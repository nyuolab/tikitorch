"""tikitorch.planning.muzero.networks — the three Tropical MuZero networks.

MuZero learns a model in latent space from three nets:
  * representation  h0 = f_repr(obs)
  * dynamics       (h', r) = g_dyn(h, a)
  * prediction     (p, v)  = f_pred(h)

Here the cores are tropical (max-plus `TropicalProjection` + `TropicalLayerNorm`),
the action is a closed gather (`nn.Embedding`), and the scalar/policy heads are
declared `nn.Linear` boundaries (the established "tropical core + boundary head"
pattern; full closure is a later refinement). The bundle `TropicalMuZeroNet`
exposes the MuZero inference interface (`initial_inference` / `recurrent_inference`)
so it drops straight into :class:`tikitorch.planning.mcts.TropicalMCTS` in place of
the Stage-3 `KnownModel`.

Backups are chosen at the MCTS level (see the verified backup study,
results/backup_sweep/FINDINGS.md): make backup a swept hyperparameter; `mean` is
the safe default, `topk` (small k) the tropical-leaning candidate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import tikitorch.nn as ttnn
from tikitorch.planning.muzero.env import Inference

__all__ = [
    "TropicalRepresentation",
    "TropicalDynamics",
    "TropicalPrediction",
    "TropicalMuZeroNet",
]


def _minmax_unit(h, eps=1e-5):
    """MuZero's latent rescale to [0,1] per sample (min/compare + one divide)."""
    lo = h.min(dim=-1, keepdim=True).values
    hi = h.max(dim=-1, keepdim=True).values
    return (h - lo) / (hi - lo + eps)


def _set_tau(module, tau):
    for m in module.modules():
        if m is module:
            continue
        if hasattr(m, "set_tau"):
            m.set_tau(tau)
        elif hasattr(m, "tau"):
            m.tau = tau


# ---------------------------------------------------------------------------
# Value / reward heads (unified interface: forward->raw, scalar(raw), loss(raw,target))
# ---------------------------------------------------------------------------


class ScalarHead(nn.Module):
    """Plain scalar regression head (raw == scalar; MSE loss)."""

    def __init__(self, in_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, 1)

    def forward(self, h):
        return self.head(h).squeeze(-1)

    def scalar(self, raw):
        return raw

    def loss(self, raw, target):
        return F.mse_loss(raw, target)


class CategoricalHead(nn.Module):
    """MuZero categorical head over an integer support [lo, hi] (spacing 1).

    ``raw`` are logits; the scalar value is ``E[support]`` under softmax, and the
    loss is cross-entropy to the two-hot encoding of the scalar target. Categorical
    value/reward are MuZero's standard stability lever vs. scalar MSE regression.
    """

    def __init__(self, in_dim, lo, hi):
        super().__init__()
        self.register_buffer("support", torch.arange(lo, hi + 1).float())
        self.head = nn.Linear(in_dim, self.support.numel())

    def forward(self, h):
        return self.head(h)                       # logits (..., N)

    def scalar(self, raw):
        return (raw.softmax(-1) * self.support).sum(-1)

    def _two_hot(self, x):
        N = self.support.numel()
        x = x.clamp(self.support[0], self.support[-1])
        lo_idx = torch.clamp((x - self.support[0]).floor().long(), 0, N - 2)
        lower = self.support[0] + lo_idx.float()
        frac = (x - lower).clamp(0.0, 1.0)
        oh = torch.zeros(*x.shape, N, device=x.device)
        oh.scatter_(-1, lo_idx.unsqueeze(-1), (1 - frac).unsqueeze(-1))
        oh.scatter_(-1, (lo_idx + 1).unsqueeze(-1), frac.unsqueeze(-1))
        return oh

    def loss(self, raw, target):
        return -(self._two_hot(target) * raw.log_softmax(-1)).sum(-1).mean()


def _make_value_head(in_dim, support):
    return CategoricalHead(in_dim, *support) if support is not None else ScalarHead(in_dim)


class TropicalRepresentation(nn.Module):
    """obs (flat) -> latent h0, tropically encoded and rescaled to [0,1]."""

    def __init__(self, obs_dim, latent_dim, tau=1.0):
        super().__init__()
        self.enc = ttnn.TropicalProjection(obs_dim, latent_dim, tau=tau)
        self.norm = ttnn.TropicalLayerNorm(latent_dim)

    def set_tau(self, tau):
        _set_tau(self, tau)

    def forward(self, obs):
        return _minmax_unit(self.norm(self.enc(obs)))


class TropicalDynamics(nn.Module):
    """(latent h, action a) -> (next latent h', reward r). Action = closed gather."""

    def __init__(self, latent_dim, n_actions, tau=1.0, depth=2, reward_support=None):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, latent_dim)   # closed gather
        self.core = nn.ModuleList(
            [ttnn.TropicalProjection(latent_dim, latent_dim, tau=tau) for _ in range(depth)])
        self.norm = nn.ModuleList(
            [ttnn.TropicalLayerNorm(latent_dim) for _ in range(depth)])
        self.reward_head = _make_value_head(latent_dim, reward_support)   # scalar or categorical

    def set_tau(self, tau):
        _set_tau(self, tau)

    def forward(self, h, action):
        # action: long tensor (B,) or int
        if not torch.is_tensor(action):
            action = torch.tensor([action], device=h.device)
        z = h + self.action_embed(action)                        # (B, D)
        for core, norm in zip(self.core, self.norm):
            z = norm(z + core(z))                                # tropical residual block
        h2 = _minmax_unit(z)                                     # rescale latent to [0,1]
        return h2, self.reward_head(h2)                          # raw reward output


class TropicalPrediction(nn.Module):
    """latent h -> (policy logits over actions, scalar value)."""

    def __init__(self, latent_dim, n_actions, tau=1.0, value_support=None):
        super().__init__()
        self.mix = ttnn.TropicalProjection(latent_dim, latent_dim, tau=tau)
        self.policy_head = nn.Linear(latent_dim, n_actions)      # declared boundary
        self.value_head = _make_value_head(latent_dim, value_support)   # scalar or categorical

    def set_tau(self, tau):
        _set_tau(self, tau)

    def forward(self, h):
        z = self.mix(h)
        return self.policy_head(z), self.value_head(z)          # (policy_logits, raw value)


class TropicalMuZeroNet(nn.Module):
    """Bundle exposing the MuZero inference interface for TropicalMCTS.

    ``initial_inference(obs)``      -> Inference(state=latent, reward=0, policy, value)
    ``recurrent_inference(h, a)``   -> (Inference(state=latent, reward, policy, value), done)

    ``obs`` is a flat tensor of shape ``(obs_dim,)`` or ``(B, obs_dim)``; ``state``
    carried through the tree is the latent tensor. ``done`` is always False (MuZero
    plans in latent space to a fixed depth; no learned terminal head here).
    """

    def __init__(self, obs_dim, n_actions, latent_dim=64, tau=1.0,
                 value_support=None, reward_support=None):
        super().__init__()
        self.n_actions = n_actions
        self.repr = TropicalRepresentation(obs_dim, latent_dim, tau=tau)
        self.dyn = TropicalDynamics(latent_dim, n_actions, tau=tau, reward_support=reward_support)
        self.pred = TropicalPrediction(latent_dim, n_actions, tau=tau, value_support=value_support)

    # scalar conversions (identity for scalar heads; expectation for categorical)
    def value_scalar(self, raw):
        return self.pred.value_head.scalar(raw)

    def reward_scalar(self, raw):
        return self.dyn.reward_head.scalar(raw)

    def set_tau(self, tau):
        self.repr.set_tau(tau)
        self.dyn.set_tau(tau)
        self.pred.set_tau(tau)

    # -------- batched forward used by training (K-step unroll) --------
    def initial(self, obs):
        h = self.repr(obs)
        p, v = self.pred(h)
        return h, p, v

    def recurrent(self, h, action):
        h2, r = self.dyn(h, action)
        p, v = self.pred(h2)
        return h2, r, p, v

    # -------- single-state inference for MCTS (Inference dataclass) --------
    @torch.no_grad()
    def initial_inference(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h, p, vraw = self.initial(obs)
        return Inference(state=h, reward=0.0, policy=F.softmax(p[0], dim=-1),
                         value=float(self.value_scalar(vraw)[0]))

    @torch.no_grad()
    def recurrent_inference(self, h, action):
        h2, rraw, p, vraw = self.recurrent(h, action)
        return Inference(state=h2, reward=float(self.reward_scalar(rraw)[0]),
                         policy=F.softmax(p[0], dim=-1),
                         value=float(self.value_scalar(vraw)[0])), False
