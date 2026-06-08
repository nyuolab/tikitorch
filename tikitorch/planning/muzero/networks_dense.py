"""tikitorch.planning.muzero.networks_dense — a MATCHED DENSE MuZero control.

A standard (multiply-based) MuZero network with the SAME shape, latent size, head
interface, and inference API as :class:`TropicalMuZeroNet`, but with dense
``nn.Linear + ReLU`` cores instead of tropical max-plus ones. It plugs into the
identical self-play / replay / MCTS / training loop, so the ONLY difference from
the tropical agent is dense-vs-tropical cores (+ the backup operator). This is the
scientific CONTROL that attributes any gap to *tropical*, not engineering --
complementing the external literature references (LightZero / muzero-general).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch.planning.muzero.env import Inference
from tikitorch.planning.muzero.networks import _minmax_unit, _make_value_head

__all__ = ["DenseMuZeroNet"]


def _mlp(d_in, d_out, hidden=None):
    hidden = hidden or d_out
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


class _DenseRepresentation(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_mult=1):
        super().__init__()
        self.net = _mlp(obs_dim, latent_dim, hidden=latent_dim * hidden_mult)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, obs):
        return _minmax_unit(self.norm(self.net(obs)))


class _DenseDynamics(nn.Module):
    def __init__(self, latent_dim, n_actions, depth=2, reward_support=None, hidden_mult=1):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, latent_dim)
        self.blocks = nn.ModuleList(
            [_mlp(latent_dim, latent_dim, hidden=latent_dim * hidden_mult) for _ in range(depth)])
        self.norm = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(depth)])
        self.reward_head = _make_value_head(latent_dim, reward_support)

    def forward(self, h, action):
        if not torch.is_tensor(action):
            action = torch.tensor([action], device=h.device)
        z = h + self.action_embed(action)
        for blk, norm in zip(self.blocks, self.norm):
            z = norm(z + blk(z))                     # dense residual block
        h2 = _minmax_unit(z)
        return h2, self.reward_head(h2)


class _DensePrediction(nn.Module):
    def __init__(self, latent_dim, n_actions, value_support=None, hidden_mult=1):
        super().__init__()
        self.mix = _mlp(latent_dim, latent_dim, hidden=latent_dim * hidden_mult)
        self.policy_head = nn.Linear(latent_dim, n_actions)
        self.value_head = _make_value_head(latent_dim, value_support)

    def forward(self, h):
        z = self.mix(h)
        return self.policy_head(z), self.value_head(z)


class DenseMuZeroNet(nn.Module):
    """Dense MuZero control with the TropicalMuZeroNet interface (drop-in)."""

    def __init__(self, obs_dim, n_actions, latent_dim=64, tau=1.0,
                 value_support=None, reward_support=None, hidden_mult=3):
        super().__init__()
        self.n_actions = n_actions
        # hidden_mult widens the dense MLP cores to PARAM-MATCH the tropical net's
        # compositional projection (else a tropical win is confounded by capacity).
        self.repr = _DenseRepresentation(obs_dim, latent_dim, hidden_mult)
        self.dyn = _DenseDynamics(latent_dim, n_actions, reward_support=reward_support,
                                  hidden_mult=hidden_mult)
        self.pred = _DensePrediction(latent_dim, n_actions, value_support=value_support,
                                     hidden_mult=hidden_mult)

    def set_tau(self, tau):
        pass                                          # no-op (no tropical softness)

    def value_scalar(self, raw):
        return self.pred.value_head.scalar(raw)

    def reward_scalar(self, raw):
        return self.dyn.reward_head.scalar(raw)

    def initial(self, obs):
        h = self.repr(obs)
        p, v = self.pred(h)
        return h, p, v

    def recurrent(self, h, action):
        h2, r = self.dyn(h, action)
        p, v = self.pred(h2)
        return h2, r, p, v

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
