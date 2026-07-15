"""Cascade PC trains fully tropical sequence models (branch pc-morphssm).

Track 1 (primary): predictive-coding training of the REAL ``tt.models.MorphSSM``
(bakeoff config) via a single-sweep cascade over frozen tropical routing — the
unrolled-graph corollary of cascade≡settled-PC (results/pc_convrnn/
THEORY_UNROLLED_CASCADE.md). At tau=0 the entire backward is gather/scatter/add:
weights enter tropical ops ADDITIVELY, so every parameter gradient is a routed
sum of errors — zero multiplies of any kind (the soft-tau phase has the
sanctioned eps·p products, same closure rung as the BP arms' soft forward).

Track 2 (extension): ConvRNNLM = MorphSSM layers + a per-channel tropical
leaky-max recurrence h_t = max(u_t, h_{t-1} - lam), lam = lam_min + |raw| > 0
(bounded by construction; the known min-plus RNN drift is the lam=0 corner).
The scan is HARD at all tau by design; its temporal adjoint is a unit-gain
gated reverse scan.

Arms (--arms arch:credit):
    morphssm:bp / morphssm:cascade / morphssm:relax_async
    convrnn:bp  / convrnn:cascade  / convrnn-norecur:bp   (paired ablation)

Protocol: exact bakeoff (nanogpt_shakespeare.py) — 1500 steps, batch 24, seq 96,
char-65, fixed val windows, tau-anneal 0.5 -> 0 by 90%, soft+hard eval logged.

Usage::

    python scripts/pc_convrnn_charlm.py --gradcheck            # G0 quick gate
    python scripts/pc_convrnn_charlm.py --arms morphssm:bp morphssm:cascade
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import tikitorch as tt                                    # noqa: E402
from tikitorch.nn import (                                # noqa: E402
    DepthwiseMorphConv1d,
    MorphSSMLayer,
    TropicalDilationModule,
    TropicalDropout,
    TropicalLayerNorm,
    TropicalTopKModule,
)
from nanogpt_shakespeare import BPEData, CharData, evaluate, sample, tau_at  # noqa: E402


# ---------------------------------------------------------------------------
# Track 2 primitive: per-channel tropical leaky-max recurrence (HARD at all tau)
# ---------------------------------------------------------------------------

class TropicalLeakyState(nn.Module):
    """h_t[c] = max(u_t[c], h_{t-1}[c] - lam[c]),  lam[c] = lam_min + |raw[c]|.

    Closed form h_t = max_k(u_{t-k} - k*lam): a content-gated running max with
    per-channel decay — bounded u_t <= h_t <= running-max(u) since lam > 0 (the
    tropical spectral radius of the only cycle, the self-loop, is -lam < 0).
    Add/compare only. HARD at all tau (a soft scan biases +tau*ln2 PER STEP and
    compounds through the recurrence — the drift the cell exists to prevent).
    lam init log-spaced (multi-timescale horizons ~ range/lam).
    """

    def __init__(self, channels: int, lam_min: float = 2 ** -6,
                 lam_init=(2 ** -6, 0.5)):
        super().__init__()
        self.channels = channels
        self.lam_min = lam_min
        lo, hi = lam_init
        lam0 = torch.logspace(math.log10(max(lo, lam_min)), math.log10(hi), channels)
        self.raw = nn.Parameter(lam0 - lam_min)          # lam = lam_min + |raw|

    def lam(self) -> torch.Tensor:
        return self.lam_min + self.raw.abs()

    def set_tau(self, tau: float) -> None:               # hard by design
        pass

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C, T); autograd path for the bp arms."""
        lam = self.lam().unsqueeze(0)                     # (1, C)
        h = u[:, :, 0]
        outs = [h]
        for t in range(1, u.shape[-1]):
            h = torch.maximum(h - lam, u[:, :, t])
            outs.append(h)
        return torch.stack(outs, dim=-1)

    @torch.no_grad()
    def forward_with_gates(self, u: torch.Tensor):
        """Returns (h, gates); gates[b,c,t] = 1 iff the carry branch won at t.

        Frozen routing for the PC cascade. gates[..., 0] = False (h_0 = u_0).
        Matches ``forward`` bit-exactly (same maximum on the same operands).
        """
        lam = self.lam().unsqueeze(0)
        B, C, T = u.shape
        gates = torch.zeros(B, C, T, dtype=torch.bool, device=u.device)
        h = u[:, :, 0]
        outs = [h]
        for t in range(1, T):
            carry = h - lam
            g = carry >= u[:, :, t]
            gates[:, :, t] = g
            h = torch.maximum(carry, u[:, :, t])
            outs.append(h)
        return torch.stack(outs, dim=-1), gates

    @torch.no_grad()
    def step(self, u_t: torch.Tensor, h_prev):
        """Streaming: (B, C) drive + (B, C) state -> new state (= output)."""
        if h_prev is None:
            return u_t
        return torch.maximum(h_prev - self.lam().unsqueeze(0), u_t)


@torch.no_grad()
def leaky_scan_adjoint(e_h: torch.Tensor, gates: torch.Tensor):
    """Frozen-gate reverse scan: the unit-gain cascade-through-time.

    e_h (B,C,T): error arriving at h_t from above. Returns (e_u, lam_grad):
        tot_t = e_h_t + carry;  e_u_t = (1-g_t)*tot_t;  carry <- g_t*tot_t;
        lam_grad = -sum_t g_t*tot_t  (the -lam edge has d/dlam = -1: a count).
    Gather/gate/add only — zero multiplies.
    """
    B, C, T = e_h.shape
    e_u = torch.empty_like(e_h)
    carry = torch.zeros(B, C, device=e_h.device, dtype=e_h.dtype)
    lam_grad = torch.zeros(C, device=e_h.device, dtype=e_h.dtype)
    zero = torch.zeros((), device=e_h.device, dtype=e_h.dtype)
    for t in range(T - 1, -1, -1):
        tot = e_h[:, :, t] + carry
        g = gates[:, :, t]
        e_u[:, :, t] = torch.where(g, zero, tot)
        carry = torch.where(g, tot, zero)
        lam_grad -= torch.where(g, tot, zero).sum(0)
    return e_u, lam_grad


# ---------------------------------------------------------------------------
# Track 2 model: MorphSSM + leaky-max recurrence (norecur == MorphSSM layout)
# ---------------------------------------------------------------------------

class ConvRNNLayer(MorphSSMLayer):
    """MorphSSMLayer with the leaky-max scan between the morph conv and dropout."""

    def __init__(self, d_model: int, *, recurrent: bool = True,
                 lam_min: float = 2 ** -6, **kw):
        super().__init__(d_model, **kw)
        assert self.out_proj_type == "tropical" and self.residual_scale == 1.0
        self.scan = TropicalLeakyState(d_model, lam_min=lam_min) if recurrent else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x).transpose(1, 2)
        h = self.morph_conv(h)
        if self.scan is not None:
            h = self.scan(h)
        h = h.transpose(1, 2)
        h = self.dropout(h)
        B, L, D = h.shape
        h = self.out_proj(h.reshape(B * L, D)).reshape(B, L, D)
        return self.post_norm(residual + h)


class ConvRNNLM(nn.Module):
    """MorphSSM LM (bakeoff layout: embed -> dropout -> layers -> norm -> topk
    head) with an optional per-layer tropical leaky-max recurrence. With
    ``recurrent=False`` this is module-for-module the MorphSSM bakeoff config.

    ``hybrid_attn=True`` inserts ONE l1-score + argmax-GATHER attention
    sublayer (pre-norm residual) after the middle layer — the RECALL ORGAN of
    the strategy note (STRATEGY_TRANSFORMER_COMPETITION.md): the mixer is
    routing-discrete (gather adjoint = a scatter; deploy-pure at tau=0, the
    measured lossless-induction recipe), projections nn.Linear for the M2
    capability gate (dyadic later for closure). BP-trained for M2; the cascade
    adjoint for this layer is gated on M2 passing."""

    task_type = "lm"

    def __init__(self, vocab: int, d_model: int = 256, n_layers: int = 7,
                 kernel_size: int = 7, dropout: float = 0.1, tau: float = 0.5,
                 recurrent: bool = True, max_dilation: int = 2048,
                 lam_min: float = 2 ** -6, hybrid_attn: bool = False):
        super().__init__()
        from tikitorch.nn import TropicalAttention
        self.embed = nn.Embedding(vocab, d_model)
        self.dropout = TropicalDropout(dropout) if dropout > 0 else nn.Identity()
        self.layers = nn.ModuleList([
            ConvRNNLayer(
                d_model, kernel_size=kernel_size,
                dilation=min(2 ** i, max_dilation),
                mode="dilation" if i % 2 == 0 else "erosion",
                dropout=dropout, use_soft=True, tau=max(tau, 1e-6),
                norm_type="tropical", recurrent=recurrent, lam_min=lam_min)
            for i in range(n_layers)])
        self.attn_at = n_layers // 2 if hybrid_attn else None
        if hybrid_attn:
            self.attn_norm = TropicalLayerNorm(d_model)
            self.attn = TropicalAttention(
                d_model, n_heads=max(1, d_model // 32), tau=max(tau, 1e-6),
                use_standard_proj=True, score_variant="l1",
                agg_variant="gather")
        self.ln_final = TropicalLayerNorm(d_model)
        self.output_head = TropicalTopKModule(d_model, vocab, k=min(8, d_model))
        for layer in self.layers:                         # MorphSSM GPT-2-style init
            nn.init.normal_(layer.out_proj.weight, std=0.02 / math.sqrt(2 * n_layers))

    def set_tau(self, tau: float) -> None:                # MorphSSM.set_tau walk
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "set_tau") and callable(m.set_tau):
                m.set_tau(tau)
            elif hasattr(m, "tau"):
                m.tau = tau
            if hasattr(m, "use_soft"):
                m.use_soft = tau > 0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.embed(ids))
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.attn_at is not None and i == self.attn_at:
                h = h + self.attn(self.attn_norm(h), causal_mask=True)
        return self.output_head(self.ln_final(h))

    def compute_loss(self, x, y):
        logits = self(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


# ---------------------------------------------------------------------------
# The PC engine: cascade + error relaxation over frozen tropical routing
# ---------------------------------------------------------------------------

def _pow2_factor(c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """The stop-grad power-of-two scale of TropicalLayerNorm (nn._pow2_normalize)."""
    scale = c.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    shift = torch.round(torch.log2(scale)).clamp_(-8.0, 8.0)
    return torch.exp2(-shift)


class MorphPCEngine:
    """Cascade / relaxation PC over a MorphSSM-shaped LM (MorphSSM or ConvRNNLM).

    Mirrors DyadicPCNet's interface (cascade_grads / error_relax_grads /
    train_step_*), predictive_coding.py:628/686. The model's forward is never
    used for training under PC — the engine runs its own forward, caching the
    FROZEN routing (conv argmax slot or softmax weights, out_proj winners,
    top-k head selection, dropout gates, recurrence carry gates, pow-2 norm
    factors), then sweeps the error once in reverse-topological order. At tau=0
    every weight gradient is a routed SUM of errors (weights enter tropical ops
    additively) — gather/scatter/add, zero multiplies. Bit-parity of the cached
    forward with ``model(x)`` is gate G0a.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        for layer in model.layers:
            assert isinstance(layer.morph_conv, DepthwiseMorphConv1d) \
                and layer.morph_conv.group_size == 1, "v1: depthwise conv only"
            assert isinstance(layer.out_proj, TropicalDilationModule)
            assert isinstance(layer.norm, TropicalLayerNorm)
            assert isinstance(layer.post_norm, TropicalLayerNorm)
        assert isinstance(model.output_head, TropicalTopKModule)

    # -- forward with frozen routing ---------------------------------------
    @torch.no_grad()
    def _norm_fwd(self, norm: TropicalLayerNorm, x: torch.Tensor):
        c = x - x.mean(dim=-1, keepdim=True)
        f = _pow2_factor(c, norm.eps)
        y = c * f
        if norm.beta is not None:
            y = y + norm.beta
        return y, f

    @torch.no_grad()
    def _drop_mask(self, mod, ref: torch.Tensor):
        """{0,1} dropout gate (TropicalDropout semantics: no survivor rescale)."""
        if not isinstance(mod, TropicalDropout) or not self.model.training \
                or mod.p == 0.0:
            return None
        return (torch.rand_like(ref) >= mod.p).to(ref.dtype)

    @torch.no_grad()
    def _forward_cache(self, ids: torch.Tensor):
        m = self.model
        h = m.embed(ids)                                   # (B, T, D)
        mask0 = self._drop_mask(m.dropout, h)
        if mask0 is not None:
            h = h * mask0
        caches = []
        for layer in m.layers:
            c = {"x_in": h}
            xn, c["f1"] = self._norm_fwd(layer.norm, h)
            xt = xn.transpose(1, 2)                        # (B, C, T)
            u, c["conv_route"] = layer.morph_conv.forward_with_routing(xt)
            if getattr(layer, "scan", None) is not None:
                u, c["gates"] = layer.scan.forward_with_gates(u)
            else:
                c["gates"] = None
            v = u.transpose(1, 2)                          # (B, T, D)
            c["drop_mask"] = self._drop_mask(layer.dropout, v)
            if c["drop_mask"] is not None:
                v = v * c["drop_mask"]
            B, T, D = v.shape
            rows = v.reshape(B * T, D)
            proj_soft = layer.out_proj.use_softmax
            c["proj_in"] = rows if proj_soft else None     # soft: recompute p later
            y, c["proj_route"] = layer.out_proj.forward_with_routing(rows)
            r = h + y.reshape(B, T, D)
            h, c["f2"] = self._norm_fwd(layer.post_norm, r)
            caches.append(c)
        hf, f_final = self._norm_fwd(m.ln_final, h)
        B, T, D = hf.shape
        rows = hf.reshape(B * T, D)
        V, k = m.output_head.out_features, m.output_head.k
        idxs, outs = [], []
        for s in range(0, rows.shape[0], 512):             # bound the (rows,V,D) transient
            expanded = m.output_head.weight + rows[s:s + 512].unsqueeze(-2)
            vals, idx = expanded.topk(k, dim=-1)
            outs.append(vals.sum(dim=-1) + m.output_head.bias)
            idxs.append(idx)
        logits = torch.cat(outs, 0).reshape(B, T, V)
        top = {"mask0": mask0, "ids": ids, "f_final": f_final,
               "head_idx": torch.cat(idxs, 0)}             # (B*T, V, k)
        return logits, caches, top

    # -- adjoint pieces ------------------------------------------------------
    @staticmethod
    def _norm_adj(e: torch.Tensor, f: torch.Tensor, norm: TropicalLayerNorm,
                  grads: bool):
        if grads and norm.beta is not None:
            g = e.reshape(-1, e.shape[-1]).sum(0)
            norm.beta.grad = g if norm.beta.grad is None else norm.beta.grad + g
        ec = e * f                                          # stop-grad pow-2 scale
        return ec - ec.mean(dim=-1, keepdim=True)           # centering adjoint

    def _conv_adj(self, layer, e_u: torch.Tensor, route, grads: bool):
        """Error at conv output (B,C,T) -> error at conv input; optional w/b grads."""
        conv = layer.morph_conv
        K, dil = conv.kernel_size, conv.dilation
        B, C, T = e_u.shape
        e_x = torch.zeros_like(e_u)
        soft = route.dtype.is_floating_point                # p (B,C,T,K) vs idx
        if soft:
            e_slots = e_u.unsqueeze(-1) * route             # sanctioned soft-phase product
            wg = e_slots.sum(dim=(0, 2))
            for j in range(K):
                if j * dil < T:
                    e_x[:, :, :T - j * dil] += e_slots[:, :, j * dil:, j]
        else:
            wg = torch.zeros_like(conv.weight)
            zero = torch.zeros((), device=e_u.device, dtype=e_u.dtype)
            for j in range(K):
                # gate as SELECTION (mux), not arithmetic — keeps the tau=0
                # sweep literally multiply-free under the dispatch audit
                contrib = torch.where(route == j, e_u, zero)
                wg[:, j] = contrib.sum(dim=(0, 2))
                if j * dil < T:
                    e_x[:, :, :T - j * dil] += contrib[:, :, j * dil:]
        if grads:
            bg = e_u.sum(dim=(0, 2))
            conv.weight.grad = wg if conv.weight.grad is None else conv.weight.grad + wg
            conv.bias.grad = bg if conv.bias.grad is None else conv.bias.grad + bg
        return e_x

    def _proj_adj(self, layer, e: torch.Tensor, cache, grads: bool):
        """Error at out_proj output (R,O) -> error at its input (R,I)."""
        proj = layer.out_proj
        O, I = proj.weight.shape
        R = e.shape[0]
        if cache["proj_route"] is not None:                 # hard: winner scatter
            idx = cache["proj_route"]
            e_x = torch.zeros(R, I, device=e.device, dtype=e.dtype)
            e_x.scatter_add_(1, idx, e)
            if grads:
                flat = (torch.arange(O, device=e.device).unsqueeze(0) * I + idx)
                wg = torch.zeros(O * I, device=e.device, dtype=e.dtype)
                wg.index_add_(0, flat.reshape(-1), e.reshape(-1))
                wg = wg.view(O, I)
        else:                                               # soft: chunked p recompute
            x = cache["proj_in"]
            e_x = torch.zeros(R, I, device=e.device, dtype=e.dtype)
            wg = torch.zeros(O, I, device=e.device, dtype=e.dtype) if grads else None
            chunk = max(1, 32_000_000 // max(1, R * I))
            for s in range(0, O, chunk):
                z = (proj.weight[s:s + chunk].unsqueeze(0) + x.unsqueeze(1)) / proj.tau
                p = torch.softmax(z, dim=-1)                # (R, chunk, I) exact Jacobian
                e_x += torch.einsum("ro,roi->ri", e[:, s:s + chunk], p)
                if grads:
                    wg[s:s + chunk] = torch.einsum("ro,roi->oi", e[:, s:s + chunk], p)
        if grads:
            bg = e.sum(0)
            proj.weight.grad = wg if proj.weight.grad is None else proj.weight.grad + wg
            proj.bias.grad = bg if proj.bias.grad is None else proj.bias.grad + bg
        return e_x

    def _layer_adjoint(self, layer, e_out: torch.Tensor, cache, grads: bool):
        """Full adjoint of one layer: error at output -> error at input."""
        B, T, D = e_out.shape
        e = self._norm_adj(e_out, cache["f2"], layer.post_norm, grads)
        e_res = e                                           # residual: unit-gain copy
        e_proj = self._proj_adj(layer, e.reshape(B * T, D), cache, grads)
        e_v = e_proj.reshape(B, T, D)
        if cache["drop_mask"] is not None:
            e_v = e_v * cache["drop_mask"]                  # {0,1} gate
        e_u = e_v.transpose(1, 2)                           # (B, C, T)
        if cache["gates"] is not None:
            e_u, lam_grad = leaky_scan_adjoint(e_u, cache["gates"])
            if grads:
                raw = layer.scan.raw
                rg = lam_grad * torch.sign(raw.detach())    # lam = lam_min + |raw|
                raw.grad = rg if raw.grad is None else raw.grad + rg
        e_xn = self._conv_adj(layer, e_u, cache["conv_route"], grads).transpose(1, 2)
        e_in = self._norm_adj(e_xn, cache["f1"], layer.norm, grads)
        return e_res + e_in

    loss_mode = "ce"      # "ce" = exact softmax (host boundary) | "lut" =
                          # shift_softmax LUT probabilities (G5 strict rung)
    loss_mask = None      # optional (B,T) bool: learn ONLY where True. Used by
                          # the embodied agent (E1) to exclude policy/separator
                          # token positions from the world-model kick per
                          # E1_SPEC ("policy tokens carry no loss"). Applied as
                          # a mux (torch.where) on the top error -> audit-clean.
    gate_mask = None      # optional (B,T) bool: the serve CE handed to the
                          # surprise gate (and returned) is computed over THESE
                          # positions, decoupled from loss_mask. E2 gates on
                          # obs-only CE while TRAINING over obs+reward, so the
                          # gate axis stays orthogonal to the objective axis.

    @torch.no_grad()
    def _top_error(self, logits, targets, caches_top, grads: bool):
        """CE top error at every position -> error at the stack output."""
        m = self.model
        B, T, V = logits.shape
        if self.loss_mode == "lut":
            from tikitorch.functional import shift_softmax
            eps = shift_softmax(logits.reshape(-1, V).float(),
                                normalize="exact").clone()
        else:
            eps = torch.softmax(logits.reshape(-1, V).float(), dim=-1)
        eps.scatter_add_(1, targets.reshape(-1, 1),
                         -torch.ones(B * T, 1, device=eps.device))
        if self.loss_mask is not None:
            # mux the error to zero at excluded positions (mult-free), and take
            # the mean over KEPT positions only
            keep = self.loss_mask.reshape(-1, 1)
            eps = torch.where(keep, eps, torch.zeros((), dtype=eps.dtype))
            denom = int(self.loss_mask.sum().clamp_min(1))
        else:
            denom = B * T
        eps = (eps / denom).to(logits.dtype)                # d(mean-CE)/d(logits)
        head, idx = m.output_head, caches_top["head_idx"]   # (R, V, k)
        R, D = B * T, head.in_features
        k = head.k
        e_exp = eps.unsqueeze(-1).expand(R, V, k)
        e_x = torch.zeros(R, D, device=eps.device, dtype=eps.dtype)
        e_x.scatter_add_(1, idx.reshape(R, V * k), e_exp.reshape(R, V * k))
        if grads:
            flat = (torch.arange(V, device=eps.device).view(1, V, 1) * D + idx)
            wg = torch.zeros(V * D, device=eps.device, dtype=eps.dtype)
            wg.index_add_(0, flat.reshape(-1), e_exp.reshape(-1))
            head.weight.grad = wg.view(V, D)
            head.bias.grad = eps.sum(0)
        e = e_x.reshape(B, T, D)
        return self._norm_adj(e, caches_top["f_final"], m.ln_final, grads)

    @torch.no_grad()
    def _embed_grads(self, e0: torch.Tensor, top):
        m = self.model
        if top["mask0"] is not None:
            e0 = e0 * top["mask0"]
        D = e0.shape[-1]
        g = torch.zeros_like(m.embed.weight)
        g.index_add_(0, top["ids"].reshape(-1), e0.reshape(-1, D))
        m.embed.weight.grad = g

    # -- entry points --------------------------------------------------------
    @torch.no_grad()
    def cascade_grads(self, ids: torch.Tensor, targets: torch.Tensor):
        """One forward (cached routing) + ONE reverse-topological sweep."""
        logits, caches, top = self._forward_cache(ids)
        e = self._top_error(logits, targets, top, grads=True)
        for layer, cache in zip(reversed(self.model.layers), reversed(caches)):
            e = self._layer_adjoint(layer, e, cache, grads=True)
        self._embed_grads(e, top)
        return logits

    @torch.no_grad()
    def error_relax_grads(self, ids, targets, n_iter=None, participation=1.0,
                          stale=False, rand_order=False, gen=None):
        """Iterative form on the frozen routing: E[l] = A_l^T E[l+1], E[L] clamped.

        Sync top-down with n_iter=1 equals :meth:`cascade_grads` bit-identically
        (triangular system). participation/stale/rand_order = the async fabric
        probe (DyadicPCNet.error_relax_grads conventions; ``gen`` is a CPU
        torch.Generator for the control-flow draws)."""
        m = self.model
        logits, caches, top = self._forward_cache(ids)
        L = len(m.layers)
        if n_iter is None:
            n_iter = L + 2
        eL = self._top_error(logits, targets, top, grads=False)  # clamped
        E = [torch.zeros_like(eL) for _ in range(L + 1)]
        E[L] = eL
        free = list(reversed(range(L)))                     # E[L-1] .. E[0]
        for _ in range(n_iter):
            snap = [t.clone() for t in E] if stale else E
            order = free
            if rand_order:
                order = [free[i] for i in
                         torch.randperm(len(free), generator=gen).tolist()]
            for l in order:
                if participation < 1.0 and \
                        torch.rand(1, generator=gen).item() >= participation:
                    continue
                E[l] = self._layer_adjoint(m.layers[l], snap[l + 1],
                                           caches[l], grads=False)
            E[L] = eL
        self._top_error(logits, targets, top, grads=True)   # head/ln_final grads
        for l in range(L):                                  # layer grads from settled E
            self._layer_adjoint(m.layers[l], E[l + 1], caches[l], grads=True)
        self._embed_grads(E[0], top)
        return logits

    @torch.no_grad()
    def serve_and_learn(self, ids, targets, opt=None, fire=None):
        """FUSED serve+learn (north-star V0): ONE forward serves the logits and
        caches the frozen routing; if ``fire`` (bool, or callable(ce)->bool for
        surprise gating) the reverse sweep runs FROM THOSE CACHES + opt.step().
        Cost: 1 forward (serve) + [1 reverse sweep + opt] iff fired — vs the
        unfused harness path (serve forward + cascade_grads re-forward + sweep
        = ~3x). Grads are bitwise identical to :meth:`cascade_grads` (same
        code path on the same caches). Returns (logits, serve_ce, fired)."""
        logits, caches, top = self._forward_cache(ids)
        V = logits.shape[-1]
        ce_mask = self.gate_mask if self.gate_mask is not None else self.loss_mask
        if ce_mask is not None:
            keep = ce_mask.reshape(-1)
            lf = logits.reshape(-1, V).float()[keep]
            ce = F.cross_entropy(lf, targets.reshape(-1)[keep]).item()
        else:
            ce = F.cross_entropy(logits.reshape(-1, V).float(),
                                 targets.reshape(-1)).item()
        fired = False
        if opt is not None:
            do = fire(ce) if callable(fire) else bool(fire)
            if do:
                opt.zero_grad(set_to_none=True)
                e = self._top_error(logits, targets, top, grads=True)
                for layer, cache in zip(reversed(self.model.layers),
                                        reversed(caches)):
                    e = self._layer_adjoint(layer, e, cache, grads=True)
                self._embed_grads(e, top)
                opt.step()
                fired = True
        return logits, ce, fired

    def train_step(self, x, y, opt, credit="cascade", **kw):
        opt.zero_grad(set_to_none=True)
        if credit == "cascade":
            logits = self.cascade_grads(x, y)
        else:
            logits = self.error_relax_grads(x, y, **kw)
        opt.step()
        with torch.no_grad():
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   y.reshape(-1))
        return loss


# ---------------------------------------------------------------------------
# G0 gradcheck: engine forward parity + cascade vs autograd, both taus
# ---------------------------------------------------------------------------

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return float((a @ b) / denom)


def build_model(arch: str, vocab: int, args) -> nn.Module:
    if arch == "morphssm":
        return tt.models.MorphSSM(
            d_model=args.d_model, n_layers=args.n_layers, kernel_size=args.kernel,
            tropical_proj=True, dropout=args.dropout, use_soft=True,
            tau=max(args.tau, 1e-6), task_type="lm", vocab_size=vocab)
    if arch in ("convrnn", "convrnn-norecur", "convrnn-hybrid"):
        return ConvRNNLM(vocab, d_model=args.d_model, n_layers=args.n_layers,
                         kernel_size=args.kernel, dropout=args.dropout,
                         tau=args.tau, recurrent=(arch != "convrnn-norecur"),
                         lam_min=args.lam_min,
                         hybrid_attn=(arch == "convrnn-hybrid"))
    raise ValueError(arch)


def gradcheck(args, out_path=None):
    """G0: forward bit-parity + per-tensor cascade-vs-autograd cosine, tau 0.5/0."""
    results = []
    for arch in ("morphssm", "convrnn"):
        for tau in (0.5, 0.0):
            for seed in range(args.gc_seeds):
                torch.manual_seed(seed)
                gc = argparse.Namespace(**{**vars(args), "d_model": 64,
                                           "n_layers": 3, "dropout": 0.0})
                model = build_model(arch, 65, gc)
                model.eval()                                # dropout off for the gate
                model.set_tau(tau)
                gen = torch.Generator().manual_seed(100 + seed)
                T = 64 if arch == "morphssm" else 192       # carry chains > conv RF
                x = torch.randint(0, 65, (4, T), generator=gen)
                y = torch.randint(0, 65, (4, T), generator=gen)

                engine = MorphPCEngine(model)
                logits_ref = model(x)
                logits_pc, _, _ = engine._forward_cache(x)
                fwd_exact = bool(torch.equal(logits_ref, logits_pc))

                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    logits_ref.reshape(-1, 65), y.reshape(-1))
                loss.backward()
                ref = {n: p.grad.detach().clone()
                       for n, p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                engine.cascade_grads(x, y)
                worst = (None, 1.0)
                for n, p in model.named_parameters():
                    if n not in ref:
                        continue
                    g = p.grad if p.grad is not None else torch.zeros_like(p)
                    c = _cos(ref[n], g)
                    rel = float((ref[n] - g).norm() /
                                ref[n].norm().clamp_min(1e-30))
                    if c < worst[1]:
                        worst = (n, c)
                    results.append(dict(gate="G0", arch=arch, tau=tau, seed=seed,
                                        param=n, cos=round(c, 8),
                                        rel_l2=round(rel, 8),
                                        fwd_exact=fwd_exact))
                status = "PASS" if (fwd_exact and worst[1] >= 0.999) else "FAIL"
                print(f"[G0] {arch:9s} tau={tau:3.1f} seed={seed} "
                      f"fwd_exact={fwd_exact} worst={worst[0]} "
                      f"cos={worst[1]:.8f}  {status}")
    if out_path:
        with open(out_path, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    ok = all(r["cos"] >= 0.999 and r["fwd_exact"] for r in results)
    print(f"[G0] overall: {'PASS' if ok else 'FAIL'} ({len(results)} tensor checks)")
    return ok


# ---------------------------------------------------------------------------
# Training run (exact bakeoff protocol) — arms: arch:credit
# ---------------------------------------------------------------------------

def run_one(arch, credit, data, args, log_path):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(arch, data.vocab, args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    engine = MorphPCEngine(model) if credit != "bp" else None

    lr = args.adam_lr if credit == "bp" else args.pc_lr
    if args.opt == "puretropical":
        import tikitorch.optim as ttoptim
        opt = ttoptim.PureTropicalSGD(model.parameters(), lr_shift=args.lr_shift,
                                      momentum_bits=4)
    elif args.opt == "tropicallion":
        import tikitorch.optim as ttoptim
        opt = ttoptim.TropicalLion(model.parameters(), lr_shift=args.lr_shift)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if engine is not None:
        engine.loss_mode = "lut" if args.loss == "lut" else "ce"
    if not args.anneal and args.tau == 0.0:
        model.set_tau(0.0)                       # hard-from-scratch (G5 receipt)
    if args.strict_audit and engine is not None:
        from tikitorch.audit import _make_catcher
        model.set_tau(0.0)
        gen0 = torch.Generator().manual_seed(0)
        xa, ya = data.batch("train", 2, args.seq_len, gen0)
        hits = []
        with _make_catcher(hits):
            engine.cascade_grads(xa.to(device), ya.to(device))
            opt.step()
        opt.zero_grad(set_to_none=True)
        waived = {"elementwise-divide", "transcendental"}
        from collections import Counter
        cats = Counter(c for c, _ in hits)
        bad = {c: n for c, n in cats.items() if c not in waived}
        row = dict(gate="G5_audit", arch=arch, opt=args.opt, loss=args.loss,
                   tau=0.0, violations=bad,
                   waived={c: n for c, n in cats.items() if c in waived},
                   training_closed=not bad)
        with (args.out / "g5_audit.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[G5-audit] training_closed={not bad} violations={bad} "
              f"waived={dict(row['waived'])}")
        model.set_tau(max(args.tau, 0.0))
    gen = torch.Generator().manual_seed(args.seed)
    async_gen = torch.Generator().manual_seed(args.seed + 7)
    tag = f"{arch}:{credit}"
    L = args.n_layers
    t0 = time.time()
    tokens_seen = 0
    rows = []

    for step in range(args.steps):
        if args.anneal:
            model.set_tau(tau_at(step, args.steps, args.tau, True, args.tau_floor))
        x, y = data.batch("train", args.batch_size, args.seq_len, gen)
        x, y = x.to(device), y.to(device)
        if credit == "bp":
            opt.zero_grad(set_to_none=True)
            loss = model.compute_loss(x, y)
            loss.backward()
            opt.step()
        elif credit == "cascade":
            loss = engine.train_step(x, y, opt, credit="cascade")
        else:                                               # relax_async
            loss = engine.train_step(
                x, y, opt, credit="relax", n_iter=3 * L,
                participation=args.participation, stale=True, rand_order=True,
                gen=async_gen)

        tokens_seen += y.numel()
        if step % args.eval_interval == 0 or step == args.steps - 1:
            cur_tau = tau_at(step, args.steps, args.tau, args.anneal, args.tau_floor)
            ce, acc = evaluate(model, data, args.batch_size, args.seq_len,
                               args.eval_batches, args.seed + 1, device)
            model.set_tau(0.0)
            ce_h, acc_h = evaluate(model, data, args.batch_size, args.seq_len,
                                   args.eval_batches, args.seed + 1, device)
            model.set_tau(cur_tau)
            row = dict(run=tag, arch=arch, credit=credit, opt="adamw", lr=lr,
                       tokenizer="char", step=step, seed=args.seed,
                       train_loss=float(loss.item()),
                       wall_s=round(time.time() - t0, 1),
                       tok_per_s=round(tokens_seen / max(time.time() - t0, 1e-9)),
                       val_ce_soft=ce, val_acc_soft=acc, tau=cur_tau,
                       val_ce_hard=ce_h, val_acc_hard=acc_h,
                       val_bpb=ce_h * data.val_tokens_per_byte / math.log(2))
            if credit != "bp":
                row.update(pc_niter=(1 if credit == "cascade" else 3 * L),
                           participation=(1.0 if credit == "cascade"
                                          else args.participation))
            scans = [l.scan for l in model.layers if getattr(l, "scan", None)]
            if scans:
                lam = torch.cat([s.lam().detach().cpu() for s in scans])
                row["lambda_p50"] = float(lam.median())
            rows.append(row)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{tag}] step {step:5d} train {loss.item():.4f} "
                  f"val_ce {ce_h:.4f} hard / {ce:.4f} soft "
                  f"({row['tok_per_s']} tok/s)", flush=True)

    model.set_tau(0.0)
    ckpt_dir = args.out / "ckpts"                       # consumed by gate G6
    ckpt_dir.mkdir(parents=True, exist_ok=True)         # (untracked: binaries)
    ckpt_name = args.ckpt_name or f"{arch}_{credit}_s{args.seed}.pt"
    torch.save({"arch": arch, "credit": credit, "seed": args.seed,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": dict(d_model=args.d_model, n_layers=args.n_layers,
                               kernel=args.kernel, dropout=args.dropout,
                               lam_min=args.lam_min, vocab=data.vocab)},
               ckpt_dir / ckpt_name)
    text = sample(model, data, "\n", args.sample_chars, args.seq_len, device,
                  seed=args.seed)
    # streaming/length-extrapolation eval (recurrence payoff metric)
    ext = {}
    for T in args.eval_long:
        if T <= args.seq_len:
            continue
        ce_l, _ = evaluate(model, data, max(2, args.batch_size // (T // 96)),
                           T, args.eval_batches, args.seed + 1, device)
        ext[f"val_ce_hard_T{T}"] = ce_l
    final = dict(rows[-1])
    final.update(final=True, n_params=n_params, sample=text,
                 total_wall_s=round(time.time() - t0, 1), **ext)
    with log_path.open("a") as f:
        f.write(json.dumps(final) + "\n")
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=["morphssm:bp", "morphssm:cascade"],
                   help="arch:credit; arch in {morphssm, convrnn, convrnn-norecur}, "
                        "credit in {bp, cascade, relax_async}")
    p.add_argument("--gradcheck", action="store_true", help="run G0 and exit")
    p.add_argument("--gc-seeds", type=int, default=3)
    p.add_argument("--text", type=Path, default=REPO / "data" / "tinyshakespeare.txt")
    p.add_argument("--out", type=Path, default=REPO / "results" / "pc_convrnn")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=7)
    p.add_argument("--kernel", type=int, default=7)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--tau-floor", type=float, default=0.0)
    p.add_argument("--no-anneal", dest="anneal", action="store_false")
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--pc-lr", type=float, default=None,
                   help="PC-arm lr (default adam_lr/3 — the measured rule)")
    p.add_argument("--participation", type=float, default=0.5)
    p.add_argument("--lam-min", type=float, default=2 ** -6)
    p.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"])
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--opt", type=str, default="adamw",
                   choices=["adamw", "puretropical", "tropicallion"])
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "lut"])
    p.add_argument("--lr-shift", type=int, default=10)
    p.add_argument("--strict-audit", action="store_true",
                   help="G5: trace one tau=0 cascade+step for hardware multiplies")
    p.add_argument("--ckpt-name", type=str, default=None,
                   help="checkpoint filename override (default arch_credit_seed.pt)")
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-long", type=int, nargs="*", default=[512, 1024],
                   help="length-extrapolation eval T's (final eval only)")
    p.add_argument("--sample-chars", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    if args.pc_lr is None:
        args.pc_lr = args.adam_lr / 3.0

    args.out.mkdir(parents=True, exist_ok=True)
    if args.gradcheck:
        ok = gradcheck(args, out_path=args.out / "g0_gradcheck.jsonl")
        raise SystemExit(0 if ok else 1)

    data = (BPEData(args.text, args.vocab_size) if args.tokenizer == "bpe"
            else CharData(args.text))
    log_path = args.out / f"g3_{args.tag}.jsonl"
    print(f"tiny-shakespeare: vocab={data.vocab} train={data.train_ids.numel()} "
          f"val={data.val_ids.numel()}")
    finals = []
    for arm in args.arms:
        arch, credit = arm.split(":")
        credit = {"relax_async": "relax_async", "relax": "relax_async",
                  "bp": "bp", "cascade": "cascade"}[credit]
        finals.append(run_one(arch, credit, data, args, log_path))
        print()
    print(f"{'arm':24s} {'params':>8s} {'hard_ce':>8s} {'bpc':>6s} {'soft_ce':>8s}")
    for r in finals:
        print(f"{r['run']:24s} {r['n_params']:8d} {r['val_ce_hard']:8.4f} "
              f"{r['val_ce_hard'] / math.log(2):6.3f} {r['val_ce_soft']:8.4f}")


if __name__ == "__main__":
    main()
