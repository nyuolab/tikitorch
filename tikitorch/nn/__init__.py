"""tikitorch.nn — tropical-semiring neural-network layers (a la ``torch.nn``).

Foundational layer modules:
    TropicalSoftMin, TropicalSoftMax          — soft (logsumexp) relaxations
    TropicalLinearModule  (alias TropicalLinear)     — min-plus erosion linear
    TropicalDilationModule (alias TropicalDilation)  — max-plus dilation linear
    CompositionalTropicalProjection (alias TropicalProjection)
    TropicalTopKModule (alias TropicalTopK)

Higher-level layers are appended below the marker. Every layer here
wraps the stateless ops in :mod:`tikitorch.functional`; the math is ported
verbatim from the validated research code.

Naming convention. Modules carry their original research names *and* a clean,
PyTorch-idiomatic alias (e.g. ``TropicalLinearModule`` is also ``TropicalLinear``).
Both names refer to the same class; prefer the clean alias in new code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch.functional import (
    adder_linear,
    tropical_asym_score,
    tropical_conv2d,
    tropical_cross_entropy,
    tropical_dilation,
    tropical_dot_product,
    tropical_hardtanh,
    tropical_hilbert_score,
    tropical_l1_score,
    tropical_l_inf_score,
    tropical_leaky_relu,
    tropical_linear,
    tropical_perceptron_loss,
    tropical_relu,
    tropical_softmax,
    tropical_softmax_asym_score,
    tropical_softmax_dot_product,
    tropical_softmax_hilbert_score,
    tropical_softmax_l_inf_score,
    tropical_softmax_value_pull,
    tropical_value_pull_topk,
    tropical_value_pull_iter_topk,
    tropical_softmax_value_pull_iter_topk,
    tropical_value_pull_min,
    tropical_softmax_value_pull_min,
    tropical_value_gather,
    tropical_softmax_value_gather,
    tropical_softmin,
    tropical_value_pull,
)


def _make_norm(norm_type: str, d_model: int):
    """Build a normalization module by name.

    ``"tropical"`` (default everywhere) is the multiplier-free
    :class:`TropicalLayerNorm` (subtract-mean + power-of-2 mean-abs rescale) that
    replaces ``nn.LayerNorm``'s variance/divide/affine multiplies — closing the
    N1 closure hole.  ``"layernorm"`` keeps standard ``nn.LayerNorm`` as a
    declared DSP boundary; ``"tropical_rms"``/``"tropical_shift"`` are the
    cheaper closed variants.  (Names resolve at call time, so the norm classes
    defined later in this module are visible.)
    """
    if norm_type in ("tropical", "tropical_layernorm"):
        return TropicalLayerNorm(d_model)
    if norm_type == "tropical_rms":
        return TropicalRMSNorm(d_model)
    if norm_type == "tropical_shift":
        return TropicalBatchShift()
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    raise ValueError(f"unknown norm_type {norm_type!r}; expected 'tropical' "
                     "(or 'tropical_layernorm'), 'tropical_rms', 'tropical_shift', "
                     "or 'layernorm'")


_SCORE_VARIANTS = ("dot_product", "l_inf", "hilbert", "hilbert_asym", "l1")


def _tropical_score(q, k, variant, tau):
    """Tropical attention scores by variant. tau>0 uses the soft relaxation.

    - ``dot_product``: max_d(Q+K) (max-plus inner product; can be K-dominated).
    - ``l_inf``: -max_d|Q-K| (L-infinity distance, symmetric).
    - ``hilbert``: -(max_d - min_d)(Q-K), the tropical/Hilbert *projective* metric
      (translation/scale-invariant, sharp polyhedral boundaries; Hashemi et al.,
      NeurIPS 2025). All three are multiplier-free.
    """
    if variant == "dot_product":
        return (tropical_softmax_dot_product(q, k, tau=tau) if tau > 0
                else tropical_dot_product(q, k))
    if variant == "l_inf":
        return (tropical_softmax_l_inf_score(q, k, tau=tau) if tau > 0
                else tropical_l_inf_score(q, k))
    if variant == "hilbert":
        return (tropical_softmax_hilbert_score(q, k, tau=tau) if tau > 0
                else tropical_hilbert_score(q, k))
    if variant == "hilbert_asym":
        return (tropical_softmax_asym_score(q, k, tau=tau) if tau > 0
                else tropical_asym_score(q, k))
    if variant == "l1":
        # -sum_d|Q-K| is already smooth + dense-gradient, so soft == hard (no
        # logsumexp relaxation needed); the same op for any tau.
        return tropical_l1_score(q, k)
    raise ValueError(f"unknown score_variant {variant!r}; expected one of {_SCORE_VARIANTS}")


_AGG_VARIANTS = ("argmax", "iter_topk", "min", "maxmin", "sinkhorn", "gather")


def _max_pull(v, scores, tau, topk_k, rank_bias, value_topk_k):
    """Max-plus (dilation) value aggregation: argmax / iter-topk, soft on tau>0."""
    if topk_k and topk_k > 1 and rank_bias is not None:
        if tau > 0:
            return tropical_softmax_value_pull_iter_topk(v, scores, k=topk_k, tau=tau, rank_bias=rank_bias)
        return tropical_value_pull_iter_topk(v, scores, k=topk_k, rank_bias=rank_bias)
    if tau > 0:
        return tropical_softmax_value_pull(v, scores, tau=tau)
    if value_topk_k > 1:
        return tropical_value_pull_topk(v, scores, value_topk_k)
    return tropical_value_pull(v, scores)


def _min_pull(v, scores, tau):
    """Min-plus (erosion) value aggregation: out[i,d] = min_j(v[j,d]+scores[i,j])."""
    if tau > 0:
        return tropical_softmax_value_pull_min(v, scores, tau=tau)
    return tropical_value_pull_min(v, scores)


def _causal_mask_scores(scores, fill):
    """Mask future keys (upper triangle, j>i) to ``fill`` on the (..., Lq, Lk) matrix."""
    Lq, Lk = scores.shape[-2], scores.shape[-1]
    m = torch.triu(torch.ones(Lq, Lk, device=scores.device, dtype=torch.bool), diagonal=1)
    return scores.masked_fill(m, fill)


def _sinkhorn_aggregate(v, s_max, tau, sink_eps, sink_iters):
    """Permutation/assignment routing (densifies the gradient + changes the routing
    CLASS from value-copy to assignment). Log-domain Sinkhorn normalizes the score
    graph S into a (causal) doubly-stochastic transport plan P, so EVERY key carries
    column mass -> dense gradient, no dark routes (vs the per-(i,d) argmax scatter).

      train (tau>0): out = P @ V   (soft transport; the P@V is the only train-time
                    multiply, deleted at deploy — north-star: train-relaxation OK)
      deploy (tau=0): out[i] = V[argmax_j logP[i,j]]   (hard gather = a generalized
                    permutation matrix = the tropical-isometric map, zero multiply)

    s_max is already causal-masked (j>i = -inf) by the caller, which makes the
    column normalization a triangular partial-OT (inherent to autoregressive masks).
    Every normalization op is +/-/logsumexp (== tropical_softmax, the on-silicon LUT).
    """
    logP = s_max / max(sink_eps, 1e-6)
    for _ in range(sink_iters):
        logP = logP - torch.logsumexp(logP, dim=-1, keepdim=True)   # rows: over keys j
        col = torch.logsumexp(logP, dim=-2, keepdim=True)           # cols: over queries i
        logP = logP - torch.nan_to_num(col, neginf=0.0)             # masked cols (all -inf) -> no-op
    if tau > 0:
        P = torch.nan_to_num(logP.exp(), nan=0.0)                   # masked entries -> 0
        return torch.einsum("...ij,...jd->...id", P, v)
    # deploy: hard per-query assignment (cheap discriminator; auction would enforce
    # a strict bijection — not needed for the prototype).
    pi = logP.argmax(dim=-1)                                        # (..., Lq)
    return torch.gather(v, -2, pi.unsqueeze(-1).expand(*pi.shape, v.shape[-1]))


def _tropical_aggregate(v, scores, variant, tau, topk_k, rank_bias, value_topk_k,
                        causal=False, sink_eps=1.0, sink_iters=5):
    """Aggregate values by tropical attention scores, by variant. tau>0 = soft.

    - ``argmax`` (default): per-(query,dim) single argmax (max-plus) over keys
      (``value_topk_k>1`` is a legacy mean-pull softening). The historical path.
    - ``iter_topk``: iterative-argmax winner-exclusion top-k, SUM-combined with an
      optional additive per-rank bias -- strictly multiplier-free, denser grad.
    - ``min``: min-plus (erosion) pull -- the natural primitive for *min*-plus DPs
      (shortest path, min-coin-change) that a pure max-plus aggregator can't express.
    - ``maxmin``: split the head dim in half -- max-plus on one half, min-plus on
      the other -- so one model carries BOTH semirings (a MaxPlusMin-style universal
      aggregator). Each half uses the full scores.
    All anneal soft<->hard on ``tau``; ``argmax``/``iter_topk`` reduce to the single
    argmax at k=1, and ``min`` is the (min,+) twin via -max(-.). Causal masking is
    applied here so it is semiring-correct: max-plus paths mask future keys with
    -inf (max ignores them); min-plus paths mask with +inf (min ignores them).
    """
    NEG, POS = float("-inf"), float("inf")
    s_max = _causal_mask_scores(scores, NEG) if causal else scores
    s_min = _causal_mask_scores(scores, POS) if causal else scores
    if variant == "iter_topk":
        rb = rank_bias if rank_bias is not None else None
        if tau > 0:
            return tropical_softmax_value_pull_iter_topk(v, s_max, k=topk_k, tau=tau, rank_bias=rb)
        return tropical_value_pull_iter_topk(v, s_max, k=topk_k, rank_bias=rb)
    if variant == "min":
        return _min_pull(v, s_min, tau)
    if variant == "maxmin":
        d = v.shape[-1]
        h = d // 2
        if h == 0:                                  # too small to split -> max-plus
            return _max_pull(v, s_max, tau, 1, None, value_topk_k)
        hi = _max_pull(v[..., :h], s_max, tau, 1, None, value_topk_k)
        lo = _min_pull(v[..., h:], s_min, tau)
        return torch.cat([hi, lo], dim=-1)
    if variant == "sinkhorn":
        return _sinkhorn_aggregate(v, s_max, tau, sink_eps, sink_iters)
    if variant == "gather":
        # argmax-GATHER: the deploy-pure limit of softmax ATTENTION (ONE key per
        # query, shared across dims) -- NOT the per-dim max-plus matmul below.
        # train (tau>0): softmax(scores)@V dense-gradient relaxation; deploy
        # (tau=0): V[argmax_j scores] (zero multiply). s_max already has future
        # keys at -inf, correct for both softmax (->0 weight) and argmax.
        if tau > 0:
            return tropical_softmax_value_gather(v, s_max, tau=tau)
        return tropical_value_gather(v, s_max)
    # "argmax" (default)
    if tau > 0:
        return tropical_softmax_value_pull(v, s_max, tau=tau)
    if value_topk_k > 1:
        return tropical_value_pull_topk(v, s_max, value_topk_k)
    return tropical_value_pull(v, s_max)


__all__ = [
    # predictive coding (local, backprop-free)
    "TropicalPCLayer",
    "TropicalPCNet",
    "DyadicPCNet",
    # soft relaxations
    "TropicalSoftMin",
    "TropicalSoftMax",
    # core linear layers
    "TropicalLinearModule",
    "TropicalLinear",
    "TropicalDilationModule",
    "TropicalDilation",
    # compositional / top-k projections
    "CompositionalTropicalProjection",
    "TropicalProjection",
    "TropicalTopKModule",
    "TropicalTopK",
    # activation + dropout modules (closure-clean)
    "TropicalReLU",
    "TropicalLeakyReLU",
    "TropicalHardtanh",
    "TropicalDropout",
    # loss modules (nn.Module wrappers over functional losses)
    "TropicalCrossEntropyLoss",
    "TropicalPerceptronLoss",
    # dyadic (power-of-two) multiplier-free oblique layers
    "DyadicLinear",
    "DyadicTropicalProjection",
    "dyadic_quantize",   # public weight-quantizer util (used by fpga export + PC layers)
    # --- higher-level layers ---
    # normalization
    "TropicalBatchShift",
    "BatchShift",
    # morphological convolutions
    "DepthwiseMorphConv1d",
    "MorphConv1d",
    "MultiScaleMorphConv1d",
    "MorphSSMLayer",
    # 2D morphological conv + pooling + CNN block (vision)
    "TropicalConv2d",
    "TropicalMaxPool2d",
    "TropicalMinPool2d",
    "TropicalAdaptiveMaxPool2d",
    "TropicalGlobalMaxPool",
    "MorphConvBlock",
    # multiplier-free normalization (replaces nn.LayerNorm)
    "TropicalLayerNorm",
    "TropicalLayerNorm2d",
    "TropicalRMSNorm",
    # tropical recurrent cells (RNN)
    "TropicalRNNCell",
    "TropicalRNN",
    # closed input embedding for continuous features
    "TropicalQuantizedEmbed",
    # deep-morph training fixes: closed scale + Max-Plus-Min universal layer
    "TropicalScale",
    "MaxPlusMin",
    # P1: dilation-erosion perceptron, opening/closing, AdderNet l1 layer
    "DilationErosionPerceptron",
    "MorphOpenClose2d",
    "AdderLinear",
    # tropical attention / transformer
    "TropicalAttention",
    "TropicalShiftAttention",
    "TropicalTransformerBlock",
    # mixture-of-tropical-experts
    "TropicalRouter",
    "MoTEBlock",
]


# ---------------------------------------------------------------------------
# Soft-min / soft-max relaxation modules
# ---------------------------------------------------------------------------


# Max-plus / min-plus linears materialize a (batch, out, in) intermediate (there is
# no GEMM equivalent for the (max,+) semiring). For a large layer -- e.g. an LM FFN
# d->4d at long sequence -- that tensor is many GB and OOMs in eager PyTorch. We tile
# over the OUTPUT dim so peak memory is bounded, reusing the SAME (tested) ops on
# weight-row slices: identical math, just chunked, and autograd flows through the
# slices. Small layers (under the threshold) take the original single-shot path with
# zero overhead. This is purely an eager-mode fix -- on FPGA/ASIC the max-plus
# reduction streams with no materialization (the whole point of being multiplier-free).
_MAXPLUS_CHUNK_ELEMS = 32_000_000   # (batch*out*in) above this -> tile over out


def _chunked_linear(op, x, weight, bias):
    """``op(x, weight_rows) -> (..., out_rows)``, tiled over output features when the
    (batch, out, in) intermediate would exceed ``_MAXPLUS_CHUNK_ELEMS``. ``x`` is
    (..., in); ``weight`` is (out, in); ``bias`` is (out,)."""
    out_features, in_features = weight.shape
    batch = x.numel() // in_features
    if batch * out_features * in_features <= _MAXPLUS_CHUNK_ELEMS:
        return op(x, weight) + bias
    chunk = max(1, _MAXPLUS_CHUNK_ELEMS // max(1, batch * in_features))
    return torch.cat([op(x, weight[s:s + chunk]) + bias[s:s + chunk]
                      for s in range(0, out_features, chunk)], dim=-1)


class TropicalSoftMin(nn.Module):
    """Differentiable tropical linear using softmin with temperature tau."""

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """x: (batch, in), weight: (out, in) -> (batch, out)."""
        expanded = weight.unsqueeze(0) + x.unsqueeze(1)  # (batch, out, in)
        return tropical_softmin(expanded, tau=self.tau, dim=-1)


class TropicalSoftMax(nn.Module):
    """Differentiable tropical dilation using softmax with temperature tau."""

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """x: (batch, in), weight: (out, in) -> (batch, out)."""
        expanded = weight.unsqueeze(0) + x.unsqueeze(1)
        return tropical_softmax(expanded, tau=self.tau, dim=-1)


# ---------------------------------------------------------------------------
# Max-plus (dilation) linear layer
# ---------------------------------------------------------------------------


class TropicalDilationModule(nn.Module):
    """Max-plus linear layer: y_j = max_i(W[j,i] + x[i]) + b_j.

    Dual of TropicalLinearModule (which computes min-plus / erosion).
    Supports both hard (custom autograd) and soft-max mode.
    """

    def __init__(self, in_features: int, out_features: int,
                 use_softmax: bool = False, tau: float = 1.0,
                 init_identity: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_softmax = use_softmax
        self.tau = tau
        # init_identity seeds the dilation as the identity map (W[j,j]=0,
        # W[j,i!=j]=-big), so y_j = max_i(W[j,i]+x[i]) ~= x_j at init. A plain
        # uniform init makes a single dilation collapse the value signal (one
        # input coord wins the max for many outputs, so per-dim diversity is
        # lost) -- fatal when a dilation is used as an attention Q/K/V projection
        # feeding per-dim max-plus aggregation. Identity init preserves per-dim
        # signal and lets training move away from it. Square layers only.
        self.init_identity = init_identity and (in_features == out_features)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._softmax = TropicalSoftMax(tau=tau)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_identity:
            with torch.no_grad():
                self.weight.fill_(-30.0)
                self.weight.fill_diagonal_(0.0)
        else:
            nn.init.uniform_(self.weight, -0.5, 0.5)
        nn.init.zeros_(self.bias)

    def set_tau(self, tau: float):
        # Flip into the soft (logsumexp) path for tau>0 and the hard (argmax)
        # path at tau==0 -- otherwise a module built use_softmax=False stays hard
        # through the whole soft->hard curriculum (the soft-anneal-freeze bug).
        self.tau = tau
        self.use_softmax = tau > 0
        self._softmax.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = self._softmax if self.use_softmax else tropical_dilation
        return _chunked_linear(op, x, self.weight, self.bias)

    @torch.no_grad()
    def forward_with_indices(self, x: torch.Tensor):
        """Forward pass that also returns argmax indices for routing.

        Unlike the standard forward, this does NOT use custom autograd —
        it's intended for MoTE routing where we need the argmax pattern
        but don't need gradients through the routing decision.

        Args:
            x: (batch, in_features)
        Returns:
            out: (batch, out_features) — max-plus output + bias
            indices: (batch, out_features) — argmax index per output unit
        """
        expanded = self.weight.unsqueeze(0) + x.unsqueeze(1)  # (batch, out, in)
        out, indices = expanded.max(dim=-1)  # (batch, out)
        return out + self.bias, indices

    @torch.no_grad()
    def forward_with_routing(self, x: torch.Tensor):
        """Forward returning the FROZEN routing for a PC cascade sweep.

        Hard path (``use_softmax=False`` / tau==0): returns ``(y, indices)`` with
        ``indices`` the (batch, out) argmax winners — the cacheable routing whose
        adjoint is a pure scatter (zero multiplies). Soft path (tau>0): returns
        ``(y, None)`` — the softmax routing weights are (batch, out, in) and are
        deliberately NOT materialized (600MB-class at d=256, B·T≈2k rows); the PC
        engine recomputes them chunk-wise from the cached input during the sweep.
        Tiled over output features like ``_chunked_linear`` so the transient stays
        bounded. Matches ``forward`` bit-exactly on both paths.
        """
        out_features, in_features = self.weight.shape
        rows = x.numel() // in_features
        xf = x.reshape(rows, in_features)
        chunk = out_features if rows * out_features * in_features <= _MAXPLUS_CHUNK_ELEMS else \
            max(1, _MAXPLUS_CHUNK_ELEMS // max(1, rows * in_features))
        outs, idxs = [], []
        for s in range(0, out_features, chunk):
            w = self.weight[s:s + chunk]
            expanded = w.unsqueeze(0) + xf.unsqueeze(1)      # (rows, chunk, in)
            if self.use_softmax:
                y = tropical_softmax(expanded, tau=self.tau, dim=-1)
            else:
                y, idx = expanded.max(dim=-1)
                idxs.append(idx)
            outs.append(y + self.bias[s:s + chunk])
        y = torch.cat(outs, dim=-1).reshape(*x.shape[:-1], out_features)
        if self.use_softmax:
            return y, None
        return y, torch.cat(idxs, dim=-1).reshape(*x.shape[:-1], out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"use_softmax={self.use_softmax}, tau={self.tau}"
        )


# ---------------------------------------------------------------------------
# Min-plus (erosion) linear layer
# ---------------------------------------------------------------------------


class TropicalLinearModule(nn.Module):
    """Min-plus linear layer (erosion): y_j = min_i(W[j,i] + x[i]) + b_j.

    Wraps TropicalLinearFn as a standard nn.Module.
    Supports both hard tropical (custom autograd) and soft-min mode.
    """

    def __init__(self, in_features: int, out_features: int, use_softmin: bool = False, tau: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_softmin = use_softmin
        self.tau = tau

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._softmin = TropicalSoftMin(tau=tau)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize near zero so min-plus sums stay bounded
        nn.init.uniform_(self.weight, -0.5, 0.5)
        nn.init.zeros_(self.bias)

    def set_tau(self, tau: float):
        """Update temperature for softmin annealing (and flip into/out of soft)."""
        # tau>0 -> soft (logsumexp) path; tau==0 -> hard (argmin). Without this
        # flip a module built use_softmin=False never softens (soft-anneal freeze).
        self.tau = tau
        self.use_softmin = tau > 0
        self._softmin.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = self._softmin if self.use_softmin else tropical_linear
        return _chunked_linear(op, x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"use_softmin={self.use_softmin}, tau={self.tau}"
        )


# ---------------------------------------------------------------------------
# Compositional projection: Dilation -> Erosion (= tropical rational map)
# ---------------------------------------------------------------------------
#
# Single-step TropicalDilationModule has insufficient capacity for QKV
# projections (1-of-d sparsity per output dim). The Dilation -> Erosion
# composition computes a tropical rational function — the full class of
# piecewise-affine maps. This is the projection capacity that matches
# nn.Linear's expressive class while staying tropical.
#
# Forward path dispatches on tau:
#   tau == 0  : hard custom-autograd path (closure-preserving, FPGA target).
#   tau > 0   : soft logsumexp path (dense gradients during training).
# Both paths share the same weight tensors W1, W2 — only the operator changes.
# This implements the Maslov-dequantization curriculum: train soft, anneal
# tau -> 0, deploy hard with no parameter copying.


class CompositionalTropicalProjection(nn.Module):
    """Dilation (max-plus) -> Erosion (min-plus) projection.

    y = min_k(W2[j,k] + max_i(W1[k,i] + x[i]) + b1[k]) + b2[j]

    in_features -> d_intermediate (max-plus) -> out_features (min-plus).
    """

    def __init__(self, in_features: int, out_features: int,
                 d_intermediate=None, tau: float = 0.0):
        super().__init__()
        if d_intermediate is None:
            d_intermediate = max(in_features, out_features) * 2
        self.in_features = in_features
        self.out_features = out_features
        self.d_intermediate = d_intermediate
        self.tau = tau

        # First layer: Dilation (max-plus): in_features -> d_intermediate
        self.W1 = nn.Parameter(torch.empty(d_intermediate, in_features))
        self.b1 = nn.Parameter(torch.zeros(d_intermediate))
        # Second layer: Erosion (min-plus): d_intermediate -> out_features
        self.W2 = nn.Parameter(torch.empty(out_features, d_intermediate))
        self.b2 = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.W1, -0.5, 0.5)
        nn.init.uniform_(self.W2, -0.5, 0.5)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features). Step 1 expands to d_intermediate via dilation.
        if self.tau > 0:
            # Soft path: logsumexp gives dense gradients.
            expanded1 = self.W1 + x.unsqueeze(-2)  # (..., d_intermediate, in)
            h = tropical_softmax(expanded1, tau=self.tau, dim=-1) + self.b1
        else:
            # Hard path: custom autograd through 1-of-d argmax.
            # tropical_dilation expects (batch, in) - (out, in) -> (batch, out).
            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features)
            h_flat = tropical_dilation(x_flat, self.W1)
            h = h_flat.reshape(*orig_shape[:-1], self.d_intermediate) + self.b1

        # Step 2: erosion (min-plus): d_intermediate -> out_features.
        if self.tau > 0:
            expanded2 = self.W2 + h.unsqueeze(-2)  # (..., out, d_intermediate)
            y = tropical_softmin(expanded2, tau=self.tau, dim=-1) + self.b2
        else:
            h_shape = h.shape
            h_flat = h.reshape(-1, self.d_intermediate)
            y_flat = tropical_linear(h_flat, self.W2)
            y = y_flat.reshape(*h_shape[:-1], self.out_features) + self.b2

        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"d_intermediate={self.d_intermediate}, "
            f"out_features={self.out_features}, tau={self.tau}"
        )


# ---------------------------------------------------------------------------
# Top-K aggregation projection
# ---------------------------------------------------------------------------
#
# Replaces y_j = max_i(W[j,i] + x[i]) with y_j = sum_topK_i(W[j,i] + x[i]).
# Top-K is comparison-only (closure preserved); sum of K values is K-1
# additions. Gradient sparsity: K/in_features, vs. 1/in_features for single
# dilation. K=1 reduces to dilation; K=in_features reduces to a sum-of-all
# linear with shifted weights.
#
# torch.topk has autograd built in — it routes gradient only through the K
# winning indices, identical to what a hand-written custom autograd would do.


class TropicalTopKModule(nn.Module):
    """Top-K max-plus aggregation: y_j = sum_{i in topK}(W[j,i] + x[i]) + b_j.

    Closure: top-K is purely comparison-based (selection); sum of K is
    K-1 additions; bias is one more addition. Pure tropical operations.
    """

    def __init__(self, in_features: int, out_features: int,
                 k: int = 4, tau: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = min(k, in_features)
        self.tau = tau  # accepted for API compatibility; topk is hard by nature

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.5, 0.5)
        nn.init.zeros_(self.bias)

    def set_tau(self, tau: float) -> None:
        # Accepted for symmetry with other tropical projections; the top-K
        # selection itself does not have a meaningful soft analogue here.
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features). Add weight and select top-K along input dim.
        expanded = self.weight + x.unsqueeze(-2)  # (..., out, in)
        topk_vals, _ = expanded.topk(self.k, dim=-1)  # (..., out, k)
        return topk_vals.sum(dim=-1) + self.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"k={self.k}, tau={self.tau}"
        )


# ---------------------------------------------------------------------------
# Dyadic (power-of-two) multiplier-free OBLIQUE layers
# ---------------------------------------------------------------------------
#
# The max-plus-linear primitive ``max_i(W[k,i] + x_i)`` has a one-hot input
# gradient, so its piecewise-linear normals are exactly the axis vectors {e_i} --
# it cannot represent OBLIQUE value/policy surfaces (the measured §2a underfit;
# see results/capacity_probe/FIX_FINDINGS.md). Obliqueness in the tropical
# semiring comes from MONOMIAL EXPONENTS (repeated addition): a weight ``w = +/-
# 2^k`` applied as ``w * x`` is a bit-SHIFT, and ``sum_i (+/-2^{k_i}) x_i`` is a
# shift+add-tree -- ZERO multiply, ZERO DSP -- whose gradient is a dense
# signed-dyadic (oblique) normal. Validated to match a dense ReLU net on oblique
# and smooth value where the max-plus-linear core underfits 1.5-1.7x
# (scripts/dyadic_probe.py, n=5).


def dyadic_quantize(w: torch.Tensor) -> torch.Tensor:
    """Round each weight to the nearest SIGNED power of two (straight-through).

    forward:  ``w -> sign(w) * 2^round(log2|w|)`` (a bit-shift on fabric);
    backward: identity (STE through the round). The exponents freeze to a fixed
    shift schedule at deploy, so inference is multiplier-free.
    """
    mag = w.abs().clamp_min(1e-12)
    q = torch.sign(w) * torch.exp2(torch.log2(mag).round())
    return w + (q - w).detach()


class DyadicLinear(nn.Linear):
    """``nn.Linear`` with weights constrained to signed powers of two (multiplier-free).

    ``w_ij = +/- 2^{k_ij}`` so ``W x`` is a sum of bit-shifts (shift + add-tree,
    zero DSP). Identical parameters/shape to ``nn.Linear``; the dyadic constraint
    is applied in ``forward`` via a straight-through estimator.

    ``_zero_dsp_layer = True`` marks this as a declared zero-DSP layer so the
    *structural* :func:`tikitorch.audit.assert_closed` does not flag it as a
    multiply (it is an ``nn.Linear`` subclass, but its deployed weights are pure
    powers of two — a shift, not a DSP multiply). The authoritative numeric check
    that the weights really are powers of two is
    :func:`tikitorch.fpga.dyadic_export.verify_schedule` (integer exponents); the
    runtime op-tracer still sees the float-sim matmul + STE, which is expected —
    closure for dyadic is a weight-value property, verified statically, not by the
    forward op trace.

    SCOPE: the marker is **forward only**. The ``W·x`` *forward* is zero-DSP. The
    autograd *backward* is NOT: the input-gradient ``Wᵀ·dL/dy`` is shift+add, but
    the weight-gradient ``dL/dy ⊗ x`` multiplies two arbitrary tensors — a real
    multiply that :func:`tikitorch.audit.find_training_multiplies` reports. A
    backward-zero-DSP training step needs a sign-only rule
    (``sign(dL/dW)=sign(dL/dy)⊕sign(x)``, then a SignSGD update in the (sign,
    exponent) domain so weights stay powers of two); that is the on-fabric training
    kernel, and it is verified with ``assert_training_closed`` — never inferred from
    this marker.
    """

    _zero_dsp_layer = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, dyadic_quantize(self.weight), self.bias)


class DyadicTropicalProjection(nn.Module):
    """Multiplier-free OBLIQUE projection: dyadic monomial -> tropical max -> dyadic monomial.

    A drop-in alternative to :class:`CompositionalTropicalProjection` for the
    components that must represent oblique/smooth surfaces (value readout,
    representation). ``in_features -> d_intermediate`` (dyadic, oblique normals)
    -> ReLU (``= max(.,0)``, the tropical non-convexity) -> ``d_intermediate ->
    out_features`` (dyadic). Everything is shift + add + compare: zero multiply,
    zero DSP. ``tau`` is accepted for interface parity with the max-plus
    projections but is inert (the dyadic forward deploys as-is).
    """

    def __init__(self, in_features: int, out_features: int,
                 d_intermediate=None, tau: float = 0.0):
        super().__init__()
        if d_intermediate is None:
            d_intermediate = max(in_features, out_features) * 2
        self.in_features = in_features
        self.out_features = out_features
        self.d_intermediate = d_intermediate
        self.tau = tau
        self.lin1 = DyadicLinear(in_features, d_intermediate)
        self.lin2 = DyadicLinear(d_intermediate, out_features)

    def set_tau(self, tau: float) -> None:
        self.tau = tau            # inert; kept for parity with TropicalProjection.set_tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.relu(self.lin1(x)))

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, d_intermediate={self.d_intermediate}, "
                f"out_features={self.out_features} (dyadic, multiplier-free)")


# ---------------------------------------------------------------------------
# PyTorch-idiomatic aliases (clean names) for the core layers
# ---------------------------------------------------------------------------

TropicalLinear = TropicalLinearModule
TropicalDilation = TropicalDilationModule
TropicalProjection = CompositionalTropicalProjection
TropicalTopK = TropicalTopKModule


# ---------------------------------------------------------------------------
# Activation + dropout MODULES (nn.Module wrappers over the closed functional ops)
# ---------------------------------------------------------------------------
#
# torch.nn.ReLU is already multiplier-free, but nn.LeakyReLU(slope) hides a non-pow2
# ``x*slope`` multiply and nn.Dropout hides the ``1/(1-p)`` survivor rescale — both
# break training closure. These give a torch-familiar, Sequential-friendly,
# closure-clean set (each carries an inert set_tau for parity with the tau cascade).


class TropicalReLU(nn.Module):
    """ReLU = ``max(x, 0)`` as a Module — multiplier-free (tropical max-with-0)."""

    def forward(self, x):
        return tropical_relu(x)

    def set_tau(self, tau):
        pass


class TropicalLeakyReLU(nn.Module):
    """Leaky ReLU with a POWER-OF-2 negative slope: ``max(x, x >> shift)`` (slope 2**-shift).

    Unlike ``nn.LeakyReLU(0.01)`` (a non-pow2 ``x*0.01`` multiply), the slope is an
    arithmetic right-shift — multiplier-free / zero-DSP.
    """

    def __init__(self, shift: int = 3):
        super().__init__()
        self.shift = int(shift)

    def forward(self, x):
        return tropical_leaky_relu(x, shift=self.shift)

    def set_tau(self, tau):
        pass

    def extra_repr(self):
        return f"slope=2**-{self.shift}"


class TropicalHardtanh(nn.Module):
    """Hard tanh = ``clamp(x, lo, hi)`` = ``min(max(x, lo), hi)``. Pure compare, no multiply."""

    def __init__(self, lo: float = -1.0, hi: float = 1.0):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return tropical_hardtanh(x, self.lo, self.hi)

    def set_tau(self, tau):
        pass

    def extra_repr(self):
        return f"lo={self.lo}, hi={self.hi}"


class TropicalDropout(nn.Module):
    """Closure-clean dropout: a ``{0,1}`` GATE, with NO ``1/(1-p)`` survivor rescale.

    Standard ``nn.Dropout`` scales survivors by ``1/(1-p)`` — a non-power-of-2 real
    multiply that silently breaks *training* closure (``assert_training_closed`` flags
    it). ``TropicalDropout`` drops by masking to zero only (gating = a mux/AND, which the
    closure audit treats as a shift, not a DSP multiply); at ``eval()`` it is the
    identity. The (small) train/eval activation-scale shift this trades away is the
    price of a zero-DSP training step.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout p must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)    # {0,1} gate (no rescale)
        return x * mask

    def set_tau(self, tau):
        pass

    def extra_repr(self):
        return f"p={self.p}"


# ---------------------------------------------------------------------------
# Loss modules (``nn.Module`` wrappers around the functional losses)
#
# PyTorch ships its losses both as functions (``F.cross_entropy``) and as
# Modules (``nn.CrossEntropyLoss``); a Module is convenient when you want the
# loss to live in a config / be moved with ``.to()`` / be ``repr``-able. These
# are thin, stateless wrappers over :mod:`tikitorch.functional` — they hold the
# closure-relevant config and forward ``(logits, target)``.
# ---------------------------------------------------------------------------


class TropicalCrossEntropyLoss(nn.Module):
    """Closure-faithful cross-entropy loss, as a Module (cf. ``nn.CrossEntropyLoss``).

    Wraps :func:`tikitorch.functional.tropical_cross_entropy`. The CE gradient is
    ``softmax(logits) - onehot(target)``; ``softmax="exact"`` conditions it with a
    host softmax (the silicon-validated path) while ``softmax="lut"`` computes the
    softmax from a subtract-max + power-of-2-addressed exp lookup + shift-normalize,
    so the *whole* loop stays multiplier-free. Extra keyword args (``lut_size``,
    ``z_range``, ``normalize``) pass through to
    :func:`tikitorch.functional.shift_softmax`.

    Args:
        softmax: ``"exact"`` (default) or ``"lut"`` (fully multiplier-free).
        **lut_kwargs: forwarded to ``shift_softmax`` when ``softmax="lut"``.
    """

    def __init__(self, softmax: str = "exact", **lut_kwargs):
        super().__init__()
        self.softmax = softmax
        self.lut_kwargs = lut_kwargs

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tropical_cross_entropy(logits, target, softmax=self.softmax, **self.lut_kwargs)

    def set_tau(self, tau):
        pass

    def extra_repr(self) -> str:
        extra = "".join(f", {k}={v}" for k, v in self.lut_kwargs.items())
        return f"softmax={self.softmax!r}{extra}"


class TropicalPerceptronLoss(nn.Module):
    """Sign-only perceptron loss, as a Module (cf. ``nn.MultiMarginLoss``).

    Wraps :func:`tikitorch.functional.tropical_perceptron_loss`: the gradient is a
    2-hot ``+/-1`` vector (push the target logit up, the argmax distractor down), so
    it is multiplier-free without any softmax. It is the cheapest closed loss but
    also the single biggest accuracy cost in the closure penalty (~-17pp on a hard
    char-LM vs cross-entropy); prefer :class:`TropicalCrossEntropyLoss` unless you
    specifically want the silicon perceptron baseline.
    """

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tropical_perceptron_loss(logits, target)

    def set_tau(self, tau):
        pass


# --- higher-level layers ---
#
# The modules below are ``nn.Module`` building blocks (NOT the full
# architectures, which live in :mod:`tikitorch.models`). They are ported
# verbatim from the validated research sources:
#
#   morphological building blocks -> TropicalBatchShift, DepthwiseMorphConv1d,
#                                    MultiScaleMorphConv1d, MorphSSMLayer
#   tropical transformer          -> TropicalAttention,
#                                    TropicalShiftAttention,
#                                    TropicalTransformerBlock
#   mixture-of-tropical-experts   -> TropicalRouter, MoTEBlock
#
# All tropical math is copied unchanged; only imports are rewired to
# tikitorch.functional / the nn primitives defined above. Every
# operation is +/max/min (comparisons + additions) — no multiplications in the
# tropical path.


# ---------------------------------------------------------------------------
# Normalization (from morphological.py)
# ---------------------------------------------------------------------------


class TropicalBatchShift(nn.Module):
    """Tropical-native normalization: subtract batch mean per feature.

    Centers activations around zero without multiplications.
    Only uses subtractions — FPGA-friendly.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)


# ---------------------------------------------------------------------------
# Depthwise Morphological Convolution (sequence mixing, from morphological.py)
# ---------------------------------------------------------------------------


class DepthwiseMorphConv1d(nn.Module):
    """Morphological 1D convolution with configurable channel grouping.

    Dilation: y[c,t] = max_k(h[c,k] + x[c, t - k*dilation])
    Erosion:  y[c,t] = min_k(h[c,k] + x[c, t - k*dilation])

    With group_size > 1, the max/min operates across groups of channels,
    enabling cross-channel feature interaction while staying in the tropical
    semiring (still only comparisons + additions).

    group_size=1: pure depthwise (original behavior)
    group_size=G: max/min over G channels × K kernel positions per output
    group_size=channels: full (non-depthwise) morphological conv

    Causal (left-padded): output at time t depends only on t and earlier.
    Hardware cost: K×G comparisons + K×G additions per output element.
    Zero multiplications.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1,
                 mode: str = "dilation", use_soft: bool = True, tau: float = 1.0,
                 group_size: int = 1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.mode = mode
        self.use_soft = use_soft
        self.tau = tau
        self.group_size = group_size

        if group_size == 1:
            # Pure depthwise: one kernel per channel
            self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        else:
            # Grouped: one kernel per group, applied to group_size channels
            assert channels % group_size == 0, f"channels ({channels}) must be divisible by group_size ({group_size})"
            n_groups = channels // group_size
            # Weight shape: (n_groups, group_size * kernel_size)
            # The max/min operates over group_size * kernel_size positions
            self.weight = nn.Parameter(torch.empty(n_groups, group_size * kernel_size))

        self.bias = nn.Parameter(torch.zeros(channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        K = self.kernel_size
        d = self.dilation
        pad = (K - 1) * d

        x_pad = F.pad(x, (pad, 0))

        # Gather dilated windows — vectorized index gather (no Python loop)
        # Build index tensor: for each output position t and kernel position k,
        # the input position is t + (K-1-k)*d in the padded tensor
        indices = torch.arange(L, device=x.device).unsqueeze(1) + \
                  torch.arange(K - 1, -1, -1, device=x.device).unsqueeze(0) * d  # (L, K)
        windows = x_pad[:, :, indices]  # (B, C, L, K) via advanced indexing

        if self.group_size == 1:
            # Pure depthwise (original fast path)
            combined = windows + self.weight.unsqueeze(0).unsqueeze(2)  # (B, C, L, K)

            if self.mode == "dilation":
                if self.use_soft:
                    y = tropical_softmax(combined, tau=self.tau, dim=-1)
                else:
                    y = combined.max(dim=-1).values
            else:
                if self.use_soft:
                    y = tropical_softmin(combined, tau=self.tau, dim=-1)
                else:
                    y = combined.min(dim=-1).values
        else:
            # Grouped: reshape channels into groups, then max/min across group × kernel
            G = self.group_size
            n_groups = C // G

            # Reshape: (B, C, L, K) -> (B, n_groups, G, L, K) -> (B, n_groups, L, G*K)
            windows_grouped = windows.reshape(B, n_groups, G, L, K)
            windows_grouped = windows_grouped.permute(0, 1, 3, 2, 4)  # (B, n_groups, L, G, K)
            windows_grouped = windows_grouped.reshape(B, n_groups, L, G * K)

            # Add structuring element: weight is (n_groups, G*K)
            combined = windows_grouped + self.weight.unsqueeze(0).unsqueeze(2)

            if self.mode == "dilation":
                if self.use_soft:
                    y = tropical_softmax(combined, tau=self.tau, dim=-1)
                else:
                    y = combined.max(dim=-1).values
            else:
                if self.use_soft:
                    y = tropical_softmin(combined, tau=self.tau, dim=-1)
                else:
                    y = combined.min(dim=-1).values

            # y shape: (B, n_groups, L) — need to expand back to (B, C, L)
            # Each group produces 1 output per position; replicate to G channels
            y = y.unsqueeze(2).expand(-1, -1, G, -1).reshape(B, C, L)

        return y + self.bias.unsqueeze(0).unsqueeze(-1)

    @torch.no_grad()
    def forward_with_routing(self, x: torch.Tensor):
        """Forward returning the FROZEN routing for a PC cascade sweep.

        Hard (``use_soft=False`` / tau==0): ``(y, idx)`` with ``idx`` (B, C, L) the
        winning kernel slot j* per output — slot j reads input position t − j·d, so
        the adjoint is a pure scatter to (t − j*·d) and to weight[c, j*] (zero
        multiplies; error to left-pad positions is dropped, matching ``F.pad``'s
        backward). Soft (tau>0): ``(y, p)`` with ``p`` (B, C, L, K) the softmax/
        softmin routing weights — the exact Jacobian of the tau-relaxed op (small:
        K per output, unlike the d×d projection). Depthwise (``group_size==1``)
        only. Matches ``forward`` bit-exactly.
        """
        assert self.group_size == 1, "forward_with_routing: depthwise only (v1)"
        B, C, L = x.shape
        K, d = self.kernel_size, self.dilation
        x_pad = F.pad(x, ((K - 1) * d, 0))
        indices = torch.arange(L, device=x.device).unsqueeze(1) + \
                  torch.arange(K - 1, -1, -1, device=x.device).unsqueeze(0) * d
        windows = x_pad[:, :, indices]                        # (B, C, L, K)
        combined = windows + self.weight.unsqueeze(0).unsqueeze(2)
        soft = self.use_soft and self.tau > 0
        if self.mode == "dilation":
            if soft:
                y = tropical_softmax(combined, tau=self.tau, dim=-1)
                route = torch.softmax(combined / self.tau, dim=-1)
            else:
                y, route = combined.max(dim=-1)
        else:
            if soft:
                y = tropical_softmin(combined, tau=self.tau, dim=-1)
                route = torch.softmax(-combined / self.tau, dim=-1)
            else:
                y, route = combined.min(dim=-1)
        return y + self.bias.unsqueeze(0).unsqueeze(-1), route

    def step(self, x_t: torch.Tensor, buffer: torch.Tensor):
        """Process single timestep incrementally.

        Args:
            x_t: (B, C) -- single timestep input
            buffer: (B, C, buf_len) -- sliding window buffer, buf_len = (K-1)*dilation + 1
        Returns:
            y_t: (B, C) -- single timestep output
            buffer: (B, C, buf_len) -- updated buffer
        """
        # Shift buffer left by 1, insert x_t at the end
        buffer = torch.cat([buffer[:, :, 1:], x_t.unsqueeze(-1)], dim=-1)

        # Extract the K dilated positions from buffer
        # The buffer holds the last (K-1)*d + 1 values
        # Dilated positions: buffer[:, :, -1], buffer[:, :, -1-d], buffer[:, :, -1-2d], ...
        K = self.kernel_size
        d = self.dilation

        if self.group_size == 1:
            # Pure depthwise path
            positions = []
            for k in range(K):
                idx = buffer.shape[-1] - 1 - k * d
                positions.append(buffer[:, :, idx])  # (B, C)
            windows = torch.stack(positions, dim=-1)  # (B, C, K)
            combined = windows + self.weight.unsqueeze(0)  # (B, C, K)

            if self.mode == "dilation":
                if self.use_soft:
                    y_t = tropical_softmax(combined, tau=self.tau, dim=-1)
                else:
                    y_t = combined.max(dim=-1).values
            else:
                if self.use_soft:
                    y_t = tropical_softmin(combined, tau=self.tau, dim=-1)
                else:
                    y_t = combined.min(dim=-1).values
        else:
            # For grouped: follow the same logic as full forward but for single step
            G = self.group_size
            C = self.channels
            n_groups = C // G
            B = x_t.shape[0]

            positions = []
            for k in range(K):
                idx = buffer.shape[-1] - 1 - k * d
                positions.append(buffer[:, :, idx])
            windows = torch.stack(positions, dim=-1)  # (B, C, K)

            windows_grouped = windows.reshape(B, n_groups, G, K)
            windows_grouped = windows_grouped.reshape(B, n_groups, G * K)
            combined = windows_grouped + self.weight.unsqueeze(0)

            if self.mode == "dilation":
                if self.use_soft:
                    y_t = tropical_softmax(combined, tau=self.tau, dim=-1)
                else:
                    y_t = combined.max(dim=-1).values
            else:
                if self.use_soft:
                    y_t = tropical_softmin(combined, tau=self.tau, dim=-1)
                else:
                    y_t = combined.min(dim=-1).values

            y_t = y_t.unsqueeze(2).expand(-1, -1, G).reshape(B, C)

        y_t = y_t + self.bias
        return y_t, buffer

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"dilation={self.dilation}, mode={self.mode}, "
            f"use_soft={self.use_soft}, tau={self.tau}, group_size={self.group_size}"
        )


# ---------------------------------------------------------------------------
# Multi-scale morphological convolution (from morphological.py)
# ---------------------------------------------------------------------------


class MultiScaleMorphConv1d(nn.Module):
    """Multi-scale morphological convolution: parallel morph convs at different kernel sizes.

    Runs K=3, K=7, K=15 (or configurable) in parallel, then combines via
    tropical max (dilation mode) or tropical min (erosion mode).
    Still only comparisons + additions — preserves algebraic closure.

    The multi-scale design captures both fine and coarse temporal features
    per layer, improving effective receptive field without more depth.
    """

    def __init__(self, channels: int, kernel_sizes=(3, 7, 15),
                 dilation: int = 1, mode: str = "dilation",
                 use_soft: bool = True, tau: float = 1.0):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.mode = mode
        self.use_soft = use_soft
        self.tau = tau
        # One morph conv per scale
        self.convs = nn.ModuleList([
            DepthwiseMorphConv1d(channels, k, dilation=dilation,
                                 mode=mode, use_soft=use_soft, tau=tau)
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) -> (B, C, L). Combine scales via tropical max/min."""
        if not self.use_soft:
            # Memory-efficient path: running max/min, never stores all scales at once
            result = self.convs[0](x)
            for conv in self.convs[1:]:
                out = conv(x)
                if self.mode == "dilation":
                    result = torch.max(result, out)
                else:
                    result = torch.min(result, out)
            return result
        else:
            # Soft path needs all scales for logsumexp
            outputs = [conv(x) for conv in self.convs]
            stacked = torch.stack(outputs, dim=-1)  # (B, C, L, n_scales)
            if self.mode == "dilation":
                return tropical_softmax(stacked, tau=self.tau, dim=-1)
            else:
                return tropical_softmin(stacked, tau=self.tau, dim=-1)


# ---------------------------------------------------------------------------
# Morphological SSM Layer (from morphological.py)
# ---------------------------------------------------------------------------


class MorphSSMLayer(nn.Module):
    """Single morphological SSM layer.

    Sequence mixing via depthwise morphological convolution (dilation or
    erosion, alternating by layer index).  Channel mixing via standard
    linear projection.

    The morphological operation itself provides nonlinearity (max/min are
    nonlinear), so no explicit activation is needed.

    Architecture (sandwich norm residual):
        x -> LayerNorm -> morph_conv -> dropout -> linear_proj -> + residual -> LayerNorm

    The post-residual LayerNorm is critical: without it, activations grow
    unboundedly during training because tropical_softmax/softmin outputs
    accumulate through residual connections without any normalizing
    mechanism (unlike attention softmax which self-normalizes).  This
    causes the model to collapse to random-chance accuracy over ~3-5K
    steps.
    """

    def __init__(self, d_model: int, kernel_size: int = 7, dilation: int = 1,
                 mode: str = "dilation", dropout: float = 0.1,
                 use_soft: bool = True, tau: float = 1.0,
                 tropical_proj: bool = False, norm_type: str = "tropical",
                 group_size: int = 1, residual_scale: float = 1.0,
                 multi_scale: bool = False, kernel_sizes=(3, 7, 15),
                 out_proj_type=None):
        super().__init__()
        self.residual_scale = residual_scale
        self.norm = _make_norm(norm_type, d_model)
        if multi_scale:
            self.morph_conv = MultiScaleMorphConv1d(
                d_model, kernel_sizes=kernel_sizes, dilation=dilation,
                mode=mode, use_soft=use_soft, tau=tau,
            )
        else:
            self.morph_conv = DepthwiseMorphConv1d(
                d_model, kernel_size, dilation=dilation,
                mode=mode, use_soft=use_soft, tau=tau, group_size=group_size,
            )
        self.dropout = TropicalDropout(dropout) if dropout > 0 else nn.Identity()
        # out_proj_type:
        #   "tropical" - TropicalDilationModule (max-plus + bias; closure-preserving, DEFAULT)
        #   "linear"   - nn.Linear (NOT closure-preserving; opt-in ablation only — closure hole H5)
        #   "identity" - no projection (closure-rigorous; relies on subsequent
        #                blocks' FFN/expert layers to do channel mixing)
        # Default is now the multiplier-free path (was "linear" — every MorphSSM/
        # MoTE block silently multiplied D^2 MACs/token, the H5 closure hole).
        if out_proj_type is None:
            out_proj_type = "tropical"
        if out_proj_type == "tropical":
            # Intra-package reference.
            self.out_proj = TropicalDilationModule(d_model, d_model, use_softmax=False, tau=tau)
        elif out_proj_type == "identity":
            self.out_proj = nn.Identity()
        else:  # "linear"
            self.out_proj = nn.Linear(d_model, d_model)
        self.out_proj_type = out_proj_type
        self.post_norm = _make_norm(norm_type, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Transpose for conv: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        x = self.morph_conv(x)
        x = x.transpose(1, 2)  # back to (B, L, D)

        x = self.dropout(x)
        if self.out_proj_type == "tropical":
            B, L, D = x.shape
            x = self.out_proj(x.reshape(B * L, D)).reshape(B, L, D)
        else:  # nn.Linear or nn.Identity both accept (B, L, D) directly
            x = self.out_proj(x)

        # residual_scale == 1.0 -> no multiply (closure: skip the constant scale).
        if self.residual_scale == 1.0:
            return self.post_norm(residual + x)
        return self.post_norm(residual + x * self.residual_scale)

    def step(self, x_t: torch.Tensor, buffer: torch.Tensor):
        """Single-timestep inference with residual and norms.

        Args:
            x_t: (B, D) -- single timestep
            buffer: (B, D, buf_len) -- conv buffer
        Returns:
            y_t: (B, D), updated buffer
        """
        residual = x_t
        # Norm across d_model (works for single timestep)
        h = self.norm(x_t.unsqueeze(1)).squeeze(1)  # (B, 1, D) -> norm -> (B, D)

        # Morph conv step
        h, buffer = self.morph_conv.step(h, buffer)

        h = self.dropout(h)

        # Output projection (both branches were identical; collapsed).
        h = self.out_proj(h)

        # Post-norm with residual. residual_scale == 1.0 -> no multiply (closure),
        # matching forward(); only scale when explicitly != 1.0.
        if self.residual_scale != 1.0:
            h = h * self.residual_scale
        out = self.post_norm((residual + h).unsqueeze(1)).squeeze(1)
        return out, buffer


# ---------------------------------------------------------------------------
# Tropical multi-head attention (from tropical_transformer.py)
# ---------------------------------------------------------------------------


class TropicalAttention(nn.Module):
    """Tropical multi-head attention with non-separable Q-K scoring.

    For each head h, query position i, output dim d:
        score_{i,j}  = max_d' (Q_{i,d'}^h + K_{j,d'}^h)
        out_{i,d}^h  = max_j  (V_{j,d}^h + score_{i,j})

    The scoring is a tropical (max-plus) inner product — depends on both Q and
    K. Each output dimension d picks its own argmax over keys j, so different
    output coordinates can attend to different tokens.

    Soft / hard switch via tau:
        tau == 0  : hard max (custom autograd, sparse gradients, FPGA-mappable).
        tau > 0   : soft logsumexp (dense gradients, training only).

    Use set_tau(...) for annealing during training.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 tau: float = 0.0, use_standard_proj: bool = False,
                 score_variant: str = "dot_product",
                 proj_variant: str = "dilation",
                 proj_d_intermediate=None,
                 proj_topk_k: int = 4,
                 value_topk_k: int = 1,
                 proj_init_identity: bool = False,
                 agg_variant: str = "argmax",
                 agg_topk_k: int = 2,
                 n_hops: int = 1,
                 sink_eps: float = 1.0,
                 sink_iters: int = 5):
        super().__init__()
        assert d_model % n_heads == 0
        assert score_variant in _SCORE_VARIANTS, score_variant
        assert proj_variant in ("dilation", "compositional", "topk"), proj_variant
        assert agg_variant in _AGG_VARIANTS, agg_variant
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.tau = tau
        # value_topk_k > 1 softens the HARD (tau==0) value pull: each output
        # (i, d) averages its k best keys instead of its single argmax, so the
        # gradient flows to k keys -- denser, still add/compare-only (no
        # multiply). An ablation knob; whether it helps is task-dependent.
        self.value_topk_k = value_topk_k
        # score_variant:
        #   "dot_product" — max-plus inner product max_d(Q + K) (variant b).
        #   "l_inf"       — negative L_inf distance -max_d|Q - K| (variant c).
        self.score_variant = score_variant
        # proj_variant:
        #   "dilation"      — single TropicalDilationModule (1-of-d, baseline).
        #   "compositional" — Dilation -> Erosion (tropical rational, Fix A).
        #   "topk"          — TropicalTopKModule (K-of-d aggregation, Fix B).
        self.proj_variant = proj_variant
        self.proj_d_intermediate = proj_d_intermediate
        self.proj_topk_k = proj_topk_k
        # Seed dilation projections as identity (fixes the value-collapse where a
        # uniform-init single dilation loses per-dim diversity; see
        # TropicalDilationModule.init_identity). Off by default (preserves prior
        # behavior); the algorithmic reasoner turns it on.
        self.proj_init_identity = proj_init_identity
        # Value-aggregation variant: "argmax" (per-dim single argmax, default,
        # preserves prior results) or "iter_topk" (iterative-argmax winner-
        # exclusion top-k, SUM-combined -- strictly multiplier-free, denser grad).
        self.agg_variant = agg_variant
        self.agg_topk_k = agg_topk_k
        # n_hops>1: Bellman-Ford diffusion — iterate the value state through the
        # content score graph S for n_hops rounds so information routes along best
        # PATHS of length <= n_hops between every token pair (tropical matrix
        # powers / shortest-path closure). Purely max-plus, data-dependent,
        # all-pairs; fills the conv mixer's fixed-receptive-field gap. n_hops=1 is
        # the historical single-hop attention (bit-exact).
        self.n_hops = n_hops
        # Sinkhorn/assignment routing (agg_variant="sinkhorn"): entropic temperature
        # + #normalization iterations. train soft transport -> deploy hard permutation.
        self.sink_eps = sink_eps
        self.sink_iters = sink_iters
        # Closed learnable per-rank additive offset for iter_topk (b_0 := 0).
        self.rank_bias = (nn.Parameter(torch.zeros(agg_topk_k))
                          if agg_variant == "iter_topk" else None)

        if use_standard_proj:
            # Mechanism-isolation test mode (closure broken by multiplications).
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            self.q_proj = self._make_proj(d_model, d_model)
            self.k_proj = self._make_proj(d_model, d_model)
            self.v_proj = self._make_proj(d_model, d_model)
            self.out_proj = self._make_proj(d_model, d_model)
        self.use_standard_proj = use_standard_proj

    def _make_proj(self, in_dim: int, out_dim: int) -> nn.Module:
        """Construct a tropical projection per self.proj_variant."""
        if self.proj_variant == "dilation":
            return TropicalDilationModule(in_dim, out_dim, use_softmax=False,
                                          init_identity=self.proj_init_identity)
        if self.proj_variant == "compositional":
            return CompositionalTropicalProjection(
                in_dim, out_dim,
                d_intermediate=self.proj_d_intermediate,
                tau=self.tau,
            )
        if self.proj_variant == "topk":
            return TropicalTopKModule(in_dim, out_dim, k=self.proj_topk_k,
                                      tau=self.tau)
        raise ValueError(self.proj_variant)

    def set_tau(self, tau: float) -> None:
        """Anneal soft <-> hard. tau=0 is hard tropical (closure-preserving).

        Propagates tau into projection sub-modules that support it (compositional).
        """
        self.tau = tau
        # Cascade tau into sub-projections that respect it.
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if hasattr(proj, "set_tau"):
                proj.set_tau(tau)

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # Project. Standard nn.Linear and Compositional/TopK projections accept
        # N-D input directly. Legacy TropicalDilationModule expects 2-D, so
        # reshape only in that case.
        if self.proj_variant == "dilation" and not self.use_standard_proj:
            q = self.q_proj(x.reshape(B * L, D)).reshape(B, L, H, Dh).transpose(1, 2)
            k = self.k_proj(x.reshape(B * L, D)).reshape(B, L, H, Dh).transpose(1, 2)
            v = self.v_proj(x.reshape(B * L, D)).reshape(B, L, H, Dh).transpose(1, 2)
        else:
            q = self.q_proj(x).reshape(B, L, H, Dh).transpose(1, 2)
            k = self.k_proj(x).reshape(B, L, H, Dh).transpose(1, 2)
            v = self.v_proj(x).reshape(B, L, H, Dh).transpose(1, 2)
        # q, k, v: (B, H, L, Dh)

        scores = _tropical_score(q, k, self.score_variant, self.tau)
        # scores: (B, H, L, L) — non-separable, depends on both Q and K. Causal
        # masking is applied INSIDE _tropical_aggregate so it is semiring-correct
        # (max-plus masks with -inf; min-plus must mask with +inf, else min picks
        # the masked key).
        # Value routing. Single-hop (n_hops<=1) = the historical 1-hop pull. Multi-
        # hop = Bellman-Ford diffusion: H^(t+1) = max(H^(t), max_j(S[i,j]+H^(t)[j,d]))
        # over a FIXED content graph S, so token i receives the best path of length
        # <= n_hops from any j (purely max-plus all-pairs routing). The max(H,.)
        # self-loop is an always-alive residual gradient path; S is shared across
        # hops so its gradient accumulates over every relaxation.
        h = v
        for _hop in range(max(1, self.n_hops)):
            pull = _tropical_aggregate(h, scores, self.agg_variant, self.tau,
                                       self.agg_topk_k, self.rank_bias, self.value_topk_k,
                                       causal=causal_mask,
                                       sink_eps=self.sink_eps, sink_iters=self.sink_iters)
            h = pull if self.n_hops <= 1 else torch.maximum(h, pull)
        out = h
        # out: (B, H, L, Dh) — each (i, d) picks its own j* (or its k best).

        out = out.transpose(1, 2).reshape(B, L, D)
        if self.proj_variant == "dilation" and not self.use_standard_proj:
            out = self.out_proj(out.reshape(B * L, D)).reshape(B, L, D)
        else:
            out = self.out_proj(out)
        return out


class TropicalShiftAttention(nn.Module):
    """Tropical attention with **no projections** — additive position shifts.

    Motivation. In the tropical semiring, classical multiplication becomes
    standard addition. So a "rotation" of the query vector — which in the
    real semiring is ``q -> R q`` (a multiplicative isometry) — becomes a
    *translation* in tropical algebra: ``q -> q + p``. This is the additive
    analogue of RoPE's rotary embeddings.

    Mathematical structure. The only tropical-isometric linear maps are
    generalized permutation matrices (one finite entry per row/column);
    everything else is non-isometric and either contracts or expands.
    Free ``nn.Linear`` projections in standard attention have no tropical
    counterpart that matches their expressivity *while preserving closure*.
    Consequence: the right "tropical projection" isn't a learned matrix at
    all — it's a learned *shift*. Q and K diverge from the same input via
    independent per-position shift streams; V passes through unchanged.

    Mechanism (variant b — max-plus dot product, applied per head):
        Q[m, d] = x[m, d] + s_q[m, d]      # additive Q-shift at position m
        K[n, d] = x[n, d] + s_k[n, d]      # additive K-shift at position n
        V[n, d] = x[n, d]                   # value passes through
        score[m, n] = max_d (Q[m, d] + K[n, d])
        out[m, d] = max_n (V[n, d] + score[m, n])

    With opposite signs (s_q[m, d] = +m·g[d], s_k[m, d] = -m·g[d]) the
    score is sensitive to *relative* position (m - n)·g[d] — the additive
    analogue of RoPE's relative-rotation property, and the key to **length
    extrapolation** (the score depends on the offset m-n, not absolute
    positions, so it is well-defined for L beyond ``max_seq_len``). With
    independent ``nn.Embedding``-parameterized shifts (``shift_mode='free'``),
    the model learns whatever shift pattern best fits the task.

    Closure. Every operation is +/max/min — no multiplications, no projection
    matrices, no norms inside the attention. The ``rope_relative`` ramp
    ``p·g[d]`` is built by *cumulative addition* of ``g`` (not a multiply), so
    even that path stays strictly multiplier-free / FPGA-mappable. Pure tropical.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 tau: float = 0.0, score_variant: str = "dot_product",
                 shift_mode: str = "free",
                 agg_variant: str = "argmax", agg_topk_k: int = 2):
        super().__init__()
        assert d_model % n_heads == 0
        assert score_variant in _SCORE_VARIANTS, score_variant
        assert shift_mode in ("free", "rope_relative"), shift_mode
        assert agg_variant in _AGG_VARIANTS, agg_variant
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.tau = tau
        self.score_variant = score_variant
        self.shift_mode = shift_mode
        self.agg_variant = agg_variant
        self.agg_topk_k = agg_topk_k
        self.rank_bias = (nn.Parameter(torch.zeros(agg_topk_k))
                          if agg_variant == "iter_topk" else None)

        if shift_mode == "free":
            # Independent learnable shift per (position, dim) for Q and K.
            self.shift_q = nn.Embedding(max_seq_len, d_model)
            self.shift_k = nn.Embedding(max_seq_len, d_model)
            nn.init.zeros_(self.shift_q.weight)
            nn.init.zeros_(self.shift_k.weight)
        else:  # "rope_relative"
            # Single learnable g[d] vector. Shifts are m·g for Q, -m·g for K
            # so the score depends only on (m - n)·g[d].
            self.g = nn.Parameter(torch.zeros(d_model))

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def _make_shifts(self, L: int, device):
        positions = torch.arange(L, device=device)
        if self.shift_mode == "free":
            return self.shift_q(positions), self.shift_k(positions)
        # rope_relative: shift[p] = p * g[d], the relative-position ramp. Built by
        # CUMULATIVE ADDITION of g (repeated add), NOT a multiply -- so the path
        # stays closure-clean / FPGA-mappable while keeping g trainable and the
        # score dependent only on (m - n)*g. (A naive `m * g` is a real multiply.)
        g = self.g.to(device)
        ramp = torch.cumsum(g.unsqueeze(0).expand(L, -1), dim=0)   # [1g, 2g, ..., Lg]
        shift = torch.cat([torch.zeros(1, g.shape[0], device=device, dtype=g.dtype),
                           ramp[:-1]], dim=0)                      # [0, 1g, ..., (L-1)g]
        return shift, -shift

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        s_q, s_k = self._make_shifts(L, x.device)  # each (L, D)
        q = (x + s_q.unsqueeze(0)).reshape(B, L, H, Dh).transpose(1, 2)
        k = (x + s_k.unsqueeze(0)).reshape(B, L, H, Dh).transpose(1, 2)
        v = x.reshape(B, L, H, Dh).transpose(1, 2)
        # q, k, v: (B, H, L, Dh)

        scores = _tropical_score(q, k, self.score_variant, self.tau)

        out = _tropical_aggregate(v, scores, self.agg_variant, self.tau,
                                  self.agg_topk_k, self.rank_bias, value_topk_k=1,
                                  causal=causal_mask)
        # No out-projection — the head/output stage in the surrounding model
        # consumes the attention output directly.
        return out.transpose(1, 2).reshape(B, L, D)


class TropicalTransformerBlock(nn.Module):
    """Tropical transformer block: norm -> attention -> residual -> norm -> tropical FFN -> residual.

    use_shift_attn=True swaps the projected ``TropicalAttention`` for the
    closure-preserving ``TropicalShiftAttention`` (no Q/K/V projections).
    Requires max_seq_len for the shift embedding tables.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 tau: float = 0.0, use_shift_attn: bool = False,
                 max_seq_len=None,
                 score_variant: str = "dot_product",
                 shift_mode: str = "free", norm_type: str = "tropical",
                 proj_init_identity: bool = False,
                 agg_variant: str = "argmax", agg_topk_k: int = 2):
        super().__init__()
        self.norm1 = _make_norm(norm_type, d_model)
        if use_shift_attn:
            assert max_seq_len is not None, "use_shift_attn=True requires max_seq_len"
            self.attn = TropicalShiftAttention(
                d_model, n_heads, max_seq_len=max_seq_len,
                tau=tau, score_variant=score_variant,
                shift_mode=shift_mode,
                agg_variant=agg_variant, agg_topk_k=agg_topk_k,
            )
        else:
            self.attn = TropicalAttention(d_model, n_heads, dropout=dropout, tau=tau,
                                          score_variant=score_variant,
                                          proj_init_identity=proj_init_identity,
                                          agg_variant=agg_variant, agg_topk_k=agg_topk_k)
        self.norm2 = _make_norm(norm_type, d_model)

        # Tropical FFN: Dilation (max-plus) -> Erosion (min-plus) = tropical rational
        self.ffn_up = TropicalDilationModule(d_model, d_ff, use_softmax=False)
        self.ffn_down = TropicalLinearModule(d_ff, d_model, use_softmin=False)
        self.dropout = TropicalDropout(dropout) if dropout > 0 else nn.Identity()

    def set_tau(self, tau: float) -> None:
        self.attn.set_tau(tau)
        # Thread tau into the FFN too -- otherwise the tropical FFN stays hard
        # through the whole soft->hard curriculum (soft-anneal freeze).
        self.ffn_up.set_tau(tau)
        self.ffn_down.set_tau(tau)

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        B, L, D = x.shape

        # Tropical attention with pre-norm
        h = self.norm1(x)
        attn_out = self.attn(h, causal_mask=causal_mask)
        x = x + self.dropout(attn_out)

        # Tropical FFN with pre-norm
        h = self.norm2(x)
        # Reshape for 2D tropical ops
        h_flat = h.reshape(B * L, D)
        ffn_out = self.ffn_up(h_flat)    # dilation (max-plus) — convex piecewise-affine
        ffn_out = self.ffn_down(ffn_out)  # erosion (min-plus) — makes it tropical rational
        ffn_out = ffn_out.reshape(B, L, D)
        x = x + self.dropout(ffn_out)

        return x


# ---------------------------------------------------------------------------
# Mixture of Tropical Experts: router + block (from mote.py)
# ---------------------------------------------------------------------------


class TropicalRouter(nn.Module):
    """Routes tokens to experts using tropical argmax patterns.

    A small TropicalDilationModule projects tokens to a low-dim routing space.
    The argmax indices from the max-plus operation form a discrete routing
    signature. Hashing these indices (via addition + modulo) produces expert IDs.

    Operations: additions + comparisons + modulo — algebraically closed
    (modulo is a bitwise mask for power-of-2 n_experts on FPGA).
    """

    def __init__(self, d_model: int, d_router: int, n_experts: int):
        super().__init__()
        self.d_model = d_model
        self.d_router = d_router
        self.n_experts = n_experts
        self.proj = TropicalDilationModule(d_model, d_router, use_softmax=False)

    def forward(self, x: torch.Tensor):
        """Route tokens to experts.

        Args:
            x: (total_tokens, d_model)
        Returns:
            expert_ids: (total_tokens,) — integer in [0, n_experts)
            aux_loss: scalar — load-balancing loss
        """
        # Get argmax indices from tropical dilation
        _, indices = self.proj.forward_with_indices(x)  # (total_tokens, d_router)

        # Hash indices to expert ID: sum of indices + positional offset mod n_experts
        # Positional offset breaks hash collisions from interleaved RVQ tokens
        # This is additions + modulo — algebraically closed
        pos_offset = torch.arange(x.shape[0], device=x.device) % self.n_experts
        expert_ids = (indices.sum(dim=-1) + pos_offset) % self.n_experts  # (total_tokens,)

        # Load-balancing auxiliary loss -- a TRAINING-only MoE regularizer (and its
        # squared-fraction term is the one non-closed op in MoTE). Skip it at
        # inference so the deployed forward is multiplier-free by construction.
        if self.training:
            aux_loss = self._load_balance_loss(expert_ids, x.shape[0])
        else:
            aux_loss = expert_ids.new_zeros((), dtype=torch.float32)

        return expert_ids, aux_loss

    def _load_balance_loss(self, expert_ids: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Compute load-balancing loss to encourage uniform expert utilization.

        Standard MoE L2 balance loss: n_experts * sum(frac_e^2).
        Minimum (uniform) = 1.0; maximum (all to one expert) = n_experts.
        """
        # Count tokens per expert
        counts = torch.zeros(self.n_experts, device=expert_ids.device)
        counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=counts.dtype))
        fractions = counts / max(n_tokens, 1)
        return self.n_experts * (fractions ** 2).sum()


class MoTEBlock(nn.Module):
    """Dispatch tokens to experts and gather results.

    Each token goes to exactly one expert (hard routing).
    No weighted combination — pure indexing, zero multiplications.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        expert_type: str = "transformer",
        expert_layers: int = 1,
        d_ff: int = 512,
        n_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        d_router: int = 32,
        use_soft: bool = True,
        tau: float = 1.0,
        capacity_factor: float = 1.5,
        use_shift_attn: bool = False,
        max_seq_len=None,
        score_variant: str = "dot_product",
    ):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        self.router = TropicalRouter(d_model, d_router, n_experts)

        # Build experts
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            if expert_type == "transformer":
                # Stack of TropicalTransformerBlocks. With use_shift_attn=True
                # the experts are fully closure-preserving (no Q/K/V projections).
                layers = nn.ModuleList([
                    TropicalTransformerBlock(
                        d_model, n_heads, d_ff, dropout,
                        use_shift_attn=use_shift_attn,
                        max_seq_len=max_seq_len,
                        score_variant=score_variant,
                    )
                    for _ in range(expert_layers)
                ])
                self.experts.append(layers)
            elif expert_type == "morph":
                # Stack of MorphSSMLayer
                layers = nn.ModuleList([
                    MorphSSMLayer(
                        d_model, kernel_size, dilation=1,
                        mode="dilation" if j % 2 == 0 else "erosion",
                        dropout=dropout, use_soft=use_soft, tau=tau,
                    )
                    for j in range(expert_layers)
                ])
                self.experts.append(layers)
            else:
                raise ValueError(f"Unknown expert_type: {expert_type}")

        self.expert_type = expert_type

    def forward(self, x: torch.Tensor):
        """Dispatch tokens to experts and gather results.

        For morph experts, tokens routed to the same expert are processed
        as a contiguous sequence (not individually), so the morphological
        convolutions have temporal context within each expert.

        Args:
            x: (B, L, D) — input tensor
        Returns:
            output: (B, L, D) — expert-processed output
            aux_loss: scalar — load-balancing loss
        """
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)

        # Route
        expert_ids, aux_loss = self.router(x_flat)

        # Dispatch and process. Initialize from the input so any token that is
        # not routed (or is dropped by the capacity cap) passes through as a
        # residual instead of becoming a zero vector -- otherwise small shapes
        # where max_tokens rounds to 0 would zero the whole block.
        output = x_flat.clone()

        for e_idx in range(self.n_experts):
            mask = expert_ids == e_idx
            if not mask.any():
                continue

            # Apply capacity factor: cap tokens per expert
            token_indices = mask.nonzero(as_tuple=True)[0]
            # Floor to >=1 so small shapes (capacity_factor*B*L/n_experts < 1)
            # do not silently drop every routed token.
            max_tokens = max(1, int(self.capacity_factor * B * L / self.n_experts))
            if len(token_indices) > max_tokens:
                token_indices = token_indices[:max_tokens]

            tokens_e = x_flat[token_indices]  # (n_e, D)
            if len(token_indices) == 0:
                continue

            # Process through expert layers as a SEQUENCE (not individual tokens)
            # This gives morph conv experts temporal context within each expert batch
            h = tokens_e.unsqueeze(0)  # (1, n_e, D) — single batch, n_e length sequence
            if self.expert_type == "transformer":
                for layer in self.experts[e_idx]:
                    h = layer(h, causal_mask=False)
            else:
                for layer in self.experts[e_idx]:
                    h = layer(h)
            tokens_out = h.squeeze(0)  # (n_e, D)

            output[token_indices] = tokens_out

        return output.reshape(B, L, D), aux_loss


# ---------------------------------------------------------------------------
# 2D morphological convolution + pooling + CNN block (vision)
# ---------------------------------------------------------------------------


class TropicalConv2d(nn.Module):
    """Morphological 2D convolution — the multiplier-free analogue of ``nn.Conv2d``.

    A learned structuring element is ADDED to each input patch and reduced by a
    single max (``mode="dilation"``) or min (``mode="erosion"``). Channel-mixing
    (full, not just depthwise): each output channel reduces over all
    ``Cin x Kh x Kw`` taps. Zero multiplies (see
    :class:`~tikitorch.functional.TropicalConv2dFn`).

    Args mirror ``nn.Conv2d``: ``in_channels, out_channels, kernel_size, stride,
    padding, dilation, bias``. Padding is zero-valued (standard, trainable).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 mode: str = "dilation", bias: bool = True,
                 use_soft: bool = False, tau: float = 0.0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert mode in ("dilation", "erosion"), mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mode = mode
        # Soft (logsumexp) reduction over taps for a dense-gradient warmup; anneal
        # tau -> 0 (set_tau) to recover the hard, multiplier-free morphological op
        # (Maslov dequantization), exactly like the rest of the library.
        self.use_soft = use_soft
        self.tau = tau
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.5, 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_tau(self, tau: float) -> None:
        self.tau = tau
        self.use_soft = tau > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_soft and self.tau > 0:
            B, Cin, H, W = x.shape
            Kh, Kw = self.kernel_size
            s, p, d = self.stride, self.padding, self.dilation
            Hout = (H + 2 * p - d * (Kh - 1) - 1) // s + 1
            Wout = (W + 2 * p - d * (Kw - 1) - 1) // s + 1
            patches = torch.nn.functional.unfold(x, (Kh, Kw), dilation=d, padding=p, stride=s)
            wflat = self.weight.reshape(self.out_channels, -1)
            combined = patches.unsqueeze(1) + wflat.unsqueeze(0).unsqueeze(-1)  # (B,Cout,P,L)
            if self.mode == "dilation":
                y = tropical_softmax(combined, tau=self.tau, dim=2)
            else:
                y = tropical_softmin(combined, tau=self.tau, dim=2)
            y = y.reshape(B, self.out_channels, Hout, Wout)
        else:
            y = tropical_conv2d(x, self.weight, self.stride, self.padding,
                                self.dilation, self.mode)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, mode={self.mode}, tau={self.tau}")


class TropicalMaxPool2d(nn.Module):
    """Max pooling — the canonical tropical (max-plus) pooling. Pure compare."""

    def __init__(self, kernel_size, stride=None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class TropicalMinPool2d(nn.Module):
    """Min pooling (min-plus dual): -max_pool2d(-x). Negation + compare."""

    def __init__(self, kernel_size, stride=None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-x, self.kernel_size, self.stride, self.padding)


class TropicalAdaptiveMaxPool2d(nn.Module):
    """Adaptive max pooling to a fixed output size. Pure compare."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_max_pool2d(x, self.output_size)


class TropicalGlobalMaxPool(nn.Module):
    """Global max pool over (H, W) -> (B, C). The CNN -> head reduction.

    Replaces a global-average-pool divide with a pure max (multiplier-free, and
    usually the better tropical inductive bias).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.amax(dim=(-2, -1))


# ---------------------------------------------------------------------------
# Multiplier-free normalization (replaces nn.LayerNorm's variance + scale + div)
# ---------------------------------------------------------------------------


def _pow2_normalize(c: torch.Tensor, dim: int, eps: float = 1e-5) -> torch.Tensor:
    """Scale ``c`` by a *power-of-two* chosen from its mean-abs (L1) along ``dim``.

    ``scale = mean(|c|)`` (sum + a constant divide-by-d, i.e. a fixed shift);
    ``shift = round(log2 scale)``; return ``c * 2^-shift`` -- an arithmetic shift
    on hardware (zero DSP), not a data-dependent float multiply. Mean-abs (rather
    than max-abs) normalizes the *average* magnitude to ~1, an RMS-like scale
    that keeps activations bounded across residual depth (max-abs only pins the
    single peak and under-normalizes the variance, which destabilized training).
    The shift is computed under ``no_grad`` and treated as a per-row constant
    (straight-through), so the backward sees the true scale and the forward stays
    multiplier-free.
    """
    with torch.no_grad():
        scale = c.abs().mean(dim=dim, keepdim=True).clamp_min(eps)
        # Clamp the shift so the norm can never amplify near-zero activations by a
        # runaway factor (the cause of late-training blow-up): bound to +-8 bits.
        shift = torch.round(torch.log2(scale)).clamp_(-8.0, 8.0)
        factor = torch.exp2(-shift)             # exact power of two in [2^-8, 2^8]
    return c * factor


class TropicalLayerNorm(nn.Module):
    """Multiplier-free LayerNorm: subtract-mean + power-of-2 (mean-abs) rescale.

    Replaces ``nn.LayerNorm`` (mean / variance / ``1/sqrt(var)`` / affine ``gamma``
    -- all multiplies/divides) with: subtract the feature mean (``/d`` folds to a
    constant shift since ``d`` is fixed), then rescale by the nearest power of two
    to the **mean-abs** (an RMS-like scale, a barrel-shift bounded to +-8 bits,
    straight-through). The only learnable affine is an additive ``beta`` bias
    (add, no multiply); ``gamma`` scaling is dropped (it is the multiply).

    .. note::
       **Stability cost (known limitation).** Without true
       variance scaling this norm is less stable than ``nn.LayerNorm`` in DEEP +
       annealed-to-hard models: shallow nets are fine (the closure-faithful-CE
       char-LM recovery holds), but a 2-layer MorphSSM on sMNIST trains to ~0.21
       vs ~0.32 with ``nn.LayerNorm`` and shows late-training noise. It is the
       default so models are closed out of the box; pass ``norm_type="layernorm"``
       for the stable (DSP-boundary) fallback when depth/stability matters.
    """

    def __init__(self, normalized_shape: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x - x.mean(dim=-1, keepdim=True)
        y = _pow2_normalize(c, dim=-1, eps=self.eps)
        if self.beta is not None:
            y = y + self.beta
        return y


class TropicalRMSNorm(nn.Module):
    """Multiplier-free RMSNorm: power-of-2 (mean-abs) rescale, no mean-subtract.

    The cheapest closed norm -- just ``x * 2^-round(log2 mean|x|)`` (a shift, with
    the exponent clamped to +-8) plus an optional additive bias. Uses mean-abs
    (L1), an RMS-like scale, via the shared :func:`_pow2_normalize`.
    """

    def __init__(self, normalized_shape: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _pow2_normalize(x, dim=-1, eps=self.eps)
        if self.beta is not None:
            y = y + self.beta
        return y


class TropicalLayerNorm2d(nn.Module):
    """Channel-wise :class:`TropicalLayerNorm` for images ``(B, C, H, W)``."""

    def __init__(self, num_channels: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(num_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x - x.mean(dim=1, keepdim=True)            # over channels
        y = _pow2_normalize(c, dim=1, eps=self.eps)
        if self.beta is not None:
            y = y + self.beta.view(1, -1, 1, 1)
        return y


class MorphConvBlock(nn.Module):
    """A fully multiplier-free CNN unit: conv -> tropical-norm -> activation -> pool.

    conv (``TropicalConv2d``) -> ``TropicalLayerNorm2d`` -> ``tropical_leaky_relu``
    -> optional ``TropicalMaxPool2d``. Every op is add/compare/shift only.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 mode: str = "dilation", pool: bool = True, leak_shift: int = 3,
                 use_soft: bool = False, tau: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv = TropicalConv2d(in_channels, out_channels, kernel_size,
                                   padding=pad, mode=mode, use_soft=use_soft, tau=tau)
        self.norm = TropicalLayerNorm2d(out_channels)
        self.leak_shift = leak_shift
        self.pool = TropicalMaxPool2d(2) if pool else None

    def set_tau(self, tau: float) -> None:
        self.conv.set_tau(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = tropical_leaky_relu(x, self.leak_shift)
        if self.pool is not None:
            x = self.pool(x)
        return x


# ---------------------------------------------------------------------------
# Quantized input embedding (closed embedding of continuous features)
# ---------------------------------------------------------------------------


class TropicalQuantizedEmbed(nn.Module):
    """Closed input embedding for continuous features: quantize -> gather -> sum.

    A tropical (add-only) linear projection of a low-dimensional *continuous*
    input is rank-degenerate -- it can shift a scalar but not scale it, so every
    output channel is the same value plus a bias, which stalls training. The
    multiplier-free fix is to **quantize** each feature into ``n_bins`` and look
    up a learned per-bin vector from an ``nn.Embedding`` (a pure gather, hence
    closed and ``assert_closed``-clean), summing across input features.

    Forward: ``x:(..., in_dim)`` in ``[lo, hi]`` -> ``(..., out_dim)``. The bin
    index ``floor((x-lo)/step)`` uses a constant ``step=(hi-lo)/n_bins`` (a fixed
    scale, not a data-dependent multiply). Features are offset into one shared
    table so each feature has its own ``n_bins`` rows.
    """

    def __init__(self, in_dim: int, out_dim: int, n_bins: int = 32,
                 lo: float = 0.0, hi: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        # nn.Linear-compatible aliases so model code that reshapes on
        # ``embed.in_features`` / ``out_features`` keeps working.
        self.in_features = in_dim
        self.out_features = out_dim
        self.n_bins = n_bins
        self.lo = lo
        self.step = (hi - lo) / n_bins
        self.embed = nn.Embedding(in_dim * n_bins, out_dim)
        self.register_buffer("offsets", torch.arange(in_dim) * n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.clamp(((x - self.lo) / self.step).long(), 0, self.n_bins - 1)
        idx = idx + self.offsets                       # per-feature bin offset
        return self.embed(idx).sum(dim=-2)             # gather + sum over features

    def set_tau(self, tau: float) -> None:             # no soft path; here for API parity
        pass


# ---------------------------------------------------------------------------
# Closed channel scaling + Max-Plus-Min layer (deep-morph training fixes)
# ---------------------------------------------------------------------------
#
# Theorem 3.2 of "Training Deep Morphological Networks as Universal
# Approximators" (arXiv:2505.09710): stacked pure max-plus layers collapse to a
# single morphological layer and have at most ~m nonzero-gradient parameters per
# layer, so deep morph nets barely train (the root of our closure penalty). Their
# fix is a learnable per-channel scale `alpha` between morph layers. We make it
# CLOSURE-CLEAN by restricting alpha to a power of two (a bit-shift, not a
# multiply), learned via a straight-through round.


class TropicalScale(nn.Module):
    """Learnable per-channel POWER-OF-2 scale: y = x * 2^round(s).

    A bit-shift on hardware (zero DSP), straight-through on the rounding so the
    learnable exponent ``s`` still receives gradient. Inserting this between
    morphological layers breaks the Theorem-3.2 gradient collapse while staying
    multiplier-free. Scales dimension ``dim`` (size ``channels``).
    """

    def __init__(self, channels: int, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.log2_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.log2_scale
        s_ste = s + (torch.round(s) - s).detach()        # forward=round(s), grad->s
        factor = torch.exp2(s_ste)                        # exact power of two
        if self.dim == -1 or self.dim == x.dim() - 1:
            return x * factor
        shape = [1] * x.dim()
        shape[self.dim] = factor.numel()
        return x * factor.view(shape)


class MaxPlusMin(nn.Module):
    """Max-Plus-Min layer (universal approximator; arXiv:2505.09710 Thm 3.4).

    Shared weights for dilation and erosion with separate biases, then a closed
    per-channel scale::

        dil = max_j(x_j + W[o,j]);  ero = min_j(x_j + W[o,j])
        y_o = scale_o * [ max(w0_o, dil) + min(m0_o, ero) ]

    All ops are add/compare/shift -- multiplier-free. ``use_soft``/``tau`` switch
    the max/min to the logsumexp relaxation for a dense-gradient warmup
    (``set_tau(0)`` hardens). N-D capable.
    """

    def __init__(self, in_features: int, out_features: int,
                 use_soft: bool = False, tau: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_soft = use_soft
        self.tau = tau
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.w0 = nn.Parameter(torch.zeros(out_features))   # dilation bias
        self.m0 = nn.Parameter(torch.zeros(out_features))   # erosion bias
        self.scale = TropicalScale(out_features)
        nn.init.uniform_(self.weight, -0.1, 0.1)

    def set_tau(self, tau: float) -> None:
        self.tau = tau
        self.use_soft = tau > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = x.unsqueeze(-2) + self.weight           # (..., out, in)
        if self.use_soft and self.tau > 0:
            dil = tropical_softmax(expanded, tau=self.tau, dim=-1)
            ero = tropical_softmin(expanded, tau=self.tau, dim=-1)
        else:
            dil = expanded.max(dim=-1).values
            ero = expanded.min(dim=-1).values
        y = torch.maximum(self.w0, dil) + torch.minimum(self.m0, ero)
        return self.scale(y)


class DilationErosionPerceptron(nn.Module):
    """Dilation-Erosion Perceptron (DEP): a power-of-2 convex combo of dil + ero.

    ``y_o = lam * max_j(x_j + Wd[o,j]) + (1-lam) * min_j(x_j + We[o,j]) + b_o``,
    separate weights for dilation/erosion. ``lam`` must be a power of two (default
    0.5 = a 1-bit shift) to stay multiplier-free. N-D capable; ``use_soft``/``tau``
    give the logsumexp warmup. (Charisopoulos & Maragos; the morphological DEP.)
    """

    def __init__(self, in_features: int, out_features: int, lam: float = 0.5,
                 use_soft: bool = False, tau: float = 0.0):
        super().__init__()
        # lam must be a power of two for closure (lam*x and (1-lam)*x are shifts).
        assert lam in (0.25, 0.5, 0.75) or (lam > 0 and (1.0 / lam).is_integer()), \
            "lam must be a power-of-two fraction (e.g. 0.5, 0.25) to stay closed"
        self.in_features = in_features
        self.out_features = out_features
        self.lam = lam
        self.use_soft = use_soft
        self.tau = tau
        self.Wd = nn.Parameter(torch.empty(out_features, in_features))
        self.We = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.uniform_(self.Wd, -0.1, 0.1)
        nn.init.uniform_(self.We, -0.1, 0.1)

    def set_tau(self, tau: float) -> None:
        self.tau = tau
        self.use_soft = tau > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ed = x.unsqueeze(-2) + self.Wd
        ee = x.unsqueeze(-2) + self.We
        if self.use_soft and self.tau > 0:
            dil = tropical_softmax(ed, tau=self.tau, dim=-1)
            ero = tropical_softmin(ee, tau=self.tau, dim=-1)
        else:
            dil = ed.max(dim=-1).values
            ero = ee.min(dim=-1).values
        return self.lam * dil + (1.0 - self.lam) * ero + self.bias


class MorphOpenClose2d(nn.Module):
    """Morphological opening / closing: a composition of erosion and dilation convs.

    ``opening``  = dilation(erosion(x))  (removes small bright features)
    ``closing``  = erosion(dilation(x))  (fills small dark gaps)

    Two channel-preserving :class:`TropicalConv2d` with the same kernel size,
    multiplier-free. Classic morphology made trainable.
    """

    def __init__(self, channels: int, kernel_size: int = 3, op: str = "opening",
                 use_soft: bool = False, tau: float = 0.0):
        super().__init__()
        assert op in ("opening", "closing"), op
        self.op = op
        pad = kernel_size // 2
        first, second = (("erosion", "dilation") if op == "opening"
                         else ("dilation", "erosion"))
        self.conv1 = TropicalConv2d(channels, channels, kernel_size, padding=pad,
                                    mode=first, use_soft=use_soft, tau=tau)
        self.conv2 = TropicalConv2d(channels, channels, kernel_size, padding=pad,
                                    mode=second, use_soft=use_soft, tau=tau)

    def set_tau(self, tau: float) -> None:
        self.conv1.set_tau(tau)
        self.conv2.set_tau(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class AdderLinear(nn.Module):
    """AdderNet l1-distance layer: y[o] = -sum_i |x_i - W[o,i]| + b_o (multiply-free).

    Adjacent to the tropical family (``|x| = max(x, -x)``); included so TikiTorch
    spans the multiply-free taxonomy and offers an AdderNet baseline.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.uniform_(self.weight, -0.5, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return adder_linear(x, self.weight) + self.bias


# ---------------------------------------------------------------------------
# Tropical recurrent cells (RNN taxonomy)
# ---------------------------------------------------------------------------


class TropicalRNNCell(nn.Module):
    """Multiplier-free recurrent cell: tropical state update.

    ``h_t = (min|max)( tropical_linear(x_t, W_x), tropical_linear(h_{t-1}, W_h) ) + b``

    The two projections are tropical (min-plus erosion or max-plus dilation), and
    the state combine is the tropical addition (elementwise min/max). Every op is
    add/compare only — no multiply, no sigmoid/tanh gate. Supports a soft (tau>0)
    warmup that hardens at tau=0, like the rest of the library.
    """

    def __init__(self, input_size: int, hidden_size: int, mode: str = "min",
                 tau: float = 0.0):
        super().__init__()
        assert mode in ("min", "max"), mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.tau = tau
        if mode == "min":
            self.x2h = TropicalLinearModule(input_size, hidden_size, use_softmin=(tau > 0), tau=tau)
            self.h2h = TropicalLinearModule(hidden_size, hidden_size, use_softmin=(tau > 0), tau=tau)
        else:
            self.x2h = TropicalDilationModule(input_size, hidden_size, use_softmax=(tau > 0), tau=tau)
            self.h2h = TropicalDilationModule(hidden_size, hidden_size, use_softmax=(tau > 0), tau=tau)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def set_tau(self, tau: float) -> None:
        self.tau = tau
        self.x2h.set_tau(tau)
        self.h2h.set_tau(tau)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        a, b = self.x2h(x_t), self.h2h(h)
        combined = torch.minimum(a, b) if self.mode == "min" else torch.maximum(a, b)
        return combined + self.bias


class TropicalRNN(nn.Module):
    """Unrolled :class:`TropicalRNNCell` over time, with a streaming ``.step()``.

    ``forward(x:(B,T,input)) -> (outputs:(B,T,hidden), h_final:(B,hidden))``.
    Multiplier-free recurrence for the RNN taxonomy row.
    """

    def __init__(self, input_size: int, hidden_size: int, mode: str = "min",
                 tau: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TropicalRNNCell(input_size, hidden_size, mode=mode, tau=tau)

    def set_tau(self, tau: float) -> None:
        self.cell.set_tau(tau)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One streaming step: (x_t:(B,input), h:(B,hidden)) -> h':(B,hidden)."""
        return self.cell(x_t, h)

    def forward(self, x: torch.Tensor, h=None):
        B, T, _ = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h


# ---------------------------------------------------------------------------
# PyTorch-idiomatic aliases (clean names) for the higher-level layers
# ---------------------------------------------------------------------------

BatchShift = TropicalBatchShift
MorphConv1d = DepthwiseMorphConv1d

# ---------------------------------------------------------------------------
# Predictive coding (local, backprop-free) — see tikitorch.nn.predictive_coding
# ---------------------------------------------------------------------------

from tikitorch.nn.predictive_coding import (  # noqa: E402,F401
    TropicalPCLayer, TropicalPCNet, DyadicPCNet,
)
