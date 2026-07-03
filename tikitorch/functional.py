"""tikitorch.functional — tropical-semiring ops and autograd Functions.

This module is the analogue of ``torch.nn.functional`` for the tropical
(max-plus / min-plus) semiring. It holds the *stateless* building blocks:

  * pure tensor ops (``tropical_matmul``, ``tropical_softmin``, ``tropical_softmax``),
  * the custom ``torch.autograd.Function`` definitions whose backward passes
    route gradients only through the winning argmin/argmax index
    (``TropicalLinearFn``, ``TropicalDilationFn``, the attention primitives),
  * convenience ``*_apply`` aliases (``tropical_linear``, ``tropical_dilation``,
    ``tropical_dot_product``, ``tropical_l_inf_score``, ``tropical_value_pull``)
    and their soft (logsumexp) relaxations.

The ``nn.Module`` wrappers that bundle parameters around these ops live in
:mod:`tikitorch.nn`.

The tropical semiring replaces (sum, product) with (min, plus) [erosion] or
(max, plus) [dilation]:

  * Tropical addition:        a (+) b = min(a, b)   (or max for the dual)
  * Tropical multiplication:  a (*) b = a + b       (standard addition)

This eliminates hardware multipliers from both the forward and the backward
pass — the basis of the algebraic-closure result the library is built around.
"""

import torch

__all__ = [
    # min-plus (erosion) linear
    "TropicalLinearFn",
    "tropical_linear",
    # max-plus (dilation) linear
    "TropicalDilationFn",
    "tropical_dilation",
    # soft relaxations
    "tropical_softmin",
    "tropical_softmax",
    # tropical matmul
    "tropical_matmul",
    # attention: dot-product scoring
    "TropicalDotProductFn",
    "tropical_dot_product",
    "tropical_softmax_dot_product",
    # attention: L_inf scoring
    "TropicalLInfScoreFn",
    "tropical_l_inf_score",
    "tropical_softmax_l_inf_score",
    # attention: Hilbert / tropical projective metric scoring
    "TropicalHilbertScoreFn",
    "tropical_hilbert_score",
    "tropical_softmax_hilbert_score",
    "tropical_asym_score",
    "tropical_softmax_asym_score",
    # l1 / AdderNet (multiply-free, tropical-adjacent)
    "adder_linear",
    "tropical_l1_score",
    # attention: value pull
    "TropicalValuePullFn",
    "tropical_value_pull",
    "tropical_softmax_value_pull",
    # attention: argmax-gather (deploy-pure limit of softmax attention)
    "tropical_value_gather",
    "tropical_softmax_value_gather",
    # attention: top-k softened value pull
    "TropicalValuePullTopKFn",
    "tropical_value_pull_topk",
    "TropicalValuePullIterTopKFn",
    "tropical_value_pull_iter_topk",
    "tropical_softmax_value_pull_iter_topk",
    "tropical_value_pull_min",
    "tropical_softmax_value_pull_min",
    # morphological 2D convolution (max-plus / min-plus)
    "TropicalConv2dFn",
    "tropical_conv2d",
    # tropical activations
    "tropical_relu",
    "tropical_leaky_relu",
    "tropical_hardtanh",
    # loss conditioning (perceptron baseline + closure-faithful cross-entropy)
    "shift_softmax",
    "exp2_lut_table",
    "TropicalPerceptronLossFn",
    "tropical_perceptron_loss",
    "TropicalCrossEntropyFn",
    "tropical_cross_entropy",
    "continuous_cross_entropy_ref",
]


# ---------------------------------------------------------------------------
# Custom autograd: hard tropical linear (min-plus / erosion)
# ---------------------------------------------------------------------------


class TropicalLinearFn(torch.autograd.Function):
    """Tropical linear: y_j = min_i(W[j,i] + x[i]).

    Custom autograd to produce sparse gradients.
    Gradient: dL/dx_i = sum_j (dL/dy_j * 1{i == argmin_i(W[j,:]+x[:])})
              dL/dW[j,i] = dL/dy_j * 1{i == argmin}
    """

    @staticmethod
    def forward(ctx, x, weight):
        # x: (batch, in_features), weight: (out_features, in_features)
        expanded = weight.unsqueeze(0) + x.unsqueeze(1)  # (batch, out, in)
        out, indices = expanded.min(dim=-1)  # (batch, out)
        ctx.save_for_backward(indices)
        ctx.shape = x.shape
        ctx.weight_shape = weight.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        batch, in_features = ctx.shape
        out_features, _ = ctx.weight_shape

        # grad_output: (batch, out_features)
        # indices:     (batch, out_features) — argmin index per output unit

        # Gradient w.r.t. x: scatter grad_output into input positions
        grad_x = torch.zeros(batch, in_features, device=grad_output.device, dtype=grad_output.dtype)
        grad_x.scatter_add_(1, indices, grad_output)

        # Gradient w.r.t. weight: one-hot at argmin, scaled by grad_output
        grad_weight = torch.zeros(
            batch, out_features, in_features,
            device=grad_output.device, dtype=grad_output.dtype,
        )
        grad_weight.scatter_(2, indices.unsqueeze(-1), grad_output.unsqueeze(-1))
        grad_weight = grad_weight.sum(dim=0)  # (out_features, in_features)

        return grad_x, grad_weight


tropical_linear = TropicalLinearFn.apply


# ---------------------------------------------------------------------------
# Soft-min relaxation
# ---------------------------------------------------------------------------


def tropical_softmin(z: torch.Tensor, tau: float = 1.0, dim: int = -1) -> torch.Tensor:
    """Differentiable relaxation of min using softmin.

    softmin(z, tau) = -tau * logsumexp(-z / tau)

    Approaches hard min as tau -> 0.
    """
    return -tau * torch.logsumexp(-z / tau, dim=dim)


# ---------------------------------------------------------------------------
# Tropical matrix multiply
# ---------------------------------------------------------------------------


def tropical_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Tropical matrix multiply: C[i,j] = min_k(A[i,k] + B[k,j]).

    A: (..., m, k), B: (..., k, n) -> C: (..., m, n)
    """
    # A[..., :, :, None] + B[..., None, :, :] -> (..., m, k, n)
    expanded = A.unsqueeze(-1) + B.unsqueeze(-3)
    return expanded.min(dim=-2).values


# ---------------------------------------------------------------------------
# Custom autograd: hard tropical dilation (max-plus)
# ---------------------------------------------------------------------------


class TropicalDilationFn(torch.autograd.Function):
    """Tropical dilation: y_j = max_i(W[j,i] + x[i]).

    Max-plus dual of TropicalLinearFn.  Gradient routing is identical
    in structure — only the argmax index receives the upstream gradient.
    """

    @staticmethod
    def forward(ctx, x, weight):
        expanded = weight.unsqueeze(0) + x.unsqueeze(1)  # (batch, out, in)
        out, indices = expanded.max(dim=-1)  # (batch, out)
        ctx.save_for_backward(indices)
        ctx.shape = x.shape
        ctx.weight_shape = weight.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        batch, in_features = ctx.shape
        out_features, _ = ctx.weight_shape

        grad_x = torch.zeros(batch, in_features, device=grad_output.device, dtype=grad_output.dtype)
        grad_x.scatter_add_(1, indices, grad_output)

        grad_weight = torch.zeros(
            batch, out_features, in_features,
            device=grad_output.device, dtype=grad_output.dtype,
        )
        grad_weight.scatter_(2, indices.unsqueeze(-1), grad_output.unsqueeze(-1))
        grad_weight = grad_weight.sum(dim=0)

        return grad_x, grad_weight


tropical_dilation = TropicalDilationFn.apply


# ---------------------------------------------------------------------------
# Soft-max relaxation (dual of soft-min)
# ---------------------------------------------------------------------------


def tropical_softmax(z: torch.Tensor, tau: float = 1.0, dim: int = -1) -> torch.Tensor:
    """Differentiable relaxation of max using softmax.

    softmax(z, tau) = tau * logsumexp(z / tau)

    Approaches hard max as tau -> 0.
    """
    return tau * torch.logsumexp(z / tau, dim=dim)


# ---------------------------------------------------------------------------
# Tropical attention primitives
# ---------------------------------------------------------------------------
#
# Two stateless ops that compose into a non-degenerate tropical attention:
#   scores = tropical_dot_product(Q, K)           # max_d (Q + K)
#   out    = tropical_value_pull(V, scores)       # per-dim max_j (V + scores)
#
# Q, K, V carry interaction information through the d-dim argmax (preserves Q-K
# coupling) and the j-dim argmax (per-output-dim key selection).
#
# Soft (tau > 0) variants apply tropical_softmax (logsumexp) and rely on
# standard autograd; hard variants use custom autograd that routes gradients
# only through the argmax positions — the FPGA-mappable form.


class TropicalDotProductFn(torch.autograd.Function):
    """Hard tropical inner product: scores[..., i, j] = max_d(Q[..., i, d] + K[..., j, d]).

    Q: (..., L_q, D), K: (..., L_k, D) -> scores: (..., L_q, L_k).

    Backward routes gradient only through the d-argmax: for each (i, j),
    grad_score flows to Q[..., i, d*] and K[..., j, d*] where d* = argmax.
    """

    @staticmethod
    def forward(ctx, Q, K):
        # expanded[..., i, j, d] = Q[..., i, d] + K[..., j, d]
        expanded = Q.unsqueeze(-2) + K.unsqueeze(-3)  # (..., L_q, L_k, D)
        scores, indices = expanded.max(dim=-1)         # (..., L_q, L_k)
        ctx.save_for_backward(indices)
        ctx.q_shape = Q.shape
        ctx.k_shape = K.shape
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        (indices,) = ctx.saved_tensors

        grad_Q = torch.zeros(ctx.q_shape, device=grad_scores.device, dtype=grad_scores.dtype)
        grad_Q.scatter_add_(-1, indices, grad_scores)

        grad_K = torch.zeros(ctx.k_shape, device=grad_scores.device, dtype=grad_scores.dtype)
        grad_K.scatter_add_(-1, indices.transpose(-2, -1), grad_scores.transpose(-2, -1))

        return grad_Q, grad_K


tropical_dot_product = TropicalDotProductFn.apply


def tropical_softmax_dot_product(Q: torch.Tensor, K: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Soft tropical inner product: dense gradient version, approaches hard as tau -> 0.

    scores[..., i, j] = tau * logsumexp((Q[..., i, d] + K[..., j, d]) / tau, dim=d)
    """
    expanded = Q.unsqueeze(-2) + K.unsqueeze(-3)  # (..., L_q, L_k, D)
    return tropical_softmax(expanded, tau=tau, dim=-1)


# ---------------------------------------------------------------------------
# Variant (c): L_infinity distance scoring (genuinely symmetric in Q and K).
#
# scores[i, j] = -max_d |Q[i, d] - K[j, d]|
#
# The dot-product variant `max_d(Q+K)` is mathematically non-degenerate but
# K-dominated in practice — the K with the single largest coordinate wins
# for almost any Q. The L_inf distance is symmetric: high score (near zero)
# only when Q ~ K dimension-by-dimension. abs(x) = max(x, -x) is tropical
# (negation + comparison), so closure is preserved.
# ---------------------------------------------------------------------------


class TropicalLInfScoreFn(torch.autograd.Function):
    """Hard L_inf-distance scoring: scores[i, j] = -max_d |Q[i, d] - K[j, d]|.

    Backward routes gradient through the (d-argmax, sign) winner.
    """

    @staticmethod
    def forward(ctx, Q, K):
        diff = Q.unsqueeze(-2) - K.unsqueeze(-3)         # (..., L_q, L_k, D)
        abs_diff = diff.abs()
        max_abs, indices = abs_diff.max(dim=-1)          # (..., L_q, L_k)
        # Sign of the winning diff: +1 if Q > K at d*, else -1.
        gathered = torch.gather(diff, -1, indices.unsqueeze(-1)).squeeze(-1)
        sign = torch.sign(gathered)
        ctx.save_for_backward(indices, sign)
        ctx.q_shape = Q.shape
        ctx.k_shape = K.shape
        return -max_abs

    @staticmethod
    def backward(ctx, grad_scores):
        # d/dQ[i, d*] of -|Q[i,d*] - K[j,d*]| = -sign(Q-K) at d*.
        # d/dK[j, d*] of -|Q[i,d*] - K[j,d*]| =  sign(Q-K) at d*.
        indices, sign = ctx.saved_tensors

        grad_Q = torch.zeros(ctx.q_shape, device=grad_scores.device, dtype=grad_scores.dtype)
        grad_Q.scatter_add_(-1, indices, -grad_scores * sign)

        grad_K = torch.zeros(ctx.k_shape, device=grad_scores.device, dtype=grad_scores.dtype)
        grad_K.scatter_add_(-1,
                             indices.transpose(-2, -1),
                             (grad_scores * sign).transpose(-2, -1))
        return grad_Q, grad_K


tropical_l_inf_score = TropicalLInfScoreFn.apply


def tropical_softmax_l_inf_score(Q: torch.Tensor, K: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Soft (dense-gradient) L_inf scoring: -tau * logsumexp(|Q-K|/tau, dim=d).

    Approaches the hard L_inf score as tau -> 0.
    """
    abs_diff = (Q.unsqueeze(-2) - K.unsqueeze(-3)).abs()  # (..., L_q, L_k, D)
    return -tropical_softmax(abs_diff, tau=tau, dim=-1)


# ---------------------------------------------------------------------------
# Variant (d): Hilbert / tropical projective metric scoring.
#
# From "Tropical Attention: Neural Algorithmic Reasoning for Combinatorial
# Algorithms" (Hashemi et al., NeurIPS 2025; github.com/Baran-phys/Tropical-Attention).
# The tropical (Hilbert projective) distance between Q_i and K_j is the *range* of
# their coordinate-wise difference:
#
#   d_trop(Q_i, K_j) = max_d (Q_i[d] - K_j[d]) - min_d (Q_i[d] - K_j[d])
#   score[i, j]      = - d_trop(Q_i, K_j)
#
# Unlike the dot-product variant (K-dominated) and the L_inf variant, this is
# *projective*: adding any constant to Q_i or K_j leaves the difference range
# unchanged, so the metric is translation/scale-invariant and preserves sharp
# polyhedral decision boundaries (the property that makes tropical attention
# strong on combinatorial/algorithmic reasoning). Multiplier-free: subtract,
# max, min only. abs(x)=max(x,-x) so the L_inf variant is a special case;
# the Hilbert range is the genuinely projective one.
# ---------------------------------------------------------------------------


class TropicalHilbertScoreFn(torch.autograd.Function):
    """Hard Hilbert/projective tropical scoring: score[i,j] = -(max_d - min_d)(Q_i - K_j).

    Backward routes gradient through the per-(i,j) argmax_d and argmin_d winners
    (a 2-index sparse scatter), like the L_inf variant but with both extremes.
    """

    @staticmethod
    def forward(ctx, Q, K):
        diff = Q.unsqueeze(-2) - K.unsqueeze(-3)          # (..., L_q, L_k, D)
        max_diff, amax = diff.max(dim=-1)                 # (..., L_q, L_k)
        min_diff, amin = diff.min(dim=-1)
        ctx.save_for_backward(amax, amin)
        ctx.q_shape = Q.shape
        ctx.k_shape = K.shape
        return min_diff - max_diff                        # = -(max - min)

    @staticmethod
    def backward(ctx, grad_scores):
        amax, amin = ctx.saved_tensors
        g = grad_scores
        # d(score)/dQ: +g at argmin_d, -g at argmax_d (diff = Q - K, dQ = +1).
        grad_Q = torch.zeros(ctx.q_shape, device=g.device, dtype=g.dtype)
        grad_Q.scatter_add_(-1, amin, g)
        grad_Q.scatter_add_(-1, amax, -g)
        # d(score)/dK: -g at argmin_d, +g at argmax_d (dK = -1).
        grad_K = torch.zeros(ctx.k_shape, device=g.device, dtype=g.dtype)
        grad_K.scatter_add_(-1, amin.transpose(-2, -1), (-g).transpose(-2, -1))
        grad_K.scatter_add_(-1, amax.transpose(-2, -1), g.transpose(-2, -1))
        return grad_Q, grad_K


tropical_hilbert_score = TropicalHilbertScoreFn.apply


def tropical_softmax_hilbert_score(Q: torch.Tensor, K: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Soft (dense-gradient) Hilbert scoring; approaches the hard projective metric as tau -> 0.

    score[i,j] = -tau*logsumexp(diff/tau, d) - tau*logsumexp(-diff/tau, d)
               = softmin_d(diff) - softmax_d(diff)   (-> min - max as tau -> 0)
    """
    diff = Q.unsqueeze(-2) - K.unsqueeze(-3)              # (..., L_q, L_k, D)
    return tropical_softmin(diff, tau=tau, dim=-1) - tropical_softmax(diff, tau=tau, dim=-1)


def tropical_asym_score(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Asymmetric tropical (one-sided projective) score (Hashemi et al., 2025).

    score[i,j] = -(sum_d(Q_i - K_j) - D * min_d(Q_i - K_j)) = -sum_d(diff - min_d diff).

    The one-sided variant of the Hilbert metric: it measures how far each
    coordinate is above the per-pair minimum. Translation-invariant in Q
    (subtracting the row min), multiplier-free (subtract, min, sum). Standard
    autograd through min/sum suffices.
    """
    diff = Q.unsqueeze(-2) - K.unsqueeze(-3)              # (..., L_q, L_k, D)
    return -(diff - diff.min(dim=-1, keepdim=True).values).sum(dim=-1)


def tropical_softmax_asym_score(Q: torch.Tensor, K: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Soft asymmetric score: uses softmin for the per-pair minimum (-> hard as tau->0)."""
    diff = Q.unsqueeze(-2) - K.unsqueeze(-3)
    m = tropical_softmin(diff, tau=tau, dim=-1).unsqueeze(-1)
    return -(diff - m).sum(dim=-1)


# ---------------------------------------------------------------------------
# l1 / AdderNet-style distance projection (multiply-free; |x| = max(x,-x))
# ---------------------------------------------------------------------------
#
# AdderNet (Chen et al., 2020) replaces the matmul with a negative l1 distance,
# y_o = -sum_i |x_i - W[o,i]|, using only subtraction, abs, and addition -- a
# multiply-free op adjacent to the tropical family (abs(x) = max(x, -x) is
# tropical). Included so TikiTorch spans the multiply-free taxonomy and offers a
# strong baseline; assert_closed-clean.


def adder_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """AdderNet l1 layer: y[..., o] = -sum_i |x[..., i] - weight[o, i]|.

    ``x: (..., in)``, ``weight: (out, in)`` -> ``(..., out)``. Multiply-free.
    """
    return -(x.unsqueeze(-2) - weight).abs().sum(dim=-1)


# ---------------------------------------------------------------------------
# l1 attention SCORE: scores[i, j] = -sum_d |Q[i, d] - K[j, d]|  (sum over ALL
# dims, the AdderNet/many-to-many analogue of the dot product). Multiplier-free
# (subtract, abs=max(x,-x), sum). UNLIKE the l_inf score (-max_d|.|, one-dim,
# sparse gradient) this accumulates over every dimension, so its gradient is
# DENSE -- measured to be the most robust multiplier-free score for content
# routing (induction probe, 2026-06-16: l1 matches dot attention at the ceiling).
# Already smooth (sum, not max), so there is no separate soft relaxation: the
# hard score IS dense-gradient. Pairs with the argmax-GATHER aggregation.
# ---------------------------------------------------------------------------


def tropical_l1_score(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """l1-distance attention score: ``scores[..., i, j] = -sum_d |Q[i,d] - K[j,d]|``.

    ``Q: (..., L_q, D)``, ``K: (..., L_k, D)`` -> ``(..., L_q, L_k)``. Multiply-free
    (subtract + abs + sum); dense gradient over all D dims (cf. the one-dim l_inf).
    """
    return -(Q.unsqueeze(-2) - K.unsqueeze(-3)).abs().sum(dim=-1)


class TropicalValuePullFn(torch.autograd.Function):
    """Hard tropical value pull: out[..., i, d] = max_j(V[..., j, d] + scores[..., i, j]).

    V: (..., L_v, D), scores: (..., L_q, L_v) -> out: (..., L_q, D).

    Per-dim aggregation: each output dimension picks its own argmax_j.
    Backward routes gradient only through that j*: grad_out[..., i, d] flows
    to V[..., j*, d] and scores[..., i, j*].
    """

    @staticmethod
    def forward(ctx, V, scores):
        # expanded[..., i, j, d] = V[..., j, d] + scores[..., i, j]
        expanded = V.unsqueeze(-3) + scores.unsqueeze(-1)  # (..., L_q, L_v, D)
        out, indices = expanded.max(dim=-2)                 # (..., L_q, D)
        ctx.save_for_backward(indices)
        ctx.v_shape = V.shape
        ctx.scores_shape = scores.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (indices,) = ctx.saved_tensors

        # grad_V[..., j*, d] += grad_out[..., i, d] where j* = indices[..., i, d]
        grad_V = torch.zeros(ctx.v_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_V.scatter_add_(-2, indices, grad_out)

        # grad_scores[..., i, j*] += grad_out[..., i, d] where j* = indices[..., i, d]
        grad_scores = torch.zeros(ctx.scores_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_scores.scatter_add_(-1, indices, grad_out)

        return grad_V, grad_scores


tropical_value_pull = TropicalValuePullFn.apply


def tropical_softmax_value_pull(V: torch.Tensor, scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Soft tropical value pull: dense gradient version, approaches hard as tau -> 0.

    out[..., i, d] = tau * logsumexp((V[..., j, d] + scores[..., i, j]) / tau, dim=j)
    """
    expanded = V.unsqueeze(-3) + scores.unsqueeze(-1)  # (..., L_q, L_v, D)
    return tropical_softmax(expanded, tau=tau, dim=-2)


# --- MIN-plus (erosion) value pull: the (min,+) twin of the max-plus pull ------
# Min-plus DPs (shortest path, min-coin-change, ...) aggregate keys by MIN, not
# MAX: out[i,d] = min_j (V[j,d] + scores[i,j]). We get it for free from the
# validated max-plus autograd via min_j x = -max_j(-x), so gradient routing,
# masking (-inf), and ties are identical and already gradchecked. Negation is a
# sign flip (aten.neg) -- still add/compare only, so closure is preserved.

def tropical_value_pull_min(V: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Hard min-plus value pull: ``out[..., i, d] = min_j (V[..., j, d] + scores[..., i, j])``."""
    return -tropical_value_pull(-V, -scores)


def tropical_softmax_value_pull_min(V: torch.Tensor, scores: torch.Tensor,
                                    tau: float = 1.0) -> torch.Tensor:
    """Soft min-plus value pull; approaches the hard min as ``tau -> 0``."""
    return -tropical_softmax_value_pull(-V, -scores, tau=tau)


# --- argmax-GATHER aggregation: the deploy-pure limit of softmax ATTENTION -----
# This is DIFFERENT from tropical_value_pull (the max-plus matmul, which picks a
# separate argmax_j PER output dim d -> a per-dimension mixture / "chimera").
# Gather picks ONE key j* per query (shared across all dims) and copies its whole
# value vector -- i.e. "attend to one position, copy it", the genuine tau->0 hard
# limit of softmax(scores)@V. Measured (induction probe, 2026-06-16): the max-plus
# matmul collapses a soft-trained attention to chance, while argmax-gather is
# LOSSLESS (soft acc == gather acc == softmax ceiling). So gather is the correct
# multiplier-free deploy op for content routing; the matmul is a different operator.
#
#   train (tau>0): out = softmax(scores/tau) @ V   (dense gradient; the @V is the
#                  only train-time multiply -- deleted at deploy, north-star OK)
#   deploy (tau=0): out[..., i, :] = V[..., argmax_j scores[..., i, j], :]
#                  (argmax + gather: ZERO multiply)

def tropical_value_gather(V: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Hard argmax-gather: ``out[..., i, :] = V[..., argmax_j scores[..., i, j], :]``.

    ``V: (..., L_v, D)``, ``scores: (..., L_q, L_v)`` -> ``(..., L_q, D)``. One key
    per query (shared across dims). Multiply-free (argmax + gather). The grad w.r.t.
    scores is zero (argmax non-differentiable) -- train via the soft variant.
    """
    jstar = scores.argmax(dim=-1)                                  # (..., L_q)
    idx = jstar.unsqueeze(-1).expand(*jstar.shape, V.shape[-1])    # (..., L_q, D)
    return torch.gather(V, -2, idx)


def tropical_softmax_value_gather(V: torch.Tensor, scores: torch.Tensor,
                                  tau: float = 1.0) -> torch.Tensor:
    """Soft attention readout (train-time relaxation of :func:`tropical_value_gather`):
    ``out[..., i, :] = sum_j softmax(scores[..., i, :] / tau)[j] * V[..., j, :]``.

    Convex combination with weights SHARED across output dims (unlike the per-dim
    softmax value pull). As ``tau -> 0`` the softmax -> one-hot at ``argmax_j`` so
    this -> :func:`tropical_value_gather`. Contains a multiply (weights @ V): a
    training-only relaxation, removed at deploy.
    """
    w = torch.softmax(scores / tau, dim=-1)                        # (..., L_q, L_v)
    return torch.einsum("...ij,...jd->...id", w, V)


# ---------------------------------------------------------------------------
# Top-k softened value pull (hard, multiplier-free; an alternative to tau>0)
# ---------------------------------------------------------------------------
#
# The hard value pull (k=1) routes each output (i, d) through a SINGLE key j*,
# so the backward is a one-index scatter. That single-winner sparsity is part of
# the hard-tropical accuracy cost. Top-k softening averages the k largest keys
# per (i, d), so the gradient flows to k keys instead of one -- a DENSER gradient
# with a DENSER gradient (the k>1 mean introduces a divide by the data-dependent
# winner count under masking -- see the note in TropicalValuePullTopKFn -- but the
# reduction itself is add/compare; for a strictly multiply-free hard path use k=1)
# while staying add/compare for the reduction (no logsumexp, unlike the
# tau>0 soft path). Whether this *helps* accuracy is an empirical, task-dependent
# question (it can hurt: it shifts the forward toward an average), so treat it as
# an ablation knob, not a guaranteed lift.


class TropicalValuePullTopKFn(torch.autograd.Function):
    """Top-k tropical value pull: mean of the k largest keys per (query, dim).

    ``out[..., i, d] = mean over the k largest j of (V[..., j, d] + scores[..., i, j])``.

    Ties resolve to the lower key index (stable descending sort), matching the
    silicon kernel's serial first-wins scan. Non-finite entries (e.g. a causal
    ``-inf`` mask, or fewer than k valid keys) are excluded from both the average
    and the gradient. ``k == 1`` reduces exactly to :func:`tropical_value_pull`.

    Backward routes ``grad_out / (#finite winners)`` to each finite winner,
    accumulating with ``scatter_add_`` across BOTH the k ranks and the output
    dims ``d`` (a key that wins for several dims sums their contributions).
    """

    @staticmethod
    def forward(ctx, V, scores, k):
        Lv = V.shape[-2]
        kk = max(1, min(int(k), Lv))
        expanded = V.unsqueeze(-3) + scores.unsqueeze(-1)          # (...,Lq,Lv,D)
        order = torch.argsort(expanded, dim=-2, descending=True, stable=True)
        topidx = order.narrow(-2, 0, kk)                          # (...,Lq,kk,D)
        topvals = torch.gather(expanded, -2, topidx)              # (...,Lq,kk,D)
        finite = torch.isfinite(topvals)
        any_finite = finite.any(dim=-2)                          # (...,Lq,D)
        cnt = finite.sum(dim=-2).clamp(min=1)                     # (...,Lq,D)
        out = torch.where(finite, topvals, torch.zeros_like(topvals)).sum(dim=-2) / cnt
        # A7: a fully-masked row (no finite key) must propagate -inf, like the
        # hard/soft value pulls -- not a deceptively-valid 0.0.
        out = torch.where(any_finite, out, torch.full_like(out, float("-inf")))
        ctx.save_for_backward(topidx, finite, cnt)
        ctx.v_shape = V.shape
        ctx.scores_shape = scores.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        topidx, finite, cnt = ctx.saved_tensors                  # topidx (...,Lq,kk,D)
        g = (grad_out / cnt).unsqueeze(-2)                        # (...,Lq,1,D)
        g_full = torch.where(finite, g.expand(finite.shape),
                             torch.zeros((), dtype=grad_out.dtype, device=grad_out.device))

        # grad_V (...,Lv,D): scatter over keys (dim=-2), flattening (Lq, kk) -> M,
        # so contributions accumulate across queries i AND the k ranks.
        D = topidx.shape[-1]
        idx_v = topidx.reshape(*topidx.shape[:-3], -1, D)
        src_v = g_full.reshape(*g_full.shape[:-3], -1, D)
        grad_V = torch.zeros(ctx.v_shape, dtype=grad_out.dtype, device=grad_out.device)
        grad_V.scatter_add_(-2, idx_v, src_v)

        # grad_scores (...,Lq,Lv): scatter over keys (dim=-1), flattening (kk, D)
        # so contributions accumulate across the k ranks AND the output dims d.
        idx_s = topidx.reshape(*topidx.shape[:-2], -1)
        src_s = g_full.reshape(*g_full.shape[:-2], -1)
        grad_scores = torch.zeros(ctx.scores_shape, dtype=grad_out.dtype, device=grad_out.device)
        grad_scores.scatter_add_(-1, idx_s, src_s)

        return grad_V, grad_scores, None


def tropical_value_pull_topk(V: torch.Tensor, scores: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Top-k softened tropical value pull (see :class:`TropicalValuePullTopKFn`)."""
    return TropicalValuePullTopKFn.apply(V, scores, k)


# ---------------------------------------------------------------------------
# Iterative-argmax (winner-exclusion) top-k value pull  — STRICTLY closed
# ---------------------------------------------------------------------------
#
# "Iterative argmax, leaving out the last winner k times": pick the best key,
# exclude it, pick the next, ... k times, then COMBINE BY SUM (with an optional
# learnable per-rank additive offset). Unlike the mean-combine top-k above, the
# sum has NO data-dependent divide, so the hard path is add/compare only -- a
# strictly multiplier-free, FPGA-mappable selective attention that gives denser
# gradients than top-1 argmax and matches the silicon serial-comparator scan.
# k == 1 reduces exactly to tropical_value_pull (the single argmax = sum of top-1).


class TropicalValuePullIterTopKFn(torch.autograd.Function):
    """Iterative-argmax top-k value pull, combined by SUM (multiplier-free).

    ``out[..., i, d] = sum over the k largest j of (V[..., j, d] + scores[..., i, j] + b_r)``
    where ``b_r`` is an optional additive per-rank bias (``b_0`` typically 0). The
    sum-combine keeps lower ranks contributing (a plain max over the top-k would
    degenerate to top-1). Non-finite entries (causal ``-inf`` mask / fewer than k
    valid keys) are excluded; a fully-masked row yields ``-inf``. Backward routes
    ``grad_out`` UNSCALED (coefficient 1) to each winner of ``V`` and ``scores``,
    and sums it per rank into ``rank_bias`` -- all add/scatter, no divide.
    """

    @staticmethod
    def forward(ctx, V, scores, k, rank_bias):
        Lv = V.shape[-2]
        kk = max(1, min(int(k), Lv))
        expanded = V.unsqueeze(-3) + scores.unsqueeze(-1)          # (...,Lq,Lv,D)
        order = torch.argsort(expanded, dim=-2, descending=True, stable=True)
        topidx = order.narrow(-2, 0, kk)                          # (...,Lq,kk,D)
        topvals = torch.gather(expanded, -2, topidx)             # (...,Lq,kk,D)
        finite = torch.isfinite(topvals)
        contrib = topvals
        if rank_bias is not None:
            # Lv can clip kk below k (e.g. a 1-token generation prompt); only the
            # first kk rank biases participate.
            rb = rank_bias[:kk].reshape(*([1] * (topvals.dim() - 2)), kk, 1)
            contrib = contrib + rb
        contrib = torch.where(finite, contrib, torch.zeros_like(contrib))
        out = contrib.sum(dim=-2)                                 # SUM-combine, no divide
        any_finite = finite.any(dim=-2)
        out = torch.where(any_finite, out, torch.full_like(out, float("-inf")))
        ctx.save_for_backward(topidx, finite)
        ctx.v_shape = V.shape
        ctx.scores_shape = scores.shape
        ctx.has_bias = rank_bias is not None
        ctx.bias_len = rank_bias.shape[0] if rank_bias is not None else 0
        ctx.kk = kk
        return out

    @staticmethod
    def backward(ctx, grad_out):
        topidx, finite = ctx.saved_tensors                       # (...,Lq,kk,D)
        g = grad_out.unsqueeze(-2)                                # (...,Lq,1,D)
        g_full = torch.where(finite, g.expand(finite.shape),
                             torch.zeros((), dtype=grad_out.dtype, device=grad_out.device))
        D = topidx.shape[-1]
        # grad_V (...,Lv,D): scatter grad to each winner key, accumulate over (i, rank).
        idx_v = topidx.reshape(*topidx.shape[:-3], -1, D)
        src_v = g_full.reshape(*g_full.shape[:-3], -1, D)
        grad_V = torch.zeros(ctx.v_shape, dtype=grad_out.dtype, device=grad_out.device)
        grad_V.scatter_add_(-2, idx_v, src_v)
        # grad_scores (...,Lq,Lv): accumulate over (rank, dim d).
        idx_s = topidx.reshape(*topidx.shape[:-2], -1)
        src_s = g_full.reshape(*g_full.shape[:-2], -1)
        grad_scores = torch.zeros(ctx.scores_shape, dtype=grad_out.dtype, device=grad_out.device)
        grad_scores.scatter_add_(-1, idx_s, src_s)
        # grad_rank_bias (kk,): sum grad over everything but the rank dim (-2).
        grad_bias = None
        if ctx.has_bias:
            reduce_dims = tuple(d for d in range(g_full.dim()) if d != g_full.dim() - 2)
            grad_bias = g_full.sum(dim=reduce_dims)
            if grad_bias.shape[0] < ctx.bias_len:   # kk was clipped by Lv < k
                grad_bias = torch.cat(
                    [grad_bias, grad_bias.new_zeros(ctx.bias_len - grad_bias.shape[0])])
        return grad_V, grad_scores, None, grad_bias


def tropical_value_pull_iter_topk(V: torch.Tensor, scores: torch.Tensor, k: int = 2,
                                  rank_bias: torch.Tensor = None) -> torch.Tensor:
    """Iterative-argmax winner-exclusion top-k value pull, SUM-combined (closed).

    See :class:`TropicalValuePullIterTopKFn`. ``rank_bias`` is an optional ``(k,)``
    additive per-rank offset (a closed, learnable parameter).
    """
    return TropicalValuePullIterTopKFn.apply(V, scores, k, rank_bias)


def tropical_softmax_value_pull_iter_topk(V: torch.Tensor, scores: torch.Tensor,
                                          k: int = 2, tau: float = 1.0,
                                          rank_bias: torch.Tensor = None) -> torch.Tensor:
    """Soft (trainable) iterative top-k value pull; anneals to the hard sum as tau->0.

    Literally mirrors argmax / remove-winner / repeat with soft operators: each
    rank is a logsumexp soft-max over keys, and the winner is softly removed by
    subtracting ``~inf`` (``tau*log1p(-w)``) at its soft-argmax location before the
    next rank. As ``tau -> 0`` this converges to
    :func:`tropical_value_pull_iter_topk`. Training-only (uses exp); the deployed
    ``tau==0`` path uses the hard, multiplier-free function.
    """
    eps = 1e-6
    m = V.unsqueeze(-3) + scores.unsqueeze(-1)                   # (...,Lq,Lv,D)
    Lv = m.shape[-2]
    kk = max(1, min(int(k), Lv))
    total = None
    for r in range(kk):
        s_r = tropical_softmax(m, tau=tau, dim=-2)               # (...,Lq,D)
        if rank_bias is not None:
            s_r = s_r + rank_bias[r]
        total = s_r if total is None else total + s_r
        if r + 1 < kk:
            w = torch.softmax(m / tau, dim=-2)                  # soft one-hot winner
            # softly remove the winner: add log(1-w) (NOT scaled by tau -- scaling
            # by tau would make the removal vanish as tau->0 and the next rank would
            # re-pick the same key). w~1 at the winner -> ~ -inf there, ~0 elsewhere.
            m = m + torch.log1p(-w.clamp(max=1.0 - eps))
    return total


# ---------------------------------------------------------------------------
# Morphological 2D convolution (max-plus dilation / min-plus erosion)
# ---------------------------------------------------------------------------
#
# The tropical analogue of a 2D convolution is exactly a grayscale
# *morphological* operation: dilation (max-plus) or erosion (min-plus) of the
# image by a learned structuring element. It is a direct 4-D lift of
# ``TropicalLinearFn`` -- patches are extracted by ``unfold`` (pure gather, no
# arithmetic), the structuring element is ADDED, and a single max/min reduces
# over the (Cin x Kh x Kw) taps. The backward routes the upstream gradient only
# through the winning tap per output (sparse one-hot scatter, then ``fold`` to
# scatter back into image space) -- no multiply anywhere.


class TropicalConv2dFn(torch.autograd.Function):
    """Morphological 2D convolution: y = (min|max) over taps of (weight + patch).

    ``x:(B,Cin,H,W)``, ``weight:(Cout,Cin,Kh,Kw)`` -> ``(B,Cout,Hout,Wout)``.

      dilation (max-plus): y[b,co,p] = max_{ci,kh,kw} (w[co,ci,kh,kw] + x_patch)
      erosion  (min-plus): y[b,co,p] = min_{ci,kh,kw} (w[co,ci,kh,kw] + x_patch)

    Multiplier-free: ``unfold``/``fold`` are gather/scatter (no arithmetic), the
    combine is an add, the reduce is a compare. Backward scatters grad_output to
    the single winning (ci,kh,kw) tap per output (one-hot), exactly like
    :class:`TropicalLinearFn` lifted to 4-D.
    """

    @staticmethod
    def forward(ctx, x, weight, stride, padding, dilation, mode):
        B, Cin, H, W = x.shape
        Cout, Cin_w, Kh, Kw = weight.shape
        assert Cin == Cin_w, f"in_channels mismatch: x has {Cin}, weight has {Cin_w}"
        s, p, d = stride, padding, dilation
        Hout = (H + 2 * p - d * (Kh - 1) - 1) // s + 1
        Wout = (W + 2 * p - d * (Kw - 1) - 1) // s + 1
        P = Cin * Kh * Kw

        patches = torch.nn.functional.unfold(
            x, (Kh, Kw), dilation=d, padding=p, stride=s)            # (B, P, L)
        wflat = weight.reshape(Cout, P)                              # (Cout, P)
        # combined[b, co, t, l] = patches[b, t, l] + wflat[co, t]
        combined = patches.unsqueeze(1) + wflat.unsqueeze(0).unsqueeze(-1)  # (B,Cout,P,L)
        if mode == "dilation":
            y, idx = combined.max(dim=2)                            # (B,Cout,L)
        elif mode == "erosion":
            y, idx = combined.min(dim=2)
        else:
            raise ValueError(f"mode must be 'dilation' or 'erosion', got {mode!r}")

        ctx.save_for_backward(idx)
        ctx.x_shape = x.shape
        ctx.w_shape = weight.shape
        ctx.params = (s, p, d, Kh, Kw, P, Cout, Hout, Wout)
        return y.reshape(B, Cout, Hout, Wout)

    @staticmethod
    def backward(ctx, grad_output):
        (idx,) = ctx.saved_tensors                                  # (B,Cout,L)
        B, Cin, H, W = ctx.x_shape
        Cout, Cin_w, Kh, Kw = ctx.w_shape
        s, p, d, Kh, Kw, P, Cout, Hout, Wout = ctx.params
        L = Hout * Wout
        g = grad_output.reshape(B, Cout, L)                         # (B,Cout,L)

        # grad_weight: accumulate over (b, l) at the winning tap per (co).
        g_co = g.permute(1, 0, 2).reshape(Cout, B * L)
        idx_co = idx.permute(1, 0, 2).reshape(Cout, B * L)
        grad_wflat = torch.zeros(Cout, P, dtype=grad_output.dtype, device=grad_output.device)
        grad_wflat.scatter_add_(1, idx_co, g_co)
        grad_weight = grad_wflat.reshape(Cout, Cin, Kh, Kw)

        # grad_patches: accumulate over co at the winning tap per (b, l), then
        # fold back into image space (fold sums overlapping patches -- the exact
        # transpose of unfold, gather/scatter only).
        grad_patches = torch.zeros(B, P, L, dtype=grad_output.dtype, device=grad_output.device)
        grad_patches.scatter_add_(1, idx, g)
        grad_x = torch.nn.functional.fold(
            grad_patches, (H, W), (Kh, Kw), dilation=d, padding=p, stride=s)

        return grad_x, grad_weight, None, None, None, None


def tropical_conv2d(x: torch.Tensor, weight: torch.Tensor, stride: int = 1,
                    padding: int = 0, dilation: int = 1, mode: str = "dilation") -> torch.Tensor:
    """Morphological 2D conv (see :class:`TropicalConv2dFn`). Multiplier-free."""
    return TropicalConv2dFn.apply(x, weight, stride, padding, dilation, mode)


# ---------------------------------------------------------------------------
# Tropical activations (all compare/shift only — zero multiply)
# ---------------------------------------------------------------------------


def tropical_relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU = max(x, 0). The canonical tropical (max-plus with 0) activation."""
    return torch.clamp_min(x, 0.0)


def tropical_leaky_relu(x: torch.Tensor, shift: int = 3) -> torch.Tensor:
    """Leaky ReLU with a power-of-2 negative slope: max(x, x >> shift).

    Slope = 2^-shift (e.g. shift=3 -> 0.125). The scaling is an arithmetic
    right-shift, so this is multiplier-free (unlike a float ``x * 0.1``).
    """
    return torch.maximum(x, x * (2.0 ** -shift))


def tropical_hardtanh(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """Hard tanh = clamp(x, lo, hi) = min(max(x, lo), hi). Pure compare."""
    return torch.clamp(x, lo, hi)


# ---------------------------------------------------------------------------
# Loss conditioning: the perceptron baseline and closure-faithful cross-entropy
# ---------------------------------------------------------------------------
#
# The tropical forward/backward above remove multipliers from the *network*.
# The loss that conditions the gradient is the last place a multiply can hide,
# and historically it did: every model here computes ``F.cross_entropy`` (a real
# softmax, with exp + divide + multiply). The zero-DSP silicon kernel cannot do
# that, so on chip the loop has used a *perceptron* gradient instead — a 2-hot
# vector, +1 at the argmax (predicted) logit and -1 at the target logit, zero
# when they coincide (see ``forge_train_step.cpp`` lines 268-277). That is
# strictly sign-valued (multiplier-free) but it is the single largest accuracy
# cost in the closure waterfall (~-17.5pp on Shakespeare char-LM) because it
# throws away all probability-mass information.
#
# ``tropical_cross_entropy`` closes that gap *without* a hardware multiply. The
# cross-entropy gradient w.r.t. the logits is exactly ``softmax(logits) -
# onehot(target)``; the only non-trivial piece is the softmax, and
# ``shift_softmax`` computes it from a subtract-max, a power-of-2-addressed exp
# lookup table, additions, and a shift-normalize — no multiply, no exp unit, no
# float divide on the shift-normalized path. Two backends:
#
#   * ``softmax="exact"`` — torch softmax. This is the HOST-CONDITIONED path:
#     the zero-DSP kernel stays multiplier-free and a tiny V-way softmax runs on
#     the host. It is the shipped, silicon-validated path (+3.2pp over the
#     perceptron baseline at the prototype's D=128/V=128 scale).
#   * ``softmax="lut"`` — ``shift_softmax``: a fully multiplier-free softmax, the
#     on-chip conditioning path, so the *entire* loop (loss included) is
#     multiplier-free at the cost of a small quantization approximation.
#
# The autograd contract (important): each loss Function returns the SCALAR loss
# from ``forward`` (a continuous value, for a readable loss curve) and the
# per-logit gradient from ``backward``. The continuous forward value and the
# custom backward gradient are intentionally decoupled (a straight-through
# pattern): the *gradient* is what conditions training and what the silicon
# emits, so it is the object we make multiplier-free. ``gradcheck`` therefore
# does NOT apply to these Functions (their forward is piecewise-constant in the
# LUT/argmax); use ``continuous_cross_entropy_ref`` for gradient checking and
# ``shift_softmax`` vs ``torch.softmax`` for the fixed-point softmax check.


def exp2_lut_table(lut_size: int = 4096, z_range: float = 16.0,
                   dtype: torch.dtype = torch.float32,
                   device=None) -> torch.Tensor:
    """Exp lookup table for :func:`shift_softmax`.

    ``table[i] = exp(-z_i)`` (natural base) for ``z_i = i * step`` with
    ``step = z_range / lut_size``, a uniform grid of non-negative magnitudes. The
    "2" in the name refers to the **power-of-two-stepped addressing** (with the
    default ``lut_size=4096, z_range=16`` the step is exactly ``2^-8``), NOT a
    base-2 exponent. On FPGA these are stored as fixed-point integers in a BRAM
    and addressed by a shifted logit (``(-z) >> z_shift``); here they are float
    for the CPU reference, but :func:`shift_softmax` addresses them by the same
    floor-of-shift index so the reference faithfully models the on-chip lookup.
    """
    step = z_range / lut_size
    i = torch.arange(lut_size, dtype=dtype, device=device)
    return torch.exp(-i * step)


def shift_softmax(logits: torch.Tensor, lut_size: int = 4096,
                  z_range: float = 16.0, normalize: str = "exact") -> torch.Tensor:
    """Multiplier-free softmax: subtract-max, exp LUT, shift-normalize.

    All ops map to comparators, adders, a BRAM exp LUT, and bit-shifts -- there
    is no multiplier and (on the ``normalize="shift"`` path) no float divide::

        z   = logits - max(logits)                 # <= 0, subtract only
        idx = clamp(round((-z) / step), 0, N-1)    # on chip: (-z) >> z_shift
        e   = exp_lut[idx]                          # BRAM lookup
        p   = e / sum(e)        (normalize="exact", host reciprocal)
            | e >> ceil(log2 sum(e))   (normalize="shift", on-chip power-of-2)

    ``step = z_range / lut_size`` is a genuine power of two with the default args
    (``2^-8``), so ``idx = floor((-z)/step)`` is exactly the shift-addressed index
    ``(-z) >> z_shift`` the hardware uses (floor, not round). The ``"exact"``
    normalization uses a true reciprocal (the cheap host-conditioned path);
    ``"shift"`` divides by the nearest power of two >= the sum, an arithmetic
    right-shift, so the whole softmax is multiplier-free at the cost of a per-row
    scale within [0.5, 1].

    Returns probabilities with the same shape as ``logits`` (softmax over dim=-1).
    """
    work = logits.float()
    z = work - work.max(dim=-1, keepdim=True).values        # <= 0
    step = z_range / lut_size
    idx = torch.clamp(((-z) / step).floor().long(), 0, lut_size - 1)
    lut = exp2_lut_table(lut_size, z_range, dtype=torch.float32, device=work.device)
    e = lut[idx]                                            # (..., V)
    s = e.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    if normalize == "exact":
        p = e / s
    elif normalize == "shift":
        shift = torch.ceil(torch.log2(s))
        p = e * torch.exp2(-shift)                          # models e >> shift
    else:
        raise ValueError(f"normalize must be 'exact' or 'shift', got {normalize!r}")
    return p.to(logits.dtype)


def _flatten_logits_target(logits: torch.Tensor, target: torch.Tensor):
    """Collapse (N, ..., V)/(N, ...) to (M, V)/(M,) like ``F.cross_entropy``."""
    if logits.dim() > 2:
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)
    return logits, target


class TropicalPerceptronLossFn(torch.autograd.Function):
    """Multiclass perceptron loss — the multiplier-free silicon baseline.

    ``forward`` returns the scalar margin ``mean(logit[argmax] - logit[target])``
    (>= 0, zero when the prediction is correct). ``backward`` returns the 2-hot
    gradient ``onehot(argmax) - onehot(target)`` averaged over the batch — the
    exact gradient the F2 kernel emits (``forge_train_step.cpp`` lines 268-277).
    Sign-valued, no multiply.
    """

    @staticmethod
    def forward(ctx, logits, target):
        logits2, target2 = _flatten_logits_target(logits, target)
        m = logits2.size(0)
        pred = logits2.argmax(dim=-1)
        rows = torch.arange(m, device=logits2.device)
        grad = torch.zeros_like(logits2)
        grad[rows, pred] += 1.0
        grad[rows, target2] -= 1.0          # +1 and -1 cancel to 0 when pred == target
        grad = grad / m
        ctx.save_for_backward(grad)
        ctx.in_shape = logits.shape
        margin = logits2[rows, pred] - logits2[rows, target2]
        return margin.mean()

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return (grad_output * grad).reshape(ctx.in_shape), None


def tropical_perceptron_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scalar multiclass perceptron loss with the 2-hot silicon gradient."""
    return TropicalPerceptronLossFn.apply(logits, target)


class TropicalCrossEntropyFn(torch.autograd.Function):
    """Closure-faithful cross-entropy: softmax conditioning, no hardware multiply.

    ``forward`` returns the scalar continuous cross-entropy (for a readable loss
    curve); ``backward`` returns ``(softmax(logits) - onehot(target)) / M`` where
    the softmax is either exact (host-conditioned) or :func:`shift_softmax` (fully
    multiplier-free, on-chip). The gradient -- not the forward value -- is the
    object that conditions training and that the kernel realizes.
    """

    @staticmethod
    def forward(ctx, logits, target, softmax, lut_kwargs):
        logits2, target2 = _flatten_logits_target(logits, target)
        m = logits2.size(0)
        if softmax == "exact":
            p = torch.softmax(logits2, dim=-1)
        elif softmax == "lut":
            p = shift_softmax(logits2, **(lut_kwargs or {}))
        else:
            raise ValueError(f"softmax must be 'exact' or 'lut', got {softmax!r}")
        onehot = torch.zeros_like(p)
        # Scatter the ROW SUM of p (not 1.0) at the target: sum(grad) == 0 per
        # row for ANY normalization. With exact softmax s == 1 (unchanged); with
        # shift_softmax(normalize="shift") s is in [0.5, 1] and an unbalanced
        # (p - onehot) gradient carries a systematic per-row bias that sign-based
        # optimizers integrate into a drift -> constant-predictor collapse
        # (measured: MorphSSM char-LM dies by step ~900). Placing the precomputed
        # s at the target index is a scatter, not a multiply -- closure-clean.
        onehot.scatter_(1, target2.view(-1, 1), p.sum(dim=-1, keepdim=True))
        grad = (p - onehot) / m
        ctx.save_for_backward(grad)
        ctx.in_shape = logits.shape
        log_z = torch.logsumexp(logits2, dim=-1)
        ce = (log_z - logits2.gather(1, target2.view(-1, 1)).squeeze(1)).mean()
        return ce

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return (grad_output * grad).reshape(ctx.in_shape), None, None, None


def tropical_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                           softmax: str = "exact", **lut_kwargs) -> torch.Tensor:
    """Closure-faithful cross-entropy loss.

    ``softmax="exact"`` host-conditions with a true softmax (the silicon-validated
    path); ``softmax="lut"`` uses :func:`shift_softmax` for a fully multiplier-free
    loop. Extra keyword args (``lut_size``, ``z_range``, ``normalize``) pass
    through to :func:`shift_softmax`.
    """
    return TropicalCrossEntropyFn.apply(logits, target, softmax, lut_kwargs)


def continuous_cross_entropy_ref(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Continuous cross-entropy (``logsumexp(logits) - logits[target]``), mean-reduced.

    Standard-autograd reference whose analytic gradient is ``softmax - onehot``.
    Use this (not the custom Functions, which are piecewise-constant) for
    ``torch.autograd.gradcheck``.
    """
    logits2, target2 = _flatten_logits_target(logits, target)
    log_z = torch.logsumexp(logits2, dim=-1)
    return (log_z - logits2.gather(1, target2.view(-1, 1)).squeeze(1)).mean()
