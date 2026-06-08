"""Tests for tikitorch.functional — the stateless tropical ops + autograd Fns.

Covers:
  * tropical_matmul correctness vs a brute-force min-plus reference,
  * forward correctness of the hard autograd ops (linear/dilation, attention),
  * gradcheck-lite: the soft (logsumexp) relaxation converges to the hard op as
    tau -> 0, and its gradients are finite/sparse-consistent,
  * torch.autograd.gradcheck on the soft relaxations (they are smooth, so this
    is a genuine numeric Jacobian check),
  * the hard custom-autograd backward routes gradient only through the winning
    argmin/argmax index.
"""
import pytest
import torch

import tikitorch.functional as F


# ---------------------------------------------------------------------------
# tropical_matmul
# ---------------------------------------------------------------------------


def _bruteforce_minplus(A, B):
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    C = torch.empty(m, n)
    for i in range(m):
        for j in range(n):
            C[i, j] = min(A[i, t] + B[t, j] for t in range(k))
    return C


def test_tropical_matmul_matches_bruteforce():
    A = torch.randn(4, 5)
    B = torch.randn(5, 3)
    out = F.tropical_matmul(A, B)
    ref = _bruteforce_minplus(A, B)
    assert out.shape == (4, 3)
    torch.testing.assert_close(out, ref)


def test_tropical_matmul_batched_shape():
    A = torch.randn(2, 4, 5)
    B = torch.randn(2, 5, 3)
    out = F.tropical_matmul(A, B)
    assert out.shape == (2, 4, 3)


def test_tropical_matmul_identity_element():
    # Tropical identity: zeros on diagonal, +inf off-diagonal, acts as identity.
    m = 4
    A = torch.randn(m, m)
    eye = torch.full((m, m), float("inf"))
    eye.fill_diagonal_(0.0)
    torch.testing.assert_close(F.tropical_matmul(A, eye), A)


# ---------------------------------------------------------------------------
# Hard tropical linear / dilation: forward correctness
# ---------------------------------------------------------------------------


def test_tropical_linear_forward():
    x = torch.randn(3, 5)
    w = torch.randn(4, 5)
    out = F.tropical_linear(x, w)
    # y[b,j] = min_i(w[j,i] + x[b,i])
    ref = (w.unsqueeze(0) + x.unsqueeze(1)).min(dim=-1).values
    assert out.shape == (3, 4)
    torch.testing.assert_close(out, ref)


def test_tropical_dilation_forward():
    x = torch.randn(3, 5)
    w = torch.randn(4, 5)
    out = F.tropical_dilation(x, w)
    ref = (w.unsqueeze(0) + x.unsqueeze(1)).max(dim=-1).values
    torch.testing.assert_close(out, ref)


# ---------------------------------------------------------------------------
# Hard backward: gradient routes only through the argmin/argmax index
# ---------------------------------------------------------------------------


def test_tropical_linear_backward_is_sparse():
    batch, in_f, out_f = 2, 6, 3
    x = torch.randn(batch, in_f, requires_grad=True)
    w = torch.randn(out_f, in_f, requires_grad=True)
    out = F.tropical_linear(x, w)
    out.sum().backward()
    # Each (batch, output) pair selects exactly one input index, and grad_w is
    # summed over the batch. So per output row the weight grad hits at most
    # `batch` distinct indices, and the whole grad_w has <= out_f*batch nnz.
    nnz_per_row = (w.grad != 0).sum(dim=-1)
    assert (nnz_per_row <= batch).all(), "min-plus weight grad: <= batch indices per output row"
    assert (w.grad != 0).sum() <= out_f * batch
    # grad_x: each output contributes to exactly one input -> bounded nnz.
    assert (x.grad != 0).sum() <= out.numel()


def test_tropical_dilation_backward_is_sparse():
    batch, in_f, out_f = 2, 6, 3
    x = torch.randn(batch, in_f, requires_grad=True)
    w = torch.randn(out_f, in_f, requires_grad=True)
    out = F.tropical_dilation(x, w)
    out.sum().backward()
    nnz_per_row = (w.grad != 0).sum(dim=-1)
    assert (nnz_per_row <= batch).all()
    assert (w.grad != 0).sum() <= out_f * batch


# ---------------------------------------------------------------------------
# Soft relaxation -> hard op as tau -> 0  (gradcheck-lite)
# ---------------------------------------------------------------------------


def test_softmin_approaches_hard_min():
    z = torch.randn(8, 10)
    hard = z.min(dim=-1).values
    soft_hi = F.tropical_softmin(z, tau=1.0, dim=-1)
    soft_lo = F.tropical_softmin(z, tau=1e-3, dim=-1)
    # softmin is a smooth lower bound that tightens to hard min as tau -> 0.
    assert (soft_hi <= hard + 1e-5).all()
    assert torch.allclose(soft_lo, hard, atol=1e-2)
    # tighter tau -> closer to hard.
    assert (soft_lo - hard).abs().max() < (soft_hi - hard).abs().max()


def test_softmax_approaches_hard_max():
    z = torch.randn(8, 10)
    hard = z.max(dim=-1).values
    soft_lo = F.tropical_softmax(z, tau=1e-3, dim=-1)
    assert torch.allclose(soft_lo, hard, atol=1e-2)


def test_soft_linear_approaches_hard_linear():
    x = torch.randn(4, 7)
    w = torch.randn(5, 7)
    hard = F.tropical_linear(x, w)
    expanded = w.unsqueeze(0) + x.unsqueeze(1)
    soft = F.tropical_softmin(expanded, tau=1e-3, dim=-1)
    torch.testing.assert_close(soft, hard, atol=2e-2, rtol=0)


# ---------------------------------------------------------------------------
# True numeric gradcheck on the smooth soft relaxations
# ---------------------------------------------------------------------------


def test_gradcheck_softmin():
    z = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: F.tropical_softmin(t, tau=0.7, dim=-1), (z,), eps=1e-6, atol=1e-4
    )


def test_gradcheck_softmax():
    z = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: F.tropical_softmax(t, tau=0.7, dim=-1), (z,), eps=1e-6, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Attention primitives: shapes, forward correctness, soft->hard limit
# ---------------------------------------------------------------------------


def test_dot_product_scores_shape_and_value():
    Q = torch.randn(2, 5, 8)
    K = torch.randn(2, 6, 8)
    scores = F.tropical_dot_product(Q, K)
    assert scores.shape == (2, 5, 6)
    ref = (Q.unsqueeze(-2) + K.unsqueeze(-3)).max(dim=-1).values
    torch.testing.assert_close(scores, ref)


def test_soft_dot_product_approaches_hard():
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    hard = F.tropical_dot_product(Q, K)
    soft = F.tropical_softmax_dot_product(Q, K, tau=1e-3)
    torch.testing.assert_close(soft, hard, atol=2e-2, rtol=0)


def test_l_inf_score_is_symmetric_and_nonpositive():
    Q = torch.randn(3, 5, 6)
    K = torch.randn(3, 5, 6)
    s = F.tropical_l_inf_score(Q, K)
    assert s.shape == (3, 5, 5)
    assert (s <= 1e-6).all(), "L_inf score = -max|.| must be <= 0"
    # Self-scoring: -max|Q-Q| == 0 on the diagonal.
    s_self = F.tropical_l_inf_score(Q, Q)
    diag = torch.diagonal(s_self, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.zeros_like(diag))


def test_value_pull_shape_and_value():
    V = torch.randn(2, 6, 8)
    scores = torch.randn(2, 5, 6)
    out = F.tropical_value_pull(V, scores)
    assert out.shape == (2, 5, 8)
    ref = (V.unsqueeze(-3) + scores.unsqueeze(-1)).max(dim=-2).values
    torch.testing.assert_close(out, ref)


def test_value_pull_backward_finite():
    V = torch.randn(2, 6, 8, requires_grad=True)
    scores = torch.randn(2, 5, 6, requires_grad=True)
    out = F.tropical_value_pull(V, scores)
    out.sum().backward()
    assert torch.isfinite(V.grad).all()
    assert torch.isfinite(scores.grad).all()


def test_attention_compose_end_to_end_grad():
    # scores = dot(Q,K) ; out = value_pull(V, scores). Full path differentiates.
    Q = torch.randn(1, 4, 8, requires_grad=True)
    K = torch.randn(1, 4, 8, requires_grad=True)
    V = torch.randn(1, 4, 8, requires_grad=True)
    scores = F.tropical_dot_product(Q, K)
    out = F.tropical_value_pull(V, scores)
    out.pow(2).sum().backward()
    for t in (Q, K, V):
        assert t.grad is not None and torch.isfinite(t.grad).all()


# ---------------------------------------------------------------------------
# Loss conditioning: perceptron baseline + closure-faithful cross-entropy
# ---------------------------------------------------------------------------


def test_perceptron_grad_is_two_hot_and_matches_silicon():
    # The silicon kernel emits grad = onehot(argmax) - onehot(target) (2-hot,
    # sign-valued). Verify the Python baseline produces exactly that.
    torch.manual_seed(0)
    n, v = 16, 32
    logits = torch.randn(n, v, requires_grad=True)
    target = torch.randint(0, v, (n,))
    F.tropical_perceptron_loss(logits, target).backward()
    g = logits.grad
    # at most 2 nonzeros per row (1 when argmax == target -> +1 and -1 cancel)
    nnz = (g != 0).sum(dim=-1)
    assert (nnz <= 2).all()
    # values are exactly +-1/n (sign-valued, multiplier-free)
    nz = g[g != 0]
    assert torch.allclose(nz.abs(), torch.full_like(nz, 1.0 / n))
    pred = logits.detach().argmax(-1)
    oh_pred = torch.zeros(n, v).scatter_(1, pred.view(-1, 1), 1.0)
    oh_tgt = torch.zeros(n, v).scatter_(1, target.view(-1, 1), 1.0)
    torch.testing.assert_close(g, (oh_pred - oh_tgt) / n)


def test_perceptron_grad_zero_when_correct():
    # Logits whose argmax already equals the target -> zero gradient (no update).
    v = 10
    logits = torch.full((1, v), -1.0)
    logits[0, 3] = 5.0
    logits.requires_grad_(True)
    target = torch.tensor([3])
    F.tropical_perceptron_loss(logits, target).backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_exact_ce_grad_equals_softmax_minus_onehot():
    torch.manual_seed(1)
    n, v = 8, 64
    logits = torch.randn(n, v, requires_grad=True)
    target = torch.randint(0, v, (n,))
    F.tropical_cross_entropy(logits, target, softmax="exact").backward()
    oh = torch.zeros(n, v).scatter_(1, target.view(-1, 1), 1.0)
    ref = (torch.softmax(logits.detach(), dim=-1) - oh) / n
    torch.testing.assert_close(logits.grad, ref)


def test_shift_softmax_matches_torch_softmax():
    # Multiplier-free softmax (subtract-max + exp LUT + exact normalize) must
    # track torch.softmax closely; this substantiates the fixed-point path.
    torch.manual_seed(2)
    logits = torch.randn(32, 128) * 3.0
    p = F.shift_softmax(logits, normalize="exact")
    ref = torch.softmax(logits, dim=-1)
    assert (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(32), atol=1e-5, rtol=0)
    assert (p - ref).abs().max() < 2e-3


def test_shift_softmax_shift_normalize_in_unit_band():
    # The on-chip shift-normalize divides by the nearest power of two >= sum,
    # so each row's mass lands in [0.5, 1] (no divide, just an arithmetic shift).
    torch.manual_seed(3)
    logits = torch.randn(16, 128) * 2.0
    p = F.shift_softmax(logits, normalize="shift")
    s = p.sum(-1)
    assert (s > 0.5 - 1e-4).all() and (s <= 1.0 + 1e-4).all()


def test_lut_ce_grad_close_to_exact_ce_grad():
    torch.manual_seed(4)
    n, v = 8, 128
    logits = torch.randn(n, v) * 3.0
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    F.tropical_cross_entropy(a, torch.zeros(n, dtype=torch.long) + 1, softmax="exact").backward()
    F.tropical_cross_entropy(b, torch.zeros(n, dtype=torch.long) + 1, softmax="lut").backward()
    assert (a.grad - b.grad).abs().max() < 5e-3


def test_gradcheck_continuous_ce_ref():
    logits = torch.randn(4, 16, dtype=torch.double, requires_grad=True)
    target = torch.randint(0, 16, (4,))
    assert torch.autograd.gradcheck(
        lambda l: F.continuous_cross_entropy_ref(l, target), (logits,), eps=1e-6, atol=1e-4
    )


def test_losses_accept_3d_logits():
    # (B, L, V) logits with (B, L) targets, like an LM batch.
    b, ll, v = 3, 7, 32
    logits = torch.randn(b, ll, v, requires_grad=True)
    target = torch.randint(0, v, (b, ll))
    for fn in (
        lambda: F.tropical_perceptron_loss(logits, target),
        lambda: F.tropical_cross_entropy(logits, target, softmax="exact"),
        lambda: F.tropical_cross_entropy(logits, target, softmax="lut"),
    ):
        logits.grad = None
        loss = fn()
        assert loss.dim() == 0
        loss.backward()
        assert logits.grad.shape == logits.shape and torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# Top-k softened value pull
# ---------------------------------------------------------------------------


def test_value_pull_topk_k1_equals_hard():
    torch.manual_seed(0)
    Vt = torch.randn(2, 3, 6, 8)
    scores = torch.randn(2, 3, 5, 6)
    torch.testing.assert_close(
        F.tropical_value_pull_topk(Vt, scores, 1), F.tropical_value_pull(Vt, scores)
    )


def test_value_pull_topk_density_rises_monotonically():
    torch.manual_seed(1)
    densities = []
    for k in (1, 2, 3, 4):
        Vt = torch.randn(2, 2, 8, 8, requires_grad=True)
        scores = torch.randn(2, 2, 6, 8, requires_grad=True)
        F.tropical_value_pull_topk(Vt, scores, k).sum().backward()
        densities.append((scores.grad != 0).float().mean().item())
    assert all(b >= a for a, b in zip(densities, densities[1:])), densities
    assert densities[-1] > densities[0]


def test_value_pull_topk_backward_accumulates_across_ranks_and_dims():
    # A key winning for several output dims / ranks must SUM (scatter_add_),
    # not overwrite. Brute-force reference on a tiny case.
    torch.manual_seed(2)
    Lv, D, Lq, k = 4, 3, 2, 2
    Vt = torch.randn(Lv, D).unsqueeze(0).unsqueeze(0).requires_grad_(True)  # (1,1,Lv,D)
    scores = torch.randn(Lq, Lv).unsqueeze(0).unsqueeze(0).requires_grad_(True)
    out = F.tropical_value_pull_topk(Vt, scores, k)
    out.sum().backward()
    # Reference: recompute with explicit loops.
    exp = Vt.detach().unsqueeze(-3) + scores.detach().unsqueeze(-1)  # (1,1,Lq,Lv,D)
    order = torch.argsort(exp, dim=-2, descending=True, stable=True)[..., :k, :]
    gV = torch.zeros_like(Vt)
    gS = torch.zeros_like(scores)
    for i in range(Lq):
        for d in range(D):
            for r in range(k):
                j = order[0, 0, i, r, d].item()
                gV[0, 0, j, d] += 1.0 / k
                gS[0, 0, i, j] += 1.0 / k
    torch.testing.assert_close(Vt.grad, gV)
    torch.testing.assert_close(scores.grad, gS)


def test_value_pull_topk_causal_mask_is_finite():
    torch.manual_seed(3)
    L = 6
    Vt = torch.randn(1, 1, L, 8, requires_grad=True)
    scores = torch.randn(1, 1, L, L)
    mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf")).requires_grad_(True)
    out = F.tropical_value_pull_topk(Vt, scores, 3)  # k > valid keys at early i
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(Vt.grad).all()
    # no gradient leaks onto masked (upper-triangular) score positions
    assert (scores.grad[mask.unsqueeze(0).unsqueeze(0).expand_as(scores.grad)] == 0).all()


def test_value_pull_topk_tie_break_lower_index():
    # All keys equal -> top-2 must be indices 0 and 1 (stable, first-wins).
    Vt = torch.zeros(1, 1, 4, 1, requires_grad=True)
    scores = torch.zeros(1, 1, 1, 4)
    F.tropical_value_pull_topk(Vt, scores, 2).sum().backward()
    winners = (Vt.grad.squeeze() != 0).tolist()
    assert winners == [True, True, False, False]


def test_attention_value_topk_knob_runs():
    from tikitorch.nn import TropicalAttention
    attn = TropicalAttention(d_model=16, n_heads=2, tau=0.0, value_topk_k=3)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = attn(x, causal_mask=True)
    y.sum().backward()
    assert y.shape == (2, 5, 16) and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Morphological 2D convolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dilation", "erosion"])
def test_conv2d_forward_matches_bruteforce(mode):
    torch.manual_seed(0)
    B, Cin, H, W = 2, 3, 6, 7
    Cout, Kh, Kw = 4, 3, 3
    s, p, d = 2, 1, 1
    x = torch.randn(B, Cin, H, W)
    w = torch.randn(Cout, Cin, Kh, Kw)
    y = F.tropical_conv2d(x, w, stride=s, padding=p, dilation=d, mode=mode)
    Hout = (H + 2 * p - d * (Kh - 1) - 1) // s + 1
    Wout = (W + 2 * p - d * (Kw - 1) - 1) // s + 1
    assert y.shape == (B, Cout, Hout, Wout)
    xp = torch.nn.functional.pad(x, (p, p, p, p), value=0.0)  # unfold zero-pads
    reduce = max if mode == "dilation" else min
    for b in range(B):
        for co in range(Cout):
            for oh in range(Hout):
                for ow in range(Wout):
                    vals = [w[co, ci, kh, kw].item() + xp[b, ci, oh * s + kh * d, ow * s + kw * d].item()
                            for ci in range(Cin) for kh in range(Kh) for kw in range(Kw)]
                    assert abs(y[b, co, oh, ow].item() - reduce(vals)) < 1e-4


def test_conv2d_backward_is_one_hot_per_output():
    # grad_output=1 on every output -> each output routes +1 to exactly one tap,
    # so sum(grad_weight) == number of outputs (sparse, sign-only, no scaling).
    x = torch.randn(1, 2, 5, 5, requires_grad=True)
    w = torch.randn(3, 2, 3, 3, requires_grad=True)
    y = F.tropical_conv2d(x, w, stride=1, padding=0, mode="dilation")
    y.sum().backward()
    assert int(round(w.grad.sum().item())) == y.numel()
    assert torch.isfinite(x.grad).all() and x.grad.shape == x.shape


def test_conv2d_gradient_finite_strided_dilated():
    x = torch.randn(2, 4, 12, 12, requires_grad=True)
    w = torch.randn(6, 4, 3, 3, requires_grad=True)
    F.tropical_conv2d(x, w, stride=2, padding=1, dilation=2, mode="erosion").pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


# ---------------------------------------------------------------------------
# Tropical activations (multiplier-free)
# ---------------------------------------------------------------------------


def test_tropical_activations():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert torch.equal(F.tropical_relu(x), torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    # leaky with power-of-2 slope 2^-3 = 0.125
    torch.testing.assert_close(F.tropical_leaky_relu(x, 3),
                               torch.tensor([-0.25, -0.125, 0.0, 1.0, 2.0]))
    assert torch.equal(F.tropical_hardtanh(x, -1.0, 1.0),
                       torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Hilbert / tropical projective metric scoring (Hashemi et al., NeurIPS 2025)
# ---------------------------------------------------------------------------


def test_hilbert_score_forward_and_projective():
    torch.manual_seed(0)
    Q = torch.randn(2, 4, 8); K = torch.randn(2, 5, 8)
    diff = Q.unsqueeze(-2) - K.unsqueeze(-3)
    ref = diff.min(-1).values - diff.max(-1).values        # -(max - min)
    s = F.tropical_hilbert_score(Q, K)
    assert s.shape == (2, 4, 5)
    torch.testing.assert_close(s, ref)
    # projective: translating a query row by a constant leaves scores unchanged
    Q2 = Q.clone(); Q2[0, 0, :] += 3.7
    torch.testing.assert_close(F.tropical_hilbert_score(Q, K),
                               F.tropical_hilbert_score(Q2, K))


def test_hilbert_score_backward_matches_autograd():
    torch.manual_seed(1)
    Qa = torch.randn(2, 4, 6, requires_grad=True); Ka = torch.randn(2, 3, 6, requires_grad=True)
    F.tropical_hilbert_score(Qa, Ka).pow(2).sum().backward()
    Qb = Qa.detach().clone().requires_grad_(True); Kb = Ka.detach().clone().requires_grad_(True)
    d = Qb.unsqueeze(-2) - Kb.unsqueeze(-3)
    (d.min(-1).values - d.max(-1).values).pow(2).sum().backward()
    torch.testing.assert_close(Qa.grad, Qb.grad)
    torch.testing.assert_close(Ka.grad, Kb.grad)


def test_soft_hilbert_approaches_hard():
    torch.manual_seed(2)
    Q = torch.randn(2, 4, 8); K = torch.randn(2, 4, 8)
    hard = F.tropical_hilbert_score(Q, K)
    soft = F.tropical_softmax_hilbert_score(Q, K, tau=1e-3)
    torch.testing.assert_close(soft, hard, atol=3e-2, rtol=0)


def test_attention_hilbert_variant_runs_and_closed():
    from tikitorch.nn import TropicalAttention
    from tikitorch.audit import find_closure_violations
    a = TropicalAttention(d_model=16, n_heads=2, tau=0.0, score_variant="hilbert")
    assert find_closure_violations(a) == []
    x = torch.randn(2, 5, 16, requires_grad=True)
    a(x, causal_mask=True).sum().backward()
    assert torch.isfinite(x.grad).all()


# --- iterative-argmax winner-exclusion top-k value pull (strictly closed) ---

def test_iter_topk_hard_is_sum_of_topk_and_k1_is_value_pull():
    torch.manual_seed(0)
    V = torch.randn(2, 3, 5, 4); scores = torch.randn(2, 3, 6, 5)
    e = V.unsqueeze(-3) + scores.unsqueeze(-1)
    ref = e.topk(2, dim=-2).values.sum(-2)
    torch.testing.assert_close(F.tropical_value_pull_iter_topk(V, scores, k=2), ref)
    torch.testing.assert_close(F.tropical_value_pull_iter_topk(V, scores, k=1),
                               F.tropical_value_pull(V, scores))


def test_iter_topk_soft_anneals_to_hard():
    torch.manual_seed(0)
    V = torch.randn(2, 3, 5, 4); scores = torch.randn(2, 3, 6, 5)
    hard = F.tropical_value_pull_iter_topk(V, scores, k=3)
    prev = float("inf")
    for tau in (1.0, 1e-1, 1e-2, 1e-3):
        d = (F.tropical_softmax_value_pull_iter_topk(V, scores, k=3, tau=tau) - hard).abs().max().item()
        prev = d
    assert prev < 0.1                                       # converges to the hard sum


def test_iter_topk_gradcheck_soft():
    Vd = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
    Sd = torch.randn(2, 5, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v, s: F.tropical_softmax_value_pull_iter_topk(v, s, k=2, tau=0.5),
        (Vd, Sd), atol=1e-4)


def test_iter_topk_rank_bias_gradient():
    torch.manual_seed(0)
    V = torch.randn(2, 5, 4, requires_grad=True)
    scores = torch.randn(2, 6, 5, requires_grad=True)
    rb = torch.zeros(3, requires_grad=True)
    out = F.tropical_value_pull_iter_topk(V, scores, k=3, rank_bias=rb)
    out.sum().backward()
    assert rb.grad is not None and rb.grad.shape == (3,)
    assert torch.isfinite(V.grad).all() and torch.isfinite(scores.grad).all()


def test_iter_topk_gradient_is_denser_than_argmax():
    torch.manual_seed(0)
    V = torch.randn(1, 6, 1)
    s1 = torch.randn(1, 1, 6, requires_grad=True)
    F.tropical_value_pull(V, s1).sum().backward()
    s3 = s1.detach().clone().requires_grad_(True)
    F.tropical_value_pull_iter_topk(V, s3, k=3).sum().backward()
    assert (s3.grad != 0).sum() > (s1.grad != 0).sum()     # k winners vs 1


def test_iter_topk_fully_masked_row_is_neg_inf():
    V = torch.zeros(1, 4, 2)
    scores = torch.full((1, 1, 4), float("-inf"))
    out = F.tropical_value_pull_iter_topk(V, scores, k=2)
    assert torch.isneginf(out).all()                       # consistent with hard/soft pulls


# --- min-plus (erosion) value pull: the (min,+) twin of the max-plus pull ----

def test_min_value_pull_is_min_over_keys():
    torch.manual_seed(0)
    V = torch.randn(2, 5, 4); scores = torch.randn(2, 6, 5)
    ref = (V.unsqueeze(-3) + scores.unsqueeze(-1)).min(dim=-2).values
    torch.testing.assert_close(F.tropical_value_pull_min(V, scores), ref)


def test_soft_min_value_pull_anneals_to_hard():
    torch.manual_seed(0)
    V = torch.randn(2, 5, 4); scores = torch.randn(2, 6, 5)
    hard = F.tropical_value_pull_min(V, scores)
    soft = F.tropical_softmax_value_pull_min(V, scores, tau=1e-3)
    torch.testing.assert_close(soft, hard, atol=2e-2, rtol=0)


def test_min_value_pull_gradcheck():
    Vd = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
    Sd = torch.randn(2, 5, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v, s: F.tropical_softmax_value_pull_min(v, s, tau=0.5), (Vd, Sd), atol=1e-4)


def test_min_value_pull_masks_with_plus_inf():
    # For MIN-plus, masked keys must be +inf (min ignores them) -- the opposite of
    # max-plus's -inf. (The attention layer applies the right sign per semiring.)
    V = torch.randn(1, 4, 2)
    scores = torch.tensor([[[0., float("inf"), float("inf"), float("inf")]]])  # only key 0 valid
    out = F.tropical_value_pull_min(V, scores)
    torch.testing.assert_close(out[0, 0], V[0, 0])     # min over the single valid key
