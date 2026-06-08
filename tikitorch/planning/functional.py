"""tikitorch.planning.functional — tropical dynamic-programming primitives.

This module adds the two stateless ops that the planning / RL layers are built
on, in the same multiplier-free spirit as :mod:`tikitorch.functional`:

  * :class:`TropicalScanFn` — the generic ``(max,+)`` / ``(min,+)`` *linear
    recurrence* (a DP scan) with argmax/argmin routing in the backward pass.
    Viterbi, DTW, edit distance, Bellman-Ford relaxation, and finite-horizon
    value iteration are all this one scan with different ``(T, emit, reduce)``
    wiring. It is the recurrent lift of :class:`tikitorch.functional.TropicalValuePullFn`.
  * :func:`max_plus_matmul` — the max-plus dual of
    :func:`tikitorch.functional.tropical_matmul` (which is min-plus and has *no*
    custom backward), with a sparse argmax backward. Repeated max-plus matmul is
    all-pairs longest/shortest path and max-plus matrix powers (value iteration).

Both reduce to comparisons and additions only — no multiplies — so they map
directly onto the inlined max-plus PE in the HLS kernels (the ``SCORE_N``
``max_d(Q+K)`` loop iterated over a step axis).
"""

import torch

from tikitorch.functional import tropical_softmax, tropical_softmin

__all__ = [
    "TropicalScanFn",
    "tropical_scan",
    "tropical_scan_soft",
    "tropical_scan_decode",
    "MaxPlusMatmulFn",
    "max_plus_matmul",
]


# ---------------------------------------------------------------------------
# Generic tropical scan (the DP engine)
# ---------------------------------------------------------------------------
#
# Recurrence (max-plus form):
#
#     h_t[i] = max_j ( h_{t-1}[j] + T[:, t-1, i, j] ) + emit[:, t-1, i]
#
# with ``reduce="min"`` giving the dual (min-plus). ``h0`` is the initial score
# vector; ``T`` is the per-step transition cost; ``emit`` is an optional per-step
# local cost. The forward stores, for each step and each target state ``i``, the
# winning source index ``j*`` (argmax/argmin). The backward is a reverse pass that
# routes the upstream gradient only through those stored winners -- one-hot
# scatter, exactly like the attention value-pull -- so it is multiply-free.


class TropicalScanFn(torch.autograd.Function):
    """Hard tropical linear-recurrence scan with argmax/argmin backward.

    Args (forward):
        h0:   (B, S)            initial scores over S states.
        T:    (B, L, S, S)      per-step transition: ``T[b, t, i, j]`` is the cost
                                added to source state ``j`` when stepping to target
                                state ``i`` at step ``t``. A time-invariant
                                transition ``(B, S, S)`` should be ``.expand``-ed to
                                ``(B, L, S, S)`` by the caller (autograd sums the
                                grad back through ``expand``).
        emit: (B, L, S) | None  optional per-step local cost added after the reduce.
        reduce: "max" | "min".

    Returns:
        H: (B, L+1, S) with ``H[:, 0] = h0`` and ``H[:, t] = h_t``.

    Backward routes ``grad_H`` through the stored winners only: for each step,
    ``grad`` on ``h_t[i]`` flows to ``h_{t-1}[j*]``, ``T[t-1, i, j*]`` and
    ``emit[t-1, i]`` where ``j* = argmax/argmin``. Add/compare only.
    """

    @staticmethod
    def forward(ctx, h0, T, emit, reduce):
        B, S = h0.shape
        L = T.shape[1]
        use_max = reduce == "max"
        H = h0.new_empty((B, L + 1, S))
        H[:, 0] = h0
        A = torch.empty((B, L, S), dtype=torch.long, device=h0.device)
        h_prev = h0
        for t in range(L):
            # cand[b, i, j] = h_prev[b, j] + T[b, t, i, j]
            cand = h_prev.unsqueeze(-2) + T[:, t]            # (B, S, S)
            if use_max:
                red, arg = cand.max(dim=-1)                  # (B, S)
            else:
                red, arg = cand.min(dim=-1)
            if emit is not None:
                red = red + emit[:, t]
            H[:, t + 1] = red
            A[:, t] = arg
            h_prev = red
        ctx.save_for_backward(A)
        ctx.h0_shape = h0.shape
        ctx.T_shape = T.shape
        ctx.has_emit = emit is not None
        ctx.emit_shape = emit.shape if emit is not None else None
        return H

    @staticmethod
    def backward(ctx, grad_H):
        (A,) = ctx.saved_tensors
        B, S = ctx.h0_shape
        L = ctx.T_shape[1]

        grad_T = torch.zeros(ctx.T_shape, device=grad_H.device, dtype=grad_H.dtype)
        grad_emit = (
            torch.zeros(ctx.emit_shape, device=grad_H.device, dtype=grad_H.dtype)
            if ctx.has_emit
            else None
        )

        # Accumulated gradient on H[:, t], processed in reverse.
        gh = grad_H[:, L].clone()                            # (B, S)
        for t in range(L, 0, -1):
            arg = A[:, t - 1]                                # (B, S): winner j for each i
            if grad_emit is not None:
                grad_emit[:, t - 1] += gh
            # grad to T[:, t-1, i, j*] : scatter gh[i] along the j axis at arg[i]
            grad_T[:, t - 1].scatter_add_(-1, arg.unsqueeze(-1), gh.unsqueeze(-1))
            # grad to h_{t-1}[j*] : accumulate gh[i] into bin arg[i]
            gh_prev = torch.zeros((B, S), device=grad_H.device, dtype=grad_H.dtype)
            gh_prev.scatter_add_(-1, arg, gh)
            gh = gh_prev + grad_H[:, t - 1]
        grad_h0 = gh
        return grad_h0, grad_T, grad_emit, None


def tropical_scan(h0, T, emit=None, reduce="max"):
    """Hard tropical DP scan. See :class:`TropicalScanFn`."""
    return TropicalScanFn.apply(h0, T, emit, reduce)


@torch.no_grad()
def tropical_scan_decode(h0, T, emit=None, reduce="max"):
    """Run the scan and return ``(H, backtrace)`` for path decoding (no grad).

    ``backtrace`` is ``(B, L, S)`` long: ``backtrace[b, t, i]`` is the source
    state ``j*`` chosen when stepping to state ``i`` at step ``t`` -- follow it in
    reverse from the terminal argmax to recover the optimal path (Viterbi/DP
    backtrace, shortest-path predecessors).
    """
    B, S = h0.shape
    L = T.shape[1]
    use_max = reduce == "max"
    H = h0.new_empty((B, L + 1, S))
    H[:, 0] = h0
    A = torch.empty((B, L, S), dtype=torch.long, device=h0.device)
    h_prev = h0
    for t in range(L):
        cand = h_prev.unsqueeze(-2) + T[:, t]
        red, arg = (cand.max(dim=-1) if use_max else cand.min(dim=-1))
        if emit is not None:
            red = red + emit[:, t]
        H[:, t + 1] = red
        A[:, t] = arg
        h_prev = red
    return H, A


def tropical_scan_soft(h0, T, emit=None, tau=1.0, reduce="max"):
    """Soft (logsumexp) DP scan -- dense-gradient, gradcheck-able relaxation.

    Approaches :func:`tropical_scan` as ``tau -> 0``. Built from
    :func:`tikitorch.functional.tropical_softmax` / ``tropical_softmin`` so it is
    still log/exp-on-host only (no closure claim; this is the training-warmup /
    verification path).
    """
    soft = tropical_softmax if reduce == "max" else tropical_softmin
    L = T.shape[1]
    h_prev = h0
    outs = [h0]
    for t in range(L):
        cand = h_prev.unsqueeze(-2) + T[:, t]                # (B, S, S)
        red = soft(cand, tau=tau, dim=-1)                    # (B, S)
        if emit is not None:
            red = red + emit[:, t]
        outs.append(red)
        h_prev = red
    return torch.stack(outs, dim=1)                          # (B, L+1, S)


# ---------------------------------------------------------------------------
# Max-plus matrix multiply (with sparse argmax backward)
# ---------------------------------------------------------------------------


class MaxPlusMatmulFn(torch.autograd.Function):
    """Max-plus matmul: ``C[..., i, j] = max_k (A[..., i, k] + B[..., k, j])``.

    The max-plus dual of :func:`tikitorch.functional.tropical_matmul` (min-plus).
    Unlike that helper it defines a custom backward that routes the gradient only
    through the winning ``k*`` -- a one-hot scatter, multiply-free -- and (in the
    decode path) exposes that index. Repeated application is all-pairs
    shortest/longest path and max-plus matrix powers (value iteration).
    """

    @staticmethod
    def forward(ctx, A, B):
        # expanded[..., i, k, j] = A[..., i, k] + B[..., k, j]
        expanded = A.unsqueeze(-1) + B.unsqueeze(-3)         # (..., m, K, n)
        out, idx = expanded.max(dim=-2)                      # (..., m, n), winner k
        ctx.save_for_backward(idx)
        ctx.a_shape = A.shape
        ctx.b_shape = B.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors                           # (..., m, n)
        grad_A = torch.zeros(ctx.a_shape, device=grad_out.device, dtype=grad_out.dtype)
        # grad_A[..., i, k*] += grad_out[..., i, j] : scatter along the k axis (-1)
        grad_A.scatter_add_(-1, idx, grad_out)
        grad_B = torch.zeros(ctx.b_shape, device=grad_out.device, dtype=grad_out.dtype)
        # grad_B[..., k*, j] += grad_out[..., i, j] : scatter along the k axis (-2)
        grad_B.scatter_add_(-2, idx, grad_out)
        return grad_A, grad_B


def max_plus_matmul(A, B):
    """Max-plus matmul ``C[i,j] = max_k(A[i,k] + B[k,j])``. See :class:`MaxPlusMatmulFn`."""
    return MaxPlusMatmulFn.apply(A, B)
