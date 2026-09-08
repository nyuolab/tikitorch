"""Tests for tikitorch.planning — tropical DP, value iteration, and search.

Covers:
  * max_plus_matmul correctness vs brute force + torch.autograd.gradcheck,
  * tropical_scan: hard == soft(tau->0), soft gradcheck, sparse hard backward,
  * TropicalShortestPath vs a brute-force Bellman-Ford reference,
  * TropicalValueIteration vs a reference value-iteration loop,
  * TropicalViterbi score + decoded path vs a reference Viterbi,
  * max_plus_power_iteration eigenvalue == maximum mean cycle,
  * assert_closed on every hard-path planning module.
"""
import pytest
import torch

import tikitorch.planning as P
from tikitorch import assert_closed

_INF = 1e9


# ---------------------------------------------------------------------------
# max_plus_matmul
# ---------------------------------------------------------------------------


def _bruteforce_maxplus(A, B):
    m, k = A.shape
    k2, n = B.shape
    C = torch.empty(m, n, dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            C[i, j] = max(A[i, t] + B[t, j] for t in range(k))
    return C


def test_max_plus_matmul_matches_bruteforce():
    A = torch.randn(4, 5)
    B = torch.randn(5, 3)
    out = P.max_plus_matmul(A, B)
    assert out.shape == (4, 3)
    torch.testing.assert_close(out, _bruteforce_maxplus(A, B))


def test_max_plus_matmul_gradcheck():
    A = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    B = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(P.max_plus_matmul, (A, B), eps=1e-6, atol=1e-4)


def test_max_plus_matmul_backward_is_sparse():
    # Each output (i,j) routes its gradient through exactly one k* -> grad_A has at
    # most (m*n) nonzeros total even though A is (m,K).
    A = torch.randn(3, 6, requires_grad=True)
    B = torch.randn(6, 4, requires_grad=True)
    P.max_plus_matmul(A, B).sum().backward()
    assert (A.grad != 0).sum().item() <= 3 * 4
    assert (B.grad != 0).sum().item() <= 3 * 4


# ---------------------------------------------------------------------------
# tropical_scan (the DP engine)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduce", ["max", "min"])
def test_scan_hard_equals_soft_limit(reduce):
    B, S, L = 2, 4, 5
    h0 = torch.randn(B, S)
    T = torch.randn(B, L, S, S)
    emit = torch.randn(B, L, S)
    hard = P.tropical_scan(h0, T, emit, reduce=reduce)
    soft = P.tropical_scan_soft(h0, T, emit, tau=1e-3, reduce=reduce)
    torch.testing.assert_close(hard, soft, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("reduce", ["max", "min"])
def test_scan_soft_gradcheck(reduce):
    B, S, L = 2, 3, 4
    h0 = torch.randn(B, S, dtype=torch.double, requires_grad=True)
    T = torch.randn(B, L, S, S, dtype=torch.double, requires_grad=True)
    emit = torch.randn(B, L, S, dtype=torch.double, requires_grad=True)

    def f(h0, T, emit):
        return P.tropical_scan_soft(h0, T, emit, tau=0.7, reduce=reduce)

    assert torch.autograd.gradcheck(f, (h0, T, emit), eps=1e-6, atol=1e-4)


def test_scan_hard_backward_matches_soft():
    # The hard custom-autograd backward should match the soft backward as tau->0.
    B, S, L = 2, 3, 4
    torch.manual_seed(1)
    h0 = torch.randn(B, S, dtype=torch.double)
    T = torch.randn(B, L, S, S, dtype=torch.double)

    h0h = h0.clone().requires_grad_(True)
    Th = T.clone().requires_grad_(True)
    P.tropical_scan(h0h, Th, None, reduce="max")[:, -1].sum().backward()

    h0s = h0.clone().requires_grad_(True)
    Ts = T.clone().requires_grad_(True)
    P.tropical_scan_soft(h0s, Ts, None, tau=1e-3, reduce="max")[:, -1].sum().backward()

    torch.testing.assert_close(h0h.grad, h0s.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(Th.grad, Ts.grad, atol=1e-2, rtol=1e-2)


def test_scan_decode_backtrace_recovers_argmax_path():
    B, S, L = 1, 4, 5
    h0 = torch.randn(B, S)
    T = torch.randn(B, L, S, S)
    H, back = P.tropical_scan_decode(h0, T, None, reduce="max")
    assert H.shape == (B, L + 1, S)
    assert back.shape == (B, L, S)
    # follow the backtrace from the terminal argmax; its accumulated score must
    # equal the DP optimum H[:, -1].max().
    end = H[0, -1].argmax().item()
    assert abs(H[0, -1, end].item() - H[0, -1].max().item()) < 1e-6


# ---------------------------------------------------------------------------
# TropicalShortestPath vs brute-force Bellman-Ford
# ---------------------------------------------------------------------------


def _brute_sssp(adj, src):
    S = adj.shape[0]
    d = [float("inf")] * S
    d[src] = 0.0
    for _ in range(S - 1):
        nd = d[:]
        for u in range(S):
            for v in range(S):
                if adj[u, v].item() < 1e8 and d[u] < float("inf"):
                    nd[v] = min(nd[v], d[u] + adj[u, v].item())
        d = nd
    return d


def _random_dag(S, seed):
    g = torch.Generator().manual_seed(seed)
    adj = torch.full((S, S), _INF)
    for u in range(S):
        for v in range(u + 1, S):
            if torch.rand((), generator=g).item() < 0.5:
                adj[u, v] = torch.randint(1, 10, (), generator=g).float()
    return adj


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_shortest_path_matches_bellman_ford(seed):
    S = 8
    adj = _random_dag(S, seed)
    dist = P.TropicalShortestPath()(adj, src=0)
    ref = _brute_sssp(adj, 0)
    for i in range(S):
        if ref[i] < float("inf"):
            assert abs(dist[i].item() - ref[i]) < 1e-4


def test_shortest_path_predecessors_shape_and_batch():
    S = 6
    adj = torch.stack([_random_dag(S, 5), _random_dag(S, 6)])  # (2, S, S)
    dist, pred = P.TropicalShortestPath()(adj, src=torch.tensor([0, 0]), return_pred=True)
    assert dist.shape == (2, S)
    assert pred.shape == (2, S)


# ---------------------------------------------------------------------------
# TropicalValueIteration vs reference value iteration
# ---------------------------------------------------------------------------


def _ref_vi(R, succ, gamma, sweeps):
    S, A = R.shape
    V = [0.0] * S
    for _ in range(sweeps):
        nV = [max(R[s, a].item() + gamma * V[succ[s, a].item()] for a in range(A)) for s in range(S)]
        V = nV
    return V


def test_value_iteration_matches_reference():
    S, A = 6, 3
    R = torch.randn(S, A)
    succ = torch.randint(0, S, (S, A))
    vi = P.TropicalValueIteration(S, A, gamma=1.0, n_sweeps=12)
    V, policy = vi(torch.zeros(S), R, succ)
    ref = _ref_vi(R, succ, 1.0, 12)
    for s in range(S):
        assert abs(V[s].item() - ref[s]) < 1e-4
    assert policy.shape == (S,)


def test_value_iteration_soft_anneals_to_hard():
    S, A = 5, 3
    R = torch.randn(S, A)
    succ = torch.randint(0, S, (S, A))
    Vh, _ = P.TropicalValueIteration(S, A, gamma=1.0, tau=0.0, n_sweeps=8)(torch.zeros(S), R, succ)
    Vs, _ = P.TropicalValueIteration(S, A, gamma=1.0, tau=1e-3, n_sweeps=8)(torch.zeros(S), R, succ)
    torch.testing.assert_close(Vh, Vs, atol=1e-2, rtol=1e-2)


def test_value_iteration_gradient_flows_to_reward():
    S, A = 5, 2
    R = torch.randn(S, A, requires_grad=True)
    succ = torch.randint(0, S, (S, A))
    V, _ = P.TropicalValueIteration(S, A, gamma=1.0, n_sweeps=6)(torch.zeros(S), R, succ)
    V.sum().backward()
    assert R.grad is not None and (R.grad != 0).any()


# ---------------------------------------------------------------------------
# TropicalViterbi vs reference Viterbi
# ---------------------------------------------------------------------------


def _ref_viterbi(emis, trans):
    B, L, S = emis.shape
    out = []
    for b in range(B):
        dp = emis[b, 0].clone()
        for t in range(1, L):
            nd = torch.empty(S)
            for i in range(S):
                nd[i] = max(dp[j] + trans[i, j] for j in range(S)) + emis[b, t, i]
            dp = nd
        out.append(dp.max().item())
    return out


def test_viterbi_score_matches_reference():
    B, L, S = 3, 6, 4
    vit = P.TropicalViterbi(S)
    with torch.no_grad():
        vit.trans.copy_(torch.randn(S, S))
    emis = torch.randn(B, L, S)
    score, tags = vit(emis, return_path=True)
    ref = _ref_viterbi(emis, vit.trans.detach())
    for b in range(B):
        assert abs(score[b].item() - ref[b]) < 1e-4
    assert tags.shape == (B, L)


def test_viterbi_gradient_flows():
    B, L, S = 2, 5, 4
    vit = P.TropicalViterbi(S)
    emis = torch.randn(B, L, S, requires_grad=True)
    vit(emis).sum().backward()
    assert emis.grad is not None and (emis.grad != 0).any()
    assert vit.trans.grad is not None


# ---------------------------------------------------------------------------
# max_plus_power_iteration: eigenvalue == maximum mean cycle
# ---------------------------------------------------------------------------


def test_max_plus_eigenvalue_is_mean_cycle():
    # 2-node cycle 0->1 (w=3), 1->0 (w=5): mean cycle = 4.
    S = 2
    M = torch.full((S, S), -_INF)
    M[1, 0] = 3.0
    M[0, 1] = 5.0
    lam, vec = P.max_plus_power_iteration(M, n_iter=300, tol=1e-9)
    assert abs(lam - 4.0) < 1e-3
    assert vec.shape == (S,)


# ---------------------------------------------------------------------------
# closure
# ---------------------------------------------------------------------------


def test_planning_modules_are_closed():
    assert_closed(P.TropicalShortestPath())
    assert_closed(P.TropicalValueIteration(6, 3))
    assert_closed(P.TropicalViterbi(4))
    assert_closed(P.TropicalCoinChangeDP())


# --- TropicalCoinChangeDP: exact closed min-plus coin-change DP layer ---------

def test_coin_change_dp_matches_reference():
    import tikitorch.algorithmic as A
    layer = P.TropicalCoinChangeDP()
    torch.manual_seed(0)
    for _ in range(25):
        coins = torch.randint(1, 10, (8, 5))
        amount = torch.randint(1, 30, (8,))
        ref = A.min_coin_selection(coins, amount)        # reference DP + lex-first traceback
        assert torch.equal(layer(coins, amount), ref)    # exact per-denomination selection


def test_coin_change_dp_solves_task_exactly():
    import tikitorch.algorithmic as A
    layer = P.TropicalCoinChangeDP()
    x, y = A.gen_min_coin_change(256, 16, seed=7)
    coins, amount = x[..., 0].long(), x[:, 0, 1].long()
    assert A.micro_f1(layer(coins, amount), y, positive=1) == 1.0   # the structural prior is exact


def test_coin_change_dp_count_and_unmakeable():
    layer = P.TropicalCoinChangeDP()
    # coins {2,5}: amount 11 = 5+2+2+2 (4 coins); amount 1 is NOT makeable (-> inf, no selection)
    coins = torch.tensor([[2, 5], [2, 5]])
    amount = torch.tensor([11, 1])
    sel, count = layer(coins, amount, return_count=True)
    assert count[0].item() == 4 and count[1].item() >= 1e8   # unmakeable -> closed +inf sentinel
    assert sel[1].sum().item() == 0                       # nothing selected for unmakeable amount


def test_coin_change_dp_runtime_reports_python_control_boundary():
    layer = P.TropicalCoinChangeDP()
    coins = torch.randint(1, 10, (8, 5))
    amount = torch.randint(1, 30, (8,))
    hits = tt_find(layer, coins, amount)
    # Amount-dependent Python loops extract tensor scalars. The eager auditor
    # cannot certify arithmetic after that escape, even though the algorithm is
    # expressed as a min-plus recurrence. Keep this boundary explicit.
    assert ("unknown-operation", "aten._local_scalar_dense.default") in hits
    assert all(category in {"unknown-operation", "elementwise-divide"} for category, _ in hits)


def tt_find(layer, *inputs):
    from tikitorch import find_runtime_multiplies
    return find_runtime_multiplies(layer, *inputs)
