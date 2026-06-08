"""tikitorch.algorithmic — a CLRS-style algorithmic-reasoning benchmark suite.

**Why tropical models for algorithmic reasoning.**
A large class of classical algorithms are *dynamic programs over the (min,+) or
(max,+) semiring*:

* all-pairs shortest paths = min-plus matrix closure (Floyd–Warshall),
* single-source shortest paths = min-plus matrix-vector iteration (Bellman–Ford),
* longest-increasing-subsequence / knapsack / subset-sum = max-plus DPs,
* a running maximum / minimum is a one-step max-plus / min-plus scan,
* sorting is a comparison network.

Multi-head **tropical attention** (Hashemi et al., *Tropical Attention*, NeurIPS
2025, arXiv:2505.17190) universally approximates tropical circuits / DP
recurrences. Crucially, because it *selects* by argmax over additive scores
rather than forming a softmax-weighted average, it is scale-invariant and
is **scale-invariant** and generalizes to larger-magnitude inputs, where a
standard softmax transformer interpolates within its trained value range and
**saturates** once values leave it. A single causal tropical-attention layer
computes a prefix-maximum *exactly*, at any magnitude; a softmax layer can only
approximate it and degrades out-of-range.

**Honest scope (what the shipped numbers do and do not show).** The defensible,
measured result is *magnitude* OOD: on the running-sum/max regression softmax
fits in-distribution but its error blows up ~20x at 8x value-scale, while the
tropical model is scale-invariant (~2x). It does **not** currently *beat* softmax
outright there (it under-fits that regression in-distribution). On *length* OOD,
both the tropical and softmax models length-generalize on position-free tasks —
softmax is not length-fragile here. (An earlier draft claimed a softmax
length-collapse on ``sort``; that was an artifact of an untrained absolute
positional embedding shared by *both* arms, not a property of softmax.) A clean
tropical-beats-softmax demonstration needs the relative-position reasoner, a
head-matched protocol, and the combinatorial DP tasks — the documented next step;
the harness to run it ships here.

**What this module provides.**

* verified data generators + reference solvers for the tropical DP tasks above,
* in-distribution **and** out-of-distribution (length / magnitude) eval splits,
* a closure-verified :class:`TropicalReasoner` model (``assert_closed``-true) and
  a matched softmax-attention baseline for head-to-head comparison,
* a one-call :func:`evaluate_task` harness that trains a model and reports
  in-distribution and OOD accuracy.

    from tikitorch.algorithmic import evaluate_task
    res = evaluate_task("cumulative_max", attention="tropical", steps=300)
    print(res["id_acc"], res["ood_len_acc"], res["ood_mag_acc"])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as tnn

import tikitorch.nn as nn
from tikitorch.backend import FpgaAccelerable

__all__ = [
    # generators
    "gen_cumulative_max", "gen_sort", "gen_lis", "gen_shortest_path",
    "gen_subset_sum", "gen_min_coin_change", "gen_knapsack",
    # reference solvers
    "lis_lengths", "bellman_ford", "floyd_warshall",
    "subset_sum_selection", "min_coin_table", "min_coin_selection",
    "knapsack_table", "knapsack_selection",
    # metrics
    "micro_f1",
    # task registry + model + harness
    "Task", "TASKS", "get_task", "TropicalReasoner", "build_reasoner",
    "evaluate_task",
]


# ===========================================================================
# Reference solvers (ground truth — pure numpy/torch, no learning)
# ===========================================================================

def lis_lengths(x: torch.Tensor) -> torch.Tensor:
    """Longest-increasing-subsequence DP table (max-plus DP).

    ``x: (B, L)`` -> ``(B, L)`` where ``dp[i] = 1 + max(dp[j] : j < i, x[j] < x[i])``
    (``dp[i] = 1`` if no such ``j``). ``dp[i]`` is the length of the longest
    strictly-increasing subsequence ending at position ``i``; ``max_i dp[i]`` is
    the LIS length. This is the canonical max-plus dynamic program.
    """
    B, L = x.shape
    dp = torch.ones(B, L, dtype=torch.float32)
    for i in range(1, L):
        # for each i, best dp[j] over j<i with x[j]<x[i]
        prev = x[:, :i]                                   # (B, i)
        mask = prev < x[:, i:i + 1]                       # (B, i)
        cand = torch.where(mask, dp[:, :i], torch.zeros_like(dp[:, :i]))
        dp[:, i] = 1.0 + cand.max(dim=1).values
    return dp


def bellman_ford(W: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Single-source shortest paths by min-plus relaxation (Bellman–Ford).

    ``W: (B, N, N)`` non-negative edge weights (``inf`` = no edge) -> ``(B, N)``
    shortest distance from ``src`` to every node. ``N-1`` rounds of the min-plus
    update ``d_v = min_u (d_u + W[u, v])`` — a min-plus matrix-vector DP.
    """
    B, N, _ = W.shape
    INF = torch.full((B, N), float("inf"))
    d = INF.clone()
    d[:, src] = 0.0
    for _ in range(N - 1):
        # relax all edges: d_v = min(d_v, min_u d_u + W[u,v])
        cand = d.unsqueeze(2) + W                         # (B, N_u, N_v)
        d = torch.minimum(d, cand.min(dim=1).values)
    return d


def floyd_warshall(W: torch.Tensor) -> torch.Tensor:
    """All-pairs shortest paths = min-plus matrix closure (Floyd–Warshall).

    ``W: (B, N, N)`` -> ``(B, N, N)``. ``D = min_k (D[i,k] + D[k,j])`` iterated
    to a fixed point — the textbook tropical (min-plus) matrix-closure algorithm.
    """
    B, N, _ = W.shape
    D = W.clone()
    for k in range(N):
        D = torch.minimum(D, D[:, :, k:k + 1] + D[:, k:k + 1, :])
    return D


# --- combinatorial min/max-plus DPs (boolean / selection reconstructions) ---

def subset_sum_selection(items: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Subset-sum membership for the lexicographically-first solution (boolean DP).

    ``items: (B, n)`` positive ints, ``target: (B,)`` -> ``sel: (B, n)`` a ``{0,1}``
    mask: ``sel[i] = 1`` iff item ``i`` is in the *lexicographically-first* subset
    summing to ``target`` (all-zeros if no subset sums to it). Lexicographically
    first = greedily prefer *including* earlier indices: scan ``i = 0..n-1`` and
    include item ``i`` whenever the remaining suffix can still complete the sum.

    The core is the boolean reachability DP ``reach[i][s] = OR over keep/drop`` —
    a max-plus DP on the Boolean ``({0,1}, OR, AND)`` semiring. We expose the
    canonical *reconstruction* (the witnessed selection), which is what a reasoner
    must execute, rather than the raw reachability table.
    """
    if items.dim() != 2:
        raise ValueError(f"items must be (B, n), got {tuple(items.shape)}")
    B, n = items.shape
    a = items.to(torch.int64)
    t = target.to(torch.int64)
    sel = torch.zeros(B, n, dtype=torch.int64)
    for b in range(B):
        sel[b] = _subset_sum_one(a[b].tolist(), int(t[b].item()))
    return sel


def _suffix_reachable(a: list, start: int, s: int) -> bool:
    """Can a subset of ``a[start:]`` sum to exactly ``s`` (boolean reachability DP)."""
    if s < 0:
        return False
    n = len(a)
    # reach[s] over suffix, filled right-to-left
    reach = [False] * (s + 1)
    reach[0] = True
    for i in range(n - 1, start - 1, -1):
        w = a[i]
        if w > s:
            continue
        for cur in range(s, w - 1, -1):
            if reach[cur - w]:
                reach[cur] = True
    return reach[s]


def _subset_sum_one(a: list, target: int) -> torch.Tensor:
    n = len(a)
    sel = torch.zeros(n, dtype=torch.int64)
    if not _suffix_reachable(a, 0, target):
        return sel                                   # no subset sums to target
    rem = target
    for i in range(n):
        # prefer KEEPING item i if the rest can still complete the sum
        if a[i] <= rem and _suffix_reachable(a, i + 1, rem - a[i]):
            sel[i] = 1
            rem -= a[i]
        # else drop i (suffix without i must reach rem, guaranteed by invariant)
    return sel


def min_coin_table(coins: torch.Tensor, amount: int) -> torch.Tensor:
    """Min-plus DP table for unbounded coin change.

    ``coins: (B, k)`` positive denominations -> ``dp: (B, amount+1)`` where
    ``dp[a] = min_c (dp[a - c] + 1)`` (``inf`` if ``a`` is unreachable, ``0`` at
    ``a = 0``). The textbook min-plus dynamic program for fewest-coins change.
    """
    if coins.dim() != 2:
        raise ValueError(f"coins must be (B, k), got {tuple(coins.shape)}")
    B = coins.shape[0]
    dp = torch.full((B, amount + 1), float("inf"))
    dp[:, 0] = 0.0
    cl = coins.to(torch.int64).tolist()
    for b in range(B):
        for a in range(1, amount + 1):
            best = float("inf")
            for c in cl[b]:
                if c <= a and dp[b, a - c] + 1 < best:
                    best = dp[b, a - c].item() + 1
            dp[b, a] = best
    return dp


def min_coin_selection(coins: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """Per-denomination usage in the lexicographically-first min-coin solution.

    ``coins: (B, k)`` (sorted descending or arbitrary), ``amount: (B,)`` ->
    ``sel: (B, k)`` ``{0,1}`` where ``sel[j] = 1`` iff denomination ``j`` is used
    (count >= 1) in a fewest-coins solution reconstructed by preferring earlier
    denominations. ``min-plus DP`` table + greedy reconstruction over the table.
    """
    if coins.dim() != 2:
        raise ValueError(f"coins must be (B, k), got {tuple(coins.shape)}")
    B, k = coins.shape
    amt = amount.to(torch.int64)
    A = int(amt.max().item()) if B else 0
    table = min_coin_table(coins, A)
    sel = torch.zeros(B, k, dtype=torch.int64)
    cl = coins.to(torch.int64).tolist()
    for b in range(B):
        rem = int(amt[b].item())
        if torch.isinf(table[b, rem]):
            continue                                 # amount not makeable
        while rem > 0:
            # pick the earliest denomination that lies on an optimal path
            for j, c in enumerate(cl[b]):
                if c <= rem and table[b, rem - c] + 1 == table[b, rem]:
                    sel[b, j] = 1
                    rem -= c
                    break
            else:
                break
    return sel


def knapsack_table(weights: torch.Tensor, values: torch.Tensor,
                   capacity: int) -> torch.Tensor:
    """Max-plus DP table for the 0/1 knapsack.

    ``weights, values: (B, n)`` -> ``dp: (B, n+1, capacity+1)`` where
    ``dp[i][c] = max(dp[i-1][c], dp[i-1][c - w_i] + v_i)`` — the canonical
    max-plus knapsack recurrence; ``dp[n][capacity]`` is the optimal value.
    """
    if weights.shape != values.shape:
        raise ValueError("weights and values must have the same shape")
    B, n = weights.shape
    w = weights.to(torch.int64).tolist()
    v = values.to(torch.int64).tolist()
    dp = torch.zeros(B, n + 1, capacity + 1, dtype=torch.float32)
    for b in range(B):
        for i in range(1, n + 1):
            wi, vi = w[b][i - 1], v[b][i - 1]
            for c in range(capacity + 1):
                best = dp[b, i - 1, c]
                if wi <= c:
                    best = max(best, dp[b, i - 1, c - wi] + vi)
                dp[b, i, c] = best
    return dp


def knapsack_selection(weights: torch.Tensor, values: torch.Tensor,
                       capacity: torch.Tensor) -> torch.Tensor:
    """Per-item membership in the optimal 0/1-knapsack set (max-plus DP).

    ``weights, values: (B, n)``, ``capacity: (B,)`` -> ``sel: (B, n)`` ``{0,1}``
    where ``sel[i] = 1`` iff item ``i`` is taken in the reconstructed optimal set.
    The standard knapsack table walk: item ``i`` is taken iff *not* taking it would
    have left value on the table, i.e. ``dp[i][c] != dp[i-1][c]`` (and it fits).
    This deterministically prefers *skipping* on a tie, so it yields the optimal
    set of fewest items / smallest weight; the brute-force reference applies the
    identical tie rule so the two agree exactly.
    """
    if weights.shape != values.shape:
        raise ValueError("weights and values must have the same shape")
    B, n = weights.shape
    cap = capacity.to(torch.int64)
    C = int(cap.max().item()) if B else 0
    table = knapsack_table(weights, values, C)
    w = weights.to(torch.int64).tolist()
    sel = torch.zeros(B, n, dtype=torch.int64)
    for b in range(B):
        c = int(cap[b].item())
        for i in range(n, 0, -1):
            wi = w[b][i - 1]
            # item i taken iff dropping it would lose value (strict improvement)
            if wi <= c and table[b, i, c] != table[b, i - 1, c]:
                sel[b, i - 1] = 1
                c -= wi
    return sel


# ===========================================================================
# Data generators
# ===========================================================================

def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def gen_cumulative_max(n: int, seq_len: int, *, scale: float = 1.0,
                       seed: int = 0):
    """Running-maximum task (one-step max-plus scan).

    ``x: (n, L, 1)`` i.i.d. uniform in ``[-scale, scale]``; ``y: (n, L, 1)`` the
    running maximum ``y[i] = max(x[0..i])``. Regression. The cleanest OOD probe:
    a causal max-plus attention computes this exactly at any length/magnitude;
    ``scale`` controls the magnitude-OOD axis.
    """
    g = _gen(seed)
    x = (torch.rand(n, seq_len, 1, generator=g) * 2 - 1) * scale
    y = torch.cummax(x, dim=1).values
    return x, y


def gen_running_max(n: int, seq_len: int, *, n_symbols: int = 20, seed: int = 0):
    """Running-maximum-token task (causal max-plus scan, classification).

    ``x: (n, L)`` integer tokens; ``y: (n, L)`` the running maximum token
    ``y[i] = max(x[0..i])``. Per-position classification — the classification
    twin of :func:`gen_cumulative_max` that sidesteps regression-scale issues.
    A single causal max-plus attention computes this *exactly* and at any length,
    so it is the cleanest probe of tropical length-generalization.
    """
    g = _gen(seed)
    x = torch.randint(0, n_symbols, (n, seq_len), generator=g)
    y = torch.cummax(x, dim=1).values
    return x, y


def gen_sort(n: int, seq_len: int, *, n_symbols: int = 20, seed: int = 0):
    """Sorting task (comparison network).

    ``x: (n, L)`` integer tokens in ``[0, n_symbols)``; ``y: (n, L)`` the same
    multiset sorted ascending. Per-position classification over ``n_symbols``.
    """
    g = _gen(seed)
    x = torch.randint(0, n_symbols, (n, seq_len), generator=g)
    y, _ = torch.sort(x, dim=1)
    return x, y


def gen_lis(n: int, seq_len: int, *, n_symbols: int = 20, seed: int = 0):
    """Longest-increasing-subsequence DP-table task (max-plus DP).

    ``x: (n, L)`` integer tokens; ``y: (n, L, 1)`` the LIS-ending-here DP table
    (see :func:`lis_lengths`). Regression — the model must *execute* the DP, not
    just report a single number.
    """
    g = _gen(seed)
    x = torch.randint(0, n_symbols, (n, seq_len), generator=g)
    y = lis_lengths(x).unsqueeze(-1)
    return x, y


def gen_shortest_path(n: int, n_nodes: int = 8, *, p_edge: float = 0.5,
                      max_w: float = 9.0, seed: int = 0):
    """Single-source shortest-path task (min-plus Bellman–Ford).

    Random directed graphs on ``n_nodes`` nodes; returns ``(W, D)`` with
    ``W: (n, N, N)`` weights (``inf`` for absent edges, ``0`` on the diagonal)
    and ``D: (n, N)`` shortest distances from node ``0`` (:func:`bellman_ford`).
    The matrix-shaped tropical task; the headline trainable tasks are the
    sequence ones, but this ships the generator + verified solver.
    """
    g = _gen(seed)
    w = torch.randint(1, int(max_w) + 1, (n, n_nodes, n_nodes), generator=g).float()
    present = torch.rand(n, n_nodes, n_nodes, generator=g) < p_edge
    W = torch.where(present, w, torch.full_like(w, float("inf")))
    eye = torch.eye(n_nodes, dtype=torch.bool)
    W[:, eye] = 0.0
    D = bellman_ford(W, src=0)
    return W, D


# --- combinatorial DP generators (multi-feature scalar inputs) --------------

def gen_subset_sum(n: int, seq_len: int, *, max_item: int = 9, seed: int = 0):
    """Subset-sum selection task (boolean / max-plus DP).

    Each example is a length-``seq_len`` sequence of positive ints in
    ``[1, max_item]`` plus a per-example target. Inputs are stacked as
    ``x: (n, L, 2)`` with channel 0 = the item value and channel 1 = the (broadcast)
    target; labels ``y: (n, L)`` are the ``{0,1}`` membership of the
    lexicographically-first subset summing to the target
    (see :func:`subset_sum_selection`). The target is *constructed* from a random
    subset so every example is solvable. Per-position binary classification.
    """
    g = _gen(seed)
    items = torch.randint(1, int(max_item) + 1, (n, seq_len), generator=g)
    # build a solvable target from a random nonempty subset (>=1 item)
    keep = torch.rand(n, seq_len, generator=g) < 0.5
    empty = ~keep.any(dim=1)
    if empty.any():
        keep[empty, 0] = True                          # force >=1 item
    target = (items * keep.to(items.dtype)).sum(dim=1)
    y = subset_sum_selection(items, target)
    x = torch.stack([items.to(torch.float32),
                     target.to(torch.float32).unsqueeze(1).expand_as(items)], dim=-1)
    return x, y


def gen_min_coin_change(n: int, seq_len: int, *, max_coin: int = 9,
                        max_amount: int = 24, seed: int = 0):
    """Min-coin-change selection task (min-plus DP).

    Each example is ``seq_len`` distinct positive coin denominations in
    ``[1, max_coin]`` plus a per-example amount in ``[1, max_amount]``; a coin of
    value ``1`` is always included so every amount is makeable. Inputs
    ``x: (n, L, 2)`` (channel 0 = denomination, channel 1 = broadcast amount);
    labels ``y: (n, L)`` = ``{0,1}`` "denomination used in the fewest-coins
    solution" (:func:`min_coin_selection`). Per-position binary classification.
    """
    g = _gen(seed)
    coins = torch.empty(n, seq_len, dtype=torch.int64)
    for b in range(n):
        # distinct denominations including 1 so any amount is reachable
        pool = torch.randperm(int(max_coin), generator=g)[:seq_len - 1] + 1
        row = torch.cat([torch.tensor([1]), pool])[:seq_len]
        if row.numel() < seq_len:                      # pad if pool too small
            extra = torch.randint(1, int(max_coin) + 1,
                                  (seq_len - row.numel(),), generator=g)
            row = torch.cat([row, extra])
        coins[b] = row
    amount = torch.randint(1, int(max_amount) + 1, (n,), generator=g)
    y = min_coin_selection(coins, amount)
    x = torch.stack([coins.to(torch.float32),
                     amount.to(torch.float32).unsqueeze(1).expand_as(coins)], dim=-1)
    return x, y


def gen_knapsack(n: int, seq_len: int, *, max_weight: int = 6, max_value: int = 9,
                 cap_frac: float = 0.5, seed: int = 0):
    """0/1-knapsack selection task (max-plus DP, multi-feature input).

    ``seq_len`` items each with a weight in ``[1, max_weight]`` and value in
    ``[1, max_value]`` plus a per-example capacity (about ``cap_frac`` of the total
    weight). Inputs ``x: (n, L, 3)`` (channels weight, value, broadcast capacity);
    labels ``y: (n, L)`` = ``{0,1}`` item membership in the optimal set
    (:func:`knapsack_selection`). Per-position binary classification; the headline
    trainable combinatorial task and the one that exercises the multi-feature
    closed embedding (weight + value + capacity, add-only).
    """
    g = _gen(seed)
    weights = torch.randint(1, int(max_weight) + 1, (n, seq_len), generator=g)
    values = torch.randint(1, int(max_value) + 1, (n, seq_len), generator=g)
    total = weights.sum(dim=1).to(torch.float32)
    capacity = (total * cap_frac).round().clamp(min=1).to(torch.int64)
    y = knapsack_selection(weights, values, capacity)
    x = torch.stack([weights.to(torch.float32), values.to(torch.float32),
                     capacity.to(torch.float32).unsqueeze(1).expand_as(weights)],
                    dim=-1)
    return x, y


# ===========================================================================
# Task registry
# ===========================================================================

@dataclass
class Task:
    """A trainable algorithmic-reasoning task.

    ``kind`` is ``"class"`` (per-position classification, cross-entropy) or
    ``"reg"`` (per-position regression, L1). ``causal`` gates the attention mask.
    ``input_kind`` is ``"tokens"`` (embedded), ``"scalar"`` (value-projected) or
    ``"multiscalar"`` (per-feature add-only projection; needs ``in_features``).
    ``metric`` selects the reported eval metric: ``"acc"`` (per-token accuracy) or
    ``"f1"`` (per-token Micro-F1 over the positive class, for boolean selection
    tasks such as subset_sum / knapsack where positives are sparse).
    """
    name: str
    kind: str                       # "class" | "reg"
    input_kind: str                 # "tokens" | "scalar" | "multiscalar"
    causal: bool
    use_pos: bool
    generate: Callable
    n_symbols: Optional[int] = None     # vocab for token tasks
    out_dim: int = 1                    # regression out width
    in_features: int = 1                # feature width for multiscalar inputs
    metric: str = "acc"                 # "acc" | "f1"


TASKS = {
    "cumulative_max": Task(
        name="cumulative_max", kind="reg", input_kind="scalar",
        causal=True, use_pos=False, generate=gen_cumulative_max),
    "running_max": Task(
        name="running_max", kind="class", input_kind="tokens",
        causal=True, use_pos=False, generate=gen_running_max, n_symbols=20),
    "sort": Task(
        name="sort", kind="class", input_kind="tokens",
        causal=False, use_pos=True, generate=gen_sort, n_symbols=20),
    "lis": Task(
        name="lis", kind="reg", input_kind="tokens",
        causal=True, use_pos=False, generate=gen_lis, n_symbols=20),
    # combinatorial selection DPs: per-item binary classification (in/out of the
    # optimal set), scored with Micro-F1 because positives are sparse. knapsack is
    # the headline trainable one (multi-feature add-only embed: weight,value,cap).
    "knapsack": Task(
        name="knapsack", kind="class", input_kind="multiscalar",
        causal=False, use_pos=True, generate=gen_knapsack,
        n_symbols=2, in_features=3, metric="f1"),
    "subset_sum": Task(
        name="subset_sum", kind="class", input_kind="multiscalar",
        causal=False, use_pos=True, generate=gen_subset_sum,
        n_symbols=2, in_features=2, metric="f1"),
    "min_coin_change": Task(
        name="min_coin_change", kind="class", input_kind="multiscalar",
        causal=False, use_pos=True, generate=gen_min_coin_change,
        n_symbols=2, in_features=2, metric="f1"),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name]


# ===========================================================================
# Closure-verified reasoner (and a matched softmax baseline)
# ===========================================================================

class _ScalarEmbed(tnn.Module):
    """Closed scalar->d_model embedding: broadcast value + per-channel bias.

    ``y[..., c] = x + b_c`` (a broadcast and an add — no multiply). Lets max-plus
    attention compare and select on the scalar value directly, so a running-max
    scan is representable exactly.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.bias = tnn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, L, 1)
        return x + self.bias


class _MultiScalarEmbed(tnn.Module):
    """Closed multi-feature -> d_model embedding (add-only, no multiply).

    For ``in_features`` scalar features per position, route each feature onto a
    contiguous slice of the ``d_model`` channels by *broadcasting* its value and
    adding a per-channel bias: ``y[..., s] = x[..., feat(s)] + b_s``. Channels are
    split as evenly as possible across features; the last feature takes the
    remainder. This is the exact multi-feature analogue of :class:`_ScalarEmbed`
    (a broadcast + an add, per feature) so a max-plus reasoner can compare and
    select on each raw feature value directly, and the network stays
    ``assert_closed``-true (no multiply / matmul anywhere in the embed).
    """

    def __init__(self, d_model: int, in_features: int):
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be >= 1")
        if d_model < in_features:
            raise ValueError("d_model must be >= in_features")
        self.in_features = in_features
        base = d_model // in_features
        sizes = [base] * in_features
        sizes[-1] += d_model - base * in_features        # remainder to last feature
        # map each output channel to its source feature index (constant, no grad)
        src = torch.cat([torch.full((s,), f, dtype=torch.long)
                         for f, s in enumerate(sizes)])
        self.register_buffer("src_index", src)           # (d_model,)
        self.bias = tnn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, L, in_features)
        # gather each output channel's source feature, then add per-channel bias
        spread = x.index_select(-1, self.src_index)       # (B, L, d_model), add-free
        return spread + self.bias


def _sinusoidal_pos(L: int, d: int, device, dtype):
    """Standard sinusoidal positional encoding (length-generalizing, not closed).

    Used only by the softmax baseline so its length-OOD comparison against the
    tropical relative-shift attention is fair (both have a position scheme that
    extends past the trained length). Not multiplier-free -- but the softmax arm
    is the unclosed baseline anyway.
    """
    pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(0, d, 2, device=device, dtype=dtype)
    div = torch.exp(-(torch.log(torch.tensor(10000.0, device=device, dtype=dtype))) * i / d)
    pe = torch.zeros(L, d, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class TropicalReasoner(FpgaAccelerable, tnn.Module):
    """A closure-verified sequence model for algorithmic reasoning.

    embed -> ``n_layers`` tropical transformer blocks (Hilbert projective
    attention by default) -> per-position tropical head. With
    ``attention="tropical"`` the whole network is ``assert_closed``-true
    (comparisons + additions only); ``attention="softmax"`` swaps in standard
    multi-head attention as the baseline (intentionally *not* closed).
    """

    def __init__(self, task: Task, *, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, d_ff: Optional[int] = None, tau: float = 1.0,
                 attention: str = "tropical", score_variant: str = "hilbert",
                 norm_type: str = "tropical_shift", max_len: int = 256,
                 pos_mode: str = "rope_shift", agg_variant: str = "argmax",
                 agg_topk_k: int = 2, head: str = "auto"):
        super().__init__()
        assert pos_mode in ("none", "abs", "rope_shift"), pos_mode
        assert head in ("auto", "tropical", "linear"), head
        self.task = task
        self.attention = attention
        # rope_shift only applies where position matters; order-free tasks need none.
        self.pos_mode = pos_mode if task.use_pos else "none"
        d_ff = d_ff or d_model * 2

        if task.input_kind == "tokens":
            self.embed = tnn.Embedding(task.n_symbols, d_model)
        elif task.input_kind == "multiscalar":
            self.embed = _MultiScalarEmbed(d_model, task.in_features)
        else:
            self.embed = _ScalarEmbed(d_model)
        # Absolute learned positions ONLY for the explicit 'abs' ablation (they do
        # NOT length-generalize -- untrained rows beyond train_len). rope_shift gets
        # position from the attention itself (closed relative shift on the tropical
        # arm; sinusoidal on the softmax arm, so length-OOD is a FAIR comparison).
        self.pos = tnn.Embedding(max_len, d_model) if self.pos_mode == "abs" else None
        use_shift = (attention == "tropical" and self.pos_mode == "rope_shift")

        if attention == "tropical":
            self.blocks = tnn.ModuleList([
                nn.TropicalTransformerBlock(
                    d_model, n_heads, d_ff, dropout=0.0, tau=tau,
                    score_variant=score_variant, norm_type=norm_type,
                    proj_init_identity=True,   # preserve per-dim value signal (fix A4)
                    use_shift_attn=use_shift, max_seq_len=max_len,
                    shift_mode="rope_relative" if use_shift else "free",
                    agg_variant=agg_variant, agg_topk_k=agg_topk_k)
                for _ in range(n_layers)])
        elif attention == "softmax":
            self.blocks = tnn.ModuleList([
                tnn.TransformerEncoderLayer(
                    d_model, n_heads, d_ff, dropout=0.0, batch_first=True)
                for _ in range(n_layers)])
        else:
            raise ValueError(f"attention must be 'tropical' or 'softmax', got {attention!r}")

        # Head: "auto" = tropical head for the tropical arm, linear for softmax.
        # "tropical"/"linear" force the SAME head on both arms (isolates attention).
        use_trop_head = (head == "tropical") or (head == "auto" and attention == "tropical")
        if task.kind == "class":
            self.head = (nn.TropicalTopKModule(d_model, task.n_symbols, k=8)
                         if use_trop_head else tnn.Linear(d_model, task.n_symbols))
        else:
            self.head = (nn.TropicalProjection(d_model, task.out_dim)
                         if use_trop_head else tnn.Linear(d_model, task.out_dim))

    def set_tau(self, tau: float) -> None:
        if self.attention == "tropical":
            for b in self.blocks:
                b.set_tau(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        L = h.shape[1]
        if self.pos_mode == "abs" and self.pos is not None:
            h = h + self.pos(torch.arange(L, device=h.device))
        elif self.pos_mode == "rope_shift" and self.attention == "softmax":
            # softmax has no closed relative analogue -> sinusoidal positions, the
            # standard length-generalizing scheme, so the length-OOD test is fair.
            h = h + _sinusoidal_pos(L, h.shape[-1], h.device, h.dtype)
        if self.attention == "tropical":
            for b in self.blocks:
                h = b(h, causal_mask=self.task.causal)
        else:
            mask = None
            if self.task.causal:
                mask = torch.triu(torch.full((L, L), float("-inf"), device=h.device), 1)
            for b in self.blocks:
                h = b(h, src_mask=mask)
        return self.head(h)


def build_reasoner(task_name: str, **kw) -> TropicalReasoner:
    return TropicalReasoner(get_task(task_name), **kw)


# Architecture kwargs of TropicalReasoner that round-trip through a saved config
# (``n_symbols`` lives in the Task, not here; the rest of a zoo config is metadata).
_REASONER_ARCH_KEYS = ("max_len", "d_model", "n_layers", "n_heads", "attention",
                       "score_variant", "norm_type", "pos_mode", "agg_variant", "head")


def load_reasoner(path: str, map_location: str = "cpu") -> TropicalReasoner:
    """Load a closure-certified ``TropicalReasoner`` saved by the algorithmic zoo.

    ``TropicalReasoner`` takes a ``Task`` (not a flat kwarg), so a plain
    ``tt.load(path)`` can't rebuild it; this reconstructs the Task vocab + the
    architecture from the saved config, then loads the state_dict bit-identically.
    """
    import torch as _torch
    import tikitorch as _tt
    cfg = _torch.load(path, map_location=map_location, weights_only=False)["config"]
    task = get_task(cfg["task"])
    if cfg.get("n_symbols") is not None and task.input_kind == "tokens":
        task = Task(**{**task.__dict__, "n_symbols": cfg["n_symbols"]})
    model = TropicalReasoner(task, **{k: cfg[k] for k in _REASONER_ARCH_KEYS if k in cfg})
    _tt.load(path, model=model, map_location=map_location)
    return model


# ===========================================================================
# Train / evaluate harness  (in-distribution + OOD length & magnitude)
# ===========================================================================

def _make_batch(task: Task, n: int, seq_len: int, *, scale: float, seed: int,
                device):
    if task.name == "cumulative_max":
        x, y = task.generate(n, seq_len, scale=scale, seed=seed)
    elif task.name in ("sort", "running_max", "lis"):
        x, y = task.generate(n, seq_len, n_symbols=task.n_symbols, seed=seed)
    else:  # combinatorial selection DPs (knapsack / subset_sum / min_coin_change)
        x, y = task.generate(n, seq_len, seed=seed)
    return x.to(device), y.to(device)


def micro_f1(pred: torch.Tensor, y: torch.Tensor, positive: int = 1) -> float:
    """Per-token Micro-F1 of the ``positive`` class over flattened masks.

    ``pred: (..., C)`` logits (argmax taken) or ``(...)`` integer labels; ``y``
    the matching integer targets. Both are flattened, reduced to a boolean
    "is positive" mask, and scored as
    ``F1 = 2*TP / (2*TP + FP + FN)`` (the micro / token-pooled F1). Returns ``1.0``
    when there are no positives in either prediction or target (degenerate but
    perfectly-agreeing case). The metric of choice for the sparse boolean-selection
    DPs (subset_sum / knapsack), where plain accuracy is dominated by the many
    "not selected" tokens.
    """
    p = pred.argmax(-1) if pred.dim() > y.dim() else pred
    p = (p.reshape(-1) == positive)
    t = (y.reshape(-1) == positive)
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0                                   # no positives anywhere, agree
    return 2.0 * tp / denom


def _accuracy(task: Task, pred: torch.Tensor, y: torch.Tensor) -> float:
    if task.kind == "class":
        if task.metric == "f1":
            return micro_f1(pred, y, positive=1)
        return (pred.argmax(-1) == y).float().mean().item()
    # regression -> "solved a cell" = rounds to the exact integer/value
    if task.name == "cumulative_max":
        return (pred - y).abs().lt(0.05 * (y.abs().mean() + 1e-6) + 1e-3).float().mean().item()
    return (pred.round() == y.round()).float().mean().item()   # DP-table exact-cell


def evaluate_task(task_name: str, *, attention: str = "tropical",
                  score_variant: str = "hilbert", norm_type: str = "tropical_shift",
                  d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                  train_len: int = 16, ood_len: int = 48, train_scale: float = 1.0,
                  ood_scale: float = 8.0, n_symbols: int = 20,
                  train_n: int = 4096, test_n: int = 512, batch_size: int = 128,
                  steps: int = 400, lr: float = 3e-3, anneal: bool = True,
                  pos_mode: str = "rope_shift", agg_variant: str = "argmax",
                  agg_topk_k: int = 2, head: str = "auto", closed: bool = False,
                  closed_opt: str = "puretropical", accum: int = 1,
                  device: str = "cpu", seed: int = 0, verbose: bool = False,
                  return_model: bool = False):
    """Train a reasoner on short/small inputs; report in-dist + OOD accuracy.

    Returns a dict with ``train_acc``, ``id_acc`` (held-out, same length/scale),
    ``ood_len_acc`` (length ``ood_len``), and, for ``cumulative_max``,
    ``ood_mag_acc`` (values scaled by ``ood_scale``). The tropical reasoner
    should hold up under OOD where the softmax baseline collapses.
    """
    task = get_task(task_name)
    # n_symbols is the configurable vocab for the *token* tasks only; the boolean
    # selection DPs have a fixed binary head (n_symbols=2) and must not be widened.
    if task.n_symbols is not None and task.input_kind == "tokens":
        task = Task(**{**task.__dict__, "n_symbols": n_symbols})
    torch.manual_seed(seed)
    model = TropicalReasoner(task, d_model=d_model, n_layers=n_layers,
                             n_heads=n_heads, attention=attention,
                             score_variant=score_variant, norm_type=norm_type,
                             pos_mode=pos_mode, agg_variant=agg_variant,
                             agg_topk_k=agg_topk_k, head=head,
                             max_len=max(train_len, ood_len) + 1).to(device)

    # Level-B closed training: hard forward (no anneal/logsumexp), a closure-faithful
    # loss (perceptron ±1 grad for class / L1 sign grad for reg), and a sign-only
    # optimizer (SignSGD, sign-gated add). The whole step is multiplier-free --
    # verify with tikitorch.audit.assert_training_closed. Non-closed default keeps
    # the Adam + tau-anneal + cross-entropy GPU proxy.
    if closed:
        import tikitorch.optim as _optim
        from tikitorch.functional import tropical_perceptron_loss
        # All closed (zero-DSP) optimizers -- verify with assert_training_closed.
        # lr = 2^-lr_shift (a power-of-2 bit-shift on FPGA).
        import math as _math
        lr_shift = max(1, min(12, round(-_math.log2(max(lr, 1e-9)))))
        if closed_opt == "tropicaladam":
            # closed adaptive (sign-streak Rprop): per-param pow-2 step.
            opt = _optim.TropicalAdam(model.parameters(), step_init=2.0 ** -lr_shift)
        elif closed_opt == "countagree":
            # closed magnitude-aware: gradient-sign agreement over the accum window.
            opt = _optim.CountAgreementSGD(model.parameters(), base_shift=lr_shift)
        elif closed_opt == "perlayer":
            # closed per-tensor adaptive pow-2 step (Rprop on a scalar shift).
            opt = _optim.PerLayerStepSGD(model.parameters(), base_shift=lr_shift)
        elif closed_opt == "frexpmag":
            # closed magnitude scaling via the frexp exponent tier.
            opt = _optim.FrexpMagnitudeSGD(model.parameters(), base_shift=lr_shift)
        elif closed_opt == "expadam":
            # closed "tropical Adam v2": counter momentum + frexp-exponent EMA.
            opt = _optim.ExponentEMAAdam(model.parameters(), base_shift=lr_shift)
        else:  # "puretropical": bit-shift lr + integer-counter momentum
            opt = _optim.PureTropicalSGD(model.parameters(), lr_shift=lr_shift)
        # CountAgreementSGD pools gradient SIGNS over the accum window via an
        # accumulate() hook (signs are scale-invariant -> do NOT divide the loss).
        uses_accum_hook = closed_opt == "countagree"
        anneal = False
        if attention == "tropical":
            model.set_tau(0.0)

        def loss_fn(pred, y):
            if task.kind == "class":
                return tropical_perceptron_loss(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
            return (pred - y).abs().mean()        # L1: grad is sign(pred-y) in {-1,0,1}
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        uses_accum_hook = False

        def loss_fn(pred, y):
            if task.kind == "class":
                return tnn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
            return (pred - y).abs().mean()

    xtr, ytr = _make_batch(task, train_n, train_len, scale=train_scale,
                           seed=seed + 1, device=device)
    for step in range(steps):
        if anneal and attention == "tropical":
            model.set_tau(max(0.05, 1.0 * (1 - step / max(1, steps - 1))))
        opt.zero_grad()
        if uses_accum_hook:
            # CountAgreementSGD: per microbatch backward() then accumulate() reads
            # sign(grad) into int counters and zeroes grad. No /accum (sign-invariant).
            opt.reset_accum()
            for _ in range(max(1, accum)):
                opt.zero_grad()
                idx = torch.randint(0, train_n, (batch_size,), device=device)
                loss = loss_fn(model(xtr[idx]), ytr[idx])
                loss.backward()
                opt.accumulate()
            opt.step()
        else:
            for _ in range(max(1, accum)):       # grad accumulation -> large effective batch
                idx = torch.randint(0, train_n, (batch_size,), device=device)
                loss = loss_fn(model(xtr[idx]), ytr[idx]) / max(1, accum)
                loss.backward()
            opt.step()
        if verbose and step % max(1, steps // 5) == 0:
            print(f"  [{task_name}/{attention}{'/closed' if closed else ''}] step {step:4d} loss {loss.item():.4f}")

    model.eval()
    if attention == "tropical":
        model.set_tau(0.0)
    with torch.no_grad():
        out = {"task": task_name, "attention": attention,
               "train_acc": _accuracy(task, model(xtr[:test_n]), ytr[:test_n])}   # A8: inside no_grad
        xid, yid = _make_batch(task, test_n, train_len, scale=train_scale, seed=seed + 2, device=device)
        out["id_acc"] = _accuracy(task, model(xid), yid)
        xol, yol = _make_batch(task, test_n, ood_len, scale=train_scale, seed=seed + 3, device=device)
        out["ood_len_acc"] = _accuracy(task, model(xol), yol)
        if task.name == "cumulative_max":
            xom, yom = _make_batch(task, test_n, train_len, scale=ood_scale, seed=seed + 4, device=device)
            out["ood_mag_acc"] = _accuracy(task, model(xom), yom)
    if return_model:
        return out, model
    return out
