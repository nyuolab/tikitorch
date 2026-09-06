"""tikitorch.planning.dp — differentiable tropical dynamic-programming layers.

These are thin wrappers over :class:`tikitorch.planning.functional.TropicalScanFn`
(one engine, several faces). Each is exact in the tropical semiring and only
*smoothed-approximate* in a dense net:

  * :class:`TropicalViterbi`  -- max-plus trellis decode (HMM / CRF Viterbi).
  * :class:`TropicalShortestPath` -- min-plus single-source shortest path
    (Bellman-Ford relaxation); repeated, it is the all-pairs / Floyd-Warshall
    cousin and the F-factor anchor task.

``set_tau(tau>0)`` swaps the hard scan for the logsumexp relaxation
(:func:`tropical_scan_soft`) -- the soft-DP / CRF-forward training path -- and
anneals to the exact hard DP for deployment.

(DTW and edit-distance / Needleman-Wunsch are the documented fast-follow: they
are 2-D wavefront DPs rather than the 1-D trellis scan here.)
"""

import torch
import torch.nn as nn

from tikitorch.planning.functional import (
    tropical_scan,
    tropical_scan_soft,
    tropical_scan_decode,
)

__all__ = ["TropicalViterbi", "TropicalShortestPath", "TropicalCoinChangeDP"]

_NEG_INF = -1e9
_POS_INF = 1e9


class TropicalViterbi(nn.Module):
    """Max-plus trellis decode with a learnable transition matrix.

    ``trans[i, j]`` is the score of transitioning from tag ``j`` to tag ``i``.

    forward(emissions, return_path=False):
        emissions: (B, L, S) per-step tag scores.
    Returns the best-path score ``(B,)`` (differentiable: sparse gradient via the
    argmax winners on the hard path, dense via logsumexp when ``tau>0``). With
    ``return_path=True`` also returns decoded tags ``(B, L)`` (no-grad backtrace).
    """

    def __init__(self, n_tags, tau=0.0):
        super().__init__()
        self.n_tags = int(n_tags)
        self.tau = float(tau)
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))

    def set_tau(self, tau):
        self.tau = float(tau)

    def forward(self, emissions, return_path=False):
        B, L, S = emissions.shape
        if S != self.n_tags:
            raise ValueError(
                f"emissions has {S} tags, expected n_tags={self.n_tags}"
            )
        h0 = emissions[:, 0]                                 # (B, S)
        if L == 1:
            score = h0.max(dim=-1).values
            if return_path:
                return score, h0.argmax(dim=-1, keepdim=True)
            return score
        T = self.trans[None, None].expand(B, L - 1, S, S)    # cand[b,i,j]=h[j]+trans[i,j]
        emit = emissions[:, 1:]                              # (B, L-1, S)
        if self.tau > 0:
            H = tropical_scan_soft(h0, T, emit, tau=self.tau, reduce="max")
        else:
            H = tropical_scan(h0, T, emit, reduce="max")
        score = H[:, -1].max(dim=-1).values                  # (B,) best-path score
        if not return_path:
            return score
        tags = self._decode(h0, T, emit)
        return score, tags

    @torch.no_grad()
    def _decode(self, h0, T, emit):
        B = h0.shape[0]
        H, back = tropical_scan_decode(h0, T, emit, reduce="max")   # H:(B,L,S) back:(B,L-1,S)
        L = H.shape[1]
        tags = torch.empty((B, L), dtype=torch.long, device=h0.device)
        cur = H[:, -1].argmax(dim=-1)                        # (B,)
        tags[:, -1] = cur
        for t in range(L - 2, -1, -1):
            cur = back[:, t].gather(-1, cur.unsqueeze(-1)).squeeze(-1)
            tags[:, t] = cur
        return tags


class TropicalShortestPath(nn.Module):
    """Min-plus single-source shortest path (Bellman-Ford relaxation).

    ``d_{k}[i] = min_j ( d_{k-1}[j] + cost(j -> i) )`` for ``S-1`` sweeps. Pure
    add/compare; the argmin winners are the predecessor pointers.

    forward(adj, src, return_pred=False):
        adj: (S, S) or (B, S, S) standard adjacency -- ``adj[u, v]`` is the weight
            of edge ``u -> v`` (use a large value / ``inf`` for absent edges).
        src: int or (B,) long source node(s).
    Returns shortest distances ``(B, S)`` (or ``(S,)`` if ``adj`` was 2-D).
    With ``return_pred=True`` also returns predecessor pointers ``(B, S)``.
    """

    def __init__(self, tau=0.0):
        super().__init__()
        self.tau = float(tau)

    def set_tau(self, tau):
        self.tau = float(tau)

    def forward(self, adj, src, return_pred=False):
        squeeze = adj.dim() == 2
        if squeeze:
            adj = adj.unsqueeze(0)                            # (1, S, S)
        B, S, _ = adj.shape
        # T[b, t, i, j] = cost(j -> i) = adj[b, j, i] (transpose), time-invariant.
        # The diagonal is forced to 0 (a zero-cost self-loop) so the relaxation
        # KEEPS the best distance found so far -- d[i] = min(d[i], min_j(...)) --
        # the reflexive min-plus closure that makes S-1 sweeps converge.
        Tt = adj.transpose(-1, -2).clone()                   # (B, S, S), Tt[i,j]=adj[j,i]
        diag = torch.arange(S, device=adj.device)
        Tt[:, diag, diag] = 0.0
        T = Tt.unsqueeze(1).expand(B, S - 1, S, S)
        if isinstance(src, int):
            src = torch.full((B,), src, dtype=torch.long, device=adj.device)
        h0 = adj.new_full((B, S), _POS_INF)
        h0.scatter_(-1, src.unsqueeze(-1), 0.0)
        if S == 1:
            dist = h0
        elif self.tau > 0:
            H = tropical_scan_soft(h0, T, None, tau=self.tau, reduce="min")
            dist = H[:, -1]
        else:
            H = tropical_scan(h0, T, None, reduce="min")
            dist = H[:, -1]                                  # (B, S)
        if not return_pred:
            return dist.squeeze(0) if squeeze else dist
        pred = self._predecessors(h0, T)
        if squeeze:
            return dist.squeeze(0), pred.squeeze(0)
        return dist, pred

    @torch.no_grad()
    def _predecessors(self, h0, T):
        # Predecessor on the final relaxation sweep (argmin source per node).
        _, back = tropical_scan_decode(h0, T, None, reduce="min")
        return back[:, -1]                                   # (B, S)


class TropicalCoinChangeDP(nn.Module):
    """Min-plus unbounded coin-change DP as a closed (multiplier-free) layer.

    The exact fewest-coins dynamic program ``dp[a] = min_c (dp[a-c] + 1)``,
    ``dp[0]=0``, evaluated over the AMOUNT axis with **add / compare / gather only**
    (no multiplies) — the "differentiable DP layer" lever (cf. ``TropicalViterbi`` /
    ``TropicalShortestPath``) realized for coin change. Equivalently a min-plus
    shortest path on the amount line (nodes ``0..A``, edges ``a -> a+c`` of weight 1).

    ``forward(coins, amount, ...)``:
        coins:  ``(B, k)`` positive integer denominations.
        amount: ``(B,)`` (or int) target amounts.
    Returns the per-denomination usage ``sel: (B, k)`` in ``{0,1}`` of the
    *lexicographically-first* optimal solution — exactly the ``min_coin_change``
    task label (``algorithmic.min_coin_selection``) — via a closed argmin traceback.
    With ``return_count=True`` also returns the fewest-coins count ``(B,)`` (``inf``
    if an amount is not makeable). The DP table is built differentiably (sub-gradient
    through the ``min`` winners); the discrete traceback is a no-grad closed backtrace
    (like ``TropicalViterbi._decode``).

    HONEST FRAMING: at unit edge cost the layer *is* the algorithm — exact by
    construction, a strong STRUCTURAL PRIOR (like a fixed-transition Viterbi), not a
    solution gradient descent "discovers". Its point is that it makes the task
    representable with a software-closed add/compare/gather recipe (which the generic
    tropical aggregations cannot—they plateau at ~0 F1) and gives an exact reference
    head; it is not a learned win or a physical zero-DSP certificate.
    """

    def __init__(self, tau: float = 0.0):
        super().__init__()
        self.tau = float(tau)              # reserved for a soft-DP relaxation (fast-follow)

    def set_tau(self, tau):
        self.tau = float(tau)

    def forward(self, coins, amount, return_count: bool = False):
        if coins.dim() != 2:
            raise ValueError(f"coins must be (B, k), got {tuple(coins.shape)}")
        B, k = coins.shape
        if isinstance(amount, int):
            amount = torch.full((B,), amount, dtype=torch.long, device=coins.device)
        amount = amount.to(torch.long).reshape(B)
        cf = coins.to(torch.long)
        A = int(amount.max().item()) if B else 0

        # --- min-plus DP table over amounts 0..A (add/compare/gather only) ---
        # built column-by-column so dp[a] can reference earlier columns dp[a-c].
        dp = coins.new_zeros((B, 1), dtype=torch.float)               # dp[:,0] = 0
        for a in range(1, A + 1):
            idx = (a - cf).clamp(min=0, max=a - 1)                    # (B,k) source amount a-c
            cand = dp.gather(1, idx) + 1.0                            # dp[a-c] + 1 (min-plus "*"=+)
            cand = torch.where(cf <= a, cand, cand.new_full((), _POS_INF))   # mask c>a
            best = cand.min(dim=1, keepdim=True).values              # min-plus reduce over coins
            dp = torch.cat([dp, best], dim=1)                        # (B, a+1)

        # --- lexicographically-first traceback for the per-denomination selection ---
        sel = torch.zeros((B, k), dtype=torch.long, device=coins.device)
        with torch.no_grad():
            dpd = dp.detach()
            rem = amount.clone()
            ar = torch.arange(B, device=coins.device)
            for _ in range(A + 1):
                if bool((rem <= 0).all()):
                    break
                rem_c = rem.clamp(0, A)
                dp_rem = dpd.gather(1, rem_c.unsqueeze(1))            # (B,1) dp[rem]
                prev = (rem.unsqueeze(1) - cf).clamp(0, A)            # (B,k)
                dp_prev = dpd.gather(1, prev)                         # dp[rem-c]
                # denominations that lie on an optimal path: c<=rem, rem>0, dp[rem-c]+1==dp[rem]
                onpath = ((cf <= rem.unsqueeze(1)) & (rem.unsqueeze(1) > 0)
                          & ((dp_prev + 1.0 - dp_rem).abs() < 0.5)
                          & (dp_rem < _POS_INF / 2))
                has = onpath.any(dim=1)
                first = onpath.to(torch.float).argmax(dim=1)         # earliest on-path denom
                if bool(has.any()):
                    sel[ar[has], first[has]] = 1
                    used = cf.gather(1, first.unsqueeze(1)).squeeze(1)
                    rem = torch.where(has, rem - used, rem)
                rem = torch.where(~has, torch.zeros_like(rem), rem)  # unmakeable -> stop

        if return_count:
            count = dp.gather(1, amount.clamp(0, A).unsqueeze(1)).squeeze(1)  # (B,)
            return sel, count
        return sel
