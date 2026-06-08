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

__all__ = ["TropicalViterbi", "TropicalShortestPath"]

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
        assert S == self.n_tags
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
