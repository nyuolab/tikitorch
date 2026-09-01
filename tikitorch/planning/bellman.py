"""tikitorch.planning.bellman — tropical value iteration and max-plus spectra.

Deterministic value iteration is exactly a ``(max,+)`` recurrence::

    V_{k+1}[s] = max_a ( R[s, a] + gamma * V_k[ succ[s, a] ] )

The successor lookup ``V_k[succ[s, a]]`` is a pure gather (the "backward = gather"
property), the per-state ``max_a`` is the argmax routing reused throughout
TikiTorch (``torch.max`` already routes its gradient only through the winner),
and the discount ``gamma`` has a multiplication-free abstract recipe when it is
``1.0`` (shortest-path / undiscounted) or can be lowered to an arithmetic shift
when it is a supported power of two. The PyTorch reference still spells this as
a multiplication. A general float ``gamma`` is a declared software-closure
boundary; physical DSP use depends on a concrete lowering and synthesis report.

:func:`max_plus_power_iteration` computes the max-plus eigenvalue (= maximum mean
cycle = optimal average-reward gain) and its eigenvector (the relative-value
function), via repeated :func:`tikitorch.planning.functional.max_plus_matmul`.
"""

import torch
import torch.nn as nn

from tikitorch.functional import tropical_softmax
from tikitorch.planning.functional import max_plus_matmul

__all__ = ["TropicalValueIteration", "max_plus_power_iteration"]


class TropicalValueIteration(nn.Module):
    """Differentiable deterministic value iteration (max-plus Bellman backup).

    ``V_{k+1}[s] = max_a ( R[s, a] + gamma * V_k[succ[s, a]] )``.

    Args:
        n_states (int): state-space dimension ``S``.
        n_actions (int): action-space dimension ``A``.
        gamma (float): discount. ``1.0`` or supported ``2**-k`` values admit an
            identity/shift lowering; a general float is a declared boundary.
        tau (float): ``0`` for the hard backup (argmax, sparse gradient); ``>0``
            for the soft logsumexp relaxation (``tropical_softmax`` over actions).
        n_sweeps (int | None): number of backups per :meth:`forward` (the
            planning horizon). If ``None``, defaults to ``n_states`` (enough for
            a deterministic MDP with unit-cost steps to propagate values across
            the whole space).

    forward(V0, R, succ):
        V0:   (B, S) or (S,)         initial values.
        R:    (B, S, A) or (S, A)    reward / learnable potential.
        succ: (S, A) long            deterministic successor index (shared over batch).
    Returns ``(V, policy)`` with ``V: (..., S)`` and ``policy: (..., S)`` the
    greedy action argmax (the stored winner) after the final sweep.
    """

    def __init__(self, n_states, n_actions, gamma=1.0, tau=0.0, n_sweeps=None):
        super().__init__()
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.n_sweeps = int(n_sweeps) if n_sweeps is not None else int(n_states)

    def set_tau(self, tau):
        self.tau = float(tau)

    def _backup(self, V, R, succ):
        # Gather V at successors: q_next[..., s, a] = V[..., succ[s, a]].
        flat = succ.reshape(-1)                              # (S*A,)
        q_next = V.index_select(-1, flat).reshape(
            *V.shape[:-1], self.n_states, self.n_actions
        )                                                    # (..., S, A)
        q = R + self.gamma * q_next                          # gamma==1.0 -> no multiply
        if self.tau > 0:
            V_new = tropical_softmax(q, tau=self.tau, dim=-1)
            policy = q.argmax(dim=-1)
        else:
            V_new, policy = q.max(dim=-1)                    # sparse argmax backward
        return V_new, policy

    def forward(self, V0, R, succ):
        V = V0
        policy = None
        for _ in range(self.n_sweeps):
            V, policy = self._backup(V, R, succ)
        return V, policy


@torch.no_grad()
def _maxplus_range(delta):
    return (delta.max() - delta.min()).item()


def max_plus_power_iteration(A, n_iter=128, tol=1e-6, x0=None):
    """Max-plus eigenvalue and eigenvector of a square matrix ``A``.

    Iterates ``x_{k+1} = A (x)_maxplus x_k`` (i.e. ``x_{k+1}[i] = max_j(A[i,j] +
    x_k[j])``) with additive renormalisation. The eigenvalue ``lambda`` (the
    maximum mean cycle of the weighted digraph ``A``, equivalently the optimal
    average-reward gain) is read off as the asymptotic per-step additive growth;
    the renormalised ``x`` is the eigenvector (the relative-value function).

    Args:
        A (torch.Tensor): (S, S) max-plus matrix (use ``-inf`` for absent edges).
        n_iter (int): power-iteration budget.
        tol (float): convergence tolerance on the range of the per-step
            increment.
        x0 (torch.Tensor | None): optional (S,) start vector (default zeros).
    Returns ``(eigenvalue: float, eigenvector: (S,) tensor)``. Differentiable in
    ``A`` through :func:`max_plus_matmul`.
    """
    S = A.shape[-1]
    x = torch.zeros(S, dtype=A.dtype, device=A.device) if x0 is None else x0
    lam = 0.0
    for _ in range(int(n_iter)):
        x_new = max_plus_matmul(A, x.unsqueeze(-1)).squeeze(-1)   # (S,)
        delta = x_new - x
        lam = 0.5 * (delta.max() + delta.min())
        # renormalise (subtract the mean increment) to keep the iterate bounded
        x = x_new - 0.5 * (delta.max() + delta.min())
        if _maxplus_range(delta) < tol:
            break
    return float(lam), x
