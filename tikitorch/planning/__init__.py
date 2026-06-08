"""tikitorch.planning — tropical dynamic programming, value iteration, and search.

Max-plus / min-plus is the native algebra of dynamic programming and planning:
value iteration ``V <- max_a(R + gamma V)`` is a ``(max,+)`` matrix-vector
product, shortest path is repeated min-plus matmul, and Viterbi / DTW are
``(max/min,+)`` trellis recurrences. This submodule provides those primitives as
differentiable, multiplier-free TikiTorch layers -- the planning/RL application
layer, building toward a tropical MCTS / Tropical MuZero (added in later stages).

Everything here reduces to comparisons and additions only, so it maps onto the
same inlined max-plus PE used by the FORGE HLS training kernels.

    import tikitorch.planning as planning

    vi = planning.TropicalValueIteration(n_states=S, n_actions=A, gamma=1.0)
    V, policy = vi(V0, R, succ)

    sp = planning.TropicalShortestPath()
    dist = sp(adj, src=0)
"""

from tikitorch.planning.functional import (  # noqa: F401
    TropicalScanFn,
    tropical_scan,
    tropical_scan_soft,
    tropical_scan_decode,
    MaxPlusMatmulFn,
    max_plus_matmul,
)
from tikitorch.planning.bellman import (  # noqa: F401
    TropicalValueIteration,
    max_plus_power_iteration,
)
from tikitorch.planning.dp import (  # noqa: F401
    TropicalViterbi,
    TropicalShortestPath,
)
from tikitorch.planning.mcts import MCTSConfig, TropicalMCTS  # noqa: F401

__all__ = [
    # functional DP primitives
    "TropicalScanFn",
    "tropical_scan",
    "tropical_scan_soft",
    "tropical_scan_decode",
    "MaxPlusMatmulFn",
    "max_plus_matmul",
    # value iteration / max-plus spectra
    "TropicalValueIteration",
    "max_plus_power_iteration",
    # differentiable DP layers
    "TropicalViterbi",
    "TropicalShortestPath",
    # tropical Monte-Carlo tree search
    "MCTSConfig",
    "TropicalMCTS",
]
