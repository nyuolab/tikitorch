"""tikitorch.planning — tropical dynamic programming, value iteration, and search.

Max-plus / min-plus is the native algebra of dynamic programming and planning:
value iteration ``V <- max_a(R + gamma V)`` is a ``(max,+)`` matrix-vector
product, shortest path is repeated min-plus matmul, and Viterbi / DTW are
``(max/min,+)`` trellis recurrences. This submodule provides hard tropical
primitives with compare/add/gather software recipes, plus soft relaxations and
planning/RL reference code that deliberately retain additional arithmetic.

Do not infer whole-module or FPGA resource closure from the package name: soft
reductions, discounting, MCTS selection/normalization, losses, and learned-model
boundaries must be audited for the exact configuration, and a physical DSP claim
requires a matching synthesis report.

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
    TropicalCoinChangeDP,
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
    "TropicalCoinChangeDP",
    # tropical Monte-Carlo tree search
    "MCTSConfig",
    "TropicalMCTS",
]
