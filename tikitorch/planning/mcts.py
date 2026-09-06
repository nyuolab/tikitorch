"""tikitorch.planning.mcts — tropical Monte-Carlo tree search.

MCTS over a learned-or-known model, with the two ingredients that are tropical:

  * **Selection** picks ``a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N)/(1+N_a) ]``
    -- a max reduction over the candidate (action) axis, the same argmax the HLS
    kernel's ``SCORE_N`` loop computes.
  * **Backup** composes ``V_parent <- backup_a(r + gamma V_child)`` along the
    search path. Three operators, selectable for ablation:
      - ``"max"``  : ``(max,+)`` Bellman step -- the pure tropical research bet,
        elegant in deterministic perfect-information planning but optimistic /
        overestimation-biased when the leaf value is noisy (a learned net).
      - ``"mean"`` : MuZero's published visit-average backup (linear, robust).
      - ``"topk"`` : the SOFTENED tropical backup -- a node's value is the mean of
        its ``topk_k`` largest backed-up returns. ``k=1`` recovers ``max``;
        ``k -> inf`` recovers ``mean``. Add/compare + a constant divide only (the
        tree analogue of ``tropical_value_pull_topk``, the library's A3 lever),
        so it interpolates between the tropical max and the robust mean and lets
        us tune away max-backup's noise fragility while staying tropical-leaning.

The PUCT exploration bonus (the ``sqrt`` and multiplies), value normalization
(a divide), and mean/top-k statistics are explicit arithmetic boundaries in the
executed selection/backup path. Tree statistics are held in a Python tree here
(the CPU reference); a flat-tensor / URAM-resident implementation is a proposed
hardware mapping, not an artifact supplied by this module.

The model is any object exposing the MuZero inference interface::

    initial_inference(state)            -> Inference(state, reward, policy, value)
    recurrent_inference(state, action)  -> (Inference(...), done)

(:class:`tikitorch.planning.muzero.env.KnownModel` wraps a deterministic env into
this interface for validating the search against exact value iteration.)
"""

import heapq
import math
import numbers
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["MCTSConfig", "TropicalMCTS"]


@dataclass
class MCTSConfig:
    num_simulations: int = 50
    c_puct: float = 1.25
    gamma: float = 1.0
    backup: str = "max"                 # "max" (tropical) | "mean" | "topk" (softened tropical)
    topk_k: int = 2                     # k for backup="topk" (1=max, large=mean)
    players: int = 1                    # 1 = single-agent; 2 = zero-sum alternating (negamax)
    dirichlet_alpha: float = 0.3
    root_exploration_frac: float = 0.0  # 0 -> deterministic (no root noise)
    seed: int = 0

    def __post_init__(self):
        if (
            isinstance(self.num_simulations, bool)
            or not isinstance(self.num_simulations, numbers.Integral)
            or self.num_simulations < 1
        ):
            raise ValueError("num_simulations must be a positive integer")
        if (
            isinstance(self.c_puct, bool)
            or not isinstance(self.c_puct, numbers.Real)
            or not math.isfinite(self.c_puct)
            or self.c_puct < 0
        ):
            raise ValueError("c_puct must be finite and non-negative")
        if (
            isinstance(self.gamma, bool)
            or not isinstance(self.gamma, numbers.Real)
            or not math.isfinite(self.gamma)
            or not 0 <= self.gamma <= 1
        ):
            raise ValueError("gamma must be finite and in [0, 1]")
        if self.backup not in ("max", "mean", "topk"):
            raise ValueError(
                f"backup must be 'max', 'mean', or 'topk', got {self.backup!r}"
            )
        if (
            isinstance(self.topk_k, bool)
            or not isinstance(self.topk_k, numbers.Integral)
            or self.topk_k < 1
        ):
            raise ValueError("topk_k must be a positive integer")
        if (
            isinstance(self.players, bool)
            or not isinstance(self.players, numbers.Integral)
            or self.players not in (1, 2)
        ):
            raise ValueError("players must be 1 or 2")
        if (
            isinstance(self.dirichlet_alpha, bool)
            or not isinstance(self.dirichlet_alpha, numbers.Real)
            or not math.isfinite(self.dirichlet_alpha)
            or self.dirichlet_alpha <= 0
        ):
            raise ValueError("dirichlet_alpha must be finite and positive")
        if (
            isinstance(self.root_exploration_frac, bool)
            or not isinstance(self.root_exploration_frac, numbers.Real)
            or not math.isfinite(self.root_exploration_frac)
            or not 0 <= self.root_exploration_frac <= 1
        ):
            raise ValueError("root_exploration_frac must be in [0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
            raise ValueError("seed must be an integer")

        # Canonical built-in scalars avoid surprises in ``range`` and NumPy RNGs
        # when callers supply numpy integer/float scalar types.
        self.num_simulations = int(self.num_simulations)
        self.c_puct = float(self.c_puct)
        self.gamma = float(self.gamma)
        self.topk_k = int(self.topk_k)
        self.players = int(self.players)
        self.dirichlet_alpha = float(self.dirichlet_alpha)
        self.root_exploration_frac = float(self.root_exploration_frac)
        self.seed = int(self.seed)


class _MinMaxStats:
    """Running min/max of observed Q values, to normalise the PUCT comparison."""

    def __init__(self):
        self.lo = float("inf")
        self.hi = -float("inf")

    def update(self, v):
        self.lo = min(self.lo, v)
        self.hi = max(self.hi, v)

    def normalize(self, v):
        if self.hi > self.lo:
            return (v - self.lo) / (self.hi - self.lo)   # declared host divide
        return v


class _Node:
    __slots__ = ("prior", "N", "value_sum", "value_max", "reward", "state", "children", "_topk")

    def __init__(self, prior):
        self.prior = float(prior)
        self.N = 0
        self.value_sum = 0.0
        self.value_max = -float("inf")
        self.reward = 0.0
        self.state = None
        self.children = {}
        self._topk = []          # min-heap holding the k largest backed-up returns

    def expanded(self):
        return len(self.children) > 0

    def record(self, g, backup, k):
        """Accumulate one backed-up return ``g`` into this node's statistics."""
        self.N += 1
        self.value_sum += g
        if g > self.value_max:
            self.value_max = g
        if backup == "topk":
            if len(self._topk) < k:
                heapq.heappush(self._topk, g)      # keep the k largest returns
            elif g > self._topk[0]:
                heapq.heapreplace(self._topk, g)

    def q(self, backup, k=1):
        if self.N == 0:
            return 0.0
        if backup == "max":
            return self.value_max
        if backup == "mean":
            return self.value_sum / self.N
        if backup == "topk":
            return sum(self._topk) / len(self._topk)   # mean of the k largest returns
        raise ValueError(f"unknown backup '{backup}'")


class TropicalMCTS:
    def __init__(self, model, config: MCTSConfig):
        self.model = model
        self.cfg = config

    # ----------------------------------------------------------------- public
    def run(self, root_state):
        """Run search from ``root_state``; return ``(visit_policy, root_value)``.

        ``visit_policy`` is the normalised root visit-count distribution over
        actions (the improved policy / MCTS target); ``root_value`` is the root's
        backed-up value (max or mean per the config).
        """
        cfg = self.cfg
        root = _Node(0.0)
        inf = self.model.initial_inference(root_state)
        root.state = inf.state
        self._n_actions = len(inf.policy)
        if self._n_actions < 1:
            raise ValueError("model policy must contain at least one action")
        self._expand(root, inf.policy, inf.legal)
        if not root.expanded():
            # An explicitly empty legal-action set is the model interface's
            # terminal-root signal. There is nothing to simulate or select.
            return torch.zeros(self._n_actions, dtype=torch.float), float(inf.value)
        if cfg.root_exploration_frac > 0:
            self._add_root_noise(root)

        minmax = _MinMaxStats()
        for _ in range(cfg.num_simulations):
            node = root
            path = [root]
            action = None
            while node.expanded():
                action, node = self._select_child(node, minmax)
                path.append(node)
            # `node` is a freshly-created, unevaluated leaf reached via `action`
            # from its parent path[-2].
            parent = path[-2]
            inf, done = self.model.recurrent_inference(parent.state, action)
            node.state = inf.state
            node.reward = inf.reward
            if not done:
                self._expand(node, inf.policy, inf.legal)
            self._backup(path, inf.value, minmax)

        A = self._n_actions
        counts = torch.tensor([root.children[a].N if a in root.children else 0
                               for a in range(A)], dtype=torch.float)
        visit_policy = counts / counts.sum().clamp_min(1.0)
        return visit_policy, root.q(cfg.backup, cfg.topk_k)

    # ---------------------------------------------------------------- internal
    def _expand(self, node, policy, legal=None):
        actions = range(len(policy)) if legal is None else list(legal)
        if any(
            not isinstance(a, numbers.Integral)
            or isinstance(a, (bool, np.bool_))
            or not 0 <= a < len(policy)
            for a in actions
        ):
            raise ValueError("legal actions must be integer indices into the policy")
        actions = [int(a) for a in actions]
        z = sum(float(policy[a]) for a in actions) or 1.0   # renormalise prior over legal
        for a in actions:
            node.children[a] = _Node(prior=float(policy[a]) / z)

    def _add_root_noise(self, root):
        children = list(root.children.values())    # legal children (sparse keys for board games)
        # Use a LOCAL seeded RNG — never mutate the process-global torch RNG (that broke
        # reproducibility under parallel self-play actors, and re-seeded to the same fixed
        # value every call so the noise was identical across moves). torch's
        # Dirichlet.sample() takes no generator=, so draw the symmetric Dirichlet from
        # numpy's seeded Generator (per-move distinct, since selfplay mutates cfg.seed;
        # isolated per actor process).
        rng = np.random.default_rng(self.cfg.seed)
        noise = rng.dirichlet([self.cfg.dirichlet_alpha] * len(children))
        frac = self.cfg.root_exploration_frac
        for child, ng in zip(children, noise):
            child.prior = child.prior * (1 - frac) + float(ng) * frac

    def _sign(self):
        # negamax: in a 2-player zero-sum game the child's value is from the OPPONENT's
        # perspective, so it enters the parent's edge value (and the backup) negated.
        return -1.0 if self.cfg.players == 2 else 1.0

    def _edge_value(self, child):
        # Q of the edge into `child` = reward(edge) + gamma * sign * V(child).
        return child.reward + self.cfg.gamma * self._sign() * child.q(self.cfg.backup, self.cfg.topk_k)

    def _select_child(self, node, minmax):
        cfg = self.cfg
        sqrtN = math.sqrt(max(node.N, 1))          # declared host sqrt
        best_score = -float("inf")
        best_a, best_child = None, None
        for a, child in sorted(node.children.items()):   # first-index tie-break (kernel order)
            q = minmax.normalize(self._edge_value(child)) if child.N > 0 else 0.0
            u = cfg.c_puct * child.prior * sqrtN / (1 + child.N)
            score = q + u                          # the load-bearing argmax
            if score > best_score:
                best_score, best_a, best_child = score, a, child
        return best_a, best_child

    def _backup(self, path, leaf_value, minmax):
        cfg = self.cfg
        sign = self._sign()
        g = float(leaf_value)
        for node in reversed(path):
            node.record(g, cfg.backup, cfg.topk_k)
            minmax.update(node.reward + cfg.gamma * sign * node.q(cfg.backup, cfg.topk_k))
            g = node.reward + cfg.gamma * sign * g  # negamax (sign=-1) / Bellman (sign=+1)
