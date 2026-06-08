"""tikitorch.planning.mcts — tropical Monte-Carlo tree search.

MCTS over a learned-or-known model, with the two ingredients the FORGE brief
flags as tropical:

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

The PUCT exploration bonus (the ``sqrt`` and the multiply) and the value
normalisation (a divide) are *declared host/DSP boundaries* -- small, per-node,
off the load-bearing argmax/backup path. Tree statistics are held in a Python
tree here (the CPU reference); the flat-tensor / URAM-resident layout is the
Stage-7 hardware mapping.

The model is any object exposing the MuZero inference interface::

    initial_inference(state)            -> Inference(state, reward, policy, value)
    recurrent_inference(state, action)  -> (Inference(...), done)

(:class:`tikitorch.planning.muzero.env.KnownModel` wraps a deterministic env into
this interface for validating the search against exact value iteration.)
"""

import heapq
import math
from dataclasses import dataclass

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
        self._expand(root, inf.policy, inf.legal)
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
        z = sum(float(policy[a]) for a in actions) or 1.0   # renormalise prior over legal
        for a in actions:
            node.children[a] = _Node(prior=float(policy[a]) / z)

    def _add_root_noise(self, root):
        children = list(root.children.values())    # legal children (sparse keys for board games)
        torch.manual_seed(self.cfg.seed)           # reproducible root exploration noise
        noise = torch.distributions.Dirichlet(
            torch.full((len(children),), self.cfg.dirichlet_alpha)).sample()
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
