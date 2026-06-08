"""tikitorch.planning.muzero.replay — trajectory replay buffer + MuZero targets.

Stores self-play games and samples K-step unroll batches with the standard MuZero
targets: n-step bootstrapped value (using the stored MCTS root values), observed
reward, and the MCTS visit-count policy. Positions past episode end are padded
with an absorbing target (value 0, reward 0, uniform policy).
"""

from dataclasses import dataclass, field
from typing import List

import torch

__all__ = ["Game", "ReplayBuffer"]


@dataclass
class Game:
    """One self-play trajectory (parallel lists; ``rewards[t]`` follows ``actions[t]``)."""
    obs: List[torch.Tensor] = field(default_factory=list)        # obs at each step (flat)
    actions: List[int] = field(default_factory=list)            # action taken at each step
    rewards: List[float] = field(default_factory=list)          # reward AFTER that action
    policies: List[torch.Tensor] = field(default_factory=list)  # MCTS visit dist (A,)
    root_values: List[float] = field(default_factory=list)      # MCTS root value at each step

    def __len__(self):
        return len(self.actions)


class ReplayBuffer:
    def __init__(self, capacity=2000, n_actions=4):
        self.capacity = capacity
        self.n_actions = n_actions
        self.games: List[Game] = []

    def save(self, game: Game):
        self.games.append(game)
        if len(self.games) > self.capacity:
            self.games.pop(0)

    def __len__(self):
        return len(self.games)

    @property
    def n_steps(self):
        return sum(len(g) for g in self.games)

    def _value_target(self, g: Game, pos: int, td: int, gamma: float) -> float:
        L = len(g)
        if pos >= L:
            return 0.0
        # bootstrap from the stored MCTS root value td steps ahead (if within episode)
        boot = (gamma ** td) * g.root_values[pos + td] if pos + td < L else 0.0
        acc = 0.0
        for i in range(td):
            if pos + i < L:
                acc += (gamma ** i) * g.rewards[pos + i]
            else:
                break
        return acc + boot

    def sample_batch(self, batch: int, unroll: int, td: int, gamma: float, gen=None):
        """Return obs0 (B,obs_dim), actions (B,K) long, value_targets (B,K+1),
        reward_targets (B,K), policy_targets (B,K+1,A)."""
        A = self.n_actions
        obs0, acts, vts, rts, pts = [], [], [], [], []
        lengths = [len(g) for g in self.games]
        tot = sum(lengths)
        for _ in range(batch):
            # sample a position proportional to game length
            r = int(torch.randint(0, max(tot, 1), (1,), generator=gen).item())
            gi = 0
            while r >= lengths[gi]:
                r -= lengths[gi]
                gi += 1
            g, pos = self.games[gi], r

            obs0.append(g.obs[pos])
            a_k, v_k, r_k, p_k = [], [self._value_target(g, pos, td, gamma)], [], [g.policies[pos]]
            for k in range(1, unroll + 1):
                p = pos + k
                a_k.append(g.actions[p - 1] if p - 1 < len(g) else 0)
                r_k.append(g.rewards[p - 1] if p - 1 < len(g) else 0.0)
                v_k.append(self._value_target(g, p, td, gamma))
                p_k.append(g.policies[p] if p < len(g) else torch.full((A,), 1.0 / A))
            acts.append(torch.tensor(a_k, dtype=torch.long))
            vts.append(torch.tensor(v_k))
            rts.append(torch.tensor(r_k))
            pts.append(torch.stack(p_k))
        return (torch.stack(obs0), torch.stack(acts), torch.stack(vts),
                torch.stack(rts), torch.stack(pts))
