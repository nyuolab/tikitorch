"""tikitorch.planning.muzero.env — deterministic perfect-information domains.

These are the planning domains where a *max* (rather than mean / visit-average)
backup is well-motivated -- the value of a state is its cost-to-go, which is
exactly a ``(max,+)`` Bellman quantity. They are deliberately small so the whole
search/learning loop runs on CPU for smoke tests and scales on a single GPU.

``GridWorldNav`` is the stepping-stone: it threads the whole chain, because the
SAME environment validates Stage-1 ``TropicalValueIteration`` (it exposes its
exact ``(R, succ)`` MDP), Stage-3 ``TropicalMCTS`` (a known model), and Stage-5
learned-model MuZero.

``KnownModel`` adapts an environment to the MuZero inference interface used by
``TropicalMCTS`` (``initial_inference`` / ``recurrent_inference``), so the search
engine is tested against a perfect model before any network is learned.
"""

from dataclasses import dataclass

import torch

# Action encoding: 0=up, 1=down, 2=left, 3=right (+ 4 diagonals when diagonal=True).
_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class GridWorldNav:
    """Deterministic grid navigation with walls and a single goal.

    Reward is ``-1`` per step (the goal-entering step included); the goal is then
    absorbing with reward 0. So the optimal value of a state is exactly the
    negative shortest-path length to the goal -- the cost-to-go that
    ``TropicalValueIteration`` / ``TropicalShortestPath`` compute exactly. Moving
    into a wall or off-grid is a no-op (still costs a step).

    Args:
        grid: list[str] map. ``'#'`` wall, ``'.'`` free, ``'G'`` goal, ``'S'``
            optional default start.
    """

    n_actions = 4

    def __init__(self, grid, diagonal=False):
        self.grid = [row for row in grid]
        self.moves = _MOVES + _DIAG if diagonal else _MOVES
        self.n_actions = len(self.moves)        # 4, or 8 with diagonals (A != 4 check)
        self.H = len(grid)
        self.W = len(grid[0])
        self.walls = set()
        self.goal = None
        self.start = None
        for r in range(self.H):
            for c in range(self.W):
                ch = grid[r][c]
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "G":
                    self.goal = (r, c)
                elif ch == "S":
                    self.start = (r, c)
        assert self.goal is not None, "grid needs a goal 'G'"
        # Enumerate non-wall cells as states.
        self.cells = [(r, c) for r in range(self.H) for c in range(self.W)
                      if (r, c) not in self.walls]
        self.cell_to_state = {cell: i for i, cell in enumerate(self.cells)}
        self.n_states = len(self.cells)
        self.goal_state = self.cell_to_state[self.goal]
        self._pos = self.start if self.start is not None else self.cells[0]

    # ------------------------------------------------------------------ MDP
    def is_terminal_state(self, s):
        return s == self.goal_state

    def _move(self, cell, a):
        if cell == self.goal:
            return cell  # absorbing
        dr, dc = self.moves[a]
        nr, nc = cell[0] + dr, cell[1] + dc
        if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in self.walls:
            return (nr, nc)
        return cell  # bumped a wall / edge

    def step_state(self, s, a):
        """(next_state, reward, done) for the tabular MDP."""
        cell = self.cells[s]
        if cell == self.goal:
            return s, 0.0, True
        ncell = self._move(cell, a)
        ns = self.cell_to_state[ncell]
        done = ncell == self.goal
        return ns, -1.0, done   # every step costs -1, so V* = -shortest_path_length

    def as_mdp(self):
        """Return ``(R, succ, terminal)`` tensors for exact value iteration.

        ``R[s, a]`` reward, ``succ[s, a]`` successor state (long), ``terminal[s]``
        bool. The goal is absorbing with reward 0.
        """
        S, A = self.n_states, self.n_actions
        R = torch.zeros(S, A)
        succ = torch.zeros(S, A, dtype=torch.long)
        terminal = torch.zeros(S, dtype=torch.bool)
        for s in range(S):
            terminal[s] = self.is_terminal_state(s)
            for a in range(A):
                ns, r, _ = self.step_state(s, a)
                R[s, a] = r
                succ[s, a] = ns
        return R, succ, terminal

    # ------------------------------------------------------------ rollout API
    def reset(self, state=None):
        if state is None:
            self._pos = self.start if self.start is not None else self.cells[0]
        else:
            self._pos = self.cells[state]
        return self.cell_to_state[self._pos]

    def step(self, a):
        s = self.cell_to_state[self._pos]
        ns, r, done = self.step_state(s, a)
        self._pos = self.cells[ns]
        return ns, r, done

    def obs(self, s):
        """One-hot (3, H, W) observation: [agent, wall, goal] channels."""
        x = torch.zeros(3, self.H, self.W)
        r, c = self.cells[s]
        x[0, r, c] = 1.0
        for (wr, wc) in self.walls:
            x[1, wr, wc] = 1.0
        gr, gc = self.goal
        x[2, gr, gc] = 1.0
        return x


@dataclass
class Inference:
    """One MuZero inference step (a known model or a learned net produces this)."""
    state: object          # latent / tabular state id
    reward: float          # reward to arrive (0 at the root)
    policy: torch.Tensor   # prior over actions, shape (A,)
    value: float           # leaf value estimate
    legal: object = None   # optional iterable of legal action ids (board games); None = all legal


class KnownModel:
    """Adapts a deterministic env to the MuZero inference interface.

    ``value_fn(state)`` supplies the leaf value (e.g. a value-iteration oracle for
    Stage-3 validation, or ``0`` to force MCTS to bootstrap from realised rewards).
    ``prior_fn(state)`` supplies the policy prior (default uniform).
    """

    def __init__(self, env, value_fn=None, prior_fn=None, gamma=1.0):
        self.env = env
        self.A = env.n_actions
        self.gamma = gamma
        self.value_fn = value_fn if value_fn is not None else (lambda s: 0.0)
        self.prior_fn = prior_fn if prior_fn is not None else (
            lambda s: torch.full((self.A,), 1.0 / self.A))

    def initial_inference(self, state):
        return Inference(state=state, reward=0.0,
                         policy=self.prior_fn(state), value=float(self.value_fn(state)))

    def recurrent_inference(self, state, action):
        ns, reward, done = self.env.step_state(state, action)
        # A terminal successor has value 0 (no further reward).
        value = 0.0 if done else float(self.value_fn(ns))
        return Inference(state=ns, reward=float(reward),
                         policy=self.prior_fn(ns), value=value), done


# A small default map used by tests / smoke runs.
DEFAULT_GRID = [
    "S....",
    ".###.",
    ".#...",
    ".#.#.",
    "...#G",
]
