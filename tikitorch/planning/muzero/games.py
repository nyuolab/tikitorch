"""tikitorch.planning.muzero.games — two-player perfect-information board games.

The deterministic perfect-information domain where the tropical / max-backup thesis
is strongest, and the canonical AlphaZero/MuZero setting (TicTacToe -> Connect4 ->
chess). This module provides ``TicTacToe`` (with an exact minimax oracle) and a
``TwoPlayerKnownModel`` adapter so the negamax ``TropicalMCTS`` (``players=2``) can
be checked against TicTacToe's exact oracle before any learned net or scale-up.

State is ``(board, to_play)`` with ``board`` a length-9 tuple (0 empty, 1/2 marks)
and ``to_play`` in {1, 2}. Actions are cell indices 0..8.
"""

from functools import lru_cache

import torch

from tikitorch.planning.muzero.env import Inference

__all__ = ["TicTacToe", "Connect4", "TwoPlayerKnownModel", "BoardGameModel", "AlphaZeroBoardModel",
           "negamax_agent", "play_match", "random_opening"]

_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
          (2, 5, 8), (0, 4, 8), (2, 4, 6)]


class TicTacToe:
    n_actions = 9

    @staticmethod
    def initial_state():
        return ((0,) * 9, 1)

    @staticmethod
    def legal_actions(board):
        return [i for i in range(9) if board[i] == 0]

    @staticmethod
    def winner(board):
        for a, b, c in _LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        return 0

    @staticmethod
    def apply(board, action, player):
        b = list(board)
        b[action] = player
        return tuple(b)

    @staticmethod
    def is_terminal(board):
        return TicTacToe.winner(board) != 0 or all(c != 0 for c in board)

    obs_dim = 18

    @staticmethod
    def obs(state):
        """Perspective-invariant obs: [own pieces, opponent pieces] (2x9 -> 18)."""
        board, tp = state
        opp = 3 - tp
        return torch.tensor([1.0 if x == tp else 0.0 for x in board]
                            + [1.0 if x == opp else 0.0 for x in board])

    @staticmethod
    @lru_cache(maxsize=None)
    def minimax(board, to_play):
        """Exact value from ``to_play``'s perspective: +1 win / 0 draw / -1 loss
        under optimal play (handles terminal states: a board with a winner means
        ``to_play`` was just beaten -> -1; a full board -> 0)."""
        if TicTacToe.winner(board) != 0:
            return -1                                  # opponent just made 3-in-a-row
        legal = TicTacToe.legal_actions(board)
        if not legal:
            return 0                                   # draw
        best = -2
        for a in legal:
            nb = TicTacToe.apply(board, a, to_play)
            best = max(best, -TicTacToe.minimax(nb, 3 - to_play))
        return best


def terminal_value(game, board, to_play):
    """Exact outcome from ``to_play``'s perspective at a terminal board: a winner
    means ``to_play`` was just beaten (-1); a full board is a draw (0)."""
    if game.winner(board) != 0:
        return -1.0
    return 0.0


class TwoPlayerKnownModel:
    """Adapts ANY two-player game to the MuZero inference interface for negamax MCTS.

    Leaf value is from the player-to-move's perspective. Terminal states always use
    the exact outcome. For non-terminal states: ``use_oracle=True`` uses ``value_fn``
    (default the game's exact ``minimax`` if it has one, e.g. TicTacToe; or a
    depth-limited negamax for Connect4); ``use_oracle=False`` uses 0 (pure search,
    bootstrapping from terminal outcomes). Reward is 0 (outcome lives in the value).
    """

    def __init__(self, game=TicTacToe, use_oracle=True, value_fn=None):
        self.game = game
        self.use_oracle = use_oracle
        self.A = game.n_actions
        if value_fn is None and hasattr(game, "minimax"):
            def value_fn(b, t):
                return float(game.minimax(b, t))
        self.value_fn = value_fn

    def _inf(self, state, reward, done):
        board, tp = state
        legal = [] if done else self.game.legal_actions(board)
        if done:
            val = terminal_value(self.game, board, tp)
        elif self.use_oracle and self.value_fn is not None:
            val = float(self.value_fn(board, tp))
        else:
            val = 0.0
        return Inference(state=state, reward=reward,
                         policy=torch.full((self.A,), 1.0 / self.A),
                         value=val, legal=legal or None)

    def initial_inference(self, state):
        return self._inf(state, 0.0, self.game.is_terminal(state[0]))

    def recurrent_inference(self, state, action):
        board, tp = state
        nb = self.game.apply(board, action, tp)
        nstate = (nb, 3 - tp)
        done = self.game.is_terminal(nb)
        return self._inf(nstate, 0.0, done), done


class Connect4:
    """6x7 Connect Four. Cells indexed r*7+c (r=0 top .. r=5 bottom); drop to lowest
    empty row. Deterministic perfect-information two-player -- the Connect4 head-to-head
    target. Full minimax is intractable (~4.5e12 states), so use the depth-limited
    :func:`negamax_value` as a skill opponent / tactical oracle (not exact from start)."""

    n_actions = 7
    H, W = 6, 7
    obs_dim = 84

    @staticmethod
    def obs(state):
        """Perspective-invariant obs: [own pieces, opponent pieces] (2x42 -> 84)."""
        board, tp = state
        opp = 3 - tp
        return torch.tensor([1.0 if x == tp else 0.0 for x in board]
                            + [1.0 if x == opp else 0.0 for x in board])

    @staticmethod
    def initial_state():
        return ((0,) * 42, 1)

    @staticmethod
    def legal_actions(board):
        return [c for c in range(7) if board[c] == 0]      # top cell of column empty

    @staticmethod
    def apply(board, col, player):
        b = list(board)
        for r in range(5, -1, -1):                          # lowest empty row in column
            if b[r * 7 + col] == 0:
                b[r * 7 + col] = player
                break
        return tuple(b)

    @staticmethod
    def winner(board):
        for r in range(6):
            for c in range(7):
                p = board[r * 7 + c]
                if p == 0:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):   # →, ↓, ↘, ↙
                    rr, cc = r + 3 * dr, c + 3 * dc
                    if 0 <= rr < 6 and 0 <= cc < 7 and all(
                            board[(r + i * dr) * 7 + (c + i * dc)] == p for i in range(4)):
                        return p
        return 0

    @staticmethod
    def is_terminal(board):
        return Connect4.winner(board) != 0 or all(board[c] != 0 for c in range(7))


def negamax_value(game, board, to_play, depth):
    """Depth-limited negamax: exact value (+1/0/-1) when the line resolves within
    ``depth``, else 0 (a heuristic-free cutoff). Exact for TicTacToe at depth>=9;
    a tactical oracle / skill opponent for Connect4 at modest depth."""
    if game.winner(board) != 0:
        return -1.0                                         # opponent just won
    legal = game.legal_actions(board)
    if not legal:
        return 0.0
    if depth == 0:
        return 0.0
    best = -2.0
    for a in legal:
        nb = game.apply(board, a, to_play)
        best = max(best, -negamax_value(game, nb, 3 - to_play, depth - 1))
        if best == 1.0:
            break                                           # found a forced win
    return best


def negamax_agent(game, depth):
    """A fixed-strength opponent: pick the move maximizing the depth-limited negamax."""
    def act(state):
        board, tp = state
        best_a, best_v = None, -2.0
        for a in game.legal_actions(board):
            v = -negamax_value(game, game.apply(board, a, tp), 3 - tp, depth - 1)
            if v > best_v:
                best_v, best_a = v, a
        return best_a
    return act


class BoardGameModel:
    """Wraps a LEARNED MuZero net + a game for two-player MCTS.

    At the ROOT (a real game state) it encodes the board to an obs, runs the net's
    initial inference, and injects the game's legal moves (legality is known only at
    the real board). At INTERNAL tree nodes it uses the net's learned latent dynamics
    (no legality / terminal info -- the standard MuZero board-game setup, where the
    policy learns to avoid illegal moves and the value net learns end-game values).
    """

    def __init__(self, net, game):
        self.net = net
        self.game = game
        self.A = game.n_actions

    def initial_inference(self, state):
        board, _ = state
        inf = self.net.initial_inference(self.game.obs(state))
        inf.legal = self.game.legal_actions(board)
        return inf

    def recurrent_inference(self, latent, action):
        return self.net.recurrent_inference(latent, action)


class AlphaZeroBoardModel:
    """Wrap a net + game for AlphaZero-style MCTS using explicit game state in the tree.

    Unlike :class:`BoardGameModel` (pure MuZero — learned latent dynamics inside the tree, legality
    only at the root), this evaluates every node at the state represented by the game object:
    transitions, terminal outcomes, and legal moves are supplied at every node. The net supplies the
    policy prior and leaf value. This can avoid learned-model and illegal-action errors, but deeper
    search is not guaranteed to improve every position or weak prior. Nor does integer-heavy game
    logic make the complete system zero-DSP: the Python MCTS path includes PUCT multiplication,
    square root, min-max division, and floating-point tree statistics, and the configured net may
    contain dense or categorical arithmetic boundaries."""

    def __init__(self, net, game):
        self.net = net
        self.game = game
        self.A = game.n_actions

    def _eval(self, state, done):
        board, tp = state
        if done:                                     # exact terminal value from tp's perspective
            return Inference(state=state, reward=0.0,
                             policy=torch.full((self.A,), 1.0 / self.A),
                             value=terminal_value(self.game, board, tp), legal=None)
        inf = self.net.initial_inference(self.game.obs(state))   # net prior+value at the REAL state
        inf.state = state                            # node carries the true state (not a latent)
        inf.legal = self.game.legal_actions(board)   # true legality at EVERY node
        return inf

    def initial_inference(self, state):
        return self._eval(state, self.game.is_terminal(state[0]))

    def recurrent_inference(self, state, action):
        board, tp = state
        nb = self.game.apply(board, action, tp)
        nstate = (nb, 3 - tp)
        done = self.game.is_terminal(nb)
        return self._eval(nstate, done), done


def random_opening(game, rng, n_plies):
    """A state after ``n_plies`` uniform-random legal moves from the start (stops early at terminal).

    Used to begin EVAL games from DIVERSE positions: a deterministic agent vs a deterministic
    opponent played from the single fixed start collapses to one game per color (a silent eval
    artifact that quantizes the score and hides gradual skill); random openings give distinct games.
    """
    state = game.initial_state()
    for _ in range(n_plies):
        if game.is_terminal(state[0]):
            break
        board, tp = state
        legal = game.legal_actions(board)
        a = legal[int(torch.randint(0, len(legal), (1,), generator=rng).item())]
        state = (game.apply(board, a, tp), 3 - tp)
    return state


def play_match(agent_x, agent_y, game=TicTacToe, start_state=None):
    """Play one full game from ``start_state`` (default the initial state); ``agent_*`` are callables
    state->action keyed by player (agent_x = player 1, agent_y = player 2). Returns winner (1/2) or 0."""
    state = start_state if start_state is not None else game.initial_state()
    agents = {1: agent_x, 2: agent_y}
    while not game.is_terminal(state[0]):
        board, tp = state
        a = agents[tp](state)
        state = (game.apply(board, a, tp), 3 - tp)
    return game.winner(state[0])
