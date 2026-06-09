"""tikitorch.planning.muzero — Tropical MuZero.

Provides the deterministic perfect-information environments used to validate
the tropical MCTS engine against exact value iteration, plus the learned
representation / dynamics / prediction networks, self-play, replay, and the full
training loop.
"""

from tikitorch.planning.muzero.env import GridWorldNav, KnownModel  # noqa: F401
from tikitorch.planning.muzero.networks import (  # noqa: F401
    TropicalRepresentation,
    TropicalDynamics,
    TropicalPrediction,
    TropicalMuZeroNet,
)
from tikitorch.planning.muzero.networks_dense import DenseMuZeroNet  # noqa: F401
from tikitorch.planning.muzero.games import (  # noqa: F401
    TicTacToe,
    Connect4,
    TwoPlayerKnownModel,
    BoardGameModel,
    negamax_agent,
    play_match,
)
# Chess needs the optional ``python-chess`` dependency; surface it only if installed
# (the package must still import on a stock environment without it).
try:
    from tikitorch.planning.muzero.chess_game import Chess  # noqa: F401
    _HAS_CHESS = True
except ImportError:                                          # pragma: no cover
    Chess = None
    _HAS_CHESS = False
from tikitorch.planning.muzero.replay import Game, ReplayBuffer  # noqa: F401
from tikitorch.planning.muzero.selfplay import play_game, play_board_game  # noqa: F401
from tikitorch.planning.muzero.agent import MuZeroConfig, TropicalMuZero  # noqa: F401
from tikitorch.planning.muzero.board_agent import BoardConfig, BoardReplay, BoardMuZero  # noqa: F401

__all__ = [
    "GridWorldNav",
    "KnownModel",
    "TropicalRepresentation",
    "TropicalDynamics",
    "TropicalPrediction",
    "TropicalMuZeroNet",
    "DenseMuZeroNet",
    "TicTacToe",
    "Connect4",
    "Chess",
    "TwoPlayerKnownModel",
    "BoardGameModel",
    "negamax_agent",
    "play_match",
    "play_board_game",
    "Game",
    "ReplayBuffer",
    "play_game",
    "MuZeroConfig",
    "TropicalMuZero",
    "BoardConfig",
    "BoardReplay",
    "BoardMuZero",
]
