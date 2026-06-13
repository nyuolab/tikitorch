"""mzgeneral_chessgame — a chess game module for muzero-general (Werner Duvaud).

muzero-general ships no chess env; this adds one so we can measure a DENSE MuZero chess
TRAINING baseline on GPU (throughput + energy), the systems half of CHESS_MUZERO_COMPARISON.md
(axes 2-3). The runner (run_mzgeneral_chess_gpu.sh) drops this into a muzero-general clone as
`games/chessgame.py` and runs `python muzero.py chessgame`.

It mirrors TikiTorch's chess encoding for a fair comparison:
  * action = from*64 + to in [0, 4096); promotions collapse to a QUEEN promotion
    (underpromotions not representable) — identical to tikitorch.planning.muzero.chess_game.
  * observation = (13, 8, 8): 12 piece planes (white P,N,B,R,Q,K then black) + 1 side-to-move plane.
    (TikiTorch uses the flat 769-d form of the same planes; muzero-general's resnet wants 3D.)

This module's PURPOSE is a representative systems measurement (self-play moves/s, train samples/s,
GPU energy/move), NOT a strength run — DeepMind chess is the literature ceiling, not a GPU contest.

Requires python-chess (`pip install chess`). Standalone self-test: `python scripts/mzgeneral_chessgame.py`.
"""
import datetime
import pathlib

import numpy

try:
    import torch
    _HAS_TORCH = True
except Exception:                       # the standalone self-test doesn't need torch
    _HAS_TORCH = False

import chess

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = 1

        ### Game
        self.observation_shape = (13, 8, 8)            # 12 piece planes + 1 side-to-move plane
        self.action_space = list(range(64 * 64))       # from*64 + to, queen-promo (matches TikiTorch)
        self.players = list(range(2))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = "random"                       # cheap progress check; doesn't affect training

        ### Self-Play
        self.num_workers = 8                            # CPU self-play actors (the runner tunes to vCPUs)
        self.selfplay_on_gpu = False                   # self-play net inference on CPU (actor); train on GPU (learner)
        self.max_moves = 80                            # match TikiTorch's ply cap
        self.num_simulations = 50                      # match the TikiTorch chess sweep (~48); report it
        self.discount = 1
        self.temperature_threshold = None
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network — representative moderate chess resnet (report the size with the throughput number)
        self.network = "resnet"
        self.support_size = 2                           # rewards/values in [-1, 1]
        self.downsample = False
        self.blocks = 6
        self.channels = 64
        self.reduced_channels_reward = 2
        self.reduced_channels_value = 2
        self.reduced_channels_policy = 4
        self.resnet_fc_reward_layers = [64]
        self.resnet_fc_value_layers = [64]
        self.resnet_fc_policy_layers = [64]
        # (fullyconnected fallback params, unused with network="resnet")
        self.encoding_size = 32
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [64]
        self.fc_value_layers = []
        self.fc_policy_layers = []

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 1000000                  # we stop via a wall-clock window in the runner, not this
        self.batch_size = 128
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = _HAS_TORCH and torch.cuda.is_available()
        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = 0.003
        self.lr_decay_rate = 1
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 10000
        self.num_unroll_steps = 5
        self.td_steps = 80                             # board game: bootstrap to game end (<= max_moves)
        self.PER = True
        self.PER_alpha = 0.5
        self.use_last_model_value = True
        self.reanalyse_on_gpu = True

        ### self-play / training ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None

    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 250e3:
            return 1.0
        elif trained_steps < 500e3:
            return 0.5
        return 0.25


def _board_to_obs(board):
    """(13, 8, 8) float32: 12 piece planes (white P,N,B,R,Q,K then black) + side-to-move plane."""
    obs = numpy.zeros((13, 8, 8), dtype="float32")
    ch = 0
    for color in (chess.WHITE, chess.BLACK):
        for pt in _PIECE_ORDER:
            for sq in board.pieces(pt, color):
                obs[ch, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            ch += 1
    obs[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return obs


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def to_play(self):
        return 0 if self.board.turn == chess.WHITE else 1

    def reset(self):
        self.board = chess.Board()
        return _board_to_obs(self.board)

    def _decode(self, action):
        frm, to = action // 64, action % 64
        fallback = None
        for m in self.board.legal_moves:
            if m.from_square == frm and m.to_square == to:
                if m.promotion in (None, chess.QUEEN):
                    return m
                fallback = fallback or m
        return fallback if fallback is not None else chess.Move(frm, to)

    def step(self, action):
        self.board.push(self._decode(action))
        done = self.board.is_game_over(claim_draw=True)
        # reward is for the player who JUST moved: +1 if that move checkmated, else 0 (draws/ongoing).
        reward = 1.0 if self.board.is_checkmate() else 0.0
        return _board_to_obs(self.board), reward, done

    def legal_actions(self):
        return sorted({m.from_square * 64 + m.to_square for m in self.board.legal_moves})

    def render(self):
        print(self.board)


# --- muzero-general AbstractGame wrapper (importlib loads `Game` + `MuZeroConfig`) ---
try:
    from .abstract_game import AbstractGame      # when imported as games.chessgame
except Exception:
    try:
        from abstract_game import AbstractGame
    except Exception:
        AbstractGame = object                    # standalone self-test path


class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = ChessEnv()

    def step(self, action):
        return self.env.step(action)

    def to_play(self):
        return self.env.to_play()

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def action_to_string(self, action_number):
        frm, to = action_number // 64, action_number % 64
        return chess.square_name(frm) + chess.square_name(to)


if __name__ == "__main__":
    # standalone correctness self-test (needs only python-chess): random rollout to a terminal/cap
    import random
    env = ChessEnv()
    obs = env.reset()
    assert obs.shape == (13, 8, 8), obs.shape
    assert len(env.legal_actions()) == 20, "start position has 20 legal moves"  # 16 pawn + 4 knight
    cfg = MuZeroConfig()
    assert len(cfg.action_space) == 4096 and cfg.observation_shape == (13, 8, 8)
    random.seed(0)
    plies = 0
    while plies < cfg.max_moves:
        la = env.legal_actions()
        if not la:
            break
        obs, r, done = env.step(random.choice(la))
        assert obs.shape == (13, 8, 8)
        plies += 1
        if done:
            break
    print(f"OK: self-test ran {plies} plies, terminal={done if 'done' in dir() else 'n/a'}, "
          f"final reward={r}, obs{obs.shape}, action_space={len(cfg.action_space)}, "
          f"start_legal=20, sims={cfg.num_simulations}, net=resnet b{cfg.blocks}c{cfg.channels}")
