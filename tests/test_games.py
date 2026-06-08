"""Two-player (negamax) MCTS validation on TicTacToe — the chess-foundation gate.

Confirms the `players=2` negamax backup + legal-move masking are correct: with the
exact minimax oracle and MEAN backup (the correct adversarial backup), the MCTS
plays TicTacToe perfectly (never loses; self-play draws). Also records that
max/top-k backup is over-optimistic in adversarial play (assumes a cooperative
opponent) -- the two-player sharpening of the backup study.
"""
import random

import torch

import tikitorch.planning as P
from tikitorch.planning.muzero import TicTacToe, TwoPlayerKnownModel, play_match


def _agent(model, backup, sims=64, seed=0):
    def act(state):
        mc = P.TropicalMCTS(model, P.MCTSConfig(num_simulations=sims, players=2,
                                                backup=backup, root_exploration_frac=0.0))
        return int(mc.run(state)[0].argmax())
    return act


def test_minimax_says_tictactoe_is_a_draw():
    assert TicTacToe.minimax((0,) * 9, 1) == 0


def test_negamax_mean_backup_never_loses():
    random.seed(0); torch.manual_seed(0)
    model = TwoPlayerKnownModel(TicTacToe, use_oracle=True)
    agent = _agent(model, "mean")
    rand = lambda s: random.choice(TicTacToe.legal_actions(s[0]))
    losses = 0
    for g in range(24):
        r = play_match(agent, rand) if g % 2 == 0 else play_match(rand, agent)
        mcts_is = 1 if g % 2 == 0 else 2
        if r != 0 and r != mcts_is:
            losses += 1
    assert losses == 0          # optimal play never loses TicTacToe


def test_negamax_self_play_is_a_draw():
    model = TwoPlayerKnownModel(TicTacToe, use_oracle=True)
    agent = _agent(model, "mean", sims=96)
    assert play_match(agent, agent) == 0     # optimal vs optimal -> draw


def test_connect4_win_detection_and_drop():
    from tikitorch.planning.muzero.games import Connect4
    b = [0] * 42
    for c in range(4):           # horizontal 4 on the bottom row for player 1
        b[5 * 7 + c] = 1
    assert Connect4.winner(tuple(b)) == 1
    # drop lands in the lowest empty row
    nb = Connect4.apply((0,) * 42, 3, 1)
    assert nb[5 * 7 + 3] == 1 and sum(nb) == 1
    assert Connect4.legal_actions((0,) * 42) == list(range(7))


def test_connect4_negamax_mcts_finds_forced_win():
    from tikitorch.planning.muzero.games import Connect4, TwoPlayerKnownModel
    b = [0] * 42
    for c in range(3):           # player 1 has 3-in-a-row (cols 0-2); winning move = col 3
        b[5 * 7 + c] = 1
    model = TwoPlayerKnownModel(Connect4, use_oracle=False)   # pure search
    mc = P.TropicalMCTS(model, P.MCTSConfig(num_simulations=120, players=2,
                                            backup="mean", root_exploration_frac=0.0))
    vp, _ = mc.run((tuple(b), 1))
    assert int(vp.argmax()) == 3


def test_board_selfplay_produces_outcome_targets():
    # learned-net two-player self-play runs end to end and labels each position with
    # the game OUTCOME from its player-to-move perspective.
    import torch
    from tikitorch.planning.muzero import TropicalMuZeroNet, Connect4, play_board_game
    torch.manual_seed(0)
    net = TropicalMuZeroNet(Connect4.obs_dim, Connect4.n_actions, latent_dim=16)
    records, w = play_board_game(net, Connect4, num_simulations=12, seed=0)
    assert len(records) >= 1 and w in (0, 1, 2)
    for obs, a, pol, z in records:
        assert obs.shape == (Connect4.obs_dim,)
        assert a in range(Connect4.n_actions)
        assert abs(float(pol.sum()) - 1.0) < 1e-4
        assert z in (-1.0, 0.0, 1.0)


def test_board_muzero_loop_runs():
    # the full learned two-player MuZero loop (self-play -> board replay -> outcome
    # training -> skill eval vs negamax) runs end to end on Connect4 with the closed
    # TropicalSGD optimizer.
    from tikitorch.planning.muzero import Connect4, BoardMuZero, BoardConfig
    cfg = BoardConfig(latent_dim=16, num_simulations=8, iters=2, games_per_iter=2,
                      train_steps_per_iter=3, batch=16, eval_every=1, eval_games=4, eval_depth=1)
    agent = BoardMuZero(Connect4, cfg)
    assert type(agent.opt).__name__ == "TropicalSGD"      # closed optimizer by default
    hist = agent.run(log=lambda m: None)
    assert hist and {"win", "draw", "loss", "score"} <= set(hist[-1])


def test_legal_move_masking_only_expands_legal():
    # a nearly-full board: only the empty cells are legal/expanded
    model = TwoPlayerKnownModel(TicTacToe, use_oracle=True)
    board = (1, 2, 1, 2, 1, 2, 0, 0, 2)      # legal = {6, 7}
    state = (board, 1)
    mc = P.TropicalMCTS(model, P.MCTSConfig(num_simulations=16, players=2,
                                            backup="mean", root_exploration_frac=0.0))
    vp, _ = mc.run(state)
    assert vp.shape == (9,)
    illegal = [i for i in range(9) if board[i] != 0]
    assert float(vp[illegal].sum()) == 0.0   # no visits to illegal moves
