"""Two-player (negamax) MCTS validation on TicTacToe — the chess-foundation gate.

Confirms the `players=2` negamax backup + legal-move masking are correct: with the
exact minimax oracle and MEAN backup (the correct adversarial backup), the MCTS
plays TicTacToe perfectly (never loses; self-play draws). Also records that
max/top-k backup is over-optimistic in adversarial play (assumes a cooperative
opponent) -- the two-player sharpening of the backup study.
"""
import random

import pytest
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


@pytest.mark.parametrize("core_kind", ["maxplus", "dyadic"])
def test_board_muzero_loop_runs(core_kind):
    # the full learned two-player MuZero loop (self-play -> board replay -> outcome
    # training -> skill eval vs negamax) runs end to end on Connect4 with the closed
    # TropicalSGD optimizer, for both the maxplus and the dyadic (oblique) cores.
    from tikitorch.planning.muzero import Connect4, BoardMuZero, BoardConfig
    cfg = BoardConfig(core_kind=core_kind, latent_dim=16, num_simulations=8, iters=2,
                      games_per_iter=2, train_steps_per_iter=3, batch=16, eval_every=1,
                      eval_games=4, eval_depth=1)
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


def test_random_opening_diversifies_eval_games():
    """Regression for the deterministic-eval artifact: a deterministic agent vs a deterministic
    opponent from the FIXED start replays one game (score quantizes, gradual skill invisible);
    random openings give distinct start states -> distinct games. Pins games.random_opening +
    play_match(start_state=...)."""
    from tikitorch.planning.muzero.games import random_opening, Connect4

    def first_legal(state):                       # a deterministic agent
        return Connect4.legal_actions(state[0])[0]

    def last_legal(state):
        return Connect4.legal_actions(state[0])[-1]

    # fixed start -> deterministic: identical winner every time
    w1 = play_match(first_legal, last_legal, Connect4)
    w2 = play_match(first_legal, last_legal, Connect4)
    assert w1 == w2

    # random openings -> diverse start states (the fix that makes eval see >2 games)
    rng = torch.Generator().manual_seed(0)
    starts = [random_opening(Connect4, rng, 4) for _ in range(16)]
    assert len({s for s in starts}) > 1                       # not all the same opening
    assert all(s != Connect4.initial_state() for s in starts)  # all moved off the fixed start
    # play_match accepts and plays from a given opening
    res = play_match(first_legal, last_legal, Connect4, start_state=starts[0])
    assert res in (0, 1, 2)


def test_alphazero_board_model_uses_true_rules_to_find_a_forced_win():
    """AlphaZeroBoardModel runs the TRUE game in the tree (exact terminals + legality at every node),
    so even an UNTRAINED net finds a tactic that the learned-model tree (BoardGameModel) misses. Here:
    a win-in-1 (bottom row cols 0,1,2 are player 1, player 1 to move -> col 3 completes four)."""
    from tikitorch.planning.muzero.games import AlphaZeroBoardModel, Connect4
    from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig

    net = BoardMuZero(Connect4, BoardConfig(net_kind="tropical", core_kind="dyadic",
                                            latent_dim=16)).net      # untrained
    net.eval()
    board = [0] * 42
    for c in (0, 1, 2):
        board[5 * 7 + c] = 1                                          # player-1 three-in-a-row, bottom
    state = (tuple(board), 1)
    assert 3 in Connect4.legal_actions(state[0])

    cfg = P.MCTSConfig(num_simulations=60, players=2, backup="mean", root_exploration_frac=0.0)
    az = P.TropicalMCTS(AlphaZeroBoardModel(net, Connect4), cfg)
    vp_az, _ = az.run(state)
    assert int(vp_az.argmax()) == 3                                  # true terminals -> finds the win

    # AlphaZero never visits illegal moves (true legality at every node): a full column gets 0 visits
    full = [0] * 42
    for r in range(6):
        full[r * 7 + 0] = 1 if r % 2 else 2                          # column 0 completely full
    st2 = (tuple(full), 1)
    assert 0 not in Connect4.legal_actions(st2[0])
    vp2, _ = P.TropicalMCTS(AlphaZeroBoardModel(net, Connect4), cfg).run(st2)
    assert float(vp2[0]) == 0.0


def test_fully_tropical_board_net_is_closure_empty_and_trains():
    """The fully multiplier-free recipe: dyadic cores + DYADIC heads (no nn.Linear) => closure audit
    returns NOTHING, and the closure-faithful training step (perceptron policy + L1 value + sign-only
    TropicalSGD) runs. This is the 'fully tropical, multiply-free' board agent."""
    from tikitorch.audit import find_closure_violations
    from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig
    from tikitorch.planning.muzero.games import Connect4

    cfg = BoardConfig(net_kind="tropical", core_kind="dyadic", dyadic_heads=True,
                      policy_loss="perceptron", optimizer="tropicalsgd", tree_model="alphazero",
                      latent_dim=32, num_simulations=16, games_per_iter=2, eval_games=2, seed=0)
    mz = BoardMuZero(Connect4, cfg)
    assert find_closure_violations(mz.net) == []          # zero nn.Linear: fully multiplier-free
    assert type(mz.opt).__name__ == "TropicalSGD"         # sign-only (multiply-free update)
    mz.self_play(temperature=1.0, base_seed=1)
    loss = mz.train_step()                                # perceptron + L1, sign-SGD step
    assert loss == loss and loss >= 0.0                   # finite, ran
