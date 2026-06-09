"""Chess two-player MuZero game env (tikitorch.planning.muzero.chess_game).

Pins the game-interface contract (obs/action encoding, legality, terminal/winner) and that
Chess plugs straight into BoardMuZero (self-play runs, records well-formed, ply cap honored).
Skipped if python-chess is unavailable. Strength is NOT asserted here (that needs the scaffold
training run + honest benchmark) — this is correctness of the env + the pipeline wiring.
"""
import pytest

chess = pytest.importorskip("chess")

from tikitorch.planning.muzero.chess_game import Chess, random_agent, material_agent  # noqa: E402
from tikitorch.planning.muzero.games import play_match  # noqa: E402


def test_interface_shapes():
    assert Chess.n_actions == 4096
    assert Chess.obs_dim == 769
    s = Chess.initial_state()
    assert s == (chess.STARTING_FEN, 1)
    assert Chess.obs(s).shape == (769,)
    # 20 legal opening moves -> 20 unique (from,to) indices
    assert len(Chess.legal_actions(s[0])) == 20


def test_apply_flips_turn_and_is_legal():
    s = Chess.initial_state()
    a = Chess.legal_actions(s[0])[0]
    nb = Chess.apply(s[0], a)
    assert chess.Board(nb).turn == chess.BLACK            # white moved -> black to move
    # the resulting position is a legal successor
    assert chess.Board(nb).is_valid()


def test_decode_round_trips_every_legal_move():
    b = chess.Board()
    for m in b.legal_moves:
        dec = Chess._decode(b, m.from_square * 64 + m.to_square)
        assert dec in b.legal_moves


def test_checkmate_is_terminal_with_winner():
    # Fool's mate: 1.f3 e5 2.g4 Qh4#  -> white to move and checkmated -> black (2) wins
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert Chess.is_terminal(fen)
    assert Chess.winner(fen) == 2
    assert Chess.legal_actions(fen) == [] or chess.Board(fen).is_checkmate()


def test_stalemate_and_draw_are_not_a_winner():
    # classic stalemate: black to move, not in check, no legal moves -> draw (winner 0)
    fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    assert Chess.is_terminal(fen)
    assert Chess.winner(fen) == 0


def test_random_and_material_agents_play_legally_to_terminal():
    # a full random-vs-random game terminates (checkmate, draw, or the play_match move budget)
    res = play_match(random_agent(1), random_agent(2), Chess)
    assert res in (0, 1, 2)
    # material agent prefers captures: from the start it has none, so it just plays a legal move
    a = material_agent()(Chess.initial_state())
    assert a in Chess.legal_actions(Chess.initial_state()[0])


def test_plugs_into_board_muzero_selfplay():
    from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig
    from tikitorch.planning.muzero.selfplay import play_board_game
    cfg = BoardConfig(net_kind="tropical", core_kind="dyadic", latent_dim=32, num_simulations=4, seed=0)
    mz = BoardMuZero(Chess, cfg)
    records, w = play_board_game(mz.net, Chess, num_simulations=4, seed=1)
    assert w in (0, 1, 2)
    assert len(records) <= Chess.max_moves                # ply cap honored
    obs, a, pol, vt = records[0]
    assert obs.shape == (769,) and 0 <= a < 4096 and pol.shape == (4096,) and vt in (-1.0, 0.0, 1.0)


def test_material_adjudication_of_capped_games():
    # the starting position is materially even -> adjudicated a draw
    assert Chess.material_score(Chess.initial_state()[0]) == 0
    assert Chess.adjudicate(Chess.initial_state()[0]) == 0
    # white up a queen -> adjudicated a white (1) win; symmetric for black
    wq = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"   # black queen removed
    assert Chess.material_score(wq) == 9
    assert Chess.adjudicate(wq) == 1
    bq = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"   # white queen removed
    assert Chess.adjudicate(bq) == 2


def test_obs_is_perspective_consistent_pieces():
    # 32 pieces at the start -> 32 set entries across the 12 piece planes; stm bit = 1 (white)
    o = Chess.obs(Chess.initial_state())
    assert float(o[:768].sum()) == 32.0
    assert float(o[768]) == 1.0
