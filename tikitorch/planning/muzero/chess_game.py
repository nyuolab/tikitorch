"""tikitorch.planning.muzero.chess_game — Chess as a two-player MuZero game.

The third rung of the AlphaZero/MuZero ladder (TicTacToe -> Connect4 -> **chess**). A thin
wrapper over ``python-chess`` exposes the same game interface as :mod:`games` (``Chess`` plugs
straight into :class:`board_agent.BoardMuZero`). This is a pipeline scaffold: it demonstrates that
the same self-play, replay, and K-step training interfaces accept a chess-sized action space. It is
not a chess-strength result or a whole-loop hardware result. Dyadic network components are intended
for shift/add lowering, while search, normalization, losses, and orchestration remain explicit
boundaries unless a concrete implementation is separately audited and synthesized.

Representation choices (kept deliberately simple for the scaffold):
  * state  = ``(fen, to_play)`` — FEN is a hashable board key; ``to_play`` 1=white, 2=black.
  * obs    = 12 piece-type planes (white P,N,B,R,Q,K then black) over 64 squares + a side-to-move
             bit = 769 floats (absolute orientation; a side-to-move feature, not perspective-flipped,
             so it stays consistent with the absolute action encoding).
  * action = ``from*64 + to`` in [0, 4096). Promotions collapse to this index and decode to a
             QUEEN promotion — UNDERPROMOTIONS ARE NOT REPRESENTABLE (a documented scaffold
             simplification; the full AlphaZero 8x8x73=4672 encoding adds the 9 underpromotion
             planes). Legality is injected at the MCTS root from ``legal_actions``.

Requires ``python-chess`` (``pip install chess``); imported at module load.
"""
import chess
import torch

__all__ = ["Chess", "random_agent", "material_agent"]

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_MATERIAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class Chess:
    n_actions = 64 * 64                 # from*64 + to (promotions -> queen)
    obs_dim = 12 * 64 + 1               # 12 piece planes + side-to-move bit
    max_moves = 80                      # self-play ply cap (then adjudicated a draw)

    @staticmethod
    def initial_state():
        return (chess.STARTING_FEN, 1)

    @staticmethod
    def legal_actions(board):
        b = chess.Board(board)
        # collapse promotion variants to one (from,to) index (decode picks queen-promo)
        return sorted({m.from_square * 64 + m.to_square for m in b.legal_moves})

    @staticmethod
    def _decode(b, action):
        """Resolve an action index to a legal chess.Move on board ``b`` (queen-promo preferred)."""
        frm, to = action // 64, action % 64
        fallback = None
        for m in b.legal_moves:
            if m.from_square == frm and m.to_square == to:
                if m.promotion in (None, chess.QUEEN):
                    return m
                fallback = fallback or m
        return fallback if fallback is not None else chess.Move(frm, to)

    @staticmethod
    def apply(board, action, player=None):
        b = chess.Board(board)
        b.push(Chess._decode(b, action))     # player is implied by the FEN's side-to-move
        return b.fen()

    @staticmethod
    def winner(board):
        """1 (white) / 2 (black) / 0 (draw or non-terminal), claiming 3-fold / 50-move draws."""
        out = chess.Board(board).outcome(claim_draw=True)
        if out is None or out.winner is None:
            return 0
        return 1 if out.winner == chess.WHITE else 2

    @staticmethod
    def is_terminal(board):
        return chess.Board(board).is_game_over(claim_draw=True)

    @staticmethod
    def material_score(board):
        """White minus black material in pawns (king excluded). A cheap position heuristic."""
        b = chess.Board(board)
        s = 0
        for sq, pc in b.piece_map().items():
            v = _MATERIAL[pc.piece_type]
            s += v if pc.color == chess.WHITE else -v
        return s

    @staticmethod
    def adjudicate(board, margin=1):
        """Pseudo-outcome for a game stopped at the ply cap (NOT a real terminal): the side
        ahead by >= ``margin`` pawns of material is scored the winner, else a draw. This gives the
        otherwise all-draw, value-signal-free self-play games a learnable target (standard
        AlphaZero-style adjudication of unfinished games)."""
        m = Chess.material_score(board)
        return 1 if m >= margin else (2 if m <= -margin else 0)

    @staticmethod
    def obs(state):
        board, _ = state
        b = chess.Board(board)
        planes = []
        for color in (chess.WHITE, chess.BLACK):
            for pt in _PIECE_ORDER:
                ss = b.pieces(pt, color)
                planes.extend(1.0 if sq in ss else 0.0 for sq in range(64))
        planes.append(1.0 if b.turn == chess.WHITE else 0.0)
        return torch.tensor(planes)


def random_agent(seed=0):
    """A uniform-random legal mover (the honest weak baseline for the chess scaffold)."""
    g = torch.Generator().manual_seed(seed)

    def act(state):
        legal = Chess.legal_actions(state[0])
        return legal[int(torch.randint(0, len(legal), (1,), generator=g).item())]
    return act


def material_agent():
    """Greedy 1-ply material maximizer (captures the most valuable hanging piece). A slightly
    stronger-than-random baseline that still needs no search/heuristic tuning."""
    def act(state):
        b = chess.Board(state[0])
        best_a, best_v = None, -1e9
        for m in b.legal_moves:
            gain = 0
            if b.is_capture(m):
                cap = b.piece_at(m.to_square)
                gain = _MATERIAL[cap.piece_type] if cap else 1     # en-passant -> a pawn
            if gain > best_v:
                best_v, best_a = gain, m.from_square * 64 + m.to_square
        return best_a
    return act
