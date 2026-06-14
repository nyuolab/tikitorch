"""tikitorch.planning.muzero.selfplay — generate self-play trajectories with MCTS.

Plays one episode in ``env`` using ``TropicalMCTS`` over the learned ``net``,
choosing actions by temperature-sampling the MCTS visit distribution (with root
Dirichlet noise for exploration), and records the data MuZero trains on.
"""

import torch

from tikitorch.planning.mcts import MCTSConfig, TropicalMCTS
from tikitorch.planning.muzero.replay import Game

__all__ = ["play_game", "play_board_game"]


def _sample_action(visit_policy, temperature, gen):
    if temperature <= 1e-3:
        return int(visit_policy.argmax())
    logits = visit_policy.clamp_min(1e-8).log() / temperature
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen).item())


def play_game(net, env, *, num_simulations=50, c_puct=1.25, gamma=1.0,
              backup="topk", topk_k=2, root_exploration_frac=0.25,
              dirichlet_alpha=0.3, temperature=1.0, max_moves=None, seed=0):
    """Run one self-play episode; return a :class:`Game`."""
    max_moves = max_moves or (4 * env.n_states)
    gen = torch.Generator().manual_seed(seed)
    cfg = MCTSConfig(num_simulations=num_simulations, c_puct=c_puct, gamma=gamma,
                     backup=backup, topk_k=topk_k,
                     root_exploration_frac=root_exploration_frac,
                     dirichlet_alpha=dirichlet_alpha, seed=seed)
    mcts = TropicalMCTS(net, cfg)

    game = Game()
    s = env.reset()
    for t in range(max_moves):
        cfg.seed = seed * 100003 + t            # fresh root noise each move
        obs = env.obs(s).flatten()
        visit_policy, root_value = mcts.run(obs)
        a = _sample_action(visit_policy, temperature, gen)
        ns, r, done = env.step_state(s, a)
        game.obs.append(obs)
        game.actions.append(a)
        game.rewards.append(float(r))
        game.policies.append(visit_policy)
        game.root_values.append(float(root_value))
        s = ns
        if done:
            break
    return game


def play_board_game(net, board_game, *, num_simulations=50, c_puct=1.25, backup="mean",
                    root_exploration_frac=0.25, dirichlet_alpha=0.3, temperature=1.0, seed=0,
                    tree_model="muzero", legal_mask=False):
    """Two-player self-play with the learned net (the agent plays BOTH sides via
    negamax MCTS). Returns ``(records, winner)`` where each record is
    ``(obs, action, mcts_policy, value_target)`` with the value target the final
    game OUTCOME from that position's player-to-move perspective (+1/0/-1).

    ``tree_model``: ``"muzero"`` runs the learned latent dynamics inside the tree (no simulator);
    ``"alphazero"`` runs the game's TRUE rules in the tree (exact transitions/terminals/legality,
    net = prior+leaf-value) — far stronger search per sim when the simulator is available."""
    from tikitorch.planning.muzero.games import BoardGameModel, AlphaZeroBoardModel
    gen = torch.Generator().manual_seed(seed)
    cfg = MCTSConfig(num_simulations=num_simulations, c_puct=c_puct, gamma=1.0,
                     backup=backup, players=2, root_exploration_frac=root_exploration_frac,
                     dirichlet_alpha=dirichlet_alpha, seed=seed)
    model = AlphaZeroBoardModel(net, board_game) if tree_model == "alphazero" else BoardGameModel(net, board_game)
    mcts = TropicalMCTS(model, cfg)

    state = board_game.initial_state()
    moves = []                                          # (obs, action, policy, to_play)
    # ply cap: an explicit max_moves (chess), else the board's cell count (H*W), else n_actions.
    max_plies = getattr(board_game, "max_moves", None) or (
        board_game.H * board_game.W if hasattr(board_game, "H") else board_game.n_actions)
    for t in range(max_plies):
        if board_game.is_terminal(state[0]):
            break
        cfg.seed = seed * 100003 + t                    # fresh root noise each move
        visit_policy, _ = mcts.run(state)
        a = _sample_action(visit_policy, temperature, gen)
        board, tp = state
        mv = (board_game.obs(state), a, visit_policy, tp)
        if legal_mask:                                  # record the legal-move mask for masked-CE training
            m = torch.zeros(board_game.n_actions, dtype=torch.uint8)
            m[board_game.legal_actions(board)] = 1
            mv = mv + (m,)
        moves.append(mv)
        state = (board_game.apply(board, a, tp), 3 - tp)

    # outcome from the final state. A game stopped at the ply cap (not truly terminal) gets a
    # pseudo-outcome from the game's optional adjudicate() hook (chess: material balance) so the
    # value target is informative instead of a forced 0 — games that never resolve otherwise carry
    # no learning signal. Games that DO end (win/draw) use the real winner().
    if not board_game.is_terminal(state[0]) and hasattr(board_game, "adjudicate"):
        w = board_game.adjudicate(state[0])
    else:
        w = board_game.winner(state[0])
    records = [(mv[0], mv[1], mv[2], (0.0 if w == 0 else (1.0 if w == mv[3] else -1.0))) + tuple(mv[4:])
               for mv in moves]
    return records, w
