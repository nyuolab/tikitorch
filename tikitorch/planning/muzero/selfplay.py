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
                    root_exploration_frac=0.25, dirichlet_alpha=0.3, temperature=1.0, seed=0):
    """Two-player self-play with the learned net (the agent plays BOTH sides via
    negamax MCTS). Returns ``(records, winner)`` where each record is
    ``(obs, action, mcts_policy, value_target)`` with the value target the final
    game OUTCOME from that position's player-to-move perspective (+1/0/-1)."""
    from tikitorch.planning.muzero.games import BoardGameModel
    gen = torch.Generator().manual_seed(seed)
    cfg = MCTSConfig(num_simulations=num_simulations, c_puct=c_puct, gamma=1.0,
                     backup=backup, players=2, root_exploration_frac=root_exploration_frac,
                     dirichlet_alpha=dirichlet_alpha, seed=seed)
    mcts = TropicalMCTS(BoardGameModel(net, board_game), cfg)

    state = board_game.initial_state()
    moves = []                                          # (obs, action, policy, to_play)
    for t in range(board_game.H * board_game.W if hasattr(board_game, "H") else board_game.n_actions):
        if board_game.is_terminal(state[0]):
            break
        cfg.seed = seed * 100003 + t                    # fresh root noise each move
        visit_policy, _ = mcts.run(state)
        a = _sample_action(visit_policy, temperature, gen)
        board, tp = state
        moves.append((board_game.obs(state), a, visit_policy, tp))
        state = (board_game.apply(board, a, tp), 3 - tp)

    w = board_game.winner(state[0])
    records = [(obs, a, pol, (0.0 if w == 0 else (1.0 if w == tp else -1.0)))
               for (obs, a, pol, tp) in moves]
    return records, w
