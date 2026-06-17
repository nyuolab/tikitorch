"""muzero_scale_bench — measure the real throughput of the self-play + train loop, to cost a scaled run.

Times the two cost drivers at the decisive-experiment config (Connect4, true-rules AZ tree, 100 sims),
for both substrates, so a scale curve can be priced honestly:
  * self-play: games/s and moves/s (CPU-bound, parallel across workers) — the dominant cost
  * train:     train_steps/s
  * net eval:  single initial-inference latency (the per-MCTS-node cost)

    PYTHONPATH=. python3 scripts/muzero_scale_bench.py [--sims 100 --games 16 --workers 8]
"""
import argparse
import time

import torch

from tikitorch.planning.muzero.games import Connect4
from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig

GAMES = {"connect4": Connect4}


def _game(name):
    if name == "chess":
        from tikitorch.planning.muzero.chess_game import Chess
        return Chess
    return GAMES[name]


def bench(net_kind, optimizer, a):
    game = _game(a.game)
    cfg = BoardConfig(net_kind=net_kind, core_kind="dyadic", latent_dim=a.latent, optimizer=optimizer,
                      tree_model=a.tree, num_simulations=a.sims, games_per_iter=a.games,
                      train_steps_per_iter=a.steps, batch=128, unroll=3, num_workers=a.workers,
                      tau_start=1.0, tau_end=0.3, seed=0)
    mz = BoardMuZero(game, cfg)
    nparams = sum(p.numel() for p in mz.net.parameters())

    # net eval latency (single initial inference, batch 1)
    obs = torch.zeros(1, game.obs_dim)
    for _ in range(5):
        mz.net.initial(obs)
    t = time.perf_counter()
    for _ in range(200):
        mz.net.initial(obs)
    eval_us = (time.perf_counter() - t) / 200 * 1e6

    # self-play: one iter of `games` games at `sims`, parallel workers
    t = time.perf_counter()
    mz.self_play(temperature=1.0, base_seed=1)
    sp_s = time.perf_counter() - t
    n_games = len(mz.buffer.games)
    n_moves = sum(len(g) for g in mz.buffer.games)

    # train throughput
    t = time.perf_counter()
    for _ in range(a.steps):
        mz.train_step()
    tr_s = time.perf_counter() - t

    return dict(net=net_kind, opt=optimizer, params=nparams, eval_us=eval_us,
                games=n_games, moves=n_moves, selfplay_s=sp_s,
                games_per_s=n_games / sp_s, moves_per_s=n_moves / sp_s,
                train_steps_per_s=a.steps / tr_s)


def main(a):
    print(f"[bench] game={a.game} tree={a.tree} sims={a.sims} games/iter={a.games} "
          f"workers={a.workers} latent={a.latent} batch=128\n")
    for net, opt in [("tropical", "tropicalsgd"), ("dense", "adamw")]:
        r = bench(net, opt, a)
        samples_per_s = r['train_steps_per_s'] * 128
        moves_per_core = r['moves_per_s'] / a.workers
        # crude net-bound fraction: per-move net time (sims x eval) vs wall per-move per worker
        net_s_per_move = a.sims * r['eval_us'] * 1e-6
        wall_s_per_move = 1.0 / max(moves_per_core, 1e-9)
        net_frac = min(net_s_per_move / wall_s_per_move, 1.0)
        print(f"=== {net} ({opt}), {r['params']:,} params ===")
        print(f"  net eval        : {r['eval_us']:.0f} us/inference (batch1, CPU)")
        print(f"  self-play       : {r['moves']} moves in {r['selfplay_s']:.1f}s "
              f"-> {r['moves_per_s']:.2f} moves/s ({a.workers} workers; {moves_per_core:.2f}/core)")
        print(f"  train (learner) : {r['train_steps_per_s']:.2f} steps/s -> {samples_per_s:,.0f} samples/s (b128)")
        print(f"  net-bound frac  : ~{net_frac:.0%} of self-play is net-eval (rest = MCTS tree, host)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["connect4", "chess"], default="connect4")
    ap.add_argument("--tree", choices=["muzero", "alphazero"], default="alphazero")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--latent", type=int, default=64)
    main(ap.parse_args())
