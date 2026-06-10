"""_one_board_run — train ONE (arm, seed) board-MuZero config and print one JSON result line.

A single-process worker (num_workers=1) so a fleet of these parallelizes cleanly across cores
(one run per core) for a robust multi-seed benchmark sweep. Prints exactly one JSON object:
{arm, seed, optimizer, game, best_score, gauntlet:{depth:score}}. Used by the board-scaling workflow.

    PYTHONPATH=<repo-root> python3 scripts/_one_board_run.py --arm dyadic --seed 0 \
        --opt tropicalsgd --game connect4 --sims 48 --iters 20 --games 16
"""
import argparse
import json

import torch

torch.set_num_threads(1)            # one core per worker; parallelism is across the fleet

from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig  # noqa: E402
from tikitorch.planning.muzero.games import Connect4, TicTacToe
from scripts.run_board_scaling import gauntlet

GAMES = {"connect4": Connect4, "tictactoe": TicTacToe}
ARM = {"maxplus": dict(net_kind="tropical", core_kind="maxplus"),
       "dyadic":  dict(net_kind="tropical", core_kind="dyadic"),
       "dense":   dict(net_kind="dense")}


def main(a):
    game = GAMES[a.game]
    cfg = BoardConfig(optimizer=a.opt, latent_dim=64, num_simulations=a.sims,
                      iters=a.iters, games_per_iter=a.games, train_steps_per_iter=30, batch=128,
                      unroll=3, eval_every=max(a.iters - 1, 1), eval_games=a.eval, eval_depth=2,
                      num_workers=a.workers, tau_start=1.0, tau_end=0.1, lr=3e-3, seed=a.seed, **ARM[a.arm])
    mz = BoardMuZero(game, cfg)
    hist = mz.run(log=lambda s: None)
    gaunt = gauntlet(mz, game, [1, 2, 3], a.eval)
    print(json.dumps(dict(arm=a.arm, seed=a.seed, optimizer=a.opt, game=a.game,
                          best_score=hist[-1]["best"], gauntlet=gaunt)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=list(ARM))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--opt", default="tropicalsgd")
    p.add_argument("--game", default="connect4", choices=list(GAMES))
    p.add_argument("--sims", type=int, default=48)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--eval", type=int, default=20)
    p.add_argument("--workers", type=int, default=1)   # parallel self-play workers within this run
    main(p.parse_args())
