"""Train and benchmark a Connect4 agent with dyadic cores and heads.

The net has dyadic tropical cores and dyadic heads (zero ``nn.Linear``). Its model closure audit is
reported separately from the exact loss, Python MCTS, and physical implementation. Training uses
AlphaZero-style explicit-game self-play and configurable policy loss, entropy, target temperature,
L1 value, and TropicalSGD. Several configurations execute powers, exact softmax/log, multiplication,
or division, so the complete loop must not be described as multiplier-free or zero-DSP.

Designed to run on EC2 (many cores -> more self-play). Prints the scoped model closure audit.

    PYTHONPATH=<repo> python3 scripts/run_fully_tropical_connect4.py [--sims 256 --iters 40 --games 48 --workers N]
"""
import argparse
import json
import os

import torch

from tikitorch.audit import find_closure_violations
from tikitorch.planning.muzero.games import Connect4, negamax_agent, random_opening, play_match, AlphaZeroBoardModel
from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig
from tikitorch.planning.mcts import TropicalMCTS, MCTSConfig

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", "board_scaling"))


def az_agent(net, sims):
    mc = TropicalMCTS(AlphaZeroBoardModel(net, Connect4),
                      MCTSConfig(num_simulations=sims, c_puct=1.25, gamma=1.0,
                                 backup="mean", players=2, root_exploration_frac=0.0))
    return lambda s: int(mc.run(s)[0].argmax())


def score_vs(net, sims, depth, n=40):
    ag, opp = az_agent(net, sims), negamax_agent(Connect4, depth)
    rng = torch.Generator().manual_seed(31 + depth)
    w = d = 0
    for g in range(n):
        start = random_opening(Connect4, rng, 4)
        me = 1 if g % 2 == 0 else 2
        ags = {me: ag, 3 - me: opp}
        r = play_match(ags[1], ags[2], Connect4, start_state=start)
        w += r == me
        d += r == 0
    return round((w + 0.5 * d) / n, 3)


def main(a):
    if a.threads > 0:
        torch.set_num_threads(a.threads)   # PIN threads: running N recipes concurrently on one box with
        # torch's default (= all cores) per process oversubscribes the eval phase ~Nx and runs for HOURS.
    cfg = BoardConfig(net_kind="tropical", core_kind="dyadic", dyadic_heads=True,
                      policy_loss=a.policy_loss, policy_entropy_coef=a.entropy_coef, target_temp=a.target_temp,
                      optimizer="tropicalsgd", tree_model="alphazero",
                      latent_dim=64, num_simulations=a.sims, iters=a.iters, games_per_iter=a.games,
                      train_steps_per_iter=40, batch=128, unroll=3, num_workers=a.workers,
                      tau_start=1.0, tau_end=0.1, lr=a.lr, seed=0, eval_every=999, eval_games=2)
    probe = BoardMuZero(Connect4, cfg)
    viol = find_closure_violations(probe.net)
    print(f"[{a.tag}] model closure audit (does not cover loss/MCTS): {viol or 'NONE'}", flush=True)
    print(f"[{a.tag}] recipe: policy_loss={a.policy_loss} entropy={a.entropy_coef} temp={a.target_temp} "
          f"lr={a.lr} | {a.sims} sims, {a.iters}x{a.games} games, {a.workers} workers...", flush=True)
    probe.run(log=lambda s: None)
    untrained = BoardMuZero(Connect4, cfg)

    grid = [int(s) for s in a.grid.split(",")]
    res = {"recipe": dict(policy_loss=a.policy_loss, entropy_coef=a.entropy_coef, target_temp=a.target_temp, lr=a.lr),
           "closure_violations": viol, "trained": {}, "untrained": {}}
    for label, mz in (("trained", probe), ("untrained", untrained)):
        for sims in grid:
            res[label][sims] = {d: score_vs(mz.net, sims, d, n=a.eval_games) for d in (2, 3)}
            print(f"[{a.tag}]  {label:9s} @ {sims:3d} sims:  d2={res[label][sims][2]}  d3={res[label][sims][3]}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"connect4_fully_tropical_{a.tag}.json")
    open(out, "w").write(json.dumps(
        {"note": "Connect4 with dyadic cores+heads and AlphaZero-style explicit-game tree. "
                 "closure_violations audits the model only, not loss/MCTS or FPGA resources. "
                 "score=win+0.5draw, 40 games.",
         "config": {"sims": a.sims, "iters": a.iters, "games": a.games}, **res}, indent=2))
    print(f"[{a.tag}] [saved] {out}")
    print(f"DONE_FT_{a.tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=256)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--policy-loss", dest="policy_loss", default="ce", choices=["ce", "perceptron", "tropce"])
    p.add_argument("--entropy-coef", dest="entropy_coef", type=float, default=0.0)
    p.add_argument("--target-temp", dest="target_temp", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threads", type=int, default=0, help="pin torch threads (use 1 per process in parallel sweeps)")
    p.add_argument("--eval-games", dest="eval_games", type=int, default=40, help="games per eval cell (manuscript: 40)")
    p.add_argument("--grid", default="100,200,400", help="eval sim grid (manuscript: 100,200,400)")
    p.add_argument("--tag", default="run")
    main(p.parse_args())
