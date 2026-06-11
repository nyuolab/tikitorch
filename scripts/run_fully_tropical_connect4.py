"""run_fully_tropical_connect4 — the FULLY multiplier-free Connect4 agent, trained + benchmarked.

Net: dyadic tropical CORES + DYADIC heads (zero nn.Linear -> closure audit returns []) — fully zero-DSP.
Training: AlphaZero-style true-model self-play (strong visit targets) distilled into the dyadic policy via
the closure-faithful PERCEPTRON loss (argmax-of-visits, 2-hot silicon grad) + L1 value + sign-only
TropicalSGD. So forward, backward (sign-product == sign(autograd grad)), and update are all multiply-free.
Benchmarks trained vs untrained at increasing sims vs negamax d2/d3 (random openings) — i.e. does training
a FULLY-tropical net buy sim-efficiency on top of the depth that already beats the wall.

Designed to run on EC2 (many cores -> more self-play). Prints the closure audit so the multiply-free
claim is self-checking.

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
    cfg = BoardConfig(net_kind="tropical", core_kind="dyadic", dyadic_heads=True,
                      policy_loss="perceptron", optimizer="tropicalsgd", tree_model="alphazero",
                      latent_dim=64, num_simulations=a.sims, iters=a.iters, games_per_iter=a.games,
                      train_steps_per_iter=40, batch=128, unroll=3, num_workers=a.workers,
                      tau_start=1.0, tau_end=0.1, lr=3e-3, seed=0, eval_every=999, eval_games=2)
    probe = BoardMuZero(Connect4, cfg)
    viol = find_closure_violations(probe.net)
    print(f"closure audit (must be empty for fully multiply-free): {viol or 'NONE'}", flush=True)
    print(f"training FULLY-tropical net (dyadic heads, perceptron+L1, SignSGD, AlphaZero tree): "
          f"{a.sims} sims, {a.iters}x{a.games} games, {a.workers} workers...", flush=True)
    probe.run(log=lambda s: None)
    untrained = BoardMuZero(Connect4, cfg)

    grid = [100, 200, 400]
    res = {"closure_violations": viol, "trained": {}, "untrained": {}}
    for label, mz in (("trained", probe), ("untrained", untrained)):
        for sims in grid:
            res[label][sims] = {d: score_vs(mz.net, sims, d) for d in (2, 3)}
            print(f"  {label:9s} @ {sims:3d} sims:  d2={res[label][sims][2]}  d3={res[label][sims][3]}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "connect4_fully_tropical.json"), "w").write(json.dumps(
        {"note": "FULLY multiplier-free Connect4 (dyadic cores+heads, perceptron+L1+SignSGD, AlphaZero "
                 "tree). closure_violations must be []. score=win+0.5draw, random openings, 40 games.",
         "config": {"sims": a.sims, "iters": a.iters, "games": a.games}, **res}, indent=2))
    print("\n[saved] connect4_fully_tropical.json")
    print("DONE_FT")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=256)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    main(p.parse_args())
