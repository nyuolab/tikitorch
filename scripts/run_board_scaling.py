"""run_board_scaling — scale the two-player MuZero loop from TicTacToe to Connect4.

Benchmarks the multiplier-free dyadic tropical MuZero against the max-plus core and a dense
control on Connect4 (6x7), the canonical AlphaZero/MuZero scale-up after TicTacToe. Board value
is the win/draw/loss OUTCOME — an OBLIQUE, non-lattice function of the position, exactly where
the max-plus core is predicted to underfit and the dyadic (multiplier-free oblique) core should
hold up. Skill is reported as score (win + 0.5*draw) in a gauntlet vs depth-limited negamax
opponents of INCREASING strength (depths 1..3), so we see skill-vs-opponent-strength, not a
single number. Reported MuZero/AlphaZero solve Connect4 (perfect first-player win); this is the
tractable-compute, zero-DSP-substrate analogue.

    PYTHONPATH=<repo-root> python3 scripts/run_board_scaling.py --mode smoke|full --game connect4
"""
import argparse
import json
import os
from collections import defaultdict

from tikitorch.planning.muzero.games import (Connect4, TicTacToe, negamax_agent,
                                             play_match)
from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "board_scaling"))

GAMES = {"connect4": Connect4, "tictactoe": TicTacToe}
ARMS = [
    ("tropical-maxplus", dict(net_kind="tropical", core_kind="maxplus")),
    ("tropical-dyadic",  dict(net_kind="tropical", core_kind="dyadic")),
    ("dense",            dict(net_kind="dense")),
]


def gauntlet(mz, game, depths, eval_games, seed0=10_000):
    """Score (win + 0.5*draw) vs negamax opponents of increasing depth, from RANDOM OPENINGS so the
    deterministic agent/opponent produce DISTINCT games (a fixed start collapses to ~2 games — a
    silent eval artifact; see games.random_opening / BoardMuZero.evaluate)."""
    import torch
    from tikitorch.planning.muzero.games import random_opening
    me = mz._mcts_agent()
    out = {}
    for d in depths:
        opp = negamax_agent(game, d)
        rng = torch.Generator().manual_seed(seed0 + d)
        w = dr = 0
        for g in range(eval_games):
            start = random_opening(game, rng, 4)
            my_id = 1 if g % 2 == 0 else 2
            agents = {my_id: me, 3 - my_id: opp}
            res = play_match(agents[1], agents[2], game, start_state=start)
            w += res == my_id
            dr += res == 0
        out[d] = round((w + 0.5 * dr) / eval_games, 3)
    return out


def run(mode, game_name, optimizer, ov):
    os.makedirs(OUT, exist_ok=True)
    game = GAMES[game_name]
    smoke = mode == "smoke"
    iters = ov.iters or (4 if smoke else 16)
    sims = ov.sims or 32
    games_pi = ov.games or 16
    arms = ARMS[1:2] if smoke else ARMS
    seeds = [0] if smoke else list(range(ov.seeds))
    depths = [1, 2] if smoke else [1, 2, 3]
    tag = ov.tag or mode
    out = os.path.join(OUT, f"scaling_{game_name}_{tag}_{optimizer}.jsonl")
    n = 0
    with open(out, "w") as fh:
        for name, kw in arms:
            for seed in seeds:
                cfg = BoardConfig(
                    optimizer=optimizer, latent_dim=64, num_simulations=sims,
                    iters=iters, games_per_iter=games_pi, train_steps_per_iter=30, batch=128,
                    unroll=3, eval_every=4, eval_games=20, eval_depth=2, num_workers=6,
                    tau_start=1.0, tau_end=0.1, lr=3e-3, seed=seed, **kw)
                mz = BoardMuZero(game, cfg)
                hist = mz.run(log=lambda s: None)            # loads best snapshot at the end
                gaunt = gauntlet(mz, game, depths, cfg.eval_games)
                fin = hist[-1]
                row = dict(arm=name, seed=seed, game=game_name, optimizer=optimizer, iters=iters,
                           best_score=fin["best"], gauntlet=gaunt,
                           score_curve=[round(h["score"], 3) for h in hist])
                fh.write(json.dumps(row) + "\n"); fh.flush()
                n += 1
                print(f"[{n}] {name:18s} s{seed} -> best={fin['best']:.2f} | "
                      f"gauntlet vs negamax {gaunt}", flush=True)
    summarize(out, tag, game_name, optimizer, depths)
    print(f"\nDONE: {n} rows -> {out}")


def summarize(out, mode, game_name, optimizer, depths):
    import numpy as np
    rows = [json.loads(line) for line in open(out)]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    hdr = " | ".join(f"vs neg-d{d}" for d in depths)
    lines = [f"# {game_name.title()} MuZero scaling — dyadic vs maxplus vs dense ({optimizer})", "",
             "_Two-player negamax-MCTS self-play; score = win + 0.5*draw in a gauntlet vs depth-limited "
             "negamax of increasing strength. Board value is the OBLIQUE win/draw/loss outcome. "
             "Reported MuZero/AlphaZero SOLVE Connect4 (perfect play); this is the tractable-compute, "
             "multiplier-free (zero-DSP) analogue — the point is the substrate scales structurally, and "
             "that the dyadic oblique core holds up where max-plus underfits._", "",
             f"| arm | n | best vs neg-d2 (train) | {hdr} (gauntlet, mean) |",
             "|---|---|---|" + "---|" * len(depths)]
    for arm in ["tropical-maxplus", "tropical-dyadic", "dense"]:
        rs = by.get(arm)
        if not rs:
            continue
        bs = np.array([r["best_score"] for r in rs])
        gs = [np.mean([r["gauntlet"][str(d)] for r in rs]) for d in depths]
        cells = " | ".join(f"{g:.2f}" for g in gs)
        lines.append(f"| {arm} | {len(rs)} | {bs.mean():.2f}±{bs.std():.2f} | {cells} |")
    path = os.path.join(OUT, f"SCALING_{game_name.upper()}_{mode}_{optimizer}.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"\n[summary] -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--game", choices=list(GAMES), default="connect4")
    ap.add_argument("--optimizer", default="tropicalsgd")
    ap.add_argument("--iters", type=int, default=0)        # 0 = mode default
    ap.add_argument("--sims", type=int, default=0)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.mode, a.game, a.optimizer, a)
