"""run_dyadic_board — dyadic vs maxplus vs dense MuZero on TicTacToe (OBLIQUE value).

Unlike GridWorld (a (min,+) cost-to-go — tropical-favorable), a board game's value is the win/draw/loss
OUTCOME — a non-lattice, OBLIQUE function of the position. This is exactly where the capacity probes
predict the max-plus core underfits and the dyadic core (multiplier-free oblique) should win, now tested
in the full two-player MuZero loop. Skill = score (win + 0.5*draw) vs a fixed depth-limited negamax.

    PYTHONPATH=<repo-root> python3 scripts/run_dyadic_board.py --mode smoke | --mode full
"""
import argparse
import json
import os

from tikitorch.planning.muzero.games import TicTacToe
from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "manuscript", "results", "capacity_probe"))

ARMS = [
    ("tropical-maxplus", dict(net_kind="tropical", core_kind="maxplus")),
    ("tropical-dyadic",  dict(net_kind="tropical", core_kind="dyadic")),
    ("dense",            dict(net_kind="dense")),
]


def run(mode, optimizer):
    os.makedirs(OUT, exist_ok=True)
    iters = 4 if mode == "smoke" else 20
    arms = ARMS[1:2] if mode == "smoke" else ARMS
    seeds = [0] if mode == "smoke" else [0, 1, 2]
    out = os.path.join(OUT, f"board_loop_{mode}_{optimizer}.jsonl")
    n = 0
    with open(out, "w") as fh:
        for name, kw in arms:
            for seed in seeds:
                cfg = BoardConfig(
                    optimizer=optimizer, latent_dim=48, num_simulations=30,
                    iters=iters, games_per_iter=16, train_steps_per_iter=20, batch=128,
                    unroll=3, eval_every=4, eval_games=20, eval_depth=2,
                    tau_start=1.0, tau_end=0.1, lr=3e-3, seed=seed, **kw)
                mz = BoardMuZero(TicTacToe, cfg)
                hist = mz.run(log=lambda s: None)
                fin = hist[-1]
                row = dict(arm=name, seed=seed, optimizer=optimizer, iters=iters,
                           final_score=fin["score"], best_score=fin["best"],
                           final_win=fin["win"], final_draw=fin["draw"], final_loss=fin["loss"],
                           score_curve=[round(h["score"], 3) for h in hist])
                fh.write(json.dumps(row) + "\n"); fh.flush()
                n += 1
                print(f"[{n}] {name:18s} s{seed} -> final score={fin['score']:.2f} best={fin['best']:.2f} "
                      f"(W{fin['win']:.0%}/D{fin['draw']:.0%}/L{fin['loss']:.0%}) | curve {[round(h['score'],2) for h in hist]}")
    summarize(out, mode, optimizer)
    print(f"\nDONE: {n} rows -> {out}")


def summarize(out, mode, optimizer):
    import numpy as np
    from collections import defaultdict
    rows = [json.loads(l) for l in open(out)]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    lines = [f"# TicTacToe MuZero (OBLIQUE value) — dyadic vs maxplus vs dense ({optimizer})", "",
             "_Two-player negamax-MCTS self-play; skill = score (win + 0.5*draw) vs depth-2 negamax. "
             "Board value is the win/draw/loss outcome (OBLIQUE, non-lattice) — where the max-plus core is "
             "predicted to underfit and the dyadic (multiplier-free oblique) core should win. best = best "
             "snapshot over training (the deployed agent)._", "",
             "| arm | n | final score (mean±std) | BEST score (mean±std) |",
             "|---|---|---|---|"]
    for arm in ["tropical-maxplus", "tropical-dyadic", "dense"]:
        rs = by.get(arm)
        if not rs:
            continue
        fs = np.array([r["final_score"] for r in rs]); bs = np.array([r["best_score"] for r in rs])
        lines.append(f"| {arm} | {len(rs)} | {fs.mean():.2f}±{fs.std():.2f} | {bs.mean():.2f}±{bs.std():.2f} |")
    path = os.path.join(OUT, f"BOARD_LOOP_FINDINGS_{mode}_{optimizer}.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"\n[summary] -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "tropicalsgd", "signsgd"])
    a = ap.parse_args()
    run(a.mode, a.optimizer)
