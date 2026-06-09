"""run_dyadic_muzero — full Tropical MuZero loop with the dyadic oblique core, end-to-end.

Trains the complete MuZero loop (self-play MCTS -> replay -> K-step unrolled train) on a GridWorld
planning task for three cores and reports the value-iteration-oracle metrics (agree = optimal-action
agreement; solve_frac = fraction of starts the greedy policy solves):

  * tropical-maxplus : current axis-aligned cores (baseline)
  * tropical-dyadic  : multiplier-free OBLIQUE cores (the fix; repr+pred dyadic, dynamics maxplus)
  * dense            : full-precision matched control (the bar)

GridWorld value is a (min,+) cost-to-go — the tropical-FAVORABLE prior — so this validates that the
dyadic core (a) trains end-to-end in the real MCTS loop and (b) does NOT regress on the favorable case;
the dyadic EXPRESSIVITY advantage itself was established on oblique/smooth value in the probes
(results/capacity_probe/FIX_FINDINGS.md). Uses AdamW to isolate the CORE effect from the optimizer; the
multiplier-free closed optimizer (TropicalSGD) is a separate axis (--optimizer).

    PYTHONPATH=<repo-root> python3 scripts/run_dyadic_muzero.py --mode smoke | --mode full
"""
import argparse
import json
import os

from tikitorch.planning.muzero.env import GridWorldNav
from tikitorch.planning.muzero.agent import TropicalMuZero, MuZeroConfig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "results", "capacity_probe"))

GRID = ["S...", "....", "....", "...G"]          # 4x4 open, fully connected (V* finite everywhere)

ARMS = [
    ("tropical-maxplus", dict(net_kind="tropical", core_kind="maxplus")),
    ("tropical-dyadic",  dict(net_kind="tropical", core_kind="dyadic")),
    ("dense",            dict(net_kind="dense")),
]


def run(mode, optimizer):
    os.makedirs(OUT, exist_ok=True)
    iters = {"smoke": 4, "full": 14, "long": 30}[mode]
    arms = ARMS[1:2] if mode == "smoke" else ARMS
    seeds = [0] if mode == "smoke" else ([0, 1, 2] if mode == "full" else [0, 1, 2, 3, 4])
    games = 8 if mode == "long" else 6
    sims = 25 if mode == "long" else 20
    out = os.path.join(OUT, f"muzero_loop_{mode}.jsonl")
    n = 0
    with open(out, "w") as fh:
        for name, kw in arms:
            for seed in seeds:
                cfg = MuZeroConfig(
                    optimizer=optimizer, latent_dim=32, num_simulations=sims,
                    iters=iters, games_per_iter=games, train_steps_per_iter=12, batch=64,
                    td_steps=5, unroll=3, eval_every=5, lr=3e-3, seed=seed, **kw)
                env = GridWorldNav(GRID)
                mz = TropicalMuZero(env, cfg)
                hist = mz.run(log=lambda s: None)
                fin = hist[-1]
                row = dict(arm=name, seed=seed, optimizer=optimizer, iters=iters,
                           final_agree=fin["agree"], final_solve_frac=fin["solve_frac"],
                           final_solved=bool(fin["solved"]),
                           agree_curve=[round(h["agree"], 3) for h in hist],
                           solve_curve=[round(h["solve_frac"], 3) for h in hist])
                fh.write(json.dumps(row) + "\n"); fh.flush()
                n += 1
                print(f"[{n}] {name:18s} s{seed} -> final agree={fin['agree']:.0%} "
                      f"solve_frac={fin['solve_frac']:.0%} solved={bool(fin['solved'])} | "
                      f"agree curve {[round(h['agree'],2) for h in hist]}")
    summarize(out, mode)
    print(f"\nDONE: {n} rows -> {out}")


def summarize(out, mode):
    import numpy as np
    from collections import defaultdict
    rows = [json.loads(l) for l in open(out)]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    lines = ["# Full Tropical MuZero loop — dyadic vs maxplus vs dense (GridWorld planning)", "",
             "_End-to-end MCTS self-play + K-step unrolled training; metrics vs the value-iteration oracle. "
             "GridWorld value is (min,+) cost-to-go (tropical-favorable), so this is a no-regression + "
             "trains-end-to-end check; the dyadic expressivity advantage is on oblique value (FIX_FINDINGS.md)._", "",
             "| arm | n | final agree (mean±std) | final solve_frac (mean±std) | start solved |",
             "|---|---|---|---|---|"]
    for arm in ["tropical-maxplus", "tropical-dyadic", "dense"]:
        rs = by.get(arm)
        if not rs:
            continue
        ag = np.array([r["final_agree"] for r in rs]); sf = np.array([r["final_solve_frac"] for r in rs])
        solved = sum(r["final_solved"] for r in rs)
        lines.append(f"| {arm} | {len(rs)} | {ag.mean():.0%}±{ag.std():.0%} | "
                     f"{sf.mean():.0%}±{sf.std():.0%} | {solved}/{len(rs)} |")
    path = os.path.join(OUT, f"MUZERO_LOOP_FINDINGS_{mode}.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"\n[summary] -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "long"], default="full")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "tropicalsgd", "signsgd"])
    a = ap.parse_args()
    run(a.mode, a.optimizer)
