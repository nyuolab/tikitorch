"""Closed-optimizer bake-off.

Runs the full grid (TASKS x ARMS x SEEDS) of ``evaluate_task`` to decide whether
any software-closed (``assert_training_closed``-verified) adaptive optimizer closes
the "closure tax" -- the gap by which sign-only closure-oriented optimizers trail the
non-closed Adam proxy on the algorithmic-reasoning tasks.

The audit classifies a software recipe; it does not establish FPGA DSP use without a
configuration-matched lowering and synthesis report.

Arms:
  * 6 CLOSED optimizers (closed=True, tropical forward): puretropical, tropicaladam,
    countagree, perlayer, frexpmag, expadam.
  * adam_ceiling  -- non-closed Adam proxy (closed=False, tropical attention): the bar.
  * softmax_ceiling -- non-closed softmax (closed=False, softmax attention): the ceiling.

Metric per cell is task-appropriate (argmax-accuracy for token tasks; micro-F1 for
the sparse binary DP tasks). We report id (in-distribution, held-out) and ood_len
(2x length) means +- std over SEEDS. Honest reporting: >=3 seeds, no pre-claiming.

Usage:
    python scripts/bakeoff_closed_optimizers.py [--quick] [--out manuscript/results/bakeoff.jsonl]

Writes one JSON line per (task, arm, seed) cell as it completes, then prints the
aggregated table to stdout (and a markdown table next to the jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

TASKS = ["running_max", "sort", "subset_sum", "knapsack", "min_coin_change"]

# Each arm -> kwargs for evaluate_task. The closed arms all share the closed recipe
# (hard tropical forward, perceptron/L1 loss, sign-based optimizer); they differ only
# in closed_opt. The two ceilings are non-closed references.
CLOSED_OPTS = ["puretropical", "tropicaladam", "countagree", "perlayer", "frexpmag", "expadam"]


# "countagree8" is the tuned CountAgreementSGD (accum=8 agreement window); it
# reuses the countagree closed_opt with a larger accum.
_ARM_TO_OPT = {a: a for a in CLOSED_OPTS}
_ARM_TO_OPT["countagree8"] = "countagree"
_ARM_ACCUM = {"countagree": None, "countagree8": 8}   # None -> default accum


def _arm_kwargs(arm: str, quick: bool, steps: int | None = None) -> dict:
    # The tropical-attention forward/backward is ~0.27 s/iter at batch 64 (single
    # thread); cost is iters = steps * accum. These settings give a clean >=3-seed
    # signal in ~27 min on a 16-core box; the closed arms share a 4-microbatch accum
    # window (effective batch 256) so CountAgreementSGD has a real agreement signal.
    if steps is None:
        steps = 120 if quick else 250
    accum = 2 if quick else 4
    train_n = 1024 if quick else 2048
    common = dict(d_model=64, n_layers=2, n_heads=4, train_len=16, ood_len=32,
                  train_n=train_n, test_n=512, batch_size=64, steps=steps)
    if arm in _ARM_TO_OPT:
        return dict(**common, attention="tropical", closed=True,
                    closed_opt=_ARM_TO_OPT[arm],
                    accum=_ARM_ACCUM.get(arm, accum) or accum, lr=2 ** -4, anneal=False)
    if arm == "adam_ceiling":
        return dict(**common, attention="tropical", closed=False, lr=3e-3, anneal=True)
    if arm == "softmax_ceiling":
        return dict(**common, attention="softmax", closed=False, lr=3e-3, anneal=True)
    raise ValueError(arm)


ALL_ARMS = CLOSED_OPTS + ["countagree8", "adam_ceiling", "softmax_ceiling"]
ARMS = CLOSED_OPTS + ["adam_ceiling", "softmax_ceiling"]   # default full grid


def _run_cell(args):
    task, arm, seed, quick, steps = args
    # keep each worker single-threaded so the pool doesn't oversubscribe the cores
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    import tikitorch.algorithmic as A
    kw = _arm_kwargs(arm, quick, steps)
    t0 = time.time()
    try:
        r = A.evaluate_task(task, seed=seed, device="cpu", **kw)
        return dict(task=task, arm=arm, seed=seed, ok=True,
                    train_acc=r["train_acc"], id_acc=r["id_acc"],
                    ood_len_acc=r["ood_len_acc"], secs=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001
        return dict(task=task, arm=arm, seed=seed, ok=False, err=repr(e),
                    secs=round(time.time() - t0, 1))


def _aggregate(rows):
    """rows -> {(task,arm): {id:[..], ood:[..]}} -> mean/std table."""
    import statistics as st
    agg = {}
    for r in rows:
        if not r.get("ok"):
            continue
        agg.setdefault((r["task"], r["arm"]), {"id": [], "ood": []})
        agg[(r["task"], r["arm"])]["id"].append(r["id_acc"])
        agg[(r["task"], r["arm"])]["ood"].append(r["ood_len_acc"])

    def ms(xs):
        if not xs:
            return (float("nan"), float("nan"))
        return (sum(xs) / len(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)

    return {k: {"id": ms(v["id"]), "ood": ms(v["ood"]), "n": len(v["id"])}
            for k, v in agg.items()}


def _print_table(table, metric="id"):
    print(f"\n=== {metric.upper()} (mean +- std over seeds) ===")
    header = f"{'task':16s} " + " ".join(f"{a:>15s}" for a in ARMS)
    print(header)
    print("-" * len(header))
    for task in TASKS:
        cells = []
        for arm in ARMS:
            v = table.get((task, arm))
            if v is None:
                cells.append(f"{'--':>15s}")
            else:
                m, s = v[metric]
                cells.append(f"{m:6.3f}+-{s:4.2f}".rjust(15))
        print(f"{task:16s} " + " ".join(cells))
    # closed-arm averages + best closed vs the Adam ceiling
    print("\n-- closed-arm mean across tasks (and gap to adam_ceiling) --")
    import statistics as st
    for arm in ARMS:
        vals = [table[(t, arm)][metric][0] for t in TASKS if (t, arm) in table]
        if vals:
            print(f"  {arm:16s} {st.mean(vals):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke grid (fewer steps)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None, help="override training steps (both budgets)")
    ap.add_argument("--arms", nargs="*", default=None,
                    help=f"subset of arms (default full grid). choices: {ALL_ARMS}")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of tasks")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default="manuscript/results/bakeoff_closed_optimizers.jsonl")
    args = ap.parse_args()

    global ARMS, TASKS
    if args.arms:
        ARMS = args.arms
    if args.tasks:
        TASKS = args.tasks
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cells = [(t, a, s, args.quick, args.steps) for t in TASKS for a in ARMS for s in range(args.seeds)]
    print(f"bake-off: {len(cells)} cells "
          f"({len(TASKS)} tasks x {len(ARMS)} arms x {args.seeds} seeds), "
          f"{args.workers} workers, quick={args.quick}")
    rows = []
    t0 = time.time()
    with open(args.out, "w") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_cell, c): c for c in cells}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done += 1
            tag = "OK " if r.get("ok") else "ERR"
            extra = (f"id={r['id_acc']:.3f} ood={r['ood_len_acc']:.3f}"
                     if r.get("ok") else r.get("err", "")[:60])
            print(f"[{done:3d}/{len(cells)}] {tag} {r['task']:16s} {r['arm']:15s} "
                  f"s{r['seed']} ({r['secs']}s) {extra}")

    print(f"\ndone in {round(time.time()-t0,1)}s -> {args.out}")
    table = _aggregate(rows)
    _print_table(table, "id")
    _print_table(table, "ood")

    # markdown summary next to the jsonl
    md = args.out.rsplit(".", 1)[0] + ".md"
    with open(md, "w") as fh:
        for metric in ("id", "ood"):
            fh.write(f"### {metric.upper()} (mean ± std over {args.seeds} seeds)\n\n")
            fh.write("| task | " + " | ".join(ARMS) + " |\n")
            fh.write("|" + "---|" * (len(ARMS) + 1) + "\n")
            for task in TASKS:
                row = [task]
                for arm in ARMS:
                    v = table.get((task, arm))
                    row.append(f"{v[metric][0]:.3f}±{v[metric][1]:.2f}" if v else "--")
                fh.write("| " + " | ".join(row) + " |\n")
            fh.write("\n")
    print(f"markdown -> {md}")


if __name__ == "__main__":
    sys.exit(main())
