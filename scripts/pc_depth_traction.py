"""Exp 2 — PC-vs-BP bpb gap vs depth: the central deep-PC-traction figure (PREREG.md).
Arms disambiguate substrate from rule: bp_float | bp_dyadic (dyadic substrate, BP rule) |
relax_async (dyadic substrate, local-PC rule). Depths {2,4,8,16} x seeds {0,1,2}, matched
data (same epochs); relax FLOP overhead reported, not hidden.

Run: PYTHONPATH=. python scripts/pc_depth_traction.py [--epochs 8] [--subset 400000]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch

from pc_charlm import load, run_one_lm, bigram_floor

OUT = "results/pc_depth_traction/exp2_traction.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--subset", type=int, default=400_000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+",
                    default=["bp_float", "bp_dyadic", "relax_async"])
    ap.add_argument("--device", default="cpu")   # DyadicPCNet relax proven on cpu/mps
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))

    tr, va, V, _, d = load(args.context, args.subset)
    floor = bigram_floor(d, V, tr[0].size(0))
    print(f"vocab={V} train={tr[0].size(0)} bigram_floor={floor:.3f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for L in args.depths:
        dims = [args.context * V] + [args.hidden] * (L - 1) + [V]
        for seed in args.seeds:
            for mode in args.modes:
                bpc, best_ep, wall = run_one_lm(
                    mode, dims, tr, va, V, seed=seed, epochs=args.epochs,
                    batch=256, lr=2e-3, wd=1e-3, drop=0.15, residual=True,
                    device=args.device, log=False)
                r = {"depth": L, "seed": seed, "mode": mode, "val_bpc": bpc,
                     "best_epoch": best_ep, "wall_s": wall,
                     "epochs": args.epochs, "hidden": args.hidden}
                rows.append(r)
                print(f"[L={L} s{seed} {mode:>11}] bpc={bpc:.3f} ({wall:.0f}s)",
                      flush=True)
                with open(OUT, "w") as f:
                    for rr in rows:
                        f.write(json.dumps(rr) + "\n")

    print("\n===== TRACTION SUMMARY (mean bpc over seeds) =====", flush=True)
    print(f"{'depth':>5} {'bp_float':>9} {'bp_dyadic':>9} {'relax_async':>11} "
          f"{'substrate tax':>13} {'rule tax':>9}")
    for L in args.depths:
        m = {}
        for mode in args.modes:
            vals = [r["val_bpc"] for r in rows if r["depth"] == L and r["mode"] == mode]
            m[mode] = sum(vals) / len(vals) if vals else float("nan")
        print(f"{L:>5} {m.get('bp_float', float('nan')):>9.3f} "
              f"{m.get('bp_dyadic', float('nan')):>9.3f} "
              f"{m.get('relax_async', float('nan')):>11.3f} "
              f"{m.get('bp_dyadic', 0) - m.get('bp_float', 0):>+13.3f} "
              f"{m.get('relax_async', 0) - m.get('bp_dyadic', 0):>+9.3f}", flush=True)


if __name__ == "__main__":
    main()
