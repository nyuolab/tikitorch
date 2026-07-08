"""plot_exp3_curves.py -- LOCAL plotting of the Experiment 2 (deep-PC-traction contrast).

Reads the fetched per-arm jsonl records (the ones written by exp3_contrast.train_one:
per-eval dicts with fields exp=="exp3", arm_key, L, seed, step, train_bpb, bpb; and the
exp=="exp3_summary" records with finals + checks). Emits PNGs to an output dir:

  (i)  curves_L{L}.png       -- train + val bpb vs step, one panel per depth, all arms.
  (ii) gap_vs_depth.png      -- the PAPER FIGURE: gap curves vs depth --
         gap_pwl   = PC_pwl - BP_pwl                (~0, flat)
         gap_trop  = PC_trop - BP_trop              (bounded)
         gap_smooth_T{t} = PC_smooth_T{t} - BP_smooth   (grows w/ depth)
         gap_Tconv = PC_smooth_Tconv(2L) - BP_smooth    (convergence check, ~0)

Usage:  python3 plot_exp3_curves.py <jsonl_dir> <out_dir>
        (jsonl_dir defaults to results/predictive_coding/exp3_runs; out to <jsonl_dir>/plots)
"""
import os, sys, json, glob, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_COLORS = {
    "BP_pwl": "#1f77b4", "PC_pwl": "#4fa3d1",
    "BP_trop": "#2ca02c", "PC_trop": "#74c476",
    "BP_smooth": "#d62728",
}
SMOOTH_T_CMAP = ["#ff7f0e", "#8c564b", "#9467bd", "#e377c2"]  # per finite-T budget
TCONV_COLOR = "#7f7f7f"


def load(jsonl_dir):
    """Return (traj, summ):
       traj[(arm_key, L, seed)] -> list of (step, train_bpb, val_bpb)
       summ[(L, seed)] -> summary dict (finals, checks, ...)"""
    traj = defaultdict(list); summ = {}
    for f in sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl"))):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("exp") == "exp3":
                k = (r.get("arm_key", r.get("arm")), r["L"], r["seed"])
                traj[k].append((r["step"], r.get("train_bpb"), r.get("bpb")))
            elif r.get("exp") == "exp3_summary":
                summ[(r["L"], r["seed"])] = r
    for k in traj:
        traj[k].sort(key=lambda t: t[0])
    return traj, summ


def arm_color(arm_key, smoothTs):
    if arm_key in ARM_COLORS:
        return ARM_COLORS[arm_key]
    if arm_key.startswith("PC_smooth_Tconv"):
        return TCONV_COLOR
    if arm_key.startswith("PC_smooth_T"):
        t = int(arm_key.split("PC_smooth_T")[1])
        return SMOOTH_T_CMAP[smoothTs.index(t) % len(SMOOTH_T_CMAP)] if t in smoothTs else "#ff7f0e"
    return "#000000"


def plot_curves(traj, out_dir):
    """Per-depth panels of train (dashed) + val (solid) bpb vs step, all arms overlaid.
    Seeds averaged by plotting each seed lightly + the mean bold is overkill; we plot
    seed 0 (or the min seed) as the representative and note seeds in the title."""
    Ls = sorted({L for (_, L, _) in traj})
    if not Ls:
        print("no trajectory records found"); return
    smoothTs = sorted({int(a.split("PC_smooth_T")[1]) for (a, _, _) in traj
                       if a.startswith("PC_smooth_T") and not a.startswith("PC_smooth_Tconv")})
    for L in Ls:
        seeds = sorted({s for (a, ll, s) in traj if ll == L})
        arms = sorted({a for (a, ll, s) in traj if ll == L})
        fig, ax = plt.subplots(figsize=(9, 6))
        for a in arms:
            col = arm_color(a, smoothTs)
            s = seeds[0]
            pts = traj.get((a, L, s), [])
            if not pts: continue
            xs = [p[0] for p in pts]
            tr = [p[1] for p in pts]; vl = [p[2] for p in pts]
            if any(v is not None for v in vl):
                ax.plot(xs, vl, "-", color=col, label=f"{a} (val)", lw=1.8)
            if any(v is not None for v in tr):
                ax.plot(xs, tr, "--", color=col, alpha=0.55, lw=1.0)
        ax.set_xlabel("step"); ax.set_ylabel("bpb")
        ax.set_title(f"Exp2 train (dashed) + val (solid) bpb vs step  -- L={L}, seed={seeds[0]}")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        p = os.path.join(out_dir, f"curves_L{L}.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        print("wrote", p)


def _mean_final(traj, arm_key, L):
    """mean best/final val bpb over seeds for an arm at depth L (uses last eval's bpb)."""
    vals = []
    for (a, ll, s), pts in traj.items():
        if a == arm_key and ll == L and pts:
            v = [p[2] for p in pts if p[2] is not None]
            if v: vals.append(v[-1])
    return sum(vals) / len(vals) if vals else None


def plot_gap_vs_depth(traj, out_dir):
    Ls = sorted({L for (_, L, _) in traj})
    smoothTs = sorted({int(a.split("PC_smooth_T")[1]) for (a, _, _) in traj
                       if a.startswith("PC_smooth_T") and not a.startswith("PC_smooth_Tconv")})
    fig, ax = plt.subplots(figsize=(8, 6))

    def gap_curve(pc_fn, bp_key, label, color, marker="o"):
        xs, ys = [], []
        for L in Ls:
            pc = pc_fn(L); bp = _mean_final(traj, bp_key, L)
            if pc is None or bp is None: continue
            xs.append(L); ys.append(pc - bp)
        if xs: ax.plot(xs, ys, marker + "-", color=color, label=label, lw=1.8)

    gap_curve(lambda L: _mean_final(traj, "PC_pwl", L), "BP_pwl",
              "gap_pwl = PC_pwl - BP_pwl  (claim ~0 flat)", ARM_COLORS["PC_pwl"])
    gap_curve(lambda L: _mean_final(traj, "PC_trop", L), "BP_trop",
              "gap_trop = PC_trop - BP_trop  (bounded)", ARM_COLORS["PC_trop"])
    for i, T in enumerate(smoothTs):
        gap_curve(lambda L, T=T: _mean_final(traj, f"PC_smooth_T{T}", L), "BP_smooth",
                  f"gap_smooth_T{T} = PC_smooth_T{T} - BP_smooth  (grows w/ depth)",
                  SMOOTH_T_CMAP[i % len(SMOOTH_T_CMAP)], marker="s")
    gap_curve(lambda L: _mean_final(traj, f"PC_smooth_Tconv{2*L}", L), "BP_smooth",
              "gap_Tconv = PC_smooth_Tconv(2L) - BP_smooth  (CHECK ~0)", TCONV_COLOR, marker="^")

    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("depth L"); ax.set_ylabel("PC - BP  final val bpb gap (nats/bit)")
    ax.set_title("Exp2 paper figure: PC-to-backprop gap vs depth")
    ax.set_xticks(Ls); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "gap_vs_depth.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


def main():
    jdir = sys.argv[1] if len(sys.argv) > 1 else "results/predictive_coding/exp3_runs"
    odir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(jdir, "plots")
    os.makedirs(odir, exist_ok=True)
    traj, summ = load(jdir)
    print(f"loaded {len(traj)} arm-trajectories, {len(summ)} summaries from {jdir}")
    plot_curves(traj, odir)
    plot_gap_vs_depth(traj, odir)
    # echo any failing checks
    fails = [(k, c) for k, s in summ.items() for c in s.get("checks", []) if not c.get("passed")]
    if fails:
        print("!! FAILING CHECKS:")
        for k, c in fails: print("  ", k, c)
    else:
        print("all logged checks PASS")


if __name__ == "__main__":
    main()
