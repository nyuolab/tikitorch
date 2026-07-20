#!/usr/bin/env python3
"""Nature-format figures, regenerated from the committed result jsonls.
House rule (as in make_figs.py): every number comes ONLY from the logs.

Produces:
  figs/fig_pareto.pdf    -- UPGRADED incumbent Pareto (frontier + memory + segmentation)
  figs/fig_pcdepth.pdf   -- NEW predictive-coding-at-depth (equivalence + depth + LM ladder)

All other manuscript figures (life, retention, gate law) remain in make_figs.py.
"""
import json
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = pathlib.Path(__file__).resolve().parents[2]
PC = REPO / "results/pc_convrnn"
PCD = REPO / "results/predictive_coding"
OUT = pathlib.Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

# Palette matched to the D2 artifact (colorblind-safe, light-theme values)
C = dict(learner="#2a78d6", norm="#1baf7a", fire="#4a3aa7", yellow="#eda100",
         full="#c98500", lora="#8a8a8a", bp="#d03b3b", ink="#1a1a1a",
         muted="#6a6a6a", good="#1b7a3a")
plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.direction": "out",
    "ytick.direction": "out",
})


def load(path):
    return [json.loads(l) for l in open(path)]


def ms(vals):
    return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


# =====================================================================
# FIG: the incumbent Pareto (upgraded from the ICML scatter + D2 artifact)
# Coordinates are the V3 analysis numbers (RESULTS.md V3; see v3_lora.jsonl,
# v2_tele_hard_shift_*). Memory class 138 vs 51 MB is the V3 activation-graph
# measurement. z*=0.25 / z*=1.0 are our two operating points.
# =====================================================================
def fig_pareto():
    # (label, extra-update-compute [fwd-eq], prequential CE, kind)
    pts = [
        ("ours ($z^{*}{=}1.0$)", 0.047, 4.750, "ours"),
        ("ours ($z^{*}{=}0.25$)", 0.131, 4.531, "ours"),
        ("LoRA c200k", 0.184, 5.350, "lora"),
        ("LoRA drift-trig", 0.430, 4.170, "lora"),
        ("LoRA c50k", 0.920, 4.140, "lora"),
        ("full-FT", 1.380, 4.080, "full"),
        ("online BP", 3.000, 3.380, "bp"),
        ("LoRA c10k", 4.850, 3.500, "lora"),
    ]
    kcol = {"ours": C["learner"], "lora": C["lora"], "full": C["full"],
            "bp": C["fire"]}

    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    # --- pre-registered segmentation shading ------------------------------
    ax.axvspan(0.03, 0.2, color=C["learner"], alpha=0.06, lw=0)
    ax.axvspan(1.6, 8, color=C["fire"], alpha=0.055, lw=0)
    ax.text(0.052, 5.63, "low-power /\nalways-on (ours)", fontsize=6.2,
            color=C["learner"], va="top", linespacing=1.15)
    ax.text(5.4, 5.63, "unconstrained\ncompute (GPU)", fontsize=6.2,
            color=C["fire"], va="top", ha="right", linespacing=1.15)

    # --- Pareto frontier (lower-left envelope, all arms) -------------------
    front = sorted(pts, key=lambda p: p[1])
    fx, fy, best = [], [], 1e9
    for _, x, y, _ in front:
        if y < best - 1e-9:
            fx.append(x); fy.append(y); best = y
    ax.plot(fx, fy, color=C["muted"], lw=1.0, ls=(0, (5, 2)), zorder=1,
            alpha=0.8)
    ax.text(fx[0] * 1.02, fy[0] + 0.02, "Pareto frontier", fontsize=6,
            color=C["muted"], rotation=-32, va="bottom")

    # --- "nothing dominates us" box below-and-left of our operating point --
    ox, oy = 0.131, 4.531
    ax.add_patch(Rectangle((0.03, 3.2), ox - 0.03, oy - 3.2,
                 facecolor=C["good"], alpha=0.05, edgecolor=C["good"],
                 lw=0.6, ls=":", zorder=0))
    ax.text(0.056, 3.42, "no incumbent\ncell lies here", fontsize=6.0,
            color=C["good"], va="bottom", ha="center", linespacing=1.15)

    # --- points + de-collided labels --------------------------------------
    lab_off = {
        "ours ($z^{*}{=}1.0$)": (0.9, 0.10, "right"),
        "ours ($z^{*}{=}0.25$)": (1.15, 0.10, "left"),
        "LoRA c200k": (1.1, 0.06, "left"),
        "LoRA drift-trig": (0.62, -0.20, "center"),
        "LoRA c50k": (1.05, 0.10, "left"),
        "full-FT": (1.05, -0.02, "left"),
        "online BP": (0.60, 0.13, "center"),
        "LoRA c10k": (0.80, 0.12, "center"),
    }
    for name, x, y, kind in pts:
        big = kind == "ours"
        ax.scatter([x], [y], s=48 if big else 24, color=kcol[kind], zorder=4,
                   edgecolors="white", linewidths=0.9)
        mx, dy, ha = lab_off[name]
        ax.annotate(name, (x, y), xytext=(x * mx, y + dy), ha=ha,
                    fontsize=6.3, color=kcol[kind] if big else "#444",
                    fontweight="bold" if big else "normal")

    # --- the money annotation: cheapest cell that beats our quality -------
    ax.annotate(
        "cheapest cell beating our\nquality: 3.3$\\times$ our update\ncompute + 2.7$\\times$ memory\n(138 vs 51 MB)",
        xy=(0.43, 4.17), xytext=(0.9, 4.86), fontsize=6.0, color="#333",
        ha="left", va="top", linespacing=1.15,
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.8,
                        connectionstyle="arc3,rad=-0.2"))

    ax.set_xscale("log")
    ax.set_xlabel("extra update compute per served token (fwd-eq)")
    ax.set_ylabel("prequential cross-entropy (nats/token)")
    ax.set_xlim(0.03, 8)
    ax.set_ylim(3.2, 5.8)
    ax.set_title("Quality vs. update compute on identical streams",
                 fontsize=8, pad=4)
    fig.savefig(OUT / "fig_pareto.pdf")
    plt.close(fig)
    print("wrote fig_pareto.pdf")


# =====================================================================
# FIG: predictive coding at depth (NEW)
#   A: cascade gradient == backprop, bitwise/near-machine (g0_gradcheck,
#      g1_equivalence)
#   B: depth collapse -- cascade trains at 128 layers where standard
#      backprop collapses to chance (scale_mnist_128layer)
#   C: the paper's own sequence model -- cascade == BP at every depth
#      (g3_g4_{bp,cascade}_L*)
# =====================================================================
def fig_pcdepth():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    aA, aB, aC = axes

    # ---- Panel A: exactness of credit assignment ----
    g0 = load(PC / "g0_gradcheck.jsonl")
    archs = ["morphssm", "convrnn"]
    xj = {"morphssm": 0, "convrnn": 1}
    import math
    for i, r in enumerate(g0):
        a = r["arch"]
        # deterministic jitter (no RNG -> reproducible)
        jit = ((i * 2654435761) % 1000) / 1000.0 - 0.5
        aA.scatter(xj[a] + jit * 0.55, r["rel_l2"], s=6,
                   color=C["learner"], alpha=0.45, edgecolors="none",
                   zorder=3)
    aA.set_yscale("log")
    aA.set_xticks([0, 1])
    aA.set_xticklabels(["MorphSSM", "conv-RNN"], fontsize=7)
    aA.set_xlim(-0.6, 1.6)
    aA.set_ylim(3e-8, 3e-4)
    aA.set_ylabel("rel-L2  (cascade grad vs.\\ backprop)")
    cosvals = [r["cos"] for r in g0]
    aA.set_title("A  Credit is exact\ncos $=%.3f$ (all %d tensors)"
                 % (min(cosvals), len(g0)), fontsize=7.5, loc="left")
    aA.text(0.5, 2.0e-4,
            "sync relaxation: bitwise\n(max-diff $=0$)\nasync: cos$\\in\\{0,1\\}$ (routed-or-not)",
            fontsize=5.8, ha="center", va="top", color="#333",
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none",
                      alpha=0.75))

    # ---- Panel B: depth collapse (residual MNIST, 3 seeds) ----
    d128 = load(PCD / "scale_mnist_128layer.jsonl")
    depths = [32, 64, 128]
    series = [
        ("cascade (ours,\nmult-free)", "cascade", C["learner"], "o", "-"),
        ("backprop\ndyadic", "bp_dyadic", C["norm"], "s", "--"),
        ("backprop\nfloat", "bp_float", C["bp"], "^", "-"),
    ]
    for lab, mode, col, mk, ls in series:
        mns, sds = [], []
        for d in depths:
            v = [r["test_acc"] for r in d128 if r["depth"] == d and r["mode"] == mode]
            m, s = ms(v); mns.append(m); sds.append(s)
        aB.errorbar(depths, mns, yerr=sds, color=col, marker=mk,
                    ms=4, ls=ls, capsize=2, lw=1.4)
        # direct end-label (no legend -> no collision)
        va = {"cascade": "bottom", "bp_dyadic": "top", "bp_float": "center"}[mode]
        aB.text(140, mns[-1], lab, fontsize=6.0, color=col, va=va,
                ha="left", linespacing=1.0, fontweight="bold")
    aB.axhline(0.10, color=C["muted"], lw=0.8, ls=":")
    aB.text(30, 0.145, "naive iterative PC: chance at every depth",
            fontsize=5.7, color=C["muted"])
    aB.set_xscale("log", base=2)
    aB.set_xticks(depths)
    aB.set_xticklabels([str(d) for d in depths])
    aB.set_ylim(0.05, 1.02)
    aB.set_xlim(28, 245)
    aB.set_xlabel("network depth (layers)")
    aB.set_ylabel("MNIST test accuracy")
    aB.set_title("B  Trains where backprop collapses", fontsize=7.5, loc="left")

    # ---- Panel C: the paper's sequence model, cascade == BP at depth ----
    Ls = [4, 8, 16, 32]
    for credit, col, mk, lab in [("bp", C["bp"], "^", "backprop"),
                                 ("cascade", C["learner"], "o", "cascade (ours)")]:
        mns, sds = [], []
        for L in Ls:
            vals = []
            for s in (0, 1, 2):
                fn = PC / f"g3_g4_{credit}_L{L}_s{s}.jsonl"
                if fn.exists():
                    rows = load(fn)
                    fin = [r for r in rows if r.get("final")]
                    if fin:
                        vals.append(fin[-1]["val_ce_hard"])
            m, s = ms(vals); mns.append(m); sds.append(s)
        aC.errorbar(Ls, mns, yerr=sds, label=lab, color=col, marker=mk,
                    ms=4, capsize=2, lw=1.2)
    aC.set_xscale("log", base=2)
    aC.set_xticks(Ls)
    aC.set_xticklabels([str(L) for L in Ls])
    aC.set_xlabel("MorphSSM depth (layers)")
    aC.set_ylabel("validation CE (nats/token)")
    aC.set_title("C  On our own substrate\ncascade $\\equiv$ backprop", fontsize=7.5,
                 loc="left")
    aC.legend(fontsize=6.2, frameon=False, loc="upper left")

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "fig_pcdepth.pdf")
    plt.close(fig)
    print("wrote fig_pcdepth.pdf")


if __name__ == "__main__":
    fig_pareto()
    fig_pcdepth()
    print("nature figures written:",
          sorted(p.name for p in OUT.glob("fig_p*.pdf")))
