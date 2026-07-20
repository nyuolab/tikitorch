#!/usr/bin/env python3
"""Camera-ready combined multi-panel Nature figures, from committed jsonls
(house rule). Reuses the plot logic of make_figs.py / make_figs_nature.py:
  fig_validation.pdf  -- a: V2 headroom bars | b: gate law | c: honest Pareto
  fig_lifelong.pdf    -- a: prequential CE | b: fire rate | c: probes | d: norm
"""
import json
import pathlib
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = pathlib.Path(__file__).resolve().parents[2]
ST = REPO / "ontogenic_learner/results/st_stream"
OUT = pathlib.Path(__file__).parent / "figs"
DOMS = ("wiki", "shakespeare", "docs", "code")
BANDC = {"wiki": "#2a78d6", "shakespeare": "#eda100", "docs": "#1baf7a", "code": "#4a3aa7"}
C = dict(learner="#2a78d6", norm="#1baf7a", fire="#4a3aa7", yellow="#c98500",
         red="#d03b3b", grey="#8a8a8a", ink="#1a1a1a", good="#1b7a3a")
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "figure.dpi": 200,
                     "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False})


def panel(ax, letter, dx=-0.02, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=11, va="top", ha="right")


def bands(ax):
    era = 5208 * 96
    t = 0
    for c in range(5):
        order = DOMS[c % 4:] + DOMS[:c % 4]
        for d in order:
            ax.axvspan(t / 1e6, (t + era) / 1e6, color=BANDC[d], alpha=0.08, lw=0)
            t += era
    for c in range(1, 5):
        ax.axvline(c * era * 4 / 1e6, color="gray", lw=0.5, ls=":")


# =====================================================================
# FIG: validation (headroom bars | gate law | Pareto)
# =====================================================================
def fig_validation():
    v2 = [json.loads(l) for l in (ST / "v2_live.jsonl").open()]
    fig, (aA, aB, aC) = plt.subplots(1, 3, figsize=(7.1, 2.5))

    # --- a: V2 headroom captured per regime (gated arm, mean over seeds) ---
    regs = [("hard_shift", "hard\nshift"), ("drift", "drift"),
            ("novel_vocab", "novel\nvocab")]
    hcs = []
    for key, _ in regs:
        vals = [r["headroom_captured"] for r in v2
                if r.get("arm") == "gated" and r.get("regime") == key
                and r.get("gate_kind") in (None, "z")
                and r.get("gate_z") in (None, 0.25)   # canonical z*=0.25 main result
                and r.get("headroom_captured") is not None
                and not r.get("lr_decay_every")]
        hcs.append(st.mean(vals) if vals else 0.0)
    xs = range(len(regs))
    aA.bar(xs, hcs, color=C["learner"], width=0.62, edgecolor="white", linewidth=0.8)
    for x, h in zip(xs, hcs):
        aA.text(x, h + 0.015, f"{h:.2f}", ha="center", fontsize=7.5, fontweight="bold")
    aA.set_xticks(list(xs)); aA.set_xticklabels([r[1] for r in regs], fontsize=7)
    aA.set_ylim(0, 0.8); aA.set_ylabel("headroom captured")
    aA.set_title("live learning pays", fontsize=8, loc="center", pad=2)
    panel(aA, "a")

    # --- b: gate law (fire rate vs threshold; z-gate fixed-quantile vs margin) ---
    zs = [0.25, 0.5, 1.0, 2.0]; zh, zsn = [], []
    for z in zs:
        for reg, acc in (("stationary", zsn), ("hard_shift", zh)):
            c = [r for r in v2 if r.get("arm") == "gated"
                 and r.get("gate_kind") in (None, "z") and r["regime"] == reg
                 and (r.get("gate_z") == z or (z == 0.25 and r.get("gate_z") is None))
                 and r["seed"] == 0 and not r.get("lr_decay_every")]
            acc.append(c[0]["fire_rate"] if c else None)
    ds = [0.25, 0.5, 1.0]; dh, dsn = [], []
    for d in ds:
        for reg, acc in (("stationary", dsn), ("hard_shift", dh)):
            c = [r for r in v2 if r.get("gate_kind") == "margin"
                 and r["regime"] == reg and r.get("delta") == d]
            acc.append(c[0]["fire_rate"] if c else None)
    aB.plot(zs, zh, "o-", color=C["red"], ms=3.5, label="$z$-gate, shift")
    aB.plot(zs, zsn, "o--", color=C["red"], ms=3.5, alpha=0.55, label="$z$-gate, stationary")
    aB.plot(ds, dh, "s-", color=C["learner"], ms=3.5, label="margin, shift")
    aB.plot(ds, dsn, "s--", color=C["learner"], ms=3.5, alpha=0.55, label="margin, stationary")
    aB.set_yscale("log"); aB.set_xlabel(r"threshold ($z^{*}$ or $\delta$ nats)")
    aB.set_ylabel("fire rate"); aB.legend(frameon=False, fontsize=5.8, loc="lower left")
    aB.set_title("the gate law", fontsize=8, loc="center", pad=2)
    panel(aB, "b")

    # --- c: honest Pareto (quality vs update compute) ---
    pts = [("ours", 0.131, 4.531, C["learner"], True),
           ("ours", 0.047, 4.750, C["learner"], True),
           ("LoRA c200k", 0.184, 5.35, C["grey"], False),
           ("LoRA drift", 0.43, 4.17, C["grey"], False),
           ("LoRA c50k", 0.92, 4.14, C["grey"], False),
           ("full-FT", 1.38, 4.08, C["yellow"], False),
           ("online BP", 3.0, 3.38, C["fire"], False),
           ("LoRA c10k", 4.85, 3.50, C["grey"], False)]
    front = sorted(pts, key=lambda p: p[1]); fx, fy, best = [], [], 1e9
    for _, x, y, _, _ in front:
        if y < best - 1e-9:
            fx.append(x); fy.append(y); best = y
    aC.plot(fx, fy, color=C["grey"], lw=0.9, ls=(0, (5, 2)), zorder=1, alpha=0.8)
    aC.add_patch(Rectangle((0.03, 3.2), 0.131 - 0.03, 4.531 - 3.2, facecolor=C["good"],
                 alpha=0.05, edgecolor=C["good"], lw=0.5, ls=":", zorder=0))
    for n, x, y, col, big in pts:
        aC.scatter([x], [y], s=40 if big else 20, color=col, zorder=4,
                   edgecolors="white", linewidths=0.8)
    aC.annotate("ours", (0.131, 4.531), xytext=(0.05, 4.2), fontsize=6.5,
                color=C["learner"], fontweight="bold")
    aC.annotate("3.3$\\times$ compute,\n2.7$\\times$ memory to\nbeat our quality",
                xy=(0.43, 4.17), xytext=(0.9, 4.9), fontsize=5.6, color="#444",
                ha="left", va="top", arrowprops=dict(arrowstyle="-|>", color="#666",
                lw=0.7, connectionstyle="arc3,rad=-0.2"))
    aC.set_xscale("log"); aC.set_xlim(0.03, 8); aC.set_ylim(3.2, 5.7)
    aC.set_xlabel("extra compute / token (fwd-eq)")
    aC.set_ylabel("prequential CE (nats/tok)")
    aC.set_title("not dominated by LoRA", fontsize=8, loc="center", pad=2)
    panel(aC, "c")

    fig.tight_layout(w_pad=1.8)
    fig.savefig(OUT / "fig_validation.pdf")
    plt.close(fig)
    print("wrote fig_validation.pdf; V2 headroom:", [round(h, 3) for h in hcs])


# =====================================================================
# FIG: lifelong (CE | fire | probes | norm), 4 stacked panels
# =====================================================================
def fig_lifelong():
    tele = [json.loads(l) for l in (ST / "v4_tele_main.jsonl").open()]
    probes = [json.loads(l) for l in (ST / "v4_probes_main.jsonl").open() if "ce_wiki" in l]
    fig, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=(6.8, 6.2), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1, 2, 1]})
    t = [r["t"] / 1e6 for r in tele]
    # a: prequential CE
    a1.plot(t, [r["ce_f"] for r in tele], color="#999", lw=0.5, ls="--", label="frozen twin")
    a1.plot(t, [r["ce_l"] for r in tele], color=C["learner"], lw=0.7, label="live learner")
    bands(a1); a1.set_ylabel("prequential CE\n(nats/tok)"); a1.set_ylim(2, 14)
    a1.legend(loc="upper right", frameon=False, fontsize=7); panel(a1, "a")
    # b: fire rate
    prev_f = prev_n = 0; fr_t, fr = [], []
    for i in range(0, len(tele), 40):
        r = tele[min(i + 39, len(tele) - 1)]; n = r["t"] / 96; f = r["fires"]
        if n > prev_n:
            fr_t.append(r["t"] / 1e6); fr.append((f - prev_f) / (n - prev_n))
        prev_f, prev_n = f, n
    a2.fill_between(fr_t, fr, color=C["fire"], alpha=0.25, lw=0)
    a2.plot(fr_t, fr, color=C["fire"], lw=0.7); bands(a2)
    for c, k in enumerate([1275, 111, 189, 96, 45]):
        a2.text((c + 0.5) * 2.0, 0.82, f"{k}", ha="center", fontsize=6.5, color="#333")
    a2.set_ylim(0, 1); a2.set_ylabel("fire rate\n(kicks/cycle)"); panel(a2, "b")
    # c: retention probes
    pt = [r["t"] / 1e6 for r in probes]
    for d in DOMS:
        a3.plot(pt, [r[f"ce_{d}"] for r in probes], color=BANDC[d], marker="o",
                ms=2.5, lw=1.0, label="shak" if d == "shakespeare" else d)
    bands(a3); a3.set_ylabel("probe CE\n(nats/tok)")
    a3.legend(ncol=4, frameon=False, fontsize=6.5, loc="upper right"); panel(a3, "c")
    # d: weight norm
    a4.plot(pt, [r["norm"] for r in probes], color=C["norm"], lw=1.3); bands(a4)
    a4.set_ylabel(r"$\|\theta\|$"); a4.set_xlabel("tokens served (millions)")
    a4.annotate("+2.3%, equilibrium", xy=(pt[-1], probes[-1]["norm"]),
                xytext=(6.2, probes[0]["norm"] + 5), fontsize=6.5, color=C["norm"])
    panel(a4, "d")
    fig.tight_layout(h_pad=0.5)
    fig.savefig(OUT / "fig_lifelong.pdf")
    plt.close(fig)
    print("wrote fig_lifelong.pdf")


if __name__ == "__main__":
    fig_validation()
    fig_lifelong()
