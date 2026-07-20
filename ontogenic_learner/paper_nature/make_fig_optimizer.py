#!/usr/bin/env python3
"""Closed-optimizer comparison on the gated drift life (headroom captured), from
the committed X1 parity-hunt jsonls. Honest picture: a multiply-free sign-based
optimizer (PureTropicalSGD) captures most of AdamW's plasticity, but no closed
arm reaches parity; the error-feedback family is collapse-immune but under-plastic."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
ST = REPO / "ontogenic_learner/results/st_stream"
OUT = pathlib.Path(__file__).parent / "figs"

HC = {}
for l in (ST / "v4_life.jsonl").open():
    r = json.loads(l)
    if str(r.get("tag", "")).startswith("x1"):
        HC[r["tag"]] = r.get("headroom_captured")

# (label, tag, closed?, color)
BARS = [
    ("AdamW\n(not closed)", "x1_adamw", False, "#8a8a8a"),
    ("PureTropical\n(closed)", "x1_puretropical", True, "#2a78d6"),
    ("EF\n(closed)", "x1_ef", True, "#1baf7a"),
    ("EF+adapt\n(closed)", "x1_hybrid", True, "#4a3aa7"),
]
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
xs = range(len(BARS))
vals = [HC.get(t) or 0 for _, t, _, _ in BARS]
cols = [c for *_, c in BARS]
ax.bar(xs, vals, color=cols, width=0.68, edgecolor="white", linewidth=0.8)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold")
adam = HC.get("x1_adamw") or 1
pure = HC.get("x1_puretropical") or 0
ax.axhline(adam, color="#8a8a8a", lw=0.8, ls=(0, (4, 3)))
ax.annotate(f"closed best =\n{100*pure/adam:.0f}% of AdamW", xy=(1, pure),
            xytext=(1.7, 0.30), fontsize=6.6, color="#2a78d6", ha="left",
            arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=0.8,
                            connectionstyle="arc3,rad=-0.2"))
ax.set_xticks(list(xs))
ax.set_xticklabels([b[0] for b in BARS], fontsize=6.8)
ax.set_ylabel("headroom captured (gated drift life)")
ax.set_ylim(0, 0.72)
ax.set_title("Multiply-free optimizers vs AdamW", fontsize=8, pad=4)
fig.savefig(OUT / "fig_optimizer.pdf")
print("wrote fig_optimizer.pdf; HC:", {k: HC[k] for k in sorted(HC) if k.startswith("x1_")})
