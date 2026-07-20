#!/usr/bin/env python3
"""EF pre-cut ablation figure: error feedback cures the sign-SGD collapse.
Regret (ce_l - ce_f) and weight-norm trajectories over the held-LR, fire-always,
stationary collapse stretch. From the committed pre-cut tele jsonls."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
ST = REPO / "ontogenic_learner/results/st_stream"
OUT = pathlib.Path(__file__).parent / "figs"

ARMS = [
    ("precut_puretropical", "PureTropicalSGD (baseline)", "#d03b3b", "-"),
    ("precut_bound_only", "sign + bound, no EF", "#c98500", "--"),
    ("precut_ef_only", "error feedback (ours)", "#2a78d6", "-"),
    ("precut_ef_bound", "error feedback + bound", "#1baf7a", ":"),
]
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.4,
                     "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
for tag, lab, col, ls in ARMS:
    p = ST / f"v4_tele_{tag}.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.open()]
    t = [r["t"] / 1e3 for r in rows]
    reg = [r["ce_l"] - r["ce_f"] for r in rows]
    nrm = [r["norm"] for r in rows]
    a1.plot(t, reg, color=col, ls=ls, label=lab)
    a2.plot(t, nrm, color=col, ls=ls, label=lab)

a1.axhline(0, color="#888", lw=0.7, ls=(0, (4, 3)))
a1.set_xlabel("tokens served (thousands)")
a1.set_ylabel("regret vs frozen (nats/token)")
a1.set_title("A  Error feedback cures the collapse", fontsize=8, loc="left")
a1.legend(fontsize=6.3, frameon=False, loc="upper left")
a1.text(0.98, 0.03, "held-LR · every window kicks · stationary · anchorless",
        transform=a1.transAxes, fontsize=5.6, color="#888", ha="right")

a2.set_xlabel("tokens served (thousands)")
a2.set_ylabel(r"weight norm $\|\theta\|$")
a2.set_title("B  ...and holds norm equilibrium", fontsize=8, loc="left")

fig.tight_layout(w_pad=1.8)
fig.savefig(OUT / "fig_efablation.pdf")
print("wrote", OUT / "fig_efablation.pdf")
