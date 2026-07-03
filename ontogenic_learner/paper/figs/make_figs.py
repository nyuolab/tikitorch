"""Regenerate paper figures from the G1 result logs. Run from repo root:
    python3 ontogenic_learner/paper/figs/make_figs.py
Reads results/g1_two_speed/{g1_full_g12_summary.json, g1_full_g12.jsonl}; writes PDFs here."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../../results/g1_two_speed")

summary = json.load(open(os.path.join(RES, "g1_full_g12_summary.json")))["summary"]

# mean applied updates per arm (energy proxy) from the event log
upd = defaultdict(lambda: [0, 0])
for line in open(os.path.join(RES, "g1_full_g12.jsonl")):
    r = json.loads(line)
    if r.get("event") == "done":
        upd[r["arm"]][0] += r["updates"]
        upd[r["arm"]][1] += 1
updates = {a: u / n for a, (u, n) in upd.items()}
BUDGET = 1800.0  # 600/domain x 3 domains

STYLE = {
    "cascade":            ("s", "#888888", "cascade (seq. BP)"),
    "rehearsal":          ("o", "#d62728", "rehearsal 25%"),
    "rehearsal_bal":      ("o", "#ff7f0e", "rehearsal 25% bal."),
    "rehearsal50":        ("v", "#d62728", "rehearsal 50%"),
    "rehearsal50_bal":    ("v", "#ff7f0e", "rehearsal 50% bal."),
    "two_speed_casc_z000": ("*", "#2ca02c", r"two-speed $\tau$=0.0"),
    "two_speed_casc":      ("*", "#17becf", r"two-speed $\tau$=0.25"),
    "two_speed_casc_z075": ("*", "#1f77b4", r"two-speed $\tau$=0.75"),
    "two_speed_casc_z150": ("*", "#9467bd", r"two-speed $\tau$=1.5"),
}

# ---------------------------------------------------- Fig 1: the Pareto plane
fig, ax = plt.subplots(figsize=(5.2, 3.9))
for arm, (mk, c, lbl) in STYLE.items():
    if arm not in summary:
        continue
    s = summary[arm]
    size = 40 + 260 * (updates.get(arm, BUDGET) / BUDGET)
    ax.scatter(s["mean_plasticity_frac"], s["mean_forgetting_bpc"], marker=mk, c=c,
               s=size, label=lbl, edgecolors="k", linewidths=0.4, zorder=3)
ax.set_xlabel("plasticity (fraction of sequential-BP adaptation)")
ax.set_ylabel("forgetting (bpc regained on prior domains)")
ax.axhline(0, color="k", lw=0.5, alpha=0.4)
ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
ax.set_title("Retention–plasticity Pareto (marker size = energy)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "pareto.pdf"))

# ---------------------------------------------------- Fig 2: the tau dial
taus = [0.0, 0.25, 0.75, 1.5]
arms = ["two_speed_casc_z000", "two_speed_casc", "two_speed_casc_z075",
        "two_speed_casc_z150"]
fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
ax1.plot(taus, [summary[a]["mean_forgetting_bpc"] for a in arms], "o-",
         c="#d62728", label="forgetting (bpc)")
ax1.plot(taus, [1 - summary[a]["mean_plasticity_frac"] for a in arms], "s-",
         c="#1f77b4", label="plasticity loss (1$-$frac)")
ax1.set_xlabel(r"surprise gate threshold $\tau$ (z-score)")
ax1.set_ylabel("forgetting / plasticity loss")
ax2 = ax1.twinx()
ax2.plot(taus, [100 * updates[a] / BUDGET for a in arms], "^--", c="#2ca02c",
         label="energy (% of budget)")
ax2.set_ylabel("applied updates (% of matched budget)", color="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="center left")
ax1.set_title(r"One scalar, three currencies: the $\tau$ dial", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "dial.pdf"))
print("wrote pareto.pdf, dial.pdf")
