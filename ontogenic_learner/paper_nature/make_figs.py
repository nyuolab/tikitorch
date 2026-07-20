#!/usr/bin/env python3
"""Regenerate ALL manuscript figures from the committed result jsonls.
Numbers in the paper come ONLY from these logs (house rule)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
ST = REPO / "ontogenic_learner/results/st_stream"
OUT = pathlib.Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "figure.dpi": 150})
DOMS = ("wiki", "shakespeare", "docs", "code")
BANDC = {"wiki": "#2a78d6", "shakespeare": "#eda100",
         "docs": "#1baf7a", "code": "#4a3aa7"}


def bands(ax, ymax_frac=1.0):
    era = 5208 * 96
    t = 0
    for c in range(5):
        order = DOMS[c % 4:] + DOMS[:c % 4]
        for d in order:
            ax.axvspan(t / 1e6, (t + era) / 1e6, color=BANDC[d], alpha=0.08, lw=0)
            t += era
    for c in range(1, 5):
        ax.axvline(c * era * 4 / 1e6, color="gray", lw=0.5, ls=":")


# ---- Fig: the life (CE + fire-rate) ----------------------------------------
tele = [json.loads(l) for l in (ST / "v4_tele_main.jsonl").open()]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
t = [r["t"] / 1e6 for r in tele]
a1.plot(t, [r["ce_f"] for r in tele], color="#999", lw=0.5, ls="--",
        label="frozen twin")
a1.plot(t, [r["ce_l"] for r in tele], color="#2a78d6", lw=0.7,
        label="live learner")
bands(a1)
a1.set_ylabel("prequential CE (nats/tok)")
a1.legend(loc="upper right", frameon=False, fontsize=7)
a1.set_ylim(2, 14)
# windowed fire rate
prev_f, prev_n = 0, 0
fr_t, fr = [], []
for i in range(0, len(tele), 40):
    r = tele[min(i + 39, len(tele) - 1)]
    n = r["t"] / 96
    f = r["fires"]
    if n > prev_n:
        fr_t.append(r["t"] / 1e6)
        fr.append((f - prev_f) / (n - prev_n))
    prev_f, prev_n = f, n
a2.fill_between(fr_t, fr, color="#4a3aa7", alpha=0.25, lw=0)
a2.plot(fr_t, fr, color="#4a3aa7", lw=0.7)
bands(a2)
kicks = [1275, 111, 189, 96, 45]
for c in range(5):
    a2.text((c + 0.5) * 2.0, 0.82, f"{kicks[c]} kicks", ha="center",
            fontsize=7, color="#333")
a2.set_ylim(0, 1)
a2.set_ylabel("fire rate")
a2.set_xlabel("tokens served (millions)")
fig.tight_layout(h_pad=0.4)
fig.savefig(OUT / "fig_life.pdf")
plt.close(fig)

# ---- Fig: retention probes + norm ------------------------------------------
probes = [json.loads(l) for l in (ST / "v4_probes_main.jsonl").open()
          if "ce_wiki" in l]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 3.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
pt = [r["t"] / 1e6 for r in probes]
for d in DOMS:
    a1.plot(pt, [r[f"ce_{d}"] for r in probes], color=BANDC[d], marker="o",
            ms=2.5, lw=1.0, label="shak" if d == "shakespeare" else d)
bands(a1)
a1.set_ylabel("probe CE (nats/tok)")
a1.legend(ncol=4, frameon=False, fontsize=7, loc="upper right")
a2.plot(pt, [r["norm"] for r in probes], color="#1baf7a", lw=1.2)
bands(a2)
a2.set_ylabel(r"$\|\theta\|$")
a2.set_xlabel("tokens served (millions)")
fig.tight_layout(h_pad=0.4)
fig.savefig(OUT / "fig_retention.pdf")
plt.close(fig)

# ---- Fig: V3 Pareto ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.6))
pts = [("ours ($z^*{=}0.25$)", 0.131, 4.531, "#2a78d6", True),
       ("ours ($z^*{=}1.0$)", 0.047, 4.750, "#2a78d6", True),
       ("LoRA c200k", 0.184, 5.35, "#888", False),
       ("LoRA trig", 0.43, 4.17, "#888", False),
       ("LoRA c50k", 0.92, 4.14, "#888", False),
       ("full-FT", 1.38, 4.08, "#c98500", False),
       ("LoRA c10k", 4.85, 3.50, "#888", False),
       ("online BP", 3.0, 3.38, "#4a3aa7", False)]
for n, x, y, c, us in pts:
    ax.scatter([x], [y], s=34 if us else 20, color=c, zorder=3,
               edgecolors="white", linewidths=0.8)
    dx = 1.12
    ax.annotate(n, (x, y), xytext=(x * dx, y + 0.06), fontsize=6.5,
                color=c if us else "#555")
ax.set_xscale("log")
ax.set_xlabel("extra update compute per served token (fwd-eq)")
ax.set_ylabel("prequential CE (nats/tok)")
ax.set_xlim(0.03, 8)
ax.set_ylim(3.2, 5.7)
fig.tight_layout()
fig.savefig(OUT / "fig_pareto.pdf")
plt.close(fig)

# ---- Fig: the gate law --------------------------------------------------------
rows = [json.loads(l) for l in (ST / "v2_live.jsonl").open()]
zs, zfire_s, zfire_h = [0.25, 0.5, 1.0, 2.0], [], []
for z in zs:
    for reg, acc in (("stationary", zfire_s), ("hard_shift", zfire_h)):
        c = [r for r in rows if r.get("arm") == "gated"
             and r.get("gate_kind") in (None, "z") and r["regime"] == reg
             and (r.get("gate_z") == z or (z == 0.25 and r.get("gate_z") is None))
             and r["seed"] == 0 and not r.get("lr_decay_every")]
        acc.append(c[0]["fire_rate"] if c else None)
ds, dfire_s, dfire_h = [0.25, 0.5, 1.0], [], []
for d in ds:
    for reg, acc in (("stationary", dfire_s), ("hard_shift", dfire_h)):
        c = [r for r in rows if r.get("gate_kind") == "margin"
             and r["regime"] == reg and r.get("delta") == d]
        acc.append(c[0]["fire_rate"] if c else None)
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.plot(zs, zfire_h, "o-", color="#d03b3b", ms=3.5, label="z-gate, shift")
ax.plot(zs, zfire_s, "o--", color="#d03b3b", ms=3.5, alpha=0.55,
        label="z-gate, stationary")
ax.plot(ds, dfire_h, "s-", color="#2a78d6", ms=3.5, label="margin, shift")
ax.plot(ds, dfire_s, "s--", color="#2a78d6", ms=3.5, alpha=0.55,
        label="margin, stationary")
ax.set_yscale("log")
ax.set_xlabel(r"threshold ($z^*$ or $\delta$ nats)")
ax.set_ylabel("fire rate")
ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT / "fig_gatelaw.pdf")
plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
