#!/usr/bin/env python3
"""Companion manuscript ("Digital Ontogeny") display items — the E6 arc.

House rule: every plotted value recomputed from committed jsonls under
ontogenic_learner/results/. Namespace fig_ont_* (the main paper's figure
scripts are untouched).

Figures:
  fig_ont_arrival.pdf   (a) arrival-era return trajectories: untreated vs
                        treated adults + treated/untreated newborn bands
                        (b) the 2x2 age-penalty abolition summary
  fig_ont_branch.pdf    3-arm falsifier: era-15 entropy after natural /
                        familiar / scrambled era-14
  fig_ont_freeze.pdf    D1: instantaneous CE through the 50k serve-only
                        window and the gradient-on recovery
  fig_ont_dose.pdf      dose-response: arrival return + resident tax vs
                        pull strength (screen + full scale)
  fig_ont_seesaw.pdf    orthogonality: non-resident probe CE by arm
                        (cure untouched) from the screen endpoints
"""
import json
import pathlib
import statistics as st

import matplotlib.pyplot as plt

import natstyle as ns

ns.apply()
HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def era_series(tele, era):
    return [r for r in tele if r["era"] == era]


def era_mean(tele, era, key="ret1k"):
    v = era_series(tele, era)
    return st.mean(r[key] for r in v) if v else None


# ---------------------------------------------------------------- arrival --
def fig_arrival():
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(ns.DOUBLE, 2.6), width_ratios=[1.6, 1.0])

    # (a) smoothed within-era arrival trajectories, full scale
    def smooth(xs, k=15):
        out = []
        for i in range(len(xs)):
            w = xs[max(0, i - k):i + 1]
            out.append(sum(w) / len(w))
        return out

    for name, color, label in (
            ("e6_tier1/t1_s1.jsonl", ns.PAL["frozen"], None),
            ("e6_tier1/t1_s2.jsonl", ns.PAL["frozen"], None),
            ("e6_tier1/t1_s3.jsonl", ns.PAL["frozen"], "untreated adult (n=4 incl. E5)"),
            ("e6_tier1/lrfixed_s0.jsonl", ns.PAL["frozen"], None)):
        t = rows(RES / name)
        e = era_series(t, 14)
        if e:
            ax.plot(range(len(e)), smooth([r["ret1k"] for r in e]),
                    color=color, lw=0.8,
                    label=label)
    for i, name in enumerate(("epw_s0", "epw_s1", "epw_s2")):
        t = rows(RES / f"e6_tier3/{name}.jsonl")
        e = era_series(t, 14)
        ax.plot(range(len(e)), smooth([r["ret1k"] for r in e]),
                color=ns.PAL["ours"], lw=1.1,
                label="treated adult (ent$_k$=$-4$)" if i == 0 else None)
    ax.set_xlabel("steps into arrival era (×10³)")
    ax.set_ylabel("return / 10³ steps")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    ns.panel(ax, "a")

    # (b) the 2x2: matched-treatment comparison (micro-grid + adults)
    def grid_means(arm):
        vals = []
        for s in range(5):
            t = rows(RES / f"e6_microgrid/{arm}_s{s}.jsonl")
            vals.append(st.mean(r["ret1k"] for r in t))
        return vals

    groups = [
        ("newborn\nuntreated", grid_means("nb-base"), ns.PAL["frozen"]),
        ("newborn\ntreated", grid_means("nb-pull"), ns.PAL["ours"]),
        ("adult\nuntreated", [0.0] * 8, ns.PAL["frozen"]),
        ("adult\ntreated", [3.117, 3.367, 3.019], ns.PAL["ours"]),
    ]
    for i, (lab, vals, c) in enumerate(groups):
        ax2.scatter([i + 0.08 * (j - len(vals) / 2) for j in range(len(vals))],
                    vals, s=6, color=c, zorder=3, clip_on=False)
        m = st.mean(vals)
        ax2.hlines(m, i - 0.28, i + 0.28, color=c, lw=1.4)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([g[0] for g in groups], fontsize=5.5)
    ax2.set_ylabel("arrival return / 10³ steps")
    ax2.axhline(0, color="0.85", lw=0.5, zorder=0)
    ns.panel(ax2, "b")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ont_arrival.pdf")
    plt.close(fig)


# ----------------------------------------------------------------- branch --
def fig_branch():
    fig, ax = plt.subplots(figsize=(ns.SINGLE, 2.2))
    src = rows(RES / "e6_tier1/t1_s3.jsonl")
    fam = rows(RES / "e6_branches/branch_familiar.jsonl")
    scr = rows(RES / "e6_branches/branch_scrambled.jsonl")
    arms = [
        ("natural\n(novel world)", era_mean(src, 15, "piH"), True),
        ("familiar\n(extra residency)", era_mean(fam, 15, "piH"), False),
        ("scrambled\n(unlearnable)", era_mean(scr, 15, "piH"), True),
    ]
    for i, (lab, v, restored) in enumerate(arms):
        ax.bar(i, v, width=0.55,
               color=ns.PAL["ours"] if restored else ns.PAL["frozen"])
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=5.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([a[0] for a in arms], fontsize=5.5)
    ax.set_ylabel("era-15 acting entropy (nats)")
    ax.set_title("after an era-13 entropy lock, era 14 spent as:",
                 fontsize=6, pad=3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ont_branch.pdf")
    plt.close(fig)


# ----------------------------------------------------------------- freeze --
def fig_freeze():
    fig, axes = plt.subplots(1, 2, figsize=(ns.DOUBLE, 2.2), sharey=True)
    for ax, (name, era, ttl) in zip(axes, (
            ("shadow_foraging", 3, "foraging revisit"),
            ("shadow_minatar", 5, "minatar revisit"))):
        t = rows(RES / f"e6_branches/{name}.jsonl")
        seq = [r for r in t if r["era"] in (era - 1, era, era + 1)]
        inst, xs, frozen = [], [], []
        for a, b in zip(seq, seq[1:]):
            if b["era"] != era:
                continue
            inst.append((b["ce"] * b["t"] - a["ce"] * a["t"]) / (b["t"] - a["t"]))
            xs.append(len(xs))
            frozen.append(b.get("frozen", 0))
        fz = [i for i, f in enumerate(frozen) if f]
        ax.plot(xs, inst, color=ns.PAL["ours"], lw=0.9)
        if fz:
            ax.axvspan(min(fz), max(fz), color="0.92", zorder=0)
            ax.text((min(fz) + max(fz)) / 2, max(inst) * 0.94,
                    "learning frozen", ha="center", fontsize=5.5, color="0.4")
        ax.set_title(ttl, fontsize=6)
        ax.set_xlabel("steps into revisit era (×10³)")
    axes[0].set_ylabel("instantaneous obs-CE (nats)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ont_freeze.pdf")
    plt.close(fig)


# ------------------------------------------------------------------- dose --
def fig_dose():
    fig, ax = plt.subplots(figsize=(ns.SINGLE, 2.3))
    ep = [json.loads(l) for l in
          open(RES / "e6_screen" / "e6_screen_endpoints.jsonl")]

    def screen_arrivals(arm):
        return [r["ignite_max"] for r in ep if r["arm"] == arm
                and isinstance(r.get("ignite_max"), (int, float))]

    pts = [
        ("0 (base)", screen_arrivals("base"), ns.PAL["frozen"]),
        ("$2^{-4}$", screen_arrivals("ent-pull-weak"), ns.PAL["ours"]),
        ("$2^{-3}$", screen_arrivals("ent-pull"), ns.PAL["compare"]),
    ]
    for i, (lab, vals, c) in enumerate(pts):
        ax.scatter([i] * len(vals), vals, s=6, color=c, zorder=3)
        ax.hlines(st.mean(vals), i - 0.22, i + 0.22, color=c, lw=1.4)
    # full-scale overlays
    ax.scatter([1] * 3, [3.117, 3.367, 3.019], s=14, marker="D",
               facecolors="none", edgecolors=ns.PAL["ours"],
               label="full scale (12M)")
    ax.scatter([2] * 3, [2.346, 2.251, 2.289], s=14, marker="D",
               facecolors="none",
               edgecolors=ns.PAL["compare"])
    ax.set_xticks(range(3))
    ax.set_xticklabels([p[0] for p in pts])
    ax.set_xlabel("pull strength $\\lambda$")
    ax.set_ylabel("best babyai era return / 10³")
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ont_dose.pdf")
    plt.close(fig)


# ----------------------------------------------------------------- seesaw --
def fig_seesaw():
    fig, ax = plt.subplots(figsize=(ns.SINGLE, 2.2))
    ep = [json.loads(l) for l in
          open(RES / "e6_screen" / "e6_screen_endpoints.jsonl")]

    def nonres(arm):
        out = []
        for r in ep:
            if r["arm"] == arm and r.get("nonres"):
                vals = [v for k, v in r["nonres"].items()
                        if k in ("foraging", "minatar")]
                if vals:
                    out.append(sum(vals) / len(vals))
        return out

    arms = [("base", "untreated"), ("ent-pull-weak", "cure ($2^{-4}$)"),
            ("anchor", "EWC anchor")]
    for i, (a, lab) in enumerate(arms):
        vals = nonres(a)
        c = ns.PAL["ours"] if "pull" in a else ns.PAL["frozen"]
        ax.scatter([i] * len(vals), vals, s=6, color=c, zorder=3)
        ax.hlines(st.mean(vals), i - 0.22, i + 0.22, color=c, lw=1.4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms], fontsize=5.5)
    ax.set_ylabel("non-resident probe obs-CE (nats)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ont_seesaw.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_arrival()
    fig_branch()
    fig_freeze()
    fig_dose()
    fig_seesaw()
    print("wrote", sorted(p.name for p in FIGS.glob("fig_ont_*.pdf")))
