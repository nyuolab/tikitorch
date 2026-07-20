"""Summarize the EF pre-cut: does error-feedback cure the PureTropicalSGD
collapse on the held-LR, fire-always stationary stretch?"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results/st_stream"
ARMS = [
    ("precut_puretropical", "PureTropicalSGD (collapse baseline)"),
    ("precut_ef_bound", "EFTropical  EF+bound  (full design)"),
    ("precut_ef_only", "EFTropical  EF, no bound"),
    ("precut_bound_only", "EFTropical  bound, no EF (naked sign)"),
]


def traj(tag):
    """Regret (ce_l-ce_f) and norm at ~start / mid / end of the tele file."""
    p = OUT / f"v4_tele_{tag}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    if not rows:
        return None
    def reg(r): return r["ce_l"] - r["ce_f"]
    k = len(rows)
    pts = {"start": rows[min(2, k - 1)], "mid": rows[k // 2], "end": rows[-1]}
    return {w: (round(reg(r), 4), round(r["norm"], 1), r["t"]) for w, r in pts.items()}, k


def verdict(tag):
    p = OUT / "v4_life.jsonl"
    if not p.exists():
        return None
    last = None
    for l in p.open():
        r = json.loads(l)
        if r.get("tag") == tag:
            last = r
    return last


print(f"\n{'ARM':44s} {'cum.regret':>10s} {'norm→':>8s} {'reg start→mid→end (ce_l-ce_f)':>34s}")
print("-" * 104)
for tag, label in ARMS:
    v = verdict(tag)
    t = traj(tag)
    if v is None or t is None:
        print(f"{label:44s}   (no data yet)")
        continue
    pts, k = t
    reg = v.get("regret_vs_frozen")
    normend = pts["end"][1]
    tr = " → ".join(f"{pts[w][0]:+.3f}" for w in ("start", "mid", "end"))
    print(f"{label:44s} {reg:>+10.4f} {normend:>8.0f}   {tr:>34s}")
print("-" * 104)
print("Read: PureTropicalSGD regret should CLIMB (collapse); EF arms should stay ~flat.")
print("Norm growth = runaway (bad); flat norm = equilibrium (good).")
