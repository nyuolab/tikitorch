#!/bin/bash
# Drift/headroom test: the REAL non-stationary life (gate ON, cycling 4 domains,
# 2 cycles = one revisit) — does the closed EFTropical+DyadicAnchor CAPTURE
# headroom competitively with the AdamW ceiling, or only avoid collapse?
set -e
cd "$(dirname "$0")/../.."
C="--era-tokens 80000 --cycles 2 --delta 0.25 --lr-decay-every 2000 --seed 0 \
  --tele-every 50 --ckpt-every 100000000 --audit-frac 0.03"
python3 ontogenic_learner/experiments/st4_life.py --tag drift_adamw       --opt adamw        --anchor-lam 0.01 $C
python3 ontogenic_learner/experiments/st4_life.py --tag drift_eftropical  --opt eftropical   --anchor-lam 0.01 --ef-bound 8 $C
python3 ontogenic_learner/experiments/st4_life.py --tag drift_puretropical --opt puretropical --anchor-lam 0    $C
echo "DRIFT DONE"
python3 - <<'PY'
import json
from pathlib import Path
OUT = Path("ontogenic_learner/results/st_stream")
tags = [("drift_adamw", "AdamW + Adam-anchor (modern ref)"),
        ("drift_eftropical", "EFTropical + DyadicAnchor (closed)"),
        ("drift_puretropical", "PureTropicalSGD (closed baseline)")]
rows = {}
for l in (OUT / "v4_life.jsonl").open():
    r = json.loads(l)
    if r.get("tag") in dict(tags):
        rows[r["tag"]] = r
print(f"\n{'ARM':38s} {'HC':>7s} {'regret':>8s} {'fire%':>6s} {'amort':>7s}")
print("-" * 70)
for t, lab in tags:
    r = rows.get(t)
    if not r:
        print(f"{lab:38s}  (no data)"); continue
    hc = r.get("headroom_captured")
    hcs = f"{hc:>7.3f}" if hc is not None else "    n/a"
    print(f"{lab:38s} {hcs} {r['regret_vs_frozen']:>+8.3f} "
          f"{100*r['fire_rate']:>5.1f}% {r['amortized_mult']:>7.4f}")
print("-" * 70)
print("HC higher = better (fraction of frozen->oracle gap captured). "
      "Closed EF should approach the AdamW HC.")
PY
