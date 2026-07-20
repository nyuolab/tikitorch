#!/bin/bash
# X1b: start from the PLASTIC champion (PureTropicalSGD, steps every fire) and
# test bigger step + collapse-insurance, aiming above 0.41 toward AdamW's 0.618.
set -e
cd "$(dirname "$0")/../.."
C="--domains wiki,shakespeare,docs,code --era-tokens 80000 --cycles 2 --delta 0.25 \
  --lr-decay-every 2000 --seed 0 --anchor none --tele-every 50 \
  --ckpt-every 100000000 --audit-frac 0.03"
run () { echo "=== $1 ==="; python3 ontogenic_learner/experiments/st4_life.py --tag "$1" $2 $C; }
run x1b_puretrop_s8     "--opt puretropical --lr-shift 8"                              # bigger step
run x1b_puretrop_s7     "--opt puretropical --lr-shift 7"                              # bigger still
run x1b_naked_mom_bound "--opt hybrid --no-ef --no-adapt --momentum --ef-bound 8 --lr-shift 8"  # PureTrop-like + bound
run x1b_naked_adapt_mom "--opt hybrid --no-ef --momentum --lr-shift 8"                 # adaptive step OFF the EF throttle
echo "X1B DONE"
python3 - <<'PY'
import json
from pathlib import Path
OUT = Path("ontogenic_learner/results/st_stream")
tags = [("x1_adamw","AdamW ref (0.618)"),("x1_puretropical","PureTropical s10 (0.411)"),
        ("x1b_puretrop_s8","PureTropical s8 (bigger)"),("x1b_puretrop_s7","PureTropical s7 (biggest)"),
        ("x1b_naked_mom_bound","sign+mom+bound (closed insured)"),
        ("x1b_naked_adapt_mom","sign+adapt+mom (no EF)")]
rows={}
for l in (OUT/"v4_life.jsonl").open():
    r=json.loads(l)
    if r.get("tag") in dict(tags): rows[r["tag"]]=r
print(f"\n{'ARM':34s} {'HC':>7s} {'regret':>8s} {'fire%':>6s}")
print("-"*60)
for t,lab in tags:
    r=rows.get(t)
    if not r: print(f"{lab:34s}  (no data)"); continue
    hc=r.get('headroom_captured'); hcs=f"{hc:>7.3f}" if hc is not None else "    n/a"
    print(f"{lab:34s} {hcs} {r['regret_vs_frozen']:>+8.3f} {100*r['fire_rate']:>5.1f}%")
print("-"*60)
PY
