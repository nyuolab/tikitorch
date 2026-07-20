#!/bin/bash
# X1 parity hunt: which CLOSED base optimizer matches AdamW's plasticity on the
# gated drift life? All anchorless (isolate the base optimizer); gate ON.
set -e
cd "$(dirname "$0")/../.."
C="--domains wiki,shakespeare,docs,code --era-tokens 80000 --cycles 2 --delta 0.25 \
  --lr-decay-every 2000 --seed 0 --anchor none --tele-every 50 \
  --ckpt-every 100000000 --audit-frac 0.03"
run () { echo "=== $1 ==="; python3 ontogenic_learner/experiments/st4_life.py --tag "$1" $2 $C; }
run x1_adamw        "--opt adamw"
run x1_puretropical "--opt puretropical"
run x1_ef           "--opt eftropical"
run x1_hybrid       "--opt hybrid"
run x1_hybrid_mom   "--opt hybrid --momentum"
run x1_hybrid_bound "--opt hybrid --ef-bound 8"
echo "X1 DONE"
python3 - <<'PY'
import json
from pathlib import Path
OUT = Path("ontogenic_learner/results/st_stream")
tags = [("x1_adamw", "AdamW (anchorless ref)"), ("x1_puretropical", "PureTropicalSGD"),
        ("x1_ef", "EFTropical (EF only)"), ("x1_hybrid", "hybrid: EF+adapt"),
        ("x1_hybrid_mom", "hybrid: EF+adapt+mom"), ("x1_hybrid_bound", "hybrid: EF+adapt+bound")]
rows = {}
for l in (OUT / "v4_life.jsonl").open():
    r = json.loads(l)
    if r.get("tag") in dict(tags):
        rows[r["tag"]] = r
print(f"\n{'ARM':26s} {'HC':>7s} {'regret':>8s} {'fire%':>6s} {'audit_viol':>10s}")
print("-" * 62)
for t, lab in tags:
    r = rows.get(t)
    if not r:
        print(f"{lab:26s}  (no data)"); continue
    hc = r.get("headroom_captured")
    hcs = f"{hc:>7.3f}" if hc is not None else "    n/a"
    print(f"{lab:26s} {hcs} {r['regret_vs_frozen']:>+8.3f} {100*r['fire_rate']:>5.1f}% "
          f"{r['audit_violations']:>10d}")
print("-" * 62)
print("Target: a CLOSED arm reaching HC ~ AdamW. Winner then re-checked for collapse-immunity.")
PY
