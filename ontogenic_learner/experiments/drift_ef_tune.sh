#!/bin/bash
# Diagnose EF's inertia on drift: unthrottle it (bigger step 2^-8, NO lr-decay)
# and bracket the anchor's effect. If it learns -> the throttles were the cause;
# if still inert -> EF fundamentally lacks the adaptive step (needs the frexp tier).
set -e
cd "$(dirname "$0")/../.."
C="--era-tokens 80000 --cycles 2 --delta 0.25 --lr-shift 8 --lr-decay-every 0 \
  --ef-bound 0 --seed 0 --tele-every 50 --ckpt-every 100000000 --audit-frac 0.02"
python3 ontogenic_learner/experiments/st4_life.py --tag drift_ef_free   --opt eftropical --anchor-lam 0    $C
python3 ontogenic_learner/experiments/st4_life.py --tag drift_ef_anchor --opt eftropical --anchor-lam 0.01 $C
echo "TUNE DONE"
python3 - <<'PY'
import json; from pathlib import Path
OUT=Path("ontogenic_learner/results/st_stream")
tags=[("drift_adamw","AdamW ref"),("drift_puretropical","PureTropical (closed base)"),
      ("drift_ef_free","EF unthrottled, no anchor"),("drift_ef_anchor","EF unthrottled + anchor"),
      ("drift_eftropical","EF original (throttled)")]
rows={}
for l in (OUT/"v4_life.jsonl").open():
    r=json.loads(l)
    if r.get("tag") in dict(tags): rows[r["tag"]]=r
print(f"\n{'ARM':32s} {'HC':>7s} {'regret':>8s} {'fire%':>6s}")
for t,lab in tags:
    r=rows.get(t)
    if not r: print(f"{lab:32s}  (no data)"); continue
    hc=r.get('headroom_captured'); hcs=f"{hc:>7.3f}" if hc is not None else "    n/a"
    print(f"{lab:32s} {hcs} {r['regret_vs_frozen']:>+8.3f} {100*r['fire_rate']:>5.1f}%")
PY
