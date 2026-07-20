#!/bin/bash
# Pre-cut: the held-LR, fire-always collapse stretch that sank PureTropicalSGD
# (E0-3). Stationary wiki, gate bypassed (every window kicks), NO lr decay, NO
# anchor -> isolates the BASE optimizer's long-horizon stability. 2x2 ablation
# of error-feedback x soft-bound, against the documented collapse baseline.
set -e
cd "$(dirname "$0")/../.."
W=${W:-5000}                       # windows (>~ the ~6k-fire collapse horizon trend)
COMMON="--domains wiki --era-tokens 800000 --cycles 1 --fire-always \
  --lr-decay-every 0 --anchor-lam 0 --max-windows $W --tele-every 25 \
  --ckpt-every 100000000 --audit-frac 0.02 --seed 0"
run () { echo "=== ARM $1 ==="; python3 ontogenic_learner/experiments/st4_life.py --tag "$1" $2 $COMMON; }

run precut_puretropical "--opt puretropical"                       # collapse baseline
run precut_ef_bound     "--opt eftropical --ef-bound 8"            # full design (EF + bound)
run precut_ef_only      "--opt eftropical --ef-bound 0"            # EF alone (no bound)
run precut_bound_only   "--opt eftropical --no-ef --ef-bound 8"   # bound alone (naked sign)
echo "PRECUT DONE"
