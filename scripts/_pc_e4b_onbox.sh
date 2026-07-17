#!/bin/bash
# _pc_e4b_onbox.sh — E4b: combo (intr+norm) + credit-horizon probe (E4B_SPEC).
# 12 cells: ctrl-F sentinel s0 | combo-F x5 | combo-M x3 | normw512-M x3.
# ONE c7i.16xlarge, single wave at P 14. MAX_MINUTES=600: review MEASURED
# w512 T-scaling at 2.9-3.45x (cache-spill superlinear, NOT the 2x FLOPs
# guess) => worst case ~470min; 480 was inside the breach band.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_reward
mkdir -p "$VOUT"
rm -f "$VOUT/e2_screen.jsonl" "$VOUT/e1_world.jsonl"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/opt/tiki
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_reward" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) pyjobs=$(pgrep -c -f st6_reward || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }
$PY -c "import minatar; from minatar import Environment; Environment('breakout').reset()" || die "minatar-import"
$PY -c "import sys; sys.path[:0]=['/opt/tiki/scripts','/opt/tiki/ontogenic_learner/experiments']; from st6_reward import Agent" || die "import-chain"

F="$PY ontogenic_learner/experiments/st6_reward.py --env foraging --window 96 --delta 0.10 --biome-steps 20000 --objective R --policy token --fire-always --mode agent --gate S --steps 200000"
M="$PY ontogenic_learner/experiments/st6_reward.py --env minatar --delta 0.05 --biome-steps 8000 --objective R --policy token --fire-always --mode agent --gate S --steps 100000"
JOBS=$(mktemp)
# longest first (w512), then sentinel, then combos — single wave at P14 anyway
for S in 0 1 2; do echo "$M --window 512 --adv-norm --seed $S" >> "$JOBS"; done
echo "$F --seed 0" >> "$JOBS"                                   # drift sentinel (0.245)
for S in 0 1 2 3 4; do echo "$F --intr-beta 0.5 --adv-norm --seed $S" >> "$JOBS"; done
for S in 0 1 2; do echo "$M --window 256 --intr-beta 0.5 --adv-norm --seed $S" >> "$JOBS"; done
wc -l "$JOBS"
xargs -P 14 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
