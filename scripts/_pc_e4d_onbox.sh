#!/bin/bash
# _pc_e4d_onbox.sh — E4d: gradient volume + sparse schedules (E4D_SPEC).
# Block A (mechanism, minatar PRIMARY): obsk-2 | obsk-3 | sub25 | head
# Block B (schedules, foraging PRIMARY): fa25 (anneal) | advf (advantage gate)
# All arms = recipe (intr0.5+norm). Sentinel first (bare polA s0 -> 0.245).
# 37 cells, ONE c7i.16xlarge (OMP 4 x P 14), MAX_MINUTES=600.
# Walls: E4/E4b anchors hold (head measured 0.97x base locally).
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

F="$PY ontogenic_learner/experiments/st6_reward.py --env foraging --window 96 --delta 0.10 --biome-steps 20000 --objective R --policy token --mode agent --gate S --steps 200000 --intr-beta 0.5 --adv-norm"
M="$PY ontogenic_learner/experiments/st6_reward.py --env minatar --window 256 --delta 0.05 --biome-steps 8000 --objective R --policy token --mode agent --gate S --steps 100000 --intr-beta 0.5 --adv-norm"
B="$PY ontogenic_learner/experiments/st6_reward.py --env foraging --window 96 --delta 0.10 --biome-steps 20000 --objective R --policy token --mode agent --gate S --steps 200000"
JOBS=$(mktemp)
# SENTINEL first (bare polA, no recipe flags)
echo "$B --fire-always --seed 0" >> "$JOBS"
# Block A: minatar PRIMARY x3 + foraging guard x3, 4 arms
for S in 0 1 2; do
  echo "$M --fire-always --obs-k -2 --seed $S" >> "$JOBS"
  echo "$M --fire-always --obs-k -3 --seed $S" >> "$JOBS"
  echo "$M --fire-always --obs-frac 0.25 --seed $S" >> "$JOBS"
  echo "$M --fire-always --action-head --seed $S" >> "$JOBS"
done
for S in 0 1 2; do
  echo "$F --fire-always --obs-k -2 --seed $S" >> "$JOBS"
  echo "$F --fire-always --obs-k -3 --seed $S" >> "$JOBS"
  echo "$F --fire-always --obs-frac 0.25 --seed $S" >> "$JOBS"
  echo "$F --fire-always --action-head --seed $S" >> "$JOBS"
done
# Block B: foraging PRIMARY x3 + minatar x3, 2 arms (gated modes: NO --fire-always)
for S in 0 1 2; do
  echo "$F --fire-anneal 0.25 --seed $S" >> "$JOBS"
  echo "$F --adv-fire --seed $S" >> "$JOBS"
  echo "$M --fire-anneal 0.25 --seed $S" >> "$JOBS"
  echo "$M --adv-fire --seed $S" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 14 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
