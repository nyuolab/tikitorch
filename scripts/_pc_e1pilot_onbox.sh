#!/bin/bash
# _pc_e1pilot_onbox.sh — E1 PILOT (E1_SPEC calibration cells before the grid):
# world 0, seed 0, 250k steps, all six arms. Questions: (a) does the birth
# pre-life give frozen_mpc > random (the world model learned SOMETHING)?
# (b) live vs frozen direction; (c) wall clocks for grid sizing.
# Launch: CPU torch, PIP_EXTRA="tokenizers", INSTANCE_TYPE=c7i.4xlarge
# MAX_MINUTES=300. No stream data needed (the world is procedural).
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_world
mkdir -p "$VOUT"
rm -f "$VOUT/e1_world.jsonl"   # start clean (no shipped/smoke rows)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_world" --only-show-errors || true; }
( while true; do sleep 60; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }

ST5="$PY ontogenic_learner/experiments/st5_world.py"
SB="$PY ontogenic_learner/experiments/st5_baselines.py"
JOBS=$(mktemp)
for ARM in live frozen_mpc oracle_mpc random; do
  echo "$ST5 --arm $ARM --steps 80000 --biome-steps 20000 --world-seed 0 --seed 0" >> "$JOBS"
done
for ARM in streaming_ac replay_dqn; do
  echo "$SB --arm $ARM --steps 80000 --biome-steps 20000 --world-seed 0 --seed 0" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 4 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
