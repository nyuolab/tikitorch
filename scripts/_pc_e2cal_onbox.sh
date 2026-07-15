#!/bin/bash
# _pc_e2cal_onbox.sh — E2cal: gate-delta fire-rate recalibration (E2_SPEC v1
# found delta=0.25 fires only 0.1-0.9% on these envs -> mechanisms unevaluable).
# Sweep delta on both envs (S gate, R objective; gate is obs-only so objective
# doesn't affect fire) to find the delta giving ~5-15% fire. ONE c7i.8xlarge
# (32 vCPU: -P 8 x OMP 4 = 32 threads — v1 died oversubscribed 2x on a 4xlarge
# with a 90-min backstop, 0 cells done), MAX_MINUTES=180,
# PIP_EXTRA="tokenizers minatar".
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_reward
mkdir -p "$VOUT"
rm -f "$VOUT/e2_screen.jsonl"   # start clean (v1 rows ship in the tarball)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/opt/tiki
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_reward" --only-show-errors || true; }
( while true; do sleep 60; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }
$PY -c "import minatar; from minatar import Environment; Environment('breakout').reset()" || die "minatar-import"

ST6="$PY ontogenic_learner/experiments/st6_reward.py --mode agent --gate S --objective R --seed 0"
JOBS=$(mktemp)
for D in 0.25 0.15 0.10 0.05 0.025 0.01; do
  echo "$ST6 --env foraging --delta $D --window 96  --steps 20000 --biome-steps 10000" >> "$JOBS"
  echo "$ST6 --env minatar  --delta $D --window 256 --steps 20000 --biome-steps 10000" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 8 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
