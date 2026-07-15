#!/bin/bash
# _pc_e2screen_onbox.sh — E2 reward-mechanism screening factorial (E2_SPEC.md).
# 2x2 {gate S|N} x {objective O|R} x {foraging, minatar} x 2 seeds = 16 agent
# cells + per-env random + frozen-R floors. Env-specific window/horizon (MinAtar
# board = 100-cell frames need a bigger window). ONE c7i.8xlarge, MAX_MINUTES=240.
# Launch: AMI default (CPU torch), PIP_EXTRA="tokenizers minatar".
# No stream data needed (foraging is procedural; minatar ships its own).
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_reward
mkdir -p "$VOUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_reward" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')kB pyjobs=$(pgrep -c -f st6_reward || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }
export PYTHONPATH=/opt/tiki
$PY -c "import minatar; from minatar import Environment; Environment('breakout').reset(); print('minatar ok')" || die "minatar-import"
$PY -c "import sys; sys.path[:0]=['/opt/tiki/scripts','/opt/tiki/ontogenic_learner/experiments']; from st6_reward import Agent; from nanogpt_shakespeare import tau_at; print('imports ok')" || die "import-chain"

ST6="$PY ontogenic_learner/experiments/st6_reward.py"
JOBS=$(mktemp)
# agent cells: env-specific window/steps/biome
for S in 0 1; do
  for G in S N; do
    for O in O R; do
      echo "$ST6 --env foraging --mode agent --gate $G --objective $O --window 96  --steps 80000 --biome-steps 20000 --seed $S" >> "$JOBS"
      echo "$ST6 --env minatar  --mode agent --gate $G --objective $O --window 256 --steps 40000 --biome-steps 8000  --seed $S" >> "$JOBS"
    done
  done
done
# floors (seed 0): random + frozen-R
echo "$ST6 --env foraging --mode random --window 96  --steps 80000 --biome-steps 20000 --seed 0" >> "$JOBS"
echo "$ST6 --env minatar  --mode random --window 256 --steps 40000 --biome-steps 8000  --seed 0" >> "$JOBS"
echo "$ST6 --env foraging --mode frozen --objective R --window 96  --steps 80000 --biome-steps 20000 --seed 0" >> "$JOBS"
echo "$ST6 --env minatar  --mode frozen --objective R --window 256 --steps 40000 --biome-steps 8000  --seed 0" >> "$JOBS"
wc -l "$JOBS"
xargs -P 8 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
