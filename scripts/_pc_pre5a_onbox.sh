#!/bin/bash
# _pc_pre5a_onbox.sh — PRE5-A: wmv (world-model-as-critic) + headobsk combo
# (PRE5_SPEC). Both arms ON the E5 head config. 13 cells, single wave P14.
# ONE c7i.16xlarge, MAX_MINUTES=420 (wmv +1 fwd/step ~ +33% over head walls).
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
$PY -c "import minatar" || die "minatar-import"
$PY -c "import sys; sys.path[:0]=['/opt/tiki/scripts','/opt/tiki/ontogenic_learner/experiments']; from st6_reward import Agent" || die "import-chain"

F="$PY ontogenic_learner/experiments/st6_reward.py --env foraging --window 96 --delta 0.10 --biome-steps 20000 --objective R --policy token --mode agent --gate S --steps 200000 --intr-beta 0.5 --adv-norm --fire-always"
M="$PY ontogenic_learner/experiments/st6_reward.py --env minatar --window 256 --delta 0.05 --biome-steps 8000 --objective R --policy token --mode agent --gate S --steps 100000 --intr-beta 0.5 --adv-norm --fire-always"
B="$PY ontogenic_learner/experiments/st6_reward.py --env foraging --window 96 --delta 0.10 --biome-steps 20000 --objective R --policy token --mode agent --gate S --steps 200000 --fire-always"
JOBS=$(mktemp)
echo "$B --seed 0" >> "$JOBS"                                  # sentinel -> 0.245
for S in 0 1 2; do
  echo "$F --action-head --wm-value --seed $S" >> "$JOBS"      # wmv
  echo "$M --action-head --wm-value --seed $S" >> "$JOBS"
  echo "$F --action-head --obs-k -2 --seed $S" >> "$JOBS"      # headobsk
  echo "$M --action-head --obs-k -2 --seed $S" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 14 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
