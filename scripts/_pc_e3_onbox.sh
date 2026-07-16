#!/bin/bash
# _pc_e3_onbox.sh — E3: the correct loss (E3_SPEC.md). Policy-as-token-
# prediction + dyadic-advantage REINFORCE through the fused cascade kick.
# Arms per env: polA(always) x3 seeds | polG(gated, E2cal delta) x3 |
# frozenTok | random | streaming-AC x3 at matched steps.
# ONE c7i.8xlarge, MAX_MINUTES=300, PIP_EXTRA="tokenizers minatar".
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_reward
mkdir -p "$VOUT"
rm -f "$VOUT/e2_screen.jsonl" "$VOUT/e1_world.jsonl"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/opt/tiki
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_reward" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) pyjobs=$(pgrep -c -f 'st6_reward|st5_baselines' || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }
$PY -c "import minatar; from minatar import Environment; Environment('breakout').reset()" || die "minatar-import"
$PY -c "import sys; sys.path[:0]=['/opt/tiki/scripts','/opt/tiki/ontogenic_learner/experiments']; from st6_reward import Agent; from st5_baselines import make_feats" || die "import-chain"

ST6F="$PY ontogenic_learner/experiments/st6_reward.py --env foraging --window 96 --delta 0.10 --biome-steps 20000 --objective R --policy token --steps 200000"
ST6M="$PY ontogenic_learner/experiments/st6_reward.py --env minatar --window 256 --delta 0.05 --biome-steps 8000 --objective R --policy token --steps 100000"
SB="$PY ontogenic_learner/experiments/st5_baselines.py --arm streaming_ac"
JOBS=$(mktemp)
# floors first (never lose the denominators)
echo "$ST6F --mode random --seed 0" >> "$JOBS"
echo "$ST6M --mode random --seed 0" >> "$JOBS"
echo "$ST6F --mode frozen --seed 0" >> "$JOBS"
echo "$ST6M --mode frozen --seed 0" >> "$JOBS"
# the question cells
for S in 0 1 2; do
  echo "$ST6F --mode agent --gate S --fire-always --seed $S" >> "$JOBS"
  echo "$ST6M --mode agent --gate S --fire-always --seed $S" >> "$JOBS"
  echo "$ST6F --mode agent --gate S --seed $S" >> "$JOBS"
done
# matched-steps streaming-AC bar
for S in 0 1 2; do
  echo "$SB --envname foraging --steps 200000 --biome-steps 20000 --seed $S" >> "$JOBS"
  echo "$SB --envname minatar --steps 100000 --biome-steps 8000 --seed $S" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 8 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
