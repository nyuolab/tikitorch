#!/bin/bash
# _pc_e2v2_onbox.sh — E2 v2: the recalibrated factorial + planning A/B
# (E2_SPEC "E2 v2 design"). ONE c7i.8xlarge, MAX_MINUTES=300,
# PIP_EXTRA="tokenizers minatar". H2 foraging cells are the long pole (~2.5h).
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_reward
mkdir -p "$VOUT"
rm -f "$VOUT/e2_screen.jsonl"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/opt/tiki
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_reward" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) pyjobs=$(pgrep -c -f st6_reward || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }
$PY -c "import minatar; from minatar import Environment; Environment('breakout').reset()" || die "minatar-import"
$PY -c "import sys; sys.path[:0]=['/opt/tiki/scripts','/opt/tiki/ontogenic_learner/experiments']; from st6_reward import Agent" || die "import-chain"

F="$PY ontogenic_learner/experiments/st6_reward.py --env foraging --window 96 --delta 0.10 --biome-steps 20000"
M="$PY ontogenic_learner/experiments/st6_reward.py --env minatar --window 256 --delta 0.05 --biome-steps 8000"
JOBS=$(mktemp)
for S in 0 1; do
  for G in S N; do
    echo "$F --mode agent --gate $G --objective R --horizon 1 --steps 80000 --seed $S" >> "$JOBS"
    echo "$F --mode agent --gate $G --objective R --horizon 2 --steps 20000 --seed $S" >> "$JOBS"
    echo "$M --mode agent --gate $G --objective R --horizon 1 --steps 40000 --seed $S" >> "$JOBS"
  done
  echo "$F --mode agent --gate S --objective O --steps 80000 --seed $S" >> "$JOBS"
  echo "$M --mode agent --gate S --objective O --steps 40000 --seed $S" >> "$JOBS"
done
echo "$F --mode random --steps 80000 --seed 0" >> "$JOBS"
echo "$F --mode frozen --objective R --steps 80000 --seed 0" >> "$JOBS"
echo "$M --mode random --steps 40000 --seed 0" >> "$JOBS"
echo "$M --mode frozen --objective R --steps 40000 --seed 0" >> "$JOBS"
wc -l "$JOBS"
xargs -P 7 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
