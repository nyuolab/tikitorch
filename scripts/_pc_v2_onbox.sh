#!/bin/bash
# _pc_v2_onbox.sh — north-star V2: does-it-help grid + D2 telemetry.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$OUT/ckpts" "$VOUT" data/experience_stream
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) pyjobs=$(pgrep -c -f st2_live || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors
for D in wiki shakespeare docs code; do
  aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors
done
aws s3 cp "$S3_RESULTS/ckpts_in/m1_s1_cascade_s0.pt" "$OUT/ckpts/" --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/v1_streams.jsonl" "$VOUT/v1_streams.jsonl" --only-show-errors
$PY ontogenic_learner/experiments/st1_streams.py --build-assets >> "$VOUT/onbox_jobs.log" 2>&1 || { echo "JOB-FAILED: assets" >> "$VOUT/onbox_jobs.log"; exit 1; }
JOBS=$(mktemp)
for R in hard_shift drift novel_vocab; do
  for A in frozen gated gated_anchor; do
    for S in 0 1 2; do
      echo "$PY ontogenic_learner/experiments/st2_live.py --arm $A --regime $R --seed $S" >> "$JOBS"
    done
  done
done
for S in 0 1 2; do
  echo "$PY ontogenic_learner/experiments/st2_live.py --arm gated --regime stationary --seed $S" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 8 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
