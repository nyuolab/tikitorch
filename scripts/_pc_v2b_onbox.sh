#!/bin/bash
# V2b: gate-calibration sweep — can a higher z* self-quiet on stationary while
# keeping the shift benefit? Runs INSIDE the same st2_live harness.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$VOUT" data/experience_stream
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --only-show-errors || true; }
( while true; do sleep 60; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors
for D in wiki shakespeare docs code; do aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors; done
aws s3 cp "$S3_RESULTS/ckpts_in/m1_s1_cascade_s0.pt" results/pc_convrnn/ckpts/ --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/v1_streams.jsonl" "$VOUT/v1_streams.jsonl" --only-show-errors
$PY ontogenic_learner/experiments/st1_streams.py --build-assets || exit 1
JOBS=$(mktemp)
for Z in 0.5 1.0 2.0; do
  for R in stationary hard_shift; do
    echo "$PY ontogenic_learner/experiments/st2_live.py --arm gated --regime $R --seed 0 --gate-z $Z" >> "$JOBS"
  done
done
xargs -P 6 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
