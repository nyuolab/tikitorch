#!/bin/bash
# _pc_v1_onbox.sh — north-star V1: stream calibration + oracle bounds
# (ontogenic_learner/NORTHSTAR_VALIDATION.md). v9-lineage hardening.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$OUT/ckpts" "$VOUT" data/experience_stream
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) load=$(cut -d' ' -f1-3 /proc/loadavg) pyjobs=$(pgrep -c -f st1_streams || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
# data: text8 (tokenizer source), experience_stream domains, S1 ckpt — all staged via S3
aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors
for D in wiki shakespeare docs code; do
  aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors
done
aws s3 cp "$S3_RESULTS/ckpts_in/m1_s1_cascade_s0.pt" "$OUT/ckpts/" --only-show-errors
ls -l data/text8_20m.txt data/experience_stream/ "$OUT/ckpts/m1_s1_cascade_s0.pt"

# phase A: assets + identity check (serial; identity FAIL aborts everything)
$PY ontogenic_learner/experiments/st1_streams.py --build-assets >> "$VOUT/onbox_jobs.log" 2>&1 || { echo "JOB-FAILED: assets" >> "$VOUT/onbox_jobs.log"; exit 1; }
$PY ontogenic_learner/experiments/st1_streams.py --identity-check >> "$VOUT/onbox_jobs.log" 2>&1 || { echo "JOB-FAILED: IDENTITY (tokenizer drift — STOP)" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }
sync_s3
# phase B: 4 oracles in parallel (explicit PIDs — never bare wait)
PIDS=""
for D in wiki shakespeare docs code; do
  $PY ontogenic_learner/experiments/st1_streams.py --oracle "$D" >> "$VOUT/onbox_jobs.log" 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
sync_s3
# phase C: headroom verdicts
$PY ontogenic_learner/experiments/st1_streams.py --headroom >> "$VOUT/onbox_jobs.log" 2>&1 || echo "JOB-FAILED: headroom" >> "$VOUT/onbox_jobs.log"
sync_s3
aws s3 sync "$VOUT/ckpts" "$S3_RESULTS/st_stream/ckpts" --only-show-errors || true
echo "ONBOX-DONE"
