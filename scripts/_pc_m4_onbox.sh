#!/bin/bash
# _pc_m4_onbox.sh — M4 FIM baseline (single job; PREREG.md M4). v9-lineage
# hardening: continuous log/health/ckpt shipping, data fetch, sized solo.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
mkdir -p "$OUT/ckpts"
sync_s3(){ aws s3 sync "$OUT" "$S3_RESULTS/pc_convrnn" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) \
load=$(cut -d' ' -f1-3 /proc/loadavg) memfree=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB \
pyjobs=$(pgrep -c -f pc_convrnn || true)" >> "$OUT/health.log"; }
( while true; do sleep 60; health; sync_s3; \
    aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; \
  done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
mkdir -p data
[ -f data/tinyshakespeare.txt ] || \
  aws s3 cp "$S3_RESULTS/data_in/tinyshakespeare.txt" data/tinyshakespeare.txt --only-show-errors || \
  curl -sL --retry 3 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare.txt
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $PY scripts/pc_convrnn_fim.py \
  --steps 6000 --credit cascade --eval-interval 500 --out $OUT \
  >> "$OUT/onbox_jobs.log" 2>&1 || echo "JOB-FAILED: m4_fim" >> "$OUT/onbox_jobs.log"
sync_s3
echo "ONBOX-DONE"
