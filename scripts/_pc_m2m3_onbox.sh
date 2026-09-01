#!/bin/bash
# _pc_m2m3_onbox.sh — on-box driver for the M2 (recall hybrid) + M3 (quality
# crossover) round, PREREG.md "M2/M3 execution round". Inherits every lesson
# from the v1-v6 saga: continuous log/health shipping, explicit-PID waits,
# dataset fetch, per-command sizing vs the backstop, chunked forwards.
#
# Phase A (parallel): M3 checkpoint training (q2 d256/6k, q3 d384/6k; 8 threads
#   each) + the full M2 battery (15 small jobs).
# Phase B (after A): G6 battery on the q2/q3 checkpoints (q1's battery = the
#   v6 rows; not re-run).
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
mkdir -p "$OUT/ckpts"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=4
NPAR=${NPAR:-10}

sync_s3(){ aws s3 sync "$OUT" "$S3_RESULTS/pc_convrnn" --only-show-errors || true; }
# v8: ckpts INCLUDED in the heartbeat — v7 lost its finished Q2 checkpoint
# because ckpts only synced after a phase that the backstop never let finish.
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) \
load=$(cut -d' ' -f1-3 /proc/loadavg) memfree=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB \
disk=$(df -h / | awk 'NR==2{print $4}') pyjobs=$(pgrep -c -f pc_convrnn || true)" \
  >> "$OUT/health.log"; }
( while true; do sleep 60; health; sync_s3; \
    aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; \
  done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health

mkdir -p data
if [ ! -f data/tinyshakespeare.txt ]; then
  aws s3 cp "$S3_RESULTS/data_in/tinyshakespeare.txt" data/tinyshakespeare.txt \
    --only-show-errors || \
  curl -sL --retry 3 \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    -o data/tinyshakespeare.txt
fi
ls -l data/tinyshakespeare.txt

# ---- Q selector (v9, two-box parallel M3): the laptop uploads config/Q =
# "q2" | "q3" | "both" right after launch; default both --------------------
QSEL="both"
for i in 1 2 3 4 5; do
  if QQ=$(aws s3 cp "$S3_RESULTS/config/Q" - 2>/dev/null) && [ -n "$QQ" ]; then
    QSEL="$QQ"; break
  fi
  sleep 30
done
echo "QSEL=$QSEL"

# ---- Phase A (v8): checkpoints SOLO at high threads — v7 ran them alongside
# 10 M2 jobs (56 threads on 32 cores) and they went 4x slower than sized; M2
# itself completed on v7 and is gated off (RUN_M2=1 re-enables).
JOBS_A=$(mktemp)
if [ "$QSEL" = "q2" ] || [ "$QSEL" = "both" ]; then
  echo "OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --adam-lr 1e-2 --pc-lr 1e-2 --steps 6000 --eval-interval 500 --seed 0 --ckpt-name m3_q2_d256_6k.pt --tag m3_q2 --out $OUT" >> "$JOBS_A"
fi
if [ "$QSEL" = "q3" ] || [ "$QSEL" = "both" ]; then
  echo "OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --adam-lr 1e-2 --pc-lr 1e-2 --steps 6000 --eval-interval 500 --d-model 384 --seed 0 --ckpt-name m3_q3_d384_6k.pt --tag m3_q3 --out $OUT" >> "$JOBS_A"
fi
if [ "${RUN_M2:-0}" = "1" ]; then
  for TASK in mqar induction; do
    for ARCH in convrnn convrnn-hybrid; do
      for S in 0 1 2; do
        echo "$PY scripts/pc_convrnn_stability.py --part 2b --tasks $TASK --steps 4000 --archs $ARCH --seed-list $S --out $OUT" >> "$JOBS_A"
      done
    done
  done
  for S in 0 1 2; do
    echo "$PY scripts/pc_convrnn_stability.py --part 2b --tasks mqar --steps 4000 --archs convrnn-norecur --seed-list $S --out $OUT" >> "$JOBS_A"
  done
fi
wc -l "$JOBS_A"
xargs -P 2 -I CMD bash -c 'CMD >> '"$OUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$OUT"'/onbox_jobs.log' < "$JOBS_A"
sync_s3

# ---- Phase B: G6 battery on the new quality points --------------------------
JOBS_B=$(mktemp)
QLIST="m3_q2_d256_6k m3_q3_d384_6k"
[ "$QSEL" = "q2" ] && QLIST="m3_q2_d256_6k"
[ "$QSEL" = "q3" ] && QLIST="m3_q3_d384_6k"
for Q in $QLIST; do
  CKPT="$OUT/ckpts/${Q}.pt"
  if [ -f "$CKPT" ]; then
    for SPAN in 8 16 32; do
      echo "$PY scripts/pc_convrnn_generative.py --gates g6b --ckpt $CKPT --n-windows 32 --spans $SPAN --relax-steps 400 --relax-lr 0.05 --tau-relax 0.5 --coord-sweeps 2 --bo-n 8 --out $OUT --tag ${Q}_s${SPAN}" >> "$JOBS_B"
    done
    echo "$PY scripts/pc_convrnn_generative.py --gates g6c --ckpt $CKPT --n-windows 32 --rates 0.1 0.25 --relax-steps 400 --out $OUT --tag ${Q}_denoise" >> "$JOBS_B"
  else
    echo "MISSING-CKPT: $CKPT" >> "$OUT/onbox_jobs.log"
  fi
done
wc -l "$JOBS_B"
xargs -P "$NPAR" -I CMD bash -c 'CMD >> '"$OUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$OUT"'/onbox_jobs.log' < "$JOBS_B"

sync_s3
aws s3 sync "$OUT/ckpts" "$S3_RESULTS/pc_convrnn/ckpts" --only-show-errors || true
echo "ONBOX-DONE"
