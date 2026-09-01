#!/bin/bash
# _pc_g5_onbox.sh — G5 strict rung (PREREG.md G5): closed optimizer + LUT loss
# on the cascade; annealed vs hard-from-scratch; machine-audited zero-multiply
# training step. v9-lineage hardening.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
mkdir -p "$OUT/ckpts"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
sync_s3(){ aws s3 sync "$OUT" "$S3_RESULTS/pc_convrnn" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) load=$(cut -d' ' -f1-3 /proc/loadavg) pyjobs=$(pgrep -c -f pc_convrnn || true)" >> "$OUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
mkdir -p data
[ -f data/tinyshakespeare.txt ] || \
  aws s3 cp "$S3_RESULTS/data_in/tinyshakespeare.txt" data/tinyshakespeare.txt --only-show-errors || \
  curl -sL --retry 3 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare.txt
JOBS=$(mktemp)
# strict-annealed x2 seeds (audit rides seed 0) at bakeoff budget
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --opt puretropical --loss lut --lr-shift 10 --strict-audit --seed 0 --tag g5_annealed_s0 --out $OUT" >> "$JOBS"
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --opt puretropical --loss lut --lr-shift 10 --seed 1 --tag g5_annealed_s1 --out $OUT" >> "$JOBS"
# budget rung: strict-annealed at 6000 steps
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --opt puretropical --loss lut --lr-shift 10 --steps 6000 --eval-interval 500 --seed 0 --ckpt-name g5_strict_6k.pt --tag g5_annealed_6k --out $OUT" >> "$JOBS"
# hard-from-scratch receipt (expected ~bigram; the anneal is load-bearing)
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --opt puretropical --loss lut --lr-shift 10 --tau 0 --no-anneal --seed 0 --tag g5_hard_receipt --out $OUT" >> "$JOBS"
wc -l "$JOBS"
xargs -P 4 -I CMD bash -c 'CMD >> '"$OUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$OUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
