#!/bin/bash
# _pc_g5_onbox.sh — G5 strict rung (PREREG.md G5): no-waiver audit gate plus
# closed-optimizer/LUT-loss quality runs, annealed vs hard-from-scratch.
# A failed strict audit stops the expensive quality sweep and the script exits nonzero.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
: "${S3_RESULTS:?S3_RESULTS is required}"
OUT=results/pc_convrnn
mkdir -p "$OUT/ckpts"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
sync_s3(){ aws s3 sync "$OUT" "$S3_RESULTS/pc_convrnn" --only-show-errors || true; }
sync_s3_strict(){
  local attempt
  for attempt in 1 2 3; do
    if aws s3 sync "$OUT" "$S3_RESULTS/pc_convrnn" --only-show-errors; then
      return 0
    fi
    [ "$attempt" -eq 3 ] || sleep "$((attempt * 5))"
  done
  return 1
}
sync_status_strict(){
  local attempt
  for attempt in 1 2 3; do
    if aws s3 cp "$OUT/onbox_status.log" \
      "$S3_RESULTS/pc_convrnn/onbox_status.log" --only-show-errors; then
      return 0
    fi
    [ "$attempt" -eq 3 ] || sleep "$((attempt * 5))"
  done
  return 1
}
record_status(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$OUT/onbox_status.log"; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) load=$(cut -d' ' -f1-3 /proc/loadavg) pyjobs=$(pgrep -c -f pc_convrnn || true)" >> "$OUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
JOBS=""
stop_sync(){
  [ -z "${SYNC_PID:-}" ] && return
  kill "$SYNC_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
  SYNC_PID=""
}
cleanup(){
  stop_sync
  [ -z "$JOBS" ] || rm -f "$JOBS"
  sync_s3
}
trap cleanup EXIT
health
mkdir -p data
[ -f data/tinyshakespeare.txt ] || \
  aws s3 cp "$S3_RESULTS/data_in/tinyshakespeare.txt" data/tinyshakespeare.txt --only-show-errors || \
  curl -sL --retry 3 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare.txt
JOBS=$(mktemp)
# Gate first: under strict-eager-v2 every classified eager operation is a violation.
# If the declared LUT/normalization boundaries remain, fail here instead of launching
# the expensive quality jobs or presenting a historical waiver result as a pass.
AUDIT_CMD=("$PY" scripts/pc_convrnn_charlm.py --arms morphssm:cascade
  --opt puretropical --loss lut --lr-shift 10 --strict-audit --seed 0
  --tag g5_annealed_s0 --out "$OUT")
if "${AUDIT_CMD[@]}" >> "$OUT/onbox_jobs.log" 2>&1; then
  :
else
  audit_rc=$?
  echo "STRICT-AUDIT-FAILED rc=$audit_rc" >> "$OUT/onbox_jobs.log"
  record_status "ONBOX-FAILED strict-audit rc=$audit_rc" || \
    echo "STATUS-WRITE-FAILED strict-audit" >&2
  sync_s3
  echo "ONBOX-FAILED strict-audit rc=$audit_rc"
  exit "$audit_rc"
fi

# Remaining quality runs start only after the no-waiver audit passes.
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --opt puretropical --loss lut --lr-shift 10 --seed 1 --tag g5_annealed_s1 --out $OUT" >> "$JOBS"
# budget rung: strict-annealed at 6000 steps
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --opt puretropical --loss lut --lr-shift 10 --steps 6000 --eval-interval 500 --seed 0 --ckpt-name g5_strict_6k.pt --tag g5_annealed_6k --out $OUT" >> "$JOBS"
# hard-from-scratch receipt (expected ~bigram; the anneal is load-bearing)
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --opt puretropical --loss lut --lr-shift 10 --tau 0 --no-anneal --seed 0 --tag g5_hard_receipt --out $OUT" >> "$JOBS"
wc -l "$JOBS"
xargs -P 3 -I CMD bash -c \
  'CMD >> '"$OUT"'/onbox_jobs.log 2>&1; rc=$?; if [ "$rc" -ne 0 ]; then echo "JOB-FAILED rc=$rc: CMD" >> '"$OUT"'/onbox_jobs.log; fi; exit "$rc"' \
  < "$JOBS"
jobs_rc=$?
if [ "$jobs_rc" -ne 0 ]; then
  record_status "ONBOX-FAILED quality-jobs rc=$jobs_rc" || \
    echo "STATUS-WRITE-FAILED quality-jobs" >&2
  sync_s3
  echo "ONBOX-FAILED quality-jobs rc=$jobs_rc"
  exit "$jobs_rc"
fi
stop_sync
if ! record_status "ONBOX-FINALIZING"; then
  echo "ONBOX-FAILED status-write"
  exit 1
fi
if ! sync_s3_strict; then
  record_status "ONBOX-FAILED artifact-upload" || \
    echo "STATUS-WRITE-FAILED artifact-upload" >&2
  sync_s3
  echo "ONBOX-FAILED artifact-upload"
  exit 1
fi
if ! record_status "ONBOX-DONE"; then
  echo "ONBOX-FAILED status-write"
  exit 1
fi
if ! sync_status_strict; then
  record_status "ONBOX-FAILED status-upload" || \
    echo "STATUS-WRITE-FAILED status-upload" >&2
  sync_s3
  echo "ONBOX-FAILED status-upload"
  exit 1
fi
echo "ONBOX-DONE"
