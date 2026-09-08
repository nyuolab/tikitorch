#!/bin/bash
# _pc_g5_onbox.sh — G5 strict rung (PREREG.md G5): no-waiver audit gate plus
# closed-optimizer/LUT-loss quality runs, annealed vs hard-from-scratch.
# A failed strict audit stops the expensive quality sweep and the script exits nonzero.
set -u
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
# Keep the displayed and executed commands identical, including quoted arguments.
BASE_CMD=("$PY" scripts/pc_convrnn_charlm.py --arms morphssm:cascade
  --opt puretropical --loss lut --lr-shift 10)
AUDIT_CMD=("${BASE_CMD[@]}" --strict-audit --seed 0 --tag g5_annealed_s0 --out "$OUT")
SEED_CMD=("${BASE_CMD[@]}" --seed 1 --tag g5_annealed_s1 --out "$OUT")
BUDGET_CMD=("${BASE_CMD[@]}" --steps 6000 --eval-interval 500 --seed 0
  --ckpt-name g5_strict_6k.pt --tag g5_annealed_6k --out "$OUT")
HARD_CMD=("${BASE_CMD[@]}" --tau 0 --no-anneal --seed 0 --tag g5_hard_receipt --out "$OUT")
case "${1:-}" in
  --print-commands)
    for command_name in AUDIT_CMD SEED_CMD BUDGET_CMD HARD_CMD; do
      # Bash 3.2 (macOS) has no namerefs; enumerate arrays explicitly below.
      case "$command_name" in
        AUDIT_CMD) printf '%q ' "${AUDIT_CMD[@]}" ;;
        SEED_CMD) printf '%q ' "${SEED_CMD[@]}" ;;
        BUDGET_CMD) printf '%q ' "${BUDGET_CMD[@]}" ;;
        HARD_CMD) printf '%q ' "${HARD_CMD[@]}" ;;
      esac
      printf '\n'
    done
    exit 0 ;;
  "") ;;
  *) echo "usage: $0 [--print-commands]" >&2; exit 2 ;;
esac
cd "${REPO:?}" || exit 2
: "${S3_RESULTS:?S3_RESULTS is required}"
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
stop_sync(){
  [ -z "${SYNC_PID:-}" ] && return
  kill "$SYNC_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
  SYNC_PID=""
}
cleanup(){
  stop_sync
  sync_s3
}
trap cleanup EXIT
health
mkdir -p data
[ -f data/tinyshakespeare.txt ] || \
  aws s3 cp "$S3_RESULTS/data_in/tinyshakespeare.txt" data/tinyshakespeare.txt --only-show-errors || \
  curl -sL --retry 3 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare.txt
# Gate first: under strict-eager-v2 every classified eager operation is a violation.
# If the declared LUT/normalization boundaries remain, fail here instead of launching
# the expensive quality jobs or presenting a historical waiver result as a pass.
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
run_quality(){
  local rc=0
  "$@" >> "$OUT/onbox_jobs.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    printf 'JOB-FAILED rc=%s:' "$rc" >> "$OUT/onbox_jobs.log"
    printf ' %q' "$@" >> "$OUT/onbox_jobs.log"
    printf '\n' >> "$OUT/onbox_jobs.log"
  fi
  return "$rc"
}
# Array execution avoids interpreting Python paths or arguments as shell programs.
run_quality "${SEED_CMD[@]}" & seed_pid=$!
run_quality "${BUDGET_CMD[@]}" & budget_pid=$!
run_quality "${HARD_CMD[@]}" & hard_pid=$!
jobs_rc=0
for job_pid in "$seed_pid" "$budget_pid" "$hard_pid"; do
  wait "$job_pid" || jobs_rc=$?
done
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
