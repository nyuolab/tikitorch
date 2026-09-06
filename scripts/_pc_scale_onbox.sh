#!/bin/bash
# _pc_scale_onbox.sh — on-box driver for the tropical predictive-coding scale tests.
# Shipped in the source tarball; invoked by run_pc_scale_gpu.sh's user-data. Runs the
# selected experiment, streams the JSONL results + driver log to S3 every 60s (so
# partial results survive an early shutdown), uploads once more at the end.
#
# Env: S3_RESULTS, REPO (synced tikitorch), PYBIN (venv python),
#      PC_TASK (mnist|cifar|convlm),
#      PC_ARGS (optional extra args).
set -u
REPO="${REPO:-$PWD}"
S3_RESULTS="${S3_RESULTS:?set S3_RESULTS}"
PYBIN="${PYBIN:-python3}"
PC_TASK="${PC_TASK:-mnist}"
PC_ARGS="${PC_ARGS:-}"
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 DATA_ROOT="$REPO/_data"
mkdir -p results/predictive_coding

MATRIX_ARGS=(--matrix)
case "$PC_TASK" in
  mnist) SCRIPT=scripts/pc_scale_mnist.py; OUT=results/predictive_coding/scale_mnist_gamma_damped.jsonl ;;
  cifar) SCRIPT=scripts/pc_scale_cifar.py; OUT=results/predictive_coding/scale_cifar.jsonl ;;
  convlm) SCRIPT=scripts/pc_charlm_conv.py; OUT=results/predictive_coding/charlm_conv_localpc.jsonl; MATRIX_ARGS=() ;;
  *) echo "[pcscale] unknown PC_TASK=$PC_TASK"; exit 2 ;;
esac
LOG=/tmp/pc_scale.log

push_best_effort(){
  aws s3 cp "$OUT" "$S3_RESULTS/$(basename "$OUT")" --only-show-errors 2>/dev/null || true
  aws s3 cp "$LOG" "$S3_RESULTS/driver.log" --only-show-errors 2>/dev/null || true
}
UP=""
ARGS_FILE=""
stop_uploader(){
  if [ -n "$UP" ]; then
    kill "$UP" 2>/dev/null || true
    wait "$UP" 2>/dev/null || true
    UP=""
  fi
  if [ -n "$ARGS_FILE" ]; then
    rm -f "$ARGS_FILE"
    ARGS_FILE=""
  fi
}
upload_with_retry(){
  local source="$1" destination="$2" attempt last_rc=1
  for attempt in 1 2 3; do
    if aws s3 cp "$source" "$destination" --only-show-errors; then
      return 0
    else
      last_rc=$?
    fi
    [ "$attempt" -eq 3 ] || sleep 2
  done
  return "$last_rc"
}
finalize(){
  local workload_rc="$1" final_rc="$1" log_rc=0 upload_rc=0 status_line
  stop_uploader
  status_line="[pcscale] workload_rc=$workload_rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\n' "$status_line"
  if ! printf '%s\n' "$status_line" >> "$LOG"; then
    log_rc=1
    echo "[pcscale] FATAL: could not append final status to $LOG" >&2
  fi
  if [ -f "$OUT" ]; then
    upload_with_retry "$OUT" "$S3_RESULTS/$(basename "$OUT")" || upload_rc=$?
  elif [ "$workload_rc" -eq 0 ]; then
    upload_rc=1
    echo "[pcscale] FATAL: successful workload produced no $OUT" >&2
  fi
  if [ -f "$LOG" ]; then
    upload_with_retry "$LOG" "$S3_RESULTS/driver.log" || upload_rc=$?
  elif [ "$workload_rc" -eq 0 ]; then
    upload_rc=1
    echo "[pcscale] FATAL: successful workload produced no $LOG" >&2
  fi
  if [ "$final_rc" -eq 0 ] && [ "$log_rc" -ne 0 ]; then
    final_rc="$log_rc"
  fi
  if [ "$final_rc" -eq 0 ] && [ "$upload_rc" -ne 0 ]; then
    final_rc="$upload_rc"
  fi
  return "$final_rc"
}
finish_on_exit(){
  local incoming_rc="$?" final_rc
  trap - EXIT
  finalize "$incoming_rc"
  final_rc=$?
  exit "$final_rc"
}
trap finish_on_exit EXIT

EXTRA_ARGS=()
if [ -n "$PC_ARGS" ]; then
  ARGS_FILE="$(mktemp /tmp/pc_scale_args.XXXXXX)" || exit 2
  if ! "$PYBIN" -c 'import os,shlex,sys; args=shlex.split(sys.argv[1]); sys.stdout.buffer.write(b"".join(os.fsencode(arg) + b"\0" for arg in args))' "$PC_ARGS" > "$ARGS_FILE"; then
    rm -f "$ARGS_FILE"
    echo "[pcscale] invalid shell-style quoting or encoding in PC_ARGS" >&2
    exit 2
  fi
  while IFS= read -r -d '' arg; do
    EXTRA_ARGS+=("$arg")
  done < "$ARGS_FILE"
  rm -f "$ARGS_FILE"
  ARGS_FILE=""
fi
COMMAND=("$PYBIN" "$SCRIPT")
if [ "${#MATRIX_ARGS[@]}" -gt 0 ]; then
  COMMAND+=("${MATRIX_ARGS[@]}")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  COMMAND+=("${EXTRA_ARGS[@]}")
fi
COMMAND+=(--out "$OUT")

rm -f "$OUT"      # start fresh; corrected runs never append to a legacy artifact
{
  echo "[pcscale] started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[pcscale] PC_TASK=$PC_TASK"
  echo "[pcscale] PC_ARGS=$PC_ARGS"
  echo "[pcscale] SOURCE_ARCHIVE_SHA256=${SOURCE_ARCHIVE_SHA256:-unknown}"
  echo "[pcscale] SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}"
  echo "[pcscale] AMI=${RUN_AMI:-unknown} INSTANCE_TYPE=${RUN_INSTANCE_TYPE:-unknown} REGION=${RUN_REGION:-unknown}"
  printf '[pcscale] command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  "$PYBIN" -c "import platform,torch,torchvision;print('python',platform.python_version(),'torch',torch.__version__,'torchvision',torchvision.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available(),'gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
} 2>&1 | tee "$LOG"
provenance_rc=("${PIPESTATUS[@]}")
rc="${provenance_rc[0]}"
if [ "$rc" -eq 0 ] && [ "${provenance_rc[1]}" -ne 0 ]; then
  rc="${provenance_rc[1]}"
fi
if [ "$rc" -ne 0 ]; then
  echo "[pcscale] FATAL: runtime provenance preflight failed (rc=$rc)" >&2
  exit "$rc"
fi

( while true; do sleep 60; push_best_effort; done ) &
UP=$!

"${COMMAND[@]}" 2>&1 | tee -a "$LOG"
pipeline_rc=("${PIPESTATUS[@]}")
rc="${pipeline_rc[0]}"
if [ "$rc" -eq 0 ] && [ "${pipeline_rc[1]}" -ne 0 ]; then
  rc="${pipeline_rc[1]}"
fi
exit "$rc"
