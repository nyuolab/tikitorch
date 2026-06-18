#!/bin/bash
# _pc_scale_onbox.sh — on-box driver for the tropical predictive-coding scale tests.
# Shipped in the source tarball; invoked by run_pc_scale_gpu.sh's user-data. Runs the
# selected experiment, streams the JSONL results + driver log to S3 every 60s (so
# partial results survive an early shutdown), uploads once more at the end.
#
# Env: S3_RESULTS, REPO (synced tikitorch), PYBIN (venv python), PC_TASK (mnist|cifar),
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

case "$PC_TASK" in
  mnist) SCRIPT=scripts/pc_scale_mnist.py; OUT=results/predictive_coding/scale_mnist.jsonl ;;
  cifar) SCRIPT=scripts/pc_scale_cifar.py; OUT=results/predictive_coding/scale_cifar.jsonl ;;
  *) echo "[pcscale] unknown PC_TASK=$PC_TASK"; exit 2 ;;
esac
LOG=/tmp/pc_scale.log

push(){
  aws s3 cp "$OUT" "$S3_RESULTS/$(basename "$OUT")" --only-show-errors 2>/dev/null || true
  aws s3 cp "$LOG" "$S3_RESULTS/driver.log" --only-show-errors 2>/dev/null || true
}

echo "[pcscale] PC_TASK=$PC_TASK $(date -u +%H:%M:%S)"
"$PYBIN" -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || true

( while true; do sleep 60; push; done ) &
UP=$!

# shellcheck disable=SC2086
"$PYBIN" "$SCRIPT" --matrix --out "$OUT" $PC_ARGS 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

kill "$UP" 2>/dev/null || true
push
echo "[pcscale] rc=$rc $(date -u +%H:%M:%S)"
exit "$rc"
