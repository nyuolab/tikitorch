#!/bin/bash
# _mzgeneral_gpu_onbox.sh — on-box driver for the dense MuZero-on-GPU chess baseline (axes 2-3 of
# CHESS_MUZERO_COMPARISON.md). Shipped in the source tarball; invoked by run_mzgeneral_chess_gpu.sh's
# user-data. Clones muzero-general, drops in our chess env + the wall-clock-bounded throughput driver,
# logs GPU power via nvidia-smi (same method as run_lm_gpu_baseline.sh), runs a bounded window, and
# uploads throughput + energy to S3.
#
# Env: S3_RESULTS, REPO (synced tikitorch, default $PWD), MZ_MAX_SECONDS (default 900),
#      MZ_NUM_WORKERS (default 12), PYBIN (python in the venv).
set -u
REPO="${REPO:-$PWD}"
S3_RESULTS="${S3_RESULTS:?set S3_RESULTS}"
PYBIN="${PYBIN:-python3}"
MZ_MAX_SECONDS="${MZ_MAX_SECONDS:-900}"
MZ_NUM_WORKERS="${MZ_NUM_WORKERS:-12}"
MZG=/opt/mzg

echo "[mzgpu] clone muzero-general $(date -u +%H:%M:%S)"
rm -rf "$MZG"
for i in 1 2 3; do git clone --depth 1 https://github.com/werner-duvaud/muzero-general.git "$MZG" && break; sleep 10; done
cp "$REPO/scripts/mzgeneral_chessgame.py" "$MZG/games/chessgame.py"
cp "$REPO/scripts/mzgeneral_gpu_driver.py" "$MZG/mz_driver.py"
cd "$MZG"

echo "[mzgpu] torch/cuda check"; "$PYBIN" -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || true

# GPU power logging (>=5 Hz, identical method to run_lm_gpu_baseline.sh) for energy = integral(power*dt)
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv,noheader -lms 200 > "$MZG/power.csv" 2>/dev/null &
SMIPID=$!

echo "[mzgpu] run bounded chess MuZero (${MZ_MAX_SECONDS}s, ${MZ_NUM_WORKERS} workers) $(date -u +%H:%M:%S)"
MZ_MAX_SECONDS="$MZ_MAX_SECONDS" MZ_NUM_WORKERS="$MZ_NUM_WORKERS" \
  "$PYBIN" mz_driver.py > "$MZG/driver.log" 2>&1
rc=$?
kill "$SMIPID" 2>/dev/null || true
echo "[mzgpu] driver rc=$rc $(date -u +%H:%M:%S)"

aws s3 cp "$MZG/mz_summary.json"     "$S3_RESULTS/mz_summary.json"     --only-show-errors || true
aws s3 cp "$MZG/mz_throughput.jsonl" "$S3_RESULTS/mz_throughput.jsonl" --only-show-errors || true
aws s3 cp "$MZG/power.csv"           "$S3_RESULTS/power.csv"           --only-show-errors || true
aws s3 cp "$MZG/driver.log"          "$S3_RESULTS/driver.log"          --only-show-errors || true
echo "[mzgpu] uploaded $(date -u +%H:%M:%S)"
exit "$rc"
