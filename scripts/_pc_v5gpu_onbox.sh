#!/bin/bash
# _pc_v5gpu_onbox.sh — V5's adapting-GPU baseline (fairness invariant 5):
# A10G BOARD power (nvidia-smi, 1 Hz) while serving the same stream frozen
# and while LoRA-periodic-adapting (the V3 center cell). Launch: AMI_PARAM=
# <base DLAMI> TORCH_INDEX="" PIP_EXTRA="tokenizers" INSTANCE_TYPE=g5.xlarge
# MAX_MINUTES=90.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$VOUT/ckpts" data/experience_stream results/pc_convrnn/ckpts
export PYTHONPATH=/opt/tiki
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --only-show-errors || true; }
( while true; do sleep 60; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }
$PY -c "import torch; assert torch.cuda.is_available()" || die "cuda-probe"

aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/bpe_text8_20m_1024.json" data/bpe_text8_20m_1024.json --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/experience_stream_bpe.pt" data/experience_stream_bpe.pt --only-show-errors
for D in wiki shakespeare docs code; do
  aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors
done
aws s3 cp "$S3_RESULTS/data_in/v1_streams.jsonl" "$VOUT/v1_streams.jsonl" --only-show-errors
aws s3 cp "$S3_RESULTS/ckpts_in/v3_dense_warm.pt" "$VOUT/ckpts/" --only-show-errors
for F in data/experience_stream_bpe.pt "$VOUT/ckpts/v3_dense_warm.pt" "$VOUT/v1_streams.jsonl"; do
  [ -s "$F" ] || die "stage $F"
done

pwr(){ # timestamped 1 Hz board power; loop form dies with its subshell (no orphan -l stream)
  while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv,noheader,nounits)"
    sleep 1
  done >> "$1" 2>&1
}
run_arm(){ # run_arm <name> <cmd...>
  local NAME=$1; shift
  local PWR="$VOUT/v5gpu_power_${NAME}.csv"
  echo "# T_START=$(date +%s)" >> "$PWR"
  pwr "$PWR" & local SPID=$!
  "$@" >> "$VOUT/onbox_jobs.log" 2>&1 || echo "JOB-FAILED: $NAME" >> "$VOUT/onbox_jobs.log"
  kill $SPID 2>/dev/null || true
  echo "# T_END=$(date +%s)" >> "$PWR"
  [ "$(grep -vc '^#' "$PWR")" -gt 30 ] || echo "JOB-FAILED: power-empty-$NAME" >> "$VOUT/onbox_jobs.log"
  sync_s3
}
ST3="$PY ontogenic_learner/experiments/st3_lora.py --device cuda --regime hard_shift --seed 0"
run_arm frozen $ST3 --arm frozen
run_arm lora $ST3 --arm lora --rank 8 --cadence 50000 --inner-k 32 --lr 1e-3
run_arm online_bp $ST3 --arm online_bp --lr 1e-3
sync_s3
echo "ONBOX-DONE"
