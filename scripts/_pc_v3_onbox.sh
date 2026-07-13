#!/bin/bash
# _pc_v3_onbox.sh — V3 honest incumbents: iso-quality dense warm start, then
# the LoRA-periodic / full-FT / online-BP sweep on the V1 streams (GPU box,
# SERIAL queue — one device). Launch: AMI_PARAM=<base DLAMI> TORCH_INDEX=""
# PIP_EXTRA="tokenizers" INSTANCE_TYPE=g5.2xlarge MAX_MINUTES=420.
# Hardened per the pre-launch adversarial review (2026-07-13): fatal CUDA
# probe; verified tokenizer + prebuilt BPE assets shipped (never rebuilt);
# every staged input hard-gated BEFORE expensive work; warm start runs
# pre-queue and gates the sweep on its two-sided iso_quality flag; --device
# cuda pinned on every job (no silent CPU fallback).
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$VOUT/ckpts" data/experience_stream results/pc_convrnn/ckpts
export PYTHONPATH=/opt/tiki
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) load=$(cut -d' ' -f1-3 /proc/loadavg) mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')kB gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1) pyjobs=$(pgrep -c -f 'st3_lora|st1_streams' || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }

$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || die "cuda-probe"
nvidia-smi | head -12 || true

# ---- staged inputs (ALL untracked -> S3 data_in/; gate EVERY one) ----------
aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors || {
  curl -sL --retry 3 http://mattmahoney.net/dc/text8.zip -o /tmp/text8.zip && \
  $PY -c "import zipfile; zipfile.ZipFile('/tmp/text8.zip').extractall('/tmp')" && \
  head -c 20000000 /tmp/text8 > data/text8_20m.txt; }
aws s3 cp "$S3_RESULTS/data_in/bpe_text8_20m_1024.json" data/bpe_text8_20m_1024.json --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/experience_stream_bpe.pt" data/experience_stream_bpe.pt --only-show-errors
for D in wiki shakespeare docs code; do
  aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors
done
aws s3 cp "$S3_RESULTS/data_in/v1_streams.jsonl" "$VOUT/v1_streams.jsonl" --only-show-errors
for F in data/text8_20m.txt data/bpe_text8_20m_1024.json \
         data/experience_stream/wiki.txt data/experience_stream/shakespeare.txt \
         data/experience_stream/docs.txt data/experience_stream/code.txt \
         "$VOUT/v1_streams.jsonl"; do
  [ -s "$F" ] || die "stage $F"
done
# prebuilt BPE assets (V1-verified) preferred; rebuild only as fallback, from
# the SHIPPED tokenizer cache (BPEData loads it verbatim — no drift)
[ -s data/experience_stream_bpe.pt ] || \
  $PY ontogenic_learner/experiments/st1_streams.py --build-assets || die "build-assets"

# ---- iso-quality warm start (pre-queue: everything depends on it) ----------
$PY ontogenic_learner/experiments/st3_lora.py --warm-start --device cuda \
  >> "$VOUT/onbox_jobs.log" 2>&1 || die "warm-start"
[ -s "$VOUT/ckpts/v3_dense_warm.pt" ] || die "warm-ckpt-missing"
$PY - <<'PYEOF' || die "warm-not-iso"
import json, pathlib
rows = [json.loads(l) for l in pathlib.Path(
    "ontogenic_learner/results/st_stream/v3_lora.jsonl").open()]
w = [r for r in rows if r.get("gate") == "V3_warm"][-1]
print("warm:", w)
raise SystemExit(0 if w["iso_quality"] else 1)
PYEOF
sync_s3

ST3="$PY ontogenic_learner/experiments/st3_lora.py --device cuda"
JOBS=$(mktemp)
# 1. frozen references (cheap, decisive denominators)
for R in hard_shift drift novel_vocab stationary; do
  echo "$ST3 --arm frozen --regime $R --seed 0" >> "$JOBS"
done
# 2. core LoRA grid on hard_shift: rank x cadence (K=32, lr=1e-3)
for RK in 4 8 16 32; do
  for C in 10000 50000 200000 trig; do
    echo "$ST3 --arm lora --rank $RK --cadence $C --inner-k 32 --lr 1e-3 --regime hard_shift --seed 0" >> "$JOBS"
  done
done
# 3. K/lr sensitivity at the center cell
echo "$ST3 --arm lora --rank 8 --cadence 50000 --inner-k 8 --lr 1e-3 --regime hard_shift --seed 0" >> "$JOBS"
echo "$ST3 --arm lora --rank 8 --cadence 50000 --inner-k 32 --lr 3e-4 --regime hard_shift --seed 0" >> "$JOBS"
echo "$ST3 --arm lora --rank 8 --cadence 50000 --inner-k 32 --lr 3e-3 --regime hard_shift --seed 0" >> "$JOBS"
# 4. full-FT-periodic + per-window online BP (the memory-class anchors)
echo "$ST3 --arm full_ft --cadence 50000 --inner-k 32 --lr 3e-4 --regime hard_shift --seed 0" >> "$JOBS"
echo "$ST3 --arm full_ft --cadence 200000 --inner-k 32 --lr 3e-4 --regime hard_shift --seed 0" >> "$JOBS"
echo "$ST3 --arm online_bp --lr 1e-3 --regime hard_shift --seed 0" >> "$JOBS"
# 5. cross-regime coverage at the center cell + stationary cadence-tax control
for R in drift novel_vocab stationary; do
  echo "$ST3 --arm lora --rank 8 --cadence 50000 --inner-k 32 --lr 1e-3 --regime $R --seed 0" >> "$JOBS"
done
for R in drift novel_vocab stationary; do
  echo "$ST3 --arm online_bp --lr 1e-3 --regime $R --seed 0" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 1 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"

sync_s3
echo "ONBOX-DONE"
