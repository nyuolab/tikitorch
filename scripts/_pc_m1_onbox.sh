#!/bin/bash
# _pc_m1_onbox.sh — M1 substrate-tax scaling ladder + riders (PREREG.md M1).
# GPU box (single GPU, SERIAL queue — parallel jobs contend on one device).
# Requires: Base DLAMI (nvidia driver) + default-index torch (CUDA wheel);
# launch with AMI_PARAM + TORCH_INDEX="" + PIP_EXTRA="tokenizers".
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
mkdir -p "$OUT/ckpts"
export PYTHONPATH=/opt/tiki    # nanogpt_shakespeare has no sys.path guard
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || echo "JOB-FAILED: cuda-probe" >> "$OUT/onbox_jobs.log"
nvidia-smi | head -12 || true
sync_s3(){ aws s3 sync "$OUT" "$S3_RESULTS/pc_convrnn" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1) pyjobs=$(pgrep -c -f 'pc_convrnn|nanogpt' || true)" >> "$OUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
mkdir -p data
[ -f data/tinyshakespeare.txt ] || \
  aws s3 cp "$S3_RESULTS/data_in/tinyshakespeare.txt" data/tinyshakespeare.txt --only-show-errors || \
  curl -sL --retry 3 https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tinyshakespeare.txt
if [ ! -f data/text8_20m.txt ]; then
  aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors || {
    curl -sL --retry 3 http://mattmahoney.net/dc/text8.zip -o /tmp/text8.zip &&     cd /tmp && unzip -o text8.zip && cd "$REPO" &&     head -c 20000000 /tmp/text8 > data/text8_20m.txt; }
fi
ls -l data/text8_20m.txt data/tinyshakespeare.txt

JOBS=$(mktemp)
T8="--tokenizer bpe --vocab-size 1024 --text data/text8_20m.txt --device cuda"
# --- S1 (0.5M): morph arms via harness; dense arms via nanogpt --------------
for S in 0 1 2; do
  echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --adam-lr 1e-2 --pc-lr 1e-2 --steps 8700 --eval-interval 500 $T8 --seed $S --ckpt-name m1_s1_cascade_s$S.pt --tag m1_s1_cascade_s$S --out $OUT" >> "$JOBS"
done
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:bp --adam-lr 1e-2 --steps 8700 --eval-interval 500 $T8 --seed 0 --tag m1_s1_bp_s0 --out $OUT" >> "$JOBS"
for S in 0 1; do
  echo "$PY scripts/nanogpt_shakespeare.py --runs dense-conv --morph-d 256 --morph-layers 7 --steps 8700 --tokenizer bpe --vocab-size 1024 --text data/text8_20m.txt --device cuda --seed $S --tag m1_s1_denseconv_s$S --out $OUT" >> "$JOBS"
  echo "$PY scripts/nanogpt_shakespeare.py --runs dense --d-model 128 --n-layers 3 --d-ff 384 --n-heads 4 --steps 8700 --tokenizer bpe --vocab-size 1024 --text data/text8_20m.txt --device cuda --seed $S --tag m1_s1_dense_s$S --out $OUT" >> "$JOBS"
done
# --- S2 (5M) -----------------------------------------------------------------
for S in 0 1; do
  echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --adam-lr 1e-2 --pc-lr 1e-2 --steps 17400 --eval-interval 1000 --d-model 832 $T8 --seed $S --ckpt-name m1_s2_cascade_s$S.pt --tag m1_s2_cascade_s$S --out $OUT" >> "$JOBS"
done
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:bp --adam-lr 1e-2 --steps 17400 --eval-interval 1000 --d-model 832 $T8 --seed 0 --tag m1_s2_bp_s0 --out $OUT" >> "$JOBS"
for S in 0 1; do
  echo "$PY scripts/nanogpt_shakespeare.py --runs dense-conv --morph-d 832 --morph-layers 7 --steps 17400 --tokenizer bpe --vocab-size 1024 --text data/text8_20m.txt --device cuda --seed $S --tag m1_s2_denseconv_s$S --out $OUT" >> "$JOBS"
  echo "$PY scripts/nanogpt_shakespeare.py --runs dense --d-model 448 --n-layers 6 --d-ff 1344 --n-heads 8 --steps 17400 --tokenizer bpe --vocab-size 1024 --text data/text8_20m.txt --device cuda --seed $S --tag m1_s2_dense_s$S --out $OUT" >> "$JOBS"
done
# --- riders -------------------------------------------------------------------
echo "$PY scripts/pc_convrnn_fim.py --family dense --d-model 128 --n-layers 3 --steps 6000 --device cuda --out $OUT" >> "$JOBS"
wc -l "$JOBS"
xargs -P 1 -I CMD bash -c 'CMD >> '"$OUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$OUT"'/onbox_jobs.log' < "$JOBS"

# --- G6 battery at scale (S2 cascade s0 ckpt) ----------------------------------
JB=$(mktemp)
CKPT="$OUT/ckpts/m1_s2_cascade_s0.pt"
if [ -f "$CKPT" ]; then
  for SPAN in 8 16 32; do
    echo "$PY scripts/pc_convrnn_generative.py --gates g6b --ckpt $CKPT --n-windows 32 --spans $SPAN --relax-steps 400 --coord-sweeps 2 --bo-n 8 --tokenizer bpe --vocab-size 1024 --text data/text8_20m.txt --out $OUT --tag m1_s2_g6_s$SPAN" >> "$JB"
  done
  echo "$PY scripts/pc_convrnn_generative.py --gates g6c --ckpt $CKPT --n-windows 32 --rates 0.1 0.25 --relax-steps 400 --tokenizer bpe --vocab-size 1024 --text data/text8_20m.txt --out $OUT --tag m1_s2_g6_denoise" >> "$JB"
else
  echo "MISSING-CKPT: $CKPT" >> "$OUT/onbox_jobs.log"
fi
xargs -P 1 -I CMD bash -c 'CMD >> '"$OUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$OUT"'/onbox_jobs.log' < "$JB"
sync_s3
echo "ONBOX-DONE"
