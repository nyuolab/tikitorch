#!/bin/bash
# _pc_v4cal_onbox.sh — V4 calibration precursor (CPU box): lock the V4-life
# constants at 800k-token scale BEFORE the 10M spend.
#   (a) margin gate delta in {0.25, 0.5, 1.0} x {stationary, hard_shift}
#       + delta=0.5 on drift — must self-quiet on stationary AND keep the
#       shift benefit (the criterion the z-gate family failed at every z*,
#       V2b) AND sustain firing under drift
#   (b) anchor lambda in {0.01, 0.1} on hard_shift (z*=1.0 gate to decouple;
#       lambda=1.0 strangled plasticity in V2)
#   (c) bit-shift lr decay smoke (halve per 200 fired kicks) on hard_shift
# Launch: CPU torch (default TORCH_INDEX), PIP_EXTRA="tokenizers",
# INSTANCE_TYPE=c7i.4xlarge MAX_MINUTES=300.
# Hardened per the pre-launch adversarial review (2026-07-13): verified
# tokenizer + prebuilt BPE assets shipped; staged inputs hard-gated; tokenizer
# identity check pre-queue; per-cell --tag on the anchor pair (tele filename
# collision); OMP=3 x -P 5 = 15 threads <= 16 vCPU; health telemetry.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$VOUT" data/experience_stream results/pc_convrnn/ckpts
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) load=$(cut -d' ' -f1-3 /proc/loadavg) mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')kB pyjobs=$(pgrep -c -f 'st2_live|st1_streams' || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }

# ---- staged inputs (ALL untracked -> S3 data_in/ckpts_in; gate EVERY one) --
aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/bpe_text8_20m_1024.json" data/bpe_text8_20m_1024.json --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/experience_stream_bpe.pt" data/experience_stream_bpe.pt --only-show-errors
for D in wiki shakespeare docs code; do
  aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors
done
aws s3 cp "$S3_RESULTS/ckpts_in/m1_s1_cascade_s0.pt" results/pc_convrnn/ckpts/ --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/v1_streams.jsonl" "$VOUT/v1_streams.jsonl" --only-show-errors
for F in data/text8_20m.txt data/bpe_text8_20m_1024.json \
         data/experience_stream/wiki.txt data/experience_stream/shakespeare.txt \
         data/experience_stream/docs.txt data/experience_stream/code.txt \
         results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt "$VOUT/v1_streams.jsonl"; do
  [ -s "$F" ] || die "stage $F"
done
[ -s data/experience_stream_bpe.pt ] || \
  $PY ontogenic_learner/experiments/st1_streams.py --build-assets || die "build-assets"
# tokenizer identity vs the ckpt's banked bpb (drift here invalidates windows)
$PY ontogenic_learner/experiments/st1_streams.py --identity-check \
  --ckpt results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt \
  >> "$VOUT/onbox_jobs.log" 2>&1 || die "tokenizer-identity"
sync_s3

ST2="$PY ontogenic_learner/experiments/st2_live.py --arm"
JOBS=$(mktemp)
for DL in 0.25 0.5 1.0; do
  for R in stationary hard_shift; do
    echo "$ST2 gated --gate margin --delta $DL --regime $R --seed 0" >> "$JOBS"
  done
done
echo "$ST2 gated --gate margin --delta 0.5 --regime drift --seed 0" >> "$JOBS"
for LAM in 0.01 0.1; do
  echo "$ST2 gated_anchor --gate z --gate-z 1.0 --anchor-lam $LAM --tag lam$LAM --regime hard_shift --seed 0" >> "$JOBS"
done
echo "$ST2 gated --gate z --gate-z 1.0 --lr-decay-every 200 --tag lrdecay --regime hard_shift --seed 0" >> "$JOBS"
wc -l "$JOBS"
xargs -P 5 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"

sync_s3
echo "ONBOX-DONE"
