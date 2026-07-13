#!/bin/bash
# _pc_v4life_onbox.sh — V4 lifetime gate: the 10M-token / 5-cycle organism
# life (main, rotating domain order) + two V4b rider lives on disjoint shards
# (Fisher-merge inputs) + the sign-drift adversary rider (fire-always +
# lr-decay on a stationary stream — the strict@6k collapse conditions).
# CPU box. Launch: default TORCH_INDEX (cpu), PIP_EXTRA="tokenizers",
# INSTANCE_TYPE=c7i.8xlarge MAX_MINUTES=1320 (V4cal measured 0.56-0.63
# s/window for this code path -> main projects 16-18h).
# Constants locked by V4cal: margin delta=0.25, anchor lambda=0.01
# (per-WINDOW EMA clock), lr halve per 2000 fires, serve dropout OFF.
# Thread budget: 10 + 4 + 4 + 4 = 22 <= 32 vCPU.
# RESUME: relaunch with RESUME_FROM=s3://<old-prefix>/results — ckpts +
# tele/probe jsonls are pulled down and every job passes --resume (st4
# validates the saved config and falls through to fresh start if no ckpt).
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$VOUT/ckpts" data/experience_stream results/pc_convrnn/ckpts
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --exclude "*.tmp" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) load=$(cut -d' ' -f1-3 /proc/loadavg) mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')kB pyjobs=$(pgrep -c -f 'st4_life' || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }

# ---- staged inputs (gate EVERY one before the 18h spend) -------------------
aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/bpe_text8_20m_1024.json" data/bpe_text8_20m_1024.json --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/experience_stream_bpe.pt" data/experience_stream_bpe.pt --only-show-errors
for D in wiki shakespeare docs code; do
  aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors
done
aws s3 cp "$S3_RESULTS/ckpts_in/m1_s1_cascade_s0.pt" results/pc_convrnn/ckpts/ --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/v1_streams.jsonl" "$VOUT/v1_streams.jsonl" --only-show-errors
for F in data/text8_20m.txt data/bpe_text8_20m_1024.json \
         data/experience_stream_bpe.pt \
         data/experience_stream/wiki.txt data/experience_stream/shakespeare.txt \
         data/experience_stream/docs.txt data/experience_stream/code.txt \
         results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt "$VOUT/v1_streams.jsonl"; do
  [ -s "$F" ] || die "stage $F"
done
$PY ontogenic_learner/experiments/st1_streams.py --identity-check \
  --ckpt results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt \
  >> "$VOUT/onbox_jobs.log" 2>&1 || die "tokenizer-identity"

# ---- resume pull-down (no-op on a fresh life) ------------------------------
if [ -n "${RESUME_FROM:-}" ]; then
  aws s3 sync "$RESUME_FROM/st_stream/ckpts" "$VOUT/ckpts" --only-show-errors
  for T in main riderA riderB drifttest; do
    aws s3 cp "$RESUME_FROM/st_stream/v4_tele_${T}.jsonl" "$VOUT/" --only-show-errors || true
    aws s3 cp "$RESUME_FROM/st_stream/v4_probes_${T}.jsonl" "$VOUT/" --only-show-errors || true
  done
  ls -la "$VOUT/ckpts/"
fi
sync_s3

ST4="$PY ontogenic_learner/experiments/st4_life.py"
JOBS=$(mktemp)
echo "OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 $ST4 --tag main --seed 0 --resume" >> "$JOBS"
echo "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $ST4 --tag riderA --domains wiki,docs --era-tokens 250000 --seed 0 --resume" >> "$JOBS"
echo "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $ST4 --tag riderB --domains shakespeare,code --era-tokens 250000 --seed 0 --resume" >> "$JOBS"
# sign-drift adversary: stationary stream, gate bypassed, 20833 fired steps
# with lr halving per 2000 -> exercises the strict@6k collapse regime directly
echo "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $ST4 --tag drifttest --domains wiki --era-tokens 2000000 --cycles 1 --fire-always --seed 0 --resume" >> "$JOBS"
wc -l "$JOBS"
xargs -P 4 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"

sync_s3
echo "ONBOX-DONE"
