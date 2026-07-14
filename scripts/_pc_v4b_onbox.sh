#!/bin/bash
# _pc_v4b_onbox.sh — V4b fleet rider: Fisher-merge the two disjoint-shard
# lives (weights-only, importance = own Adam second moments) and probe-eval
# on all four domains. Micro job. Launch: CPU torch, PIP_EXTRA="tokenizers",
# INSTANCE_TYPE=c7i.xlarge MAX_MINUTES=60.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$VOUT/ckpts" data/experience_stream results/pc_convrnn/ckpts
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --exclude "*.tmp" --exclude "ckpts/*" --only-show-errors || true; }
( while true; do sleep 60; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }

aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/bpe_text8_20m_1024.json" data/bpe_text8_20m_1024.json --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/experience_stream_bpe.pt" data/experience_stream_bpe.pt --only-show-errors
for D in wiki shakespeare docs code; do
  aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors
done
aws s3 cp "$S3_RESULTS/ckpts_in/m1_s1_cascade_s0.pt" results/pc_convrnn/ckpts/ --only-show-errors
for T in riderA riderB; do
  aws s3 cp "$S3_RESULTS/ckpts_in/v4_life_${T}.pt" "$VOUT/ckpts/" --only-show-errors
done
for F in data/experience_stream_bpe.pt results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt \
         "$VOUT/ckpts/v4_life_riderA.pt" "$VOUT/ckpts/v4_life_riderB.pt"; do
  [ -s "$F" ] || die "stage $F"
done
$PY ontogenic_learner/experiments/st4b_merge.py >> "$VOUT/onbox_jobs.log" 2>&1 || die "st4b"
sync_s3
echo "ONBOX-DONE"
