#!/bin/bash
# _pc_e0_onbox.sh — E0 (ROADMAP_EMBODIED.md): the paper-adjacent closures.
# ONE c7i.8xlarge box (32 vCPU), MAX_MINUTES=720, PIP_EXTRA="tokenizers":
#   E0-1  excess-surprise gate vs margin gate on the FIXED-order two_cycle
#         regime (the re-entry deadband case) + excess sanity on
#         stationary/hard_shift            [st2_live, 5 cells, OMP 3]
#   E0-2  V4 life seeds 1 and 2 (the multi-seed closure)  [st4_life, OMP 8 x2]
#   E0-3  strict shift-optimizer sign-drift adversary: PureTropicalSGD +
#         bit-shift decay, fire-always stationary 20.8k steps, anchorless
#                                            [st4_life, OMP 4]
# Thread budget: 2x8 (lives) + 4 (adversary) + 5x3 (cells, first wave) = 35
# on 32 vCPU briefly — cells are ~2h, lives ~6h; acceptable transient.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_stream
mkdir -p "$VOUT/ckpts" data/experience_stream results/pc_convrnn/ckpts
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_stream" --exclude "*.tmp" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')kB pyjobs=$(pgrep -c -f 'st2_live|st4_life' || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }

aws s3 cp "$S3_RESULTS/data_in/text8_20m.txt" data/text8_20m.txt --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/bpe_text8_20m_1024.json" data/bpe_text8_20m_1024.json --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/experience_stream_bpe.pt" data/experience_stream_bpe.pt --only-show-errors
for D in wiki shakespeare docs code; do
  aws s3 cp "$S3_RESULTS/data_in/${D}.txt" "data/experience_stream/${D}.txt" --only-show-errors
done
aws s3 cp "$S3_RESULTS/ckpts_in/m1_s1_cascade_s0.pt" results/pc_convrnn/ckpts/ --only-show-errors
aws s3 cp "$S3_RESULTS/data_in/v1_streams.jsonl" "$VOUT/v1_streams.jsonl" --only-show-errors
for F in data/text8_20m.txt data/bpe_text8_20m_1024.json data/experience_stream_bpe.pt \
         data/experience_stream/wiki.txt data/experience_stream/shakespeare.txt \
         data/experience_stream/docs.txt data/experience_stream/code.txt \
         results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt "$VOUT/v1_streams.jsonl"; do
  [ -s "$F" ] || die "stage $F"
done
$PY ontogenic_learner/experiments/st1_streams.py --identity-check \
  --ckpt results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt \
  >> "$VOUT/onbox_jobs.log" 2>&1 || die "tokenizer-identity"
sync_s3

# resume pull-down (no-op on a fresh launch): RESUME_FROM=s3://<old>/results
if [ -n "${RESUME_FROM:-}" ]; then
  aws s3 sync "$RESUME_FROM/st_stream/ckpts" "$VOUT/ckpts" --only-show-errors
  for T in main_s1 main_s2 stricttest; do
    aws s3 cp "$RESUME_FROM/st_stream/v4_tele_${T}.jsonl" "$VOUT/" --only-show-errors || true
    aws s3 cp "$RESUME_FROM/st_stream/v4_probes_${T}.jsonl" "$VOUT/" --only-show-errors || true
  done
  ls -la "$VOUT/ckpts/"
fi

ST2="$PY ontogenic_learner/experiments/st2_live.py --arm gated"
ST4="$PY ontogenic_learner/experiments/st4_life.py"
JOBS=$(mktemp)
# E0-2: the multi-seed lives (V4 config verbatim, seeds 1 and 2)
echo "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $ST4 --tag main_s1 --seed 1 --resume" >> "$JOBS"
echo "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $ST4 --tag main_s2 --seed 2 --resume" >> "$JOBS"
# E0-3: strict shift-optimizer adversary (anchorless, bit-shift decay)
echo "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $ST4 --tag stricttest --domains wiki --era-tokens 2000000 --cycles 1 --fire-always --opt puretropical --anchor-lam 0 --seed 0 --resume" >> "$JOBS"
# E0-1: gate cells on the deadband regime + sanity
echo "OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $ST2 --gate margin --delta 0.25 --regime two_cycle --seed 0" >> "$JOBS"
echo "OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $ST2 --gate excess --delta 0.25 --regime two_cycle --seed 0" >> "$JOBS"
echo "OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $ST2 --gate excess --delta 0.5 --regime two_cycle --seed 0" >> "$JOBS"
echo "OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $ST2 --gate excess --delta 0.25 --regime stationary --seed 0" >> "$JOBS"
echo "OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $ST2 --gate excess --delta 0.25 --regime hard_shift --seed 0" >> "$JOBS"
wc -l "$JOBS"
xargs -P 8 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"

sync_s3
echo "ONBOX-DONE"
