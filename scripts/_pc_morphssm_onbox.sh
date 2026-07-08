#!/bin/bash
# _pc_morphssm_onbox.sh — on-box driver for the pc-morphssm EC2 batches
# (results/pc_convrnn/PREREG.md "EC2 execution plan"). Runs under the
# run_connect4_prior_sweep_ec2.sh machinery: env REPO=/opt/tiki,
# S3_RESULTS=s3://.../results, python at /opt/venv/bin/python.
#
# Batches (B4/G5 deferred: the harness's closed-optimizer path is not built yet):
#   B0  cascade-trained checkpoints for G6 (matched-lr, seeds deterministic)
#   B1  G6 generative battery: g6a control; g6b infilling sweep; g6c denoising
#   B2  G2b fallback ladder round 1: selective copy 10k steps x lr {3e-3,1e-2}
#   B3  G4 depth ladder: d=128, L in {4,8,16,32} x {bp, cascade} x 3 seeds
#       + relax_async n=1 at L=16
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=4
NPAR=${NPAR:-8}

sync_s3(){ aws s3 sync "$OUT" "$S3_RESULTS/pc_convrnn" --exclude "ckpts/*" \
             --only-show-errors || true; }
( while true; do sleep 120; sync_s3; done ) &   # heartbeat sync
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT

# ---- B0: G6 checkpoints (must finish before B1 g6b/g6c) --------------------
$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --adam-lr 1e-2 \
  --pc-lr 1e-2 --seed 0 --tag onbox_morphssm_cascade_s0 --out "$OUT" &
$PY scripts/pc_convrnn_charlm.py --arms convrnn:cascade --adam-lr 1e-2 \
  --pc-lr 1e-2 --seed 0 --tag onbox_convrnn_cascade_s0 --out "$OUT" &
wait
sync_s3

# ---- job queue for B1-B3 (xargs -P over command lines) ---------------------
JOBS=$(mktemp)

# B1: G6 battery
echo "$PY scripts/pc_convrnn_generative.py --gates g6a --settle-steps 1000 --out $OUT --tag onbox" >> "$JOBS"
for CK in morphssm convrnn; do
  CKPT="$OUT/ckpts/${CK}_cascade_s0.pt"
  # primary infilling cells + ablations (steps/lr/tau/init)
  echo "$PY scripts/pc_convrnn_generative.py --gates g6b --ckpt $CKPT --n-windows 64 --spans 8 16 32 --relax-steps 400 --relax-lr 0.05 --tau-relax 0.5 --coord-sweeps 3 --out $OUT --tag ${CK}_main" >> "$JOBS"
  echo "$PY scripts/pc_convrnn_generative.py --gates g6b --ckpt $CKPT --n-windows 64 --spans 16 --relax-steps 1600 --relax-lr 0.2 --tau-relax 0.25 --out $OUT --tag ${CK}_hi" >> "$JOBS"
  echo "$PY scripts/pc_convrnn_generative.py --gates g6b --ckpt $CKPT --n-windows 64 --spans 16 --relax-steps 400 --relax-lr 0.05 --relax-init mean --out $OUT --tag ${CK}_meaninit" >> "$JOBS"
  echo "$PY scripts/pc_convrnn_generative.py --gates g6c --ckpt $CKPT --n-windows 64 --rates 0.1 0.25 0.5 --relax-steps 400 --out $OUT --tag ${CK}_denoise" >> "$JOBS"
done

# B2: G2b fallback round 1 (budget/lr) — selective copy at 10k steps
for LR in 3e-3 1e-2; do
  echo "$PY scripts/pc_convrnn_stability.py --part 2b --tasks copy --steps 10000 --lr $LR --seeds 3 --out $OUT" >> "$JOBS"
done

# B3: G4 depth ladder (d=128; matched lr both arms)
for L in 4 8 16 32; do
  for S in 0 1 2; do
    echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:bp --adam-lr 1e-2 --d-model 128 --n-layers $L --seed $S --tag g4_bp_L${L}_s${S} --out $OUT" >> "$JOBS"
    echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --adam-lr 1e-2 --pc-lr 1e-2 --d-model 128 --n-layers $L --seed $S --tag g4_cascade_L${L}_s${S} --out $OUT" >> "$JOBS"
  done
done
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:relax_async --adam-lr 1e-2 --pc-lr 1e-2 --d-model 128 --n-layers 16 --seed 0 --tag g4_async_L16_s0 --out $OUT" >> "$JOBS"

wc -l "$JOBS"
# run the queue; log each job's rc; never abort the batch on one failure
xargs -P "$NPAR" -I CMD bash -c 'CMD >> '"$OUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$OUT"'/onbox_jobs.log' < "$JOBS"

sync_s3
aws s3 sync "$OUT/ckpts" "$S3_RESULTS/pc_convrnn/ckpts" --only-show-errors || true
echo "ONBOX-DONE"
