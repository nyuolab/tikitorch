#!/bin/bash
# _pc_morphssm_onbox.sh — on-box driver for the pc-morphssm EC2 batches
# (results/pc_convrnn/PREREG.md "EC2 execution plan"). Runs under the
# run_connect4_prior_sweep_ec2.sh machinery: env REPO=/opt/tiki,
# S3_RESULTS=s3://.../results, python at /opt/venv/bin/python.
#
# v2 (speed fixes measured against the first launch):
#   * B0 checkpoint training replaced by S3 fetch (the laptop uploads the
#     matched-lr cascade ckpts to $S3_RESULTS/ckpts_in/ right after launch);
#     on-box retrain kept only as the fallback if the fetch never lands.
#   * B2 split into per-(lr, seed) single-arm commands — the v1 form looped
#     2 archs x 3 seeds inside ONE process (~10-12 h serial, past the
#     backstop). norecur controls dropped at 10k steps (measured at chance;
#     the ladder question is the recurrent arm's bar).
#   * G4 runs at --eval-interval 300 (evals were ~20% of wall).
#   * NPAR/OMP sized for c7i.16xlarge (64 vCPU): 14 jobs x 4 threads.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
OUT=results/pc_convrnn
mkdir -p "$OUT/ckpts"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=4
NPAR=${NPAR:-10}

# v3: the v2 box died ~4 min into the queue with its trace log never re-uploaded
# (the EBS died with the diagnosis). The heartbeat now ships the live trace log
# AND a health line (mem/disk/load/proc count) every 60s so a recurrence is
# diagnosable from S3 alone.
sync_s3(){ aws s3 sync "$OUT" "$S3_RESULTS/pc_convrnn" --exclude "ckpts/*" \
             --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) up=$(cut -d' ' -f1 /proc/uptime) \
load=$(cut -d' ' -f1-3 /proc/loadavg) memfree=$(awk '/MemAvailable/{print $2}' /proc/meminfo)kB \
disk=$(df -h / | awk 'NR==2{print $4}') pyjobs=$(pgrep -c -f pc_convrnn || true)" \
  >> "$OUT/health.log"; }
( while true; do sleep 60; health; sync_s3; \
    aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; \
  done ) &   # heartbeat: results + trace log + health telemetry
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
health

# ---- data: git archive ships only COMMITTED files, and tinyshakespeare.txt
# is untracked (the v4 box lost 37/43 jobs to this) — fetch from the sweep
# prefix, else the public source --------------------------------------------
mkdir -p data
if [ ! -f data/tinyshakespeare.txt ]; then
  aws s3 cp "$S3_RESULTS/data_in/tinyshakespeare.txt" data/tinyshakespeare.txt \
    --only-show-errors || \
  curl -sL --retry 3 \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    -o data/tinyshakespeare.txt
fi
ls -l data/tinyshakespeare.txt

# ---- B0: fetch matched-lr cascade ckpts (uploaded by the laptop); retrain
# only as fallback ------------------------------------------------------------
for i in 1 2 3 4 5; do
  aws s3 cp "$S3_RESULTS/ckpts_in/morphssm_cascade_s0.pt" "$OUT/ckpts/" --only-show-errors \
    && aws s3 cp "$S3_RESULTS/ckpts_in/convrnn_cascade_s0.pt" "$OUT/ckpts/" --only-show-errors \
    && break
  sleep 60
done
# CAUTION: bare `wait` blocks on ALL background jobs INCLUDING the infinite
# heartbeat subshell — that hang (one line before the queue) is what zombied
# boxes v1-v3. Wait on explicit B0 PIDs only.
B0_PIDS=""
if [ ! -f "$OUT/ckpts/morphssm_cascade_s0.pt" ]; then
  $PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --adam-lr 1e-2 \
    --pc-lr 1e-2 --seed 0 --tag onbox_morphssm_cascade_s0 --out "$OUT" &
  B0_PIDS="$B0_PIDS $!"
fi
if [ ! -f "$OUT/ckpts/convrnn_cascade_s0.pt" ]; then
  $PY scripts/pc_convrnn_charlm.py --arms convrnn:cascade --adam-lr 1e-2 \
    --pc-lr 1e-2 --seed 0 --tag onbox_convrnn_cascade_s0 --out "$OUT" &
  B0_PIDS="$B0_PIDS $!"
fi
[ -n "$B0_PIDS" ] && wait $B0_PIDS
sync_s3

# ---- job queue for B1-B3 (xargs -P over command lines) ---------------------
JOBS=$(mktemp)

# B3: G4 depth ladder (d=128; matched lr both arms; light eval cadence)
for L in 4 8 16 32; do
  for S in 0 1 2; do
    echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:bp --adam-lr 1e-2 --d-model 128 --n-layers $L --seed $S --eval-interval 300 --tag g4_bp_L${L}_s${S} --out $OUT" >> "$JOBS"
    echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:cascade --adam-lr 1e-2 --pc-lr 1e-2 --d-model 128 --n-layers $L --seed $S --eval-interval 300 --tag g4_cascade_L${L}_s${S} --out $OUT" >> "$JOBS"
  done
done
echo "$PY scripts/pc_convrnn_charlm.py --arms morphssm:relax_async --adam-lr 1e-2 --pc-lr 1e-2 --d-model 128 --n-layers 16 --seed 0 --eval-interval 300 --tag g4_async_L16_s0 --out $OUT" >> "$JOBS"

# B1: G6 battery (v6: AFTER the cheap G4 jobs — v5 ran the multi-hour G6
# cells first and starved the ladder; 32 windows / best-of-8 / 2 coord
# sweeps: the O(T^2) l2r baselines dominated wall)
echo "$PY scripts/pc_convrnn_generative.py --gates g6a --settle-steps 1000 --out $OUT --tag onbox" >> "$JOBS"
for CK in morphssm convrnn; do
  CKPT="$OUT/ckpts/${CK}_cascade_s0.pt"
  # per-span commands (span-32 coord descent is the heavy tail — parallelize it)
  for SPAN in 8 16 32; do
    echo "$PY scripts/pc_convrnn_generative.py --gates g6b --ckpt $CKPT --n-windows 32 --spans $SPAN --relax-steps 400 --relax-lr 0.05 --tau-relax 0.5 --coord-sweeps 2 --bo-n 8 --out $OUT --tag ${CK}_main_s${SPAN}" >> "$JOBS"
  done
  echo "$PY scripts/pc_convrnn_generative.py --gates g6b --ckpt $CKPT --n-windows 32 --spans 16 --relax-steps 1600 --relax-lr 0.2 --tau-relax 0.25 --out $OUT --tag ${CK}_hi" >> "$JOBS"
  echo "$PY scripts/pc_convrnn_generative.py --gates g6b --ckpt $CKPT --n-windows 32 --spans 16 --relax-steps 400 --relax-lr 0.05 --relax-init mean --out $OUT --tag ${CK}_meaninit" >> "$JOBS"
  echo "$PY scripts/pc_convrnn_generative.py --gates g6c --ckpt $CKPT --n-windows 32 --rates 0.1 0.25 0.5 --relax-steps 400 --out $OUT --tag ${CK}_denoise" >> "$JOBS"
done

# B2: G2b fallback round 1 — COMPLETED on the v4 box (its synthetic-data jobs
# survived the missing-dataset failure); re-enable for fresh sweeps if needed.
if [ "${RUN_B2:-0}" = "1" ]; then
  for LR in 3e-3 1e-2; do
    for S in 0 1 2; do
      echo "$PY scripts/pc_convrnn_stability.py --part 2b --tasks copy --steps 10000 --lr $LR --archs convrnn --seed-list $S --out $OUT" >> "$JOBS"
    done
  done
fi

wc -l "$JOBS"
# run the queue; log each job's rc; never abort the batch on one failure
xargs -P "$NPAR" -I CMD bash -c 'CMD >> '"$OUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$OUT"'/onbox_jobs.log' < "$JOBS"

sync_s3
aws s3 sync "$OUT/ckpts" "$S3_RESULTS/pc_convrnn/ckpts" --only-show-errors || true
echo "ONBOX-DONE"
