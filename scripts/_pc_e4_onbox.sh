#!/bin/bash
# _pc_e4_onbox.sh — E4: ignition reliability, four levers in parallel
# (E4_SPEC.md). Arms = E3 polA + ONE lever each:
#   intr (surprise reward b=0.5) | norm (A/sqrt var) | mb (MPC warmup 20k)
#   | ent (2^-4 uniform pull) | ctrl (bare polA; s0-s2 = the E3 sentinel).
# Foraging 5 seeds (primary, ent 3), minatar 3 seeds replication. 32 cells.
# ONE c7i.16xlarge (64 vCPU): OMP 4 x P 14. MAX_MINUTES=540.
# SENTINELS FIRST, then foraging primary, minatar last.
set -x
cd "${REPO:?}"
PY=${PY:-/opt/venv/bin/python}
VOUT=ontogenic_learner/results/st_reward
mkdir -p "$VOUT"
rm -f "$VOUT/e2_screen.jsonl" "$VOUT/e1_world.jsonl"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/opt/tiki
sync_s3(){ aws s3 sync "$VOUT" "$S3_RESULTS/st_reward" --only-show-errors || true; }
health(){ echo "$(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) pyjobs=$(pgrep -c -f st6_reward || true)" >> "$VOUT/health.log"; }
( while true; do sleep 60; health; sync_s3; aws s3 cp /var/log/tiki-sweep.log "$S3_RESULTS/boot.log" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_s3' EXIT
die(){ echo "JOB-FAILED: $1" >> "$VOUT/onbox_jobs.log"; sync_s3; exit 1; }
$PY -c "import minatar; from minatar import Environment; Environment('breakout').reset()" || die "minatar-import"
$PY -c "import sys; sys.path[:0]=['/opt/tiki/scripts','/opt/tiki/ontogenic_learner/experiments']; from st6_reward import Agent" || die "import-chain"

F="$PY ontogenic_learner/experiments/st6_reward.py --env foraging --window 96 --delta 0.10 --biome-steps 20000 --objective R --policy token --fire-always --mode agent --gate S --steps 200000"
M="$PY ontogenic_learner/experiments/st6_reward.py --env minatar --window 256 --delta 0.05 --biome-steps 8000 --objective R --policy token --fire-always --mode agent --gate S --steps 100000"
JOBS=$(mktemp)
# SENTINELS FIRST: ctrl s0-s2 must reproduce E3 polA {0.24, 29.54, 22.02}
for S in 0 1 2; do echo "$F --seed $S" >> "$JOBS"; done
for S in 3 4;   do echo "$F --seed $S" >> "$JOBS"; done
# foraging primary arms, 5 seeds
for S in 0 1 2 3 4; do
  echo "$F --intr-beta 0.5 --seed $S" >> "$JOBS"
  echo "$F --adv-norm --seed $S" >> "$JOBS"
  echo "$F --mb-warmup 20000 --seed $S" >> "$JOBS"
done
# ent (weakest-motivated): 3 seeds
for S in 0 1 2; do echo "$F --ent-k -4 --seed $S" >> "$JOBS"; done
# minatar replication, 3 seeds
for S in 0 1 2; do
  echo "$M --intr-beta 0.5 --seed $S" >> "$JOBS"
  echo "$M --adv-norm --seed $S" >> "$JOBS"
  echo "$M --mb-warmup 10000 --seed $S" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 14 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
