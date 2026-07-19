#!/bin/bash
# _pc_e5b_onbox.sh — E5b: the BabyAI gate screen (E5_SPEC "third world").
# 9 cells: random x3 | frozen x3 | recipe(headobsk) x3 on babyai, 100k steps.
# Bar: recipe > random floor on return OR clean obs-CE learning -> BabyAI
# eras join the life from cycle 6. ONE c7i.4xlarge, MAX_MINUTES=300.
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
$PY -c "import minigrid, gymnasium" || die "minigrid-import"
$PY -c "import sys; sys.path[:0]=['/opt/tiki/scripts','/opt/tiki/ontogenic_learner/experiments']; from st5_envs import make_env; make_env('babyai', biome_steps=100)" || die "babyai-env"

BB="$PY ontogenic_learner/experiments/st6_reward.py --env babyai --window 96 --delta 0.05 --biome-steps 25000 --objective R --policy token --steps 100000"
JOBS=$(mktemp)
for S in 0 1 2; do echo "$BB --mode random --seed $S" >> "$JOBS"; done
for S in 0 1 2; do echo "$BB --mode frozen --seed $S" >> "$JOBS"; done
for S in 0 1 2; do
  echo "$BB --mode agent --gate S --fire-always --intr-beta 0.5 --adv-norm --action-head --obs-k -2 --seed $S" >> "$JOBS"
done
wc -l "$JOBS"
xargs -P 8 -I CMD bash -c 'CMD >> '"$VOUT"'/onbox_jobs.log 2>&1 || echo "JOB-FAILED: CMD" >> '"$VOUT"'/onbox_jobs.log' < "$JOBS"
sync_s3
echo "ONBOX-DONE"
