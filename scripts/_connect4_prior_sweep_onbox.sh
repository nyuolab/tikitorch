#!/bin/bash
# _connect4_prior_sweep_onbox.sh — runs the fully-multiply-free Connect4 anti-collapse
# prior sweep ON the EC2 box. This is the on-box half of run_connect4_prior_sweep_ec2.sh;
# it is shipped inside the source tarball and invoked by that runner's user-data.
#
# WHY THIS EXISTS / WHAT IT FIXES (see manuscript/results/board_scaling/CONNECT4_PRIOR_SWEEP.md
# "OPS MISTAKE"): the first sweep ran the recipes concurrently but each `main` used
# torch's DEFAULT thread count (= all cores) per process, so the eval phase
# oversubscribed ~200 threads onto 36 cores and the run took ~13 h instead of ~30 min.
# Here every recipe is pinned to 1 torch thread (`--threads 1`) and a small self-play
# worker count, so 7 recipes share the box without oversubscription. Each recipe's JSON
# is uploaded to S3 the instant it finishes, so a box that dies (or hits the hard
# shutdown backstop) mid-sweep never loses already-completed recipes — the failure that
# lost the apples-to-apples table before.
#
# The script run_fully_tropical_connect4.py runs trained AND untrained per recipe over
# the full sim grid [100,200,400], so a single clean pass of these recipes IS the
# apples-to-apples table (trained-vs-untrained, same run) that the prior partial run
# never retrieved.
#
# Env:
#   S3_RESULTS = s3://bucket/prefix/results   (per-recipe JSONs + logs land here)
#   REPO       = path to the synced tikitorch repo (default: $PWD)
#   WORKERS    = self-play workers per recipe   (default 4; 7*4=28 < 36 vCPU)
set -u
REPO="${REPO:-$PWD}"
WORKERS="${WORKERS:-4}"
S3_RESULTS="${S3_RESULTS:?set S3_RESULTS to the s3:// results prefix}"
cd "$REPO" || exit 2
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
mkdir -p logs manuscript/results/board_scaling

# tag | extra-args  — EXACTLY the rows of CONNECT4_PRIOR_SWEEP.md. The args contain only
# simple flags (no spaces inside a token), so the unquoted expansion below is intentional.
recipes=(
  "perceptron|--policy-loss perceptron"
  "ce|--policy-loss ce"
  "ce_ent0.1|--policy-loss ce --entropy-coef 0.1"
  "ce_ent0.3|--policy-loss ce --entropy-coef 0.3"
  "ce_ent0.1_lr1e3|--policy-loss ce --entropy-coef 0.1 --lr 1e-3"
  "ce_temp2|--policy-loss ce --target-temp 2.0"
  "perceptron_ent0.1|--policy-loss perceptron --entropy-coef 0.1"
)

run_one(){
  local tag="$1" extra="$2"
  local log="logs/sweep_${tag}.log"
  local json="manuscript/results/board_scaling/connect4_fully_tropical_${tag}.json"
  echo "[$tag] START $(date -u +%H:%M:%S)  args: $extra"
  # shellcheck disable=SC2086
  python3 scripts/run_fully_tropical_connect4.py --tag "$tag" --threads 1 --workers "$WORKERS" $extra \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$json" ]; then
    aws s3 cp "$json" "$S3_RESULTS/connect4_fully_tropical_${tag}.json" --only-show-errors \
      && echo "[$tag] DONE+UPLOADED $(date -u +%H:%M:%S)" \
      || echo "[$tag] DONE but UPLOAD FAILED $(date -u +%H:%M:%S)"
  else
    echo "[$tag] FAILED rc=$rc (no json or nonzero exit) $(date -u +%H:%M:%S)"
  fi
  aws s3 cp "$log" "$S3_RESULTS/logs/sweep_${tag}.log" --only-show-errors || true
}

echo "[sweep] launching ${#recipes[@]} recipes, WORKERS=$WORKERS/recipe, threads pinned to 1"
pids=()
for r in "${recipes[@]}"; do
  tag="${r%%|*}"; extra="${r#*|}"
  run_one "$tag" "$extra" &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[sweep] ALL DONE fail=$fail $(date -u +%H:%M:%S)"
exit "$fail"
