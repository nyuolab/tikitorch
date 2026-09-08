#!/bin/bash
# _chess_strength_onbox.sh — on-box driver for the chess STRENGTH sweep (the "lift chess off the floor"
# experiment). Shipped in the source tarball; invoked by run_connect4_prior_sweep_ec2.sh's user-data when
# launched with DRIVER=_chess_strength_onbox.sh RESULT_GLOB='chess_strength_*.json'.
#
# Chess MCTS is ~100x costlier than Connect4 (measured ~0.64 s/ply, ~84 s/eval-game @64 sims), so eval is
# the dominant cost and is serial per recipe. We therefore run the recipes CONCURRENTLY (each recipe's
# serial eval on its own core, recipes overlapping), threads pinned to 1, with a bounded self-play worker
# count so 4 recipes share the box (4*WORKERS self-play procs). Each recipe uploads its JSON to S3 the
# instant it finishes (a dead box / backstop never loses completed recipes).
#
# The question each recipe answers: with the STRONG (AlphaZero true-rules) tree, does TRAINING the
# multiply-free prior beat the UNTRAINED control (same tree)? Baselines to beat (chess_alphazero.json):
# untrained AlphaZero-tree 0.625 vs random; untrained learned-tree 0.375; the old trained scaffold 0.071.
#
# Env: S3_RESULTS (s3:// prefix), REPO (default $PWD), WORKERS (self-play workers/recipe, default 8).
set -u
REPO="${REPO:-$PWD}"
WORKERS="${WORKERS:-8}"
S3_RESULTS="${S3_RESULTS:?set S3_RESULTS to the s3:// results prefix}"
cd "$REPO" || exit 2
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
mkdir -p logs manuscript/results/board_scaling

# bounded params (sized from measured chess cost; ~2-2.5h wall-clock on a 36-vCPU box, well under backstop)
SIMS=48; ITERS=12; GAMES=16; STEPS=30; EVAL=12; ETRACK=6; EEVERY=6

# tag | extra-args. az_* compare trained-vs-untrained with the STRONG tree; mz_ent1 isolates the tree lever.
recipes=(
  "az_ce|--tree alphazero --policy-loss ce --entropy 0.0 --target-temp 1.0"
  "az_ent1|--tree alphazero --policy-loss ce --entropy 0.1 --target-temp 1.5"
  "az_ent3|--tree alphazero --policy-loss ce --entropy 0.3 --target-temp 2.0"
  "mz_ent1|--tree muzero --policy-loss ce --entropy 0.1 --target-temp 1.5"
)

run_one(){
  local tag="$1" extra="$2"
  local log="logs/chess_${tag}.log"
  local json="manuscript/results/board_scaling/chess_strength_${tag}.json"
  echo "[$tag] START $(date -u +%H:%M:%S)  args: $extra"
  # shellcheck disable=SC2086
  python3 scripts/run_chess_strength.py --tag "$tag" --threads 1 --workers "$WORKERS" \
    --sims $SIMS --iters $ITERS --games $GAMES --steps $STEPS \
    --eval $EVAL --eval-track $ETRACK --eval-every $EEVERY $extra > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$json" ]; then
    aws s3 cp "$json" "$S3_RESULTS/chess_strength_${tag}.json" --only-show-errors \
      && echo "[$tag] DONE+UPLOADED $(date -u +%H:%M:%S)" || echo "[$tag] DONE but UPLOAD FAILED"
  else
    echo "[$tag] FAILED rc=$rc $(date -u +%H:%M:%S)"
  fi
  aws s3 cp "$log" "$S3_RESULTS/logs/chess_${tag}.log" --only-show-errors || true
}

echo "[chess] launching ${#recipes[@]} recipes, WORKERS=$WORKERS/recipe, threads pinned, sims=$SIMS iters=$ITERS games=$GAMES"
pids=()
for r in "${recipes[@]}"; do
  tag="${r%%|*}"; extra="${r#*|}"
  run_one "$tag" "$extra" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[chess] ALL DONE fail=$fail $(date -u +%H:%M:%S)"
exit "$fail"
