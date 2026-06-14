#!/bin/bash
# _chess_lift_onbox.sh — the SCALED chess LIFT test (the real test of whether legal-masking + much more
# self-play lifts the multiply-free chess agent above its untrained control). On-box driver for
# run_connect4_prior_sweep_ec2.sh (DRIVER=_chess_lift_onbox.sh RESULT_GLOB='chess_strength_*.json' PIP_EXTRA=chess).
#
# Unlike the first chess sweep (parallel recipes, serial eval), here recipes run SEQUENTIALLY, each using the
# WHOLE box: many self-play workers + PARALLEL eval (chess eval is the bottleneck — run_chess_strength.py now
# fans eval games across --eval-workers). Scaled ~7x self-play (40x32=1280 games) + sims 64 + legal masking.
# Each recipe uploads its JSON the instant it finishes (a dead box / backstop never loses completed recipes).
#
# Recipes isolate the two new levers:
#   lift_az_mask   = AlphaZero tree + anti-collapse + LEGAL MASK (the faithful candidate)
#   lift_az_nomask = same WITHOUT mask (isolates masking's effect at this scale)
#   lift_mz_mask   = learned-tree control + mask (the mz_ent1 that hung before; LAST so the AZ recipes land first)
set -u
REPO="${REPO:-$PWD}"
WORKERS="${WORKERS:-34}"
S3_RESULTS="${S3_RESULTS:?set S3_RESULTS}"
cd "$REPO" || exit 2
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
mkdir -p logs results/board_scaling

# CONSERVATIVE + reliable (prior run was under-sized: serial tracking eval -> recipe 1 exceeded the backstop).
# Now: parallel tracking AND final eval, modest 480-game scale-up (2.5x the 192 baseline), sims 48. The
# per-phase timing prints in run_chess_strength.py reveal the real cost so a later scale-up is sized, not guessed.
ITERS=20; GAMES=24; SIMS=48; STEPS=40; EVAL=20; ETRACK=10; EEVERY=10

# 2 recipes only (the key comparison): isolate masking at a fixed self-play budget. The learned-tree control
# (mz) is dropped here (slow 4096-way tree; run it separately if needed).
recipes=(
  "lift_az_mask|--tree alphazero --policy-loss ce --entropy 0.1 --target-temp 1.5 --legal-mask"
  "lift_az_nomask|--tree alphazero --policy-loss ce --entropy 0.1 --target-temp 1.5"
)

run_one(){
  local tag="$1" extra="$2"
  local log="logs/chess_${tag}.log" json="results/board_scaling/chess_strength_${tag}.json"
  echo "[$tag] START $(date -u +%H:%M:%S)  args: $extra"
  # shellcheck disable=SC2086
  python3 scripts/run_chess_strength.py --tag "$tag" --threads 1 --workers "$WORKERS" --eval-workers "$WORKERS" \
    --sims $SIMS --iters $ITERS --games $GAMES --steps $STEPS \
    --eval $EVAL --eval-track $ETRACK --eval-every $EEVERY $extra > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$json" ]; then
    aws s3 cp "$json" "$S3_RESULTS/chess_strength_${tag}.json" --only-show-errors \
      && echo "[$tag] DONE+UPLOADED $(date -u +%H:%M:%S)" || echo "[$tag] UPLOAD FAILED"
  else
    echo "[$tag] FAILED rc=$rc $(date -u +%H:%M:%S)"
  fi
  aws s3 cp "$log" "$S3_RESULTS/logs/chess_${tag}.log" --only-show-errors || true
}

echo "[chess-lift] ${#recipes[@]} recipes SEQUENTIAL, full-box (WORKERS=$WORKERS self-play + parallel eval), sims=$SIMS iters=$ITERS games=$GAMES eval=$EVAL"
fail=0
for r in "${recipes[@]}"; do
  tag="${r%%|*}"; extra="${r#*|}"
  run_one "$tag" "$extra" || fail=1
done
echo "[chess-lift] ALL DONE fail=$fail $(date -u +%H:%M:%S)"
exit "$fail"
