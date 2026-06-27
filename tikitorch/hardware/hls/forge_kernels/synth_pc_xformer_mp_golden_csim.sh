#!/usr/bin/env bash
# synth_pc_xformer_mp_golden_csim.sh — build + run the Gate-G2 CASCADE multi-precision sweep.
# Plain int64/double C++ golden (no Vitis / no ap_int needed). Local g++.
#
#   ./synth_pc_xformer_mp_golden_csim.sh            # build + run all 4 modes, STEPS=2500
#   STEPS=4000 ./synth_pc_xformer_mp_golden_csim.sh # longer sweep
#   ./synth_pc_xformer_mp_golden_csim.sh build      # just build
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN=/tmp/mp_xformer_test
STEPS="${STEPS:-2500}"
SHK="${SHAKESPEARE:-$HOME/.cache/tiur/tinyshakespeare.txt}"

echo "== building =="
g++ -O2 -I"$HERE" "$HERE/forge_pc_xformer_mp_golden_test.cpp" -o "$BIN"
echo "built $BIN"
[ "${1:-}" = "build" ] && exit 0

for MODE in int fix16 fix24 dbl; do
  echo ""
  echo "==================== MP_OPT=$MODE  (STEPS=$STEPS) ===================="
  MP_OPT="$MODE" SHAKESPEARE="$SHK" STEPS="$STEPS" "$BIN"
done
