#!/usr/bin/env bash
# Build the LOCAL TikiTorch FPGA bridge (_libtikifpga) — no board, no Xilinx
# toolchain, just a C++17 compiler. It compiles the bundled bridge.cpp + one
# fused-step kernel as plain C++, so `model.fpga()` BINDS and runs the REAL
# kernel math on the host CPU. This is how you verify the multiplier-free /
# zero-DSP TRAINING properties with no FPGA. (For AWS F2 silicon, build the XRT
# variant instead — see tikitorch/fpga/BUILD_NATIVE.md.)
#
# Usage:
#   bash scripts/build_fpga_local.sh                      # 1-layer LM kernel (the one with a runner)
#   eval "$(bash scripts/build_fpga_local.sh --export)"   # build AND set TIKITORCH_FPGA_LIB
#   KERNEL=forge_train_step_morph bash scripts/build_fpga_local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

KERNEL="${KERNEL:-forge_train_step}"
case "$KERNEL" in
  forge_train_step)           DEF=TT_KERNEL_1LAYER    ; SRC=forge_train_step.cpp ;;
  forge_train_step_2layer)    DEF=TT_KERNEL_2LAYER    ; SRC=forge_train_step_2layer.cpp ;;
  forge_train_step_3layer)    DEF=TT_KERNEL_3LAYER    ; SRC=forge_train_step_3layer.cpp ;;
  forge_train_step_morph)     DEF=TT_KERNEL_MORPH     ; SRC=forge_train_step_morph.cpp ;;
  forge_train_step_morph_ecg) DEF=TT_KERNEL_MORPH_ECG ; SRC=forge_train_step_morph_ecg.cpp ;;
  *) echo "unknown KERNEL='$KERNEL' (see tikitorch/fpga/BUILD_NATIVE.md)" >&2; exit 2 ;;
esac

HLS="$(python3 -c 'import tikitorch.hardware as h; print(h.hls_dir())')/forge_kernels"
CXX="${CXX:-$(command -v clang++ || command -v g++ || true)}"
[ -n "$CXX" ] || { echo "no C++ compiler found (need clang++ or g++)" >&2; exit 1; }
EXT=so; [ "$(uname)" = "Darwin" ] && EXT=dylib
OUT="${TIKITORCH_FPGA_LIB:-$ROOT/_libtikifpga.$EXT}"

echo "building '$KERNEL' -> $OUT  (CXX=$CXX)" >&2
"$CXX" -O2 -std=c++17 -fPIC -shared -DCSIM_ONLY -D"$DEF" -I"$HLS" \
    "$ROOT/tikitorch/fpga/bridge.cpp" "$HLS/$SRC" -o "$OUT"

if [ "${1:-}" = "--export" ]; then
  echo "export TIKITORCH_FPGA_LIB=$OUT"            # stdout: eval-able
else
  echo "done. enable it with:  export TIKITORCH_FPGA_LIB=$OUT" >&2
fi
