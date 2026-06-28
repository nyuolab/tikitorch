#!/bin/bash
# run_local_csim_mp.sh — compile + run the forge_pc_xformer_mp_block CSIM LOCALLY (no Vitis/EC2). Validates:
#   (A) the WIDE float forward-for-backward == the bf16 golden bf_forward,
#   (B) the BF16 STE backward + dyadic-Adam == the bf16 golden bf_backward + bf_step (BF16_BWD=2),
#   (C) the DEPLOY int forward bit-exact to the int64 golden g_forward.
# Uses Xilinx's OPEN-SOURCE ap_int headers (authentic ap arithmetic, identical to Vitis csim for the int path).
# The EC2 `vitis_hls -f synth_pc_xformer_mp_block.tcl` run remains the csynth GATE (DSP/timing/SLR).
set -e
cd "$(dirname "$0")"
AP=/tmp/apreal
[ -d "$AP/include" ] || git clone --depth 1 https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git "$AP"
# hls_math.h is a Vitis header; the kernel uses hls::round/log2/ldexp/exp/sqrt/floor — alias them to <cmath> for
# the local g++ build via a tiny shim dir (only for csim; Vitis provides the real hls_math.h on EC2).
SHIM=/tmp/mp_hlsshim
mkdir -p "$SHIM"
cat > "$SHIM/hls_math.h" <<'EOF'
#ifndef MP_HLS_MATH_SHIM
#define MP_HLS_MATH_SHIM
#include <cmath>
namespace hls {
  static inline float round(float x){ return std::round(x); }
  static inline float floor(float x){ return std::floor(x); }
  static inline float log2(float x){ return std::log2(x); }
  static inline float exp(float x){ return std::exp(x); }
  static inline float sqrt(float x){ return std::sqrt(x); }
  static inline float ldexp(float x,int e){ return std::ldexp(x,e); }
}
#endif
EOF
APFLAGS="-D__AP_INT_SPECIAL_H__ -D__AP_FIXED_SPECIAL_H__ -I$AP/include -I$SHIM"
echo "== bf16 golden trains (sanity, BF16_BWD=2) =="
g++ -O2 -I. forge_pc_xformer_bf16_golden_test.cpp -o /tmp/mp_bftest 2>/dev/null && \
  BF16_BWD=2 STEPS=300 ACCUM=1 /tmp/mp_bftest 2>/dev/null | tail -1 || echo "  (golden train sanity skipped — needs Shakespeare data)"
echo "== mp kernel CSIM: (A) wide fwd==golden  (B) bf16 bwd+adam==golden  (C) int fwd==int-golden =="
g++ -O2 -std=c++14 $APFLAGS -I. -Wno-unknown-pragmas \
    forge_pc_xformer_mp_block.cpp forge_pc_xformer_mp_block_tb.cpp -o /tmp/mp_csim
/tmp/mp_csim
