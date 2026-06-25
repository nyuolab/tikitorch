#!/bin/bash
# run_local_csim.sh — compile + run the forge_pc_xformer_block CSIM LOCALLY (no Vitis/EC2), validating the kernel
# fwd+bwd+dyadic-Adam BIT-EXACT against the int64 golden. Uses Xilinx's OPEN-SOURCE ap_int headers
# (github.com/Xilinx/HLS_arbitrary_Precision_Types) so the arithmetic is AUTHENTIC ap_int — identical to what
# Vitis csim runs (the std::complex<ap_int> special headers are skipped via their guards; the kernel never uses
# them). This is the fast, authoritative inner loop; the EC2 `vitis_hls -f synth_pc_xformer_block.tcl` run is for
# the csynth GATE (DSP/timing/SLR), which the shim/headers can't produce.
set -e
cd "$(dirname "$0")"
AP=/tmp/apreal
[ -d "$AP/include" ] || git clone --depth 1 https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git "$AP"
APFLAGS="-D__AP_INT_SPECIAL_H__ -D__AP_FIXED_SPECIAL_H__ -I$AP/include"
echo "== golden trains (sanity) =="
g++ -O2 -I. forge_pc_xformer_golden_test.cpp -o /tmp/gtest && /tmp/gtest | tail -2
echo "== kernel CSIM bit-exact vs golden (AUTHENTIC ap_int) =="
g++ -O2 -std=c++14 $APFLAGS -I. -Wno-unknown-pragmas \
    forge_pc_xformer_block.cpp forge_pc_xformer_block_tb.cpp -o /tmp/csim_real
/tmp/csim_real
