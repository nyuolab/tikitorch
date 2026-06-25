#!/bin/bash
# run_local_csim.sh — compile + run the forge_pc_xformer_block CSIM LOCALLY (no Vitis/EC2) using the minimal
# _apshim/ap_int.h shim, validating the kernel fwd+bwd+dyadic-Adam BIT-EXACT against the int64 golden. This is
# the fast inner loop; the EC2 Vitis run (synth_pc_xformer_block.tcl) is the authoritative csim + the csynth gate.
set -e
cd "$(dirname "$0")"
echo "== golden trains (sanity) =="
g++ -O2 -I. forge_pc_xformer_golden_test.cpp -o /tmp/gtest && /tmp/gtest | tail -2
echo "== kernel CSIM bit-exact vs golden =="
g++ -O2 -std=c++14 -I_apshim -I. -Wno-unknown-pragmas \
    forge_pc_xformer_block.cpp forge_pc_xformer_block_tb.cpp -o /tmp/csim
/tmp/csim
