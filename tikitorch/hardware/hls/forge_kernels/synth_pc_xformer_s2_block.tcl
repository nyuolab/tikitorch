# synth_pc_xformer_s2_block.tcl — Vitis HLS csim + CSYNTH gate for the SCALE-2 (30M, D=512) DDR-streaming +
# K-wide-MAC transformer-block kernel. This is the PHASE-1 G0 gate of SCALE_30M_124M_DESIGN.md: read the
# per-loop II table and CONFIRM (before ANY AFI) —
#   * s2_treeK / s2_gemm inner MAC loop  : inner II=1, cyc/output = ceil(I/K)
#   * the m_axi weight-tile burst (bload_f) and adam/gacc RMW streams : II=1
#   * attn-bwd + FFN-down working set inside the 18.25 MB SLR ; pick the max K that closes 150 MHz
#   * forward matmul DSP_MULT=0 (dyad-shift) ; backward route/wgrad/Jacobian bf16 mults bind to DSP
#
# K SWEEP: pass -DS2_K=<16|32|64|128> (default 32). Optionally -DS2_TILE_O / -DS2_TILE_I / -DS2_SEQ_TILE.
#   vitis_hls -f synth_pc_xformer_s2_block.tcl                      ;# K=32 default, D=512
#   S2_K=64  vitis_hls -f synth_pc_xformer_s2_block.tcl            ;# sweep a lane count
# NOTE: no Vitis locally — the parent launches this CSYNTH_ONLY on EC2; csim already passed at D=64 in the TB.
set proj  forge_pc_xformer_s2_block_proj
set top   forge_pc_xformer_s2_block
set part  "xcvu47p-fsvh2892-2L-e"
set period 4.0
if { [info exists ::env(S2_CLK_NS)] && $::env(S2_CLK_NS) ne "" } { set period $::env(S2_CLK_NS) }   ;# e.g. 6.667 = 150MHz
# compile-time knobs from env (K sweep + optional tile caps); default D=512 dims from the config header.
# NOTE: empty env vars are SKIPPED (an exported-but-empty S2_K once produced `-DS2_K=` -> "module not found").
set kdef 32
if { [info exists ::env(S2_K)] && $::env(S2_K) ne "" }       { set kdef $::env(S2_K) }
set flags "-I. -DS2_K=$kdef"
foreach v {S2_TILE_O S2_TILE_I S2_SEQ_TILE S2_D S2_H S2_DH S2_FFN S2_SEQ S2_VOCAB S2_AGGRESSIVE_SCHED S2_DEPIPELINE_REDUCE} {
    if { [info exists ::env($v)] && $::env($v) ne "" } { append flags " -D$v=$::env($v)" }
}
puts "==== s2 csynth flags: $flags ===="

# S2_DEBUG_GRADS: the TB compares the kernel's raw grad DUMPS, so BOTH the kernel and TB need the flag for csim.
# The dumps are a handful of bstore_f at the END of the backward (outside every MAC loop) — they DO NOT change
# any inner-loop II, the s2_gemm cyc/output, or DSP_MULT, and add negligible resource. For the last, cleanest
# resource number you may drop -DS2_DEBUG_GRADS from both add_files below (the II/DSP conclusions are unchanged).
open_project -reset $proj
add_files forge_pc_xformer_s2_block.cpp -cflags "$flags -DS2_DEBUG_GRADS"
add_files -tb forge_pc_xformer_s2_tb.cpp -cflags "$flags -DS2_DEBUG_GRADS"
open_solution -reset solution1
set_top $top
set_part $part
create_clock -period $period
# csim wrapped so a hiccup still lets csynth (the GATE) emit its report
if { [catch { csim_design } cerr] } { puts "==== CSIM NOTE: $cerr (continuing to csynth) ====" } else { puts "==== CSIM PASS ====" }
csynth_design
puts "==== report: $proj/solution1/syn/report/${top}_csynth.rpt ===="
puts "==== G0 CHECK: s2_gemm inner MAC II=1 + cyc/output=ceil(I/K); bload_f burst II=1; adam/gacc RMW II=1;"
puts "==== fwd matmul DSP_MULT=0; bwd route/wgrad/Jacobian bf16 mults on DSP; WS < 18.25 MB SLR; timing <= ${period}ns ===="
exit
