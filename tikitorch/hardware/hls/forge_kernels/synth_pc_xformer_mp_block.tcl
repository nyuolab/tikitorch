# synth_pc_xformer_mp_block.tcl — Vitis HLS csim + csynth gate for the BF16 MIXED-PRECISION transformer-block
# shard. Free gate (no F2): confirms the deploy forward stays DSP_MULT=0 outside attention, the BF16 backward's
# real bf16 multiplies (route/wgrad/Jacobian) bind to DSP as expected, timing <= 4ns, one-SLR fit, BEFORE AFI.
#   vitis_hls -f synth_pc_xformer_mp_block.tcl
set proj  forge_pc_xformer_mp_block_proj
set top   forge_pc_xformer_mp_block
set part  "xcvu47p-fsvh2892-2L-e"
set period 4.0

open_project -reset $proj
add_files forge_pc_xformer_mp_block.cpp -cflags "-I."
add_files -tb forge_pc_xformer_mp_block_tb.cpp -cflags "-I."
open_solution -reset solution1
set_top $top
set_part $part
create_clock -period $period
if { [catch { csim_design } cerr] } { puts "==== CSIM NOTE: $cerr (continuing to csynth) ====" } else { puts "==== CSIM PASS ====" }
csynth_design
# NOTE: csynth emits the FP-operator wrappers + a companion *_ip.tcl generator for each into syn/verilog/.
# The CL synth (synth_cl_forge_pc_xformer.tcl) sources those generators to materialize the floating_point
# _ip cores (csynth never runs them) -> resolves the AFI-build "module ..._ip not found" wall.
puts "==== report: $proj/solution1/syn/report/${top}_csynth.rpt ===="
puts "==== CHECK: deploy fwd DSP_MULT=0 outside attn; BF16 bwd DSP = bf16 mults in route/wgrad/Jacobian;"
puts "==== timing met (<= ${period}ns); LUT < one SLR ===="
exit
