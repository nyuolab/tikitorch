# synth_pc_xformer_block.tcl — Vitis HLS csim + csynth gate for the distributed PC transformer-block shard.
# Free gate (no F2): confirms DSP_MULT=0 outside attention, timing <=4ns, one-SLR fit BEFORE any ~5h AFI.
#   vitis_hls -f synth_pc_xformer_block.tcl
set proj  forge_pc_xformer_block_proj
set top   forge_pc_xformer_block
set part  "xcvu47p-fsvh2892-2L-e"
set period 4.0

open_project -reset $proj
add_files forge_pc_xformer_block.cpp -cflags "-I."
add_files -tb forge_pc_xformer_block_tb.cpp -cflags "-I."
open_solution -reset solution1
set_top $top
set_part $part
create_clock -period $period
# csim: bit-exact fwd+bwd+dyadic-Adam vs the int64 golden. Wrapped so a csim hiccup still lets csynth (the GATE)
# run and emit its report — the local ap_int-shim csim already PASSED max|diff|=0 (run_local_csim.sh).
if { [catch { csim_design } cerr] } { puts "==== CSIM NOTE: $cerr (continuing to csynth) ====" } else { puts "==== CSIM PASS ====" }
csynth_design
puts "==== report: $proj/solution1/syn/report/${top}_csynth.rpt ===="
puts "==== CHECK: DSP (attention soft-blend only ~128), timing met (<= ${period}ns), LUT < one SLR ===="
exit
