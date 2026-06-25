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
csim_design          ;# bit-exact fwd+bwd+dyadic-Adam vs the int64 golden (forge_pc_xformer_block_tb.cpp)
csynth_design
puts "==== report: $proj/solution1/syn/report/${top}_csynth.rpt ===="
puts "==== CHECK: DSP usage (attention soft-blend only ~128), timing met (<= ${period}ns), LUT < one SLR ===="
exit
