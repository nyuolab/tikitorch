# synth_muzero_train_chess.tcl — Vitis HLS C-synthesis of the dyadic Tropical MuZero TRAIN kernel
# (backward + SignSGD) at CHESS scale. With the now-bounded nearest_pow2 loop (was data-dependent ->
# HLS left the train-kernel latency UNRESOLVED '?'), csynth resolves a bounded train-step latency.
#   vitis_hls -f synth_muzero_train_chess.tcl
set proj_dir forge_muzero_train_chess_proj
set top      forge_train_step_muzero_train
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_train_step_muzero_train.cpp -cflags "-I.. -DMZ_OBS_OVERRIDE=769 -DMZ_L_OVERRIDE=64 -DMZ_A_OVERRIDE=4096"

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns
# ZERO-DSP GUARANTEE: this is a multiply-FREE datapath (dyadic shift+add, SignSGD sign-select). The only
# multiplies HLS can find are integer ADDRESS arithmetic (o*IN), which it maps to a DSP48 BY DEFAULT — even
# when IN is a compile-time constant. Force ALL mul ops to LUT fabric so DSP=0 is guaranteed at synthesis,
# independent of HLS's binding heuristics. The source pointer-walks already remove o*IN, so this is a backstop
# (nothing left to map); it is the correct constraint to DECLARE for a zero-DSP design.
config_op mul -impl fabric
csynth_design

puts "==== chess-scale train csynth report ===="
puts "  $proj_dir/solution1/syn/report/${top}_csynth.rpt"
exit
