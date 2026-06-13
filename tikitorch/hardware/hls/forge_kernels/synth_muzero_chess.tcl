# synth_muzero_chess.tcl — Vitis HLS C-synthesis of the dyadic Tropical MuZero FORWARD kernel
# at CHESS scale (OBS=769, latent=64, A=4096; matches tikitorch.planning.muzero.chess_game), to get
# the chess-scale silicon estimate (DSP / bounded latency / LUT-FF-BRAM-URAM) for the comparison's
# axis 4 — upgrading it from the gridworld-only number. Run on the AWS FPGA Developer AMI:
#   vitis_hls -f synth_muzero_chess.tcl
set proj_dir forge_muzero_chess_proj
set top      forge_train_step_muzero
# target_part = the F2 part (xcvu47p); period 4.0 ns = 250 MHz
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_train_step_muzero.cpp -cflags "-I.. -DMZ_OBS_OVERRIDE=769 -DMZ_L_OVERRIDE=64 -DMZ_A_OVERRIDE=4096"

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns
csynth_design

puts "==== chess-scale forward csynth report ===="
puts "  $proj_dir/solution1/syn/report/${top}_csynth.rpt"
exit
