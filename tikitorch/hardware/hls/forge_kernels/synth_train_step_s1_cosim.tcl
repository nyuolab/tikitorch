# CO-SIMULATION script for forge_train_step_s1 — the CHEAP RTL-no-op catcher.
# ----------------------------------------------------------------------------
# Mirrors synth_train_step_s1.tcl but compiles the testbench with -DS1_COSIM_MIN so main() runs ONLY the
# minimal mode-2 INIT repro (stage W_fp masters -> do_opt(S1_OPT_INIT) -> assert the LIVE code/value regions
# became nonzero) and returns 0/1. cosim_design then RE-RUNS that same TB against the RTL: if the optimizer's
# writeback is dropped in RTL (the do_opt silicon no-op class — csim+csynth pass but the live regions are
# left ZERO on real hardware), cosim FAILS here on CPU, catching the bug for ~tens of $ instead of an AFI
# build. A PASS means the fused wide read+write commits in RTL exactly as in csim.
#   -trace_level none keeps cosim fast (no waveform dump); -rtl verilog matches the AFI flow.
set proj_dir forge_train_step_s1_proj
set top      forge_train_step_s1
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_train_step_s1.cpp
add_files -tb forge_train_step_s1_tb.cpp -cflags "-DS1_COSIM_MIN"

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csim_design                                  ;# CPU repro: confirms the minimal TB returns 0 (PASS) on csim
csynth_design                                ;# synth the kernel (DSP=128 attn-blend, 4.0ns target)
cosim_design -rtl verilog -trace_level none  ;# RE-RUN the minimal TB vs the RTL — catches the writeback drop

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"
puts "==== Cosim: a PASS proves the mode-2 INIT writeback COMMITS in RTL (no do_opt no-op) ===="

exit
