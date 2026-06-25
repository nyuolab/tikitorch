# CO-SIMULATION script for forge_train_step_s1 — the mode-2 STEP RTL-no-op catcher.
# ----------------------------------------------------------------------------
# Mirrors synth_train_step_s1_cosim.tcl but compiles the testbench with -DS1_COSIM_STEP so main() runs ONLY
# the minimal mode-2 STEP repro and returns. The S1_COSIM_MIN cosim only ever tested mode-2 INIT (the
# value-copy / re-quantize that writes ONE wide_buf slot per iteration), which COMMITS in RTL. The SILICON
# FACT is narrower: do_opt(STEP) is a NO-OP on the synthesized RTL — a host-driven per-step loop is EXACTLY
# flat, weights never move, even though csim CONVERGES. The STEP path's PASS A writes FIVE wide_buf slots
# per iteration (wfp/m/v/g/dst) vs INIT's ONE; that write fan-out on the single gmem bundle is what gets
# dropped. This harness stages a known nonzero gradient, kicks ONE mode-2 STEP, and asserts a LIVE weight
# (W_HEAD/EMBED deploy + code W_fp masters) actually MOVED and is bit-exact to a tiny CPU golden. If the
# STEP writeback is dropped in RTL, cosim FAILS here on CPU, catching the bug for ~tens of $ instead of an
# AFI build. A PASS means the fused wide read+write STEP commits in RTL exactly as in csim.
#   -trace_level none keeps cosim fast (no waveform dump); -rtl verilog matches the AFI flow.
set proj_dir forge_train_step_s1_step_proj
set top      forge_train_step_s1
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_train_step_s1.cpp
add_files -tb forge_train_step_s1_tb.cpp -cflags "-DS1_COSIM_STEP"

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csim_design                                  ;# CPU repro: confirms the minimal STEP TB returns 0 (PASS) on csim
csynth_design                                ;# synth the kernel (DSP=128 attn-blend, 4.0ns target)
cosim_design -rtl verilog -trace_level none  ;# RE-RUN the minimal STEP TB vs the RTL — catches the STEP writeback drop

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"
puts "==== Cosim: a PASS proves the mode-2 STEP writeback COMMITS in RTL (no do_opt STEP no-op) ===="

exit
