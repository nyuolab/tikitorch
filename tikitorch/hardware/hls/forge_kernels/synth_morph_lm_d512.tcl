# synth_morph_lm_d512.tcl — D=512, REDUCED-CLOCK HLS csynth driver for the A3 build.
#
# PARALLEL to synth_morph_lm.tcl (which stays byte-identical for the D=128 sentinel
# and its 4.0 ns target). This variant ALWAYS builds -DML_D=512 and targets a
# REDUCED clock (env TT_CLK_NS, default 6.0 ns = ~166 MHz; A3 accepts 6-8 ns), so
# the recsynth measures the clock actually achievable with the #if ML_D>=512
# URAM/partition pragmas engaged. Emits the same report path as synth_morph_lm.tcl.
#
# ML_D is fixed to 512 here (this is the D=512 driver); ML_LOG2_D is DERIVED in
# forge_buffer_morph_lm.hpp. TT_CLK_NS only sets the create_clock target — it does
# NOT change the D=128 kernel (that uses the other tcl) and does not affect the
# committed sentinel RTL.

set proj_dir forge_train_step_morph_lm_proj
set top      forge_train_step_morph_lm
set target_part "xcvu47p-fsvh2892-2L-e"

set ml_d 512
if {[info exists ::env(ML_D)]} { set ml_d $::env(ML_D) }
set clk_ns 6.0
if {[info exists ::env(TT_CLK_NS)]} { set clk_ns $::env(TT_CLK_NS) }
puts "D512 driver: -DML_D=$ml_d ; target_period_ns=$clk_ns"

open_project -reset $proj_dir
add_files forge_train_step_morph_lm.cpp -cflags "-DML_D=$ml_d"

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $clk_ns

csynth_design

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== D512 synthesis report ===="
puts "  $rpt (target ${clk_ns} ns)"

exit
