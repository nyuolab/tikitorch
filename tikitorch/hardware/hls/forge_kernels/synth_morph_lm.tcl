set proj_dir forge_train_step_morph_lm_proj
set top      forge_train_step_morph_lm
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
# Model width ML_D is build-time selectable (default 128 = the deployed sentinel
# AFI; export ML_D=512 for the larger deployed artifact). When ML_D is UNSET the
# kernel is added with NO -D flag, so the default csynth is byte-identical to the
# sentinel build (the header's `#ifndef ML_D` supplies 128). ML_LOG2_D and every
# WORDS_*/OFF_* are derived from ML_D in forge_buffer_morph_lm.hpp.
if {[info exists ::env(ML_D)]} {
    puts "ML_D override: -DML_D=$::env(ML_D)"
    add_files forge_train_step_morph_lm.cpp -cflags "-DML_D=$::env(ML_D)"
} else {
    add_files forge_train_step_morph_lm.cpp
}
# Skip csim (validated locally bit-exact vs the numpy reference + end-to-end
# training; scripts/_validate_kernel_morph_lm.py).

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csynth_design

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"

exit
