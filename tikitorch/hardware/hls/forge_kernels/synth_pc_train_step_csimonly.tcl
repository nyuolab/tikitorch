set proj_dir forge_pc_train_step_proj
set top      forge_pc_train_step
set target_part "xcvu47p-fsvh2892-2L-e"

open_project -reset $proj_dir
add_files forge_pc_train_step.cpp
add_files -tb forge_pc_train_step_tb.cpp

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period 4.0

csim_design                                  ;# CSIM ONLY (csynth already confirmed DSP=0/timing/fit for TN=32)
exit
