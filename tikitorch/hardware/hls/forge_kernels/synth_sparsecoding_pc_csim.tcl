set proj_dir forge_sparsecoding_pc_proj
set top      forge_sparsecoding_pc
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_sparsecoding_pc.cpp
add_files -tb forge_sparsecoding_pc_tb.cpp

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csim_design
csynth_design
exit
