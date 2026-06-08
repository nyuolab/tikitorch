set proj_dir forge_train_step_2layer_proj
set top      forge_train_step_2layer
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_train_step_2layer.cpp
# Skip csim (validated locally; HDK build path uses csynth only).

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csynth_design

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"

exit
