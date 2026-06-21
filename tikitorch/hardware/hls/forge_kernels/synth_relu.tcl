set proj_dir forge_relu_proj
set top      forge_relu
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_relu.cpp
add_files -tb forge_relu_tb.cpp

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csim_design                                  ;# run the actual kernel C++ (fwd+bwd) vs the int64 reference
csynth_design                                ;# confirm DSP=0 + 4.0 ns timing (pure compare/select)

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"

exit
