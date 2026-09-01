set proj_dir forge_pc_wgrad_proj
set top      forge_pc_wgrad
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_pc_wgrad.cpp
add_files -tb forge_pc_wgrad_tb.cpp

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csim_design                                  ;# run the actual kernel C++ vs the int64 reference
csynth_design                                ;# re-confirm DSP=0 + timing with the faithful dyad

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"

exit
