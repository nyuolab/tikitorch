set proj_dir forge_xgrad_wide_proj
set top      forge_xgrad_wide
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_xgrad_wide.cpp
add_files -tb forge_xgrad_wide_tb.cpp

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csim_design                                  ;# run the actual kernel C++ vs the int64 reference
csynth_design                                ;# confirm DSP=0 (no multiplier) + 4ns timing with the faithful shift

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"

exit
