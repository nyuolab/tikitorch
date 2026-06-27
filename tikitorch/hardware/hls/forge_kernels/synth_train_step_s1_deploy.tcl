# Deploy csynth of the s1 kernel as the fully-MULTIPLIER-FREE tropical-transformer forward:
#   S1_ATTN_HARD=1 (argmax-gather attention, no soft-blend)  +  S1_HEAD_MODE=1 (dyadic-GEMM head)
#   + S1_POS_EMBED=1 (positional embeddings)  +  S1_N_LAYERS=2 (the D=64/L2 on-chip LLM, hard 2.548 bpc).
# Confirms the deploy AFI fits + closes timing and the forward is DSP=0 (the attn soft-blend removed).
set proj_dir forge_train_step_s1_proj
set top      forge_train_step_s1
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_train_step_s1.cpp -cflags "-DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1 -DS1_N_LAYERS=2"

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns

csynth_design     ;# deploy fit/timing; expect DSP=0 on the forward (no attn soft-blend)

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"
exit
