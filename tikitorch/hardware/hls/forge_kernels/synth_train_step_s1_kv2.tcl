# Kernel-v2 deploy csynth: the COHERENT strictly-multiplier-free (DSP=0) tropical-transformer forward.
#   S1_KV2=1 (4-term CSD weights via wmac4 + per-matrix pow2 rescale + 4-term CSD-reciprocal LayerNorm)
#   D=128 / L4 / H4 / FFN512  (the deploy model, BIT-EXACT to the sim, gates 2.42 bpc, beats bigram 3.47)
#   + S1_ATTN_HARD=1 (argmax-gather) + S1_HEAD_MODE=1 (dyadic-GEMM head) + S1_POS_EMBED=1 + S1_FWD_ONLY=1.
# Checks: DSP_MULT must be 0 (all shift-add), fit (<= 1 SLR ideally), timing met @ 4.0ns.
# D=128 RISK: the D=128 monolithic TRAINER missed timing before, but this is FWD-ONLY (much smaller).
set proj_dir forge_train_step_s1_kv2_proj
set top      forge_train_step_s1
set target_part "xcvu47p-fsvh2892-2L-e"
set target_period_ns 4.0

open_project -reset $proj_dir
add_files forge_train_step_s1.cpp -cflags "-DS1_KV2=1 -DS1_D=128 -DS1_LOG2_D=7 -DS1_H=4 -DS1_DH=32 -DS1_FFN=512 -DS1_N_LAYERS=4 -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1"

open_solution -reset solution1
set_top $top
set_part $target_part
create_clock -period $target_period_ns
csynth_design

set rpt $proj_dir/solution1/syn/report/${top}_csynth.rpt
puts "==== Synthesis report ===="
puts "  $rpt"
exit
