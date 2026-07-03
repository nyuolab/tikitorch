# xdma_shell_cl_pnr_user.xdc — CL place-and-route user timing constraints.
# NOTE on the filename: the F2 HDK shell reads ${SHELL_MODE}_cl_pnr_user.xdc during P&R
# (build_level_1_cl.tcl:56; SHELL_MODE defaults to xdma_shell). The older cl_timing_user.xdc
# / cl_synth_user.xdc names are NOT read by this shell — this is the file that actually applies.

# --- Reset-CDC false path (the s2 D=512 kernel clock-domain crossing) ---
# cl_forge_pc_xformer_s2.sv runs the kernel on a derived clk_kernel (clk_out1_clk_div_kernel,
# 150 MHz) to close the heavy BF16 / K-wide-MAC datapath. Its reset synchronizer `rst_kernel_pipe`
# takes an ASYNC clear from rst_main_n (clk_main_a0 / 250 MHz domain) and is an async-assert /
# sync-deassert synchronizer, so the recovery/removal timing on that crossing is intentionally
# NOT a real constraint (the synchronizer resolves metastability on the synchronized deassert).
# On the mp CL, post-route showed these recovery checks as the ONLY timing failures (WNS -3.270 ns
# on rst_kernel_pipe_reg[*]/CLR) while the 150 MHz kernel compute paths MEET (+0.206 ns). The s2
# CDC + reset synchronizer are identical, so the same false-path applies. False-path the async reset
# crossing -> the design closes.
set_false_path -to [get_pins -hierarchical -filter {NAME =~ "*rst_kernel_pipe_reg*/CLR"}]
