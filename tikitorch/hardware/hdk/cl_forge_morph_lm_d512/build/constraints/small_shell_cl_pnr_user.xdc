# =============================================================================
# small_shell_cl_pnr_user.xdc — D=512 MorphSSM-LM v3 (CLKGEN) LIVE P&R constraints.
#
# PACKAGING (the S1-proven lesson, verified in its aws_build.full.log): the F2
# HDK P&R flow reads ONLY ${SHELL_MODE}_cl_pnr_user.xdc = THIS FILE from the CL
# constraints dir. cl_pnr_user_floorplan.xdc / cl_timing_user.xdc are NOT read
# at P&R — the v2 build's floorplan pblock was a silent no-op (and free
# placement still routed; the v2 failure was intrinsic path length at 4 ns,
# not placement).
#
# v3 constraints, deliberately MINIMAL:
#   (1) ASYNC CLOCK GROUPS — the one constraint v3 CANNOT ship without. The
#       user design runs on the AWS_CLK_GEN MMCM_A output, which Vivado names
#       **clk_out1_clk_mmcm_a** (NOT "clk_extra_a1" — that is a net name; the
#       S1 v13 build proved get_clocks clk_extra_a1 returns empty and the
#       async group silently never applies). clk_main_a0 (shell 250) and
#       clk_out1_clk_mmcm_a (user 125) cross ONLY through the two async
#       axi_clock_converters (PCIS_CDC / OCL_CDC); declaring them async stops
#       Vivado timing the phantom inter-clock crossings. Bare command (XDC
#       forbids Tcl control flow).
#   (2) rst_pipe recovery/removal false path (async-assert synchronizer).
#   (3) NO pblock: at 861/960 URAM the design must span all SLRs anyway —
#       free placement IS the all-SLR floorplan (the S1 v13 lesson: an extra
#       pblock on the interconnect made timing WORSE). At 8 ns the v2-measured
#       worst path (6.7 ns) closes with >1.2 ns slack without help.
# =============================================================================

set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks clk_main_a0] \
  -group [get_clocks -include_generated_clocks clk_out1_clk_mmcm_a]

# Async-reset synchronizer CLR pins: recovery/removal is a false path.
set_false_path -to [get_pins -hier -filter {NAME =~ *KINST*rst_pipe_reg*/CLR}]
