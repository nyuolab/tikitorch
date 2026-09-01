# =============================================================================
# small_shell_cl_pnr_user.xdc -- S1 multiplier-free transformer-trainer AFI floorplan
# v14 = v12's PROVEN config (kernel pblock + FREE crossbar = WNS -0.062 ns) + the
#       async clock-groups finally applied with the CORRECT clock name.
#
# PACKAGING (verified): the F2 HDK P&R flow reads ONLY ${SHELL_MODE}_cl_pnr_user.xdc
# = small_shell_cl_pnr_user.xdc from the CL constraints dir (aws_build.full.log line
# 7265-7267). cl_timing_user.xdc / cl_synth_user.xdc are NOT read. pblock_CL is
# created by the shell BEFORE this file is read, so `set_property parent pblock_CL`
# resolves.
#
# THE MISS + ITERATION HISTORY (clk_main_a0, 250 MHz, 4 ns):
#   v9/v10  -0.409  no effective constraints (the async-group authored in
#                   cl_timing_user.xdc was a silent no-op -- wrong file).
#   v11     -0.543  LANE=4, no floorplan -> kernel SIZE is irrelevant; the miss is
#                   the DDR crossbar (XBAR_DDR) wready mux crossing SLR boundaries.
#   v12     -0.062  + this kernel pblock (FREE crossbar) -> floorplan WORKS, 62 ps away.
#   v13     -0.302  + ALSO confining XBAR_DDR to SLR1 X4/X5 -> WORSE: it pulled the
#                   crossbar away from the STATIC DMA-PCIS shell endpoint, lengthening
#                   the fo=577 wready net (the real cost). => do NOT pblock the crossbar;
#                   v12's free placement is the better compromise. (Reverted here.)
#   The worst path is the SH_DDR async-FIFO wready -> XBAR_DDR wready mux -> a
#   high-fanout (fo=577) net to a STATIC (shell DMA-PCIS) endpoint, ~89% route.
#
# THE FIX (v14):
#   (1) KEEP v12's kernel pblock: soft-confine the trainer compute (u_kernel + the
#       new 512-bit kernel-mem CDC GMEM_CDC) to the full compute region (SLR2 +
#       SLR1/SLR0 X0-X3 blocks), OFF the narrow X5 DDR column, so XBAR_DDR + SH_DDR +
#       DMA-PCIS place compactly there like the proven muzero_ddr build. This is what
#       took the miss from -0.409 to -0.062. (IS_SOFT is force-overridden to FALSE by
#       the PR/hierarchical flow -- set FALSE explicitly; the HARD pblock gave -0.062.)
#   (2) ASYNC CLOCK GROUPS with the CORRECT kernel clock name. The kernel runs on the
#       AWS_CLK_GEN MMCM_A output, which Vivado names **clk_out1_clk_mmcm_a** (NOT
#       "clk_extra_a1" -- that name is a net/comment, not a clock object; v13's
#       get_clocks clk_extra_a1 returned empty -> CRITICAL WARNING 12-4739, async never
#       applied; "clocked by clk_out1_clk_mmcm_a" is confirmed in the v13 timing report).
#       clk_main_a0 (shell 250) and clk_out1_clk_mmcm_a (kernel 125) are async and cross
#       ONLY through the s1_axil_cdc / s1_axi_cdc converters. Declaring them async stops
#       Vivado timing the ~255 phantom inter-clock crossings (WARNING Timing 38-164),
#       freeing the router for the last 62 ps. Bare command (XDC forbids Tcl `if`).
#
# pblock_CL CR ownership (verified vs results/fpga_silicon/ddr_arch/*small_shell* AWS
# refs): SLR2 X0Y8:X7Y11 (full); SLR1 X0Y4:X3Y7 (+X5Y4:X5Y7 DDR col +X4 partials);
# SLR0 X0Y0:X3Y3 (+partials). Kernel uses only X0Y8:X7Y11, X0Y4:X3Y7, X0Y0:X3Y3 --
# all owned. The X5 DDR column is deliberately left to XBAR_DDR/SH_DDR. SH_DDR and
# AWS_CLK_GEN are NOT pblocked (shell/IO-driven). Kernel HLS + CL wiring UNCHANGED.
# =============================================================================

###############################################################################
# Kernel pblock: soft-confine u_kernel + GMEM_CDC to the compute region (SLR2 +
# SLR1/SLR0 X0-X3 blocks), OFF the narrow X5 DDR column. (v12-proven: -0.062 ns.)
###############################################################################
create_pblock pblock_S1_KERNEL
resize_pblock pblock_S1_KERNEL -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
resize_pblock pblock_S1_KERNEL -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y7}
resize_pblock pblock_S1_KERNEL -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3}
set_property SNAPPING_MODE ON    [get_pblocks pblock_S1_KERNEL]
set_property IS_SOFT       FALSE [get_pblocks pblock_S1_KERNEL]
set_property parent  pblock_CL   [get_pblocks pblock_S1_KERNEL]
add_cells_to_pblock pblock_S1_KERNEL [get_cells [list \
    WRAPPER/CL/u_kernel \
    WRAPPER/CL/GMEM_CDC ]]

###############################################################################
# Async clock groups: kernel clk_out1_clk_mmcm_a (125) is async to shell
# clk_main_a0 (250); they cross only through the CDC converters. Bare command
# (no Tcl control flow). Correct clock name (v13 proved clk_extra_a1 is not it).
###############################################################################
set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks clk_main_a0] \
  -group [get_clocks -include_generated_clocks clk_out1_clk_mmcm_a]
