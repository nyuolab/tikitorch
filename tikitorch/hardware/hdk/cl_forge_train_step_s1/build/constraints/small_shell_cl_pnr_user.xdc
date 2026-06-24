# =============================================================================
# small_shell_cl_pnr_user.xdc  --  S1 multiplier-free transformer-trainer AFI floorplan
# CANDIDATE A -- "kernel-to-SLR2".
#
# PACKAGING (VERIFIED, read first): the AWS HDK P&R flow reads
# "${SHELL_MODE}_cl_pnr_user.xdc". On the F2 small shell SHELL_MODE=small_shell,
# so the file the flow actually parses is "small_shell_cl_pnr_user.xdc", NOT a
# plain "cl_pnr_user.xdc". CONFIRMED on the real v11 S1 build log
# (results/fpga_silicon/s1_afibuild/aws_build.full.log line 6837-6839):
#     ## read_xdc ${constraints_dir}/${SHELL_MODE}_cl_pnr_user.xdc
#     Parsing XDC File [.../constraints/small_shell_cl_pnr_user.xdc]
# That v11 build parsed an EMPTY small_shell_cl_pnr_user.xdc (no create_pblock
# from it) -> pure auto-placement -> WNS -0.409 ns. The runner
# (scripts/run_s1_afi_build.sh line 140) stages EVERY *.xdc from the package
# build/constraints/ into the CL, so this file MUST exist HERE under the
# small_shell_ name to take effect. A plain cl_pnr_user.xdc is staged but never
# read (silent no-op). pblock_CL is created by the shell's
# small_shell_level_1_fp_cl.xdc (build log line 6812, parsed BEFORE this file at
# 6837), so `set_property parent pblock_CL` below resolves.
#
# ---------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# The S1 trainer AFI post-routes at WNS -0.409 ns on clk_main_a0 (250 MHz). The
# worst path (v10t.rpt, VERIFIED) is the crossbar<->(sh_ddr / DMA-PCIS) `wready`
# handshake forced through the narrow DDR column:
#   WRAPPER/CL/SH_DDR/...asyncfifo_axi/...ram_full_i_reg/C  (SLICE_X140Y492, SLR1)
#     -> s_axi_wready_INST_0 (SLICE_X140Y491, SLR1)
#     -> net fo=3, 0.590 ns LATERAL hop X140 -> X163 -->
#   WRAPPER/CL/XBAR_DDR/...gen_wmux.wmux_aw_fifo/...wready..._replica
#                                            (SLICE_X163Y489 / X163Y484, SLR1)
#     -> net fo=577, 1.167 ns crossing SLR1 -> SLR0 --> DMA-PCIS dest (SLR0).
# Data path 3.752 ns, ~89% route. The proven muzero_ddr build
# (agfi-085dca07f374aa1cd) has the IDENTICAL crossbar + sh_ddr + DMA-PCIS
# structure and the SAME pblock_CL geometry, and CLOSED (+0.123 ns) with EMPTY
# constraints. So the SLR1->SLR0 crossing is NOT fatal by itself; S1 misses ONLY
# because its extra CL logic crowds the narrow X5 DDR column + the SLR0/SLR1
# boundary, degrading that wready net's placement and replication. v11 (LANE=4,
# half-size kernel, EMPTY floorplan) STILL missed at -0.409 ns -> the cause is
# DDR-column congestion, not raw kernel size, so a FLOORPLAN move is required.
#
# THE FIX (Candidate A -- kernel-to-SLR2)
# Pull the big compute (u_kernel + its kernel-side memory CDC GMEM_CDC) OFF the
# congested SLR0/SLR1 boundary and OUT of the narrow X5 DDR column, soft-confining
# it to the full-width compute region (all of SLR2 + the X0-X3 compute blocks in
# SLR1/SLR0). This frees the X5 strip + the SLR boundary so the DDR datapath
# (XBAR_DDR + SH_DDR + DMA-PCIS) can place compactly there exactly like the
# proven muzero_ddr build -- restoring its short, closeable wready handshake.
#
# This is the SAME nested-child-of-pblock_CL idiom the AWS reference floorplans
# use (cl_mem_perf / cl_dram_hbm_dma small_shell_cl_pnr_user.xdc): they create a
# pblock_CL_SLR2 = CLOCKREGION_X0Y8:X7Y11 child, parent it to pblock_CL, and map
# compute masters into it. We touch ONLY the CL logic we own; NOTHING here
# pblocks SH_DDR (shell IP, IO-driven) or AWS_CLK_GEN (MMCM). The kernel HLS RTL
# and the CL functional wiring are UNCHANGED.
#
# SAFETY POSTURE
#  - Every clock region used is OWNED by pblock_CL. VERIFIED against the AWS
#    small-shell reference floorplans (results/fpga_silicon/ddr_arch/
#    cl_mem_perf_build_small_shell_cl_pnr_user.xdc +
#    cl_dram_hbm_dma_build_small_shell_cl_pnr_user.xdc):
#       pblock_CL SLR2 = X0Y8:X7Y11 (full)
#       pblock_CL SLR1 = X0Y4:X3Y7  (+ X5Y4:X5Y7 DDR col + X4Y4 + X4Y6:X4Y7)
#       pblock_CL SLR0 = X0Y0:X3Y3  (+ X4Y0:X7Y0 + X4Y3 + X6Y1 + X5Y1:X5Y3)
#    -> X0Y8:X7Y11, X0Y4:X3Y7, X0Y0:X3Y3 are ALL within pblock_CL. We add NO CR
#    that pblock_CL lacks, so no "CR not in parent" build ERROR. We deliberately
#    EXCLUDE the X5 DDR column so the DDR datapath keeps it.
#  - IS_SOFT TRUE: a placement GUIDE (attractor), not a hard wall. The kernel
#    mass is pulled into SLR2 + the compute blocks and away from the DDR column,
#    but the placer may relax the boundary if needed -- so a mis-estimate of the
#    LANE=4 kernel footprint CANNOT hard-fail the ~6 h build (no place-infeasible
#    error, no forced timing regression). The region is generous (full SLR2 +
#    both compute blocks ~= the entire CL compute area, ~64 clock regions), so
#    the half-size LANE=4 kernel (which already auto-placed in these blocks)
#    fits with wide margin.
#  - SNAPPING_MODE ON (AWS-canonical for nested pblocks): all ranges are whole
#    clock regions, so snapping is a no-op alignment, but ON keeps the derived
#    ranges inside the parent, consistent with the proven AWS pattern.
# =============================================================================

###############################################################################
# Child Pblock: soft-confine the trainer compute (u_kernel + kernel-side
# GMEM_CDC) to the full-width compute region (SLR2 + SLR1/SLR0 X0-X3 blocks),
# OFF the narrow X5 DDR column and the congested SLR0/SLR1 DDR boundary.
###############################################################################
create_pblock pblock_S1_KERNEL

# All of SLR2 (full-width compute, 32 clock regions; the largest contiguous
# compute area pblock_CL owns -- where the trainer kernel mass belongs).
resize_pblock pblock_S1_KERNEL -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}

# SLR1 compute block (X0-X3, Y4-Y7) -- the X0-X3 columns pblock_CL owns in SLR1.
# NOTE: the X5Y4:X5Y7 DDR column is intentionally NOT added (left for the xbar).
resize_pblock pblock_S1_KERNEL -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y7}

# SLR0 compute block (X0-X3, Y0-Y3) -- the X0-X3 columns pblock_CL owns in SLR0.
# NOTE: the X5Y1:X5Y3 / X4*/X6Y1 DDR-column partials are intentionally NOT added.
resize_pblock pblock_S1_KERNEL -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3}

# Nested-pblock hygiene + soft (guide, not wall) -- see SAFETY POSTURE above.
set_property SNAPPING_MODE ON   [get_pblocks pblock_S1_KERNEL]
set_property IS_SOFT       TRUE [get_pblocks pblock_S1_KERNEL]
set_property parent  pblock_CL  [get_pblocks pblock_S1_KERNEL]

# ---- Cell mapping (exact full hier paths; confirmed against the CL .sv) ----
# u_kernel  = forge_train_step_s1 (cl_forge_train_step_s1.sv line 192), the big
#             multiplier-free trainer compute.
# GMEM_CDC  = s1_axi_cdc (line 227), the NEW 512-bit kernel-memory async-FIFO CDC
#             that muzero_ddr never had -- the prime crowding suspect. Pulling it
#             into the compute region (away from the X5 strip where the sh_ddr /
#             crossbar async FIFOs live) is the core of the fix. Its 250 MHz
#             master output then routes from the compute block to XBAR_DDR SI0;
#             that hop has slack today (the failing path is the wready handshake,
#             not this CDC master fan-out) and IS_SOFT lets the master side
#             drift toward X5 if P&R prefers.
# We deliberately do NOT map XBAR_DDR or SH_DDR here -- they must stay free to
# place in/near the X5 DDR column + SLR boundary (the muzero_ddr-proven home).
add_cells_to_pblock pblock_S1_KERNEL [get_cells [list \
    WRAPPER/CL/u_kernel \
    WRAPPER/CL/GMEM_CDC ]]

# =============================================================================
# ASYNC CLOCK GROUPS (the kernel clk_extra_a1=125 MHz is async to the shell
# clk_main_a0=250 MHz; they cross ONLY through the s1_axil_cdc / s1_axi_cdc
# axi_clock_converter CDC IPs which carry their own synchronizers). Without this,
# Vivado TIMES every clk_main_a0<->clk_extra_a1 crossing as synchronous -- the v10
# build log shows exactly that (WARNING [Timing 38-164] "Inter clock paths are
# considered valid unless explicitly excluded by ... set_clock_groups") with 255
# such phantom paths, wasting router effort on impossible crossings. This was
# AUTHORED earlier in cl_timing_user.xdc but that file is NEVER read by the F2
# flow (it reads ONLY ${SHELL_MODE}_cl_pnr_user.xdc = small_shell_cl_pnr_user.xdc,
# confirmed at aws_build.full.log line 7265-7267) -- so it was a silent no-op the
# whole time (why v9 == v10 byte-identical). Declaring the groups async here (the
# file that IS read) stops Vivado timing the crossings, freeing the router to help
# the marginal -0.409 ns sh_ddr<->crossbar wready path. Guarded: if either clock
# is absent at P&R the get_clocks return empty and the whole block is skipped, so
# a name mismatch CANNOT error the ~6 h build.
# =============================================================================
set _cmain  [get_clocks -quiet -include_generated_clocks clk_main_a0]
set _cextra [get_clocks -quiet -include_generated_clocks clk_extra_a1]
if {[llength $_cmain] > 0 && [llength $_cextra] > 0} {
  set_clock_groups -asynchronous -group $_cmain -group $_cextra
  puts "INFO: S1 cl_pnr_user -- set_clock_groups async clk_main_a0 <-> clk_extra_a1 applied"
} else {
  puts "WARNING: S1 cl_pnr_user -- clk_main_a0 or clk_extra_a1 not found; async group SKIPPED (m=[llength $_cmain] e=[llength $_cextra])"
}
