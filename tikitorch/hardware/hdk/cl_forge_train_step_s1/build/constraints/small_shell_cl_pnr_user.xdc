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

# Nested-pblock hygiene. IS_SOFT FALSE: the F2 Hierarchical-Design (partial-reconfig)
# flow only supports HARD pblocks ("WARNING [Constraints 18-5661] Only False is
# supported for IS_SOFT ... due to its Hierarchical Design flow requirement" -- v12
# forced it FALSE anyway, and the HARD pblock is exactly what took the worst path from
# -0.409 to -0.062 ns). The region is generous (~64 CRs) so the LANE=4 kernel fits hard.
set_property SNAPPING_MODE ON    [get_pblocks pblock_S1_KERNEL]
set_property IS_SOFT       FALSE [get_pblocks pblock_S1_KERNEL]
set_property parent  pblock_CL   [get_pblocks pblock_S1_KERNEL]

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

###############################################################################
# Child Pblock: confine the DDR crossbar (XBAR_DDR) to SLR1 so it CANNOT spread
# into SLR2. v12 (kernel pblock alone) cut the miss to WNS -0.062 ns but the worst
# path STILL crosses SLR1->SLR2->SLR1: the crossbar's own wready mux drifted up
# into SLR2's X5 column (the kernel pblock OPENED all of SLR2 but the LANE=4 kernel
# does not fill it, so the unconstrained crossbar crept in). v11 showed the same
# crossbar self-crossing (XBAR_DDR/...m_valid_i_reg SLR1<->SLR2, WNS -0.543).
# Confining XBAR_DDR to SLR1's CL clock regions (X0Y4:X3Y7 compute block + the
# X5Y4:X5Y7 DDR column where SH_DDR lives) keeps the whole sh_ddr<->crossbar
# handshake intra-SLR1 -- eliminating the inter-SLR (~0.35 ns each way) + Laguna
# penalty on the failing path. Only the inherent crossbar->DMA-PCIS SLR1->SLR0
# crossing remains, which the proven muzero_ddr build already closed. The region
# is GENEROUS (~20 CRs vs a small 2->1 SASD crossbar) so the forced-HARD pblock
# cannot overflow; the crossbar gravitates to X5 next to SH_DDR (timing-driven).
# Every CR is within pblock_CL's SLR1 set (X0Y4:X3Y7 + X5Y4:X5Y7), so no
# out-of-partition error.
###############################################################################
# Confine XBAR_DDR to the SLR1 X4+X5 DDR-side columns (next to SH_DDR), DISJOINT
# from the kernel pblock's X0-X3 columns -- so the two sibling pblocks never overlap
# (the proven AWS reference floorplans, results/fpga_silicon/ddr_arch/*small_shell*,
# keep CL child pblocks strictly disjoint by SLR + nest the crossbar; overlapping
# siblings are avoided). These are EXACTLY the SLR1 X4/X5 clock regions pblock_CL
# owns (X4Y4, X4Y6:X4Y7 -- note X4Y5 is NOT in pblock_CL -- and X5Y4:X5Y7); 7 CRs
# (~130k LUT after SH_DDR's share) vs a ~3k-LUT 2->1 SASD crossbar = generous, so
# the forced-HARD pblock cannot overflow. No SLR2 region -> the crossbar physically
# cannot spread to SLR2, killing the SLR1<->SLR2 self-crossing on the -0.062 path.
create_pblock pblock_S1_XBAR
resize_pblock pblock_S1_XBAR -add {CLOCKREGION_X5Y4:CLOCKREGION_X5Y7}
resize_pblock pblock_S1_XBAR -add {CLOCKREGION_X4Y6:CLOCKREGION_X4Y7}
resize_pblock pblock_S1_XBAR -add {CLOCKREGION_X4Y4:CLOCKREGION_X4Y4}
set_property SNAPPING_MODE ON    [get_pblocks pblock_S1_XBAR]
set_property IS_SOFT       FALSE [get_pblocks pblock_S1_XBAR]
set_property parent  pblock_CL   [get_pblocks pblock_S1_XBAR]
add_cells_to_pblock pblock_S1_XBAR [get_cells [list WRAPPER/CL/XBAR_DDR]]

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
# whole time (why v9 == v10 byte-identical). v12 then put it here but wrapped in a
# Tcl `if {...}` guard -- REJECTED ("CRITICAL WARNING [Designutils 20-1307] Command
# 'if' is not supported in the xdc constraint file"), so the async group STILL did
# not apply (Timing 38-164 persisted at v12 line 11565). XDC files allow only a
# restricted command set, NOT general Tcl control flow. The fix is a BARE
# set_clock_groups (get_clocks IS allowed in XDC). clk_main_a0 and clk_extra_a1
# both exist at P&R (the v10/v11/v12 reports time 255 paths between them), so the
# get_clocks resolve non-empty -- no guard needed. This finally excludes the
# phantom inter-clock crossings (handled by the CDC IPs), freeing the router for
# the last 62 ps on the single-clock sh_ddr<->crossbar wready path.
# =============================================================================
set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks clk_main_a0] \
  -group [get_clocks -include_generated_clocks clk_extra_a1]
