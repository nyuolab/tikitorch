# =============================================================================
# !!! NOT READ BY THE F2 SMALL-SHELL P&R FLOW (S1-proven: the flow reads ONLY
# !!! ${SHELL_MODE}_cl_pnr_user.xdc). The LIVE v3 constraints are in
# !!! small_shell_cl_pnr_user.xdc (async clock groups + rst false path, NO
# !!! pblock). This file is kept as the URAM-budget documentation record.
# =============================================================================
# D=512 SINGLE-KERNEL (NK=1) floorplan — ALL-SLR.
#
# PARALLEL to the D=128 sentinel floorplan (../../cl_forge_morph_lm, NK=4,
# WNS -0.173 ns), which is untouched. At D=512 there is exactly ONE kernel
# instance (KINST[0]) but its footprint spans all three SLRs (v2 two-bank):
#   * 8 MB staging URAM (2 x 4 MB BANK[b].BUF_URAM)       ~= 256 URAM blocks
#   * kernel-internal URAM (resident weights + on-chip grads under the
#     champion flags; MEASURED at csynth, forge_train_step 4.0 ns run) = 605
#   -> 861 of the device's 960 URAM (per-SLR budget is only ~320), so the
#      single instance MUST span SLR0+SLR1+SLR2. Hence one pblock, all clock
#      regions. 861/960 = 90% URAM: if placement fails on URAM, the next lever
#      is 64b-packing the kernel's two 4 MB int32 arrays (mp_buf/g_mp_oc,
#      ~2.25x width waste in 72b URAM lines -> kernel ~605 -> ~350).
#
# As in the D=128 package, the RS_DMA / RS_OCL register slices are EXCLUDED from
# the kernel pblock so they float to the SLR boundaries and Vivado routes the
# crossbar->slice and OCL-slice->kernel hops reg-to-reg through Laguna (the
# recurring inter-SLR failure mode). With a single all-SLR pblock the kernel's
# own logic now also crosses SLR boundaries internally; the shell-provided Laguna
# columns carry those nets. The reset-synchronizer CLR is a false path.
#
# ---------------------------------------------------------------------------
# TODO (REQUIRES FIRST-PASS P&R FEEDBACK — do NOT guess silicon-invalid ranges):
#   The clock-region rectangle below is the FULL user area X0Y0:X7Y11. The AWS
#   shell RESERVES columns/regions in SLR0 and SLR1 (URAM + the shell's own
#   logic), so the first `aws_build_dcp_from_cl.py` run WILL report URAM/BRAM
#   over-utilization or unroutable Laguna in specific regions. Use that run's
#   `utilization` + `timing_summary` reports to:
#     1. Carve the shell-reserved regions out of pb_k0 (remove_cells / resize to
#        the regions the shell utilization report shows as FREE per SLR).
#     2. If URAM in any single SLR still overflows, Pblock the BUF_URAM
#        (xpm_memory_tdpram) instance across SLRs explicitly, separate from the
#        kernel-logic pblock.
#     3. Confirm RS_DMA/RS_OCL land in the boundary regions with Laguna RS to the
#        shared crossbar / OCL slice (as the D=128 -1.147/-0.576 ns fixes did).
#   Until then this spans the whole user area and lets SNAPPING_MODE + the shell's
#   own exclusions place it; expect to iterate on WNS at the reduced clock.
# ---------------------------------------------------------------------------

create_pblock pb_k0
add_cells_to_pblock pb_k0 [get_cells -hier -regexp {.*KINST\[0\]\..*}]
# RS_OCL sits at KINST[0] level; RS_DMA and RS_KRN are per-bank
# (KINST[0].BANK[b].RS_{DMA,KRN} in the v2 two-bank design) — all float to the
# SLR boundaries (reg-to-reg Laguna on both sides of each crossing).
remove_cells_from_pblock pb_k0 [get_cells -hier -regexp {.*KINST\[0\]\.(BANK\[[0-9]+\]\.)?RS_(DMA|OCL|KRN).*}]
# Full user area SLR0 (Y0-3) + SLR1 (Y4-7) + SLR2 (Y8-11), all columns X0-X7.
# TODO: tighten per the first P&R utilization report (shell-reserved regions).
resize_pblock pb_k0 -add {CLOCKREGION_X0Y0:CLOCKREGION_X7Y11}
set_property SNAPPING_MODE ON [get_pblocks pb_k0]

# Async-reset synchronizer CLR pins: recovery/removal is a false path.
set_false_path -to [get_pins -hier -filter {NAME =~ *KINST*rst_pipe_reg*/CLR}]
