# S1 AFI floorplan notes — the SLR-crossing timing miss + the fix decision tree

## The miss (v9 = v10, byte-identical worst path)
- WNS **-0.409 ns** / TNS -89.181 (clk_main_a0, 250 MHz, 4 ns budget).
- Worst path: `WRAPPER/CL/SH_DDR/...AXI_CCF/gen_async_conv.asyncfifo_axi/...ram_full_i_reg`
  (SLICE_X140Y492, **SLR1**, CR ~X5Y4) → `s_axi_wready_INST_0` → **net fo=3 0.590 ns** →
  `WRAPPER/CL/XBAR_DDR/...gen_wmux.wmux_aw_fifo/...wready..._replica` (SLICE_X163Y489, SLR1) →
  **net fo=577 1.167 ns crossing SLR1→SLR0** → destination SLICE_X177Y322 (**SLR0**, CR ~X5Y2).
- Data path 3.752 ns, **89% route**. It is the crossbar↔(sh_ddr / DMA-PCIS) `wready` handshake.
- `set_clock_groups -asynchronous` (v10) correctly did NOT change it — single-clock clk_main_a0 path.

## Why it misses here but NOT in the proven muzero_ddr build
muzero_ddr (agfi-085dca07f374aa1cd) has the SAME crossbar + sh_ddr + DMA-PCIS structure and the
SAME pblock geometry, and it CLOSED (+0.123 ns) with EMPTY constraints (pure auto-placement). The
structural SLR0↔SLR1 crossing (DMA-PCIS is shell-fixed in SLR0; the DDR controller is in SLR1) is
therefore NOT fatal by itself. S1 misses because its extra logic **crowds the narrow DDR column**,
degrading that net's placement + the replication of the fo=577 wready.

## pblock_CL geometry (from aws-fpga hdk/common/shell_stable .../small_shell_level_1_fp_cl.xdc)
- **SLR2 (compute):** `CLOCKREGION_X0Y8:X7Y11` (full).
- **SLR1:** `CLOCKREGION_X0Y4:X3Y7` (compute block) **+ `CLOCKREGION_X5Y1:X5Y7`** (the narrow DDR column).
- **SLR0:** `CLOCKREGION_X0Y0:X3Y3` (compute block) + partials (X4*, X6Y1).
- ⇒ the DDR datapath lives in the **CR X5 column, Y1–Y7** — a 2-SLR-tall narrow strip. The compute
  region (kernel) is the X0–X3 blocks + all of SLR2. SLICE↔CR: ~120 SLICE-rows per CR row;
  SLR0 = SLICE Y0–479 (CR Y0–Y3), SLR1 = Y480–959 (CR Y4–Y7), SLR2 = Y960+ (CR Y8–Y11).
- `SNAPPING_MODE ON`, `IS_SOFT 0`. Top/bottom pblock edges align to CRs → Y480 edge tiles are
  PROHIBITED (routing-containment) — relevant: the failing path sits right at that SLR0/SLR1 edge.

## The real suspect (refined): s1_axi_cdc in the X5 strip
The NEW logic vs muzero_ddr that lands in/near the X5 DDR column:
1. **s1_axi_cdc** — the kernel-mem CDC (512-bit AXI4 async FIFO, 125→250). muzero's kernel went
   straight to the crossbar with no CDC. A 512-bit async FIFO crammed into the narrow X5 strip
   alongside the crossbar + sh_ddr async FIFOs is the likely crowding source.
2. AWS_CLK_GEN MMCM (small, clock-region resource).
3. The bigger trainer kernel (but it should sit in X0–X3/SLR2, not the X5 strip).

⇒ **LANE=4 (v11) reduces only the kernel** — it helps if the kernel was spilling into SLR1's X5
neighborhood, but may NOT fix crowding caused by s1_axi_cdc itself. v11's worst-path will tell us:
if the path is unchanged again, the s1_axi_cdc hypothesis is confirmed and the fix is floorplan.

## Fix decision tree (pick after v11's result + margin are known)
1. **v11 = LANE=4** [TESTING]: de-crowd via smaller kernel. Cheap, bit-exact. If it closes → done.
2. **pblock s1_axi_cdc OUT of the X5 strip** into the compute block (e.g. SLR0 X0–X3) so the X5
   column holds only crossbar + sh_ddr (muzero-like). Risk: the CDC 250 MHz master output then
   routes X3→X5 to the crossbar (could be long). cl_pnr_user.xdc, no kernel re-csim.
3. **pblock the crossbar (XBAR_DDR) compactly in CR X5Y4 (SLR1) next to sh_ddr** + force wready
   replication, so the sh_ddr→crossbar hop is tight and only the (inherent) crossbar→DMA-PCIS
   crossing remains (which muzero tolerated). cl_pnr_user.xdc.
4. **Replace the SASD crossbar with a trivial 2:1 time-disjoint mux** (host-stage vs kernel-compute
   are time-disjoint per the F2-run de-risk) — eliminates the wready-mux-replica logic that is the
   exact failing cells. CL change, no kernel change, geometry-free.
5. **On-chip scratchpad** (D=64 working set = 1.543 MB fits in URAM/BRAM): replace sh_ddr+crossbar+
   s1_axi_cdc with a BRAM/URAM AXI slave — eliminates the entire failing subsystem. Biggest rework;
   host stages ~228 KB of weights+LUTs via OCL (~57 ms one-time) since DMA-PCIS→DDR goes away.

Recommended order if v11 misses: (4) trivial-mux or (2) pblock-the-CDC-out — both directly target
the identified cause without touching the proven kernel. Use a workflow to author+adversarially-vet
whichever, with the geometry above.
