# cl_forge_pc_ddr — F2 CL for the multiplier-free PC backward kernel (forge_pc_wgrad)

Deploys the DSP=0, bit-exact **predictive-coding cascade backward** layer-step
(`forge_pc_wgrad`; see `results/predictive_coding/PC_BACKWARD_SILICON.md`) on F2 silicon, so
the gradient produced **on the chip** can be checked against tikitorch's `cascade_grads`.

Reuses the proven drivable on-card-DDR architecture of `cl_forge_muzero_ddr` verbatim (host
stages the I/O buffer into DDR over DMA-PCIS, the kernel reads/writes it via its m_axi — both
reach DDR through a 2→1 AXI crossbar; no host IOMMU in the loop):

```
ocl (AXI4-Lite) --[pc_ocl_slice]--> forge_pc_wgrad s_axi_control   (5-bit control map)
sh_cl_dma_pcis (512b, host staging) --\
                                       >--[axi_crossbar_ddr 2->1]--> sh_ddr --> on-card DDR4
forge_pc_wgrad m_axi gmem (512b)     --/
```

**What differs from cl_forge_muzero_ddr:** only the kernel instance (`forge_pc_wgrad`) and its
**5-bit** `s_axi_control` slice (the kernel has a single `buffer` arg → a small register map,
vs the muzero train-step's 13-bit). The kernel's gmem is a **native 512b word bus** (16 int32
lanes/word; `burst_load`/`burst_store` pack/unpack) so it wires straight to the 512b crossbar
— **no width converter**.

**Buffer at DDR addr 0** holds the flat int32 layout
`[ e | W | acts | relu_mask | skip_gate | gW | e_down | gB ]` (D=256, N=16; all regions
16-int32-aligned). The host DMA-PCIS-writes the inputs, sets the kernel BUFFER pointer reg to
0, kicks `ap_start`, the kernel writes `gW`/`e_down`/`gB` back to DDR, the host reads them out
and compares to `cascade_grads` (must be bit-exact — proven in CSIM).

**Build inputs (committed):** `design/cl_forge_pc_ddr.sv` (+ `*_defines.vh`, `cl_id_defines.vh`),
`build/scripts/gen_ip.tcl` (`pc_ocl_slice` AXI4-Lite slice + `axi_crossbar_ddr` 2→1 SASD 512b/
64-addr/16-ID), `build/scripts/synth_cl_forge_pc_ddr.tcl` (reads `sh_ddr` deps + the two IP),
`build/constraints/*.xdc`.

**Build** (isolated from the parallel muzero/LM AFI builds — distinct state/tag/S3):
```
CL=cl_forge_pc_ddr PKG_REL=tikitorch/hardware/hdk/cl_forge_pc_ddr \
  bash scripts/run_pc_afi_build.sh launch     # HLS forge_pc_wgrad -> scaffold CL -> synth/P&R -> AFI tarball
bash scripts/run_pc_afi_build.sh status         # poll S3 (fail-fast at synth ~20m before the ~3h P&R)
```
Then `create-fpga-image` on the tarball → AGFI → load on f2.6xlarge, stage DDR via
`fpga_dma`/BAR4, kick, read back, compare to `cascade_grads`.
