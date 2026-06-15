# cl_forge_muzero_ddr — DDR variant of the forward MuZero chess CL (drivable for dynamic power)

The sibling `cl_forge_muzero` (PCIM→host-DRAM) routes + loads on F2 and its PCIM master is electrically
clean, but it **can't be driven for inference from EC2 userspace** — a raw host buffer isn't IOMMU-granted to
the FPGA, so CL-mastered PCIM reads never complete (see `../../../../results/fpga_silicon/MUZERO_CHESS_FWD_AFI.md`
§3b). This variant moves the ~12 MB weight buffer to **on-card DDR4**, the drivable home:

```
ocl (AXI4-Lite) --[reg slice]--> kernel s_axi_control
sh_cl_dma_pcis (host staging) --\
                                 >--[axi_crossbar 2->1, SASD]--> sh_ddr (DDR_PRESENT=1) --> on-card DDR4
kernel m_axi gmem (reads)     --/
```

Host stages the buffer into DDR over DMA-PCIS (`fpga_dma`), kernel reads it via m_axi — both reach DDR through
the crossbar. **No host IOMMU in the loop.** The forward kernel HLS (`forge_train_step_muzero`, mode 0) is
unchanged; only the master's destination moves PCIM→DDR. PCIM is tied off here.

Build inputs (committed): `design/cl_forge_muzero_ddr.sv` (CL top: OCL + kernel + crossbar + sh_ddr),
`design/*_defines.vh` + `cl_id_defines.vh`, `build/scripts/gen_ip.tcl` (OCL slice + `axi_crossbar_ddr` 2→1
SASD, 512b/64-addr/16-ID to match `cl_sh_ddr_axi`), `build/scripts/synth_cl_forge_muzero_ddr.tcl` (reads
`sh_ddr` from `$HDK_SHELL_DIR/design/sh_ddr/synth` + the two IP), `build/constraints/*.xdc`.

Build: `CL=cl_forge_muzero_ddr PKG_REL=tikitorch/hardware/hdk/cl_forge_muzero_ddr bash
scripts/run_muzero_afi_build.sh launch`. Drive: stage DDR via `fpga_dma_burst_write` + the proven OCL control
(bridge.cpp offsets), kick mode 0, sample Vccint → dynamic J/inference.
