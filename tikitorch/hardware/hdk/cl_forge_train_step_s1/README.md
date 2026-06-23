# cl_forge_train_step_s1 — DDR variant CL for the S1 transformer-block TRAINER on F2

This CL puts the S1 integrated transformer trainer kernel `forge_train_step_s1` (forward + backward +
dyadic-Adam + the one-kick fused trainer, multiplier-free except the single attention soft p*V blend) on
real AWS F2 silicon, reading/writing its whole working set in **on-card DDR4**.

It is a **byte-for-byte clone of `cl_forge_muzero_ddr`** (the DDR CL that already runs on F2,
agfi-085dca07f374aa1cd: DDR P&R closed WNS +0.154 ns, AFI loads, 108 inf/s measured) with **exactly 3 edits**
to the SV plus 2 IP-name renames in the build scripts. The whole shell DDR path (sh_ddr + 2->1 SASD crossbar
+ DMA-PCIS staging + OCL control slice) is unchanged.

## Why DDR (not PCIM)

The sibling PCIM->host-DRAM path loads on F2 but cannot be DRIVEN for inference from EC2 userspace — a raw
host buffer isn't IOMMU-granted to the FPGA, so CL-mastered PCIM reads never complete (the Stage-3b finding).
On-card DDR is the drivable home for the S1 working set (`S1_BUFLEN` int32 = ~6.15 MB): the host **stages** it
into DDR over the shell DMA-PCIS (BAR4), the kernel **reads and writes** it via its m_axi gmem — both reach
DDR through a 2->1 AXI crossbar; no host IOMMU in the loop. PCIM is tied off.

```
ocl (AXI4-Lite, BAR0) --[s1_ocl_slice]--> kernel s_axi_control[12:0]   (buffer ptr + mode + a1/a2/a3)
sh_cl_dma_pcis (512b, host staging, BAR4) --\
                                            >--[axi_crossbar_ddr 2->1, SASD]--> sh_ddr (DDR_PRESENT=1) --> DDR4
kernel m_axi gmem (512b, reads AND writes)  --/                                  (BUFFER ptr reg = DDR addr 0)
```

DMA staging and kernel m_axi never overlap in time (host stages fully -> kicks -> kernel runs N steps resident
in DDR), so the SASD crossbar needs no concurrent arbitration and the master ID stays 16-bit.

## The 3 SV edits vs cl_forge_muzero_ddr.sv (everything else VERBATIM)

1. **Module name + include**: `cl_forge_muzero_ddr` -> `cl_forge_train_step_s1`; include
   `cl_forge_train_step_s1_defines.vh`. PCIM stays tied off (`unused_pcim_template.inc`).
2. **OCL slice instance**: `muzero_ocl_slice OCL_REG_SLC` -> `s1_ocl_slice OCL_REG_SLC` (renamed so it does
   not collide with the HDK's own `axi_register_slice_light`; all port connections unchanged).
3. **Kernel instance**: `forge_train_step_muzero u_kernel` -> `forge_train_step_s1 u_kernel`. The **13-bit**
   control address slice is KEPT (`.s_axi_control_AWADDR(ocl_aw_addr_pre[12:0])`,
   `.s_axi_control_ARADDR(ocl_ar_addr_pre[12:0])`) — the S1 control bundle carries buffer ptr (0x18/0x1c) +
   mode (0x24) + a1/a2/a3 (0x2c/0x34/0x3c), so it must NOT be narrowed to [4:0]. The standard HLS
   `m_axi_gmem_*` port set is identical (`bundle=gmem`, `max_widen_bitwidth=512` -> 512-bit, wires straight to
   the 512b crossbar with no width converter), so the `krn_*` hookup is unchanged.

Verified against the kernel's exported interface: `forge_train_step_s1` is
`extern "C" int forge_train_step_s1(int32_t* buffer, int mode, int a1, int a2, int a3)` with
`#pragma HLS INTERFACE m_axi port=buffer ... bundle=gmem max_widen_bitwidth=512` (=> `m_axi_gmem_*`) and
`#pragma HLS INTERFACE s_axilite ... bundle=control` for buffer/mode/a1/a2/a3/return (=> `s_axi_control_*`),
plus `ap_clk / ap_rst_n / interrupt` — the SAME port set the muzero kernel emits (muzero's args are named
arg1/arg2/arg3, S1's a1/a2/a3, but the SV connects only the bus, not the named register ports, and the
declaration order buffer->mode->a1->a2->a3->return preserves the register offset map).

## Build-script renames

- `build/scripts/gen_ip.tcl`: the OCL slice module is `s1_ocl_slice` (was `muzero_ocl_slice`). The
  `axi_crossbar_ddr` config is VERBATIM (NUM_SI=2, NUM_MI=1, AXI4, DATA_WIDTH 512, ADDR_WIDTH 64,
  ID_WIDTH 16, R_REGISTER 1, M00 base 0x0, ADDR_WIDTH 34). ID_WIDTH must stay 16 (matches cl_sh_ddr_axi).
- `build/scripts/synth_cl_forge_train_step_s1.tcl`: the OCL-slice IP refs are `s1_ocl_slice` (read_ip glob +
  synth_ip). The sh_ddr dependency-IP `read_ip` list (cl_ddr4, axi_register_slice[_light],
  cl_axi3_256b_reg_slice, cl_axi_clock_converter[_light]) + the `read_verilog -sv` glob + the OOC
  `synth_design` + header/footer are VERBATIM. **NEVER `read_verilog` sh_ddr.sv** (encrypted envelope error).
- `build/constraints/{cl_synth_user,cl_timing_user}.xdc`: EMPTY (muzero_ddr shipped them empty; add
  constraints only on a measured P&R miss — first mitigation is a slower user clock recipe in the build).

## Build + run

Build (clone of run_muzero_afi_build.sh with the S1 CL/KERNEL params; see results/fpga_training_rig/S1_AFI_PLAN.md §2):
```
CL=cl_forge_train_step_s1 \
PKG_REL=tikitorch/hardware/hdk/cl_forge_train_step_s1 \
KERNEL_TCL=synth_train_step_s1.tcl KERNEL_PROJ=forge_train_step_s1_proj \
bash scripts/run_s1_afi_build.sh launch
```
The build runner's launch guard checks `$PKG_REL/design/$CL.sv` is in the committed tree, so this package must
be committed before launch.

Run (clone of run_pc_train_f2.sh; stage W_fp + BOTH exp2 LUTs (S1_OFF_CE_LUT, S1_OFF_ATTN_LUT — BRAM static
init reads ZERO on AFI) over BAR4, kick mode 3 with a1=N_STEPS, a2=S1_OPT_TRAIN_ALL, poll AP_DONE, read back
LOSS / W_fp / LOGITS for the bit-exact check vs the CSIM golden). Register map (proven, muzero):
R_AP_CTRL=0x00, R_AP_RETURN=0x10, R_BUFFER_0=0x18, R_BUFFER_1=0x1c, R_MODE=0x24, R_ARG1=0x2c, R_ARG2=0x34,
R_ARG3=0x3c; AP_START=0x01, AP_DONE=0x02; DDR_BUF=0x0.

## First-AFI clock

Build with the **same clock recipe `cl_forge_muzero_ddr` used** (the build runner default — do NOT invent a
recipe). The kernel closes csynth at 3.197 ns (target 4.0 ns, +0.803 ns margin), so it meets 250 MHz with
margin; a slower recipe only adds slack. S1 is ~5x the muzero kernel's LUT/BRAM, so post-route timing is the
realistic risk — the §4 mitigation order is: lower the user clock to ~200 MHz, then pblocks (SLR0+SLR1
kernel / SLR2 shell), then S1_LANE 16->8.
