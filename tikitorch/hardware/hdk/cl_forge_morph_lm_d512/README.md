# cl_forge_morph_lm_d512 — F2 HDK Custom Logic package (D=512, A3 reduced-clock)

PARALLEL to `../cl_forge_morph_lm` (the DEPLOYED **D=128** sentinel AFI
`agfi-0433370d93cb58a78`, NK=4, which MUST stay byte-identical). This package
wraps the **D=512** build of `forge_train_step_morph_lm` (`-DML_D=512`) toward the
best achievable clock under the A3 approved plan (a REDUCED clock, ~125-166 MHz,
is accepted). Nothing here edits the D=128 package.

## Why a separate package (from the HDK width-prep report, wf_007d5672)

The D=512 training buffer is **12.25 MB** (vs 1.66 MB at D=128), so one kernel
needs a **16 MB** staging URAM — larger than a single SLR's ~11.25 MB URAM
budget. That forces, versus the D=128 package:

| delta | D=128 | D=512 | where |
|---|---|---|---|
| replication NK | 4 | **1** | `design/cl_forge_morph_lm_d512.sv` localparam |
| staging URAM | 2 MB | **16 MB** | `BUF_URAM` MEMORY_SIZE 16777216→134217728 bits, ADDR_WIDTH 15→18 |
| URAM line select | `[20:6]` | **`[23:6]`** | `BUF_URAM` addra/addrb |
| BRAM-ctl addr decode | `[20:0]` | **`[23:0]`** | BRAM_CTL_DMA/KRN s_axi_a{w,r}addr, bram_{a,b}_addr |
| `axi_bram_ctrl` MEM_DEPTH | 32768 | **262144** | `build/scripts/gen_ip.tcl` |
| `axi_crossbar_dma` | NUM_MI 4, M0x AW 21 | **NUM_MI 1, M00 AW 24** | `build/scripts/gen_ip.tcl` |
| floorplan | 4 SLR-local pblocks | **1 all-SLR pblock** | `build/constraints/cl_pnr_user_floorplan.xdc` |

The kernel *itself* also moves its out-proj / LM-head weight arrays (`mp_buf`
4 MB, `wh_buf` 0.25 MB) to URAM via `#if ML_D>=512` `BIND_STORAGE`/`ARRAY_PARTITION`
guards in `forge_train_step_morph_lm.cpp` (attacks the 11.425 ns csynth critical
path and frees BRAM). That URAM is internal to the kernel and adds to the per-SLR
URAM budget the floorplan must satisfy.

## Files renamed vs the D=128 package
- `design/cl_forge_morph_lm_d512.sv` — top module **`cl_forge_morph_lm_d512`**.
- `build/scripts/synth_cl_forge_morph_lm_d512.tcl` — CL synth driver (content is
  `${CL}`-generic; only the filename must be `synth_<CL>.tcl` for the HDK harness).
- `design/ip/` — the stale flat `.xci` were removed; **`gen_ip.tcl` regenerates
  all 5 IP** with the D=512 params (see `design/ip/REGENERATED_BY_GEN_IP.txt`).

## Open TODOs needing P&R feedback (marked in the files)
- `cl_pnr_user_floorplan.xdc`: the exact clock-region carve-out (shell-reserved
  URAM/logic regions per SLR) must come from the FIRST `aws_build_dcp_from_cl.py`
  utilization/timing report. It currently spans the full user area X0Y0:X7Y11.

## Build
Reduced-clock, D=512. Driven by `scripts/run_morph_lm_d512_afi.sh` (HLS the
kernel with `synth_morph_lm_d512.tcl`, scaffold this CL, gen IP, P&R, then
`create-fpga-image`). Confirm the achievable clock first with
`scripts/run_morph_lm_d512_recsynth.sh` and pass it as the AFI `CLOCK_RECIPE_A`.
