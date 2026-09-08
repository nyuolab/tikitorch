# cl_forge_muzero — F2 HDK Custom Logic package (MUST stay committed)

The AWS FPGA HDK Custom-Logic wrapper that takes the synthesized **dyadic
(zero-DSP) Tropical MuZero chess kernel** (`forge_train_step_muzero`) to an AFI.
Adapts the sibling `cl_forge_morph_lm` package. **These files are build INPUTS,
not regenerable artifacts** — commit them; never `.gitignore` them. (The morph-LM
set was lost once to an uncommitted session and cost a multi-hour
reconstruction.)

## The one architectural difference from cl_forge_morph_lm: PCIM, not URAM

morph-LM staged a ~2 MB int32 training buffer into an **on-CL URAM** over
DMA-PCIS and the kernel mastered that URAM. The chess MuZero buffer does **not
fit on-chip** — ~12 MB forward / ~6 MB train (the `repr.enc` WR1 matrix alone is
~4.7 MB). So:

| | cl_forge_morph_lm | **cl_forge_muzero** |
|---|---|---|
| buffer location | on-CL URAM (~2 MB) | **host DRAM (~12 MB)** |
| kernel `m_axi gmem` target | URAM via `axi_bram_ctrl` | **shell PCIM → host DRAM** |
| DMA-PCIS | used (host→URAM staging) | **unused** (tied off) |
| PCIM | unused (tied off) | **used** (the whole point) |
| replication | NK=4 (cheap, SLR-local) | **NK=1** (one kernel saturates a PCIM port) |
| IP needed | bram_ctrl_dma, bram_ctrl_krn, reg_slice_light | **reg_slice_light + reg_slice_pcim** |

The kernel's HLS interface is **unchanged** — one flat `int32_t* buffer` over
`m_axi bundle=gmem` (512-bit, `offset=slave`). Only the master's *destination*
changes (PCIM vs URAM). The host fills its DRAM buffer, passes the PCIM bus
address into the kernel via the OCL `buffer` offset register, writes
`mode`/`arg1..3`, kicks `AP_START`, polls `AP_DONE`, reads the scalar loss back.
The streaming layout (`forge_buffer_muzero*.hpp`: WR1 streamed in aligned 512-bit
beats, smaller matrices burst-loaded resident) was built for exactly this — it is
the same code path whether the far end is URAM or PCIM.

## Forward-first plan

The **forward** kernel is the AFI-worthy artifact today: chess-scale C-synth =
7.92 ms, 512-bit gmem, **DSP=0**, 250 MHz, CSIM bit-exact. The **train** kernel
is CSIM bit-exact (gridworld loss converges identically) but its 4096-way policy
head makes C-synth slow — a yellow flag for P&R feasibility, not a correctness
issue. **Build/validate the forward AFI first**; bring the train kernel to
silicon only after forward P&R closes timing on the F2 part. Swap which kernel
this CL wraps by changing the kernel name in `cl_forge_muzero.sv` (the kernel
instantiation) and the HLS `.tcl` the build runs — the CL wiring is identical
(same `m_axi gmem` / `s_axi_control` contract).

## Must-commit, and why each is load-bearing

- `design/cl_forge_muzero.sv` — the CL top: OCL AXI-Lite control → kernel
  s_axilite, kernel `m_axi gmem` → PCIM register slice → shell PCIM, DMA-PCIS
  tied off, 2026 F2 OCL port family (`ocl_cl_*`/`cl_ocl_*`). **The PCIM section is
  a sketch pending F2 reconciliation — see below.**
- `design/cl_forge_muzero_defines.vh`, `design/cl_id_defines.vh` — wiring +
  manifest IDs. The manifest generator parses LITERAL `CL_SH_ID0/1` defines;
  include-redirects break AFI registration (`MANIFEST_PARSE_FAILED`).
  IDs: `CL_SH_ID0=32'hF010_1D0F`, `CL_SH_ID1=32'h1D81_1D0F`.
- `build/scripts/gen_ip.tcl` — regenerates the two IP (`axi_register_slice_light`
  AXI4-Lite 32b; `axi_register_slice_pcim` AXI4 512b ID_WIDTH=6) from scratch:
  `vivado -mode batch -source gen_ip.tcl -tclargs <out_ip_dir>`.
- `build/scripts/synth_cl_forge_muzero.tcl` — the CL synth driver. Reads the flat
  CL+kernel verilog, then **`read_ip` + `synth_ip`** the two IP (the non-project
  OOC flow does NOT auto-synthesize IP — `synth_ip` is required or you get
  `module 'axi_register_slice_*' not found`).
- `build/constraints/cl_synth_user.xdc`, `cl_timing_user.xdc` — timing/placement
  (currently empty; populate if WNS closure needs it — a single small kernel may
  not).

NOT committed (gitignored, regenerable): `build/checkpoints/`, `*.dcp`,
`*.vivado.log`, `build/src_post_encryption/`, `to_aws/` tarballs, generated
`design/ip/*` (rebuilt by `gen_ip.tcl`).

## PCIM wiring — RESOLVED (probed against the F2 shell, 2026-06-15)

The PCIM signal contract was extracted from the real F2 shell on a build box
(`run_f2_hdk_probe.sh` → `manuscript/results/fpga_silicon/hdk_probe/`): aws-fpga **rc/2.3.2**,
`HDK_SHELL_DIR=.../hdk/common/shell_stable`. Ground truth that shaped the wiring:

- PCIM ports are `cl_sh_pcim_*` (CL→shell out) / `sh_cl_pcim_*` (in), **ID width
  16** (`cl_sh_pcim_awid/arid [15:0]`) — the kernel's 1-bit `m_axi` ID is
  zero-extended to 16 (`{15'h0, krn_awid}`); `gen_ip.tcl` sets `ID_WIDTH=16`.
- `cl_sh_pcim_awuser/aruser` are **55-bit** ([54:0]); the shell also exposes
  `cl_sh_pcim_wid [15:0]` + `wuser [63:0]` that an AXI4 master doesn't drive —
  all four are tied to 0 in the `.sv`.
- OCL is the renamed F2 family `ocl_cl_*`/`cl_ocl_*` (already used).
- `unused_dma_pcis_template.inc` **is** the correct DMA-PCIS tie-off include
  (confirmed present in the shell's `design/interfaces/`).

The `PCIM_REG_SLC` instance is now wired (not a sketch). **Still do a synth-only
smoke first** (fast) before the ~3 h P&R, to catch any `axi_register_slice` IP
port-list mismatch. Everything else (IDs, status, reset sync, OCL slice, kernel
control) is the proven morph-LM wiring.

## Rebuild recipe (fresh F2 box)

```bash
source /opt/Xilinx/Vivado/2024.1/settings64.sh
sudo apt-get install -y git-lfs && (cd ~/aws-fpga && git lfs pull)   # shell files
cd ~/aws-fpga && source hdk_setup.sh                                 # do NOT pipe (subshell eats exports)
# 1. HLS the kernel -> verilog  (forward first; train later)
cd <repo>/tikitorch/hardware/hls/forge_kernels && vitis_hls -f synth_muzero_chess.tcl
# 2. reconcile the PCIM section in cl_forge_muzero.sv against the HDK cl_ports.vh (see above)
# 3. scaffold CL from CL_TEMPLATE, overlay design/*, flatten kernel *.v into design/, gen IP
vivado -mode batch -source build/scripts/gen_ip.tcl -tclargs <CL>/design/ip
# 4. build (default clocks)
export CL_DIR=<CL_path>
cd $CL_DIR/build/scripts && ./aws_build_dcp_from_cl.py --cl cl_forge_muzero --no-encrypt
```

Full walkthrough + the silicon-only bug classes: `../../BUILD_GUIDE.md` and
`../../../fpga/silicon/SILICON_FINDINGS.md`. Sibling reference AFI (proven flow):
`cl_forge_morph_lm_v8_aligned` (`agfi-0433370d93cb58a78`).
