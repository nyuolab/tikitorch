# cl_forge_morph_lm — F2 HDK Custom Logic package (MUST stay committed)

The AWS FPGA HDK Custom-Logic wrapper that takes the synthesized
`forge_train_step_morph_lm` kernel to an AFI. **These files are build INPUTS, not
regenerable artifacts** — they were lost once (a prior session never committed
them) and cost a multi-hour from-scratch reconstruction. Commit them; never
`.gitignore` them.

Must-commit, and why each is load-bearing:

- `design/cl_forge_morph_lm.sv` — the CL top: OCL AXI-Lite control → kernel
  s_axilite, the kernel `m_axi` ↔ on-CL URAM buffer (DMA-PCIS dual-port), and the
  2026 F2 shell port names (`ocl_cl_*`/`cl_ocl_*` — the renamed family; May-era
  names synthesize with a *floating* control plane).
- `design/cl_forge_morph_lm_defines.vh`, `design/cl_id_defines.vh` — wiring +
  manifest IDs. The manifest generator parses LITERAL `CL_SH_ID0/1` defines;
  include-redirects break AFI registration (`MANIFEST_PARSE_FAILED`).
- `design/ip/*.xci` — the three IP configs (`axi_bram_ctrl_dma`,
  `axi_bram_ctrl_krn`, `axi_register_slice_light`). Commit BOTH the `.xci` and the
  generator below so they rebuild deterministically.
- `build/scripts/gen_ip.tcl` — regenerates the three IP from scratch
  (`vivado -mode batch -source gen_ip.tcl -tclargs <out_ip_dir>`).
- `build/scripts/synth_cl_forge_morph_lm.tcl` — the CL synth driver. Reads the
  flat (encryption-flattened) CL+kernel verilog, then **`read_ip` + `synth_ip`**
  the three IP (the non-project OOC flow does NOT auto-synthesize IP — `synth_ip`
  is required or you get `module 'axi_bram_ctrl_*' not found`).
- `build/constraints/cl_synth_user.xdc`, `cl_timing_user.xdc` — timing/placement;
  WNS closure depends on these.

NOT committed (gitignored): `build/checkpoints/`, `*.dcp`, `*.vivado.log`,
`build/src_post_encryption/`, `to_aws/` tarballs — all regenerable artifacts.

## Rebuild recipe (fresh F2 box)

```bash
source /opt/Xilinx/Vivado/2024.1/settings64.sh
sudo apt-get install -y git-lfs && (cd ~/aws-fpga && git lfs pull)   # shell files
cd ~/aws-fpga && source hdk_setup.sh                                 # do NOT pipe (subshell eats exports)
# 1. HLS the kernel -> verilog
cd <repo>/tikitorch/hardware/hls/forge_kernels && vitis_hls -f synth_morph_lm.tcl
# 2. scaffold CL from CL_TEMPLATE, overlay design/*, flatten kernel *.v into design/, gen IP
vivado -mode batch -source build/scripts/gen_ip.tcl -tclargs <CL>/design/ip
# 3. build (default clocks; custom recipes need --aws_clk_gen, which we don't use)
export CL_DIR=<CL_path>
cd $CL_DIR/build/scripts && ./aws_build_dcp_from_cl.py --cl cl_forge_morph_lm --no-encrypt
```

Full walkthrough + the silicon-only bug classes: `../../BUILD_GUIDE.md` (§2, §4)
and `../../../fpga/silicon/SILICON_FINDINGS.md`. Validated AFI:
`cl_forge_morph_lm_v8_aligned` (`agfi-0433370d93cb58a78`) — fused mode 3, DSP=0,
WNS −0.039 ns, fully-tropical CE training bit-exact on silicon.
