# Forward MuZero chess kernel — AFI build record (silicon)

The dyadic, zero-DSP Tropical **MuZero chess FORWARD** kernel taken to a routed,
registered AWS F2 AFI. This is the first MuZero-family kernel placed-and-routed on
silicon (the morph-LM sibling is the only other FORGE AFI).

## Result — P&R timing CLOSED, first pass

| item | value |
|---|---|
| CL | `cl_forge_muzero` (kernel `forge_train_step_muzero`, mode=forward inference) |
| part | xcvu47p-fsvh2892-2L-e (F2), 250 MHz / 4.00 ns target |
| **post-route WNS** | **+0.123 ns (MET)** — "SUCCESS: Design has no negative slack path" |
| DSP | **0** (multiply-free datapath; dyadic shift+add) |
| LUT | 61,542 (14% of one SLR — comfortable single-SLR fit) |
| data path | kernel `m_axi gmem` (512-bit) → `axi_register_slice_pcim` → shell **PCIM → host DRAM** |
| Vivado build time | ~20 min (synth+opt+place+phys_opt+route); ~29 min incl. HLS+scaffold |
| reproducibility | timing closed identically across two independent builds (deterministic) |
| **AFI** | `afi-0ae3ac657a1fbbc7e` |
| **AGFI** | `agfi-0b2c228a17e516e0a` |
| name | `cl_forge_muzero_chess_fwd_v1` |
| reports | `afibuild/reports/cl_forge_muzero.2026_06_15-112652.post_route_timing.rpt` |

## What this validates
- The **PCIM-to-host-DRAM** architecture (our one departure from morph-LM's
  on-CL-URAM design) is silicon-grade: it elaborated with 0 errors and placed +
  routed clean against the real F2 shell. The probe-derived contract (PCIM ID
  width 16, 55-bit aw/ar user, `wid`/`wuser` tie-offs) was correct.
- Chess-scale **zero-DSP** forward inference is a routed reality, not just C-synth.
- First-pass closure with +0.123 ns beats the morph-LM precedent (which needed
  retries to reach −0.04…−0.17 ns) — the forward kernel's 2.920 ns C-synth
  estimate left enough margin to survive route.

## Build provenance (reproduce)
- `scripts/run_muzero_afi_build.sh` (committed): HLS → scaffold CL from the F2
  `CL_TEMPLATE` → gen-IP → `aws_build_dcp_from_cl.py` → `Developer_CL.tar`.
- `scripts/run_f2_hdk_probe.sh`: extracted the F2 shell PCIM contract (rc/2.3.2).
- Flow bring-up took 5 cheap (~$0.10 each) fail-fast smokes: $HOME, git-lfs,
  CL_TEMPLATE name, dangling symlinks, `mkdir design/ip`, and the synth driver
  (header path / `src_post_enc_dir` / `design/ip` / the `synth_design` call).

## Stage 3 — ON SILICON (measured on f2.6xlarge, 2026-06-15)

The AFI **loads on a real F2** and the PCIM path is healthy:

| item | value |
|---|---|
| load | `agfi-0b2c228a17e516e0a` → slot 0, state **loaded / ok** (device 0xf010 / vendor 0x1d0f) |
| PCIM health | **0 protocol errors** across 5 samples (`pcim-axi-protocol-error=0`, `pcim-range-error=0`, no timeouts) — the PCIM-to-host-DRAM wiring is clean on hardware |
| **Vccint power** | **8–10 W** (last/avg 8–9 W, max 10 W) — **loaded, kernel IDLE** |
| tool | `fpga-describe-local-image -M`, 5 samples; runner `scripts/run_muzero_afi_f2_power.sh` |

**Honest reading:** 8–10 W is the **loaded/idle** Vccint rail = **shell + DSP=0 CL static** (shell-dominated;
our CL is 14% of one SLR with no DSPs). It is **NOT** inference power, and you must **not** multiply it by the
7.92 ms latency to get J/inference — the dynamic draw of the kernel is unmeasured. What is proven on silicon:
the routed DSP=0 design **configures and the PCIM interface works**.

## Stage 3b — dynamic power ATTEMPTED; blocked by an EC2 PCIM-to-host IOMMU wall (measured 2026-06-15)

Built and ran a real PCIM host driver on F2 (`muzero_pcim_power.c` + `run_muzero_afi_f2_dynpower.sh`):
1 GB hugepage (contiguous) → `/proc/self/pagemap` guest-physical (`0x180000000`) → kernel `BUFFER_0/1`
register → kick `AP_START` (mode 0). Buffer = **MZ_BUFFER_WORDS = 3,132,144 = 11.95 MB**.

**Outcome — definitive diagnosis, not a clean number:**
- The CL's PCIM master **WORKS**: `pcim-read-count` went 0 → 8 with **0 range errors, 0 protocol errors** —
  the kernel issued clean PCIM read transactions to the programmed host guest-physical address. Wiring +
  addressing + control-register poke are all correct on silicon.
- But the kernel **STALLED** (warm kick: no `AP_DONE` in 5 s; rc=6; Vccint stayed at idle 7 W during the
  "loop" → no inference ran). The 8 reads were *issued* but never *completed*.
- **Root cause:** on EC2/Nitro the FPGA's PCIM accesses to host memory must be **IOMMU-mapped/granted to the
  FPGA's DMA domain**. A raw `mmap` hugepage is visible to the host but **not** granted to the FPGA, so the
  reads are issued (shell range-check passes → counted) but silently dropped downstream (no response → m_axi
  stalls). This is an EC2 host-memory-access requirement, **not** a CL/AFI defect.

**What this still proves on silicon:** the PCIM-to-host datapath is electrically + protocol correct (clean
issued transactions). The only missing piece is the host-side DMA mapping.

**Paths to a real J/inference (a genuine fork — not yet chosen):**
- **(A) IOMMU-map the buffer for the FPGA** (VFIO, or allocate via the AWS XDMA/`fpga_dma` driver — whose
  buffers ARE IOMMU-mapped — and point the CL m_axi at that IOVA). Smaller experiment; uncertain whether the
  app-PF exposes this for a raw m_axi PCIM master.
- **(B) Rebuild the AFI to read on-card DDR** (`cl_sh_ddr`) instead of host PCIM — the architecturally-correct
  home for a 12 MB buffer that doesn't fit URAM. Host stages into card DDR (DMA-PCIS→DDR / `fpga_dma`), kernel
  reads DDR; no host IOMMU needed. ~½-day rebuild (rewire CL, new host-load path, rebuild AFI, new driver).
  **Recommended** if dynamic power is wanted — it's the drivable architecture and reuses proven host staging.

The loaded/idle Vccint (7–9 W) above stands; the inference-power figure remains **to-measure** pending one of
the above.
