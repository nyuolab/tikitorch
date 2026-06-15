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

## Pending (Stage 3)
On-silicon **Vccint power** during kernel execution → fills the energy axis of
`results/board_scaling/CHESS_MUZERO_COMPARISON.md`. Needs an F2 instance + a host
driver that provides a PCIM-addressable buffer (the kernel masters host DRAM).
