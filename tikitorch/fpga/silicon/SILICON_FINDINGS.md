# Silicon bring-up: morph-LM training on F2

> Historical bring-up log, last updated 2026-06-15. AGFIs are recorded for
> provenance and may be private, deregistered, or account/region-specific. The
> small negative WNS values below mean the headline builds missed timing; the
> throughput measurements are preliminary rather than release guarantees.

## HEADLINE — measured N-way replication with about 93% efficiency

For this D128 build family, the whole multiplier-free training loop runs on F2
fabric and the build reports DSP=0. Each instance's logged observable record
matched its host reference. Replicating the kernel across the VU47P's three SLRs
increased aggregate throughput with about 93% replica efficiency; completion
times show that the instances ran in parallel, but scaling was not perfect:

| config | tok/s | scaling | WNS | AGFI |
|---|---|---|---|---|
| 1 kernel (rw32) | 13,261 | 1.00× | −0.030 | agfi-0f6afc0dc37c0ad76 |
| 1 kernel (rw64, 64 lanes) | 14,236 | — | −0.020 | agfi-050b219d4090fb8d6 |
| N=2 | 26,522 | 1.86× vs rw64 N=1 | −0.097 | agfi-04a1a35443210687e |
| **N=4 (SLR-local max)** | **53,001** | **3.72× vs rw64 N=1** | **−0.173** | **agfi-0c4cde06e2161099a** |
| N=8 extrapolation | ~106,000 | projected, not measured | — | needs URAM rebalance |

At N=4, all four logged weight checksums were
`14650122959732499763`; the aggregate rate was 53,001 tok/s and the observed
Vccint core rail was ~4 W. The arithmetic ratio 53,001 / 107,091 is 49.5%, but
the nanoGPT denominator uses a different model and optimizer, so this is not an
accelerator-speedup or GPU-parity comparison. Average N=4 per-replica throughput
was 13,250 tok/s, about 6.9% below the 14,236-token/s rw64 N=1 measurement.
Single-kernel evolution: v8 fused 3,827 → +on-chip-grad
(w32) 12,077 → +resident-W (rw32) 13,261 → +64-lane (rw64) 14,236.

### Replication recipe (what it took to scale past N=2)

1. **BRAM density — `TT_URAM_W`.** Resident weights + on-chip grads are 423
   BRAM_18K/kernel — 2 kernels overflow one SLR (708 > 672 tiles). `#pragma HLS
   bind_storage ... impl=URAM` (in `do_train_run`) moves them to URAM → BRAM
   190/kernel. The observable-record gate passed after the change. URAM and BRAM
   are both one-cycle memories here, but the aggregate measurements still show
   the per-replica reduction noted above.
2. **Inter-SLR interconnect.** One shared DMA crossbar + broadcast OCL feeding
   kernels in different SLRs makes SLL-crossing critical paths. Fixes, in order
   (each halved the violation: −2.2 → −1.75 → −1.147 → −0.576 → −0.173):
   - per-instance reset synchronizer (async-assert/sync-deassert) + `set_false_path`
     on its CLR pins (the high-fanout shell reset can't cross an SLR in 1 cycle);
   - DMA crossbar `CONFIG.STRATEGY {2}` (registered outputs);
   - a full-AXI4 register slice (`RS_DMA`) **and** an AXI4-Lite slice (`RS_OCL`)
     per instance, **excluded from the kernel pblock** so they float to the SLR
     boundary → crossbar/OCL↔kernel hops are reg-to-reg (Laguna), not combinational.
   - **Lesson:** putting the register slice INSIDE the kernel pblock made it WORSE
     (−2.42) — its backward `s_ready` then crossed the SLR. Float the slice.
3. **Floorplan.** Per-SLR pblocks pin each kernel SLR-local (K0→SLR0, K1→SLR1,
   K2/K3→SLR2 halves). OCL control is BROADCAST (one AP_START kicks all N) +
   address-muxed reads; the host (`morph_lm_silicon_n4.py`) stages window i at
   word offset i·0x80000, polls each instance's done (`tt_fpga_done`).

### The N=4 ceiling (why not N≥6 yet)

Binding resource is **per-SLR CL URAM**, not total. Per kernel ≈ 122 URAM
(58 resident + 64 staging); the CL pblock holds only **176 / 192 / ~320** URAM
in SLR0 / SLR1 / SLR2 (the shell reserves the rest). So SLR-local capacity =
1 + 1 + 2 = **N=4**. N=6 (2/SLR) fails `DRC UTLZ-1` (needs 244, has 176/192).
An N=8 experiment would need URAM/BRAM rebalancing + staging compaction (the 5 grad
regions are confirmed dead in the staging buffer — `morph_lm_silicon_n4.py` never
reads them — so the staging window can shrink ~0.4 MB once the layout is compacted).
No N=8 image was built or measured.

## FINAL RESULT — fully-tropical CE training runs entirely on F2 silicon (measured observable-record match)

AFI `cl_forge_morph_lm_v8_aligned` (`agfi-0433370d93cb58a78`): fused whole-run
trainer (mode 3), DSP=0, WNS −0.039 ns @250 MHz. The negative WNS means this
artifact missed timing and is not a timing-closed release image. One host kick
runs the entire 30-step training loop (fwd + balanced-LUT-CE loss + backward +
signSGD) on the fabric, tokens/targets/weights/LUT resident in the buffer:
- the silicon JSONL observable record was **byte-identical** to the host CSIM
  record (`wcrc 14650122959732499763`, acc 0→0.30, and sampled predictions);
  this does not compare every internal signal;
- **1.003 s wall for 30 steps (3,840 token-updates) = ~3,827 tok/s**, one kick,
  no per-position host round-trips;
- 3 W Vccint core → ~0.78 mJ/tok core-rail (NOT total card).

Two bugs had to be fixed to get CE working on silicon (both were CSIM-invisible —
the host scalar path masked them; only the silicon wide path tripped them):
1. **Dead exp2 ROM.** A `static const` ROM (`FT_CE_EXP2_Q16`) read all-zero on the
   synthesized AFI → CE grad = 0 → training froze. Fix: host-stage the 256-word
   table into the buffer (`OFF_CE_LUT`) and read it from there.
2. **Misaligned wide burst.** `OFF_CE_LUT` sat at `OFF_CFG+2` (not 16-word
   aligned); the kernel's 512-bit `burst_load` truncates the offset to `>>4`, so
   it read the LUT shifted by 2 words → wrong CE grad → wrong weights, *while the
   logged forward predictions still matched* (preds don't use the LUT). Fix: round
   `OFF_CE_LUT` up to the next 16. Isolated by a 1-step on-silicon run (preds
   matched golden, weight-crc didn't).

Also: per-position modes 0/1/2 are HOST-only (`#if !SYNTHESIS`) so only fused
mode 3 synthesizes → CL area 190k→90k LUT, WNS −0.346→−0.039 ns. The whole HDK
CL build (IP regen via `build/scripts/gen_ip.tcl`, `synth_cl_forge_morph_lm.tcl`)
is now committed under `tikitorch/hardware/hdk/cl_forge_morph_lm/`.

---

## Earlier milestone — perceptron path (AFI cl_forge_morph_lm_v6_bufram)

AFI `agfi-006127c2d7aaabb42` (v2 buffer-RAM CL: DMA-PCIS↔URAM↔kernel, DSP=0,
WNS −0.190 ns @250 MHz). This artifact also missed timing. Its exactness gate
compares the logged silicon JSONL observable fields with the host CSIM record
(same `buffer_init.bin`, same schedule), not all internal state.

## RESULT: the fully-tropical morph-LM kernel trains on silicon (measured observable-record match)

Running the perceptron loss (CFG `loss_kind=0`, `TT_SGD_THR=0`), 30 steps over a
128-token window, the silicon trajectory is **byte-identical** to the host golden
(`perc_silicon_traj.jsonl` == `perc_golden_traj.jsonl`, sha `b61482d3…`):
- genuine learning: accuracy 0.000 → 0.195 (peak 0.234), weights evolve each step;
- measured throughput: **3,840 tokens fwd+bwd in 1.03 s → ~3,732 tok/s** (signsgd <1%);
- measured core power: **3 W Vccint** (fpga-describe `-M`), constant idle↔active
  in this observation; the build reports DSP=0. Dividing that core rail by the
  measured rate gives ~0.80 mJ/token **core-rail**.
  NB: Vccint is the user-logic rail only — NOT total-card power; a board-to-board
  energy comparison vs the GPU's 72 W total needs external card instrumentation.

The archived probes matched the host reference for forward argmax, the sampled
`AD_HEAD`/`BS_STREAM`/`GRAD_LOGIT` writes, reverse-backward readouts, reported
gradient nonzero counts (G_EMBED/MH/MP/MB/W_HEAD), and the SignSGD record. This
is exactness of those observed fields, not a claim about every internal signal.

## ROOT-CAUSED BUG: the LUT-CE exp2 ROM reads as zero on this AFI

With the better-quality balanced LUT cross-entropy loss (`loss_kind=1`), training
FREEZES on silicon: `GRAD_LOGIT` is all-zero after forward while the host golden is
dense (16384/16384 nz). Localized by staged silicon probes:
- `OFF_CFG` reads `[1,4]` correctly (loss path selected right);
- PRED matches golden ⇒ forward logits + argmax correct;
- the perceptron 2-hot grad (no ROM) is correct on silicon (`[pred]=+1,[tgt]=−1`);
- only the CE branch — which indexes `FT_CE_EXP2_Q16` (forge_ce_lut.hpp) — yields
  zero, *consistently exactly zero* (not jitter).
⇒ the exp2 ROM's init contents are not present in the synthesized/packaged BRAM on
this CL (a ROM-init/packaging issue, not the −0.190 ns timing). FIX for next build:
ensure the `FT_CE_EXP2_Q16` table synthesizes as an initialized ROM in the CL
(e.g. force a BRAM ROM with a `.mem`/inline init that survives encryption+place),
then CE training will run on silicon at the same throughput.

## Latent kernel bug (separate, see commit a6568f6): signsgd threshold truncates
to ap_int<16> on the synthesis path — harmless at thr≤32767, fix to ap_int<32>.
