# Silicon bring-up: morph-LM training on F2

## FINAL RESULT — fully-tropical CE training runs ENTIRELY on F2 silicon (measured, bit-exact)

AFI `cl_forge_morph_lm_v8_aligned` (`agfi-0433370d93cb58a78`): fused whole-run
trainer (mode 3), DSP=0, WNS −0.039 ns @250 MHz. One host kick runs the entire
30-step training loop (fwd + balanced-LUT-CE loss + backward + signSGD) on the
fabric, tokens/targets/weights/LUT resident in the buffer:
- silicon JSONL **byte-identical** to the host CSIM golden
  (`wcrc 14650122959732499763`, acc 0→0.30 quality-grade CE);
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
   forward/preds stayed bit-exact* (preds don't use the LUT). Fix: round
   `OFF_CE_LUT` up to the next 16. Isolated by a 1-step on-silicon run (preds
   matched golden, weight-crc didn't).

Also: per-position modes 0/1/2 are HOST-only (`#if !SYNTHESIS`) so only fused
mode 3 synthesizes → CL area 190k→90k LUT, WNS −0.346→−0.039 ns. The whole HDK
CL build (IP regen via `build/scripts/gen_ip.tcl`, `synth_cl_forge_morph_lm.tcl`)
is now committed under `tikitorch/hardware/hdk/cl_forge_morph_lm/`.

---

## Earlier milestone — perceptron path (AFI cl_forge_morph_lm_v6_bufram)

AFI `agfi-006127c2d7aaabb42` (v2 buffer-RAM CL: DMA-PCIS↔URAM↔kernel, DSP=0,
WNS −0.190 ns @250 MHz). Bit-exact gate = silicon JSONL must equal the host
CSIM golden (same `buffer_init.bin`, same schedule).

## RESULT: the fully-tropical morph-LM kernel TRAINS on silicon (measured, bit-exact)

Running the perceptron loss (CFG `loss_kind=0`, `TT_SGD_THR=0`), 30 steps over a
128-token window, the silicon trajectory is **byte-identical** to the host golden
(`perc_silicon_traj.jsonl` == `perc_golden_traj.jsonl`, sha `b61482d3…`):
- genuine learning: accuracy 0.000 → 0.195 (peak 0.234), weights evolve each step;
- measured throughput: **3,840 tokens fwd+bwd in 1.03 s → ~3,732 tok/s** (signsgd <1%);
- measured core power: **3 W Vccint** (fpga-describe `-M`), constant idle↔active
  (the DSP=0 design barely moves the core rail) → ~0.80 mJ/token **core-rail**.
  NB: Vccint is the user-logic rail only — NOT total-card power; a board-to-board
  energy comparison vs the GPU's 72 W total needs external card instrumentation.

The read/forward/backward/update datapath is bit-exact to host: forward argmax
(PRED matches golden), `AD_HEAD`/`BS_STREAM`/`GRAD_LOGIT` writes, the full reverse
backward, grad propagation (G_EMBED/MH/MP/MB/W_HEAD nz all match host exactly),
and signSGD all verified on silicon.

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
