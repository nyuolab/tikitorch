# Silicon bring-up: morph-LM training on F2 (AFI cl_forge_morph_lm_v6_bufram)

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
