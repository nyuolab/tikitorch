# Deploying the tropical LLM on F2 (on-chip, D=64) — steps + handoff

**Date:** 2026-06-27. Path chosen: **on-chip now** — deploy the D=64 on-chip-fittable tropical LLM for inference
on real F2, fully multiplier-free.

## The model (ready)
`results/lm_shakespeare/tropical_llm_d64.pt` — tropical transformer, **D=64 / H=2 / DH=32 / FFN=256 / T=64 /
L=2 / V=39** (108k params), full validated recipe (max-plus argmax-gather attention + dyadic ±2^k linears +
positional embeddings, τ-annealed warm-train → hard deploy). **HARD (fully multiplier-free) bpc = 2.548 ≈ soft
2.554** (clean deploy), beats bigram 3.472 by ~0.92 bits, generates char-Shakespeare. Dims match the s1 routable
config (D=64; only N_LAYERS bumped 1→2).

## The kernel (ready)
The s1 forward, built fully multiplier-free via compile flags:
`-DS1_ATTN_HARD=1` (argmax-gather attention, no soft-blend) `-DS1_HEAD_MODE=1` (dyadic-GEMM head)
`-DS1_POS_EMBED=1` (positional embeddings) `-DS1_N_LAYERS=2`. Every op DSP=0. **csynth-CONFIRMED (forward-only, S1_FWD_ONLY=1): DSP=0, timing 2.920 ns <= 4.0 (closes at 250 MHz), ~15% LUT /
~21% BRAM of one SLR, URAM=0 -- fits one SLR, fast. The full-kernel csynth's 13.1ns/384-DSP/155%-SLR was the
TRAINING path (bwd/opt), gated out by S1_FWD_ONLY.** (`synth_train_step_s1_deploy.tcl`; report committed.)

## Remaining steps to "trained LLM generating on F2"
1. **csim deploy gate (no F2):** export the trained weights to the s1 fixed-point buffer format and run the s1
   kernel g++ forward (mode-0) over held-out windows; confirm bpc matches the torch model within the int-deploy
   margin (~2.5–2.7, per the int16 gate). This is the bit-exact-ish "the kernel implements the model" check.
   - **Weight export** (to build): torch `tropical_llm_d64.pt` → s1 buffer. Map: Wq/Wk/Wv/Wo/W_up/W_down →
     dyadic CODE regions (the kernel's `s1_encode` of each weight's ±2^k, matching `dyad()`); EMBED/W_HEAD/POS →
     int16 value regions; LN γ/β; CE/attn exp2 LUTs (host-staged). The host helpers (`h_s1_encode_fine`, the
     `S1_OPT_REGIONS` table) already exist in `forge_train_step_s1_f2.c`.
   - **Inference loop** (to build): in the driver, stage the exported weights, then per eval window kick
     `MODE_FWD` (mode-0), read back `S1_OFF_LOGITS`, compute bpc / sample for generation. (The driver already has
     staging + the mode-0 kick + logit readback; needs the eval/generate loop instead of the training loop.)
2. **AFI build (gated, USER runs F2):** rebuild `cl_forge_train_step_s1` with the deploy cflags above (via
   `run_s1_afi_build.sh`, the D=64 config that routed at L=1 — L=2 fit confirmed by the deploy csynth). ~5h P&R.
   The mult-free flags only *remove* the attn soft-blend DSPs, so it fits no worse than the proven L=1 trainer AFI.
   Fallback if L=2 P&R is tight: `-DS1_N_LAYERS=1` (the proven-routable config; model L=1 hard 2.751).
3. **Run on F2:** load the AFI, stage the exported weights, run the inference loop → on-silicon held-out bpc +
   generated text, fully multiplier-free (DSP=0). This is the tropical LLM running on real silicon.

## Status
- Model trained + validated (2.548 hard, clean, beats bigram). ✅
- Mult-free forward-only deploy kernel csynth-CONFIRMED: DSP=0, 2.920ns, fits one SLR. ✅
- Weight-export + inference loop: the one integration to build (csim gate) before the AFI.
- AFI + on-silicon run: the gated build (user runs F2).


## ★ UPDATE (kernel-faithful deploy is VIABLE)
The first deploy gate caught a train/deploy numeric MISMATCH: the std-LN-trained model deploys at **4.05 bpc**
(beats unigram, not bigram) through the s1 kernel, because the kernel uses mean-abs-deviation LayerNorm + pow2
gamma + operand-quant (not std-LN + float). NOT a tuning miss (~700 configs). **FIX CONFIRMED:** training
kernel-faithful (MAD-LN + pow2-gamma + dyadic + int16, `scripts/tropical_llm_kernelfaithful.py`) →
**HARD 2.650 bpc, beats bigram by 0.82** (only 0.09 behind std-LN). The on-chip kernel CAN host a useful LM.
Deployable model: `results/lm_shakespeare/tropical_llm_d64_faithful.pt` (D=64/H2/DH32/FFN256/T64/L2, MAD-LN).
Also: the gate found+fixed a kernel bug (S1_OFF_LNF_G offset at N_LAYERS>=2).

### Remaining (precise)
1. **LN bit-match + re-gate:** make the torch MAD-LN bit-identical to the kernel's 2-bit-mantissa fine-dyadic LN,
   retrain, re-export (`tropical_llm_export_s1.py`, extend to MADLN keys), re-run `s1_deploy_eval.cpp` → confirm
   the real s1 forward reproduces ~2.65. (The export+harness are bit-exact-verified; this closes the last csim gap.)
2. **AFI (USER runs F2):** `forge_train_step_s1 -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1
   -DS1_N_LAYERS=2` via `run_s1_afi_build.sh` (csynth-confirmed: DSP=0, 2.92ns, fits one SLR).
3. **Inference driver:** stage exported weights + mode-0 forward + logits → bpc/generate (driver has staging+mode-0).
