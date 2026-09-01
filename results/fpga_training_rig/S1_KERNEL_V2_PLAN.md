# S1 kernel-v2 — the coherent strictly-multiplier-free (DSP=0) transformer LM

Goal: take the sim-proven kernel-v2 recipe (DSP=0, beats bigram, generates recognizable Shakespeare)
to real F2 silicon. The sim (scripts/tropical_llm_kernelsim.py) is bit-faithful to the kernel, so a
kernel that implements this recipe will reproduce the trained bpc on silicon.

## What changed vs the current kernel (forge_train_step_s1.cpp, S1_GEMM_QX=0)
The current kernel deploys at bigram-class (~3.56). Three shift-add changes (all DSP=0) take it to
~2.7 bpc (recognizable LM). Measured deltas (D=64): plain 3.56 → +these → 2.7-3.0.

1. **Weights → 4-term dyadic (CSD).** Each weight = round(w*SW) approximated as a sum of 4 signed
   powers of 2 (canonical-signed-digit), instead of one ±2^k. The GEMM does 4 shift-adds per weight
   (4 shifted copies of the int16 activation accumulated) instead of 1. DSP=0 (barrel shifts + adds).
   - Encoding: 4 code_t per weight (sign+exp+nz each), so the weight ROM is 4× larger.
   - SW (weight scale) raised to 64 per matrix (so weights aren't rounded to zero — the old SW=2
     zeroed ~95% of weights, the root cause of the bigram wall).

2. **Per-matrix pow2 accumulator rescale.** After each GEMM, right-shift the accumulator by
   log2(SW)=6 before sat16, so the big SW doesn't overflow the int16 residual path. One shift per
   output. DSP=0. (Without it the residual-branch scale explodes ~SW²/block — the confound that hid
   the recipe for ~20 experiments.)

3. **LayerNorm → 4-term CSD reciprocal.** Replace the coarse `fdyad` nq-mantissa reciprocal with:
   compute 1/meanabs, represent it as 4 signed powers of 2 (CSD), apply to the centered c as 4
   shift-adds: r = Σ_k (c shifted by e_k). DSP=0. THIS was the single binding constraint — with an
   accurate reciprocal, exact-recip→2.91 vs coarse-fdyad→3.57. K'=4 terms recovers it (2.99).
   - Reciprocal source: a LUT meanabs→(4 CSD exponents). meanabs is a bounded int (sum|c|>>6).
     Alternative: on-the-fly CSD via priority encoders (4 iterations of leading-bit detect + subtract).

Activations stay int16 (measured fine — exact-LN+int-act = 3.19, beats bigram). Attention stays
L1 argmax-gather (DSP=0). Head stays dyadic-GEMM (would also become 4-term CSD).

## Dimensions (deploy target = D=128, 4 layers)
D=128 (LOG2_D=7), H=4, DH=32, FFN=512, T=64, V=39/64, N_LAYERS=4. Chosen over D=256/L4 (≈ same
bpc within run noise, ~2.7-2.8) because D=256 + 4-term CSD is a much larger AFI (4× weight ROM +
16× GEMM width) — likely multi-SLR / hard build. D=128 is the buildable sweet spot. Depth>4 is OUT
(D=256/L6 plateaued at 4.2 — int16 residual / rescale compounds badly past 4 layers).

## Work items
- [ ] config: S1_D 64→128, S1_LOG2_D 6→7, S1_N_LAYERS→4, S1_FFN→512, S1_H→4; recompute all S1_OFF_*.
- [ ] fwd_gemm: 4-term CSD weight decode (4 codes/weight) + 4 shift-adds + per-matrix rescale shift.
- [ ] layernorm_tok: 4-term CSD reciprocal (LUT or on-the-fly) replacing the fdyad path.
- [ ] head: 4-term CSD dyadic GEMM (same as fwd_gemm).
- [ ] export (tropical_llm_export_s1_faithful.py): emit 4 CSD codes/weight + rescale shifts + recip LUT.
- [ ] csim: bit-exact (or <0.05 bpc) vs scripts/tropical_llm_kernelsim.py on the deploy model.
- [ ] AFI build (own runner) → register → run on a DEDICATED f2 instance (NEVER tiki-f2-silicon).
- [ ] autoregressive generate on silicon.

## Risk / open
- Area: 4-term CSD GEMM at D=128 — confirm csynth DSP_MULT=0 + fits 1 SLR + timing. If tight, the
  CSD GEMM can be serialized (4 passes over the 1-term engine) to trade latency for area.
- LN reciprocal LUT size (meanabs range) vs on-the-fly priority-encoder CSD (smaller, more logic).
- Model quality is ~2.7 (recognizable, not GPT-coherent); lower needs bigger model = bigger build.
