# A strictly multiplier-free tropical transformer LM generating on real F2 silicon (2026-07-01)

**THE RESULT:** a coherent, strictly multiplier-free (DSP=0, all shift-add) tropical transformer
character-LM, trained via kernel-faithful QAT, running INFERENCE on a real AWS F2 FPGA — generating
recognizable Shakespeare and measuring **held-out bpc = 2.4230 on silicon, BIT-EXACT to the g++
reference (2.4230) and beating bigram (3.47) by ~1.0**.

## On-silicon run (agfi-082aa26aac7495798, f2.6xlarge, kernel clk 62.5 MHz)
```
[infer] forge_train_step_s1 FORWARD on F2: S1_BUFLEN=4507664 (17.195 MB) D=128 H=4 FFN=512 L_SEQ=64 V=64(real=39) N_LAYERS=4
[infer] AWS_CLK_GEN: MMCM_LOCK A locked -> resets released; clk_extra_a1 (kernel clk) live
[infer] staged 17.195 MB into DDR via BAR4 in 0.014 s ; POST-STAGE READBACK: MATCH (0 mismatches / 32 probed)
[infer] GENERATE 400 chars (rolling ctx<=64, temp=128):
goood let your dealy half canatst torsul in him falt nor hath warh
worst indo your kho? romeos! gone!
for bore time franidius tits isue: and men,
and the field had fagante-siciuliath,old unto eventy
lack anticent it tog'd.
be ducatina:
comppenes:
set upidece i grantle, mercrtaine, '3 him own to the glace.
-warpen holsd did his done argloucester's yen ludinak-sewe camestiomalle.
if it you mai
[infer] generated 400 chars in 170.614 s (426.5 ms/char, forward-bound)
[infer] HELD-OUT bpc = 2.4230 | bigram = 3.47 | beats_bigram = YES
```
Recognizable Shakespeare: character-name-colon play format, real words ("let your", "nor hath",
"and the field had", "if it you"), and actual character names emerging (**romeos**, **gloucester's**).

## What made it work (the recipe, all DSP=0 / shift-add)
- **Weights:** 4-term dyadic CSD (sum of ±2^k) at per-matrix scale 64, packed into the existing int32
  slot (via `wmac4`); K=1 single-pow2 is UNSTABLE (2/3 seeds collapse), K=4 stabilizes.
- **GEMM:** per-matrix pow2 accumulator rescale (backbone >>6, head >>3).
- **LayerNorm:** 1/meanabs as a 4-term CSD reciprocal (Q24 integer floor-div + top-4 set bits) with
  guard-bit cross-terms — the binding-constraint fix that took it from bigram-class to coherent.
- **Activations** int16, **attention** L1 argmax-gather, **head** dyadic GEMM. Model D=128/L4, 2.42 bpc.
- Trained via QAT through a bit-exact integer sim mirror of the kernel (float QAT is insufficient
  because the argmax attention is discontinuous — a ±1 flips a gather).

## The bit-exactness grind (3 real kernel bugs found via per-stage diff vs the sim)
1. LN reciprocal: invalid CSD terms (erec=-64) not skipped -> `cv>>48` sign-extends to -1 for cv<0.
2. LN meanabs computed from UNSATURATED cv, not the saturated c[d] (the trained model uses saturated).
3. `Logits_all` stride declared [L_SEQ][max(D,V)] but fwd_gemm writes stride O=V -> rows 1+ misaligned
   when D>V. DORMANT at the old D=64=V, exposed at D=128 — the killer.

## FPGA build
csynth Fmax 342 MHz, DSP-clean. D=128/L4 FWD-ONLY FITS 1 SLR (fully placed+routed). Missed the
125 MHz kernel clock by -0.101 ns -> rebuilt at 62.5 MHz (CLOCK_RECIPE=A0) -> closed WNS +0.392,
hold met. AFI agfi-082aa26aac7495798. Runners: run_s1_afi_build.sh (KERNEL_TCL=synth_train_step_s1_kv2.tcl),
run_s1_infer_f2.sh (dedicated tiki-s1 box). Deploy model tropical_llm_d128L4_DEPLOY.pt.

## Honest caveats
- 426 ms/char (62.5 MHz, D=128/L4 forward, NOT throughput-optimized — this is a CORRECTNESS/coherence
  proof, not a tok/s result). Replication + a faster clock are the throughput levers.
- 2.42 bpc is small-transformer quality (recognizable, not GPT-fluent); larger D would improve it at
  a bigger build. The point proven: strictly multiplier-free CAN do coherent LM, bit-exact on silicon.
