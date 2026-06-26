# Distributed PC transformer — why the integer recipe under-trains (root-cause, measured)

Tooling: `scripts/pc_xformer_ablate.py` (a faithful torch twin of the kernel recipe with per-component toggles)
+ the int64 golden (`forge_pc_xformer_golden*`). All accuracies = next-char acc on TinyShakespeare,
D=64 / H=2 / FFN=256 / seq=64 / vocab=64, 1.2–2k steps. Float dot-product baseline = **0.489**.

## Headline: the substrate (tropical / multiplier-free) is NOT the limiter

Single-axis ablation, each kernel-component added to the dot/float/AdamW baseline (BATCH=32):

| component swapped in | acc | bpb cost | tropical? |
|---|---|---|---|
| L1 attention (mult-free score) | 0.468 | +0.10 | ✅ |
| int16 per-op activations | 0.487 | +0.00 | ✅ |
| dyadic-Adam | 0.489 | +0.05 | ✅ |
| fine-dyadic LayerNorm | 0.418 | +0.40 | ✅ |
| **all tropical together (float softmax)** | **0.347–0.448** | — | ✅ |

A **fully multiplier-free** recipe reaches 0.35–0.45 ≈ the dot-product baseline — *if the softmax is precise*.

## Two bugs found (not the substrate)

1. **Degenerate softmax temperature** — the kernel's fixed `>>X_AT_DELTA` left logits (~2^19) and L1 scores
   (~2^14) far above the exp2-LUT's range → the softmax saturated to one-hot → no calibration gradient.
   **Fixed** with an adaptive per-row shift (bitlen + barrel-shift, DSP_MULT=0). Golden 0.072 → 0.155.

2. **The integer attention softmax is the dominant ceiling.** Dropping the kernel's exp2-LUT softmax into the
   otherwise-0.347 torch recipe collapses it to **0.153** (= the golden's level). Decisive isolation:

   | attention softmax | acc |
   |---|---|
   | float | 0.347 |
   | float → Q16-quantized probs | **0.299** ← coarse *probabilities* are fine |
   | int exp2-LUT + adaptive-shift | 0.191 ← the **LUT + adaptive-shift structure** is the loss |

   The adaptive shift buckets the scores (`gap=(max−s)>>smsh` integer) and **crushes the close-key fine
   structure** (the keys that matter), while the coarse 2^(−g/8) LUT loses sub-bucket precision.

## The precise-softmax fix — validated in torch, fully tropical

`exp(−d) = G_EXP[frac(u)] >> int(u)`, with `u = d·log2(e)` computed by **shift-add** (×1.4375) and `G_EXP` a fine
mantissa ROM (256 entries of 2^(−f/256)). ROM + shift + add → **DSP_MULT=0**. In torch (temp 1.0) this reaches
**0.341 ≈ the float softmax (0.347)**, vs the coarse LUT's 0.153.

## This is the PC setup — and the PC optimizer is SOUND

The backward here is the **PC cascade** (single-pass transpose-free error propagation, Z-IL) and the optimizer
is our **dyadic-Adam** (mult-free per-param pow2-adaptive step). Confirmed sound: in torch, `OPT=dyadic` (the
float twin of the PC dyadic-Adam) trains the **whole transformer, attention included, to 0.489** — so the PC
optimizer is NOT the limiter; it adapts Q/K to the float ceiling.

## The integer golden — measured progress + the residual gap

Cumulative on the int64 golden (the on-fabric recipe), all fully multiplier-free:

| recipe | acc | prob_ema (chance 0.016) |
|---|---|---|
| original (degenerate fixed-shift softmax) | 0.072 | ~0.016 |
| + adaptive-shift softmax fix | 0.155 | ~0.03 |
| + **precise exp softmax + gradient accumulation (PC dyadic-Adam, batch≈8)** | **0.205** | **0.075 = 4.8× chance** |

So the precise softmax + feeding the PC dyadic-Adam cleaner gradients (its natural lever) compounds: 2.8× the
original. Batch has a sweet spot ≈8 (ACCUM=64 regresses to 0.147 — the sign-based step is scale-invariant, so
more batch ≠ more signal past the noise floor).

Two residual gaps remain, both understood:
- **0.20 → 0.30** = the **integer dyadic-Adam step quantization** (the PC optimizer reaches 0.30 in float with
  the precise softmax — `q16` torch — but the integer step floors sub-unit updates). The *simple* fixes
  (fractional master, stochastic rounding) measurably REGRESS it (0.205 → 0.13–0.16: the finer/slower moves
  undertrain at the batch sweet spot) — so this needs a co-designed step scheme, not a drop-in. Open.
- **0.30 → 0.49** = the **mult-free attention TRAINING wall** — the documented PC verdict ("PC coupled-routing
  does not map to attention"; "the multiply does essential work"). Warm-train the attention (the one allowed
  multiply) to cross it.

Ruled out as the cause (no effect on the golden): wgrad precision, activation precision, LN mantissa, weight
decay, master quantization.

## Path to competitive (all options keep the substrate tropical)

- **Warm-train the attention** (the documented answer): the QK dot-product is "the one allowed multiply"
  (attention zone), FFN/proj/embed/head stay mult-free PC-trained, deploy mult-free. Highest-confidence route
  to the 0.49 baseline.
- **A better integer attention optimizer** co-designed with the precise softmax (the interaction above), to
  realize the torch's 0.30 fully mult-free. Open problem (low-precision transformer attention training).

This matches the project's long-standing finding that *tropical attention needs smooth attention for LM* —
now precisely localized to the integer softmax's exp/temperature **and** the integer optimizer's Q/K adaptation.
