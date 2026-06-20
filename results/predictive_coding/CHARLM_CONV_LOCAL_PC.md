# Rung 2: local async PC on a competitive dyadic-conv char-LM — a qualified result

**Date:** 2026-06-20 · **Code:** `scripts/pc_charlm_conv.py` (`DyadicConvPC1d`) ·
**Data:** tiny-shakespeare (char) · **Hardware:** EC2 g5.4xlarge (A10G).

## The question

Rung 1 (`CHARLM_LOCAL_PC.md`) showed **local** tropical PC — async error-relaxation, not
the cascade — matches backprop on a feedforward *context-MLP* char-LM, multiplier-free.
Rung 2 asks the same on a **competitive convolutional** architecture (the step toward
"local PC trains models worth scaling"): a causal dilated **dyadic-conv** char-LM, with
credit by frozen-routing error-relaxation routed through the conv stack.

Standard dyadic conv (not morphological): the error route `A_lᵀ = conv_transpose1d(E·relu_mask,
±2^k W)` is an **exact linear transpose**, weight grad = `conv1d_weight` on ±2^k operands;
the async loop (participation/stale/random-order) is the fabric form. All-position
prediction; sliding windows; warmup+cosine lr schedule.

## What's solid

- **The conv adjoint is exact** — the cheapest-first gradient gate is cosine **1.0000 on
  every layer** (float), confirming the causal-transpose alignment is correct.
- **`relax` == `relax_async` bit-identically** (every run): the async/fabric form — 50%
  participation + stale reads + random update order — converges to the *identical* result
  as sync on the conv. The parallelizable credit is **robust** on a conv too.
- **Local PC trains the competitive conv** well above the bigram floor (3.58).

## The honest result (converged, depth-5, RF-125, full data, best-val)

| arm | credit | val bpc |
|---|---|---|
| bp_float | float backprop | 2.511 |
| **bp_dyadic** | backprop, ±2^k forward, **real-multiply backward** | **2.451** |
| relax (sync local PC) | local, **fully multiplier-free** (operand-quantized weight grad) | 2.680 |
| **relax_async** (async/fabric local PC) | neighbour-local, partial+stale+any-order, fully mult-free | **2.680** |
| *bigram floor* | — | 3.58 |

**Verdict: a qualified win.** Local async PC trains the competitive conv (multiplier-free,
async-robust), but the **fully-multiplier-free** form trails the real-backward `bp_dyadic`
by **+0.23 bpc** at convergence. This is **not** an lr/undertraining artifact — the cosine
schedule converged the PC arm (it plateaus/oscillates at ~2.68 from ep35–44 while lr
decays, no further improvement and no degrade). The gap is the cost of the
**operand-quantized conv weight gradient**: the gradient *direction* is right
(gradcheck cosine ≈ 0.96 to backprop), but quantizing both conv-weight-grad operands to
±2^k is **lossier on conv than on the MLP** — where the same coarsening was ≈ free
(Rung 1 matched). On conv it costs ~0.23 bpc, concentrated where the per-layer grad
quantization is worst (the gate showed the top conv at cosine 0.61).

## Honest scope & the open lever

- This is a **credit-assignment** comparison on the same conv arch (all-position metric,
  single seed). `bp_dyadic` slightly beats `bp_float` here (a ±2^k-as-regularizer effect
  on this small data).
- The +0.23 is the price of **multiplier-free backward** (operand quantization), *not* the
  local/async form (sync `relax` == async `relax_async`; both = the same quantized
  gradient; the *form* is fine — the *quantization* is the cost).
- **Plausibly reducible (untested here):** magnitude-aware dyadic backward (the Rung-1
  MLP achieved "no acc cost" with magnitude-kept-within-2× dyadic grads), or not
  quantizing the worst layer (the readout / top conv), or a sign-product update there.
  This is the concrete next lever if closing the +0.23 matters.

## The journey (why this took several runs — honestly)

1. Non-overlapping windows **undersampled** (~34 steps/epoch) → everything undertrained;
   seq≈RF inflated the metric. Fixed: sliding windows (~7× more updates) + moderate RF.
2. A fixed PC lr kept missing the **narrow sweet spot** (8e-4 early-peaked@ep24 then
   degraded; 5e-4 too slow). Fixed: warmup + cosine-decay schedule (high early lr, decays
   to ~0). This **converged** the PC arm and exposed the real, modest +0.23 gap above.

## Reproduce

```
PYTHONPATH=. python scripts/pc_charlm_conv.py --gradcheck     # adjoint gate (cosine 1.0)
PC_TASK=convlm PC_ARGS="--depth 5 --seq 256 --stride 32 --width 128 --epochs 80 \
  --lr 1.5e-3 --pc-lr 7e-4 --pc-niter 30 --wd 2e-4" bash scripts/run_pc_scale_gpu.sh launch
```
