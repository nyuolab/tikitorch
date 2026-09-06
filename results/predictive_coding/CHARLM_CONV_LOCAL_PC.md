# Rung 2: local async PC on a competitive dyadic-conv char-LM — a qualified result

**Date:** 2026-06-20 · **Code:** `scripts/pc_charlm_conv.py` (`DyadicConvPC1d`) ·
**Data:** tiny-shakespeare (char) · **Hardware:** EC2 g5.4xlarge (A10G).

## The question

Rung 1 (`CHARLM_LOCAL_PC.md`) showed **local** tropical PC — async error-relaxation, not
the cascade — matches backprop on a feedforward *context-MLP* char-LM with a
dyadic internal weight/activation operand.
Rung 2 asks the same on a **competitive convolutional** architecture (the step toward
"local PC trains models worth scaling"): a causal dilated **dyadic-conv** char-LM, with
credit by frozen-routing error-relaxation routed through the conv stack.

Standard dyadic conv (not morphological): the error route `A_lᵀ = conv_transpose1d(E·relu_mask,
±2^k W)` is an **exact linear transpose**, and the internal weight-gradient kernel uses
`conv1d_weight` with a dyadic activation and a real error operand; the async loop
(participation/stale/random-order) is the candidate fabric form. Cross-entropy, Adam,
the learning-rate schedule, and data handling remain ordinary boundaries. All-position
prediction; sliding windows.

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
| **relax_async** (async/fabric local PC) | neighbour-local, partial+stale+any-order, single-dyadic-operand weight gradient | **2.617** |
| relax (sync) — identical to async | local, single-dyadic-operand weight gradient | 2.617 |
| *bigram floor* | — | 3.58 |

**Verdict: a qualified win — local async PC with a dyadic activation operand trains
the competitive conv to within ~0.1 bpc of float backprop** (2.617 vs bp_float 2.511), and the async/fabric
form is robust (`relax_async` == `relax` bit-identically — partial participation + stale +
any-order converge to the same result as sync). The residual **+0.17 vs `bp_dyadic`** (2.451)
is the measured cost of using **single-operand quantization in the internal weight-gradient
kernel**: `bp_dyadic` uses a ±2^k *forward* but an ordinary float backward, while
`relax_async` uses `±2^k activation × real error` in that kernel. This operator can
map to shifts; it is not a claim that Adam or the whole training recipe is closed.
That operand quantization is ~free
on the MLP (Rung 1 matched) but costs ~0.17 bpc on conv.

**This was driven down with two principled fixes (not lr-guessing):**
1. **Single-operand quantization** (error real × ±2^k activation = a shift) instead of
   quantizing both operands — raised the per-layer weight-grad cosine to ~0.98 (top conv
   0.61 → 0.98). [Helped the gradient direction; on its own did not move training — the
   floor was a *bias*, not magnitude.]
2. **Unbiased stochastic dyadic rounding** (round ±2^k up/down with linear probability,
   E[q]=w) — averages out the deterministic-rounding *bias floor* over training: **2.680 →
   2.617** (closed ~30% of the original +0.23). The residual is inherent ±2^k weight-grad
   *direction* noise (a direction problem; lr/optimizer/schedule can't fix it — only finer
   quantization, e.g. 2-term dyadic, untested). The lr/convergence question was settled
   separately (the cosine schedule converged the PC arm cleanly — the residual is real, not
   undertraining).

## Honest scope & the open lever

- This is a **credit-assignment** comparison on the same conv arch (all-position metric,
  single seed). `bp_dyadic` slightly beats `bp_float` here (a ±2^k-as-regularizer effect
  on this small data).
- The +0.23 is the price of a **dyadic activation operand in the weight-gradient kernel**, *not* the
  local/async form (sync `relax` == async `relax_async`; both = the same quantized
  gradient; the *form* is fine — the *quantization* is the cost).
- **Plausibly reducible (untested here):** magnitude-aware dyadic backward (the Rung-1
  MLP achieved "no acc cost" with magnitude-kept-within-2× dyadic grads), or not
  quantizing the worst layer (the readout / top conv), or a sign-product update there.
  This is the concrete next lever if closing the +0.23 matters.
- **Whole-recipe closure is unmeasured:** loss, Adam/schedule, data processing, and
  evaluation are outside the network-internal dyadic credit-path claim.

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
