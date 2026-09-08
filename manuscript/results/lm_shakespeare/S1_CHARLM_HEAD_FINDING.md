# S1 tropical transformer char-LM — root cause: the max-plus head, and the mult-free fix

**Date:** 2026-06-26 · CSIM (CPU, g++ #else int64 = silicon-faithful) + a torch probe. No F2 spent.

## Question
Can the existing v17 AFI (D=64/H2/DH32/FFN256/L_SEQ64/V64/1-layer multiplier-free transformer) be trained into a
real char-LM on tinyshakespeare (lowercased, 39-char vocab), measured by held-out **bits/char**? Baselines on the
val split: **unigram 4.42, bigram(cond) 3.47–3.52 bpc**.

## Findings (all CSIM, driving the REAL kernel)
1. **The kernel ±1 dyadic-Adam optimizer is too coarse for an LM.** `step = m>>round(log2 v) ∈ {-1,0,+1}`, and
   `S1_LR_SHIFT` is inert (≥1 rounds the step to 0). Batch-1 SGD on noisy text ratchets the most-frequent char up →
   collapse to space-mode → bpc *diverges*. (`s1_charlm.cpp`, optimizer `kernel-pm1`.)
2. **It is NOT the optimizer.** Two stronger host optimizers were added (the optimizer runs host-side on F2, so this
   needs no rebuild): `m1` = mult-free dyadic-Adam + grad-accumulation over a power-of-2 batch; `m2` = an FP-master
   QAT-Adam upper bound (real float Adam on a high-precision host master, straight-through through the mult-free
   quantized fwd/bwd; the host code-quantizer is **bit-exact** to the kernel `s1_encode`). Even the FP-master upper
   bound saturates at **~4.66 bpc** across all K/lr — *worse than unigram*. Temperature sweep explodes when sharpened
   (the logits carry no held-out info). m2-FREEZE (perfect Adam on embed+head only) also tops at 4.66.
3. **It is NOT capacity.** Recompiling the real kernel at D=128/L2 and D=256/L4 (16× the params) gives the *same*
   ~4.7–4.9 bpc plateau, val top1 stuck at space-frequency. More size does not help.
4. **The wall is the MAX-PLUS HEAD.** `logits[v] = max_d(W_HEAD[v][d] + R_final[d])` — an argmax-over-d morphological
   readout, sitting after LayerNorm — collapses every input to the unigram marginal. A torch probe (D=64/1-layer,
   AdamW, faithful 1-layer transformer) settles it:

   | head | val bits/char | vs bigram (3.47) |
   |---|---|---|
   | max-plus (kernel) | 3.59 | does **not** beat bigram |
   | linear (float dot-product) | 2.59 | beats by 0.9 |
   | **dyadic (mult-free, ±2^k weights, STE)** | **2.63** | **beats by 0.84, within 0.04 of float** |

## Conclusion / fix
Replace the max-plus head with a **multiplier-free dyadic-GEMM head** (`logits[v] = Σ_d ±2^k · R_final[d]`, the same
shift-add primitive the backbone already uses, DSP=0). **D=64/1-layer suffices** — no bigger model needed. This makes
the tropical transformer a genuine char-LM (~2.6 bpc, beats bigram) that is still fully multiplier-free.

Next: implement the dyadic head in the kernel (fwd+bwd, finite-difference-checked), csim-validate it beats bigram on
the real kernel, then AFI rebuild (gated) → train the tropical transformer char-LM on real F2.

Artifacts: `tikitorch/hardware/hls/forge_kernels/s1_charlm.cpp` (real-kernel char-LM csim trainer; optimizers
kernel-pm1 / mf-accum-m1 / fpmaster-m2; true held-out log2-CE bpc), `scripts/charlm_head_probe.py` (the torch
head-comparison probe), `s1_converge_gate.cpp` (the convergence gate driving the real kernel).
