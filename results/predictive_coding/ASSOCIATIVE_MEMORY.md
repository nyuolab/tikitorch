# Associative memory: multiplier-free sparse-PC vs backprop autoencoder

**Date:** 2026-06-20. Script: `scripts/pc_associative_memory.py` (reproducible, seed 0).
This is the multiplier-free analogue of Salvatori et al. 2021 ("Associative memories via
predictive coding", NeurIPS) — store K patterns, recall each from a partial cue, measure
the fraction recalled below an error threshold. Salvatori's float PCN strongly beats a
backprop autoencoder; we reproduce that gap under the **multiplier-free** constraint.

## Setup

- Store **K=200 MNIST** images. Both models see only these 200 (pure memorization).
- **Multiplier-free SparseCodingPC** (`n_pixels=784, n_atoms=512, n_active=20`): ±2^k
  (dyadic) dictionary, top-k WTA sparse code, local dyadic dict update. Memorize via 120
  `train_step`s. Recall by `complete(X, mask)` — settle the code from the *observed*
  pixels, reconstruct the whole image.
- **Backprop autoencoder** (784-256-64-256-784, ReLU, Adam 2e-3, 2000 steps): recall by
  feeding the masked cue forward `ae(cue)`.
- Cue = random pixel mask (keep 50% / 25%). Retrieval-fraction = images with recon
  MSE < threshold.

## Result (measured)

| kept | threshold | sparse-PC (mult-free) | backprop-AE |
|---|---|---|---|
| 50% | 0.001 | 0% | 0% |
| 50% | 0.005 | 1% | 0% |
| **50%** | **0.01** | **43%** | **0%** |
| 25% | 0.001 | 0% | 0% |
| 25% | 0.005 | 0% | 0% |
| **25%** | **0.01** | **3%** | **0%** |

Mean reconstruction MSE (lower = better):

| kept | sparse-PC | backprop-AE | PC advantage |
|---|---|---|---|
| 50% | **0.0104** | 0.0384 | **3.7× lower** |
| 25% | **0.0210** | 0.0729 | **3.5× lower** |

## Reading it honestly

- **The PC≫AE associative-memory gap reproduces — multiplier-free.** At 50% pixels the
  sparse-PC recalls 43% of the stored set below MSE 0.01 while the backprop AE recalls
  **none**, and PC's mean reconstruction error is 3.5–3.7× lower at both mask levels.
  Same qualitative result as Salvatori (PCN ≫ backprop AE), here with a ±2^k dictionary
  and top-k WTA code — no multipliers.
- **Why the AE loses:** trained only to reconstruct, it learns a near-mean map and does
  not do *inference from a partial observation* — feeding a masked cue forward gives a
  blurred average, not the stored pattern. PC settles its latent from the observed pixels
  (Bayesian-style completion), which is exactly the capability backprop AEs lack. This is
  the one regime where PC genuinely beats backprop, and we get it multiplier-free.
- **Scope / honest gaps vs the float SOTA:** ours is **single-layer** sparse coding on
  **MNIST**; Salvatori is a 10-layer PCN on Tiny-ImageNet with a stricter
  perfect-retrieval metric and reports much higher absolute fractions (98% at ½ pixels).
  Our threshold (MSE<0.01) is loose, and absolute retrieval at the tight thresholds
  (0.001) is ~0 for both. So this is a **capability demonstration under the harder
  hardware constraint**, not a SOTA associative-memory result. The 25%-cue column is weak
  (3%) — partial-observation completion degrades fast at single-layer / heavy masking.

## Source

Salvatori, Song, Lukasiewicz, et al., "Associative memories via predictive coding,"
NeurIPS 2021 (float PCN ≫ backprop AE at partial-cue retrieval). Our column is
multiplier-free (±2^k dictionary, top-k WTA code, local dyadic update).
