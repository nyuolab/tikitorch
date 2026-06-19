# Benchmark vs competing predictive-coding works (honest)

**Date:** 2026-06-19. Our numbers: `SCALE_TESTS.md`, `GENERATIVE_PC.md` (committed,
fact-checked). Competitor numbers: published values from Qi et al. 2025
(arXiv 2506.23800), μPC (2505.13124), the PC-benchmark / Z-IL line
(Pinchetti/Salvatori; Song et al.), and generative/associative-memory PC
(Salvatori et al. NeurIPS 2021) + classical sparse coding (Elad-Aharon K-SVD).

## TL;DR — the honest read

Two regimes, two different stories:

1. **Discriminative accuracy.** The deep-PC field's core problem is that
   *iterative-settling* PC **collapses with depth** and has to be rescued by heavy
   machinery — and even then trails backprop on the hard cases. We **sidestep that
   problem entirely**: our cascade / error-relaxation *is* backprop (the Z-IL limit),
   so it matches backprop at depth **and is multiplier-free**, which **no competitor
   is**. So on accuracy we are competitive-to-ahead of the deep-PC SOTA — but honestly
   *because backprop ≥ deep-settling-PC at depth*, not because we did deeper settling.
2. **Generative inference.** Here PC genuinely beats backprop, and the literature
   already shows it strongly (Salvatori PCN ≫ backprop autoencoder at in-painting).
   Our contribution is doing it **multiplier-free** (sparse coding) — demonstrated at
   modest scale (MNIST), behind the float PC SOTA on scale/quality.

**The axis nobody else has:** every competing PC work is **FP32 with real multipliers**
(and settling-PC does *more* multiplies than backprop, T inference iterations per
update). Ours is the only **multiplier-free** (±2^k shift-add) one. That is the
genuine differentiator across both regimes.

## Discriminative: accuracy vs the deep-PC literature

All competitor numbers are full-precision float. "plain PC" = vanilla iterative
settling (the collapse baseline); "best PC" = their enhanced variant.

| task / arch | plain settling-PC | enhanced PC | backprop | **ours (multiplier-free)** |
|---|---|---|---|---|
| CIFAR-10 VGG-10 | 72.75 (Qi, collapse) | 92.07 (Qi S+F) | 92.21 (Qi) | **91.4** cascade / 92.2 float |
| CIFAR-100 VGG-10 | 45.86 (Qi, collapse) | **63.35** (Qi S+F) | 69.35 (Qi) | **71.0** cascade / 71.5 float |
| Tiny-ImageNet VGG-15 | 22.95 (Qi, collapse) | 42.51 (Qi S+F) | 44.52 (Qi) | — *(not run)* |
| CIFAR-10 ResNet-18 | 70.44 (best PC, −22pt) | — | 92.83 | — *(not run)* |
| MNIST MLP (shallow) | 98.1 / iPC 98.45 | — | 98.07 | 97.4 (d8 deep) |
| MNIST 128-layer ResNet | — | ~98 (μPC) | (no BP given) | — *(we did d32)* |

Reading this honestly:

- **Where it's a clean win:** on **CIFAR-100 VGG-10**, our multiplier-free cascade
  (71.0%, matching our 71.5% float backprop) is **ahead of Qi's float S+F settling-PC
  (63.35%)** — because their enhanced PC still trails backprop by ~6pt there while we
  *are* backprop. Same pattern wherever settling-PC collapses (ResNet-18: PC 70.44 vs
  BP 92.83). (Caveat: our VGG-10 / training schedule isn't identical to Qi's — our BP
  baseline is 71.5 vs their 69.35 — so read it as "we match backprop; they don't,"
  not a like-for-like 71.0-vs-63.35 sweep.)
- **Where it's a tie:** CIFAR-10 (everyone ≈ backprop ≈ 92%); shallow MNIST/Fashion
  (PC ≈ backprop has been known since Whittington-Bogacz).
- **The honest catch:** this "win" is **because our discriminative PC is backprop**
  (Z-IL), not deep-settling-PC. Qi and μPC are solving a *harder, different* problem —
  making genuine iterative-settling PC converge at 10–128 layers — which we did **not**
  do (our own attempt to reach the true settling equilibrium ≈ backprop too;
  `SCALE_TESTS.md`). So: we win on *accuracy-per-multiplier* and on *not-collapsing*,
  not on "better deep PC."
- **Scale gap (real):** we have **not** run **Tiny-ImageNet** (Qi's hardest) or
  **ResNet / 128-layer** (Qi / μPC). Our deepest is VGG-13 (CIFAR) and 32-layer MLP.
  A true head-to-head needs those.

## Generative: in-painting / inference vs the generative-PC literature

PC's real advantage over backprop is here — and it's established:

| work | task | PC | backprop / baseline |
|---|---|---|---|
| Salvatori 2021 (float, 10-layer PCN) | Tiny-ImageNet, ½ pixels given | **98% retrieved** | 72% (backprop AE) |
| Salvatori 2021 | Tiny-ImageNet, ¼ pixels given | 74% | 48% (backprop AE) |
| Salvatori 2021 | CIFAR-10 corrupted retrieval | PCN ≫ | Modern Hopfield ~9 imgs |
| K-SVD (float sparse coding) | image denoising (Lena/Barbara…) | 34.86 dB PSNR | 34.62 (wavelet BLS-GSM) |
| **ours (multiplier-free SparseCodingPC)** | **MNIST, 50% random pixels** | **60% better than mean-fill** | (no backprop AE can answer arbitrary masks) |

Reading this honestly:

- The PC>backprop *generative* gap is **real and established** (Salvatori: 98% vs 72%).
  Our result **confirms the same capability** — one model, arbitrary masks, by settling
  — but does it **multiplier-free** (±2^k dictionary, top-k WTA sparse code), which the
  float literature does not.
- We are **behind the float PC SOTA on scale and quality**: ours is single-layer on
  MNIST; Salvatori is 10-layer on Tiny-ImageNet, K-SVD is the gold standard on natural
  images. Our metric (MSE-vs-mean-fill) is also weaker than theirs (perfect-retrieval
  fraction, PSNR). So this is a *capability demonstration under the harder
  (multiplier-free) constraint*, not a SOTA generative result.

## Verdict

- **Multiplier-free is ours alone**, across both regimes — that is the defensible,
  unique contribution. Everyone else is float, and settling-PC is *more* multiply-heavy
  than backprop.
- **Discriminative:** competitive-to-ahead on accuracy (we match backprop where
  deep-settling-PC collapses or trails), but via backprop-equivalence — we did *not*
  beat Qi/μPC at *their* problem (deep settling-PC); we sidestepped it.
- **Generative:** we reproduce PC's genuine over-backprop advantage (in-painting by
  settling) multiplier-free, at modest (MNIST, single-layer) scale — behind the float
  PC SOTA, ahead on the hardware constraint.

## To make it a true head-to-head (runnable next, on EC2)

1. **Tiny-ImageNet VGG-15** and **CIFAR ResNet-18** with the dyadic cascade — Qi's and
   the benchmark's hardest discriminative cases.
2. **Deeper / 128-layer** dyadic nets — μPC's regime (depth as the axis).
3. **Scale up the generative PC** — hierarchical (multi-layer) sparse/generative PC on
   Tiny-ImageNet with the *retrieval-fraction* metric, for a direct Salvatori
   comparison; report PSNR for denoising to compare K-SVD.

## Sources

Qi et al. 2025 (2506.23800); μPC / Innocenti et al. 2025 (2505.13124); PC-benchmark /
Z-IL line (Pinchetti & Salvatori; Song et al. "Can the brain do backprop");
Whittington-Bogacz 2017; Salvatori et al. "Associative memories via predictive coding"
(NeurIPS 2021); Elad-Aharon K-SVD denoising (2006). All competitor numbers are
full-precision float; the ours column is multiplier-free (±2^k).
