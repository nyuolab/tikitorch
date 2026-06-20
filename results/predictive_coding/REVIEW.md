# Where tropical/dyadic PC stands relative to the predictive-coding literature

**Date:** 2026-06-20. Consolidated read of the whole effort against the PC field.
Sources for our numbers (all committed, each adversarially audited): `SCALE_TESTS.md`
(discriminative), `GENERATIVE_PC.md` + `ASSOCIATIVE_MEMORY.md` (generative),
`PARALLELISM.md` (the hardware thesis), `BENCHMARK.md` (head-to-head table). Competitor
numbers are published full-precision-float values: Qi et al. 2025 (arXiv 2506.23800),
μPC / Innocenti et al. 2025 (2505.13124), Salvatori et al. NeurIPS 2021 (associative
memory), Whittington–Bogacz 2017, Song et al. (Z-IL), Elad–Aharon K-SVD 2006.

---

## 1. The one-paragraph verdict

The deep-PC field is fighting one central battle: **iterative-settling PC collapses with
depth**, and the recent SOTA papers (Qi, μPC) are machinery to rescue it — and even
rescued, it trails or only ties backprop. We do **not** win that battle on its own terms
(we did not make genuine deep *settling*-PC converge; our own attempt to reach the true
settling equilibrium just reproduces backprop). Instead we **sidestep** it: our cascade /
error-relaxation **is** backprop (the Z-IL limit), so it matches backprop at depth — and
it does so **multiplier-free** (±2^k shift-add, top-k WTA), which **no competitor is**.
On the generative side, PC's real over-backprop advantage (inference / associative
recall) **is** genuine and established in the literature, and we **reproduce it
multiplier-free**. So the defensible, unique contribution across both regimes is one
axis nobody else has — **multiplier-free + synchronization-free credit assignment that
reaches backprop quality** — not "more accurate" and not "better deep settling-PC."

---

## 2. Discriminative accuracy — vs Qi / μPC

The field's baseline pathology: *plain* iterative-settling PC loses 20–23 points to
backprop as nets deepen, and the SOTA variants claw most of it back with extra
machinery. Our multiplier-free cascade tracks backprop wherever settling-PC collapses.

| task / arch | plain settling-PC | enhanced PC (lit) | backprop | **ours (multiplier-free)** |
|---|---|---|---|---|
| CIFAR-10 VGG-10 | 72.75 (Qi, collapse) | 92.07 (Qi S+F) | 92.21 (Qi) | **91.4** cascade / 92.2 float |
| CIFAR-100 VGG-10 | 45.86 (Qi, collapse) | 63.35 (Qi S+F) | 69.35 (Qi) | **71.0** cascade / 71.5 float |
| **CIFAR-10 ResNet-18** | **70.44 (best PC, −23pt)** | — | 93.7 (ours) | **93.3** cascade |
| CIFAR-100 ResNet-18 | — | — | 74.3 (ours) | **73.8** cascade |
| CIFAR-10 VGG-15 (deep) | — | — | 92.1 (ours) | 91.4 cascade |
| MNIST MLP (shallow) | 98.1 / iPC 98.45 | — | 98.07 | 97.4 (d8) |
| **MNIST 128-layer (residual)** | — | ~98 (μPC) | **24.1 (float BP collapses)** | **93.8** cascade |
| Tiny-ImageNet ResNet-18 (200-way) | 22.95 (Qi VGG-15, collapse) | 42.51 (Qi S+F) | — | *run in flight* |

**Three clean structural wins:**

1. **ResNet-18 (vs the benchmark's collapse).** The literature's best settling-PC drops
   to **70.44%** on ResNet-18 (−23pt); our multiplier-free cascade is **93.3%**, matching
   float backprop (93.7%) within 0.4pp. Same on CIFAR-100 ResNet-18 (73.8 ≈ 74.3). This
   is the sharpest case: *wherever settling-PC collapses, our multiplier-free cascade
   matches backprop.*
2. **128 layers (vs μPC).** μPC's entire contribution is a parameterisation (Depth-μP)
   that lets *float* PC train at 128 layers. We measured a 128-layer residual MLP three
   ways: multiplier-free **cascade = 93.8%** (flat from d32), `bp_dyadic` 86.6%, and
   **float additive-residual backprop collapses to 24.1%**. The **tropical max-residual**
   `max(relu(Wx), x)` — bounded and idempotent, so it does not accumulate variance the
   way additive float residuals do — is doing μPC's depth-scaling job, multiplier-free,
   at a depth where the float additive-residual net dies.
3. **Qi-scale parity (calibrated).** Our float VGG-10 (92.17%) reproduces Qi's backprop
   number (92.21%), so the baseline is calibrated to their setup — and the fully
   multiplier-free cascade is within 0.8pp of it, ahead of Qi's float S+F on CIFAR-100.

**The honest catch (stated plainly):** these wins are because *our discriminative PC is
backprop* (Z-IL / cascade), not deep-settling-PC. Qi and μPC are solving a harder,
different problem — making genuine iterative-*settling* PC converge at 10–128 layers —
which we did **not** do. Our own attempt to reach the true settling equilibrium (CG-exact
`settle_nudge`) just reproduces backprop with **no** prospective-config or continual-
learning edge (honest negative; `SCALE_TESTS.md`). So we win on **accuracy-per-multiplier**
and on **not-collapsing**, not on "better deep PC." Where everyone ties (CIFAR-10 ≈ 92%,
shallow MNIST) it has been known since Whittington–Bogacz that PC ≈ backprop.

---

## 3. Generative inference — vs Salvatori / K-SVD

This is the regime where PC genuinely beats backprop, and the literature already shows
it strongly. Our contribution is doing it **multiplier-free**.

| work | task | PC | backprop / baseline |
|---|---|---|---|
| Salvatori 2021 (float, 10-layer PCN) | Tiny-ImageNet, ½ pixels | **98% retrieved** | 72% (backprop AE) |
| Salvatori 2021 | Tiny-ImageNet, ¼ pixels | 74% | 48% (backprop AE) |
| K-SVD (float sparse coding) | image denoising | 34.86 dB PSNR | 34.62 (wavelet) |
| **ours (mult-free sparse-PC)** | **MNIST, 50% random pixels** | **60% better than mean-fill** | (no fixed AE answers arbitrary masks) |
| **ours (mult-free, assoc. memory)** | **store K=200, recall from 50% cue** | **43% recalled; recon MSE 0.0104** | **0% recalled; MSE 0.0384** (backprop AE) |

**Two measured results, both reproducing the PC≫backprop pattern multiplier-free:**

- **In-painting** (`GENERATIVE_PC.md`): one `SparseCodingPC` answers *any* mask by
  settling its top-k code from observed pixels — recovers MNIST digits from 50% random
  masking 60% better than mean-fill. A discriminative backprop net structurally can't do
  this (it is a fixed input→label map).
- **Associative memory** (`ASSOCIATIVE_MEMORY.md`, new 2026-06-20): store K=200 MNIST,
  recall from a 50% cue — sparse-PC recalls **43%** below MSE 0.01, the backprop AE
  recalls **0%**, and PC's reconstruction error is 3.5–3.7× lower. Same qualitative gap
  Salvatori report (PCN ≫ AE), here with a ±2^k dictionary and WTA code.

**Honest scope:** ours is **single-layer on MNIST**; Salvatori is 10-layer on
Tiny-ImageNet with a stricter perfect-retrieval metric and far higher absolute numbers
(98% vs our 43% at a looser threshold), K-SVD is the float gold standard on natural
images. So this is a **capability demonstration under the harder (multiplier-free)
constraint**, well behind the float PC SOTA on scale and quality.

---

## 4. The parallelism thesis — the axis nobody else competes on

Since accuracy is backprop-equivalent, accuracy is **not** the value proposition
(`PARALLELISM.md`). The value of being a *legit* PC (local energy, settling, no global
backward) is that it removes the synchronization backprop needs. Backprop imposes three
locks (Jaderberg 2017): forward / update / backward. **PC breaks update + backward:**
the weight update is local (`ΔW_l ∝ ε_l ⊗ x_l`), every layer updates at once, no
sequential backward sweep, no global gradient to route.

We **measured** the part that matters: async error-relaxation on full MNIST is **sync ==
cascade == backprop bit-exact**, and **async within ~0.1pp** under 50% participation +
stale reads + random update order at d4–d32. I.e. half the workers silent each round, no
global clock, lagged communication — and it trains to the same accuracy. A
pipeline-throughput model turns the missing backward sweep + staleness-tolerance into
**2.3× (no latency) → 3.9× (latency-bound) → grows with depth** speedup over backprop.

The two axes none of the competitors (all FP32, real multipliers, settling-PC *more*
multiply-heavy than backprop) have, and they compound:
- **Multiplier-free** — ±2^k shift-add, top-k WTA, no DSP multiplies.
- **Synchronization-free** — local, async, delay/partial-tolerant; no global backward,
  no global clock, no all-to-all gradient routing.

That combination is what maps onto neuromorphic / in-memory / wafer-scale / chiplet-async
substrates, where backprop's global synchronous backward is the blocker, not the FLOPs.
**Honest scope:** the throughput numbers are a dependency-structure *model*, not measured
wall-clock; the staleness-tolerance that lets PC hide latency is the one piece we
*measured*. Real-hardware async throughput is the next claim to substantiate.

---

## 5. Scorecard

| claim | status |
|---|---|
| Multiplier-free PC reaching backprop accuracy at depth | **Won, measured** (cascade = BP; ResNet-18, 128-layer, VGG) |
| Matches backprop where settling-PC collapses | **Won, measured** (ResNet-18 93.3 vs 70.4; 128-layer 93.8 vs float-BP 24.1) |
| Qi-scale discriminative parity (calibrated) | **Won, measured** (VGG-10 cascade 91.4, float 92.2 = Qi 92.21) |
| PC≫backprop generative inference, multiplier-free | **Won, measured** (in-painting + associative memory) |
| Sync-free / async credit assignment survives parallel-HW conditions | **Won, measured** (async = sync within 0.1pp) |
| Throughput advantage from broken locks | **Modeled** (2.3–3.9×), not measured wall-clock |
| Better *deep settling*-PC than Qi / μPC (their problem) | **No** — we sidestepped it (cascade = backprop), did not beat it |
| Prospective-config / continual-learning PC edge | **No** (honest negative — true equilibrium ≈ backprop) |
| Generative SOTA scale/quality | **No** — single-layer MNIST, behind Salvatori/K-SVD |
| Tiny-ImageNet (Qi's hardest discriminative) | **In flight** (ResNet-18 cascade vs BP, EC2) |
| Measured deploy energy/throughput on fabric | **Open** (accuracy is GPU-sim; the deploy win is unproven here) |

---

## 6. Bottom line vs the field

- **Against Qi / μPC (discriminative):** competitive-to-ahead *on accuracy*, but via
  backprop-equivalence — we match backprop where deep-settling-PC collapses, multiplier-
  free. We did not beat them at *their* problem (deep settling-PC); we made it moot.
- **Against Salvatori / K-SVD (generative):** we reproduce PC's genuine over-backprop
  advantage (inference, associative recall) multiplier-free, at modest (MNIST,
  single-layer) scale — behind the float SOTA, ahead on the hardware constraint.
- **The defensible contribution:** **multiplier-free + synchronization-free credit
  assignment that reaches backprop quality** — the one axis the entire float, synchronous
  PC literature does not occupy. The remaining work to make it a true head-to-head is
  Tiny-ImageNet (in flight) and **measured** deploy energy/throughput on real
  multiplier-free / async hardware.
