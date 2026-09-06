# Where tropical/dyadic PC stands relative to the predictive-coding literature

**Date:** 2026-06-20. Consolidated read of the whole effort against the PC field.
Sources for our numbers (all committed, each adversarially audited): `SCALE_TESTS.md`
(discriminative), `GENERATIVE_PC.md` + `ASSOCIATIVE_MEMORY.md` (generative),
`PARALLELISM.md` (the hardware thesis), `BENCHMARK.md` (head-to-head table). Competitor
numbers are published full-precision-float values: Qi et al. 2025 (arXiv 2506.23800),
μPC / Innocenti et al. 2025 (2505.13124), Salvatori et al. NeurIPS 2021 (associative
memory), Whittington–Bogacz 2017, Song et al. (Z-IL), Elad–Aharon K-SVD 2006.

> **Correction (2026-09-06):** empirical claims here about TikiTorch's
> state-settling `settle_nudge` path inherit archived runs that accidentally used
> an effective unit relaxation step because the configured `gamma` was not
> applied. Those claims are provisional pending rerun. Cascade, error-relaxation,
> and generative results are unaffected.

---

## 1. The one-paragraph verdict

The deep-PC field is fighting one central battle: **iterative-settling PC collapses with
depth**, and the recent SOTA papers (Qi, μPC) are machinery to rescue it — and even
rescued, it trails or only ties backprop. TikiTorch has not yet shown that its corrected
gamma-damped state-settling path wins that battle; the archived unit-step path did not.
Instead we **sidestep** it with the currently validated methods: our cascade /
error-relaxation **is** backprop (the Z-IL limit), so it matches backprop at depth — and
its network-internal forward/credit operands are dyadic (±2^k), a constraint the
compared methods do not target.
On the generative side, PC's real over-backprop advantage (inference / associative
recall) **is** genuine and established in the literature, and we **reproduce it
with a dyadic dictionary/top-k sparse-code core**. So the defensible contribution is
accuracy evidence under dyadic internal paths plus neighbour-local credit scheduling.
Those properties create an analog/in-memory hardware opportunity; they do not by
themselves prove full-step closure or deployment. The reported scale recipes retain
declared loss/optimizer/non-dyadic boundaries, and `settle_nudge` uses exact diagonal
division. We claim neither "better deep settling-PC" nor a free GPU speedup from
"synchronization-freedom" (staleness-tolerance is shared with async-SGD; see §4).

---

## 2. Discriminative accuracy — vs Qi / μPC

The field's baseline pathology: *plain* iterative-settling PC loses 20–23 points to
backprop as nets deepen, and the SOTA variants claw most of it back with extra
machinery. Our dyadic-internal-path cascade tracks backprop wherever settling-PC
collapses.

| task / arch | plain settling-PC | enhanced PC (lit) | backprop | **ours (dyadic internal path)** |
|---|---|---|---|---|
| CIFAR-10 VGG-10 | 72.75 (Qi, collapse) | 92.07 (Qi S+F) | 92.21 (Qi) | **91.4** cascade / 92.2 float |
| CIFAR-100 VGG-10 | 45.86 (Qi, collapse) | 63.35 (Qi S+F) | 69.35 (Qi) | **71.0** cascade / 71.5 float |
| **CIFAR-10 ResNet-18** | **70.44 (best PC, −23pt)** | — | 93.7 (ours) | **93.3** cascade |
| CIFAR-100 ResNet-18 | — | — | 74.3 (ours) | **73.8** cascade |
| CIFAR-10 VGG-15 (deep) | — | — | 92.1 (ours) | 91.4 cascade |
| MNIST MLP (shallow) | 98.1 / iPC 98.45 | — | 98.07 | 97.4 (d8) |
| **MNIST 128-layer (residual)** | — | ~98 (μPC, 5 ep) | **24.1 (float *additive*-res. BP, all seeds)** | **93.8** cascade (*max*-res.) |
| Tiny-ImageNet 200-way (Qi: VGG-15; ours: ResNet-18) | 22.95 (Qi VGG-15) | 42.51 (Qi S+F) | 63.0 (ours RN-18) | **61.6** cascade (−1.4pp, n=1) |

**Three clean structural wins:**

1. **ResNet-18, including on Tiny-ImageNet (vs the benchmark's collapse).** The
   literature's best settling-PC drops to **70.44%** on CIFAR-10 ResNet-18 (−23pt); our
   dyadic-credit cascade is **93.3%**, matching float backprop (93.7%) within 0.4pp.
   Same on CIFAR-100 ResNet-18 (73.8 ≈ 74.3). And now on **Tiny-ImageNet** (200-way, Qi's
   hardest, where Qi's VGG-15 plain settling-PC collapses to 22.95): our dyadic-credit
   cascade ResNet-18 = **61.6%** vs our float backprop **63.0%** (−1.4pp, n=1 seed). This
   is the pattern across the board: *wherever settling-PC collapses, our dyadic-credit
   cascade matches backprop* — now confirmed on the hardest discriminative case.
2. **128 layers (vs μPC) — the lever is the residual *type*.** μPC's contribution is a
   parameterisation (Depth-μP) that lets *float* PC train at 128 layers (≈98%, 5 epochs).
   We measured a 128-layer residual MLP three ways (3 seeds; the residual type is
   code-verified: `bp_float` uses **additive** `relu(Wx)+x`, cascade/`bp_dyadic` use the
   tropical **max** `max(relu(Wx),x)`). Per-seed at d128: **float additive-residual BP
   collapses on all 3 seeds → 24.1%** (robust); the dyadic-credit **max-residual cascade
   trains on all 3 seeds → 93.8%** (the only *reliably*-training variant); `bp_dyadic`
   (dyadic conv/linear fwd, max-residual) is bimodal (mean 86.6, but 2/3 seeds ~94.4 — so its mean
   *understates* it). So the **tropical max-residual** — bounded/idempotent, so it does
   not accumulate variance the way additive float residuals do — does μPC's depth-scaling
   job with a dyadic forward/internal-credit path, at a depth where the *additive*
   float net dies on every seed.
   (Honest: cascade *is* backprop credit; the lever is the residual type, not the cascade.
   The float d64=68.2 mean is a one-collapsed-seed artifact, median 88.7 — the collapse is
   sharp only at d128, not a smooth slide.)
3. **Qi-scale parity (calibrated).** Our float VGG-10 (92.17%) reproduces Qi's backprop
   number (92.21%), so the baseline is calibrated to their setup — and the
   dyadic-credit cascade is within 0.8pp of it, ahead of Qi's float S+F on CIFAR-100.

**The honest catch (stated plainly):** these wins are because *our discriminative PC is
backprop* (Z-IL / cascade), not deep-settling-PC. Qi and μPC are solving a harder,
different problem — making genuine iterative-*settling* PC converge at 10–128 layers —
which we have **not yet demonstrated** with the corrected gamma-damped path. A
separate strong-inner-solver, hard-clamp equilibrium experiment is documented in
`TROPICAL_PC_DEPTH.md`; it is not a benchmark of the corrected `settle_nudge`
implementation. So we win on **accuracy under a dyadic internal-path constraint**
and on **not-collapsing**, not on "better deep PC." Where everyone ties (CIFAR-10 ≈ 92%,
shallow MNIST) it has been known since Whittington–Bogacz that PC ≈ backprop.

---

## 3. Generative inference — vs Salvatori / K-SVD

This is the regime where PC genuinely beats backprop, and the literature already shows
it strongly. Our contribution is testing a dyadic dictionary/top-k sparse-code core.

| work | task | PC | backprop / baseline |
|---|---|---|---|
| Salvatori 2021 (float, 10-layer PCN) | Tiny-ImageNet, ½ pixels | **98% retrieved** | 72% (backprop AE) |
| Salvatori 2021 | Tiny-ImageNet, ¼ pixels | 74% | 48% (backprop AE) |
| K-SVD (float sparse coding) | image denoising | 34.86 dB PSNR | 34.62 (wavelet) |
| **ours (dyadic-dictionary sparse-PC)** | **MNIST, 50% random pixels** | **60% better than mean-fill** | (no fixed AE answers arbitrary masks) |
| **ours (dyadic dictionary, assoc. memory)** | **store K=200, recall from 50% cue** | **43% recalled; recon MSE 0.0104** | **0% recalled; MSE 0.0384** (backprop AE) |

**Two measured results reproduce the PC≫backprop pattern with that dyadic core:**

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
images. So this is a **capability demonstration under a dyadic-operator
constraint**, well behind the float PC SOTA on scale and quality and without a
full-workflow closure claim.

---

## 4. The parallelism thesis — re-scoped to what actually survives scrutiny

This is the half the whole effort points at, and an adversarial review (2026-06-20,
4-lens panel + async-SGD rebuttal) forced it to be re-scoped. The honest version is
**narrower and more defensible** than "synchronization-free → 2.3–3.9× over backprop."

**First, two corrections that matter.** (1) The "no backward sweep / all layers at once"
claim was attached to the wrong method. The *accuracy-matching* learner is the
error-relaxation, which is **bit-exact equal to the cascade = backprop credit** — it
*has* a directional top-down credit dependency (`E_l = A_l^T E_{l+1}`); async execution
*reorders/delays* that dependency, it does not *eliminate* it. The method that genuinely
has no directional sweep (state-settling `settle_nudge`) is the one whose archived
unit-step runs **diverged at scale** and capped at ~85% of backprop; corrected-gamma
results are pending. Current validated evidence therefore does not establish
"matches backprop" and "breaks the backward lock" in one method. (2)
**Staleness-tolerance is *not* PC-unique.** Async-SGD, Hogwild!, Downpour,
PipeDream/1F1B already hide pipeline latency and train under bounded-stale gradients for
backprop. A throughput model that compares PC only against *synchronous* backprop (as our
`scripts/pc_throughput_model.py` did) measures against a strawman; the 2.3–3.9× is now demoted to
an explicitly-modeled illustration with a sensitivity band (floor ~1.7×), not a headline.

**What genuinely survives the async-SGD rebuttal.** The validated cascade/error-relax
path retains `E_l = A_l^T E_{l+1}`, so it does **not** eliminate the directional
backward relation. It expresses that relation as adjacent-layer messages, reuses the
forward parameter tensor instead of learning a separate feedback matrix, and uses
dyadic operands in the internal weight-gradient path. Partial/stale/arbitrary-order
schedules can delay and reorder those messages, but information still propagates
through depth. This representation creates a plausible **analog / in-memory /
crossbar implementation path**; code-level locality and tensor reuse are not proof of
a physical reverse datapath, full-recipe closure, or deployability. State settling is
the path that could remove an ordered reverse phase, but corrected-gamma scale results
and end-to-end hardware closure remain open.

**What we measured (and the new measurement that grounds the cost).** Async
error-relaxation on full MNIST is **sync == cascade == backprop bit-exact**, and **async
within ~0.1pp** under 50% participation + stale reads + random order at d4–d32 — the
credit signal converges to the *exact* backprop gradient under fair, sufficiently long
updates (the frozen-routing system is triangular/nilpotent). To stop booking the cost of asynchrony as zero, we now
also measured the **round-tax** (`scripts/pc_staleness_frontier.py`,
`staleness_frontier.png`): how many local message-passing rounds async needs to reach the
exact backprop gradient (cosine ≥ 0.999) vs the single ordered sync pass. **The result is
two-sided and honest.** *Positive:* when the async gradient reached the threshold, it
approached the exact backprop gradient (cosine → 1.000) under partial participation +
lagged reads + random order, consistent with finite propagation under fair updates.
*Falsifier:* the round-tax **grows ~O(L /
participation)** (measured: stale = L, partial-50% = 1.5·L, 50%+stale+rand ≈ 4–6·L, and
25%+stale+rand doesn't converge within 128 rounds at depth ≥ 32). Since that is the *same
order* as the backward sweep it replaces, **there is no free, depth-growing throughput
speedup** — the earlier "speedup grows with depth" claim is refuted by our own
measurement. What async actually buys is **scheduling freedom with correct credit** in
the converged runs, at O(L) rounds — a candidate substrate-mapping property, not a
pipeline win or hardware deployment.

**Bottom line on parallelism:** the durable claims are dyadic internal operands,
neighbour-local messages, no separately learned feedback matrix, and measured schedule
tolerance in simulation. These support a settling/in-memory **research opportunity**;
they do not establish zero-DSP/full-step closure or real-hardware deployment. The
contestable claim ("free GPU speedup via synchronization-freedom") is dropped:
staleness-tolerance is shared with async-SGD, and the throughput multiplier is a model,
not a measurement.

---

## 5. Scorecard

| claim | status |
|---|---|
| Dyadic internal-credit path reaching backprop accuracy at depth | **Won, measured** (cascade = BP; ResNet-18, 128-layer, VGG) |
| Matches backprop where settling-PC collapses | **Won, measured** (ResNet-18 93.3 vs 70.4; 128-layer 93.8 vs float-BP 24.1) |
| Qi-scale discriminative parity (calibrated) | **Won, measured** (VGG-10 cascade 91.4, float 92.2 = Qi 92.21) |
| PC≫backprop generative inference with a dyadic dictionary/top-k core | **Won, measured** (in-painting + associative memory) |
| Async credit approaches the exact backprop gradient under stale/partial/random-order | **Measured in simulation** (cosine→1.000 for converged conditions; the hardest depth/participation cells were censored at 128 rounds; async accuracy within 0.1pp on MNIST) |
| Neighbour-local credit + no separately learned feedback matrix | **Established in code**; hardware mapping and reverse-datapath cost remain unmeasured |
| Whole benchmark training recipe is multiplier-free / zero-DSP closed | **Open** (loss, optimizer, non-dyadic model ops, and state-settling division are declared boundaries) |
| Round-tax of asynchrony bounded → depth-growing throughput win | **No, measured** (`scripts/pc_staleness_frontier.py`) — tax grows ~O(L/p); refutes the "grows with depth" claim |
| Synchronization-free → free GPU speedup over backprop | **Dropped** — staleness shared with async-SGD/Hogwild; throughput straddles 1× (overlap-dependent); 2.3–3.9× was vs a *synchronous* strawman |
| "Break the backward lock" **and** match backprop accuracy simultaneously | **Open for corrected state settling** — the validated accuracy-matching form has an ordered dependency; the archived unit-step lock-breaking form trailed ~15% |
| Better *deep settling*-PC than Qi / μPC (their problem) | **No** — we sidestepped it (cascade = backprop), did not beat it |
| Prospective-config / continual-learning PC edge | **No clean win in the tested task** (stability improved, plasticity/mean accuracy worsened) |
| Generative SOTA scale/quality | **No** — single-layer MNIST, behind Salvatori/K-SVD |
| Tiny-ImageNet 200-way (Qi's hardest discriminative) | **Won, measured** (cascade 61.6 vs BP 63.0, −1.4pp; ResNet-18, n=1) |
| Measured deploy energy/throughput on real settling/in-memory fabric | **Open** (accuracy/async evidence is simulation; the substrate mapping is an opportunity) |

---

## 6. Bottom line vs the field

- **Against Qi / μPC (discriminative):** competitive-to-ahead *on accuracy*, but via
  backprop-equivalence — our dyadic-internal-path cascade matches backprop where
  deep-settling-PC collapses. We did not beat them at *their* problem (deep
  settling-PC); we sidestepped it.
- **Against Salvatori / K-SVD (generative):** we reproduce PC's genuine over-backprop
  advantage (inference, associative recall) with a dyadic dictionary/top-k core, at
  modest (MNIST, single-layer) scale — behind the float SOTA.
- **The defensible contribution (re-scoped after adversarial review):** a dyadic
  network-internal credit path that reaches **backprop quality**, with neighbour-local
  messages and no separately learned feedback matrix. That combination motivates an
  **analog / in-memory / settling-hardware opportunity**, but the repository has not
  closed or deployed the complete recipe on that substrate. We explicitly **do not**
  claim a free GPU speedup from "synchronization-freedom": staleness-tolerance is shared
  with async-SGD, and the throughput multiplier is a model, not a measurement. With
  Tiny-ImageNet now run (cascade 61.6 ≈ BP 63.0) and the round-tax now measured
  (`scripts/pc_staleness_frontier.py`: async → exact backprop gradient, but at O(L/p) rounds), the
  remaining work includes a recipe-specific closure audit and **measured** deploy
  energy/throughput on real hardware; the substrate mapping is a structural hypothesis,
  not a completed result.
