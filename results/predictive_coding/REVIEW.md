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
axis nobody else has — credit assignment at **backprop quality** that is **multiplier-
free** (±2^k), **weight-transport-free**, and **credit-local-in-space**, hence deployable
on analog/in-memory/settling hardware backprop cannot run on — not "more accurate," not
"better deep settling-PC," and (after adversarial review) **not** a free GPU speedup from
"synchronization-freedom" (staleness-tolerance is shared with async-SGD; see §4).

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
| **MNIST 128-layer (residual)** | — | ~98 (μPC, 5 ep) | **24.1 (float *additive*-res. BP, all seeds)** | **93.8** cascade (*max*-res.) |
| Tiny-ImageNet 200-way (Qi: VGG-15; ours: ResNet-18) | 22.95 (Qi VGG-15) | 42.51 (Qi S+F) | — | *ResNet-18 run in flight* |

**Three clean structural wins:**

1. **ResNet-18 (vs the benchmark's collapse).** The literature's best settling-PC drops
   to **70.44%** on ResNet-18 (−23pt); our multiplier-free cascade is **93.3%**, matching
   float backprop (93.7%) within 0.4pp. Same on CIFAR-100 ResNet-18 (73.8 ≈ 74.3). This
   is the sharpest case: *wherever settling-PC collapses, our multiplier-free cascade
   matches backprop.*
2. **128 layers (vs μPC) — the lever is the residual *type*.** μPC's contribution is a
   parameterisation (Depth-μP) that lets *float* PC train at 128 layers (≈98%, 5 epochs).
   We measured a 128-layer residual MLP three ways (3 seeds; the residual type is
   code-verified: `bp_float` uses **additive** `relu(Wx)+x`, cascade/`bp_dyadic` use the
   tropical **max** `max(relu(Wx),x)`). Per-seed at d128: **float additive-residual BP
   collapses on all 3 seeds → 24.1%** (robust); the multiplier-free **max-residual cascade
   trains on all 3 seeds → 93.8%** (the only *reliably*-training variant); `bp_dyadic`
   (mult-free fwd, max-residual) is bimodal (mean 86.6, but 2/3 seeds ~94.4 — so its mean
   *understates* it). So the **tropical max-residual** — bounded/idempotent, so it does
   not accumulate variance the way additive float residuals do — does μPC's depth-scaling
   job multiplier-free, at a depth where the *additive* float net dies on every seed.
   (Honest: cascade *is* backprop credit; the lever is the residual type, not the cascade.
   The float d64=68.2 mean is a one-collapsed-seed artifact, median 88.7 — the collapse is
   sharp only at d128, not a smooth slide.)
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

## 4. The parallelism thesis — re-scoped to what actually survives scrutiny

This is the half the whole effort points at, and an adversarial review (2026-06-20,
4-lens panel + async-SGD rebuttal) forced it to be re-scoped. The honest version is
**narrower and more defensible** than "synchronization-free → 2.3–3.9× over backprop."

**First, two corrections that matter.** (1) The "no backward sweep / all layers at once"
claim was attached to the wrong method. The *accuracy-matching* learner is the
error-relaxation, which is **bit-exact equal to the cascade = backprop credit** — it
*has* a directional top-down credit dependency (`E_l = A_l^T E_{l+1}`); async execution
*reorders/delays* that dependency, it does not *eliminate* it. The method that genuinely
has no directional sweep (state-settling `settle_nudge`) is the one that **diverges at
scale** and caps at ~85% of backprop. So "matches backprop" and "breaks the backward
lock" come from **two different, non-coinciding methods** — you cannot claim both at
once. (2) **Staleness-tolerance is *not* PC-unique.** Async-SGD, Hogwild!, Downpour,
PipeDream/1F1B already hide pipeline latency and train under bounded-stale gradients for
backprop. A throughput model that compares PC only against *synchronous* backprop (as our
`pc_throughput_model.py` did) measures against a strawman; the 2.3–3.9× is now demoted to
an explicitly-modeled illustration with a sensitivity band (floor ~1.7×), not a headline.

**What genuinely survives the async-SGD rebuttal — the one axis async-backprop cannot
occupy.** PC eliminates the **backward graph itself**: (a) **no weight transport** — the
*same* forward `±2^k` weight carries the error (`_route_T`; the cascade's error step
reuses `self.layers[l].weight`, never a transposed copy), and (b) **no separate
error-carrying datapath** — credit is a **local neighbour-only message-passing fixpoint**,
not a global all-to-all gradient object that must be routed end-to-end before any update
is correct. Async-SGD/Hogwild parallelize the **data/batch axis** and still compute a full
backward with a transpose-weight datapath; they parallelize *where/when* the backward
runs, never *whether it exists*. PC's distinct contribution is on the **depth/credit
axis**: it makes the credit assignment runnable on **analog / in-memory / crossbar
settling hardware where backprop has no physical realization at all** (the EqProp /
memristor lineage). That is a **hardware-substrate** property, not a GPU throughput win —
and it is true in the code today, independent of any throughput model.

**What we measured (and the new measurement that grounds the cost).** Async
error-relaxation on full MNIST is **sync == cascade == backprop bit-exact**, and **async
within ~0.1pp** under 50% participation + stale reads + random order at d4–d32 — the
credit signal provably converges to the *exact* backprop gradient (the frozen-routing
system is triangular/nilpotent). To stop booking the cost of asynchrony as zero, we now
also measured the **round-tax** (`scripts/pc_staleness_frontier.py`,
`staleness_frontier.png`): how many local message-passing rounds async needs to reach the
exact backprop gradient (cosine ≥ 0.999) vs the single ordered sync pass. **The result is
two-sided and honest.** *Positive:* the async gradient **always converges to the exact
backprop gradient** (cosine → 1.000) under partial participation + lagged reads + random
order — it never lands on the wrong credit. *Falsifier:* the round-tax **grows ~O(L /
participation)** (measured: stale = L, partial-50% = 1.5·L, 50%+stale+rand ≈ 4–6·L, and
25%+stale+rand doesn't converge within 128 rounds at depth ≥ 32). Since that is the *same
order* as the backward sweep it replaces, **there is no free, depth-growing throughput
speedup** — the earlier "speedup grows with depth" claim is refuted by our own
measurement. What async actually buys is **scheduling freedom with correct credit** (any
order, any lag, partial participation → still the exact gradient), at O(L) rounds — a
substrate-enablement property (it runs on a clockless async fabric), not a pipeline win.

**Bottom line on parallelism:** the durable, code-true claims are **multiplier-free**
(±2^k shift-add, no DSP multiplies), **weight-transport-free**, and **credit-local-in-
space** — which together make this learner deployable on settling/in-memory substrates
backprop cannot run on. The contestable claim ("free GPU speedup via synchronization-
freedom") is dropped: staleness-tolerance is shared with async-SGD, and the throughput
multiplier is a model, not a measurement.

---

## 5. Scorecard

| claim | status |
|---|---|
| Multiplier-free PC reaching backprop accuracy at depth | **Won, measured** (cascade = BP; ResNet-18, 128-layer, VGG) |
| Matches backprop where settling-PC collapses | **Won, measured** (ResNet-18 93.3 vs 70.4; 128-layer 93.8 vs float-BP 24.1) |
| Qi-scale discriminative parity (calibrated) | **Won, measured** (VGG-10 cascade 91.4, float 92.2 = Qi 92.21) |
| PC≫backprop generative inference, multiplier-free | **Won, measured** (in-painting + associative memory) |
| Async credit converges to the *exact* backprop gradient under stale/partial/random-order | **Won, measured** (cosine→1.000 every condition; async acc = sync within 0.1pp, MNIST sim) |
| **Weight-transport-free + credit-local-in-space** (runnable on settling/in-memory HW) | **Won, code-verified** (same forward weight carries the error; neighbour-only fixpoint) |
| Round-tax of asynchrony bounded → depth-growing throughput win | **No, measured** (`pc_staleness_frontier.py`) — tax grows ~O(L/p); refutes the "grows with depth" claim |
| Synchronization-free → free GPU speedup over backprop | **Dropped** — staleness shared with async-SGD/Hogwild; throughput straddles 1× (overlap-dependent); 2.3–3.9× was vs a *synchronous* strawman |
| "Break the backward lock" **and** match backprop accuracy simultaneously | **No** — those are two different methods (accuracy-matching form *is* an ordered backward; the lock-breaking form trails ~15%) |
| Better *deep settling*-PC than Qi / μPC (their problem) | **No** — we sidestepped it (cascade = backprop), did not beat it |
| Prospective-config / continual-learning PC edge | **No** (honest negative — true equilibrium ≈ backprop) |
| Generative SOTA scale/quality | **No** — single-layer MNIST, behind Salvatori/K-SVD |
| Tiny-ImageNet (Qi's hardest discriminative) | **In flight** (ResNet-18 cascade vs BP, EC2) |
| Measured deploy energy/throughput on real settling/in-memory fabric | **Open** (accuracy is GPU-sim; the substrate win is structural, not yet HW-measured) |

---

## 6. Bottom line vs the field

- **Against Qi / μPC (discriminative):** competitive-to-ahead *on accuracy*, but via
  backprop-equivalence — we match backprop where deep-settling-PC collapses, multiplier-
  free. We did not beat them at *their* problem (deep settling-PC); we made it moot.
- **Against Salvatori / K-SVD (generative):** we reproduce PC's genuine over-backprop
  advantage (inference, associative recall) multiplier-free, at modest (MNIST,
  single-layer) scale — behind the float SOTA, ahead on the hardware constraint.
- **The defensible contribution (re-scoped after adversarial review):** a credit-
  assignment that reaches **backprop quality** while being **multiplier-free** (±2^k),
  **weight-transport-free**, and **credit-local-in-space** (a neighbour-only fixpoint, not
  a global gradient object). That combination is deployable on **analog / in-memory /
  settling hardware where backprop has no physical realization** — the axis the entire
  float, transpose-weight PC and DL literature does not occupy. We explicitly **do not**
  claim a free GPU speedup from "synchronization-freedom": staleness-tolerance is shared
  with async-SGD, and the throughput multiplier is a model, not a measurement. The
  remaining work to make it a true head-to-head: Tiny-ImageNet (in flight), the **measured
  round-tax** of asynchrony (`pc_staleness_frontier.py`), and **measured** deploy
  energy/throughput on real multiplier-free / settling hardware.
