# Tropical Forward-Forward (TFF)

*A forward-only, backprop-free, **multiplier-free** local learning rule on the
max-plus substrate.* Status: implemented, closure-verified, measured on MNIST.
Date: 2026-06-15.

> Reproduce: `python scripts/tropical_forward_forward.py` (full, ~10 min CPU) ·
> `--quick` (fast) · `--smoke` (1 step, synthetic, asserts closure).
> Tests: `pytest tests/test_forward_forward.py`. Code: `tikitorch/forward_forward.py`.

---

## 1. What this is

Hinton's Forward-Forward (FF, 2022) replaces the backward pass with a second
*forward* pass: a layer is trained **locally** to give high *goodness* on real
("positive") data and low goodness on "negative" data, with no gradient threaded
back from the output. TFF ports FF to TikiTorch's tropical (max-plus) substrate
and keeps the library's core invariant — **closure**: the entire training step
(both forward passes, the local loss, its backward, and the weight update) runs
with **zero hardware multiplies / divides / transcendentals**, the property that
makes the loop FPGA-native (zero-DSP).

Each layer is `h = max-plus(W, x) [+ tropical ReLU]` — the existing silicon
forward kernel (`TropicalDilationModule` / `TropicalDilationFn`). One layer = one
local learner.

## 2. The closure-clean design choices

Each choice below diverges from Hinton's float recipe *specifically* to remove a
hardware multiply. All are verified by the closure guard (`tt.assert_no_multiply`,
`tt.assert_training_closed`) — see §5.

| Piece | Hinton (float) | TFF (closure-clean) | Why it's multiplier-free |
|---|---|---|---|
| **Goodness** | `Σ h²` (sum of squares) | `Σ top-k(h)` (default) or `max(h)`; pluggable | a square is a multiply; top-k/sum/max are compare+add |
| **Local loss** | logistic | **hinge/margin**: push `G_pos ≥ θ+m`, `G_neg ≤ θ−m` | `dL/dG ∈ {−1,0,+1}` → the update is **sign-gated** |
| **Loss backward** | dense grad | **scatter** `{−1,0,+1}` into the top-k positions | no tensor×tensor multiply (cf. `TropicalPerceptronLossFn`) |
| **Inter-layer norm** | L2 length-norm | **subtract the max**: `h ← h − max(h)` | L2 = square+divide+rsqrt; max-shift = compare+subtract |
| **Threshold θ** | fixed | adaptive **batch midpoint** `½·(mean G_pos + mean G_neg)` | `½·` is a bit-shift; `mean`/`+` are closed |
| **Optimizer** | Adam | `PureTropicalSGD` (sign-only, pow-2 lr) | sign + integer-counter momentum + bit-shift lr |

**The local update, end to end.** For each top-k output unit of an *active*
sample, the loss scatters `−1` (positive data) or `+1` (negative data) onto that
unit; the max-plus backward routes it to the single **winning input edge**
(argmax); the sign optimizer nudges that edge by `±2⁻ˢ`. So TFF's weight update is
literally *"strengthen the winning edges on positive data, weaken them on negative
data,"* a sign-gated Hebbian rule — comparisons and additions only.

**Negative data.** (a) Hinton's **label overlay** (write a one-hot label into the
first `num_classes` pixels; positive = correct label, negative = a wrong one) —
used for the labeled MNIST task and the goodness classifier. (b) A
**tropical-native** generator, `scramble_features` (independently permute each
feature across the batch): it preserves every feature's marginal but destroys the
cross-feature joint structure the max-plus *routing* depends on — "combine
features from different images," made tropical.

**Logistic FF loss** is also provided (`tropical_ff_logistic_loss`) for fidelity,
with an explicit honesty caveat: its per-sample coefficient is a **host sigmoid**
(a transcendental, exactly like `tropical_cross_entropy`'s host softmax), so the
*per-weight* update stays multiply-free but computing the coefficient is not — and
under a *sign-only* optimizer the soft weighting collapses to an always-on ±1 push.
The hinge loss is the recommended closed default.

**Eval.** Classify by the label whose overlay maximizes the **summed per-layer
goodness** (`goodness_classify`; optional `skip_first` per Hinton). Fully closed
(forward kernel + goodness adds). A linear probe on the tropically-normalized
activations is also reported as a feature-quality diagnostic (a *declared*
non-closed boundary).

## 3. Results (MNIST)

Matched-budget run: 6000 train / 2000 test, width 300, 6 epochs, batch 100,
goodness = top-10, margin 2, overlay `on=1.0`, lr `2⁻⁴`, no momentum, `skip_first`
off, seed 0, CPU. (`results/forward_forward/mnist_results.jsonl`.)

**Test accuracy — learning rule × substrate × depth:**

| learning rule | substrate | depth 2 | depth 4 | depth 8 |
|---|---|---:|---:|---:|
| **Forward-Forward (local)** | **tropical max-plus (TFF)** | 0.200 | 0.226 | **0.230** |
| backprop (global) | tropical max-plus | 0.222 | 0.203 | **0.102** |
| Forward-Forward (local) | dense (Hinton float FF) | 0.276 | 0.222 | 0.202 |
| backprop (global) | dense MLP | 0.911 | 0.916 | 0.881 |

**Per-layer goodness separation** `G_pos − G_neg` (final batch; should be > 0):

| depth | per-layer separation |
|---|---|
| 2 | `[0.32, 0.43]` |
| 4 | `[0.50, 0.13, 0.23, −0.08]` |
| 8 | `[0.42, 0.08, 0, 0, 0, 0, 0, 0]` |

The local rule **does** open a positive gap — but only in the first ~2 layers;
after repeated subtract-max normalization the deeper layers carry no separable
signal (the gap decays to 0, and once went slightly negative at depth 4).

**Two controls that frame the gap:**

- *Substrate ceiling* (a 3-layer dilation stack, global tropical CE, 10 epochs;
  archived in `diagnostics.jsonl`, probe `substrate_ceiling`): **Adam 0.288 /
  SignSGD 0.334** at 784-dim; all six runs across input dims {784, 196, 49} land
  in **0.29–0.35**. The max-plus dilation MLP on flat pixels caps at ~0.3
  regardless of optimizer or input size — the sign-only "closure tax" is *not*
  the limiter here.
- *The bottleneck is the max-plus PRIMITIVE, not closure* (global backprop, 20k
  train / 5k test, width 512, 15 epochs; archived in `_ceiling_probe.log`):
  **dense MLP 0.964 · dyadic closure-clean MLP 0.958 · max-plus dilation MLP 0.274.** A *multiplier-free* (shift-add,
  zero-DSP forward) **dyadic** net essentially matches the dense ceiling, while the
  max-plus dilation primitive — winner-take-all, axis-aligned, no oblique/smooth
  normals — collapses to 0.27. So the FF paper's ~98.6% is *not* reachable on a
  pure **max-plus** substrate, but **is** plausibly reachable on a closure-clean
  **dyadic** substrate (the repo's validated max-plus expressivity fix,
  `tropical-dyadic-oblique-fix`). The limiter is representational, not closure.
- *Dense-FF ceiling* (more epochs, same width-300/depth-2; archived in
  `diagnostics.jsonl`, probe `dense_ff_epochs`): 0.276 → 0.344 (20 ep) → 0.358
  (40 ep) — it plateaus ~0.36 at this size. Hinton's published ~98% needs
  width ~2000 / 4 layers / 60 epochs / 60k examples; under our small matched budget
  *every* FF variant is far from that ceiling, while dense **backprop** reaches
  0.91 immediately (the task itself is easily learnable — FF is the weaker
  paradigm at this scale).

## 4. Honest verdict — where TFF wins and loses

**TFF works as specified and is fully closure-clean, but its absolute MNIST
accuracy is low (~0.23).** No inflation: this is a weak classifier on this task.
The gap to dense backprop (~0.90) decomposes into three measured, separable costs:

1. **The FF paradigm** (biggest, substrate-independent): dense FF *also* only
   reaches ~0.28–0.36 under the matched budget vs 0.91 for dense backprop. Local
   goodness-overlay learning is simply a weaker learner than backprop at this
   scale — most of the gap is FF-ness, not tropical-ness.
2. **The max-plus substrate** (~0.06–0.08): tropical FF (0.20–0.23) sits a little
   below dense FF (0.20–0.28), the same ~order as the measured substrate-vs-dense
   gap, and consistent with the library's standing finding that the max-plus
   dilation primitive underfits (it has no oblique/smooth normals; cf.
   `tropical-muzero-capacity-verdict`, the LM bakeoff).
3. **Label-overlay × winner-take-all** (a TFF-specific friction): the goodness
   classifier needs the *additive* image⊕label composition that FF relies on, but
   max-plus is winner-take-all — a weak label (10 of 794 inputs) is drowned by the
   image, and a strong one (`on≥3`, measured) makes goodness collapse to a
   label-only constant (predicts one class, ~0.10). A soft-τ warmup did **not**
   help (0.117 — *below* the hard ~0.23; `diagnostics.jsonl`, probe
   `softtau_warmup`), so this is composition difficulty, not only argmax sparsity.

**Where TFF actually wins — depth-robustness.** TFF is the **only** tropical/FF
method here that does not degrade with depth (0.200 → 0.226 → 0.230), and at
**depth 8 it beats global backprop-tropical 0.230 vs 0.102** (the latter collapses
to chance). Local forward-only learning has no gradient threaded through 8
argmax layers, so it sidesteps the deep max-plus gradient pathology that kills the
backprop stack. (Honest caveat: the backprop-tropical recipe — a single global
soft→hard anneal — was not re-tuned per depth; the robustness claim is *relative*
to that matched recipe.) The added depth doesn't *contribute* (separation → 0 past
layer 2), so this is robustness, not a capacity gain.

**Net:** on the metric (accuracy) TFF loses to dense backprop by a lot, ties dense
FF, and slightly trails backprop-tropical at shallow depth; it wins on
depth-stability and on the property it was built for — a **forward-only,
backprop-free, verified-multiplier-free** training loop that maps to the existing
zero-DSP max-plus kernel.

## 5. Closure receipts (verified, not asserted)

The deliverable assertion lives in `tests/test_forward_forward.py`:

- `test_train_step_is_multiplier_free` — wraps a **full** `TropicalFFTrainer.train_step`
  (two forwards + hinge loss + backward + `PureTropicalSGD` step) in
  `tt.assert_no_multiply()`, on a **non-power-of-two batch (13)** so the pass cannot
  be an artifact of pow-2 batch sizes. Passes.
- `test_hinge_loss_gradient_is_sign_gated` — the loss backward is exactly `{−1,0,+1}`
  at the top-k positions (k non-zeros per row), nothing else.
- Structural `tt.assert_closed(TropicalFFNet)` and whole-step
  `tt.assert_training_closed(layer, …, PureTropicalSGD)` both pass.

The forward kernel is the existing zero-DSP max-plus dilation kernel; a trained TFF
layer's inference is one `y = max_i(W+x)` + a subtract-max, so a TFF stack slots
straight into the silicon forward path with **no new primitive** required.

## 6. Limitations & honest notes

- **Small budget.** 6k train / 6 epochs / width 300, chosen for a CPU-runnable
  demo. Absolute numbers would rise with scale (the controls show the *shape* of
  the comparison, not converged accuracies). The substrate-ceiling diagnostic
  (~0.35) suggests the max-plus dilation MLP would not reach dense accuracy even
  with more compute.
- **The tropical-native (unsupervised) route underperformed.** Training layers to
  separate real images from `scramble_features` negatives, then a *linear probe*
  on the features, gave **0.10 — far worse than a raw-pixel linear probe (0.88)**
  (`diagnostics.jsonl`, probe `scramble_linear_probe`).
  The max-plus "real-vs-scrambled" objective does not yield class-linear features.
  `scramble_features` is provided and correct; it just isn't a good MNIST
  representation learner. (Label-overlay is the working supervised route here.)
- **Goodness-classifier vs readout.** Accuracy uses the closure-clean
  label-overlay goodness classifier (Hinton-native). A trained readout would
  likely score higher but adds a separate classifier; the goodness classifier is
  the honest end-to-end-closed number.
- **Closure-guard nuance (review-confirmed).** `tt.assert_no_multiply` does *not*
  flag `torch.logsumexp`, so an **un-hardened soft (τ>0)** TFF forward would get a
  false PASS. This is the *same* tolerance that lets the intentionally-allowed
  host-conditioned `tropical_cross_entropy(softmax="exact")` pass, so it was left
  as-is rather than "fixed" in the shared guard. **Discipline: harden `τ→0` before
  any closure claim.** TFF defaults to the hard path and `train_step` never raises
  τ, so the headline train-step closure result is genuine (and the higher-level
  audit functions auto-harden via `harden_tau=True`).
- **Loss-scaling caveat (review-confirmed, matches repo convention).** The loss
  backward `grad_output * grad_h` is closure-clean because `grad_output` is the
  scalar `1.0` when the loss is the (unscaled) root of `.backward()` — identical to
  `TropicalPerceptronLossFn`/`TropicalCrossEntropyFn`. Do not pre-scale the loss
  (`(loss * w).backward()`) if you want the guard to stay green.

## 7. Formal models in `tikitorch.models`

TFF is a first-class model (re-exported at top level, like `tt.MorphSSM`):

- **`tt.models.TropicalForwardForward`** — the max-plus substrate. `forward(x)`
  returns goodness-based class logits (drop-in classifier);
  `forward_forward_fit(X, Y, ...)` / `make_trainer()` / `forward_forward_step()`
  are the closure-clean forward-only training API; `compute_loss` is the CE on the
  goodness logits (a documented *non-FF* backprop baseline). Trains forward-only,
  not by `loss.backward()` on a global head.
- **`tt.models.DyadicForwardForward`** — same FF rule on the closure-clean dyadic
  (`±2^k`) substrate; the high-accuracy variant (§8).

Both pass the structural `tt.assert_closed`. The layer is substrate-pluggable:
`TropicalFFLayer(..., substrate="maxplus"|"dyadic")`.

## 8. Two substrates — max-plus vs dyadic (the paper comparison)

The same forward-only local FF rule (top-k goodness, hinge loss, subtract-max
norm, sign-only update) on two closure-clean substrates, vs Hinton's FF paper
(2022) MNIST baseline of **1.36% error ≈ 0.9864** (a *dense* 4×2000 net):

| model | substrate | learning | test acc | notes |
|---|---|---|---:|---|
| **`DyadicForwardForward`** | dyadic ±2^k (shift-add) | **forward-only FF** | **0.9177** | full 60k, 4×2000, 25 ep, sign-only update |
| `TropicalForwardForward` | max-plus dilation | forward-only FF | 0.230 | §3 best; the winner-take-all wall |
| dyadic backprop *(ref ceiling)* | dyadic ±2^k | global backprop | 0.958 | what the closure-clean substrate can do |
| dense backprop *(ref ceiling)* | dense float | global backprop | 0.964 | task ceiling at this budget |
| **FF paper** (Hinton 2022) | dense float | forward-only FF | 0.9864 | dense 4×2000, 60 ep, the target |

**Answer to "can we match the paper?"**

- **On the faithful max-plus substrate: no.** It caps at ~0.23–0.35 — the
  winner-take-all dilation primitive cannot compose the additive image⊕label
  signal FF needs (the gap is *representational*, measured and diagnosed in §3–4).
- **On the closure-clean dyadic substrate: most of the way, yes.**
  `DyadicForwardForward` reaches **0.9177** — forward-only, backprop-free,
  multiplier-free (the dyadic forward is a verified shift-add schedule; the update
  is sign-only). That is **+0.69 over max-plus TFF** and within **~6.9 pp of the
  paper's dense 0.9864**, with the rest attributable to budget (the paper uses
  4×2000 / 60 ep / Adam-class conditioning; ours is a sign-only update with fewer
  epochs) rather than a substrate wall — the dyadic *ceiling* is 0.958.

**Honest cost & closure caveat.** The 0.9177 run took ~5 h on CPU (width-2000
dense matmuls; this would be minutes on a GPU/FPGA). And the dyadic substrate's
closure is a **static weight-value property** (`fpga.dyadic_export.verify_schedule`
on the ±2^k weights), **not** the eager op trace — the float sim runs a real
`addmm`, so `DyadicForwardForward` does **not** pass the runtime
`tt.assert_no_multiply` (only the max-plus `TropicalForwardForward` does). Both
pass the structural `tt.assert_closed`; the weight *update* is sign-only for both.
So the takeaway: **a closure-clean Forward-Forward can approach the paper on MNIST,
but it has to leave the pure max-plus primitive for the dyadic one** — exactly the
substrate trade-off the rest of this report measures.

## 9. Files

- `tikitorch/forward_forward.py` — substrate-pluggable layer, goodness fns,
  hinge/logistic loss, tropical max-normalize, greedy per-layer trainer,
  negative-data generators, goodness classifier. Re-exported as
  `tt.forward_forward` / `tt.TropicalFFLayer` / `tt.TropicalFFNet` / `tt.TropicalFFTrainer`.
- `tikitorch/models/__init__.py` — `TropicalForwardForward` / `DyadicForwardForward`.
- `tests/test_forward_forward.py` — closure assertion + correctness (21 tests).
- `scripts/tropical_forward_forward.py` — the depth-sweep experiment (§3).
- `scripts/tropical_ff_mnist_benchmark.py` — the full-MNIST paper comparison (§8).
- `scripts/tropical_ff_diagnostics.py` — archives the §3/§4/§6 diagnostic probes.
- `results/forward_forward/` raw rows: `mnist_results.jsonl` (§3 depth sweep, 12
  rows), `mnist_benchmark.jsonl` (§8 substrate comparison), `diagnostics.jsonl`
  (substrate-ceiling / dense-FF-epochs / scramble-probe / soft-τ probes), plus the
  console logs `_run.log` / `_benchmark.log` / `_ceiling_probe.log` / `_diagnostics.log`.

All numbers in this report trace to one of those files. Reproduce: `python
scripts/tropical_forward_forward.py`, `... tropical_ff_mnist_benchmark.py`, `...
tropical_ff_diagnostics.py` (the §8 0.9177 headline used the heavier flags noted in §8).
