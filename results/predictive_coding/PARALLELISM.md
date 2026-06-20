# The actual value of legit PC: a credit signal that runs where backprop physically can't

**Date:** 2026-06-20 (re-scoped after an adversarial 4-lens review + async-SGD rebuttal;
supersedes the earlier "synchronization-free → 2.3–3.9× over backprop" framing).
Evidence: the async error-relaxation results in `SCALE_TESTS.md` (committed, multi-seed),
the measured round-tax in `scripts/pc_staleness_frontier.py`, and the code in
`tikitorch/nn/predictive_coding.py`. Framing per Jaderberg et al. 2017 (locking) and the
EqProp / in-memory-compute lineage (Scellier–Bengio; memristor crossbars).

## The reframe (and the correction the review forced)

This effort established, honestly, that on **accuracy** our PC ≈ backprop (the cascade /
sync error-relaxation *are* backprop via Z-IL; `SCALE_TESTS.md`). So accuracy is **not**
the value proposition. The earlier version of this doc claimed the value was a
**synchronization-free throughput speedup** (2.3–3.9×). An adversarial review showed that
framing does not survive scrutiny, for two concrete reasons:

1. **Staleness-tolerance is not PC-unique.** Async-SGD, Hogwild!, Downpour, and
   PipeDream/1F1B already hide pipeline latency and train backprop under bounded-stale
   gradients. A throughput model that pits PC against **strictly synchronous** backprop
   (as the old table did) is measuring against a strawman; the speedup it shows is mostly
   the latency that *async-backprop also hides*.
2. **"Matches backprop" and "breaks the backward lock" come from different methods.** The
   accuracy-matching learner (cascade / sync error-relaxation) is **bit-exact equal to
   backprop credit** and therefore *has* a directional top-down dependency; the method
   that genuinely has no directional sweep (state-settling `settle_nudge`) **diverges at
   scale** and caps at ~85% of backprop. You cannot claim both properties in one breath.

So the defensible thesis is narrower — and it is a **hardware-substrate** claim, not a
GPU-throughput claim.

## What genuinely survives the async-SGD rebuttal

The one thing async-backprop categorically **cannot** do is eliminate the **backward
graph itself**. Every async/stale/pipeline backprop variant still computes a full
backward pass with a **transpose-weight (or autograd-stored) datapath**; they parallelize
*where/when* the backward runs, never *whether it exists*. Our learner removes it, and
this is true in the code today:

- **No weight transport.** The error is routed through the **same** forward `±2^k`
  weight, not a separate transposed copy: `_route_T` and the cascade error step reuse
  `self.layers[l].weight` directly. (Backprop's `W` and `Wᵀ` are two objects that must be
  kept in sync — "weight transport," biologically and physically implausible.)
- **No separate error-carrying datapath; credit is local-in-space.** `E_l = A_l^T E_{l+1}`
  is a **neighbour-only** message-passing fixpoint over cached routing — not a global,
  all-to-all gradient object that must be routed end-to-end before any update is correct.
- **Update-local + multiplier-free.** Once errors are settled, `ΔW_l = q(ε_l) ⊗ q(x_l)`
  uses only layer-`l`-adjacent quantities (all `L` updates mutually independent), and
  every operand is `±2^k` (shift-add, no DSP multiply).

Together these make the learner runnable on **analog / in-memory / crossbar settling
hardware** (the EqProp / memristor regime): the same physical conductances that run the
forward MVM carry the credit signal, the energy-relaxation *is* the inference, and there
is no global backward controller to implement. **Backprop has no physical realization on
that substrate at all** — regardless of how you parallelize it. *That* is the axis no
competing PC work (all FP32, transpose-weight) and no async-backprop method occupies.

What we explicitly **concede**: async-SGD/Hogwild parallelize the **data/batch axis** and
tolerate staleness — PC does **not** beat them there. PC's distinct contribution is on the
**depth/credit axis** (local credit, no backward graph), which is orthogonal.

## Backprop's locks — which PC breaks, stated honestly

| lock (Jaderberg 2017) | backprop | our PC, honestly |
|---|---|---|
| **forward** (layer l waits for l−1) | yes | **kept** — there is a forward pass |
| **update** (no layer updates until the whole backward + barrier completes) | yes | **broken** — `ΔW_l` is local; given settled errors, all L updates are independent, no global barrier |
| **weight transport** (Wᵀ datapath, kept in sync with W) | yes | **broken** — same forward `±2^k` weight carries the error |
| **backward** (directional end-to-end credit sweep) | yes | **only the state-settling form removes it — and that form trails ~15% and is scale-fragile**; the accuracy-matching error-relaxation *relaxes* (tolerates async/partial/stale execution of) the sweep but does not eliminate the directional dependency |

The precise, defensible headline is therefore: **PC decomposes the backward credit-
routing into purely-local, weight-transport-free neighbour updates that tolerate
asynchronous, partial, and stale execution** — not "PC has no backward sweep."

## What is measured (vs modeled)

**Measured (the credit signal survives the conditions parallel/async hardware imposes):**

- **Sync == cascade == backprop, bit-exact** (`SCALE_TESTS.md`; unit test asserts
  `torch.equal`) — the local/iterative form computes the *identical* gradient when run
  synchronously, so nothing is given up in the easy case.
- **Async = sync within ~0.1–0.6pp** under 50% participation + stale reads + random update
  order, d4–d32, full MNIST, multi-seed — and the async gradient provably **converges to
  the exact backprop gradient** (cosine → 1.000), because the frozen-routing system is
  triangular/nilpotent.
- **Round-tax** (`scripts/pc_staleness_frontier.py`, `staleness_frontier.{json,png}`,
  2026-06-20): the new measurement that stops booking asynchrony's cost as zero — how many
  local message-passing rounds async needs to reach the **exact** backprop gradient
  (cosine ≥ 0.999) vs the single ordered sync pass. Two findings, one positive, one that
  **falsifies the optimistic throughput claim** (residual MLP, MNIST, 6 trials):

  | condition | d8 | d16 | d32 | d64 | law |
  |---|---|---|---|---|---|
  | **sync** | 1 | 1 | 1 | 1 | one ordered pass = backprop |
  | **stale** (lag only, full participation) | 8 | 16 | 32 | 64 | **= L** |
  | **partial 50%** | 12 | 24 | 48 | 96 | **1.5·L** |
  | **50% + stale + random-order** | 32 | 48 | 96 | >128 | ~4–6·L |
  | **25% + stale + random-order** | 48 | 96 | >128 | >128 | doesn't converge ≤128 |

  - **Positive:** the async gradient *always* converges to the **exact** backprop gradient
    (cosine → 1.000) under every condition tested — partial participation, lagged reads,
    no global update order. Async never converges to the *wrong* credit; the triangular/
    nilpotent structure guarantees it. So the credit assignment is correct and robust to
    exactly the conditions a clockless, partially-participating substrate imposes.
  - **Falsifier (honest):** the round-tax **grows ~O(L / participation)** — it is *not*
    bounded. Since backprop's avoided backward sweep is also O(L), the async round-count
    *cancels* the depth advantage: there is **no free, depth-growing throughput speedup**.
    (The earlier draft's "speedup grows with depth" is refuted by our own measurement.)
    Caveat that keeps it from being purely negative: accuracy-adequate training tolerated
    the default `n_iter ≈ L` (async accuracy within 0.1pp of sync; `SCALE_TESTS.md`) —
    full cosine-0.999 convergence is stricter than accuracy needs — so the *practical*
    tax for accuracy is ~O(L), the same order as the backward sweep, not the O(L/p) of the
    exact-gradient tax.

  **Net:** async error-relaxation buys **scheduling freedom with correct credit** (any
  order, any lag, partial participation → still the exact backprop gradient), at a
  round-count that is **O(L), not free**. That is a substrate-enablement property (it runs
  correctly on a clockless async fabric), **not** a throughput win.

**Modeled (now demoted to an explicitly-bounded illustration, not a headline):** the
pipeline-throughput model (`scripts/pc_throughput_model.py`) is a dependency-structure
algebra with hand-set constants (τ_f=1, τ_b=2, τ_u=0.3) and a zero-latency-for-PC
assumption. Sensitivity sweeps show the *direction* (flat-in-latency, grows-with-depth)
survives any reasonable constants, but the *magnitude* does not: τ_b=1,τ_u=1 → 1.0× (no
advantage); a fair "PC pays half the latency" term collapses the latency-bound case from
3.9× to ~2.8×; and the model never included an **async-backprop** comparator (PipeDream/
1F1B), against which most of the modeled gap evaporates. So the throughput multiplier is
quoted only as "modeled vs synchronous backprop, floor ~1.7×, magnitude unreliable" —
the substrate claim above, not this number, is the contribution.

## Honest scope

- The substrate advantage (weight-transport-free, credit-local, multiplier-free → runs on
  analog/in-memory settling hardware) is a **structural / code-verified** property, **not**
  a wall-clock measurement on real neuromorphic hardware. Real-hardware energy/throughput
  on such a substrate is the next claim to substantiate.
- All async-robustness is **simulated on a single GPU** (participation coin + permutation +
  stale snapshots), MNIST-only for the full-scale confirmation. It demonstrates the credit
  signal's tolerance, not a deployed async system.
- The accuracy-matching form is, by our own bit-exact result, **backprop credit**; the
  genuinely-different equilibrium learner (state-settling) is the scale-fragile one. The
  value is **substrate-enablement** (runs where backprop can't), not free speedup on a GPU.
