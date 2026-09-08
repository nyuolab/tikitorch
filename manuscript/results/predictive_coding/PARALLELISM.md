# Local dyadic PC: measured scheduling tolerance and a hardware opportunity

**Date:** 2026-06-20 (re-scoped after an adversarial 4-lens review + async-SGD rebuttal;
supersedes the earlier "synchronization-free → 2.3–3.9× over backprop" framing).
Evidence: the async error-relaxation results in `SCALE_TESTS.md` (committed, multi-seed),
the measured round-tax in `scripts/pc_staleness_frontier.py`, and the code in
`tikitorch/nn/predictive_coding.py`. Framing per Jaderberg et al. 2017 (locking) and the
EqProp / in-memory-compute lineage (Scellier–Bengio; memristor crossbars).

> **Correction (2026-09-06):** state-settling `settle_nudge` measurements cited
> here came from an implementation that did not apply its configured relaxation
> factor `gamma` and therefore used a unit step. Treat those accuracy and
> scale-fragility numbers as historical pending rerun. The cascade and
> error-relaxation measurements are unaffected.

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
   that genuinely has no directional sweep (state-settling `settle_nudge`) **diverged in
   the archived unit-step runs** and capped at ~85% of backprop; corrected-gamma
   results are pending. Current validated evidence therefore does not establish
   both properties in one method.

So the defensible thesis is narrower: a measured **local-message scheduling** result
plus a **hardware-substrate opportunity**, not a GPU-throughput or completed-hardware
claim.

## What genuinely survives the async-SGD rebuttal

The validated cascade/error-relaxation path does **not** eliminate the backward
dependency: it computes `E_l = A_l^T E_{l+1}`, the same directional credit relation as
backprop. What changes is how that relation can be represented and scheduled:

- **No separately learned feedback matrix.** `_route_T` and the cascade/error-relax
  steps reuse the forward layer's parameter tensor. The mathematical transpose action
  remains; code-level tensor reuse should not be mistaken for proof that a physical
  implementation needs no reverse datapath or data movement.
- **Credit is neighbour-local in space.** Each relaxation update communicates only
  between adjacent layers. Partial, stale, and arbitrary-order schedules converge in
  the measured frozen-routing experiments, although information still has to propagate
  directionally through depth.
- **Updates use local dyadic operands.** Once errors are available,
  `ΔW_l = q(ε_l) ⊗ q(x_l)` uses layer-adjacent quantities and ±2^k operands. That
  network-internal kernel is shift-add-oriented. Ordinary loss, optimizer, and other
  recipe operations are outside this claim.

These properties motivate an **analog / in-memory / crossbar implementation path**:
local state, reused parameters, and dyadic operands may reduce communication and
multiplier cost on an appropriate substrate. The repository has not demonstrated that
end-to-end mapping on such hardware. Only state settling is structurally capable of
removing an ordered reverse phase; its corrected-gamma accuracy and hardware closure
remain open, and the software solver currently includes exact diagonal division.

What we explicitly **concede**: async-SGD/Hogwild parallelize the **data/batch axis** and
tolerate staleness — PC does **not** beat them there. PC's distinct contribution is on the
**depth/credit representation and scheduling axis**, which is orthogonal.

## Backprop's locks — which PC breaks, stated honestly

| lock (Jaderberg 2017) | backprop | our PC, honestly |
|---|---|---|
| **forward** (layer l waits for l−1) | yes | **kept** — there is a forward pass |
| **update** (no layer updates until the whole backward + barrier completes) | yes | **relaxed after credit arrives** — `ΔW_l` is local and layer updates are independent once their errors are available |
| **weight transport** (separate feedback parameters) | transpose action uses forward parameters | **no separate learned feedback matrix** — the same parameter tensor is reused, while the transpose action and a physical reverse path still remain |
| **backward** (directional end-to-end credit sweep) | yes | **only the state-settling form removes it**; archived unit-step runs trailed ~15% and were scale-fragile, while corrected-gamma results are pending. The accuracy-matching error-relaxation *relaxes* (tolerates async/partial/stale execution of) the sweep but does not eliminate the directional dependency |

The precise, defensible headline is therefore: **PC expresses the backward credit
relation as neighbour-local updates, without a separately learned feedback matrix, and
the measured implementation tolerates asynchronous, partial, and stale execution** —
not "PC has no backward sweep."

## What is measured (vs modeled)

**Measured (the credit signal survives the conditions parallel/async hardware imposes):**

- **Sync == cascade == backprop, bit-exact** (`SCALE_TESTS.md`; unit test asserts
  `torch.equal`) — the local/iterative form computes the *identical* gradient when run
  synchronously, so nothing is given up in the easy case.
- **Async = sync within ~0.1–0.6pp** under 50% participation + stale reads + random update
  order, d4–d32, full MNIST, multi-seed — and under fair, sufficiently long updates the
  triangular/nilpotent frozen-routing system converges to the exact cascade/backprop
  credit.
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

  - **Positive:** every configuration that reached the threshold approached the
    **exact** backprop gradient (cosine → 1.000) under partial participation, lagged
    reads, and no global update order. The triangular/nilpotent structure explains
    convergence under fair updates; the `>128` cells are censored, not measured passes.
  - **Falsifier (honest):** the round-tax **grows ~O(L / participation)** — it is *not*
    bounded. Since backprop's avoided backward sweep is also O(L), the async round-count
    *cancels* the depth advantage: there is **no free, depth-growing throughput speedup**.
    (The earlier draft's "speedup grows with depth" is refuted by our own measurement.)
    Caveat that keeps it from being purely negative: accuracy-adequate training tolerated
    the default `n_iter ≈ L` (async accuracy within 0.1pp of sync; `SCALE_TESTS.md`) —
    full cosine-0.999 convergence is stricter than accuracy needs — so the *practical*
    tax for accuracy is ~O(L), the same order as the backward sweep, not the O(L/p) of the
    exact-gradient tax.

  **Net:** async error-relaxation buys **scheduling freedom with correct credit** in the
  converged runs, at a round-count that is **O(L), not free**. That supports a clockless-
  fabric implementation hypothesis; it is **not** a throughput or deployment result.

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

- The neighbour-local scheduling and dyadic internal operands are
  **structural/code-verified** properties. Their proposed mapping to analog/in-memory
  settling hardware is an opportunity, not proof of end-to-end deployability or a
  wall-clock/energy result. A recipe-specific closure audit and real-hardware
  measurements are still required.
- All async-robustness is **simulated on a single GPU** (participation coin + permutation +
  stale snapshots), MNIST-only for the full-scale confirmation. It demonstrates the credit
  signal's tolerance, not a deployed async system.
- The accuracy-matching form is, by our own bit-exact result, **backprop credit**; the
  genuinely-different equilibrium learner's archived unit-step runs were scale-fragile,
  and its corrected-gamma scale behavior remains to be measured. The
  value under current evidence is **schedule tolerance plus a substrate hypothesis**,
  not proof that it runs where backprop cannot or a free speedup on a GPU.
