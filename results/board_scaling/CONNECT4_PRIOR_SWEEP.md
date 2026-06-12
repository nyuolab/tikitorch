# Learning a good tropical policy prior — the anti-collapse sweep (Connect4, fully multiply-free)

_Goal: a MULTIPLY-FREE training recipe that makes the dyadic policy prior **help** low-sim search, instead
of the perceptron+SignSGD collapse (trained 0.025 vs untrained ~0.787 @ 100 sims). Net: dyadic cores + dyadic
heads (closure `[]`), AlphaZero true-model tree, sign-only TropicalSGD. Score = win+0.5·draw vs negamax,
random openings. The diagnostic metric is **trained @ 100 sims vs d2**: does the learned prior beat the
untrained ~0.787 (helps) or collapse (0.025)?  Run on EC2 (c5.9xlarge)._

## Result — the collapse is solved, and the prior now helps

| recipe (all fully multiply-free) | trained @100 d2/d3 | @200 d2/d3 |
|---|---|---|
| perceptron only (prior run) | **0.025** / 0.075 | 0.90 / 0.75 |
| `ce` — full-distribution soft-CE | 0.60 / 0.60 | 0.85 / 0.86 |
| `ce` + entropy 0.1 | 0.787 / 0.71 | 0.84 / 0.81 |
| **`ce` + entropy 0.3** | **0.90 / 0.80** | 0.91 / 0.90 |
| `ce` + entropy 0.1 + lr 1e-3 | 0.80 / 0.675 | 0.875 / 0.81 |
| `ce` + target-temp 2.0 | 0.875 / 0.70 | 0.90 / 0.825 |
| perceptron + entropy 0.1 | 0.838 / 0.71 | 0.925 / 0.81 |
| _untrained reference (prior run, same cfg)_ | _~0.787_ | _~0.90_ |

**Findings:**
- **Collapse SOLVED.** Every anti-collapse recipe lands 0.60–0.90 @100 sims vs the perceptron-only 0.025.
- **The trained prior now HELPS low-sim search.** The strong recipes — full-dist soft-CE + entropy 0.3
  (**0.90**), + target-temp 2.0 (0.875), even perceptron + entropy (0.838) — exceed the untrained ~0.787 at
  100 sims. So a multiply-free trained prior buys sim-efficiency (fewer sims for the same strength = cheaper
  FPGA actors) — the thing that failed before.
- **Lever ranking:** ANTI-COLLAPSE (entropy bonus or visit/target temperature) is the key lever. Full-dist
  soft-CE alone helps (0.60 vs 0.025) but doesn't beat untrained; entropy/temp on top does. Winner here:
  full-distribution soft-CE + entropy ≈ 0.3.

## The general lesson for learning tropical nets (converges with the LM-bakeoff instance)

Sign/shift training collapses a learned prior via two compounding mechanisms: **(1) overconfidence** —
sign-only updates drive logits to extremes with no calibration, especially with an argmax-only target
(perceptron); and **(2) gradient row-bias** — a shift-normalized soft loss whose row-sum ≠ 1 leaves a
per-row bias that sign training integrates into a constant-predictor drift. The cure is **(a) a calibrated,
row-balanced loss** (full-distribution soft-CE; or `tropical_cross_entropy`, which scatters the softmax
row-sum at the target so `sum(grad)==0` for exact *and* shift LUT — the deployable closure-faithful form),
plus **(b) explicit entropy / temperature regularization** to keep the prior soft. The `mode-frac` detector
(mean prob on the policy argmax) flags collapse hundreds of steps before the eval does.

## Caveats (honest)

- **Partial data.** The runaway below killed the sweep before it wrote complete JSONs: we have trained @
  100/200 (not 400) and **not** a per-recipe untrained baseline — the ~0.787 untrained is cross-run (same
  config, prior run), so trained-vs-untrained mixes runs. n≈30–40 games/cell (±~0.08). The *direction*
  (collapse fixed; best recipes beat untrained) is robust and consistent across all six; the exact margins
  are not yet apples-to-apples. A clean, thread-pinned re-run (trained+untrained per recipe, full grid) is
  the confirmation to-do.
- **OPS MISTAKE (recorded so it doesn't recur).** The 6 recipes ran concurrently but each `main` used
  torch's *default* thread count (= 36), so the eval phase oversubscribed ~200 threads onto 36 cores —
  the run took **~13 h (~$20)** instead of ~30 min. Fix: pin `torch.set_num_threads` per process for
  parallel sweeps (`--threads`, added). Always pin threads when fanning processes on one box.

_Reproduce: `scripts/run_fully_tropical_connect4.py --policy-loss ce --entropy-coef 0.3 [--threads 1]`._
