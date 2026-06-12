# Connect4 — fully multiplier-free net + multiply-free training (the honest result)

_Run on EC2 (c5.9xlarge, 36 cores). Net: dyadic tropical CORES **and** DyadicLinear heads — closure audit
returns `[]` (fully zero-DSP, no nn.Linear). Training: AlphaZero-style true-model self-play (256 sims,
40×48 games) distilled into the dyadic policy via the closure-faithful **perceptron** loss (argmax of
MCTS visits) + **L1** value + sign-only **TropicalSGD**. So forward, backward, and update are all
multiplier-free. Score = win + 0.5·draw vs negamax, random openings, 40 games._

| sims | TRAINED d2 / d3 | UNTRAINED d2 / d3 |
|---|---|---|
| 100 | **0.025 / 0.075** | 0.787 / 0.700 |
| 200 | 0.900 / 0.750 | 0.900 / 0.738 |
| 400 | 0.875 / 0.725 | 0.975 / 0.838 |

## What this shows (and the negative result)

1. **The fully-tropical net is real and competitive via depth.** `closure_violations == []` (multiplier-free
   confirmed on-box), and the **untrained** net + AlphaZero search **crushes negamax d2/d3** (0.79→0.98
   vs d2 as sims rise). Competitive Connect4, fully multiply-free, is achieved — by the *search*.

2. **The multiply-free TRAINING recipe (perceptron + SignSGD) COLLAPSES the policy — it makes the agent
   WORSE.** Trained vs untrained is catastrophic at low sims (**0.025 vs 0.787** at 100 sims) and no better
   at high sims (0.875 vs 0.975 at 400). The trained dyadic policy head learns an **overconfident, wrong
   prior** that actively misleads low-sim PUCT search; only at 200+ sims does the search explore enough to
   override it. A *good* prior should help low-sim play, not destroy it.

## Why (diagnosis)

The perceptron loss distills only the **argmax** of the visit counts (discarding the distribution), and
**sign-only SignSGD** drives the logits to extreme magnitudes — together they produce a **peaked,
miscalibrated** policy prior. When that confident prior is wrong, PUCT wastes its few (100) simulations on
the wrong move. This is the same "training narrows the policy" pathology seen earlier with the hybrid net
(CE was marginal/noisy), but **worse** here because argmax-only + sign-only is maximally overconfident.

## So: how DO we learn a good tropical policy?

Honest state after this run: **none of the recipes tried learn a policy prior that beats uniform+search.**
The AlphaZero search carries the strength; the net's prior is, at best, neutral and at worst harmful. The
clear next levers (all keep it multiply-free), as a falsifiable to-do, not a claim:
- **Distill the FULL visit distribution, not the argmax** — a closure-faithful soft loss (e.g.
  `TropicalCrossEntropyFn`'s `(shift_softmax − onehot)` gradient) preserves calibration the perceptron throws away.
- **Anti-collapse**: entropy/KL-to-uniform regularization or a visit-count temperature so the prior stays soft.
- **Gentler updates**: sign-only SignSGD with no lr decay over-sharpens logits; an integer lr_shift schedule
  (decay) or magnitude-aware-but-still-closed update may stop the over-confidence.
- **Or lean on the VALUE head** (which guides search more robustly than a collapsed policy prior) and treat
  the policy prior as secondary.

Bottom line: **fully multiply-free + competitive (via depth) is done; a multiply-free TRAINING recipe that
*improves* the prior is still open** — and this run rules out naive perceptron+SignSGD as that recipe.

_Reproduce: `scripts/run_fully_tropical_connect4.py`. Data: `connect4_fully_tropical.json`._
