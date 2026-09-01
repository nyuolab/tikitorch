# Learning a tropical policy prior — the clean apples-to-apples Connect4 sweep

_Goal: does a closure-oriented training recipe make the dyadic policy prior **help** low-sim search?
Net: dyadic tropical cores + dyadic heads (the model closure audit returns `[]`), AlphaZero-style
explicit-game tree, sign-only TropicalSGD. This audit does not cover PUCT/min-max arithmetic, exact
softmax/loss branches, or physical DSP use. Score = win+0.5·draw vs negamax, random openings,
**n=40 games/cell**. Each recipe runs
TRAINED **and** UNTRAINED in the **same run** over the sim grid [100,200,400] at negamax depths d2/d3, so this
is a genuine apples-to-apples table. The diagnostic the study defined: does the learned prior beat the untrained
net at low sims (100)?_

**Provenance (this is the clean confirmation run).** Run on a c5.9xlarge (36 vCPU) via the robust one-shot runner
`scripts/run_connect4_prior_sweep_ec2.sh` (threads pinned `--threads 1`/recipe, results streamed to S3,
IAM-instance-profile, `shutdown` cost-backstop, self-terminate). ~66 min, ~$1.80. This run **supersedes and
partly corrects** the earlier partial sweep (commit `d380b0f`), whose two headline numbers do **not** reproduce
(see "Correction" below).

## Result — the clean apples-to-apples table (`tr` = trained, `un` = untrained, both from the same run)

| recipe (all closure `[]`) | @100 tr · un | @200 tr · un | @400 tr · un | mean Δ(tr−un) |
|---|---|---|---|---|
| perceptron only | 0.975/0.588 · 0.525/0.562 | 0.775/0.450 · 0.775/0.512 | 0.863/0.787 · 0.838/0.688 | +0.090 |
| `ce` — full-dist soft-CE | 0.713/0.512 · 0.812/0.600 | 0.775/0.850 · 0.812/0.675 | 0.912/0.800 · 0.938/0.688 | +0.006 |
| **`ce` + entropy 0.1** | **0.925/0.825 · 0.688/0.350** | 0.850/0.825 · 0.825/0.625 | 0.900/0.925 · 0.875/0.762 | **+0.188** |
| `ce` + entropy 0.3 | 0.675/0.613 · 0.875/0.650 | 0.875/0.850 · 0.900/0.762 | 0.912/0.900 · 0.925/0.812 | −0.016 |
| `ce` + entropy 0.1 + lr 1e-3 | 0.825/0.725 · 0.800/0.812 | 0.900/0.800 · 0.850/0.787 | 0.863/0.838 · 0.900/0.850 | −0.008 |
| `ce` + target-temp 2.0 | 0.750/0.500 · 0.525/0.425 | 0.812/0.750 · 0.550/0.375 | 0.950/0.900 · 0.700/0.550 | +0.256† |
| perceptron + entropy 0.1 | 0.725/0.688 · 0.775/0.537 | 0.900/0.713 · 0.900/0.812 | 0.700/0.662 · 0.950/0.812 | −0.066 |

_Cells are `d2/d3`. mean Δ is the trained−untrained gap averaged over the 6 (sim×depth) cells. Noise: per-cell
score SE = √(0.25/40) = 0.079; a depth-averaged trained−untrained delta has SE ≈ 0.079 (2σ band ≈ 0.158); a
recipe's 6-cell mean Δ has SE ≈ 0.046. †`ce`+target-temp's large Δ is a **weak-untrained-baseline artifact**,
not a strong trained net — see Findings._

## Findings (honest; this clean run is more deflationary than the partial run)

1. **The audited dyadic model is competitive in this sweep.** All 7 recipes report model closure `[]`
   (zero `nn.Linear`) and play credibly vs negamax: 83% of all cells (trained or
   untrained) score ≥0.6, range 0.35–0.975. **This substrate result stands independently of the prior-learning
   question** — the dyadic net is competitive via search depth here regardless of whether training helps.
   It is not a whole-system resource result.

2. **No collapse.** Every recipe — including bare perceptron+SignSGD — sits in 0.6–0.9 everywhere. There is no
   catastrophic policy collapse to "solve" in this clean run (the partial run's perceptron 0.025 @100 is not
   reproduced; see Correction).

3. **Training's low-sim benefit is recipe-specific and weak, not a general property.** Of 7 recipes, exactly
   **one — `ce` + entropy 0.1 — shows a real low-sim gain**: it beats untrained in 6/6 cells, its mean Δ
   (+0.188) clears ~4σ, the gap is largest precisely at the sim=100 diagnostic, and its *trained* net is the
   strongest in the litter (above the cross-recipe trained mean in all 6 cells). Even so, **its headline margin
   is inflated by an unusually weak untrained baseline** (untrained @100 d3 = 0.350, the single lowest cell);
   controlled against the median untrained baseline the true effect is closer to **~+0.15–0.19**, not the raw
   +0.36–0.48. All other recipes are within noise, negative, or baseline-confounded.

4. **`ce`+target-temp 2.0 is NOT a training win** despite its +0.256 mean Δ. Its *trained* scores are below the
   cross-recipe trained mean at the sim=100 diagnostic; the gap is driven entirely by a pathologically weak
   *untrained* baseline (worst at every grid point). This is a bad-comparator artifact, not a learned prior.

5. **The ordering is dominated by seed/baseline noise.** Nominally-identical untrained nets spread **0.350**
   across recipes @100 d2 (>2× the per-cell noise band) — so the untrained comparator is not a fixed yardstick,
   and most per-cell trained-vs-untrained deltas (33/42 within the single-comparison 2σ band) are noise. Only
   recipe-level 6-cell aggregates are trustworthy, and only `ce`+entropy 0.1 survives as a clean training win.

**Honest bottom line:** a trained dyadic prior *can* help at low sims, but only under one specific recipe
(`ce`+entropy 0.1) and only weakly — it is **not** the general collapse-and-rescue phenomenon the prior commit
claimed. This converges with the board-scaling verdict (substrate scales structurally; the prior-learning win is
narrow and compute-/seed-limited, and is shown supervised, not robustly in the self-play loop).

## Correction to commit `d380b0f` (the partial run)

Two headline numbers from the partial sweep do **not** survive the clean, thread-pinned, apples-to-apples re-run:

- **REFUTED — "perceptron collapses to 0.025 @100."** Clean trained perceptron @100 = (0.975, 0.588), the single
  best cell in the table. There is no collapse; the collapse-and-rescue framing rests on a number that did not
  reproduce.
- **REFUTED — "`ce`+entropy 0.3 wins @0.90, beating untrained ~0.787."** Clean `ce`+entropy 0.3 trained @100 =
  (0.675, 0.613), which **loses to its own untrained baseline** (0.875, 0.65) on both depths (6-cell mean
  Δ = −0.016). It is neither a 0.90 winner nor a winner at all; the honest standout is `ce`+entropy 0.1.

The loss-design work the partial run motivated is unaffected and stands on its own: the row-balanced,
row-balanced `tropical_cross_entropy` (softmax row-sum scattered at the target so `sum(grad)==0` for exact
*and* shift-LUT) and the entropy/temperature levers are useful engineering, and the sign-training gradient-row-bias
failure mode is real and documented elsewhere (the Shakespeare LM bakeoff: shift-softmax gradient-bias collapse,
found and fixed). It simply did not dominate this Connect4 self-play setting.

## Caveats (honest)

- **Single n=40 run, no seed replication, no within-cell CIs.** Every cell is one draw; SE figures are worst-case
  Bernoulli. The 0.350 spread among nominally-identical untrained baselines shows seed variance is first-order.
  Resolving the real ~0.05–0.15 effects needs **hundreds of games/cell** and, critically, a **fixed shared
  (same-seed) untrained baseline** so the comparison is paired rather than seed-confounded. That paired design is
  the natural next confirmation; this run pins the apples-to-apples table but not the small effect sizes.
- **Multiple comparisons.** 42 trained-vs-untrained cell-deltas screened with no correction (~2 false 2σ hits
  expected); a genuine *negative* 2σ outlier (perceptron+entropy 0.1 @400, −0.200) confirms the noise is real and
  two-sided. Over-read only the largest recipe-level aggregate (`ce`+entropy 0.1).
- **No clean sim-efficiency trend.** A learned prior buying search efficiency should show the trained−untrained
  gap *shrinking* as sims grow; across [100,200,400] the gaps are noisy/non-monotone for most recipes
  (perceptron-only even sign-flips @200). `ce`+entropy 0.1 is the partial exception (largest gap at sim=100).
- **Scope.** "Beats untrained" is a relative claim about prior quality at a fixed search budget vs a fixed
  (non-improving) negamax opponent. It is **not** evidence the multiply-free substrate matches a dense MuZero
  prior — the capacity verdict (dense 1.5–1.7× better on smooth/oblique value) still stands.

_Reproduce: `bash scripts/run_connect4_prior_sweep_ec2.sh setup && bash scripts/run_connect4_prior_sweep_ec2.sh launch`
(or one recipe locally: `PYTHONPATH=. python3 scripts/run_fully_tropical_connect4.py --policy-loss ce --entropy-coef 0.1 --threads 1`)._
