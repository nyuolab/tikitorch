# Making TikiTorch the definitive tropical deep-learning library

A landscape survey of tropical / max-plus ML and a gap-driven roadmap. Goal: the
one open, PyTorch-shaped library covering tropical **deep learning**, the
**classical** tropical ML toolkit, the **geometry/theory** utilities, and the
unique **zero-DSP silicon** backend — all `assert_closed`-verifiable.

## The landscape (with the load-bearing references)

**1. Theory — tropical geometry of neural nets.**
ReLU/maxout nets *are* tropical rational maps; decision boundaries are tropical
hypersurfaces; linear regions = vertices of Newton polytopes; depth ⇒ exponential
expressivity. (Zhang, Naitzat, Lim, *Tropical Geometry of Deep Neural Networks*,
ICML 2018; *Tropical Expressivity of Neural Networks*, 2024.) We already use a
ReLU↔MorphSSM duality theorem in the manuscript.

**2. Morphological neural networks (our core lineage).**
Dilation/erosion (max-plus/min-plus) layers, the Dilation-Erosion Perceptron,
morphological pooling, opening/closing. *Training Deep Morphological Networks as
Universal Approximators* (2025, arXiv:2505.09710) is the most actionable:
- **Theorem 3.2 (gradient collapse):** stacked pure max-plus layers reduce to a
  single morphological layer (tropical associativity), and *at most m parameters
  per layer have nonzero gradient* → deep morph nets barely train. **This is the
  theoretical root of our closure-cost waterfall and sequence-task stalls.**
- **Fix:** a learnable scaling `αᵢ` (diagonal) between morph layers breaks the
  collapse. **MPM (Max-Plus-Min) layer**: shared weights for dilation+erosion,
  λ=1/2, separate biases — a *universal approximator* (Thm 3.4). Hybrid
  (linear∘morph) is universal and subsumes ReLU/maxout (Thm 3.5).
- **Practical:** needs **large batches** (their Hybrid-MLP fails at batch 64,
  trains at ≥6400); residuals help; init Glorot + α=1 + bias so dil/ero means
  cancel. Results: MPM 94.7% / Hybrid ~97.4% MNIST.
- Pruning/compressibility is a known morph-net strength (Dimitriadis & Maragos;
  *Max-plus operators for filter selection & model pruning*, 2019; *Advances in
  training/pruning/shape constraints of Morphological NNs using Tropical
  Algebra*, 2020).

**3. Tropical attention (Hashemi et al., NeurIPS 2025, arXiv:2505.17190).**
Hilbert projective-metric scoring (ADOPTED as `score_variant="hilbert"`),
1-Lipschitz / scale-invariant, MHTA universally approximates tropical circuits /
DP algorithms, symmetric+asymmetric variants, a `tropical-gemm` backend, and an
11-task combinatorial-reasoning benchmark (Floyd-Warshall, SCC, convex hull,
quickselect, 3SUM, knapsack, bin packing, subset sum, ...).

**4. Adjacent multiply-free DL (the "below-dense-FMA" neighborhood).**
AdderNet (ℓ1-distance conv — note |x|=max(x,−x), a tropical op), ShiftAddNet
(shift+add), Log-Number-System nets, binary/ternary nets. TikiTorch's *algebraic
closure* + `assert_closed` is the unifying lens; these are the natural baselines
and the framing for FORGE_TikiTorch_competing_below_dense_FMA.

**5. Classical tropical ML (broaden beyond DL).**
Tropical SVM (LP, max-min tropical distance), tropical PCA (tropical polytope),
tropical k-means / hierarchical clustering over the projective torus, tropical
Grassmannian / phylogenetics. (Yoshida et al.; Maragos *ICASSP 2024 tutorial*;
the **TML package**, arXiv:2309.01082 — R/Julia, not PyTorch.)

**6. Software.** cuASR (CUDA semiring GEMM, min-plus), GraphBLAST (GPU semiring
graph), BTAS, PALMA (ARM SIMD tropical), **algebraic-arrays** (Python
NumPy/JAX/PyTorch semirings with smooth variants — closest existing PyTorch
tropical lib). None do tropical **training on zero-DSP silicon** — our moat.

## What TikiTorch has vs. what's missing

| area | have | gap to be definitive |
|---|---|---|
| primitives | linear/dilation/erosion, conv1d/2d, pooling, attention (dot/L∞/**hilbert**), value-pull/top-k, BatchShift, **TropicalLayerNorm/RMS**, RNN cell, quantized embed | **α-scale between morph layers** (the collapse fix), **MPM layer**, opening/closing, **DEP**, ℓ1/Adder layer, asymmetric attention |
| optimizers | SignSGD, TropicalSGD, PureTropicalSGD, (Lion) | DC/Convex-Concave for DEP; tropical structured **pruning** |
| models | MLP, **CNN**, RNN, Transformer, SSM, MoE — all `assert_closed` | Hybrid (closed) universal-approx config; tropical-attention algo-reasoning model |
| theory tools | duality theorem (paper) | `tikitorch.geometry`: Newton polytope, **linear-region count**, tropical-hypersurface / decision-boundary extraction |
| classical ML | tropical distance, k-means, nearest-centroid, **TropicalSVM** (max-margin), PCA (MDS) | tropical-polytope PCA; exact LP solver for TropicalSVM |
| benchmarks | MNIST/CIFAR/sMNIST/sCIFAR/ECG; **`tikitorch.algorithmic`** (CLRS-style: sort, running/cumulative-max, LIS, shortest-path) with verified solvers + OOD harness | closed-reasoner training to tropical-beats-softmax parity on the full combinatorial suite |
| backends | HLS zero-DSP (unique), PyTorch | fast max-plus GEMM (Triton / cuASR interop) |
| verification | **`assert_closed`** | — (our differentiator; keep) |

## Roadmap (prioritized)

**P0 — fix the training collapse (directly lifts every closed number).**
1. **Closed α-scaling**: a learnable *power-of-2* per-channel scale between morph
   layers (a shift, so closure-clean) — the multiplier-free analogue of the
   morph-NN paper's `α` that breaks the Theorem-3.2 gradient collapse. *(Built
   below.)*
2. **MPM layer** (`MaxPlusMin`): shared-weight dilation+erosion, λ=1/2, separate
   biases — universal approximator. *(Built below.)*
3. Document the **large-batch** requirement + ship large-batch benchmark configs
   (our small-batch runs under-state closed accuracy).
4. Cite Theorem 3.2 in the paper as the *theoretical* explanation of the closure
   penalty — turns a limitation into an understood mechanism.

**P1 — complete the primitive set: DONE.** `MorphOpenClose2d` (opening/closing),
`DilationErosionPerceptron`, `AdderLinear` (ℓ1/AdderNet taxonomy + baseline),
asymmetric attention (`score_variant="hilbert_asym"`), and `tikitorch.prune`
(`tropical_prune`, structured pruning of never-winning taps). Plus the P0 deep-
morph fixes `TropicalScale` + `MaxPlusMin`. (DC/Convex-Concave training for DEP
and 1-Lipschitz attention remain as refinements.)

**P2 — one-stop tropical ML: DONE.** `tikitorch.geometry`
(`count_linear_regions`, `decision_regions_2d`, `newton_polytope`) and
`tikitorch.classical` (`tropical_distance`, `tropical_kmeans`,
`TropicalNearestCentroid`, `tropical_pca`). The max-margin LP **TropicalSVM** and
tropical-polytope PCA remain as the documented next extensions.

**P3 — performance, benchmarks, positioning: IN PROGRESS.**

*Done.* `tikitorch.algorithmic` — a CLRS-style algorithmic-reasoning suite:
verified data generators + reference solvers for the tropical dynamic programs
(`gen_running_max`/`gen_cumulative_max` = max-plus scan, `gen_sort`, `gen_lis` +
`lis_lengths` = max-plus DP, `gen_shortest_path` + `bellman_ford` / `floyd_warshall`
= min-plus shortest-path closure), a closure-verified `TropicalReasoner` (and a
matched softmax baseline), and an `evaluate_task` harness with in-distribution
**and** OOD (length / magnitude) splits. Plus `tikitorch.classical.TropicalSVM`
(max-margin tropical classifier; multiplier-free at inference). Demo:
`scripts/run_algorithmic_reasoning.py`.

*Honest findings so far.* (a) The reference solvers agree (Bellman–Ford ==
Floyd–Warshall) — the ground truth is correct. (b) A **closed** (`assert_closed`-true)
max-plus reasoner executes the prefix-max DP and **extrapolates** to 8× the
training length (≈0.99). (c) Standard softmax attention is **OOD-fragile**: it fits
sorting in-distribution (≈0.93) but collapses at 2× length (≈0.11). We do **not**
yet claim tropical-beats-softmax on the harder combinatorial tasks: at present the
closed reasoner under-fits sorting in-distribution, mirroring the sMNIST stall and
Theorem 3.2. Reproducing the Hashemi et al. (NeurIPS 2025) wins on the full
Floyd-Warshall/knapsack/3SUM suite (relative positional encodings, their recipe)
is the next step — the infrastructure to benchmark it now ships here.

*Update (P3 round 2).* Added, after an adversarial audit of the attention stack:
- **Iterative-argmax winner-exclusion top-k attention** (`agg_variant="iter_topk"`):
  pick best key, exclude, repeat k times, SUM-combine (+ learnable additive
  per-rank bias). Strictly multiplier-free (no `/cnt` divide), denser gradient
  than top-1, soft-anneals to the hard sum. `tikitorch.functional.tropical_value_pull_iter_topk`.
- **Closed relative-position reasoner** (`pos_mode="rope_shift"`): wires
  `TropicalShiftAttention(shift_mode="rope_relative")` into `TropicalReasoner`. Its
  ramp `p·g` is built by *cumulative addition* (was a real multiply — a genuine
  closure bug the audit caught), so it is multiplier-free AND length-extrapolates.
- **`assert_closed_runtime`**: traces a forward via `TorchDispatchMode` and flags
  any real multiply/divide (pow-2 = shift), the *sufficient* check the structural
  `assert_closed` could not provide (it had missed the rope multiply).
- Fixes: dilation Q/K/V identity-init (value-collapse), `-inf` on fully-masked
  top-k rows, harness `score_variant` default, `no_grad` eval.

*Honesty correction.* An earlier draft claimed "softmax collapses OOD on sort";
the audit proved that was an untrained-absolute-positional-embedding artifact
shared by *both* arms, not a softmax property. Corrected everywhere. The
defensible result is **magnitude OOD** (softmax saturates ~23x; tropical is
scale-invariant). On `sort`, the tropical reasoner still under-fits in-distribution
even with rope_shift+iter_topk+identity-init — a clean tropical-beats-softmax
demonstration is **not yet achieved** and remains the open item.

*Update (P3 round 3 — shipped).* The paper-faithful combinatorial DP tasks are in:
`gen_subset_sum` / `gen_min_coin_change` / `gen_knapsack` with brute-force-verified
solvers (`subset_sum_selection`, `min_coin_table/selection`, `knapsack_table/selection`),
a per-token `micro_f1` metric, and a closed multi-feature input embed
(`input_kind="multiscalar"`, add-only) so `evaluate_task("knapsack", ...)` trains
`assert_closed`-true. Also landed the general-use refinements: `tt.save`/`tt.load`
(state_dict + constructor config + closure manifest), `tt.profile_model` +
`python -m tikitorch <Model>` (prints `multiplications: 0` for a closed model),
the closure-aware `tt.Trainer` (tau curriculum + deploy-time `assert_closed`),
GitHub Actions CI + an import-safety test, and `TROPICAL_BELOW_FMA.md` (the
AdderNet/ShiftAdd/LNS taxonomy with `assert_closed` as the unifying check). Full
suite: **304 passed / 1 skipped**.

*GPU assessment DONE (2026-06-07, A10G, 3 seeds, head-matched, train→2× size-OOD;
`scripts/run_algorithmic_gpu.py`).* HONEST VERDICT: **the closed tropical reasoner
does NOT beat softmax** on these tasks — tie on subset_sum (0.71 vs 0.70 OOD-F1),
competitive on knapsack (0.85 vs 0.92) and running_max (0.95 vs 1.00), under-fits
sort (id 0.49 vs 0.996; both collapse OOD ~0.1, a positional issue), and **fails
min_coin_change outright (0.00 vs 0.40)**. So we do NOT reproduce the
Tropical-Attention paper's advantage with this reasoner. Two concrete leads: (a)
the min_coin 0.00 is a likely **semiring mismatch** — it's a *min*-plus DP but the
reasoner aggregates *max*-plus (`iter_topk`); add a min-plus aggregation variant;
(b) the in-distribution under-fit on sort/min_coin is the gating problem.

*Min-plus aggregation — built, but the semiring-mismatch hypothesis did NOT pan
out (honest negative result, 2026-06-07).* Added `agg_variant="min"` (erosion
value pull, the (min,+) twin via `-max(-.)`) and `"maxmin"` (split-head MaxPlusMin,
both semirings), with semiring-correct causal masking (max-plus masks future keys
with -inf; min-plus with +inf — min would otherwise *select* the masked key). All
closure-clean + gradchecked (314 tests). But 3-seed data shows it does **not** fix
min_coin_change (min 0.09±0.13 vs iter_topk 0.09±0.12 — a wash; min_coin training
is bimodal/unstable, not semiring-bound; the earlier 1-seed 0.25 was noise), and
the effect elsewhere is **task-dependent with no clear winner**: maxmin slightly
helps running_max OOD (0.94→0.997) but min-plus *collapses* subset_sum OOD
(0.69→0.01, overfits to train size). So: kept as correct, semiring-complete
primitives, but they do NOT make tropical beat softmax. The real blocker for the
hard tasks is **training stability + the in-dist under-fit**, not the aggregator.

*Remaining.* (1) min_coin/sort: attack training STABILITY (longer schedules,
init, class-imbalance / focal loss for the 16%-positive selection targets) — the
aggregator is not the lever; (2) Triton/cuASR max-plus GEMM backend (GPU-perf only
— low priority; removes the O(n^3) intermediate-materialization bottleneck, changes
nothing about correctness/closure/silicon); (3) an AFI rebuilt from the bundled
`forge_train_step` kernel to recover a live on-silicon loss curve (the registered
AFIs are layout-rebuilds; `model.fpga()` training itself is already verified on F2
— 99.5% of params evolve via on-board sign-SGD).

**Large-batch finding (load-bearing).** Fully-closed sign-SGD sequence models
(sMNIST/sCIFAR) stall at chance with small batches and only train at large
*effective* batch (gradient accumulation: `sign(Σ grads)`): sMNIST 0.11→**0.40**,
sCIFAR →**0.37** at effective batch 384-512. Confirms the morph-net large-batch
requirement (arXiv:2505.09710).

## Level-B closure: the training loop, not just inference (2026-06-07)

Decision: the target is **Level B — full closure with no compromises** (forward +
backward + optimizer all multiplier-free / zero-DSP, the FORGE on-silicon vision),
**stronger than the Tropical-Attention paper**, which is NOT closed (it uses
`log`/`exp` I/O, ordinary multiply-bearing transformer layers, and AdamW on a GPU;
"tropical" there = the *geometry*, not the hardware).

Consequence: the `evaluate_task` GPU proxy was itself **not** Level B (Adam +
`exp`-anneal + cross-entropy). So all prior "under-fit" numbers were a non-closed
proxy. New tooling + a closed path:
- **`tikitorch.audit.assert_training_closed` / `find_training_multiplies`** — trace
  the WHOLE training step (fwd + loss + backward + optimizer) for multiplies. Novel
  (prior closure checks were inference-only). It immediately caught real Level-B
  breaks: SignSGD's float momentum (`buf.mul_(0.9)`), the need for **power-of-2
  `d_model`** (so the mean-subtract norm is a shift), and Adam/CE.
- **`evaluate_task(..., closed=True)`** — hard forward (no anneal), perceptron loss
  (±1 grad) / L1, `PureTropicalSGD` (bit-shift lr, integer momentum). The weight
  update is verified multiplier-free (the only residual is the float loss-VALUE's
  sign-invariant `/n`, which is host-side, not on-chip).

**Honest 3-seed Level-B numbers (id / ood-2×; f1 for the combinatorial DPs):**

| task | tropical CLOSED (Level B) | tropical non-closed proxy | softmax |
|---|---|---|---|
| running_max | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| subset_sum | 0.73 / 0.65 | 0.80 / 0.78 | 0.78 / 0.71 |
| knapsack | 0.82 / 0.77 | 0.87 / 0.88 | 0.91 / 0.90 |
| sort | 0.20 / 0.09 | 0.39 / 0.11 | 0.99 / 0.13 |
| min_coin_change | **0.31 / 0.20** | 0.08 / 0.08 | 0.76 / 0.49 |

Verdict (correcting an over-eager interim claim that "closed trains better"): it is
**mixed**, not a uniform win. Closed is **much better on min_coin** (0.31 vs 0.08,
3-seed-confirmed — the hard/temperature-free recipe genuinely helps that crisp DP),
**tied on running_max**, but **modestly worse on sort / subset_sum / knapsack**
(5-9 pts) than the Adam+anneal proxy — a real, small **closure tax** on those.
Closed-tropical does **not** beat softmax on any task. So Level B is *achievable
and now verifiable*, with a modest accuracy cost on most tasks and one task where
hard-closed is clearly best. Next levers stay: training **stability** (large batch,
α-scale, class-imbalance loss for the 16%-positive selection targets).
*(Update: the P4 RESOLVED study below tightened this to ≈1pp on the across-task mean
with 5 seeds — the 5-9pt per-task figure was largely seed/recipe variance.)*

## P4 — Tropical optimizers (the lever for the closure tax)

The 3-seed Level-B study localized the closure tax to the **optimizer**, not the
architecture: the non-closed proxy's edge on sort/subset_sum/knapsack comes from
**Adam's adaptivity + the soft anneal**, both of which we drop for zero-DSP. So the
highest-leverage closed-DL research direction is a *better multiplier-free
optimizer* — one that recovers Adam-like adaptivity using only compare/add/shift.
All must pass `assert_training_closed` (the new gate).

Have: `SignSGD` (sign-only; **float momentum is NOT closed** — use `momentum=0`),
`TropicalSGD`, `PureTropicalSGD` (bit-shift lr + **integer-counter momentum** —
the verified-closed default), plus `Lion` (sign-based).

Explore (each must trace clean under `assert_training_closed`):
1. **Closed adaptive rates** — per-parameter learning rate as a *bit-shift chosen
   from a sign-L1 / sign-streak counter* (the `adaptive=True` path in
   `PureTropicalSGD` is currently EXPERIMENTAL/not-closed; make it closed). A
   "tropical Adam": second-moment as an exponent (shift amount), not a float.
2. **Integer/LFSR momentum + stochastic rounding** — denser, lower-variance sign
   estimates without a float EMA (addresses the τ=0 gradient-sparsity pathology
   from the optimizer side, complementing large-batch majority-vote sign).
3. **Sign-streak / count-based step sizing** — grow the shift when a parameter's
   gradient sign is consistent, shrink on flips (a closed Rprop analogue).
4. **Pure-tropical Lion** — verify/close the momentum interpolation (it likely has
   the same float-momentum multiply as SignSGD).
Goal: a closed optimizer that closes the 5-9pt Level-B tax on sort/subset_sum/
knapsack. Benchmark via `evaluate_task(closed=True)` + the algorithmic suite.

*Built: `TropicalAdam` (2026-06-07) — and an honest negative result.* A closed
**sign-streak Rprop** optimizer: per-parameter power-of-2 step, doubled on
`streak` consecutive same-sign grads, halved on a sign flip; update is
`p -= sign(grad)·2^-shift`. Verified zero-DSP (`assert_training_closed` clean,
`tt.TropicalAdam`, `evaluate_task(closed_opt="tropicaladam")`, 4 optim tests),
converges on a toy, and the step provably stays a power of two. BUT 3-seed +
streak∈{1,2,4} sweep: it is only **competitive with `PureTropicalSGD`, not a win**
(id/ood — running_max 0.97/0.99, sort 0.21/0.10, subset_sum 0.73/0.62, knapsack
0.80/0.59 @streak2 but 0.83/0.80 @streak1, min_coin 0.23/0.18) — no setting closes
the tax, and neither closed optimizer reaches the non-closed Adam proxy or softmax.
**Diagnosis:** sign-only methods (SignSGD/Rprop/TropicalAdam) can't access Adam's
*magnitude*-aware scaling (`1/√v`), and tropical gradients are *sparse* (a param is
updated only when it wins an argmax), so per-param sign-flip history is unreliable.
**Revised next direction — a closed MAGNITUDE proxy:** scale each param's pow-2
step by the *integer count of agreeing examples in a large batch* (`sign(Σ grad)`
already used for the update; the count is a closed magnitude surrogate → bucket to
a shift). That, not sign-streak, is the candidate to actually close the tax — or it
confirms the tax is intrinsic to sign-only optimization (itself a paper-worthy
finding).

### RESOLVED (2026-06-07) — the closure tax was ≈1pp all along

Built **all four** designed closed adaptive optimizers (each `assert_training_closed`-
clean, run twice; step magnitude from `torch.frexp` + a constant pow-2 table gather,
never `2**tier`/`ldexp`): `FrexpMagnitudeSGD` (frexp-tier magnitude, EMA-normalized),
`CountAgreementSGD` (the count-based magnitude proxy above; sign-agreement over an
`accumulate()` window, accum≥8), `ExponentEMAAdam` (counter momentum + frexp-exponent
2nd-moment EMA), `PerLayerStepSGD` (Rprop on a scalar per-tensor shift). All exported,
wired into `evaluate_task(closed_opt=...)`, 35 optim tests green.

Bake-off: 120-cell 3-seed grid → a 6-agent adversarial verification workflow → a
definitive 5-seed confirmation at short (50-step) and full (250-step) budgets
(`scripts/bakeoff_closed_optimizers.py`; `results/confirm_{short50,full250}.md`).

**Findings (honest, cross-checked by two independent adversarial skeptics):**
1. **The "closure tax" is ≈1pp, not 5-9pp.** Full-budget 5-seed across-task mean:
   `PureTropicalSGD` (closed) 0.521 ID / 0.490 OOD vs the non-closed Adam proxy
   0.531 / 0.497 — within seed noise; closed *wins* on running_max (+24pp) and
   min_coin (+19pp; Adam gets 0.002 there). The earlier 5-9pp figure was a
   recipe/seed artifact. **Full closure is essentially free.**
2. **At full budget `PureTropicalSGD` is still the best closed optimizer** — the
   adaptive ones don't beat it asymptotically (the honest negative stands for the
   asymptote).
3. **At short budget (≤50 steps — the on-silicon regime) the adaptive ones WIN:**
   `FrexpMagnitudeSGD` ≥2× step-efficient (running_max 0.78 vs 0.44 at 50 steps);
   `CountAgreementSGD` (accum≥8) is the stability play (knapsack 0.778±0.01 vs the
   baseline's 0.711±0.13). So they ship with a real, characterized niche.
4. **The residual gap to softmax is architectural** (sort: 0.92 vs ~0.12), not an
   optimizer gap. Instability is bimodal + task/init-bound and afflicts the baseline
   too; the accum≥8 window is the best variance reducer.

**Default policy:** `PureTropicalSGD` all-purpose; `FrexpMagnitudeSGD` for tight /
silicon budgets; `CountAgreementSGD` (accum≥8) when DP-task stability matters.
