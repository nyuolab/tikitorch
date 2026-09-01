# Tropical Predictive Coding — depth study

**Date:** 2026-06-15
**Code:** `tikitorch/nn/predictive_coding.py` (`TropicalPCLayer`, `TropicalPCNet`),
tests `tests/test_predictive_coding.py`, repro `scripts/pc_depth_probe.py`.

## What this is

A local, backprop-free, **multiplier-free** learning rule for the tropical
semiring — predictive coding (Rao–Ballard / Whittington–Bogacz) ported to
max-plus / min-plus layers. Each edge predicts its output state from the state
below with the tropical primitive `mu[j] = (max|min)_i(W[j,i] + x[i])`; the
prediction error `eps = x - mu` is **tropical division** (the ⊗-inverse of the
semiring's `+`, i.e. a subtractor — closure-clean, the semiring only lacks the ⊕
/ max inverse, which PC never needs); credit assignment rides the **cached
argmax** (`S^T` is a `scatter_add` over the same indices the forward pass
produced — no transposed weights, no weight transport, no multiply).

Two depth levers are built in: **forward-init** (latents start at the
feed-forward pass; interior errors begin at 0 so only the clamped top carries
signal) and optional **incremental PC** (`learn_every`). Depth is made
representationally real by **alternating dilation/erosion** — a pure max-plus
stack collapses to a single max-plus-linear map, whereas min-plus∘max-plus is a
genuine tropical-rational map.

**Closure verified:** a full settle + local-update step runs **zero hardware
multiplies** (`tt.assert_no_multiply()` passes — adds, compares, scatters,
power-of-two shifts only). This is the property that maps the rule onto the
silicon-ready multiplier-free forward kernel.

## Headline result

Two separate questions, two answers:

1. **Can tropical PC train *deep* (optimisation)?** **Yes** — identity-initialised
   interior layers train to depth 32 with no degradation, where the naive
   baseline collapses to chance by depth ~8.
2. **Does depth buy *accuracy* on real data?** **No — and the limit is the
   tropical primitive, not PC.** Even full backprop + Adam with a soft→hard
   anneal (the best possible rule) caps deep alternating tropical at < 0.5 on
   MNIST and *degrades* with depth (0.49→0.37→0.12 for depth 2/4/8), while a
   dense MLP solves it at 0.91 at every depth. A max-plus/min-plus layer is
   information-lossy, so stacking compounds the loss instead of building
   features.

So the depth study succeeds as an optimisation result and lands a clean negative
on *pure max-plus* depth as a route to quality.

3. **Does fixing the primitive fix the quality — and can training be fully
   multiplier-free?** **Yes to both.** Swapping the lossy max-plus mixing for a
   **dyadic ±2^k shift-add** mixing + ReLU (`DyadicPCNet` — a multiplier-free
   oblique MLP) makes depth pay off: the local PC rule trains flat at **~0.88–
   0.91 to depth 12**, matching the dyadic backprop ceiling within noise. And the
   **fully multiplier-free weight update** — quantizing *both* the error and the
   activity to ±2^k so the weight-grad contraction is a shift-add tree —
   **matches Adam** (~0.89/0.92/0.88 at depth 2/4/8). The naive pure-±1 sign rule
   lagged at ~0.65–0.68; keeping the *magnitude* (within 2×) closed the gap. So
   forward, inference, AND training are multiplier-free with no accuracy cost.
   See "Path 2" below.

4. **Can we go genuinely deep (toward ResNet)?** **Yes — the depth wall is
   solved.** Iterative settling still walls at ~depth 48 (a PC-settling problem;
   backprop on the same net is fine). The fix exploits a uniquely-tropical fact —
   **hard routing is discrete and cacheable** — to replace O(depth) settling with
   a **single-pass cached-routing cascade** backward. Fully multiplier-free, it is
   **flat from depth 16 to 100 (0.84–0.90), matching backprop** (and fast: depth
   100 in ~30 s). See "Path 4" below.

Details and mechanism below.

## The failure, and the mechanism

Naive (uniform-init) deep tropical PC collapses to chance at depth ≥ 8. Holding
a recipe that trains perfectly at shallow depth (4-class separable blobs, d=24,
fixed target, gamma=2, 10 settle steps) and sweeping depth:

| depth | uniform-init test acc |
|------:|----------------------:|
| 2 | 0.92 |
| 3 | 1.00 |
| 5 | 0.97 |
| 8 | **0.26** (chance) |
| 12 | **0.26** |

Instrumenting per-layer error (L1, input-side → output-side) at depth 12 shows
why — the top-clamp error never reaches the early layers:

```
[0, 0, 0, 0, 0, 0, 0.001, 0.002, 0.004, 0.007, 0.013, 0.437]
 \__________ bottom (no signal) __________/        top (clamp)
```

The injected error decays ~2× per layer and the bottom six layers see exactly
zero. Two corrections to the original theory, recorded honestly:

- This is **not** the unit-gain routing first claimed: the settling update has a
  `1/gamma` damping *and* propagates ~one layer per sweep, so with finite sweeps
  the signal attenuates before reaching the bottom. The 0/1 Jacobian keeps the
  error *magnitude* bounded (logits stay in ±2 even at depth 12 — no
  vanish/explode), but bounded ≠ delivered.
- The deeper problem is **representation collapse**: random deep tropical layers
  destroy class information before it reaches the readout, and the layers that
  would fix it get no learning signal (chicken-and-egg).

## The fix: identity init (lever isolation)

Square interior layers, four recipes, 8 epochs:

| depth | A uniform,g2,s10 | B identity,g2,s10 | C identity,g2,s3·d | D identity,g1,s3·d |
|------:|-----------------:|------------------:|-------------------:|-------------------:|
| 5 | 0.670 | **0.978** | 0.973 | 0.895 |
| 8 | 0.235 | **0.985** | 0.980 | 0.967 |
| 12 | 0.235 | **0.985** | 0.985 | 0.967 |
| 16 | 0.233 | **0.985** | 0.995 | 0.970 |

**Identity init alone (B) carries the whole effect** — flat accuracy with depth.
Extra settle steps (C) and less damping (D) add nothing. Identity init works
because (1) the deep stack starts as a near-identity map so the readout sees real
features regardless of depth, and (2) error routes straight down the diagonal at
true unit gain. Flatness confirmed to depth 32 on blobs (depth 8/16/24/32 all =
**0.985**).

## The honest limitation: real-data quality

MNIST (12×12 = 144-d, 6000 train, 3 epochs, square interior):

| depth | uniform-init | identity-init |
|------:|-------------:|--------------:|
| 2 | 0.376 | 0.352 |
| 4 | 0.261 | 0.198 |
| 8 | 0.113 | **0.275** |
| 12 | 0.089 | **0.275** |

Identity init does its job — it **prevents the depth collapse** (0.089 → 0.275 at
depth 12). But the **absolute accuracy is bad** (≤ ~0.38 vs ~0.92 for a linear
classifier). This is a *capped* ceiling, not undertraining: at depth 2, raising
the learning rate (1/16, 1/32) leaves accuracy stuck at chance (0.087, flat over
12 epochs) — higher lr destabilises, lower lr (1/128) gives the best ~0.35.

Diagnosis: with identity init the interior layers stay ≈ identity (their local
errors are tiny, so the update barely moves them) — depth is **inert**. The
effective classifier is a single max-plus readout on raw pixels, and a max-plus
output selects exactly **one** input via argmax, so it can only key off
individual pixels. Weak by construction.

## Why the interior can't learn: hard routing is bistable

Attempt to wake the inert interior by shrinking the **identity gap** (how far
below the diagonal the off-diagonals start) + tie-break noise. Depth 8, MNIST,
sweeping the gap:

| identity gap | test acc | active % (off-diagonal argmax winners) |
|-------------:|---------:|---------------------------------------:|
| 30 | 0.211 | 0.0 |
| 8 | 0.211 | 0.0 |
| 4 | 0.211 | 0.0 |
| 2 | 0.211 | 0.0 |
| **1** | **0.087** | **97.8** |
| 0.5 | 0.087 | 97.9 |

There is **no sweet spot — the interior is bistable.** Gap ≥ 2: the diagonal
always wins the argmax → the layer is frozen as a permutation, off-diagonal
weights never get a gradient (`active%`=0, inert). Gap ≤ 1: off-diagonals start
winning → `active%` jumps to 98%, but the deep winner-take-all dynamics
immediately scramble the representation → **collapse to chance**. The transition
is sharp because the argmax is all-or-nothing. **Hard max-plus routing cannot be
*gently* plastic under a local rule** — it is either frozen or chaotic. (Depth at
gap=2 confirms the frozen regime gains nothing: acc 0.27/0.32/0.21/0.17 for depth
2/4/8/16, `active%`≈0 throughout.)

## Where the ceiling comes from: the primitive, not PC

To separate "PC's local rule" from "tropical features", train a deep alternating
dilation/erosion stack with the **best** rule — full backprop + Adam + soft→hard
tau anneal (the repo's intended training mode; NOT closure-clean, a diagnostic).
MNIST 12×12, 8000 train, 12 epochs:

| depth | tropical (soft→hard) | tropical (hard, from scratch) | dense MLP |
|------:|---------------------:|------------------------------:|----------:|
| 2 | 0.491 | 0.087 (chance) | 0.939 |
| 4 | 0.366 | 0.087 | 0.914 |
| 8 | 0.117 | 0.087 | 0.912 |

- **Dense solves the task at ~0.91 at every depth** — the task is easy, depth is
  fine for a normal net.
- **Hard tropical from scratch is untrainable** (chance at all depths, even with
  Adam): the argmax gives gradient to a single input.
- **Soft→hard tropical caps at 0.49 and *degrades* with depth.** The best rule
  available still cannot make deep tropical features work.

Conclusion: the low ceiling is a property of the **max-plus/min-plus primitive**
(information-lossy; depth compounds the loss), not of predictive coding. This
matches the §2a capacity verdict (max-plus underfits) and the LM bakeoff finding
that tropical attention is the broken piece. PC was never the bottleneck — it
trains shallow tropical to 0.98 (blobs), multiplier-free and local.

## Path 2: fix the primitive — the dyadic shift-add fix (it works)

The ceiling probe says the limiter is the max-plus *mixing* (one input survives
each max). Replace it with a **dyadic** mix — `DyadicLinear`, weights snapped to
±2^k via STE so `W·x` is a shift+add tree (multiplier-free) — plus ReLU. That is
an oblique, non-lossy, multiplier-free MLP.

**Step 1 — does the primitive survive depth (backprop+Adam, MNIST 12×12)?**

| depth | dyadic (±2^k) | dense | pure max-plus (soft→hard) |
|------:|--------------:|------:|--------------------------:|
| 2 | 0.938 | 0.939 | 0.491 |
| 4 | 0.923 | 0.914 | 0.366 |
| 8 | 0.890 | 0.912 | 0.117 |
| 16 | 0.677 | 0.117 | — |

Dyadic ≈ dense to depth 8 and far outlasts max-plus. (At depth 16 both dyadic
and dense fall — the generic plain-MLP depth wall, no residuals — and dyadic
actually beats dense there.)

**Step 2 — can the LOCAL PC rule train it deep?** First attempt collapsed at
depth ≥ 4 (chance) — but the cause was settling instability, not a barrier:

| γ (settle step) | sweeps | lr | iPC | depth-4 acc |
|----------------:|-------:|---:|----:|------------:|
| 0.50 | 8 | 1e-2 | – | 0.087 (collapse) |
| 0.10 | 40 | 1e-2 | – | 0.839 |
| 0.10 | 40 | 1e-3 | – | **0.909** |
| 0.05 | 60 | 1e-3 | 10 | 0.911 |

A small enough settling step (γ≈0.1, not 0.5 — the deeper energy's curvature)
with sweeps scaling as ~10·depth trains it. **Depth curve, local PC with the
ADAM weight update** (single seed; the Adam update is a real multiply — see
closure below):

| depth | dyadic PC (Adam) | dyadic backprop ceiling |
|------:|-----------------:|------------------------:|
| 2 | 0.906 | 0.938 |
| 4 | 0.909 | 0.923 |
| 8 | 0.908 | 0.890 |
| 12 | 0.883 | *not measured* (≈0.84 interp.) |

Accuracy is **flat with depth**. At depth 8 the local rule **matches the
backprop ceiling within noise** (0.908 vs 0.890, single seed, and the two runs
use slightly different budgets so this is "on par", not a robust "beat"). The
depth-12 ceiling was **not measured** — the dyadic backprop probe only ran
depths 2/4/8/16 (0.938/0.923/0.890/0.677); ≈0.84 is an interpolation, so we make
no depth-12 "beats backprop" claim. The takeaway that *is* supported: the local
PC rule stays flat to depth 12 where it would otherwise need residuals, and a
plain dense MLP collapses by depth 16 (0.117).

**Step 3 — make the weight update multiplier-free too.** Adam's update
`(relu'⊙(mu−x))·xᵀ` is a real outer-product multiply. Two multiplier-free
surrogates (`DyadicPCNet.set_mf_grads`, `multiplier_free=True`), both quantizing
the operands so every product is a shift and the contraction is an add-tree:

| depth | mf "dyadic" (magnitude-aware) | mf "sign" (pure ±1) | Adam | backprop ceiling |
|------:|------------------------------:|--------------------:|-----:|-----------------:|
| 2 | **0.892** | 0.654 | 0.906 | 0.938 |
| 4 | **0.919** | 0.674 | 0.909 | 0.923 |
| 8 | **0.881** | 0.675 | 0.908 | 0.890 |

- **`mf_mode="sign"`** — pure ±1 sign-product `sign(dE/dW[j,i]) = sign(mu_j −
  x_{l+1,j}) ⊕ sign(x_{l,i})` (note `mu−x`, i.e. `−sign(eps)`; per-sample identity
  verified exactly, `test_dyadic_sign_product_identity_and_ternary`). It learns
  but lags (~0.65–0.68): the per-sample sign is exact, but the batch sum-of-signs
  is a lossy majority vote that disagrees with the magnitude-weighted gradient
  sign on ~25–30% of weights. (Smaller batches made it *worse*, not better;
  `PureTropicalSGD` integer momentum also hurt, 0.13–0.18.)
- **`mf_mode="dyadic"` (recommended)** — quantize *both* the error and the
  activity to ±2^k (`q(v)=sign(v)·2^round(log2|v|)`, a leading-bit extraction in
  HW), so each product is `±2^(k₁+k₂)` (a shift) and the contraction is a
  shift-add tree. Keeping the magnitude within 2× makes `sign(Σ q·q)` track the
  true-gradient sign closely — it **matches Adam** (~0.88–0.92) with **no
  multiply and no accuracy cost**, in fewer epochs. This is the proper *full
  dyadic backward* (both forward and weight-grad multiplier-free).

**Closure (honest), per part.**
- Forward + inference **deploy** multiplier-free — `W·x` / `Wᵀ·error` with ±2^k
  weights are shift+add, `relu'` is a 0/1 mask. Weight-VALUE property verified
  statically (`test_dyadic_pc_layer_deploys_quantized_weights`); the float-sim
  op-trace legitimately shows matmuls (as for `DyadicLinear`).
- The **weight update** (`mf_mode` dyadic or sign) quantizes its operands to
  ±2^k / ternary, so the contraction is a shift-add tree on fabric (the float-sim
  shows `aten.mm`; closure is the quantized-operand property,
  `test_dyadic_sign_product_identity_and_ternary`).
- The one remaining sim-only multiply is the settling **step** `states -= γ·g`
  with γ=0.1 (non-power-of-two; γ=0.5, the nearest shift, diverges → needs a
  shift *schedule*). The energy's `**2` is likewise a sim artifact — on fabric
  the settling routes the error (a subtractor) directly, so the square never
  deploys.

## Path 3: tropical-native stability — pushing the depth wall

Dyadic PC still has a depth wall — plain dyadic PC collapses to chance by ~depth
24 (and is already down to 0.52–0.72 at depth 16, settling-budget dependent; see
the matched depth-scaling table further below for 0.524 at n_steps=96). This is
the no-residual wall that Qi et al. 2025 fight with precision-weighting +
residual-energy control, and μPC with Depth-μP. Their fixes are *multiplicative*
(precision matrices, √depth scaling). Since the tropical semiring is the log
domain (⊗ = +), the natural duals are *additive* / shift-based and so
multiplier-free. Tested at depth 16 (MNIST 12×12, Adam probe, n_steps=128):

| mechanism | depth-16 acc | note |
|-----------|-------------:|------|
| baseline (none) | 0.717 | the wall (n_steps=128) |
| **tropical residual** `max(relu(Wx), x)` | **0.909** | a free `max` — wall gone |
| discounted residual `max(relu(Wx), x−1)` | 0.713 | over-discounted; skip killed |
| max-shift norm `x − max(x)` | 0.117 | collapsed — reintroduced the target-scale bug |
| power-of-two precision `2^l·ε_l` | 0.087 | collapsed — 2^l gain blew up settling |

The **tropical residual wins decisively**, and it scales — depth 16/24/32, with
both Adam and the fully multiplier-free dyadic-backward update (plain dyadic PC
shown for contrast):

| depth | residual (Adam) | residual (multiplier-free) | plain (Adam) |
|------:|----------------:|---------------------------:|-------------:|
| 16 | 0.905 | 0.869 | 0.524 |
| 24 | 0.906 | 0.854 | 0.087 (chance) |
| 32 | **0.909** \* | **0.856** \* | 0.096 † |

\* depth-32 residual re-settled at `n_steps=200` (the d16/d24 rows are the
`n_steps=min(6·depth,120)` scale run). † plain at depth 32 was *not* re-run at
200 — 0.096 is the undersettled `n_steps=120` value, so the d32 row is not a
matched-budget comparison for the plain column (it is at depths 16/24).

Both the Adam and the fully multiplier-free update stay **flat to depth 32**
(0.91 / 0.86) where plain dyadic PC is at chance by depth 24. Note the depth-32
result took a correction: a first pass with a capped `n_steps=120` gave only
0.571, which **looked** like a new wall — but it was **undersettling**, not a
barrier. PC inference needs O(depth) sweeps; with `n_steps≈6·depth` (200) the
residual recovers to 0.909 (Adam) / 0.856 (multiplier-free), 0.913 at 300
sweeps. So:

| depth-32 settling sweeps | residual (Adam) acc |
|-------------------------:|--------------------:|
| 120 (undersettled) | 0.571 |
| 200 (~6·depth) | **0.909** |
| 300 | 0.913 |

So a single `max(relu(Wx), x)` — tropical addition of the identity path, a
comparison, no multiply — **keeps dyadic PC flat to depth 32** (0.91) where plain
dyadic PC collapses to chance, given enough settling. It works *because* `max` is
bounded and idempotent: residual energy can't accumulate the way an additive
float skip does — exactly Qi's "residual energy propagates too fast" failure,
solved natively and multiplier-free. Exposed as `DyadicPCNet(residual=True)`.

Direction A (the last sim-only multiply): the settling **step** `states -= γ·g`
with **γ=2⁻³=0.125 (a right shift)** trains as well as the non-power-of-two 0.1
(depth 16: 0.907 vs 0.905; γ=2⁻⁴ also fine — only γ=2⁻¹=0.5 was too big). Now the
default, so the settling step is multiplier-free; the deploy datapath is shift-add
end to end (the remaining op-trace multiplies — energy `**2`, dyadic matmul — are
the documented sim artifacts that deploy as a subtractor / ±2^k shift-add).

Honest negatives: the two other duals **did not help** — max-shift normalization
collapsed training (reintroduced the target-scale pathology: logits shifted below
the fixed one-hot target → no discrimination), and even a *gentle* capped (4×)
precision schedule slightly **hurt** at depth 32 (0.888 vs 0.909) once settling
was adequate; the un-capped `2^l` schedule diverged outright. So the residual +
enough settling + a shift-step carry the result; precision-weighting and
normalization, as tried, are not needed (and norm needs a margin target to work
at all). Parked, not load-bearing.

## Path 4: the cached-routing cascade — the depth wall, solved

Even with the residual, **iterative settling has a PC-specific wall**: trained
with the standard settle loop (width 64, MNIST), dyadic PC matches a same-arch
*backprop* baseline to depth ~32 but then **collapses — 0.21 at depth 48, chance
at 64 — while backprop on the identical residual net is flat to depth 64**. So
the architecture supports depth; PC's *iterative settling* is what fails (the
problem Qi et al. 2025 / μPC address in float).

The uniquely-tropical fix exploits the fact that **tropical routing is discrete
and cacheable**, which float PC cannot do. Forward once, caching the hard routing
(ReLU masks, residual `max`-gates); then route error top-down in a **single
O(depth) pass** (the Z-IL single-ordered-pass equivalence) instead of O(depth)
iterative sweeps. It is fully multiplier-free:

- error propagation `delta @ W` with ±2^k weights is **shift-add even for real
  `delta`** (real × 2^k = exponent shift) — so error flows *exactly*, with no
  unstable settling map to iterate;
- residual skip-wins are **unit-gain highways** (error passes straight down where
  the skip won the `max`);
- only the weight-grad outer product is quantized (dyadic ±2^k, or ternary
  sign) — a shift-add / sign-XOR weight grad.

**Result — flat to depth 100, matching backprop, fully multiplier-free** (width
64, MNIST 12×12, 15 epochs):

| depth | iterative settling | **cascade (multiplier-free)** | backprop ref |
|------:|-------------------:|------------------------------:|-------------:|
| 16 | 0.866 | 0.866 | 0.868 |
| 32 | 0.836 | 0.890 | 0.867 |
| 48 | **0.209** (wall) | **0.896** | 0.844 |
| 64 | **0.105** (chance) | 0.878 | 0.817 |
| 100 | — | **0.841** | 0.850 |

The cascade is **flat from depth 16 to 100** (0.84–0.90), matching backprop at
every depth (and beating it at 48/64), where settling walls at 48. It is also
**fast** — depth 100 in ~30 s (single pass), vs settling's hundreds of seconds
*and* failure. A bonus tropical synergy: at the very deepest, the **sign**-based
weight update is *more* depth-robust than Adam on the same cascade grads (Adam is
confused by the dyadic grad's magnitude drift; the sign ignores it).

Exposed as `DyadicPCNet.cascade_grads` / `train_step_cascade`. Honest caveats: it
is the *single-pass* (Z-IL) form of PC, not the iterative-relaxation form — i.e.
local, multiplier-free backprop enabled by cached hard routing; and the numbers
are width-64, single-seed, 4k-subset MNIST (an MLP, not a conv ResNet). But the
core claim holds: **the deep-PC depth wall is solved on the tropical substrate,
multiplier-free, via hard-routing.**

## Path 5: can we keep the ITERATIVE relaxation (not just the cascade) at depth?

The cascade is single-pass; we also tried to make the genuinely *iterative*
(equilibrium-reaching) PC work at depth, to keep its prospective-configuration /
inference advantages. Extensive exploration (Jacobi, Gauss-Seidel, diagonal-
preconditioned GS, damped, annealed step, cascade-warm-start "skip", momentum,
two-grid) — and the honest result is a **double obstacle**, not just a slow solver:

1. **Stiffness.** Settling is a stiff 1-D chain BVP; conditioning ~depth², so the
   equilibrium needs ~O(depth²) iterations. Measured: depth 48 needs 960+ sweeps
   to reach only 0.67 (still creeping); depth 64 doesn't converge in any practical
   budget. The cheap solvers either don't accelerate it (GS no faster than Jacobi)
   or accelerate by taking big steps that **leave the prospective-config basin**
   (preconditioned/momentum → worse fixed point). The best cheap one was your
   **skip** (cascade-warm-start), 0.39 @ depth 48 — helps but doesn't close it.
2. **The hard-clamp equilibrium isn't backprop.** A *decisive* test — settle with
   a strong inner solver (Adam on the latents) + the **exact** autograd PC
   equilibrium update — gives 0.60 @ depth 48 / chance @ 64 vs backprop 0.79/0.62.
   So even a perfect solver wouldn't reach backprop: with a **hard** label clamp
   the PC equilibrium update is a *different* (and, on deep classification, worse)
   update than backprop. (Theory: PC = backprop only in the *small-nudge* limit.)

| depth | iterative eq (strong solver) | backprop |
|------:|-----------------------------:|---------:|
| 48 | 0.60 | 0.79 |
| 64 | 0.11 (chance) | 0.62 |

**Reconciling insight.** The cascade is *not* a competitor to iterative PC — it
**is** iterative PC in the small-nudge / Z-IL regime collapsed to its exact single
pass, i.e. precisely the regime where PC matches backprop. So *backprop-matching
iterative tropical PC already exists — it is the cascade* (depth 100). Iterating
harder (hard clamp, full relaxation) moves **away** from backprop into the
prospective-configuration regime — a different update whose value is on continual-
learning / inference, **not** deep-classification accuracy, and which is practical
only to depth ~32. So "iterative PC matching backprop at deep classification" was
the wrong target: it asks the prospective-config regime to beat backprop on
backprop's home turf, where it is both stiff to compute and inherently different.

**Standing result for the iterative regime:** it works to **depth ~32** (0.84 =
backprop, true equilibrium, keeps PC's advantages, multiplier-free) — already
4–5× past the standard 5–7-layer PC wall. Deeper, use the cascade.

**Does the iterative regime *earn its keep*? (prospective configuration test.)**
We tested the canonical PC advantage — less catastrophic forgetting — on
permuted-MNIST (`scripts/pc_continual.py`), iterative `train_step` vs the cascade,
same net/optimizer/data. The result is an honest **tradeoff, not a win**:
iterative PC is consistently **more stable** (lower forgetting; markedly better
*distant*-task retention — oldest-task accuracy 0.21–0.42 vs the cascade's
0.06–0.27 — the genuine prospective-configuration signature) but much **less
plastic** (a freshly-trained task reaches only ~0.41 vs the cascade's ~0.86; its
conservative equilibrium update learns ~4× slower, and more epochs made it
*worse*). On the average-accuracy metric the plasticity deficit dominates, so the
cascade wins (0.33 vs 0.23–0.30). The classic PC inference advantage
(denoise/inpaint) would need a *generative* PC; ours is discriminative, so it
isn't a fair vehicle. **Bottom line for the iterative regime: its
prospective-configuration character is real (stability over plasticity) but does
not net a clean win over the cascade on the tropical discriminative substrate.**
The cascade remains the robust practical learner.

## Discarded approaches

- **Margin target** (`target = feed-forward logits + margin`): no fixed point —
  the network chases its own logits, which run away to ~1e9; accuracy at chance.
  Removed as a recommended mode (`target_mode="fixed"` is the default).
- **Undamped settling** (`gamma_shift=0`, full Euler step): diverges to NaN at
  every depth. Non-expansiveness bounds the max-plus *map*, not the PC settling
  *update*; a sub-unit step is required.

## Verdict

- **The PC machinery is sound and delivers what it promised:** a local,
  backprop-free, **multiplier-free** learning rule (closure verified) that trains
  shallow tropical nets well (0.98 on separable blobs), with the cached-argmax
  free adjoint and forward-init working as designed.
- **The deep-PC *optimisation* problem is solved** — identity init + forward-init
  train to depth 32 without the collapse that kills the baseline at depth 8.
- **But depth does not buy accuracy, and this is the substrate's fault, not
  PC's.** Hard max-plus routing is bistable (frozen or chaotic) under any local
  rule, and even full backprop + Adam caps deep tropical at < 0.5 on MNIST and
  degrades with depth. The max-plus/min-plus layer is information-lossy.

- **Path 2 resolved it, including full training closure (`DyadicPCNet`):** fixing
  the *primitive* — dyadic ±2^k shift-add mixing + ReLU instead of lossy max —
  makes depth pay off. The local PC rule trains flat to depth 12 (~0.88–0.91), on
  par with the dyadic backprop ceiling. And the **fully multiplier-free
  magnitude-aware update** (quantize both backward operands to ±2^k →
  shift-add weight grad) **matches Adam** (~0.89/0.92/0.88) at **no accuracy
  cost** — forward, inference, and the weight update are all multiplier-free.

- **Path 3 — tropical residual stabilizes settling to depth 32.** A **tropical
  residual** `max(relu(Wx), x)` (a free `max`, bounded/idempotent) keeps the
  *iterative-settling* dyadic PC flat to depth 32 (0.91), where plain dyadic PC
  collapses; **γ=2⁻³** makes the settling step a shift. But settling still walls
  at ~48 even with the residual (backprop on the same net is fine — it's a
  PC-settling problem).

- **Path 4 — the cached-routing cascade SOLVES the depth wall.** Tropical routing
  is *discrete and cacheable* (float PC's isn't): forward once caching the hard
  routing, then route error top-down in a **single O(depth) pass** instead of
  iterating settling. Fully multiplier-free (real-`delta` × ±2^k = shift-add
  error flow; unit-gain skip highways; dyadic/sign weight grad). It is **flat
  from depth 16 to 100 (0.84–0.90), matching backprop**, where settling died at
  48 — and fast (depth 100 in ~30 s, single pass). This is the depth result.

- **Path 5 — the *iterative-relaxation* regime: a double obstacle, and a
  reconciliation.** Keeping the full iterative equilibrium (not the cascade) works
  to depth ~32 but not beyond: settling is stiff (~O(depth²)) AND the hard-clamp
  equilibrium update isn't backprop (and underperforms it on deep classification —
  0.60 @ 48 / chance @ 64 even with a strong solver). The reconciliation: the
  **cascade IS iterative PC in the small-nudge/Z-IL regime** (the regime where PC
  matches backprop), so backprop-matching iterative tropical PC already exists.
  Iterating harder is a *different* (prospective-config) regime whose value is
  continual-learning / inference, not deep-classification accuracy.

**Bottom line.** Pure max-plus PC is fully multiplier-free but can't use depth
(lossy primitive); the **dyadic** primitive + **magnitude-aware dyadic backward**
give depth at the backprop ceiling with a shift-add training step; a **tropical
residual** (`max`) stabilizes it; and the **cached-routing cascade** — exploiting
discrete, cacheable hard routing that float PC lacks — **solves the deep-PC depth
wall, flat to depth 100 matching backprop, multiplier-free and single-pass** (and
it *is* the backprop-matching limit of iterative PC). The full-relaxation regime
is practical to depth ~32 and is where PC's prospective-config advantages live.
The through-line: deep PC's failures are multiplicative, and the tropical
substrate supplies the additive/discrete duals (bounded `max`, ±2^k shifts, cached
routing) that fix them with only adds, compares, and shifts.

## Next steps (toward a deep tropical-PC ResNet)

1. **Multi-seed + full MNIST + CIFAR** — the depth curves are single-seed,
   width-64, 4k-subset MNIST (MLP). Confirm the depth-100 cascade result with
   error bars, full MNIST, and a real dataset.
2. **Dyadic conv + true residual blocks** — replace the dyadic Linear stack with
   dyadic conv (the repo has a tropical conv) and proper residual blocks for a
   genuine ResNet, not an MLP.
3. **Push past depth 100** — the cascade is O(depth) and stable; test 128/200 to
   match μPC's regime, multiplier-free.
4. **On-fabric mapping** — dyadic forward + the cascade backward (real-`delta` ×
   ±2^k = shift-add error flow, sign-XOR/popcount weight grad, `max` residual)
   all land on the multiplier-free kernel; sketch the datapath.
5. **Then transformers** — revisit tropical attention (the LM instance has
   progress here) once the deep-conv-ResNet path is solid.

**Separate track — where the *iterative* regime earns its keep:** demonstrate
the prospective-configuration advantage at depth ~32 (where iterative settling is
practical) on a task that *rewards* relaxation — **continual learning** (less
catastrophic forgetting) or **inference** (denoise / inpaint / missing-data) —
where iterative tropical PC should beat the cascade/backprop. That is the right
way to show *why* one would iterate at all, rather than chasing deep-classification
accuracy (a non-target, per Path 5).

## Reproduce

```
TT_REPO="$(git rev-parse --show-toplevel)"              # canonical repo (see env note)
PYTHONPATH=$TT_REPO python scripts/pc_depth_probe.py smoke    # closure + sanity learn
PYTHONPATH=$TT_REPO python scripts/pc_depth_probe.py depth    # baseline vs identity, depth 2-32 (blobs)
PYTHONPATH=$TT_REPO python scripts/pc_depth_probe.py mnist    # MNIST baseline vs identity
PYTHONPATH=$TT_REPO python scripts/pc_active_probe.py         # identity-gap bistability sweep
PYTHONPATH=$TT_REPO python scripts/pc_ceiling_probe.py        # max-plus primitive ceiling (backprop+Adam)
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_probe.py         # PATH 2 step 1: dyadic vs dense ceiling
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_train.py         # PATH 2 step 2: dyadic PC depth curve (Adam)
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_train.py rescue  #   (settle-stability rescue grid)
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_train.py signsgd # multiplier-free pure-sign update + per-sample identity
PYTHONPATH=$P python scripts/pc_dyadic_train.py signbest #   (tune the no-momentum pure-sign update)
PYTHONPATH=$P python scripts/pc_dyadic_train.py signfix # magnitude-aware dyadic-backward update (matches Adam)
PYTHONPATH=$P python scripts/pc_dyadic_train.py deepwall # PATH 3: locate the depth wall (no tricks)
PYTHONPATH=$P python scripts/pc_dyadic_deep.py 16        # PATH 3: stability-machinery sweep at depth 16
PYTHONPATH=$P python scripts/pc_dyadic_deep.py scale     # PATH 3: tropical residual depth scaling (16/24/32)
PYTHONPATH=$P python scripts/pc_dyadic_deep.py powgamma  # PATH 3 (A): power-of-two settling step (a shift)
PYTHONPATH=$P python scripts/pc_dyadic_deep.py deepen    # PATH 3 (B): depth-32 settling + gentle precision
PYTHONPATH=$P python scripts/pc_dyadic_deep.py deepmf    # PATH 3: multiplier-free residual at depth 32
PYTHONPATH=$P python scripts/pc_dyadic_deep.py resnet 64 16 32 48 64  # depth ceiling: settling walls at 48
PYTHONPATH=$P python scripts/pc_dyadic_deep.py cascade 32 48 64 100   # PATH 4: cached-routing cascade vs BP
PYTHONPATH=$P python scripts/pc_dyadic_deep.py cascadefinal           # PATH 4: multiplier-free cascade, flat to 100
PYTHONPATH=$P python scripts/pc_dyadic_deep.py gs 16 32 48            # PATH 5: Gauss-Seidel settling
PYTHONPATH=$P python scripts/pc_dyadic_deep.py explore 32 48 64       # PATH 5: damped/anneal/skip
PYTHONPATH=$P python scripts/pc_dyadic_deep.py deepfix 48 64          # PATH 5: multigrid + momentum
PYTHONPATH=$P python scripts/pc_dyadic_deep.py eqceiling 48 64        # PATH 5: equilibrium-ceiling (decisive)
PYTHONPATH=$P python -m pytest tests/test_predictive_coding.py -q
```

Note: the editable install points at the kanervasforge submodule checkout, so
`PYTHONPATH` must force the canonical repo to pick up this code (see below).
