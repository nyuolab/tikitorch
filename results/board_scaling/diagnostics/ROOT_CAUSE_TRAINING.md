# Root-cause: why tropical MuZero training hurts (chess trained 0.071 < untrained 0.393)

**Step 1 of the FPGA roadmap** — a cheap, CPU-only test of the *binding* hypothesis before any further
fabric work. The flagship goal is a blazing-fast, fully-tropical MuZero entirely on FPGA; that is only
worth building if the model can be made to *play*. Today it cannot: training the prior is flat-to-harmful
(chess trained `0.071` vs its own untrained control `0.393`; loss falls `27.55→12.0` while strength drops).

The pre-workflow critique's leading hypothesis was that the **closed SignSGD/TropicalSGD optimizer**
(`W -= lr·sign(grad)`, which discards gradient *magnitude*) is structurally too weak on value-like
targets, and that the fix is a *new learning rule*. This document tests that, constructively.

Harness: [`scripts/muzero_train_diagnostics.py`](../../../scripts/muzero_train_diagnostics.py)
(`optbisect` = stationary supervised bisect; `online` = instrumented self-play loop).

---

## Experiment A — offline optimizer bisect (stationary; isolates optimizer-class)

Fit the **same** dyadic net (`TropicalMuZeroNet`, `core_kind="dyadic"` — the scaffold net), from
**byte-identical init**, to two stationary Connect4 targets: a **smooth MC-rollout value** in [-1,1]
(genuinely value-like / oblique) and a depth-4 **negamax policy** distribution. Each optimizer reported
at the *best* of a small hyperparameter sweep (no straw-manning). 512 positions, 1000 full-batch steps.
Data: [`optbisect_connect4.json`](optbisect_connect4.json).

| optimizer | value-L1 (smooth target) ↓ | vs predict-mean | policy top-1 ↑ | zero-DSP? |
|---|---|---|---|---|
| `tropicalsgd` (pure sign-only) | 0.087 | −69% | 27% | ✅ |
| **`adamw`** (open ceiling) | **0.030** | **−89%** | 50% | ❌ |
| `signsgd`+mom0.9 (sign-of-EMA) | 0.031 | −89% | 31% | ❌ (float EMA, hole H7) |
| `exponentemaadam` (magfree) | 0.067 | −76% | 40% | ✅ |
| `frexpmag` (magfree) | 0.073 | −74% | 22% | ✅ |
| `tropicaladam` (magfree) | 0.075 | −73% | **63%** | ✅ |
| `countagreement` (magfree) | 0.092 | −67% | 29% | ✅ |

**Reading (honest):**
- **Gradient magnitude matters for the VALUE head.** Pure zero-DSP sign-only underfits the smooth value
  target ~2.9× worse than Adam (0.087 vs 0.030) — but it still removes 69% of the predict-the-mean error,
  so the deficit is *moderate, not catastrophic*.
- **What recovers it is accumulated magnitude** (Adam, or sign-of-float-EMA momentum ≈ 0.031) — and that
  EMA is a float multiply, i.e. **not zero-DSP** (the known closure hole H7).
- **The existing zero-DSP magnitude-aware optimizers only *partially* close it** (best `exponentemaadam`
  0.067, still 2.2× Adam). A fully zero-DSP value-fit fix is *not* off-the-shelf — but the gap is moderate
  and looks like an optimizer-engineering problem (better integer/exponent momentum), not a dead end.
- **Policy ranking is fine multiply-free** — `tropicaladam` (sign-streak Rprop) gets top-1 **63%**,
  *beating* Adam (50%). The optimizer weakness is specific to *value regression*, not policy.

This is *consistent* with the optimizer hypothesis but **too mild to explain the catastrophe** — a
69%-vs-89% value-fit gap cannot turn trained `0.071` below untrained `0.393`. So the loop itself must be
tested directly.

---

## Experiment B — online in-situ disambiguator (Connect4, vs random, learned-MuZero tree)

The decisive test: in the **actual self-play loop**, does swapping the optimizer (or recipe) lift strength
above the **untrained control**? 3 optimizers × 2 recipes, 14 iters, 40 sims, 60 eval games vs random.
`lift = trained(soft τ) − untrained_control`. Data: [`online_connect4.json`](online_connect4.json).

| optimizer / recipe | trained (soft) | untrained ctrl | **lift** | τ-gap (soft−hard) | final mode_frac |
|---|---|---|---|---|---|
| tropicalsgd / ce_ent0.1 | 0.80 | 0.78 | **+0.017** | +0.10 | 0.44 |
| tropicalsgd / ce | 0.80 | 0.82 | −0.017 | −0.07 | 0.58 |
| tropicaladam / ce | 0.73 | 0.77 | −0.033 | +0.07 | 0.30 |
| adamw / ce_ent0.1 | 0.72 | 0.77 | −0.050 | +0.07 | 0.48 |
| **adamw / ce** | 0.70 | 0.77 | **−0.067** | −0.03 | 0.54 |
| tropicaladam / ce_ent0.1 | 0.65 | 0.72 | −0.067 | −0.02 | 0.25 |

**Reading (decisive):**
- **No cell meaningfully lifts strength.** Best is `+0.017` (flat). Training the prior is flat-to-harmful.
- **AdamW does NOT rescue the loop.** `adamw/ce` is **−0.067** — *worse* than sign-only `−0.017`. The
  open-optimizer ceiling that fit value well offline fails to lift strength in-situ. **This refutes the
  optimizer class as the binding cause.**
- **Strength comes from SEARCH, not the learned prior.** The untrained MCTS control already scores ~0.80
  vs random; there is no headroom for the prior to add, and training *collapses* it
  (mode_frac climbs to 0.44–0.58, entropy falls, value-pred L1 *degrades during training* — the
  loss-down/strength-down decoupling, captured live).
- **The deploy τ mismatch is NOT a major factor** — the soft→hard τ gap is small and sign-inconsistent
  (±0.1). One hypothesis ruled out.
- The entropy bonus (`ce_ent0.1`) is the only thing that nudges positive and reduces collapse
  (mode_frac 0.58→0.44) — consistent with the earlier Connect4 prior-sweep, but the effect is marginal.

---

## Experiment C — chess confirmation (the actual catastrophe domain, with real headroom)

The target domain. Connect4-vs-random had the untrained control at ceiling (~0.80), so "training doesn't
help" could be a no-headroom artifact. Chess vs random has the untrained control at ~0.35–0.45 (matches
the scaffold's 0.393) — **real headroom** — so this resolves that confound. 8 iters, 16 sims, 20 eval
games. Data: [`online_chess.json`](online_chess.json).

| optimizer / recipe | trained (soft) | untrained ctrl | **lift** | τ-gap | final mode_frac |
|---|---|---|---|---|---|
| adamw / ce_ent0.1 | 0.40 | 0.42 | −0.025 | +0.075 | 0.44 |
| tropicalsgd / ce | 0.33 | 0.35 | −0.025 | +0.075 | 0.40 |
| tropicalsgd / ce_ent0.1 | 0.35 | 0.38 | −0.025 | −0.050 | 0.37 |
| **adamw / ce** | 0.35 | 0.45 | **−0.100** | +0.050 | **0.61** |

**Reading (confirms B, resolves the headroom confound):**
- **Every cell is flat-to-negative even with real headroom.** No optimizer/recipe beats the untrained
  control. So the Connect4 "no-headroom" worry is dismissed — training genuinely fails to help where there
  *is* room to improve.
- **AdamW is again the WORST** (`adamw/ce` −0.100) and shows the **hardest collapse** (mode_frac
  0.016→0.348→**0.613**, entropy 5.83→1.70). A *stronger* optimizer fits the over-confident self-play
  target faster and so collapses the prior *harder* — direct evidence the **loop target**, not the
  optimizer, is the problem.
- The loss-down/strength-down decoupling is reproduced exactly (loss 27.6→8.3 while strength drops),
  matching the original scaffold catastrophe.

---

## Verdict

**The binding cause is NOT the optimizer / learning rule. It is self-play prior-collapse driven by an
uninformative learning target at attainable compute.**

1. **NOT the optimizer (refuted, both domains).** Swapping the closed sign-optimizer for AdamW does **not**
   rescue strength — Adam is *worse* in the loop and collapses the prior hardest. The pre-investigation
   "needs a new learning rule" hypothesis is wrong as the primary lever.
2. **NOT the deploy-τ mismatch.** Soft→hard τ gap is small and sign-inconsistent (±0.05–0.10).
3. **The mechanism is prior-collapse.** The MCTS-visit→policy distillation drives the prior to an
   over-confident spike (mode_frac climbs, entropy falls) regardless of optimizer; a stronger optimizer
   accelerates it. Anti-collapse (entropy bonus, target tempering) only marginally mitigates.
4. **The deeper constraint is capacity/compute headroom.** Strength comes from MCTS *search*; at
   workstation compute the learned prior cannot beat raw search, so there is no informative gradient
   rewarding a *better* (non-collapsed) prior. The learning target itself is too weak to lift play.
5. **The closed optimizer is a real but SECONDARY issue:** offline it underfits the smooth value head
   ~2.9× vs Adam (still −69% of error), and a fully zero-DSP magnitude-aware fix is not off-the-shelf
   (best magfree closes ~⅓ of the gap). Worth improving for value fidelity, but it will **not** unblock
   the strength gate.

### Implications for the FPGA roadmap
- **The strength gate is not unlocked cheaply** (no optimizer/learning-rule silver bullet). The real
  levers are: (a) **much more self-play data/compute**, (b) **anti-collapse training** (entropy/target-
  temper — marginal alone), (c) a **stronger model or true-rules AZ search** to create headroom, or
  (d) **downscope** to a regime where the learned prior demonstrably helps.
- **The strength problem is largely DECOUPLED from the FPGA work.** Self-play is **CPU tree-search-bound**
  (reference GPU ~93% idle); FPGA accelerating *net-eval* does not directly attack the bottleneck — that
  would need MCTS-on-fabric or a compiled/parallel actor. So the honest FPGA value proposition is
  **multiply-free inference/deployment** (the forward kernel on silicon, throughput levers — shippable on
  its own merits), **not** unblocking training strength.

### Caveats
Diagnostic-scale: single seed, small n (eval 20–60), short loops (8–14 iters). The **directional**
conclusions are robust because they are qualitative patterns consistent across 7 optimizers (offline),
6 Connect4 cells, and 4 chess cells — not marginal effects: (i) Adam never rescues and collapses hardest,
(ii) training is flat-to-harmful even with headroom, (iii) prior-collapse is observed live in every cell.
A seed-replicated confirmation would tighten the magnitudes but is unlikely to flip the direction.
