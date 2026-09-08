# Root-cause: why tropical MuZero training hurts (chess trained 0.071 < untrained 0.393)

**Step 1 of the FPGA roadmap** — a cheap, CPU-only test of the *binding* hypothesis before any further
fabric work. The flagship goal is a blazing-fast, fully-tropical MuZero entirely on FPGA; that is only
worth building if the model can be made to *play*. Today it cannot: training the prior is flat-to-harmful
(chess trained `0.071` vs its own untrained control `0.393`; loss falls `27.55→12.0` while strength drops).

The pre-workflow critique's leading hypothesis was that the **closed SignSGD/TropicalSGD optimizer**
(`W -= lr·sign(grad)`, which discards gradient *magnitude*) is structurally too weak on value-like
targets, and that the fix is a *new learning rule*. This document tests that, constructively.

Harness: [`scripts/muzero_train_diagnostics.py`](../../../../scripts/muzero_train_diagnostics.py)
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
`lift = trained(soft τ) − untrained_control`. Data: [`online_connect4.json`](online_connect4.json)
(learned-MuZero tree, 6 cells).

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

## Experiment D — AZ true-rules tree + headroom + seed replication (the go/no-go)

Does giving the net a *strong, real-rules* search (so learned-model fidelity is not the ceiling) + a
stronger opponent (negamax-d3, real headroom) let the trained prior beat its untrained control? And does
it hold across seeds? 16 iters, 48 sims, 80 eval games. Data: `online_connect4_az_s0.json`,
`online_connect4_azrep_s{1,2,3}.json`.

| seed | trained (soft) | untrained ctrl | **lift** |
|---|---|---|---|
| 0 | 0.331 | **0.025** | +0.306 ← weak-baseline OUTLIER |
| 1 | 0.200 | 0.438 | −0.237 |
| 2 | 0.250 | 0.569 | −0.319 |
| 3 | 0.219 | 0.362 | −0.144 |
| **mean** | | | **−0.099** |

**Reading (this corrected a mid-investigation false positive):** seed 0 alone showed a large +0.306 lift and
looked like "AZ-tree rescues training." **Replication refuted it** — that seed had an anomalously weak
untrained baseline (0.025); seeds 1–3, with normal baselines (0.36–0.57), all show training **hurts**
(−0.14 to −0.32). Mean lift **−0.099**: even with a true-rules tree and real headroom, training the
zero-DSP prior is **net negative** across seeds. The untrained, spread-out prior is on average *better* for
shallow-search PUCT than the trained, over-sharpened one (`mode_frac` climbs every run). This is the
seed-variance-is-first-order caveat made concrete — and a reminder that single-seed lifts here are noise.

## Experiment E — dense vs tropical, fair search budget, head-to-head (the substrate-vs-scale test)

Experiments B–D used starved search (16–48 sims) and an external opponent (ceiling/floor confounds). This
fixes both: 100 sims, true-rules AZ tree, and the confound-free metric **trained vs its OWN untrained self**
(>0.5 = training helped). Dense MuZero (Adam) is the control: if dense *also* fails to benefit, "doesn't
play well" is the MuZero-at-scale wall, not the tropical substrate. n=5 seeds each. Data:
`online_connect4_decisive_{dense,dyadic}_s{0..4}.json`.

| substrate | H2H mean (trained vs untrained) | range | final mode_frac |
|---|---|---|---|
| dense (Adam) | **0.505 ± 0.115** | 0.42–0.70 | 0.267 |
| tropical (TropicalSGD) | **0.500 ± 0.091** | 0.42–0.64 | 0.386 |

**Reading — two big corrections:**
- **The catastrophe was a STARVED-SEARCH artifact, not a permanent property.** At a fair sim budget the
  trained-≪-untrained collapse (Experiments B–D) is GONE. Training is now **net-neutral, H2H ≈ 0.50** — it
  no longer actively hurts. My earlier "firm NO-GO" described the starved regime; it over-generalized.
- **Dense ≈ tropical — statistically identical (0.505 vs 0.500).** The gold-standard dense substrate is
  *exactly as unable* to benefit from training at this scale. So "doesn't play well via training" is NOT a
  tropical defect — it is the well-known MuZero compute/data wall, substrate-independent. (Residual tropical
  signal: it collapses a bit more, mode_frac 0.39 vs 0.27 — a value-precision gap — but it does not produce
  a head-to-head disadvantage.)
- **The `lift`-vs-opponent metric (used in B–D) is unreliable** and partly manufactured the earlier
  catastrophe. In E it swings ±0.4 across seeds (untrained baseline alone ranges 0.06–0.84) while the
  shared-opponent-cancelling head-to-head sits stable at ~0.50. So the B–D "trained ≪ untrained" was
  substantially baseline-variance noise on a bad metric, not only starved search. Head-to-head is the metric
  to trust.

Net: at workstation iteration/game scale, MuZero training adds **nothing measurable over raw search**, for
dense OR tropical. Strength rides entirely on search; the learned net is along for the ride. Making training
*contribute* requires far more self-play (games/iters) — a pure compute problem, where tropical has no
disadvantage vs dense.

## Verdict

**"Doesn't play well via training" is the MuZero compute/data wall — substrate-independent (dense ≈
tropical) — NOT a tropical defect, NOT the optimizer, and NOT a permanent collapse. The training effect is
search-budget-dependent: harmful when search is starved, neutral (H2H ≈ 0.50) at a fair budget, for dense
AND tropical alike. Making training actually HELP requires more self-play scale, where tropical has no
disadvantage vs dense.** (Supersedes the earlier "firm NO-GO," which described only the starved-search
regime — see Experiment E.)

1. **NOT the optimizer (refuted, both domains).** Swapping the closed sign-optimizer for AdamW does **not**
   rescue strength — Adam is *worse* in the loop and collapses the prior hardest. The pre-investigation
   "needs a new learning rule" hypothesis is wrong as the primary lever.
2. **NOT the deploy-τ mismatch.** Soft→hard τ gap is small and sign-inconsistent (±0.05–0.10).
3. **The mechanism is prior-collapse.** The MCTS-visit→policy distillation drives the prior to an
   over-confident spike (mode_frac climbs, entropy falls) regardless of optimizer; a stronger optimizer
   accelerates it. Anti-collapse (entropy bonus, target tempering) only marginally mitigates.
4. **The deeper constraint is capacity/compute headroom — and even true-rules search + headroom +
   seed replication does NOT rescue it (Experiment D).** Strength comes from MCTS *search*; at workstation
   compute the trained prior cannot reliably beat the untrained one. Giving the net a strong real-rules
   AZ tree (removing learned-model fidelity as the ceiling) still yields **mean lift −0.099** across 4
   seeds — training the prior is net negative. The single seed that looked positive (+0.306) was a
   weak-baseline artifact, caught by replication. So this is a firm **NO-GO on the current self-play
   training recipe**, not a "needs the right tree/opponent" issue.
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
