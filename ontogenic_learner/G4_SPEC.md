# G4 — coaxing genuine continual learning out of the PC learner

*Pre-registered 2026-07-08, prompted by the pilot's measured failures. Designed by a 5-lens ×
3-critic adversarial workflow (22 candidate mechanisms → 8 families), then **verified against
the live pilot checkpoint** before anything was built.*

## 1. The corrected hypothesis

The original hypothesis — "PC's special properties enable continuous learning" — failed **as
stated**, and we proved why ourselves: the cascade is exactly backprop (grad cos = 1.000000,
all depths), so the *update rule* carries no CL advantage. What the organism deploys is BP + a
gate + a buffer, and it behaves like it. The corrected hypothesis: **PC's continual-learning
potential lives in the terms we discarded when reducing full Bayesian PC to the cascade** —
weight priors, precision weighting (unit and synaptic), iterative explaining-away, layerwise
error signals. G4 restores them incrementally and measures which ones buy retention.

## 2. Verified causal map (pilot, day 0–2.7)

| # | Pathology (measured) | Verified cause |
|---|---|---|
| P1 | norm 26→488, near-linear | no weight prior; slope bends at shift (novelty-starvation sharpening) |
| P2 | stationary wiki drift 2.47→2.88 | z-gate never satiates (hardest-30% forever) × P1 miscalibration |
| P3 | shak 28.6→4.10 then rising | 1 MB corpus re-epoched ~every 20 min, no prior → memorization |
| P4 | wiki 2.88→6.48 under shak | **NOT inverted sleep** (see §3) — replay-mass arithmetic: wiki restore ≈ 10.5 kicks × 3e-4 vs shak push ≈ 45 × 1e-3 per 150 steps = **14× imbalance**, and zero consolidation resistance |
| P5 | adapt_gain 2.1→7.4 | gradient-scale artifact of P1 (observability bug, not plasticity) |

## 3. Honest negative from verification (record it)

The panel's highest-drama claim — sleep inverted by frozen shift-time priorities (z≈360,
~98.6% of rehearsal on the new domain) — was **refuted** on the live checkpoint at step 6.19M:
shak holds 23.2% of slots / 24.7% of priority mass; priorities are near-uniform (mean ≈1.0,
max 33). The EMA adapts within ~10² steps of a shift so few inserts carry spike-z, and
reservoir-sampling replacement dilutes those. Sleep rehearses ~75% wiki and wiki erodes
*anyway* — which is what makes the replay-mass arithmetic (P4) the real story. Lesson kept:
verify panel claims against ground truth before building.

## 4. The G4 mechanism package (ranked; PC-nativeness labeled honestly)

- **M1 — weight prior (decay).** Restores the p(W) term of the PC free energy → decay toward
  0 (dyadic: one shift-subtract in the Adam kernel later; AdamW host-side now). *PC-native
  (restored term).* Targets P1/P2/P3/P5.
- **M2 — satiating gate (epistemic surprise).** Replace the self-normalizing z with an
  absolute epistemic gap: fire when batch CE − H(p̂) exceeds a floor (well-predicted data →
  CE→H → gate closes; a shift re-opens it). *PC-flavored (errors explained as noise stop
  driving learning); partly engineering.* Targets P2/P3.
- **M3 — synaptic-precision consolidation.** Per-parameter fast-kick damping by the slow
  Adam's `exp_avg_sq` (weight-posterior precision — the exact ingredient that cut 97% of the
  G3 Fisher-merge oracle gap, turned inward): grad ×= 1/(1 + c·v̂). *PC-native ingredient
  (free), EWC-shaped use.* Targets P4.
- **M4 — replay repair.** `balanced=True` sleep sampling (exists, unwired — b1's 89.7%
  rehearsal used it) + raise sleep mass toward parity with the fast stream (sleep_updates
  15→30 as a start). *Engineering (CLS), labeled as such.* Targets P4.
- **M5 (phase 2) — layerwise error precision Π in the cascade.** The panel's top-ranked
  (7.7/10, 3/3 keeps): learned per-layer error variances anneal the effective gain under
  stationarity (Kalman-gain view) and re-open on shift — the principled version of both the
  LR schedule and the gate. Medium build; spec'd after M1–M4 read out.
- **M6 (phase 3 / research) — k-WTA tropical pattern separation.** Sparse orthogonal codes
  are substrate-native (max/argmax is the fabric primitive; skipped lanes = real throughput).
  Kernel rebuild → after the host-side story is proven.

## 5. The decisive experiment (`experiments/g4_cl_probe.py`)

Compressed organism life, local: **wiki 30k steps → shak 15k → wiki-revisit 15k** (organism
corpora + vocab; DyadicPCNet ctx 12 / hidden 256 / depth 4; same two-speed constants),
val-tail probes every 2k steps. Arms (cumulative, isolating each mechanism's marginal):

`base` → `+M1` → `+M1M2` → `+M1M2M3` → `+M1M2M3M4` (+3 seeds on base and full)

**Metrics:** (a) stationary stability = wiki-val slope in era 1; (b) adaptation speed = steps
to shak < 3.0 after shift; (c) retention = wiki val at end of shak era; (d) recovery = steps
to wiki within 0.1 of era-1 best on revisit; (e) norm trajectory throughout.

**Pre-registered decision rules:** the package GOES if, vs base: stationary drift ≈
eliminated (|slope| < 0.02 bpc/10k), interference cut ≥ 50% at (c), recovery ≥ 2× faster at
(d), and adaptation (b) degraded ≤ 20%. Any single mechanism whose marginal is ~0 across all
five metrics gets dropped from v0.1 (parsimony). If M1 alone fixes P2/P3, the "satiety"
narrative transfers from the gate to the prior — report it that way (honesty over story).

**Confound control (critic-demanded):** norm fixes (M1) inflate apparent retention by
shrinking logit scale; retention is therefore also reported as *rank-accuracy* (top-1
agreement with era-1 snapshot) which is scale-invariant.

## 5b. RESULTS (2026-07-08 · 18 runs, 12 arms · `results/g4_cl/probe.jsonl`)

Compressed life wiki 30k → shak 15k → wiki 15k; means over seeds (base/m1a/m1at n=3, others n=1):

| arm | era-1 wiki | shak adapt | interference | ret@peak | recovery | ret@end | norm | duty |
|---|---|---|---|---|---|---|---|---|
| base | 2.401 | 2.634 | +0.489 | 0.556 | 2.409 | 0.627 | 37.9 | 0.39 |
| m1 (decay) | 2.385 | 2.646 | +0.497 | 0.505 | 2.376 | 0.661 | 35.7 | 0.39 |
| m12 (epi-gate) | 2.712 | 2.832 | +0.339 | 0.560 | 2.583 | 0.604 | 18.3 | 0.10 |
| m13 (grad-damp) | 2.426 | 2.618 | +0.531 | 0.542 | 2.416 | 0.640 | 30.1 | 0.39 |
| m14 (replay+) | 2.476 | 2.694 | +0.436 | 0.536 | 2.485 | 0.612 | 35.7 | 0.38 |
| m1p (parity replay) | 2.903 | 3.167 | **+1.145** | 0.468 | 2.830 | 0.517 | 37.0 | 0.32 |
| m1t (fire-less) | 2.583 | 2.783 | +0.380 | 0.551 | 2.552 | 0.636 | 21.7 | 0.15 |
| **m1a (anchor)** | 2.497 | **2.626** | **+0.303** | **0.626** | **2.415** | **0.728** | **18.1** | 0.39 |
| m1at (anchor+τ) | 2.625 | 2.771 | **+0.262** | 0.609 | 2.566 | 0.664 | **15.2** | **0.15** |

Seed spread on the deciders (no overlap): interference base .478/.485/.503 vs m1a .260/.312/.338;
ret@end base .605/.630/.646 vs m1a .712/.731/.740.

**Verdict: the precision anchor is the single surviving mechanism — v0.1 GO (partial vs
pre-reg).** m1a cuts interference **38%** and lifts end-of-life retention **+10pp** at **zero
adaptation cost** (shak 2.626 vs base 2.634) with **half the norm, flattening** (the first
equilibrium norm trajectory observed). The pre-registered 50% interference bar was not met by
any host-side mechanism (m1at reaches 46% but pays ~0.2 bpc quality; keep τ as the
SPJS-deployment flag). Parsimony drops everything else:

- **M2 epistemic gate: broken as built** — CE−H ≈ 0 at ignorant init (conflates "nothing to
  learn" with "nothing learned"), firing *rises* with training (anti-satiating). Replacement
  candidate: learning-progress/habituation gate. Honest negative.
- **M3 gradient damping: nullified by Adam** — per-param grad rescaling washes out of
  `m/√v` within ~1k steps. Consolidation must move the *fixed point* (hence the anchor).
- **M4/parity replay: fails, informatively** — replaying a 4096-slot buffer at fast-lr mass
  *degrades everything* (buffer memorization, +1.15 interference). **Weight-space memory
  beats data-space memory at organism buffer sizes** — and the weight-space mechanism is the
  PC-native one.
- **M1 decay alone: null at 60k steps** (time-constant 1/(lr·wd) = 100k steps — its test is
  the pilot timescale; it ships inside m1a regardless).

**v0.1 birth config:** two-speed + AdamW(1e-2) + precision anchor (λ=1.0, θ_EMA τ=1e-4,
v̂ = slow-Adam exp_avg_sq/mean, clamp 100) — the restored `p(W) = N(θ_posterior, 1/v)` term.
Residual interference gap (~0.26–0.30) is representation-level → phase-2/3 (layerwise Π,
tropical k-WTA pattern separation).

## 6. Consequences

- v0.1 organism birth config = whatever survives §5 (candidate: M1+M2+M3+M4).
- The paper gains its strongest section: *deployed the reduction → failed exactly as the
  discarded Bayesian terms predict → restored them one at a time → measured which ones carry
  continual learning.* The pilot (untouched, running) is the control life.
- Fabric cost of the surviving set (silicon lens): M1 = one shift-subtract in dyadic-Adam;
  M2/M4 = host-side; M3 = exponent-difference damping (dyadic-friendly); layerwise gating
  (M5-adjacent) *reduces* DDR traffic — CL machinery is a perf win on this substrate, not a tax.
