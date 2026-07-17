# E4 — ignition reliability: four levers in parallel (prereg, FROZEN 2026-07-16)

*User direction: "let's sweep all of these in parallel for E4." E3 established
that the fused kick executes REINFORCE faithfully and that 3/12 policy cells
ignited into compounding policy learning — but the composed serve+learn loss
has a dark-room attractor that captures cold starts by luck. E4 tests the four
pre-identified levers for making ignition reliable, side by side, one screen.*

## What E3 fixed as ground truth (controls reused, not re-run)

- polA foraging = {0.24, 22.0, 29.5}, ignitions 2/3; minatar polA 0/3.
- Floors (3 seeds): foraging frozen 28.2 / random 36.3; minatar 23.3 / 22.0.
- streaming-AC matched-steps: foraging 138.1; minatar 31.9.
- Mechanism: during droughts ~99% of kicks carry zero policy term; the dense
  obs-CE term sculpts a predictable dither (dark room). Ignition = early
  advantage pulses compounding before the trap closes.

## The four arms (each = E3 polA + ONE lever; no combinations this screen)

1. **intr** — intrinsic surprise reward. The model's own prediction error
   becomes reward: after each serve, the NEWEST FRAME's per-frame obs-CE
   excess over its slow EMA (clipped at 0) is back-filled as intrinsic reward
   on that frame's credit channel (as built — sharper credit than the
   window-level gate CE originally sketched; declared pre-launch):
   `r_used = r_ext + beta * max(ce - ce_ema, 0)`, beta = 0.5, ce_ema decay
   0.999. Extrinsic-only everywhere else (the [RWD] token, salience EMA, and
   ALL reported ret/1k stay extrinsic). Rationale: a surprise-seeking term
   directly opposes the predictability pull; zero new machinery (the signal
   already exists for the gate).
2. **norm** — advantage normalization. `A_hat = (G − baseline) /
   sqrt(var_ema + 1e-4)` with var_ema = EMA(0.99) of (G − baseline)^2, THEN
   dyadic-quantize (same k ∈ [−4, 2], same drop-at-|A_hat|<2^-5). Lifts small
   consistent advantages to weight ~1 — raises the policy term's share of
   gradient volume during near-drought phases.
3. **mbwarm** — model-based birth actor. For the first W = 10% of life (foraging 20k, minatar 10k steps) the
   behaviour policy mixes in the world-model's reward posterior: with
   probability p(t) = max(0, 1 − t/W), act via the E2 MPC head (batched
   K-forward P(reward+ | a), softmax at tau_act = 0.02); otherwise act_token.
   Credit/learning unchanged (REINFORCE on the action actually taken;
   epsilon-greedy-style warmup off-policyness accepted and declared). This is
   the enumerative form of the clamped-settle P(a | reward+) posterior —
   iterative settling is only informative when clamped, and enumeration is
   its cheapest implementation at K ≤ 6. Cost: +K fwds/step for ≤ W steps.
4. **ent** — always-on uniform pull at action positions (as built; supersedes
   the earlier multiplicative-smoothing sketch). Engine hook adds
   `2^-4 · (softmax − uniform_K-slice)` at action positions AFTER the dyadic
   mux, so it acts even where the advantage drops the position — droughts are
   exactly when it must counter the world-model sculpting. Dyadic magnitude
   (ldexp, shift-class); unit-tested: flags-off bitwise ≡ cascade (R1), the
   pull reaches dropped positions (R2), and it raises slice entropy toward
   ln K (R3). Declared the weakest-motivated arm (E3's dither was NOT
   low-entropy; this partly proxies 'stay near the random policy', whose
   floor is 36.3); included per user directive; reduced seed count.

Plus **ctrl** — polA unchanged (the E3 configuration, flags off). Foraging
seeds 0–2 must reproduce E3 polA bitwise-or-near (same code path, same seeds)
— the regression sentinel; seeds 3–4 extend the control distribution.

## Design

- Envs/configs exactly E3: foraging (200k steps, w96, δ n/a — all arms
  fire-always like polA) and minatar (100k, w256). Gating is NOT swept here
  (E3 showed it starves ignition; one lever at a time).
- **Foraging (primary): 5 seeds** (0–4) × {intr, norm, mbwarm, ctrl} + 3 seeds
  (0–2) × ent = 23 cells. **Minatar (replication): 3 seeds** × {intr, norm,
  mbwarm} = 9 cells. 32 cells total, one c7i.16xlarge, foraging-first.
- New telemetry: policy entropy at the act_token probe every 1k steps
  (attractor-state readout); ce_ema and intrinsic-reward mass logged for intr.
- All reported returns are EXTRINSIC-only, including intr's.

## Pre-registered readout (distribution over seeds — the E3 lesson: no binary
arm-level PASS/KILL on a bistable phenomenon)

- **Ignite(seed)** := max(Q3, Q4 mean extrinsic ret/1k) > 36.3 (the E3 random
  floor mean, the higher floor). E3 control rate by this rule: 2/3.
- **Primary comparison:** foraging ignition rate per arm out of 5 vs ctrl out
  of 5. **WIN** for a lever: rate ≥ 4/5 AND arm mean ret/1k > ctrl mean.
  **NULL:** rate within ±1 seed of ctrl. **HARM:** rate ≤ ctrl − 2 or mean
  collapse. Small-n honesty: rates reported as fractions with the seed lists;
  no significance theatre at n=5.
- Secondary: minatar rates (0/3 E3 baseline — ANY minatar ignition by a lever
  is reportable progress); time-to-ignition (first 1k-bucket with smoothed
  ret > floor); dark-room occupancy (fraction of life with pi-entropy-CE in
  the trap signature); intr's intrinsic-mass audit (did shaped reward stay
  subordinate to extrinsic scale?).
- Sentinel: ctrl foraging s0–s2 must land within noise of E3's {0.24, 29.54,
  22.02} (same seeds, same path). A sentinel miss = code drift ⇒ the whole
  screen is quarantined before interpretation.
- Combos (e.g. intr+norm) are E4b, gated on this screen's winners.

## Costs & closure

Engine additions: `loss_uniform_pull` (hook-off path bitwise ≡ cascade_grads
— regression bar R1, verified) — the dyadic advantage path is unchanged. Agent additions are host-side Python (intrinsic
backfill, variance EMA, MPC warmup mixture). Closure: the kick still receives
only mask + sign·2^k weights + (new) a λ-mix of one-hot/uniform targets —
uniform over K is a shift-class constant (1/K with K ∈ {5,6}: NOT a power of
two; declared as a HOST-side target constant, same status as the E3 advantage
baseline EMA — the on-fabric story quantizes it later, waiver declared now).
One c7i.16xlarge (64 vCPU, OMP 4 × P 14), MAX_MINUTES = 540, ~7.5h, ≈ $26.
House rules: adversarial review before launch, sentinel-first job order,
honest branches, EC2-only.

---

## RESULTS (2026-07-17, box i-045d3e0c24ede4d45, 32/32 cells, 0 failures, 6h07m, ≈$21)

Data: `results/st_reward/e4_screen.jsonl` + tele (same dir; lever-suffixed arm
names; ctrl tele supersedes E3's, verified identical on shared fields + piH
added). Sentinel: **ctrl s0–s2 reproduce E3 polA EXACTLY** {0.245, 29.54,
22.02} — screen quarantine-free. Live dashboard:
https://claude.ai/code/artifact/11430f3b-a443-4e3f-bfae-09300f3dd6ce

ret/1k (full life), foraging 5 seeds | seeds | mean | Ignite (max(Q3,Q4)>36.3) | sustained (post-hoc: Q3 AND Q4 > 36.3)
---|---|---|---|---
ctrl (E3 polA) | 0.2 / 29.5 / 22.0 / 25.3 / 31.5 | 21.7 | 4/5* | **0/5**
**intr** | 43.2 / 46.5 / 42.4 / 48.9 / 45.4 | **45.3** | **5/5** | **4/5**
**norm** | 41.3 / 31.0 / 48.9 / 50.6 / 52.2 | **44.8** | **5/5** | 3/5
mbwarm | 29.9 / 31.8 / 32.7 / 18.3 / 34.7 | 29.5 | 0/5 | 0/5
ent (3 seeds) | 36.6 / 36.9 / 40.2 | 37.9 | 3/3 | 1/3

minatar 3 seeds | seeds | mean | Ignite (>23.3) | sustained
---|---|---|---|---
intr | 8.4 / 11.6 / 1.2 | 7.0 | 1/3 | 0/3
norm | 20.3 / 19.2 / 26.1 | 21.9 | **3/3** | **2/3**
mbwarm | 7.4 / 7.8 / 1.3 | 5.5 | 0/3 | 0/3
(E3 ctrl minatar: 1.3 / 7.3 / 7.0, 0/3 by either metric.)

*The frozen Ignite metric counts TRANSIENT crossings: ctrl's 4/5 are
single-quarter biome spikes (s1 Q3=11→Q4=44; s2 Q3=45→Q4=25), which is why
the post-hoc SUSTAINED column (declared post-hoc, both quarters) is the
sharper discriminator: ctrl 0/5 vs intr 4/5, norm 3/5.

### Verdicts per prereg

- **intr — WIN** (rate 5/5 ≥ 4/5 ✓, mean 45.3 > ctrl 21.7 ✓). Audit caveat
  reported per prereg: intrinsic mass ran CO-DOMINANT with extrinsic on
  foraging (~1.3–1.5×), not subordinate — the returns above are extrinsic-
  only, but the credit signal was substantially curiosity-driven.
- **norm — WIN** (5/5 ✓, 44.8 > 21.7 ✓), including the two best single-cell
  results of the program (50.6, 52.2) — and the only lever that moves
  MINATAR: 3/3 Ignite, 2/3 sustained, with late-life rates (Q3 up to 47)
  above streaming-AC's full-life 31.9 — the program's first sustained
  above-floor policy performance on the standard benchmark. Mechanistically
  cleanest: no shaped reward, no extra forwards, just unit-scaled advantage.
- **mbwarm — HARM on the frozen rate metric** (0/5 vs ctrl 4/5), mean
  UPLIFTED (29.5 > 21.7), no collapse: the decaying MPC prior is a VARIANCE
  COMPRESSOR — it reliably prevents the dark room (worst seed 18.3 vs ctrl
  0.2) but anchors behaviour to the model's mediocre reward posterior and
  never compounds past random. Both tails cut.
- **ent — formally NULL by seed count** (3 seeds cannot reach ≥4/5); 3/3
  Ignite with mean 37.9 pinned AT the random floor (36.3) — exactly the
  rescue-to-random behaviour predicted at design time. Prevents the trap,
  learns nothing beyond it.

### What E4 settles

The dark-room attractor is FIXABLE at zero substrate cost: two independent
levers (surprise-credit and advantage normalization) each turn E3's
bistable coin-flip into 5/5 reliable ignition on the primary env, at the
same 1-forward act / 1-kick learn budget. The E3 KILL-branch worry
(substrate/kick) is now doubly dead: the same kick that dark-roomed under
the naked composed loss sustains compounding policy learning when the
policy term's gradient share is protected. Environment geometry remains the
frontier: minatar's 200:2 obs:action imbalance + ~2-frame credit horizon
yields only norm's near-floor performance — the window-geometry fix (longer
credit horizon per window) is a design lever E4 did not touch.

### E4b candidates (user decides)

(a) **intr+norm combo** at 5 seeds both envs (the two WINs are mechanistically
independent: shaped credit × protected gradient share); (b) norm + longer
minatar credit horizon (window 512 or frame-thinned obs) targeting the
geometry frontier; (c) gate re-entry: norm/intr under polG-style gating (E3
showed gating starves ignition — do the levers change that?); (d) β/scale
sweeps only if the combo disappoints.
