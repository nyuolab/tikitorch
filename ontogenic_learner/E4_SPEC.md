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
