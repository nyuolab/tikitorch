# E4c — gating re-entry: is the recipe cheap AND reliable? (prereg, FROZEN 2026-07-17)

*Last cell of the ignition arc (E3 → E4 → E4b → this). E3's gated arm (polG)
starved: 2.6–15.6% fire → no foraging ignition. E4b's recipe (intr + norm)
ignites reliably but fire-always — the kick on every step. The north star
wants BOTH: learn only when it matters (the gate is the efficiency story)
AND ignite reliably. One question: does the recipe transfer to gating?*

## Why it plausibly transfers (declared hypothesis)

The margin gate fires on obs-CE surprise — the SAME signal intr converts to
credit. Under the recipe, intrinsic credit accrues EVERY step (the backfill
rides the serve pass, no kick needed), while kicks concentrate exactly where
surprise (and thus shaped advantage) is largest; norm then unit-scales at
kick time. Gated kicks should carry a HIGHER policy-gradient fraction than
fire-always kicks. Counter-hypothesis: 3–15% fire is simply too few kicks to
compound before the (rarely-updated) world model drifts behaviour.

## Arms (all recipe: --intr-beta 0.5 --adv-norm, policy token; NO code change)

- **comboG** = recipe, gated (margin δ = E2cal: foraging 0.10, minatar 0.05):
  foraging ×5 seeds, minatar ×3.
- **comboA** reference = E4b's fire-always cells (reused, not re-run).
- **ctrl-F s0** drift sentinel (must reproduce 0.245; code untouched).

9 cells, one c7i.16xlarge (single wave at P 14; gated cells are serve-heavy
≈ frozen walls), MAX_MINUTES = 360, ≈ $9.

## Pre-registered readout (SUSTAINED primary, as E4b)

- **CHEAP-AND-RELIABLE (the arc's capstone):** comboG foraging sustained
  ≥ 3/5 at mean fire ≤ 20%, with mean ret/1k ≥ 40 (within ~20% of comboA's
  48.5). Then the whole story holds at the gated price point and the
  north-star loop learns policies only when surprised.
- **STARVED (E3-polG repeats):** sustained ≤ 1/5 ⇒ kicks-per-life is the
  binder; fire-always stays the recipe's price and the efficiency claim
  stays prediction-side only. Published either way.
- Between: PARTIAL — report the frontier point (sustained rate, fire rate,
  mean) as the measured trade.
- Secondary: minatar comboG vs comboA (2/3 sustained, mean 28.2); fire-rate
  trajectories (does the gate self-quiet as CE falls?); piH; intrinsic-mass
  audit at the gated cadence.
- Sentinel: ctrl-F s0 = 0.245 or the screen quarantines.

## House rules

Adversarial review before launch (surface: driver + flag composition only —
no code change), sentinel-first, honest branches, EC2-only. This closes the
ignition arc; next decisions (manuscript integration, E5 persistence life,
the 2.8× streaming-AC gap) go to the user with the arc's consolidated
verdict.

---

## RESULTS (2026-07-17, box i-0e09dff4c4f7879e5, 9/9 cells, 0 failures, ~1.8h, ≈$6)

Sentinel: 0.245 exact (fourth consecutive screen, bitwise-stable baseline).
Data: `results/st_reward/e4c_screen.jsonl` + tele.

arm | seeds ret/1k | mean | fire | SUSTAIN
---|---|---|---|---
comboG foraging | 9.5 / 8.9 / 11.5 / 7.3 / 17.2 | 10.9 | 6.0% | **0/5**
comboG minatar | 22.7 / 25.4 / 20.6 | 22.9 | 12.7% | 1/3
(comboA refs: foraging 48.5 & 4/5; minatar 28.2 & 2/3.)

### Verdict per prereg

- **Foraging — STARVED** (sustained 0/5 ≤ the 1/5 bar). At 6% fire (~12k
  kicks/life) the recipe cannot compound: mean 10.9 vs comboA's 48.5.
  Better than E3's naked polG (4.9) but nowhere near the floors.
  Kicks-per-life is the binder; **fire-always remains the recipe's price on
  the primary env**, and the efficiency claim stays prediction-side only.
- **Minatar — PARTIAL, and mechanism-confirming.** At 12.7% fire (2.1× the
  foraging rate) the gated recipe reaches mean 22.9 = 81% of comboA's 28.2,
  sustained 1/3 with two near-misses (s0 Q3=46→Q4=23, s2 Q3=39→Q4=22 —
  both Q4 dips land ON the 23.3 floor). The cross-env fire-rate gradient
  (6% starves, 13% nearly holds) is exactly what the kicks-per-life binder
  predicts. A δ-sweep trading fire-rate against sustained-rate would map
  the frontier, but the arc's question is answered: gating is not free.

### The ignition arc, closed (E3 → E4 → E4b → E4c, ~$85 total)

1. E3: the composed serve+learn loss has a dark-room attractor; ignition is
   a coin-flip (substrate exonerated by direction test).
2. E4: two independent one-flag levers (surprise credit, advantage
   normalization) each make ignition reliable; two others map the failure
   space (variance compression, rescue-to-random).
3. E4b: the levers STACK — the recipe (token-policy + dyadic advantage +
   intr + norm, fire-always) delivers 4/5 sustained, mean 48.5 foraging and
   program-best minatar, at 1-forward-act + 1-kick-learn.
4. E4c: the recipe does NOT survive aggressive gating on the primary env —
   learning-only-when-surprised undershoots policy ignition at ~6% fire;
   the fire-rate gradient across envs confirms kicks-per-life as the binder.

Open fronts for the next phase: the 2.8× streaming-AC gap at matched steps
(capability), minatar sustained universality (geometry/gradient-volume), a
δ/fire-rate frontier sweep (efficiency), the E5 persistence life with the
recipe, and manuscript integration of the arc.
