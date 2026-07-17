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
