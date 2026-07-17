# E4b — combo + credit-horizon (prereg, FROZEN 2026-07-17)

*Follow-on to E4's two WINs (E4_SPEC RESULTS), per the standing keep-moving
directive. Two questions, one screen, ZERO new code (arms compose existing
E4 flags):*

1. **Do the two independent WIN levers stack?** `combo` = intr (β=0.5) +
   norm together. Mechanistic hypothesis: independent (shaped credit ×
   protected gradient share) → combo ≥ max(single levers). Declared risk:
   norm normalizes the SHAPED advantage, so intr's co-dominant intrinsic
   mass gets unit-scaled too — could amplify curiosity noise into a variance
   floor (the E0-1 échos). That interaction is exactly what the cells test.
2. **Does minatar's credit horizon bind?** `normw512` = norm at window 512
   (vs 256): ~4–5 action positions per window instead of ~2 — the cheapest
   possible geometry probe (one flag). If sustained rate and mean rise vs
   E4's norm-w256, the near-bandit horizon was binding; if not, the
   obs:action gradient-volume imbalance (unchanged by window) is the binder.

## Arms (fire-always, policy token, all else = E4)

- **combo** foraging ×5 seeds (200k, w96) + minatar ×3 (100k, w256)
- **normw512** minatar ×3 (100k, w512; biome-steps 8000 unchanged)
- **ctrl** foraging ×1 (s0) — drift sentinel only (code untouched since E4;
  must reproduce 0.245)

12 cells, one c7i.16xlarge (OMP 4 × P 14 — single wave), MAX_MINUTES = 480
(long pole = w512 minatar ≈ 2× E4's ~2.7h ≈ 5.5h), ≈ $22.

## Pre-registered readout — SUSTAINED ignition is PRIMARY (the E4 metric lesson)

- **Sustain(seed)** := Q3 mean AND Q4 mean, both > floor (36.3 foraging /
  23.3 minatar). E4 baselines on THIS metric: ctrl 0/5; intr 4/5; norm 3/5
  (foraging) · norm 2/3, intr 0/3 (minatar).
- **combo verdicts (foraging):** STACK = sustained ≥ 4/5 AND mean ≥
  max(45.3, 44.8); REDUNDANT = between the singles; INTERFERE = below
  min(single levers) − 1 seed or mean < 40.
- **combo (minatar):** reportable if sustained > 2/3 (beats norm alone).
- **normw512 verdicts:** HORIZON-BINDS = sustained ≥ 2/3 AND mean > E4
  norm-w256 mean (21.9) + 25% (i.e. > 27.4, above the frozen floor);
  HORIZON-MOOT = within ±25%; note walls (w512 costs ~2× — the horizon fix
  must justify its serve-cost tax in any north-star accounting).
- Sentinel: ctrl-F s0 must reproduce 0.245 (code untouched — any drift
  quarantines the screen).
- Full-life means + the E4 Ignite metric reported alongside for continuity;
  piH + intr_sum audits as in E4.

## House rules

Adversarial review before launch (focused: flag composition + w512 wall
math), sentinel-first ordering, honest branches, EC2-only. Combos beyond
intr+norm, β/scale sweeps, and gating re-entry stay gated on this screen.
