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

12 cells, one c7i.16xlarge (OMP 4 × P 14 — single wave), MAX_MINUTES = 600
(review amendment: w512 T-scaling MEASURED at 2.9–3.45× w256 — cache-spill
superlinearity, not the 2× FLOPs guess; correct w256 anchors 5.7–8.2 ks →
worst case ≈ 470 min), ≈ $25. The horizon fix's serve-cost tax is now a
measured 2.9×+, priced into any north-star accounting.

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

---

## RESULTS (2026-07-17, box i-03d5706b92415b4a4, 12/12 cells, 0 failures, ~4.8h, ≈$22)

Sentinel: ctrl-F s0 = 0.245 (Q3=0, Q4=0) — exact dark-room reproduction,
screen quarantine-free. Data: `results/st_reward/e4b_screen.jsonl` + tele.

arm | seeds ret/1k | mean | SUSTAIN | Ignite
---|---|---|---|---
combo foraging (w96) | 48.9 / 46.9 / 47.5 / 52.6 / 46.6 | **48.5** | **4/5** | 5/5
combo minatar (w256) | 24.3 / 39.9 / 20.4 | **28.2** | 2/3 | 3/3
normw512 minatar | 35.0 / 25.4 / 20.6 | 27.0 | 2/3 | 3/3

(E4 baselines: intr-F 45.3 & 4/5 sus; norm-F 44.8 & 3/5; norm-M-w256 21.9 &
2/3; ctrl-F sustained 0/5.)

### Verdicts per prereg

- **combo — STACK.** Foraging: sustained 4/5 ✓ AND mean 48.5 ≥ 45.3 ✓, with
  EVERY seed (min 46.6) above both single-lever means and the tightest seed
  spread of any arm (sd 2.5 vs norm's 8.5). Minatar: sustained 2/3 (ties
  norm — the "> 2/3" reportability bar not met) but mean +29% (28.2 vs 21.9)
  and s1 = 39.9 full-life — the program's best minatar cell (Q3=51, Q4=53,
  both ≈ 60% above streaming-AC's full-life 31.9). The levers are
  complementary as hypothesized: shaped credit supplies signal during
  droughts, unit-scaling keeps it full-weight. The declared variance-floor
  risk did not materialize (foraging CE ~0.47–0.51, stable; no noise
  amplification signature).
- **normw512 — HORIZON-MOOT** (by the letter: mean 27.0 vs the 27.4 bar,
  sustained 2/3 ✓). Honest nuance: +23% mean over w256 is real but
  sub-threshold, bought at the MEASURED 2.9×+ serve-cost tax — and the combo
  at w256 achieves MORE (28.2) for FREE. The credit horizon is not minatar's
  binder; the obs:action gradient-volume imbalance (unchanged by window)
  remains the prime suspect. The geometry lever is dominated; retire it.

### What E4b settles

The program now has a RELIABLE, CHEAP policy-learning recipe on the fused
serve+learn substrate: **policy-as-token-prediction + dyadic advantage +
surprise credit (β=0.5) + advantage normalization, fire-always** — 4/5
sustained above floors on foraging (mean 48.5), best-ever minatar means, at
the unchanged 1-forward-act / 1-kick-learn budget. Honest gaps: streaming-AC
still 2.8× ahead on foraging (138 vs 48.5) at matched steps; minatar
sustained is 2/3, not universal; and the loop is fire-always — the
GATING/efficiency dimension (E3 polG starved at 3% fire) is the one
ignition-adjacent question this arc has not re-tested with the working
recipe. That is E4c.
