# E4d — gradient volume + sparse schedules (prereg, FROZEN 2026-07-17)

*Per the approved plan (E4d → E5 → BabyAI arrival). Two question blocks, one
screen. All arms run ON TOP of the recipe (intr β=0.5 + adv-norm), fire-always
unless the arm itself changes firing. References reused from E4b: comboA
foraging 48.5 mean & 4/5 sustained; minatar 28.2 & 2/3. SUSTAINED ignition
(Q3 AND Q4 > floor: 36.3 foraging / 23.3 minatar) is PRIMARY.*

## Block A — the mechanism (is minatar's binder the obs:action gradient volume?)

- **A1 obsw2** (`--obs-k -2`), **A2 obsw3** (`--obs-k -3`): dyadic
  down-weight of obs-position errors via the existing per-position sign·2^k
  machinery (zero engine code). Rebalances ~200:2 → ~50:2 / ~25:2.
- **A3 sub25** (`--obs-frac 0.25`): seeded random 25% of obs positions kept
  in loss_mask per window (gate CE unchanged — gate_mask untouched).
  Same volume cut, stochastic rather than uniform.
- **A4 head** (`--action-head`): separate K-logit TropicalTopKModule action
  head on the shared trunk ("one network, one kick, two readouts" — ruled
  in-bounds by the user). Structural isolation: dark-room pressure reached
  the action marginal through the SHARED head; this blocks the last hop.
  Advantage weighting applies to the action-head error; obs loss untouched.
  SCALE (review amendment, pre-launch): both heads normalize by the SHARED
  learn+act denominator, matching the shared-head design's per-position
  policy-gradient scale exactly — a head win is attributable to structure,
  not a hidden 10-100x policy-lr boost. Birth BCs the random policy THROUGH
  the head (review launch-blocker: the first implementation cloned into the
  LM head and left the acting head at random init — the A1 lottery).

Cells: 4 arms × (minatar ×3 PRIMARY + foraging ×3 guard) = 24.

**Bars:** MECHANISM-CONFIRMED for an arm = minatar sustained 3/3 AND mean >
28.2, with the foraging guard ≥ 3/5 sustained and mean ≥ 40 (no harm). If
only A4 wins → the binder is head-sharing, not raw volume (reported as the
distinction). If A1/A2/A3 win → volume confirmed directly. All-null →
the imbalance hypothesis is refuted; minatar's binder remains open
(published as such; E5 proceeds regardless — minatar eras are lived, not
gated on this).

## Block B — sparse schedules (can the recipe learn only when it matters?)

- **B1 anneal** (`--fire-anneal 0.25`): fire-always for the first 25% of
  life, margin-gated (E2cal δ) after — the user's critical-period idea.
  Expected lifetime fire ≈ 25% + 6%×0.75 ≈ 30%.
- **B2 advfire** (`--adv-fire`): fire on (obs-surprise gate) OR (any nonzero
  quantized advantage in the window). The E4c diagnosis says obs surprise is
  the WRONG gate signal for policy learning; advantage pulses are the right
  one and are intrinsically sparse. NOTE: with intr active, shaped credit
  makes nonzero advantages more common than raw reward density — the
  measured fire rate is itself a readout.

Cells: 2 arms × (foraging ×3 PRIMARY + minatar ×3) = 12.

**Bars:** B1 DEVELOPMENTAL-WIN = foraging sustained ≥ 3/5 at lifetime fire
≤ 40% and mean ≥ 40. B2 SPARSE-WIN = same at fire ≤ 20%. Winner(s) feed
E5's fire_schedule; none → E5 runs fire-always (declared in the plan).

## Sentinel + logistics

Bare polA foraging s0 first (must reproduce 0.245 — 5th consecutive screen;
miss = quarantine). 37 cells, one c7i.16xlarge (OMP 4 × P 14), MAX_MINUTES
600, ≈$30. Wall anchors: foraging recipe cells 7.4-8.3 ks, minatar 5.7-10.2
ks (E4/E4b measured); the action-head adds one small head to fwd+adjoint
(≤10% est — verified by local timing before launch). House rules: local
smokes + engine regression tests (flags-off bitwise ≡ cascade_grads; ported
direction test on the action head), adversarial review before launch,
honest branches, EC2-only.

## Deferred by design

β-anneal (revisit only if audits show intrinsic co-dominance harming);
combinations of A×B winners (E5 config decision, not a new screen unless
the user asks); w512 (retired by E4b).

---

## RESULTS (2026-07-17, box i-076a26726172d54e1, 37/37 cells, 0 failures, ~6.9h, ≈$24)

Sentinel exact (0.245, 6th consecutive screen). Data:
`results/st_reward/e4d_screen.jsonl` + tele.

arm | foraging (mean · sustain) | minatar (mean · sustain) | fire
---|---|---|---
**head** | **53.3 · 3/3** (43.3/62.8/53.8) | **33.0** · 2/3 (43.4/16.1/39.3) | 1.0
obsk-2 | 41.6 · 1/3 | 29.6 · **3/3** | 1.0
obsk-3 | 37.4 · 1/3 | 23.6 · 3/3 | 1.0
sub25 | 39.9 · 1/3 | 27.8 · 3/3 | 1.0
fa25 | 35.7 · 0/3 | 23.8 · 1/3 | 0.36
advf | 50.3 · 2/3 | 27.1 · 3/3 | 0.95–1.0
(refs: recipe/comboA foraging 48.5 · 4/5; minatar 28.2 · 2/3; floors 36.3/23.3)

### Verdicts per prereg

- **No arm meets strict MECHANISM-CONFIRMED** (minatar 3/3 + mean > 28.2 +
  clean foraging guard). obsk-2 hits the minatar clause (3/3, 29.6) but its
  foraging sustained crashes to 1/3; the head sweeps foraging (3/3, mean
  53.3 — NEW PROGRAM RECORD on every seed, lowest CE ~0.28) and leads the
  minatar mean (33.0, cells 43.4/39.3 with quarters up to 69) but keeps the
  2/3 reliability hole.
- **The richer finding — a TWO-FACTOR mechanism map:** gradient volume is
  the RELIABILITY binder (every volume lever → minatar sustained 3/3, but
  means capped near floors, and foraging sustained collapses — the
  world-model tax is real); head-sharing is the CEILING binder (the
  separate readout unlocks record performance on both envs with NO obs tax,
  but not reliability). The prereg's either/or was the wrong shape; both
  factors are real and orthogonal. (A head+obsk-2 combo is the obvious
  follow-up; out of scope per spec — combos deferred.)
- **B1 fa25 — NOT a DEVELOPMENTAL-WIN** (foraging sustained 0/3; Q3/Q4
  24–31 vs the 36.3 floor at 36% fire). The critical period builds but the
  gated adulthood does not maintain at δ=0.10. One minatar seed (55/54
  quarters at 35% fire) shows holding is possible — a δ-frontier question,
  not closed here.
- **B2 advf — degenerate as predicted-risk:** with intr active, ~every
  window carries quantized advantage → fire 0.95–1.0. The sparsity premise
  dies on its declared readout (intr densifies advantage). Returns are
  recipe-class (foraging 50.3), i.e. it is fire-always by another name.
- **E5 config consequence (per the approved plan):** mech_lever = HEAD
  (dominant, no tax); fire_schedule = ALWAYS (no sparse winner — declared).

### For the record

The E4d screen also delivered the program's best-ever cells on both envs
(foraging 62.8; minatar Q3 = 69 under the head) and localized minatar's
residual unreliability to something neither gradient volume nor readout
structure fully controls — cold-start reward luck remains a factor at 100k
steps (head minatar s1: dark-roomish quarters 14/22 despite the head).
