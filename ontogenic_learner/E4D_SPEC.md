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
