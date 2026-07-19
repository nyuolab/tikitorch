# PRE5 — resolve scale, environments, algorithm before E5 (prereg, FROZEN 2026-07-18)

*User direction after E4d: resolve (1) scaling, (2) environments, (3)
algorithm + thesis audit before the life. Approved decisions: targeted
scaling with lr control; the frozen env trio stands (XLand-MiniGrid
designated the future generalization gate); test critic-from-world-model +
the head+obsk-2 combo; the life adopts d256 ONLY on a ≥25%-both-envs bar.*

## PRE5-A — algorithm screen (c7i.16xlarge, 13 cells, ≈$16)

Both arms ON the E5 config (head + intr β=0.5 + norm, fire-always); E4d head
references (foraging 53.3 & 3/3, minatar 33.0 & 2/3) are the comparators.

1. **wmv** (`--wm-value`): the world model is its own critic. After choosing
   the action, ONE probe forward `[ctx, ACT, a, RWD]` (the E2 score_actions
   machinery) reads p⁺ = P̂(reward⁺ | ctx, a); the reward-to-go reverse loop
   initializes its accumulator at v_tail = p⁺/(1−γ) (geometric persistence
   bootstrap) instead of 0, with done-gating unchanged — closing the
   window-credit horizon that explains the 2.6× streaming-AC gap on
   foraging. ZERO new learners: this is CE-on-observed prediction reused
   for credit, NOT a TD-trained value head — the E2 minimality boundary is
   untouched. Declared cost: acting 2 fwd/step → ≈4× serve-equivalent
   total; measured and reported beside any win (SPJS framing).
2. **headobsk** (`--action-head --obs-k -2`): the deferred ceiling
   (head) + reliability (mild volume rebalance) combination.

Sentinel: bare polA foraging s0 = 0.245 (7th consecutive screen).

**Bars.** wmv WIN = foraging mean ≥ 64 (+20% over the head) AND minatar
no-harm (mean ≥ 30); if won, wmv joins the E5 config and the life's compute
budget is re-flagged. headobsk WIN = minatar sustained 3/3 with mean ≥ 33
AND foraging mean ≥ 48. NULLs are published; no post-hoc bar movement.

## PRE5-S — scaling screen (c7i.24xlarge, OMP 4 × P 22 single wave, 19 cells, backstop 780, ≈$47)

All at the E5 head config, 3 seeds:
- **d256/4L × lr {1e-3, 5e-4} × both envs** (12 cells). The lr grid is the
  scaling-confound control — declared: if 5e-4 wins at d256, the fixed-lr
  d128 history carries a caveat. Params: 77k → ~297k (unified-vocab basis).
- **Context at d128-head**: foraging w256 ×3, minatar w512 ×3 (6 cells).
  Untested post-head (E4b's horizon-moot was pre-head). Measured w-scaling
  walls 2.9–3.45× apply.
- Sentinel: bare polA foraging s0 (this box's own quarantine).

Anchor honesty, declared up front: streaming-AC beats us on foraging with
22k params, so foraging scaling gains are NOT expected; minatar capacity and
context are the live hypotheses.

**Bars.** SCALE-WINS = ≥25% mean uplift on BOTH envs at the best lr
(foraging ≥ 66.6, minatar ≥ 41.3) → the life runs d256 (budget $150–250,
user-approved). One-env ≥25% = SCALE-HELPS (paper point; life stays d128).
CONTEXT-WINS = same 25% rule at matched steps. NULL on all = capacity/
context are not the binder at this tier — the 77k-agent result stands as a
paper point. Sustained quarters reported throughout (E4 pattern).

## Documents (no compute)

- **SYSTEM_AUDIT.md** — the current system against the thesis: components,
  closure ledger (shift-class vs host-side declared waivers), measured cost
  ledger, open problems, claim-language guardrails.
- **ENV_DECISION_2.md** — the environment matrix; the trio stands for E5;
  XLand-MiniGrid = the designated generalization gate (task distributions
  test a different claim; JAX vectorization is moot at single-env CPU);
  rejected alternatives with reasons; 2026 task-diversity/continual-RL
  prior art engaged.

## Process

House rules as all six prior screens: local smokes + unit tests (wmv credit
closed-form incl. done-gating; flags-off bitwise) → commit → focused
adversarial review (wmv math, d256 walls, dual-box launch state handling) →
launch BOTH boxes together (A ~3h, S ~11h) → sustained analysis vs bars →
winners fold into E5_SPEC mechanically → roadmap + memory. Two-box state:
the launcher's run.env is snapshotted per box; monitors track S3 prefixes
directly; both boxes self-terminate.

---

## PRE5-A RESULTS (2026-07-18, box i-09016e959a638d92e, 13/13, 0 failures, ~3.6h, ≈$11)

Sentinel exact (0.245, 7th consecutive screen).
Data: `results/st_reward/pre5a_screen.jsonl` + tele.

arm | foraging (mean · SUS) | minatar (mean · SUS)
---|---|---
**headobsk** | **58.3 · 2/3** (47.8/**78.1**/48.9) | **36.5 · 3/3** (41.8/38.9/28.8)
wmv | 44.1 · 2/3 | 29.2 · 2/3
(head refs: 53.3 · 3/3 | 33.0 · 2/3; bars: headobsk = minatar 3/3 & ≥33 &
foraging ≥48 ✓✓✓; wmv = foraging ≥64 ✗)

- **headobsk — WIN on its full bar.** The ceiling lever (separate readout)
  and the reliability lever (mild obs down-weight) COMPOSE: minatar 3/3
  sustained at mean 36.5 (above streaming-AC's matched-steps 31.9), foraging
  mean 58.3 with the program's best-ever cell (78.1). E4d's strict
  reliability+performance standard is finally met — by the two-factor
  combination the E4d verdict predicted. → joins the E5 config
  (mech_lever = headobsk).
- **wmv — NULL.** 44.1 foraging (below the 64 bar AND below head-alone),
  29.2 minatar (marginal no-harm miss), at the declared 4× serve cost.
  The geometric-persistence value readout does not close the sAC gap:
  either window truncation is not the binding mechanism, or a cruder-than-
  needed bootstrap (early p⁺ miscalibration feeds the baseline). Published
  as the honest negative; does NOT join E5. The sAC gap remains attributed
  to per-step TD credit vs windowed REINFORCE, now with one competing
  explanation eliminated.

---

## PRE5-S RESULTS (2026-07-19, box i-0c67af2dba8683eb4, 13 full + 6 truncated rows, ≈$70)

**Incident, declared:** the 960-min backstop fired with the six d256-minatar
cells at 96–97k/100k steps — the d256×w256 compound wall (>5.4× d128)
exceeded even the review-raised estimate. The 60s sync preserved telemetry;
verdicts below use the delivered 96%+ trajectories with the truncation
stated. (Monitor gap also found: a fully-GC'd instance reads "unknown," not
"terminated" — fixed in the next monitor.) Sentinel exact (8th screen).

cell | foraging | minatar
---|---|---
d256 lr 5e-4 | **85.5** (68.7/82.6/105.2) | 25.6 (trunc.)
d256 lr 1e-3 | 52.7 (flat vs d128) | **35.4** · 3/3 SUS (trunc.)
d128 w256/w512 (context) | 25.9 (≪ 53.3) | 27.7 (< 33.0)
(d128-head refs 53.3 / 33.0; bars: foraging ≥66.6 ✓ at 5e-4; minatar ≥41.3 ✗)

- **SCALE-HELPS, not SCALE-WINS → the life stays d128** (mechanical bar).
  Foraging scales spectacularly (+60% at the SCALED lr — s1 = 105.2, the
  program record; at the old lr d256 is FLAT: the lr-control arm was
  load-bearing, exactly the declared confound). Minatar scales modestly
  (+7% at 1e-3; and note the best lr FLIPS by env — 5e-4 foraging, 1e-3
  minatar — declared). headobsk at d128 (36.5) already beats d256-head
  (35.4) on minatar at ~4× less compute.
- **CONTEXT — NULL/harm at this tier, both envs.** Retired.
- Paper points: (i) the 77k→297k foraging curve with the lr interaction;
  (ii) the honest "capacity is not minatar's binder; readout structure is"
  line now has scale evidence; (iii) d256+5e-4+headobsk is the untested
  "big agent" combo — noted, not run.

## PRE5 CLOSED — the E5 configuration, final

d128/4L · headobsk (action head + obs-k −2) · intr β=0.5 · adv-norm ·
fire-always · window 96 · lr 1e-3 · the frozen env trio · life budget ≤$75.
