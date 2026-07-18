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
