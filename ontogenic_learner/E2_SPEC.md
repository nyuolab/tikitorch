# E2 — reward in the loop: a mechanism-screening factorial (prereg DRAFT)

*2026-07-15. The decisive gate promoted by E1's kill (prediction ≠ control).
Status: FROZEN 2026-07-15 (user sign-off). DECISIONS: (1) first screen = 2 envs
(foraging + MinAtar), BabyAI joins post-2x2; (2) neuromod gate uses REALIZED
reward for a clean 2x2 — predicted-salience is tested only IF a+b wins. Design intent (user, 2026-07-15): (a)
and (b) are likely both needed eventually but it's too early to pick; test the
mechanisms as a factorial ACROSS environments of differing reward density
before drilling.*

## The question

E1 established that prediction-only in-process learning improves the world
model but does NOT yield control under greedy MPC (prediction and control
decoupled, even anti-correlated). E2 asks: **what is the minimal way reward
must enter the single-process live loop for it to produce CONTROL — and does
the answer travel across environments?**

## Two orthogonal reward mechanisms (the factorial axes)

- **GATE axis — when to learn.**
  - `S` surprise-only: the V4 margin gate on observation-prediction error
    (the E1 setting; control condition).
  - `N` neuromodulated (mechanism a): fire on surprise AND reward-salience —
    gate = margin(obs-CE) fires AND |realized reward on the window| above its
    running baseline (a product/AND gate). Reward enters the SCHEDULE only; no
    gradient path, no multiply → strict closure preserved. Uses REALIZED
    reward, so independent of the objective axis.
- **OBJECTIVE axis — what to model.**
  - `O` observation-only world model (E1 setting).
  - `R` observation + reward token (mechanism b): each frame carries `[RWD] r`,
    r ∈ {none, +}; the model predicts reward as a token; the MPC preference
    becomes the env-agnostic **P(next-frame reward = + | history, a_cand)**.
    Reward enters the OBJECTIVE/representation.

2×2 cells: {S,O} = E1 baseline · {N,O} = a-only · {S,R} = b-only ·
{N,R} = a+b (the interaction the design targets).

## Environments = a deliberate reward-density axis

- `foraging` (bespoke World): dense-ish reward, fully controlled — every knob
  understood; the mechanism cross-check.
- `minatar` (game 0 = Breakout first): dense game-score reward; our
  streaming-RL baseline (Elsayed et al.) has published numbers here.
- `babyai`: sparse episodic goal reward — JOINS after the 2×2 picks a
  mechanism (its sparsity is itself the discriminating test; adapter cost
  only pays once the reward mechanism exists). Not in the first screen.

Hypothesis (pre-registered): (b) dominates as reward gets sparser; (a)
dominates as reward gets denser → the mechanism ranking is reward-density
dependent, which is WHY both are eventually needed. The screen tests this.

## Screen design (pilot scale, cheap — E1's "pilot before grid" lesson)

4 cells × 2 envs (foraging, minatar) × 2 seeds = 16 runs + per-env frozen +
random floors. Horizon ~80–150k steps (E1-pilot scale), batched-MPC
throughput fix (58×) applied. One CPU box, ~$20–40. Metric primary =
reward-return/1k (E1: prediction ≠ control, so return leads); secondary =
prequential obs-CE, fire-rate, amortized cost multiplier, biome-revisit
retention, sampled zero-multiply audit.

## Pre-registered readouts + pass/kill

- **Main effect of N** (gate): mean return lift of {N,·} over {S,·}.
- **Main effect of R** (objective): lift of {·,R} over {·,O}.
- **Interaction (the target):** is {N,R} > max({N,O}, {S,R}) with
  non-overlapping seed CIs? YES ⇒ a and b are complementary (user's
  hypothesis confirmed) → drill a+b. NO/redundant ⇒ the cheaper single wins;
  report which.
- **Env-consistency:** same cell ranking on foraging and minatar, or
  reward-density dependent? A flip is a FINDING (motivates carrying both
  mechanisms), not a failure.
- **PASS (E2 gate):** at least one reward-mechanism cell beats surprise-only
  {S,O} on return with seed CIs in at least one env, AND live > frozen, AND
  the amortized cost stays ≤ ~1.1× acting. **KILL:** no cell beats {S,O} in
  EITHER env ⇒ reward-in-the-loop via these mechanisms does not produce
  control at this horizon; escalate to a planning-horizon MPC (separate
  lever) or a value estimate, published as the boundary.
- (c) advantage-weighted cascade HELD OUT of the first screen (a third axis +
  a declared audit implication: one scalar multiply/window). Added only if the
  objective axis (R) proves to matter, as a refinement of reward-weighting.

## Build order

1. Freeze this prereg.
2. `st5_envs.py`: Env interface + `ForagingEnv` + `MinAtarEnv` (reward token
   codec shared); local tokenization/throughput smoke.
3. Extend the agent: reward-token frames (R), neuromod gate (N) — both small
   deltas on the existing serve_and_learn loop; the loss_mask already excludes
   non-tile targets and now includes the reward token as a learnable target.
4. Adversarial pre-launch review (the E1 discipline — it caught 3
   false-negative bugs there).
5. Screen box (one CPU, ~$30). Then drill the winner; BabyAI joins.

House rules throughout: pre-registration, honest-negative branches, EC2-only,
audit coverage, cheap-pilot-before-grid.
