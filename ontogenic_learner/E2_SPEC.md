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

## Amendments from the pre-launch review (2026-07-15, before any run)

Four confirmed factorial-validity fixes (the E1 discipline — the review hunts
false-result bugs):
- **A1 gate orthogonality:** the surprise gate now reads OBS-ONLY CE in both
  objectives (new `MorphPCEngine.gate_mask`, decoupled from `loss_mask` which
  trains obs+reward in R). Without this, R cells gated on reward-prediction
  surprise too, leaking the objective axis into the gate axis. Regression:
  masks-unset serve_and_learn stays bit-exact vs cascade_grads.
- **A2 salience baseline:** the neuromod salience EMA is evaluated BEFORE the
  AND, so the reward baseline always updates (Python short-circuit had frozen
  it to an over-surprising-windows-only EMA).
- **A3 clean O objective:** O = REWARD-FREE = random policy on BOTH envs. The
  foraging obs-heuristic P(food) is itself a reward proxy; using it made O
  env-inconsistent (heuristic on foraging, random on minatar) and foraging-O ≈
  foraging-R. Now O is the uniform "no reward in the objective" control (SO/NO
  are the negative controls; the R column carries the control signal).
- **A4 minatar reporting:** report MinAtar PER-GAME, not pooled — sparse-reward
  games (freeway/seaquest/asterix) make the R-objective reward token ~always
  'none' → MPC ≈ random and the N gate ~never fires (near-frozen online). That
  IS the reward-density finding, but the S-vs-N contrast is only fair on the
  dense games (breakout, space_invaders); log/report per-cell online fire-rate
  (already in tele) beside return.

## E2 screen v1 RESULT (2026-07-15): INCONCLUSIVE — gate under-fired (confound)

Return/1k (mean[seeds]) — foraging: random 26.6, frozen-R 39.7 |
SO=NO 33.7, SR=NR 34.1. minatar: random 22.9, frozen-R 13.9 | all cells ~13.5.
- **The GATE axis (a) does LITERALLY NOTHING:** NR=SR and NO=SO return
  BIT-IDENTICALLY per seed (e.g. foraging s0 both 39.688). Root cause: the
  margin gate δ=0.25 (tuned for the LM streams) fires only **0.1–0.9%** here —
  after the tau-annealed birth the model predicts its own obs stream well, so
  obs-CE surprise rarely exceeds δ. With ~no online learning, the two gates
  have nothing to modulate and actions stay RNG-dominated (identical). The
  neuromod mechanism is therefore UNEVALUABLE from this screen.
- **Main effects within seed noise:** main-N +0.00/−0.11, main-R
  +0.39/+0.04, interaction +0.00/−0.21 — all << the ±8 seed spread. The a+b
  interaction (the design target) CANNOT be assessed while a is unevaluable.
- **Live learning ≤ frozen birth:** frozen-R ≥ every learning cell on both
  envs → past birth, the online kicks (what little fire there is) add nothing.
- **MinAtar controller ≤ random** (13.5 vs 22.9); per-game, space_invaders
  dominates (51/1k) while the sparse games sit at ~4–5 — the greedy 1-step
  reward-MPC is myopic (E1's control finding, recurring), and reward density
  drives everything.

**Verdict: NOT a clean pass or kill — the screen is UNDER-POWERED on the gate
axis (δ mis-calibrated → ~0 fire), so it cannot fairly test the mechanisms.**
Pre-registered next step BEFORE any E2 mechanism claim: **E2cal — a gate-δ
fire-rate recalibration** (sweep δ on both envs to a target ~5–15% fire, the
V4cal pattern), then re-run the factorial. Separately, MinAtar's controller <
random re-flags that greedy 1-step MPC is the bottleneck (a planning-horizon
variant is the other lever). Cost of v1: ~$20 — cheap to have learned the
calibration is wrong before drawing a mechanism conclusion.

## Planning-horizon A/B: built, with a token-scale cost finding (2026-07-15)

The H2 controller (faithful 2-step lookahead: imagine the next obs frame
greedily, then max over the second action's predicted reward; score =
P(r1|a1) + γ·max_a2 P(r2)) is built (`Agent.score_actions_h2`). Measured
per-step throughput vs greedy H1 (d128, CPU):
  foraging  H1 59 sps  | H2 3.4 sps  (17x — imagined frame = obs_len+1 fwds)
  minatar   H1 27 sps  | H2 0.1 sps  (270x — 100-cell board rollout)
**FINDING: model-based planning with a GENERATIVE TOKEN world model does not
scale** — the imagined-frame rollout is O(obs_len) fwds per candidate per ply,
so H2 is infeasible on MinAtar's 100-token board (~99h/cell) though tractable
on foraging (~1.6h/cell at 20k steps). Implication: if a planning horizon is
what control needs, the substrate needs a COMPACT/LATENT world model, not a
token one (future-work lever).

## Re-run design (E2 v2, pending E2cal's calibrated δ)

- **foraging** (H2 tractable): full {gate S|N} × {horizon H1|H2} × 2 seeds on
  the R objective + O controls + floors. Tests the neuromod gate AND the
  planning A/B (and their interaction) at the recalibrated δ.
- **minatar** (H2 infeasible): {gate S|N} × H1 × 2 seeds on R + O controls +
  floors, dense games emphasized (per-game report). H2 documented infeasible.
- Same pre-registered readouts + the added horizon main-effect/interaction;
  δ from E2cal (target ~5–15% fire).
