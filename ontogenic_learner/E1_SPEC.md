# E1 — action without an actor/learner split (pre-registration DRAFT)

*2026-07-14. The decisive gate of ROADMAP_EMBODIED.md. Status: DRAFT — freeze
before any run; harness build next. Everything below is fixed before results
exist; amendments get dated entries.*

## The claim under test

One process, one model: the SAME forward computation that produces the
agent's behavior is the sufficient statistic for its learning, in a closed
sensorimotor loop — no value function, no replay buffer, no separate learner,
no second memory class. E1 tests the weakest version that is still an agent:
**model-predictive acting over a live world model that learns by gated
cascade kicks on its own prediction errors.** Prediction is the only teacher;
competence improves because, and only because, the world model improves.
(Reward entering the LEARNING rule is E2's question, not E1's.)

## World (token-native, so the whole validated substrate is reused)

A procedural 2D foraging gridworld rendered AS a token stream. Each timestep
appends: `[OBS] <local 3x3 view, 9 tiles> [ACT] <action>` — a fixed 12-token
frame over a ~24-symbol vocabulary (tile types: empty, wall, food, agent;
5 actions: N/S/E/W/stay; separators). The MorphSSM consumes the interleaved
stream; its next-token predictions over the OBS positions of the NEXT frame
are the action-conditional world model.

Nonstationarity (the gate must have work): food regrows under a drifting
spatial distribution; **biomes** (wall layouts + food statistics) switch on a
schedule with revisits — the V4 rotation discipline ported to a world. World
generator is pure Python, seeded, committed; three world seeds per arm.

## Acting (forward passes only)

At each step, for each of the 5 candidate actions, run the forward with that
action token appended and score the predicted next-OBS distribution for food
adjacency (a fixed, hand-written preference functional over predicted tiles
— no learned value). Choose argmax with epsilon=0.05 exploration.
Acting cost: 5 forwards/step (model-predictive control at K=5). This is the
SERVE baseline that all cost multipliers are measured against.

## Learning (the fused loop, unchanged)

After the environment returns the realized next OBS frame, the excess-margin
gate (E0-1's gate if it passes; else the margin gate + rotated biome
schedule) decides whether the already-computed forward for the CHOSEN action
becomes a cascade kick on the realized-OBS prediction error. Anchor
lambda=0.01, bit-shift decay per E0-3's verdict. tau=0 throughout. The
policy tokens carry no loss in E1.

## Arms (all serve the same worlds, same seeds)

1. **live** — the agent above (gated learning on).
2. **frozen-MPC** — identical acting, learning off (the frozen-twin analog;
   isolates the value of live world-model learning for CONTROL).
3. **random** — uniform actions (floor).
4. **streaming-RL** — a streaming actor-critic per Elsayed et al. 2024
   (their hyperparameters, adapted minimally), matched to the live arm's
   TOTAL compute (acting + learning fwd-eq) and separately to matched
   wall-clock; its replay-free but distinct actor/learner computation is the
   contrast the manuscript's §6 names.
5. **replay actor-learner** — a small DQN with replay buffer at the same
   matched budgets (the deployed-RL shape).

Warm start: none (world vocab is new) — the world model trains from scratch
with the tau-anneal birth recipe on a short pre-life in the FIRST biome only,
then the life begins (mirrors V-program warm-start discipline; the pre-life
is identical across arms 1-2).

## Metrics and the pre-registered verdict

Primary: **foraging return per 1k steps** over a 1M-step life (mean of last
quarter vs first quarter after each biome entry), 3 world seeds x 3 agent
seeds. Secondary: prequential OBS-CE vs frozen (the perception claim);
fire-rate; amortized cost multiplier over acting; biome-revisit retention
(return recovery latency); sampled zero-multiply audit; norm equilibrium.

**Pass:** live return > frozen-MPC return with non-overlapping seed CIs in
at least 2 of 3 biomes AND live amortized cost <= 1.1x acting AND retention
criteria (V4 form) hold. **Competitive context (not pass/kill):** live vs
streaming-RL/replay at matched budgets, published as a Pareto whatever it
shows. **Kill:** live return <= frozen-MPC despite OBS-CE improving =>
"perception learns in-process; control does not benefit at this horizon" —
published as the boundary, and E2 (reward in the gate) becomes the required
next step rather than an option.

## Cost & infra

Harness: `experiments/st5_world.py` (world gen + frame codec + MPC loop
reusing MorphPCEngine/serve_and_learn/gates verbatim) + `_pc_e1_onbox.sh`.
CPU boxes; est. wall ~2-4h per (arm x world-seed) at d=128; full grid ~$60-90
across two boxes. Baselines use torch on the same boxes (small nets).
Review-before-launch, honest-negative branches, EC2-only — house rules.

## Known risks, stated now

(i) MPC with a hand-written preference functional may be too weak to forage
even with a perfect model — the random and frozen floors expose this early
(a cheap pilot cell runs FIRST: frozen-MPC with an ORACLE world model; if
oracle-MPC does not beat random, the preference functional gets one
pre-registered redesign before the grid). (ii) 12-token frames make the
recurrent state's usable horizon ~5 frames at window 96 — biome statistics
must be locally inferable. (iii) From-scratch birth may be the bottleneck
(V-program always warm-started); the pre-life length gets one calibration
cell. (iv) Compute-matching RL baselines is contentious — we publish both
matched-compute and matched-wall, per V3 practice.

## Amendments (dated, pre-launch)

**A1 (2026-07-14, design-time):** OBS tiles are encoded CENTER-FIRST, so one
forward per candidate action reads P(landing tile = food | history, action)
at the final position — the exact greedy foraging signal with no
autoregressive rollout (K=5 forwards/step total). The original row-major
read-out was semantically wrong (it scored frame-tail token predictions) and
never ran.

**A2 (2026-07-14, from the oracle pilot smoke — the pre-registered
one-redesign):** strict argmax over preference scores degenerates when all
candidates score flat zero (no food adjacent): the agent walks its fixed
first-argmax direction into a wall and parks, losing to random walk
(oracle-MPC 12 vs random 39 per 1k). Fix: uniform-random tie-breaking over
near-max scores — random-walk exploration when the signal is flat, greedy
foraging when it is not. Post-fix pilot: oracle-MPC 90.7 vs random 39.0 per
1k (2.3x) — the pilot bar (oracle beats random) is cleared and the
preference functional is FROZEN.

**A3 (2026-07-15, from the pilot-review harness-learnability smoke):** three
result-corrupting harness bugs were caught by the pre-launch review and fixed
before any spend (all three would have produced a FALSE NEGATIVE — killing
the E1 gate on a bug, not the claim): (i) the frame codec was
cause-after-effect (action written after the obs it produced → causally
invisible; now ACTION-FIRST: `[ACT] a [OBS] center...`); (ii) food was
consumed before the frame was rendered, so the realized stream showed
center==food 56/200k times vs the oracle's view (now render-then-consume:
555 center-food frames per 8k oracle-birth steps); (iii) policy tokens
carried loss, violating the spec's "policy tokens carry no loss" (now a
mux-based `loss_mask` in MorphPCEngine — audit-verified zero unwaived
multiplies — restricts the kick, gate, and reported CE to tile-target
positions). BIRTH POLICY: random-walk birth leaves the world model at the
action-INDEPENDENT marginal (food-adjacency too rare: spread 0.000);
ORACLE-GUIDED birth (eps=0.3 explore) over-samples the food-landing
transitions and yields a correctly action-conditional model (controlled
food-to-north probe: N=0.150 vs stay=0.043, spread 0.11). Birth is
world-model SHAPING, identical across live/frozen via the shared seed; the
LIFE still tests continued GATED learning. Verdict criteria unchanged.

## E1 VERDICT (2026-07-15): KILL on the pre-registered branch — a mechanism, not a null

**Result (pilot world 0 + multi-seed birth calibration, softmax controller A4):**
- Perception LEARNS in one process: live obs-CE 0.7355 < frozen 0.751; the
  action-conditional signal grows with birth length (food-north-probe spread
  0.00 → 0.46 → 0.53) and the model correctly scores the food-direction
  highest on the clean test state.
- Control does NOT benefit — and gets WORSE as prediction improves:
  foraging return/1k vs birth length (2 seeds): 8k-birth 26.0/19.5 (~random)
  → 20k-birth 11.0/2.0 → 40k-birth 0.0/1.5. A better world model forages
  FURTHER BELOW random walk (~20/1k). Both pilot MPC arms sat at ~0
  (live 0.175, frozen 0.037) vs random 26.6, oracle-MPC 113, streaming-AC 175.

**Mechanism (the finding):** prediction quality and greedy-control quality are
DECOUPLED, even anti-correlated, here. Greedy 1-step model-predictive control
confidently exploits the model's residual miscalibration; a 1-cell preference
functional only "sees" adjacent food, so between rare adjacencies the agent
executes confident-but-wrong moves instead of covering ground. Random walk
stumbles onto food; a myopic exploiter of an imperfect model does not. This is
robust to the action-selection rule (softmax A4 did not rescue it) and to seed.

**Interpretation (honest, scoped):** the live-loop MECHANISM (Goal A) is intact
— one process learns its world model in place at ~1× acting cost, and
perception measurably improves. What E1 shows is that this mechanism, by
itself, does NOT yield competent CONTROL under greedy 1-step MPC with a
myopic preference. Prediction alone is not control. This does not touch the
V-program's Goal-A result (serving/prediction); it bounds the naive embodied
extension.

**Consequence:** E2 (reward IN THE LOOP) is promoted from option to REQUIRED
next gate — the finding is its motivation. A planning horizon (multi-step MPC)
is a separate future-work lever. The E1 harness, world, baselines, and the
decoupling result are the publishable artifact.
