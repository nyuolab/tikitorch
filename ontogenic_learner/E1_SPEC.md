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
