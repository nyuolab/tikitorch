# E2 environments: standard benchmarks alongside the bespoke world

*2026-07-15. Feasibility check (web-grounded + adversarially fact-checked +
MinAtar verified hands-on) of running MinAtar and BabyAI in parallel with the
bespoke foraging world through one E2 harness. Decision doc; freeze before the
E2 build.*

## The filter (why not just grab a standard benchmark)

A standard benchmark helps ONLY where the axis is one we're strong on —
streaming/continual adaptation under non-stationarity, prequential, at matched
compute. On capability/scale axes the multiply-free substrate loses by
construction (V3 + the capability ceiling). So we borrow standard TASK
definitions (external validity, anti-cherry-pick) and keep OUR metric
(prequential regret, headroom-captured, cost-multiplier, SPJS). That is the E6
"ship the metric, not a new environment" play, brought forward.

## Verdicts

### MinAtar — ADOPT (primary standard env)
- **Verified hands-on (minatar 1.0.15):** pip-installs; raw
  `Environment.act(a)` accepts a constant **6 actions across all 5 games**
  (the gym `-v1` wrapper's variable per-game minimal sets would break a
  fixed-K MPC head — use the raw API); 10×10×C binary board tokenizes to a
  tiny per-game vocab (max 9 → ~13 shared with separators, = our current
  VOCAB); env runs 90k–700k steps/s (never the bottleneck); random agent
  scores on 4/5 games (breakout 44, space_invaders 65, asterix 5, seaquest
  1.3, freeway 0) — learning headroom.
- **Why it's the primary pick:** the streaming-RL method we ALREADY baseline
  against (Elsayed et al. 2410.14606, our `st5_baselines`) has PUBLISHED
  MinAtar numbers — an apples-to-apples anchor no other candidate gives us.
  Game-cycle biomes (Breakout→Freeway→Asterix→Seaquest→SpaceInvaders→revisit)
  are a strong, natural retention story: a SHARED token codebook makes the
  same symbol mean different things per game → genuine interference/retention
  signal, better than the synthetic wall-density biomes.
- **Caveats (all manageable):** GPL-3.0 — keep behind a pip/process boundary,
  do NOT vendor its source into the repo (internal EC2 gate = fine). Heavy
  transitive deps (scipy/pandas/matplotlib/seaborn) — installs on a box, add
  to PIP_EXTRA. Full board can't be cropped → ~100-token frames (~8× our
  per-step cost) → budget fewer total steps / shorter frame-history. MinAtar
  did NOT exhibit the streaming-instability barrier (authors' own words) →
  it's a comparability anchor, NOT a streaming stress test; don't overclaim.

### BabyAI / MiniGrid — ADOPT-WITH-CAVEATS (the gated stretch)
- Apache-2.0 (cleaner than MinAtar); `pip install minigrid`; pure-Python/CPU
  (pygame-ce is a mandatory dep but rendering-only, headless-fine); 7×7×3
  egocentric SYMBOLIC partial view (matches our local-view design better than
  MinAtar's full board); 7 constant discrete actions (prunable to 3 nav-only);
  40+ level curriculum = a natural biome-cycle-with-revisits.
- **Why it's the STRETCH not the first pick:** sparse episodic goal reward +
  episodic resets clash with our dense, continuing-stream, one-step-MPC loop.
  A one-step preference is near-useless when the informative event (reaching a
  goal) is many steps away — it needs a shaped preference (distance-to-goal)
  or a planning horizon, PLUS an episodic-reset adapter (catch
  terminated/truncated, reset internally, emit a boundary token). And there is
  NO published streaming-RL baseline on MiniGrid/BabyAI — no comparability
  numbers. (Two "continual-RL uses MiniGrid" citations in the raw research
  were fabricated arXiv IDs — flagged and discarded.)
- Use nav-only single-instruction levels (skip the mission-language channel)
  to keep the vocab small and the task one-headed.

### Not adopting: image CL suites (CORe50/CLEAR), heavy continual-RL
(Continual World/MuJoCo) — GPU-scale, domains the substrate isn't built for;
we'd defend capability instead of demonstrating cheap adaptation.

## The unifying design insight (env-agnostic reward, and it IS E2)

Our foraging MPC scored a bespoke functional (P(center=food)). MinAtar/BabyAI
have arbitrary game reward. The clean generalization, which every adapter
shares: **add a REWARD TOKEN to each frame** (`[RWD] r` where r ∈ {none, +})
so the world model predicts reward as just another token, and the MPC
preference becomes the env-agnostic **P(next-frame reward token = + | history,
a_cand)**. This is exactly E2 mechanism (b) "reward-as-prediction" from the
roadmap — so the env-adapter design and the E2 reward mechanism CONVERGE:
reward enters the loop as a predicted token, and the gate/MPC read it. One
codec serves the bespoke world, MinAtar, and BabyAI.

## Plan: one adapter interface, three envs, sequenced

`experiments/st5_envs.py`: a common `Env` interface —
`reset() · step(a)->(reward,obs,done) · n_actions · biome · encode_frame(obs,
a, reward) · feat(obs, last_a)` (+ optional `predict_true`). Adapters:
`ForagingEnv` (wrap the existing World — trivial), `MinAtarEnv` (raw API,
board→cell-symbols, game-cycle biomes, reward token), `BabyAIEnv`
(nav-only levels, distance-shaped reward, reset→boundary token).

**Sequence (E1's lesson: pilot cheap, don't build all at once):**
1. Env interface + `ForagingEnv` (conforms already) + `MinAtarEnv`; local
   tokenization/throughput smoke.
2. E2 reward mechanism on MinAtar first (dense reward → the cleanest test of
   "reward in the loop"): neuromodulated gate (a) and/or reward-as-prediction
   (b). Pilot at world/game 0 before any grid.
3. Bespoke foraging = the controlled cross-check (we understand every knob).
4. BabyAI = the sparse-reward/partial-obs stretch, GATED on E2 working on
   MinAtar (its adapter cost only pays once the reward mechanism exists).

Standard V-program disciplines throughout: pre-registration, pre-launch
adversarial review, honest-negative branches, EC2-only, audit coverage.
