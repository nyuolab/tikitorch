# Roadmap: from a persistently learning device to an embodied agent

*2026-07-14. Successor roadmap to the V-gate program (NORTHSTAR_VALIDATION.md,
all gates closed). The organizing idea, carried over from the manuscript's
§6: an embodied agent whose experience IS its training — acting and learning
as one computation — versus every deployed RL system's actor/learner split.*

## What the backbone already gives us

| capability | evidence |
|---|---|
| acting and learning share one forward pass, exactly (≡BP bitwise) | V0 |
| learning costs +0.38 fwd-eq per lesson, ~1.006× amortized over a life | V0, V4 |
| a *when-to-learn* rule that self-quiets (and the law its predecessors break) | V2b, V4cal |
| retention machinery validated at 10M tokens (anchor, decay, probes, audit) | V4 |
| kill-9-resumable always-on operation (organism daemon, ledger) | organism v0 |
| silicon path with bit-exact kernels; learning power below telemetry floor | V5 |
| honest boundaries: capability ceiling, fleet-merge failure, entropy-blind gate | V3, V4b, V4cal |

## E-gates (each pre-registered before running, per house rule)

### E0 — paper-adjacent closures (LAUNCHED 2026-07-14; CPU box)
**E0-1 excess-surprise gate.** Difficulty estimate = online add-1 unigram CE
(prequential-clean, O(V) memory, adds only); gate fires on the two-timescale
margin of ce_model − ce_unigram. Test = the FIXED-order two_cycle regime
(the deadband case the margin gate is blind to). **Pass (pre-registered):**
excess HC(two_cycle) ≥ margin HC(two_cycle) + 0.05, AND excess stationary
fire < 1%, AND excess HC(hard_shift) ≥ 0.9× margin's V4cal hard_shift HC
(0.616). **Kill:** excess ≤ margin on two_cycle ⇒ unigram difficulty doesn't
capture re-entry; E1 keeps rotation and the finding is published.
**E0-2 multi-seed life.** V4 main-life config verbatim, seeds 1 and 2.
**Pass:** both reproduce the V4 criteria (HC ≥ 0.5, norm-slope deceleration,
per-cycle fire collapse). **Kill:** either fails ⇒ the V4 verdict carries a
seed-fragility caveat.
**E0-2 VERDICT (2026-07-15): PASS — V4 reproduces across 3 seeds.** HC
0.602/0.547/0.567 (mean 0.572 ± 0.028, bar ≥0.5 ✓ all); regret −2.62/−2.38/
−2.46 (mean −2.485 ± 0.122); fire 1.65/1.36/1.72% (mean 1.58%); amortized
1.0063/1.0052/1.0065× (mean 1.0060×); norm +2.17/+2.33% (equilibrium); **0
audit violations across all 3 lives.** The V4 single-seed caveat is CLOSED;
N3 upgrades to a 3-seed number.
**E0-1 VERDICT (2026-07-15): KILL on the self-quieting condition.** The
excess gate met the benefit bars (hard_shift HC 0.687 ≥ 0.554 ✓; two_cycle
HC 0.649 vs margin+0.05 = 0.648 — met by 0.001, single seed, i.e. no real
edge) but FAILED stationary self-quieting decisively (fire 17.6% vs the <1%
bar): subtracting a second noisy estimator (unigram CE) adds variance to
the gated quantity and the margin fires on that noise. The symmetric
counterpart to the V2b law: self-normalization erases regime information;
difficulty-normalization via a noisy reference reintroduces a variance
floor. Exploratory (tele-sampled, biased): cycle-2 regrets are essentially
tied (−2.54 margin vs −2.59 excess). **Per the kill branch: E1 runs the
plain margin gate δ=0.25 + rotated biome schedule.** A low-variance
difficulty estimate (slow-EMA'd or probe-based) is the follow-up idea, not
assumed.

**E0-3 strict adversary.** PureTropicalSGD (lr = 2^-10, momentum 4 bits) +
bit-shift decay (lr_shift+1 per 2000 fires), anchorless, fire-always
stationary 20,833 steps — the SAME optimizer family whose 6k-step collapse
set the horizon. **Pass:** end prequential CE ≤ start (no collapse) ⇒ the
V4 AdamW caveat closes. **Kill:** collapse ⇒ the caveat stands; decay
cadence retuned before any strict-config life.
**E0-3 VERDICT (2026-07-15): KILL.** Cumulative preq CE 5.34 vs frozen 3.88
(regret +1.45) despite all 10 halvings firing on schedule — the drift is
DECELERATED by the decay (visible in the trajectory) but the damage
accrues before the cadence bites. Audit: 0 violations (the closed path is
clean; it is not yet STABLE at this cadence). The V4 AdamW caveat stands.
Retune direction for the follow-up cell: faster early decay (halve per
~500 fires) and/or birth lr_shift 12.

### E1 — action without an actor/learner split (the decisive gate) — KILL (2026-07-15)
**VERDICT: KILL, mechanism mapped (E1_SPEC "E1 VERDICT").** Prediction-only
in-process learning improves the world model (obs-CE down; action-conditional
signal up with birth) but greedy 1-step MPC over it does NOT forage — and a
BETTER-predicting model forages WORSE than random (return/1k 26→11→2→0 as
spread 0.00→0.46→0.53, 2 seeds, under the softmax controller). Prediction and
control are decoupled/anti-correlated: confident myopic exploitation of
residual miscalibration underperforms random ground-covering. The live-loop
mechanism (Goal A) is intact; prediction ALONE is not control. Pilot caught
this at ~$8 (no grid launched). **E2 is now the REQUIRED next gate.** Original
design below (retained for the record).


**Question:** can the SAME forward pass produce an action *and* be the
sufficient statistic for learning, in a closed sensorimotor loop?
**Design:** an agent in a procedural text/grid world; the substrate's forward
emits (i) next-action logits and (ii) a prediction of the next observation
*conditioned on the chosen action* (action-conditional world model — the
predictive-processing move). Learning = the fused cascade kick on observation
prediction error, margin-gated; NO value function, NO replay buffer, NO
separate learner. **Baselines:** streaming DRL (Elsayed et al.) and a
replay-buffer actor-learner at matched compute and matched wall.
**Pass:** the agent's competence (task return + prediction quality) improves
over its life at ≤1.1× acting cost, with the same retention/audit properties
as V4. **Kill:** if pure prediction-error learning cannot move the policy,
the claim demotes to "perception learns in-process; control still needs a
second process" — published as the boundary.

### E2 — reward IN THE LOOP — KILL (2026-07-16, pre-registered branch)
**VERDICT (E2_SPEC "E2 v2 RESULT"):** with the gate calibrated (E2cal) and
fair (13–17% fire), neither mechanism produces control: nothing beats
surprise-only in either env; frozen-birth BEATS all live cells on foraging
(online kicks degrade control while improving prediction — E1's decoupling at
fair firing); 2-ply planning makes it worse, not better; a+b interaction
moot. The remaining escalation (a value estimate) reintroduces the
actor/learner machinery the minimality claim excludes — that IS the boundary.
E1+E2 together map it with mechanisms at every step (~$60). The V-program's
prediction-side results are untouched. Original design below (for the record).

### E2 — reward IN THE LOOP (PROMOTED to the decisive gate by E1's kill; three mechanisms)
**Environments (E2_ENV_DECISION.md, 2026-07-15):** run standard benchmarks
ALONGSIDE the bespoke world through one adapter. MinAtar = ADOPT/primary
(verified: constant K=6 raw API, ~13-symbol board vocab, our streaming-RL
baseline has published numbers here, game-cycle biomes); BabyAI/MiniGrid =
adopt-with-caveats/stretch (Apache, partial-obs, but sparse episodic reward
clashes with one-step MPC — gated on E2 working on MinAtar). Unifying design:
a REWARD TOKEN per frame makes the MPC preference env-agnostic (P(next reward))
— which IS reward-mechanism (b). Sequence: MinAtar first, bespoke cross-check,
BabyAI stretch.
(a) **Neuromodulated gate:** fire the kick only on surprise × reward-salience
coincidence (a scalar gate input, not a gradient path) — keeps strict
closure. (b) **Reward-as-prediction:** add a reward channel to the predicted
stream; act by settling the generative PC iteration toward
high-predicted-reward futures (revives the G6 iterative machinery for
planning — pay iteration only where you want generative behavior, exactly
the original two-mode north star). (c) **Advantage-weighted cascade:** scale
routed errors by a scalar advantage — one scalar multiply per window; audit
implication declared (dyadic-quantize it or waive it explicitly).
Run (a) first: it is closure-clean and needs no new theory.

### E3 — the correct loss (policy-as-token-prediction) — BISTABLE IGNITION (2026-07-16)
**VERDICT (E3_SPEC "RESULTS"):** neither prereg branch as worded. Not PASS
(polA under both floors at arm level: foraging mean 17.3 vs frozen 28.2 /
random 36.3). Not KILL-as-attributed: the kick IS REINFORCE (one-kick
direction test) and 3/12 policy cells IGNITED into compounding policy
learning through the fused kick — foraging polA s1 Q4 = 44.2 (above both
floors), s2 trap→ESCAPE→44.5, minatar polG s2 full-life 39.7 (above floors
AND the matched-steps streaming-AC 31.9). The failure is OBJECTIVE
COMPOSITION: the dense world-model CE (~60–200 obs positions/step) shares the
trunk with a sparse advantage term (~99% of kicks carry none), creating a
dark-room attractor (learned predictable dither, CE ≪ frozen, zero reward)
that captures cold starts by luck; gate-rate is a dose (polG mostly starves
ignition), environment geometry a gradient (minatar polA 0/3). The user's
"correct loss" hypothesis vindicated: E1/E2's boundary was the loss, and the
substrate learns policies when given one — reliability of IGNITION is the
open problem (E4 candidates in E3_SPEC: intrinsic-surprise reward, advantage
normalization, model-based birth actor, entropy bonus). ~$10.

### E4 — ignition reliability: four levers — TWO WINS (2026-07-17)
**VERDICT (E4_SPEC "RESULTS"):** the dark-room attractor is FIXABLE at zero
substrate cost. Sentinels reproduced E3 exactly (screen quarantine-free).
**intr WIN** (surprise-credit: 5/5 foraging ignitions, mean 45.3 vs ctrl
21.7; audit: intrinsic mass co-dominant, returns extrinsic-only) and
**norm WIN** (unit-scaled advantage: 5/5, mean 44.8, program-best cells
50.6/52.2, AND the only lever that moves minatar — 3/3 Ignite, 2/3
sustained, late-life above streaming-AC's full-life). mbwarm = variance
compressor (HARM on rate, mean uplifted — cuts both tails); ent =
rescue-to-random (pinned at the 36.3 floor), as predicted. Post-hoc
sustained metric: ctrl 0/5 vs intr 4/5 / norm 3/5. Two independent levers
each turn E3's bistable coin-flip into reliable compounding policy learning
at the same serve+learn budget. Frontier: minatar window geometry. E4b
candidates in spec (intr+norm combo; norm+longer credit horizon; gating
re-entry). ≈$21.

### E4b — combo + credit horizon — STACK / HORIZON-MOOT (2026-07-17)
**VERDICT (E4B_SPEC "RESULTS"):** combo (intr+norm) STACKS — foraging 4/5
sustained, mean 48.5 with EVERY seed above both single-lever means (tightest
spread of any arm); minatar mean +29% with the program-best cell (39.9
full-life, Q3/Q4 ≈ 60% above streaming-AC). normw512 HORIZON-MOOT: +23%
mean at a MEASURED 2.9x serve tax, dominated by the free combo — the
credit horizon is not minatar's binder (obs:action gradient volume remains
prime suspect). THE RECIPE: token-policy + dyadic advantage + surprise
credit + advantage normalization, fire-always = reliable cheap policy
learning on the fused substrate. Open: gating/efficiency (E4c), the 2.8x
gap to streaming-AC on foraging, minatar universality. ≈$22.

### E5 — the second life (months-long persistence)
100M+ tokens / multi-month always-on agent on one box: organism daemon +
V4-life config + excess-surprise gate + periodic Fisher-probe consolidation.
Weekly probe suites; the pre-registered adversary is slow drift (the 10M
life's open tail). This is cheap (CPU/one F2) and is the artifact that makes
the "persistent embodied agent" claim visceral — a thing that has been alive
and learning for months, every update attributable.

### E5 — the agent at the sensor (silicon)
Port the fused loop end-to-end to the F2 kernel path (gate: the localized s2
backward divergence); then the E1 loop on-fabric. V5's floor number feeds
the ASIC study: the target artifact is a milliwatt-class always-plastic
sensor-edge learner, where the GPU's 79 W stream-sitting floor is the
comparison that matters.

### E6 — collective ontogeny (fleet)
N embodied agents, disjoint experiences, firing-normalized/probe-based merge
(V4b.1 fix). Pass = merged-beats-solo on held-out worlds WITHOUT sharing
data. This is the fleet-experience-scaling bet, now with the V4b failure
mode designed out rather than assumed away.

### E7 — the public benchmark
SPJS harness (score-per-joule-per-stream) with the embodied stream tasks +
GPU baselines, shipped as the community artifact; asymmetric honesty (the
fabric loses columns) as the adoption mechanism.

## Sequencing and spend

E0 ✅($8) → E1 KILL($8) → E2 KILL($60) → E3 BISTABLE($10) — the ladder
re-sequenced by results: the actionable next gate is an E4 DESIGN DECISION on
ignition reliability (E3_SPEC candidates); the persistence life (E4 above)
starts once a policy loop is reliable OR runs prediction-only on the V-config;
silicon (E5) gated on the s2 fix; fleet/benchmark (E6/E7) after E4.
(Renumbered 2026-07-16 when the correct-loss gate took the E3 slot.)
Every gate keeps the V-program disciplines: pre-registration, honest-negative
branches, adversarial pre-launch review, EC2-only compute, audit coverage.
