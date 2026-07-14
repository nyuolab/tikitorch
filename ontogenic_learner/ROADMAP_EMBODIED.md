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

### E0 — paper-adjacent closures (weeks; CPU boxes)
Excess-surprise gate (margin vs a cheap difficulty estimate — fixes
easier-domain-entry blindness); multi-seed V4 life (the single-seed
limitation); strict shift-optimizer decay (close the last AdamW dependency).
Kill criteria inherited from V4.

### E1 — action without an actor/learner split (the decisive gate)
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

### E2 — reward without a learner process (three pre-registered mechanisms)
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

### E3 — the second life (months-long persistence)
100M+ tokens / multi-month always-on agent on one box: organism daemon +
V4-life config + excess-surprise gate + periodic Fisher-probe consolidation.
Weekly probe suites; the pre-registered adversary is slow drift (the 10M
life's open tail). This is cheap (CPU/one F2) and is the artifact that makes
the "persistent embodied agent" claim visceral — a thing that has been alive
and learning for months, every update attributable.

### E4 — the agent at the sensor (silicon)
Port the fused loop end-to-end to the F2 kernel path (gate: the localized s2
backward divergence); then the E1 loop on-fabric. V5's floor number feeds
the ASIC study: the target artifact is a milliwatt-class always-plastic
sensor-edge learner, where the GPU's 79 W stream-sitting floor is the
comparison that matters.

### E5 — collective ontogeny (fleet)
N embodied agents, disjoint experiences, firing-normalized/probe-based merge
(V4b.1 fix). Pass = merged-beats-solo on held-out worlds WITHOUT sharing
data. This is the fleet-experience-scaling bet, now with the V4b failure
mode designed out rather than assumed away.

### E6 — the public benchmark
SPJS harness (score-per-joule-per-stream) with the embodied stream tasks +
GPU baselines, shipped as the community artifact; asymmetric honesty (the
fabric loses columns) as the adoption mechanism.

## Sequencing and spend

E0 now (≈$20, days) → E1 (the decisive gate; ≈$50-100, 2-3 weeks incl. world
+ baselines) → E2a riding E1's harness → E3 starts the moment E1 passes
(background, ≈$30/month) → E4 gated on the s2 fix → E5/E6 after E1+E3.
Every gate keeps the V-program disciplines: pre-registration, honest-negative
branches, adversarial pre-launch review, EC2-only compute, audit coverage.
