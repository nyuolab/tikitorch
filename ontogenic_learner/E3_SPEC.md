# E3 — the correct loss: policy-as-token-prediction (prereg, FROZEN 2026-07-16)

*User direction (2026-07-16): E1/E2's failures may be a LOSS problem, not a
substrate problem — the loop never contained a policy-improvement operator.
E3 gives it one, minimally, and tests whether the substrate learns policies
when actually asked to.*

## The diagnosis E3 tests

E1+E2 mapped: prediction-only learning + hand-derived control ≠ policy
learning. Nothing in those results says the substrate cannot learn policies —
its only learning signal had a PREDICTOR as its fixed point. The control:
streaming-AC (22k-param MLP) learned these same tasks at the same step
budgets (175–223/1k foraging vs our ~34), so sample budget was not the
binder; the loss was.

## The mechanism: the action tokens ARE the policy

Frames already carry action tokens; the model already predicts them (masked
out of the loss until now — "policy tokens carry no loss" was the E1 rule).
E3 flips that rule deliberately:
- **Acting:** sample the next action from the model's own action-token
  distribution (temperature tau_pi). Policy and world model are the SAME
  network; acting costs ONE forward/step (cheaper than the K-forward MPC).
- **Learning:** the same fused cascade kick, with a per-position error weight:
  obs (+reward) tokens keep weight 1 (world-model loss); each frame's
  action-token error is weighted by the frame's ADVANTAGE
  A_t = (discounted reward-to-go over the next n frames) − (baseline EMA).
  Gradient of −A·log π(a_t) at the logits = A·(softmax − onehot) — i.e.
  REINFORCE-with-baseline implemented through the existing kick. One network,
  one forward, one kick: the policy-gradient form of "serving is learning."
- **Closure declaration:** A_t is quantized to sign·2^k (dyadic advantage,
  k clamped to [−4, 2]); the weighting is applied as exponent arithmetic
  (shift-class, the same family as the substrate's pow-2 LN factors). Zero
  quantized advantage ⇒ the action position drops out of the kick.

## Arms (per env; envs = foraging + minatar per-game)

1. **pol-always** — policy-token learning, every window fires (give the loss
   its best chance; the gate question is secondary here).
2. **pol-gated** — same, margin gate at the E2cal δ (the efficiency variant).
3. **frozen** — birth model, sampling policy, no learning (floor).
4. **random** — uniform actions (floor).
5. **streaming-AC** — at MATCHED steps (the published-comparable bar).

3 seeds. Horizons: foraging 200k steps, minatar 100k (acting is 1 fwd/step —
cheap). Birth: same tau-annealed prelife (action tokens unweighted during
birth — the birth policy is random, there is nothing to reinforce).

## Pre-registered verdict

- **PASS (the substrate learns policies):** pol-always return beats frozen
  AND random with non-overlapping 3-seed CIs on foraging; trajectory shows
  sustained improvement (last-quarter > first-quarter post-birth). Beating
  streaming-AC is NOT required (it's the mature-baseline bar; parity or a
  fraction at matched steps is reported as-is).
- **KILL (the substrate/kick, not the loss):** pol-always ≤ floors at 200k
  on foraging ⇒ with a genuine policy gradient, matched budget, and a task
  the MLP solves, the failure localizes to the substrate/kick machinery —
  the strongest negative the E-program could produce, published as such.
- Secondary: pol-gated vs pol-always (can gating coexist with REINFORCE
  variance?); minatar per-game; obs-CE (does policy learning damage the
  world model?); audit (dyadic-advantage declared).

## Costs

Engine: a per-position dyadic loss-weight (exponent arithmetic) — regression
bar: weights-all-one path stays bit-exact vs cascade_grads. Agent: sampling
act + reward-to-go ring + advantage quantizer. One c7i.8xlarge screen ~2-3h
(~$10). House rules: review-before-launch, floors-first, honest branches.
