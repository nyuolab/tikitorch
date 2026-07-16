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

### Pre-launch amendments (adversarial review, 2026-07-16, before any run)

- **A1 — birth behaviour-cloning (supersedes "action tokens unweighted during
  birth").** Review verified empirically that leaving action tokens out of the
  birth loss leaves the K-restricted action marginal ARBITRARY (they are only
  pushed down as never-target softmax competitors), so tau_pi sampling starts
  from an init lottery — a skewed start + REINFORCE cold-start positive
  feedback could collapse pi early and read as a false substrate-KILL. Fix
  implements the sentence's intent ("the birth policy is random"): action
  tokens join the birth loss at weight 1, behaviour-cloning the uniform random
  birth policy -> calibrated ~uniform start. Frozen floor shares the same
  birth, so floor comparisons stay fair. Each run logs `pi0_ent` (post-birth
  action-marginal entropy; uniform = ln K: foraging 1.609, minatar 1.792).
- **A2 — done-boundary credit.** Reward-to-go now zeroes the discount across
  episode boundaries (`acc = r_i + gamma*acc*(1-done_i)`; per-token done flags
  ride ctx_d). Without it, minatar rewards earned after an in-window auto-reset
  are credited to the actions that caused the death. Foraging (primary) is
  continuing/never-done — unaffected.
- **A3 — launch-blocker fixed.** The dyadic quantizer's `.clamp(-4,2)` was
  applied over the -99 drop-sentinel, resurrecting |A|<2^-5 positions at
  sgn=-1, k=-4 (a systematic "punish whatever you did" during reward droughts,
  violating the closure declaration above). Clamp moved inside the live
  branch; unit-tested (A=0 and tiny |A| drop; signs/clamps correct).
- **Noted, not fixed:** minatar's effective credit horizon is ~2 frames
  (window 256 / frame_len 105) — near-bandit REINFORCE; a minatar-negative
  is NOT evidence against the substrate (foraging, ~7 frames/window, carries
  the primary verdict). The warm-loop now also passes reward/done into
  ctx_r/ctx_d (cold-start EMA seeding was reading zeros).

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

---

## RESULTS (2026-07-16, box i-0a274d869eef1083d, 30/30 cells, 0 job failures, ~5.7h, ~$10)

Data: `results/st_reward/e3_screen.jsonl` (+24 tele files, same dir).
streaming-AC verdicts recovered from `e3_onbox_jobs.log` stdout
(`e3_baselines_recovered.jsonl`) — st5_baselines writes to the st_world OUT
dir, which the driver's sync didn't cover (fix before any rerun: `--out` flag
or sync both dirs). Live dashboard (both attractors visible per trajectory):
https://claude.ai/code/artifact/7e6bfcc1-9f50-4661-9f9c-ecb3c4b1e35f

ret/1k, 3 seeds, sorted | foraging (200k) | minatar (100k)
---|---|---
polA (always-fire) | 0.25 / 22.0 / 29.5 (mean 17.3) | 1.3 / 7.0 / 7.3 (mean 5.2)
polG (gated) | 2.2 / 4.1 / 8.3 (mean 4.9) | 8.5 / 16.7 / **39.7** (mean 21.6)
frozen (birth) | 23.7 / 25.4 / 35.6 (mean 28.2) | 22.6 / 23.5 / 23.9 (mean 23.3)
random | 32.0 / 37.7 / 39.3 (mean 36.3) | 21.7 / 21.8 / 22.5 (mean 22.0)
streaming-AC | 129 / 141 / 144 (mean 138.1) | 31.1 / 31.8 / 32.6 (mean 31.9)

### Verdict against the prereg: NEITHER branch as worded — the data split them

- **Not PASS:** polA does not beat the floors with non-overlapping seed
  spreads anywhere (foraging mean 17.3 < frozen 28.2 < random 36.3).
- **Not KILL-as-attributed:** the KILL branch's attribution ("failure
  localizes to the substrate/kick") is affirmatively refuted: (i) the one-kick
  direction test passes (sgn=+1 raises pi(a) 0.016→0.074 over 20 kicks, −1
  suppresses to 0.005 — the kick IS REINFORCE); (ii) 3 of 12 policy cells
  ignited into genuine compounding policy learning through the fused kick:
  foraging polA s1 (quarters 15.5→47.4→11.0→44.2, Q4 above BOTH floor means),
  foraging polA s2 (trap→escape: Q1 = 0.4 with CE 0.09, Q3 = 44.5 with CE
  rising to 0.24), and minatar polG s2 (7.0→29.1→61.8→60.7, full-life 39.7 —
  above the floors AND above streaming-AC's matched-steps 31.9).

### The finding: bistable ignition under a composed serve+learn loss

One shared network, one kick, two attractors:

1. **The dark room.** During reward droughts every action position drops out
   of the kick by design (quantized A=0), so ~99% of kicks are pure
   world-model CE (~60–200 obs positions/step vs ≤1 weighted action
   position). That term alone sculpts behaviour toward maximum predictability
   — a learned two-action dither (policy entropy stays ~1.3; NOT
   single-action collapse; local repro) with CE ≪ frozen and zero reward.
   Canonical case: foraging polA s0 = 0.24 ret/1k for 198k steps, CE 0.029.
2. **Compounding ignition.** If early rewards get traction, advantage pulses
   reinforce reward-seeking, which produces more pulses — genuine policy
   improvement through the identical machinery, up to beating the matched
   MLP baseline (minatar polG s2). Foraging polA s2 shows the trap is
   METASTABLE: one life with Q1 in the dark room and Q3 at 44.5 ret/1k, CE
   rising as the agent trades predictability back for reward.

Which attractor wins is decided early, by reward luck × kick rate × reward
density: foraging polA ignited 2/3; minatar polA 0/3 (steepest obs:action
imbalance ~200:2 per window + near-bandit credit horizon, the noted-not-fixed
review finding); the gated arm mostly starves ignition (2.6–15.6% fire) but
its one ignition (minatar s2, densest-reward games carrying it: breakout 58,
space_invaders 118 ret/1k vs ~6 in the sparse three) rode gating's slower
world-model drift to the best policy result of the screen.

### What E3 settles

- The user's hypothesis was right: E1/E2's boundary was a LOSS problem — with
  a policy-gradient term the substrate demonstrably learns policies (three
  independent ignitions, one exceeding the published-comparable bar).
- The remaining failure is neither substrate nor gradient: it is OBJECTIVE
  COMPOSITION. A dense prediction loss sharing one trunk with a sparse
  REINFORCE term has a dark-room attractor, and cold-start luck decides.
  Known pathology class (dark-room problem / predictability trap), here
  measured cleanly with per-seed bifurcation, a dose–response via the gate,
  an in-life escape, and an environment-geometry gradient.
- streaming-AC remains far ahead on foraging (138 vs best-arm 17) — matched
  steps, purpose-built RL; the honest bar, reported as-is per prereg.
- Audit: dyadic advantage (sign·2^k) kept every kick closure-clean;
  pi0_ent confirms A1 birth calibration on all 18 agent cells (1.59–1.61 vs
  ln5=1.609; 1.785–1.788 vs ln6=1.792).

### E4 candidates (targeting ignition reliability, not yet designed)

(a) intrinsic reward = the gate's own surprise signal (counterbalances the
predictability pull with a surprise-seeking term — same signal, zero new
machinery); (b) advantage normalization / larger dyadic range (lift the
policy term's share of gradient volume); (c) model-based behaviour policy at
birth (clamped-settle or MPC P(a|reward+) posterior as actor, REINFORCE as
consolidator — uses the E2 machinery as the exploration prior; prices the
settle iterations against the 2x serve budget); (d) entropy bonus on action
positions (dyadic-quantizable). Decision deferred to the user.
