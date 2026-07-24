# External peer feedback (GPT Sol, 2026-07-24) — adjudicated

User looped an external model in on the E6 interim brief. Each point checked
against our implementation before adoption.

## 1. ent-pull framing — ADOPTED (and it sharpens our dissociation)
Verified against the engine: action_uniform_pull adds 2^k·(softmax − U) at
action positions = the gradient of λ·CE(U, π) = λ[log K + D_KL(U ∥ π)] —
a **uniform-target FORWARD-KL anti-saturation regularizer**, NOT SAC-style
max-ent (which is reverse-KL D_KL(π ∥ U) up to a constant). Forward KL
punishes near-zero action probabilities much harder — which is exactly why
it desaturates a frozen head while the ε-mix (which never touches the
learner's gradients) failed 0/5. The theory now MATCHES the measured
ε-floor/ent-pull dissociation. Writeup citations: label smoothing
(Szegedy et al. 2016), confidence penalty (Pereyra et al. 2017),
max-ent RL (Ziebart; SAC) — with the KL-direction distinction stated.

## 2. "Entropy homeostasis" language — ADOPTED (retreat)
Correct epistemic call, matching our own skeptic: the cross-task
restoration event (world-A entropy restored during world-B learning) is
n=1 and not yet demonstrated by current instrumentation. Language until
the Tier-3 branch control reports: "cross-task entropy RESTORATION
(observed once; no direct precedent located)"; "homeostasis" only if the
novel-vs-familiar-vs-scrambled branches confirm the causal attribution.

## 3. IDBD θ too fast — ADOPTED for a possible IDBD-v2 mini-screen
With Autostep bounding |Δβ| ≤ θ per update, θ=0.01 traverses one octave
per 100 aligned kicks and the full 8-octave range in 800 — inside a
6M-kick correlated life, effectively per-weight bang-bang. Fire-always is
NOT the conceptual violation (IDBD was designed for high-frequency
updates); the meta-rate is. If IDBD is pursued past the screen negative:
sweep θ ∈ 1e-5…1e-3; add θ=0 (AdamW-parity) and no-decay controls as LIFE
arms, not just unit tests; log per-module β-bound saturation (the beta
telemetry exists); try **IDBD on the action head only** before all-weights
(cheap, targeted at the actual failure site).

## 4. E6-S capability composite — ADOPTED into the E6-S draft
Per-skill normalized competence c_k(t) = (R_k(t) − R_k^random) /
(R_k^single-task − R_k^random). We already possess both anchors:
random floors (E5b + screen floors) and single-task references (the st6
single-env screens). Primary outputs: mean retained competence,
worst-skill floor, #skills above threshold, acquisition AUC,
forward-transfer gap closure, normalized world-model headroom, lifetime
utility, joules. Single scalar only if unavoidable: PRE-REGISTERED
geometric mean of (policy, model, transfer) gated by a worst-skill
retention floor.

## 5. The word "law" — ADOPTED as the E6-S bar
Return-rises-with-steps is every RL curve. E6-S may claim a "scaling law"
ONLY if a fitted relation across (lifetime, capacity, task diversity)
predicts HELD-OUT scales: ≥3-4 model sizes, fresh task-generator seeds,
randomized era orders, frozen algorithm. This sharpens (does not replace)
the existing existence+shape bars incl. the honest flat-curve negative.
