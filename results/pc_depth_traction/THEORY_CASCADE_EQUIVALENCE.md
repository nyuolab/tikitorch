# When does the single-sweep cascade equal settled iterative PC?

**Claim.** For a feedforward DyadicPCNet with the CE readout, the one-pass **cascade**
gradient equals the gradient produced by **iterative predictive coding settled to its energy
minimum**, exactly, under three conditions — and the repo's measured gradient checks are the
empirical confirmation of each.

## Setup

PC minimizes the layerwise energy `E = Σ_l ½‖x_l − f_l(x_{l−1})‖²` (top layer clamped to the
target). *Iterative* PC settles the free activations `x_1..x_{L−1}` to `∂E/∂x = 0` (inference
phase), then takes a local weight step `∂E/∂W_l` at the settled state. *Cascade* skips the
settling loop: it does one forward pass, injects the top error, and sweeps it down once,
updating each `W_l` from the adjacent errors.

## The three conditions for exact equality

**C1 — feedforward (acyclic) graph.** With no lateral/recurrent coupling, the stationarity
condition `∂E/∂x_l = 0` is a single back-substitution from the top: the settled error at
layer `l` is exactly the top error propagated through the transposed Jacobians of layers
`l+1..L`. One downward sweep computes that back-substitution in closed form — no iteration is
needed to reach the fixed point. (On a cyclic graph the fixed point requires true relaxation;
this is why Salvatori et al.'s arbitrary-graph PC needs settling and the cascade does not
apply.)

**C2 — the readout/loss is the standard chain-rule target.** With the CE readout and the top
layer clamped to the target, the injected top error equals `∂CE/∂logits`; the downward sweep
then reproduces the chain rule term-for-term. This is exactly the repo's finding that
**`cascade_grads` with `loss="ce"` matches backprop at depth** (pc-xformer-build-decision:
single + 2-block-chained grad-checks give sign=1.0, cos=1.000000, relL2 ≈ 2e-12 vs torch
autograd). The lone by-design deviations are the dyadic pow-2 weight-grad rounding
(sign-correct) and the stop-grad LN scale — i.e. equality holds up to the deliberately
quantized wgrad, not because the algorithm differs.

**C3 — the settling is run to convergence with a stable step.** `SCALE_TESTS.md` records the
operational form: at `dE/dx = 0` the PC errors ARE the backprop errors, so a converged
iterative run reproduces the cascade. The failure modes are all C3 violations: **truncated**
settling (n_iter too small for the depth) or an **unstable** step (the U-gate result: plain
fixed-step descent diverges at width 768; RMS-normalization restores stability but not yet
credit quality). Async/partial relaxation (participation < 1, stale messages) is a *stochastic
truncation* of C3 — hence the repo's measured ~0.6pp async tax and the Exp-1 question of
whether that truncation stays bounded or compounds with depth.

## Consequences (what the two experiments test)

- **Exp 1** measures C1+C2 directly: sync relaxation run long enough must match the cascade
  (cos → 1) — a positive control for the equivalence — while async/truncated runs quantify the
  C3 gap by depth position.
- **Exp 2** asks the orthogonal question the equivalence *cannot* answer: even where cascade ≡
  settled-PC on gradients, does the resulting **trained** model retain quality as depth grows,
  and how much of any gap is the dyadic substrate (bp_dyadic − bp_float) vs the local rule
  (relax_async − bp_dyadic)?

**Bottom line.** Cascade is not an approximation to iterative PC on feedforward nets with a
chain-rule loss and converged, stable settling — it is the closed-form evaluation of the same
fixed point. Deep-PC "traction" therefore hinges not on cascade-vs-iterative (they agree) but
on (a) keeping C3 satisfied at scale — the conditioning problem the U-gate is chasing — and
(b) the substrate/rule taxes Exp 2 measures.
