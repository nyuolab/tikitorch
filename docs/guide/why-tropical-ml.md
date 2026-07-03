# Why tropical algebra for ML/DL

Tropical algebra is not just a curiosity — restricting a network to the min-plus /
max-plus semiring buys concrete properties that ordinary (multiply-based) deep
learning cannot have. It also costs something, measurably. This page is the honest
case for *why* you would do it.

## 1. Multiplier-free → zero-DSP hardware

A tropical layer is `min`/`max` of `(weight + activation)`: only **adders and
comparators**, never a multiplier. That is the whole point. On an FPGA, multiplies
consume scarce **DSP blocks**; a network whose forward pass, backward pass, *and*
optimizer are all tropical synthesizes to **zero DSP blocks** — the silicon budget
goes to more parallel adder lanes instead.

TikiTorch is built so the *entire learning loop* stays closed:

- **Forward**: tropical linears/convs/attention (compare + add).
- **Backward**: gradients route through the argmin/argmax (a scatter, no multiply).
- **Optimizer**: sign-based (`SignSGD`) or power-of-two-step (`TropicalAdam`,
  `PureTropicalSGD`) — the learning-rate is a **bit-shift**.
- **Loss**: `tropical_cross_entropy(softmax="lut")` computes softmax from a
  subtract-max, a power-of-two-addressed exp lookup, and a shift-normalize.

The architectures shipped here were trained **end-to-end on AWS F2 silicon**
(Virtex UltraScale+) at single-digit millijoules per token, DSP=0. See the measured
throughput/energy table in the [README](https://github.com/nyuolab/tikitorch#readme)
and [notebook 01](https://github.com/nyuolab/tikitorch/blob/main/notebooks/01_intro.ipynb).

## 2. A routing inductive bias

Because each output picks a single winning input, a tropical net **routes** rather
than **mixes**. This is a qualitatively different prior:

- It is exactly the right primitive for **algorithmic / combinatorial** tasks —
  shortest paths, dynamic programming, sorting — which *are* min-plus recurrences
  (see [Beyond deep learning](beyond-deep-learning.md)).
- **Tropical attention** routes by an L1/L∞/Hilbert distance + an argmax *gather*;
  on content-routing tasks (induction/copy) a fully multiplier-free attention matches
  the softmax ceiling, *if* deployed as argmax-gather and trained with the soft
  relaxation.
- Max-plus residuals (`max(relu(Wx), x)`) are more depth-stable than additive
  residuals in several of our scaling probes.

## 3. Quantization- and robustness-friendly

Adds and shifts compose cleanly in fixed point — there is no multiply to blow up the
bit-width, so INT16 (or narrower) is natural and the quantization penalty is small
(measured ~−1pp; see below). The piecewise-linear, order-based structure also gives
**exact, certifiable** behavior in places where smooth nets are only approximate
(the tropical geometry of decision regions is literally a polytope arrangement —
[Tropical geometry of neural nets](tropical-geometry.md)).

## The closure penalty

Forcing the whole loop into the semiring with sign-only updates costs accuracy versus
an unconstrained `AdamW + softmax-CE + FP32` baseline — the **closure penalty**. The
point of TikiTorch is *not* to beat a GPU transformer on perplexity; it is to make the
loop closed and to give you the levers to climb the penalty back. On a hard
character-LM the penalty **decomposes (measured)** roughly as:

| source | cost |
|---|---|
| SignSGD vs AdamW | ≈ −13 pp |
| perceptron loss vs cross-entropy | ≈ −17 pp |
| INT16 quantization | ≈ −1 pp |
| hard tropical ops (vs soft) | ≈ −5 pp |

The two big terms are *optimizer* and *loss conditioning*, not the tropical ops
themselves — and both have multiplier-free fixes:

- **Loss**: swap the perceptron loss for `tropical_cross_entropy` /
  `nn.TropicalCrossEntropyLoss` — `softmax="lut"` recovers most of the −17pp **without
  a hardware multiply** ([notebook 04](https://github.com/nyuolab/tikitorch/blob/main/notebooks/04_sequence.ipynb)
  reproduces the recovery live).
- **Optimizer**: the adaptive family (`TropicalAdam`, the magnitude-aware optimizers)
  closes much of the SignSGD gap with power-of-two per-parameter steps.
- **Ops**: the `tau` soft→hard curriculum recovers the −5pp by training smooth and only
  hardening at the end.

On easier tasks (e.g. MNIST/CIFAR classification) the penalty is a few points and a
declared-`nn.Linear`-head configuration reaches ~96–97%. Always report both the closed
number and the boundary-head number — that is the honest comparison, and it is the
discipline the whole library is built around (`assert_closed` makes the claim
mechanical, not rhetorical).

## When *not* to use it

If you have a GPU, plenty of DSP/energy budget, and only care about raw perplexity,
ordinary dense deep learning wins — the multiply does real work. Tropical pays off
when the constraint is **hardware** (no multipliers / energy / FPGA-ASIC deployment),
when the task has **combinatorial structure**, or when you want **certifiable,
piecewise-linear** behavior. See [Choosing a model](choosing-a-model.md) for the
decision recipe.
