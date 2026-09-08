# Iterative tropical PC: a local/async hardware-mapping candidate

**Date:** 2026-06-17
**Code:** `tikitorch/nn/predictive_coding.py` (`DyadicPCNet.settle_nudge`,
`train_step_nudge`); experiments `scripts/pc_dyadic_deep.py` (`harden`, `hybrid`,
`async`, `nudge*` modes); tests in `tests/test_predictive_coding.py`.

> **Implementation correction (2026-09-06):** the archived `nudge`, `harden`,
> `hybrid`, and async-nudge measurements below predate a fix that made
> `settle_nudge` apply its configured relaxation factor `gamma`; those runs used
> an effective unit step. The values remain provenance for the historical run,
> but are not current-code benchmarks and must be rerun before being cited as
> such. The conceptual hardware motivation remains, but corrected-gamma behavior
> and end-to-end physical closure are separate, unmeasured claims.

This documents the *iterative-relaxation* regime of dyadic tropical PC — distinct
from the single-pass cascade (see `TROPICAL_PC_DEPTH.md`, Path 4). The cascade
matches backprop and is the right tool when you have a controller. **Iterative PC
is a candidate for substrates without that controller** — distributed / async /
neuromorphic / analog — because it replaces an ordered reverse phase with local
state dynamics. This document does not establish a fabricated implementation.

## Why iterative ≠ cascade (the computational model is the point)

| property | backprop | cascade | **iterative (nudged) PC** |
|---|---|---|---|
| separately learned feedback weights | no; transpose action reuses forward parameters | none | **none** |
| separate ordered **backward phase** | yes | yes (top-down sweep, directional) | **no** |
| local *in time* / async-able | no | no — `e_l` needs `e_{l+1}` | **yes** |
| candidate mapping to physical/analog **relaxation** | not evaluated here | no | **yes, structurally; unbuilt** |

The cascade is *PC reduced to backprop's two-phase structure*: it reuses the
forward parameters but retains a **global, ordered backward sweep**.
Iterative PC has **no backward pass at all** — feedback emerges from a local
energy relaxation where every neuron updates from its own prediction error and
its neighbours', in any order, asynchronously. Scoring it on accuracy-vs-backprop
is only one axis; its proposed value is a **deployment opportunity on hardware
whose dynamics implement relaxation**. That opportunity still requires a
recipe-specific closure audit and hardware measurement.

## Intended damped soft-*nudge* regime (not a hard clamp)

A hard one-hot output clamp makes the settling stiff *and* pushes the equilibrium
away from backprop. The intended regime is the **small-nudge / EqProp limit**:
leave the output state free, add a gentle `β·(state − target)` pull. That keeps
the equilibrium near the feed-forward basin; convergence remains
configuration-dependent. `settle_nudge` / `train_step_nudge` implement the
gamma-damped update.

The archived effective-unit-step runs reached ~0.76 at depth 16 and ~85% of
backprop at depth 32 (0.59 vs 0.69; more sweeps barely helped). Those measurements
do **not** establish the corrected-gamma accuracy, sweep requirement, or an
inherent accuracy cost for fully local/async settling.

## Archived distributed robustness (unit-step, multi-seed)

Depth 16, 3 seeds, 20 local rounds, legacy nudged settling (`harden` mode). In
that effective-unit-step run, the two
*hardest* distributed conditions — lagged/stale reads and no global schedule —
cost almost nothing:

| condition | accuracy (mean ± std) | vs sync |
|---|---|---|
| sync | 0.761 ± 0.014 | — |
| **stale reads (Jacobi)** | 0.750 ± 0.015 | **99%** |
| **random update order** | 0.746 ± 0.021 | **98%** |
| partial participation 50% | 0.724 ± 0.030 | 95% |
| partial participation 25% | 0.604 ± 0.061 | 79% |
| **HARD: 50% + stale + rand** | 0.673 ± 0.032 | **88%** |

The legacy run therefore showed training under communication lag, no global
clock, and half the nodes silent per round. Corrected-gamma robustness must be
remeasured. **The cascade has no async form** (its backward sweep is strictly
ordered), so it is N/A for every row here.

## Archived accuracy ↔ async frontier (unit-step)

The legacy run varied a fraction `p` of cascade steps (sync, exact gradient —
needs a controller) against fully async unit-step nudge updates
(partial+stale+rand). Depth 16 (`hybrid` mode):

| sync fraction `p` | accuracy |
|---:|---:|
| 0.00 (pure async) | 0.558 |
| 0.10 | 0.629 |
| 0.25 | 0.646 |
| 0.50 | 0.711 |
| 1.00 (cascade) | 0.774 |

In that run, 10% sync recovered ~1/3 of the measured gap and 50% recovered ~70%.
Whether the corrected gamma-damped method has the same frontier is unmeasured.

## Physical / analog mapping (the efficiency endpoint)

Iterative PC is an **energy-minimisation** process — the latents relax to
minimise `E = Σ ½‖x_l − f_l(x_{l-1})‖²`. That form suggests an analog /
in-memory implementation whose circuit dynamics approximate the software
relaxation. The correspondences below are design hypotheses, not measurements:

- **States → node voltages/charges** on the substrate (capacitors, crossbar nodes).
- **Energy → a candidate circuit energy**; prediction errors `ε_l` could be
  represented as local voltage differences.
- **Settling → physical relaxation** (RC / Kirchhoff-law dynamics), if a circuit
  realizes the same objective and update rule within acceptable nonidealities.
- **Dyadic ±2^k weights → quantised conductances.** `W·x` is Ohm/Kirchhoff current
  summation in a prospective crossbar. Reciprocal use of that crossbar could
  implement the transpose action without separately learned feedback weights,
  subject to peripheral/data-movement costs.
- **ReLU / max (tropical) → prospective analog rectifiers / comparators** (diodes,
  max cells).
- **Local weight update → in-memory conductance update** from local pre/post
  voltages (`ΔW ∝ ε·activity`), no global error routed.
- **Async operation → matches the clockless, variation-prone reality of analog
  hardware.** The legacy unit-step results suggest tolerance to stale reads,
  random order, and partial participation, but the corrected gamma-damped method
  still needs the same multi-seed validation.

This sits in the lineage of **Equilibrium Propagation** (Scellier & Bengio) —
training energy-based models with local rules and bidirectional weights, designed
for analog substrates — and **memristor-crossbar in-memory computing**. The
contribution here is a tropical / dyadic version whose forward/feedback weights
and local weight-gradient operands are ±2^k. The current software state-settling
loop still has an explicit arithmetic boundary in its diagonal division by `D`;
an end-to-end multiplier-free physical realization and corrected-gamma async
robustness are not yet measured. **The cascade can't ride the physical-relaxation
mapping**: physics settles to an equilibrium; it does not execute an ordered
backward sweep.

## Verdict

Two tools, two proposed hardware targets:
- **Cascade** — matches backprop and uses dyadic network-internal credit operands,
  but needs a controller to run the ordered backward (FPGA / GPU). Loss, optimizer,
  and other recipe operations remain outside that internal-path claim.
- **Iterative (nudged) PC** — structurally has **no ordered backward phase** and
  uses the same weights for forward and feedback. It is the candidate for
  neuromorphic / in-memory / distributed substrates; its corrected-gamma accuracy,
  async robustness, and end-to-end physical closure remain validation targets.

## Rerun with the corrected method

These commands exercise the current gamma-damped implementation (the nudge
paths explicitly use `gamma=0.125`); they are not expected to reproduce the archived
numeric tables above exactly. Record fresh outputs separately from the legacy
tables.

```
TT_REPO="$(git rev-parse --show-toplevel)"
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_deep.py nudge 16 32 48 64   # corrected-gamma accuracy vs backprop
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_deep.py harden             # corrected-gamma multi-seed async rerun
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_deep.py hybrid 16          # corrected-gamma frontier rerun
PYTHONPATH=$TT_REPO python -m pytest tests/test_predictive_coding.py -q -k "nudge or async"  # regression/smoke, not benchmark reproduction
```
