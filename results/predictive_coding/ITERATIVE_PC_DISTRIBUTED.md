# Iterative tropical PC: the fully-local / async / physically-realizable regime

**Date:** 2026-06-17
**Code:** `tikitorch/nn/predictive_coding.py` (`DyadicPCNet.settle_nudge`,
`train_step_nudge`); experiments `scripts/pc_dyadic_deep.py` (`harden`, `hybrid`,
`async`, `nudge*` modes); tests in `tests/test_predictive_coding.py`.

This documents the *iterative-relaxation* regime of dyadic tropical PC — distinct
from the single-pass cascade (see `TROPICAL_PC_DEPTH.md`, Path 4). The cascade
matches backprop and is the right tool when you have a controller. **Iterative PC
is the right tool for substrates where you don't** — distributed / async /
neuromorphic / analog — because it keeps PC's defining computational property
that the cascade gives up.

## Why iterative ≠ cascade (the computational model is the point)

| property | backprop | cascade | **iterative (nudged) PC** |
|---|---|---|---|
| weight transport (separate ⊤ weights) | needs it | none (same ±2^k weights) | **none** |
| separate ordered **backward phase** | yes | yes (top-down sweep, directional) | **no** |
| local *in time* / async-able | no | no — `e_l` needs `e_{l+1}` | **yes** |
| maps to physical/analog **relaxation** | no | no | **yes** (energy min = circuit settling) |

The cascade is *PC reduced to backprop's two-phase structure*: it keeps the
no-weight-transport win but reintroduces a **global, ordered backward sweep**.
Iterative PC has **no backward pass at all** — feedback emerges from a local
energy relaxation where every neuron updates from its own prediction error and
its neighbours', in any order, asynchronously. Scoring it on accuracy-vs-backprop
is the wrong axis (backprop is the ceiling by construction); its value is
**deployability on hardware the cascade can't run on.**

## Convergence: the soft-*nudge* regime (not a hard clamp)

A hard one-hot output clamp makes the settling stiff *and* pushes the equilibrium
away from backprop. The convergent regime is the **small-nudge / EqProp limit**:
leave the output state free, add a gentle `β·(state − target)` pull. That keeps
the equilibrium near the feed-forward basin, so diagonal-preconditioned
Gauss-Seidel converges there. `settle_nudge` / `train_step_nudge` implement it.

It converges to a **usable** equilibrium (~0.76 at depth 16) but **not** to
backprop accuracy — it plateaus at ~85% of backprop at depth (depth 32: 0.59 vs
0.69; more sweeps barely help). That gap is *inherent* to the equilibrium regime
(the cascade matches backprop precisely because it computes the exact gradient in
one ordered pass). The ~15% is the price of being fully local/async.

## Distributed robustness (hardened, multi-seed)

Depth 16, 3 seeds, 20 local rounds, nudged settling (`harden` mode). The two
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

So iterative PC trains under communication lag, no global clock, and half the
nodes silent per round — the conditions analog/distributed hardware *imposes*.
**The cascade has no async form** (its backward sweep is strictly ordered), so it
is N/A for every row here.

## The accuracy ↔ async frontier

You can trade synchronization for accuracy smoothly. A fraction `p` of steps use
the cascade (sync, exact gradient — needs a controller); the rest are fully async
nudge (partial+stale+rand). Depth 16 (`hybrid` mode):

| sync fraction `p` | accuracy |
|---:|---:|
| 0.00 (pure async) | 0.558 |
| 0.10 | 0.629 |
| 0.25 | 0.646 |
| 0.50 | 0.711 |
| 1.00 (cascade) | 0.774 |

Even occasional synchronization (which most distributed systems can afford
periodically) buys back accuracy: 10% sync recovers ~1/3 of the gap, 50% recovers
~70%. So one can operate anywhere on the frontier per the substrate's
capabilities.

## Physical / analog mapping (the efficiency endpoint)

Iterative PC is an **energy-minimisation** process — the latents relax to
minimise `E = Σ ½‖x_l − f_l(x_{l-1})‖²`. Energy minimisation is what physical
systems do *for free*, so this regime maps onto analog / in-memory hardware that
optimises by *settling*, not by digital gradient computation:

- **States → node voltages/charges** on the substrate (capacitors, crossbar nodes).
- **Energy → the network's physical energy**; prediction errors `ε_l` are local
  voltage differences.
- **Settling → physical relaxation** (RC / Kirchhoff-law dynamics). No digital
  gradient is computed — the hardware settles to the minimum.
- **Dyadic ±2^k weights → quantised conductances.** `W·x` is Ohm/Kirchhoff current
  summation — analog matrix–vector *for free*, multiplier-free (it's physics).
  The transpose `Wᵀ` (error feedback) uses the **same** crossbar — **no weight
  transport** (the same synapses carry forward and feedback). This is the core
  PC/EqProp hardware advantage.
- **ReLU / max (tropical) → analog rectifiers / comparators** (diodes, max cells).
- **Local weight update → in-memory conductance update** from local pre/post
  voltages (`ΔW ∝ ε·activity`), no global error routed.
- **Async-robustness → matches the clockless, variation-prone reality of analog
  hardware.** The measured robustness to stale reads (99%), random order (98%),
  and partial participation is exactly the tolerance such hardware requires — our
  numbers *validate* that the algorithm survives those conditions.

This sits in the lineage of **Equilibrium Propagation** (Scellier & Bengio) —
training energy-based models with local rules and bidirectional weights, designed
for analog substrates — and **memristor-crossbar in-memory computing**. The
contribution here is a **tropical / dyadic (multiplier-free, ±2^k)** version that
maps onto *quantised-conductance* crossbars, with measured async robustness. **The
cascade can't ride this**: physics settles to an equilibrium; it does not execute
an ordered backward sweep.

## Verdict

Two tools, two hardware targets:
- **Cascade** — matches backprop, multiplier-free, but needs a controller to run
  the ordered backward (FPGA / GPU). The fast digital learner.
- **Iterative (nudged) PC** — ~85% of backprop, but **no backward phase, no weight
  transport, async-robust, physically realizable.** The enabler for
  neuromorphic / in-memory / distributed substrates — the ultra-efficient hardware
  this whole multiplier-free direction is ultimately aiming at, where backprop
  *cannot run at all*.

## Reproduce

```
TT_REPO="$(git rev-parse --show-toplevel)"
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_deep.py nudge 16 32 48 64   # soft-nudge convergence vs backprop
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_deep.py harden             # multi-seed async robustness + depth scan
PYTHONPATH=$TT_REPO python scripts/pc_dyadic_deep.py hybrid 16          # accuracy<->async frontier
PYTHONPATH=$TT_REPO python -m pytest tests/test_predictive_coding.py -q -k "nudge or async"
```
