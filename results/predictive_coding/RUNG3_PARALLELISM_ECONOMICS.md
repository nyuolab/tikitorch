# Rung 3: the measured parallelism economics — when local PC wins on a fabric

**Date:** 2026-06-20. The capstone of the scaling-thesis arc (Rung 1 `CHARLM_LOCAL_PC.md`,
Rung 2 `CHARLM_CONV_LOCAL_PC.md`, Rung 3A `conv_parallelism.json`, and the audited
`PARALLELISM.md`). The goal: account *honestly* for the trade local PC makes on a fabric,
grounded only in measured quantities.

## The trade

Local PC swaps backprop's **global** operations for **local** ones:

| | backprop pays | local PC pays |
|---|---|---|
| backward credit | a global, ordered **backward sweep** | **O(L/p) local neighbour rounds** (measured) |
| weight transport | keep `Wᵀ` in sync with `W` | **none** — the same forward `±2^k` weight carries the error (code-verified) |
| gradient comms | a global **all-reduce** | **none** — credit is a local message-passing fixpoint |
| synchronization | a global **barrier** per step | **none** — converges under any order / lag / partial participation (measured) |

The economics question is just: **is the avoided global cost worth the O(L/p) local rounds,
and where?**

## What is measured (the load-bearing facts)

1. **The round-tax is O(L/p), depth-scale, and NOT inflated by the conv receptive field.**
   Rounds for the async error-relaxation gradient to reach the *exact* backprop gradient
   (cosine ≥ 0.999):
   - MLP (`staleness_frontier`): stale = L, partial-50% = 1.5L, worst ≈ 4–6L, ~O(L/p).
   - **Conv (Rung 3A, `conv_parallelism.json`, depth-5 / RF-125):** sync 1, stale **4**,
     partial-50% **4**, worst(50+stale+rand) **12**, partial-25% **24** — all ≈ O(depth/p),
     **far below RF 125.** Credit propagates layer-by-layer (depth hops); the dilation is
     handled *inside* each layer's transpose, not as extra rounds. So a long receptive
     field does **not** inflate the fabric cost (the design's main open risk, retired).
2. **The async credit is *correct*, not approximate.** The async gradient converges to the
   *exact* backprop gradient (cosine → 1.000) under partial participation + stale reads +
   random order — the frozen-routing system is triangular/nilpotent. And the async form
   reaches the **same final result** as sync (Rung 1 MLP within 0.1pp; Rung 2 conv
   **bit-identical** `relax_async` == `relax`).
3. **No weight transport, credit local-in-space** — code-verified: the error route reuses
   the forward `±2^k` weight (no transposed copy), and `E_l = A_lᵀ E_{l+1}` is a
   neighbour-only fixpoint, not a global gradient object.
4. **The accuracy precondition holds** (Rungs 1–2): the local async credit reaches backprop
   quality — MLP *exact* match; competitive **conv within ~0.1 bpc** of float backprop,
   fully multiplier-free. So the economics are **not** bought with an accuracy sacrifice.

## The honest crossover

- **vs *synchronous* backprop** (GPipe-style): PC avoids the backward sweep, the barrier,
  and the all-reduce, paying O(L/p) bounded local rounds. Where the global ops dominate
  (large device count N, high inter-node latency), PC wins. (The `throughput_model.py`
  sensitivity band: a structural ~2× from the missing backward sweep, before any comms.)
- **vs *asynchronous* backprop** (Hogwild! / Downpour / PipeDream-1F1B): staleness-tolerance
  is **shared** — *not* PC-unique. Against this fair baseline, the throughput advantage
  largely evaporates; the band straddles 1× (overlap-dependent). **This is conceded.**
- **What survives, and is the actual prize:** PC eliminates the **backward graph itself** —
  no weight transport, no separate error datapath, credit is a local fixpoint. That makes
  it runnable on **analog / in-memory / settling hardware where backprop has no physical
  realization at all** (the EqProp / memristor lineage). There, the O(L/p) "rounds" are not
  clocked digital steps — they are **continuous-time physical settling** (gate delays, ns),
  and the round-tax being **depth-scale and bounded** (measured, Rung 3A) is exactly what
  makes that settling converge quickly. This is a **substrate** property, not a GPU
  throughput win.

## The economics, stated plainly

`net fabric advantage ≈ (global ops avoided) / (round-tax O(L/p))`. The denominator is now
**measured and bounded** (depth-scale, RF-independent), and the credit it produces is
**exact** and reaches **backprop quality**. So the thesis no longer rests on any
*algorithmic* unknown — the local async multiplier-free learner demonstrably works at
backprop quality on MLP and competitive conv, async-robust, weight-transport-free, with a
bounded round-tax. **It rests only on the hardware substrate existing**: an analog /
in-memory / clockless-async fabric where backprop's global backward graph is the blocker
and continuous-time local settling is cheap. That is the in-memory-compute / neuromorphic
bet — a hardware question, not a learning-algorithm question.

## Honest scope (what is still NOT shown)

- All of this is **structural + simulation-measured** (round-tax via gradient cosine;
  async-robustness emulated in one process; weight-transport-freeness from the code). It is
  **not** wall-clock throughput/energy on real async or analog hardware. That measurement —
  on a multi-node-async or in-memory-compute substrate — is the remaining claim to
  substantiate, and is the natural Rung 4.
- The "avoided global ops" side is argued from dependency structure, not measured on a
  real distributed system; the fair comparator (async-backprop) shares the staleness axis,
  so the GPU-throughput framing is explicitly **dropped** in favour of the substrate one.

## Bottom line of the arc

The scaling thesis — "local PC unlocks parallelism to outscale on fabrics" — is, after
three rungs, **honestly bounded**: the *carrier* is real and measured (local async
multiplier-free credit = backprop quality, async-robust, weight-transport-free,
round-tax O(L/p) and RF-independent), so the only thing between here and the vision is a
**substrate that rewards it** — which is a hardware bet (analog/in-memory/clockless), not
an algorithmic gap. The GPU-throughput version of the claim does not survive the
async-backprop comparison and is dropped; the substrate-enablement version stands.
