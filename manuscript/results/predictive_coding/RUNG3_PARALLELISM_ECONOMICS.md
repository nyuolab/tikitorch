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
| feedback parameters | transpose action reuses forward parameters | **no separately learned feedback matrix**; a reverse transpose action still remains |
| credit communication | implementation-dependent gradient collectives | adjacent-layer messages; no all-reduce in the measured sharded carrier |
| scheduling | ordered dependencies remain | partial/stale/arbitrary-order schedules converge in the measured error-relax simulations |

The economics question is just: **is replacing globally scheduled work with O(L/p)
local rounds worthwhile,
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
3. **No separately learned feedback matrix; credit local-in-space** — code-verified:
   the error route reuses the forward `±2^k` parameter tensor, and
   `E_l = A_lᵀ E_{l+1}` is a neighbour-only fixpoint. The transpose action and
   directional propagation through depth still remain.
4. **The accuracy precondition holds** (Rungs 1–2): the local async credit reaches backprop
   quality — MLP *exact* match; competitive **conv within ~0.1 bpc** of float backprop,
   with dyadic internal credit operands. Loss, optimizer, and data-path boundaries are
   outside that operand-level claim.

## The honest crossover

- **vs *synchronous* backprop** (GPipe-style): error-relaxation replaces the ordered
  backward schedule and its collectives with O(L/p) local rounds. Where global ops dominate
  (large device count N, high inter-node latency), PC wins. (The
  `scripts/pc_throughput_model.py`
  sensitivity band: a structural ~2× from the missing backward sweep, before any comms.)
- **vs *asynchronous* backprop** (Hogwild! / Downpour / PipeDream-1F1B): staleness-tolerance
  is **shared** — *not* PC-unique. Against this fair baseline, the throughput advantage
  largely evaporates; the band straddles 1× (overlap-dependent). **This is conceded.**
- **What survives:** error-relaxation expresses the same directional backward relation
  as neighbour-local messages, reuses the forward parameter tensor, and tolerates
  partial/stale/arbitrary-order scheduling. Those properties motivate an **analog /
  in-memory / settling-hardware implementation**, where local dynamics might make the
  measured O(L/p) propagation cost cheap. The repository has not built or timed that
  mapping, and state settling — the path that could remove the ordered reverse phase —
  still lacks corrected-gamma scale evidence and end-to-end closure. This is a
  **substrate hypothesis**, not a GPU throughput win.

## The economics, stated plainly

`net fabric advantage ≈ (global scheduling/communication replaced) / (round-tax O(L/p))`. The denominator is now
**measured** (depth-scale and RF-independent over the tested configurations), and the credit it produces is
**exact** and reaches **backprop quality**. So the thesis no longer rests on any
unknown about the frozen-routing credit signal in these tests: the dyadic internal path
works at backprop-comparable quality on an MLP and conv and tolerates simulated async
schedules. It still rests on untested algorithm/hardware interfaces, including the
corrected state-settling behavior, full-recipe closure, physical reverse datapath, and
real substrate economics. That is the in-memory-compute / neuromorphic bet.

## Honest scope (what is still NOT shown)

- All of this is **structural + simulation-measured** (round-tax via gradient cosine;
  async-robustness emulated in one process; parameter-tensor reuse from the code). It is
  **not** wall-clock throughput/energy on real async or analog hardware. That measurement —
  on a multi-node-async or in-memory-compute substrate — is the remaining claim to
  substantiate, and is the natural Rung 4.
- The "avoided global ops" side is argued from dependency structure, not measured on a
  real distributed system; the fair comparator (async-backprop) shares the staleness axis,
  so the GPU-throughput framing is explicitly **dropped** in favour of the substrate one.

## Bottom line of the arc

The scaling thesis — "local PC unlocks parallelism to outscale on fabrics" — is, after
three rungs, **honestly bounded**: the frozen-routing carrier is measured (dyadic
internal credit reaches backprop-comparable quality, tolerates simulated async schedules,
and has O(L/p), RF-independent round-tax in the tested range). The path from that carrier
to the vision still includes full-recipe closure, corrected state-settling validation,
and a substrate that rewards locality. The GPU-throughput version of the claim does not
survive the async-backprop comparison; the substrate version remains a research target.
