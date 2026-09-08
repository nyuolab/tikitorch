# Rung 4: fabric throughput with MEASURED constants — the honest (deflating) verdict

**Date:** 2026-06-20. `scripts/pc_fabric_sim.py`, `fabric_sim.json`. The goal was to take
the parallelism claim from a hand-set-τ *model* to a **measured-constant** comparison, and
test honestly whether local PC actually out-throughputs backprop on a fabric.

## Every constant is measured

- **Compute** (timed on the real `DyadicPCNet` ops): `t_fwd ≈ 0.46ms`, **`t_bwd ≈ 1.40ms`
  (= 3.06 × t_fwd)**, `t_round ≈ 0.42ms` (one error-relaxation round ≈ a forward, **~3×
  cheaper than a backward**).
- **Global sync** (real `torch.distributed` gloo all-reduce, N localhost processes):
  **1.7ms (N=2) → 2.5ms (N=4) → 9.0ms (N=8)** — the all-reduce backprop pays and PC avoids,
  growing with N.
- **Round-tax** R (rounds for the async local credit to converge): measured ~O(L/p)
  (Rung 3A / staleness-frontier); swept here R ∈ {L, 1.5L, 3L, 6L}.

## A bias I caught and fixed (transparency)

My first formula favoured PC two ways: it mixed a microbatch-pipeline term with the
per-step relaxation, and it **omitted the weight-grad pass PC still must do**, modelling
PC's backward as just `R·t_round`. Corrected: PC pays `forward + R·(error-route round) +
ONE weight-grad pass`, compared against **async-backprop (1F1B)** — the fair baseline
(latency hidden, all-reduce amortised), not synchronous backprop.

## Result — local PC does NOT win digital throughput

Speedup vs async-backprop (>1 = PC faster), δ=0.1ms:

| round-tax R | L=16 | L=32 | L=64 |
|---|---|---|---|
| R = L (light async) | 1.0–1.06 | 1.0–1.03 | 1.0–1.02 |
| **R = 1.5L** (measured robust point) | **0.90–0.94** | **0.89–0.92** | **0.89–0.90** |
| R = 3L | 0.67–0.71 | 0.67–0.69 | 0.67–0.68 |
| R = 6L (heavy async) | 0.45–0.47 | 0.45–0.46 | 0.45 |

**PC is break-even at light async and slower as async deepens** — because it still pays the
weight-grad pass *plus* R relaxation rounds, which together exceed backprop's backward
sweep + (amortised) all-reduce at the measured all-reduce cost.

**Crossover** — the all-reduce cost at which PC breaks even (R=1.5L): **>17ms (L=16),
>34ms (L=32), >67ms (L=64)**. The measured localhost all-reduce (9ms) is *below* these, so
**async-backprop wins on a small fabric.** PC needs a *real-cluster-scale* all-reduce
(large N, slow interconnect, large gradients) to cross over — and even then, real
frameworks **overlap** all-reduce with compute and amortise it over microbatches, making
the crossover harder still than this model (which puts a fraction of it on the critical
path) suggests.

## Verdict (confirms Rung 3B, by measurement)

**On a digital fabric, the throughput version of "local PC outscales backprop" does not
survive measurement.** Local PC ≈ break-even-to-slower vs well-optimised async-backprop;
the avoided all-reduce only helps at extreme scale, where frameworks overlap it anyway.
This is the same conclusion Rung 3B reached structurally — now grounded in measured
compute + measured all-reduce.

**What is NOT refuted — and remains an opportunity:** the **substrate** axis. The
validated error-relax path uses neighbour-local credit messages, no separately learned
feedback matrix, and dyadic internal operands. Those properties may map efficiently to
**analog / in-memory / clockless settling hardware**, where local dynamics could change
the cost of O(L/p) rounds. The mathematical transpose action and directional propagation
still remain, however, and no end-to-end physical implementation or closure audit is
reported here. **A digital fabric sim cannot settle that regime**, so this Rung-4 result
neither confirms nor refutes the substrate opportunity; it kills only the
*digital-throughput* version.

## Honest scope

- Measured constants (compute + all-reduce) but a **dependency-structure model** of the
  schedule, not end-to-end training on a real cluster. The async-backprop comparator is
  modelled (1F1B latency-hiding + amortised all-reduce), not benchmarked end-to-end.
- The all-reduce was measured on **localhost** (lower bound); a real interconnect at scale
  is larger (helps PC) but is also overlapped by real frameworks (helps backprop). The net
  is genuinely uncertain and not settled here.
- The **substrate** regime (analog/in-memory) — the remaining proposed route to a win —
  is **not measured** by any digital experiment. Substantiating it needs real
  neuromorphic / in-memory-compute hardware. That is the genuine open frontier.
