# The PC cascade backward is multiplier-free silicon — synthesized, simulated, bit-exact

**Date:** 2026-06-21. This is the first *synthesized* (not by-analogy) realization of the
local predictive-coding credit-assignment backward as a single fused FPGA kernel. It turns
Rung-5's structural claim — "*the PC backward IS the already-DSP=0 kernels*" — into a fact
established by Vitis HLS C-synthesis + C-simulation against the measured carrier.

## What was built

`tikitorch/hardware/hls/forge_kernels/forge_pc_wgrad.cpp` — the **complete** `cascade_grads`
layer-step for a square interior layer (O==I==D=256, microbatch N=16), fusing all four ops
of the PC backward into one kernel, every product carrying a ±2^k operand (a **shift**, never
a multiply) and every routing decision a cached {0,1} gate (a **mux**):

| op | formula | mult-free mechanism |
|----|---------|---------------------|
| routed adjoint | `delta = e · relu_mask · skip_gate` | {0,1}·{0,1} gate → SELECT |
| weight grad | `gW[o][i] = Σ_n q(delta[n][o])·q(acts[n][i])` | both dyadic → constant shift `1<<(e_d+e_a)` |
| bias grad | `gB[o] = Σ_n q(delta[n][o])` | signed sum of ±2^k → adds only |
| error route | `e_down[n][i] = Σ_o delta[n][o]·q(W[o][i]) + e·(1−skip)` | W dyadic → variable shift `delta<<e_w`; skip = unit-gain mux/add |

It composes the two already-silicon-proven DSP=0 PEs (`forge_wgrad_probe` = both-dyadic
weight grad; `forge_fwd_gemm` = real×dyadic route) plus the gating and the residual
skip-highway that the standalone PEs did not fuse.

## The result (all measured this session)

**1. Synthesizes multiplier-free, at timing, on the F2 part.** csynth on
`xcvu47p-fsvh2892-2L-e` @ 4.0 ns target:

- **DSP = 0**, and **zero** `DSP48 / mul / mac_muladd / multiplier` instances anywhere in
  the report (the closure invariant is DSP_MULT=0; here it is DSP=0 outright).
- **Timing closes**: estimated **2.920 ns** vs 4.00 ns target (uncertainty 1.08 ns) → positive WNS, 37% margin.
- **Fits comfortably**: 2 % BRAM / 3 % LUT / 23 % FF / 0 % DSP / 0 % URAM of the device
  (one-SLR: 7 / 9 / 71 / 0 / 0 %). A single PE instance; replication balances across SLRs.

**2. Literally bit-exact to the measured PC carrier.** The kernel's `dyad()` rounds via a
16-entry `ceil(√2·2^k)` threshold LUT (`DYAD_THRESH`) + a comparator — which reproduces
`tikitorch.nn.dyadic_quantize` (round-half-in-log) **bit-for-bit over all 65 535 int16
magnitudes (0 mismatch)**, and is itself **multiplier-free** (exact `round(log2|v|)` the
obvious way needs `|v|²` — a multiply — which we avoid). So:

- `pc_wgrad_kernel_check.py` (CPU mirror): `gW`, `gB`, `e_down` all `max|kernel − carrier| = 0`
  against both the `cascade_grads` algorithm and `dyadic_quantize` directly.
- `forge_pc_wgrad_tb.cpp` (HLS **C-simulation**, runs the *actual kernel C++* against an
  int64 reference): `gW`, `gB`, `e_down` all `max|diff| = 0`, **0 errors**, with realistic
  fixed-point operands the peak `|gW|`/`|e_down|` were 1.63 M / 4.53 M — comfortably under
  2³¹, confirming the int32 buffer contract holds at the operating point.

**3. Adversarially reviewed before any silicon spend.** A 21-agent review workflow (4
independent lenses → adversarial verify each finding) raised ~17 plausible "high" issues —
int32 truncation, II=1/array-partition, signed-shift, exp_t overflow — and the verifiers
**rejected all of them** as inherited contracts the proven PEs already rely on, or as
empirically fine (csynth closed timing; CSIM confirmed in-range). It surfaced exactly **one**
real gap: the kernel originally omitted `bias.grad`. That is now the `gB` output above, and
re-validated bit-exact. (The cheaper bump-on-next-bit 1.5× round was separately shown
**training-equivalent** to the √2 round on MNIST depth-8/16 cascade — mean Δacc +0.0012,
max 0.0039, within seed noise, `pc_wgrad_round_ablation.py` — so the faithful round is used
for *rigor*, not necessity.)

## Why this matters for the thesis

The training-at-scale program (`TRAINING_AT_SCALE.md`) rests on PC removing backprop's
**global** ops (ordered backward sweep, weight transport, all-reduce, barrier). The carrier
was proven to *do* that — async-robust, weight-transport-free, model-parallel under real
sharding (`pc_model_parallel.py`). The missing piece was that the credit-assignment backward
is realizable on the **multiplier-free fabric** the thesis targets. It now is: the full
local PC backward layer-step **computes the measured carrier's gradient bit-for-bit, on a
zero-multiply datapath that closes timing on the F2 silicon**. The error route reuses the
*same* forward dyadic weight (`q(W)`) — **weight transport is structurally absent in the
kernel**, not just the math.

## Honest scope — what this is and is NOT

- It **is** the backward *layer-step* kernel, C-synthesized and C-simulated, DSP=0, bit-exact.
- It is **not yet** AFI-deployed on F2 (next: a `cl_forge_pc_ddr` CL + host driver → measure
  the gradient produced *on real silicon* matches `cascade_grads`).
- It is **not** a full training loop on fabric, and **not** multi-F2 sharded yet.
- The **digital-throughput economics still stand** (Rung 4): this does not make PC faster
  than backprop on a digital fabric. The win remains **structural/substrate** — no global
  coordination — which the multi-F2 sharded demonstration (one model across ≥2 FPGAs,
  neighbour-only messages, no global gradient) is meant to exhibit on real separate devices.

## Reproduce

```
# bit-exact mirror (CPU)
PYTHONPATH=. python scripts/pc_wgrad_kernel_check.py
# rounding ablation (training-equivalence of the cheap round)
PYTHONPATH=. DATA_ROOT=/tmp/_tiki_data python scripts/pc_wgrad_round_ablation.py
# csim + csynth on a build box (no FPGA needed, ~$0.17)
KTOP=forge_pc_wgrad KTCL=synth_pc_wgrad_csim.tcl bash scripts/run_kernel_csynth.sh launch
```
