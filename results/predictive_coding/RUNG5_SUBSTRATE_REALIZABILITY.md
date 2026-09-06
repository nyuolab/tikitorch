# Rung 5: a multiplier-free PC layer-step on silicon — measured substrate evidence

**Date:** 2026-06-21. The standing claim after Rungs 1–4: the digital-throughput win is
refuted (`RUNG4_FABRIC_SIM.md`); what's left is the proposed **substrate** axis — a
neighbour-local credit path with dyadic internal operands and no separately learned
feedback matrix. Rung 5 asks the first, most concrete version of that: **does one tropical
PC backward layer-step synthesize to zero-multiplier hardware?**

> **Later evidence / scope correction:** `PC_BACKWARD_SILICON.md` supersedes the
> engineering-status statements below: a fixed-geometry interior cascade layer-step is
> now P&R'd and bit-exact on F2, and `MULTI_F2_SCALE_OUT.md` demonstrates a two-device
> sharded backward. These are kernel/backward results, not closure of the loss,
> optimizer, full PC training loop, or an analog/in-memory deployment. “No weight
> transport” below means no separately learned feedback matrix; the transpose action and
> physical reverse datapath still exist.

The honest answer is **yes, and it's already measured** — because the two operations the PC
backward needs are *already* the FPGA kernels the FPGA workstream has csynth-validated at
DSP=0, and a sibling mult-free training loop already runs on F2 silicon. The contribution
here is the **structural mapping** (which had not been drawn) plus an honest scope.

## The structural mapping: the PC backward IS the already-DSP=0 kernels

`_route_T`, `cascade_grads`, and `error_relax_grads`
(`tikitorch/nn/predictive_coding.py`) implement the internal operations summarized below,
which map to kernels already proven multiplier-free:

| PC backward op (CPU) | what it is | FPGA kernel | measured |
|---|---|---|---|
| `weight.grad = q(δ)ᵀ @ q(acts)` | operand-quantized (±2^k × ±2^k) outer product | **`forge_wgrad_probe`** | csynth **2.92ns @ 4ns target, DSP=0**, 49% SLR |
| `e_down = δ @ dyadic_quantize(W)` | real error × ±2^k weight = a barrel shift | **`forge_fwd_gemm`** (transposed read) | csynth **DSP=0**, 2.92ns |
| `δ = e · skip_gate · relu_mask` | cached {0,1} gates → conditional **select** | a mux on the cached gate | no multiply by construction |

Every product in the PC backward has **at least one ±2^k operand → it is a shift, not a
multiply**, and the routing is **gather/scatter through cached {0,1} selectors → muxes**.
There is no real multiplier anywhere. This is the tropical/dyadic enabler made concrete:
hard discrete routing (relu/argmax → {0,1} selectors) + dyadic weights (±2^k → shifts) means
the *backward* is shift-add + select, the same as the forward — so it synthesizes to the
same zero-DSP fabric. (Closure invariant audited as **DSP_MULT = 0**, not DSP=0 — DSP48s as
adders/accumulators are allowed and wanted; see `PC_BACKWARD_SILICON.md`.)

## The measured evidence (already on silicon / in csynth)

- **`forge_wgrad_probe`** (`results/fpga_training_rig/forge_wgrad_probe_csynth.rpt`): the PC
  weight-grad outer product, **DSP=0, 2.920ns** (meets the 4ns target), fits one SLR. The
  synth TCL does **not** force `config_op mul -impl fabric` and still reports DSP=0 — the
  multiply-freeness is **structural**, not a synthesis crutch.
- **`forge_fwd_gemm`** (the PC error-route, transposed): **DSP=0, 2.92ns**.
- **`forge_train_step_morph_lm`** — a *full* multiplier-free training loop (fwd-loss +
  frozen-routing backward + SignSGD) **AFI-built and running on F2 silicon**
  (agfi-085dca07f374aa1cd, DDR-resident, ~53k tok/s, ~4W loaded). This is the existence proof
  that mult-free *training* — not just inference — closes P&R and runs on real fabric.

So: the PC backward's arithmetic is silicon-realizable at zero multipliers, demonstrated at
the op level (csynth) and, for an equivalent mult-free training loop, on deployed silicon.

## What this does NOT claim (the honest scope)

- **Realizability, not superiority.** No efficiency or throughput win is claimed — Rung 4
  already refuted the digital-fabric throughput advantage (PC ≈ 0.9× async-backprop). This
  rung says the mult-free PC backward *can be built in zero-DSP hardware*, which is the
  substrate *precondition*, not a performance result.
- **Kernel closure ≠ training closure.** The later fused `forge_pc_wgrad` result closes
  timing at DSP=0 and matches its fixed-point reference on F2 for one interior-layer
  geometry. It does not include the loss, optimizer, complete model, state-settling
  division, or a full training schedule.
- **No kernel-isolated energy number exists.** The measured F2 Vccint (morph-LM/MuZero, ~4–9W)
  is **shell-dominated** — the DSP=0 core's dynamic draw is *below the board sensor floor*
  ("statistically identical to idle";
  `results/fpga_silicon/MUZERO_CHESS_FWD_AFI.md`). So the FPGA energy is a
  whole-rail *bound* (~0.074 J/inference for MuZero), **not comparable** to a GPU
  whole-board number. An energy *comparison* is therefore **not made** here — it would be
  dishonest given the measurement boundaries differ. The GPU dense-backprop baseline
  (`results/lm_gpu_baseline/`, ~11–21k tok/s ramping, val bpb ~2.7) is reported only as
  context.
- **Async/distributed PC stays CPU-side.** The participation/staleness/random-order machinery
  is control-flow, not on the fabric; the kernel covers the sync cascade arithmetic.

## Bottom line of Rung 5

The measured precondition is narrower: the internal cascade layer-step can synthesize to
a zero-multiplier FPGA kernel, and later work verified one such kernel on F2. What remains
is a full PC training loop with recipe-level closure, a **kernel-isolated** energy
measurement, and the analog/in-memory investigation. The existing digital results do not
establish an end-to-end substrate advantage.
