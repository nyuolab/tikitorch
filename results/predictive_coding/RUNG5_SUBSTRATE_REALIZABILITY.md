# Rung 5: is multiplier-free PC training silicon-realizable? — the measured substrate evidence

**Date:** 2026-06-21. The standing claim after Rungs 1–4: the digital-throughput win is
refuted (`RUNG4_FABRIC_SIM.md`); what's left is the **substrate** axis — a credit-assignment
that is multiplier-free, weight-transport-free, and local, hence expressible on hardware
backprop can't use. Rung 5 asks the first, most concrete version of that: **does the tropical
PC training backward actually synthesize to zero-multiplier hardware?**

The honest answer is **yes, and it's already measured** — because the two operations the PC
backward needs are *already* the FPGA kernels the FPGA workstream has csynth-validated at
DSP=0, and a sibling mult-free training loop already runs on F2 silicon. The contribution
here is the **structural mapping** (which had not been drawn) plus an honest scope.

## The structural mapping: the PC backward IS the already-DSP=0 kernels

`cascade_grads` / `error_relax_grads` (`predictive_coding.py:628–744`) do exactly three
arithmetic things per layer-step, and each maps to a kernel already proven multiplier-free:

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
adders/accumulators are allowed and wanted; `multiplier-free-not-dsp0.md`.)

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
- **csynth ≠ a deployed PC kernel.** No fused PC-adjoint kernel has been P&R'd / AFI-built /
  run on F2. The MuZero-train precedent shows csynth-close can fail P&R at scale. The one
  remaining engineering confirmation is a **fused `forge_pc_wgrad` csynth** (masked-transpose
  error-route + skip-highway fused with the dyadic-MAC PE) → expected DSP_MULT=0 by
  composition, but not yet run. (Scoped, collision-free, ~$0.17/20min, for the FPGA
  workstream to run.)
- **No kernel-isolated energy number exists.** The measured F2 Vccint (morph-LM/MuZero, ~4–9W)
  is **shell-dominated** — the DSP=0 core's dynamic draw is *below the board sensor floor*
  ("statistically identical to idle"; `MUZERO_CHESS_FWD_AFI.md`). So the FPGA energy is a
  whole-rail *bound* (~0.074 J/inference for MuZero), **not comparable** to a GPU
  whole-board number. An energy *comparison* is therefore **not made** here — it would be
  dishonest given the measurement boundaries differ. The GPU dense-backprop baseline
  (`results/lm_gpu_baseline/`, ~11–21k tok/s ramping, val bpb ~2.7) is reported only as
  context.
- **Async/distributed PC stays CPU-side.** The participation/staleness/random-order machinery
  is control-flow, not on the fabric; the kernel covers the sync cascade arithmetic.

## Bottom line of Rung 5

The substrate **precondition is real and measured**: the tropical-PC training backward —
the local, weight-transport-free, multiplier-free credit assignment validated to match
backprop in Rungs 1–2 — synthesizes to **zero-multiplier hardware**, because its two ops are
the already-DSP=0 FPGA kernels and its routing is muxes; and an equivalent mult-free training
loop already runs on F2 silicon. What remains to convert "realizable" into "demonstrated
advantage" is the genuine frontier and is **not** a digital-fabric result: (1) a fused
PC-kernel deployed + P&R'd, (2) a **kernel-isolated** energy measurement (the field's open
problem — current board sensors can't see the DSP=0 core), and (3) the analog/in-memory
endgame where the O(L/p) relaxation rounds become continuous-time settling. Rungs 1–5 have
established the *carrier* end-to-end (matches backprop, async-robust, bounded round-tax,
silicon-expressible at zero DSP); the *win* still lives one substrate beyond what any digital
measurement — FPGA included — can show.
