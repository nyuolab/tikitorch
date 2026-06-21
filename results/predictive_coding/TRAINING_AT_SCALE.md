# Training at scale: a multiplier-free LM, sharded across a fabric, trained by local PC

**Date:** 2026-06-21. A research-direction doc (not a result) locking in the synthesis of
the two workstreams — the multiplier-free LM that runs on FPGA silicon, and the local PC
credit-assignment carrier (Rungs 1–5) — into one honest program: **a model too large for
any backprop cluster, sharded across a massive fabric of multiplier-free chips, trained by
local PC credit with no device ever holding the global model or the global gradient.**

## The idea

Backprop's wall at extreme scale is not FLOPs — it is the **global** operations: a globally
*ordered* backward sweep across every shard, **weight transport** (Wᵀ kept in sync with W),
a global **all-reduce** of the gradient, and a global **barrier**. These are what make
model-parallel training comm/sync-bound at thousands+ of devices.

**Local PC has none of them.** Once states/errors are settled, `ΔW_l ∝ ε_l ⊗ x_l` uses only
layer-l-adjacent quantities; the error route reuses the *same* forward weight (no transpose
copy); credit is a neighbour-only message-passing fixpoint (no all-reduce); and it converges
under any update order / lag / partial participation (no barrier, no global ordering). So a
model can be **sharded across a fabric where each chip holds one layer/block, exchanges only
neighbour messages, and updates locally** — and no chip ever holds the whole model or the
whole gradient. That is the structural mechanism by which this could *outscale* what
backprop can coordinate.

## What is already real (measured, both workstreams)

- **A multiplier-free LM trains end-to-end on F2 silicon** (`forge_train_step_morph_lm`,
  agfi-085dca07, ~53k tok/s, ~4W; AFI-deployed). *Caveat:* it trains by **SignSGD —
  multiplier-free *backprop*** (frozen-routing global backward), **not** PC; and the
  measured linear scaling is **data-parallel replication** (N independent copies), **not**
  one model sharded.
- **The PC carrier is established end-to-end** (Rungs 1–5): local async multiplier-free PC =
  backprop quality (MLP exact; conv within ~0.1 bpc), async-robust (== sync), weight-
  transport-free, round-tax bounded O(L/p) and receptive-field-independent, and
  **silicon-expressible at zero DSP** (the PC backward = the already-DSP=0 `forge_wgrad_probe`
  + `forge_fwd_gemm` kernels + {0,1} routing muxes).

So both halves exist; the combination — **PC local credit, model-parallel, on the
mult-free fabric** — does not yet, and it inherits walls we have already measured.

## The honest walls (each measured, not hand-waved)

1. **Not built.** No PC training has been deployed model-parallel on the fabric LM; the
   silicon LM is SignSGD-backprop. Wiring PC's cascade/error-relaxation onto a sharded
   fabric is the unbuilt core.
2. **Digital throughput does not favour PC (Rung 4, `RUNG4_FABRIC_SIM.md`).** With measured
   constants, local PC is break-even at light async and slower as async deepens — it still
   pays R relaxation rounds + a weight-grad pass, and pipeline-parallel backprop
   (GPipe/PipeDream) already scales model-parallel with overlapped/amortised all-reduce.
   PC's "no global ordering" advantage converts to a *win* only where global ordering is
   the bottleneck **and** the relaxation rounds are cheap — extreme scale **and/or** an
   analog/in-memory substrate.
3. **The attention wall caps the model.** A *competitive* LM needs smooth content-adaptive
   attention, which is tropical-walled; the fully-mult-free LM (morph-SSM / MBR-attention)
   trails softmax by ~0.2–1 bpb. Scaling this via PC scales a sub-competitive LM unless the
   architecture problem is solved (SSM-friendly, attention-dodging).
4. **Replication ≠ sharding.** The measured linear scaling is independent copies (no PC
   needed); the PC-relevant case (one model sharded, credit crossing chips locally) is the
   harder, unbuilt thing.

## The version that could genuinely win

A model **too large for any backprop cluster** (memory + comm walls), **sharded across a
massive fabric of multiplier-free chips**, trained by **local PC credit** (no global
gradient / all-reduce / ordering), on a substrate where **global ops dominate or relaxation
is continuous-time physics**. It requires, in order: (a) a **PC-trainable competitive
architecture** that dodges the attention wall; (b) **PC deployed model-parallel** on the
fabric; (c) a **substrate that rewards locality** (analog/in-memory, or extreme-comm-bound
scale where Rung-4's economics flip).

## The open tests (the path, in dependency order)

1. **Model-parallel PC correctness** *(`pc_model_parallel.py`)* — **DONE, PASSED.** Sharded
   one MLP char-LM across **D real `torch.distributed` (gloo) processes, one layer each**,
   neighbour-only point-to-point messages, **no all-reduce, no global barrier, no process
   ever holding the global model or gradient**, each updating its layer from its own local
   credit. **Measured: it tracks single-process backprop step-for-step and reaches the same
   quality** — D=3: sharded 3.618 vs backprop 3.621; D=5: sharded **3.0179** vs backprop
   **3.0182**. So the carrier survives *real* sharding (not a single-process simulation
   artifact). [This is correctness, not throughput — gloo localhost; Rung 4 already settled
   the digital-throughput economics.]
2. **Amortised relaxation** *(`pc_amortized_relax.py`, the Rung-4 revisit)* — **DONE, FAILED
   (a clean negative that *closes* Rung 4).** Tested whether few error-propagation sweeps
   per step (K<L), warm-started across steps, hold quality. **They do not:** at K<L quality
   collapses (~4.7 vs backprop 2.9 bpc on a depth-6 char-LM), and warm-start is *worse* than
   cold. The reason is fundamental — partial propagation means the **error never reaches the
   bottom layers** (they get no signal, don't train); that is an *incomplete* gradient, not
   a *stale* one (which async-SGD tolerates), and persisting it across steps just injects
   noise from a stale weight/routing config. So **there is no algorithmic shortcut**: PC's
   error-route is the full O(L) sweep per step, its digital per-step cost is real, and
   **Rung 4 stands** — the win is **substrate-only** (on analog/in-memory that O(L) sweep is
   continuous-time settling, which no digital shortcut replicates). [Pipelining the backward
   would require learned synthetic gradients (DNI) — a different, costed mechanism, not PC.]
3. **A PC-trainable competitive sequence arch** (the architecture wall — SSM direction).
4. **The substrate.** *Backward kernel: DONE (synthesized + simulated + bit-exact),
   `PC_BACKWARD_SILICON.md`.* The complete `cascade_grads` layer-step (routed adjoint +
   weight grad + bias grad + error route + residual skip highway) is one fused FPGA kernel,
   **DSP=0 / zero multiply instances, timing closed 2.920 ns on the F2 part**, and **bit-for-bit
   identical to the measured carrier** (CPU mirror + HLS C-sim both `max|diff|=0`; the `dyad`
   reproduces `dyadic_quantize` over all int16). Weight transport is structurally absent (the
   route reuses the same forward `q(W)`). *Remaining:* AFI-deploy on F2 (gradient produced on
   real silicon == `cascade_grads`), then the multi-F2 sharded run (one model across ≥2 FPGAs,
   neighbour-only messages, no global gradient), then a kernel-isolated energy number and the
   analog/in-memory endgame.

## Honest status

This is a **coherent research program with every gate mapped**, not a result. The pieces are
real in isolation and the mechanism (PC removes backprop's global ops) is sound; the *win*
is a substrate bet gated by the architecture wall and unbuilt deployment. Tests 1–2 are
measurable now and decide whether the carrier survives real sharding and whether amortised
relaxation cracks the digital economics — the two things that would move this from "in
theory" to "measured."
