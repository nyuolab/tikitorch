# Training at scale: a dyadic-fabric/local-PC research program

**Date:** 2026-06-21. A research-direction doc (not a result) locking in the synthesis of
the two workstreams — the multiplier-free LM that runs on FPGA silicon, and the local PC
credit-assignment carrier (Rungs 1–5) — into one honest program: **a model too large for
any backprop cluster, sharded across a massive fabric of multiplier-free chips, trained by
local PC credit with no device ever holding the global model or the global gradient.**

## The idea

At extreme scale, globally scheduled reverse computation and gradient collectives can
become communication bottlenecks. The proposed local-PC carrier represents the same
directional credit relation as adjacent-layer messages: once a layer's error arrives,
`ΔW_l ∝ ε_l ⊗ x_l` uses local quantities, and no separately learned feedback matrix is
needed. The frozen-routing method still requires information to propagate through depth
and a mathematical transpose action; its measured advantage is tolerance to partial,
stale, arbitrary-order scheduling, not elimination of the backward relation. A model
could therefore be **sharded across a fabric where each chip holds one layer/block and
exchanges neighbour messages**. Whether that outscales optimized backprop is an open
hardware/economics question.

## What is already real (measured, both workstreams)

- **A multiplier-free LM trains end-to-end on F2 silicon** (`forge_train_step_morph_lm`,
  agfi-085dca07, ~53k tok/s, ~4W; AFI-deployed). *Caveat:* it trains by **SignSGD —
  multiplier-free *backprop*** (frozen-routing global backward), **not** PC; and the
  measured linear scaling is **data-parallel replication** (N independent copies), **not**
  one model sharded.
- **Pieces of the PC carrier are established:** frozen-routing local credit reaches
  backprop-comparable quality (MLP exact; conv within ~0.1 bpc) and tolerates simulated
  async schedules; its round-tax is O(L/p) and receptive-field-independent in the tested
  range. A fixed-geometry interior layer-step is bit-exact and DSP=0 on F2. That kernel
  result does not close the loss, optimizer, inter-layer transport, or a full PC training
  loop.

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

A model beyond a practical backprop cluster's memory/communication envelope, **sharded
across a massive fabric of multiplier-free chips**, trained by **local PC credit**
(neighbour-scheduled credit without a global gradient all-reduce), on a substrate where
global operations dominate or physical relaxation is unusually cheap. It requires, in
order: (a) a **PC-trainable competitive
architecture** that dodges the attention wall; (b) **PC deployed model-parallel** on the
fabric; (c) a **substrate that rewards locality** (analog/in-memory, or extreme-comm-bound
scale where Rung-4's economics flip).

## The open tests (the path, in dependency order)

1. **Model-parallel PC correctness** *(`scripts/pc_model_parallel.py`)* — **DONE, PASSED.** Sharded
   one MLP char-LM across **D real `torch.distributed` (gloo) processes, one layer each**,
   neighbour-only point-to-point messages, **no all-reduce, no global barrier, no process
   ever holding the global model or gradient**, each updating its layer from its own local
   credit. **Measured: it tracks single-process backprop step-for-step and reaches the same
   quality** — D=3: sharded 3.618 vs backprop 3.621; D=5: sharded **3.0179** vs backprop
   **3.0182**. So the carrier survives *real* sharding (not a single-process simulation
   artifact). [This is correctness, not throughput — gloo localhost; Rung 4 already settled
   the digital-throughput economics.] **NOW ALSO ON REAL SILICON** (`MULTI_F2_SCALE_OUT.md`):
   a 2-layer backward sharded across **two physical FPGAs** (slots 0+1 of an f2.12xlarge, both
   running `agfi-063e2dd0d3255dcaa`), each chip one layer, **only a single neighbour error
   message** between them (no global gradient / weight transport / barrier) → gradients
   **bit-exact** to single-process `cascade_grads`. The structural mechanism, demonstrated on
   real separate accelerators.
2. **Amortised relaxation** *(`scripts/pc_amortized_relax.py`, the Rung-4 revisit)* — **DONE, FAILED
   (a clean negative that *closes* Rung 4).** Tested whether few error-propagation sweeps
   per step (K<L), warm-started across steps, hold quality. **They do not:** at K<L quality
   collapses (~4.7 vs backprop 2.9 bpc on a depth-6 char-LM), and warm-start is *worse* than
   cold. The reason is fundamental — partial propagation means the **error never reaches the
   bottom layers** (they get no signal, don't train); that is an *incomplete* gradient, not
   a *stale* one (which async-SGD tolerates), and persisting it across steps just injects
   noise from a stale weight/routing config. So **there is no algorithmic shortcut**: PC's
   error-route is the full O(L) sweep per step, its digital per-step cost is real, and
   **Rung 4 stands for the tested digital model**. Analog/in-memory dynamics remain a
   possible route to changing that O(L) cost, but are unbuilt and unmeasured.
   [Pipelining the backward
   would require learned synthetic gradients (DNI) — a different, costed mechanism, not PC.]
3. **A PC-trainable competitive sequence arch** (the architecture wall — SSM direction).
4. **The substrate.** *Backward kernel: DONE — VERIFIED BIT-EXACT ON F2 SILICON,
   `PC_BACKWARD_SILICON.md`.* The complete `cascade_grads` layer-step (routed adjoint + weight
   grad + bias grad + error route + residual skip highway) is one fused FPGA kernel, **DSP=0 /
   zero multiply instances**, built into a timing-closed AFI (`agfi-063e2dd0d3255dcaa`, WNS
   +0.077 ns), and the gradient it produces **on the chip is bit-for-bit identical to
   `cascade_grads`** (on-card DDR readback: known-answer + random both `max|diff|=0`). No
   separately learned feedback matrix is present (the route reuses the same forward
   `q(W)` values). On-silicon
   verification caught a synthesis-only bug C-sim cannot see (a const-ROM dyad threshold whose
   init didn't survive synth; fixed with the table-free 1.5× dyad). *Remaining:* a full
   PC training loop across the sharded devices, recipe-level closure, a kernel-isolated
   energy number, and the analog/in-memory investigation.

## Honest status

This is a **coherent research program with every gate mapped**, not a result. The pieces are
real in isolation; the proposed *win* from replacing global scheduling/collectives with
neighbour messages is a substrate bet gated by the architecture wall, recipe-level
closure, and unbuilt full-PC deployment. The completed sharding and amortisation tests
establish correctness and a digital-throughput negative; they do not establish the
end-to-end substrate result.
