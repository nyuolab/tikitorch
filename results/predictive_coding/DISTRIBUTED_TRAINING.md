# Distributed (model-parallel) PC training across multiple F2 FPGAs — the "outscale backprop" flagship

**Status (2026-06-23):** CPU-validated + csynth-ready; the F2 csynth → AFI build → f2.12xlarge run are
user-run (gated by a free csynth pass). This doc is the design + de-risk; silicon numbers land after the build.

## The claim

Train ONE network MODEL-PARALLEL across N physical FPGAs: each chip runs **one layer**, holds **only that
layer's** weights+momentum in its **own** on-card DDR, and exchanges **only** a neighbour activation
(downstream, forward) and a neighbour error (upstream, backward). **No chip ever holds the global model or the
global gradient — no weight transport, no all-reduce, no barrier.** This is the on-silicon realization of the
project's "outscale backprop" thesis: the credit-assignment scheme that removes backprop's *global* ops
(global ordered backward, Wᵀ transport, gradient all-reduce, barrier) runs across physically separate
accelerators with neighbour-only messaging.

## What is real (measured on CPU; the fabric arithmetic, bit-for-bit)

`scripts/pc_shard_train_ref.py` simulates the 2-chip split with **forge_pc_block.cpp's exact per-layer
arithmetic** (dyadic shift-add matmul, fixed `norm_shift = bit_length−target`, max-shift activation norm,
int16 cross-chip message rescale, SignSGD-momentum). A 2-layer net (chip0 = hidden D→D ReLU, chip1 = output
D→C) trained model-parallel on 16×16 MNIST:

```
init 0.078 → epoch1 0.8912 → 0.9047 → 0.9021 → 0.9044 → 0.9077 → 0.9080 → epoch7 0.9095 ...
```

**0.91 distributed vs 0.7475 single-chip** (the earlier fused on-fabric run, `ONFABRIC_TRAINING.md`). Two
reasons it is higher: (1) the new block kernel builds in the **off-by-one-correct** `norm_shift`
(`bit_length−target`) from the start, so `tgt=8` works with no host compensation and the recipe reaches its
true ~0.90 ceiling; (2) randn init + shuffle. The int16 cross-chip rescale is **provably benign**: a uniform
shift of the neighbour error scales every `gW` term by the same `2^−s`, so `sign(gW)` — all SignSGD uses — is
invariant; training is unaffected by the message rescale (confirmed empirically: 0.91 with the rescale).

## Artifacts

- **Kernel** `tikitorch/hardware/hls/forge_kernels/forge_pc_block.cpp` (top `forge_pc_block`): ONE layer as a
  distributed block. Role (HIDDEN/OUTPUT) + phase (FWD/BWD) + update-enable read from a CFG word in DDR, so the
  AXI-Lite control map is byte-identical to forge_pc_train_step and **one AFI loads on every slot**. FWD =
  q(W)·a_in → (hidden) relu+max-shift / (output) logits+max-shift; BWD = build error (OUTPUT: a_out−tgt;
  HIDDEN: neighbour e_in) → delta=e·mask → gW=q(delta)ᵀq(a_in) → e_down=delta·q(W) maxshifted to int16
  (upstream msg) → SignSGD update. Reuses the validated dyad / norm_shift / burst I/O / SignSGD. `DSP_MULT=0`
  (every op shift/add/compare/mux). Two kicks/step/chip (FWD then BWD; BWD reads the FWD's cached mask/output
  from DDR). Transpose-free: the error route reads the forward q(W) in place.
- **Reference** `scripts/pc_shard_train_ref.py` — the CPU de-risk above + golden-vector dump (`PC_GOLDEN=...`).
- **csim testbench** `forge_pc_block_tb.cpp` — orchestrates one 2-chip step (2 buffers) and asserts the kernel
  matches an int64 reference of the same step **bit-for-bit** (W0′, W1′, and the AOUT/EOUT messages); only
  AOUT (down) and EOUT (up) cross between buffers.
- **Driver** `tikitorch/fpga/silicon/forge_pc_train_shard.c` — attaches L slots (2 ≤ L ≤ 8), forward sweep
  (passing each layer's activation downstream) + backward sweep (passing each layer's error upstream), weights
  persist per chip; per-epoch host eval; **Gate-C audit** counts cross-chip bytes by kind (messages vs WEIGHTS
  vs GRADIENTS — the last two stay 0 by construction). Generalized to L chips for f2.48xlarge.
- **CL** `tikitorch/hardware/hdk/cl_forge_pc_block/` (clone of the proven DDR CL; kernel instance
  `forge_pc_block`, slice `pcblock_ocl_slice`). **csynth TCL** `synth_pc_block{,_csim}.tcl`.
- **Runners** `scripts/run_pc_shard_train_f2.sh` (f2.12xlarge, loads the one AGFI on both slots, builds+runs
  the shard driver) + the generic `run_kernel_csynth.sh` / `run_pc_afi_build.sh` (invoked with `CL=` /
  `KERNEL_TCL=` env overrides for the block — see Build/run below).

## Verification — three gates (the 3rd is the headline)

- **Gate A — bit-exact.** csim (`forge_pc_block_tb.cpp`) and a one-step on-silicon verify: the distributed
  2-chip step equals the int64 reference (`max|diff|=0`). Proves the sharded computation == single-process.
- **Gate B — it learns.** CPU de-risk already shows **0.91** model-parallel; the f2.12xlarge run must show the
  same accuracy climbing on-fabric, cross-checked by an independent re-eval of the read-back weights.
- **Gate C — nothing global crosses.** The driver's audit prints `messages=… (activations+errors),
  WEIGHTS=0, GRADIENTS=0` — the verifiable evidence that only neighbour messages cross. Negative control:
  zeroing the neighbour error stops the hidden layer learning.

## Honest framing

- The cascade **computes backprop's gradient** (Z-IL) — the claim is **not a new gradient**, it is the
  **locality/communication structure**: the *same* gradient computed with strictly neighbour-only messages and
  per-chip-local weight storage+update, which is exactly what removes backprop's global ops. Gate C makes that
  verifiable, not rhetorical.
- **Structural, not a throughput win.** A 2-layer/256-d net split across two FPGAs is *slower* wall-clock than
  one chip (extra PCIe round-trips); the value is that the no-global-coordination training algorithm runs
  across physically separate accelerators. (Throughput parity/loss at this size is expected and stated.)
- One chip = one layer-pair here; ≥3-chip (f2.48xlarge) and cross-box TCP messaging are the next steps.

## Build / run (user)

```
# 1. free csynth gate (no FPGA): confirm DSP_MULT=0, timing, one-SLR fit + csim BIT-EXACT
KTOP=forge_pc_block KTCL=synth_pc_block_csim.tcl bash scripts/run_kernel_csynth.sh launch
# 2. AFI build (~5h) then register with create-fpga-image
CL=cl_forge_pc_block PKG_REL=tikitorch/hardware/hdk/cl_forge_pc_block \
  KERNEL_TCL=synth_pc_block.tcl KERNEL_PROJ=forge_pc_block_proj bash scripts/run_pc_afi_build.sh launch
# 3. distributed training run on f2.12xlarge (AGFI on both slots; Gates B + C)
AGFI=agfi-... EPOCHS=8 bash scripts/run_pc_shard_train_f2.sh launch
```
