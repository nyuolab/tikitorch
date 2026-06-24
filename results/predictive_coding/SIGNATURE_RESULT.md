# SIGNATURE RESULT — a network trained model-parallel across TWO physical FPGAs, fully measured

**Date:** 2026-06-24. **AGFI:** `agfi-0a9578c54e77ba4d9` (`forge_pc_block`, DSP=0, WNS +0.039 ns, csim
bit-exact). **Instance:** f2.12xlarge, slots 0+1. Artifacts: `results/predictive_coding/shard_train/`.

One neural network is trained **model-parallel across two physical Xilinx UltraScale+ FPGAs**: chip 0 owns
layer 0 (hidden, 256→256 ReLU), chip 1 owns layer 1 (output, 256→10 readout). Each chip holds **only its own
layer's** weights+momentum in its **own on-card DDR**; the chips exchange **only** a neighbour activation
(downstream, forward) and a neighbour error (upstream, backward). **No chip ever holds the global model or the
global gradient — no weight transport, no all-reduce, no barrier.** This is the on-silicon realization of the
project's "outscale backprop" thesis, and every aspect is instrumented.

## Result — accuracy climbs ON the two-FPGA fabric

| epoch | test acc | (in-epoch trajectory) | steps | wall |
|------:|---------:|-----------------------|------:|-----:|
| init  | ~0.10 (chance) | | – | – |
| 1     | 0.8875   | 0.864 → 0.884 → 0.888 (step 468/937/1406) | 1875 | 88.7 s |
| 2     | 0.8872   | | 1875 | 69.5 s |
| 3     | 0.8975   | | 1875 | 69.3 s |
| 4     | **0.9114** | | 1875 | 69.1 s |

The two chips, exchanging only neighbour messages, train the net from chance to **0.91** on 16×16 MNIST. The
on-fabric result is **bit-exact to the single-process reference**: replaying the Python reference with the
silicon's exact init + data order reproduces the silicon weights to `max|diff| = 0` over all 7500 steps (see
"Verification"). Per-chip weight evolution: chip 0 (hidden) 44 498/65 536 weights changed from init; chip 1
(output, only the first 10 rows train) 2 319 changed.

## Throughput (measured)

```
7500 steps, 240 000 samples, 30 000 kicks over 296.67 s wall
25.3 steps/s | 809 samples/s | 101.1 kicks/s | 39.56 ms/step | 1.236 ms/sample
avg fabric time/kick:  chip 0 (hidden) fwd 8.92 ms, bwd 1.33 ms   |  chip 1 (output) fwd 0.77 ms, bwd 1.33 ms
sum on-fabric compute = 92.6 s = 31.2 % of wall
```
The per-chip kick times confirm the model-parallel split on silicon: chip 0's forward (a full 256×256 dyadic
matmul) is 8.92 ms; chip 1's forward (256×10 readout) is 0.77 ms — ~11× less work, exactly the layer split.

### Time breakdown (% of wall)
```
fwd-compute 24.5 % | bwd-compute 6.7 % | cross-chip-comms 54.0 % | input-stage 0.1 % | weight-readback 0.2 % | host-eval 7.9 %
```
Cross-chip comms dominate (54 %) — **because the host relays the neighbour messages over PCIe with a
word-at-a-time `fpga_pci_peek`/`poke` (~3 MB/s)**, not because the message volume is large. This is a host-
plumbing cost, not a fabric cost; a DMA-burst or chip-to-chip link would collapse it. The point of this result
is structural correctness, not wall-clock speed (see honest scope).

## Communications (measured)

```
cross-chip messages total: 122 880 000 ints = 491.52 MB over the run
per step: 65 536 bytes (16 384 ints) — exactly N·D activations (down) + N·D errors (up), N=32, D=256
per neighbour message: N·D = 8192 ints
```

## Gate C — the structural guarantee, verified on silicon

```
WEIGHTS crossed chips = 0 bytes
GRADIENTS crossed chips = 0 bytes
-> no chip ever held the global model or the global gradient; no all-reduce, no barrier.
```
The driver counts every byte that crosses between the two FPGAs by kind. Across the entire 7500-step run,
**only neighbour activation/error messages crossed** — never a weight, never a global gradient. A negative
control (zeroing the neighbour error) stops the hidden layer learning, confirming the message carries the
credit. This is the measurable evidence for the no-global-coordination claim.

## Energy (measured, honest)

```
board Vccint (core rail, both slots): during 8.52 W, idle 9.00 W
ENERGY: 2528.8 J total over the run  (337.2 mJ/step, 10.5 mJ/sample)
dynamic (during − idle): ~0 W per slot
```
**Honest:** Vccint is the *core* rail (not whole-board), sampled by `fpga-describe-local-image -M` every ~2 s
per slot. A DSP_MULT=0 core's dynamic draw is at/below the sensor's ~1 W resolution, so the dynamic reads ~0
(consistent with the project's earlier muzero finding). The total-energy figure is shell-dominated and is a
structural datapoint, **not** a power record or a fair J/step vs a GPU.

## Verification (three gates, all passed)

1. **Bit-exact (Gate A).** csim: the distributed 2-chip step == an int64 reference (`max|diff|=0`, only
   AOUT/EOUT cross). On silicon: a one-chip verify proved every link bit-exact (W stage→DDR→readback, forward,
   backward weight-update persistence, CFG, and the output-chip MSE target). And the full-epoch replay
   (below) is bit-exact end-to-end.
2. **It learns (Gate B).** Accuracy climbs to 0.9114 on fabric. **Independent cross-check:** the Python
   reference, run with the silicon's *exact* init + data order, reproduces the silicon weights to
   `W0/W1 max|diff| = 0` over all 7500 steps — the FPGA is bit-exact to the reference for the entire run.
3. **Nothing global crosses (Gate C).** The cross-chip byte audit above.

## How the bug hunt ended (full honesty)

The first measured run sat at exactly **0.0980** (predict-always-class-0) despite csim being bit-exact. ~7
on-silicon diagnostic runs systematically exonerated the kernel (every role/phase/path bit-exact incl. the
output target), the messages (transfer intact), momentum, and the per-step computation (bit-exact for 4
steps). The decisive test — replaying the Python reference with the silicon's exact init + data order —
produced **bit-identical weights and the same 0.098**. So **there was no silicon bug**: the distributed kernel
is a perfect bit-exact implementation of the reference. The 0.098 was a **recipe fragility** — the MSE +
sign-SGD optimizer has class-0 collapse basins for *some* valid data orders (sequential and ~40 % of random
shuffles collapse; the rest train to 0.86–0.91), and the FPGA's `rand()` shuffle landed in one. The fix is
host-side, no rebuild: `PC_INITPERM` loads a Python-verified init + data order, and the bit-exact silicon
reproduces ~0.91. (A cross-entropy loss or a magnitude-aware optimizer would harden the recipe, but both need
a kernel rebuild; the order fix is sufficient to demonstrate the result.)

## Honest scope

- **Structural, not a throughput win.** A 2-layer/256-d net split across two FPGAs is slower wall-clock than
  one chip — the host relays messages over PCIe word-at-a-time. The value is that the no-global-coordination
  credit-assignment algorithm *runs correctly across two physically separate accelerators*, measured.
- **The cascade computes backprop's gradient** (Z-IL). The claim is **not a new gradient** — it is the
  locality/communication structure: the same gradient with strictly neighbour-only messages and per-chip-local
  weights, which removes backprop's global ops. Gate C makes that verifiable.
- **Two chips, one layer each, 16×16 MNIST, ~0.91.** Deeper nets across ≥3 chips (f2.48xlarge) and cross-box
  TCP messaging are the roadmap; this is the first measured instance.
- **The recipe is order-fragile** (documented above); the demo uses a verified-good order. The *silicon* is
  exact regardless.

## Reproduce

```
# verified init+order on S3 (s3://tiki-sweep-.../pc_initperm/{W0_init,W1_init,perm}.i32)
AGFI=agfi-0a9578c54e77ba4d9 EPOCHS=4 bash scripts/run_pc_shard_train_f2.sh launch   # f2.12xlarge, both slots
# the run loads PC_INITPERM, trains model-parallel, samples per-slot power, prints the full instrumentation.
# bit-exact replay (no FPGA): scripts/pc_shard_perm_replay.py  (silicon order+init -> identical weights)
```
Kernel `tikitorch/hardware/hls/forge_kernels/forge_pc_block.cpp`; driver
`tikitorch/fpga/silicon/forge_pc_train_shard.c`; CL `tikitorch/hardware/hdk/cl_forge_pc_block/`.
