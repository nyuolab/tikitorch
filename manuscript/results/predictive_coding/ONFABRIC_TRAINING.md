# A network trained entirely on multiplier-free FPGA silicon (local-PC cascade)

**Date:** 2026-06-23. **AGFI:** `agfi-0fdc38db9da6add74` (`forge_pc_train_step`). **Instance:**
f2.6xlarge, slot 0.

The AGFI identifies the historical account/region-specific experiment; it is
not bundled with TikiTorch or guaranteed to remain public or accessible.

A 2-layer dyadic MLP (256 → 256 → 10) is trained on 16×16-downsampled MNIST with **every step of
credit assignment running on the FPGA fabric** — forward, top-error, cascade backward, gradient,
momentum, and the weight update. The host only streams each minibatch (pixels + a one-hot label
target) into on-card DDR and kicks the kernel; **weights and momentum persist in DDR across kicks
and are never updated by the host.** Periodically the host reads the weights out to *measure*
accuracy (observation, not part of the training loop).

This was independently honesty-audited (5 adversarial reviewers reading the kernel, driver, recipe
and logs); their required corrections are folded into the framing and caveats below.

## Result (measured on silicon)

| epoch | test acc (on-fabric driver) | kicks | wall |
|------:|----------------------------:|------:|-----:|
| (chance baseline) | ~0.10 (1/10; analytic — the fixed driver's first measurement is epoch 1) | – | – |
| 1     | 0.7101                      | 1875  | 24.8 s |
| 2     | 0.7430                      | 1875  | 24.6 s |
| 3     | 0.7399                      | 1875  | 24.6 s |
| 4     | 0.7479                      | 1875  | 24.5 s |
| 5     | **0.7475**                  | 1875  | 24.8 s |

- Batch 32 → 1875 kicks/epoch over 60 000 train images. The ~24.6 s/epoch wall **also includes** a
  host weight read-back + a full 10 000-image host eval, so it is not a training-rate figure;
  the archive does not isolate per-kick fabric time. This is a correctness/structure demonstration,
  not a throughput claim.
- After a single kick, 37 729 / 65 536 W0 weights had changed from init → the on-fabric update is
  live and persists in DDR (`[diag]` line in `train.txt`).
- Predictions are spread across most classes (not collapsed to one), though **imbalanced** — the
  epoch-5 histogram `[984 1157 2555 873 1070 284 1001 968 5 1103]` has one class at ~26% and one
  predicted 5/10 000, consistent with the modest accuracy.

### Independent cross-check (splits "weights are good" from "predict() is good")

The driver dumped the final integer weights; a **separate** code path (`pc_eval_onfabric_weights.py`,
the Python recipe forward) re-evaluated them on a **freshly downloaded** torchvision MNIST test set
(committed: `onfabric_train/crosscheck.txt`):

```
RECIPE-EVAL of ON-FABRIC weights: test acc = 0.7413   (driver reported 0.7475)
```

The two differ by 0.62 percentage points. That difference is **unresolved
pipeline variation**: the archive did not freeze identical evaluator inputs,
preprocessing output, and software versions for the two paths. Platform/version
differences in `F.interpolate(area)` are a plausible source, but the current
records do not isolate the cause and therefore cannot distinguish preprocessing
from other evaluator-implementation variation. The separate evaluation does
establish that the exported fabric-trained weights retain substantial accuracy on a freshly
loaded test set; it is not a bit-for-bit parity test.

## What runs on the fabric, exactly (kernel `forge_pc_train_step.cpp`)

One kick = one minibatch training step, fully on-chip:

1. **Forward** — `pre = Σ q(W)·x` via dyadic shift-add (exponent add + barrel shift), ReLU, then a
   **max-shift activation norm** (right-shift so |max| lands in a fixed band) to keep magnitudes bounded.
2. **Top error (MSE)** — `e = (logits ≫ sh) − tgt`. A subtraction; **no exp / no softmax** on fabric.
3. **Cascade backward** — `delta = e · relu_mask`; error route `e_down = Σ e · q(W1)` reading the
   **forward** weight's dyadic exponents in place (line 202 reuses `w1E` produced once at line 161) →
   **no materialized transposed weight buffer** (the transpose is an access pattern, not a stored/
   synced copy); `delta0 = e_down · mask0` with its own max-shift.
4. **Gradients** — `gW = q(delta)ᵀ · q(acts)` again via exponent-add + shift.
5. **Update** — integer-momentum **SignSGD**: a leaky sign-counter `mom += sign(gW)` (capped),
   then `W −= sign(mom)`. All integer add / compare.

**`DSP_MULT=0`:** every matmul (forward, error route, gW0, gW1) is exponent-addition + barrel-shift +
sign-mux add — there is no data multiply anywhere in the kernel; SignSGD is compare/int-add. Synthesis
binds **one** DSP as an *adder*, zero as multipliers (the project invariant is no-multiply, not no-DSP —
see [[multiplier-free-not-dsp0]]). The WNS +0.196 ns / DSP figures are from the **AFI build report**
that produced `agfi-0fdc38db9da6add74` (this run's box only compiles the host driver, not HLS); see
`PC_BACKWARD_SILICON.md`.

### Honest note on "transport"

The backward **reuses the forward weights** (transposed access of `W1`). That is *not* the
biologically-motivated "weight-transport-free" property (which would require separate or random
feedback weights, à la feedback alignment) — this run does **not** claim that. What is true and what
matters for hardware: there is **no separately-stored, separately-synchronized transposed weight
buffer** — forward and backward read the *same* DDR-resident weights. In the multi-chip shard
([[pc-backward-silicon]], `MULTI_F2_SCALE_OUT.md`) this becomes the meaningful property: no weight is
copied between chips for the backward; only a neighbour error message crosses.

## The bug that almost buried it (and the honest fix)

The first run sat at **exactly 0.098** (predict always class 0) even though the
weights demonstrably evolved. Root cause: the kernel's
`norm_shift` computes `s = highest_bit_pos − target` (= `bit_length − 1 − target`), **one shift less**
than the validated recipe's `bit_length − target`. argmax is shift-invariant, so *evaluation* was
unaffected — but in *training* the kernel's activations/logits come out 2× larger, which halves the
effective logit-vs-target gap and collapses the dyad-rounded gradient's class asymmetry → no
learning. A C reference carrying the same off-by-one convention would not, by
itself, detect this recipe mismatch. Importantly, the archived training run did
**not** execute its optional full-train-kernel one-step verifier
(`onfabric_train/pctrain.log` records `VERIFY=`); no such KAT result is claimed
for this run.

Reproduced exactly in the recipe (`PC_KERNEL_SHIFT=1` → stuck at 0.098, identical weight growth;
committed: `onfabric_train/recipe_diag.txt`). **Fixed host-side, no AFI rebuild:** double the
true-class target 8 → 16, which precisely compensates the 2× scale. The target is a loss *data input*
the host already supplies, so this moves **no computation off the fabric** — the training-loop claim
is intact. Recipe confirmation: off-by-one + tgt=16 → 0.896; among the tested
target values, 16 performed best (24 → 0.864, 32 → 0.789). *(A future clean
rebuild could set `norm_shift = bit_length − target` so tgt=8 works. That rebuild
is unnecessary for this target-compensated demonstration, but it is required
before claiming recipe-parity at the original target scale.)*

**Why 0.7475 and not 0.896 — non-parity context.** The 0.896 recipe run used
**8 epochs / batch 128 / `randn*4` initialization**. The fabric run used **5
epochs / batch 32 (TN=32) / uniform-int(−4..4) initialization**. A closer recipe
configuration reached ~0.81 (`recipe_diag.txt`: 0.7996 / 0.8079 / 0.8094 over
three epochs), but it still used a different RNG path and did not freeze the
same evaluator preprocessing. These records show that the runs are not a
controlled parity comparison; they do not attribute the remaining accuracy
difference. The independent evaluation confirms useful exported weights, but
the archive does not contain full-train-kernel KAT or full-trajectory parity
evidence and therefore does not exclude every residual kernel issue.

## Honest scope — what this is and is not

- **Is:** a complete credit-assignment loop (fwd + local-cascade bwd + update) closing entirely on
  one multiplier-free FPGA, training a real net from chance to ~0.75 on downsampled MNIST. The
  exported weights were independently re-evaluated at 0.7413; the 0.62-point evaluator difference
  remains unresolved.
- **Is not** a SOTA-accuracy claim: 16×16 MNIST, ~0.75, 2 layers. The point is that the *loop closes
  on fabric*, not the number.
- **Is not** a throughput claim (see the table note).
- **Cascade computes backprop's gradient** (Z-IL); its property *demonstrated here* is **multiplier-
  free** with the backward reusing the forward weights in place (no transposed buffer materialized) —
  not a learning-quality win over backprop, and not feedback-alignment.
- **One chip, one layer-pair in one kernel.** The distributed neighbour-only multi-chip PC vision is
  the multi-F2 shard ([[pc-backward-silicon]], `MULTI_F2_SCALE_OUT.md`) and the roadmap, not this run.
- **Relation to the LM on-fabric result** (`TRAINING_AT_SCALE.md`): complementary. The LM trains by
  SignSGD multiplier-free *backprop* (frozen-routing global backward); this trains by the **cascade
  local-credit layer-step**. Both fully on F2.

## Reproduce

```
# recipe reproduction of the off-by-one diagnosis (committed: onfabric_train/recipe_diag.txt)
PYTHONPATH=. PC_KERNEL_SHIFT=1            python scripts/pc_onfabric_train_ref.py   # -> stuck 0.098
PYTHONPATH=. PC_KERNEL_SHIFT=1 PC_TGT=16  python scripts/pc_onfabric_train_ref.py   # -> 0.896
# on-fabric run (host stages batch + kicks; weights persist in DDR; eval read back per epoch)
# requires an authorized, compatible AFI in the caller's AWS account/region
AWS_PROFILE=... AGFI=agfi-0fdc38db9da6add74 EPOCHS=5 bash scripts/run_pc_train_f2.sh launch
# independent cross-check of the dumped on-fabric weights (committed: onfabric_train/crosscheck.txt)
PYTHONPATH=. python scripts/pc_eval_onfabric_weights.py <trainedW0.i32> <trainedW1.i32>
```

Artifacts: `results/predictive_coding/onfabric_train/` (train.txt, trainedW0/W1.i32, crosscheck.txt,
recipe_diag.txt, img.txt, pctrain.log). Kernel
`tikitorch/hardware/hls/forge_kernels/forge_pc_train_step.cpp`; driver
`tikitorch/fpga/silicon/forge_pc_train_f2.c`; recipe `scripts/pc_onfabric_train_ref.py`.
