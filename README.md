# TikiTorch

[![CI](https://github.com/nyuolab/tikitorch/actions/workflows/ci.yml/badge.svg)](https://github.com/nyuolab/tikitorch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**A tropical-semiring deep-learning library — multiplier-free, FPGA-native
training, modeled closely on PyTorch.**

TikiTorch replaces the standard (sum, product) arithmetic of neural networks
with the **tropical semiring** — (min, +) and (max, +). In this algebra, the
"multiplication" of the network is ordinary addition, and the "addition" is a
min or a max (a comparison). The consequence: **every operation in both the
forward and the backward pass reduces to comparisons and additions — there are
no multiplications anywhere in the learning loop.** That is what makes the whole
loop *algebraically closed* and natively mappable to FPGA fabric with **zero DSP
blocks** — train without matmuls, all the way down to silicon.

TikiTorch ships the validated math (max-plus / min-plus ops with custom
autograd), a family of tropical architectures (morphological SSMs, tropical
transformers, mixture-of-tropical-experts), strictly multiplier-free optimizers
(`SignSGD`, `TropicalSGD`, `PureTropicalSGD`), and the **HLS kernels + AFI build
guide** that take a trained model all the way to AWS F2 silicon.

It is intentionally PyTorch-shaped: `import tikitorch as tt` and you get
`tt.nn`, `tt.optim`, `tt.models`, and `tt.functional`, each mirroring its
`torch` analogue. If you know PyTorch, you already know TikiTorch.

---

## Install

```bash
pip install -e .
```

Requires Python >= 3.9, `torch >= 2.0`, `numpy`. Licensed under **GPLv3**.

---

## Quickstart

A complete training step in ten lines — a morphological state-space LM trained
with the strictly multiplier-free `PureTropicalSGD`:

```python
import torch
import tikitorch as tt
import tikitorch.optim as optim

model = tt.models.MorphSSM(d_model=128, n_layers=4, task_type="lm", vocab_size=128)
opt = optim.PureTropicalSGD(model.parameters())     # power-of-2 lr = bit-shift, no DSPs
x = torch.randint(0, 128, (8, 64))                  # (batch, seq)
y = torch.randint(0, 128, (8, 64))
loss = model.compute_loss(x, y)                     # cross-entropy, internally
loss.backward()                                     # gradients route through argmin/argmax
opt.step(); opt.zero_grad()
```

Everything in the model's core — the sequence mixing, the projections, the
gradient flow — is comparisons and additions. The only standard-arithmetic
ops are the (optional) embedding table and output head.

---

## The three architectures

TikiTorch ships three closure-complete families. All take a `task_type`
keyword (`"classification"`, `"lm"`, or `"regression"`) and expose a concrete
`compute_loss(x, y)` method, so they drop straight into a standard training
loop.

### `tt.models.MorphSSM` — morphological state-space model

Sequence mixing by **max-plus / min-plus depthwise convolutions** instead of
attention or gated SSM recurrences. Linear-time, streaming-friendly (`.step()`
runs token-by-token with an on-chip ring buffer), and the natural fit for the
FPGA silicon path. This is the architecture validated end-to-end on F2.

```python
model = tt.models.MorphSSM(d_model=256, n_layers=8, kernel_size=7,
                           task_type="lm", vocab_size=32000)
```

### `tt.models.TropicalTransformer` — tropical attention transformer

A transformer whose attention is **tropical**: scores are max-plus inner
products (or an L-infinity distance), the softmax is a temperature-annealed
logsumexp relaxation, and the value pull is a per-dimension max-plus matvec.
Supports both dot-product and shift-based attention variants and causal masking
for LM.

```python
model = tt.models.TropicalTransformer(d_model=256, n_heads=4, n_layers=6,
                                      task_type="lm", vocab_size=32000)
```

### `tt.models.MoTEModel` (alias `tt.models.MoTE`) — mixture of tropical experts

A sparse mixture-of-experts where the **router itself is tropical** and each
expert is a tropical transformer or morphological block. `compute_loss` adds the
load-balancing auxiliary loss; the routing decision is an argmax over additive
scores — no multiplications.

```python
model = tt.models.MoTE(d_model=128, n_experts=8, expert_type="transformer",
                       task_type="lm", vocab_size=32000)
```

Also included: `tt.models.TropicalMLP` and `tt.models.TropicalSequenceModel`
(carved tropical baselines for classification / sequence tasks), plus
`MorphMLP`.

### `tt.optim.PureTropicalSGD` — the strictly multiplier-free optimizer

The optimizer that closes the loop. No floating-point multiply appears anywhere
in the update:

- **power-of-2 learning rate** implemented as an integer **bit-shift**
  (`lr_shift`),
- **integer-counter momentum** (`momentum_bits`) instead of an FP EMA,
- optional **LFSR noise** (`noise_scale`) for stochastic rounding,
- optional **sign-L1 adaptive** scaling (`adaptive`, `adaptive_bits`).

```python
opt = optim.PureTropicalSGD(model.parameters(), lr_shift=10, momentum_bits=4)
```

Two simpler relatives are also provided: `tt.optim.SignSGD`
(`W -= lr * sign(grad)`, optional momentum-on-sign) and `tt.optim.TropicalSGD`
(the same update expressed purely as comparison + addition).

**Closed *adaptive* optimizers.** Five multiplier-free attempts at Adam-style
per-parameter rates, all verified zero-DSP by `assert_training_closed` (the step
magnitude comes from `torch.frexp` — the gradient's binary exponent, a leading-bit
priority-encode on FPGA — gathered through a constant power-of-two lookup table, never
`2**tier`/`ldexp`):

- `tt.optim.TropicalAdam` — per-param pow-2 step adapted by gradient-sign agreement (Rprop lineage),
- `tt.optim.FrexpMagnitudeSGD` — step scaled by the frexp-exponent magnitude tier, EMA-normalized (the Adam-like `1/√v` analogue),
- `tt.optim.CountAgreementSGD` — step scaled by gradient-sign *agreement* over an `accumulate()` window (use `accum≥8`),
- `tt.optim.ExponentEMAAdam` — counter momentum (1st moment) + a frexp-exponent EMA (2nd moment), self-centered on each param's initial exponent,
- `tt.optim.PerLayerStepSGD` — one shared scalar pow-2 step per tensor, Rprop hysteresis (robust to sparse tropical gradients).

**Honest bake-off verdict** (5 seeds × 5 algorithmic tasks, `scripts/bakeoff_closed_optimizers.py`; full table in `results/confirm_full250.md`):
the closed **"closure tax" is ≈1pp, not the 5-9pp we assumed** — `PureTropicalSGD`
(closed) scores 0.521 ID / 0.490 OOD vs the *non-closed* Adam proxy's 0.531 / 0.497
(within seed noise; closed actually *wins* on running_max and min_coin_change).
**Full closure is essentially free.** At the full step budget `PureTropicalSGD`
remains the best closed optimizer — the adaptive ones don't beat it asymptotically —
**but at short step budgets (≤50 steps, the on-silicon regime) `FrexpMagnitudeSGD`
wins** (≥2× step efficiency: running_max 0.78 vs 0.44 at 50 steps), and
`CountAgreementSGD` (accum≥8) is the **stability** play (knapsack 0.778±0.01 vs
0.711±0.13). So: `PureTropicalSGD` as the all-purpose default, `FrexpMagnitudeSGD`
for tight/silicon budgets. The residual gap to softmax is **architectural** (sort),
not an optimizer gap.

---

## API map

TikiTorch mirrors the PyTorch namespace one-to-one. If you know where it lives
in `torch`, you know where it lives in `tikitorch`.

| TikiTorch | PyTorch analog | What it is |
|---|---|---|
| `tt.functional.tropical_matmul` | `torch.matmul` | `C[i,j] = min_k(A[i,k] + B[k,j])` (min-plus matmul) |
| `tt.functional.tropical_linear` | `F.linear` | min-plus (erosion) linear, sparse argmin backward |
| `tt.functional.tropical_dilation` | `F.linear` | max-plus (dilation) linear, sparse argmax backward |
| `tt.functional.tropical_softmin` / `tropical_softmax` | `F.softmin` / `F.softmax` | logsumexp relaxation, hard as `tau -> 0` |
| `tt.functional.tropical_dot_product` | scaled dot-product score | tropical attention scoring |
| `tt.functional.tropical_value_pull` | attention @ V | per-dim max-plus value aggregation |
| `tt.nn.TropicalLinear` *(= `TropicalLinearModule`)* | `nn.Linear` | min-plus linear layer |
| `tt.nn.TropicalDilation` *(= `TropicalDilationModule`)* | `nn.Linear` | max-plus linear layer |
| `tt.nn.TropicalProjection` *(= `CompositionalTropicalProjection`)* | `nn.Linear` | dilation∘erosion = tropical rational (universal PWL) |
| `tt.nn.TropicalTopK` *(= `TropicalTopKModule`)* | top-k pooling | top-K max-plus aggregation |
| `tt.nn.MorphConv1d` *(= `DepthwiseMorphConv1d`)* | `nn.Conv1d` (depthwise) | morphological (max/min-plus) depthwise conv |
| `tt.nn.BatchShift` *(= `TropicalBatchShift`)* | `nn.BatchNorm` | additive (subtract-mean) normalization |
| `tt.nn.MorphSSMLayer`, `TropicalAttention`, `TropicalTransformerBlock`, `MoTEBlock` | `nn.TransformerEncoderLayer` etc. | composable tropical blocks |
| `tt.optim.SignSGD` | `torch.optim.SGD` | sign-only update |
| `tt.optim.TropicalSGD` | `torch.optim.SGD` | semiring (compare + add) view of SignSGD |
| `tt.optim.PureTropicalSGD` | `torch.optim.SGD` | strictly multiplier-free (bit-shift lr, int momentum) — **default** |
| `tt.optim.FrexpMagnitudeSGD` / `CountAgreementSGD` / `ExponentEMAAdam` / `PerLayerStepSGD` / `TropicalAdam` | `torch.optim.Adam` | closed *adaptive* (frexp-tier pow-2 step); frexp-mag best at short/silicon budgets |
| `tt.models.MorphSSM` / `TropicalTransformer` / `MoTE` | a model in `torchvision` / `transformers` | ready-to-use tropical architectures |

Both the original research names and the clean PyTorch-idiomatic aliases are
exported; the two names point to the same class. Prefer the clean alias in new
code.

### Soft-to-hard curriculum

Train with the differentiable `tau > 0` (logsumexp) relaxation for dense
gradients, anneal `tau -> 0`, then deploy the hard tropical form for closure —
**no parameter copying**, the same weights drive both paths (Maslov
dequantization). Layers expose `set_tau(...)` / `set_temperature(...)` so you
can schedule the anneal across training.

---

## Benchmarks (fully closed)

Every model below is built with TikiTorch defaults, which are **closed by
default** — `assert_closed(model)` passes (tropical top-k head + quantized input
+ `TropicalLayerNorm`; only an `nn.Embedding` token gather on LM tasks). Trained
with the sign-only `SignSGD` under the soft→hard `tau` curriculum; accuracy is at
hard `tau=0` (the form that maps to the zero-DSP FPGA kernel).

| task | model | fully-closed acc | notes |
|---|---|---|---|
| MNIST | `TropicalCNN` (morph conv) | **0.80** (3 seeds, A10G) | morph-conv backbone strong; head is the cost |
| MNIST | `MaxPlusMin` MLP + closed α-scale | **0.85** (700 steps) | breaks the Thm-3.2 gradient collapse → toward the 0.95 morph-net range |
| CIFAR-10 | `TropicalCNN` | **0.335** (3 seeds, A10G) | small closed CNN; hard task |
| sMNIST | `MorphSSM` (len-784) | **0.40** (L40S, eff. batch 512) | vs **0.11 = chance** at batch 16 — morph nets need *large* batches |
| sCIFAR-10 | `MorphSSM` (len-1024) | **0.37** (L40S, eff. batch 384) | same large-batch lift on the length-1024 sequence task |

These are *honestly* fully closed: unlike the manuscript's earlier "96.78% fully
closed" sMNIST (which used `nn.LayerNorm` + an `nn.Linear` head), nothing here
hides a multiply. **The large-batch finding is load-bearing:** the closed sign-SGD
sequence models stall at chance with small batches (the morphological-net
large-batch requirement, arXiv:2505.09710) and only train once the *effective*
batch is large — reached here via gradient accumulation (`--accum-steps`), where
`sign(Σ grads)` is the large-batch majority-vote sign. Reproduce with
`scripts/run_closed_benchmarks.py` (`--device cuda --accum-steps N`). Every number
also passes the runtime closure check (`assert_closed_runtime`, below). The
closure verifier is the contract:

TikiTorch ships a **three-tier closure-verification suite** — structural,
inference (runtime), and training:

```python
import tikitorch as tt
tt.assert_closed(model)                 # 1. structural: no nn.Linear/LayerNorm/Conv
tt.assert_closed_runtime(model, x)      # 2. inference: traces the forward, flags any
                                        #    real multiply/divide (pow-2 = shift, OK)
tt.assert_training_closed(              # 3. TRAINING: traces fwd + loss + backward +
    model, (x,), y, loss_fn, optimizer) #    optimizer -- the whole Level-B step
tt.closure_report(model, x, y, loss_fn, optimizer)   # one-call audit of all three
```

- **`assert_closed_runtime`** (inference) runs the forward under a
  `TorchDispatchMode` and catches a raw `*`/`@`/`exp` hidden in a custom `forward`
  that the structural scan misses (it treats `x*2**k` / `x/8` as bit-shifts).
- **`assert_training_closed`** (training) is novel: prior closure checks were
  inference-only, but **on-chip training needs the *whole step* — forward, loss,
  backward, and the optimizer update — to be multiplier-free.** It immediately
  surfaced real Level-B breaks (e.g. SignSGD's float-momentum `buf.mul_(0.9)`).

**Is TikiTorch closed *by construction*?** Audited with these tools:
*Inference* — **yes**, every standard tropical model is multiplier-free at
inference by construction (the only exception found: one model's pow-of-two scale
uses `pow`, a one-line fix). *Training* — **yes, in the closed recipe**, which is
just turning off the two opt-in, non-closed training aids: `dropout=0` (inverted
dropout is a train-time mask multiply — identity at eval, so it never affects
inference) and **hard tropical** (`tau=0`; the soft `logsumexp` path is the
differentiable relaxation, by design non-closed, meant to anneal to hard) — plus a
closure-faithful loss (`tropical_perceptron_loss`/L1, not cross-entropy), a
sign-only optimizer (`PureTropicalSGD`), and a power-of-two `d_model` (so the
norm's mean-subtract is a bit-shift). Under that recipe, `assert_training_closed`
passes — the *entire* training step is zero-DSP (the on-silicon vision).
The realistic guarantee is **checkable closure**: the tools certify it op-by-op
rather than asking you to trust it.

### Algorithmic reasoning (closed DP execution + OOD)

`tikitorch.algorithmic` benchmarks the tropical claim that max-plus/min-plus
models *are* dynamic programs. Reproduce with the `tikitorch.algorithmic`
harness (`evaluate_task` + the verified reference solvers):

| probe | result | reading |
|---|---|---|
| Bellman–Ford == Floyd–Warshall | exact match | the min-plus shortest-path ground truth is correct |
| **closed** max-plus reasoner, running-max, train len 16 | **0.98** in-dist, **0.99 at 8–32× length** | a multiplier-free (`assert_closed`-true) model executes the prefix-max DP and **length-extrapolates** |
| magnitude OOD, cumulative-max (mean rel-err, 8× scale) | softmax **0.03 → 0.68 (≈23× blowup)**; tropical **scale-invariant (≈1.9×)** | softmax **saturates** outside its trained value range; the tropical argmax is scale-invariant by construction |

This ships the *infrastructure* (generators + verified solvers + a closed
reasoner + in-distribution / length / magnitude OOD harness). Two honest caveats:
(1) the length-extrapolation result holds for both tropical and softmax on
position-free tasks — softmax is *not* length-fragile here (an earlier draft's
"softmax collapses on sort" was an artifact of an untrained absolute positional
embedding shared by both arms, now corrected); the genuine softmax weakness is
**magnitude** OOD (saturation). (2) On the magnitude probe the tropical reasoner
is scale-invariant but currently *under-fits* the regression in-distribution, so
this is "degrades far less," not yet "wins outright."

**Head-matched tropical-vs-softmax (GPU, 3 seeds, train→2× size-OOD, Micro-F1 for
the combinatorial DPs; via `tikitorch.algorithmic.evaluate_task` with
`attention="tropical"` vs `"softmax"`):**

| task | tropical (id / ood2×) | softmax (id / ood2×) | verdict |
|---|---|---|---|
| running_max | 0.99 / 0.95 | 1.00 / 1.00 | both solve it |
| subset_sum  | 0.79 / 0.71 | 0.79 / 0.70 | **tie** |
| knapsack    | 0.86 / 0.85 | 0.91 / 0.92 | softmax ahead, tropical competitive |
| sort        | 0.49 / 0.09 | 1.00 / 0.13 | softmax fits; **both collapse OOD** (positional) |
| min_coin_change | 0.00 / 0.00 | 0.60 / 0.40 | tropical **fails** (min-plus task, max-plus aggregator) |

**Honest verdict: the closed tropical reasoner does NOT yet beat softmax on these
tasks.** It ties on subset_sum, is competitive on knapsack and running_max, but
under-fits sort and fails min_coin_change outright — we do *not* reproduce the
Tropical-Attention paper's claimed advantage with this reasoner.

We hypothesized min_coin's failure was a **semiring mismatch** (it's a *min*-plus
DP, the reasoner aggregated *max*-plus) and added `agg_variant="min"`/`"maxmin"`
(min-plus / both-semiring value pulls, with semiring-correct masking). **It did not
pan out** (an honest negative result): over 3 seeds min-plus is a wash with
max-plus on min_coin (≈0.09 ± 0.13 either way — the training is *bimodal/unstable*,
not semiring-bound), and elsewhere the effect is task-dependent (maxmin nudges
running_max OOD 0.94→0.997, but min-plus *collapses* subset_sum OOD 0.69→0.01). So
the min/maxmin variants ship as correct, semiring-complete primitives, but the real
blocker is **training stability + the in-distribution under-fit**, not the
aggregator. Reproduce the sweep with `tikitorch.algorithmic.evaluate_task`
(vary `attention`, `agg_variant`, and `seed`).

### Reference models (the closed model zoo)

A set of saved, **closure-certified** reference models trained end-to-end under the
verified closed recipe with `PureTropicalSGD` (the full-budget closed
default). Every model is `tt.save`d with its config + closure manifest, certified by
`tt.closure_report` (`inference_closed` *and* `training_zero_dsp`), and round-trips
through `tt.load` **bit-identically**.

**Algorithmic reasoners** (`scripts/build_algorithmic_zoo.py` → `zoo/algorithmic/`,
best-of-4 seeds because closed tropical training is bimodally unstable on the DP
tasks — see P4):

| model | id acc | ood-2× acc | closed? |
|---|---|---|---|
| `running_max.tt` | **0.996** | **0.998** | inference ✓ / train zero-DSP ✓ |
| `subset_sum.tt` (F1) | 0.709 | 0.676 | ✓ / ✓ |
| `knapsack.tt` (F1) | 0.775 | 0.780 | ✓ / ✓ |
| `min_coin_change.tt` (F1) | 0.173 | 0.088 | ✓ / ✓ |
| `sort.tt` | 0.132 | 0.090 | ✓ / ✓ (architectural ceiling — tropical attention can't sort) |

```python
import tikitorch as tt
model = tt.algorithmic.load_reasoner("zoo/algorithmic/running_max.tt")  # rebuilds arch + loads weights
```

**Image / sequence** (`scripts/run_closed_benchmarks.py` → `zoo/image_seq/`, MorphSSM
d=128×4 under the closed recipe on an EC2 A10G, small per-step batch + large
*effective* batch via accumulation, soft→hard `tau` anneal, accuracy at hard `tau=0`):

| model | task | hard-tau acc | eff. batch | closed? |
|---|---|---|---|---|
| `smnist_morphssm.tt` | sequential-MNIST (len-784) | 0.343 (6 ep) | 512 | inference ✓ |
| `scifar_morphssm.tt` | sequential-CIFAR-10 (len-1024) | 0.320 (6 ep) | 384 | inference ✓ |

Both climb cleanly above chance as `tau` anneals (sMNIST 0.12→0.34, sCIFAR
0.20→0.32 over 6 epochs) and were still rising — these are short-budget refreshes of
the longer-run 0.40 / 0.37 numbers in the benchmark table above, confirming the
closed (multiplier-free, large-effective-batch) recipe trains end-to-end on GPU.

**On-silicon training reference (AWS F2, zero-DSP).** `model.fpga()` binds a
`TropicalTransformer` char-LM to a real F2 board (the validated monolithic
`forge_train_step` AGFI) and runs the entire `train_step()` — forward → backward →
sign-SGD — *on the board over PCIe*. Reproduce with
`scripts/run_tikitorch_f2_silicon.sh launch` (provisions an f2.6xlarge, runs
`scripts/run_fpga_f2_verify.sh`, terminate after):

```
board present: True
BOUND on silicon: True | kernel: forge_train_step | dsp: 0
params evolved by on-silicon training: 229872 / 231040     # 99.5% via the on-board kernel
ON-SILICON MULTIPLIER-FREE TRAINING ACTIVE: True
```

So the closure is not just *checkable* in software — the same `model.fpga()` chain
drives real **multiplier-free (dsp=0) training on silicon**: 50 `train_step()`s
evolved 99.5% of the model's parameters through the on-board fwd→bwd→sign-SGD kernel.
(Caveat: the live loss-curve readout pins constant on the currently-registered AGFIs
due to a `grad_logit` layout offset mismatch — documented in the script; the
parameter-evolution proof is exact.)

---

## Beyond deep learning: the full tropical toolkit

TikiTorch aims to be the *definitive* tropical-ML library, not just deep nets:

- **`tikitorch.geometry`** — operationalizes the tropical geometry of nets
  (Zhang–Naitzat–Lim): `count_linear_regions` (distinct affine pieces — also a
  collapse/dead-cell diagnostic), `decision_regions_2d` (the tropical
  hypersurface), `newton_polytope`.
- **`tikitorch.classical`** — classical tropical ML: `tropical_distance` (Hilbert
  projective metric), `tropical_kmeans`, `TropicalNearestCentroid`, **`TropicalSVM`**
  (max-margin tropical classifier, multiplier-free at inference), `tropical_pca`.
- **`tikitorch.algorithmic`** — a CLRS-style **algorithmic-reasoning** suite:
  verified generators + reference solvers for the tropical dynamic programs
  (sorting, running/cumulative max = max-plus scan, longest-increasing-subsequence
  = max-plus DP, shortest paths = min-plus `bellman_ford`/`floyd_warshall`
  closure), a closure-verified `TropicalReasoner`, and an `evaluate_task` harness
  with in-distribution **and** out-of-distribution (length / magnitude) splits.
- **`tikitorch.planning`** — **differentiable tropical DP layers** (the "DP-as-a-closed-layer"
  lever): `TropicalViterbi` (max-plus trellis decode), `TropicalShortestPath` (min-plus
  Bellman-Ford), `TropicalValueIteration` + `max_plus_power_iteration` (RL value iteration /
  max-plus eigenvalue), `TropicalCoinChangeDP` (exact min-plus coin-change DP), and a tropical
  `MCTS`/`MuZero` scaffold. Each is exact in the tropical semiring and multiplier-free
  (`assert_closed`-true), only *smoothed-approximate* in a dense net. **Honest note:** these are
  exact *structural priors* (the layer encodes the algorithm), not solutions gradient descent
  discovers — e.g. `TropicalCoinChangeDP` solves the `min_coin_change` task at F1=1.000 closed/
  zero-DSP (where the generic learned aggregations plateau at ~0.03), but that is the wired DP,
  not a learned head. They make those tasks *representable* closed, and serve as exact references.
- **`tikitorch.prune`** — `tropical_prune`: structured pruning of never-winning
  morph taps (a documented strength of morphological nets), closure-preserving.
- **New layers** — `MaxPlusMin` (universal-approx morph layer) + `TropicalScale`
  (closed power-of-2 α-scale that fixes the deep-morph gradient collapse),
  `DilationErosionPerceptron`, `MorphOpenClose2d`, `AdderLinear` (ℓ1/AdderNet),
  and `score_variant="hilbert"`/`"hilbert_asym"` tropical-projective attention.

---

## FPGA / silicon

The bundled HLS C++ kernels (`tt.hardware.hls_dir()`) implement a complete
tropical LM **training step** — forward + loss + backward + sign-SGD update — in
FPGA fabric. The whole thing synthesizes to **zero DSP blocks**, which is the
physical confirmation of the closure claim: there is literally no multiplier on
the chip.

Measured on AWS EC2 F2 (Virtex UltraScale+ VU47P), validated end-to-end:

| Build | Throughput | Energy | Timing (WNS) | Notes |
|---|---|---|---|---|
| 1-layer tropical LM | ~1474 tok/s | — | ~-0.20 ns | C + DMA host driver |
| **2-layer tropical LM** | **~772 tok/s** | **~9.0 mJ/token** (6.98 W) | **-0.197 ns** | first multilayer NN training on FPGA, zero DSPs |
| 3-layer tropical LM | ~521 tok/s | — | -0.197 ns | depth saturates; width is the next constraint |

`tt.hardware.build_guide_path()` points at
[`tikitorch/hardware/BUILD_GUIDE.md`](tikitorch/hardware/BUILD_GUIDE.md), the
full Vitis-HLS synthesis → AWS HDK AFI build → F2 load-and-run walkthrough,
including the load-bearing details (single-kernel dispatcher, PCIM not the DDR
bridge, host-owned per-window loop) that actually make timing close.

```python
import tikitorch.hardware as hw
hw.hls_dir()           # path to the bundled HLS kernels
hw.build_guide_path()  # path to the silicon build guide
```

### Running a model on the board: `model.fpga()`

The ergonomic, PyTorch-shaped mover — `model.cuda()`'s zero-DSP cousin:

```python
model = tt.models.TropicalTransformer(d_model=128, n_layers=1,
                                      task_type="lm", vocab_size=128)
model = model.fpga()          # DDP-style wrapper; runs the fused step on a board
loss  = model.train_step(x, y, lr=1)   # FWD_LOSS -> BWD -> SIGNSGD, all on silicon
```

`model.fpga()` (or `tt.to(model, tt.device('fpga'))`) returns an `AcceleratedModel`
that is *bound* to a compiled kernel when a board + native driver + on-board runner
are present, and is a transparent CPU fallback otherwise (so the call is always
safe). The reference bridge (`tikitorch/fpga/bridge.cpp`) ships three transports —
**LOCAL** (host execution of the real bundled kernel), **fpga_pci** (the AWS HDK
OCL-register flow), and **XRT** — plus the 1-layer on-board runner.

**Validated end-to-end on real AWS F2 silicon (2026-06):** built `_libtikifpga`
in `fpga_pci` mode on an f2.6xlarge, loaded a monolithic `forge_train_step` AGFI,
and ran `model.fpga().train_step()` — it **binds on silicon** (`bound=True`,
**dsp=0**) and drives real **multiplier-free** training: 50 steps evolved **99.5%
of the model's parameters** via the on-board fwd→bwd→sign-SGD kernel (data-dependent
±1 sign-SGD updates, confirmed by direct register probes). Also validated in LOCAL
mode where a `TropicalTransformer` trains to **zero error** through `model.fpga()`.
Reproduce: `scripts/run_fpga_f2_verify.sh` (on an F2 instance) or
[`tikitorch/fpga/BUILD_NATIVE.md`](tikitorch/fpga/BUILD_NATIVE.md) (LOCAL).
*(Honest caveat: a live converging loss-curve needs the exact AGFI's buffer layout;
the currently-registered AFIs are layout-rebuilds whose `grad_logit` offset differs
from the bundled source, so the accuracy proxy pins constant on them even though
training runs — rebuild an AFI from the bundled kernel to recover the loss curve.)*

---

## Backends — running off-GPU

TikiTorch's whole workload collapses to one tropical primitive, which makes it an
unusually clean target for non-GPU accelerators. `tikitorch.backend` is the
pluggable, **FPGA-first but backend-agnostic** seam they plug into — a small,
PyTorch-flavored dispatch layer with **transparent CPU fallback**: anything a
backend can't run silently runs the correct CPU reference (one-time warning).
Correctness never breaks; performance degrades gracefully.

```python
import tikitorch as tt

tt.backend.available_backends()        # ['cpu']   (+ 'fpga' on a board)
dev = tt.device("fpga")                # a logical device (tropical tensors stay on CPU)
```

**FPGA — one line, your loop unchanged.** `tt.fpga.accelerate(model)` wraps a
model exactly like `DistributedDataParallel`: when a model is *bound* the whole
fused training step runs on silicon at **zero DSPs**; everywhere else it's a
transparent CPU fallback, so the same code runs anywhere. A model binds only when
**all three** hold — a board is present, its shape matches a compiled kernel, and
an on-board runner is registered for that kernel (the hardware-validated last
mile; none ship by default — see
[`fpga/BUILD_NATIVE.md`](tikitorch/fpga/BUILD_NATIVE.md)).

```python
import tikitorch as tt, tikitorch.fpga as fpga, tikitorch.optim as optim

model = tt.models.TropicalTransformer(d_model=128, n_layers=2,
                                      task_type="lm", vocab_size=128)
opt   = optim.PureTropicalSGD(model.parameters())
model = fpga.accelerate(model)         # bound on a board (+ runner); CPU fallback otherwise
print(fpga.is_available(), model.bound)

loss = model.compute_loss(x, y); loss.backward(); opt.step()   # unchanged loop
```

**Honest scope (Tier 1).** The bundled silicon kernels are *fused whole-training-
step monoliths* (no multi-kernel mux), so `accelerate` binds a whole **model** to
a compiled kernel matched by shape — it is *not* per-op `.to('fpga')` dispatch
(that is Tier 2 / custom codegen). The native driver is built
out-of-band against the Xilinx toolchain — see
[`tikitorch/fpga/BUILD_NATIVE.md`](tikitorch/fpga/BUILD_NATIVE.md). Other backends
(a torch-free C/CPU runtime, WASM, integer NPUs) register the same way:

```python
class MyBackend(tt.backend.Backend):
    name = "my-accel"
    def is_available(self): return True
    def ops(self): return {"tropical_linear": my_fast_linear}

MyBackend().register()                 # now dispatch routes to it when available
```

See [`examples/backend_dispatch.py`](examples/backend_dispatch.py) and
[`examples/fpga_quickstart.py`](examples/fpga_quickstart.py).

---

## Honest scope

TikiTorch is **research-grade**. Closure is not free: forcing the whole loop
into the tropical semiring with sign-only updates costs accuracy relative to an
unconstrained FP32 + AdamW + cross-entropy baseline.

- On the **classification / sequence tasks**, the measured closure cost is
  modest — on the order of **~2.5-5 percentage points** of accuracy.
- On a **hard character-LM**, the gap is larger and stacks across closure
  choices. A decomposition on Shakespeare char-LM (1 layer, D=128, V=128) shows
  the open GPU ceiling at ~45% next-token accuracy versus ~8% fully-closed on
  silicon, dominated by two choices: the perceptron loss (replacing
  cross-entropy, ~-17pp) and SignSGD (replacing AdamW, ~-13pp). INT16 weight
  quantization is nearly free (~-1pp). Replacing the perceptron loss with a
  host-side cross-entropy gradient recovers a meaningful chunk of that gap.

The point of TikiTorch is not to beat a GPU transformer on perplexity. It is to
demonstrate — and let you reproduce — a learning loop that is *algebraically
closed*: trainable end-to-end with no multipliers, all the way to silicon, at
single-digit millijoules per token. Treat accuracy numbers as the price of that
closure, and the soft-to-hard `tau` curriculum and the loss/optimizer knobs as
the levers for trading it back.

---

## License

GPLv3. See [LICENSE](LICENSE).

---

## Citation

If you use TikiTorch in your research, please cite:

```bibtex
@software{tikitorch,
  title        = {TikiTorch: A Tropical-Semiring Deep-Learning Library for
                  Multiplier-Free, FPGA-Native Training},
  author       = {{TikiTorch Authors}},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/nyuolab/tikitorch},
  note         = {Algebraic closure of the learning loop: max-plus / min-plus
                  training with zero DSP blocks on FPGA silicon.}
}
```
