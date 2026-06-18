# Scaling tropical predictive coding to real tasks (MNIST / Fashion-MNIST / CIFAR-10)

**Date:** 2026-06-18 · **Hardware:** AWS EC2 g5.4xlarge (1× A10G) · 3 seeds/config
**Code:** `scripts/pc_scale_mnist.py`, `scripts/pc_scale_cifar.py`, driven by
`scripts/run_pc_scale_gpu.sh` (PC_TASK=mnist|cifar). Raw data:
`results/predictive_coding/scale_mnist.jsonl` (144 rows), `scale_cifar.jsonl` (18 rows),
`scale_mnist_iterative.jsonl` (error-relaxation: sync / async / cascade).
**Audited:** every headline claim below was adversarially re-checked against the raw
JSONL (a workflow of independent verifiers); corrections from that audit are folded in.

## TL;DR

Moving dyadic tropical PC off the 12×12 toy onto full datasets gives one strong
positive, one confirmation, one genuine tropical win, and two honest negatives:

- **CIFAR-10 (the headline):** a *fully multiplier-free* (±2^k) VGG-10 trained by
  the cascade reaches **91.4%**, vs **92.2%** float (−0.80pp) — and the float twin
  (92.17% mean) essentially matches Qi et al.'s VGG-10 backprop number (92.21%), so
  this is a calibrated **near-parity at Qi's scale under the harder multiplier-free
  constraint they don't impose.**
- **Multiplier-free *inference* is nearly free on conv** (`bp_dyadic` ≤1.4pp below
  float on VGG; on the residual MLP it actually matches/beats float — see below).
- **Tropical win:** the tropical max-residual is *more* depth-stable than float's
  additive residual (MNIST d32: dyadic 0.971 vs float 0.915).
- **Negative 1 (correction):** the multiplier-free *cascade* matches backprop on
  CIFAR (−0.35pp) but on the deep MLP it **trails by ~2–5pp, growing with depth**.
- **Negative 2 → RESOLVED (see "The iterative regime, fixed" below):** the
  state-settling **nudge** regime collapses to chance at full scale (its energy
  diverges as weights grow). The fix is **error-relaxation** (frozen routing), which
  is unconditionally stable, matches the cascade exactly, and keeps the async value.

## Setup

- **Datasets:** MNIST, Fashion-MNIST (784-dim, MLP); CIFAR-10 (VGG conv).
- **Modes:**
  - `bp_float` — plain float net, backprop (the ceiling).
  - `bp_dyadic` — ±2^k forward (`DyadicLinear`/`DyadicConv2d`), backprop weight
    update. Tests multiplier-free **inference** (what the FPGA forward kernel does).
  - `cascade` — fully multiplier-free: weight grads from operand-quantized (±2^k)
    operands. On MLP this is `DyadicPCNet.train_step_cascade` (cached-routing
    backward); on CIFAR it is the gradient-equivalent realized by `MFGrad` hooks
    (autograd through the dyadic forward + per-layer operand quantization).
  - `nudge` — the iterative soft-nudge settling regime (`train_step_nudge`).
- **MLP:** hidden 256, depths 4/8/16/32; non-residual baseline (5 epochs; nudge 3)
  and a residual run (tropical max-skip; 8 epochs; depths 8/16/32).
- **VGG:** vgg7/vgg10 (conv+BN+ReLU+maxpool, single FC head), 30 epochs, SGD+cosine.
- Reference: Qi et al. 2025 report VGG-10 CIFAR-10 backprop = 92.21%.

## CIFAR-10 — multiplier-free dyadic VGG (3 seeds, 30 epochs)

| net | `bp_float` | `bp_dyadic` (±2^k fwd) | `cascade` (fully mult-free) |
|---|---|---|---|
| VGG-7  | 0.894 | 0.880 | 0.879 |
| VGG-10 | **0.922** | **0.917** | **0.914** |

- `bp_float` VGG-10 = 0.9217 mean (seed-0 0.9221) ≈ Qi's 0.9221 → **baseline
  calibrated to their setup.**
- `bp_dyadic` trails float by only **0.46pp** at VGG-10 (1.36pp at VGG-7):
  multiplier-free inference is nearly free.
- `cascade` vs `bp_dyadic` = **−0.35pp** at VGG-10 (a small but *consistent* deficit,
  ~1.5× the VGG-10 cascade seed spread of 0.23pp; VGG-7 spread is wider at 0.74pp);
  vs the float ceiling, `cascade` is **−0.80pp** (VGG-10), **−1.46pp** (VGG-7). So on
  conv, fully multiplier-free training costs well under a point → **near-parity.**
- Cost: cascade wall-clock ≈ 2× (285s vs 146s/run at VGG-10) — purely the `MFGrad`
  hooks recomputing weight grads in sim; on fabric that accumulation is shift-add.

## MNIST / Fashion-MNIST — deep MLP

### Non-residual baseline (5 epochs; mean test acc, 3 seeds)

| dataset | mode | d4 | d8 | d16 | d32 |
|---|---|---|---|---|---|
| MNIST | cascade | 0.977 | 0.968 | 0.737 | **0.103** |
| MNIST | bp_dyadic | 0.976 | 0.972 | 0.855 | **0.114** |
| MNIST | bp_float | 0.977 | 0.973 | 0.892 | **0.114** |
| Fashion | bp_float | 0.874 | 0.865 | 0.741 | **0.100** |

**Plain deep MLPs collapse to chance at d32 — *even float backprop*** (train acc is
also chance → vanishing, not under-training). So the depth wall is the classic
plain-deep-MLP pathology, **not** a dyadic or cascade defect. (This qualifies the
earlier "cascade flat to d100" note, which was 12×12 with identity/residual
machinery, not plain deep MLPs on full data.)

### Residual depth-fix (tropical/additive max-skip, 8 epochs)

| dataset | mode | d8 | d16 | d32 |
|---|---|---|---|---|
| MNIST | bp_dyadic | 0.976 | 0.974 | **0.971** |
| MNIST | bp_float | 0.976 | 0.962 | 0.915 |
| MNIST | cascade | 0.958 | 0.950 | 0.921 |
| Fashion | bp_dyadic | 0.877 | 0.874 | **0.873** |
| Fashion | bp_float | 0.880 | 0.870 | 0.859 |
| Fashion | cascade | 0.852 | 0.851 | 0.845 |

- **Residual recovers depth:** every mode stays far above chance through d32
  (vs the non-residual collapse). Depth is representationally usable again.
- **Tropical win:** `bp_dyadic` (tropical max-skip `max(relu(Wx),x)`) is *flatter*
  than `bp_float` (additive skip) at depth — MNIST d32 **0.971 vs 0.915**, Fashion
  d32 0.873 vs 0.859. The bounded, idempotent max-skip is more depth-stable than the
  additive one.

### Correction: cascade vs backprop on the MLP

The cascade matches backprop at shallow depth (d4: +0.1pp) but **trails `bp_dyadic`
by a systematic, depth-growing margin** on the deep MLP. The deficit is uniform in
direction (cascade < bp_dyadic in every config) and, at d8/d16, well-resolved (both
modes' 3-seed spreads ≤0.4pp). At d32 the cascade is noisier (spread up to 3pp on
MNIST) so the −4.9pp point is less tightly bounded — but even the cascade's *best*
d32 seed (0.932) trails `bp_dyadic`'s *worst* (0.970):

| | MNIST d8 | d16 | d32 | Fashion d8 | d16 | d32 |
|---|---|---|---|---|---|---|
| cascade − bp_dyadic | −1.8pp | −2.4pp | **−4.9pp** | −2.5pp | −2.3pp | −2.7pp |

So "multiplier-free training = backprop" holds **on CIFAR conv (MFGrad)** but **not
on the deep MLP cached-routing cascade**, where the operand-quantization deficit
compounds with depth. The two `cascade` mechanisms behave differently; reconciling
them is open work.

### Negative: the iterative nudge regime

`nudge` collapses to **chance (~0.10 test *and* train) at every tested depth
(4/8/16; d32 not run) on both datasets**. On the 12×12 toy it reached ~0.76 at d16;
at 784-dim it learns nothing with the toy's hyperparameters (γ/β/sweeps). Root-caused
and fixed below.

## The iterative regime, fixed (error-relaxation)

**Root cause (instrumented).** `settle_nudge` settles the *state* energy by
preconditioned Gauss–Seidel. That iteration's stability depends on the weight
magnitude, and as the dyadic weights grow during training the settle **diverges**:
on full MNIST d4, epoch 0 reaches 0.81, then the settle energy blows up to ~10²⁰ by
epoch 1 and accuracy falls to chance. Tuning cannot save it — β/sweeps best is 0.43
(vs cascade 0.92), and *more* settling is *worse* (it over-settles out of the
feed-forward basin). Two-phase EqProp doesn't help either: here the free phase **is**
the feed-forward pass (eps_free ≈ 0), so the contrastive update reduces to the
single-phase one.

**The fix.** At the true equilibrium `dE/dx_l = 0` gives `E_l = A_l^T E_{l+1}` — the
equilibrium errors *are* the backprop/cascade error propagated top-down. So relax the
**error** variables directly (with routing **frozen** at the feed-forward pass, like
the cascade) instead of the state energy. With frozen routing this is a **triangular
(nilpotent)** system: it converges to the exact cascade gradient in ≤depth local
rounds and is **unconditionally stable regardless of weight magnitude**. Diagnostic:
the frozen-relaxed gradient is cosine-similarity **0.99** with the cascade gradient
per layer, and the synchronous form is **bit-exact equal** to `cascade_grads` (a unit
test asserts `torch.equal`). Exposed as `DyadicPCNet.error_relax_grads` /
`train_step_relax`; data `scale_mnist_iterative.jsonl`.

Full MNIST, residual, 8 epochs, 3 seeds:

| depth | cascade | error-relax **sync** | error-relax **async** (50% participation + stale + random-order) |
|---|---|---|---|
| d8  | 0.958 | 0.958 | **0.958** |
| d16 | 0.950 | 0.950 | **0.950** |
| d32 | 0.921 | 0.921 | **0.921** |

- **Sync == cascade** (it is the iterative solve of the same triangular system).
- **Async ≈ sync within ~0.1pp** — with half the nodes silent per round, lagged
  (stale) reads, and no global update order. Given enough local rounds the system
  fully converges, so at d32 it reaches the *identical* per-seed values. (The earlier
  single-seed depth-scan on a 5k subset showed the same: d4–d32 async within ~0.6pp.)

So the iterative regime now **works at full scale** — local, no global backward
controller, robust to partial participation / lag / no clock — i.e. it keeps the
distributed/neuromorphic value the cascade's strict ordered pass cannot offer, while
reaching the same credit signal. Honest framing: sync error-relaxation *is* the
cascade; the **async capability** is the value-add, not new accuracy. Cost: the async
draws (participation coin + order permutation) serialize on a GPU so the sim is slow;
on genuinely async hardware they are free.

## Caveats & limitations

- **Two different "cascade"s:** MLP uses `cascade_grads`; CIFAR uses `MFGrad` (the
  gradient-equivalent, not the literal settling loop). They are *not* the same code
  path and they behave differently (CIFAR matches BP, MLP trails) — do not conflate.
- **Epoch budget:** non-residual baseline ran 5 epochs, residual 8. The d32 collapse
  is genuine (chance-level *train* acc = vanishing, not slow), but the 5-vs-8 epoch
  difference is a confound; non-residual was not re-run at 8 epochs.
- **Narrow sweep:** single hidden width (256), single-FC-head VGG, 3 seeds, CIFAR-10
  only (no CIFAR-100 / Tiny-ImageNet yet), no data-aug ablation.
- **MNIST d16 is high-variance** (non-residual cascade spread 0.574–0.922) — both
  cascade and BP are mid-collapse there; treat d16 non-residual as transitional.
- **Energy/throughput not measured** — these are accuracy results on GPU sim; the
  multiplier-free advantage is a deploy-time (FPGA/ASIC) property, not shown here.
- **Async robustness is demonstrated in *simulation*** (we emulate partial
  participation / stale reads / random order); it is not a measurement on real
  asynchronous hardware.
- **The error-relaxation async confirmation is MNIST-only** (the GPU run hit its
  time backstop on the sim-slow async mode before Fashion; the single-seed depth-scan
  covered both). State-settling `settle_nudge` is retained for the analog/physical
  (voltage-relaxation) reading but is scale-fragile — use `train_step_relax`.

## Verdict

Tropical/dyadic PC **scales to real tasks for the forward/cascade regime**: a fully
multiplier-free VGG-10 hits CIFAR-10 parity with float backprop at Qi's scale, the
±2^k forward is nearly free, and the tropical max-residual is a genuine depth-stability
win. The **iterative/distributed regime now also works at full scale** via
error-relaxation (sync == cascade; async within ~0.1pp), where the naive state-settling
diverged. The remaining honest cost: the MLP cascade trails backprop by a depth-growing
2–5pp (mechanism-dependent vs the CIFAR `MFGrad` path).

## Next steps

1. **CIFAR-100 / Tiny-ImageNet, deeper VGG (vgg13/15)** — extend the calibrated
   Qi-scale comparison; this is the strongest direction.
2. **Reconcile the two cascades** — port the CIFAR `MFGrad` operand-quantization into
   the MLP path (or vice-versa) and find why the MLP cascade compounds error with
   depth.
3. **Error-relaxation at depth + on conv** — extend the async confirmation to Fashion
   and to the dyadic VGG (CIFAR), and push depth past 32.
4. **Measure deploy metrics** (energy/throughput) for the dyadic VGG forward — that is
   where the multiplier-free constraint is supposed to pay off.

## Reproduce

```
export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
PC_TASK=mnist bash scripts/run_pc_scale_gpu.sh launch          # Phase 0 baseline
PC_TASK=mnist PC_ARGS="--residual --depths 8 16 32 --modes cascade bp_dyadic bp_float --epochs 8" \
  bash scripts/run_pc_scale_gpu.sh launch                      # Phase 0b depth-fix
PC_TASK=mnist PC_ARGS="--residual --depths 8 16 32 --modes relax relax_async cascade --epochs 8" \
  bash scripts/run_pc_scale_gpu.sh launch                      # iterative regime (error-relaxation)
PC_TASK=cifar MAX_MINUTES=300 bash scripts/run_pc_scale_gpu.sh launch   # Phase 1
# status | fetch | terminate ; results stream to S3 and self-terminate
```
