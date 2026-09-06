# Scaling tropical predictive coding to real tasks (MNIST / Fashion-MNIST / CIFAR-10)

**Date:** 2026-06-18 · **Hardware:** AWS EC2 g5.4xlarge (1× A10G) · 3 seeds/config
**Code:** `scripts/pc_scale_mnist.py`, `scripts/pc_scale_cifar.py`, driven by
`scripts/run_pc_scale_gpu.sh` (PC_TASK=mnist|cifar). Raw data:
`results/predictive_coding/scale_mnist.jsonl` (144 rows), `scale_cifar.jsonl` (18 rows),
`scale_mnist_iterative.jsonl` (error-relaxation: sync / async / cascade),
`scale_mnist_cascade_ce.jsonl` (CE cascade reconciliation), `scale_cifar100.jsonl`
(CIFAR-100 VGG-10/13).
**Audited:** every headline claim below was adversarially re-checked against the raw
JSONL (a workflow of independent verifiers); corrections from that audit are folded in.

> **Implementation correction (2026-09-06):** only the state-settling `nudge`
> rows in `scale_mnist.jsonl` were produced before `settle_nudge` began applying
> its configured `gamma`; they therefore used an effective unit relaxation step.
> Treat the nudge-collapse measurements and conclusions as historical, pending a
> rerun with current code. Cascade, error-relaxation, backprop, and CIFAR results
> do not use that method and are unaffected. The 18 affected raw records are
> `scale_mnist.jsonl` lines 37–45 and 82–90; they intentionally remain unchanged
> as provenance and do not record `gamma`. Corrected `pc_scale_mnist.py` reruns
> default to `scale_mnist_gamma_damped.jsonl` and record `algorithm_revision`,
> `protocol`, `gamma`, `beta`, `n_sweeps`, `epochs`, `residual`, `lr`, `batch`,
> `smoke`, `n_train`, and `n_test`; do not append them to or merge them with the
> legacy file. The default
> `matched` protocol applies the requested epoch/residual settings to every mode.
> `archived-budget` applies the archived three-epoch, non-residual budget to
> `nudge` only, so corrected `gamma=0.125` and legacy-effective `gamma=1` can be
> compared under the same historical budget.

## TL;DR

Moving dyadic tropical PC off the 12×12 toy onto full datasets gives one strong
positive, one confirmation, one genuine tropical win, one resolved negative, and
one historical result awaiting a corrected-gamma rerun:

- **CIFAR-10 (the headline):** a VGG-10 whose dyadic convolution/linear forward
  and cascade weight-gradient operands are ±2^k reaches **91.4%**, vs **92.2%**
  float (−0.80pp) — and the float twin
  (92.17% mean) essentially matches Qi et al.'s VGG-10 backprop number (92.21%), so
  this is calibrated **near-parity under a dyadic internal-datapath constraint they
  do not impose.** BatchNorm, cross-entropy, and the SGD/Adam update remain outside
  that measured closure boundary.
- **The dyadic conv/linear *forward path* costs little accuracy** (`bp_dyadic`
  ≤1.4pp below float on VGG; on the residual MLP it actually matches/beats float
  — see below). This is not an audit of every inference operator.
- **Tropical win:** the tropical max-residual is *more* depth-stable than float's
  additive residual (MNIST d32: dyadic 0.971 vs float 0.915).
- **Negative 1 → RESOLVED:** the MLP cascade trailed backprop by ~2–5pp *only because
  it used an MSE top-error*; switching to **cross-entropy** (the new default) closes the
  gap to within seed noise (MNIST −0.1pp) — the same algorithm as the CIFAR `MFGrad`
  path. The two cascades are reconciled.
- **Historical negative 2 (see the correction above):** the archived
  state-settling **nudge** runs collapsed to chance at full scale while using an
  unintended unit relaxation step. **Error-relaxation** (frozen routing) remains
  unconditionally stable, matches the cascade exactly, and keeps the async value;
  current-gamma state settling still needs a full-scale rerun.

## Setup

- **Datasets:** MNIST, Fashion-MNIST (784-dim, MLP); CIFAR-10 (VGG conv).
- **Modes:**
  - `bp_float` — plain float net, backprop (the ceiling).
  - `bp_dyadic` — ±2^k convolution/linear forward
    (`DyadicLinear`/`DyadicConv2d`), with an ordinary backprop weight update. It
    isolates the deployed dyadic layer representation, not whole-model closure.
  - `cascade` — weight gradients use operand-quantized (±2^k) operands. On MLP
    this is `DyadicPCNet.train_step_cascade` (cached-routing backward); on CIFAR
    it is the gradient-equivalent realized by `MFGrad` hooks (autograd through
    the dyadic forward + per-layer operand quantization). CE, Adam/SGD, and
    non-dyadic model operators are declared boundaries.
  - `nudge` — the iterative soft-nudge settling regime (`train_step_nudge`); the
    archived rows used an unintended effective `gamma=1`, while current-code
    scale behavior is unmeasured.
- **MLP:** hidden 256, depths 4/8/16/32; non-residual baseline (5 epochs; nudge 3)
  and a residual run (tropical max-skip; 8 epochs; depths 8/16/32).
- **VGG:** vgg7/vgg10 (conv+BN+ReLU+maxpool, single FC head), 30 epochs, SGD+cosine.
- Reference: Qi et al. 2025 report VGG-10 CIFAR-10 backprop = 92.21%.

## CIFAR-10 — dyadic-internal-path VGG (3 seeds, 30 epochs)

| net | `bp_float` | `bp_dyadic` (±2^k conv/linear fwd) | `cascade` (dyadic weight-gradient operands) |
|---|---|---|---|
| VGG-7  | 0.894 | 0.880 | 0.879 |
| VGG-10 | **0.922** | **0.917** | **0.914** |

- `bp_float` VGG-10 = 0.9217 mean (seed-0 0.9221) ≈ Qi's 0.9221 → **baseline
  calibrated to their setup.**
- `bp_dyadic` trails float by only **0.46pp** at VGG-10 (1.36pp at VGG-7):
  the dyadic convolution/linear forward path has little measured accuracy cost.
- `cascade` vs `bp_dyadic` = **−0.35pp** at VGG-10 (a small but *consistent* deficit,
  ~1.5× the VGG-10 cascade seed spread of 0.23pp; VGG-7 spread is wider at 0.74pp);
  vs the float ceiling, `cascade` is **−0.80pp** (VGG-10), **−1.46pp** (VGG-7). So on
  conv, dyadic weight-gradient operands cost well under a point → **near-parity.**
- Cost: cascade wall-clock ≈ 2× (285s vs 146s/run at VGG-10) — purely the `MFGrad`
  hooks recomputing weight grads in sim; on fabric that accumulation is shift-add.

### CIFAR-100 + deeper VGG (3 seeds, 40 epochs) — parity holds at 100 classes

| | `bp_float` | `bp_dyadic` | `cascade` |
|---|---|---|---|
| VGG-10 | 0.715 | 0.712 | 0.710 |
| VGG-13 | 0.718 | 0.709 | 0.707 |

The dyadic convolution/linear forward (`bp_dyadic`) is within **−0.29pp** of float
at VGG-10 and **−0.90pp** at VGG-13; the dyadic-gradient `cascade` is within
**−0.50 / −1.12pp**
of float (and ~−0.2pp vs `bp_dyadic`). So the ±2^k parity extends cleanly from 10 to
**100 classes**, and the deeper VGG-13 holds (no collapse), though it does not improve
over VGG-10 on this dataset. Data: `scale_cifar100.jsonl`.

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
| MNIST | cascade (CE) | 0.974 | 0.973 | **0.970** |
| Fashion | bp_dyadic | 0.877 | 0.874 | **0.873** |
| Fashion | bp_float | 0.880 | 0.870 | 0.859 |
| Fashion | cascade (CE) | 0.867 | 0.867 | 0.868 |

(The cascade now uses the cross-entropy top-error — the default — which matches
backprop at depth; the earlier MSE cascade that trailed is in "Resolved" below.)

- **Residual recovers depth:** every mode stays far above chance through d32
  (vs the non-residual collapse). Depth is representationally usable again.
- **Tropical win:** `bp_dyadic` (tropical max-skip `max(relu(Wx),x)`) is *flatter*
  than `bp_float` (additive skip) at depth — MNIST d32 **0.971 vs 0.915**, Fashion
  d32 0.873 vs 0.859. The bounded, idempotent max-skip is more depth-stable than the
  additive one.

### Resolved: cross-entropy closes the cascade-vs-backprop gap

The MLP cascade originally **trailed `bp_dyadic` by a depth-growing 2–5pp** (MSE
cascade: MNIST −1.8/−2.4/−4.9pp at d8/16/32; Fashion −2.5/−2.3/−2.7). Reconciliation
shows the cause is the **loss function, not anything tropical**: `cascade_grads`
injected the MSE/L2-energy top-error (`logits − target`), while `bp_dyadic` and the
CIFAR `MFGrad` path both backprop **cross-entropy**. Operand-quantization harshness was
ruled out first (2/3-term dyadic quantization did not help). Switching the cascade's
top-error to CE (`softmax(logits) − target`, the new default) closes the gap entirely
— and dyadic operand quantization then costs ~nothing (CE+quantized ≈ CE+real):

| cascade(CE) − bp_dyadic | MNIST d8 | d16 | d32 | Fashion d8 | d16 | d32 |
|---|---|---|---|---|---|---|
| (3-seed mean) | −0.15pp | −0.06pp | −0.10pp | −0.97pp | −0.71pp | −0.48pp |

On MNIST the cascade now matches `bp_dyadic` within seed noise and is **flat to d32**
(0.974→0.973→0.970, flatter than float's 0.976→0.962→0.915); on Fashion it is within
~1pp and the gap *shrinks* with depth (the opposite of the old MSE behaviour). The
softmax is a per-sample boundary op (over the 10 output classes); the network-internal
credit path retains dyadic operands. Because the error-relaxation shares the
top-error, **the iterative regime inherits the same fix** (sync == cascade(CE)). So
the **dyadic internal credit path** now tracks backprop at depth on both the MLP and
CIFAR conv — the two paths are reconciled (same algorithm; the MLP was just
handicapped by MSE). This result does not close the CE or optimizer boundary.

### Historical negative: the legacy unit-step nudge regime

The archived `nudge` run collapsed to **chance (~0.10 test *and* train) at every
tested depth (4/8/16; d32 not run) on both datasets**. On the 12×12 toy it reached
~0.76 at d16. Because the archived implementation ignored its configured `gamma`
and used a unit relaxation step, this is evidence about that historical recipe,
not the corrected implementation. A full-scale rerun is required.

## A separately validated iterative regime: error-relaxation

**Historical diagnosis (instrumented).** `settle_nudge` settles the *state* energy
by preconditioned Gauss–Seidel. In the archived unit-step run, that iteration's
stability depended on the weight magnitude, and as the dyadic weights grew during
training the settle **diverged**:
on full MNIST d4, epoch 0 reaches 0.81, then the settle energy blows up to ~10²⁰ by
epoch 1 and accuracy falls to chance. Within that unit-step implementation,
β/sweeps best was 0.43 (vs cascade 0.92), and *more* settling was *worse* (it
over-settled out of the feed-forward basin). A two-phase EqProp variant did not
help that archived configuration either: here the free phase **is** the
feed-forward pass (eps_free ≈ 0), so the contrastive update reduced to the
single-phase one.

**A separately validated alternative.** At the true equilibrium `dE/dx_l = 0`
gives `E_l = A_l^T E_{l+1}` — the
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

So error-relaxation now **works at full MNIST scale in simulation**: its neighbour-local
updates tolerate partial participation, lag, and arbitrary scheduling while reaching
the same credit signal. It still has the same directional top-down dependency as the
cascade, so it does not eliminate the backward relation. Honest framing: sync
error-relaxation *is* the cascade; **schedule tolerance** is the value-add, not new
accuracy or demonstrated hardware closure. Cost: the async
draws (participation coin + order permutation) serialize on a GPU so the sim is slow;
on genuinely async hardware they are free.

## Caveats & limitations

- **Two different "cascade"s:** MLP uses `cascade_grads`; CIFAR uses `MFGrad` (the
  gradient-equivalent, not the literal settling loop). They are *not* the same code
  path, so code-path equivalence should not be inferred from the shared label. With
  matched cross-entropy top errors, both track their corresponding dyadic-backprop
  baseline in the reported runs.
- **Epoch budget:** non-residual baseline ran 5 epochs, residual 8. The d32 collapse
  is genuine (chance-level *train* acc = vanishing, not slow), but the 5-vs-8 epoch
  difference is a confound; non-residual was not re-run at 8 epochs.
- **Narrow sweep:** single hidden width (256), single-FC-head VGG/ResNet variants,
  three seeds for the principal MNIST/CIFAR runs, but only one Tiny-ImageNet seed;
  no data-augmentation ablation or matched-architecture comparison to Qi's VGG-15.
- **MNIST d16 is high-variance** (non-residual cascade spread 0.574–0.922) — both
  cascade and BP are mid-collapse there; treat d16 non-residual as transitional.
- **Energy/throughput not measured** — these are accuracy results on GPU sim; the
  dyadic operands create a shift-add deployment opportunity, not a measured
  whole-recipe or real-hardware advantage here.
- **Async robustness is demonstrated in *simulation*** (we emulate partial
  participation / stale reads / random order); it is not a measurement on real
  asynchronous hardware.
- **The error-relaxation async confirmation is MNIST-only** (the GPU run hit its
  time backstop on the sim-slow async mode before Fashion; the single-seed depth-scan
  covered both). State-settling `settle_nudge` is retained for the analog/physical
  (voltage-relaxation) reading; its current-gamma scale behavior is unmeasured, so
  use the validated `train_step_relax` path for now.

## Verdict

Tropical/dyadic PC **scales to real tasks for the forward/cascade regime**: a VGG-10
with dyadic convolution/linear and internal weight-gradient operands hits CIFAR-10
near-parity with float backprop at Qi's scale, and the tropical max-residual is a
genuine depth-stability win. The **error-relaxation regime also works at full MNIST
scale in simulation** (sync == cascade; async within ~0.1pp), where the archived unit-step
state-settling run diverged. And the MLP cascade now **matches backprop at depth**
once it uses a cross-entropy top-error (the prior 2–5pp gap was an MSE artifact) —
reconciling it with the CIFAR `MFGrad` path. Net: the dyadic internal-credit variants
track backprop at scale; full-step closure and measured deploy energy remain separate,
open claims.

## Next steps

1. **Corrected state settling at scale** — run the matched default and the
   `archived-budget` gamma control above before drawing conclusions about convergence
   or scale fragility.
2. **Matched Tiny-ImageNet architecture + multiple seeds** — replace the current
   one-seed ResNet-18 result with a like-for-like VGG-15 comparison and uncertainty.
3. **Error-relaxation at depth + on conv** — extend the async confirmation to Fashion
   and to the dyadic VGG (CIFAR), and push depth past 32.
4. **Audit and measure the deploy recipe** — declare or close BatchNorm/pooling/loss/
   optimizer boundaries, then measure energy/throughput for the resulting hard model.

## Reproduce

The MNIST runner now defaults to `--protocol matched` and a three-epoch budget sized
for the EC2 backstop: requested `--epochs` and `--residual` settings apply uniformly
across modes. It writes
`scale_mnist_gamma_damped.jsonl` by default and records `protocol`, effective epochs,
effective residual setting, settling hyperparameters, algorithm revision, source
hashes, runtime metadata, smoke/full status, and actual train/test counts per row.
Keep this output separate from the historical file.

For a controlled comparison against the affected archived nudge budget, use
`--protocol archived-budget --modes nudge`. That protocol applies three epochs and
`residual=false` to nudge only. Run `gamma=0.125` and `gamma=1` to isolate damping;
the latter recreates the old *effective step*, but both rows come from corrected code
and must not be presented as literal reproduction of the old binary/source state.

```
export AWS_PROFILE=my-profile
PC_TASK=mnist bash scripts/run_pc_scale_gpu.sh launch          # matched protocol (default)
PC_TASK=mnist PC_ARGS="--residual --depths 8 16 32 --modes cascade bp_dyadic bp_float --epochs 8" \
  bash scripts/run_pc_scale_gpu.sh launch                      # Phase 0b depth-fix
PC_TASK=mnist PC_ARGS="--residual --depths 8 16 32 --modes relax relax_async cascade --epochs 8" \
  bash scripts/run_pc_scale_gpu.sh launch                      # iterative regime (error-relaxation)
PC_TASK=cifar MAX_MINUTES=300 bash scripts/run_pc_scale_gpu.sh launch   # Phase 1
# status | fetch | terminate ; results stream to S3 and self-terminate

# Controlled gamma comparison (run in the prepared environment; distinct outputs):
PYTHONPATH=. python scripts/pc_scale_mnist.py --matrix --protocol archived-budget \
  --modes nudge --gamma 0.125 \
  --out results/predictive_coding/scale_mnist_nudge_archived_budget_g0125.jsonl
PYTHONPATH=. python scripts/pc_scale_mnist.py --matrix --protocol archived-budget \
  --modes nudge --gamma 1 \
  --out results/predictive_coding/scale_mnist_nudge_archived_budget_g1.jsonl
```
