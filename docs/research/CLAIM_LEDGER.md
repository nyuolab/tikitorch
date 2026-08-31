# TikiTorch research claim ledger

Status: canonical working ledger, 2026-08-31.

This file controls research wording for TikiTorch. A result may be described
more narrowly than this ledger, but not more broadly without adding evidence
and updating the ledger. Raw artifacts and executable tests outrank summaries;
this ledger outranks manuscript prose and the top-level README when wording
conflicts.

## Repository identity

TikiTorch is **PyTorch-shaped infrastructure for tropical and hard-routed deep
learning**. Its scope includes tropical/dyadic operators, modules, models,
optimizers, closure auditing, serialization, reference differentiation, and
FPGA export and kernels.

“PyTorch-shaped” means familiar namespaces and training-loop conventions such
as `tt.nn`, `tt.functional`, `tt.models`, and `tt.optim`. It does not mean API,
numerical, or performance compatibility with every PyTorch operator.

TikiTorch is **not** the ontogenic-agent repository. Persistent lives,
developmental transfer, action-support plasticity, continual memory, embodiment,
and the true-agent specification belong in the sibling `ontogenic-agent`
project. That project may consume a tagged TikiTorch release; it must not be
used to expand TikiTorch's claims.

## Status vocabulary

- **Established:** checked by committed implementation and tests.
- **Measured:** supported by an archived run, within the stated configuration.
- **Draft evidence:** numerically supported, but not yet reproduced from a clean
  archived TikiTorch commit.
- **Preliminary:** informative, but a named gate prevents headline use.
- **Negative:** a tested claim that failed or was narrowed.
- **Open:** proposed work, not evidence.

## Canonical claims

### TT-LIB-001 — PyTorch-shaped tropical-learning library

**Status:** Established.

**Allowed wording:** TikiTorch supplies a PyTorch-shaped Python surface for
tropical and dyadic neural-network research, including operators, models,
optimizers, auditing, serialization, and hardware paths.

**Evidence:** [package metadata](../../pyproject.toml), [public import
tests](../../tests/test_import.py), [architecture guide](../guide/architecture.md),
and the [`tikitorch` package](../../tikitorch/).

**Do not claim:** complete PyTorch compatibility; production maturity; optimized
performance for every operator; arbitrary-model FPGA compilation; or that every
shipped configuration is multiplier-free.

**Upgrade gate:** green supported-platform CI, consistent versioning, a tagged
release and DOI, stable API documentation, and an external-use record.

### TT-CLOSE-001 — Recipe-scoped closure auditing

**Status:** Established as a software audit; physical closure remains
build-specific.

**Allowed wording:** TikiTorch can audit a declared recipe at three software
levels: registered-module structure, an executed forward trace, and an executed
forward--loss--backward--optimizer step. Power-of-two scaling is classified as
a shift. HLS reports and silicon checks separately validate a compiled kernel.

**Evidence:** [`tikitorch.audit`](../../tikitorch/audit.py), [audit
tests](../../tests/test_audit.py), [training-closure
tests](../../tests/test_dyadic_backward_closure.py), and FPGA synthesis evidence
listed below.

**Required boundary language:**

- `assert_closed` is a structural deny-list and cannot see raw arithmetic in a
  custom `forward`.
- Runtime tracing covers only executed paths and representative inputs.
- A dyadic layer's exported forward schedule can be shift--add while its eager
  host implementation calls BLAS.
- A dyadic forward does not imply a multiplier-free autograd weight gradient.
- Soft routing (`tau > 0`), ordinary cross-entropy, dropout, dense heads,
  conventional normalization, and adaptive optimizers can leave the closed
  subset.
- A software pass is not proof of zero multipliers after synthesis.

**Do not claim:** “every TikiTorch operation contains no multiplication” or
“the auditor proves zero DSPs.” Closure is a property of a concrete hard model,
loss, reverse rule, optimizer, input path, and compiled build.

**Upgrade gate:** every headline hardware configuration must bind a closure
report to exact source, build flags, HLS operator/resource reports, routed
reports, and an on-silicon known-answer test.

### TT-ROUTE-001 — Exact selected-VJP certificates

**Status:** Draft evidence; implementation and artifacts are being re-homed in
TikiTorch.

**Allowed wording:** For a declared finite graph of trace-sufficient hard
operators, the forward route plus required auxiliary fields and a supplied
output cotangent determines the selected state and additive-parameter VJPs.
Certificates compose through the registered fan-out, residual, shared-parameter,
and finite-unrolling rules under deterministic ties.

**Evidence:** [route-certificate paper workspace](../../papers/route_certificates/),
[claim boundary and preserved evidence](../../papers/route_certificates/README.md),
[working implementation](../../tikitorch/autodiff/), [working result
archive](../../results/route_certificates/), and route tests currently under
[`tests/`](../../tests/).

**Exactness evidence currently used by the draft:** 38 registered reference
tests; a small residual-DAG comparison with maximum FP32 parameter-gradient
difference `4.77e-7` from eager autograd; and maximum parameter difference
`2.98e-8` after five SGD steps. Packed replay and the direct route reverse use
the same selected reverse-mode computation.

**Do not claim:** a new gradient; a biologically distinct learning rule; generic
activation-free training; route-index-only sufficiency; or coverage of dense
parameter outer products, smooth attention, ordinary normalization, nonlinear
loss construction, and optimizer state. The existing
`pc_convrnn.RouteCertificate` support inventory is not by itself the portable
VJP certificate.

**Upgrade gate:** archive the versioned schema and operator registry; re-run all
tests and matched trajectories in a clean checkout; freeze the tie and
output-cotangent boundary; and regenerate tables from append-only native
TikiTorch results.

### TT-ROUTE-002 — MorphSSM retained-state representation

**Status:** Draft evidence, configuration-specific.

**Allowed wording:** On three measured hard-routed MorphSSM configurations, the
packed certificate payload is 2.45--3.07x smaller than the registered
checkpoint-retained-state comparator without recomputation. At
`d=128, L=4, T=96`, payload is 87,588 bytes, or 912.4 bytes/token, and the
reported ratio is 2.97x.

**Accounting boundary:** those ratios use certificate payload. Including the
portable directory and separately disclosed loss logits changes the three
ratios to 1.70x, 2.49x, and 2.80x. Report both views.

**Evidence:** [paper README](../../papers/route_certificates/README.md),
[manuscript table](../../papers/route_certificates/main.tex), [accounting
sensitivity](../../papers/route_certificates/supplement.tex), and [raw working
results](../../results/route_certificates/actual_morphssm/).

**Do not claim:** lower process RSS, allocator peak, accelerator peak, DRAM
traffic, latency, energy, or generic whole-model activation memory. The small
Python prototype was 3.3--3.8x slower than eager autograd; the compiled five-arm
comparison is not complete.

**Upgrade gate:** matched eager, compiled-autograd, checkpoint, compiled-unpacked
VJP, and compiled-packed-replay arms with exact trajectory equivalence, full
cycle latency, retained bytes, process/device peak, recomputation, and measured
bytes moved.

### TT-FPGA-INF-001 — Coherent multiplier-free LM inference on F2

**Status:** Measured.

**Allowed wording:** A host-QAT-trained, four-layer `D=128` tropical transformer
runs forward-only inference on AWS F2 with a DSP-free shift--add datapath. It
measured 2.4230 held-out bits/character over 3,840 positions and matched the
kernel-faithful C++ reference. The 62.5 MHz build closed timing with WNS
`+0.392 ns`.

**Evidence:** [result report](../../results/lm_shakespeare/S1_KV2_ON_SILICON.md)
and [raw F2 log](../../results/lm_shakespeare/onfabric_gen/s1_kv2_f2_gen.txt).

**Do not claim:** on-fabric training of this quality result; competitive
throughput; large-language-model scale; or model-generated text as a quantitative
quality metric. Generation took about 426 ms/character and is a correctness
demonstration.

**Upgrade gate:** throughput-oriented timing-closed build, matched reference
quality and tokenization, multiple seeds/checkpoints, and a stable export
manifest.

**Secondary inference artifact:** the chess-scale forward kernel measured
9.22 ms/inference (108.4 inference/s) over 4,000 F2 kicks with DSP=0 and routed
timing closure. It is forward inference only; the corresponding training kernel
misses timing and exceeds one SLR. See the [AFI report](../../results/fpga_silicon/MUZERO_CHESS_FWD_AFI.md)
and [comparison audit](../../results/board_scaling/CHESS_MUZERO_COMPARISON.md).

### TT-FPGA-TRAIN-001 — Exact D512 fused learning-loop execution

**Status:** Measured execution; learning quality remains open.

**Allowed wording:** A four-layer, width-512 MorphSSM kernel executes forward,
LUT-loss gradient, reverse, and SignSGD entirely in one F2 invocation. A
30-update run is bit-exact to its C-simulation record. A 2,000-update invocation
processed 256,000 token positions in 418.3963 seconds, sustaining 612 token/s,
with persistent weight-state change.

**Evidence:** [D512 verification archive](../../results/fpga_silicon/d512_verify/),
especially [`BITEXACT`](../../results/fpga_silicon/d512_verify/BITEXACT),
[`train_2000.err`](../../results/fpga_silicon/d512_verify/train_2000.err), and
the [fused kernel](../../tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp).

**Do not claim:** held-out language learning, 256,000 distinct tokens, a quality
improvement, or a competitive accelerator result. The harness repeats one
resident 128-token window for all 2,000 updates. The 30-step record, not every
internal signal or the full 2,000-step trajectory, is compared to C simulation.

**Upgrade gate:** disjoint streaming windows, held-out evaluation, matched
reference checkpoints, multiple seeds, archived final HLS/routed reports, and
an iso-model CUDA arm.

### TT-FPGA-LEARN-001 — Held-out MNIST learning from fabric-only updates

**Status:** Measured, small-scale.

**Allowed wording:** A two-layer dyadic network trained on 16-by-16 MNIST with
forward, error routing, gradient accumulation, momentum, and update running on
F2 reached 74.75% test accuracy. An independent Python path evaluated the
exported fabric-trained weights at 74.13% on a freshly loaded test set.

**Evidence:** [result report](../../results/predictive_coding/ONFABRIC_TRAINING.md)
and [raw artifacts](../../results/predictive_coding/onfabric_train/).

**Physical wording:** the model-data path has no multiplication. Synthesis used
one DSP block as an adder and zero DSP multipliers; therefore call this
“multiplier-free,” not universally “zero DSP.”

**Do not claim:** state-of-the-art MNIST accuracy, training throughput, weight-
transport-free learning, feedback alignment, or a result on full-resolution
MNIST. The implementation reuses forward weights through transposed access,
predictions are imbalanced, and epoch wall time includes host readback and
evaluation. The target-scale compensation and preprocessing differences must
remain disclosed.

**Upgrade gate:** frozen preprocessing and initialization, multiple seeds,
matched CPU/GPU recipe, isolated kernel timing, and a larger held-out task.

### TT-FPGA-SCALE-001 — D128 MorphSSM replica throughput

**Status:** Preliminary; not a submission headline.

**Allowed wording:** The archived D128 build family measured 14,236 token/s for
one kernel, 26,522 token/s for two replicas, and 53,001 token/s for four replicas;
instances matched their reference records and used zero DSP blocks.

**Evidence:** [hallmark result and provenance](../../results/lm_shakespeare/fig_hallmark_lm.md).

**Do not claim:** timing-robust throughput, GPU parity, iso-model speedup, or an
energy win. Headline routed builds have small negative WNS (`-0.020 ns` and
`-0.173 ns`), the current GPU denominator is a different model/optimizer, and
FPGA power is a core rail rather than full-card power. The N=8 result is a
projection, not a measurement.

**Upgrade gate:** timing closure with margin, repeat measurements, iso-model and
iso-quality baselines, and full-card power.

## Negative and narrowing results that must remain visible

### TT-NEG-001 — Exact cascade credit is not a new gradient

The matched cascade/synchronous route reverse computes the selected backprop
gradient. Its value is representation and hardware scheduling, not a distinct
credit direction. State-settling variants that genuinely alter the dependency
story are the scale-fragile variants. Evidence: [parallelism
audit](../../results/predictive_coding/PARALLELISM.md) and [depth
study](../../results/pc_depth_traction/EXP_RESULTS.md).

### TT-NEG-002 — Asynchrony does not yield a free depth speedup

The exact-gradient async round tax grows roughly with depth and inverse
participation. This removes the prior claim of a depth-growing throughput win
over backprop. The surviving claim is scheduling tolerance on a suitable
substrate, not measured GPU acceleration. Evidence: [parallelism
audit](../../results/predictive_coding/PARALLELISM.md).

### TT-NEG-003 — Strict closure currently carries a language-quality cost

In the matched Tiny Shakespeare host study, strict MorphSSM reached 2.793 bpb
versus 2.273 for parameter-matched nanoGPT at the reported budgets. The 2.423
bpc F2 inference model is a different character setup trained by kernel-faithful
QAT and must not be substituted into that comparison. Evidence: [hallmark
ledger](../../results/lm_shakespeare/fig_hallmark_lm.md) and the [fixed-point
investigation](../../results/lm_shakespeare/S1_MULTFREE_CHARLM_WALL.md).

### TT-NEG-004 — No present end-to-end energy win

Core-rail measurements, WNS-derived estimates, and projected replicas do not
support a board-to-board energy claim. Full-card FPGA power under a matched
quality/workload protocol remains open. Evidence: [hallmark
qualifications](../../results/lm_shakespeare/fig_hallmark_lm.md) and the [FPGA
paper draft](../../papers/multiplier_free_fpga/main.tex).

## Next claim gates, in order

1. **Archive:** commit the route-certificate implementation, tests, scripts,
   raw rows, manuscripts, and this ledger on a fetched remote branch; bind every
   result to a clean commit and environment manifest.
2. **Library:** restore ordinary green CI, reconcile package/version/docs claims,
   and tag a reproducible release.
3. **Route certificates:** complete the exact five-arm compiled comparison and
   measure full-cycle latency, retained bytes, peak, recomputation, and traffic.
4. **FPGA learning:** run D512 on disjoint windows with held-out evaluation and a
   matched reference trajectory.
5. **FPGA systems comparison:** close timing with margin; add iso-model CUDA and
   full-card power; then freeze the paper tables.
6. **External validity:** repeat fabric-only learning beyond downsampled MNIST
   before generalizing the learning claim.

No ontogenic-agent result is a gate for these TikiTorch publications.
