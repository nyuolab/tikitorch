# TikiTorch research claim ledger

Status: canonical working ledger, updated 2026-09-15.

This file controls research wording for TikiTorch. A result may be described
more narrowly than this ledger, but not more broadly without adding evidence
and updating the ledger. Raw artifacts and executable tests outrank summaries;
this ledger outranks manuscript prose and the top-level README when wording
conflicts.

The single active paper is now the [unified TikiTorch manuscript](https://github.com/nyuolab/tikitorch/blob/main/manuscript/README.md).
The two scientific drafts in `manuscript/archive/` and their raw results are preserved as
historical sources, not competing current claim authorities. The manuscript's
[evidence ledger](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/EVIDENCE.md) records the selected
current and historical evidence separately.

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

- **Established:** checked by implementation and tests; this status alone does
  not assert a clean committed release or remote CI validation.
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

**Evidence:** [package metadata](https://github.com/nyuolab/tikitorch/blob/main/pyproject.toml), [public import
tests](https://github.com/nyuolab/tikitorch/blob/main/tests/test_import.py), [architecture guide](../guide/architecture.md),
and the [`tikitorch` package](https://github.com/nyuolab/tikitorch/tree/main/tikitorch/).

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
forward--loss--backward--optimizer step. The current executed classifier is
fail-closed (`safe`, `unsafe`, `unknown`), with `strict-eager-v3` semantics and
report schema version 2. Only supported static scaling with the declared
operation/dtype semantics is classified as a shift; a runtime tensor is not
accepted merely because its observed values are powers of two. HLS reports and
silicon checks separately validate a compiled kernel.

**Evidence:** [`tikitorch.audit`](https://github.com/nyuolab/tikitorch/blob/main/tikitorch/audit.py), [audit
tests](https://github.com/nyuolab/tikitorch/blob/main/tests/test_audit.py), [training-closure
tests](https://github.com/nyuolab/tikitorch/blob/main/tests/test_dyadic_backward_closure.py), and FPGA synthesis evidence
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
- Historical research artifacts that set `training_closed=true` after waiving
  division or transcendental operations are not passes under the public
  no-waiver `strict-eager-v3` semantics; preserve the rows as provenance and
  label the narrower internal-path result instead.
- Earlier `strict-eager-v2` reports are historical results, not automatically
  re-certified under the newer fail-closed classifier.
- A software pass is not proof of zero multipliers after synthesis.

**Do not claim:** “every TikiTorch operation contains no multiplication” or
“the auditor proves zero DSPs.” Closure is a property of a concrete hard model,
loss, reverse rule, optimizer, input path, and compiled build.

**Upgrade gate:** every headline hardware configuration must bind a closure
report to exact source, build flags, HLS operator/resource reports, routed
reports, and an on-silicon known-answer test.

### TT-ROUTE-001 — Exact selected-VJP certificates

**Status:** Draft evidence. The implementation and working artifacts are now in
TikiTorch; clean release-candidate reproduction remains pending.

**Allowed wording:** For a declared finite executed graph of trace-sufficient hard
operators, the forward route plus required auxiliary fields and a supplied
output cotangent determines the selected state and additive-parameter VJPs.
Certificates compose through the registered fan-out, residual, shared-parameter,
and finite-unrolling rules under declared deterministic ties. At differentiable
points this is the classical VJP. At ties it is the specified local reverse
program, not necessarily a Clarke subgradient of the composed function.

**Evidence:** [unified manuscript and proofs](https://github.com/nyuolab/tikitorch/blob/main/manuscript/README.md),
[claim boundary and preserved evidence](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/EVIDENCE.md),
[working implementation](https://github.com/nyuolab/tikitorch/tree/main/tikitorch/autodiff/), [working result
archive](https://github.com/nyuolab/tikitorch/tree/main/manuscript/results/route_certificates/), and route tests currently under
[`tests/`](https://github.com/nyuolab/tikitorch/tree/main/tests/).

**Exactness evidence currently used by the draft:** current source-bound
device-tape measurements are TT-ROUTE-003. The executable tests additionally
exercise primitive/composed VJPs, ties, sharing, serialization and supported
training trajectories; numerical tolerance agreement is not bit identity.
Historical reference tests and small residual-DAG results remain in the old
archive, not as a current test-count or current device-performance claim.
The integrated `route_logits` MorphSSM adapter is narrower than the graph
theorem: it rejects parameter aliases, hooks/overrides, custom route-less
backends, recurrent scans and other unsupported configurations.

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

**Allowed wording:** In the historical three-cell hard-routed MorphSSM archive, the
portable certificate is 1.97x, 2.74x, and 2.96x smaller than the registered
shape-derived layer/embedding checkpoint boundary set when both sides are
given the output cotangent. At `d=128, L=4, T=96`, the payload is 87,588 bytes,
or 912.4 bytes/token; the portable certificate is 89,799 bytes and the
boundary-matched ratio is 2.74x.

**Accounting boundary:** these historical supplied-cotangent ratios include the
portable certificate directory. When the historical replay interface retains
loss logits and int64 targets, include those tensors on both sides; the resulting
ratios are 1.69x, 2.48x, and 2.79x. Historical payload-only ratios (2.04x, 2.81x,
and 2.98x) are coding diagnostics. The independently measured current device
payload comparison has its own record and boundary in TT-ROUTE-003.

**Evidence:** [historical paper README](https://github.com/nyuolab/tikitorch/blob/main/manuscript/archive/route_certificates/README.md),
[archived manuscript table](https://github.com/nyuolab/tikitorch/blob/main/manuscript/archive/route_certificates/main.tex), [accounting
sensitivity](https://github.com/nyuolab/tikitorch/blob/main/manuscript/archive/route_certificates/supplement.tex), and [raw working
results](https://github.com/nyuolab/tikitorch/tree/main/manuscript/results/route_certificates/actual_morphssm/).

**Do not claim:** lower process RSS, allocator peak, accelerator peak, DRAM
traffic, latency, energy, or generic whole-model activation memory. The small
historical Python prototype was 3.3--3.8x slower than eager autograd; that is not
a measurement of current device replay. The five-arm full-step benchmark is
now implemented, but deployment-scale CUDA comparisons remain unrun. A
shape-derived boundary is not an optimal executed checkpoint policy.

**Upgrade gate:** matched eager, compiled-autograd, checkpoint, compiled-unpacked
VJP, and compiled-packed-replay arms with exact trajectory equivalence, full
cycle latency, retained bytes, process/device peak, recomputation, and measured
bytes moved.

### TT-ROUTE-003 — Current device-route retained-state audit

**Status:** Draft evidence: freshly measured from source-fingerprinted current
code; clean release-candidate reproduction remains pending.

**Allowed wording:** For three declared FP32 hard-MorphSSM cells, actual device
tape payloads are 36,084, 87,588 and 329,928 bytes, respectively 2.04x, 2.81x
and 2.98x smaller than a specified analytical layer/embedding boundary set.
All cells pass exact logits, canonical integer route consistency and
parameter-VJP comparison at predeclared `atol=rtol=1e-5`; maximum absolute
gradient errors are `7.30e-7`, `2.92e-6` and `1.49e-6`.

**Configuration/accounting:** `(D,L,T)` is `(64,2,96)`, `(128,4,96)` or
`(256,4,192)`; batch 1, vocabulary 38, top-k sum with k=8, kernel 7, no dropout.
Parameter, input and cotangent seeds are 0, 1 and 2; these are three distinct
seed roles, not a multi-seed task study. The common supplied cotangent has
unit L2 norm. The comparator is `(L+1)*B*T*D*4`, not measured checkpoint peak.
Parameters, optimizer, loss, common cotangent, metadata and transient
workspace are excluded. Portable framing is reported separately: 38,298,
91,342 and 333,715 bytes. Its values differ from the historical format.

**Evidence:** [current source-bound record](https://github.com/nyuolab/tikitorch/blob/main/manuscript/evidence/retained_state_layout_2026-09-09.json),
[measurement/check command](https://github.com/nyuolab/tikitorch/blob/main/manuscript/tools/measure_retained_state.py),
[table and boundary ledger](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/EVIDENCE.md).
The route-consistency oracle shares canonical primitives; it is not an
independent algorithm. The record binds all library Python sources and the
measurement script and explicitly reports a dirty worktree.

**Do not claim:** measured peak savings, an optimal checkpoint comparison,
speed, energy, quality, independent route-oracle validation, or general
activation-free training. A byte comparison is not a full-step benchmark.

**Upgrade gate:** clean independent reproduction, then five-arm same-model
measurements and multi-seed useful-task quality/cost evidence under the
[declared experiment protocol](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/PLAN.md#comparison-protocol).

### TT-FPGA-INF-001 — Coherent multiplier-free LM inference on F2

**Status:** Measured.

**Allowed wording:** A host-QAT-trained, four-layer `D=128` tropical transformer
runs forward-only inference on AWS F2 with a DSP-free shift--add datapath. It
measured 2.4230 held-out bits/character over 3,840 positions and matched the
kernel-faithful C++ reference. The 62.5 MHz build closed timing with WNS
`+0.392 ns`.

**Evidence:** [result report](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/lm_shakespeare/S1_KV2_ON_SILICON.md)
and [raw F2 log](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/lm_shakespeare/onfabric_gen/s1_kv2_f2_gen.txt).

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
misses timing and exceeds one SLR. See the [AFI report](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/fpga_silicon/MUZERO_CHESS_FWD_AFI.md)
and [comparison audit](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/board_scaling/CHESS_MUZERO_COMPARISON.md).

### TT-FPGA-TRAIN-001 — D512 fused execution with an exact observable-record match

**Status:** Measured execution; learning quality remains open.

**Allowed wording:** A four-layer, width-512 MorphSSM kernel executes forward,
LUT-loss gradient, reverse, and SignSGD entirely in one F2 invocation. The
archived resident accuracy, weight CRC, and first 16 predictions from a
30-update run match its C-simulation record bit for bit. A 2,000-update
invocation processed 256,000 token positions in 418.3963 seconds, sustaining
612 token/s, with persistent weight-state change.

**Evidence:** [D512 verification archive](https://github.com/nyuolab/tikitorch/tree/main/manuscript/results/fpga_silicon/d512_verify/),
especially [`BITEXACT`](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/fpga_silicon/d512_verify/BITEXACT),
[`train_2000.err`](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/fpga_silicon/d512_verify/train_2000.err), and
the [fused kernel](https://github.com/nyuolab/tikitorch/blob/main/tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp).

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
exported fabric-trained weights at 74.13% on a freshly loaded test set. The
0.62 percentage-point difference is unresolved pipeline variation because the
two evaluations did not freeze identical inputs and preprocessing.

**Evidence:** [result report](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/predictive_coding/ONFABRIC_TRAINING.md)
and [raw artifacts](https://github.com/nyuolab/tikitorch/tree/main/manuscript/results/predictive_coding/onfabric_train/).

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
instances matched their reference records and used zero DSP blocks. Relative to
the one-kernel measurement, aggregate throughput was 1.86x at two replicas and
3.72x at four replicas; these are within-family scaling ratios, not accelerator
speedups.

**Evidence:** [hallmark result and provenance](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/lm_shakespeare/fig_hallmark_lm.md).

**Do not claim:** 2x/4x measured scaling, perfect linearity, timing-robust
throughput, GPU parity, iso-model speedup, or an energy win. Headline routed
builds have small negative WNS (`-0.020 ns`, `-0.097 ns`, and `-0.173 ns` for
N=1, N=2, and N=4), the current GPU
denominator is a different model/optimizer, and FPGA power is a core rail rather
than full-card power. The N=8 result is a projection, not a measurement.

**Upgrade gate:** timing closure with margin, repeat measurements, iso-model and
iso-quality baselines, and full-card power.

## Negative and narrowing results that must remain visible

### TT-NEG-001 — Exact cascade credit is not a new gradient

The matched cascade/synchronous route reverse computes the selected backprop
gradient. Its value is representation and hardware scheduling, not a distinct
credit direction. State-settling variants genuinely alter the dependency story,
but the archived scale-fragility evidence for `settle_nudge` came from an
implementation that ignored its configured `gamma` and used a unit step.
Corrected-gamma scale behavior is pending rerun. Evidence: [parallelism
audit](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/predictive_coding/PARALLELISM.md) and [depth
study](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/pc_depth_traction/EXP_RESULTS.md).

### TT-NEG-002 — Asynchrony does not yield a free depth speedup

The exact-gradient async round tax grows roughly with depth and inverse
participation. This removes the prior claim of a depth-growing throughput win
over backprop. The surviving claim is scheduling tolerance on a suitable
substrate, not measured GPU acceleration. Evidence: [parallelism
audit](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/predictive_coding/PARALLELISM.md).

### TT-NEG-003 — Historical language endpoints favor the dense baseline

In the historical Tiny Shakespeare host study, the selected annealed MorphSSM
endpoint reached 2.793 bpb versus 2.273 for nanoGPT, but with unequal token
budgets (13.8M versus 3.46M) and unequal parameter counts (1,004,032 versus
770,816 in the raw rows). The archived summary's parameter-matched wording
does not hold for these selected endpoints. The MorphSSM run starts at
`tau=0.5` and ends at zero; its legacy ROM/LZC audit waivers are not a
`strict-eager-v3` full-training closure certificate. This is a negative
historical quality result, not a matched-compute causal cost of strict closure
or a measurement of the current device-tape implementation. The raw selected
endpoints are [MorphSSM](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/lm_shakespeare/strictH_bpe_anneal_s9decay_6k.jsonl)
and [dense GPT](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/lm_shakespeare/bpeA_dense.jsonl).
The 2.423
bpc F2 inference model is a different character setup trained by kernel-faithful
QAT and must not be substituted into that comparison. Evidence: [hallmark
ledger](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/lm_shakespeare/fig_hallmark_lm.md) and the [fixed-point
investigation](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/lm_shakespeare/S1_MULTFREE_CHARLM_WALL.md).

### TT-NEG-004 — No present end-to-end energy win

Core-rail measurements, WNS-derived estimates, and projected replicas do not
support a board-to-board energy claim. Full-card FPGA power under a matched
quality/workload protocol remains open. Evidence: [hallmark
qualifications](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/lm_shakespeare/fig_hallmark_lm.md) and the [FPGA
evidence qualifications](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/EVIDENCE.md).

### TT-NEG-005 — Default AdamW can amplify compiled-gradient roundoff

On the audited local FP32 stack, ordinary native compiled autograd and packed
replay both fail the repeated-update parity check against eager AdamW, with
approximately 0.002 maximum parameter difference after two updates. Separate
momentum-SGD integration passes are not AdamW equivalence. The model/reverse path
is compiled; **the stock AdamW update itself is eager**. The new diagnostic
isolates initial-normalization mean-reduction rounding and cancellation near
zero gradients. Its explicit `eps=1e-4` recipe passes 20 fresh-batch updates
across three seeds and all three compiled paths at unchanged tolerances on the
small CPU fixture. This is a bounded mitigation, not a repaired default, a new
optimizer or evidence of task quality/GPU stability. Keep the failed default
visible; do not loosen tolerances to obtain a speed ranking.
Evidence: [optimizer diagnosis and record](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/EVIDENCE.md#adamw-numerical-diagnostic),
[original local codebase audit](../codebase_audit_2026-09-08.md) and the
[full-step benchmark](https://github.com/nyuolab/tikitorch/blob/main/scripts/route_certificates/training_benchmark.py).

### TT-NEG-006 — Packed CPU pilot is not a speedup; native selective support is scoped

The current ten-case, one-layer CPU pilot passes measured-final-state checks
but packed replay is not faster than the best compiled-checkpoint baseline
(`0.375` versus `0.349` ms median). Unpacked replay measures `0.265` ms. These
five fixed-order samples at one tiny shape are not task-scale speed, peak-memory
or quality evidence. A separate two-layer momentum-SGD compiled-selective worker
terminates with `SIGSEGV`; it is retained as a failed cell, not a denominator.
The cause is unresolved and the one-layer selective success remains valid.
Evidence: [checked pilot and native failure](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/EVIDENCE.md#current-cpu-comparator-pilot).

## Next claim gates, in order

1. **Archive:** finish and review the selected `main` release-candidate commit,
   including the route-certificate implementation, tests, scripts, raw rows,
   manuscripts, and this ledger; bind every result to that clean commit and an
   environment manifest. Complete the separate all-ref provenance/history gate
   before exposing the repository.
2. **Library:** local tests/docs/package checks passed in the implementation
   audit; obtain green remote candidate CI across the supported matrix and
   independent installed-artifact reproduction before authorizing a release.
3. **Practical route utility:** resolve or scope native AdamW support, complete
   the six-arm CUDA comparison, and measure full-cycle latency, retained bytes,
   peak and recomputation. Add multi-seed real-task quality/cost comparisons;
   traffic or energy requires separate measurement, not inference from bytes.
4. **FPGA learning:** preserve the completed 64-update D512 checkpoint-equivalence
   pilot. The varied-text CPU recipe still trails GRU in every tested order;
   close the quality gate before scaling, then independently validate any changed
   integer recipe in C-simulation and silicon. CPU readout diagnostics do not
   establish a deployable FPGA head. See the [current handoff](https://github.com/nyuolab/tikitorch/blob/main/manuscript/README.md#handoff-for-xiang).
5. **FPGA systems comparison:** close timing with margin; add iso-model CUDA and
   full-card power; then freeze the paper tables.
6. **External validity:** repeat fabric-only learning beyond downsampled MNIST
   before generalizing the learning claim.

No ontogenic-agent result is a gate for these TikiTorch publications.

The detailed acceptance criteria and implementation priorities are in the
[release/impact roadmap](../release_impact_roadmap.md). A narrower software
paper need not complete every hardware project unless its claims depend on it.
