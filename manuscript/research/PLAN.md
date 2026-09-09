# Research plan: useful learning on smaller hardware

Status: proposed program and draft comparison protocol, consolidated 2026-09-09
from the 2026-09-08 research plan and draft protocol 2. The immediate milestone
is the [contributor handoff](../README.md), **not execution of these studies**.
No student, compute allocation, spending limit, final evaluation budget or
submission schedule is assumed. Select an owner and approve a pilot protocol
before downloading research datasets, scheduling accelerator runs or building
new hardware images. Separate pilot tuning from final evaluation; never call
choices made after viewing results preregistered.

## The result we want

> At a useful level of task performance, TikiTorch makes learning cheaper or
> accessible on smaller hardware.

Test ECG classification first and language-model pretraining as a complementary
track, followed by an optional FPGA capstone. Compact route records are the
mechanism, not the practical result: saved-state bytes alone establish neither
lower peak memory, shorter time to quality, better quality, lower energy nor
lower cost. The existing source-bound three-cell CPU protocol in
`manuscript/tools/measure_retained_state.py` answers a narrow representation
question, not the remaining systems or modeling questions.

A useful first recipe may combine a route-trained tropical body with ordinary
PyTorch input/output layers, losses and optimizers. Charge these boundaries in
all resource measurements. A hybrid result is valid without a multiplier-free
claim; a fully constrained recipe is a separate later goal.

Keep these questions distinct:

| Question | Comparison or check | Evidence needed |
|---|---|---|
| Representation | Declared selected reverse and retained fields | Actual storage and scaling, including auxiliary state |
| Same-model systems utility | Eager/compiled autograd, strong checkpoint policies, unpacked/packed routes | Equivalent outputs, selected gradients and repeated updates; complete-step time, measured peak memory and tested capacity |
| Modeling utility | Tuned tropical/hybrid and conventional/low-arithmetic architectures | Held-out quality versus total time, memory and cost under controlled data and tuning budgets |
| Closure | Exact forward/loss/reverse/update recipe | No-waiver executed arithmetic audit, separate from quality and physical synthesis |

The primary practical outcome is **time to a predeclared quality target under
a fixed device-memory budget**, with quality/cost frontiers alongside it.
Select the quality target, acceptable difference, metric aggregation and
hardware budget from training/validation pilots before final test evaluation.
Report failures to reach the target, retaining their consumed budgets.

An aspirational target is about **2× lower measured peak training memory than
compiled autograd**, with competitive quality and better time to quality than
the best compatible checkpoint policy at the constrained budget. This is a
research target, not a forecast, required positive outcome or current result.
A smaller robust capacity/cost advantage may still be worthwhile. An unrelated
GPU model, failed baseline or synthetic successor task cannot establish it.

## Comparison protocol

### Freeze before final runs

Record the clean candidate Git revision and source archive/hash; environment,
hardware and driver/compiler identities; dataset license, byte hashes and
train/validation/test identities; tokenizer/preprocessing; loss, optimizer,
precision, seeds, shapes, update counts, numerical tolerances and timeout policy.
Archive exact commands and deterministic reducers. Distinguish pilot, tuning,
validation-selection and final-test runs in the manifest. Fill in the exact
dataset configuration, hardware/budget, owner and candidate identity in a
task-specific approved protocol revision before claim-bearing runs.

Use seeds **0, 1 and 2** for the initial multi-seed study and retain every
outcome. Additional seeds require an appended protocol version, not selection
of favorable seeds. Fix tolerances before performance rankings: failed route
identity or nonfinite values always fail equivalence, and optimizer/dtype
changes define new recipes. Preserve OOMs, timeouts, unsupported compiler cells,
instability and negative findings; report maximum *tested* capacity only.

### Same-model systems comparison

Start from `scripts/route_certificates/training_benchmark.py` on one existing
authorized CUDA runner, piloting the smallest shape's compile/resource costs
before a full matrix. No paid provisioning is part of this protocol.

The proposed development ladder is `(D,L,T) = (64,2,128), (128,4,512),
(256,4,2048)`, batches 1, 4 and 16, fixed task vocabulary and all declared seeds.
This is not the final ECG or pretraining configuration. Use justified task
resolutions/lengths and capacities; do not silently truncate ECG or shrink an
LM vocabulary to fit the ladder. Adjust it only in a documented pilot revision.

Compare eager autograd, native compiled autograd, checkpointing (per-layer,
grouped and selective where available), native compiled unpacked routes and
native compiled packed routes. Match model, initialization, input order, loss,
optimizer, precision, effective batch and update count. Fix warmups/repetitions
after the pilot and retain per-repetition values. Include packing, decoding and
optimizer time; separate compile/startup from steady-state work.

**Prerequisites, not completed features:** the current checkpoint arms are
eager. Implement and validate compatible compiled-checkpoint policies before
final rankings, preserving unsupported cells. Instrument actual selective
saving/recomputation, including the current `amax`/`amin` reductions; a policy
name does not prove coverage.

Compare the **measured final state** after all updates against eager, not only
a fresh one-step run. Also retain intermediate losses/gradients/parameters to
identify first divergence, and test repeated updates on fresh batches alongside
fixed-batch timings. Resolve or explicitly limit the current native compiled
AdamW discrepancy; a momentum-SGD study is a distinct configuration, not its fix.

Report complete-step latency median/dispersion, allocated and reserved device
peaks, maximum tested validated workload, route bytes, transient workspace and
compile time separately. Select the best successful checkpoint policy within
the declared allocator budget and show the whole Pareto frontier. Report
memory traffic only if measured, with the profiler boundary; neither arithmetic
counts nor allocator metrics establish energy.

Give all relevant arms compatible gradient accumulation, chunking, offloading,
mixed precision and optimized loss implementations. Report host/offloaded
memory too: moving storage is a trade-off, not elimination. Charge preprocessing,
tuning, failed runs and all soft-to-hard phases to their declared budgets.
End-to-end time includes compile/startup; a final hard phase may be reported
separately only as a labeled component, never as the whole pretraining cost.

### Useful task quality and selection

The final practical claim needs a legally usable **non-synthetic task** with
fixed held-out data. TinyStories and synthetic successor-token tests are
diagnostic, not fulfillment of this gate. A corpus needs distinct training,
validation and test partitions; the convenience runner's 85/15 split is not
a complete model-selection protocol. Freeze preprocessing/tokenization and
remove known overlap/leakage. Use validation for hyperparameters, stopping and
checkpoint selection; evaluate the untouched test split once per selected
seed/configuration, confirming the predeclared quality region.

Compare the same hard model across differentiation implementations to isolate
systems effects, and separately tuned architectures at controlled budgets to
assess modeling utility. Include a conventional dense sequence baseline and a
directly relevant morphological/low-arithmetic baseline selected from the
[literature review](LITERATURE.md), plus no-update and simple baseline controls.
Match/report parameters, examples/tokens, update and tuning budgets. Equal
parameter counts alone do not establish matched quality; unrelated published
throughput or scores are not controlled denominators.

Report each seed and uncertainty across seeds; correlated tokens or timing
repetitions are not independent training runs. If the hard model does not learn
well, preserve that result and investigate before scaling hardware claims.

### Ablations that explain the result

1. Packed versus unpacked selected state, charging decoding and workspace.
2. Several checkpoint policies, including a hybrid route/checkpoint path when
   uncovered boundaries dominate.
3. Vocabulary/top-k head size: `B*T*V*k*ceil(log2 D)` can dominate storage.
4. Depth, active-weight coverage, gradient density and route churn: a selected
   derivative does not guarantee adequate optimization coverage.
5. Hard-only versus soft-to-hard training, with each phase's separate audit/tape
   applicability; the soft phase is not activation-free or automatically closed.
6. Loss/optimizer choice and closure status, charging dense or smooth boundaries
   even when the hard body has a compact certificate.

Freeze primary hypotheses and budget first; label secondary analyses exploratory.
No outcome is assumed positive.

## Track A: useful ECG classification

### Task and inexpensive first gate

Use [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), five-label
diagnostic-superclass prediction, and recommended patient-separated folds
(1–8 train, 9 validation, 10 test). Freeze the annotation-to-label map, eligible
record IDs and hashes, and confirm the release's data terms. Never randomly
split windows from one patient across partitions.

Begin at 100 Hz; advance to 500 Hz only for a useful reason established by a
validation-only pilot. Fix preprocessing, lead order, missing-value handling
and normalization from training data only. This retrospective benchmark does
not establish clinical readiness or safety.

Use five independent logits with a multilabel loss such as `BCEWithLogitsLoss`,
not the library's default single-label cross-entropy. Select class weighting
and thresholds on training/validation data only. Proposed primary metric:
macro-AUROC, with macro-AUPRC and per-class metrics alongside it. Freeze exact
reductions and missing-label handling before final runs.

### Required implementation and decision gate

1. Separate the route-trained body from input, pooling and output boundaries.
   The current exact-class MorphSSM route adapter supports a narrower LM, not
   this ECG classifier.
2. Specify pooled-classification reverse semantics and test body/boundary
   parameter gradients against autograd, including repeated optimizer updates.
3. Compare continuous input projection with a declared quantized alternative.
   The default tropical input projection uses 32 bins over `[0, 1]`, without a
   straight-through input derivative. Raw signed ECG cannot simply use that
   default without defined scaling, clipping and quantization effects.
4. Report the actual receptive field and aggregation rule. Longer tensors
   alone do not establish longer represented temporal dependencies.

Tune conventional ECG networks, beginning with the ResNet/InceptionTime families
in the [PTB-XL benchmark study](https://arxiv.org/abs/2004.13701), alongside the
shared protocol's controls. Use the same release and split; published results
from different pipelines remain context.

First establish competitive learning in a small CPU or approved existing-device
pilot, then attempt the memory-constrained study. Smaller conventional networks,
100-Hz inputs or checkpointing that reach equal quality more cheaply count
against the claim. Do not manufacture an advantage through unnecessary length,
resolution or oversized baselines.

Deliverable: frozen multi-seed quality/time/memory evidence of more accessible
ECG learning, or a documented negative result and stop/revision decision.

## Track B: language-model pretraining

### Development pilot, then natural text

Pretrain from scratch; a final hard fine-tuning phase is not the whole result.
The proposed ladder is:

1. **10–30M parameters on
   [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** to
   investigate optimization, route scaling and head cost. This development
   milestone establishes neither general LLM competence nor natural-text success.
2. After quality/systems gates pass, **100–150M parameters on a frozen
   [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
   subset**. Select the exact configuration/revision, tokenizer, deduplication/
   contamination controls and separate train/validation/test partitions first.
   Set token/tuning budgets from pilot throughput on explicitly approved hardware.

These sizes are proposed scopes, not current capabilities or approved allocations.
Freeze parameter counting, sequence lengths, tokens, vocabulary, optimizer/schedule
and effective batch. Report held-out token cross-entropy/perplexity with the same
tokenizer; cross-tokenizer comparisons need a separately specified comparable
metric. A prespecified small downstream suite belongs only at the later gate,
with its selection and cost separately recorded.

### Solve the vocabulary-head bottleneck

The current tropical top-k head retains
`B * T * V * k * ceil(log2(D))` bits before framing/alignment. At
`B=1, D=512, T=2048, V=32768, k=8`, its head alone needs **576 MiB of logical
route payload**. This shape calculation is not measured peak memory and shows
why small-vocabulary evidence is insufficient.

- **Practical first:** route-trained body, conventional linear readout and
  memory-efficient cross-entropy. Retaining the final hidden-state boundary
  may beat vocabulary-wide routes. Use an exact loss or an explicitly declared
  numerical approximation. Disable gradient filtering for exact-comparison arms,
  or label and independently validate the changed recipe. Give dense baselines
  the same loss. [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) is a design
  reference, not an already integrated dependency.
- **Fully tropical path:** bounded/tiled readout with selective recomputation
  or another validated head representation. Charge boundaries, recomputation
  and workspace; tiling alone need not remove retained reverse information.

The hybrid adapter, scalable head and compiled-checkpoint controls remain
prerequisites. Do not relax route-adapter guards to imply unsupported coverage.

### Baselines and decision gate

Use same-model systems arms, and separately a tuned conventional decoder-only
transformer and relevant low-arithmetic baseline from the literature review.
Include smaller dense models for smaller-hardware claims. Control token/tuning
budgets and report quality/cost frontiers, not merely equal parameter counts.

Stop scaling if quality is uncompetitive, head memory dominates or checkpointing
removes the useful advantage. Preserve all hard-only/soft-to-hard phase costs
and their distinct closure status. Historical small-language results do not
establish matched-budget LLM pretraining success.

Deliverable: reproducible from-scratch time-to-quality/capacity evidence at a
stated scale, without presenting the pilot as a broadly capable or billion-
parameter LLM.

## Track C: optional FPGA learning capstone

Pursue this after software usefulness; it is not a prerequisite for either
software result. Start with ECG's smaller pooled output boundary rather than
a per-token language vocabulary head.

Target a useful recipe whose **forward, pooling/head, loss, backward and
parameter updates all execute on FPGA**, without a GPU performing learning.
A host may orchestrate I/O/evaluation, with explicit cost; offloading embedding,
head, loss, gradient formation or updates to the host does not satisfy this
complete-learning boundary. Existing ECG hardware is a starting point, not
evidence that the target is already achieved.

Verify these independently:

- **No data multiplication:** no-waiver executed software audit plus applicable
  synthesis/operator evidence for the frozen numeric implementation.
- **No DSPs in that learning implementation:** verified zero DSP utilization in
  its synthesized/routed hierarchy, separately disclosing shell/out-of-boundary
  resources. A multiplier-free design can use DSP adders; zero DSP use does not
  exclude multiplication implemented in logic.
- **No GPU learning:** a complete execution/host-boundary record, not simply
  absence of a GPU library import in a kernel.

Specify quantization, normalization, loss and optimizer anew and reproduce
quality in software first. Quantized training is not the floating-point route
theorem. Match the exact recipe, model, data order, update and state transition
between reference and hardware. Require full-state one-step/repeated-step
known-answer tests, disjoint training examples, independent held-out evaluation,
timing-closed reports and exact source/image/toolchain identities.

Historical appendix feasibility needs no new speed/energy claim. Expanding
those claims requires resolving frozen MNIST evaluator/KAT differences and
D512 resident-window limits, with archived HLS/routed reports and bitstream
identities. Measure full-card idle/active power under synchronized matched
workloads with uncertainty. Show host staging/readback, setup and build costs;
infer neither energy nor cost from operator counts. Toolchain, board access,
build costs and artifact rights require explicit approval.

## Sequence and required artifacts

1. **Now:** use the single [manuscript README/handoff](../README.md) for independent
   contributor review. Documentation consolidation does not launch studies.
2. **After an owner accepts:** reproduce onboarding/correctness checks, review
   numerical/provenance limits, and agree a bounded pilot and hardware/data/budget.
3. **Shared engineering:** validate body/boundary APIs and strong checkpoint
   controls, then inexpensive ECG quality and LM readout pilots. Parallelize only
   if ownership/resources permit.
4. **Claim-bearing software:** freeze task protocols and run multi-seed comparisons
   only for tracks that passed their gates; revisit manuscript emphasis using
   actual outcomes, including negative results.
5. **Optional capstone:** separately approve the complete FPGA recipe after the
   software case is established.

Store research evidence beneath [`manuscript/results/`](../results/README.md),
separate from the reusable library. Use append-only run IDs, raw rows and incident
records; keep historical logs/manifests immutable and labeled historical. Retain
source/environment/data identities, all comparator configurations and unsupported
cases, deterministic plots/tables with pinned input hashes, exact evidence/claim
mapping and an independent reproduction report.

Large datasets, checkpoints and board images belong in approved external storage
with durable links/checksums, not casual commits of binaries or credentials.
Follow [evidence preservation](EVIDENCE.md) and the
[release provenance gate](../../docs/release_provenance.json).

This plan does not authorize publication, visibility changes, history rewriting,
cloud provisioning, dataset redistribution or submission. Authorship, supervision
and venue decisions remain human decisions; release/submission requires separate
authorization after scientific and rights review.
