# TikiTorch research plan: useful learning on smaller hardware

Status: proposed research program, 2026-09-08. The immediate milestone is a
maintainer/student handoff, **not execution of these experiments**. No student,
compute allocation, spending limit, final evaluation budget or submission
schedule is assumed. Select an owner and approve a pilot protocol before
downloading research datasets, scheduling accelerator runs or building new
hardware images.

## The result we want

> At a useful level of task performance, TikiTorch makes learning cheaper or
> accessible on smaller hardware.

Test this with two software workloads—ECG classification and language-model
pretraining—followed by an optional FPGA capstone. Compact route records are
the mechanism, not themselves the practical result. A small saved-state
payload does not establish lower peak memory, shorter time to quality, better
task performance, lower energy or lower cost.

The initial software recipe may combine a route-trained tropical body with
ordinary PyTorch input/output layers, losses and optimizers. These boundaries
must be charged in all resource measurements. A useful hybrid result is valid
without a multiplier-free claim. A fully constrained learning recipe is a
separate, later goal rather than a reason to sacrifice useful learning early.

## Shared design and acceptance criteria

Keep two questions separate:

| Question | Comparison | Evidence needed |
|---|---|---|
| Does route differentiation improve execution? | The same model and training recipe with eager/compiled autograd, tuned checkpointing, and unpacked/packed routes | Selected-route and numerical equivalence; complete-step time, measured peak memory and tested capacity |
| Is that model useful? | Tuned tropical/hybrid and conventional architectures | Held-out quality versus total training time, memory and cost, with controlled data and tuning budgets |

The primary practical outcome is **time to a predeclared quality target under a
fixed device-memory budget**, with quality/cost frontiers alongside it. Choose
the quality target, acceptable quality difference, metric aggregation and
hardware budget using training/validation pilots, before final test evaluation.
Report configurations that never reach the target as failures to reach it;
retain their consumed budget rather than dropping them from averages.

A useful aspirational target is about **2× lower measured peak training memory
than compiled autograd**, while reaching competitive quality and improving
time to quality over the best compatible checkpoint policy at the constrained
budget. This is a target for investigation, not a forecast, required positive
outcome or a claim supported by the current payload table. A smaller but robust
capacity or cost advantage may still be worthwhile.

For all claim-bearing software comparisons:

- Freeze source/environment/data identities and train/validation/test splits;
  declare at least three training seeds and retain every result.
- Match initialization, input order, loss, optimizer, precision, effective
  batch size and updates for same-model systems comparisons. Fix numerical
  tolerances before ranking implementations. Different architectures need
  controlled tuning and quality/cost comparisons, not parameter-wise parity.
- Include compatible compiled checkpoint policies, gradient accumulation and
  equivalent chunking where applicable. Give conventional baselines the same
  optimized loss and memory-saving opportunities.
- Measure allocated/reserved device peaks, host/offloaded memory where used,
  packing/decoding workspace and full optimizer-step time. Separate compile
  startup from steady state, and report end-to-end training cost too.
- Charge all soft-to-hard warmup, preprocessing, tuning and failed runs to
  their declared budgets. Report a final hard phase separately only as a
  labeled component, never as the whole cost of pretraining.
- Record OOMs, timeouts, unsupported compiler cells, instability and negative
  findings. Do not estimate maximum capacity from configurations never run.
- Keep software closure reports separate from task quality and physical
  hardware reports. Neither payload bytes nor operation counts imply energy.

The [experiment protocol](EXPERIMENT_PROTOCOL.md) defines the common controls.
It must receive a task-specific, owner-approved revision before final runs.

## Track A: useful ECG classification

### Task and inexpensive first gate

Start with [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and its
five-label diagnostic-superclass task. Use the recommended patient-separated
folds (1–8 training, 9 validation, 10 test), freeze the annotation-to-label map,
and record which records qualify for this task. Do not randomly split windows
from the same patient across partitions. Confirm the selected release's data
terms and freeze record IDs and hashes before a claim-bearing study.

Begin with the 100-Hz recordings. Only advance to 500 Hz if a validation-only
pilot establishes a useful reason for the extra resolution. Fix preprocessing,
lead ordering, missing-value handling and normalization from training data
only. This is a retrospective research benchmark, not evidence of clinical
readiness or safety.

The task is multilabel. Use five independent logits with a suitable loss such
as `BCEWithLogitsLoss`, not the library's default single-label classification
cross-entropy. Any class weighting or decision thresholds must be selected on
training/validation data, not the test set. Propose macro-AUROC as the primary
quality metric and macro-AUPRC/per-class metrics as complementary reporting;
freeze the exact reduction and handling of missing labels in the protocol.

### Required implementation work, not existing functionality

1. Separate the route-trained body from supported input, pooling and output
   boundaries. The current exact-class MorphSSM route adapter supports a
   narrower language-model configuration; it does not yet cover this ECG
   classifier.
2. Implement and test a pooled classification path whose pooling reverse is
   explicitly specified. Verify gradients for both body and boundary parameters
   against ordinary autograd, including repeated optimizer updates.
3. Compare a conventional continuous input projection with a declared
   quantized-input alternative. The current default tropical input projection
   uses 32 bins over `[0, 1]` and does not provide a straight-through input
   derivative. Raw signed ECG values cannot simply be passed through that
   default without specifying scaling, clipping and quantization effects.
4. Report the actual receptive field and aggregation rule. A longer input
   tensor alone is not evidence that the model represents longer temporal
   dependencies.

### Baselines and decision gate

Use tuned conventional ECG networks, beginning with ResNet/InceptionTime
families from the [PTB-XL benchmark study](https://arxiv.org/abs/2004.13701),
alongside a simple sanity baseline and the same-model systems arms. Use the
same dataset version and split protocol; published scores from a different
pipeline are context, not a controlled denominator.

First establish that the tropical/hybrid model learns competitively in a small
CPU or approved existing-device pilot. Only then run the memory-constrained
study. A smaller conventional network, 100-Hz input or ordinary checkpointing
that reaches the same quality more cheaply counts against the practical claim.
Do not manufacture an advantage with unnecessary sequence length, resolution
or an oversized baseline.

Deliverable: a frozen multi-seed quality/time/memory comparison showing when
ECG learning becomes more accessible, or a documented negative result and a
reasoned stop/revision decision.

## Track B: language-model pretraining

### Development pilot, then natural text

Pretrain from scratch rather than measuring only a final hard fine-tuning phase.
The proposed ladder is deliberately gated:

1. Roughly **10–30M parameters on
   [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** to
   investigate optimization, route-state scaling and output-head cost. This
   is a small-language-model development milestone, not evidence of general
   LLM competence. Its synthetic corpus does not satisfy the natural-text
   evidence gate in the main protocol.
2. If quality and systems pilots pass, roughly **100–150M parameters on a
   frozen [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
   subset**. Select the exact dataset configuration/revision, tokenizer,
   deduplication/contamination controls and distinct training, validation and
   test partitions before running. Set token and tuning budgets after measuring
   pilot throughput on available, explicitly approved hardware.

These sizes are proposed scopes, not approved allocations or current
capabilities. Freeze parameter counting, sequence lengths, training tokens,
tokenizer vocabulary, optimizer/schedule and effective batch. Report held-out
token cross-entropy/perplexity with the same tokenizer. Cross-tokenizer
comparisons require a separately specified comparable metric, not raw
perplexity. Use a prespecified small downstream evaluation suite only at the
later gate, charging and documenting its selection separately.

### Solve the vocabulary-head bottleneck explicitly

The current top-k tropical vocabulary head retains route information scaling
as `B * T * V * k * ceil(log2(D))` bits before framing/alignment overhead.
For `B=1, D=512, T=2048, V=32768, k=8`, this is **576 MiB of logical route
payload for the head alone**. This is a shape calculation, not measured peak
memory, and shows why small-vocabulary evidence is insufficient.

Investigate two honest alternatives:

- **Practical first:** a route-trained body with a conventional linear readout
  and memory-efficient cross-entropy. Retaining a final hidden-state boundary
  can be preferable to retaining vocabulary-wide routes. Use an exact loss or
  explicitly declared numerical approximation; if using gradient-filtering
  optimizations, disable filtering for exact-comparison arms or label and
  separately validate the altered recipe. Give dense baselines the same loss
  implementation. [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) is a
  relevant design reference, not an already integrated dependency.
- **Fully tropical research path:** bounded/tiled readout evaluation with
  selective recomputation or another validated head representation. Charge
  all retained boundaries, recomputation and temporary workspace. Do not
  assume tiling alone removes the information retained for reverse execution.

The hybrid adapter, scalable head and compatible compiled-checkpoint controls
are prerequisites, not completed features. Do not silently relax the current
route adapter's checks to make an unsupported head appear covered.

### Baselines and decision gate

Compare same-model differentiation implementations, and separately a tuned
conventional decoder-only transformer and a relevant low-arithmetic baseline
from the [literature review](LITERATURE_REVIEW.md). Control token and tuning
budgets and show quality/cost frontiers; equal parameter counts alone do not
establish a fair practical comparison. Include smaller dense models when the
claim is access on smaller hardware.

Stop scaling if optimization quality is not competitive, the head dominates
memory, or improved checkpointing removes the useful advantage. Preserve all
hard-only/soft-to-hard phase costs and their distinct closure status. Existing
small historical language results do not establish matched-budget LLM
pretraining success.

Deliverable: reproducible time-to-quality and capacity evidence for pretraining
from scratch at a clearly stated scale, without presenting the pilot as a
billion-parameter or broadly capable LLM result.

## Track C: FPGA learning without GPUs or DSPs

This is a capstone after software usefulness, not a prerequisite for either
software paper result. Start with ECG because a small pooled output boundary
is a more bounded engineering target than a per-token language vocabulary head.

The target is a useful numerical recipe whose **forward computation, pooling
and head, loss, backward computation and parameter updates all execute on
FPGA**, with no GPU participating in those computations. A host may orchestrate
I/O and evaluation, but its role and cost must be explicit. Moving embedding,
head, loss, gradient formation or updates onto a CPU host does not meet this
complete-learning boundary. Existing ECG hardware is a starting point, not
evidence that this boundary has already been achieved.

Treat these as different checks:

- No data multiplications in the declared learning recipe: executed software
  audit plus applicable synthesis/operator evidence for the frozen numeric
  implementation, without undisclosed waivers.
- No DSPs used by that learning implementation: verified zero DSP utilization
  in its synthesized and routed hierarchy, with the platform shell and any
  out-of-boundary DSP resources separately disclosed. A multiplier-free design
  can still use a DSP as an adder; zero DSP use alone does not prove that no
  multiplication was implemented elsewhere in logic.
- No GPU used for learning: complete execution/host-boundary record, not merely
  absence of a GPU library import in the FPGA kernel.

Define the quantization, normalization, loss and optimizer recipe anew and
first reproduce its quality in software. Do not equate quantized training with
the floating-point route theorem. Require full-state one-step and repeated-step
known-answer tests, independent held-out evaluation, disjoint training examples,
timing-closed implementation reports and exact source/image/toolchain identities.
Only claim lower energy or cost after matched-workload, end-to-end measurements
with host/board boundaries, setup costs and uncertainty. Toolchain, board access,
build costs and artifact rights require explicit approval.

## Sequencing, ownership and artifact policy

1. **Now:** document the program, separate historical research artifacts from
   the reusable library, finish a bounded cleanup and offer the
   [student handoff](STUDENT_HANDOFF.md). Do not launch the studies.
2. **After an owner accepts:** independently reproduce onboarding and current
   correctness checks, review known numerical/provenance limits, choose a
   tractable first pilot and obtain hardware/data/budget approval.
3. **Shared engineering:** validate body/boundary interfaces and strong
   checkpoint controls. Run inexpensive ECG quality and LM readout pilots;
   parallelize only if ownership and resources permit.
4. **Claim-bearing software study:** freeze task protocols and execute the
   multi-seed quality/cost comparisons only for tracks that passed their gates.
5. **Optional capstone:** approve and execute the fully specified FPGA recipe
   after the software case is established. Revisit the manuscript's emphasis
   using actual outcomes, including negative results.

Store research evidence beneath [`manuscript/results/`](results/README.md),
separate from the importable library. Keep original logs/manifests immutable,
identify historical runs as historical, and produce derived tables with pinned
input hashes. Use append-only run IDs. Large datasets, model checkpoints and
board images belong in approved external storage with durable links and
checksums, not casual commits of binaries or credentials. Follow the
[evidence ledger](EVIDENCE_LEDGER.md) and
[release provenance gate](../docs/release_provenance.json).

The plan does not authorize publication, repository visibility changes,
history rewriting, cloud provisioning, dataset redistribution or submission.
Authorship, student supervision and target-venue decisions remain human decisions.
