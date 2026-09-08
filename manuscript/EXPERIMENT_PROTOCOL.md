# Proposed design for the remaining TikiTorch study

Version: draft protocol 1, 2026-09-08. This document specifies proposed
experiments, not measurements. Dataset, hardware/budget and candidate identity
must be filled and approved before claim-bearing runs. Pilot tuning and final
evaluation must be separated; never label choices made after viewing results
as preregistered.

## Questions and decision criteria

- **Representation:** what fields suffice for the selected reverse, and how
  does their actual storage scale? The source-bound three-cell CPU protocol in
  `measure_retained_state.py` already answers this narrowly.
- **Same-model systems utility:** at equivalent model outputs, selected
  gradients and repeated updates, does route storage improve the measured
  latency/memory/capacity frontier over strong eager/compiled/checkpoint arms?
- **Modeling utility:** does a hard tropical model reach useful held-out quality
  under a controlled training budget, and what resource cost achieves that
  quality compared with tuned dense and relevant low-arithmetic alternatives?
- **Closure:** which exact forward/loss/reverse/update recipes pass the
  no-waiver executed arithmetic audit? This is independent of task quality and
  physical synthesis.

Do not use an unrelated GPU model or a failed baseline as a denominator. A
compact byte record is not acceptance of the systems hypothesis. A synthetic
successor task is not acceptance of the modeling hypothesis.

## Freeze before final runs

Record the candidate source archive/hash, clean Git revision, environment,
hardware and driver/compiler identifiers, dataset license and byte hashes,
tokenizer/preprocessing, train/validation/test identities, optimizer and loss,
seeds, shapes, update counts, numerical tolerances and timeout policy. Archive
the exact command and deterministic reducer. A manifest must identify pilot,
tuning, validation-selection and final-test runs distinctly.

Use declared seeds **0, 1 and 2** for the initial multi-seed study and retain
every outcome. If more seeds are needed, append a new protocol version rather
than selecting favorable seeds. Choose numerical tolerances before examining
performance rankings. Failed route identity or nonfinite values always fail
equivalence; optimizer/dtype changes define new recipes.

## A. Same-model systems comparison

Use the existing `scripts/route_certificates/training_benchmark.py` numerical
paths. Begin on one existing authorized CUDA runner; no paid provisioning is
part of this protocol. Pilot compile/resource costs on the smallest shape
before authorizing a full matrix.

The initial proposed shape ladder is `(D,L,T) = (64,2,128), (128,4,512),
(256,4,2048)`, with batches 1, 4 and 16, fixed task vocabulary and all declared
seeds. Adjust this ladder only in a documented pilot revision before final
runs. Record actual OOMs and tested fits; do not extrapolate a maximum capacity.

Arms: eager autograd, native compiled autograd, checkpoint (per-layer, grouped,
selective where available), native compiled unpacked routes and native compiled
packed routes. The same model, initialization, input order, loss, optimizer,
precision and update count are mandatory. Fix warmups and repetitions after
the pilot, retain per-repetition values, separate compile/startup from steady
work and include packing, decoding and optimizer time.

The current checkpoint arms are eager. Before final rankings, implement and
validate compatible compiled-checkpoint policies, retaining explicit unsupported
cells where necessary. Instrument selective saving/recomputation on actual
executed operators, including the current `amax`/`amin` reductions; a policy
name is not proof of coverage. These are protocol prerequisites, not already
completed benchmark features.

Compare the **measured final state** after the full update count with eager,
not only a new one-step run. In the larger study, also checkpoint intermediate
losses/gradients/parameters to identify the first divergence. Test repeated
updates using fresh data batches in addition to fixed-batch timing. Resolve
the current AdamW numerical limitation, or label a momentum-SGD study as that
distinct configuration.

Primary systems reporting: complete-step latency median and dispersion;
allocated and reserved device peaks; maximum tested validated workload;
route bytes and transient workspace separately; compile time. Select the best
successful checkpoint policy within a specified allocator budget and show the
whole Pareto frontier. If instrumented, report measured memory traffic with
its profiler boundary; otherwise leave it unmeasured. Never infer board energy
from arithmetic counts or allocator metrics.

## B. Useful task quality

Choose a legally usable non-synthetic task with a fixed held-out split before
final evaluation. A corpus study needs distinct training, validation and test
partitions; the current convenience 85/15 runner is not a full model-selection
protocol. Freeze preprocessing/tokenization and remove known overlap or leakage.
Use validation only for hyperparameters, stopping and checkpoint selection;
evaluate the untouched test split once per selected seed/configuration.

Compare (i) the same hard model across differentiation implementations, to
isolate systems effects, and separately (ii) tuned architectures at controlled
budgets, to assess modeling utility. At minimum include a conventional dense
sequence baseline and one directly relevant morphological/low-arithmetic
baseline chosen from `LITERATURE_REVIEW.md`. Match/report parameter count,
training examples/tokens, update budget and tuning budget. Equal parameter count
alone is not matched quality. Compare quality/cost frontiers rather than
unrelated published throughput numbers.

Report each seed and uncertainty across seeds; do not treat correlated tokens
or timing repetitions as independent training runs. The primary held-out metric
and desired quality region must be fixed before final runs. Include no-update
and simple baseline controls. If the hard model fails to learn well, report
that and investigate before scaling hardware claims.

## C. Ablations that explain the result

1. Packed versus unpacked selected state; include decoding and workspace costs.
2. Several checkpoint policies, plus a hybrid route/checkpoint path when
   uncovered boundaries dominate.
3. Vocabulary and top-k head size, since `B*T*V*k*ceil(log2 D)` can dominate
   route storage even if the trunk is compact.
4. Depth, active-weight coverage, gradient density and route churn. A declared
   selected derivative does not imply adequate optimization coverage.
5. Hard-only versus soft-to-hard training, with separate audit/tape applicability
   for each phase; do not present the soft phase as activation-free or closed.
6. Loss/optimizer choice and closure status. Dense or smooth boundaries must
   be charged even when a hard trunk has a compact certificate.

Freeze the primary hypotheses and budget first; label secondary analyses as
exploratory. No outcome is assumed positive.

## D. Optional hardware extension

Retaining the current appendix as historical feasibility does not require a
new speed/energy claim. To add one, match the exact numeric recipe, model,
data order, update rule and state transition across reference and hardware;
resolve frozen MNIST evaluator/KAT differences and D512 resident-window limits.
Archive HLS and timing-closed routed reports plus bitstream/build identities.
Measure full-card idle/active power under synchronized workloads with
uncertainty. Host staging/readback and setup costs must be visible.

## Required final artifacts

Append-only raw rows and incident records; immutable source/environment/data
identities; deterministic tables/plots; all comparator configurations; explicit
unsupported cases; exact evidence/claim mapping; and an independent reproduction
report. A release publication or paper submission is a separate authorized
action after scientific and rights review.
