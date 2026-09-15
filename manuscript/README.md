# TikiTorch manuscript and handoff

**[Read the current manuscript (PDF)](TikiTorch.pdf)**

Status: **2026-09-15 — one ICLR-oriented working draft, not yet submission-ready.**
This is the shared starting point for students, Xiang, contributors and reviewers;
there is no separate handoff or readiness document to maintain.

*TikiTorch: Learning with Additive Hard Routing* starts with the mathematical
payoff: **additive hard selection turns backward into routing and accumulation**.
For `y_j = max_i(x_i + w_ji)`, the winner index and incoming gradient determine
both input and weight gradients; their original values are not needed. Forward
still evaluates the candidates. This property does not extend to a hard gate
around a dense expert, whose weight gradient still needs its input values.

TikiTorch connects the math to composable operators, layers and models,
device-resident route tapes and ordinary PyTorch training interfaces. The
practical question is whether this makes **useful learning cheaper or accessible
on smaller hardware**. The research priority is now **FPGA language-model
pretraining**: custom hardware can exploit the add/compare/select and routed
backward structure even if generic GPU execution shows no savings. That is a
hypothesis to test at useful held-out quality, not an established advantage.
The PyTorch interfaces remain the modeling and comparison layer.

## Handoff for Xiang

**Decision to make:** choose the next bounded quality experiment after reviewing
this readout-conditioning result. The route mathematics and small FPGA
execution are supported; useful language-model quality and a complete-system
efficiency advantage remain open. The strongest current development comparator
is the GRU. Hardware scaling still depends on closing the declared quality gate.

**Review package:** [current PDF](TikiTorch.pdf),
[evidence and negative results](research/EVIDENCE.md),
[frozen experimental protocols](research/PLAN.md), and the
[library claim ledger](../docs/research/CLAIM_LEDGER.md).
The PDF is a working draft; the newest development diagnostics are documented
here and in the evidence ledger, rather than incorporated into paper tables.

**Transfer requirement:** this is an uncommitted working checkout based on
`ea951fd`. Many experiment helpers, tests and public receipts are new files.
Raw checkpoints/caches under ignored `manuscript/build/` and the ignored corpus
are required to reproduce the diagnostic chain; a Git checkout or wheel alone
is insufficient. Preserve their hashes and saved declarations when arranging
artifact access. Runner receipts pin this machine's absolute paths, so moving
them requires a reviewed reproduction configuration rather than edits to old
receipts. Remote CI, a clean candidate commit, portable artifact packaging and
the separate rights/history review remain handoff tasks before release.

### Current results

Latest completed diagnostic: the [same-objective conditioned affine readout](research/EVIDENCE.md#same-objective-gram-preconditioned-readout-2026-09-15)
and its independent audit both complete successfully. It reaches the fixed
200-iteration cap with sampled-training/development CE **1.9788 / 2.6039**.
Conditioning improves fitting but worsens development CE against the prior
affine endpoint. It does not pass the numerical or quality gate.

| Fixed-cohort comparison | Sampled-training CE | Development CE |
| --- | ---: | ---: |
| Frozen-core head | 2.1534 | 2.5984 |
| Bias-only correction | 2.1495 | 2.5908 |
| Earlier affine correction, capped | 2.0612 | 2.5736 |
| Conditioned affine correction, capped | 1.9788 | 2.6039 |
| First-pass GRU control | 1.9480 | 2.3712 |

Lower CE is better. Both affine runs use the same frozen features, initialization,
regularization and iteration/call ceilings; actual solver work and wall time
are not matched. Readout fitting uses 1,024 training windows; the table scores
64 sampled-training and 64 repeatedly used development windows. GRU is an
inherited reference, not a readout-budget-matched refit. The conditioning run
improves the full-training regularized objective by **0.05856**, but stops with
an original-coordinate numerical gap estimate of **0.01504** against a target
of **0.00001**. That estimate is not a measured remaining loss gap or a formal
roundoff certificate. Near-optimality and the cause of the development deficit
remain unresolved. The affine head is a multiply-heavy CPU diagnostic, not an
FPGA-ready learning recipe. Review the fitting/development tradeoff before
choosing another experiment; no automatic extension follows the cap.

The [earlier frozen-feature comparison](research/EVIDENCE.md#frozen-feature-residual-readout-oracle-2026-09-15)
and its independent audit remain preserved, including the earlier affine
improvement and the bias-only numerical convergence.

Broader comparison: the [six-cell varied-text comparison](research/EVIDENCE.md#varied-text-exact-ce-development-comparison-2026-09-14)
has independently audited mean development CE **2.6211 tropical vs 2.3725 GRU
vs 2.6854 bigram**. There is improvement beyond the prior, but the paired GRU
quality gate fails in every order. All gradients, states and declared scores
pass verification. Each model/order used one pass over 1,024 training windows;
evaluation uses development windows 88–151 only. Reserved validation 152–863
and the final test remain untouched. No FPGA or paid run is launched.

The [audited frozen-endpoint check](research/EVIDENCE.md#frozen-endpoint-fitting-diagnostic-2026-09-14)
finds most of the gap already on a fixed training subset: **2.1661 tropical vs
1.9469 GRU**, a 0.2191-nat difference versus 0.2485 on development. This is a
fitting deficit at the current budget, not just a development-only problem;
optimization, conditioning and representation limits remain to be separated.
The check made no updates or new development evaluations. The latest combined
regression coverage totals **5,156 passing tests** across the default run and
successful TCP rerun (14 skipped, 10 deselected). The default sandbox run had
5,153 passes and three TCP failures caused by blocked loopback sockets; all
eight TCP tests then passed outside that restriction. This includes 92
routing-contrast, 87 readout-oracle and 82 conditioning tests.

The [audited extra-pass control](research/EVIDENCE.md#paired-order-0-adam-continuation-2026-09-14)
improves both models but widens the training gap from 0.2179 to 0.3019 nat and
development gap from 0.2372 to 0.3069. Observed channel support stays fixed while
temporal routes adapt. More updates alone do not catch the GRU at this budget.

The [audited core-bias control](research/EVIDENCE.md#frozen-core-bias-control-2026-09-14)
starts again from the original initialization and freezes projection weights
and core biases while training embeddings, temporal weights and head weights.
It modestly improves sampled-training CE **2.1660 → 2.1534** and development
CE **2.6083 → 2.5984**; the paired GRU remains at **1.9480 / 2.3712**.
Development accuracy is slightly lower despite the CE improvement. Neither
single-order control closes the gap or proves its cause. The
[audited two-candidate diagnostic](research/EVIDENCE.md#two-candidate-channel-routing-control-2026-09-14)
adds one channel alternative per projection row and requires initial-function and
first-gradient parity, plus an explicit same-checkpoint activation check.
Its implementation has 79 passing synthetic tests, and the frozen-source,
input-only admission check passes. The single real-data producer and its
independent numerical audit both completed successfully. Both choices become
active within the final checkpoint at **2,031 / 2,048** development layer/row
pairs, but CE is
**2.1869 sampled training / 2.6083 development**, worse than the frozen-core
control. Activating competition with this fixed-offset recipe did not improve
quality; that does not prove a capacity limit
or identify the cause. The [completed three-arm follow-up](research/EVIDENCE.md#three-arm-learned-routing-contrast-block-2026-09-15)
tests learned relative scores, a ten-times-smaller relative-score learning
rate, and an additional learned pair-mean coordinate. All three producers and
independent numerical audits finish successfully. None improves on frozen-core:
sampled-training/development CE is **2.1973 / 2.6147**, **2.1857 / 2.6203**,
and **2.1970 / 2.6161**, respectively. The slower update trades better fitting
against worse development loss; learning the pair mean does not recover the
deficit. These controls change output values and physical update scales as
well as routes. They neither establish a capacity limit nor justify scaling.

The earlier [512-position fitting check](research/EVIDENCE.md#exact-ce-rate-only-comparison-v1)
reaches tropical CE **0.124 / 100% accuracy**, versus GRU **0.172 / 97.3%**.
Its independent audit and [byte-exact guarded-path replay](research/EVIDENCE.md#guarded-full-trajectory-parity-2026-09-13)
both pass; the two older failed audit attempts remain preserved. That closes
the tiny fitting gap, not the broader quality gap above. Untouched-cohort
confirmation and an implementable low-arithmetic optimizer still precede FPGA
scaling.

## What is done

- The two papers are combined into one narrative with theory, methods, references
  and qualified historical FPGA results. Superseded scientific sources are
  byte-preserved in [archive/](archive/README.md), not competing drafts.
- The ICLR narrative leads with a worked gradient example and the library stack;
  the draft states the joint state/parameter reverse contract, local tie rules,
  proofs and counterexamples. Selected local derivatives are explicitly **not**
  guaranteed to be global Clarke subgradients; the coding bound is conditional.
- The library has a narrow, tested device-tape/PyTorch route-training path,
  fail-closed arithmetic audits and a tiny runnable onboarding workflow.
- Source-snapshotted CPU measurements check logits, routes and parameter VJPs. The
  retained-state ratios are **2.04–2.98×** against a shape-defined checkpoint
  boundary; these are representation measurements, **not measured peak memory,
  training speed, energy or useful-task quality improvements**.
- The selected CPU measurements are preserved with their original 71 source
  files. `python -m manuscript.tools.historical_cpu_evidence --check` verifies
  them using their original strict checkers; it does not rerun experiments or
  claim freshness against subsequently added library code.
- Tables are derived from checked records. The PDF build verifies evidence,
  pinned venue styles, references and main-body pagination before refreshing the
  review copy. Historical measurements remain separate and append-only.
- AdamW divergence is traced to upstream reduction/cancellation roundoff amplified
  by its default epsilon, not a compiled optimizer implementation. Explicit
  `eps=1e-4` passes the bounded 20-update, three-seed CPU diagnostic across compiled
  autograd and packed/unpacked paths; the default failure remains recorded.
  See [optimizer evidence](research/EVIDENCE.md#adamw-numerical-diagnostic).
- The same explicit recipe also passes a width-32, two-layer, length-24
  three-seed follow-up. The comparator now includes compiled checkpointing and
  observed selective-policy coverage, with scoped CPU pilot evidence. These
  checks do not establish task-scale learning or accelerator support.
- The new fixed-point FPGA workflow passes local native-layout, checked-PCI
  fault-injection and independent frozen-reference tests. Four local CSIM
  pilots are recorded, including D512 training on 1,024 fresh windows. Validation
  LUT NLL improves **4.8520 → 3.4087**, but removing temporal context leaves its
  final metrics unchanged and simple whole-training-partition baselines remain
  stronger. These CPU records do not establish useful contextual learning;
  see the [development results](research/EVIDENCE.md#fresh-window-fpga-lm-development-2026-09-09).
  A checked follow-up identifies shared head winners and exactly cancelled
  head-input gradients at **92.68%** of the inspected validation positions.
  The subsequent head-repair screen below addresses this initialization problem.
- A current-code physical FPGA pilot now completes **64 fresh-window updates**
  (8,192 token exposures) on the 1,196,032-parameter character model. All full
  weight payloads match C-simulation at updates **0, 32 and 64**; the final
  checkpoint also matches the earlier local reference. This is an audited
  hardware-correctness milestone, not useful LLM pretraining or a cost advantage.
  See the [paired pilot](research/EVIDENCE.md#current-code-fpga-checkpoint-equivalence-pilot).
- A twelve-run CPU/C-simulation screen now demonstrates **context-sensitive
  learning across three seeds** after changing head initialization and LUT scale.
  The anchored head with finer LUT reaches validation NLL **2.708–2.741**,
  beating the matched-data unigram (**3.225**) but not the bigram (**2.621**).
  Removing past temporal taps worsens its NLL by **0.184–0.328** nats/token.
  Both anchored recipes pass the prespecified gradient-flow/context gates;
  no arm passes the larger-paid-learning gate. These are host-staged changes
  compatible with the existing kernel interface, **not yet revalidated on FPGA**.
  See the [screen results and limitations](research/EVIDENCE.md#head-routing-repair-screen-v1).
- The declared six-run, **1,024-window CPU extension** also completes. On its
  new, fixed validation cohort, coarse-scale mean NLL improves **3.0635 → 2.7366**
  from updates256 to1,024. Fine-scale accuracy improves, but mean NLL worsens
  **2.8045 → 2.9006**, with more extremely low-probability targets. Both recipes
  still fail the matched-bigram loss gate (**2.5933**). This is a useful negative
  result: repaired context use alone does not establish a competitive language
  model. See the [budget-extension evidence](research/EVIDENCE.md#cpu-learning-budget-extension-v1).
- A 60-cell **fixed-checkpoint confidence diagnostic** confirms a growing severe
  error tail, but the tested evaluation rescalings do not close the loss gap.
  All original-shift outputs match saved native arrays; weights and historical
  gates are unchanged. This rules out a simple fix within the tested five-scale
  grid, not every possible calibration or training improvement. See the
  [confidence results](research/EVIDENCE.md#fixed-checkpoint-confidence-diagnostic).
- The intermediate-scale follow-up completes all 12 cells. Its best mean
  (**2.78355**) still trails matched bigram (**2.66015**) on the shared new
  cohort. A separate native-checked construction represents that bigram within
  **0.00031 nats**, so quantization alone does not explain the training gap.
  A separately versioned, magnitude-preserving integer optimizer is now tested;
  see the [scale results](research/EVIDENCE.md#intermediate-training-scales-v1)
  and [representability control](research/EVIDENCE.md#bigram-representability-control).
- The completed integer error-feedback study has a promising **count-fitted
  k18** candidate: NLL **2.6184** versus stronger bigram **2.6602**, with **0.0530**
  nats of temporal-context benefit. All four cells and full optimizer states
  are audited. This is a host-initialized, one-seed CPU result; from-initialization
  runs still trail bigram. See the [results and exposure accounting](research/EVIDENCE.md#integer-error-feedback-learning-v1).
- Three training-order replicates on a fresh development slice all beat bigram
  (mean advantage **0.0494 nats**), but one misses the fixed context threshold.
  The combined scaling gate therefore **fails**. See the
  [confirmation and limitations](research/EVIDENCE.md#error-feedback-training-order-confirmation-v1).
- The exact fitted-start SignSGD control also completes: final NLL **2.8248**
  versus EF's **2.6184**, with every native weight update checked. That supports
  this specific optimizer choice, not universal EF superiority. See the
  [matched control](research/EVIDENCE.md#matched-fitted-start-optimizer-control-v1).
- A forward-preserving projection initializer now completes three new runs and
  three matched historical controls. It passes the unchanged context/bigram
  screen, but has **worse NLL in every paired order**: mean **2.6113** versus
  historical **2.5793** and stronger bigram **2.6353**. One new run slightly
  trails bigram. This is a context-reliability/quality trade-off, not a better
  learning recipe. All 18 full optimizer states and the evidence package pass
  the [comparison audits](research/EVIDENCE.md#forward-preserving-projection-comparison-v1).
- The [learning-contract diagnostic](research/EVIDENCE.md#learning-contract-diagnostic-v1)
  now establishes contextual FP64 toy learning and exact independent/native
  gradient and full-state agreement in six reset D512 probes. Near-teacher
  training and synthetic-evaluation KL fall by over 99.5%; a structured-cold
  control also learns. Integer weight rounding suppresses immediate descent
  much more than the tested LUT approximation; EF retains the deferred updates.
  This identifies a precision/step-size question, not the cause of the full
  validation gap or a successful new LM recipe. No AWS resources or held-out
  corpus windows are used.
- The [persistent-trajectory diagnostic](research/EVIDENCE.md#persistent-trajectory-diagnostic-v1)
  completes four 32-update trajectories and 120 later counterfactuals. Every
  native update and saved-gradient state replay agrees. On the three fitted
  training probes, removing weight rounding is only 0.000528 CE better with
  historical initialization but 0.008131 worse with the diverse initialization.
  No tested own-window update shows ascent. More precision alone is therefore
  not a demonstrated repair; the update-scale comparison below tests a separate
  intervention.
- The [paired update-scale screen](research/EVIDENCE.md#paired-update-scale-screen-v1)
  completes six new 1,024-update runs with six unchanged historical controls.
  For the primary diverse initialization, fourfold-smaller updates improve mean
  NLL by only **0.002858 nat**, regress in two of three orders and miss the
  required bigram margin. The recipe is **not promoted**. Independent replay
  matches all 896 evaluation windows and twelve full-gradient/native-state
  anchors. Reserved development data and the final test remain untouched.
- The verified arithmetic is integrated as the experimental public CPU primitive
  `tt.optim.fixed_point_ef_step`, with checked overflow, exact persistent carry
  and a runnable `python -m examples.fixed_point_workflow` example. It is distinct
  from `EFTropicalSGD`; no optimizer defaults or hardware guarantees change.
- The [four-window accumulation diagnostic](research/EVIDENCE.md#four-window-gradient-accumulation-v1)
  is complete and negative. At the same per-example scale, final training CE
  worsens to **5.376293**, versus sequential **2.409066**; batch4 is worse on all
  32 final windows under both CE and LUT NLL. Independent replay matches every
  full gradient and score readout. Arithmetic is intact; batching does not repair
  this workload. No held-out data, new native training or cloud resource is used.
- The [first-failing-block fraction probe](research/EVIDENCE.md#first-failing-batch-step-fractions-v1)
  supports a local oversized-step explanation. Dynamic CE is **2.595076** with
  no update, **2.519917** with a quarter fresh-gradient contribution, and
  **3.261983** with the full contribution. The full step also fails in the
  unrounded fixed-affine view; reduced fractions improve every tested window
  and view. All candidates are uncommitted: this does not establish a stable
  training recipe or promote the earlier failed k20 screen.
- The [normalized-accumulation comparison](research/EVIDENCE.md#normalized-accumulation-order-diagnostic-v1)
  finishes nine short trajectories on the same 32 training windows. Batching
  avoids the previous explosion, but mean CE is **2.434009**, versus matched-scale
  sequential **2.432145** and original-scale sequential **2.412051**. It regresses
  against the matched-scale control in two orders and fails its numerical gate.
  Completion checks and independent replay of all **288 full gradients and
  434 readouts** pass. No recipe is promoted.
- The [relative head-weighting comparison](research/EVIDENCE.md#relative-head-weighting-diagnostic-v1)
  improves mean training CE by **0.0169 nat** with uniform token weighting,
  but regresses in one of three orders and fails its consistency gate. All
  **192 gradients, 198 states and 302 readouts** pass independent replay.
- The [exact-CE fitting comparison](research/EVIDENCE.md#exact-ce-fitted-start-topology-comparison-v1)
  and its full independent audit pass: after 128 updates on the same 512 training positions,
  CE is **1.532 original tropical / 1.427 cross-class tropical / 0.172 GRU**,
  from identical initial probabilities. All beat the empirical current-token
  floor, and cross-class wiring helps by 0.106 nat, but a large GRU gap remained
  at rate 0.001. Tropical learning had not plateaued; the completed
  [rate-only diagnostic](research/PLAN.md#exact-ce-rate-only-comparison-v1-declared-2026-09-11)
  below closes that local gap with tenfold-larger Adam steps and the same model/budget.
  These are local training fits, not a repaired LM or an FPGA-ready recipe.
- The [rate-only follow-up](research/EVIDENCE.md#exact-ce-rate-only-comparison-v1)
  completes its producer at **0.124361 CE / 100% training accuracy**, below the
  frozen GRU's 0.172225. Only Adam's rate changes; initial states and scores
  match exactly, and head/projection supports never switch. Both local-fitting
  thresholds and the full independent numerical audit **pass** under the new
  pinned-predecessor protocol. All 512 window VJPs, 128 batch gradients, 129
  complete Adam states and 12 scores agree. Both earlier
  [failure receipts](research/EVIDENCE.md#exact-ce-rate-only-comparison-v1)
  remain failures, distinct from the successful 11-minute audit. No recipe is
  promoted, and broader generalization and
  hardware suitability remain untested.
- The [guarded implementation replay](research/EVIDENCE.md#guarded-full-trajectory-parity-2026-09-13)
  passes **all 524 forward traces, 128 gradients and 129 complete Adam states
  byte-for-byte**. Every covered projection/head operation certifies, with zero
  dense fallbacks. The complete bounded invocation takes 114 seconds, retaining
  full parameters and optimizer state. This establishes implementation parity
  and local replay feasibility, not a matched speedup, memory saving or FPGA result.

Sharing-pass validation (2026-09-13): **4,428 regression checks pass,
14 skipped, 10 slow/native tests deselected**, plus **35 separately run guarded-
verifier cases**. The full socket-enabled regression run excludes only that
then-in-progress new test module and takes 435.59 seconds with one expected
backend-fallback warning; the final new module passes in 10.11 seconds. These
are two invocations, not a claimed single 4,463-case run. No timing threshold was
loosened or skip added. Strict documentation, artifact-layout and PDF freshness
checks pass. An offline source-distribution-to-wheel build passes the existing
distribution-layout checks and an isolated public-API import/workflow check.
Scientific quality remains a separate gate; these checks do not authorize scaling
or establish hardware performance.
The public integer primitive, carry/resume example, update-scale runner and
independent auditor add **236 focused checks**, covering signed arithmetic,
overflow, exact state conservation, provenance and failure lifecycle. They do
not run the corpus study as part of the default test suite.
The matched accumulation helper and diagnostic add a further **214 tests**,
including full tiny-model query/readout replay and exact artifact membership.
The uncommitted fraction helper and first-failing-block runner add another
**175 tests**, including all five real tiny-model readout views, exact carry
units, parent-state controls and failure-safe publication. Both complete
scientific runs and their separately bounded readout replays pass; the default
test suite itself does not execute either corpus diagnostic.
The normalized-accumulation runner adds **125 tests**, including all nine
tiny-model cells, 216 direct integer state recurrences, 288 independently
recomputed gradients, 434 readouts, fixed-k carry releases, shuffled exposure
accounting and failure-safe publication. This synthetic test result is separate
from the declared corpus comparison and its separately bounded audit.
The head-weighting diagnostic adds **112 tests**; the topology primitive and
exact-CE/GRU comparison add **86**. They include every tiny-model Adam update
checked against a separate formula, independent model/gradient replay, causal
route-forgery rejection and nested-deadline preservation. The combined focused
suite has 198 passing checks; these do not run either corpus experiment.
The separate rate-control runner adds **41 tests**, bringing that focused
combination to **239 passes**. The guarded-forward prototype adds
**42 synthetic parity/fallback tests**. Its new full-trajectory verifier has
**35 tests**, and the pinned-predecessor independent rate auditor has **28**.
The original dense training producers remain unchanged. These synthetic tests
do not run the corpus studies or establish a speed advantage.
The learning-contract reference, numeric helpers and runner add 237 synthetic
checks covering independent derivatives, arithmetic, provenance and failure
lifecycle; their separate declared experiment is reported above.
The persistent-trajectory reference, arithmetic and harness add 249 further
synthetic checks, including full saved-gradient/checkpoint replay and rejected
completion-record handling. Neither test suite runs the declared corpus study.
The new scale/capacity helpers pass 213 focused checks; the sign diagnostic and
native optimizer pass 74; the error-feedback study runner passes 68 and the
confirmation harness passes 123; the matched optimizer control passes 129;
the projection initializer and study harness pass 169. Native
tests include full D512 weight/remainder parity for two synthetic updates,
state-preserving evaluation/resume, and isolated overflow/clipping failures.
The new budget-extension, audit and confidence helpers pass **188 focused
checks**, including scalar LUT arithmetic, native-array comparisons and
artifact/claim-boundary rejection cases; these are included in the full total.
The separate finalized
head-screen/observer/auditor suite passes **143 checks**, including one isolated
native test covering six configurations and every parameter-gradient word.
The new isolated native
fresh-window test also passes. The separate native-compiler optimizer diagnostic
suite passed all 22 tests, including the three-seed, 20-update CPU checks.
The native comparator suite passed 46 tests with one explicit optional-policy
skip for the reproduced two-layer selective-checkpoint crash.
Lint, strict documentation and artifact/PDF freshness checks pass. The earlier
successful PDF build remains current; this pass does not add these development
results to the paper or rebuild it. This is not remote CI, independent scientific reproduction or
GPU validation.
Cloud pagination changes also pass their focused lifecycle tests;
see the integration records for what has actually been exercised on AWS.

## What needs to happen next

1. **Review the fitting/development tradeoff and choose the next bounded
   quality experiment.** The varied-text tropical model beats bigram on mean
   development CE but trails the matched-data GRU in all three orders. Extra
   training, added channel competition and learned routing contrasts do not
   close that gap. The frozen-feature affine readout improves the frozen-core
   endpoint, but its capped solve leaves near-optimality unknown. The completed
   [same-objective conditioning experiment](research/PLAN.md#same-objective-gram-preconditioned-readout-v1-declared-2026-09-15-utc)
   improves fitting of the same objective but worsens development CE, and its
   capped endpoint still misses the numerical target. Use its audited result
   to choose a bounded follow-up; do not extend a capped
   solve automatically or infer a backbone capacity limit from it. These CPU
   FP64/Adam and affine-readout diagnostics are distinct from the deployed
   integer/LUT/SignSGD FPGA recipe. An eligible hardware recipe must independently
   pass quality, complete-state C-simulation and silicon equivalence gates.
   Reserve validation windows 152–863 and the final test for a separately frozen
   confirmation protocol. The repeatedly inspected development slice is tuning
   evidence. Earlier controls and negative results remain in the
   [evidence ledger](research/EVIDENCE.md).
2. **Reproduce and profile the complete FPGA system.** The
   owner has authorized AWS spending below **$1,000 cumulative** for this
   continuation. Start with a $10 envelope, not the whole allocation. The
   [AWS helper](../scripts/fpga_lm_aws.py) supports offline preparation and
   explicit read-only preflight; its direct launch remains disabled. The
   [separate controller adapter](../scripts/fpga_lm_cloud.py) requires explicit
   execution, a durable reservation and a prearmed AWS termination watchdog.
   Pilot G passed native build, image checks, training, checkpoint comparison
   and artifact retrieval after explicit source-upload approval. Earlier
   integration failures are retained. **$60 of reservations remain counted**,
   with actual invoicing pending. All five instances launched for these pilots
   are confirmed terminated, with disposable disks/interfaces absent; the old
   historical sentinel remains untouched. Preserve these failures and every
   complete reservation in the [integration evidence](research/EVIDENCE.md#live-cloud-integration-failures-are-evidence-not-training-results);
   repeat the paired gate after any model/numeric change. Measure cold setup,
   training, evaluation, transfers and costs separately; runner timing is not
   cloud end-to-end timing or an optimized CPU/GPU comparison.
   See the [research plan](research/PLAN.md) for controls and integration status.
3. **Scale only after correctness and learning gates.** Compare held-out
   learning with the initial/no-update model, train-fitted unigram/bigram and
   context-removal controls; preserve negative results and leave test data
   untouched. The current D512 design nearly fills its on-chip memory and does
   not use external HBM/DDR for model state. Larger models require a measured
   memory/compute redesign, not simply longer runs or independent replicas.
   Measure end-to-end cost and quality against credible conventional baselines;
   zero DSPs alone is not an efficiency result. GPU route-training benchmarks
   and ECG remain secondary opportunities, not prerequisites for this pilot.
4. **Freeze and independently reproduce the candidate.** Archive exact source,
   environment and commands, rerun append-only evidence, regenerate the PDF and
   resolve the human/distribution gates below. A source hash alone is not an
   independent reproduction or a tagged release.

For a student or collaborator, start with the checks below and agree on an owner,
reviewer, deliverable and stop condition. Before research runs, confirm dataset
terms/splits, permitted downloads, hardware and runtime/storage/spending limits.
The previous local optimizer diagnostic did not authorize paid compute; the
new, bounded AWS authorization and first-pilot controls are recorded in the
research plan. They do not authorize unrelated account changes or spending
above the cumulative limit.

The [research plan and comparison protocol](research/PLAN.md) specify the seeds,
shape ladder, baselines, quality targets, ablations, pilot/final-test separation
and hardware controls. Agree on tuning budgets and retention before measuring;
do not choose final-test targets after seeing results. A second supported
architecture needs contracts and equivalence tests, not removal of adapter guards.

## First contribution

Use a fresh checkout and virtual environment following [Contributing](../CONTRIBUTING.md)
and the [test guide](../tests/README.md). From the repository root, without a GPU,
FPGA, dataset download or LaTeX installation:

```bash
python -m examples.closure_workflow
python -m pytest -q tests/test_route_training.py tests/test_morphssm_tape_parity.py
python scripts/check_artifact_layout.py
python manuscript/build.py --check
```

Then run the standard lint and full test suite. Share commit/worktree state,
Python/PyTorch/platform versions, actual pass/skip/fail results and one scoped
issue or PR. Keep environment limitations and skip reasons visible. Do not
widen tolerances or rewrite evidence hashes to make a check pass; a source-bound
mismatch needs review and an append-only remeasurement, as described in
[the evidence policy](research/EVIDENCE.md).

## Reproduce the local head-routing screen

With the pinned corpus already present at `data/tinyshakespeare.txt`, NumPy and
a C++17 compiler, run the fixed CPU/C-simulation protocol in a **new** directory:

```bash
python manuscript/tools/screen_fpga_lm_heads.py \
  --corpus data/tinyshakespeare.txt \
  --output manuscript/build/head-screen-reproduction
python manuscript/tools/audit_fpga_lm_head_screen.py \
  --run manuscript/build/head-screen-reproduction \
  --output manuscript/build/head-screen-reproduction-audit.json
```

The screen fixes four arms, three seeds, 256 fresh windows per run, and separate
initial/final validation checks. Each worker has a 600-second limit. It builds
the unchanged native kernel locally, retains full checkpoints and native output
arrays, and fits unigram/bigram controls to exactly the same supervised pairs.
The audit recomputes validation outputs, context controls and route diagnostics
from those checkpoints. It does not independently replay every training step.
Existing output paths are rejected; keep unsuccessful runs and use new paths.

No AWS credentials, FPGA, GPU, download or final-test evaluation is involved.
Large binary artifacts stay in ignored `manuscript/build/`; the compact audit
record can be shared separately. A fresh checkout needs the recorded corpus and
a new execution, or access to the original binary artifacts. These are
development diagnostics, not a useful-LLM or hardware-efficiency benchmark.

The versioned learning-budget extension uses a completed v1 run as its
checkpoint reference. It restarts both anchored recipes from initialization,
trains 1,024 fresh windows per seed, and evaluates a new validation cohort:

```bash
python manuscript/tools/extend_fpga_lm_budget.py \
  --corpus data/tinyshakespeare.txt \
  --v1-run manuscript/build/head-screen-reproduction \
  --output manuscript/build/budget-extension-reproduction
python manuscript/tools/audit_fpga_lm_budget.py \
  --run manuscript/build/budget-extension-reproduction \
  --output manuscript/build/budget-extension-reproduction-audit.json
```

All six workers have a 900-second limit. Update 256 must reproduce the actual
v1 checkpoint bytes; controls are refitted separately at updates 256 and 1,024
on exactly the supervised data seen so far. Initialization is a no-update
control, not a comparison with a fitted baseline. Preserve the original v1
artifact directory and sources: the extension verifies those references as
well as its own outputs, and rejects existing output paths.

After the extension audit succeeds, reproduce the separate **post-hoc,
evaluation-only** confidence grid:

```bash
python manuscript/tools/diagnose_fpga_lm_calibration.py \
  --run manuscript/build/budget-extension-reproduction \
  --audit-record manuscript/build/budget-extension-reproduction-audit.json \
  --output manuscript/build/calibration-reproduction.json
```

This retains all 60 combinations of trained recipe, seed, checkpoint and
evaluation LUT shift. Original-shift outputs are checked against saved native
arrays; alternative shifts use fixed host-recomputed logits. No model is
retrained, no gate is changed, and no scale is selected by the tool. The same
development validation data are reused, not a new generalization test.

The completed intermediate training study and independent checkpoint audit use
the same immutable budget extension as controls:

```bash
python manuscript/tools/train_fpga_lm_scales.py \
  --corpus data/tinyshakespeare.txt \
  --budget-run manuscript/build/budget-extension-reproduction \
  --budget-audit manuscript/build/budget-extension-reproduction-audit.json \
  --output manuscript/build/scales-reproduction
python manuscript/tools/audit_fpga_lm_scales.py \
  --run manuscript/build/scales-reproduction \
  --output manuscript/build/scales-reproduction-audit.json
python manuscript/tools/check_fpga_lm_bigram_capacity.py \
  --corpus data/tinyshakespeare.txt \
  --output manuscript/build/capacity-reproduction \
  --record manuscript/build/capacity-reproduction-record.json
```

The capacity run fits counts on the host; it performs zero gradient updates.
All paths must be new, and evidence records must sit outside raw run trees.
The separate error-feedback experiment requires a local C++17 compiler with
undefined-behavior sanitization, no AWS credentials or FPGA image:

```bash
python manuscript/tools/train_fpga_lm_efsgd.py \
  --corpus data/tinyshakespeare.txt \
  --output manuscript/build/efsgd-reproduction \
  --record manuscript/build/efsgd-reproduction-record.json
```

Follow its [scope and data accounting](research/PLAN.md#integer-optimizer-repair-v1-completed).
This is a CPU research adapter with a **new** optimizer/state ABI. It cannot
be substituted into the existing physical-pilot runner or deployed FPGA image.

The fixed confirmation requires that complete four-cell run and its audited
record. It imports the same fitted initial state, shuffles training order with
three declared seeds, and evaluates a new development cohort:

```bash
python manuscript/tools/confirm_fpga_lm_efsgd.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/efsgd-reproduction \
  --prior-record manuscript/build/efsgd-reproduction-record.json \
  --output manuscript/build/efsgd-confirmation-reproduction \
  --record manuscript/build/efsgd-confirmation-reproduction-record.json
```

These are training-order replicates, not independent initializations or data
samples. Inherited fitting history is counted separately from new computation.
All three cells must finish before the unchanged quality gates are evaluated;
a passing result does not launch or authorize cloud execution.

The separate matched optimizer control uses the original exploratory cohort
and sequential order, with the exact same fitted weights:

```bash
python manuscript/tools/compare_fpga_lm_warm_signsgd.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/efsgd-reproduction \
  --prior-record manuscript/build/efsgd-reproduction-record.json \
  --output manuscript/build/warm-signsgd-reproduction \
  --record manuscript/build/warm-signsgd-reproduction-record.json
```

This checks every native update against a full host gradient/weight oracle,
retains failures, and compares the two fixed optimizers. It does not isolate
the remainder mechanism alone or replace three-run quality confirmation.

The declared projection-initializer study requires the complete confirmation
run (including its failed context gate), not the SignSGD control. It trains
three new cells and reevaluates all three historical controls on one fresh
development cohort:

```bash
python manuscript/tools/train_fpga_lm_diverse_projection.py \
  --corpus data/tinyshakespeare.txt \
  --confirmation-run manuscript/build/efsgd-confirmation-reproduction \
  --confirmation-record manuscript/build/efsgd-confirmation-reproduction-record.json \
  --output manuscript/build/diverse-projection-reproduction \
  --record manuscript/build/diverse-projection-reproduction-record.json
```

Only projection initialization changes. Historical controls import their exact
saved states and execute no new training; all six cells are required before
either arm's fixed quality gates are evaluated.

The completed CPU-only learning-contract diagnostic uses the complete
projection comparison as immutable input. Reproduce into new paths:

```bash
python manuscript/tools/diagnose_fpga_lm_learning_contract.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/diverse-projection-reproduction \
  --prior-record manuscript/build/diverse-projection-reproduction-record.json \
  --output manuscript/build/learning-contract-reproduction \
  --record manuscript/build/learning-contract-reproduction-record.json
python manuscript/tools/diagnose_fpga_lm_learning_contract.py --check \
  --output manuscript/build/learning-contract-reproduction \
  --record manuscript/build/learning-contract-reproduction-record.json
```

The first command executes the declared toy learning, algebra controls and
six separately reset native updates / 90 numerical counterfactuals. It does
not evaluate held-out corpus data or launch cloud resources. The second
verifies sources, inputs, raw artifact hashes and the public receipt without
repeating the scientific trajectories. Raw build artifacts stay local/ignored;
the compact checked record is under `evidence/` for the recorded original run.

The follow-up compares persistent EF with unrounded FP64 application of the
**same native learning rule**, not smooth exact-CE SGD. Its 32-window pairs
and later saved-checkpoint probes are declared in the
[research plan](research/PLAN.md#persistent-trajectory-diagnostic-v1-completed).
With the complete learning-contract run available, reproduce at new paths:

```bash
python manuscript/tools/diagnose_fpga_lm_persistent_trajectory.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/learning-contract-reproduction \
  --prior-record manuscript/build/learning-contract-reproduction-record.json \
  --output manuscript/build/persistent-trajectory-reproduction \
  --record manuscript/build/persistent-trajectory-reproduction-record.json
python manuscript/tools/diagnose_fpga_lm_persistent_trajectory.py --check \
  --output manuscript/build/persistent-trajectory-reproduction \
  --record manuscript/build/persistent-trajectory-reproduction-record.json
```

The checker replays saved full-gradient optimizer arithmetic and verifies
complete native/FP64 checkpoints, but does not rerun the model's forward or
backward. Both commands are CPU-only and make no cloud calls.

The paired update-scale screen uses the complete projection study as its source
of fitted starts and unchanged k18 controls. It trains six new k20 cells and
checks all 12 endpoints on a separately declared development cohort:

```bash
python manuscript/tools/test_fpga_lm_update_scale.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/diverse-projection-reproduction \
  --prior-record manuscript/build/diverse-projection-reproduction-record.json \
  --output manuscript/build/update-scale-reproduction \
  --record manuscript/build/update-scale-reproduction-record.json
# Integrity, coverage, aggregation and saved library/native update replay:
python manuscript/tools/test_fpga_lm_update_scale.py --check \
  --output manuscript/build/update-scale-reproduction \
  --record manuscript/build/update-scale-reproduction-record.json
# Also independently recompute all evaluation windows and 12 gradient anchors:
python manuscript/tools/test_fpga_lm_update_scale.py --check --recompute-evaluation \
  --output manuscript/build/update-scale-reproduction \
  --record manuscript/build/update-scale-reproduction-record.json
```

All paths must be new for a reproduction; the public record's parent directory
must exist and its path must be outside the raw run. The two checks are read-only.
Neither replays every training update or authorizes scaling. The library-facing
carry/resume example is `python -m examples.fixed_point_workflow`.

The [four-window accumulation diagnostic](research/PLAN.md#four-window-gradient-accumulation-v1-declared-2026-09-10-before-execution)
uses only the already-inspected training windows and the complete persistent
diagnostic as its predecessor. It changes gradient-query and integer-update
release timing, with no change to the per-example scale:

```bash
python manuscript/tools/diagnose_fpga_lm_gradient_accumulation.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/persistent-trajectory-reproduction \
  --prior-record manuscript/build/persistent-trajectory-reproduction-record.json \
  --output manuscript/build/gradient-accumulation-reproduction \
  --record manuscript/build/gradient-accumulation-reproduction-record.json
# Complete saved-gradient arithmetic, metrics, source and artifact checks:
python manuscript/tools/diagnose_fpga_lm_gradient_accumulation.py --check \
  --output manuscript/build/gradient-accumulation-reproduction \
  --record manuscript/build/gradient-accumulation-reproduction-record.json
# Also independently recompute every full gradient and loss readout:
python manuscript/tools/diagnose_fpga_lm_gradient_accumulation.py --check --recompute \
  --output manuscript/build/gradient-accumulation-reproduction \
  --record manuscript/build/gradient-accumulation-reproduction-record.json
```

Use fresh output/record paths. Checks are read-only and bounded to 1,800 seconds;
neither opens validation/test data, invokes a native training process or permits
scaling. The accumulated arm is a host reference, not a new FPGA implementation.

The [first-failing-block fraction probe](research/PLAN.md#first-failing-batch-step-fractions-v1-declared-2026-09-10-before-execution)
uses that completed run's saved state and four gradients. All fractions remain
uncommitted; it does not train another model:

```bash
python manuscript/tools/diagnose_fpga_lm_batch_step.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/gradient-accumulation-reproduction \
  --prior-record manuscript/build/gradient-accumulation-reproduction-record.json \
  --output manuscript/build/batch-step-reproduction \
  --record manuscript/build/batch-step-reproduction-record.json
python manuscript/tools/diagnose_fpga_lm_batch_step.py --check \
  --output manuscript/build/batch-step-reproduction \
  --record manuscript/build/batch-step-reproduction-record.json
python manuscript/tools/diagnose_fpga_lm_batch_step.py --check --recompute \
  --output manuscript/build/batch-step-reproduction \
  --record manuscript/build/batch-step-reproduction-record.json
```

Again use fresh output/record paths. The replay recomputes the four saved-point
gradients and 88 readouts. Dynamic integer candidates have a separate forward
oracle; affine and fractional-shadow readouts reuse their defining helpers.
This is a local diagnostic, not a rate-selection or hardware-release command.

The [normalized-accumulation order diagnostic](research/PLAN.md#normalized-accumulation-order-diagnostic-v1-declared-2026-09-11-before-execution)
compares sequential k18, sequential k20 and batch4 k20 in three fixed orders
of the same 32 training windows. It starts each trajectory with zero carry and
keeps its denominator fixed; it does not repeatedly rescale live carry:

```bash
python manuscript/tools/diagnose_fpga_lm_normalized_accumulation.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/gradient-accumulation-reproduction \
  --prior-record manuscript/build/gradient-accumulation-reproduction-record.json \
  --step-run manuscript/build/batch-step-reproduction \
  --step-record manuscript/build/batch-step-reproduction-record.json \
  --output manuscript/build/normalized-accumulation-reproduction \
  --record manuscript/build/normalized-accumulation-reproduction-record.json
python manuscript/tools/diagnose_fpga_lm_normalized_accumulation.py --check \
  --output manuscript/build/normalized-accumulation-reproduction \
  --record manuscript/build/normalized-accumulation-reproduction-record.json
python manuscript/tools/diagnose_fpga_lm_normalized_accumulation.py --check --recompute \
  --output manuscript/build/normalized-accumulation-reproduction \
  --record manuscript/build/normalized-accumulation-reproduction-record.json
```

Use new output/record paths and completed, mutually consistent predecessors.
The run and read-only audit each have a 3,600-second deadline. The replay
checks all 288 full gradients and 434 scored readouts. Any numerical training
gate remains conditional on successful independent replay and is not quality
promotion, reserved-data access or permission to scale hardware.

The head-weighting result has a completed independent CLI audit receipt.
To recheck the retained run (requires its complete source-bound raw ancestry):

```bash
python manuscript/tools/diagnose_fpga_lm_head_weighting.py --check --recompute \
  --output manuscript/build/fpga_lm_head_weighting_2026-09-11/v1 \
  --record manuscript/evidence/fpga_lm_head_weighting_2026-09-11.json
```

The new exact-CE fitting protocol is independently declared, rather than
automatically authorized by the failed head-weighting gate. A new run requires
the completed head-weighting run and its actual successful audit receipt;
do not manufacture a receipt for an audit that has not run:

```bash
python manuscript/tools/diagnose_fpga_lm_exact_fit.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/fpga_lm_head_weighting_2026-09-11/v1 \
  --prior-record manuscript/evidence/fpga_lm_head_weighting_2026-09-11.json \
  --prior-audit manuscript/evidence/fpga_lm_head_weighting_audit_2026-09-11.json \
  --output manuscript/build/exact-fit-reproduction \
  --record manuscript/build/exact-fit-reproduction-record.json
python manuscript/tools/diagnose_fpga_lm_exact_fit.py --check --recompute \
  --output manuscript/build/exact-fit-reproduction \
  --record manuscript/build/exact-fit-reproduction-record.json
```

Use exclusive new output paths and retain failures. Each invocation is bounded
to 3,600 seconds. The independent audit reconstructs 1,024 tropical window
VJPs, compares 256 summed tropical and 128 GRU batch gradients, verifies every
Adam state/update and rechecks all 36 score-window readouts. Training-fit success
would still need development-quality confirmation and a low-arithmetic numeric
implementation before hardware scaling. Archived receipts bind original paths
and sources; moving to another machine requires regenerating the corresponding
ancestry, not editing old evidence into a purported successful reproduction.

The separate one-rate follow-up requires the completed exact-fit audit above.
It changes only the mixed model's Adam learning rate and retains every state:

```bash
python manuscript/tools/diagnose_fpga_lm_exact_rate.py \
  --corpus data/tinyshakespeare.txt \
  --prior-run manuscript/build/fpga_lm_exact_fit_2026-09-11/v1 \
  --prior-record manuscript/evidence/fpga_lm_exact_fit_2026-09-11.json \
  --prior-audit manuscript/evidence/fpga_lm_exact_fit_audit_2026-09-11.json \
  --output manuscript/build/exact-rate-reproduction \
  --record manuscript/build/exact-rate-reproduction-record.json
python manuscript/tools/diagnose_fpga_lm_exact_rate.py --check --recompute \
  --output manuscript/build/exact-rate-reproduction \
  --record manuscript/build/exact-rate-reproduction-record.json
```

Each invocation has a 3,600-second deadline, including ancestor validation.
The new run binds 61 source/test files and retains 403 payloads; its independent
audit checks 512 window VJPs, 128 summed gradients, 129 complete Adam states and
12 score readouts. Checked-in summary JSON alone is insufficient: these research
runners require the retained raw ancestry. A self-contained distributable
benchmark bundle remains release work; do not claim that a fresh checkout can
replay the whole diagnostic chain without those artifacts. The separate guarded
forward is not enabled by either command and makes no timing claim.
These commands describe the preserved full-reproduction protocol, not permission
for another attempt. Two historical recursive audits hit their deadlines and
remain failures; the separately declared September 13 pinned-predecessor audit
successfully verifies the original rate trajectory without rerunning predecessor
numerical trajectories.

The new pinned auditor and guarded-parity verifier are narrowly bound to the
retained **original** run, source bytes, public record and completed audit receipt.
They are not generic or portable checkers for the fresh reproduction commands
above: different paths, new producer hashes or a newly timestamped audit receipt
intentionally fail their pins. Fresh reproductions require their own full
ancestry/semantic verification and separately reviewed pinning, without editing
historical receipts. Source-distributable benchmark packaging remains open work.

## Build and share the PDF

`TikiTorch.pdf` is the checked review copy, including references and appendices.
Send it together with this README for an
initial scientific discussion. [Its manifest](evidence/review_pdf.json) binds
the PDF to the source, build helpers and selected evidence. `--check` verifies
freshness without LaTeX or network access.

```bash
# Requires Python, latexmk and TeX Live/MacTeX (forloop, placeins, pgf, xurl).
# First build: fetch the pinned official, unmodified venue template.
python manuscript/build.py --fetch-style
# Later builds can use the cached template offline:
python manuscript/build.py
# Read-only check of the committed review copy:
python manuscript/build.py --check
```

Alternatively pass `--template-zip /path/to/iclr-2027-style-files.zip`. Successful builds refresh
the review PDF; failed checks leave the previous copy intact. Downloaded styles,
TeX dependencies and intermediate outputs stay in ignored `build/`. Paper sources
and the PDF are excluded from library distributions.

## Submission and release gates

- **Scientific scope:** `route_logits` currently supports a narrow hard MorphSSM
  language model, not arbitrary graphs, pooled ECG classifiers or conventional
  vocabulary heads. Closure belongs to a complete executed forward/loss/backward/
  update recipe; a disclosed hybrid boundary is not a fully closed pipeline.
- **Evidence:** real-task multi-seed quality/resource measurements and the intended
  GPU stack are unvalidated. Historical FPGA rows show scoped feasibility, not
  current-Python export equivalence, timing closure or a matched GPU/energy advantage.
  Unreconciled evaluators remain visible in the [evidence ledger](research/EVIDENCE.md).
- **Distribution:** AWS-derived RTL and retained-history review remain open in
  [release provenance](../docs/release_provenance.json). Passing integrity checks
  does not grant redistribution rights; the release gate may intentionally fail.
  Preserve required licenses/attribution, including in any separately reviewed
  anonymous source bundle. No publication, relicensing or history rewrite is authorized.
- **Human review:** agree on contributions/authorship, order, affiliations,
  acknowledgments, funding and conflicts; independently check proofs and citations.
  Assignment does not establish authorship or permission to submit. This work used
  AI assistance; human authors remain responsible and must review applicable
  disclosure requirements and describe assistance accurately.
- **Venue:** the official [ICLR 2027 guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
  and template were verified on 2026-09-09. Initial submissions allow nine
  main-text pages, with references/appendices separate from that limit. The
  [AI-use statement](https://iclr.cc/Conferences/2027/AIPolicyForAuthors) is mandatory;
  the draft includes it plus ethics and reproducibility statements. The PDF is
  explicitly unsubmitted. Formatting does not establish scientific readiness,
  authorship approval or a commitment to a particular deadline.

For scientific review, ask: what does the integrated contract enable beyond a
custom VJP or tuned checkpoint schedule; does it survive heads, optimizers and
live workspaces at useful quality; which exactness notion is tested; and can an
independent researcher reproduce the result without privileged hardware or
unreviewed source rights?

## Where things live

| Location | Purpose |
|---|---|
| [research/PLAN.md](research/PLAN.md) | One research plan and experimental protocol |
| [research/EVIDENCE.md](research/EVIDENCE.md) | Claim boundaries, record index and artifact policy |
| [research/LITERATURE.md](research/LITERATURE.md) | Closest primary literature and novelty positioning |
| [source/](source/) | Active LaTeX, bibliography, template manifest and generated tables |
| [tools/](tools/) / [evidence/](evidence/) | Measurement/reduction helpers and append-only records |
| [results/](results/README.md) / [archive/](archive/README.md) | Preserved raw research and superseded papers |

Reusable code stays in `tikitorch/`, examples in `examples/`, and research runners
in `scripts/` and the unstable `tikitorch.research` namespace. The
[route-training guide](../docs/guide/route-training.md),
[claim ledger](../docs/research/CLAIM_LEDGER.md) and
[release/impact roadmap](../docs/release_impact_roadmap.md) cover library contracts
and release work; this README is the manuscript status and handoff.
