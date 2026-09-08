# TikiTorch collaboration handoff

One starting point for prospective contributors, students, Xiang and other
reviewers. The immediate milestone is independent onboarding and agreement on
a bounded project—not launching the research program. No owner, compute budget,
authorship or submission decision is presumed.

## The project and the question

TikiTorch is an independent PyTorch companion for tropical and hard-routed
learning. Can compact records of hard operators' routing decisions make
**useful learning cheaper or accessible on smaller hardware**, while keeping
the arithmetic of each learning recipe auditable?

The [single active manuscript](README.md) connects route recording, selected
first-order reverse computation, PyTorch integration and closure audits. The
chain rule itself is not the novelty. The [research plan](RESEARCH_PLAN.md)
tests practical ECG classification and language-model pretraining, followed by
an optional complete-learning FPGA capstone. The [older drafts](archive/README.md)
are historical reference, not additional papers to maintain.

## Where to start

| Purpose | Canonical reference |
|---|---|
| Install, contribute and run the library checks | [Contributing](../CONTRIBUTING.md) and [test guide](../tests/README.md) |
| Understand supported route training | [Route-training guide](../docs/guide/route-training.md) |
| Review current versus historical evidence | [Evidence ledger](EVIDENCE_LEDGER.md) and [claim ledger](../docs/research/CLAIM_LEDGER.md) |
| Choose and design a research project | [Research plan](RESEARCH_PLAN.md) and [experiment protocol](EXPERIMENT_PROTOCOL.md) |
| Evaluate submission and software-release gaps | [ICML readiness](ICML_READINESS.md) and [release/impact roadmap](../docs/release_impact_roadmap.md) |

Reusable code is in `tikitorch/`, small examples in `examples/`, tests in
`tests/`, and research runners in `scripts/` plus the unstable
`tikitorch.research` namespace. Paper sources, evidence and archived drafts
share this `manuscript/` workspace.

## First contribution: reproduce and explain

Follow the [development setup](../CONTRIBUTING.md#dev-setup) in a fresh Git
checkout and virtual environment. No GPU, FPGA, dataset download or manuscript
rewrite is required. From the repository root:

```bash
python -m examples.closure_workflow
python -m pytest -q tests/test_route_training.py tests/test_morphssm_tape_parity.py
python scripts/check_artifact_layout.py
python manuscript/generate_tables.py --check
python manuscript/measure_retained_state.py --check
```

Then run the standard lint and full test checks in the contributing guide.
Deliver a short note with commit/worktree state, Python/PyTorch/platform
versions, commands, actual pass/skip/fail outcomes, and one scoped issue or PR.
Preserve skip reasons and environment limitations, including unavailable
localhost sockets; a configured CI matrix is not proof every environment passed.

Good first contributions clarify an error, fix a documented command with a
regression test, or explain an actual unsupported configuration. Do not widen
numerical tolerances or modify evidence hashes to make a check pass. A
source-bound evidence mismatch needs maintainer review and an append-only
remeasurement; see [artifact preservation](ARTIFACTS.md). PDF build instructions
are in the [manuscript overview](README.md), if needed for review.

## Boundaries to preserve

- `route_logits` covers a narrow hard MorphSSM language-model configuration,
  not arbitrary graphs, pooled ECG classifiers or conventional vocabulary
  heads. Extensions need contracts and equivalence tests, not removal of guards.
- Selected-route VJPs use declared local tie rules; those choices need not
  produce a global Clarke subgradient. Smaller retained bytes alone do not
  establish lower measured peak memory, faster learning or competitive quality.
- Native compiled AdamW has a documented update-parity failure on the tested
  stack, including compiled ordinary autograd. Passing SGD does not fix it.
- Closure belongs to an executed forward/loss/backward/update recipe. A
  disclosed hybrid boundary is allowed but is not fully closed. Software
  checks do not certify DSP use, energy or timing closure.
- Historical FPGA records establish scoped feasibility, not current-Python
  equivalence or a matched GPU advantage. Unreconciled evaluators and other
  limitations remain visible in the evidence ledger.
- Third-party RTL and retained-history review remain release gates. Onboarding
  does not authorize publication, relicensing, deleting provenance or rewriting
  history. The release checker may intentionally fail.

## Agree on a project and review its contribution

The preferred first research milestone is an **ECG learning-quality pilot**
after pooled-classifier and multilabel-loss tests exist. A complementary
systems project is the **LM head/boundary prototype**, validating scalable
readout before larger pretraining runs. Neither commits a student to both
studies or the FPGA capstone.

Before starting, contributors and maintainers should agree on:

1. Owner, supervisor/reviewer, first deliverable and stop/review condition.
2. Exact dataset release/terms, splits, preprocessing and storage; no patient
   information or dataset binaries in pull requests.
3. Existing authorized hardware, runtime/storage/spending limits and permitted
   downloads or external services.
4. Baselines, quality metric, pilot versus final-test roles, tuning budget and
   artifact retention. Do not select final-test targets after seeing results.
5. Expected contribution, supervision and authorship discussion; an assignment
   alone does not establish authorship or permission to submit.

For Xiang and other scientific reviewers: does the integrated contract plus a
useful real-task quality/resource result support a strong contribution? Which
result is essential, who can own it, and which approvals or rights remain open?
Use the readiness checklist and evidence ledger to answer those questions;
the provisional venue template is not a verified next-cycle submission guarantee.
