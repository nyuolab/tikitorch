# Student handoff: TikiTorch

Status: available for discussion, 2026-09-08. No student assignment or research
run is presumed. The immediate objective is to make a first contribution from
a clean checkout, then agree on a bounded project with a maintainer. An FPGA,
GPU account or manuscript rewrite is not required for this first milestone.

## What the project is about

TikiTorch is an independent PyTorch companion for tropical and hard-routed
research. Its distinctive hypothesis is that a compact record of hard
operators' routing decisions can support useful training with less retained
state, while arithmetic audits make each recipe's constraints explicit.

The practical question is **whether useful learning becomes cheaper or
accessible on smaller hardware**, not whether a route encoding is smaller in
isolation. The [research plan](RESEARCH_PLAN.md) proposes ECG classification and
language-model pretraining, with a later complete-learning FPGA capstone.

## First reading and repository map

- [Library README](../README.md) and [contributing guide](../CONTRIBUTING.md):
  setup, API scope and contributor expectations.
- [Route-training guide](../docs/guide/route-training.md): current adapter and
  integration limits, plus same-model benchmarking.
- [Manuscript overview](README.md), [evidence ledger](EVIDENCE_LEDGER.md) and
  [readiness checklist](ICML_READINESS.md): what the current paper does and does
  not establish.
- [Research results](results/README.md): historical logs, figures and manifests.
  These are evidence, not reusable package APIs or clean reproductions of
  today's source by default.
- [Release/impact roadmap](../docs/release_impact_roadmap.md): maintainers'
  unresolved engineering and publication gates.

Reusable code lives in `tikitorch/`; small executable examples in `examples/`;
tests in `tests/`; research runners in `scripts/` and the explicitly unstable
`tikitorch.research` namespace. The single active paper and research evidence
live under `manuscript/`. The former scientific sources under
[`manuscript/archive/`](archive/README.md) are historical reference, not two
additional papers to edit. The [Xiang handoff](XIANG_HANDOFF.md) is in this same workspace.

## First contribution: reproduce and explain, without research hardware

Use a supported Python version (currently 3.9–3.12; 3.11 is a useful starting
point). From a source checkout in a new virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m examples.closure_workflow
python -m pytest -q tests/test_route_training.py tests/test_morphssm_tape_parity.py
python -m ruff check .
python -m pytest -q
```

The default suite deliberately excludes slow learning and native-kernel
markers; hardware-dependent tests may skip. Preserve the actual pass/skip/fail
summary and skip reasons. First-order numerical or compiler failures are
findings to report, not reasons to widen tolerances until tests pass. If the
environment cannot install dependencies or permit required localhost sockets,
record that limitation rather than treating it as a scientific result.

Then check the paper's evidence bindings:

```bash
python manuscript/generate_tables.py --check
python manuscript/measure_retained_state.py --check
python scripts/check_artifact_layout.py
python -m pytest -q tests/test_manuscript_tables.py tests/test_manuscript_artifact.py
```

These checks are not new learning experiments. If current sources differ from
a source-bound record, stop and report the mismatch. Do not edit old hashes or
raw measurements to make it pass; a maintainer must approve a new append-only
record and any corresponding manuscript table update. Building the PDF is
optional for initial library work; see [build instructions](README.md).

Deliver a short reproduction note with commit/worktree status, Python/PyTorch/
platform versions, exact commands, outcomes and one scoped issue or pull
request. Good first contributions include improving an unclear error message
with a regression test, fixing a broken documented command, or documenting an
actual unsupported configuration. Keep unrelated cleanup out of the change.

## Boundaries that must remain explicit

- `route_logits` currently supports a narrow hard MorphSSM language-model
  configuration, not arbitrary PyTorch graphs, pooled ECG classification or
  a conventional vocabulary head. These extensions require contracts and
  equivalence tests, not simply removal of guards.
- Selected hard-route VJPs are first-order derivatives under declared local
  tie rules. Local tie choices need not produce a global Clarke subgradient.
- Smaller retained route bytes are not measured peak-memory or speed gains.
  The current checkpoint-boundary comparator is explicitly shape-defined.
- Native compiled AdamW has a documented update-parity failure on the tested
  stack, including ordinary compiled autograd. A passing SGD configuration
  neither fixes nor establishes AdamW equivalence.
- Closure belongs to a specific executed forward/loss/backward/update recipe.
  A hybrid PyTorch boundary is allowed, but must not be called fully closed.
  Software closure is not a zero-DSP, energy or FPGA timing certificate.
- Third-party RTL provenance and retained-history review remain maintainer
  release gates. Do not publish, remove provenance records or rewrite history
  as an onboarding task. The release checker may intentionally fail.

## Choose a project with the maintainer before starting experiments

The preferred first research milestone is the **ECG learning-quality pilot**,
after its pooled-classifier interface and multilabel-loss tests exist. A
complementary systems project is the **LM head/boundary prototype**, measuring
head-state growth and validating a hybrid readout before large training runs.
Neither requires committing now to both full studies or the FPGA capstone.

Before any pilot, agree in writing on:

1. Owner, supervisor/reviewer, first deliverable and stop/review condition.
2. Exact dataset release and terms, split/preprocessing policy and storage
   location; no patient information or dataset binaries in pull requests.
3. Existing authorized hardware, maximum runtime/storage/spending, and whether
   any external service or download is allowed.
4. Baselines, quality metric, pilot versus final-evaluation roles, and artifact
   retention. Do not choose final-test targets after inspecting test results.
5. Expected contribution and authorship discussion; a task assignment alone
   does not settle authorship or authorize submission.

Use [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the sequence and
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for rigorous comparisons. The
current handoff intentionally stops before those experiments are launched.
