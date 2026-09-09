# Library and research handoff cleanup — 2026-09-08

This bounded pass starts from `197b2644441877a9662d5a33b84b118bef169bc5`.
It records the proposed research program, relocates artifacts and prepares a
student-facing library handoff. It does not execute the ECG/pretraining plan,
change the scientific manuscript, publish a release or authorize cloud work.
The earlier [implementation audit](codebase_audit_2026-09-08.md) describes a
different pass and retains its original measurements.

## Completed

- A proposed ECG and language-pretraining program, quality/resource comparison
  gates, and later complete-learning FPGA capstone; CPU-first student onboarding
  and an explicit pause before research execution.
- `results/` moved to `manuscript/results/`. All 686 locally present files were
  hash-identical immediately after the move. The durable inventory covers 544
  tracked artifacts: 538 unchanged and six navigation-only Markdown edits.
  Active paths and generated provenance keys follow the move; raw evidence,
  S3 object keys and the prior source-bound measurement remain unchanged.
- A fail-closed relocation checker and CI check, plus library-only source
  archive boundaries. Research manuscripts, results, runners and checkout-only
  tests are not distributed as part of the Python library archives.
- Canonical no-waiver LM strict auditing of the configured batch/device and
  training state; hardened candidates are distinct. Soft schedules, temperature
  floors and untraced CountAgreement accumulation cannot certify strict training.
  Hard-from-scratch initialization now actually uses zero temperature before
  auditing and the first update, including replaced heads.
- The PC runner's legacy batch-2 hardened cascade/optimizer probe is explicitly
  scoped in its JSON and console output, not advertised as proof of configured
  soft/annealed training. Research data/output defaults resolve a checkout or
  caller workspace, never the installed package directory.
- Two historical GPU/F2 launchers now require explicit launch, transfer the
  required package and scripts, expose side-effect-free help, propagate remote
  pipeline failures, clear stale completion markers, and retain local state
  when termination fails. Tests use local fixtures/mocked commands, not AWS.
- Input-range, quantization-gradient, multilabel-loss and hardware-claim
  clarifications in library documentation, with research/student navigation.

## Validation and evidence

Local stack: macOS arm64, Python 3.11.5, PyTorch 2.10.0, no CUDA device.
Validation is local, not confirmation that the configured remote CI matrix ran.

Final default suite: **1,015 passed, 8 skipped, 8 deliberately deselected** in
63.30 seconds, with the expected backend-fallback warning. Localhost socket
access was permitted for the distributed tests. All 40 tracked Bash scripts
passed syntax checks without execution. Relocation verification covers all
544 frozen files; actual source/wheel archive checks passed (257 / 212 members).

The source-bound retained-state protocol was rerun for its existing three
CPU cells, under the append-only record
[`retained_state_handoff_2026-09-08.json`](https://github.com/nyuolab/tikitorch/blob/main/manuscript/evidence/retained_state_handoff_2026-09-08.json).
It reproduces 36,084 / 87,588 / 329,928 packed bytes and passes the existing
exact-logit/canonical-route and tolerance-gated VJP checks. The generated
scientific tables are unchanged. Historical reducer source keys now use the
relocated paths, with identical input digests and numerical outputs. This is
representation-regression evidence, not a new learning, memory-peak or speed study.

The strict documentation build, Ruff, onboarding example, evidence validators
and offline manuscript PDF check pass. The PDF retains six main-text pages
and 16 total pages; this pass does not revise the scientific prose.

Source archive and wheel built in an isolated environment using locally cached
build dependencies with Python 3.11. Twine metadata checks pass. The installed
wheel, tested outside the checkout, passes import/version, route backward,
research-workspace fallback and CLI smoke checks. No distribution was uploaded.

## Remaining gates and next action

The next action is a student/maintainer review of the
[handoff](https://github.com/nyuolab/tikitorch/blob/main/manuscript/README.md)
and [research plan](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/PLAN.md).
No dataset download, ECG/LM pilot, accelerator allocation, FPGA synthesis or
paid service call was initiated. Library cleanup is not evidence of a practical
workload advantage, and research adapters in the plan remain future work.

Native compiled AdamW update parity remains a known failure on the documented
stack; this pass neither fixes it nor widens tolerances. CUDA and hardware
claims still require actual-device validation. Other historical cloud helpers
remain revision-bound and require source inspection before use.

The provenance inventory passes integrity checks but reports `release_ready:
false`: four AWS-derived RTL records and retained Git history still require
human review/resolution. Moving or excluding artifacts from a Python archive
does not clear publication of the repository or its history. See
[release validation](release_validation.md) and the
[remaining engineering roadmap](release_impact_roadmap.md).
