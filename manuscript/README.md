# TikiTorch: Route-Sufficient Differentiation for Auditable Tropical Learning

This directory is the **single active manuscript**, combining the former
route-certificate and multiplier-free FPGA drafts around TikiTorch.
The older scientific sources remain in `papers/` as historical reference;
they are not parallel submission drafts.

The main contribution is an executable connection between the derivative state
needed by additive hard operators and the arithmetic executed by a learning
recipe. It is not a claim to have invented morphological backpropagation,
compact switches, or multiplication-free training.

## Read and build

- [Main paper](main.tex), [theory appendix](theory_appendix.tex), and
  [methods/evidence appendix](methods_appendix.tex).
- [Evidence ledger](EVIDENCE_LEDGER.md) and [primary-literature comparison](LITERATURE_REVIEW.md).
- [ICML readiness](ICML_READINESS.md) and [remaining experiment protocol](EXPERIMENT_PROTOCOL.md).
- [ECG, language pretraining and FPGA research plan](RESEARCH_PLAN.md) and
  [student handoff](STUDENT_HANDOFF.md).
- [Research results and historical artifacts](results/README.md), kept here
  separately from the reusable library.
- [Artifact preservation and relocation policy](ARTIFACTS.md).
- [Post-manuscript codebase/release roadmap](../docs/release_impact_roadmap.md).

The next step is a student/maintainer handoff and independent onboarding, not
launching the proposed studies. The plan targets useful learning on smaller
hardware, with practical ECG and language-pretraining results before a complete
FPGA learning capstone. Hardware, data, runtime and spending decisions remain
unapproved; a student can begin with local correctness checks without a GPU.

From the repository root:

```bash
# Requires Python, latexmk and TeX Live/MacTeX (including forloop, placeins, pgf, xurl).
# First build only: fetch the pinned official, unmodified template.
python manuscript/build.py --fetch-style
# Subsequent builds are offline:
python manuscript/build.py
```

An already-downloaded official archive can be supplied with
`--template-zip /path/to/icml2026.zip`. Venue styles and PDF/auxiliary files
stay in ignored `manuscript/build/`; they are not vendored into the package.
The PDF is `manuscript/build/main.pdf`. The build checks both evidence reducers,
style hashes, unresolved citations, overfull boxes and the main-body page marker.

The latest verified **2026** ICML style is a provisional drafting aid.
The 2026 conference has passed; next-cycle instructions and dates have not been
verified. Anonymous formatting is not confirmation of authorship or permission
to submit. Do not use the accepted/camera-ready option or invent author details.

## Evidence boundaries

The new current-source, three-cell CPU audit records device payloads of
36,084 / 87,588 / 329,928 bytes, with exact logits and canonical route checks
and tolerance-gated parameter VJPs. The 2.04 / 2.81 / 2.98 ratios compare a
**shape-defined layer/embedding checkpoint retained set to device payload**,
not peak memory, the best checkpoint schedule or speed.

```bash
python manuscript/measure_retained_state.py --check
python manuscript/generate_tables.py --check
python -m pytest -q tests/test_manuscript_tables.py tests/test_manuscript_artifact.py
```

The selected record is
[`evidence/retained_state_handoff_2026-09-08.json`](evidence/retained_state_handoff_2026-09-08.json),
an append-only remeasurement after the library cleanup. It reproduces the same
table; the original `evidence/current_retained_state.json` remains unchanged
as an earlier source snapshot. The current record hashes every library Python source and its measurement
script. If code changes, the check fails: produce a new append-only record,
audit it, and deliberately update which table the manuscript cites. The
current worktree is not a tagged release. To rerun without overwriting evidence:

```bash
python manuscript/measure_retained_state.py --run \
  --output manuscript/build/reproduction.json \
  --table manuscript/build/reproduction.tex
```

Use new output names for subsequent runs. Historical portable-tape rows are
kept separately in the appendix, with raw-source hashes. Hardware records
establish scoped feasibility, not current-Python export equivalence or a
matched GPU/energy comparison. Native compiled AdamW update parity remains
a documented failure on the tested stack; the SGD smoke is a separate recipe.

## Submission status

The combined scientific draft is substantially more rigorous but **not yet
submission-ready**. The main missing result is useful real-task training with
multi-seed, matched-quality memory/latency evidence, including the strongest
feasible checkpoint baseline. GPU measurements, clean archived reproduction,
artifact rights review and human authorship/content approval remain open.
The [readiness checklist](ICML_READINESS.md) makes these gates explicit.

The pre-cleanup manuscript pass on 2026-09-08 had 899 tests passed, 8 skipped
and 8 deliberately deselected; 21 checked manuscript/evidence integrity.
See the [subsequent library handoff audit](../docs/library_handoff_audit_2026-09-08.md)
for the cleanup and its validation.
The strict documentation build, lint and PDF build passed. The anonymous PDF
has six main-text pages and 16 pages total, with resolved references and no
overfull boxes. These are local artifact checks, not remote CI, independent
reproduction or scientific acceptance. The first sandboxed suite was unable
to open localhost sockets; the full rerun with loopback permitted passed.

Nothing in this manuscript pass submits a paper, creates a public release,
changes repository visibility, rewrites history, or authorizes a paid run.
