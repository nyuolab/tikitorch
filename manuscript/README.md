# TikiTorch manuscript and handoff

**[Read the current manuscript (PDF)](TikiTorch.pdf)**

Status: **2026-09-09 — one working draft, not yet submission-ready.**
This is the shared starting point for students, Xiang, contributors and reviewers;
there is no separate handoff or readiness document to maintain.

*TikiTorch: Route-Sufficient Differentiation for Auditable Tropical Learning*
asks whether compact routing records can make **useful learning cheaper or
accessible on smaller hardware**, with auditable arithmetic for each learning
recipe. TikiTorch is an independent PyTorch companion. Its contribution connects
derivative state, selected reverse computation and arithmetic contracts—not a
claim to have invented morphological backpropagation or multiplication-free training.

## What is done

- The two papers are combined into one narrative with theory, methods, references
  and qualified historical FPGA results. Superseded scientific sources are
  byte-preserved in [archive/](archive/README.md), not competing drafts.
- The draft states the joint state/parameter reverse contract, local tie rules,
  proofs and counterexamples. Selected local derivatives are explicitly **not**
  guaranteed to be global Clarke subgradients; the coding bound is conditional.
- The library has a narrow, tested device-tape/PyTorch route-training path,
  fail-closed arithmetic audits and a tiny runnable onboarding workflow.
- Source-bound CPU measurements check logits, routes and parameter VJPs. Current
  retained-state ratios are **2.04–2.98×** against a shape-defined checkpoint
  boundary; these are representation measurements, **not measured peak memory,
  training speed, energy or useful-task quality improvements**.
- Tables are derived from checked records. The PDF build verifies evidence,
  pinned venue styles, references and main-body pagination before refreshing the
  review copy. Historical measurements remain separate and append-only.

Local sharing-pass validation (2026-09-09): **1,150 tests passed, 8 skipped,
8 slow tests deselected**; lint, strict documentation, artifact/PDF integrity
checks and the actual PDF build passed. This is not remote CI, independent
scientific reproduction or GPU validation.

## What needs to happen next

1. **Independent onboarding and a bounded project.** Reproduce the checks below;
   agree on an owner, reviewer, first deliverable and stop condition. Confirm the
   exact dataset release/terms, splits, permitted downloads, existing hardware
   and runtime/storage/spending limits before experiments. No study is launched
   or compute budget approved by this plan.
2. **Make the comparison trustworthy.** Test compatible compiled-checkpoint
   baselines and selective-save coverage; diagnose native compiled AdamW's first
   numerical divergence. Passing SGD is a separate recipe, not an AdamW fix.
   Include complete-step costs, memory, failures and parity—not retained bytes alone.
3. **Enable one useful ECG workload.** First add a tested route-trained body,
   pooling/readout boundary and multilabel loss. Then run a quality pilot against
   a tuned conventional model and strong checkpoint baselines. Fix selection
   rules on validation data before multi-seed held-out quality/resource results.
   A smaller-hardware or memory advantage at matched useful quality is compelling
   even without a speedup. If none survives, narrow the claim.
4. **Gate language pretraining on its head.** Prototype the conventional
   vocabulary head/loss boundary and measure its full cost before larger runs.
   ECG and LM need not both become prerequisites for a focused paper. A complete
   FPGA learning system is an optional later capstone, not an automatic ICML blocker.
5. **Freeze and independently reproduce the candidate.** Archive exact source,
   environment and commands, rerun append-only evidence, regenerate the PDF and
   resolve the human/distribution gates below. A source hash alone is not an
   independent reproduction or a tagged release.

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

Alternatively pass `--template-zip /path/to/icml2026.zip`. Successful builds refresh
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
- **Venue:** the anonymous ICML 2026 style is a provisional drafting aid. Its
  [author instructions](https://icml.cc/Conferences/2026/AuthorInstructions) allow
  eight main-text pages with references/appendices in the same PDF and require an
  impact statement. That conference has passed; next-cycle rules and dates are
  **not verified**. Recheck the actual target cycle before submission.

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
