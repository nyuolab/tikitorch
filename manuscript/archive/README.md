# Archived TikiTorch drafts

The single active, ICML-targeted manuscript is in [the parent workspace](../README.md):

**TikiTorch: Route-Sufficient Differentiation for Auditable Tropical Learning**

It combines the derivative-state and arithmetic-closure strands of the former
two-paper plan. Theory and current-code experiments anchor the main text;
historical portable-tape and FPGA results retain their own provenance and
qualifications in the appendix. Combining the narrative does not make these
different implementations or measurements interchangeable.

## Historical sources, not active submissions

- [Route-certificate draft](route_certificates/): preserved former scientific
  source, methods, supplement and bibliography.
- [Multiplier-free FPGA draft](multiplier_free_fpga/): preserved former
  scientific source and bibliography.

These directories are historical references, not independent active submission
tracks. Their old venue, ownership, test-count and “must remain separate”
instructions are superseded by the current user-directed consolidation.
Do not edit their numerical records to make them agree with new software.
Make scientific revisions in `manuscript/source/`, not this archive.

For contributors and reviewers, use the [manuscript README and handoff](../README.md). The canonical
[research claim ledger](../../docs/research/CLAIM_LEDGER.md) continues to constrain
claim wording; the unified paper's [evidence ledger](../research/EVIDENCE.md)
maps individual rows to raw artifacts. The
[release/impact roadmap](../../docs/release_impact_roadmap.md) separates code maturity,
empirical evidence and distribution authority.

Ontogenic-learner/agent work remains outside this paper and repository scope.

## Consolidation record

On 2026-09-08 the former root `papers/` tree was consolidated into `manuscript/`:

| Former path | Current path (from the repository root) |
|---|---|
| `papers/XIANG_HANDOFF.md` | `manuscript/README.md` (now includes student/reviewer handoff) |
| `papers/README.md` | `manuscript/archive/README.md` (this index) |
| `papers/route_certificates/` | `manuscript/archive/route_certificates/` |
| `papers/multiplier_free_fpga/` | `manuscript/archive/multiplier_free_fpga/` |

All 12 tracked files were hash-identical immediately after the move. Only the
four Markdown navigation/readme files were then updated. The eight scientific
sources, bibliographies and `.latexmkrc` files remain byte-identical to commit
`1b4b00308a53919dee6afbc8fdca86011659d348`; their hashes and executable modes
are recorded in [source_manifest.json](source_manifest.json) and regression-tested.
Locally ignored `build/` products moved with each draft and remain ignored.
No draft or result was deleted, combined a second time, or relabeled as current evidence.
The later documentation cleanup combined the separate Xiang and student handoffs
into `manuscript/HANDOFF.md`. On 2026-09-09 that handoff and the readiness checklist
were absorbed into `manuscript/README.md`; the former documents remain in Git history.

Archived drafts can still be built from their own directories using their
documented local build commands. Their scientific claims, dependencies, venue
choices and reported test counts remain historical. The active paper's build
entry point is `python manuscript/build.py` from the repository root.
