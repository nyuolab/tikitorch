# Research scope and evidence

Library documentation and research claims have different jobs. The user guide explains
how to use the current Python surface. Research claims require a frozen configuration,
archived evidence, explicit limitations, and a reproduction path.

## Evidence hierarchy

When descriptions disagree, use this order:

1. Raw artifacts and executable tests.
2. The canonical [research claim ledger](CLAIM_LEDGER.md).
3. Paper workspaces and result summaries.
4. General documentation, examples, and manuscript prose.

The ledger uses `Established`, `Measured`, `Draft evidence`, `Preliminary`, `Negative`,
and `Open` as distinct statuses. A result should be described no more broadly than its
ledger entry.

## Current workstreams

The proposed next studies are ECG classification and language-model pretraining,
with an eventual complete-learning FPGA capstone. Start with the
[research plan](https://github.com/nyuolab/tikitorch/blob/main/manuscript/RESEARCH_PLAN.md)
and [collaboration handoff](https://github.com/nyuolab/tikitorch/blob/main/manuscript/HANDOFF.md).
These are planning documents, not established workload advantages or approved runs.
Historical artifacts now live under `manuscript/results/`; the
[artifact policy](https://github.com/nyuolab/tikitorch/blob/main/manuscript/ARTIFACTS.md)
explains unchanged raw records and old embedded paths.

| Workstream | Workspace | Boundary |
| --- | --- | --- |
| TikiTorch library | [source and API docs](../api/index.md) | PyTorch-shaped, not complete PyTorch compatibility or production maturity |
| Unified TikiTorch manuscript | [`manuscript`](https://github.com/nyuolab/tikitorch/tree/main/manuscript) | Route-sufficient selected reverse programs and independent arithmetic auditing; current-code representation measurements and historical FPGA feasibility remain separately scoped |

The older route-certificate and FPGA drafts under `manuscript/archive/` are preserved
historical sources, not independent active submission tracks. See the
[release and scientific-impact roadmap](../release_impact_roadmap.md) for the
post-manuscript priorities and required acceptance evidence.

Paper source is kept separate from the installed package and user documentation. Draft
manuscripts are working research artifacts, not release guarantees.

## Reproducing results

Begin with the relevant paper README and follow its exact commit, environment, script,
and artifact references. The
[results map](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/README.md)
distinguishes the major evidence areas, and the
[runner map](https://github.com/nyuolab/tikitorch/blob/main/scripts/README.md) explains which
scripts are public examples versus revision-bound research tools. Large FPGA build
products may remain outside Git; the
[artifact manifest](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/fpga_silicon/s1_afibuild/ARTIFACTS.md)
records retrieval locations and checksums.

For software behavior, use the default test suite and the focused command documented by
the workstream. For hardware claims, a CPU fallback is not reproduction: verify the
selected backend plus the compiled resource, timing, and known-answer records.

## Project boundary

Persistent developmental lives, developmental transfer, continual memory, embodiment,
and the true-agent specification belong to the separate `ontogenic-agent` programme.
They do not expand TikiTorch's software or systems claims.
