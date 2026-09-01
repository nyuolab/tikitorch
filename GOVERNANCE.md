# Project governance

TikiTorch is maintained as an independent open research project for the PyTorch
ecosystem. It is not an official PyTorch project or endorsement.

## Decision making

Maintainers review contributions, releases, compatibility decisions, and changes
to the public claim surface. Routine decisions are made by review consensus. When
consensus is not available, the repository owner makes the final call and records
the rationale in the pull request. CODEOWNERS requests review but only a separately
configured repository ruleset or branch-protection rule can enforce it.

Changes to closure semantics, route-certificate schemas, serialized formats, or
claim-bearing evidence require tests, documentation, and review from the relevant
code owner. Research results do not become library guarantees merely by being
committed to `results/` or described in a paper draft.

## Project surfaces

- `tikitorch/` and the documented API are the reusable software surface.
- `scripts/`, `results/`, and `papers/` are research artifacts with narrower
  reproducibility and compatibility promises.
- AWS/Xilinx toolchains and PyTorch remain independently governed dependencies.

## Releases

Only maintainers may create tags or publish distributions. A release must satisfy
[RELEASING.md](RELEASING.md), preserve the evidence boundaries in the research
claim ledger, and avoid implying an official relationship with PyTorch.
