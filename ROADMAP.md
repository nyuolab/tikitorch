# TikiTorch roadmap

This page is the roadmap entry point. The detailed priorities and acceptance
criteria live in the [release/impact roadmap](docs/release_impact_roadmap.md);
research task design lives in the [research plan](manuscript/research/PLAN.md).
Keep implementation checklists there rather than maintaining conflicting copies.

## Current order of work

1. **Independent collaborator review.** Use the [shared handoff](manuscript/README.md)
   to reproduce onboarding and select a bounded contribution before running studies.
2. **Release foundations.** Resolve third-party RTL and retained-history review,
   verify the candidate on the supported CI matrix, and review version/API policy.
   Local tests and successful packaging do not themselves authorize publication.
3. **Numerical and systems foundations.** Resolve or explicitly scope compiled
   AdamW parity; strengthen checkpoint controls and supported body/head boundaries.
   Preserve negative results and do not widen tolerances to pass a comparison.
4. **Useful-workload evidence.** With an owner and approved protocol/resources,
   test ECG and language-pretraining quality versus complete training cost and
   measured memory. The full-learning FPGA path is a later optional capstone.

The current stopping point is collaborator review, not research execution,
cloud provisioning or release publication.

## Canonical references

- [Contributing](CONTRIBUTING.md): development setup, checks and change policy.
- [Release validation](docs/release_validation.md): runnable checks and publication gates.
- [Release/impact roadmap](docs/release_impact_roadmap.md): engineering priorities and acceptance evidence.
- [Research plan](manuscript/research/PLAN.md): ECG, language-pretraining and FPGA study sequence.
- [Claim ledger](docs/research/CLAIM_LEDGER.md): what the evidence permits us to claim.

TikiTorch remains an independent PyTorch companion, not a complete PyTorch
replacement or a general FPGA compiler. The single active manuscript is in
`manuscript/`; superseded drafts are in `manuscript/archive/`. Agent/organism
research remains outside this repository's scope.
