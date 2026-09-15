# TikiTorch roadmap

This page is the roadmap entry point. The detailed priorities and acceptance
criteria live in the [release/impact roadmap](docs/release_impact_roadmap.md);
research task design lives in the [research plan](manuscript/research/PLAN.md).
Keep implementation checklists there rather than maintaining conflicting copies.

## Current order of work

1. **Close the quality gate for FPGA-first language pretraining.** The existing
   D512 image passed 64 fresh updates with complete checkpoint equivalence to
   C-simulation. The later CPU development recipe beats bigram on mean loss but
   trails GRU in every tested order. Review the frozen-readout conditioning
   diagnostic through the [current handoff](manuscript/README.md#handoff-for-xiang)
   before selecting the next bounded experiment. CPU/GPU speedups are not a
   prerequisite; the CPU recipe and deployed integer SignSGD remain distinct.
2. **Preserve the completed cloud pilot and repeat parity after recipe changes.**
   Retain source/image identities, negative integration results and teardown
   evidence. Future work uses the owner's cumulative AWS authorization below
   $1,000, conservative reservations and confirmed teardown; that authorization
   does not bypass the quality gate or authorize an automatic new run.
3. **Profile, then scale the hardware.** Demonstrate learning before new synthesis;
   remove measured bottlenecks and move beyond the nearly full on-chip staging
   design toward HBM/DDR-backed models. The current 1.2M-character-model prototype
   is not yet an LLM pretraining result. ECG remains secondary.
4. **Independent review and release foundations.** Use the
   [shared handoff](manuscript/README.md) for onboarding. Resolve third-party RTL
   and retained-history review, verify the supported CI matrix and candidate API.
   Local tests and packaging do not themselves authorize publication.

The FPGA research continuation and bounded AWS spending are authorized. New
resources still require validated inputs and cost/termination controls, and
spending above $1,000 needs further approval. Release publication is not implied.

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
