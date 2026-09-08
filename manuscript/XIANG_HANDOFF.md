# Unified TikiTorch discussion handoff for Xiang

The active draft is [TikiTorch: Route-Sufficient Differentiation for Auditable
Tropical Learning](README.md). The prior two-paper portfolio is
superseded by one TikiTorch manuscript; the old sources remain historical.

The immediate next step is to see whether a student wants to take ownership,
after local library cleanup and onboarding. The proposed
[research plan](RESEARCH_PLAN.md) and
[student handoff](STUDENT_HANDOFF.md) are written down; neither
constitutes a launched experiment or approved compute budget. Research results
now live separately under [`manuscript/results/`](results/README.md).
The superseded drafts are preserved in [archive/](archive/README.md).

## The scientific question

Can native additive hard operators make both the state required for reverse
differentiation and the arithmetic of a learning recipe explicit enough to
deliver useful resource-constrained training?

The paper joins typed route recording, first-order replay, ordinary PyTorch
integration and fail-closed auditing. It gives scoped proofs and counterexamples,
including why deterministic local ties do not imply a global Clarke subgradient.
It credits prior morphological backpropagation, backward-state encodings and
complete multiplication-free training.

## What the evidence says

- A fresh current-code three-cell CPU audit passes exact logits/routes and
  selected-VJP checks. Device payloads are 2.04–2.98× smaller than an explicitly
  shape-defined checkpoint boundary set. This is not measured peak-memory or
  latency superiority.
- The previous portable-tape table remains appendix-only, with its own legacy
  source identity and accounting.
- FPGA evidence is a scoped feasibility case: matching D512 observable fields
  and separate fabric-trained MNIST weights. Repeated windows, unreconciled
  evaluators, numerical recipe differences and missing physical comparisons
  are disclosed.
- Native compiled AdamW remains a strict update-parity failure on the tested
  stack; passing a separate SGD recipe is not an AdamW-equivalence claim.

## Decisions for author review

1. Does the integrated contract plus a useful real-task resource frontier make
   a sufficiently strong ICML contribution? The direct chain-rule proof alone
   is not the novelty.
2. Who can own the first bounded pilot? The agreed directions are ECG
   classification and language pretraining, followed by a possible full-learning
   FPGA capstone. Confirm task-specific baselines, protocol and resources before
   beginning; the student need not undertake all three tracks.
3. Which author contributions, institutional approvals and artifact rights
   must be confirmed before release and submission?

Read the [readiness checklist](ICML_READINESS.md),
[experiment protocol](EXPERIMENT_PROTOCOL.md) and
[codebase/release roadmap](../docs/release_impact_roadmap.md) after the PDF.
The current official style is a provisional 2026 template, not a verified
next-cycle submission guarantee. No manuscript has been submitted.
