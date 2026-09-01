# Two-manuscript discussion handoff for Xiang

Status: viable technical-review drafts, not submission-ready manuscripts.

The two papers now have distinct claims, evidence boundaries, and submission
gates. They should be reviewed separately rather than combined into a single
TikiTorch paper. The canonical wording authority for both is the
[research claim ledger](../docs/research/CLAIM_LEDGER.md).

## Draft 1: forward traces as differentiation certificates

- Source and build: [`route_certificates/`](route_certificates/)
- Root handoff: [`../manuscript/`](../manuscript/)
- Durable contribution: an operator-scoped contract showing when a typed hard
  forward trace suffices to replay selected state and additive-parameter VJPs,
  plus exactness tests and a portable packed representation.
- Current measured result: portable certificates are 1.97--2.96x smaller than
  checkpoint-retained state at a common supplied-cotangent boundary across
  three MorphSSM configurations; the loss-inclusive range is 1.69--2.79x.
- Principal open gate: complete the frozen five-arm compiled comparison and
  measure full-cycle latency, retained state, peak allocation, recomputation,
  and bytes moved.
- Venue decision: MLSys only if the compiled systems gate closes; otherwise
  TMLR for the theorem, executable contract, and representation result.

Questions for discussion:

1. Is the operator contract and typed tape a sufficiently sharp conceptual
   contribution, or should the paper lead with the portable representation?
2. Which five-arm metric should decide the MLSys-versus-TMLR route?
3. Who should be included as authors, and what contribution evidence is still
   needed before fixing the order?

## Draft 2: multiplier-free learning on FPGA

- Source and build: [`multiplier_free_fpga/`](multiplier_free_fpga/)
- Durable contribution: a recipe-level closure contract and on-silicon
  execution of a fused forward--loss--reverse--update transition, paired with
  a separate small held-out learning result from fabric-only updates.
- Current measured result: the archived observable record of a 30-update D512
  run matches C simulation bit for bit; a 2,000-update resident-window run
  sustains 612 token/s. A separate 16-by-16 MNIST model reaches 74.75% held-out
  accuracy (74.13% under an independent evaluation path).
- Principal open gates: disjoint-stream learning, frozen MNIST reproduction,
  timing-closed headline builds, an iso-model CUDA arm, full-card power, and an
  immutable artifact bundle.
- Venue decision: ACM FPGA, FPL, or FCCM only after the systems comparison and
  artifact gates close.

Questions for discussion:

1. Is feasibility of the complete multiplier-free state transition, with the
   current scoped exactness evidence, enough to anchor the paper?
2. Which learning-quality experiment is the minimum credible successor to the
   repeated-window D512 run?
3. Should the preliminary D128 replication ladder remain context or move
   entirely to supplementary material until timing closes?

## Suggested review order

1. Read each abstract and the red/open-gate language before reviewing prose.
2. Confirm that the durable contribution survives even if every open systems
   comparison is unfavorable.
3. Agree on authorship, target audience, and the single decision experiment
   for each paper.
4. Freeze protocols before collecting new headline numbers; failed runs and
   negative comparisons remain part of the record.

Build commands and generated-PDF locations are documented in each paper's
README. Generated PDFs are review artifacts and are intentionally not tracked.
