# Forward traces as exact differentiation certificates

> Historical draft, superseded on 2026-09-08 by the unified
> [TikiTorch manuscript](../../README.md). The scientific sources below
> are preserved for provenance, not maintained as a separate submission. Former
> instructions requiring separate papers, old test counts and venue choices are
> historical. Make new scientific edits in `manuscript/`.
> This directory moved from `papers/route_certificates/` to
> `manuscript/archive/route_certificates/`; the remaining narrative describes
> the former draft, including its old "canonical" and separate-paper wording.

This directory is the canonical TikiTorch manuscript workspace for the
route-certificate result. It is deliberately separate from the FPGA learning
paper and from any continual, developmental, or embodied-agent programme.

Status: professional discussion draft; theorem, scoped exactness,
implementation migration, and boundary-matched retained-state results are
written. The migrated focused suite passes 75 tests in the current worktree
(38 core theorem/operator tests plus 37 integration, serialization, and
benchmark-hardening tests). The compiled five-arm systems comparison is not
complete, so the draft does not claim a latency, allocator-peak, traffic,
energy, or whole-model activation-memory advantage. Authorship and venue style
remain deliberately unresolved.

## Files

- `main.tex` — standalone paper, tables, current results, limitations, and
  explicit artifact gates.
- `methods.tex` — operator semantics, packing, exactness tests, state
  accounting, and the target five-arm protocol.
- `supplement.tex` — theorem proof, tie/stability arguments, information bound,
  counterexamples, accounting sensitivity, and submission checklist.
- `refs.bib` — bibliography used by this draft.
- `.latexmkrc` — sends generated PDF and auxiliary files to ignored `build/`.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The local `.latexmkrc` writes the PDF and auxiliary files to `build/`, which is
ignored by Git. Clean them with `latexmk -C main.tex`.

## Claim boundary

The paper may claim:

- an exact selected-VJP certificate theorem for finite executed graphs whose
  requested local transpose actions are trace-sufficient;
- state and additive-parameter sufficiency for declared hard max/min-plus
  operators with an explicit tie contract;
- composition through fan-out, residual addition, shared parameters, and a
  stated finite recurrent unrolling;
- exactness in the registered reference tests and matched trajectories; and
- configuration-specific retained-state savings on the measured MorphSSM
  scale ladder, with boundary accounting disclosed.

The paper must not claim:

- a new gradient or learning rule;
- generic activation-free training or whole-model coverage;
- route sufficiency for dense parameter outer products, smooth attention,
  ordinary normalization, loss construction, or optimizer state;
- an optimized latency, process/allocator peak, memory-traffic, energy, or
  deployment advantage before the corresponding gate passes; or
- a hardware result. Multiplier-free FPGA learning is a separate manuscript.

The existing `tikitorch.research.pc_convrnn.RouteCertificate` is a useful
target-free support/multiplicity inventory. It is not, by itself, the complete
portable VJP certificate defined in this manuscript. The submission artifact
should unite the typed replay fields, packed schema, and public route inventory
behind one versioned TikiTorch interface.

## Preserved evidence used by the draft

The numeric draft was reconstructed from preserved, append-only records and
the previous route-trace manuscript, then reframed with TikiTorch as the
canonical software and artifact repository. No agent-learning result is used.

- Historical TikiTorch code identity for the actual-model audit:
  `ee3c775281b6046781144b0768a050ac26dfbc19`.
- Actual-model saved-state run:
  `529c45f4-7274-4999-b35c-c59d1db917e6`.
- Selected versioned matched-DAG run:
  `5f6b7714-1a69-4bdc-b412-c1f0d768be2a`. The same file preserves 12
  pre-schema rows without a run identifier; those legacy rows are not used for
  the draft's 3.3--3.8x serializer range.
- Route-stability runs:
  `72050d96-60b1-43ab-96bc-3a6b50259efb` and
  `82d15dd7-6c77-4158-b311-c40b6ba96894`. Each is a complete 42-row run and
  independently yields the reported total of 2,353 conservative certified
  cases, so the draft value does not select between them.
- Cells: `(d=64,L=2,T=96)`, `(128,4,96)`, and `(256,4,192)`, batch one,
  vocabulary 38, hard routing, FP32.
- At a common supplied-output-cotangent boundary, portable serialized
  certificates are 1.97x, 2.74x, and 2.96x smaller than checkpoint-retained
  layer/embedding state. Payload-only coding ratios at that same boundary are
  2.04x, 2.81x, and 2.98x.
- Portable serialized sizes: 37,431, 89,799, and 332,173 bytes.
- The current MorphSSM replay derives its output cotangent from retained logits
  and targets. Adding those tensors symmetrically to both methods gives
  loss-inclusive portable ratios of 1.69x, 2.48x, and 2.79x.
- The middle payload is 87,588 bytes, or 912.4 bytes/token.
- Focused current suite: 38 passing core tests plus 37 passing integration,
  serialization, and benchmark-hardening tests. The historical claim-bearing
  record covers the 38-test core.
- Small matched residual-DAG audit: maximum FP32 parameter-gradient difference
  from eager autograd `4.77e-7`; maximum parameter difference after five SGD
  steps `2.98e-8`; serialized state 5.0–6.3x below checkpoint nonparameter
  state; prototype Python packing 3.3–3.8x slower than eager autograd.

These records have been migrated into TikiTorch but must be regenerated under a
clean, archived submission commit before the manuscript is artifact-ready.
Until then, the values are evidence-backed draft text rather than a
clean-checkout reproduction. The migrated archive also retains repeated
historical runs and incident rows. Draft values are tied to the explicit run
identifiers above, never selected as the best observation. The submission
reducer must reject ambiguous duplicate successful cell keys unless a manifest
marks one run as superseding another, while preserving all incidents.

## Required artifact layout

The intended canonical layout is:

```text
tikitorch/autodiff/                      # schema, reference operators, replay
tests/test_route_*.py                    # graph, gradient, and composition tests
tests/test_*sufficiency.py               # state and certificate sufficiency tests
scripts/route_certificates/              # audit and five-arm runners
manuscript/results/route_certificates/              # append-only raw rows and manifests
manuscript/archive/route_certificates/   # this historical manuscript
```

Do not create placeholder scientific rows. Each claim-bearing result should
bind the command, TikiTorch commit, worktree digest, environment, device, seed,
complete configuration, model/input digests, schema version, and incident
status.

## Submission gates

1. Re-run the migrated theorem tests, packed serializer, matched DAG audit, and actual
   MorphSSM saved-state audit from a clean archived TikiTorch commit.
2. Freeze one output-cotangent/loss boundary and report the boundary sensitivity
   for every comparator.
3. Complete the actual MorphSSM target five-arm comparison:
   eager autograd, compiled autograd, checkpoint/recompute, compiled unpacked
   custom VJP, and compiled packed replay.
4. Require output, route, gradient, and optimization-trajectory equivalence
   before retaining a performance row.
5. Measure full-cycle latency, process or accelerator peak memory, retained
   bytes, recomputation events, and bytes moved. Include packing and decoding;
   retain OOMs, graph breaks, and compiler fallbacks.
6. Regenerate every table from raw append-only results with a deterministic
   reducer in a clean checkout.
7. Confirm authorship, select the venue style, and run a final
   claim/citation/artifact audit.

The intended first venue is MLSys if the compiled systems gates close. TMLR is
the appropriate rolling fallback if the result remains primarily theorem,
exactness, and representation.
