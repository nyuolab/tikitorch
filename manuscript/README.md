# Route-certificate manuscript handoff

This is the root-level entry point for the discussion draft prepared for Xiang:
**Forward Traces as Exact Differentiation Certificates in Hard-Routed
Networks**.

There is deliberately only one scientific source. The canonical LaTeX lives
in [`papers/route_certificates/`](../papers/route_certificates/); this
directory supplies a zero-copy build launcher and the publication handoff.
Edit the canonical [`main.tex`](../papers/route_certificates/main.tex),
[`methods.tex`](../papers/route_certificates/methods.tex),
[`supplement.tex`](../papers/route_certificates/supplement.tex), and
[`refs.bib`](../papers/route_certificates/refs.bib). Do not copy them into this
directory.

The multiplier-free FPGA result is a separate paper. It must not be folded
into this manuscript or used to imply a software latency, memory, traffic, or
energy result.

## Build

From the repository root:

```bash
cd manuscript
latexmk -pdf main.tex
```

The PDF is written to `manuscript/build/main.pdf`; `build/` is ignored by Git.
Use `latexmk -C main.tex` from this directory to clean generated files.

Before changing a claim, check the
[`claim ledger`](../docs/research/CLAIM_LEDGER.md) and the canonical
[`paper README`](../papers/route_certificates/README.md).

## Xiang's publication brief

The paper's durable contribution is an operator-scoped theorem and executable
contract: for declared hard-routed operators, a forward trace can be a
sufficient certificate for state and additive-parameter VJPs under a declared
derivative selection. It is ordinary reverse accumulation over a smaller,
explicitly typed tape—not a new gradient rule and not generic activation-free
learning. The graph composition proof is intentionally direct; the scientific
value is the operator contract, concrete replay fields, counterexamples,
portable representation, and evidence.

Do not promote the current retained-state result into a peak-memory or speed
claim. Under a common supplied-output-cotangent boundary, the portable
certificate is 1.97--2.96x smaller than checkpoint-retained state on the three
measured MorphSSM configurations. Under the current replay interface, which
retains logits and targets on both sides, the portable ratio is 1.69--2.79x.
The current focused suite passes 38 core tests plus 37 integration,
serialization, and benchmark-hardening tests.

### Gates that must close before submission

1. **Freeze the protocol.** Pin a clean TikiTorch commit, environment, output
   cotangent/loss boundary, schemas, model/input digests, configurations,
   seeds, commands, and deterministic reducer before collecting final rows.
2. **Prove equivalence first.** For every retained comparison row, require
   matching outputs and routes plus tolerance-gated parameter gradients and
   optimization trajectories. Preserve graph breaks, fallbacks, OOMs, and
   failed runs instead of filtering them away.
3. **Complete all five target arms on the actual MorphSSM workload:** eager autograd,
   compiled autograd, checkpoint/recompute, compiled unpacked custom VJP, and
   compiled packed replay. Include packing and decoding in the measured cycle.
   The checked-in Python cascade/packed runner is scaffolding, not this result.
4. **Measure the systems question.** Report full-cycle latency, retained bytes,
   process or accelerator peak memory, recomputation events, and bytes moved at
   each scale under the same boundary accounting. Report warm-up, compilation,
   and steady-state timing separately.
5. **Regenerate every table.** Tables must come from append-only raw rows with
   provenance, not hand-edited manuscript values. Re-run the theorem,
   serializer, matched-DAG, trajectory, and actual-model audits from the pinned
   clean checkout.
6. **Finish the submission surface.** Confirm authorship, lock the
   venue style, complete the closest-prior-art matrix, audit theorem notation
   against implementation fields, and package a one-command artifact with an
   immutable release identifier.

### Venue decision

- Target **MLSys** only if the compiled packed certificate closes a meaningful
  peak-memory or memory-traffic gate while reporting the full latency cost.
- Target **TMLR** if the durable result remains the theorem, exact replay, and
  retained-state representation. That is a strong paper boundary, not a
  failed MLSys paper.
- Do not submit to preserve a deadline. Make the venue decision only after the
  five-arm reducer produces its frozen decision table.

The first deliverable should be a checked-in protocol manifest and a clean-run
reproduction report—not revised headline prose.
