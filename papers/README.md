# TikiTorch publication map

TikiTorch is the **PyTorch-shaped tropical deep-learning infrastructure**. Its
publication programme is intentionally split into independent, auditable works.
Ontogenic agents, developmental transfer, continual memory, embodied plasticity,
and the true-agent specification belong in the sibling `ontogenic-agent`
repository, not in these papers.

The canonical wording authority is the [research claim
ledger](../docs/research/CLAIM_LEDGER.md).

## Paper 1 — Forward traces as exact differentiation certificates

Workspace: [`route_certificates/`](route_certificates/)

Root-level student handoff and build entry point:
[`../manuscript/`](../manuscript/). The scientific source remains canonical in
`papers/route_certificates/`; the handoff does not maintain a second copy.

**Contribution:** a scoped theorem and executable interface showing when a hard
forward trace plus declared auxiliary fields is sufficient to replay selected
state and additive-parameter VJPs; exactness tests; and configuration-specific
MorphSSM retained-state measurements.

**Owns:**

- theorem, tie contract, supported-operator inventory, and counterexamples;
- typed and packed certificate schemas;
- route-replay exactness and trajectory equivalence;
- retained-state accounting and its boundary sensitivity; and
- the compiled five-arm software comparison.

**Does not own:** FPGA throughput, silicon learning, multiplier-free energy, a
new gradient, generic activation-free learning, or ontogenic-agent behavior.

**Submission route:** MLSys if the compiled latency/memory/traffic gates close;
TMLR if the durable contribution remains theorem, exactness, and representation.

**Before submission:** re-run from a clean archived TikiTorch commit; freeze the
loss/output-cotangent boundary; compare eager autograd, compiled autograd,
checkpoint/recompute, compiled unpacked VJP, and compiled packed replay; require
equivalent routes, outputs, gradients, and optimization trajectories; measure
full-cycle latency, retained bytes, process/device peak, recomputation, and
bytes moved.

## Paper 2 — Multiplier-free learning on FPGA

Workspace: [`multiplier_free_fpga/`](multiplier_free_fpga/)

**Contribution:** co-design and physical validation of complete
forward--loss--backward--update loops built from compare, add, shift, mux, and
lookup operations, including exact D512 execution and held-out learning from a
small fabric-trained classifier.

**Owns:**

- closure-to-HLS evidence and the distinction between multiplier-free and
  zero-DSP;
- AWS F2 kernel architecture and bit-exact known-answer tests;
- D512 fused learning-loop execution;
- fabric-only MNIST learning;
- multiplier-free LM and secondary inference artifacts; and
- timing, throughput, utilization, and power measurements when properly gated.

**Does not own:** the route-certificate retained-state theorem, ontogenic-agent
learning, a general TikiTorch compiler, GPU superiority, or an energy win before
matched measurement.

**Submission route:** a hardware venue such as ACM FPGA, FPL, or FCCM after the
systems gates close. Do not submit merely to preserve a deadline.

**Before submission:** run disjoint-stream D512 learning with held-out quality;
archive final HLS and routed reports; require timing closure with margin; add an
iso-model/iso-update CUDA arm; measure full-card FPGA power; freeze MNIST
preprocessing and repeat with seeds and matched controls.

## Paper 3 — TikiTorch software

Workspace: not opened yet.

**Contribution:** the library as reusable research infrastructure: PyTorch-shaped
tropical operators, models, optimizers, closure auditing, serialization,
documentation, and hardware/export seams.

**Owns:** API design, usability, software architecture, testing, documentation,
release engineering, and examples. Research-paper numbers may demonstrate use
but should be cited rather than republished as software-paper novelty.

**Submission route:** JOSS after the repository has sufficient public history,
green supported-platform CI, a tagged/archived release, stable documentation,
and evidence of external research use.

## Result ownership

| Result | Canonical paper | May be background elsewhere? |
|---|---|---|
| Selected-VJP certificate theorem and packing | Route certificates | Yes, one-paragraph citation |
| 2.45--3.07x retained-state payload result | Route certificates | Yes, with the accounting boundary |
| D512 fused F2 execution | FPGA | Yes, as hardware feasibility only |
| Fabric-trained MNIST held-out accuracy | FPGA | Yes, as hardware feasibility only |
| 2.4230 bpc F2 forward inference | FPGA | Yes, as deployment context |
| Library API, audit surface, docs, release | Software | Yes, as artifact description |
| Developmental transfer or true-agent evidence | Neither | No; belongs in `ontogenic-agent` |

## Portfolio rules

1. Do not build a combined “everything” manuscript. Each paper needs one claim,
   one comparator family, and one artifact boundary.
2. Do not report the same measurement as independent evidence in two papers.
   Cite the canonical paper and state why the result is background.
3. Keep negative results visible: route packing is not yet a latency win; D512
   repeats a resident window; MNIST is small; replica WNS is negative; and
   full-card energy is unmeasured.
4. Every table must be generated from append-only raw rows bound to a clean
   TikiTorch commit, environment, command, configuration, and reducer.
5. Draft prose cannot promote a claim beyond the [claim
   ledger](../docs/research/CLAIM_LEDGER.md).
6. Ontogenic-agent experiments consume a tagged TikiTorch dependency. They do
   not live in this repository and do not broaden its software or hardware
   claims.

## Immediate order of work

1. Archive the current TikiTorch research branch and working artifacts remotely.
2. Make the route-certificate integration and ordinary CI green.
3. Close the route paper's five-arm systems experiment.
4. Close the FPGA paper's disjoint-stream, timing, CUDA, and power gates.
5. Cut a stable TikiTorch release; pursue the software paper after adoption.
