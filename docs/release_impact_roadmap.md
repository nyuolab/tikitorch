# TikiTorch: release and scientific-impact roadmap

Review date: 2026-09-08, after the codebase audit and combined-manuscript pass.
This is a prioritized plan, not a claim that GPU, hardware or publication gates
have run. It separates release authority, implementation maturity and research
evidence so one cannot substitute for another.

The strongest opportunity is a usable PyTorch companion in which hard
operators expose a compact exact derivative record and an explicit arithmetic
contract. The immediate goal is to make that opportunity independently
reproducible on a real training workload. Expanding the model zoo is less
valuable until that central result is demonstrated.

## First: resolve publication authority and freeze a candidate

**R1 — Provenance is the release blocker, not a cosmetic cleanup.**

The four AWS-derived RTL records and retained-history review in
[`release_provenance.json`](release_provenance.json) remain unresolved. A
maintainer must choose documented upstream permission/notices, an independently
implemented replacement, or deliberate exclusion from all intended release
artifacts/refs. Do not infer authorization to remove source or rewrite history.

Acceptance: reviewed evidence and exact digests for every resolved record,
named human review of all intended public refs, and successful
`python scripts/check_release_provenance.py` without an inventory-only waiver.
Archive the chosen candidate commit and distribution contents. Building a
wheel or making a repository public does not itself satisfy this gate.

**R2 — Treat supported environments as tested release configurations.**

The local audit is macOS arm64/Python 3.11/PyTorch 2.10. CI defines a Python 3.9–3.12
matrix, minimum dependencies and native CPU compilation; their configuration
is not evidence that the remote jobs ran. Package metadata remains 0.2.0/Alpha.

Acceptance: green candidate CI matrix; one source archive and wheel built from
the same clean revision; installed-wheel smoke outside the checkout; archived
dependency/platform manifests; a reviewed version/changelog/API compatibility
decision. Publish a support table distinguishing eager reference, compiled CPU,
compiled CUDA and FPGA rather than giving one undifferentiated support badge.

## Highest scientific priority: demonstrate useful training, not only smaller codes

**S1 — Run the actual five-arm protocol on an existing authorized GPU.**

`scripts/route_certificates/training_benchmark.py` now has real eager, compiled,
checkpoint, unpacked and packed paths. The historical saved-state table is a
portable representation comparison against a declared checkpoint set, not an
allocator measurement. The current seven-case CPU SGD smoke is intentionally
tiny. Neither establishes the central practical benefit.

Before ranking arms, strengthen the checkpoint baseline: the current checkpoint
policies execute eagerly while the compiled-autograd/route arms compile. Add
compatible compiled-checkpoint variants or explicitly report unsupported cells.
The selective save list includes `max.dim`/`min.dim`, while current hard
projection reductions use `amax`/`amin`; instrument which operations are actually
saved and recomputed instead of assuming the policy covers them.

Acceptance: source-bound CUDA runs over preregistered widths, lengths and batch
sizes; identical initialization, data and update count; best successful
checkpoint policy within a declared memory budget; validated final gradients,
parameters and losses; full-step latency, compile startup, allocated/reserved
peaks, and all OOM/timeout/parity failures preserved. Report maximum *tested*
feasible workloads, not extrapolated capacity. Measure packing/decoding and
retained state separately from transient peaks. Use the existing manual GPU
workflow only after approval of a trusted runner; no cloud provisioning is
implied by this plan.

Expand actual-device conformance beyond the current CUDA smoke fixtures:
parameterize route/tie identity, empty and boundary shapes, supported dtypes,
and repeated optimizer trajectories on the claimed accelerator stack.

**S2 — Resolve or explicitly scope native AdamW parity before making it the default comparison.**

The audited default native-Inductor AdamW comparison fails strict update parity
on the local stack, including ordinary compiled autograd. Small near-zero
gradient differences can be amplified by the optimizer. Switching to momentum
SGD is a legitimate separate configuration, not a fix or an AdamW pass.

Acceptance: minimal deterministic reproducer identifying the first differing
operator/gradient/update, recorded dtype/compiler/device versions, and either a
demonstrated correction or an explicit supported-configuration limitation.
Choose numerical tolerances and optimizer epsilon before looking at benchmark
rankings; retain strict route identity. Keep failing rows and test the decision
in CI. Do not broaden tolerances merely to turn the comparison green.

**S3 — Add a frozen real-task, multi-seed quality/cost study.**

`training_benchmark.py --corpus` currently offers an 85/15 contiguous character
split, training-only vocabulary and a GRU comparison. This is a useful protocol
hook, not a parameter-, budget- or quality-matched study. Its synthetic
successor-token regression is valuable for CI but not external task evidence.

Acceptance: legally redistributable dataset manifest with input hashes,
train/validation/test boundaries, tokenizer/preprocessing and contamination
checks; at least three declared seeds; a fixed model-selection budget;
checkpoint selection on validation only; final held-out metric with per-seed
results and uncertainty. Compare the same hard model across differentiation
arms to isolate systems effects, and separately compare a tuned conventional
baseline at matched parameter/training budget and quality. Include an
unpacked-versus-packed ablation, checkpoint policies, hard-versus-soft training
where relevant, and accuracy/cost Pareto curves. Preserve negative outcomes.

An impactful result can be a measured capacity frontier at equal quality or a
useful memory/latency trade-off; it does not have to be a universal speedup.

## Engineering priorities after the core measurement

**E0 — Remove contradictory closure labels from research entry points before release.**

Addressed in the student-handoff cleanup: `tikitorch/research/nanogpt.py` now
uses the canonical versioned report with no operation waivers. It audits the
configured batch/device/training temperature separately from a hardened
deployment candidate, rejecting soft schedules and untraced accumulation
windows under `--strict`. The legacy `pc_convrnn --strict-audit` receipt is
explicitly labeled as a batch-2 hardened cascade/optimizer probe, not proof
of the configured training run. Historical logs remain unchanged.

Acceptance: route current research commands through the canonical report, or
explicitly label waived/historical checks as non-certifying. Add regression
tests that soft routing, waived operations and unrecognized numeric operations
cannot print an unqualified closure pass. Keep raw historical logs unchanged.
This is release-facing correctness work, not a request to rewrite past results.

**E1 — Make the route contract an extension interface, not just a MorphSSM bridge.**

`morphssm_tape._assert_hard` deliberately accepts a narrow exact-class model
configuration and rejects aliases, hooks, custom backend routes and recurrent
scans. This is safe. It also means the graph-composition theorem is broader
than the stable integrated API. `pc_convrnn.RouteCertificate` describes support,
whereas portable/device tapes provide actual reverse state; the distinction
must remain explicit.

Acceptance: a versioned operator registration contract specifying forward
route, selected state/parameter VJP, tie/NaN behavior, auxiliary fields,
parameter sharing and device/dtype/compiler support; conformance tests that
reject incomplete contracts; a small non-MorphSSM residual DAG and a shared
parameter case through the public interface. Keep the existing narrow path
until those tests pass. Define first-order and serialization compatibility
policy before promising arbitrary-model support.

**E2 — Profile the portable tensor kernels before adding specialized kernels.**

The current hard reductions bound intermediates and the device tape packs
bounded tensors; these are strong correctness baselines. The explicit
first-winner implementation was required for native-compiler route parity.
Stable top-k sorting and per-field packing/decoding may still dominate on
accelerators. No current measurement identifies the main bottleneck.

Acceptance: operator and end-to-end profiles on the S1 workloads, measured
temporary allocation and launch counts, and documented bottleneck shares.
Only then add a fused CUDA/Triton or alternative backend implementation for
the expensive primitives. Require the same route, gradient, exceptional-value,
FP16/FP32/FP64 and empty/boundary-shape contracts; demonstrate a whole-step gain
without losing the reference fallback. Do not optimize only payload size while
increasing peak or decoding time.

**E3 — Stabilize the FPGA ABI and evidence bundle separately from Python dispatch.**

`tikitorch/fpga/backend.py` correctly advertises a fail-closed whole-model seam,
not per-op FPGA dispatch; safe native execution requires a separately supplied
runner. `BUILD_NATIVE.md` describes ABI, shape and buffer contracts. This is
not yet a turn-key compiler from a normal TikiTorch model to a new bitstream.

Acceptance: for one supported kernel, a versioned manifest binds source/image
identity, shape, numerical recipe, buffer layout, native symbols and transport
to a one-command host-reference KAT. Validate wrong shape/image/ABI/transport
rejection and distinguish LOCAL reference from real silicon availability.
Provide pinned toolchain build instructions and exact report checksums. Do not
claim that successful Python fallback means FPGA execution.

**E4 — Turn reproducibility and adoption into maintained product surfaces.**

The tiny runnable workflow is now a good entry point. Next, reduce the distance
from that workflow to an independently reproduced contribution.

Acceptance: one compact notebook/script showing ordinary `loss.backward()`
route training, inspecting route bytes, running a deliberately non-closed
recipe through the auditor, and comparing against eager/checkpoint on a fixed
small task. Execute it in CI. Add a documented support/error matrix,
serialization round-trip across supported versions, migration/deprecation
policy, contributor issue templates and a minimal custom-operator example.
Bind paper tables to raw hashes (`manuscript/generate_tables.py --check`) and
release artifacts to a reviewed tag; register archival metadata/DOI only when
authorized. Have an external contributor reproduce the workflow from a clean
environment before describing the package as release-ready.

## Hardware experiments: required only for the claims retained

For a stronger FPGA learning claim, first freeze the MNIST input tensors and
preprocessing, run the missing one-step full-training KAT, reconcile the 74.75%
and 74.13% evaluators, and repeat seeds against the same software recipe. Rebuild
or explicitly retain the historical normalization/target compensation.

For D512 language learning, replace the repeated resident window with disjoint
streaming windows, periodic full-state/reference checkpoints, a held-out metric
and no-update control. Archive the exact final HLS/routed reports. For any
systems headline, require timing closure with margin, a matched CUDA workload,
and synchronized full-card energy with uncertainty. Negative-WNS replica
results and core-rail power remain appendix feasibility context, not speed or
energy superiority.

A focused TikiTorch software submission can proceed without claiming FPGA
superiority. The strongest near-term investment is S1–S3 plus a stable public
route interface; expanding hardware claims before those results would spread
the contribution too thin.

## Suggested decision sequence

1. Resolve R1 with maintainers; freeze R2 candidate and empirical protocols.
2. Keep E0 regression coverage green, strengthen the checkpoint baseline, then complete S1/S2 on an
   approved existing GPU, retaining failures.
3. Complete S3 and choose the strongest supported memory/quality/cost claim.
4. Implement E1/E2 against the measured need; retain E3 as a clearly scoped
   experimental backend unless its full acceptance bundle exists.
5. Finish E4, external reproduction, author review and venue-compliant artifact
   packaging; then authorize the public release and submission separately.

No repository visibility change, push, history rewrite, paid run, new release
or manuscript submission was performed by this review.
