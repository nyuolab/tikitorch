# Codebase implementation audit — 2026-09-08

This audit covers the codebase improvement pass on top of `ea84903`, not a
manuscript revision or authorization to publish. Changes remain local and
uncommitted. No manuscript, historical experiment result or hardware report was
rewritten.

## Implemented

- Fail-closed executed-operation auditing with safe/unsafe/unknown classifications,
  versioned coverage metadata and tensor-only local-learning diagnostics.
- Shape-safe tiled tropical reductions, bounded backward intermediates, consistent
  first-index routes/top-k ties and validated temperatures.
- Coherent public backend dispatch, atomic replacement, capability enforcement,
  and explicit unsupported-backend/compiler boundaries.
- Device-resident packed/unpacked route tapes, streaming hard-MorphSSM forward,
  loss-independent VJPs and first-order PyTorch autograd integration.
- A separate full-step benchmark covering eager, compiled, checkpointed, unpacked
  and packed training; matched final-state checks, bounded learning gates,
  subprocess failure handling and source-bound manifests.
- Runner hygiene, offline shell tests, native compiler CI, manual existing-runner
  GPU validation and a fail-closed release provenance inventory/gate.

Independent reviews covered operator/backend correctness, audit semantics and
benchmark/tape claim boundaries. Reported implementation blockers were fixed and
regression-tested before the final validation below.

## Local validation

Environment: macOS arm64, Python 3.11.5, PyTorch 2.10.0; no CUDA device.

| Check | Result |
| --- | --- |
| Default full suite | 878 passed, 8 skipped, 8 slow/native-kernel tests deselected |
| Explicit native Inductor/route benchmark suite | 15 passed, 1 CUDA skip |
| Ruff and `git diff --check` | Passed |
| Strict documentation build and runnable closure onboarding | Passed |
| Isolated source archive/wheel build and Twine metadata check | Passed |
| Installed wheel outside the checkout: import, route backward, CLI | Passed |
| Provenance inventory integrity | Passed; publication gate remains blocked |

The final full-step CPU protocol smoke passed all seven execution cases (three
checkpoint policies), comparing measured final gradients/parameters/loss to eager
after five updates. Configuration: float32, width 16, one layer, vocabulary 11,
batch 2, sequence 12, seed 0, momentum SGD, one warmup and three timed repetitions.
The five-step route trajectory and 25-update synthetic held-out learning gate
also passed; executable source hashes were unchanged during the run.
Run ID: `14f00d16-e3ea-4622-9c80-c03d30bb259f`. This tiny CPU smoke is not evidence
of competitive task quality, GPU capacity, general speedups or hardware savings.

Reproduce the protocol using the commands in [route training](guide/route-training.md).
The supported interpreter/dependency matrix is configured in CI; this local audit
does not claim that remote workflows or all matrix combinations were executed.

## Remaining boundaries

- The native Inductor **AdamW** comparison still fails the strict update-parity
  gate on this machine, for ordinary compiled autograd as well as packed routes
  (about 0.002 maximum parameter discrepancy after two updates in the small test).
  This result remains visible and inconclusive; tolerances were not loosened.
  Passing the separate momentum-SGD protocol does not establish AdamW equivalence.
- GPU tests and physical hardware measurements require an existing authorized
  runner/device. No cloud resource was provisioned or workflow dispatched.
- Four AWS-derived RTL provenance records and retained Git-history review remain
  unresolved. The [publication gate](release_validation.md) intentionally exits
  nonzero. Building the package does not clear distribution rights.
- Real-task, multi-seed, matched-quality experiments remain research work. The
  optional corpus protocol records held-out metrics without claiming quality
  superiority. Manuscript positioning should reflect these evidence boundaries.
