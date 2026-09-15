# Route-certificate experiments

These scripts own the software experiments for the unified TikiTorch manuscript. Run them
as modules from the TikiTorch repository root so provenance resolves against this
checkout.

## Entry points

| Script | Purpose |
|---|---|
| `matched_benchmark.py` | Small residual-DAG exactness, trajectory, retained-state, and timing comparison |
| `trajectory_audit.py` | Deep composed-graph gradient and update-trajectory equivalence |
| `route_stability.py` | Route stability under controlled perturbations and dtypes |
| `actual_model_benchmark.py` | Actual MorphSSM benchmark with isolated workers and incident records |
| `saved_state_audit.py` | Actual MorphSSM retained-state accounting |
| `compiled_resource_benchmark.py` | Consolidated five-arm grid, peak/proxy accounting, and frozen systems gates |
| `training_benchmark.py` | Current device-native six-arm full-step protocol, eager/compiled checkpoint policies, observed selective coverage, repeated-update parity and learning checks |
| `optimizer_diagnostic.py` | Untimed AdamW first-divergence controls, explicit epsilon sweeps, fresh-batch trajectories and offline evidence checking |
| `provenance.py` | Shared manifests, digests, environment records, and append-only helpers |

Smoke-check the actual-model configurations without starting a measured run:

```bash
python -m scripts.route_certificates.actual_model_benchmark --dry-run
python -m scripts.route_certificates.saved_state_audit --dry-run
```

Run the reference comparison with:

```bash
python -m scripts.route_certificates.matched_benchmark
python -m scripts.route_certificates.trajectory_audit
python -m scripts.route_certificates.route_stability
```

Defaults append native JSONL rows beneath `manuscript/results/route_certificates/`. Do not edit a
scientific row after creation. Preserve failed runs and fallbacks as incidents, and bind
every retained row to a clean commit, command, environment, configuration, input/model
digests, and schema version.

The historical Python/serialized protocols and current device-native protocol
must not share claim-bearing rows. `training_benchmark.py` now executes native
compiled numerical paths, but the small CPU validation is not a completed
multi-scale GPU or matched-quality study. See the unified
[experiment protocol](../../manuscript/research/PLAN.md#comparison-protocol) and
[evidence ledger](../../manuscript/research/EVIDENCE.md).

For the optimizer investigation, start with
`python -m scripts.route_certificates.optimizer_diagnostic --check`; this only
verifies the preserved source-bound record. See the
[route-training guide](../../docs/guide/route-training.md#adamw-numerical-diagnosis)
for reproduction commands and the bounded `--adamw-eps 1e-4` CPU recipe.
Default epsilon and failed trajectories remain unchanged. Benchmark expansion
and language pretraining follow review of the numerical configuration.
