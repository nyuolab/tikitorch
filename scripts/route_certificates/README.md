# Route-certificate experiments

These scripts own the software experiments for the route-certificate paper. Run them
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

Defaults append native JSONL rows beneath `results/route_certificates/`. Do not edit a
scientific row after creation. Preserve failed runs and fallbacks as incidents, and bind
every retained row to a clean commit, command, environment, configuration, input/model
digests, and schema version.

The current Python implementations establish exactness and accounting. They are not the
paper's pending compiled five-arm systems benchmark.
