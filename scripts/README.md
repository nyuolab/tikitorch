# Research and hardware scripts

These are experiment runners, diagnostics, and hardware orchestration helpers—not a
compatibility-stable command-line API. The installed public command is `tt-closure`;
the scripts in this directory are tied to their documented data, environment, and
source revision.

## Finding an entry point

| Prefix or area | Purpose |
| --- | --- |
| [`route_certificates/`](route_certificates/README.md) | Matched, replay, stability, and retained-state experiments |
| `tropical_*`, `induction_*`, `charlm_*` | Tropical model and language/induction experiments |
| `pc_*`, `bench_cascade_vs_bp.py` | Predictive-coding, cascade, scale, and local-learning studies |
| `muzero_*`, `run_*board*`, `run_*chess*`, `run_*connect4*` | Planning and board-game experiments |
| `run_*f2*`, `run_*afi*`, `run_*csynth*`, `build_fpga_local.sh` | Hardware builds, C-simulation, and board execution |
| `_*.py`, `_*.sh` | Internal helpers invoked by another runner; not primary entry points |

For library onboarding, use [`examples/`](../examples/README.md), not these research
runners. For evidence interpretation, start at [`results/`](../results/README.md) and
the [claim ledger](../docs/research/CLAIM_LEDGER.md).

## Running safely

1. Read the script docstring and `--help`; begin with a `--dry-run` or smoke option
   where one exists.
2. Record the exact Git revision, command, seed, dependencies, data digest, and hardware
   configuration. Do not infer those values from a filename.
3. Supply AWS profiles, keys, security groups, AGFIs, vendor-tool locations, and data
   paths explicitly. The repository provides no credentials or implicit AWS profile;
   some revision-bound runners retain documented S3 artifact locations as provenance
   defaults, so inspect and override them before launching in another account.
4. Write new outputs to a fresh run path. Promote only curated summaries and manifests
   to Git; keep large generated products in external artifact storage.

A safe, download-free first check is:

```bash
python -m examples.closure_workflow
python -m scripts.route_certificates.actual_model_benchmark --dry-run
```
