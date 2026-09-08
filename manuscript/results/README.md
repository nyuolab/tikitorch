# Research results and evidence

This tree contains curated outputs from TikiTorch experiments. It is not part of the
stable Python API, and a file's presence does not promote an exploratory observation
into a release claim. Start with the
[claim ledger](../../docs/research/CLAIM_LEDGER.md), which records the supported wording,
scope, status, and evidence path for claim-bearing results.

## Map

| Area | Directories and entry points |
| --- | --- |
| Route-certificate replay | [`route_certificates/`](route_certificates/README.md) |
| FPGA builds and execution | `fpga_silicon/`, `fpga_training_rig/`, `a2_power/` |
| Language-model experiments | `lm_shakespeare/`, `lm_gpu_baseline/`, `tropical_attention_routing/` |
| Predictive coding and local learning | `predictive_coding/`, `pc_convrnn/`, `pc_depth_traction/`, `forward_forward/` |
| Planning and board-game scaling | [`board_scaling/`](board_scaling/README.md), `chess_gpu_baseline/` |
| Diagnostic sweeps | `capacity_probe/`, `closed_benchmarks_remote/`, root `confirm_*` and `bakeoff_*` summaries |
| Publication graphics | [`graphics/`](graphics/README.md) |

Many directories mix confirmatory records with diagnostics, failed runs, or design
notes. Read the nearest README/report and its provenance fields before interpreting a
number. Configuration-specific FPGA reports establish only the named build and
workload; software closure reports do not establish routed hardware resource use.

## Adding evidence

- Keep small, reviewable summaries, machine-readable result rows, checksums, and
  provenance manifests in Git.
- Store large checkpoints, raw vendor build products, and repeated binary outputs in
  the project's external artifact store. Commit a stable URI, digest, byte size,
  generating command, source revision, and toolchain/environment record instead.
- Never overwrite a cited result. Add a new run or correction that identifies the
  superseded record.
- Keep credentials, private endpoints, and licensed vendor sources out of result
  files. A deliberately published artifact-store URI may retain its non-secret bucket
  or account identifier when that identifier is necessary to retrieve the artifact.

Research runners live under [`scripts/`](../../scripts/README.md). Manuscript workspaces
consume selected evidence but are maintained separately.
