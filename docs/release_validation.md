# Release validation and remaining authority

The supported interpreter window is Python **3.9–3.12** (`>=3.9,<3.13` in
package metadata), with PyTorch >=2.0. Ordinary CI covers each interpreter and a
separate Python 3.9 / PyTorch 2.0.1 / NumPy 1.24.4 floor. Extend metadata and CI
together when validating a newer interpreter.

## Automatic checks

CI runs the CPU suite, deterministic bounded learning/route-training regressions,
shell syntax checks, offline runner argument/render tests, HLS C simulation,
closure auditing, documentation and distribution builds. A dedicated CPU job
requires native fullgraph Inductor execution and matched momentum-SGD updates
for compiled autograd, unpacked routes and packed routes. These checks do not
establish CUDA behavior, physical hardware utilization or publication clearance.

The three PC launch helpers have side-effect-free inspection interfaces:

```bash
bash scripts/_pc_g5_onbox.sh --print-commands
PYBIN=python3 PC_TASK=mnist PC_ARGS='--seed 1' bash scripts/_pc_scale_onbox.sh --print-commands
bash scripts/run_pc_scale_gpu.sh render s3://example/src.tar.gz s3://example/results COMMIT SHA256
```

These modes neither contact AWS nor start workloads; `render` writes user-data to
stdout and needs no AWS profile. The older `DRY_RUN=1 ... launch` still performs
AWS preflight reads and creates a local source archive. Generated user-data is a
program for a remote machine: inspect it rather than executing it locally.

## Explicit GPU/compiler validation

`.github/workflows/gpu-validation.yml` is manual-only and restricted to `main`.
It uses an **existing** trusted Linux x64 self-hosted runner with label
`tikitorch-gpu`, Python 3.9–3.12, a CUDA-capable PyTorch installation and a working
Inductor toolchain. It does not create EC2 instances or other paid resources.
Before enabling it, maintainers must configure the `gpu-validation` environment
with required reviewers and supply an isolated, preferably ephemeral runner.
Do not attach a privileged production host or expose secrets to test code.

The job requires an explicit existing-runner confirmation and fails if CUDA or
fullgraph Inductor execution is unavailable. It records source/environment
details and runs public-operator and route-tape tests. An unconfigured runner
leaves the job queued; it does not count as a pass. No workflow has been dispatched
as part of adding this configuration.

## Fail-closed provenance gate

Research-layout integrity is a separate check:

```bash
python scripts/check_artifact_layout.py
```

It verifies the frozen archive now under `manuscript/results/`, not permission
to redistribute it. Library source archives and wheels exclude `manuscript/`
(including the archived drafts), `scripts/` and `tests/`; use a Git checkout for the full development
and research workflow. See the
[artifact policy](https://github.com/nyuolab/tikitorch/blob/main/manuscript/ARTIFACTS.md).
The former root `papers/` directory is gone; its packaging exclusion remains
as a guard against accidental reintroduction.
After building, verify actual archive contents (also enforced in package CI):

```bash
python scripts/check_distribution_layout.py dist/*.tar.gz dist/*.whl
```

```bash
python scripts/check_release_provenance.py --inventory-only  # record integrity only
python scripts/check_release_provenance.py                   # publication gate
```

The first command is ordinary CI and reports `release_ready: false` while known
issues remain. The second exits nonzero until all records are resolved; the manual
`release-readiness.yml` workflow runs this gate without publishing anything.
Passing ordinary CI or building a wheel is not permission to publish the source
tree, change visibility, or expose retained Git history.

The machine-readable [inventory](release_provenance.json) fingerprints four RTL
wrappers and records the separate retained-history review. Local evidence establishes
adaptation, not distribution permission: the MuZero header identifies the HDK
`unused_ddr_template`; the PC header identifies verbatim reuse of that wiring;
the other wrappers retain related DDR structures. The historical
[`VENDOR_SNAPSHOT_MANIFEST.tsv`](https://github.com/nyuolab/tikitorch/blob/main/manuscript/results/fpga_silicon/ddr_arch/VENDOR_SNAPSHOT_MANIFEST.tsv)
identifies the captured template blob but does **not** establish an upstream commit,
complete chain of rights, or license compatibility. No source has been relicensed
or independently rewritten by this release-preparation change.

To resolve each source record, a maintainer must supply one of:

- An independently implemented replacement, with documented source boundaries and
  functional/synthesis verification; or
- The precise upstream source/revision, applicable complete notices and license,
  adapted-line inventory, and an approved distribution/licensing decision; or
- A deliberate release exclusion, including verification that no released archive
  or published ref redistributes the excluded material.

Record the evidence references, named reviewer, resolution status and current file
digest in the inventory. The checker verifies completeness and source identity;
it cannot verify the substance of a legal approval. A second human review is still
required. Review every ref intended for publication for vendor snapshots, routed
checkpoints and `Developer_CL` archives, documenting scope and disposition. Any
history rewrite or removal requires separate authorization; this gate performs none.
