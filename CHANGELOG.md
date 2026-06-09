# Changelog

All notable changes to TikiTorch are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scripts/build_fpga_local.sh` and an upgraded `examples/fpga_quickstart.py`:
  one command builds the LOCAL host-faithful bridge so `model.fpga()` binds and
  runs a real, multiplier-free (dsp=0) training step with no FPGA.
- CI now compiles and runs the HLS kernel CSIM testbenches on every push.

### Changed
- AWS scripts (`run_tikitorch_f2_silicon.sh`, `run_tikitorch_gpu_benchmarks.sh`)
  no longer default to any account: `AWS_PROFILE` / `KEY_NAME` / `SG` / `AGFI`
  are required env vars; the library ships no AFI (build your own).
- `BUILD_NATIVE.md`: corrected the LOCAL build command (it was missing the
  `-DCSIM_ONLY -DTT_KERNEL_*` defines, without which the bridge returns rc -99).

### Fixed
- `prune.py` citation: the "Max-plus Operators … Pruning" paper is Zhang,
  Blusseau et al. (ISMM 2019), not Dimitriadis & Maragos.

## [0.1.0] — 2026-06-08

Initial public release — a tropical-semiring deep-learning library whose entire
training loop is multiplier-free and maps to zero-DSP FPGA fabric.

### Added
- **`tikitorch.functional`** — max-plus / min-plus ops with custom autograd
  (tropical linear/dilation, softmin/softmax relaxations, tropical attention
  scores, value-pull / top-k aggregation, tropical conv2d).
- **`tikitorch.nn`** — tropical / morphological layer modules (TropicalLinear,
  dilation/erosion, TropicalLayerNorm, TropicalTopK, tropical-attention and
  morphological blocks, MoTE).
- **`tikitorch.optim`** — strictly multiplier-free optimizers: SignSGD,
  TropicalSGD, PureTropicalSGD, plus the closed adaptive variants (TropicalAdam,
  FrexpMagnitudeSGD, CountAgreementSGD, ExponentEMAAdam, PerLayerStepSGD).
- **`tikitorch.models`** — MorphSSM, TropicalTransformer, MoTEModel, TropicalMLP,
  TropicalSequenceModel, MorphMLP.
- **`tikitorch.audit`** — the three-tier closure audit: `assert_closed`,
  `find_runtime_multiplies` / `assert_closed_runtime`, and `closure_report`.
- **`tikitorch.fpga`** — `model.fpga()` whole-model binding to the bundled,
  silicon-validated fused-training-step kernels (LOCAL and AWS F2 paths).
- **`tikitorch.classical` / `.geometry` / `.prune` / `.algorithmic` /
  `.planning`** — classical tropical ML, tropical-geometry diagnostics, tropical
  structured pruning, the algorithmic-reasoning harness, and Tropical MuZero/MCTS.
- Bundled HLS kernels + silicon build guide; a closed model zoo (`zoo/`); CI
  (ruff + pytest).

[Unreleased]: https://github.com/nyuolab/tikitorch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nyuolab/tikitorch/releases/tag/v0.1.0
