# Changelog

All notable changes to TikiTorch are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Exact forward-route certificates** — `tikitorch.autodiff` provides a typed
  operator registry, packed certificates, and selected-VJP replay for the
  explicitly supported hard-routed graph. Exactness, composition, tie, saved-state,
  and actual-MorphSSM tests accompany the implementation.
- **Independent research workspaces** — paper sources for route certificates and
  multiplier-free FPGA learning now live under `papers/`, with a shared publication
  map and evidence-controlled claim ledger.
- **Reproducible route benchmarks** — matched, trajectory, stability, saved-state,
  and actual-model entry points under `scripts/route_certificates/`, with preserved
  reference rows under `results/route_certificates/`.
- **Citation metadata** — `CITATION.cff` identifies the software artifact while the
  two research papers maintain separate bibliographies.
- **Loss modules** — `tt.nn.TropicalCrossEntropyLoss` and `tt.nn.TropicalPerceptronLoss`,
  thin `nn.Module` wrappers over the functional losses (cf. `torch.nn.CrossEntropyLoss`)
  for users who want the loss in a config / `repr`-able. `softmax="lut"` keeps the CE
  loss multiplier-free.
- **`MIGRATION.md`** — side-by-side "your PyTorch loop → closed TikiTorch loop": the swap
  table, the `set_tau` soft→hard curriculum, `assert_closed` as the new invariant, and
  what intentionally differs. Canonical copy at `docs/guide/migration.md`.
- **Guide docs** (`docs/guide/`, wired into the mkdocs nav) — Foundations, Why tropical
  for ML/DL, Tropical geometry of neural nets, Beyond deep learning, Architecture &
  diagrams, Choosing a model, Migrating from PyTorch, FAQ & troubleshooting.
- **Notebooks 05 & 06** — `05_tropical_geometry` (linear regions, decision regions,
  Newton polytopes via `tikitorch.geometry`) and `06_beyond_deep_learning` (shortest-path
  DP as min-plus matmul; tropical k-means / SVM / PCA via `tikitorch.classical` /
  `tikitorch.algorithmic`).
- `dyadic_quantize` promoted to `tikitorch.nn.__all__` (already public and imported across
  the codebase); new `tests/test_exports.py` guards `__all__` hygiene.
- **Tropical Forward-Forward** — Hinton's Forward-Forward adapted to the tropical
  substrate: a forward-only, backprop-free *local* learner whose whole training
  step is verified multiplier-free (`tt.assert_no_multiply`). Closure-clean choices:
  top-k goodness (not sum-of-squares), a hinge/margin local loss with a sign-gated
  scatter gradient, subtract-the-max normalization, and a sign-only update.
  Shipped as formal models `tt.models.TropicalForwardForward` (max-plus) and
  `tt.models.DyadicForwardForward` (dyadic ±2^k — dense/additive, reaches ~0.92 on
  MNIST vs ~0.23 for max-plus; the bottleneck is the max-plus primitive, not
  closure). Core in `tikitorch/forward_forward.py` (substrate-pluggable
  `TropicalFFLayer`, goodness fns, hinge/logistic loss, `TropicalFFTrainer`,
  negative-data generators, goodness classifier); the models train via
  `forward_forward_fit`, not `loss.backward()`. Experiments + honest report under
  `results/forward_forward/` (`scripts/tropical_forward_forward.py`,
  `tropical_ff_mnist_benchmark.py`, `tropical_ff_diagnostics.py`).
- `scripts/build_fpga_local.sh` and an upgraded `examples/fpga_quickstart.py`:
  one command builds the LOCAL host-faithful bridge so `model.fpga()` binds and
  runs a real, multiplier-free (dsp=0) training step with no FPGA.
- CI now compiles and runs the HLS kernel CSIM testbenches on every push.

### Changed
- The public description is library-first and recipe-scoped: tropical operators do
  not make every loss, backward rule, optimizer, host call, or synthesized build
  multiplier-free. Closure claims now point to concrete audit and physical-evidence
  boundaries.
- The package export version now agrees with project metadata at `0.2.0`.
- AWS scripts (`run_tikitorch_f2_silicon.sh`, `run_tikitorch_gpu_benchmarks.sh`)
  no longer default to any account: `AWS_PROFILE` / `KEY_NAME` / `SG` / `AGFI`
  are required env vars; the library ships no AFI (build your own).
- `BUILD_NATIVE.md`: corrected the LOCAL build command (it was missing the
  `-DCSIM_ONLY -DTT_KERNEL_*` defines, without which the bridge returns rc -99).

### Fixed
- `prune.py` citation: the "Max-plus Operators … Pruning" paper is Zhang,
  Blusseau et al. (ISMM 2019), not Dimitriadis & Maragos.

## [0.1.0] — 2026-06-08

Initial public development baseline for the tropical-semiring library. It included
specific closure-clean learning recipes and FPGA research paths; closure was not a
blanket property of every shipped model or training loop.

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

[Unreleased]: https://github.com/nyuolab/tikitorch/compare/3edd5f3...HEAD
[0.1.0]: https://github.com/nyuolab/tikitorch/tree/3edd5f3
