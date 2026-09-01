# Changelog

All notable changes to TikiTorch are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Public-release governance and safety files** — contribution governance,
  code of conduct, security reporting, third-party notices, and a release
  checklist now make repository responsibilities and release gates explicit.
- **Two-paper Xiang handoff** — one review entry point now separates the
  route-certificate and multiplier-free-FPGA drafts, their durable claims, and
  the experiments still required before submission.
- **Exact forward-route certificates** — `tikitorch.autodiff` provides a typed
  operator registry, packed certificates, and selected-VJP replay for the
  explicitly supported hard-routed graph. Exactness, composition, tie, saved-state,
  and actual-MorphSSM tests accompany the implementation.
- **Independent research workspaces** — paper sources for route certificates and
  multiplier-free FPGA learning now live under `papers/`, with a shared publication
  map and evidence-controlled claim ledger.
- **Root manuscript handoff** — `manuscript/` is a zero-copy build entry point and
  publication brief for the canonical route-certificate source under `papers/`.
- **Library-first documentation** — installation, five-minute quickstart, recipes,
  closure boundaries, API index, research map, and namespace reference now form a
  progressive PyTorch-style documentation path.
- **Reproducible route benchmarks** — matched, trajectory, stability, saved-state,
  and actual-model entry points under `scripts/route_certificates/`, with preserved
  reference rows under `results/route_certificates/`.
- **Citation metadata** — `CITATION.cff` identifies the software artifact while the
  two research papers maintain separate bibliographies.
- **Loss modules** — `tt.nn.TropicalCrossEntropyLoss` and `tt.nn.TropicalPerceptronLoss`,
  thin `nn.Module` wrappers over the functional losses (cf. `torch.nn.CrossEntropyLoss`)
  for users who want the loss in a config / `repr`-able. `softmax="lut"` exposes the
  lookup/shift conditioning path; full-step closure remains recipe-scoped.
- **`MIGRATION.md`** — side-by-side PyTorch/TikiTorch namespace and training-loop map,
  the `set_tau` soft→hard curriculum, explicit arithmetic boundaries, and what
  intentionally differs. Canonical copy at `docs/guide/migration.md`.
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
  substrate: a layer-local learner without end-to-end backpropagation, with a
  registered full-step software audit. Its constrained choices include
  top-k goodness (not sum-of-squares), a hinge/margin local loss with a sign-gated
  scatter gradient, subtract-the-max normalization, and a sign-only update.
  Shipped as experimental models `tt.models.TropicalForwardForward` (max-plus) and
  `tt.models.DyadicForwardForward` (dyadic signed powers of two). Core in
  `tikitorch/forward_forward.py` (substrate-pluggable
  `TropicalFFLayer`, goodness fns, hinge/logistic loss, `TropicalFFTrainer`,
  negative-data generators, goodness classifier); the models train via
  `forward_forward_fit`, not `loss.backward()`. Experiments + honest report under
  `results/forward_forward/` (`scripts/tropical_forward_forward.py`,
  `tropical_ff_mnist_benchmark.py`, `tropical_ff_diagnostics.py`).
- `scripts/build_fpga_local.sh` and an upgraded `examples/fpga_quickstart.py`:
  one command builds the LOCAL host-reference endpoint, while the example
  distinguishes catalog lookup, structural matching, endpoint availability,
  explicit runner registration, and PyTorch fallback.
- CI now compiles and runs the HLS kernel CSIM testbenches on every push.
- CI now builds and installs the wheel and source distribution, validates both
  with Twine, and exercises the installed closure CLI.

### Changed
- Merged the `pc-morphssm` research line into `main`, including route
  certificates, local-learning experiments, silicon evidence, and both
  manuscript workspaces.
- TikiTorch is now described as an independent companion in the PyTorch
  ecosystem, with an explicit comparison of what it reuses from PyTorch and
  what its tropical, route-replay, closure-audit, and hardware paths add.
- Public subpackages import eagerly so broken required dependencies or export
  errors fail at import time instead of silently replacing namespaces with
  `None`.
- The unsafe one-layer FPGA reference runner is retired and its compatibility
  hook now fails loudly. It flattened parameters in PyTorch registration order,
  which did not match the HLS named regions, and omitted B1/B2 synchronization;
  native dispatch now requires an explicitly registered, validated runner.
- FPGA binding rejects non-FPGA logical devices and unsupported endpoint indices.
  The MorphSSM catalog remains inspectable, but whole-model matching now fails
  closed until the public zero-padding boundary is made equivalent to the HLS
  kernel's skip-invalid causal boundary.
- Route-certificate retained-state summaries now use symmetric boundaries:
  1.97--2.96x for portable certificates with a supplied cotangent and
  1.69--2.79x for the current loss-inclusive replay.
- Vendor-licensed AWS/Xilinx snapshot sources are no longer present in the
  release tree; a manifest preserves object-level provenance while repository
  history review remains a required pre-publication gate.
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
- Distribution metadata now uses an SPDX license expression with explicit
  license files, and README links are absolute so the PyPI long description
  does not produce repository-relative broken links.

### Fixed
- `prune.py` citation: the "Max-plus Operators … Pruning" paper is Zhang,
  Blusseau et al. (ISMM 2019), not Dimitriadis & Maragos.
- TCP neighbour-link frames are rank-, axis-, and payload-bounded before memory
  allocation, and split-chain listeners default to loopback unless a reachable
  interface is selected explicitly. The transport remains unauthenticated and
  unencrypted and is documented for trusted networks or protected tunnels only.

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
- **`tikitorch.optim`** — sign-, shift-, and counter-oriented optimizers: SignSGD,
  TropicalSGD, PureTropicalSGD, plus adaptive research variants (TropicalAdam,
  FrexpMagnitudeSGD, CountAgreementSGD, ExponentEMAAdam, PerLayerStepSGD).
- **`tikitorch.models`** — MorphSSM, TropicalTransformer, MoTEModel, TropicalMLP,
  TropicalSequenceModel, MorphMLP.
- **`tikitorch.audit`** — the three-tier closure audit: `assert_closed`,
  `find_runtime_multiplies` / `assert_closed_runtime`, and `closure_report`.
- **`tikitorch.fpga`** — experimental `model.fpga()` whole-model binding and
  runners for specific local-reference and AWS F2 kernels.
- **`tikitorch.classical` / `.geometry` / `.prune` / `.algorithmic` /
  `.planning`** — classical tropical ML, tropical-geometry diagnostics, tropical
  structured pruning, the algorithmic-reasoning harness, and Tropical MuZero/MCTS.
- Bundled HLS kernels + silicon build guide; a research model zoo (`zoo/`); CI
  (ruff + pytest).

[Unreleased]: https://github.com/nyuolab/tikitorch/compare/3edd5f3...HEAD
[0.1.0]: https://github.com/nyuolab/tikitorch/tree/3edd5f3
