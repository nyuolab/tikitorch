# TikiTorch test suite

Use a Git source checkout and run commands from its root. Release wheels and
source archives contain the library, not these tests or their research fixtures.

```bash
python -m pip install -e ".[dev]"
```

Then choose the checks relevant to your change:

```bash
python -m pytest -q                          # default correctness/regression suite
python -m pytest -q tests/test_functional.py  # one focused module
python -m pytest -m slow                     # stochastic learning checks only
python -m pytest -m kernel                   # native kernel loading; run separately
python -m ruff check .                       # CI's pinned lint contract
```

The default marker expression in `pyproject.toml` excludes `slow` and `kernel`.
Slow tests validate learning behavior rather than basic library correctness. Kernel
tests compile and load native HLS shims in-process and are isolated because a native
fault can terminate the Python process; CI's `csim` job independently compiles and
runs the C++ testbenches.

## Coverage map

| Area | Representative tests |
| --- | --- |
| Public imports, exports, serialization, CLI | `test_import.py`, `test_exports.py`, `test_serialization.py`, `test_smoke.py`, `test_cli.py` |
| Tropical operators, tie rules, layers, models | `test_functional.py`, `test_tie_rules.py`, `test_nn.py`, `test_models.py`, `test_transformer_stage.py` |
| Optimizers and full-recipe closure | `test_optim.py`, `test_audit.py`, `test_closure_guard.py`, `test_dyadic_backward_closure.py` |
| Route certificates and reverse replay | `test_route_*.py`, `test_reverse_action_primitives.py`, `test_saved_state_sufficiency.py`, `test_morphssm_tape_parity.py` |
| Local learning and distributed fabric | `test_forward_forward.py`, `test_predictive_coding.py`, `test_pc_convrnn_gradient_packet.py`, `test_fabric.py`, `test_tcp_link.py` |
| Planning and algorithmic APIs | `test_algorithmic.py`, `test_games.py`, `test_mcts.py`, `test_muzero.py`, `test_planning.py` |
| FPGA/backend integration | `test_backend.py`, `test_fpga.py`, `test_muzero_kernel.py`, `test_muzero_train_kernel.py` |
| Classical, geometry, profiling, pruning, training | `test_classical.py`, `test_geometry.py`, `test_profile.py`, `test_prune.py`, `test_train.py` |
| Compiler/device contracts and training comparisons | `test_accelerator_contract.py`, `test_route_training.py`, `test_training_benchmark.py`, `test_device_tape.py` |
| Research and release boundaries | `test_artifact_layout.py`, `test_distribution_layout.py`, `test_release_provenance.py`, `test_manuscript_layout.py`, `test_research_workspace.py` |
| Research recipe auditing and offline launchers | `test_nanogpt_strict_gate.py`, `test_shell_runners.py` |

Tests for public modules are active, not placeholder-guarded. Hardware-dependent
entry points skip only when their declared external board/toolchain prerequisite is
unavailable.

The TCP transport tests use loopback sockets; a sandbox that forbids local socket
binding cannot run those tests. Native compilation and CUDA checks have explicit
availability requirements; a skipped test does not validate the missing backend.
Shell-runner tests use fake external commands and local fixtures, never AWS
requests or training jobs.

Research evidence and distribution layouts also have standalone, read-only checks:

```bash
python scripts/check_artifact_layout.py
python scripts/check_release_provenance.py --inventory-only
# After building distributions with the build extra installed:
python scripts/check_distribution_layout.py dist/*.tar.gz dist/*.whl
```

Passing integrity checks is not publication clearance. See
[release validation](../docs/release_validation.md) for the separate provenance
gate, native compiler requirements, and maintainer-approved GPU workflow.

CI runs Ruff once, the fast suite on every supported Python version, a separate
advertised-minimum-dependency lane, focused closure checks, C-simulation, a strict
documentation build, and package build/install smoke checks.
