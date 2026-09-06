# TikiTorch test suite

Run commands from the repository root:

```bash
pytest -q                         # fast library suite (the default)
pytest tests/test_functional.py  # one focused module
pytest -m slow                   # stochastic learning checks only
pytest -m kernel                 # native kernel loading; run separately
```

The default marker expression in `pyproject.toml` excludes `slow` and `kernel`.
Slow tests validate learning behavior rather than basic library correctness. Kernel
tests compile and load native HLS shims in-process and are isolated because a native
fault can terminate the Python process; CI's `csim` job independently compiles and
runs the C++ testbenches.

## Coverage map

| Area | Representative tests |
| --- | --- |
| Public imports, exports, serialization | `test_import.py`, `test_exports.py`, `test_serialization.py`, `test_smoke.py` |
| Tropical operators, tie rules, layers, models | `test_functional.py`, `test_tie_rules.py`, `test_nn.py`, `test_models.py`, `test_transformer_stage.py` |
| Optimizers and full-recipe closure | `test_optim.py`, `test_audit.py`, `test_closure_guard.py`, `test_dyadic_backward_closure.py` |
| Route certificates and reverse replay | `test_route_*.py`, `test_reverse_action_primitives.py`, `test_saved_state_sufficiency.py`, `test_morphssm_tape_parity.py` |
| Local learning and distributed fabric | `test_forward_forward.py`, `test_predictive_coding.py`, `test_pc_convrnn_gradient_packet.py`, `test_fabric.py`, `test_tcp_link.py` |
| Planning and algorithmic APIs | `test_algorithmic.py`, `test_games.py`, `test_mcts.py`, `test_muzero.py`, `test_planning.py` |
| FPGA/backend integration | `test_backend.py`, `test_fpga.py`, `test_muzero_kernel.py`, `test_muzero_train_kernel.py` |
| Classical, geometry, profiling, pruning, training | `test_classical.py`, `test_geometry.py`, `test_profile.py`, `test_prune.py`, `test_train.py` |

Tests for public modules are active, not placeholder-guarded. Hardware-dependent
entry points skip only when their declared external board/toolchain prerequisite is
unavailable.

CI runs Ruff once, the fast suite on every supported Python version, a separate
advertised-minimum-dependency lane, focused closure checks, C-simulation, a strict
documentation build, and package build/install smoke checks.
