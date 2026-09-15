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
| Experimental integer EF and public carry/resume workflow | `test_fixed_point_ef.py`, `test_fixed_point_workflow.py` |
| Route certificates and reverse replay | `test_route_*.py`, `test_reverse_action_primitives.py`, `test_saved_state_sufficiency.py`, `test_morphssm_tape_parity.py` |
| Local learning and distributed fabric | `test_forward_forward.py`, `test_predictive_coding.py`, `test_pc_convrnn_gradient_packet.py`, `test_fabric.py`, `test_tcp_link.py` |
| Planning and algorithmic APIs | `test_algorithmic.py`, `test_games.py`, `test_mcts.py`, `test_muzero.py`, `test_planning.py` |
| FPGA/backend integration | `test_backend.py`, `test_fpga.py`, `test_muzero_kernel.py`, `test_muzero_train_kernel.py` |
| Fresh-window FPGA LM, context diagnostics, checked PCI errors and offline cloud safety | `test_fpga_lm_pretrain.py`, `test_fpga_lm_context_diagnostic.py`, `test_fpga_bridge_errors.py`, `test_fpga_lm_aws.py`, `test_fpga_legacy_guard.py` |
| FP64 routed reference, integer numerical ladder and learning-diagnostic provenance | `test_fpga_lm_float_reference.py`, `test_fpga_lm_numeric_diagnostic.py`, `test_fpga_lm_learning_contract.py` |
| Persistent EF/unrounded trajectories, full-gradient arithmetic replay and lifecycle | `test_fpga_lm_trajectory_reference.py`, `test_fpga_lm_trajectory_arithmetic.py`, `test_fpga_lm_persistent_trajectory.py`, `test_fpga_lm_persistent_lifecycle.py` |
| Paired update-scale protocol, frozen controls, independent replay and fail-closed publication | `test_fpga_lm_update_scale.py`, `test_fpga_lm_update_scale_audit.py` |
| Matched gradient accumulation, signed grouping, complete replay and failure lifecycle | `test_fpga_lm_accumulation_arithmetic.py`, `test_fpga_lm_gradient_accumulation.py`, `test_fpga_lm_gradient_accumulation_replay.py` |
| Uncommitted step fractions, live-carry units and affine/integer diagnostic views | `test_fpga_lm_step_fraction.py`, `test_fpga_lm_batch_step.py` |
| Normalized accumulation, matched-scale order controls and complete integer replay | `test_fpga_lm_normalized_accumulation.py`, `test_fpga_lm_normalized_accumulation_replay.py` |
| Relative head weighting, fractional forward/reverse replay and ancestry guards | `test_fpga_lm_head_weighting.py`, `test_fpga_lm_head_weighting_replay.py` |
| Function-preserving cross-class topology, exact-CE/GRU controls and complete Adam replay | `test_fpga_lm_projection_topology.py`, `test_fpga_lm_exact_fit.py`, `test_fpga_lm_exact_fit_replay.py` |
| Single-rate exact-CE control, identical fitted initialization and independent full-state replay | `test_fpga_lm_exact_rate.py` |
| Pinned completed predecessor reuse, full independent rate replay and failure-safe audit publication | `test_fpga_lm_exact_rate_audit.py` |
| Experimental guarded FP64 forward, exact dense fallback and unchanged reverse semantics | `test_fpga_lm_guarded_forward.py` |
| Guarded versus independently audited dense trajectory, byte-exact gradients/full Adam states and bounded publication | `test_fpga_lm_guarded_rate.py` |
| Varied-text development, extra-pass and frozen-backbone controls | `test_fpga_lm_development.py`, `test_fpga_lm_continuation.py`, `test_fpga_lm_core_bias.py`, `test_fpga_lm_endpoint_fit.py` |
| Channel competition and learned routing contrasts | `test_fpga_lm_channel_choice.py`, `test_fpga_lm_routing_contrasts.py` |
| Frozen-feature affine readout, physical-coordinate conditioning and audited endpoint provenance | `test_fpga_lm_readout_oracle.py`, `test_fpga_lm_readout_preconditioned.py` |
| Historical CPU source snapshots and tamper rejection | `test_historical_cpu_evidence.py` |
| Classical, geometry, profiling, pruning, training | `test_classical.py`, `test_geometry.py`, `test_profile.py`, `test_prune.py`, `test_train.py` |
| Compiler/device contracts and training comparisons | `test_accelerator_contract.py`, `test_route_training.py`, `test_training_benchmark.py`, `test_optimizer_diagnostic.py`, `test_device_tape.py` |
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

The learning-contract tests use synthetic arithmetic/gradient fixtures and
mocked orchestration. They do not execute the declared teacher optimization
or corpus probes; reproduce those separately using the
[manuscript instructions](../manuscript/README.md). Ordinary metadata/integrity
checks verify retained evidence without repeating its scientific trajectories.
The frozen-readout tests use synthetic caches. Their separately invoked research
audits check endpoint gradients and scores; the conditioning audit also checks
the training Gram and coordinate transform. They do not replay every solver
trial or establish hardware performance.
The explicitly invoked pinned-rate numerical audit and guarded-parity verifier
recompute a fixed saved trajectory; their pytest coverage uses only synthetic
tiny-model fixtures, never the research corpus.
The persistent-trajectory checker additionally replays saved full-gradient
optimizer arithmetic and checks complete native/FP64 checkpoint payloads;
it does not independently recompute the model's gradients or learning run.
Update-scale tests likewise use synthetic/fake-native fixtures, not the research
corpus. The study's separate `--check --recompute-evaluation` mode independently
recomputes all development readouts and selected full-gradient parity anchors,
but does not replay every training update. Historical CPU snapshot checks
validate original source-bound results, not current-library performance.

The new PCI fault-injection tests compile against fake AWS headers and run
without a board. They exercise error propagation, not real PCI performance.
The fresh-window runner's native C-simulation test uses an isolated subprocess:

```bash
python -m pytest -q -m kernel tests/test_fpga_lm_pretrain.py
```

It verifies the independent integer forward/LUT-gradient reference, frozen
weights and native layout/backend rejection. It does not validate a loaded AFI
or large-scale language quality. Cloud-helper tests use fake read-only responses;
even the `launch` command remains disabled until lifecycle controls are ready.

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
