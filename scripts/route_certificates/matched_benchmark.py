"""Matched CPU benchmark for route-sufficient reverse differentiation.

Every benchmark mode executes the same residual DAG::

    branch_l[j] = extreme_i(h_l[i] + weight_l[j, i]) + bias_l[j]
    h_{l+1} = h_l + branch_l

Layers alternate max-plus and min-plus.  The scalar objective is a fixed
linear readout ``sum(h_L * output_adjoint)``.  Consequently the objective's
output adjoint is identical in every mode and the local additive-parameter
VJPs need only the upstream adjoint and the selected routes.

This is a small, clarity-first systems experiment.  It does not benchmark the
full MorphSSM implementation and it does not claim that the direct and
"cascade" realizations are different algorithms: in this DAG they are the
same reverse-topological computation, and the result record says so.

Run from the repository root, for example::

    python -m scripts.route_certificates.matched_benchmark

The default grid is CPU-only, uses three seeds, two widths, two depths, five
warmups, twenty measured repetitions, and a five-step matched SGD trajectory.
Results are appended as one JSON object per configuration.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import uuid
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.checkpoint import checkpoint

from tikitorch.autodiff.packed_tape import PackedRouteTape, bits_for_cardinality
from tikitorch.autodiff.primitives import additive_extreme, additive_extreme_vjp


Mode = Literal["autograd", "direct", "cascade", "checkpoint", "packed"]
MODES: tuple[Mode, ...] = ("autograd", "direct", "cascade", "checkpoint", "packed")


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int
    width: int
    depth: int
    batch_size: int
    dtype: str
    learning_rate: float
    warmups: int
    repetitions: int
    trajectory_steps: int


@dataclass
class StepResult:
    parameters: list[tuple[Tensor, Tensor]]
    gradients: list[tuple[Tensor, Tensor]]
    output: Tensor
    loss: float
    routes: list[Tensor] | None
    times_ns: dict[str, int]
    memory: dict[str, int]
    recomputation_count: int
    recomputation_bytes: int


class SavedTensorTracker:
    """Measure tensors retained by autograd, de-duplicated by storage."""

    def __init__(self, parameter_keys: set[tuple[str, int, int]]) -> None:
        self.parameter_keys = parameter_keys
        self.saved_calls = 0
        self.saved_logical_bytes = 0
        self._storages: dict[tuple[str, int, int], int] = {}

    @staticmethod
    def storage_key(tensor: Tensor) -> tuple[str, int, int]:
        storage = tensor.untyped_storage()
        return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))

    def pack(self, tensor: Tensor) -> Tensor:
        self.saved_calls += 1
        self.saved_logical_bytes += tensor.numel() * tensor.element_size()
        key = self.storage_key(tensor)
        self._storages[key] = key[2]
        return tensor

    @staticmethod
    def unpack(tensor: Tensor) -> Tensor:
        return tensor

    def summary(self) -> dict[str, int]:
        all_bytes = sum(self._storages.values())
        parameter_bytes = sum(
            nbytes for key, nbytes in self._storages.items() if key in self.parameter_keys
        )
        return {
            "saved_tensor_calls": self.saved_calls,
            "saved_tensor_logical_bytes_without_dedup": self.saved_logical_bytes,
            "saved_storage_bytes_deduplicated": all_bytes,
            "saved_nonparameter_storage_bytes_deduplicated": all_bytes - parameter_bytes,
            "saved_parameter_storage_bytes_deduplicated": parameter_bytes,
            "saved_unique_storages": len(self._storages),
        }


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("this benchmark supports float32 and float64")
    return dtype


def _mode_for_layer(layer: int) -> Literal["max", "min"]:
    return "max" if layer % 2 == 0 else "min"


def make_problem(
    config: BenchmarkConfig,
) -> tuple[list[tuple[Tensor, Tensor]], Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    dtype = _dtype_from_name(config.dtype)
    scale = 1.0 / math.sqrt(config.width)
    parameters = [
        (
            torch.randn(config.width, config.width, generator=generator, dtype=dtype) * scale,
            torch.randn(config.width, generator=generator, dtype=dtype) * scale,
        )
        for _ in range(config.depth)
    ]
    inputs = torch.randn(
        config.batch_size, config.width, generator=generator, dtype=dtype
    )
    output_adjoint = torch.randn(
        config.batch_size, config.width, generator=generator, dtype=dtype
    )
    output_adjoint /= output_adjoint.norm().clamp_min(torch.finfo(dtype).tiny)
    return parameters, inputs, output_adjoint


def clone_parameters(
    parameters: Sequence[tuple[Tensor, Tensor]], *, requires_grad: bool = False
) -> list[tuple[Tensor, Tensor]]:
    return [
        (
            weight.detach().clone().requires_grad_(requires_grad),
            bias.detach().clone().requires_grad_(requires_grad),
        )
        for weight, bias in parameters
    ]


def parameter_storage_keys(parameters: Sequence[tuple[Tensor, Tensor]]) -> set[tuple[str, int, int]]:
    return {
        SavedTensorTracker.storage_key(parameter)
        for pair in parameters
        for parameter in pair
    }


def forward_with_routes(
    inputs: Tensor, parameters: Sequence[tuple[Tensor, Tensor]]
) -> tuple[Tensor, list[Tensor]]:
    hidden = inputs
    routes: list[Tensor] = []
    for layer, (weight, bias) in enumerate(parameters):
        branch, route = additive_extreme(
            hidden, weight, bias, mode=_mode_for_layer(layer)
        )
        hidden = hidden + branch
        routes.append(route)
    return hidden, routes


def forward_with_packed_routes(
    inputs: Tensor, parameters: Sequence[tuple[Tensor, Tensor]]
) -> tuple[Tensor, PackedRouteTape]:
    hidden = inputs
    tape = PackedRouteTape()
    width = inputs.shape[-1]
    for layer, (weight, bias) in enumerate(parameters):
        branch, route = additive_extreme(
            hidden, weight, bias, mode=_mode_for_layer(layer)
        )
        hidden = hidden + branch
        tape.append(f"layer_{layer}", route, cardinality=width)
    return hidden, tape


def direct_route_reverse(
    output_adjoint: Tensor,
    routes: Sequence[Tensor],
    width: int,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    """Reverse the residual DAG from route tensors and an output adjoint."""

    hidden_adjoint = output_adjoint
    gradients_reversed: list[tuple[Tensor, Tensor]] = []
    for route in reversed(routes):
        branch_input_adjoint, grad_weight, grad_bias = additive_extreme_vjp(
            hidden_adjoint, route, width
        )
        gradients_reversed.append((grad_weight, grad_bias))
        # Residual VJP: one copy follows the skip edge and one the branch.
        hidden_adjoint = hidden_adjoint + branch_input_adjoint
    return hidden_adjoint, list(reversed(gradients_reversed))


def cascade_reverse(
    output_adjoint: Tensor,
    routes: Sequence[Tensor],
    width: int,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    """Cascade realization; exactly the direct reverse on this acyclic DAG."""

    return direct_route_reverse(output_adjoint, routes, width)


def packed_route_reverse(
    output_adjoint: Tensor,
    tape: PackedRouteTape,
    *,
    depth: int,
    width: int,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    routes = [tape.read(f"layer_{layer}") for layer in range(depth)]
    return direct_route_reverse(output_adjoint, routes, width)


def _loss(output: Tensor, output_adjoint: Tensor) -> Tensor:
    return (output * output_adjoint).sum()


def _sgd_update(
    parameters: Sequence[tuple[Tensor, Tensor]],
    gradients: Sequence[tuple[Tensor, Tensor]],
    learning_rate: float,
) -> None:
    with torch.no_grad():
        for (weight, bias), (grad_weight, grad_bias) in zip(parameters, gradients, strict=True):
            weight.add_(grad_weight, alpha=-learning_rate)
            bias.add_(grad_bias, alpha=-learning_rate)


def _autograd_step(
    mode: Literal["autograd", "checkpoint"],
    initial_parameters: Sequence[tuple[Tensor, Tensor]],
    inputs: Tensor,
    output_adjoint: Tensor,
    learning_rate: float,
    *,
    capture_memory: bool,
) -> StepResult:
    parameters = clone_parameters(initial_parameters, requires_grad=True)
    tracker = SavedTensorTracker(parameter_storage_keys(parameters))
    layer_calls = [0]

    def plain_forward() -> Tensor:
        output, _ = forward_with_routes(inputs, parameters)
        layer_calls[0] += len(parameters)
        return output

    def checkpoint_forward() -> Tensor:
        hidden = inputs
        for layer, (weight, bias) in enumerate(parameters):
            layer_mode = _mode_for_layer(layer)

            def layer_function(
                value: Tensor,
                layer_weight: Tensor,
                layer_bias: Tensor,
                *,
                route_mode: Literal["max", "min"] = layer_mode,
            ) -> Tensor:
                layer_calls[0] += 1
                branch, _ = additive_extreme(
                    value, layer_weight, layer_bias, mode=route_mode
                )
                return value + branch

            hidden = checkpoint(
                layer_function,
                hidden,
                weight,
                bias,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return hidden

    forward_fn: Callable[[], Tensor] = plain_forward if mode == "autograd" else checkpoint_forward
    start = time.perf_counter_ns()
    if capture_memory:
        with saved_tensors_hooks(tracker.pack, tracker.unpack):
            output = forward_fn()
            loss = _loss(output, output_adjoint)
    else:
        output = forward_fn()
        loss = _loss(output, output_adjoint)
    after_forward = time.perf_counter_ns()
    loss.backward()
    after_reverse = time.perf_counter_ns()
    gradients = [
        (weight.grad.detach().clone(), bias.grad.detach().clone())  # type: ignore[union-attr]
        for weight, bias in parameters
    ]
    with torch.no_grad():
        for weight, bias in parameters:
            weight.add_(weight.grad, alpha=-learning_rate)  # type: ignore[arg-type]
            bias.add_(bias.grad, alpha=-learning_rate)  # type: ignore[arg-type]
    after_update = time.perf_counter_ns()

    extra_layer_calls = max(0, layer_calls[0] - len(parameters))
    recomputation_bytes = extra_layer_calls * output.numel() * output.element_size()
    return StepResult(
        parameters=parameters,
        gradients=gradients,
        output=output.detach(),
        loss=float(loss.detach()),
        routes=None,
        times_ns={
            "forward": after_forward - start,
            "reverse": after_reverse - after_forward,
            "update": after_update - after_reverse,
        },
        memory=tracker.summary() if capture_memory else {},
        recomputation_count=extra_layer_calls,
        recomputation_bytes=recomputation_bytes,
    )


def _route_step(
    mode: Literal["direct", "cascade", "packed"],
    initial_parameters: Sequence[tuple[Tensor, Tensor]],
    inputs: Tensor,
    output_adjoint: Tensor,
    learning_rate: float,
) -> StepResult:
    parameters = clone_parameters(initial_parameters)
    start = time.perf_counter_ns()
    if mode == "packed":
        output, tape = forward_with_packed_routes(inputs, parameters)
        routes: list[Tensor] | None = None
        trace_memory = {
            "logical_trace_bits": tape.bit_length,
            "packed_payload_bytes": tape.payload_nbytes,
            "packed_serialized_bytes": len(tape.to_bytes()),
            "packed_directory_bytes": len(tape.to_bytes()) - tape.payload_nbytes,
            "unpacked_route_tensor_bytes": 0,
        }
    else:
        output, routes = forward_with_routes(inputs, parameters)
        tape = None
        trace_memory = {
            "logical_trace_bits": (
                len(parameters)
                * inputs.shape[0]
                * inputs.shape[1]
                * bits_for_cardinality(inputs.shape[1])
            ),
            "packed_payload_bytes": 0,
            "packed_serialized_bytes": 0,
            "packed_directory_bytes": 0,
            "unpacked_route_tensor_bytes": sum(
                route.numel() * route.element_size() for route in routes
            ),
        }
    loss = _loss(output, output_adjoint)
    after_forward = time.perf_counter_ns()
    if mode == "packed":
        assert tape is not None
        _, gradients = packed_route_reverse(
            output_adjoint, tape, depth=len(parameters), width=inputs.shape[-1]
        )
    elif mode == "cascade":
        assert routes is not None
        _, gradients = cascade_reverse(output_adjoint, routes, inputs.shape[-1])
    else:
        assert routes is not None
        _, gradients = direct_route_reverse(output_adjoint, routes, inputs.shape[-1])
    after_reverse = time.perf_counter_ns()
    _sgd_update(parameters, gradients, learning_rate)
    after_update = time.perf_counter_ns()
    return StepResult(
        parameters=parameters,
        gradients=gradients,
        output=output.detach(),
        loss=float(loss.detach()),
        routes=routes,
        times_ns={
            "forward": after_forward - start,
            "reverse": after_reverse - after_forward,
            "update": after_update - after_reverse,
        },
        memory=trace_memory,
        recomputation_count=0,
        recomputation_bytes=0,
    )


def run_step(
    mode: Mode,
    initial_parameters: Sequence[tuple[Tensor, Tensor]],
    inputs: Tensor,
    output_adjoint: Tensor,
    learning_rate: float,
    *,
    capture_memory: bool = False,
) -> StepResult:
    if mode in ("autograd", "checkpoint"):
        return _autograd_step(
            mode,
            initial_parameters,
            inputs,
            output_adjoint,
            learning_rate,
            capture_memory=capture_memory,
        )
    return _route_step(mode, initial_parameters, inputs, output_adjoint, learning_rate)


def _max_parameter_difference(
    left: Sequence[tuple[Tensor, Tensor]], right: Sequence[tuple[Tensor, Tensor]]
) -> float:
    return max(
        float((a.detach() - b.detach()).abs().max())
        for left_pair, right_pair in zip(left, right, strict=True)
        for a, b in zip(left_pair, right_pair, strict=True)
    )


def _assert_parameter_close(
    actual: Sequence[tuple[Tensor, Tensor]],
    expected: Sequence[tuple[Tensor, Tensor]],
    *,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.float64:
        rtol, atol = 1e-12, 1e-12
    else:
        rtol, atol = 2e-6, 2e-6
    for actual_pair, expected_pair in zip(actual, expected, strict=True):
        for actual_tensor, expected_tensor in zip(actual_pair, expected_pair, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)


def _route_digest(routes: Iterable[Tensor]) -> str:
    digest = hashlib.sha256()
    for route in routes:
        digest.update(route.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_single_step(
    config: BenchmarkConfig,
    parameters: Sequence[tuple[Tensor, Tensor]],
    inputs: Tensor,
    output_adjoint: Tensor,
) -> tuple[dict[str, Any], dict[Mode, StepResult]]:
    dtype = _dtype_from_name(config.dtype)
    results = {
        mode: run_step(
            mode,
            parameters,
            inputs,
            output_adjoint,
            config.learning_rate,
            capture_memory=True,
        )
        for mode in MODES
    }
    reference = results["autograd"]
    output_differences: dict[str, float] = {}
    gradient_differences: dict[str, float] = {}
    update_differences: dict[str, float] = {}
    for mode, result in results.items():
        torch.testing.assert_close(result.output, reference.output, rtol=0.0, atol=0.0)
        _assert_parameter_close(result.gradients, reference.gradients, dtype=dtype)
        _assert_parameter_close(result.parameters, reference.parameters, dtype=dtype)
        output_differences[mode] = float((result.output - reference.output).abs().max())
        gradient_differences[mode] = _max_parameter_difference(
            result.gradients, reference.gradients
        )
        update_differences[mode] = _max_parameter_difference(
            result.parameters, reference.parameters
        )

    direct_routes = results["direct"].routes
    cascade_routes = results["cascade"].routes
    assert direct_routes is not None and cascade_routes is not None
    route_equal = all(
        torch.equal(direct, cascade)
        for direct, cascade in zip(direct_routes, cascade_routes, strict=True)
    )
    if not route_equal:
        raise AssertionError("direct and cascade forward routes differ")
    _, packed_tape = forward_with_packed_routes(inputs, parameters)
    packed_equal = all(
        torch.equal(route, packed_tape.read(f"layer_{layer}"))
        for layer, route in enumerate(direct_routes)
    )
    if not packed_equal:
        raise AssertionError("packed route round trip differs")

    return (
        {
            "passed": True,
            "comparison_reference": "autograd",
            "output_max_abs_difference": output_differences,
            "parameter_gradient_max_abs_difference": gradient_differences,
            "post_update_parameter_max_abs_difference": update_differences,
            "direct_cascade_routes_equal": route_equal,
            "packed_routes_equal": packed_equal,
            "route_sha256": _route_digest(direct_routes),
            "tolerances": {
                "output_rtol": 0.0,
                "output_atol": 0.0,
                "parameter_rtol": 1e-12 if dtype == torch.float64 else 2e-6,
                "parameter_atol": 1e-12 if dtype == torch.float64 else 2e-6,
            },
        },
        results,
    )


def validate_trajectory(
    config: BenchmarkConfig,
    initial_parameters: Sequence[tuple[Tensor, Tensor]],
) -> dict[str, Any]:
    dtype = _dtype_from_name(config.dtype)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1_000_003)
    batches = [
        (
            torch.randn(
                config.batch_size, config.width, generator=generator, dtype=dtype
            ),
            torch.randn(
                config.batch_size, config.width, generator=generator, dtype=dtype
            ),
        )
        for _ in range(config.trajectory_steps)
    ]
    batches = [
        (inputs, adjoint / adjoint.norm().clamp_min(torch.finfo(dtype).tiny))
        for inputs, adjoint in batches
    ]
    states = {mode: clone_parameters(initial_parameters) for mode in MODES}
    maximum_state_difference = {mode: 0.0 for mode in MODES}
    maximum_loss_difference = {mode: 0.0 for mode in MODES}
    for inputs, output_adjoint in batches:
        step_results = {
            mode: run_step(
                mode,
                states[mode],
                inputs,
                output_adjoint,
                config.learning_rate,
            )
            for mode in MODES
        }
        states = {mode: result.parameters for mode, result in step_results.items()}
        reference = step_results["autograd"]
        for mode, result in step_results.items():
            _assert_parameter_close(result.parameters, reference.parameters, dtype=dtype)
            maximum_state_difference[mode] = max(
                maximum_state_difference[mode],
                _max_parameter_difference(result.parameters, reference.parameters),
            )
            maximum_loss_difference[mode] = max(
                maximum_loss_difference[mode], abs(result.loss - reference.loss)
            )
    return {
        "passed": True,
        "steps": config.trajectory_steps,
        "optimizer": "plain_sgd_without_momentum",
        "learning_rate": config.learning_rate,
        "maximum_parameter_difference_from_autograd": maximum_state_difference,
        "maximum_loss_difference_from_autograd": maximum_loss_difference,
    }


def _timing_summary(samples_ns: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(samples_ns)
    count = len(ordered)

    def percentile(fraction: float) -> float:
        position = (count - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower] / 1e6
        weight = position - lower
        return (ordered[lower] * (1.0 - weight) + ordered[upper] * weight) / 1e6

    median_ns = statistics.median(ordered)
    mad_ns = statistics.median(abs(value - median_ns) for value in ordered)
    return {
        "count": count,
        "median_ms": median_ns / 1e6,
        "median_absolute_deviation_ms": mad_ns / 1e6,
        "mean_ms": statistics.mean(ordered) / 1e6,
        "stdev_ms": statistics.stdev(ordered) / 1e6 if count > 1 else 0.0,
        "minimum_ms": ordered[0] / 1e6,
        "p10_ms": percentile(0.10),
        "p90_ms": percentile(0.90),
        "maximum_ms": ordered[-1] / 1e6,
    }


def benchmark_mode(
    mode: Mode,
    config: BenchmarkConfig,
    parameters: Sequence[tuple[Tensor, Tensor]],
    inputs: Tensor,
    output_adjoint: Tensor,
    validation_result: StepResult,
) -> dict[str, Any]:
    for _ in range(config.warmups):
        run_step(mode, parameters, inputs, output_adjoint, config.learning_rate)
    samples: dict[str, list[int]] = {"forward": [], "reverse": [], "update": [], "total": []}
    recomputation_counts: list[int] = []
    recomputation_bytes: list[int] = []
    for _ in range(config.repetitions):
        result = run_step(mode, parameters, inputs, output_adjoint, config.learning_rate)
        for phase in ("forward", "reverse", "update"):
            samples[phase].append(result.times_ns[phase])
        samples["total"].append(sum(result.times_ns.values()))
        recomputation_counts.append(result.recomputation_count)
        recomputation_bytes.append(result.recomputation_bytes)
    return {
        "timing": {phase: _timing_summary(values) for phase, values in samples.items()},
        "memory": validation_result.memory,
        "recomputation_layer_calls_per_step": {
            "minimum": min(recomputation_counts),
            "maximum": max(recomputation_counts),
            "median": statistics.median(recomputation_counts),
        },
        "recomputation_materialized_output_bytes_per_step": {
            "minimum": min(recomputation_bytes),
            "maximum": max(recomputation_bytes),
            "median": statistics.median(recomputation_bytes),
            "definition": "extra layer calls times batch*width*dtype_bytes; traffic, not retained or peak memory",
        },
    }


def _git_value(arguments: Sequence[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def environment_metadata(repository: Path) -> dict[str, Any]:
    tikitorch_repository = repository.parent / "tikitorch"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "git_commit": _git_value(["rev-parse", "HEAD"], repository),
        "git_branch": _git_value(["branch", "--show-current"], repository),
        "git_status_porcelain": _git_value(["status", "--porcelain"], repository),
        "tikitorch_git_commit": _git_value(["-C", str(tikitorch_repository), "rev-parse", "HEAD"], repository),
    }


def tensor_digest(tensors: Iterable[Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_record(
    config: BenchmarkConfig,
    repository: Path,
    command: str,
    run_id: str,
) -> dict[str, Any]:
    parameters, inputs, output_adjoint = make_problem(config)
    validation, validation_results = validate_single_step(
        config, parameters, inputs, output_adjoint
    )
    trajectory = validate_trajectory(config, parameters)
    modes = {
        mode: benchmark_mode(
            mode,
            config,
            parameters,
            inputs,
            output_adjoint,
            validation_results[mode],
        )
        for mode in MODES
    }
    parameter_count = sum(parameter.numel() for pair in parameters for parameter in pair)
    logical_bits = (
        config.depth
        * config.batch_size
        * config.width
        * bits_for_cardinality(config.width)
    )
    return {
        "schema": "route_sufficient_matched_benchmark.v1",
        "experiment_ids": ["RS-006", "RS-007"],
        "run_id": run_id,
        "status": "complete",
        "incident_status": "none",
        "scope": "small_cpu_reference_dag",
        "command": command,
        "config": {
            **config.__dict__,
            "device": "cpu",
            "dag": "alternating hard max/min-plus additive branch plus unit residual",
            "objective": "fixed linear readout sum(output * output_adjoint)",
            "tie_rule": "first candidate index",
            "optimizer": "plain SGD without momentum",
            "parameter_count": parameter_count,
            "input_shape": list(inputs.shape),
            "route_cardinality": config.width,
            "bits_per_route": bits_for_cardinality(config.width),
            "logical_trace_bits": logical_bits,
            "deterministic_problem_generation": True,
        },
        "input_and_adjoint_sha256": tensor_digest([inputs, output_adjoint]),
        "initial_parameter_sha256": tensor_digest(
            parameter for pair in parameters for parameter in pair
        ),
        "checkpoint_sha256": "not_applicable_synthetic_problem",
        "implementation_relations": {
            "cascade_vs_direct": (
                "identical reverse-topological computation and shared kernel for this acyclic DAG; "
                "timings are repeated labels, not evidence of an algorithmic speed distinction"
            ),
            "packed_vs_direct": "same reverse kernel after bit-packed route decoding",
            "checkpoint_vs_autograd": "same framework graph with per-layer non-reentrant recomputation",
        },
        "validation": validation,
        "trajectory_validation": trajectory,
        "modes": modes,
        "environment": environment_metadata(repository),
        "limitations": [
            "CPU microbenchmark of a reference residual DAG, not the full MorphSSM stack.",
            "Wall-clock values include Python dispatch and are not accelerator or hardware projections.",
            "Packed reverse decodes the complete tape before applying the reference reverse kernel.",
            "Autograd memory is saved-tensor storage, not process peak RSS; parameter and nonparameter bytes are separated.",
            "Checkpoint recomputation bytes are materialized output traffic, not retained or peak memory.",
            "The fixed linear loss isolates differentiation after supplying the output adjoint; nonlinear loss state is out of scope.",
        ],
    }


def parse_int_list(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_seed_list(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of nonnegative integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=parse_int_list, default=[32, 64])
    parser.add_argument("--depths", type=parse_int_list, default=[2, 4])
    parser.add_argument("--seeds", type=parse_seed_list, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trajectory-steps", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/results/route_certificates/reference/matched_benchmark.jsonl"),
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.warmups < 0 or args.repetitions < 1:
        parser.error("batch-size and repetitions must be positive; warmups must be nonnegative")
    if args.trajectory_steps < 1 or args.learning_rate <= 0 or args.torch_threads < 1:
        parser.error("trajectory-steps, learning-rate, and torch-threads must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    repository = Path(__file__).resolve().parents[2]
    output = args.output if args.output.is_absolute() else repository / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = "python -m scripts.route_certificates.matched_benchmark " + " ".join(
        [
            f"--widths {','.join(map(str, args.widths))}",
            f"--depths {','.join(map(str, args.depths))}",
            f"--seeds {','.join(map(str, args.seeds))}",
            f"--batch-size {args.batch_size}",
            f"--dtype {args.dtype}",
            f"--learning-rate {args.learning_rate}",
            f"--warmups {args.warmups}",
            f"--repetitions {args.repetitions}",
            f"--trajectory-steps {args.trajectory_steps}",
            f"--torch-threads {args.torch_threads}",
            f"--output {args.output}",
        ]
    )
    run_id = str(uuid.uuid4())
    total = len(args.seeds) * len(args.widths) * len(args.depths)
    completed = 0
    with output.open("a", encoding="utf-8") as handle:
        for seed in args.seeds:
            for width in args.widths:
                for depth in args.depths:
                    config = BenchmarkConfig(
                        seed=seed,
                        width=width,
                        depth=depth,
                        batch_size=args.batch_size,
                        dtype=args.dtype,
                        learning_rate=args.learning_rate,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                        trajectory_steps=args.trajectory_steps,
                    )
                    record = build_record(config, repository, command, run_id)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    completed += 1
                    medians = {
                        mode: round(record["modes"][mode]["timing"]["total"]["median_ms"], 3)
                        for mode in MODES
                    }
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": total,
                                "seed": seed,
                                "width": width,
                                "depth": depth,
                                "total_median_ms": medians,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
