"""Strict endpoint-bound receipts for the preconditioned readout oracle.

This module evaluates no objective, model, data Gram, or optimizer. Scalar
trial records are provenance only; only retained initial/final arrays are
bound to independently recomputed physical endpoint statistics. Coordinate
materialization and gradient pullback use the same immutable saved transform.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from manuscript.tools import fpga_lm_readout_oracle as oracle
from manuscript.tools import fpga_lm_readout_preconditioned as preconditioned


def _equal(actual, expected, label):
    if type(actual) is not type(expected):
        raise ValueError(label + ": literal types differ")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(label + ": fields differ")
        for key in expected:
            _equal(actual[key], expected[key], label + ": " + key)
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise ValueError(label + ": lengths differ")
        for left, right in zip(actual, expected):
            _equal(left, right, label)
    elif actual != expected:
        raise ValueError(label + ": values differ")


def _finite(value, label):
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError(label + ": finite nonnegative literal float required")
    return value


def _integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(label + ": bounded literal integer required")
    return value


def _close(actual, expected, label):
    if not math.isfinite(expected) or not math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-11):
        raise ValueError(label + ": independently checked endpoint differs")


def _tensor(value, shape, label):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float64 or value.layout != torch.strided
            or value.requires_grad or value.shape != shape or not bool(torch.isfinite(value).all())):
        raise ValueError(label + ": finite detached complete CPU FP64 tensor required")
    return value.contiguous().numpy()


def _bytes_equal(actual, expected, label):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
        raise ValueError(label + ": exact array bytes differ")


def validate_solver_record(solver, endpoints, arrays, data, transform):
    """Validate physical stopping and accepted coordinate/physical archives.

    No full-trial replay is implied. One counted evaluation may lack a complete
    history row only when the solve was interrupted by its declared deadline.
    """
    if not isinstance(data, dict) or set(data) != {"h", "logits", "targets"}:
        raise ValueError("complete immutable training data mapping required")
    positions, width, vocab = oracle.validate_data(data["h"], data["logits"], data["targets"])
    size = oracle.parameter_count(width, vocab, "affine")
    if not isinstance(transform, dict) or set(transform) != {"M", "L", "S", "summary"}:
        raise ValueError("complete saved preconditioner required")
    transform_hash = preconditioned.transform_digest(transform)
    if not isinstance(solver, dict) or set(solver) != {"summary", "history", "iteration_history"}:
        raise ValueError("complete solver scalar record required")
    summary, history, accepted = (solver[key] for key in ("summary", "history", "iteration_history"))
    if not isinstance(summary, dict) or type(history) is not list or type(accepted) is not list:
        raise ValueError("typed solver summary and complete histories required")
    physical_keys = {"objective", "ce", "penalty", "gradient_norm_sq", "numerical_gap_estimate", "accuracy"}
    scalar_keys = physical_keys | {"transformed_gradient_norm_sq"}
    clock_keys = {"elapsed_utc_seconds", "elapsed_monotonic_seconds"}
    summary_keys = scalar_keys | clock_keys | {
        "schema", "arm", "positions", "features", "vocabulary", "parameter_count", "regularization",
        "options", "function_calls", "complete_function_evaluations", "iterations", "stopping_reason",
        "scipy_status", "scipy_message", "scipy_success", "numerical_target_met", "numerically_converged",
        "inconclusive", "formal_certification", "rounding_error_bound", "gap_target", "deadline_seconds",
        "zero_start", "returned_point", "restarts", "trial_history_scope", "transform_sha256", "stopping_gradient"}
    _equal(set(summary), summary_keys, "complete solver summary fields")
    expected = {
        "schema": preconditioned.SCHEMA, "arm": "affine", "positions": positions, "features": width,
        "vocabulary": vocab, "parameter_count": size, "regularization": oracle.REGULARIZATION,
        "gap_target": oracle.GAP_TARGET, "deadline_seconds": preconditioned.DEADLINE_SECONDS,
        "zero_start": True, "restarts": 0, "formal_certification": False, "rounding_error_bound": None,
        "returned_point": "last evaluated accepted iterate",
        "trial_history_scope": "scalar provenance only; independent audit is endpoint-only",
        "transform_sha256": transform_hash, "stopping_gradient": "physical",
        "options": {"maxiter": preconditioned.MAX_ITERATIONS, "maxfun": preconditioned.MAX_FUNCTION_CALLS,
                    "maxls": 20, "maxcor": 10, "ftol": 0.0, "gtol": 0.0}}
    for key, value in expected.items():
        _equal(summary[key], value, "fixed solver declaration: " + key)

    def statistics(row, label, *, transformed=True):
        for key in scalar_keys if transformed else physical_keys:
            _finite(row[key], label + ": " + key)
        if row["accuracy"] > 1:
            raise ValueError("accuracy must lie in [0,1]")
        _equal(row["objective"], row["ce"] + row["penalty"], label + ": physical CE plus ridge")
        _equal(row["numerical_gap_estimate"], row["gradient_norm_sq"] / (2 * oracle.REGULARIZATION),
               label + ": physical gradient-norm gap estimate")
        _equal(row["numerical_target_met"], row["numerical_gap_estimate"] <= oracle.GAP_TARGET,
               label + ": actual physical numerical target")

    statistics(summary, "solver endpoint summary")
    for key in clock_keys:
        _finite(summary[key], "solver " + key)
    calls = _integer(summary["function_calls"], 1, preconditioned.MAX_FUNCTION_CALLS, "function-call count")
    complete = _integer(summary["complete_function_evaluations"], 1, calls, "complete evaluation count")
    iterations = _integer(summary["iterations"], 0, preconditioned.MAX_ITERATIONS, "accepted iteration count")
    _equal(len(history), complete, "all completed evaluations retained")
    _equal(len(accepted), iterations, "all accepted iterations retained")
    reason = summary["stopping_reason"]
    if type(reason) is not str or reason not in {
            "numerical_gap_target", "deadline", "function_call_limit", "iteration_limit",
            "scipy_stopped_above_numerical_target"}:
        raise ValueError("declared terminal solver reason required")
    if calls - complete not in (0, 1) or (calls != complete and reason != "deadline"):
        raise ValueError("only deadline may interrupt one counted full evaluation")
    expired = max(summary[key] for key in clock_keys) >= preconditioned.DEADLINE_SECONDS
    _equal(reason == "deadline", expired, "deadline matches actual clocks")
    if reason == "function_call_limit":
        _equal(calls, preconditioned.MAX_FUNCTION_CALLS, "actual function-call cap reached")
    if reason == "iteration_limit":
        _equal(iterations, preconditioned.MAX_ITERATIONS, "actual iteration cap reached")
    if reason == "numerical_gap_target" and not summary["numerical_target_met"]:
        raise ValueError("numerical stopping reason requires the physical endpoint target")
    if reason == "scipy_stopped_above_numerical_target" and summary["numerical_target_met"]:
        raise ValueError("above-target solver stop cannot claim the target")
    converged = reason == "numerical_gap_target" and summary["numerical_target_met"]
    _equal(summary["numerically_converged"], converged, "actual numerical convergence")
    _equal(summary["inconclusive"], not converged, "bounded nonconvergence remains inconclusive")
    status, message, success = (summary[key] for key in ("scipy_status", "scipy_message", "scipy_success"))
    if type(success) is not bool:
        raise ValueError("literal SciPy success flag required")
    if status is None:
        if message is not None or success or reason == "scipy_stopped_above_numerical_target":
            raise ValueError("interrupted solver cannot fabricate a SciPy return")
    else:
        _integer(status, 0, 2, "SciPy L-BFGS-B status")
        if type(message) is not str or not message or success != (status == 0):
            raise ValueError("consistent actual SciPy status/message required")

    hash_keys = {key + "_sha256" for key in ("U", "theta", "gradient", "transformed_gradient")}
    row_keys = scalar_keys | clock_keys | hash_keys | {"numerical_target_met", "function_call"}
    previous_mono = 0.0
    for ordinal, row in enumerate(history, 1):
        if not isinstance(row, dict):
            raise ValueError("complete evaluation row required")
        _equal(set(row), row_keys, "exact evaluation provenance fields")
        _equal(_integer(row["function_call"], 1, calls, "evaluation ordinal"), ordinal, "ordered evaluations")
        statistics(row, "function evaluation")
        for key in clock_keys:
            _finite(row[key], "evaluation " + key)
        if not previous_mono <= row["elapsed_monotonic_seconds"] <= summary["elapsed_monotonic_seconds"]:
            raise ValueError("ordered monotonic evaluation clocks required")
        previous_mono = row["elapsed_monotonic_seconds"]
        for key in hash_keys:
            digest = row[key]
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("literal complete evaluation SHA-256 required")
    previous_call, previous_mono = 1, 0.0
    for ordinal, row in enumerate(accepted, 1):
        if not isinstance(row, dict):
            raise ValueError("complete accepted-iteration row required")
        _equal(set(row), row_keys | {"iteration"}, "exact accepted provenance fields")
        _equal(_integer(row["iteration"], 1, iterations, "accepted ordinal"), ordinal, "ordered accepted iterations")
        call = _integer(row["function_call"], previous_call + 1, complete, "accepted evaluation reference")
        reference = history[call - 1]
        for key in row_keys - clock_keys:
            _equal(row[key], reference[key], "accepted iterate binds a complete evaluation")
        for key in clock_keys:
            _finite(row[key], "accepted " + key)
        mono = row["elapsed_monotonic_seconds"]
        if not max(previous_mono, reference["elapsed_monotonic_seconds"]) <= mono <= summary["elapsed_monotonic_seconds"]:
            raise ValueError("accepted-iteration monotonic clocks required")
        previous_call, previous_mono = call, mono

    if not isinstance(arrays, dict) or not isinstance(endpoints, dict):
        raise ValueError("typed endpoint arrays and statistics required")
    _equal(set(arrays), {"initial", "final"}, "both actual endpoint archives")
    _equal(set(endpoints), {"initial", "final"}, "both recomputed endpoint statistics")
    final_row = accepted[-1] if accepted else history[0]
    for name, row in (("initial", history[0]), ("final", final_row)):
        archive = arrays[name]
        if not isinstance(archive, dict):
            raise ValueError("typed endpoint archive required")
        _equal(set(archive), {"U", "theta", "gradient", "transformed_gradient"}, "complete endpoint archive")
        values = {}
        for key, value in archive.items():
            shape = (vocab, width + 1) if key in {"U", "transformed_gradient"} else (size,)
            values[key] = _tensor(value, shape, "archived " + key)
            _equal(hashlib.sha256(values[key].tobytes()).hexdigest(), row[key + "_sha256"],
                   "accepted endpoint binds archived " + key)
        if name == "initial":
            for key in ("U", "theta"):
                _bytes_equal(values[key], np.zeros_like(values[key]), "positive-zero initial " + key)
        _bytes_equal(values["theta"], preconditioned.materialize(values["U"], transform["S"]),
                     "physical coefficients materialize actual coordinates")
        _bytes_equal(values["transformed_gradient"], preconditioned.pullback(values["gradient"], transform["S"]),
                     "actual physical-gradient coordinate pullback")
        _close(row["penalty"], oracle.REGULARIZATION / 2 * float(np.dot(values["theta"], values["theta"])),
               "actual physical ridge")
        _close(row["gradient_norm_sq"], float(np.dot(values["gradient"], values["gradient"])), "physical gradient norm")
        transformed = values["transformed_gradient"].ravel()
        _close(row["transformed_gradient_norm_sq"], float(np.dot(transformed, transformed)), "coordinate gradient norm")
        endpoint = endpoints[name]
        if not isinstance(endpoint, dict):
            raise ValueError("typed recomputed endpoint statistics required")
        _equal(set(endpoint), physical_keys | {"numerical_target_met", "positions", "regularization", "formal_certification"},
               "complete recomputed physical endpoint statistics")
        statistics(endpoint, "recomputed physical endpoint", transformed=False)
        for key, value in {"positions": positions, "regularization": oracle.REGULARIZATION,
                           "formal_certification": False}.items():
            _equal(endpoint[key], value, "recomputed endpoint declaration")
        for key in physical_keys:
            _close(row[key], endpoint[key], "recomputed endpoint " + key)
    for key in scalar_keys | {"numerical_target_met"}:
        _equal(summary[key], final_row[key], "summary is the last accepted evaluated endpoint")
