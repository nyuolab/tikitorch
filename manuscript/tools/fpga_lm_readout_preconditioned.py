"""Training-Gram coordinates for the unchanged frozen-feature affine objective.

Only optimization coordinates change. Physical ridge, logits, and the stopping
gradient remain those of fpga_lm_readout_oracle; no feature whitening changes
the statistical objective and no floating-point certificate is asserted.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from manuscript.tools import fpga_lm_readout_oracle as oracle

SCHEMA = "tikitorch.fpga_lm_readout_preconditioned.v1"
REGULARIZATION, GAP_TARGET = oracle.REGULARIZATION, oracle.GAP_TARGET
MAX_ITERATIONS, MAX_FUNCTION_CALLS = 200, 400
DEADLINE_SECONDS, CHUNK_SIZE = 3600, 2048
ATOL, RTOL = 1e-9, 1e-11
BACKWARD_FACTOR = 64.0
INVERSE_ABSOLUTE_LIMIT = 1e-6


def source_identity():
    path = Path(__file__).resolve()
    return {**oracle.source_identity(),
            path.relative_to(path.parents[2]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}


def _features(h):
    oracle._array(h, "training features")
    if h.ndim != 2:
        raise ValueError("complete training feature matrix required")
    return h.shape


def _near(actual, expected, label):
    oracle._array(actual, label)
    oracle._array(expected, label)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        raise ValueError(label + " independent numerical mismatch")


def transform_digest(transform):
    digest = hashlib.sha256()
    for name in ("M", "L", "S"):
        value = transform[name]
        oracle._array(value, "preconditioner " + name)
        digest.update(str((name, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _residuals(matrix, lower, inverse):
    """Fixed backward-error checks, not exact whitening or a roundoff proof."""
    for name, value in (("M", matrix), ("L", lower), ("S", inverse)):
        oracle._array(value, "preconditioner " + name)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or lower.shape != matrix.shape or inverse.shape != matrix.shape):
        raise ValueError("complete square preconditioner geometry required")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("preconditioner M must be exactly symmetric")
    if not np.array_equal(lower, np.tril(lower)) or not np.all(np.diag(lower) > 0):
        raise ValueError("positive-diagonal lower Cholesky factor required")
    n = matrix.shape[0]
    identity = np.eye(n, dtype=np.float64)
    def norm(value):
        return float(np.linalg.norm(value, ord=np.inf))
    matrix_norm, lower_norm, inverse_norm = norm(matrix), norm(lower.T), norm(inverse)
    factor_error = norm(lower @ lower.T - matrix)
    inverse_error = norm(lower.T @ inverse - identity)
    factor_relative = factor_error / max(matrix_norm, np.finfo(np.float64).tiny)
    inverse_scale = 1 + lower_norm * inverse_norm
    inverse_relative = inverse_error / inverse_scale
    whitening_error = norm(inverse.T @ matrix @ inverse - identity)
    # Receipts must preserve literal Python scalar types across JSON roundtrip.
    threshold = float(BACKWARD_FACTOR * n * np.finfo(np.float64).eps)
    values = (matrix_norm, lower_norm, inverse_norm, factor_error, inverse_error,
              factor_relative, inverse_relative, inverse_scale, whitening_error, threshold)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite preconditioner residual or norm")
    if factor_relative > threshold or inverse_relative > threshold or inverse_error > INVERSE_ABSOLUTE_LIMIT:
        raise ValueError("preconditioner violates fixed backward-error or inverse sanity guards")
    return {"cholesky_relative_residual": factor_relative,
            "inverse_relative_residual": inverse_relative,
            "inverse_absolute_residual": inverse_error,
            "whitening_absolute_residual": whitening_error,
            "backward_tolerance": threshold, "inverse_absolute_limit": INVERSE_ABSOLUTE_LIMIT}


def _gram(h, chunk_size, checkpoint, independent=False):
    positions, d = _features(h)
    oracle._positive(chunk_size, "Gram chunk size")
    width = d + 1
    if independent:
        result = torch.eye(width, dtype=torch.float64) * REGULARIZATION
        for start in range(0, positions, chunk_size):
            if checkpoint is not None:
                checkpoint()
            block = torch.tensor(h[start:start + chunk_size], dtype=torch.float64)
            augmented = torch.cat((block, torch.ones((len(block), 1), dtype=torch.float64)), dim=1)
            result = result + augmented.T @ augmented / (2 * positions)
            if checkpoint is not None:
                checkpoint()
        matrix = result.numpy()
    else:
        matrix = np.eye(width, dtype=np.float64) * REGULARIZATION
        for start in range(0, positions, chunk_size):
            if checkpoint is not None:
                checkpoint()
            block = h[start:start + chunk_size]
            augmented = np.concatenate((block, np.ones((len(block), 1), dtype=np.float64)), axis=1)
            matrix += augmented.T @ augmented / (2 * positions)
            if checkpoint is not None:
                checkpoint()
    oracle._array(matrix, "complete regularized training Gram")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("training Gram accumulation lost exact symmetry")
    return matrix


def build_preconditioner(h, *, chunk_size=CHUNK_SIZE, checkpoint=None):
    """M=lambda I+X.T X/(2N), X=[training h,1]; no labels or dev input."""
    from scipy.linalg import solve_triangular

    positions, d = _features(h)
    matrix = _gram(h, chunk_size, checkpoint)
    if checkpoint is not None:
        checkpoint()
    lower = np.linalg.cholesky(matrix)
    inverse = solve_triangular(lower.T, np.eye(d + 1, dtype=np.float64), lower=False, check_finite=True)
    # Persist and use one canonical memory layout. LAPACK triangular solves
    # may return Fortran arrays; switching layouts only inside solve could
    # otherwise change BLAS reduction choices during exact endpoint readback.
    result = {name: np.ascontiguousarray(value) for name, value in (("M", matrix), ("L", lower), ("S", inverse))}
    result["summary"] = _summary(h, result, chunk_size)
    if checkpoint is not None:
        checkpoint()
    return result


def _summary(h, transform, chunk_size):
    positions, d = _features(h)
    matrix, lower, inverse = (transform[key] for key in ("M", "L", "S"))
    residuals = _residuals(matrix, lower, inverse)
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition < 1:
        raise ValueError("finite estimated Gram condition number required")
    return {"schema": SCHEMA, "positions": positions, "features": d,
                         "augmented_features": d + 1, "regularization": REGULARIZATION,
                         "chunk_size": chunk_size,
                         "construction": "lambda*I + sum_chunks(X.T@X/(2*N)), X=[training h,ones]",
                         "orientation": "M=L@L.T; S=L^(-T); A=U@S.T; G_U=G_A@S",
                         "training_feature_sha256": oracle._digest(h),
                         "transform_sha256": transform_digest(transform),
                         "estimated_condition_number": condition, **residuals,
                         "independent_atol": ATOL, "independent_rtol": RTOL,
                         "formal_certification": False, "development_used": False,
                         "labels_used": False, "physical_objective_changed": False}


def check_preconditioner(transform, h, *, independent=False, chunk_size=CHUNK_SIZE, checkpoint=None):
    """Check the train-only construction and factors; do not modify the map."""
    if type(independent) is not bool or not isinstance(transform, dict) or set(transform) != {"M", "L", "S", "summary"}:
        raise ValueError("complete typed training preconditioner required")
    positions, d = _features(h)
    summary = transform["summary"]
    if not isinstance(summary, dict):
        raise ValueError("complete preconditioner provenance required")
    for name in ("M", "L", "S"):
        oracle._array(transform[name], "saved preconditioner " + name)
        if not transform[name].flags.c_contiguous:
            raise ValueError("saved preconditioner arrays require canonical C-contiguous layout")
    if transform["M"].shape != (d + 1, d + 1):
        raise ValueError("preconditioner differs from actual feature width")
    expected = _summary(h, transform, chunk_size)
    if set(summary) != set(expected):
        raise ValueError("complete exact preconditioner summary fields required")
    for key, value in expected.items():
        if type(summary.get(key)) is not type(value) or summary[key] != value:
            raise ValueError("preconditioner provenance differs: " + key)
    matrix = _gram(h, chunk_size, checkpoint, independent)
    if independent:
        _near(transform["M"], matrix, "independent training Gram")
        with torch.no_grad():
            lower = torch.linalg.cholesky(torch.tensor(matrix))
            inverse = torch.linalg.solve_triangular(lower.T, torch.eye(d + 1, dtype=torch.float64), upper=True)
        _residuals(matrix, lower.numpy(), inverse.numpy())
        _residuals(matrix, transform["L"], transform["S"])
    elif transform["M"].tobytes() != matrix.tobytes():
        raise ValueError("exact original training Gram bytes differ")
    if checkpoint is not None:
        checkpoint()
    return True


def verify_preconditioner(h, factor, *, independent=False, chunk_size=CHUNK_SIZE, checkpoint=None):
    return check_preconditioner(factor, h, independent=independent,
                                chunk_size=chunk_size, checkpoint=checkpoint)


def materialize(U, S):
    """Physical flat W-then-b coordinates from row-major U[V,D+1]."""
    oracle._array(U, "optimization coordinates U")
    oracle._array(S, "coordinate map S")
    if (U.ndim != 2 or S.ndim != 2 or S.shape[0] != S.shape[1]
            or U.shape[1] != S.shape[0] or U.shape[0] < 2 or U.shape[1] < 2):
        raise ValueError("complete U[V,D+1] and S[D+1,D+1] required")
    physical = np.zeros_like(U) if not np.any(U) else U @ S.T
    oracle._array(physical, "materialized physical affine coefficients")
    return np.concatenate((physical[:, :-1].ravel(), physical[:, -1]))


def pullback(gradient, S):
    oracle._array(gradient, "physical full gradient")
    oracle._array(S, "coordinate map S")
    if gradient.ndim != 1 or S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 2:
        raise ValueError("full flat physical gradient and square coordinate map required")
    width = S.shape[0]
    if len(gradient) % width or len(gradient) // width < 2:
        raise ValueError("physical gradient geometry differs from coordinate map")
    vocab, d = len(gradient) // width, width - 1
    weights, bias = oracle.unpack(gradient, d, vocab, "affine")
    result = np.concatenate((weights, bias[:, None]), axis=1) @ S
    oracle._array(result, "transformed full gradient")
    return result


def objective(U, h, z0, targets, transform, *, independent=False, chunk_size=CHUNK_SIZE, checkpoint=None):
    """Evaluate the ORIGINAL physical objective, then pull its full gradient."""
    if type(independent) is not bool:
        raise ValueError("literal independent mode required")
    oracle._array(U, "optimization coordinates U")
    if not isinstance(transform, dict) or "S" not in transform:
        raise ValueError("explicit bound coordinate map required")
    _, d, vocab = oracle.validate_data(h, z0, targets)
    if U.shape != (vocab, d + 1):
        raise ValueError("optimization coordinates differ from physical data geometry")
    S = transform["S"]
    theta = materialize(U, S)
    function = oracle.independent_objective if independent else oracle.objective
    physical = function(theta, h, z0, targets, "affine", chunk_size=chunk_size, checkpoint=checkpoint)
    physical_gradient = physical["gradient"]
    if independent:
        leaves = torch.tensor(U, dtype=torch.float64, requires_grad=True)
        with torch.enable_grad():
            values = leaves @ torch.tensor(S).T
            flat = torch.cat((values[:, :-1].reshape(-1), values[:, -1]))
            coordinate = torch.autograd.grad(flat, leaves, grad_outputs=torch.tensor(physical_gradient))[0].detach().numpy()
        _near(coordinate, pullback(physical_gradient, S), "independent materialization pullback")
        _near(flat.detach().numpy(), theta, "independent physical coefficient materialization")
    else:
        coordinate = pullback(physical_gradient, S)
    norm = float(np.dot(coordinate.ravel(), coordinate.ravel()))
    if not math.isfinite(norm):
        raise ValueError("nonfinite transformed gradient norm")
    return {**{key: value for key, value in physical.items() if key != "gradient"},
            "theta": theta, "physical_gradient": physical_gradient,
            "coordinate_gradient": coordinate, "transformed_gradient_norm_sq": norm}


def solve(h, z0, targets, *, preconditioner=None, chunk_size=CHUNK_SIZE, checkpoint=None):
    """One zero-start coordinate solve, stopping ONLY on physical full G.

    A supplied map receives a fresh training-Gram verification before fitting;
    an omitted map is built once. This separately accounted preparation uses
    the caller's global checkpoint; the fit timer starts after it completes.
    All function calls are counted before work, accepted endpoint evaluations
    are cached, and no extra final query or restart is performed on a cap.
    The 3600-second per-fit cap is checkpoint-enforced; a runner's global
    watchdog separately covers in-flight native operations.
    """
    from scipy.optimize import minimize

    positions, d, vocab = oracle.validate_data(h, z0, targets)
    oracle._positive(chunk_size, "chunk size")
    shape = (vocab, d + 1)
    size = vocab * (d + 1)
    zero = np.zeros(size, dtype=np.float64)
    history, iteration_history = [], []
    initial = accepted = last = None
    calls, iterations = 0, 0
    scipy_status, scipy_message, scipy_success = None, None, False
    reason = "not_started"

    def clocks():
        elapsed_utc = time.time() - started_utc
        elapsed_mono = time.monotonic() - started_mono
        if (not math.isfinite(elapsed_utc) or not math.isfinite(elapsed_mono)
                or elapsed_utc < 0 or elapsed_mono < 0):
            raise RuntimeError("invalid preconditioned solve clock")
        return elapsed_utc, elapsed_mono

    def check():
        if checkpoint is not None:
            checkpoint()
        if max(clocks()) >= DEADLINE_SECONDS:
            raise oracle._Stop("deadline")

    # Admission failures are not an evaluable capped fit: no physical endpoint
    # objective/gradient exists until the first complete counted FG evaluation.
    if preconditioner is None:
        transform = build_preconditioner(h, chunk_size=chunk_size, checkpoint=checkpoint)
    else:
        check_preconditioner(preconditioner, h, chunk_size=chunk_size, checkpoint=checkpoint)
        transform = {name: value.copy() for name, value in preconditioner.items() if name != "summary"}
        transform["summary"] = dict(preconditioner["summary"])
    original_transform = transform_digest(transform)
    started_utc, started_mono = time.time(), time.monotonic()

    def row(point, value):
        utc, mono = clocks()
        result = {key: value[key] for key in ("objective", "ce", "penalty", "gradient_norm_sq",
                                             "numerical_gap_estimate", "numerical_target_met", "accuracy",
                                             "transformed_gradient_norm_sq")}
        return {**result, "function_call": calls,
                "U_sha256": oracle._digest(point),
                "theta_sha256": oracle._digest(value["theta"]),
                "gradient_sha256": oracle._digest(value["physical_gradient"]),
                "transformed_gradient_sha256": oracle._digest(value["coordinate_gradient"]),
                "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": mono}

    def fg(point):
        nonlocal calls, initial, accepted, last
        check()
        if calls >= MAX_FUNCTION_CALLS:
            raise oracle._Stop("function_call_limit")
        calls += 1
        value = objective(point.reshape(shape), h, z0, targets, transform,
                          chunk_size=chunk_size, checkpoint=check)
        point = point.copy()
        last = (point, value)
        history.append(row(point, value))
        if initial is None:
            if point.tobytes() != zero.tobytes():
                raise RuntimeError("L-BFGS-B did not evaluate the prescribed zero U")
            initial = accepted = last
            if value["numerical_target_met"]:
                raise oracle._Stop("numerical_gap_target")
        return value["objective"], value["coordinate_gradient"].ravel().copy()

    def callback(point):
        nonlocal accepted, iterations
        if last is None or point.tobytes() != last[0].tobytes():
            raise RuntimeError("accepted iterate lacks its actual counted coordinate/physical FG")
        accepted = last
        iterations += 1
        iteration_history.append({**row(*accepted), "iteration": iterations})
        if iterations % 10 == 0:
            print(json.dumps({"stage": "preconditioned_iteration", "iteration": iterations,
                              "function_calls": calls, "objective": accepted[1]["objective"],
                              "ce": accepted[1]["ce"],
                              "physical_numerical_gap_estimate": accepted[1]["numerical_gap_estimate"]}), flush=True)
        check()
        if accepted[1]["numerical_target_met"]:
            raise oracle._Stop("numerical_gap_target")
        if iterations >= MAX_ITERATIONS:
            raise oracle._Stop("iteration_limit")

    # A transformed max-component gradient tolerance is NOT the physical
    # gradient-gap criterion. Disable it; inspect the latter at accepted U.
    options = {"maxiter": MAX_ITERATIONS, "maxfun": MAX_FUNCTION_CALLS,
               "maxls": 20, "maxcor": 10, "ftol": 0.0, "gtol": 0.0}
    try:
        result = minimize(fg, zero, method="L-BFGS-B", jac=True, bounds=None,
                          callback=callback, options=options)
        scipy_status, scipy_message, scipy_success = int(result.status), str(result.message), bool(result.success)
        if accepted is None or result.x.tobytes() != accepted[0].tobytes():
            raise RuntimeError("returned point differs from the last evaluated accepted coordinate iterate")
        if accepted[1]["numerical_target_met"]:
            reason = "numerical_gap_target"
        elif calls >= MAX_FUNCTION_CALLS:
            reason = "function_call_limit"
        elif iterations >= MAX_ITERATIONS:
            reason = "iteration_limit"
        else:
            reason = "scipy_stopped_above_numerical_target"
    except oracle._Stop as stopped:
        reason = stopped.reason
    if initial is None or accepted is None:
        raise RuntimeError("solve stopped before a complete initial physical objective/gradient")
    if transform_digest(transform) != original_transform:
        raise RuntimeError("coordinate map changed during the fixed solve")
    point, value = accepted
    utc, mono = clocks()
    if max(utc, mono) >= DEADLINE_SECONDS:
        reason = "deadline"
    target = bool(value["numerical_target_met"])
    completed = reason == "numerical_gap_target" and target
    summary = {"schema": SCHEMA, "arm": "affine", "positions": positions, "features": d, "vocabulary": vocab,
               "parameter_count": size, "regularization": REGULARIZATION, "options": options,
               "function_calls": calls, "complete_function_evaluations": len(history), "iterations": iterations,
               "stopping_reason": reason, "scipy_status": scipy_status, "scipy_message": scipy_message,
               "scipy_success": scipy_success, "numerical_target_met": target, "numerically_converged": completed,
               "inconclusive": not completed, "formal_certification": False, "rounding_error_bound": None,
               "gap_target": GAP_TARGET, "numerical_gap_estimate": value["numerical_gap_estimate"],
               "objective": value["objective"], "ce": value["ce"], "penalty": value["penalty"],
               "gradient_norm_sq": value["gradient_norm_sq"], "accuracy": value["accuracy"],
               "transformed_gradient_norm_sq": value["transformed_gradient_norm_sq"],
               "transform_sha256": original_transform, "stopping_gradient": "physical",
               "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": mono,
               "deadline_seconds": DEADLINE_SECONDS, "zero_start": True,
               "returned_point": "last evaluated accepted iterate", "restarts": 0,
               "trial_history_scope": "scalar provenance only; independent audit is endpoint-only"}

    def endpoint(record):
        coordinates, values = record
        return {"U": coordinates.reshape(shape).copy(), "theta": values["theta"].copy(),
                "physical_gradient": values["physical_gradient"].copy(),
                "coordinate_gradient": values["coordinate_gradient"].copy(),
                "statistics": {key: item for key, item in values.items()
                               if key not in ("theta", "physical_gradient", "coordinate_gradient")}}

    final = endpoint(accepted)
    return {**final, "preconditioner": transform, "summary": summary, "history": history,
            "iteration_history": iteration_history, "initial": endpoint(initial), "final": final}
