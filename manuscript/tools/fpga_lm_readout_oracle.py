"""Frozen-feature residual readout controls; no corpus or model fitting on import.

The numerical strong-convexity gap estimate is not a rounding-certified bound.
Features and baseline logits are never rescaled, centered, or differentiated.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch

from manuscript.tools import fpga_lm_exact_fit as helper
from manuscript.tools import fpga_lm_guarded_forward as guarded

SCHEMA = "tikitorch.fpga_lm_readout_oracle.v1"
ARMS = ("bias", "affine")
REGULARIZATION = 1e-3
GAP_TARGET = 1e-5
MAX_ITERATIONS = 200
MAX_FUNCTION_CALLS = 400
DEADLINE_SECONDS = 3600
CHUNK_SIZE = 2048
GUARD_KEYS = ("structure_certified", "whole_model_dense_fallback",
              "projection_fast_operations", "projection_dense_operations",
              "head_fast_operations", "head_dense_operations")


def source_identity():
    path = Path(__file__).resolve()
    return {**guarded.source_identity(),
            path.relative_to(path.parents[2]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(label + " must be a positive literal integer")


def _exact(actual, expected, label):
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.contiguous().numpy().tobytes() != expected.contiguous().numpy().tobytes()):
        raise ValueError(label + " differs in shape, dtype, or tensor bytes")


@torch.no_grad()
def _from_tape(weights, tokens, base):
    """Gather the complete head-input vector, retaining producer association."""
    helper.trace(base)
    geometry = helper.reference._geometry(weights, tokens)
    routes, offsets = helper.reference._base(base, geometry, tokens, base["use_context"])
    if base.get("parameter_sha256") != helper.parameter_digest(weights):
        raise ValueError("feature tape belongs to different parameter bytes")
    d, n, length = (geometry[key] for key in ("D", "N", "T"))
    coordinates = torch.arange(d)
    classes = torch.arange(geometry["V"])
    history = [[] for _ in range(n)]
    features, logits = [], []
    for position in range(length):
        residual = weights["EMBED"][tokens[position]]
        for layer in range(n):
            centered = residual - residual.mean() + offsets[layer, position]
            helper._finite(centered, "feature tape centering", detached=True)
            history[layer].append(centered)
            tap = routes["TAP_IDX"][layer, position]
            past = torch.stack(history[layer])
            conv = past[position - tap * (1 << min(layer, 6)), coordinates] + weights["MH"][layer, coordinates, tap]
            helper._finite(conv, "feature tape temporal selection", detached=True)
            projection = routes["PROJ_IDX"][layer, position]
            projected = weights["MP"][layer, coordinates, projection] + conv[projection]
            projected = projected + weights["MB"][layer]
            helper._finite(projected, "feature tape projection plus bias", detached=True)
            residual = residual + projected
            helper._finite(residual, "feature tape residual", detached=True)
        head = routes["AD_HEAD"][position]
        logits.append(weights["W_HEAD"][classes, head] + residual[head])
        features.append(residual.clone())
    features, logits = torch.stack(features), torch.stack(logits)
    _exact(logits, base["logits"], "feature-reconstructed frozen logits")
    return features


@torch.no_grad()
def _dense_features(weights, tokens, context):
    """Fresh dynamic full-D selectors; never consumes a saved route tape."""
    geometry = helper.reference._geometry(weights, tokens)
    n, k, length = (geometry[key] for key in ("N", "K", "T"))
    history = [[] for _ in range(n)]
    taps, projections = ([[] for _ in range(n)] for _ in range(2))
    features, logits, heads = [], [], []
    for position in range(length):
        residual = weights["EMBED"][tokens[position]]
        for layer in range(n):
            centered = residual - residual.mean() + residual.new_zeros(())
            history[layer].append(helper._finite(centered, "independent feature centering"))
            dilation = 1 << min(layer, 6)
            count = min(k, position // dilation + 1) if context else 1
            candidates = torch.stack([history[layer][position - tap * dilation]
                                      + weights["MH"][layer, :, tap] for tap in range(count)])
            conv, tap = candidates.max(dim=0) if layer % 2 == 0 else candidates.min(dim=0)
            helper._finite(conv, "independent feature temporal selection", detached=True)
            projected, projection = (weights["MP"][layer] + conv[None, :]).max(dim=1)
            residual = residual + (projected + weights["MB"][layer])
            helper._finite(residual, "independent feature residual", detached=True)
            taps[layer].append(tap)
            projections[layer].append(projection)
        scores, head = (weights["W_HEAD"] + residual[None, :]).max(dim=1)
        features.append(residual.clone())
        logits.append(helper._finite(scores, "independent feature logits", detached=True))
        heads.append(head)
    base = {"schema": helper.reference.SCHEMA, "mode": "smooth", "geometry": geometry,
            "tokens": tokens.clone(), "use_context": context, "logits": torch.stack(logits),
            "centering_offsets": torch.zeros((n, length), dtype=torch.float64),
            "routes": {"AD_HEAD": torch.stack(heads),
                       "PROJ_IDX": torch.stack([torch.stack(rows) for rows in projections]),
                       "TAP_IDX": torch.stack([torch.stack(rows) for rows in taps])},
            "helper_schema": helper.SCHEMA, "units": "dimensionless",
            "parameter_sha256": helper.parameter_digest(weights), "native_execution": False,
            "independent_forward": True}
    helper.trace(base)
    return torch.stack(features), base


def extract_features(weights, tokens, independent=False, *, base=None, context=True):
    """Return detached h[T,D], z0[T,V], full trace and actual guard metadata.

    Production uses the unchanged guarded forward and reconstructs h from its
    complete tape. Independent mode dynamically selects from full matrices and
    has no guard metadata; supplemental guard validation belongs to the caller.
    """
    helper.validate_weights(weights)
    helper._tokens(tokens, weights["EMBED"].shape[0])
    if type(independent) is not bool or type(context) is not bool:
        raise ValueError("literal independent/context booleans required")
    if independent and base is not None:
        raise ValueError("independent features must not consume a saved base")
    if independent:
        features, base = _dense_features(weights, tokens, context)
        counts = None
    else:
        base = guarded.forward(weights, tokens, context=context) if base is None else base
        if base.get("use_context") is not context:
            raise ValueError("feature base context differs")
        features = _from_tape(weights, tokens, base)
        diagnostics = base.get("guard_diagnostics")
        if (not isinstance(diagnostics, dict) or set(diagnostics) != {"schema", *GUARD_KEYS}
                or diagnostics["schema"] != guarded.SCHEMA
                or any(type(diagnostics[key]) is not bool for key in GUARD_KEYS[:2])
                or any(type(diagnostics[key]) is not int or diagnostics[key] < 0 for key in GUARD_KEYS[2:])):
            raise ValueError("producer features require actual guarded-forward diagnostics")
        if (diagnostics[GUARD_KEYS[0]] == diagnostics[GUARD_KEYS[1]]
                or diagnostics[GUARD_KEYS[2]] + diagnostics[GUARD_KEYS[3]] != base["geometry"]["N"] * len(tokens)
                or diagnostics[GUARD_KEYS[4]] + diagnostics[GUARD_KEYS[5]] != len(tokens)
                or (diagnostics[GUARD_KEYS[1]] and (diagnostics[GUARD_KEYS[2]] or diagnostics[GUARD_KEYS[4]]))):
            raise ValueError("inconsistent actual guard operation counts")
        counts = torch.tensor([[int(diagnostics[key]) for key in GUARD_KEYS]], dtype=torch.int64)
    return {"schema": SCHEMA, "h": features, "logits": base["logits"].clone(),
            "base": base, "trace": helper.trace(base), "guard_counts": counts,
            "independent": independent, "features_differentiated": False}


def _array(value, label, *, dtype=np.float64):
    if (not isinstance(value, np.ndarray) or value.dtype != dtype or not value.size
            or not np.isfinite(value).all()):
        raise ValueError(label + " requires a nonempty finite NumPy " + str(dtype) + " array")
    return value


def validate_data(h, z0, targets):
    _array(h, "features")
    _array(z0, "baseline logits")
    _array(targets, "targets", dtype=np.int64)
    if (h.ndim != 2 or z0.ndim != 2 or targets.ndim != 1
            or h.shape[0] != z0.shape[0] or targets.shape != (h.shape[0],)
            or z0.shape[1] < 2 or np.any((targets < 0) | (targets >= z0.shape[1]))):
        raise ValueError("complete aligned feature/logit/target geometry required")
    return h.shape[0], h.shape[1], z0.shape[1]


def parameter_count(d, vocab, arm):
    _positive(d, "feature width")
    _positive(vocab, "vocabulary")
    if arm not in ARMS:
        raise ValueError("allowlisted residual-readout arm required")
    return vocab if arm == "bias" else vocab * d + vocab


def unpack(theta, d, vocab, arm):
    _array(theta, "readout coordinates")
    if theta.shape != (parameter_count(d, vocab, arm),):
        raise ValueError("complete flat readout coordinates required")
    if arm == "bias":
        return None, theta
    return theta[:vocab * d].reshape(vocab, d), theta[vocab * d:]


def theta_size(h, z0, arm):
    _array(h, "features")
    _array(z0, "baseline logits")
    if h.ndim != 2 or z0.ndim != 2 or h.shape[0] != z0.shape[0] or z0.shape[1] < 2:
        raise ValueError("aligned complete feature/logit geometry required")
    return parameter_count(h.shape[1], z0.shape[1], arm)


def _logits(theta, h, z0, weights, bias):
    # Preserve the actual frozen baseline bytes at the prescribed zero start,
    # including signed zero; adding an all-zero residual need not preserve it.
    if not np.any(theta):
        return z0.copy()
    result = z0 + bias if weights is None else z0 + h @ weights.T + bias
    if not np.isfinite(result).all():
        raise ValueError("nonfinite residual logits")
    return result


def predict(theta, h, z0, arm, *, chunk_size=CHUNK_SIZE):
    """Fresh residual logits in row order; the zero start is byte-identical."""
    theta_size(h, z0, arm)
    _positive(chunk_size, "chunk size")
    weights, bias = unpack(theta, h.shape[1], z0.shape[1], arm)
    result = np.empty_like(z0)
    for start in range(0, len(h), chunk_size):
        end = min(start + chunk_size, len(h))
        result[start:end] = _logits(theta, h[start:end], z0[start:end], weights, bias)
    return result


def _statistics(ce, theta, gradient, correct, positions):
    penalty = REGULARIZATION / 2 * float(np.dot(theta, theta))
    gradient_norm_sq = float(np.dot(gradient, gradient))
    estimate = gradient_norm_sq / (2 * REGULARIZATION)
    values = (ce, penalty, ce + penalty, gradient_norm_sq, estimate)
    if not all(math.isfinite(value) for value in values) or not np.isfinite(gradient).all():
        raise ValueError("nonfinite residual objective, gradient, or gap estimate")
    return {"objective": ce + penalty, "ce": ce, "penalty": penalty,
            "gradient": gradient, "gradient_norm_sq": gradient_norm_sq,
            "numerical_gap_estimate": estimate, "numerical_target_met": estimate <= GAP_TARGET,
            "accuracy": correct / positions, "positions": positions,
            "regularization": REGULARIZATION, "formal_certification": False}


def objective(theta, h, z0, targets, arm, *, chunk_size=CHUNK_SIZE, checkpoint=None):
    """Chunked NumPy exact-softmax CE and analytic residual-only gradient."""
    positions, d, vocab = validate_data(h, z0, targets)
    _positive(chunk_size, "chunk size")
    weights, bias = unpack(theta, d, vocab, arm)
    gw = None if weights is None else np.zeros_like(weights)
    gb = np.zeros_like(bias)
    ce, correct = 0.0, 0
    for start in range(0, positions, chunk_size):
        if checkpoint is not None:
            checkpoint()
        end = min(start + chunk_size, positions)
        logits = _logits(theta, h[start:end], z0[start:end], weights, bias)
        shifted = logits - logits.max(axis=1, keepdims=True)
        energies = np.exp(shifted)
        totals = energies.sum(axis=1)
        rows = np.arange(end - start)
        ce += float((np.log(totals) - shifted[rows, targets[start:end]]).sum()) / positions
        correct += int((logits.argmax(axis=1) == targets[start:end]).sum())
        cotangent = energies / totals[:, None]
        cotangent[rows, targets[start:end]] -= 1
        cotangent /= positions
        gb += cotangent.sum(axis=0)
        if gw is not None:
            gw += cotangent.T @ h[start:end]
        if checkpoint is not None:
            checkpoint()
    gradient = gb if gw is None else np.concatenate((gw.ravel(), gb))
    gradient += REGULARIZATION * theta
    return _statistics(ce, theta, gradient, correct, positions)


def independent_objective(theta, h, z0, targets, arm, *, chunk_size=CHUNK_SIZE, checkpoint=None):
    """Independent Torch softmax/autograd; no model or feature derivatives."""
    positions, d, vocab = validate_data(h, z0, targets)
    _positive(chunk_size, "chunk size")
    unpack(theta, d, vocab, arm)
    leaves = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    gradient = torch.zeros_like(leaves)
    ce, correct = 0.0, 0
    with torch.enable_grad():
        for start in range(0, positions, chunk_size):
            if checkpoint is not None:
                checkpoint()
            end = min(start + chunk_size, positions)
            logits = torch.tensor(z0[start:end])
            if arm == "affine":
                logits = logits + torch.tensor(h[start:end]) @ leaves[:vocab * d].reshape(vocab, d).T
            logits = logits + (leaves if arm == "bias" else leaves[vocab * d:])
            ids = torch.tensor(targets[start:end], dtype=torch.int64)
            value = torch.nn.functional.cross_entropy(logits, ids, reduction="sum") / positions
            gradient += torch.autograd.grad(value, leaves)[0]
            ce += float(value.detach())
            correct += int((logits.argmax(dim=1) == ids).sum())
            if checkpoint is not None:
                checkpoint()
        penalty = REGULARIZATION / 2 * leaves.square().sum()
        gradient += torch.autograd.grad(penalty, leaves)[0]
    return _statistics(ce, theta, gradient.detach().numpy(), correct, positions)


class _Stop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _digest(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def solve(h, z0, targets, arm, *, chunk_size=CHUNK_SIZE, checkpoint=None):
    """One zero-start, fixed-budget L-BFGS-B solve; no selection or restart.

    Only evaluated accepted endpoints can be returned. The final full analytic
    gradient is reused from its counted evaluation, never obtained through an
    uncounted post-cap query. Scalar trial hashes are provenance, not an audit
    of the L-BFGS path. The caller independently checks initial/final endpoints.
    A stopped solve above the numerical target is explicitly inconclusive.

    The per-solve clocks are checked before/after every data chunk. A caller's
    global watchdog must cover an in-flight BLAS/SciPy operation as well.
    """
    started_utc, started_mono = time.time(), time.monotonic()
    from scipy.optimize import minimize

    positions, d, vocab = validate_data(h, z0, targets)
    _positive(chunk_size, "chunk size")
    size = parameter_count(d, vocab, arm)
    theta0 = np.zeros(size, dtype=np.float64)
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
            raise RuntimeError("invalid readout solve clock")
        return elapsed_utc, elapsed_mono

    def check():
        if checkpoint is not None:
            checkpoint()
        if max(clocks()) >= DEADLINE_SECONDS:
            raise _Stop("deadline")

    def row(point, value):
        utc, mono = clocks()
        return {key: value[key] for key in ("objective", "ce", "penalty", "gradient_norm_sq",
                                           "numerical_gap_estimate", "numerical_target_met", "accuracy")} | {
            "function_call": calls, "theta_sha256": _digest(point),
            "gradient_sha256": _digest(value["gradient"]),
            "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": mono}

    def fg(point):
        nonlocal calls, initial, accepted, last
        check()
        if calls >= MAX_FUNCTION_CALLS:
            raise _Stop("function_call_limit")
        calls += 1
        value = objective(point, h, z0, targets, arm, chunk_size=chunk_size, checkpoint=check)
        point = point.copy()
        last = (point, value)
        history.append(row(point, value))
        if initial is None:
            if point.tobytes() != theta0.tobytes():
                raise RuntimeError("L-BFGS-B did not evaluate the prescribed zero start")
            initial = accepted = last
            if value["numerical_target_met"]:
                raise _Stop("numerical_gap_target")
        return value["objective"], value["gradient"].copy()

    def callback(point):
        nonlocal accepted, iterations
        if last is None or point.tobytes() != last[0].tobytes():
            raise RuntimeError("accepted iterate lacks its actual counted objective/gradient")
        accepted = last
        iterations += 1
        iteration_history.append({**row(*accepted), "iteration": iterations})
        check()
        if accepted[1]["numerical_target_met"]:
            raise _Stop("numerical_gap_target")
        if iterations >= MAX_ITERATIONS:
            raise _Stop("iteration_limit")

    options = {"maxiter": MAX_ITERATIONS, "maxfun": MAX_FUNCTION_CALLS, "maxls": 20,
               "maxcor": 10, "ftol": 0.0,
               "gtol": math.sqrt(2 * REGULARIZATION * GAP_TARGET / size)}
    try:
        result = minimize(fg, theta0, method="L-BFGS-B", jac=True, bounds=None,
                          callback=callback, options=options)
        scipy_status, scipy_message, scipy_success = int(result.status), str(result.message), bool(result.success)
        if accepted is None or result.x.tobytes() != accepted[0].tobytes():
            raise RuntimeError("returned L-BFGS-B point differs from the last evaluated accepted iterate")
        if accepted[1]["numerical_target_met"]:
            reason = "numerical_gap_target"
        elif calls >= MAX_FUNCTION_CALLS:
            reason = "function_call_limit"
        elif iterations >= MAX_ITERATIONS:
            reason = "iteration_limit"
        else:
            reason = "scipy_stopped_above_numerical_target"
    except _Stop as stopped:
        reason = stopped.reason
    if initial is None or accepted is None:
        raise RuntimeError("readout solve stopped before a complete initial objective/gradient")
    point, value = accepted
    utc, mono = clocks()
    if max(utc, mono) >= DEADLINE_SECONDS:
        reason = "deadline"
    weights, bias = unpack(point, d, vocab, arm)
    target = bool(value["numerical_target_met"])
    completed = reason == "numerical_gap_target" and target
    summary = {"schema": SCHEMA, "arm": arm, "positions": positions, "features": d, "vocabulary": vocab,
               "parameter_count": size, "regularization": REGULARIZATION, "options": options,
               "function_calls": calls, "complete_function_evaluations": len(history),
               "iterations": iterations, "stopping_reason": reason, "scipy_status": scipy_status,
               "scipy_message": scipy_message, "scipy_success": scipy_success,
               "numerical_target_met": target, "numerically_converged": completed,
               "inconclusive": not completed, "formal_certification": False,
               "rounding_error_bound": None, "gap_target": GAP_TARGET,
               "numerical_gap_estimate": value["numerical_gap_estimate"],
               "objective": value["objective"], "ce": value["ce"], "penalty": value["penalty"],
               "gradient_norm_sq": value["gradient_norm_sq"], "accuracy": value["accuracy"],
               "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": mono,
               "deadline_seconds": DEADLINE_SECONDS, "zero_start": True,
               "returned_point": "last evaluated accepted iterate", "restarts": 0,
               "trial_history_scope": "scalar provenance only; independent audit is endpoint-only"}
    return {"theta": point.copy(), "W": None if weights is None else weights.copy(), "b": bias.copy(),
            "gradient": value["gradient"].copy(), "summary": summary,
            "history": history, "iteration_history": iteration_history,
            "initial": {"theta": initial[0].copy(), "gradient": initial[1]["gradient"].copy(),
                        "statistics": {key: item for key, item in initial[1].items() if key != "gradient"}}}
