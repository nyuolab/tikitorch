"""Exact paired EF/unrounded arithmetic; no forward model or experiment.

All mappings use the five natural parameter shapes and detached CPU FP64
tensors. Native weights, remainders and raw gradients are integers; unrounded
weights lie on the 2**-k lattice. The two trajectories may query different
gradients. A shadow tracks the EF trajectory's *realized* gradients, not those
of an autonomous floating trajectory, and no identity promises loss descent.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from manuscript.tools import fpga_lm_numeric_diagnostic as numeric

SCHEMA = "tikitorch.fpga_lm_trajectory_arithmetic.v1"
WEIGHTS = ("EMBED", "MH", "MP", "MB", "W_HEAD")
MAX_EXACT_INTEGER = (1 << 53) - 1


def source_identity():
    path = Path(__file__).resolve()
    return {**numeric.source_identity(),
            path.relative_to(path.parents[2]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}


def _denominator(k):
    if type(k) is not int or not 0 <= k <= 30:
        raise ValueError("k must be a literal integer in 0..30")
    return 1 << k


def _mapping(values, label, template=None):
    if not isinstance(values, dict) or set(values) != set(WEIGHTS):
        raise ValueError(label + " requires all and only five parameter regions")
    for name in WEIGHTS:
        value = values[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
                or value.device.type != "cpu" or value.layout != torch.strided
                or value.requires_grad or not value.numel() or not torch.isfinite(value).all()):
            raise ValueError(label + "." + name + " requires detached finite CPU FP64 values")
    embed, mh = values["EMBED"], values["MH"]
    if embed.ndim != 2 or mh.ndim != 3:
        raise ValueError(label + " requires EMBED[V,D] and MH[N,D,K]")
    v, d = embed.shape
    n, md, k = mh.shape
    shapes = {"EMBED": (v, d), "MH": (n, d, k), "MP": (n, d, d),
              "MB": (n, d), "W_HEAD": (v, d)}
    if md != d or any(values[name].shape != shape for name, shape in shapes.items()):
        raise ValueError(label + " has inconsistent parameter geometry")
    if template is not None and any(values[name].shape != template[name].shape for name in WEIGHTS):
        raise ValueError(label + " geometry differs; broadcasting is forbidden")
    return values


def _integers(values, lower, upper, label):
    for name, value in values.items():
        if bool(((value < lower) | (value > upper) | (value != value.round())).any()):
            raise ValueError(label + "." + name + " is not an in-range exact integer")


def _native(weights, remainders, denominator):
    _mapping(weights, "weights")
    _mapping(remainders, "remainders", weights)
    _integers(weights, -32768, 32767, "native weights")
    _integers(remainders, -denominator + 1, denominator - 1, "canonical remainders")


def _gradient(gradient, template, label="gradient"):
    _mapping(gradient, label, template)
    _integers(gradient, -(1 << 31), (1 << 31) - 1, label)


def _floating(weights, denominator, template=None):
    _mapping(weights, "unrounded weights", template)
    for name, value in weights.items():
        scaled = value * denominator
        if bool(((value < -32768) | (value > 32767) | (scaled != scaled.round())).any()):
            raise ValueError("unrounded weights." + name + " must be bounded exact dyadics")


def shadow(weights, remainders, k=18):
    """Return z=w-r/2**k; canonical remainders guarantee |z-w|<1.

    A shadow may lie just outside int16 even when visible weights are valid;
    it is an arithmetic state, not permission to execute a clipping forward.
    """
    denominator = _denominator(k)
    _native(weights, remainders, denominator)
    return {name: weights[name] - remainders[name] / denominator for name in WEIGHTS}


def _ef_step(weights, remainders, gradient, k):
    result_w, result_r = {}, {}
    for name in WEIGHTS:
        result = numeric.ef_rounding(weights[name].to(torch.int64).numpy(),
                                    remainders[name].to(torch.int64).numpy(),
                                    gradient[name].to(torch.int64).numpy(), k)
        result_w[name] = torch.from_numpy(result["weights"]).to(torch.float64)
        result_r[name] = torch.from_numpy(result["remainders"]).to(torch.float64)
    return {"weights": result_w, "remainders": result_r}


def ef_step(weights, remainders, gradient, k=18):
    """Transactional pure native step via the frozen integer EF oracle.

    Failures return no partial result and never mutate any caller tensor.
    """
    denominator = _denominator(k)
    _native(weights, remainders, denominator)
    _gradient(gradient, weights)
    return _ef_step(weights, remainders, gradient, k)


def _unrounded_step(weights, gradient, denominator):
    result = {name: weights[name] - gradient[name] / denominator for name in WEIGHTS}
    for name, value in result.items():
        if bool(((value < -32768) | (value > 32767)).any()):
            raise OverflowError("unrounded candidate would exceed int16 range: " + name)
    return result


def unrounded_step(weights, gradient, k=18):
    """Apply a raw integer gradient without rounding; not exact-CE SGD.

    The helper does not choose where gradients are evaluated. It enforces the
    dyadic lattice and numerical range, without clipping or an STE backward.
    """
    denominator = _denominator(k)
    _floating(weights, denominator)
    _gradient(gradient, weights)
    return _unrounded_step(weights, gradient, denominator)


def _unit_and_norm(value):
    scale = float(value.abs().max())
    if not scale:
        return None, 0.0
    if not math.isfinite(scale):
        raise OverflowError("comparison statistics overflow")
    scaled = value / scale
    scaled_norm = float(torch.linalg.vector_norm(scaled))
    norm = scale * scaled_norm
    if not math.isfinite(norm):
        raise OverflowError("comparison statistics overflow")
    return scaled / scaled_norm, norm


def _statistics(left, right):
    delta = left - right
    _, l2 = _unit_and_norm(delta)
    l1, maximum = float(delta.abs().sum()), float(delta.abs().max())
    unit_a, _ = _unit_and_norm(left)
    unit_b, _ = _unit_and_norm(right)
    if not all(math.isfinite(value) for value in (l1, l2, maximum)):
        raise OverflowError("comparison statistics overflow")
    # Scaling first avoids underflow for small nonzero values or overflow of
    # norm_a * norm_b; None means an actually zero vector, not a tiny norm.
    cosine = float(unit_a @ unit_b) if unit_a is not None and unit_b is not None else None
    if cosine is not None:
        if not math.isfinite(cosine):
            raise OverflowError("comparison cosine overflow")
        cosine = min(1.0, max(-1.0, cosine))
    return {"l1": l1, "l2": l2, "max_abs": maximum,
            "differing_words": int((left != right).sum()), "cosine": cosine}


def compare(left, right):
    """Exact differing-word counts and FP64 discrepancy/norm statistics."""
    _mapping(left, "left")
    _mapping(right, "right", left)
    regions = {name: _statistics(left[name].reshape(-1), right[name].reshape(-1)) for name in WEIGHTS}
    return {"global": _statistics(torch.cat([left[name].reshape(-1) for name in WEIGHTS]),
                                  torch.cat([right[name].reshape(-1) for name in WEIGHTS])),
            "regions": regions}


def _shadow_numerator(weights, remainders, name, denominator):
    return weights[name].to(torch.int64) * denominator - remainders[name].to(torch.int64)


def _assert_pair(weights, remainders, float_weights, cumulative, denominator, label):
    for name in WEIGHTS:
        residual = (_shadow_numerator(weights, remainders, name, denominator)
                    - (float_weights[name] * denominator).to(torch.int64)
                    + cumulative[name].to(torch.int64))
        if bool(residual.any()):
            raise ValueError(label + " paired-gradient identity failed in " + name)


def transition(weights, remainders, float_weights, native_gradient, float_gradient,
               cumulative_gradient_difference, k=18):
    """Advance paired trajectories and assert exact integer-numerator identities.

    The incoming states must obey z-f=-C/2**k, where C is the sum of previous
    (G_native-G_float), normally initialized to zero at a common full state.
    This function updates C using each trajectory's supplied raw gradient. It
    does NOT infer that both gradients came from the same weights or function.
    C is stored as exact FP64 integers but checked/added using signed int64.
    """
    denominator = _denominator(k)
    _native(weights, remainders, denominator)
    _floating(float_weights, denominator, weights)
    _gradient(native_gradient, weights, "native gradient")
    _gradient(float_gradient, weights, "float gradient")
    _mapping(cumulative_gradient_difference, "cumulative gradient difference", weights)
    _integers(cumulative_gradient_difference, -MAX_EXACT_INTEGER, MAX_EXACT_INTEGER,
              "cumulative gradient difference")
    _assert_pair(weights, remainders, float_weights, cumulative_gradient_difference,
                 denominator, "incoming")
    stepped = _ef_step(weights, remainders, native_gradient, k)
    next_float = _unrounded_step(float_weights, float_gradient, denominator)
    next_cumulative = {}
    for name in WEIGHTS:
        cumulative = (cumulative_gradient_difference[name].to(torch.int64)
                      + native_gradient[name].to(torch.int64) - float_gradient[name].to(torch.int64))
        if bool(((cumulative < -MAX_EXACT_INTEGER) | (cumulative > MAX_EXACT_INTEGER)).any()):
            raise OverflowError("cumulative gradient difference exceeds exact FP64 integers")
        next_cumulative[name] = cumulative.to(torch.float64)
        before = _shadow_numerator(weights, remainders, name, denominator)
        after = _shadow_numerator(stepped["weights"], stepped["remainders"], name, denominator)
        if bool((after - before + native_gradient[name].to(torch.int64)).any()):
            raise ValueError("EF shadow recurrence failed in " + name)
        float_before = (float_weights[name] * denominator).to(torch.int64)
        float_after = (next_float[name] * denominator).to(torch.int64)
        if bool((float_after - float_before + float_gradient[name].to(torch.int64)).any()):
            raise ValueError("unrounded recurrence failed in " + name)
    _assert_pair(stepped["weights"], stepped["remainders"], next_float, next_cumulative,
                 denominator, "outgoing")
    return {**stepped, "float_weights": next_float,
            "cumulative_gradient_difference": next_cumulative,
            "identities": {"exact_shadow_recurrence": True, "exact_unrounded_recurrence": True,
                           "exact_paired_gradient_identity": True,
                           "shadow_recurrence_max_abs": 0.0, "paired_identity_max_abs": 0.0}}
