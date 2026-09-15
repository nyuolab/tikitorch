"""CPU-only, one-factor arithmetic helpers; no model, dataset, or experiment runner.

Raw logit/weight units are retained throughout. The fixed shift-2 ROM approximates
exp(beta * logits), beta = log(2)/64. Supplied gradients are diagnostic vector
fields, NOT derivatives of the discontinuous LUT loss. No native code is loaded.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

SCHEMA = "tikitorch.fpga_lm_numeric_diagnostic.v1"
BETA = math.log(2.0) / 64.0
Q16 = 65536
ZSHIFT = 2
LOSS_VARIANTS = ("exact_ce", "lut_normalized", "truncated_normalized", "variable_mass")
CENTER_MODES = ("exact", "floor_ste", "legacy_floor")
ROM = tuple(round(Q16 * 2.0 ** (-index / 16.0)) for index in range(256))


def source_identity():
    path = Path(__file__).resolve()
    return {path.relative_to(path.parents[2]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}


def recipe():
    return {"schema": SCHEMA, "beta": BETA, "lut_shift": ZSHIFT, "q16": Q16,
            "loss_variants": list(LOSS_VARIANTS), "centering_modes": list(CENTER_MODES),
            "gradient_units": "derivative per raw logit unit; per-token gradients before reduction",
            "surrogate_scalar_value": "true continuous cross entropy; backward explicitly overridden",
            "rom_distribution": "E/S before truncation; r=E>>s, R=sum(r), q=r/R",
            "normalization_shift": "least nonnegative s for which floor(S/2**s)<=65536",
            "legacy_centering": "convert cotangent to declared native units before floor",
            "ef_rounding": "integer raw-weight update; no objective or gradient-unit rescaling",
            "training_performed": False, "native_execution": False, "cloud_operations": False}


def _tensor(value, name, ndim=None):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
            or value.device.type != "cpu" or value.numel() == 0
            or (ndim is not None and value.ndim != ndim) or not torch.isfinite(value).all()):
        raise ValueError(name + " must be a nonempty finite CPU float64 tensor")
    return value


def _integer_array(value, name):
    def reject_bool(item):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(name + " cannot contain booleans")
        if isinstance(item, (list, tuple)):
            for child in item:
                reject_bool(child)
    reject_bool(value)
    array = np.asarray(value)
    if array.size == 0 or array.dtype.kind not in "iu":
        raise ValueError(name + " must contain nonempty integer values")
    if np.any(array < -(1 << 63)) or np.any(array > (1 << 63) - 1):
        raise ValueError(name + " is outside signed int64")
    return array.astype(np.int64, copy=True)


class _SuppliedGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, value, gradient):
        ctx.save_for_backward(gradient)
        return value.clone()

    @staticmethod
    def backward(ctx, output_gradient):
        return output_gradient * ctx.saved_tensors[0], None, None


def loss_head(logits, targets, variant="exact_ce", *, reduction="sum"):
    """Return CE/LUT metrics and a scalar with the declared supplied backward.

    ``supplied_gradient`` is always per-token, unreduced. ``loss.backward()``
    applies the requested sum/mean reduction. For non-CE variants, ``loss`` has
    the true CE *value*, but its overridden backward is explicitly not the
    derivative of that value or of LUT NLL. The rounded q may assign zero mass
    to the target; actual E/S LUT NLL remains finite. Predictions use raw logits.
    """
    _tensor(logits, "logits", 2)
    if variant not in LOSS_VARIANTS or reduction not in ("sum", "mean"):
        raise ValueError("unknown head variant or reduction")
    ids = _integer_array(targets, "targets")
    rows, vocab = logits.shape
    if ids.shape != (rows,) or np.any(ids < 0) or np.any(ids >= vocab):
        raise ValueError("one in-vocabulary target is required per logit row")
    index = torch.from_numpy(ids)
    detached = logits.detach()
    gaps = detached.max(dim=1, keepdim=True).values - detached
    if not torch.isfinite(gaps).all():
        raise OverflowError("raw logit difference overflow")
    raw_indices = torch.floor(gaps / (1 << ZSHIFT))
    indices = raw_indices.clamp(max=255).to(torch.int64)
    energies = torch.tensor(ROM, dtype=torch.int64)[indices]
    # Integer totals/shifts are deliberately not replaced with log2 rounding.
    totals = energies.sum(dim=1)
    shifts = []
    for total in totals.tolist():
        if total <= 0:
            raise OverflowError("invalid ROM energy total")
        shift = 0
        while total >> shift > Q16:
            shift += 1
        shifts.append(shift)
    shifts = torch.tensor(shifts, dtype=torch.int64)
    rounded = torch.bitwise_right_shift(energies, shifts[:, None])
    masses = rounded.sum(dim=1)
    if torch.any(masses <= 0):
        raise ValueError("truncated distribution has zero total mass")
    p = energies.to(torch.float64) / totals[:, None]
    q = rounded.to(torch.float64) / masses[:, None]
    exact_log_p = torch.log_softmax(logits * BETA, dim=1)
    ce_tokens = -exact_log_p[torch.arange(rows), index]
    lut_tokens = torch.log(totals.to(torch.float64)) - torch.log(
        energies[torch.arange(rows), index].to(torch.float64))
    onehot = torch.nn.functional.one_hot(index, vocab).to(torch.float64)
    native = rounded.clone()
    native[torch.arange(rows), index] -= masses
    gradients = {
        "exact_ce": BETA * (exact_log_p.detach().exp() - onehot),
        "lut_normalized": BETA * (p - onehot),
        "truncated_normalized": BETA * (q - onehot),
        "variable_mass": (BETA / Q16) * native.to(torch.float64),
    }
    gradient = gradients[variant]
    divisor = rows if reduction == "mean" else 1
    true_ce = ce_tokens.sum() / divisor
    scalar = (true_ce if variant == "exact_ce" else _SuppliedGradient.apply(
        logits, true_ce.detach(), gradient / divisor))
    if not torch.isfinite(ce_tokens).all() or not torch.isfinite(lut_tokens).all():
        raise OverflowError("nonfinite head loss")
    return {"loss": scalar, "true_ce": true_ce, "lut_nll": lut_tokens.sum() / divisor,
            "true_ce_per_token": ce_tokens.detach(), "lut_nll_per_token": lut_tokens,
            "true_ce_mean": ce_tokens.detach().mean(), "lut_nll_mean": lut_tokens.mean(),
            "supplied_gradient": gradient, "native_gradient": native,
            "native_gradient_scale": BETA / (Q16 * divisor),
            "energies": energies, "energy_totals": totals, "normalization_shifts": shifts,
            "rounded_energies": rounded, "rounded_totals": masses,
            "lut_probabilities": p, "truncated_probabilities": q,
            "mass_factor": masses.to(torch.float64) / Q16,
            "rounded_zero_target_mask": rounded[torch.arange(rows), index] == 0,
            "target_at_last_bin_mask": raw_indices[torch.arange(rows), index] >= 255,
            "target_above_last_bin_mask": raw_indices[torch.arange(rows), index] > 255,
            "predictions": detached.argmax(dim=1), "variant": variant, "reduction": reduction,
            "scalar_semantics": "true_ce" if variant == "exact_ce" else "true_ce_value_supplied_backward"}


def center_vjp(gradient, mode="exact", *, gradient_scale=1.0):
    """Last-axis centering VJP; legacy floor is a nonlinear surrogate.

    For native variable-mass cotangents, use beta/65536 (or beta/(65536*T)
    after mean reduction). Floating arithmetic near a lattice boundary is not
    promised bitwise native parity; use legacy_center_vjp_raw for that check.
    """
    _tensor(gradient, "gradient")
    if gradient.ndim == 0 or mode not in CENTER_MODES:
        raise ValueError("centering requires a nonempty last axis and a known mode")
    if type(gradient_scale) not in (float, int) or not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be positive and finite, not bool")
    if mode == "legacy_floor":
        native_units = gradient / gradient_scale
        if not torch.isfinite(native_units).all():
            raise OverflowError("native cotangent conversion overflow")
        mean = torch.floor(native_units.mean(dim=-1, keepdim=True)) * gradient_scale
    else:
        mean = gradient.mean(dim=-1, keepdim=True)
    result = gradient - mean
    if not torch.isfinite(result).all():
        raise OverflowError("centering VJP overflow")
    return result


class _Center(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, mode, gradient_scale):
        ctx.mode, ctx.gradient_scale = mode, gradient_scale
        mean = value.mean(dim=-1, keepdim=True)
        return value - (mean if mode == "exact" else torch.floor(mean))

    @staticmethod
    def backward(ctx, gradient):
        return center_vjp(gradient, ctx.mode, gradient_scale=ctx.gradient_scale), None, None


def center(value, mode="exact", *, gradient_scale=1.0):
    """Exact center, floor-forward/exact-STE, or floor-forward/legacy backward."""
    _tensor(value, "value")
    # Validate configuration before constructing an autograd node.
    center_vjp(torch.zeros_like(value), mode, gradient_scale=gradient_scale)
    result = _Center.apply(value, mode, gradient_scale)
    if not torch.isfinite(result).all():
        raise OverflowError("centering forward overflow")
    return result


def legacy_center_vjp_raw(gradient):
    """Exact integer last-axis g-floor(sum(g)/D), with native int32 bounds."""
    array = _integer_array(gradient, "gradient")
    if array.ndim == 0 or np.any(array < -(1 << 31)) or np.any(array > (1 << 31) - 1):
        raise ValueError("raw centering requires int32 rows")
    if array.shape[-1] > (1 << 31):
        raise OverflowError("row too wide for checked int64 summation")
    prefixes = array.cumsum(axis=-1)
    if np.any(prefixes < -(1 << 31)) or np.any(prefixes > (1 << 31) - 1):
        raise OverflowError("native int32 centering ordered sum overflow")
    total = prefixes[..., -1:]
    result = array - total // array.shape[-1]
    if np.any(result < -(1 << 31)) or np.any(result > (1 << 31) - 1):
        raise OverflowError("native int32 centering output overflow")
    return result


def ef_rounding(weights, remainders, gradient, k):
    """Exact scalar/array raw-unit EF step; no input mutation or silent clipping.

    next_w-w = -G/2**k + (next_r-r)/2**k. The fractional shadow state
    w-r/2**k follows unrounded SGD exactly. This identity isolates optimizer
    quantization; it does not normalize variable head mass or change objective.
    """
    if type(k) is not int or not 0 <= k <= 30:
        raise ValueError("k must be a literal integer in0..30")
    w, r, g = (_integer_array(value, name) for value, name in
               ((weights, "weights"), (remainders, "remainders"), (gradient, "gradient")))
    if w.shape != r.shape or w.shape != g.shape:
        raise ValueError("weights/remainders/gradient require equal shapes; no broadcasting")
    denominator = 1 << k
    if np.any(w < -32768) or np.any(w > 32767):
        raise ValueError("weights are outside int16")
    if np.any(r <= -denominator) or np.any(r >= denominator):
        raise ValueError("noncanonical remainder")
    if np.any(g < -(1 << 31)) or np.any(g > (1 << 31) - 1):
        raise ValueError("gradient is outside int32")
    accumulator = r + g
    if np.any(accumulator < -(1 << 31)) or np.any(accumulator > (1 << 31) - 1):
        raise OverflowError("native int32 accumulator overflow")
    quotient = np.abs(accumulator) // denominator
    quotient = np.where(accumulator < 0, -quotient, quotient)
    next_w, next_r = w - quotient, accumulator - quotient * denominator
    if np.any(next_w < -32768) or np.any(next_w > 32767):
        raise OverflowError("candidate weight would clip")
    identity = (w - next_w) * denominator + next_r - r - g
    if np.any(identity != 0):
        raise AssertionError("internal EF conservation failure")
    return {"weights": next_w, "remainders": next_r, "quotient": quotient,
            "integer_identity_residual": identity,
            "unrounded_delta": -g.astype(np.float64) / denominator,
            "rounding_delta": (next_r - r).astype(np.float64) / denominator,
            "shadow_before": w - r.astype(np.float64) / denominator,
            "shadow_after": next_w - next_r.astype(np.float64) / denominator,
            "denominator": denominator, "k": k}


def nominal_raw_sgd_learning_rate(k):
    """Rate matching G_native/2**k to summed beta*G_native/65536 units.

    This is a unit conversion only, not a recommended learning rate. A mean
    objective needs an additional factor T; normalized-head mass changes remain
    separate. Rounding and legacy backward differences are not erased.
    """
    if type(k) is not int or not 0 <= k <= 30:
        raise ValueError("k must be a literal integer in0..30")
    return Q16 / (BETA * (1 << k))


def smoothing_stationarity(counts, alpha=1.0):
    """Closed-form expected CE gradients at an additive-count-smoothed fit.

    Positive alpha generally makes the fit nonstationary for the *unsmoothed*
    empirical objective. It is stationary for the smoothed target distribution.
    This says nothing about a model's ability to realize the fitted logits.
    """
    values = _integer_array(counts, "counts")
    if values.ndim not in (1, 2) or np.any(values < 0) or np.any(values > (1 << 31) - 1):
        raise ValueError("counts require one or more nonnegative int32 rows")
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive, not bool")
    totals = values.astype(np.float64).sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("empirical distribution needs positive observations in every row")
    empirical = values / totals
    smoothed = (values + float(alpha)) / (totals + values.shape[-1] * float(alpha))
    if not np.isfinite(smoothed).all() or not np.all(smoothed > 0):
        raise OverflowError("nonfinite smoothed distribution")
    gradient = BETA * (smoothed - empirical)
    return {"empirical_probabilities": empirical, "smoothed_probabilities": smoothed,
            "unsmoothed_expected_gradient": gradient,
            "smoothed_expected_gradient": np.zeros_like(gradient),
            "unsmoothed_gradient_l1": np.abs(gradient).sum(axis=-1), "alpha": alpha}


def duplicate_bias_step(first, second, target, learning_rate):
    """Scalar quadratic identity for two equally updated additive parameters.

    For L=(first+second-target)^2/2, both gradients equal the residual. Updating
    both at eta changes their sum exactly as updating one parameter at 2*eta.
    This is an algebraic control, not a diagnosis of the model's nonlinear loss.
    """
    for value in (first, second, target, learning_rate):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("scalar bias control requires finite real values, not bool")
    if learning_rate < 0:
        raise ValueError("learning rate cannot be negative")
    before = first + second
    residual = before - target
    after_first, after_second = first - learning_rate * residual, second - learning_rate * residual
    result = {"sum_before": before, "gradient_each": residual,
              "first_after": after_first, "second_after": after_second,
              "sum_after": after_first + after_second,
              "single_parameter_after": before - learning_rate * residual,
              "double_rate_single_after": before - 2 * learning_rate * residual,
              "loss_before": 0.5 * residual ** 2,
              "loss_after": 0.5 * (after_first + after_second - target) ** 2}
    if any(not math.isfinite(value) for value in result.values()):
        raise OverflowError("scalar bias control overflow")
    return result
