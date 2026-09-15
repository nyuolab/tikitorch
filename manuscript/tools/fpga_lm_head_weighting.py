"""Matched nominal head-mass diagnostics on a common fractional FP64 carrier.

No corpus access, optimizer state, native execution, or experiment lives here.
The supplied backward is a real-centering, selected-route STE: it is not the
derivative of the discontinuous floor forward or of normalized LUT NLL.
Equal mean head mass does not imply equal parameter-gradient/update norms.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from manuscript.tools import fpga_lm_trajectory_reference as extension
from manuscript.tools import fpga_lm_trajectory_arithmetic as arithmetic

reference, numeric = extension.reference, extension.numeric
WEIGHTS = extension.WEIGHTS
ARMS = ("relative_mass", "uniform_mass")
SCHEMA = "tikitorch.fpga_lm_head_weighting_helper.v1"
ATOL, RTOL = 1e-9, 1e-11


def source_identity():
    path = Path(__file__).resolve()
    return {**extension.source_identity(), **arithmetic.source_identity(),
            path.relative_to(path.parents[2]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_weights(weights):
    arithmetic._mapping(weights, "fractional weights")
    if weights["EMBED"].shape[1] & (weights["EMBED"].shape[1] - 1):
        raise ValueError("floor-forward width must be a power of two")
    extension.parameter_digest(weights)


def near(actual, expected, label):
    """Fixed FP64 reverse/head tolerance; integer tapes remain exact."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label + " complete mapping differs")
        for key in expected:
            near(actual[key], expected[key], label + "." + key)
        return
    if (not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor)
            or actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.device.type != "cpu" or expected.device.type != "cpu"
            or actual.dtype not in (torch.float64, torch.int64)
            or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()):
        raise ValueError(label + " complete finite tensor contract differs")
    same = (torch.allclose(actual, expected, atol=ATOL, rtol=RTOL) if expected.dtype == torch.float64
            else torch.equal(actual, expected))
    if not same:
        raise ValueError(label + " numerical replay differs")


def _bounded(value, label):
    if not bool(torch.isfinite(value).all()) or bool(((value < -32768) | (value > 32767)).any()):
        raise ValueError(label + " would clip or become nonfinite")
    return value


@torch.no_grad()
def _independent_forward(weights, tokens, use_context):
    """Separate full-matrix selector, not the production chunked selector.

    Keep the same FP64 addition/mean/floor order; a changed reduction order at
    a floor boundary would define a different discontinuous forward. Forward
    selection is independent of the saved tape. Reverse uses a separate
    fixed-route autograd implementation below, not this no-grad evaluator.
    """
    geometry = reference._geometry(weights, tokens)
    d, n, k, v, length = (geometry[key] for key in ("D", "N", "K", "V", "T"))
    history = [[] for _ in range(n)]
    tap_rows, projection_rows, offset_rows = ([[] for _ in range(n)] for _ in range(3))
    head_rows, outputs = [], []
    for position in range(length):
        hidden = weights["EMBED"][int(tokens[position])]
        for layer in range(n):
            mean = hidden.mean()
            floor = mean.floor()
            centered = _bounded(hidden - floor, "independent centering")
            history[layer].append(centered)
            offset_rows[layer].append(mean - floor)
            dilation = 1 << min(layer, 6)
            count = min(k, position // dilation + 1) if use_context else 1
            temporal = torch.stack([history[layer][position - tap * dilation]
                                    + weights["MH"][layer, :, tap] for tap in range(count)])
            conv, tap = temporal.max(dim=0) if layer % 2 == 0 else temporal.min(dim=0)
            _bounded(conv, "independent temporal selection")
            projected, projection = (weights["MP"][layer] + conv.unsqueeze(0)).max(dim=1)
            projected = _bounded(projected + weights["MB"][layer], "independent projection plus bias")
            hidden = _bounded(hidden + projected, "independent residual")
            tap_rows[layer].append(tap)
            projection_rows[layer].append(projection)
        logits, head = (weights["W_HEAD"] + hidden.unsqueeze(0)).max(dim=1)
        outputs.append(_bounded(logits, "independent head"))
        head_rows.append(head)
    return {"schema": extension.SCHEMA, "mode": extension.MODE, "geometry": geometry,
            "tokens": tokens.detach().clone(), "use_context": use_context,
            "logits": torch.stack(outputs),
            "routes": {"AD_HEAD": torch.stack(head_rows),
                       "PROJ_IDX": torch.stack([torch.stack(rows) for rows in projection_rows]),
                       "TAP_IDX": torch.stack([torch.stack(rows) for rows in tap_rows])},
            "centering_offsets": torch.stack([torch.stack(rows) for rows in offset_rows]),
            "parameter_sha256": extension.parameter_digest(weights), "autograd_forward": False,
            "native_execution": False, "saturation_events": 0}


def forward(weights, tokens, use_context=True, independent=False):
    validate_weights(weights)
    if type(use_context) is not bool or type(independent) is not bool:
        raise ValueError("literal context and independent booleans required")
    extension._geometry(weights, tokens)
    return (_independent_forward(weights, tokens, use_context) if independent
            else extension.forward(weights, tokens, use_context=use_context))


def trace(result):
    extension.reference_base(result)
    return {"logits": result["logits"], "centering_offsets": result["centering_offsets"], **result["routes"]}


def head(logits, targets, c, arm, independent=False):
    """Detached supplied cotangent; no gradient through probabilities/masses.

    c is a frozen Python float calibrated externally from saved initial masses.
    relative_weights always records m/mean(m); row_weights records the actually
    applied relative or uniform weighting. There is no division by sequence T.
    """
    numeric._tensor(logits, "logits", 2)
    _bounded(logits, "head input logits")
    if type(c) is not float or not math.isfinite(c) or not 0 < c <= 1:
        raise ValueError("c must be a literal finite float in (0,1]")
    if type(arm) is not str or arm not in ARMS or type(independent) is not bool:
        raise ValueError("declared literal arm and independent boolean required")
    ids = numeric._integer_array(targets, "targets")
    length, vocab = logits.shape
    if ids.shape != (length,) or np.any(ids < 0) or np.any(ids >= vocab):
        raise ValueError("one in-vocabulary target per logit row required")
    if independent:
        values = logits.detach().numpy()
        indices = np.minimum(np.floor((values.max(axis=1, keepdims=True) - values) / 4), 255).astype(np.int64)
        energies = np.asarray(numeric.ROM, dtype=np.int64)[indices]
        energy_sum = energies.sum(axis=1)
        shifts = np.array([max(0, int(total).bit_length() - 17) for total in energy_sum], dtype=np.int64)
        # Equality at Q16 is permitted: correct the bit-length candidate using
        # literal integer comparisons, independently of the producer's loop.
        shifts += (np.right_shift(energy_sum, shifts) > numeric.Q16).astype(np.int64)
        rounded = np.right_shift(energies, shifts[:, None])
        masses = rounded.sum(axis=1)
        if np.any(masses <= 0) or np.any(masses > numeric.Q16):
            raise ValueError("invalid independent truncated probability mass")
        q = rounded.astype(np.float64) / masses[:, None]
        factors = masses.astype(np.float64) / numeric.Q16
        relative = factors / factors.mean()
        applied = relative if arm == "relative_mass" else np.ones_like(relative)
        raw_head = rounded.astype(np.float64)
        raw_head[np.arange(length), ids] -= masses
        onehot = np.zeros_like(q)
        onehot[np.arange(length), ids] = 1
        cotangent = ((c / factors.mean()) * raw_head if arm == "relative_mass"
                     else (c * numeric.Q16) * (q - onehot))
        return {key: torch.from_numpy(value.copy()) for key, value in {
            "cotangent": cotangent, "rounded_energies": rounded, "rounded_totals": masses,
            "mass_factor": factors, "relative_weights": relative, "row_weights": applied,
            "truncated_probabilities": q}.items()}
    row = numeric.loss_head(logits.detach(), ids, "variable_mass", reduction="sum")
    factors = row["mass_factor"]
    relative = factors / factors.mean()
    applied = relative if arm == "relative_mass" else torch.ones_like(relative)
    q = row["truncated_probabilities"]
    onehot = torch.nn.functional.one_hot(torch.from_numpy(ids), vocab).to(torch.float64)
    cotangent = ((c / float(factors.mean())) * row["native_gradient"].to(torch.float64)
                 if arm == "relative_mass" else (c * numeric.Q16) * (q - onehot))
    result = {"cotangent": cotangent, "rounded_energies": row["rounded_energies"],
              "rounded_totals": row["rounded_totals"], "mass_factor": factors,
              "relative_weights": relative, "row_weights": applied, "truncated_probabilities": q}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite supplied head")
    return {key: value.detach().clone() for key, value in result.items()}


def gradient(weights, tokens, base, supplied_head, independent=False):
    validate_weights(weights)
    if type(independent) is not bool:
        raise ValueError("independent must be a literal boolean")
    tape = extension._bound_result(weights, tokens, base)
    cotangent = supplied_head["cotangent"]
    numeric._tensor(cotangent, "supplied cotangent", 2)
    if cotangent.shape != base["logits"].shape or cotangent.requires_grad:
        raise ValueError("complete detached head cotangent required")
    if independent:
        leaves = {key: value.detach().clone().requires_grad_(True) for key, value in weights.items()}
        with torch.enable_grad():
            affine = reference.forward(leaves, tokens, mode="fixed_routes", base=tape)
            near(affine["logits"].detach(), base["logits"], "affine replay at the source point")
            values = torch.autograd.grad(affine["logits"], tuple(leaves.values()), grad_outputs=cotangent.detach())
        result = {key: value.detach() for key, value in zip(leaves, values)}
    else:
        result = reference.manual_vjp(weights, tokens, cotangent.detach(), tape)
    arithmetic._mapping(result, "full fractional gradient", weights)
    return result


def step(weights, full_gradient, k=18):
    """Transactional FP64 subtraction, not integer EF or exact shadow algebra."""
    validate_weights(weights)
    if type(k) is not int or k != 18:
        raise ValueError("this diagnostic fixes literal k18")
    arithmetic._mapping(full_gradient, "fractional gradient", weights)
    candidate = {key: weights[key] - full_gradient[key] / (1 << k) for key in WEIGHTS}
    validate_weights(candidate)
    return candidate
