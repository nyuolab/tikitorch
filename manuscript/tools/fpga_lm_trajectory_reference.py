"""CPU FP64 weights with floor-forward centering, for bounded diagnostics only.

This is neither native integer execution nor an autograd model. Dynamic first-
index max/min selections and floor centering match the frozen integer forward
at integer weights when no clipping occurs. Fractional weights are permitted;
all weight and selected-activation int16 range guards remain active.

The real-centering backward options are declared straight-through surrogates,
not derivatives of this discontinuous forward. ``reference_base`` provides an
explicit, origin-labeled affine routing/offset tape for the old reference.
No dataset, training loop, native process, or artifact writer lives here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from manuscript.tools import fpga_lm_float_reference as reference
from manuscript.tools import fpga_lm_numeric_diagnostic as numeric

SCHEMA = "tikitorch.fpga_lm_trajectory_reference.v1"
MODE = "continuous_floor"
VARIANTS = (*numeric.LOSS_VARIANTS, "legacy")
WEIGHTS = reference.WEIGHTS
GRADIENT_UNITS = "Q16/beta times unreduced normalized derivative per raw logit unit"


def source_identity():
    sources = {**reference.source_identity(), **numeric.source_identity()}
    path = Path(__file__).resolve()
    sources[path.relative_to(path.parents[2]).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def recipe():
    return {"schema": SCHEMA, "forward": MODE, "cpu_only": True,
            "dtype": "float64", "weights": "continuous in the int16 value range",
            "centering": "subtract floor(sum(residual)/D), D a power of two",
            "selection": "dynamic first-index causal max/min, max-plus projection and head",
            "gradient_variants": list(VARIANTS), "gradient_units": GRADIENT_UNITS,
            "legacy_backward": "checked int32 raw-head integer reverse before any global scaling",
            "other_backward": "real-centering fixed-route STE; not derivative of floor",
            "parameter_digest": "EMBED/MH/MP/MB/W_HEAD concatenated row-major little-endian float64",
            "native_execution": False, "training_performed": False}


def _tensor(value, name, *, dtype=torch.float64):
    if (not isinstance(value, torch.Tensor) or value.dtype != dtype
            or value.device.type != "cpu" or value.numel() == 0
            or not bool(torch.isfinite(value).all())):
        raise ValueError(name + " requires nonempty finite CPU " + str(dtype))


def _bounded(value, name):
    _tensor(value, name)
    if bool(((value < -32768) | (value > 32767)).any()):
        raise ValueError("continuous-floor forward would clip at " + name)
    return value


def _geometry(weights, tokens):
    geometry = reference._geometry(weights, tokens)
    if geometry["D"] & (geometry["D"] - 1):
        raise ValueError("continuous-floor diagnostic requires power-of-two D")
    _tensor(tokens, "tokens", dtype=torch.int64)
    for name in WEIGHTS:
        _bounded(weights[name], name)
    return geometry


def parameter_digest(weights):
    """Bind all five validated parameter tensors in an explicit portable format."""
    if not isinstance(weights, dict) or set(weights) != set(WEIGHTS):
        raise ValueError("all and only five parameter regions are required")
    digest = hashlib.sha256()
    for name in WEIGHTS:
        _bounded(weights[name], name)
        digest.update(weights[name].detach().numpy().astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _choose(candidates, maximum=True):
    if not bool(torch.isfinite(candidates).all()):
        raise ValueError("nonfinite selection candidates")
    indices = candidates.argmax(dim=-1) if maximum else candidates.argmin(dim=-1)
    return candidates.gather(-1, indices.unsqueeze(-1)).squeeze(-1), indices


@torch.no_grad()
def forward(weights, tokens, *, use_context=True, chunk_size=64):
    """Dynamic floor-forward for fractional weights; returns no autograd graph."""
    geometry = _geometry(weights, tokens)
    if type(use_context) is not bool or type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("literal context boolean and positive integer chunk size required")
    d, n, k, vocab, length = (geometry[name] for name in ("D", "N", "K", "V", "T"))
    history = [[] for _ in range(n)]
    taps, projections, offsets = ([[] for _ in range(n)] for _ in range(3))
    heads, logits = [], []
    for position in range(length):
        residual = weights["EMBED"][tokens[position]]
        for layer in range(n):
            mean = residual.mean()
            if not bool(torch.isfinite(mean)):
                raise ValueError("nonfinite centering mean")
            floor = torch.floor(mean)
            centered = _bounded(residual - floor, "centering")
            history[layer].append(centered)
            offsets[layer].append(mean - floor)
            dilation = 1 << min(layer, 6)
            count = min(k, position // dilation + 1) if use_context else 1
            candidates = torch.stack([history[layer][position - tap * dilation] + weights["MH"][layer, :, tap]
                                      for tap in range(count)], dim=-1)
            conv, tap = _choose(candidates, layer % 2 == 0)
            conv = _bounded(conv, "temporal selection")
            selected, indices = [], []
            for start in range(0, d, chunk_size):
                value, index = _choose(weights["MP"][layer, start:start + chunk_size] + conv[None, :])
                selected.append(value)
                indices.append(index)
            projected = _bounded(torch.cat(selected) + weights["MB"][layer], "projection plus bias")
            residual = _bounded(residual + projected, "residual")
            taps[layer].append(tap)
            projections[layer].append(torch.cat(indices))
        selected, indices = [], []
        for start in range(0, vocab, chunk_size):
            value, index = _choose(weights["W_HEAD"][start:start + chunk_size] + residual[None, :])
            selected.append(value)
            indices.append(index)
        logits.append(_bounded(torch.cat(selected), "head"))
        heads.append(torch.cat(indices))
    return {"schema": SCHEMA, "mode": MODE, "geometry": geometry,
            "tokens": tokens.detach().clone(), "use_context": use_context,
            "logits": torch.stack(logits),
            "routes": {"AD_HEAD": torch.stack(heads),
                       "PROJ_IDX": torch.stack([torch.stack(row) for row in projections]),
                       "TAP_IDX": torch.stack([torch.stack(row) for row in taps])},
            "centering_offsets": torch.stack([torch.stack(row) for row in offsets]),
            "parameter_sha256": parameter_digest(weights), "autograd_forward": False,
            "native_execution": False, "saturation_events": 0}


def reference_base(result):
    """Adapt a validated saved selection tape, not the original forward model.

    The resulting old-reference ``fixed_routes`` tape uses constant fractional
    centering offsets. Replay agrees at the source point up to FP64 addition
    roundoff; perturbations describe its affine surrogate, not dynamic floor.
    The origin is explicit and all tensor payloads are copied.
    """
    if (not isinstance(result, dict) or result.get("schema") != SCHEMA or result.get("mode") != MODE
            or result.get("native_execution") is not False or result.get("autograd_forward") is not False
            or type(result.get("saturation_events")) is not int or result["saturation_events"] != 0):
        raise ValueError("expected a no-clipping continuous-floor result")
    geometry = result.get("geometry")
    if (not isinstance(geometry, dict) or set(geometry) != {"D", "N", "K", "V", "T"}
            or any(type(value) is not int or value <= 0 for value in geometry.values())
            or geometry["D"] & (geometry["D"] - 1)):
        raise ValueError("invalid complete tape geometry")
    tokens = result.get("tokens")
    _tensor(tokens, "tape tokens", dtype=torch.int64)
    if (tokens.shape != (geometry["T"],) or bool(((tokens < 0) | (tokens >= geometry["V"])).any())):
        raise ValueError("tape token geometry differs")
    if type(result.get("use_context")) is not bool:
        raise ValueError("tape context must be literal bool")
    logits = result.get("logits")
    _bounded(logits, "tape logits")
    if logits.shape != (geometry["T"], geometry["V"]):
        raise ValueError("tape logit geometry differs")
    sha = result.get("parameter_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("tape must bind the source parameter bytes")
    adapted = {"schema": reference.SCHEMA, "mode": "fixed_routes", "geometry": dict(geometry),
               "tokens": tokens, "use_context": result["use_context"], "logits": logits,
               "routes": result.get("routes"), "centering_offsets": result.get("centering_offsets")}
    routes, offsets = reference._base(adapted, geometry, tokens, result["use_context"])
    for name, value in routes.items():
        _tensor(value, name, dtype=torch.int64)
    _tensor(offsets, "tape offsets")
    return {**adapted, "tokens": tokens.detach().clone(), "logits": logits.detach().clone(),
            "routes": {name: value.detach().clone() for name, value in routes.items()},
            "centering_offsets": offsets.detach().clone(),
            "tape_origin": {"schema": SCHEMA, "forward_mode": MODE, "parameter_sha256": sha},
            "tape_semantics": "fixed selections and offsets; affine surrogate only, not native integer forward"}


def _bound_result(weights, tokens, result):
    geometry = _geometry(weights, tokens)
    tape = reference_base(result)
    if (tape["geometry"] != geometry or not torch.equal(tape["tokens"], tokens)
            or result["parameter_sha256"] != parameter_digest(weights)):
        raise ValueError("saved forward tape differs from current parameters or tokens")
    return tape


def _checked(values, name):
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < -(1 << 31)) or np.any(values > (1 << 31) - 1):
        raise OverflowError("native int32 backward overflow: " + name)
    return values


def _scatter(destination, indices, values, name):
    """Check every ordered duplicate-index prefix before publishing any scatter."""
    indices, values = np.asarray(indices, dtype=np.int64), _checked(values, name)
    flat = destination.reshape(-1)
    if indices.ndim != 1 or values.shape != indices.shape or np.any(indices < 0) or np.any(indices >= flat.size):
        raise ValueError("invalid internal scatter geometry")
    order = np.argsort(indices, kind="stable")
    sorted_ids, sorted_values = indices[order], values[order]
    prefixes = sorted_values.cumsum()
    starts = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
    beginnings = np.maximum.accumulate(np.where(starts, np.arange(len(indices)), 0))
    group_offset = np.where(beginnings > 0, prefixes[np.maximum(beginnings - 1, 0)], 0)
    _checked(flat[sorted_ids] + prefixes - group_offset, name + " ordered prefix")
    np.add.at(flat, indices, values)


def raw_legacy_vjp(weights, tokens, raw_logit_gradient, result):
    """Checked native-rule reverse with exact integer head cotangents.

    Integer gradients and all accumulation prefixes are checked even though
    weights/logits may be fractional. Returns CPU float64 *integer* gradients;
    no beta/Q16 scaling is applied here. No forward is recomputed.
    """
    tape = _bound_result(weights, tokens, result)
    _tensor(raw_logit_gradient, "raw logit gradient")
    if raw_logit_gradient.shape != result["logits"].shape or not torch.equal(raw_logit_gradient, raw_logit_gradient.trunc()):
        raise ValueError("raw head requires a complete exact-integer T-by-V cotangent")
    if bool(((raw_logit_gradient < -(1 << 31)) | (raw_logit_gradient > (1 << 31) - 1)).any()):
        raise OverflowError("raw head exceeds int32")
    raw = raw_logit_gradient.detach().numpy().astype(np.int64)
    geometry = result["geometry"]
    d, n, t, vocab = (geometry[key] for key in ("D", "N", "T", "V"))
    route = {name: value.numpy() for name, value in tape["routes"].items()}
    gradients = {name: np.zeros(tuple(value.shape), dtype=np.int64) for name, value in weights.items()}
    stream = np.zeros((n, t, d), dtype=np.int64)
    coordinates, classes = np.arange(d), np.arange(vocab)
    for position in range(t - 1, -1, -1):
        head = route["AD_HEAD"][position]
        g = np.zeros(d, dtype=np.int64)
        _scatter(g, head, raw[position], "head scatter")
        _scatter(gradients["W_HEAD"], classes * d + head, raw[position], "head parameter")
        for layer in range(n - 1, -1, -1):
            projection = route["PROJ_IDX"][layer, position]
            gradients["MB"][layer] = _checked(gradients["MB"][layer] + g, "bias accumulation")
            _scatter(gradients["MP"][layer], coordinates * d + projection, g, "projection parameter")
            conv = np.zeros(d, dtype=np.int64)
            _scatter(conv, projection, g, "projection scatter")
            tap = route["TAP_IDX"][layer, position]
            _scatter(gradients["MH"][layer], coordinates * geometry["K"] + tap, conv, "temporal parameter")
            destination = position - tap * (1 << min(layer, 6))
            _scatter(stream[layer], destination * d + coordinates, conv, "temporal stream")
            centered = numeric.legacy_center_vjp_raw(stream[layer, position])
            g = _checked(g + centered, "residual backward")
        token = int(tokens[position])
        gradients["EMBED"][token] = _checked(gradients["EMBED"][token] + g, "embedding accumulation")
    return {name: torch.from_numpy(value.astype(np.float64)) for name, value in gradients.items()}


def gradient(weights, tokens, targets, result, *, variant="legacy"):
    """Head plus VJP in one common raw-equivalent scale, without an update.

    Legacy preserves raw integer units throughout backward. All other variants
    use real-centering STE and old manual fixed-route reverse. ``variable_mass``
    therefore isolates backward-centering arithmetic from legacy. Exact-CE is
    an STE for this floor-forward model, not its mathematical floor derivative.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown declared gradient variant")
    tape = _bound_result(weights, tokens, result)
    head = numeric.loss_head(result["logits"], targets, "variable_mass" if variant == "legacy" else variant,
                             reduction="sum")
    raw_head = head["native_gradient"].to(torch.float64)
    if variant == "legacy":
        gradients = raw_legacy_vjp(weights, tokens, raw_head, result)
    else:
        cotangent = raw_head if variant == "variable_mass" else head["supplied_gradient"] * (numeric.Q16 / numeric.BETA)
        gradients = reference.manual_vjp(weights, tokens, cotangent, tape)
    return {"gradients": gradients, "head": head, "variant": variant,
            "gradient_units": GRADIENT_UNITS, "native_integer_gradient": variant == "legacy",
            "backward_semantics": "raw integer surrogate" if variant == "legacy" else "real-centering STE"}
