"""FP64 diagnostic reference for the integer LM architecture, not FPGA code.

All five parameter regions have their natural tensor shapes. ``smooth`` uses
real mean centering and first-index selected max/min derivatives.
``integer_anchor`` requires integer int16 weights, reproduces the nonclipped
integer forward, and uses the real centering VJP (an explicit straight-through
surrogate, NOT the derivative of floor). ``fixed_routes`` replays a supplied
base's selections and constant centering offsets: its logits are jointly affine
in every parameter. It never materializes projection candidate matrices.
``fixed_integer`` holds only the selections fixed and recomputes floor centering
on integer weights, separating rounding from route changes. Its autograd path
is an STE, not a derivative of its discontinuous forward.

The latter is an oracle for that affine surrogate, not a promise of descent for
the discrete forward or after dynamic routes change. No fitting, corpus access,
training loop, mutable global state, cloud operation, or artifact write lives
here. Source/environment binding belongs to the invoking diagnostic.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

SCHEMA = "tikitorch.fpga_lm_float_reference.v1"
WEIGHTS = ("EMBED", "MH", "MP", "MB", "W_HEAD")
BETA = math.log(2.0) / 64.0


def source_identity():
    path = Path(__file__).resolve()
    return {path.relative_to(path.parents[2]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}


def _finite_tensor(value, *, name, dtype=torch.float64, device=None):
    if not isinstance(value, torch.Tensor) or value.dtype != dtype:
        raise ValueError(name + " must be an exact " + str(dtype) + " tensor")
    if device is not None and value.device != device:
        raise ValueError(name + " must share the parameter device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")


def _geometry(weights, tokens):
    if not isinstance(weights, dict) or set(weights) != set(WEIGHTS):
        raise ValueError("all and only the five parameter regions are required")
    for name in WEIGHTS:
        _finite_tensor(weights[name], name=name)
    embed, mh = weights["EMBED"], weights["MH"]
    if embed.ndim != 2 or mh.ndim != 3:
        raise ValueError("EMBED[V,D] and MH[N,D,K] determine the geometry")
    vocab, width = embed.shape
    layers, mh_width, taps = mh.shape
    if min(vocab, width, layers, taps) < 1 or mh_width != width:
        raise ValueError("positive consistent vocabulary/width/layer/tap sizes required")
    shapes = {"EMBED": (vocab, width), "MH": (layers, width, taps),
              "MP": (layers, width, width), "MB": (layers, width), "W_HEAD": (vocab, width)}
    for name, shape in shapes.items():
        if weights[name].shape != shape or weights[name].device != embed.device:
            raise ValueError("inconsistent full parameter geometry or device: " + name)
    _finite_tensor(tokens, name="tokens", dtype=torch.int64, device=embed.device)
    if tokens.ndim != 1 or not tokens.numel() or bool(((tokens < 0) | (tokens >= vocab)).any()):
        raise ValueError("one nonempty complete in-vocabulary token window required")
    return {"D": width, "V": vocab, "N": layers, "K": taps, "T": tokens.numel()}


def _geometry_equal(actual, expected):
    return (isinstance(actual, dict) and set(actual) == set(expected)
            and all(type(actual[name]) is int and actual[name] == value for name, value in expected.items()))


def _base(base, geometry, tokens, use_context):
    if not isinstance(base, dict) or base.get("schema") != SCHEMA:
        raise ValueError("fixed replay requires a base forward from this reference")
    if base.get("mode") not in ("smooth", "integer_anchor", "fixed_routes", "fixed_integer"):
        raise ValueError("base must declare its originating reference mode")
    if not _geometry_equal(base.get("geometry"), geometry):
        raise ValueError("base geometry differs")
    _finite_tensor(base.get("tokens"), name="base tokens", dtype=torch.int64, device=tokens.device)
    if not torch.equal(base["tokens"], tokens):
        raise ValueError("base must bind the identical token window")
    if type(base.get("use_context")) is not bool or base["use_context"] != use_context:
        raise ValueError("base context policy differs")
    d, n, v, t = (geometry[key] for key in ("D", "N", "V", "T"))
    routes = base.get("routes")
    shapes = {"AD_HEAD": (t, v), "PROJ_IDX": (n, t, d), "TAP_IDX": (n, t, d)}
    if not isinstance(routes, dict) or set(routes) != set(shapes):
        raise ValueError("complete head/projection/tap routes required")
    for name, shape in shapes.items():
        _finite_tensor(routes[name], name=name, dtype=torch.int64, device=tokens.device)
        upper = geometry["K"] if name == "TAP_IDX" else d
        if routes[name].shape != shape or bool(((routes[name] < 0) | (routes[name] >= upper)).any()):
            raise ValueError("invalid full route geometry or index: " + name)
    for layer in range(n):
        times = torch.arange(t, device=tokens.device)[:, None] - routes["TAP_IDX"][layer] * (1 << min(layer, 6))
        if bool((times < 0).any()) or (not use_context and bool(routes["TAP_IDX"][layer].any())):
            raise ValueError("saved route selects an excluded causal tap")
    offsets = base.get("centering_offsets")
    _finite_tensor(offsets, name="centering offsets", device=tokens.device)
    if offsets.shape != (n, t) or bool(((offsets < 0) | (offsets >= 1)).any()):
        raise ValueError("complete scalar centering offsets required")
    if base["mode"] == "smooth" and bool(offsets.any()):
        raise ValueError("smooth forward cannot supply integer rounding offsets")
    # Constant replay offsets cannot inherit gradients from a different graph.
    return {name: values.detach() for name, values in routes.items()}, offsets.detach()


def _check_values(values, anchor, name):
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite " + name)
    if anchor and bool(((values < -32768) | (values > 32767)).any()):
        raise ValueError("integer-anchor forward would clip at " + name)
    return values


def _select(candidates, maximum):
    """First-index tie selection with a selected-coordinate autograd path."""
    with torch.no_grad():
        index = candidates.argmax(dim=-1) if maximum else candidates.argmin(dim=-1)
        if candidates.shape[-1] == 1:
            margin = torch.full_like(candidates[..., 0], float("inf"))
        else:
            pair = torch.topk(candidates.detach(), 2, dim=-1, largest=maximum).values
            margin = pair[..., 0] - pair[..., 1] if maximum else pair[..., 1] - pair[..., 0]
    selected = candidates.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return selected, index, margin


def forward(weights, tokens, *, mode="smooth", base=None, chunk_size=64, use_context=None):
    """Return logits, selected routes, base offsets, margins and their scope.

    Routes have shapes AD_HEAD[T,V], PROJ_IDX/TAP_IDX[N,T,D]. Margins are
    winner-minus-runner-up (reversed for min). A lone causal candidate has +inf
    margin, identified separately by candidate_counts. Fixed-route replay does
    not recompute margins; returns ``margins=None`` rather than stale claims.
    ``fixed_integer`` freezes routes but recomputes floor means; its forward
    is not affine. Both integer modes reject would-clip states.
    ``chunk_size`` bounds dynamic projection/head output-row candidate batches.
    Integer anchoring rejects clipping; smooth/fixed modes do not quantize or
    clip and make no integer/hardware equivalence claim away from the base.
    """
    geometry = _geometry(weights, tokens)
    if mode not in ("smooth", "integer_anchor", "fixed_routes", "fixed_integer"):
        raise ValueError("undeclared reference mode")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive literal integer")
    if use_context is None:
        use_context = base.get("use_context") if mode in ("fixed_routes", "fixed_integer") and isinstance(base, dict) else True
    if type(use_context) is not bool:
        raise ValueError("use_context must be literal bool")
    anchor = mode in ("integer_anchor", "fixed_integer")
    fixed = mode in ("fixed_routes", "fixed_integer")
    if base is not None and not fixed:
        raise ValueError("a base is used only by fixed-route replay")
    if anchor:
        if geometry["D"] & (geometry["D"] - 1):
            raise ValueError("integer centering requires a power-of-two width")
        for name, value in weights.items():
            if bool(((value != value.round()) | (value < -32768) | (value > 32767)).any()):
                raise ValueError("integer-anchor weights must be exact int16 integers: " + name)
    routes, offsets = _base(base, geometry, tokens, use_context) if fixed else (None, None)
    d, n, k, v, t = (geometry[key] for key in ("D", "N", "K", "V", "T"))
    coordinates = torch.arange(d, device=tokens.device)
    history = [[] for _ in range(n)]
    tap_rows, proj_rows, offset_rows = ([[] for _ in range(n)] for _ in range(3))
    tap_margins, proj_margins, tap_counts = ([[] for _ in range(n)] for _ in range(3))
    logits, head_rows, head_margins = [], [], []
    for position in range(t):
        residual = weights["EMBED"][tokens[position]]
        for layer in range(n):
            mean = residual.mean()
            offset = offsets[layer, position] if mode == "fixed_routes" else ((mean - mean.floor()).detach() if anchor else mean.new_zeros(()))
            centered = _check_values(residual - mean + offset, anchor, "centering")
            history[layer].append(centered)
            offset_rows[layer].append(offset.detach())
            dilation = 1 << min(layer, 6)
            count = min(k, position // dilation + 1) if use_context else 1
            if fixed:
                tap = routes["TAP_IDX"][layer, position]
                past = torch.stack(history[layer])
                conv = past[position - tap * dilation, coordinates] + weights["MH"][layer, coordinates, tap]
            else:
                candidates = torch.stack([history[layer][position - index * dilation] + weights["MH"][layer, :, index]
                                          for index in range(count)], dim=-1)
                conv, tap, margin = _select(candidates, maximum=(layer % 2 == 0))
                tap_margins[layer].append(margin)
            conv = _check_values(conv, anchor, "temporal selection")
            if fixed:
                projection = routes["PROJ_IDX"][layer, position]
                projected = conv[projection] + weights["MP"][layer, coordinates, projection]
            else:
                selected_rows, indices, margins = [], [], []
                for start in range(0, d, chunk_size):
                    candidates = weights["MP"][layer, start:start + chunk_size] + conv[None, :]
                    selected, index, margin = _select(candidates, maximum=True)
                    selected_rows.append(selected)
                    indices.append(index)
                    margins.append(margin)
                projected, projection = torch.cat(selected_rows), torch.cat(indices)
                proj_margins[layer].append(torch.cat(margins))
            projected = _check_values(projected + weights["MB"][layer], anchor, "projection plus bias")
            residual = _check_values(residual + projected, anchor, "residual")
            tap_rows[layer].append(tap)
            proj_rows[layer].append(projection)
            tap_counts[layer].append(torch.full((d,), count, dtype=torch.int64, device=tokens.device))
        if fixed:
            head = routes["AD_HEAD"][position]
            scores = residual[head] + weights["W_HEAD"][torch.arange(v, device=tokens.device), head]
        else:
            selected_rows, indices, margins = [], [], []
            for start in range(0, v, chunk_size):
                selected, index, margin = _select(weights["W_HEAD"][start:start + chunk_size] + residual[None, :], True)
                selected_rows.append(selected)
                indices.append(index)
                margins.append(margin)
            scores, head = torch.cat(selected_rows), torch.cat(indices)
            head_margins.append(torch.cat(margins))
        logits.append(_check_values(scores, anchor, "head"))
        head_rows.append(head)
    packed_routes = {"AD_HEAD": torch.stack(head_rows).detach(),
        "PROJ_IDX": torch.stack([torch.stack(rows) for rows in proj_rows]).detach(),
        "TAP_IDX": torch.stack([torch.stack(rows) for rows in tap_rows]).detach()}
    margins = None if fixed else {"AD_HEAD": torch.stack(head_margins),
        "PROJ_IDX": torch.stack([torch.stack(rows) for rows in proj_margins]),
        "TAP_IDX": torch.stack([torch.stack(rows) for rows in tap_margins])}
    return {"schema": SCHEMA, "mode": mode, "geometry": geometry, "tokens": tokens.detach().clone(),
        "use_context": use_context, "logits": torch.stack(logits), "routes": packed_routes,
        "centering_offsets": torch.stack([torch.stack(rows) for rows in offset_rows]).detach(),
        "margins": margins, "margin_scope": "not recomputed for fixed-route replay" if fixed else "current dynamic forward",
        "candidate_counts": {"AD_HEAD": torch.full((t, v), d, dtype=torch.int64, device=tokens.device),
            "PROJ_IDX": torch.full((n, t, d), d, dtype=torch.int64, device=tokens.device),
            "TAP_IDX": torch.stack([torch.stack(rows) for rows in tap_counts])}}


def manual_vjp(weights, tokens, cotangent, base, *, center_vjp=None):
    """Independent, non-autograd fixed-route reverse through all five regions.

    The default is the exact mean-centering VJP. An optional pure callback on
    each complete D-vector is explicitly a different diagnostic conditioner;
    callers must bind its source and gradient units. No parameters, base arrays
    or input cotangents are mutated. Returned tensors carry no autograd graph.
    """
    geometry = _geometry(weights, tokens)
    routes, _ = _base(base, geometry, tokens, base.get("use_context") if isinstance(base, dict) else None)
    _finite_tensor(cotangent, name="logit cotangent", device=tokens.device)
    if cotangent.shape != (geometry["T"], geometry["V"]):
        raise ValueError("one complete T-by-V logit cotangent required")
    if center_vjp is not None and not callable(center_vjp):
        raise ValueError("centering VJP override must be a callable")
    d, n, t = (geometry[key] for key in ("D", "N", "T"))
    with torch.no_grad():
        result = {name: torch.zeros_like(value) for name, value in weights.items()}
        stream = weights["EMBED"].new_zeros((n, t, d))
        coordinates = torch.arange(d, device=tokens.device)
        classes = torch.arange(geometry["V"], device=tokens.device)
        for position in range(t - 1, -1, -1):
            head = routes["AD_HEAD"][position]
            g = weights["EMBED"].new_zeros(d)
            g.scatter_add_(0, head, cotangent[position])
            result["W_HEAD"][classes, head] += cotangent[position]
            for layer in range(n - 1, -1, -1):
                projection = routes["PROJ_IDX"][layer, position]
                result["MB"][layer] += g
                result["MP"][layer, coordinates, projection] += g
                conv = g.new_zeros(d)
                conv.scatter_add_(0, projection, g)
                tap = routes["TAP_IDX"][layer, position]
                result["MH"][layer, coordinates, tap] += conv
                destination = position - tap * (1 << min(layer, 6))
                stream[layer, destination, coordinates] += conv
                incoming = stream[layer, position]
                centered = incoming - incoming.mean() if center_vjp is None else center_vjp(incoming.clone())
                _finite_tensor(centered, name="centering VJP result", device=tokens.device)
                if centered.shape != (d,):
                    raise ValueError("centering VJP changed vector shape")
                g = g + centered
            result["EMBED"][tokens[position]] += g
        if any(not bool(torch.isfinite(value).all()) for value in result.values()):
            raise ValueError("nonfinite parameter VJP")
    return result


def loss(logits, targets, *, beta=BETA):
    """Mean true cross-entropy at the declared raw-logit temperature."""
    _finite_tensor(logits, name="logits")
    _finite_tensor(targets, name="targets", dtype=torch.int64, device=logits.device)
    if (logits.ndim != 2 or not logits.shape[0] or not logits.shape[1]
            or targets.shape != (logits.shape[0],)
            or bool(((targets < 0) | (targets >= logits.shape[1])).any())):
        raise ValueError("complete nonempty logits/target geometry required")
    if type(beta) not in (float, int) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and strictly positive")
    return torch.nn.functional.cross_entropy(logits * beta, targets, reduction="mean")
