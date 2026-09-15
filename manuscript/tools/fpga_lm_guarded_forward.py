"""Experimental exact guarded FP64 forward with unchanged dense fallback.

This helper is not used by any frozen experiment. It certifies winners for
matrices with exactly one nonconstant entry per row. An uncertified structure
uses the complete original forward; an uncertified operation uses the original
dense selector. It neither changes the reverse rule nor compresses parameters,
Adam state, or checkpoints, and makes no timing or hardware claim.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from manuscript.tools import fpga_lm_exact_fit as helper

SCHEMA = "tikitorch.fpga_lm_guarded_forward.v1"
OFF_WEIGHT = helper.BETA * -8192.0


def source_identity():
    path = Path(__file__).resolve()
    root = path.parents[2]
    result = dict(helper.source_identity())
    for name in ("manuscript/tools/fpga_lm_guarded_forward.py", "tests/test_fpga_lm_guarded_forward.py"):
        result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return result


def _support(matrix, off_weight):
    different = matrix != off_weight
    if not bool((different.sum(dim=-1) == 1).all()):
        return None
    return different.to(torch.int64).argmax(dim=-1)


def _max_operation(matrix, vector, support, off_weight):
    """Strict sufficient certificate; equality must take first-tie dense path.

    C + max(vector) includes even the designated column, hence can only be
    conservative. Each actual off-support candidate is at most this bound.
    All rows must certify before the whole operation uses the gathered scores.
    """
    rows = torch.arange(matrix.shape[0])
    selected = matrix[rows, support] + vector[support]
    upper, lower = off_weight + vector.max(), off_weight + vector.min()
    finite = bool(torch.isfinite(selected).all()) and bool(torch.isfinite(upper)) and bool(torch.isfinite(lower))
    if finite and bool((selected > upper).all()):
        return selected, support.clone(), True
    values, indices = [], []
    for start in range(0, matrix.shape[0], 64):
        value, index, _ = helper.reference._select(matrix[start:start + 64] + vector[None, :], True)
        values.append(value)
        indices.append(index)
    return torch.cat(values), torch.cat(indices), False


@torch.no_grad()
def forward(weights, tokens, context=True, *, off_weight=OFF_WEIGHT):
    """Return an exact-fit-compatible detached base plus guard diagnostics.

    Params remain full natural-shaped CPU FP64 arrays. Off-weight changes that
    break the structural family trigger an ordinary dense call immediately.
    Core, selected-support, and embedding values can change freely, provided
    the frozen helper's finite-value/geometry contract continues to hold.

    The returned logits, route tape and centering offsets have the same meaning
    as ``helper.forward`` and may be passed to its unchanged dense manual VJP.
    Margin arrays are not computed by this helper; no substitute margin claim
    is made by the sufficient winner certificate.
    """
    helper.validate_weights(weights)
    helper._tokens(tokens, weights["EMBED"].shape[0])
    if type(context) is not bool or type(off_weight) is not float or not math.isfinite(off_weight):
        raise ValueError("literal context bool and finite literal float off_weight required")
    geometry = helper.reference._geometry(weights, tokens)
    mp_support = _support(weights["MP"], off_weight)
    head_support = _support(weights["W_HEAD"], off_weight)
    diagnostics = {"schema": SCHEMA, "structure_certified": mp_support is not None and head_support is not None,
                   "whole_model_dense_fallback": mp_support is None or head_support is None,
                   "projection_fast_operations": 0, "projection_dense_operations": 0,
                   "head_fast_operations": 0, "head_dense_operations": 0}
    if diagnostics["whole_model_dense_fallback"]:
        result = helper.forward(weights, tokens, context=context)
        diagnostics["projection_dense_operations"] = geometry["N"] * geometry["T"]
        diagnostics["head_dense_operations"] = geometry["T"]
        return {**result, "guard_diagnostics": diagnostics}
    n, length, k = (geometry[key] for key in ("N", "T", "K"))
    history = [[] for _ in range(n)]
    taps, projections, offsets = ([[] for _ in range(n)] for _ in range(3))
    logits, heads = [], []
    for position in range(length):
        residual = weights["EMBED"][tokens[position]]
        for layer in range(n):
            mean = residual.mean()
            offset = mean.new_zeros(())
            # The explicit +0 matches the original smooth-forward expression,
            # including IEEE signed-zero behavior and operation association.
            centered = helper._finite(residual - mean + offset, "guarded real centering", detached=True)
            history[layer].append(centered)
            offsets[layer].append(offset)
            dilation = 1 << min(layer, 6)
            count = min(k, position // dilation + 1) if context else 1
            candidates = torch.stack([history[layer][position - tap * dilation] + weights["MH"][layer, :, tap]
                                      for tap in range(count)], dim=-1)
            conv, tap, _ = helper.reference._select(candidates, maximum=(layer % 2 == 0))
            conv = helper._finite(conv, "guarded temporal selection", detached=True)
            projected, projection, certified = _max_operation(weights["MP"][layer], conv, mp_support[layer], off_weight)
            key = "projection_fast_operations" if certified else "projection_dense_operations"
            diagnostics[key] += 1
            projected = helper._finite(projected + weights["MB"][layer], "guarded projection plus bias", detached=True)
            residual = helper._finite(residual + projected, "guarded residual", detached=True)
            taps[layer].append(tap)
            projections[layer].append(projection)
        scores, head, certified = _max_operation(weights["W_HEAD"], residual, head_support, off_weight)
        diagnostics["head_fast_operations" if certified else "head_dense_operations"] += 1
        logits.append(helper._finite(scores, "guarded head", detached=True))
        heads.append(head)
    result = {"schema": helper.reference.SCHEMA, "mode": "smooth", "geometry": geometry,
              "tokens": tokens.detach().clone(), "use_context": context,
              "logits": torch.stack(logits), "centering_offsets": torch.stack([torch.stack(rows) for rows in offsets]),
              "routes": {"AD_HEAD": torch.stack(heads),
                         "PROJ_IDX": torch.stack([torch.stack(rows) for rows in projections]),
                         "TAP_IDX": torch.stack([torch.stack(rows) for rows in taps])},
              "helper_schema": helper.SCHEMA, "units": "dimensionless",
              "parameter_sha256": helper.parameter_digest(weights), "native_execution": False,
              "independent_forward": False, "guard_diagnostics": diagnostics,
              "margins": None, "margin_scope": "not computed; sufficient guard bounds are not runner-up margins"}
    helper.trace(result)
    return result
