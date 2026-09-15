"""Checked summed-gradient error feedback; no model, corpus, or training run.

Mappings have the five natural parameter shapes and detached CPU FP64 exact
integers used by the frozen trajectory arithmetic. Gradients are summed, never
averaged. The caller supplies their order and where they were evaluated; this
helper does not imply equivalence to sequential, separately evaluated updates.
Even for a fixed gradient list, grouping can change visible weights/remainders
under signed truncation while preserving the exact final shadow state.
Checking supplied gradients cannot certify the earlier backward accumulations
that produced each one; those retain their own range and native-oracle gates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from manuscript.tools import fpga_lm_trajectory_arithmetic as arithmetic
from tikitorch.optim import fixed_point_ef_step

SCHEMA = "tikitorch.fpga_lm_accumulation_arithmetic.v1"
WEIGHTS = arithmetic.WEIGHTS
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = (Path(__file__).resolve(), root / "tikitorch/optim/fixed_point.py",
             root / "tikitorch/optim/__init__.py")
    return {**arithmetic.source_identity(),
            **{path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in paths}}


def _check_int32(value, label):
    if bool(((value < INT32_MIN) | (value > INT32_MAX)).any()):
        raise OverflowError(label + " would overflow int32")


def _checked_sum(gradients, remainders):
    if not isinstance(gradients, (list, tuple)) or not gradients:
        raise ValueError("gradients must be a nonempty list or tuple of mappings")
    arithmetic._mapping(remainders, "remainders")
    arithmetic._integers(remainders, INT32_MIN, INT32_MAX, "remainders")
    wide_r = {name: remainders[name].to(torch.int64) for name in WEIGHTS}
    total = {name: torch.zeros_like(wide_r[name]) for name in WEIGHTS}
    maxima = {"gradient": 0,
              "remainder_plus_gradient": max(int(value.abs().max()) for value in wide_r.values())}
    for index, gradient in enumerate(gradients, 1):
        arithmetic._gradient(gradient, remainders, "gradient[" + str(index - 1) + "]")
        for name in WEIGHTS:
            # Every incoming total and individual gradient fits int32, so this
            # addition and the following addition of r cannot overflow int64.
            prefix = total[name] + gradient[name].to(torch.int64)
            label = "prefix " + str(index) + " in " + name
            _check_int32(prefix, "gradient " + label)
            accumulated = wide_r[name] + prefix
            _check_int32(accumulated, "remainder + gradient " + label)
            maxima["gradient"] = max(maxima["gradient"], int(prefix.abs().max()))
            maxima["remainder_plus_gradient"] = max(
                maxima["remainder_plus_gradient"], int(accumulated.abs().max()))
            total[name] = prefix
    return {name: total[name].to(torch.float64) for name in WEIGHTS}, maxima


def checked_sum(gradients, remainders):
    """Return a fresh full gradient after checking EVERY ordered int32 prefix.

    Both sum(G[:i]) and r+sum(G[:i]) must fit int32; later cancellation cannot
    excuse an earlier overflow. No input is mutated, even on failure. With no
    shift argument this function checks int32 remainders, not canonicality;
    ``apply_block`` additionally requires abs(r)<2**k. List/tuple input keeps
    block membership explicit and rejects empty or unbounded generators.
    """
    return _checked_sum(gradients, remainders)[0]


def apply_block(weights, remainders, gradients, k=18):
    """Apply ONE EF update to the checked sum, asserting independent parity.

    Returns fresh FP64 mappings ``weights``, ``remainders``, and ``gradient``;
    ``prefix_max_abs`` reports global absolute maxima for the raw prefix and
    remainder-plus-prefix (including the zero prefix). The literal parity and
    shadow flags are emitted only after checking public-library results against
    the frozen oracle and Q*w_next-r_next == Q*w-r-sum(G) in exact int64.
    This is a range/algebra check, not a loss-descent or hardware claim.
    """
    denominator = arithmetic._denominator(k)
    arithmetic._native(weights, remainders, denominator)
    gradient, maxima = _checked_sum(gradients, remainders)
    reference = arithmetic.ef_step(weights, remainders, gradient, k)
    result_w, result_r = {}, {}
    for name in WEIGHTS:
        public_w, public_r = fixed_point_ef_step(
            weights[name].to(torch.int16), remainders[name].to(torch.int32),
            gradient[name].to(torch.int32), k=k)
        # Assert storage contract before comparing values: torch.equal alone
        # allows tensors of different dtypes to compare equal.
        for value, dtype, label in ((public_w, torch.int16, "weights"),
                                    (public_r, torch.int32, "remainders")):
            if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                    or value.device.type != "cpu" or value.layout != torch.strided
                    or value.requires_grad or value.shape != weights[name].shape):
                raise ValueError("public EF " + label + " contract differs in " + name)
        result_w[name] = public_w.to(torch.float64)
        result_r[name] = public_r.to(torch.float64)
        if (not torch.equal(result_w[name], reference["weights"][name])
                or not torch.equal(result_r[name], reference["remainders"][name])):
            raise ValueError("public and frozen EF results differ in " + name)
        before = weights[name].to(torch.int64) * denominator - remainders[name].to(torch.int64)
        after = public_w.to(torch.int64) * denominator - public_r.to(torch.int64)
        if bool((after - before + gradient[name].to(torch.int64)).any()):
            raise ValueError("exact EF shadow recurrence failed in " + name)
    return {"weights": result_w, "remainders": result_r, "gradient": gradient,
            "prefix_max_abs": maxima, "public_reference_equal": True,
            "exact_shadow_recurrence": True}
