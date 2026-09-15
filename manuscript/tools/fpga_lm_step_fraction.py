"""Uncommitted dyadic step counterfactuals preserving an existing EF shadow.

The base state uses k18 and the four supplied raw window gradients are summed,
never averaged. This is not a new native optimizer or permission to train.
For alpha=2**-j, scale the old carry by 2**j BEFORE applying the unchanged raw
gradient at denominator 2**(18+j). The inherited shadow is therefore unchanged.
Arithmetic-only shadow/ideal tensors are not permission to execute an out-of-
range forward; callers must enforce the appropriate forward-model guards.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from manuscript.tools import fpga_lm_accumulation_arithmetic as accumulation

arithmetic = accumulation.arithmetic
WEIGHTS = arithmetic.WEIGHTS
SCHEMA = "tikitorch.fpga_lm_step_fraction.v1"
BASE_K = 18
SHIFTS = (None, 4, 2, 0)
GRADIENTS_PER_BLOCK = 4


def source_identity():
    path = Path(__file__).resolve()
    return {**accumulation.source_identity(),
            path.relative_to(path.parents[2]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}


def step_fraction(weights, remainders, gradients, shift):
    """Return a fresh, uncommitted candidate for one declared step fraction.

    Inputs are the five full natural-shaped, detached CPU FP64 exact-integer
    mappings used by the frozen arithmetic oracle. Exactly four gradients in
    their declared order are required. ``shift=None`` means alpha0, not k0:
    it preserves the complete original state and checks raw gradient prefixes,
    but never checks a hypothetical nonzero update. Other literal shifts are
    4,2,0 for alpha1/16,1/4,1 respectively; booleans/floats are rejected.

    Every raw prefix and every scaled-carry-plus-prefix fits int32 for the
    actual nonzero fraction. Candidate integer weights fit int16, with no
    clipping. Public/frozen parity and Q'*w'-r'=Q'*z-G prove preservation of
    the realized compensated state, not loss descent or native execution.
    """
    if shift is not None and (type(shift) is not int or shift not in (4, 2, 0)):
        raise ValueError("shift must be exactly None, 4, 2, or 0")
    if not isinstance(gradients, (list, tuple)) or len(gradients) != GRADIENTS_PER_BLOCK:
        raise ValueError("exactly four ordered full gradients are required")
    arithmetic._native(weights, remainders, 1 << BASE_K)
    shadow = arithmetic.shadow(weights, remainders, k=BASE_K)
    if shift is None:
        zero = {name: torch.zeros_like(weights[name]) for name in WEIGHTS}
        # The unused real G is validated, but is not applied to r or w.
        gradient, raw_maxima = accumulation._checked_sum(gradients, zero)
        candidate = accumulation.apply_block(weights, remainders, [zero], k=BASE_K)
        effective_k, alpha, factor = BASE_K, 0.0, 1
        maxima = {"gradient": raw_maxima["gradient"],
                  "remainder_plus_gradient": candidate["prefix_max_abs"]["remainder_plus_gradient"]}
        prefix_scope = "raw G prefixes checked; actual zero-update carry unchanged"
    else:
        effective_k, alpha, factor = BASE_K + shift, 2.0 ** (-shift), 1 << shift
        # Canonical base r is below 2**18, so this multiplication is exact and
        # within int32 for all declared j. apply_block checks every later sum.
        scaled_r = {name: remainders[name] * factor for name in WEIGHTS}
        candidate = accumulation.apply_block(weights, scaled_r, gradients, k=effective_k)
        gradient, maxima = candidate["gradient"], candidate["prefix_max_abs"]
        prefix_scope = "raw G and scaled-old-carry-plus-G at every ordered prefix"
    denominator = 1 << effective_k
    fresh_ideal, ideal = {}, {}
    for name in WEIGHTS:
        decrement = torch.zeros_like(gradient[name]) if shift is None else gradient[name] / denominator
        fresh_ideal[name] = weights[name] - decrement
        ideal[name] = shadow[name] - decrement
        before = (weights[name].to(torch.int64) * (1 << BASE_K)
                  - remainders[name].to(torch.int64)) * factor
        after = (candidate["weights"][name].to(torch.int64) * denominator
                 - candidate["remainders"][name].to(torch.int64))
        applied = torch.zeros_like(before) if shift is None else gradient[name].to(torch.int64)
        if bool((after - before + applied).any()):
            raise ValueError("exact carry-preserving fraction identity failed in " + name)
        if not torch.equal(after.to(torch.float64) / denominator, ideal[name]):
            raise ValueError("dyadic ideal and candidate shadow differ in " + name)
    if shift is None:
        for name in WEIGHTS:
            if (not torch.equal(candidate["weights"][name], weights[name])
                    or not torch.equal(candidate["remainders"][name], remainders[name])):
                raise ValueError("zero fraction changed original state")
    return {"schema": SCHEMA, "shift": shift, "alpha": alpha, "base_k": BASE_K,
            "k": effective_k, "weights": candidate["weights"], "remainders": candidate["remainders"],
            "gradient": gradient, "shadow_before": shadow, "ideal": ideal, "fresh_ideal": fresh_ideal,
            "prefix_max_abs": maxima, "prefix_scope": prefix_scope,
            "public_reference_equal": candidate["public_reference_equal"],
            "exact_shadow_recurrence": True, "committed": False, "native_execution": False}
