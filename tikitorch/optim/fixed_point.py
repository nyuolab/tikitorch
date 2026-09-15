"""Experimental, functional CPU reference for fixed-point error feedback.

This integer update is distinct from ``EFTropicalSGD``. It is not an autograd
optimizer, a hardware implementation, or evidence of a learning/speed advantage.
"""
from __future__ import annotations

import torch


def fixed_point_ef_step(
    weights: torch.Tensor,
    remainder: torch.Tensor,
    gradient: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fresh ``(weights, remainder)`` after one checked integer EF step.

    With ``Q = 2**k``, the coordinate-wise rule in raw integer units is::

        accumulator = remainder + gradient
        update = trunc_toward_zero(accumulator / Q)
        next_weights = weights - update
        next_remainder = accumulator - update * Q

    All inputs must be dense strided CPU tensors with identical shapes: int16
    weights and int32 remainder/gradient. Scalars, empty tensors and strided
    views are supported; broadcasting and floating-point conversion are not.
    ``k`` must be a literal Python integer in [0, 30], and every input remainder
    must satisfy ``abs(remainder) < Q``. The caller owns and checkpoints BOTH
    returned tensors and the fixed ``k``; no optimizer state is stored here.

    The addition is widened for checking but must fit int32, matching the
    native retained-accumulator contract. Every candidate weight must fit
    int16. Overflow raises instead of clipping or wrapping, and no input is
    mutated even if another coordinate fails. Outputs never alias inputs.
    This check cannot certify earlier accumulation steps used to form the
    supplied gradient, or certify that that gradient is a loss derivative.

    The exact identity ``(weights-next_weights)*Q + next_remainder =
    remainder + gradient`` preserves fractional updates. Equivalently the
    shadow state ``weights-remainder/Q`` advances by ``-gradient/Q`` for the
    supplied gradient sequence; this is not a loss-descent guarantee or a
    claim of equality to a separately evolving floating-point trajectory.

    Raises:
        TypeError: A tensor, dtype, or literal-integer requirement is violated.
        ValueError: A shape, layout, device, shift, or remainder is invalid.
        OverflowError: The int32 accumulator or int16 candidate would overflow.
    """
    if type(k) is not int:
        raise TypeError("k must be a literal Python integer")
    if not 0 <= k <= 30:
        raise ValueError("k must be in [0, 30]")
    for name, value, dtype in (
        ("weights", weights, torch.int16),
        ("remainder", remainder, torch.int32),
        ("gradient", gradient, torch.int32),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if value.layout != torch.strided or value.is_nested:
            raise ValueError(f"{name} must be a dense strided tensor")
    if weights.device != remainder.device or weights.device != gradient.device:
        raise ValueError("weights, remainder and gradient must have identical devices")
    if weights.shape != remainder.shape or weights.shape != gradient.shape:
        raise ValueError("weights, remainder and gradient must have identical shapes")

    denominator = 1 << k
    wide_remainder = remainder.to(torch.int64)
    if torch.any((wide_remainder <= -denominator) | (wide_remainder >= denominator)):
        raise ValueError("remainder is not canonical: require abs(remainder) < 2**k")
    accumulator = wide_remainder + gradient.to(torch.int64)
    if torch.any((accumulator < -(1 << 31)) | (accumulator > (1 << 31) - 1)):
        raise OverflowError("remainder + gradient would overflow int32")
    update = torch.div(accumulator, denominator, rounding_mode="trunc")
    next_weights = weights.to(torch.int64) - update
    if torch.any((next_weights < -32768) | (next_weights > 32767)):
        raise OverflowError("candidate weight would overflow int16")
    next_remainder = accumulator - update * denominator
    # Both dtype conversions allocate independent storage, including for k=0.
    return next_weights.to(torch.int16), next_remainder.to(torch.int32)
