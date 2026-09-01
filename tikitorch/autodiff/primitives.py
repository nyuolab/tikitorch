"""Reference primitives for route-sufficient reverse differentiation.

The functions in this module deliberately separate a hard forward operation
from its reverse action.  A forward returns both its value and a deterministic
route.  The corresponding VJP consumes only the upstream adjoint, the route,
and static shape/configuration metadata; it never consumes the forward
activation values.

These are clarity-first theorem-validation kernels, not performance kernels.
All ties use the lowest flattened candidate index ("first wins").
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import torch
import torch.nn.functional as F


Tensor = torch.Tensor
ExtremeMode = Literal["max", "min"]


def _check_mode(mode: str) -> ExtremeMode:
    if mode not in ("max", "min"):
        raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
    return mode  # type: ignore[return-value]


def first_extreme(values: Tensor, *, mode: ExtremeMode = "max", dim: int = -1) -> tuple[Tensor, Tensor]:
    """Return an extreme and its deterministic first-index route.

    ``torch.maximum`` splits its autograd derivative at a two-way tie.  This
    helper instead emits an explicit selected subgradient: among exactly equal
    extrema, the lowest candidate index wins.  The returned value is gathered
    through that index so framework autograd observes the same selection.
    """

    mode = _check_mode(mode)
    if values.ndim == 0:
        raise ValueError("first_extreme requires at least one candidate dimension")
    dim = dim % values.ndim
    if values.shape[dim] == 0:
        raise ValueError("cannot select an extreme from an empty dimension")
    if torch.isnan(values).any():
        raise ValueError("NaN candidates have no deterministic extreme route")

    extreme = values.amax(dim=dim, keepdim=True) if mode == "max" else values.amin(dim=dim, keepdim=True)
    # argmax on a boolean/byte equality mask returns the first True index.
    route = (values == extreme).to(torch.uint8).argmax(dim=dim)
    selected = values.gather(dim, route.unsqueeze(dim)).squeeze(dim)
    return selected, route


def routing_margin(values: Tensor, *, mode: ExtremeMode = "max", dim: int = -1) -> Tensor:
    """Gap between the selected and second-best candidates.

    A positive margin identifies a locally stable route.  A zero margin is an
    exact tie.  A singleton candidate dimension has infinite margin.
    """

    mode = _check_mode(mode)
    dim = dim % values.ndim
    if values.shape[dim] == 0:
        raise ValueError("cannot compute a margin for an empty dimension")
    if values.shape[dim] == 1:
        shape = list(values.shape)
        del shape[dim]
        return torch.full(shape, float("inf"), dtype=values.dtype, device=values.device)
    ordered = torch.sort(values, dim=dim, descending=(mode == "max"), stable=True).values
    best = ordered.select(dim, 0)
    second = ordered.select(dim, 1)
    return best - second if mode == "max" else second - best


def additive_extreme(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    mode: ExtremeMode = "max",
) -> tuple[Tensor, Tensor]:
    """Hard additive max/min mixing with an explicit winner route.

    ``x`` has shape ``(..., in_features)`` and ``weight`` has shape
    ``(out_features, in_features)``.  The result and route have shape
    ``(..., out_features)``.
    """

    mode = _check_mode(mode)
    if x.ndim < 1 or weight.ndim != 2:
        raise ValueError("x must be (..., I) and weight must be (O, I)")
    out_features, in_features = weight.shape
    if x.shape[-1] != in_features:
        raise ValueError(f"input has {x.shape[-1]} features, weight expects {in_features}")
    if bias is not None and bias.shape != (out_features,):
        raise ValueError(f"bias must have shape {(out_features,)}, got {tuple(bias.shape)}")

    combined = x.unsqueeze(-2) + weight
    value, route = first_extreme(combined, mode=mode, dim=-1)
    if bias is not None:
        value = value + bias
    return value, route


def max_plus(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``max_i(x_i + weight[j, i]) + bias[j]`` with first-index routes."""

    return additive_extreme(x, weight, bias, mode="max")


def min_plus(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``min_i(x_i + weight[j, i]) + bias[j]`` with first-index routes."""

    return additive_extreme(x, weight, bias, mode="min")


def _validate_single_route(grad_output: Tensor, route: Tensor, input_features: int) -> Tensor:
    if input_features <= 0:
        raise ValueError("input_features must be positive")
    if route.shape != grad_output.shape:
        raise ValueError(
            f"route shape {tuple(route.shape)} must equal grad_output shape {tuple(grad_output.shape)}"
        )
    route = route.to(device=grad_output.device, dtype=torch.long)
    if route.numel() and (int(route.min()) < 0 or int(route.max()) >= input_features):
        raise ValueError("route index is outside the input feature range")
    return route


def additive_extreme_vjp(
    grad_output: Tensor,
    route: Tensor,
    input_features: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """State, additive-weight, and bias VJPs from a winner route alone.

    The output dimension is the final dimension of ``grad_output``.  Leading
    dimensions are independent examples/positions and are accumulated into the
    shared parameter gradients.
    """

    route = _validate_single_route(grad_output, route, input_features)
    out_features = grad_output.shape[-1]
    rows = grad_output.numel() // out_features
    g = grad_output.reshape(rows, out_features)
    r = route.reshape(rows, out_features)

    grad_input = torch.zeros(rows, input_features, dtype=g.dtype, device=g.device)
    grad_input.scatter_add_(1, r, g)

    output_rows = torch.arange(out_features, device=g.device).view(1, out_features)
    flat_route = output_rows * input_features + r
    grad_weight = torch.zeros(
        out_features * input_features, dtype=g.dtype, device=g.device
    )
    grad_weight.scatter_add_(0, flat_route.reshape(-1), g.reshape(-1))
    grad_weight = grad_weight.reshape(out_features, input_features)
    grad_bias = g.sum(dim=0)
    return grad_input.reshape(*grad_output.shape[:-1], input_features), grad_weight, grad_bias


def additive_parameter_vjp(grad_output: Tensor, route: Tensor, input_features: int) -> tuple[Tensor, Tensor]:
    """Return only additive weight and bias VJPs, without forward activations."""

    _, grad_weight, grad_bias = additive_extreme_vjp(grad_output, route, input_features)
    return grad_weight, grad_bias


max_plus_vjp = additive_extreme_vjp
min_plus_vjp = additive_extreme_vjp


def stable_topk(
    values: Tensor,
    k: int,
    *,
    largest: bool = True,
    dim: int = -1,
) -> tuple[Tensor, Tensor]:
    """Stable top-k selection whose ties retain ascending candidate order."""

    if values.ndim == 0:
        raise ValueError("stable_topk requires a candidate dimension")
    dim = dim % values.ndim
    n = values.shape[dim]
    if not 1 <= k <= n:
        raise ValueError(f"k must lie in [1, {n}], got {k}")
    if torch.isnan(values).any():
        raise ValueError("NaN candidates have no deterministic top-k route")
    order = torch.argsort(values, dim=dim, descending=largest, stable=True)
    route = order.narrow(dim, 0, k)
    return values.gather(dim, route), route


def topk_additive(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    k: int,
    mode: ExtremeMode = "max",
) -> tuple[Tensor, Tensor]:
    """Sum the stable top-k additive candidates and return all selected routes."""

    mode = _check_mode(mode)
    if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("x must be (..., I) and weight must be (O, I)")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError(f"bias must have shape {(weight.shape[0],)}")
    selected, route = stable_topk(
        x.unsqueeze(-2) + weight,
        k,
        largest=(mode == "max"),
        dim=-1,
    )
    value = selected.sum(dim=-1)
    if bias is not None:
        value = value + bias
    return value, route


def topk_max_plus(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, *, k: int
) -> tuple[Tensor, Tensor]:
    return topk_additive(x, weight, bias, k=k, mode="max")


def topk_min_plus(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, *, k: int
) -> tuple[Tensor, Tensor]:
    return topk_additive(x, weight, bias, k=k, mode="min")


def topk_additive_vjp(
    grad_output: Tensor,
    route: Tensor,
    input_features: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """VJP for :func:`topk_additive` from its ``(..., O, k)`` route."""

    if input_features <= 0 or route.ndim != grad_output.ndim + 1:
        raise ValueError("route must have shape (*grad_output.shape, k)")
    if route.shape[:-1] != grad_output.shape or route.shape[-1] == 0:
        raise ValueError("route must have shape (*grad_output.shape, nonzero k)")
    route = route.to(device=grad_output.device, dtype=torch.long)
    if route.numel() and (int(route.min()) < 0 or int(route.max()) >= input_features):
        raise ValueError("route index is outside the input feature range")

    out_features, k = grad_output.shape[-1], route.shape[-1]
    rows = grad_output.numel() // out_features
    g = grad_output.reshape(rows, out_features)
    r = route.reshape(rows, out_features, k)
    routed_g = g.unsqueeze(-1).expand_as(r).to(g.dtype)

    grad_input = torch.zeros(rows, input_features, dtype=g.dtype, device=g.device)
    grad_input.scatter_add_(1, r.reshape(rows, out_features * k), routed_g.reshape(rows, -1))

    output_rows = torch.arange(out_features, device=g.device).view(1, out_features, 1)
    flat_route = output_rows * input_features + r
    grad_weight = torch.zeros(
        out_features * input_features, dtype=g.dtype, device=g.device
    )
    grad_weight.scatter_add_(0, flat_route.reshape(-1), routed_g.reshape(-1))
    grad_bias = g.sum(dim=0)
    return (
        grad_input.reshape(*grad_output.shape[:-1], input_features),
        grad_weight.reshape(out_features, input_features),
        grad_bias,
    )


def morphological_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: int = 1,
    mode: ExtremeMode = "max",
    padding_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Causal depthwise additive morphological convolution.

    ``x`` is ``(batch, channels, length)`` and ``weight`` is
    ``(channels, kernel_size)``.  Route slot ``j`` reads ``x[..., t-j*dilation]``;
    positions before the sequence use the fixed ``padding_value``.  The output
    and route both have shape ``(batch, channels, length)``.
    """

    mode = _check_mode(mode)
    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError("x must be (B, C, L) and weight must be (C, K)")
    batch, channels, length = x.shape
    if weight.shape[0] != channels or weight.shape[1] == 0:
        raise ValueError("weight must have shape (x.shape[1], nonzero kernel_size)")
    if dilation <= 0:
        raise ValueError("dilation must be positive")
    if bias is not None and bias.shape != (channels,):
        raise ValueError(f"bias must have shape {(channels,)}")

    kernel_size = weight.shape[1]
    left = (kernel_size - 1) * dilation
    padded = F.pad(x, (left, 0), value=padding_value)
    # Candidate j is current input for j=0, then increasingly old inputs.
    windows = torch.stack(
        [padded[..., left - j * dilation : left - j * dilation + length] for j in range(kernel_size)],
        dim=-1,
    )
    combined = windows + weight.view(1, channels, 1, kernel_size)
    value, route = first_extreme(combined, mode=mode, dim=-1)
    if bias is not None:
        value = value + bias.view(1, channels, 1)
    return value, route


def morphological_conv1d_vjp(
    grad_output: Tensor,
    route: Tensor,
    *,
    kernel_size: int,
    dilation: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Route-only VJP for :func:`morphological_conv1d`.

    Contributions selecting left-padding still reach the additive kernel
    parameter but, correctly, do not reach an input element.
    """

    if grad_output.ndim != 3 or route.shape != grad_output.shape:
        raise ValueError("grad_output and route must both be (B, C, L)")
    if kernel_size <= 0 or dilation <= 0:
        raise ValueError("kernel_size and dilation must be positive")
    route = route.to(device=grad_output.device, dtype=torch.long)
    if route.numel() and (int(route.min()) < 0 or int(route.max()) >= kernel_size):
        raise ValueError("route index is outside the kernel range")

    _, channels, length = grad_output.shape
    grad_input = torch.zeros_like(grad_output)
    grad_weight = torch.zeros(
        channels, kernel_size, dtype=grad_output.dtype, device=grad_output.device
    )
    zero = torch.zeros((), dtype=grad_output.dtype, device=grad_output.device)
    for j in range(kernel_size):
        contribution = torch.where(route == j, grad_output, zero)
        grad_weight[:, j] = contribution.sum(dim=(0, 2))
        delay = j * dilation
        if delay < length:
            grad_input[..., : length - delay] += contribution[..., delay:]
    grad_bias = grad_output.sum(dim=(0, 2))
    return grad_input, grad_weight, grad_bias


def residual_add_vjp(grad_output: Tensor) -> tuple[Tensor, Tensor]:
    """VJP of ``skip + branch``: copy the adjoint to both parents."""

    return grad_output, grad_output


def fanout_vjp(*adjoints: Tensor) -> Tensor:
    """Accumulate contributions arriving through multiple outgoing edges."""

    if not adjoints:
        raise ValueError("fanout_vjp needs at least one adjoint")
    total = adjoints[0]
    for contribution in adjoints[1:]:
        if contribution.shape != total.shape:
            raise ValueError("all fan-out adjoints must have the same shape")
        total = total + contribution
    return total


accumulate_fanout = fanout_vjp


def concatenation_vjp(grad_output: Tensor, sizes: Sequence[int], *, dim: int = -1) -> tuple[Tensor, ...]:
    """VJP of concatenation: split the output adjoint at the saved static sizes."""

    if not sizes or any(size < 0 for size in sizes):
        raise ValueError("sizes must be a nonempty sequence of nonnegative integers")
    dim = dim % grad_output.ndim
    if sum(sizes) != grad_output.shape[dim]:
        raise ValueError("sizes do not sum to the concatenated dimension")
    return tuple(torch.split(grad_output, tuple(sizes), dim=dim))


def compose_vjp(
    output_adjoint: Tensor,
    reverse_actions: Iterable[Callable[[Tensor], Tensor]],
) -> Tensor:
    """Apply local transpose actions in output-to-input order."""

    adjoint = output_adjoint
    for action in reverse_actions:
        adjoint = action(adjoint)
    return adjoint


reverse_compose = compose_vjp


# Descriptive aliases used by the theorem text and downstream experiments.
max_plus_forward = max_plus
min_plus_forward = min_plus
morph_conv1d = morphological_conv1d
morph_conv1d_vjp = morphological_conv1d_vjp


__all__ = [
    "accumulate_fanout",
    "additive_extreme",
    "additive_extreme_vjp",
    "additive_parameter_vjp",
    "compose_vjp",
    "concatenation_vjp",
    "fanout_vjp",
    "first_extreme",
    "max_plus",
    "max_plus_forward",
    "max_plus_vjp",
    "min_plus",
    "min_plus_forward",
    "min_plus_vjp",
    "morph_conv1d",
    "morph_conv1d_vjp",
    "morphological_conv1d",
    "morphological_conv1d_vjp",
    "residual_add_vjp",
    "reverse_compose",
    "routing_margin",
    "stable_topk",
    "topk_additive",
    "topk_additive_vjp",
    "topk_max_plus",
    "topk_min_plus",
]


# ---------------------------------------------------------------------------
# Reverse actions added for the actual-model bridge (ontogenic_v2 Exp-1,
# work-list item 1; producing spec revision/ontogenic_v2/PROGRAMME.md).
# Ported record-for-record from tikitorch/tikitorch/research/pc_convrnn.py
# (_norm_adj :354-361, dropout mux :430-431, _embed_grads :604-605,
# leaky_scan_adjoint :122-142). Route data consumed: integer shift exponents,
# boolean gate masks, integer token ids, boolean scan gates — never
# activations.
# ---------------------------------------------------------------------------


def pow2_norm_vjp(adjoint: torch.Tensor, shift: torch.Tensor,
                  want_beta_grad: bool = True):
    """Reverse of the tropical layer norm y = (x - mean(x)) * 2^-shift + beta.

    ``shift`` is the integer dyadic exponent tape record (17 codes in
    [-8, 8] for the current model). Uses ``torch.ldexp`` — an exponent
    add, not a float multiply (closure-purity improvement over the
    inherited ``e * f``; bit-identical for power-of-two f).

    Returns ``(e_input, beta_grad)``; ``beta_grad`` is ``None`` when not
    requested.
    """
    if adjoint.isnan().any():
        raise ValueError("adjoint contains NaN")
    if shift.dtype not in (torch.int8, torch.int16, torch.int32,
                           torch.int64):
        raise TypeError(f"shift must be integer, got {shift.dtype}")
    beta_grad = (adjoint.reshape(-1, adjoint.shape[-1]).sum(0)
                 if want_beta_grad else None)
    ec = torch.ldexp(adjoint, (-shift).to(adjoint.device))
    return ec - ec.mean(dim=-1, keepdim=True), beta_grad


def gate_vjp(adjoint: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse of a {0,1} mux gate (tropical dropout / keep-mask): pass the
    adjoint where the gate kept, zero where it dropped. A mux, not a
    multiply."""
    if adjoint.shape != mask.shape:
        raise ValueError(f"shape mismatch {adjoint.shape} vs {mask.shape}")
    zero = torch.zeros((), dtype=adjoint.dtype, device=adjoint.device)
    return torch.where(mask.bool(), adjoint, zero)


def embedding_gather_vjp(adjoint: torch.Tensor, ids: torch.Tensor,
                         num_embeddings: int) -> torch.Tensor:
    """Reverse of an embedding row gather: scatter-accumulate the adjoint
    into the gathered rows. ``ids`` is the token-id tape record."""
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"ids must be integer, got {ids.dtype}")
    d = adjoint.shape[-1]
    flat_e = adjoint.reshape(-1, d)
    flat_i = ids.reshape(-1)
    if flat_e.shape[0] != flat_i.shape[0]:
        raise ValueError("adjoint/ids length mismatch")
    if int(flat_i.min()) < 0 or int(flat_i.max()) >= num_embeddings:
        raise ValueError("ids out of range")
    out = torch.zeros(num_embeddings, d, dtype=adjoint.dtype,
                      device=adjoint.device)
    out.index_add_(0, flat_i, flat_e)
    return out


def leaky_scan_vjp(adjoint: torch.Tensor, gates: torch.Tensor):
    """Frozen-gate reverse of h_t = max(u_t, h_{t-1} - lam): the unit-gain
    cascade-through-time. ``gates`` (bool, (B,C,T)) is the tape record:
    True where the carry branch won. Returns ``(e_u, lam_grad)`` with
    ``lam_grad`` the -1-edge count accumulation. Gather/gate/add only.
    Exact under frozen routes; a route-changing counterexample is a
    separate reporting category (never silently mixed)."""
    if adjoint.shape != gates.shape:
        raise ValueError(f"shape mismatch {adjoint.shape} vs {gates.shape}")
    if gates.dtype != torch.bool:
        raise TypeError(f"gates must be bool, got {gates.dtype}")
    B, C, T = adjoint.shape
    e_u = torch.empty_like(adjoint)
    carry = torch.zeros(B, C, dtype=adjoint.dtype, device=adjoint.device)
    lam_grad = torch.zeros(C, dtype=adjoint.dtype, device=adjoint.device)
    zero = torch.zeros((), dtype=adjoint.dtype, device=adjoint.device)
    for t in range(T - 1, -1, -1):
        tot = adjoint[:, :, t] + carry
        g = gates[:, :, t]
        e_u[:, :, t] = torch.where(g, zero, tot)
        carry = torch.where(g, tot, zero)
        lam_grad -= torch.where(g, tot, zero).sum(0)
    return e_u, lam_grad
