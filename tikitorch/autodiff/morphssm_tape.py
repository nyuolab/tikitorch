"""Device-native route training and portable certificates for hard MorphSSM.

``forward_tape`` streams only token ids, first-index winner routes, top-k
indices, dropout gates and dyadic normalization shifts into a DeviceRouteTape.
``vjp`` consumes those decisions with a caller-supplied output cotangent;
``route_logits`` integrates the same first-order reverse sweep with autograd.
These APIs support the validated plain token-embedding LM at tau=0, including
ties under the shared first-index convention. Soft layers, custom route-less
backends and modified model behavior are rejected.

The historical engine-cache adapter and mean-cross-entropy ``replay`` wrapper
remain available. Portable format-v2 tapes are configuration-fingerprinted;
older fingerprints/tie conventions require their archived source revision.
"""
from __future__ import annotations

from typing import Any

import torch

from .packed_tape import PackedRouteTape
from .device_tape import DeviceRouteTape
from .primitives import embedding_gather_vjp, pow2_norm_vjp

SHIFT_LO, SHIFT_HI = -8, 8            # the model's clamp (pc_convrnn:245)


def _dropout_gate(x, gate):
    """Match TropicalDropout's IEEE multiplication, including NaNs/signed zero."""
    return x * gate.to(x.dtype)


def model_fingerprint(model, tau: float = 0.0) -> dict[str, Any]:
    """Configuration binding for tape serialization (format v2)."""
    layer0 = model.layers[0]
    return dict(
        arch="morphssm",
        d_model=int(model.embed.weight.shape[1]),
        vocab=int(model.embed.weight.shape[0]),
        n_layers=len(model.layers),
        kernel=int(layer0.morph_conv.kernel_size),
        kernels=[int(layer.morph_conv.kernel_size) for layer in model.layers],
        dilations=[int(layer.morph_conv.dilation) for layer in model.layers],
        modes=[layer.morph_conv.mode for layer in model.layers],
        norm_eps=[float(n.eps) for layer in model.layers for n in (layer.norm, layer.post_norm)]
                 + [float(model.ln_final.eps)],
        head_k=int(model.output_head.k),
        tau=float(tau),
        tie_rule="first-index-v1",
        dtype=str(model.embed.weight.dtype),
        parameters=[[name, list(p.shape), str(p.dtype)] for name, p in model.named_parameters()],
    )


def _assert_hard(model) -> None:
    from tikitorch.models import MorphSSM
    from tikitorch.nn import DepthwiseMorphConv1d, MorphSSMLayer, TropicalDilationModule, TropicalDropout, TropicalLayerNorm, TropicalTopKModule, _uses_reference_tropical_op
    if type(model) is not MorphSSM:
        raise ValueError("route replay requires the unmodified MorphSSM implementation")
    from torch.nn.modules import module as module_runtime
    if any(getattr(module_runtime, name, None) for name in (
        "_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
        "_global_backward_pre_hooks"
    )):
        raise ValueError("global module hooks are outside the route contract")
    for module in model.modules():
        if "forward" in module.__dict__ or any(getattr(module, name, None) for name in (
            "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"
        )):
            raise ValueError("custom forward overrides and module hooks are outside the route contract")
    if getattr(model, "task_type", None) != "lm" or type(model.embed) is not torch.nn.Embedding:
        raise ValueError("route replay requires the MorphSSM token-embedding LM configuration")
    if not len(model.layers) or type(model.output_head) is not TropicalTopKModule or type(model.ln_final) is not TropicalLayerNorm:
        raise ValueError("route replay requires layers, tropical normalization and a top-k head")
    embed = model.embed
    if embed.padding_idx is not None or embed.scale_grad_by_freq or embed.sparse or embed.max_norm is not None:
        raise ValueError("route replay requires a plain dense embedding without padding/max_norm/frequency scaling")
    if any(type(drop) not in (torch.nn.Identity, TropicalDropout)
           for drop in [model.dropout, *(layer.dropout for layer in model.layers)]):
        raise ValueError("route replay supports only Identity or TropicalDropout gates")
    for layer in model.layers:
        if not _uses_reference_tropical_op("tropical_dilation", getattr(layer.out_proj, "_tt_device", None),
                                           tensor=model.embed.weight):
            raise ValueError("custom backend kernels need an explicit route-derivative contract")
    if not _uses_reference_tropical_op("stable_topk", getattr(model.output_head, "_tt_device", None),
                                       tensor=model.embed.weight):
        raise ValueError("custom top-k kernels need an explicit route-derivative contract")
    names = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in names}) != len(names):
        raise ValueError("aliased parameters require an explicit alias-accumulation contract")
    if getattr(model, "action_head", None) is not None or getattr(model, "attn_at", None) is not None:
        raise ValueError("route replay supports the plain MorphSSM readout only")
    for i, layer in enumerate(model.layers):
        if type(layer) is not MorphSSMLayer or type(layer.morph_conv) is not DepthwiseMorphConv1d or layer.morph_conv.group_size != 1:
            raise ValueError("route replay requires single-scale depthwise convolution")
        if type(layer.out_proj) is not TropicalDilationModule or not all(
            type(n) is TropicalLayerNorm for n in (layer.norm, layer.post_norm)
        ) or layer.residual_scale != 1.0:
            raise ValueError("route replay requires tropical projections/norms and unit residual scale")
        if getattr(layer.out_proj, "use_softmax", False):
            raise AssertionError(f"layer {i}: out_proj soft (tau>0) — "
                                 "packed tape is a tau=0 claim only")
        conv = layer.morph_conv
        if getattr(conv, "use_soft", False) and getattr(conv, "tau", 0) > 0:
            raise AssertionError(f"layer {i}: morph_conv soft — tau=0 only")
        if getattr(layer, "scan", None) is not None:
            raise AssertionError("ConvRNN scan present — use the carry-gate "
                                 "record (out of MorphSSM scope)")


def _shift_of(f: torch.Tensor, name: str) -> torch.Tensor:
    """Recover the integer dyadic exponent from a pow-2 factor tensor,
    validating exactness (f must be exp2(-shift), shift in [-8, 8])."""
    shift = torch.round(-torch.log2(f)).to(torch.int64)
    if not torch.equal(torch.exp2((-shift).to(f.dtype)), f):
        raise ValueError(f"{name}: factor is not an exact power of two")
    if int(shift.min()) < SHIFT_LO or int(shift.max()) > SHIFT_HI:
        raise ValueError(f"{name}: shift outside [{SHIFT_LO}, {SHIFT_HI}]")
    return shift


def pack_engine_cache(caches, top, model, ids: torch.Tensor,
                      ) -> PackedRouteTape:
    """caches/top: the `MorphPCEngine._forward_cache` outputs; ids: the
    input token tensor (B, T). Returns the minimal route tape."""
    _assert_hard(model)
    D = int(model.embed.weight.shape[1])
    V = int(model.embed.weight.shape[0])
    tape = PackedRouteTape()
    tape._fingerprint = model_fingerprint(model)
    for li, (layer, c) in enumerate(zip(model.layers, caches)):
        K = int(layer.morph_conv.kernel_size)
        cr = c["conv_route"]
        if cr is None or cr.is_floating_point():
            raise AssertionError(f"layer {li}: no hard conv route")
        tape.append(f"layer{li}.conv_winner", cr, cardinality=K)
        pr = c["proj_route"]
        if pr is None:
            raise AssertionError(f"layer {li}: no hard proj route")
        tape.append(f"layer{li}.proj_winner", pr, cardinality=D)
        if c["drop_mask"] is not None:
            tape.append_mask(f"layer{li}.drop_gate",
                             c["drop_mask"].to(torch.bool))
        tape.append_signed(f"layer{li}.f1_shift",
                           _shift_of(c["f1"], f"layer{li}.f1"),
                           lo=SHIFT_LO, hi=SHIFT_HI)
        tape.append_signed(f"layer{li}.f2_shift",
                           _shift_of(c["f2"], f"layer{li}.f2"),
                           lo=SHIFT_LO, hi=SHIFT_HI)
    if top["mask0"] is not None:
        tape.append_mask("embed_drop_gate", top["mask0"].to(torch.bool))
    tape.append_signed("f_final_shift",
                       _shift_of(top["f_final"], "f_final"),
                       lo=SHIFT_LO, hi=SHIFT_HI)
    tape.append("head_winner", top["head_idx"], cardinality=D)
    tape.append("token_ids", ids, cardinality=V)
    return tape


@torch.no_grad()
def forward_tape(model, ids: torch.Tensor, *, packed: bool = True):
    """Execute hard MorphSSM and record routes as each layer completes.

    Returns ``(logits, DeviceRouteTape)``. No MorphPCEngine or activation-cache
    list is created. Normalization shifts and dropout gates are auxiliary state,
    explicitly included in tape bytes. Packing is on the input device.
    """
    _assert_hard(model)
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0:
        raise ValueError("ids must be a nonempty (batch, sequence) tensor")
    if ids.device != model.embed.weight.device:
        raise ValueError("ids and model must be on the same device")
    tape = DeviceRouteTape(device=ids.device, packed=packed,
                           fingerprint=model_fingerprint(model), validate=False)
    D, V = model.embed.weight.shape[1], model.embed.weight.shape[0]

    def norm_forward(norm, x, field):
        centered = x - x.mean(dim=-1, keepdim=True)
        scale = centered.abs().mean(dim=-1, keepdim=True).clamp_min(norm.eps)
        shift = torch.round(torch.log2(scale)).clamp(-8, 8)
        tape.append_signed(field, shift.to(torch.long), lo=SHIFT_LO, hi=SHIFT_HI)
        y = centered * torch.exp2(-shift)
        return y if norm.beta is None else y + norm.beta

    def dropout_forward(module, x, field):
        from tikitorch.nn import TropicalDropout
        if isinstance(module, TropicalDropout) and module.training and module.p:
            gate = torch.rand_like(x) >= module.p
            tape.append_mask(field, gate)
            return _dropout_gate(x, gate)
        return x

    h = dropout_forward(model.dropout, model.embed(ids), "embed_drop_gate")
    B, T = ids.shape
    for li, layer in enumerate(model.layers):
        prefix = f"layer{li}"
        xn = norm_forward(layer.norm, h, f"{prefix}.f1_shift")
        u, route = layer.morph_conv.forward_with_routing(xn.transpose(1, 2))
        tape.append(f"{prefix}.conv_winner", route,
                    cardinality=int(layer.morph_conv.kernel_size))
        v = dropout_forward(layer.dropout, u.transpose(1, 2), f"{prefix}.drop_gate")
        y, route = layer.out_proj.forward_with_routing(v.reshape(B * T, D))
        tape.append(f"{prefix}.proj_winner", route, cardinality=D)
        h = norm_forward(layer.post_norm, h + y.reshape(B, T, D), f"{prefix}.f2_shift")
    hf = norm_forward(model.ln_final, h, "f_final_shift")
    rows = hf.reshape(B * T, D)
    logits, idx = model.output_head.forward_with_routing(rows)
    tape.append("head_winner", idx, cardinality=D)
    tape.append("token_ids", ids, cardinality=V)
    return logits.reshape(B, T, V), tape


def serialize(tape: PackedRouteTape, model, tau: float = 0.0) -> bytes:
    return tape.to_bytes(fingerprint=model_fingerprint(model, tau))


def deserialize(blob: bytes, model, tau: float = 0.0) -> PackedRouteTape:
    return PackedRouteTape.from_bytes(
        blob, expected_fingerprint=model_fingerprint(model, tau))


@torch.no_grad()
def replay(tape: PackedRouteTape, model, logits: torch.Tensor,
           targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean-CE convenience boundary matching the historical cascade recipe.

    For another loss, pass its output cotangent to :func:`vjp` instead.
    """
    B, T, V = logits.shape
    R = B * T
    if R == 0 or targets.shape != (B, T) or targets.device != logits.device:
        raise ValueError("targets must match the nonempty logits batch and device")
    eps = torch.softmax(logits.reshape(-1, V).float(), dim=-1)
    eps.scatter_add_(1, targets.reshape(-1, 1),
                     -torch.ones(R, 1, device=eps.device))
    return vjp(tape, model, (eps / R).to(logits.dtype).reshape(B, T, V))


@torch.no_grad()
def vjp(tape: PackedRouteTape | DeviceRouteTape, model,
        output_cotangent: torch.Tensor) -> dict[str, torch.Tensor]:
    """Loss-independent selected parameter VJP; never reads forward activations.

    Does not modify parameter ``.grad``. The tape binds configuration, not weight
    values: it describes the recorded forward even if live parameter values later
    change. The supported hard operators' parameter VJPs do not use those values.
    """
    _assert_hard(model)
    if tape.fingerprint is not None and tape.fingerprint != model_fingerprint(model):
        raise ValueError("route tape fingerprint mismatch")
    if output_cotangent.ndim != 3:
        raise ValueError("output cotangent must have shape (batch, sequence, vocabulary)")
    B, T, V = output_cotangent.shape
    token_field = next(f for f in tape.fields if f.name == "token_ids")
    if token_field.shape != (B, T) or V != model.embed.weight.shape[0] or B * T == 0:
        raise ValueError("output cotangent shape does not match the recorded forward")
    if output_cotangent.device != model.embed.weight.device or output_cotangent.dtype != model.embed.weight.dtype:
        raise ValueError("output cotangent must match model dtype and device")
    D = int(model.embed.weight.shape[1])
    R, dt, dev = B * T, output_cotangent.dtype, output_cotangent.device
    grads: dict[str, torch.Tensor] = {}
    eps = output_cotangent.reshape(R, V)
    def read(name):
        return tape.read(name, device=dev)

    head = model.output_head
    k = head.k
    idx = read("head_winner")                           # (R, V, k)
    e_exp = eps.unsqueeze(-1).expand(R, V, k)
    e_x = torch.zeros(R, D, dtype=dt, device=dev)
    e_x.scatter_add_(1, idx.reshape(R, V * k), e_exp.reshape(R, V * k))
    flat = (torch.arange(0, V * D, D, device=dev).view(1, V, 1) + idx)
    wg = torch.zeros(V * D, dtype=dt, device=dev)
    wg.index_add_(0, flat.reshape(-1), e_exp.reshape(-1))
    grads["output_head.weight"] = wg.view(V, D)
    grads["output_head.bias"] = eps.sum(0)
    e = e_x.reshape(B, T, D)
    e, bg = pow2_norm_vjp(e, read("f_final_shift"), validate=False)
    grads["ln_final.beta"] = bg

    # ---- layer sweep (mirrors _layer_adjoint, hard path) -------------------
    for li in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[li]
        e2, bg2 = pow2_norm_vjp(e, read(f"layer{li}.f2_shift"), validate=False)
        grads[f"layers.{li}.post_norm.beta"] = bg2
        e_res = e2
        # proj adjoint (hard winner scatter; O == I == D)
        pe = e2.reshape(R, D)
        pidx = read(f"layer{li}.proj_winner")           # (R, D)
        e_proj = torch.zeros(R, D, dtype=dt, device=dev)
        e_proj.scatter_add_(1, pidx, pe)
        pflat = (torch.arange(0, D * D, D, device=dev).unsqueeze(0) + pidx)
        pwg = torch.zeros(D * D, dtype=dt, device=dev)
        pwg.index_add_(0, pflat.reshape(-1), pe.reshape(-1))
        grads[f"layers.{li}.out_proj.weight"] = pwg.view(D, D)
        grads[f"layers.{li}.out_proj.bias"] = pe.sum(0)
        e_v = e_proj.reshape(B, T, D)
        try:
            gate = read(f"layer{li}.drop_gate")
        except KeyError:
            gate = None
        if gate is not None:
            e_v = _dropout_gate(e_v, gate)
        e_u = e_v.transpose(1, 2)                        # (B, C, T)
        # conv adjoint (hard mux)
        conv = layer.morph_conv
        K, dil = int(conv.kernel_size), int(conv.dilation)
        route = read(f"layer{li}.conv_winner")          # (B, C, T)
        e_in_c = torch.zeros_like(e_u)
        cwg = torch.zeros(D, K, dtype=dt, device=dev)
        zero = torch.zeros((), dtype=dt, device=dev)
        for j in range(K):
            contrib = torch.where(route == j, e_u, zero)
            cwg[:, j] = contrib.sum(dim=(0, 2))
            if j * dil < T:
                e_in_c[:, :, :T - j * dil] += contrib[:, :, j * dil:]
        grads[f"layers.{li}.morph_conv.weight"] = cwg
        grads[f"layers.{li}.morph_conv.bias"] = e_u.sum(dim=(0, 2))
        e_xn = e_in_c.transpose(1, 2)
        e1, bg1 = pow2_norm_vjp(e_xn, read(f"layer{li}.f1_shift"), validate=False)
        grads[f"layers.{li}.norm.beta"] = bg1
        e = e_res + e1

    # ---- embedding ---------------------------------------------------------
    try:
        m0 = read("embed_drop_gate")
    except KeyError:
        m0 = None
    if m0 is not None:
        e = _dropout_gate(e, m0)
    grads["embed.weight"] = embedding_gather_vjp(
        e, read("token_ids"), int(model.embed.weight.shape[0]), validate=False)
    return {name: grad for name, grad in grads.items() if name in dict(model.named_parameters())}


class _RouteForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, packed, ids, *parameters):
        logits, tape = forward_tape(model, ids, packed=packed)
        ctx.model = model
        ctx.names = tuple(name for name, _ in model.named_parameters())
        # Route tensors participate in PyTorch saved-tensor hooks/offloading.
        ctx.save_for_backward(*tape._storage.values())
        ctx.storage_names = tuple(tape._storage)
        tape._storage = {}
        ctx.tape = tape
        return logits

    @staticmethod
    def backward(ctx, output_cotangent):
        if torch.is_grad_enabled():
            raise RuntimeError("route training supports first-order derivatives only")
        tape = ctx.tape.clone()
        tape._storage = dict(zip(ctx.storage_names, ctx.saved_tensors))
        grads = vjp(tape, ctx.model, output_cotangent)
        return (None, None, None, *(grads.get(name) for name in ctx.names))


def route_logits(model, ids: torch.Tensor, *, packed: bool = True):
    """Hard MorphSSM logits with route-only first-order PyTorch autograd.

    Use an ordinary loss, ``loss.backward()``, and a PyTorch optimizer. The
    supplied-cotangent replay API remains available for manual training loops.
    Soft layers, recurrence/action-head extensions and higher derivatives are
    explicitly unsupported rather than silently approximated.
    """
    return _RouteForward.apply(model, packed, ids, *model.parameters())
