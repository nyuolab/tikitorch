"""Actual-MorphSSM bridge for packed route certificates.

Packs :class:`tikitorch.research.pc_convrnn.MorphPCEngine` forward-cache output into a
:class:`PackedRouteTape` (format v2, model-fingerprinted) retaining ONLY the
mathematically required decisions — conv winners (3b at K=7), proj winners
(log2 D bits), head top-k indices, dropout gates (1b), the 17-code dyadic
norm shifts (5b signed), and token ids — and replays the exact reverse
sweep of `MorphPCEngine.cascade_grads` from the tape plus declared boundary
state (logits, targets).

Contract (preregistered):
- tau = 0 ONLY. Every layer is asserted hard (`use_softmax`/`use_soft`
  False); the soft phase retains activations by construction and is out of
  scope for the packed claim.
- Routes are CONSUMED from the model's own forward cache, never recomputed
  (two documented tie-rule divergences vs the reference kernels).
- replay(tape) ≡ cascade_grads ALWAYS (same recorded decisions);
  ≡ framework autograd exactly when no route sits at a tie (ties are a
  separate reporting category).
- Plain LM loss (mean CE over all positions). The engine's mask/dyadic/
  uniform-pull/action-head extensions are position-wise muxes on the top
  error derived from boundary inputs, not saved state; they are out of scope here and
  declared.
"""
from __future__ import annotations

from typing import Any

import torch

from .packed_tape import PackedRouteTape
from .primitives import embedding_gather_vjp, gate_vjp, pow2_norm_vjp

SHIFT_LO, SHIFT_HI = -8, 8            # the model's clamp (pc_convrnn:245)


def model_fingerprint(model, tau: float = 0.0) -> dict[str, Any]:
    """Configuration binding for tape serialization (format v2)."""
    layer0 = model.layers[0]
    return dict(
        arch="morphssm",
        d_model=int(model.embed.weight.shape[1]),
        vocab=int(model.embed.weight.shape[0]),
        n_layers=len(model.layers),
        kernel=int(layer0.morph_conv.kernel_size),
        dilations=[int(layer.morph_conv.dilation) for layer in model.layers],
        head_k=int(model.output_head.k),
        tau=float(tau),
        tie_rule="model-produced",
        dtype=str(model.embed.weight.dtype),
    )


def _assert_hard(model) -> None:
    for i, layer in enumerate(model.layers):
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


def serialize(tape: PackedRouteTape, model, tau: float = 0.0) -> bytes:
    return tape.to_bytes(fingerprint=model_fingerprint(model, tau))


def deserialize(blob: bytes, model, tau: float = 0.0) -> PackedRouteTape:
    return PackedRouteTape.from_bytes(
        blob, expected_fingerprint=model_fingerprint(model, tau))


@torch.no_grad()
def replay(tape: PackedRouteTape, model, logits: torch.Tensor,
           targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """The exact `cascade_grads` reverse sweep from the tape + boundary
    state. Returns {parameter_name: grad} without touching `.grad`."""
    _assert_hard(model)
    B, T, V = logits.shape
    D = int(model.embed.weight.shape[1])
    R = B * T
    dt = logits.dtype
    grads: dict[str, torch.Tensor] = {}

    # ---- top error (mirrors _top_error, plain-LM path) ---------------------
    eps = torch.softmax(logits.reshape(-1, V).float(), dim=-1)
    eps.scatter_add_(1, targets.reshape(-1, 1),
                     -torch.ones(R, 1, device=eps.device))
    eps = (eps / R).to(dt)
    head = model.output_head
    k = head.k
    idx = tape.read("head_winner")                       # (R, V, k)
    e_exp = eps.unsqueeze(-1).expand(R, V, k)
    e_x = torch.zeros(R, D, dtype=dt)
    e_x.scatter_add_(1, idx.reshape(R, V * k), e_exp.reshape(R, V * k))
    flat = (torch.arange(V).view(1, V, 1) * D + idx)
    wg = torch.zeros(V * D, dtype=dt)
    wg.index_add_(0, flat.reshape(-1), e_exp.reshape(-1))
    grads["output_head.weight"] = wg.view(V, D)
    grads["output_head.bias"] = eps.sum(0)
    e = e_x.reshape(B, T, D)
    e, bg = pow2_norm_vjp(e, tape.read("f_final_shift"))
    grads["ln_final.beta"] = bg

    # ---- layer sweep (mirrors _layer_adjoint, hard path) -------------------
    for li in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[li]
        e2, bg2 = pow2_norm_vjp(e, tape.read(f"layer{li}.f2_shift"))
        grads[f"layers.{li}.post_norm.beta"] = bg2
        e_res = e2
        # proj adjoint (hard winner scatter; O == I == D)
        pe = e2.reshape(R, D)
        pidx = tape.read(f"layer{li}.proj_winner")       # (R, D)
        e_proj = torch.zeros(R, D, dtype=dt)
        e_proj.scatter_add_(1, pidx, pe)
        pflat = (torch.arange(D).unsqueeze(0) * D + pidx)
        pwg = torch.zeros(D * D, dtype=dt)
        pwg.index_add_(0, pflat.reshape(-1), pe.reshape(-1))
        grads[f"layers.{li}.out_proj.weight"] = pwg.view(D, D)
        grads[f"layers.{li}.out_proj.bias"] = pe.sum(0)
        e_v = e_proj.reshape(B, T, D)
        try:
            gate = tape.read(f"layer{li}.drop_gate")
        except KeyError:
            gate = None
        if gate is not None:
            e_v = gate_vjp(e_v, gate.bool())
        e_u = e_v.transpose(1, 2)                        # (B, C, T)
        # conv adjoint (hard mux)
        conv = layer.morph_conv
        K, dil = int(conv.kernel_size), int(conv.dilation)
        route = tape.read(f"layer{li}.conv_winner")      # (B, C, T)
        e_in_c = torch.zeros_like(e_u)
        cwg = torch.zeros(D, K, dtype=dt)
        zero = torch.zeros((), dtype=dt)
        for j in range(K):
            contrib = torch.where(route == j, e_u, zero)
            cwg[:, j] = contrib.sum(dim=(0, 2))
            if j * dil < T:
                e_in_c[:, :, :T - j * dil] += contrib[:, :, j * dil:]
        grads[f"layers.{li}.morph_conv.weight"] = cwg
        grads[f"layers.{li}.morph_conv.bias"] = e_u.sum(dim=(0, 2))
        e_xn = e_in_c.transpose(1, 2)
        e1, bg1 = pow2_norm_vjp(e_xn, tape.read(f"layer{li}.f1_shift"))
        grads[f"layers.{li}.norm.beta"] = bg1
        e = e_res + e1

    # ---- embedding ---------------------------------------------------------
    try:
        m0 = tape.read("embed_drop_gate")
    except KeyError:
        m0 = None
    if m0 is not None:
        e = gate_vjp(e, m0.bool())
    grads["embed.weight"] = embedding_gather_vjp(
        e, tape.read("token_ids"), int(model.embed.weight.shape[0]))
    return grads
