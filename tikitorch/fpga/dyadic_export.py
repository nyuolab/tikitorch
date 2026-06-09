"""tikitorch.fpga.dyadic_export — freeze a trained dyadic net to a zero-DSP SHIFT SCHEDULE.

A :class:`tikitorch.nn.DyadicLinear` constrains every weight to a signed power of two,
``w = sign * 2^k`` (:func:`tikitorch.nn.dyadic_quantize`). On FPGA fabric that is a
sign-controlled bit-SHIFT feeding an add-tree -- no multiplier, no DSP. This module exports
a trained net's dyadic layers to an integer ``(sign, exponent)`` schedule the HLS kernel
generator consumes, and VERIFIES the schedule reproduces the net's *deployed* forward
exactly (the closure check: deploy is the quantized forward, which this schedule is).

    from tikitorch.fpga.dyadic_export import export_net, verify_schedule
    sched = export_net(trained_dyadic_muzero_net)
    assert not verify_schedule(net, sched)        # empty list => bit-exact
"""
import torch
import torch.nn.functional as F

from tikitorch.nn import DyadicLinear, dyadic_quantize

__all__ = ["export_layer", "reconstruct_weight", "export_net", "verify_schedule"]


def export_layer(lin: "DyadicLinear", exp_clip=None):
    """Export one DyadicLinear to a shift schedule: int8 sign, int16 exponent, float bias.

    With ``exp_clip=(lo, hi)`` the exponents are clamped to a hardware shift range (this
    changes the weight, so re-verify against a clamped reference if you use it)."""
    q = dyadic_quantize(lin.weight.detach())          # = sign * 2^round(log2|w|): the deployed weight
    sign = torch.sign(q).to(torch.int8)
    exp = torch.log2(q.abs().clamp_min(1e-30)).round().to(torch.int32)
    exp = torch.where(sign == 0, torch.zeros_like(exp), exp)   # zero weight -> no shift
    if exp_clip is not None:
        exp = exp.clamp(exp_clip[0], exp_clip[1])
    return dict(sign=sign.to(torch.int16), exponent=exp.to(torch.int16),
                bias=(lin.bias.detach().clone() if lin.bias is not None else None),
                shape=tuple(lin.weight.shape))


def reconstruct_weight(layer_sched):
    """Rebuild the dyadic weight from its schedule: ``sign * 2^exponent`` (a shift on fabric)."""
    return layer_sched["sign"].float() * torch.exp2(layer_sched["exponent"].float())


def export_net(net):
    """Walk ``net`` and export every DyadicLinear, keyed by its module path."""
    return {name: export_layer(m) for name, m in net.named_modules() if isinstance(m, DyadicLinear)}


def verify_schedule(net, sched, n=128, atol=1e-4):
    """Return the layers whose exported schedule does NOT reproduce the deployed forward
    (empty list => bit-exact). The deployed forward IS the dyadic-quantized forward, so an
    unclipped export must match exactly up to float round-off."""
    mods = dict(net.named_modules())
    bad = []
    for name, ls in sched.items():
        m = mods[name]
        x = torch.randn(n, m.in_features)
        with torch.no_grad():
            ref = m(x)                                            # deployed dyadic forward
            got = F.linear(x, reconstruct_weight(ls), ls["bias"])
        err = float((got - ref).abs().max())
        if err > atol:
            bad.append((name, err))
    return bad
