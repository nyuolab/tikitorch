"""Dyadic forward is zero-DSP; dyadic *backward* is only zero-DSP under a sign-only rule.

This file is the executable record of an investigation. The dyadic fix
(:class:`tikitorch.nn.DyadicLinear`, signed-power-of-two weights) makes the FORWARD
multiplier-free — ``W·x`` is a sign+shift. The ``_zero_dsp_layer`` marker and the
structural :func:`assert_closed` therefore pass it. But the marker is FORWARD-scoped:
the autograd *backward* of a dyadic linear is NOT multiplier-free, because the
weight-gradient ``dL/dW = dL/dy ⊗ x`` multiplies two arbitrary tensors. TikiTorch's
runtime tracers were never fooled by this — they report the backward ``mm`` — but
nothing *gated* on them for the dyadic net, so the forward-only property could read as
a whole-model one.

These tests pin both halves so the gap cannot silently reopen:
  * ``test_autograd_dyadic_backward_is_not_zero_dsp`` — the gap: a normal autograd
    training step on a dyadic layer trips the closure guard (dense-multiply).
  * ``test_signonly_dyadic_rule_is_zero_dsp`` — the fix: the on-fabric training rule
    (forward shift, grad-input shift, weight-grad SIGN = sign-product, SignSGD update)
    is certified multiplier-free by ``assert_no_multiply``. This is the exact algebra
    of the HLS kernel forge_train_step_muzero_train.cpp.
"""
import torch
import torch.nn.functional as F

import tikitorch as tt
from tikitorch.audit import closure_report
from tikitorch.nn import DyadicLinear


def test_autograd_dyadic_backward_is_not_zero_dsp():
    """THE GAP. A dyadic layer is forward-zero-DSP, but training it with autograd is not:
    the weight-gradient is a real multiply. The closure tooling reports this truthfully."""
    lin = DyadicLinear(4, 3)
    x = torch.randn(1, 4)

    # the forward-scoped marker makes the STRUCTURAL check pass ...
    tt.assert_closed(lin)

    # ... but a full training step trips the runtime guard with the backward multiply.
    raised = False
    try:
        with tt.assert_no_multiply():
            F.l1_loss(lin(x), torch.zeros(1, 3)).backward()
    except tt.ClosureViolation as e:
        raised = True
        assert "multiply" in str(e)
    assert raised, "autograd dyadic backward should NOT be multiplier-free"

    # closure_report names it precisely: training is not zero-DSP, and a dense-multiply appears.
    rep = closure_report(DyadicLinear(4, 3), torch.randn(1, 4), torch.zeros(1, 3),
                         F.l1_loss, optimizer=None)
    assert rep["training_zero_dsp"] is False
    assert any(cat == "dense-multiply" for cat, _ in rep["training"])


def _shift(x, exp):
    """sgnless arithmetic shift by a per-element signed exponent tensor — a bit-shift, no multiply."""
    return torch.where(exp >= 0, torch.bitwise_left_shift(x, exp.clamp_min(0)),
                                 torch.bitwise_right_shift(x, (-exp).clamp_min(0)))


def test_signonly_dyadic_rule_is_zero_dsp():
    """THE FIX. The on-fabric training rule — forward via shift, grad-input via shift,
    weight-gradient via a 1-bit SIGN-product (never forming dL/dy ⊗ x), and a SignSGD
    update in the (sign, exponent) domain — is certified multiplier-free by the guard.
    Every op is shift / add / compare / int-sign; this is the algebra of the HLS
    training kernel, re-expressed in torch so assert_no_multiply can trace it."""
    torch.manual_seed(0)
    exp = torch.randint(-2, 3, (3, 4))           # a frozen dyadic schedule: w = sgn * 2^exp
    sgn = torch.randint(0, 2, (3, 4)) * 2 - 1
    x = torch.randint(-8, 9, (1, 4))             # int activations (batch=1, as the kernel)
    tgt = torch.tensor([[5, -3, 2]])

    with tt.assert_no_multiply():                # raises ClosureViolation on any hardware multiply
        # forward: y[o] = sum_i sgn[o,i] * (x[i] << exp[o,i])
        xe = x[:, None, :].expand(1, 3, 4)
        y = (_shift(xe, exp) * sgn).sum(dim=2)            # int-sign mul + add-tree
        gy = torch.sign(y - tgt)                          # L1 grad in {-1,0,+1}

        # grad wrt input: dL/dx = q(W)^T · gy  — multiplies by the *pow2* weights -> shift+add
        gye = gy[:, :, None].expand(1, 3, 4)
        gx = (_shift(gye, exp) * sgn).sum(dim=1)          # noqa: F841 (the closed grad-input)

        # weight-grad SIGN = sign(gy) (x) sign(x): a {-1,0,+1} outer product (int = logic, no DSP).
        gwsign = torch.sign(gy)[0][:, None] * torch.sign(x)[0][None, :]

        # SignSGD in the (sign, exponent) domain: nudge |w| up/down by 1 (int add) toward -grad.
        d = -gwsign
        agree = (d == sgn)
        exp = exp + torch.where(d == 0, torch.zeros_like(exp), torch.where(agree, 1, -1))
    # reached here without ClosureViolation => the rule is multiplier-free per TikiTorch's tracer.
    assert exp.dtype == torch.int64
