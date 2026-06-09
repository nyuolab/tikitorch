"""Direct unit tests for the dyadic (signed-power-of-two) multiplier-free primitives:
dyadic_quantize, DyadicLinear, DyadicTropicalProjection.

These are the zero-DSP cores behind the §2a oblique fix. Closure for them is a STATIC
weight-value property (weights are exact powers of two -> shift+add), recognized
structurally by assert_closed (the _zero_dsp_layer marker) and verified numerically by
fpga.verify_schedule (see test_fpga) — NOT by the runtime op-tracer, which sees the
float-sim matmul (that documented policy is pinned below).
"""
import pytest
import torch

import tikitorch as tt
import tikitorch.nn as ttnn
from tikitorch.nn import dyadic_quantize


def test_dyadic_quantize_edge_cases_and_ste_grad():
    w = torch.tensor([-3.0, -0.3, 0.0, 1.0, 1e-15], requires_grad=True)
    q = dyadic_quantize(w)
    # nearest SIGNED power of two; 0 -> 0; tiny magnitude clamps (>=1e-12) -> 2^-40, never inf/NaN
    assert torch.allclose(q.detach(), torch.tensor([-4.0, -0.25, 0.0, 1.0, 2.0 ** -40])), q.detach()
    nz = q.detach()[q.detach() != 0].abs()
    assert torch.allclose(torch.log2(nz), torch.log2(nz).round())   # every nonzero result is a pure pow2
    assert torch.isfinite(q.detach()).all()
    # straight-through estimator: gradient is identity (1) everywhere
    q.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_dyadic_linear_weights_pow2_and_marked():
    lin = ttnn.DyadicLinear(8, 5)
    qw = dyadic_quantize(lin.weight).detach().abs()
    nz = qw[qw > 0]
    assert torch.allclose(torch.log2(nz), torch.log2(nz).round())   # weights snap to powers of two
    assert getattr(lin, "_zero_dsp_layer", False) is True            # declared zero-DSP for assert_closed
    assert lin(torch.randn(4, 8)).shape == (4, 5)


def test_dyadic_projection_oblique_and_drop_in_shape():
    proj = ttnn.DyadicTropicalProjection(16, 16)
    assert proj(torch.randn(3, 16)).shape == (3, 16)                 # drop-in for TropicalProjection
    # OBLIQUE: input gradient routes to MANY inputs, unlike the axis-aligned max-plus core (~1)
    x = torch.randn(1, 16, requires_grad=True)
    proj(x).sum().backward()
    assert int((x.grad.abs() > 1e-9).sum()) > 1


def test_dyadic_structural_closed_but_runtime_sees_matmul():
    # Documented policy: assert_closed (structural) treats DyadicLinear as a zero-DSP shift
    # layer (no violation); the runtime op-tracer still flags the float-sim addmm as a
    # dense-multiply. Dyadic closure is the static weight property, checked by
    # fpga.verify_schedule, not by the forward trace.
    proj = ttnn.DyadicTropicalProjection(8, 8)
    tt.assert_closed(proj)                       # structural: passes
    with pytest.raises(tt.ClosureViolation):     # runtime: the matmul is seen as a multiply
        with tt.assert_no_multiply():
            proj(torch.randn(2, 8))
