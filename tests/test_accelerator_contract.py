"""Opt-in accelerator preflight; an explicitly requested GPU run cannot skip."""

import os

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("TIKITORCH_REQUIRE_CUDA") != "1",
    reason="explicit GPU validation only (TIKITORCH_REQUIRE_CUDA=1)",
)
def test_requested_cuda_is_available_and_executes_backward():
    assert torch.cuda.is_available(), "GPU validation requested but CUDA is unavailable"
    x = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    x.square().sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(x.grad, torch.tensor([2.0, 4.0], device="cuda"))


@pytest.mark.skipif(
    os.environ.get("TIKITORCH_REQUIRE_COMPILE") != "1",
    reason="explicit compiler validation only (TIKITORCH_REQUIRE_COMPILE=1)",
)
def test_requested_inductor_executes_fullgraph_forward_and_backward():
    assert hasattr(torch, "compile"), "compiler validation requested but torch.compile is absent"
    device = "cuda" if os.environ.get("TIKITORCH_REQUIRE_CUDA") == "1" else "cpu"
    if device == "cuda":
        assert torch.cuda.is_available(), "compiled CUDA validation requested without CUDA"
    from torch import _dynamo

    def kernel(x):
        return (x.square() + x).sum()

    # A missing compiler or compilation error must fail, never fall back to eager.
    with _dynamo.config.patch(suppress_errors=False):
        compiled = torch.compile(kernel, backend="inductor", fullgraph=True)
        x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        result = compiled(x)
        result.backward()
        torch.testing.assert_close(result, torch.tensor(8.0, device=device))
        torch.testing.assert_close(x.grad, torch.tensor([3.0, 5.0], device=device))
        if device == "cuda":
            torch.cuda.synchronize()
