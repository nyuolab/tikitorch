"""State-VJP validation for theorem-bearing route primitives."""

import pytest
import torch

from tikitorch.autodiff import (
    additive_extreme,
    additive_extreme_vjp,
    morphological_conv1d,
    morphological_conv1d_vjp,
    routing_margin,
)


def _finite_difference_scalar(function, x, index, eps=1e-6):
    plus = x.detach().clone()
    minus = x.detach().clone()
    plus[index] += eps
    minus[index] -= eps
    return (function(plus) - function(minus)) / (2.0 * eps)


@pytest.mark.parametrize("mode", ["max", "min"])
def test_additive_route_state_vjp_matches_autograd_and_finite_difference(mode):
    x = torch.tensor(
        [[0.1, -1.2, 2.5, 0.7], [-0.9, 0.4, 1.1, 3.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    weight = torch.tensor(
        [[0.3, -0.4, 0.1, 1.2], [-0.8, 0.9, 0.2, -0.3], [1.3, -0.7, 0.6, 0.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    upstream = torch.tensor([[0.5, -1.25, 2.0], [1.75, 0.25, -0.5]], dtype=torch.double)

    combined = x.unsqueeze(-2) + weight
    assert torch.all(routing_margin(combined, mode=mode) > 1e-3)
    value, route = additive_extreme(x, weight, mode=mode)
    (value * upstream).sum().backward()
    route_grad_x, _, _ = additive_extreme_vjp(upstream, route, x.shape[-1])
    torch.testing.assert_close(route_grad_x, x.grad, rtol=0, atol=0)

    def contracted(candidate):
        return (additive_extreme(candidate, weight.detach(), mode=mode)[0] * upstream).sum()

    fd = torch.empty_like(x)
    for row in range(x.shape[0]):
        for feature in range(x.shape[1]):
            fd[row, feature] = _finite_difference_scalar(contracted, x, (row, feature))
    torch.testing.assert_close(route_grad_x, fd, rtol=1e-7, atol=1e-8)

    # Registered randomized coverage: twenty independent away-from-tie
    # executions per extremum type, in addition to the finite-difference case.
    for seed in range(20):
        generator = torch.Generator().manual_seed(1_000 + seed)
        random_x = torch.randn(3, 6, generator=generator, dtype=torch.double).requires_grad_()
        random_weight = torch.randn(4, 6, generator=generator, dtype=torch.double).requires_grad_()
        random_upstream = torch.randn(3, 4, generator=generator, dtype=torch.double)
        random_output, random_route = additive_extreme(
            random_x, random_weight, mode=mode
        )
        (random_output * random_upstream).sum().backward()
        random_route_grad, _, _ = additive_extreme_vjp(
            random_upstream, random_route, input_features=6
        )
        torch.testing.assert_close(random_route_grad, random_x.grad, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("mode,sign", [("max", 1.0), ("min", -1.0)])
def test_morphological_conv_state_vjp_handles_overlap(mode, sign):
    # The large central input wins three overlapping windows (slots 0, 1, 2).
    x = (sign * torch.tensor([[[-4.0, -3.0, 10.0, -2.0, -1.0]]], dtype=torch.double)).requires_grad_()
    weight = (sign * torch.tensor([[0.0, -0.1, -0.2]], dtype=torch.double)).requires_grad_()
    upstream = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=torch.double)

    value, route = morphological_conv1d(x, weight, dilation=1, mode=mode)
    assert route.tolist() == [[[1, 2, 0, 1, 2]]]
    (value * upstream).sum().backward()
    route_grad_x, _, _ = morphological_conv1d_vjp(
        upstream, route, kernel_size=3, dilation=1
    )
    torch.testing.assert_close(route_grad_x, x.grad, rtol=0, atol=0)
    assert route_grad_x[0, 0, 2].item() == 12.0

    def contracted(candidate):
        output, _ = morphological_conv1d(
            candidate, weight.detach(), dilation=1, mode=mode
        )
        return (output * upstream).sum()

    fd = torch.empty_like(x)
    for time in range(x.shape[-1]):
        fd[0, 0, time] = _finite_difference_scalar(contracted, x, (0, 0, time))
    torch.testing.assert_close(route_grad_x, fd, rtol=1e-7, atol=1e-8)

    for seed in range(20):
        generator = torch.Generator().manual_seed(2_000 + seed)
        random_x = torch.randn(2, 2, 11, generator=generator, dtype=torch.double).requires_grad_()
        random_weight = torch.randn(2, 3, generator=generator, dtype=torch.double).requires_grad_()
        random_upstream = torch.randn(2, 2, 11, generator=generator, dtype=torch.double)
        random_output, random_route = morphological_conv1d(
            random_x, random_weight, dilation=2, mode=mode
        )
        (random_output * random_upstream).sum().backward()
        random_route_grad, _, _ = morphological_conv1d_vjp(
            random_upstream, random_route, kernel_size=3, dilation=2
        )
        torch.testing.assert_close(
            random_route_grad, random_x.grad, rtol=1e-14, atol=1e-14
        )


def test_route_scatter_accumulates_multiple_outputs_exactly():
    upstream = torch.tensor([[2.0, -3.0, 5.0, 7.0]])
    route = torch.tensor([[1, 1, 1, 0]])
    grad_input, _, _ = additive_extreme_vjp(upstream, route, input_features=3)
    torch.testing.assert_close(grad_input, torch.tensor([[7.0, 4.0, 0.0]]), rtol=0, atol=0)
