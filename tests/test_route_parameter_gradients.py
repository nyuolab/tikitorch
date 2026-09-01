"""Parameter-VJP sufficiency tests and explicit non-sufficient boundaries."""

import pytest
import torch
import torch.nn.functional as F

from tikitorch.autodiff import (
    additive_extreme,
    additive_extreme_vjp,
    morphological_conv1d,
    morphological_conv1d_vjp,
    topk_additive,
    topk_additive_vjp,
)


@pytest.mark.parametrize("mode", ["max", "min"])
def test_additive_parameter_vjp_matches_autograd_without_input_values(mode):
    for seed in range(20):
        generator = torch.Generator().manual_seed(3_000 + seed)
        x = torch.randn(4, 5, generator=generator, dtype=torch.double).requires_grad_()
        weight = torch.randn(3, 5, generator=generator, dtype=torch.double).requires_grad_()
        bias = torch.randn(3, generator=generator, dtype=torch.double).requires_grad_()
        upstream = torch.randn(4, 3, generator=generator, dtype=torch.double)

        output, route = additive_extreme(x, weight, bias, mode=mode)
        (output * upstream).sum().backward()
        _, grad_weight, grad_bias = additive_extreme_vjp(upstream, route, input_features=5)
        # Framework reduction order may differ by a few ulps from the explicit
        # route accumulation even though the selected real-arithmetic VJP is equal.
        torch.testing.assert_close(grad_weight, weight.grad, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(grad_bias, bias.grad, rtol=1e-14, atol=1e-14)

        # A common scalar translation changes every retained activation but not
        # the route or selected additive-parameter Jacobian.
        translated = (x.detach() + 100.0).requires_grad_()
        weight_2 = weight.detach().clone().requires_grad_()
        bias_2 = bias.detach().clone().requires_grad_()
        output_2, route_2 = additive_extreme(translated, weight_2, bias_2, mode=mode)
        assert torch.equal(route, route_2)
        (output_2 * upstream).sum().backward()
        torch.testing.assert_close(weight_2.grad, grad_weight, rtol=0, atol=0)
        torch.testing.assert_close(bias_2.grad, grad_bias, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["max", "min"])
def test_stable_topk_parameter_vjp_matches_autograd(mode):
    x = torch.tensor(
        [[-1.0, 0.2, 2.0, 0.8], [1.1, -2.0, 0.4, 3.2]],
        dtype=torch.double,
        requires_grad=True,
    )
    weight = torch.tensor(
        [[0.1, 0.3, -0.2, 0.7], [-0.8, 0.5, 0.9, -0.1]],
        dtype=torch.double,
        requires_grad=True,
    )
    bias = torch.tensor([0.4, -0.6], dtype=torch.double, requires_grad=True)
    upstream = torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.double)

    output, route = topk_additive(x, weight, bias, k=2, mode=mode)
    (output * upstream).sum().backward()
    grad_x, grad_weight, grad_bias = topk_additive_vjp(upstream, route, input_features=4)
    torch.testing.assert_close(grad_x, x.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_weight, weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_bias, bias.grad, rtol=0, atol=0)

    for seed in range(20):
        generator = torch.Generator().manual_seed(4_000 + seed)
        random_x = torch.randn(3, 6, generator=generator, dtype=torch.double).requires_grad_()
        random_weight = torch.randn(4, 6, generator=generator, dtype=torch.double).requires_grad_()
        random_bias = torch.randn(4, generator=generator, dtype=torch.double).requires_grad_()
        random_upstream = torch.randn(3, 4, generator=generator, dtype=torch.double)
        random_output, random_route = topk_additive(
            random_x, random_weight, random_bias, k=3, mode=mode
        )
        (random_output * random_upstream).sum().backward()
        random_grads = topk_additive_vjp(random_upstream, random_route, 6)
        for actual, expected in zip(
            random_grads, (random_x.grad, random_weight.grad, random_bias.grad)
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("mode", ["max", "min"])
def test_morphological_parameter_vjp_matches_autograd(mode):
    for seed in range(20):
        generator = torch.Generator().manual_seed(5_000 + seed)
        x = torch.randn(2, 3, 9, generator=generator, dtype=torch.double).requires_grad_()
        weight = torch.randn(3, 4, generator=generator, dtype=torch.double).requires_grad_()
        bias = torch.randn(3, generator=generator, dtype=torch.double).requires_grad_()
        upstream = torch.randn(2, 3, 9, generator=generator, dtype=torch.double)

        output, route = morphological_conv1d(
            x, weight, bias, dilation=2, mode=mode, padding_value=0.0
        )
        (output * upstream).sum().backward()
        grad_x, grad_weight, grad_bias = morphological_conv1d_vjp(
            upstream, route, kernel_size=4, dilation=2
        )
        torch.testing.assert_close(grad_x, x.grad, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(grad_weight, weight.grad, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(grad_bias, bias.grad, rtol=1e-14, atol=1e-14)


def test_dense_linear_weight_gradient_is_not_route_sufficient():
    """An empty/constant route cannot distinguish dense-linear weight Jacobians."""

    weight = torch.tensor([[0.5, -0.25, 1.0], [-1.0, 0.75, 0.2]], dtype=torch.double)
    upstream = torch.tensor([[2.0, -3.0]], dtype=torch.double)
    x_a = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.double, requires_grad=True)
    x_b = torch.tensor([[3.0, 2.0, -1.0]], dtype=torch.double, requires_grad=True)

    def weight_gradient(x):
        w = weight.clone().requires_grad_()
        (F.linear(x, w) * upstream).sum().backward()
        return w.grad

    grad_a = weight_gradient(x_a)
    grad_b = weight_gradient(x_b)
    assert not torch.equal(grad_a, grad_b)
    torch.testing.assert_close(grad_a, upstream.transpose(0, 1) @ x_a.detach())
    torch.testing.assert_close(grad_b, upstream.transpose(0, 1) @ x_b.detach())

    # In contrast, the state Jacobian is the same fixed weight matrix.  This
    # isolates parameter-Jacobian insufficiency rather than a route change.
    torch.testing.assert_close(x_a.grad if x_a.grad is not None else upstream @ weight, upstream @ weight)
