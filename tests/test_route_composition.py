"""Composition, fan-out, concatenation, and residual-adjoint tests."""

import torch

from tikitorch.autodiff import (
    additive_extreme,
    additive_extreme_vjp,
    compose_vjp,
    concatenation_vjp,
    fanout_vjp,
    residual_add_vjp,
)


def test_two_layer_routes_compose_to_exact_state_and_parameter_vjps():
    for seed in range(20):
        generator = torch.Generator().manual_seed(6_000 + seed)
        x = torch.randn(3, 5, generator=generator, dtype=torch.double).requires_grad_()
        w1 = torch.randn(4, 5, generator=generator, dtype=torch.double).requires_grad_()
        b1 = torch.randn(4, generator=generator, dtype=torch.double).requires_grad_()
        w2 = torch.randn(2, 4, generator=generator, dtype=torch.double).requires_grad_()
        b2 = torch.randn(2, generator=generator, dtype=torch.double).requires_grad_()
        upstream = torch.randn(3, 2, generator=generator, dtype=torch.double)

        hidden, route1 = additive_extreme(x, w1, b1, mode="max")
        output, route2 = additive_extreme(hidden, w2, b2, mode="min")
        (output * upstream).sum().backward()

        grad_hidden, grad_w2, grad_b2 = additive_extreme_vjp(upstream, route2, 4)
        grad_x, grad_w1, grad_b1 = additive_extreme_vjp(grad_hidden, route1, 5)
        torch.testing.assert_close(grad_x, x.grad, rtol=0, atol=0)
        torch.testing.assert_close(grad_w1, w1.grad, rtol=0, atol=0)
        torch.testing.assert_close(grad_b1, b1.grad, rtol=0, atol=0)
        torch.testing.assert_close(grad_w2, w2.grad, rtol=0, atol=0)
        torch.testing.assert_close(grad_b2, b2.grad, rtol=0, atol=0)

        chained = compose_vjp(
            upstream,
            [
                lambda adjoint: additive_extreme_vjp(adjoint, route2, 4)[0],
                lambda adjoint: additive_extreme_vjp(adjoint, route1, 5)[0],
            ],
        )
        torch.testing.assert_close(chained, grad_x, rtol=0, atol=0)


def test_fanout_and_residual_accumulate_all_reverse_paths():
    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.double, requires_grad=True)
    w_a = torch.tensor(
        [[0.2, -0.5, 1.0], [-1.0, 0.1, 0.3], [0.7, -0.2, -0.4]],
        dtype=torch.double,
        requires_grad=True,
    )
    w_b = torch.tensor(
        [[-0.4, 0.8, 0.0], [1.2, -0.3, 0.1], [-0.1, 0.4, 0.9]],
        dtype=torch.double,
        requires_grad=True,
    )
    upstream = torch.tensor([[2.0, -3.0, 5.0]], dtype=torch.double)

    branch_a, route_a = additive_extreme(x, w_a, mode="max")
    branch_b, route_b = additive_extreme(x, w_b, mode="min")
    output = x + branch_a + branch_b
    (output * upstream).sum().backward()

    direct, branch_seed = residual_add_vjp(upstream)
    # The second addition copies branch_seed to both routed branches.
    seed_a, seed_b = residual_add_vjp(branch_seed)
    grad_a, grad_w_a, _ = additive_extreme_vjp(seed_a, route_a, 3)
    grad_b, grad_w_b, _ = additive_extreme_vjp(seed_b, route_b, 3)
    reconstructed = fanout_vjp(direct, grad_a, grad_b)

    torch.testing.assert_close(reconstructed, x.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_w_a, w_a.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_w_b, w_b.grad, rtol=0, atol=0)
    # At least one input receives contributions from multiple reverse paths;
    # the global Jacobian is therefore not a partial permutation.
    assert torch.count_nonzero(reconstructed - direct) > 0


def test_concatenation_vjp_splits_static_parent_ranges():
    left = torch.randn(2, 3, requires_grad=True)
    middle = torch.randn(2, 1, requires_grad=True)
    right = torch.randn(2, 4, requires_grad=True)
    output = torch.cat((left, middle, right), dim=-1)
    upstream = torch.arange(16, dtype=output.dtype).reshape_as(output)
    (output * upstream).sum().backward()

    split = concatenation_vjp(upstream, (3, 1, 4))
    torch.testing.assert_close(split[0], left.grad, rtol=0, atol=0)
    torch.testing.assert_close(split[1], middle.grad, rtol=0, atol=0)
    torch.testing.assert_close(split[2], right.grad, rtol=0, atol=0)
