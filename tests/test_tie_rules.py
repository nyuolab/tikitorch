"""Deterministic selected-subgradient and route-margin tests."""

import pytest
import torch

from tikitorch.autodiff import (
    additive_extreme,
    additive_extreme_vjp,
    morphological_conv1d,
    routing_margin,
    stable_topk,
)


@pytest.mark.parametrize("mode", ["max", "min"])
def test_exact_ties_choose_the_first_candidate_and_selected_subgradient(mode):
    x = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)
    weight = torch.zeros(2, 3, requires_grad=True)
    upstream = torch.tensor([[3.0, -5.0]])

    output, route = additive_extreme(x, weight, mode=mode)
    assert route.tolist() == [[0, 0]]
    (output * upstream).sum().backward()
    grad_x, grad_weight, _ = additive_extreme_vjp(upstream, route, input_features=3)
    torch.testing.assert_close(x.grad, grad_x, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, grad_weight, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, torch.tensor([[-2.0, 0.0, 0.0]]), rtol=0, atol=0)


@pytest.mark.parametrize("largest", [True, False])
def test_stable_topk_ties_keep_lowest_indices(largest):
    values = torch.tensor([[4.0, 4.0, 4.0, 4.0]])
    selected, route = stable_topk(values, 3, largest=largest)
    assert route.tolist() == [[0, 1, 2]]
    torch.testing.assert_close(selected, torch.tensor([[4.0, 4.0, 4.0]]))
    for _ in range(5):
        assert torch.equal(stable_topk(values, 3, largest=largest)[1], route)


@pytest.mark.parametrize("mode", ["max", "min"])
def test_morphological_ties_choose_kernel_slot_zero(mode):
    x = torch.zeros(2, 3, 7)
    weight = torch.zeros(3, 4)
    _, route = morphological_conv1d(x, weight, dilation=2, mode=mode)
    assert torch.equal(route, torch.zeros_like(route))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_near_ties_are_resolved_after_dtype_rounding(dtype):
    eps = torch.finfo(dtype).eps
    rounded_tie = torch.tensor([1.0, 1.0 + eps / 4], dtype=dtype)
    resolved = torch.tensor([1.0, 1.0 + 2 * eps], dtype=dtype)

    tied_value = rounded_tie.view(1, 1, 2)
    resolved_value = resolved.view(1, 1, 2)
    _, tie_route = additive_extreme(
        rounded_tie.view(1, 2), torch.zeros(1, 2, dtype=dtype), mode="max"
    )
    _, resolved_route = additive_extreme(
        resolved.view(1, 2), torch.zeros(1, 2, dtype=dtype), mode="max"
    )
    assert tie_route.item() == 0
    assert resolved_route.item() == 1
    assert routing_margin(tied_value, mode="max").item() == 0.0
    assert routing_margin(resolved_value, mode="max").item() > 0.0


def test_minimum_near_tie_has_the_same_first_index_contract():
    eps = torch.finfo(torch.float32).eps
    rounded_tie = torch.tensor([[1.0, 1.0 - eps / 4]], dtype=torch.float32)
    resolved = torch.tensor([[1.0, 1.0 - 2 * eps]], dtype=torch.float32)
    weight = torch.zeros(1, 2)
    assert additive_extreme(rounded_tie, weight, mode="min")[1].item() == 0
    assert additive_extreme(resolved, weight, mode="min")[1].item() == 1
