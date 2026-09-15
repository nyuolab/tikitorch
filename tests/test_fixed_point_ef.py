"""Synthetic integer recurrence tests; no model, corpus, or native execution."""
import random

import numpy as np
import pytest
import torch

from tikitorch.optim.fixed_point import fixed_point_ef_step


def inputs(weights, remainder, gradient):
    return (torch.tensor(weights, dtype=torch.int16),
            torch.tensor(remainder, dtype=torch.int32),
            torch.tensor(gradient, dtype=torch.int32))


def scalar_oracle(weight, remainder, gradient, k):
    total = remainder + gradient
    quotient = abs(total) // (1 << k)
    if total < 0:
        quotient = -quotient
    return weight - quotient, total - quotient * (1 << k)


@pytest.mark.parametrize("k", [0, 1, 2, 8, 18, 30])
def test_scalar_oracle_and_exact_integer_conservation(k):
    rng = random.Random(111 + k)
    denominator = 1 << k
    weights = [rng.randint(-1000, 1000) for _ in range(257)]
    remainders = [rng.randint(-denominator + 1, denominator - 1) for _ in weights]
    limit = min(denominator * 100, (1 << 31) - denominator)
    gradients = [rng.randint(-limit, limit) for _ in weights]
    w, r, g = inputs(weights, remainders, gradients)
    next_w, next_r = fixed_point_ef_step(w, r, g, k=k)
    expected = [scalar_oracle(*values, k) for values in zip(weights, remainders, gradients)]
    assert next_w.tolist() == [value[0] for value in expected]
    assert next_r.tolist() == [value[1] for value in expected]
    assert next_w.dtype == torch.int16 and next_r.dtype == torch.int32
    assert torch.all(next_r.to(torch.int64).abs() < denominator)
    # Int64 identities do not rely on floating-point tolerances.
    assert torch.equal((w.long() - next_w.long()) * denominator + next_r.long(),
                       r.long() + g.long())
    assert torch.equal(next_w.long() * denominator - next_r.long(),
                       w.long() * denominator - r.long() - g.long())


@pytest.mark.parametrize("weight,remainder,gradient,k", [
    (0, 0, -1, 2), (0, -3, -2, 2), (0, 3, 2, 2), (0, -3, 4, 2),
    (0, 3, -4, 2), (0, 0, -4, 2), (0, 0, -(1 << 31), 30),
    (0, 0, (1 << 31) - 1, 30), (0, 0, -(1 << 31), 16),
    (-1, 0, -(1 << 31), 16), (32767, 0, 32767, 0), (-32768, 0, -32768, 0),
])
def test_boundaries_match_frozen_native_scalar_recurrence(weight, remainder, gradient, k):
    # Importing the frozen oracle does not build/load/run its native adapter.
    from manuscript.tools.fpga_lm_efsgd import scalar_update

    w, r, g = inputs(weight, remainder, gradient)
    expected = scalar_oracle(weight, remainder, gradient, k)
    if not -32768 <= expected[0] <= 32767:
        with pytest.raises(OverflowError):
            fixed_point_ef_step(w, r, g, k=k)
        with pytest.raises(OverflowError):
            scalar_update(weight, remainder, gradient, k)
    else:
        next_w, next_r = fixed_point_ef_step(w, r, g, k=k)
        assert (next_w.item(), next_r.item()) == expected
        assert expected == scalar_update(weight, remainder, gradient, k)[:2]


@pytest.mark.parametrize("k", [0, 1, 18, 30])
def test_array_parity_with_frozen_diagnostic_helper(k):
    from manuscript.tools.fpga_lm_numeric_diagnostic import ef_rounding

    w, r, g = inputs([[100, -100], [0, 12]], [[0, 0], [1 if k else 0, 0]],
                     [[17, -19], [-23, 42]])
    expected = ef_rounding(w.numpy(), r.numpy(), g.numpy(), k)
    next_w, next_r = fixed_point_ef_step(w, r, g, k=k)
    assert np.array_equal(next_w.numpy(), expected["weights"])
    assert np.array_equal(next_r.numpy(), expected["remainders"])


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3, 4)])
def test_shape_scalar_and_empty_support(shape):
    w = torch.zeros(shape, dtype=torch.int16)
    r = torch.zeros(shape, dtype=torch.int32)
    g = torch.ones(shape, dtype=torch.int32)
    next_w, next_r = fixed_point_ef_step(w, r, g, k=2)
    assert next_w.shape == next_r.shape == w.shape
    assert torch.equal(next_w, w) and torch.equal(next_r, g)
    assert next_w is not w and next_r is not r and next_r is not g
    assert next_w._base is None and next_r._base is None


def test_noncontiguous_inputs_and_independent_outputs():
    w = torch.arange(24, dtype=torch.int16).reshape(4, 6)[:, ::2].T
    r = torch.ones((3, 4), dtype=torch.int32).T.contiguous().T
    g = torch.full((3, 4), -7, dtype=torch.int32)
    assert not w.is_contiguous() and not r.is_contiguous()
    originals = [value.clone() for value in (w, r, g)]
    next_w, next_r = fixed_point_ef_step(w, r, g, k=2)
    assert torch.equal(next_w, w + 1)
    assert torch.equal(next_r, torch.full_like(r, -2))
    for output in (next_w, next_r):
        for value in (w, r, g):
            assert output.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
    assert next_w.untyped_storage().data_ptr() != next_r.untyped_storage().data_ptr()
    next_w.fill_(99)
    next_r.fill_(-99)
    for value, before in zip((w, r, g), originals):
        assert torch.equal(value, before)


def test_remainder_and_gradient_may_alias_without_mutation():
    w = torch.tensor([7, -7], dtype=torch.int16)
    shared = torch.tensor([3, -3], dtype=torch.int32)
    next_w, next_r = fixed_point_ef_step(w, shared, shared, k=2)
    assert next_w.tolist() == [6, -6] and next_r.tolist() == [2, -2]
    assert shared.tolist() == [3, -3]


def test_persistent_resume_is_exact_and_remainders_are_not_optional():
    rng = random.Random(812)
    gradients = [torch.tensor([rng.randint(-91, 91) for _ in range(11)], dtype=torch.int32)
                 for _ in range(40)]
    initial = (torch.zeros(11, dtype=torch.int16), torch.zeros(11, dtype=torch.int32))
    state = initial
    for gradient in gradients[:17]:
        state = fixed_point_ef_step(*state, gradient, k=7)
    checkpoint = tuple(value.clone() for value in state)
    assert torch.count_nonzero(checkpoint[1]) > 0
    uninterrupted = state
    resumed = checkpoint
    cleared = (checkpoint[0].clone(), torch.zeros_like(checkpoint[1]))
    for gradient in gradients[17:]:
        uninterrupted = fixed_point_ef_step(*uninterrupted, gradient, k=7)
        resumed = fixed_point_ef_step(*resumed, gradient, k=7)
        cleared = fixed_point_ef_step(*cleared, gradient, k=7)
    assert all(torch.equal(a, b) for a, b in zip(uninterrupted, resumed))
    assert any(not torch.equal(a, b) for a, b in zip(uninterrupted, cleared))
    total = sum((gradient.long() for gradient in gradients), torch.zeros(11, dtype=torch.int64))
    assert torch.equal(resumed[0].long() * 128 - resumed[1].long(), -total)
    assert torch.count_nonzero(initial[0]) == torch.count_nonzero(initial[1]) == 0


@pytest.mark.parametrize("k", [True, False, 1.0, np.int64(1), None, "1"])
def test_shift_must_be_literal_integer(k):
    with pytest.raises(TypeError, match="literal Python integer"):
        fixed_point_ef_step(*inputs([0], [0], [0]), k=k)


@pytest.mark.parametrize("k", [-1, 31, 100000])
def test_shift_range(k):
    with pytest.raises(ValueError, match="in \\[0, 30\\]"):
        fixed_point_ef_step(*inputs([0], [0], [0]), k=k)


def test_shift_is_required_and_keyword_only():
    tensors = inputs([0], [0], [0])
    with pytest.raises(TypeError):
        fixed_point_ef_step(*tensors)
    with pytest.raises(TypeError):
        fixed_point_ef_step(*tensors, 2)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("replacement", [[0], None, 0])
def test_all_arguments_must_be_tensors(index, replacement):
    values = list(inputs([0], [0], [0]))
    values[index] = replacement
    with pytest.raises(TypeError, match="torch.Tensor"):
        fixed_point_ef_step(*values, k=2)


@pytest.mark.parametrize("index,dtype", [
    (0, torch.int32), (0, torch.int64), (0, torch.float32), (0, torch.bool),
    (1, torch.int16), (1, torch.int64), (1, torch.float64),
    (2, torch.int16), (2, torch.int64), (2, torch.float32), (2, torch.bool),
])
def test_exact_dtype_contract(index, dtype):
    values = list(inputs([0], [0], [0]))
    values[index] = values[index].to(dtype)
    with pytest.raises(TypeError, match="dtype"):
        fixed_point_ef_step(*values, k=2)


@pytest.mark.parametrize("index", range(3))
def test_shape_mismatch_is_not_broadcast(index):
    values = list(inputs([0, 0], [0, 0], [0, 0]))
    values[index] = values[index][:1]
    with pytest.raises(ValueError, match="identical shapes"):
        fixed_point_ef_step(*values, k=2)


@pytest.mark.parametrize("index", range(3))
def test_meta_tensors_fail_before_data_access(index):
    values = list(inputs([0], [0], [0]))
    values[index] = torch.empty((1,), dtype=values[index].dtype, device="meta")
    with pytest.raises(ValueError, match="CPU"):
        fixed_point_ef_step(*values, k=2)


@pytest.mark.parametrize("index", range(3))
def test_sparse_tensors_are_rejected(index):
    values = list(inputs([0], [0], [0]))
    values[index] = values[index].to_sparse()
    with pytest.raises(ValueError, match="dense strided"):
        fixed_point_ef_step(*values, k=2)


@pytest.mark.parametrize("k", [0, 1, 18, 30])
@pytest.mark.parametrize("sign", [-1, 1])
def test_canonical_remainder_boundary(k, sign):
    with pytest.raises(ValueError, match="canonical"):
        fixed_point_ef_step(*inputs([0], [sign * (1 << k)], [0]), k=k)


@pytest.mark.parametrize("remainder,gradient,k", [
    (1, (1 << 31) - 1, 1), (-1, -(1 << 31), 1),
    ((1 << 30) - 1, (1 << 31) - 1, 30), (-(1 << 30) + 1, -(1 << 31), 30),
])
def test_combined_int32_overflow_is_transactional(remainder, gradient, k):
    tensors = inputs([0, 0], [0, remainder], [1, gradient])
    snapshots = [value.clone() for value in tensors]
    with pytest.raises(OverflowError, match="int32"):
        fixed_point_ef_step(*tensors, k=k)
    assert all(torch.equal(value, before) for value, before in zip(tensors, snapshots))


@pytest.mark.parametrize("weight,gradient", [(32767, -4), (-32768, 4)])
def test_candidate_int16_overflow_is_transactional(weight, gradient):
    tensors = inputs([0, weight], [0, 0], [4, gradient])
    snapshots = [value.clone() for value in tensors]
    with pytest.raises(OverflowError, match="int16"):
        fixed_point_ef_step(*tensors, k=2)
    assert all(torch.equal(value, before) for value, before in zip(tensors, snapshots))


def test_boundary_weights_with_subunit_update_are_valid():
    next_w, next_r = fixed_point_ef_step(*inputs([32767, -32768], [0, 0], [-3, 3]), k=2)
    assert next_w.tolist() == [32767, -32768]
    assert next_r.tolist() == [-3, 3]
