"""Packed-route reconstruction, coding-bound, and bounded-retention tests."""

import pytest
import torch

from tikitorch.autodiff import (
    PackedRouteTape,
    RouteTapeRingBuffer,
    additive_extreme,
    additive_extreme_vjp,
    bits_for_cardinality,
)


def test_packed_routes_reconstruct_state_and_parameter_vjps_without_activations():
    torch.manual_seed(101)
    x = torch.randn(6, 7, dtype=torch.double, requires_grad=True)
    weight = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
    bias = torch.randn(5, dtype=torch.double, requires_grad=True)
    upstream = torch.randn(6, 5, dtype=torch.double)
    output, route = additive_extreme(x, weight, bias, mode="max")
    (output * upstream).sum().backward()
    expected = (x.grad.clone(), weight.grad.clone(), bias.grad.clone())

    tape = PackedRouteTape()
    field = tape.append("layer0.winner", route, cardinality=7)
    tape.append_mask("dropout", torch.tensor([[True, False, True], [False, True, False]]))
    assert field.bit_width == 3
    assert tape.bit_length == route.numel() * 3 + 6
    assert tape.payload_nbytes == (tape.bit_length + 7) // 8
    record = tape.to_bytes()

    # The reconstructed VJP receives no x, weight, output, or framework graph.
    del x, weight, bias, output, route
    restored = PackedRouteTape.from_bytes(record)
    decoded_route = restored.read("layer0.winner")
    reconstructed = additive_extreme_vjp(upstream, decoded_route, input_features=7)
    for actual, reference in zip(reconstructed, expected):
        # Scatter/reduction order can differ from framework autograd by a few
        # floating-point ulps; the exact-integer routing test covers bit parity.
        torch.testing.assert_close(actual, reference, rtol=1e-14, atol=1e-14)
    assert restored.read("dropout").tolist() == [[1, 0, 1], [0, 1, 0]]


@pytest.mark.parametrize(
    "cardinality,expected_bits",
    [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (256, 8), (257, 9)],
)
def test_fixed_width_code_meets_the_worst_case_coding_bound(cardinality, expected_bits):
    assert bits_for_cardinality(cardinality) == expected_bits
    values = torch.arange(cardinality, dtype=torch.long)
    tape = PackedRouteTape()
    field = tape.append("route", values, cardinality=cardinality)
    assert field.bit_width == expected_bits
    assert tape.bit_length == cardinality * expected_bits
    assert torch.equal(tape.read("route"), values)


def test_packed_field_validation_rejects_unrepresentable_routes():
    with pytest.raises(ValueError, match="outside cardinality"):
        PackedRouteTape().append("bad", torch.tensor([0, 3]), cardinality=3)
    with pytest.raises(ValueError, match="cannot encode cardinality"):
        PackedRouteTape().append("narrow", torch.tensor([0]), cardinality=5, bit_width=2)
    with pytest.raises(ValueError, match="unsigned"):
        PackedRouteTape().append("negative", torch.tensor([-1]), bit_width=2)
    with pytest.raises(ValueError, match="positive integer"):
        bits_for_cardinality(0)


def _single_value_tape(value):
    tape = PackedRouteTape()
    tape.append("route", torch.tensor([value]), cardinality=4)
    return tape


def test_fixed_capacity_ring_overwrites_oldest_and_roundtrips_state():
    ring = RouteTapeRingBuffer(capacity=2, slot_bytes=256)
    first = _single_value_tape(0)
    ring.append(first)
    # Records are frozen on insertion rather than aliased to the source object.
    first.append_mask("later_mutation", torch.tensor([True]))
    assert "later_mutation" not in ring.oldest

    ring.append(_single_value_tape(1))
    ring.append(_single_value_tape(2))
    assert len(ring) == 2
    assert [tape.read("route").item() for tape in ring] == [1, 2]
    assert ring.allocated_bytes == 512

    restored = RouteTapeRingBuffer.from_state_dict(ring.state_dict())
    assert [tape.read("route").item() for tape in restored] == [1, 2]
    assert restored.pop_oldest().read("route").item() == 1
    assert restored.newest.read("route").item() == 2


def test_ring_rejects_a_record_larger_than_its_fixed_slot():
    tape = _single_value_tape(3)
    record_size = len(tape.to_bytes())
    ring = RouteTapeRingBuffer(capacity=1, slot_bytes=record_size - 1)
    with pytest.raises(ValueError, match="exceeding slot_bytes"):
        ring.append(tape)
