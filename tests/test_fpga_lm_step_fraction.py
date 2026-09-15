"""Pure dyadic fraction tests: no model, native execution, corpus or training."""
import hashlib
from pathlib import Path

import pytest
import torch

from manuscript.tools import fpga_lm_step_fraction as fraction


def mapping(value=0):
    shapes = {"EMBED": (2, 3), "MH": (2, 3, 3), "MP": (2, 3, 3),
              "MB": (2, 3), "W_HEAD": (2, 3)}
    return {name: torch.full(shape, float(value), dtype=torch.float64) for name, shape in shapes.items()}


def gradients(value=0):
    return [mapping(value), mapping(), mapping(), mapping()]


def clone(values):
    return {name: tensor.clone() for name, tensor in values.items()}


def assert_mapping(actual, expected):
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name].dtype == torch.float64
        assert actual[name].device.type == "cpu"
        assert not actual[name].requires_grad
        assert actual[name].shape == expected[name].shape
        assert torch.equal(actual[name], expected[name])


def signed_trunc(numerator, denominator):
    return abs(numerator) // denominator * (-1 if numerator < 0 else 1)


def test_source_identity_and_fixed_counterfactual_protocol():
    assert fraction.BASE_K == 18
    assert fraction.SHIFTS == (None, 4, 2, 0)
    assert fraction.GRADIENTS_PER_BLOCK == 4
    root = Path(__file__).resolve().parents[1]
    sources = fraction.source_identity()
    assert set(sources) == set(fraction.accumulation.source_identity()) | {
        "manuscript/tools/fpga_lm_step_fraction.py"}
    assert "tikitorch/optim/__init__.py" in sources
    for relative, digest in sources.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("shift", fraction.SHIFTS)
@pytest.mark.parametrize("container", [list, tuple])
def test_all_coordinates_against_independent_python_integer_oracle(shift, container):
    q = 1 << 18
    w, r, gs = mapping(), mapping(), gradients()
    for region_index, name in enumerate(fraction.WEIGHTS):
        for index in range(w[name].numel()):
            w[name].reshape(-1)[index] = index - 9
            r[name].reshape(-1)[index] = (index * 17513 + region_index * 71) % (2*q-1) - q + 1
            for position, gradient in enumerate(gs):
                gradient[name].reshape(-1)[index] = (index * 977 + position * 37111 + region_index * 37) % (5*q) - 2*q
    saved = [clone(item) for item in (w, r, *gs)]
    result = fraction.step_fraction(w, r, container(gs), shift)
    factor = 1 if shift is None else 1 << shift
    denominator = q * factor
    assert type(result["alpha"]) is float
    assert result["alpha"] == (0 if shift is None else 1/factor)
    assert result["k"] == (18 if shift is None else 18 + shift)
    assert result["base_k"] == 18 and result["shift"] == shift
    for name in fraction.WEIGHTS:
        for index in range(w[name].numel()):
            wi, ri = int(w[name].reshape(-1)[index]), int(r[name].reshape(-1)[index])
            gi = sum(int(g[name].reshape(-1)[index]) for g in gs)
            numerator = ri if shift is None else factor*ri + gi
            quotient = 0 if shift is None else signed_trunc(numerator, denominator)
            next_w = wi - quotient
            next_r = numerator - quotient * denominator
            actual_w = int(result["weights"][name].reshape(-1)[index])
            actual_r = int(result["remainders"][name].reshape(-1)[index])
            assert (actual_w, actual_r) == (next_w, next_r)
            assert abs(actual_r) < denominator
            assert actual_w*denominator-actual_r == factor*(wi*q-ri) - (0 if shift is None else gi)
            assert result["gradient"][name].reshape(-1)[index] == gi
            assert result["shadow_before"][name].reshape(-1)[index] == wi-ri/q
            assert result["ideal"][name].reshape(-1)[index] == (next_w*denominator-next_r)/denominator
            assert result["fresh_ideal"][name].reshape(-1)[index] == wi - (0 if shift is None else gi/denominator)
    for actual, expected in zip((w, r, *gs), saved):
        assert_mapping(actual, expected)
    for key in ("public_reference_equal", "exact_shadow_recurrence"):
        assert result[key] is True
    assert result["committed"] is result["native_execution"] is False
    for output in ("weights", "remainders", "gradient", "shadow_before", "ideal", "fresh_ideal"):
        for name in fraction.WEIGHTS:
            assert all(result[output][name].data_ptr() != item[name].data_ptr() for item in (w, r, *gs))


@pytest.mark.parametrize("shift", [4, 2])
def test_zero_gradient_rescales_carry_without_changing_inherited_shadow(shift):
    w, r = mapping(7), mapping(-17513)
    result = fraction.step_fraction(w, r, gradients(), shift)
    assert_mapping(result["weights"], w)
    assert_mapping(result["remainders"], {name: value*(1 << shift) for name, value in r.items()})
    expected = {name: w[name]-r[name]/(1 << 18) for name in fraction.WEIGHTS}
    assert_mapping(result["ideal"], expected)
    for name in fraction.WEIGHTS:
        naive = w[name]-r[name]/(1 << (18+shift))
        assert not torch.equal(naive, expected[name])


def test_zero_fraction_preserves_full_state_and_uses_actual_zero_public_update(monkeypatch):
    w, r, gs = mapping(-12), mapping(37), gradients(2**20)
    original = fraction.accumulation.fixed_point_ef_step
    calls = []

    def public(weights, remainder, gradient, **kwargs):
        calls.append((gradient.clone(), kwargs["k"]))
        return original(weights, remainder, gradient, **kwargs)

    monkeypatch.setattr(fraction.accumulation, "fixed_point_ef_step", public)
    result = fraction.step_fraction(w, r, gs, None)
    assert_mapping(result["weights"], w)
    assert_mapping(result["remainders"], r)
    assert result["k"] == 18
    assert len(calls) == 5
    assert all(k == 18 and not torch.count_nonzero(g) for g, k in calls)
    assert_mapping(result["fresh_ideal"], w)
    assert_mapping(result["ideal"], result["shadow_before"])


def test_zero_does_not_check_hypothetical_nonzero_carry_overflow():
    w, r = mapping(), mapping((1 << 18)-1)
    gs = gradients((1 << 31)-1)
    zero = fraction.step_fraction(w, r, gs, None)
    assert_mapping(zero["weights"], w)
    assert_mapping(zero["remainders"], r)
    with pytest.raises(OverflowError, match="remainder"):
        fraction.step_fraction(w, r, gs, 0)


def test_zero_and_small_fraction_do_not_require_unit_step_weight_validity():
    w, r, gs = mapping(32767), mapping(), gradients(-(1 << 18))
    zero = fraction.step_fraction(w, r, gs, None)
    small = fraction.step_fraction(w, r, gs, 4)
    assert_mapping(zero["weights"], w)
    assert_mapping(small["weights"], w)
    # The ideal is arithmetic, not permission to execute an out-of-range forward.
    assert float(small["ideal"]["EMBED"][0, 0]) == 32767.0625
    with pytest.raises((ValueError, OverflowError)):
        fraction.step_fraction(w, r, gs, 0)


@pytest.mark.parametrize("shift", fraction.SHIFTS)
@pytest.mark.parametrize("sign", [-1, 1])
def test_raw_prefix_overflow_cannot_be_excused_by_later_cancellation(shift, sign):
    limit = (1 << 31)-1 if sign > 0 else -(1 << 31)
    gs = [mapping(limit), mapping(sign), mapping(-sign), mapping()]
    with pytest.raises(OverflowError, match="gradient prefix 2"):
        fraction.step_fraction(mapping(), mapping(), gs, shift)


@pytest.mark.parametrize("shift", [4, 2])
def test_scaled_carry_prefix_checked_even_when_original_k18_sum_fits(shift):
    r = mapping((1 << 18)-1)
    gs = gradients((1 << 31)-1-((1 << 18)-1))
    fraction.step_fraction(mapping(), r, gs, 0)
    with pytest.raises(OverflowError, match="remainder"):
        fraction.step_fraction(mapping(), r, gs, shift)


@pytest.mark.parametrize("shift", [False, True, 0.0, 4.0, -1, 1, 3, 5, 18, "0", (), {}])
def test_only_declared_literal_shift_ids_are_allowed(shift):
    with pytest.raises(ValueError, match="shift"):
        fraction.step_fraction(mapping(), mapping(), gradients(), shift)


@pytest.mark.parametrize("gs", [None, [], (), {}, [mapping()], [mapping()]*3, [mapping()]*5, iter([mapping()]*4)])
def test_exact_four_ordered_gradients_required(gs):
    with pytest.raises(ValueError, match="four"):
        fraction.step_fraction(mapping(), mapping(), gs, None)


@pytest.mark.parametrize("region", fraction.WEIGHTS)
@pytest.mark.parametrize("kind", ["nonfinite", "fractional", "int32_overflow", "dtype", "shape", "autograd"])
def test_every_gradient_region_is_strict_even_for_zero_fraction(region, kind):
    gs = gradients()
    value = gs[2][region]
    if kind in ("nonfinite", "fractional", "int32_overflow"):
        value.reshape(-1)[0] = {"nonfinite": float("nan"), "fractional": .5, "int32_overflow": 1 << 31}[kind]
    elif kind == "dtype":
        gs[2][region] = value.to(torch.int64)
    elif kind == "shape":
        gs[2][region] = value.flatten()
    else:
        value.requires_grad_()
    with pytest.raises(ValueError):
        fraction.step_fraction(mapping(), mapping(), gs, None)


@pytest.mark.parametrize("kind", ["missing_weight", "extra_gradient", "missing_remainder", "int16_weight", "fractional_weight", "canonical_r", "fractional_r"])
def test_base_state_and_mapping_membership_strict(kind):
    w, r, gs = mapping(), mapping(), gradients()
    if kind == "missing_weight":
        w.pop("MP")
    elif kind == "extra_gradient":
        gs[0]["extra"] = torch.zeros(1, dtype=torch.float64)
    elif kind == "missing_remainder":
        r.pop("MH")
    else:
        target, value = {"int16_weight": (w, 32768), "fractional_weight": (w, .1),
                         "canonical_r": (r, 1 << 18), "fractional_r": (r, .1)}[kind]
        target["EMBED"][0, 0] = value
    with pytest.raises(ValueError):
        fraction.step_fraction(w, r, gs, None)


def test_nonzero_partial_failure_never_mutates_input_regions():
    w, r, gs = mapping(), mapping(), gradients(1 << 18)
    w["W_HEAD"][:] = -32768
    saved = [clone(item) for item in (w, r, *gs)]
    with pytest.raises((ValueError, OverflowError)):
        fraction.step_fraction(w, r, gs, 0)
    for actual, expected in zip((w, r, *gs), saved):
        assert_mapping(actual, expected)


@pytest.mark.parametrize("shift", fraction.SHIFTS)
def test_public_arithmetic_disagreement_fails_not_a_false_parity_flag(monkeypatch, shift):
    original = fraction.accumulation.fixed_point_ef_step

    def wrong(*args, **kwargs):
        w, r = original(*args, **kwargs)
        return w + 1, r

    monkeypatch.setattr(fraction.accumulation, "fixed_point_ef_step", wrong)
    with pytest.raises(ValueError, match="differ"):
        fraction.step_fraction(mapping(), mapping(), gradients(7), shift)
