"""Synthetic arithmetic tests only: no corpus, forward model or training run."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from manuscript.tools import fpga_lm_trajectory_arithmetic as arithmetic


def mapping(value=0.0, *, width=4, vocab=3, layers=2, taps=3):
    shapes = {"EMBED": (vocab, width), "MH": (layers, width, taps),
              "MP": (layers, width, width), "MB": (layers, width),
              "W_HEAD": (vocab, width)}
    return {name: torch.full(shape, float(value), dtype=torch.float64) for name, shape in shapes.items()}


def copy(values):
    return {name: value.clone() for name, value in values.items()}


def equal(left, right):
    assert set(left) == set(right)
    assert all(torch.equal(left[name], right[name]) for name in left)


def coordinate(values, value, name="EMBED"):
    values[name].reshape(-1)[0] = value
    return values


def test_source_identity_binds_own_and_frozen_numeric_dependency():
    root = Path(__file__).resolve().parents[1]
    identity = arithmetic.source_identity()
    assert set(identity) == {"manuscript/tools/fpga_lm_trajectory_arithmetic.py",
                             "manuscript/tools/fpga_lm_numeric_diagnostic.py"}
    for path, digest in identity.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("k", [0, 1, 18, 30])
def test_shadow_is_exact_and_strictly_less_than_one_lsb(k):
    denominator = 1 << k
    w, r = mapping(14), mapping(denominator - 1)
    result = arithmetic.shadow(w, r, k)
    equal(result, mapping(14 - (denominator - 1) / denominator))
    assert max(float((result[name] - w[name]).abs().max()) for name in w) < 1


def test_shadow_can_be_outside_int16_without_claiming_a_safe_forward():
    result = arithmetic.shadow(mapping(-32768), mapping(7), 3)
    equal(result, mapping(-32768.875))


@pytest.mark.parametrize("r,g,wanted_w,wanted_r", [
    (7, 1, 9, 0), (-7, -1, 11, 0), (7, -6, 10, 1),
    (-7, 6, 10, -1), (0, -1, 10, -1), (0, 17, 8, 1), (0, -17, 12, -1),
])
def test_signed_truncation_release_and_cancellation(r, g, wanted_w, wanted_r):
    weights, remainders, gradient = mapping(10), mapping(r), mapping(g)
    originals = [copy(x) for x in (weights, remainders, gradient)]
    result = arithmetic.ef_step(weights, remainders, gradient, 3)
    equal(result["weights"], mapping(wanted_w))
    equal(result["remainders"], mapping(wanted_r))
    before = arithmetic.shadow(weights, remainders, 3)
    after = arithmetic.shadow(result["weights"], result["remainders"], 3)
    equal(after, {name: before[name] - gradient[name] / 8 for name in before})
    for current, original in zip((weights, remainders, gradient), originals):
        equal(current, original)


def test_native_step_calls_the_frozen_integer_oracle(monkeypatch):
    original, calls = arithmetic.numeric.ef_rounding, []

    def recording(weights, remainders, gradient, k):
        calls.append((weights.shape, weights.dtype.kind, remainders.dtype.kind, gradient.dtype.kind, k))
        return original(weights, remainders, gradient, k)

    monkeypatch.setattr(arithmetic.numeric, "ef_rounding", recording)
    arithmetic.ef_step(mapping(2), mapping(), mapping(1), 3)
    assert len(calls) == 5
    assert all(row[1:] == ("i", "i", "i", 3) for row in calls)


def test_int32_min_gradient_is_valid_when_sum_and_weight_are_safe():
    result = arithmetic.ef_step(mapping(), mapping(), mapping(-(1 << 31)), 18)
    equal(result["weights"], mapping(8192))
    equal(result["remainders"], mapping())
    equal(arithmetic.unrounded_step(mapping(), mapping(-(1 << 31)), 18), mapping(8192))


@pytest.mark.parametrize("k", [0, 1, 18, 30])
def test_unrounded_step_preserves_exact_dyadic_lattice(k):
    denominator = 1 << k
    result = arithmetic.unrounded_step(mapping(3 / denominator), mapping(2), k)
    equal(result, mapping(1 / denominator))
    for value in result.values():
        assert torch.equal(value * denominator, (value * denominator).round())


def test_first_native_weights_can_stay_fixed_while_shadow_moves():
    result = arithmetic.transition(mapping(), mapping(), mapping(), mapping(7), mapping(7), mapping(), 3)
    equal(result["weights"], mapping())
    equal(result["remainders"], mapping(7))
    equal(result["float_weights"], mapping(-0.875))
    equal(arithmetic.shadow(result["weights"], result["remainders"], 3), result["float_weights"])
    assert all(result["identities"][name] is True for name in
               ("exact_shadow_recurrence", "exact_unrounded_recurrence", "exact_paired_gradient_identity"))


def test_same_current_gradient_can_release_old_remainder():
    first = arithmetic.transition(mapping(), mapping(), mapping(), mapping(7), mapping(7), mapping(), 3)
    second = arithmetic.transition(first["weights"], first["remainders"], first["float_weights"],
                                   mapping(1), mapping(1), first["cumulative_gradient_difference"], 3)
    equal(second["weights"], mapping(-1))
    equal(second["remainders"], mapping())
    equal(second["float_weights"], mapping(-1))
    # Dropping the inherited remainder would miss this native visible update.
    equal(arithmetic.ef_step(first["weights"], mapping(), mapping(1), 3)["weights"], mapping())


def test_paired_trajectory_identity_for_distinct_gradient_streams():
    generator = torch.Generator().manual_seed(1977)
    weights, remainders, floating, cumulative = mapping(10), mapping(), mapping(10), mapping()
    summed_native, summed_float = mapping(), mapping()
    initial = copy(weights)
    for _ in range(32):
        native = {name: torch.randint(-40000, 40001, value.shape, generator=generator).to(torch.float64)
                  for name, value in weights.items()}
        other = {name: torch.randint(-40000, 40001, value.shape, generator=generator).to(torch.float64)
                 for name, value in weights.items()}
        original = [copy(x) for x in (weights, remainders, floating, native, other, cumulative)]
        result = arithmetic.transition(weights, remainders, floating, native, other, cumulative)
        for current, saved in zip((weights, remainders, floating, native, other, cumulative), original):
            equal(current, saved)
        weights, remainders = result["weights"], result["remainders"]
        floating, cumulative = result["float_weights"], result["cumulative_gradient_difference"]
        for name in weights:
            summed_native[name] += native[name]
            summed_float[name] += other[name]
        expected_shadow = {name: initial[name] - summed_native[name] / (1 << 18) for name in weights}
        expected_float = {name: initial[name] - summed_float[name] / (1 << 18) for name in weights}
        equal(arithmetic.shadow(weights, remainders), expected_shadow)
        equal(floating, expected_float)
        equal(cumulative, {name: summed_native[name] - summed_float[name] for name in weights})


def test_valid_resumed_nonzero_cumulative_and_remainder():
    # z=-7/8, f=-3/8, so C=4 exactly records past gradient divergence.
    result = arithmetic.transition(mapping(), mapping(7), mapping(-3 / 8), mapping(2), mapping(-1), mapping(4), 3)
    equal(result["cumulative_gradient_difference"], mapping(7))
    equal(result["float_weights"], mapping(-0.25))
    equal(arithmetic.shadow(result["weights"], result["remainders"], 3), mapping(-1.125))


def test_gradient_queried_at_visible_weight_does_not_guarantee_shadow_descent():
    # L(x)=(x+1/8)^2/2, w=0, z=-7/8. G(w)=1 raw unit at Q=8.
    before = arithmetic.shadow(mapping(), mapping(7), 3)["EMBED"][0, 0].item()
    result = arithmetic.ef_step(mapping(), mapping(7), mapping(1), 3)
    after = arithmetic.shadow(result["weights"], result["remainders"], 3)["EMBED"][0, 0].item()
    assert 0.5 * (after + 0.125) ** 2 > 0.5 * (before + 0.125) ** 2
    fresh_only = arithmetic.unrounded_step(mapping(), mapping(1), 3)["EMBED"][0, 0].item()
    assert fresh_only == -0.125


def test_comparison_counts_all_regions_and_uses_difference_norms():
    left, right = mapping(), mapping()
    coordinate(right, 3, "EMBED")
    coordinate(right, 4, "MH")
    result = arithmetic.compare(left, right)
    assert result["global"] == {"l1": 7.0, "l2": 5.0, "max_abs": 4.0,
                                "differing_words": 2, "cosine": None}
    assert set(result["regions"]) == set(arithmetic.WEIGHTS)
    assert result["regions"]["EMBED"]["l2"] == 3
    assert result["regions"]["MH"]["l2"] == 4
    assert result["regions"]["MP"]["differing_words"] == 0


@pytest.mark.parametrize("left,right,expected", [(1, 2, 1), (1, -2, -1), (0, 1, None), (0, 0, None)])
def test_comparison_cosine_and_zero_vectors(left, right, expected):
    result = arithmetic.compare(mapping(left), mapping(right))
    if expected is None:
        assert result["global"]["cosine"] is None
    else:
        assert result["global"]["cosine"] == pytest.approx(expected, abs=1e-14)


def test_small_nonzero_comparison_does_not_mislabel_as_zero():
    result = arithmetic.compare(mapping(1e-300), mapping(-1e-300))
    assert result["global"]["cosine"] == pytest.approx(-1)
    assert result["global"]["l2"] > 0


def test_results_are_detached_copies_even_on_zero_update():
    weights, remainders, gradient = mapping(1), mapping(), mapping()
    for result in (arithmetic.shadow(weights, remainders), arithmetic.ef_step(weights, remainders, gradient)["weights"],
                   arithmetic.unrounded_step(weights, gradient)):
        for name in weights:
            assert result[name].data_ptr() != weights[name].data_ptr()
            assert result[name].grad_fn is None and not result[name].requires_grad
        coordinate(result, 300)
    equal(weights, mapping(1))


@pytest.mark.parametrize("bad_k", [True, False, -1, 31, 18.0, "18", None])
def test_k_is_strict_literal_bounded_integer(bad_k):
    for function in (lambda: arithmetic.shadow(mapping(), mapping(), bad_k),
                     lambda: arithmetic.ef_step(mapping(), mapping(), mapping(), bad_k),
                     lambda: arithmetic.unrounded_step(mapping(), mapping(), bad_k),
                     lambda: arithmetic.transition(*[mapping() for _ in range(6)], k=bad_k)):
        with pytest.raises(ValueError, match="literal integer"):
            function()


@pytest.mark.parametrize("mutation", ["missing", "extra", "float32", "int64", "requires_grad", "nan", "inf",
                                       "wrong_embed_rank", "wrong_mh_width", "wrong_mb", "empty"])
def test_canonical_mapping_rejects_malformed_fields(mutation):
    values = mapping()
    if mutation == "missing":
        del values["MB"]
    elif mutation == "extra":
        values["EXTRA"] = torch.zeros(1)
    elif mutation in ("float32", "int64"):
        values["EMBED"] = values["EMBED"].to(getattr(torch, mutation))
    elif mutation == "requires_grad":
        values["EMBED"].requires_grad_(True)
    elif mutation in ("nan", "inf"):
        coordinate(values, float(mutation))
    elif mutation == "wrong_embed_rank":
        values["EMBED"] = values["EMBED"].reshape(-1)
    elif mutation == "wrong_mh_width":
        values["MH"] = torch.zeros((2, 3, 3), dtype=torch.float64)
    elif mutation == "wrong_mb":
        values["MB"] = torch.zeros((2, 4, 1), dtype=torch.float64)
    else:
        values["EMBED"] = torch.empty((0, 4), dtype=torch.float64)
    with pytest.raises(ValueError):
        arithmetic.compare(values, mapping())


@pytest.mark.parametrize("region", arithmetic.WEIGHTS)
def test_integer_fields_reject_fractions_in_every_region(region):
    with pytest.raises(ValueError):
        arithmetic.shadow(coordinate(mapping(), 0.25, region), mapping())
    with pytest.raises(ValueError):
        arithmetic.shadow(mapping(), coordinate(mapping(), 0.25, region))
    with pytest.raises(ValueError):
        arithmetic.unrounded_step(mapping(), coordinate(mapping(), 0.25, region))


@pytest.mark.parametrize("value", [-32769, 32768])
def test_native_and_floating_start_weights_must_fit_int16_range(value):
    with pytest.raises(ValueError):
        arithmetic.shadow(mapping(value), mapping())
    with pytest.raises(ValueError):
        arithmetic.unrounded_step(mapping(value), mapping())


@pytest.mark.parametrize("value", [-8, 8])
def test_noncanonical_remainder_rejected(value):
    with pytest.raises(ValueError, match="canonical remainders"):
        arithmetic.shadow(mapping(), mapping(value), 3)


@pytest.mark.parametrize("value", [-(1 << 31) - 1, 1 << 31])
def test_raw_gradient_must_be_int32(value):
    with pytest.raises(ValueError):
        arithmetic.ef_step(mapping(), mapping(), mapping(value))
    with pytest.raises(ValueError):
        arithmetic.unrounded_step(mapping(), mapping(value))


@pytest.mark.parametrize("value", [0.1, 0.0625])
def test_floating_weight_requires_declared_dyadic_lattice(value):
    with pytest.raises(ValueError, match="exact dyadics"):
        arithmetic.unrounded_step(mapping(value), mapping(), 3)


@pytest.mark.parametrize("r,g", [(1, (1 << 31) - 1), (-1, -(1 << 31))])
def test_native_accumulator_overflow_is_fail_closed(r, g):
    with pytest.raises(OverflowError, match="accumulator"):
        arithmetic.ef_step(mapping(), mapping(r), mapping(g), 18)


@pytest.mark.parametrize("region", arithmetic.WEIGHTS)
def test_failed_candidate_never_partially_mutates_input(region):
    weights, remainders, gradient = mapping(), mapping(), mapping(8)
    coordinate(weights, 32767, region)
    coordinate(gradient, -8, region)
    saved = [copy(x) for x in (weights, remainders, gradient)]
    with pytest.raises(OverflowError, match="clip"):
        arithmetic.ef_step(weights, remainders, gradient, 3)
    with pytest.raises(OverflowError, match="range"):
        arithmetic.unrounded_step(weights, gradient, 3)
    for current, original in zip((weights, remainders, gradient), saved):
        equal(current, original)


def test_unrounded_fractional_range_violation_is_not_rounded_back_to_safe():
    with pytest.raises(OverflowError, match="range"):
        arithmetic.unrounded_step(mapping(32767), mapping(-1), 18)


@pytest.mark.parametrize("which", ["remainders", "gradient", "float_weights", "cumulative"])
def test_matching_dictionary_geometry_required_without_broadcasting(which):
    wrong = mapping(width=8)
    if which == "remainders":
        call = lambda: arithmetic.shadow(mapping(), wrong)
    elif which == "gradient":
        call = lambda: arithmetic.ef_step(mapping(), mapping(), wrong)
    elif which == "float_weights":
        call = lambda: arithmetic.transition(mapping(), mapping(), wrong, mapping(), mapping(), mapping())
    else:
        call = lambda: arithmetic.transition(mapping(), mapping(), mapping(), mapping(), mapping(), wrong)
    with pytest.raises(ValueError, match="broadcasting"):
        call()


@pytest.mark.parametrize("value", [1, 0.5, float(1 << 53)])
def test_incoming_paired_identity_or_exact_integer_guard_rejects_bad_history(value):
    with pytest.raises(ValueError):
        arithmetic.transition(mapping(), mapping(), mapping(), mapping(), mapping(), mapping(value))


def test_incoming_shadow_not_visible_weight_must_match_float_history():
    with pytest.raises(ValueError, match="incoming paired"):
        arithmetic.transition(mapping(), mapping(1), mapping(), mapping(), mapping(), mapping())


def test_corrupted_integer_oracle_result_is_detected_by_shadow_identity(monkeypatch):
    original = arithmetic.numeric.ef_rounding

    def corrupt(*args):
        result = original(*args)
        result["remainders"].reshape(-1)[0] += 1
        return result

    monkeypatch.setattr(arithmetic.numeric, "ef_rounding", corrupt)
    with pytest.raises(ValueError, match="shadow recurrence"):
        arithmetic.transition(*[mapping() for _ in range(6)])


def test_comparison_nonfinite_statistics_fail_closed():
    with pytest.raises(OverflowError, match="statistics overflow"):
        arithmetic.compare(mapping(1e308), mapping(-1e308))
