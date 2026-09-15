"""Synthetic accumulation tests only; no corpus, native model, or study run."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from manuscript.tools import fpga_lm_accumulation_arithmetic as accumulation


def mapping(value=0, *, width=3, vocab=2, layers=2, taps=3):
    shapes = {"EMBED": (vocab, width), "MH": (layers, width, taps),
              "MP": (layers, width, width), "MB": (layers, width),
              "W_HEAD": (vocab, width)}
    return {name: torch.full(shape, float(value), dtype=torch.float64)
            for name, shape in shapes.items()}


def clone(values):
    return {name: value.clone() for name, value in values.items()}


def equal(left, right):
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def put(values, value, region="EMBED"):
    values[region].reshape(-1)[0] = value
    return values


def test_source_identity_binds_public_export_oracle_and_own_source():
    root = Path(__file__).resolve().parents[1]
    identity = accumulation.source_identity()
    assert set(identity) == {
        "manuscript/tools/fpga_lm_accumulation_arithmetic.py",
        "manuscript/tools/fpga_lm_trajectory_arithmetic.py",
        "manuscript/tools/fpga_lm_numeric_diagnostic.py",
        "tikitorch/optim/fixed_point.py", "tikitorch/optim/__init__.py"}
    for path, digest in identity.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("container", [list, tuple])
def test_checked_sum_is_full_raw_sum_without_averaging(container):
    gradients, remainder = container([mapping(7), mapping(-2), mapping(4)]), mapping(-1)
    saved = [clone(x) for x in (*gradients, remainder)]
    result = accumulation.checked_sum(gradients, remainder)
    equal(result, mapping(9))
    for now, before in zip((*gradients, remainder), saved):
        equal(now, before)
    for name in result:
        assert result[name].dtype == torch.float64
        assert result[name].device.type == "cpu"
        assert not result[name].requires_grad
        assert all(result[name].data_ptr() != x[name].data_ptr() for x in (*gradients, remainder))


@pytest.mark.parametrize("gradients", [[], (), None, {}, mapping(), iter([mapping()])])
def test_block_membership_must_be_nonempty_explicit_sequence(gradients):
    with pytest.raises(ValueError, match="nonempty list or tuple"):
        accumulation.checked_sum(gradients, mapping())


@pytest.mark.parametrize("region", accumulation.WEIGHTS)
@pytest.mark.parametrize("sign", [-1, 1])
def test_raw_prefix_overflow_rejected_even_if_later_gradient_cancels(region, sign):
    edge = accumulation.INT32_MAX if sign > 0 else accumulation.INT32_MIN
    gradients = [put(mapping(), edge, region), put(mapping(), sign, region),
                 put(mapping(), -sign, region)]
    before = [clone(g) for g in gradients]
    with pytest.raises(OverflowError, match="gradient prefix 2 in " + region):
        accumulation.checked_sum(gradients, mapping())
    for now, saved in zip(gradients, before):
        equal(now, saved)


@pytest.mark.parametrize("sign", [-1, 1])
def test_remainder_plus_prefix_overflow_cannot_be_excused_by_final_cancellation(sign):
    edge = accumulation.INT32_MAX if sign > 0 else accumulation.INT32_MIN
    gradients, remainder = [mapping(edge), mapping(-edge if sign > 0 else accumulation.INT32_MAX)], mapping(sign)
    saved = [clone(x) for x in (*gradients, remainder)]
    with pytest.raises(OverflowError, match="remainder \\+ gradient prefix 1"):
        accumulation.checked_sum(gradients, remainder)
    for now, before in zip((*gradients, remainder), saved):
        equal(now, before)


def test_raw_prefix_is_checked_even_when_negative_remainder_would_hide_overflow():
    gradients = [mapping(accumulation.INT32_MAX), mapping(1), mapping(-1)]
    with pytest.raises(OverflowError, match="gradient prefix 2"):
        accumulation.checked_sum(gradients, mapping(-1))


def test_int32_min_is_allowed_and_abs_stat_does_not_wrap():
    result = accumulation.apply_block(mapping(), mapping(), [mapping(accumulation.INT32_MIN)])
    equal(result["weights"], mapping(8192))
    equal(result["remainders"], mapping())
    assert result["prefix_max_abs"] == {"gradient": 1 << 31, "remainder_plus_gradient": 1 << 31}


@pytest.mark.parametrize("k", [0, 1, 18, 30])
def test_public_reference_parity_and_exact_shadow_for_all_supported_scale_edges(k):
    q = 1 << k
    weights, remainder = mapping(10), mapping(q - 1)
    gradients = [mapping(1), mapping(-2), mapping(3)]
    result = accumulation.apply_block(weights, remainder, gradients, k)
    reference = accumulation.arithmetic.ef_step(weights, remainder, mapping(2), k)
    equal(result["weights"], reference["weights"])
    equal(result["remainders"], reference["remainders"])
    equal(result["gradient"], mapping(2))
    assert result["public_reference_equal"] is True
    assert result["exact_shadow_recurrence"] is True
    for name in weights:
        before = weights[name].to(torch.int64) * q - remainder[name].to(torch.int64)
        after = result["weights"][name].to(torch.int64) * q - result["remainders"][name].to(torch.int64)
        assert torch.equal(after, before - 2)


def test_signed_grouping_changes_visible_state_but_not_shadow():
    weights, remainder, first, second = mapping(), mapping(), mapping(9), mapping(-2)
    sequential = accumulation.arithmetic.ef_step(weights, remainder, first, 3)
    sequential = accumulation.arithmetic.ef_step(sequential["weights"], sequential["remainders"], second, 3)
    grouped = accumulation.apply_block(weights, remainder, [first, second], 3)
    equal(sequential["weights"], mapping(-1))
    equal(sequential["remainders"], mapping(-1))
    equal(grouped["weights"], mapping())
    equal(grouped["remainders"], mapping(7))
    for result in (sequential, grouped):
        equal(accumulation.arithmetic.shadow(result["weights"], result["remainders"], 3), mapping(-7 / 8))


@pytest.mark.parametrize("r,gradients,wanted_w,wanted_r", [
    (7, [1], 9, 0), (-7, [-1], 11, 0), (7, [-6, -2], 10, -1),
    (-7, [6, 2], 10, 1), (0, [-9, 2], 10, -7), (0, [9, -2], 10, 7),
])
def test_signed_release_and_cancellation(r, gradients, wanted_w, wanted_r):
    result = accumulation.apply_block(mapping(10), mapping(r), [mapping(g) for g in gradients], 3)
    equal(result["weights"], mapping(wanted_w))
    equal(result["remainders"], mapping(wanted_r))


def test_maxima_include_all_prefixes_regions_and_zero_prefix_remainder():
    remainder = put(mapping(), -7, "MP")
    gradients = [put(mapping(), 5, "W_HEAD"), put(mapping(), -4, "W_HEAD")]
    result = accumulation.apply_block(mapping(), remainder, gradients, 3)
    assert result["prefix_max_abs"] == {"gradient": 5, "remainder_plus_gradient": 7}


def test_checked_sum_does_not_claim_remainder_canonicality_without_k():
    equal(accumulation.checked_sum([mapping(-8)], mapping(8)), mapping(-8))
    with pytest.raises(ValueError, match="canonical remainders"):
        accumulation.apply_block(mapping(), mapping(8), [mapping(-8)], 3)


@pytest.mark.parametrize("bad_k", [True, False, -1, 31, 18.0, "18", None])
def test_k_is_literal_and_bounded(bad_k):
    with pytest.raises(ValueError, match="literal integer"):
        accumulation.apply_block(mapping(), mapping(), [mapping()], bad_k)


@pytest.mark.parametrize("kind", ["weights", "remainders", "gradient"])
@pytest.mark.parametrize("region", accumulation.WEIGHTS)
def test_fractional_inputs_rejected_in_every_region(kind, region):
    inputs = {"weights": mapping(), "remainders": mapping(), "gradient": mapping()}
    put(inputs[kind], 0.25, region)
    with pytest.raises(ValueError, match="exact integer"):
        accumulation.apply_block(inputs["weights"], inputs["remainders"], [inputs["gradient"]])


@pytest.mark.parametrize("kind", ["weights", "remainders", "gradient"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "float32", "int64", "nan", "inf",
                                       "requires_grad", "empty", "rank", "broadcast", "geometry"])
def test_invalid_mapping_contracts_fail_closed(kind, mutation):
    values = mapping()
    if mutation == "missing":
        del values["MB"]
    elif mutation == "extra":
        values["extra"] = torch.zeros(1, dtype=torch.float64)
    elif mutation in ("float32", "int64"):
        values["EMBED"] = values["EMBED"].to(getattr(torch, mutation))
    elif mutation in ("nan", "inf"):
        put(values, float(mutation))
    elif mutation == "requires_grad":
        values["EMBED"].requires_grad_(True)
    elif mutation == "empty":
        values["EMBED"] = torch.empty((0, 3), dtype=torch.float64)
    elif mutation == "rank":
        values["EMBED"] = values["EMBED"].reshape(-1)
    elif mutation == "broadcast":
        values["MB"] = values["MB"][:, :1]
    else:
        values = mapping(width=4)
    inputs = {"weights": mapping(), "remainders": mapping(), "gradient": mapping()}
    inputs[kind] = values
    with pytest.raises(ValueError):
        accumulation.apply_block(inputs["weights"], inputs["remainders"], [inputs["gradient"]])


@pytest.mark.parametrize("kind,bad", [("weights", -32769), ("weights", 32768),
                                       ("gradient", -(1 << 31) - 1), ("gradient", 1 << 31),
                                       ("remainders", -(1 << 31) - 1), ("remainders", 1 << 31)])
def test_input_integer_ranges_are_checked_before_conversion(kind, bad):
    inputs = {"weights": mapping(), "remainders": mapping(), "gradient": mapping()}
    inputs[kind] = mapping(bad)
    with pytest.raises(ValueError):
        accumulation.apply_block(inputs["weights"], inputs["remainders"], [inputs["gradient"]])


def test_geometry_matching_noncontiguous_views_supported_without_aliasing():
    weights, remainder, gradient = mapping(10), mapping(), mapping(1)
    for values in (weights, remainder, gradient):
        for name, value in list(values.items()):
            values[name] = value.unsqueeze(-1).expand(*value.shape, 2).clone()[..., 0]
            assert not values[name].is_contiguous()
    result = accumulation.apply_block(weights, remainder, [gradient], 0)
    equal(result["weights"], mapping(9))
    equal(result["remainders"], mapping())
    for name in weights:
        for returned in (result["weights"], result["remainders"], result["gradient"]):
            assert all(returned[name].data_ptr() != values[name].data_ptr()
                       for values in (weights, remainder, gradient))


def test_weight_overflow_is_transactional_even_in_last_region():
    weights, remainder, gradient = mapping(), mapping(), mapping(1)
    put(weights, -32768, "W_HEAD")
    before = [clone(x) for x in (weights, remainder, gradient)]
    with pytest.raises(OverflowError):
        accumulation.apply_block(weights, remainder, [gradient], 0)
    for now, saved in zip((weights, remainder, gradient), before):
        equal(now, saved)


def test_every_public_region_and_frozen_oracle_are_really_invoked(monkeypatch):
    public, oracle = accumulation.fixed_point_ef_step, accumulation.arithmetic.ef_step
    public_calls, oracle_calls = [], []

    def recording_public(weights, remainder, gradient, *, k):
        public_calls.append((weights.dtype, remainder.dtype, gradient.dtype, k))
        return public(weights, remainder, gradient, k=k)

    def recording_oracle(weights, remainder, gradient, k):
        oracle_calls.append(k)
        return oracle(weights, remainder, gradient, k)

    monkeypatch.setattr(accumulation, "fixed_point_ef_step", recording_public)
    monkeypatch.setattr(accumulation.arithmetic, "ef_step", recording_oracle)
    accumulation.apply_block(mapping(), mapping(), [mapping(1), mapping(2)], 3)
    assert public_calls == [(torch.int16, torch.int32, torch.int32, 3)] * 5
    assert oracle_calls == [3]


@pytest.mark.parametrize("mutation", ["weight_value", "remainder_value", "weight_dtype", "remainder_dtype"])
def test_public_parity_flag_cannot_pass_without_real_value_and_dtype_checks(monkeypatch, mutation):
    public = accumulation.fixed_point_ef_step

    def wrong(weights, remainder, gradient, *, k):
        w, r = public(weights, remainder, gradient, k=k)
        if mutation == "weight_value":
            w = w + 1
        elif mutation == "remainder_value":
            r = r + 1
        elif mutation == "weight_dtype":
            w = w.to(torch.int32)
        else:
            r = r.to(torch.int64)
        return w, r

    monkeypatch.setattr(accumulation, "fixed_point_ef_step", wrong)
    with pytest.raises(ValueError, match="public"):
        accumulation.apply_block(mapping(), mapping(), [mapping(1)], 3)


def test_shadow_check_is_independent_of_matching_public_and_reference_values(monkeypatch):
    def wrong_public(weights, remainder, gradient, *, k):
        return weights.clone(), remainder.clone()

    def wrong_reference(weights, remainders, gradient, k):
        return {"weights": clone(weights), "remainders": clone(remainders)}

    monkeypatch.setattr(accumulation, "fixed_point_ef_step", wrong_public)
    monkeypatch.setattr(accumulation.arithmetic, "ef_step", wrong_reference)
    with pytest.raises(ValueError, match="shadow recurrence"):
        accumulation.apply_block(mapping(), mapping(), [mapping(1)], 3)


def test_late_public_failure_leaves_all_caller_tensors_unchanged(monkeypatch):
    public, calls = accumulation.fixed_point_ef_step, []
    weights, remainder, gradients = mapping(10), mapping(1), [mapping(2), mapping(-1)]
    saved = [clone(x) for x in (weights, remainder, *gradients)]

    def fail_last(w, r, g, *, k):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("controlled last-region failure")
        return public(w, r, g, k=k)

    monkeypatch.setattr(accumulation, "fixed_point_ef_step", fail_last)
    with pytest.raises(RuntimeError, match="controlled"):
        accumulation.apply_block(weights, remainder, gradients, 3)
    for now, before in zip((weights, remainder, *gradients), saved):
        equal(now, before)
