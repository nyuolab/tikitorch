"""Synthetic exactness contracts, not a corpus, FPGA, or speed experiment."""
from __future__ import annotations

import importlib
import hashlib
import math
from pathlib import Path

import pytest
import torch


@pytest.fixture
def modules():
    return (importlib.import_module("manuscript.tools.fpga_lm_guarded_forward"),
            importlib.import_module("manuscript.tools.fpga_lm_exact_fit"))


def parameters(off_weight, *, d=16, v=4, n=4, k=7, seed=13, amplitude=.125):
    generator = torch.Generator().manual_seed(seed)
    result = {
        "EMBED": torch.randn((v, d), generator=generator, dtype=torch.float64) * amplitude,
        "MH": torch.randn((n, d, k), generator=generator, dtype=torch.float64) * amplitude,
        "MP": torch.full((n, d, d), off_weight, dtype=torch.float64),
        "MB": torch.randn((n, d), generator=generator, dtype=torch.float64) * amplitude,
        "W_HEAD": torch.full((v, d), off_weight, dtype=torch.float64),
    }
    for layer in range(n):
        selected = (torch.arange(d) + layer + 1) % d
        result["MP"][layer, torch.arange(d), selected] = torch.randn(
            d, generator=generator, dtype=torch.float64) * amplitude
    result["W_HEAD"][torch.arange(v), torch.arange(v) * (d // v)] = torch.randn(
        v, generator=generator, dtype=torch.float64) * amplitude
    return result


def clones(values):
    return {name: value.clone() for name, value in values.items()}


def exact(actual, expected):
    """Require IEEE payload equality, including signed zeros, not just allclose."""
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        candidate = actual[name]
        assert candidate.dtype == value.dtype and candidate.shape == value.shape, name
        if value.dtype == torch.float64:
            assert torch.equal(candidate.contiguous().view(torch.int64),
                               value.contiguous().view(torch.int64)), name
        else:
            assert torch.equal(candidate, value), name


def compare_forward(modules, values, tokens, *, context=True, **kwargs):
    guarded, helper = modules
    before = clones(values)
    actual = guarded.forward(values, tokens, context=context, **kwargs)
    expected = helper.forward(values, tokens, context=context)
    exact(helper.trace(actual), helper.trace(expected))
    assert actual["parameter_sha256"] == expected["parameter_sha256"]
    assert actual["geometry"] == expected["geometry"]
    assert actual["mode"] == expected["mode"] == "smooth"
    assert actual["units"] == "dimensionless"
    assert actual["use_context"] is context
    assert actual["native_execution"] is False
    exact(values, before)
    return actual, expected


@pytest.mark.parametrize("seed", [0, 7, 29])
@pytest.mark.parametrize("context", [False, True])
def test_rich_d16_forward_is_bitwise_dense_with_learned_active_weights(modules, seed, context):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT, seed=seed)
    tokens = torch.tensor([0, 1, 2, 3, 1, 0, 3, 2, 0, 2, 1, 3], dtype=torch.int64)
    actual, _ = compare_forward(modules, values, tokens, context=context)
    facts = actual["guard_diagnostics"]
    assert facts["structure_certified"] is True
    assert facts["whole_model_dense_fallback"] is False
    assert facts["projection_fast_operations"] == len(tokens) * 4
    assert facts["head_fast_operations"] == len(tokens)
    assert facts["projection_dense_operations"] == facts["head_dense_operations"] == 0


@pytest.mark.parametrize("context", [False, True])
def test_full_parameter_vjp_is_same_dense_reverse_and_independent_autograd(modules, context):
    guarded, helper = modules
    values = parameters(guarded.OFF_WEIGHT, seed=43)
    tokens = torch.tensor([0, 1, 3, 2, 0, 2, 1, 3, 0], dtype=torch.int64)
    targets = tokens.roll(-1)
    actual, expected = compare_forward(modules, values, tokens, context=context)
    cotangent = torch.randn((len(tokens), 4), dtype=torch.float64,
                            generator=torch.Generator().manual_seed(11))
    exact(helper.reference.manual_vjp(values, tokens, cotangent, actual),
          helper.reference.manual_vjp(values, tokens, cotangent, expected))
    # An arbitrary signed cotangent checks more than the constrained CE head.
    leaves = {name: value.clone().requires_grad_(True) for name, value in values.items()}
    replay = helper.reference.forward(leaves, tokens, mode="fixed_routes", base=actual,
                                      use_context=context)
    independent = torch.autograd.grad(replay["logits"], tuple(leaves.values()), grad_outputs=cotangent)
    manual = helper.reference.manual_vjp(values, tokens, cotangent, actual)
    for name, gradient in zip(leaves, independent):
        torch.testing.assert_close(manual[name], gradient, atol=1e-12, rtol=1e-11)
    produced = helper.gradient(values, tokens, targets, actual, len(tokens) * 4)
    exact(produced, helper.gradient(values, tokens, targets, expected, len(tokens) * 4))
    checked = helper.gradient(values, tokens, targets, actual, len(tokens) * 4, independent=True)
    for name in produced:
        torch.testing.assert_close(produced[name], checked[name], atol=1e-12, rtol=1e-11)


def zero_parameters(off_weight=-1.0, *, n=1):
    values = parameters(off_weight, d=4, v=2, n=n, k=3, amplitude=0)
    values["MP"].fill_(off_weight)
    values["MP"][:, torch.arange(4), torch.tensor([3, 2, 1, 0])] = 0
    values["W_HEAD"].fill_(off_weight)
    values["W_HEAD"][torch.arange(2), torch.tensor([3, 2])] = 0
    return values


@pytest.mark.parametrize("peak", [1.0, 2.0])
def test_equal_or_failed_projection_guard_falls_back_with_first_index_ties(modules, peak):
    values = zero_parameters()
    values["EMBED"][0, 0] = peak
    actual, _ = compare_forward(modules, values, torch.tensor([0]), off_weight=-1.0)
    facts = actual["guard_diagnostics"]
    assert facts["structure_certified"] is True
    assert facts["whole_model_dense_fallback"] is False
    assert facts["projection_dense_operations"] == 1
    # At peak=1 the designated input3 and off input0 tie exactly; dense first
    # index must win, not the designated sparse input. At peak=2 it wins strictly.
    assert actual["routes"]["PROJ_IDX"][0, 0, 0].item() == 0


@pytest.mark.parametrize("peak", [1.0, 2.0])
def test_equal_or_failed_head_guard_falls_back(modules, peak):
    values = zero_parameters()
    # Temporal/projection values remain zero, so MB directly creates hidden.
    values["MB"][0, 0] = peak
    actual, _ = compare_forward(modules, values, torch.tensor([0]), off_weight=-1.0)
    facts = actual["guard_diagnostics"]
    assert facts["structure_certified"] is True
    assert facts["projection_fast_operations"] == 1
    assert facts["head_dense_operations"] == 1
    assert actual["routes"]["AD_HEAD"][0, 0].item() == 0


@pytest.mark.parametrize("region", ["MP", "W_HEAD"])
@pytest.mark.parametrize("case", ["none", "two", "all"])
def test_unsupported_row_structure_calls_complete_dense_fallback(modules, region, case):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT)
    row = values[region][0, 0] if region == "MP" else values[region][0]
    if case == "none":
        row.fill_(guarded.OFF_WEIGHT)
    elif case == "two":
        available = torch.nonzero(row == guarded.OFF_WEIGHT)[0].item()
        row[available] = .375
    else:
        row.zero_()
    actual, _ = compare_forward(modules, values, torch.tensor([0, 1, 2, 3]))
    assert actual["guard_diagnostics"]["structure_certified"] is False
    assert actual["guard_diagnostics"]["whole_model_dense_fallback"] is True


@pytest.mark.parametrize("region", ["MP", "W_HEAD"])
def test_no_stale_structural_certificate_after_off_weight_mutates(modules, region):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT)
    tokens = torch.tensor([0, 1, 2, 3])
    first, _ = compare_forward(modules, values, tokens)
    assert first["guard_diagnostics"]["structure_certified"] is True
    row = values[region][0, 0] if region == "MP" else values[region][0]
    available = torch.nonzero(row == guarded.OFF_WEIGHT)[0].item()
    row[available] = torch.nextafter(row[available], row.new_tensor(float("inf")))
    second, _ = compare_forward(modules, values, tokens)
    assert second["guard_diagnostics"]["structure_certified"] is False
    assert second["parameter_sha256"] != first["parameter_sha256"]


def test_dynamic_temporal_max_min_padding_and_first_ties(modules):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT, amplitude=0)
    tokens = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    tied, _ = compare_forward(modules, values, tokens)
    assert torch.count_nonzero(tied["routes"]["TAP_IDX"]) == 0
    # Prefer older eligible taps in BOTH max and min layers. Invalid future
    # indices must never participate, and each layer has its own dilation.
    for layer in range(4):
        values["MH"][layer] = ((1 if layer % 2 == 0 else -1)
                               * torch.arange(7, dtype=torch.float64)[None, :])
    routed, _ = compare_forward(modules, values, tokens)
    for layer in range(4):
        expected = torch.minimum(torch.arange(len(tokens)) // (2 ** layer), torch.tensor(6))
        assert torch.equal(routed["routes"]["TAP_IDX"][layer], expected[:, None].expand(-1, 16))
    disabled, _ = compare_forward(modules, values, tokens, context=False)
    assert torch.count_nonzero(disabled["routes"]["TAP_IDX"]) == 0


def test_causality_and_full_window_context_reset(modules):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT)
    tokens = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0])
    changed = tokens.clone()
    changed[-1] = 2
    first, _ = compare_forward(modules, values, tokens)
    second, _ = compare_forward(modules, values, changed)
    assert torch.equal(first["logits"][:-1], second["logits"][:-1])
    disabled, _ = compare_forward(modules, values, tokens, context=False)
    one_at_a_time = torch.cat([guarded.forward(values, token.reshape(1))["logits"] for token in tokens])
    assert torch.equal(disabled["logits"], one_at_a_time)


def test_default_off_weight_is_the_existing_dimensionless_barrier(modules):
    guarded, helper = modules
    assert type(guarded.OFF_WEIGHT) is float
    assert guarded.OFF_WEIGHT == helper.BETA * -8192.0
    assert guarded.OFF_WEIGHT != -8192.0


def test_signed_zero_and_nonassociative_bias_residual_order_match_payloads(modules):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT, d=4, v=2, n=2, k=3, amplitude=0)
    for name in ("EMBED", "MH", "MB"):
        values[name].fill_(-0.0)
    values["MP"][values["MP"] != guarded.OFF_WEIGHT] = -0.0
    values["W_HEAD"][values["W_HEAD"] != guarded.OFF_WEIGHT] = -0.0
    compare_forward(modules, values, torch.tensor([0, 1, 0]))
    # Same addition schedule is essential: (projection+MB)+residual is not
    # generally the floating-point value of projection+(MB+residual).
    values = zero_parameters(off_weight=-float(2 ** 60))
    values["EMBED"].fill_(float(2 ** 53))
    values["MP"][values["MP"] != -float(2 ** 60)] = -float(2 ** 53)
    values["MB"].fill_(1.0)
    actual, _ = compare_forward(modules, values, torch.tensor([0]), off_weight=-float(2 ** 60))
    assert torch.equal(actual["logits"], torch.ones_like(actual["logits"]))
    # This second arrangement also detects (residual+projection)+MB.
    values = zero_parameters(off_weight=-float(2 ** 60))
    values["EMBED"].fill_(1.0)
    values["MP"][values["MP"] != -float(2 ** 60)] = float(2 ** 53)
    values["MB"].fill_(-float(2 ** 53))
    actual, _ = compare_forward(modules, values, torch.tensor([0]), off_weight=-float(2 ** 60))
    assert torch.equal(actual["logits"], torch.ones_like(actual["logits"]))


def test_short_d512_forward_uses_same_full_dimensional_mean_and_routes(modules):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT, d=512, v=128, seed=17, amplitude=.03125)
    tokens = torch.tensor([0, 91, 127], dtype=torch.int64)
    actual, _ = compare_forward(modules, values, tokens)
    assert actual["guard_diagnostics"]["projection_fast_operations"] == 12
    assert actual["guard_diagnostics"]["head_fast_operations"] == 3


@pytest.mark.parametrize("bad", [True, 0, float("inf"), float("-inf"), float("nan")])
def test_off_weight_requires_finite_literal_float(modules, bad):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT)
    with pytest.raises(ValueError):
        guarded.forward(values, torch.tensor([0]), off_weight=bad)


@pytest.mark.parametrize("case", ["missing", "dtype", "attached", "nonfinite", "tokens", "context"])
def test_invalid_inputs_reject_without_mutating_valid_arrays(modules, case):
    guarded, _ = modules
    values = parameters(guarded.OFF_WEIGHT)
    tokens, context = torch.tensor([0, 1]), True
    if case == "missing":
        del values["MB"]
    elif case == "dtype":
        values["MB"] = values["MB"].float()
    elif case == "attached":
        values["MB"].requires_grad_(True)
    elif case == "nonfinite":
        values["MB"][0, 0] = math.inf
    elif case == "tokens":
        tokens = tokens.double()
    else:
        context = 1
    before = clones(values)
    with pytest.raises(ValueError):
        guarded.forward(values, tokens, context=context)
    exact(values, before)


@pytest.mark.parametrize("whole_model", [False, True])
def test_dense_fallback_tapes_preserve_complete_selected_reverse(modules, whole_model):
    guarded, helper = modules
    values = zero_parameters(n=2)
    values["EMBED"][0, 0] = 1.0
    if whole_model:
        values["MP"][0, 0, 1] = .125
    tokens = torch.tensor([0, 1, 0, 1, 0])
    actual, expected = compare_forward(modules, values, tokens, off_weight=-1.0)
    assert actual["guard_diagnostics"]["whole_model_dense_fallback"] is whole_model
    cotangent = torch.tensor([[1., -2.], [-3., 4.], [5., -6.], [-7., 8.], [9., -10.]], dtype=torch.float64)
    direct = helper.reference.manual_vjp(values, tokens, cotangent, actual)
    exact(direct, helper.reference.manual_vjp(values, tokens, cotangent, expected))
    leaves = {name: value.clone().requires_grad_(True) for name, value in values.items()}
    replay = helper.reference.forward(leaves, tokens, mode="fixed_routes", base=actual)
    checked = torch.autograd.grad(replay["logits"], tuple(leaves.values()), grad_outputs=cotangent)
    for name, gradient in zip(leaves, checked):
        torch.testing.assert_close(direct[name], gradient, atol=1e-12, rtol=1e-11)


def test_unsupported_structure_delegates_to_frozen_whole_model(modules, monkeypatch):
    guarded, helper = modules
    values = parameters(guarded.OFF_WEIGHT)
    values["W_HEAD"][0].zero_()
    tokens = torch.tensor([0, 1, 2])
    original, calls = helper.forward, []

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(helper, "forward", wrapped)
    actual = guarded.forward(values, tokens, context=False)
    assert len(calls) == 1 and calls[0][1] == {"context": False}
    exact(helper.trace(actual), helper.trace(original(values, tokens, context=False)))


def test_selected_weight_need_not_exceed_the_constant_background(modules):
    values = zero_parameters()
    values["MP"][0, 0, 3] = -2.0
    actual, _ = compare_forward(modules, values, torch.tensor([0]), off_weight=-1.0)
    assert actual["guard_diagnostics"]["structure_certified"] is True
    assert actual["guard_diagnostics"]["projection_dense_operations"] == 1
    assert actual["routes"]["PROJ_IDX"][0, 0, 0].item() == 0


def test_finite_inputs_with_overflowing_center_do_not_hide_dense_failure(modules):
    guarded, helper = modules
    values = parameters(guarded.OFF_WEIGHT)
    values["EMBED"].fill_(torch.finfo(torch.float64).max)
    before = clones(values)
    for forward in (guarded.forward, helper.forward):
        with pytest.raises(ValueError):
            forward(values, torch.tensor([0]))
        exact(values, before)


def test_source_identity_binds_helper_tests_and_frozen_dependencies(modules):
    guarded, helper = modules
    sources = guarded.source_identity()
    root = Path(__file__).resolve().parents[1]
    for name, digest in helper.source_identity().items():
        assert sources[name] == digest
    for name in ("manuscript/tools/fpga_lm_guarded_forward.py", "tests/test_fpga_lm_guarded_forward.py"):
        assert sources[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
