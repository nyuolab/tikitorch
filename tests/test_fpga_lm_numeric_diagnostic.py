"""Synthetic arithmetic tests only: no corpus, native execution, or training study."""
import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from manuscript.tools import fpga_lm_numeric_diagnostic as numeric  # noqa: E402


def tensor(values, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def scalar_head(row, target):
    beta = math.log(2) / 64
    top = max(row)
    exact = [math.exp(beta * (value - top)) for value in row]
    ce = math.log(sum(exact)) - beta * (row[target] - top)
    energies = [round(65536 * 2 ** (-min(math.floor((top - value) / 4), 255) / 16))
                for value in row]
    total = sum(energies)
    shift = 0
    while total >> shift > 65536:
        shift += 1
    rounded = [value >> shift for value in energies]
    mass = sum(rounded)
    native = rounded.copy()
    native[target] -= mass
    return ce, math.log(total) - math.log(energies[target]), energies, shift, rounded, native


@pytest.mark.parametrize("vocab", [1, 4, 17, 128])
@pytest.mark.parametrize("variant", numeric.LOSS_VARIANTS)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_head_independent_scalar_rom_oracle_and_backward(vocab, variant, reduction):
    rng = random.Random(87)
    rows = [[rng.randint(-2000, 2000) / 2 for _ in range(vocab)] for _ in range(3)]
    targets = [rng.randrange(vocab) for _ in rows]
    logits = tensor(rows, grad=True)
    result = numeric.loss_head(logits, targets, variant, reduction=reduction)
    oracle = [scalar_head(row, target) for row, target in zip(rows, targets)]
    assert result["energies"].tolist() == [item[2] for item in oracle]
    assert result["normalization_shifts"].tolist() == [item[3] for item in oracle]
    assert result["rounded_energies"].tolist() == [item[4] for item in oracle]
    assert result["native_gradient"].tolist() == [item[5] for item in oracle]
    np.testing.assert_allclose(result["true_ce_per_token"].numpy(), [item[0] for item in oracle], atol=1e-14)
    np.testing.assert_allclose(result["lut_nll_per_token"].numpy(), [item[1] for item in oracle], atol=1e-14)
    assert result["loss"].item() == result["true_ce"].item()
    result["loss"].backward()
    expected = result["supplied_gradient"] / (3 if reduction == "mean" else 1)
    torch.testing.assert_close(logits.grad, expected, rtol=1e-13, atol=1e-17)


def test_true_ce_gradient_finite_difference():
    logits = tensor([[0.3, -2, 1], [1, -1, 0.5]], grad=True)
    value = numeric.loss_head(logits, [0, 2], "exact_ce")["loss"]
    value.backward()
    for row in range(2):
        for column in range(3):
            plus, minus = logits.detach().clone(), logits.detach().clone()
            plus[row, column] += 1e-5
            minus[row, column] -= 1e-5
            delta = (numeric.loss_head(plus, [0, 2])["loss"] - numeric.loss_head(minus, [0, 2])["loss"]) / 2e-5
            assert delta.item() == pytest.approx(logits.grad[row, column].item(), abs=3e-11)


def test_mass_factor_changes_scale_not_truncated_normalized_direction():
    logits = tensor([[0, -1, -1000, -1800], [0, -25, -37, -60]])
    normalized = numeric.loss_head(logits, [3, 1], "truncated_normalized")
    variable = numeric.loss_head(logits, [3, 1], "variable_mass")
    torch.testing.assert_close(variable["supplied_gradient"],
                               normalized["supplied_gradient"] * variable["mass_factor"][:, None])
    assert variable["mass_factor"][0] != variable["mass_factor"][1]
    assert torch.all(variable["mass_factor"] > 0)
    assert torch.all(variable["mass_factor"] <= 1)
    assert variable["native_gradient"].sum(dim=1).tolist() == [0, 0]
    assert variable["rounded_zero_target_mask"].tolist() == [True, False]
    assert variable["native_gradient"][0, 3] < 0
    assert math.isfinite(variable["lut_nll_mean"].item())


def test_prediction_not_lut_tied_argmax_and_last_bin_boundary():
    result = numeric.loss_head(tensor([[1, 2, -1018], [0, -1024, -1020]]), [2, 2])
    assert result["predictions"].tolist() == [1, 0]
    assert result["target_at_last_bin_mask"].tolist() == [True, True]
    assert result["target_above_last_bin_mask"].tolist() == [False, False]
    assert numeric.loss_head(tensor([[0, -1024]]), [1])["target_above_last_bin_mask"].item()


def test_shift_invariance_and_global_gradient_unit_conversion():
    logits = tensor([[0, -20, -300], [42, 50, -2]])
    for variant in numeric.LOSS_VARIANTS:
        original = numeric.loss_head(logits, [1, 0], variant)
        translated = numeric.loss_head(logits + 1234, [1, 0], variant)
        torch.testing.assert_close(original["supplied_gradient"], translated["supplied_gradient"])
    for reduction, divisor in (("sum", 1), ("mean", 2)):
        result = numeric.loss_head(logits, [1, 0], "variable_mass", reduction=reduction)
        assert result["native_gradient_scale"] == numeric.BETA / (65536 * divisor)
        for k in (0, 18, 30):
            eta = numeric.nominal_raw_sgd_learning_rate(k) * divisor
            torch.testing.assert_close(eta * result["supplied_gradient"] / divisor,
                                       result["native_gradient"].double() / (1 << k))


@pytest.mark.parametrize("logits,targets", [([], []), ([[float("nan"), 0]], [0]),
    ([[float("inf"), 0]], [0]), ([[0, 1]], [True]), ([[0, 1]], [0.0]),
    ([[0, 1]], [-1]), ([[0, 1]], [2]), ([[0, 1]], []), ([[0, 1]], [[0]]),
    ([[0, 1], [1, 0]], [0]), ([0, 1], [0])])
def test_head_rejects_invalid_inputs(logits, targets):
    with pytest.raises(ValueError):
        numeric.loss_head(tensor(logits), targets)


def test_head_rejects_unknown_recipe_dtype_zero_rounded_distribution_and_overflow():
    with pytest.raises(ValueError):
        numeric.loss_head(torch.zeros((1, 2)), [0])
    with pytest.raises(ValueError):
        numeric.loss_head(tensor([[0, 1]]), [0], "softmax_magic")
    with pytest.raises(ValueError):
        numeric.loss_head(tensor([[0, 1]]), [0], reduction="none")
    with pytest.raises(ValueError, match="zero total mass"):
        numeric.loss_head(torch.zeros((1, 65537), dtype=torch.float64), [0])
    with pytest.raises(OverflowError):
        numeric.loss_head(tensor([[1e308, -1e308]]), [0])


@pytest.mark.parametrize("mode", numeric.CENTER_MODES)
def test_center_forward_and_custom_backward(mode):
    value = tensor([[0, 0, -1], [2, -1, 0]], grad=True)
    upstream = tensor([[1, 0, 0], [-2, 1, -1]])
    output = numeric.center(value, mode)
    mean = value.detach().mean(dim=-1, keepdim=True)
    expected = value.detach() - (mean if mode == "exact" else mean.floor())
    torch.testing.assert_close(output, expected)
    output.backward(upstream)
    torch.testing.assert_close(value.grad, numeric.center_vjp(upstream, mode))


def test_floor_ste_changes_only_forward_and_legacy_is_not_linear():
    value = tensor([[0, 0, -1]])
    assert not torch.equal(numeric.center(value, "exact"), numeric.center(value, "floor_ste"))
    gradient = tensor([[0.5, 0]])
    torch.testing.assert_close(numeric.center_vjp(gradient, "exact"), numeric.center_vjp(gradient, "floor_ste"))
    floor = numeric.center_vjp(gradient, "legacy_floor")
    assert not torch.equal(4 * floor, numeric.center_vjp(4 * gradient, "legacy_floor"))
    assert floor.sum().item() != 0
    assert numeric.center_vjp(gradient, "exact").sum().item() == 0


def test_native_unit_floor_and_integer_raw_negative_sum():
    raw = np.asarray([[1, 0, 0, 0], [-3, 0, 0, 0]], dtype=np.int64)
    expected = np.asarray([[1, 0, 0, 0], [-2, 1, 1, 1]])
    np.testing.assert_array_equal(numeric.legacy_center_vjp_raw(raw), expected)
    for scale in (1.0, 0.125, numeric.BETA / 65536, numeric.BETA / (65536 * 128)):
        result = numeric.center_vjp(tensor(raw) * scale, "legacy_floor", gradient_scale=scale)
        torch.testing.assert_close(result, tensor(expected) * scale)


@pytest.mark.parametrize("scale", [True, 0, -1, float("nan"), float("inf")])
def test_center_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        numeric.center(tensor([[1, 2]]), "legacy_floor", gradient_scale=scale)


@pytest.mark.parametrize("raw", [1, [[True, 1]], [[1.0, 0]], [[2**31, 0]]])
def test_integer_center_rejects_invalid_inputs(raw):
    with pytest.raises(ValueError):
        numeric.legacy_center_vjp_raw(raw)


def test_integer_center_rejects_sum_and_output_overflow():
    with pytest.raises(OverflowError):
        numeric.legacy_center_vjp_raw([[2**31 - 1, 1]])
    with pytest.raises(OverflowError):
        numeric.legacy_center_vjp_raw([[-2**31, 2**31 - 1, 2**31 - 1]])
    with pytest.raises(OverflowError, match="ordered sum"):
        numeric.legacy_center_vjp_raw([[2**31 - 1, 1, -1]])


def test_floor_requires_raw_units_before_scaling_for_bitwise_integer_parity():
    raw = tensor([[-65534, -65534]])
    scale = numeric.BETA / 65536
    raw_first = numeric.center_vjp(raw, "legacy_floor") * scale
    scaled_first = numeric.center_vjp(raw * scale, "legacy_floor", gradient_scale=scale)
    assert torch.equal(raw_first, tensor([[0, 0]]))
    # This deliberately records why a scaled callback is not a native parity gate.
    assert not torch.equal(raw_first, scaled_first)


@pytest.mark.parametrize("k", [0, 1, 4, 18, 30])
def test_exact_ef_scalar_and_array_oracle_conservation(k):
    rng = random.Random(k)
    denominator = 1 << k
    weights = np.asarray([rng.randint(-100, 100) for _ in range(40)])
    rem = np.asarray([rng.randrange(-denominator + 1, denominator) for _ in weights])
    grad = np.asarray([rng.randint(-1000, 1000) for _ in weights])
    old = (weights.copy(), rem.copy(), grad.copy())
    result = numeric.ef_rounding(weights, rem, grad, k)
    expected = []
    for weight, remainder, gradient in zip(weights.tolist(), rem.tolist(), grad.tolist()):
        accumulator = remainder + gradient
        quotient = abs(accumulator) // denominator * (-1 if accumulator < 0 else 1)
        expected.append((weight - quotient, accumulator - quotient * denominator, quotient))
    np.testing.assert_array_equal(result["weights"], [row[0] for row in expected])
    np.testing.assert_array_equal(result["remainders"], [row[1] for row in expected])
    np.testing.assert_array_equal(result["quotient"], [row[2] for row in expected])
    np.testing.assert_array_equal(result["integer_identity_residual"], np.zeros(40))
    np.testing.assert_array_equal(result["weights"] - weights, result["unrounded_delta"] + result["rounding_delta"])
    np.testing.assert_array_equal(result["shadow_after"], result["shadow_before"] + result["unrounded_delta"])
    for before, after in zip(old, (weights, rem, grad)):
        np.testing.assert_array_equal(before, after)
    scalar = numeric.ef_rounding(0, 0, -9, 3)
    assert scalar["weights"] == 1 and scalar["remainders"] == -1


def test_ef_cumulative_exact_identity_and_state_resume():
    gradients = [3, -1, -7, 2, -6, 4, 9, -13]
    w, r = 0, 0
    for index, gradient in enumerate(gradients):
        state = numeric.ef_rounding(w, r, gradient, 3)
        w, r = int(state["weights"]), int(state["remainders"])
        assert -w * 8 + r == sum(gradients[:index + 1])
    assert numeric.ef_rounding(0, 0, -(1 << 31), 30)["weights"] == 2


def test_ef_rounding_adjustment_can_exceed_one_when_remainder_changes_sign():
    result = numeric.ef_rounding(0, 7, -14, 3)
    assert result["weights"] == 0 and result["remainders"] == -7
    assert result["rounding_delta"] == -1.75
    assert result["unrounded_delta"] == 1.75


@pytest.mark.parametrize("args,error", [((0, 0, 1, True), ValueError), ((0, 0, 1, -1), ValueError),
    ((0, 0, 1, 31), ValueError), ((0, 0, 1, 1.0), ValueError), ((0, 2, 1, 1), ValueError),
    ((0, -2, 1, 1), ValueError), ((False, 0, 1, 1), ValueError),
    (([0, 0], [0], [1], 1), ValueError), (([0, True], [0, 0], [1, 1], 1), ValueError),
    ((0, (1 << 30) - 1, (1 << 31) - 1, 30), OverflowError),
    ((0, -(1 << 30) + 1, -(1 << 31), 30), OverflowError),
    ((32767, 0, -1, 0), OverflowError), ((-32768, 0, 1, 0), OverflowError)])
def test_ef_rejects_invalid_state_overflow_or_clipping(args, error):
    with pytest.raises(error):
        numeric.ef_rounding(*args)


def test_smoothed_count_fit_stationarity_depends_on_objective():
    result = numeric.smoothing_stationarity([9, 1, 0], 1)
    np.testing.assert_allclose(result["smoothed_probabilities"], np.asarray([10, 2, 1]) / 13)
    np.testing.assert_allclose(result["unsmoothed_expected_gradient"],
                               numeric.BETA * (np.asarray([10, 2, 1]) / 13 - np.asarray([.9, .1, 0])))
    assert result["unsmoothed_gradient_l1"] > 0
    assert np.all(result["smoothed_expected_gradient"] == 0)
    assert numeric.smoothing_stationarity([4, 4, 4])["unsmoothed_gradient_l1"] == 0
    np.testing.assert_allclose(result["unsmoothed_expected_gradient"].sum(), 0, atol=1e-18)


@pytest.mark.parametrize("counts,alpha", [([0, 0], 1), ([-1, 2], 1), ([True, 2], 1),
    ([1, 2], True), ([1, 2], 0), ([1, 2], float("nan")), ([1, 2], float("inf"))])
def test_smoothing_rejects_invalid(counts, alpha):
    with pytest.raises(ValueError):
        numeric.smoothing_stationarity(counts, alpha)


def test_duplicate_bias_fixed_task_exact_double_step_not_universal_failure():
    safe = numeric.duplicate_bias_step(0, 0, 1, 0.25)
    assert safe["sum_after"] == safe["double_rate_single_after"] == 0.5
    assert safe["single_parameter_after"] == 0.25
    assert safe["loss_after"] < safe["loss_before"]
    overshoot = numeric.duplicate_bias_step(0, 0, 1, 1.1)
    assert overshoot["loss_after"] > overshoot["loss_before"]
    assert 0.5 * (overshoot["single_parameter_after"] - 1) ** 2 < overshoot["loss_before"]


def test_recipe_scope_and_source_binding():
    recipe = numeric.recipe()
    assert recipe["beta"] == math.log(2) / 64
    assert recipe["lut_shift"] == 2
    assert recipe["training_performed"] is False and recipe["native_execution"] is False
    assert recipe["cloud_operations"] is False
    identity = numeric.source_identity()
    assert list(identity) == ["manuscript/tools/fpga_lm_numeric_diagnostic.py"]
    assert len(next(iter(identity.values()))) == 64
