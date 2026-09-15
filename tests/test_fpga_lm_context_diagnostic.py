"""No native execution: input transforms and observation-boundary comparisons."""
import pytest

from manuscript.tools import diagnose_fpga_lm_context as diagnostic


@pytest.mark.parametrize("variant,expected", [("original", [1, 2, 3]),
                                              ("context_disabled", [1, 2, 3]),
                                              ("all_space", [32, 32, 32]),
                                              ("reversed_tokens", [3, 2, 1])])
def test_transform_is_deterministic_and_does_not_mutate(variant, expected):
    source = [1, 2, 3]
    assert diagnostic.transformed(source, variant) == expected
    assert source == [1, 2, 3]


def test_comparison_reports_tested_outputs_not_unobserved_logits():
    left = {"predictions": [0, 1], "gradients": [3, -3, 2, -2], "lut_normalized_nll": 2.0}
    right = {"predictions": [1, 1], "gradients": [4, -4, 2, -2], "lut_normalized_nll": 2.5}
    assert diagnostic.compare(left, right) == {
        "prediction_mismatches": 1, "rounded_gradient_mismatches": 2,
        "rounded_gradient_max_abs_difference": 1, "lut_nll_difference": 0.5}


def test_array_digest_binds_order():
    assert diagnostic.array_digest([1, 2]) != diagnostic.array_digest([2, 1])


def test_failed_run_rejected_before_loading_checkpoint(tmp_path):
    (tmp_path / "failure.json").write_text('{"complete": false}')
    with pytest.raises(ValueError, match="failed run"):
        diagnostic.diagnose(tmp_path)


def test_nonfinite_run_json_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text('{"value": NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        diagnostic.diagnose(tmp_path)


def test_shared_head_route_exactly_cancels_balanced_gradient():
    shape = diagnostic.pilot.layout(16)
    weights = {name: [0] * getattr(shape, "WORDS_" + name) for name in diagnostic.pilot.WEIGHTS}
    reference = diagnostic.pilot.integer_forward(weights, [0] * 128, [1] * 128, shape)
    observation = diagnostic.head_route_summary(weights, [0] * 128, [1] * 128, shape, reference)
    assert observation["single_shared_head_winner_positions"] == 128
    assert observation["exact_zero_head_input_vjp_positions"] == 128
    assert observation["max_abs_head_input_vjp"] == 0


@pytest.mark.parametrize("candidate", [{"predictions": [], "gradients": []},
                                       {"predictions": [0], "gradients": [1, -1]},
                                       {"predictions": [0, 1], "gradients": [1]}])
def test_incomplete_observations_rejected(candidate):
    original = {"predictions": [0, 1], "gradients": [1, -1], "lut_normalized_nll": 1.0}
    with pytest.raises(ValueError, match="complete nonempty"):
        diagnostic.compare(original, candidate)


def test_mixed_head_routes_retain_nonzero_hidden_gradient():
    shape = diagnostic.pilot.layout(16)
    weights = {name: [0] * getattr(shape, "WORDS_" + name) for name in diagnostic.pilot.WEIGHTS}
    for row in range(128):
        weights["W_HEAD"][row * 16 + (0 if row < 64 else 1)] = 10
    reference = diagnostic.pilot.integer_forward(weights, [0] * 128, [1] * 128, shape)
    observation = diagnostic.head_route_summary(weights, [0] * 128, [1] * 128, shape, reference)
    assert observation["single_shared_head_winner_positions"] == 0
    assert observation["unique_head_winners_min"] == observation["unique_head_winners_max"] == 2
    assert observation["exact_zero_head_input_vjp_positions"] == 0
    assert observation["max_abs_head_input_vjp"] == 32768
    reference["gradients"][0] += 1
    with pytest.raises(ValueError, match="LUT gradient mismatch"):
        diagnostic.head_route_summary(weights, [0] * 128, [1] * 128, shape, reference)
