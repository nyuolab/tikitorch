"""Synthetic contracts for a distribution-preserving projection initializer.

No corpus is read or fitted here. The native test evaluates synthetic windows
and one synthetic optimizer unit step, never a scientific training experiment.
"""

import importlib
import math
import shutil

import numpy as np
import pytest

from manuscript.tools import check_fpga_lm_bigram_capacity as capacity
from manuscript.tools import fpga_lm_efsgd as optimizer
from manuscript.tools.observe_fpga_lm_routes import observe
from scripts import fpga_lm_pretrain as pilot


@pytest.fixture
def geometry():
    return importlib.import_module("manuscript.tools.fpga_lm_diverse_projection")


@pytest.fixture(scope="module")
def synthetic_case():
    shape = pilot.layout(512)
    counts = [[1 + ((token * 17 + target * 3) % 13) for target in range(128)]
              for token in range(128)]
    weights, facts = capacity.construct_weights(counts, shape, 2)
    return shape, weights, facts


def expected_mapping():
    anchors = [(5 * value) % 512 for value in range(128)]
    unused = sorted(set(range(512)) - set(anchors))
    groups = [unused[start:start + 128] for start in (0, 128, 256)]
    target = [-1] * 512
    for a, b, c, d in zip(anchors, *groups):
        target[a], target[b], target[c], target[d] = b, c, d, b
    return {"A": anchors, "B": groups[0], "C": groups[1], "D": groups[2], "target_by_row": target}


def test_map_is_exact_balanced_chain_on_distinct_unused_coordinates(geometry, synthetic_case):
    shape, _, _ = synthetic_case
    actual, expected = geometry.mapping(shape), expected_mapping()
    for key, value in expected.items():
        assert actual[key] == value
    groups = [actual[key] for key in ("A", "B", "C", "D")]
    assert sorted(sum(groups, [])) == list(range(512))
    assert set(actual["target_by_row"]) == set(sum(groups[1:], []))
    preimages = np.bincount(actual["target_by_row"], minlength=512)
    assert np.count_nonzero(preimages == 2) == 128
    assert np.count_nonzero(preimages == 1) == 256
    assert np.count_nonzero(preimages == 0) == 128
    actual["target_by_row"][0] = -1
    assert geometry.mapping(shape)["target_by_row"] == expected["target_by_row"]


def test_transform_changes_only_mp_and_never_mutates_or_aliases_input(geometry, synthetic_case):
    shape, weights, _ = synthetic_case
    before = pilot.weight_bytes(weights)
    transformed, _ = geometry.transform(weights, shape)
    assert pilot.weight_bytes(weights) == before
    target = expected_mapping()["target_by_row"]
    matrices = np.asarray(transformed["MP"]).reshape(4, 512, 512)
    expected = np.full((512, 512), -8192, dtype=np.int32)
    expected[np.arange(512), target] = 0
    for layer in matrices:
        assert np.array_equal(layer, expected)
    for name in pilot.WEIGHTS:
        assert transformed[name] is not weights[name]
        if name != "MP":
            assert transformed[name] == weights[name]
    transformed["EMBED"][0] += 4
    assert pilot.weight_bytes(weights) == before


def test_initial_forward_bounds_are_finite_and_strict(geometry, synthetic_case):
    shape, weights, facts = synthetic_case
    transformed, metadata = geometry.transform(weights, shape)
    assert isinstance(metadata, dict)
    bound = facts["maximum_embedding_magnitude"]
    common = math.ceil(128 * bound / 512)
    assert geometry.MAX_MAGNITUDE == 756
    assert bound <= 756 and common <= 189
    assert metadata["initial_upper_increment_bound"] == common
    assert metadata["initial_residual_lower_bound"] == -bound
    assert metadata["initial_residual_upper_bound"] == 4 * common
    assert metadata["initial_head_margin_lower_bound"] == 8192 - bound
    assert metadata["initial_l1_preserved_projection_layers"] == [3, 2, 1]
    assert metadata["bottom_projection_may_merge_A_and_D"] is True
    assert metadata["past_context_use_guaranteed"] is False
    assert -bound > -32768 and 4 * common < 32767
    assert 8192 - bound > 0
    assert all(-32768 <= value <= 32767 for region in transformed.values() for value in region)
    # The mathematical bound does not depend on a heldout window or its labels.
    for token in (0, 1, 7, 127):
        row = weights["EMBED"][token * 512:(token + 1) * 512]
        mean = sum(row) // 512
        assert 0 <= -mean <= common
        centered = [value - mean for value in row]
        assert min(centered) >= -bound and max(centered) <= common


@pytest.mark.parametrize("constant", [False, True])
def test_synthetic_forward_preserved_and_top_three_projection_vjps_do_not_cancel(geometry, synthetic_case, constant):
    shape, weights, _ = synthetic_case
    transformed, _ = geometry.transform(weights, shape)
    tokens = [7] * 128 if constant else [(index * 7 + index // 3) % 128 for index in range(128)]
    targets = tokens[1:] + [3]
    original = observe(weights, tokens, targets, shape, zshift=2)
    modified = observe(transformed, tokens, targets, shape, zshift=2)
    for key in ("predictions", "gradients", "accuracy", "lut_normalized_nll",
                "saturation_events", "rounded_zero_target_mass"):
        assert modified[key] == original[key]
    assert modified["saturation_events"] == 0
    for name in ("AD_HEAD", "TAP_IDX"):
        assert modified["native_routes"][name] == original["native_routes"][name]
    target = expected_mapping()["target_by_row"]
    assert modified["native_routes"]["PROJ_IDX"] == target * (4 * 128)
    routes = modified["route_diagnostics"]
    assert routes["exact_zero_head_input_vjp_on_nonzero_gradient_positions"] == 0
    assert routes["nonzero_logit_gradient_positions"] == 128
    for layer in routes["projection_layers"]:
        assert layer["nonzero_centering_remainder_positions"] == 0
        if layer["layer"] in (1, 2, 3):
            assert layer["projection_incoming_l1"] > 0
            assert layer["projection_incoming_l1"] == layer["projection_scattered_l1"]
            assert layer["zero_projection_input_vjp_on_nonzero_incoming_positions"] == 0
        if constant:
            assert layer["nonzero_temporal_carry_positions"] == 0
    if constant:
        assert not any(modified["native_routes"]["TAP_IDX"])


def test_bottom_projection_has_a_null_direction_so_global_l1_preservation_is_not_claimed():
    """Algebraic operator example, not a claim this cotangent occurs in data."""
    mapping = expected_mapping()
    z = np.zeros(128, dtype=np.int64)
    z[0], z[1] = 7, -7
    incoming = np.zeros(512, dtype=np.int64)
    incoming[mapping["A"]], incoming[mapping["D"]] = z, -z
    scattered = np.zeros(512, dtype=np.int64)
    np.add.at(scattered, mapping["target_by_row"], incoming)
    assert incoming.sum() == 0 and np.abs(incoming).sum() == 28
    assert not np.any(scattered)


@pytest.mark.parametrize("active_groups", [("A",), ("A", "B"), ("A", "B", "C")])
def test_map_preserves_l1_for_every_coordinate_on_each_proven_support(active_groups):
    mapping = expected_mapping()
    support = sum((mapping[name] for name in active_groups), [])
    destinations = [mapping["target_by_row"][index] for index in support]
    assert len(set(destinations)) == len(support)
    incoming = np.zeros(512, dtype=np.int64)
    for index, coordinate in enumerate(support):
        incoming[coordinate] = (index + 1) * (1 if index % 2 else -1)
    scattered = np.zeros(512, dtype=np.int64)
    np.add.at(scattered, mapping["target_by_row"], incoming)
    assert np.abs(scattered).sum() == np.abs(incoming).sum()


@pytest.mark.parametrize("region", ["MH", "MP", "MB"])
def test_transform_rejects_nonzero_fitted_zero_regions(geometry, synthetic_case, region):
    shape, original, _ = synthetic_case
    weights = {name: list(values) for name, values in original.items()}
    weights[region][-1] = 1
    before = pilot.weight_bytes(weights)
    with pytest.raises(ValueError):
        geometry.transform(weights, shape)
    assert pilot.weight_bytes(weights) == before


@pytest.mark.parametrize("mutation", ["unused_embedding", "positive_anchor", "off_grid_anchor",
                                       "unbounded_anchor", "head_anchor", "head_other"])
def test_transform_requires_the_exact_fitted_structure_and_safe_bounds(geometry, synthetic_case, mutation):
    shape, original, _ = synthetic_case
    weights = {name: list(values) for name, values in original.items()}
    mapping = expected_mapping()
    if mutation == "unused_embedding":
        weights["EMBED"][mapping["B"][0]] = -4
    elif mutation == "positive_anchor":
        weights["EMBED"][mapping["A"][0]] = 4
    elif mutation == "off_grid_anchor":
        weights["EMBED"][mapping["A"][0]] = -1
    elif mutation == "unbounded_anchor":
        weights["EMBED"][mapping["A"][0]] = -760
    elif mutation == "head_anchor":
        weights["W_HEAD"][mapping["A"][0]] = -1
    else:
        weights["W_HEAD"][mapping["B"][0]] = -8191
    with pytest.raises(ValueError):
        geometry.transform(weights, shape)


@pytest.mark.parametrize("value", [True, False, 0.0, "0", None])
def test_transform_rejects_boolean_or_noninteger_weight_values(geometry, synthetic_case, value):
    shape, original, _ = synthetic_case
    weights = {name: list(values) for name, values in original.items()}
    weights["MP"][0] = value
    with pytest.raises(ValueError):
        geometry.transform(weights, shape)


@pytest.mark.parametrize("region", pilot.WEIGHTS)
def test_transform_requires_complete_exact_parameter_regions(geometry, synthetic_case, region):
    shape, original, _ = synthetic_case
    weights = {name: list(values) for name, values in original.items()}
    weights[region].pop()
    with pytest.raises(ValueError):
        geometry.transform(weights, shape)


def test_exact_bound_is_accepted_but_loss_of_row_zero_is_not(geometry, synthetic_case):
    shape, original, _ = synthetic_case
    weights = {name: list(values) for name, values in original.items()}
    banks = expected_mapping()
    for token in range(128):
        for column in banks["A"]:
            weights["EMBED"][token * 512 + column] = -756
        weights["EMBED"][token * 512 + banks["A"][-1]] = 0
    changed, facts = geometry.transform(weights, shape)
    assert facts["maximum_embedding_magnitude"] == 756
    assert facts["initial_upper_increment_bound"] == 189
    assert facts["initial_head_margin_lower_bound"] == 7436
    result = observe(changed, list(range(128)), list(range(1, 128)) + [0], shape, zshift=2)
    assert result["saturation_events"] == 0
    assert result["route_diagnostics"]["backward_int32_overflow_detected"] is False
    weights["EMBED"][banks["A"][-1]] = -4
    with pytest.raises(ValueError):
        geometry.transform(weights, shape)


@pytest.mark.parametrize("width", [16, 128, 256, 1024])
def test_mapping_rejects_undeclared_layout(geometry, width):
    with pytest.raises(ValueError):
        geometry.mapping(pilot.layout(width))


def test_equivalence_helper_checks_complete_initial_outputs_and_narrow_scope(geometry, synthetic_case):
    shape, original, _ = synthetic_case
    transformed, _ = geometry.transform(original, shape)
    tokens = [(index * 7) % 128 for index in range(128)]
    result = geometry.check_initial_equivalence(original, transformed, tokens, tokens[1:] + [3], shape)
    for key in ("initial_logits_equal", "initial_predictions_gradients_equal",
                "initial_head_tap_routes_equal", "initial_projection_routes_equal_mapping",
                "initial_upper_three_projection_l1_preserved"):
        assert result[key] is True
    assert result["source_weights_sha256"] == pilot.digest(pilot.weight_bytes(original))
    assert result["transformed_weights_sha256"] == pilot.digest(pilot.weight_bytes(transformed))
    assert "initialization only" in result["scope"]


@pytest.mark.parametrize("mutation", ["prediction", "gradient", "head", "tap", "projection",
                                       "l1", "saturation", "overflow"])
def test_equivalence_helper_rejects_observation_mismatch(geometry, synthetic_case, mutation):
    shape, original, _ = synthetic_case
    transformed, _ = geometry.transform(original, shape)
    tokens = [7] * 128
    calls = []

    def corrupted(weights, *args, **kwargs):
        result = observe(weights, *args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            if mutation == "prediction":
                result["predictions"][0] ^= 1
            elif mutation == "gradient":
                result["gradients"][0] += 1
            elif mutation in ("head", "tap", "projection"):
                name = {"head": "AD_HEAD", "tap": "TAP_IDX", "projection": "PROJ_IDX"}[mutation]
                result["native_routes"][name][0] += 1
            elif mutation == "l1":
                result["route_diagnostics"]["projection_layers"][3]["projection_scattered_l1"] += 1
            elif mutation == "saturation":
                result["saturation_events"] = True
            else:
                result["route_diagnostics"]["backward_int32_overflow_detected"] = True
        return result

    with pytest.raises(ValueError):
        geometry.check_initial_equivalence(original, transformed, tokens, tokens, shape, observer=corrupted)


def test_native_synthetic_forward_routes_and_complete_state_match_frozen_observer(geometry, synthetic_case, tmp_path):
    if not (shutil.which("clang++") or shutil.which("g++")):
        pytest.skip("native synthetic check needs a C++17 compiler")
    shape, original, _ = synthetic_case
    transformed, _ = geometry.transform(original, shape)
    executable = optimizer.build(tmp_path / "native", width=512)
    tokens = [(index * 7 + index // 3) % 128 for index in range(128)]
    targets = tokens[1:] + [3]
    results = []
    for index, weights in enumerate((original, transformed)):
        initial = shape.build_init(0)
        initial[shape.OFF_CFG:shape.OFF_CFG + 2] = [1, 2]
        for name, values in weights.items():
            offset = getattr(shape, "OFF_" + name)
            initial[offset:offset + len(values)] = values
        expected = observe(weights, tokens, targets, shape, zshift=2,
                           include_parameter_gradients=(index == 1))
        with optimizer.EFSGD(executable, shape, initial, 18, timeout_s=60,
                             log_path=tmp_path / ("native_%d.log" % index)) as native:
            before = native.export_state()
            actual = native.evaluate(tokens, targets)
            assert actual["step"] == 0
            assert native.export_state() == before
            for name in ("predictions", "gradients", "native_routes"):
                assert actual[name] == expected[name]
            results.append(actual)
            if index == 1:
                # One synthetic optimizer unit step: independently apply the
                # exact signed truncation/remainder law to all 1,196,032 Gs.
                update = native.train(tokens, targets)
                assert update["step"] == 1
                for name in ("predictions", "gradients"):
                    assert update[name] == expected[name]
                assert update["native_routes"] == {}  # Training ABI does not transmit route arrays.
                after = optimizer.decode_state(native.export_state(), shape)
                assert after["step"] == 1
                checked = 0
                for name in pilot.WEIGHTS:
                    gradient = np.asarray(expected["parameter_gradients"][name], dtype=np.int64)
                    quotient = np.sign(gradient) * (np.abs(gradient) // (1 << 18))
                    remainder = gradient - quotient * (1 << 18)
                    candidate = np.asarray(weights[name], dtype=np.int64) - quotient
                    assert np.all(candidate >= -32768) and np.all(candidate <= 32767)
                    assert np.array_equal(after["weights"][name], candidate)
                    assert np.array_equal(after["remainders"][name], remainder)
                    assert np.all(np.abs(remainder) < (1 << 18))
                    checked += gradient.size
                assert checked == 1_196_032
    for name in ("predictions", "gradients"):
        assert results[0][name] == results[1][name]
    for name in ("AD_HEAD", "TAP_IDX"):
        assert results[0]["native_routes"][name] == results[1]["native_routes"][name]
    assert results[1]["native_routes"]["PROJ_IDX"] == expected_mapping()["target_by_row"] * (4 * 128)
