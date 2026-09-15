"""Pure topology/initialization tests; synthetic weights only, no corpus or training."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from manuscript.tools import fpga_lm_projection_topology as topology


@pytest.fixture(scope="module")
def fitted():
    shape = topology.pilot.layout(512)
    banks = topology.initializer.mapping(shape)
    weights = {name: np.zeros(getattr(shape, "WORDS_" + name), dtype=np.int64) for name in topology.pilot.WEIGHTS}
    embedding = weights["EMBED"].reshape(128, 512)
    for token in range(128):
        embedding[token, banks["A"]] = -4 * ((np.arange(128) * (token % 3 + 1)) % 190)
    head = weights["W_HEAD"].reshape(128, 512)
    head.fill(-8192)
    head[np.arange(128), banks["A"]] = 0
    original, _ = topology.initializer.transform(weights, shape)
    changed, facts = topology.transform(original, shape)
    return shape, original, changed, facts


def test_canonical_seed_and_global_rng_are_stable(fitted):
    shape, _, _, _ = fitted
    before = random.getstate()
    spec = topology.mapping(shape)
    assert random.getstate() == before
    sigma = list(range(128))
    random.Random(0).shuffle(sigma)
    assert spec["sigma"] == sigma
    for layer, permutation in enumerate(spec["class_permutations"]):
        assert sorted(permutation) == list(range(128))
        assert all(sigma[permutation[value]] == (sigma[value] + 2**layer) % 128 for value in range(128))
    assert len({tuple(row) for row in spec["target_by_layer"]}) == 4


def test_all_sixteen_paths_have_distinct_explicit_class_ancestors(fitted):
    shape, _, _, _ = fitted
    spec = topology.mapping(shape)
    certificate = topology.selected_ancestor_certificate(shape)
    topology.validate_certificate(certificate, shape)
    for head, record in enumerate(certificate["heads"]):
        leaves = []
        for choices in range(16):
            coordinate = spec["A"][head]
            for layer in range(3, -1, -1):
                if choices & (1 << layer):
                    coordinate = spec["target_by_layer"][layer][coordinate]
            leaves.append(coordinate)
        assert len(set(leaves)) == 16
        assert sorted(leaves) == record["coordinates"]
        assert [len(level["coordinates"]) for level in record["levels"]] == [1, 2, 4, 8, 16]
        assert len(record["original_coordinates"]) == 4
        assert set(leaves) & set(spec["A"]) == {spec["A"][head]}
    assert certificate["dense_mean_centering_path_excluded"] is True
    assert certificate["unused_bank_mixing_injects_other_fitted_anchor_features"] is False


def test_transform_preserves_non_mp_bytes_and_changes_only_declared_support(fitted):
    shape, original, changed, facts = fitted
    before = copy.deepcopy(original)
    repeated, repeated_facts = topology.transform(original, shape)
    assert repeated == changed and repeated_facts == facts and original == before
    for key in topology.pilot.WEIGHTS:
        if key != "MP":
            assert np.asarray(original[key], dtype="<i4").tobytes() == np.asarray(changed[key], dtype="<i4").tobytes()
            assert original[key] is not changed[key]
    projection = np.asarray(changed["MP"]).reshape(4, 512, 512)
    assert np.count_nonzero(projection == 0) == 4 * 512
    assert set(np.unique(projection)) == {-8192, 0}
    for layer in range(4):
        assert np.array_equal(projection[layer].argmax(axis=1), facts["mapping"]["target_by_layer"][layer])
    assert facts["source_weights_sha256"] != facts["transformed_weights_sha256"]


@pytest.mark.parametrize("tokens", [[7, 7, 7], [0, 1, 2, 7, 12]])
def test_synthetic_fitted_forward_and_selected_gradient_guarantees(fitted, tokens):
    shape, original, changed, facts = fitted
    tokens = [tokens[index % len(tokens)] for index in range(shape.ML_L_SEQ)]
    targets = [(token + 1) % 128 for token in tokens]
    old = topology.initializer.observe(original, tokens, targets, shape, zshift=2)
    new = topology.initializer.observe(changed, tokens, targets, shape, zshift=2)
    old_logits = topology.initializer.calibration.integer_logits(original, tokens, shape)
    new_logits = topology.initializer.calibration.integer_logits(changed, tokens, shape)
    assert old_logits == new_logits
    for key in ("predictions", "gradients", "lut_normalized_nll", "accuracy", "rounded_zero_target_mass"):
        assert old[key] == new[key]
    for key in ("AD_HEAD", "TAP_IDX"):
        assert old["native_routes"][key] == new["native_routes"][key]
    expected = [target for layer in facts["mapping"]["target_by_layer"] for _ in tokens for target in layer]
    assert new["native_routes"]["PROJ_IDX"] == expected
    for index in (3, 2, 1):
        layer = new["route_diagnostics"]["projection_layers"][index]
        assert layer["projection_incoming_l1"] == layer["projection_scattered_l1"]
    assert old["saturation_events"] == new["saturation_events"] == 0


@pytest.mark.parametrize("mutation", ["missing", "extra", "float", "bool", "short", "shape", "range", "mp", "zero_mp", "mh", "mb", "head", "unused", "positive", "multiple", "anchor_max", "magnitude"])
def test_rejects_noncanonical_or_trained_inputs(fitted, mutation):
    shape, original, _, _ = fitted
    weights = copy.deepcopy(original)
    banks = topology.mapping(shape)
    if mutation == "missing":
        del weights["MP"]
    elif mutation == "extra":
        weights["unused"] = [0]
    elif mutation == "float":
        weights["MB"] = np.asarray(weights["MB"], dtype=np.float64)
    elif mutation == "bool":
        weights["MB"][0] = False
    elif mutation == "short":
        weights["MH"].pop()
    elif mutation == "shape":
        weights["MH"] = np.asarray(weights["MH"]).reshape(4, 512, 7)
    elif mutation == "range":
        weights["MH"][0] = 32768
    elif mutation == "mp":
        weights["MP"][0] += 1
    elif mutation == "zero_mp":
        weights["MP"] = [0] * shape.WORDS_MP
    elif mutation in ("mh", "mb"):
        weights[mutation.upper()][0] = 1
    elif mutation == "head":
        weights["W_HEAD"][banks["A"][0]] = -1
    elif mutation == "unused":
        weights["EMBED"][banks["B"][0]] = -4
    elif mutation == "positive":
        weights["EMBED"][banks["A"][0]] = 4
    elif mutation == "multiple":
        weights["EMBED"][banks["A"][1]] = -3
    elif mutation == "anchor_max":
        weights["EMBED"][banks["A"][0]] = -4
    elif mutation == "magnitude":
        weights["EMBED"][banks["A"][1]] = -760
    with pytest.raises(ValueError):
        topology.transform(weights, shape)


@pytest.mark.parametrize("width", [16, 128, 256])
def test_rejects_other_layouts(width):
    with pytest.raises(ValueError):
        topology.mapping(topology.pilot.layout(width))


@pytest.mark.parametrize("field,value", [("seed", False), ("seed", 1), ("sigma", []), ("target_by_layer", []), ("schema", "other")])
def test_mapping_identity_is_literal_and_complete(fitted, field, value):
    shape = fitted[0]
    spec = topology.mapping(shape)
    spec[field] = value
    with pytest.raises(ValueError):
        topology.validate_mapping(spec, shape)


def test_certificate_rejects_count_and_scope_forgery(fitted):
    shape = fitted[0]
    for key, value in (("all_heads_new_count", 16.0), ("dense_mean_centering_path_excluded", False),
                       ("unused_bank_mixing_injects_other_fitted_anchor_features", True)):
        certificate = topology.selected_ancestor_certificate(shape)
        certificate[key] = value
        with pytest.raises(ValueError):
            topology.validate_certificate(certificate, shape)


def test_source_identity_includes_both_new_files():
    sources = topology.source_identity()
    for name in ("manuscript/tools/fpga_lm_projection_topology.py", "tests/test_fpga_lm_projection_topology.py"):
        assert sources[name] == topology.pilot.digest((topology.pilot.REPO / name).read_bytes())
