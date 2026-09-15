"""Initialization-only, function-preserving cross-bank projection topology.

The input must be the exact structural family of the frozen diverse fitted
initializer, not a trained checkpoint. Only MP changes. The selected-ancestor
certificate excludes the dense mean-centering path and does not assert that
unused channels initially contain other classes' fitted anchor features.
"""
from __future__ import annotations

import random

import numpy as np

from manuscript.tools import fpga_lm_diverse_projection as initializer

pilot = initializer.pilot
SCHEMA = "tikitorch.fpga_lm_projection_topology.v1"
SEED, MARGIN = 0, 8192


def source_identity():
    result = initializer.source_identity()
    for name in ("manuscript/tools/fpga_lm_projection_topology.py", "tests/test_fpga_lm_projection_topology.py"):
        result[name] = pilot.digest((pilot.REPO / name).read_bytes())
    return result


def _equal(actual, expected, label):
    if type(actual) is not type(expected):
        raise ValueError(label + " literal type differs")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(label + " fields differ")
        for key in expected:
            _equal(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(label + " length differs")
        for left, right in zip(actual, expected):
            _equal(left, right, label)
    elif actual != expected:
        raise ValueError(label + " value differs")


def mapping(shape):
    banks = initializer.mapping(shape)  # Exact D512/V128/N4/K7 layout guard.
    sigma = list(range(128))
    random.Random(SEED).shuffle(sigma)
    inverse = [0] * 128
    for index, value in enumerate(sigma):
        inverse[value] = index
    permutations, targets = [], []
    for layer in range(4):
        permutation = [inverse[(sigma[index] + (1 << layer)) % 128] for index in range(128)]
        target = [-1] * 512
        for source_bank, target_bank in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")):
            for index, source in enumerate(banks[source_bank]):
                target[source] = banks[target_bank][permutation[index]]
        permutations.append(permutation)
        targets.append(target)
    return {"schema": SCHEMA, "seed": SEED, "sigma": sigma, "inverse_sigma": inverse,
            "class_permutations": permutations, "target_by_layer": targets,
            "original_target_by_row": banks["target_by_row"],
            **{key: banks[key] for key in ("A", "B", "C", "D")},
            "formula": "p_l(v)=inverse_sigma[(sigma[v]+2**l)%128]"}


def validate_mapping(value, shape):
    _equal(value, mapping(shape), "canonical topology mapping")


def selected_ancestor_certificate(shape):
    spec = mapping(shape)
    bank_class = {coordinate: index for bank in ("A", "B", "C", "D")
                  for index, coordinate in enumerate(spec[bank])}
    records = []
    for head in range(128):
        current, original = {spec["A"][head]}, {spec["A"][head]}
        levels = [{"after_reverse_layers": 0, "coordinates": sorted(current)}]
        for count, layer in enumerate(range(3, -1, -1), start=1):
            current |= {spec["target_by_layer"][layer][index] for index in current}
            original |= {spec["original_target_by_row"][index] for index in original}
            levels.append({"after_reverse_layers": count, "coordinates": sorted(current)})
        classes = sorted(bank_class[index] for index in current)
        expected_classes = sorted(spec["inverse_sigma"][(spec["sigma"][head] + offset) % 128] for offset in range(16))
        if len(current) != 16 or len(original) != 4 or classes != expected_classes:
            raise ValueError("explicit selected-ancestor construction failed")
        if current & set(spec["A"]) != {spec["A"][head]}:
            raise ValueError("topology must not introduce a different fitted anchor")
        records.append({"head": head, "levels": levels, "coordinates": sorted(current),
                        "class_bank_labels": classes, "original_coordinates": sorted(original),
                        "original_count": 4, "new_count": 16,
                        "fitted_anchor_coordinates": [spec["A"][head]]})
    return {"schema": SCHEMA, "heads": records, "all_heads_original_count": 4, "all_heads_new_count": 16,
            "proof": "residual/projection paths choose subsets of shifts8,4,2,1; sums0..15 are distinct modulo128",
            "scope": "explicit selected residual/projection channel ancestry; temporal positions not counted",
            "dense_mean_centering_path_excluded": True,
            "unused_bank_mixing_injects_other_fitted_anchor_features": False,
            "post_training_route_lock_guaranteed": False, "learning_improvement_guaranteed": False}


def validate_certificate(value, shape):
    _equal(value, selected_ancestor_certificate(shape), "canonical selected-ancestor certificate")


def transform(weights, shape):
    """Copy a strict original-diverse fitted initializer and change only MP.

    Full checkpoint/source/corpus identity must be bound by the experiment
    using this structural primitive. No optimizer state is accepted or changed.
    """
    spec = mapping(shape)
    if not isinstance(weights, dict) or set(weights) != set(pilot.WEIGHTS):
        raise ValueError("all and only the five full weight regions required")
    arrays = {}
    for key in pilot.WEIGHTS:
        value = weights[key]
        if isinstance(value, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in value):
            raise ValueError("boolean entries are not integer weights")
        array = np.asarray(value)
        if (array.ndim != 1 or array.size != getattr(shape, "WORDS_" + key)
                or array.dtype.kind not in "iu" or np.any(array < -32768) or np.any(array > 32767)):
            raise ValueError("complete flattened int16-range integer weights required: " + key)
        arrays[key] = array.astype(np.int64, copy=True)
    recovered = {key: array.tolist() for key, array in arrays.items()}
    recovered["MP"] = [0] * shape.WORDS_MP
    canonical_diverse, source_facts = initializer.transform(recovered, shape)
    input_bytes = pilot.weight_bytes({key: array.tolist() for key, array in arrays.items()})
    if input_bytes != pilot.weight_bytes(canonical_diverse):
        raise ValueError("input is not the exact original diverse fitted initialization")
    result = {key: array.tolist() for key, array in arrays.items()}
    projection = np.full((4, 512, 512), -MARGIN, dtype=np.int64)
    for layer in range(4):
        projection[layer, np.arange(512), spec["target_by_layer"][layer]] = 0
    result["MP"] = projection.ravel().tolist()
    facts = {"schema": SCHEMA, "mapping": spec, "selected_ancestor_certificate": selected_ancestor_certificate(shape),
             "source_weights_sha256": pilot.digest(input_bytes),
             "transformed_weights_sha256": pilot.digest(pilot.weight_bytes(result)),
             "unchanged_regions": [key for key in pilot.WEIGHTS if key != "MP"],
             "margin": MARGIN, "maximum_embedding_magnitude": source_facts["maximum_embedding_magnitude"],
             "initial_raw_forward_preservation": "every designated input is in a tied unused bank at the strict fitted source point",
             "initial_head_and_tap_routes_preserved": True,
             "initial_projection_routes": "canonical layer-specific mapping, deliberately changed",
             "initial_upper_three_projection_scatter_l1_preserved": True,
             "guarantee_scope": "initialization only; selected routing is a surrogate, not derivative of rounding",
             "native_execution": False, "optimizer_state_changed": False, "training_performed": False,
             "paid_execution_authorized": False, "recipe_promoted": False}
    return result, facts
