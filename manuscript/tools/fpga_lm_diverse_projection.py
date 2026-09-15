"""Strict, initialization-only route diversification for the fitted D512 model.

This changes MP only. It preserves the initial forward computation because
the three unused feature banks are tied maxima at the fitted source point.
There is no assertion that those ties, margins, or gradient properties persist
after an update. The historical byte binding is enforced by the study engine.
"""
from __future__ import annotations

from pathlib import Path

from manuscript.tools import diagnose_fpga_lm_calibration as calibration
from manuscript.tools.observe_fpga_lm_routes import observe
from scripts import fpga_lm_pretrain as pilot

SCHEMA = "tikitorch.fpga_lm_diverse_projection_initializer.v1"
MARGIN, MAX_MAGNITUDE = 8192, 756


def source_identity():
    result = calibration.source_identity()
    path = Path(__file__).resolve()
    result[path.relative_to(pilot.REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def mapping(shape):
    if pilot.layout_values(shape) != pilot.layout_values(pilot.layout(512)):
        raise ValueError("initializer requires the exact frozen D512 layout")
    a = [(5 * value) % 512 for value in range(128)]
    unused = sorted(set(range(512)) - set(a))
    b, c, d = unused[:128], unused[128:256], unused[256:]
    phi = [None] * 512
    for source, target in ((a, b), (b, c), (c, d), (d, b)):
        for left, right in zip(source, target):
            phi[left] = right
    return {"A": a, "B": b, "C": c, "D": d, "target_by_row": phi}


def transform(weights, shape):
    """Validate the strict fitted structure and return copied weights/facts."""
    import numpy as np
    banks = mapping(shape)
    if set(weights) != set(pilot.WEIGHTS):
        raise ValueError("all and only the five full weight regions are required")
    arrays = {}
    for name in pilot.WEIGHTS:
        values = weights[name]
        if isinstance(values, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("boolean weights are not literal integer weights")
        array = np.asarray(values)
        if (array.ndim != 1 or array.size != getattr(shape, "WORDS_" + name)
                or array.dtype.kind not in "iu" or np.any(array < -32768) or np.any(array > 32767)):
            raise ValueError("full bounded integer weight regions required")
        arrays[name] = array.astype(np.int64)
    if any(np.any(arrays[name]) for name in ("MH", "MP", "MB")):
        raise ValueError("only the zero-core count-fitted source point is supported")
    embedded = arrays["EMBED"].reshape(128, 512)
    anchors = embedded[:, banks["A"]]
    if (np.any(embedded[:, banks["B"] + banks["C"] + banks["D"]])
            or np.any(anchors > 0) or np.any(anchors < -MAX_MAGNITUDE)
            or np.any(anchors % 4) or np.any(anchors.max(axis=1) != 0)):
        raise ValueError("expected z2 fitted embedding: zero unused banks, nonpositive bounded anchor multiples of four")
    expected_head = np.full((128, 512), -MARGIN, dtype=np.int64)
    expected_head[np.arange(128), banks["A"]] = 0
    if not np.array_equal(arrays["W_HEAD"].reshape(128, 512), expected_head):
        raise ValueError("the exact fitted vocabulary anchors and head margins are required")
    changed = {name: array.tolist() for name, array in arrays.items()}
    projection = np.full((shape.ML_N_LAYERS, 512, 512), -MARGIN, dtype=np.int64)
    projection[:, np.arange(512), banks["target_by_row"]] = 0
    changed["MP"] = projection.ravel().tolist()
    magnitude = int(-anchors.min())
    u = (128 * magnitude + 511) // 512
    facts = {"schema": SCHEMA, "mapping": banks, "margin": MARGIN,
             "maximum_embedding_magnitude": magnitude, "initial_upper_increment_bound": u,
             "initial_residual_lower_bound": -magnitude, "initial_residual_upper_bound": 4 * u,
             "initial_head_margin_lower_bound": MARGIN - magnitude,
             "unchanged_regions": [name for name in pilot.WEIGHTS if name != "MP"],
             "source_weights_sha256": pilot.digest(pilot.weight_bytes(weights)),
             "transformed_weights_sha256": pilot.digest(pilot.weight_bytes(changed)),
             "guarantee_scope": "initial fitted source point only; no post-update guarantee",
             "initial_l1_preserved_projection_layers": [3, 2, 1],
             "bottom_projection_may_merge_A_and_D": True,
             "past_context_use_guaranteed": False}
    return changed, facts


def check_initial_equivalence(original, changed, tokens, targets, shape, observer=None):
    """Read-only exact forward/output/route check at one initialization window."""
    observer = observe if observer is None else observer
    expected, facts = transform(original, shape)
    if pilot.weight_bytes(changed) != pilot.weight_bytes(expected):
        raise ValueError("changed weights are not the exact declared MP-only transform")
    old = observer(original, tokens, targets, shape, zshift=2)
    new = observer(changed, tokens, targets, shape, zshift=2)
    for value in (old, new):
        if (type(value["saturation_events"]) is not int or value["saturation_events"] != 0
                or value["route_diagnostics"]["backward_int32_overflow_detected"] is not False):
            raise ValueError("initial forward/backward guard failed")
    fields = ("predictions", "gradients", "lut_normalized_nll", "accuracy", "rounded_zero_target_mass")
    if any(old[name] != new[name] for name in fields):
        raise ValueError("initial prediction/gradient/distribution equality failed")
    if any(old["native_routes"][name] != new["native_routes"][name] for name in ("AD_HEAD", "TAP_IDX")):
        raise ValueError("initial head or temporal routes changed")
    routes = facts["mapping"]["target_by_row"] * (len(tokens) * shape.ML_N_LAYERS)
    if new["native_routes"]["PROJ_IDX"] != routes:
        raise ValueError("initial projection routes differ from the declared mapping")
    old_logits = calibration.integer_logits(original, tokens, shape)["logits"]
    new_logits = calibration.integer_logits(changed, tokens, shape)["logits"]
    if old_logits != new_logits:
        raise ValueError("initial saturated logits differ")
    layers = new["route_diagnostics"]["projection_layers"]
    if any(layers[index]["projection_incoming_l1"] != layers[index]["projection_scattered_l1"]
           for index in (3, 2, 1)):
        raise ValueError("initial upper-three projection L1 preservation failed")
    return {"initial_logits_equal": True, "initial_predictions_gradients_equal": True,
            "initial_head_tap_routes_equal": True, "initial_projection_routes_equal_mapping": True,
            "initial_upper_three_projection_l1_preserved": True,
            "logits_sha256": pilot.digest(calibration.screen._i32([x for row in old_logits for x in row])),
            "source_weights_sha256": facts["source_weights_sha256"],
            "transformed_weights_sha256": facts["transformed_weights_sha256"],
            "scope": "initialization only, not guaranteed after any update"}
