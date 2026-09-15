"""Training-only fixed-gradient accounting at the host-fitted bigram checkpoint.

This does not evolve model weights, execute native code, train a model, select
an optimizer from validation, or authorize paid work. All declared windows and
integer update scales are retained. The exact construction makes projection
input cotangents zero, isolating the direct embedding/head gradient geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import audit_fpga_lm_head_screen as audit  # noqa: E402
from manuscript.tools import check_fpga_lm_bigram_capacity as capacity  # noqa: E402
from manuscript.tools import observe_fpga_lm_routes as observer  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_sign_bias.v1"
UPDATE_SHIFTS = (16, 18, 20)
OBSERVER_WINDOWS = (0, 1, 2)
ZSHIFT = 2
THRESHOLD = 2048


def source_identity():
    result = capacity.source_identity()
    path = Path(__file__).resolve()
    result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def recipe():
    return {"construction": "exact completed host-fitted capacity checkpoint, shift2",
            "training_indices": list(range(1024)), "supervised_pairs_observed": 131072,
            "validation_indices_used_for_optimizer_selection": [],
            "inherited_validation_cohort": list(range(40, 56)),
            "threshold": THRESHOLD, "update_shifts": list(UPDATE_SHIFTS),
            "observer_check_windows": list(OBSERVER_WINDOWS),
            "gradient_formula": "G_EMBED[t,v]=n_t*r[t,v]-k_tv*R[t]; head/bias sum over t",
            "r": "post-power-of-two-truncation LUT mass", "R": "sum of that truncated mass",
            "counterfactual_update": "a=remainder+G; q=trunc_toward_zero(a/2^k); remainder=a-q*2^k",
            "gradient_policy": "all gradients evaluated at the same unchanged fitted checkpoint",
            "state_policy": "zero initial counterfactual remainder; virtual steps never feed back into logits",
            "scope": "deterministic finite-sample diagnostic, not an expectation theorem or evolving optimizer trial",
            "gradient_training": False, "native_execution": False, "cloud_operations": False,
            "physical_fpga_execution": False, "optimizer_selected": False, "quality_gate_candidate": False}


def row_masses(weights, shape, zshift=ZSHIFT):
    import numpy as np
    anchors = [(5 * v) % shape.ML_D for v in range(shape.ML_V)]
    embedded = np.asarray(weights["EMBED"], dtype=np.int64).reshape(shape.ML_V, shape.ML_D)[:, anchors]
    indices = (embedded.max(axis=1, keepdims=True) - embedded) >> zshift
    lut = np.asarray([round(65536 * 2.0 ** (-i / 16.0)) for i in range(256)], dtype=np.int64)
    energies = lut[np.minimum(indices, 255)]
    probabilities = np.empty_like(energies)
    for token, row in enumerate(energies):
        total, shift = int(row.sum()), 0
        while total >> shift > 65536:
            shift += 1
        probabilities[token] = row >> shift
    totals = probabilities.sum(axis=1)
    if np.any(totals <= 0) or np.any(totals > 65536):
        raise ValueError("invalid post-truncation LUT row mass")
    return probabilities, totals


def window_gradients(tokens, targets, probabilities, totals):
    """Exact class-coordinate gradients for one window; no floating differentiation."""
    import numpy as np
    probabilities, totals = np.asarray(probabilities), np.asarray(totals)
    if (totals.ndim != 1 or not len(totals) or probabilities.shape != (len(totals), len(totals))
            or probabilities.dtype.kind not in "iu" or totals.dtype.kind not in "iu"
            or np.any(probabilities < 0) or np.any(probabilities > 65536)
            or np.any(totals <= 0) or np.any(totals > 65536)
            or np.any(probabilities.sum(axis=1) != totals)):
        raise ValueError("row masses must be exact nonnegative integer balanced-LUT probabilities")
    probabilities, totals = probabilities.astype(np.int64), totals.astype(np.int64)
    vocab = len(totals)
    if (not isinstance(tokens, list) or not isinstance(targets, list) or not tokens or len(tokens) != len(targets)
            or any(type(value) is not int or not 0 <= value < vocab for value in tokens + targets)):
        raise ValueError("window requires equal nonempty literal-integer token/target arrays")
    pairs = np.bincount(np.asarray(tokens) * vocab + targets, minlength=vocab * vocab).reshape(vocab, vocab)
    occurrences = pairs.sum(axis=1)
    embedding = occurrences[:, None] * probabilities - pairs * totals[:, None]
    per_position = probabilities[tokens].copy()
    per_position[np.arange(len(tokens)), targets] -= totals[tokens]
    if np.any(embedding.sum(axis=1)) or np.any(per_position.sum(axis=1)):
        raise ValueError("balanced surrogate gradients must sum to exactly zero")
    return {"embedding": embedding, "head": embedding.sum(axis=0), "logits": per_position,
            "occurrences": occurrences, "pairs": pairs}


def remainder_step(remainder, gradient, k):
    """Checked integer reference for the proposed, separately versioned updater."""
    import numpy as np
    if type(k) is not int or not 0 <= k <= 30:
        raise ValueError("integer update shift must be a literal integer in0..30")
    remainder, gradient = np.asarray(remainder), np.asarray(gradient)
    if (remainder.shape != gradient.shape or remainder.dtype.kind not in "iu" or gradient.dtype.kind not in "iu"
            or np.any(remainder < -(1 << 31)) or np.any(remainder > (1 << 31) - 1)
            or np.any(gradient < -(1 << 31)) or np.any(gradient > (1 << 31) - 1)):
        raise ValueError("remainder/gradient require matching signed-int32-valued integer arrays")
    divisor = 1 << k
    remainder, gradient = remainder.astype(np.int64), gradient.astype(np.int64)
    if np.any(np.abs(remainder) >= divisor):
        raise ValueError("initial remainder lies outside the canonical interval")
    accumulated = remainder + gradient
    if np.any(accumulated < -(1 << 31)) or np.any(accumulated > (1 << 31) - 1):
        raise ValueError("int32 remainder accumulation would overflow")
    quotient = np.sign(accumulated) * (np.abs(accumulated) // divisor)
    new_remainder = accumulated - quotient * divisor
    if np.any(np.abs(new_remainder) >= divisor):
        raise ValueError("integer remainder bound violated")
    return quotient, new_remainder


class GradientAccounting:
    """Streaming frozen-gradient accounting, not a model optimizer."""

    def __init__(self, initial_weights, threshold=THRESHOLD, shifts=UPDATE_SHIFTS):
        import numpy as np
        if type(threshold) is not int or threshold < 0:
            raise ValueError("threshold must be a nonnegative literal integer")
        if not shifts or len(set(shifts)) != len(shifts) or any(type(k) is not int or not 0 <= k <= 30 for k in shifts):
            raise ValueError("update shifts must be unique literal integers in0..30")
        initial = np.asarray(initial_weights)
        if initial.dtype.kind not in "iu" or np.any(initial < -32768) or np.any(initial > 32767):
            raise ValueError("initial weight coordinates must fit int16")
        self.initial, self.threshold = initial.astype(np.int64), threshold
        self.windows = 0
        self.sum_gradient = np.zeros_like(self.initial)
        self.sum_sign = np.zeros_like(self.initial)
        self.positive_active = np.zeros_like(self.initial)
        self.negative_active = np.zeros_like(self.initial)
        self.nonzero = np.zeros_like(self.initial)
        self.deadband = np.zeros_like(self.initial)
        self.states = {k: {"remainder": np.zeros_like(self.initial), "sum_steps": np.zeros_like(self.initial),
                           "virtual_int16_exceed_windows": 0, "maximum_absolute_window_step": 0} for k in shifts}

    def add(self, gradient):
        import numpy as np
        gradient = np.asarray(gradient)
        if (gradient.shape != self.initial.shape or gradient.dtype.kind not in "iu"
                or np.any(gradient < -(1 << 31)) or np.any(gradient > (1 << 31) - 1)):
            raise ValueError("fixed gradient shape/int32 range differs")
        gradient = gradient.astype(np.int64)
        active = np.abs(gradient) > self.threshold
        self.sum_gradient += gradient
        self.sum_sign += np.sign(gradient) * active
        self.positive_active += active & (gradient > 0)
        self.negative_active += active & (gradient < 0)
        self.nonzero += gradient != 0
        self.deadband += (gradient != 0) & ~active
        self.windows += 1
        for k, state in self.states.items():
            steps, state["remainder"] = remainder_step(state["remainder"], gradient, k)
            state["sum_steps"] += steps
            state["maximum_absolute_window_step"] = max(state["maximum_absolute_window_step"], int(np.abs(steps).max()))
            virtual = self.initial - state["sum_steps"]
            state["virtual_int16_exceed_windows"] += int(np.any((virtual < -32768) | (virtual > 32767)))

    def finish(self):
        import numpy as np
        if not self.windows:
            raise ValueError("empty fixed-gradient stream")
        gradient, signs = self.sum_gradient, self.sum_sign
        g, s = gradient.astype(np.float64).ravel(), signs.astype(np.float64).ravel()
        denominator = float(np.linalg.norm(g) * np.linalg.norm(s))
        variants = {}
        for k, state in self.states.items():
            if np.any(gradient != (1 << k) * state["sum_steps"] + state["remainder"]):
                raise ValueError("error-feedback telescoping identity failed")
            variants[str(k)] = {name: value.ravel().tolist() if isinstance(value, np.ndarray) else value
                                for name, value in state.items()}
            variants[str(k)].update(exact_telescoping_identity=True,
                                     maximum_absolute_remainder=int(np.abs(state["remainder"]).max()),
                                     cumulative_fractional_step_error_bound_exclusive=1)
        return {"windows": self.windows, "coordinate_shape": list(self.initial.shape), "coordinate_count": self.initial.size,
                "ever_nonzero_coordinates": int(np.count_nonzero(self.nonzero)),
                "nonzero_mean_gradient_coordinates": int(np.count_nonzero(gradient)),
                "nonzero_sign_drift_coordinates": int(np.count_nonzero(signs)),
                "opposite_nonzero_mean_gradient_and_sign_drift_coordinates": int(np.count_nonzero(gradient * signs < 0)),
                "nonzero_sign_drift_on_exact_zero_sum_gradient_coordinates": int(np.count_nonzero((gradient == 0) & (signs != 0))),
                "mean_gradient_sign_drift_cosine": float(np.dot(g, s) / denominator) if denominator else None,
                "sum_gradient": gradient.ravel().tolist(), "sum_thresholded_sign": signs.ravel().tolist(),
                "positive_active_windows": self.positive_active.ravel().tolist(),
                "negative_active_windows": self.negative_active.ravel().tolist(),
                "nonzero_gradient_windows": self.nonzero.ravel().tolist(),
                "deadband_nonzero_gradient_windows": self.deadband.ravel().tolist(),
                "weight_update_sign_convention": "native SignSGD weight change is minus the reported thresholded sign",
                "fixed_gradient_error_feedback": variants}


def observer_check(weights, counts, tokens, targets, shape, zshift=ZSHIFT):
    """Independent all-parameter scatter at the constructed common-projection point."""
    import numpy as np
    expected_weights, facts = capacity.construct_weights(counts, shape, zshift)
    if weights != expected_weights:
        raise ValueError("observer diagnostic requires the exact construction")
    probabilities, totals = row_masses(weights, shape, zshift)
    gradients = window_gradients(tokens, targets, probabilities, totals)
    expected = {name: np.zeros(len(values), dtype=np.int64) for name, values in weights.items()}
    vocab, d, layers = shape.ML_V, shape.ML_D, shape.ML_N_LAYERS
    anchors = np.asarray(facts["anchors"])
    expected["EMBED"].reshape(vocab, d)[:, anchors] = gradients["embedding"]
    expected["W_HEAD"].reshape(vocab, d)[np.arange(vocab), anchors] = gradients["head"]
    expected["MB"].reshape(layers, d)[:, anchors] = gradients["head"]
    embedding = np.asarray(weights["EMBED"], dtype=np.int64).reshape(vocab, d)
    centered = embedding - (embedding.sum(axis=1, keepdims=True) >> shape.ML_LOG2_D)
    shared_stars = []
    for layer in range(layers):
        stars = []
        for position in range(len(tokens)):
            candidates = np.stack([centered[tokens[position - (tap << min(layer, 6))]]
                                   for tap in range(shape.ML_K) if position >= (tap << min(layer, 6))])
            convolved = candidates.max(axis=0) if layer % 2 == 0 else candidates.min(axis=0)
            star = int(convolved.argmax())
            stars.append(star)
            expected["MP"].reshape(layers, d, d)[layer, anchors, star] += gradients["logits"][position]
        shared_stars.extend([star for star in stars for _ in range(d)])
    result = observer.observe(weights, tokens, targets, shape, zshift=zshift, include_parameter_gradients=True)
    audit._assert_equal(result["gradients"], gradients["logits"].ravel().tolist(), "exact per-position balanced logits")
    audit._assert_equal(result["native_routes"]["PROJ_IDX"], shared_stars, "independently derived shared projection routes")
    audit._assert_equal(result["native_routes"]["AD_HEAD"], facts["anchors"] * len(tokens), "constructed head routes")
    hashes = {}
    for name, values in expected.items():
        audit._assert_equal(result["parameter_gradients"][name], values.tolist(), "exact constructed parameter gradient " + name)
        hashes[name] = {"entries": len(values), "sha256": pilot.digest(capacity.screen._i32(values.tolist()))}
    return {"all_parameter_gradients_equal": True, "all_logit_gradients_equal": True,
            "shared_projection_routes_equal": True, "parameter_gradient_hashes": hashes,
            "parameter_gradient_entries_checked": sum(len(values) for values in expected.values()),
            "native_execution": False, "comparison": "frozen host observer, not native parameter-gradient readback"}


def _seal(root):
    payload, declared = capacity.calibration._read_record(root / "artifact_manifest.json")
    actual = capacity._artifacts(root)
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    audit._assert_equal(declared, actual, "completed capacity artifacts")
    return payload, declared


def validate_inputs(run, record):
    frozen_sources = source_identity()
    plan_bytes = capacity.PLAN.read_bytes()
    root, record_path = Path(run).resolve(strict=True), Path(record).resolve(strict=True)
    record_bytes, receipt = capacity.calibration._read_record(record_path)
    manifest = capacity._read(root / "manifest.json")
    summary = capacity._read(root / "summary.json")
    if manifest["schema"] != capacity.SCHEMA or summary["schema"] != capacity.SCHEMA:
        raise ValueError("requires the completed declared capacity study")
    required = {"complete": True, "host_count_fitting": True, "gradient_training": False,
                "native_training_updates": 0, "declared_cells": 4, "completed_cells": 4,
                "all_scale_outputs_equal": True, "failures": [], "physical_fpga_execution": False,
                "cloud_operations": False, "quality_gate_candidate": False, "heldout_test_evaluated": False}
    audit._assert_equal({name: summary.get(name) for name in required}, required, "complete capacity scope")
    audit._assert_equal({name: receipt[name] for name in summary}, summary, "capacity publication summary")
    audit._assert_equal(manifest["recipe"], capacity.recipe(), "capacity construction protocol")
    audit._assert_equal(summary["source_hashes"], manifest["source_hashes"], "capacity source publication link")
    library = capacity._read(root / "library.json")
    capacity._frozen(manifest, library)
    audit._assert_equal(audit._verify_library(root, library, manifest), summary["build"], "actual source-bound capacity build")
    artifact_bytes, artifacts = _seal(root)
    if any("failure" in Path(row["file"]).name for row in artifacts["files"]):
        raise ValueError("failed capacity artifacts cannot support a successful diagnostic")
    required_receipt = {"run_directory": str(root), "artifact_manifest_sha256": pilot.digest(artifact_bytes),
                        "artifact_files_verified": len(artifacts["files"]),
                        "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
                        "constructed_full_checkpoints_verified": 4, "checkpoint_validation_windows": 64,
                        "saved_native_predictions_recomputed": 8192, "saved_native_gradient_entries_recomputed": 1048576,
                        "saved_native_head_route_entries_recomputed": 1048576}
    audit._assert_equal({name: receipt[name] for name in required_receipt}, required_receipt, "capacity publication coverage")
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], 1024, 56)
    audit._assert_equal(dataset, manifest["dataset"], "capacity source corpus")
    controls = capacity._read(root / "counts.json")
    expected_controls = {"alpha": 1, "vocabulary": 128, "training_pairs": 131072,
                         "train_indices": list(range(1024)), "validation_indices": list(range(40, 56))}
    audit._assert_equal({name: controls[name] for name in expected_controls}, expected_controls,
                        "source count-fit budget and inherited cohort")
    counts = [[1] * 128 for _ in range(128)]
    for index in range(1024):
        tokens, targets = pilot.window(raw, dataset, "train", index)
        for token, target in zip(tokens, targets):
            counts[token][target] += 1
    audit._assert_equal(counts, controls["bigram_pair_counts"], "same exact training-only alpha1 counts")
    shape = pilot.layout(512)
    audit._assert_equal(pilot.layout_values(shape), manifest["layout"], "frozen capacity layout")
    weights, facts = capacity.construct_weights(counts, shape, ZSHIFT)
    payload = (root / "shift_2/checkpoint_000000.i32").read_bytes()
    if payload != pilot.weight_bytes(weights):
        raise ValueError("actual source-bound capacity checkpoint differs from construction")
    if source_identity() != frozen_sources:
        raise ValueError("diagnostic sources changed during input validation")
    return {"root": root, "record_path": record_path, "record_bytes": record_bytes,
            "artifact_bytes": artifact_bytes, "manifest": manifest, "library": library,
            "raw": raw, "dataset": dataset, "counts": counts, "shape": shape, "weights": weights,
            "facts": facts, "source_hashes": frozen_sources,
            "checkpoint_sha256": pilot.digest(payload), "plan_bytes": plan_bytes}


def _recheck(context):
    capacity._frozen(context["manifest"], context["library"])
    if (source_identity() != context["source_hashes"]
            or context["record_path"].read_bytes() != context["record_bytes"]
            or capacity.PLAN.read_bytes() != context["plan_bytes"]
            or _seal(context["root"])[0] != context["artifact_bytes"]):
        raise ValueError("fixed-gradient diagnostic source evidence changed")


def diagnose(run, record):
    import numpy as np
    context = validate_inputs(run, record)
    weights, shape, counts = context["weights"], context["shape"], context["counts"]
    probabilities, totals = row_masses(weights, shape)
    anchors = context["facts"]["anchors"]
    initial = np.asarray(weights["EMBED"], dtype=np.int64).reshape(128, 512)[:, anchors]
    accumulators = {"EMBED_anchor_coordinates": GradientAccounting(initial),
                    "W_HEAD_anchor_coordinates": GradientAccounting(np.zeros(128, dtype=np.int64)),
                    "MB_anchor_coordinates": GradientAccounting(np.zeros((4, 128), dtype=np.int64))}
    buckets = {name: {"context_window_rows": 0, "supervised_positions": 0, "positive_active_coordinates": 0,
                      "negative_active_coordinates": 0, "nonzero_deadband_coordinates": 0,
                      "positive_gradient_mass": 0, "negative_gradient_absolute_mass": 0}
               for name in ("1", "2", "3", "4-7", "8-15", "16-127", "128")}
    observer_checks = []
    for index in range(1024):
        tokens, targets = pilot.window(context["raw"], context["dataset"], "train", index)
        gradients = window_gradients(tokens, targets, probabilities, totals)
        accumulators["EMBED_anchor_coordinates"].add(gradients["embedding"])
        accumulators["W_HEAD_anchor_coordinates"].add(gradients["head"])
        accumulators["MB_anchor_coordinates"].add(np.tile(gradients["head"], (4, 1)))
        for token, count in enumerate(gradients["occurrences"]):
            if not count:
                continue
            name = str(count) if count < 4 or count == 128 else "4-7" if count < 8 else "8-15" if count < 16 else "16-127"
            bucket, values = buckets[name], gradients["embedding"][token]
            bucket["context_window_rows"] += 1
            bucket["supervised_positions"] += int(count)
            bucket["positive_active_coordinates"] += int(np.count_nonzero(values > THRESHOLD))
            bucket["negative_active_coordinates"] += int(np.count_nonzero(values < -THRESHOLD))
            bucket["nonzero_deadband_coordinates"] += int(np.count_nonzero((values != 0) & (np.abs(values) <= THRESHOLD)))
            bucket["positive_gradient_mass"] += int(values[values > 0].sum())
            bucket["negative_gradient_absolute_mass"] += int(-values[values < 0].sum())
        if index in OBSERVER_WINDOWS:
            observer_checks.append({"window_index": index, **observer_check(weights, counts, tokens, targets, shape)})
    if sum(bucket["supervised_positions"] for bucket in buckets.values()) != 131072:
        raise ValueError("incomplete training-window accounting")
    regions = {name: value.finish() for name, value in accumulators.items()}
    _recheck(context)
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(),
            "source_hashes": context["source_hashes"], "dataset": context["dataset"],
            "declared_plan_sha256": pilot.digest(context["plan_bytes"]),
            "environment": {"python": sys.version, "numpy": np.__version__},
            "capacity_run_directory": str(context["root"]),
            "capacity_record": {"file": str(context["record_path"]), "sha256": pilot.digest(context["record_bytes"])},
            "capacity_artifact_manifest_sha256": pilot.digest(context["artifact_bytes"]),
            "capacity_checkpoint_sha256": context["checkpoint_sha256"],
            "training_windows_observed": 1024, "supervised_pairs_observed": 131072,
            "post_truncation_row_mass_minimum": int(totals.min()), "post_truncation_row_mass_maximum": int(totals.max()),
            "regions": regions, "embedding_context_occurrence_buckets": buckets, "observer_checks": observer_checks,
            "native_execution": False, "gradient_training": False, "cloud_operations": False,
            "optimizer_selected": False, "quality_gate_candidate": False,
            "limitations": ["All gradients are frozen at a host-fitted capacity checkpoint; virtual updates never alter them.",
                            "Finite actual windows diagnose update geometry, not a population expectation or training outcome.",
                            "Alpha1 smoothing, LUT approximation and variable truncated row mass mean aggregate gradients need not vanish at this fit.",
                            "The balanced rounded LUT surrogate is separately different from the reported LUT-normalized NLL derivative.",
                            "Full-stream accounting covers anchored embedding/head/bias coordinates; not the evolving full projection optimizer.",
                            "Only three declared windows compare all parameter gradients against the frozen host observer, not native readback.",
                            "All fixed update shifts are retained; no validation result selected an optimizer or passed a quality gate."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed capacity raw run")
    parser.add_argument("--record", type=Path, required=True, help="completed capacity compact record")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = capacity._new_external_path(args.output)
    result = diagnose(args.run, args.record)
    pilot.write_json(output, result)
    print(json.dumps({"complete": True, "output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
