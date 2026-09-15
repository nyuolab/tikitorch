"""Exploratory, evaluation-only confidence grid on the completed budget extension.

Both arms, seeds0..2, checkpoints256/1024, validation24..39 and LUT shifts0..4
are fixed before evaluation. This reuses development data: it neither selects a
winner nor changes historical trained-recipe gates. No training, native-library
execution, cloud operations or final-test evaluation is performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import extend_fpga_lm_budget as experiment  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_calibration_diagnostic.v1"
ARMS = ("anchors", "anchors_fine")
SEEDS = (0, 1, 2)
STEPS = (256, 1024)
SHIFTS = (0, 1, 2, 3, 4)
VALIDATION_INDICES = tuple(range(24, 40))


def source_identity():
    from manuscript.tools import audit_fpga_lm_budget as audit
    result = audit.source_identity()
    path = Path(__file__).resolve()
    result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def recipe():
    return {"arms": list(ARMS), "seeds": list(SEEDS), "checkpoints": list(STEPS),
            "evaluation_lut_shifts": list(SHIFTS), "validation_indices": list(VALIDATION_INDICES),
            "forward_policy": "fixed checkpoint logits and argmax; only evaluation LUT shift changes",
            "distribution": "integer LUT energies normalized before power-of-two gradient truncation",
            "quantiles": {"probabilities": [0, 0.5, 0.9, 0.95, 0.99, 1], "method": "linear"},
            "last_bin_policy": "raw target index >=255 includes boundary; actual clipping requires >255",
            "subgroup_policy": "clipping and rounded-zero masks overlap; their contributions are not additive partitions",
            "scope": "post-hoc development calibration diagnostic; all grid cells retained, no selection or gate promotion",
            "training": False, "native_execution": False, "cloud_operations": False,
            "heldout_test_evaluated": False, "historical_gates_modified": False}


def _integer_array(values, name):
    import numpy as np
    array = np.asarray(values)
    if array.dtype.kind not in "iu" or array.size == 0:
        raise ValueError(name + " must be a nonempty integer array")
    return array


def integer_logits(weights, tokens, shape):
    """Independent saturated integer forward exposing logits, not native execution."""
    import numpy as np
    d, n, k, vocab, length = (shape.ML_D, shape.ML_N_LAYERS, shape.ML_K,
                             shape.ML_V, shape.ML_L_SEQ)
    if (min(d, n, k, vocab, length) <= 0 or d & (d - 1)
            or shape.ML_LOG2_D != d.bit_length() - 1):
        raise ValueError("invalid integer-forward geometry")
    ids = _integer_array(tokens, "tokens")
    if ids.ndim != 1 or len(ids) != length or np.any(ids >= vocab):
        raise ValueError("tokens require one complete in-vocabulary window")
    if np.any(ids < 0):
        raise ValueError("token outside vocabulary")
    dims = {"EMBED": (vocab, d), "MH": (n, d, k), "MP": (n, d, d),
            "MB": (n, d), "W_HEAD": (vocab, d)}
    arrays = {}
    for name, dimensions in dims.items():
        values = _integer_array(weights[name], name)
        if (values.ndim != 1 or values.size != math.prod(dimensions)
                or np.any(values < -(1 << 31)) or np.any(values > (1 << 31) - 1)):
            raise ValueError("weight region does not match int32 storage geometry: " + name)
        values = values.astype(np.int64)
        arrays[name] = (((values + 32768) % 65536) - 32768).reshape(dimensions)
    history = np.zeros((n, length, d), dtype=np.int64)
    logits, predictions, saturation_events = [], [], 0

    def sat(values):
        nonlocal saturation_events
        saturation_events += int(np.count_nonzero((values < -32768) | (values > 32767)))
        return np.clip(values, -32768, 32767)

    for position, token in enumerate(ids):
        residual = arrays["EMBED"][token].copy()
        for layer in range(n):
            history[layer, position] = sat(residual - (int(residual.sum()) >> shape.ML_LOG2_D))
            candidates = np.stack([
                history[layer, position - (tap << min(layer, 6))] + arrays["MH"][layer, :, tap]
                for tap in range(k) if position >= (tap << min(layer, 6))])
            conv = sat(candidates.max(axis=0) if layer % 2 == 0 else candidates.min(axis=0))
            projected = sat((arrays["MP"][layer] + conv).max(axis=1) + arrays["MB"][layer])
            residual = sat(residual + projected)
        scores = sat((arrays["W_HEAD"] + residual).max(axis=1))
        logits.append(scores.tolist())
        predictions.append(int(scores.argmax()))
    return {"logits": logits, "predictions": predictions, "saturation_events": saturation_events}


def score_logits(logits, targets, zshift):
    """Exact LUT scoring and balanced gradients for generic nonempty 2D int16 logits.

    The helper accepts shifts0..30; the declared diagnostic uses only0..4.
    Probabilities/NLL precede surrogate truncation. A rounded-zero target mass
    therefore has finite NLL and normally a nonzero negative target gradient.
    """
    import numpy as np
    if type(zshift) is not int or not 0 <= zshift <= 30:
        raise ValueError("zshift must be an integer in0..30, not bool")
    scores = _integer_array(logits, "logits")
    ids = _integer_array(targets, "targets")
    if (scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0
            or np.any(scores < -32768) or np.any(scores > 32767)):
        raise ValueError("logits must be a nonempty 2D saturated int16 array")
    if ids.ndim != 1 or len(ids) != len(scores) or np.any(ids < 0) or np.any(ids >= scores.shape[1]):
        raise ValueError("targets require one valid vocabulary index per logit row")
    scores, ids = scores.astype(np.int64), ids.astype(np.int64)
    lut = np.asarray([round(65536 * 2.0 ** (-index / 16.0)) for index in range(256)], dtype=np.int64)
    predictions, gradients, losses, confidences = [], [], [], []
    zeros, at_last, above_last, correct = [], [], [], []
    for row, target in zip(scores, ids):
        prediction = int(row.argmax())
        indices = (int(row[prediction]) - row) >> zshift
        energies = lut[np.minimum(indices, 255)]
        total = int(energies.sum())
        losses.append(math.log(total) - math.log(int(energies[target])))
        confidences.append(int(energies.max()) / total)
        shift = 0
        while total >> shift > 65536:
            shift += 1
        gradient = energies >> shift
        zeros.append(bool(gradient[target] == 0))
        at_last.append(bool(indices[target] >= 255))
        above_last.append(bool(indices[target] > 255))
        gradient[target] -= gradient.sum()
        predictions.append(prediction)
        gradients.extend(gradient.tolist())
        correct.append(prediction == int(target))
    return {"predictions": predictions, "gradients": gradients,
            "lut_normalized_nll": sum(losses) / len(losses), "accuracy": sum(correct) / len(correct),
            "rounded_zero_target_mass": sum(zeros), "token_nll": losses,
            "maximum_probabilities": confidences, "rounded_zero_target_mask": zeros,
            "target_index_at_or_above_last_bin_mask": at_last,
            "target_index_above_last_bin_mask": above_last, "correct_mask": correct}


def summarize_scores(windows):
    """Aggregate exact token losses; overlapping subgroup contributions stay separate."""
    import numpy as np
    if not windows:
        raise ValueError("cannot summarize an empty scoring cohort")
    fields = ("token_nll", "maximum_probabilities", "rounded_zero_target_mask",
              "target_index_at_or_above_last_bin_mask", "target_index_above_last_bin_mask", "correct_mask")
    combined = {field: [] for field in fields}
    for row in windows:
        count = len(row["token_nll"])
        if not count or any(len(row[field]) != count for field in fields):
            raise ValueError("score arrays must have equal nonzero token counts")
        for field in fields:
            combined[field].extend(row[field])
    for field in fields[2:]:
        if any(type(value) is not bool for value in combined[field]):
            raise ValueError("score masks must contain literal booleans")
    losses, probabilities = combined["token_nll"], combined["maximum_probabilities"]
    if (any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in losses)
            or any(type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 1
                   for value in probabilities)):
        raise ValueError("losses/probabilities must be finite and valid")
    if any(above and not at for above, at in zip(combined["target_index_above_last_bin_mask"],
                                                combined["target_index_at_or_above_last_bin_mask"])):
        raise ValueError("actual clipping must be a subset of the last-bin boundary mask")
    total = sum(losses)

    def group(mask):
        selected = [loss for loss, included in zip(losses, mask) if included]
        contribution = sum(selected)
        return {"count": len(selected), "total_nll": contribution,
                "mean_nll": contribution / len(selected) if selected else None,
                "nll_contribution_per_token": contribution / len(losses),
                "fraction_of_total_nll": contribution / total if total else None}

    correct = combined["correct_mask"]
    quantiles = np.quantile(np.asarray(losses), [0, 0.5, 0.9, 0.95, 0.99, 1], method="linear")
    return {"tokens": len(losses), "lut_normalized_nll": total / len(losses), "total_nll": total,
            "accuracy": sum(correct) / len(correct), "mean_maximum_probability": sum(probabilities) / len(probabilities),
            "rounded_zero_target_mass": sum(combined["rounded_zero_target_mask"]),
            "target_index_at_or_above_last_bin_count": sum(combined["target_index_at_or_above_last_bin_mask"]),
            "target_index_above_last_bin_count": sum(combined["target_index_above_last_bin_mask"]),
            "token_nll_quantiles": {name: float(value) for name, value in zip(
                ("minimum", "median", "p90", "p95", "p99", "maximum"), quantiles)},
            "loss_contributions": {
                "rounded_zero_target": group(combined["rounded_zero_target_mask"]),
                "target_index_at_or_above_last_bin": group(combined["target_index_at_or_above_last_bin_mask"]),
                "target_index_above_last_bin": group(combined["target_index_above_last_bin_mask"]),
                "correct": group(correct), "incorrect": group([not value for value in correct])}}


def _read_record(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key: " + key)
            result[key] = value
        return result
    payload = Path(path).read_bytes()
    json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
    return payload, pilot._finite_json(payload.decode("utf-8"))


def validate_inputs(run, audit_record):
    """Verify the completed audit and actual source-bound payloads, without replaying its observer."""
    from manuscript.tools import audit_fpga_lm_budget as audit
    from manuscript.tools import audit_fpga_lm_head_screen as prior
    root = Path(run).resolve(strict=True)
    audit_path = Path(audit_record).resolve(strict=True)
    frozen = source_identity()
    audit_bytes, receipt = _read_record(audit_path)
    manifest_bytes, manifest = _read_record(root / "manifest.json")
    _, summary = _read_record(root / "summary.json")
    audit._verify_root_status(manifest, summary)
    if manifest["recipe"] != experiment.recipe() or manifest["source_hashes"] != experiment.source_identity():
        raise ValueError("diagnostic requires the exact completed extension sources and protocol")
    artifacts = experiment._verify_artifacts(root, experiment.SCHEMA)
    artifact_bytes = (root / "artifact_manifest.json").read_bytes()
    prior._assert_equal(prior._read(root / "protocol.json"),
                        {"schema": experiment.SCHEMA, **manifest["recipe"]}, "extension protocol")
    library = prior._read(root / "library.json")
    build = audit._verify_library(root, library, manifest)
    shape = pilot.layout(512)
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], 1024, 40)
    if dataset != manifest["dataset"] or pilot.layout_values(shape) != manifest["layout"]:
        raise ValueError("corpus or native layout differs from the audited run")
    bound_v1 = experiment.v1_binding(manifest["v1_evidence"]["run_directory"], dataset)
    prior._assert_equal(bound_v1, manifest["v1_evidence"], "historical v1 evidence")
    required = {"schema": "tikitorch.fpga_lm_budget_audit.v1", "complete": True,
                "run_directory": str(root), "recipe": manifest["recipe"], "dataset": dataset,
                "source_hashes": manifest["source_hashes"], "audit_source_hashes": audit.source_identity(),
                "declared_plan_sha256": manifest["plan_document"]["sha256"],
                "v1_evidence": bound_v1, "library_sha256": library["sha256"], "build": build,
                "artifact_manifest_sha256": pilot.digest(artifact_bytes),
                "artifact_files_verified": len(artifacts["files"]),
                "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
                "full_checkpoints_verified": 18, "v1_full_checkpoint_256_matches": 6,
                "initial_weights_paired": True, "distinct_validation_positions": 2048,
                "checkpoint_validation_windows": 288, "saved_native_predictions_recomputed": 36864,
                "saved_native_gradient_entries_recomputed": 4718592,
                "saved_route_digest_entries_recomputed": 288 * (shape.WORDS_AD_HEAD + shape.WORDS_PROJ_IDX + shape.WORDS_TAP_IDX),
                "physical_fpga_execution": False, "heldout_test_evaluated": False,
                "audit_performs_cloud_operations": False}
    prior._assert_equal({name: receipt.get(name) for name in required}, required, "completed source-bound audit receipt")
    cells, by_arm = [], {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            name = "%s_seed_%d" % (arm, seed)
            directory = root / name
            worker = experiment.verify_worker(directory, manifest, arm, seed, library["sha256"])
            execution = prior._read(root / (name + "_execution.json"))
            if (type(execution.get("returncode")) is not int or execution["returncode"] != 0
                    or execution.get("timed_out") is not False or execution.get("timeout_s") != 900):
                raise ValueError("extension worker lacks bounded successful execution")
            cell = {"arm": arm, "seed": seed, "steps": worker["steps"],
                    "training_pairs": worker["training_pairs"], "v1_checkpoint_256_equal": True}
            for key in ("initial", "intermediate", "final"):
                evaluation = worker[key]
                cell[key] = {**prior._compact_evaluation(evaluation),
                             "baseline_training_pairs": evaluation["baseline_training_pairs"]}
            cells.append(cell)
            by_arm[arm].append(worker)
    prior._assert_equal(receipt["cells"], cells, "audited checkpoint/metric/native-output links")
    gates = {arm: screen.gate_arm(rows) for arm, rows in by_arm.items()}
    prior._assert_equal(receipt["gates"], gates, "historical audited gates")
    prior._assert_equal(summary["gates"], gates, "historical run gates")
    prior._assert_equal(summary["measurements"],
                        {arm: [experiment._compact_worker(row) for row in rows] for arm, rows in by_arm.items()},
                        "historical parent measurements")
    prior._assert_equal(summary["paired_initialization_checks"], experiment._verify_paired_initializations(root),
                        "paired initial checkpoints")
    compact_controls = {step: {name: controls[name] for name in ("unigram", "bigram")}
                        for step, controls in by_arm[ARMS[0]][0]["baselines_by_step"].items()}
    prior._assert_equal(receipt["baselines_by_step"], compact_controls, "audited matched controls")
    prior._assert_equal(receipt["environment"], prior._read(root / "anchors_seed_0/manifest.json")["environment"],
                        "audited runtime")
    return {"root": root, "audit_path": audit_path, "audit_bytes": audit_bytes,
            "artifact_bytes": artifact_bytes, "manifest_bytes": manifest_bytes,
            "manifest": manifest, "receipt": receipt, "by_arm": by_arm,
            "raw": raw, "dataset": dataset, "shape": shape, "frozen_sources": frozen}


def _recheck_inputs(context):
    if source_identity() != context["frozen_sources"]:
        raise ValueError("diagnostic or dependency sources changed during evaluation")
    if context["audit_path"].read_bytes() != context["audit_bytes"]:
        raise ValueError("completed audit receipt changed during evaluation")
    if (context["root"] / "manifest.json").read_bytes() != context["manifest_bytes"]:
        raise ValueError("source run manifest changed during evaluation")
    if (context["root"] / "artifact_manifest.json").read_bytes() != context["artifact_bytes"]:
        raise ValueError("source artifact manifest changed during evaluation")
    experiment._verify_artifacts(context["root"], experiment.SCHEMA)
    manifest = context["manifest"]
    if experiment.v1_binding(manifest["v1_evidence"]["run_directory"], context["dataset"]) != manifest["v1_evidence"]:
        raise ValueError("historical v1 evidence changed during evaluation")
    if pilot.digest(Path(context["dataset"]["path"]).read_bytes()) != context["dataset"]["sha256"]:
        raise ValueError("source corpus changed during evaluation")


def _canonical_check(directory, row, weights, tokens, targets, shape, extracted, score, zshift):
    from manuscript.tools import audit_fpga_lm_head_screen as prior
    reference = pilot.integer_forward(weights, tokens, targets, shape, zshift=zshift)
    own = {"predictions": score["predictions"], "gradients": score["gradients"],
           "lut_normalized_nll": score["lut_normalized_nll"], "accuracy": score["accuracy"],
           "saturation_events": extracted["saturation_events"],
           "rounded_zero_target_mass": score["rounded_zero_target_mass"]}
    prior._assert_equal(own, reference, "canonical independent integer forward")
    if extracted["predictions"] != score["predictions"]:
        raise ValueError("logit extraction and native argmax semantics differ")
    for name in ("predictions", "gradients"):
        saved = prior._unpack(directory / row["native_arrays"][name]["file"])
        if saved != score[name]:
            raise ValueError("canonical shift differs from actual saved native " + name)
    for name in ("lut_normalized_nll", "accuracy", "saturation_events", "rounded_zero_target_mass"):
        prior._assert_equal(own[name], row[name], "canonical saved metric " + name)


def diagnostic(run, audit_record):
    """Compute every fixed grid cell; never train, execute native code, select or promote."""
    import numpy as np
    from manuscript.tools import audit_fpga_lm_head_screen as prior
    context = validate_inputs(run, audit_record)
    root, shape, dataset, raw = (context[name] for name in ("root", "shape", "dataset", "raw"))
    grid, checked = [], {"windows": 0, "predictions": 0, "gradient_entries": 0}
    for arm in ARMS:
        for worker in context["by_arm"][arm]:
            seed = worker["seed"]
            directory = root / ("%s_seed_%d" % (arm, seed))
            original_shift = 0 if arm == "anchors_fine" else 4
            for step, key in ((256, "intermediate"), (1024, "final")):
                evaluation = worker[key]
                weights = prior._weights(directory, evaluation["checkpoint"], shape)
                scores_by_shift = {shift: [] for shift in SHIFTS}
                rows_by_shift = {shift: [] for shift in SHIFTS}
                for row in evaluation["per_window"]:
                    index = row["window_index"]
                    tokens, targets = pilot.window(raw, dataset, "validation", index)
                    extracted = integer_logits(weights, tokens, shape)
                    original = score_logits(extracted["logits"], targets, original_shift)
                    _canonical_check(directory, row, weights, tokens, targets, shape, extracted, original, original_shift)
                    checked["windows"] += 1
                    checked["predictions"] += len(original["predictions"])
                    checked["gradient_entries"] += len(original["gradients"])
                    logit_hash = pilot.digest(screen._i32([value for values in extracted["logits"] for value in values]))
                    for shift in SHIFTS:
                        score = original if shift == original_shift else score_logits(extracted["logits"], targets, shift)
                        if score["predictions"] != original["predictions"]:
                            raise ValueError("evaluation LUT shift unexpectedly changed raw-logit predictions")
                        scores_by_shift[shift].append(score)
                        rows_by_shift[shift].append({"window_index": index, **summarize_scores([score]),
                                                     "saturation_events": extracted["saturation_events"],
                                                     "logits_sha256": logit_hash,
                                                     "prediction_sha256": pilot.digest(screen._i32(score["predictions"])),
                                                     "gradient_sha256": pilot.digest(screen._i32(score["gradients"]))})
                for shift in SHIFTS:
                    grid.append({"arm": arm, "seed": seed, "checkpoint_step": step,
                                 "checkpoint_sha256": evaluation["checkpoint"]["sha256"],
                                 "training_lut_shift": original_shift, "evaluation_lut_shift": shift,
                                 "is_original_recipe_shift": shift == original_shift,
                                 "metrics": summarize_scores(scores_by_shift[shift]),
                                 "per_window": rows_by_shift[shift],
                                 "canonical_saved_native_outputs_equal": True,
                                 "raw_logit_predictions_invariant_across_grid": True})
                print("Diagnosed calibration %s seed%d checkpoint%d" % (arm, seed, step), flush=True)
    if len(grid) != 60 or checked != {"windows": 192, "predictions": 24576, "gradient_entries": 3145728}:
        raise ValueError("incomplete fixed calibration grid or canonical native-output coverage")
    _recheck_inputs(context)
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "run_directory": str(root),
            "audit_record": {"file": str(context["audit_path"]), "sha256": pilot.digest(context["audit_bytes"])},
            "artifact_manifest_sha256": pilot.digest(context["artifact_bytes"]),
            "source_hashes": context["frozen_sources"], "dataset": dataset,
            "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
            "canonical_checks": checked, "grid_cells": len(grid), "grid": grid,
            "historical_trained_recipe_gates": context["receipt"]["gates"],
            "historical_gates_modified": False, "winner_selected": False,
            "training_performed": False, "native_execution": False, "cloud_operations": False,
            "heldout_test_evaluated": False,
            "limitations": ["Post-hoc scoring on development windows already used for the budget extension.",
                            "Evaluation shift changes confidence, not checkpoint training, logits, routes or argmax accuracy.",
                            "No grid cell is selected or substituted into the historical trained-recipe gates.",
                            "A future calibration/training policy requires a separately declared evaluation protocol.",
                            "Normalized LUT probabilities precede the rounded balanced-gradient surrogate.",
                            "Rounded-zero target mass is not zero normalized probability or zero target gradient.",
                            "Integer LUT energies may reach their minimum below index255; last-bin counts do not cover every energy-floor event.",
                            "Last-bin boundary/clipping and rounded-zero groups overlap; do not sum their loss contributions.",
                            "No conventional softmax perplexity, statistical significance, new silicon, speed or energy claim."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    config = parser.parse_args(argv)
    output, root = config.output.resolve(), config.run.resolve(strict=True)
    _, manifest = _read_record(root / "manifest.json")
    historical = Path(manifest["v1_evidence"]["run_directory"]).resolve(strict=True)
    if (output.exists() or output == root or root in output.parents
            or output == historical or historical in output.parents):
        parser.error("output must be a new path outside both immutable run directories")
    result = diagnostic(root, config.audit_record)
    pilot.write_json(output, result)
    print(json.dumps({"complete": result["complete"], "output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
