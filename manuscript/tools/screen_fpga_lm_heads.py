"""Declared v1 CPU/CSIM head-repair screen; no FPGA, tuning or speed claims.

Four fixed arms, three paired seeds, 256 fresh training windows and validation
windows 8..23. Each arm/seed runs in a new process because the frozen D512 kernel
has static resident weights. The test partition is reserved, never evaluated.
All output directories and records are exclusive; failures are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_head_screen.v1"
ARMS = ("reference", "wider", "anchors", "anchors_fine")
SEEDS = (0, 1, 2)
WIDTH = 512
TRAIN_STEPS = 256
VALIDATION_INDICES = tuple(range(8, 24))
WORKER_TIMEOUT_S = 600
PLAN = REPO / "manuscript/research/PLAN.md"
OBSERVER = REPO / "manuscript/tools/observe_fpga_lm_routes.py"


def source_identity():
    """Executable identities; the plan is separately archived as a historical hash."""
    records = pilot.source_identity()
    for path in (Path(__file__).resolve(), OBSERVER):
        records[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return records


def recipe():
    return {
        "width": WIDTH, "vocabulary": 128, "sequence": 128, "layers": 4, "batch": 1,
        "arms": list(ARMS), "seeds": list(SEEDS), "steps": TRAIN_STEPS,
        "training_indices": list(range(TRAIN_STEPS)),
        "validation_indices": list(VALIDATION_INDICES), "validation_steps": [0, TRAIN_STEPS],
        "lr": 1, "threshold": 2048, "loss": "balanced LUT surrogate",
        "head_initialization": {
            "reference": "unchanged existing LCG jitter [-4,4]; zshift=4",
            "wider": "separate existing LCG seed XOR 0xA5A5A5A5; randint(-512,512); zshift=4",
            "anchors": "retain jitter at (5*v+seed)%512, subtract8192 elsewhere; zshift=4",
            "anchors_fine": "same weights as anchors; zshift=0"},
        "baseline": "alpha=1,128 symbols; target counts and within-window token-target pairs only",
        "mechanism_evaluation": "final validation, separately for every seed",
        "gates": {"maximum_zero_head_vjp_fraction": 0.25,
                  "minimum_fraction_with_16_head_winners": 0.75,
                  "maximum_saturation_events": 0, "unigram_advantage_nats": 0.05,
                  "final_context_advantage_nats": 0.02, "context_advantage_growth_nats": 0.02,
                  "mean_bigram_advantage_nats": 0.02, "maximum_single_seed_bigram_loss_nats": 0.02},
        "gate_float_boundary_tolerance_nats": 1e-12,
        "worker_timeout_s": WORKER_TIMEOUT_S, "heldout_test_evaluated": False,
        "scope": "fixed CPU/CSIM mechanism screen; no adaptive extension, statistical significance, "
                 "ordinary softmax perplexity, speed advantage or hardware execution claim"}


def initialize_buffer(shape, seed, arm):
    """Generate the original buffer before changing only the declared head/config."""
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("screen requires one of the four declared arms and seeds 0,1,2")
    if shape.ML_D != 512 or shape.ML_V != 128:
        raise ValueError("the declared head interventions require D512 and vocabulary128")
    initial = shape.build_init(seed)
    start = shape.OFF_W_HEAD
    if arm == "wider":
        rng = shape._LCG(seed ^ 0xA5A5A5A5)
        initial[start:start + shape.WORDS_W_HEAD] = [
            rng.randint(-512, 512) for _ in range(shape.WORDS_W_HEAD)]
    elif arm in ("anchors", "anchors_fine"):
        for vocabulary in range(shape.ML_V):
            anchor = (5 * vocabulary + seed) % shape.ML_D
            for feature in range(shape.ML_D):
                if feature != anchor:
                    initial[start + vocabulary * shape.ML_D + feature] -= 8192
    initial[shape.OFF_CFG] = 1
    initial[shape.OFF_CFG + 1] = 0 if arm == "anchors_fine" else 4
    return initial


def matched_baselines(raw, dataset, train_indices=tuple(range(256)),
                      eval_indices=tuple(range(8, 24)), sequence=128):
    """Fit alpha-one controls to exactly the same supervised observations."""
    train_indices, eval_indices = tuple(train_indices), tuple(eval_indices)
    if (not train_indices or not eval_indices or len(set(train_indices)) != len(train_indices)
            or len(set(eval_indices)) != len(eval_indices)):
        raise ValueError("baseline windows must be nonempty and unrepeated")
    vocab = 128
    unigram = [1] * vocab
    bigram = [[1] * vocab for _ in range(vocab)]
    for index in train_indices:
        tokens, targets = pilot.window(raw, dataset, "train", index, sequence)
        if len(tokens) != sequence or len(targets) != sequence:
            raise ValueError("incomplete baseline training window")
        for token, target in zip(tokens, targets):
            if not 0 <= token < vocab or not 0 <= target < vocab:
                raise ValueError("baseline token outside ASCII128")
            unigram[target] += 1
            bigram[token][target] += 1
    uni_total, bi_totals = sum(unigram), [sum(row) for row in bigram]
    uni_prediction = max(range(vocab), key=lambda i: unigram[i])
    bi_predictions = [max(range(vocab), key=lambda i: row[i]) for row in bigram]
    scores = {"unigram": [], "bigram": []}
    for index in eval_indices:
        tokens, targets = pilot.window(raw, dataset, "validation", index, sequence)
        for name in scores:
            if name == "unigram":
                losses = [math.log(uni_total) - math.log(unigram[target]) for target in targets]
                predictions = [uni_prediction] * sequence
            else:
                losses = [math.log(bi_totals[token]) - math.log(bigram[token][target])
                          for token, target in zip(tokens, targets)]
                predictions = [bi_predictions[token] for token in tokens]
            scores[name].append({"window_index": index, "nll": sum(losses) / sequence,
                                 "accuracy": sum(a == b for a, b in zip(predictions, targets)) / sequence})
    return {"training_pairs": len(train_indices) * sequence, "alpha": 1, "vocabulary": vocab,
            "train_indices": list(train_indices), "validation_indices": list(eval_indices),
            "unigram_target_counts": unigram, "bigram_pair_counts": bigram,
            **{name: {"nll": sum(row["nll"] for row in rows) / len(rows),
                      "accuracy": sum(row["accuracy"] for row in rows) / len(rows),
                      "per_window": rows} for name, rows in scores.items()}}


def gate_arm(seed_records):
    """Fixed all-seed screening rules, never a significance or paid-launch decision."""
    pilot._finite_json(json.dumps(seed_records, allow_nan=False))
    if (len(seed_records) != len(SEEDS)
            or sorted(row["seed"] for row in seed_records) != list(SEEDS)
            or any(type(row["seed"]) is not int for row in seed_records)):
        raise ValueError("gates require exactly one complete record for each seed 0,1,2")
    rows = []
    for record in sorted(seed_records, key=lambda item: item["seed"]):
        initial, final = record["initial"], record["final"]
        routes = final["route_diagnostics"]
        positions = routes["positions"]
        nonzero = routes["nonzero_logit_gradient_positions"]
        zero = routes["exact_zero_head_input_vjp_on_nonzero_gradient_positions"]
        diverse = routes["head_winners_ge16_positions"]
        if (any(type(value) is not int for value in (positions, nonzero, zero, diverse))
                or not 0 <= zero <= nonzero <= positions or not 0 <= diverse <= positions):
            raise ValueError("invalid route diagnostic counts")
        zero_fraction = zero / nonzero if nonzero else None
        diverse_fraction = diverse / positions if positions else None
        initial_context = initial["context_disabled_nll"] - initial["lut_normalized_nll"]
        final_context = final["context_disabled_nll"] - final["lut_normalized_nll"]
        unigram_advantage = record["baselines"]["unigram"]["nll"] - final["lut_normalized_nll"]
        bigram_advantage = record["baselines"]["bigram"]["nll"] - final["lut_normalized_nll"]
        mechanism = (nonzero > 0 and positions == len(VALIDATION_INDICES) * 128 and zero_fraction <= 0.25
                     and diverse_fraction >= 0.75 and final["saturation_events"] == 0
                     and final["native_reference_equal"] is True
                     and final["native_routes_equal"] is True
                     and routes.get("backward_int32_overflow_detected") is False)
        context = (unigram_advantage >= 0.05 - 1e-12 and final_context >= 0.02 - 1e-12
                   and final_context - initial_context >= 0.02 - 1e-12)
        rows.append({"seed": record["seed"], "mechanism": mechanism, "context": context,
                     "zero_head_vjp_fraction_on_nonzero_logit_gradients": zero_fraction,
                     "fraction_with_at_least_16_head_winners": diverse_fraction,
                     "unigram_advantage_nats": unigram_advantage,
                     "bigram_advantage_nats": bigram_advantage,
                     "initial_context_advantage_nats": initial_context,
                     "final_context_advantage_nats": final_context,
                     "context_advantage_growth_nats": final_context - initial_context})
    mechanism = all(row["mechanism"] for row in rows)
    context = all(row["context"] for row in rows)
    mean_bigram = sum(row["bigram_advantage_nats"] for row in rows) / len(rows)
    bigram = (mean_bigram >= 0.02 - 1e-12
              and all(row["bigram_advantage_nats"] >= -0.02 - 1e-12 for row in rows))
    return {"per_seed": rows, "mechanism": mechanism, "context": context,
            "mean_bigram_advantage_nats": mean_bigram, "bigram": bigram,
            "larger_paid_learning_benchmark_gate": mechanism and context and bigram,
            "statistical_significance_claim": False, "paid_execution_authorized": False}


def _i32(values):
    return struct.pack("<%di" % len(values), *values)


def _save_array(directory, filename, values):
    payload = _i32(values)
    with (directory / filename).open("xb") as stream:
        stream.write(payload)
    return {"file": filename, "words": len(values), "bytes": len(payload),
            "sha256": pilot.digest(payload), "format": "little-endian signed int32"}


def _scalar_result(result):
    return {key: value for key, value in result.items()
            if key not in ("predictions", "gradients", "native_routes", "parameter_gradients")}


def _input_probe(original, candidate):
    if (len(original["predictions"]) != len(candidate["predictions"])
            or len(original["gradients"]) != len(candidate["gradients"])
            or not original["predictions"] or not original["gradients"]):
        raise ValueError("probe array length mismatch")
    return {"lut_normalized_nll": candidate["lut_normalized_nll"], "accuracy": candidate["accuracy"],
            "prediction_mismatches": sum(a != b for a, b in zip(original["predictions"], candidate["predictions"])),
            "rounded_gradient_mismatches": sum(a != b for a, b in zip(original["gradients"], candidate["gradients"])),
            "maximum_absolute_rounded_gradient_difference": max(
                abs(a - b) for a, b in zip(original["gradients"], candidate["gradients"])),
            "prediction_sha256": pilot.digest(_i32(candidate["predictions"])),
            "gradient_sha256": pilot.digest(_i32(candidate["gradients"])),
            "saturation_events": candidate["saturation_events"],
            "rounded_zero_target_mass": candidate["rounded_zero_target_mass"]}


def _aggregate(rows):
    if not rows:
        raise ValueError("cannot aggregate empty validation")
    count_names = ("positions", "nonzero_logit_gradient_positions",
                   "exact_zero_head_input_vjp_on_nonzero_gradient_positions",
                   "head_winners_ge16_positions", "single_shared_head_winner_positions",
                   "nonzero_embedding_input_vjp_positions")
    routes = {name: sum(row["route_diagnostics"][name] for row in rows) for name in count_names}
    for name in ("max_abs_head_input_vjp", "max_abs_embedding_input_vjp"):
        routes[name] = max(row["route_diagnostics"][name] for row in rows)
    routes["backward_int32_overflow_detected"] = any(
        row["route_diagnostics"]["backward_int32_overflow_detected"] for row in rows)
    routes["head_winner_counts"] = [value for row in rows for value in row["route_diagnostics"]["head_winner_counts"]]
    return {"lut_normalized_nll": sum(row["lut_normalized_nll"] for row in rows) / len(rows),
            "accuracy": sum(row["accuracy"] for row in rows) / len(rows),
            "context_disabled_nll": sum(row["probes"]["context_disabled"]["lut_normalized_nll"] for row in rows) / len(rows),
            "saturation_events": sum(row["saturation_events"] for row in rows),
            "rounded_zero_target_mass": sum(row["rounded_zero_target_mass"] for row in rows),
            "native_reference_equal": all(row["native_reference_equal"] is True for row in rows),
            "native_routes_equal": all(row["native_routes_equal"] is True for row in rows),
            "route_diagnostics": routes, "per_window": rows}


def _artifact_manifest(directory):
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks cannot be screen artifacts")
        if path.is_file():
            entries.append({"file": path.relative_to(directory).as_posix(),
                            "bytes": path.stat().st_size, "sha256": pilot.digest(path.read_bytes())})
    return {"schema": SCHEMA, "files": entries}


def _assert_frozen(expected, library, library_hash):
    if source_identity() != expected or pilot.digest(Path(library).read_bytes()) != library_hash:
        raise RuntimeError("screen source/library changed during execution; evidence is invalid")


def _common_manifest(corpus):
    shape = pilot.layout(WIDTH)
    raw, dataset = pilot.corpus_plan(corpus, TRAIN_STEPS, max(VALIDATION_INDICES) + 1, shape.ML_L_SEQ)
    return raw, shape, {"schema": SCHEMA, "backend": "csim", "recipe": recipe(),
                        "dataset": dataset, "layout": pilot.layout_values(shape),
                        "source_hashes": source_identity(),
                        "plan_document": {"file": PLAN.relative_to(REPO).as_posix(),
                                          "sha256": pilot.digest(PLAN.read_bytes())}}


def run_worker(config, bridge_factory=None, observer=None):
    """One fresh process owns exactly one arm/seed; injectable transport for unit tests."""
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bridge, completed = None, 0
    try:
        import numpy as np
        if observer is None:
            from manuscript.tools.observe_fpga_lm_routes import observe
            observer = observe
        raw, shape, manifest = _common_manifest(config.corpus)
        initial = initialize_buffer(shape, config.seed, config.arm)
        library = Path(config.library).resolve(strict=True)
        library_hash = pilot.digest(library.read_bytes())
        parent_manifest = getattr(config, "parent_manifest", None)
        if parent_manifest:
            expected = pilot._finite_json(Path(parent_manifest).read_text())
            if any(manifest[key] != expected[key] for key in manifest):
                raise RuntimeError("worker recipe/source/corpus differs from parent manifest")
            binding = pilot._finite_json((Path(parent_manifest).parent / "library.json").read_text())
            if library_hash != binding["sha256"] or str(library) != binding["path"]:
                raise RuntimeError("worker library differs from parent build")
        manifest.update(arm=config.arm, seed=config.seed,
                        zshift=initial[shape.OFF_CFG + 1], library_sha256=library_hash,
                        started_utc=datetime.now(timezone.utc).isoformat(),
                        environment={"python": sys.version, "numpy": np.__version__,
                                     "platform": platform.platform()},
                        initial_buffer_sha256=pilot.digest(_i32(initial)))
        controls = matched_baselines(raw, manifest["dataset"], range(TRAIN_STEPS),
                                     VALIDATION_INDICES, shape.ML_L_SEQ)
        pilot.write_json(output / "manifest.json", manifest)
        pilot.write_json(output / "baselines.json", controls)
        bridge = (bridge_factory or pilot.NativeBridge)(library, "csim", shape)
        bridge.write(0, initial)
        if bridge.read(0, len(initial)) != initial:
            raise RuntimeError("complete initialization transport mismatch")
        weights = pilot.read_weights(bridge, shape)
        zshift = manifest["zshift"]

        def evaluate(step, current):
            snapshot = pilot.checkpoint(output, step, current, shape)
            rows = []
            for index in VALIDATION_INDICES:
                tokens, targets = pilot.window(raw, manifest["dataset"], "validation", index, shape.ML_L_SEQ)
                observed = observer(current, tokens, targets, shape, zshift=zshift)
                reference = pilot.integer_forward(current, tokens, targets, shape, zshift=zshift)
                if (any(observed[key] != reference[key] for key in reference if key != "lut_normalized_nll")
                        or not math.isclose(observed["lut_normalized_nll"], reference["lut_normalized_nll"],
                                            rel_tol=0, abs_tol=1e-12)):
                    raise RuntimeError("route observer differs from frozen independent forward")
                pilot.stage_window(bridge, shape, tokens, targets)
                bridge.step(0, 2048)
                predictions = bridge.read(shape.OFF_PRED, shape.ML_L_SEQ)
                gradients = bridge.read(shape.OFF_GRAD_LOGIT, shape.WORDS_GRAD_LOGIT)
                prefix = "eval_%06d_window_%03d" % (step, index)
                arrays = {"predictions": _save_array(output, prefix + "_predictions.i32", predictions),
                          "gradients": _save_array(output, prefix + "_gradients.i32", gradients)}
                # Store native evidence even when it disagrees; never bless a partial match.
                if predictions != reference["predictions"] or gradients != reference["gradients"]:
                    raise RuntimeError("native prediction or balanced loss-gradient mismatch")
                route_hashes = {}
                for name, expected in observed["native_routes"].items():
                    count = getattr(shape, "WORDS_" + name)
                    actual = bridge.read(getattr(shape, "OFF_" + name), count)
                    if len(expected) != count or actual != expected:
                        raise RuntimeError("native saved-route mismatch: " + name)
                    route_hashes[name] = {"words": count, "sha256": pilot.digest(_i32(actual))}
                probes = {}
                for name, probe_tokens, context in (
                        ("context_disabled", tokens, False), ("all_space", [32] * len(tokens), True),
                        ("reversed_tokens", list(reversed(tokens)), True)):
                    candidate = pilot.integer_forward(current, probe_tokens, targets, shape,
                                                      zshift=zshift, use_context=context)
                    probes[name] = _input_probe(reference, candidate)
                paired = {name: controls[name]["per_window"][len(rows)]["nll"] - reference["lut_normalized_nll"]
                          for name in ("unigram", "bigram")}
                row = {"window_index": index, **_scalar_result(observed), "probes": probes,
                       "native_arrays": arrays, "native_route_hashes": route_hashes,
                       "native_reference_equal": True, "native_routes_equal": True,
                       "baseline_minus_model_nll": paired}
                pilot.write_json(output / (prefix + ".json"), row)
                rows.append(row)
            if pilot.weight_bytes(pilot.read_weights(bridge, shape)) != pilot.weight_bytes(current):
                raise RuntimeError("frozen evaluation changed full weights")
            result = {"step": step, "checkpoint": snapshot, "weights_unchanged": True, **_aggregate(rows)}
            pilot.write_json(output / ("evaluation_%06d.json" % step), result)
            return result

        initial_eval = evaluate(0, weights)
        with (output / "training.jsonl").open("x", encoding="utf-8") as log, \
                (output / "training_predictions.i32").open("xb") as predictions_log, \
                (output / "training_gradients.i32").open("xb") as gradients_log:
            for index in range(TRAIN_STEPS):
                tokens, targets = pilot.window(raw, manifest["dataset"], "train", index, shape.ML_L_SEQ)
                pilot.stage_window(bridge, shape, tokens, targets)
                bridge.step(1, 2048)
                predictions = bridge.read(shape.OFF_PRED, shape.ML_L_SEQ)
                gradients = bridge.read(shape.OFF_GRAD_LOGIT, shape.WORDS_GRAD_LOGIT)
                predictions_log.write(_i32(predictions))
                gradients_log.write(_i32(gradients))
                completed = index + 1
                log.write(json.dumps({"step": completed, "window_index": index,
                                      "token_pairs": shape.ML_L_SEQ, "input_readback_equal": True,
                                      "predictions_sha256": pilot.digest(_i32(predictions)),
                                      "gradients_sha256": pilot.digest(_i32(gradients))}, allow_nan=False) + "\n")
                log.flush()
        final_eval = evaluate(TRAIN_STEPS, pilot.read_weights(bridge, shape))
        to_close, bridge = bridge, None
        to_close.close()
        _assert_frozen(manifest["source_hashes"], library, library_hash)
        summary = {"schema": SCHEMA, "complete": True, "arm": config.arm, "seed": config.seed,
                   "steps": completed, "training_pairs": completed * shape.ML_L_SEQ,
                   "initial": initial_eval, "final": final_eval, "baselines": controls,
                   "heldout_test_evaluated": False, "hardware_execution": False,
                   "speed_advantage_claim": False}
        pilot.write_json(output / "summary.json", summary)
        pilot.write_json(output / "artifact_manifest.json", _artifact_manifest(output))
        return summary
    except BaseException as error:
        pilot.write_json(output / "failure.json", {"complete": False, "steps_completed": completed,
                                                   "error": str(error), "type": type(error).__name__})
        raise
    finally:
        if bridge is not None:
            try:
                bridge.close()
            except Exception as error:
                pilot.write_json(output / "cleanup_failure.json", {"error": str(error), "type": type(error).__name__})


def verify_worker(directory, expected, arm, seed, library_hash):
    """Verify actual payload bytes and declared coverage, not merely child exit status."""
    directory = Path(directory)
    if any((directory / name).exists() for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("failed worker cannot count as completed evidence")
    artifacts = pilot._finite_json((directory / "artifact_manifest.json").read_text())
    actual = _artifact_manifest(directory)
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    if artifacts != actual:
        raise ValueError("worker artifact manifest does not cover actual unchanged files")
    manifest = pilot._finite_json((directory / "manifest.json").read_text())
    if any(manifest[key] != expected[key] for key in expected):
        raise ValueError("worker source/protocol/corpus identity mismatch")
    if (manifest.get("arm") != arm or manifest.get("seed") != seed
            or manifest.get("library_sha256") != library_hash):
        raise ValueError("worker identity differs from declared arm/seed/library")
    result = pilot._finite_json((directory / "summary.json").read_text())
    if (result.get("schema") != SCHEMA or result.get("complete") is not True
            or result.get("steps") != TRAIN_STEPS or result.get("arm") != arm or result.get("seed") != seed
            or result.get("training_pairs") != TRAIN_STEPS * 128
            or result.get("heldout_test_evaluated") is not False):
        raise ValueError("worker summary lacks complete declared recipe")
    shape = pilot.layout(WIDTH)
    expected_initial = initialize_buffer(shape, seed, arm)
    initial_weights = {name: expected_initial[getattr(shape, "OFF_" + name):
                                             getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
                       for name in pilot.WEIGHTS}
    if (manifest.get("initial_buffer_sha256") != pilot.digest(_i32(expected_initial))
            or manifest.get("zshift") != (0 if arm == "anchors_fine" else 4)):
        raise ValueError("worker initial buffer differs from declared recipe")
    baseline_record = pilot._finite_json((directory / "baselines.json").read_text())
    if result["baselines"] != baseline_record:
        raise ValueError("worker baseline record link differs")
    raw = Path(expected["dataset"]["path"]).read_bytes()
    if pilot.digest(raw) != expected["dataset"]["sha256"]:
        raise ValueError("corpus changed while verifying worker")
    if baseline_record != matched_baselines(raw, expected["dataset"], range(TRAIN_STEPS),
                                           VALIDATION_INDICES, shape.ML_L_SEQ):
        raise ValueError("worker baselines do not use the declared matched data")
    for step, key in ((0, "initial"), (TRAIN_STEPS, "final")):
        record = result[key]
        checkpoint = record["checkpoint"]
        filename = "checkpoint_%06d.i32" % step
        expected_regions = [{"name": name, "words": getattr(shape, "WORDS_" + name)} for name in pilot.WEIGHTS]
        payload = (directory / filename).read_bytes()
        if step == 0 and payload != pilot.weight_bytes(initial_weights):
            raise ValueError("actual initial checkpoint does not reproduce declared arm/seed")
        if (checkpoint["file"] != filename or checkpoint["regions"] != expected_regions
                or checkpoint["step"] != step or checkpoint["bytes"] != sum(r["words"] for r in expected_regions) * 4
                or checkpoint["bytes"] != len(payload) or checkpoint["sha256"] != pilot.digest(payload)
                or record["step"] != step or record["weights_unchanged"] is not True):
            raise ValueError("worker full checkpoint coverage mismatch")
        if pilot._finite_json((directory / ("checkpoint_%06d.json" % step)).read_text()) != checkpoint:
            raise ValueError("worker checkpoint link differs from actual record")
        if pilot._finite_json((directory / ("evaluation_%06d.json" % step)).read_text()) != record:
            raise ValueError("worker evaluation link differs from actual record")
        if [row["window_index"] for row in record["per_window"]] != list(VALIDATION_INDICES):
            raise ValueError("worker validation coverage mismatch")
        for name, value in _aggregate(record["per_window"]).items():
            if record[name] != value:
                raise ValueError("worker aggregate differs from complete per-window evidence")
        for index, row in enumerate(record["per_window"]):
            prefix = "eval_%06d_window_%03d" % (step, row["window_index"])
            if pilot._finite_json((directory / (prefix + ".json")).read_text()) != row:
                raise ValueError("worker validation row link mismatch")
            if row["native_reference_equal"] is not True or row["native_routes_equal"] is not True:
                raise ValueError("worker lacks independent native agreement")
            differences = {name: baseline_record[name]["per_window"][index]["nll"] - row["lut_normalized_nll"]
                           for name in ("unigram", "bigram")}
            if row["baseline_minus_model_nll"] != differences:
                raise ValueError("worker paired differences do not match refitted baselines")
            for name, count in (("predictions", shape.ML_L_SEQ), ("gradients", shape.WORDS_GRAD_LOGIT)):
                array = row["native_arrays"][name]
                filename = prefix + "_" + name + ".i32"
                payload = (directory / filename).read_bytes()
                if (array["file"] != filename or array["words"] != count or array["bytes"] != count * 4
                        or len(payload) != count * 4 or array["sha256"] != pilot.digest(payload)):
                    raise ValueError("worker native array coverage/hash mismatch")
    training = [pilot._finite_json(line) for line in (directory / "training.jsonl").read_text().splitlines()]
    if ([row["window_index"] for row in training] != list(range(TRAIN_STEPS))
            or [row["step"] for row in training] != list(range(1, TRAIN_STEPS + 1))):
        raise ValueError("worker fresh training sequence mismatch")
    for name, words in (("predictions", shape.ML_L_SEQ), ("gradients", shape.WORDS_GRAD_LOGIT)):
        payload = (directory / ("training_" + name + ".i32")).read_bytes()
        stride = words * 4
        if len(payload) != TRAIN_STEPS * stride or any(
                pilot.digest(payload[index * stride:(index + 1) * stride]) != row[name + "_sha256"]
                for index, row in enumerate(training)):
            raise ValueError("worker training native array coverage/hash mismatch")
    return result


def _compact_worker(result):
    """Portable small record; full route arrays and native payloads remain in worker artifacts."""
    record = {"arm": result["arm"], "seed": result["seed"], "steps": result["steps"],
              "training_pairs": result["training_pairs"]}
    for key in ("initial", "final"):
        evaluation = result[key]
        record[key] = {name: value for name, value in evaluation.items()
                       if name not in ("per_window", "route_diagnostics")}
        record[key]["route_diagnostics"] = {name: value for name, value in evaluation["route_diagnostics"].items()
                                             if name != "head_winner_counts"}
        record[key]["per_window"] = [
            {"window_index": row["window_index"], "lut_normalized_nll": row["lut_normalized_nll"],
             "accuracy": row["accuracy"], "baseline_minus_model_nll": row["baseline_minus_model_nll"],
             "context_disabled_nll": row["probes"]["context_disabled"]["lut_normalized_nll"]}
            for row in evaluation["per_window"]]
    record["baselines"] = {name: result["baselines"][name] for name in ("unigram", "bigram")}
    return record


def _verify_paired_initializations(output):
    shape = pilot.layout(WIDTH)
    non_head_bytes = sum(getattr(shape, "WORDS_" + name) for name in pilot.WEIGHTS[:-1]) * 4
    for seed in SEEDS:
        payloads = [(output / (arm + "_seed_%d" % seed) / "checkpoint_000000.i32").read_bytes()
                    for arm in ARMS]
        if any(payload[:non_head_bytes] != payloads[0][:non_head_bytes] for payload in payloads[1:]):
            raise ValueError("paired arms do not share byte-identical non-head initial weights")
        if payloads[2] != payloads[3]:
            raise ValueError("anchor and fine-LUT arms must have identical initial full weights")
    return {"non_head_weights_equal_for_every_paired_seed": True,
            "anchor_full_weights_equal_for_every_paired_seed": True}


def _run_process(command, log_path):
    """A fixed timeout terminates only the owned process group, including descendants."""
    with Path(log_path).open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(REPO), start_new_session=True)
        try:
            return {"returncode": process.wait(timeout=WORKER_TIMEOUT_S), "timed_out": False}
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            return {"returncode": process.returncode, "timed_out": True}
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            raise


def run_screen(config):
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        raw, shape, manifest = _common_manifest(config.corpus)
        del raw, shape
        pilot.write_json(output / "manifest.json", manifest)
        pilot.write_json(output / "protocol.json", {"schema": SCHEMA, **manifest["recipe"]})
        library = pilot.build_csim(output / "build", WIDTH).resolve()
        library_hash = pilot.digest(library.read_bytes())
        pilot.write_json(output / "library.json", {"path": str(library), "sha256": library_hash})
        results, failures = {arm: [] for arm in ARMS}, []
        for arm in ARMS:
            for seed in SEEDS:
                _assert_frozen(manifest["source_hashes"], library, library_hash)
                name = arm + "_seed_%d" % seed
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm,
                           "--seed", str(seed), "--corpus", str(Path(config.corpus).resolve()),
                           "--output", str(output / name), "--library", str(library),
                           "--parent-manifest", str(output / "manifest.json")]
                execution = {"command": command, "timeout_s": WORKER_TIMEOUT_S,
                             **_run_process(command, output / (name + ".log"))}
                pilot.write_json(output / (name + "_execution.json"), execution)
                try:
                    _assert_frozen(manifest["source_hashes"], library, library_hash)
                    if execution["returncode"] != 0 or execution["timed_out"]:
                        raise RuntimeError("worker process failed or timed out")
                    results[arm].append(verify_worker(output / name, manifest, arm, seed, library_hash))
                except Exception as error:
                    failure = {"arm": arm, "seed": seed, "error": str(error), "type": type(error).__name__}
                    pilot.write_json(output / (name + "_failure.json"), failure)
                    worker_output = output / name
                    worker_output.mkdir(exist_ok=True)
                    if not (worker_output / "failure.json").exists():
                        pilot.write_json(worker_output / "failure.json", {"complete": False,
                                         "reported_by": "parent after process exit/timeout or artifact rejection", **failure})
                    failures.append(failure)
                # Failures never choose another recipe or extra updates. All twelve
                # predeclared cells are attempted; incomplete arms cannot pass gates.
        _assert_frozen(manifest["source_hashes"], library, library_hash)
        pairing = _verify_paired_initializations(output) if not failures else {"complete": False}
        gates = {arm: gate_arm(rows) if not failures
                 else {"complete": False, "larger_paid_learning_benchmark_gate": False,
                       "reason": "all twelve declared workers must complete before screening decisions"}
                 for arm, rows in results.items()}
        summary = {"schema": SCHEMA, "complete": not failures, "failures": failures,
                   "gates": gates, "declared_workers": len(ARMS) * len(SEEDS),
                   "completed_workers": sum(map(len, results.values())),
                   "paired_initialization_checks": pairing,
                   "measurements": {arm: [_compact_worker(row) for row in rows] for arm, rows in results.items()},
                   "heldout_test_evaluated": False, "statistical_significance_claim": False,
                   "speed_advantage_claim": False, "hardware_execution": False}
        if failures:
            pilot.write_json(output / "failure.json", {"complete": False, "failures": failures})
        pilot.write_json(output / "summary.json", summary)
        pilot.write_json(output / "artifact_manifest.json", _artifact_manifest(output))
        return summary
    except BaseException as error:
        if not (output / "failure.json").exists():
            pilot.write_json(output / "failure.json", {"complete": False, "error": str(error),
                                                       "type": type(error).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new exclusive output directory")
    parser.add_argument("--corpus", required=True, type=Path, help="existing local strict ASCII128 corpus")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--library", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parent-manifest", type=Path, help=argparse.SUPPRESS)
    config = parser.parse_args(argv)
    if config.worker:
        if config.arm is None or config.seed is None or config.library is None:
            parser.error("internal worker requires --arm, --seed, --library")
        result = run_worker(config)
    else:
        if any(value is not None for value in (config.arm, config.seed, config.library, config.parent_manifest)):
            parser.error("arm/seed/library overrides are only valid for isolated internal workers")
        result = run_screen(config)
    print(json.dumps({"complete": result["complete"], "output": str(config.output)}, allow_nan=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
