"""Versioned six-cell CPU/CSIM learning-budget extension of the frozen head screen.

Anchored heads, coarse/fine LUT, seeds0..2, 1024 fresh updates from initialization.
Validation is fixed at windows24..39 at updates0/256/1024. Update256 full weights
must reproduce v1 despite the different frozen validation cohort. No cloud API,
test-set evaluation, adaptive extension, or hardware/speed claim is permitted.
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
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_budget_extension.v1"
ARMS = ("anchors", "anchors_fine")
SEEDS = (0, 1, 2)
WIDTH = 512
TRAIN_STEPS = 1024
EVAL_STEPS = (0, 256, 1024)
VALIDATION_INDICES = tuple(range(24, 40))
WORKER_TIMEOUT_S = 900
PLAN = REPO / "manuscript/research/PLAN.md"
initialize_buffer = screen.initialize_buffer
matched_baselines = screen.matched_baselines
gate_arm = screen.gate_arm


def source_identity():
    records = screen.source_identity()
    path = Path(__file__).resolve()
    records[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return records


def recipe():
    value = screen.recipe()
    value.update(arms=list(ARMS), seeds=list(SEEDS), steps=TRAIN_STEPS,
                 training_indices=list(range(TRAIN_STEPS)),
                 validation_indices=list(VALIDATION_INDICES), validation_steps=list(EVAL_STEPS),
                 worker_timeout_s=WORKER_TIMEOUT_S,
                 baseline_budgets={str(step): step * 128 for step in EVAL_STEPS if step},
                 initial_baseline_comparison=False,
                 initialization_policy="from original initialization; never resume a v1 checkpoint",
                 projection_initialization="unchanged original LCG jitter for every cell",
                 checkpoint_256_policy="complete actual bytes must equal corresponding v1 arm/seed",
                 validation_cohort="new development windows24..39, not v1 paired scores or final test",
                 scope="six-cell fixed CPU/CSIM budget extension; no adaptive extension, hardware execution, "
                       "ordinary softmax perplexity, statistical significance or speed claims")
    value["head_initialization"] = {arm: value["head_initialization"][arm] for arm in ARMS}
    return value


def _read(path):
    return pilot._finite_json(Path(path).read_text())


def _artifact_manifest(directory):
    result = screen._artifact_manifest(Path(directory))
    result["schema"] = SCHEMA
    return result


def _verify_artifacts(directory, schema):
    directory = Path(directory)
    if any((directory / name).exists() for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("failed evidence cannot be accepted")
    recorded = _read(directory / "artifact_manifest.json")
    actual = screen._artifact_manifest(directory)
    actual["schema"] = schema
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    if recorded != actual:
        raise ValueError("artifact manifest does not cover actual unchanged regular files")
    return recorded


def v1_binding(v1_run, dataset):
    """Bind complete historical artifacts and the six actual 256-update checkpoints."""
    root = Path(v1_run).resolve(strict=True)
    _verify_artifacts(root, screen.SCHEMA)
    manifest, summary = _read(root / "manifest.json"), _read(root / "summary.json")
    if (manifest.get("schema") != screen.SCHEMA or manifest.get("backend") != "csim"
            or manifest.get("recipe") != screen.recipe()
            or manifest.get("source_hashes") != screen.source_identity()
            or manifest.get("layout") != pilot.layout_values(pilot.layout(WIDTH))
            or summary.get("schema") != screen.SCHEMA or summary.get("complete") is not True
            or summary.get("failures") != [] or summary.get("declared_workers") != 12
            or summary.get("completed_workers") != 12
            or any(summary.get(name) is not False for name in (
                "heldout_test_evaluated", "hardware_execution", "speed_advantage_claim",
                "statistical_significance_claim"))):
        raise ValueError("v1 reference must be the complete unchanged declared twelve-cell screen")
    if ({key: value for key, value in dataset.items() if key != "path"}
            != {key: value for key, value in manifest["dataset"].items() if key != "path"}):
        raise ValueError("extension corpus/splits differ from v1")
    library = _read(root / "library.json")
    name = Path(library["path"]).name
    if name not in ("libmorphlm.so", "libmorphlm.dylib"):
        raise ValueError("unsupported v1 library name")
    build = _read(root / "build/build.json")
    if (pilot.digest((root / "build" / name).read_bytes()) != library["sha256"]
            or build["library_sha256"] != library["sha256"]
            or build["source_hashes"] != pilot.source_identity()
            or build["layout"] != manifest["layout"]):
        raise ValueError("v1 actual library/source/layout identity differs")
    workers = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = "%s_seed_%d" % (arm, seed)
            directory = root / name
            result = screen.verify_worker(directory, manifest, arm, seed, library["sha256"])
            checkpoint = result["final"]["checkpoint"]
            payload = (directory / "checkpoint_000256.i32").read_bytes()
            if checkpoint["bytes"] != len(payload) or checkpoint["sha256"] != pilot.digest(payload):
                raise ValueError("v1 actual full checkpoint differs from its verified record")
            workers[name] = {"manifest_sha256": pilot.digest((directory / "manifest.json").read_bytes()),
                             "artifact_manifest_sha256": pilot.digest((directory / "artifact_manifest.json").read_bytes()),
                             "checkpoint_256": checkpoint}
    return {"run_directory": str(root), "schema": screen.SCHEMA,
            "manifest_sha256": pilot.digest((root / "manifest.json").read_bytes()),
            "artifact_manifest_sha256": pilot.digest((root / "artifact_manifest.json").read_bytes()),
            "workers": workers}


def read_v1_checkpoint(binding, arm, seed):
    name = "%s_seed_%d" % (arm, seed)
    checkpoint = binding["workers"][name]["checkpoint_256"]
    if arm not in ARMS or seed not in SEEDS or checkpoint["file"] != "checkpoint_000256.i32":
        raise ValueError("unexpected v1 checkpoint identity")
    root = Path(binding["run_directory"]).resolve(strict=True)
    path = root / name / "checkpoint_000256.i32"
    if path.is_symlink():
        raise ValueError("v1 checkpoint cannot be a symlink")
    payload = path.read_bytes()
    expected_bytes = sum(getattr(pilot.layout(WIDTH), "WORDS_" + region) for region in pilot.WEIGHTS) * 4
    if (len(payload) != expected_bytes or checkpoint["bytes"] != expected_bytes
            or pilot.digest(payload) != checkpoint["sha256"]):
        raise ValueError("v1 full checkpoint payload changed")
    return payload


def _common_manifest(corpus, v1_run):
    shape = pilot.layout(WIDTH)
    raw, dataset = pilot.corpus_plan(corpus, TRAIN_STEPS, max(VALIDATION_INDICES) + 1, shape.ML_L_SEQ)
    return raw, shape, {"schema": SCHEMA, "backend": "csim", "recipe": recipe(),
                        "dataset": dataset, "layout": pilot.layout_values(shape),
                        "source_hashes": source_identity(), "v1_evidence": v1_binding(v1_run, dataset),
                        "plan_document": {"file": PLAN.relative_to(REPO).as_posix(),
                                          "sha256": pilot.digest(PLAN.read_bytes())}}


def _assert_frozen(manifest, library, library_hash):
    if source_identity() != manifest["source_hashes"] or pilot.digest(Path(library).read_bytes()) != library_hash:
        raise RuntimeError("extension source/library changed during execution")
    if v1_binding(manifest["v1_evidence"]["run_directory"], manifest["dataset"]) != manifest["v1_evidence"]:
        raise RuntimeError("bound v1 evidence changed during execution")


def _evaluate(output, bridge, current, shape, raw, manifest, step, observer, controls):
    """Read-only model evaluation; all native prediction/gradient payloads are retained."""
    snapshot = pilot.checkpoint(output, step, current, shape)
    rows, zshift = [], manifest["zshift"]
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
        arrays = {"predictions": screen._save_array(output, prefix + "_predictions.i32", predictions),
                  "gradients": screen._save_array(output, prefix + "_gradients.i32", gradients)}
        if predictions != reference["predictions"] or gradients != reference["gradients"]:
            raise RuntimeError("native prediction or balanced loss-gradient mismatch")
        route_hashes = {}
        for name, expected in observed["native_routes"].items():
            count = getattr(shape, "WORDS_" + name)
            actual = bridge.read(getattr(shape, "OFF_" + name), count)
            if len(expected) != count or actual != expected:
                raise RuntimeError("native saved-route mismatch: " + name)
            route_hashes[name] = {"words": count, "sha256": pilot.digest(screen._i32(actual))}
        probes = {}
        for name, probe_tokens, context in (
                ("context_disabled", tokens, False), ("all_space", [32] * len(tokens), True),
                ("reversed_tokens", list(reversed(tokens)), True)):
            candidate = pilot.integer_forward(current, probe_tokens, targets, shape,
                                              zshift=zshift, use_context=context)
            probes[name] = screen._input_probe(reference, candidate)
        paired = ({name: controls[name]["per_window"][len(rows)]["nll"] - reference["lut_normalized_nll"]
                   for name in ("unigram", "bigram")} if controls is not None else {})
        row = {"window_index": index, **screen._scalar_result(observed), "probes": probes,
               "native_arrays": arrays, "native_route_hashes": route_hashes,
               "native_reference_equal": True, "native_routes_equal": True,
               "baseline_minus_model_nll": paired}
        pilot.write_json(output / (prefix + ".json"), row)
        rows.append(row)
    after = pilot.weight_bytes(pilot.read_weights(bridge, shape))
    if after != pilot.weight_bytes(current):
        raise RuntimeError("frozen evaluation changed full weights")
    result = {"step": step, "checkpoint": snapshot, "weights_unchanged": True,
              "baseline_training_pairs": step * shape.ML_L_SEQ if step else None,
              **screen._aggregate(rows)}
    if step == EVAL_STEPS[1]:
        expected = read_v1_checkpoint(manifest["v1_evidence"], manifest["arm"], manifest["seed"])
        if after != expected:
            raise RuntimeError("update256 full weights do not reproduce corresponding v1 arm/seed")
        result["v1_checkpoint_equal"] = True
    pilot.write_json(output / ("evaluation_%06d.json" % step), result)
    return result


def run_worker(config, bridge_factory=None, observer=None):
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bridge, completed = None, 0
    try:
        import numpy as np
        if observer is None:
            from manuscript.tools.observe_fpga_lm_routes import observe
            observer = observe
        if config.arm not in ARMS or type(config.seed) is not int or config.seed not in SEEDS:
            raise ValueError("extension requires the two anchored arms and seeds0,1,2")
        raw, shape, manifest = _common_manifest(config.corpus, config.v1_run)
        initial = initialize_buffer(shape, config.seed, config.arm)
        library = Path(config.library).resolve(strict=True)
        library_hash = pilot.digest(library.read_bytes())
        parent_manifest = getattr(config, "parent_manifest", None)
        if parent_manifest:
            expected = _read(parent_manifest)
            if any(manifest[key] != expected[key] for key in manifest):
                raise RuntimeError("worker protocol/source/corpus/v1 evidence differs from parent")
            binding = _read(Path(parent_manifest).parent / "library.json")
            if binding["path"] != str(library) or binding["sha256"] != library_hash:
                raise RuntimeError("worker library differs from parent build")
        manifest.update(arm=config.arm, seed=config.seed, zshift=initial[shape.OFF_CFG + 1],
                        library_sha256=library_hash, initial_buffer_sha256=pilot.digest(screen._i32(initial)),
                        started_utc=datetime.now(timezone.utc).isoformat(),
                        environment={"python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
        pilot.write_json(output / "manifest.json", manifest)
        bridge = (bridge_factory or pilot.NativeBridge)(library, "csim", shape)
        bridge.write(0, initial)
        if bridge.read(0, len(initial)) != initial:
            raise RuntimeError("complete initialization transport mismatch")
        evaluations, controls_by_step = {}, {}
        evaluations[0] = _evaluate(output, bridge, pilot.read_weights(bridge, shape), shape, raw,
                                   manifest, 0, observer, None)
        with (output / "training.jsonl").open("x", encoding="utf-8") as log, \
                (output / "training_predictions.i32").open("xb") as predictions_log, \
                (output / "training_gradients.i32").open("xb") as gradients_log:
            for index in range(TRAIN_STEPS):
                tokens, targets = pilot.window(raw, manifest["dataset"], "train", index, shape.ML_L_SEQ)
                pilot.stage_window(bridge, shape, tokens, targets)
                bridge.step(1, 2048)
                predictions = bridge.read(shape.OFF_PRED, shape.ML_L_SEQ)
                gradients = bridge.read(shape.OFF_GRAD_LOGIT, shape.WORDS_GRAD_LOGIT)
                predictions_log.write(screen._i32(predictions))
                gradients_log.write(screen._i32(gradients))
                completed = index + 1
                log.write(json.dumps({"step": completed, "window_index": index,
                                      "token_pairs": shape.ML_L_SEQ, "input_readback_equal": True,
                                      "predictions_sha256": pilot.digest(screen._i32(predictions)),
                                      "gradients_sha256": pilot.digest(screen._i32(gradients))}, allow_nan=False) + "\n")
                log.flush()
                if completed in EVAL_STEPS:
                    controls = matched_baselines(raw, manifest["dataset"], range(completed),
                                                 VALIDATION_INDICES, shape.ML_L_SEQ)
                    controls_by_step[str(completed)] = controls
                    pilot.write_json(output / ("baselines_%06d.json" % completed), controls)
                    evaluations[completed] = _evaluate(output, bridge, pilot.read_weights(bridge, shape),
                                                       shape, raw, manifest, completed, observer, controls)
        to_close, bridge = bridge, None
        to_close.close()
        _assert_frozen(manifest, library, library_hash)
        final_controls = controls_by_step[str(TRAIN_STEPS)]
        pilot.write_json(output / "baselines.json", final_controls)
        summary = {"schema": SCHEMA, "complete": True, "arm": config.arm, "seed": config.seed,
                   "steps": completed, "training_pairs": completed * shape.ML_L_SEQ,
                   "initial": evaluations[0], "intermediate": evaluations[EVAL_STEPS[1]],
                   "final": evaluations[TRAIN_STEPS], "baselines_by_step": controls_by_step,
                   "baselines": final_controls, "v1_checkpoint_256_equal": True,
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


def _verify_checkpoint(directory, evaluation, step, shape):
    checkpoint = evaluation["checkpoint"]
    filename = "checkpoint_%06d.i32" % step
    regions = [{"name": name, "words": getattr(shape, "WORDS_" + name)} for name in pilot.WEIGHTS]
    payload = (directory / filename).read_bytes()
    if (checkpoint.get("file") != filename or checkpoint.get("regions") != regions
            or checkpoint.get("step") != step or checkpoint.get("bytes") != sum(row["words"] for row in regions) * 4
            or checkpoint["bytes"] != len(payload) or checkpoint.get("sha256") != pilot.digest(payload)
            or checkpoint.get("location_portable") is not True or evaluation.get("step") != step
            or evaluation.get("weights_unchanged") is not True
            or _read(directory / ("checkpoint_%06d.json" % step)) != checkpoint):
        raise ValueError("full checkpoint payload/coverage/link mismatch")
    return payload


def _verify_evaluation(directory, evaluation, step, controls, shape):
    if _read(directory / ("evaluation_%06d.json" % step)) != evaluation:
        raise ValueError("evaluation link differs from actual record")
    if ([row["window_index"] for row in evaluation["per_window"]] != list(VALIDATION_INDICES)
            or evaluation.get("baseline_training_pairs") != (step * shape.ML_L_SEQ if step else None)):
        raise ValueError("validation cohort or baseline training budget differs")
    for name, value in screen._aggregate(evaluation["per_window"]).items():
        if evaluation[name] != value:
            raise ValueError("evaluation aggregate differs from complete per-window evidence")
    for index, row in enumerate(evaluation["per_window"]):
        prefix = "eval_%06d_window_%03d" % (step, row["window_index"])
        if _read(directory / (prefix + ".json")) != row:
            raise ValueError("validation row link differs from actual record")
        if row["native_reference_equal"] is not True or row["native_routes_equal"] is not True:
            raise ValueError("missing independent native prediction/gradient/route agreement")
        differences = ({name: controls[name]["per_window"][index]["nll"] - row["lut_normalized_nll"]
                        for name in ("unigram", "bigram")} if controls is not None else {})
        if row["baseline_minus_model_nll"] != differences:
            raise ValueError("baseline differences do not match the checkpoint-specific supervised budget")
        for name, count in (("predictions", shape.ML_L_SEQ), ("gradients", shape.WORDS_GRAD_LOGIT)):
            record = row["native_arrays"][name]
            filename = prefix + "_" + name + ".i32"
            payload = (directory / filename).read_bytes()
            if (record["file"] != filename or record["words"] != count or record["bytes"] != count * 4
                    or len(payload) != count * 4 or record["sha256"] != pilot.digest(payload)):
                raise ValueError("native array payload/coverage/hash mismatch")


def verify_worker(directory, expected, arm, seed, library_hash):
    directory = Path(directory)
    _verify_artifacts(directory, SCHEMA)
    manifest, result = _read(directory / "manifest.json"), _read(directory / "summary.json")
    if any(manifest[key] != expected[key] for key in expected):
        raise ValueError("worker protocol/source/corpus/v1 reference differs from expected manifest")
    if (manifest.get("arm") != arm or manifest.get("seed") != seed
            or manifest.get("library_sha256") != library_hash
            or result.get("schema") != SCHEMA or result.get("complete") is not True
            or result.get("arm") != arm or result.get("seed") != seed
            or result.get("steps") != TRAIN_STEPS or result.get("training_pairs") != TRAIN_STEPS * 128
            or result.get("v1_checkpoint_256_equal") is not True
            or any(result.get(name) is not False for name in (
                "heldout_test_evaluated", "hardware_execution", "speed_advantage_claim"))):
        raise ValueError("worker identity or complete declared recipe differs")
    shape = pilot.layout(WIDTH)
    initial = initialize_buffer(shape, seed, arm)
    initial_weights = {name: initial[getattr(shape, "OFF_" + name):
                                    getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
                       for name in pilot.WEIGHTS}
    if (manifest.get("initial_buffer_sha256") != pilot.digest(screen._i32(initial))
            or manifest.get("zshift") != (0 if arm == "anchors_fine" else 4)):
        raise ValueError("initial full buffer differs from declared anchored recipe")
    raw = Path(expected["dataset"]["path"]).read_bytes()
    if pilot.digest(raw) != expected["dataset"]["sha256"]:
        raise ValueError("corpus changed while verifying extension")
    if (set(result["baselines_by_step"]) != {str(step) for step in EVAL_STEPS if step}
            or result["baselines"] != result["baselines_by_step"][str(TRAIN_STEPS)]
            or result["baselines"] != _read(directory / "baselines.json")):
        raise ValueError("checkpoint-specific baseline coverage or final link differs")
    for step, key in zip(EVAL_STEPS, ("initial", "intermediate", "final")):
        evaluation = result[key]
        payload = _verify_checkpoint(directory, evaluation, step, shape)
        if step == 0 and payload != pilot.weight_bytes(initial_weights):
            raise ValueError("actual initial full weights do not reproduce declared arm/seed")
        if step == EVAL_STEPS[1]:
            if (payload != read_v1_checkpoint(expected["v1_evidence"], arm, seed)
                    or evaluation.get("v1_checkpoint_equal") is not True):
                raise ValueError("actual update256 full checkpoint differs from bound v1")
        controls = None
        if step:
            controls = matched_baselines(raw, expected["dataset"], range(step), VALIDATION_INDICES, shape.ML_L_SEQ)
            if (result["baselines_by_step"][str(step)] != controls
                    or _read(directory / ("baselines_%06d.json" % step)) != controls):
                raise ValueError("controls do not use exact checkpoint-specific supervised pairs")
        _verify_evaluation(directory, evaluation, step, controls, shape)
    training = [_read_json_line(line) for line in (directory / "training.jsonl").read_text().splitlines()]
    if ([row["window_index"] for row in training] != list(range(TRAIN_STEPS))
            or [row["step"] for row in training] != list(range(1, TRAIN_STEPS + 1))
            or any(row.get("input_readback_equal") is not True or row.get("token_pairs") != shape.ML_L_SEQ
                   for row in training)):
        raise ValueError("fresh training sequence/readback coverage differs")
    for name, words in (("predictions", shape.ML_L_SEQ), ("gradients", shape.WORDS_GRAD_LOGIT)):
        payload = (directory / ("training_" + name + ".i32")).read_bytes()
        stride = words * 4
        if len(payload) != TRAIN_STEPS * stride or any(
                pilot.digest(payload[index * stride:(index + 1) * stride]) != row[name + "_sha256"]
                for index, row in enumerate(training)):
            raise ValueError("training native array coverage/hash mismatch")
    return result


def _read_json_line(line):
    return pilot._finite_json(line)


def _compact_worker(result):
    value = screen._compact_worker(result)
    intermediate = dict(result, initial=result["intermediate"])
    value["intermediate"] = screen._compact_worker(intermediate)["initial"]
    value["baselines_by_step"] = {
        step: {name: controls[name] for name in ("unigram", "bigram")}
        for step, controls in result["baselines_by_step"].items()}
    value["v1_checkpoint_256_equal"] = result["v1_checkpoint_256_equal"]
    return value


def _verify_paired_initializations(root):
    root = Path(root)
    for seed in SEEDS:
        payloads = [(root / ("%s_seed_%d" % (arm, seed)) / "checkpoint_000000.i32").read_bytes()
                    for arm in ARMS]
        if payloads[0] != payloads[1]:
            raise ValueError("coarse/fine anchored arms must have identical full initial weights")
    return {"anchor_full_weights_equal_for_every_paired_seed": True,
            "non_head_weights_equal_for_every_paired_seed": True}


def _run_process(command, log_path):
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


def run_extension(config):
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        raw, shape, manifest = _common_manifest(config.corpus, config.v1_run)
        del raw, shape
        pilot.write_json(output / "manifest.json", manifest)
        pilot.write_json(output / "protocol.json", {"schema": SCHEMA, **manifest["recipe"]})
        library = pilot.build_csim(output / "build", WIDTH).resolve()
        library_hash = pilot.digest(library.read_bytes())
        pilot.write_json(output / "library.json", {"path": str(library), "sha256": library_hash})
        results, failures = {arm: [] for arm in ARMS}, []
        for arm in ARMS:
            for seed in SEEDS:
                _assert_frozen(manifest, library, library_hash)
                name = "%s_seed_%d" % (arm, seed)
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm,
                           "--seed", str(seed), "--corpus", str(Path(config.corpus).resolve()),
                           "--v1-run", manifest["v1_evidence"]["run_directory"],
                           "--output", str(output / name), "--library", str(library),
                           "--parent-manifest", str(output / "manifest.json")]
                execution = {"command": command, "timeout_s": WORKER_TIMEOUT_S,
                             **_run_process(command, output / (name + ".log"))}
                pilot.write_json(output / (name + "_execution.json"), execution)
                try:
                    _assert_frozen(manifest, library, library_hash)
                    if execution["returncode"] != 0 or execution["timed_out"]:
                        raise RuntimeError("worker failed or timed out")
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
        _assert_frozen(manifest, library, library_hash)
        pairing = _verify_paired_initializations(output) if not failures else {"complete": False}
        gates = {arm: gate_arm(rows) if not failures else {
                 "complete": False, "larger_paid_learning_benchmark_gate": False,
                 "reason": "all six declared cells must complete before screening decisions"}
                 for arm, rows in results.items()}
        summary = {"schema": SCHEMA, "complete": not failures, "failures": failures,
                   "gates": gates, "declared_workers": len(ARMS) * len(SEEDS),
                   "completed_workers": sum(map(len, results.values())),
                   "paired_initialization_checks": pairing,
                   "measurements": {arm: [_compact_worker(row) for row in rows] for arm, rows in results.items()},
                   "heldout_test_evaluated": False, "hardware_execution": False,
                   "speed_advantage_claim": False, "statistical_significance_claim": False}
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
    parser.add_argument("--v1-run", required=True, type=Path, help="complete unchanged v1 head-screen evidence")
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
            parser.error("arm/seed/library overrides are valid only for isolated workers")
        result = run_extension(config)
    print(json.dumps({"complete": result["complete"], "output": str(config.output)}, allow_nan=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
