"""Fixed intermediate-scale CSIM study with immutable same-cohort controls.

Six new shift1/2 training cells and six re-evaluated shift0/4 controls, seeds0..2.
Every cell evaluates checkpoints0/256/1024 on validation40..55. Controls use a
fresh native process per checkpoint: static resident weights cannot be safely
replaced by rewriting the host buffer in an existing D512 kernel process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_calibration as calibration  # noqa: E402
from manuscript.tools import extend_fpga_lm_budget as budget  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_training_scales.v1"
TRAIN_ARMS = ("anchors_shift1", "anchors_shift2")
CONTROL_ARMS = ("anchors", "anchors_fine")
ARMS = TRAIN_ARMS + CONTROL_ARMS
ARM_SHIFTS = {"anchors_shift1": 1, "anchors_shift2": 2, "anchors": 4, "anchors_fine": 0}
SEEDS = (0, 1, 2)
WIDTH = 512
TRAIN_STEPS = 1024
EVAL_STEPS = (0, 256, 1024)
VALIDATION_INDICES = tuple(range(40, 56))
WORKER_TIMEOUT_S = 900
CHECKPOINT_TIMEOUT_S = 240
PLAN = REPO / "manuscript/research/PLAN.md"
matched_baselines = screen.matched_baselines
gate_arm = screen.gate_arm
_run_process = budget._run_process


def source_identity():
    result = calibration.source_identity()
    path = Path(__file__).resolve()
    result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def recipe():
    value = budget.recipe()
    value.update(arms=list(ARMS), training_arms=list(TRAIN_ARMS), control_arms=list(CONTROL_ARMS),
                 arm_shifts=ARM_SHIFTS.copy(), seeds=list(SEEDS), steps=TRAIN_STEPS,
                 training_indices=list(range(TRAIN_STEPS)), validation_steps=list(EVAL_STEPS),
                 validation_indices=list(VALIDATION_INDICES), worker_timeout_s=WORKER_TIMEOUT_S,
                 control_checkpoint_timeout_s=CHECKPOINT_TIMEOUT_S,
                 baseline_budgets={str(step): step * 128 for step in EVAL_STEPS if step},
                 initialization_policy="new training starts from anchored initial weights, byte-matched to completed controls",
                 checkpoint_256_policy="different training shifts are not expected to reproduce trained control weights",
                 control_policy="immutable prior0/256/1024 weights, no retraining, fresh CSIM process per checkpoint",
                 training_budget_fields="steps/training_pairs describe checkpoint training history; "
                                        "training_updates_performed distinguishes1024 new updates from0 control updates",
                 validation_cohort="new fixed development windows40..55; original control cohort scores are not substituted",
                 scope="six new training cells plus six immutable control cells; all twelve must finish before unchanged gates; "
                       "no cloud, final test, adaptive extension, geometry change or hardware/speed claim")
    value["head_initialization"] = {arm: "same original distinct anchored head; training and evaluation zshift=%d" % ARM_SHIFTS[arm]
                                    for arm in ARMS}
    return value


def initialize_buffer(shape, seed, arm):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("unknown declared scale arm or seed")
    initial = screen.initialize_buffer(shape, seed, "anchors")
    initial[shape.OFF_CFG + 1] = ARM_SHIFTS[arm]
    return initial


def _read(path):
    return pilot._finite_json(Path(path).read_text())


def _guard_output(output, budget_run, allow_existing=False):
    """Reject writes inside either sealed historical run before creating anything."""
    target = Path(output).resolve()
    if target.exists() and not allow_existing:
        raise FileExistsError("output must be a new exclusive directory")
    historical = Path(budget_run).resolve(strict=True)
    manifest = _read(historical / "manifest.json")
    earlier = Path(manifest["v1_evidence"]["run_directory"]).resolve(strict=True)
    if any(target == root or root in target.parents for root in (historical, earlier)):
        raise ValueError("output cannot be nested inside either immutable historical run")
    return target


def _artifact_manifest(directory):
    result = screen._artifact_manifest(Path(directory))
    result["schema"] = SCHEMA
    return result


def budget_binding(run, audit, dataset):
    """Reuse the frozen completed-audit validator, never its scoring/grid execution."""
    context = calibration.validate_inputs(run, audit)
    if ({key: value for key, value in context["dataset"].items() if key != "path"}
            != {key: value for key, value in dataset.items() if key != "path"}):
        raise ValueError("new study corpus/splits differ from completed budget controls")
    workers = {}
    for arm in CONTROL_ARMS:
        for worker in context["by_arm"][arm]:
            name = "%s_seed_%d" % (arm, worker["seed"])
            directory = context["root"] / name
            workers[name] = {
                "manifest_sha256": pilot.digest((directory / "manifest.json").read_bytes()),
                "artifact_manifest_sha256": pilot.digest((directory / "artifact_manifest.json").read_bytes()),
                "checkpoints": {str(step): worker[key]["checkpoint"]
                                for step, key in zip(EVAL_STEPS, ("initial", "intermediate", "final"))}}
    return {"run_directory": str(context["root"]),
            "audit_record": {"file": str(context["audit_path"]), "sha256": pilot.digest(context["audit_bytes"])},
            "manifest_sha256": pilot.digest(context["manifest_bytes"]),
            "artifact_manifest_sha256": pilot.digest(context["artifact_bytes"]), "workers": workers}


def read_budget_checkpoint(binding, arm, seed, step):
    if (arm not in ARMS or type(seed) is not int or seed not in SEEDS
            or type(step) is not int or step not in EVAL_STEPS or (arm in TRAIN_ARMS and step != 0)):
        raise ValueError("only initial new-arm weights or declared control checkpoints can bind to history")
    original_arm = "anchors" if arm in TRAIN_ARMS else arm
    name = "%s_seed_%d" % (original_arm, seed)
    record = binding["workers"][name]["checkpoints"][str(step)]
    filename = "checkpoint_%06d.i32" % step
    root = Path(binding["run_directory"]).resolve(strict=True)
    path = root / name / filename
    if path.is_symlink() or (root / name).is_symlink() or record["file"] != filename:
        raise ValueError("control checkpoint path must be the exact regular bound file")
    shape = pilot.layout(WIDTH)
    regions = [{"name": region, "words": getattr(shape, "WORDS_" + region)} for region in pilot.WEIGHTS]
    payload = path.read_bytes()
    if (record["step"] != step or record["regions"] != regions or record["location_portable"] is not True
            or len(payload) != sum(row["words"] for row in regions) * 4 or record["bytes"] != len(payload)
            or record["sha256"] != pilot.digest(payload)):
        raise ValueError("actual historical full checkpoint differs from binding")
    return payload


def _common_manifest(corpus, budget_run, budget_audit):
    shape = pilot.layout(WIDTH)
    raw, dataset = pilot.corpus_plan(corpus, TRAIN_STEPS, max(VALIDATION_INDICES) + 1, shape.ML_L_SEQ)
    return raw, shape, {"schema": SCHEMA, "backend": "csim", "recipe": recipe(),
                        "dataset": dataset, "layout": pilot.layout_values(shape),
                        "source_hashes": source_identity(),
                        "budget_evidence": budget_binding(budget_run, budget_audit, dataset),
                        "plan_document": {"file": PLAN.relative_to(REPO).as_posix(),
                                          "sha256": pilot.digest(PLAN.read_bytes())}}


def _assert_frozen(manifest, library, library_hash):
    if source_identity() != manifest["source_hashes"] or pilot.digest(Path(library).read_bytes()) != library_hash:
        raise RuntimeError("scale study sources/library changed")
    bound = manifest["budget_evidence"]
    if budget_binding(bound["run_directory"], bound["audit_record"]["file"], manifest["dataset"]) != bound:
        raise RuntimeError("immutable completed control evidence changed")


def _weights_from_payload(payload, shape):
    import struct
    count = sum(getattr(shape, "WORDS_" + name) for name in pilot.WEIGHTS)
    if len(payload) != count * 4:
        raise ValueError("unsupported full checkpoint byte count")
    values = list(struct.unpack("<%di" % count, payload))
    weights, start = {}, 0
    for name in pilot.WEIGHTS:
        stop = start + getattr(shape, "WORDS_" + name)
        weights[name], start = values[start:stop], stop
    return weights


def _evaluate(output, bridge, current, shape, raw, manifest, step, observer, controls):
    snapshot = pilot.checkpoint(output, step, current, shape)
    rows, zshift = [], manifest["zshift"]
    for index in VALIDATION_INDICES:
        tokens, targets = pilot.window(raw, manifest["dataset"], "validation", index, shape.ML_L_SEQ)
        observed = observer(current, tokens, targets, shape, zshift=zshift)
        reference = pilot.integer_forward(current, tokens, targets, shape, zshift=zshift)
        if (any(observed[key] != reference[key] for key in reference if key != "lut_normalized_nll")
                or not math.isclose(observed["lut_normalized_nll"], reference["lut_normalized_nll"], rel_tol=0, abs_tol=1e-12)):
            raise RuntimeError("observer differs from frozen independent forward")
        pilot.stage_window(bridge, shape, tokens, targets)
        bridge.step(0, 2048)
        prefix = "eval_%06d_window_%03d" % (step, index)
        arrays = {}
        for name, offset, count in (("predictions", shape.OFF_PRED, shape.ML_L_SEQ),
                                    ("gradients", shape.OFF_GRAD_LOGIT, shape.WORDS_GRAD_LOGIT)):
            values = bridge.read(offset, count)
            arrays[name] = screen._save_array(output, prefix + "_" + name + ".i32", values)
            if values != reference[name]:
                raise RuntimeError("native " + name + " differ from independent forward")
        route_hashes = {}
        for name, expected in observed["native_routes"].items():
            count = getattr(shape, "WORDS_" + name)
            values = bridge.read(getattr(shape, "OFF_" + name), count)
            if len(expected) != count or values != expected:
                raise RuntimeError("native route mismatch: " + name)
            route_hashes[name] = {"words": count, "sha256": pilot.digest(screen._i32(values))}
        probes = {}
        for name, probe_tokens, context in (
                ("context_disabled", tokens, False), ("all_space", [32] * len(tokens), True),
                ("reversed_tokens", list(reversed(tokens)), True)):
            candidate = pilot.integer_forward(current, probe_tokens, targets, shape, zshift=zshift, use_context=context)
            probes[name] = screen._input_probe(reference, candidate)
        paired = ({name: controls[name]["per_window"][len(rows)]["nll"] - reference["lut_normalized_nll"]
                   for name in ("unigram", "bigram")} if controls is not None else {})
        row = {"window_index": index, **screen._scalar_result(observed), "probes": probes,
               "native_arrays": arrays, "native_route_hashes": route_hashes,
               "native_reference_equal": True, "native_routes_equal": True, "baseline_minus_model_nll": paired}
        pilot.write_json(output / (prefix + ".json"), row)
        rows.append(row)
    payload = pilot.weight_bytes(pilot.read_weights(bridge, shape))
    if payload != pilot.weight_bytes(current):
        raise RuntimeError("frozen evaluation changed full weights")
    result = {"step": step, "checkpoint": snapshot, "weights_unchanged": True,
              "baseline_training_pairs": step * shape.ML_L_SEQ if step else None, **screen._aggregate(rows)}
    if manifest["arm"] in CONTROL_ARMS:
        if payload != read_budget_checkpoint(manifest["budget_evidence"], manifest["arm"], manifest["seed"], step):
            raise RuntimeError("control evaluation changed actual historical full checkpoint")
        result["historical_checkpoint_equal"] = True
    pilot.write_json(output / ("evaluation_%06d.json" % step), result)
    return result


def _stage_initial(bridge, initial):
    bridge.write(0, initial)
    if bridge.read(0, len(initial)) != initial:
        raise RuntimeError("complete initialization transport mismatch")


def _load_observer(observer):
    if observer is None:
        from manuscript.tools.observe_fpga_lm_routes import observe
        return observe
    return observer


def _run_checkpoint_process(command, log_path):
    """Checkpoint child inherits the outer worker's owned group, so outer timeout kills both."""
    with Path(log_path).open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(REPO), start_new_session=False)
        try:
            return {"returncode": process.wait(timeout=CHECKPOINT_TIMEOUT_S), "timed_out": False}
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            return {"returncode": process.returncode, "timed_out": True}
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            raise


def run_checkpoint_worker(config, bridge_factory=None, observer=None):
    """One fresh process, one immutable control checkpoint, lr0 only."""
    output = Path(config.output).resolve(strict=True)
    step = config.checkpoint_step
    if (config.arm not in CONTROL_ARMS or type(config.seed) is not int or config.seed not in SEEDS
            or type(step) is not int or step not in EVAL_STEPS
            or Path(config.parent_manifest).resolve(strict=True) != output / "manifest.json"):
        raise ValueError("checkpoint child requires an owned control manifest and declared checkpoint")
    _guard_output(output, config.budget_run, allow_existing=True)
    authorized = _read(output / "manifest.json")
    if (authorized.get("schema") != SCHEMA or authorized.get("arm") != config.arm
            or authorized.get("seed") != config.seed or authorized.get("training_performed") is not False):
        raise ValueError("checkpoint child output is not an owned scale-study control directory")
    job = output / "checkpoint_jobs" / ("step_%06d" % step)
    job.mkdir(parents=True, exist_ok=False)
    bridge = None
    try:
        manifest = _read(output / "manifest.json")
        raw, shape, common = _common_manifest(config.corpus, config.budget_run, config.budget_audit)
        if (any(manifest[key] != common[key] for key in common)
                or manifest["arm"] != config.arm or manifest["seed"] != config.seed
                or manifest["training_performed"] is not False or manifest["training_updates_performed"] != 0
                or manifest["zshift"] != ARM_SHIFTS[config.arm]):
            raise ValueError("checkpoint child differs from authorized control recipe")
        library = Path(config.library).resolve(strict=True)
        library_hash = pilot.digest(library.read_bytes())
        if manifest["library_path"] != str(library) or manifest["library_sha256"] != library_hash:
            raise ValueError("checkpoint child library differs from its control manifest")
        pilot.write_json(job / "manifest.json", {"schema": SCHEMA, "arm": config.arm, "seed": config.seed,
                         "step": step, "parent_manifest_sha256": pilot.digest((output / "manifest.json").read_bytes()),
                         "source_hashes": common["source_hashes"], "library_sha256": library_hash,
                         "process_id": os.getpid(), "native_training_updates": 0})
        payload = read_budget_checkpoint(manifest["budget_evidence"], config.arm, config.seed, step)
        current = _weights_from_payload(payload, shape)
        initial = initialize_buffer(shape, config.seed, config.arm)
        for name, values in current.items():
            offset = getattr(shape, "OFF_" + name)
            initial[offset:offset + len(values)] = values
        bridge = (bridge_factory or pilot.NativeBridge)(library, "csim", shape)
        _stage_initial(bridge, initial)
        controls = None
        if step:
            controls = matched_baselines(raw, manifest["dataset"], range(step), VALIDATION_INDICES, shape.ML_L_SEQ)
            pilot.write_json(output / ("baselines_%06d.json" % step), controls)
        evaluation = _evaluate(output, bridge, current, shape, raw, manifest, step, _load_observer(observer), controls)
        to_close, bridge = bridge, None
        to_close.close()
        _assert_frozen(manifest, library, library_hash)
        receipt = {"schema": SCHEMA, "complete": True, "arm": config.arm, "seed": config.seed,
                   "step": step, "training_updates_performed": 0, "native_learning_rate": 0,
                   "native_validation_windows": len(VALIDATION_INDICES),
                   "historical_checkpoint_sha256": pilot.digest(payload),
                   "evaluation_sha256": pilot.digest((output / ("evaluation_%06d.json" % step)).read_bytes()),
                   "checkpoint_sha256": evaluation["checkpoint"]["sha256"]}
        pilot.write_json(job / "summary.json", receipt)
        return receipt
    except BaseException as error:
        pilot.write_json(job / "failure.json", {"complete": False, "error": str(error), "type": type(error).__name__})
        raise
    finally:
        if bridge is not None:
            try:
                bridge.close()
            except Exception as error:
                pilot.write_json(job / "cleanup_failure.json", {"error": str(error), "type": type(error).__name__})


def _control_evaluations(config, manifest, runner):
    output = Path(config.output).resolve()
    evaluations, controls = {}, {}
    for step in EVAL_STEPS:
        command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint-worker", "--checkpoint-step", str(step),
                   "--arm", config.arm, "--seed", str(config.seed), "--corpus", str(Path(config.corpus).resolve()),
                   "--budget-run", str(Path(config.budget_run).resolve()),
                   "--budget-audit", str(Path(config.budget_audit).resolve()),
                   "--output", str(output), "--library", manifest["library_path"],
                   "--parent-manifest", str(output / "manifest.json")]
        execution = {"command": command, "timeout_s": CHECKPOINT_TIMEOUT_S,
                     "inherits_outer_worker_process_group": True,
                     **runner(command, output / ("checkpoint_%06d.log" % step))}
        pilot.write_json(output / ("checkpoint_%06d_execution.json" % step), execution)
        if execution["returncode"] != 0 or execution["timed_out"]:
            raise RuntimeError("fresh control checkpoint process failed or timed out")
        receipt = _read(output / "checkpoint_jobs" / ("step_%06d" % step) / "summary.json")
        if receipt.get("complete") is not True or receipt.get("training_updates_performed") != 0:
            raise ValueError("control checkpoint process did not complete read-only evaluation")
        evaluations[step] = _read(output / ("evaluation_%06d.json" % step))
        if step:
            controls[str(step)] = _read(output / ("baselines_%06d.json" % step))
    return evaluations, controls


def run_worker(config, bridge_factory=None, observer=None, checkpoint_runner=None):
    output = _guard_output(config.output, config.budget_run)
    output.mkdir(parents=True, exist_ok=False)
    bridge, completed = None, 0
    try:
        import numpy as np
        if config.arm not in ARMS or type(config.seed) is not int or config.seed not in SEEDS:
            raise ValueError("unknown declared scale arm or seed")
        raw, shape, manifest = _common_manifest(config.corpus, config.budget_run, config.budget_audit)
        initial = initialize_buffer(shape, config.seed, config.arm)
        weights = {name: initial[getattr(shape, "OFF_" + name):
                                getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
                   for name in pilot.WEIGHTS}
        if pilot.weight_bytes(weights) != read_budget_checkpoint(manifest["budget_evidence"], config.arm, config.seed, 0):
            raise ValueError("actual anchored initial weights differ from immutable control initialization")
        library = Path(config.library).resolve(strict=True)
        library_hash = pilot.digest(library.read_bytes())
        parent_manifest = getattr(config, "parent_manifest", None)
        if parent_manifest:
            expected = _read(parent_manifest)
            if any(manifest[key] != expected[key] for key in manifest):
                raise ValueError("worker source/protocol/corpus/control history differs from parent")
            binding = _read(Path(parent_manifest).parent / "library.json")
            if binding["path"] != str(library) or binding["sha256"] != library_hash:
                raise ValueError("worker library differs from parent build")
        training = config.arm in TRAIN_ARMS
        manifest.update(arm=config.arm, seed=config.seed, zshift=ARM_SHIFTS[config.arm],
                        training_performed=training, training_updates_performed=TRAIN_STEPS if training else 0,
                        library_path=str(library), library_sha256=library_hash,
                        initial_buffer_sha256=pilot.digest(screen._i32(initial)),
                        started_utc=datetime.now(timezone.utc).isoformat(),
                        environment={"python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
        pilot.write_json(output / "manifest.json", manifest)
        if training:
            observer = _load_observer(observer)
            bridge = (bridge_factory or pilot.NativeBridge)(library, "csim", shape)
            _stage_initial(bridge, initial)
            evaluations = {0: _evaluate(output, bridge, weights, shape, raw, manifest, 0, observer, None)}
            controls = {}
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
                    log.write(json.dumps({"step": completed, "window_index": index, "token_pairs": shape.ML_L_SEQ,
                                          "input_readback_equal": True, "predictions_sha256": pilot.digest(screen._i32(predictions)),
                                          "gradients_sha256": pilot.digest(screen._i32(gradients))}, allow_nan=False) + "\n")
                    log.flush()
                    if completed in EVAL_STEPS:
                        fitted = matched_baselines(raw, manifest["dataset"], range(completed), VALIDATION_INDICES, shape.ML_L_SEQ)
                        controls[str(completed)] = fitted
                        pilot.write_json(output / ("baselines_%06d.json" % completed), fitted)
                        evaluations[completed] = _evaluate(output, bridge, pilot.read_weights(bridge, shape), shape,
                                                          raw, manifest, completed, observer, fitted)
            to_close, bridge = bridge, None
            to_close.close()
        else:
            evaluations, controls = _control_evaluations(config, manifest, checkpoint_runner or _run_checkpoint_process)
        _assert_frozen(manifest, library, library_hash)
        pilot.write_json(output / "baselines.json", controls[str(TRAIN_STEPS)])
        summary = {"schema": SCHEMA, "complete": True, "arm": config.arm, "seed": config.seed,
                   "steps": TRAIN_STEPS, "training_pairs": TRAIN_STEPS * shape.ML_L_SEQ,
                   "training_performed": training, "training_updates_performed": completed,
                   "initial_weights_equal_budget_control": True,
                   "initial": evaluations[0], "intermediate": evaluations[EVAL_STEPS[1]],
                   "final": evaluations[TRAIN_STEPS], "baselines_by_step": controls,
                   "baselines": controls[str(TRAIN_STEPS)], "heldout_test_evaluated": False,
                   "hardware_execution": False, "speed_advantage_claim": False}
        pilot.write_json(output / "summary.json", summary)
        pilot.write_json(output / "artifact_manifest.json", _artifact_manifest(output))
        return summary
    except BaseException as error:
        pilot.write_json(output / "failure.json", {"complete": False, "training_updates_completed": completed,
                                                   "error": str(error), "type": type(error).__name__})
        raise
    finally:
        if bridge is not None:
            try:
                bridge.close()
            except Exception as error:
                pilot.write_json(output / "cleanup_failure.json", {"error": str(error), "type": type(error).__name__})


def _verify_evaluation(directory, evaluation, step, controls, shape):
    if (_read(directory / ("evaluation_%06d.json" % step)) != evaluation
            or [row["window_index"] for row in evaluation["per_window"]] != list(VALIDATION_INDICES)
            or evaluation["baseline_training_pairs"] != (step * shape.ML_L_SEQ if step else None)):
        raise ValueError("evaluation link/cohort/baseline budget differs")
    for name, value in screen._aggregate(evaluation["per_window"]).items():
        if evaluation[name] != value:
            raise ValueError("evaluation aggregate differs from per-window evidence")
    for index, row in enumerate(evaluation["per_window"]):
        prefix = "eval_%06d_window_%03d" % (step, row["window_index"])
        if (_read(directory / (prefix + ".json")) != row or row["native_reference_equal"] is not True
                or row["native_routes_equal"] is not True):
            raise ValueError("native-checked validation row differs")
        differences = ({name: controls[name]["per_window"][index]["nll"] - row["lut_normalized_nll"]
                        for name in ("unigram", "bigram")} if controls is not None else {})
        if row["baseline_minus_model_nll"] != differences:
            raise ValueError("baseline differences do not match exact checkpoint-specific controls")
        for name, count in (("predictions", shape.ML_L_SEQ), ("gradients", shape.WORDS_GRAD_LOGIT)):
            record = row["native_arrays"][name]
            filename = prefix + "_" + name + ".i32"
            payload = (directory / filename).read_bytes()
            if (record["file"] != filename or record["words"] != count or record["bytes"] != count * 4
                    or len(payload) != count * 4 or record["sha256"] != pilot.digest(payload)):
                raise ValueError("native array payload/coverage/hash mismatch")


def _verify_control_job(directory, manifest, evaluation, step):
    job = directory / "checkpoint_jobs" / ("step_%06d" % step)
    if any((job / name).exists() for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("failed checkpoint child cannot count as a valid control")
    job_manifest = _read(job / "manifest.json")
    required = {"schema": SCHEMA, "arm": manifest["arm"], "seed": manifest["seed"], "step": step,
                "parent_manifest_sha256": pilot.digest((directory / "manifest.json").read_bytes()),
                "source_hashes": manifest["source_hashes"], "library_sha256": manifest["library_sha256"],
                "native_training_updates": 0}
    if (any(job_manifest.get(name) != value for name, value in required.items())
            or any(type(job_manifest.get(name)) is not int for name in ("seed", "step", "native_training_updates"))
            or type(job_manifest.get("process_id")) is not int or job_manifest["process_id"] <= 0):
        raise ValueError("fresh checkpoint process provenance differs")
    receipt = _read(job / "summary.json")
    expected = {"schema": SCHEMA, "complete": True, "arm": manifest["arm"], "seed": manifest["seed"],
                "step": step, "training_updates_performed": 0, "native_learning_rate": 0,
                "native_validation_windows": len(VALIDATION_INDICES),
                "historical_checkpoint_sha256": evaluation["checkpoint"]["sha256"],
                "checkpoint_sha256": evaluation["checkpoint"]["sha256"],
                "evaluation_sha256": pilot.digest((directory / ("evaluation_%06d.json" % step)).read_bytes())}
    if (receipt != expected or receipt.get("complete") is not True
            or any(type(receipt.get(name)) is not int for name in
                   ("seed", "step", "training_updates_performed", "native_learning_rate", "native_validation_windows"))):
        raise ValueError("read-only checkpoint child receipt differs")
    execution = _read(directory / ("checkpoint_%06d_execution.json" % step))
    if (type(execution.get("returncode")) is not int or execution["returncode"] != 0
            or type(execution.get("timeout_s")) is not int
            or execution.get("timed_out") is not False or execution.get("timeout_s") != CHECKPOINT_TIMEOUT_S
            or execution.get("inherits_outer_worker_process_group") is not True):
        raise ValueError("control checkpoint lacks bounded inherited-group successful execution")


def verify_worker(directory, expected, arm, seed, library_hash):
    directory = Path(directory)
    budget._verify_artifacts(directory, SCHEMA)
    manifest, result = _read(directory / "manifest.json"), _read(directory / "summary.json")
    training = arm in TRAIN_ARMS
    updates = TRAIN_STEPS if training else 0
    if any(manifest[key] != expected[key] for key in expected):
        raise ValueError("worker protocol/source/corpus/control binding differs from parent")
    for record in (manifest, result):
        if (record.get("schema") != SCHEMA or record.get("arm") != arm or record.get("seed") != seed
                or record.get("training_performed") is not training
                or type(record.get("training_updates_performed")) is not int
                or record["training_updates_performed"] != updates):
            raise ValueError("worker training/control role or actual-update count differs")
    if (manifest.get("library_sha256") != library_hash or manifest.get("zshift") != ARM_SHIFTS[arm]
            or result.get("complete") is not True or result.get("steps") != TRAIN_STEPS
            or result.get("training_pairs") != TRAIN_STEPS * 128
            or result.get("initial_weights_equal_budget_control") is not True
            or any(result.get(name) is not False for name in ("heldout_test_evaluated", "hardware_execution", "speed_advantage_claim"))):
        raise ValueError("worker recipe/library or complete checkpoint training budget differs")
    shape = pilot.layout(WIDTH)
    initial = initialize_buffer(shape, seed, arm)
    initial_weights = {name: initial[getattr(shape, "OFF_" + name):
                                    getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
                       for name in pilot.WEIGHTS}
    if manifest["initial_buffer_sha256"] != pilot.digest(screen._i32(initial)):
        raise ValueError("initial buffer differs from declared head and LUT configuration")
    raw = Path(expected["dataset"]["path"]).read_bytes()
    if pilot.digest(raw) != expected["dataset"]["sha256"]:
        raise ValueError("study corpus changed")
    if (set(result["baselines_by_step"]) != {str(step) for step in EVAL_STEPS if step}
            or result["baselines"] != result["baselines_by_step"][str(TRAIN_STEPS)]
            or result["baselines"] != _read(directory / "baselines.json")):
        raise ValueError("checkpoint-specific controls or final baseline link differ")
    for step, key in zip(EVAL_STEPS, ("initial", "intermediate", "final")):
        evaluation = result[key]
        payload = budget._verify_checkpoint(directory, evaluation, step, shape)
        if step == 0 and (payload != pilot.weight_bytes(initial_weights)
                          or payload != read_budget_checkpoint(expected["budget_evidence"], arm, seed, step)):
            raise ValueError("actual initial full checkpoint differs from the paired anchored control")
        if not training:
            if (payload != read_budget_checkpoint(expected["budget_evidence"], arm, seed, step)
                    or evaluation.get("historical_checkpoint_equal") is not True):
                raise ValueError("re-evaluated control is not the actual unchanged historical checkpoint")
            _verify_control_job(directory, manifest, evaluation, step)
        controls = None
        if step:
            controls = matched_baselines(raw, expected["dataset"], range(step), VALIDATION_INDICES, shape.ML_L_SEQ)
            if (result["baselines_by_step"][str(step)] != controls
                    or _read(directory / ("baselines_%06d.json" % step)) != controls):
                raise ValueError("controls do not match exact supervised pairs and the new cohort")
        _verify_evaluation(directory, evaluation, step, controls, shape)
    training_files = ("training.jsonl", "training_predictions.i32", "training_gradients.i32")
    if not training:
        if any((directory / name).exists() for name in training_files):
            raise ValueError("read-only controls cannot publish a new training stream")
    else:
        rows = [pilot._finite_json(line) for line in (directory / "training.jsonl").read_text().splitlines()]
        if ([row["window_index"] for row in rows] != list(range(TRAIN_STEPS))
                or [row["step"] for row in rows] != list(range(1, TRAIN_STEPS + 1))
                or any(row.get("token_pairs") != shape.ML_L_SEQ or row.get("input_readback_equal") is not True for row in rows)):
            raise ValueError("fresh training sequence or input-readback coverage differs")
        for name, count in (("predictions", shape.ML_L_SEQ), ("gradients", shape.WORDS_GRAD_LOGIT)):
            payload = (directory / ("training_" + name + ".i32")).read_bytes()
            stride = count * 4
            if len(payload) != TRAIN_STEPS * stride or any(
                    pilot.digest(payload[index * stride:(index + 1) * stride]) != row[name + "_sha256"]
                    for index, row in enumerate(rows)):
                raise ValueError("training native array byte coverage/hash mismatch")
    return result


def _compact_worker(result):
    value = screen._compact_worker(result)
    value["intermediate"] = screen._compact_worker(dict(result, initial=result["intermediate"]))["initial"]
    value["baselines_by_step"] = {step: {name: controls[name] for name in ("unigram", "bigram")}
                                    for step, controls in result["baselines_by_step"].items()}
    value.update(training_performed=result["training_performed"],
                 training_updates_performed=result["training_updates_performed"],
                 initial_weights_equal_budget_control=result["initial_weights_equal_budget_control"])
    return value


def _verify_paired_initializations(root, binding):
    root = Path(root)
    for seed in SEEDS:
        expected = read_budget_checkpoint(binding, "anchors", seed, 0)
        for arm in ARMS:
            if (root / ("%s_seed_%d" % (arm, seed)) / "checkpoint_000000.i32").read_bytes() != expected:
                raise ValueError("all four scale arms must share the actual full anchored initial weights")
    return {"all_arm_full_weights_equal_for_every_paired_seed": True,
            "initial_weights_equal_immutable_budget_controls": True}


def run_study(config):
    output = _guard_output(config.output, config.budget_run)
    output.mkdir(parents=True, exist_ok=False)
    try:
        raw, shape, manifest = _common_manifest(config.corpus, config.budget_run, config.budget_audit)
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
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm, "--seed", str(seed),
                           "--corpus", str(Path(config.corpus).resolve()), "--budget-run", str(Path(config.budget_run).resolve()),
                           "--budget-audit", str(Path(config.budget_audit).resolve()), "--output", str(output / name),
                           "--library", str(library), "--parent-manifest", str(output / "manifest.json")]
                execution = {"command": command, "timeout_s": WORKER_TIMEOUT_S,
                             **_run_process(command, output / (name + ".log"))}
                pilot.write_json(output / (name + "_execution.json"), execution)
                try:
                    _assert_frozen(manifest, library, library_hash)
                    if execution["returncode"] != 0 or execution["timed_out"]:
                        raise RuntimeError("bounded worker process failed or timed out")
                    results[arm].append(verify_worker(output / name, manifest, arm, seed, library_hash))
                except Exception as error:
                    failure = {"arm": arm, "seed": seed, "error": str(error), "type": type(error).__name__}
                    pilot.write_json(output / (name + "_failure.json"), failure)
                    worker_output = output / name
                    worker_output.mkdir(exist_ok=True)
                    if not (worker_output / "failure.json").exists():
                        pilot.write_json(worker_output / "failure.json", {"complete": False, "reported_by": "parent", **failure})
                    failures.append(failure)
        _assert_frozen(manifest, library, library_hash)
        pairing = _verify_paired_initializations(output, manifest["budget_evidence"]) if not failures else {"complete": False}
        gates = {arm: gate_arm(rows) if not failures else {
                 "complete": False, "larger_paid_learning_benchmark_gate": False,
                 "reason": "all twelve new-training and immutable-control cells must complete"}
                 for arm, rows in results.items()}
        summary = {"schema": SCHEMA, "complete": not failures, "failures": failures,
                   "gates": gates, "declared_workers": len(ARMS) * len(SEEDS),
                   "completed_workers": sum(map(len, results.values())),
                   "declared_training_workers": len(TRAIN_ARMS) * len(SEEDS),
                   "declared_control_workers": len(CONTROL_ARMS) * len(SEEDS),
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
            pilot.write_json(output / "failure.json", {"complete": False, "error": str(error), "type": type(error).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--budget-run", required=True, type=Path)
    parser.add_argument("--budget-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-step", type=int, choices=EVAL_STEPS, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--library", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--parent-manifest", type=Path, help=argparse.SUPPRESS)
    config = parser.parse_args(argv)
    if config.worker and config.checkpoint_worker:
        parser.error("top-level worker and checkpoint-child modes are mutually exclusive")
    if config.worker or config.checkpoint_worker:
        if config.arm is None or config.seed is None or config.library is None:
            parser.error("internal workers require --arm, --seed, --library")
        if config.checkpoint_worker:
            if config.checkpoint_step is None or config.parent_manifest is None or config.arm not in CONTROL_ARMS:
                parser.error("checkpoint child requires a control arm, checkpoint step and owned parent manifest")
            result = run_checkpoint_worker(config)
        else:
            if config.checkpoint_step is not None:
                parser.error("checkpoint-step is valid only for a checkpoint child")
            result = run_worker(config)
    else:
        if any(value is not None for value in (config.arm, config.seed, config.library, config.parent_manifest, config.checkpoint_step)):
            parser.error("internal overrides require an explicit worker mode")
        result = run_study(config)
    print(json.dumps({"complete": result["complete"], "output": str(config.output)}, allow_nan=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
