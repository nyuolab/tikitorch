"""Four-cell, CPU-only integer error-feedback learning diagnostic.

This is a new optimizer and state ABI, NOT executable on the existing FPGA
image. One seed and two declared scales cannot pass the three-seed paid gate.
Host-fitted initializers are charged a separate full count-fitting pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import audit_fpga_lm_head_screen as audit  # noqa: E402
from manuscript.tools import check_fpga_lm_bigram_capacity as capacity  # noqa: E402
from manuscript.tools import fpga_lm_efsgd as optimizer  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from manuscript.tools.observe_fpga_lm_routes import observe  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_efsgd_study.v1"
ARMS = ("anchors", "count_fitted")
K_VALUES = (18, 20)
WIDTH, ZSHIFT, STEPS = 512, 2, 1024
CHECKPOINTS = (0, 256, 1024)
VALIDATION = tuple(range(40, 56))
WORKER_TIMEOUT_S = 1200
PLAN = REPO / "manuscript/research/PLAN.md"


def source_identity():
    result = capacity.source_identity()
    result.update(optimizer.source_identity())
    path = Path(__file__).resolve()
    result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def recipe():
    return {"arms": list(ARMS), "k_values": list(K_VALUES), "seed": 0,
            "width": WIDTH, "zshift": ZSHIFT, "training_indices": list(range(STEPS)),
            "checkpoints": list(CHECKPOINTS), "validation_indices": list(VALIDATION),
            "optimizer": "a=r+G; step=trunc_toward_zero(a/2**k); w-=step; r=a-step*2**k",
            "overflow_policy": "nonrecovering UBSan; reject out-of-int16 updates; no successful poisoned state",
            "scale_rationale": "one target occurrence has about32768..65536 gradient units: "
                               "fractional weight motion0.125..0.25 at k18 or0.03125..0.0625 at k20",
            "initialization": {"anchors": "original anchored seed0, zshift2, no data fitting",
                               "count_fitted": "exact frozen bigram construction, alpha1 on windows0..1023"},
            "training_order": "each arm uses windows0..1023 once in order; count_fitted replays its fitting data",
            "budget": "anchors131072 gradient pairs; count_fitted adds131072 host-fit pairs; "
                      "both have131072 distinct supervised pairs at final",
            "baselines": "alpha1 unigram/bigram fitted on every actual exposure (including prefix replay); "
                         "count_fitted also reports one-pass fit and uses the lower NLL comparator",
            "evaluation": "same development cohort40..55; not fresh holdout; no test or validation-fitted parameters",
            "native_state": "weights, all integer gradient remainders, step count and optimizer/loss configuration",
            "checkpoint_verification": "full state save/import; forward-only evaluation must preserve every state byte; "
                                       "native predictions, loss gradients and routes compared to frozen observer",
            "worker_timeout_s": WORKER_TIMEOUT_S,
            "scope": "four exploratory cells, all retained; no three-seed gate, hardware or speed claim"}


def _read(path):
    return pilot._finite_json(Path(path).read_text())


def _artifacts(root):
    result = screen._artifact_manifest(root)
    result["schema"] = SCHEMA
    result["files"] = [row for row in result["files"] if row["file"] != "artifact_manifest.json"]
    return result


def _seal(root):
    pilot.write_json(root / "artifact_manifest.json", _artifacts(root))


def _run_process(command, path):
    started = time.monotonic()
    with path.open("xb") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            raise
        finally:
            # A worker can exit abruptly before its native child's context
            # manager runs. Its descendants still belong to our owned group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return {"returncode": process.returncode, "timed_out": timed_out,
            "timeout_s": WORKER_TIMEOUT_S, "elapsed_s": time.monotonic() - started}


def exposure_baseline(raw, dataset, arm, step):
    """Alpha-one controls count actual data accesses, including prefix replay."""
    if arm not in ARMS or type(step) is not int or step not in CHECKPOINTS:
        raise ValueError("undeclared budget")
    indices = (list(range(STEPS)) if arm == "count_fitted" else []) + list(range(step))
    if not indices:
        return None
    unigram, bigram = [1] * 128, [[1] * 128 for _ in range(128)]
    for index in indices:
        tokens, targets = pilot.window(raw, dataset, "train", index)
        for token, target in zip(tokens, targets):
            unigram[target] += 1
            bigram[token][target] += 1
    result = {"supervised_exposures": len(indices) * 128,
              "distinct_supervised_pairs": len(set(indices)) * 128, "alpha": 1}
    for name, counts in (("unigram", [unigram] * 128), ("bigram", bigram)):
        totals = list(map(sum, counts))
        predictions = [max(range(128), key=lambda target: row[target]) for row in counts]
        losses, correct = [], 0
        for index in VALIDATION:
            tokens, targets = pilot.window(raw, dataset, "validation", index)
            losses.extend(math.log(totals[token]) - math.log(counts[token][target])
                          for token, target in zip(tokens, targets))
            correct += sum(predictions[token] == target for token, target in zip(tokens, targets))
        result[name] = {"nll": sum(losses) / len(losses), "accuracy": correct / len(losses)}
    return result


def initial_buffer(shape, raw, dataset, arm):
    if arm not in ARMS:
        raise ValueError("undeclared initializer")
    initial = screen.initialize_buffer(shape, 0, "anchors")
    initial[shape.OFF_CFG + 1] = ZSHIFT
    if arm == "count_fitted":
        controls = screen.matched_baselines(raw, dataset, range(STEPS), VALIDATION)
        weights, _ = capacity.construct_weights(controls["bigram_pair_counts"], shape, ZSHIFT)
        for name, values in weights.items():
            offset = getattr(shape, "OFF_" + name)
            initial[offset:offset + len(values)] = values
    return initial


def _frozen(manifest):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("wrong study schema")
    audit._assert_equal(manifest.get("recipe"), recipe(), "declared frozen study recipe")
    if (manifest["source_hashes"] != source_identity()
            or pilot.digest(Path(manifest["dataset"]["path"]).read_bytes()) != manifest["dataset"]["sha256"]
            or pilot.digest(Path(manifest["executable"]["path"]).read_bytes()) != manifest["executable"]["sha256"]):
        raise ValueError("study source, corpus or executable changed")


def _native_identity(manifest, shape):
    executable = Path(manifest["executable"]["path"]).resolve(strict=True)
    build_bytes = (executable.parent / "build.json").read_bytes()
    build = pilot._finite_json(build_bytes.decode())
    sources = optimizer.source_identity()
    expected = {"schema": optimizer.SCHEMA, "source_hashes": sources,
                "source_sha256": optimizer.source_digest(sources), "layout": pilot.layout_values(shape),
                "sanitizer": "undefined, nonrecovering", "executable_sha256": manifest["executable"]["sha256"],
                "cpu_only": True, "existing_afi_compatible": False}
    audit._assert_equal({key: build.get(key) for key in expected}, expected, "native build declaration")
    command = build.get("command")
    if (not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command)
            or command[1:] != [*optimizer._flags(shape.ML_D, sources), "-I" + str(pilot.HLS),
                               str(optimizer.CPP), "-o", str(executable)]
            or (executable.parent / "failure.json").exists()):
        raise ValueError("native compile recipe differs or build failed")
    return {"schema": optimizer.SCHEMA, "executable_sha256": manifest["executable"]["sha256"],
            "build_sha256": pilot.digest(build_bytes), "source_hashes": sources,
            "layout": pilot.layout_values(shape), "cpu_only": True, "existing_afi_compatible": False}


def evaluate_cell(native, root, step, raw, dataset, shape):
    payload = native.export_state()
    state = optimizer.decode_state(payload, shape)
    if (state["step"] != step or state["k"] not in K_VALUES
            or state["config"]["loss_kind"] != 1 or state["config"]["zshift"] != ZSHIFT):
        raise ValueError("optimizer checkpoint step/config mismatch")
    checkpoint = native.save_checkpoint(root, "state_%06d" % step)
    weights = state["weights"]
    rows = []
    for index in VALIDATION:
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        actual = native.evaluate(tokens, targets)
        expected = observe(weights, tokens, targets, shape, zshift=ZSHIFT)
        arrays = {}
        for name in ("predictions", "gradients"):
            audit._assert_equal(actual[name], expected[name], "native " + name)
            arrays[name] = screen._save_array(root, "eval_%06d_%03d_%s.i32" % (step, index, name), actual[name])
        audit._assert_equal(actual["native_routes"], expected["native_routes"], "native routes")
        for name, values in actual["native_routes"].items():
            arrays[name] = screen._save_array(root, "eval_%06d_%03d_%s.i32" % (step, index, name), values)
        disabled = pilot.integer_forward(weights, tokens, targets, shape, zshift=ZSHIFT, use_context=False)
        row = {**screen._scalar_result(expected), "window_index": index,
               "native_arrays": arrays, "native_reference_equal": True, "native_routes_equal": True,
               "probes": {"context_disabled": screen._scalar_result(disabled)}}
        pilot.write_json(root / ("eval_%06d_%03d.json" % (step, index)), row)
        rows.append(row)
    if native.export_state() != payload:
        raise ValueError("evaluation changed weights or optimizer remainder/config state")
    native.import_state(payload)
    if native.export_state() != payload:
        raise ValueError("full optimizer state import differs from export")
    return {"step": step, "checkpoint": checkpoint, "state_sha256": pilot.digest(payload),
            "full_state_evaluation_invariant": True, "full_state_import_equal": True, **screen._aggregate(rows)}


def run_worker(config):
    parent = Path(config.parent).resolve(strict=True)
    root = Path(config.output).resolve()
    if config.arm not in ARMS or type(config.k) is not int or config.k not in K_VALUES:
        raise ValueError("undeclared cell")
    if root != parent / ("%s_k%d" % (config.arm, config.k)):
        raise ValueError("worker must own exactly its declared child path")
    manifest = _read(parent / "manifest.json")
    audit._assert_equal(manifest["recipe"], recipe(), "frozen recipe")
    if manifest["schema"] != SCHEMA:
        raise ValueError("wrong parent schema")
    _frozen(manifest)
    raw, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    audit._assert_equal(dataset, manifest["dataset"], "corpus")
    shape = pilot.layout(WIDTH)
    initial = initial_buffer(shape, raw, dataset, config.arm)
    root.mkdir(parents=True, exist_ok=False)
    try:
        pilot.write_json(root / "manifest.json", {**manifest, "arm": config.arm, "k": config.k,
                                                  "initial_buffer_sha256": pilot.digest(screen._i32(initial))})
        evaluations = []
        with optimizer.EFSGD(Path(manifest["executable"]["path"]), shape, initial, config.k,
                             timeout_s=60, log_path=root / "native.log") as native:
            evaluations.append(evaluate_cell(native, root, 0, raw, dataset, shape))
            with (root / "training.jsonl").open("x") as log, (root / "training_predictions.i32").open("xb") as predlog, \
                    (root / "training_gradients.i32").open("xb") as gradlog:
                for index in range(STEPS):
                    tokens, targets = pilot.window(raw, dataset, "train", index)
                    result = native.train(tokens, targets)
                    if result["step"] != index + 1:
                        raise ValueError("native update counter differs from actual window count")
                    pred, grad = screen._i32(result["predictions"]), screen._i32(result["gradients"])
                    if len(pred) != 128 * 4 or len(grad) != 128 * 128 * 4:
                        raise ValueError("incomplete native training arrays")
                    predlog.write(pred)
                    gradlog.write(grad)
                    log.write(json.dumps({"step": index + 1, "window_index": index,
                                          "predictions_sha256": pilot.digest(pred), "gradients_sha256": pilot.digest(grad)}) + "\n")
                    if index + 1 in CHECKPOINTS:
                        evaluations.append(evaluate_cell(native, root, index + 1, raw, dataset, shape))
        baselines = {str(step): exposure_baseline(raw, dataset, config.arm, step) for step in CHECKPOINTS}
        one_pass = exposure_baseline(raw, dataset, "anchors", STEPS)
        final, initial_eval = evaluations[-1], evaluations[0]
        comparator = min(baselines[str(STEPS)]["bigram"]["nll"], one_pass["bigram"]["nll"])
        context = final["context_disabled_nll"] - final["lut_normalized_nll"]
        summary = {"schema": SCHEMA, "complete": True, "arm": config.arm, "k": config.k, "seed": 0,
                   "training_updates": STEPS, "gradient_pair_exposures": STEPS * 128,
                   "host_fitting_pair_exposures": STEPS * 128 if config.arm == "count_fitted" else 0,
                   "distinct_training_pairs": STEPS * 128, "baselines_by_step": baselines,
                   "one_pass_final_baseline": one_pass, "evaluations": evaluations,
                   "conservative_bigram_advantage_nats": comparator - final["lut_normalized_nll"],
                   "context_advantage_nats": context,
                   "context_advantage_growth_nats": context - (initial_eval["context_disabled_nll"] - initial_eval["lut_normalized_nll"]),
                   "heldout_test_evaluated": False, "physical_fpga_execution": False,
                   "three_seed_quality_gate_evaluated": False, "larger_paid_learning_benchmark_gate": False}
        _frozen(manifest)
        pilot.write_json(root / "summary.json", summary)
        _seal(root)
        return summary
    except BaseException as error:
        pilot.write_json(root / "failure.json", {"complete": False, "type": type(error).__name__, "error": str(error)})
        _seal(root)
        raise


def verify_worker(root, manifest, arm, k):
    """Parent validates exact artifacts and recomputes all checkpoint evaluations."""
    _frozen(manifest)
    audit._assert_equal(_read(root / "artifact_manifest.json"), _artifacts(root), "sealed cell artifacts")
    initial = initial_buffer(pilot.layout(WIDTH), Path(manifest["dataset"]["path"]).read_bytes(), manifest["dataset"], arm)
    audit._assert_equal(_read(root / "manifest.json"),
                        {**manifest, "arm": arm, "k": k,
                         "initial_buffer_sha256": pilot.digest(screen._i32(initial))}, "cell declaration")
    summary = _read(root / "summary.json")
    expected = {"schema": SCHEMA, "complete": True, "arm": arm, "k": k, "seed": 0,
                "training_updates": STEPS, "gradient_pair_exposures": STEPS * 128,
                "host_fitting_pair_exposures": STEPS * 128 if arm == "count_fitted" else 0,
                "distinct_training_pairs": STEPS * 128, "heldout_test_evaluated": False,
                "physical_fpga_execution": False, "three_seed_quality_gate_evaluated": False,
                "larger_paid_learning_benchmark_gate": False}
    audit._assert_equal({key: summary.get(key) for key in expected}, expected, "cell coverage and scope")
    if (root / "failure.json").exists():
        raise ValueError("failed cell cannot be reported complete")
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], STEPS, max(VALIDATION) + 1)
    audit._assert_equal(dataset, manifest["dataset"], "verified corpus/split declaration")
    shape = pilot.layout(WIDTH)
    identity = _native_identity(manifest, shape)
    if (any(type(row["step"]) is not int for row in summary["evaluations"])
            or [row["step"] for row in summary["evaluations"]] != list(CHECKPOINTS)):
        raise ValueError("all checkpoints required")
    for evaluation in summary["evaluations"]:
        step = evaluation["step"]
        payload = (root / ("state_%06d.i32" % step)).read_bytes()
        state = optimizer.decode_state(payload, shape)
        checkpoint = _read(root / ("state_%06d.json" % step))
        audit._assert_equal(evaluation["checkpoint"], checkpoint, "checkpoint metadata")
        expected_checkpoint = {"schema": optimizer.SCHEMA, "file": "state_%06d.i32" % step,
                               "bytes": len(payload), "sha256": pilot.digest(payload),
                               "step": state["step"], "k": state["k"], "config": state["config"],
                               "identity": identity,
                               "regions": [{"name": name, "weight_words": getattr(shape, "WORDS_" + name),
                                            "remainder_words": getattr(shape, "WORDS_" + name)} for name in pilot.WEIGHTS],
                               "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}
        audit._assert_equal(checkpoint, expected_checkpoint, "complete optimizer checkpoint metadata")
        if (state["step"] != step or state["k"] != k or pilot.digest(payload) != evaluation["state_sha256"]
                or checkpoint["file"] != "state_%06d.i32" % step or checkpoint["sha256"] != pilot.digest(payload)
                or type(checkpoint["bytes"]) is not int or checkpoint["bytes"] != len(payload)
                or state["config"]["loss_kind"] != 1 or state["config"]["zshift"] != ZSHIFT
                or evaluation["full_state_evaluation_invariant"] is not True
                or evaluation["full_state_import_equal"] is not True):
            raise ValueError("full optimizer state checkpoint differs")
        if step == 0:
            initial_weights = {name: initial[getattr(shape, "OFF_" + name):
                                           getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
                               for name in pilot.WEIGHTS}
            if state["weights"] != initial_weights or any(any(values) for values in state["remainders"].values()):
                raise ValueError("initial weights or zero optimizer remainders differ from declaration")
        rows = []
        for index in VALIDATION:
            tokens, targets = pilot.window(raw, dataset, "validation", index)
            expected_window = observe(state["weights"], tokens, targets, shape, zshift=ZSHIFT)
            row = _read(root / ("eval_%06d_%03d.json" % (step, index)))
            audit._assert_equal({key: row.get(key) for key in ("window_index", "native_reference_equal", "native_routes_equal")},
                                {"window_index": index, "native_reference_equal": True, "native_routes_equal": True},
                                "actual native window coverage and equality")
            if (type(row.get("saturation_events")) is not int or row["saturation_events"] != 0
                    or expected_window["saturation_events"] != 0):
                raise ValueError("clipping violates the native optimizer's fail-closed study protocol")
            expected_arrays = {}
            for name, values in {"predictions": expected_window["predictions"], "gradients": expected_window["gradients"],
                                 **expected_window["native_routes"]}.items():
                actual_path = root / ("eval_%06d_%03d_%s.i32" % (step, index, name))
                array_payload = screen._i32(values)
                if actual_path.read_bytes() != array_payload:
                    raise ValueError("saved native array differs from independent checkpoint reference")
                expected_arrays[name] = {"file": actual_path.name, "words": len(values), "bytes": len(array_payload),
                                         "sha256": pilot.digest(array_payload), "format": "little-endian signed int32"}
            audit._assert_equal(row["native_arrays"], expected_arrays, "native array metadata")
            audit._assert_equal({key: row[key] for key in screen._scalar_result(expected_window)},
                                screen._scalar_result(expected_window), "checkpoint diagnostic")
            disabled = pilot.integer_forward(state["weights"], tokens, targets, shape, zshift=ZSHIFT, use_context=False)
            audit._assert_equal(row["probes"]["context_disabled"], screen._scalar_result(disabled), "context probe")
            rows.append(row)
        audit._assert_equal({key: evaluation[key] for key in screen._aggregate(rows)}, screen._aggregate(rows), "aggregate")
    audit._assert_equal(summary["baselines_by_step"],
                        {str(step): exposure_baseline(raw, dataset, arm, step) for step in CHECKPOINTS}, "exposure controls")
    one_pass = exposure_baseline(raw, dataset, "anchors", STEPS)
    audit._assert_equal(summary["one_pass_final_baseline"], one_pass, "one-pass comparator")
    final, first = summary["evaluations"][-1], summary["evaluations"][0]
    comparator = min(summary["baselines_by_step"][str(STEPS)]["bigram"]["nll"], one_pass["bigram"]["nll"])
    context = final["context_disabled_nll"] - final["lut_normalized_nll"]
    for key, value in {"conservative_bigram_advantage_nats": comparator - final["lut_normalized_nll"],
                       "context_advantage_nats": context,
                       "context_advantage_growth_nats": context - (first["context_disabled_nll"] - first["lut_normalized_nll"])}.items():
        audit._assert_equal(summary[key], value, key)
    lines = (root / "training.jsonl").read_text().splitlines()
    if len(lines) != STEPS:
        raise ValueError("training log coverage differs")
    for name, count in (("predictions", 128), ("gradients", 16384)):
        data = (root / ("training_%s.i32" % name)).read_bytes()
        if len(data) != STEPS * count * 4:
            raise ValueError("training native array coverage differs")
        for index, line in enumerate(lines):
            item = pilot._finite_json(line)
            if (type(item["step"]) is not int or item["step"] != index + 1
                    or type(item["window_index"]) is not int or item["window_index"] != index
                    or pilot.digest(data[index * count * 4:(index + 1) * count * 4]) != item[name + "_sha256"]):
                raise ValueError("training array/log identity differs")
    _frozen(manifest)
    audit._assert_equal(_read(root / "artifact_manifest.json"), _artifacts(root), "final cell artifacts")
    return summary


def _run_in_directory(config, root, dataset):
    executable = optimizer.build(root / "build", width=WIDTH)
    manifest = {"schema": SCHEMA, "recipe": recipe(), "dataset": dataset,
                "source_hashes": source_identity(), "plan_sha256": pilot.digest(PLAN.read_bytes()),
                "executable": {"path": str(executable.resolve()), "sha256": pilot.digest(executable.read_bytes())}}
    pilot.write_json(root / "manifest.json", manifest)
    rows, failures = [], []
    for arm in ARMS:
        for k in K_VALUES:
            name = "%s_k%d" % (arm, k)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--corpus", dataset["path"],
                       "--output", str(root / name), "--parent", str(root), "--arm", arm, "--k", str(k)]
            try:
                execution = _run_process(command, root / (name + ".log"))
                pilot.write_json(root / (name + ".execution.json"), execution)
                if (type(execution["returncode"]) is not int or execution["returncode"] != 0
                        or execution["timed_out"] is not False or type(execution["timeout_s"]) is not int
                        or execution["timeout_s"] != WORKER_TIMEOUT_S
                        or type(execution["elapsed_s"]) not in (int, float)
                        or not math.isfinite(execution["elapsed_s"]) or execution["elapsed_s"] < 0):
                    raise RuntimeError("native cell failed or exceeded its fixed timeout")
                rows.append(verify_worker(root / name, manifest, arm, k))
            except Exception as error:
                failures.append({"cell": name, "type": type(error).__name__, "error": str(error)})
    _frozen(manifest)
    result = {"schema": SCHEMA, "complete": not failures and len(rows) == 4, "recipe": recipe(),
              "source_hashes": manifest["source_hashes"], "declared_cells": 4, "completed_cells": len(rows),
              "cells": rows, "failures": failures, "heldout_test_evaluated": False,
              "physical_fpga_execution": False, "speed_advantage_claim": False,
              "three_seed_quality_gate_evaluated": False, "larger_paid_learning_benchmark_gate": False,
              "limitations": ["One initialization seed, reused development validation, no selection-adjusted confidence.",
                              "Count-fitted initialization is a separate charged host learning pass, not FPGA pretraining.",
                              "Requires a new synthesized optimizer/state interface and physical checkpoint verification.",
                              "Training arrays are coverage/hash checked; not every training update is independently replayed."]}
    pilot.write_json(root / "summary.json", result)
    _seal(root)
    return result


def run(config):
    root = capacity._new_external_path(config.output)
    raw, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    del raw
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _run_in_directory(config, root, dataset)
    except BaseException as error:
        if not (root / "failure.json").exists():
            pilot.write_json(root / "failure.json", {"complete": False, "type": type(error).__name__, "error": str(error)})
        if not (root / "artifact_manifest.json").exists():
            _seal(root)
        raise


def publish_record(path, root, result):
    target = capacity._new_external_path(path)
    root = Path(root).resolve(strict=True)
    if target == root or root in target.parents:
        raise ValueError("compact record must be outside the raw run")
    expected = {"schema": SCHEMA, "complete": True, "declared_cells": 4, "completed_cells": 4, "failures": [],
                "heldout_test_evaluated": False, "physical_fpga_execution": False, "speed_advantage_claim": False,
                "three_seed_quality_gate_evaluated": False, "larger_paid_learning_benchmark_gate": False}
    audit._assert_equal({name: result.get(name) for name in expected}, expected, "complete publication scope")
    audit._assert_equal(result["recipe"], recipe(), "published protocol")
    audit._assert_equal([[row["arm"], row["k"]] for row in result["cells"]],
                        [[arm, k] for arm in ARMS for k in K_VALUES], "all declared cells")
    audit._assert_equal(_read(root / "summary.json"), result, "raw summary")
    manifest = _read(root / "manifest.json")
    _frozen(manifest)
    audit._assert_equal(result["source_hashes"], manifest["source_hashes"], "published source identities")
    for row in result["cells"]:
        audit._assert_equal(row, verify_worker(root / ("%s_k%d" % (row["arm"], row["k"])),
                                              manifest, row["arm"], row["k"]), "published cell")
    declared = _read(root / "artifact_manifest.json")
    audit._assert_equal(declared, _artifacts(root), "entire raw artifact tree")
    if any("failure" in Path(row["file"]).name for row in declared["files"]):
        raise ValueError("cannot publish a successful record over any retained failure")
    cells = []
    for row in result["cells"]:
        compact = {key: value for key, value in row.items() if key != "evaluations"}
        compact["evaluations"] = [{key: value for key, value in evaluation.items() if key != "per_window"}
                                  for evaluation in row["evaluations"]]
        cells.append(compact)
    value = {**result, "cells": cells, "run_directory": str(root),
             "artifact_manifest_sha256": pilot.digest((root / "artifact_manifest.json").read_bytes()),
             "artifact_files_verified": len(declared["files"]),
             "artifact_payload_bytes": sum(row["bytes"] for row in declared["files"]),
             "full_optimizer_checkpoints_verified": 12, "checkpoint_validation_windows": 192,
             "distinct_validation_positions": 2048, "new_native_updates": 4096,
             "gradient_pair_exposures": 524288, "host_fitting_pair_exposures": 262144,
             "distinct_training_pairs": 131072, "native_training_trajectory_independently_replayed": False}
    _frozen(manifest)
    pilot.write_json(target, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, help="new compact evidence record outside all raw artifact trees")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--k", type=int, choices=K_VALUES, help=argparse.SUPPRESS)
    config = parser.parse_args(argv)
    if config.worker:
        if config.parent is None or config.arm is None or config.k is None or config.record is not None:
            parser.error("internal worker requires parent, arm and k")
        result = run_worker(config)
    else:
        if any(value is not None for value in (config.parent, config.arm, config.k)):
            parser.error("cell overrides are not a different declared public protocol")
        output = capacity._new_external_path(config.output)
        if config.record is not None:
            target = capacity._new_external_path(config.record)
            if target == output or output in target.parents:
                parser.error("record must be outside the raw output")
        result = run(config)
        if config.record is not None and result["complete"] is True:
            publish_record(config.record, config.output, result)
    print(json.dumps({"complete": result["complete"], "output": str(config.output)}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
