"""One matched warm-start SignSGD development control, using the frozen kernel.

This imports the completed count-fitted EF k18 initial weights; it does not fit
new model counts. One sequential-order CPU cell is compared with that historical
EF cell. It is neither confirmation across seeds nor authorization for hardware.
Every training step is preflighted with the frozen host route observer and its
complete weight update is checked. The external 1200-second limit remains fixed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import confirm_fpga_lm_efsgd as confirmation  # noqa: E402
from manuscript.tools import train_fpga_lm_efsgd as prior  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from manuscript.tools.observe_fpga_lm_routes import observe  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_warm_signsgd.v1"
WIDTH, ZSHIFT, LR, THRESHOLD, STEPS = 512, 2, 1, 2048, 1024
CHECKPOINTS = (0, 256, 1024)
VALIDATION = tuple(range(40, 56))
WORKER_TIMEOUT_S = 1200
PLAN = REPO / "manuscript/research/PLAN.md"
_read, _equal, _run_process = prior._read, prior.audit._assert_equal, prior._run_process
bind_prior = confirmation.bind_prior
initial_buffer = confirmation.initial_buffer
confirm, optimizer = confirmation, prior.optimizer


def source_identity():
    result = confirmation.source_identity()
    path = Path(__file__).resolve()
    result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def recipe():
    return {"width": WIDTH, "zshift": ZSHIFT, "lr": LR, "threshold": THRESHOLD,
            "steps": STEPS, "training_indices": list(range(STEPS)), "checkpoints": list(CHECKPOINTS),
            "validation_indices": list(VALIDATION), "declared_cells": 1,
            "initialization": "exact full weight payload from completed prior count_fitted_k18 state0; zero native gradients",
            "optimizer": "frozen integer SignSGD: w-=sign(G)*lr iff abs(G)>threshold; clear gradients",
            "comparison": "completed count_fitted_k18 EF cell; same original weights, sequential windows, z2 and development40..55",
            "inherited_host_fitting_pair_exposures": STEPS * 128, "new_host_fitting_pair_exposures": 0,
            "initialization_refitted": False, "new_gradient_pair_exposures": STEPS * 128,
            "model_history_pair_exposures": 2 * STEPS * 128, "distinct_training_pairs": STEPS * 128,
            "baseline_policy": "alpha1 full inherited fit plus actual sequential replay prefix; retain one-pass comparator too",
            "training_guard": "host observer every window: no forward saturation/int32 overflow; preflight all int16 updates; "
                              "compare actual native predictions/logit gradients/routes and all post-update weights",
            "evaluation": "native lr0 mode3 still executes backward; complete weights unchanged; fixed context-disabled host probe",
            "verification": "every training step checked by worker; posthoc verifier rehashes training arrays/weight-chain receipts "
                            "and recomputes checkpoints, not a second complete training replay",
            "worker_timeout_s": WORKER_TIMEOUT_S,
            "scope": "one causal development control, not three-seed confirmation, final test, speed, physical FPGA or paid gate"}


def _bounded_regions(weights, gradients, shape):
    """Validate the shared native integer and abs(G) preconditions."""
    import numpy as np
    if set(weights) != set(pilot.WEIGHTS) or set(gradients) != set(pilot.WEIGHTS):
        raise ValueError("all declared weight and parameter-gradient regions are required")
    for name in pilot.WEIGHTS:
        for values in (weights[name], gradients[name]):
            if isinstance(values, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in values):
                raise ValueError("boolean values are not integer weights or gradients")
        w, g = np.asarray(weights[name]), np.asarray(gradients[name])
        count = getattr(shape, "WORDS_" + name)
        if (w.ndim != 1 or g.ndim != 1 or len(w) != count or len(g) != count
                or w.dtype.kind not in "iu" or g.dtype.kind not in "iu"
                or np.any(w < -32768) or np.any(w > 32767)
                or np.any(g < -(1 << 31)) or np.any(g >= (1 << 31))):
            raise ValueError("exact bounded full weight/gradient integer arrays are required")
        w, g = w.astype(np.int64), g.astype(np.int64)
        # Frozen C++ computes abs on int32: reject INT_MIN even though the
        # mathematical int64 oracle itself could represent its absolute value.
        if np.any(g == -(1 << 31)):
            raise ValueError("frozen SignSGD abs(INT_MIN) would overflow")
        yield name, w, g


def validate_parameter_gradients(weights, gradients, shape):
    """lr0 evaluation executes abs(G), but must not test an lr1 update."""
    for _ in _bounded_regions(weights, gradients, shape):
        pass


def signsgd_candidate(weights, gradients, shape):
    """Exact lr1 no-clipping update oracle; never mutate caller-owned arrays."""
    import numpy as np
    result = {}
    for name, w, g in _bounded_regions(weights, gradients, shape):
        candidate = w - np.where(np.abs(g) > THRESHOLD, np.sign(g) * LR, 0)
        if np.any(candidate < -32768) or np.any(candidate > 32767):
            raise ValueError("native SignSGD weight update would clip")
        result[name] = candidate.tolist()
    return result


def _checked_observe(weights, tokens, targets, shape, observer=None, parameters=False):
    observer = observe if observer is None else observer
    result = observer(weights, tokens, targets, shape, zshift=ZSHIFT,
                      include_parameter_gradients=parameters)
    if (type(result.get("saturation_events")) is not int or result["saturation_events"] != 0
            or result["route_diagnostics"].get("backward_int32_overflow_detected") is not False):
        raise ValueError("observer clipping or int32-overflow guard failed")
    return result


def _artifacts(root):
    result = prior._artifacts(root)
    result["schema"] = SCHEMA
    return result


def _seal(root):
    pilot.write_json(root / "artifact_manifest.json", _artifacts(root))


def _verify_library(manifest):
    root = Path(manifest["run_directory"]).resolve(strict=True)
    library = manifest["library"]
    path = Path(library["path"])
    if path != root / "build" / path.name or path.name not in ("libmorphlm.so", "libmorphlm.dylib"):
        raise ValueError("native library must be the exact owned source-bound build")
    build = prior.audit._verify_library(root, library, manifest)
    if pilot.digest((root / "build/build.json").read_bytes()) != manifest["build_record_sha256"]:
        raise ValueError("native build receipt changed")
    return build


def _frozen(manifest):
    _equal(manifest["schema"], SCHEMA, "warm SignSGD schema")
    _equal(manifest["recipe"], recipe(), "fixed warm SignSGD recipe")
    _equal(manifest["source_hashes"], source_identity(), "warm SignSGD frozen sources")
    if pilot.digest(Path(manifest["dataset"]["path"]).read_bytes()) != manifest["dataset"]["sha256"]:
        raise ValueError("source corpus changed")
    binding = manifest["prior_evidence"]
    _equal(binding, bind_prior(binding["run_directory"], binding["record"]["file"], manifest["dataset"], deep=False),
           "unchanged complete prior raw evidence and publication")
    _equal(manifest["layout"], pilot.layout_values(pilot.layout(WIDTH)), "native model layout")
    _verify_library(manifest)
    root = Path(manifest["run_directory"])
    _equal(_read(root / "protocol.json"), {"schema": SCHEMA, **recipe()}, "pre-execution declaration")


def _weights_from_initial(shape, binding):
    payload = confirmation.read_initial_state(binding, shape)
    return prior.optimizer.decode_state(payload, shape)["weights"]


def exposure_baseline(raw, dataset, step):
    if type(step) is not int or step not in CHECKPOINTS:
        raise ValueError("undeclared sequential replay checkpoint")
    # This frozen helper has precisely the original sequential-order policy
    # and original development cohort. No global constants are modified.
    return prior.exposure_baseline(raw, dataset, "count_fitted", step)


def _native_arrays(bridge, shape):
    result = {"predictions": bridge.read(shape.OFF_PRED, shape.WORDS_PRED),
              "gradients": bridge.read(shape.OFF_GRAD_LOGIT, shape.WORDS_GRAD_LOGIT)}
    result.update({name: bridge.read(getattr(shape, "OFF_" + name), getattr(shape, "WORDS_" + name))
                   for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")})
    return result


def _expected_arrays(observed):
    return {"predictions": observed["predictions"], "gradients": observed["gradients"],
            **observed["native_routes"]}


def _array_receipt(filename, values):
    payload = screen._i32(values)
    return {"file": filename, "words": len(values), "bytes": len(payload),
            "sha256": pilot.digest(payload), "format": "little-endian signed int32"}


def _context_probe(weights, tokens, targets, shape):
    result = pilot.integer_forward(weights, tokens, targets, shape, zshift=ZSHIFT, use_context=False)
    if type(result.get("saturation_events")) is not int or result["saturation_events"] != 0:
        raise ValueError("context-disabled forward probe encountered clipping")
    return result


def evaluate_cell(bridge, root, step, weights, raw, dataset, shape, observer=None):
    expected_weight_bytes = pilot.weight_bytes(weights)
    if pilot.weight_bytes(pilot.read_weights(bridge, shape)) != expected_weight_bytes:
        raise ValueError("weights changed before evaluation")
    checkpoint = pilot.checkpoint(root, step, weights, shape)
    rows = []
    for index in VALIDATION:
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        expected = _checked_observe(weights, tokens, targets, shape, observer, parameters=True)
        # mode3/lr0 still invokes abs(G), but applies no weight update.
        validate_parameter_gradients(weights, expected["parameter_gradients"], shape)
        pilot.stage_window(bridge, shape, tokens, targets)
        bridge.step(0, THRESHOLD)
        actual = _native_arrays(bridge, shape)
        arrays = {name: screen._save_array(root, "eval_%06d_%03d_%s.i32" % (step, index, name), values)
                  for name, values in actual.items()}
        _equal(actual, _expected_arrays(expected), "native evaluation predictions, gradients and routes")
        disabled = _context_probe(weights, tokens, targets, shape)
        row = {**screen._scalar_result(expected), "window_index": index, "native_arrays": arrays,
               "native_reference_equal": True, "native_routes_equal": True,
               "probes": {"context_disabled": screen._scalar_result(disabled)}}
        pilot.write_json(root / ("eval_%06d_%03d.json" % (step, index)), row)
        rows.append(row)
    if pilot.weight_bytes(pilot.read_weights(bridge, shape)) != expected_weight_bytes:
        raise ValueError("lr0 validation changed complete weights")
    return {"step": step, "checkpoint": checkpoint, "full_weights_evaluation_invariant": True,
            **screen._aggregate(rows)}


def _ef_comparator(binding):
    root = Path(binding["run_directory"]) / "count_fitted_k18"
    cell = _read(root / "summary.json")
    # Full historical verification happens before the parent executes a build;
    # every later _frozen call checks its entire immutable artifact tree.
    fields = ("step", "lut_normalized_nll", "accuracy", "context_disabled_nll",
              "saturation_events", "rounded_zero_target_mass", "native_reference_equal", "native_routes_equal")
    return {"arm": "count_fitted", "k": 18,
            "historical_cell_summary_sha256": pilot.digest((root / "summary.json").read_bytes()),
            "evaluations": [{name: row[name] for name in fields} for row in cell["evaluations"]]}


def _cell_summary(evaluations, raw, dataset, binding):
    controls = {str(step): exposure_baseline(raw, dataset, step) for step in CHECKPOINTS}
    one = prior.exposure_baseline(raw, dataset, "anchors", STEPS)
    ef = _ef_comparator(binding)
    if [row["step"] for row in ef["evaluations"]] != list(CHECKPOINTS):
        raise ValueError("historical EF comparator lacks the exact declared checkpoints")
    first, final = evaluations[0], evaluations[-1]
    _equal({key: first[key] for key in ("lut_normalized_nll", "accuracy", "context_disabled_nll")},
           {key: ef["evaluations"][0][key] for key in ("lut_normalized_nll", "accuracy", "context_disabled_nll")},
           "same actual initialization and development cohort")
    comparator = min(controls[str(STEPS)]["bigram"]["nll"], one["bigram"]["nll"])
    context = final["context_disabled_nll"] - final["lut_normalized_nll"]
    return {"schema": SCHEMA, "complete": True, "optimizer": "frozen SignSGD", "training_updates": STEPS,
            "initial_weights_equal_prior": True, "inherited_host_fitting_pair_exposures": STEPS * 128,
            "new_host_fitting_pair_exposures": 0, "initialization_refitted": False,
            "new_gradient_pair_exposures": STEPS * 128, "model_history_pair_exposures": 2 * STEPS * 128,
            "distinct_training_pairs": STEPS * 128, "evaluations": evaluations,
            "baselines_by_step": controls, "one_pass_baseline": one, "paired_historical_ef": ef,
            "conservative_bigram_advantage_nats": comparator - final["lut_normalized_nll"],
            "context_advantage_nats": context,
            "context_advantage_growth_nats": context - (first["context_disabled_nll"] - first["lut_normalized_nll"]),
            "signsgd_minus_ef_final_nll": final["lut_normalized_nll"] - ef["evaluations"][-1]["lut_normalized_nll"],
            "training_steps_guarded_and_compared_by_worker": STEPS,
            "posthoc_complete_training_trajectory_replayed": False,
            "heldout_test_evaluated": False, "physical_fpga_execution": False,
            "quality_gate_candidate": False, "three_seed_quality_gate_evaluated": False,
            "larger_paid_learning_benchmark_gate": False, "paid_execution_authorized": False,
            "statistical_significance_claim": False, "speed_advantage_claim": False}


def run_worker(config, bridge_factory=None, observer=None):
    observer = observe if observer is None else observer
    parent, root = Path(config.parent).resolve(strict=True), Path(config.output).resolve()
    if root != parent / "signsgd":
        raise ValueError("worker must own exactly the declared signsgd child directory")
    manifest = _read(parent / "manifest.json")
    if manifest["run_directory"] != str(parent):
        raise ValueError("worker parent does not own this declaration")
    _frozen(manifest)
    raw, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    _equal(dataset, manifest["dataset"], "worker corpus/split identity")
    shape = pilot.layout(WIDTH)
    initial = initial_buffer(shape, manifest["prior_evidence"])
    weights = _weights_from_initial(shape, manifest["prior_evidence"])
    root.mkdir(parents=True, exist_ok=False)
    bridge, completed, failure = None, 0, None
    try:
        pilot.write_json(root / "manifest.json", {**manifest, "initial_buffer_sha256": pilot.digest(screen._i32(initial))})
        bridge = (bridge_factory or pilot.NativeBridge)(Path(manifest["library"]["path"]), "csim", shape)
        bridge.write(0, initial)
        if bridge.read(0, len(initial)) != initial:
            raise ValueError("complete initial buffer transport differs")
        evaluations = [evaluate_cell(bridge, root, 0, weights, raw, dataset, shape, observer)]
        with (root / "training.jsonl").open("x") as log, (root / "training_predictions.i32").open("xb") as predlog, \
                (root / "training_gradients.i32").open("xb") as gradlog:
            for index in range(STEPS):
                tokens, targets = pilot.window(raw, dataset, "train", index)
                before = pilot.weight_bytes(weights)
                expected = _checked_observe(weights, tokens, targets, shape, observer, parameters=True)
                candidate = signsgd_candidate(weights, expected["parameter_gradients"], shape)
                after = pilot.weight_bytes(candidate)
                pilot.stage_window(bridge, shape, tokens, targets)
                bridge.step(LR, THRESHOLD)
                actual = _native_arrays(bridge, shape)
                pred, grad = screen._i32(actual["predictions"]), screen._i32(actual["gradients"])
                predlog.write(pred)
                gradlog.write(grad)
                _equal(actual, _expected_arrays(expected), "native training outputs and saved routes")
                if pilot.weight_bytes(pilot.read_weights(bridge, shape)) != after:
                    raise ValueError("complete native post-update weights differ from guarded SignSGD oracle")
                completed = index + 1
                log.write(json.dumps({"step": completed, "window_index": index, "token_pairs": 128,
                    "input_readback_equal": True, "saturation_events": 0, "backward_int32_overflow_detected": False,
                    "weight_update_clipping": False, "native_reference_equal": True, "native_routes_equal": True,
                    "full_weight_update_equal": True, "before_weights_sha256": pilot.digest(before),
                    "after_weights_sha256": pilot.digest(after), "predictions_sha256": pilot.digest(pred),
                    "gradients_sha256": pilot.digest(grad),
                    "route_hashes": {name: {"words": len(values), "sha256": pilot.digest(screen._i32(values))}
                                     for name, values in actual.items() if name not in ("predictions", "gradients")}}) + "\n")
                log.flush()
                weights = candidate
                if completed in CHECKPOINTS:
                    evaluations.append(evaluate_cell(bridge, root, completed, weights, raw, dataset, shape, observer))
        closing, bridge = bridge, None
        closing.close()
        summary = _cell_summary(evaluations, raw, dataset, manifest["prior_evidence"])
        _frozen(manifest)
        pilot.write_json(root / "summary.json", summary)
    except BaseException as error:
        failure = error
        pilot.write_json(root / "failure.json", {"complete": False, "training_updates_completed": completed,
                                                 "type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if bridge is not None:
            try:
                bridge.close()
            except BaseException as error:
                pilot.write_json(root / "cleanup_failure.json", {"type": type(error).__name__, "error": str(error)})
                if failure is None:
                    raise
        _seal(root)
    return summary


def _checkpoint_weights(root, step, shape, record):
    path = root / ("checkpoint_%06d.i32" % step)
    payload = path.read_bytes()
    expected = {"step": step, "file": path.name, "location_portable": True,
                "sha256": pilot.digest(payload), "bytes": len(payload),
                "format": "little-endian signed int32, low16 weights",
                "regions": [{"name": name, "words": getattr(shape, "WORDS_" + name)} for name in pilot.WEIGHTS]}
    _equal(record, expected, "exact full weight checkpoint metadata")
    _equal(_read(path.with_suffix(".json")), expected, "saved weight checkpoint metadata")
    values, offset, weights = prior.optimizer._unpack(payload), 0, {}
    for name in pilot.WEIGHTS:
        count = getattr(shape, "WORDS_" + name)
        weights[name] = values[offset:offset + count]
        if len(weights[name]) != count or any(not -32768 <= value <= 32767 for value in weights[name]):
            raise ValueError("invalid complete low16 checkpoint weights")
        offset += count
    if offset != len(values):
        raise ValueError("checkpoint has extra data")
    return weights


def _verify_evaluation(root, evaluation, shape, raw, dataset):
    step = evaluation["step"]
    if evaluation.get("full_weights_evaluation_invariant") is not True:
        raise ValueError("validation weight invariance receipt required")
    weights = _checkpoint_weights(root, step, shape, evaluation["checkpoint"])
    rows = []
    for index in VALIDATION:
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        expected = _checked_observe(weights, tokens, targets, shape, parameters=True)
        validate_parameter_gradients(weights, expected["parameter_gradients"], shape)
        arrays = {}
        for name, values in _expected_arrays(expected).items():
            filename = "eval_%06d_%03d_%s.i32" % (step, index, name)
            if (root / filename).read_bytes() != screen._i32(values):
                raise ValueError("native checkpoint array differs from independent observer")
            arrays[name] = _array_receipt(filename, values)
        disabled = _context_probe(weights, tokens, targets, shape)
        row = {**screen._scalar_result(expected), "window_index": index, "native_arrays": arrays,
               "native_reference_equal": True, "native_routes_equal": True,
               "probes": {"context_disabled": screen._scalar_result(disabled)}}
        _equal(_read(root / ("eval_%06d_%03d.json" % (step, index))), row, "complete native checkpoint row")
        rows.append(row)
    _equal(evaluation, {"step": step, "checkpoint": evaluation["checkpoint"],
                        "full_weights_evaluation_invariant": True, **screen._aggregate(rows)}, "checkpoint aggregate")
    return weights


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("exact lowercase SHA256 required")
    return value


def verify_worker(root, manifest):
    """Recompute saved evaluations; verify every training byte and guard receipt.

The source-bound worker itself checks every update against the complete host
oracle. This posthoc pass does not rerun all 1024 training gradient evaluations.
"""
    root = Path(root).resolve(strict=True)
    if root != Path(manifest["run_directory"]) / "signsgd":
        raise ValueError("unexpected worker directory")
    _frozen(manifest)
    artifact_bytes = (root / "artifact_manifest.json").read_bytes()
    _equal(_read(root / "artifact_manifest.json"), _artifacts(root), "complete worker artifact tree")
    if any("failure" in Path(row["file"]).name for row in _artifacts(root)["files"]):
        raise ValueError("a failed cell cannot be complete evidence")
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], STEPS, max(VALIDATION) + 1)
    _equal(dataset, manifest["dataset"], "source corpus/split identity")
    shape = pilot.layout(WIDTH)
    initial = initial_buffer(shape, manifest["prior_evidence"])
    _equal(_read(root / "manifest.json"), {**manifest, "initial_buffer_sha256": pilot.digest(screen._i32(initial))},
           "worker declaration and complete initialization")
    summary = _read(root / "summary.json")
    evaluations = summary["evaluations"]
    if (any(type(row["step"]) is not int for row in evaluations)
            or [row["step"] for row in evaluations] != list(CHECKPOINTS)):
        raise ValueError("every declared checkpoint is required")
    checkpoint_hashes = {}
    for row in evaluations:
        weights = _verify_evaluation(root, row, shape, raw, dataset)
        checkpoint_hashes[row["step"]] = pilot.digest(pilot.weight_bytes(weights))
        if row["step"] == 0 and pilot.weight_bytes(weights) != pilot.weight_bytes(
                _weights_from_initial(shape, manifest["prior_evidence"])):
            raise ValueError("actual initial full weights differ from the completed EF state")
    _equal(summary, _cell_summary(evaluations, raw, dataset, manifest["prior_evidence"]), "complete causal-control scope")
    lines = (root / "training.jsonl").read_text().splitlines()
    if len(lines) != STEPS:
        raise ValueError("training coverage differs from the fixed budget")
    arrays = {name: (root / ("training_%s.i32" % name)).read_bytes() for name in ("predictions", "gradients")}
    words = {"predictions": 128, "gradients": 16384}
    if any(len(arrays[name]) != STEPS * count * 4 for name, count in words.items()):
        raise ValueError("training array coverage is incomplete")
    previous = checkpoint_hashes[0]
    for index, line in enumerate(lines):
        row = pilot._finite_json(line)
        expected = {"step": index + 1, "window_index": index, "token_pairs": 128,
                    "input_readback_equal": True, "saturation_events": 0,
                    "backward_int32_overflow_detected": False, "weight_update_clipping": False,
                    "native_reference_equal": True, "native_routes_equal": True, "full_weight_update_equal": True,
                    "before_weights_sha256": previous, "after_weights_sha256": _sha(row["after_weights_sha256"]),
                    "route_hashes": {name: {"words": getattr(shape, "WORDS_" + name),
                                             "sha256": _sha(row["route_hashes"][name]["sha256"])}
                                     for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}
        for name, count in words.items():
            expected[name + "_sha256"] = pilot.digest(arrays[name][index * count * 4:(index + 1) * count * 4])
        _equal(row, expected, "training order, arrays, complete weight chain and runtime guard receipts")
        previous = row["after_weights_sha256"]
        if index + 1 in checkpoint_hashes and previous != checkpoint_hashes[index + 1]:
            raise ValueError("training weight chain does not reach the actual saved checkpoint")
    _frozen(manifest)
    if (root / "artifact_manifest.json").read_bytes() != artifact_bytes:
        raise ValueError("worker seal changed during verification")
    _equal(_read(root / "artifact_manifest.json"), _artifacts(root), "worker artifacts unchanged during verification")
    return summary


def _expected_command(root, manifest):
    binding = manifest["prior_evidence"]
    return [sys.executable, str(Path(__file__).resolve()), "--worker", "--corpus", manifest["dataset"]["path"],
            "--prior-run", binding["run_directory"], "--prior-record", binding["record"]["file"],
            "--output", str(root / "signsgd"), "--parent", str(root)]


def _verify_execution(root, manifest):
    row = _read(root / "signsgd.execution.json")
    if (type(row.get("returncode")) is not int or row["returncode"] != 0
            or row.get("timed_out") is not False or type(row.get("timeout_s")) is not int
            or row["timeout_s"] != WORKER_TIMEOUT_S or type(row.get("elapsed_s")) not in (int, float)
            or not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < 0):
        raise ValueError("native worker did not finish successfully inside the declared bound")
    _equal(row, {"command": _expected_command(root, manifest), "returncode": 0, "timed_out": False,
                 "timeout_s": WORKER_TIMEOUT_S, "elapsed_s": row["elapsed_s"]}, "exact owned worker command and execution")


def _parent_summary(manifest, cells, failures):
    return {"schema": SCHEMA, "complete": len(cells) == 1 and not failures,
            "recipe": recipe(), "dataset": manifest["dataset"], "source_hashes": manifest["source_hashes"],
            "prior_evidence": manifest["prior_evidence"], "declared_cells": 1, "completed_cells": len(cells),
            "cells": cells, "failures": failures, "physical_fpga_execution": False, "cloud_operations": False,
            "heldout_test_evaluated": False, "quality_gate_candidate": False,
            "three_seed_quality_gate_evaluated": False, "larger_paid_learning_benchmark_gate": False,
            "paid_execution_authorized": False, "statistical_significance_claim": False, "speed_advantage_claim": False,
            "limitations": ["A single sequential-order development control, not new quality confirmation.",
                            "Inherited host count fitting is part of model history, not new fitting or FPGA learning.",
                            "Optimizer comparison changes magnitude/deadband/update scaling together; it does not isolate remainder alone.",
                            "Native SignSGD is the frozen algorithm, but this is CPU CSIM, not physical image equivalence.",
                            "Training update comparisons occurred in the worker; posthoc checks rehash receipts/arrays and recompute checkpoints.",
                            "Training route digests are retained, not complete raw training route arrays."]}


def _run_in_directory(config, root, dataset, binding, frozen_sources):
    # The fixed machine-readable declaration is written before native building.
    pilot.write_json(root / "protocol.json", {"schema": SCHEMA, **recipe()})
    if source_identity() != frozen_sources:
        raise ValueError("control sources changed before native build")
    library_path = pilot.build_csim(root / "build", WIDTH).resolve()
    if source_identity() != frozen_sources:
        raise ValueError("control sources changed during native build")
    manifest = {"schema": SCHEMA, "recipe": recipe(), "run_directory": str(root), "dataset": dataset,
                "prior_evidence": binding, "layout": pilot.layout_values(pilot.layout(WIDTH)),
                "source_hashes": frozen_sources, "plan_sha256": pilot.digest(PLAN.read_bytes()),
                "library": {"path": str(library_path), "sha256": pilot.digest(library_path.read_bytes())},
                "build_record_sha256": pilot.digest((root / "build/build.json").read_bytes())}
    pilot.write_json(root / "manifest.json", manifest)
    _frozen(manifest)
    command = _expected_command(root, manifest)
    cells, failures = [], []
    try:
        execution = _run_process(command, root / "signsgd.log")
        pilot.write_json(root / "signsgd.execution.json", {"command": command, **execution})
        _verify_execution(root, manifest)
        cells.append(verify_worker(root / "signsgd", manifest))
    except Exception as error:
        failure = {"type": type(error).__name__, "error": str(error)}
        pilot.write_json(root / "signsgd.failure.json", failure)
        failures.append(failure)
    _frozen(manifest)
    summary = _parent_summary(manifest, cells, failures)
    pilot.write_json(root / "summary.json", summary)
    _seal(root)
    return summary


def run(config):
    root = prior.capacity._new_external_path(config.output)
    frozen_sources = source_identity()
    _, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    binding = bind_prior(config.prior_run, config.prior_record, dataset, deep=True)
    if source_identity() != frozen_sources:
        raise ValueError("control sources changed during prior-evidence binding")
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _run_in_directory(config, root, dataset, binding, frozen_sources)
    except BaseException as error:
        if not (root / "failure.json").exists():
            pilot.write_json(root / "failure.json", {"complete": False, "type": type(error).__name__, "error": str(error)})
        if not (root / "artifact_manifest.json").exists():
            _seal(root)
        raise


def publish_record(path, root, result):
    target = prior.capacity._new_external_path(path)
    root = Path(root).resolve(strict=True)
    if target == root or root in target.parents:
        raise ValueError("public record must be outside the immutable raw output")
    original_seal = (root / "artifact_manifest.json").read_bytes()
    manifest = _read(root / "manifest.json")
    if manifest["run_directory"] != str(root):
        raise ValueError("publication directory does not own the manifest")
    _frozen(manifest)
    _verify_execution(root, manifest)
    cell = verify_worker(root / "signsgd", manifest)
    expected = _parent_summary(manifest, [cell], [])
    _equal(result, expected, "complete one-cell publication scope and comparison")
    _equal(_read(root / "summary.json"), expected, "actual raw parent summary")
    artifacts = _read(root / "artifact_manifest.json")
    _equal(artifacts, _artifacts(root), "complete actual raw artifact tree")
    if any("failure" in Path(row["file"]).name for row in artifacts["files"]):
        raise ValueError("failure artifacts prevent successful publication")
    compact = {key: value for key, value in cell.items() if key != "evaluations"}
    compact["evaluations"] = [{key: value for key, value in row.items() if key != "per_window"}
                              for row in cell["evaluations"]]
    value = {**result, "cells": [compact], "run_directory": str(root),
             "artifact_manifest_sha256": pilot.digest(original_seal), "artifact_files_verified": len(artifacts["files"]),
             "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
             "full_weight_checkpoints_verified": len(CHECKPOINTS),
             "checkpoint_validation_windows": len(CHECKPOINTS) * len(VALIDATION),
             "distinct_validation_positions": len(VALIDATION) * 128,
             "new_native_updates": STEPS, "new_gradient_pair_exposures": STEPS * 128,
             "inherited_host_fitting_pair_exposures": STEPS * 128, "new_host_fitting_pair_exposures": 0,
             "initialization_refitted": False, "model_history_pair_exposures": 2 * STEPS * 128,
             "distinct_training_pairs": STEPS * 128,
             "training_steps_guarded_and_compared_by_worker": STEPS,
             "posthoc_complete_training_trajectory_replayed": False}
    _frozen(manifest)
    if (root / "artifact_manifest.json").read_bytes() != original_seal:
        raise ValueError("artifact manifest changed during publication")
    _equal(artifacts, _artifacts(root), "payload tree unchanged throughout publication")
    prior.capacity._new_external_path(target)
    pilot.write_json(target, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--prior-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent", type=Path, help=argparse.SUPPRESS)
    config = parser.parse_args(argv)
    if config.worker:
        if config.parent is None or config.record is not None:
            parser.error("worker requires its exact parent and no publication path")
        result = run_worker(config)
    else:
        if config.parent is not None:
            parser.error("parent is internal to worker execution")
        output = prior.capacity._new_external_path(config.output)
        if config.record is not None:
            target = prior.capacity._new_external_path(config.record)
            if target == output or output in target.parents:
                parser.error("record must be outside the raw output")
        result = run(config)
        if config.record is not None and result["complete"] is True:
            publish_record(config.record, config.output, result)
    print(json.dumps({"complete": result["complete"], "output": str(config.output)}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
