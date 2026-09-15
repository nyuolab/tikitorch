"""Fixed MP-route repair study with three new runs and three historical controls.

Only diverse cells train. Historical cells import completed confirmation states
and evaluate them on the same new development cohort. Shared inherited model
history is charged explicitly; it is not counted as newly executed fitting or
training. No numerical outcome authorizes hardware or establishes generalization.
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
from manuscript.tools import fpga_lm_diverse_projection as initializer  # noqa: E402
from manuscript.tools import fpga_lm_efsgd as optimizer  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from manuscript.tools.observe_fpga_lm_routes import observe  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_diverse_projection_study.v1"
ARMS, SEEDS = ("diverse", "historical"), (0, 1, 2)
WIDTH, ZSHIFT, K, STEPS = 512, 2, 18, 1024
CHECKPOINTS, VALIDATION = (0, 256, 1024), tuple(range(72, 88))
WORKER_TIMEOUT_S = 1200
PLAN = REPO / "manuscript/research/PLAN.md"
prior = confirmation.prior
_read, _equal = confirmation._read, confirmation._equal
_native_identity, _run_process = confirmation._native_identity, confirmation._run_process


def source_identity():
    result = confirmation.source_identity()
    result.update(initializer.source_identity())
    path = Path(__file__).resolve()
    result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def training_order(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("only the three declared order seeds are supported")
    order = confirmation.training_order(seed)
    if sorted(order) != list(range(STEPS)):
        raise ValueError("saved confirmation order must match the exact declared budget")
    return order


def recipe():
    return {"arms": list(ARMS), "seeds": list(SEEDS), "width": WIDTH, "zshift": ZSHIFT, "k": K,
            "steps": STEPS, "checkpoints": list(CHECKPOINTS), "validation_indices": list(VALIDATION),
            "training_orders": {str(seed): training_order(seed) for seed in SEEDS},
            "seed_meaning": "exact historical confirmation permutations; not weight initialization or data seeds",
            "initializer_schema": initializer.SCHEMA, "initializer": "MP-only A->B, B->C, C->D, D->B, all four layers; designated0/other-8192",
            "initialization_guarantee": "strict fitted source point only; exact logits/distribution/head/tap parity and mapped projection routes",
            "optimizer": "unchanged EF k18: a=r+G; q=trunc_toward_zero(a/2**18); w-=q; r=a-q*2**18",
            "declared_cells": 6, "new_training_cells": 3, "historical_evaluation_cells": 3,
            "new_updates_by_arm": {"diverse": STEPS, "historical": 0},
            "inherited_fitting_pairs_per_cell": STEPS * 128,
            "inherited_gradient_pairs_by_arm": {"diverse": 0, "historical": STEPS * 128},
            "new_gradient_pairs_by_arm": {"diverse": STEPS * 128, "historical": 0},
            "new_host_fitting_pair_exposures": 0, "initialization_refitted": False,
            "model_history_pairs_per_cell": 2 * STEPS * 128, "distinct_training_pairs": STEPS * 128,
            "baseline_policy": "alpha1; inherited full fit plus actual shuffled prefix; report one-pass/replay and use stronger final comparator",
            "state_policy": "complete native optimizer state, import/evaluation invariance; historical bytes unchanged at every checkpoint",
            "gate_policy": "same frozen per-arm three-seed mechanism/context/bigram gates; only after all six cells complete",
            "worker_timeout_s": WORKER_TIMEOUT_S,
            "scope": "CPU development72..87 only; no adaptive retries, final test, paid authorization, significance, speed or physical FPGA claim"}


def _baseline(raw, dataset, indices):
    uni, bi = [1] * 128, [[1] * 128 for _ in range(128)]
    for index in indices:
        tokens, targets = pilot.window(raw, dataset, "train", index)
        for token, target in zip(tokens, targets):
            uni[target] += 1
            bi[token][target] += 1
    result = {"supervised_exposures": len(indices) * 128, "distinct_supervised_pairs": len(set(indices)) * 128,
              "alpha": 1, "vocabulary": 128, "exposure_indices": list(indices), "validation_indices": list(VALIDATION)}
    for name, counts in (("unigram", [uni] * 128), ("bigram", bi)):
        totals = list(map(sum, counts))
        predictions = [max(range(128), key=row.__getitem__) for row in counts]
        rows = []
        for index in VALIDATION:
            tokens, targets = pilot.window(raw, dataset, "validation", index)
            losses = [math.log(totals[t]) - math.log(counts[t][y]) for t, y in zip(tokens, targets)]
            rows.append({"window_index": index, "nll": sum(losses) / len(losses),
                         "accuracy": sum(predictions[t] == y for t, y in zip(tokens, targets)) / len(targets)})
        result[name] = {"nll": sum(row["nll"] for row in rows) / len(rows),
                        "accuracy": sum(row["accuracy"] for row in rows) / len(rows), "per_window": rows}
    return result


def exposure_baseline(raw, dataset, seed, step):
    if type(step) is not int or step not in CHECKPOINTS:
        raise ValueError("undeclared checkpoint")
    return _baseline(raw, dataset, list(range(STEPS)) + training_order(seed)[:step])


def one_pass_baseline(raw, dataset):
    return _baseline(raw, dataset, list(range(STEPS)))


def _artifacts(root):
    value = prior._artifacts(Path(root))
    value["schema"] = SCHEMA
    return value


def _seal(root):
    pilot.write_json(root / "artifact_manifest.json", _artifacts(root))


def confirmation_binding(run, record, dataset, deep=True):
    """A failed quality gate is valid history; incomplete execution is not."""
    if type(deep) is not bool:
        raise ValueError("literal verification depth required")
    root, path = Path(run).resolve(strict=True), Path(record).resolve(strict=True)
    if root == path or root in path.parents:
        raise ValueError("historical public record must be outside its raw run")
    names = ("manifest.json", "summary.json", "artifact_manifest.json")
    original = {name: (root / name).read_bytes() for name in names}
    record_bytes = path.read_bytes()
    manifest, summary, artifacts = (_read(root / name) for name in names)
    receipt = _read(path)
    confirmation._frozen(manifest)
    _equal(manifest["dataset"], dataset, "historical corpus/split identity")
    _equal(artifacts, confirmation._artifacts(root), "every historical raw payload")
    if any("failure" in Path(row["file"]).name for row in artifacts["files"]):
        raise ValueError("failed execution artifact cannot initialize this study")
    expected_scope = {"schema": confirmation.SCHEMA, "complete": True, "declared_cells": 3, "completed_cells": 3,
        "failures": [], "heldout_test_evaluated": False, "physical_fpga_execution": False,
        "speed_advantage_claim": False, "statistical_significance_claim": False, "paid_execution_authorized": False}
    _equal({name: summary.get(name) for name in expected_scope}, expected_scope, "completed confirmation scope")
    _equal(summary["recipe"], confirmation.recipe(), "historical recipe")
    _equal(summary["source_hashes"], manifest["source_hashes"], "historical source bindings")
    _equal(summary["prior_evidence"], manifest["prior_evidence"], "original fitted-state ancestry")
    _equal([cell["seed"] for cell in summary["cells"]], list(SEEDS), "all historical seed cells")
    states = {}
    for cell in summary["cells"]:
        seed = cell["seed"]
        directory = root / ("seed_%d" % seed)
        confirmation._verify_execution(root, manifest, seed)
        _equal(cell, _read(directory / "summary.json"), "historical cell summary")
        if deep:
            _equal(cell, confirmation.verify_worker(directory, manifest, seed), "complete historical cell audit")
        _equal(cell["training_order"], training_order(seed), "actual saved permutation")
        states[str(seed)] = {}
        for step in CHECKPOINTS:
            state_path = directory / ("state_%06d.i32" % step)
            payload = state_path.read_bytes()
            states[str(seed)][str(step)] = {"file": str(state_path), "bytes": len(payload), "sha256": pilot.digest(payload)}
    _equal(summary["gates"], confirmation.gate_results(summary["cells"]), "historical gates, whether pass or fail")
    expected_receipt = {**summary, "cells": [confirmation._compact_cell(cell) for cell in summary["cells"]],
        "run_directory": str(root), "artifact_manifest_sha256": pilot.digest(original["artifact_manifest.json"]),
        "artifact_files_verified": len(artifacts["files"]), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
        "full_optimizer_checkpoints_verified": 9, "checkpoint_validation_windows": 144, "distinct_validation_positions": 2048,
        "new_native_updates": 3072, "gradient_pair_exposures": 393216, "inherited_host_fitting_pair_exposures": 393216,
        "new_host_fitting_pair_exposures": 0, "initialization_refitted": False, "model_history_pair_exposures": 786432,
        "inherited_exposure_accounting": "sum of per-seed histories; the same old fit is shared, not newly repeated",
        "distinct_training_pairs": 131072, "native_training_trajectory_independently_replayed": False}
    _equal(receipt, expected_receipt, "exact completed public confirmation receipt")
    binding = {"run_directory": str(root), "record": {"file": str(path), "sha256": pilot.digest(record_bytes)},
        "manifest_sha256": pilot.digest(original["manifest.json"]), "summary_sha256": pilot.digest(original["summary.json"]),
        "artifact_manifest_sha256": pilot.digest(original["artifact_manifest.json"]),
        "prior_evidence": manifest["prior_evidence"], "states": states}
    shape = pilot.layout(WIDTH)
    for seed in SEEDS:
        for step in CHECKPOINTS:
            read_confirmation_state(binding, seed, step, shape)
    confirmation._frozen(manifest)
    if path.read_bytes() != record_bytes or any((root / name).read_bytes() != original[name] for name in names):
        raise ValueError("historical metadata changed during binding")
    _equal(artifacts, confirmation._artifacts(root), "historical bytes unchanged through binding")
    return binding


def read_confirmation_state(binding, seed, step, shape):
    training_order(seed)
    if type(step) is not int or step not in CHECKPOINTS:
        raise ValueError("undeclared historical checkpoint")
    root = Path(binding["run_directory"]).resolve(strict=True)
    path = root / ("seed_%d" % seed) / ("state_%06d.i32" % step)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("historical state must be an exact regular owned file")
    payload = path.read_bytes()
    _equal(binding["states"][str(seed)][str(step)],
           {"file": str(path), "bytes": len(payload), "sha256": pilot.digest(payload)}, "actual full historical state")
    state = optimizer.decode_state(payload, shape)
    if (state["step"] != step or state["k"] != K or state["config"]["loss_kind"] != 1 or state["config"]["zshift"] != ZSHIFT):
        raise ValueError("historical state config/source/count mismatch")
    if step == 0 and payload != confirmation.read_initial_state(binding["prior_evidence"], shape):
        raise ValueError("historical initialization differs from exact original fitted full state")
    return payload


def initial_state(binding, arm, seed, shape):
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    payload = read_confirmation_state(binding, seed, 0, shape)
    state = optimizer.decode_state(payload, shape)
    changed, facts = initializer.transform(state["weights"], shape)
    if arm == "historical":
        return payload, facts
    values = optimizer._unpack(payload)
    offset = optimizer.STATE_HEADER + shape.WORDS_CE_LUT
    for name in pilot.WEIGHTS:
        count = getattr(shape, "WORDS_" + name)
        values[offset:offset + count] = changed[name]
        offset += count
    changed_payload = optimizer._pack(values)
    decoded = optimizer.decode_state(changed_payload, shape)
    _equal(decoded["remainders"], state["remainders"], "unchanged zero optimizer initialization")
    return changed_payload, facts


def initial_buffer(binding, arm, seed, shape):
    payload, facts = initial_state(binding, arm, seed, shape)
    state = optimizer.decode_state(payload, shape)
    buffer = shape.build_init(0)
    for name, values in state["weights"].items():
        offset = getattr(shape, "OFF_" + name)
        buffer[offset:offset + len(values)] = values
    buffer[shape.OFF_CFG:shape.OFF_CFG + 2] = [1, ZSHIFT]
    buffer[shape.OFF_CE_LUT:shape.OFF_CE_LUT + shape.WORDS_CE_LUT] = state["config"]["lut"]
    return buffer, payload, facts


def _frozen(manifest):
    _equal(manifest["schema"], SCHEMA, "study schema")
    _equal(manifest["recipe"], recipe(), "fixed declared protocol")
    _equal(manifest["source_hashes"], source_identity(), "frozen source dependencies")
    root = Path(manifest["run_directory"]).resolve(strict=True)
    executable = Path(manifest["executable"]["path"])
    if executable.parent != root / "build" or executable.is_symlink():
        raise ValueError("native executable must be the exact owned build")
    if pilot.digest(executable.read_bytes()) != manifest["executable"]["sha256"]:
        raise ValueError("native executable changed")
    if pilot.digest((root / "build/build.json").read_bytes()) != manifest["build_record_sha256"]:
        raise ValueError("native build receipt changed")
    _native_identity(manifest, pilot.layout(WIDTH))
    _equal(_read(root / "protocol.json"), {"schema": SCHEMA, **recipe()}, "pre-execution declaration")
    binding = manifest["confirmation_evidence"]
    _equal(binding, confirmation_binding(binding["run_directory"], binding["record"]["file"],
                                         manifest["dataset"], deep=False), "unchanged complete historical evidence")


def _checked_observer(weights, tokens, targets, shape):
    result = observe(weights, tokens, targets, shape, zshift=ZSHIFT)
    if (type(result["saturation_events"]) is not int or result["saturation_events"] != 0
            or result["route_diagnostics"]["backward_int32_overflow_detected"] is not False):
        raise ValueError("observed forward clipping or backward overflow")
    return result


def _probe(weights, tokens, targets, shape):
    result = pilot.integer_forward(weights, tokens, targets, shape, zshift=ZSHIFT, use_context=False)
    if type(result["saturation_events"]) is not int or result["saturation_events"] != 0:
        raise ValueError("context-disabled probe clips")
    return screen._scalar_result(result)


def _initial_equivalence(binding, seed, tokens, targets, shape):
    old = optimizer.decode_state(read_confirmation_state(binding, seed, 0, shape), shape)["weights"]
    changed, _ = initializer.transform(old, shape)
    return initializer.check_initial_equivalence(old, changed, tokens, targets, shape)


def _state_config(payload, shape, step):
    state = optimizer.decode_state(payload, shape)
    if (state["step"] != step or state["k"] != K or state["config"]["loss_kind"] != 1
            or state["config"]["zshift"] != ZSHIFT):
        raise ValueError("complete state step/config/source mismatch")
    return state


def evaluate_cell(native, root, step, raw, dataset, shape, binding, seed):
    payload = native.export_state()
    state = _state_config(payload, shape, step)
    checkpoint = native.save_checkpoint(root, "state_%06d" % step)
    rows = []
    for index in VALIDATION:
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        actual = native.evaluate(tokens, targets)
        if type(actual["step"]) is not int or actual["step"] != step:
            raise ValueError("evaluation advanced optimizer state")
        expected = _checked_observer(state["weights"], tokens, targets, shape)
        arrays = {}
        for name, values in {"predictions": expected["predictions"], "gradients": expected["gradients"],
                             **expected["native_routes"]}.items():
            found = actual[name] if name in ("predictions", "gradients") else actual["native_routes"][name]
            arrays[name] = screen._save_array(root, "eval_%06d_%03d_%s.i32" % (step, index, name), found)
            _equal(found, values, "native evaluation arrays and routes")
        row = {**screen._scalar_result(expected), "window_index": index, "native_arrays": arrays,
               "native_reference_equal": True, "native_routes_equal": True,
               "probes": {"context_disabled": _probe(state["weights"], tokens, targets, shape)},
               "initial_equivalence": _initial_equivalence(binding, seed, tokens, targets, shape) if step == 0 else None}
        pilot.write_json(root / ("eval_%06d_%03d.json" % (step, index)), row)
        rows.append(row)
    if native.export_state() != payload:
        raise ValueError("validation modified complete optimizer state")
    native.import_state(payload)
    if native.export_state() != payload:
        raise ValueError("checkpoint import is not byte-invariant")
    return {"step": step, "checkpoint": checkpoint, "state_sha256": pilot.digest(payload),
            "full_state_evaluation_invariant": True, "full_state_import_equal": True, **screen._aggregate(rows)}


def _cell_summary(arm, seed, evaluations, baselines, one_pass, facts):
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    order = training_order(seed)
    new_steps = STEPS if arm == "diverse" else 0
    inherited_steps = 0 if arm == "diverse" else STEPS
    comparators = confirmation._comparators(baselines, one_pass)
    first, final = evaluations[0], evaluations[-1]
    context = final["context_disabled_nll"] - final["lut_normalized_nll"]
    return {"schema": SCHEMA, "complete": True, "arm": arm, "seed": seed, "k": K, "zshift": ZSHIFT,
        "seed_meaning": "saved confirmation training permutation only", "training_order": order,
        "training_performed": arm == "diverse", "training_updates": new_steps,
        "new_gradient_pair_exposures": new_steps * 128, "inherited_gradient_pair_exposures": inherited_steps * 128,
        "inherited_host_fitting_pair_exposures": STEPS * 128, "new_host_fitting_pair_exposures": 0,
        "initialization_refitted": False, "model_history_pair_exposures": STEPS * 256,
        "distinct_training_pairs": STEPS * 128, "initial_state_equal_declared_arm": True,
        "historical_full_states_equal": arm == "historical",
        "initializer": {"applied": arm == "diverse", "proposal": facts},
        "evaluations": evaluations, "baselines_by_step": baselines, "one_pass_baseline": one_pass,
        "conservative_comparators": comparators,
        "conservative_bigram_advantage_nats": comparators["bigram"]["nll"] - final["lut_normalized_nll"],
        "context_advantage_nats": context,
        "context_advantage_growth_nats": context - (first["context_disabled_nll"] - first["lut_normalized_nll"]),
        "heldout_test_evaluated": False, "physical_fpga_execution": False, "speed_advantage_claim": False,
        "paid_execution_authorized": False, "native_training_trajectory_independently_replayed": False}


def run_worker(config):
    arm, seed = config.arm, config.seed
    order = training_order(seed)
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    parent, root = Path(config.parent).resolve(strict=True), Path(config.output).resolve()
    if root != parent / ("%s_seed_%d" % (arm, seed)):
        raise ValueError("worker must own exactly its declared child directory")
    manifest = _read(parent / "manifest.json")
    if manifest["run_directory"] != str(parent):
        raise ValueError("worker parent does not own manifest")
    _frozen(manifest)
    raw, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    _equal(dataset, manifest["dataset"], "worker corpus and splits")
    shape, binding = pilot.layout(WIDTH), manifest["confirmation_evidence"]
    buffer, payload, facts = initial_buffer(binding, arm, seed, shape)
    root.mkdir(parents=True, exist_ok=False)
    completed = 0
    try:
        pilot.write_json(root / "manifest.json", {**manifest, "arm": arm, "seed": seed,
            "training_order": order, "initial_buffer_sha256": pilot.digest(screen._i32(buffer)),
            "initial_state_sha256": pilot.digest(payload),
            "initializer": {"applied": arm == "diverse", "proposal": facts}})
        evaluations = []
        with optimizer.EFSGD(Path(manifest["executable"]["path"]), shape, buffer, K,
                             timeout_s=60, log_path=root / "native.log") as native:
            if native.export_state() != payload:
                raise ValueError("actual native initialization differs byte-for-byte")
            with (root / "training.jsonl").open("x") as log, (root / "training_predictions.i32").open("xb") as predlog, \
                    (root / "training_gradients.i32").open("xb") as gradlog:
                if arm == "historical":
                    for step in CHECKPOINTS:
                        previous = read_confirmation_state(binding, seed, step, shape)
                        native.import_state(previous)
                        if native.export_state() != previous:
                            raise ValueError("historical full-state import differs")
                        evaluations.append(evaluate_cell(native, root, step, raw, dataset, shape, binding, seed))
                        if native.export_state() != previous:
                            raise ValueError("historical full-state evaluation changed bytes")
                else:
                    evaluations.append(evaluate_cell(native, root, 0, raw, dataset, shape, binding, seed))
                    for index in order:
                        tokens, targets = pilot.window(raw, dataset, "train", index)
                        result = native.train(tokens, targets)
                        if type(result["step"]) is not int or result["step"] != completed + 1:
                            raise ValueError("actual new update count differs")
                        pred, grad = screen._i32(result["predictions"]), screen._i32(result["gradients"])
                        if len(pred) != 128 * 4 or len(grad) != 16384 * 4:
                            raise ValueError("incomplete native training arrays")
                        predlog.write(pred)
                        gradlog.write(grad)
                        completed += 1
                        log.write(json.dumps({"step": completed, "window_index": index,
                            "predictions_sha256": pilot.digest(pred), "gradients_sha256": pilot.digest(grad)}) + "\n")
                        log.flush()
                        if completed in CHECKPOINTS:
                            evaluations.append(evaluate_cell(native, root, completed, raw, dataset, shape, binding, seed))
        baselines = {str(step): exposure_baseline(raw, dataset, seed, step) for step in CHECKPOINTS}
        result = _cell_summary(arm, seed, evaluations, baselines, one_pass_baseline(raw, dataset), facts)
        _frozen(manifest)
        pilot.write_json(root / "summary.json", result)
        _seal(root)
        return result
    except BaseException as error:
        pilot.write_json(root / "failure.json", {"complete": False, "new_updates_completed": completed,
            "type": type(error).__name__, "error": str(error)})
        _seal(root)
        raise


def _verify_evaluation(root, evaluation, step, shape, identity, raw, dataset, binding, seed):
    path = root / ("state_%06d.i32" % step)
    payload = path.read_bytes()
    state = _state_config(payload, shape, step)
    expected_checkpoint = {"schema": optimizer.SCHEMA, "file": path.name, "bytes": len(payload),
        "sha256": pilot.digest(payload), "step": step, "k": K, "config": state["config"], "identity": identity,
        "regions": [{"name": name, "weight_words": getattr(shape, "WORDS_" + name),
                     "remainder_words": getattr(shape, "WORDS_" + name)} for name in pilot.WEIGHTS],
        "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}
    _equal(evaluation["checkpoint"], expected_checkpoint, "complete optimizer checkpoint metadata")
    _equal(_read(path.with_suffix(".json")), expected_checkpoint, "actual optimizer checkpoint receipt")
    rows = []
    for index in VALIDATION:
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        expected = _checked_observer(state["weights"], tokens, targets, shape)
        arrays = {}
        for name, values in {"predictions": expected["predictions"], "gradients": expected["gradients"],
                             **expected["native_routes"]}.items():
            filename = "eval_%06d_%03d_%s.i32" % (step, index, name)
            encoded = screen._i32(values)
            if (root / filename).read_bytes() != encoded:
                raise ValueError("native checkpoint arrays differ from independent observer")
            arrays[name] = {"file": filename, "bytes": len(encoded), "words": len(values),
                           "sha256": pilot.digest(encoded), "format": "little-endian signed int32"}
        row = {**screen._scalar_result(expected), "window_index": index, "native_arrays": arrays,
               "native_reference_equal": True, "native_routes_equal": True,
               "probes": {"context_disabled": _probe(state["weights"], tokens, targets, shape)},
               "initial_equivalence": _initial_equivalence(binding, seed, tokens, targets, shape) if step == 0 else None}
        _equal(_read(root / ("eval_%06d_%03d.json" % (step, index))), row, "actual checkpoint observation")
        rows.append(row)
    _equal(evaluation, {"step": step, "checkpoint": expected_checkpoint, "state_sha256": pilot.digest(payload),
        "full_state_evaluation_invariant": True, "full_state_import_equal": True, **screen._aggregate(rows)},
        "all checkpoint fields and aggregate")
    return payload


def verify_worker(root, manifest, arm, seed):
    order = training_order(seed)
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    root = Path(root).resolve(strict=True)
    if root != Path(manifest["run_directory"]) / ("%s_seed_%d" % (arm, seed)):
        raise ValueError("unexpected worker directory")
    _frozen(manifest)
    seal = (root / "artifact_manifest.json").read_bytes()
    artifacts = _read(root / "artifact_manifest.json")
    _equal(artifacts, _artifacts(root), "actual complete worker artifact tree")
    if any("failure" in Path(row["file"]).name for row in artifacts["files"]):
        raise ValueError("failed worker cannot be complete evidence")
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], STEPS, max(VALIDATION) + 1)
    _equal(dataset, manifest["dataset"], "actual corpus/split declaration")
    shape, binding = pilot.layout(WIDTH), manifest["confirmation_evidence"]
    buffer, initial, facts = initial_buffer(binding, arm, seed, shape)
    _equal(_read(root / "manifest.json"), {**manifest, "arm": arm, "seed": seed, "training_order": order,
        "initial_buffer_sha256": pilot.digest(screen._i32(buffer)), "initial_state_sha256": pilot.digest(initial),
        "initializer": {"applied": arm == "diverse", "proposal": facts}}, "exact worker initialization and declaration")
    result = _read(root / "summary.json")
    evaluations = result["evaluations"]
    _equal([row["step"] for row in evaluations], list(CHECKPOINTS), "all declared checkpoints")
    identity = _native_identity(manifest, shape)
    for row in evaluations:
        payload = _verify_evaluation(root, row, row["step"], shape, identity, raw, dataset, binding, seed)
        if row["step"] == 0 and payload != initial:
            raise ValueError("actual initial complete state is not the declared transform")
        if arm == "historical" and payload != read_confirmation_state(binding, seed, row["step"], shape):
            raise ValueError("historical checkpoint differs from the actual saved prior bytes")
    baselines = {str(step): exposure_baseline(raw, dataset, seed, step) for step in CHECKPOINTS}
    _equal(result, _cell_summary(arm, seed, evaluations, baselines, one_pass_baseline(raw, dataset), facts),
           "complete worker scope and exposure accounting")
    lines = (root / "training.jsonl").read_text().splitlines()
    updates = STEPS if arm == "diverse" else 0
    if len(lines) != updates:
        raise ValueError("new update coverage differs from declared arm")
    arrays = {name: (root / ("training_%s.i32" % name)).read_bytes() for name in ("predictions", "gradients")}
    sizes = {"predictions": 128, "gradients": 16384}
    if any(len(arrays[name]) != updates * count * 4 for name, count in sizes.items()):
        raise ValueError("actual training array coverage differs")
    for index, line in enumerate(lines):
        expected = {"step": index + 1, "window_index": order[index]}
        for name, count in sizes.items():
            expected[name + "_sha256"] = pilot.digest(arrays[name][index * count * 4:(index + 1) * count * 4])
        _equal(pilot._finite_json(line), expected, "actual new training arrays and saved order")
    _frozen(manifest)
    if (root / "artifact_manifest.json").read_bytes() != seal:
        raise ValueError("worker seal changed during audit")
    _equal(artifacts, _artifacts(root), "worker bytes unchanged during audit")
    return result


def gate_results(cells):
    expected = [[arm, seed] for arm in ARMS for seed in SEEDS]
    _equal([[row["arm"], row["seed"]] for row in cells], expected, "all six cells before gates")
    if any(row.get("complete") is not True for row in cells):
        raise ValueError("all six cells must be complete before either arm is assessed")
    return {arm: confirmation.gate_results([cell for cell in cells if cell["arm"] == arm]) for arm in ARMS}


def _verify_paired_initializations(root):
    """Compare actual independently saved native bytes, not just equality flags."""
    shape = pilot.layout(WIDTH)
    words = {"predictions": 128, "gradients": 16384,
             "AD_HEAD": shape.WORDS_AD_HEAD, "TAP_IDX": shape.WORDS_TAP_IDX}
    for seed in SEEDS:
        old = root / ("historical_seed_%d" % seed)
        new = root / ("diverse_seed_%d" % seed)
        for index in VALIDATION:
            for name in ("predictions", "gradients", "AD_HEAD", "TAP_IDX"):
                filename = "eval_000000_%03d_%s.i32" % (index, name)
                left, right = (old / filename).read_bytes(), (new / filename).read_bytes()
                if len(left) != words[name] * 4 or left != right:
                    raise ValueError("actual paired initial native predictions/gradients/head/tap bytes differ")


def _expected_command(root, manifest, arm, seed):
    training_order(seed)
    if arm not in ARMS:
        raise ValueError("undeclared arm")
    return [sys.executable, str(Path(__file__).resolve()), "--worker", "--corpus", manifest["dataset"]["path"],
        "--confirmation-run", manifest["confirmation_evidence"]["run_directory"],
        "--confirmation-record", manifest["confirmation_evidence"]["record"]["file"],
        "--output", str(root / ("%s_seed_%d" % (arm, seed))), "--parent", str(root), "--arm", arm, "--seed", str(seed)]


def _verify_execution(root, manifest, arm, seed):
    row = _read(root / ("%s_seed_%d.execution.json" % (arm, seed)))
    if (type(row.get("returncode")) is not int or row["returncode"] != 0 or row.get("timed_out") is not False
            or type(row.get("timeout_s")) is not int or row["timeout_s"] != WORKER_TIMEOUT_S
            or type(row.get("elapsed_s")) not in (int, float) or not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < 0):
        raise ValueError("worker execution was not successful, bounded and finite")
    _equal(row, {"command": _expected_command(root, manifest, arm, seed), "returncode": 0,
        "timed_out": False, "timeout_s": WORKER_TIMEOUT_S, "elapsed_s": row["elapsed_s"]}, "exact executed worker command")


def _parent_summary(manifest, cells, failures):
    complete = not failures and len(cells) == len(ARMS) * len(SEEDS)
    return {"schema": SCHEMA, "complete": complete, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
        "confirmation_evidence": manifest["confirmation_evidence"], "declared_cells": 6, "completed_cells": len(cells),
        "cells": cells, "failures": failures, "gates": gate_results(cells) if complete else None,
        "paired_initial_native_outputs_equal": True if complete else None,
        "heldout_test_evaluated": False, "physical_fpga_execution": False, "speed_advantage_claim": False,
        "statistical_significance_claim": False, "paid_execution_authorized": False,
        "limitations": ["Three order replicates share initialization/data; historical controls perform no new training.",
            "MP initialization preserves the initial function only; residual dynamics may change after updates.",
            "This tests one route-geometry repair, not proof that cancellation is the unique causal bottleneck.",
            "Historical host fitting and historical control updates are inherited, not newly executed exposures.",
            "New development cohort is not final test; frozen numerical gates never authorize hardware.",
            "EF adapter requires new synthesis and physical verification; this study is CPU-only.",
            "Training arrays are hash/coverage checked, not an independent replay of every native update."]}


def _run_in_directory(config, root, dataset, binding, sources):
    pilot.write_json(root / "protocol.json", {"schema": SCHEMA, **recipe()})
    _equal(source_identity(), sources, "sources unchanged before native build")
    executable = optimizer.build(root / "build", width=WIDTH)
    _equal(source_identity(), sources, "sources unchanged across native build")
    manifest = {"schema": SCHEMA, "recipe": recipe(), "run_directory": str(root), "dataset": dataset,
        "confirmation_evidence": binding, "source_hashes": sources, "plan_sha256": pilot.digest(PLAN.read_bytes()),
        "executable": {"path": str(executable.resolve()), "sha256": pilot.digest(executable.read_bytes())},
        "build_record_sha256": pilot.digest((root / "build/build.json").read_bytes())}
    pilot.write_json(root / "manifest.json", manifest)
    cells, failures = [], []
    for arm in ARMS:
        for seed in SEEDS:
            _frozen(manifest)
            name = "%s_seed_%d" % (arm, seed)
            command = _expected_command(root, manifest, arm, seed)
            try:
                execution = _run_process(command, root / (name + ".log"))
                pilot.write_json(root / (name + ".execution.json"), {"command": command, **execution})
                _verify_execution(root, manifest, arm, seed)
                cells.append(verify_worker(root / name, manifest, arm, seed))
            except Exception as error:
                failure = {"arm": arm, "seed": seed, "type": type(error).__name__, "error": str(error)}
                pilot.write_json(root / (name + ".failure.json"), failure)
                failures.append(failure)
    _frozen(manifest)
    if not failures and len(cells) == len(ARMS) * len(SEEDS):
        _verify_paired_initializations(root)
    result = _parent_summary(manifest, cells, failures)
    pilot.write_json(root / "summary.json", result)
    _seal(root)
    return result


def run(config):
    root = prior.capacity._new_external_path(config.output)
    sources = source_identity()
    _, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    binding = confirmation_binding(config.confirmation_run, config.confirmation_record, dataset, deep=True)
    _equal(source_identity(), sources, "source identity across full history verification")
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _run_in_directory(config, root, dataset, binding, sources)
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
        raise ValueError("record must be outside the immutable raw run")
    seal = (root / "artifact_manifest.json").read_bytes()
    manifest = _read(root / "manifest.json")
    if manifest["run_directory"] != str(root):
        raise ValueError("raw output does not own manifest")
    _frozen(manifest)
    cells = []
    for arm in ARMS:
        for seed in SEEDS:
            _verify_execution(root, manifest, arm, seed)
            cells.append(verify_worker(root / ("%s_seed_%d" % (arm, seed)), manifest, arm, seed))
    _verify_paired_initializations(root)
    expected = _parent_summary(manifest, cells, [])
    _equal(result, expected, "complete six-cell publication scope")
    _equal(_read(root / "summary.json"), expected, "actual parent result")
    artifacts = _read(root / "artifact_manifest.json")
    _equal(artifacts, _artifacts(root), "every actual raw artifact")
    if any("failure" in Path(row["file"]).name for row in artifacts["files"]):
        raise ValueError("failure artifacts prevent successful publication")
    value = {**result, "cells": [confirmation._compact_cell(cell) for cell in cells], "run_directory": str(root),
        "artifact_manifest_sha256": pilot.digest(seal), "artifact_files_verified": len(artifacts["files"]),
        "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
        "full_optimizer_checkpoints_verified": 6 * len(CHECKPOINTS),
        "checkpoint_validation_windows": 6 * len(CHECKPOINTS) * len(VALIDATION),
        "distinct_validation_positions": len(VALIDATION) * 128,
        "new_native_updates": len(SEEDS) * STEPS, "new_gradient_pair_exposures": len(SEEDS) * STEPS * 128,
        "inherited_gradient_pair_exposures": len(SEEDS) * STEPS * 128,
        "inherited_host_fitting_pair_exposures": 6 * STEPS * 128,
        "new_host_fitting_pair_exposures": 0, "initialization_refitted": False,
        "model_history_pair_exposures": 6 * STEPS * 256, "distinct_training_pairs": STEPS * 128,
        "inherited_exposure_accounting": "sum of per-cell histories; shared old fitting/control updates were not reexecuted",
        "native_training_trajectory_independently_replayed": False}
    _frozen(manifest)
    if (root / "artifact_manifest.json").read_bytes() != seal:
        raise ValueError("raw seal changed during publication")
    _equal(artifacts, _artifacts(root), "actual raw bytes unchanged across publication")
    pilot.write_json(target, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "confirmation-run", "confirmation-record", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    config = parser.parse_args(argv)
    if config.worker:
        if config.parent is None or config.arm is None or config.seed is None or config.record is not None:
            parser.error("worker requires exact parent/arm/seed and no public record")
        result = run_worker(config)
    else:
        if any(getattr(config, name) is not None for name in ("parent", "arm", "seed")):
            parser.error("arm/order overrides are internal worker arguments")
        output = prior.capacity._new_external_path(config.output)
        if config.record is not None:
            target = prior.capacity._new_external_path(config.record)
            if target == output or output in target.parents:
                parser.error("record must be outside raw output")
        result = run(config)
        if config.record is not None and result["complete"] is True:
            publish_record(config.record, config.output, result)
    print(json.dumps({"complete": result["complete"], "output": str(config.output)}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
