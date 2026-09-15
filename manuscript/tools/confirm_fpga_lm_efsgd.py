"""Fixed shuffled-order confirmation of the count-initialized CPU EFSGD recipe.

Seeds select Python Random shuffles, NOT weight initializations or independent
data samples. All cells start from the exact prior count-fitted/k18 state, learn
the same 1024 windows, and evaluate new development windows56..71. This cannot
authorize cloud resources or establish physical FPGA/throughput equivalence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import train_fpga_lm_efsgd as prior  # noqa: E402
from manuscript.tools import fpga_lm_efsgd as optimizer  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from manuscript.tools.observe_fpga_lm_routes import observe  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_efsgd_confirmation.v1"
SEEDS = (0, 1, 2)
WIDTH, ZSHIFT, K, STEPS = 512, 2, 18, 1024
CHECKPOINTS = (0, 256, 1024)
VALIDATION = tuple(range(56, 72))
WORKER_TIMEOUT_S = 1200
PLAN = REPO / "manuscript/research/PLAN.md"
_read = prior._read
_run_process = prior._run_process
_native_identity = prior._native_identity
_equal = prior.audit._assert_equal


def source_identity():
    result = prior.source_identity()
    path = Path(__file__).resolve()
    result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def training_order(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed must be one of the three declared order seeds")
    result = list(range(STEPS))
    random.Random(seed).shuffle(result)
    return result


def recipe():
    return {"width": WIDTH, "zshift": ZSHIFT, "k": K, "steps": STEPS, "seeds": list(SEEDS),
            "seed_meaning": "random.Random(seed).shuffle of training-window order only; identical weights/data",
            "training_orders": {str(seed): training_order(seed) for seed in SEEDS},
            "checkpoints": list(CHECKPOINTS), "validation_indices": list(VALIDATION),
            "initialization": "byte-identical complete state0 from completed count_fitted_k18 prior cell",
            "optimizer": "a=r+G; q=trunc_toward_zero(a/2**18); w-=q; r=a-q*2**18",
            "fitting_indices": list(range(STEPS)), "inherited_host_fitting_pair_exposures_per_seed": STEPS * 128,
            "new_host_fitting_pair_exposures_per_seed": 0, "initialization_refitted": False,
            "model_history_pair_exposures_per_seed": 2 * STEPS * 128,
            "gradient_pair_exposures_per_seed": STEPS * 128, "distinct_training_pairs_per_seed": STEPS * 128,
            "baseline_policy": "alpha1; all inherited fitting windows plus actual shuffled prefix at each checkpoint; "
                               "report one-pass and replay controls and use the lower final NLL for each comparator",
            "state_policy": "full weights/remainders/step/config/source; forward-only evaluation and import are byte-invariant",
            "safety_policy": "frozen nonrecovering UBSan adapter; reject observed forward/update clipping",
            "worker_timeout_s": WORKER_TIMEOUT_S,
            "gate_policy": "unchanged all-seed mechanism/context and mean bigram thresholds; no paid authorization",
            "scope": "three order replicates on new development56..71; no final test, adaptive extension, "
                     "weight-seed independence, significance, hardware or speed claim"}


def _baseline(raw, dataset, indices):
    if not indices:
        raise ValueError("a baseline requires supervised exposures")
    unigram, bigram = [1] * 128, [[1] * 128 for _ in range(128)]
    for index in indices:
        tokens, targets = pilot.window(raw, dataset, "train", index)
        for token, target in zip(tokens, targets):
            unigram[target] += 1
            bigram[token][target] += 1
    result = {"supervised_exposures": len(indices) * 128, "distinct_supervised_pairs": len(set(indices)) * 128,
              "alpha": 1, "vocabulary": 128, "exposure_indices": list(indices),
              "validation_indices": list(VALIDATION)}
    for name, counts in (("unigram", [unigram] * 128), ("bigram", bigram)):
        totals = list(map(sum, counts))
        predictions = [max(range(128), key=lambda target: row[target]) for row in counts]
        rows = []
        for index in VALIDATION:
            tokens, targets = pilot.window(raw, dataset, "validation", index)
            losses = [math.log(totals[token]) - math.log(counts[token][target]) for token, target in zip(tokens, targets)]
            rows.append({"window_index": index, "nll": sum(losses) / len(losses),
                         "accuracy": sum(predictions[token] == target for token, target in zip(tokens, targets)) / len(targets)})
        result[name] = {"nll": sum(row["nll"] for row in rows) / len(rows),
                        "accuracy": sum(row["accuracy"] for row in rows) / len(rows), "per_window": rows}
    return result


def exposure_baseline(raw, dataset, seed, step):
    if type(step) is not int or step not in CHECKPOINTS:
        raise ValueError("undeclared checkpoint budget")
    return _baseline(raw, dataset, list(range(STEPS)) + training_order(seed)[:step])


def one_pass_baseline(raw, dataset):
    return _baseline(raw, dataset, list(range(STEPS)))


def _compact_cell(cell):
    result = {key: value for key, value in cell.items() if key != "evaluations"}
    result["evaluations"] = [{key: value for key, value in row.items() if key != "per_window"}
                             for row in cell["evaluations"]]
    return result


def bind_prior(run, record, dataset, deep=True):
    """Bind completed audited prior evidence; no execution may use a partial run."""
    if type(deep) is not bool:
        raise ValueError("prior verification depth must be literal boolean")
    root = Path(run).resolve(strict=True)
    record_path = Path(record).resolve(strict=True)
    if root == record_path or root in record_path.parents:
        raise ValueError("prior audit/public record must be outside its sealed raw run")
    manifest_bytes = (root / "manifest.json").read_bytes()
    artifact_bytes = (root / "artifact_manifest.json").read_bytes()
    summary_bytes = (root / "summary.json").read_bytes()
    record_bytes = record_path.read_bytes()
    manifest, summary, receipt = _read(root / "manifest.json"), _read(root / "summary.json"), _read(record_path)
    prior._frozen(manifest)
    _equal(manifest["dataset"], dataset, "prior corpus and split identity")
    expected = {"schema": prior.SCHEMA, "complete": True, "declared_cells": 4, "completed_cells": 4,
                "failures": [], "heldout_test_evaluated": False, "physical_fpga_execution": False,
                "speed_advantage_claim": False, "three_seed_quality_gate_evaluated": False,
                "larger_paid_learning_benchmark_gate": False}
    for value in (summary, receipt):
        _equal({key: value.get(key) for key in expected}, expected, "completed prior scope")
        _equal(value["recipe"], prior.recipe(), "prior recipe")
        _equal(value["source_hashes"], manifest["source_hashes"], "prior sources")
        _equal([[cell["arm"], cell["k"]] for cell in value["cells"]],
               [[arm, k] for arm in prior.ARMS for k in prior.K_VALUES], "prior cell membership")
    artifacts = _read(root / "artifact_manifest.json")
    _equal(artifacts, prior._artifacts(root), "every actual prior artifact")
    if any("failure" in Path(row["file"]).name for row in artifacts["files"]):
        raise ValueError("failed prior artifact cannot initialize confirmation")
    for cell in summary["cells"]:
        directory = root / ("%s_k%d" % (cell["arm"], cell["k"]))
        _equal(cell, _read(directory / "summary.json"), "prior raw cell summary")
        if deep:
            _equal(cell, prior.verify_worker(directory, manifest, cell["arm"], cell["k"]), "deep verified prior cell")
    expected_receipt = {**summary, "cells": [_compact_cell(cell) for cell in summary["cells"]],
                        "run_directory": str(root), "artifact_manifest_sha256": pilot.digest(artifact_bytes),
                        "artifact_files_verified": len(artifacts["files"]),
                        "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
                        "full_optimizer_checkpoints_verified": 12, "checkpoint_validation_windows": 192,
                        "distinct_validation_positions": 2048, "new_native_updates": 4096,
                        "gradient_pair_exposures": 524288, "host_fitting_pair_exposures": 262144,
                        "distinct_training_pairs": 131072, "native_training_trajectory_independently_replayed": False}
    _equal(receipt, expected_receipt, "completed prior audit/public receipt")
    initial_path = root / "count_fitted_k18" / "state_000000.i32"
    payload = initial_path.read_bytes()
    state = optimizer.decode_state(payload, pilot.layout(WIDTH))
    if (state["step"] != 0 or state["k"] != K or state["config"]["loss_kind"] != 1
            or state["config"]["zshift"] != ZSHIFT or any(any(row) for row in state["remainders"].values())):
        raise ValueError("prior initialization has wrong config, nonzero step or nonzero remainder")
    prior._frozen(manifest)
    if ((root / "manifest.json").read_bytes() != manifest_bytes
            or (root / "artifact_manifest.json").read_bytes() != artifact_bytes
            or (root / "summary.json").read_bytes() != summary_bytes or record_path.read_bytes() != record_bytes):
        raise ValueError("prior evidence changed during binding")
    _equal(prior._artifacts(root), artifacts, "prior payloads unchanged throughout binding")
    return {"run_directory": str(root), "record": {"file": str(record_path), "sha256": pilot.digest(record_bytes)},
            "manifest_sha256": pilot.digest(manifest_bytes), "summary_sha256": pilot.digest(summary_bytes),
            "artifact_manifest_sha256": pilot.digest(artifact_bytes),
            "initial_state": {"file": str(initial_path), "bytes": len(payload), "sha256": pilot.digest(payload)}}


def read_initial_state(binding, shape):
    root = Path(binding["run_directory"]).resolve(strict=True)
    path = root / "count_fitted_k18" / "state_000000.i32"
    if path.is_symlink() or path.parent.is_symlink() or binding["initial_state"]["file"] != str(path):
        raise ValueError("prior initial-state path is not the exact regular bound file")
    payload = path.read_bytes()
    _equal(binding["initial_state"],
           {"file": str(path), "bytes": len(payload), "sha256": pilot.digest(payload)}, "actual full initial state")
    state = optimizer.decode_state(payload, shape)
    if (state["step"] != 0 or state["k"] != K or state["config"]["loss_kind"] != 1
            or state["config"]["zshift"] != ZSHIFT or any(any(row) for row in state["remainders"].values())):
        raise ValueError("prior initial state is not the declared unchanged optimizer initialization")
    return payload


def initial_buffer(shape, binding):
    state = optimizer.decode_state(read_initial_state(binding, shape), shape)
    initial = shape.build_init(0)
    for name, values in state["weights"].items():
        offset = getattr(shape, "OFF_" + name)
        initial[offset:offset + len(values)] = values
    initial[shape.OFF_CFG] = state["config"]["loss_kind"]
    initial[shape.OFF_CFG + 1] = state["config"]["zshift"]
    initial[shape.OFF_CE_LUT:shape.OFF_CE_LUT + shape.WORDS_CE_LUT] = state["config"]["lut"]
    return initial


def _artifacts(root):
    result = prior._artifacts(root)
    result["schema"] = SCHEMA
    return result


def _seal(root):
    pilot.write_json(root / "artifact_manifest.json", _artifacts(root))


def _frozen(manifest):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("wrong confirmation schema")
    _equal(manifest["recipe"], recipe(), "declared confirmation recipe")
    if (manifest["source_hashes"] != source_identity()
            or pilot.digest(Path(manifest["dataset"]["path"]).read_bytes()) != manifest["dataset"]["sha256"]
            or pilot.digest(Path(manifest["executable"]["path"]).read_bytes()) != manifest["executable"]["sha256"]):
        raise ValueError("confirmation source/corpus/native executable changed")
    binding = manifest["prior_evidence"]
    _equal(binding, bind_prior(binding["run_directory"], binding["record"]["file"], manifest["dataset"], deep=False),
           "unchanged complete prior evidence")


def evaluate_cell(native, root, step, raw, dataset, shape):
    payload = native.export_state()
    state = optimizer.decode_state(payload, shape)
    if (state["step"] != step or state["k"] != K or state["config"]["loss_kind"] != 1
            or state["config"]["zshift"] != ZSHIFT):
        raise ValueError("optimizer checkpoint step/config differs")
    checkpoint = native.save_checkpoint(root, "state_%06d" % step)
    rows = []
    for index in VALIDATION:
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        actual = native.evaluate(tokens, targets)
        expected = observe(state["weights"], tokens, targets, shape, zshift=ZSHIFT)
        if (type(actual["step"]) is not int or actual["step"] != step
                or type(expected["saturation_events"]) is not int or expected["saturation_events"] != 0):
            raise ValueError("evaluation advanced state or encountered clipping")
        arrays = {}
        for name, values in {"predictions": expected["predictions"], "gradients": expected["gradients"],
                             **expected["native_routes"]}.items():
            found = actual[name] if name in ("predictions", "gradients") else actual["native_routes"][name]
            arrays[name] = screen._save_array(root, "eval_%06d_%03d_%s.i32" % (step, index, name), found)
            _equal(found, values, "native checkpoint arrays")
        disabled = pilot.integer_forward(state["weights"], tokens, targets, shape, zshift=ZSHIFT, use_context=False)
        row = {**screen._scalar_result(expected), "window_index": index, "native_arrays": arrays,
               "native_reference_equal": True, "native_routes_equal": True,
               "probes": {"context_disabled": screen._scalar_result(disabled)}}
        pilot.write_json(root / ("eval_%06d_%03d.json" % (step, index)), row)
        rows.append(row)
    if native.export_state() != payload:
        raise ValueError("validation changed the complete optimizer state")
    native.import_state(payload)
    if native.export_state() != payload:
        raise ValueError("import differs from the complete saved optimizer state")
    return {"step": step, "checkpoint": checkpoint, "state_sha256": pilot.digest(payload),
            "full_state_evaluation_invariant": True, "full_state_import_equal": True, **screen._aggregate(rows)}


def _comparators(baselines, one_pass):
    return {name: {"nll": min(baselines[str(STEPS)][name]["nll"], one_pass[name]["nll"])}
            for name in ("unigram", "bigram")}


def _cell_summary(seed, evaluations, baselines, one_pass):
    first, final = evaluations[0], evaluations[-1]
    context = final["context_disabled_nll"] - final["lut_normalized_nll"]
    comparators = _comparators(baselines, one_pass)
    return {"schema": SCHEMA, "complete": True, "seed": seed, "k": K, "zshift": ZSHIFT,
            "seed_meaning": "training-window permutation only", "training_order": training_order(seed),
            "training_updates": STEPS, "gradient_pair_exposures": STEPS * 128,
            "inherited_host_fitting_pair_exposures": STEPS * 128, "new_host_fitting_pair_exposures": 0,
            "initialization_refitted": False, "model_history_pair_exposures": 2 * STEPS * 128,
            "distinct_training_pairs": STEPS * 128,
            "initial_full_state_equal_prior": True, "evaluations": evaluations,
            "baselines_by_step": baselines, "one_pass_baseline": one_pass, "conservative_comparators": comparators,
            "conservative_bigram_advantage_nats": comparators["bigram"]["nll"] - final["lut_normalized_nll"],
            "context_advantage_nats": context,
            "context_advantage_growth_nats": context - (first["context_disabled_nll"] - first["lut_normalized_nll"]),
            "heldout_test_evaluated": False, "physical_fpga_execution": False,
            "speed_advantage_claim": False, "paid_execution_authorized": False}


def gate_results(cells):
    if (len(cells) != len(SEEDS) or any(type(row["seed"]) is not int for row in cells)
            or sorted(row["seed"] for row in cells) != list(SEEDS) or any(row.get("complete") is not True for row in cells)):
        raise ValueError("gates require all three complete declared order seeds")
    rows = [{"seed": row["seed"], "initial": row["evaluations"][0], "final": row["evaluations"][-1],
             "baselines": row["conservative_comparators"]} for row in cells]
    result = screen.gate_arm(rows)
    result["seed_meaning"] = "order replicates, not independent data samples or weight initializations"
    result["scope"] = "unchanged numerical screening thresholds; passing never authorizes cloud execution"
    return result


def run_worker(config):
    seed = config.seed
    order = training_order(seed)
    parent = Path(config.parent).resolve(strict=True)
    root = Path(config.output).resolve()
    if root != parent / ("seed_%d" % seed):
        raise ValueError("worker must own its exact declared child directory")
    manifest = _read(parent / "manifest.json")
    _frozen(manifest)
    history = Path(manifest["prior_evidence"]["run_directory"]).resolve(strict=True)
    if root == history or history in root.parents:
        raise ValueError("worker output cannot modify historical evidence")
    raw, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    _equal(dataset, manifest["dataset"], "worker corpus/split identity")
    shape = pilot.layout(WIDTH)
    initial = initial_buffer(shape, manifest["prior_evidence"])
    expected_initial = read_initial_state(manifest["prior_evidence"], shape)
    root.mkdir(parents=True, exist_ok=False)
    completed = 0
    try:
        pilot.write_json(root / "manifest.json", {**manifest, "seed": seed, "training_order": order,
                                                  "initial_buffer_sha256": pilot.digest(screen._i32(initial))})
        evaluations = []
        with optimizer.EFSGD(Path(manifest["executable"]["path"]), shape, initial, K,
                             timeout_s=60, log_path=root / "native.log") as native:
            if native.export_state() != expected_initial:
                raise ValueError("complete native initial state differs from the prior byte for byte")
            evaluations.append(evaluate_cell(native, root, 0, raw, dataset, shape))
            with (root / "training.jsonl").open("x") as log, (root / "training_predictions.i32").open("xb") as predlog, \
                    (root / "training_gradients.i32").open("xb") as gradlog:
                for completed, index in enumerate(order, 1):
                    tokens, targets = pilot.window(raw, dataset, "train", index)
                    result = native.train(tokens, targets)
                    if type(result["step"]) is not int or result["step"] != completed:
                        raise ValueError("native update count differs from actual shuffled-window budget")
                    pred, grad = screen._i32(result["predictions"]), screen._i32(result["gradients"])
                    if len(pred) != 128 * 4 or len(grad) != 128 * 128 * 4:
                        raise ValueError("incomplete native training outputs")
                    predlog.write(pred)
                    gradlog.write(grad)
                    log.write(json.dumps({"step": completed, "window_index": index,
                                          "predictions_sha256": pilot.digest(pred), "gradients_sha256": pilot.digest(grad)}) + "\n")
                    if completed in CHECKPOINTS:
                        evaluations.append(evaluate_cell(native, root, completed, raw, dataset, shape))
        baselines = {str(step): exposure_baseline(raw, dataset, seed, step) for step in CHECKPOINTS}
        one_pass = one_pass_baseline(raw, dataset)
        summary = _cell_summary(seed, evaluations, baselines, one_pass)
        _frozen(manifest)
        pilot.write_json(root / "summary.json", summary)
        _seal(root)
        return summary
    except BaseException as error:
        pilot.write_json(root / "failure.json", {"complete": False, "updates_attempted": completed,
                                                 "type": type(error).__name__, "error": str(error)})
        _seal(root)
        raise


def _verify_evaluation(root, evaluation, step, shape, identity, raw, dataset):
    payload = (root / ("state_%06d.i32" % step)).read_bytes()
    state = optimizer.decode_state(payload, shape)
    expected_checkpoint = {"schema": optimizer.SCHEMA, "file": "state_%06d.i32" % step,
                           "bytes": len(payload), "sha256": pilot.digest(payload), "step": state["step"],
                           "k": state["k"], "config": state["config"], "identity": identity,
                           "regions": [{"name": name, "weight_words": getattr(shape, "WORDS_" + name),
                                        "remainder_words": getattr(shape, "WORDS_" + name)} for name in pilot.WEIGHTS],
                           "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}
    _equal(evaluation["checkpoint"], expected_checkpoint, "full-state checkpoint metadata")
    _equal(_read(root / ("state_%06d.json" % step)), expected_checkpoint, "saved checkpoint metadata")
    if (state["step"] != step or state["k"] != K or state["config"]["loss_kind"] != 1
            or state["config"]["zshift"] != ZSHIFT or evaluation["state_sha256"] != pilot.digest(payload)
            or evaluation["full_state_evaluation_invariant"] is not True or evaluation["full_state_import_equal"] is not True):
        raise ValueError("state count/config/evaluation/import receipt differs")
    rows = []
    for index in VALIDATION:
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        expected = observe(state["weights"], tokens, targets, shape, zshift=ZSHIFT)
        if type(expected["saturation_events"]) is not int or expected["saturation_events"] != 0:
            raise ValueError("checkpoint contains observed clipping")
        row = _read(root / ("eval_%06d_%03d.json" % (step, index)))
        _equal({key: row.get(key) for key in ("window_index", "native_reference_equal", "native_routes_equal")},
               {"window_index": index, "native_reference_equal": True, "native_routes_equal": True}, "native row flags")
        arrays = {}
        for name, values in {"predictions": expected["predictions"], "gradients": expected["gradients"],
                             **expected["native_routes"]}.items():
            filename = "eval_%06d_%03d_%s.i32" % (step, index, name)
            encoded = screen._i32(values)
            if (root / filename).read_bytes() != encoded:
                raise ValueError("actual native arrays differ from independently recomputed checkpoint")
            arrays[name] = {"file": filename, "words": len(values), "bytes": len(encoded),
                            "sha256": pilot.digest(encoded), "format": "little-endian signed int32"}
        _equal(row["native_arrays"], arrays, "actual native array metadata")
        _equal({key: row[key] for key in screen._scalar_result(expected)}, screen._scalar_result(expected), "checkpoint observer")
        disabled = pilot.integer_forward(state["weights"], tokens, targets, shape, zshift=ZSHIFT, use_context=False)
        _equal(row["probes"]["context_disabled"], screen._scalar_result(disabled), "context-disabled probe")
        rows.append(row)
    aggregate = screen._aggregate(rows)
    _equal({key: evaluation[key] for key in aggregate}, aggregate, "evaluation aggregate and cohort rows")
    return payload


def verify_worker(root, manifest, seed):
    root = Path(root)
    order = training_order(seed)
    _frozen(manifest)
    _equal(_read(root / "artifact_manifest.json"), _artifacts(root), "complete cell artifact tree")
    if (root / "failure.json").exists():
        raise ValueError("failed cell cannot count as complete")
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], STEPS, max(VALIDATION) + 1)
    _equal(dataset, manifest["dataset"], "actual corpus/split declaration")
    shape = pilot.layout(WIDTH)
    initial = initial_buffer(shape, manifest["prior_evidence"])
    _equal(_read(root / "manifest.json"), {**manifest, "seed": seed, "training_order": order,
                                          "initial_buffer_sha256": pilot.digest(screen._i32(initial))}, "cell manifest")
    summary = _read(root / "summary.json")
    evaluations = summary["evaluations"]
    if (any(type(row["step"]) is not int for row in evaluations)
            or [row["step"] for row in evaluations] != list(CHECKPOINTS)):
        raise ValueError("all three declared checkpoints required")
    identity = _native_identity(manifest, shape)
    for evaluation in evaluations:
        payload = _verify_evaluation(root, evaluation, evaluation["step"], shape, identity, raw, dataset)
        if evaluation["step"] == 0 and payload != read_initial_state(manifest["prior_evidence"], shape):
            raise ValueError("actual full initial state is not byte-identical to completed prior initialization")
    baselines = {str(step): exposure_baseline(raw, dataset, seed, step) for step in CHECKPOINTS}
    expected_summary = _cell_summary(seed, evaluations, baselines, one_pass_baseline(raw, dataset))
    _equal(summary, expected_summary, "complete cell scope and exact exposure controls")
    lines = (root / "training.jsonl").read_text().splitlines()
    if len(lines) != STEPS:
        raise ValueError("training window count differs")
    for name, count in (("predictions", 128), ("gradients", 128 * 128)):
        payload = (root / ("training_%s.i32" % name)).read_bytes()
        if len(payload) != STEPS * count * 4:
            raise ValueError("training array byte coverage differs")
        for index, line in enumerate(lines):
            row = pilot._finite_json(line)
            if (type(row["step"]) is not int or row["step"] != index + 1
                    or type(row["window_index"]) is not int or row["window_index"] != order[index]
                    or row[name + "_sha256"] != pilot.digest(payload[index * count * 4:(index + 1) * count * 4])):
                raise ValueError("actual training arrays/order/exposure identity differs")
    _frozen(manifest)
    _equal(_read(root / "artifact_manifest.json"), _artifacts(root), "cell artifacts unchanged during verification")
    return summary


def _expected_command(root, manifest, seed):
    training_order(seed)
    root = Path(root)
    binding = manifest["prior_evidence"]
    return [sys.executable, str(Path(__file__).resolve()), "--worker", "--corpus", manifest["dataset"]["path"],
            "--prior-run", binding["run_directory"], "--prior-record", binding["record"]["file"],
            "--output", str(root / ("seed_%d" % seed)), "--parent", str(root), "--seed", str(seed)]


def _verify_execution(root, manifest, seed):
    execution = _read(Path(root) / ("seed_%d.execution.json" % seed))
    if (type(execution.get("returncode")) is not int or execution["returncode"] != 0
            or execution.get("timed_out") is not False or type(execution.get("timeout_s")) is not int
            or execution["timeout_s"] != WORKER_TIMEOUT_S
            or type(execution.get("elapsed_s")) not in (int, float)
            or not math.isfinite(execution["elapsed_s"]) or execution["elapsed_s"] < 0):
        raise ValueError("cell execution must be successful, bounded and finite")
    expected = {"command": _expected_command(root, manifest, seed), "returncode": 0, "timed_out": False,
                "timeout_s": WORKER_TIMEOUT_S, "elapsed_s": execution["elapsed_s"]}
    _equal(execution, expected, "exact declared bounded worker execution")
    return execution


def _run_in_directory(config, root, dataset, binding):
    executable = optimizer.build(root / "build", width=WIDTH)
    manifest = {"schema": SCHEMA, "recipe": recipe(), "dataset": dataset, "prior_evidence": binding,
                "source_hashes": source_identity(), "plan_sha256": pilot.digest(PLAN.read_bytes()),
                "executable": {"path": str(executable.resolve()), "sha256": pilot.digest(executable.read_bytes())}}
    pilot.write_json(root / "manifest.json", manifest)
    rows, failures = [], []
    for seed in SEEDS:
        _frozen(manifest)
        name = "seed_%d" % seed
        command = _expected_command(root, manifest, seed)
        try:
            execution = _run_process(command, root / (name + ".log"))
            pilot.write_json(root / (name + ".execution.json"), {"command": command, **execution})
            _verify_execution(root, manifest, seed)
            rows.append(verify_worker(root / name, manifest, seed))
        except Exception as error:
            failure = {"seed": seed, "type": type(error).__name__, "error": str(error)}
            pilot.write_json(root / (name + ".failure.json"), failure)
            failures.append(failure)
    _frozen(manifest)
    complete = not failures and len(rows) == len(SEEDS)
    result = {"schema": SCHEMA, "complete": complete, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
              "prior_evidence": binding, "declared_cells": len(SEEDS), "completed_cells": len(rows),
              "cells": rows, "failures": failures, "gates": gate_results(rows) if complete else None,
              "heldout_test_evaluated": False, "physical_fpga_execution": False, "speed_advantage_claim": False,
              "statistical_significance_claim": False, "paid_execution_authorized": False,
              "limitations": ["Order seeds are not independent weight initializations or data samples.",
                              "The charged count-fitting pass is inherited model history, not a new fit or FPGA pretraining.",
                              "New development cohort is not the untouched final test.",
                              "Requires new synthesis, optimizer-state ABI and physical verification before FPGA use.",
                              "Native training arrays are hash/coverage checked; not every update is independently replayed."]}
    pilot.write_json(root / "summary.json", result)
    _seal(root)
    return result


def run(config):
    root = prior.capacity._new_external_path(config.output)
    _, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    # A complete previous audit/public receipt is required BEFORE build/native work.
    binding = bind_prior(config.prior_run, config.prior_record, dataset, deep=True)
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _run_in_directory(config, root, dataset, binding)
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
        raise ValueError("public record must be outside the sealed raw run")
    original_artifact_bytes = (root / "artifact_manifest.json").read_bytes()
    expected = {"schema": SCHEMA, "complete": True, "declared_cells": len(SEEDS), "completed_cells": len(SEEDS),
                "failures": [], "heldout_test_evaluated": False, "physical_fpga_execution": False,
                "speed_advantage_claim": False, "statistical_significance_claim": False, "paid_execution_authorized": False}
    _equal({key: result.get(key) for key in expected}, expected, "complete publication scope")
    _equal(result["recipe"], recipe(), "published recipe")
    _equal([row["seed"] for row in result["cells"]], list(SEEDS), "all three order seeds")
    _equal(_read(root / "summary.json"), result, "actual parent summary")
    manifest = _read(root / "manifest.json")
    _frozen(manifest)
    _equal(result["source_hashes"], manifest["source_hashes"], "published sources")
    _equal(result["prior_evidence"], manifest["prior_evidence"], "published historical binding")
    for row in result["cells"]:
        _verify_execution(root, manifest, row["seed"])
        _equal(row, verify_worker(root / ("seed_%d" % row["seed"]), manifest, row["seed"]), "verified published cell")
    _equal(result["gates"], gate_results(result["cells"]), "unchanged recomputed numerical gates")
    declared = _read(root / "artifact_manifest.json")
    _equal(declared, _artifacts(root), "complete raw artifact membership and hashes")
    if any("failure" in Path(row["file"]).name for row in declared["files"]):
        raise ValueError("cannot publish success over a retained failure")
    value = {**result, "cells": [_compact_cell(row) for row in result["cells"]], "run_directory": str(root),
             "artifact_manifest_sha256": pilot.digest((root / "artifact_manifest.json").read_bytes()),
             "artifact_files_verified": len(declared["files"]),
             "artifact_payload_bytes": sum(row["bytes"] for row in declared["files"]),
             "full_optimizer_checkpoints_verified": len(SEEDS) * len(CHECKPOINTS),
             "checkpoint_validation_windows": len(SEEDS) * len(CHECKPOINTS) * len(VALIDATION),
             "distinct_validation_positions": len(VALIDATION) * 128,
             "new_native_updates": len(SEEDS) * STEPS, "gradient_pair_exposures": len(SEEDS) * STEPS * 128,
             "inherited_host_fitting_pair_exposures": len(SEEDS) * STEPS * 128,
             "new_host_fitting_pair_exposures": 0, "initialization_refitted": False,
             "model_history_pair_exposures": 2 * len(SEEDS) * STEPS * 128,
             "inherited_exposure_accounting": "sum of per-seed histories; the same old fit is shared, not newly repeated",
             "distinct_training_pairs": STEPS * 128,
             "native_training_trajectory_independently_replayed": False}
    _frozen(manifest)
    if (root / "artifact_manifest.json").read_bytes() != original_artifact_bytes:
        raise ValueError("raw artifact manifest changed during publication")
    _equal(declared, _artifacts(root), "actual artifact bytes unchanged throughout publication")
    pilot.write_json(target, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--prior-run", required=True, type=Path)
    parser.add_argument("--prior-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    config = parser.parse_args(argv)
    if config.worker:
        if config.parent is None or config.seed is None or config.record is not None:
            parser.error("worker requires its parent, order seed and no publication path")
        result = run_worker(config)
    else:
        if config.parent is not None or config.seed is not None:
            parser.error("order overrides require internal worker mode")
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
