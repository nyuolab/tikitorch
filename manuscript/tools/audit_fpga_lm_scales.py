"""Recompute the complete intermediate-training-scale study from saved checkpoints.

The audit loads no native library and performs no training or cloud operations.
It verifies both new training recipes and the old fixed-checkpoint controls on
the same declared development cohort; no result may substitute an older cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import audit_fpga_lm_head_screen as prior  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from manuscript.tools import train_fpga_lm_scales as experiment  # noqa: E402
from manuscript.tools.observe_fpga_lm_routes import observe  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_scales_audit.v1"
EVALUATIONS = ((0, "initial"), (256, "intermediate"), (1024, "final"))


def source_identity():
    result = experiment.source_identity()
    for path in (Path(__file__).resolve(), Path(prior.__file__).resolve()):
        result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def _verify_root_status(manifest, summary):
    if (manifest.get("schema") != experiment.SCHEMA or manifest.get("backend") != "csim"
            or summary.get("schema") != experiment.SCHEMA or summary.get("complete") is not True
            or summary.get("failures") != []
            or any(type(summary.get(name)) is not int or summary[name] != 12
                   for name in ("declared_workers", "completed_workers"))
            or any(summary.get(name) is not False for name in (
                "heldout_test_evaluated", "hardware_execution", "speed_advantage_claim",
                "statistical_significance_claim"))):
        raise ValueError("requires all twelve training/control cells without expanded claims")


def _controls_for_step(step, controls):
    if type(step) is not int or step not in (0, 256, 1024):
        raise ValueError("unsupported checkpoint step")
    if step == 0:
        return None
    result = controls[str(step)]
    expected = {"training_pairs": step * 128, "train_indices": list(range(step)),
                "validation_indices": list(range(40, 56)), "alpha": 1, "vocabulary": 128}
    prior._assert_equal({name: result.get(name) for name in expected}, expected,
                        "checkpoint control budget and new validation cohort")
    return result


def _verify_artifacts(root, original=None):
    payload = (root / "artifact_manifest.json").read_bytes()
    if original is not None and original != payload:
        raise ValueError("artifact manifest changed during audit")
    declared = pilot._finite_json(payload.decode("utf-8"))
    actual = experiment._artifact_manifest(root)
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    prior._assert_equal(actual, declared, "scale-study artifacts")
    return payload, declared


def _recompute_evaluation(root, evaluation, weights, raw, dataset, shape, zshift, controls):
    prior._assert_equal([row["window_index"] for row in evaluation["per_window"]],
                        list(range(40, 56)), "new validation cohort")
    counts = {"predictions": 0, "gradient_entries": 0, "route_digest_entries": 0}
    for offset, row in enumerate(evaluation["per_window"]):
        tokens, targets = pilot.window(raw, dataset, "validation", row["window_index"])
        result = observe(weights, tokens, targets, shape, zshift=zshift)
        scalar = screen._scalar_result(result)
        prior._assert_equal({name: row[name] for name in scalar}, scalar, "checkpoint observer")
        for name, count in (("predictions", 128), ("gradients", 128 * 128)):
            saved = prior._unpack(root / row["native_arrays"][name]["file"])
            if len(saved) != count or saved != result[name]:
                raise ValueError("saved native output differs from recomputed checkpoint")
        counts["predictions"] += 128
        counts["gradient_entries"] += 128 * 128
        for name, values in result["native_routes"].items():
            prior._assert_equal(row["native_route_hashes"][name],
                                {"words": len(values), "sha256": pilot.digest(screen._i32(values))},
                                "native route digest")
            counts["route_digest_entries"] += len(values)
        for name, probe_tokens, context in (
                ("context_disabled", tokens, False), ("all_space", [32] * 128, True),
                ("reversed_tokens", list(reversed(tokens)), True)):
            candidate = pilot.integer_forward(weights, probe_tokens, targets, shape,
                                              zshift=zshift, use_context=context)
            prior._assert_equal(row["probes"][name], screen._input_probe(result, candidate), "input probe")
        differences = ({} if controls is None else
                       {name: controls[name]["per_window"][offset]["nll"] - result["lut_normalized_nll"]
                        for name in ("unigram", "bigram")})
        prior._assert_equal(row["baseline_minus_model_nll"], differences, "checkpoint-specific baseline")
    aggregate = screen._aggregate(evaluation["per_window"])
    prior._assert_equal({name: evaluation[name] for name in aggregate}, aggregate, "evaluation aggregate")
    return counts


def _verify_training_scope(worker, arm):
    if arm not in experiment.ARMS:
        raise ValueError("undeclared training/control arm")
    training = arm in experiment.TRAIN_ARMS
    prior._assert_equal(worker["training_performed"], training, "new training scope")
    prior._assert_equal(worker["training_updates_performed"], 1024 if training else 0,
                        "new training update count")
    prior._assert_equal(worker["steps"], 1024, "final checkpoint training history")
    prior._assert_equal(worker["training_pairs"], 131072, "final checkpoint supervised budget")


def _budget_binding(manifest):
    binding = manifest["budget_evidence"]
    return experiment.budget_binding(binding["run_directory"], binding["audit_record"]["file"],
                                     manifest["dataset"])


def audit(directory):
    root = Path(directory).resolve(strict=True)
    manifest, summary = prior._read(root / "manifest.json"), prior._read(root / "summary.json")
    _verify_root_status(manifest, summary)
    prior._assert_equal(summary["declared_training_workers"], 6, "declared new-training cells")
    prior._assert_equal(summary["declared_control_workers"], 6, "declared fixed-checkpoint control cells")
    if any((root / name).exists() for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("cannot publish failed scale-study evidence")
    if manifest["recipe"] != experiment.recipe() or manifest["source_hashes"] != experiment.source_identity():
        raise ValueError("use the exact frozen scale-study sources and protocol")
    frozen = source_identity()
    artifact_bytes, artifacts = _verify_artifacts(root)
    prior._assert_equal(prior._read(root / "protocol.json"),
                        {"schema": experiment.SCHEMA, **manifest["recipe"]}, "scale-study protocol")
    library = prior._read(root / "library.json")
    build = prior._verify_library(root, library, manifest)
    environment = prior._read(root / "anchors_shift1_seed_0/manifest.json")["environment"]
    shape = pilot.layout(512)
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], 1024, 56)
    if dataset != manifest["dataset"] or pilot.layout_values(shape) != manifest["layout"]:
        raise ValueError("scale-study corpus or layout changed")
    bound = _budget_binding(manifest)
    prior._assert_equal(manifest["budget_evidence"], bound, "actual historical budget evidence")
    controls = {str(step): screen.matched_baselines(raw, dataset, range(step), range(40, 56))
                for step in (256, 1024)}
    cells, by_arm, paired = [], {}, {}
    counts = {"predictions": 0, "gradient_entries": 0, "route_digest_entries": 0}
    control_matches = new_initial_matches = new_updates = 0
    for arm in experiment.ARMS:
        by_arm[arm] = []
        for seed in experiment.SEEDS:
            name = "%s_seed_%d" % (arm, seed)
            worker_root = root / name
            worker = experiment.verify_worker(worker_root, manifest, arm, seed, library["sha256"])
            _verify_training_scope(worker, arm)
            execution = prior._read(root / (name + "_execution.json"))
            if (type(execution.get("returncode")) is not int or execution["returncode"] != 0
                    or execution.get("timed_out") is not False
                    or type(execution.get("timeout_s")) is not int or execution["timeout_s"] != 900):
                raise ValueError("worker lacks declared bounded successful completion")
            prior._assert_equal(worker["baselines_by_step"], controls, "both matched-budget controls")
            prior._assert_equal(worker["baselines"], controls["1024"], "final gate baseline")
            initial = experiment.initialize_buffer(shape, seed, arm)
            expected_weights = {region: initial[getattr(shape, "OFF_" + region):
                                                getattr(shape, "OFF_" + region) + getattr(shape, "WORDS_" + region)]
                                for region in pilot.WEIGHTS}
            initial_payload = (worker_root / "checkpoint_000000.i32").read_bytes()
            if initial_payload != pilot.weight_bytes(expected_weights):
                raise ValueError("actual initial weights differ from the anchored initialization")
            if initial_payload != experiment.read_budget_checkpoint(bound, "anchors", seed, 0):
                raise ValueError("actual initial weights differ from the historical paired control")
            if paired.setdefault(seed, initial_payload) != initial_payload:
                raise ValueError("paired initial weights differ across training/control arms")
            if arm in experiment.TRAIN_ARMS:
                new_initial_matches += 1
                new_updates += worker["training_updates_performed"]
                training = [pilot._finite_json(line)
                            for line in (worker_root / "training.jsonl").read_text().splitlines()]
                if any(type(row.get("token_pairs")) is not int or row["token_pairs"] != 128
                       or row.get("input_readback_equal") is not True for row in training):
                    raise ValueError("training record lacks complete-window input checks")
            cell = {"arm": arm, "seed": seed, "zshift": experiment.ARM_SHIFTS[arm],
                    "steps": worker["steps"], "training_pairs": worker["training_pairs"],
                    "training_performed": worker["training_performed"],
                    "training_updates_performed": worker["training_updates_performed"]}
            for step, key in EVALUATIONS:
                evaluation = worker[key]
                prior._assert_equal(evaluation["step"], step, "checkpoint step")
                prior._assert_equal(evaluation["baseline_training_pairs"], step * 128 if step else None,
                                    "checkpoint comparison budget")
                if arm in experiment.CONTROL_ARMS:
                    payload = (worker_root / evaluation["checkpoint"]["file"]).read_bytes()
                    if payload != experiment.read_budget_checkpoint(bound, arm, seed, step):
                        raise ValueError("control checkpoint differs from actual historical weights")
                    control_matches += 1
                weights = prior._weights(worker_root, evaluation["checkpoint"], shape)
                observed = _recompute_evaluation(worker_root, evaluation, weights, raw, dataset, shape,
                                                experiment.ARM_SHIFTS[arm], _controls_for_step(step, controls))
                for field in counts:
                    counts[field] += observed[field]
                cell[key] = {**prior._compact_evaluation(evaluation),
                             "baseline_training_pairs": evaluation["baseline_training_pairs"]}
            by_arm[arm].append(worker)
            cells.append(cell)
            print("Audited training scale %s seed %d" % (arm, seed), flush=True)
    gates = {arm: screen.gate_arm(rows) for arm, rows in by_arm.items()}
    prior._assert_equal(summary["gates"], gates, "complete new-cohort gates")
    prior._assert_equal(summary["measurements"],
                        {arm: [experiment._compact_worker(row) for row in rows] for arm, rows in by_arm.items()},
                        "parent measurements")
    prior._assert_equal(summary["paired_initialization_checks"],
                        experiment._verify_paired_initializations(root, bound), "paired initialization summary")
    expected_counts = {"predictions": 73728, "gradient_entries": 9437184,
                       "route_digest_entries": 576 * (shape.WORDS_AD_HEAD + shape.WORDS_PROJ_IDX + shape.WORDS_TAP_IDX)}
    prior._assert_equal(counts, expected_counts, "all checkpoint/native evaluation coverage")
    if control_matches != 18 or new_initial_matches != 6 or new_updates != 6144:
        raise ValueError("incomplete control-checkpoint or new-training coverage")
    prior._assert_equal(_budget_binding(manifest), bound, "historical artifacts changed during audit")
    if source_identity() != frozen or pilot.digest(Path(dataset["path"]).read_bytes()) != dataset["sha256"]:
        raise ValueError("audit sources or corpus changed during recomputation")
    _verify_artifacts(root, artifact_bytes)
    return {"schema": SCHEMA, "complete": True, "run_directory": str(root),
            "recipe": manifest["recipe"], "dataset": dataset,
            "source_hashes": manifest["source_hashes"], "audit_source_hashes": frozen,
            "declared_plan_sha256": manifest["plan_document"]["sha256"],
            "budget_evidence": bound, "library_sha256": library["sha256"],
            "build": build, "environment": environment,
            "artifact_manifest_sha256": pilot.digest(artifact_bytes),
            "artifact_files_verified": len(artifacts["files"]),
            "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
            "full_checkpoints_verified": 36, "historical_control_checkpoint_matches": control_matches,
            "new_initial_checkpoint_matches": new_initial_matches, "initial_weights_paired": True,
            "new_training_updates": new_updates, "new_training_pair_exposures": new_updates * 128,
            "distinct_training_supervised_pairs": 131072, "control_training_updates": 0,
            "distinct_validation_positions": 2048, "checkpoint_validation_windows": 576,
            "saved_native_predictions_recomputed": counts["predictions"],
            "saved_native_gradient_entries_recomputed": counts["gradient_entries"],
            "saved_route_digest_entries_recomputed": counts["route_digest_entries"],
            "baselines_by_step": {step: {name: control[name] for name in ("unigram", "bigram")}
                                  for step, control in controls.items()},
            "cells": cells, "gates": gates, "physical_fpga_execution": False,
            "heldout_test_evaluated": False, "audit_performs_cloud_operations": False,
            "limitations": ["Development study with fixed recipes and cohort, not final-test or significance evidence.",
                            "All arms are evaluated on windows40..55; earlier cohorts are not paired controls.",
                            "Only six cells train anew; six controls re-evaluate actual historical checkpoints.",
                            "Each final trained checkpoint saw the same131072 pairs, not twelve times as much distinct data.",
                            "Initial no-update controls have no fitted-baseline comparison.",
                            "LUT-normalized NLL is not conventional softmax perplexity or the exact surrogate derivative.",
                            "Validation native predictions/gradients are recomputed; route arrays have saved digests.",
                            "Training arrays are hash/coverage verified, not replayed through every native update.",
                            "The audit shares the frozen host observer; it is not an outside-group reproduction.",
                            "No new FPGA, speed, power, energy or cost-to-quality result follows.",
                            "Binary inputs remain local and ignored; this JSON is not a full checkpoint bundle."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.run.resolve(strict=True), args.output.resolve()
    manifest = prior._read(root / "manifest.json")
    historical = Path(manifest["budget_evidence"]["run_directory"]).resolve(strict=True)
    older = Path(prior._read(historical / "manifest.json")["v1_evidence"]["run_directory"]).resolve(strict=True)
    if output.exists() or any(output == path or path in output.parents for path in (root, historical, older)):
        parser.error("output must be new and outside every immutable run directory")
    result = audit(root)
    pilot.write_json(output, result)
    print(json.dumps({"complete": result["complete"], "output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
