"""Audit the fixed 1024-window CPU budget extension without running native code.

Reuses the frozen v1 observer and artifact helpers, with new checkpoint,
validation-cohort and matched-budget contracts. Binary artifacts remain local;
the compact output is a source-bound development record, not hardware evidence.
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
from manuscript.tools import extend_fpga_lm_budget as experiment  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from manuscript.tools.observe_fpga_lm_routes import observe  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

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
            or any(type(summary.get(name)) is not int or summary[name] != 6
                   for name in ("declared_workers", "completed_workers"))
            or any(summary.get(name) is not False for name in (
                "heldout_test_evaluated", "hardware_execution", "speed_advantage_claim",
                "statistical_significance_claim"))):
        raise ValueError("requires all six declared CPU extension cells without expanded claims")


def _verify_library(root, library, manifest):
    return prior._verify_library(root, library, manifest)


def _controls_for_step(step, controls):
    if type(step) is not int or step not in (0, 256, 1024):
        raise ValueError("unsupported checkpoint step")
    if step == 0:
        return None
    result = controls[str(step)]
    expected = {"training_pairs": step * 128, "train_indices": list(range(step)),
                "validation_indices": list(range(24, 40)), "alpha": 1, "vocabulary": 128}
    prior._assert_equal({name: result.get(name) for name in expected}, expected,
                        "checkpoint control budget and validation cohort")
    return result


def _verify_artifacts(root, original=None):
    payload = (root / "artifact_manifest.json").read_bytes()
    if original is not None and original != payload:
        raise ValueError("artifact manifest changed during audit")
    declared = pilot._finite_json(payload.decode("utf-8"))
    actual = experiment._artifact_manifest(root)
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    prior._assert_equal(actual, declared, "extension artifacts")
    return payload, declared


def _recompute_evaluation(root, evaluation, weights, raw, dataset, shape, zshift, controls):
    if [row["window_index"] for row in evaluation["per_window"]] != list(range(24, 40)):
        raise ValueError("validation cohort differs from declared windows24..39")
    counts = {"predictions": 0, "gradient_entries": 0, "route_digest_entries": 0}
    for offset, row in enumerate(evaluation["per_window"]):
        tokens, targets = pilot.window(raw, dataset, "validation", row["window_index"])
        result = observe(weights, tokens, targets, shape, zshift=zshift)
        scalar = screen._scalar_result(result)
        prior._assert_equal({name: row[name] for name in scalar}, scalar, "checkpoint observer")
        for name, count in (("predictions", 128), ("gradients", 128 * 128)):
            saved = prior._unpack(root / row["native_arrays"][name]["file"])
            if len(saved) != count or saved != result[name]:
                raise ValueError("saved native validation output differs from recomputed checkpoint")
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


def audit(directory):
    root = Path(directory).resolve(strict=True)
    manifest = prior._read(root / "manifest.json")
    summary = prior._read(root / "summary.json")
    _verify_root_status(manifest, summary)
    if any((root / name).exists() for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("cannot publish failed extension evidence")
    if manifest["recipe"] != experiment.recipe() or manifest["source_hashes"] != experiment.source_identity():
        raise ValueError("use the exact frozen extension sources and protocol")
    frozen = source_identity()
    artifact_bytes, artifacts = _verify_artifacts(root)
    prior._assert_equal(prior._read(root / "protocol.json"),
                        {"schema": experiment.SCHEMA, **manifest["recipe"]}, "extension protocol")
    library = prior._read(root / "library.json")
    build = _verify_library(root, library, manifest)
    environment = prior._read(root / "anchors_seed_0/manifest.json")["environment"]
    shape = pilot.layout(512)
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], 1024, 40)
    if dataset != manifest["dataset"] or pilot.layout_values(shape) != manifest["layout"]:
        raise ValueError("extension corpus or layout changed")
    bound_v1 = experiment.v1_binding(manifest["v1_evidence"]["run_directory"], dataset)
    prior._assert_equal(manifest["v1_evidence"], bound_v1, "original v1 linkage")
    controls = {str(step): screen.matched_baselines(raw, dataset, range(step), range(24, 40))
                for step in (256, 1024)}
    cells, by_arm, paired = [], {}, {}
    counts = {"predictions": 0, "gradient_entries": 0, "route_digest_entries": 0}
    for arm in experiment.ARMS:
        by_arm[arm] = []
        for seed in experiment.SEEDS:
            name = "%s_seed_%d" % (arm, seed)
            worker_root = root / name
            worker = experiment.verify_worker(worker_root, manifest, arm, seed, library["sha256"])
            execution = prior._read(root / (name + "_execution.json"))
            if (type(execution.get("returncode")) is not int or execution["returncode"] != 0
                    or execution.get("timed_out") is not False or execution.get("timeout_s") != 900):
                raise ValueError("worker lacks declared bounded successful completion")
            training = [pilot._finite_json(line)
                        for line in (worker_root / "training.jsonl").read_text().splitlines()]
            if any(type(row.get("token_pairs")) is not int or row["token_pairs"] != 128
                   or row.get("input_readback_equal") is not True for row in training):
                raise ValueError("training record lacks complete-window input checks")
            prior._assert_equal(worker["baselines_by_step"], controls, "both matched-budget baselines")
            prior._assert_equal(worker["baselines"], controls["1024"], "final gate baseline")
            initial = screen.initialize_buffer(shape, seed, arm)
            expected_weights = {region: initial[getattr(shape, "OFF_" + region):
                                                getattr(shape, "OFF_" + region) + getattr(shape, "WORDS_" + region)]
                                for region in pilot.WEIGHTS}
            initial_payload = (worker_root / "checkpoint_000000.i32").read_bytes()
            if initial_payload != pilot.weight_bytes(expected_weights):
                raise ValueError("extension did not restart from the declared initial weights")
            if paired.setdefault(seed, initial_payload) != initial_payload:
                raise ValueError("coarse and fine LUT arms have different initial weights")
            intermediate = (worker_root / "checkpoint_000256.i32").read_bytes()
            if intermediate != experiment.read_v1_checkpoint(bound_v1, arm, seed):
                raise ValueError("new validation cohort changed the full update256 checkpoint")
            cell = {"arm": arm, "seed": seed, "steps": worker["steps"],
                    "training_pairs": worker["training_pairs"], "v1_checkpoint_256_equal": True}
            for step, key in EVALUATIONS:
                evaluation = worker[key]
                prior._assert_equal(evaluation["step"], step, "checkpoint step")
                prior._assert_equal(evaluation["baseline_training_pairs"], step * 128 if step else None,
                                    "checkpoint comparison budget")
                weights = prior._weights(worker_root, evaluation["checkpoint"], shape)
                observed = _recompute_evaluation(worker_root, evaluation, weights, raw, dataset, shape,
                                                0 if arm == "anchors_fine" else 4,
                                                _controls_for_step(step, controls))
                for field in counts:
                    counts[field] += observed[field]
                cell[key] = {**prior._compact_evaluation(evaluation),
                             "baseline_training_pairs": evaluation["baseline_training_pairs"]}
            by_arm[arm].append(worker)
            cells.append(cell)
            print("Audited budget extension %s seed %d" % (arm, seed), flush=True)
    gates = {arm: screen.gate_arm(rows) for arm, rows in by_arm.items()}
    prior._assert_equal(summary["gates"], gates, "final matched-budget gates")
    prior._assert_equal(summary["measurements"],
                        {arm: [experiment._compact_worker(row) for row in rows] for arm, rows in by_arm.items()},
                        "parent measurements")
    prior._assert_equal(summary["paired_initialization_checks"], experiment._verify_paired_initializations(root),
                        "paired initialization summary")
    prior._assert_equal(experiment.v1_binding(bound_v1["run_directory"], dataset), bound_v1,
                        "v1 artifacts changed during audit")
    if source_identity() != frozen:
        raise ValueError("audit sources changed during recomputation")
    _verify_artifacts(root, artifact_bytes)
    return {"schema": "tikitorch.fpga_lm_budget_audit.v1", "complete": True,
            "run_directory": str(root), "recipe": manifest["recipe"], "dataset": dataset,
            "source_hashes": manifest["source_hashes"], "audit_source_hashes": frozen,
            "declared_plan_sha256": manifest["plan_document"]["sha256"],
            "v1_evidence": bound_v1, "library_sha256": library["sha256"],
            "build": build, "environment": environment,
            "artifact_manifest_sha256": pilot.digest(artifact_bytes),
            "artifact_files_verified": len(artifacts["files"]),
            "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
            "full_checkpoints_verified": 18, "v1_full_checkpoint_256_matches": 6,
            "initial_weights_paired": True,
            "distinct_validation_positions": 2048, "checkpoint_validation_windows": 288,
            "saved_native_predictions_recomputed": counts["predictions"],
            "saved_native_gradient_entries_recomputed": counts["gradient_entries"],
            "saved_route_digest_entries_recomputed": counts["route_digest_entries"],
            "baselines_by_step": {step: {name: control[name] for name in ("unigram", "bigram")}
                                  for step, control in controls.items()},
            "cells": cells, "gates": gates, "physical_fpga_execution": False,
            "heldout_test_evaluated": False, "audit_performs_cloud_operations": False,
            "limitations": ["Fixed development budget experiment, not a final-test or significance result.",
                            "Validation windows24..39 differ from v1 windows8..23; their scores are not paired.",
                            "Matched controls are refitted separately at256 and1024 updates; none at initialization.",
                            "The same131072 training pairs and2048 validation positions are reused across six cells.",
                            "LUT-normalized NLL is not ordinary softmax perplexity or the exact surrogate derivative.",
                            "Native validation arrays are recomputed; native routes are checked against saved hashes.",
                            "Training arrays are hash/coverage verified, not independently replayed every step.",
                            "The audit reuses the host observer; this is not outside-group reproduction.",
                            "No new silicon, accelerator speed, energy or cost-to-quality result.",
                            "Binary artifacts are local and ignored; fresh clones need these or complete reruns."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(args.run.resolve()):
        parser.error("output must be a new path outside the immutable run directory")
    result = audit(args.run)
    pilot.write_json(output, result)
    print(json.dumps({"complete": result["complete"], "output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
