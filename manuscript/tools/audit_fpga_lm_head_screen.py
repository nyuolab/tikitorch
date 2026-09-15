"""Recompute a completed head screen from full checkpoints and saved native arrays.

No native library is loaded, no training is performed and no cloud API is used.
The compact output is suitable for manuscript/evidence; binary inputs stay in
manuscript/build. Use the recorded executable sources to re-audit old runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from manuscript.tools.observe_fpga_lm_routes import observe  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402


def _read(path):
    return pilot._finite_json(Path(path).read_text())


def _unpack(path):
    payload = Path(path).read_bytes()
    if len(payload) % 4:
        raise ValueError("incomplete int32 payload")
    return list(struct.unpack("<%di" % (len(payload) // 4), payload))


def _assert_equal(actual, expected, path="record"):
    """Exact structure/integer checks; tolerate only tiny log-rounding differences."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(path + ": keys differ")
        for name in expected:
            _assert_equal(actual[name], expected[name], path + "." + name)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(path + ": length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_equal(left, right, path + "[%d]" % index)
    elif type(expected) is float:
        if (type(actual) not in (float, int) or not math.isfinite(expected) or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(path + ": number differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(path + ": value differs")


def _weights(directory, record, shape):
    values = _unpack(directory / record["file"])
    weights, offset = {}, 0
    for name in pilot.WEIGHTS:
        count = getattr(shape, "WORDS_" + name)
        weights[name] = values[offset:offset + count]
        offset += count
    if offset != len(values):
        raise ValueError("checkpoint has an unsupported number of weights")
    return weights


def _verify_root_status(manifest, summary):
    if (manifest.get("schema") != screen.SCHEMA or manifest.get("backend") != "csim"
            or summary.get("schema") != screen.SCHEMA or summary.get("complete") is not True
            or summary.get("failures") != [] or summary.get("declared_workers") != 12
            or summary.get("completed_workers") != 12
            or type(summary.get("declared_workers")) is not int
            or type(summary.get("completed_workers")) is not int
            or any(summary.get(name) is not False for name in (
                "heldout_test_evaluated", "hardware_execution", "speed_advantage_claim",
                "statistical_significance_claim"))):
        raise ValueError("requires the complete declared twelve-cell CPU screen without expanded claims")


def _verify_library(root, library, manifest):
    name = Path(library["path"]).name
    if name not in ("libmorphlm.so", "libmorphlm.dylib"):
        raise ValueError("unsupported built library name")
    binary = root / "build" / name
    build = _read(root / "build/build.json")
    if (binary.is_symlink() or pilot.digest(binary.read_bytes()) != library["sha256"]
            or build.get("library_sha256") != library["sha256"]
            or build.get("source_hashes") != pilot.source_identity()
            or build.get("layout") != manifest["layout"]):
        raise ValueError("actual built library/source/layout identity differs")
    return build


def _compact_evaluation(evaluation):
    result = {name: evaluation[name] for name in (
        "step", "checkpoint", "lut_normalized_nll", "accuracy", "context_disabled_nll",
        "saturation_events", "rounded_zero_target_mass", "native_reference_equal", "native_routes_equal")}
    result["route_diagnostics"] = {name: value for name, value in evaluation["route_diagnostics"].items()
                                   if name != "head_winner_counts"}
    counts = evaluation["route_diagnostics"]["head_winner_counts"]
    result["route_diagnostics"].update(head_winners_min=min(counts), head_winners_max=max(counts))
    layers = []
    for layer in range(4):
        rows = [row["route_diagnostics"]["projection_layers"][layer] for row in evaluation["per_window"]]
        combined = {"layer": layer}
        for name in rows[0]:
            if name != "layer":
                reducer = max if name.startswith("max_") else sum
                combined[name] = reducer(row[name] for row in rows)
        layers.append(combined)
    result["route_diagnostics"]["projection_layers"] = layers
    result["per_window"] = [{"window_index": row["window_index"],
                             "lut_normalized_nll": row["lut_normalized_nll"],
                             "accuracy": row["accuracy"],
                             "context_disabled_nll": row["probes"]["context_disabled"]["lut_normalized_nll"],
                             "baseline_minus_model_nll": row["baseline_minus_model_nll"],
                             "input_sensitivity_probes": row["probes"]}
                            for row in evaluation["per_window"]]
    return result


def audit(directory):
    root = Path(directory).resolve(strict=True)
    manifest = _read(root / "manifest.json")
    summary = _read(root / "summary.json")
    _verify_root_status(manifest, summary)
    if any((root / name).exists() for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("cannot publish failed screen evidence")
    if manifest["recipe"] != screen.recipe() or manifest["source_hashes"] != screen.source_identity():
        raise ValueError("use the exact frozen screen sources and protocol to audit this run")
    source_before = {**screen.source_identity(),
                     Path(__file__).relative_to(REPO).as_posix(): pilot.digest(Path(__file__).read_bytes())}
    artifact_bytes = (root / "artifact_manifest.json").read_bytes()
    artifacts = pilot._finite_json(artifact_bytes.decode("utf-8"))
    actual = screen._artifact_manifest(root)
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    _assert_equal(actual, artifacts, "artifacts")
    _assert_equal(_read(root / "protocol.json"), {"schema": screen.SCHEMA, **manifest["recipe"]}, "protocol")
    library = _read(root / "library.json")
    build = _verify_library(root, library, manifest)
    environment = _read(root / "reference_seed_0/manifest.json")["environment"]
    shape = pilot.layout(512)
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], 256, 24)
    if dataset != manifest["dataset"] or pilot.layout_values(shape) != manifest["layout"]:
        raise ValueError("corpus or layout changed")
    controls = screen.matched_baselines(raw, dataset)
    cells, per_arm, paired_nonhead = [], {}, {}
    checked_predictions = checked_gradients = checked_routes = 0
    for arm in screen.ARMS:
        per_arm[arm] = []
        for seed in screen.SEEDS:
            path = root / (arm + "_seed_%d" % seed)
            worker = screen.verify_worker(path, manifest, arm, seed, library["sha256"])
            execution = _read(root / (arm + "_seed_%d_execution.json" % seed))
            if (type(execution.get("returncode")) is not int or execution.get("returncode") != 0
                    or execution.get("timed_out") is not False):
                raise ValueError("worker lacks successful process completion")
            training = [pilot._finite_json(line) for line in (path / "training.jsonl").read_text().splitlines()]
            if any(type(row.get("token_pairs")) is not int or row.get("token_pairs") != 128
                   or row.get("input_readback_equal") is not True for row in training):
                raise ValueError("training record lacks full-window input checks")
            _assert_equal(worker["baselines"], controls, "matched controls")
            _assert_equal(_read(path / "baselines.json"), controls, "controls payload")
            initial = screen.initialize_buffer(shape, seed, arm)
            initial_weights = {name: initial[getattr(shape, "OFF_" + name):
                                            getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
                               for name in pilot.WEIGHTS}
            if pilot.weight_bytes(initial_weights) != (path / "checkpoint_000000.i32").read_bytes():
                raise ValueError("initial checkpoint does not follow declared intervention")
            nonhead = b"".join(screen._i32(initial_weights[name]) for name in pilot.WEIGHTS if name != "W_HEAD")
            if paired_nonhead.setdefault(seed, nonhead) != nonhead:
                raise ValueError("paired arms have different initial non-head weights")
            zshift = 0 if arm == "anchors_fine" else 4
            for key in ("initial", "final"):
                evaluation = worker[key]
                weights = _weights(path, evaluation["checkpoint"], shape)
                for row in evaluation["per_window"]:
                    index = row["window_index"]
                    tokens, targets = pilot.window(raw, dataset, "validation", index)
                    result = observe(weights, tokens, targets, shape, zshift=zshift)
                    scalar = screen._scalar_result(result)
                    _assert_equal({name: row[name] for name in scalar}, scalar, "recomputed observer")
                    for name, count in (("predictions", 128), ("gradients", 128 * 128)):
                        saved = _unpack(path / row["native_arrays"][name]["file"])
                        if saved != result[name] or len(saved) != count:
                            raise ValueError("saved native array differs from recomputed checkpoint")
                    checked_predictions += 128
                    checked_gradients += 128 * 128
                    for name, values in result["native_routes"].items():
                        _assert_equal(row["native_route_hashes"][name],
                                      {"words": len(values), "sha256": pilot.digest(screen._i32(values))},
                                      "native route digest")
                        checked_routes += len(values)
                    for name, probe_tokens, context in (
                        ("context_disabled", tokens, False), ("all_space", [32] * 128, True),
                        ("reversed_tokens", list(reversed(tokens)), True)):
                        candidate = pilot.integer_forward(weights, probe_tokens, targets, shape,
                                                          zshift=zshift, use_context=context)
                        _assert_equal(row["probes"][name], screen._input_probe(result, candidate), "input probe")
                    expected_delta = {name: controls[name]["per_window"][index - 8]["nll"]
                                      - result["lut_normalized_nll"] for name in ("unigram", "bigram")}
                    _assert_equal(row["baseline_minus_model_nll"], expected_delta, "paired baseline delta")
                aggregate = screen._aggregate(evaluation["per_window"])
                _assert_equal({name: evaluation[name] for name in aggregate}, aggregate, "validation aggregate")
            per_arm[arm].append(worker)
            cells.append({"arm": arm, "seed": seed, "steps": worker["steps"],
                          "training_pairs": worker["training_pairs"],
                          "initial": _compact_evaluation(worker["initial"]),
                          "final": _compact_evaluation(worker["final"])})
            print("Audited %s seed %d" % (arm, seed), flush=True)
    gates = {arm: screen.gate_arm(rows) for arm, rows in per_arm.items()}
    _assert_equal(summary["gates"], gates, "screening gates")
    _assert_equal(summary["measurements"],
                  {arm: [screen._compact_worker(row) for row in rows] for arm, rows in per_arm.items()},
                  "parent measurements")
    _assert_equal(summary["paired_initialization_checks"], screen._verify_paired_initializations(root),
                  "parent initialization checks")
    source_after = {**screen.source_identity(),
                    Path(__file__).relative_to(REPO).as_posix(): pilot.digest(Path(__file__).read_bytes())}
    if source_after != source_before:
        raise ValueError("audit sources changed during recomputation")
    final_artifacts = screen._artifact_manifest(root)
    final_artifacts["files"] = [row for row in final_artifacts["files"] if row["file"] != "artifact_manifest.json"]
    _assert_equal(final_artifacts, artifacts, "artifacts changed during audit")
    if (root / "artifact_manifest.json").read_bytes() != artifact_bytes:
        raise ValueError("artifact manifest changed during audit")
    return {"schema": "tikitorch.fpga_lm_head_screen_audit.v1", "complete": True,
            "run_directory": str(root), "recipe": manifest["recipe"], "dataset": dataset,
            "source_hashes": manifest["source_hashes"], "audit_source_hashes": source_before,
            "declared_plan_sha256": manifest["plan_document"]["sha256"],
            "library_sha256": library["sha256"],
            "build": build,
            "environment": environment,
            "artifact_manifest_sha256": pilot.digest(artifact_bytes),
            "artifact_files_verified": len(artifacts["files"]),
            "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
            "full_checkpoints_verified": 24, "initial_nonhead_weights_paired": True,
            "saved_native_predictions_recomputed": checked_predictions,
            "saved_native_gradient_entries_recomputed": checked_gradients,
            "saved_route_digest_entries_recomputed": checked_routes,
            "baselines": {name: controls[name] for name in ("unigram", "bigram")},
            "cells": cells, "gates": gates, "physical_fpga_execution": False,
            "heldout_test_evaluated": False, "additional_aws_instances_launched": False,
            "limitations": ["Fixed development screen, not a final-test or statistical-significance result.",
                            "LUT-normalized NLL is not ordinary softmax perplexity or the exact surrogate derivative.",
                            "Checkpoint validation arrays are directly recomputed; native route arrays are hash-bound only.",
                            "Training arrays are hash/coverage verified, not independently replayed step by step.",
                            "This audit shares the host observer with the screen; it is not an outside-group reproduction.",
                            "No hardware speed, power, GPU savings or useful LLM pretraining claim.",
                            "Binary inputs are local ignored artifacts; a fresh clone needs these or a new complete run."]}


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
