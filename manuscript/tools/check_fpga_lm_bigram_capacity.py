"""Host-fitted bigram representability in the frozen integer LM, not gradient learning.

Fit alpha-one counts to 1024 training windows, construct four checkpoints with
shifts0/1/2/4, and evaluate all on development windows40..55. Every native call
uses lr0 in a fresh CSIM process per checkpoint. This does not train on an FPGA,
test a warm start, select a scale, change a quality gate, or evaluate final test.
"""
from __future__ import annotations

import argparse
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

from manuscript.tools import audit_fpga_lm_head_screen as audit  # noqa: E402
from manuscript.tools import diagnose_fpga_lm_calibration as calibration  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_bigram_capacity.v1"
SHIFTS = (0, 1, 2, 4)
TRAIN_STEPS = 1024
VALIDATION_INDICES = tuple(range(40, 56))
WIDTH = 512
WORKER_TIMEOUT_S = 240
PLAN = REPO / "manuscript/research/PLAN.md"


def source_identity():
    result = calibration.source_identity()
    path = Path(__file__).resolve()
    result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def recipe():
    return {"width": WIDTH, "vocabulary": 128, "sequence": 128, "layers": 4,
            "seed": 0, "shifts": list(SHIFTS), "training_indices": list(range(TRAIN_STEPS)),
            "fitted_supervised_pairs": 131072, "alpha": 1,
            "validation_indices": list(VALIDATION_INDICES),
            "anchor": "d_v=(5*v)%512; W_HEAD[v,d_v]=0, all other head entries=-8192",
            "embedding": "E[t,d_v]=-(1<<z)*floor(16*log2(Cmax_t/C_tv)+0.5); unused columns=0",
            "rounding": "nearest integer, ties upward; index is not clipped before embedding",
            "other_weights": "MH=MP=MB=0", "loss": "unchanged balanced LUT surrogate",
            "native_learning_rate": 0, "native_threshold": 2048,
            "host_count_fitting": True, "gradient_training": False, "native_training_updates": 0,
            "native_execution": "fresh CSIM process per constructed checkpoint, one shared source-bound library",
            "worker_timeout_s": WORKER_TIMEOUT_S,
            "scope": "representability diagnostic, all four scales retained, no model selection or quality-gate promotion",
            "warm_start_policy": "not evaluated; any future count-fit plus gradient pass must count both data accesses",
            "physical_fpga_execution": False, "cloud_operations": False, "heldout_test_evaluated": False}


def construct_weights(counts, shape, zshift):
    """Generic small-shape helper; the executable protocol requires D512/V128.

    With B=max(abs(E)), residual coordinates lie conservatively in
    [-(L+1)B, LB]. No saturation plus an anchor margin8192-B>0 makes
    logit differences exactly the constructed embedding differences.
    """
    if type(zshift) is not int or zshift not in SHIFTS:
        raise ValueError("construction requires a declared literal integer shift")
    d, vocab, layers = shape.ML_D, shape.ML_V, shape.ML_N_LAYERS
    if (any(type(value) is not int or value <= 0 for value in (d, vocab, layers))
            or d & (d - 1) or d < vocab or shape.ML_LOG2_D != d.bit_length() - 1):
        raise ValueError("distinct anchors require a valid power-of-two width at least vocabulary")
    if (not isinstance(counts, list) or len(counts) != vocab
            or any(not isinstance(row, list) or len(row) != vocab
                   or any(type(value) is not int or not 1 <= value <= 131073 for value in row) for row in counts)):
        raise ValueError("counts must be a positive integer vocabulary-square matrix bounded by131073")
    indices = [[math.floor(16 * math.log2(max(row) / value) + 0.5) for value in row] for row in counts]
    bound = max(max(row) for row in indices) * (1 << zshift)
    if bound >= 8192 or (layers + 1) * bound > 32767:
        raise ValueError("construction lacks a strict head margin or conservative no-saturation bound")
    anchors = [(5 * v) % d for v in range(vocab)]
    if len(set(anchors)) != vocab:
        raise ValueError("anchors are not distinct")
    dimensions = {"EMBED": vocab * d, "MH": layers * d * shape.ML_K,
                  "MP": layers * d * d, "MB": layers * d, "W_HEAD": vocab * d}
    weights = {name: [0] * count for name, count in dimensions.items()}
    weights["W_HEAD"] = [-8192] * dimensions["W_HEAD"]
    for v, anchor in enumerate(anchors):
        weights["W_HEAD"][v * d + anchor] = 0
    for token, row in enumerate(indices):
        for target, index in enumerate(row):
            weights["EMBED"][token * d + anchors[target]] = -(1 << zshift) * index
    return weights, {"anchors": anchors, "maximum_unclipped_lut_index": max(max(row) for row in indices),
                     "maximum_embedding_magnitude": bound, "minimum_head_margin_bound": 8192 - bound,
                     "residual_lower_bound": -(layers + 1) * bound, "residual_upper_bound": layers * bound,
                     "count_pairs_excluding_pseudocounts": sum(map(sum, counts)) - vocab * vocab}


def analyze_window(weights, counts, tokens, targets, shape, zshift):
    """Two frozen forwards plus direct constructed-logit scoring; no native calls."""
    expected_weights, facts = construct_weights(counts, shape, zshift)
    if weights != expected_weights:
        raise ValueError("window weights differ from the exact host-fitted construction")
    detailed = calibration.integer_logits(weights, tokens, shape)
    scores = calibration.score_logits(detailed["logits"], targets, zshift)
    reference = pilot.integer_forward(weights, tokens, targets, shape, zshift=zshift)
    audit._assert_equal({name: scores[name] if name != "saturation_events" else detailed[name]
                         for name in reference}, reference, "two independent frozen forwards")
    if detailed["saturation_events"] != 0:
        raise ValueError("representation proof requires no int16 saturation")
    embedded_logits = [[weights["EMBED"][token * shape.ML_D + anchor] for anchor in facts["anchors"]]
                       for token in tokens]
    if any(len({actual - expected for actual, expected in zip(row, embedded)}) != 1
           for row, embedded in zip(detailed["logits"], embedded_logits)):
        raise ValueError("residual network did not preserve constructed logit differences")
    audit._assert_equal(calibration.score_logits(embedded_logits, targets, zshift), scores,
                        "direct constructed logits versus full model")
    context_off = pilot.integer_forward(weights, tokens, targets, shape, zshift=zshift, use_context=False)
    audit._assert_equal(context_off, reference, "constructed distribution must not use context")
    floating_predictions = [max(range(shape.ML_V), key=lambda v: counts[token][v]) for token in tokens]
    quantized_tie_changes = sum(a != b for a, b in zip(floating_predictions, scores["predictions"]))
    return {"scores": scores, "logits": detailed["logits"],
            "head_routes": facts["anchors"] * len(tokens),
            "facts": facts, "quantized_top1_changes_from_floating_counts": quantized_tie_changes,
            "saturation_events": 0, "context_distribution_equal": True,
            "logit_differences_equal_constructed_embedding": True}


def _read(path):
    return pilot._finite_json(Path(path).read_text())


def _artifacts(directory):
    result = screen._artifact_manifest(Path(directory))
    result["schema"] = SCHEMA
    return result


def _new_external_path(path):
    """Never place new runs/records inside any active or historical LM artifact tree."""
    candidate = Path(path).resolve()
    if candidate.exists():
        raise FileExistsError("output must be new: " + str(candidate))
    for ancestor in (candidate, *candidate.parents):
        manifest = ancestor / "manifest.json"
        if manifest.is_file():
            value = _read(manifest)
            if isinstance(value, dict) and str(value.get("schema", "")).startswith("tikitorch.fpga_lm"):
                raise ValueError("new output cannot be nested inside a source-bound LM run")
    return candidate


def _worker_paths(config):
    if type(config.zshift) is not int or config.zshift not in SHIFTS:
        raise ValueError("worker requires a declared literal integer shift")
    parent = Path(config.parent).resolve(strict=True)
    output = Path(config.output).resolve()
    if output != parent / ("shift_%d" % config.zshift):
        raise ValueError("worker output must be the exact owned shift child of its parent")
    manifest = _read(parent / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("backend") != "csim":
        raise ValueError("worker parent is not the capacity protocol")
    audit._assert_equal(manifest["recipe"], recipe(), "worker parent fixed recipe")
    audit._assert_equal(manifest["source_hashes"], source_identity(), "worker parent frozen sources")
    return parent, output


def _common(corpus):
    raw, dataset = pilot.corpus_plan(corpus, TRAIN_STEPS, max(VALIDATION_INDICES) + 1)
    shape = pilot.layout(WIDTH)
    controls = screen.matched_baselines(raw, dataset, range(TRAIN_STEPS), VALIDATION_INDICES)
    return raw, shape, controls, {"schema": SCHEMA, "recipe": recipe(), "backend": "csim",
                                "dataset": dataset, "layout": pilot.layout_values(shape),
                                "source_hashes": source_identity(),
                                "plan_sha256": pilot.digest(PLAN.read_bytes())}


def _frozen(manifest, library):
    if (source_identity() != manifest["source_hashes"]
            or pilot.digest(Path(manifest["dataset"]["path"]).read_bytes()) != manifest["dataset"]["sha256"]
            or pilot.digest(Path(library["path"]).read_bytes()) != library["sha256"]):
        raise ValueError("capacity sources, corpus or source-bound library changed")


def _summary(rows):
    return {"windows": len(rows), "positions": sum(len(row["scores"]["predictions"]) for row in rows),
            "lut_normalized_nll": sum(row["scores"]["lut_normalized_nll"] for row in rows) / len(rows),
            "accuracy": sum(row["scores"]["accuracy"] for row in rows) / len(rows),
            "rounded_zero_target_mass": sum(row["scores"]["rounded_zero_target_mass"] for row in rows),
            "target_index_at_or_above255": sum(sum(row["scores"]["target_index_at_or_above_last_bin_mask"]) for row in rows),
            "target_index_above255": sum(sum(row["scores"]["target_index_above_last_bin_mask"]) for row in rows),
            "quantized_top1_changes_from_floating_counts": sum(row["quantized_top1_changes_from_floating_counts"] for row in rows),
            "saturation_events": 0, "context_distribution_equal": True,
            "confidence_and_loss": calibration.summarize_scores([row["scores"] for row in rows])}


def run_worker(config, bridge_factory=None):
    parent, output = _worker_paths(config)
    output.mkdir(parents=True, exist_ok=False)
    bridge = None
    try:
        raw, shape, controls, manifest = _common(config.corpus)
        audit._assert_equal(_read(parent / "manifest.json"), manifest, "parent capacity protocol")
        audit._assert_equal(_read(parent / "counts.json"), controls, "parent fitted counts")
        library = _read(parent / "library.json")
        audit._verify_library(parent, library, manifest)
        weights, facts = construct_weights(controls["bigram_pair_counts"], shape, config.zshift)
        if facts["count_pairs_excluding_pseudocounts"] != 131072:
            raise ValueError("host fitting did not use the declared supervised pairs")
        import numpy as np
        manifest.update(zshift=config.zshift, construction=facts, library=library,
                        environment={"python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
        pilot.write_json(output / "manifest.json", manifest)
        initial = shape.build_init(0)
        for name, values in weights.items():
            initial[getattr(shape, "OFF_" + name):getattr(shape, "OFF_" + name) + len(values)] = values
        initial[shape.OFF_CFG:shape.OFF_CFG + 2] = [1, config.zshift]
        bridge = (bridge_factory or pilot.NativeBridge)(Path(library["path"]), "csim", shape)
        bridge.write(0, initial)
        if bridge.read(0, len(initial)) != initial:
            raise ValueError("constructed full buffer transport differs")
        checkpoint = pilot.checkpoint(output, 0, weights, shape)
        rows = []
        for index in VALIDATION_INDICES:
            tokens, targets = pilot.window(raw, manifest["dataset"], "validation", index)
            row = analyze_window(weights, controls["bigram_pair_counts"], tokens, targets, shape, config.zshift)
            pilot.stage_window(bridge, shape, tokens, targets)
            bridge.step(0, 2048)
            arrays = {}
            for name, offset, expected in (
                    ("predictions", shape.OFF_PRED, row["scores"]["predictions"]),
                    ("gradients", shape.OFF_GRAD_LOGIT, row["scores"]["gradients"]),
                    ("head_routes", shape.OFF_AD_HEAD, row["head_routes"])):
                values = bridge.read(offset, len(expected))
                if values != expected:
                    raise ValueError("actual native " + name + " differ from constructed reference")
                arrays[name] = screen._save_array(output, "window_%03d_%s.i32" % (index, name), values)
            row.update(window_index=index, native_arrays=arrays, native_reference_equal=True)
            pilot.write_json(output / ("window_%03d.json" % index), row)
            rows.append(row)
        if pilot.weight_bytes(pilot.read_weights(bridge, shape)) != pilot.weight_bytes(weights):
            raise ValueError("lr0 evaluation changed constructed full weights")
        closing, bridge = bridge, None
        closing.close()
        _frozen(manifest, library)
        summary = {"schema": SCHEMA, "complete": True, "zshift": config.zshift,
                   "host_count_fitting": True, "gradient_training": False, "native_training_updates": 0,
                   "checkpoint": checkpoint, "weights_unchanged": True, "native_reference_equal": True,
                   "construction": facts, **_summary(rows)}
        pilot.write_json(output / "summary.json", summary)
        pilot.write_json(output / "artifact_manifest.json", _artifacts(output))
        return summary
    except BaseException as error:
        pilot.write_json(output / "failure.json", {"complete": False, "error": str(error), "type": type(error).__name__})
        raise
    finally:
        if bridge is not None:
            try:
                bridge.close()
            except Exception as error:
                pilot.write_json(output / "cleanup_failure.json", {"error": str(error), "type": type(error).__name__})


def _run_process(command, log_path):
    with Path(log_path).open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO), start_new_session=True)
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


def verify_worker(directory, manifest, controls, raw, shape, zshift, library):
    """Recompute all retained host outputs and compare every saved native array."""
    directory = Path(directory)
    if any((directory / name).exists() for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("a failed capacity cell cannot pass verification")
    artifact_bytes = (directory / "artifact_manifest.json").read_bytes()
    actual = _artifacts(directory)
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    audit._assert_equal(_read(directory / "artifact_manifest.json"), actual, "capacity cell artifacts")
    worker = _read(directory / "manifest.json")
    audit._assert_equal({name: worker[name] for name in manifest}, manifest, "capacity cell protocol")
    audit._assert_equal(worker["zshift"], zshift, "capacity cell shift")
    audit._assert_equal(worker["library"], library, "capacity cell source-bound library")
    weights, facts = construct_weights(controls["bigram_pair_counts"], shape, zshift)
    audit._assert_equal(worker["construction"], facts, "capacity construction proof")
    summary = _read(directory / "summary.json")
    checkpoint = summary["checkpoint"]
    expected = {"step": 0, "file": "checkpoint_000000.i32", "location_portable": True,
                "sha256": pilot.digest(pilot.weight_bytes(weights)), "bytes": len(pilot.weight_bytes(weights)),
                "format": "little-endian signed int32, low16 weights",
                "regions": [{"name": name, "words": len(weights[name])} for name in pilot.WEIGHTS]}
    audit._assert_equal(checkpoint, expected, "constructed full checkpoint")
    audit._assert_equal(_read(directory / "checkpoint_000000.json"), expected, "checkpoint metadata")
    if (directory / "checkpoint_000000.i32").read_bytes() != pilot.weight_bytes(weights):
        raise ValueError("actual full checkpoint is not the host-fitted construction")
    rows = []
    for index in VALIDATION_INDICES:
        tokens, targets = pilot.window(raw, manifest["dataset"], "validation", index)
        expected_row = analyze_window(weights, controls["bigram_pair_counts"], tokens, targets, shape, zshift)
        row = _read(directory / ("window_%03d.json" % index))
        audit._assert_equal({name: row[name] for name in expected_row}, expected_row, "recomputed capacity window")
        audit._assert_equal(row["window_index"], index, "capacity window index")
        audit._assert_equal(row["native_reference_equal"], True, "native equality receipt")
        for name, values in (("predictions", expected_row["scores"]["predictions"]),
                             ("gradients", expected_row["scores"]["gradients"]),
                             ("head_routes", expected_row["head_routes"])):
            payload = screen._i32(values)
            filename = "window_%03d_%s.i32" % (index, name)
            audit._assert_equal(row["native_arrays"][name],
                                {"file": filename, "bytes": len(payload), "words": len(values),
                                 "sha256": pilot.digest(payload), "format": "little-endian signed int32"},
                                "native array metadata")
            if (directory / filename).read_bytes() != payload:
                raise ValueError("saved native array differs from recomputation")
        rows.append(row)
    expected_summary = {"schema": SCHEMA, "complete": True, "zshift": zshift,
                        "host_count_fitting": True, "gradient_training": False, "native_training_updates": 0,
                        "checkpoint": checkpoint, "weights_unchanged": True, "native_reference_equal": True,
                        "construction": facts, **_summary(rows)}
    audit._assert_equal(summary, expected_summary, "all capacity cell claims")
    actual = _artifacts(directory)
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    if (directory / "artifact_manifest.json").read_bytes() != artifact_bytes:
        raise ValueError("capacity artifact manifest changed during verification")
    audit._assert_equal(_read(directory / "artifact_manifest.json"), actual, "final capacity artifacts")
    return summary


def run(config):
    output = _new_external_path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        raw, shape, controls, manifest = _common(config.corpus)
        pilot.write_json(output / "manifest.json", manifest)
        pilot.write_json(output / "protocol.json", {"schema": SCHEMA, **recipe()})
        pilot.write_json(output / "counts.json", controls)
        library_path = pilot.build_csim(output / "build", WIDTH).resolve()
        library = {"path": str(library_path), "sha256": pilot.digest(library_path.read_bytes())}
        pilot.write_json(output / "library.json", library)
        build = audit._verify_library(output, library, manifest)
        root_bytes = {name: (output / name).read_bytes()
                      for name in ("manifest.json", "protocol.json", "counts.json", "library.json", "build/build.json")}
        results, failures = [], []
        sealed_workers = {}
        for shift in SHIFTS:
            _frozen(manifest, library)
            name = "shift_%d" % shift
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--corpus", str(Path(config.corpus).resolve()),
                       "--output", str(output / name), "--parent", str(output), "--zshift", str(shift)]
            execution = {"command": command, "timeout_s": WORKER_TIMEOUT_S,
                         **_run_process(command, output / (name + ".log"))}
            pilot.write_json(output / (name + "_execution.json"), execution)
            try:
                if type(execution["returncode"]) is not int or execution["returncode"] != 0 or execution["timed_out"] is not False:
                    raise ValueError("bounded capacity process failed or timed out")
                results.append(verify_worker(output / name, manifest, controls, raw, shape, shift, library))
                sealed_workers[name] = (output / name / "artifact_manifest.json").read_bytes()
            except Exception as error:
                failure = {"zshift": shift, "error": str(error), "type": type(error).__name__}
                failures.append(failure)
                pilot.write_json(output / (name + "_failure.json"), failure)
        _frozen(manifest, library)
        for name, payload in root_bytes.items():
            if (output / name).read_bytes() != payload:
                raise ValueError("root capacity input/provenance changed: " + name)
        for name, artifact_bytes in sealed_workers.items():
            directory = output / name
            if (directory / "artifact_manifest.json").read_bytes() != artifact_bytes:
                raise ValueError("verified capacity cell seal changed")
            actual = _artifacts(directory)
            actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
            audit._assert_equal(_read(directory / "artifact_manifest.json"), actual, "final sealed capacity cell")
        if not failures:
            for result in results[1:]:
                fields = ("lut_normalized_nll", "accuracy", "rounded_zero_target_mass", "target_index_at_or_above255",
                          "target_index_above255", "quantized_top1_changes_from_floating_counts", "confidence_and_loss")
                audit._assert_equal({name: result[name] for name in fields},
                                    {name: results[0][name] for name in fields}, "four exactly spaced LUT representations")
            for shift in SHIFTS[1:]:
                for index in VALIDATION_INDICES:
                    for name in ("predictions", "gradients", "head_routes"):
                        filename = "window_%03d_%s.i32" % (index, name)
                        if (output / ("shift_%d" % shift) / filename).read_bytes() != (output / "shift_0" / filename).read_bytes():
                            raise ValueError("cross-scale native distributions/gradients/routes differ")
        summary = {"schema": SCHEMA, "complete": not failures, "failures": failures,
                   "host_count_fitting": True, "gradient_training": False, "native_training_updates": 0,
                   "fitted_supervised_pairs": 131072, "distinct_validation_positions": 2048,
                   "declared_cells": len(SHIFTS), "completed_cells": len(results),
                   "all_scale_outputs_equal": not failures, "cells": results,
                   "baselines": {name: controls[name] for name in ("unigram", "bigram")},
                   "recipe": recipe(), "dataset": manifest["dataset"], "source_hashes": manifest["source_hashes"],
                   "plan_sha256": manifest["plan_sha256"], "build": build,
                   "physical_fpga_execution": False, "cloud_operations": False, "quality_gate_candidate": False,
                   "historical_gates_modified": False, "heldout_test_evaluated": False,
                   "limitations": ["Host-fitted parameters, not gradient-trained or FPGA-learned weights.",
                                   "Represents a count conditional without learned temporal context.",
                                   "Nearest log rounding and the LUT floor/cap can change count probabilities and top1 ties.",
                                   "LUT-normalized NLL is not the exact derivative used by balanced integer SignSGD.",
                                   "A future warm start must separately account for count fitting and subsequent gradient data passes.",
                                   "All four scales are retained; no selection, quality-gate promotion or efficiency result."]}
        if failures:
            pilot.write_json(output / "failure.json", {"complete": False, "failures": failures})
        pilot.write_json(output / "summary.json", summary)
        pilot.write_json(output / "artifact_manifest.json", _artifacts(output))
        return summary
    except BaseException as error:
        if not (output / "failure.json").exists():
            pilot.write_json(output / "failure.json", {"complete": False, "error": str(error), "type": type(error).__name__})
        raise


def publish_record(record, output, result):
    """Publish only the complete, unchanged sealed study; never overwrite evidence."""
    target, root = _new_external_path(record), Path(output).resolve(strict=True)
    if target == root or root in target.parents:
        raise ValueError("publication must be outside the raw run")
    expected = {"schema": SCHEMA, "complete": True, "failures": [], "host_count_fitting": True,
                "gradient_training": False, "native_training_updates": 0, "fitted_supervised_pairs": 131072,
                "distinct_validation_positions": 2048, "declared_cells": 4, "completed_cells": 4,
                "all_scale_outputs_equal": True, "physical_fpga_execution": False, "cloud_operations": False,
                "quality_gate_candidate": False, "historical_gates_modified": False, "heldout_test_evaluated": False}
    audit._assert_equal({name: result.get(name) for name in expected}, expected, "publishable capacity scope")
    audit._assert_equal(_read(root / "summary.json"), result, "raw summary publication link")
    audit._assert_equal(result["recipe"], recipe(), "published capacity recipe")
    audit._assert_equal([cell["zshift"] for cell in result["cells"]], list(SHIFTS), "all declared capacity cells")
    for cell in result["cells"]:
        expected_cell = {"schema": SCHEMA, "complete": True, "host_count_fitting": True,
                         "gradient_training": False, "native_training_updates": 0,
                         "windows": 16, "positions": 2048, "weights_unchanged": True,
                         "native_reference_equal": True, "saturation_events": 0, "context_distribution_equal": True}
        audit._assert_equal({name: cell.get(name) for name in expected_cell}, expected_cell, "full capacity cell coverage")
    manifest, library = _read(root / "manifest.json"), _read(root / "library.json")
    audit._assert_equal(result["source_hashes"], manifest["source_hashes"], "published source identities")
    _frozen(manifest, library)
    artifact_bytes = (root / "artifact_manifest.json").read_bytes()
    declared = _read(root / "artifact_manifest.json")
    actual = _artifacts(root)
    actual["files"] = [row for row in actual["files"] if row["file"] != "artifact_manifest.json"]
    audit._assert_equal(declared, actual, "entire sealed capacity output")
    if any("failure" in Path(row["file"]).name for row in declared["files"]):
        raise ValueError("failed raw studies cannot publish successful capacity records")
    _frozen(manifest, library)
    if (root / "artifact_manifest.json").read_bytes() != artifact_bytes:
        raise ValueError("root artifact seal changed during publication")
    value = {**result, "run_directory": str(root), "artifact_manifest_sha256": pilot.digest(artifact_bytes),
             "artifact_files_verified": len(declared["files"]),
             "artifact_payload_bytes": sum(row["bytes"] for row in declared["files"]),
             "constructed_full_checkpoints_verified": 4, "checkpoint_validation_windows": 64,
             "saved_native_predictions_recomputed": 8192, "saved_native_gradient_entries_recomputed": 1048576,
             "saved_native_head_route_entries_recomputed": 1048576}
    pilot.write_json(target, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, help="new compact record outside all raw run trees; written only on complete success")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--zshift", type=int, choices=SHIFTS, help=argparse.SUPPRESS)
    config = parser.parse_args(argv)
    if config.worker:
        if config.parent is None or config.zshift is None or config.record is not None:
            parser.error("worker requires parent and declared zshift")
        result = run_worker(config)
    else:
        if config.parent is not None or config.zshift is not None:
            parser.error("worker-only overrides are not a different public recipe")
        output = _new_external_path(config.output)
        if config.record is not None:
            target = _new_external_path(config.record)
            if target == output or output in target.parents:
                parser.error("record must be outside the raw output")
        result = run(config)
        if config.record is not None and result["complete"] is True:
            publish_record(config.record, config.output, result)
    print(json.dumps({"complete": result["complete"], "output": str(config.output)}, allow_nan=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
