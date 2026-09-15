"""Six-cell FP64 relative-head-mass control; no native or held-out execution.

Ancestry semantics are checked once per public invocation. The private,
invocation-local validation object is never serialized as trusted evidence;
all subsequent guards rehash the complete recognized dependency closure.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_normalized_accumulation as normalized  # noqa: E402
from manuscript.tools import fpga_lm_head_weighting as helper  # noqa: E402

parent, old, prior = normalized.parent, normalized.old, normalized.prior
pilot, read, equal, WEIGHTS = normalized.pilot, normalized.read, normalized.equal, normalized.WEIGHTS
SCHEMA = "tikitorch.fpga_lm_head_weighting.v1"
ARMS, ORDERS = ("relative_mass", "uniform_mass"), normalized.ORDERS
WINDOWS, PROBES, PROBE_EXPOSURES = tuple(range(32)), (0, 37, 511), (8, 32)
K, DEADLINE_S, GATE_TOLERANCE = 18, 3600, 1e-12
PLAN = REPO / "manuscript/research/PLAN.md"
training_order = normalized.training_order


def source_identity():
    sources = dict(normalized.source_identity())
    for name in ("manuscript/tools/fpga_lm_head_weighting.py",
                 "manuscript/tools/diagnose_fpga_lm_head_weighting.py",
                 "tests/test_fpga_lm_head_weighting.py",
                 "tests/test_fpga_lm_head_weighting_replay.py"):
        sources[name] = old.digest((REPO / name).read_bytes())
    return sources


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "orders": list(ORDERS),
            "training_orders": {order: training_order(order) for order in ORDERS},
            "windows": list(WINDOWS), "partition": "train", "divisor_k": K,
            "initialization": "identical inherited diverse fitted weights; no new fitting",
            "state": "FP64 weights only; no EF remainder or native optimizer state",
            "forward": "dynamic first-tie routes and floor centering, continuous bounded FP64 weights",
            "backward": "common real-centering selected-route STE; not the derivative of floor",
            "relative_mass_head": "c*65536*(m/mean_window(m))*(q-y)",
            "uniform_mass_head": "c*65536*(q-y)",
            "head_definitions": "r=ROM_energy>>normalization_shift; R=sum(r); q=r/R; m=R/65536",
            "c": "arithmetic mean of m over all tokens in the 32 saved normalized shared-initial train logits",
            "scale_scope": "equal mean nominal head mass, not equal parameter-gradient norm or realized update norm",
            "orders_scope": "fixed order sensitivity on the same examples, not independent initialization seeds",
            "shared_initial_readouts": {"training_windows": list(WINDOWS), "probes": list(PROBES), "context": [True, False]},
            "cell_readouts": {"final_training_windows": list(WINDOWS), "probe_exposures": list(PROBE_EXPOSURES),
                              "probes": list(PROBES), "context": [True, False]},
            "declared_cells": 6, "full_gradient_queries": 192, "optimizer_commits": 192,
            "complete_saved_states": 198, "score_readouts": 302,
            "new_gradient_pair_exposures": 24576, "distinct_gradient_pairs": 4096,
            "new_fitting_pair_exposures": 0, "summed_inherited_fitting_pair_exposures": 786432,
            "summed_endpoint_history_pair_exposures": 811008, "distinct_fitted_pairs": 131072,
            "gradient_audit": "independent reverse via autograd on detached-cotangent fixed-route affine replay",
            "forward_audit": "separate fractional floor-forward selector; full logits/routes/offsets checked",
            "recurrence": "stored FP64 W-G/2**18; no exact integer cumulative-shadow claim",
            "gate": {"uniform_improves_initial_every_order_both_losses": True,
                     "mean_relative_minus_uniform_minimum_nats": 0.01,
                     "uniform_no_order_regression_both_losses": True,
                     "inclusive_comparison_tolerance": GATE_TOLERANCE},
            "torch_threads": 1, "deadline_s": DEADLINE_S, "new_native_training_calls": 0,
            "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
            "paid_execution_authorized": False, "recipe_promoted": False,
            "adaptive_extension": False, "statistical_significance_claim": False}


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular non-symlink input required: " + str(path))
    return path.resolve(strict=True)


def ancestry_snapshot(root, record, audit):
    """Hash recognized sealed run roots and all declared external input files.

    Discovery is read-only and bounded to manifests with their own seals.
    It is not a replacement for the original semantic checker.
    """
    pending, roots, files = [Path(root).resolve(strict=True)], {}, set()
    files.update(_regular(path) for path in (record, audit))
    run_keys = {"prior_run", "step_run", "learning_contract_run", "run_directory",
                "budget_run", "confirmation_run", "v1_run", "capacity_run", "screen_run"}

    def discover(value, key=""):
        if isinstance(value, dict):
            descriptor = value.get("path") if "sha256" in value else None
            if descriptor is None and "sha256" in value and isinstance(value.get("file"), str):
                if Path(value["file"]).is_absolute():
                    descriptor = value["file"]
            if descriptor is not None:
                path = _regular(descriptor)
                actual = old.file_identity(path)
                equal(actual["sha256"], value["sha256"], "declared ancestor external file hash")
                if "bytes" in value:
                    equal(actual["bytes"], value["bytes"], "declared ancestor external file size")
                files.add(path)
            for name, item in value.items():
                discover(item, name)
        elif isinstance(value, list):
            for item in value:
                discover(item, key)
        elif isinstance(value, str):
            if key in run_keys:
                path = Path(value)
                if path.is_symlink() or not path.is_dir():
                    raise ValueError("non-symlink sealed ancestor directory required")
                pending.append(path.resolve(strict=True))
            elif (key.endswith(("_record", "_audit")) or key == "initial_state") and Path(value).is_absolute():
                files.add(_regular(value))

    while pending:
        directory = pending.pop()
        if str(directory) in roots:
            continue
        if len(roots) >= 64:
            raise ValueError("ancestor closure exceeds fixed safety bound")
        manifest_path, seal_path = directory / "manifest.json", directory / "artifact_manifest.json"
        _regular(manifest_path)
        _regular(seal_path)
        manifest = read(manifest_path)
        if not isinstance(manifest, dict) or not str(manifest.get("schema", "")).startswith("tikitorch.fpga_lm"):
            raise ValueError("recognized source-bound LM manifest required")
        artifacts = prior.artifact_manifest(directory)
        seal = read(seal_path)
        if isinstance(seal, dict):
            equal(set(seal), {"schema", "files"}, "complete versioned ancestor seal fields")
            equal(seal["schema"], manifest["schema"], "ancestor manifest/seal schema")
            equal(seal["files"], artifacts, "sealed versioned complete ancestor inventory")
        else:
            equal(seal, artifacts, "sealed complete ancestor inventory")
        if any("failure" in row["file"] for row in artifacts):
            raise ValueError("failed artifact tree is not a complete ancestor")
        roots[str(directory)] = {"artifacts": artifacts, "seal": old.file_identity(seal_path)}
        discover(manifest)
    return {"roots": {key: roots[key] for key in sorted(roots)},
            "files": [old.file_identity(path) for path in sorted(files)]}


@dataclass(frozen=True)
class _Validation:
    binding: dict
    sources: dict

    def __post_init__(self):
        # The working manifest must not alias this semantic-validation receipt.
        object.__setattr__(self, "binding", copy.deepcopy(self.binding))
        object.__setattr__(self, "sources", copy.deepcopy(self.sources))


def _audit_receipt(root, record, audit):
    value = read(audit)
    artifacts = prior.artifact_manifest(root)
    previous = read(root / "manifest.json")
    expected = {"schema": "tikitorch.fpga_lm_normalized_accumulation.cli_audit_receipt.v1",
                "complete": True, "independent_recompute": True, "exit_code": 0,
                "deadline_seconds": 3600,
                "parent_record_sha256": old.digest(Path(record).read_bytes()),
                "run_manifest_sha256": old.digest((root / "manifest.json").read_bytes()),
                "artifact_manifest_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
                "runner_sha256": normalized.source_identity()["manuscript/tools/diagnose_fpga_lm_normalized_accumulation.py"],
                "source_files_verified": len(normalized.source_identity()),
                "direct_input_files_verified": len(previous["binding"]["inputs"]),
                "artifact_files_verified": len(artifacts),
                "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
                "independent_full_gradient_queries": 288, "independent_score_readouts": 434,
                "complete_saved_states_verified": 225, "optimizer_commits_replayed": 216,
                "ordered_progress_blocks_verified": 72, "recomputed_gradient_pair_observations": 36864,
                "complete_parent_ancestry_verified": True, "source_input_raw_seal_public_bytes_unchanged": True,
                "new_training_updates": 0, "new_gradient_pair_exposures": 0,
                "new_native_training_calls": 0, "new_optimizer_trajectory_executed": False,
                "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
                "paid_execution_authorized": False, "recipe_promoted": False, "broader_study_authorized": False}
    for key, item in expected.items():
        equal(value.get(key), item, "successful bounded predecessor audit: " + key)
    command = value.get("command")
    if (not isinstance(command, list) or len(command) != 8 or any(type(item) is not str for item in command)
            or command[0] != "python" or command[2:5] != ["--check", "--recompute", "--output"]
            or command[6] != "--record"):
        raise ValueError("canonical predecessor read-only audit command required")
    def command_path(value):
        path = Path(value)
        return (path if path.is_absolute() else REPO / path).resolve(strict=True)
    equal(command_path(command[1]), REPO / "manuscript/tools/diagnose_fpga_lm_normalized_accumulation.py", "audit command source")
    equal(command_path(command[5]), root, "audit command raw tree")
    equal(command_path(command[7]), Path(record), "audit command public record")
    try:
        timestamps = [datetime.strptime(value[key], "%Y-%m-%d %H:%M:%S UTC")
                      for key in ("started_at_utc", "completed_at_utc")]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("canonical UTC predecessor audit timestamps required") from error
    for key, timestamp in zip(("started_at_utc", "completed_at_utc"), timestamps):
        equal(value[key], timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "canonical timestamp spelling")
    elapsed = (timestamps[1] - timestamps[0]).total_seconds()
    if not 0 <= elapsed <= 3600:
        raise ValueError("predecessor audit completion is outside its declared deadline")
    equal(value.get("numerical_training_gate_pass"), read(record)["gates"]["numerical_training_gate_pass"],
          "historical numerical gate preserved, including failure")
    return value


def calibration(root, raw, dataset):
    masses, identities = [], []
    for index in WINDOWS:
        path = Path(root) / "shared" / f"score_initial_{index:03d}_1.npz"
        arrays = parent.load(path)
        equal(set(arrays), {"logits"}, "saved initial logit payload")
        _, targets = parent.window(raw, dataset, index)
        info = helper.head(arrays["logits"], targets, 1.0, "uniform_mass")
        masses.append(info["mass_factor"])
        identities.append(old.file_identity(path))
    c = float(torch.cat(masses).mean())
    if not 0 < c <= 1:
        raise ValueError("invalid frozen initial mean head mass")
    return {"c": c, "window_indices": list(WINDOWS), "tokens": sum(item.numel() for item in masses),
            "source_logits": identities, "scope": "frozen before new trajectories; no target-based tuning"}


def bind_inputs(config):
    root = Path(config.prior_run).resolve(strict=True)
    record, audit = _regular(config.prior_record), _regular(config.prior_audit)
    sources = source_identity()
    snapshot = ancestry_snapshot(root, record, audit)
    normalized.check(root, record, recompute=False)
    _audit_receipt(root, record, audit)
    previous = read(root / "manifest.json")
    raw, dataset = pilot.corpus_plan(Path(config.corpus), 1024, 88)
    equal(dataset, previous["binding"]["dataset"], "same full corpus and partition")
    binding = {"prior_run": str(root), "prior_record": str(record), "prior_audit": str(audit),
               "dataset": dataset, "initial_state": previous["binding"]["initial_state"],
               "calibration": calibration(root, raw, dataset), "closure": snapshot}
    equal(ancestry_snapshot(root, record, audit), snapshot, "complete ancestry unchanged during semantic validation")
    equal(source_identity(), sources, "sources unchanged during semantic validation")
    return binding, raw, _Validation(binding, sources)


def frozen(manifest, validation):
    if not isinstance(validation, _Validation):
        raise ValueError("private per-invocation semantic validation required")
    equal(manifest["schema"], SCHEMA, "schema")
    equal(manifest["recipe"], recipe(), "declared protocol")
    equal(manifest["source_hashes"], source_identity(), "complete source/test bytes")
    equal(manifest["source_hashes"], validation.sources, "same invocation source bytes")
    binding = manifest["binding"]
    equal(binding, validation.binding, "same semantically validated binding")
    equal(ancestry_snapshot(binding["prior_run"], binding["prior_record"], binding["prior_audit"]),
          binding["closure"], "complete ancestor membership/payload/receipt bytes unchanged")


def state(path):
    weights = parent.load(path)
    equal(set(weights), set(WEIGHTS), "complete weights-only state")
    helper.validate_weights(weights)
    return weights


def state_record(directory, consumed, weights, *, write=False):
    if type(consumed) is not int or not 0 <= consumed <= 32:
        raise ValueError("literal consumed-window count required")
    helper.validate_weights(weights)
    path = Path(directory) / f"state_{consumed:03d}.npz"
    if write:
        parent.save(path, weights)
    parent.same(state(path), weights, "exact saved FP64 parameter state")
    payload = path.read_bytes()
    result = {"schema": SCHEMA, "file": path.name, "bytes": len(payload), "sha256": old.digest(payload),
              "windows_consumed": consumed, "optimizer_commits": consumed, "divisor_k": K,
              "dtype": "float64", "regions": list(WEIGHTS), "weights_sha256": old.tensor_hash(weights),
              "contains_optimizer_remainder": False, "native_state": False}
    record = path.with_suffix(".json")
    if write:
        pilot.write_json(record, result)
    equal(read(record), result, "complete weights-only state metadata")
    return result


TRACE_KEYS = ("logits", "centering_offsets", "AD_HEAD", "PROJ_IDX", "TAP_IDX")
HEAD_KEYS = ("cotangent", "rounded_energies", "rounded_totals", "mass_factor",
             "relative_weights", "row_weights", "truncated_probabilities")


def _base(arrays, weights, tokens, context=True):
    equal(set(arrays), set(TRACE_KEYS), "complete saved forward trace")
    helper.validate_weights(weights)
    extension = parent.extension
    result = {"schema": extension.SCHEMA, "mode": extension.MODE,
              "geometry": extension._geometry(weights, tokens), "tokens": tokens,
              "use_context": context, "logits": arrays["logits"],
              "centering_offsets": arrays["centering_offsets"],
              "routes": {key: arrays[key] for key in TRACE_KEYS[2:]},
              "parameter_sha256": extension.parameter_digest(weights),
              "autograd_forward": False, "native_execution": False, "saturation_events": 0}
    extension.reference_base(result)
    return result


def region_stats(values):
    equal(set(values), set(WEIGHTS), "all five statistics regions")
    result = {}
    for key in WEIGHTS:
        value = values[key]
        if value.dtype != torch.float64 or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("finite CPU FP64 statistic values required")
        result[key] = {"elements": value.numel(), "l1": float(value.abs().sum()),
                       "l2": float(value.norm()), "linf": float(value.abs().max())}
    return result


def collect_scores(directory, label, weights, indices, raw, dataset, context=True, *, write=False, recompute=False):
    rows = []
    for index in indices:
        tokens, targets = parent.window(raw, dataset, index)
        path = Path(directory) / f"score_{label}_{index:03d}_{int(context)}.npz"
        if write:
            parent.save(path, helper.trace(helper.forward(weights, tokens, use_context=context)))
        arrays = parent.load(path)
        _base(arrays, weights, tokens, context)
        if recompute:
            independent = helper.forward(weights, tokens, use_context=context, independent=True)
            parent.same(arrays, helper.trace(independent), "independent complete score forward")
        rows.append(prior.metrics(arrays["logits"], targets))
    return rows


def query(directory, index, weights, raw, dataset, c, arm, *, write=False, recompute=False):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("declared literal head-weighting arm required")
    tokens, targets = parent.window(raw, dataset, index)
    path = Path(directory) / f"query_{index:03d}.npz"
    if write:
        base = helper.forward(weights, tokens)
        head = helper.head(base["logits"], targets, c, arm)
        gradient = helper.gradient(weights, tokens, base, head)
        parent.save(path, {**helper.trace(base), **{"head_" + key: head[key] for key in HEAD_KEYS},
                           **{"g_" + key: gradient[key] for key in WEIGHTS}})
    arrays = parent.load(path)
    equal(set(arrays), set(TRACE_KEYS) | {"head_" + key for key in HEAD_KEYS} | {"g_" + key for key in WEIGHTS},
          "complete FP64 head/gradient/trace archive")
    trace = {key: arrays[key] for key in TRACE_KEYS}
    base = _base(trace, weights, tokens)
    actual_head = {key: arrays["head_" + key] for key in HEAD_KEYS}
    expected_head = helper.head(base["logits"], targets, c, arm)
    parent.same(actual_head, {key: expected_head[key] for key in HEAD_KEYS}, "exact supplied head rule from saved logits")
    gradient = {key: arrays["g_" + key] for key in WEIGHTS}
    for key in WEIGHTS:
        value = gradient[key]
        if (value.dtype != torch.float64 or value.device.type != "cpu" or value.shape != weights[key].shape
                or not torch.isfinite(value).all()):
            raise ValueError("complete finite FP64 parameter gradient required: " + key)
    if recompute:
        independent = helper.forward(weights, tokens, independent=True)
        parent.same(trace, helper.trace(independent), "independent dynamic query selection/floor/logits")
        independent_head = helper.head(independent["logits"], targets, c, arm, independent=True)
        helper.near(actual_head, {key: independent_head[key] for key in HEAD_KEYS}, "independent normalized head")
        independent_gradient = helper.gradient(weights, tokens, independent, independent_head, independent=True)
        helper.near(gradient, independent_gradient, "all five gradients versus independent autograd reverse")
    return gradient


def _query_row(directory, index, position, weights, gradient, raw, dataset):
    path = Path(directory) / f"query_{index:03d}.npz"
    arrays = parent.load(path)
    _, targets = parent.window(raw, dataset, index)
    payload = path.read_bytes()
    mass, row_weights = arrays["head_mass_factor"], arrays["head_row_weights"]
    return {"position": position, "window_index": index, "file": path.name,
            "bytes": len(payload), "sha256": old.digest(payload),
            "weights_sha256": old.tensor_hash(weights), "gradient_sha256": old.tensor_hash(gradient),
            "metrics": prior.metrics(arrays["logits"], targets),
            "mass_mean": float(mass.mean()), "mass_min": float(mass.min()), "mass_max": float(mass.max()),
            "row_weight_mean": float(row_weights.mean()), "row_weight_min": float(row_weights.min()),
            "row_weight_max": float(row_weights.max()), "gradient_statistics": region_stats(gradient)}


def _probe(directory, consumed, weights, raw, dataset, *, write, recompute):
    return {"windows_consumed": consumed,
            "context_on": collect_scores(directory, f"probe_{consumed:03d}", weights, PROBES, raw, dataset,
                                         write=write, recompute=recompute),
            "context_off": collect_scores(directory, f"probe_{consumed:03d}", weights, PROBES, raw, dataset,
                                          False, write=write, recompute=recompute)}


def _cell(manifest, output, raw, start_w, arm, order, *, write=False, recompute=False):
    directory = Path(output) / arm / order
    if write:
        directory.mkdir(parents=True, exist_ok=False)
    dataset = manifest["binding"]["dataset"]
    c, indices = manifest["binding"]["calibration"]["c"], training_order(order)
    weights = prior.clone(start_w)
    previous_state = state_record(directory, 0, weights, write=write)
    rows, transitions, probes, blocks = [], [], [], []
    for position, index in enumerate(indices):
        gradient = query(directory, index, weights, raw, dataset, c, arm, write=write, recompute=recompute)
        row = _query_row(directory, index, position, weights, gradient, raw, dataset)
        next_weights = helper.step(weights, gradient, k=K)
        consumed = position + 1
        next_state = state_record(directory, consumed, next_weights, write=write)
        transition = {"schema": SCHEMA, "arm": arm, "order": order, "divisor_k": K, "c": c,
                      "optimizer_commit": consumed, "windows_consumed_before": position,
                      "windows_consumed_after": consumed, "window_index": index, "query": row,
                      "before_state": previous_state, "after_state": next_state,
                      "gradient_statistics": region_stats(gradient),
                      "realized_update_statistics": region_stats(old.subtract(next_weights, weights)),
                      "stored_fp64_recurrence_equal": True, "integer_conservation_claim": False,
                      "native_execution": False}
        path = directory / f"transition_{consumed:03d}.json"
        if write:
            pilot.write_json(path, transition)
        equal(read(path), transition, "full reconstructed FP64 transition receipt")
        rows.append(row)
        transitions.append(transition)
        weights, previous_state = next_weights, next_state
        if consumed in PROBE_EXPOSURES:
            probes.append(_probe(directory, consumed, weights, raw, dataset, write=write, recompute=recompute))
        if consumed % 4 == 0:
            block_rows = rows[-4:]
            blocks.append({"windows_consumed": consumed, "window_indices": indices[consumed - 4:consumed],
                           "query_mean": parent.mean_metrics([item["metrics"] for item in block_rows]),
                           "gradients_share_query_weights": False})
            if write or recompute:
                print(json.dumps({"arm": arm, "order": order, "windows_consumed": consumed, "recompute": recompute}), flush=True)
    final = collect_scores(directory, "final", weights, WINDOWS, raw, dataset, write=write, recompute=recompute)
    cell = {"schema": SCHEMA, "arm": arm, "order": order, "training_order": indices, "c": c, "divisor_k": K,
            "complete": True, "mechanics_pass": True, "gradient_queries": len(rows), "optimizer_commits": len(transitions),
            "windows_consumed": 32, "gradient_pair_exposures": 4096, "initial_weights_sha256": old.tensor_hash(start_w),
            "final_weights_sha256": old.tensor_hash(weights), "query_metrics": rows, "transitions": transitions,
            "blocks": blocks, "probes": probes, "final_scores": final, "final_mean": parent.mean_metrics(final),
            "net_parameter_motion": region_stats(old.subtract(weights, start_w)),
            "new_fitting_pair_exposures": 0, "inherited_fitting_pair_exposures": 131072,
            "new_native_training_calls": 0, "stored_fp64_recurrence_equal": True}
    equal([len(rows), len(transitions), len(probes), len(final)], [32, 32, 2, 32], "complete cell counts")
    if write:
        pilot.write_json(directory / "summary.json", cell)
    equal(read(directory / "summary.json"), cell, "complete cell summary")
    return cell


def execute(manifest, output, raw, *, write=False, recompute=False):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("literal disjoint write/recompute modes required")
    output = Path(output)
    binding, dataset = manifest["binding"], manifest["binding"]["dataset"]
    decoded = old.native.decode_state(Path(binding["initial_state"]).read_bytes(), pilot.layout(512))
    equal([decoded["step"], decoded["k"], decoded["config"]["loss_kind"], decoded["config"]["zshift"]],
          [0, 18, 1, 2], "unchanged fitted initial state")
    start_w, start_r = old.state_tensors(decoded)
    old.require_zero(start_r, "initial inherited carry is exactly zero and not used")
    helper.validate_weights(start_w)
    shared = output / "shared"
    if write:
        shared.mkdir()
    initial_scores = collect_scores(shared, "initial", start_w, WINDOWS, raw, dataset, write=write, recompute=recompute)
    equal(binding["calibration"]["window_indices"], list(WINDOWS), "complete fixed calibration window order")
    equal(len(binding["calibration"]["source_logits"]), len(WINDOWS), "complete calibration payload count")
    for index, identity in zip(WINDOWS, binding["calibration"]["source_logits"]):
        equal(old.file_identity(identity["path"]), identity, "original calibration payload bytes")
        original = parent.load(identity["path"])
        equal(set(original), {"logits"}, "original initial-logit archive")
        current = parent.load(shared / f"score_initial_{index:03d}_1.npz")
        parent.same({"logits": current["logits"]}, original, "new initial forward equals actual predecessor logits")
    initial_probe = _probe(shared, 0, start_w, raw, dataset, write=write, recompute=recompute)
    initial = {"training_scores": initial_scores, "mean": parent.mean_metrics(initial_scores),
               "context_on": initial_probe["context_on"], "context_off": initial_probe["context_off"],
               "weights_sha256": old.tensor_hash(start_w), "calibration": binding["calibration"]}
    cells = [_cell(manifest, output, raw, start_w, arm, order, write=write, recompute=recompute)
             for arm in ARMS for order in ORDERS]
    equal([len(cells), sum(cell["gradient_queries"] for cell in cells), sum(cell["optimizer_commits"] for cell in cells)],
          [6, 192, 192], "all six complete trajectories required")
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "shared_initial": initial, "cells": cells, "gates": gates(cells, initial["mean"]),
            "declared_cells": 6, "completed_cells": 6, "full_gradient_queries": 192,
            "optimizer_commits": 192, "complete_saved_states": 198, "score_readouts": 302,
            "new_gradient_pair_exposures": 24576, "new_fitting_pair_exposures": 0,
            "new_native_training_calls": 0, "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
            "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "FP64 selected-route training mechanism control; no native recipe, held-out quality or hardware claim"}


def expected_artifacts():
    files = {"manifest.json", "protocol.json", "declaration.md", "summary.json"}
    files.update(f"shared/score_initial_{index:03d}_1.npz" for index in WINDOWS)
    files.update(f"shared/score_probe_000_{index:03d}_{context}.npz" for index in PROBES for context in (0, 1))
    for arm in ARMS:
        for order in ORDERS:
            prefix = arm + "/" + order + "/"
            files.add(prefix + "summary.json")
            files.update(prefix + f"query_{index:03d}.npz" for index in WINDOWS)
            files.update(prefix + f"state_{consumed:03d}." + suffix for consumed in range(33) for suffix in ("npz", "json"))
            files.update(prefix + f"transition_{consumed:03d}.json" for consumed in range(1, 33))
            files.update(prefix + f"score_final_{index:03d}_1.npz" for index in WINDOWS)
            files.update(prefix + f"score_probe_{consumed:03d}_{index:03d}_{context}.npz"
                         for consumed in PROBE_EXPOSURES for index in PROBES for context in (0, 1))
    return files


def gates(cells, initial_metrics):
    expected = {(arm, order) for arm in ARMS for order in ORDERS}
    if not isinstance(cells, list) or len(cells) != len(expected):
        raise ValueError("all six cells required before gates")
    lookup = {(cell["arm"], cell["order"]): cell for cell in cells}
    if set(lookup) != expected or any(cell.get("complete") is not True or cell.get("mechanics_pass") is not True for cell in cells):
        raise ValueError("complete unique healthy arm/order membership required")
    initial = parent.mean_metrics([initial_metrics])
    endpoints = {key: parent.mean_metrics([cell["final_mean"]]) for key, cell in lookup.items()}
    losses = ("true_ce", "lut_nll")
    means = {arm: parent.mean_metrics([endpoints[(arm, order)] for order in ORDERS]) for arm in ARMS}
    paired = {order: {loss: endpoints[("relative_mass", order)][loss] - endpoints[("uniform_mass", order)][loss]
                      for loss in losses} for order in ORDERS}
    gains = {loss: means["relative_mass"][loss] - means["uniform_mass"][loss] for loss in losses}
    clauses = {"all_mechanics_pass": True,
               "uniform_improves_initial_every_order_both_losses": all(endpoints[("uniform_mass", order)][loss] < initial[loss]
                                                                         for order in ORDERS for loss in losses),
               "uniform_mean_gain_both_losses": all(value + GATE_TOLERANCE >= 0.01 for value in gains.values()),
               "uniform_no_order_regression_both_losses": all(value + GATE_TOLERANCE >= 0 for row in paired.values() for value in row.values())}
    return {"clauses": clauses, "numerical_training_gate_pass": all(clauses.values()),
            "initial_mean": initial, "arm_means": means, "relative_minus_uniform_by_order": paired,
            "mean_relative_minus_uniform": gains, "requires_separate_successful_independent_recompute": True,
            "broader_study_authorized": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "training mechanism threshold only; no automatic optimizer or hardware promotion"}


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root),
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(item["bytes"] for item in artifacts),
            "check_scope": "complete saved FP64 recurrence and metric replay; --recompute additionally checks "
                           "192 full gradients and 302 readouts using separate forward/head/reverse algorithms; no new training"}


def _check(root, record, recompute, validation, raw):
    if type(recompute) is not bool:
        raise ValueError("literal recompute boolean required")
    root, record = Path(root).resolve(strict=True), _regular(record)
    manifest = read(root / "manifest.json")
    frozen(manifest, validation)
    equal(manifest["run_directory"], str(root), "owned raw directory")
    equal(read(root / "protocol.json"), recipe(), "saved complete recipe")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "saved declaration bytes")
    artifacts = prior.artifact_manifest(root)
    seal_bytes, public_bytes = (root / "artifact_manifest.json").read_bytes(), record.read_bytes()
    equal(read(root / "artifact_manifest.json"), artifacts, "complete raw artifact inventory")
    equal({item["file"] for item in artifacts}, expected_artifacts(), "exact artifact membership")
    summary = execute(manifest, root, raw, recompute=recompute)
    equal(read(root / "summary.json"), summary, "full reconstructed summary")
    expected = public_record(root, summary, artifacts)
    equal(read(record), expected, "public/raw result equality")
    frozen(manifest, validation)
    equal(prior.artifact_manifest(root), artifacts, "raw tree unchanged through check")
    equal((root / "artifact_manifest.json").read_bytes(), seal_bytes, "seal bytes unchanged through check")
    equal(record.read_bytes(), public_bytes, "public bytes unchanged through check")
    return expected


def _bound_config(manifest):
    binding = manifest["binding"]
    return SimpleNamespace(prior_run=binding["prior_run"], prior_record=binding["prior_record"],
                           prior_audit=binding["prior_audit"], corpus=binding["dataset"]["path"])


def check(root, record, recompute=False):
    if type(recompute) is not bool:
        raise ValueError("literal recompute boolean required")
    previous, installed = None, False
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed head-weighting read-only audit deadline exceeded")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        manifest = read(Path(root) / "manifest.json")
        binding, raw, validation = bind_inputs(_bound_config(manifest))
        equal(manifest["binding"], binding, "fresh canonical complete binding")
        result = _check(root, record, recompute, validation, raw)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        installed = False
        return result
    finally:
        if installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                signal.signal(signal.SIGALRM, previous)


def run(config):
    output, target = (prior.capacity._new_external_path(path) for path in (config.output, config.record))
    if target == output or output in target.parents or target in output.parents:
        raise ValueError("public receipt must be outside raw output")
    if not target.parent.is_dir():
        raise FileNotFoundError("receipt parent must already exist")
    sources = source_identity()
    output.mkdir(parents=True, exist_ok=False)
    previous, installed, published = None, False, None
    staged, staging_directory, owned_inode = None, None, None
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed head-weighting run deadline exceeded")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        binding, raw, validation = bind_inputs(config)
        equal(source_identity(), sources, "sources unchanged before run declaration")
        declaration = PLAN.read_bytes()
        manifest = {"schema": SCHEMA, "run_directory": str(output), "recipe": recipe(), "binding": binding,
                    "source_hashes": sources, "plan_sha256": old.digest(declaration),
                    "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"}}
        pilot.write_json(output / "manifest.json", manifest)
        pilot.write_json(output / "protocol.json", recipe())
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        frozen(manifest, validation)
        summary = execute(manifest, output, raw, write=True)
        frozen(manifest, validation)
        pilot.write_json(output / "summary.json", summary)
        artifacts = prior.artifact_manifest(output)
        equal({item["file"] for item in artifacts}, expected_artifacts(), "complete payload before publication")
        pilot.write_json(output / "artifact_manifest.json", artifacts)
        public = public_record(output, summary, artifacts)
        published = old.digest((json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-head-weighting-", dir=target.parent))
        staged = staging_directory / "receipt.json"
        pilot.write_json(staged, public)
        identity = staged.stat()
        owned_inode = (identity.st_dev, identity.st_ino)
        os.link(staged, target)
        result = _check(output, target, False, validation, raw)
        staged.unlink()
        staging_directory.rmdir()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        installed = False
        return result
    except BaseException as error:
        if target.is_file() and not target.is_symlink() and owned_inode is not None:
            identity = target.stat()
            if (identity.st_dev, identity.st_ino) == owned_inode and old.digest(target.read_bytes()) == published:
                rejected = output / "rejected_public.json"
                if rejected.exists():
                    raise RuntimeError("cannot overwrite rejected receipt") from error
                target.rename(rejected)
        if not (output / "failure.json").exists():
            pilot.write_json(output / "failure.json", {"complete": False, "type": type(error).__name__, "error": str(error)})
        raise
    finally:
        try:
            if staged is not None and staged.exists():
                staged.unlink()
            if staging_directory is not None and staging_directory.exists():
                staging_directory.rmdir()
        finally:
            if installed:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                finally:
                    signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--prior-record", type=Path)
    parser.add_argument("--prior-audit", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    if args.recompute and not args.check:
        parser.error("--recompute requires --check")
    if args.check:
        result = check(args.output, args.record, args.recompute)
    else:
        if any(getattr(args, name) is None for name in ("prior_run", "prior_record", "prior_audit", "corpus")):
            parser.error("a new run requires --prior-run, --prior-record, --prior-audit and --corpus")
        result = run(args)
    print(json.dumps({"complete": result["complete"], "gates": result["gates"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
