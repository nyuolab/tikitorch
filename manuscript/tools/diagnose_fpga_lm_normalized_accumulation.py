"""Fixed nine-cell CPU normalized-accumulation mechanism study.

All cells restart from the same fitted diverse weights and zero carry. There
is no native execution, held-out evaluation, adaptive extension, or automatic
optimizer promotion. --check repeats saved state algebra; --recompute also
uses independent integer observers/forwards at the already-recorded points.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_gradient_accumulation as parent  # noqa: E402
from manuscript.tools import diagnose_fpga_lm_batch_step as step_study  # noqa: E402

old, prior, pilot = parent.old, parent.prior, parent.pilot
arithmetic, accumulation = parent.arithmetic, parent.accumulation
read, WEIGHTS = parent.read, parent.WEIGHTS
SCHEMA = "tikitorch.fpga_lm_normalized_accumulation.v1"
ARMS = ("seq_k18", "seq_k20", "batch4_k20")
ORDERS = ("consecutive", "shuffle_0", "shuffle_1")
ARM_CONFIG = {"seq_k18": {"batch_size": 1, "k": 18},
              "seq_k20": {"batch_size": 1, "k": 20},
              "batch4_k20": {"batch_size": 4, "k": 20}}
WINDOWS = tuple(range(32))
PROBES, PROBE_EXPOSURES = (0, 37, 511), (8, 32)
DEADLINE_S, GATE_TOLERANCE = 3600, 1e-12
PLAN = REPO / "manuscript/research/PLAN.md"


def equal(actual, expected, label):
    """New receipts retain exact integer/float/bool types without old edits."""
    if type(expected) in (int, float, bool) and type(actual) is not type(expected):
        raise ValueError(label + " numeric or boolean type differs")
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected.keys() & actual.keys():
            equal(actual[key], expected[key], label + "." + str(key))
    elif isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, label + "[" + str(index) + "]")
    parent.equal(actual, expected, label)


def source_identity():
    sources = {**parent.source_identity(), **step_study.source_identity()}
    for name in ("manuscript/tools/diagnose_fpga_lm_normalized_accumulation.py",
                 "tests/test_fpga_lm_normalized_accumulation.py",
                 "tests/test_fpga_lm_normalized_accumulation_replay.py"):
        sources[name] = old.digest((REPO / name).read_bytes())
    return sources


def training_order(order):
    if type(order) is not str or order not in ORDERS:
        raise ValueError("one declared order label is required")
    result = list(WINDOWS)
    if order != "consecutive":
        random.Random(0 if order == "shuffle_0" else 1).shuffle(result)
    return result


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "orders": list(ORDERS),
            "arm_config": {arm: dict(ARM_CONFIG[arm]) for arm in ARMS},
            "training_orders": {order: training_order(order) for order in ORDERS},
            "order_meaning": "fixed order-sensitivity replicates; same data/weight initialization, not independent seeds",
            "initialization": "exact original diverse fitted weights; zero carry for every cell; fixed k throughout",
            "partition": "train", "training_windows": list(WINDOWS),
            "loss": "unchanged raw legacy LUT zshift2 cotangents and integer floor-selected backward",
            "normalization": "batch4 sum(G)/2**20 equals mean(G)/2**18; no extra division or midtrajectory carry rescaling",
            "comparison": "batch4k20 versus seqk20 matches nominal per-exposure scale but changes query/release timing; "
                          "versus seqk18 changes both scale and timing",
            "shared_initial_readouts": {"training_windows": list(WINDOWS), "probes": list(PROBES), "context": [True, False]},
            "cell_readouts": {"final_training_windows": list(WINDOWS), "probe_exposures": list(PROBE_EXPOSURES),
                              "probes": list(PROBES), "context": [True, False]},
            "block_metrics": "means of saved actual query logits only; sequential queries do not share block-start weights",
            "declared_cells": 9, "full_gradient_queries": 288, "optimizer_commits": 216,
            "complete_saved_states": 225, "score_readouts": 434,
            "new_gradient_pair_exposures": 36864, "distinct_gradient_pairs": 4096,
            "new_fitting_pair_exposures": 0, "summed_inherited_fitting_pair_exposures": 1179648,
            "summed_endpoint_history_pair_exposures": 1216512, "distinct_fitted_pairs": 131072,
            "consecutive_control": "all32 seqk18 gradients/states match the completed frozen native-control chain",
            "checks": "all ordered gradient/carry int32 prefixes, int16 candidates, canonical fixed-k carry, "
                      "public/frozen EF parity and exact per-cell cumulative shadow identity",
            "gate": {"all_mechanics": True, "batch_improves_initial_every_order_both_losses": True,
                     "batch_mean_gain_vs_seqk20_minimum_nats": 0.01,
                     "batch_no_order_regression_vs_seqk20_both_losses": True,
                     "batch_mean_not_worse_than_seqk18_both_losses": True,
                     "inclusive_comparison_tolerance": GATE_TOLERANCE,
                     "scope": "training mechanism gate only; does not select an optimizer default or authorize scaling"},
            "torch_threads": 1, "deadline_s": DEADLINE_S,
            "new_native_training_calls": 0, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False,
            "recipe_promoted": False, "statistical_significance_claim": False, "adaptive_extension": False}


def bind_inputs(config):
    root, record = Path(config.prior_run).resolve(strict=True), Path(config.prior_record).resolve(strict=True)
    step_root, step_record = Path(config.step_run).resolve(strict=True), Path(config.step_record).resolve(strict=True)
    snapshots = [(path, prior.artifact_manifest(path), (path / "artifact_manifest.json").read_bytes(), receipt.read_bytes())
                 for path, receipt in ((root, record), (step_root, step_record))]
    parent.check(root, record, recompute=False)
    step_study.check(step_root, step_record, recompute=False)
    previous, step_manifest = read(root / "manifest.json"), read(step_root / "manifest.json")
    equal([step_manifest["binding"]["prior_run"], step_manifest["binding"]["prior_record"]],
          [str(root), str(record)], "batch-step rationale binds the identical accumulation predecessor")
    persistent = previous["binding"]
    raw, dataset = pilot.corpus_plan(Path(config.corpus), 1024, 88)
    equal(dataset, persistent["dataset"], "identical complete corpus/split")
    paths = [record, step_record, Path(dataset["path"]), Path(persistent["initial_state"])]
    paths += [path / name for path in (root, step_root)
              for name in ("manifest.json", "summary.json", "artifact_manifest.json")]
    binding = {"prior_run": str(root), "prior_record": str(record), "step_run": str(step_root),
               "step_record": str(step_record), "persistent_binding": persistent,
               "initial_state": persistent["initial_state"], "dataset": dataset,
               "inputs": [old.file_identity(path) for path in paths]}
    for (path, tree, seal, public), receipt in zip(snapshots, (record, step_record)):
        equal(prior.artifact_manifest(path), tree, "complete predecessor tree unchanged")
        equal((path / "artifact_manifest.json").read_bytes(), seal, "predecessor seal bytes unchanged")
        equal(receipt.read_bytes(), public, "predecessor public bytes unchanged")
    return binding, raw


def frozen(manifest):
    equal(manifest["schema"], SCHEMA, "schema")
    equal(manifest["recipe"], recipe(), "exact declared protocol")
    equal(manifest["source_hashes"], source_identity(), "frozen complete sources/tests")
    binding = manifest["binding"]
    for item in binding["inputs"]:
        equal(old.file_identity(item["path"]), item, "bound predecessor/input unchanged")
    rebuilt, _ = bind_inputs(SimpleNamespace(prior_run=binding["prior_run"], prior_record=binding["prior_record"],
                                            step_run=binding["step_run"], step_record=binding["step_record"],
                                            corpus=binding["dataset"]["path"]))
    equal(rebuilt, binding, "complete canonical recursive binding")


def state(path, k):
    if type(k) is not int or k not in (18, 20):
        raise ValueError("state requires literal fixed k18 or k20")
    arrays = parent.load(path)
    equal(set(arrays), {prefix + key for prefix in ("w_", "r_") for key in WEIGHTS}, "complete state arrays")
    weights, remainders = ({key: arrays[prefix + key] for key in WEIGHTS} for prefix in ("w_", "r_"))
    arithmetic.shadow(weights, remainders, k=k)
    return weights, remainders


def state_record(directory, consumed, commits, weights, remainders, k, *, write=False):
    if type(consumed) is not int or type(commits) is not int or not 0 <= commits <= consumed <= 32:
        raise ValueError("literal valid exposure/commit counters required")
    path = directory / f"state_{consumed:03d}.npz"
    if write:
        parent.save(path, parent.state_arrays(weights, remainders))
    actual_w, actual_r = state(path, k)
    parent.same(parent.state_arrays(actual_w, actual_r), parent.state_arrays(weights, remainders), "every complete state")
    payload = path.read_bytes()
    result = {"schema": SCHEMA, "file": path.name, "bytes": len(payload), "sha256": old.digest(payload),
              "k": k, "windows_consumed": consumed, "optimizer_commits": commits,
              "weights_sha256": old.tensor_hash(weights), "remainders_sha256": old.tensor_hash(remainders)}
    metadata_path = path.with_suffix(".json")
    if write:
        pilot.write_json(metadata_path, result)
    equal(read(metadata_path), result, "complete state k/shape/bytes/counters")
    return result


def _query_metrics(directory, index, raw, dataset):
    _, targets = parent.window(raw, dataset, index)
    return prior.metrics(parent.load(directory / f"query_{index:03d}.npz")["logits"], targets)


def expected_artifacts():
    files = {"manifest.json", "protocol.json", "declaration.md", "summary.json"}
    files.update(f"shared/score_initial_{index:03d}_1.npz" for index in WINDOWS)
    files.update(f"shared/score_probe_000_{index:03d}_{context}.npz" for index in PROBES for context in (0, 1))
    for arm in ARMS:
        batch_size = ARM_CONFIG[arm]["batch_size"]
        for order in ORDERS:
            prefix = arm + "/" + order + "/"
            files.add(prefix + "summary.json")
            files.update(prefix + f"query_{index:03d}.npz" for index in WINDOWS)
            for consumed in range(0, 33, batch_size):
                files.update(prefix + f"state_{consumed:03d}." + suffix for suffix in ("npz", "json"))
            files.update(prefix + f"transition_{commit:03d}.json" for commit in range(1, 32 // batch_size + 1))
            files.update(prefix + f"score_final_{index:03d}_1.npz" for index in WINDOWS)
            files.update(prefix + f"score_probe_{consumed:03d}_{index:03d}_{context}.npz"
                         for consumed in PROBE_EXPOSURES for index in PROBES for context in (0, 1))
    return files


def _probe(directory, consumed, weights, raw, dataset, *, write, recompute):
    return {"windows_consumed": consumed,
            "context_on": parent.collect_scores(directory, f"probe_{consumed:03d}", weights, PROBES, raw, dataset,
                                                write=write, recompute=recompute),
            "context_off": parent.collect_scores(directory, f"probe_{consumed:03d}", weights, PROBES, raw, dataset,
                                                 False, write=write, recompute=recompute)}


def _cell(manifest, output, raw, start_w, start_r, arm, order, *, write=False, recompute=False):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("declared literal arm required")
    order_indices = training_order(order)
    k, batch_size = ARM_CONFIG[arm]["k"], ARM_CONFIG[arm]["batch_size"]
    binding, dataset = manifest["binding"], manifest["binding"]["dataset"]
    directory = output / arm / order
    if write:
        directory.mkdir(parents=True)
    weights, remainders = prior.clone(start_w), prior.clone(start_r)
    previous_state = state_record(directory, 0, 0, weights, remainders, k, write=write)
    parent.same(weights, start_w, "same full fitted initial weights")
    old.require_zero(remainders, "same zero initial carry")
    total = {key: torch.zeros_like(weights[key]) for key in WEIGHTS}
    transitions, query_rows, blocks, probes = [], [], [], []
    commits = 0
    for block_start in range(0, 32, 4):
        block_indices = order_indices[block_start:block_start + 4]
        block_gradients, block_query_metrics = [], []
        for start in range(block_start, block_start + 4, batch_size):
            indices = order_indices[start:start + batch_size]
            gradients, inputs = [], []
            for position, index in enumerate(indices, start=start):
                gradient = parent.query(directory, index, weights, raw, dataset, write=write, recompute=recompute)
                gradients.append(gradient)
                block_gradients.append(gradient)
                total = {key: total[key] + gradient[key] for key in WEIGHTS}
                metrics = _query_metrics(directory, index, raw, dataset)
                block_query_metrics.append(metrics)
                path = directory / f"query_{index:03d}.npz"
                payload = path.read_bytes()
                query_row = {"position": position, "window_index": index, "file": path.name,
                             "bytes": len(payload), "sha256": old.digest(payload), "metrics": metrics,
                             "weights_sha256": old.tensor_hash(weights), "remainders_sha256": old.tensor_hash(remainders),
                             "gradient_sha256": old.tensor_hash(gradient)}
                inputs.append(query_row)
                query_rows.append(query_row)
            # Every query in a batch uses the exact same visible W; pending
            # raw gradients are separate from the canonical retained carry.
            summed = accumulation.checked_sum(gradients, remainders)
            result = accumulation.apply_block(weights, remainders, gradients, k=k)
            equal([result["public_reference_equal"], result["exact_shadow_recurrence"]], [True, True],
                  "public/frozen optimizer and exact shadow parity")
            parent.same(result["gradient"], summed, "complete committed raw gradient sum")
            next_w, next_r = result["weights"], result["remainders"]
            old.require_zero({key: (next_w[key] - weights[key]) * (1 << k)
                              - next_r[key] + remainders[key] + summed[key] for key in WEIGHTS},
                             "exact per-commit shadow recurrence")
            old.require_zero({key: (next_w[key] - start_w[key]) * (1 << k)
                              - next_r[key] + start_r[key] + total[key] for key in WEIGHTS},
                             "exact per-commit cumulative shadow identity")
            if arm == "seq_k18" and order == "consecutive":
                parent.sequential_control(binding["persistent_binding"], indices[0], weights, remainders, gradients[0], result)
            consumed, commits = start + batch_size, commits + 1
            next_state = state_record(directory, consumed, commits, next_w, next_r, k, write=write)
            transition = {"schema": SCHEMA, "arm": arm, "order": order, "k": k, "batch_size": batch_size,
                          "optimizer_commit": commits, "windows_consumed_before": start, "windows_consumed_after": consumed,
                          "window_indices": indices, "queries": inputs, "before_state": previous_state, "after_state": next_state,
                          "gradient_sum_sha256": old.tensor_hash(summed), "gradient_sum_statistics": old.stats(summed),
                          "ordered_prefix_max_abs": result["prefix_max_abs"],
                          "visible_motion": old.stats(old.subtract(next_w, weights)),
                          "remainder_statistics": old.stats(next_r), "public_reference_equal": True,
                          "exact_shadow_recurrence": True, "exact_cumulative_shadow_identity": True}
            path = directory / f"transition_{commits:03d}.json"
            if write:
                pilot.write_json(path, transition)
            equal(read(path), transition, "complete transition/query/state receipt")
            transitions.append(transition)
            weights, remainders, previous_state = next_w, next_r, next_state
        vectors = [prior.vector(gradient) for gradient in block_gradients]
        denominator = sum(float(vector.norm()) for vector in vectors)
        cancellation_ratio = float(sum(vectors).norm()) / denominator if denominator else None
        blocks.append({"windows_consumed": block_start + 4, "window_indices": block_indices,
                       "query_metrics": block_query_metrics, "query_mean": parent.mean_metrics(block_query_metrics),
                       "gradients_share_query_weights": batch_size == 4,
                       "gradient_sum_norm_over_sum_norms": cancellation_ratio,
                       "cumulative_gradient_sha256": old.tensor_hash(total),
                       "cumulative_shadow_identity": True})
        if block_start + 4 in PROBE_EXPOSURES:
            probes.append(_probe(directory, block_start + 4, weights, raw, dataset, write=write, recompute=recompute))
        if write or recompute:
            print(json.dumps({"arm": arm, "order": order, "windows_consumed": block_start + 4,
                              "commits": commits, "recompute": recompute}), flush=True)
    final_scores = parent.collect_scores(directory, "final", weights, WINDOWS, raw, dataset,
                                         write=write, recompute=recompute)
    cell = {"schema": SCHEMA, "arm": arm, "order": order, "k": k, "batch_size": batch_size,
            "training_order": order_indices, "complete": True, "mechanics_pass": True,
            "gradient_queries": len(query_rows), "optimizer_commits": commits, "windows_consumed": 32,
            "gradient_pair_exposures": 4096, "query_metrics": query_rows, "transitions": transitions,
            "blocks": blocks, "probes": probes, "final_scores": final_scores,
            "final_mean": parent.mean_metrics(final_scores), "exact_cumulative_shadow_identity": True,
            "parent_native_control_matches": True if arm == "seq_k18" and order == "consecutive" else None}
    equal([len(query_rows), commits, len(probes), len(final_scores)], [32, 32 // batch_size, 2, 32], "complete cell exposure/readout counts")
    if write:
        pilot.write_json(directory / "summary.json", cell)
    equal(read(directory / "summary.json"), cell, "complete reconstructed cell")
    return cell


def execute(manifest, output, raw, *, write=False, recompute=False):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("literal disjoint write/recompute modes required")
    output = Path(output)
    binding, dataset = manifest["binding"], manifest["binding"]["dataset"]
    decoded = old.native.decode_state(Path(binding["initial_state"]).read_bytes(), pilot.layout(512))
    equal([decoded["step"], decoded["k"], decoded["config"]["loss_kind"], decoded["config"]["zshift"]],
          [0, 18, 1, 2], "exact original fitted zero-step source state")
    start_w, start_r = old.state_tensors(decoded)
    arithmetic.shadow(start_w, start_r, k=18)
    old.require_zero(start_r, "original zero carry")
    shared = output / "shared"
    if write:
        shared.mkdir()
    training_scores = parent.collect_scores(shared, "initial", start_w, WINDOWS, raw, dataset, write=write, recompute=recompute)
    initial_probe = _probe(shared, 0, start_w, raw, dataset, write=write, recompute=recompute)
    shared_initial = {"training_scores": training_scores, "mean": parent.mean_metrics(training_scores),
                      "context_on": initial_probe["context_on"], "context_off": initial_probe["context_off"],
                      "weights_sha256": old.tensor_hash(start_w), "remainders_sha256": old.tensor_hash(start_r)}
    cells = [_cell(manifest, output, raw, start_w, start_r, arm, order, write=write, recompute=recompute)
             for arm in ARMS for order in ORDERS]
    equal([len(cells), sum(cell["gradient_queries"] for cell in cells), sum(cell["optimizer_commits"] for cell in cells)],
          [9, 288, 216], "all nine complete trajectories required")
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "shared_initial": shared_initial, "cells": cells, "gates": gates(cells, shared_initial["mean"]),
            "declared_cells": 9, "completed_cells": 9, "full_gradient_queries": 288,
            "optimizer_commits": 216, "complete_saved_states": 225, "score_readouts": 434,
            "new_gradient_pair_exposures": 36864, "new_fitting_pair_exposures": 0,
            "new_native_training_calls": 0, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "nine-cell host integer training-mechanism diagnostic; no held-out quality or physical hardware result"}


def gates(cells, initial_metrics):
    expected = {(arm, order) for arm in ARMS for order in ORDERS}
    if not isinstance(cells, list) or len(cells) != len(expected):
        raise ValueError("all nine cells are required before interpreting gates")
    lookup = {(cell["arm"], cell["order"]): cell for cell in cells}
    if set(lookup) != expected:
        raise ValueError("exact unique arm/order membership required")
    initial = parent.mean_metrics([initial_metrics])
    if any(cell.get("complete") is not True or cell.get("mechanics_pass") is not True for cell in cells):
        raise ValueError("all cells must be complete with checked mechanics before gates")
    endpoints = {key: parent.mean_metrics([value["final_mean"]]) for key, value in lookup.items()}
    losses = ("true_ce", "lut_nll")
    means = {arm: parent.mean_metrics([endpoints[(arm, order)] for order in ORDERS]) for arm in ARMS}
    paired = {order: {loss: endpoints[("seq_k20", order)][loss] - endpoints[("batch4_k20", order)][loss]
                      for loss in losses} for order in ORDERS}
    gains = {loss: means["seq_k20"][loss] - means["batch4_k20"][loss] for loss in losses}
    clauses = {
        "all_mechanics_pass": True,
        "batch_improves_initial_every_order_both_losses": all(endpoints[("batch4_k20", order)][loss] < initial[loss]
                                                               for order in ORDERS for loss in losses),
        "batch_mean_gain_vs_seqk20_both_losses": all(value + GATE_TOLERANCE >= 0.01 for value in gains.values()),
        "batch_no_order_regression_vs_seqk20_both_losses": all(value + GATE_TOLERANCE >= 0
                                                               for row in paired.values() for value in row.values()),
        "batch_mean_not_worse_than_seqk18_both_losses": all(means["batch4_k20"][loss] <= means["seq_k18"][loss] + GATE_TOLERANCE
                                                           for loss in losses)}
    return {"clauses": clauses, "numerical_training_gate_pass": all(clauses.values()),
            "initial_mean": initial, "arm_means": means, "seqk20_minus_batch_by_order": paired,
            "mean_seqk20_minus_batch": gains, "requires_separate_successful_independent_recompute": True,
            "broader_study_authorized": False, "recipe_promoted": False, "paid_execution_authorized": False,
            "scope": "exploratory training mechanism threshold; no optimizer default, generalization or hardware claim"}


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root),
            "artifact_manifest_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts),
            "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "check_scope": "complete saved-gradient state/metric replay; --recompute independently checks "
                           "all288 full gradients and434 score readouts; no new training trajectory"}


def check(root, record, recompute=False):
    if type(recompute) is not bool:
        raise ValueError("recompute must be a literal boolean")
    torch.set_num_threads(1)
    root = Path(root).resolve(strict=True)
    manifest = read(root / "manifest.json")
    frozen(manifest)
    equal(manifest["run_directory"], str(root), "owned run path")
    equal(read(root / "protocol.json"), recipe(), "frozen protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "saved declaration")
    artifacts = prior.artifact_manifest(root)
    seal_bytes, public_bytes = (root / "artifact_manifest.json").read_bytes(), Path(record).read_bytes()
    equal(read(root / "artifact_manifest.json"), artifacts, "complete artifact inventory")
    if any("failure" in row["file"] for row in artifacts):
        raise ValueError("failed run cannot be complete")
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact declared artifact membership")
    raw, dataset = pilot.corpus_plan(Path(manifest["binding"]["dataset"]["path"]), 1024, 88)
    equal(dataset, manifest["binding"]["dataset"], "same bound corpus")
    summary = execute(manifest, root, raw, recompute=recompute)
    equal(read(root / "summary.json"), summary, "fully reconstructed summary")
    expected = public_record(root, summary, artifacts)
    equal(read(record), expected, "public/raw record equality")
    frozen(manifest)
    equal(prior.artifact_manifest(root), artifacts, "run unchanged through check")
    equal(read(root / "artifact_manifest.json"), artifacts, "seal unchanged through check")
    equal(read(record), expected, "public unchanged through check")
    equal((root / "artifact_manifest.json").read_bytes(), seal_bytes, "seal bytes unchanged through check")
    equal(Path(record).read_bytes(), public_bytes, "public bytes unchanged through check")
    return expected


def run(config):
    output, target = (prior.capacity._new_external_path(path) for path in (config.output, config.record))
    if target == output or output in target.parents or target in output.parents:
        raise ValueError("public receipt must be outside raw output")
    if not target.parent.is_dir():
        raise FileNotFoundError("receipt parent must exist")
    sources = source_identity()
    output.mkdir(parents=True, exist_ok=False)
    previous, installed, published = None, False, None
    staging_directory, staged, owned = None, None, False
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed normalized-accumulation diagnostic deadline exceeded")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        binding, raw = bind_inputs(config)
        declaration = PLAN.read_bytes()
        manifest = {"schema": SCHEMA, "run_directory": str(output), "recipe": recipe(), "binding": binding,
                    "source_hashes": sources, "plan_sha256": old.digest(declaration),
                    "environment": {"python": sys.version, "numpy": np.__version__,
                                    "torch": torch.__version__, "device": "cpu"}}
        pilot.write_json(output / "manifest.json", manifest)
        pilot.write_json(output / "protocol.json", recipe())
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        frozen(manifest)
        summary = execute(manifest, output, raw, write=True)
        frozen(manifest)
        pilot.write_json(output / "summary.json", summary)
        artifacts = prior.artifact_manifest(output)
        pilot.write_json(output / "artifact_manifest.json", artifacts)
        public = public_record(output, summary, artifacts)
        published = old.digest((json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-normalized-accumulation-", dir=target.parent))
        staged = staging_directory / "receipt.json"
        pilot.write_json(staged, public)
        os.link(staged, target)
        owned = True
        result = check(output, target)
        staged.unlink()
        staging_directory.rmdir()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        installed = False
        return result
    except BaseException as error:
        same_inode = staged is not None and staged.is_file() and target.is_file() and os.path.samefile(staged, target)
        if (owned or same_inode) and published is not None and target.is_file() and old.digest(target.read_bytes()) == published:
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
    for name in ("output", "record", "corpus", "prior-run", "prior-record", "step-run", "step-record"):
        parser.add_argument("--" + name, type=Path, required=name in ("output", "record"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    config = parser.parse_args(argv)
    if config.recompute and not config.check:
        parser.error("--recompute requires --check")
    if not config.check and any(getattr(config, key) is None for key in
                                ("corpus", "prior_run", "prior_record", "step_run", "step_record")):
        parser.error("run requires corpus and complete accumulation and batch-step predecessor run/record pairs")
    if config.check:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed read-only normalized-accumulation check deadline exceeded")
        previous = signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
            result = check(config.output, config.record, config.recompute)
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                signal.signal(signal.SIGALRM, previous)
    else:
        result = run(config)
    print(json.dumps({"complete": result["complete"], "gates": result["gates"],
                      "check_only": config.check, "independent_recompute": config.recompute}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
