"""Forward-only sampled-training endpoint diagnostic; no fitting or updates.

The successful development audit is an explicit immutable trust prerequisite.
Initial and final models are evaluated only on the fixed sparse training
cohort; development metrics are reused from the bound saved evidence.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import compare_fpga_lm_development as parent  # noqa: E402

guard, helper, old, prior, pilot = parent.guard, parent.helper, parent.old, parent.prior, parent.pilot
read, equal, load, tensor_hash = parent.read, parent.equal, parent.load, parent.tensor_hash
SCHEMA = "tikitorch.fpga_lm_endpoint_fit.v1"
ARMS, ORDER_SEEDS = parent.ARMS, parent.ORDER_SEEDS
TRAIN_WINDOWS = tuple(range(0, 1024, 16))
SEQUENCE, VOCAB = parent.SEQUENCE, parent.VOCAB
DEADLINE_S, STORAGE_CAP, FREE_RESERVE = 1800, 256 * (1 << 20), 4 * (1 << 30)
PLAN = REPO / "manuscript/research/PLAN.md"
# Admission remains impossible until the actual current independent audit
# completes and its exact successful receipt is pinned before source freeze.
AUDIT_RECEIPT_SHA256 = "500147cf9f03d8ac4b91362c74961ab8bd14aae438da2d6676a0f8a9b9a85e0a"


def source_identity():
    values = dict(parent.source_identity())
    for name in ("manuscript/tools/diagnose_fpga_lm_endpoint_fit.py", "tests/test_fpga_lm_endpoint_fit.py"):
        values[name] = old.digest((REPO / name).read_bytes())
    return values


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "order_seeds": list(ORDER_SEEDS),
            "training_windows": list(TRAIN_WINDOWS), "partition": "train", "context": True,
            "sequence": SEQUENCE, "vocabulary": VOCAB, "shared_initial_models": len(ARMS),
            "final_models": len(ARMS) * len(ORDER_SEEDS),
            "model_window_readouts": len(TRAIN_WINDOWS) * len(ARMS) * (1 + len(ORDER_SEEDS)),
            "baseline_window_readouts": 2 * len(TRAIN_WINDOWS),
            "audit": "independent dense tropical and expanded GRU forward; supplemental exact guarded trace/counter replay",
            "sample_scope": "fixed evenly spaced deterministic subset, not full training loss, random sample or convergence proof",
            "development": "reuse complete saved initial/final per-window development evidence only; no new dev forward",
            "dtype": "float64", "device": "cpu", "torch_threads": 1,
            "deadline_seconds": DEADLINE_S, "combined_new_output_cap_bytes": STORAGE_CAP, "free_reserve_bytes": FREE_RESERVE,
            "new_gradient_queries": 0, "optimizer_steps": 0, "new_training_updates": 0,
            "new_gradient_pair_exposures": 0, "new_model_count_fit": False,
            "new_development_evaluations": 0, "heldout_test_evaluated": False,
            "reserved_development_evaluated": False, "physical_fpga_execution": False,
            "paid_execution_authorized": False, "recipe_promoted": False,
            "adaptive_selection": False, "quality_gate": "none; all comparisons descriptive"}


class Deadline(parent.Deadline):
    def check(self):
        elapsed = self.elapsed()
        if not all(math.isfinite(value) for value in elapsed) or min(elapsed) < 0:
            raise RuntimeError("endpoint diagnostic clock moved backwards")
        if max(elapsed) >= DEADLINE_S:
            raise TimeoutError("fixed endpoint diagnostic deadline exceeded")
        return elapsed

    def _expired(self, signum, frame):
        raise TimeoutError("fixed endpoint diagnostic deadline exceeded")

    def arm(self):
        self.check()
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        self.installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S - max(self.elapsed()))
        self.watcher = threading.Thread(target=self._watch, name="endpoint-diagnostic-deadline", daemon=True)
        self.watcher.start()


class Storage(parent.Storage):
    def reserve(self, path, worst_case):
        self.clock.check()
        path = Path(path)
        if not any(root in path.resolve().parents for root in self.allowed_roots):
            raise ValueError("write outside explicitly owned endpoint outputs")
        if path.exists() or path.is_symlink():
            raise FileExistsError("append-only endpoint output required")
        if type(worst_case) is not int or worst_case < 0 or self.used + worst_case + (1 << 20) > STORAGE_CAP:
            raise RuntimeError("combined endpoint output storage cap would be exceeded")
        if shutil.disk_usage(path.parent).free - worst_case < FREE_RESERVE:
            raise RuntimeError("free disk reserve would be violated")

    def reconcile(self):
        size = super().reconcile()
        if size > STORAGE_CAP:
            raise RuntimeError("actual aggregate endpoint output exceeds cap")
        return size


def _audit_capsule(path, root, record):
    if (type(AUDIT_RECEIPT_SHA256) is not str or len(AUDIT_RECEIPT_SHA256) != 64
            or any(character not in "0123456789abcdef" for character in AUDIT_RECEIPT_SHA256)):
        raise ValueError("completed current independent development audit is not yet pinned")
    path = parent.rate.head._regular(path)
    equal(old.digest(path.read_bytes()), AUDIT_RECEIPT_SHA256, "actual successful development audit pin")
    value = read(path)
    expected = {"schema": parent.SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                "run_directory": str(root), "record": str(record), "source_hashes": parent.source_identity(),
                "producer_public_sha256": old.digest(record.read_bytes()),
                "producer_seal_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
                "deadline_seconds": parent.DEADLINE_S, "cell_deadline_seconds": parent.CELL_DEADLINE_S,
                "independent_tropical_window_vjps": 3072, "independent_tropical_window_forwards": 3520,
                "independent_gru_batch_queries": 768, "independent_gru_single_window_scores": 448,
                "supplemental_guarded_window_forwards": 3520, "full_batch_gradients_verified": 1536,
                "complete_saved_states_verified": 1542, "optimizer_commits_replayed": 1536,
                "score_window_readouts": 896, "baseline_window_readouts": 128,
                "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
                "new_gradient_pair_exposures": 0, "new_model_count_fit": False,
                "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                "reserved_development_evaluated": False, "physical_fpga_execution": False,
                "paid_execution_authorized": False, "recipe_promoted": False}
    for key, wanted in expected.items():
        equal(value.get(key), wanted, "successful development audit: " + key)
    equal(value["development_gates"], read(record)["gates"], "preserve actual predecessor quality outcome, including failure")
    for key in ("verification_elapsed_utc_seconds", "verification_elapsed_monotonic_seconds"):
        elapsed = value[key]
        if type(elapsed) is not float or not math.isfinite(elapsed) or not 0 <= elapsed < parent.DEADLINE_S:
            raise ValueError("completed audit must be within its actual bound")
    directory = Path(value["audit_directory"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("actual successful development audit directory required")
    directory = directory.resolve(strict=True)
    names = {"audit_manifest.json", "declaration.md", "verification.json"}
    equal({item.name for item in directory.iterdir()}, names, "exact complete independent audit capsule")
    descriptors = {name: old.file_identity(parent.rate.head._regular(directory / name)) for name in names}
    equal(descriptors["audit_manifest.json"]["sha256"], value["audit_manifest_sha256"], "actual audit manifest identity")
    equal((directory / "verification.json").read_bytes(), path.read_bytes(), "actual successful public/private audit receipt bytes")
    equal(read(directory / "verification.json"), value, "actual retained successful audit verification")
    manifest = read(directory / "audit_manifest.json")
    for key, wanted in {"schema": parent.SCHEMA, "complete": False, "mode": "independent_audit",
                        "run_directory": str(root), "record": str(record), "audit_directory": str(directory),
                        "audit_record": str(path), "source_hashes": parent.source_identity(),
                        "producer_public_sha256": value["producer_public_sha256"],
                        "producer_seal_sha256": value["producer_seal_sha256"],
                        "declaration_sha256": descriptors["declaration.md"]["sha256"],
                        "runtime": value["runtime"], "started_at_utc": value["started_at_utc"],
                        "deadline_seconds": parent.DEADLINE_S, "cell_deadline_seconds": parent.CELL_DEADLINE_S}.items():
        equal(manifest[key], wanted, "actual bound development audit declaration")
    return value, {"receipt": old.file_identity(path), "files": descriptors}


@dataclass(frozen=True)
class _Validation:
    binding: dict
    sources: dict
    parent_manifest: dict
    parent_validation: object

    def __post_init__(self):
        for key in ("binding", "sources", "parent_manifest"):
            object.__setattr__(self, key, copy.deepcopy(getattr(self, key)))


def bind_inputs(config):
    sources = source_identity()
    root, record = Path(config.prior_run).resolve(strict=True), parent.rate.head._regular(config.prior_record)
    _, capsule = _audit_capsule(config.prior_audit, root, record)
    previous = read(root / "manifest.json")
    previous_binding, raw, token = parent.bind_inputs(parent._bound_config(previous))
    equal(previous_binding, previous["binding"], "canonical fully byte-verified development ancestry")
    parent.frozen(previous, token)
    artifacts = parent._inventory(root)
    summary = read(root / "summary.json")
    equal(read(record), parent.public_record(root, summary, artifacts), "actual completed audited development publication")
    equal(summary["complete"], True, "all audited development cells complete")
    equal(Path(config.corpus).resolve(strict=True), Path(previous_binding["dataset"]["path"]), "same original corpus path")
    binding = {"prior_run": str(root), "prior_record": str(record), "prior_audit": str(parent.rate.head._regular(config.prior_audit)),
               "audit_receipt_sha256": AUDIT_RECEIPT_SHA256, "audit_capsule": capsule,
               "parent_binding": previous_binding, "dataset": previous_binding["dataset"],
               "parent_files": {name: old.file_identity(root / name) for name in ("manifest.json", "artifact_manifest.json", "summary.json")},
               "parent_public": old.file_identity(record), "parent_artifacts": artifacts}
    validation = _Validation(binding, sources, previous, token)
    frozen({"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "binding": binding}, validation)
    return binding, raw, validation


def frozen(manifest, validation):
    if not isinstance(validation, _Validation):
        raise ValueError("private invocation-local endpoint validation required")
    equal(manifest["schema"], SCHEMA, "endpoint schema")
    equal(manifest["recipe"], recipe(), "fixed sparse training diagnostic")
    equal(manifest["source_hashes"], source_identity(), "all71 source identities")
    equal(manifest["source_hashes"], validation.sources, "original invocation identities")
    equal(manifest["binding"], validation.binding, "canonical validated endpoint ancestry")
    binding = manifest["binding"]
    equal(AUDIT_RECEIPT_SHA256, binding["audit_receipt_sha256"], "unchanged successful audit admission pin")
    parent.frozen(validation.parent_manifest, validation.parent_validation)
    equal(parent._inventory(Path(binding["prior_run"])), binding["parent_artifacts"], "all development payloads unchanged")
    _, capsule = _audit_capsule(binding["prior_audit"], Path(binding["prior_run"]), Path(binding["prior_record"]))
    equal(capsule, binding["audit_capsule"], "complete audit proof unchanged")
    for descriptor in (*binding["parent_files"].values(), binding["parent_public"]):
        equal(old.file_identity(descriptor["path"]), descriptor, "actual parent raw/seal/public identity")


def _checkpoint(path, cell, arm, step):
    """Validate full safe tensor checkpoint; never instantiate an optimizer."""
    path = Path(path)
    values, metadata = load(path), read(path.with_suffix(".json"))
    payload = path.read_bytes()
    expected_keys = {"parameter/" + name for name in cell["parameters"]}
    if step:
        expected_keys |= {prefix + "/" + name for name in cell["parameters"] for prefix in ("step", "exp_avg", "exp_avg_sq")}
    equal(set(values), expected_keys, "complete saved full Adam checkpoint inventory")
    shapes = {}
    for name, parameter in cell["parameters"].items():
        value = values["parameter/" + name]
        helper._finite(value, "actual saved parameter", detached=True)
        if value.shape != parameter.shape:
            raise ValueError("actual saved parameter shape differs")
        shapes[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "initialized": step != 0}
        if step:
            counter = values["step/" + name]
            if (counter.device.type != "cpu" or counter.dtype not in (torch.float32, torch.float64)
                    or counter.shape != torch.Size([]) or not bool(torch.isfinite(counter)) or float(counter) != step):
                raise ValueError("exact endpoint Adam counter required")
            shapes[name]["step_dtype"] = str(counter.dtype)
            for key in ("exp_avg", "exp_avg_sq"):
                moment = values[key + "/" + name]
                helper._finite(moment, "saved endpoint moment", detached=True)
                if moment.shape != value.shape or (key == "exp_avg_sq" and bool((moment < 0).any())):
                    raise ValueError("complete finite endpoint moments required")
    group = metadata["optimizer_group"]
    expected_group = {"lr": 0.01 if arm == "mixed" else 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.0, "amsgrad": False, "maximize": False, "foreach": False,
                      "fused": False, "capturable": False, "differentiable": False}
    if "decoupled_weight_decay" in group:
        expected_group["decoupled_weight_decay"] = False
    equal(group, expected_group, "actual inherited endpoint optimizer options")
    parameters = {name: values["parameter/" + name] for name in cell["parameters"]}
    expected = {"schema": parent.SCHEMA, "arm": arm, "step": step, "file": path.name, "bytes": len(payload),
                "sha256": old.digest(payload), "parameter_order": list(cell["parameters"]), "parameter_state": shapes,
                "optimizer_group": group, "full_state_tensor_sha256": tensor_hash(values),
                "parameters_sha256": tensor_hash(parameters), "initial_optimizer_state_empty": step == 0}
    equal(metadata, expected, "complete original checkpoint metadata and tensor payload identity")
    return parameters, expected


@torch.no_grad()
def model(binding, arm, seed=None):
    if arm not in ARMS or (seed is not None and (type(seed) is not int or seed not in ORDER_SEEDS)):
        raise ValueError("one declared original/endpoint model required")
    cell = parent.initial_models(binding["parent_binding"])[arm]
    step, source_seed = (0, ORDER_SEEDS[0]) if seed is None else (parent.UPDATES, seed)
    path = Path(binding["prior_run"]) / parent.cell_name(arm, source_seed) / f"state_{step:06d}.npz"
    parameters, state = _checkpoint(path, cell, arm, step)
    if seed is None:
        guard.byte_equal(cell["parameters"], parameters, "exact original initial parameters")
    else:
        for name, parameter in cell["parameters"].items():
            parameter.copy_(parameters[name])
    guard.byte_equal(cell["parameters"], parameters, "exact safely imported endpoint parameter bytes")
    parent.parent._finite_parameters(cell)
    cell["endpoint_source"] = {"file": old.file_identity(path), "metadata": old.file_identity(path.with_suffix(".json")),
                               "full_state_tensor_sha256": state["full_state_tensor_sha256"], "step": step}
    return cell


def batch_data(raw, dataset, index):
    if type(index) is not int or index not in TRAIN_WINDOWS:
        raise ValueError("only the declared sparse training indices may be read")
    tokens, targets = pilot.window(raw, dataset, "train", index, SEQUENCE)
    if len(tokens) != SEQUENCE or len(targets) != SEQUENCE:
        raise ValueError("complete sampled training window required")
    return torch.tensor(tokens, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)


def baselines(raw, dataset, inherited):
    equal(inherited["alpha"], 1, "inherited alpha-one table")
    equal(inherited["train_indices"], list(parent.TRAIN_WINDOWS), "inherited complete fit window history")
    equal(inherited["training_pairs"], len(parent.TRAIN_WINDOWS) * SEQUENCE, "inherited count-fit history")
    counts = inherited["bigram_pair_counts"]
    if (not isinstance(counts, list) or len(counts) != VOCAB or any(not isinstance(row, list) or len(row) != VOCAB
            or any(type(value) is not int or value < 1 for value in row) for row in counts)):
        raise ValueError("complete positive inherited count table required")
    equal(sum(map(sum, counts)) - VOCAB * VOCAB, len(parent.TRAIN_WINDOWS) * SEQUENCE, "actual raw inherited total")
    result = {}
    for name, table in (("alpha_one", counts), ("repeat_alpha_one", [[2 * value - 1 for value in row] for row in counts])):
        totals = [sum(row) for row in table]
        predictions = [max(range(VOCAB), key=lambda index: row[index]) for row in table]
        rows = []
        for index in TRAIN_WINDOWS:
            tokens, targets = batch_data(raw, dataset, index)
            rows.append({"window": index,
                         "ce": math.fsum(math.log(totals[token]) - math.log(table[token][target]) for token, target in zip(tokens.tolist(), targets.tolist())) / SEQUENCE,
                         "accuracy": sum(predictions[token] == target for token, target in zip(tokens.tolist(), targets.tolist())) / SEQUENCE})
        result[name] = {"per_window": rows, "mean": parent._mean(rows), "new_model_count_fit": False}
    return result


@torch.no_grad()
def collect_scores(directory, cell, raw, dataset, *, write=False, recompute=False, storage=None, clock=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal forward-only modes required")
    before = tensor_hash(cell["parameters"])
    prior_before = None if cell["prior_logits"] is None else tensor_hash({"prior": cell["prior_logits"]})
    rows = []
    for position, index in enumerate(TRAIN_WINDOWS):
        if clock is not None:
            clock.check()
        tokens, targets = batch_data(raw, dataset, index)
        path = Path(directory) / f"score_{index:04d}.npz"
        if write:
            if cell["kind"] == "tropical":
                base = guard.guarded.forward(cell["parameters"], tokens, context=True)
                values = {**helper.trace(base), "guard_counts": torch.tensor([parent._guard_arrays(base)], dtype=torch.int64)}
            else:
                values = {"logits": cell["model"](tokens[None, :], cell["prior_logits"], context=True)[0]}
            storage.arrays(path, values)
        arrays = load(path)
        if cell["kind"] == "tropical":
            equal(set(arrays), set(helper.TRACE_KEYS) | {"guard_counts"}, "complete sampled tropical trace")
            trace = {key: arrays[key] for key in helper.TRACE_KEYS}
            parent.parent._base(trace, cell["parameters"], tokens, True)
            if arrays["guard_counts"].shape != (1, 6):
                raise ValueError("one complete sampled guard row required")
            parent._validate_guard_rows(arrays["guard_counts"].repeat(parent.BATCH_SIZE, 1), cell, SEQUENCE)
            if recompute:
                independent = helper.forward(cell["parameters"], tokens, context=True, independent=True)
                parent.rate.head.helper.near(trace, helper.trace(independent), "independent sampled dense forward")
                actual = guard.guarded.forward(cell["parameters"], tokens, context=True)
                guard.byte_equal(trace, helper.trace(actual), "exact supplemental sampled guarded trace")
                equal(arrays["guard_counts"][0].tolist(), parent._guard_arrays(actual), "exact sampled guarded counts")
        else:
            equal(set(arrays), {"logits"}, "complete sampled GRU logits")
            if arrays["logits"].shape != (SEQUENCE, cell["model"].vocab):
                raise ValueError("sampled GRU score shape differs")
            if recompute:
                independent = cell["model"](tokens[None, :], cell["prior_logits"], context=True, independent=True)[0]
                parent.rate.head.helper.near(arrays["logits"], independent, "independent sampled expanded GRU")
        rows.append({"window": index, **parent.parent.metrics(arrays["logits"], targets),
                     "trace_sha256": tensor_hash(arrays), "file_sha256": old.digest(path.read_bytes()),
                     "guard_counts": arrays["guard_counts"].tolist() if cell["kind"] == "tropical" else None})
        if (position + 1) % 16 == 0:
            print(json.dumps({"stage": "endpoint_scores", "directory": Path(directory).name,
                              "windows_complete": position + 1, "recompute": recompute}), flush=True)
    equal(tensor_hash(cell["parameters"]), before, "all endpoint weights unchanged by readouts")
    if prior_before is not None:
        equal(tensor_hash({"prior": cell["prior_logits"]}), prior_before, "frozen GRU prior unchanged")
    if any(parameter.grad is not None for parameter in cell["parameters"].values()):
        raise ValueError("forward-only endpoint readout produced gradients")
    if "endpoint_source" in cell:
        for key in ("file", "metadata"):
            descriptor = cell["endpoint_source"][key]
            equal(old.file_identity(descriptor["path"]), descriptor, "actual complete checkpoint unchanged after scoring")
    return rows


def _development_rows(binding, arm, seed=None):
    root = Path(binding["prior_run"])
    summary = read(root / "summary.json")
    if seed is None:
        saved = summary["shared_initial"]["architectures"][arm]
        directory, label = root / ("shared_" + arm), "initial"
        rows, mean = saved["scores"], saved["mean"]
    else:
        matches = [cell for cell in summary["cells"] if cell["arm"] == arm and cell["order_seed"] == seed]
        if len(matches) != 1:
            raise ValueError("exactly one bound final development cell required")
        rows, mean = matches[0]["final_scores"], matches[0]["final_mean"]
        directory, label = root / parent.cell_name(arm, seed), "final"
    equal([row["window"] for row in rows], list(parent.DEV_WINDOWS), "all saved original development windows")
    equal(parent._mean(rows), mean, "actual saved development row means")
    for row in rows:
        path = directory / f"score_{label}_{row['window']:03d}_1.npz"
        equal(old.digest(path.read_bytes()), row["file_sha256"], "actual saved development score file identity")
    return {"per_window": rows, "mean": mean, "new_forward_evaluations": 0}


def execute(manifest, output, raw, *, write=False, recompute=False, storage=None, clock=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal endpoint modes required")
    output, binding = Path(output), manifest["binding"]
    clock = clock or Deadline()
    if write and storage is None:
        storage = Storage(output, clock)
        storage.used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    inherited = read(binding["parent_binding"]["capacity"]["identities"]["counts.json"]["path"])
    baseline = baselines(raw, binding["dataset"], inherited)
    if write:
        storage.json(output / "baselines.json", baseline)
    equal(read(output / "baselines.json"), baseline, "same inherited-count sampled baseline entries")
    initial, cells = {}, []
    for arm in ARMS:
        cell = model(binding, arm)
        directory = output / ("initial_" + arm)
        if write:
            directory.mkdir(exist_ok=False)
        rows = collect_scores(directory, cell, raw, binding["dataset"], write=write, recompute=recompute, storage=storage, clock=clock)
        initial[arm] = {"train_scores": rows, "train_mean": parent._mean(rows),
                        "development": _development_rows(binding, arm), "source_state": cell["endpoint_source"]}
    probability_error = 0.0
    for index in TRAIN_WINDOWS:
        probabilities = [load(output / ("initial_" + arm) / f"score_{index:04d}.npz")["logits"].softmax(-1) for arm in ARMS]
        if probabilities[0].shape != probabilities[1].shape or not torch.allclose(probabilities[0], probabilities[1], atol=1e-12, rtol=1e-12):
            raise ValueError("shared original probabilities differ on sampled training windows")
        probability_error = max(probability_error, float((probabilities[0] - probabilities[1]).abs().max()))
    for seed in ORDER_SEEDS:
        for arm in ARMS:
            cell = model(binding, arm, seed)
            directory = output / parent.cell_name(arm, seed)
            if write:
                directory.mkdir(exist_ok=False)
            rows = collect_scores(directory, cell, raw, binding["dataset"], write=write, recompute=recompute, storage=storage, clock=clock)
            train, dev = parent._mean(rows), _development_rows(binding, arm, seed)
            train0, dev0 = initial[arm]["train_mean"], initial[arm]["development"]["mean"]
            train_improvement, dev_improvement = train0["ce"] - train["ce"], dev0["ce"] - dev["mean"]["ce"]
            gap0, gap = dev0["ce"] - train0["ce"], dev["mean"]["ce"] - train["ce"]
            if not math.isclose(gap - gap0, train_improvement - dev_improvement, abs_tol=1e-12, rel_tol=0):
                raise ValueError("relative train/development gap identity differs")
            cells.append({"arm": arm, "order_seed": seed, "complete": True, "source_state": cell["endpoint_source"],
                          "train_scores": rows, "train_mean": train, "development": dev,
                          "train_improvement_from_initial": train_improvement, "development_improvement_from_initial": dev_improvement,
                          "initial_development_minus_train": gap0, "final_development_minus_train": gap,
                          "development_minus_train_change": gap - gap0,
                          "train_advantage_over_baselines": {name: value["mean"]["ce"] - train["ce"] for name, value in baseline.items()}})
    paired = []
    for seed in ORDER_SEEDS:
        left, right = (next(cell for cell in cells if cell["arm"] == arm and cell["order_seed"] == seed) for arm in ARMS)
        paired.append({"order_seed": seed, "tropical_minus_gru_sampled_train_ce": left["train_mean"]["ce"] - right["train_mean"]["ce"],
                       "tropical_minus_gru_saved_development_ce": left["development"]["mean"]["ce"] - right["development"]["mean"]["ce"]})
    def spread(values):
        return {"mean": math.fsum(values) / len(values), "minimum": min(values), "maximum": max(values), "range": max(values) - min(values)}
    aggregate = {}
    for arm in ARMS:
        rows = [cell for cell in cells if cell["arm"] == arm]
        aggregate[arm] = {"sampled_train_ce": spread([row["train_mean"]["ce"] for row in rows]),
                          "saved_development_ce": spread([row["development"]["mean"]["ce"] for row in rows]),
                          **{key: spread([row[key] for row in rows]) for key in
                             ("train_improvement_from_initial", "development_improvement_from_initial", "development_minus_train_change")}}
    paired_aggregate = {key: spread([row[key] for row in paired]) for key in
                        ("tropical_minus_gru_sampled_train_ce", "tropical_minus_gru_saved_development_ce")}
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "initial": initial, "cells": cells, "paired_comparisons": paired, "baselines": baseline,
            "initial_probabilities_equal": True, "initial_probability_max_abs_error": probability_error,
            "descriptive_order_aggregates": aggregate, "descriptive_paired_aggregates": paired_aggregate,
            "spread_scope": "mean/minimum/maximum/range across the same three order replicas, not iid or significance",
            "model_window_readouts": len(TRAIN_WINDOWS) * len(ARMS) * (1 + len(ORDER_SEEDS)),
            "baseline_window_readouts": 2 * len(TRAIN_WINDOWS), "new_gradients": 0, "optimizer_steps": 0,
            "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_model_count_fit": False,
            "new_development_evaluations": 0, "heldout_test_evaluated": False, "reserved_development_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "descriptive deterministic sampled-training endpoint losses and changes relative to inherited initialization; not whole-training fit or convergence proof"}


def expected_artifacts():
    names = {"manifest.json", "protocol.json", "declaration.md", "baselines.json", "summary.json"}
    for name in ["initial_" + arm for arm in ARMS] + [parent.cell_name(arm, seed) for seed in ORDER_SEEDS for arm in ARMS]:
        names.update(f"{name}/score_{index:04d}.npz" for index in TRAIN_WINDOWS)
    return names


def _inventory(root):
    root = Path(root)
    artifacts = prior.artifact_manifest(root)
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact complete forward-only endpoint inventory")
    directories = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}
    equal(directories, {"initial_" + arm for arm in ARMS} | {parent.cell_name(arm, seed) for arm in ARMS for seed in ORDER_SEEDS},
          "only declared readout directories")
    equal(read(root / "artifact_manifest.json"), {"schema": SCHEMA, "files": artifacts}, "exact sealed endpoint payloads")
    return artifacts


def public_record(root, result, artifacts):
    return {**result, "run_directory": str(root), "runtime": read(Path(root) / "manifest.json")["runtime"],
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "independent_forward_audit_required": True,
            "check_scope": "strict existing data/state/trace/metric verification; separate audit independently rederives same sparse training forwards"}


def _bound_config(manifest):
    from types import SimpleNamespace
    binding = manifest["binding"]
    return SimpleNamespace(prior_run=Path(binding["prior_run"]), prior_record=Path(binding["prior_record"]),
                           prior_audit=Path(binding["prior_audit"]), corpus=Path(binding["dataset"]["path"]))


def _check(root, record, recompute, validation, raw, clock):
    if type(recompute) is not bool:
        raise ValueError("literal independent readout mode required")
    root = Path(root).resolve(strict=True)
    record = None if record is None else parent.rate.head._regular(record)
    originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
    public_bytes = None if record is None else record.read_bytes()
    manifest = read(root / "manifest.json")
    equal(manifest["run_directory"], str(root), "actual endpoint run path")
    equal(manifest["runtime"], parent.runtime(), "exact endpoint runtime")
    equal(read(root / "protocol.json"), recipe(), "actual saved sparse-window protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["declaration_sha256"], "actual saved declaration")
    frozen(manifest, validation)
    artifacts = _inventory(root)
    result = execute(manifest, root, raw, recompute=recompute, clock=clock)
    equal(read(root / "summary.json"), result, "complete recomputed endpoint summary")
    published = public_record(root, result, artifacts)
    if record is not None:
        equal(read(record), published, "actual endpoint publication")
    frozen(manifest, validation)
    equal(_inventory(root), artifacts, "endpoint payloads unchanged during verification")
    for name, payload in originals.items():
        equal((root / name).read_bytes(), payload, "exact original endpoint metadata bytes")
    if record is not None:
        equal(record.read_bytes(), public_bytes, "exact original endpoint public bytes")
    clock.check()
    return published


def _failure(output, clock, stage, error):
    if output is not None and output.is_dir() and not (output / "failure.json").exists():
        utc, monotonic = clock.elapsed()
        rejected = output / "rejected_public.json"
        retained = old.file_identity(rejected) if rejected.is_file() and not rejected.is_symlink() else None
        pilot.write_json(output / "failure.json", {"schema": SCHEMA, "complete": False, "stage": stage,
            "type": type(error).__name__, "error": str(error)[:100000], "retained_rejected_public": retained,
            "started_at_utc": guard.parent._utc(clock.started_utc), "failure_observed_at_utc": guard.parent._utc(time.time()),
            "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic,
            "new_training_updates": 0, "optimizer_steps": 0, "recipe_promoted": False})


def _invoke(config, independent):
    clock, output, publication = Deadline(), None, None
    created = closed = cleanup_done = False
    stage = "setup"
    try:
        clock.arm()
        output, target = parent._new_outputs(config.audit_output, config.audit_record) if independent else parent._new_outputs(config.output, config.record)
        sources, declaration = source_identity(), PLAN.read_bytes()
        output.mkdir(parents=True, exist_ok=False)
        created = True
        storage = Storage(output, clock)
        torch.set_num_threads(1)
        stage = "input_validation"
        print(json.dumps({"stage": stage, "independent_forward_audit": independent, "new_model_readouts": 0}), flush=True)
        if independent:
            root, record = Path(config.output).resolve(strict=True), parent.rate.head._regular(config.record)
            manifest = read(root / "manifest.json")
            originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
            public_bytes = record.read_bytes()
            artifacts = _inventory(root)
            storage.base_bytes = sum(row["bytes"] for row in artifacts) + len(originals["artifact_manifest.json"]) + len(public_bytes)
            storage.used = storage.base_bytes
            binding, raw, validation = bind_inputs(_bound_config(manifest))
            equal(binding, manifest["binding"], "fresh independent endpoint input binding")
            private = {"schema": SCHEMA, "complete": False, "mode": "independent_forward_audit",
                       "run_directory": str(root), "record": str(record), "audit_directory": str(output), "audit_record": str(target),
                       "producer_public_sha256": old.digest(public_bytes), "producer_seal_sha256": old.digest(originals["artifact_manifest.json"]),
                       "source_hashes": sources, "runtime": parent.runtime(), "started_at_utc": guard.parent._utc(clock.started_utc),
                       "deadline_seconds": DEADLINE_S, "declaration_sha256": old.digest(declaration)}
            storage.json(output / "audit_manifest.json", private)
            private_bytes = (output / "audit_manifest.json").read_bytes()
        else:
            root, record = output, target
            binding, raw, validation = bind_inputs(config)
            manifest = {"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "binding": binding,
                        "run_directory": str(root), "record": str(target), "runtime": parent.runtime(),
                        "started_at_utc": guard.parent._utc(clock.started_utc), "declaration_sha256": old.digest(declaration)}
            storage.json(output / "manifest.json", manifest)
            storage.json(output / "protocol.json", recipe())
        equal(source_identity(), sources, "no source relabeling during initial validation")
        storage.reserve(output / "declaration.md", len(declaration))
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        storage.used += len(declaration)
        frozen(manifest, validation)
        clock.check()
        stage = "independent_forward_audit" if independent else "fixed_endpoint_readouts"
        if independent:
            result = _check(root, record, True, validation, raw, clock)
            utc, monotonic = clock.check()
            per_arm = len(TRAIN_WINDOWS) * (1 + len(ORDER_SEEDS))
            published = {"schema": SCHEMA + ".audit", "complete": True, "independent_forward_recompute_completed": True,
                         "run_directory": str(root), "record": str(record), "audit_directory": str(output),
                         "producer_public_sha256": old.digest(public_bytes), "producer_seal_sha256": old.digest(originals["artifact_manifest.json"]),
                         "audit_manifest_sha256": old.digest(private_bytes), "source_hashes": sources, "runtime": parent.runtime(),
                         "started_at_utc": guard.parent._utc(clock.started_utc),
                         "verification_completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                         "verification_elapsed_utc_seconds": utc, "verification_elapsed_monotonic_seconds": monotonic,
                         "deadline_seconds": DEADLINE_S, "independent_dense_tropical_forwards": per_arm,
                         "independent_expanded_gru_forwards": per_arm, "supplemental_guarded_forwards": per_arm,
                         "model_window_readouts": result["model_window_readouts"], "baseline_window_readouts": result["baseline_window_readouts"],
                         "source_input_raw_seal_public_bytes_unchanged": True, "new_gradients": 0, "optimizer_steps": 0,
                         "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_model_count_fit": False,
                         "new_development_evaluations": 0, "heldout_test_evaluated": False, "reserved_development_evaluated": False,
                         "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
                         "scope": "independent forward-only reconstruction of fixed sampled-training endpoints; no new learning or convergence claim"}
            storage.json(output / "verification.json", published)
            verification_bytes = (output / "verification.json").read_bytes()
            equal(read(output / "verification.json"), published, "actual independent verification result")
        else:
            result = execute(manifest, output, raw, write=True, storage=storage, clock=clock)
            storage.json(output / "summary.json", result)
            artifacts = prior.artifact_manifest(root)
            equal({row["file"] for row in artifacts}, expected_artifacts(), "all fixed endpoint scores before sealing")
            storage.json(root / "artifact_manifest.json", {"schema": SCHEMA, "files": artifacts})
            originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
            stage = "ordinary_readback"
            published = _check(root, None, False, validation, raw, clock)
            equal(published["artifact_manifest_sha256"], old.digest(originals["artifact_manifest.json"]), "public binds original exact seal")
        storage.reconcile()
        stage = "publication"
        publication = parent._Publication(target, output, storage)
        publication.publish(published)
        frozen(manifest, validation)
        equal(_inventory(root), artifacts, "all endpoint bytes unchanged during publication")
        for name, payload in originals.items():
            equal((root / name).read_bytes(), payload, "original endpoint metadata unchanged during publication")
        if independent:
            equal(record.read_bytes(), public_bytes, "original endpoint publication unchanged during audit")
            equal((output / "audit_manifest.json").read_bytes(), private_bytes, "exact private audit manifest")
            equal((output / "verification.json").read_bytes(), verification_bytes, "exact private audit result")
            equal(read(output / "verification.json"), published, "private/public audit result agreement")
            equal({path.name for path in output.iterdir()}, {"audit_manifest.json", "declaration.md", "verification.json"}, "exact successful endpoint audit capsule")
        equal((output / "declaration.md").read_bytes(), declaration, "saved endpoint declaration unchanged")
        publication.verify()
        clock.check()
        publication.cleanup()
        cleanup_done = True
        clock.close()
        closed = True
        utc, monotonic = clock.check()
        print(json.dumps({"stage": "complete", "independent_forward_audit": independent,
                          "completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                          "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic,
                          "record_sha256": publication.digest}), flush=True)
        return published
    except BaseException as error:
        if publication is not None:
            publication.retract()
        if created:
            _failure(output, clock, stage, error)
        raise
    finally:
        try:
            if publication is not None and not cleanup_done:
                publication.cleanup()
        finally:
            if not closed:
                clock.close()


def run(config):
    return _invoke(config, False)


def audit(config):
    return _invoke(config, True)


def check(output, record):
    clock = Deadline()
    try:
        clock.arm()
        torch.set_num_threads(1)
        root = Path(output).resolve(strict=True)
        manifest = read(root / "manifest.json")
        binding, raw, validation = bind_inputs(_bound_config(manifest))
        equal(binding, manifest["binding"], "fresh canonical endpoint checker binding")
        result = _check(root, record, False, validation, raw, clock)
        clock.check()
        return result
    finally:
        clock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("run", "check", "audit"):
        modes.add_argument("--" + name, action="store_true")
    for name in ("output", "record"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("prior-run", "prior-record", "prior-audit", "corpus", "audit-output", "audit-record"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.run and any(getattr(args, key) is None for key in ("prior_run", "prior_record", "prior_audit", "corpus")):
        parser.error("--run requires prior run/record/successful audit and original corpus")
    if args.audit and (args.audit_output is None or args.audit_record is None):
        parser.error("--audit requires exclusive audit output/record paths")
    result = run(args) if args.run else audit(args) if args.audit else check(args.output, args.record)
    print(json.dumps({"complete": result["complete"], "optimizer_steps": 0, "recipe_promoted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
