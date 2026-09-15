"""Fixed six-cell varied-text development comparison; CPU FP64 only.

One pass through the same1024 training windows per cell. Initializations and
historical audits are immutable; independent checking rederives all queries
but applies SAVED gradients to reconstruct exact complete Adam states.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import sys
import tempfile
import threading
import time

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import verify_fpga_lm_guarded_rate as guard  # noqa: E402

rate, parent, helper = guard.rate, guard.rate.parent, guard.helper
old, prior, pilot = rate.old, rate.prior, rate.pilot
read, equal, load, tensor_hash = rate.read, rate.equal, rate.load, rate.tensor_hash
SCHEMA = "tikitorch.fpga_lm_development.v1"
ARMS, ORDER_SEEDS = ("mixed", "prior_gru"), (0, 1, 2)
TRAIN_WINDOWS, DEV_WINDOWS = tuple(range(1024)), tuple(range(88, 152))
BATCH_SIZE, UPDATES, SEQUENCE, VOCAB = 4, 256, 128, 128
DEADLINE_S, CELL_DEADLINE_S = 10800, 3600
STORAGE_CAP, FREE_RESERVE = 12 * (1 << 30), 4 * (1 << 30)
GATE_TOLERANCE = 1e-12
PLAN = REPO / "manuscript/research/PLAN.md"
GUARD_RECEIPT_SHA256 = "7a21df2543ece923fbdcd388ae326770db0aa0ad6623d1ff578c505ae4ae5a82"
# Filled only from the actual retained completed capacity evidence before freeze.
CAPACITY_PINS = {"counts.json": "d61f20b3108fe6443a388f27ac6469859219e1965126afc4c6d43d6bba605244",
                 "manifest.json": "8268966aa461bd5a7ede14b8363c262337034f2f0ceb56c75d872d8ee383a318",
                 "artifact_manifest.json": "eb6605c83590eeb4019ade6164a2d79a6051004a00b33d9c427a0b80fe3f195b",
                 "public": "027b012522fafce2309a4bfb5d548ebd1f5e1dd0921f3628dcc3b501f62e9dc9"}
CAPACITY_ROOT = REPO / "manuscript/build/fpga_lm_bigram_capacity_2026-09-09/v1"
CAPACITY_RECORD = REPO / "manuscript/evidence/fpga_lm_bigram_capacity_2026-09-09.json"


def source_identity():
    values = dict(guard.source_identity())
    for name in ("manuscript/tools/compare_fpga_lm_development.py", "tests/test_fpga_lm_development.py"):
        values[name] = old.digest((REPO / name).read_bytes())
    return values


def training_order(seed):
    if type(seed) is not int or seed not in ORDER_SEEDS:
        raise ValueError("declared literal order seed required")
    values = list(TRAIN_WINDOWS)
    random.Random(seed).shuffle(values)
    return values


def cell_name(arm, seed):
    if arm not in ARMS or type(seed) is not int or seed not in ORDER_SEEDS:
        raise ValueError("one declared architecture/order pair required")
    return f"{arm}_seed_{seed}"


def recipe():
    cells = len(ARMS) * len(ORDER_SEEDS)
    return {"schema": SCHEMA, "arms": list(ARMS), "order_seeds": list(ORDER_SEEDS),
            "training_orders": {str(seed): training_order(seed) for seed in ORDER_SEEDS},
            "train_windows": list(TRAIN_WINDOWS), "development_windows": list(DEV_WINDOWS),
            "reserved_development_windows": [152, 863], "test_policy": "reserved, never evaluated",
            "batch_size": BATCH_SIZE, "sequence": SEQUENCE, "updates_per_cell": UPDATES,
            "initialization_seed": 0, "seed_meaning": "order-only; identical fitted initialization and empty Adam",
            "initialization": "original fitted mixed tropical and zero-correction GRU; not previously trained endpoints",
            "dtype": "float64", "device": "cpu", "torch_threads": 1,
            "loss": "exact mean CE over all batch positions, dimensionless logits; no second beta",
            "optimizer": {"mixed_lr": 0.01, "prior_gru_lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
                          "weight_decay": 0.0, "foreach": False, "fused": False, "gradient_clipping": False},
            "tropical": "guarded current-winner forward with exact dense fallback; unchanged dense selected-route VJP",
            "gru": "frozen embedding128/hidden128/layer1 recipe with externally frozen fitted log prior",
            "audit": "independent dense tropical selector/autograd and explicit GRU cells; SAVED G drives exact Adam replay",
            "baseline": "inherited alpha1 counts C; repeat-count alpha1=2*C-1; no new model count fit",
            "declared_cells": cells, "full_batch_gradient_queries": cells * UPDATES,
            "optimizer_commits": cells * UPDATES, "complete_saved_states": cells * (UPDATES + 1),
            "shared_initial_readouts": len(ARMS) * len(DEV_WINDOWS),
            "final_readouts": cells * 2 * len(DEV_WINDOWS),
            "score_window_readouts": len(ARMS) * len(DEV_WINDOWS) + cells * 2 * len(DEV_WINDOWS),
            "baseline_window_readouts": 2 * len(DEV_WINDOWS), "development_evaluated": True,
            "reserved_development_evaluated": False, "heldout_test_evaluated": False,
            "new_gradient_pair_exposures": cells * len(TRAIN_WINDOWS) * SEQUENCE,
            "distinct_gradient_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
            "new_fitting_pair_exposures": 0,
            "summed_inherited_fitting_pair_exposures": cells * len(TRAIN_WINDOWS) * SEQUENCE,
            "gate": {"each_tropical_improves_own_initial": True, "mean_tropical_below_stronger_bigram_nats": 0.02,
                     "each_tropical_at_most_paired_gru_plus_nats": 0.05, "inclusive_tolerance": GATE_TOLERANCE},
            "global_deadline_seconds": DEADLINE_S, "cell_deadline_seconds": CELL_DEADLINE_S,
            "storage_cap_bytes": STORAGE_CAP, "free_reserve_bytes": FREE_RESERVE,
            "adaptive_extension": False, "early_loss_stopping": False, "new_native_training_calls": 0,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False}


class Deadline(guard.parent.Deadline):
    def __init__(self):
        super().__init__()
        self.cell_started = None

    def check(self):
        elapsed = self.elapsed()
        if not all(math.isfinite(value) for value in elapsed) or min(elapsed) < 0:
            raise RuntimeError("development invocation clock moved backwards")
        if max(elapsed) >= DEADLINE_S:
            raise TimeoutError("global development deadline exceeded")
        if self.cell_started is not None:
            cell = (time.time() - self.cell_started[0], time.monotonic() - self.cell_started[1])
            if not all(math.isfinite(value) for value in cell) or min(cell) < 0 or max(cell) >= CELL_DEADLINE_S:
                raise TimeoutError("development cell deadline exceeded")
        return elapsed

    def arm(self):
        self.check()
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        self.installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S - max(self.elapsed()))
        self.watcher = threading.Thread(target=self._watch, name="development-deadline", daemon=True)
        self.watcher.start()

    def _expired(self, signum, frame):
        raise TimeoutError("declared development invocation/cell deadline exceeded")

    def begin_cell(self):
        self.check()
        self.cell_started = (time.time(), time.monotonic())

    def end_cell(self):
        self.check()
        self.cell_started = None


class Storage:
    """Account owned bytes, guard each write's conservative uncompressed bound."""
    def __init__(self, root, clock):
        self.root, self.clock, self.used = Path(root), clock, 0
        self.allowed_roots = [self.root.resolve()]
        self.base_bytes, self.external_paths = 0, []

    def reconcile(self):
        seen, size = set(), self.base_bytes
        paths = [path for root in self.allowed_roots for path in root.rglob("*") if path.is_file()]
        paths.extend(path for path in self.external_paths if path.exists())
        for path in paths:
            if path.is_symlink():
                raise ValueError("no owned output symlinks")
            stat = path.stat()
            identity = stat.st_dev, stat.st_ino
            if identity not in seen:
                size += stat.st_size
                seen.add(identity)
        if size != self.used or size > STORAGE_CAP:
            raise RuntimeError("actual aggregate output bytes differ from bounded write accounting")
        return size

    def reserve(self, path, worst_case):
        self.clock.check()
        path = Path(path)
        if not any(root in path.resolve().parents for root in self.allowed_roots):
            raise ValueError("write outside explicitly owned output roots")
        if path.exists() or path.is_symlink():
            raise FileExistsError("append-only artifact required")
        if type(worst_case) is not int or worst_case < 0 or self.used + worst_case + (1 << 20) > STORAGE_CAP:
            raise RuntimeError("declared storage cap would be exceeded")
        if shutil.disk_usage(path.parent).free - worst_case < FREE_RESERVE:
            raise RuntimeError("declared free-space reserve would be violated")

    def json(self, path, value):
        payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        self.reserve(path, len(payload))
        with Path(path).open("xb") as stream:
            stream.write(payload)
        self.used += len(payload)

    def arrays(self, path, values):
        # Deflate worst-case overhead plus NPY/ZIP names/headers, per member.
        bound = sum(value.numel() * value.element_size() + value.numel() * value.element_size() // 1000
                    + 8192 + 4 * len(name.encode()) for name, value in values.items()) + 65536
        self.reserve(path, bound)
        parent.save(path, values)
        size = Path(path).stat().st_size
        if size > bound:
            raise RuntimeError("NPZ exceeded declared conservative bound")
        self.used += size


def batch_data(raw, dataset, indices, partition="train"):
    allowed = TRAIN_WINDOWS if partition == "train" else DEV_WINDOWS if partition == "validation" else ()
    if not indices or any(type(index) is not int or index not in allowed for index in indices):
        raise ValueError("only the declared training/development windows may be read")
    pairs = [pilot.window(raw, dataset, partition, index, SEQUENCE) for index in indices]
    tokens = torch.tensor([pair[0] for pair in pairs], dtype=torch.int64)
    targets = torch.tensor([pair[1] for pair in pairs], dtype=torch.int64)
    if tokens.shape != (len(indices), SEQUENCE) or targets.shape != tokens.shape:
        raise ValueError("complete aligned fixed-length windows required")
    return tokens, targets


def baselines(raw, dataset, inherited_counts):
    if type(inherited_counts.get("alpha")) is not int or inherited_counts["alpha"] != 1:
        raise ValueError("actual inherited alpha-one table required")
    equal(inherited_counts["training_pairs"], len(TRAIN_WINDOWS) * SEQUENCE, "inherited count exposure budget")
    equal(inherited_counts["train_indices"], list(TRAIN_WINDOWS), "exact inherited count windows")
    counts = inherited_counts["bigram_pair_counts"]
    if (not isinstance(counts, list) or len(counts) != VOCAB or any(not isinstance(row, list) or len(row) != VOCAB
            or any(type(value) is not int or value < 1 for value in row) for row in counts)):
        raise ValueError("complete positive literal-integer inherited count table required")
    equal(sum(sum(row) for row in counts) - VOCAB * VOCAB, len(TRAIN_WINDOWS) * SEQUENCE,
          "raw inherited count total excluding pseudo-counts")
    # Reconstruct only supervised integer counts to verify ancestry. This is
    # not a new model count fit or an observation on old validation cohorts.
    observed = torch.ones((VOCAB, VOCAB), dtype=torch.int64)
    for start in range(0, len(TRAIN_WINDOWS), BATCH_SIZE):
        tokens, targets = batch_data(raw, dataset, TRAIN_WINDOWS[start:start + BATCH_SIZE])
        observed += torch.bincount((tokens * VOCAB + targets).reshape(-1), minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB)
    guard.byte_equal({"counts": observed}, {"counts": torch.tensor(counts, dtype=torch.int64)}, "actual inherited training counts")
    result = {}
    for name, table in (("alpha_one", counts), ("repeat_alpha_one", [[2 * value - 1 for value in row] for row in counts])):
        totals = [sum(row) for row in table]
        predictions = [max(range(VOCAB), key=lambda index: row[index]) for row in table]
        rows = []
        for index in DEV_WINDOWS:
            tokens, targets = batch_data(raw, dataset, (index,), "validation")
            losses = [math.log(totals[token]) - math.log(table[token][target]) for token, target in zip(tokens[0].tolist(), targets[0].tolist())]
            rows.append({"window": index, "ce": sum(losses) / len(losses),
                         "accuracy": sum(predictions[token] == target for token, target in zip(tokens[0].tolist(), targets[0].tolist())) / len(losses)})
        result[name] = {"per_window": rows, "mean": parent.mean_metrics([{key: row[key] for key in ("ce", "accuracy")} for row in rows]),
                        "effective_count_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE * (1 if name == "alpha_one" else 2),
                        "distinct_count_pairs": len(TRAIN_WINDOWS) * SEQUENCE, "new_model_count_fit": False}
    result["stronger_ce"] = min(result[name]["mean"]["ce"] for name in ("alpha_one", "repeat_alpha_one"))
    result["new_model_fitting_pair_exposures"] = 0
    return result


def _capacity(dataset):
    root, record = Path(CAPACITY_ROOT).resolve(strict=True), rate.head._regular(CAPACITY_RECORD)
    identities = {}
    for key, expected in CAPACITY_PINS.items():
        path = record if key == "public" else root / key
        identity = old.file_identity(rate.head._regular(path))
        equal(identity["sha256"], expected, "hard-pinned inherited capacity: " + key)
        identities[key] = identity
    equal(set(identities), {"counts.json", "manifest.json", "artifact_manifest.json", "public"}, "all capacity trust pins")
    manifest, seal, published = read(root / "manifest.json"), read(root / "artifact_manifest.json"), read(record)
    artifacts = prior.artifact_manifest(root)
    equal(seal, {"schema": manifest["schema"], "files": artifacts}, "complete immutable capacity inventory")
    equal(manifest["dataset"], dataset, "capacity inherited count corpus/splits")
    equal(published["complete"], True, "completed original capacity control")
    equal(published["run_directory"], str(root), "actual capacity root")
    equal(published["artifact_manifest_sha256"], identities["artifact_manifest.json"]["sha256"], "capacity public seal")
    sources = source_identity()
    for name, digest in manifest["source_hashes"].items():
        equal(sources[name], digest, "unchanged inherited capacity source")
    return {"root": str(root), "record": str(record), "identities": identities, "artifacts": artifacts,
            "source_hashes": manifest["source_hashes"]}


def _guard_capsule(receipt_path, root, record):
    receipt_path = rate.head._regular(receipt_path)
    equal(old.digest(receipt_path.read_bytes()), GUARD_RECEIPT_SHA256, "actual successful guarded parity receipt pin")
    value = read(receipt_path)
    for key, wanted in {"schema": guard.SCHEMA, "complete": True, "guarded_parity_replay_completed": True,
                        "rate_run_directory": str(root), "rate_public_record": str(record),
                        "source_hashes": guard.source_identity(), "independent_autograd_audit_reused": True,
                        "source_input_raw_seal_public_bytes_unchanged": True,
                        "new_training_updates": 0, "new_optimizer_trajectory_executed": False,
                        "paid_execution_authorized": False, "recipe_promoted": False,
                        "publication_postchecks_within_deadline": True, "deadline_seconds": 3600}.items():
        equal(value.get(key), wanted, "completed guarded parity proof: " + key)
    capsule = Path(value["audit_directory"])
    if capsule.is_symlink() or not capsule.is_dir():
        raise ValueError("actual successful guarded audit capsule required")
    capsule = capsule.resolve(strict=True)
    names = {"audit_manifest.json", "declaration.md", "verification.json", "parity.json"}
    equal({path.name for path in capsule.iterdir()}, names, "exact four-file successful guarded capsule")
    equal(set(value["private_files"]), names, "all four guarded payload descriptors")
    descriptors = {name: old.file_identity(rate.head._regular(capsule / name)) for name in names}
    equal(descriptors, value["private_files"], "actual guarded capsule payload bytes")
    equal(read(capsule / "verification.json"), value["verification"], "actual guard verification facts")
    equal(read(capsule / "parity.json"), value["parity"], "actual complete guard replay payload")
    audit_manifest = read(capsule / "audit_manifest.json")
    equal(descriptors["audit_manifest.json"]["sha256"], value["audit_manifest_sha256"], "guard private manifest pin")
    for key, wanted in {"schema": guard.SCHEMA, "complete": False, "run_directory": str(root), "record": str(record),
                        "audit_directory": str(capsule), "receipt": str(receipt_path),
                        "source_hashes": guard.source_identity(), "deadline_seconds": 3600,
                        "declaration_sha256": descriptors["declaration.md"]["sha256"]}.items():
        equal(audit_manifest[key], wanted, "actual guarded declaration: " + key)
    for key, count in {"guarded_window_forwards": 524, "manual_window_vjps": 512,
                       "full_batch_gradients_verified": 128, "optimizer_commits_replayed": 128,
                       "complete_saved_states_verified": 129, "score_readouts_verified": 12}.items():
        equal(value["parity"][key], count, "completed guarded numerical coverage")
    return value, {"receipt": old.file_identity(receipt_path), "files": descriptors}


@dataclass(frozen=True)
class _Validation:
    binding: dict
    sources: dict
    rate_manifest: dict
    rate_validation: object
    guard_facts: dict

    def __post_init__(self):
        for name in ("binding", "sources", "rate_manifest", "guard_facts"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))


def bind_inputs(config):
    sources = source_identity()
    root, record = Path(config.prior_run).resolve(strict=True), rate.head._regular(config.prior_record)
    audited, capsule = _guard_capsule(config.guard_receipt, root, record)
    previous, old_raw, token, facts = guard.validate_inputs(root, record,
        read(Path(audited["audit_directory"]) / "audit_manifest.json")["successful_audit_receipt"])
    equal(audited["verification"], facts, "successful guard proof versus complete fresh ancestry validation")
    raw, dataset = pilot.corpus_plan(Path(config.corpus), len(TRAIN_WINDOWS), max(DEV_WINDOWS) + 1, SEQUENCE)
    equal(raw, old_raw, "same immutable corpus bytes")
    equal(dataset, previous["binding"]["dataset"], "same immutable corpus/splits")
    capacity = _capacity(dataset)
    initial_root = Path(previous["binding"]["prior_run"])
    binding = {"prior_run": str(root), "prior_record": str(record), "guard_receipt": str(rate.head._regular(config.guard_receipt)),
               "guard_capsule": capsule, "guard_receipt_sha256": GUARD_RECEIPT_SHA256,
               "rate_binding": previous["binding"], "dataset": dataset, "capacity": capacity,
               "initial_states": {arm: old.file_identity(initial_root / arm / "state_000000.npz") for arm in ARMS},
               "frozen_prior": old.file_identity(initial_root / "frozen_prior.npz")}
    validation = _Validation(binding, sources, previous, token, facts)
    frozen({"schema": SCHEMA, "recipe": recipe(), "binding": binding, "source_hashes": sources}, validation)
    return binding, raw, validation


def frozen(manifest, validation):
    if not isinstance(validation, _Validation):
        raise ValueError("private invocation-local input validation required")
    equal(manifest["schema"], SCHEMA, "new development schema")
    equal(manifest["recipe"], recipe(), "fixed development recipe")
    equal(manifest["source_hashes"], source_identity(), "all69current sources")
    equal(manifest["source_hashes"], validation.sources, "original invocation source identity")
    equal(manifest["binding"], validation.binding, "canonical complete immutable binding")
    binding = manifest["binding"]
    guard._unchanged(Path(binding["prior_run"]), Path(binding["prior_record"]), validation.rate_manifest,
                     validation.rate_validation, validation.guard_facts)
    _, capsule = _guard_capsule(binding["guard_receipt"], Path(binding["prior_run"]), Path(binding["prior_record"]))
    equal(capsule, binding["guard_capsule"], "unchanged trusted guard capsule")
    equal(_capacity(binding["dataset"]), binding["capacity"], "all capacity provenance unchanged")
    for descriptor in (*binding["initial_states"].values(), binding["frozen_prior"]):
        equal(old.file_identity(descriptor["path"]), descriptor, "actual initial model identity")


def initial_models(binding):
    models = parent.initial_models(binding["rate_binding"]["prior_binding"])
    result = {arm: models[arm] for arm in ARMS}
    for arm, cell in result.items():
        descriptor = binding["initial_states"][arm]
        equal(old.file_identity(descriptor["path"]), descriptor, "actual original initial checkpoint")
        guard.byte_equal({"parameter/" + name: value for name, value in cell["parameters"].items()},
                         load(descriptor["path"]), "exact original parameters and empty Adam state")
    guard.byte_equal({"logits": result["prior_gru"]["prior_logits"]}, load(binding["frozen_prior"]["path"]),
                     "exact inherited frozen GRU prior")
    return result


def make_optimizer(arm, parameters):
    if arm not in ARMS:
        raise ValueError("declared optimizer arm required")
    optimizer = helper.make_optimizer(parameters)
    optimizer.param_groups[0]["lr"] = 0.01 if arm == "mixed" else 0.001
    return optimizer


def state_record(directory, step, cell, optimizer, arm, *, write=False, storage=None):
    if type(step) is not int or not 0 <= step <= UPDATES or type(write) is not bool or arm not in ARMS:
        raise ValueError("literal declared state step/arm/write required")
    parent._finite_parameters(cell)
    group = (rate._optimizer_group if arm == "mixed" else parent._optimizer_group)(cell, optimizer)
    if set(optimizer.state) - set(cell["parameters"].values()):
        raise ValueError("unowned optimizer state")
    arrays, shapes = {}, {}
    for name, parameter in cell["parameters"].items():
        arrays["parameter/" + name] = parameter.detach()
        moment = optimizer.state.get(parameter, {})
        equal(set(moment), set() if step == 0 else {"step", "exp_avg", "exp_avg_sq"}, "complete actual Adam state")
        shapes[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype), "initialized": step != 0}
        if step:
            counter = moment["step"]
            if (counter.device.type != "cpu" or counter.dtype not in (torch.float32, torch.float64)
                    or counter.shape != torch.Size([]) or not bool(torch.isfinite(counter)) or float(counter) != step):
                raise ValueError("exact actual Adam scalar counter required")
            for key in ("exp_avg", "exp_avg_sq"):
                value = moment[key]
                helper._finite(value, "Adam moment", detached=True)
                if value.shape != parameter.shape or (key == "exp_avg_sq" and bool((value < 0).any())):
                    raise ValueError("complete bounded Adam moments required")
                arrays[key + "/" + name] = value
            arrays["step/" + name] = counter
            shapes[name]["step_dtype"] = str(counter.dtype)
    path = Path(directory) / f"state_{step:06d}.npz"
    if write:
        storage.arrays(path, arrays)
    guard.byte_equal(load(path), arrays, "every actual parameter/moment/counter byte")
    payload = path.read_bytes()
    result = {"schema": SCHEMA, "arm": arm, "step": step, "file": path.name, "bytes": len(payload),
              "sha256": old.digest(payload), "parameter_order": list(cell["parameters"]), "parameter_state": shapes,
              "optimizer_group": group, "full_state_tensor_sha256": tensor_hash(arrays),
              "parameters_sha256": tensor_hash(cell["parameters"]), "initial_optimizer_state_empty": step == 0}
    if write:
        storage.json(path.with_suffix(".json"), result)
    equal(read(path.with_suffix(".json")), result, "complete typed Adam metadata")
    return result


def _guard_arrays(base):
    value = guard._guards(base)
    return [int(value["structure_certified"]), int(value["whole_model_dense_fallback"]), *[value[key] for key in guard.GUARD_KEYS]]


def _validate_guard_rows(rows, cell, length):
    if rows.dtype != torch.int64 or rows.shape != (BATCH_SIZE, 6):
        raise ValueError("complete integer producer guard counters required")
    n = cell["parameters"]["MH"].shape[0]
    for row in rows.tolist():
        if min(row) < 0 or row[0] not in (0, 1) or row[1] != 1 - row[0]:
            raise ValueError("consistent producer guard flags required")
        if row[2] + row[3] != n * length or row[4] + row[5] != length or (row[1] and (row[2] or row[4])):
            raise ValueError("complete consistent producer guard counts required")


def query_batch(directory, step, cell, tokens, targets, indices, *, write=False, recompute=False, storage=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal query modes required")
    if type(step) is not int or not 0 <= step < UPDATES or len(indices) != BATCH_SIZE:
        raise ValueError("declared full batch and step required")
    parent._finite_parameters(cell)
    path = Path(directory) / f"query_{step:06d}.npz"
    if write:
        arrays = {"tokens": tokens, "targets": targets, "window_indices": torch.tensor(indices, dtype=torch.int64)}
        if cell["kind"] == "tropical":
            gradients = {name: torch.zeros_like(value) for name, value in cell["parameters"].items()}
            guards = []
            for row, index in enumerate(indices):
                base = guard.guarded.forward(cell["parameters"], tokens[row])
                values = helper.gradient(cell["parameters"], tokens[row], targets[row], base, targets.numel())
                for name in gradients:
                    gradients[name] = gradients[name] + values[name]
                arrays.update({f"window_{index}/" + name: value for name, value in helper.trace(base).items()})
                guards.append(_guard_arrays(base))
            arrays["guard_counts"] = torch.tensor(guards, dtype=torch.int64)
        else:
            arrays["logits"], gradients = parent._gru_query(cell, tokens, targets)
        parent._gradients(cell, gradients)
        arrays.update({"g_" + name: value for name, value in gradients.items()})
        storage.arrays(path, arrays)
    arrays = load(path)
    expected = {"tokens", "targets", "window_indices"} | {"g_" + name for name in cell["parameters"]}
    expected |= ({f"window_{index}/" + name for index in indices for name in helper.TRACE_KEYS} | {"guard_counts"}
                 if cell["kind"] == "tropical" else {"logits"})
    equal(set(arrays), expected, "complete declared batch query inventory")
    guard.byte_equal({name: arrays[name] for name in ("tokens", "targets", "window_indices")},
                     {"tokens": tokens, "targets": targets, "window_indices": torch.tensor(indices, dtype=torch.int64)},
                     "actual ordered training pairs")
    gradients = {name: arrays["g_" + name] for name in cell["parameters"]}
    parent._gradients(cell, gradients)
    metrics = []
    if cell["kind"] == "tropical":
        _validate_guard_rows(arrays["guard_counts"], cell, tokens.shape[1])
        independent = {name: torch.zeros_like(value) for name, value in gradients.items()}
        for row, index in enumerate(indices):
            trace = {name: arrays[f"window_{index}/" + name] for name in helper.TRACE_KEYS}
            parent._base(trace, cell["parameters"], tokens[row], True)
            if recompute:
                base = helper.forward(cell["parameters"], tokens[row], independent=True)
                rate.head.helper.near(trace, helper.trace(base), "independent dense development query")
                values = helper.gradient(cell["parameters"], tokens[row], targets[row], base, targets.numel(), independent=True)
                for name in values:
                    independent[name] = independent[name] + values[name]
                guarded = guard.guarded.forward(cell["parameters"], tokens[row])
                guard.byte_equal(trace, helper.trace(guarded), "supplemental exact guarded query replay")
                equal(arrays["guard_counts"][row].tolist(), _guard_arrays(guarded), "actual recomputed guard query counters")
            metrics.append({"window": index, **parent.metrics(trace["logits"], targets[row])})
        if recompute:
            rate.head.helper.near(gradients, independent, "independent full tropical batch G")
    else:
        if arrays["logits"].shape != (*tokens.shape, cell["model"].vocab):
            raise ValueError("complete GRU batch logits required")
        if recompute:
            logits, independent = parent._gru_query(cell, tokens, targets, independent=True)
            rate.head.helper.near(arrays["logits"], logits, "independent expanded GRU batch")
            rate.head.helper.near(gradients, independent, "independent expanded GRU batch G")
        metrics = [{"window": index, **parent.metrics(arrays["logits"][row], targets[row])} for row, index in enumerate(indices)]
    return gradients, {"step": step, "window_indices": list(indices), "per_window": metrics,
                       "gradient_sha256": tensor_hash(gradients), "query_sha256": old.digest(path.read_bytes()),
                       "guard_counts": arrays["guard_counts"].tolist() if cell["kind"] == "tropical" else None}


def collect_scores(directory, label, cell, raw, dataset, context, *, write=False, recompute=False, storage=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal score modes required")
    if type(context) is not bool or label not in ("initial", "final"):
        raise ValueError("declared score label/context required")
    parent._finite_parameters(cell)
    rows = []
    for index in DEV_WINDOWS:
        tokens, targets = batch_data(raw, dataset, (index,), "validation")
        path = Path(directory) / f"score_{label}_{index:03d}_{int(context)}.npz"
        if write:
            if cell["kind"] == "tropical":
                base = guard.guarded.forward(cell["parameters"], tokens[0], context=context)
                arrays = {**helper.trace(base), "guard_counts": torch.tensor([_guard_arrays(base)], dtype=torch.int64)}
            else:
                with torch.no_grad():
                    arrays = {"logits": cell["model"](tokens, cell["prior_logits"], context=context)[0]}
            storage.arrays(path, arrays)
        arrays = load(path)
        if cell["kind"] == "tropical":
            equal(set(arrays), set(helper.TRACE_KEYS) | {"guard_counts"}, "complete score trace and guards")
            trace = {name: arrays[name] for name in helper.TRACE_KEYS}
            parent._base(trace, cell["parameters"], tokens[0], context)
            # The same counter invariants, repeated only to satisfy a full batch
            # shape, avoid changing the frozen guarded helper's contract.
            if arrays["guard_counts"].shape != (1, 6):
                raise ValueError("one score-window guard row required")
            _validate_guard_rows(arrays["guard_counts"].repeat(BATCH_SIZE, 1), cell, tokens.shape[1])
            if recompute:
                base = helper.forward(cell["parameters"], tokens[0], context=context, independent=True)
                rate.head.helper.near(trace, helper.trace(base), "independent dense development score")
                guarded = guard.guarded.forward(cell["parameters"], tokens[0], context=context)
                guard.byte_equal(trace, helper.trace(guarded), "supplemental exact guarded score replay")
                equal(arrays["guard_counts"][0].tolist(), _guard_arrays(guarded), "actual recomputed score guard counters")
        else:
            equal(set(arrays), {"logits"}, "complete GRU score array")
            if arrays["logits"].shape != (tokens.shape[1], cell["model"].vocab):
                raise ValueError("complete GRU score geometry")
            if recompute:
                with torch.no_grad():
                    logits = cell["model"](tokens, cell["prior_logits"], context=context, independent=True)[0]
                rate.head.helper.near(arrays["logits"], logits, "independent expanded GRU score")
        rows.append({"window": index, **parent.metrics(arrays["logits"], targets[0]),
                     "trace_sha256": tensor_hash(arrays), "file_sha256": old.digest(path.read_bytes()),
                     "guard_counts": arrays["guard_counts"].tolist() if cell["kind"] == "tropical" else None})
    return rows


def _mean(rows):
    return parent.mean_metrics([{key: row[key] for key in ("ce", "accuracy")} for row in rows])


def gates(cells, baseline):
    expected = [(arm, seed) for seed in ORDER_SEEDS for arm in ARMS]
    if not isinstance(cells, list) or len(cells) != len(expected):
        raise ValueError("all six architecture/order cells required before gates")
    equal([(row["arm"], row["order_seed"]) for row in cells], expected, "exact ordered cell membership")
    for row in cells:
        if row.get("complete") is not True or row.get("numerics_pass") is not True or type(row["order_seed"]) is not int:
            raise ValueError("complete literal healthy cell required before gates")
        parent.mean_metrics([row["initial_mean"], row["final_mean"], row["context_off_mean"]])
    by_id = {(row["arm"], row["order_seed"]): row for row in cells}
    scores = [by_id["mixed", seed]["final_mean"]["ce"] for seed in ORDER_SEEDS]
    comparator = baseline["stronger_ce"]
    if type(comparator) is not float or not math.isfinite(comparator) or comparator < 0:
        raise ValueError("finite exact stronger baseline required")
    mean = sum(scores) / len(scores)
    improves = {str(seed): by_id["mixed", seed]["final_mean"]["ce"] < by_id["mixed", seed]["initial_mean"]["ce"]
                for seed in ORDER_SEEDS}
    paired = {str(seed): by_id["mixed", seed]["final_mean"]["ce"] <= by_id["prior_gru", seed]["final_mean"]["ce"] + 0.05 + GATE_TOLERANCE
              for seed in ORDER_SEEDS}
    bigram = mean <= comparator - 0.02 + GATE_TOLERANCE
    return {"all_numerics_pass": True, "tropical_mean_ce": mean, "stronger_bigram_ce": comparator,
            "improves_initial_per_order": improves, "paired_gru_margin_pass_per_order": paired,
            "mean_bigram_margin_pass": bigram, "development_gate_candidate": all(improves.values()) and all(paired.values()) and bigram,
            "requires_successful_separate_independent_audit": True,
            "recipe_promoted": False, "paid_execution_authorized": False,
            "scope": "fixed development screen, reused development cohort and three order replicas; not final-test generalization or significance"}


def execute(manifest, output, raw, *, write=False, recompute=False, clock=None, storage=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal execution modes required")
    equal(UPDATES * BATCH_SIZE, len(TRAIN_WINDOWS), "exact one-pass update/window budget")
    output, binding = Path(output), manifest["binding"]
    clock = clock or Deadline()
    if write and storage is None:
        storage = Storage(output, clock)
        storage.used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    counts = read(binding["capacity"]["identities"]["counts.json"]["path"])
    baseline = baselines(raw, binding["dataset"], counts)
    if write:
        storage.json(output / "baselines.json", baseline)
    equal(read(output / "baselines.json"), baseline, "exact inherited-count development baselines")
    models = initial_models(binding)
    prior_values = {"logits": models["prior_gru"]["prior_logits"]}
    if write:
        storage.arrays(output / "frozen_prior.npz", prior_values)
    guard.byte_equal(load(output / "frozen_prior.npz"), prior_values, "actual immutable inherited GRU prior")
    shared, maximum_error = {}, 0.0
    for arm in ARMS:
        clock.check()
        directory = output / ("shared_" + arm)
        if write:
            directory.mkdir(exist_ok=False)
        rows = collect_scores(directory, "initial", models[arm], raw, binding["dataset"], True,
                              write=write, recompute=recompute, storage=storage)
        shared[arm] = {"parameters_sha256": tensor_hash(models[arm]["parameters"]), "scores": rows, "mean": _mean(rows)}
    for index in DEV_WINDOWS:
        probabilities = [load(output / ("shared_" + arm) / f"score_initial_{index:03d}_1.npz")["logits"].softmax(-1) for arm in ARMS]
        if probabilities[0].shape != probabilities[1].shape or not torch.allclose(probabilities[0], probabilities[1], atol=1e-12, rtol=1e-12):
            raise ValueError("original model initial probabilities differ")
        maximum_error = max(maximum_error, float((probabilities[0] - probabilities[1]).abs().max()))
    shared_record = {"schema": SCHEMA, "architectures": shared, "initial_probabilities_equal": True,
                     "maximum_probability_error": maximum_error}
    if write:
        storage.json(output / "shared_initial.json", shared_record)
    equal(read(output / "shared_initial.json"), shared_record, "exact shared initial readouts")
    del models
    cells = []
    for seed in ORDER_SEEDS:
        for arm in ARMS:
            clock.begin_cell()
            name = cell_name(arm, seed)
            print(json.dumps({"cell": name, "stage": "starting", "recompute": recompute, "write": write}), flush=True)
            directory = output / name
            if write:
                directory.mkdir(exist_ok=False)
            cell = initial_models(binding)[arm]
            equal(tensor_hash(cell["parameters"]), shared[arm]["parameters_sha256"], "identical order-only initialization")
            optimizer = make_optimizer(arm, cell["parameters"].values())
            states = [state_record(directory, 0, cell, optimizer, arm, write=write, storage=storage)]
            order, rows = training_order(seed), []
            before_prior = None if cell["prior_logits"] is None else cell["prior_logits"].detach().clone()
            for step in range(UPDATES):
                clock.check()
                indices = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
                tokens, targets = batch_data(raw, binding["dataset"], indices)
                before = tensor_hash(cell["parameters"])
                gradient, row = query_batch(directory, step, cell, tokens, targets, indices,
                                            write=write, recompute=recompute, storage=storage)
                # Independent G is compared inside query_batch; the returned
                # G is ALWAYS the saved payload, preserving exact Adam replay.
                optimizer.zero_grad(set_to_none=True)
                for key, parameter in cell["parameters"].items():
                    parameter.grad = gradient[key].detach().clone()
                optimizer.step()
                parent._finite_parameters(cell)
                states.append(state_record(directory, step + 1, cell, optimizer, arm, write=write, storage=storage))
                if before_prior is not None:
                    guard.byte_equal({"prior": cell["prior_logits"]}, {"prior": before_prior}, "frozen external GRU prior")
                row.update(before_parameters_sha256=before, after_parameters_sha256=tensor_hash(cell["parameters"]),
                           state_sha256=states[-1]["full_state_tensor_sha256"], saved_gradient_adam_replay=True)
                rows.append(row)
                clock.check()
                if (step + 1) % 16 == 0:
                    print(json.dumps({"cell": name, "step": step + 1, "query_ce": _mean(row["per_window"])["ce"],
                                      "recompute": recompute}), flush=True)
            final = collect_scores(directory, "final", cell, raw, binding["dataset"], True,
                                   write=write, recompute=recompute, storage=storage)
            disabled = collect_scores(directory, "final", cell, raw, binding["dataset"], False,
                                      write=write, recompute=recompute, storage=storage)
            result = {"schema": SCHEMA, "arm": arm, "order_seed": seed, "name": name, "complete": True,
                      "numerics_pass": True, "training_order": order, "initial_mean": shared[arm]["mean"],
                      "shared_initial_parameters_sha256": shared[arm]["parameters_sha256"],
                      "final_scores": final, "final_mean": _mean(final), "context_off_scores": disabled,
                      "context_off_mean": _mean(disabled), "context_ce_penalty": _mean(disabled)["ce"] - _mean(final)["ce"],
                      "optimizer_commits": UPDATES, "complete_saved_states": UPDATES + 1,
                      "full_batch_gradient_queries": UPDATES, "query_rows": rows, "states": states,
                      "new_gradient_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
                      "inherited_fitting_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
                      "endpoint_history_pair_exposures": 2 * len(TRAIN_WINDOWS) * SEQUENCE,
                      "distinct_fitted_pairs": len(TRAIN_WINDOWS) * SEQUENCE, "new_model_count_fit": False,
                      "guard_count_scope": "ordinary complete membership checks; audit supplements independent dense selectors with exact guarded trace/counter replay"}
            if write:
                storage.json(directory / "summary.json", result)
            equal(read(directory / "summary.json"), result, "complete actual development cell")
            cells.append(result)
            clock.end_cell()
            print(json.dumps({"cell": name, "stage": "complete", "final_ce": result["final_mean"]["ce"], "recompute": recompute}), flush=True)
    count = len(ARMS) * len(ORDER_SEEDS)
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "shared_initial": shared_record, "baselines": baseline, "cells": cells, "gates": gates(cells, baseline),
            "completed_cells": count, "full_batch_gradient_queries": count * UPDATES,
            "optimizer_commits": count * UPDATES, "complete_saved_states": count * (UPDATES + 1),
            "score_window_readouts": len(DEV_WINDOWS) * (len(ARMS) + 2 * count),
            "baseline_window_readouts": 2 * len(DEV_WINDOWS),
            "new_gradient_pair_exposures": count * len(TRAIN_WINDOWS) * SEQUENCE,
            "distinct_gradient_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
            "summed_inherited_fitting_pair_exposures": count * len(TRAIN_WINDOWS) * SEQUENCE,
            "summed_endpoint_history_pair_exposures": 2 * count * len(TRAIN_WINDOWS) * SEQUENCE,
            "new_fitting_pair_exposures": 0, "new_native_training_calls": 0,
            "development_evaluated": True, "reserved_development_evaluated": False, "heldout_test_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "independent_audit_required": True, "statistical_significance_claim": False,
            "scope": "fixed development comparison with order-only replicas and inherited fitting history; not final-test or hardware evidence"}


def expected_artifacts():
    names = {"manifest.json", "protocol.json", "declaration.md", "baselines.json", "frozen_prior.npz", "shared_initial.json", "summary.json"}
    for arm in ARMS:
        names.update(f"shared_{arm}/score_initial_{index:03d}_1.npz" for index in DEV_WINDOWS)
        for seed in ORDER_SEEDS:
            name = cell_name(arm, seed)
            names.add(name + "/summary.json")
            names.update(f"{name}/query_{step:06d}.npz" for step in range(UPDATES))
            names.update(f"{name}/state_{step:06d}.{suffix}" for step in range(UPDATES + 1) for suffix in ("npz", "json"))
            names.update(f"{name}/score_final_{index:03d}_{context}.npz" for index in DEV_WINDOWS for context in (0, 1))
    return names


def runtime():
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": parent.np.__version__,
            "device": "cpu", "torch_threads": 1}


def _bound_config(manifest):
    from types import SimpleNamespace
    binding = manifest["binding"]
    return SimpleNamespace(prior_run=Path(binding["prior_run"]), prior_record=Path(binding["prior_record"]),
                           guard_receipt=Path(binding["guard_receipt"]), corpus=Path(binding["dataset"]["path"]))


def _inventory(root):
    artifacts = prior.artifact_manifest(root)
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact complete development payload inventory")
    directories = {path.relative_to(root).as_posix() for path in Path(root).rglob("*") if path.is_dir()}
    equal(directories, {"shared_" + arm for arm in ARMS} | {cell_name(arm, seed) for arm in ARMS for seed in ORDER_SEEDS},
          "no undeclared output directories")
    equal(read(Path(root) / "artifact_manifest.json"), {"schema": SCHEMA, "files": artifacts}, "exact sealed development bytes")
    return artifacts


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root), "runtime": read(Path(root) / "manifest.json")["runtime"],
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "check_scope": "ordinary all-state saved-gradient replay; separate independent dense/autograd/expanded-GRU audit required; development not test"}


def _check(root, record, recompute, validation, raw, clock):
    root = Path(root).resolve(strict=True)
    record = None if record is None else rate.head._regular(record)
    if type(recompute) is not bool:
        raise ValueError("literal audit mode required")
    manifest_bytes, seal_bytes = (root / "manifest.json").read_bytes(), (root / "artifact_manifest.json").read_bytes()
    public_bytes = None if record is None else record.read_bytes()
    manifest = read(root / "manifest.json")
    equal(manifest["run_directory"], str(root), "actual development run root")
    equal(manifest["runtime"], runtime(), "exact declared runtime versions and CPU thread setting")
    equal(read(root / "protocol.json"), recipe(), "actual saved protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["declaration_sha256"], "saved declaration hash")
    frozen(manifest, validation)
    artifacts = _inventory(root)
    result = execute(manifest, root, raw, recompute=recompute, clock=clock)
    equal(result, read(root / "summary.json"), "complete rederived development summary")
    published = public_record(root, result, artifacts)
    if record is not None:
        equal(read(record), published, "actual unchanged development publication")
    frozen(manifest, validation)
    equal(_inventory(root), artifacts, "all development payloads unchanged during replay")
    equal((root / "manifest.json").read_bytes(), manifest_bytes, "manifest bytes unchanged")
    equal((root / "artifact_manifest.json").read_bytes(), seal_bytes, "seal bytes unchanged")
    if record is not None:
        equal(record.read_bytes(), public_bytes, "public bytes unchanged")
    clock.check()
    return published


class _Publication:
    def __init__(self, target, output, storage):
        self.target, self.output, self.storage = Path(target), Path(output), storage
        self.directory = self.staged = self.inode = self.digest = None

    def publish(self, value):
        self.directory = Path(tempfile.mkdtemp(prefix=".tt-development-publication-", dir=self.target.parent))
        self.storage.allowed_roots.append(self.directory.resolve())
        self.staged = self.directory / "receipt.json"
        self.storage.json(self.staged, value)
        self.digest = old.digest(self.staged.read_bytes())
        stat = self.staged.stat()
        self.inode = stat.st_dev, stat.st_ino
        self.storage.external_paths.append(self.target)
        self.storage.clock.check()
        os.link(self.staged, self.target)

    def verify(self):
        equal(old.digest(rate.head._regular(self.target).read_bytes()), self.digest, "exact newly published record bytes")
        self.storage.reconcile()

    def retract(self):
        if self.inode is not None and self.target.is_file() and not self.target.is_symlink():
            stat = self.target.stat()
            if (stat.st_dev, stat.st_ino) == self.inode:
                rejected = self.output / "rejected_public.json"
                if rejected.exists():
                    raise RuntimeError("cannot overwrite rejected publication")
                self.target.rename(rejected)

    def cleanup(self):
        if self.staged is not None and self.staged.exists():
            self.staged.unlink()
        if self.directory is not None and self.directory.exists():
            self.directory.rmdir()


def _new_outputs(output, record):
    output, record = guard._new_path(output), guard._new_path(record)
    if output == record or output in record.parents or record in output.parents:
        raise ValueError("public record must be outside exclusive private output")
    if not record.parent.is_dir():
        raise FileNotFoundError("public record parent must already exist")
    return output, record


def _failure(output, clock, stage, error):
    # One MiB is reserved by every normal write for retained failure metadata.
    if output is not None and output.is_dir() and not (output / "failure.json").exists():
        utc, monotonic = clock.elapsed()
        rejected = output / "rejected_public.json"
        retained = old.file_identity(rejected) if rejected.is_file() and not rejected.is_symlink() else None
        pilot.write_json(output / "failure.json", {"schema": SCHEMA, "complete": False, "stage": stage,
            "type": type(error).__name__, "error": str(error)[:100000],
            "started_at_utc": guard.parent._utc(clock.started_utc), "failure_observed_at_utc": guard.parent._utc(time.time()),
            "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic,
            "retained_rejected_public": retained, "recipe_promoted": False, "paid_execution_authorized": False})


def run(config):
    clock, output, publication = Deadline(), None, None
    created = closed = cleanup_done = False
    stage = "setup"
    try:
        clock.arm()
        output, record = _new_outputs(config.output, config.record)
        sources, declaration = source_identity(), PLAN.read_bytes()
        output.mkdir(parents=True, exist_ok=False)
        created = True
        storage = Storage(output, clock)
        stage = "input_validation"
        print(json.dumps({"stage": stage, "new_model_readouts": 0}), flush=True)
        binding, raw, validation = bind_inputs(config)
        equal(sources, source_identity(), "no source relabeling during binding")
        torch.set_num_threads(1)
        manifest = {"schema": SCHEMA, "run_directory": str(output), "record": str(record), "recipe": recipe(),
                    "source_hashes": sources, "binding": binding, "runtime": runtime(),
                    "started_at_utc": guard.parent._utc(clock.started_utc), "declaration_sha256": old.digest(declaration)}
        storage.json(output / "manifest.json", manifest)
        storage.json(output / "protocol.json", recipe())
        storage.reserve(output / "declaration.md", len(declaration))
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        storage.used += len(declaration)
        frozen(manifest, validation)
        clock.check()
        stage = "producer"
        result = execute(manifest, output, raw, write=True, clock=clock, storage=storage)
        storage.json(output / "summary.json", result)
        artifacts = prior.artifact_manifest(output)
        equal({row["file"] for row in artifacts}, expected_artifacts(), "all six cells before sealing")
        storage.json(output / "artifact_manifest.json", {"schema": SCHEMA, "files": artifacts})
        storage.reconcile()
        stage = "ordinary_saved_gradient_replay"
        seal_bytes = (output / "artifact_manifest.json").read_bytes()
        published = _check(output, None, False, validation, raw, clock)
        equal(published["artifact_manifest_sha256"], old.digest(seal_bytes), "public record binds original exact seal")
        stage = "publication"
        publication = _Publication(record, output, storage)
        publication.publish(published)
        frozen(manifest, validation)
        equal(_inventory(output), artifacts, "all source-bound producer bytes unchanged through publication")
        equal((output / "artifact_manifest.json").read_bytes(), seal_bytes, "exact producer seal bytes unchanged through publication")
        publication.verify()
        clock.check()
        publication.cleanup()
        cleanup_done = True
        clock.close()
        closed = True
        utc, monotonic = clock.check()
        print(json.dumps({"stage": "complete", "completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                          "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic,
                          "independent_audit_completed": False, "record_sha256": publication.digest}), flush=True)
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


def check(output, record):
    clock = Deadline()
    try:
        clock.arm()
        torch.set_num_threads(1)
        root = Path(output).resolve(strict=True)
        manifest = read(root / "manifest.json")
        binding, raw, validation = bind_inputs(_bound_config(manifest))
        equal(binding, manifest["binding"], "fresh canonical checker input binding")
        result = _check(root, record, False, validation, raw, clock)
        clock.check()
        return result
    finally:
        clock.close()


def audit(config):
    clock, output, publication = Deadline(), None, None
    created = closed = cleanup_done = False
    stage = "setup"
    try:
        clock.arm()
        output, target = _new_outputs(config.audit_output, config.audit_record)
        root, record = Path(config.output).resolve(strict=True), rate.head._regular(config.record)
        sources, declaration = source_identity(), PLAN.read_bytes()
        output.mkdir(parents=True, exist_ok=False)
        created = True
        storage = Storage(output, clock)
        # Charge retained NEW producer/public bytes again only as aggregate
        # storage occupancy, not as new writes or gradient exposures.
        producer_artifacts = _inventory(root)
        storage.base_bytes = sum(row["bytes"] for row in producer_artifacts) + (root / "artifact_manifest.json").stat().st_size + record.stat().st_size
        storage.used = storage.base_bytes
        manifest, public_bytes, seal_bytes = read(root / "manifest.json"), record.read_bytes(), (root / "artifact_manifest.json").read_bytes()
        stage = "input_validation"
        binding, raw, validation = bind_inputs(_bound_config(manifest))
        equal(binding, manifest["binding"], "fresh canonical independent audit binding")
        equal(source_identity(), sources, "no source relabeling during audit validation")
        torch.set_num_threads(1)
        private = {"schema": SCHEMA, "complete": False, "mode": "independent_audit", "source_hashes": sources,
                   "run_directory": str(root), "record": str(record), "audit_directory": str(output), "audit_record": str(target),
                   "producer_public_sha256": old.digest(public_bytes), "producer_seal_sha256": old.digest(seal_bytes),
                   "runtime": runtime(), "started_at_utc": guard.parent._utc(clock.started_utc),
                   "deadline_seconds": DEADLINE_S, "cell_deadline_seconds": CELL_DEADLINE_S,
                   "declaration_sha256": old.digest(declaration)}
        storage.json(output / "audit_manifest.json", private)
        storage.reserve(output / "declaration.md", len(declaration))
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        storage.used += len(declaration)
        private_bytes = (output / "audit_manifest.json").read_bytes()
        stage = "independent_numerical_replay"
        print(json.dumps({"stage": stage, "old_numerical_trajectories_replayed": False}), flush=True)
        result = _check(root, record, True, validation, raw, clock)
        equal(record.read_bytes(), public_bytes, "original producer public bytes unchanged")
        equal((root / "artifact_manifest.json").read_bytes(), seal_bytes, "original producer seal bytes unchanged")
        utc, monotonic = clock.check()
        trop_windows = len(ORDER_SEEDS) * UPDATES * BATCH_SIZE
        score_per_arm = len(DEV_WINDOWS) * (1 + 2 * len(ORDER_SEEDS))
        receipt = {"schema": SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                   "run_directory": str(root), "record": str(record), "audit_directory": str(output),
                   "producer_public_sha256": old.digest(public_bytes), "producer_seal_sha256": old.digest(seal_bytes),
                   "audit_manifest_sha256": old.digest(private_bytes), "source_hashes": sources, "runtime": runtime(),
                   "started_at_utc": guard.parent._utc(clock.started_utc),
                   "verification_completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                   "verification_elapsed_utc_seconds": utc, "verification_elapsed_monotonic_seconds": monotonic,
                   "deadline_seconds": DEADLINE_S, "cell_deadline_seconds": CELL_DEADLINE_S,
                   "independent_tropical_window_vjps": trop_windows,
                   "independent_tropical_window_forwards": trop_windows + score_per_arm,
                   "independent_gru_batch_queries": len(ORDER_SEEDS) * UPDATES,
                   "independent_gru_single_window_scores": score_per_arm,
                   "supplemental_guarded_window_forwards": trop_windows + score_per_arm,
                   "full_batch_gradients_verified": result["full_batch_gradient_queries"],
                   "complete_saved_states_verified": result["complete_saved_states"],
                   "optimizer_commits_replayed": result["optimizer_commits"],
                   "score_window_readouts": result["score_window_readouts"], "baseline_window_readouts": result["baseline_window_readouts"],
                   "development_gates": result["gates"], "source_input_raw_seal_public_bytes_unchanged": True,
                   "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_model_count_fit": False,
                   "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                   "reserved_development_evaluated": False, "physical_fpga_execution": False,
                   "paid_execution_authorized": False, "recipe_promoted": False,
                   "scope": "independent numerical audit of fixed development comparison, not new learning or final-test validation"}
        storage.json(output / "verification.json", receipt)
        verification_bytes = (output / "verification.json").read_bytes()
        stage = "publication"
        publication = _Publication(target, output, storage)
        publication.publish(receipt)
        frozen(manifest, validation)
        equal(_inventory(root), producer_artifacts, "complete producer bytes unchanged during audit publication")
        equal(record.read_bytes(), public_bytes, "producer public unchanged through audit publication")
        equal((root / "artifact_manifest.json").read_bytes(), seal_bytes, "producer seal unchanged through audit publication")
        equal((output / "audit_manifest.json").read_bytes(), private_bytes, "private audit identity unchanged")
        equal((output / "declaration.md").read_bytes(), declaration, "saved audit declaration unchanged")
        equal((output / "verification.json").read_bytes(), verification_bytes, "actual audit verification unchanged")
        equal(read(output / "verification.json"), receipt, "actual private/public verification agreement")
        equal({path.name for path in output.iterdir()}, {"audit_manifest.json", "declaration.md", "verification.json"}, "exact audit capsule inventory")
        publication.verify()
        clock.check()
        publication.cleanup()
        cleanup_done = True
        clock.close()
        closed = True
        utc, monotonic = clock.check()
        print(json.dumps({"stage": "complete", "independent_recompute_completed": True,
                          "completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                          "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic,
                          "receipt_sha256": publication.digest}), flush=True)
        return receipt
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("run", "check", "audit"):
        modes.add_argument("--" + name, action="store_true")
    for name in ("output", "record"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("prior-run", "prior-record", "guard-receipt", "corpus", "audit-output", "audit-record"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.run and any(getattr(args, name) is None for name in ("prior_run", "prior_record", "guard_receipt", "corpus")):
        parser.error("--run requires prior run/record, guard receipt and corpus")
    if args.audit and (args.audit_output is None or args.audit_record is None):
        parser.error("--audit requires exclusive --audit-output and --audit-record")
    result = run(args) if args.run else audit(args) if args.audit else check(args.output, args.record)
    print(json.dumps({"complete": result["complete"], "recipe_promoted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
