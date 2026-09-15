"""Three declared, separately invoked pair-coordinate routing controls.

Physical projection pairs are m+d and m-d. This is ordinary FP64 coordinate
Adam, not an integer recipe or a claim of matched physical update scales.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_channel_choice as channel  # noqa: E402

core, continuation, endpoint, parent = channel.core, channel.continuation, channel.endpoint, channel.parent
guard, helper, old, prior, pilot = channel.guard, channel.helper, channel.old, channel.prior, channel.pilot
read, equal, load, tensor_hash = channel.read, channel.equal, channel.load, channel.tensor_hash
SCHEMA = "tikitorch.fpga_lm_routing_contrasts.v1"
ARMS = ("contrast", "contrast_slow", "contrast_mean")
BODY = ("EMBED", "MH", "W_HEAD")
TRAIN_WINDOWS, SCORE_TRAIN_WINDOWS, DEV_WINDOWS = channel.TRAIN_WINDOWS, channel.SCORE_TRAIN_WINDOWS, channel.DEV_WINDOWS
BATCH_SIZE, SEQUENCE, UPDATES = channel.BATCH_SIZE, channel.SEQUENCE, channel.UPDATES
DEADLINE_S, CELL_DEADLINE_S = channel.DEADLINE_S, channel.CELL_DEADLINE_S
STORAGE_CAP, FREE_RESERVE = channel.STORAGE_CAP, channel.FREE_RESERVE
Deadline, Storage = channel.Deadline, channel.Storage
batch_data = parent.batch_data
PLAN = REPO / "manuscript/research/PLAN.md"
PRODUCER_SHA256 = "f8e7f3ddc66fff4f3d1d41ff1228eca030da8c183f437d38cf4181befa2ecbb8"
PRODUCER_SEAL_SHA256 = "e17ed897707852ce5124f60a7726529cff7a9760567e5941ce797232882e3767"
AUDIT_MANIFEST_SHA256 = "6e9b8e56f5d5d2ee41ce41bff3e63855ad95a3698901d42860f484f30022c2ec"
AUDIT_RECEIPT_SHA256 = "96e824375caf67522059a6b2a8864ff58b1c2ba98f668e62f90b6680d53d8e72"


def arm_spec(arm):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("one exact predeclared routing-contrast arm required")
    return {"arm": arm, "body_lr": 0.01, "contrast_lr": 0.001 if arm == "contrast_slow" else 0.01,
            "mean_trainable": arm == "contrast_mean", "mean_lr": 0.01 if arm == "contrast_mean" else None}


def source_identity():
    values = dict(channel.source_identity())
    for name in ("manuscript/tools/diagnose_fpga_lm_routing_contrasts.py", "tests/test_fpga_lm_routing_contrasts.py"):
        values[name] = old.digest((REPO / name).read_bytes())
    return values


def training_order():
    order = parent.training_order(0)
    equal(sorted(order), list(TRAIN_WINDOWS), "same full order-zero training membership")
    return order


def recipe(arm):
    settings = arm_spec(arm)
    values = channel.recipe()
    values.pop("learning_rate")
    values.update(schema=SCHEMA, arm=arm, arm_settings=settings, training_order=training_order(),
        body_learning_rate=0.01,
        initialization="same archived channel-choice state0, m=-.005,d=.005 and empty coordinate Adam",
        intervention="primary=m+d, alternate=m-d in original true-bank pair map; all noncandidate MP and MB fixed",
        optimizer_coordinates=list(BODY) + ["d"] + (["m"] if settings["mean_trainable"] else []),
        coordinate_pullback={"d": "g_primary-g_alternate (no half factor)", "m": "g_primary+g_alternate"},
        mean_constraint="fixed coordinate m in A/B, not a bitwise physical p+a==2m promise or a fixed functional bias",
        frozen_regions=["MB", "MP noncandidate slots"], trainable_regions=list(BODY) + ["MP pair coordinates"],
        freezing_rule="no full MP/MB optimizer; retain all full physical raw G, apply only exact coordinate pullback",
        forward="unchanged full-D guarded/dense selector; no restricted pair selector or new fast path",
        state_format=SCHEMA, state_format_arm=arm,
        controls="reuse audited channel-choice, frozen-core, original mixed and fitted-prior GRU endpoints",
        first_batch_parity="query0 inputs/targets/all traces/full raw G byte-exact versus audited channel-choice query0 before first commit; original core equality is transitive",
        audit="one independent dense physical VJP loop plus autograd materialization pullback; exact saved coordinate-G/full coordinate and materialized-state replay; all scores and actual guards",
        scale_caveat="C starts with g_m=g_d when g_alternate=0; in real arithmetic its first primary move is twice A's; not matched physical-step norms",
        quality_gate="none; complete three declared independent arms before comparing, no adaptive selection or promotion")
    return values


def _coordinate_names(arm):
    return list(BODY) + ["d"] + (["m"] if arm_spec(arm)["mean_trainable"] else [])


def _physical(coordinates, fixed, spec):
    """Canonical differentiable construction; addition/subtraction order fixed."""
    v, d, n, _ = channel._validate_mapping(spec)
    mean = coordinates["m"] if "m" in coordinates else fixed["m"]
    contrast = coordinates["d"]
    projection = torch.full((n, d, d), channel.OFF_WEIGHT, dtype=torch.float64)
    for layer in range(n):
        projection[layer, torch.arange(d), spec["primary_by_layer"][layer]] = mean[layer] + contrast[layer]
        projection[layer, torch.arange(d), spec["alternate_by_layer"][layer]] = mean[layer] - contrast[layer]
    return {"EMBED": coordinates["EMBED"], "MH": coordinates["MH"], "MP": projection,
            "MB": fixed["MB"], "W_HEAD": coordinates["W_HEAD"]}


def _validate_coordinates(cell):
    names = _coordinate_names(cell["arm"])
    equal(list(cell["optimizer_parameters"]), names, "exact ordered active coordinate parameters")
    v, d, n, k = channel._validate_mapping(cell["mapping"])
    shapes = {"EMBED": (v, d), "MH": (n, d, k), "W_HEAD": (v, d), "d": (n, d), "m": (n, d)}
    equal(set(cell["fixed"]), {"MB"} if "m" in names else {"MB", "m"}, "fixed coordinates not simultaneously optimized")
    for name, value in cell["optimizer_parameters"].items():
        helper._finite(value, "active coordinate", detached=True)
        if tuple(value.shape) != shapes[name] or not value.is_leaf:
            raise ValueError("complete detached natural coordinate shape required")
    for name, value in cell["fixed"].items():
        helper._finite(value, "fixed coordinate", detached=True)
        if tuple(value.shape) != (n, d):
            raise ValueError("full fixed pair-mean/core-bias buffer required")
        expected = torch.zeros_like(value) if name == "MB" else torch.full_like(value, -0.005)
        guard.byte_equal({name: value}, {name: expected}, "unchanged original fixed buffer")
    if len({id(value) for value in cell["optimizer_parameters"].values()}) != len(names):
        raise ValueError("unique coordinate tensor ownership required")
    buffers = [*cell["optimizer_parameters"].values(), *cell["fixed"].values()]
    if len({value.untyped_storage().data_ptr() for value in buffers}) != len(buffers):
        raise ValueError("independent coordinate/fixed backing storage required")


def materialize(cell):
    _validate_coordinates(cell)
    values = _physical(cell["optimizer_parameters"], cell["fixed"], cell["mapping"])
    helper.validate_weights(values)
    cell["parameters"] = values
    return values


def pullback(raw, cell, *, independent=False):
    if type(independent) is not bool:
        raise ValueError("literal coordinate pullback mode required")
    _validate_coordinates(cell)
    helper.validate_weights(raw)
    for name, value in raw.items():
        helper._finite(value, "full physical raw gradient", detached=True)
        if value.shape != cell["parameters"][name].shape:
            raise ValueError("complete raw gradient geometry required")
    if independent:
        leaves = {name: value.detach().clone().requires_grad_() for name, value in cell["optimizer_parameters"].items()}
        weights = _physical(leaves, cell["fixed"], cell["mapping"])
        regions = list(BODY) + ["MP"]
        gradients = torch.autograd.grad(tuple(weights[name] for name in regions), tuple(leaves.values()),
                                        grad_outputs=tuple(raw[name] for name in regions), allow_unused=False)
        result = {name: value.detach().clone() for name, value in zip(leaves, gradients)}
    else:
        spec = cell["mapping"]
        p = torch.tensor(spec["primary_by_layer"], dtype=torch.int64).unsqueeze(-1)
        a = torch.tensor(spec["alternate_by_layer"], dtype=torch.int64).unsqueeze(-1)
        gp, ga = raw["MP"].gather(-1, p).squeeze(-1), raw["MP"].gather(-1, a).squeeze(-1)
        result = {name: raw[name].detach().clone() for name in BODY}
        result["d"] = gp - ga
        if "m" in cell["optimizer_parameters"]:
            result["m"] = gp + ga
    for name, value in result.items():
        helper._finite(value, "applied coordinate gradient", detached=True)
        if value.shape != cell["optimizer_parameters"][name].shape:
            raise ValueError("complete applied coordinate shape required")
    return result


def make_optimizer(cell):
    _validate_coordinates(cell)
    settings, coordinates = arm_spec(cell["arm"]), cell["optimizer_parameters"]
    optimizer = helper.make_optimizer([coordinates[name] for name in BODY])
    optimizer.param_groups[0]["lr"] = settings["body_lr"]
    optimizer.add_param_group({"params": [coordinates["d"]], "lr": settings["contrast_lr"]})
    if settings["mean_trainable"]:
        optimizer.add_param_group({"params": [coordinates["m"]], "lr": settings["mean_lr"]})
    _optimizer_groups(cell, optimizer)
    return optimizer


def _optimizer_groups(cell, optimizer):
    settings = arm_spec(cell["arm"])
    groups = [(list(BODY), settings["body_lr"]), (["d"], settings["contrast_lr"])]
    if settings["mean_trainable"]:
        groups.append((["m"], settings["mean_lr"]))
    equal(len(optimizer.param_groups), len(groups), "exact declared coordinate Adam groups")
    if set(optimizer.state) - set(cell["optimizer_parameters"].values()):
        raise ValueError("no optimizer state outside declared coordinates")
    result = []
    for actual, (names, lr) in zip(optimizer.param_groups, groups):
        wanted = [cell["optimizer_parameters"][name] for name in names]
        if len(actual["params"]) != len(wanted) or any(a is not b for a, b in zip(actual["params"], wanted)):
            raise ValueError("exact ordered coordinate optimizer ownership required")
        expected = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0,
                    "amsgrad": False, "maximize": False, "foreach": False, "fused": False,
                    "capturable": False, "differentiable": False}
        for key, value in expected.items():
            equal(actual.get(key), value, "unchanged eager coordinate Adam option: " + key)
        if set(actual) - set(expected) - {"params", "decoupled_weight_decay"}:
            raise ValueError("unrecognized coordinate Adam option")
        if "decoupled_weight_decay" in actual:
            equal(actual["decoupled_weight_decay"], False, "no AdamW conversion")
        result.append({"parameter_names": names, "options": json.loads(json.dumps(
            {key: value for key, value in actual.items() if key != "params"}, allow_nan=False))})
    return result


def state_record(directory, step, cell, optimizer, *, write=False, storage=None):
    if type(step) is not int or not 0 <= step <= UPDATES or type(write) is not bool:
        raise ValueError("literal declared coordinate state step/write required")
    _validate_coordinates(cell)
    helper.validate_weights(cell["parameters"])
    guard.byte_equal(cell["parameters"], _physical(cell["optimizer_parameters"], cell["fixed"], cell["mapping"]),
                     "physical state equals canonical coordinate materialization byte-for-byte")
    groups = _optimizer_groups(cell, optimizer)
    equal(set(optimizer.state), set() if step == 0 else set(cell["optimizer_parameters"].values()),
          "genuinely empty initial or exactly complete active coordinate Adam states")
    arrays = {"physical/" + name: value.detach() for name, value in cell["parameters"].items()}
    arrays.update({"fixed/" + name: value for name, value in cell["fixed"].items()})
    shapes = {}
    for name, parameter in cell["optimizer_parameters"].items():
        arrays["coordinate/" + name] = parameter.detach()
        moment = optimizer.state.get(parameter, {})
        equal(set(moment), set() if step == 0 else {"step", "exp_avg", "exp_avg_sq"}, "complete actual coordinate Adam state")
        shapes[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype), "initialized": step != 0}
        if step:
            counter = moment["step"]
            if (counter.dtype not in (torch.float32, torch.float64) or counter.device.type != "cpu"
                    or counter.shape != torch.Size([]) or not bool(torch.isfinite(counter)) or float(counter) != step):
                raise ValueError("exact actual coordinate Adam counter required")
            for key in ("exp_avg", "exp_avg_sq"):
                value = moment[key]
                helper._finite(value, "coordinate Adam moment", detached=True)
                if value.shape != parameter.shape or (key == "exp_avg_sq" and bool((value < 0).any())):
                    raise ValueError("complete valid coordinate Adam moments required")
                arrays[key + "/" + name] = value
            arrays["step/" + name] = counter
            shapes[name]["step_dtype"] = str(counter.dtype)
    path = Path(directory) / f"state_{step:06d}.npz"
    if write:
        storage.arrays(path, arrays)
    guard.byte_equal(load(path), arrays, "all physical/coordinate/fixed/moment/counter bytes")
    payload = path.read_bytes()
    result = {"schema": SCHEMA, "arm": cell["arm"], "step": step, "file": path.name,
              "bytes": len(payload), "sha256": old.digest(payload), "coordinate_order": list(cell["optimizer_parameters"]),
              "coordinate_state": shapes, "optimizer_groups": groups, "full_state_tensor_sha256": tensor_hash(arrays),
              "physical_parameters_sha256": tensor_hash(cell["parameters"]),
              "coordinate_parameters_sha256": tensor_hash(cell["optimizer_parameters"]), "initial_optimizer_state_empty": step == 0}
    if write:
        storage.json(path.with_suffix(".json"), result)
    equal(read(path.with_suffix(".json")), result, "exact typed full coordinate state receipt")
    return result

def _audit_capsule(path, root, record):
    if (type(AUDIT_RECEIPT_SHA256) is not str or len(AUDIT_RECEIPT_SHA256) != 64
            or any(char not in "0123456789abcdef" for char in AUDIT_RECEIPT_SHA256)):
        raise ValueError("successful current channel independent audit is not yet pinned")
    path = parent.rate.head._regular(path)
    equal(old.digest(path.read_bytes()), AUDIT_RECEIPT_SHA256, "actual successful channel audit pin")
    equal(old.digest(record.read_bytes()), PRODUCER_SHA256, "actual complete channel producer pin")
    equal(old.digest((root / "artifact_manifest.json").read_bytes()), PRODUCER_SEAL_SHA256, "actual channel seal pin")
    value = read(path)
    expected = {"schema": channel.SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                "run_directory": str(root), "record": str(record), "source_hashes": channel.source_identity(),
                "producer_public_sha256": PRODUCER_SHA256, "producer_seal_sha256": PRODUCER_SEAL_SHA256,
                "audit_manifest_sha256": AUDIT_MANIFEST_SHA256,
                "deadline_seconds": channel.DEADLINE_S, "cell_deadline_seconds": channel.CELL_DEADLINE_S,
                "independent_tropical_window_vjps": 1024, "independent_tropical_window_forwards": 1280,
                "independent_tropical_batch_gradients": 256, "applied_gradient_masks_verified": 256,
                "initial_score_parity_checks": 128, "first_batch_parity_checks": 1,
                "all_whole_model_dense_fallback_counts_verified": True,
                "frozen_regions_verified": list(channel.FROZEN_REGIONS), "supplemental_guarded_window_forwards": 1280,
                "full_batch_gradients_verified": 256, "complete_saved_states_verified": 257,
                "optimizer_commits_replayed": 256, "score_window_readouts": 256,
                "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
                "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
                "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                "reserved_development_evaluated": False, "physical_fpga_execution": False,
                "paid_execution_authorized": False, "recipe_promoted": False}
    for key, wanted in expected.items():
        equal(value.get(key), wanted, "successful channel audit: " + key)
    for key in ("verification_elapsed_utc_seconds", "verification_elapsed_monotonic_seconds"):
        elapsed = value[key]
        if type(elapsed) is not float or not math.isfinite(elapsed) or not 0 <= elapsed < channel.DEADLINE_S:
            raise ValueError("actual successful audit must finish within declared bound")
    directory = Path(value["audit_directory"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("actual successful channel audit directory required")
    directory = directory.resolve(strict=True)
    names = {"audit_manifest.json", "declaration.md", "verification.json"}
    equal({p.name for p in directory.iterdir()}, names, "exact successful channel audit capsule")
    files = {name: old.file_identity(parent.rate.head._regular(directory / name)) for name in names}
    equal((directory / "verification.json").read_bytes(), path.read_bytes(), "exact successful private/public verification bytes")
    equal(files["audit_manifest.json"]["sha256"], AUDIT_MANIFEST_SHA256, "actual channel audit manifest pin")
    manifest = read(directory / "audit_manifest.json")
    for key, wanted in {"schema": channel.SCHEMA, "complete": False, "mode": "independent_audit",
                        "run_directory": str(root), "record": str(record), "audit_directory": str(directory),
                        "audit_record": str(path), "source_hashes": channel.source_identity(),
                        "producer_public_sha256": PRODUCER_SHA256, "producer_seal_sha256": PRODUCER_SEAL_SHA256,
                        "runtime": value["runtime"], "started_at_utc": value["started_at_utc"],
                        "deadline_seconds": channel.DEADLINE_S, "cell_deadline_seconds": channel.CELL_DEADLINE_S,
                        "declaration_sha256": files["declaration.md"]["sha256"]}.items():
        equal(manifest[key], wanted, "successful channel audit manifest: " + key)
    return value, {"receipt": old.file_identity(path), "files": files}


@dataclass(frozen=True)
class _Validation:
    binding: dict
    sources: dict
    channel_manifest: dict
    channel_validation: object

    def __post_init__(self):
        for key in ("binding", "sources", "channel_manifest"):
            object.__setattr__(self, key, copy.deepcopy(getattr(self, key)))


def bind_inputs(config):
    arm_spec(config.arm)
    sources = source_identity()
    root, record = Path(config.prior_run).resolve(strict=True), parent.rate.head._regular(config.prior_record)
    _, capsule = _audit_capsule(config.prior_audit, root, record)
    previous = read(root / "manifest.json")
    equal(previous["runtime"], parent.runtime(), "same original software runtime")
    ancestry, raw, token = channel.bind_inputs(channel._bound_config(previous))
    equal(ancestry, previous["binding"], "canonical fully byte-verified channel ancestry")
    channel.frozen(previous, token)
    artifacts = channel._inventory(root)
    summary = read(root / "summary.json")
    equal(read(record), channel.public_record(root, summary, artifacts), "actual complete audited channel publication")
    equal(summary["complete"], True, "complete prerequisite channel control")
    equal(Path(config.corpus).resolve(strict=True), Path(ancestry["dataset"]["path"]), "same corpus path")
    source = root / channel.ARM / "state_000000.npz"
    binding = {"prior_run": str(root), "prior_record": str(record), "prior_audit": str(parent.rate.head._regular(config.prior_audit)),
               "audit_receipt_sha256": AUDIT_RECEIPT_SHA256, "audit_capsule": capsule,
               "channel_binding": ancestry, "core_binding": ancestry["core_binding"],
               "continuation_binding": ancestry["continuation_binding"], "endpoint_binding": ancestry["endpoint_binding"],
               "development_binding": ancestry["development_binding"], "dataset": ancestry["dataset"],
               "initial_scores": copy.deepcopy(ancestry["initial_scores"]),
               "initial_query": old.file_identity(root / channel.ARM / "query_000000.npz"),
               "initial_state": {"file": old.file_identity(source), "metadata": old.file_identity(source.with_suffix(".json"))},
               "parent_files": {name: old.file_identity(root / name) for name in ("manifest.json", "artifact_manifest.json", "summary.json")},
               "parent_public": old.file_identity(record), "parent_artifacts": artifacts}
    validation = _Validation(binding, sources, previous, token)
    frozen({"schema": SCHEMA, "arm": config.arm, "recipe": recipe(config.arm), "source_hashes": sources, "binding": binding}, validation)
    return binding, raw, validation


def frozen(manifest, validation):
    if not isinstance(validation, _Validation):
        raise ValueError("private invocation-local routing-contrast validation required")
    equal(manifest["schema"], SCHEMA, "routing-contrast schema")
    equal(manifest["recipe"], recipe(manifest["arm"]), "one exact declared routing-coordinate arm")
    equal(manifest["source_hashes"], source_identity(), "all79 source identities")
    equal(manifest["source_hashes"], validation.sources, "original invocation sources")
    equal(manifest["binding"], validation.binding, "canonical audited full ancestry")
    binding = manifest["binding"]
    equal(binding["audit_receipt_sha256"], AUDIT_RECEIPT_SHA256, "actual successful admission pin")
    channel.frozen(validation.channel_manifest, validation.channel_validation)
    equal(channel._inventory(Path(binding["prior_run"])), binding["parent_artifacts"], "all channel artifacts unchanged")
    _, capsule = _audit_capsule(binding["prior_audit"], Path(binding["prior_run"]), Path(binding["prior_record"]))
    equal(capsule, binding["audit_capsule"], "complete channel audit proof unchanged")
    for descriptor in (*binding["parent_files"].values(), binding["parent_public"], binding["initial_query"],
                       *binding["initial_scores"]["train"], *binding["initial_scores"]["development"],
                       *binding["initial_state"].values()):
        equal(old.file_identity(descriptor["path"]), descriptor, "actual input raw/seal/public/initial-control identity")


def initial_model(binding, arm):
    settings = arm_spec(arm)
    cell = channel.initial_model(binding["channel_binding"])
    original = cell["parameters"]
    source = binding["initial_state"]
    for descriptor in source.values():
        equal(old.file_identity(descriptor["path"]), descriptor, "actual audited channel state0 bytes")
    parameters, metadata = endpoint._checkpoint(Path(source["file"]["path"]), cell, "mixed", 0)
    guard.byte_equal(original, parameters, "same actual complete physical channel initialization")
    n, d = original["MB"].shape
    coordinates = {name: original[name].detach().clone() for name in BODY}
    coordinates["d"] = torch.full((n, d), 0.005, dtype=torch.float64)
    fixed = {"MB": original["MB"].detach().clone()}
    if settings["mean_trainable"]:
        coordinates["m"] = torch.full((n, d), -0.005, dtype=torch.float64)
    else:
        fixed["m"] = torch.full((n, d), -0.005, dtype=torch.float64)
    previous_transform = cell["transform"]
    cell.update(arm=arm, optimizer_parameters=coordinates, fixed=fixed,
                mapping=copy.deepcopy(previous_transform["mapping"]))
    guard.byte_equal(materialize(cell), original, "m+d/m-d reproduces every initial physical byte exactly")
    cell["channel_source"] = {**copy.deepcopy(source), "full_state_tensor_sha256": metadata["full_state_tensor_sha256"], "step": 0}
    cell["transform"] = {"schema": SCHEMA + ".coordinates", "arm": arm, "arm_settings": settings,
                         "mapping": cell["mapping"], "channel_transform": previous_transform,
                         "initial_mean": -0.005, "initial_contrast": 0.005,
                         "initial_physical_parameters_sha256": tensor_hash(original),
                         "initial_coordinate_parameters_sha256": tensor_hash(coordinates),
                         "active_coordinate_parameters": sum(value.numel() for value in coordinates.values()),
                         "initial_physical_bytes_equal": True,
                         "scope": "canonical add/sub materialization, not FP64 pair-sum or optimizer-equivalence claim"}
    return cell


def query_batch(directory, step, cell, tokens, targets, indices, *, write=False, recompute=False, storage=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal coordinate query modes required")
    materialize(cell)
    # Reuse production/archive semantics, but own the independent loop so its
    # full VJP is available to the independent coordinate pullback without a
    # second set of forward/reverse queries.
    raw, row = parent.query_batch(directory, step, cell, tokens, targets, indices,
                                  write=write, recompute=False, storage=storage)
    applied = pullback(raw, cell)
    if recompute:
        arrays = load(Path(directory) / f"query_{step:06d}.npz")
        independent_raw = {name: torch.zeros_like(value) for name, value in raw.items()}
        for position, index in enumerate(indices):
            trace = {name: arrays[f"window_{index}/{name}"] for name in helper.TRACE_KEYS}
            base = helper.forward(cell["parameters"], tokens[position], independent=True)
            parent.rate.head.helper.near(trace, helper.trace(base), "independent dense physical coordinate query")
            gradient = helper.gradient(cell["parameters"], tokens[position], targets[position], base,
                                       targets.numel(), independent=True)
            for name in independent_raw:
                independent_raw[name] = independent_raw[name] + gradient[name]
            guarded = guard.guarded.forward(cell["parameters"], tokens[position])
            guard.byte_equal(trace, helper.trace(guarded), "supplemental exact actual guarded coordinate query")
            equal(arrays["guard_counts"][position].tolist(), parent._guard_arrays(guarded), "actual coordinate query guard counters")
        parent.rate.head.helper.near(raw, independent_raw, "independent full physical batch raw G")
        independent_applied = pullback(independent_raw, cell, independent=True)
        parent.rate.head.helper.near(applied, independent_applied, "independent dense VJP and coordinate-map autograd pullback")
    row["coordinate_gradient_sha256"] = tensor_hash(applied)
    row["coordinate_pullback_rule"] = "d=Gp-Ga; m=Gp+Ga when active; body unchanged"
    return raw, row


def applied_record(directory, step, raw, cell, *, write=False, storage=None):
    if type(step) is not int or not 0 <= step < UPDATES or type(write) is not bool:
        raise ValueError("literal local coordinate-gradient step/write required")
    expected = pullback(raw, cell)
    path = Path(directory) / f"applied_{step:06d}.npz"
    if write:
        storage.arrays(path, expected)
    saved = load(path)
    guard.byte_equal(saved, expected, "exact full raw-to-applied coordinate pullback, including body bytes")
    return saved, {"step": step, "coordinate_order": list(cell["optimizer_parameters"]),
                   "raw_gradient_sha256": tensor_hash(raw), "applied_gradient_sha256": tensor_hash(saved),
                   "file": path.name, "file_sha256": old.digest(path.read_bytes()),
                   "saved_coordinate_gradient_drives_adam": True}


def trace_arrays(arrays, cell):
    """Validate actual guards without assuming learned pairs always fall back."""
    trace = {name: arrays[name] for name in helper.TRACE_KEYS}
    t = trace["logits"].shape[0]
    counts = arrays["guard_counts"]
    if counts.shape != (1, 6):
        raise ValueError("one complete guard row per saved trace required")
    parent._validate_guard_rows(counts.repeat(BATCH_SIZE, 1), cell, t)
    channel.classify_routes([trace], cell["mapping"])
    return trace


def controls(binding):
    values = channel.controls(binding["channel_binding"])
    root = Path(binding["prior_run"])
    summary = read(root / "summary.json")
    values["fixed_channel"] = {**summary["after"], "initial_source": summary["initial_source"],
        "paths": {"train": [str(root / channel.ARM / "train" / f"score_{index:04d}.npz") for index in SCORE_TRAIN_WINDOWS],
                  "development": [str(root / channel.ARM / f"score_final_{index:03d}_1.npz") for index in DEV_WINDOWS]}}
    equal(values["fixed_channel"]["initial_source"], values["mixed"]["original_initial"]["source_state"],
          "all fixed controls refer to the same actual original initializer")
    return values


def comparisons(after, inherited):
    result = channel.comparisons(after, inherited)
    for cohort, indices in (("train", SCORE_TRAIN_WINDOWS), ("development", DEV_WINDOWS)):
        current, fixed = after[cohort], inherited["fixed_channel"][cohort]
        equal([row["window"] for row in fixed["per_window"]], list(indices), "complete fixed-channel comparator cohort")
        equal(parent._mean(fixed["per_window"]), fixed["mean"], "actual fixed-channel comparator mean")
        result[cohort].update(fixed_channel=fixed["mean"],
            new_minus_fixed_channel_ce=current["mean"]["ce"] - fixed["mean"]["ce"],
            new_minus_fixed_channel_accuracy=current["mean"]["accuracy"] - fixed["mean"]["accuracy"])
        for index, row in enumerate(result[cohort]["per_window"]):
            row["new_minus_fixed_channel_ce"] = current["per_window"][index]["ce"] - fixed["per_window"][index]["ce"]
            row["accuracy_differences"]["fixed_channel"] = current["per_window"][index]["accuracy"] - fixed["per_window"][index]["accuracy"]
    return result


def delta_statistics(before, after):
    equal(set(before), set(after), "same measured parameter-delta fields")
    result = {}
    for name in before:
        delta = after[name].detach() - before[name]
        helper._finite(delta, "measured actual parameter step", detached=True)
        maximum, norm = float(delta.abs().max()), float(torch.linalg.vector_norm(delta))
        if not math.isfinite(norm):
            raise ValueError("finite actual parameter-step statistic required")
        result[name] = {"max_abs": maximum, "l2": norm}
    return result


def pair_values(cell):
    projection = cell["parameters"]["MP"]
    spec = cell["mapping"]
    p = projection.gather(-1, torch.tensor(spec["primary_by_layer"], dtype=torch.int64).unsqueeze(-1)).squeeze(-1)
    a = projection.gather(-1, torch.tensor(spec["alternate_by_layer"], dtype=torch.int64).unsqueeze(-1)).squeeze(-1)
    return {"primary": p.detach().clone(), "alternate": a.detach().clone(), "gap": (p - a).detach()}


def _fixed_record(cell, step):
    _validate_coordinates(cell)
    _, d, n, _ = channel._validate_mapping(cell["mapping"])
    return {"step": step, "fixed_buffer_sha256": tensor_hash(cell["fixed"]),
            "noncandidate_projection_entries": n * d * (d - 2), "noncandidate_sentinels_unchanged": True,
            "mb_fixed": True, "mean_buffer_fixed": "m" in cell["fixed"],
            "scope": "full state byte-equality to canonical materialization proves fixed-slot values; no physical pair-sum identity"}


def execute(manifest, output, raw, *, write=False, recompute=False, clock=None, storage=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal coordinate execution modes required")
    arm = manifest["arm"]
    arm_spec(arm)
    equal(UPDATES * BATCH_SIZE, len(TRAIN_WINDOWS), "one exact original-order training pass")
    output, binding = Path(output), manifest["binding"]
    clock = clock or Deadline()
    if write and storage is None:
        storage = Storage(output, clock)
        storage.used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    clock.begin_cell()
    directory, initial_directory = output / arm, output / "initial"
    if write:
        for path in (directory, directory / "train", initial_directory, initial_directory / "train"):
            path.mkdir(exist_ok=False)
    inherited = controls(binding)
    cell = initial_model(binding, arm)
    equal(cell["initial_source"], inherited["mixed"]["original_initial"]["source_state"],
          "all compared models share the actual original fitted source")
    if write:
        storage.json(output / "transform.json", cell["transform"])
    equal(read(output / "transform.json"), cell["transform"], "exact arm-specific coordinate/map initialization")
    optimizer = make_optimizer(cell)
    equal(len(optimizer.state), 0, "genuinely empty coordinate Adam")
    head_reference = {"W_HEAD": continuation.support_reference(cell)["W_HEAD"]}
    states = [state_record(directory, 0, cell, optimizer, write=write, storage=storage)]
    invariants = [_fixed_record(cell, 0)]
    head_states = [{"step": 0, "regions": None, "optimizer_moments_empty": True}]
    print(json.dumps({"stage": "coordinate_initial_parity", "arm": arm, "recompute": recompute}), flush=True)
    initial_scores = channel.collect_pair(initial_directory, cell, raw, binding["dataset"], "initial", write=write,
                                          recompute=recompute, storage=storage, clock=clock)
    parity = channel.initial_parity(initial_directory, initial_scores, binding, cell["mapping"])
    equal(state_record(directory, 0, cell, optimizer), states[0], "initial scores leave all coordinate/physical/Adam bytes unchanged")
    order, query_rows, applied_rows = training_order(), [], []
    for step in range(UPDATES):
        clock.check()
        indices = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        tokens, targets = batch_data(raw, binding["dataset"], indices)
        before = tensor_hash(cell["parameters"])
        before_coordinates = {name: value.detach().clone() for name, value in cell["optimizer_parameters"].items()}
        before_pairs = pair_values(cell)
        raw_gradients, row = query_batch(directory, step, cell, tokens, targets, indices,
                                        write=write, recompute=recompute, storage=storage)
        query_path = directory / f"query_{step:06d}.npz"
        arrays = load(query_path)
        traces = [trace_arrays({**{key: arrays[f"window_{index}/{key}"] for key in helper.TRACE_KEYS},
                                "guard_counts": arrays["guard_counts"][position:position + 1]}, cell)
                  for position, index in enumerate(indices)]
        row["channel_choices"] = channel.classify_routes(traces, cell["mapping"])
        if step == 0:
            parity["first_batch"] = channel.first_batch_parity(query_path, binding["initial_query"])
            if write:
                storage.json(output / "initial_parity.json", parity)
            equal(read(output / "initial_parity.json"), parity, "all initial checks precede first coordinate commit")
        applied, applied_row = applied_record(directory, step, raw_gradients, cell, write=write, storage=storage)
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in cell["optimizer_parameters"].items():
            parameter.grad = applied[name].detach().clone()
        optimizer.step()
        materialize(cell)
        states.append(state_record(directory, step + 1, cell, optimizer, write=write, storage=storage))
        invariants.append(_fixed_record(cell, step + 1))
        head_states.append({"step": step + 1, "regions": continuation.support_state(cell, optimizer, head_reference),
                            "optimizer_moments_empty": False})
        row.update(before_physical_parameters_sha256=before, after_physical_parameters_sha256=tensor_hash(cell["parameters"]),
                   applied_gradient_sha256=applied_row["applied_gradient_sha256"], state_sha256=states[-1]["full_state_tensor_sha256"],
                   actual_coordinate_step=delta_statistics(before_coordinates, cell["optimizer_parameters"]),
                   actual_physical_pair_step=delta_statistics(before_pairs, pair_values(cell)),
                   saved_coordinate_gradient_adam_replay=True)
        query_rows.append(row)
        applied_rows.append(applied_row)
        clock.check()
        if (step + 1) % 16 == 0:
            print(json.dumps({"stage": "coordinate_update", "arm": arm, "step": step + 1,
                              "query_ce": parent._mean(row["per_window"])["ce"], "recompute": recompute}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    before_scores = state_record(directory, UPDATES, cell, optimizer)
    after = channel.collect_pair(directory, cell, raw, binding["dataset"], "final", write=write,
                                 recompute=recompute, storage=storage, clock=clock)
    equal(state_record(directory, UPDATES, cell, optimizer), before_scores, "final scores leave all actual state bytes unchanged")
    changes, final_choices = {}, {}
    for cohort, indices in (("train", SCORE_TRAIN_WINDOWS), ("development", DEV_WINDOWS)):
        equal([row["window"] for row in after[cohort]["per_window"]], list(indices), "complete matched final cohort")
        traces = [trace_arrays(load(channel._score_path(directory, cohort, index, "final")), cell) for index in indices]
        final_choices[cohort] = channel.classify_routes(traces, cell["mapping"])
        changes[cohort] = {}
        for control in ("mixed", "frozen_core", "fixed_channel"):
            paths = inherited[control]["paths"][cohort]
            equal(len(paths), len(indices), "complete actual fixed control final tapes")
            paired = [{"window": index, "routes": continuation.route_changes(load(path), trace)}
                      for index, path, trace in zip(indices, paths, traces)]
            changes[cohort][control] = {"per_window": paired,
                "totals": {key: {field: sum(item["routes"][key][field] for item in paired) for field in ("changed", "total")}
                           for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}
    for source in (cell["initial_source"], cell["channel_source"]):
        for key in ("file", "metadata"):
            descriptor = source[key]
            equal(old.file_identity(descriptor["path"]), descriptor, "actual original/control checkpoint remains untouched")
    choices = [row["channel_choices"] for row in query_rows]
    query_totals = {key: sum(value["total"][key] for value in choices) for key in ("primary", "alternate", "other", "total")}
    query_totals.update(both_choice_row_checkpoints=sum(value["total"]["both_choice_rows"] for value in choices),
                        total_row_checkpoints=sum(value["total"]["total_rows"] for value in choices))
    guard_rows = [row for item in query_rows for row in item["guard_counts"]]
    for scores in (initial_scores, after):
        guard_rows.extend(row for cohort in scores.values() for item in cohort["per_window"] for row in item["guard_counts"])
    guard_totals = {"forward_windows": len(guard_rows), "structure_certified_windows": sum(row[0] for row in guard_rows),
                    "whole_model_dense_fallback_windows": sum(row[1] for row in guard_rows),
                    **{key: sum(row[position + 2] for row in guard_rows) for position, key in enumerate(guard.GUARD_KEYS)}}
    comparison = comparisons(after, inherited)
    material = {cohort: comparison[cohort]["new"]["ce"] <= comparison[cohort]["frozen_core"]["ce"] - 0.01
                for cohort in ("train", "development")}
    result = {"schema": SCHEMA, "complete": True, "arm": arm, "arm_settings": arm_spec(arm), "recipe": recipe(arm),
              "source_hashes": manifest["source_hashes"], "order_seed": 0, "training_order": order,
              "initial_source": cell["initial_source"], "channel_source": cell["channel_source"], "transform": cell["transform"],
              "initial_scores": initial_scores, "initial_parity": parity, "inherited_controls": inherited,
              "after": after, "comparisons": comparison,
              "material_progress": {"per_cohort": material, "both_cohorts": all(material.values()), "margin_nat": 0.01,
                                    "requires_successful_independent_audit": True, "scope": "descriptive replication trigger, not promotion or significance"},
              "fixed_prior_baselines": read(Path(binding["prior_run"]) / "summary.json")["fixed_prior_baselines"],
              "query_rows": query_rows, "applied_rows": applied_rows, "states": states,
              "fixed_coordinate_invariants": invariants, "head_support_states": head_states,
              "query_channel_choice_totals": query_totals, "final_channel_choices": final_choices,
              "between_recipe_final_route_changes": changes, "actual_guard_totals": guard_totals,
              "route_scope": "query BOTH counts are row-checkpoints; final cohorts are separate fixed checkpoints; valid other winners are not censored",
              "raw_gradient_queries": UPDATES, "applied_gradient_archives": UPDATES, "optimizer_commits": UPDATES,
              "complete_saved_states": UPDATES + 1, "score_window_readouts": 2 * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
              "initial_score_parity_checks": len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS), "first_batch_parity_checks": 1,
              "producer_forward_windows": UPDATES * BATCH_SIZE + 2 * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
              "new_gradient_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
              "inherited_fitting_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE, "inherited_gradient_pair_exposures": 0,
              "endpoint_history_pair_exposures": 2 * len(TRAIN_WINDOWS) * SEQUENCE, "distinct_gradient_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
              "new_fitting_pair_exposures": 0, "new_baseline_evaluations": 0,
              "new_initial_score_evaluations": len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS), "new_original_model_forwards": 0,
              "new_gru_training": False, "heldout_test_evaluated": False, "reserved_development_evaluated": False,
              "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
              "scope": "one of three declared original-start coordinate controls; all terminal outcomes must enter the block review; no matched-physical-step/capacity/hardware claim"}
    clock.end_cell()
    return result


def expected_artifacts(arm):
    arm_spec(arm)
    names = {"manifest.json", "protocol.json", "declaration.md", "summary.json", "transform.json", "initial_parity.json"}
    names.update(f"{arm}/{prefix}_{step:06d}.npz" for step in range(UPDATES) for prefix in ("query", "applied"))
    names.update(f"{arm}/state_{step:06d}.{suffix}" for step in range(UPDATES + 1) for suffix in ("npz", "json"))
    for directory, label in ((arm, "final"), ("initial", "initial")):
        names.update(f"{directory}/train/score_{index:04d}.npz" for index in SCORE_TRAIN_WINDOWS)
        names.update(f"{directory}/score_{label}_{index:03d}_1.npz" for index in DEV_WINDOWS)
    return names


def _inventory(root):
    root = Path(root)
    arm = read(root / "manifest.json")["arm"]
    artifacts = prior.artifact_manifest(root)
    equal({row["file"] for row in artifacts}, expected_artifacts(arm), "exact full declared per-arm payload inventory")
    equal({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()},
          {arm, arm + "/train", "initial", "initial/train"}, "only declared coordinate-arm directories")
    equal(read(root / "artifact_manifest.json"), {"schema": SCHEMA, "files": artifacts}, "complete sealed coordinate payloads")
    return artifacts


def public_record(root, result, artifacts):
    return {**result, "run_directory": str(root), "runtime": read(Path(root) / "manifest.json")["runtime"],
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "independent_audit_required": True,
            "check_scope": "initial parity, exact coordinate pullback and saved-coordinate-G full physical/coordinate Adam states; independent raw-G/score audit separate"}


def _bound_config(manifest):
    config = channel._bound_config(manifest)
    config.arm = manifest["arm"]
    return config

def _check(root, record, recompute, validation, raw, clock):
    if type(recompute) is not bool:
        raise ValueError("literal independent replay mode required")
    root = Path(root).resolve(strict=True)
    record = None if record is None else parent.rate.head._regular(record)
    originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
    public_bytes = None if record is None else record.read_bytes()
    manifest = read(root / "manifest.json")
    equal(manifest["run_directory"], str(root), "actual routing-contrast output path")
    equal(manifest["runtime"], parent.runtime(), "unchanged current software runtime")
    equal(read(root / "protocol.json"), recipe(manifest["arm"]), "saved fixed routing-contrast protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["declaration_sha256"], "actual saved declaration")
    frozen(manifest, validation)
    artifacts = _inventory(root)
    result = execute(manifest, root, raw, recompute=recompute, clock=clock)
    equal(read(root / "summary.json"), result, "complete recomputed routing-contrast summary")
    published = public_record(root, result, artifacts)
    if record is not None:
        equal(read(record), published, "actual complete routing-contrast publication")
    frozen(manifest, validation)
    equal(_inventory(root), artifacts, "all routing-contrast payload bytes unchanged during replay")
    for name, payload in originals.items():
        equal((root / name).read_bytes(), payload, "exact original routing-contrast metadata bytes")
    if record is not None:
        equal(record.read_bytes(), public_bytes, "exact original public bytes")
    clock.check()
    return published


def _failure(output, clock, stage, error, independent=False, arm=None):
    if output is not None and output.is_dir() and not (output / "failure.json").exists():
        utc, monotonic = clock.elapsed()
        rejected = output / "rejected_public.json"
        retained = old.file_identity(rejected) if rejected.is_file() and not rejected.is_symlink() else None
        pilot.write_json(output / "failure.json", {"schema": SCHEMA, "complete": False, "stage": stage,
            "arm": arm, "type": type(error).__name__, "error": str(error)[:100000], "retained_rejected_public": retained,
            "started_at_utc": guard.parent._utc(clock.started_utc), "failure_observed_at_utc": guard.parent._utc(time.time()),
            "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic, "audit_only": independent,
            "new_training_updates": 0 if independent else None,
            "partial_work_scope": "retained files describe any completed partial work; no automatic retry or false completion",
            "recipe_promoted": False})


def _invoke(config, independent):
    clock, output, publication = Deadline(), None, None
    created = closed = cleanup_done = False
    stage = "setup"
    try:
        clock.arm()
        arm_spec(config.arm)
        output, target = parent._new_outputs(config.audit_output, config.audit_record) if independent else parent._new_outputs(config.output, config.record)
        sources, declaration = source_identity(), PLAN.read_bytes()
        output.mkdir(parents=True, exist_ok=False)
        created = True
        storage = Storage(output, clock)
        torch.set_num_threads(1)
        stage = "input_validation"
        print(json.dumps({"stage": stage, "independent_audit": independent, "new_model_readouts": 0}), flush=True)
        if independent:
            root, record = Path(config.output).resolve(strict=True), parent.rate.head._regular(config.record)
            manifest = read(root / "manifest.json")
            equal(config.arm, manifest["arm"], "requested audit arm matches immutable producer")
            originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
            public_bytes = record.read_bytes()
            artifacts = _inventory(root)
            storage.base_bytes = sum(row["bytes"] for row in artifacts) + len(originals["artifact_manifest.json"]) + len(public_bytes)
            storage.used = storage.base_bytes
            binding, raw, validation = bind_inputs(_bound_config(manifest))
            equal(binding, manifest["binding"], "fresh canonical routing-contrast audit binding")
            private = {"schema": SCHEMA, "arm": config.arm, "complete": False, "mode": "independent_audit",
                       "run_directory": str(root), "record": str(record), "audit_directory": str(output), "audit_record": str(target),
                       "producer_public_sha256": old.digest(public_bytes), "producer_seal_sha256": old.digest(originals["artifact_manifest.json"]),
                       "source_hashes": sources, "runtime": parent.runtime(), "started_at_utc": guard.parent._utc(clock.started_utc),
                       "deadline_seconds": DEADLINE_S, "cell_deadline_seconds": CELL_DEADLINE_S,
                       "declaration_sha256": old.digest(declaration)}
            storage.json(output / "audit_manifest.json", private)
            private_bytes = (output / "audit_manifest.json").read_bytes()
        else:
            root, record = output, target
            binding, raw, validation = bind_inputs(config)
            manifest = {"schema": SCHEMA, "arm": config.arm, "recipe": recipe(config.arm), "source_hashes": sources, "binding": binding,
                        "run_directory": str(root), "record": str(target), "runtime": parent.runtime(),
                        "started_at_utc": guard.parent._utc(clock.started_utc), "declaration_sha256": old.digest(declaration)}
            storage.json(output / "manifest.json", manifest)
            storage.json(output / "protocol.json", recipe(config.arm))
        equal(source_identity(), sources, "no source relabeling during initial validation")
        storage.reserve(output / "declaration.md", len(declaration))
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        storage.used += len(declaration)
        frozen(manifest, validation)
        clock.check()
        stage = "independent_numerical_replay" if independent else "fixed_routing_contrast_control"
        if independent:
            result = _check(root, record, True, validation, raw, clock)
            utc, monotonic = clock.check()
            windows, scores = UPDATES * BATCH_SIZE, 2 * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS))
            published = {"schema": SCHEMA + ".audit", "arm": config.arm, "complete": True, "independent_recompute_completed": True,
                         "run_directory": str(root), "record": str(record), "audit_directory": str(output),
                         "producer_public_sha256": old.digest(public_bytes), "producer_seal_sha256": old.digest(originals["artifact_manifest.json"]),
                         "audit_manifest_sha256": old.digest(private_bytes), "source_hashes": sources, "runtime": parent.runtime(),
                         "started_at_utc": guard.parent._utc(clock.started_utc),
                         "verification_completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                         "verification_elapsed_utc_seconds": utc, "verification_elapsed_monotonic_seconds": monotonic,
                         "deadline_seconds": DEADLINE_S, "cell_deadline_seconds": CELL_DEADLINE_S,
                         "independent_tropical_window_vjps": windows, "independent_tropical_window_forwards": windows + scores,
                         "independent_tropical_batch_gradients": UPDATES, "applied_coordinate_gradients_verified": UPDATES, "independent_coordinate_pullbacks_verified": UPDATES,
                         "fixed_regions_verified": ["MB", "MP noncandidate slots"], "supplemental_guarded_window_forwards": windows + scores,
                         "initial_score_parity_checks": result["initial_score_parity_checks"], "first_batch_parity_checks": 1,
                         "actual_guard_counters_verified": True,
                         "full_batch_gradients_verified": result["raw_gradient_queries"],
                         "complete_saved_states_verified": result["complete_saved_states"],
                         "optimizer_commits_replayed": result["optimizer_commits"], "score_window_readouts": result["score_window_readouts"],
                         "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
                         "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
                         "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                         "reserved_development_evaluated": False, "physical_fpga_execution": False,
                         "paid_execution_authorized": False, "recipe_promoted": False,
                         "scope": "independent reconstruction of the fixed routing-contrast control and final readouts, not new learning, seed confirmation or promotion"}
            storage.json(output / "verification.json", published)
            verification_bytes = (output / "verification.json").read_bytes()
            equal(read(output / "verification.json"), published, "actual independent verification result")
        else:
            result = execute(manifest, output, raw, write=True, storage=storage, clock=clock)
            storage.json(output / "summary.json", result)
            artifacts = prior.artifact_manifest(root)
            equal({row["file"] for row in artifacts}, expected_artifacts(config.arm), "complete routing-contrast payloads before sealing")
            storage.json(root / "artifact_manifest.json", {"schema": SCHEMA, "files": artifacts})
            originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
            stage = "ordinary_saved_gradient_replay"
            published = _check(root, None, False, validation, raw, clock)
            equal(published["artifact_manifest_sha256"], old.digest(originals["artifact_manifest.json"]), "public binds original exact seal")
        storage.reconcile()
        stage = "publication"
        publication = parent._Publication(target, output, storage)
        publication.publish(published)
        frozen(manifest, validation)
        equal(_inventory(root), artifacts, "all routing-contrast bytes unchanged during publication")
        for name, payload in originals.items():
            equal((root / name).read_bytes(), payload, "original routing-contrast metadata unchanged during publication")
        if independent:
            equal(record.read_bytes(), public_bytes, "original producer publication unchanged during audit")
            equal((output / "audit_manifest.json").read_bytes(), private_bytes, "exact private audit manifest")
            equal((output / "verification.json").read_bytes(), verification_bytes, "exact private audit result")
            equal(read(output / "verification.json"), published, "private/public audit result agreement")
            equal({path.name for path in output.iterdir()}, {"audit_manifest.json", "declaration.md", "verification.json"}, "exact successful audit capsule")
        equal((output / "declaration.md").read_bytes(), declaration, "saved declaration unchanged")
        publication.verify()
        clock.check()
        publication.cleanup()
        cleanup_done = True
        clock.close()
        closed = True
        utc, monotonic = clock.check()
        print(json.dumps({"stage": "complete", "independent_audit": independent,
                          "completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                          "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic,
                          "record_sha256": publication.digest}), flush=True)
        return published
    except BaseException as error:
        if publication is not None:
            publication.retract()
        if created:
            _failure(output, clock, stage, error, independent, config.arm)
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


def check(output, record, arm=None):
    clock = Deadline()
    try:
        clock.arm()
        torch.set_num_threads(1)
        root = Path(output).resolve(strict=True)
        manifest = read(root / "manifest.json")
        if arm is not None:
            equal(arm, manifest["arm"], "requested checker arm matches producer")
        binding, raw, validation = bind_inputs(_bound_config(manifest))
        equal(binding, manifest["binding"], "fresh canonical routing-contrast checker binding")
        result = _check(root, record, False, validation, raw, clock)
        clock.check()
        return result
    finally:
        clock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("run", "check", "audit"):
        modes.add_argument("--" + name, action="store_true")
    for name in ("output", "record"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("prior-run", "prior-record", "prior-audit", "corpus", "audit-output", "audit-record"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.run and any(getattr(args, key) is None for key in ("prior_run", "prior_record", "prior_audit", "corpus")):
        parser.error("--run requires prior channel-choice run/record/successful audit and original corpus")
    if args.audit and (args.audit_output is None or args.audit_record is None):
        parser.error("--audit requires exclusive audit output/record paths")
    result = run(args) if args.run else audit(args) if args.audit else check(args.output, args.record, args.arm)
    print(json.dumps({"complete": result["complete"], "recipe_promoted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
