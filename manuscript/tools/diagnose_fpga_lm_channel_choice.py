"""Fixed original-start two-candidate projection control, not a new fast path.

All candidates remain in the unchanged full-dimensional max-plus selector.
MP/MB are frozen by applied zero gradients; raw gradients remain evidence.
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

from manuscript.tools import diagnose_fpga_lm_core_bias as core  # noqa: E402
from manuscript.tools import fpga_lm_projection_topology as topology  # noqa: E402

continuation, endpoint, parent = core.continuation, core.endpoint, core.parent
guard, helper, old, prior, pilot = core.guard, core.helper, core.old, core.prior, core.pilot
read, equal, load, tensor_hash = core.read, core.equal, core.load, core.tensor_hash
SCHEMA = "tikitorch.fpga_lm_channel_choice.v1"
ARM = "mixed_channel_choice"
FROZEN_REGIONS, TRAINABLE_REGIONS = core.FROZEN_REGIONS, core.TRAINABLE_REGIONS
TRAIN_WINDOWS, SCORE_TRAIN_WINDOWS, DEV_WINDOWS = core.TRAIN_WINDOWS, core.SCORE_TRAIN_WINDOWS, core.DEV_WINDOWS
BATCH_SIZE, SEQUENCE, UPDATES = core.BATCH_SIZE, core.SEQUENCE, core.UPDATES
DEADLINE_S, CELL_DEADLINE_S = core.DEADLINE_S, core.CELL_DEADLINE_S
STORAGE_CAP, FREE_RESERVE = core.STORAGE_CAP, core.FREE_RESERVE
Deadline, Storage = core.Deadline, core.Storage
query_batch, batch_data = parent.query_batch, parent.batch_data
applied_gradient, applied_record, frozen_state = core.applied_gradient, core.applied_record, core.frozen_state
PLAN = REPO / "manuscript/research/PLAN.md"
ALTERNATE_WEIGHT = -0.01
OFF_WEIGHT = -8192.0 * helper.BETA
PRODUCER_SHA256 = "b5d16c31d3bba43ce8ad7750ae62ff8b0fc278f8720e8f84b94008c90f13d23c"
PRODUCER_SEAL_SHA256 = "6ca008ad1681d2a0db14509ec47889fb4bbc4960c6637f3565a06db5185eb6ee"
AUDIT_MANIFEST_SHA256 = "0a1f2507d6d95616ccab735d0457dc76b97662fe9e8bef16a64f1d04bcd6d4dd"
AUDIT_RECEIPT_SHA256 = "c9185adcdef0db345fbcbd27d7372c9590a69f7979d688faeaa327196f2a7fcf"  # Actual completed proof is inserted only at reviewed admission.


def source_identity():
    values = dict(core.source_identity())
    for name in ("manuscript/tools/diagnose_fpga_lm_channel_choice.py", "tests/test_fpga_lm_channel_choice.py"):
        values[name] = old.digest((REPO / name).read_bytes())
    return values


def training_order():
    order = parent.training_order(0)
    equal(sorted(order), list(TRAIN_WINDOWS), "unchanged order-zero full training membership")
    return order


def mapping(shape=None):
    shape = pilot.layout(512) if shape is None else shape
    original = topology.mapping(shape)  # Strict supported D512/V128/N4/K7.
    lookup = {coordinate: (bank, index) for bank in ("B", "C", "D")
              for index, coordinate in enumerate(original[bank])}
    primary = original["target_by_layer"]
    alternate = []
    for layer in primary:
        targets = []
        for selected in layer:
            bank, index = lookup[selected]
            successor = original["inverse_sigma"][(original["sigma"][index] + 1) % 128]
            targets.append(original[bank][successor])
        alternate.append(targets)
    return {"schema": SCHEMA + ".mapping", "seed": 0,
            "geometry": {"vocab": 128, "width": 512, "layers": 4, "taps": 7},
            **{key: original[key] for key in ("A", "B", "C", "D", "sigma", "inverse_sigma")},
            "primary_by_layer": primary, "alternate_by_layer": alternate,
            "alternate_rule": "true-bank Y[inverse_sigma[(sigma[class(primary)]+1)%V]], not physical blocks"}


def _validate_mapping(spec):
    geometry = spec["geometry"]
    if set(geometry) != {"vocab", "width", "layers", "taps"}:
        raise ValueError("complete natural mapping geometry required")
    v, d, n, k = (geometry[key] for key in ("vocab", "width", "layers", "taps"))
    if any(type(value) is not int or value <= 0 for value in (v, d, n, k)) or d != 4 * v or v < 2:
        raise ValueError("literal four-bank geometry required")
    for bank in ("A", "B", "C", "D"):
        values = spec[bank]
        if not isinstance(values, list) or len(values) != v or any(type(x) is not int for x in values):
            raise ValueError("complete literal bank coordinates required")
    equal(sorted(sum((spec[bank] for bank in ("A", "B", "C", "D")), [])), list(range(d)), "disjoint exhaustive physical banks")
    for name in ("sigma", "inverse_sigma"):
        values = spec[name]
        if not isinstance(values, list) or any(type(x) is not int for x in values):
            raise ValueError("literal class permutations required")
        equal(sorted(values), list(range(v)), "complete class permutation")
    equal([spec["inverse_sigma"][x] for x in spec["sigma"]], list(range(v)), "actual inverse class permutation")
    p, a = spec["primary_by_layer"], spec["alternate_by_layer"]
    if not isinstance(p, list) or not isinstance(a, list) or len(p) != n or len(a) != n:
        raise ValueError("all layer maps required")
    for layer in range(n):
        expected_p, expected_a = [-1] * d, [-1] * d
        for source, target in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")):
            for index, row in enumerate(spec[source]):
                selected = spec["inverse_sigma"][(spec["sigma"][index] + (1 << layer)) % v]
                successor = spec["inverse_sigma"][(spec["sigma"][selected] + 1) % v]
                expected_p[row], expected_a[row] = spec[target][selected], spec[target][successor]
        for values, expected in ((p[layer], expected_p), (a[layer], expected_a)):
            if not isinstance(values, list) or any(type(x) is not int for x in values):
                raise ValueError("literal complete projection maps required")
            equal(values, expected, "canonical true-bank seeded map")
        if any(left == right for left, right in zip(p[layer], a[layer])):
            raise ValueError("alternate must never overwrite primary")
    return v, d, n, k


def _transform(weights, spec):
    """Validate a complete fitted family, then change only alternate MP slots."""
    helper.validate_weights(weights)
    v, d, n, k = _validate_mapping(spec)
    expected_shapes = {"EMBED": (v, d), "MH": (n, d, k), "MP": (n, d, d),
                       "MB": (n, d), "W_HEAD": (v, d)}
    for name, shape in expected_shapes.items():
        helper._finite(weights[name], "detached fitted parameter", detached=True)
        if tuple(weights[name].shape) != shape:
            raise ValueError("parameter geometry differs from exact mapping")
    expected = {name: torch.zeros_like(value) for name, value in weights.items()}
    anchors = weights["EMBED"][:, spec["A"]]
    raw = torch.round(anchors / helper.BETA)
    if (bool((raw > 0).any()) or bool((raw < -topology.initializer.MAX_MAGNITUDE).any()) or bool((raw.remainder(4) != 0).any())
            or bool((raw.max(dim=1).values != 0).any())):
        raise ValueError("bounded original fitted anchor multiples of four required")
    expected["EMBED"][:, spec["A"]] = raw * helper.BETA
    expected["MP"].fill_(OFF_WEIGHT)
    expected["W_HEAD"].fill_(OFF_WEIGHT)
    expected["W_HEAD"][torch.arange(v), spec["A"]] = 0.0
    for layer in range(n):
        expected["MP"][layer, torch.arange(d), spec["primary_by_layer"][layer]] = 0.0
    guard.byte_equal(weights, expected, "exact original mixed fitted family, including every unused/core/sentinel byte")
    result = {name: value.detach().clone() for name, value in weights.items()}
    for layer in range(n):
        result["MP"][layer, torch.arange(d), spec["alternate_by_layer"][layer]] = ALTERNATE_WEIGHT
    guard.byte_equal({name: result[name] for name in result if name != "MP"},
                     {name: weights[name] for name in weights if name != "MP"}, "all non-MP bytes preserved")
    return result, {"schema": SCHEMA + ".transform", "mapping": copy.deepcopy(spec),
                    "source_parameters_sha256": tensor_hash(weights), "transformed_parameters_sha256": tensor_hash(result),
                    "alternate_weight": ALTERNATE_WEIGHT, "primary_weight": 0.0, "other_weight": OFF_WEIGHT,
                    "changed_projection_slots": n * d, "non_projection_bytes_unchanged": True,
                    "scope": "initial tied-bank construction only; actual parity must be checked; unchanged full-D selector"}


def transform(weights, shape=None):
    return _transform(weights, mapping(shape))


def classify_routes(traces, spec):
    """Unweighted selections at ONE checkpoint, not across evolving parameters."""
    v, d, n, k = _validate_mapping(spec)
    if not isinstance(traces, (list, tuple)) or not traces:
        raise ValueError("nonempty ordered same-checkpoint traces required")
    primary = torch.tensor(spec["primary_by_layer"], dtype=torch.int64)[:, None, :]
    alternate = torch.tensor(spec["alternate_by_layer"], dtype=torch.int64)[:, None, :]
    hit_p, hit_a = torch.zeros((n, d), dtype=torch.bool), torch.zeros((n, d), dtype=torch.bool)
    counts = [{"primary": 0, "alternate": 0, "other": 0, "total": 0} for _ in range(n)]
    head_changed = head_total = 0
    for trace in traces:
        if not isinstance(trace, dict) or not set(helper.TRACE_KEYS).issubset(trace):
            raise ValueError("complete actual helper trace required")
        logits = trace["logits"]
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != v:
            raise ValueError("actual vocabulary logit geometry required")
        t = logits.shape[0]
        expected = {"logits": (t, v), "centering_offsets": (n, t), "AD_HEAD": (t, v),
                    "PROJ_IDX": (n, t, d), "TAP_IDX": (n, t, d)}
        for name, shape in expected.items():
            value = trace[name]
            dtype = torch.int64 if name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX") else torch.float64
            helper._finite(value, "actual routing trace", dtype=dtype, detached=True)
            if tuple(value.shape) != shape:
                raise ValueError("complete trace shape required")
        for name, limit in (("AD_HEAD", d), ("PROJ_IDX", d), ("TAP_IDX", k)):
            if bool((trace[name] < 0).any()) or bool((trace[name] >= limit).any()):
                raise ValueError("route outside full selector geometry")
        limits = torch.stack([torch.arange(t) // (1 << min(layer, 6)) for layer in range(n)])[:, :, None]
        if bool((trace["TAP_IDX"] > limits).any()):
            raise ValueError("noncausal temporal route")
        p, a = trace["PROJ_IDX"] == primary, trace["PROJ_IDX"] == alternate
        hit_p |= p.any(dim=1)
        hit_a |= a.any(dim=1)
        for layer in range(n):
            np, na, total = int(p[layer].sum()), int(a[layer].sum()), p[layer].numel()
            for key, value in (("primary", np), ("alternate", na), ("other", total - np - na), ("total", total)):
                counts[layer][key] += value
        head_changed += int((trace["AD_HEAD"] != torch.tensor(spec["A"])[None, :]).sum())
        head_total += trace["AD_HEAD"].numel()
    for layer in range(n):
        counts[layer].update(layer=layer, both_choice_rows=int((hit_p[layer] & hit_a[layer]).sum()), total_rows=d)
    return {"per_layer": counts,
            "total": {key: sum(row[key] for row in counts) for key in ("primary", "alternate", "other", "total", "both_choice_rows", "total_rows")},
            "head": {"departures_from_initial_anchor": head_changed, "total": head_total},
            "scope": "one fixed checkpoint across the listed positions/windows; other is a legitimate finite-sentinel winner"}


def recipe():
    values = core.recipe()
    values.update(schema=SCHEMA, arm=ARM, training_order=training_order(),
                  initialization="exact original fitted mixed state0 plus canonical alternate MP=-.01, genuinely empty Adam",
                  intervention="one true-bank seeded-successor alternate per MP row; primary zero, others unchanged sentinel; full MP/MB frozen",
                  mapping=mapping(), alternate_weight=ALTERNATE_WEIGHT,
                  raw_gradient_queries=UPDATES, applied_gradient_archives=UPDATES, optimizer_commits=UPDATES,
                  complete_saved_states=UPDATES + 1, sampled_training_windows=list(SCORE_TRAIN_WINDOWS),
                  development_windows=list(DEV_WINDOWS), score_window_readouts=2 * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
                  initial_scores="new candidate traces byte-equal bound ORIGINAL initial traces; no original-model forward",
                  new_initial_score_evaluations=len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS),
                  first_batch_parity="query0 inputs/targets/all traces/full raw G byte-exact versus saved core query0 before first commit; guard counters intentionally differ",
                  forward="unchanged full-D dynamic selector; existing guarded wrapper whole-model dense fallback; no two-column fast path",
                  controls="audited original first-pass mixed, frozen-core mixed and fitted-prior GRU endpoints; no new comparator run",
                  route_scope="exhaustive primary/alternate/other and BOTH-choice rows at EACH checkpoint; head and temporal changes separate; activation not a quality gate",
                  quality_gate="none; one fixed map/margin/order/budget, no global capacity or hardware claim")
    return values


def _audit_capsule(path, root, record):
    if (type(AUDIT_RECEIPT_SHA256) is not str or len(AUDIT_RECEIPT_SHA256) != 64
            or any(char not in "0123456789abcdef" for char in AUDIT_RECEIPT_SHA256)):
        raise ValueError("successful current core independent audit is not yet pinned")
    path = parent.rate.head._regular(path)
    equal(old.digest(path.read_bytes()), AUDIT_RECEIPT_SHA256, "actual successful core audit pin")
    equal(old.digest(record.read_bytes()), PRODUCER_SHA256, "actual complete core producer pin")
    equal(old.digest((root / "artifact_manifest.json").read_bytes()), PRODUCER_SEAL_SHA256, "actual core seal pin")
    value = read(path)
    expected = {"schema": core.SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                "run_directory": str(root), "record": str(record), "source_hashes": core.source_identity(),
                "producer_public_sha256": PRODUCER_SHA256, "producer_seal_sha256": PRODUCER_SEAL_SHA256,
                "audit_manifest_sha256": AUDIT_MANIFEST_SHA256,
                "deadline_seconds": core.DEADLINE_S, "cell_deadline_seconds": core.CELL_DEADLINE_S,
                "independent_tropical_window_vjps": 1024, "independent_tropical_window_forwards": 1152,
                "independent_tropical_batch_gradients": 256, "applied_gradient_masks_verified": 256,
                "frozen_regions_verified": list(FROZEN_REGIONS), "supplemental_guarded_window_forwards": 1152,
                "full_batch_gradients_verified": 256, "complete_saved_states_verified": 257,
                "optimizer_commits_replayed": 256, "score_window_readouts": 128,
                "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
                "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
                "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                "reserved_development_evaluated": False, "physical_fpga_execution": False,
                "paid_execution_authorized": False, "recipe_promoted": False}
    for key, wanted in expected.items():
        equal(value.get(key), wanted, "successful core audit: " + key)
    for key in ("verification_elapsed_utc_seconds", "verification_elapsed_monotonic_seconds"):
        elapsed = value[key]
        if type(elapsed) is not float or not math.isfinite(elapsed) or not 0 <= elapsed < core.DEADLINE_S:
            raise ValueError("actual successful audit must finish within declared bound")
    directory = Path(value["audit_directory"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("actual successful core audit directory required")
    directory = directory.resolve(strict=True)
    names = {"audit_manifest.json", "declaration.md", "verification.json"}
    equal({p.name for p in directory.iterdir()}, names, "exact successful core audit capsule")
    files = {name: old.file_identity(parent.rate.head._regular(directory / name)) for name in names}
    equal((directory / "verification.json").read_bytes(), path.read_bytes(), "exact successful private/public verification bytes")
    equal(files["audit_manifest.json"]["sha256"], AUDIT_MANIFEST_SHA256, "actual core audit manifest pin")
    manifest = read(directory / "audit_manifest.json")
    for key, wanted in {"schema": core.SCHEMA, "complete": False, "mode": "independent_audit",
                        "run_directory": str(root), "record": str(record), "audit_directory": str(directory),
                        "audit_record": str(path), "source_hashes": core.source_identity(),
                        "producer_public_sha256": PRODUCER_SHA256, "producer_seal_sha256": PRODUCER_SEAL_SHA256,
                        "runtime": value["runtime"], "started_at_utc": value["started_at_utc"],
                        "deadline_seconds": core.DEADLINE_S, "cell_deadline_seconds": core.CELL_DEADLINE_S,
                        "declaration_sha256": files["declaration.md"]["sha256"]}.items():
        equal(manifest[key], wanted, "successful core audit manifest: " + key)
    return value, {"receipt": old.file_identity(path), "files": files}


@dataclass(frozen=True)
class _Validation:
    binding: dict
    sources: dict
    core_manifest: dict
    core_validation: object

    def __post_init__(self):
        for key in ("binding", "sources", "core_manifest"):
            object.__setattr__(self, key, copy.deepcopy(getattr(self, key)))


def bind_inputs(config):
    sources = source_identity()
    root, record = Path(config.prior_run).resolve(strict=True), parent.rate.head._regular(config.prior_record)
    _, capsule = _audit_capsule(config.prior_audit, root, record)
    previous = read(root / "manifest.json")
    equal(previous["runtime"], parent.runtime(), "same original software runtime")
    ancestry, raw, token = core.bind_inputs(core._bound_config(previous))
    equal(ancestry, previous["binding"], "canonical fully byte-verified core ancestry")
    core.frozen(previous, token)
    artifacts = core._inventory(root)
    summary = read(root / "summary.json")
    equal(read(record), core.public_record(root, summary, artifacts), "actual complete audited core publication")
    equal(summary["complete"], True, "complete prerequisite core control")
    equal(Path(config.corpus).resolve(strict=True), Path(ancestry["dataset"]["path"]), "same corpus path")
    initial_paths = {
        "train": [Path(ancestry["continuation_binding"]["prior_run"]) / "initial_mixed" / f"score_{index:04d}.npz"
                  for index in SCORE_TRAIN_WINDOWS],
        "development": [Path(ancestry["endpoint_binding"]["prior_run"]) / "shared_mixed" / f"score_initial_{index:03d}_1.npz"
                        for index in DEV_WINDOWS]}
    binding = {"prior_run": str(root), "prior_record": str(record), "prior_audit": str(parent.rate.head._regular(config.prior_audit)),
               "audit_receipt_sha256": AUDIT_RECEIPT_SHA256, "audit_capsule": capsule,
               "core_binding": ancestry, "continuation_binding": ancestry["continuation_binding"],
               "endpoint_binding": ancestry["endpoint_binding"], "development_binding": ancestry["development_binding"],
               "dataset": ancestry["dataset"],
               "initial_scores": {key: [old.file_identity(path) for path in paths] for key, paths in initial_paths.items()},
               "initial_query": old.file_identity(root / core.ARM / "query_000000.npz"),
               "parent_files": {name: old.file_identity(root / name) for name in ("manifest.json", "artifact_manifest.json", "summary.json")},
               "parent_public": old.file_identity(record), "parent_artifacts": artifacts}
    validation = _Validation(binding, sources, previous, token)
    frozen({"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "binding": binding}, validation)
    return binding, raw, validation


def frozen(manifest, validation):
    if not isinstance(validation, _Validation):
        raise ValueError("private invocation-local channel-choice validation required")
    equal(manifest["schema"], SCHEMA, "channel-choice schema")
    equal(manifest["recipe"], recipe(), "fixed two-candidate control")
    equal(manifest["source_hashes"], source_identity(), "all77 source identities")
    equal(manifest["source_hashes"], validation.sources, "original invocation sources")
    equal(manifest["binding"], validation.binding, "canonical audited full ancestry")
    binding = manifest["binding"]
    equal(binding["audit_receipt_sha256"], AUDIT_RECEIPT_SHA256, "actual successful admission pin")
    core.frozen(validation.core_manifest, validation.core_validation)
    equal(core._inventory(Path(binding["prior_run"])), binding["parent_artifacts"], "all core artifacts unchanged")
    _, capsule = _audit_capsule(binding["prior_audit"], Path(binding["prior_run"]), Path(binding["prior_record"]))
    equal(capsule, binding["audit_capsule"], "complete core audit proof unchanged")
    for descriptor in (*binding["parent_files"].values(), binding["parent_public"], binding["initial_query"],
                       *binding["initial_scores"]["train"], *binding["initial_scores"]["development"]):
        equal(old.file_identity(descriptor["path"]), descriptor, "actual input raw/seal/public/initial-control identity")


def initial_model(binding):
    cell = core.initial_model(binding["core_binding"])
    changed, facts = transform(cell["parameters"])
    cell["parameters"] = changed
    cell["transform"] = facts
    return cell


def controls(binding):
    result = core.controls(binding["core_binding"])
    root = Path(binding["prior_run"])
    summary = read(root / "summary.json")
    result["frozen_core"] = {**summary["after"], "initial_source": summary["initial_source"],
        "paths": {"train": [str(root / core.ARM / "train" / f"score_{index:04d}.npz") for index in SCORE_TRAIN_WINDOWS],
                  "development": [str(root / core.ARM / f"score_final_{index:03d}_1.npz") for index in DEV_WINDOWS]}}
    equal(result["frozen_core"]["initial_source"], result["mixed"]["original_initial"]["source_state"],
          "audited core and original controls start at exact same actual source")
    return result


def _dense_trace(arrays):
    trace = {name: arrays[name] for name in helper.TRACE_KEYS}
    counts = arrays["guard_counts"]
    n, t = trace["PROJ_IDX"].shape[:2]
    guard.byte_equal({"guard_counts": counts},
                     {"guard_counts": torch.tensor([[0, 1, 0, n * t, 0, t]], dtype=torch.int64)},
                     "two-active MP invokes unchanged whole-model dense fallback")
    return trace


def first_batch_parity(path, descriptor):
    equal(old.file_identity(descriptor["path"]), descriptor, "bound actual original core query0")
    original, actual = load(descriptor["path"]), load(path)
    # Deliberately exclude only guard bookkeeping: original one-support rows
    # could certify fast selectors, while this model must use dense fallback.
    guard.byte_equal({key: value for key, value in actual.items() if key != "guard_counts"},
                     {key: value for key, value in original.items() if key != "guard_counts"},
                     "actual query0 inputs/targets/all traces/full RAW G equal original core before any commit")
    equal(old.file_identity(descriptor["path"]), descriptor, "original first-query bytes unchanged")
    return {"complete": True, "original_query": descriptor, "new_query": old.file_identity(path),
            "all_inputs_targets_traces_raw_gradient_bytes_equal": True, "guard_counters_compared": False,
            "additional_gradient_queries": 0, "optimizer_commits_before_check": 0}


def _score_path(directory, cohort, index, label):
    return (Path(directory) / "train" / f"score_{index:04d}.npz" if cohort == "train"
            else Path(directory) / f"score_{label}_{index:03d}_1.npz")


def collect_pair(directory, cell, raw, dataset, label, *, write=False, recompute=False, storage=None, clock=None):
    train = endpoint.collect_scores(Path(directory) / "train", cell, raw, dataset, write=write,
                                    recompute=recompute, storage=storage, clock=clock)
    with torch.no_grad():
        dev = parent.collect_scores(directory, label, cell, raw, dataset, True,
                                    write=write, recompute=recompute, storage=storage)
    return {"train": {"per_window": train, "mean": parent._mean(train)},
            "development": {"per_window": dev, "mean": parent._mean(dev)}}


def initial_parity(directory, scores, binding, spec):
    rows, support = {}, {}
    for cohort, indices in (("train", SCORE_TRAIN_WINDOWS), ("development", DEV_WINDOWS)):
        equal([row["window"] for row in scores[cohort]["per_window"]], list(indices), "complete new initial cohort")
        descriptors = binding["initial_scores"][cohort]
        equal(len(descriptors), len(indices), "complete original initial trace cohort")
        rows[cohort], traces = [], []
        for index, descriptor in zip(indices, descriptors):
            equal(old.file_identity(descriptor["path"]), descriptor, "actual archived ORIGINAL initial trace")
            original = load(descriptor["path"])
            path = _score_path(directory, cohort, index, "initial")
            trace = _dense_trace(load(path))
            guard.byte_equal(trace, {name: original[name] for name in helper.TRACE_KEYS},
                             "128 scheduled candidate initial traces byte-exact to original initial function/routes")
            traces.append(trace)
            rows[cohort].append({"window": index, "original": descriptor, "candidate": old.file_identity(path),
                                 "trace_tensor_sha256": tensor_hash(trace), "byte_exact": True})
        support[cohort] = classify_routes(traces, spec)
    return {"schema": SCHEMA + ".initial_parity", "complete": True, "cohorts": rows, "support": support,
            "score_window_parity_checks": sum(len(value) for value in rows.values()),
            "additional_original_model_forwards": 0, "guard_counters_compared_to_original": False}


def comparisons(after, inherited):
    result = core.comparisons(after, inherited)
    for cohort, indices in (("train", SCORE_TRAIN_WINDOWS), ("development", DEV_WINDOWS)):
        current, frozen_core = after[cohort], inherited["frozen_core"][cohort]
        equal([row["window"] for row in frozen_core["per_window"]], list(indices), "complete audited frozen-core cohort")
        equal(parent._mean(frozen_core["per_window"]), frozen_core["mean"], "actual frozen-core mean")
        result[cohort].update(frozen_core=frozen_core["mean"],
                             new_minus_frozen_core_ce=current["mean"]["ce"] - frozen_core["mean"]["ce"],
                             new_minus_frozen_core_accuracy=current["mean"]["accuracy"] - frozen_core["mean"]["accuracy"])
        for index, row in enumerate(result[cohort]["per_window"]):
            row["new_minus_frozen_core_ce"] = current["per_window"][index]["ce"] - frozen_core["per_window"][index]["ce"]
            row["accuracy_differences"] = {
                name: current["per_window"][index]["accuracy"] - inherited[name][cohort]["per_window"][index]["accuracy"]
                for name in ("mixed", "prior_gru", "frozen_core")}
    return result


def execute(manifest, output, raw, *, write=False, recompute=False, clock=None, storage=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal channel-choice execution modes required")
    equal(UPDATES * BATCH_SIZE, len(TRAIN_WINDOWS), "one exact original-order training pass")
    output, binding = Path(output), manifest["binding"]
    clock = clock or Deadline()
    if write and storage is None:
        storage = Storage(output, clock)
        storage.used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    clock.begin_cell()
    directory, initial_directory = output / ARM, output / "initial"
    if write:
        for path in (directory, directory / "train", initial_directory, initial_directory / "train"):
            path.mkdir(exist_ok=False)
    inherited = controls(binding)
    cell = initial_model(binding)
    equal(cell["initial_source"], inherited["mixed"]["original_initial"]["source_state"],
          "original initial score and transformed-model source are the same actual archived checkpoint")
    facts, spec = cell["transform"], cell["transform"]["mapping"]
    if write:
        storage.json(output / "transform.json", facts)
    equal(read(output / "transform.json"), facts, "exact immutable initialization transform and map")
    optimizer = parent.make_optimizer("mixed", cell["parameters"].values())
    equal(len(optimizer.state), 0, "genuinely empty original Adam state")
    frozen_initial = {key: cell["parameters"][key].detach().clone() for key in FROZEN_REGIONS}
    # Head remains trainable. This old one-support observer is used ONLY for
    # head state; projection selections use the explicit two-candidate maps.
    head_reference = {"W_HEAD": continuation.support_reference(cell)["W_HEAD"]}
    invariants = [frozen_state(cell, optimizer, frozen_initial, 0)]
    head_states = [{"step": 0, "regions": None, "optimizer_moments_empty": True}]
    states = [parent.state_record(directory, 0, cell, optimizer, "mixed", write=write, storage=storage)]
    print(json.dumps({"stage": "candidate_initial_parity", "recompute": recompute, "additional_original_forwards": 0}), flush=True)
    initial_scores = collect_pair(initial_directory, cell, raw, binding["dataset"], "initial", write=write,
                                  recompute=recompute, storage=storage, clock=clock)
    parity = initial_parity(initial_directory, initial_scores, binding, spec)
    equal(parent.state_record(directory, 0, cell, optimizer, "mixed"), states[0], "initial readouts preserve full empty-Adam state")
    order, query_rows, applied_rows = training_order(), [], []
    for step in range(UPDATES):
        clock.check()
        indices = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        tokens, targets = batch_data(raw, binding["dataset"], indices)
        before = tensor_hash(cell["parameters"])
        raw_gradients, row = query_batch(directory, step, cell, tokens, targets, indices,
                                        write=write, recompute=recompute, storage=storage)
        query_path = directory / f"query_{step:06d}.npz"
        arrays = load(query_path)
        traces = []
        for position, index in enumerate(indices):
            trace = {key: arrays[f"window_{index}/{key}"] for key in helper.TRACE_KEYS}
            traces.append(_dense_trace({**trace, "guard_counts": arrays["guard_counts"][position:position + 1]}))
        row["channel_choices"] = classify_routes(traces, spec)
        if step == 0:
            parity["first_batch"] = first_batch_parity(query_path, binding["initial_query"])
            if write:
                storage.json(output / "initial_parity.json", parity)
            equal(read(output / "initial_parity.json"), parity, "all initial parity checks complete BEFORE first update")
        applied, applied_row = applied_record(directory, step, raw_gradients, write=write, storage=storage)
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in cell["parameters"].items():
            parameter.grad = applied[name].detach().clone()
        optimizer.step()
        invariants.append(frozen_state(cell, optimizer, frozen_initial, step + 1))
        states.append(parent.state_record(directory, step + 1, cell, optimizer, "mixed", write=write, storage=storage))
        head_states.append({"step": step + 1, "regions": continuation.support_state(cell, optimizer, head_reference),
                            "optimizer_moments_empty": False})
        row.update(before_parameters_sha256=before, after_parameters_sha256=tensor_hash(cell["parameters"]),
                   applied_gradient_sha256=applied_row["applied_gradient_sha256"], state_sha256=states[-1]["full_state_tensor_sha256"],
                   saved_applied_gradient_adam_replay=True)
        query_rows.append(row)
        applied_rows.append(applied_row)
        clock.check()
        if (step + 1) % 16 == 0:
            print(json.dumps({"stage": "two_candidate_update", "step": step + 1,
                              "query_ce": parent._mean(row["per_window"])["ce"], "recompute": recompute}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    before_scores = parent.state_record(directory, UPDATES, cell, optimizer, "mixed")
    after = collect_pair(directory, cell, raw, binding["dataset"], "final", write=write,
                         recompute=recompute, storage=storage, clock=clock)
    equal(parent.state_record(directory, UPDATES, cell, optimizer, "mixed"), before_scores, "final scores preserve complete Adam state")
    frozen_state(cell, optimizer, frozen_initial, UPDATES)
    changes, final_choices = {}, {}
    for cohort, indices in (("train", SCORE_TRAIN_WINDOWS), ("development", DEV_WINDOWS)):
        equal([row["window"] for row in after[cohort]["per_window"]], list(indices), "complete matched final cohort")
        traces = [_dense_trace(load(_score_path(directory, cohort, index, "final"))) for index in indices]
        final_choices[cohort] = classify_routes(traces, spec)
        changes[cohort] = {}
        for control in ("mixed", "frozen_core"):
            paths = inherited[control]["paths"][cohort]
            equal(len(paths), len(indices), "complete actual control final-trace inventory")
            paired = [{"window": index, "routes": continuation.route_changes(load(path), trace)}
                      for index, path, trace in zip(indices, paths, traces)]
            changes[cohort][control] = {"per_window": paired,
                "totals": {key: {field: sum(item["routes"][key][field] for item in paired) for field in ("changed", "total")}
                           for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}
    for key in ("file", "metadata"):
        descriptor = cell["initial_source"][key]
        equal(old.file_identity(descriptor["path"]), descriptor, "actual original source checkpoint remains untouched")
    # Do not merge traces across updates: BOTH-choice is computed independently
    # above. These sums count row-checkpoints, never distinct fixed rows.
    choices = [row["channel_choices"] for row in query_rows]
    query_totals = {key: sum(value["total"][key] for value in choices) for key in ("primary", "alternate", "other", "total")}
    query_totals.update(both_choice_row_checkpoints=sum(value["total"]["both_choice_rows"] for value in choices),
                        total_row_checkpoints=sum(value["total"]["total_rows"] for value in choices))
    result = {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
              "arm": ARM, "order_seed": 0, "initial_source": cell["initial_source"], "transform": facts,
              "initial_scores": initial_scores, "initial_parity": parity, "training_order": order,
              "inherited_state_format": {"schema": parent.SCHEMA, "arm": "mixed", "steps": list(range(UPDATES + 1))},
              "inherited_controls": inherited, "after": after, "comparisons": comparisons(after, inherited),
              "fixed_prior_baselines": read(Path(binding["prior_run"]) / "summary.json")["fixed_prior_baselines"],
              "query_rows": query_rows, "applied_rows": applied_rows, "states": states,
              "frozen_region_invariants": invariants, "head_support_states": head_states,
              "query_channel_choice_totals": query_totals, "final_channel_choices": final_choices,
              "between_recipe_final_route_changes": changes,
              "route_scope": "query BOTH counts are row-checkpoints; final cohorts are separate fixed checkpoints; cross-recipe final differences are not within-trajectory stability or gradient mass",
              "raw_gradient_queries": UPDATES, "applied_gradient_archives": UPDATES, "optimizer_commits": UPDATES,
              "complete_saved_states": UPDATES + 1, "score_window_readouts": 2 * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
              "initial_score_parity_checks": len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS), "first_batch_parity_checks": 1,
              "producer_whole_model_dense_forward_windows": UPDATES * BATCH_SIZE + 2 * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
              "new_gradient_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
              "inherited_fitting_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE, "inherited_gradient_pair_exposures": 0,
              "endpoint_history_pair_exposures": 2 * len(TRAIN_WINDOWS) * SEQUENCE, "distinct_gradient_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
              "new_fitting_pair_exposures": 0, "new_baseline_evaluations": 0,
              "new_initial_score_evaluations": len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS),
              "new_original_model_forwards": 0, "new_gru_training": False, "heldout_test_evaluated": False,
              "reserved_development_evaluated": False, "physical_fpga_execution": False,
              "paid_execution_authorized": False, "recipe_promoted": False,
              "scope": "single original-start fixed two-candidate routing control; activation is descriptive, not a quality gate; no global capacity or hardware claim"}
    clock.end_cell()
    return result


def expected_artifacts():
    names = {"manifest.json", "protocol.json", "declaration.md", "summary.json", "transform.json", "initial_parity.json"}
    names.update(f"{ARM}/{prefix}_{step:06d}.npz" for step in range(UPDATES) for prefix in ("query", "applied"))
    names.update(f"{ARM}/state_{step:06d}.{suffix}" for step in range(UPDATES + 1) for suffix in ("npz", "json"))
    for directory, label in ((ARM, "final"), ("initial", "initial")):
        names.update(f"{directory}/train/score_{index:04d}.npz" for index in SCORE_TRAIN_WINDOWS)
        names.update(f"{directory}/score_{label}_{index:03d}_1.npz" for index in DEV_WINDOWS)
    return names


def _inventory(root):
    root = Path(root)
    artifacts = prior.artifact_manifest(root)
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact full channel-choice payload inventory")
    equal({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()},
          {ARM, ARM + "/train", "initial", "initial/train"}, "only declared channel-choice directories")
    equal(read(root / "artifact_manifest.json"), {"schema": SCHEMA, "files": artifacts}, "complete sealed control payloads")
    return artifacts


def public_record(root, result, artifacts):
    return {**result, "run_directory": str(root), "runtime": read(Path(root) / "manifest.json")["runtime"],
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "independent_audit_required": True,
            "check_scope": "initial parity/raw-applied masks/full saved-applied-G Adam states exact; separate independent raw-G/score audit required"}


_bound_config = core._bound_config

def _check(root, record, recompute, validation, raw, clock):
    if type(recompute) is not bool:
        raise ValueError("literal independent replay mode required")
    root = Path(root).resolve(strict=True)
    record = None if record is None else parent.rate.head._regular(record)
    originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
    public_bytes = None if record is None else record.read_bytes()
    manifest = read(root / "manifest.json")
    equal(manifest["run_directory"], str(root), "actual channel-choice output path")
    equal(manifest["runtime"], parent.runtime(), "unchanged current software runtime")
    equal(read(root / "protocol.json"), recipe(), "saved fixed channel-choice protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["declaration_sha256"], "actual saved declaration")
    frozen(manifest, validation)
    artifacts = _inventory(root)
    result = execute(manifest, root, raw, recompute=recompute, clock=clock)
    equal(read(root / "summary.json"), result, "complete recomputed channel-choice summary")
    published = public_record(root, result, artifacts)
    if record is not None:
        equal(read(record), published, "actual complete channel-choice publication")
    frozen(manifest, validation)
    equal(_inventory(root), artifacts, "all channel-choice payload bytes unchanged during replay")
    for name, payload in originals.items():
        equal((root / name).read_bytes(), payload, "exact original channel-choice metadata bytes")
    if record is not None:
        equal(record.read_bytes(), public_bytes, "exact original public bytes")
    clock.check()
    return published


def _failure(output, clock, stage, error, independent=False):
    if output is not None and output.is_dir() and not (output / "failure.json").exists():
        utc, monotonic = clock.elapsed()
        rejected = output / "rejected_public.json"
        retained = old.file_identity(rejected) if rejected.is_file() and not rejected.is_symlink() else None
        pilot.write_json(output / "failure.json", {"schema": SCHEMA, "complete": False, "stage": stage,
            "type": type(error).__name__, "error": str(error)[:100000], "retained_rejected_public": retained,
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
            originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
            public_bytes = record.read_bytes()
            artifacts = _inventory(root)
            storage.base_bytes = sum(row["bytes"] for row in artifacts) + len(originals["artifact_manifest.json"]) + len(public_bytes)
            storage.used = storage.base_bytes
            binding, raw, validation = bind_inputs(_bound_config(manifest))
            equal(binding, manifest["binding"], "fresh canonical channel-choice audit binding")
            private = {"schema": SCHEMA, "complete": False, "mode": "independent_audit",
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
        stage = "independent_numerical_replay" if independent else "fixed_channel_choice_control"
        if independent:
            result = _check(root, record, True, validation, raw, clock)
            utc, monotonic = clock.check()
            windows, scores = UPDATES * BATCH_SIZE, 2 * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS))
            published = {"schema": SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                         "run_directory": str(root), "record": str(record), "audit_directory": str(output),
                         "producer_public_sha256": old.digest(public_bytes), "producer_seal_sha256": old.digest(originals["artifact_manifest.json"]),
                         "audit_manifest_sha256": old.digest(private_bytes), "source_hashes": sources, "runtime": parent.runtime(),
                         "started_at_utc": guard.parent._utc(clock.started_utc),
                         "verification_completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                         "verification_elapsed_utc_seconds": utc, "verification_elapsed_monotonic_seconds": monotonic,
                         "deadline_seconds": DEADLINE_S, "cell_deadline_seconds": CELL_DEADLINE_S,
                         "independent_tropical_window_vjps": windows, "independent_tropical_window_forwards": windows + scores,
                         "independent_tropical_batch_gradients": UPDATES, "applied_gradient_masks_verified": UPDATES,
                         "frozen_regions_verified": list(FROZEN_REGIONS), "supplemental_guarded_window_forwards": windows + scores,
                         "initial_score_parity_checks": result["initial_score_parity_checks"], "first_batch_parity_checks": 1,
                         "all_whole_model_dense_fallback_counts_verified": True,
                         "full_batch_gradients_verified": result["raw_gradient_queries"],
                         "complete_saved_states_verified": result["complete_saved_states"],
                         "optimizer_commits_replayed": result["optimizer_commits"], "score_window_readouts": result["score_window_readouts"],
                         "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
                         "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
                         "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                         "reserved_development_evaluated": False, "physical_fpga_execution": False,
                         "paid_execution_authorized": False, "recipe_promoted": False,
                         "scope": "independent reconstruction of the fixed channel-choice control and final readouts, not new learning, seed confirmation or promotion"}
            storage.json(output / "verification.json", published)
            verification_bytes = (output / "verification.json").read_bytes()
            equal(read(output / "verification.json"), published, "actual independent verification result")
        else:
            result = execute(manifest, output, raw, write=True, storage=storage, clock=clock)
            storage.json(output / "summary.json", result)
            artifacts = prior.artifact_manifest(root)
            equal({row["file"] for row in artifacts}, expected_artifacts(), "complete channel-choice payloads before sealing")
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
        equal(_inventory(root), artifacts, "all channel-choice bytes unchanged during publication")
        for name, payload in originals.items():
            equal((root / name).read_bytes(), payload, "original channel-choice metadata unchanged during publication")
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
            _failure(output, clock, stage, error, independent)
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
        equal(binding, manifest["binding"], "fresh canonical channel-choice checker binding")
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
        parser.error("--run requires prior core run/record/successful audit and original corpus")
    if args.audit and (args.audit_output is None or args.audit_record is None):
        parser.error("--audit requires exclusive audit output/record paths")
    result = run(args) if args.run else audit(args) if args.audit else check(args.output, args.record)
    print(json.dumps({"complete": result["complete"], "recipe_promoted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
