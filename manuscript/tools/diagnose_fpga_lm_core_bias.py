"""One-pass original-start control with MP/MB zero applied gradients.

Raw gradients remain unmodified evidence. Applied gradients alone freeze the
two regions; this is an optimization-control ablation, not a new architecture
or a claim that all projection parameters are globally redundant.
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

from manuscript.tools import continue_fpga_lm_development as continuation  # noqa: E402

endpoint, parent = continuation.endpoint, continuation.parent
guard, helper, old, prior, pilot = continuation.guard, continuation.helper, continuation.old, continuation.prior, continuation.pilot
read, equal, load, tensor_hash = continuation.read, continuation.equal, continuation.load, continuation.tensor_hash
SCHEMA = "tikitorch.fpga_lm_core_bias.v1"
ARM = "mixed_core_bias_frozen"
FROZEN_REGIONS = ("MP", "MB")
TRAINABLE_REGIONS = ("EMBED", "MH", "W_HEAD")
TRAIN_WINDOWS, SCORE_TRAIN_WINDOWS, DEV_WINDOWS = parent.TRAIN_WINDOWS, endpoint.TRAIN_WINDOWS, parent.DEV_WINDOWS
BATCH_SIZE, SEQUENCE, UPDATES = parent.BATCH_SIZE, parent.SEQUENCE, 256
DEADLINE_S, CELL_DEADLINE_S = 10800, 3600
STORAGE_CAP, FREE_RESERVE = 8 * (1 << 30), 4 * (1 << 30)
Deadline, Storage = continuation.Deadline, continuation.Storage
query_batch, batch_data = parent.query_batch, parent.batch_data
PLAN = REPO / "manuscript/research/PLAN.md"
PRODUCER_SHA256 = "07f18f23ce6cde30da7a6816eed661c4c901c4faeb96aa8f411149cebe69b813"
# Admission is impossible until actual successful independent completion.
AUDIT_RECEIPT_SHA256 = "ebc8bf0b4b4b3b723120ba94c0927818d70159e0b06fd24bc86ea4d63666b25c"


def source_identity():
    values = dict(continuation.source_identity())
    for name in ("manuscript/tools/diagnose_fpga_lm_core_bias.py", "tests/test_fpga_lm_core_bias.py"):
        values[name] = old.digest((REPO / name).read_bytes())
    return values


def training_order():
    order = parent.training_order(0)
    equal(sorted(order), list(TRAIN_WINDOWS), "exact unchanged order-zero membership")
    return order


def recipe():
    return {"schema": SCHEMA, "arm": ARM, "order_seed": 0, "training_order": training_order(),
            "initialization": "actual original mixed fitted state0 and empty Adam, never a trained endpoint",
            "frozen_regions": list(FROZEN_REGIONS), "trainable_regions": list(TRAINABLE_REGIONS),
            "freezing_rule": "preserve raw G; set only applied MP/MB G to +0; zero initial moments keep exact frozen weights/moments; counters advance",
            "batch_size": BATCH_SIZE, "sequence": SEQUENCE, "updates": UPDATES, "learning_rate": 0.01,
            "dtype": "float64", "device": "cpu", "torch_threads": 1, "context": True,
            "optimizer": "unchanged eager Adam .9/.999 eps1e-8, zero decay, no foreach/fused/clipping",
            "loss": "exact mean CE over full batch in dimensionless units; no second beta",
            "raw_gradient_queries": UPDATES, "applied_gradient_archives": UPDATES,
            "optimizer_commits": UPDATES, "complete_saved_states": UPDATES + 1,
            "state_format": parent.SCHEMA, "state_format_arm": "mixed",
            "sampled_training_windows": list(SCORE_TRAIN_WINDOWS), "development_windows": list(DEV_WINDOWS),
            "score_window_readouts": len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS),
            "initial_scores": "reuse bound original initial per-window evidence only",
            "controls": "reuse frozen first-pass order-zero mixed and GRU endpoints; no new comparator trajectory",
            "audit": "independent raw G/dense VJP; exact mask validation; saved applied G drives exact full-Adam replay; all scores and supplemental guards",
            "new_gradient_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
            "inherited_fitting_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
            "inherited_gradient_pair_exposures": 0, "endpoint_history_pair_exposures": 2 * len(TRAIN_WINDOWS) * SEQUENCE,
            "distinct_training_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
            "deadline_seconds": DEADLINE_S, "cell_deadline_seconds": CELL_DEADLINE_S,
            "combined_new_output_cap_bytes": STORAGE_CAP, "free_reserve_bytes": FREE_RESERVE,
            "new_model_count_fit": False, "new_baseline_evaluations": 0, "new_initial_score_evaluations": 0,
            "new_gru_training": False, "adaptive_selection": False, "recipe_promoted": False,
            "heldout_test_evaluated": False, "reserved_development_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False,
            "quality_gate": "none; fixed one-order region-freezing control; no global redundancy or convergence claim"}


def _audit_capsule(path, root, record):
    if (type(AUDIT_RECEIPT_SHA256) is not str or len(AUDIT_RECEIPT_SHA256) != 64
            or any(char not in "0123456789abcdef" for char in AUDIT_RECEIPT_SHA256)):
        raise ValueError("successful current continuation independent audit is not yet pinned")
    path = parent.rate.head._regular(path)
    equal(old.digest(path.read_bytes()), AUDIT_RECEIPT_SHA256, "actual successful continuation audit pin")
    equal(old.digest(record.read_bytes()), PRODUCER_SHA256, "actual continuation publication pin")
    value = read(path)
    expected = {"schema": continuation.SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                "run_directory": str(root), "record": str(record), "source_hashes": continuation.source_identity(),
                "producer_public_sha256": PRODUCER_SHA256,
                "producer_seal_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
                "deadline_seconds": continuation.DEADLINE_S, "cell_deadline_seconds": continuation.CELL_DEADLINE_S,
                "independent_tropical_window_vjps": 1024, "independent_tropical_window_forwards": 1152,
                "independent_tropical_batch_gradients": 256, "independent_gru_batch_gradients": 256,
                "independent_gru_single_window_scores": 128, "supplemental_guarded_window_forwards": 1152,
                "full_batch_gradients_verified": 512, "complete_saved_states_verified": 514,
                "optimizer_commits_replayed": 512, "score_window_readouts": 256,
                "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
                "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
                "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                "reserved_development_evaluated": False, "physical_fpga_execution": False,
                "paid_execution_authorized": False, "recipe_promoted": False}
    for key, wanted in expected.items():
        equal(value.get(key), wanted, "successful continuation audit: " + key)
    for key in ("verification_elapsed_utc_seconds", "verification_elapsed_monotonic_seconds"):
        elapsed = value[key]
        if type(elapsed) is not float or not math.isfinite(elapsed) or not 0 <= elapsed < continuation.DEADLINE_S:
            raise ValueError("actual successful audit must finish within declared bound")
    directory = Path(value["audit_directory"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("actual successful continuation audit directory required")
    directory = directory.resolve(strict=True)
    names = {"audit_manifest.json", "declaration.md", "verification.json"}
    equal({p.name for p in directory.iterdir()}, names, "exact successful continuation audit capsule")
    files = {name: old.file_identity(parent.rate.head._regular(directory / name)) for name in names}
    equal((directory / "verification.json").read_bytes(), path.read_bytes(), "exact successful private/public verification bytes")
    equal(files["audit_manifest.json"]["sha256"], value["audit_manifest_sha256"], "actual audited manifest identity")
    manifest = read(directory / "audit_manifest.json")
    for key, wanted in {"schema": continuation.SCHEMA, "complete": False, "mode": "independent_audit",
                        "run_directory": str(root), "record": str(record), "audit_directory": str(directory),
                        "audit_record": str(path), "source_hashes": continuation.source_identity(),
                        "producer_public_sha256": PRODUCER_SHA256, "producer_seal_sha256": value["producer_seal_sha256"],
                        "runtime": value["runtime"], "started_at_utc": value["started_at_utc"],
                        "deadline_seconds": continuation.DEADLINE_S, "cell_deadline_seconds": continuation.CELL_DEADLINE_S,
                        "declaration_sha256": files["declaration.md"]["sha256"]}.items():
        equal(manifest[key], wanted, "successful continuation audit manifest: " + key)
    return value, {"receipt": old.file_identity(path), "files": files}


@dataclass(frozen=True)
class _Validation:
    binding: dict
    sources: dict
    continuation_manifest: dict
    continuation_validation: object

    def __post_init__(self):
        for key in ("binding", "sources", "continuation_manifest"):
            object.__setattr__(self, key, copy.deepcopy(getattr(self, key)))


def bind_inputs(config):
    sources = source_identity()
    root, record = Path(config.prior_run).resolve(strict=True), parent.rate.head._regular(config.prior_record)
    _, capsule = _audit_capsule(config.prior_audit, root, record)
    previous = read(root / "manifest.json")
    equal(previous["runtime"], parent.runtime(), "same original software runtime")
    ancestry, raw, token = continuation.bind_inputs(continuation._bound_config(previous))
    equal(ancestry, previous["binding"], "canonical fully byte-verified continuation ancestry")
    continuation.frozen(previous, token)
    artifacts = continuation._inventory(root)
    summary = read(root / "summary.json")
    equal(read(record), continuation.public_record(root, summary, artifacts), "actual complete audited continuation publication")
    equal(summary["complete"], True, "all prerequisite continuation cells complete")
    equal(Path(config.corpus).resolve(strict=True), Path(ancestry["dataset"]["path"]), "same corpus path")
    binding = {"prior_run": str(root), "prior_record": str(record), "prior_audit": str(parent.rate.head._regular(config.prior_audit)),
               "audit_receipt_sha256": AUDIT_RECEIPT_SHA256, "audit_capsule": capsule,
               "continuation_binding": ancestry, "endpoint_binding": ancestry["endpoint_binding"],
               "development_binding": ancestry["endpoint_binding"]["parent_binding"], "dataset": ancestry["dataset"],
               "parent_files": {name: old.file_identity(root / name) for name in ("manifest.json", "artifact_manifest.json", "summary.json")},
               "parent_public": old.file_identity(record), "parent_artifacts": artifacts}
    validation = _Validation(binding, sources, previous, token)
    frozen({"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "binding": binding}, validation)
    return binding, raw, validation


def frozen(manifest, validation):
    if not isinstance(validation, _Validation):
        raise ValueError("private invocation-local core-bias validation required")
    equal(manifest["schema"], SCHEMA, "core-bias schema")
    equal(manifest["recipe"], recipe(), "fixed original-start freeze control")
    equal(manifest["source_hashes"], source_identity(), "all75 source identities")
    equal(manifest["source_hashes"], validation.sources, "original invocation sources")
    equal(manifest["binding"], validation.binding, "canonical audited full ancestry")
    binding = manifest["binding"]
    equal(binding["audit_receipt_sha256"], AUDIT_RECEIPT_SHA256, "actual successful admission pin")
    continuation.frozen(validation.continuation_manifest, validation.continuation_validation)
    equal(continuation._inventory(Path(binding["prior_run"])), binding["parent_artifacts"], "all continuation artifacts unchanged")
    _, capsule = _audit_capsule(binding["prior_audit"], Path(binding["prior_run"]), Path(binding["prior_record"]))
    equal(capsule, binding["audit_capsule"], "complete audit proof unchanged")
    for descriptor in (*binding["parent_files"].values(), binding["parent_public"]):
        equal(old.file_identity(descriptor["path"]), descriptor, "actual input raw/seal/public identity")


def initial_model(binding):
    cell = parent.initial_models(binding["development_binding"])["mixed"]
    source = Path(binding["endpoint_binding"]["prior_run"]) / parent.cell_name("mixed", 0) / "state_000000.npz"
    parameters, metadata = endpoint._checkpoint(source, cell, "mixed", 0)
    guard.byte_equal(cell["parameters"], parameters, "exact original mixed order-zero initial parameter bytes")
    cell["initial_source"] = {"file": old.file_identity(source), "metadata": old.file_identity(source.with_suffix(".json")),
                              "full_state_tensor_sha256": metadata["full_state_tensor_sha256"], "step": 0}
    return cell


def applied_gradient(raw):
    if not isinstance(raw, dict) or set(raw) != set(FROZEN_REGIONS) | set(TRAINABLE_REGIONS):
        raise ValueError("all five raw gradient regions required")
    helper.validate_weights(raw)  # Natural finite FP64 geometry, no integer or magnitude bound.
    result = {}
    for name, value in raw.items():
        helper._finite(value, "raw gradient", detached=True)
        result[name] = torch.zeros_like(value) if name in FROZEN_REGIONS else value.detach().clone()
    return result


def applied_record(directory, step, raw, *, write=False, storage=None):
    if type(step) is not int or not 0 <= step < UPDATES or type(write) is not bool:
        raise ValueError("literal local applied-gradient step and write flag required")
    expected = applied_gradient(raw)
    path = Path(directory) / f"applied_{step:06d}.npz"
    if write:
        storage.arrays(path, expected)
    saved = load(path)
    guard.byte_equal(saved, expected, "exact applied raw-G mask, including unchanged trainable bytes")
    return saved, {"step": step, "raw_gradient_sha256": tensor_hash(raw), "applied_gradient_sha256": tensor_hash(saved),
                   "file": path.name, "file_sha256": old.digest(path.read_bytes()),
                   "frozen_regions": list(FROZEN_REGIONS), "trainable_regions_unchanged": True}


def frozen_state(cell, optimizer, initial, step):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError("literal frozen-state step required")
    guard.byte_equal({key: cell["parameters"][key].detach() for key in FROZEN_REGIONS}, initial,
                     "all frozen MP/MB parameter bytes unchanged")
    for key in FROZEN_REGIONS:
        moment = optimizer.state.get(cell["parameters"][key], {})
        equal(set(moment), set() if step == 0 else {"step", "exp_avg", "exp_avg_sq"}, "complete frozen-region Adam state")
        if step:
            if (moment["step"].dtype not in (torch.float32, torch.float64) or moment["step"].shape != torch.Size([])
                    or moment["step"].device.type != "cpu" or not bool(torch.isfinite(moment["step"])) or float(moment["step"]) != step):
                raise ValueError("frozen region counter still advances exactly")
            guard.byte_equal({name: moment[name] for name in ("exp_avg", "exp_avg_sq")},
                             {name: torch.zeros_like(initial[key]) for name in ("exp_avg", "exp_avg_sq")},
                             "frozen region moments remain exact initialized zeros")
    return {"step": step, "frozen_parameters_sha256": tensor_hash(initial), "parameters_byte_unchanged": True,
            "moments_empty": step == 0, "moments_exact_zero": step > 0, "all_counters": step}


def controls(binding):
    return {arm: continuation.prior_scores(binding["continuation_binding"], arm) for arm in parent.ARMS}


def comparisons(after, inherited):
    result = {}
    initial = inherited["mixed"]["original_initial"]
    for cohort, indices in (("train", SCORE_TRAIN_WINDOWS), ("development", DEV_WINDOWS)):
        current, mixed, gru = after[cohort], inherited["mixed"][cohort], inherited["prior_gru"][cohort]
        original = {"per_window": initial["train_scores"], "mean": initial["train_mean"]} if cohort == "train" else initial["development"]
        for value in (current, mixed, gru, original):
            equal([row["window"] for row in value["per_window"]], list(indices), "complete fixed cohort for actual comparisons")
            equal(parent._mean(value["per_window"]), value["mean"], "actual paired per-window mean")
        paired = []
        for a, b, c, d in zip(current["per_window"], mixed["per_window"], gru["per_window"], original["per_window"]):
            paired.append({"window": a["window"], "new_minus_original_mixed_ce": a["ce"] - b["ce"],
                           "new_minus_gru_ce": a["ce"] - c["ce"], "ce_improvement_from_initial": d["ce"] - a["ce"]})
        result[cohort] = {"new": current["mean"], "original_mixed": mixed["mean"], "original_gru": gru["mean"],
                          "original_initial": original["mean"],
                          "ce_improvement_from_initial": original["mean"]["ce"] - current["mean"]["ce"],
                          "accuracy_improvement_from_initial": current["mean"]["accuracy"] - original["mean"]["accuracy"],
                          "new_minus_original_mixed_ce": current["mean"]["ce"] - mixed["mean"]["ce"],
                          "new_minus_gru_ce": current["mean"]["ce"] - gru["mean"]["ce"],
                          "per_window": paired}
    return result


def execute(manifest, output, raw, *, write=False, recompute=False, clock=None, storage=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal control execution modes required")
    equal(UPDATES * BATCH_SIZE, len(TRAIN_WINDOWS), "one exact original-order training pass")
    output, binding = Path(output), manifest["binding"]
    clock = clock or Deadline()
    if write and storage is None:
        storage = Storage(output, clock)
        storage.used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    clock.begin_cell()
    directory = output / ARM
    if write:
        directory.mkdir(exist_ok=False)
        (directory / "train").mkdir(exist_ok=False)
    inherited = controls(binding)
    cell = initial_model(binding)
    equal(cell["initial_source"], inherited["mixed"]["original_initial"]["source_state"],
          "original shared initial scores refer to exact current initial state")
    optimizer = parent.make_optimizer("mixed", cell["parameters"].values())
    equal(len(optimizer.state), 0, "genuinely empty original Adam state")
    frozen_initial = {key: cell["parameters"][key].detach().clone() for key in FROZEN_REGIONS}
    reference = continuation.support_reference(cell)
    invariants = [frozen_state(cell, optimizer, frozen_initial, 0)]
    support_rows = [{"step": 0, "regions": None, "optimizer_moments_empty": True}]
    states = [parent.state_record(directory, 0, cell, optimizer, "mixed", write=write, storage=storage)]
    guard.byte_equal(load(directory / "state_000000.npz"), load(cell["initial_source"]["file"]["path"]),
                     "all actual initial parameters and empty moments match frozen original state")
    order, query_rows, applied_rows, observed = training_order(), [], [], []
    print(json.dumps({"stage": "original_start", "arm": ARM, "recompute": recompute, "frozen_regions": list(FROZEN_REGIONS)}), flush=True)
    for step in range(UPDATES):
        clock.check()
        indices = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        tokens, targets = batch_data(raw, binding["dataset"], indices)
        before = tensor_hash(cell["parameters"])
        raw_gradients, row = query_batch(directory, step, cell, tokens, targets, indices,
                                        write=write, recompute=recompute, storage=storage)
        # The independent oracle compares RAW G inside query_batch. Its saved
        # payload is masked and independently byte-bound in the separate file.
        applied, applied_row = applied_record(directory, step, raw_gradients, write=write, storage=storage)
        arrays = load(directory / f"query_{step:06d}.npz")
        selections = [continuation.support_selections({key: arrays[f"window_{index}/{key}"] for key in helper.TRACE_KEYS}, reference)
                      for index in indices]
        row["observed_support"] = continuation._sum_selections(selections)
        observed.extend(selections)
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in cell["parameters"].items():
            parameter.grad = applied[name].detach().clone()
        optimizer.step()
        invariants.append(frozen_state(cell, optimizer, frozen_initial, step + 1))
        states.append(parent.state_record(directory, step + 1, cell, optimizer, "mixed", write=write, storage=storage))
        support_rows.append({"step": step + 1, "regions": continuation.support_state(cell, optimizer, reference), "optimizer_moments_empty": False})
        row.update(before_parameters_sha256=before, after_parameters_sha256=tensor_hash(cell["parameters"]),
                   applied_gradient_sha256=applied_row["applied_gradient_sha256"], state_sha256=states[-1]["full_state_tensor_sha256"],
                   saved_applied_gradient_adam_replay=True)
        query_rows.append(row)
        applied_rows.append(applied_row)
        clock.check()
        if (step + 1) % 16 == 0:
            print(json.dumps({"stage": "constrained_update", "step": step + 1,
                              "query_ce": parent._mean(row["per_window"])["ce"], "recompute": recompute}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    before_scores = parent.state_record(directory, UPDATES, cell, optimizer, "mixed")
    train = endpoint.collect_scores(directory / "train", cell, raw, binding["dataset"], write=write,
                                    recompute=recompute, storage=storage, clock=clock)
    with torch.no_grad():
        dev = parent.collect_scores(directory, "final", cell, raw, binding["dataset"], True,
                                    write=write, recompute=recompute, storage=storage)
    equal(parent.state_record(directory, UPDATES, cell, optimizer, "mixed"), before_scores, "scores preserve complete final Adam state")
    frozen_state(cell, optimizer, frozen_initial, UPDATES)
    after = {"train": {"per_window": train, "mean": parent._mean(train)},
             "development": {"per_window": dev, "mean": parent._mean(dev)}}
    changes = {}
    for cohort, rows in (("train", train), ("development", dev)):
        expected = list(SCORE_TRAIN_WINDOWS if cohort == "train" else DEV_WINDOWS)
        equal([row["window"] for row in rows], expected, "complete matched recipe-final cohort")
        paths = inherited["mixed"]["paths"][cohort]
        equal(len(paths), len(expected), "all actual old mixed endpoint trace paths")
        paired = []
        for old_path, row in zip(paths, rows):
            new_path = directory / "train" / f"score_{row['window']:04d}.npz" if cohort == "train" else directory / f"score_final_{row['window']:03d}_1.npz"
            old_arrays, new_arrays = load(old_path), load(new_path)
            paired.append({"window": row["window"], "routes": continuation.route_changes(old_arrays, new_arrays)})
            observed.append(continuation.support_selections(new_arrays, reference))
        changes[cohort] = {"per_window": paired,
                           "totals": {key: {field: sum(item["routes"][key][field] for item in paired)
                                            for field in ("changed", "total")} for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}
    for key in ("file", "metadata"):
        descriptor = cell["initial_source"][key]
        equal(old.file_identity(descriptor["path"]), descriptor, "actual original initial checkpoint unchanged")
    fixed_baselines = {"train": read(Path(binding["continuation_binding"]["prior_run"]) / "summary.json")["baselines"],
                       "development": read(Path(binding["endpoint_binding"]["prior_run"]) / "summary.json")["baselines"]}
    result = {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
              "arm": ARM, "order_seed": 0, "initial_source": cell["initial_source"], "training_order": order,
              "inherited_state_format": {"schema": parent.SCHEMA, "arm": "mixed", "steps": list(range(UPDATES + 1))},
              "inherited_controls": inherited, "after": after, "comparisons": comparisons(after, inherited),
              "fixed_prior_baselines": fixed_baselines, "query_rows": query_rows, "applied_rows": applied_rows, "states": states,
              "frozen_region_invariants": invariants, "inactive_support_states": support_rows,
              "support_reference": {key: value["selected"].tolist() for key, value in reference.items()},
              "all_query_and_final_score_support": continuation._sum_selections(observed),
              "between_recipe_final_route_changes": changes,
              "route_scope": "same-cohort final old-mixed versus final frozen-core recipe; not within-trajectory stability; temporal changes are legitimate adaptation",
              "raw_gradient_queries": UPDATES, "applied_gradient_archives": UPDATES, "optimizer_commits": UPDATES,
              "complete_saved_states": UPDATES + 1, "score_window_readouts": len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS),
              "new_gradient_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
              "inherited_fitting_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE, "inherited_gradient_pair_exposures": 0,
              "endpoint_history_pair_exposures": 2 * len(TRAIN_WINDOWS) * SEQUENCE, "distinct_gradient_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
              "new_fitting_pair_exposures": 0, "new_baseline_evaluations": 0, "new_initial_score_evaluations": 0,
              "new_gru_training": False, "heldout_test_evaluated": False, "reserved_development_evaluated": False,
              "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
              "scope": "single-order frozen-core conditioning control; predecessor continuation is proof dependency only, not model history; no global redundancy or promotion claim"}
    clock.end_cell()
    return result


def expected_artifacts():
    names = {"manifest.json", "protocol.json", "declaration.md", "summary.json"}
    names.update(f"{ARM}/{prefix}_{step:06d}.npz" for step in range(UPDATES) for prefix in ("query", "applied"))
    names.update(f"{ARM}/state_{step:06d}.{suffix}" for step in range(UPDATES + 1) for suffix in ("npz", "json"))
    names.update(f"{ARM}/train/score_{index:04d}.npz" for index in SCORE_TRAIN_WINDOWS)
    names.update(f"{ARM}/score_final_{index:03d}_1.npz" for index in DEV_WINDOWS)
    return names


def _inventory(root):
    root = Path(root)
    artifacts = prior.artifact_manifest(root)
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact full core-bias payload inventory")
    equal({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()},
          {ARM, ARM + "/train"}, "only declared control directories")
    equal(read(root / "artifact_manifest.json"), {"schema": SCHEMA, "files": artifacts}, "complete sealed control payloads")
    return artifacts


def public_record(root, result, artifacts):
    return {**result, "run_directory": str(root), "runtime": read(Path(root) / "manifest.json")["runtime"],
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "independent_audit_required": True,
            "check_scope": "all raw/applied masks and saved-applied-G fullAdam states exact; separate independent raw-G/score audit required"}


_bound_config = continuation._bound_config


def _check(root, record, recompute, validation, raw, clock):
    if type(recompute) is not bool:
        raise ValueError("literal independent replay mode required")
    root = Path(root).resolve(strict=True)
    record = None if record is None else parent.rate.head._regular(record)
    originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
    public_bytes = None if record is None else record.read_bytes()
    manifest = read(root / "manifest.json")
    equal(manifest["run_directory"], str(root), "actual core-bias output path")
    equal(manifest["runtime"], parent.runtime(), "unchanged current software runtime")
    equal(read(root / "protocol.json"), recipe(), "saved fixed core-bias protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["declaration_sha256"], "actual saved declaration")
    frozen(manifest, validation)
    artifacts = _inventory(root)
    result = execute(manifest, root, raw, recompute=recompute, clock=clock)
    equal(read(root / "summary.json"), result, "complete recomputed core-bias summary")
    published = public_record(root, result, artifacts)
    if record is not None:
        equal(read(record), published, "actual complete core-bias publication")
    frozen(manifest, validation)
    equal(_inventory(root), artifacts, "all core-bias payload bytes unchanged during replay")
    for name, payload in originals.items():
        equal((root / name).read_bytes(), payload, "exact original core-bias metadata bytes")
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
            equal(binding, manifest["binding"], "fresh canonical core-bias audit binding")
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
        stage = "independent_numerical_replay" if independent else "fixed_core_bias_control"
        if independent:
            result = _check(root, record, True, validation, raw, clock)
            utc, monotonic = clock.check()
            windows, scores = UPDATES * BATCH_SIZE, len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)
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
                         "full_batch_gradients_verified": result["raw_gradient_queries"],
                         "complete_saved_states_verified": result["complete_saved_states"],
                         "optimizer_commits_replayed": result["optimizer_commits"], "score_window_readouts": result["score_window_readouts"],
                         "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
                         "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
                         "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                         "reserved_development_evaluated": False, "physical_fpga_execution": False,
                         "paid_execution_authorized": False, "recipe_promoted": False,
                         "scope": "independent reconstruction of the fixed core-bias control and final readouts, not new learning, seed confirmation or promotion"}
            storage.json(output / "verification.json", published)
            verification_bytes = (output / "verification.json").read_bytes()
            equal(read(output / "verification.json"), published, "actual independent verification result")
        else:
            result = execute(manifest, output, raw, write=True, storage=storage, clock=clock)
            storage.json(output / "summary.json", result)
            artifacts = prior.artifact_manifest(root)
            equal({row["file"] for row in artifacts}, expected_artifacts(), "complete core-bias payloads before sealing")
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
        equal(_inventory(root), artifacts, "all core-bias bytes unchanged during publication")
        for name, payload in originals.items():
            equal((root / name).read_bytes(), payload, "original core-bias metadata unchanged during publication")
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
        equal(binding, manifest["binding"], "fresh canonical core-bias checker binding")
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
        parser.error("--run requires prior continuation run/record/successful audit and original corpus")
    if args.audit and (args.audit_output is None or args.audit_record is None):
        parser.error("--audit requires exclusive audit output/record paths")
    result = run(args) if args.run else audit(args) if args.audit else check(args.output, args.record)
    print(json.dumps({"complete": result["complete"], "recipe_promoted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
