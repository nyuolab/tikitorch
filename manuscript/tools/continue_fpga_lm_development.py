"""Fixed order-zero second pass: exact full-Adam continuation, not promotion."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_endpoint_fit as endpoint  # noqa: E402

parent = endpoint.parent
guard, helper, old, prior, pilot = endpoint.guard, endpoint.helper, endpoint.old, endpoint.prior, endpoint.pilot
read, equal, load, tensor_hash = endpoint.read, endpoint.equal, endpoint.load, endpoint.tensor_hash
SCHEMA = "tikitorch.fpga_lm_continuation.v1"
ARMS = parent.ARMS
TRAIN_WINDOWS, SCORE_TRAIN_WINDOWS, DEV_WINDOWS = parent.TRAIN_WINDOWS, endpoint.TRAIN_WINDOWS, parent.DEV_WINDOWS
BATCH_SIZE, SEQUENCE, UPDATES, START_STEP = parent.BATCH_SIZE, parent.SEQUENCE, 256, 256
DEADLINE_S, CELL_DEADLINE_S = 10800, 3600
STORAGE_CAP, FREE_RESERVE = 8 * (1 << 30), 4 * (1 << 30)
PLAN = REPO / "manuscript/research/PLAN.md"
TRUST = {"endpoint_public": "6bb2ae3f379049378475196506197942c8fefb1f3aab34f3981d4c5d9bae3e3e",
         "endpoint_audit": "8a10febeb610962a6b6e86a8aacc5e94f934d7fc2730a9f23e7835d763291fbc",
         "development_public": "2ec2fdef77c5e99f14e0402bafb3f0f1219b4c4d9cb084be737a1c035db65149"}
query_batch, batch_data = parent.query_batch, parent.batch_data
Deadline = parent.Deadline


def source_identity():
    values = dict(endpoint.source_identity())
    for name in ("manuscript/tools/continue_fpga_lm_development.py", "tests/test_fpga_lm_continuation.py"):
        values[name] = old.digest((REPO / name).read_bytes())
    return values


def training_order():
    order = parent.training_order(0)
    equal(sorted(order), list(TRAIN_WINDOWS), "exact unchanged order-zero training membership")
    return order


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "order_seed": 0, "training_order": training_order(),
            "batch_size": BATCH_SIZE, "sequence": SEQUENCE, "additional_updates_per_arm": UPDATES,
            "start_step": START_STEP, "final_step": START_STEP + UPDATES,
            "sampled_training_windows": list(SCORE_TRAIN_WINDOWS), "development_windows": list(DEV_WINDOWS),
            "context": True, "dtype": "float64", "device": "cpu", "torch_threads": 1,
            "optimizer": "unchanged eager Adam; full parameters, moments, counters and group restored",
            "learning_rates": {"mixed": 0.01, "prior_gru": 0.001},
            "loss": "exact CE averaged over the complete batch; dimensionless u, no second beta",
            "gradient_queries": len(ARMS) * UPDATES, "optimizer_commits": len(ARMS) * UPDATES,
            "complete_saved_states": len(ARMS) * (UPDATES + 1),
            "score_window_readouts": len(ARMS) * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
            "new_gradient_pair_exposures": len(ARMS) * len(TRAIN_WINDOWS) * SEQUENCE,
            "distinct_gradient_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
            "inherited_fitting_pairs_per_arm": len(TRAIN_WINDOWS) * SEQUENCE,
            "inherited_gradient_pairs_per_arm": len(TRAIN_WINDOWS) * SEQUENCE,
            "endpoint_history_pairs_per_arm": 3 * len(TRAIN_WINDOWS) * SEQUENCE,
            "baseline_scope": "reuse fixed predecessor references; not exposure-matched to second-pass history",
            "initial_readouts": "reuse original and step256 per-window evidence; no new initial scores",
            "audit": "independent dense/autograd tropical and expanded GRU G; saved G drives byte-exact Adam replay; supplemental guarded traces",
            "support_scope": "descriptive observed selections and inactive sentinel/moment changes; no route freezing or stability gate",
            "deadline_seconds": DEADLINE_S, "cell_deadline_seconds": CELL_DEADLINE_S,
            "combined_new_output_cap_bytes": STORAGE_CAP, "free_reserve_bytes": FREE_RESERVE,
            "new_model_count_fit": False, "new_baseline_evaluations": 0, "adaptive_selection": False,
            "heldout_test_evaluated": False, "reserved_development_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "quality_gate": "none; one-order descriptive continuation, not confirmation or full-training convergence"}


class Storage(parent.Storage):
    def reserve(self, path, worst_case):
        self.clock.check()
        path = Path(path)
        if not any(root in path.resolve().parents for root in self.allowed_roots):
            raise ValueError("write outside owned continuation outputs")
        if path.exists() or path.is_symlink():
            raise FileExistsError("append-only continuation output required")
        if type(worst_case) is not int or worst_case < 0 or self.used + worst_case + (1 << 20) > STORAGE_CAP:
            raise RuntimeError("combined continuation storage cap would be exceeded")
        if shutil.disk_usage(path.parent).free - worst_case < FREE_RESERVE:
            raise RuntimeError("free disk reserve would be violated")

    def reconcile(self):
        size = super().reconcile()
        if size > STORAGE_CAP:
            raise RuntimeError("actual combined continuation storage cap exceeded")
        return size


def _audit_capsule(path, root, record):
    path = parent.rate.head._regular(path)
    equal(old.digest(path.read_bytes()), TRUST["endpoint_audit"], "actual successful endpoint audit pin")
    equal(old.digest(record.read_bytes()), TRUST["endpoint_public"], "actual endpoint publication pin")
    value = read(path)
    wanted = {"schema": endpoint.SCHEMA + ".audit", "complete": True,
              "independent_forward_recompute_completed": True, "run_directory": str(root), "record": str(record),
              "source_hashes": endpoint.source_identity(), "producer_public_sha256": TRUST["endpoint_public"],
              "producer_seal_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
              "deadline_seconds": endpoint.DEADLINE_S, "independent_dense_tropical_forwards": 256,
              "independent_expanded_gru_forwards": 256, "supplemental_guarded_forwards": 256,
              "model_window_readouts": 512, "baseline_window_readouts": 128,
              "source_input_raw_seal_public_bytes_unchanged": True, "new_gradients": 0, "optimizer_steps": 0,
              "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_model_count_fit": False,
              "new_development_evaluations": 0, "heldout_test_evaluated": False,
              "reserved_development_evaluated": False, "physical_fpga_execution": False,
              "paid_execution_authorized": False, "recipe_promoted": False}
    for key, expected in wanted.items():
        equal(value.get(key), expected, "successful endpoint audit: " + key)
    for key in ("verification_elapsed_utc_seconds", "verification_elapsed_monotonic_seconds"):
        elapsed = value[key]
        if type(elapsed) is not float or not math.isfinite(elapsed) or not 0 <= elapsed < endpoint.DEADLINE_S:
            raise ValueError("successful endpoint audit must finish within its declared bound")
    directory = Path(value["audit_directory"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("actual completed endpoint audit capsule required")
    directory = directory.resolve(strict=True)
    names = {"audit_manifest.json", "declaration.md", "verification.json"}
    equal({p.name for p in directory.iterdir()}, names, "exact successful endpoint audit inventory")
    files = {name: old.file_identity(parent.rate.head._regular(directory / name)) for name in names}
    equal((directory / "verification.json").read_bytes(), path.read_bytes(), "exact pinned private/public audit bytes")
    equal(files["audit_manifest.json"]["sha256"], value["audit_manifest_sha256"], "actual endpoint audit manifest")
    manifest = read(directory / "audit_manifest.json")
    for key, expected in {"schema": endpoint.SCHEMA, "complete": False, "mode": "independent_forward_audit",
                          "run_directory": str(root), "record": str(record), "audit_directory": str(directory),
                          "audit_record": str(path), "source_hashes": endpoint.source_identity(),
                          "producer_public_sha256": TRUST["endpoint_public"], "producer_seal_sha256": value["producer_seal_sha256"],
                          "runtime": value["runtime"], "started_at_utc": value["started_at_utc"],
                          "deadline_seconds": endpoint.DEADLINE_S, "declaration_sha256": files["declaration.md"]["sha256"]}.items():
        equal(manifest[key], expected, "successful endpoint audit manifest: " + key)
    return value, {"receipt": old.file_identity(path), "files": files}


@dataclass(frozen=True)
class _Validation:
    binding: dict
    sources: dict
    endpoint_manifest: dict
    endpoint_validation: object

    def __post_init__(self):
        for key in ("binding", "sources", "endpoint_manifest"):
            object.__setattr__(self, key, copy.deepcopy(getattr(self, key)))


def bind_inputs(config):
    sources = source_identity()
    root, record = Path(config.prior_run).resolve(strict=True), parent.rate.head._regular(config.prior_record)
    _, capsule = _audit_capsule(config.prior_audit, root, record)
    previous = read(root / "manifest.json")
    equal(previous["runtime"], parent.runtime(), "unchanged endpoint software runtime")
    binding, raw, token = endpoint.bind_inputs(endpoint._bound_config(previous))
    equal(binding, previous["binding"], "canonical fully byte-verified endpoint ancestry")
    endpoint.frozen(previous, token)
    equal(read(Path(binding["prior_run"]) / "manifest.json")["runtime"], parent.runtime(), "unchanged development software runtime")
    equal(old.digest(Path(binding["prior_record"]).read_bytes()), TRUST["development_public"], "actual development publication pin")
    artifacts = endpoint._inventory(root)
    summary = read(root / "summary.json")
    equal(read(record), endpoint.public_record(root, summary, artifacts), "actual complete audited endpoint publication")
    equal(summary["complete"], True, "all prior endpoints complete")
    equal(Path(config.corpus).resolve(strict=True), Path(binding["dataset"]["path"]), "unchanged corpus path")
    result = {"prior_run": str(root), "prior_record": str(record), "prior_audit": str(parent.rate.head._regular(config.prior_audit)),
              "audit_capsule": capsule, "trust": dict(TRUST), "endpoint_binding": binding, "dataset": binding["dataset"],
              "parent_files": {name: old.file_identity(root / name) for name in ("manifest.json", "artifact_manifest.json", "summary.json")},
              "parent_public": old.file_identity(record), "parent_artifacts": artifacts}
    validation = _Validation(result, sources, previous, token)
    frozen({"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "binding": result}, validation)
    return result, raw, validation


def frozen(manifest, validation):
    if not isinstance(validation, _Validation):
        raise ValueError("private invocation-local continuation validation required")
    equal(manifest["schema"], SCHEMA, "continuation schema")
    equal(manifest["recipe"], recipe(), "fixed second-pass recipe")
    equal(manifest["source_hashes"], source_identity(), "all73 source identities")
    equal(manifest["source_hashes"], validation.sources, "original invocation sources")
    equal(manifest["binding"], validation.binding, "canonical full audited ancestry")
    binding = manifest["binding"]
    equal(binding["trust"], TRUST, "fixed actual prior admission pins")
    endpoint.frozen(validation.endpoint_manifest, validation.endpoint_validation)
    equal(endpoint._inventory(Path(binding["prior_run"])), binding["parent_artifacts"], "all endpoint artifacts unchanged")
    _, capsule = _audit_capsule(binding["prior_audit"], Path(binding["prior_run"]), Path(binding["prior_record"]))
    equal(capsule, binding["audit_capsule"], "complete successful audit capsule unchanged")
    for descriptor in (*binding["parent_files"].values(), binding["parent_public"]):
        equal(old.file_identity(descriptor["path"]), descriptor, "actual ancestor raw/seal/public bytes unchanged")


def state_arrays(cell, optimizer, arm, step):
    if arm not in ARMS or type(step) is not int or not START_STEP <= step <= START_STEP + UPDATES:
        raise ValueError("literal absolute continuation state step required")
    parent.parent._finite_parameters(cell)
    group = (parent.rate._optimizer_group if arm == "mixed" else parent.parent._optimizer_group)(cell, optimizer)
    if set(optimizer.state) != set(cell["parameters"].values()):
        raise ValueError("all and only resumed parameter optimizer states required")
    arrays, shapes = {}, {}
    for name, parameter in cell["parameters"].items():
        arrays["parameter/" + name] = parameter.detach()
        moment = optimizer.state[parameter]
        equal(set(moment), {"step", "exp_avg", "exp_avg_sq"}, "complete resumed Adam state")
        counter = moment["step"]
        if (counter.device.type != "cpu" or counter.dtype not in (torch.float32, torch.float64)
                or counter.shape != torch.Size([]) or not bool(torch.isfinite(counter)) or float(counter) != step):
            raise ValueError("exact absolute resumed Adam counter required")
        shapes[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype), "initialized": True,
                        "step_dtype": str(counter.dtype)}
        for key in ("exp_avg", "exp_avg_sq"):
            value = moment[key]
            helper._finite(value, "resumed Adam moment", detached=True)
            if value.shape != parameter.shape or (key == "exp_avg_sq" and bool((value < 0).any())):
                raise ValueError("finite full-shaped Adam moments required")
            arrays[key + "/" + name] = value
        arrays["step/" + name] = counter
    return arrays, shapes, group


def restore_optimizer(cell, path, arm):
    parameters, metadata = endpoint._checkpoint(path, cell, arm, START_STEP)
    saved = load(path)
    equal(tensor_hash(saved), metadata["full_state_tensor_sha256"], "second safe load binds validated full state")
    guard.byte_equal({name: saved["parameter/" + name] for name in parameters}, parameters, "second safe load binds validated parameters")
    optimizer = parent.make_optimizer(arm, cell["parameters"].values())
    with torch.no_grad():
        for name, parameter in cell["parameters"].items():
            parameter.copy_(parameters[name])
            parameter.grad = None
            optimizer.state[parameter] = {key: saved[key + "/" + name].detach().clone()
                                          for key in ("step", "exp_avg", "exp_avg_sq")}
    actual, _, group = state_arrays(cell, optimizer, arm, START_STEP)
    equal(group, metadata["optimizer_group"], "exact restored original Adam group")
    guard.byte_equal(actual, saved, "every restored parameter/moment/counter byte before new G")
    return optimizer


def resume_model(binding, arm):
    cell = endpoint.model(binding["endpoint_binding"], arm, 0)
    source = copy.deepcopy(cell["endpoint_source"])
    optimizer = restore_optimizer(cell, source["file"]["path"], arm)
    return cell, optimizer, source


def state_record(directory, step, cell, optimizer, arm, *, write=False, storage=None):
    if type(write) is not bool:
        raise ValueError("literal state write flag required")
    arrays, shapes, group = state_arrays(cell, optimizer, arm, step)
    path = Path(directory) / f"state_{step:06d}.npz"
    if write:
        storage.arrays(path, arrays)
    guard.byte_equal(load(path), arrays, "every actual continued state tensor byte")
    payload = path.read_bytes()
    result = {"schema": SCHEMA, "arm": arm, "step": step, "additional_updates": step - START_STEP,
              "file": path.name, "bytes": len(payload), "sha256": old.digest(payload),
              "parameter_order": list(cell["parameters"]), "parameter_state": shapes, "optimizer_group": group,
              "full_state_tensor_sha256": tensor_hash(arrays), "parameters_sha256": tensor_hash(cell["parameters"]),
              "initial_optimizer_state_empty": False, "resumed_full_state": True}
    if write:
        storage.json(path.with_suffix(".json"), result)
    equal(read(path.with_suffix(".json")), result, "complete typed absolute-step state metadata")
    return result


def support_reference(cell):
    """Static resumed support; ambiguous rows are reported, never forbidden."""
    if cell["kind"] != "tropical":
        return None
    result = {}
    for key in ("MP", "W_HEAD"):
        values = cell["parameters"][key].detach()
        inactive = values == guard.guarded.OFF_WEIGHT
        count = (~inactive).sum(-1)
        selected = (~inactive).to(torch.int64).argmax(-1)
        selected = torch.where(count == 1, selected, torch.full_like(selected, -1))
        result[key] = {"inactive": inactive.clone(), "selected": selected.clone()}
    return result


def support_state(cell, optimizer, reference):
    if reference is None:
        return None
    result = {}
    for key, item in reference.items():
        values = cell["parameters"][key].detach()
        inactive, selected = item["inactive"], item["selected"]
        moments = optimizer.state[cell["parameters"][key]]
        count = (values != guard.guarded.OFF_WEIGHT).sum(-1)
        current = (values != guard.guarded.OFF_WEIGHT).to(torch.int64).argmax(-1)
        known = selected >= 0
        result[key] = {"resumed_inactive_entries": int(inactive.sum()),
                       "inactive_parameters_different_from_sentinel": int(((values != guard.guarded.OFF_WEIGHT) & inactive).sum()),
                       "inactive_first_moments_nonzero": int(((moments["exp_avg"] != 0) & inactive).sum()),
                       "inactive_second_moments_nonzero": int(((moments["exp_avg_sq"] != 0) & inactive).sum()),
                       "resumed_single_active_rows": int(known.sum()), "resumed_ambiguous_rows": int((~known).sum()),
                       "current_single_active_rows": int((count == 1).sum()),
                       "changed_single_active_support_rows": int((known & ((count != 1) | (current != selected))).sum())}
    return result


def support_selections(trace, reference):
    if reference is None:
        return None
    result = {}
    for route, key in (("AD_HEAD", "W_HEAD"), ("PROJ_IDX", "MP")):
        actual = trace[route]
        expected = reference[key]["selected"]
        expected = expected[None, :] if route == "AD_HEAD" else expected[:, None, :]
        expected = expected.expand_as(actual)
        known = expected >= 0
        if actual.dtype != torch.int64:
            raise ValueError("integer observed route tape required")
        result[route] = {"departures": int(((actual != expected) & known).sum()), "compared_entries": int(known.sum()),
                         "unclassified_entries": int((~known).sum()), "total_entries": actual.numel()}
    return result


def _sum_selections(rows):
    return {route: {key: sum(row[route][key] for row in rows) for key in
                   ("departures", "compared_entries", "unclassified_entries", "total_entries")}
            for route in ("AD_HEAD", "PROJ_IDX")}


def route_changes(before, after):
    result = {}
    for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX"):
        a, b = before[key], after[key]
        if a.dtype != torch.int64 or b.dtype != torch.int64 or a.shape != b.shape:
            raise ValueError("same-cohort complete integer route arrays required")
        result[key] = {"changed": int((a != b).sum()), "total": a.numel()}
    return result


def prior_scores(binding, arm):
    root = Path(binding["prior_run"])
    summary = read(root / "summary.json")
    matches = [row for row in summary["cells"] if row["arm"] == arm and row["order_seed"] == 0]
    equal(len(matches), 1, "one actual bound order-zero endpoint")
    saved = matches[0]
    equal(saved["complete"], True, "complete prior endpoint")
    train = saved["train_scores"]
    equal([row["window"] for row in train], list(SCORE_TRAIN_WINDOWS), "exact prior sampled training cohort")
    equal(parent._mean(train), saved["train_mean"], "actual prior sampled training mean")
    paths = {"train": [], "development": []}
    for row in train:
        path = root / parent.cell_name(arm, 0) / f"score_{row['window']:04d}.npz"
        equal(old.digest(path.read_bytes()), row["file_sha256"], "actual sampled prior trace")
        paths["train"].append(str(path))
    dev = endpoint._development_rows(binding["endpoint_binding"], arm, 0)
    equal(dev, saved["development"], "same actually bound development rows")
    for row in dev["per_window"]:
        paths["development"].append(str(Path(binding["endpoint_binding"]["prior_run"]) / parent.cell_name(arm, 0)
                                        / f"score_final_{row['window']:03d}_1.npz"))
    return {"train": {"per_window": train, "mean": saved["train_mean"]}, "development": dev,
            "original_initial": summary["initial"][arm], "source_state": saved["source_state"], "paths": paths}


def comparisons(cells):
    equal([row["arm"] for row in cells], list(ARMS), "exact two-arm continuation order")
    result = {}
    for cohort in ("train", "development"):
        left, right = cells
        expected = list(SCORE_TRAIN_WINDOWS if cohort == "train" else DEV_WINDOWS)
        for cell in cells:
            if cell.get("complete") is not True or cell.get("numerics_pass") is not True:
                raise ValueError("both actual complete healthy cells required for comparisons")
            for stage in ("before", "after"):
                values = cell[stage][cohort]
                equal([row["window"] for row in values["per_window"]], expected, "complete paired cohort membership")
                equal(parent._mean(values["per_window"]), values["mean"], "paired cohort actual row mean")
        before = left["before"][cohort]["mean"]["ce"] - right["before"][cohort]["mean"]["ce"]
        after = left["after"][cohort]["mean"]["ce"] - right["after"][cohort]["mean"]["ce"]
        pairs = []
        for a0, b0, a1, b1 in zip(left["before"][cohort]["per_window"], right["before"][cohort]["per_window"],
                                  left["after"][cohort]["per_window"], right["after"][cohort]["per_window"]):
            equal([a0["window"], b0["window"], a1["window"], b1["window"]], [a0["window"]] * 4, "actual paired windows")
            delta0, delta1 = a0["ce"] - b0["ce"], a1["ce"] - b1["ce"]
            pairs.append({"window": a0["window"], "before_tropical_minus_gru_ce": delta0,
                          "after_tropical_minus_gru_ce": delta1, "paired_deficit_change": delta1 - delta0})
        result[cohort] = {"before_tropical_minus_gru_ce": before, "after_tropical_minus_gru_ce": after,
                          "paired_deficit_change": after - before, "deficit_narrowed": after < before,
                          "tropical_ce_improved": left["after"][cohort]["mean"]["ce"] < left["before"][cohort]["mean"]["ce"],
                          "gru_ce_improved": right["after"][cohort]["mean"]["ce"] < right["before"][cohort]["mean"]["ce"],
                          "per_window": pairs}
    return result


def execute(manifest, output, raw, *, write=False, recompute=False, clock=None, storage=None):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("disjoint literal continuation modes required")
    equal(UPDATES * BATCH_SIZE, len(TRAIN_WINDOWS), "one exact additional pass")
    output, binding = Path(output), manifest["binding"]
    clock = clock or Deadline()
    if write and storage is None:
        storage = Storage(output, clock)
        storage.used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    cells = []
    for arm in ARMS:
        clock.begin_cell()
        print(json.dumps({"stage": "resume", "arm": arm, "absolute_step": START_STEP, "recompute": recompute}), flush=True)
        directory = output / arm
        if write:
            directory.mkdir(exist_ok=False)
            (directory / "train").mkdir(exist_ok=False)
        cell, optimizer, source = resume_model(binding, arm)
        before = prior_scores(binding, arm)
        equal(source, before["source_state"], "actual scored predecessor is the complete resumed state")
        reference = support_reference(cell)
        support_rows = [{"step": START_STEP, "regions": support_state(cell, optimizer, reference)}] if reference is not None else []
        observed = []
        states = [state_record(directory, START_STEP, cell, optimizer, arm, write=write, storage=storage)]
        guard.byte_equal(load(directory / f"state_{START_STEP:06d}.npz"), load(source["file"]["path"]), "actual complete imported state archive")
        prior_before = None if cell["prior_logits"] is None else cell["prior_logits"].detach().clone()
        order, rows = training_order(), []
        for local_step in range(UPDATES):
            clock.check()
            indices = order[local_step * BATCH_SIZE:(local_step + 1) * BATCH_SIZE]
            tokens, targets = batch_data(raw, binding["dataset"], indices)
            parameters_before = tensor_hash(cell["parameters"])
            gradients, row = query_batch(directory, local_step, cell, tokens, targets, indices,
                                         write=write, recompute=recompute, storage=storage)
            if reference is not None:
                arrays = load(directory / f"query_{local_step:06d}.npz")
                selections = []
                for index in indices:
                    trace = {key: arrays[f"window_{index}/{key}"] for key in helper.TRACE_KEYS}
                    selections.append(support_selections(trace, reference))
                row["observed_support"] = _sum_selections(selections)
                observed.extend(selections)
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in cell["parameters"].items():
                parameter.grad = gradients[name].detach().clone()
            optimizer.step()
            absolute = START_STEP + local_step + 1
            states.append(state_record(directory, absolute, cell, optimizer, arm, write=write, storage=storage))
            if reference is not None:
                support_rows.append({"step": absolute, "regions": support_state(cell, optimizer, reference)})
            if prior_before is not None:
                guard.byte_equal({"prior": cell["prior_logits"]}, {"prior": prior_before}, "frozen inherited GRU prior through continuation")
            row.update(absolute_step_before=absolute - 1, absolute_step_after=absolute,
                       before_parameters_sha256=parameters_before, after_parameters_sha256=tensor_hash(cell["parameters"]),
                       state_sha256=states[-1]["full_state_tensor_sha256"], saved_gradient_adam_replay=True)
            rows.append(row)
            clock.check()
            if (local_step + 1) % 16 == 0:
                print(json.dumps({"arm": arm, "additional_updates": local_step + 1, "absolute_step": absolute,
                                  "query_ce": parent._mean(row["per_window"])["ce"], "recompute": recompute}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        state_before_scores = tensor_hash(state_arrays(cell, optimizer, arm, START_STEP + UPDATES)[0])
        train = endpoint.collect_scores(directory / "train", cell, raw, binding["dataset"], write=write,
                                        recompute=recompute, storage=storage, clock=clock)
        with torch.no_grad():
            dev = parent.collect_scores(directory, "final", cell, raw, binding["dataset"], True,
                                        write=write, recompute=recompute, storage=storage)
        equal(tensor_hash(state_arrays(cell, optimizer, arm, START_STEP + UPDATES)[0]), state_before_scores, "scores preserve complete final Adam state")
        after = {"train": {"per_window": train, "mean": parent._mean(train)},
                 "development": {"per_window": dev, "mean": parent._mean(dev)}}
        changes = {}
        for cohort, final_rows in (("train", train), ("development", dev)):
            if reference is not None:
                expected = list(SCORE_TRAIN_WINDOWS if cohort == "train" else DEV_WINDOWS)
                equal([row["window"] for row in final_rows], expected, "complete same-cohort final route membership")
                equal(len(before["paths"][cohort]), len(expected), "all actual prior route paths present")
                per_window = []
                for old_path, row in zip(before["paths"][cohort], final_rows):
                    new_path = directory / "train" / f"score_{row['window']:04d}.npz" if cohort == "train" else directory / f"score_final_{row['window']:03d}_1.npz"
                    old_arrays, new_arrays = load(old_path), load(new_path)
                    per_window.append({"window": row["window"], "routes": route_changes(old_arrays, new_arrays)})
                    observed.append(support_selections(new_arrays, reference))
                changes[cohort] = {"per_window": per_window,
                                   "totals": {key: {field: sum(item["routes"][key][field] for item in per_window)
                                                    for field in ("changed", "total")} for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}
        improvements = {cohort: {"ce": before[cohort]["mean"]["ce"] - after[cohort]["mean"]["ce"],
                                 "accuracy": after[cohort]["mean"]["accuracy"] - before[cohort]["mean"]["accuracy"]}
                        for cohort in ("train", "development")}
        separation0 = before["development"]["mean"]["ce"] - before["train"]["mean"]["ce"]
        separation1 = after["development"]["mean"]["ce"] - after["train"]["mean"]["ce"]
        result = {"schema": SCHEMA, "arm": arm, "order_seed": 0, "complete": True, "numerics_pass": True,
                  "source_state": source, "training_order": order, "start_step": START_STEP, "final_step": START_STEP + UPDATES,
                  "before": before, "after": after, "improvements": improvements,
                  "development_minus_sampled_train_before": separation0, "development_minus_sampled_train_after": separation1,
                  "development_minus_sampled_train_change": separation1 - separation0,
                  "query_rows": rows, "states": states, "optimizer_commits": UPDATES,
                  "full_batch_gradient_queries": UPDATES, "complete_saved_states": UPDATES + 1,
                  "new_gradient_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
                  "inherited_fitting_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
                  "inherited_gradient_pair_exposures": len(TRAIN_WINDOWS) * SEQUENCE,
                  "endpoint_history_pair_exposures": 3 * len(TRAIN_WINDOWS) * SEQUENCE,
                  "distinct_training_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
                  "support_reference": None if reference is None else {key: value["selected"].tolist() for key, value in reference.items()},
                  "inactive_support_states": support_rows, "same_cohort_route_changes": changes,
                  "all_query_and_final_score_support": None if reference is None else _sum_selections(observed),
                  "support_scope": "observed resumed-support departures and same-window endpoint changes only; no fixed-route or quality gate"}
        for key in ("file", "metadata"):
            equal(old.file_identity(source[key]["path"]), source[key], "actual predecessor checkpoint unchanged")
        if write:
            storage.json(directory / "summary.json", result)
        equal(read(directory / "summary.json"), result, "complete actual continued cell")
        cells.append(result)
        clock.end_cell()
    fixed_baselines = {"train": read(Path(binding["prior_run"]) / "summary.json")["baselines"],
                       "development": read(Path(binding["endpoint_binding"]["prior_run"]) / "summary.json")["baselines"]}
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "cells": cells, "paired_comparisons": comparisons(cells), "fixed_prior_baselines": fixed_baselines,
            "completed_cells": len(ARMS), "full_batch_gradient_queries": len(ARMS) * UPDATES,
            "optimizer_commits": len(ARMS) * UPDATES, "complete_saved_states": len(ARMS) * (UPDATES + 1),
            "score_window_readouts": len(ARMS) * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
            "new_gradient_pair_exposures": len(ARMS) * len(TRAIN_WINDOWS) * SEQUENCE,
            "distinct_gradient_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
            "summed_endpoint_history_pair_exposures": 3 * len(ARMS) * len(TRAIN_WINDOWS) * SEQUENCE,
            "new_fitting_pair_exposures": 0, "new_baseline_evaluations": 0,
            "heldout_test_evaluated": False, "reserved_development_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "fixed one-order second-pass achieved fitting and reused-development diagnostic; not full training convergence, seed confirmation or significance"}


def expected_artifacts():
    names = {"manifest.json", "protocol.json", "declaration.md", "summary.json"}
    for arm in ARMS:
        names.add(arm + "/summary.json")
        names.update(f"{arm}/query_{step:06d}.npz" for step in range(UPDATES))
        names.update(f"{arm}/state_{step:06d}.{suffix}" for step in range(START_STEP, START_STEP + UPDATES + 1)
                     for suffix in ("npz", "json"))
        names.update(f"{arm}/train/score_{index:04d}.npz" for index in SCORE_TRAIN_WINDOWS)
        names.update(f"{arm}/score_final_{index:03d}_1.npz" for index in DEV_WINDOWS)
    return names


def _inventory(root):
    root = Path(root)
    artifacts = prior.artifact_manifest(root)
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact complete continuation payload inventory")
    equal({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()},
          set(ARMS) | {arm + "/train" for arm in ARMS}, "only declared continuation directories")
    equal(read(root / "artifact_manifest.json"), {"schema": SCHEMA, "files": artifacts}, "exact sealed continuation payloads")
    return artifacts


def public_record(root, result, artifacts):
    return {**result, "run_directory": str(root), "runtime": read(Path(root) / "manifest.json")["runtime"],
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "independent_audit_required": True,
            "check_scope": "all saved-G full-Adam states replayed exactly; separate audit independently reconstructs new gradients and scores"}


def _bound_config(manifest):
    from types import SimpleNamespace
    binding = manifest["binding"]
    return SimpleNamespace(prior_run=Path(binding["prior_run"]), prior_record=Path(binding["prior_record"]),
                           prior_audit=Path(binding["prior_audit"]), corpus=Path(binding["dataset"]["path"]))


def _check(root, record, recompute, validation, raw, clock):
    if type(recompute) is not bool:
        raise ValueError("literal independent replay mode required")
    root = Path(root).resolve(strict=True)
    record = None if record is None else parent.rate.head._regular(record)
    originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json")}
    public_bytes = None if record is None else record.read_bytes()
    manifest = read(root / "manifest.json")
    equal(manifest["run_directory"], str(root), "actual continuation output path")
    equal(manifest["runtime"], parent.runtime(), "unchanged current software runtime")
    equal(read(root / "protocol.json"), recipe(), "saved fixed continuation protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["declaration_sha256"], "actual saved declaration")
    frozen(manifest, validation)
    artifacts = _inventory(root)
    result = execute(manifest, root, raw, recompute=recompute, clock=clock)
    equal(read(root / "summary.json"), result, "complete recomputed continuation summary")
    published = public_record(root, result, artifacts)
    if record is not None:
        equal(read(record), published, "actual complete continuation publication")
    frozen(manifest, validation)
    equal(_inventory(root), artifacts, "all continuation payload bytes unchanged during replay")
    for name, payload in originals.items():
        equal((root / name).read_bytes(), payload, "exact original continuation metadata bytes")
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
            equal(binding, manifest["binding"], "fresh canonical continuation audit binding")
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
        stage = "independent_numerical_replay" if independent else "fixed_second_pass"
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
                         "independent_tropical_batch_gradients": UPDATES, "independent_gru_batch_gradients": UPDATES,
                         "independent_gru_single_window_scores": scores, "supplemental_guarded_window_forwards": windows + scores,
                         "full_batch_gradients_verified": result["full_batch_gradient_queries"],
                         "complete_saved_states_verified": result["complete_saved_states"],
                         "optimizer_commits_replayed": result["optimizer_commits"], "score_window_readouts": result["score_window_readouts"],
                         "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
                         "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
                         "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
                         "reserved_development_evaluated": False, "physical_fpga_execution": False,
                         "paid_execution_authorized": False, "recipe_promoted": False,
                         "scope": "independent reconstruction of the fixed second pass and final readouts, not new learning, seed confirmation or promotion"}
            storage.json(output / "verification.json", published)
            verification_bytes = (output / "verification.json").read_bytes()
            equal(read(output / "verification.json"), published, "actual independent verification result")
        else:
            result = execute(manifest, output, raw, write=True, storage=storage, clock=clock)
            storage.json(output / "summary.json", result)
            artifacts = prior.artifact_manifest(root)
            equal({row["file"] for row in artifacts}, expected_artifacts(), "complete continuation payloads before sealing")
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
        equal(_inventory(root), artifacts, "all continuation bytes unchanged during publication")
        for name, payload in originals.items():
            equal((root / name).read_bytes(), payload, "original continuation metadata unchanged during publication")
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
        equal(binding, manifest["binding"], "fresh canonical continuation checker binding")
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
        parser.error("--run requires prior endpoint run/record/successful audit and original corpus")
    if args.audit and (args.audit_output is None or args.audit_record is None):
        parser.error("--audit requires exclusive audit output/record paths")
    result = run(args) if args.run else audit(args) if args.audit else check(args.output, args.record)
    print(json.dumps({"complete": result["complete"], "recipe_promoted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
