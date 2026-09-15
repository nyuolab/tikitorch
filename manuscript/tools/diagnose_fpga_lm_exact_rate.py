"""One preregistered FP64 exact-CE rate control; no native or held-out claim.

The completed mixed-topology initializer and all numeric rules are inherited
unchanged. Only eager Adam's learning rate changes from .001 to .01. Ordinary
checking replays saved gradients; independent checking rederives all queries.
"""
from __future__ import annotations

import argparse
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

from manuscript.tools import diagnose_fpga_lm_exact_fit as parent  # noqa: E402

helper, head = parent.helper, parent.head
old, prior, pilot = parent.old, parent.prior, parent.pilot
read, equal, save, load, same = parent.read, parent.equal, parent.save, parent.load, parent.same
query_batch, collect_scores = parent.query_batch, parent.collect_scores
batch_data, tensor_hash, statistics = parent.batch_data, parent.tensor_hash, parent.statistics
SCHEMA = "tikitorch.fpga_lm_exact_rate.v1"
ARM, LR, UPDATES, WINDOWS = "mixed_rate_01", 0.01, 128, (0, 1, 2, 3)
DEADLINE_S, GATE_TOLERANCE = 3600, 1e-12
PLAN = REPO / "manuscript/research/PLAN.md"


def source_identity():
    result = dict(parent.source_identity())
    for name in ("manuscript/tools/diagnose_fpga_lm_exact_rate.py", "tests/test_fpga_lm_exact_rate.py"):
        result[name] = old.digest((REPO / name).read_bytes())
    return result


def recipe():
    return {"schema": SCHEMA, "arm": ARM, "training_windows": list(WINDOWS),
            "updates": UPDATES, "seed": 0, "dtype": "float64", "device": "cpu",
            "batch": "same four full 128-token windows; reset state at each boundary",
            "sole_recipe_change": "eager Adam lr .001 -> .01; same original mixed initializer, not endpoint continuation",
            "units": "u=beta*W, dimensionless logits, exact full-batch mean CE, real centering",
            "topology": "same source-bound mixed conjugated-rotation initializer",
            "optimizer": {"name": "eager Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-8,
                          "weight_decay": 0.0, "foreach": False, "fused": False, "gradient_clipping": False},
            "full_batch_gradient_queries": 128, "optimizer_commits": 128, "complete_saved_states": 129,
            "score_window_readouts": 12, "independent_tropical_window_vjps": 512,
            "independent_full_batch_gradients": 128,
            "gradient_audit": "reconstruct 512 window VJPs and compare 128 saved sums; exact saved-gradient Adam replay",
            "readouts": "four initial context-on and four final context-on/off; same training cohort only",
            "initial_state_and_scores": "exact actual prior mixed step-zero parameter arrays and initial traces",
            "independent_tolerance": {"atol": 1e-9, "rtol": 1e-11, "integer_routes": "exact"},
            "gate": {"fraction_of_prior_mixed_to_gru_ce_gap_closed": 0.5, "near_prior_gru_ce_margin": 0.05,
                     "inclusive_tolerance": GATE_TOLERANCE, "comparators": "completed unchanged prior mixed and GRU endpoints"},
            "new_gradient_pair_exposures": 65536, "distinct_gradient_pairs": 512,
            "new_fitting_pair_exposures": 0, "inherited_fitting_pair_exposures": 131072,
            "endpoint_history_pair_exposures": 196608, "distinct_fitted_pairs": 131072,
            "historical_comparator_updates_reexecuted": 0, "adaptive_extension": False,
            "learning_rate_sweep": False, "torch_threads": 1, "deadline_s": DEADLINE_S,
            "new_native_training_calls": 0, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False}


def _audit_receipt(root, record, audit):
    value, artifacts = read(audit), prior.artifact_manifest(root)
    expected = {"schema": "tikitorch.fpga_lm_exact_fit.cli_audit_receipt.v1", "complete": True,
                "independent_recompute": True, "exit_code": 0, "deadline_seconds": 3600,
                "parent_record_sha256": old.digest(Path(record).read_bytes()),
                "run_manifest_sha256": old.digest((root / "manifest.json").read_bytes()),
                "artifact_manifest_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
                "runner_sha256": parent.source_identity()["manuscript/tools/diagnose_fpga_lm_exact_fit.py"],
                "source_files_verified": len(parent.source_identity()), "artifact_files_verified": len(artifacts),
                "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
                "independent_full_batch_gradient_queries": 384, "independent_tropical_window_vjps": 1024,
                "independent_tropical_full_batch_gradients": 256, "independent_gru_full_batch_gradients": 128,
                "independent_score_readouts": 36, "complete_saved_states_verified": 387, "optimizer_commits_replayed": 384,
                "complete_parent_ancestry_verified": True, "source_input_raw_seal_public_bytes_unchanged": True,
                "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
                "new_optimizer_trajectory_executed": False, "heldout_corpus_evaluated": False,
                "physical_fpga_execution": False, "paid_execution_authorized": False,
                "recipe_promoted": False, "broader_study_authorized": False}
    for key, wanted in expected.items():
        equal(value.get(key), wanted, "successful exact-fit audit: " + key)
    for key in ("per_arm_local_fitting_gate", "mixed_gain_at_least_002"):
        equal(value.get(key), read(record)["gates"][key], "retain audited predecessor gate")
    command = value.get("command")
    if (not isinstance(command, list) or len(command) != 8 or any(type(item) is not str for item in command)
            or command[0] != "python" or command[2:5] != ["--check", "--recompute", "--output"] or command[6] != "--record"):
        raise ValueError("canonical read-only predecessor command required")
    for actual, wanted in ((command[1], REPO / "manuscript/tools/diagnose_fpga_lm_exact_fit.py"),
                           (command[5], root), (command[7], Path(record))):
        path = Path(actual)
        equal((path if path.is_absolute() else REPO / path).resolve(strict=True), wanted, "audit command path")
    try:
        times = [datetime.strptime(value[key], "%Y-%m-%d %H:%M:%S UTC") for key in ("started_at_utc", "completed_at_utc")]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("canonical successful audit timestamps required") from error
    for key, timestamp in zip(("started_at_utc", "completed_at_utc"), times):
        equal(value[key], timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "audit timestamp spelling")
    if not 0 <= (times[1] - times[0]).total_seconds() <= 3600:
        raise ValueError("predecessor audit exceeded deadline")


def _controls(record):
    cells = read(record)["cells"]
    equal([cell["arm"] for cell in cells], list(parent.ARMS), "all predecessor controls retained")
    result = {}
    for cell in cells:
        if cell["arm"] in ("mixed", "prior_gru"):
            equal(cell["complete"], True, "complete prior control")
            equal(cell["optimizer_commits"], UPDATES, "matched prior update budget")
            result[cell["arm"]] = {key: parent.mean_metrics([cell[key]]) for key in ("initial_mean", "final_mean")}
    return result


def bind_inputs(config):
    root = Path(config.prior_run).resolve(strict=True)
    record, audit = head._regular(config.prior_record), head._regular(config.prior_audit)
    sources = source_identity()
    closure = head.ancestry_snapshot(root, record, audit)
    previous = read(root / "manifest.json")
    # Both public check wrappers own SIGALRM. Invoke the unchanged semantic
    # bodies under this invocation's timer; never cancel/reset an outer timer.
    previous_binding, raw, previous_validation = parent.bind_inputs(parent._bound_config(previous))
    equal(previous["binding"], previous_binding, "canonical complete predecessor binding")
    parent._check(root, record, False, previous_validation, raw)
    _audit_receipt(root, record, audit)
    supplied_raw, dataset = pilot.corpus_plan(Path(config.corpus), 1024, 88)
    equal(dataset, previous_binding["dataset"], "same full corpus and partition")
    equal(supplied_raw, raw, "same complete corpus bytes")
    initial_path = root / "mixed/state_000000.npz"
    binding = {"prior_run": str(root), "prior_record": str(record), "prior_audit": str(audit),
               "prior_binding": previous_binding, "dataset": dataset, "initial_state": str(initial_path),
               "initial_state_identity": old.file_identity(initial_path),
               "initial_scores": [old.file_identity(root / "mixed" / f"score_initial_{index:03d}_1.npz") for index in WINDOWS],
               "controls": _controls(record), "closure": closure}
    equal(head.ancestry_snapshot(root, record, audit), closure, "ancestor bytes unchanged through semantics")
    equal(source_identity(), sources, "source bytes unchanged through semantics")
    return binding, raw, head._Validation(binding, sources)


def frozen(manifest, validation):
    if not isinstance(validation, head._Validation):
        raise ValueError("invocation-local semantic validation required")
    equal(manifest["schema"], SCHEMA, "new versioned schema")
    equal(manifest["recipe"], recipe(), "fixed rate-only recipe")
    equal(manifest["source_hashes"], source_identity(), "complete source identity")
    equal(manifest["source_hashes"], validation.sources, "invocation source identity")
    equal(manifest["binding"], validation.binding, "original canonical semantic binding")
    binding = manifest["binding"]
    equal(head.ancestry_snapshot(binding["prior_run"], binding["prior_record"], binding["prior_audit"]),
          binding["closure"], "complete sealed ancestry unchanged")


def initial_model(binding):
    cell = parent.initial_models(binding["prior_binding"])["mixed"]
    previous = load(binding["initial_state"])
    same(previous, {"parameter/" + name: value for name, value in cell["parameters"].items()},
         "exact inherited mixed initial parameters and empty Adam state")
    equal(old.file_identity(binding["initial_state"]), binding["initial_state_identity"], "initial checkpoint identity")
    return cell


def make_optimizer(parameters):
    # Constructor semantics are frozen; explicitly change only the declared lr.
    optimizer = helper.make_optimizer(parameters)
    optimizer.param_groups[0]["lr"] = LR
    return optimizer


def _optimizer_group(cell, optimizer):
    if len(optimizer.param_groups) != 1:
        raise ValueError("one exact eager Adam group required")
    group = optimizer.param_groups[0]
    parameters = list(cell["parameters"].values())
    if len(group["params"]) != len(parameters) or any(left is not right for left, right in zip(group["params"], parameters)):
        raise ValueError("exact ordered parameter ownership required")
    expected = {"lr": LR, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0,
                "amsgrad": False, "maximize": False, "foreach": False, "fused": False,
                "capturable": False, "differentiable": False}
    for key, wanted in expected.items():
        equal(group.get(key), wanted, "exact rate-only Adam option: " + key)
    if set(group) - set(expected) - {"params", "decoupled_weight_decay"}:
        raise ValueError("unknown Adam option")
    if "decoupled_weight_decay" in group:
        equal(group["decoupled_weight_decay"], False, "no AdamW conversion")
    return json.loads(json.dumps({key: value for key, value in group.items() if key != "params"}, allow_nan=False))


def state_record(directory, step, cell, optimizer, *, write=False):
    if type(step) is not int or not 0 <= step <= UPDATES or type(write) is not bool:
        raise ValueError("literal declared state step/write mode required")
    parent._finite_parameters(cell)
    group = _optimizer_group(cell, optimizer)
    parameters, arrays, shapes = cell["parameters"], {}, {}
    if set(optimizer.state) - set(parameters.values()):
        raise ValueError("unowned Adam state")
    for name, parameter in parameters.items():
        arrays["parameter/" + name] = parameter.detach()
        moment = optimizer.state.get(parameter, {})
        equal(set(moment), set() if step == 0 else {"step", "exp_avg", "exp_avg_sq"}, "complete actual Adam state")
        shapes[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype), "initialized": step != 0}
        if step:
            counter = moment["step"]
            if (counter.device.type != "cpu" or counter.dtype not in (torch.float32, torch.float64)
                    or counter.shape != torch.Size([]) or not torch.isfinite(counter) or float(counter) != step):
                raise ValueError("exact finite scalar Adam counter required")
            for key in ("exp_avg", "exp_avg_sq"):
                value = moment[key]
                helper._finite(value, "Adam moment", detached=True)
                if value.shape != parameter.shape or (key == "exp_avg_sq" and bool((value < 0).any())):
                    raise ValueError("complete valid first/second moments required")
                arrays[key + "/" + name] = value
            arrays["step/" + name] = counter
            shapes[name]["step_dtype"] = str(counter.dtype)
    path = Path(directory) / f"state_{step:06d}.npz"
    if write:
        save(path, arrays)
    same(load(path), arrays, "all parameters/moments/counters")
    payload = path.read_bytes()
    result = {"schema": SCHEMA, "file": path.name, "bytes": len(payload), "sha256": old.digest(payload),
              "step": step, "parameter_order": list(parameters), "parameter_state": shapes,
              "optimizer_group": group, "initial_uninitialized_state_is_empty": step == 0,
              "parameters_sha256": tensor_hash(parameters)}
    if write:
        pilot.write_json(path.with_suffix(".json"), result)
    equal(read(path.with_suffix(".json")), result, "complete checkpoint metadata")
    return result


def _initial_parity(directory, binding):
    same(load(Path(directory) / "state_000000.npz"), load(binding["initial_state"]),
         "actual full initial parameters and empty Adam state equal prior mixed")
    equal(old.file_identity(binding["initial_state"]), binding["initial_state_identity"], "prior initial checkpoint bytes")
    descriptors = binding["initial_scores"]
    if not isinstance(descriptors, list) or len(descriptors) != len(WINDOWS):
        raise ValueError("all four actual initial score descriptors required")
    for index, descriptor in zip(WINDOWS, descriptors):
        source = Path(binding["prior_run"]) / "mixed" / f"score_initial_{index:03d}_1.npz"
        equal(old.file_identity(source), descriptor, "exact ordered original score identity")
        same(load(Path(directory) / f"score_initial_{index:03d}_1.npz"), load(source),
             "exact prior mixed initial logits and all selected routes")


def _support_counts(arrays, head_indices, projection_indices, *, prefix=""):
    head_routes, projection_routes = arrays[prefix + "AD_HEAD"], arrays[prefix + "PROJ_IDX"]
    return {"head_entries": head_routes.numel(), "head_off_initial_support": int((head_routes != head_indices[None, :]).sum()),
            "projection_entries": projection_routes.numel(),
            "projection_off_initial_support": int((projection_routes != projection_indices[:, None, :]).sum())}


def gates(cell, controls):
    if cell.get("complete") is not True or cell.get("numerics_pass") is not True or cell.get("arm") != ARM:
        raise ValueError("one complete healthy declared rate cell required")
    equal(set(controls), {"mixed", "prior_gru"}, "exact historical controls")
    for control in controls.values():
        equal(set(control), {"initial_mean", "final_mean"}, "complete control metrics")
        for metric in control.values():
            parent.mean_metrics([metric])
    new = parent.mean_metrics([cell["final_mean"]])["ce"]
    baseline, gru = (controls[key]["final_mean"]["ce"] for key in ("mixed", "prior_gru"))
    if baseline <= gru:
        raise ValueError("bound historical mixed-to-GRU gap must be positive")
    threshold, near = (baseline + gru) / 2, gru + 0.05
    return {"all_numerics_pass": True, "prior_mixed_ce": baseline, "prior_gru_ce": gru, "new_ce": new,
            "half_gap_ce_threshold": threshold, "near_gru_ce_threshold": near,
            "fraction_of_historical_gap_closed": (baseline - new) / (baseline - gru),
            "half_gap_closed": new <= threshold + GATE_TOLERANCE,
            "near_gru_local_fit": new <= near + GATE_TOLERANCE,
            "requires_separate_successful_independent_recompute": True,
            "recipe_promoted": False, "paid_execution_authorized": False, "broader_study_authorized": False,
            "scope": "one training-data-informed rate, same local fitting task; not independent tuning or generalization"}


def execute(manifest, output, raw, *, write=False, recompute=False):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("literal disjoint write/recompute modes required")
    output, binding = Path(output), manifest["binding"]
    tokens, targets = batch_data(raw, binding["dataset"])
    cell = initial_model(binding)
    equal(cell["kind"], "tropical", "only mixed tropical cell")
    parent._finite_parameters(cell)
    vocab = cell["parameters"]["EMBED"].shape[0]
    helper._tokens(tokens, vocab, ndim=2)
    helper._tokens(targets, vocab, ndim=2)
    if tokens.shape != targets.shape or tokens.shape[0] != len(WINDOWS):
        raise ValueError("all four complete aligned training windows required")
    directory = output / ARM
    if write:
        directory.mkdir(exist_ok=False)
    optimizer = make_optimizer(cell["parameters"].values())
    initial_parameters = {name: value.detach().clone() for name, value in cell["parameters"].items()}
    head_indices = initial_parameters["W_HEAD"].argmax(dim=1)
    projection_indices = initial_parameters["MP"].argmax(dim=2)
    states = [state_record(directory, 0, cell, optimizer, write=write)]
    initial_scores = collect_scores(directory, "initial", cell, tokens, targets, True, write=write, recompute=recompute)
    _initial_parity(directory, binding)
    equal(parent.mean_metrics(initial_scores), binding["controls"]["mixed"]["initial_mean"], "identical historical initial metrics")
    rows, support_totals = [], {key: 0 for key in ("head_entries", "head_off_initial_support", "projection_entries", "projection_off_initial_support")}
    for step in range(UPDATES):
        gradient = query_batch(directory, step, cell, tokens, targets, write=write, recompute=recompute)
        path = directory / f"query_{step:06d}.npz"
        query_scores, query_mean = parent._query_metrics(path, cell, targets)
        arrays = load(path)
        supports = [_support_counts(arrays, head_indices, projection_indices, prefix=f"window_{index}/") for index in WINDOWS]
        for support in supports:
            for key in support_totals:
                support_totals[key] += support[key]
        before = {name: value.detach().clone() for name, value in cell["parameters"].items()}
        parent._gradients(cell, gradient)
        optimizer.zero_grad(set_to_none=True)
        for name, value in cell["parameters"].items():
            value.grad = gradient[name].detach().clone()
        optimizer.step()
        parent._finite_parameters(cell)
        states.append(state_record(directory, step + 1, cell, optimizer, write=write))
        payload = path.read_bytes()
        rows.append({"step_before": step, "step_after": step + 1, "query_file": path.name,
                     "query_bytes": len(payload), "query_sha256": old.digest(payload),
                     "before_parameters_sha256": tensor_hash(before), "after_parameters_sha256": tensor_hash(cell["parameters"]),
                     "gradient_sha256": tensor_hash(gradient), "query_scores": query_scores, "query_mean": query_mean,
                     "gradient_statistics": statistics(gradient), "query_supports": supports,
                     "update_statistics": statistics({name: value.detach() - before[name] for name, value in cell["parameters"].items()}),
                     "complete_saved_gradient_adam_replay": True})
        if (write or recompute) and (step + 1) % 16 == 0:
            print(json.dumps({"arm": ARM, "step": step + 1, "query_ce": query_mean["ce"], "recompute": recompute}), flush=True)
    final = collect_scores(directory, "final", cell, tokens, targets, True, write=write, recompute=recompute)
    disabled = collect_scores(directory, "final", cell, tokens, targets, False, write=write, recompute=recompute)
    final_supports = [_support_counts(load(directory / f"score_final_{index:03d}_1.npz"), head_indices, projection_indices) for index in WINDOWS]
    result = {"schema": SCHEMA, "arm": ARM, "kind": "tropical", "complete": True, "numerics_pass": True,
              "optimizer_commits": UPDATES, "full_batch_gradient_queries": UPDATES, "complete_saved_states": UPDATES + 1,
              "initial_scores": initial_scores, "initial_mean": parent.mean_metrics(initial_scores),
              "final_scores": final, "final_mean": parent.mean_metrics(final),
              "context_off_scores": disabled, "context_off_mean": parent.mean_metrics(disabled),
              "context_ce_penalty": parent.mean_metrics(disabled)["ce"] - parent.mean_metrics(final)["ce"],
              "query_learning_curve": rows, "state_receipts": states, "query_support_totals": support_totals,
              "final_supports": final_supports, "initial_parameters_and_score_traces_equal_prior_mixed": True,
              "net_parameter_motion": statistics({name: value.detach() - initial_parameters[name] for name, value in cell["parameters"].items()}),
              "trainable_parameters": sum(value.numel() for value in cell["parameters"].values()),
              "new_gradient_pair_exposures": UPDATES * tokens.numel(), "distinct_gradient_pairs": tokens.numel(),
              "new_fitting_pair_exposures": 0, "inherited_fitting_pair_exposures": 131072,
              "endpoint_history_pair_exposures": 131072 + UPDATES * tokens.numel(), "new_native_training_calls": 0}
    if write:
        pilot.write_json(directory / "summary.json", result)
    equal(read(directory / "summary.json"), result, "complete reconstructed rate cell")
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "cell": result, "gates": gates(result, binding["controls"]), "historical_controls": binding["controls"],
            "full_batch_gradient_queries": 128, "optimizer_commits": 128, "complete_saved_states": 129,
            "score_window_readouts": 12, "independent_tropical_window_vjps": 512,
            "new_gradient_pair_exposures": UPDATES * tokens.numel(), "distinct_gradient_pairs": tokens.numel(),
            "new_fitting_pair_exposures": 0, "inherited_fitting_pair_exposures": 131072,
            "endpoint_history_pair_exposures": 131072 + UPDATES * tokens.numel(), "distinct_fitted_pairs": 131072,
            "historical_comparator_updates_reexecuted": 0, "new_native_training_calls": 0,
            "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
            "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "one fixed local rate comparison; historical controls are not new replications or matched tuning budgets"}


def expected_artifacts():
    files = {"manifest.json", "protocol.json", "declaration.md", "summary.json", ARM + "/summary.json"}
    files.update(f"{ARM}/query_{step:06d}.npz" for step in range(UPDATES))
    files.update(f"{ARM}/state_{step:06d}.{suffix}" for step in range(UPDATES + 1) for suffix in ("npz", "json"))
    files.update(f"{ARM}/score_initial_{index:03d}_1.npz" for index in WINDOWS)
    files.update(f"{ARM}/score_final_{index:03d}_{context}.npz" for index in WINDOWS for context in (0, 1))
    return files


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root), "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "check_scope": "exact 128 saved-gradient Adam transitions/129 full states; separate --recompute reconstructs 512 window "
                           "VJPs and compares 128 batch gradients plus 12 score readouts; no new optimizer trajectory"}


def _check(root, record, recompute, validation, raw):
    if type(recompute) is not bool:
        raise ValueError("literal independent check mode required")
    root, record = Path(root).resolve(strict=True), head._regular(record)
    manifest = read(root / "manifest.json")
    frozen(manifest, validation)
    equal(manifest["run_directory"], str(root), "owned run directory")
    equal(read(root / "protocol.json"), recipe(), "exact saved recipe")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "saved declaration bytes")
    artifacts = prior.artifact_manifest(root)
    seal_bytes, public_bytes = (root / "artifact_manifest.json").read_bytes(), record.read_bytes()
    equal(read(root / "artifact_manifest.json"), artifacts, "complete sealed inventory")
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact payload membership")
    summary = execute(manifest, root, raw, recompute=recompute)
    equal(read(root / "summary.json"), summary, "complete reconstructed result")
    expected = public_record(root, summary, artifacts)
    equal(read(record), expected, "public/raw equality")
    frozen(manifest, validation)
    equal(prior.artifact_manifest(root), artifacts, "actual payload unchanged through check")
    equal((root / "artifact_manifest.json").read_bytes(), seal_bytes, "exact seal bytes unchanged")
    equal(record.read_bytes(), public_bytes, "exact public bytes unchanged")
    return expected


def _bound_config(manifest):
    binding = manifest["binding"]
    return SimpleNamespace(prior_run=binding["prior_run"], prior_record=binding["prior_record"],
                           prior_audit=binding["prior_audit"], corpus=binding["dataset"]["path"])


def check(root, record, recompute=False):
    if type(recompute) is not bool:
        raise ValueError("literal independent check mode required")
    previous, installed = None, False
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed rate-control independent audit deadline exceeded")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        manifest = read(Path(root) / "manifest.json")
        binding, raw, validation = bind_inputs(_bound_config(manifest))
        equal(manifest["binding"], binding, "fresh canonical input semantics")
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
        raise FileNotFoundError("public parent must already exist")
    sources = source_identity()
    output.mkdir(parents=True, exist_ok=False)
    previous, installed, published = None, False, None
    staged, staging_directory, owned_inode = None, None, None
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed rate-control run deadline exceeded")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        binding, raw, validation = bind_inputs(config)
        equal(source_identity(), sources, "source bytes unchanged before declaration")
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
        equal({row["file"] for row in artifacts}, expected_artifacts(), "complete payload before publication")
        pilot.write_json(output / "artifact_manifest.json", artifacts)
        public = public_record(output, summary, artifacts)
        published = old.digest((json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-exact-rate-", dir=target.parent))
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
    for name in ("prior-run", "prior-record", "prior-audit", "corpus"):
        parser.add_argument("--" + name, type=Path)
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
            parser.error("new run requires prior run/record/audit and corpus")
        result = run(args)
    print(json.dumps({"complete": result["complete"], "gates": result["gates"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
