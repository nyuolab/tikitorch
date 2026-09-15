"""Three fixed exact-CE fitted-start controls; training data only, CPU FP64.

Every saved Adam transition is replayed exactly. Independent checking uses
separate tropical selection/reverse and explicit GRU-cell implementations,
without changing any recorded optimizer trajectory or selecting a recipe.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_head_weighting as parent  # noqa: E402
from manuscript.tools import fpga_lm_exact_fit as helper  # noqa: E402
from manuscript.tools import fpga_lm_projection_topology as topology  # noqa: E402

old, prior, pilot = parent.old, parent.prior, parent.pilot
head = parent
read, equal, save = parent.read, parent.equal, parent.parent.save
SCHEMA = "tikitorch.fpga_lm_exact_fit.v1"
ARMS, WINDOWS = ("isolated", "mixed", "prior_gru"), (0, 1, 2, 3)
UPDATES, DEADLINE_S, GATE_TOLERANCE = 128, 3600, 1e-12
PLAN = REPO / "manuscript/research/PLAN.md"


def source_identity():
    result = dict(parent.source_identity())
    for name in ("manuscript/tools/fpga_lm_projection_topology.py", "tests/test_fpga_lm_projection_topology.py",
                 "manuscript/tools/fpga_lm_exact_fit.py", "tests/test_fpga_lm_exact_fit.py",
                 "manuscript/tools/diagnose_fpga_lm_exact_fit.py", "tests/test_fpga_lm_exact_fit_replay.py"):
        result[name] = old.digest((REPO / name).read_bytes())
    return result


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "training_windows": list(WINDOWS),
            "updates_per_arm": UPDATES, "batch": "same four complete 128-token windows, reset state at boundaries",
            "seed": 0, "dtype": "float64", "device": "cpu", "tropical_units": "u=beta*W; logits dimensionless; beta used once",
            "tropical_forward": "real mean centering; dynamic first-index max/min selection; no floor/LUT/EF",
            "tropical_backward": "selected derivative of exact mean CE, away from routing ties; declared first selector at ties",
            "topology": "fixed seed0 conjugated rotations2**layer; only MP changes; initial function preserved",
            "gru": {"layers": 1, "embedding": 128, "hidden": 128, "vocabulary": 128,
                    "prior": "frozen log_softmax(beta*(E[token,A_v]+head[v,A_v]))", "initial_correction": "zero head weight and bias",
                    "trainable_parameters": 131968},
            "tropical_parameters_per_arm": 1196032, "cost_or_parameter_matching": False,
            "optimizer": {"name": "eager Adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
                          "weight_decay": 0.0, "foreach": False, "fused": False, "gradient_clipping": False},
            "initial_probability_tolerance": {"atol": 1e-12, "rtol": 1e-12},
            "independent_tolerance": {"atol": 1e-9, "rtol": 1e-11, "integer_routes": "exact"},
            "readouts": "four windows initially context-on; final context-on and context-off, for every arm",
            "tropical_maximum_past_offset": 90, "gru_maximum_past_offset": 127,
            "context_off": "one tropical causal tap; reset GRU hidden state at each position",
            "declared_cells": 3, "full_batch_gradient_queries": 384, "optimizer_commits": 384,
            "complete_saved_states": 387, "score_window_readouts": 36,
            "independent_tropical_window_queries": 1024, "independent_gru_batch_queries": 128,
            "gradient_audit": "reconstruct 1024 tropical window VJPs, sum in original order and compare 256 saved full-batch gradients; "
                              "compare 128 saved GRU full-batch gradients to explicit-cell autograd",
            "new_gradient_pair_exposures": 196608, "distinct_gradient_pairs": 512,
            "new_fitting_pair_exposures": 0, "summed_inherited_fitting_pair_exposures": 393216,
            "summed_endpoint_history_pair_exposures": 589824, "distinct_fitted_pairs": 131072,
            "floor": "unsmoothed empirical H(Y|current token) on the same512pairs; descriptive counts, not a new fit",
            "gate": {"per_arm_gain_below_current_token_floor_nats": 0.02, "mixed_gain_over_isolated_nats": 0.02,
                     "inclusive_tolerance": GATE_TOLERANCE},
            "torch_threads": 1, "deadline_s": DEADLINE_S, "adaptive_extension": False,
            "new_native_training_calls": 0, "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
            "paid_execution_authorized": False, "recipe_promoted": False, "statistical_significance_claim": False}


def _audit_receipt(root, record, audit):
    value, artifacts = read(audit), prior.artifact_manifest(root)
    expected = {"schema": "tikitorch.fpga_lm_head_weighting.cli_audit_receipt.v1",
                "complete": True, "independent_recompute": True, "exit_code": 0, "deadline_seconds": 3600,
                "parent_record_sha256": old.digest(Path(record).read_bytes()),
                "run_manifest_sha256": old.digest((root / "manifest.json").read_bytes()),
                "artifact_manifest_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
                "runner_sha256": parent.source_identity()["manuscript/tools/diagnose_fpga_lm_head_weighting.py"],
                "source_files_verified": len(parent.source_identity()),
                "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
                "independent_full_gradient_queries": 192, "independent_score_readouts": 302,
                "complete_saved_states_verified": 198, "optimizer_commits_replayed": 192,
                "complete_parent_ancestry_verified": True, "source_input_raw_seal_public_bytes_unchanged": True,
                "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
                "new_optimizer_trajectory_executed": False, "heldout_corpus_evaluated": False,
                "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
                "broader_study_authorized": False}
    for key, expected_value in expected.items():
        equal(value.get(key), expected_value, "completed head-weighting audit: " + key)
    equal(value.get("numerical_training_gate_pass"), read(record)["gates"]["numerical_training_gate_pass"], "retain prior gate result")
    command = value.get("command")
    if (not isinstance(command, list) or len(command) != 8 or any(type(item) is not str for item in command)
            or command[0] != "python" or command[2:5] != ["--check", "--recompute", "--output"] or command[6] != "--record"):
        raise ValueError("canonical predecessor audit command required")
    for actual, expected_path in ((command[1], REPO / "manuscript/tools/diagnose_fpga_lm_head_weighting.py"),
                                  (command[5], root), (command[7], Path(record))):
        path = Path(actual)
        equal((path if path.is_absolute() else REPO / path).resolve(strict=True), expected_path, "audit command path")
    try:
        times = [datetime.strptime(value[key], "%Y-%m-%d %H:%M:%S UTC") for key in ("started_at_utc", "completed_at_utc")]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("canonical audit timestamps required") from error
    for key, timestamp in zip(("started_at_utc", "completed_at_utc"), times):
        equal(value[key], timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "audit timestamp spelling")
    if not 0 <= (times[1] - times[0]).total_seconds() <= 3600:
        raise ValueError("predecessor audit exceeded its declared deadline")


def bind_inputs(config):
    root = Path(config.prior_run).resolve(strict=True)
    record, audit = parent._regular(config.prior_record), parent._regular(config.prior_audit)
    sources = source_identity()
    closure = parent.ancestry_snapshot(root, record, audit)
    # Reuse the exact semantic body, not the public timer-owning wrapper:
    # parent.check would cancel this invocation's already-armed deadline.
    previous = read(root / "manifest.json")
    parent_binding, parent_raw, parent_validation = parent.bind_inputs(parent._bound_config(previous))
    equal(previous["binding"], parent_binding, "fresh canonical head-study binding")
    parent._check(root, record, False, parent_validation, parent_raw)
    _audit_receipt(root, record, audit)
    previous = read(root / "manifest.json")
    raw, dataset = pilot.corpus_plan(Path(config.corpus), 1024, 88)
    equal(dataset, previous["binding"]["dataset"], "same complete corpus/split")
    binding = {"prior_run": str(root), "prior_record": str(record), "prior_audit": str(audit),
               "dataset": dataset, "initial_state": previous["binding"]["initial_state"], "closure": closure}
    equal(parent.ancestry_snapshot(root, record, audit), closure, "complete ancestor bytes unchanged during validation")
    equal(source_identity(), sources, "source bytes unchanged during validation")
    return binding, raw, parent._Validation(binding, sources)


def frozen(manifest, validation):
    if not isinstance(validation, parent._Validation):
        raise ValueError("private invocation-local validation required")
    equal(manifest["schema"], SCHEMA, "schema")
    equal(manifest["recipe"], recipe(), "declared recipe")
    equal(manifest["source_hashes"], source_identity(), "complete source identity")
    equal(manifest["source_hashes"], validation.sources, "invocation source identity")
    binding = manifest["binding"]
    equal(binding, validation.binding, "canonical semantically checked binding")
    equal(parent.ancestry_snapshot(binding["prior_run"], binding["prior_record"], binding["prior_audit"]),
          binding["closure"], "full ancestor trees/external receipts unchanged")


def initial_models(binding):
    shape = pilot.layout(512)
    decoded = old.native.decode_state(Path(binding["initial_state"]).read_bytes(), shape)
    equal([decoded["step"], decoded["k"], decoded["config"]["loss_kind"], decoded["config"]["zshift"]],
          [0, 18, 1, 2], "bound original diverse fitted state")
    raw_weights, remainders = old.state_tensors(decoded)
    old.require_zero(remainders, "initial inherited remainders exactly zero")
    flat = {key: raw_weights[key].to(torch.int64).reshape(-1).tolist() for key in helper.WEIGHTS}
    mixed, facts = topology.transform(flat, shape)
    mixed_weights = {key: torch.tensor(mixed[key], dtype=torch.float64).reshape(raw_weights[key].shape) for key in helper.WEIGHTS}
    anchors = topology.mapping(shape)["A"]
    logits = helper.BETA * (raw_weights["EMBED"][:, anchors]
                            + raw_weights["W_HEAD"][torch.arange(128), anchors][None, :])
    prior_logits = logits.log_softmax(dim=-1).detach()
    model = helper.make_gru(seed=0, vocab=128, embedding=128, hidden=128)
    return {"isolated": {"kind": "tropical", "parameters": helper.dimensionless(raw_weights), "model": None,
                         "prior_logits": None, "initialization": {"topology": "original", "weights_sha256": old.tensor_hash(raw_weights)}},
            "mixed": {"kind": "tropical", "parameters": helper.dimensionless(mixed_weights), "model": None,
                      "prior_logits": None, "initialization": facts},
            "prior_gru": {"kind": "gru", "parameters": dict(model.named_parameters()), "model": model,
                          "prior_logits": prior_logits, "initialization": {"seed": 0, "zero_output_correction": True}}}


def batch_data(raw, dataset):
    pairs = [parent.parent.window(raw, dataset, index) for index in WINDOWS]
    return torch.stack([item[0] for item in pairs]), torch.tensor([item[1] for item in pairs], dtype=torch.int64)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate tensor keys")
        result = {}
        for key in archive.files:
            array = archive[key]
            if array.dtype not in (np.dtype("float32"), np.dtype("float64"), np.dtype("int64")) or not np.isfinite(array).all():
                raise ValueError("finite FP32/FP64/int64 arrays required")
            result[key] = torch.from_numpy(array.copy())
    return result


def same(actual, expected, label):
    equal(set(actual), set(expected), label + " fields")
    for key in expected:
        left, right = actual[key], expected[key]
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(label + " tensor differs: " + key)


def tensor_hash(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def statistics(values):
    result = {}
    for name, value in values.items():
        value = value.detach()
        helper._finite(value, "parameter/gradient statistic")
        result[name] = {"elements": value.numel(), "l1": float(value.abs().sum()), "l2": float(value.norm()),
                        "linf": float(value.abs().max()), "nonzero": int(torch.count_nonzero(value))}
    return result


def _finite_parameters(cell):
    if cell["kind"] == "tropical":
        helper.validate_weights(cell["parameters"])
    elif cell["kind"] == "gru":
        cell["model"]._validate()
        equal(list(cell["parameters"]), list(dict(cell["model"].named_parameters())), "canonical GRU parameter order")
        for key, value in cell["model"].named_parameters():
            if cell["parameters"][key] is not value:
                raise ValueError("GRU parameter mapping does not own the module parameters")
        helper._finite(cell["prior_logits"], "frozen fitted prior", detached=True)
    else:
        raise ValueError("declared model kind required")


def _optimizer_group(cell, optimizer):
    if len(optimizer.param_groups) != 1:
        raise ValueError("exactly one eager Adam parameter group required")
    group = optimizer.param_groups[0]
    if len(group["params"]) != len(cell["parameters"]) or any(left is not right for left, right in zip(group["params"], cell["parameters"].values())):
        raise ValueError("exact ordered Adam parameter ownership required")
    expected = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0,
                "amsgrad": False, "maximize": False, "foreach": False, "fused": False,
                "capturable": False, "differentiable": False}
    for key, value in expected.items():
        equal(group.get(key), value, "fixed eager Adam option: " + key)
    if set(group) - set(expected) - {"params", "decoupled_weight_decay"}:
        raise ValueError("unrecognized Adam options")
    if "decoupled_weight_decay" in group:
        equal(group["decoupled_weight_decay"], False, "no AdamW conversion")
    return json.loads(json.dumps({key: value for key, value in group.items() if key != "params"}, allow_nan=False))


def state_record(directory, step, cell, optimizer, *, write=False):
    if type(step) is not int or not 0 <= step <= UPDATES:
        raise ValueError("literal declared Adam step required")
    _finite_parameters(cell)
    group = _optimizer_group(cell, optimizer)
    parameters = cell["parameters"]
    arrays, state_shapes = {}, {}
    if set(optimizer.state) - set(parameters.values()):
        raise ValueError("unowned Adam state")
    for name, parameter in parameters.items():
        arrays["parameter/" + name] = parameter.detach()
        moment = optimizer.state.get(parameter, {})
        equal(set(moment), set() if step == 0 else {"step", "exp_avg", "exp_avg_sq"}, "complete actual Adam state")
        state_shapes[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype), "initialized": step != 0}
        if step:
            counter = moment["step"]
            if (counter.device.type != "cpu" or counter.dtype not in (torch.float32, torch.float64)
                    or counter.shape != torch.Size([]) or not torch.isfinite(counter) or float(counter) != step):
                raise ValueError("exact finite scalar Adam counter required")
            for key in ("exp_avg", "exp_avg_sq"):
                value = moment[key]
                helper._finite(value, "Adam " + key, detached=True)
                if value.shape != parameter.shape or (key == "exp_avg_sq" and bool((value < 0).any())):
                    raise ValueError("complete nonnegative second moment/valid first moment required")
                arrays[key + "/" + name] = value
            arrays["step/" + name] = counter
            state_shapes[name]["step_dtype"] = str(counter.dtype)
    path = Path(directory) / f"state_{step:06d}.npz"
    if write:
        save(path, arrays)
    same(load(path), arrays, "complete actual parameter and Adam-state bytes")
    payload = path.read_bytes()
    record = {"schema": SCHEMA, "file": path.name, "bytes": len(payload), "sha256": old.digest(payload),
              "step": step, "parameter_order": list(parameters), "parameter_state": state_shapes,
              "optimizer_group": group, "initial_uninitialized_state_is_empty": step == 0,
              "parameters_sha256": tensor_hash(parameters)}
    if write:
        pilot.write_json(path.with_suffix(".json"), record)
    equal(read(path.with_suffix(".json")), record, "complete Adam checkpoint metadata")
    return record


def metrics(logits, targets):
    helper._finite(logits, "dimensionless score logits", detached=True)
    if logits.ndim != 2:
        raise ValueError("complete per-window logits required")
    ids = helper._targets(targets, logits.shape[0], logits.shape[1])
    return {"ce": float(F.cross_entropy(logits, ids)),
            "accuracy": float((logits.argmax(-1) == ids).to(torch.float64).mean())}


def mean_metrics(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty per-window metrics required")
    for row in rows:
        equal(set(row), {"ce", "accuracy"}, "exact CE metric fields")
        if any(type(row[key]) is not float or not math.isfinite(row[key]) for key in row):
            raise ValueError("finite literal metric values required")
        if row["ce"] < 0 or not 0 <= row["accuracy"] <= 1:
            raise ValueError("metric range invalid")
    return {key: math.fsum(row[key] for row in rows) / len(rows) for key in ("ce", "accuracy")}


def _base(trace, parameters, tokens, context):
    equal(set(trace), set(helper.TRACE_KEYS), "complete tropical trace")
    base = {"schema": helper.reference.SCHEMA, "mode": "smooth", "helper_schema": helper.SCHEMA,
            "units": "dimensionless", "geometry": helper.reference._geometry(parameters, tokens),
            "tokens": tokens, "use_context": context, "logits": trace["logits"],
            "centering_offsets": trace["centering_offsets"],
            "routes": {key: trace[key] for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")},
            "parameter_sha256": helper.parameter_digest(parameters), "native_execution": False,
            "independent_forward": False}
    helper.trace(base)
    if bool(trace["centering_offsets"].count_nonzero()):
        raise ValueError("real-centering trace must have zero floor offsets")
    return base


def _gradients(cell, values):
    equal(set(values), set(cell["parameters"]), "all parameter gradients")
    for name, parameter in cell["parameters"].items():
        value = values[name]
        helper._finite(value, "full batch gradient", detached=True)
        if value.shape != parameter.shape:
            raise ValueError("full batch gradient geometry differs")


def _gru_query(cell, tokens, targets, independent=False):
    with torch.enable_grad():
        logits = cell["model"](tokens, cell["prior_logits"], independent=independent)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        gradients = torch.autograd.grad(loss, tuple(cell["parameters"].values()))
    return logits.detach(), {name: value.detach() for name, value in zip(cell["parameters"], gradients)}


def query_batch(directory, step, cell, tokens, targets, *, write=False, recompute=False):
    _finite_parameters(cell)
    path = Path(directory) / f"query_{step:06d}.npz"
    positions = targets.numel()
    if write:
        arrays = {"tokens": tokens, "targets": targets}
        if cell["kind"] == "tropical":
            gradients = {name: torch.zeros_like(value) for name, value in cell["parameters"].items()}
            for row, index in enumerate(WINDOWS):
                base = helper.forward(cell["parameters"], tokens[row])
                gradient = helper.gradient(cell["parameters"], tokens[row], targets[row], base, positions)
                for name in gradients:
                    gradients[name] = gradients[name] + gradient[name]
                arrays.update({f"window_{index}/" + name: value for name, value in helper.trace(base).items()})
        else:
            arrays["logits"], gradients = _gru_query(cell, tokens, targets)
        _gradients(cell, gradients)
        arrays.update({"g_" + name: value for name, value in gradients.items()})
        save(path, arrays)
    arrays = load(path)
    expected = {"tokens", "targets"} | {"g_" + name for name in cell["parameters"]}
    expected |= ({f"window_{index}/" + key for index in WINDOWS for key in helper.TRACE_KEYS}
                 if cell["kind"] == "tropical" else {"logits"})
    equal(set(arrays), expected, "complete batch query payload")
    same({key: arrays[key] for key in ("tokens", "targets")}, {"tokens": tokens, "targets": targets}, "exact bound batch pairs")
    gradients = {name: arrays["g_" + name] for name in cell["parameters"]}
    _gradients(cell, gradients)
    if cell["kind"] == "tropical":
        independent_gradient = {name: torch.zeros_like(value) for name, value in cell["parameters"].items()}
        for row, index in enumerate(WINDOWS):
            trace = {key: arrays[f"window_{index}/" + key] for key in helper.TRACE_KEYS}
            _base(trace, cell["parameters"], tokens[row], True)
            if recompute:
                base = helper.forward(cell["parameters"], tokens[row], independent=True)
                head.helper.near(trace, helper.trace(base), "independent tropical batch forward")
                values = helper.gradient(cell["parameters"], tokens[row], targets[row], base, positions, independent=True)
                for name in values:
                    independent_gradient[name] = independent_gradient[name] + values[name]
        if recompute:
            head.helper.near(gradients, independent_gradient, "independent full tropical batch gradient")
    else:
        logits = helper._finite(arrays["logits"], "complete GRU query logits", detached=True)
        if logits.shape != (*tokens.shape, cell["model"].vocab):
            raise ValueError("complete GRU batch logits required")
        if recompute:
            independent_logits, independent_gradient = _gru_query(cell, tokens, targets, independent=True)
            head.helper.near(logits, independent_logits, "independent expanded GRU cell forward")
            head.helper.near(gradients, independent_gradient, "independent expanded GRU full gradient")
    return gradients


def _query_metrics(path, cell, targets):
    arrays = load(path)
    logits = ([arrays[f"window_{index}/logits"] for index in WINDOWS] if cell["kind"] == "tropical"
              else list(arrays["logits"]))
    rows = [metrics(value, targets[row]) for row, value in enumerate(logits)]
    return rows, mean_metrics(rows)


def collect_scores(directory, label, cell, tokens, targets, context=True, *, write=False, recompute=False):
    _finite_parameters(cell)
    gru_logits = None
    if cell["kind"] == "gru" and (write or recompute):
        with torch.no_grad():
            gru_logits = cell["model"](tokens, cell["prior_logits"], context=context, independent=recompute)
    rows = []
    for row, index in enumerate(WINDOWS):
        path = Path(directory) / f"score_{label}_{index:03d}_{int(context)}.npz"
        if write:
            trace = (helper.trace(helper.forward(cell["parameters"], tokens[row], context=context))
                     if cell["kind"] == "tropical" else {"logits": gru_logits[row]})
            save(path, trace)
        arrays = load(path)
        if cell["kind"] == "tropical":
            _base(arrays, cell["parameters"], tokens[row], context)
            if recompute:
                base = helper.forward(cell["parameters"], tokens[row], context=context, independent=True)
                head.helper.near(arrays, helper.trace(base), "independent complete tropical score")
        else:
            equal(set(arrays), {"logits"}, "complete GRU score archive")
            if arrays["logits"].shape != (tokens.shape[1], cell["model"].vocab):
                raise ValueError("GRU score shape mismatch")
            if recompute:
                head.helper.near(arrays["logits"], gru_logits[row], "independent complete GRU score")
        rows.append(metrics(arrays["logits"], targets[row]))
    return rows


def _cell(directory, arm, cell, tokens, targets, initial_scores, *, write=False, recompute=False):
    optimizer = helper.make_optimizer(cell["parameters"].values())
    state_receipts = [state_record(directory, 0, cell, optimizer, write=write)]
    initial_parameters = {name: value.detach().clone() for name, value in cell["parameters"].items()}
    prior_before = None if cell["prior_logits"] is None else cell["prior_logits"].detach().clone()
    rows = []
    for step in range(UPDATES):
        gradient = query_batch(directory, step, cell, tokens, targets, write=write, recompute=recompute)
        query_path = Path(directory) / f"query_{step:06d}.npz"
        query_rows, query_mean = _query_metrics(query_path, cell, targets)
        before = {name: value.detach().clone() for name, value in cell["parameters"].items()}
        _gradients(cell, gradient)
        optimizer.zero_grad(set_to_none=True)
        for name, value in cell["parameters"].items():
            value.grad = gradient[name].detach().clone()
        optimizer.step()
        _finite_parameters(cell)
        state_receipts.append(state_record(directory, step + 1, cell, optimizer, write=write))
        if prior_before is not None:
            same({"prior": cell["prior_logits"]}, {"prior": prior_before}, "frozen fitted GRU prior")
        payload = query_path.read_bytes()
        rows.append({"step_before": step, "step_after": step + 1, "query_file": query_path.name,
                     "query_bytes": len(payload), "query_sha256": old.digest(payload),
                     "before_parameters_sha256": tensor_hash(before), "after_parameters_sha256": tensor_hash(cell["parameters"]),
                     "gradient_sha256": tensor_hash(gradient), "query_scores": query_rows, "query_mean": query_mean,
                     "gradient_statistics": statistics(gradient),
                     "update_statistics": statistics({name: value.detach() - before[name] for name, value in cell["parameters"].items()}),
                     "complete_saved_gradient_adam_replay": True})
        if (write or recompute) and (step + 1) % 16 == 0:
            print(json.dumps({"arm": arm, "step": step + 1, "query_ce": query_mean["ce"], "recompute": recompute}), flush=True)
    final = collect_scores(directory, "final", cell, tokens, targets, True, write=write, recompute=recompute)
    disabled = collect_scores(directory, "final", cell, tokens, targets, False, write=write, recompute=recompute)
    result = {"schema": SCHEMA, "arm": arm, "kind": cell["kind"], "complete": True, "numerics_pass": True,
              "optimizer_commits": UPDATES, "full_batch_gradient_queries": UPDATES, "complete_saved_states": UPDATES + 1,
              "initial_scores": initial_scores, "initial_mean": mean_metrics(initial_scores),
              "final_scores": final, "final_mean": mean_metrics(final),
              "context_off_scores": disabled, "context_off_mean": mean_metrics(disabled),
              "context_ce_penalty": mean_metrics(disabled)["ce"] - mean_metrics(final)["ce"],
              "query_learning_curve": rows, "state_receipts": state_receipts,
              "net_parameter_motion": statistics({name: value.detach() - initial_parameters[name] for name, value in cell["parameters"].items()}),
              "trainable_parameters": sum(value.numel() for value in cell["parameters"].values()),
              "new_gradient_pair_exposures": UPDATES * tokens.numel(), "new_fitting_pair_exposures": 0,
              "inherited_fitting_pair_exposures": 131072, "new_native_training_calls": 0}
    if write:
        pilot.write_json(Path(directory) / "summary.json", result)
    equal(read(Path(directory) / "summary.json"), result, "fully reconstructed model result")
    return result


def gates(cells, floor):
    if type(floor) is not float or not math.isfinite(floor) or floor < 0:
        raise ValueError("finite empirical conditional entropy required")
    if not isinstance(cells, list) or len(cells) != len(ARMS):
        raise ValueError("all three controls required before gates")
    lookup = {cell["arm"]: cell for cell in cells}
    if set(lookup) != set(ARMS) or any(cell.get("complete") is not True or cell.get("numerics_pass") is not True for cell in cells):
        raise ValueError("complete unique healthy model controls required")
    losses = {arm: mean_metrics([lookup[arm]["final_mean"]])["ce"] for arm in ARMS}
    gains = {arm: floor - value for arm, value in losses.items()}
    gap = losses["isolated"] - losses["mixed"]
    return {"all_numerics_pass": True, "current_token_empirical_floor": floor, "final_ce": losses,
            "gain_below_floor": gains, "per_arm_local_fitting_gate": {arm: gain + GATE_TOLERANCE >= 0.02 for arm, gain in gains.items()},
            "isolated_minus_mixed_ce": gap, "mixed_gain_at_least_002": gap + GATE_TOLERANCE >= 0.02,
            "requires_separate_successful_independent_recompute": True,
            "recipe_promoted": False, "paid_execution_authorized": False, "broader_study_authorized": False,
            "scope": "local training fit on the same512positions, possibly using position; not generalization or architecture impossibility"}


def execute(manifest, output, raw, *, write=False, recompute=False):
    if type(write) is not bool or type(recompute) is not bool or (write and recompute):
        raise ValueError("literal disjoint write/recompute modes required")
    output = Path(output)
    tokens, targets = batch_data(raw, manifest["binding"]["dataset"])
    models = initial_models(manifest["binding"])
    equal(list(models), list(ARMS), "exact model order and membership")
    vocab = models["isolated"]["parameters"]["EMBED"].shape[0]
    helper._tokens(tokens, vocab, ndim=2)
    helper._tokens(targets, vocab, ndim=2)
    if tokens.shape != targets.shape or tokens.shape[0] != len(WINDOWS):
        raise ValueError("complete aligned four-window batch required")
    floor = helper.empirical_bigram_floor(tokens, targets, vocab)
    for arm in ARMS:
        _finite_parameters(models[arm])
        equal(models[arm]["kind"], "gru" if arm == "prior_gru" else "tropical", "declared model family")
        if write:
            (output / arm).mkdir(exist_ok=False)
    prior_path = output / "frozen_prior.npz"
    prior_values = {"logits": models["prior_gru"]["prior_logits"]}
    if write:
        save(prior_path, prior_values)
    same(load(prior_path), prior_values, "complete frozen fitted GRU prior")
    initialization = {arm: {"kind": cell["kind"], "parameter_order": list(cell["parameters"]),
                           "parameter_shapes": {name: list(value.shape) for name, value in cell["parameters"].items()},
                           "parameters_sha256": tensor_hash(cell["parameters"]), "facts": cell.get("initialization", {})}
                      for arm, cell in models.items()}
    initialization["frozen_prior_sha256"] = tensor_hash(prior_values)
    if write:
        pilot.write_json(output / "initialization.json", initialization)
    equal(read(output / "initialization.json"), initialization, "complete initial model/source identity")
    initial = {arm: collect_scores(output / arm, "initial", cell, tokens, targets, True, write=write, recompute=recompute)
               for arm, cell in models.items()}
    maximum_probability_error = 0.0
    for index in WINDOWS:
        probabilities = [load(output / arm / f"score_initial_{index:03d}_1.npz")["logits"].softmax(-1) for arm in ARMS]
        for actual in probabilities[1:]:
            if actual.shape != probabilities[0].shape or not torch.allclose(actual, probabilities[0], atol=1e-12, rtol=1e-12):
                raise ValueError("fitted initial probabilities differ")
            maximum_probability_error = max(maximum_probability_error, float((actual - probabilities[0]).abs().max()))
    cells = [_cell(output / arm, arm, cell, tokens, targets, initial[arm], write=write, recompute=recompute)
             for arm, cell in models.items()]
    equal([len(cells), sum(cell["optimizer_commits"] for cell in cells), sum(cell["complete_saved_states"] for cell in cells)],
          [3, 384, 387], "complete fixed fitting budget")
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "cells": cells, "gates": gates(cells, floor), "current_token_empirical_floor": floor,
            "initial_probability_max_abs_error": maximum_probability_error,
            "initial_probabilities_equal": True, "declared_cells": 3, "completed_cells": 3,
            "full_batch_gradient_queries": 384, "optimizer_commits": 384, "complete_saved_states": 387,
            "score_window_readouts": 36, "independent_tropical_window_queries": 1024,
            "independent_gru_batch_queries": 128, "new_gradient_pair_exposures": 3 * UPDATES * tokens.numel(),
            "distinct_gradient_pairs": tokens.numel(), "new_fitting_pair_exposures": 0,
            "summed_inherited_fitting_pair_exposures": 393216, "summed_endpoint_history_pair_exposures": 393216 + 3 * UPDATES * tokens.numel(),
            "distinct_fitted_pairs": 131072, "new_native_training_calls": 0, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "fixed-budget eager FP64 training-fit comparison, jointly different from native recipes; no held-out/hardware claim"}


def expected_artifacts():
    files = {"manifest.json", "protocol.json", "declaration.md", "summary.json", "frozen_prior.npz", "initialization.json"}
    for arm in ARMS:
        files.add(arm + "/summary.json")
        files.update(f"{arm}/query_{step:06d}.npz" for step in range(UPDATES))
        files.update(f"{arm}/state_{step:06d}.{suffix}" for step in range(UPDATES + 1) for suffix in ("npz", "json"))
        files.update(f"{arm}/score_initial_{index:03d}_1.npz" for index in WINDOWS)
        files.update(f"{arm}/score_final_{index:03d}_{context}.npz" for index in WINDOWS for context in (0, 1))
    return files


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root), "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(item["bytes"] for item in artifacts),
            "check_scope": "complete saved-gradient Adam replay; separate --recompute reconstructs 1024 tropical window "
                           "forwards/VJPs and compares 256 summed full-batch gradients, compares 128 GRU full-batch "
                           "gradients/forwards, and checks 36 score-window readouts; no new optimizer trajectory"}


def _check(root, record, recompute, validation, raw):
    if type(recompute) is not bool:
        raise ValueError("literal recompute boolean required")
    root, record = Path(root).resolve(strict=True), parent._regular(record)
    manifest = read(root / "manifest.json")
    frozen(manifest, validation)
    equal(manifest["run_directory"], str(root), "owned raw directory")
    equal(read(root / "protocol.json"), recipe(), "saved recipe")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "saved declaration bytes")
    artifacts = prior.artifact_manifest(root)
    seal_bytes, public_bytes = (root / "artifact_manifest.json").read_bytes(), record.read_bytes()
    equal(read(root / "artifact_manifest.json"), artifacts, "complete artifact inventory")
    equal({item["file"] for item in artifacts}, expected_artifacts(), "exact complete artifact membership")
    summary = execute(manifest, root, raw, recompute=recompute)
    equal(read(root / "summary.json"), summary, "full reconstructed result")
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
            raise TimeoutError("fixed exact-fit read-only audit deadline exceeded")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        manifest = read(Path(root) / "manifest.json")
        binding, raw, validation = bind_inputs(_bound_config(manifest))
        equal(manifest["binding"], binding, "fresh canonical parent binding")
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
            raise TimeoutError("fixed exact-fit run deadline exceeded")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        binding, raw, validation = bind_inputs(config)
        equal(source_identity(), sources, "sources unchanged before declaration")
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
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-exact-fit-", dir=target.parent))
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
            parser.error("new run requires --prior-run, --prior-record, --prior-audit and --corpus")
        result = run(args)
    print(json.dumps({"complete": result["complete"], "gates": result["gates"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
