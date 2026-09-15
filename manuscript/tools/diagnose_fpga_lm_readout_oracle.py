"""Bounded frozen-feature readout sufficiency diagnostic, not an FPGA head.

Two predeclared convex corrections share one frozen backbone feature cache.
The separate audit reconstructs features and endpoint objectives/gradients;
it does not claim to replay the internal L-BFGS trajectory or formally bound
floating-point roundoff in the numerical suboptimality estimate.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_routing_contrasts as previous  # noqa: E402
from manuscript.tools import fpga_lm_readout_oracle as oracle  # noqa: E402

channel, core, parent = previous.channel, previous.core, previous.parent
helper, guard, old, prior = previous.helper, previous.guard, previous.old, previous.prior
read, equal, load = previous.read, previous.equal, previous.load
SCHEMA = "tikitorch.fpga_lm_readout_oracle.v1"
ARMS = ("bias", "affine")
TRAIN_WINDOWS = previous.TRAIN_WINDOWS
SCORE_TRAIN_WINDOWS = previous.SCORE_TRAIN_WINDOWS
DEV_WINDOWS = previous.DEV_WINDOWS
SEQUENCE = previous.SEQUENCE
PLAN = REPO / "manuscript/research/PLAN.md"
Deadline, Storage = previous.Deadline, previous.Storage
DEADLINE_S, STORAGE_CAP = 10800, 8 * (1 << 30)


def source_identity():
    values = dict(previous.source_identity())
    for name in ("manuscript/tools/fpga_lm_readout_oracle.py",
                 "manuscript/tools/diagnose_fpga_lm_readout_oracle.py",
                 "tests/test_fpga_lm_readout_oracle.py"):
        values[name] = old.digest((REPO / name).read_bytes())
    return values


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "frozen_checkpoint": "audited frozen-core order0 state256",
            "fit_windows": list(TRAIN_WINDOWS), "sampled_training_windows": list(SCORE_TRAIN_WINDOWS),
            "development_windows": list(DEV_WINDOWS), "sequence": SEQUENCE,
            "correction": "z=z0+h@W.T+b; bias arm omits W; zero correction reproduces actual z0",
            "features": "unscaled final head-input residuals, context ON; all original parameters frozen",
            "objective": "mean exact CE plus .001/2 times squared norm of ALL correction weights and biases",
            "lambda": 0.001, "numerical_gap_estimate_target": 1e-5,
            "numerical_gap_estimate": "squared full gradient Euclidean norm/(2*lambda); not a formal roundoff-certified bound",
            "max_iterations_per_arm": 200, "max_objective_calls_per_arm": 400,
            "solver_seconds_per_arm": 3600, "solver": "SciPy L-BFGS-B from zero, analytic full-data gradient",
            "solver_deadline_scope": "checked before/after each data chunk and accepted callback; in-flight native calls covered by global watchdog",
            "chunk_size": 2048, "torch_threads": 1, "blas_threads": 1,
            "cache_windows": len(TRAIN_WINDOWS) + len(DEV_WINDOWS),
            "backbone_gradient_queries": 0, "backbone_optimizer_steps": 0,
            "independent_audit": "all cached features/traces via fresh full-D selection; readout initial/final full-data autograd gradients and all fixed-cohort scores",
            "optimizer_trajectory_replayed": False, "shared_cache_refitted": False,
            "baseline": "reuse audited core, original mixed and GRU readouts; no comparator training",
            "material_progress_margin_nat": 0.01, "deadline_seconds": DEADLINE_S,
            "combined_output_cap_bytes": STORAGE_CAP, "free_reserve_bytes": 4 * (1 << 30),
            "heldout_test_evaluated": False, "reserved_development_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "richer-readout oracle with a larger optimization budget, not a parameter/exposure-matched architecture comparison"}


def runtime():
    import scipy
    import scipy.optimize._lbfgsb as lbfgsb_binary
    import scipy.optimize._lbfgsb_py as lbfgsb_python
    import threadpoolctl
    return {**parent.runtime(), "scipy": scipy.__version__, "threadpoolctl": threadpoolctl.__version__,
            "threadpools": threadpoolctl.threadpool_info(),
            "optimizer_implementation": {"python": old.file_identity(lbfgsb_python.__file__),
                                         "binary": old.file_identity(lbfgsb_binary.__file__)}}


@dataclass(frozen=True)
class Validation:
    binding: dict
    sources: dict
    channel_validation: object

    def __post_init__(self):
        for name in ("binding", "sources"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))


def bind_inputs(config):
    # This existing binder pins the actual core producer, seal, audit capsule,
    # safe endpoint ancestry and corpus, with no numerical trajectory replay.
    sources = source_identity()
    bound, raw, token = channel.bind_inputs(config)
    checkpoint = Path(bound["prior_run"]) / core.ARM / "state_000256.npz"
    binding = {"core": bound, "checkpoint": old.file_identity(checkpoint),
               "checkpoint_metadata": old.file_identity(checkpoint.with_suffix(".json"))}
    equal(source_identity(), sources, "no source change during input binding")
    validation = Validation(binding, sources, token)
    return binding, raw, validation


def frozen(binding, validation):
    if not isinstance(validation, Validation):
        raise ValueError("invocation-local readout validation required")
    equal(binding, validation.binding, "unchanged readout input binding")
    equal(source_identity(), validation.sources, "all frozen readout sources")
    channel.frozen({"schema": channel.SCHEMA, "recipe": channel.recipe(),
                    "source_hashes": channel.source_identity(), "binding": binding["core"]},
                   validation.channel_validation)
    for key in ("checkpoint", "checkpoint_metadata"):
        descriptor = binding[key]
        equal(old.file_identity(descriptor["path"]), descriptor, "actual frozen endpoint input bytes")


def frozen_model(binding):
    cell = core.initial_model(binding["core"]["core_binding"])
    parameters, _ = previous.endpoint._checkpoint(binding["checkpoint"]["path"], cell, "mixed", 256)
    cell["parameters"] = parameters
    for value in parameters.values():
        if value.requires_grad:
            raise ValueError("backbone must remain detached")
    return cell


def cache_path(root, cohort, index):
    if cohort not in ("train", "development"):
        raise ValueError("declared cache cohort required")
    allowed = TRAIN_WINDOWS if cohort == "train" else DEV_WINDOWS
    if type(index) is not int or index not in allowed:
        raise ValueError("only previously used training/development windows allowed")
    return Path(root) / "cache" / cohort / f"window_{index:04d}.npz"


def baseline_path(binding, cohort, index):
    base = Path(binding["core"]["prior_run"]) / core.ARM
    return (base / "train" / f"score_{index:04d}.npz" if cohort == "train"
            else base / f"score_final_{index:03d}_1.npz")


def near(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(label + ": exact fields required")
    for name, wanted in expected.items():
        value = actual[name]
        if value.shape != wanted.shape or value.dtype != wanted.dtype:
            raise ValueError(label + ": shape/dtype mismatch: " + name)
        if value.dtype == torch.float64:
            if not bool(torch.isfinite(value).all()) or not torch.allclose(value, wanted, atol=1e-9, rtol=1e-11):
                raise ValueError(label + ": independent numerical mismatch: " + name)
        else:
            guard.byte_equal({name: value}, {name: wanted}, label)


def collect_cache(root, binding, raw, cell, *, storage=None, independent=False, clock=None):
    """Write once, or validate immutable cache; sampled train scores reuse it."""
    writing = storage is not None
    if writing and independent:
        raise ValueError("independent audit cannot write a feature cache")
    result, rows, parity, guard_totals = {}, [], 0, [0] * 6
    before = helper.parameter_digest(cell["parameters"])
    for cohort, indices in (("train", TRAIN_WINDOWS), ("development", DEV_WINDOWS)):
        collected = {key: [] for key in ("h", "logits", "targets")}
        for position, index in enumerate(indices):
            clock.check()
            tokens, targets = parent.batch_data(raw, binding["core"]["dataset"], (index,),
                                                "train" if cohort == "train" else "validation")
            path = cache_path(root, cohort, index)
            if writing:
                extracted = oracle.extract_features(cell["parameters"], tokens[0])
                arrays = {**extracted["trace"], "h": extracted["h"], "tokens": tokens[0],
                          "targets": targets[0], "guard_counts": extracted["guard_counts"]}
                storage.arrays(path, arrays)
            arrays = load(path)
            equal(set(arrays), set(helper.TRACE_KEYS) | {"h", "tokens", "targets", "guard_counts"}, "complete cache inventory")
            guard.byte_equal({"tokens": arrays["tokens"], "targets": arrays["targets"]},
                             {"tokens": tokens[0], "targets": targets[0]}, "exact cache data alignment")
            h = arrays["h"]
            if h.dtype != torch.float64 or h.shape != (SEQUENCE, cell["parameters"]["EMBED"].shape[1]) or not bool(torch.isfinite(h).all()):
                raise ValueError("finite full-width FP64 cached features required")
            trace = {key: arrays[key] for key in helper.TRACE_KEYS}
            parent.parent._base(trace, cell["parameters"], tokens[0], True)
            if arrays["guard_counts"].shape != (1, 6):
                raise ValueError("complete actual guard row required")
            parent._validate_guard_rows(arrays["guard_counts"].repeat(parent.BATCH_SIZE, 1), cell, SEQUENCE)
            if independent:
                dense = oracle.extract_features(cell["parameters"], tokens[0], independent=True)
                near({**trace, "h": h}, {**dense["trace"], "h": dense["h"]}, "fresh full-D cache reconstruction")
                supplemental = oracle.extract_features(cell["parameters"], tokens[0])
                guard.byte_equal({**trace, "h": h, "guard_counts": arrays["guard_counts"]},
                                 {**supplemental["trace"], "h": supplemental["h"], "guard_counts": supplemental["guard_counts"]},
                                 "exact supplemental guarded cache")
            if cohort == "development" or index in SCORE_TRAIN_WINDOWS:
                inherited = load(baseline_path(binding, cohort, index))
                guard.byte_equal(trace, {key: inherited[key] for key in helper.TRACE_KEYS}, "exact frozen-core initial-function parity")
                parity += 1
            for key in range(6):
                guard_totals[key] += int(arrays["guard_counts"][0, key])
            for key in collected:
                collected[key].append(arrays[key].numpy())
            rows.append({"cohort": cohort, "window": index, "sha256": old.digest(path.read_bytes())})
            if (position + 1) % 64 == 0 or position + 1 == len(indices):
                print(json.dumps({"stage": "feature_cache", "cohort": cohort, "completed": position + 1,
                                  "independent": independent, "writing": writing}), flush=True)
        result[cohort] = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    equal(helper.parameter_digest(cell["parameters"]), before, "no backbone mutation during cache work")
    equal(parity, len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS), "all original endpoint parity checks")
    return result, {"files": rows, "initial_function_parity_checks": parity,
                    "guard_counts": guard_totals, "backbone_parameter_sha256": before}


def score(theta, cache, arm, *, independent=False):
    results = {}
    for cohort, indices, all_indices in (("train", SCORE_TRAIN_WINDOWS, TRAIN_WINDOWS),
                                         ("development", DEV_WINDOWS, DEV_WINDOWS)):
        data, rows = cache[cohort], []
        for index in indices:
            start = list(all_indices).index(index) * SEQUENCE
            region = slice(start, start + SEQUENCE)
            logits = oracle.predict(theta, data["h"][region], data["logits"][region], arm)
            if independent:
                weights, bias = oracle.unpack(theta, data["h"].shape[1], data["logits"].shape[1], arm)
                alternate = torch.from_numpy(data["logits"][region].copy())
                if bool((theta != 0).any()):
                    if weights is not None:
                        alternate = alternate + torch.from_numpy(data["h"][region]) @ torch.from_numpy(weights).T
                    alternate = alternate + torch.from_numpy(bias)
                near({"logits": torch.from_numpy(logits)}, {"logits": alternate}, "independent readout score logits")
                logits = alternate.numpy()
            rows.append({"window": index, **parent.parent.metrics(torch.from_numpy(logits), torch.from_numpy(data["targets"][region]))})
        results[cohort] = {"mean": parent._mean(rows), "per_window": rows}
    return results


def paired(after, before):
    result = {}
    for cohort in ("train", "development"):
        left, right = after[cohort], before[cohort]
        equal([r["window"] for r in left["per_window"]], [r["window"] for r in right["per_window"]], "paired score window ordering")
        result[cohort] = {"ce_difference": left["mean"]["ce"] - right["mean"]["ce"],
                          "accuracy_difference": left["mean"]["accuracy"] - right["mean"]["accuracy"],
                          "lower_ce_windows": sum(a["ce"] < b["ce"] for a, b in zip(left["per_window"], right["per_window"])),
                          "per_window": [{"window": a["window"], "ce_difference": a["ce"] - b["ce"],
                                          "accuracy_difference": a["accuracy"] - b["accuracy"]}
                                         for a, b in zip(left["per_window"], right["per_window"])]}
    return result


def _endpoint(theta, data, arm, independent=False):
    fn = oracle.independent_objective if independent else oracle.objective
    return fn(theta, data["h"], data["logits"], data["targets"], arm)


def validate_solver_record(solver, endpoints, arrays, data, arm):
    """Validate endpoint-bound scalar provenance, not unretained trial numerics.

    This performs no objective, gradient, model or optimizer evaluation. One
    counted evaluation may be incomplete only when the solver hit its deadline.
    """
    import math

    if type(arm) is not str or arm not in ARMS:
        raise ValueError("one declared solver arm required")
    positions, width, vocab = oracle.validate_data(data["h"], data["logits"], data["targets"])
    size = oracle.parameter_count(width, vocab, arm)
    if not isinstance(solver, dict) or set(solver) != {"summary", "history", "iteration_history"}:
        raise ValueError("complete solver scalar record required")
    summary, history, accepted = (solver[key] for key in ("summary", "history", "iteration_history"))
    if not isinstance(summary, dict) or type(history) is not list or type(accepted) is not list:
        raise ValueError("typed solver summary and complete histories required")
    scalar_keys = {"objective", "ce", "penalty", "gradient_norm_sq", "numerical_gap_estimate", "accuracy"}
    clock_keys = {"elapsed_utc_seconds", "elapsed_monotonic_seconds"}
    summary_keys = scalar_keys | clock_keys | {
        "schema", "arm", "positions", "features", "vocabulary", "parameter_count", "regularization",
        "options", "function_calls", "complete_function_evaluations", "iterations", "stopping_reason",
        "scipy_status", "scipy_message", "scipy_success", "numerical_target_met", "numerically_converged",
        "inconclusive", "formal_certification", "rounding_error_bound", "gap_target", "deadline_seconds",
        "zero_start", "returned_point", "restarts", "trial_history_scope"}
    equal(set(summary), summary_keys, "complete solver summary fields")
    expected = {"schema": oracle.SCHEMA, "arm": arm, "positions": positions, "features": width,
        "vocabulary": vocab, "parameter_count": size, "regularization": oracle.REGULARIZATION,
        "gap_target": oracle.GAP_TARGET, "deadline_seconds": oracle.DEADLINE_SECONDS,
        "zero_start": True, "restarts": 0, "formal_certification": False, "rounding_error_bound": None,
        "returned_point": "last evaluated accepted iterate",
        "trial_history_scope": "scalar provenance only; independent audit is endpoint-only",
        "options": {"maxiter": oracle.MAX_ITERATIONS, "maxfun": oracle.MAX_FUNCTION_CALLS,
                    "maxls": 20, "maxcor": 10, "ftol": 0.0,
                    "gtol": math.sqrt(2 * oracle.REGULARIZATION * oracle.GAP_TARGET / size)}}
    for key, value in expected.items():
        equal(summary[key], value, "fixed solver declaration: " + key)

    def finite(value, label):
        if type(value) is not float or not math.isfinite(value) or value < 0:
            raise ValueError(label + " requires a finite nonnegative literal float")
        return value

    def integer(value, lower, upper, label):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(label + " requires a bounded literal integer")
        return value

    def close(actual, expected_value, label):
        if not math.isfinite(expected_value) or not math.isclose(actual, expected_value, abs_tol=1e-9, rel_tol=1e-11):
            raise ValueError(label + " differs from the independently checked endpoint")

    def statistics(row, label):
        for key in scalar_keys:
            finite(row[key], label + ": " + key)
        if row["accuracy"] > 1:
            raise ValueError("accuracy must lie in [0,1]")
        equal(row["objective"], row["ce"] + row["penalty"], label + ": CE plus ridge")
        equal(row["numerical_gap_estimate"], row["gradient_norm_sq"] / (2 * oracle.REGULARIZATION),
              label + ": gradient-norm gap estimate")
        equal(row["numerical_target_met"], row["numerical_gap_estimate"] <= oracle.GAP_TARGET,
              label + ": actual numerical target flag")

    statistics(summary, "solver endpoint summary")
    for key in clock_keys:
        finite(summary[key], "solver " + key)
    calls = integer(summary["function_calls"], 1, oracle.MAX_FUNCTION_CALLS, "function-call count")
    complete = integer(summary["complete_function_evaluations"], 1, calls, "complete evaluation count")
    iterations = integer(summary["iterations"], 0, oracle.MAX_ITERATIONS, "accepted iteration count")
    equal(len(history), complete, "all completed evaluation rows retained")
    equal(len(accepted), iterations, "all accepted iteration rows retained")
    reasons = {"numerical_gap_target", "deadline", "function_call_limit", "iteration_limit",
               "scipy_stopped_above_numerical_target"}
    reason = summary["stopping_reason"]
    if type(reason) is not str or reason not in reasons:
        raise ValueError("declared terminal solver reason required")
    if calls - complete not in (0, 1) or (calls != complete and reason != "deadline"):
        raise ValueError("only deadline may interrupt one counted full evaluation")
    expired = max(summary[key] for key in clock_keys) >= oracle.DEADLINE_SECONDS
    equal(reason == "deadline", expired, "deadline reason matches actual elapsed clocks")
    if reason == "function_call_limit":
        equal(calls, oracle.MAX_FUNCTION_CALLS, "actual function-call cap reached")
    if reason == "iteration_limit":
        equal(iterations, oracle.MAX_ITERATIONS, "actual iteration cap reached")
    if reason == "numerical_gap_target" and not summary["numerical_target_met"]:
        raise ValueError("numerical stopping reason requires the endpoint target")
    if reason == "scipy_stopped_above_numerical_target" and summary["numerical_target_met"]:
        raise ValueError("above-target solver stop cannot claim the target")
    converged = reason == "numerical_gap_target" and summary["numerical_target_met"]
    equal(summary["numerically_converged"], converged, "actual numerical convergence status")
    equal(summary["inconclusive"], not converged, "bounded failure remains inconclusive")
    status, message, success = (summary[key] for key in ("scipy_status", "scipy_message", "scipy_success"))
    if type(success) is not bool:
        raise ValueError("literal SciPy success flag required")
    if status is None:
        if message is not None or success or reason == "scipy_stopped_above_numerical_target":
            raise ValueError("interrupted solver cannot fabricate a SciPy return")
    else:
        integer(status, 0, 2, "SciPy L-BFGS-B status")
        if type(message) is not str or not message or success != (status == 0):
            raise ValueError("consistent actual SciPy status/message required")

    row_keys = scalar_keys | clock_keys | {"numerical_target_met", "function_call", "theta_sha256", "gradient_sha256"}
    previous_mono = 0.0
    for ordinal, row in enumerate(history, 1):
        if not isinstance(row, dict):
            raise ValueError("complete function evaluation record required")
        equal(set(row), row_keys, "exact function evaluation provenance fields")
        equal(integer(row["function_call"], 1, calls, "evaluation ordinal"), ordinal, "ordered complete evaluations")
        statistics(row, "function evaluation")
        for key in clock_keys:
            finite(row[key], "evaluation " + key)
        if not previous_mono <= row["elapsed_monotonic_seconds"] <= summary["elapsed_monotonic_seconds"]:
            raise ValueError("ordered monotonic evaluation clocks required")
        previous_mono = row["elapsed_monotonic_seconds"]
        for key in ("theta_sha256", "gradient_sha256"):
            digest = row[key]
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("literal complete evaluation SHA-256 required")
    previous_call, previous_mono = 1, 0.0
    for ordinal, row in enumerate(accepted, 1):
        if not isinstance(row, dict):
            raise ValueError("complete accepted-iteration row required")
        equal(set(row), row_keys | {"iteration"}, "exact accepted-iteration provenance fields")
        equal(integer(row["iteration"], 1, iterations, "accepted ordinal"), ordinal, "ordered accepted iterations")
        call = integer(row["function_call"], previous_call + 1, complete, "accepted evaluation reference")
        reference = history[call - 1]
        for key in row_keys - clock_keys:
            equal(row[key], reference[key], "accepted iterate refers to its actual completed evaluation")
        for key in clock_keys:
            finite(row[key], "accepted " + key)
        mono = row["elapsed_monotonic_seconds"]
        if not max(previous_mono, reference["elapsed_monotonic_seconds"]) <= mono <= summary["elapsed_monotonic_seconds"]:
            raise ValueError("accepted-iteration monotonic clocks required")
        previous_call, previous_mono = call, mono

    equal(set(arrays), {"initial", "final"}, "both actual endpoint archives required")
    equal(set(endpoints), {"initial", "final"}, "both recomputed endpoint statistics required")
    final_row = accepted[-1] if accepted else history[0]
    for name, row in (("initial", history[0]), ("final", final_row)):
        archive = arrays[name]
        equal(set(archive), {"theta", "gradient"}, "complete endpoint coefficient/gradient archive")
        for key, value in archive.items():
            helper._finite(value, "archived " + key, detached=True)
            if value.shape != (size,):
                raise ValueError("complete flat endpoint coefficient/gradient required")
            equal(old.digest(value.contiguous().numpy().tobytes()), row[key + "_sha256"],
                  "accepted endpoint actually binds archived " + key)
        if name == "initial":
            guard.byte_equal({"theta": archive["theta"]}, {"theta": torch.zeros_like(archive["theta"])},
                             "literal positive-zero initial correction")
        theta, gradient = archive["theta"].numpy(), archive["gradient"].numpy()
        close(row["penalty"], oracle.REGULARIZATION / 2 * float(np.dot(theta, theta)), "actual endpoint ridge")
        close(row["gradient_norm_sq"], float(np.dot(gradient, gradient)), "actual endpoint gradient norm")
        endpoint = endpoints[name]
        equal(set(endpoint), scalar_keys | {"numerical_target_met", "positions", "regularization", "formal_certification"},
              "complete recomputed endpoint statistics")
        statistics(endpoint, "recomputed endpoint")
        for key, value in {"positions": positions, "regularization": oracle.REGULARIZATION,
                           "formal_certification": False}.items():
            equal(endpoint[key], value, "recomputed endpoint declaration")
        for key in scalar_keys:
            close(row[key], endpoint[key], "recomputed endpoint " + key)
    for key in scalar_keys | {"numerical_target_met"}:
        equal(summary[key], final_row[key], "summary is the last accepted evaluated endpoint")


def fit_arms(root, cache, *, storage=None, independent=False, clock=None):
    data, outcomes = cache["train"], {}
    for arm in ARMS:
        clock.check()
        directory = Path(root) / arm
        if storage is not None:
            fit = oracle.solve(data["h"], data["logits"], data["targets"], arm, checkpoint=clock.check)
            theta = fit["theta"]
            storage.arrays(directory / "initial.npz", {"theta": torch.from_numpy(fit["initial"]["theta"]),
                                                      "gradient": torch.from_numpy(fit["initial"]["gradient"])})
            storage.arrays(directory / "final.npz", {"theta": torch.from_numpy(theta), "gradient": torch.from_numpy(fit["gradient"])})
            storage.json(directory / "solver.json", {"summary": fit["summary"], "history": fit["history"],
                                                     "iteration_history": fit["iteration_history"]})
        solver = read(directory / "solver.json")
        endpoints, endpoint_arrays = {}, {}
        for name in ("initial", "final"):
            values = load(directory / (name + ".npz"))
            endpoint_arrays[name] = values
            equal(set(values), {"theta", "gradient"}, "complete readout endpoint state")
            theta, gradient = values["theta"].numpy(), values["gradient"].numpy()
            if name == "initial" and (theta != 0).any():
                raise ValueError("readout must start from exact zero correction")
            result = _endpoint(theta, data, arm, independent)
            near({"gradient": torch.from_numpy(gradient)}, {"gradient": torch.from_numpy(result["gradient"])}, "full-data readout endpoint gradient")
            endpoints[name] = {key: value for key, value in result.items() if key != "gradient"}
        validate_solver_record(solver, endpoints, endpoint_arrays, data, arm)
        initial_scores = score(load(directory / "initial.npz")["theta"].numpy(), cache, arm, independent=independent)
        after = score(theta, cache, arm, independent=independent)
        outcomes[arm] = {"solver": solver, "endpoints": endpoints, "initial_scores": initial_scores,
                         "after": after, "new_minus_frozen_core": paired(after, initial_scores),
                         "numerical_gap_estimate_target_met": endpoints["final"]["numerical_gap_estimate"] <= 1e-5,
                         "solver_complete_labeled_position_evaluations": solver["summary"]["complete_function_evaluations"] * len(data["targets"]),
                         "solver_attempted_labeled_position_evaluations_upper_bound": solver["summary"]["function_calls"] * len(data["targets"]),
                         "producer_endpoint_verification_full_gradients": 4,
                         "independent_endpoint_audit_full_gradients": 2,
                         "formal_optimization_certificate": False}
        print(json.dumps({"stage": "readout_complete", "arm": arm, "independent": independent,
                          "scores": {key: value["mean"] for key, value in after.items()},
                          "numerical_gap_estimate": endpoints["final"]["numerical_gap_estimate"]}), flush=True)
    return outcomes


def expected_artifacts():
    names = {"manifest.json", "declaration.md", "summary.json"}
    names.update(f"cache/{cohort}/window_{index:04d}.npz" for cohort, indices in
                 (("train", TRAIN_WINDOWS), ("development", DEV_WINDOWS)) for index in indices)
    names.update(f"{arm}/{name}" for arm in ARMS for name in ("initial.npz", "final.npz", "solver.json"))
    return names


def inventory(root):
    root = Path(root)
    artifacts = prior.artifact_manifest(root)
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact sealed readout payload membership")
    equal(read(root / "artifact_manifest.json"), {"schema": SCHEMA, "files": artifacts}, "all sealed readout payload hashes")
    equal({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()},
          {"cache", "cache/train", "cache/development", *ARMS}, "exact readout directories")
    return artifacts


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root), "artifact_files_verified": len(artifacts),
            "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "independent_audit_required": True}


def _config(binding):
    bound = binding["core"]
    return argparse.Namespace(prior_run=bound["prior_run"], prior_record=bound["prior_record"],
                              prior_audit=bound["prior_audit"], corpus=bound["dataset"]["path"])


def execute(root, binding, raw, *, storage=None, independent=False, clock=None):
    cell = frozen_model(binding)
    cache, cache_record = collect_cache(root, binding, raw, cell, storage=storage, independent=independent, clock=clock)
    outcomes = fit_arms(root, cache, storage=storage, independent=independent, clock=clock)
    controls = channel.controls(binding["core"])
    for outcome in outcomes.values():
        outcome["new_minus_gru"] = paired(outcome["after"], controls["prior_gru"])
        outcome["new_minus_original_mixed"] = paired(outcome["after"], controls["mixed"])
        for cohort in ("train", "development"):
            equal(outcome["initial_scores"][cohort]["mean"], controls["frozen_core"][cohort]["mean"], "original endpoint metric parity")
        progress = {cohort: outcome["new_minus_frozen_core"][cohort]["ce_difference"] <= -0.01
                    for cohort in ("train", "development")}
        outcome["material_progress"] = {"per_cohort": progress, "both_cohorts": all(progress.values()),
                                        "numerically_completed": outcome["solver"]["summary"]["numerically_converged"],
                                        "requires_successful_independent_audit": True,
                                        "scope": "descriptive next-review flag, not confirmation or promotion"}
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": source_identity(),
            "binding": binding, "cache": cache_record, "arms": outcomes,
            "affine_minus_bias": paired(outcomes["affine"]["after"], outcomes["bias"]["after"]),
            "controls": {name: {cohort: value[cohort]["mean"] for cohort in ("train", "development")}
                         for name, value in controls.items()},
            "backbone_gradient_queries": 0, "backbone_optimizer_steps": 0,
            "cache_windows": len(TRAIN_WINDOWS) + len(DEV_WINDOWS),
            "distinct_readout_training_pairs": len(TRAIN_WINDOWS) * SEQUENCE,
            "new_distinct_training_pairs": 0,
            "exposure_scope": "same distinct training pairs; every full readout objective query reuses all pairs; not matched to GRU optimization budget",
            "heldout_test_evaluated": False, "reserved_development_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False,
            "recipe_promoted": False, "formal_optimization_certificate": False,
            "scope": "frozen-endpoint residual readout adequacy only; bias control distinguishes recalibration; no backbone retraining or hardware claim"}


def _invoke(config, independent=False):
    from threadpoolctl import threadpool_limits
    clock, publication, output = Deadline(), None, None
    created, closed, cleanup_done = False, False, False
    stage = "setup"
    try:
        clock.arm()
        torch.set_num_threads(1)
        runtime()  # Load SciPy/BLAS before limiting every discovered thread pool.
        with threadpool_limits(limits=1):
            output, target = parent._new_outputs(config.audit_output, config.audit_record) if independent else parent._new_outputs(config.output, config.record)
            input_root = Path(config.output if independent else config.prior_run).resolve(strict=True)
            for new_path in (output, target):
                if new_path == input_root or input_root in new_path.parents or new_path in input_root.parents:
                    raise ValueError("new outputs must be disjoint from the immutable input run")
                for ancestor in new_path.parents:
                    if (ancestor / "artifact_manifest.json").exists() or (ancestor / "audit_manifest.json").exists():
                        raise ValueError("new outputs cannot be nested inside any sealed run or audit capsule")
            sources, declaration = source_identity(), PLAN.read_bytes()
            output.mkdir(parents=True, exist_ok=False)
            created = True
            storage = Storage(output, clock)
            root = Path(config.output).resolve(strict=True) if independent else output
            print(json.dumps({"stage": "input_validation", "independent": independent}), flush=True)
            stage = "input_validation"
            if independent:
                record = parent.rate.head._regular(config.record)
                manifest = read(root / "manifest.json")
                equal(manifest["schema"], SCHEMA, "exact producer manifest schema")
                equal(manifest["run_directory"], str(root), "actual producer root")
                equal(manifest["record"], str(record), "actual producer public path")
                equal(old.digest((root / "declaration.md").read_bytes()), manifest["declaration_sha256"], "producer saved declaration pin")
                equal((root / "declaration.md").read_bytes(), declaration, "same frozen declaration across producer and audit")
                equal(manifest["source_hashes"], sources, "same producer source closure")
                equal(manifest["recipe"], recipe(), "exact readout recipe")
                equal(manifest["runtime"], runtime(), "same readout runtime")
                artifacts = inventory(root)
                original = record.read_bytes()
                summary = read(root / "summary.json")
                originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json", "declaration.md")}
                equal(read(record), public_record(root, summary, artifacts), "producer publication exactly binds sealed data")
                storage.base_bytes = sum(row["bytes"] for row in artifacts) + (root / "artifact_manifest.json").stat().st_size + len(original)
                storage.used = storage.base_bytes
                binding, raw, validation = bind_inputs(_config(manifest["binding"]))
                equal(binding, manifest["binding"], "fresh canonical readout binding")
                private = {"schema": SCHEMA, "mode": "independent_endpoint_audit", "complete": False,
                           "producer_public_sha256": old.digest(original), "producer_seal_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
                           "run_directory": str(root), "record": str(record), "audit_directory": str(output),
                           "audit_record": str(target), "source_hashes": sources, "runtime": runtime(),
                           "declaration_sha256": old.digest(declaration), "started_at_utc": guard.parent._utc(clock.started_utc)}
                storage.json(output / "audit_manifest.json", private)
                private_bytes = (output / "audit_manifest.json").read_bytes()
            else:
                binding, raw, validation = bind_inputs(config)
                manifest = {"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "binding": binding,
                            "runtime": runtime(), "run_directory": str(root), "record": str(target),
                            "declaration_sha256": old.digest(declaration), "started_at_utc": guard.parent._utc(clock.started_utc)}
                storage.json(root / "manifest.json", manifest)
                for name in ("cache/train", "cache/development", *ARMS):
                    (root / name).mkdir(parents=True, exist_ok=False)
            equal(source_identity(), sources, "no source relabeling during invocation admission")
            equal(validation.sources, sources, "validation binds exact pre-admission sources")
            storage.reserve(output / "declaration.md", len(declaration))
            with (output / "declaration.md").open("xb") as stream:
                stream.write(declaration)
            storage.used += len(declaration)
            frozen(binding, validation)
            stage = "independent_endpoint_audit" if independent else "frozen_feature_fit"
            result = execute(root, binding, raw, storage=None if independent else storage, independent=independent, clock=clock)
            if independent:
                # Endpoint gradients are independently checked at declared tolerance;
                # all score/gradient-derived scalar fields are compared numerically.
                compare_results(result, summary)
                utc, monotonic = clock.check()
                published = {"schema": SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                             **{key: private[key] for key in ("producer_public_sha256", "producer_seal_sha256", "run_directory", "record", "audit_directory", "source_hashes", "runtime", "started_at_utc")},
                             "audit_manifest_sha256": old.digest((output / "audit_manifest.json").read_bytes()),
                             "verification_completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                             "verification_elapsed_utc_seconds": utc, "verification_elapsed_monotonic_seconds": monotonic,
                             "independent_dense_feature_windows": len(TRAIN_WINDOWS) + len(DEV_WINDOWS),
                             "supplemental_guarded_feature_windows": len(TRAIN_WINDOWS) + len(DEV_WINDOWS),
                             "independent_readout_full_gradients": 2 * len(ARMS),
                             "independent_initial_readout_score_windows": len(ARMS) * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
                             "independent_final_readout_score_windows": len(ARMS) * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
                             "initial_function_parity_checks": result["cache"]["initial_function_parity_checks"],
                             "optimizer_trajectory_replayed": False, "formal_optimization_certificate": False,
                             "new_training_updates": 0, "backbone_gradient_queries": 0,
                             "source_input_raw_seal_public_bytes_unchanged": True,
                             "heldout_test_evaluated": False, "reserved_development_evaluated": False,
                             "physical_fpga_execution": False, "paid_execution_authorized": False,
                             "recipe_promoted": False}
                storage.json(output / "verification.json", published)
                verification_bytes = (output / "verification.json").read_bytes()
            else:
                storage.json(root / "summary.json", result)
                artifacts = prior.artifact_manifest(root)
                equal({row["file"] for row in artifacts}, expected_artifacts(), "exact payloads before sealing")
                storage.json(root / "artifact_manifest.json", {"schema": SCHEMA, "files": artifacts})
                originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json", "declaration.md")}
                stage = "ordinary_endpoint_readback"
                readback = execute(root, binding, raw, clock=clock)
                equal(readback, result, "exact saved endpoint/cache/metric readback")
                published = public_record(root, result, inventory(root))
            storage.reconcile()
            stage = "publication"
            publication = parent._Publication(target, output, storage)
            publication.publish(published)
            frozen(binding, validation)
            equal(runtime(), manifest["runtime"], "runtime and optimizer implementation unchanged")
            equal(inventory(root), artifacts, "all sealed producer bytes unchanged")
            for name, payload in originals.items():
                equal((root / name).read_bytes(), payload, "exact original producer metadata bytes")
            if independent:
                equal(record.read_bytes(), original, "producer publication unchanged during audit")
                equal((output / "audit_manifest.json").read_bytes(), private_bytes, "exact original audit manifest bytes")
                equal((output / "verification.json").read_bytes(), verification_bytes, "exact original verification bytes")
                equal((output / "verification.json").read_bytes(), target.read_bytes(), "exact private/public audit receipt")
                equal({path.name for path in output.iterdir()}, {"audit_manifest.json", "declaration.md", "verification.json"}, "exact audit capsule")
            equal(PLAN.read_bytes(), declaration, "protocol remains frozen throughout invocation")
            publication.verify()
            publication.cleanup()
            cleanup_done = True
            clock.close()
            closed = True
            utc, monotonic = clock.check()
            print(json.dumps({"stage": "complete", "independent": independent,
                              "completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                              "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic,
                              "record_sha256": publication.digest}), flush=True)
            return published
    except BaseException as error:
        if publication is not None:
            publication.retract()
        if created:
            rejected = output / "rejected_public.json"
            payload = {"schema": SCHEMA, "complete": False, "stage": stage, "independent": independent,
                       "error": str(error)[:100000], "exception_type": type(error).__name__, "time_utc": time.time(),
                       "retained_rejected_public": old.file_identity(rejected) if rejected.is_file() and not rejected.is_symlink() else None,
                       "recipe_promoted": False, "paid_execution_authorized": False}
            try:
                with (output / "failure.json").open("x") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            except OSError:
                pass
        raise
    finally:
        try:
            if publication is not None and not cleanup_done:
                publication.cleanup()
        finally:
            if not closed:
                clock.close()


def compare_results(actual, expected):
    """Tolerance only for floating-point scalar recomputation; structure exact."""
    if type(actual) is not type(expected):
        raise ValueError("independent result type mismatch")
    if isinstance(actual, dict):
        equal(set(actual), set(expected), "independent result fields")
        for key in actual:
            compare_results(actual[key], expected[key])
    elif isinstance(actual, list):
        equal(len(actual), len(expected), "independent result lengths")
        for left, right in zip(actual, expected):
            compare_results(left, right)
    elif type(actual) is float:
        if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual, expected, atol=1e-9, rtol=1e-11):
            raise ValueError("independent result scalar mismatch")
    else:
        equal(actual, expected, "independent exact result")


def run(config):
    return _invoke(config, False)


def audit(config):
    return _invoke(config, True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    for name in ("output", "record"):
        parser.add_argument("--" + name, required=True)
    for name in ("prior-run", "prior-record", "prior-audit", "corpus", "audit-output", "audit-record"):
        parser.add_argument("--" + name)
    config = parser.parse_args(argv)
    required = ("audit_output", "audit_record") if config.audit else ("prior_run", "prior_record", "prior_audit", "corpus")
    if any(getattr(config, name) is None for name in required):
        parser.error("all input/output paths for the selected mode are required")
    audit(config) if config.audit else run(config)


if __name__ == "__main__":
    main()
