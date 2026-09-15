"""Guarded implementation parity against an independently audited trajectory.

No new training trajectory, corpus cohort, optimizer, or numerical recipe is
chosen here. Guarded queries must reproduce saved dense IEEE payloads exactly;
their recomputed identical gradients replay all complete saved Adam states.
Timings describe this parity workload, not a matched dense speed comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import audit_fpga_lm_exact_rate as parent  # noqa: E402
from manuscript.tools import fpga_lm_guarded_forward as guarded  # noqa: E402

rate, helper = parent.rate, parent.rate.helper
SCHEMA = "tikitorch.fpga_lm_guarded_rate_parity.v1"
DEADLINE_S = 3600
PLAN = REPO / "manuscript/research/PLAN.md"
# Actual successful receipt observed after the complete independent audit.
# No CLI/config override exists; tests replace this module constant only.
AUDIT_RECEIPT_SHA256 = "56d4388eb440ee77e7416e7fe60592cdf594c7b3cbf066dbc03bc2ff079a940b"
TIMING_GROUPS = ("guarded_forward", "manual_vjp_and_batch_sum", "adam_update",
                 "saved_payload_io", "exact_payload_comparison", "state_and_metadata_validation")
GUARD_KEYS = ("projection_fast_operations", "projection_dense_operations",
              "head_fast_operations", "head_dense_operations")


def source_identity():
    result = dict(parent.source_identity())
    for name in ("manuscript/tools/fpga_lm_guarded_forward.py", "tests/test_fpga_lm_guarded_forward.py",
                 "manuscript/tools/verify_fpga_lm_guarded_rate.py", "tests/test_fpga_lm_guarded_rate.py"):
        result[name] = rate.old.digest((REPO / name).read_bytes())
    return result


def byte_equal(actual, expected, label):
    """Full tensor identity, including signed zero, shape, dtype and key set."""
    rate.equal(set(actual), set(expected), label + " fields")
    for name in expected:
        left, right = actual[name], expected[name]
        if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
                or left.device.type != "cpu" or right.device.type != "cpu"
                or left.layout != torch.strided or right.layout != torch.strided
                or left.dtype != right.dtype or left.shape != right.shape
                or left.detach().contiguous().numpy().tobytes() != right.detach().contiguous().numpy().tobytes()):
            raise ValueError(label + " exact tensor bytes differ: " + name)


def _successful_audit(root, record, receipt_path):
    if (type(AUDIT_RECEIPT_SHA256) is not str or len(AUDIT_RECEIPT_SHA256) != 64
            or any(character not in "0123456789abcdef" for character in AUDIT_RECEIPT_SHA256)):
        raise ValueError("successful independent rate-audit receipt has not been pinned")
    receipt_path = rate.head._regular(receipt_path)
    rate.equal(rate.old.digest(receipt_path.read_bytes()), AUDIT_RECEIPT_SHA256, "actual successful audit pin")
    value = rate.read(receipt_path)
    expected = {"schema": parent.SCHEMA, "complete": True, "independent_recompute_completed": True,
                "rate_run_directory": str(root), "rate_public_record": str(record), "deadline_seconds": DEADLINE_S,
                "publication_postchecks_within_deadline": True, "independent_tropical_window_vjps": 512,
                "independent_full_batch_gradients": 128, "independent_score_readouts": 12,
                "independent_tropical_window_forwards": 524, "complete_saved_states_verified": 129,
                "optimizer_commits_replayed": 128, "rate_numerical_replay_complete": True,
                "source_input_raw_seal_public_bytes_unchanged": True,
                "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
                "new_optimizer_trajectory_executed": False, "heldout_corpus_evaluated": False,
                "physical_fpga_execution": False, "paid_execution_authorized": False,
                "recipe_promoted": False, "broader_study_authorized": False}
    for key, wanted in expected.items():
        rate.equal(value.get(key), wanted, "completed independent audit: " + key)
    for key in ("verification_elapsed_utc_seconds", "verification_elapsed_monotonic_seconds"):
        elapsed = value.get(key)
        if type(elapsed) is not float or not math.isfinite(elapsed) or not 0 <= elapsed < DEADLINE_S:
            raise ValueError("successful audit verification elapsed time invalid")
    audit_directory = Path(value["audit_directory"])
    if audit_directory.is_symlink() or not audit_directory.is_dir():
        raise ValueError("actual private independent-audit directory required")
    audit_directory = audit_directory.resolve(strict=True)
    rate.equal({path.name for path in audit_directory.iterdir()},
               {"audit_manifest.json", "declaration.md", "verification.json"}, "exact successful private audit inventory")
    for path in audit_directory.iterdir():
        rate.head._regular(path)
    manifest_path = rate.head._regular(audit_directory / "audit_manifest.json")
    rate.equal(rate.old.digest(manifest_path.read_bytes()), value["audit_manifest_sha256"], "actual audit manifest hash")
    manifest = rate.read(manifest_path)
    for key, wanted in {"schema": parent.SCHEMA, "complete": False,
                        "operation": "independent rate audit with pinned completed ancestor",
                        "run_directory": str(root), "record": str(record), "audit_directory": str(audit_directory),
                        "receipt": str(receipt_path), "source_hashes": parent.source_identity(),
                        "trust": parent.TRUST, "deadline_seconds": DEADLINE_S,
                        "started_at_utc": value["started_at_utc"]}.items():
        rate.equal(manifest.get(key), wanted, "bound audit manifest: " + key)
    command = ["python", "manuscript/tools/audit_fpga_lm_exact_rate.py", "--run", str(root),
               "--record", str(record), "--output", str(audit_directory), "--receipt", str(receipt_path)]
    rate.equal(manifest["command"], command, "completed audit exact command")
    declaration = rate.head._regular(audit_directory / "declaration.md")
    rate.equal(rate.old.digest(declaration.read_bytes()), manifest["declaration_sha256"], "audit saved declaration")
    verification = rate.head._regular(audit_directory / "verification.json")
    rate.equal(rate.read(verification), value["verification"], "actual independent audit verification facts")
    return value, [rate.old.file_identity(path) for path in (receipt_path, manifest_path, declaration, verification)]


def validate_inputs(run, record, audit_receipt):
    sources = source_identity()
    root, record = Path(run).resolve(strict=True), rate.head._regular(record)
    audited, identities = _successful_audit(root, record, audit_receipt)
    manifest, raw, validation, facts = parent.validate_inputs(root, record)
    rate.equal(audited["verification"], facts, "successful receipt matches all fresh byte/source verification")
    rate.equal(audited["historical_numerical_gates"], rate.read(record)["gates"], "unchanged original numerical gates")
    rate.equal(audited["recomputed_gradient_pair_observations"], rate.read(record)["new_gradient_pair_exposures"],
               "successful audit covers all original gradient observations")
    rate.equal(source_identity(), sources, "all guarded verifier sources unchanged through validation")
    for identity in identities:
        rate.equal(rate.old.file_identity(identity["path"]), identity, "successful audit files unchanged")
    return manifest, raw, validation, {"pinned_validation": facts, "successful_audit_files": identities,
                                      "successful_audit_receipt_sha256": AUDIT_RECEIPT_SHA256,
                                      "source_hashes": sources, "independent_autograd_audit_reused": True}


class Timings:
    def __init__(self):
        self.values = {name: {"wall_seconds": 0.0, "cpu_seconds": 0.0, "calls": 0} for name in TIMING_GROUPS}

    def measure(self, name, function):
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            return function()
        finally:
            row = self.values[name]
            row["wall_seconds"] += time.perf_counter() - wall
            row["cpu_seconds"] += time.process_time() - cpu
            row["calls"] += 1


def _guards(base):
    value = base["guard_diagnostics"]
    rate.equal(set(value), {"schema", "structure_certified", "whole_model_dense_fallback", *GUARD_KEYS}, "full guard metadata")
    rate.equal(value["schema"], guarded.SCHEMA, "guard schema")
    if type(value["structure_certified"]) is not bool or type(value["whole_model_dense_fallback"]) is not bool:
        raise ValueError("literal guard booleans required")
    if value["structure_certified"] == value["whole_model_dense_fallback"]:
        raise ValueError("guard structure/fallback flags disagree")
    if any(type(value[key]) is not int or value[key] < 0 for key in GUARD_KEYS):
        raise ValueError("literal nonnegative operation counts required")
    if value["whole_model_dense_fallback"] and (value["projection_fast_operations"] or value["head_fast_operations"]):
        raise ValueError("whole-model dense fallback cannot contain fast operations")
    geometry = base["geometry"]
    rate.equal(value["projection_fast_operations"] + value["projection_dense_operations"], geometry["N"] * geometry["T"], "all projection operations")
    rate.equal(value["head_fast_operations"] + value["head_dense_operations"], geometry["T"], "all head operations")
    return dict(value)


def query(directory, step, cell, tokens, targets, timings):
    """Guarded requery of one saved four-window batch; return recomputed G."""
    path = Path(directory) / f"query_{step:06d}.npz"
    arrays = timings.measure("saved_payload_io", lambda: rate.load(path))
    expected_keys = {"tokens", "targets"} | {"g_" + name for name in cell["parameters"]}
    expected_keys |= {f"window_{index}/" + key for index in rate.WINDOWS for key in helper.TRACE_KEYS}
    rate.equal(set(arrays), expected_keys, "complete original dense query")
    byte_equal({key: arrays[key] for key in ("tokens", "targets")}, {"tokens": tokens, "targets": targets}, "bound original pairs")
    gradients = {name: torch.zeros_like(value) for name, value in cell["parameters"].items()}
    rows = []
    for row, index in enumerate(rate.WINDOWS):
        base = timings.measure("guarded_forward", lambda: guarded.forward(cell["parameters"], tokens[row], context=True))
        trace = helper.trace(base)
        expected = {key: arrays[f"window_{index}/" + key] for key in helper.TRACE_KEYS}
        timings.measure("exact_payload_comparison", lambda: byte_equal(trace, expected, "guarded versus saved dense query trace"))
        def backward_and_sum():
            gradient = helper.gradient(cell["parameters"], tokens[row], targets[row], base, targets.numel())
            for name in gradients:
                gradients[name] = gradients[name] + gradient[name]
        timings.measure("manual_vjp_and_batch_sum", backward_and_sum)
        rows.append({"window": index, "trace_sha256": rate.tensor_hash(trace), "guards": _guards(base)})
    rate.parent._gradients(cell, gradients)
    timings.measure("exact_payload_comparison", lambda: byte_equal(gradients, {name: arrays["g_" + name] for name in gradients},
                                                                  "all summed guarded gradient IEEE bytes"))
    return gradients, {"step_before": step, "query_sha256": rate.old.digest(path.read_bytes()),
                       "gradient_sha256": rate.tensor_hash(gradients), "windows": rows}


def state(directory, step, cell, optimizer, timings):
    """Validate frozen metadata AND every actual Adam tensor's IEEE payload."""
    metadata = timings.measure("state_and_metadata_validation",
                              lambda: rate.state_record(directory, step, cell, optimizer, write=False))
    actual = {"parameter/" + name: parameter.detach() for name, parameter in cell["parameters"].items()}
    for name, parameter in cell["parameters"].items():
        for key, value in optimizer.state.get(parameter, {}).items():
            actual[key + "/" + name] = value
    saved = timings.measure("saved_payload_io", lambda: rate.load(Path(directory) / f"state_{step:06d}.npz"))
    timings.measure("exact_payload_comparison", lambda: byte_equal(actual, saved, "complete actual guarded Adam state"))
    return {"step": step, "state_file_sha256": metadata["sha256"],
            "full_state_tensor_sha256": rate.tensor_hash(actual), "parameters_sha256": metadata["parameters_sha256"],
            "optimizer_group": metadata["optimizer_group"]}


def scores(directory, label, cell, tokens, targets, context, timings):
    rows = []
    for row, index in enumerate(rate.WINDOWS):
        path = Path(directory) / f"score_{label}_{index:03d}_{int(context)}.npz"
        saved = timings.measure("saved_payload_io", lambda: rate.load(path))
        base = timings.measure("guarded_forward", lambda: guarded.forward(cell["parameters"], tokens[row], context=context))
        trace = helper.trace(base)
        timings.measure("exact_payload_comparison", lambda: byte_equal(trace, saved, "guarded versus dense score trace"))
        rows.append({"label": label, "context": context, "window": index,
                     "trace_sha256": rate.tensor_hash(trace), "file_sha256": rate.old.digest(path.read_bytes()),
                     "metrics": rate.parent.metrics(trace["logits"], targets[row]), "guards": _guards(base)})
    return rows


def replay(manifest, raw):
    """Recompute identical guarded G, apply that G, check all saved states."""
    wall, cpu = time.perf_counter(), time.process_time()
    timings = Timings()
    binding = manifest["binding"]
    directory = Path(manifest["run_directory"]) / rate.ARM
    tokens, targets = rate.batch_data(raw, binding["dataset"])
    cell = rate.initial_model(binding)
    rate.equal(cell["kind"], "tropical", "original mixed tropical architecture")
    rate.parent._finite_parameters(cell)
    helper._tokens(tokens, cell["parameters"]["EMBED"].shape[0], ndim=2)
    helper._tokens(targets, cell["parameters"]["EMBED"].shape[0], ndim=2)
    if tokens.shape != targets.shape or tokens.shape[0] != len(rate.WINDOWS):
        raise ValueError("complete original four-window batch required")
    optimizer = rate.make_optimizer(cell["parameters"].values())
    history = rate.read(directory / "summary.json")
    states = [state(directory, 0, cell, optimizer, timings)]
    # Strengthen the frozen numerical initial checks to exact tensor payloads.
    initial = timings.measure("saved_payload_io", lambda: rate.load(binding["initial_state"]))
    byte_equal({"parameter/" + name: value for name, value in cell["parameters"].items()}, initial,
               "actual inherited initial model and empty optimizer")
    score_rows = scores(directory, "initial", cell, tokens, targets, True, timings)
    rate.equal([row["metrics"] for row in score_rows], history["initial_scores"], "all original initial scores")
    rate.equal(rate.parent.mean_metrics([row["metrics"] for row in score_rows]),
               binding["controls"]["mixed"]["initial_mean"], "matched historical initial score")
    rows = []
    for step in range(rate.UPDATES):
        before = rate.tensor_hash(cell["parameters"])
        gradient, row = query(directory, step, cell, tokens, targets, timings)
        def apply_recomputed_gradient():
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in cell["parameters"].items():
                parameter.grad = gradient[name].detach().clone()
            optimizer.step()
        timings.measure("adam_update", apply_recomputed_gradient)
        states.append(state(directory, step + 1, cell, optimizer, timings))
        after = rate.tensor_hash(cell["parameters"])
        saved_row = history["query_learning_curve"][step]
        for key, wanted in {"step_before": step, "step_after": step + 1,
                            "before_parameters_sha256": before, "after_parameters_sha256": after,
                            "gradient_sha256": row["gradient_sha256"], "query_sha256": row["query_sha256"]}.items():
            rate.equal(saved_row[key], wanted, "original complete trajectory receipt: " + key)
        row.update(step_after=step + 1, before_parameters_sha256=before, after_parameters_sha256=after,
                   full_state_tensor_sha256=states[-1]["full_state_tensor_sha256"],
                   recomputed_identical_gradient_applied=True)
        rows.append(row)
        if (step + 1) % 16 == 0:
            print(json.dumps({"stage": "guarded_parity_replay", "step": step + 1,
                              "saved_dense_trace_gradient_and_full_state_byte_equal": True}), flush=True)
    final = scores(directory, "final", cell, tokens, targets, True, timings)
    disabled = scores(directory, "final", cell, tokens, targets, False, timings)
    rate.equal([row["metrics"] for row in final], history["final_scores"], "all original final scores")
    rate.equal([row["metrics"] for row in disabled], history["context_off_scores"], "all original disabled-context scores")
    score_rows.extend(final + disabled)
    rate.equal(len(history["query_learning_curve"]), rate.UPDATES, "complete original update membership")
    rate.equal(len(rows), 128, "all declared batch-gradient checks")
    rate.equal(len(states), 129, "all complete initial and updated states")
    rate.equal(len(score_rows), 12, "all declared score-window checks")
    all_windows = [window for row in rows for window in row["windows"]] + score_rows
    guards = {key: sum(window["guards"][key] for window in all_windows) for key in GUARD_KEYS}
    guards["whole_model_dense_fallback_forwards"] = sum(window["guards"]["whole_model_dense_fallback"] for window in all_windows)
    return {"schema": SCHEMA, "complete": True, "byte_exact_traces_gradients_and_complete_states": True,
            "guarded_window_forwards": len(all_windows), "manual_window_vjps": len(rows) * len(rate.WINDOWS),
            "full_batch_gradients_verified": len(rows), "optimizer_commits_replayed": len(rows),
            "complete_saved_states_verified": len(states), "score_readouts_verified": len(score_rows),
            "replayed_gradient_pair_observations": len(rows) * targets.numel(),
            "recomputed_identical_gradients_applied": True, "guard_totals": guards,
            "query_rows": rows, "state_rows": states, "score_rows": score_rows,
            "timings": timings.values, "replay_wall_seconds": time.perf_counter() - wall,
            "replay_cpu_seconds": time.process_time() - cpu,
            "timing_scope": "non-exhaustive named scopes within total guarded parity replay; state validation includes frozen metadata/I/O; unbucketed hashing, guards and Python orchestration remain in replay totals",
            "speedup_claim": False, "independent_autograd_rerun": False,
            "new_optimizer_trajectory_executed": False, "new_training_updates": 0,
            "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
            "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
            "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "implementation parity against a separately independently audited saved dense trajectory, not a new fit or independent-data performance experiment"}


def _unchanged(root, record, manifest, validation, facts):
    parent._unchanged(root, record, manifest, validation, facts["pinned_validation"])
    rate.equal(source_identity(), facts["source_hashes"], "all 67 guarded verifier sources unchanged")
    rate.equal(AUDIT_RECEIPT_SHA256, facts["successful_audit_receipt_sha256"], "successful audit trust pin unchanged")
    _, identities = _successful_audit(root, record, facts["successful_audit_files"][0]["path"])
    rate.equal(identities, facts["successful_audit_files"], "complete prior successful audit capsule unchanged")


def _new_path(path):
    candidate = rate.prior.capacity._new_external_path(path)
    # Historical audit capsules use audit_manifest.json, not manifest.json.
    # Reject before mkdir, so a failed invocation cannot add its failure files
    # inside the successful audit it is supposed to preserve.
    for ancestor in (candidate, *candidate.parents):
        manifest = ancestor / "audit_manifest.json"
        if manifest.is_file():
            value = rate.read(rate.head._regular(manifest))
            if isinstance(value, dict) and str(value.get("schema", "")).startswith("tikitorch.fpga_lm"):
                raise ValueError("new output cannot be nested inside a source-bound audit capsule")
    return candidate


def run(config):
    clock = parent.Deadline()
    output = target = staged = staging_directory = None
    published = owned_inode = None
    created = closed = False
    stage = "setup"
    try:
        clock.arm()
        output, target = (_new_path(path) for path in (config.output, config.receipt))
        if target == output or output in target.parents or target in output.parents:
            raise ValueError("new public receipt must be outside private parity output")
        if not target.parent.is_dir():
            raise FileNotFoundError("new receipt parent must already exist")
        root, record = Path(config.run).resolve(strict=True), rate.head._regular(config.record)
        audited_receipt = rate.head._regular(config.audit_receipt)
        sources, pin = source_identity(), AUDIT_RECEIPT_SHA256
        output.mkdir(parents=True, exist_ok=False)
        created = True
        declaration = PLAN.read_bytes()
        private = {"schema": SCHEMA, "complete": False, "operation": "guarded versus audited saved dense full trajectory parity",
                   "run_directory": str(root), "record": str(record), "successful_audit_receipt": str(audited_receipt),
                   "successful_audit_receipt_sha256": pin, "audit_directory": str(output), "receipt": str(target),
                   "source_hashes": sources, "started_at_utc": parent._utc(clock.started_utc),
                   "deadline_seconds": DEADLINE_S, "declaration_sha256": rate.old.digest(declaration),
                   "command": ["python", "manuscript/tools/verify_fpga_lm_guarded_rate.py", "--run", str(root),
                               "--record", str(record), "--audit-receipt", str(audited_receipt),
                               "--output", str(output), "--receipt", str(target)]}
        rate.pilot.write_json(output / "audit_manifest.json", private)
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        private_bytes = (output / "audit_manifest.json").read_bytes()
        torch.set_num_threads(1)
        stage = "input_validation"
        print(json.dumps({"stage": stage, "deadline_seconds": DEADLINE_S, "ancestor_numerical_replay": False}), flush=True)
        wall, cpu = time.perf_counter(), time.process_time()
        manifest, raw, validation, facts = validate_inputs(root, record, audited_receipt)
        validation_timing = {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu}
        rate.equal(facts["source_hashes"], sources, "no source relabeling during validation")
        rate.equal(facts["successful_audit_receipt_sha256"], pin, "no audit pin relabeling during validation")
        rate.pilot.write_json(output / "verification.json", facts)
        rate.equal(rate.read(output / "verification.json"), facts, "actual written verification facts")
        verification_bytes = (output / "verification.json").read_bytes()
        clock.check()
        stage = "guarded_parity_replay"
        print(json.dumps({"stage": stage, "guarded_window_forwards": 524, "manual_window_vjps": 512,
                          "independent_autograd_rerun": False}), flush=True)
        result = replay(manifest, raw)
        clock.check()
        rate.pilot.write_json(output / "parity.json", result)
        rate.equal(rate.read(output / "parity.json"), result, "actual written parity result")
        parity_bytes = (output / "parity.json").read_bytes()
        stage = "final_integrity"
        wall, cpu = time.perf_counter(), time.process_time()
        _unchanged(root, record, manifest, validation, facts)
        integrity_timing = {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu}
        rate.equal((output / "verification.json").read_bytes(), verification_bytes, "private verification unchanged before sealing")
        rate.equal((output / "parity.json").read_bytes(), parity_bytes, "private parity unchanged before sealing")
        rate.equal(rate.read(output / "verification.json"), facts, "sealed verification matches actual checks")
        rate.equal(rate.read(output / "parity.json"), result, "sealed parity matches actual replay")
        private_files = {name: rate.old.file_identity(output / name) for name in
                         ("audit_manifest.json", "declaration.md", "verification.json", "parity.json")}
        rate.equal((output / "audit_manifest.json").read_bytes(), private_bytes, "private declaration identity")
        rate.equal((output / "declaration.md").read_bytes(), declaration, "saved protocol bytes")
        elapsed_utc, elapsed_monotonic = clock.check()
        receipt = {"schema": SCHEMA, "complete": True, "guarded_parity_replay_completed": True,
                   "rate_run_directory": str(root), "rate_public_record": str(record), "audit_directory": str(output),
                   "audit_manifest_sha256": rate.old.digest(private_bytes), "private_files": private_files,
                   "source_hashes": sources, "verification": facts,
                   "successful_independent_audit_receipt_sha256": pin,
                   "started_at_utc": parent._utc(clock.started_utc),
                   "verification_completed_at_utc": parent._utc(clock.started_utc + elapsed_utc),
                   "verification_elapsed_utc_seconds": elapsed_utc,
                   "verification_elapsed_monotonic_seconds": elapsed_monotonic, "deadline_seconds": DEADLINE_S,
                   "publication_postchecks_within_deadline": True,
                   "timing_scope": "verification time precedes publication; final stdout records terminal completion; no matched dense benchmark",
                   "input_validation_timing": validation_timing, "prepublication_integrity_timing": integrity_timing,
                   "parity": result, "historical_numerical_gates": rate.read(record)["gates"],
                   "source_input_raw_seal_public_bytes_unchanged": True,
                   "independent_autograd_audit_reused": True, "independent_autograd_rerun": False,
                   "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
                   "new_optimizer_trajectory_executed": False, "heldout_corpus_evaluated": False,
                   "physical_fpga_execution": False, "paid_execution_authorized": False,
                   "recipe_promoted": False, "broader_study_authorized": False, "speedup_claim": False}
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-guarded-rate-parity-", dir=target.parent))
        staged = staging_directory / "receipt.json"
        rate.pilot.write_json(staged, receipt)
        published = rate.old.digest(staged.read_bytes())
        identity = staged.stat()
        owned_inode = identity.st_dev, identity.st_ino
        stage = "publication"
        clock.check()
        os.link(staged, target)
        _unchanged(root, record, manifest, validation, facts)
        rate.equal(rate.old.digest(target.read_bytes()), published, "exact published parity receipt")
        rate.equal({path.name for path in output.iterdir()}, set(private_files), "exact successful parity output membership")
        for name, descriptor in private_files.items():
            rate.equal(rate.old.file_identity(output / name), descriptor, "private parity payload unchanged")
        clock.check()
        staged.unlink()
        staging_directory.rmdir()
        clock.close()
        closed = True
        elapsed_utc, elapsed_monotonic = clock.check()
        print(json.dumps({"stage": "complete", "guarded_parity_replay_completed": True,
                          "completed_at_utc": parent._utc(clock.started_utc + elapsed_utc),
                          "elapsed_utc_seconds": elapsed_utc, "elapsed_monotonic_seconds": elapsed_monotonic,
                          "receipt_sha256": published}), flush=True)
        return receipt
    except BaseException as error:
        if target is not None and target.is_file() and not target.is_symlink() and owned_inode is not None:
            identity = target.stat()
            if (identity.st_dev, identity.st_ino) == owned_inode and rate.old.digest(target.read_bytes()) == published:
                rejected = output / "rejected_public.json"
                if rejected.exists():
                    raise RuntimeError("cannot overwrite retained rejected parity receipt") from error
                target.rename(rejected)
        if created and not (output / "failure.json").exists():
            elapsed_utc, elapsed_monotonic = clock.elapsed()
            rate.pilot.write_json(output / "failure.json", {"schema": SCHEMA, "complete": False,
                "guarded_parity_replay_completed": False, "stage": stage, "type": type(error).__name__, "error": str(error),
                "started_at_utc": parent._utc(clock.started_utc), "failure_observed_at_utc": parent._utc(time.time()),
                "elapsed_utc_seconds": elapsed_utc, "elapsed_monotonic_seconds": elapsed_monotonic,
                "deadline_seconds": DEADLINE_S, "new_training_updates": 0, "recipe_promoted": False})
        raise
    finally:
        try:
            if staged is not None and staged.exists():
                staged.unlink()
            if staging_directory is not None and staging_directory.exists():
                staging_directory.rmdir()
        finally:
            if not closed:
                clock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "record", "audit-receipt", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args)
    print(json.dumps({"complete": receipt["complete"], "receipt": str(args.receipt), "speedup_claim": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
