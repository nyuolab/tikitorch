"""Pinned-history audit tests over synthetic arrays, never the real corpus."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_fpga_lm_exact_fit_replay import tiny_models


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def auditor():
    return importlib.import_module("manuscript.tools.audit_fpga_lm_exact_rate")


def make_parent_receipt(rate, root, record):
    artifacts = rate.prior.artifact_manifest(root)
    sources = rate.parent.source_identity()
    source_name = "manuscript/tools/diagnose_fpga_lm_exact_fit.py"
    gates = rate.read(record)["gates"]
    return {"schema": "tikitorch.fpga_lm_exact_fit.cli_audit_receipt.v1", "complete": True,
        "independent_recompute": True, "exit_code": 0, "deadline_seconds": 3600,
        "parent_record_sha256": rate.old.digest(record.read_bytes()),
        "run_manifest_sha256": rate.old.digest((root / "manifest.json").read_bytes()),
        "artifact_manifest_sha256": rate.old.digest((root / "artifact_manifest.json").read_bytes()),
        "runner_sha256": sources[source_name], "source_files_verified": len(sources),
        "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
        "independent_full_batch_gradient_queries": 384, "independent_tropical_window_vjps": 1024,
        "independent_tropical_full_batch_gradients": 256, "independent_gru_full_batch_gradients": 128,
        "independent_score_readouts": 36, "complete_saved_states_verified": 387, "optimizer_commits_replayed": 384,
        "complete_parent_ancestry_verified": True, "source_input_raw_seal_public_bytes_unchanged": True,
        "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
        "new_optimizer_trajectory_executed": False, "heldout_corpus_evaluated": False,
        "physical_fpga_execution": False, "paid_execution_authorized": False,
        "recipe_promoted": False, "broader_study_authorized": False, **gates,
        "ancestor_raw_trees_verified": 0, "external_ancestor_file_identities_verified": 1,
        "independent_tropical_window_forwards": 1048, "independent_gru_batch_forwards": 131,
        "independent_forward_window_equivalents": 1572, "recomputed_gradient_pair_observations": 196608,
        "command": ["python", source_name, "--check", "--recompute", "--output", str(root), "--record", str(record)],
        "started_at_utc": "2026-09-11 12:00:00 UTC", "completed_at_utc": "2026-09-11 12:30:00 UTC"}


def build_tiny_rate(auditor, temporary, patch):
    """Real tiny numerical payload; only the input/model-size factories differ."""
    rate = auditor.rate
    parent_root, raw_root = temporary / "parent", temporary / "rate"
    (parent_root / "mixed").mkdir(parents=True)
    raw_root.mkdir()
    corpus, parent_record, parent_audit, record = [temporary / name for name in
        ("synthetic.txt", "parent.json", "parent-audit.json", "rate.json")]
    corpus.write_bytes(b"Synthetic four-window unit fixture; never corpus evidence.\n")
    raw = corpus.read_bytes()
    dataset = {"path": str(corpus), "synthetic": True}
    tokens = torch.tensor([[0, 1, 0, 2], [1, 3, 1, 0], [2, 0, 2, 3], [3, 2, 3, 1]], dtype=torch.int64)
    targets = torch.tensor([[1, 0, 2, 1], [3, 1, 0, 3], [0, 2, 3, 0], [2, 3, 1, 2]], dtype=torch.int64)
    model = tiny_models(rate.helper)["mixed"]
    initial_state = parent_root / "mixed/state_000000.npz"
    rate.save(initial_state, {"parameter/" + name: value for name, value in model["parameters"].items()})
    initial_scores, scores = [], []
    for index in range(4):
        trace = rate.helper.trace(rate.helper.forward(model["parameters"], tokens[index]))
        path = parent_root / "mixed" / f"score_initial_{index:03d}_1.npz"
        rate.save(path, trace)
        initial_scores.append(rate.old.file_identity(path))
        scores.append(rate.parent.metrics(trace["logits"], targets[index]))
    initial_mean = rate.parent.mean_metrics(scores)
    prior_binding = {"dataset": dataset, "initial_state": str(initial_state),
                     "inputs": [rate.old.file_identity(corpus)],
                     "closure": {"roots": {}, "files": [rate.old.file_identity(corpus)]}}
    write_json(parent_root / "manifest.json", {"schema": rate.parent.SCHEMA,
               "run_directory": str(parent_root), "recipe": rate.parent.recipe(),
               "binding": prior_binding, "source_hashes": rate.parent.source_identity()})
    cells = [{"arm": arm, "complete": True, "optimizer_commits": 128,
              "initial_mean": initial_mean, "final_mean": {"ce": .2 if arm == "prior_gru" else 1.0, "accuracy": .5}}
             for arm in rate.parent.ARMS]
    parent_summary = {"schema": rate.parent.SCHEMA, "complete": True,
               "cells": cells, "source_hashes": rate.parent.source_identity(),
               "gates": {"per_arm_local_fitting_gate": {arm: True for arm in rate.parent.ARMS},
                         "mixed_gain_at_least_002": True}}
    write_json(parent_root / "summary.json", parent_summary)
    parent_artifacts = rate.prior.artifact_manifest(parent_root)
    write_json(parent_root / "artifact_manifest.json", parent_artifacts)
    write_json(parent_record, rate.parent.public_record(parent_root, parent_summary, parent_artifacts))
    # This is a synthetic pinned ancestor, not a claim of re-running the old
    # three-arm experiment. Its exact smaller declared inventory is verified.
    patch.setattr(rate.parent, "expected_artifacts", lambda: {row["file"] for row in parent_artifacts})
    write_json(parent_audit, make_parent_receipt(rate, parent_root, parent_record))
    binding = {"prior_run": str(parent_root), "prior_record": str(parent_record), "prior_audit": str(parent_audit),
               "prior_binding": prior_binding, "dataset": dataset, "initial_state": str(initial_state),
               "initial_state_identity": rate.old.file_identity(initial_state), "initial_scores": initial_scores,
               "controls": rate._controls(parent_record),
               "closure": rate.head.ancestry_snapshot(parent_root, parent_record, parent_audit)}
    patch.setattr(rate, "initial_model", lambda binding: tiny_models(rate.helper)["mixed"])
    patch.setattr(rate, "batch_data", lambda raw, dataset: (tokens.clone(), targets.clone()))

    def corpus_plan(path, *args):
        assert Path(path) == corpus
        return corpus.read_bytes(), dict(dataset)

    patch.setattr(rate.pilot, "corpus_plan", corpus_plan)
    declaration = b"Synthetic pinned-audit test, not a scientific declaration.\n"
    manifest = {"schema": rate.SCHEMA, "run_directory": str(raw_root), "recipe": rate.recipe(),
                "source_hashes": rate.source_identity(), "binding": binding,
                "plan_sha256": rate.old.digest(declaration)}
    write_json(raw_root / "manifest.json", manifest)
    write_json(raw_root / "protocol.json", rate.recipe())
    (raw_root / "declaration.md").write_bytes(declaration)
    summary = rate.execute(manifest, raw_root, raw, write=True)
    write_json(raw_root / "summary.json", summary)
    artifacts = rate.prior.artifact_manifest(raw_root)
    write_json(raw_root / "artifact_manifest.json", artifacts)
    write_json(record, rate.public_record(raw_root, summary, artifacts))
    return SimpleNamespace(rate=rate, root=raw_root, record=record, parent_root=parent_root,
                           parent_record=parent_record, parent_audit=parent_audit,
                           corpus=corpus, raw=raw, dataset=dataset, binding=binding, manifest=manifest,
                           artifacts=artifacts, temporary=temporary, tokens=tokens, targets=targets)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    auditor = importlib.import_module("manuscript.tools.audit_fpga_lm_exact_rate")
    patch = pytest.MonkeyPatch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        value = build_tiny_rate(auditor, tmp_path_factory.mktemp("pinned-rate-audit"), patch)
        pins = dict(auditor.TRUST)
        for key, path in {
            "rate_public_sha256": value.record, "rate_manifest_sha256": value.root / "manifest.json",
            "rate_seal_sha256": value.root / "artifact_manifest.json", "parent_audit_sha256": value.parent_audit,
            "parent_public_sha256": value.parent_record, "parent_manifest_sha256": value.parent_root / "manifest.json",
            "parent_seal_sha256": value.parent_root / "artifact_manifest.json",
        }.items():
            pins[key] = value.rate.old.digest(path.read_bytes())
        ancestors = value.rate.prior.artifact_manifest(value.parent_root)
        pins.update(rate_source_count=len(value.rate.source_identity()), rate_artifact_count=len(value.artifacts),
                    rate_artifact_bytes=sum(row["bytes"] for row in value.artifacts),
                    parent_source_count=len(value.rate.parent.source_identity()), parent_artifact_count=len(ancestors),
                    parent_artifact_bytes=sum(row["bytes"] for row in ancestors),
                    ancestor_tree_count=len(value.binding["closure"]["roots"]),
                    ancestor_external_file_count=len(value.binding["closure"]["files"]))
        patch.setattr(auditor, "TRUST", pins)

        def forbidden(*args, **kwargs):
            pytest.fail("historical numerical replay or timer-owning old checker was called")

        for module, names in ((value.rate, ("bind_inputs", "check")),
                              (value.rate.parent, ("bind_inputs", "_check", "check")),
                              (value.rate.head, ("bind_inputs", "_check", "check"))):
            for name in names:
                patch.setattr(module, name, forbidden)
        calls = {"gradient": 0, "independent_gradient": 0, "forward": 0, "independent_forward": 0}
        forward, gradient = value.rate.helper.forward, value.rate.helper.gradient

        def traced_forward(*args, **kwargs):
            calls["independent_forward" if kwargs.get("independent") else "forward"] += 1
            return forward(*args, **kwargs)

        def traced_gradient(*args, **kwargs):
            calls["independent_gradient" if kwargs.get("independent") else "gradient"] += 1
            return gradient(*args, **kwargs)

        patch.setattr(value.rate.helper, "forward", traced_forward)
        patch.setattr(value.rate.helper, "gradient", traced_gradient)
        value.auditor, value.calls, value.pins = auditor, calls, pins
        yield value
    finally:
        patch.undo()
        torch.set_num_threads(previous_threads)


def test_complete_pinned_inputs_verified_without_any_parent_or_new_model_query(synthetic):
    value = synthetic
    before = dict(value.calls)
    manifest, raw, validation, facts = value.auditor.validate_inputs(value.root, value.record)
    assert manifest == value.manifest and raw == value.raw
    assert validation.binding == value.binding
    assert value.calls == before
    assert facts["rate_artifact_files_verified"] == 403
    assert facts["completed_predecessor_audit_reused"] is True
    assert facts["predecessor_numerical_replay_executed"] is False
    assert facts["rate_source_files_verified"] == 61


@pytest.mark.parametrize("target", ["rate_public", "rate_manifest", "rate_seal", "parent_public",
                                    "parent_manifest", "parent_seal", "parent_audit"])
def test_any_changed_pinned_record_bytes_are_rejected(synthetic, target):
    value = synthetic
    paths = {"rate_public": value.record, "rate_manifest": value.root / "manifest.json",
             "rate_seal": value.root / "artifact_manifest.json", "parent_public": value.parent_record,
             "parent_manifest": value.parent_root / "manifest.json", "parent_seal": value.parent_root / "artifact_manifest.json",
             "parent_audit": value.parent_audit}
    path = paths[target]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError):
            value.auditor.validate_inputs(value.root, value.record)
    finally:
        path.write_bytes(original)


def test_resealed_ancestor_mutation_does_not_create_new_trusted_history(synthetic):
    value, rate = synthetic, synthetic.rate
    path = value.parent_root / "mixed/score_initial_000_1.npz"
    originals = {file: file.read_bytes() for file in
                 (path, value.parent_root / "artifact_manifest.json", value.parent_record, value.parent_audit)}
    try:
        path.write_bytes(path.read_bytes() + b"changed prior payload")
        artifacts = rate.prior.artifact_manifest(value.parent_root)
        write_json(value.parent_root / "artifact_manifest.json", artifacts)
        write_json(value.parent_record, rate.parent.public_record(value.parent_root,
                   rate.read(value.parent_root / "summary.json"), artifacts))
        write_json(value.parent_audit, make_parent_receipt(rate, value.parent_root, value.parent_record))
        with pytest.raises(ValueError):
            value.auditor.validate_inputs(value.root, value.record)
    finally:
        for file, payload in originals.items():
            file.write_bytes(payload)


def test_unlisted_ancestor_member_is_rejected_even_with_unchanged_pins(synthetic):
    path = synthetic.parent_root / "extra.bin"
    try:
        path.write_bytes(b"unlisted ancestor payload")
        with pytest.raises(ValueError):
            synthetic.auditor.validate_inputs(synthetic.root, synthetic.record)
    finally:
        path.unlink()


def test_changed_external_corpus_is_rejected_without_parsing_or_model_execution(synthetic):
    original, before = synthetic.corpus.read_bytes(), dict(synthetic.calls)
    try:
        synthetic.corpus.write_bytes(original + b"changed")
        with pytest.raises(ValueError):
            synthetic.auditor.validate_inputs(synthetic.root, synthetic.record)
        assert synthetic.calls == before
    finally:
        synthetic.corpus.write_bytes(original)


def test_current_source_drift_is_not_waived_by_pinned_old_audit(synthetic, monkeypatch):
    original = synthetic.rate.source_identity()
    key = next(iter(original))
    monkeypatch.setattr(synthetic.rate, "source_identity", lambda: {**original, key: "f" * 64})
    with pytest.raises(ValueError):
        synthetic.auditor.validate_inputs(synthetic.root, synthetic.record)


def test_resealed_rate_gradient_payload_is_not_a_new_trusted_producer(synthetic):
    value, rate = synthetic, synthetic.rate
    path = value.root / rate.ARM / "query_000000.npz"
    originals = {file: file.read_bytes() for file in (path, value.root / "artifact_manifest.json", value.record)}
    arrays = rate.load(path)
    arrays["g_EMBED"].reshape(-1)[0] += .125
    try:
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{key: item.numpy() for key, item in arrays.items()})
        artifacts = rate.prior.artifact_manifest(value.root)
        write_json(value.root / "artifact_manifest.json", artifacts)
        write_json(value.record, rate.public_record(value.root, rate.read(value.root / "summary.json"), artifacts))
        with pytest.raises(ValueError):
            value.auditor.validate_inputs(value.root, value.record)
    finally:
        for file, payload in originals.items():
            file.write_bytes(payload)


@pytest.mark.parametrize("utc_delta,monotonic_delta,error", [
    (3599.0, 3599.0, None), (3600.0, 1.0, TimeoutError),
    (1.0, 3600.0, TimeoutError), (-.1, 1.0, RuntimeError), (1.0, -.1, RuntimeError),
])
def test_both_independent_clocks_bound_the_audit(auditor, monkeypatch, utc_delta, monotonic_delta, error):
    values = SimpleNamespace(utc=10000.0, monotonic=100.0)
    monkeypatch.setattr(auditor, "time", SimpleNamespace(time=lambda: values.utc, monotonic=lambda: values.monotonic))
    deadline = auditor.Deadline()
    values.utc += utc_delta
    values.monotonic += monotonic_delta
    if error:
        with pytest.raises(error):
            deadline.check()
    else:
        assert deadline.check() == (utc_delta, monotonic_delta)


def test_deadline_restores_handler_and_stops_owned_watcher(auditor):
    handler = signal.getsignal(signal.SIGALRM)
    deadline = auditor.Deadline()
    try:
        deadline.arm()
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= auditor.DEADLINE_S
        assert deadline.watcher.is_alive()
    finally:
        deadline.close()
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert not deadline.watcher.is_alive()


def test_full_new_rate_replay_and_receipt_without_parent_numerical_replay(synthetic, monkeypatch):
    value, auditor, rate = synthetic, synthetic.auditor, synthetic.rate
    before = {"raw": tree_bytes(value.root), "parent": tree_bytes(value.parent_root),
              "record": value.record.read_bytes(), "parent_record": value.parent_record.read_bytes(),
              "parent_audit": value.parent_audit.read_bytes(), "corpus": value.corpus.read_bytes()}
    calls_before = dict(value.calls)
    recorded_steps = []
    state_record = rate.state_record

    def counted_state(directory, step, *args, **kwargs):
        recorded_steps.append(step)
        return state_record(directory, step, *args, **kwargs)

    monkeypatch.setattr(rate, "state_record", counted_state)
    config = SimpleNamespace(run=value.root, record=value.record, output=value.temporary / "audit",
                             receipt=value.temporary / "audit.json")
    receipt = auditor.run(config)
    assert receipt["complete"] is True and receipt["independent_recompute_completed"] is True
    assert receipt["complete_saved_states_verified"] == 129 and receipt["optimizer_commits_replayed"] == 128
    assert receipt["independent_tropical_window_vjps"] == 512
    assert receipt["independent_full_batch_gradients"] == 128
    assert receipt["independent_tropical_window_forwards"] == 524
    assert receipt["independent_score_readouts"] == 12
    assert recorded_steps == list(range(129))
    assert value.calls["independent_forward"] - calls_before["independent_forward"] == 524
    assert value.calls["independent_gradient"] - calls_before["independent_gradient"] == 512
    assert value.calls["forward"] == calls_before["forward"]
    assert value.calls["gradient"] == calls_before["gradient"]
    assert receipt["verification"]["predecessor_numerical_replay_executed"] is False
    assert receipt["source_input_raw_seal_public_bytes_unchanged"] is True
    assert receipt["new_training_updates"] == receipt["new_native_training_calls"] == 0
    assert receipt["new_optimizer_trajectory_executed"] is False
    assert receipt["recipe_promoted"] is False and receipt["paid_execution_authorized"] is False
    assert receipt["recomputed_gradient_pair_observations"] == 128 * value.tokens.numel()
    assert rate.read(config.receipt) == receipt
    assert receipt["publication_postchecks_within_deadline"] is True
    assert 0 <= receipt["verification_elapsed_utc_seconds"] < auditor.DEADLINE_S
    assert 0 <= receipt["verification_elapsed_monotonic_seconds"] < auditor.DEADLINE_S
    assert not (config.output / "failure.json").exists()
    assert before == {"raw": tree_bytes(value.root), "parent": tree_bytes(value.parent_root),
                      "record": value.record.read_bytes(), "parent_record": value.parent_record.read_bytes(),
                      "parent_audit": value.parent_audit.read_bytes(), "corpus": value.corpus.read_bytes()}
    # Independent algebra on saved arrays, not a second forward or trajectory.
    directory = value.root / rate.ARM
    previous = rate.load(directory / "state_000000.npz")
    names = [key.removeprefix("parameter/") for key in previous]
    for step in range(1, 129):
        current = rate.load(directory / f"state_{step:06d}.npz")
        query = rate.load(directory / f"query_{step - 1:06d}.npz")
        for name in names:
            parameter = previous["parameter/" + name]
            gradient = query["g_" + name]
            first = .9 * previous.get("exp_avg/" + name, torch.zeros_like(parameter)) + .1 * gradient
            second = .999 * previous.get("exp_avg_sq/" + name, torch.zeros_like(parameter)) + .001 * gradient.square()
            expected = parameter - (.01 / (1 - .9 ** step)) * first / (
                second.sqrt() / ((1 - .999 ** step) ** .5) + 1e-8)
            for prefix, wanted in (("parameter/", expected), ("exp_avg/", first), ("exp_avg_sq/", second)):
                torch.testing.assert_close(current[prefix + name], wanted, atol=1e-12, rtol=1e-11)
            assert float(current["step/" + name]) == step
        previous = current


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt, TimeoutError])
def test_numerical_replay_failure_is_retained_never_published(synthetic, monkeypatch, tmp_path, failure):
    value, auditor = synthetic, synthetic.auditor
    config = SimpleNamespace(run=value.root, record=value.record, output=tmp_path / "failed",
                             receipt=tmp_path / "receipt.json")

    def fail(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        raise failure("injected numerical audit failure")

    monkeypatch.setattr(value.rate, "_check", fail)
    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(failure):
        auditor.run(config)
    record = value.rate.read(config.output / "failure.json")
    assert record["complete"] is False and record["independent_recompute_completed"] is False
    assert record["stage"] == "independent_rate_replay"
    assert not config.receipt.exists()
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("foreign", [False, True])
def test_postpublication_failure_retracts_only_owned_receipt(synthetic, monkeypatch, tmp_path, foreign):
    value, auditor = synthetic, synthetic.auditor
    config = SimpleNamespace(run=value.root, record=value.record, output=tmp_path / "audit",
                             receipt=tmp_path / "receipt.json")
    # This fixture tests publication only. The real complete replay is tested
    # above; this injected return must never be used as scientific evidence.
    monkeypatch.setattr(value.rate, "_check", lambda *args: value.rate.read(value.record))
    unchanged = auditor._unchanged
    checks = []

    def fail_after_link(*args):
        unchanged(*args)
        checks.append(True)
        if len(checks) == 2:
            assert config.receipt.is_file()
            if foreign:
                payload = config.receipt.read_bytes()
                config.receipt.unlink()
                config.receipt.write_bytes(payload)
            raise ValueError("injected publication postcheck failure")

    monkeypatch.setattr(auditor, "_unchanged", fail_after_link)
    with pytest.raises(ValueError, match="publication postcheck"):
        auditor.run(config)
    assert (config.output / "failure.json").is_file()
    assert config.receipt.exists() is foreign
    assert (config.output / "rejected_public.json").exists() is (not foreign)
    assert not list(tmp_path.glob(".tt-pinned-rate-audit-*"))
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("which", ["output", "receipt", "nested"])
def test_exclusive_external_outputs_preserve_existing_history(synthetic, tmp_path, which):
    value = synthetic
    config = SimpleNamespace(run=value.root, record=value.record, output=tmp_path / "audit",
                             receipt=tmp_path / "receipt.json")
    if which == "output":
        config.output.mkdir()
        protected = config.output / "user.bin"
    elif which == "receipt":
        protected = config.receipt
    else:
        config.output = value.parent_root / "new-audit"
        protected = value.parent_root / "manifest.json"
    if which != "nested":
        protected.write_bytes(b"pre-existing user bytes")
    before = protected.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        value.auditor.run(config)
    assert protected.read_bytes() == before
    assert not (config.output / "failure.json").exists()
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
