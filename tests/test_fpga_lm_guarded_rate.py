"""Synthetic guarded-vs-saved-dense trajectory parity, not new learning."""
from __future__ import annotations

import importlib
import json
import signal
from types import SimpleNamespace

import pytest
import torch

from test_fpga_lm_exact_rate_audit import build_tiny_rate


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def verifier():
    return importlib.import_module("manuscript.tools.verify_fpga_lm_guarded_rate")


def test_source_closure_includes_guard_and_new_verifier_without_relabeling_old_sources(verifier):
    sources = verifier.source_identity()
    assert len(sources) == 67
    for name in ("manuscript/tools/fpga_lm_guarded_forward.py", "tests/test_fpga_lm_guarded_forward.py",
                 "manuscript/tools/verify_fpga_lm_guarded_rate.py", "tests/test_fpga_lm_guarded_rate.py"):
        assert sources[name] == verifier.rate.old.digest((verifier.REPO / name).read_bytes())
    assert all(sources[name] == digest for name, digest in verifier.parent.source_identity().items())


@pytest.mark.parametrize("case", ["signed_zero", "dtype", "shape", "missing", "different_value"])
def test_byte_identity_rejects_differences_hidden_by_numeric_equality(verifier, case):
    expected = {"x": torch.tensor([0., 1.], dtype=torch.float64)}
    actual = {"x": expected["x"].clone()}
    if case == "signed_zero":
        actual["x"][0] = -0.0
        assert torch.equal(actual["x"], expected["x"])
    elif case == "dtype":
        actual["x"] = actual["x"].float()
    elif case == "shape":
        actual["x"] = actual["x"].reshape(1, 2)
    elif case == "missing":
        actual = {}
    else:
        actual["x"][1] += .1
    with pytest.raises(ValueError):
        verifier.byte_equal(actual, expected, "synthetic exact IEEE payload")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    verifier = importlib.import_module("manuscript.tools.verify_fpga_lm_guarded_rate")
    parent, rate = verifier.parent, verifier.rate
    patch = pytest.MonkeyPatch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        temporary = tmp_path_factory.mktemp("guarded-rate-replay")
        value = build_tiny_rate(parent, temporary, patch)
        pins = dict(parent.TRUST)
        for key, path in {
            "rate_public_sha256": value.record, "rate_manifest_sha256": value.root / "manifest.json",
            "rate_seal_sha256": value.root / "artifact_manifest.json", "parent_audit_sha256": value.parent_audit,
            "parent_public_sha256": value.parent_record, "parent_manifest_sha256": value.parent_root / "manifest.json",
            "parent_seal_sha256": value.parent_root / "artifact_manifest.json",
        }.items():
            pins[key] = rate.old.digest(path.read_bytes())
        ancestors = rate.prior.artifact_manifest(value.parent_root)
        pins.update(rate_source_count=len(rate.source_identity()), rate_artifact_count=len(value.artifacts),
                    rate_artifact_bytes=sum(row["bytes"] for row in value.artifacts),
                    parent_source_count=len(rate.parent.source_identity()), parent_artifact_count=len(ancestors),
                    parent_artifact_bytes=sum(row["bytes"] for row in ancestors),
                    ancestor_tree_count=len(value.binding["closure"]["roots"]),
                    ancestor_external_file_count=len(value.binding["closure"]["files"]))
        patch.setattr(parent, "TRUST", pins)

        def forbidden(*args, **kwargs):
            pytest.fail("unapproved prior numerical trajectory/check wrapper")

        for module, names in ((rate, ("bind_inputs", "check")),
                              (rate.parent, ("bind_inputs", "_check", "check")),
                              (rate.head, ("bind_inputs", "_check", "check"))):
            for name in names:
                patch.setattr(module, name, forbidden)
        audit_config = SimpleNamespace(run=value.root, record=value.record, output=temporary / "dense-audit",
                                       receipt=temporary / "dense-audit.json")
        # A real independent tiny audit establishes this fixture's new trust
        # anchor; the guarded verifier must not repeat its autograd replay.
        audit_receipt = parent.run(audit_config)
        patch.setattr(verifier, "AUDIT_RECEIPT_SHA256", rate.old.digest(audit_config.receipt.read_bytes()))
        patch.setattr(parent, "run", forbidden)
        patch.setattr(rate, "_check", forbidden)
        value.verifier, value.audit_receipt, value.audit_config = verifier, audit_receipt, audit_config
        yield value
    finally:
        patch.undo()
        torch.set_num_threads(previous_threads)


def test_successful_actual_tiny_audit_is_bound_without_a_second_audit(synthetic):
    value, verifier = synthetic, synthetic.verifier
    manifest, raw, token, facts = verifier.validate_inputs(value.root, value.record, value.audit_config.receipt)
    assert manifest == value.manifest and raw == value.raw and token.binding == value.binding
    assert facts["independent_autograd_audit_reused"] is True
    assert facts["successful_audit_receipt_sha256"] == value.rate.old.digest(value.audit_config.receipt.read_bytes())
    assert len(facts["source_hashes"]) == 67


@pytest.mark.parametrize("case", ["unpinned", "receipt", "private_manifest", "failure_file"])
def test_missing_or_changed_successful_audit_trust_is_rejected(synthetic, monkeypatch, case):
    value, verifier = synthetic, synthetic.verifier
    if case == "unpinned":
        monkeypatch.setattr(verifier, "AUDIT_RECEIPT_SHA256", None)
        with pytest.raises(ValueError):
            verifier.validate_inputs(value.root, value.record, value.audit_config.receipt)
        return
    path = (value.audit_config.receipt if case == "receipt" else
            value.audit_config.output / ("failure.json" if case == "failure_file" else "audit_manifest.json"))
    original = path.read_bytes() if path.exists() else None
    try:
        path.write_bytes((original or b'{}') + b"\n")
        with pytest.raises(ValueError):
            verifier.validate_inputs(value.root, value.record, value.audit_config.receipt)
    finally:
        if original is None:
            path.unlink()
        else:
            path.write_bytes(original)


@pytest.mark.parametrize("fallback", ["none", "whole", "operation"])
def test_query_gradients_are_exact_with_certified_or_dense_fallback_paths(synthetic, monkeypatch, fallback):
    value, verifier, rate = synthetic, synthetic.verifier, synthetic.rate
    cell = rate.initial_model(value.binding)
    before = {name: parameter.clone() for name, parameter in cell["parameters"].items()}
    if fallback == "whole":
        original = verifier.guarded.forward
        monkeypatch.setattr(verifier.guarded, "forward", lambda *args, **kwargs:
            original(*args, **kwargs, off_weight=verifier.guarded.OFF_WEIGHT + .125))
    elif fallback == "operation":
        original = verifier.guarded._max_operation
        monkeypatch.setattr(verifier.guarded, "_max_operation", lambda matrix, vector, support, off_weight:
            original(matrix, vector, support, off_weight + 1000000.0))
    timings = verifier.Timings()
    gradients, row = verifier.query(value.root / rate.ARM, 0, cell, value.tokens, value.targets, timings)
    saved = rate.load(value.root / rate.ARM / "query_000000.npz")
    verifier.byte_equal(gradients, {name: saved["g_" + name] for name in gradients}, "complete recomputed batch G")
    verifier.byte_equal(cell["parameters"], before, "query does not update parameters")
    assert timings.values["guarded_forward"]["calls"] == 4
    assert timings.values["manual_vjp_and_batch_sum"]["calls"] == 4
    assert len(row["windows"]) == 4
    for window in row["windows"]:
        guard = window["guards"]
        assert guard["projection_fast_operations"] + guard["projection_dense_operations"] == 16
        assert guard["head_fast_operations"] + guard["head_dense_operations"] == 4
        assert guard["whole_model_dense_fallback"] is (fallback == "whole")
        assert guard["projection_fast_operations"] == (16 if fallback == "none" else 0)
        assert guard["head_fast_operations"] == (4 if fallback == "none" else 0)


@pytest.mark.parametrize("case", ["gradient", "route"])
def test_query_rejects_recomputed_gradient_or_trace_corruption(synthetic, monkeypatch, case):
    value, verifier, rate = synthetic, synthetic.verifier, synthetic.rate
    cell = rate.initial_model(value.binding)
    if case == "gradient":
        original = verifier.helper.gradient

        def corrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            result["EMBED"] = result["EMBED"].clone()
            result["EMBED"].reshape(-1)[0] += .125
            return result

        monkeypatch.setattr(verifier.helper, "gradient", corrupted)
    else:
        original = verifier.guarded.forward

        def corrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            result["routes"]["AD_HEAD"] = result["routes"]["AD_HEAD"].clone()
            result["routes"]["AD_HEAD"][0, 0] = (result["routes"]["AD_HEAD"][0, 0] + 1) % 16
            return result

        monkeypatch.setattr(verifier.guarded, "forward", corrupted)
    with pytest.raises(ValueError, match="bytes differ"):
        verifier.query(value.root / rate.ARM, 0, cell, value.tokens, value.targets, verifier.Timings())


@pytest.mark.parametrize("case", ["boolean_count", "wrong_total", "contradictory_fallback"])
def test_guard_counts_require_literal_complete_consistent_operations(synthetic, case):
    value, verifier = synthetic, synthetic.verifier
    cell = value.rate.initial_model(value.binding)
    base = verifier.guarded.forward(cell["parameters"], value.tokens[0])
    if case == "boolean_count":
        base["guard_diagnostics"]["head_fast_operations"] = True
    elif case == "wrong_total":
        base["guard_diagnostics"]["projection_fast_operations"] -= 1
    else:
        base["guard_diagnostics"]["structure_certified"] = False
        base["guard_diagnostics"]["whole_model_dense_fallback"] = True
    with pytest.raises(ValueError):
        verifier._guards(base)


def test_all128_guarded_updates_match_saved_dense_traces_gradients_and_adam_states(synthetic, monkeypatch):
    value, verifier, rate = synthetic, synthetic.verifier, synthetic.rate
    before = tree_bytes(value.root)
    forward, gradient, state_record = verifier.guarded.forward, verifier.helper.gradient, rate.state_record
    bases, gradients, states = [], [], []

    def guarded_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        bases.append((result["geometry"], dict(result["guard_diagnostics"])))
        return result

    def manual_gradient(*args, **kwargs):
        assert kwargs.get("independent", False) is False
        gradients.append(True)
        return gradient(*args, **kwargs)

    def checked_state(directory, step, *args, **kwargs):
        assert kwargs.get("write", False) is False
        states.append(step)
        return state_record(directory, step, *args, **kwargs)

    def no_dense_model(*args, **kwargs):
        pytest.fail("healthy synthetic guard certificate unexpectedly called a dense model")

    monkeypatch.setattr(verifier.guarded, "forward", guarded_forward)
    monkeypatch.setattr(verifier.helper, "gradient", manual_gradient)
    monkeypatch.setattr(verifier.helper, "forward", no_dense_model)
    monkeypatch.setattr(rate, "state_record", checked_state)
    config = SimpleNamespace(run=value.root, record=value.record, audit_receipt=value.audit_config.receipt,
                             output=value.temporary / "guarded-complete", receipt=value.temporary / "guarded-complete.json")
    receipt = verifier.run(config)
    report = receipt["parity"]
    assert report["complete"] is True and report["byte_exact_traces_gradients_and_complete_states"] is True
    assert rate.read(config.receipt) == receipt
    assert receipt["guarded_parity_replay_completed"] is True
    assert receipt["independent_autograd_rerun"] is False
    assert receipt["independent_autograd_audit_reused"] is True
    assert receipt["source_input_raw_seal_public_bytes_unchanged"] is True
    assert receipt["speedup_claim"] is False and receipt["physical_fpga_execution"] is False
    assert report["guarded_window_forwards"] == 524 and report["manual_window_vjps"] == 512
    assert report["complete_saved_states_verified"] == 129 and report["optimizer_commits_replayed"] == 128
    assert report["score_readouts_verified"] == 12
    assert report["recomputed_identical_gradients_applied"] is True
    assert len(bases) == 524 and len(gradients) == 512
    assert states == list(range(129))
    assert sum(guard["projection_fast_operations"] for _, guard in bases) == 524 * 4 * 4
    assert sum(guard["head_fast_operations"] for _, guard in bases) == 524 * 4
    for geometry, guard in bases:
        assert geometry["T"] == 4 and geometry["N"] == 4
        assert guard["structure_certified"] is True
        assert guard["whole_model_dense_fallback"] is False
        assert guard["projection_dense_operations"] == guard["head_dense_operations"] == 0
    assert tree_bytes(value.root) == before


def test_replay_applies_returned_guarded_gradient_not_saved_gradient(synthetic, monkeypatch):
    value, verifier = synthetic, synthetic.verifier
    query = verifier.query

    def corrupted_after_verified_comparison(*args, **kwargs):
        gradient, row = query(*args, **kwargs)
        gradient["EMBED"] = gradient["EMBED"].clone()
        gradient["EMBED"].reshape(-1)[0] += .125
        return gradient, row

    monkeypatch.setattr(verifier, "query", corrupted_after_verified_comparison)
    with pytest.raises(ValueError):
        verifier.replay(value.manifest, value.raw)


@pytest.mark.parametrize("case", ["moment", "options"])
def test_replay_rejects_changed_actual_adam_moment_or_options(synthetic, monkeypatch, case):
    value, verifier, rate = synthetic, synthetic.verifier, synthetic.rate
    make_optimizer = rate.make_optimizer

    def invalid_optimizer(parameters):
        optimizer = make_optimizer(parameters)
        if case == "options":
            optimizer.param_groups[0]["lr"] = .001
        else:
            step = optimizer.step

            def corrupt_step(*args, **kwargs):
                result = step(*args, **kwargs)
                parameter = optimizer.param_groups[0]["params"][0]
                optimizer.state[parameter]["exp_avg"].reshape(-1)[0] += .125
                return result

            optimizer.step = corrupt_step
        return optimizer

    monkeypatch.setattr(rate, "make_optimizer", invalid_optimizer)
    with pytest.raises(ValueError):
        verifier.replay(value.manifest, value.raw)


def test_full_state_check_rejects_signed_zero_moment_difference(synthetic):
    value, verifier, rate = synthetic, synthetic.verifier, synthetic.rate
    cell = rate.initial_model(value.binding)
    optimizer = rate.make_optimizer(cell["parameters"].values())
    timings = verifier.Timings()
    gradients, _ = verifier.query(value.root / rate.ARM, 0, cell, value.tokens, value.targets, timings)
    for name, parameter in cell["parameters"].items():
        parameter.grad = gradients[name].clone()
    optimizer.step()
    verifier.state(value.root / rate.ARM, 1, cell, optimizer, timings)
    moment = optimizer.state[cell["parameters"]["MP"]]["exp_avg"]
    original = moment.clone()
    index = int(torch.nonzero(moment.reshape(-1) == 0)[0])
    moment.reshape(-1)[index] = 0.0 if torch.signbit(moment.reshape(-1)[index]) else -0.0
    assert torch.equal(moment, original), "numeric-only equality cannot detect this corruption"
    with pytest.raises(ValueError, match="bytes differ"):
        verifier.state(value.root / rate.ARM, 1, cell, optimizer, timings)


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt, TimeoutError])
def test_failed_guarded_replay_is_retained_without_public_success(synthetic, monkeypatch, tmp_path, failure):
    value, verifier = synthetic, synthetic.verifier
    config = SimpleNamespace(run=value.root, record=value.record, audit_receipt=value.audit_config.receipt,
                             output=tmp_path / "guarded", receipt=tmp_path / "guarded.json")

    def fail(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        raise failure("injected guarded replay failure")

    monkeypatch.setattr(verifier, "replay", fail)
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(failure):
        verifier.run(config)
    assert (config.output / "failure.json").is_file()
    assert not config.receipt.exists()
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("which", ["output", "receipt", "history", "audit_capsule"])
def test_exclusive_external_publication_never_overwrites_history(synthetic, tmp_path, which):
    value, verifier = synthetic, synthetic.verifier
    config = SimpleNamespace(run=value.root, record=value.record, audit_receipt=value.audit_config.receipt,
                             output=tmp_path / "guarded", receipt=tmp_path / "guarded.json")
    if which == "output":
        config.output.mkdir()
        protected = config.output / "user.bin"
    elif which == "receipt":
        protected = config.receipt
    elif which == "history":
        config.output = value.root / "new-guarded-run"
        protected = value.root / "manifest.json"
    else:
        config.output = value.audit_config.output / "new-guarded-run"
        protected = value.audit_config.output / "audit_manifest.json"
    if which in ("output", "receipt"):
        protected.write_bytes(b"user-owned existing bytes")
    before = protected.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        verifier.run(config)
    assert protected.read_bytes() == before
    assert not (config.output / "failure.json").exists()
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("foreign", [False, True])
def test_postpublication_failure_retracts_only_owned_success_receipt(synthetic, monkeypatch, tmp_path, foreign):
    value, verifier = synthetic, synthetic.verifier
    config = SimpleNamespace(run=value.root, record=value.record, audit_receipt=value.audit_config.receipt,
                             output=tmp_path / "guarded", receipt=tmp_path / "guarded.json")
    # This tests only the publication failure transaction. Full genuine replay
    # is exercised above; the injected report is never returned as success.
    monkeypatch.setattr(verifier, "replay", lambda *args: {"complete": True, "synthetic_failure_fixture": True})
    original = verifier._unchanged
    calls = []

    def fail_after_link(*args):
        original(*args)
        calls.append(True)
        if len(calls) == 2:
            assert config.receipt.is_file()
            if foreign:
                payload = config.receipt.read_bytes()
                config.receipt.unlink()
                config.receipt.write_bytes(payload)
            raise ValueError("injected postpublication parity failure")

    monkeypatch.setattr(verifier, "_unchanged", fail_after_link)
    with pytest.raises(ValueError, match="postpublication parity"):
        verifier.run(config)
    assert (config.output / "failure.json").is_file()
    assert config.receipt.exists() is foreign
    assert (config.output / "rejected_public.json").exists() is (not foreign)
    assert not list(tmp_path.glob(".tt-guarded-rate-parity-*"))
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("filename", ["verification.json", "parity.json"])
def test_private_result_mutation_before_sealing_cannot_be_published(synthetic, monkeypatch, tmp_path, filename):
    value, verifier = synthetic, synthetic.verifier
    config = SimpleNamespace(run=value.root, record=value.record, audit_receipt=value.audit_config.receipt,
                             output=tmp_path / "guarded", receipt=tmp_path / "guarded.json")
    monkeypatch.setattr(verifier, "replay", lambda *args: {"complete": True, "synthetic_failure_fixture": True})
    original = verifier._unchanged

    def mutate_before_descriptor_capture(*args):
        original(*args)
        write_json(config.output / filename, {"changed": True})

    monkeypatch.setattr(verifier, "_unchanged", mutate_before_descriptor_capture)
    with pytest.raises(ValueError):
        verifier.run(config)
    assert (config.output / "failure.json").is_file()
    assert not config.receipt.exists()
    assert not list(tmp_path.glob(".tt-guarded-rate-parity-*"))
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
