"""Synthetic rate-only fitting replay and publication; no corpus experiment."""
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


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_exact_rate")


def test_only_learning_rate_changes_optimizer_contract(study):
    assert study.LR == .01 and study.UPDATES == 128
    assert study.WINDOWS == (0, 1, 2, 3) and study.DEADLINE_S == 3600
    parameter = torch.tensor([.1, -.2], dtype=torch.float64)
    optimizer = study.make_optimizer([parameter])
    previous = study.parent.helper.make_optimizer([parameter.clone()])
    old = {key: value for key, value in previous.param_groups[0].items() if key != "params"}
    new = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    assert new == {**old, "lr": .01}


def test_exact_artifact_membership_bounds_all_states_queries_scores(study):
    files = study.expected_artifacts()
    assert len(files) == 403
    assert sum("/state_" in name and name.endswith(".npz") for name in files) == 129
    assert sum("/state_" in name and name.endswith(".json") for name in files) == 129
    assert sum("/query_" in name for name in files) == 128
    assert sum("/score_" in name for name in files) == 12


def test_declared_rate_budget_and_claim_scope(study):
    recipe = study.recipe()
    assert recipe["training_windows"] == [0, 1, 2, 3]
    assert recipe["updates"] == recipe["optimizer_commits"] == recipe["full_batch_gradient_queries"] == 128
    assert recipe["complete_saved_states"] == 129 and recipe["score_window_readouts"] == 12
    assert recipe["new_gradient_pair_exposures"] == 65536 and recipe["distinct_gradient_pairs"] == 512
    assert recipe["inherited_fitting_pair_exposures"] == 131072
    assert recipe["endpoint_history_pair_exposures"] == 196608
    assert recipe["new_fitting_pair_exposures"] == recipe["historical_comparator_updates_reexecuted"] == 0
    for key in ("learning_rate_sweep", "adaptive_extension", "heldout_corpus_evaluated",
                "physical_fpga_execution", "paid_execution_authorized", "recipe_promoted"):
        assert recipe[key] is False


@pytest.mark.parametrize("ce,half,near", [(.8, True, False), (.8 + 2e-12, False, False),
                                         (.25, True, True), (.25 + 2e-12, True, False)])
def test_gap_gates_are_derived_from_controls_and_never_promote_recipe(study, ce, half, near):
    controls = {arm: {"initial_mean": {"ce": 2.4, "accuracy": .25},
                     "final_mean": {"ce": value, "accuracy": .5}}
                for arm, value in (("mixed", 1.4), ("prior_gru", .2))}
    cell = {"arm": study.ARM, "complete": True, "numerics_pass": True,
            "final_mean": {"ce": ce, "accuracy": .75}}
    result = study.gates(cell, controls)
    assert result["half_gap_ce_threshold"] == pytest.approx(.8)
    assert result["near_gru_ce_threshold"] == .25
    assert result["half_gap_closed"] is half and result["near_gru_local_fit"] is near
    assert result["fraction_of_historical_gap_closed"] == pytest.approx((1.4 - ce) / 1.2)
    assert result["requires_separate_successful_independent_recompute"] is True
    for key in ("recipe_promoted", "paid_execution_authorized", "broader_study_authorized"):
        assert result[key] is False


@pytest.mark.parametrize("field,value", [("complete", 1), ("numerics_pass", False), ("arm", "mixed")])
def test_incomplete_or_wrong_cell_cannot_pass_gate(study, field, value):
    controls = {arm: {"initial_mean": {"ce": 2.4, "accuracy": .25},
                     "final_mean": {"ce": ce, "accuracy": .5}}
                for arm, ce in (("mixed", 1.4), ("prior_gru", .2))}
    cell = {"arm": study.ARM, "complete": True, "numerics_pass": True,
            "final_mean": {"ce": .1, "accuracy": .9}}
    cell[field] = value
    with pytest.raises(ValueError):
        study.gates(cell, controls)


def control_rows():
    return [{"arm": arm, "complete": True, "optimizer_commits": 128,
             "initial_mean": {"ce": 2.4, "accuracy": .25},
             "final_mean": {"ce": .2 if arm == "prior_gru" else 1.4, "accuracy": .75}}
            for arm in ("isolated", "mixed", "prior_gru")]


@pytest.mark.parametrize("case", ["valid", "missing", "incomplete", "float_budget", "nan", "reordered"])
def test_exact_historical_control_coverage_and_numeric_metadata(study, tmp_path, case):
    rows = control_rows()
    if case == "missing":
        rows.pop()
    elif case == "incomplete":
        rows[1]["complete"] = 1
    elif case == "float_budget":
        rows[1]["optimizer_commits"] = 128.0
    elif case == "nan":
        rows[1]["final_mean"]["ce"] = float("nan")
    elif case == "reordered":
        rows.reverse()
    record = tmp_path / "prior.json"
    record.write_text(json.dumps({"cells": rows}))
    if case == "valid":
        actual = study._controls(record)
        assert set(actual) == {"mixed", "prior_gru"}
        assert actual["mixed"]["final_mean"]["ce"] == 1.4
    else:
        with pytest.raises(ValueError):
            study._controls(record)


def test_initial_model_requires_actual_original_full_state(study, tmp_path, monkeypatch):
    cell = tiny_models(study.helper)["mixed"]
    path = tmp_path / "state_000000.npz"
    arrays = {"parameter/" + name: value for name, value in cell["parameters"].items()}
    study.save(path, arrays)
    binding = {"prior_binding": {"synthetic": True}, "initial_state": str(path),
               "initial_state_identity": study.old.file_identity(path)}
    monkeypatch.setattr(study.parent, "initial_models", lambda value: {"mixed": cell})
    assert study.initial_model(binding) is cell
    # A validly encoded but changed original parameter must still be rejected.
    changed = {key: value.clone() for key, value in arrays.items()}
    changed["parameter/MB"][0, 0] += .01
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{key: value.numpy() for key, value in changed.items()})
    binding["initial_state_identity"] = study.old.file_identity(path)
    with pytest.raises(ValueError):
        study.initial_model(binding)


@pytest.mark.parametrize("case", ["wrong_rate", "parameter_order", "boolean_step", "negative_second_moment", "incomplete_moments"])
def test_state_receipt_rejects_optimizer_recipe_or_state_drift(study, tmp_path, case):
    cell = tiny_models(study.helper)["mixed"]
    optimizer = study.make_optimizer(cell["parameters"].values())
    for parameter in cell["parameters"].values():
        parameter.grad = torch.ones_like(parameter) / 100
    optimizer.step()
    step = 1
    parameter = next(iter(cell["parameters"].values()))
    if case == "wrong_rate":
        optimizer.param_groups[0]["lr"] = .001
    elif case == "parameter_order":
        optimizer.param_groups[0]["params"].reverse()
    elif case == "boolean_step":
        step = True
    elif case == "negative_second_moment":
        optimizer.state[parameter]["exp_avg_sq"].reshape(-1)[0] = -.1
    else:
        del optimizer.state[parameter]["exp_avg"]
    with pytest.raises(ValueError):
        study.state_record(tmp_path, step, cell, optimizer, write=True)
    assert not list(tmp_path.iterdir()), "invalid complete state must be rejected before writing"


@pytest.fixture
def predecessor(study, tmp_path, monkeypatch):
    root = tmp_path / "prior"
    (root / "mixed").mkdir(parents=True)
    record, audit, corpus = (tmp_path / name for name in ("record.json", "audit.json", "corpus.txt"))
    corpus.write_text("synthetic opaque input, never decoded as a corpus")
    dataset = {"path": str(corpus), "synthetic": True}
    binding = {"dataset": dataset, "initial_state": str(corpus)}
    write_json(root / "manifest.json", {"binding": binding})
    for name in ["state_000000.npz", *[f"score_initial_{index:03d}_1.npz" for index in range(4)]]:
        (root / "mixed" / name).write_bytes(b"opaque prior file; no native or model execution")
    gate = {"per_arm_local_fitting_gate": {arm: True for arm in ("isolated", "mixed", "prior_gru")},
            "mixed_gain_at_least_002": True}
    write_json(record, {"cells": control_rows(), "gates": gate})
    artifacts = study.prior.artifact_manifest(root)
    write_json(root / "artifact_manifest.json", artifacts)
    source_name = "manuscript/tools/diagnose_fpga_lm_exact_fit.py"
    sources = {source_name: study.old.digest((study.REPO / source_name).read_bytes())}
    monkeypatch.setattr(study.parent, "source_identity", lambda: dict(sources))
    receipt = {"schema": "tikitorch.fpga_lm_exact_fit.cli_audit_receipt.v1", "complete": True,
        "independent_recompute": True, "exit_code": 0, "deadline_seconds": 3600,
        "parent_record_sha256": study.old.digest(record.read_bytes()),
        "run_manifest_sha256": study.old.digest((root / "manifest.json").read_bytes()),
        "artifact_manifest_sha256": study.old.digest((root / "artifact_manifest.json").read_bytes()),
        "runner_sha256": sources[source_name], "source_files_verified": 1,
        "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
        "independent_full_batch_gradient_queries": 384, "independent_tropical_window_vjps": 1024,
        "independent_tropical_full_batch_gradients": 256, "independent_gru_full_batch_gradients": 128,
        "independent_score_readouts": 36, "complete_saved_states_verified": 387, "optimizer_commits_replayed": 384,
        "complete_parent_ancestry_verified": True, "source_input_raw_seal_public_bytes_unchanged": True,
        "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
        "new_optimizer_trajectory_executed": False, "heldout_corpus_evaluated": False,
        "physical_fpga_execution": False, "paid_execution_authorized": False,
        "recipe_promoted": False, "broader_study_authorized": False, **gate,
        "command": ["python", source_name, "--check", "--recompute", "--output", str(root), "--record", str(record)],
        "started_at_utc": "2026-09-11 12:00:00 UTC", "completed_at_utc": "2026-09-11 12:30:00 UTC"}
    write_json(audit, receipt)
    return SimpleNamespace(root=root, record=record, audit=audit, corpus=corpus, binding=binding,
                           config=SimpleNamespace(prior_run=root, prior_record=record, prior_audit=audit, corpus=corpus))


@pytest.mark.parametrize("field,value", [("complete", 1), ("independent_recompute", False),
    ("parent_record_sha256", "0" * 64), ("optimizer_commits_replayed", 383),
    ("completed_at_utc", "2026-09-11 13:00:01 UTC")])
def test_predecessor_audit_receipt_requires_exact_success_and_bounded_duration(study, predecessor, field, value):
    study._audit_receipt(predecessor.root, predecessor.record, predecessor.audit)
    receipt = study.read(predecessor.audit)
    receipt[field] = value
    write_json(predecessor.audit, receipt)
    with pytest.raises(ValueError):
        study._audit_receipt(predecessor.root, predecessor.record, predecessor.audit)


def test_predecessor_semantic_check_keeps_outer_deadline_and_binds_actual_controls(study, predecessor, monkeypatch):
    value = predecessor
    sources = study.source_identity()
    proof = study.head._Validation(value.binding, sources)
    closure = {"roots": {}, "files": [study.old.file_identity(value.corpus)]}
    monkeypatch.setattr(study.head, "ancestry_snapshot", lambda *args: closure)
    monkeypatch.setattr(study.parent, "_bound_config", lambda manifest: object())
    monkeypatch.setattr(study.parent, "bind_inputs", lambda config: (value.binding, b"synthetic", proof))
    monkeypatch.setattr(study.pilot, "corpus_plan", lambda *args: (b"synthetic", value.binding["dataset"]))
    calls = []

    def semantic(*args):
        assert args == (value.root, value.record, False, proof, b"synthetic")
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        calls.append(True)

    def forbidden(*args, **kwargs):
        pytest.fail("public predecessor wrapper would overwrite the outer timer")

    monkeypatch.setattr(study.parent, "_check", semantic)
    monkeypatch.setattr(study.parent, "check", forbidden)
    timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.setitimer(signal.ITIMER_REAL, 30)
        binding, raw, validation = study.bind_inputs(value.config)
        assert calls == [True] and raw == b"synthetic"
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 30
        assert binding["controls"] == study._controls(value.record)
        assert binding["initial_state"] == str(value.root / "mixed/state_000000.npz")
        assert binding["initial_state_identity"] == study.old.file_identity(binding["initial_state"])
        assert [item["path"] for item in binding["initial_scores"]] == [
            str(value.root / "mixed" / f"score_initial_{index:03d}_1.npz") for index in range(4)]
        manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "source_hashes": sources, "binding": binding}
        study.frozen(manifest, validation)
        binding["controls"]["mixed"]["final_mean"]["ce"] += .1
        with pytest.raises(ValueError):
            study.frozen(manifest, validation)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Only immutable-ancestor/D512 input factories are replaced, not math."""
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_exact_rate")
    patch = pytest.MonkeyPatch()
    temporary = tmp_path_factory.mktemp("exact-rate-replay")
    bound_file = temporary / "synthetic-input.txt"
    bound_file.write_text("Synthetic unit fixture, not corpus evidence.\n")
    declaration = temporary / "PLAN.md"
    declaration.write_text("Synthetic rate-only unit fixture.\n")
    tokens = torch.tensor([[0, 1, 0, 2], [1, 3, 1, 0], [2, 0, 2, 3], [3, 2, 3, 1]], dtype=torch.int64)
    targets = torch.tensor([[1, 0, 2, 1], [3, 1, 0, 3], [0, 2, 3, 0], [2, 3, 1, 2]], dtype=torch.int64)
    historical = temporary / "mixed"
    historical.mkdir()
    starting = tiny_models(study.helper)["mixed"]
    initial_path = historical / "state_000000.npz"
    study.save(initial_path, {"parameter/" + name: value for name, value in starting["parameters"].items()})
    initial_scores, initial_metrics = [], []
    for index in range(4):
        trace = study.helper.trace(study.helper.forward(starting["parameters"], tokens[index]))
        path = historical / f"score_initial_{index:03d}_1.npz"
        study.save(path, trace)
        initial_scores.append(study.old.file_identity(path))
        initial_metrics.append(study.parent.metrics(trace["logits"], targets[index]))
    initial_mean = study.parent.mean_metrics(initial_metrics)
    sources = {"synthetic": "0" * 64}
    identity = study.old.file_identity(bound_file)
    binding = {"prior_run": str(temporary), "prior_record": str(bound_file),
               "prior_audit": str(bound_file), "initial_state": str(initial_path),
               "initial_state_identity": study.old.file_identity(initial_path), "initial_scores": initial_scores,
               "controls": {"mixed": {"initial_mean": initial_mean, "final_mean": {"ce": 1.0, "accuracy": .5}},
                            "prior_gru": {"initial_mean": initial_mean, "final_mean": {"ce": .2, "accuracy": .95}}},
               "dataset": {"path": str(bound_file)}, "closure": {"roots": {}, "files": [identity]}}
    validation = study.head._Validation(binding, sources)

    def frozen(manifest, token):
        assert isinstance(token, study.head._Validation)
        study.equal(manifest["binding"], binding, "tiny complete binding")
        study.equal(token.binding, binding, "tiny private binding")
        study.equal(manifest["source_hashes"], sources, "tiny source identity")
        study.equal(study.source_identity(), sources, "tiny live source identity")
        study.equal(study.old.file_identity(bound_file), identity, "tiny unchanged input")

    patch.setattr(study, "PLAN", declaration)
    patch.setattr(study, "source_identity", lambda: dict(sources))
    patch.setattr(study, "bind_inputs", lambda config: (binding, b"synthetic", validation))
    patch.setattr(study, "frozen", frozen)
    patch.setattr(study, "initial_model", lambda binding: tiny_models(study.helper)["mixed"])
    # The runner reuses the frozen parent's batch parser; this replaces only
    # that input boundary, never forward, gradients, Adam or saved replay.
    patch.setattr(study, "batch_data", lambda raw, dataset: (tokens.clone(), targets.clone()))
    calls = {"forward": 0, "independent_forward": 0, "gradient": 0, "independent_gradient": 0}
    original_forward, original_gradient = study.helper.forward, study.helper.gradient

    def traced_forward(*args, **kwargs):
        calls["independent_forward" if kwargs.get("independent") else "forward"] += 1
        return original_forward(*args, **kwargs)

    def traced_gradient(*args, **kwargs):
        calls["independent_gradient" if kwargs.get("independent") else "gradient"] += 1
        return original_gradient(*args, **kwargs)

    patch.setattr(study.helper, "forward", traced_forward)
    patch.setattr(study.helper, "gradient", traced_gradient)
    root, record = temporary / "run", temporary / "public.json"
    config = SimpleNamespace(output=root, record=record, prior_run=temporary,
                             prior_record=bound_file, prior_audit=bound_file, corpus=bound_file)
    try:
        result = study.run(config)

        def reseal():
            artifacts = study.prior.artifact_manifest(root)
            write_json(root / "artifact_manifest.json", artifacts)
            write_json(record, study.public_record(root, study.read(root / "summary.json"), artifacts))

        yield SimpleNamespace(study=study, root=root, record=record, result=result, calls=calls,
                              tokens=tokens, targets=targets, reseal=reseal, config=config,
                              temporary=temporary, bound_file=bound_file, binding=binding,
                              validation=validation, sources=sources)
    finally:
        patch.undo()


def test_complete_rate_only_trajectory_and_independent_adam_replay(synthetic):
    value, study = synthetic, synthetic.study
    assert value.result["complete"] is True
    assert value.result["recipe_promoted"] is False
    assert value.result["heldout_corpus_evaluated"] is False
    assert value.result["optimizer_commits"] == 128
    assert value.result["complete_saved_states"] == 129
    directory = value.root / study.ARM
    previous = study.parent.load(directory / "state_000000.npz")
    names = [key.removeprefix("parameter/") for key in previous]
    for step in range(1, 129):
        current = study.parent.load(directory / f"state_{step:06d}.npz")
        query = study.parent.load(directory / f"query_{step - 1:06d}.npz")
        for name in names:
            parameter = previous["parameter/" + name]
            gradient = query["g_" + name]
            m = previous.get("exp_avg/" + name, torch.zeros_like(parameter))
            v = previous.get("exp_avg_sq/" + name, torch.zeros_like(parameter))
            next_m = .9 * m + .1 * gradient
            next_v = .999 * v + .001 * gradient.square()
            expected = parameter - (.01 / (1 - .9 ** step)) * next_m / (
                next_v.sqrt() / ((1 - .999 ** step) ** .5) + 1e-8)
            for key, expected_value in (("parameter/", expected), ("exp_avg/", next_m),
                                        ("exp_avg_sq/", next_v)):
                torch.testing.assert_close(current[key + name], expected_value, atol=1e-12, rtol=1e-11)
            assert float(current["step/" + name]) == step
        previous = current
    before = dict(value.calls)
    study.check(value.root, value.record)
    assert value.calls == before, "ordinary integrity check must not query a model"
    audited = study.check(value.root, value.record, recompute=True)
    assert audited == value.result
    assert value.calls["independent_gradient"] - before["independent_gradient"] == 512
    assert value.calls["independent_forward"] - before["independent_forward"] == 524


def test_resealed_gradient_corruption_rejected_without_new_forward(synthetic):
    value, study = synthetic, synthetic.study
    path = value.root / study.ARM / "query_000000.npz"
    original = path.read_bytes()
    arrays = study.parent.load(path)
    arrays["g_EMBED"].reshape(-1)[0] += .125
    before = dict(value.calls)
    try:
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{name: array.numpy() for name, array in arrays.items()})
        value.reseal()
        with pytest.raises(ValueError):
            study.check(value.root, value.record)
        assert value.calls == before
    finally:
        path.write_bytes(original)
        value.reseal()


def test_changed_bound_input_rejected_on_fresh_check(synthetic):
    original = synthetic.bound_file.read_bytes()
    try:
        synthetic.bound_file.write_bytes(original + b"changed")
        with pytest.raises(ValueError):
            synthetic.study.check(synthetic.root, synthetic.record)
    finally:
        synthetic.bound_file.write_bytes(original)


def test_changed_current_source_rejected_on_fresh_check(synthetic, monkeypatch):
    monkeypatch.setattr(synthetic.study, "source_identity", lambda: {"synthetic": "f" * 64})
    with pytest.raises(ValueError):
        synthetic.study.check(synthetic.root, synthetic.record)


def test_resealed_legal_route_forgery_fails_independent_forward(synthetic):
    value, study = synthetic, synthetic.study
    path = value.root / study.ARM / "query_000001.npz"
    originals = {file: file.read_bytes() for file in value.root.rglob("*.json")}
    original = path.read_bytes()
    old_sha = study.old.digest(original)
    arrays = study.parent.load(path)
    taps = arrays["window_0/TAP_IDX"]
    taps[0, 1, 0] = 1 - taps[0, 1, 0]
    try:
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{name: array.numpy() for name, array in arrays.items()})
        payload = path.read_bytes()

        def forge(row):
            if isinstance(row, dict):
                for hash_key, size_key in (("sha256", "bytes"), ("query_sha256", "query_bytes")):
                    if row.get(hash_key) == old_sha:
                        row[hash_key] = study.old.digest(payload)
                        if size_key in row:
                            row[size_key] = len(payload)
                for item in row.values():
                    forge(item)
            elif isinstance(row, list):
                for item in row:
                    forge(item)

        for file in originals:
            row = study.read(file)
            forge(row)
            write_json(file, row)
        value.reseal()
        study.check(value.root, value.record)
        with pytest.raises(ValueError, match="independent"):
            study.check(value.root, value.record, recompute=True)
    finally:
        path.write_bytes(original)
        for file, content in originals.items():
            file.write_bytes(content)
        value.reseal()


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt, TimeoutError])
def test_failed_execution_retains_failure_and_no_publication(synthetic, monkeypatch, tmp_path, failure):
    study = synthetic.study
    config = SimpleNamespace(**vars(synthetic.config))
    config.output, config.record = tmp_path / "failed-run", tmp_path / "failed-public.json"

    def fail(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        raise failure("injected failure before publication")

    before = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(study, "execute", fail)
    with pytest.raises(failure):
        study.run(config)
    assert not config.record.exists()
    assert (config.output / "failure.json").is_file()
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("foreign", [False, True])
def test_final_check_failure_retracts_only_owned_public_inode(synthetic, monkeypatch, tmp_path, foreign):
    value, study = synthetic, synthetic.study
    config = SimpleNamespace(**vars(value.config))
    config.output, config.record = tmp_path / "copied-run", tmp_path / "copied-public.json"
    root_metadata = {"manifest.json", "protocol.json", "declaration.md", "summary.json"}

    def copy_saved_payload(manifest, output, raw, *, write=False, recompute=False):
        assert write is True and recompute is False
        # Replay is not repeated: valid tiny payload bytes are copied solely
        # to exercise the final publication transaction and its failure path.
        for relative in study.expected_artifacts() - root_metadata:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((value.root / relative).read_bytes())
        return study.read(value.root / "summary.json")

    def fail_after_publication(root, record, *args):
        assert record.is_file()
        if foreign:
            payload = record.read_bytes()
            record.unlink()
            record.write_bytes(payload)  # Same bytes, different owner/inode.
        raise ValueError("injected final verification failure")

    monkeypatch.setattr(study, "execute", copy_saved_payload)
    monkeypatch.setattr(study, "_check", fail_after_publication)
    with pytest.raises(ValueError, match="injected final"):
        study.run(config)
    assert (config.output / "failure.json").is_file()
    assert config.record.exists() is foreign
    assert (config.output / "rejected_public.json").exists() is (not foreign)
    assert not list(tmp_path.glob(".tt-exact-rate-*"))
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("which", ["output", "record"])
def test_existing_outputs_are_never_overwritten(synthetic, tmp_path, which):
    config = SimpleNamespace(**vars(synthetic.config))
    config.output, config.record = tmp_path / "existing-run", tmp_path / "existing-public.json"
    if which == "output":
        config.output.mkdir()
        protected = config.output / "user.txt"
    else:
        protected = config.record
    protected.write_bytes(b"pre-existing user-owned bytes")
    with pytest.raises(FileExistsError):
        synthetic.study.run(config)
    assert protected.read_bytes() == b"pre-existing user-owned bytes"


def test_changed_original_score_payload_is_not_accepted_as_same_initialization(synthetic):
    path = Path(synthetic.binding["initial_scores"][0]["path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"changed source bytes")
        with pytest.raises(ValueError):
            synthetic.study.check(synthetic.root, synthetic.record)
    finally:
        path.write_bytes(original)
