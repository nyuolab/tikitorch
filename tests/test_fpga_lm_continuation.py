"""Synthetic full-Adam continuation contracts; no corpus or cloud execution."""
from __future__ import annotations

import copy
import importlib
import json
import random
import signal
from types import SimpleNamespace

import pytest
import torch

from test_fpga_lm_exact_fit_replay import tiny_models


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def models(helper):
    values = tiny_models(helper)
    return {arm: values[arm] for arm in ("mixed", "prior_gru")}


def synthetic_gradient(cell, step):
    """Deterministic algebraic input, not a corpus/model backward pass."""
    result = {}
    for index, (name, parameter) in enumerate(cell["parameters"].items()):
        values = torch.arange(parameter.numel(), dtype=torch.float64).reshape(parameter.shape)
        values = (((values + 3 * step + index) % 11) - 5) / 1000
        if name in ("MP", "W_HEAD"):
            # Tiny fixture's inactive entries are about -88.7; active entries
            # remain near zero during these bounded synthetic updates.
            values = values.masked_fill(parameter.detach() < -10, 0)
        result[name] = values
    return result


def advance(cell, optimizer, gradients):
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in cell["parameters"].items():
        parameter.grad = gradients[name].clone()
    optimizer.step()


def optimizer_arrays(cell, optimizer):
    result = {"parameter/" + name: parameter.detach().clone()
              for name, parameter in cell["parameters"].items()}
    for name, parameter in cell["parameters"].items():
        for key, value in optimizer.state[parameter].items():
            result[key + "/" + name] = value.detach().clone()
    return result


def checkpoint(study, directory, arm, counter_dtype=torch.float32):
    """A genuine eager Adam history, entirely driven by synthetic gradients."""
    directory.mkdir(parents=True)
    cell = models(study.helper)[arm]
    optimizer = study.parent.make_optimizer(arm, cell["parameters"].values())
    for step in range(256):
        advance(cell, optimizer, synthetic_gradient(cell, step))
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        state["step"] = state["step"].to(counter_dtype)
    storage = study.parent.Storage(directory, SimpleNamespace(check=lambda: None))
    receipt = study.parent.state_record(directory, 256, cell, optimizer, arm, write=True, storage=storage)
    return cell, optimizer, directory / "state_000256.npz", receipt


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.continue_fpga_lm_development")


def test_source_closure_contains_only_two_new_continuation_files(study):
    sources = study.source_identity()
    assert len(sources) == 73
    previous = study.endpoint.source_identity()
    assert {name: sources[name] for name in previous} == previous
    assert set(sources) - set(previous) == {
        "manuscript/tools/continue_fpga_lm_development.py",
        "tests/test_fpga_lm_continuation.py",
    }
    for name in set(sources) - set(previous):
        assert sources[name] == study.old.digest((study.REPO / name).read_bytes())


def test_order_zero_reuses_the_exact_complete_second_pass_without_global_rng(study):
    before = random.getstate()
    expected = list(range(1024))
    random.Random(0).shuffle(expected)
    assert study.training_order() == expected
    assert random.getstate() == before
    assert sorted(expected) == list(range(1024)) and len(set(expected)) == 1024
    altered = study.training_order()
    altered[0] = -1
    assert study.training_order() == expected


@pytest.mark.parametrize("axis,value,error", [(0, 10800., TimeoutError), (1, 10800., TimeoutError),
                                               (0, -1., RuntimeError), (1, float("nan"), RuntimeError)])
def test_continuation_deadline_checks_both_clock_domains(study, monkeypatch, axis, value, error):
    clock = study.Deadline()
    elapsed = [10., 10.]
    elapsed[axis] = value
    monkeypatch.setattr(clock, "elapsed", lambda: tuple(elapsed))
    with pytest.raises(error):
        clock.check()


def test_continuation_deadline_restores_signal_without_waiting(study):
    clock = study.Deadline()
    previous = signal.getsignal(signal.SIGALRM)
    try:
        clock.arm()
        assert 10700 < signal.getitimer(signal.ITIMER_REAL)[0] <= 10800
    finally:
        clock.close()
    assert not clock.watcher.is_alive()
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("case", ["cap", "free", "outside", "existing", "boolean_size"])
def test_storage_preserves_existing_bytes_and_enforces_eight_gib_bound(study, tmp_path, monkeypatch, case):
    assert study.STORAGE_CAP == 8 * 2**30 and study.FREE_RESERVE == 4 * 2**30
    root = tmp_path / "owned"
    root.mkdir()
    storage = study.Storage(root, SimpleNamespace(check=lambda: None))
    target = root / "new.json"
    monkeypatch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=study.FREE_RESERVE + 100))
    if case == "cap":
        storage.used = study.STORAGE_CAP
    elif case == "outside":
        target = tmp_path / "unowned.json"
    elif case == "existing":
        target.write_bytes(b"original user bytes")
    before = tree_bytes(tmp_path)
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        storage.reserve(target, True if case == "boolean_size" else 101 if case == "free" else 1)
    assert tree_bytes(tmp_path) == before


@pytest.fixture
def inherited_state(study, tmp_path, monkeypatch):
    prior_root = tmp_path / "prior"
    entries = {}
    for arm in ("mixed", "prior_gru"):
        cell, optimizer, path, receipt = checkpoint(study, prior_root / f"{arm}_seed_0", arm)
        entries[arm] = SimpleNamespace(cell=cell, optimizer=optimizer, path=path, receipt=receipt)
    monkeypatch.setattr(study.parent, "initial_models", lambda binding: models(study.helper))
    return SimpleNamespace(root=prior_root, entries=entries)


def assert_next_adam_formula(before, after, gradients, lr, step):
    """Independent scalar Adam algebra, including the inherited counter."""
    for name, gradient in gradients.items():
        first = .9 * before["exp_avg/" + name] + .1 * gradient
        second = .999 * before["exp_avg_sq/" + name] + .001 * gradient.square()
        wanted = before["parameter/" + name] - lr * first / (1 - .9 ** step) / (
            second.sqrt() / ((1 - .999 ** step) ** .5) + 1e-8)
        for prefix, value in (("parameter/", wanted), ("exp_avg/", first), ("exp_avg_sq/", second)):
            torch.testing.assert_close(after[prefix + name], value, atol=1e-12, rtol=1e-11)
        assert float(after["step/" + name]) == step


@pytest.mark.parametrize("arm", ["mixed", "prior_gru"])
@pytest.mark.parametrize("counter_dtype", [torch.float32, torch.float64])
def test_full_adam_restore_and_next257_update_equal_uninterrupted(study, tmp_path, arm, counter_dtype):
    old_cell, uninterrupted, path, receipt = checkpoint(study, tmp_path / arm, arm, counter_dtype)
    before = optimizer_arrays(old_cell, uninterrupted)
    protected = tree_bytes(path.parent)
    cell = models(study.helper)[arm]
    optimizer = study.restore_optimizer(cell, path, arm)
    study.guard.byte_equal(optimizer_arrays(cell, optimizer), before, "every inherited state byte restored")
    assert optimizer.param_groups[0]["params"] == list(cell["parameters"].values())
    assert receipt["initial_optimizer_state_empty"] is False
    for name, parameter in cell["parameters"].items():
        state = optimizer.state[parameter]
        assert state["step"].dtype == counter_dtype and float(state["step"]) == 256
        assert state["exp_avg"].data_ptr() != uninterrupted.state[old_cell["parameters"][name]]["exp_avg"].data_ptr()
    assert any(bool(value.any()) for key, value in before.items() if key.startswith("exp_avg/"))
    gradients = synthetic_gradient(cell, 256)
    advance(old_cell, uninterrupted, gradients)
    advance(cell, optimizer, gradients)
    after = optimizer_arrays(cell, optimizer)
    study.guard.byte_equal(after, optimizer_arrays(old_cell, uninterrupted), "resumed next step equals uninterrupted")
    assert_next_adam_formula(before, after, gradients, .01 if arm == "mixed" else .001, 257)
    assert tree_bytes(path.parent) == protected


def test_parameters_only_adam_reset_is_detectably_not_the_same_continuation(study, inherited_state):
    previous = inherited_state.entries["mixed"]
    continued = models(study.helper)["mixed"]
    optimizer = study.restore_optimizer(continued, previous.path, "mixed")
    reset = models(study.helper)["mixed"]
    with torch.no_grad():
        for name, value in reset["parameters"].items():
            value.copy_(continued["parameters"][name])
    empty = study.parent.make_optimizer("mixed", reset["parameters"].values())
    gradients = synthetic_gradient(continued, 256)
    advance(continued, optimizer, gradients)
    advance(reset, empty, gradients)
    assert any(not torch.equal(continued["parameters"][name], reset["parameters"][name]) for name in gradients)
    assert all(float(s["step"]) == 257 for s in optimizer.state.values())
    assert all(float(s["step"]) == 1 for s in empty.state.values())


@pytest.mark.parametrize("case", ["missing_first", "missing_second", "counter_reset", "counter_vector",
                                  "counter_int", "moment_float32", "moment_shape", "negative_second",
                                  "nan_first", "params_only", "extra_state", "wrong_rate",
                                  "wrong_order", "metadata_step_bool", "metadata_hash"])
def test_restore_rejects_incomplete_or_mismatched_full_state(study, inherited_state, case):
    previous = inherited_state.entries["mixed"]
    path = previous.path
    arrays, metadata = study.load(path), study.read(path.with_suffix(".json"))
    name = metadata["parameter_order"][0]
    if case == "missing_first":
        del arrays["exp_avg/" + name]
    elif case == "missing_second":
        del arrays["exp_avg_sq/" + name]
    elif case == "counter_reset":
        arrays["step/" + name].zero_()
    elif case == "counter_vector":
        arrays["step/" + name] = arrays["step/" + name].reshape(1)
    elif case == "counter_int":
        arrays["step/" + name] = arrays["step/" + name].long()
    elif case == "moment_float32":
        arrays["exp_avg/" + name] = arrays["exp_avg/" + name].float()
    elif case == "moment_shape":
        arrays["exp_avg/" + name] = arrays["exp_avg/" + name].flatten()
    elif case == "negative_second":
        arrays["exp_avg_sq/" + name].reshape(-1)[0] = -1
    elif case == "nan_first":
        arrays["exp_avg/" + name].reshape(-1)[0] = float("nan")
    elif case == "params_only":
        arrays = {key: value for key, value in arrays.items() if key.startswith("parameter/")}
    elif case == "extra_state":
        arrays["exp_avg/unowned"] = torch.ones(1, dtype=torch.float64)
    elif case == "wrong_rate":
        metadata["optimizer_group"]["lr"] = .001
    elif case == "wrong_order":
        metadata["parameter_order"].reverse()
    elif case == "metadata_step_bool":
        metadata["step"] = True
    else:
        metadata["parameters_sha256"] = "0" * 64
    if case not in ("wrong_rate", "wrong_order", "metadata_step_bool", "metadata_hash"):
        path.unlink()
        study.parent.parent.save(path, arrays)
        metadata.update(bytes=path.stat().st_size, sha256=study.old.digest(path.read_bytes()))
    write_json(path.with_suffix(".json"), metadata)
    cell = models(study.helper)["mixed"]
    before = {key: value.clone() for key, value in cell["parameters"].items()}
    with pytest.raises((ValueError, TypeError)):
        study.restore_optimizer(cell, path, "mixed")
    study.guard.byte_equal(cell["parameters"], before, "invalid checkpoint rejected before mutating model")


@pytest.mark.parametrize("step", [0, 255, 513, True, 256.0, "256"])
def test_new_state_receipts_only_accept_literal_global256_through512(study, inherited_state, tmp_path, step):
    previous = inherited_state.entries["mixed"]
    cell = models(study.helper)["mixed"]
    optimizer = study.restore_optimizer(cell, previous.path, "mixed")
    output = tmp_path / "continuation"
    output.mkdir()
    with pytest.raises(ValueError):
        study.state_record(output, step, cell, optimizer, "mixed", write=True,
                           storage=study.Storage(output, SimpleNamespace(check=lambda: None)))
    assert not list(output.iterdir())


def test_first_new_full_state_keeps_global_counter_and_original_start_bytes(study, inherited_state, tmp_path):
    previous = inherited_state.entries["mixed"]
    cell = models(study.helper)["mixed"]
    optimizer = study.restore_optimizer(cell, previous.path, "mixed")
    output = tmp_path / "continuation"
    output.mkdir()
    storage = study.Storage(output, SimpleNamespace(check=lambda: None))
    initial = study.state_record(output, 256, cell, optimizer, "mixed", write=True, storage=storage)
    study.guard.byte_equal(study.load(output / "state_000256.npz"), study.load(previous.path), "exact restored start")
    assert initial["step"] == 256 and initial["initial_optimizer_state_empty"] is False
    gradients = synthetic_gradient(cell, 256)
    advance(cell, optimizer, gradients)
    result = study.state_record(output, 257, cell, optimizer, "mixed", write=True, storage=storage)
    assert result["step"] == 257 and result["initial_optimizer_state_empty"] is False
    arrays = study.load(output / "state_000257.npz")
    assert_next_adam_formula(study.load(previous.path), arrays, gradients, .01, 257)
    assert set(output.glob("*.npz")) == {output / "state_000256.npz", output / "state_000257.npz"}


def test_second_checkpoint_load_cannot_substitute_unvalidated_moments(study, inherited_state, monkeypatch):
    previous = inherited_state.entries["mixed"]
    original = study.load

    def altered(path):
        arrays = original(path)
        arrays["exp_avg/EMBED"].reshape(-1)[0] += .125
        return arrays

    monkeypatch.setattr(study, "load", altered)
    cell = models(study.helper)["mixed"]
    before = {key: value.clone() for key, value in cell["parameters"].items()}
    with pytest.raises(ValueError):
        study.restore_optimizer(cell, previous.path, "mixed")
    study.guard.byte_equal(cell["parameters"], before, "second-read substitution rejected transactionally")


def test_declared_two_arm_fixed_second_pass_and_no_new_selection(study):
    recipe = study.recipe()
    assert study.ARMS == ("mixed", "prior_gru")
    assert (study.START_STEP, study.UPDATES, study.BATCH_SIZE, study.SEQUENCE) == (256, 256, 4, 128)
    assert study.TRAIN_WINDOWS == tuple(range(1024))
    assert study.SCORE_TRAIN_WINDOWS == tuple(range(0, 1024, 16))
    assert study.DEV_WINDOWS == tuple(range(88, 152))
    assert recipe["order_seed"] == 0 and recipe["start_step"] == 256 and recipe["final_step"] == 512
    assert recipe["gradient_queries"] == recipe["optimizer_commits"] == 512
    assert recipe["complete_saved_states"] == 514 and recipe["score_window_readouts"] == 256
    assert recipe["new_gradient_pair_exposures"] == 262144
    assert recipe["distinct_gradient_pairs"] == 131072 and recipe["endpoint_history_pairs_per_arm"] == 393216
    assert recipe["inherited_fitting_pairs_per_arm"] == recipe["inherited_gradient_pairs_per_arm"] == 131072
    assert recipe["learning_rates"] == {"mixed": .01, "prior_gru": .001}
    assert recipe["new_baseline_evaluations"] == 0 and recipe["context"] is True
    assert recipe["torch_threads"] == 1 and recipe["dtype"] == "float64" and recipe["device"] == "cpu"
    assert "none" in recipe["quality_gate"] and "no route freezing" in recipe["support_scope"]
    for key in ("new_model_count_fit", "adaptive_selection", "heldout_test_evaluated", "reserved_development_evaluated",
                "physical_fpga_execution", "paid_execution_authorized", "recipe_promoted"):
        assert recipe[key] is False


@pytest.mark.parametrize("indices,partition", [((152,), "validation"), ((87,), "validation"),
                                               ((864,), "validation"), ((0,), "test"),
                                               ((1024,), "train"), ((True,), "train"), ((), "train")])
def test_data_fences_fail_before_any_corpus_access(study, monkeypatch, indices, partition):
    monkeypatch.setattr(study.pilot, "window", lambda *args: pytest.fail("unapproved data read"))
    with pytest.raises(ValueError):
        study.batch_data(b"synthetic", {}, indices, partition)


def test_private_validation_token_is_a_deep_snapshot(study):
    binding, sources, previous = ({"nested": [i]} for i in range(3))
    expected = copy.deepcopy([binding, sources, previous])
    token = study._Validation(binding, sources, previous, object())
    for value in (binding, sources, previous):
        value["nested"].append("mutated")
    assert [token.binding, token.sources, token.endpoint_manifest] == expected


@pytest.fixture
def audit_capsule(study, tmp_path, monkeypatch):
    root, capsule = tmp_path / "prior_endpoint", tmp_path / "prior_audit"
    root.mkdir()
    capsule.mkdir()
    public, receipt = tmp_path / "endpoint.json", tmp_path / "audit.json"
    write_json(root / "artifact_manifest.json", {"synthetic": True})
    write_json(public, {"complete": True, "scope": "synthetic endpoint fixture"})
    sources = {"synthetic_ancestor": "a" * 64}
    monkeypatch.setattr(study.endpoint, "source_identity", lambda: dict(sources))
    declaration = capsule / "declaration.md"
    declaration.write_text("Synthetic successful forward audit; no scientific evidence.\n")
    value = {"schema": study.endpoint.SCHEMA + ".audit", "complete": True,
             "independent_forward_recompute_completed": True, "run_directory": str(root), "record": str(public),
             "source_hashes": sources, "producer_public_sha256": study.old.digest(public.read_bytes()),
             "producer_seal_sha256": study.old.digest((root / "artifact_manifest.json").read_bytes()),
             "deadline_seconds": 1800, "independent_dense_tropical_forwards": 256,
             "independent_expanded_gru_forwards": 256, "supplemental_guarded_forwards": 256,
             "model_window_readouts": 512, "baseline_window_readouts": 128,
             "source_input_raw_seal_public_bytes_unchanged": True, "new_gradients": 0, "optimizer_steps": 0,
             "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_model_count_fit": False,
             "new_development_evaluations": 0, "heldout_test_evaluated": False,
             "reserved_development_evaluated": False, "physical_fpga_execution": False,
             "paid_execution_authorized": False, "recipe_promoted": False,
             "verification_elapsed_utc_seconds": 100., "verification_elapsed_monotonic_seconds": 100.,
             "audit_directory": str(capsule), "runtime": study.parent.runtime(),
             "started_at_utc": "2026-09-14 00:00:00 UTC"}
    manifest = {key: value[key] for key in ("run_directory", "record", "source_hashes", "producer_public_sha256",
                "producer_seal_sha256", "runtime", "started_at_utc", "deadline_seconds")}
    manifest.update(schema=study.endpoint.SCHEMA, complete=False, mode="independent_forward_audit",
                    audit_directory=str(capsule), audit_record=str(receipt),
                    declaration_sha256=study.old.digest(declaration.read_bytes()))
    write_json(capsule / "audit_manifest.json", manifest)
    value["audit_manifest_sha256"] = study.old.digest((capsule / "audit_manifest.json").read_bytes())

    def reseal():
        write_json(capsule / "verification.json", value)
        write_json(receipt, value)
        monkeypatch.setattr(study, "TRUST", {**study.TRUST, "endpoint_public": study.old.digest(public.read_bytes()),
                                           "endpoint_audit": study.old.digest(receipt.read_bytes())})

    reseal()
    return SimpleNamespace(root=root, capsule=capsule, public=public, receipt=receipt, value=value, reseal=reseal)


def test_successful_pinned_endpoint_audit_admission_is_file_only(study, audit_capsule, monkeypatch):
    for name in ("check", "execute", "audit", "model", "collect_scores"):
        monkeypatch.setattr(study.endpoint, name, lambda *args, **kwargs: pytest.fail("old endpoint numerical replay"))
    value = audit_capsule
    result, identities = study._audit_capsule(value.receipt, value.root, value.public)
    assert result == value.value
    assert set(identities["files"]) == {"audit_manifest.json", "declaration.md", "verification.json"}
    assert identities["receipt"] == study.old.file_identity(value.receipt)


@pytest.mark.parametrize("target", ["public", "receipt", "audit_manifest.json", "verification.json",
                                     "declaration.md", "failure.json", "symlink"])
def test_capsule_requires_hard_pins_exact_private_bytes_and_no_extra_files(study, audit_capsule, target):
    value = audit_capsule
    if target == "symlink":
        path = value.capsule / "declaration.md"
        other = value.capsule.parent / "other.md"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    else:
        path = getattr(value, target) if target in ("public", "receipt") else value.capsule / target
        path.write_bytes((path.read_bytes() if path.exists() else b"{}") + b"\n")
    with pytest.raises(ValueError):
        study._audit_capsule(value.receipt, value.root, value.public)


@pytest.mark.parametrize("field,bad", [("complete", 1), ("independent_forward_recompute_completed", False),
                                       ("model_window_readouts", 511), ("new_training_updates", False),
                                       ("verification_elapsed_utc_seconds", 1800.),
                                       ("verification_elapsed_monotonic_seconds", -1.)])
def test_resealed_incomplete_or_nonliteral_audit_does_not_authorize_continuation(study, audit_capsule, field, bad):
    value = audit_capsule
    value.value[field] = bad
    value.reseal()
    with pytest.raises(ValueError):
        study._audit_capsule(value.receipt, value.root, value.public)


@pytest.mark.parametrize("arm", ["mixed", "prior_gru"])
def test_independent_query_compares_but_returns_saved_gradient_for_exact_adam_replay(
        study, inherited_state, tmp_path, monkeypatch, arm):
    previous = inherited_state.entries[arm]
    cell = models(study.helper)[arm]
    optimizer = study.restore_optimizer(cell, previous.path, arm)
    before = optimizer_arrays(cell, optimizer)
    output = tmp_path / "query"
    output.mkdir()
    storage = study.Storage(output, SimpleNamespace(check=lambda: None))
    tokens = torch.tensor([[0, 1, 0, 2], [1, 3, 1, 0], [2, 0, 2, 3], [3, 2, 3, 1]], dtype=torch.int64)
    targets = (tokens.roll(-1, 1) + 1) % 4
    indices = study.training_order()[:4]
    produced, row = study.query_batch(output, 0, cell, tokens, targets, indices, write=True, storage=storage)
    calls = []
    if arm == "mixed":
        original = study.helper.gradient

        def oracle(*args, **kwargs):
            values = original(*args, **kwargs)
            if kwargs.get("independent"):
                calls.append(args[4])
                return {name: value + 1e-13 for name, value in values.items()}
            return values

        monkeypatch.setattr(study.helper, "gradient", oracle)
    else:
        original = study.parent.parent._gru_query

        def oracle(*args, **kwargs):
            logits, values = original(*args, **kwargs)
            if kwargs.get("independent"):
                calls.append(True)
                values = {name: value + 1e-13 for name, value in values.items()}
            return logits, values

        monkeypatch.setattr(study.parent.parent, "_gru_query", oracle)
    actual, checked = study.query_batch(output, 0, cell, tokens, targets, indices, recompute=True)
    study.guard.byte_equal(actual, produced, "independent tolerance cannot replace exact saved G")
    study.guard.byte_equal(optimizer_arrays(cell, optimizer), before, "queries leave all Adam state untouched")
    assert checked == row and row["window_indices"] == indices
    assert calls == ([16] * 4 if arm == "mixed" else [True])


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Real tiny two-arm replay; only data geometry and ancestry are synthetic."""
    study = importlib.import_module("manuscript.tools.continue_fpga_lm_development")
    patch = pytest.MonkeyPatch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    temporary = tmp_path_factory.mktemp("continuation-two-arm")
    source, corpus, declaration = (temporary / name for name in ("source.txt", "corpus.txt", "PLAN.md"))
    source.write_text("Synthetic source identity, not executable scientific code.\n")
    corpus.write_bytes(b"Synthetic bytes; all real corpus selectors are blocked.\n")
    declaration.write_text("Synthetic continuation fixture, not a scientific study.\n")
    for module, values in (
            (study, {"TRAIN_WINDOWS": tuple(range(8)), "SCORE_TRAIN_WINDOWS": (0, 4), "DEV_WINDOWS": (88, 89),
                     "UPDATES": 2, "SEQUENCE": 4, "PLAN": declaration}),
            (study.parent, {"TRAIN_WINDOWS": tuple(range(8)), "DEV_WINDOWS": (88, 89), "SEQUENCE": 4, "VOCAB": 4}),
            (study.endpoint, {"TRAIN_WINDOWS": (0, 4), "SEQUENCE": 4, "VOCAB": 4})):
        for name, value in values.items():
            patch.setattr(module, name, value)
    patch.setattr(study.parent, "initial_models", lambda binding: models(study.helper))
    patch.setattr(study.pilot, "window", lambda *a, **k: pytest.fail("actual corpus selector forbidden"))
    reads = []

    def batch(raw, dataset, indices, partition="train"):
        assert raw == corpus.read_bytes() and dataset == {"path": str(corpus)}
        assert partition in ("train", "validation")
        assert all(index in (study.TRAIN_WINDOWS if partition == "train" else study.DEV_WINDOWS) for index in indices)
        reads.extend((partition, index) for index in indices)
        tokens = torch.stack([(torch.arange(4, dtype=torch.int64) + index % 4) % 4 for index in indices])
        targets = torch.stack([(row.roll(-1) + index // 4 % 3) % 4 for row, index in zip(tokens, indices)])
        return tokens, targets

    patch.setattr(study, "batch_data", batch)
    patch.setattr(study.parent, "batch_data", batch)

    def endpoint_batch(raw, dataset, index):
        assert index in study.SCORE_TRAIN_WINDOWS
        return tuple(value[0] for value in batch(raw, dataset, (index,)))

    patch.setattr(study.endpoint, "batch_data", endpoint_batch)
    prior_root, endpoint_root = temporary / "old_development", temporary / "old_endpoint"
    prior_root.mkdir()
    endpoint_root.mkdir()
    dataset = {"path": str(corpus)}
    endpoint_binding = {"prior_run": str(prior_root), "parent_binding": {"synthetic": True}, "dataset": dataset}
    old_cells, endpoint_cells, initial = [], [], {}
    storage = study.parent.Storage(endpoint_root, SimpleNamespace(check=lambda: None))
    development_storage = study.parent.Storage(prior_root, SimpleNamespace(check=lambda: None))
    for arm in study.ARMS:
        checkpoint(study, prior_root / f"{arm}_seed_0", arm)
        cell = study.endpoint.model(endpoint_binding, arm, 0)
        directory = endpoint_root / f"{arm}_seed_0"
        directory.mkdir()
        train = study.endpoint.collect_scores(directory, cell, corpus.read_bytes(), dataset,
                                              write=True, storage=storage)
        with torch.no_grad():
            dev = study.parent.collect_scores(prior_root / f"{arm}_seed_0", "final", cell,
                                               corpus.read_bytes(), dataset, True, write=True,
                                               storage=development_storage)
        old_cells.append({"arm": arm, "order_seed": 0, "final_scores": dev, "final_mean": study.parent._mean(dev)})
        development = {"per_window": dev, "mean": study.parent._mean(dev), "new_forward_evaluations": 0}
        endpoint_cells.append({"arm": arm, "order_seed": 0, "complete": True,
                               "train_scores": train, "train_mean": study.parent._mean(train),
                               "development": development, "source_state": cell["endpoint_source"]})
        initial[arm] = {"synthetic_original_initial_evidence": True}
    baseline = {"synthetic_fixed_baseline": {"ce": 3.}}
    write_json(prior_root / "summary.json", {"cells": old_cells, "baselines": baseline})
    write_json(endpoint_root / "summary.json", {"cells": endpoint_cells, "initial": initial, "baselines": baseline})
    write_json(prior_root / "manifest.json", {"schema": study.parent.SCHEMA})
    write_json(endpoint_root / "manifest.json", {"schema": study.endpoint.SCHEMA})
    binding = {"prior_run": str(endpoint_root), "prior_record": str(corpus), "prior_audit": str(corpus),
               "endpoint_binding": endpoint_binding, "dataset": dataset, "trust": dict(study.TRUST)}
    protected = [study.old.file_identity(path) for root in (prior_root, endpoint_root) for path in root.rglob("*") if path.is_file()]
    protected.extend(study.old.file_identity(path) for path in (source, corpus))
    sources = {"synthetic_source": study.old.digest(source.read_bytes())}
    patch.setattr(study, "source_identity", lambda: {"synthetic_source": study.old.digest(source.read_bytes())})
    validation = study._Validation(binding, sources, {}, object())
    binds = []

    def bind(config):
        binds.append(True)
        return copy.deepcopy(binding), corpus.read_bytes(), validation

    def frozen(manifest, token):
        assert isinstance(token, study._Validation)
        study.equal(manifest["schema"], study.SCHEMA, "synthetic schema")
        study.equal(manifest["recipe"], study.recipe(), "synthetic fixed protocol")
        study.equal(manifest["binding"], token.binding, "synthetic canonical binding")
        study.equal(token.binding, binding, "synthetic original immutable binding")
        study.equal(manifest["source_hashes"], token.sources, "synthetic source closure")
        study.equal(token.sources, study.source_identity(), "synthetic source bytes")
        for descriptor in protected:
            study.equal(study.old.file_identity(descriptor["path"]), descriptor, "synthetic original input bytes")

    patch.setattr(study, "bind_inputs", bind)
    patch.setattr(study, "frozen", frozen)
    patch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=32 * 2**30))
    for module, names in ((study.parent, ("run", "execute", "check", "audit", "baselines")),
                          (study.endpoint, ("run", "execute", "check", "audit", "baselines"))):
        for name in names:
            patch.setattr(module, name, lambda *a, **k: pytest.fail("old trajectory/count-fitting execution"))
    calls = {"guarded": 0, "dense": 0, "manual_g": 0, "independent_g": 0, "gru": 0, "independent_gru": 0}
    original_guard, original_forward = study.guard.guarded.forward, study.helper.forward
    original_gradient, original_gru = study.helper.gradient, study.helper.FittedPriorGRU.forward

    def guarded(*args, **kwargs):
        calls["guarded"] += 1
        return original_guard(*args, **kwargs)

    def forward(*args, **kwargs):
        if kwargs.get("independent"):
            calls["dense"] += 1
        return original_forward(*args, **kwargs)

    def gradient(*args, **kwargs):
        calls["independent_g" if kwargs.get("independent") else "manual_g"] += 1
        return original_gradient(*args, **kwargs)

    def gru(self, *args, **kwargs):
        calls["independent_gru" if kwargs.get("independent") else "gru"] += 1
        return original_gru(self, *args, **kwargs)

    patch.setattr(study.guard.guarded, "forward", guarded)
    patch.setattr(study.helper, "forward", forward)
    patch.setattr(study.helper, "gradient", gradient)
    patch.setattr(study.helper.FittedPriorGRU, "forward", gru)
    reads.clear()
    config = SimpleNamespace(prior_run=endpoint_root, prior_record=corpus, prior_audit=corpus,
                             corpus=corpus, output=temporary / "run", record=temporary / "public.json")
    try:
        result = study.run(config)
        yield SimpleNamespace(study=study, temporary=temporary, config=config, root=config.output,
                              record=config.record, result=result, binding=binding, validation=validation,
                              source=source, corpus=corpus, prior_root=prior_root, endpoint_root=endpoint_root,
                              protected=protected, calls=calls, reads=reads, binds=binds)
    finally:
        patch.undo()
        torch.set_num_threads(previous_threads)


def test_exact_production_inventory_has_all_new_queries_absolute_states_and_endpoints(study):
    names = study.expected_artifacts()
    assert len(names) == 1802
    assert sum("/query_" in name for name in names) == 512
    assert sum("/state_" in name and name.endswith(".npz") for name in names) == 514
    assert sum("/state_" in name and name.endswith(".json") for name in names) == 514
    assert sum("/score_" in name for name in names) == 256
    assert not any("state_000000" in name or "initial" in name or "baseline" in name for name in names)


def test_real_two_arm_producer_replays_inherited_adam_and_only_new_readouts(synthetic):
    value, study = synthetic, synthetic.study
    result = value.result
    assert result["complete"] is True and result["completed_cells"] == 2
    assert result["full_batch_gradient_queries"] == result["optimizer_commits"] == 4
    assert result["complete_saved_states"] == 6 and result["score_window_readouts"] == 8
    assert result["new_gradient_pair_exposures"] == 64 and result["distinct_gradient_pairs"] == 32
    assert result["new_fitting_pair_exposures"] == result["new_baseline_evaluations"] == 0
    assert [cell["arm"] for cell in result["cells"]] == ["mixed", "prior_gru"]
    assert value.calls == {"guarded": 12, "dense": 0, "manual_g": 8, "independent_g": 0,
                           "gru": 6, "independent_gru": 0}
    for cell in result["cells"]:
        assert cell["start_step"] == 256 and cell["final_step"] == 258
        assert cell["training_order"] == study.training_order()
        assert [index for row in cell["query_rows"] for index in row["window_indices"]] == study.training_order()
        directory = value.root / cell["arm"]
        source = study.load(cell["source_state"]["file"]["path"])
        study.guard.byte_equal(study.load(directory / "state_000256.npz"), source, "actual full inherited start")
        previous = source
        for local in range(2):
            row = cell["query_rows"][local]
            assert row["absolute_step_before"] == 256 + local and row["absolute_step_after"] == 257 + local
            assert row["saved_gradient_adam_replay"] is True
            arrays = study.load(directory / f"query_{local:06d}.npz")
            gradients = {name[2:]: val for name, val in arrays.items() if name.startswith("g_")}
            after = study.load(directory / f"state_{257 + local:06d}.npz")
            assert_next_adam_formula(previous, after, gradients, .01 if cell["arm"] == "mixed" else .001, 257 + local)
            previous = after
        assert [row["window"] for row in cell["after"]["train"]["per_window"]] == [0, 4]
        assert [row["window"] for row in cell["after"]["development"]["per_window"]] == [88, 89]
        assert cell["development_minus_sampled_train_change"] == pytest.approx(
            cell["improvements"]["train"]["ce"] - cell["improvements"]["development"]["ce"], abs=1e-12)
    assert set(value.reads) <= {("train", i) for i in range(8)} | {("validation", i) for i in (88, 89)}


def test_ordinary_check_has_no_new_model_calls(synthetic):
    value = synthetic
    before = dict(value.calls)
    assert value.study.check(value.root, value.record) == value.result
    assert value.calls == before


def test_full_independent_two_arm_audit_checks_all_new_gradients_states_and_scores(synthetic, tmp_path):
    value, study = synthetic, synthetic.study
    before = dict(value.calls)
    original = tree_bytes(value.root)
    public = value.record.read_bytes()
    config = SimpleNamespace(**vars(value.config), audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    result = study.audit(config)
    assert result["complete"] is True and result["independent_recompute_completed"] is True
    delta = {key: value.calls[key] - before[key] for key in before}
    assert delta == {"guarded": 12, "dense": 12, "manual_g": 0, "independent_g": 8,
                     "gru": 0, "independent_gru": 6}
    assert result["full_batch_gradients_verified"] == 4 and result["complete_saved_states_verified"] == 6
    assert result["optimizer_commits_replayed"] == 4 and result["score_window_readouts"] == 8
    assert {p.name for p in config.audit_output.iterdir()} == {"audit_manifest.json", "declaration.md", "verification.json"}
    assert (config.audit_output / "verification.json").read_bytes() == config.audit_record.read_bytes()
    assert tree_bytes(value.root) == original and value.record.read_bytes() == public


def test_off_support_state_changes_are_reported_not_rejected(study, inherited_state):
    previous = inherited_state.entries["mixed"]
    cell = models(study.helper)["mixed"]
    optimizer = study.restore_optimizer(cell, previous.path, "mixed")
    reference = study.support_reference(cell)
    initial = study.support_state(cell, optimizer, reference)
    for key in ("MP", "W_HEAD"):
        assert initial[key]["resumed_ambiguous_rows"] == 0
        assert initial[key]["inactive_parameters_different_from_sentinel"] == 0
        index = tuple(reference[key]["inactive"].nonzero()[0])
        with torch.no_grad():
            cell["parameters"][key][index] += 1
            optimizer.state[cell["parameters"][key]]["exp_avg"][index] = .1
            optimizer.state[cell["parameters"][key]]["exp_avg_sq"][index] = .2
    result = study.support_state(cell, optimizer, reference)
    for key in ("MP", "W_HEAD"):
        assert result[key]["inactive_parameters_different_from_sentinel"] == 1
        assert result[key]["inactive_first_moments_nonzero"] == result[key]["inactive_second_moments_nonzero"] == 1
        assert result[key]["changed_single_active_support_rows"] == 1
    study.state_arrays(cell, optimizer, "mixed", 256)  # Valid state even if the sparse structure changed.


def test_observed_projection_head_departures_and_temporal_changes_are_distinct(study, inherited_state):
    cell = inherited_state.entries["mixed"].cell
    reference = study.support_reference(cell)
    trace = {"AD_HEAD": reference["W_HEAD"]["selected"][None, :].expand(3, -1).clone(),
             "PROJ_IDX": reference["MP"]["selected"][:, None, :].expand(-1, 3, -1).clone(),
             "TAP_IDX": torch.zeros((4, 3, 16), dtype=torch.int64)}
    after = {key: value.clone() for key, value in trace.items()}
    after["AD_HEAD"][0, 0] = (after["AD_HEAD"][0, 0] + 1) % 16
    after["PROJ_IDX"][0, 0, 0] = (after["PROJ_IDX"][0, 0, 0] + 1) % 16
    after["TAP_IDX"].fill_(1)
    selected = study.support_selections(after, reference)
    assert selected["AD_HEAD"] == {"departures": 1, "compared_entries": 12, "unclassified_entries": 0, "total_entries": 12}
    assert selected["PROJ_IDX"] == {"departures": 1, "compared_entries": 192, "unclassified_entries": 0, "total_entries": 192}
    assert study.route_changes(trace, after) == {"AD_HEAD": {"changed": 1, "total": 12},
                                                "PROJ_IDX": {"changed": 1, "total": 192},
                                                "TAP_IDX": {"changed": 192, "total": 192}}
    ambiguous = copy.deepcopy(reference)
    ambiguous["MP"]["selected"][0, 0] = -1
    assert study.support_selections(after, ambiguous)["PROJ_IDX"]["unclassified_entries"] == 3


def test_narrowing_paired_gap_cannot_masquerade_as_tropical_improvement(study, monkeypatch):
    monkeypatch.setattr(study, "SCORE_TRAIN_WINDOWS", (0,))
    monkeypatch.setattr(study, "DEV_WINDOWS", (0,))
    cells = []
    for arm, before, after in (("mixed", 3., 3.1), ("prior_gru", 2., 2.5)):
        def cohort(value):
            return {"mean": {"ce": value, "accuracy": .25}, "per_window": [{"window": 0, "ce": value, "accuracy": .25}]}
        cells.append({"arm": arm, "complete": True, "numerics_pass": True,
                      "before": {key: cohort(before) for key in ("train", "development")},
                      "after": {key: cohort(after) for key in ("train", "development")}})
    result = study.comparisons(cells)
    for row in result.values():
        assert row["deficit_narrowed"] is True and row["paired_deficit_change"] == pytest.approx(-.4)
        assert row["tropical_ce_improved"] is False and row["gru_ce_improved"] is False


@pytest.mark.parametrize("target", ["gradient", "route", "moment"])
def test_raw_gradient_route_or_moment_corruption_fails_numerical_reconstruction(synthetic, target):
    value, study = synthetic, synthetic.study
    path = value.root / "mixed" / ("state_000257.npz" if target == "moment" else "query_000000.npz")
    original = path.read_bytes()
    arrays = study.load(path)
    if target == "gradient":
        arrays["g_EMBED"].reshape(-1)[0] += .1
    elif target == "moment":
        arrays["exp_avg/EMBED"].reshape(-1)[0] += .1
    else:
        key = next(key for key in arrays if key.endswith("/AD_HEAD"))
        arrays[key].reshape(-1)[0] = (arrays[key].reshape(-1)[0] + 1) % 16
    path.unlink()
    study.parent.parent.save(path, arrays)
    try:
        with pytest.raises(ValueError):
            study.execute(study.read(value.root / "manifest.json"), value.root, value.corpus.read_bytes(), recompute=True)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("target", ["source", "input", "inherited_state", "public", "seal"])
def test_source_input_and_publication_mutations_fail_before_new_model_calls(synthetic, target):
    value, study = synthetic, synthetic.study
    path = {"source": value.source, "input": value.corpus, "public": value.record,
            "seal": value.root / "artifact_manifest.json",
            "inherited_state": value.prior_root / "mixed_seed_0/state_000256.npz"}[target]
    original, before = path.read_bytes(), dict(value.calls)
    if target == "public":
        # An unpinned producer record is compared semantically; exact byte
        # identity is enforced during publication and by later audit pins.
        published = study.read(path)
        published["complete"] = False
        write_json(path, published)
    else:
        path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError):
            study.check(value.root, value.record)
    finally:
        path.write_bytes(original)
    assert value.calls == before


def mode_config(value, temporary, mode):
    if mode == "producer":
        config = SimpleNamespace(**{**vars(value.config), "output": temporary / "run", "record": temporary / "public.json"})
        return config, value.study.run, config.output, config.record
    config = SimpleNamespace(**vars(value.config), audit_output=temporary / "audit", audit_record=temporary / "audit.json")
    return config, value.study.audit, config.audit_output, config.audit_record


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt, TimeoutError])
def test_failures_interruptions_and_timeouts_retain_partial_work_not_success(synthetic, tmp_path, monkeypatch, mode, error):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)

    def fail(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0 and torch.get_num_threads() == 1
        raise error("synthetic bounded interruption")

    monkeypatch.setattr(study, "execute" if mode == "producer" else "_check", fail)
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(error):
        invoke(config)
    failure = study.read(output / "failure.json")
    assert failure["complete"] is False and failure["recipe_promoted"] is False
    assert failure["stage"] == ("fixed_second_pass" if mode == "producer" else "independent_numerical_replay")
    assert not target.exists()
    assert signal.getsignal(signal.SIGALRM) == handler and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("replacement", ["owned_corrupted", "foreign"])
def test_publication_failure_retracts_only_owned_inode_and_preserves_evidence(synthetic, tmp_path, monkeypatch, mode, replacement):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.result)
    original = study.frozen
    payloads = []

    def frozen(*args):
        original(*args)
        if target.exists():
            payload = target.read_bytes()
            if replacement == "foreign":
                target.unlink()
                target.write_bytes(payload)
            else:
                payload += b"\n"
                target.write_bytes(payload)
            payloads.append(payload)
            raise ValueError("synthetic publication postcheck failure")

    monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError, match="postcheck"):
        invoke(config)
    failure = study.read(output / "failure.json")
    assert failure["complete"] is False and target.exists() is (replacement == "foreign")
    if replacement == "owned_corrupted":
        rejected = output / "rejected_public.json"
        assert rejected.read_bytes() == payloads[0]
        assert failure["retained_rejected_public"] == study.old.file_identity(rejected)
    else:
        assert target.read_bytes() == payloads[0]


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("case", ["existing_output", "existing_public", "nested_prior"])
def test_exclusive_outputs_preserve_old_evidence_without_partial_writes(synthetic, tmp_path, mode, case):
    value = synthetic
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if case == "existing_output":
        output.mkdir()
        protected = output / "user.txt"
        protected.write_bytes(b"original user bytes")
    elif case == "existing_public":
        protected = target
        target.write_bytes(b"existing publication")
    else:
        output = value.endpoint_root / "unapproved_child"
        setattr(config, "output" if mode == "producer" else "audit_output", output)
        protected = value.endpoint_root / "manifest.json"
    before = protected.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        invoke(config)
    assert protected.read_bytes() == before and not (output / "failure.json").exists()
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_producer_rejects_resealed_bytes_changed_after_ordinary_check(synthetic, tmp_path, monkeypatch):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, "producer")
    original = study._check

    def check(*args):
        result = original(*args)
        path = output / "artifact_manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(study, "_check", check)
    with pytest.raises((ValueError, RuntimeError)):
        invoke(config)
    assert not target.exists() and (output / "failure.json").is_file()


@pytest.mark.parametrize("mode", ["producer", "audit"])
def test_successful_cleanup_is_not_repeated_outside_failure_boundary(synthetic, tmp_path, monkeypatch, mode):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.result)
    original = study.parent._Publication.cleanup
    calls = []

    def cleanup(self):
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("unexpected duplicate cleanup")
        return original(self)

    monkeypatch.setattr(study.parent._Publication, "cleanup", cleanup)
    assert invoke(config)["complete"] is True and target.is_file()
    assert calls == [True] and not (output / "failure.json").exists()
