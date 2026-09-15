"""Synthetic endpoint-only scoring; no scientific corpus or training calls."""
from __future__ import annotations

import importlib
import json
import copy
import math
import signal
from types import SimpleNamespace

import pytest
import torch

from test_fpga_lm_development import models


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_endpoint_fit")


def test_new_endpoint_source_closure_preserves_the_frozen_development_inputs(study):
    sources = study.source_identity()
    assert len(sources) == 71
    for name in ("manuscript/tools/diagnose_fpga_lm_endpoint_fit.py", "tests/test_fpga_lm_endpoint_fit.py"):
        assert sources[name] == study.old.digest((study.REPO / name).read_bytes())
    assert all(sources[name] == digest for name, digest in study.parent.source_identity().items())


def test_fixed_endpoint_recipe_is_sampled_training_inference_only(study):
    recipe = study.recipe()
    assert study.TRAIN_WINDOWS == tuple(range(0, 1024, 16))
    assert recipe["partition"] == "train" and recipe["context"] is True
    assert recipe["training_windows"] == list(range(0, 1024, 16))
    assert recipe["sequence"] == recipe["vocabulary"] == 128
    assert recipe["shared_initial_models"] == 2 and recipe["final_models"] == 6
    assert recipe["model_window_readouts"] == 512 and recipe["baseline_window_readouts"] == 128
    assert recipe["deadline_seconds"] == 1800 and recipe["combined_new_output_cap_bytes"] == 256 * 2**20
    assert recipe["free_reserve_bytes"] == 4 * 2**30 and recipe["torch_threads"] == 1
    assert "not full training loss" in recipe["sample_scope"]
    assert "none" in recipe["quality_gate"]
    for key in ("new_gradient_queries", "optimizer_steps", "new_training_updates",
                "new_gradient_pair_exposures", "new_development_evaluations"):
        assert type(recipe[key]) is int and recipe[key] == 0
    for key in ("new_model_count_fit", "heldout_test_evaluated", "reserved_development_evaluated",
                "physical_fpga_execution", "paid_execution_authorized", "recipe_promoted", "adaptive_selection"):
        assert recipe[key] is False


@pytest.mark.parametrize("pin", [None, "", "0" * 63, "G" * 64, 1])
def test_missing_successful_audit_pin_fails_before_reading_any_evidence(study, monkeypatch, tmp_path, pin):
    monkeypatch.setattr(study, "AUDIT_RECEIPT_SHA256", pin)
    monkeypatch.setattr(study.parent.rate.head, "_regular", lambda *a: pytest.fail("unpinned audit accessed a file"))
    with pytest.raises(ValueError, match="not yet pinned"):
        study._audit_capsule(tmp_path / "missing.json", tmp_path / "missing", tmp_path / "public.json")


@pytest.mark.parametrize("axis,value,error", [(0, 1800., TimeoutError), (1, 1800., TimeoutError),
                                               (0, -1., RuntimeError), (1, float("nan"), RuntimeError)])
def test_endpoint_deadline_checks_both_clock_domains(study, monkeypatch, axis, value, error):
    clock = study.Deadline()
    elapsed = [10., 10.]
    elapsed[axis] = value
    monkeypatch.setattr(clock, "elapsed", lambda: tuple(elapsed))
    with pytest.raises(error):
        clock.check()


def test_endpoint_timer_uses_its_own_bound_and_restores_signal(study):
    clock = study.Deadline()
    previous = signal.getsignal(signal.SIGALRM)
    try:
        clock.arm()
        assert 1700 < signal.getitimer(signal.ITIMER_REAL)[0] <= 1800
    finally:
        clock.close()
    assert not clock.watcher.is_alive()
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("case", ["cap", "free", "outside", "existing", "boolean_size"])
def test_endpoint_storage_is_exclusive_and_bounded_before_write(study, tmp_path, monkeypatch, case):
    output = tmp_path / "owned"
    output.mkdir()
    storage = study.Storage(output, SimpleNamespace(check=lambda: None))
    path = output / "new.json"
    monkeypatch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=study.FREE_RESERVE + 100))
    if case == "cap":
        storage.used = study.STORAGE_CAP - 2**20
    elif case == "outside":
        path = tmp_path / "unowned.json"
    elif case == "existing":
        path.write_bytes(b"user bytes")
    before = tree_bytes(tmp_path)
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        storage.reserve(path, 101 if case == "free" else True if case == "boolean_size" else 1)
    assert tree_bytes(tmp_path) == before


@pytest.fixture
def audit_capsule(study, tmp_path, monkeypatch):
    root, capsule = tmp_path / "prior", tmp_path / "audit"
    root.mkdir()
    capsule.mkdir()
    public, receipt = tmp_path / "prior.json", tmp_path / "audit.json"
    write_json(root / "artifact_manifest.json", {"synthetic": True})
    gates = {"development_gate_candidate": False, "recipe_promoted": False}
    write_json(public, {"gates": gates})
    sources = {"synthetic_ancestor": "a" * 64}
    monkeypatch.setattr(study.parent, "source_identity", lambda: dict(sources))
    declaration = capsule / "declaration.md"
    declaration.write_text("Synthetic completed audit; no real scientific evidence.\n")
    value = {"schema": study.parent.SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
             "run_directory": str(root), "record": str(public), "source_hashes": sources,
             "producer_public_sha256": study.old.digest(public.read_bytes()),
             "producer_seal_sha256": study.old.digest((root / "artifact_manifest.json").read_bytes()),
             "deadline_seconds": 10800, "cell_deadline_seconds": 3600,
             "independent_tropical_window_vjps": 3072, "independent_tropical_window_forwards": 3520,
             "independent_gru_batch_queries": 768, "independent_gru_single_window_scores": 448,
             "supplemental_guarded_window_forwards": 3520, "full_batch_gradients_verified": 1536,
             "complete_saved_states_verified": 1542, "optimizer_commits_replayed": 1536,
             "score_window_readouts": 896, "baseline_window_readouts": 128,
             "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
             "new_gradient_pair_exposures": 0, "new_model_count_fit": False,
             "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
             "reserved_development_evaluated": False, "physical_fpga_execution": False,
             "paid_execution_authorized": False, "recipe_promoted": False, "development_gates": gates,
             "verification_elapsed_utc_seconds": 100., "verification_elapsed_monotonic_seconds": 100.,
             "audit_directory": str(capsule), "runtime": study.parent.runtime(),
             "started_at_utc": "2026-09-14 00:00:00 UTC"}
    manifest = {key: value[key] for key in ("run_directory", "record", "source_hashes", "producer_public_sha256",
                "producer_seal_sha256", "runtime", "started_at_utc", "deadline_seconds", "cell_deadline_seconds")}
    manifest.update(schema=study.parent.SCHEMA, complete=False, mode="independent_audit", audit_directory=str(capsule),
                    audit_record=str(receipt), declaration_sha256=study.old.digest(declaration.read_bytes()))
    write_json(capsule / "audit_manifest.json", manifest)
    value["audit_manifest_sha256"] = study.old.digest((capsule / "audit_manifest.json").read_bytes())

    def reseal():
        write_json(capsule / "verification.json", value)
        write_json(receipt, value)
        monkeypatch.setattr(study, "AUDIT_RECEIPT_SHA256", study.old.digest(receipt.read_bytes()))

    reseal()
    return SimpleNamespace(root=root, capsule=capsule, public=public, receipt=receipt, value=value, reseal=reseal)


def test_complete_audit_of_failed_quality_gate_is_valid_inference_prerequisite(study, audit_capsule):
    value = audit_capsule
    receipt, identity = study._audit_capsule(value.receipt, value.root, value.public)
    assert receipt == value.value and receipt["development_gates"]["development_gate_candidate"] is False
    assert identity["receipt"] == study.old.file_identity(value.receipt)
    assert set(identity["files"]) == {"audit_manifest.json", "declaration.md", "verification.json"}


@pytest.mark.parametrize("case", ["untrusted_bytes", "manifest", "private_receipt", "declaration", "extra_failure", "missing"])
def test_audit_trust_requires_exact_complete_unchanged_capsule(study, audit_capsule, case):
    value = audit_capsule
    path = (value.receipt if case == "untrusted_bytes" else value.capsule / {
        "manifest": "audit_manifest.json", "private_receipt": "verification.json", "declaration": "declaration.md",
        "extra_failure": "failure.json", "missing": "verification.json"}[case])
    if case == "missing":
        path.unlink()
    else:
        path.write_bytes((path.read_bytes() if path.exists() else b"{}") + b"\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        study._audit_capsule(value.receipt, value.root, value.public)


@pytest.mark.parametrize("case", ["complete_int", "not_audited", "missing_state", "float_count", "expired", "nan_elapsed", "changed_gate"])
def test_resealed_incomplete_or_malformed_audit_cannot_authorize_readouts(study, audit_capsule, case):
    value = audit_capsule
    if case == "complete_int":
        value.value["complete"] = 1
    elif case == "not_audited":
        value.value["independent_recompute_completed"] = False
    elif case == "missing_state":
        value.value["complete_saved_states_verified"] = 1541
    elif case == "float_count":
        value.value["score_window_readouts"] = 896.
    elif case == "expired":
        value.value["verification_elapsed_utc_seconds"] = 10800.
    elif case == "nan_elapsed":
        value.value["verification_elapsed_monotonic_seconds"] = float("nan")
    else:
        value.value["development_gates"] = {"development_gate_candidate": True, "recipe_promoted": False}
    if case == "nan_elapsed":
        # Deliberately malformed JSON numeric value must reach the validator.
        for path in (value.receipt, value.capsule / "verification.json"):
            path.write_text(json.dumps(value.value))
        study.AUDIT_RECEIPT_SHA256 = study.old.digest(value.receipt.read_bytes())
    else:
        value.reseal()
    with pytest.raises(ValueError):
        study._audit_capsule(value.receipt, value.root, value.public)


def test_validation_proof_does_not_alias_mutable_binding(study):
    binding, sources, previous = ({"nested": [i]} for i in range(3))
    originals = copy.deepcopy((binding, sources, previous))
    proof = study._Validation(binding, sources, previous, object())
    for value in (binding, sources, previous):
        value["nested"].append("changed")
    assert (proof.binding, proof.sources, proof.parent_manifest) == originals


def checkpoint(study, path, cell, arm, step, seed=0):
    """Handcraft synthetic frozen evidence, without constructing an optimizer."""
    parameters = {key: value.detach().clone() for key, value in cell["parameters"].items()}
    if step:
        key = "EMBED" if arm == "mixed" else "head.weight"
        parameters[key] += torch.arange(parameters[key].numel(), dtype=torch.float64).reshape(parameters[key].shape).remainder(7) * (.002 + seed * .001)
    arrays = {"parameter/" + key: value for key, value in parameters.items()}
    shapes = {}
    for key, value in parameters.items():
        shapes[key] = {"shape": list(value.shape), "dtype": "torch.float64", "initialized": step != 0}
        if step:
            arrays.update({"exp_avg/" + key: torch.zeros_like(value), "exp_avg_sq/" + key: torch.zeros_like(value),
                           "step/" + key: torch.tensor(float(step), dtype=torch.float32)})
            shapes[key]["step_dtype"] = "torch.float32"
    study.parent.parent.save(path, arrays)
    group = {"lr": .01 if arm == "mixed" else .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
             "amsgrad": False, "maximize": False, "foreach": False, "fused": False, "capturable": False,
             "differentiable": False, "decoupled_weight_decay": False}
    metadata = {"schema": study.parent.SCHEMA, "arm": arm, "step": step, "file": path.name,
                "bytes": path.stat().st_size, "sha256": study.old.digest(path.read_bytes()),
                "parameter_order": list(parameters), "parameter_state": shapes, "optimizer_group": group,
                "full_state_tensor_sha256": study.tensor_hash(arrays), "parameters_sha256": study.tensor_hash(parameters),
                "initial_optimizer_state_empty": step == 0}
    write_json(path.with_suffix(".json"), metadata)
    return parameters


@pytest.mark.parametrize("index", [True, 0.0, 1, 15, 1024, -16, "16"])
def test_only_fixed_sparse_training_positions_can_be_read(study, monkeypatch, index):
    monkeypatch.setattr(study.pilot, "window", lambda *a: pytest.fail("undeclared raw data access"))
    with pytest.raises(ValueError):
        study.batch_data(b"synthetic", {}, index)


def test_window_reader_never_requests_development_or_test_partition(study, monkeypatch):
    calls = []

    def window(raw, dataset, partition, index, sequence):
        calls.append((partition, index, sequence))
        return [0] * sequence, [1] * sequence

    monkeypatch.setattr(study.pilot, "window", window)
    for index in study.TRAIN_WINDOWS:
        tokens, targets = study.batch_data(b"synthetic", {}, index)
        assert tokens.shape == targets.shape == (128,)
    assert calls == [("train", index, 128) for index in range(0, 1024, 16)]


@pytest.mark.parametrize("case", ["missing_moment", "negative_second", "wrong_step", "float_metadata_step", "wrong_rate", "parameter_shape"])
def test_endpoint_parameter_extraction_requires_complete_original_full_state(study, tmp_path, case):
    cell = models(study.helper)["mixed"]
    path = tmp_path / "state_000256.npz"
    checkpoint(study, path, cell, "mixed", 256)
    arrays, metadata = study.load(path), study.read(path.with_suffix(".json"))
    if case == "float_metadata_step":
        metadata["step"] = 256.
    elif case == "wrong_rate":
        metadata["optimizer_group"]["lr"] = .001
    else:
        if case == "missing_moment":
            del arrays["exp_avg/EMBED"]
        elif case == "negative_second":
            arrays["exp_avg_sq/EMBED"][0, 0] = -.01
        elif case == "wrong_step":
            arrays["step/EMBED"].fill_(255.)
        else:
            arrays["parameter/EMBED"] = arrays["parameter/EMBED"].reshape(-1)
        path.unlink()
        study.parent.parent.save(path, arrays)
        metadata.update(bytes=path.stat().st_size, sha256=study.old.digest(path.read_bytes()),
                        full_state_tensor_sha256=study.tensor_hash(arrays))
    write_json(path.with_suffix(".json"), metadata)
    with pytest.raises(ValueError):
        study._checkpoint(path, cell, "mixed", 256)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_endpoint_fit")
    patch = pytest.MonkeyPatch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    temporary = tmp_path_factory.mktemp("endpoint-forward-only")
    prior_root = temporary / "prior"
    prior_root.mkdir()
    source, corpus, declaration = (temporary / name for name in ("source.txt", "input.txt", "PLAN.md"))
    source.write_bytes(b"synthetic source proof")
    corpus.write_bytes(b"Opaque synthetic input; never read as a real corpus.")
    declaration.write_text("Synthetic endpoint inference unit test.\n")
    patch.setattr(study, "SEQUENCE", 4)
    patch.setattr(study, "VOCAB", 4)
    patch.setattr(study, "PLAN", declaration)
    sources = {"synthetic": study.old.digest(source.read_bytes())}
    patch.setattr(study, "source_identity", lambda: {"synthetic": study.old.digest(source.read_bytes())})
    patch.setattr(study.parent, "initial_models", lambda binding: models(study.helper))
    dataset = {"path": str(corpus)}
    reads = []

    def window(raw, actual_dataset, partition, index, sequence):
        assert raw == corpus.read_bytes() and actual_dataset == dataset
        assert partition == "train" and index in study.TRAIN_WINDOWS and sequence == 4
        reads.append(index)
        tokens = [(position + index // 16) % 4 for position in range(4)]
        targets = [(token + 1 + index // 64 % 2) % 4 for token in tokens]
        return tokens, targets

    patch.setattr(study.pilot, "window", window)
    summary = {"shared_initial": {"architectures": {}}, "cells": []}

    def development(arm, seed=None):
        directory = prior_root / ("shared_" + arm if seed is None else study.parent.cell_name(arm, seed))
        directory.mkdir(exist_ok=True)
        rows = []
        for index in study.parent.DEV_WINDOWS:
            path = directory / f"score_{'initial' if seed is None else 'final'}_{index:03d}_1.npz"
            path.write_bytes(f"synthetic sealed historical development payload {arm} {seed} {index}".encode())
            rows.append({"window": index, "ce": 3. if seed is None else (2.8 if arm == "mixed" else 2.6) + seed * .01,
                         "accuracy": .25, "file_sha256": study.old.digest(path.read_bytes())})
        return rows, study.parent._mean(rows)

    for arm in study.ARMS:
        rows, mean = development(arm)
        summary["shared_initial"]["architectures"][arm] = {"scores": rows, "mean": mean}
        for seed in study.ORDER_SEEDS:
            directory = prior_root / study.parent.cell_name(arm, seed)
            directory.mkdir(exist_ok=True)
            cell = models(study.helper)[arm]
            checkpoint(study, directory / "state_000000.npz", cell, arm, 0)
            checkpoint(study, directory / "state_000256.npz", cell, arm, 256, seed)
            rows, mean = development(arm, seed)
            summary["cells"].append({"arm": arm, "order_seed": seed, "final_scores": rows, "final_mean": mean})
    write_json(prior_root / "summary.json", summary)
    write_json(prior_root / "manifest.json", {"schema": study.parent.SCHEMA, "synthetic": True})
    counts = temporary / "counts.json"
    write_json(counts, {"alpha": 1, "train_indices": list(range(1024)), "training_pairs": 4096,
                        "bigram_pair_counts": [[257] * 4 for _ in range(4)]})
    binding = {"prior_run": str(prior_root), "prior_record": str(corpus), "prior_audit": str(corpus),
               "dataset": dataset, "parent_binding": {"capacity": {"identities": {"counts.json": study.old.file_identity(counts)}}}}
    proof = study._Validation(binding, sources, {}, object())
    initial_prior_bytes = tree_bytes(prior_root)
    protected = [study.old.file_identity(path) for path in (source, corpus, counts)]
    binds = []

    def bind(config):
        binds.append(True)
        return copy.deepcopy(binding), corpus.read_bytes(), proof

    def frozen(manifest, token):
        assert isinstance(token, study._Validation)
        study.equal(manifest["schema"], study.SCHEMA, "synthetic endpoint schema")
        study.equal(manifest["recipe"], study.recipe(), "synthetic endpoint recipe")
        study.equal(manifest["binding"], token.binding, "synthetic immutable ancestry")
        study.equal(token.binding, binding, "synthetic original binding")
        study.equal(manifest["source_hashes"], token.sources, "synthetic declared sources")
        study.equal(token.sources, study.source_identity(), "synthetic current source")
        study.equal(tree_bytes(prior_root), initial_prior_bytes, "synthetic full prior byte closure")
        for item in protected:
            study.equal(study.old.file_identity(item["path"]), item, "synthetic protected input")

    patch.setattr(study, "bind_inputs", bind)
    patch.setattr(study, "frozen", frozen)
    patch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=32 * 2**30))

    def forbidden(*args, **kwargs):
        pytest.fail("forward-only endpoint diagnostic called a gradient, optimizer, count fit, or old trajectory")

    for owner, names in ((study.helper, ("gradient", "make_optimizer")),
                         (study.parent, ("make_optimizer", "query_batch", "baselines", "execute", "check", "audit")),
                         (study.parent.parent, ("_gru_query",)), (torch.autograd, ("grad",)),
                         (torch.optim, ("Adam",)), (torch.Tensor, ("backward",))):
        for name in names:
            patch.setattr(owner, name, forbidden)
    calls = {"guarded": 0, "independent_dense": 0, "gru": 0, "independent_gru": 0}
    original_guard, original_dense, original_gru = study.guard.guarded.forward, study.helper.forward, study.helper.FittedPriorGRU.forward

    def guarded(*args, **kwargs):
        assert not torch.is_grad_enabled()
        calls["guarded"] += 1
        return original_guard(*args, **kwargs)

    def dense(*args, **kwargs):
        assert not torch.is_grad_enabled()
        if kwargs.get("independent"):
            calls["independent_dense"] += 1
        return original_dense(*args, **kwargs)

    def gru(self, *args, **kwargs):
        assert not torch.is_grad_enabled()
        calls["independent_gru" if kwargs.get("independent") else "gru"] += 1
        return original_gru(self, *args, **kwargs)

    patch.setattr(study.guard.guarded, "forward", guarded)
    patch.setattr(study.helper, "forward", dense)
    patch.setattr(study.helper.FittedPriorGRU, "forward", gru)
    root, record = temporary / "run", temporary / "public.json"
    config = SimpleNamespace(prior_run=prior_root, prior_record=corpus, prior_audit=corpus,
                             corpus=corpus, output=root, record=record)
    try:
        result = study.run(config)
        yield SimpleNamespace(study=study, temporary=temporary, root=root, record=record, config=config,
                              result=result, binding=binding, proof=proof, calls=calls, reads=reads, binds=binds,
                              prior_root=prior_root, prior_bytes=initial_prior_bytes, source=source, corpus=corpus)
    finally:
        patch.undo()
        torch.set_num_threads(previous_threads)


def test_all512_real_tiny_readouts_and_independent_audit_without_any_training(synthetic):
    value, study = synthetic, synthetic.study
    result = value.result
    assert result["complete"] is True and result["model_window_readouts"] == 512
    assert result["baseline_window_readouts"] == 128 and result["artifact_files_verified"] == 517
    assert result["initial_probabilities_equal"] is True
    assert result["initial_probability_max_abs_error"] <= 1e-12
    assert value.calls == {"guarded": 256, "independent_dense": 0, "gru": 256, "independent_gru": 0}
    assert set(value.reads) == set(range(0, 1024, 16))
    assert [(row["arm"], row["order_seed"]) for row in result["cells"]] == [
        (arm, seed) for seed in (0, 1, 2) for arm in ("mixed", "prior_gru")]
    for initial in result["initial"].values():
        assert initial["source_state"]["step"] == 0
        assert [row["window"] for row in initial["train_scores"]] == list(range(0, 1024, 16))
        assert initial["development"]["new_forward_evaluations"] == 0
    for row in result["cells"]:
        assert row["source_state"]["step"] == 256
        assert [score["window"] for score in row["train_scores"]] == list(range(0, 1024, 16))
        assert [score["window"] for score in row["development"]["per_window"]] == list(range(88, 152))
        initial = result["initial"][row["arm"]]
        train_gain = initial["train_mean"]["ce"] - row["train_mean"]["ce"]
        dev_gain = initial["development"]["mean"]["ce"] - row["development"]["mean"]["ce"]
        assert row["train_improvement_from_initial"] == train_gain
        assert row["development_improvement_from_initial"] == dev_gain
        assert row["development_minus_train_change"] == pytest.approx(train_gain - dev_gain, abs=1e-12)
        assert row["development_minus_train_change"] == pytest.approx(
            row["final_development_minus_train"] - row["initial_development_minus_train"], abs=1e-12)
        for descriptor in (row["source_state"]["file"], row["source_state"]["metadata"]):
            assert study.old.file_identity(descriptor["path"]) == descriptor
    for arm, aggregates in result["descriptive_order_aggregates"].items():
        rows = [row for row in result["cells"] if row["arm"] == arm]
        for key, aggregate in aggregates.items():
            values = [(row["train_mean"]["ce"] if key == "sampled_train_ce" else row["development"]["mean"]["ce"]
                       if key == "saved_development_ce" else row[key]) for row in rows]
            assert aggregate == {"mean": math.fsum(values) / 3, "minimum": min(values),
                                 "maximum": max(values), "range": max(values) - min(values)}
    for key, aggregate in result["descriptive_paired_aggregates"].items():
        values = [row[key] for row in result["paired_comparisons"]]
        assert aggregate == {"mean": math.fsum(values) / 3, "minimum": min(values),
                             "maximum": max(values), "range": max(values) - min(values)}
    before = tree_bytes(value.root), value.record.read_bytes(), dict(value.calls)
    assert study.check(value.root, value.record) == result
    assert value.calls == before[2], "ordinary checking must not issue any new model forward"
    config = SimpleNamespace(**vars(value.config), audit_output=value.temporary / "audit", audit_record=value.temporary / "audit.json")
    receipt = study.audit(config)
    assert receipt["complete"] is True and receipt["independent_forward_recompute_completed"] is True
    assert receipt["independent_dense_tropical_forwards"] == 256
    assert receipt["independent_expanded_gru_forwards"] == 256
    assert receipt["supplemental_guarded_forwards"] == 256
    assert value.calls["independent_dense"] == value.calls["independent_gru"] == 256
    assert value.calls["guarded"] == 512 and value.calls["gru"] == 256
    for actual in (result, receipt):
        for key in ("new_gradients", "optimizer_steps", "new_training_updates", "new_gradient_pair_exposures", "new_development_evaluations"):
            assert type(actual[key]) is int and actual[key] == 0
        for key in ("new_model_count_fit", "heldout_test_evaluated", "reserved_development_evaluated",
                    "physical_fpga_execution", "paid_execution_authorized", "recipe_promoted"):
            assert actual[key] is False
    assert receipt["source_input_raw_seal_public_bytes_unchanged"] is True
    assert study.read(config.audit_record) == receipt
    assert set(tree_bytes(config.audit_output)) == {"audit_manifest.json", "declaration.md", "verification.json"}
    assert (tree_bytes(value.root), value.record.read_bytes()) == before[:2]
    assert tree_bytes(value.prior_root) == value.prior_bytes


@pytest.mark.parametrize("arm", ["mixed", "prior_gru"])
def test_endpoint_state_extraction_returns_fresh_exact_parameter_copies(synthetic, arm):
    value, study = synthetic, synthetic.study
    first = study.model(value.binding, arm, 1)
    expected = {key: parameter.detach().clone() for key, parameter in first["parameters"].items()}
    with torch.no_grad():
        next(iter(first["parameters"].values())).reshape(-1)[0] += .125
    another = study.model(value.binding, arm, 1)
    study.guard.byte_equal(another["parameters"], expected, "actual frozen full endpoint parameters")
    assert all(parameter.grad is None for parameter in another["parameters"].values())
    assert tree_bytes(value.prior_root) == value.prior_bytes


@pytest.mark.parametrize("case", ["logits", "route", "plausible_counter"])
def test_independent_readout_audit_rejects_validly_encoded_forged_payloads(synthetic, case):
    value, study = synthetic, synthetic.study
    path = value.root / "initial_mixed" / "score_0000.npz"
    original = path.read_bytes()
    arrays = study.load(path)
    if case == "logits":
        arrays["logits"][0, 0] += .125
    elif case == "route":
        arrays["AD_HEAD"][0, 0] = (arrays["AD_HEAD"][0, 0] + 1) % 16
    else:
        arrays["guard_counts"][0, 2] -= 1
        arrays["guard_counts"][0, 3] += 1
    try:
        path.unlink()
        study.parent.parent.save(path, arrays)
        with pytest.raises(ValueError):
            study.collect_scores(path.parent, study.model(value.binding, "mixed"), value.corpus.read_bytes(),
                                 value.binding["dataset"], recompute=True)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("case", ["source", "prior_state", "saved_development", "extra_output", "public"])
def test_fresh_check_rejects_source_input_and_artifact_changes_without_readouts(synthetic, case):
    value, study = synthetic, synthetic.study
    path = (value.source if case == "source" else value.prior_root / "mixed_seed_1/state_000256.npz" if case == "prior_state"
            else value.prior_root / "mixed_seed_1/score_final_088_1.npz" if case == "saved_development"
            else value.root / "undeclared.bin" if case == "extra_output" else value.record)
    original = path.read_bytes() if path.exists() else None
    before = dict(value.calls)
    try:
        if case == "public":
            record = study.read(path)
            record["new_development_evaluations"] = 1
            write_json(path, record)
        else:
            path.write_bytes((original or b"") + b"changed")
        with pytest.raises(ValueError):
            study.check(value.root, value.record)
    finally:
        if original is None:
            path.unlink()
        else:
            path.write_bytes(original)
    assert value.calls == before


def mode_config(value, temporary, mode):
    if mode == "producer":
        config = SimpleNamespace(**{**vars(value.config), "output": temporary / "run", "record": temporary / "public.json"})
        return config, value.study.run, config.output, config.record
    config = SimpleNamespace(**vars(value.config), audit_output=temporary / "audit", audit_record=temporary / "audit.json")
    return config, value.study.audit, config.audit_output, config.audit_record


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt, TimeoutError])
def test_failed_forward_operation_retains_failure_without_success_publication(synthetic, tmp_path, monkeypatch, mode, failure):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)

    def fail(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0 and torch.get_num_threads() == 1
        raise failure("synthetic endpoint interruption")

    monkeypatch.setattr(study, "execute" if mode == "producer" else "_check", fail)
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(failure):
        invoke(config)
    receipt = study.read(output / "failure.json")
    assert receipt["complete"] is False and receipt["new_training_updates"] == receipt["optimizer_steps"] == 0
    assert receipt["stage"] == ("fixed_endpoint_readouts" if mode == "producer" else "independent_forward_audit")
    assert not target.exists()
    assert signal.getsignal(signal.SIGALRM) == handler and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("replacement", ["owned", "owned_corrupted", "foreign"])
def test_publication_failure_retracts_only_owned_inode_and_retains_its_bytes(synthetic, tmp_path, monkeypatch, mode, replacement):
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
            elif replacement == "owned_corrupted":
                payload += b"\n"
                target.write_bytes(payload)
            payloads.append(payload)
            raise ValueError("synthetic endpoint publication postcheck failure")

    monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError, match="postcheck"):
        invoke(config)
    failure = study.read(output / "failure.json")
    assert failure["complete"] is False
    assert target.exists() is (replacement == "foreign")
    if replacement != "foreign":
        rejected = output / "rejected_public.json"
        assert rejected.read_bytes() == payloads[0]
        assert failure["retained_rejected_public"] == study.old.file_identity(rejected)
    else:
        assert target.read_bytes() == payloads[0]
    assert not list(tmp_path.glob(".tt-development-publication-*"))


def test_producer_cannot_adopt_changed_seal_after_ordinary_check_returns(synthetic, tmp_path, monkeypatch):
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
def test_successful_cleanup_is_not_repeated_outside_retained_failure_handling(synthetic, tmp_path, monkeypatch, mode):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.result)
    original = study.parent._Publication.cleanup
    calls = []

    def cleanup(self):
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("second cleanup would bypass failure handling")
        return original(self)

    monkeypatch.setattr(study.parent._Publication, "cleanup", cleanup)
    assert invoke(config)["complete"] is True and target.is_file()
    assert calls == [True] and not (output / "failure.json").exists()


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("case", ["existing_output", "existing_public", "nested_prior"])
def test_exclusive_endpoint_paths_preserve_preexisting_evidence(synthetic, tmp_path, mode, case):
    value = synthetic
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if case == "existing_output":
        output.mkdir()
        protected = output / "user.txt"
        protected.write_bytes(b"user bytes")
    elif case == "existing_public":
        protected = target
        target.write_bytes(b"existing public record")
    else:
        output = value.prior_root / "unapproved_child"
        setattr(config, "output" if mode == "producer" else "audit_output", output)
        protected = value.prior_root / "manifest.json"
    before = protected.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        invoke(config)
    assert protected.read_bytes() == before
    assert not (output / "failure.json").exists()
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
