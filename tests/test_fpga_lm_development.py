"""Synthetic development-screen contracts; no scientific corpus or cloud runs."""
from __future__ import annotations

import importlib
import json
import math
import random
import signal
import copy
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
    result = tiny_models(helper)
    return {name: result[name] for name in ("mixed", "prior_gru")}


def check_adam_recurrences(study, directory, updates, lr):
    """Independent algebra on every saved full state, not optimizer.step()."""
    previous = study.load(directory / "state_000000.npz")
    names = [key.removeprefix("parameter/") for key in previous]
    assert all(key.startswith("parameter/") for key in previous)
    for step in range(1, updates + 1):
        current = study.load(directory / f"state_{step:06d}.npz")
        query = study.load(directory / f"query_{step - 1:06d}.npz")
        assert set(current) == {prefix + name for name in names
                                for prefix in ("parameter/", "exp_avg/", "exp_avg_sq/", "step/")}
        for name in names:
            parameter = previous["parameter/" + name]
            gradient = query["g_" + name]
            first = .9 * previous.get("exp_avg/" + name, torch.zeros_like(parameter)) + .1 * gradient
            second = .999 * previous.get("exp_avg_sq/" + name, torch.zeros_like(parameter)) + .001 * gradient.square()
            expected = parameter - (lr / (1 - .9 ** step)) * first / (
                second.sqrt() / ((1 - .999 ** step) ** .5) + 1e-8)
            for prefix, wanted in (("parameter/", expected), ("exp_avg/", first), ("exp_avg_sq/", second)):
                torch.testing.assert_close(current[prefix + name], wanted, atol=1e-12, rtol=1e-11)
            assert float(current["step/" + name]) == step
        previous = current


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.compare_fpga_lm_development")


def test_source_closure_adds_only_the_new_development_runner_and_tests(study):
    sources = study.source_identity()
    assert len(sources) == 69
    for name in ("manuscript/tools/compare_fpga_lm_development.py",
                 "tests/test_fpga_lm_development.py"):
        assert sources[name] == study.old.digest((study.REPO / name).read_bytes())
    assert all(sources[name] == digest for name, digest in study.guard.source_identity().items())
    assert study.GUARD_RECEIPT_SHA256 == "7a21df2543ece923fbdcd388ae326770db0aa0ad6623d1ff578c505ae4ae5a82"


def test_fixed_six_cell_protocol_and_accounting(study):
    recipe = study.recipe()
    assert study.ARMS == ("mixed", "prior_gru") and study.ORDER_SEEDS == (0, 1, 2)
    assert study.TRAIN_WINDOWS == tuple(range(1024)) and study.DEV_WINDOWS == tuple(range(88, 152))
    assert (study.BATCH_SIZE, study.UPDATES, study.SEQUENCE, study.VOCAB) == (4, 256, 128, 128)
    assert recipe["declared_cells"] == 6
    assert recipe["full_batch_gradient_queries"] == recipe["optimizer_commits"] == 1536
    assert recipe["complete_saved_states"] == 1542
    assert recipe["shared_initial_readouts"] == 128 and recipe["final_readouts"] == 768
    assert recipe["score_window_readouts"] == 896
    assert recipe["new_gradient_pair_exposures"] == recipe["summed_inherited_fitting_pair_exposures"] == 786432
    assert recipe["distinct_gradient_pairs"] == 131072 and recipe["new_fitting_pair_exposures"] == 0
    assert recipe["torch_threads"] == 1 and recipe["dtype"] == "float64" and recipe["device"] == "cpu"
    assert recipe["global_deadline_seconds"] == 10800 and recipe["cell_deadline_seconds"] == 3600
    assert recipe["storage_cap_bytes"] == 12 * 2**30 and recipe["free_reserve_bytes"] == 4 * 2**30
    for key in ("adaptive_extension", "early_loss_stopping", "physical_fpga_execution",
                "paid_execution_authorized", "recipe_promoted"):
        assert recipe[key] is False


def test_order_replicates_are_isolated_deterministic_complete_permutations(study):
    before = random.getstate()
    orders = []
    for seed in (0, 1, 2):
        expected = list(range(1024))
        random.Random(seed).shuffle(expected)
        assert study.training_order(seed) == expected
        assert sorted(expected) == list(range(1024))
        assert len(set(expected)) == 1024
        orders.append(expected)
    assert random.getstate() == before
    assert len({tuple(order) for order in orders}) == 3
    changed = study.training_order(0)
    changed[0] = -1
    assert study.training_order(0) == orders[0]


@pytest.mark.parametrize("seed", [True, False, 0.0, "0", -1, 3])
def test_undeclared_or_nonliteral_order_seed_is_rejected(study, seed):
    with pytest.raises(ValueError):
        study.training_order(seed)
    with pytest.raises(ValueError):
        study.cell_name("mixed", seed)


@pytest.mark.parametrize("indices,partition", [((152,), "validation"), ((87,), "validation"),
                                               ((864,), "validation"), ((0,), "test"),
                                               ((1024,), "train"), ((True,), "train"),
                                               ((0.0,), "train"), ((), "train")])
def test_reserved_windows_rejected_before_any_data_access(study, monkeypatch, indices, partition):
    monkeypatch.setattr(study.pilot, "window", lambda *args: pytest.fail("reserved corpus window read"))
    with pytest.raises(ValueError):
        study.batch_data(b"synthetic", {}, indices, partition)


def test_batch_reader_uses_declared_exact_sequence_and_aligned_pairs(study, monkeypatch):
    calls = []

    def window(raw, dataset, partition, index, sequence):
        calls.append((partition, index, sequence))
        return [index % 128] * sequence, [(index + 1) % 128] * sequence

    monkeypatch.setattr(study.pilot, "window", window)
    tokens, targets = study.batch_data(b"synthetic", {}, (12, 9, 3, 700))
    assert tokens.shape == targets.shape == (4, 128)
    assert tokens.dtype == targets.dtype == torch.int64
    assert calls == [("train", i, 128) for i in (12, 9, 3, 700)]
    assert targets[0, 0] == 13


@pytest.fixture
def baseline_fixture(study, monkeypatch):
    monkeypatch.setattr(study, "TRAIN_WINDOWS", (0, 1))
    monkeypatch.setattr(study, "DEV_WINDOWS", (88, 89))
    monkeypatch.setattr(study, "SEQUENCE", 3)
    monkeypatch.setattr(study, "VOCAB", 2)
    train = {0: ([0, 0, 1], [0, 1, 1]), 1: ([1, 1, 0], [0, 1, 1])}
    dev = {88: ([0, 1, 0], [0, 0, 1]), 89: ([1, 0, 1], [1, 1, 0])}
    calls = []

    def batch(raw, dataset, indices, partition="train"):
        table = train if partition == "train" else dev
        calls.extend((partition, index) for index in indices)
        return tuple(torch.tensor([table[index][side] for index in indices], dtype=torch.int64)
                     for side in (0, 1))

    monkeypatch.setattr(study, "batch_data", batch)
    counts = {"alpha": 1, "training_pairs": 6, "train_indices": [0, 1],
              "bigram_pair_counts": [[2, 3], [2, 3]]}
    return SimpleNamespace(counts=counts, dev=dev, calls=calls)


def test_bigram_controls_use_counts_not_rounded_logit_doubling(study, baseline_fixture):
    fixture = baseline_fixture
    result = study.baselines(b"synthetic", {}, fixture.counts)
    expected = {}
    for name, counts in (("alpha_one", [[2, 3], [2, 3]]), ("repeat_alpha_one", [[3, 5], [3, 5]])):
        losses = []
        for index, (tokens, targets) in fixture.dev.items():
            value = sum(-math.log(counts[token][target] / sum(counts[token]))
                        for token, target in zip(tokens, targets)) / len(tokens)
            losses.append(value)
            row = next(row for row in result[name]["per_window"] if row["window"] == index)
            assert row["ce"] == pytest.approx(value, abs=1e-14)
        expected[name] = sum(losses) / len(losses)
        assert result[name]["mean"]["ce"] == pytest.approx(expected[name], abs=1e-14)
    assert result["stronger_ce"] == min(result[name]["mean"]["ce"] for name in expected)
    assert result["new_model_fitting_pair_exposures"] == 0
    assert set(fixture.calls) == {("train", 0), ("train", 1), ("validation", 88), ("validation", 89)}


@pytest.mark.parametrize("case", ["alpha_bool", "budget_float", "count_bool", "wrong_same_total_counts", "wrong_order"])
def test_inherited_count_table_must_match_full_actual_training_counts(study, baseline_fixture, case):
    counts = baseline_fixture.counts
    if case == "alpha_bool":
        counts["alpha"] = True
    elif case == "budget_float":
        counts["training_pairs"] = 6.0
    elif case == "count_bool":
        counts["bigram_pair_counts"][0][0] = True
    elif case == "wrong_same_total_counts":
        counts["bigram_pair_counts"][0] = [3, 2]
    else:
        counts["train_indices"].reverse()
    with pytest.raises(ValueError):
        study.baselines(b"synthetic", {}, counts)


@pytest.mark.parametrize("axis,amount,error", [(0, 10800, TimeoutError), (1, 10800, TimeoutError),
                                               (0, -1, RuntimeError), (1, float("nan"), RuntimeError)])
def test_global_deadline_checks_both_clocks_without_sleep(study, monkeypatch, axis, amount, error):
    clock = study.Deadline()
    elapsed = [10., 10.]
    elapsed[axis] = amount
    monkeypatch.setattr(clock, "elapsed", lambda: tuple(elapsed))
    with pytest.raises(error):
        clock.check()


@pytest.mark.parametrize("axis", [0, 1])
def test_cell_deadline_remains_bounded_under_global_budget(study, monkeypatch, axis):
    clock = study.Deadline()
    monkeypatch.setattr(clock, "elapsed", lambda: (4000., 4000.))
    monkeypatch.setattr(study.time, "time", lambda: 5000.)
    monkeypatch.setattr(study.time, "monotonic", lambda: 5000.)
    clock.cell_started = tuple(1400. if i == axis else 2000. for i in range(2))
    with pytest.raises(TimeoutError, match="cell"):
        clock.check()


def test_deadline_restores_signal_and_stops_owned_watcher(study):
    original = signal.getsignal(signal.SIGALRM)
    clock = study.Deadline()
    try:
        clock.arm()
        assert 3600 < signal.getitimer(signal.ITIMER_REAL)[0] <= 10800
        clock.begin_cell()
        assert clock.cell_started is not None
        clock.end_cell()
        assert clock.cell_started is None
    finally:
        clock.close()
    assert signal.getsignal(signal.SIGALRM) == original
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert not clock.watcher.is_alive()


@pytest.mark.parametrize("case", ["cap", "free", "boolean_size", "existing", "expired"])
def test_storage_rejects_before_writing_or_overwriting(study, tmp_path, monkeypatch, case):
    def check():
        if case == "expired":
            raise TimeoutError("synthetic deadline")

    storage = study.Storage(tmp_path, SimpleNamespace(check=check))
    path = tmp_path / "new.json"
    monkeypatch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=study.FREE_RESERVE + 100))
    size = 101 if case == "free" else True if case == "boolean_size" else 1
    if case == "cap":
        storage.used = study.STORAGE_CAP
    if case == "existing":
        path.write_bytes(b"user bytes")
    before = tree_bytes(tmp_path)
    with pytest.raises((RuntimeError, FileExistsError, TimeoutError)):
        storage.reserve(path, size)
    assert tree_bytes(tmp_path) == before


def test_storage_counts_actual_compressed_payloads_and_json(study, tmp_path, monkeypatch):
    storage = study.Storage(tmp_path, SimpleNamespace(check=lambda: None))
    monkeypatch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=study.FREE_RESERVE + 2**20))
    storage.json(tmp_path / "metadata.json", {"synthetic": True})
    storage.arrays(tmp_path / "state.npz", {"x": torch.zeros(16, dtype=torch.float64)})
    assert storage.used == sum(path.stat().st_size for path in tmp_path.iterdir())


@pytest.fixture
def guard_capsule(study, tmp_path, monkeypatch):
    root, capsule = tmp_path / "prior", tmp_path / "guard"
    root.mkdir()
    capsule.mkdir()
    public, receipt = tmp_path / "prior.json", tmp_path / "guard.json"
    public.write_text("{}\n")
    sources = {"synthetic-old-source": "a" * 64}
    monkeypatch.setattr(study.guard, "source_identity", lambda: dict(sources))
    (capsule / "declaration.md").write_text("Synthetic guarded audit; no real experiment.\n")
    manifest = {"schema": study.guard.SCHEMA, "complete": False, "run_directory": str(root),
                "record": str(public), "audit_directory": str(capsule), "receipt": str(receipt),
                "source_hashes": sources, "deadline_seconds": 3600,
                "declaration_sha256": study.old.digest((capsule / "declaration.md").read_bytes())}
    parity = {"guarded_window_forwards": 524, "manual_window_vjps": 512,
              "full_batch_gradients_verified": 128, "optimizer_commits_replayed": 128,
              "complete_saved_states_verified": 129, "score_readouts_verified": 12}
    verification = {"synthetic": True}
    for name, value in (("audit_manifest.json", manifest), ("parity.json", parity),
                        ("verification.json", verification)):
        write_json(capsule / name, value)
    value = {"schema": study.guard.SCHEMA, "complete": True, "guarded_parity_replay_completed": True,
             "rate_run_directory": str(root), "rate_public_record": str(public), "audit_directory": str(capsule),
             "source_hashes": sources, "independent_autograd_audit_reused": True,
             "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
             "new_optimizer_trajectory_executed": False, "paid_execution_authorized": False,
             "recipe_promoted": False, "publication_postchecks_within_deadline": True,
             "deadline_seconds": 3600, "verification": verification, "parity": parity,
             "audit_manifest_sha256": study.old.digest((capsule / "audit_manifest.json").read_bytes()),
             "private_files": {path.name: study.old.file_identity(path) for path in capsule.iterdir()}}
    write_json(receipt, value)
    monkeypatch.setattr(study, "GUARD_RECEIPT_SHA256", study.old.digest(receipt.read_bytes()))
    return SimpleNamespace(root=root, capsule=capsule, public=public, receipt=receipt, value=value)


def test_successful_guard_trust_is_file_only_and_binds_exact_private_capsule(study, guard_capsule, monkeypatch):
    monkeypatch.setattr(study.guard, "replay", lambda *a, **k: pytest.fail("old guarded model replay"))
    value = guard_capsule
    result, identities = study._guard_capsule(value.receipt, value.root, value.public)
    assert result == value.value
    assert identities["receipt"] == study.old.file_identity(value.receipt)
    assert set(identities["files"]) == {"audit_manifest.json", "declaration.md", "verification.json", "parity.json"}


@pytest.mark.parametrize("target", ["receipt", "audit_manifest.json", "declaration.md", "verification.json",
                                     "parity.json", "failure.json", "symlink"])
def test_guard_trust_rejects_changed_or_extra_capsule_bytes(study, guard_capsule, target):
    value = guard_capsule
    if target == "receipt":
        value.receipt.write_bytes(value.receipt.read_bytes() + b"\n")
    elif target == "symlink":
        path = value.capsule / "declaration.md"
        other = value.capsule.parent / "declaration.md"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    else:
        path = value.capsule / target
        path.write_bytes((path.read_bytes() if path.exists() else b"{}") + b"\n")
    with pytest.raises(ValueError):
        study._guard_capsule(value.receipt, value.root, value.public)


@pytest.fixture
def original_models(study, tmp_path, monkeypatch):
    model_cells = models(study.helper)
    initial = {}
    for arm, cell in model_cells.items():
        path = tmp_path / f"{arm}_initial.npz"
        study.parent.save(path, {"parameter/" + key: value.detach() for key, value in cell["parameters"].items()})
        initial[arm] = study.old.file_identity(path)
    prior_path = tmp_path / "frozen_prior.npz"
    study.parent.save(prior_path, {"logits": model_cells["prior_gru"]["prior_logits"]})
    binding = {"rate_binding": {"prior_binding": {"synthetic": True}}, "initial_states": initial,
               "frozen_prior": study.old.file_identity(prior_path)}
    monkeypatch.setattr(study.parent, "initial_models", lambda binding: models(study.helper))
    return binding


def test_actual_original_initial_weights_and_prior_match_for_every_order(study, original_models):
    first = study.initial_models(original_models)
    for _ in range(2):
        another = study.initial_models(original_models)
        for arm in study.ARMS:
            study.guard.byte_equal(another[arm]["parameters"], first[arm]["parameters"], "paired initial state")
            for key in first[arm]["parameters"]:
                assert first[arm]["parameters"][key].data_ptr() != another[arm]["parameters"][key].data_ptr()


@pytest.mark.parametrize("target", ["mixed", "prior_gru", "frozen_prior", "extra_moments"])
def test_initial_factory_rejects_endpoint_or_prior_drift_even_with_updated_file_hash(study, original_models, target):
    descriptor = original_models["frozen_prior"] if target == "frozen_prior" else original_models["initial_states"][
        "mixed" if target == "extra_moments" else target]
    path = study.Path(descriptor["path"])
    arrays = study.load(path)
    if target == "extra_moments":
        arrays["exp_avg/EMBED"] = torch.zeros_like(arrays["parameter/EMBED"])
    else:
        next(iter(arrays.values())).reshape(-1)[0] += .125
    path.unlink()
    study.parent.save(path, arrays)
    descriptor.update(study.old.file_identity(path))
    with pytest.raises(ValueError):
        study.initial_models(original_models)


def test_validation_token_does_not_alias_mutable_working_metadata(study):
    binding, sources, previous, facts = ({"nested": [i]} for i in range(4))
    token = study._Validation(binding, sources, previous, object(), facts)
    snapshots = [copy.deepcopy(value) for value in (binding, sources, previous, facts)]
    for value in (binding, sources, previous, facts):
        value["nested"].append("changed")
    assert [token.binding, token.sources, token.rate_manifest, token.guard_facts] == snapshots


@pytest.mark.parametrize("arm", ["mixed", "prior_gru"])
def test_real_tiny_query_uses_one_full_batch_mean_and_independent_reverse_rule(study, tmp_path, monkeypatch, arm):
    cell = models(study.helper)[arm]
    tokens = torch.tensor([[0, 1, 0, 2], [1, 3, 1, 0], [2, 0, 2, 3], [3, 2, 3, 1]], dtype=torch.int64)
    targets = (tokens.roll(-1, 1) + 1) % 4
    storage = study.Storage(tmp_path, SimpleNamespace(check=lambda: None))
    seen = []
    original = study.helper.gradient

    def gradient(*args, **kwargs):
        seen.append((args[4], kwargs.get("independent", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(study.helper, "gradient", gradient)
    before = {name: value.detach().clone() for name, value in cell["parameters"].items()}
    saved, row = study.query_batch(tmp_path, 0, cell, tokens, targets, (19, 8, 12, 4), write=True, storage=storage)
    audited, audited_row = study.query_batch(tmp_path, 0, cell, tokens, targets, (19, 8, 12, 4), recompute=True)
    study.guard.byte_equal(saved, audited, "independent query returns the exact saved batch G")
    assert row == audited_row and row["window_indices"] == [19, 8, 12, 4]
    study.guard.byte_equal(cell["parameters"], before, "query never updates parameters")
    if arm == "mixed":
        assert seen == [(16, False)] * 4 + [(16, True)] * 4
    else:
        _, expected = study.parent._gru_query(cell, tokens, targets, independent=True)
        for name in saved:
            torch.testing.assert_close(saved[name], expected[name], atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("case", ["wrong_rate", "parameter_order", "boolean_step", "negative_second_moment", "missing_moment"])
def test_complete_adam_state_rejects_recipe_or_metadata_drift_before_write(study, tmp_path, case):
    cell = models(study.helper)["mixed"]
    optimizer = study.make_optimizer("mixed", cell["parameters"].values())
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
    storage = study.Storage(tmp_path, SimpleNamespace(check=lambda: None))
    with pytest.raises(ValueError):
        study.state_record(tmp_path, step, cell, optimizer, "mixed", write=True, storage=storage)
    assert not list(tmp_path.iterdir())


def test_full256_update_tiny_cell_preserves_all_queries_states_and_exact_adam_recurrence(study, tmp_path):
    """Keep the real one-pass update budget; shrink only tensor/data geometry."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cell = models(study.helper)["mixed"]
        optimizer = study.make_optimizer("mixed", cell["parameters"].values())
        storage = study.Storage(tmp_path, SimpleNamespace(check=lambda: None))
        study.state_record(tmp_path, 0, cell, optimizer, "mixed", write=True, storage=storage)
        order = study.training_order(1)
        consumed = []
        for step in range(256):
            indices = order[step * 4:(step + 1) * 4]
            tokens = torch.stack([(torch.arange(4, dtype=torch.int64) + index % 4) % 4 for index in indices])
            targets = torch.stack([(row.roll(-1) + index // 4 % 3) % 4 for row, index in zip(tokens, indices)])
            gradient, row = study.query_batch(tmp_path, step, cell, tokens, targets, indices, write=True, storage=storage)
            checked, actual = study.query_batch(tmp_path, step, cell, tokens, targets, indices, recompute=True)
            study.guard.byte_equal(gradient, checked, "full-budget independently checked gradient")
            assert row == actual
            consumed.extend(row["window_indices"])
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in cell["parameters"].items():
                parameter.grad = gradient[name].detach().clone()
            optimizer.step()
            study.state_record(tmp_path, step + 1, cell, optimizer, "mixed", write=True, storage=storage)
        assert consumed == order and sorted(consumed) == list(range(1024))
        assert len(list(tmp_path.glob("query_*.npz"))) == 256
        assert len(list(tmp_path.glob("state_*.npz"))) == len(list(tmp_path.glob("state_*.json"))) == 257
        check_adam_recurrences(study, tmp_path, 256, .01)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    study = importlib.import_module("manuscript.tools.compare_fpga_lm_development")
    patch = pytest.MonkeyPatch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    temporary = tmp_path_factory.mktemp("development-six-cell")
    source = temporary / "synthetic_source.txt"
    source.write_text("Synthetic invocation source proof; not executable research code.\n")
    corpus = temporary / "synthetic_input.txt"
    corpus.write_bytes(b"Opaque synthetic input; actual corpus/model factories are blocked.\n")
    declaration = temporary / "PLAN.md"
    declaration.write_text("Synthetic six-cell unit fixture, not a scientific study.\n")
    for name, value in {"TRAIN_WINDOWS": tuple(range(8)), "DEV_WINDOWS": (88, 89),
                        "UPDATES": 2, "SEQUENCE": 4, "VOCAB": 4, "PLAN": declaration}.items():
        patch.setattr(study, name, value)
    sources = {"synthetic_source": study.old.digest(source.read_bytes())}
    patch.setattr(study, "source_identity", lambda: {"synthetic_source": study.old.digest(source.read_bytes())})
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
    patch.setattr(study.parent, "initial_models", lambda binding: models(study.helper))
    initial = {}
    for arm, cell in models(study.helper).items():
        path = temporary / f"original_{arm}.npz"
        study.parent.save(path, {"parameter/" + key: value.detach() for key, value in cell["parameters"].items()})
        initial[arm] = study.old.file_identity(path)
    prior_path = temporary / "original_frozen_prior.npz"
    study.parent.save(prior_path, {"logits": models(study.helper)["prior_gru"]["prior_logits"]})
    tokens, targets = batch(corpus.read_bytes(), {"path": str(corpus)}, study.TRAIN_WINDOWS)
    table = torch.ones((4, 4), dtype=torch.int64)
    table += torch.bincount((tokens * 4 + targets).reshape(-1), minlength=16).reshape(4, 4)
    counts = temporary / "counts.json"
    write_json(counts, {"alpha": 1, "training_pairs": 32, "train_indices": list(range(8)),
                        "bigram_pair_counts": table.tolist()})
    binding = {"prior_run": str(temporary), "prior_record": str(corpus), "guard_receipt": str(corpus),
               "dataset": {"path": str(corpus)}, "rate_binding": {"prior_binding": {"synthetic": True}},
               "initial_states": initial, "frozen_prior": study.old.file_identity(prior_path),
               "capacity": {"identities": {"counts.json": study.old.file_identity(counts)}}}
    protected = [study.old.file_identity(path) for path in (source, corpus, counts, prior_path)] + list(initial.values())
    validation = study._Validation(binding, sources, {}, None, {})
    bind_calls = []

    def bind(config):
        bind_calls.append(True)
        return copy.deepcopy(binding), corpus.read_bytes(), validation

    def frozen(manifest, token):
        assert isinstance(token, study._Validation)
        study.equal(manifest["schema"], study.SCHEMA, "synthetic schema")
        study.equal(manifest["recipe"], study.recipe(), "synthetic fixed recipe")
        study.equal(manifest["binding"], token.binding, "synthetic complete binding")
        study.equal(token.binding, binding, "synthetic original binding")
        study.equal(manifest["source_hashes"], token.sources, "synthetic invocation source proof")
        study.equal(token.sources, study.source_identity(), "synthetic current source proof")
        for descriptor in protected:
            study.equal(study.old.file_identity(descriptor["path"]), descriptor, "synthetic immutable input")

    patch.setattr(study, "bind_inputs", bind)
    patch.setattr(study, "frozen", frozen)
    patch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=32 * 2**30))
    calls = {"guarded": 0, "independent_forward": 0, "manual_gradient": 0,
             "independent_gradient": 0, "gru": 0, "independent_gru": 0}
    original_guard, original_forward = study.guard.guarded.forward, study.helper.forward
    original_gradient, original_gru = study.helper.gradient, study.helper.FittedPriorGRU.forward

    def guarded(*args, **kwargs):
        calls["guarded"] += 1
        return original_guard(*args, **kwargs)

    def forward(*args, **kwargs):
        if kwargs.get("independent"):
            calls["independent_forward"] += 1
        return original_forward(*args, **kwargs)

    def gradient(*args, **kwargs):
        calls["independent_gradient" if kwargs.get("independent") else "manual_gradient"] += 1
        return original_gradient(*args, **kwargs)

    def gru(self, *args, **kwargs):
        calls["independent_gru" if kwargs.get("independent") else "gru"] += 1
        return original_gru(self, *args, **kwargs)

    patch.setattr(study.guard.guarded, "forward", guarded)
    patch.setattr(study.helper, "forward", forward)
    patch.setattr(study.helper, "gradient", gradient)
    patch.setattr(study.helper.FittedPriorGRU, "forward", gru)
    root, record = temporary / "run", temporary / "public.json"
    config = SimpleNamespace(prior_run=temporary, prior_record=corpus, guard_receipt=corpus,
                             corpus=corpus, output=root, record=record)
    try:
        result = study.run(config)
        yield SimpleNamespace(study=study, temporary=temporary, root=root, record=record, config=config,
                              result=result, binding=binding, validation=validation, source=source,
                              corpus=corpus, reads=reads, calls=calls, bind_calls=bind_calls)
    finally:
        patch.undo()
        torch.set_num_threads(previous_threads)


def gate_cells(study):
    return [{"arm": arm, "order_seed": seed, "complete": True, "numerics_pass": True,
             "initial_mean": {"ce": 3.1, "accuracy": .25},
             "final_mean": {"ce": 2.8 if arm == "mixed" else 2.75, "accuracy": .5},
             "context_off_mean": {"ce": 3.4, "accuracy": .2}}
            for seed in study.ORDER_SEEDS for arm in study.ARMS]


def test_production_inventory_accounts_every_query_state_and_score(study):
    names = study.expected_artifacts()
    assert len(names) == 5529
    assert sum(name.endswith(".npz") and "/query_" in name for name in names) == 1536
    assert sum(name.endswith(".npz") and "/state_" in name for name in names) == 1542
    assert sum(name.endswith(".json") and "/state_" in name for name in names) == 1542
    assert sum("/score_" in name for name in names) == 896


def test_development_gate_requires_all_order_losses_not_context_or_accuracy(study):
    cells = gate_cells(study)
    result = study.gates(cells, {"stronger_ce": 2.82})
    assert result["development_gate_candidate"] is True
    assert result["requires_successful_separate_independent_audit"] is True
    assert result["recipe_promoted"] is False and result["paid_execution_authorized"] is False
    cells[0]["final_mean"] = {"ce": 3.1, "accuracy": 1.0}
    cells[0]["context_off_mean"] = {"ce": 100., "accuracy": 0.0}
    result = study.gates(cells, {"stronger_ce": 10.})
    assert result["improves_initial_per_order"]["0"] is False
    assert result["development_gate_candidate"] is False


@pytest.mark.parametrize("margin,delta,passes", [("gru", 0., True), ("gru", 2e-12, False),
                                                ("bigram", 0., True), ("bigram", 2e-12, False)])
def test_declared_loss_margins_have_only_the_fixed_inclusive_tolerance(study, margin, delta, passes):
    cells = gate_cells(study)
    if margin == "gru":
        cells[0]["final_mean"]["ce"] = 2.8 + delta
        baseline = {"stronger_ce": 4.}
    else:
        for row in cells:
            if row["arm"] == "mixed":
                row["final_mean"]["ce"] = 2.8 + delta
            else:
                row["final_mean"]["ce"] = 3.
        baseline = {"stronger_ce": 2.82}
    result = study.gates(cells, baseline)
    assert result["development_gate_candidate"] is passes


@pytest.mark.parametrize("case", ["missing", "duplicate", "reordered", "false_complete", "float_seed", "unhealthy", "nan"])
def test_incomplete_or_malformed_cells_cannot_receive_a_quality_gate(study, case):
    cells = gate_cells(study)
    if case == "missing":
        cells.pop()
    elif case == "duplicate":
        cells[-1] = copy.deepcopy(cells[0])
    elif case == "reordered":
        cells.reverse()
    elif case == "false_complete":
        cells[0]["complete"] = 1
    elif case == "float_seed":
        cells[0]["order_seed"] = 0.
    elif case == "unhealthy":
        cells[0]["numerics_pass"] = False
    else:
        cells[0]["final_mean"]["ce"] = float("nan")
    with pytest.raises(ValueError):
        study.gates(cells, {"stronger_ce": 3.})


@pytest.mark.parametrize("case", ["valid", "source", "initial_bytes", "binding", "ancestor", "capsule"])
def test_frozen_guard_rechecks_sources_original_inputs_and_cached_ancestry(study, original_models, tmp_path, monkeypatch, case):
    source = tmp_path / "source.txt"
    source.write_bytes(b"synthetic source")
    sources = {"synthetic": study.old.digest(source.read_bytes())}
    monkeypatch.setattr(study, "source_identity", lambda: {"synthetic": study.old.digest(source.read_bytes())})
    binding = copy.deepcopy(original_models)
    binding.update(prior_run=str(tmp_path), prior_record=str(source), guard_receipt=str(source),
                   guard_capsule={"proof": "fixed"}, capacity={"proof": "fixed"}, dataset={"synthetic": True})
    calls = []

    def unchanged(*args):
        calls.append("complete source-bound ancestry")
        if case == "ancestor":
            raise ValueError("changed ancestor bytes")

    monkeypatch.setattr(study.guard, "_unchanged", unchanged)
    monkeypatch.setattr(study, "_guard_capsule", lambda *args: ({}, {"proof": "changed" if case == "capsule" else "fixed"}))
    monkeypatch.setattr(study, "_capacity", lambda dataset: {"proof": "fixed"})
    token = study._Validation(binding, sources, {}, object(), {})
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "binding": copy.deepcopy(binding),
                "source_hashes": dict(sources)}
    if case == "source":
        source.write_bytes(b"different synthetic source")
    elif case == "initial_bytes":
        path = study.Path(binding["initial_states"]["mixed"]["path"])
        path.write_bytes(path.read_bytes() + b"changed")
    elif case == "binding":
        manifest["binding"]["dataset"]["synthetic"] = False
    if case == "valid":
        study.frozen(manifest, token)
        assert calls == ["complete source-bound ancestry"]
    else:
        with pytest.raises(ValueError):
            study.frozen(manifest, token)


@pytest.mark.parametrize("case", ["gradient", "route", "plausible_guard_count", "tokens"])
def test_independent_query_check_rejects_reencoded_raw_payload_corruption(study, tmp_path, case):
    cell = models(study.helper)["mixed"]
    tokens = torch.tensor([[0, 1, 0, 2], [1, 3, 1, 0], [2, 0, 2, 3], [3, 2, 3, 1]], dtype=torch.int64)
    targets = (tokens.roll(-1, 1) + 1) % 4
    storage = study.Storage(tmp_path, SimpleNamespace(check=lambda: None))
    study.query_batch(tmp_path, 0, cell, tokens, targets, [0, 1, 2, 3], write=True, storage=storage)
    path = tmp_path / "query_000000.npz"
    arrays = study.load(path)
    if case == "gradient":
        arrays["g_EMBED"][0, 0] += .125
    elif case == "route":
        arrays["window_0/AD_HEAD"][0, 0] = (arrays["window_0/AD_HEAD"][0, 0] + 1) % 16
    elif case == "plausible_guard_count":
        arrays["guard_counts"][0, 2] -= 1
        arrays["guard_counts"][0, 3] += 1
    else:
        arrays["tokens"][0, 0] = (arrays["tokens"][0, 0] + 1) % 4
    path.unlink()
    study.parent.save(path, arrays)
    with pytest.raises(ValueError):
        study.query_batch(tmp_path, 0, cell, tokens, targets, [0, 1, 2, 3], recompute=True)


def test_real_six_cell_producer_check_and_separate_independent_audit(synthetic):
    value, study = synthetic, synthetic.study
    result = value.result
    assert result["complete"] is True and result["completed_cells"] == 6
    assert result["optimizer_commits"] == result["full_batch_gradient_queries"] == 12
    assert result["complete_saved_states"] == 18 and result["score_window_readouts"] == 28
    assert result["new_gradient_pair_exposures"] == 192 and result["distinct_gradient_pairs"] == 32
    assert result["summed_endpoint_history_pair_exposures"] == 384
    assert result["new_fitting_pair_exposures"] == 0
    assert result["shared_initial"]["initial_probabilities_equal"] is True
    assert [(row["arm"], row["order_seed"]) for row in result["cells"]] == [
        (arm, seed) for seed in (0, 1, 2) for arm in ("mixed", "prior_gru")]
    for row in result["cells"]:
        assert row["training_order"] == study.training_order(row["order_seed"])
        assert [index for batch in row["query_rows"] for index in batch["window_indices"]] == row["training_order"]
        assert row["shared_initial_parameters_sha256"] == result["shared_initial"]["architectures"][row["arm"]]["parameters_sha256"]
        check_adam_recurrences(study, value.root / row["name"], 2, .01 if row["arm"] == "mixed" else .001)
    before = tree_bytes(value.root), value.record.read_bytes(), dict(value.calls)
    bind_count = len(value.bind_calls)
    assert study.check(value.root, value.record) == result
    assert len(value.bind_calls) == bind_count + 1
    assert value.calls == before[2], "ordinary saved-gradient checking must not run any forward/reverse model"
    config = SimpleNamespace(**vars(value.config), audit_output=value.temporary / "audit",
                             audit_record=value.temporary / "audit.json")
    receipt = study.audit(config)
    assert receipt["complete"] is True and receipt["independent_recompute_completed"] is True
    assert receipt["independent_tropical_window_vjps"] == 24
    assert receipt["independent_tropical_window_forwards"] == 38
    assert receipt["supplemental_guarded_window_forwards"] == 38
    assert receipt["independent_gru_batch_queries"] == 6
    assert receipt["independent_gru_single_window_scores"] == 14
    assert receipt["full_batch_gradients_verified"] == receipt["optimizer_commits_replayed"] == 12
    assert receipt["complete_saved_states_verified"] == 18
    assert value.calls["independent_gradient"] - before[2]["independent_gradient"] == 24
    assert value.calls["independent_forward"] - before[2]["independent_forward"] == 38
    assert value.calls["guarded"] - before[2]["guarded"] == 38
    assert value.calls["independent_gru"] - before[2]["independent_gru"] == 20
    assert value.calls["manual_gradient"] == before[2]["manual_gradient"]
    assert value.calls["gru"] == before[2]["gru"]
    assert receipt["source_input_raw_seal_public_bytes_unchanged"] is True
    assert receipt["new_training_updates"] == receipt["new_gradient_pair_exposures"] == 0
    assert receipt["development_gates"] == result["gates"]
    assert receipt["old_numerical_trajectories_replayed"] is False
    assert receipt["heldout_test_evaluated"] is False and receipt["reserved_development_evaluated"] is False
    assert receipt["recipe_promoted"] is False and receipt["physical_fpga_execution"] is False
    assert study.read(config.audit_record) == receipt
    assert set(tree_bytes(config.audit_output)) == {"audit_manifest.json", "declaration.md", "verification.json"}
    assert (tree_bytes(value.root), value.record.read_bytes()) == before[:2]
    assert {index for partition, index in value.reads if partition == "validation"} == {88, 89}


def mode_config(value, temporary, mode):
    if mode == "producer":
        config = SimpleNamespace(**{**vars(value.config), "output": temporary / "run", "record": temporary / "public.json"})
        return config, value.study.run, config.output, config.record
    config = SimpleNamespace(**vars(value.config), audit_output=temporary / "audit", audit_record=temporary / "audit.json")
    return config, value.study.audit, config.audit_output, config.audit_record


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt, TimeoutError])
def test_failure_or_interruption_retains_diagnostics_and_never_publishes_success(synthetic, tmp_path, monkeypatch, mode, failure):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)

    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 1
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        raise failure("synthetic interrupted development operation")

    monkeypatch.setattr(study, "execute" if mode == "producer" else "_check", fail)
    original_handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(failure):
        invoke(config)
    receipt = study.read(output / "failure.json")
    assert receipt["complete"] is False and receipt["recipe_promoted"] is False
    assert receipt["stage"] == ("producer" if mode == "producer" else "independent_numerical_replay")
    assert not target.exists()
    assert signal.getsignal(signal.SIGALRM) == original_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("mode", ["producer", "audit"])
def test_successful_publication_cleanup_is_not_repeated_after_deadline_close(synthetic, tmp_path, monkeypatch, mode):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.result)
    original, calls = study._Publication.cleanup, []

    def cleanup(self):
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("redundant final cleanup would bypass retained failure handling")
        return original(self)

    monkeypatch.setattr(study._Publication, "cleanup", cleanup)
    assert invoke(config)["complete"] is True
    assert calls == [True] and target.is_file()
    assert not (output / "failure.json").exists()
    assert not list(tmp_path.glob(".tt-development-publication-*"))


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("foreign", [False, True])
def test_postpublication_failure_retracts_owned_receipt_but_preserves_foreign_replacement(synthetic, tmp_path, monkeypatch, mode, foreign):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.result)
    original = study.frozen

    def frozen(*args):
        original(*args)
        if target.exists():
            if foreign:
                payload = target.read_bytes()
                target.unlink()
                target.write_bytes(payload)
            raise ValueError("synthetic publication postcheck failure")

    monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError, match="postcheck"):
        invoke(config)
    assert (output / "failure.json").is_file()
    assert target.exists() is foreign
    assert (output / "rejected_public.json").exists() is (not foreign)
    assert not list(tmp_path.glob(".tt-development-publication-*"))
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_producer_publication_rejects_even_semantically_identical_changed_seal_bytes(synthetic, tmp_path, monkeypatch):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, "producer")
    original = study.frozen

    def frozen(*args):
        original(*args)
        if target.exists():
            path = output / "artifact_manifest.json"
            path.write_bytes(path.read_bytes() + b"\n")

    monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError, match="seal"):
        invoke(config)
    assert not target.exists() and (output / "rejected_public.json").is_file()
    assert (output / "failure.json").is_file()


def test_producer_cannot_adopt_a_changed_seal_after_ordinary_checker_returns(synthetic, tmp_path, monkeypatch):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, "producer")
    original = study._check

    def check(*args):
        result = original(*args)
        path = output / "artifact_manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(study, "_check", check)
    with pytest.raises(ValueError, match="seal"):
        invoke(config)
    assert not target.exists()
    assert (output / "failure.json").is_file()


@pytest.mark.parametrize("mode", ["producer", "audit"])
def test_failed_publication_retains_corrupted_owned_inode_only_as_rejected_evidence(synthetic, tmp_path, monkeypatch, mode):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.result)
    original = study.frozen
    corrupted = []

    def frozen(*args):
        original(*args)
        if target.exists():
            inode = target.stat().st_ino
            payload = target.read_bytes() + b"\n"
            target.write_bytes(payload)
            assert target.stat().st_ino == inode
            corrupted.append(payload)
            raise ValueError("same-owned-inode public corruption")

    monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError, match="same-owned-inode"):
        invoke(config)
    assert not target.exists()
    assert (output / "rejected_public.json").read_bytes() == corrupted[0]
    assert study.read(output / "failure.json")["complete"] is False
    assert not list(tmp_path.glob(".tt-development-publication-*"))


@pytest.mark.parametrize("case", ["source", "input", "extra_payload", "missing_payload", "public"])
def test_fresh_ordinary_check_rejects_integrity_drift_without_any_model_calls(synthetic, case):
    value, study = synthetic, synthetic.study
    path = (value.source if case == "source" else value.corpus if case == "input" else
            value.record if case == "public" else value.root / "undeclared.json" if case == "extra_payload" else
            value.root / "mixed_seed_0" / "state_000001.npz")
    original = path.read_bytes() if path.exists() else None
    calls = dict(value.calls)
    try:
        if case == "missing_payload":
            path.unlink()
        elif case == "public":
            payload = study.read(path)
            payload["recipe_promoted"] = True
            write_json(path, payload)
        else:
            path.write_bytes((original or b"{}") + b"changed")
        with pytest.raises((ValueError, FileNotFoundError)):
            study.check(value.root, value.record)
    finally:
        if original is None:
            path.unlink()
        else:
            path.write_bytes(original)
    assert value.calls == calls


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("which", ["existing_output", "existing_public", "nested_history"])
def test_exclusive_paths_never_modify_existing_history(synthetic, tmp_path, mode, which):
    value = synthetic
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if which == "existing_output":
        output.mkdir()
        protected = output / "user.bin"
        protected.write_bytes(b"unrelated user file")
    elif which == "existing_public":
        protected = target
        protected.write_bytes(b"existing public bytes")
    else:
        output = value.root / "unapproved_child"
        setattr(config, "output" if mode == "producer" else "audit_output", output)
        protected = value.root / "manifest.json"
    before = protected.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        invoke(config)
    assert protected.read_bytes() == before
    assert not (output / "failure.json").exists()
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
