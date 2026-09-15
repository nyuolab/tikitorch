"""Synthetic frozen-core gradient/state controls; no corpus or cloud runs."""
from __future__ import annotations

import copy
import importlib
import json
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


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_core_bias")


def cell(study):
    return tiny_models(study.helper)["mixed"]


def raw_gradient(study):
    return {name: (torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape) % 11 - 5) / 100
            for name, value in cell(study)["parameters"].items()}


def test_source_closure_extends_frozen_continuation_by_only_two_files(study):
    previous, sources = study.continuation.source_identity(), study.source_identity()
    assert len(previous) == 73 and len(sources) == 75
    assert {name: sources[name] for name in previous} == previous
    assert set(sources) - set(previous) == {"manuscript/tools/diagnose_fpga_lm_core_bias.py",
                                          "tests/test_fpga_lm_core_bias.py"}


def test_applied_gradient_zeroes_only_mp_mb_and_never_aliases_raw(study):
    raw = raw_gradient(study)
    before = {name: value.clone() for name, value in raw.items()}
    applied = study.applied_gradient(raw)
    assert set(applied) == set(raw)
    for name, value in applied.items():
        assert value.dtype == torch.float64 and value.device.type == "cpu" and not value.requires_grad
        assert value.shape == raw[name].shape and value.data_ptr() != raw[name].data_ptr()
        assert torch.equal(value, torch.zeros_like(value) if name in ("MP", "MB") else raw[name])
        value.add_(100)
    study.guard.byte_equal(raw, before, "mask cannot mutate raw gradients")


@pytest.mark.parametrize("tied_taps", [False, True])
def test_static_projection_bias_offsets_can_move_to_head_with_full_mean(study, tied_taps):
    """Conditional representability, not an equality of optimizer trajectories."""
    width, vocab, layers, taps = 8, 4, 4, 7
    embeddings = torch.tensor([[(3 * v + 5 * i) % 11 - 5 for i in range(width)]
                               for v in range(vocab)], dtype=torch.float64) / 64
    mh = torch.zeros((layers, width, taps), dtype=torch.float64)
    if not tied_taps:
        mh = torch.tensor([[[(l + 2 * i + k) % 7 - 3 for k in range(taps)]
                            for i in range(width)] for l in range(layers)], dtype=torch.float64) / 32
    mp = torch.full((layers, width, width), -100., dtype=torch.float64)
    head = torch.full((vocab, width), -100., dtype=torch.float64)
    head[torch.arange(vocab), torch.arange(vocab)] = torch.arange(vocab, dtype=torch.float64) / 32
    maps = [[(2 * i + l) % width for i in range(width)] for l in range(layers)]
    for l, mapping in enumerate(maps):
        mp[l, torch.arange(width), mapping] = 0
    baseline = {"EMBED": embeddings, "MH": mh, "MP": mp,
                "MB": torch.zeros((layers, width), dtype=torch.float64), "W_HEAD": head}
    changed = {name: value.clone() for name, value in baseline.items()}
    differences = []
    for l, mapping in enumerate(maps):
        row = []
        for i, j in enumerate(mapping):
            selected = ((l + i) % 7 - 3) / 64
            bias = ((2 * l + i) % 5 - 2) / 128
            changed["MP"][l, i, j] = selected
            changed["MB"][l, i] = bias
            row.append(selected + bias)
        differences.append(row)
    offset, missing_mean = [0.] * width, [0.] * width
    for mapping, bias in zip(maps, differences):
        mean = sum(offset) / width
        offset = [offset[i] + offset[mapping[i]] - mean + bias[i] for i in range(width)]
        missing_mean = [missing_mean[i] + missing_mean[mapping[i]] + bias[i] for i in range(width)]
    compensated = {name: value.clone() for name, value in baseline.items()}
    wrong = {name: value.clone() for name, value in baseline.items()}
    for v in range(vocab):
        compensated["W_HEAD"][v, v] += offset[v]
        wrong["W_HEAD"][v, v] += missing_mean[v]
    tokens = torch.tensor([0, 1, 0, 2, 3, 1, 2, 0, 3, 2, 1, 0], dtype=torch.int64)
    actual = study.helper.trace(study.helper.forward(changed, tokens, independent=True))
    matched = study.helper.trace(study.helper.forward(compensated, tokens, independent=True))
    wrong_trace = study.helper.trace(study.helper.forward(wrong, tokens, independent=True))
    torch.testing.assert_close(actual["logits"], matched["logits"], atol=1e-12, rtol=1e-12)
    for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX"):
        assert torch.equal(actual[key], matched[key])
    assert not torch.allclose(actual["logits"], wrong_trace["logits"], atol=1e-12, rtol=1e-12)
    for l, mapping in enumerate(maps):
        assert torch.equal(actual["PROJ_IDX"][l], torch.tensor(mapping).expand(len(tokens), -1))
    assert torch.equal(actual["AD_HEAD"], torch.arange(vocab).expand(len(tokens), -1))
    # Earliest positions exclude causal padding even when all valid taps tie.
    assert torch.count_nonzero(actual["TAP_IDX"][:, 0]) == 0


def test_matching_but_time_varying_projection_tapes_do_not_imply_bias_equivalence(study):
    """A constant MB change yields token-dependent class differences."""
    baseline = {"EMBED": torch.tensor([[2., -2.], [-2., 2.]], dtype=torch.float64),
                "MH": torch.zeros((2, 2, 1), dtype=torch.float64),
                "MP": torch.tensor([[[0., -100.], [-100., 0.]], [[0., 0.], [-100., 0.]]], dtype=torch.float64),
                "MB": torch.zeros((2, 2), dtype=torch.float64),
                "W_HEAD": torch.tensor([[0., -100.], [-100., 0.]], dtype=torch.float64)}
    changed = {name: value.clone() for name, value in baseline.items()}
    changed["MB"][0, 0] = 1
    tokens = torch.tensor([0, 1], dtype=torch.int64)
    a = study.helper.trace(study.helper.forward(changed, tokens, independent=True))
    b = study.helper.trace(study.helper.forward(baseline, tokens, independent=True))
    assert torch.equal(a["PROJ_IDX"], b["PROJ_IDX"])
    assert a["PROJ_IDX"][1, :, 0].tolist() == [0, 1]
    delta = a["logits"] - b["logits"]
    torch.testing.assert_close(delta, torch.tensor([[1.5, -.5], [.5, -.5]], dtype=torch.float64), atol=0, rtol=0)
    assert float(delta[0, 0] - delta[0, 1]) != float(delta[1, 0] - delta[1, 1])


@pytest.mark.parametrize("case", ["missing", "extra", "float32", "integer", "nan", "attached", "shape"])
def test_mask_rejects_incomplete_or_invalid_raw_geometry(study, case):
    raw = raw_gradient(study)
    if case == "missing":
        del raw["MB"]
    elif case == "extra":
        raw["unexpected"] = torch.zeros(1, dtype=torch.float64)
    elif case == "float32":
        raw["EMBED"] = raw["EMBED"].float()
    elif case == "integer":
        raw["MH"] = raw["MH"].long()
    elif case == "nan":
        raw["MP"].reshape(-1)[0] = float("nan")
    elif case == "attached":
        raw["MB"].requires_grad_(True)
    else:
        raw["MB"] = raw["MB"][:, :1]
    with pytest.raises((ValueError, TypeError)):
        study.applied_gradient(raw)


def test_unbounded_fractional_raw_g_is_not_mistaken_for_integer_parameter_state(study):
    raw = raw_gradient(study)
    raw["EMBED"].reshape(-1)[0] = 1e12 + .25
    raw["MP"].fill_(-0.)
    applied = study.applied_gradient(raw)
    study.guard.byte_equal({"EMBED": applied["EMBED"]}, {"EMBED": raw["EMBED"]}, "unrounded raw gradient units")
    assert not torch.signbit(applied["MP"]).any() and torch.signbit(raw["MP"]).all()


def test_all256_zero_applied_core_updates_keep_weights_moments_but_advance_counters(study):
    model = cell(study)
    initial = {name: model["parameters"][name].clone() for name in study.FROZEN_REGIONS}
    optimizer = study.parent.make_optimizer("mixed", model["parameters"].values())
    assert study.frozen_state(model, optimizer, initial, 0)["moments_empty"] is True
    raw = raw_gradient(study)
    assert bool(raw["MP"].any()) and bool(raw["MB"].any())
    for step in range(1, 257):
        applied = study.applied_gradient({name: value * (1 if step % 2 else -.5) for name, value in raw.items()})
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model["parameters"].items():
            parameter.grad = applied[name].clone()
        optimizer.step()
        receipt = study.frozen_state(model, optimizer, initial, step)
        assert receipt["moments_exact_zero"] is True and receipt["all_counters"] == step
        for moment in optimizer.state.values():
            assert float(moment["step"]) == step
    for name in study.TRAINABLE_REGIONS:
        assert bool(optimizer.state[model["parameters"][name]]["exp_avg"].any())
    study.guard.byte_equal({key: model["parameters"][key] for key in initial}, initial, "entire core frozen, not selected entries only")


@pytest.mark.parametrize("case", ["weight", "first", "second", "reset_counter", "missing", "signed_zero"])
def test_frozen_state_guard_rejects_carried_moments_parameter_drift_or_false_counters(study, case):
    model = cell(study)
    initial = {name: model["parameters"][name].clone() for name in study.FROZEN_REGIONS}
    optimizer = study.parent.make_optimizer("mixed", model["parameters"].values())
    applied = study.applied_gradient(raw_gradient(study))
    for name, parameter in model["parameters"].items():
        parameter.grad = applied[name]
    optimizer.step()
    value = model["parameters"]["MP"]
    state = optimizer.state[value]
    if case == "weight":
        value.reshape(-1)[0] += .1
    elif case in ("first", "second"):
        state["exp_avg" if case == "first" else "exp_avg_sq"].reshape(-1)[0] = .1
    elif case == "reset_counter":
        state["step"].zero_()
    elif case == "missing":
        del state["exp_avg"]
    else:
        state["exp_avg"].fill_(-0.)
    with pytest.raises(ValueError):
        study.frozen_state(model, optimizer, initial, 1)


@pytest.mark.parametrize("case", ["trainable_zeroed", "core_nonzero", "negative_zero"])
def test_applied_archive_cannot_change_raw_trainable_g_or_core_zero_bytes(study, tmp_path, case):
    raw = raw_gradient(study)
    storage = study.Storage(tmp_path, SimpleNamespace(check=lambda: None))
    actual, receipt = study.applied_record(tmp_path, 0, raw, write=True, storage=storage)
    assert receipt["raw_gradient_sha256"] == study.tensor_hash(raw)
    assert receipt["applied_gradient_sha256"] == study.tensor_hash(actual)
    path = tmp_path / receipt["file"]
    values = study.load(path)
    if case == "trainable_zeroed":
        values["EMBED"].zero_()
    elif case == "core_nonzero":
        values["MB"].reshape(-1)[0] = .01
    else:
        values["MP"].fill_(-0.)
    path.unlink()
    study.parent.parent.save(path, values)
    with pytest.raises(ValueError):
        study.applied_record(tmp_path, 0, raw)


@pytest.mark.parametrize("step", [-1, 256, True, 0.0, "0"])
def test_applied_archives_require_literal_bounded_local_steps(study, tmp_path, step):
    with pytest.raises(ValueError):
        study.applied_record(tmp_path, step, raw_gradient(study), write=True,
                             storage=study.Storage(tmp_path, SimpleNamespace(check=lambda: None)))
    assert not list(tmp_path.iterdir())


def test_fixed_original_start_single_arm_one_pass_recipe(study):
    recipe = study.recipe()
    assert recipe["arm"] == "mixed_core_bias_frozen" and recipe["order_seed"] == 0
    assert recipe["frozen_regions"] == ["MP", "MB"] and recipe["trainable_regions"] == ["EMBED", "MH", "W_HEAD"]
    assert recipe["raw_gradient_queries"] == recipe["applied_gradient_archives"] == recipe["optimizer_commits"] == 256
    assert recipe["complete_saved_states"] == 257 and recipe["score_window_readouts"] == 128
    assert recipe["learning_rate"] == .01 and recipe["batch_size"] == 4 and recipe["sequence"] == 128
    assert recipe["new_gradient_pair_exposures"] == recipe["inherited_fitting_pair_exposures"] == 131072
    assert recipe["inherited_gradient_pair_exposures"] == 0 and recipe["endpoint_history_pair_exposures"] == 262144
    assert recipe["sampled_training_windows"] == list(range(0, 1024, 16))
    assert recipe["development_windows"] == list(range(88, 152))
    assert recipe["training_order"] == study.parent.training_order(0)
    assert recipe["new_initial_score_evaluations"] == recipe["new_baseline_evaluations"] == 0
    assert recipe["state_format"] == study.parent.SCHEMA and recipe["state_format_arm"] == "mixed"
    assert "none" in recipe["quality_gate"] and recipe["new_gru_training"] is False
    for key in ("new_model_count_fit", "adaptive_selection", "recipe_promoted", "heldout_test_evaluated",
                "reserved_development_evaluated", "physical_fpga_execution", "paid_execution_authorized"):
        assert recipe[key] is False


@pytest.mark.parametrize("pin", [None, "", "a" * 63, "G" * 64, 1])
def test_missing_successful_prerequisite_audit_blocks_before_input_or_model_access(study, monkeypatch, tmp_path, pin):
    monkeypatch.setattr(study, "AUDIT_RECEIPT_SHA256", pin)
    monkeypatch.setattr(study.parent.rate.head, "_regular", lambda *a: pytest.fail("unpinned audit opened input"))
    with pytest.raises(ValueError, match="not yet pinned"):
        study._audit_capsule(tmp_path / "missing", tmp_path / "missing_root", tmp_path / "public.json")


@pytest.fixture
def initial_state(study, tmp_path, monkeypatch):
    root = tmp_path / "original_development"
    directory = root / "mixed_seed_0"
    directory.mkdir(parents=True)
    model = cell(study)
    optimizer = study.parent.make_optimizer("mixed", model["parameters"].values())
    storage = study.parent.Storage(root, SimpleNamespace(check=lambda: None))
    study.parent.state_record(directory, 0, model, optimizer, "mixed", write=True, storage=storage)
    monkeypatch.setattr(study.parent, "initial_models", lambda binding: {"mixed": cell(study)})
    return {"development_binding": {"synthetic": True}, "endpoint_binding": {"prior_run": str(root)}}


def test_initial_factory_uses_actual_original_state_zero_with_no_prior_moments(study, initial_state):
    first = study.initial_model(initial_state)
    second = study.initial_model(initial_state)
    assert first["initial_source"]["step"] == 0
    assert first["initial_source"]["file"]["path"].endswith("mixed_seed_0/state_000000.npz")
    for name in first["parameters"]:
        assert torch.equal(first["parameters"][name], second["parameters"][name])
        assert first["parameters"][name].data_ptr() != second["parameters"][name].data_ptr()
    assert set(study.load(first["initial_source"]["file"]["path"])) == {"parameter/" + name for name in first["parameters"]}


@pytest.mark.parametrize("case", ["parameter_drift", "moments_inserted"])
def test_initial_factory_rejects_trained_or_nonempty_optimizer_start(study, initial_state, case):
    path = study.Path(initial_state["endpoint_binding"]["prior_run"]) / "mixed_seed_0/state_000000.npz"
    values = study.load(path)
    if case == "parameter_drift":
        values["parameter/EMBED"].reshape(-1)[0] += .1
    else:
        values["exp_avg/EMBED"] = torch.zeros_like(values["parameter/EMBED"])
    path.unlink()
    study.parent.parent.save(path, values)
    with pytest.raises(ValueError):
        study.initial_model(initial_state)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Real tiny raw/applied-gradient run; historical inputs are synthetic."""
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_core_bias")
    patch = pytest.MonkeyPatch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    temporary = tmp_path_factory.mktemp("core-bias-control")
    source, corpus, declaration = (temporary / name for name in ("source.txt", "corpus.txt", "PLAN.md"))
    source.write_text("Synthetic invocation source proof.\n")
    corpus.write_bytes(b"Opaque synthetic input; real corpus access blocked.\n")
    declaration.write_text("Synthetic one-arm core-bias test, not scientific evidence.\n")
    for module, values in (
            (study, {"TRAIN_WINDOWS": tuple(range(8)), "SCORE_TRAIN_WINDOWS": (0, 4), "DEV_WINDOWS": (88, 89),
                     "UPDATES": 2, "SEQUENCE": 4, "PLAN": declaration}),
            (study.parent, {"TRAIN_WINDOWS": tuple(range(8)), "DEV_WINDOWS": (88, 89), "SEQUENCE": 4, "VOCAB": 4}),
            (study.endpoint, {"TRAIN_WINDOWS": (0, 4), "SEQUENCE": 4, "VOCAB": 4}),
            (study.continuation, {"SCORE_TRAIN_WINDOWS": (0, 4), "DEV_WINDOWS": (88, 89)})):
        for name, value in values.items():
            patch.setattr(module, name, value)
    patch.setattr(study.parent, "initial_models", lambda binding: tiny_models(study.helper))
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
    dev_root, endpoint_root, prior_root = (temporary / name for name in ("old_development", "old_endpoint", "old_continuation"))
    for root in (dev_root, endpoint_root, prior_root):
        root.mkdir()
    dataset = {"path": str(corpus)}
    endpoint_binding = {"prior_run": str(dev_root), "parent_binding": {"synthetic": True}, "dataset": dataset}
    continuation_binding = {"prior_run": str(endpoint_root), "endpoint_binding": endpoint_binding}
    old_cells, endpoint_cells, initials = [], [], {}
    es = study.parent.Storage(endpoint_root, SimpleNamespace(check=lambda: None))
    ds = study.parent.Storage(dev_root, SimpleNamespace(check=lambda: None))
    for arm in ("mixed", "prior_gru"):
        directory = dev_root / f"{arm}_seed_0"
        directory.mkdir()
        original = tiny_models(study.helper)[arm]
        optimizer = study.parent.make_optimizer(arm, original["parameters"].values())
        study.parent.state_record(directory, 0, original, optimizer, arm, write=True, storage=ds)
        for label, step in (("initial", 0), ("final", 256)):
            if step:
                # Synthetic historical endpoint: full typed state without any
                # model fitting or gradient operation in fixture preparation.
                with torch.no_grad():
                    next(iter(original["parameters"].values())).reshape(-1)[0] += .03125
                for parameter in original["parameters"].values():
                    optimizer.state[parameter] = {"step": torch.tensor(256., dtype=torch.float32),
                                                  "exp_avg": torch.zeros_like(parameter),
                                                  "exp_avg_sq": torch.zeros_like(parameter)}
                study.parent.state_record(directory, step, original, optimizer, arm, write=True, storage=ds)
            model = study.endpoint.model(endpoint_binding, arm, None if step == 0 else 0)
            train_dir = endpoint_root / ("initial_" + arm if step == 0 else f"{arm}_seed_0")
            train_dir.mkdir()
            train = study.endpoint.collect_scores(train_dir, model, corpus.read_bytes(), dataset, write=True, storage=es)
            dev_dir = dev_root / ("shared_" + arm) if step == 0 else directory
            dev_dir.mkdir(exist_ok=True)
            with torch.no_grad():
                dev = study.parent.collect_scores(dev_dir, label, model, corpus.read_bytes(), dataset, True,
                                                   write=True, storage=ds)
            development = {"per_window": dev, "mean": study.parent._mean(dev), "new_forward_evaluations": 0}
            row = {"train_scores": train, "train_mean": study.parent._mean(train),
                   "development": development, "source_state": model["endpoint_source"]}
            if step == 0:
                initials[arm] = row
            else:
                endpoint_cells.append({"arm": arm, "order_seed": 0, "complete": True, **row})
                old_cells.append({"arm": arm, "order_seed": 0, "final_scores": dev, "final_mean": study.parent._mean(dev)})
    baseline = {"synthetic_fixed_baseline": {"ce": 3.}}
    write_json(dev_root / "summary.json", {"cells": old_cells, "baselines": baseline})
    write_json(endpoint_root / "summary.json", {"cells": endpoint_cells, "initial": initials, "baselines": baseline})
    for root, schema in ((dev_root, study.parent.SCHEMA), (endpoint_root, study.endpoint.SCHEMA),
                         (prior_root, study.continuation.SCHEMA)):
        write_json(root / "manifest.json", {"schema": schema})
    binding = {"prior_run": str(prior_root), "prior_record": str(corpus), "prior_audit": str(corpus),
               "continuation_binding": continuation_binding, "endpoint_binding": endpoint_binding,
               "development_binding": {"synthetic": True}, "dataset": dataset, "audit_receipt_sha256": "a" * 64}
    protected = [study.old.file_identity(path) for root in (dev_root, endpoint_root, prior_root)
                 for path in root.rglob("*") if path.is_file()]
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
        study.equal(manifest["recipe"], study.recipe(), "synthetic fixed recipe")
        study.equal(manifest["binding"], token.binding, "synthetic complete binding")
        study.equal(token.binding, binding, "synthetic original binding")
        study.equal(manifest["source_hashes"], token.sources, "synthetic invocation source")
        study.equal(token.sources, study.source_identity(), "synthetic source bytes")
        for descriptor in protected:
            study.equal(study.old.file_identity(descriptor["path"]), descriptor, "synthetic immutable historical input")

    patch.setattr(study, "bind_inputs", bind)
    patch.setattr(study, "frozen", frozen)
    patch.setattr(study.continuation.shutil, "disk_usage", lambda path: SimpleNamespace(free=32 * 2**30))
    for module in (study.continuation, study.endpoint, study.parent):
        for name in ("run", "execute", "check", "audit"):
            patch.setattr(module, name, lambda *a, **k: pytest.fail("old numerical execution forbidden"))
    for module in (study.endpoint, study.parent):
        patch.setattr(module, "baselines", lambda *a, **k: pytest.fail("new count fit/baseline scoring forbidden"))
    patch.setattr(study.helper.FittedPriorGRU, "forward", lambda *a, **k: pytest.fail("new GRU model call forbidden"))
    calls = {"guarded": 0, "dense": 0, "manual_g": 0, "independent_g": 0}
    og, of, ov = study.guard.guarded.forward, study.helper.forward, study.helper.gradient

    def guarded(*args, **kwargs):
        calls["guarded"] += 1
        return og(*args, **kwargs)

    def forward(*args, **kwargs):
        if kwargs.get("independent"):
            calls["dense"] += 1
        return of(*args, **kwargs)

    def gradient(*args, **kwargs):
        calls["independent_g" if kwargs.get("independent") else "manual_g"] += 1
        return ov(*args, **kwargs)

    patch.setattr(study.guard.guarded, "forward", guarded)
    patch.setattr(study.helper, "forward", forward)
    patch.setattr(study.helper, "gradient", gradient)
    reads.clear()
    config = SimpleNamespace(prior_run=prior_root, prior_record=corpus, prior_audit=corpus,
                             corpus=corpus, output=temporary / "run", record=temporary / "public.json")
    try:
        result = study.run(config)
        yield SimpleNamespace(study=study, temporary=temporary, config=config, root=config.output,
                              record=config.record, result=result, binding=binding, source=source, corpus=corpus,
                              prior_root=prior_root, dev_root=dev_root, endpoint_root=endpoint_root,
                              protected=protected, calls=calls, reads=reads, binds=binds)
    finally:
        patch.undo()
        torch.set_num_threads(previous_threads)


def test_real_single_arm_run_keeps_raw_and_applied_gradients_and_complete_adam(synthetic):
    value, study = synthetic, synthetic.study
    result = value.result
    assert result["complete"] is True and result["arm"] == study.ARM
    assert result["raw_gradient_queries"] == result["applied_gradient_archives"] == result["optimizer_commits"] == 2
    assert result["complete_saved_states"] == 3 and result["score_window_readouts"] == 4
    assert value.calls == {"guarded": 12, "dense": 0, "manual_g": 8, "independent_g": 0}
    assert result["new_initial_score_evaluations"] == result["new_baseline_evaluations"] == 0
    assert result["new_gru_training"] is False and result["inherited_gradient_pair_exposures"] == 0
    assert result["training_order"] == study.training_order()
    directory = value.root / study.ARM
    previous = study.load(directory / "state_000000.npz")
    initial = {key: previous["parameter/" + key].clone() for key in study.FROZEN_REGIONS}
    source = result["initial_source"]["file"]["path"]
    study.guard.byte_equal(previous, study.load(source), "actual original state0, not trained endpoint")
    nonzero_frozen_raw = False
    for step in range(2):
        query = study.load(directory / f"query_{step:06d}.npz")
        raw = {key[2:]: val for key, val in query.items() if key.startswith("g_")}
        applied = study.load(directory / f"applied_{step:06d}.npz")
        study.guard.byte_equal(applied, study.applied_gradient(raw), "actual masked raw gradient")
        nonzero_frozen_raw |= any(bool(raw[key].any()) for key in study.FROZEN_REGIONS)
        after = study.load(directory / f"state_{step + 1:06d}.npz")
        for name, gradient in applied.items():
            parameter = previous["parameter/" + name]
            first = .9 * previous.get("exp_avg/" + name, torch.zeros_like(parameter)) + .1 * gradient
            second = .999 * previous.get("exp_avg_sq/" + name, torch.zeros_like(parameter)) + .001 * gradient.square()
            expected = parameter - .01 * first / (1 - .9 ** (step + 1)) / (second.sqrt() / ((1 - .999 ** (step + 1)) ** .5) + 1e-8)
            torch.testing.assert_close(after["parameter/" + name], expected, atol=1e-12, rtol=1e-11)
            assert float(after["step/" + name]) == step + 1
            if name in study.FROZEN_REGIONS:
                study.guard.byte_equal({"weight": after["parameter/" + name]}, {"weight": initial[name]}, "whole region frozen")
                assert not after["exp_avg/" + name].any() and not after["exp_avg_sq/" + name].any()
        previous = after
    assert nonzero_frozen_raw, "fixture must distinguish retained raw G from zero applied G"
    assert set(value.reads) <= {("train", i) for i in range(8)} | {("validation", i) for i in (88, 89)}


def test_ordinary_replay_uses_saved_applied_g_without_new_model_calls(synthetic):
    value = synthetic
    before = dict(value.calls)
    assert value.study.check(value.root, value.record) == value.result
    assert value.calls == before


def test_independent_audit_rederives_all_raw_g_and_checks_mask_state_and_scores(synthetic, tmp_path):
    value, study = synthetic, synthetic.study
    before, original = dict(value.calls), tree_bytes(value.root)
    config = SimpleNamespace(**vars(value.config), audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    result = study.audit(config)
    assert result["complete"] is True and result["independent_recompute_completed"] is True
    assert {key: value.calls[key] - before[key] for key in before} == {
        "guarded": 12, "dense": 12, "manual_g": 0, "independent_g": 8}
    assert result["full_batch_gradients_verified"] == result["applied_gradient_masks_verified"] == 2
    assert result["complete_saved_states_verified"] == 3 and result["score_window_readouts"] == 4
    assert (config.audit_output / "verification.json").read_bytes() == config.audit_record.read_bytes()
    assert tree_bytes(value.root) == original


def test_production_inventory_keeps_raw_applied_states_and_final_only_scores(study):
    # These source constants are checked before the module-scoped tiny fixture
    # is entered when this test is selected on its own.
    expected = 4 + 2 * study.UPDATES + 2 * (study.UPDATES + 1) + len(study.SCORE_TRAIN_WINDOWS) + len(study.DEV_WINDOWS)
    names = study.expected_artifacts()
    assert len(names) == expected
    assert sum("/query_" in name for name in names) == study.UPDATES
    assert sum("/applied_" in name for name in names) == study.UPDATES
    assert sum("/state_" in name and name.endswith(".npz") for name in names) == study.UPDATES + 1
    assert not any("initial" in name or "baseline" in name or "gru" in name for name in names)


@pytest.fixture
def audit_capsule(study, tmp_path, monkeypatch):
    root, capsule = tmp_path / "previous", tmp_path / "audit"
    root.mkdir()
    capsule.mkdir()
    public, receipt = tmp_path / "producer.json", tmp_path / "receipt.json"
    write_json(root / "artifact_manifest.json", {"synthetic": True})
    write_json(public, {"complete": True})
    sources = {"synthetic_source": "a" * 64}
    monkeypatch.setattr(study.continuation, "source_identity", lambda: sources)
    declaration = capsule / "declaration.md"
    declaration.write_text("Synthetic complete continuation audit capsule.\n")
    value = {"schema": study.continuation.SCHEMA + ".audit", "complete": True,
             "independent_recompute_completed": True, "run_directory": str(root), "record": str(public),
             "source_hashes": sources, "producer_public_sha256": study.old.digest(public.read_bytes()),
             "producer_seal_sha256": study.old.digest((root / "artifact_manifest.json").read_bytes()),
             "deadline_seconds": 10800, "cell_deadline_seconds": 3600,
             "independent_tropical_window_vjps": 1024, "independent_tropical_window_forwards": 1152,
             "independent_tropical_batch_gradients": 256, "independent_gru_batch_gradients": 256,
             "independent_gru_single_window_scores": 128, "supplemental_guarded_window_forwards": 1152,
             "full_batch_gradients_verified": 512, "complete_saved_states_verified": 514,
             "optimizer_commits_replayed": 512, "score_window_readouts": 256,
             "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
             "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
             "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
             "reserved_development_evaluated": False, "physical_fpga_execution": False,
             "paid_execution_authorized": False, "recipe_promoted": False,
             "verification_elapsed_utc_seconds": 100., "verification_elapsed_monotonic_seconds": 100.,
             "audit_directory": str(capsule), "runtime": study.parent.runtime(),
             "started_at_utc": "2026-09-14 00:00:00 UTC"}
    manifest = {key: value[key] for key in ("run_directory", "record", "source_hashes", "producer_public_sha256",
                "producer_seal_sha256", "runtime", "started_at_utc", "deadline_seconds", "cell_deadline_seconds")}
    manifest.update(schema=study.continuation.SCHEMA, complete=False, mode="independent_audit",
                    audit_directory=str(capsule), audit_record=str(receipt),
                    declaration_sha256=study.old.digest(declaration.read_bytes()))
    write_json(capsule / "audit_manifest.json", manifest)
    value["audit_manifest_sha256"] = study.old.digest((capsule / "audit_manifest.json").read_bytes())

    def reseal():
        write_json(capsule / "verification.json", value)
        write_json(receipt, value)
        monkeypatch.setattr(study, "PRODUCER_SHA256", study.old.digest(public.read_bytes()))
        monkeypatch.setattr(study, "AUDIT_RECEIPT_SHA256", study.old.digest(receipt.read_bytes()))

    reseal()
    return SimpleNamespace(root=root, capsule=capsule, public=public, receipt=receipt, value=value, reseal=reseal)


def test_successful_audit_admission_verifies_exact_capsule_without_older_numerical_replay(study, audit_capsule, monkeypatch):
    for name in ("check", "execute", "audit", "resume_model"):
        monkeypatch.setattr(study.continuation, name, lambda *a, **k: pytest.fail("old numerical method called"))
    value = audit_capsule
    actual, identities = study._audit_capsule(value.receipt, value.root, value.public)
    assert actual == value.value
    assert set(identities["files"]) == {"audit_manifest.json", "declaration.md", "verification.json"}


@pytest.mark.parametrize("case", ["private_whitespace", "failure_file", "public_pin", "incomplete", "missing_state", "expired"])
def test_altered_or_resealed_incomplete_predecessor_cannot_admit_control(study, audit_capsule, case):
    value = audit_capsule
    if case == "private_whitespace":
        path = value.capsule / "verification.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif case == "failure_file":
        write_json(value.capsule / "failure.json", {"complete": False})
    elif case == "public_pin":
        value.public.write_bytes(value.public.read_bytes() + b"\n")
    else:
        key, bad = {"incomplete": ("complete", 1), "missing_state": ("complete_saved_states_verified", 513),
                    "expired": ("verification_elapsed_monotonic_seconds", 10800.)}[case]
        value.value[key] = bad
        value.reseal()
    with pytest.raises(ValueError):
        study._audit_capsule(value.receipt, value.root, value.public)


@pytest.mark.parametrize("target", ["raw_g", "applied_g", "moment", "route"])
def test_numerical_audit_rejects_raw_mask_state_and_route_substitution(synthetic, target):
    value, study = synthetic, synthetic.study
    directory = value.root / study.ARM
    path = directory / ({"applied_g": "applied_000000.npz", "moment": "state_000001.npz"}.get(target, "query_000000.npz"))
    original = path.read_bytes()
    values = study.load(path)
    if target == "raw_g":
        values["g_MP"].reshape(-1)[0] += .1
    elif target == "applied_g":
        values["EMBED"].zero_()
    elif target == "moment":
        values["exp_avg/MP"].reshape(-1)[0] = .1
    else:
        key = next(key for key in values if key.endswith("/AD_HEAD"))
        values[key].reshape(-1)[0] = (values[key].reshape(-1)[0] + 1) % 16
    path.unlink()
    study.parent.parent.save(path, values)
    try:
        with pytest.raises(ValueError):
            study.execute(study.read(value.root / "manifest.json"), value.root, value.corpus.read_bytes(), recompute=True)
    finally:
        path.write_bytes(original)


def mode_config(value, temporary, mode):
    if mode == "producer":
        config = SimpleNamespace(**{**vars(value.config), "output": temporary / "run", "record": temporary / "public.json"})
        return config, value.study.run, config.output, config.record
    config = SimpleNamespace(**vars(value.config), audit_output=temporary / "audit", audit_record=temporary / "audit.json")
    return config, value.study.audit, config.audit_output, config.audit_record


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt, TimeoutError])
def test_interrupted_or_failed_control_retains_private_attempt_without_success(synthetic, tmp_path, monkeypatch, mode, error):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)

    def fail(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0 and torch.get_num_threads() == 1
        raise error("synthetic control interruption")

    monkeypatch.setattr(study, "execute" if mode == "producer" else "_check", fail)
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(error):
        invoke(config)
    assert study.read(output / "failure.json")["complete"] is False and not target.exists()
    assert signal.getsignal(signal.SIGALRM) == handler and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("replacement", ["owned_corrupted", "foreign"])
def test_publication_failure_preserves_rejected_owned_bytes_and_foreign_replacement(synthetic, tmp_path, monkeypatch, mode, replacement):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.result)
    original, payloads = study.frozen, []

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


@pytest.mark.parametrize("target", ["source", "input", "initial_checkpoint", "control_summary"])
def test_immutable_source_initial_and_comparator_inputs_fail_before_model_calls(synthetic, target):
    value, study = synthetic, synthetic.study
    path = {"source": value.source, "input": value.corpus,
            "initial_checkpoint": value.dev_root / "mixed_seed_0/state_000000.npz",
            "control_summary": value.endpoint_root / "summary.json"}[target]
    original, calls = path.read_bytes(), dict(value.calls)
    path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError):
            study.check(value.root, value.record)
    finally:
        path.write_bytes(original)
    assert value.calls == calls
