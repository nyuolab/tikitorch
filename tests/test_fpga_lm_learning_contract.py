"""Synthetic protocol/provenance tests; never execute either diagnostic study."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from manuscript.tools import diagnose_fpga_lm_learning_contract as study


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tests must not execute native or diagnostic studies")

    monkeypatch.setattr(study, "toy_diagnostic", forbidden)
    monkeypatch.setattr(study, "actual_diagnostic", forbidden)
    monkeypatch.setattr(study, "algebra_diagnostic", forbidden)
    monkeypatch.setattr(study.native, "EFSGD", forbidden)


def test_declared_protocol_keeps_three_toys_and_six_independent_native_probes():
    recipe = study.recipe()
    assert recipe["teacher_geometry"] == {"D": 16, "V": 4, "N": 4, "K": 7, "T": 12}
    assert recipe["teacher_cells"] == ["fixed_near", "dynamic_near", "dynamic_cold"]
    assert recipe["teacher_updates_per_cell"] == 64
    assert recipe["teacher_train_windows"] == recipe["teacher_evaluation_windows"] == 4
    assert recipe["native_arms"] == ["historical", "diverse"]
    assert recipe["native_training_windows"] == [0, 37, 511]
    assert recipe["native_updates"] == 6
    assert recipe["native_updates_are_a_training_trajectory"] is False
    assert recipe["numerical_variants"] == ["exact_ce", "lut_normalized", "truncated_normalized", "variable_mass", "legacy"]
    assert recipe["counterfactual_scales"] == [0.0625, 0.25, 1.0]
    assert len(study.ARMS) * len(study.WINDOWS) * len(study.VARIANTS) * len(study.SCALES) == 90
    assert recipe["overall_deadline_s"] == 1800
    for field in ("heldout_corpus_evaluated", "physical_fpga_execution", "paid_execution_authorized",
                  "statistical_significance_claim", "adaptive_retry_or_extension"):
        assert recipe[field] is False


def test_toy_windows_are_deterministic_distinct_complete_and_in_vocabulary():
    train, evaluation = study.toy_windows(601), study.toy_windows(602)
    again = study.toy_windows(601)
    assert len(train) == len(evaluation) == 4
    assert all(torch.equal(a, b) for a, b in zip(train, again))
    assert all(row.dtype == torch.int64 and row.shape == (12,) for row in train + evaluation)
    assert all(bool(((row >= 0) & (row < 4)).all()) for row in train + evaluation)
    assert len({tuple(row.tolist()) for row in train + evaluation}) == 8
    train[0][0] = 99
    assert int(study.toy_windows(601)[0][0]) != 99


def test_teacher_recipe_constructs_weights_without_consuming_global_rng():
    previous = torch.random.get_rng_state().clone()
    first, second = study.teacher_weights(), study.teacher_weights()
    assert torch.equal(previous, torch.random.get_rng_state())
    shapes = {"EMBED": (4, 16), "MH": (4, 16, 7), "MP": (4, 16, 16),
              "MB": (4, 16), "W_HEAD": (4, 16)}
    for name, size in shapes.items():
        assert first[name].shape == size and first[name].dtype == torch.float64
        assert torch.equal(first[name], second[name])
        assert first[name].data_ptr() != second[name].data_ptr()
    expected = torch.full((4, 16, 16), -8192.0, dtype=torch.float64)
    expected[:, torch.arange(16), torch.arange(16)] = 0
    assert torch.equal(first["MP"], expected)
    assert torch.equal(first["MH"][0, :, 1], torch.full((16,), 32.0, dtype=torch.float64))
    assert bool((first["MH"][1, :, 1:] == 256).all())
    assert bool((first["MH"][2, :, 1:] == -256).all())
    assert bool((first["MH"][3, :, 1:] == 256).all())


@pytest.mark.parametrize("cold", [False, True])
def test_student_initialization_is_seeded_copied_and_finite(cold):
    teacher = study.teacher_weights()
    before = study.vector_hash(study.vector(teacher))
    left, right = study.student_weights(teacher, cold), study.student_weights(teacher, cold)
    assert study.vector_hash(study.vector(teacher)) == before
    assert study.vector_hash(study.vector(left)) == study.vector_hash(study.vector(right))
    assert study.vector_hash(study.vector(left)) != before
    for name in teacher:
        assert left[name].data_ptr() != teacher[name].data_ptr()
        assert bool(torch.isfinite(left[name]).all())


def test_clone_and_vector_have_explicit_order_and_no_aliases():
    weights = {name: torch.tensor([float(i)], dtype=torch.float64)
               for i, name in enumerate(study.reference.WEIGHTS)}
    assert study.vector(weights).tolist() == list(range(5))
    copied = study.clone(weights, grad=True)
    assert all(value.requires_grad for value in copied.values())
    assert all(copied[name].data_ptr() != weights[name].data_ptr() for name in weights)
    raw = study.vector(weights).numpy().astype("<f8").tobytes()
    assert study.vector_hash(study.vector(weights)) == hashlib.sha256(raw).hexdigest()


def test_soft_metrics_stationarity_and_common_shift_on_different_synthetic_window():
    logits = torch.tensor([[0.1, -0.3, 0.7], [-0.4, 0.2, 0.8]], dtype=torch.float64, requires_grad=True)
    targets = torch.softmax(logits.detach() * study.reference.BETA, dim=-1)
    ce, kl = study.soft_metrics(logits, targets)
    assert float(kl.detach()) == pytest.approx(0.0, abs=1e-14)
    gradient, = torch.autograd.grad(ce, logits)
    assert float(gradient.abs().max()) < 1e-15
    shifted = logits.detach() + torch.tensor([[5.0], [-3.0]], dtype=torch.float64)
    assert float(study.soft_metrics(shifted, targets)[0]) == pytest.approx(float(ce.detach()), abs=1e-14)


def test_gradient_statistics_preserve_zero_direction_and_negative_alignment():
    result = study.gradient_stats({"a": torch.tensor([0., 0.], dtype=torch.float64),
                                  "b": torch.tensor([-1., -2.], dtype=torch.float64)},
                                 {"a": torch.tensor([1., 2.], dtype=torch.float64),
                                  "b": torch.tensor([1., 2.], dtype=torch.float64)})
    assert result["a"]["exact_cosine"] is None
    assert result["a"]["max_difference"] == 2
    assert result["b"]["exact_cosine"] == pytest.approx(-1)
    assert result["b"]["l1"] == 3


def test_route_changes_counts_actual_arrays():
    before = {"routes": {name: torch.tensor([0, 1, 2]) for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}
    after = copy.deepcopy(before)
    after["routes"]["TAP_IDX"][1] = 0
    assert study.route_changes(before, after) == {"AD_HEAD": 0, "PROJ_IDX": 0, "TAP_IDX": 1}


def test_artifact_manifest_covers_bytes_and_excludes_only_its_own_seal(tmp_path):
    (tmp_path / "payload.bin").write_bytes(b"synthetic\x00payload")
    write_json(tmp_path / "child" / "record.json", {"value": 1})
    write_json(tmp_path / "artifact_manifest.json", [])
    rows = study.artifact_manifest(tmp_path)
    assert [row["file"] for row in rows] == ["child/record.json", "payload.bin"]
    for row in rows:
        data = (tmp_path / row["file"]).read_bytes()
        assert row["bytes"] == len(data)
        assert row["sha256"] == hashlib.sha256(data).hexdigest()


def test_artifact_manifest_rejects_even_internal_symlinks(tmp_path):
    (tmp_path / "actual").write_bytes(b"x")
    (tmp_path / "alias").symlink_to(tmp_path / "actual")
    with pytest.raises(ValueError, match="symlink"):
        study.artifact_manifest(tmp_path)


@pytest.fixture
def input_manifest(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    path.write_bytes(b"immutable input")
    sources = {"synthetic/source.py": "a" * 64}
    monkeypatch.setattr(study, "source_identity", lambda: copy.deepcopy(sources))
    return {"schema": study.SCHEMA, "recipe": study.recipe(), "source_hashes": sources,
            "prior": {"inputs": [{"path": str(path), "bytes": path.stat().st_size,
                                   "sha256": study.digest(path.read_bytes())}]}}


def test_verify_inputs_accepts_bound_synthetic_bytes(input_manifest):
    study.verify_inputs(input_manifest)


@pytest.mark.parametrize("mutation", ["source", "recipe", "payload", "size", "hash"])
def test_verify_inputs_rejects_mutated_binding(input_manifest, mutation):
    value = copy.deepcopy(input_manifest)
    if mutation == "source":
        value["source_hashes"]["synthetic/source.py"] = "b" * 64
    elif mutation == "recipe":
        value["recipe"]["native_updates"] = 7
    elif mutation == "payload":
        Path(value["prior"]["inputs"][0]["path"]).write_bytes(b"replaced bytes")
    elif mutation == "size":
        value["prior"]["inputs"][0]["bytes"] += 1
    else:
        value["prior"]["inputs"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        study.verify_inputs(value)


@pytest.fixture
def mocked_run(tmp_path, monkeypatch):
    """Lifecycle-only fixture: both diagnostic bodies are pure data mocks."""
    source = {"synthetic.py": "1" * 64}
    plan = tmp_path / "PLAN.md"
    plan.write_text("Synthetic test declaration; no experiment is executed.\n")
    config = SimpleNamespace(output=tmp_path / "new-run", record=tmp_path / "public.json",
                             prior_run=tmp_path / "old-run", prior_record=tmp_path / "old.json",
                             corpus=tmp_path / "corpus.txt")
    calls, timers, handlers = [], [], []
    prior = {"inputs": [], "prior_run": str(config.prior_run), "dataset": {"synthetic": True}}
    monkeypatch.setattr(study, "PLAN", plan)
    monkeypatch.setattr(study, "source_identity", lambda: copy.deepcopy(source))
    monkeypatch.setattr(study, "prior_binding", lambda *args: (copy.deepcopy(prior), b"", {}))
    monkeypatch.setattr(study.torch, "set_num_threads", lambda count: calls.append(("threads", count)))

    def fake_signal(which, handler):
        handlers.append((which, handler))
        return "old-handler"

    monkeypatch.setattr(study.signal, "signal", fake_signal)
    monkeypatch.setattr(study.signal, "setitimer", lambda *args: timers.append(args))
    toy = {"teacher": {"synthetic": True}, "cells": [{"cell": cell, "updates": 64} for cell in study.TOY_CELLS],
           "new_fp64_updates": 192, "soft_target_training_position_exposures": 9216,
           "training_position_indices": 48, "synthetic_evaluation_position_indices": 48,
           "native_training": False, "nearby_success_is_not_from_scratch_learning": True}
    actual = {"cells": [{"arm": arm, "window_index": window, "independent_native_updates": 1}
                        for arm in study.ARMS for window in study.WINDOWS],
              "new_native_updates": 6, "new_native_gradient_pair_exposures": 768,
              "distinct_probe_pairs": 384, "new_model_fits": 0, "inherited_fit_pairs_per_reset": 131072,
              "summed_inherited_fit_pair_histories": 786432, "summed_native_model_history_pair_exposures": 787200,
              "distinct_model_history_pairs": 131072,
              "scope": "synthetic fixture only"}

    def fake_toy(root):
        calls.append("toy-mock")
        write_json(root / "synthetic-toy.json", toy)
        return copy.deepcopy(toy)

    def fake_actual(root, binding, raw, decoded):
        calls.append("actual-mock")
        assert raw == b"" and decoded == {}
        write_json(root / "synthetic-actual.json", actual)
        return copy.deepcopy(actual)

    monkeypatch.setattr(study, "toy_diagnostic", fake_toy)
    monkeypatch.setattr(study, "actual_diagnostic", fake_actual)
    monkeypatch.setattr(study, "algebra_diagnostic", lambda root: {"synthetic": True})
    # Lifecycle assertions do not claim to exercise numerical receipt validation.
    monkeypatch.setattr(study, "validate_summary", lambda *args: None)
    monkeypatch.setattr(study, "check", lambda root, record: study.read(record))
    return SimpleNamespace(config=config, calls=calls, timers=timers, handlers=handlers,
                           source=source, toy=toy, actual=actual, plan=plan)


def test_mocked_run_publishes_exclusive_bound_records_and_restores_alarm(mocked_run):
    case = mocked_run
    result = study.run(case.config)
    assert case.calls == [("threads", 1), "toy-mock", "actual-mock"]
    assert result["complete"] is True
    assert result["actual"]["new_native_updates"] == 6
    assert result["actual"]["new_native_gradient_pair_exposures"] == 768
    assert result["actual"]["distinct_probe_pairs"] == 384
    assert result["actual"]["new_model_fits"] == 0
    assert len(result["toy"]["cells"]) == 3
    assert len(result["actual"]["cells"]) == 6
    assert result["physical_fpga_execution"] is False
    assert result["paid_execution_authorized"] is False
    assert (case.config.output / "declaration.md").read_bytes() == case.plan.read_bytes()
    assert study.read(case.config.output / "protocol.json") == study.recipe()
    assert case.timers == [(study.signal.ITIMER_REAL, study.DEADLINE_S), (study.signal.ITIMER_REAL, 0)]
    assert case.handlers[-1] == (study.signal.SIGALRM, "old-handler")
    assert not (case.config.output / "failure.json").exists()
    before = case.config.record.read_bytes()
    with pytest.raises(FileExistsError):
        study.run(case.config)
    assert case.config.record.read_bytes() == before


@pytest.mark.parametrize("error", [RuntimeError("synthetic failure"), KeyboardInterrupt(), TimeoutError("synthetic deadline")])
def test_mocked_run_retains_failure_without_public_success(mocked_run, monkeypatch, error):
    def fail(root):
        (root / "partial.bin").write_bytes(b"retain me")
        raise error

    monkeypatch.setattr(study, "toy_diagnostic", fail)
    with pytest.raises(type(error)):
        study.run(mocked_run.config)
    failure = study.read(mocked_run.config.output / "failure.json")
    assert failure["complete"] is False
    assert failure["type"] == type(error).__name__
    assert (mocked_run.config.output / "partial.bin").read_bytes() == b"retain me"
    assert not mocked_run.config.record.exists()
    assert "actual-mock" not in mocked_run.calls
    assert mocked_run.timers[-1] == (study.signal.ITIMER_REAL, 0)
    assert mocked_run.handlers[-1] == (study.signal.SIGALRM, "old-handler")


def test_mocked_alarm_handler_fails_closed_without_waiting(mocked_run, monkeypatch):
    def trigger(root):
        mocked_run.handlers[0][1](study.signal.SIGALRM, None)

    monkeypatch.setattr(study, "toy_diagnostic", trigger)
    with pytest.raises(TimeoutError, match="deadline"):
        study.run(mocked_run.config)
    assert study.read(mocked_run.config.output / "failure.json")["complete"] is False
    assert not mocked_run.config.record.exists()


def test_source_mutation_after_mocked_work_prevents_publication(mocked_run, monkeypatch):
    def changed(root, binding, raw, decoded):
        mocked_run.source["synthetic.py"] = "2" * 64
        return mocked_run.actual

    monkeypatch.setattr(study, "actual_diagnostic", changed)
    with pytest.raises(ValueError, match="source|protocol"):
        study.run(mocked_run.config)
    assert (mocked_run.config.output / "failure.json").exists()
    assert not mocked_run.config.record.exists()


def test_invalid_prior_does_not_create_run_or_publish(mocked_run, monkeypatch):
    def reject(*args):
        raise ValueError("invalid prior fixture")

    monkeypatch.setattr(study, "prior_binding", reject)
    with pytest.raises(ValueError, match="invalid prior"):
        study.run(mocked_run.config)
    assert not mocked_run.config.output.exists()
    assert not mocked_run.config.record.exists()
    assert mocked_run.timers == []


def test_run_refuses_nested_publication_before_any_work(mocked_run):
    mocked_run.config.record = mocked_run.config.output / "public.json"
    with pytest.raises(ValueError, match="outside"):
        study.run(mocked_run.config)
    assert not mocked_run.config.output.exists()
    assert not mocked_run.calls


def test_run_refuses_output_inside_historical_source_bound_tree(mocked_run):
    old = mocked_run.config.prior_run
    write_json(old / "manifest.json", {"schema": "tikitorch.fpga_lm_fake.v1"})
    mocked_run.config.output = old / "new-diagnostic"
    with pytest.raises(ValueError, match="nested"):
        study.run(mocked_run.config)
    assert not mocked_run.config.output.exists()
    assert not mocked_run.calls


@pytest.fixture
def prior_case(tmp_path, monkeypatch):
    """Real files/hashes, synthetic state decoder and synthetic prior gate only."""
    root, repo = tmp_path / "prior", tmp_path / "repo"
    root.mkdir()
    repo.mkdir()
    (repo / "source.py").write_text("# synthetic source identity\n")
    sources = {"source.py": study.digest((repo / "source.py").read_bytes())}
    corpus, record = tmp_path / "corpus.bin", tmp_path / "prior-record.json"
    corpus.write_bytes(b"not a scientific corpus; corpus_plan is mocked")
    executable = root / "build" / "native"
    executable.parent.mkdir()
    executable.write_bytes(b"not an executable")
    write_json(executable.parent / "build.json", {"synthetic": True})
    dataset = {"path": str(corpus), "synthetic": True}
    monkeypatch.setattr(study, "REPO", repo)
    monkeypatch.setattr(study.historical, "source_identity", lambda: copy.deepcopy(sources))
    monkeypatch.setattr(study.historical, "recipe", lambda: {"synthetic_protocol": True})
    monkeypatch.setattr(study.pilot, "corpus_plan", lambda *args: (b"fixture", copy.deepcopy(dataset)))
    monkeypatch.setattr(study.native, "decode_state", lambda data, *args: json.loads(data))
    monkeypatch.setattr(study.initializer, "transform", lambda weights, shape: ({**copy.deepcopy(weights), "MP": [1]}, {}))

    def gates(cells):
        if [[row["arm"], row["seed"]] for row in cells] != [[arm, seed] for arm in ("diverse", "historical") for seed in range(3)]:
            raise ValueError("all six synthetic prior cells required")
        return {"diverse": {"gate": True}, "historical": {"gate": False}}

    monkeypatch.setattr(study.historical, "gate_results", gates)
    states = {}
    for arm in study.ARMS:
        states[arm] = {"step": 0, "k": 18, "config": {"loss_kind": 1, "zshift": 2},
                       "weights": {name: [1 if name == "MP" and arm == "diverse" else 0]
                                   for name in study.reference.WEIGHTS},
                       "remainders": {name: [0] for name in study.reference.WEIGHTS}}
        write_json(root / (arm + "_seed_0/state_000000.i32"), states[arm])
    manifest = {"schema": study.historical.SCHEMA, "source_hashes": copy.deepcopy(sources),
                "dataset": dataset, "executable": {"path": str(executable), "sha256": study.digest(executable.read_bytes())},
                "build_record_sha256": study.digest((executable.parent / "build.json").read_bytes())}
    summary = {"schema": study.historical.SCHEMA, "complete": True, "declared_cells": 6, "completed_cells": 6,
               "source_hashes": copy.deepcopy(sources), "recipe": study.historical.recipe(), "failures": [],
               "cells": [{"arm": arm, "seed": seed, "complete": True, "evaluations": []}
                         for arm in ("diverse", "historical") for seed in range(3)]}
    summary["gates"] = gates(summary["cells"])

    def reseal():
        write_json(root / "manifest.json", manifest)
        write_json(root / "summary.json", summary)
        files = study.artifact_manifest(root)
        write_json(root / "artifact_manifest.json", {"schema": study.historical.SCHEMA, "files": files})
        public = {**summary, "cells": [study.historical.confirmation._compact_cell(cell) for cell in summary["cells"]],
                  "run_directory": str(root), "artifact_manifest_sha256": study.digest((root / "artifact_manifest.json").read_bytes()),
                  "artifact_files_verified": len(files), "artifact_payload_bytes": sum(row["bytes"] for row in files),
                  "full_optimizer_checkpoints_verified": 18, "checkpoint_validation_windows": 288,
                  "distinct_validation_positions": 2048, "new_native_updates": 3072,
                  "new_gradient_pair_exposures": 393216, "inherited_gradient_pair_exposures": 393216,
                  "inherited_host_fitting_pair_exposures": 786432, "new_host_fitting_pair_exposures": 0,
                  "initialization_refitted": False, "model_history_pair_exposures": 1572864,
                  "distinct_training_pairs": 131072,
                  "inherited_exposure_accounting": "sum of per-cell histories; shared old fitting/control updates were not reexecuted",
                  "native_training_trajectory_independently_replayed": False}
        write_json(record, public)
        return public

    reseal()
    study.prior_binding(root, record, corpus)
    return SimpleNamespace(root=root, repo=repo, corpus=corpus, record=record, sources=sources,
                           manifest=manifest, summary=summary, states=states, reseal=reseal)


def test_prior_binding_accepts_complete_failed_quality_gate_history(prior_case):
    case = prior_case
    binding, raw, decoded = study.prior_binding(case.root, case.record, case.corpus)
    assert raw == b"fixture"
    assert decoded == case.states
    assert set(binding["states"]) == {"historical", "diverse"}
    assert len(binding["inputs"]) == 9
    for row in binding["inputs"]:
        payload = Path(row["path"]).read_bytes()
        assert row["bytes"] == len(payload) and row["sha256"] == study.digest(payload)
    assert case.summary["gates"]["historical"]["gate"] is False


@pytest.mark.parametrize("field,value", [("complete", False), ("declared_cells", 5), ("completed_cells", 5),
                                         ("new_native_updates", 3073), ("artifact_files_verified", 1)])
def test_prior_binding_rejects_public_receipt_mutations(prior_case, field, value):
    public = study.read(prior_case.record)
    public[field] = value
    write_json(prior_case.record, public)
    with pytest.raises(ValueError):
        study.prior_binding(prior_case.root, prior_case.record, prior_case.corpus)


@pytest.mark.parametrize("mutation", ["unlisted", "missing", "modified", "failure", "symlink", "partial_inventory"])
def test_prior_binding_rejects_incomplete_or_unsafe_actual_tree(prior_case, mutation):
    case = prior_case
    state = case.root / "historical_seed_0/state_000000.i32"
    if mutation == "unlisted":
        (case.root / "unlisted.bin").write_bytes(b"unlisted")
    elif mutation == "missing":
        state.unlink()
    elif mutation == "modified":
        state.write_bytes(b"mutated")
    elif mutation == "failure":
        write_json(case.root / "failure.json", {"complete": False})
        case.reseal()
    elif mutation == "symlink":
        (case.root / "alias.i32").symlink_to(state)
    else:
        artifacts = study.read(case.root / "artifact_manifest.json")
        artifacts["files"] = artifacts["files"][:-1]
        write_json(case.root / "artifact_manifest.json", artifacts)
        public = study.read(case.record)
        public["artifact_manifest_sha256"] = study.digest((case.root / "artifact_manifest.json").read_bytes())
        write_json(case.record, public)
    with pytest.raises(ValueError):
        study.prior_binding(case.root, case.record, case.corpus)


@pytest.mark.parametrize("mutation", ["source", "missing_source_binding", "wrong_pair", "nonzero_remainder", "step", "k", "loss", "scale"])
def test_prior_binding_rejects_resealed_wrong_sources_or_initial_state(prior_case, mutation):
    case = prior_case
    if mutation == "source":
        (case.repo / "source.py").write_text("# changed\n")
    elif mutation == "missing_source_binding":
        case.manifest["source_hashes"] = {}
        case.summary["source_hashes"] = {}
    else:
        state = case.states["diverse"]
        if mutation == "wrong_pair":
            state["weights"]["EMBED"][0] = 7
        elif mutation == "nonzero_remainder":
            state["remainders"]["MP"][0] = 1
        elif mutation in ("step", "k"):
            state[mutation] += 1
        elif mutation == "loss":
            state["config"]["loss_kind"] = 0
        else:
            state["config"]["zshift"] = 4
        write_json(case.root / "diverse_seed_0/state_000000.i32", state)
    case.reseal()
    with pytest.raises(ValueError):
        study.prior_binding(case.root, case.record, case.corpus)


@pytest.fixture
def summary_case(tmp_path, monkeypatch, input_manifest):
    """Synthetic receipts only; native byte decoding is tested elsewhere."""
    root = tmp_path / "summary-run"
    root.mkdir()
    record = tmp_path / "summary-public.json"
    manifest = copy.deepcopy(input_manifest)
    manifest["run_directory"] = str(root)
    declaration = b"Synthetic receipt fixture, not a scientific execution.\n"
    manifest["plan_sha256"] = study.digest(declaration)
    executable = tmp_path / "fake-build" / "native"
    executable.parent.mkdir()
    executable.write_bytes(b"not executable")
    write_json(executable.parent / "build.json", {"synthetic": True})
    manifest["prior"]["executable"] = {"path": str(executable), "sha256": study.digest(executable.read_bytes())}
    shape = study.pilot.layout(512)
    state = {"step": 1, "k": 18, "config": {"loss_kind": 1, "zshift": 2, "lut": list(study.numeric.ROM)}}
    monkeypatch.setattr(study.native, "decode_state", lambda *args: copy.deepcopy(state))
    identity = {"schema": study.native.SCHEMA, "executable_sha256": manifest["prior"]["executable"]["sha256"],
                "build_sha256": study.digest((executable.parent / "build.json").read_bytes()),
                "source_hashes": study.native.source_identity(), "layout": study.pilot.layout_values(shape),
                "cpu_only": True, "existing_afi_compatible": False}
    teacher = {"weight_sha256": "a" * 64, "train_windows": [row.tolist() for row in study.toy_windows(601)],
               "evaluation_windows": [row.tolist() for row in study.toy_windows(602)],
               "context_disabled_kl": 0.1, "same_current_token_probability_difference": 0.2,
               "minimum_nontrivial_route_margin": 1.0,
               "train_target_sha256": "b" * 64, "evaluation_target_sha256": "c" * 64}
    toy_cells = []
    for name in study.TOY_CELLS:
        weights = {key: [0.0] for key in study.reference.WEIGHTS}
        weight_file = name + "_final_weights.json"
        write_json(root / weight_file, weights)
        trace = [{"step": step, "train_ce": 1 + 0.5 / (step + 1), "train_kl": 0.5 / (step + 1),
                  "evaluation_kl": 0.7 / (step + 1),
                  "route_changes_from_initial": {key: 0 for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}
                 for step in range(65)]
        final_hash = study.vector_hash(study.vector({key: torch.tensor(value, dtype=torch.float64) for key, value in weights.items()}))
        toy_cells.append({"cell": name, "updates": 64, "trajectory": trace,
            "manual_autodiff_max_error": 0.0, "directional_error": 0.0, "affine_error": 0.0,
            "common_shift_probability_error": 0.0, "train_ce_monotone": True,
            "relative_train_kl": trace[-1]["train_kl"] / trace[0]["train_kl"],
            "relative_evaluation_kl": trace[-1]["evaluation_kl"] / trace[0]["evaluation_kl"],
            "local_kl_goal_met": True, "final_dynamic_routes_equal_initial": True,
            "final_dynamic_route_changes_from_initial": {key: 0 for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")},
            "final_weights_file": weight_file, "final_weights_file_sha256": study.digest((root / weight_file).read_bytes()),
            "final_weight_sha256": final_hash})
    cells = []
    for arm in study.ARMS:
        for index in study.WINDOWS:
            filename = "%s_%03d_after.i32" % (arm, index)
            payload = ("synthetic checkpoint " + filename).encode()
            (root / filename).write_bytes(payload)
            checkpoint = {"schema": study.native.SCHEMA, "file": filename, "bytes": len(payload),
                          "sha256": study.digest(payload), "step": 1, "k": 18, "config": copy.deepcopy(state["config"]),
                          "identity": copy.deepcopy(identity),
                          "regions": [{"name": name, "weight_words": getattr(shape, "WORDS_" + name),
                                       "remainder_words": getattr(shape, "WORDS_" + name)} for name in study.native.WEIGHTS],
                          "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}
            cells.append({"arm": arm, "window_index": index, "initial_step": 0, "final_step": 1,
                "independent_native_updates": 1, "independent_raw_logits_equal": True,
                "frozen_full_parameter_gradient_equal": True, "native_prediction_gradient_route_equal": True,
                "native_full_weight_remainder_equal": True, "native_checkpoint": checkpoint,
                "numerical_ladder": [{"variant": variant, "candidates": [
                    {"scale": scale, "legacy_scale1_matches_actual_native_weights":
                     True if variant == "legacy" and scale == 1.0 else None}
                    for scale in study.SCALES]} for variant in study.VARIANTS]})
    summary = {"schema": study.SCHEMA, "complete": True, "recipe": study.recipe(),
               "source_hashes": manifest["source_hashes"], "algebra": {"synthetic": True},
               "toy": {"teacher": teacher, "cells": toy_cells, "native_training": False,
                       "new_fp64_updates": 192, "soft_target_training_position_exposures": 9216,
                       "training_position_indices": 48, "synthetic_evaluation_position_indices": 48,
                       "nearby_success_is_not_from_scratch_learning": True},
               "actual": {"cells": cells, "new_native_updates": 6, "new_native_gradient_pair_exposures": 768,
                          "distinct_probe_pairs": 384, "new_model_fits": 0, "inherited_fit_pairs_per_reset": 131072,
                          "summed_inherited_fit_pair_histories": 786432, "summed_native_model_history_pair_exposures": 787200,
                          "distinct_model_history_pairs": 131072},
               "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
               "paid_execution_authorized": False}

    def reseal():
        write_json(root / "manifest.json", manifest)
        write_json(root / "protocol.json", study.recipe())
        (root / "declaration.md").write_bytes(declaration)
        write_json(root / "teacher.json", summary["toy"]["teacher"])
        write_json(root / "algebra.json", summary["algebra"])
        for cell in summary["toy"]["cells"]:
            write_json(root / (cell["cell"] + ".json"), cell)
        for cell in summary["actual"]["cells"]:
            write_json(root / ("%s_%03d.json" % (cell["arm"], cell["window_index"])), cell)
            checkpoint = cell["native_checkpoint"]
            # Tests alter paths after sealing; never write attacker-chosen paths.
            if Path(checkpoint["file"]).name == checkpoint["file"]:
                write_json(root / Path(checkpoint["file"]).with_suffix(".json"), checkpoint)
        write_json(root / "summary.json", summary)
        artifacts = study.artifact_manifest(root)
        write_json(root / "artifact_manifest.json", artifacts)
        public = {**summary, "run_directory": str(root),
                  "artifact_manifest_sha256": study.digest((root / "artifact_manifest.json").read_bytes()),
                  "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts)}
        write_json(record, public)
        return public

    reseal()
    # Every negative case starts from a genuinely accepted fixture, so missing
    # new required fields cannot make a tampering test pass for the wrong cause.
    study.check(root, record)
    return SimpleNamespace(root=root, record=record, manifest=manifest, summary=summary, state=state, reseal=reseal)


def test_integrity_check_accepts_complete_synthetic_receipts(summary_case):
    case = summary_case
    result = study.check(case.root, case.record)
    assert result["complete"] is True
    assert len(result["actual"]["cells"]) == 6
    assert len(result["toy"]["cells"]) == 3


@pytest.mark.parametrize("field", ["independent_raw_logits_equal", "frozen_full_parameter_gradient_equal",
                                    "native_prediction_gradient_route_equal", "native_full_weight_remainder_equal"])
@pytest.mark.parametrize("value", [False, 1])
def test_resealed_parity_flag_cannot_bypass_literal_true(summary_case, field, value):
    case = summary_case
    case.summary["actual"]["cells"][0][field] = value
    case.reseal()
    with pytest.raises(ValueError):
        study.check(case.root, case.record)


@pytest.mark.parametrize("mutation", ["missing_toy", "missing_native", "variant", "scale", "legacy_parity",
                                       "new_updates", "exposures", "distinct", "new_fit", "float_count",
                                       "bool_step", "float_step", "bool_trace_step", "float_trace_step",
                                       "ratio", "goal", "monotonicity", "negative_error", "bool_error",
                                       "affine_error", "weight_hash", "route_claim", "hardware", "schema"])
def test_resealed_summary_guard_rejects_false_coverage_or_claims(summary_case, mutation):
    case, summary = summary_case, summary_case.summary
    native_cell, toy_cell = summary["actual"]["cells"][0], summary["toy"]["cells"][0]
    if mutation == "missing_toy":
        summary["toy"]["cells"].pop()
    elif mutation == "missing_native":
        summary["actual"]["cells"].pop()
    elif mutation == "variant":
        native_cell["numerical_ladder"].pop()
    elif mutation == "scale":
        native_cell["numerical_ladder"][0]["candidates"].pop()
    elif mutation == "legacy_parity":
        native_cell["numerical_ladder"][-1]["candidates"][-1]["legacy_scale1_matches_actual_native_weights"] = False
    elif mutation in ("new_updates", "exposures", "distinct", "new_fit", "float_count"):
        field = {"new_updates": "new_native_updates", "exposures": "new_native_gradient_pair_exposures",
                 "distinct": "distinct_probe_pairs", "new_fit": "new_model_fits", "float_count": "new_native_updates"}[mutation]
        summary["actual"][field] = 6.0 if mutation == "float_count" else summary["actual"][field] + 1
    elif mutation in ("bool_step", "float_step"):
        native_cell["final_step"] = True if mutation == "bool_step" else 1.0
    elif mutation in ("bool_trace_step", "float_trace_step"):
        toy_cell["trajectory"][1]["step"] = True if mutation == "bool_trace_step" else 1.0
    elif mutation == "ratio":
        toy_cell["relative_train_kl"] = 0.01
    elif mutation == "goal":
        toy_cell["local_kl_goal_met"] = False
    elif mutation == "monotonicity":
        toy_cell["trajectory"][1]["train_ce"] = 100.0
    elif mutation == "negative_error":
        toy_cell["manual_autodiff_max_error"] = -1.0
    elif mutation == "bool_error":
        toy_cell["manual_autodiff_max_error"] = False
    elif mutation == "affine_error":
        toy_cell["affine_error"] = 1e-5
    elif mutation == "weight_hash":
        toy_cell["final_weight_sha256"] = "f" * 64
    elif mutation == "route_claim":
        toy_cell["final_dynamic_routes_equal_initial"] = False
    elif mutation == "hardware":
        summary["physical_fpga_execution"] = True
    else:
        summary["schema"] = "wrong"
    case.reseal()
    with pytest.raises(ValueError):
        study.check(case.root, case.record)


@pytest.mark.parametrize("field", ["step", "k", "config", "identity", "regions", "format"])
def test_resealed_checkpoint_metadata_must_match_native_contract(summary_case, field):
    case = summary_case
    checkpoint = case.summary["actual"]["cells"][0]["native_checkpoint"]
    if field in ("step", "k"):
        checkpoint[field] += 1
    elif field == "config":
        checkpoint[field]["zshift"] = 4
    elif field == "identity":
        checkpoint[field]["cpu_only"] = False
    elif field == "regions":
        checkpoint[field].pop()
    else:
        checkpoint[field] = "weights only"
    case.reseal()
    with pytest.raises(ValueError):
        study.check(case.root, case.record)


@pytest.mark.parametrize("path", ["../escape.i32", "/tmp/escape.i32", "child/historical_000_after.i32"])
def test_checkpoint_paths_cannot_escape_owned_flat_directory(summary_case, path):
    case = summary_case
    case.summary["actual"]["cells"][0]["native_checkpoint"]["file"] = path
    case.reseal()
    with pytest.raises(ValueError):
        study.check(case.root, case.record)


@pytest.mark.parametrize("mutation", ["declaration", "public", "payload", "extra", "failure"])
def test_integrity_check_rejects_corruption_and_resealed_failure(summary_case, mutation):
    case = summary_case
    if mutation == "declaration":
        (case.root / "declaration.md").write_bytes(b"different declaration")
    elif mutation == "public":
        public = study.read(case.record)
        public["artifact_payload_bytes"] += 1
        write_json(case.record, public)
    elif mutation == "payload":
        (case.root / "historical_000_after.i32").write_bytes(b"changed checkpoint")
    elif mutation == "extra":
        (case.root / "unlisted.bin").write_bytes(b"unexpected")
    else:
        write_json(case.root / "failure.json", {"complete": False})
        case.reseal()
    with pytest.raises(ValueError):
        study.check(case.root, case.record)
