"""Synthetic protocol, arithmetic and lifecycle tests; no scientific execution."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_gradient_accumulation")


@pytest.fixture(autouse=True)
def no_science(study, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unmocked model, native executable, or corpus use")

    for owner, name in ((study.extension, "forward"), (study.extension, "gradient"),
                        (study.pilot, "window"), (study.pilot, "corpus_plan"),
                        (study.old.native, "EFSGD"), (study.prior.observer, "observe"),
                        (study.prior.reference, "forward")):
        monkeypatch.setattr(owner, name, forbidden)


def mapping(value=0):
    shapes = {"EMBED": (3, 2), "MH": (4, 2, 7), "MP": (4, 2, 2),
              "MB": (4, 2), "W_HEAD": (3, 2)}
    return {name: torch.full(shape, float(value), dtype=torch.float64) for name, shape in shapes.items()}


def metrics(ce=2, lut=2, accuracy=.25):
    return {"true_ce": ce, "lut_nll": lut, "accuracy": accuracy}


def endpoints(study, ce=1.9, lut=1.9):
    return {"sequential": [metrics() for _ in study.WINDOWS],
            "batch4": [metrics(ce, lut) for _ in study.WINDOWS]}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def test_protocol_fixed_sum_scale_windows_and_scope(study):
    assert study.ARMS == ("sequential", "batch4")
    assert study.WINDOWS == tuple(range(32))
    assert study.PROBES == (0, 37, 511)
    assert study.PROBE_EXPOSURES == (0, 8, 32)
    assert (study.K, study.BLOCK, study.DEADLINE_S) == (18, 4, 1800)
    recipe = study.recipe()
    assert recipe["initialization"] == "diverse seed0 fitted step0"
    assert recipe["partition"] == "train"
    assert recipe["normalization"] == "sum four gradients at unchanged block weights; divide by 2**18, not 4*2**18"
    assert recipe["optimizer_commits"] == {"sequential": 32, "batch4": 8}
    assert recipe["gradient_queries"] == {"sequential": 32, "batch4": 32}
    assert recipe["boundary_exposures"] == list(range(0, 33, 4))
    assert "signed-quantization release" in recipe["intervention"]
    assert recipe["new_native_updates"] == recipe["new_fit_pair_exposures"] == 0
    assert recipe["new_gradient_pair_exposures"] == 8192
    assert recipe["distinct_new_gradient_pairs"] == 4096
    assert recipe["summed_inherited_fit_pair_exposures"] == 262144
    assert recipe["summed_endpoint_history_pair_exposures"] == 270336
    for field in ("replay_is_training", "heldout_corpus_evaluated", "physical_fpga_execution",
                  "paid_execution_authorized", "recipe_promoted", "statistical_significance_claim", "adaptive_extension"):
        assert recipe[field] is False


def test_sources_bind_runner_tests_and_public_function_export(study):
    sources = study.source_identity()
    required = {"manuscript/tools/diagnose_fpga_lm_gradient_accumulation.py",
                "tests/test_fpga_lm_gradient_accumulation.py",
                "manuscript/tools/fpga_lm_accumulation_arithmetic.py",
                "tests/test_fpga_lm_accumulation_arithmetic.py", "tikitorch/optim/__init__.py",
                "tikitorch/optim/fixed_point.py", "tests/test_fixed_point_ef.py"}
    assert required <= set(sources)
    for name in required:
        assert sources[name] == study.old.digest((study.REPO / name).read_bytes())


@pytest.mark.parametrize("ce,lut,expected", [(1.9, 1.9, "both_training_losses_improve"),
    (2, 1.9, "mixed_or_nonpositive_training_result"), (1.9, 2, "mixed_or_nonpositive_training_result"),
    (2, 2, "mixed_or_nonpositive_training_result"), (2.1, 1.9, "mixed_or_nonpositive_training_result"),
    (1.9, 2.1, "mixed_or_nonpositive_training_result"), (2.1, 2.1, "mixed_or_nonpositive_training_result")])
def test_decision_is_paired_direction_only_never_promotion(study, ce, lut, expected):
    result = study.decision(endpoints(study, ce, lut))
    assert result["direction"] == expected
    assert result["recipe_promoted"] is False
    assert result["scope"] == "descriptive fitted-training mechanism diagnostic only"
    assert result["sequential_minus_batch4"] == pytest.approx({"true_ce": 2-ce, "lut_nll": 2-lut})


@pytest.mark.parametrize("arm", ["sequential", "batch4"])
@pytest.mark.parametrize("count", [0, 1, 31, 33])
def test_decision_requires_all_exact32_endpoint_windows(study, arm, count):
    rows = endpoints(study)
    rows[arm] = [metrics() for _ in range(count)]
    with pytest.raises(ValueError):
        study.decision(rows)


@pytest.mark.parametrize("field,value", [("true_ce", True), ("lut_nll", False), ("accuracy", True),
    ("true_ce", -1), ("lut_nll", -.1), ("accuracy", -.1), ("accuracy", 1.1),
    ("true_ce", float("nan")), ("lut_nll", float("inf")), ("accuracy", float("-inf"))])
def test_metrics_reject_nonphysical_nonfinite_or_boolean_values(study, field, value):
    row = metrics()
    row[field] = value
    with pytest.raises(ValueError):
        study.mean_metrics([row])


@pytest.mark.parametrize("mutation", ["missing_arm", "extra_arm", "missing_metric", "extra_metric"])
def test_decision_rejects_incomplete_or_ambiguous_schema(study, mutation):
    rows = endpoints(study)
    if mutation == "missing_arm":
        rows.pop("batch4")
    elif mutation == "extra_arm":
        rows["other"] = rows["batch4"]
    elif mutation == "missing_metric":
        rows["batch4"][0].pop("accuracy")
    else:
        rows["batch4"][0]["selection"] = 1
    with pytest.raises(ValueError):
        study.decision(rows)


@pytest.mark.parametrize("index", [True, False, 0.0, "0", -1, 32, 38, 88, 152, 864])
def test_window_guard_never_looks_up_undeclared_or_heldout_data(study, index):
    with pytest.raises(ValueError):
        study.window(b"unused", {}, index)


@pytest.mark.parametrize("index", [0, 31, 37, 511])
def test_window_guard_requests_train_partition_only(study, monkeypatch, index):
    calls = []
    monkeypatch.setattr(study.pilot, "window", lambda *args: (calls.append(args) or ([2, 1], [1, 0])))
    tokens, targets = study.window(b"synthetic", {"synthetic": True}, index)
    assert calls == [(b"synthetic", {"synthetic": True}, "train", index)]
    assert tokens.dtype == torch.int64 and tokens.tolist() == [2, 1]
    assert targets == [1, 0]


def test_signed_partition_has_same_shadow_not_same_visible_state(study):
    w, r = mapping(), mapping()
    g1, g2 = mapping(), mapping()
    g1["EMBED"][0, 0] = (1 << study.K) + 1
    g2["EMBED"][0, 0] = -2
    batch = study.accumulation.apply_block(w, r, [g1, g2], k=study.K)
    result = study.fixed_partition(w, r, [g1, g2], batch)
    assert result["same_shadow"] is True
    assert result["counterfactual_committed"] is False
    assert result["visible_difference"]["max_abs"] == 1
    assert result["remainder_difference"]["max_abs"] == 1 << study.K
    assert torch.count_nonzero(w["EMBED"]) == torch.count_nonzero(r["EMBED"]) == 0


@pytest.mark.parametrize("tamper", ["missing", "extra", "dtype", "shape", "nan", "weight_bound", "fractional", "remainder_bound"])
def test_complete_state_arrays_reject_mutation(study, tmp_path, tamper):
    arrays = study.state_arrays(mapping(), mapping())
    if tamper == "missing":
        arrays.pop("r_MH")
    elif tamper == "extra":
        arrays["extra"] = torch.zeros(1, dtype=torch.float64)
    elif tamper == "dtype":
        arrays["w_EMBED"] = arrays["w_EMBED"].to(torch.int64)
    elif tamper == "shape":
        arrays["w_EMBED"] = arrays["w_EMBED"].flatten()
    else:
        key = "r_EMBED" if tamper == "remainder_bound" else "w_EMBED"
        value = {"nan": float("nan"), "weight_bound": 32768, "fractional": .5,
                 "remainder_bound": 1 << study.K}[tamper]
        arrays[key].reshape(-1)[0] = value
    path = tmp_path / "state.npz"
    study.save(path, arrays)
    with pytest.raises(ValueError):
        study.state(path)


def test_save_is_exclusive_and_same_checks_dtype_shape_and_values(study, tmp_path):
    path = tmp_path / "state.npz"
    value = {"a": torch.zeros((2, 2), dtype=torch.float64)}
    study.save(path, value)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        study.save(path, value)
    assert path.read_bytes() == original
    for changed in ({"b": value["a"]}, {"a": value["a"].to(torch.int64)},
                    {"a": value["a"].flatten()}, {"a": value["a"] + 1}):
        with pytest.raises(ValueError):
            study.same(value, changed, "synthetic")


@pytest.fixture
def fake_run(study, tmp_path, monkeypatch):
    """Real loop/state/publication, tiny fake queries and readouts, no model."""
    initial_path = tmp_path / "initial.i32"
    initial_path.write_bytes(b"synthetic fixed start, never passed to a native process")
    initial = {"step": 0, "k": 18, "config": {"loss_kind": 1, "zshift": 2}}
    monkeypatch.setattr(study.old.native, "decode_state", lambda *args: copy.deepcopy(initial))
    monkeypatch.setattr(study.old, "state_tensors", lambda *_: (mapping(), mapping()))
    controls = SimpleNamespace(queries=[], scores=[], control_indices=[], fail_query=False)

    def fake_query(directory, index, weights, raw, dataset, *, write=False, recompute=False):
        if controls.fail_query:
            raise RuntimeError("synthetic query failure")
        gradient = mapping()
        gradient["EMBED"][0, 0] = (1 << 17) + 64 * weights["EMBED"][0, 0]
        path = directory / f"query_{index:03d}.npz"
        if write:
            study.save(path, gradient)
            controls.queries.append((directory.name, index, float(weights["EMBED"][0, 0])))
        study.same(study.load(path), gradient, "synthetic stored query")
        return gradient

    def fake_scores(directory, label, weights, indices, raw, dataset, context=True, *, write=False, recompute=False):
        rows = []
        for index in indices:
            value = float(weights["EMBED"][0, 0])
            row = metrics(2 + value / 100 + index / 10000 + (0 if context else .1),
                          2.1 + value / 100 + index / 10000 + (0 if context else .1))
            path = directory / f"score_{label}_{index:03d}_{int(context)}.npz"
            arrays = {name: torch.tensor([number], dtype=torch.float64) for name, number in row.items()}
            if write:
                study.save(path, arrays)
                controls.scores.append((directory.name, label, index, context))
            study.same(study.load(path), arrays, "synthetic saved score")
            rows.append(row)
        return rows

    def fake_control(binding, index, before_w, before_r, gradient, result):
        controls.control_indices.append(index)
        expected = study.arithmetic.ef_step(before_w, before_r, gradient, k=18)
        study.same(result["weights"], expected["weights"], "fake old control weights")
        study.same(result["remainders"], expected["remainders"], "fake old control remainders")

    monkeypatch.setattr(study, "query", fake_query)
    monkeypatch.setattr(study, "collect_scores", fake_scores)
    monkeypatch.setattr(study, "sequential_control", fake_control)
    binding = {"prior_run": str(tmp_path / "old"), "prior_record": str(tmp_path / "old.json"),
        "initial_state": str(initial_path), "dataset": {"path": str(tmp_path / "synthetic.txt")},
        "inputs": [study.old.file_identity(initial_path)]}
    sources = {"synthetic-source": "frozen"}
    monkeypatch.setattr(study, "source_identity", lambda: copy.deepcopy(sources))
    monkeypatch.setattr(study, "bind_inputs", lambda _: (copy.deepcopy(binding), b"synthetic"))
    monkeypatch.setattr(study.pilot, "corpus_plan", lambda *args: (b"synthetic", copy.deepcopy(binding["dataset"])))
    plan = tmp_path / "plan.md"
    plan.write_text("Synthetic protocol fixture, not a scientific declaration\n")
    monkeypatch.setattr(study, "PLAN", plan)
    config = SimpleNamespace(output=tmp_path / "run", record=tmp_path / "record.json",
        prior_run=tmp_path / "old", prior_record=tmp_path / "old.json", corpus=tmp_path / "synthetic.txt")
    return SimpleNamespace(config=config, controls=controls, binding=binding, sources=sources)


def test_fake_lifecycle_exact_queries_commits_exposures_and_readonly_replay(study, fake_run):
    case = fake_run
    result = study.run(case.config)
    assert result["complete"] is True and result["recipe_promoted"] is False
    assert result["paired_shadow_identity"] is True
    assert len(case.controls.queries) == 64
    assert len(case.controls.scores) == 292
    for arm, commits in (("sequential", 32), ("batch4", 8)):
        rows = [row for row in case.controls.queries if row[0] == arm]
        assert [row[1] for row in rows] == list(range(32))
        assert len(list((case.config.output / arm).glob("state_*.npz"))) == commits + 1
        summary = next(row for row in result["arms"] if row["arm"] == arm)
        assert summary["optimizer_commits"] == commits
        assert summary["gradient_queries"] == summary["windows_consumed"] == 32
        assert [row["windows_consumed"] for row in summary["probes"]] == [0, 8, 32]
        if arm == "batch4":
            for start in range(0, 32, 4):
                assert len({row[2] for row in rows[start:start+4]}) == 1
    assert len({row[2] for row in case.controls.queries if row[0] == "sequential"}) > 1
    assert set(row[2] for row in case.controls.scores) <= set(study.WINDOWS) | set(study.PROBES)
    original = case.config.record.read_bytes()
    assert study.check(case.config.output, case.config.record, recompute=True) == result
    assert case.config.record.read_bytes() == original
    assert len(case.controls.queries) == 64  # replay loads retained gradients; no new committed query


@pytest.mark.parametrize("failure", ["query", "final_check", "after_link", "before_link", "foreign_target"])
def test_lifecycle_failure_retains_evidence_and_retracts_only_owned_receipt(study, fake_run, monkeypatch, failure):
    case = fake_run
    previous = study.signal.getsignal(study.signal.SIGALRM)
    if failure == "query":
        case.controls.fail_query = True
    elif failure == "final_check":
        monkeypatch.setattr(study, "check", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("synthetic check failure")))
    else:
        original = study.os.link

        def fail_link(source, target):
            if failure == "after_link":
                original(source, target)
                raise KeyboardInterrupt("synthetic post-link interrupt")
            if failure == "foreign_target":
                Path(target).write_bytes(b"unrelated user receipt")
                original(source, target)
            raise OSError("synthetic pre-link failure")

        monkeypatch.setattr(study.os, "link", fail_link)
    with pytest.raises((RuntimeError, ValueError, OSError, KeyboardInterrupt)):
        study.run(case.config)
    assert study.signal.getsignal(study.signal.SIGALRM) == previous
    assert study.signal.getitimer(study.signal.ITIMER_REAL)[0] == 0
    assert study.read(case.config.output / "failure.json")["complete"] is False
    if failure == "foreign_target":
        assert case.config.record.read_bytes() == b"unrelated user receipt"
    else:
        assert not case.config.record.exists()
    assert (case.config.output / "rejected_public.json").exists() is (failure in ("after_link", "final_check"))
    assert not list(case.config.record.parent.glob(".tt-accumulation-*"))


@pytest.mark.parametrize("mutation", ["missing_input", "initial_redirect", "source", "input_bytes", "recipe"])
def test_frozen_rebuilds_exact_ancestry_not_only_claimed_hashes(study, fake_run, mutation):
    case = fake_run
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "source_hashes": copy.deepcopy(case.sources),
                "binding": copy.deepcopy(case.binding)}
    study.frozen(manifest)
    if mutation == "missing_input":
        manifest["binding"]["inputs"] = []
    elif mutation == "initial_redirect":
        manifest["binding"]["initial_state"] += ".other"
    elif mutation == "source":
        manifest["source_hashes"]["synthetic-source"] = "changed"
    elif mutation == "input_bytes":
        Path(case.binding["initial_state"]).write_bytes(b"changed synthetic state")
    else:
        manifest["recipe"]["k"] = 20
    with pytest.raises(ValueError):
        study.frozen(manifest)


def test_expected_artifact_set_is_complete_and_has_no_unselected_windows(study):
    names = study.expected_artifacts()
    assert len(names) == 402
    assert len([name for name in names if "/query_" in name]) == 64
    assert len([name for name in names if "/score_" in name]) == 292
    assert len([name for name in names if "/state_" in name]) == 42
    assert "batch4/state_001.npz" not in names
    assert "batch4/state_004.npz" in names
    assert all("validation" not in name and "test" not in name for name in names)


@pytest.mark.parametrize("mutation", ["extra", "missing", "public", "seal", "failure", "state", "query"])
def test_completed_run_rejects_artifact_and_receipt_mutations(study, fake_run, mutation):
    case = fake_run
    study.run(case.config)
    if mutation == "extra":
        (case.config.output / "extra.txt").write_text("unsealed")
    elif mutation == "missing":
        (case.config.output / "batch4/query_002.npz").unlink()
    elif mutation == "public":
        value = study.read(case.config.record)
        value["recipe_promoted"] = True
        write_json(case.config.record, value)
    elif mutation == "seal":
        write_json(case.config.output / "artifact_manifest.json", [])
    elif mutation == "failure":
        write_json(case.config.output / "failure.json", {"complete": False})
    else:
        path = case.config.output / "batch4" / ("state_004.npz" if mutation == "state" else "query_002.npz")
        arrays = study.load(path)
        key = "w_EMBED" if mutation == "state" else "EMBED"
        arrays[key].reshape(-1)[0] += 1
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{name: value.numpy() for name, value in arrays.items()})
        # Direct replay isolates the full-state/query chain from outer seal checks.
        with pytest.raises(ValueError):
            study.execute(study.read(case.config.output / "manifest.json"), case.config.output, b"synthetic")
    with pytest.raises(ValueError):
        study.check(case.config.output, case.config.record)


@pytest.mark.parametrize("target", ["public", "seal"])
def test_final_check_rehashes_exact_receipt_bytes_not_only_parsed_json(study, fake_run, monkeypatch, target):
    case = fake_run
    study.run(case.config)
    calls = 0
    original = study.frozen
    path = case.config.record if target == "public" else case.config.output / "artifact_manifest.json"

    def change_after_replay(manifest):
        nonlocal calls
        original(manifest)
        calls += 1
        if calls == 2:
            path.write_bytes(path.read_bytes() + b"\n")

    monkeypatch.setattr(study, "frozen", change_after_replay)
    with pytest.raises(ValueError, match="bytes unchanged"):
        study.check(case.config.output, case.config.record)


@pytest.mark.parametrize("failure", ["binding", "signal_setup", "staging_cleanup"])
def test_setup_and_cleanup_failures_retain_failure_and_restore_alarm(study, fake_run, monkeypatch, failure):
    case = fake_run
    previous = study.signal.getsignal(study.signal.SIGALRM)

    def fail(*args, **kwargs):
        raise OSError("synthetic " + failure)

    if failure == "binding":
        monkeypatch.setattr(study, "bind_inputs", fail)
    elif failure == "signal_setup":
        monkeypatch.setattr(study.signal, "getsignal", fail)
    else:
        original = Path.unlink

        def fail_unlink(path, *args, **kwargs):
            if path.name == "receipt.json" and path.parent.name.startswith(".tt-accumulation-"):
                fail()
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="synthetic " + failure):
        study.run(case.config)
    assert study.read(case.config.output / "failure.json")["complete"] is False
    assert not case.config.record.exists()
    if failure != "signal_setup":
        assert study.signal.getsignal(study.signal.SIGALRM) == previous
    assert study.signal.getitimer(study.signal.ITIMER_REAL)[0] == 0
    if failure != "staging_cleanup":
        assert not case.controls.queries


@pytest.mark.parametrize("existing", ["output", "record"])
def test_existing_paths_remain_untouched_without_execution(study, fake_run, existing):
    path = getattr(fake_run.config, existing)
    path.write_bytes(b"existing user bytes")
    with pytest.raises((FileExistsError, ValueError)):
        study.run(fake_run.config)
    assert path.read_bytes() == b"existing user bytes"
    assert not fake_run.controls.queries


@pytest.mark.parametrize("nested", ["same", "under_raw", "raw_under_record"])
def test_public_and_raw_namespaces_cannot_overlap(study, fake_run, nested):
    config = fake_run.config
    if nested == "same":
        config.record = config.output
    elif nested == "under_raw":
        config.record = config.output / "record.json"
    else:
        config.output = config.record / "run"
    with pytest.raises(ValueError, match="outside"):
        study.run(config)
    assert not fake_run.controls.queries
