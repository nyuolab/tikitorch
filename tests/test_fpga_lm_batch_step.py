"""Synthetic batch-step protocol/replay tests; no corpus or scientific run."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_batch_step")


@pytest.fixture(autouse=True)
def no_science(study, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unmocked corpus, native process, or scientific model execution")

    originals = {}
    for owner, name in ((study.extension, "forward"), (study.extension, "gradient"),
                        (study.prior.reference, "forward"), (study.prior.observer, "observe"),
                        (study.pilot, "window"), (study.pilot, "corpus_plan"),
                        (study.old.native, "EFSGD"), (study.old.native, "decode_state")):
        originals[(id(owner), name)] = getattr(owner, name)
        monkeypatch.setattr(owner, name, forbidden)
    return originals


@pytest.fixture
def tiny_models(study, no_science, monkeypatch):
    """Explicitly allow only D2/V3/N4/T3 synthetic forward observations."""
    calls = []
    for owner in (study.extension, study.prior.reference):
        original = no_science[(id(owner), "forward")]

        def bounded(weights, tokens, _original=original, **kwargs):
            assert weights["EMBED"].shape == (3, 2)
            assert weights["MH"].shape == (4, 2, 7)
            assert tokens.tolist() == [0, 1, 2]
            calls.append(kwargs.get("mode", "continuous_floor"))
            return _original(weights, tokens, **kwargs)

        monkeypatch.setattr(owner, "forward", bounded)
    return calls


def mapping(value=0):
    shapes = {"EMBED": (3, 2), "MH": (4, 2, 7), "MP": (4, 2, 2),
              "MB": (4, 2), "W_HEAD": (3, 2)}
    return {name: torch.full(shape, float(value), dtype=torch.float64)
            for name, shape in shapes.items()}


def clone(values):
    return {name: value.clone() for name, value in values.items()}


def metric(ce=2.0, lut=2.1, accuracy=.25):
    return {"true_ce": ce, "lut_nll": lut, "accuracy": accuracy}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_source_map_binds_new_driver_helper_and_tests(study):
    root = Path(__file__).resolve().parents[1]
    sources = study.source_identity()
    expected = {"manuscript/tools/diagnose_fpga_lm_batch_step.py",
                "manuscript/tools/fpga_lm_step_fraction.py",
                "tests/test_fpga_lm_batch_step.py", "tests/test_fpga_lm_step_fraction.py",
                "manuscript/tools/diagnose_fpga_lm_gradient_accumulation.py",
                "manuscript/tools/fpga_lm_accumulation_arithmetic.py"}
    assert expected <= set(sources)
    for name in expected:
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sources[name]


def test_recipe_is_repeatable_and_does_not_share_mutable_lists(study):
    original = copy.deepcopy(study.recipe())
    changed = study.recipe()
    changed["synthetic_test_only"] = True
    assert study.recipe() == original


def test_declared_one_block_four_uncommitted_fractions_and_readout_budget(study):
    assert study.WINDOWS == (4, 5, 6, 7)
    assert study.SHIFTS == (None, 4, 2, 0)
    assert study.LABELS == ("zero", "sixteenth", "quarter", "full")
    assert study.MODELS == ("fresh_affine", "carry_affine", "rounded_affine", "fixed_integer", "dynamic_integer")
    assert study.BASELINES == ("visible", "shadow")
    assert (study.K, study.DEADLINE_S) == (18, 1800)
    recipe = study.recipe()
    assert recipe["partition"] == "train" and recipe["windows"] == [4, 5, 6, 7]
    assert recipe["parent_arm"] == "batch4"
    assert recipe["parent_before_exposures"] == 4
    assert recipe["parent_after_exposures"] == 8
    assert recipe["fractions"] == [0.0, .0625, .25, 1.0]
    assert recipe["readouts"] == 88
    assert recipe["candidate_readouts"] == 80 and recipe["baseline_readouts"] == 8
    assert recipe["recompute_gradient_queries"] == 4
    assert recipe["recompute_gradient_pair_observations"] == 512
    assert recipe["new_training_updates"] == recipe["new_gradient_pair_exposures"] == recipe["new_fitting_pair_exposures"] == 0
    assert recipe["selected_fraction"] is None
    for key in ("candidates_committed", "heldout_corpus_evaluated", "physical_fpga_execution",
                "paid_execution_authorized", "recipe_promoted", "statistical_significance_claim", "adaptive_extension"):
        assert recipe[key] is False


def test_artifact_inventory_requires_every_fraction_model_and_only_declared_block(study):
    files = study.expected_artifacts()
    assert len(files) == 107
    assert {name for name in files if name.startswith("query_")} == {f"query_{i:03d}.npz" for i in (4, 5, 6, 7)}
    assert len([name for name in files if name.startswith("baseline_")]) == 8
    for label in study.LABELS:
        assert len([name for name in files if name.startswith(label + "/")]) == 22
        assert label + "/state.npz" in files
        assert label + "/candidate.json" in files
        for model in study.MODELS:
            assert {f"{label}/{model}_{i:03d}.npz" for i in study.WINDOWS} <= files
    assert "parent_state_004.npz" in files and "parent_state_008.npz" in files
    assert not any("validation" in name or "test" in name for name in files)


@pytest.mark.parametrize("index", [True, False, 4.0, "4", -1, 0, 3, 8, 31, 37, 511])
def test_window_guard_rejects_every_undeclared_request_without_data_access(study, index):
    with pytest.raises(ValueError, match="fixed training windows"):
        study._window(b"unused", {}, index)


@pytest.mark.parametrize("index", [4, 5, 6, 7])
def test_window_guard_uses_training_partition_only(study, monkeypatch, index):
    calls = []
    monkeypatch.setattr(study.pilot, "window", lambda *args: (calls.append(args) or ([0, 1, 2], [1, 2, 0])))
    tokens, targets = study._window(b"synthetic", {"fixture": True}, index)
    assert calls == [(b"synthetic", {"fixture": True}, "train", index)]
    assert tokens.tolist() == [0, 1, 2] and targets == [1, 2, 0]


def test_parent_payload_copy_is_exact_exclusive_and_replay_does_not_write(study, tmp_path):
    source, target = tmp_path / "source.npz", tmp_path / "copied.npz"
    source.write_bytes(b"synthetic opaque parent payload")
    study._copy_exact(source, target, True)
    assert source.read_bytes() == target.read_bytes()
    with pytest.raises(FileExistsError):
        study._copy_exact(source, target, True)
    original = target.read_bytes()
    study._copy_exact(source, target, False)
    assert target.read_bytes() == original
    source.write_bytes(b"changed parent")
    with pytest.raises(ValueError):
        study._copy_exact(source, target, False)
    assert target.read_bytes() == original


def small_state():
    weights, remainders = mapping(), mapping()
    weights["W_HEAD"].fill_(-12)
    weights["W_HEAD"][:, 0] = 0
    remainders["W_HEAD"][:, 0] = 1 << 17
    gradients = []
    for index in range(4):
        gradient = mapping()
        gradient["W_HEAD"][0, 0] = (1 << 16) + index
        gradient["W_HEAD"][1, 0] = -(1 << 15)
        gradients.append(gradient)
    return weights, remainders, gradients


def test_real_tiny_all_five_readouts_and_shadow_baseline_roundtrip(study, tiny_models, tmp_path):
    weights, remainder, gradients = small_state()
    originals = [clone(value) for value in (weights, remainder, *gradients)]
    tokens, targets = torch.tensor([0, 1, 2], dtype=torch.int64), [1, 2, 0]
    visible_row, visible = study.readout(tmp_path / "visible.npz", weights, tokens, targets,
                                          mode="dynamic_integer", write=True, recompute=True)
    base = study._base_from_trace(visible, weights, tokens)
    shadow = study.arithmetic.shadow(weights, remainder, study.K)
    shadow_row, shadow_trace = study.readout(tmp_path / "shadow.npz", shadow, tokens, targets,
                                             mode="shadow", write=True, recompute=True)
    assert not torch.equal(visible["logits"], shadow_trace["logits"])
    assert all(torch.isfinite(torch.tensor(list(row["metrics"].values()))).all()
               for row in (visible_row, shadow_row))
    for label, shift in zip(study.LABELS, study.SHIFTS):
        candidate = study.fraction.step_fraction(weights, remainder, gradients, shift)
        specifications = {
            "fresh_affine": (candidate["fresh_ideal"], "fixed_routes", base),
            "carry_affine": (candidate["ideal"], "fixed_routes", base),
            "rounded_affine": (candidate["weights"], "fixed_routes", base),
            "fixed_integer": (candidate["weights"], "fixed_integer", base),
            "dynamic_integer": (candidate["weights"], "dynamic_integer", None)}
        for model, (candidate_weights, mode, tape) in specifications.items():
            path = tmp_path / f"{label}_{model}.npz"
            row, arrays = study.readout(path, candidate_weights, tokens, targets,
                                        mode=mode, base=tape, write=True, recompute=True)
            before = path.read_bytes()
            replay_row, replay_arrays = study.readout(path, candidate_weights, tokens, targets,
                                                      mode=mode, base=tape, recompute=True)
            assert row == replay_row and path.read_bytes() == before
            study.parent.same(arrays, replay_arrays, "tiny exact replay")
            assert set(arrays) == {"logits", "centering_offsets", "AD_HEAD", "PROJ_IDX", "TAP_IDX"}
            if label == "zero" and model != "carry_affine":
                assert torch.equal(arrays["logits"], visible["logits"])
            if label == "zero" and model == "carry_affine":
                assert torch.equal(arrays["logits"], shadow_trace["logits"])
    for value, original in zip((weights, remainder, *gradients), originals):
        study.parent.same(value, original, "no input mutation")
    assert {"continuous_floor", "integer_anchor", "fixed_routes", "fixed_integer"} <= set(tiny_models)


@pytest.mark.parametrize("mutation", ["missing", "extra", "logit_dtype", "logit_shape", "logit_nan",
                                       "integer_fraction", "integer_clip", "route_dtype", "route_range",
                                       "tap_noncausal", "offset_dtype", "offset_shape", "offset_range"])
def test_trace_archives_reject_incomplete_or_noncanonical_receipts(study, tiny_models, tmp_path, mutation):
    weights, _, _ = small_state()
    tokens, targets = torch.tensor([0, 1, 2], dtype=torch.int64), [1, 2, 0]
    path = tmp_path / "trace.npz"
    _, arrays = study.readout(path, weights, tokens, targets, mode="dynamic_integer", write=True)
    if mutation == "missing":
        arrays.pop("PROJ_IDX")
    elif mutation == "extra":
        arrays["extra"] = torch.zeros(1, dtype=torch.float64)
    elif mutation == "logit_dtype":
        arrays["logits"] = arrays["logits"].to(torch.int64)
    elif mutation == "logit_shape":
        arrays["logits"] = arrays["logits"].flatten()
    elif mutation in ("logit_nan", "integer_fraction", "integer_clip"):
        arrays["logits"][0, 0] = {"logit_nan": float("nan"), "integer_fraction": .5, "integer_clip": 32768}[mutation]
    elif mutation == "route_dtype":
        arrays["AD_HEAD"] = arrays["AD_HEAD"].to(torch.float64)
    elif mutation == "route_range":
        arrays["PROJ_IDX"][0, 0, 0] = 2
    elif mutation == "tap_noncausal":
        arrays["TAP_IDX"][0, 0, 0] = 1
    elif mutation == "offset_dtype":
        arrays["centering_offsets"] = arrays["centering_offsets"].to(torch.int64)
    elif mutation == "offset_shape":
        arrays["centering_offsets"] = arrays["centering_offsets"].flatten()
    else:
        arrays["centering_offsets"][0, 0] = 1.0
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
    with pytest.raises(ValueError):
        study.readout(path, weights, tokens, targets, mode="dynamic_integer")


@pytest.mark.parametrize("field", ["logits", "AD_HEAD", "centering_offsets"])
def test_recompute_rejects_plausible_but_forged_complete_trace(study, tiny_models, tmp_path, field):
    weights, _, _ = small_state()
    tokens, targets = torch.tensor([0, 1, 2], dtype=torch.int64), [1, 2, 0]
    path = tmp_path / "trace.npz"
    _, arrays = study.readout(path, weights, tokens, targets, mode="dynamic_integer", write=True)
    arrays[field].reshape(-1)[0] += .5 if field == "centering_offsets" else 1
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
    # Correct geometry/ranges alone are not a proof of the saved observation.
    with pytest.raises(ValueError, match="recomputed"):
        study.readout(path, weights, tokens, targets, mode="dynamic_integer", recompute=True)


@pytest.mark.parametrize("field", ["AD_HEAD", "PROJ_IDX", "centering_offsets"])
def test_fixed_affine_tape_requires_exact_original_routes_and_offsets(study, tiny_models, tmp_path, field):
    weights, _, _ = small_state()
    tokens, targets = torch.tensor([0, 1, 2], dtype=torch.int64), [1, 2, 0]
    _, visible = study.readout(tmp_path / "visible.npz", weights, tokens, targets, mode="dynamic_integer", write=True)
    base = study._base_from_trace(visible, weights, tokens)
    path = tmp_path / "fixed.npz"
    _, arrays = study.readout(path, weights, tokens, targets, mode="fixed_routes", base=base, write=True)
    arrays[field].reshape(-1)[0] += .5 if field == "centering_offsets" else 1
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
    with pytest.raises(ValueError, match="fixed"):
        study.readout(path, weights, tokens, targets, mode="fixed_routes", base=base)


@pytest.fixture
def fake_run(study, tiny_models, tmp_path, monkeypatch):
    """Real tiny readouts/arithmetic/receipts, synthetic parent and no gradients."""
    weights, remainders, gradients = small_state()
    prior_root, prior_record = tmp_path / "prior", tmp_path / "prior.json"
    (prior_root / "batch4").mkdir(parents=True)
    corpus = tmp_path / "synthetic.txt"
    corpus.write_bytes(b"synthetic token fixture; never read by a corpus loader")
    dataset = {"path": str(corpus), "synthetic": True}
    study.parent.save(prior_root / "batch4/state_004.npz", study.parent.state_arrays(weights, remainders))
    full = study.fraction.step_fraction(weights, remainders, gradients, 0)
    study.parent.save(prior_root / "batch4/state_008.npz", study.parent.state_arrays(full["weights"], full["remainders"]))
    tokens, targets = torch.tensor([0, 1, 2], dtype=torch.int64), [1, 2, 0]
    visible = study.extension.forward(weights, tokens)
    for index, gradient in zip(study.WINDOWS, gradients):
        study.parent.save(prior_root / f"batch4/query_{index:03d}.npz",
            {**{"g_" + key: value for key, value in gradient.items()}, "logits": visible["logits"], **visible["routes"]})
    write_json(prior_root / "manifest.json", {"schema": study.parent.SCHEMA, "binding": {"dataset": dataset}})
    write_json(prior_root / "summary.json", {"complete": True, "synthetic": True})
    write_json(prior_root / "artifact_manifest.json", study.prior.artifact_manifest(prior_root))
    write_json(prior_record, {"complete": True, "synthetic": True})
    controls = SimpleNamespace(queries=[], windows=[], readouts=[], bindings=[])

    def parent_check(root, record, recompute=False):
        assert Path(root) == prior_root and Path(record) == prior_record
        assert recompute is False
        controls.bindings.append((root, record))
        study.equal(study.read(prior_root / "artifact_manifest.json"),
                    study.prior.artifact_manifest(prior_root), "synthetic predecessor seal")
        return study.read(prior_record)

    def corpus_plan(path, steps, validation):
        assert Path(path) == corpus and (steps, validation) == (1024, 88)
        return b"synthetic", copy.deepcopy(dataset)

    def fake_window(raw, supplied, partition, index):
        assert raw == b"synthetic" and supplied == dataset and partition == "train"
        assert type(index) is int and index in (4, 5, 6, 7)
        controls.windows.append(index)
        return tokens.tolist(), targets[:]

    def parent_query(directory, index, current_weights, raw, supplied, *, recompute=False, **kwargs):
        assert not kwargs.get("write", False)
        assert index in study.WINDOWS and raw == b"synthetic" and supplied == dataset
        study.parent.same(current_weights, weights, "every supplied G shares the original point")
        controls.queries.append((index, recompute))
        arrays = study.parent.load(Path(directory) / f"query_{index:03d}.npz")
        result = {key: arrays["g_" + key] for key in study.WEIGHTS}
        study.parent.same(result, gradients[index - 4], "complete synthetic saved parent gradient")
        return result

    original_readout = study.readout

    def counted_readout(*args, **kwargs):
        if kwargs.get("write", False):
            controls.readouts.append((Path(args[0]).name, kwargs["mode"]))
        return original_readout(*args, **kwargs)

    monkeypatch.setattr(study.parent, "check", parent_check)
    monkeypatch.setattr(study.parent, "query", parent_query)
    monkeypatch.setattr(study.pilot, "corpus_plan", corpus_plan)
    monkeypatch.setattr(study.pilot, "window", fake_window)
    monkeypatch.setattr(study, "readout", counted_readout)
    sources = {"synthetic-source": "frozen"}
    monkeypatch.setattr(study, "source_identity", lambda: copy.deepcopy(sources))
    plan = tmp_path / "synthetic_plan.md"
    plan.write_text("Synthetic unit-test declaration; no corpus study\n")
    monkeypatch.setattr(study, "PLAN", plan)
    config = SimpleNamespace(output=tmp_path / "run", record=tmp_path / "record.json",
        prior_run=prior_root, prior_record=prior_record, corpus=corpus)
    return SimpleNamespace(config=config, controls=controls, dataset=dataset, sources=sources,
        weights=weights, remainders=remainders, gradients=gradients, full=full,
        parent_bytes=tree_bytes(prior_root), parent_record_bytes=prior_record.read_bytes())


def test_full_tiny_run_seals_all_fractions_without_commit_and_replays_readonly(study, fake_run):
    case = fake_run
    result = study.run(case.config)
    assert result["complete"] is True and result["completed_fractions"] == 4
    assert result["readouts"] == len(case.controls.readouts) == 88
    assert result["artifact_files_verified"] == 107
    assert result["candidates_committed"] is False and result["selected_fraction"] is None
    assert result["new_training_updates"] == result["new_gradient_pair_exposures"] == 0
    assert result["zero_fraction_original_state_equal"] is True
    assert result["full_fraction_parent_endpoint_equal"] is True
    assert [row["label"] for row in result["candidates"]] == list(study.LABELS)
    assert [row["candidate"]["k"] for row in result["candidates"]] == [18, 22, 20, 18]
    for row in result["candidates"]:
        assert [window["window_index"] for window in row["windows"]] == [4, 5, 6, 7]
        assert all(set(window["models"]) == set(study.MODELS) for window in row["windows"])
        assert row["fraction_committed"] is False
    assert set(case.controls.windows) == {4, 5, 6, 7}
    before = tree_bytes(case.config.output)
    record_before = case.config.record.read_bytes()
    replay = study.check(case.config.output, case.config.record, recompute=True)
    assert replay == result
    assert tree_bytes(case.config.output) == before
    assert case.config.record.read_bytes() == record_before
    assert tree_bytes(case.config.prior_run) == case.parent_bytes
    assert case.config.prior_record.read_bytes() == case.parent_record_bytes
    assert len(case.controls.readouts) == 88
    assert [index for index, recompute in case.controls.queries if recompute] == [4, 5, 6, 7]


def test_real_bind_inputs_has_exact_parent_states_queries_corpus_and_receipts(study, fake_run):
    config = fake_run.config
    binding, raw = study.bind_inputs(config)
    assert raw == b"synthetic"
    expected = {config.prior_run / name for name in ("manifest.json", "summary.json", "artifact_manifest.json")}
    expected |= {config.prior_record, config.corpus, config.prior_run / "batch4/state_004.npz",
                 config.prior_run / "batch4/state_008.npz"}
    expected |= {config.prior_run / f"batch4/query_{index:03d}.npz" for index in study.WINDOWS}
    assert {Path(item["path"]) for item in binding["inputs"]} == expected
    assert len(binding["inputs"]) == len(expected) == 11
    assert binding["state_steps"] == [4, 8] and binding["query_windows"] == [4, 5, 6, 7]
    for item in binding["inputs"]:
        payload = Path(item["path"]).read_bytes()
        assert item["bytes"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("part", ["weights", "remainders"])
def test_full_fraction_rejects_wrong_actual_parent_endpoint(study, fake_run, part):
    path = fake_run.config.prior_run / "batch4/state_008.npz"
    arrays = study.parent.load(path)
    arrays[("w_" if part == "weights" else "r_") + "W_HEAD"][0, 0] += 1
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
    # A synthetic re-sealed parent still cannot bypass the full algebraic anchor.
    write_json(fake_run.config.prior_run / "artifact_manifest.json", study.prior.artifact_manifest(fake_run.config.prior_run))
    with pytest.raises(ValueError, match="parent endpoint"):
        study.run(fake_run.config)
    assert not fake_run.config.record.exists()
    assert study.read(fake_run.config.output / "failure.json")["complete"] is False


@pytest.mark.parametrize("mode", ["fixed_routes", "shadow", "dynamic_integer", "fixed_integer"])
def test_readout_rejects_out_of_range_parameters_before_loading_or_forward(study, tmp_path, mode):
    weights = mapping()
    weights["EMBED"][0, 0] = 32767.5
    with pytest.raises(ValueError, match="clip|range"):
        study.readout(tmp_path / "absent.npz", weights, torch.tensor([0, 1, 2]), [1, 2, 0], mode=mode)


@pytest.mark.parametrize("mutation", ["missing_input", "query_windows", "state_steps", "source", "recipe", "input_bytes"])
def test_frozen_verifies_complete_canonical_parent_binding(study, fake_run, mutation):
    binding, _ = study.bind_inputs(fake_run.config)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(),
                "source_hashes": copy.deepcopy(fake_run.sources), "binding": copy.deepcopy(binding)}
    study.frozen(manifest)
    if mutation == "missing_input":
        manifest["binding"]["inputs"].pop()
    elif mutation == "query_windows":
        manifest["binding"]["query_windows"] = [0, 1, 2, 3]
    elif mutation == "state_steps":
        manifest["binding"]["state_steps"] = [0, 4]
    elif mutation == "source":
        manifest["source_hashes"]["synthetic-source"] = "changed"
    elif mutation == "recipe":
        manifest["recipe"]["selected_fraction"] = "quarter"
    else:
        fake_run.config.corpus.write_bytes(b"changed synthetic corpus bytes")
    with pytest.raises(ValueError):
        study.frozen(manifest)


@pytest.mark.parametrize("field,bad", [("alpha", 1), ("k", True), ("committed", 0),
                                       ("public_reference_equal", 1), ("exact_shadow_recurrence", False)])
def test_candidate_claims_require_literal_types_and_actual_parity(study, field, bad):
    weights, remainder, gradients = small_state()
    candidate = study.fraction.step_fraction(weights, remainder, gradients, 0)
    candidate[field] = bad
    summed = study.parent.accumulation.checked_sum(gradients, mapping())
    with pytest.raises(ValueError):
        study._check_candidate(candidate, weights, remainder, summed, 0)


@pytest.mark.parametrize("failure", ["binding", "candidate", "final_check", "before_link", "after_link",
                                      "foreign_target", "cleanup_disarm", "cleanup_staging"])
def test_failures_retract_only_owned_public_and_retain_private_failure(study, fake_run, monkeypatch, failure):
    config = fake_run.config
    previous = study.signal.getsignal(study.signal.SIGALRM)

    def fail(*args, **kwargs):
        raise OSError("synthetic " + failure)

    if failure == "binding":
        monkeypatch.setattr(study, "bind_inputs", fail)
    elif failure == "candidate":
        original = study.fraction.step_fraction

        def fail_candidate(weights, remainder, gradients, shift):
            if shift == 2:
                raise TimeoutError("synthetic candidate timeout")
            return original(weights, remainder, gradients, shift)

        monkeypatch.setattr(study.fraction, "step_fraction", fail_candidate)
    elif failure == "final_check":
        monkeypatch.setattr(study, "check", fail)
    elif failure in ("before_link", "after_link", "foreign_target"):
        original = study.os.link

        def fail_link(source, target):
            if failure == "after_link":
                original(source, target)
                raise KeyboardInterrupt("synthetic interruption after owned link")
            if failure == "foreign_target":
                Path(target).write_bytes(b"unrelated user receipt")
                return original(source, target)
            fail()

        monkeypatch.setattr(study.os, "link", fail_link)
    elif failure == "cleanup_disarm":
        original, failed = study.signal.setitimer, []

        def fail_disarm(which, seconds, *args):
            if seconds == 0 and not failed:
                failed.append(True)
                fail()
            return original(which, seconds, *args)

        monkeypatch.setattr(study.signal, "setitimer", fail_disarm)
    else:
        original, failed = Path.unlink, []

        def fail_unlink(path, *args, **kwargs):
            if path.name == "receipt.json" and path.parent.name.startswith(".tt-batch-step-") and not failed:
                failed.append(True)
                fail()
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises((OSError, TimeoutError, KeyboardInterrupt)):
        study.run(config)
    assert study.read(config.output / "failure.json")["complete"] is False
    assert study.signal.getsignal(study.signal.SIGALRM) == previous
    assert study.signal.getitimer(study.signal.ITIMER_REAL)[0] == 0
    if failure == "foreign_target":
        assert config.record.read_bytes() == b"unrelated user receipt"
    else:
        assert not config.record.exists()
    assert (config.output / "rejected_public.json").exists() is (failure in
        {"final_check", "after_link", "cleanup_disarm", "cleanup_staging"})
    assert not list(config.record.parent.glob(".tt-batch-step-*"))
    assert tree_bytes(config.prior_run) == fake_run.parent_bytes
    assert config.prior_record.read_bytes() == fake_run.parent_record_bytes


def test_final_check_failure_preserves_concurrently_edited_public_receipt(study, fake_run, monkeypatch):
    def changed_then_fail(*args, **kwargs):
        fake_run.config.record.write_bytes(b"concurrent user revision")
        raise ValueError("controlled final check failure")

    monkeypatch.setattr(study, "check", changed_then_fail)
    with pytest.raises(ValueError, match="controlled"):
        study.run(fake_run.config)
    assert fake_run.config.record.read_bytes() == b"concurrent user revision"
    assert study.read(fake_run.config.output / "failure.json")["complete"] is False
    assert not (fake_run.config.output / "rejected_public.json").exists()


@pytest.mark.parametrize("mutation", ["missing_fraction", "candidate_metadata", "candidate_state", "gradient_sum",
                                       "extra", "failure", "summary", "public"])
def test_completed_record_rejects_missing_or_tampered_sealed_evidence(study, fake_run, mutation):
    config = fake_run.config
    study.run(config)
    if mutation == "missing_fraction":
        (config.output / "sixteenth/candidate.json").unlink()
    elif mutation == "candidate_metadata":
        path = config.output / "quarter/candidate.json"
        row = study.read(path)
        row["committed"] = True
        write_json(path, row)
    elif mutation in ("candidate_state", "gradient_sum"):
        path = config.output / ("quarter/state.npz" if mutation == "candidate_state" else "gradient_sum.npz")
        arrays = study.parent.load(path)
        key = "w_W_HEAD" if mutation == "candidate_state" else "W_HEAD"
        arrays[key].reshape(-1)[0] += 1
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
    elif mutation == "extra":
        (config.output / "extra.npz").write_bytes(b"unselected artifact")
    elif mutation == "failure":
        write_json(config.output / "failure.json", {"complete": False})
    else:
        path = config.output / "summary.json" if mutation == "summary" else config.record
        row = study.read(path)
        row["selected_fraction"] = "quarter"
        write_json(path, row)
    if mutation in ("candidate_metadata", "candidate_state", "gradient_sum"):
        # Bypass the outer seal to exercise full-state/metadata algebra replay.
        with pytest.raises(ValueError):
            study.execute(study.read(config.output / "manifest.json"), config.output, b"synthetic")
    with pytest.raises(ValueError):
        study.check(config.output, config.record)


@pytest.mark.parametrize("target", ["public", "seal"])
def test_check_rehashes_exact_bytes_after_replay_not_only_parsed_values(study, fake_run, monkeypatch, target):
    config = fake_run.config
    study.run(config)
    original, calls = study.frozen, []
    path = config.record if target == "public" else config.output / "artifact_manifest.json"

    def edit_at_end(manifest):
        original(manifest)
        calls.append(True)
        if len(calls) == 2:
            path.write_bytes(path.read_bytes() + b"\n")

    monkeypatch.setattr(study, "frozen", edit_at_end)
    with pytest.raises(ValueError, match="bytes unchanged"):
        study.check(config.output, config.record)


@pytest.mark.parametrize("existing", ["output", "record"])
def test_existing_outputs_are_preserved_without_execution(study, fake_run, existing):
    path = getattr(fake_run.config, existing)
    path.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError):
        study.run(fake_run.config)
    assert path.read_bytes() == b"existing user data"
    assert not fake_run.controls.readouts


@pytest.mark.parametrize("nested", ["same", "public_in_raw", "raw_in_public", "historical_raw", "historical_public"])
def test_new_namespaces_cannot_overlap_or_modify_source_bound_history(study, fake_run, nested):
    config = fake_run.config
    if nested == "same":
        config.record = config.output
    elif nested == "public_in_raw":
        config.record = config.output / "record.json"
    elif nested == "raw_in_public":
        config.output = config.record / "run"
    elif nested == "historical_raw":
        config.output = config.prior_run / "new_run"
    else:
        config.record = config.prior_run / "new_record.json"
    with pytest.raises(ValueError):
        study.run(config)
    assert not fake_run.controls.readouts
    assert tree_bytes(config.prior_run) == fake_run.parent_bytes


@pytest.mark.parametrize("recompute", [1, 0, "true", None])
def test_check_mode_requires_literal_boolean_before_io(study, recompute):
    with pytest.raises(ValueError, match="literal boolean"):
        study.check(Path("does-not-exist"), Path("does-not-exist.json"), recompute=recompute)
