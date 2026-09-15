"""Complete tiny synthetic artifact replay; no corpus, native process, or AWS.

The historical ABI/binding are deliberately mocked. All new forward queries,
independent observer gradients, summed updates, saved states, score readouts,
and artifact replay use real helpers on a four-feature/four-class model.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manuscript.tools import diagnose_fpga_lm_gradient_accumulation as study


def _blocked(*args, **kwargs):
    raise AssertionError("real corpus/native execution is forbidden in synthetic replay tests")


def _rewrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _payload(weights, remainders, step):
    return {"weights": {key: value.reshape(-1).tolist() for key, value in weights.items()},
            "remainders": {key: value.reshape(-1).tolist() for key, value in remainders.items()},
            "step": step, "k": 18, "config": {"loss_kind": 1, "zshift": 2}}


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    shape = SimpleNamespace(ML_D=4, ML_V=4, ML_N_LAYERS=2, ML_K=3,
                            ML_L_SEQ=4, ML_LOG2_D=2)
    raw = b"synthetic fixture bytes only"
    corpus = tmp_path / "synthetic-input.txt"
    corpus.write_bytes(raw)
    dataset = {"path": str(corpus), "synthetic": True}
    allowed = set(study.WINDOWS) | set(study.PROBES)

    def window(actual_raw, actual_dataset, partition, index):
        assert actual_raw == raw and actual_dataset == dataset
        assert partition == "train" and type(index) is int and index in allowed
        tokens = [(index + position) % 4 for position in range(4)]
        targets = [(token + 1 + index % 2) % 4 for token in tokens]
        return tokens, targets

    def corpus_plan(path, steps, evaluation_windows):
        assert Path(path) == corpus and steps == 1024 and evaluation_windows == 88
        return raw, dataset

    monkeypatch.setattr(study.pilot, "window", window)
    monkeypatch.setattr(study.pilot, "corpus_plan", corpus_plan)
    monkeypatch.setattr(study.pilot, "layout", lambda width: shape if width == 512 else _blocked())
    monkeypatch.setattr(study.old.native, "EFSGD", _blocked)
    monkeypatch.setattr(study.old.native, "build", _blocked)
    monkeypatch.setattr(study.old.native, "decode_state", lambda payload, layout: json.loads(payload))
    monkeypatch.setattr(study, "bind_inputs", _blocked)
    original_forward = study.extension.forward
    original_observer = study.prior.observer.observe
    original_reference = study.prior.reference.forward
    calls = {"forward": 0, "observer": 0, "reference": 0}

    def forward(weights, tokens, **kwargs):
        assert weights["EMBED"].shape == (4, 4) and tokens.shape == (4,)
        calls["forward"] += 1
        return original_forward(weights, tokens, **kwargs)

    def observe(weights, tokens, targets, layout, **kwargs):
        assert layout is shape and len(weights["EMBED"]) == 16
        calls["observer"] += 1
        return original_observer(weights, tokens, targets, layout, **kwargs)

    def reference(weights, tokens, **kwargs):
        assert weights["EMBED"].shape == (4, 4) and tokens.shape == (4,)
        assert kwargs["mode"] == "integer_anchor"
        calls["reference"] += 1
        return original_reference(weights, tokens, **kwargs)

    monkeypatch.setattr(study.extension, "forward", forward)
    monkeypatch.setattr(study.prior.observer, "observe", observe)
    monkeypatch.setattr(study.prior.reference, "forward", reference)
    weights = {"EMBED": torch.tensor([[0, -8, -4, -12], [-12, 0, -8, -4],
                                     [-4, -12, 0, -8], [-8, -4, -12, 0]], dtype=torch.float64),
               "MH": torch.zeros((2, 4, 3), dtype=torch.float64),
               "MP": torch.full((2, 4, 4), -32.0, dtype=torch.float64),
               "MB": torch.zeros((2, 4), dtype=torch.float64),
               "W_HEAD": torch.full((4, 4), -32.0, dtype=torch.float64)}
    for index in range(4):
        weights["MP"][:, index, index] = 0
        weights["W_HEAD"][index, index] = 0
    remainders = {key: torch.zeros_like(value) for key, value in weights.items()}
    initial = tmp_path / "initial.json"
    _rewrite(initial, _payload(weights, remainders, 0))
    history = tmp_path / "history"
    directory = history / "diverse"
    directory.mkdir(parents=True)
    current_w, current_r = weights, remainders
    # Construct the synthetic predecessor with the independent NumPy observer
    # and frozen scalar EF helper, not the new query/update implementation.
    for index in study.WINDOWS:
        tokens, targets = window(raw, dataset, "train", index)
        observed = original_observer({key: value.to(torch.int64).reshape(-1).tolist()
                                      for key, value in current_w.items()}, tokens, targets,
                                     shape, zshift=2, include_parameter_gradients=True)
        gradient = {key: torch.tensor(observed["parameter_gradients"][key], dtype=torch.float64)
                    .reshape(current_w[key].shape) for key in study.WEIGHTS}
        step = study.arithmetic.ef_step(current_w, current_r, gradient, k=18)
        study.old.save_arrays(directory / f"gradients_{index + 1:06d}.npz",
            {**{"native_" + key: gradient[key] for key in study.WEIGHTS},
             **{"float_" + key: torch.zeros_like(gradient[key]) for key in study.WEIGHTS}})
        _rewrite(directory / f"step_{index + 1:06d}.json", {
            "before_hashes": {"native": study.old.tensor_hash(current_w),
                              "remainder": study.old.tensor_hash(current_r)},
            "after_hashes": {"native": study.old.tensor_hash(step["weights"]),
                             "remainder": study.old.tensor_hash(step["remainders"])}})
        if index + 1 in study.old.CHECKPOINTS:
            _rewrite(directory / f"native_{index + 1:06d}.i32",
                     _payload(step["weights"], step["remainders"], index + 1))
        current_w, current_r = step["weights"], step["remainders"]
    binding = {"prior_run": str(history), "dataset": dataset, "initial_state": str(initial),
               "inputs": [study.old.file_identity(path) for path in
                          [initial, corpus, *sorted(directory.iterdir())]]}
    root = tmp_path / "run"
    root.mkdir()
    declaration = b"Synthetic artifact fixture; no declared scientific run.\n"
    (root / "declaration.md").write_bytes(declaration)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "binding": binding,
                "source_hashes": {"synthetic": "0" * 64}, "run_directory": str(root),
                "plan_sha256": study.old.digest(declaration)}

    def frozen(actual):
        study.equal(actual, manifest, "synthetic bound manifest")
        for item in actual["binding"]["inputs"]:
            study.equal(study.old.file_identity(item["path"]), item, "synthetic bound input")

    monkeypatch.setattr(study, "frozen", frozen)
    _rewrite(root / "manifest.json", manifest)
    _rewrite(root / "protocol.json", study.recipe())
    summary = study.execute(manifest, root, raw, write=True)
    _rewrite(root / "summary.json", summary)
    record = tmp_path / "public.json"

    def reseal():
        artifacts = study.prior.artifact_manifest(root)
        _rewrite(root / "artifact_manifest.json", artifacts)
        _rewrite(record, study.public_record(root, study.read(root / "summary.json"), artifacts))

    reseal()
    case = SimpleNamespace(root=root, record=record, manifest=manifest, raw=raw, dataset=dataset,
                           initial_w=weights, initial_r=remainders, calls=calls, reseal=reseal)
    yield case
    torch.set_num_threads(previous_threads)


def _mutate_arrays(path, change):
    arrays = study.load(path)
    change(arrays)
    # Overwrite only a disposable test artifact to exercise resealed corruption.
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})


def test_full_synthetic_write_saved_replay_and_independent_recompute(synthetic):
    result = study.check(synthetic.root, synthetic.record)
    assert result["complete"] is True
    assert result["artifact_files_verified"] == 402
    assert [arm["optimizer_commits"] for arm in result["arms"]] == [32, 8]
    assert [arm["gradient_queries"] for arm in result["arms"]] == [32, 32]
    before = dict(synthetic.calls)
    assert study.check(synthetic.root, synthetic.record, recompute=True) == result
    assert synthetic.calls["forward"] == before["forward"]
    assert synthetic.calls["observer"] - before["observer"] == 64
    assert synthetic.calls["reference"] - before["reference"] == 292
    assert result["paired_shadow_identity"] is True
    assert result["recipe_promoted"] is False
    assert all(block["fixed_gradient_partition"]["same_shadow"]
               for block in result["arms"][1]["blocks"])


@pytest.mark.parametrize("relative,field", [
    ("sequential/state_001.npz", "w_EMBED"),
    ("sequential/state_008.npz", "r_MH"),
    ("batch4/state_004.npz", "w_MP"),
    ("batch4/state_032.npz", "r_W_HEAD"),
    ("batch4/state_000.npz", "w_MB"),
])
def test_resealed_full_state_mutation_fails(synthetic, relative, field):
    _mutate_arrays(synthetic.root / relative, lambda arrays: arrays[field].view(-1).__setitem__(0, arrays[field].view(-1)[0] + 1))
    synthetic.reseal()
    with pytest.raises(ValueError):
        study.check(synthetic.root, synthetic.record)


@pytest.mark.parametrize("field,change", [
    ("logits", lambda value: value + 0.5),
    ("logits", lambda value: value.reshape(-1)),
    ("AD_HEAD", lambda value: value.to(torch.float64)),
    ("PROJ_IDX", lambda value: value.reshape(-1)),
    ("TAP_IDX", lambda value: torch.ones_like(value)),
    ("g_MH", lambda value: value + 0.5),
    ("predictions", lambda value: torch.full_like(value, 4)),
])
def test_query_schema_rejects_resealed_malformed_arrays_without_recompute(synthetic, field, change):
    _mutate_arrays(synthetic.root / "batch4/query_000.npz", lambda arrays: arrays.__setitem__(field, change(arrays[field])))
    synthetic.reseal()
    with pytest.raises(ValueError):
        study.check(synthetic.root, synthetic.record)


def test_independent_replay_catches_valid_shape_wrong_route(synthetic):
    path = synthetic.root / "batch4/query_000.npz"
    _mutate_arrays(path, lambda arrays: arrays["AD_HEAD"].view(-1).__setitem__(0, (arrays["AD_HEAD"].view(-1)[0] + 1) % 4))
    synthetic.reseal()
    # Saved-gradient/algebra replay intentionally does not redo route selection.
    assert study.check(synthetic.root, synthetic.record)["complete"] is True
    with pytest.raises(ValueError, match="routes"):
        study.check(synthetic.root, synthetic.record, recompute=True)


@pytest.mark.parametrize("kind", ["missing_query", "missing_score", "missing_state", "extra", "failure", "symlink"])
def test_exact_artifact_membership_is_required(synthetic, kind):
    paths = {"missing_query": "batch4/query_031.npz", "missing_score": "sequential/score_final_031_1.npz",
             "missing_state": "batch4/state_032.npz"}
    if kind in paths:
        (synthetic.root / paths[kind]).unlink()
    elif kind == "symlink":
        (synthetic.root / "unexpected").symlink_to(synthetic.root / "summary.json")
        with pytest.raises(ValueError, match="symlink"):
            study.check(synthetic.root, synthetic.record)
        return
    else:
        (synthetic.root / ("failure.json" if kind == "failure" else "unexpected.json")).write_text("{}\n")
    synthetic.reseal()
    with pytest.raises((ValueError, FileNotFoundError)):
        study.check(synthetic.root, synthetic.record)


def test_sequential_query_is_bound_to_actual_predecessor_gradient(synthetic):
    path = synthetic.root / "sequential/query_000.npz"
    _mutate_arrays(path, lambda arrays: arrays["g_EMBED"].view(-1).__setitem__(0, arrays["g_EMBED"].view(-1)[0] + 1))
    synthetic.reseal()
    with pytest.raises(ValueError, match="historical native gradient"):
        study.check(synthetic.root, synthetic.record)


def test_summary_cannot_promote_or_forge_commit_counts(synthetic):
    summary = study.read(synthetic.root / "summary.json")
    summary["recipe_promoted"] = True
    summary["arms"][1]["optimizer_commits"] = 32
    _rewrite(synthetic.root / "summary.json", summary)
    synthetic.reseal()
    with pytest.raises(ValueError):
        study.check(synthetic.root, synthetic.record)


@pytest.mark.parametrize("which", ["artifact_manifest.json", "public.json"])
def test_byte_only_receipt_mutation_during_final_guard_is_rejected(synthetic, monkeypatch, which):
    original = study.frozen
    calls = 0

    def mutate_on_last(manifest):
        nonlocal calls
        original(manifest)
        calls += 1
        if calls == 2:
            path = synthetic.record if which == "public.json" else synthetic.root / which
            path.write_text(path.read_text() + " \n")

    monkeypatch.setattr(study, "frozen", mutate_on_last)
    with pytest.raises(ValueError):
        study.check(synthetic.root, synthetic.record)
