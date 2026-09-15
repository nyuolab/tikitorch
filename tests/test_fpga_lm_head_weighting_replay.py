"""Six genuine tiny FP64 trajectories and independent replay, without real data.

The predecessor fixture supplies synthetic corpus/ABI boundaries only. The
new head, fractional forward, all-region VJP, and update paths remain real.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manuscript.tools import diagnose_fpga_lm_head_weighting as study


def _predecessor_fixture():
    path = Path(__file__).with_name("test_fpga_lm_gradient_accumulation_replay.py")
    spec = importlib.util.spec_from_file_location("_head_weighting_synthetic_predecessor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthetic


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    temporary = tmp_path_factory.mktemp("head-weighting-replay")
    generator = _predecessor_fixture().__wrapped__(temporary, patch)
    previous = next(generator)
    try:
        # The inherited fixture's old score spy intentionally accepts only
        # integer-anchor mode. This new audit also needs the genuine frozen
        # reference's fixed-route affine mode for fractional-state autograd.
        spy = study.prior.reference.forward
        captured = dict(zip(spy.__code__.co_freevars, (cell.cell_contents for cell in spy.__closure__)))
        patch.setattr(study.prior.reference, "forward", captured["original_reference"])
        root = temporary / "head-weighting"
        root.mkdir()
        declaration = b"Synthetic unit fixture; not a scientific run.\n"
        (root / "declaration.md").write_bytes(declaration)
        # Filled from actual saved synthetic initial-logit artifacts, never
        # from a replacement constant or new corpus/model evaluation.
        calibration_root = temporary / "synthetic-normalized-prior"
        shared = calibration_root / "shared"
        shared.mkdir(parents=True)
        for index in range(32):
            name = f"score_initial_{index:03d}_1.npz"
            (shared / name).write_bytes((previous.root / "sequential" / name).read_bytes())
        calibration = study.calibration(calibration_root, previous.raw, previous.dataset)
        binding = {"prior_run": str(calibration_root), "prior_record": str(previous.record),
                   "prior_audit": str(previous.record), "dataset": previous.dataset,
                   "initial_state": previous.manifest["binding"]["initial_state"],
                   "calibration": calibration,
                   "closure": {"roots": {}, "files": previous.manifest["binding"]["inputs"]
                               + calibration["source_logits"]}}
        manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "binding": binding,
                    "source_hashes": {"synthetic": "0" * 64}, "run_directory": str(root),
                    "plan_sha256": study.old.digest(declaration)}

        validation = study._Validation(binding, manifest["source_hashes"])

        def frozen(actual, token):
            assert isinstance(token, study._Validation)
            study.equal(actual, manifest, "synthetic complete binding")
            study.equal(token.binding, binding, "synthetic per-invocation binding")
            for item in actual["binding"]["closure"]["files"]:
                study.equal(study.old.file_identity(item["path"]), item, "synthetic input bytes")

        patch.setattr(study, "frozen", frozen)
        patch.setattr(study, "bind_inputs", lambda config: (binding, previous.raw, validation))
        calls = {"forward": 0, "independent_forward": 0, "gradient": 0, "independent_gradient": 0}
        forward, gradient = study.helper.forward, study.helper.gradient

        def traced_forward(weights, tokens, **kwargs):
            assert weights["EMBED"].shape == (4, 4) and len(tokens) == 4
            calls["independent_forward" if kwargs.get("independent") else "forward"] += 1
            return forward(weights, tokens, **kwargs)

        def traced_gradient(weights, tokens, base, head, **kwargs):
            assert weights["EMBED"].shape == (4, 4) and len(tokens) == 4
            calls["independent_gradient" if kwargs.get("independent") else "gradient"] += 1
            return gradient(weights, tokens, base, head, **kwargs)

        patch.setattr(study.helper, "forward", traced_forward)
        patch.setattr(study.helper, "gradient", traced_gradient)
        _write(root / "manifest.json", manifest)
        _write(root / "protocol.json", study.recipe())
        summary = study.execute(manifest, root, previous.raw, write=True)
        _write(root / "summary.json", summary)
        record = temporary / "head-weighting-public.json"

        def reseal():
            artifacts = study.prior.artifact_manifest(root)
            _write(root / "artifact_manifest.json", artifacts)
            _write(record, study.public_record(root, study.read(root / "summary.json"), artifacts))

        reseal()
        yield SimpleNamespace(root=root, record=record, summary=summary, manifest=manifest,
                              initial_w=previous.initial_w, reseal=reseal, calls=calls)
    finally:
        try:
            next(generator)
        except StopIteration:
            pass
        patch.undo()


def test_real_six_cell_fp64_updates_and_independent_replay(synthetic):
    before = dict(synthetic.calls)
    assert before == {"forward": 494, "gradient": 192,
                      "independent_forward": 0, "independent_gradient": 0}
    result = study.check(synthetic.root, synthetic.record)
    assert synthetic.calls == before
    assert result["complete"] is True
    assert result["recipe_promoted"] is False
    assert result["heldout_corpus_evaluated"] is False
    assert result["full_gradient_queries"] == 192
    assert result["optimizer_commits"] == 192
    assert result["complete_saved_states"] == 198
    assert result["score_readouts"] == 302
    assert result["artifact_files_verified"] == 1092
    calibration = synthetic.manifest["binding"]["calibration"]
    assert calibration["tokens"] == 32 * 4
    masses = []
    for index, identity in enumerate(calibration["source_logits"]):
        saved = study.parent.load(Path(identity["path"]))["logits"]
        current = study.parent.load(synthetic.root / "shared" / f"score_initial_{index:03d}_1.npz")["logits"]
        assert torch.equal(current, saved)
        # The scalar is determined by actual saved initial distributions;
        # target choice cannot affect ROM row masses.
        masses.append(study.helper.numeric.loss_head(saved, [0] * 4, "variable_mass")["mass_factor"])
    assert calibration["c"] == float(torch.cat(masses).mean())
    commits = 0
    fractional_arms = set()
    for cell in result["cells"]:
        arm, order = cell["arm"], cell["order"]
        assert cell["c"] == calibration["c"]
        assert cell["training_order"] == study.training_order(order)
        assert [row["position"] for row in cell["query_metrics"]] == list(range(32))
        directory = synthetic.root / arm / order
        weights = {key: value.clone() for key, value in synthetic.initial_w.items()}
        for consumed, index in enumerate(study.training_order(order), 1):
            query = study.parent.load(directory / f"query_{index:03d}.npz")
            transition = study.read(directory / f"transition_{consumed:03d}.json")
            assert transition["c"] == calibration["c"]
            assert transition["window_index"] == index
            assert transition["windows_consumed_after"] == consumed
            assert transition["query"]["weights_sha256"] == study.old.tensor_hash(weights)
            assert not query["head_cotangent"].requires_grad
            weights = {key: weights[key] - query["g_" + key] / (1 << 18) for key in study.WEIGHTS}
            actual = study.state(directory / f"state_{consumed:03d}.npz")
            for key in study.WEIGHTS:
                assert actual[key].dtype is torch.float64
                assert torch.equal(actual[key], weights[key])
                if bool((actual[key] != actual[key].round()).any()):
                    fractional_arms.add(arm)
            commits += 1
    assert commits == 192
    assert fractional_arms == set(study.ARMS)
    assert study.check(synthetic.root, synthetic.record, recompute=True) == result
    assert synthetic.calls["forward"] == before["forward"]
    assert synthetic.calls["gradient"] == before["gradient"]
    assert synthetic.calls["independent_forward"] == 494
    assert synthetic.calls["independent_gradient"] == 192


def test_resealed_fractional_weight_corruption_fails(synthetic):
    path = synthetic.root / "uniform_mass/shuffle_0/state_008.npz"
    original = path.read_bytes()
    try:
        arrays = study.parent.load(path)
        key = next(key for key in arrays if "EMBED" in key)
        arrays[key].reshape(-1)[0] += 0.125
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
        synthetic.reseal()
        with pytest.raises(ValueError):
            study.check(synthetic.root, synthetic.record)
    finally:
        path.write_bytes(original)
        synthetic.reseal()


def test_resealed_query_gradient_corruption_fails(synthetic):
    path = synthetic.root / "relative_mass/consecutive/query_000.npz"
    original = path.read_bytes()
    try:
        arrays = study.parent.load(path)
        arrays["g_EMBED"].reshape(-1)[0] += 0.125
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
        synthetic.reseal()
        with pytest.raises(ValueError):
            study.check(synthetic.root, synthetic.record)
    finally:
        path.write_bytes(original)
        synthetic.reseal()
