"""Real tiny-model nine-cell replay; no real corpus, native process, or AWS.

Reuse the frozen synthetic predecessor fixture, including its independent
observer/native-chain construction. Only ancestry/ABI boundaries are mocked;
all forward queries, saved gradients, fixed-k updates and readouts are real.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manuscript.tools import diagnose_fpga_lm_normalized_accumulation as study


def _old_fixture():
    path = Path(__file__).with_name("test_fpga_lm_gradient_accumulation_replay.py")
    spec = importlib.util.spec_from_file_location("_frozen_accumulation_replay_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    temporary = tmp_path_factory.mktemp("normalized-accumulation-replay")
    predecessor_generator = _old_fixture().synthetic.__wrapped__(temporary, patch)
    predecessor = next(predecessor_generator)
    try:
        root = temporary / "normalized"
        root.mkdir()
        declaration = b"Tiny synthetic replay fixture, not a scientific study.\n"
        (root / "declaration.md").write_bytes(declaration)
        binding = {**predecessor.manifest["binding"],
                   "persistent_binding": predecessor.manifest["binding"]}
        manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "binding": binding,
                    "source_hashes": {"synthetic": "0" * 64}, "run_directory": str(root),
                    "plan_sha256": study.old.digest(declaration)}

        def frozen(actual):
            study.equal(actual, manifest, "synthetic complete binding")
            for item in actual["binding"]["inputs"]:
                study.equal(study.old.file_identity(item["path"]), item, "synthetic input bytes")

        patch.setattr(study, "frozen", frozen)
        trace = []
        query = study.parent.query

        def traced_query(directory, index, weights, *args, **kwargs):
            if kwargs.get("write"):
                trace.append((directory.parent.name, directory.name, index,
                              study.old.tensor_hash(weights)))
            return query(directory, index, weights, *args, **kwargs)

        patch.setattr(study.parent, "query", traced_query)
        _write(root / "manifest.json", manifest)
        _write(root / "protocol.json", study.recipe())
        before = dict(predecessor.calls)
        summary = study.execute(manifest, root, predecessor.raw, write=True)
        _write(root / "summary.json", summary)
        record = temporary / "normalized-public.json"

        def reseal():
            artifacts = study.prior.artifact_manifest(root)
            _write(root / "artifact_manifest.json", artifacts)
            _write(record, study.public_record(root, study.read(root / "summary.json"), artifacts))

        reseal()
        yield SimpleNamespace(root=root, record=record, calls=predecessor.calls, before=before,
                              trace=trace, summary=summary, reseal=reseal,
                              initial_w=predecessor.initial_w, initial_r=predecessor.initial_r)
    finally:
        try:
            next(predecessor_generator)
        except StopIteration:
            pass
        patch.undo()


def test_real_nine_cell_replay_and_independent_recompute(synthetic):
    assert synthetic.calls["forward"] - synthetic.before["forward"] == 288 + 434
    before = dict(synthetic.calls)
    result = study.check(synthetic.root, synthetic.record)
    assert synthetic.calls == before
    assert result["complete"] is True
    assert result["artifact_files_verified"] == 1401
    assert result["full_gradient_queries"] == 288
    assert result["optimizer_commits"] == 216
    assert result["complete_saved_states"] == 225
    assert result["score_readouts"] == 434
    assert result["recipe_promoted"] is False
    assert result["heldout_corpus_evaluated"] is False
    assert result["gates"]["broader_study_authorized"] is False

    # Independently replay every committed full state using direct int64
    # signed truncation, not the runner's accumulation helper. A constant Q
    # is used per cell; retained carry is never rescaled between commits.
    total_commits = 0
    k20_has_motion = False
    for cell in result["cells"]:
        arm, order = cell["arm"], cell["order"]
        k, batch = study.ARM_CONFIG[arm]["k"], study.ARM_CONFIG[arm]["batch_size"]
        denominator = 1 << k
        directory = synthetic.root / arm / order
        weights = {key: value.to(torch.int64) for key, value in synthetic.initial_w.items()}
        remainders = {key: value.to(torch.int64) for key, value in synthetic.initial_r.items()}
        cumulative = {key: torch.zeros_like(value) for key, value in weights.items()}
        indices = study.training_order(order)
        traces = [row for row in synthetic.trace if row[:2] == (arm, order)]
        assert [row[2] for row in traces] == indices
        assert [row["window_index"] for row in cell["query_metrics"]] == indices
        assert [row["position"] for row in cell["query_metrics"]] == list(range(32))
        for start in range(0, 32, batch):
            expected_hash = study.old.tensor_hash({key: value.to(torch.float64) for key, value in weights.items()})
            assert all(row[3] == expected_hash for row in traces[start:start + batch])
            gradients = [study.parent.load(directory / f"query_{index:03d}.npz")
                         for index in indices[start:start + batch]]
            for key in study.WEIGHTS:
                summed = sum((row["g_" + key].to(torch.int64) for row in gradients),
                             torch.zeros_like(weights[key]))
                cumulative[key] += summed
                accumulator = remainders[key] + summed
                quotient = torch.where(accumulator >= 0, accumulator // denominator,
                                       -((-accumulator) // denominator))
                weights[key] = weights[key] - quotient
                remainders[key] = accumulator - quotient * denominator
            consumed = start + batch
            actual_w, actual_r = study.state(directory / f"state_{consumed:03d}.npz", k)
            metadata = study.read(directory / f"state_{consumed:03d}.json")
            assert metadata["k"] == k
            assert metadata["windows_consumed"] == consumed
            assert metadata["optimizer_commits"] == consumed // batch
            for key in study.WEIGHTS:
                assert torch.equal(actual_w[key], weights[key].to(torch.float64))
                assert torch.equal(actual_r[key], remainders[key].to(torch.float64))
                assert bool((remainders[key].abs() < denominator).all())
                # Exact dyadic cumulative relation at every equal exposure
                # boundary, for each arm's actually queried gradient stream.
                numerator = (weights[key] - synthetic.initial_w[key].to(torch.int64)) * denominator
                assert torch.equal(numerator - remainders[key] + cumulative[key],
                                   torch.zeros_like(numerator))
                if k == 20 and not torch.equal(actual_w[key], synthetic.initial_w[key]):
                    k20_has_motion = True
            total_commits += 1
    assert total_commits == 216
    assert k20_has_motion, "the synthetic k20 trajectory must exercise actual carry releases"
    assert study.check(synthetic.root, synthetic.record, recompute=True) == result
    assert synthetic.calls["forward"] == before["forward"]
    assert synthetic.calls["observer"] - before["observer"] == 288
    assert synthetic.calls["reference"] - before["reference"] == 434


def test_resealed_k20_carry_mutation_is_rejected(synthetic):
    path = synthetic.root / "batch4_k20/shuffle_0/state_008.npz"
    original = path.read_bytes()
    try:
        arrays = study.parent.load(path)
        arrays["r_EMBED"].reshape(-1)[0] += 1
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
        synthetic.reseal()
        with pytest.raises(ValueError):
            study.check(synthetic.root, synthetic.record)
    finally:
        path.write_bytes(original)
        synthetic.reseal()


def test_resealed_exposure_counter_cannot_be_a_shuffled_window_id(synthetic):
    path = synthetic.root / "batch4_k20/shuffle_1/transition_001.json"
    original = path.read_bytes()
    try:
        transition = study.read(path)
        assert transition["window_indices"] != [0, 1, 2, 3]
        transition["windows_consumed_after"] = transition["window_indices"][-1]
        assert transition["windows_consumed_after"] != 4
        _write(path, transition)
        synthetic.reseal()
        with pytest.raises(ValueError):
            study.check(synthetic.root, synthetic.record)
    finally:
        path.write_bytes(original)
        synthetic.reseal()
