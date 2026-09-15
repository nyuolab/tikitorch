"""Genuine tiny fitted-model/Adam trajectories; no corpus or cloud execution."""
from __future__ import annotations

import importlib
import json
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tiny_models(helper):
    """Only the real-corpus/D512 factory boundary is replaced in this fixture."""
    vocab, width = 4, 16
    a, b, c, d = (list(range(offset, offset + vocab)) for offset in range(0, width, vocab))
    raw = {"EMBED": torch.zeros(vocab, width, dtype=torch.float64),
           "MH": torch.zeros(4, width, 3, dtype=torch.float64),
           "MP": torch.full((4, width, width), -8192., dtype=torch.float64),
           "MB": torch.zeros(4, width, dtype=torch.float64),
           "W_HEAD": torch.full((vocab, width), -8192., dtype=torch.float64)}
    for token in range(vocab):
        raw["EMBED"][token, a] = torch.tensor([-4 * ((token + target) % vocab)
                                              for target in range(vocab)], dtype=torch.float64)
    raw["W_HEAD"][torch.arange(vocab), a] = 0
    for source, target in ((a, b), (b, c), (c, d), (d, b)):
        raw["MP"][:, source, target] = 0
    mixed = {name: value.clone() for name, value in raw.items()}
    mixed["MP"].fill_(-8192.)
    for layer in range(4):
        for source, target in ((a, b), (b, c), (c, d), (d, b)):
            destination = [target[(index + (1 << layer)) % vocab] for index in range(vocab)]
            mixed["MP"][layer, source, destination] = 0
    prior = (helper.BETA * raw["EMBED"][:, a]).log_softmax(-1)
    gru = helper.make_gru(seed=0, vocab=vocab, embedding=4, hidden=4)
    return {
        "isolated": {"kind": "tropical", "parameters": helper.dimensionless(raw),
                     "model": None, "prior_logits": None},
        "mixed": {"kind": "tropical", "parameters": helper.dimensionless(mixed),
                  "model": None, "prior_logits": None},
        "prior_gru": {"kind": "gru", "parameters": dict(gru.named_parameters()),
                      "model": gru, "prior_logits": prior},
    }


def test_predecessor_validation_preserves_outer_deadline(tmp_path, monkeypatch):
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_exact_fit")
    root = tmp_path / "parent"
    root.mkdir()
    record, audit, corpus = (tmp_path / name for name in ("record.json", "audit.json", "corpus.txt"))
    for path in (record, audit, corpus):
        path.write_text("synthetic\n")
    dataset = {"path": str(corpus)}
    parent_binding = {"dataset": dataset, "initial_state": str(corpus)}
    _write(root / "manifest.json", {"binding": parent_binding})
    sources, closure = {"synthetic": "0" * 64}, {"roots": {}, "files": []}
    proof = study.head._Validation(parent_binding, sources)
    calls = []
    monkeypatch.setattr(study, "source_identity", lambda: sources)
    monkeypatch.setattr(study.head, "ancestry_snapshot", lambda *args: closure)
    monkeypatch.setattr(study.head, "_bound_config", lambda manifest: object())
    monkeypatch.setattr(study.head, "bind_inputs", lambda config: (parent_binding, b"synthetic", proof))
    monkeypatch.setattr(study, "_audit_receipt", lambda *args: None)
    monkeypatch.setattr(study.pilot, "corpus_plan", lambda *args: (b"synthetic", dataset))

    def semantic_check(*args):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        assert args == (root, record, False, proof, b"synthetic")
        calls.append("complete parent check")

    def timer_owning_wrapper(*args, **kwargs):
        raise AssertionError("the public predecessor wrapper would clobber the outer timer")

    monkeypatch.setattr(study.head, "_check", semantic_check)
    monkeypatch.setattr(study.head, "check", timer_owning_wrapper)
    original_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.setitimer(signal.ITIMER_REAL, 30)
        binding, raw, validation = study.bind_inputs(SimpleNamespace(
            prior_run=root, prior_record=record, prior_audit=audit, corpus=corpus))
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 30
        assert calls == ["complete parent check"]
        assert binding["dataset"] == dataset and raw == b"synthetic"
        assert validation.binding == binding
    finally:
        signal.setitimer(signal.ITIMER_REAL, *original_timer)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_exact_fit")
    patch = pytest.MonkeyPatch()
    temporary = tmp_path_factory.mktemp("exact-fit-replay")
    bound_file = temporary / "synthetic-input.txt"
    bound_file.write_text("Synthetic unit fixture, not corpus evidence.\n")
    declaration = temporary / "PLAN.md"
    declaration.write_text("Synthetic exact-fit test; not a scientific declaration.\n")
    tokens = torch.tensor([[0, 1, 0, 2], [1, 3, 1, 0], [2, 0, 2, 3], [3, 2, 3, 1]], dtype=torch.int64)
    targets = torch.tensor([[1, 0, 2, 1], [3, 1, 0, 3], [0, 2, 3, 0], [2, 3, 1, 2]], dtype=torch.int64)
    sources = {"synthetic": "0" * 64}
    identity = study.old.file_identity(bound_file)
    binding = {"prior_run": str(temporary), "prior_record": str(bound_file),
               "prior_audit": str(bound_file), "initial_state": str(bound_file),
               "dataset": {"path": str(bound_file)}, "closure": {"roots": {}, "files": [identity]}}
    validation = study.head._Validation(binding, sources)

    def frozen(manifest, token):
        assert isinstance(token, study.head._Validation)
        study.equal(manifest["binding"], binding, "tiny complete binding")
        study.equal(token.binding, binding, "tiny private binding")
        study.equal(manifest["source_hashes"], sources, "tiny source identity")
        study.equal(study.old.file_identity(bound_file), identity, "tiny unchanged input")

    patch.setattr(study, "PLAN", declaration)
    patch.setattr(study, "source_identity", lambda: dict(sources))
    patch.setattr(study, "bind_inputs", lambda config: (binding, b"synthetic", validation))
    patch.setattr(study, "frozen", frozen)
    patch.setattr(study, "initial_models", lambda binding: tiny_models(study.helper))
    patch.setattr(study, "batch_data", lambda raw, dataset: (tokens.clone(), targets.clone()))
    calls = {"forward": 0, "independent_forward": 0, "gradient": 0,
             "independent_gradient": 0, "gru": 0, "independent_gru": 0}
    original_forward, original_gradient = study.helper.forward, study.helper.gradient
    original_gru = study.helper.FittedPriorGRU.forward

    def traced_forward(*args, **kwargs):
        calls["independent_forward" if kwargs.get("independent") else "forward"] += 1
        return original_forward(*args, **kwargs)

    def traced_gradient(*args, **kwargs):
        calls["independent_gradient" if kwargs.get("independent") else "gradient"] += 1
        return original_gradient(*args, **kwargs)

    def traced_gru(self, *args, **kwargs):
        calls["independent_gru" if kwargs.get("independent") else "gru"] += 1
        return original_gru(self, *args, **kwargs)

    patch.setattr(study.helper, "forward", traced_forward)
    patch.setattr(study.helper, "gradient", traced_gradient)
    patch.setattr(study.helper.FittedPriorGRU, "forward", traced_gru)
    root, record = temporary / "run", temporary / "public.json"
    config = SimpleNamespace(output=root, record=record, prior_run=temporary,
                             prior_record=bound_file, prior_audit=bound_file, corpus=bound_file)
    try:
        result = study.run(config)

        def reseal():
            artifacts = study.prior.artifact_manifest(root)
            _write(root / "artifact_manifest.json", artifacts)
            _write(record, study.public_record(root, study.read(root / "summary.json"), artifacts))

        yield SimpleNamespace(study=study, root=root, record=record, result=result, calls=calls,
                              tokens=tokens, targets=targets, reseal=reseal, config=config,
                              temporary=temporary, bound_file=bound_file)
    finally:
        patch.undo()


def test_complete_three_arm_fitted_training_and_independent_replay(synthetic):
    value, study = synthetic, synthetic.study
    assert value.result["complete"] is True
    assert value.result["recipe_promoted"] is False
    assert value.result["heldout_corpus_evaluated"] is False
    assert value.result["optimizer_commits"] == 384
    assert value.result["complete_saved_states"] == 387
    # A separate closed-form Adam calculation checks every saved transition,
    # in addition to the runner's exact eager-Adam replay from saved gradients.
    for arm in ("isolated", "mixed", "prior_gru"):
        directory = value.root / arm
        previous = study.load(directory / "state_000000.npz")
        names = [key.removeprefix("parameter/") for key in previous]
        for step in range(1, 129):
            current = study.load(directory / f"state_{step:06d}.npz")
            query = study.load(directory / f"query_{step - 1:06d}.npz")
            for name in names:
                parameter = previous["parameter/" + name]
                gradient = query["g_" + name]
                m = previous.get("exp_avg/" + name, torch.zeros_like(parameter))
                v = previous.get("exp_avg_sq/" + name, torch.zeros_like(parameter))
                next_m = .9 * m + .1 * gradient
                next_v = .999 * v + .001 * gradient.square()
                expected = parameter - (.001 / (1 - .9 ** step)) * next_m / (
                    next_v.sqrt() / ((1 - .999 ** step) ** .5) + 1e-8)
                for key, expected_value in (("parameter/", expected), ("exp_avg/", next_m),
                                            ("exp_avg_sq/", next_v)):
                    torch.testing.assert_close(current[key + name], expected_value, atol=1e-12, rtol=1e-11)
                assert float(current["step/" + name]) == step
            previous = current
    before = dict(value.calls)
    study.check(value.root, value.record)
    assert value.calls == before, "ordinary audit must not issue new model queries"
    audited = study.check(value.root, value.record, recompute=True)
    assert audited == value.result
    assert value.calls["independent_gradient"] - before["independent_gradient"] == 1024
    assert value.calls["independent_forward"] - before["independent_forward"] == 1048
    assert value.calls["independent_gru"] - before["independent_gru"] == 131


def test_resealed_gradient_corruption_is_not_a_valid_training_run(synthetic):
    value, study = synthetic, synthetic.study
    path = sorted((value.root / "isolated").glob("query_*.npz"))[0]
    original = path.read_bytes()
    arrays = study.load(path)
    key = next(key for key in arrays if key.startswith("g_"))
    arrays[key].reshape(-1)[0] += 0.125
    try:
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{name: array.numpy() for name, array in arrays.items()})
        value.reseal()
        with pytest.raises(ValueError):
            study.check(value.root, value.record)
    finally:
        path.write_bytes(original)
        value.reseal()


def test_changed_bound_input_rejected_on_fresh_check(synthetic):
    original = synthetic.bound_file.read_bytes()
    try:
        synthetic.bound_file.write_bytes(original + b"changed")
        with pytest.raises(ValueError):
            synthetic.study.check(synthetic.root, synthetic.record)
    finally:
        synthetic.bound_file.write_bytes(original)


def test_resealed_causal_route_forgery_requires_independent_forward(synthetic):
    value, study = synthetic, synthetic.study
    path = value.root / "isolated" / "query_000001.npz"
    originals = {file: file.read_bytes() for file in value.root.rglob("*.json")}
    original = path.read_bytes()
    old_sha = study.old.digest(original)
    arrays = study.load(path)
    taps = arrays["window_0/TAP_IDX"]
    taps[0, 1, 0] = 1 - taps[0, 1, 0]  # Another legal causal selector at t=1.
    try:
        with path.open("wb") as stream:
            np.savez_compressed(stream, **{name: array.numpy() for name, array in arrays.items()})
        payload = path.read_bytes()

        def forge(row):
            if isinstance(row, dict):
                for hash_key, size_key in (("sha256", "bytes"), ("query_sha256", "query_bytes")):
                    if row.get(hash_key) == old_sha:
                        row[hash_key] = study.old.digest(payload)
                        if size_key in row:
                            row[size_key] = len(payload)
                for item in row.values():
                    forge(item)
            elif isinstance(row, list):
                for item in row:
                    forge(item)

        for file in originals:
            row = study.read(file)
            forge(row)
            _write(file, row)
        value.reseal()
        study.check(value.root, value.record)
        with pytest.raises(ValueError, match="independent"):
            study.check(value.root, value.record, recompute=True)
    finally:
        path.write_bytes(original)
        for file, content in originals.items():
            file.write_bytes(content)
        value.reseal()


def test_failed_execution_does_not_publish(synthetic, monkeypatch):
    study = synthetic.study
    config = SimpleNamespace(**vars(synthetic.config))
    config.output = synthetic.temporary / "failed-run"
    config.record = synthetic.temporary / "failed-public.json"

    def fail(*args, **kwargs):
        raise ValueError("injected prepublication numerical failure")

    monkeypatch.setattr(study, "execute", fail)
    with pytest.raises(ValueError, match="injected prepublication"):
        study.run(config)
    assert not config.record.exists()
    assert (config.output / "failure.json").is_file()
