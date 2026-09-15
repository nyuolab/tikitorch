"""Synthetic protocol, artifact and replay tests; no scientific trajectories."""

import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from manuscript.tools import diagnose_fpga_lm_persistent_trajectory as study


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scientific/native execution is forbidden in these tests")

    for name in ("trajectories", "later_probes"):
        if hasattr(study, name):
            monkeypatch.setattr(study, name, forbidden)
    monkeypatch.setattr(study.native, "EFSGD", forbidden)
    monkeypatch.setattr(study.extension, "forward", forbidden)
    monkeypatch.setattr(study.extension, "gradient", forbidden)


def test_declared_protocol_is_four_persistent_32_step_trajectories():
    recipe = study.recipe()
    assert recipe["arms"] == ["historical", "diverse"]
    assert recipe["trajectories"] == ["native_ef", "unrounded_native_rule"]
    assert recipe["training_windows"] == list(range(32))
    assert recipe["updates_per_trajectory"] == 32
    assert recipe["checkpoints"] == [0, 1, 8, 32]
    assert recipe["probe_windows"] == [0, 37, 511]
    assert recipe["checkpoint_models"] == ["native_visible", "native_shadow", "unrounded_native_rule"]
    assert recipe["checkpoint_context"] == [True, False]
    assert recipe["float_control_is_exact_ce_sgd"] is False
    assert recipe["native_updates"] == recipe["fp64_updates"] == 64
    assert recipe["new_gradient_pair_exposures"] == 4 * 32 * 128 == 16384
    assert recipe["distinct_new_gradient_pairs"] == 32 * 128 == 4096
    assert recipe["summed_inherited_fit_exposures"] == 4 * 131072 == 524288
    assert recipe["summed_trajectory_history_exposures"] == 540672
    assert recipe["distinct_history_pairs"] == 131072
    assert recipe["new_count_fits"] == 0
    assert recipe["overall_deadline_s"] == 1800


def test_late_probes_are_same_point_counterfactuals_not_additional_updates():
    recipe = study.recipe()
    assert recipe["anchors"] == [{"source": source, "step": step} for source, step in
                                  [("new", 8), ("new", 32), ("historical", 256), ("historical", 1024)]]
    assert recipe["probe_scales"] == [1.0]
    assert recipe["probe_variants"] == ["exact_ce", "lut_normalized", "truncated_normalized", "variable_mass", "legacy"]
    assert len(study.ARMS) * len(study.ANCHORS) * len(study.PROBES) * len(study.VARIANTS) == 120
    assert recipe["counterfactual_candidates"] == 120
    assert recipe["counterfactual_gradient_pair_evaluations_not_updates"] == 120 * 128
    assert "NOT later checkpoints" in recipe["historical_anchor_scope"]
    assert "preserve anchor r" in recipe["probe_remainders"]
    for name in ("heldout_corpus_evaluated", "physical_fpga_execution", "paid_execution_authorized",
                 "statistical_significance_claim", "adaptive_retry_or_extension"):
        assert recipe[name] is False


def test_array_archive_roundtrip_is_exclusive_and_non_aliasing(tmp_path):
    path = tmp_path / "synthetic.npz"
    values = {"first": torch.arange(6, dtype=torch.float64).reshape(2, 3),
              "second": torch.tensor([-0.25, 7.5], dtype=torch.float64)}
    before = {key: value.clone() for key, value in values.items()}
    record = study.save_arrays(path, values)
    assert record == {"file": path.name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    loaded = study.load_arrays(path, values)
    for key in values:
        assert torch.equal(values[key], before[key]) and torch.equal(loaded[key], values[key])
        assert loaded[key].data_ptr() != values[key].data_ptr()
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        study.save_arrays(path, values)
    assert path.read_bytes() == original


@pytest.mark.parametrize("mutation", ["missing", "extra", "float32", "integer", "nan", "inf", "object"])
def test_archive_rejects_missing_non_fp64_or_nonfinite_tensors(tmp_path, mutation):
    values = {"a": np.asarray([0.5], dtype=np.float64), "b": np.asarray([1.0], dtype=np.float64)}
    if mutation == "missing":
        values.pop("b")
    elif mutation == "extra":
        values["c"] = values["b"]
    elif mutation == "float32":
        values["a"] = values["a"].astype(np.float32)
    elif mutation == "integer":
        values["a"] = values["a"].astype(np.int64)
    elif mutation in ("nan", "inf"):
        values["a"][0] = float(mutation)
    else:
        values["a"] = np.asarray([{"untrusted": True}], dtype=object)
    path = tmp_path / "bad.npz"
    with path.open("xb") as handle:
        np.savez(handle, **values)
    with pytest.raises(ValueError):
        study.load_arrays(path, ["a", "b"])


def test_archive_duplicate_member_is_not_complete_unique_coverage(tmp_path):
    path = tmp_path / "duplicate.npz"
    payload = io.BytesIO()
    np.save(payload, np.asarray([1.0], dtype=np.float64), allow_pickle=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.npy", payload.getvalue())
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("a.npy", payload.getvalue())
    with pytest.raises(ValueError):
        study.load_arrays(path, ["a"])


@pytest.fixture
def tiny_state(monkeypatch):
    shape = SimpleNamespace(ML_V=3, ML_D=4, ML_N_LAYERS=2, ML_K=2)
    dimensions = {"EMBED": (3, 4), "MH": (2, 4, 2), "MP": (2, 4, 4), "MB": (2, 4), "W_HEAD": (3, 4)}
    monkeypatch.setattr(study.pilot, "layout", lambda width: shape)
    weights = {name: torch.zeros(size, dtype=torch.float64) for name, size in dimensions.items()}
    remainders = {name: torch.full(size, 17.0, dtype=torch.float64) for name, size in dimensions.items()}
    gradients = {name: torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape) + 262145
                 for name, value in weights.items()}
    return SimpleNamespace(shape=shape, weights=weights, remainders=remainders, gradients=gradients)


def test_stats_hash_and_exact_zero_preserve_tiny_full_region_coverage(tiny_state):
    values = tiny_state.weights
    assert study.stats(values) == {"l1": 0.0, "l2": 0.0, "max_abs": 0.0, "nonzero_words": 0}
    assert study.require_zero(values, "zero fixture") == 0
    original = study.tensor_hash(values)
    values["MB"][0, 0] = 0.25
    assert study.tensor_hash(values) != original
    assert study.stats(values)["nonzero_words"] == 1
    with pytest.raises(ValueError, match="fixture"):
        study.require_zero(values, "fixture")


def test_native_parity_compares_actual_outputs_and_all_weight_remainders(tiny_state):
    case = tiny_state
    result = study.arithmetic.ef_step(case.weights, case.remainders, case.gradients, k=18)
    after = {"weights": {name: value.reshape(-1).tolist() for name, value in result["weights"].items()},
             "remainders": {name: value.reshape(-1).tolist() for name, value in result["remainders"].items()}}
    base = {"routes": {"AD_HEAD": torch.tensor([[0, 1, 2]]), "PROJ_IDX": torch.tensor([[[0, 1, 2, 3]]]),
                       "TAP_IDX": torch.tensor([[[0, 0, 0, 0]]])}}
    info = {"head": {"predictions": torch.tensor([2]), "native_gradient": torch.tensor([[1, 1, -2]])}}
    outputs = {"predictions": [2], "gradients": [1, 1, -2],
               "native_routes": {name: values.reshape(-1).tolist() for name, values in base["routes"].items()}}
    found_w, found_r = study.native_parity(outputs, base, info, case.gradients, case.weights, case.remainders, after)
    for name in study.WEIGHTS:
        assert torch.equal(found_w[name], result["weights"][name])
        assert torch.equal(found_r[name], result["remainders"][name])
    for field in ("predictions", "gradients", "native_routes", "weight", "remainder"):
        out, state = copy.deepcopy(outputs), copy.deepcopy(after)
        if field == "native_routes":
            out[field]["TAP_IDX"][0] = 1
        elif field in ("predictions", "gradients"):
            out[field][0] += 1
        else:
            state["weights" if field == "weight" else "remainders"]["MP"][-1] += 1
        with pytest.raises(ValueError):
            study.native_parity(out, base, info, case.gradients, case.weights, case.remainders, state)


@pytest.fixture
def input_manifest(sealed_replay):
    return sealed_replay.manifest


def test_verify_inputs_accepts_exact_synthetic_source_and_input_binding(input_manifest):
    study.verify_inputs(input_manifest)


@pytest.mark.parametrize("field", ["schema", "source", "recipe", "input_size", "input_hash", "input_bytes"])
def test_verify_inputs_refuses_mutated_provenance(input_manifest, field):
    row = input_manifest["binding"]["inputs"][0]
    if field == "schema":
        input_manifest["schema"] = "different"
    elif field == "source":
        input_manifest["source_hashes"] = {}
    elif field == "recipe":
        input_manifest["recipe"]["training_windows"] = list(range(1, 33))
    elif field == "input_size":
        row["bytes"] += 1
    elif field == "input_hash":
        row["sha256"] = "b" * 64
    else:
        Path(row["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        study.verify_inputs(input_manifest)


@pytest.fixture
def prior_inputs(tmp_path, monkeypatch):
    previous, historical = tmp_path / "learning-contract", tmp_path / "projection-study"
    previous.mkdir()
    historical.mkdir()
    record, older_record = tmp_path / "learning.json", tmp_path / "projection.json"
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"synthetic fixture; no corpus parser is called")
    write_json(record, {"synthetic": True})
    write_json(older_record, {"synthetic": True})
    write_json(previous / "manifest.json", {"prior": {"prior_run": str(historical), "prior_record": str(older_record)}})
    write_json(previous / "summary.json", {"synthetic": True})
    write_json(previous / "artifact_manifest.json", [])
    decoded = {arm: {"synthetic_initial": arm} for arm in study.ARMS}
    binding = {"prior_run": str(historical), "inputs": [study.file_identity(corpus)]}
    calls = []

    def checked(run, public):
        calls.append(("checked", Path(run), Path(public)))
        return {"recipe": study.prior.recipe()}

    def bound(run, public, path):
        calls.append(("bound", Path(run), Path(public), Path(path)))
        return copy.deepcopy(binding), b"synthetic bytes", copy.deepcopy(decoded)

    monkeypatch.setattr(study.prior, "check", checked)
    monkeypatch.setattr(study.prior, "prior_binding", bound)
    monkeypatch.setattr(study.native, "decode_state", lambda data, shape: json.loads(data))
    for arm in study.ARMS:
        for step in (256, 1024):
            write_json(historical / (arm + "_seed_0") / ("state_%06d.i32" % step),
                       {"step": step, "k": 18, "config": {"loss_kind": 1, "zshift": 2}})
    config = SimpleNamespace(prior_run=previous, prior_record=record, corpus=corpus)
    return SimpleNamespace(config=config, previous=previous, historical=historical, record=record,
                           older_record=older_record, corpus=corpus, decoded=decoded, binding=binding, calls=calls)


def test_prior_mapping_uses_complete_predecessor_and_exact_seed0_late_states(prior_inputs):
    case = prior_inputs
    binding, raw, decoded = study.bind_inputs(case.config)
    assert raw == b"synthetic bytes" and decoded == case.decoded
    assert case.calls == [("checked", case.previous, case.record),
                          ("bound", case.historical, case.older_record, case.corpus)]
    assert len(binding["inputs"]) == 9
    for arm in study.ARMS:
        for step in (256, 1024):
            assert binding["late_states"][arm][str(step)] == str(case.historical / (arm + "_seed_0") / ("state_%06d.i32" % step))
    for item in binding["inputs"]:
        assert study.file_identity(item["path"]) == item


@pytest.mark.parametrize("field,value", [("step", 255), ("step", 256.0), ("k", 17), ("k", 18.0),
                                         ("loss_kind", 0), ("loss_kind", True), ("zshift", 4)])
def test_late_anchor_config_must_match_exact_declared_native_state(prior_inputs, field, value):
    path = prior_inputs.historical / "historical_seed_0/state_000256.i32"
    state = study.read(path)
    if field in ("step", "k"):
        state[field] = value
    else:
        state["config"][field] = value
    write_json(path, state)
    with pytest.raises(ValueError):
        study.bind_inputs(prior_inputs.config)


def test_invalid_predecessor_stops_before_late_state_binding(prior_inputs, monkeypatch):
    def fail(*args):
        raise ValueError("incomplete predecessor")

    monkeypatch.setattr(study.prior, "check", fail)
    with pytest.raises(ValueError, match="incomplete"):
        study.bind_inputs(prior_inputs.config)
    assert not prior_inputs.calls


@pytest.mark.parametrize("failure", [None, "changed_initial_float", "mutated_state"])
def test_checkpoint_checks_initial_logits_and_full_state_invariance(tmp_path, monkeypatch, tiny_state, failure):
    class Endpoint:
        payload = b"synthetic full state"

        def save_checkpoint(self, root, stem):
            path = root / (stem + ".i32")
            path.write_bytes(self.payload)
            return {"file": path.name, "bytes": len(self.payload), "sha256": study.digest(self.payload)}

        def export_state(self):
            return self.payload

    endpoint = Endpoint()
    monkeypatch.setattr(study, "PROBES", (2, 5))
    monkeypatch.setattr(study.pilot, "window", lambda *args: ([0, 1], [1, 0]))
    weights = tiny_state.weights
    remainders = {name: torch.zeros_like(value) for name, value in weights.items()}
    floating = study.prior.clone(weights)
    if failure == "changed_initial_float":
        floating["EMBED"][0, 0] = 0.5
    calls = []

    def synthetic_score(values, tokens, targets, use_context=True):
        calls.append(use_context)
        if failure == "mutated_state":
            endpoint.payload = b"changed by probe"
        signature = float(values["EMBED"][0, 0])
        return {"true_ce": 1.0, "lut_nll": 1.25, "accuracy": 0.5}, {
            "logits": torch.tensor([[signature, 0.0]], dtype=torch.float64),
            "routes": {name: torch.tensor([0]) for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}

    monkeypatch.setattr(study, "score", synthetic_score)
    if failure:
        with pytest.raises(ValueError):
            study.checkpoint(tmp_path, 0, endpoint, weights, remainders, floating, b"", {})
    else:
        record = study.checkpoint(tmp_path, 0, endpoint, weights, remainders, floating, b"", {})
        assert record["step"] == 0 and record["inference_state_preserved"] is True
        assert len(record["probe_scores"]) == 6
        assert calls == [True, False] * 6
        assert endpoint.payload == b"synthetic full state"


@pytest.fixture
def sealed_replay(tmp_path, monkeypatch, tiny_state):
    """Complete tiny synthetic optimizer-algebra record, without a forward pass."""
    torch.set_num_threads(1)
    monkeypatch.setattr(study, "STEPS", 2)
    monkeypatch.setattr(study, "CHECKPOINTS", (0, 1, 2))
    monkeypatch.setattr(study, "PROBES", (2,))
    monkeypatch.setattr(study, "ANCHORS", (("new", 1), ("historical", 256)))
    sources = {"synthetic-test-source": "c" * 64}
    monkeypatch.setattr(study, "source_identity", lambda: copy.deepcopy(sources))
    monkeypatch.setattr(study.native, "source_identity", lambda: {"synthetic-native": "d" * 64})
    monkeypatch.setattr(study.native, "decode_state", lambda data, shape: json.loads(data))
    monkeypatch.setattr(study.pilot, "layout_values", lambda shape: [shape.ML_D, shape.ML_V])
    for name, values in tiny_state.weights.items():
        setattr(tiny_state.shape, "WORDS_" + name, values.numel())
    root, inputs = tmp_path / "run", tmp_path / "inputs"
    root.mkdir()
    inputs.mkdir()
    original, learning = inputs / "original", inputs / "learning"
    original.mkdir()
    learning.mkdir()
    original_record, learning_record = inputs / "original.json", inputs / "learning.json"
    corpus = inputs / "synthetic-corpus.txt"
    corpus.write_bytes(b"Not parsed; synthetic provenance fixture.")
    for path in (original_record, learning_record, original / "manifest.json", original / "summary.json",
                 original / "artifact_manifest.json", learning / "summary.json", learning / "artifact_manifest.json"):
        write_json(path, {"synthetic": True})
    executable = inputs / "never-executed"
    executable.write_bytes(b"synthetic executable identity only")
    write_json(inputs / "build.json", {"synthetic": True})
    identity = {"schema": study.native.SCHEMA, "executable_sha256": study.digest(executable.read_bytes()),
                "build_sha256": study.digest((inputs / "build.json").read_bytes()),
                "source_hashes": study.native.source_identity(), "layout": study.pilot.layout_values(tiny_state.shape),
                "cpu_only": True, "existing_afi_compatible": False}
    native_config = {"loss_kind": 1, "zshift": 2, "lut": [3, 2, 1]}

    def save_state(path, step, weights, remainders):
        write_json(path, {"step": step, "k": study.K, "config": native_config,
                         "weights": {key: value.reshape(-1).tolist() for key, value in weights.items()},
                         "remainders": {key: value.reshape(-1).tolist() for key, value in remainders.items()}})
        return {"schema": study.native.SCHEMA, "file": path.name, "bytes": path.stat().st_size,
                "sha256": study.digest(path.read_bytes()), "step": step, "k": study.K,
                "config": copy.deepcopy(native_config), "identity": copy.deepcopy(identity),
                "regions": [{"name": key, "weight_words": value.numel(), "remainder_words": value.numel()}
                            for key, value in weights.items()],
                "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}

    binding = {"initial": {"states": {}, "executable": study.file_identity(executable),
                            "prior_run": str(original), "prior_record": str(original_record), "inputs": []},
               "late_states": {}, "inputs": [], "learning_contract_run": str(learning),
               "learning_contract_record": str(learning_record)}
    summary = {"schema": study.SCHEMA, "complete": True, "recipe": study.recipe(), "source_hashes": sources,
               "trajectories": [], "probes": [], "heldout_corpus_evaluated": False,
               "physical_fpga_execution": False, "paid_execution_authorized": False,
               "scope": "CPU learning-rule diagnostic; no held-out quality or FPGA claim"}
    probe_dir = root / "probes"
    probe_dir.mkdir()
    for arm_index, arm in enumerate(study.ARMS):
        directory = root / arm
        directory.mkdir()
        weights = {key: value + arm_index * 4 for key, value in tiny_state.weights.items()}
        remainders = {key: torch.zeros_like(value) for key, value in weights.items()}
        floating = study.prior.clone(weights)
        initial_path = original / (arm + "_seed_0/state_000000.i32")
        save_state(initial_path, 0, weights, remainders)
        binding["initial"]["states"][arm] = str(initial_path)
        # The historical anchor is deliberately not a new-trajectory checkpoint.
        historical_path = original / (arm + "_seed_0/state_000256.i32")
        historical_r = {key: torch.full_like(value, 262143) for key, value in weights.items()}
        save_state(historical_path, 256, weights, historical_r)
        other_path = original / (arm + "_seed_0/state_001024.i32")
        save_state(other_path, 1024, weights, historical_r)
        binding["late_states"][arm] = {"256": str(historical_path), "1024": str(other_path)}
        pair = {"arm": arm, "updates_per_trajectory": study.STEPS, "steps": [], "checkpoints": []}
        for step in range(study.STEPS + 1):
            receipt = save_state(directory / ("native_%06d.i32" % step), step, weights, remainders)
            float_receipt = study.save_arrays(directory / ("float_%06d.npz" % step), floating)
            pair["checkpoints"].append({"step": step, "native": receipt, "float": float_receipt,
                "float_tensor_sha256": study.tensor_hash(floating), "inference_state_preserved": True,
                "probe_scores": [{"model": model, "window_index": 2,
                    "context_on": {"true_ce": 1.0, "lut_nll": 1.25},
                    "context_off": {"true_ce": 1.25, "lut_nll": 1.75},
                    "context_benefit": {"true_ce": 0.25, "lut_nll": 0.5}}
                    for model in study.recipe()["checkpoint_models"]]})
            if step == study.STEPS:
                break
            # Exact dyadic recurrence with distinct native/float gradients and nonzero carry.
            ng = {key: torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape) + 131079 + step
                  for key, value in weights.items()}
            fg = {key: value - 3 for key, value in ng.items()}
            arrays = {prefix + key: values[key] for prefix, values in (("native_", ng), ("float_", fg))
                      for key in study.WEIGHTS}
            archive = study.save_arrays(directory / ("gradients_%06d.npz" % (step + 1)), arrays)
            before_shadow = study.arithmetic.shadow(weights, remainders, k=study.K)
            result = study.arithmetic.ef_step(weights, remainders, ng, k=study.K)
            after_w, after_r = result["weights"], result["remainders"]
            after_f = study.arithmetic.unrounded_step(floating, fg, k=study.K)
            after_shadow = study.arithmetic.shadow(after_w, after_r, k=study.K)
            row = {"step": step + 1, "window_index": step, "gradient_file": archive,
                "native_full_gradient_weight_remainder_parity": True,
                "shadow_recurrence_error": 0.0, "paired_gradient_identity_error": 0.0,
                "before_hashes": {"native": study.tensor_hash(weights), "remainder": study.tensor_hash(remainders),
                                  "float": study.tensor_hash(floating)},
                "after_hashes": {"native": study.tensor_hash(after_w), "remainder": study.tensor_hash(after_r),
                                 "float": study.tensor_hash(after_f)},
                "native_gradient_vs_float": study.prior.gradient_stats(ng, fg)}
            for label, left, right in (("native_visible_motion", after_w, weights),
                ("native_shadow_motion", after_shadow, before_shadow), ("float_motion", after_f, floating),
                ("pending_native_motion", after_w, after_shadow), ("native_shadow_minus_float", after_shadow, after_f),
                ("native_visible_minus_float", after_w, after_f)):
                row[label] = study.stats(study.subtract(left, right))
            pair["steps"].append(row)
            weights, remainders, floating = after_w, after_r, after_f
        summary["trajectories"].append(pair)
        for source, step in study.ANCHORS:
            path = directory / ("native_%06d.i32" % step) if source == "new" else historical_path
            weights, remainders = study.state_tensors(study.native.decode_state(path.read_bytes(), tiny_state.shape))
            stem = "%s_%s_%06d_%03d" % (arm, source, step, 2)
            # Nonzero carry changes the rounded counterfactual versus a fresh accumulator.
            gradient = {key: torch.full_like(value, 11) for key, value in weights.items()}
            rounded = {key: weights[key] - torch.trunc((remainders[key] + gradient[key]) / (1 << study.K))
                       for key in study.WEIGHTS}
            arrays = {variant + "_" + key: gradient[key] for variant in study.VARIANTS for key in study.WEIGHTS}
            summary["probes"].append({"arm": arm, "source": source, "step": step, "window_index": 2,
                "anchor": study.file_identity(path), "initial_remainder_sha256": study.tensor_hash(remainders),
                "pending_motion": study.stats(study.subtract(weights, study.arithmetic.shadow(weights, remainders, k=study.K))),
                "gradient_query_point": "visible anchor weights, not shadow weights",
                "independent_raw_logits_routes_head_parameter_gradient_equal": True, "candidate_committed": False,
                "gradient_file": study.save_arrays(probe_dir / (stem + ".npz"), arrays),
                "candidates": [{"variant": variant, "scale": 1.0, "raw_gradient_sha256": study.tensor_hash(gradient),
                    "gradient_statistics": study.prior.gradient_stats(gradient, gradient),
                    "rounded_weight_sha256": study.tensor_hash(rounded),
                    "changed_weight_words": study.stats(study.subtract(rounded, weights))["nonzero_words"],
                    "legacy_ef_algebra_checked": True if variant == "legacy" else None} for variant in study.VARIANTS]})
    initial_paths = [original_record, original / "manifest.json", original / "summary.json",
                     original / "artifact_manifest.json", executable, inputs / "build.json", corpus,
                     *[Path(path) for path in binding["initial"]["states"].values()]]
    binding["initial"]["inputs"] = [study.file_identity(path) for path in initial_paths]
    write_json(learning / "manifest.json", {"prior": binding["initial"]})
    binding["inputs"] = [study.file_identity(path) for path in sorted(inputs.rglob("*")) if path.is_file()]
    declaration = b"Synthetic two-step fixture; not an experimental declaration.\n"
    (root / "declaration.md").write_bytes(declaration)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "source_hashes": sources,
                "binding": binding, "run_directory": str(root), "plan_sha256": study.digest(declaration)}
    record = tmp_path / "public.json"

    def reseal():
        write_json(root / "manifest.json", manifest)
        write_json(root / "summary.json", summary)
        write_json(root / "protocol.json", study.recipe())
        for pair in summary["trajectories"]:
            directory = root / pair["arm"]
            for row in pair["steps"]:
                write_json(directory / ("step_%06d.json" % row["step"]), row)
            for row in pair["checkpoints"]:
                write_json(directory / ("checkpoint_%06d.json" % row["step"]), row)
                write_json(directory / ("native_%06d.json" % row["step"]), row["native"])
        for row in summary["probes"]:
            stem = "%s_%s_%06d_%03d" % (row["arm"], row["source"], row["step"], row["window_index"])
            write_json(probe_dir / (stem + ".json"), row)
        artifacts = study.prior.artifact_manifest(root)
        write_json(root / "artifact_manifest.json", artifacts)
        write_json(record, {**summary, "run_directory": str(root),
            "artifact_manifest_sha256": study.digest((root / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "check_scope": "saved full-gradient optimizer algebra replay plus integrity; no forward or training replay"})

    reseal()
    assert study.check(root, record)["complete"] is True
    return SimpleNamespace(root=root, record=record, summary=summary, manifest=manifest, reseal=reseal,
                           save_state=save_state, shape=tiny_state.shape)


def test_complete_small_replay_checks_full_arrays_states_and_publication_without_forward(sealed_replay):
    case = sealed_replay
    result = study.check(case.root, case.record)
    assert len(result["trajectories"]) == 2
    assert sum(len(row["steps"]) for row in result["trajectories"]) == 4
    assert len(result["probes"]) == 4
    assert sum(len(row["candidates"]) for row in result["probes"]) == 20
    assert result["check_scope"] == "saved full-gradient optimizer algebra replay plus integrity; no forward or training replay"
    actual = study.prior.artifact_manifest(case.root)
    assert result["artifact_files_verified"] == len(actual)
    assert result["artifact_payload_bytes"] == sum(row["bytes"] for row in actual)


@pytest.mark.parametrize("field", ["pair", "step", "checkpoint", "probe", "variant", "native_parity",
    "inference_flag", "candidate_committed", "legacy_parity", "initial_remainder", "context_benefit",
    "motion", "gradient_comparison", "step_bool", "checkpoint_bool", "scale_bool", "scale_int", "receipt_regions", "receipt_identity"])
def test_resealed_partial_or_forged_receipts_fail_full_replay(sealed_replay, field):
    case = sealed_replay
    pair, probe = case.summary["trajectories"][0], case.summary["probes"][0]
    row, checkpoint_row = pair["steps"][0], pair["checkpoints"][1]
    if field in ("pair", "step", "checkpoint", "probe", "variant"):
        {"pair": case.summary["trajectories"], "step": pair["steps"], "checkpoint": pair["checkpoints"],
         "probe": case.summary["probes"], "variant": probe["candidates"]}[field].pop()
    elif field == "native_parity":
        row["native_full_gradient_weight_remainder_parity"] = 1
    elif field == "inference_flag":
        checkpoint_row["inference_state_preserved"] = 1
    elif field == "candidate_committed":
        probe["candidate_committed"] = True
    elif field == "legacy_parity":
        probe["candidates"][-1]["legacy_ef_algebra_checked"] = 1
    elif field == "initial_remainder":
        probe["initial_remainder_sha256"] = "0" * 64
    elif field == "context_benefit":
        checkpoint_row["probe_scores"][0]["context_benefit"]["true_ce"] += 0.25
    elif field == "motion":
        row["native_shadow_minus_float"]["l1"] += 1.0
    elif field == "gradient_comparison":
        row["native_gradient_vs_float"]["EMBED"]["l1"] += 1.0
    elif field == "step_bool":
        row["step"] = True
    elif field == "checkpoint_bool":
        checkpoint_row["step"] = True
    elif field == "scale_bool":
        probe["candidates"][0]["scale"] = True
    elif field == "scale_int":
        probe["candidates"][0]["scale"] = 1
    elif field == "receipt_regions":
        checkpoint_row["native"]["regions"].pop()
    else:
        checkpoint_row["native"]["identity"]["cpu_only"] = 1
    case.reseal()
    with pytest.raises(ValueError):
        study.check(case.root, case.record)


@pytest.mark.parametrize("field", ["native_weight", "native_remainder", "native_config", "native_gradient",
                                  "float_gradient", "float_weight", "float_shape", "probe_carry"])
def test_resealed_full_payload_mutations_cannot_fake_optimizer_replay(sealed_replay, field):
    case = sealed_replay
    pair = case.summary["trajectories"][0]
    checkpoint_row = pair["checkpoints"][-1]
    directory = case.root / pair["arm"]
    if field.startswith("native_") and field != "native_gradient":
        path = directory / checkpoint_row["native"]["file"]
        state = study.read(path)
        if field == "native_config":
            state["k"] = 17
        else:
            state["weights" if field == "native_weight" else "remainders"]["MP"][-1] += 1
        write_json(path, state)
        checkpoint_row["native"]["bytes"] = path.stat().st_size
        checkpoint_row["native"]["sha256"] = study.digest(path.read_bytes())
    elif field in ("native_gradient", "float_gradient"):
        row = pair["steps"][0]
        path = directory / row["gradient_file"]["file"]
        keys = [prefix + key for prefix in ("native_", "float_") for key in study.WEIGHTS]
        values = study.load_arrays(path, keys)
        values["native_MP" if field == "native_gradient" else "float_MP"][-1, -1, -1] += 1
        # Replace only a disposable test artifact; re-seal its full byte receipt.
        path.unlink()
        row["gradient_file"] = study.save_arrays(path, values)
    elif field in ("float_weight", "float_shape"):
        path = directory / checkpoint_row["float"]["file"]
        values = study.load_arrays(path, study.WEIGHTS)
        if field == "float_weight":
            values["MP"][-1, -1, -1] += 2 ** -18
        else:
            values["MB"] = values["MB"].reshape(-1)
        path.unlink()
        checkpoint_row["float"] = study.save_arrays(path, values)
        checkpoint_row["float_tensor_sha256"] = study.tensor_hash(values)
    else:
        probe = next(row for row in case.summary["probes"] if row["source"] == "historical")
        state = study.native.decode_state(Path(probe["anchor"]["path"]).read_bytes(), case.shape)
        weights, _ = study.state_tensors(state)
        # Gradient 11 is insufficient without the existing 262143 remainder.
        for item in probe["candidates"]:
            item["rounded_weight_sha256"] = study.tensor_hash(weights)
            item["changed_weight_words"] = 0
    case.reseal()
    with pytest.raises(ValueError):
        study.check(case.root, case.record)


@pytest.mark.parametrize("field", ["source", "input", "declaration", "unlisted", "failure", "public_count", "public_scope"])
def test_sealed_check_rejects_changed_sources_inputs_membership_or_public_claims(sealed_replay, field):
    case = sealed_replay
    if field == "source":
        case.manifest["source_hashes"] = {}
        case.reseal()
    elif field == "input":
        Path(case.manifest["binding"]["inputs"][0]["path"]).write_bytes(b"changed")
    elif field == "declaration":
        (case.root / "declaration.md").write_bytes(b"different declaration")
    elif field == "unlisted":
        (case.root / "unlisted.bin").write_bytes(b"undeclared")
    elif field == "failure":
        write_json(case.root / "failure.json", {"complete": False})
        case.reseal()
    else:
        public = study.read(case.record)
        if field == "public_count":
            public["artifact_files_verified"] += 1
        else:
            public["check_scope"] = "independently reproduced training"
        write_json(case.record, public)
    with pytest.raises(ValueError):
        study.check(case.root, case.record)


@pytest.mark.parametrize("relative", ["../outside", "/outside", "./payload.bin", "nested/../payload.bin", ""])
def test_local_artifacts_require_canonical_in_root_paths(tmp_path, relative):
    with pytest.raises((ValueError, FileNotFoundError)):
        study.local_file(tmp_path, relative)


def test_artifact_symlinks_are_rejected_even_with_resealed_matching_bytes(sealed_replay):
    case = sealed_replay
    path = case.root / "link.bin"
    path.symlink_to(case.record)
    with pytest.raises(ValueError, match="symlink"):
        study.prior.artifact_manifest(case.root)
    with pytest.raises(ValueError):
        study.local_file(case.root, path.name)


@pytest.mark.parametrize("field", ["initial", "late"])
def test_referenced_states_cannot_be_redirected_to_unbound_identical_files(sealed_replay, field):
    case = sealed_replay
    binding = case.manifest["binding"]
    mapping = binding["initial"]["states"] if field == "initial" else binding["late_states"]["historical"]
    key = "historical" if field == "initial" else "256"
    original = Path(mapping[key])
    redirected = original.with_name("unbound-" + original.name)
    redirected.write_bytes(original.read_bytes())
    mapping[key] = str(redirected)
    if field == "late":
        for row in case.summary["probes"]:
            if row["source"] == "historical" and row["arm"] == "historical":
                row["anchor"] = study.file_identity(redirected)
    case.reseal()
    with pytest.raises(ValueError):
        study.check(case.root, case.record)


@pytest.mark.parametrize("mutation", ["duplicate_input", "unbound_executable", "missing_arm", "predecessor_binding"])
def test_input_inventory_and_predecessor_binding_are_complete_and_unique(sealed_replay, mutation):
    case = sealed_replay
    binding = case.manifest["binding"]
    if mutation == "duplicate_input":
        binding["inputs"].append(copy.deepcopy(binding["inputs"][0]))
    elif mutation == "unbound_executable":
        executable = binding["initial"]["executable"]["path"]
        binding["inputs"] = [row for row in binding["inputs"] if row["path"] != executable]
    elif mutation == "missing_arm":
        binding["initial"]["states"].pop("diverse")
    else:
        binding["initial"]["dataset"] = {"new_unbound_metadata": True}
    with pytest.raises(ValueError):
        study.verify_inputs(case.manifest)
