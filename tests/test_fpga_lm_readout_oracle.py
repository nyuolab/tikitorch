"""Synthetic frozen-feature readout contracts; never a real-data study."""
from __future__ import annotations

import importlib
import copy
import json
import math
from pathlib import Path
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def helper():
    return importlib.import_module("manuscript.tools.fpga_lm_readout_oracle")


@pytest.fixture(autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def features():
    generator = np.random.default_rng(19)
    return (generator.normal(size=(7, 3)).astype(np.float64),
            generator.normal(size=(7, 4)).astype(np.float64),
            np.array([0, 1, 2, 3, 1, 1, 0], dtype=np.int64))


def weights():
    generator = torch.Generator().manual_seed(4)
    shapes = {"EMBED": (4, 4), "MH": (2, 4, 3), "MP": (2, 4, 4),
              "MB": (2, 4), "W_HEAD": (4, 4)}
    return {name: torch.randn(shape, dtype=torch.float64, generator=generator) * .2
            for name, shape in shapes.items()}


@pytest.mark.parametrize("arm", ["bias", "affine"])
@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_true_mean_ce_full_l2_and_gradient_match_independent_autograd_and_finite_difference(helper, arm, chunk):
    h, z0, y = features()
    size = helper.parameter_count(3, 4, arm)
    theta = np.linspace(-.17, .23, size, dtype=np.float64)
    before = [value.tobytes() for value in (theta, h, z0, y)]
    result = helper.objective(theta, h, z0, y, arm, chunk_size=chunk)
    audited = helper.independent_objective(theta, h, z0, y, arm, chunk_size=chunk)
    for key in ("objective", "ce", "penalty", "gradient_norm_sq", "numerical_gap_estimate", "accuracy"):
        assert result[key] == pytest.approx(audited[key], abs=1e-12, rel=1e-12)
    np.testing.assert_allclose(result["gradient"], audited["gradient"], atol=1e-12, rtol=1e-12)
    for index in range(size):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical = (helper.objective(plus, h, z0, y, arm)["objective"]
                     - helper.objective(minus, h, z0, y, arm)["objective"]) / 2e-6
        assert result["gradient"][index] == pytest.approx(numerical, abs=5e-10, rel=1e-7)
    assert result["penalty"] == pytest.approx(.0005 * float(theta @ theta), abs=1e-15)
    assert result["numerical_gap_estimate"] == pytest.approx(float(result["gradient"] @ result["gradient"]) / .002)
    assert result["formal_certification"] is False and result["positions"] == 7
    assert [value.tobytes() for value in (theta, h, z0, y)] == before
    assert not np.shares_memory(result["gradient"], theta)


def test_zero_affine_is_exact_initial_bias_nested_and_no_feature_normalization(helper):
    h, z0, y = features()
    bias = np.array([.1, -.2, .3, -.4], dtype=np.float64)
    affine = np.concatenate((np.zeros(12, dtype=np.float64), bias))
    b = helper.objective(bias, h, z0, y, "bias")
    a = helper.objective(affine, h, z0, y, "affine")
    assert b["ce"] == a["ce"] and b["penalty"] == a["penalty"]
    np.testing.assert_array_equal(a["gradient"][-4:], b["gradient"])
    initial = helper.objective(np.zeros(16, dtype=np.float64), h, z0, y, "affine")
    logits = torch.tensor(z0, dtype=torch.float64)
    assert initial["ce"] == pytest.approx(float(torch.nn.functional.cross_entropy(logits, torch.tensor(y))), abs=1e-14)
    scaled = helper.objective(np.zeros(16, dtype=np.float64), h * 8, z0, y, "affine")
    np.testing.assert_allclose(scaled["gradient"][:12], initial["gradient"][:12] * 8, atol=1e-14)
    np.testing.assert_array_equal(scaled["gradient"][-4:], initial["gradient"][-4:])
    assert scaled["ce"] == initial["ce"]


def test_all_coordinates_including_common_bias_are_regularized(helper):
    h, z0, y = features()
    theta = np.array([.1, -.2, .3, -.4], dtype=np.float64)
    common = .75
    first = helper.objective(theta, h, z0, y, "bias")
    second = helper.objective(theta + common, h, z0, y, "bias")
    assert first["ce"] == pytest.approx(second["ce"], abs=1e-14)
    np.testing.assert_allclose(second["gradient"] - first["gradient"], np.full(4, .001 * common), atol=1e-14)
    assert second["objective"] > first["objective"]


@pytest.mark.parametrize("independent", [False, True])
def test_full_position_mean_is_not_mean_of_unequal_chunks(helper, independent):
    h = np.zeros((3, 2), dtype=np.float64)
    z0 = np.zeros((3, 2), dtype=np.float64)
    targets = np.array([0, 0, 1], dtype=np.int64)
    call = helper.independent_objective if independent else helper.objective
    result = call(np.zeros(2, dtype=np.float64), h, z0, targets, "bias", chunk_size=2)
    assert result["ce"] == pytest.approx(math.log(2), abs=1e-15)
    np.testing.assert_allclose(result["gradient"], [-1 / 6, 1 / 6], atol=1e-15)


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_actual_features_logits_and_selectors_match_fresh_full_matrix_forward(helper, context, ties):
    w = weights()
    if ties:
        w = {key: torch.zeros_like(value) for key, value in w.items()}
    before = {key: value.numpy().tobytes() for key, value in w.items()}
    tokens = torch.tensor([0, 1, 3, 2, 0], dtype=torch.int64)
    actual = helper.extract_features(w, tokens, context=context)
    independent = helper.extract_features(w, tokens, context=context, independent=True)
    assert actual["h"].shape == (5, 4) and actual["logits"].shape == (5, 4)
    for name in ("h", "logits"):
        torch.testing.assert_close(actual[name], independent[name], atol=1e-12, rtol=1e-12)
        assert not actual[name].requires_grad and actual[name].dtype == torch.float64
    for name in helper.helper.TRACE_KEYS:
        if name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX"):
            assert torch.equal(actual["trace"][name], independent["trace"][name])
            if ties:
                assert not actual["trace"][name].any()
        else:
            torch.testing.assert_close(actual["trace"][name], independent["trace"][name], atol=1e-12, rtol=1e-12)
    assert actual["guard_counts"].shape == (1, 6) and actual["guard_counts"].dtype == torch.int64
    assert independent["guard_counts"] is None and actual["features_differentiated"] is False
    assert {key: value.numpy().tobytes() for key, value in w.items()} == before
    actual["h"].zero_()
    assert {key: value.numpy().tobytes() for key, value in w.items()} == before


def test_feature_extraction_rejects_wrong_checkpoint_context_or_saved_independent_tape(helper):
    w = weights()
    tokens = torch.tensor([0, 1, 3, 2], dtype=torch.int64)
    base = helper.guarded.forward(w, tokens)
    changed = {key: value.clone() for key, value in w.items()}
    changed["MH"].reshape(-1)[0] += .1
    with pytest.raises(ValueError, match="parameter"):
        helper.extract_features(changed, tokens, base=base)
    with pytest.raises(ValueError, match="context"):
        helper.extract_features(w, tokens, base=base, context=False)
    with pytest.raises(ValueError, match="saved base"):
        helper.extract_features(w, tokens, base=base, independent=True)


@pytest.mark.parametrize("case", ["dtype", "nan", "targets_dtype", "targets_range", "shape", "empty", "theta_shape", "theta_dtype", "chunk_bool", "chunk_zero", "arm"])
def test_objective_rejects_invalid_or_broadcasting_inputs(helper, case):
    h, z0, y = features()
    theta, arm, chunk = np.zeros(4, dtype=np.float64), "bias", 2
    if case == "dtype": h = h.astype(np.float32)
    elif case == "nan": z0[0, 0] = np.nan
    elif case == "targets_dtype": y = y.astype(np.float64)
    elif case == "targets_range": y[0] = 4
    elif case == "shape": z0 = z0[:1]
    elif case == "empty": h, z0, y = h[:0], z0[:0], y[:0]
    elif case == "theta_shape": theta = theta.reshape(1, 4)
    elif case == "theta_dtype": theta = theta.astype(np.float32)
    elif case == "chunk_bool": chunk = True
    elif case == "chunk_zero": chunk = 0
    elif case == "arm": arm = "other"
    for call in (helper.objective, helper.independent_objective):
        with pytest.raises(ValueError):
            call(theta, h, z0, y, arm, chunk_size=chunk)


@pytest.mark.parametrize("independent", [False, True])
def test_objective_checkpoint_propagates_deadline_without_mutating_inputs(helper, independent):
    h, z0, y = features()
    theta = np.zeros(4, dtype=np.float64)
    before = [value.tobytes() for value in (theta, h, z0, y)]
    calls = []

    def deadline():
        calls.append(True)
        raise TimeoutError("synthetic fixed deadline")

    call = helper.independent_objective if independent else helper.objective
    with pytest.raises(TimeoutError):
        call(theta, h, z0, y, "bias", checkpoint=deadline)
    assert len(calls) == 1 and [value.tobytes() for value in (theta, h, z0, y)] == before


@pytest.mark.parametrize("arm", ["bias", "affine"])
def test_real_tiny_convex_solver_returns_counted_endpoints_not_a_formal_certificate(helper, arm):
    pytest.importorskip("scipy.optimize")
    h, z0, y = features()
    before = [value.tobytes() for value in (h, z0, y)]
    fit = helper.solve(h, z0, y, arm, chunk_size=3)
    result = fit["summary"]
    assert result["zero_start"] is True and result["restarts"] == 0
    assert result["formal_certification"] is False and result["rounding_error_bound"] is None
    assert result["function_calls"] == result["complete_function_evaluations"] == len(fit["history"])
    assert 1 <= result["function_calls"] <= 400 and result["iterations"] <= 200
    assert result["ce"] < fit["initial"]["statistics"]["ce"]
    assert result["numerical_target_met"] is True and result["numerically_converged"] is True
    assert result["numerical_gap_estimate"] <= 1e-5
    assert not fit["initial"]["theta"].any()
    checked = helper.independent_objective(fit["theta"], h, z0, y, arm, chunk_size=2)
    np.testing.assert_allclose(fit["gradient"], checked["gradient"], atol=1e-12, rtol=1e-11)
    assert result["objective"] == pytest.approx(checked["objective"], abs=1e-12)
    assert [value.tobytes() for value in (h, z0, y)] == before


def test_solver_function_cap_returns_last_accepted_not_unevaluated_or_trial_point(helper, monkeypatch):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    h, z0, y = features()
    calls = []
    original = helper.objective

    def objective(*args, **kwargs):
        calls.append(args[0].copy())
        return original(*args, **kwargs)

    def minimize(function, initial, *, method, jac, bounds, callback, options):
        assert method == "L-BFGS-B" and jac is True and bounds is None
        function(initial)
        function(initial + .01)  # Evaluated line-search trial, never accepted.
        function(initial + .02)  # The strict cap must stop before evaluation.
        pytest.fail("function cap did not stop the third call")

    monkeypatch.setattr(helper, "MAX_FUNCTION_CALLS", 2)
    monkeypatch.setattr(helper, "objective", objective)
    monkeypatch.setattr(scipy_optimize, "minimize", minimize)
    fit = helper.solve(h, z0, y, "bias")
    assert len(calls) == 2 and len(fit["history"]) == 2
    assert fit["summary"]["function_calls"] == 2 and fit["summary"]["iterations"] == 0
    assert fit["summary"]["stopping_reason"] == "function_call_limit"
    assert fit["summary"]["inconclusive"] is True and fit["summary"]["numerically_converged"] is False
    np.testing.assert_array_equal(fit["theta"], np.zeros(4))
    np.testing.assert_array_equal(fit["gradient"], fit["initial"]["gradient"])


@pytest.mark.parametrize("reason", ["iteration_limit", "scipy_stopped_above_numerical_target", "deadline"])
def test_bounded_solver_stop_is_evaluable_inconclusive_and_does_not_retry(helper, monkeypatch, reason):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    h, z0, y = features()
    start = 1000.
    elapsed = [0.]
    original = helper.objective
    calls, solver_calls = [], []

    def objective(*args, **kwargs):
        calls.append(args[0].copy())
        return original(*args, **kwargs)

    def minimize(function, initial, **kwargs):
        solver_calls.append(True)
        function(initial)
        if reason == "deadline":
            elapsed[0] = 3600.
            function(initial + .01)
            pytest.fail("deadline ignored")
        if reason == "iteration_limit":
            point = initial + .01
            function(point)
            kwargs["callback"](point)
            pytest.fail("iteration cap ignored")
        return SimpleNamespace(status=2, message="synthetic finite stop", success=False, x=initial.copy())

    monkeypatch.setattr(helper, "MAX_ITERATIONS", 1)
    monkeypatch.setattr(helper.time, "time", lambda: start + elapsed[0])
    monkeypatch.setattr(helper.time, "monotonic", lambda: start + elapsed[0])
    monkeypatch.setattr(helper, "objective", objective)
    monkeypatch.setattr(scipy_optimize, "minimize", minimize)
    fit = helper.solve(h, z0, y, "bias")
    assert solver_calls == [True]
    assert fit["summary"]["stopping_reason"] == reason
    assert fit["summary"]["inconclusive"] is True and fit["summary"]["formal_certification"] is False
    expected = calls[-1] if reason == "iteration_limit" else calls[0]
    np.testing.assert_array_equal(fit["theta"], expected)
    assert fit["summary"]["function_calls"] == len(calls)


@pytest.mark.parametrize("case", ["wrong_zero_start", "unevaluated_accepted", "unevaluated_return"])
def test_solver_cannot_return_unbound_or_nonzero_start_points(helper, monkeypatch, case):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    h, z0, y = features()

    def minimize(function, initial, **kwargs):
        if case == "wrong_zero_start":
            function(initial + .01)
        else:
            function(initial)
            if case == "unevaluated_accepted":
                kwargs["callback"](initial + .01)
            else:
                return SimpleNamespace(status=0, message="forged success", success=True, x=initial + .01)

    monkeypatch.setattr(scipy_optimize, "minimize", minimize)
    with pytest.raises(RuntimeError):
        helper.solve(h, z0, y, "bias")


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_readout_oracle")


def test_source_closure_and_fixed_budget_do_not_relabel_existing_studies(study):
    before, now = study.previous.source_identity(), study.source_identity()
    assert len(before) == 79 and len(now) == 82
    assert {key: now[key] for key in before} == before
    assert set(now) - set(before) == {"manuscript/tools/fpga_lm_readout_oracle.py",
        "manuscript/tools/diagnose_fpga_lm_readout_oracle.py", "tests/test_fpga_lm_readout_oracle.py"}
    recipe = study.recipe()
    assert recipe["arms"] == ["bias", "affine"]
    assert recipe["fit_windows"] == list(range(1024))
    assert recipe["sampled_training_windows"] == list(range(0, 1024, 16))
    assert recipe["development_windows"] == list(range(88, 152))
    assert recipe["lambda"] == .001 and recipe["numerical_gap_estimate_target"] == 1e-5
    assert recipe["max_iterations_per_arm"] == 200 and recipe["max_objective_calls_per_arm"] == 400
    assert recipe["solver_seconds_per_arm"] == 3600 and recipe["cache_windows"] == 1088
    assert recipe["backbone_gradient_queries"] == recipe["backbone_optimizer_steps"] == 0
    assert recipe["optimizer_trajectory_replayed"] is False and recipe["recipe_promoted"] is False
    assert len(study.expected_artifacts()) == 1097


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Actual tiny feature/fit/endpoint audit, with synthetic admission only."""
    pytest.importorskip("scipy.optimize")
    pytest.importorskip("threadpoolctl")
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_readout_oracle")
    real_bind, real_model = study.bind_inputs, study.frozen_model
    patch = pytest.MonkeyPatch()
    temporary = tmp_path_factory.mktemp("readout_oracle")
    source, plan, corpus = [temporary / name for name in ("source.txt", "PLAN.md", "synthetic.txt")]
    source.write_text("Synthetic readout source closure.\n")
    plan.write_text("Synthetic readout protocol, not scientific evidence.\n")
    corpus.write_bytes(b"opaque synthetic corpus stand-in; no real corpus reads")
    prior_root = temporary / "prior"
    prior_root.mkdir()
    for name, value in {"TRAIN_WINDOWS": (0, 1, 2, 3), "SCORE_TRAIN_WINDOWS": (0, 2),
                        "DEV_WINDOWS": (88, 89), "SEQUENCE": 4, "PLAN": plan}.items():
        patch.setattr(study, name, value)
    parameters = weights()
    checkpoint = prior_root / "state_000256.npz"
    study.parent.parent.save(checkpoint, parameters)
    metadata = checkpoint.with_suffix(".json")
    metadata.write_text('{"synthetic_checkpoint": true}\n')
    binding = {"core": {"prior_run": str(prior_root), "prior_record": str(source), "prior_audit": str(source),
                         "dataset": {"path": str(corpus)}, "core_binding": {}},
               "checkpoint": study.old.file_identity(checkpoint), "checkpoint_metadata": study.old.file_identity(metadata)}
    controls, reads, calls = {}, [], {"producer_features": 0, "independent_features": 0, "solve": 0, "independent_objective": 0}

    def batch_data(raw, dataset, indices, partition="train"):
        assert raw == corpus.read_bytes() and dataset == binding["core"]["dataset"]
        assert len(indices) == 1
        index = indices[0]
        assert (partition == "train" and index in (0, 1, 2, 3)) or (partition == "validation" and index in (88, 89))
        reads.append((partition, index))
        tokens = torch.tensor([(index + i) % 4 for i in range(4)], dtype=torch.int64)
        targets = torch.tensor([(2 * index + i + (i % 2)) % 4 for i in range(4)], dtype=torch.int64)
        return tokens[None, :], targets[None, :]

    for cohort, indices in (("train", (0, 2)), ("development", (88, 89))):
        rows = []
        for index in indices:
            tokens, targets = batch_data(corpus.read_bytes(), binding["core"]["dataset"], (index,),
                                         "train" if cohort == "train" else "validation")
            original = study.oracle.extract_features(parameters, tokens[0])
            path = study.baseline_path(binding, cohort, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            study.parent.parent.save(path, original["trace"])
            rows.append({"window": index, **study.parent.parent.metrics(original["logits"], targets[0])})
        controls[cohort] = {"per_window": rows, "mean": study.parent._mean(rows)}
    protected = [study.old.file_identity(path) for path in (source, corpus, checkpoint, metadata)]
    protected.extend(study.old.file_identity(path) for path in prior_root.rglob("*.npz"))
    identity = lambda: {"synthetic_source": study.old.digest(source.read_bytes())}
    patch.setattr(study, "source_identity", identity)
    token = study.Validation(binding, identity(), object())

    def bind(config):
        return copy.deepcopy(binding), corpus.read_bytes(), token

    def frozen(actual, proof):
        study.equal(actual, proof.binding, "immutable synthetic readout binding")
        study.equal(proof.binding, binding, "original synthetic admission")
        study.equal(proof.sources, identity(), "current source bytes")
        for descriptor in protected:
            study.equal(study.old.file_identity(descriptor["path"]), descriptor, "immutable input/control/checkpoint")

    def model(actual):
        assert actual == binding
        return {"kind": "tropical", "parameters": study.load(checkpoint), "model": None, "prior_logits": None}

    patch.setattr(study.parent, "batch_data", batch_data)
    patch.setattr(study, "bind_inputs", bind)
    patch.setattr(study, "frozen", frozen)
    patch.setattr(study, "frozen_model", model)
    patch.setattr(study.channel, "controls", lambda actual: {key: copy.deepcopy(controls) for key in ("mixed", "prior_gru", "frozen_core")})
    original_extract, original_solve, original_independent = study.oracle.extract_features, study.oracle.solve, study.oracle.independent_objective

    def extract(*args, **kwargs):
        calls["independent_features" if kwargs.get("independent", False) else "producer_features"] += 1
        return original_extract(*args, **kwargs)

    def solve(*args, **kwargs):
        calls["solve"] += 1
        return original_solve(*args, **kwargs)

    def independent(*args, **kwargs):
        calls["independent_objective"] += 1
        return original_independent(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("backbone gradient/optimizer or old numerical trajectory is forbidden")

    patch.setattr(study.oracle, "extract_features", extract)
    patch.setattr(study.oracle, "solve", solve)
    patch.setattr(study.oracle, "independent_objective", independent)
    patch.setattr(study.helper, "gradient", forbidden)
    patch.setattr(study.helper, "make_optimizer", forbidden)
    patch.setattr(torch.optim.Adam, "step", forbidden)
    for module in (study.previous, study.channel, study.core):
        for name in ("run", "audit", "execute", "check"):
            if hasattr(module, name):
                patch.setattr(module, name, forbidden)
    config = SimpleNamespace(prior_run=prior_root, prior_record=source, prior_audit=source, corpus=corpus,
                             output=temporary / "producer", record=temporary / "public.json")
    reads.clear()
    try:
        result = study.run(config)
        yield SimpleNamespace(study=study, temporary=temporary, source=source, plan=plan, corpus=corpus, binding=binding,
            protected=protected, config=config, root=config.output, record=config.record, result=result, calls=calls,
            reads=reads, producer_calls=dict(calls), token=token, real_bind=real_bind, real_model=real_model)
    finally:
        patch.undo()


def test_actual_shared_cache_both_fits_and_all_fixed_cohort_readouts(synthetic):
    value, study = synthetic, synthetic.study
    result = value.result
    assert result["complete"] is True and set(result["arms"]) == {"bias", "affine"}
    assert value.producer_calls == {"producer_features": 6, "independent_features": 0, "solve": 2, "independent_objective": 0}
    assert result["cache_windows"] == 6 and result["cache"]["initial_function_parity_checks"] == 4
    assert result["backbone_gradient_queries"] == result["backbone_optimizer_steps"] == 0
    assert result["distinct_readout_training_pairs"] == 16 and result["new_distinct_training_pairs"] == 0
    assert result["formal_optimization_certificate"] is False
    assert set(value.reads) == {("train", i) for i in range(4)} | {("validation", i) for i in (88, 89)}
    assert len(study.inventory(value.root)) == len(study.expected_artifacts()) == 15
    for arm, outcome in result["arms"].items():
        assert outcome["solver_complete_labeled_position_evaluations"] == outcome["solver"]["summary"]["complete_function_evaluations"] * 16
        assert outcome["solver_attempted_labeled_position_evaluations_upper_bound"] == outcome["solver"]["summary"]["function_calls"] * 16
        assert outcome["endpoints"]["initial"]["positions"] == 16
        assert outcome["endpoints"]["final"]["positions"] == 16
        assert [row["window"] for row in outcome["after"]["train"]["per_window"]] == [0, 2]
        assert [row["window"] for row in outcome["after"]["development"]["per_window"]] == [88, 89]
        initial = study.load(value.root / arm / "initial.npz")
        assert not initial["theta"].any()
    assert result["arms"]["bias"]["initial_scores"] == result["arms"]["affine"]["initial_scores"]


def test_real_independent_endpoint_audit_checks_all_features_and_gradients_without_refitting(synthetic, tmp_path):
    value, study = synthetic, synthetic.study
    original, before = tree_bytes(value.root), dict(value.calls)
    config = SimpleNamespace(**vars(value.config), audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    result = study.audit(config)
    assert result["complete"] is True and result["independent_recompute_completed"] is True
    assert result["independent_dense_feature_windows"] == result["supplemental_guarded_feature_windows"] == 6
    assert result["independent_readout_full_gradients"] == 4 and result["initial_function_parity_checks"] == 4
    assert result["optimizer_trajectory_replayed"] is False and result["new_training_updates"] == 0
    assert {k: value.calls[k] - before[k] for k in before} == {
        "producer_features": 6, "independent_features": 6, "solve": 0, "independent_objective": 4}
    assert tree_bytes(value.root) == original
    assert {path.name for path in config.audit_output.iterdir()} == {"audit_manifest.json", "declaration.md", "verification.json"}
    assert (config.audit_output / "verification.json").read_bytes() == config.audit_record.read_bytes()


def test_invocation_proof_does_not_alias_mutable_working_binding_or_sources(study):
    binding, sources = {"nested": ["original"]}, {"file": "original_hash"}
    proof = study.Validation(binding, sources, object())
    binding["nested"][0] = "changed"
    sources["file"] = "changed_hash"
    assert proof.binding == {"nested": ["original"]} and proof.sources == {"file": "original_hash"}


@pytest.mark.parametrize("case", ["h", "logits", "route", "guard", "gradient", "theta", "initial_signed_zero"])
def test_independent_reconstruction_rejects_cache_or_endpoint_corruption(synthetic, case):
    value, study = synthetic, synthetic.study
    path = (study.cache_path(value.root, "train", 1) if case in ("h", "logits", "route", "guard")
            else value.root / "affine" / ("initial.npz" if case == "initial_signed_zero" else "final.npz"))
    original, arrays = path.read_bytes(), study.load(path)
    if case == "route":
        arrays["PROJ_IDX"].reshape(-1)[0] = (arrays["PROJ_IDX"].reshape(-1)[0] + 1) % 4
    elif case == "guard":
        arrays["guard_counts"][0] = torch.tensor([1, 0, 8, 0, 4, 0])
    elif case == "initial_signed_zero":
        arrays["theta"][0] = -0.
    else:
        arrays[case].reshape(-1)[0] += .125
    path.unlink()
    study.parent.parent.save(path, arrays)
    try:
        with pytest.raises(ValueError):
            study.execute(value.root, value.binding, value.corpus.read_bytes(), independent=True,
                          clock=SimpleNamespace(check=lambda: None))
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("case", ["source", "checkpoint", "initial_parity"])
def test_immutable_sources_checkpoint_and_original_traces_fail_before_new_forward(synthetic, tmp_path, case):
    value, study = synthetic, synthetic.study
    path = {"source": value.source, "checkpoint": Path(value.binding["checkpoint"]["path"]),
            "initial_parity": study.baseline_path(value.binding, "train", 0)}[case]
    original, before = path.read_bytes(), dict(value.calls)
    path.write_bytes(original + b"\n")
    config = SimpleNamespace(**vars(value.config), audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    try:
        with pytest.raises(ValueError):
            study.audit(config)
    finally:
        path.write_bytes(original)
    assert value.calls == before and not config.audit_record.exists()
    assert study.read(config.audit_output / "failure.json")["complete"] is False


@pytest.mark.parametrize("case", ["calls", "iterations", "option", "endpoint_hash", "objective", "gap", "zero_start", "accepted_reference"])
def test_solver_scalar_provenance_must_match_actual_endpoints_and_fixed_caps(synthetic, case):
    value, study = synthetic, synthetic.study
    path = value.root / "affine" / "solver.json"
    original = path.read_bytes()
    solver = study.read(path)
    if case == "calls": solver["summary"]["function_calls"] = 401
    elif case == "iterations": solver["summary"]["iterations"] = 201
    elif case == "option": solver["summary"]["options"]["ftol"] = .5
    elif case == "endpoint_hash": solver["iteration_history"][-1]["theta_sha256"] = "a" * 64
    elif case == "objective": solver["summary"]["objective"] += .1
    elif case == "gap": solver["summary"]["numerical_gap_estimate"] = 0.
    elif case == "zero_start": solver["summary"]["zero_start"] = 1
    elif case == "accepted_reference": solver["iteration_history"][-1]["function_call"] = 0
    path.write_text(json.dumps(solver, sort_keys=True, indent=2, allow_nan=False))
    try:
        with pytest.raises(ValueError):
            study.execute(value.root, value.binding, value.corpus.read_bytes(), independent=False,
                          clock=SimpleNamespace(check=lambda: None))
    finally:
        path.write_bytes(original)


def invocation(value, tmp_path, mode):
    if mode == "producer":
        config = SimpleNamespace(**{**vars(value.config), "output": tmp_path / "run", "record": tmp_path / "public.json"})
        return config, value.study.run, config.output, config.record
    config = SimpleNamespace(**vars(value.config), audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    return config, value.study.audit, config.audit_output, config.audit_record


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt, TimeoutError])
def test_failure_and_deadline_do_not_publish_completion_or_leave_alarm(synthetic, tmp_path, monkeypatch, mode, error):
    value, study = synthetic, synthetic.study
    config, call, output, record = invocation(value, tmp_path, mode)

    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 1 and signal.getitimer(signal.ITIMER_REAL)[0] > 0
        raise error("synthetic readout interruption")

    monkeypatch.setattr(study, "execute", fail)
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(error):
        call(config)
    assert study.read(output / "failure.json")["complete"] is False and not record.exists()
    assert signal.getsignal(signal.SIGALRM) == handler and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("replacement", ["owned", "foreign"])
def test_publication_failure_retracts_only_owned_receipt_and_preserves_bytes(synthetic, tmp_path, monkeypatch, mode, replacement):
    value, study = synthetic, synthetic.study
    config, call, output, record = invocation(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "execute", lambda *args, **kwargs: study.read(value.root / "summary.json"))
    original, changed = study.frozen, []

    def frozen(*args):
        original(*args)
        if record.exists():
            payload = record.read_bytes()
            if replacement == "foreign":
                record.unlink()
            else:
                payload += b"\n"
            record.write_bytes(payload)
            changed.append(payload)
            raise ValueError("synthetic publication postcheck failure")

    monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError, match="postcheck"):
        call(config)
    assert study.read(output / "failure.json")["complete"] is False
    if replacement == "owned":
        assert not record.exists() and (output / "rejected_public.json").read_bytes() == changed[0]
    else:
        assert record.read_bytes() == changed[0] and not (output / "rejected_public.json").exists()


@pytest.mark.parametrize("mode", ["producer", "audit"])
def test_existing_output_is_preserved_and_rejected_before_numerical_work(synthetic, tmp_path, mode):
    value = synthetic
    config, call, output, record = invocation(value, tmp_path, mode)
    output.mkdir()
    (output / "user.txt").write_bytes(b"user-owned existing output")
    before = dict(value.calls)
    with pytest.raises((ValueError, FileExistsError)):
        call(config)
    assert tree_bytes(output) == {"user.txt": b"user-owned existing output"}
    assert value.calls == before and not record.exists()


def test_source_drift_during_binding_cannot_be_adopted_into_new_token(synthetic, tmp_path, monkeypatch):
    value, study = synthetic, synthetic.study
    config, call, output, record = invocation(value, tmp_path, "producer")
    original_bind, original_source, before = study.bind_inputs, value.source.read_bytes(), dict(value.calls)

    def bind(config):
        binding, raw, token = original_bind(config)
        value.source.write_bytes(original_source + b"changed during binding")
        return binding, raw, study.Validation(binding, study.source_identity(), token.channel_validation)

    monkeypatch.setattr(study, "bind_inputs", bind)
    try:
        with pytest.raises(ValueError, match="source"):
            call(config)
    finally:
        value.source.write_bytes(original_source)
    assert value.calls == before and not record.exists()
    assert study.read(output / "failure.json")["complete"] is False


@pytest.mark.parametrize("mode", ["producer", "audit"])
def test_exact_seal_or_private_receipt_cannot_be_rewritten_during_publication(synthetic, tmp_path, monkeypatch, mode):
    value, study = synthetic, synthetic.study
    config, call, output, record = invocation(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "execute", lambda *args, **kwargs: study.read(value.root / "summary.json"))
    original = study.frozen

    def frozen(*args):
        original(*args)
        if record.exists():
            path = output / ("artifact_manifest.json" if mode == "producer" else "verification.json")
            before = path.read_bytes()
            changed = before.replace(b"\n  ", b"\n\t ", 1)
            assert before != changed and len(before) == len(changed)
            path.write_bytes(changed)

    monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError):
        call(config)
    assert not record.exists() and study.read(output / "failure.json")["complete"] is False


def test_finite_solver_stop_remains_evaluable_inconclusive_not_failed_or_retried(synthetic, tmp_path, monkeypatch):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    value = synthetic
    config, call, output, record = invocation(value, tmp_path, "producer")
    calls = []

    def minimize(function, initial, **kwargs):
        calls.append(initial.copy())
        function(initial)
        return SimpleNamespace(status=2, message="synthetic finite solver stop", success=False, x=initial.copy())

    monkeypatch.setattr(scipy_optimize, "minimize", minimize)
    result = call(config)
    assert len(calls) == 2 and result["complete"] is True and record.is_file()
    assert not (output / "failure.json").exists()
    for outcome in result["arms"].values():
        summary = outcome["solver"]["summary"]
        assert summary["inconclusive"] is True and summary["numerically_converged"] is False
        assert summary["stopping_reason"] == "scipy_stopped_above_numerical_target"
        assert summary["function_calls"] == 1 and summary["iterations"] == 0 and summary["restarts"] == 0


def test_audit_rejects_changed_frozen_plan_before_recomputing_any_features(synthetic, tmp_path):
    value, study = synthetic, synthetic.study
    original, before = value.plan.read_bytes(), dict(value.calls)
    value.plan.write_bytes(original + b"new unadmitted protocol\n")
    config, call, output, record = invocation(value, tmp_path, "audit")
    try:
        with pytest.raises(ValueError, match="declaration"):
            call(config)
    finally:
        value.plan.write_bytes(original)
    assert value.calls == before and not record.exists()
    assert study.read(output / "failure.json")["complete"] is False


@pytest.mark.parametrize("field", ["schema", "run_directory", "record"])
def test_resealed_producer_manifest_identity_is_checked_before_model_access(synthetic, tmp_path, field):
    value, study = synthetic, synthetic.study
    path, seal = value.root / "manifest.json", value.root / "artifact_manifest.json"
    original, seal_before = path.read_bytes(), seal.read_bytes()
    data = study.read(path)
    data[field] = "unrelated-retargeted-value"
    path.write_text(json.dumps(data, sort_keys=True, indent=2))
    seal.write_text(json.dumps({"schema": study.SCHEMA, "files": study.prior.artifact_manifest(value.root)}))
    before = dict(value.calls)
    config, call, output, record = invocation(value, tmp_path, "audit")
    try:
        with pytest.raises(ValueError):
            call(config)
    finally:
        path.write_bytes(original)
        seal.write_bytes(seal_before)
    assert value.calls == before and not record.exists()
    assert study.read(output / "failure.json")["complete"] is False


def test_canonical_binding_selects_actual_core_step256_without_old_numerical_calls(synthetic, tmp_path, monkeypatch):
    study = synthetic.study
    directory = tmp_path / study.core.ARM
    directory.mkdir()
    path = directory / "state_000256.npz"
    path.write_bytes(b"synthetic old full checkpoint payload")
    path.with_suffix(".json").write_bytes(b"synthetic old checkpoint metadata")
    bound, token = {"prior_run": str(tmp_path), "core_binding": {"original": True}}, object()
    monkeypatch.setattr(study.channel, "bind_inputs", lambda config: (bound, b"synthetic raw bytes", token))
    for name in ("execute", "run", "audit", "check"):
        monkeypatch.setattr(study.channel, name, lambda *args, **kwargs: pytest.fail("old numerical calls forbidden"))
    binding, raw, proof = synthetic.real_bind(SimpleNamespace())
    assert raw == b"synthetic raw bytes" and proof.channel_validation is token
    assert binding["checkpoint"] == study.old.file_identity(path)
    assert binding["checkpoint_metadata"] == study.old.file_identity(path.with_suffix(".json"))
    assert binding["core"] == bound


def test_frozen_model_loads_full_audited_endpoint_not_initial_or_trained_replacement(synthetic, tmp_path, monkeypatch):
    study = synthetic.study
    initial, final = weights(), weights()
    final["EMBED"] += .125
    body = {"kind": "tropical", "parameters": initial, "model": None, "prior_logits": None}
    binding = {"core": {"core_binding": {"actual_old_inputs": True}}, "checkpoint": {"path": str(tmp_path / "state_000256.npz")}}
    calls = []

    def initial_model(bound):
        assert bound == binding["core"]["core_binding"]
        return body

    def checkpoint(path, model, arm, step):
        calls.append((path, arm, step))
        assert model is body
        return final, {"synthetic": True}

    monkeypatch.setattr(study.core, "initial_model", initial_model)
    monkeypatch.setattr(study.previous.endpoint, "_checkpoint", checkpoint)
    cell = synthetic.real_model(binding)
    assert cell["parameters"] is final
    assert calls == [(binding["checkpoint"]["path"], "mixed", 256)]
    assert not any(value.requires_grad for value in cell["parameters"].values())


@pytest.mark.parametrize("target", ["producer_output", "producer_public", "audit_output"])
@pytest.mark.parametrize("marker", ["artifact_manifest.json", "audit_manifest.json"])
def test_unrelated_sealed_tree_cannot_receive_new_output_or_publication(synthetic, tmp_path, target, marker):
    value = synthetic
    mode = "audit" if target == "audit_output" else "producer"
    config, call, output, record = invocation(value, tmp_path, mode)
    sealed = tmp_path / "unrelated_sealed_history"
    sealed.mkdir()
    (sealed / marker).write_bytes(b'{"synthetic_existing_seal": true}\n')
    (sealed / "user.txt").write_bytes(b"immutable unrelated payload")
    forbidden = sealed / "forbidden"
    if target == "producer_output":
        config.output = forbidden
    elif target == "producer_public":
        config.record = forbidden
    else:
        config.audit_output = forbidden
    before, calls = tree_bytes(sealed), dict(value.calls)
    with pytest.raises(ValueError):
        call(config)
    assert tree_bytes(sealed) == before and value.calls == calls
    assert not forbidden.exists() and not output.exists() and not record.exists()
