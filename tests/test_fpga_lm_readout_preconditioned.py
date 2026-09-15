"""Synthetic coordinate-only preconditioning; no real cached-data fitting."""
from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def helper():
    pytest.importorskip("scipy.linalg")
    return importlib.import_module("manuscript.tools.fpga_lm_readout_preconditioned")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def data():
    rng = np.random.default_rng(23)
    h = rng.normal(size=(9, 3)).astype(np.float64)
    h[:, 1] = 3 * h[:, 0] + h[:, 1] * .1
    z0 = rng.normal(size=(9, 4)).astype(np.float64)
    targets = np.array([0, 1, 1, 3, 2, 3, 1, 0, 2], dtype=np.int64)
    return h, z0, targets


def pack(matrix):
    return np.concatenate((matrix[:, :-1].ravel(), matrix[:, -1]))


def unpack(vector, vocab, width):
    return np.column_stack((vector[:vocab * width].reshape(vocab, width), vector[vocab * width:]))


@pytest.mark.parametrize("chunk_size", [1, 4, 100])
def test_preconditioner_uses_exact_training_augmented_gram_and_physical_ridge(helper, chunk_size):
    h, _, _ = data()
    before = h.tobytes()
    factor = helper.build_preconditioner(h, chunk_size=chunk_size)
    x = np.column_stack((h, np.ones(len(h), dtype=np.float64)))
    expected = .001 * np.eye(4) + x.T @ x / (2 * len(h))
    np.testing.assert_allclose(factor["M"], expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(factor["L"] @ factor["L"].T, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(factor["S"].T @ expected @ factor["S"], np.eye(4), atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(factor["L"].T @ factor["S"], np.eye(4), atol=1e-14, rtol=1e-14)
    assert h.tobytes() == before
    assert factor["M"][-1, -1] == pytest.approx(.501, abs=1e-15)


def test_rank_deficient_features_remain_invertible_without_changing_lambda(helper):
    h = np.zeros((5, 3), dtype=np.float64)
    factor = helper.build_preconditioner(h)
    np.testing.assert_array_equal(factor["M"], np.diag([.001, .001, .001, .501]))
    np.testing.assert_allclose(factor["S"], np.diag(1 / np.sqrt([.001, .001, .001, .501])), atol=1e-14)


def test_nonsymmetric_transform_orientation_packing_and_pullback_are_unambiguous(helper):
    s = np.array([[2., -.25, .75], [0., .5, -.125], [0., 0., 1.5]], dtype=np.float64)
    u = np.array([[.2, -.3, .4], [1.1, -.7, -.9]], dtype=np.float64)
    physical_gradient = np.array([.3, -.4, .5, .6, -.2, .7], dtype=np.float64)
    before = [value.tobytes() for value in (s, u, physical_gradient)]
    theta = helper.materialize(u, s)
    np.testing.assert_array_equal(theta, pack(u @ s.T))
    coordinate_gradient = helper.pullback(physical_gradient, s)
    np.testing.assert_array_equal(coordinate_gradient, unpack(physical_gradient, 2, 2) @ s)
    x = np.array([[.2, -.7, 1.], [.8, .1, 1.]], dtype=np.float64)
    np.testing.assert_allclose(x @ unpack(theta, 2, 2).T, (x @ s) @ u.T, atol=1e-14)
    for row in range(2):
        for col in range(3):
            plus, minus = u.copy(), u.copy()
            plus[row, col] += 1e-6
            minus[row, col] -= 1e-6
            finite_difference = (helper.materialize(plus, s) @ physical_gradient - helper.materialize(minus, s) @ physical_gradient) / 2e-6
            assert coordinate_gradient[row, col] == pytest.approx(finite_difference, abs=2e-10)
    assert [value.tobytes() for value in (s, u, physical_gradient)] == before
    assert not np.shares_memory(theta, u) and not np.shares_memory(coordinate_gradient, physical_gradient)


@pytest.mark.parametrize("independent", [False, True])
def test_coordinate_objective_preserves_physical_ridge_and_full_gradient(helper, independent):
    old = importlib.import_module("manuscript.tools.fpga_lm_readout_oracle")
    h, z0, targets = data()
    factor = helper.build_preconditioner(h)
    u = np.linspace(-.02, .03, 16, dtype=np.float64).reshape(4, 4)
    theta = pack(u @ factor["S"].T)
    reference = old.independent_objective(theta, h, z0, targets, "affine", chunk_size=4)
    result = helper.objective(u, h, z0, targets, factor, independent=independent, chunk_size=4)
    np.testing.assert_array_equal(result["theta"], theta)
    np.testing.assert_allclose(result["physical_gradient"], reference["gradient"], atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(result["coordinate_gradient"], unpack(reference["gradient"], 4, 3) @ factor["S"], atol=1e-12, rtol=1e-11)
    for key in ("objective", "ce", "penalty", "numerical_gap_estimate"):
        assert result[key] == pytest.approx(reference[key], abs=1e-12, rel=1e-11)
    assert result["penalty"] == pytest.approx(.0005 * float(theta @ theta), abs=1e-14)
    assert abs(result["penalty"] - .0005 * float(np.sum(u * u))) > 1e-5
    for row in range(4):
        for col in range(4):
            plus, minus = u.copy(), u.copy()
            plus[row, col] += 1e-6
            minus[row, col] -= 1e-6
            finite_difference = (helper.objective(plus, h, z0, targets, factor)["objective"]
                                 - helper.objective(minus, h, z0, targets, factor)["objective"]) / 2e-6
            assert result["coordinate_gradient"][row, col] == pytest.approx(finite_difference, abs=1e-8, rel=1e-6)


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_numerical_stop_estimate_is_physical_not_coordinate_gradient_norm(helper, scale):
    h, z0, targets = data()
    factor = helper.build_preconditioner(h)
    factor = {**factor, "S": np.eye(4, dtype=np.float64) * scale}
    # Objective coordinates can be tested with an invertible explicit S;
    # admission separately binds the one prescribed training-derived factor.
    result = helper.objective(np.zeros((4, 4), dtype=np.float64), h, z0, targets, factor)
    physical = float(result["physical_gradient"] @ result["physical_gradient"]) / .002
    transformed = float(np.sum(result["coordinate_gradient"] ** 2)) / .002
    assert result["numerical_gap_estimate"] == pytest.approx(physical, rel=1e-13)
    assert result["numerical_target_met"] is (physical <= 1e-5)
    assert transformed / physical == pytest.approx(scale ** 2, rel=1e-12)
    if scale < 1:
        assert transformed < 1e-5 < physical


@pytest.mark.parametrize("independent", [False, True])
def test_saved_factor_is_canonical_and_materializes_identical_roundtrip_bytes(helper, independent):
    h, _, _ = data()
    factor = helper.build_preconditioner(h)
    assert all(factor[key].flags.c_contiguous for key in ("M", "L", "S"))
    assert helper.verify_preconditioner(h, factor, independent=independent) is True
    stream = io.BytesIO()
    np.savez(stream, **{key: factor[key] for key in ("M", "L", "S")})
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    saved["summary"] = json.loads(json.dumps(factor["summary"]))
    assert helper.verify_preconditioner(h, saved, independent=independent) is True
    u = np.arange(16, dtype=np.float64).reshape(4, 4) / 31
    assert helper.materialize(u, factor["S"]).tobytes() == helper.materialize(u, saved["S"]).tobytes()
    assert helper.transform_digest(factor) == helper.transform_digest(saved)


@pytest.mark.parametrize("case", ["M", "L", "S", "fortran", "condition", "construction", "orientation", "feature_hash", "bool_alias"])
def test_factor_or_complete_construction_receipt_cannot_be_relabelled(helper, case):
    h, _, _ = data()
    factor = helper.build_preconditioner(h)
    if case in ("M", "L", "S"):
        factor[case][0, 0] += .01
    elif case == "fortran":
        factor["S"] = np.asfortranarray(factor["S"])
    else:
        key = {"condition": "estimated_condition_number", "feature_hash": "training_feature_sha256",
               "bool_alias": "development_used"}.get(case, case)
        factor["summary"][key] = 0 if case == "bool_alias" else "forged"
    with pytest.raises(ValueError):
        helper.verify_preconditioner(h, factor, independent=True)


@pytest.mark.parametrize("case", ["h_float32", "h_nan", "h_empty", "h_rank", "u_float32", "u_shape", "s_shape", "g_shape", "g_nan"])
def test_no_fractional_dtype_broadcast_or_nonfinite_coordinate_inputs(helper, case):
    h, _, _ = data()
    u, s, g = np.zeros((4, 4)), np.eye(4), np.zeros(16)
    with pytest.raises(ValueError):
        if case.startswith("h_"):
            bad = {"h_float32": h.astype(np.float32), "h_nan": h * np.nan,
                   "h_empty": h[:0], "h_rank": h[0]}[case]
            helper.build_preconditioner(bad)
        elif case == "g_shape":
            helper.pullback(g[:-1], s)
        elif case == "g_nan":
            helper.pullback(g + np.nan, s)
        else:
            if case == "u_float32":
                u = u.astype(np.float32)
            elif case == "u_shape":
                u = u[:1]
            else:
                s = s[:, :1]
            helper.materialize(u, s)


def validate_fit(helper, fit, values, independent=False):
    validator = importlib.import_module("manuscript.tools.fpga_lm_readout_preconditioned_validation")
    h, z0, targets = values
    arrays, endpoints = {}, {}
    for name in ("initial", "final"):
        saved = fit[name]
        arrays[name] = {key: torch.from_numpy(saved[source].copy()) for key, source in
                        {"U": "U", "theta": "theta", "gradient": "physical_gradient",
                         "transformed_gradient": "coordinate_gradient"}.items()}
        result = helper.oracle.independent_objective(saved["theta"], h, z0, targets, "affine") if independent else helper.oracle.objective(saved["theta"], h, z0, targets, "affine")
        endpoints[name] = {key: value for key, value in result.items() if key != "gradient"}
    record = {key: fit[key] for key in ("summary", "history", "iteration_history")}
    validator.validate_solver_record(record, endpoints, arrays, {"h": h, "logits": z0, "targets": targets}, fit["preconditioner"])
    return validator, record, endpoints, arrays


def test_real_tiny_solver_and_independent_physical_endpoint_validator(helper):
    values = data()
    before = [array.tobytes() for array in values]
    fit = helper.solve(*values)
    summary = fit["summary"]
    assert summary["numerically_converged"] is True and summary["numerical_gap_estimate"] <= 1e-5
    assert summary["stopping_gradient"] == "physical" and summary["zero_start"] is True
    assert summary["formal_certification"] is False and summary["restarts"] == 0
    assert 1 <= summary["function_calls"] == len(fit["history"]) <= 400
    assert summary["iterations"] == len(fit["iteration_history"]) <= 200
    validate_fit(helper, fit, values, independent=True)
    assert [array.tobytes() for array in values] == before


@pytest.mark.parametrize("reason", ["function_call_limit", "iteration_limit", "deadline", "scipy_stopped_above_numerical_target"])
def test_solver_caps_preserve_only_complete_accepted_point_and_never_retry(helper, monkeypatch, reason):
    scipy = pytest.importorskip("scipy.optimize")
    values = data()
    elapsed, calls, attempts = [0.], [], []
    original = helper.objective

    def objective(*args, **kwargs):
        calls.append(args[0].copy())
        return original(*args, **kwargs)

    def minimize(function, initial, **kwargs):
        attempts.append(True)
        assert kwargs["options"]["ftol"] == kwargs["options"]["gtol"] == 0.0
        function(initial)
        if reason == "function_call_limit":
            function(initial + .01)  # A trial, never an accepted endpoint.
            function(initial + .02)
        elif reason == "iteration_limit":
            point = initial + .01
            function(point)
            kwargs["callback"](point)
        elif reason == "deadline":
            elapsed[0] = 3600.
            function(initial + .01)
        else:
            return SimpleNamespace(status=2, message="synthetic finite stop", success=False, x=initial.copy())
        pytest.fail("bounded stop was ignored")

    monkeypatch.setattr(helper, "MAX_FUNCTION_CALLS", 2)
    monkeypatch.setattr(helper, "MAX_ITERATIONS", 1)
    monkeypatch.setattr(helper.time, "time", lambda: 1000. + elapsed[0])
    monkeypatch.setattr(helper.time, "monotonic", lambda: 1000. + elapsed[0])
    monkeypatch.setattr(helper, "objective", objective)
    monkeypatch.setattr(scipy, "minimize", minimize)
    fit = helper.solve(*values)
    assert attempts == [True] and fit["summary"]["stopping_reason"] == reason
    assert fit["summary"]["inconclusive"] is True and fit["summary"]["numerically_converged"] is False
    expected = calls[-1] if reason == "iteration_limit" else calls[0]
    assert fit["U"].tobytes() == expected.tobytes()
    assert len(calls) == fit["summary"]["function_calls"]
    validate_fit(helper, fit, values)


def test_scipy_success_with_small_transformed_gradient_is_not_physical_convergence(helper, monkeypatch):
    scipy = pytest.importorskip("scipy.optimize")
    h = np.array([[-1e6], [1e6]], dtype=np.float64)
    z0 = np.array([[np.log(1e8), 0.], [0., np.log(1e8)]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.int64)

    def minimize(function, initial, **kwargs):
        assert kwargs["options"]["gtol"] == 0.0
        _, gradient = function(initial)
        assert float(gradient @ gradient) / .002 < 1e-5
        return SimpleNamespace(status=0, message="synthetic transformed stop", success=True, x=initial.copy())

    monkeypatch.setattr(scipy, "minimize", minimize)
    fit = helper.solve(h, z0, y)
    assert fit["summary"]["numerical_gap_estimate"] > 1e-5
    assert fit["summary"]["stopping_reason"] == "scipy_stopped_above_numerical_target"
    assert fit["summary"]["numerically_converged"] is False
    validate_fit(helper, fit, (h, z0, y))


@pytest.mark.parametrize("case", ["nonzero_initial", "unbound_callback", "unaccepted_return"])
def test_solver_cannot_fabricate_initial_or_accepted_coordinates(helper, monkeypatch, case):
    scipy = pytest.importorskip("scipy.optimize")

    def minimize(function, initial, **kwargs):
        if case == "nonzero_initial":
            function(initial + .01)
        else:
            function(initial)
            if case == "unbound_callback":
                kwargs["callback"](initial + .01)
            else:
                return SimpleNamespace(status=0, message="forged", success=True, x=initial + .01)

    monkeypatch.setattr(scipy, "minimize", minimize)
    with pytest.raises(RuntimeError):
        helper.solve(*data())


def test_source_closure_and_unchanged_single_affine_protocol():
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_readout_preconditioned")
    sources = study.source_identity()
    assert len(sources) == 86 and study.previous.source_identity().items() <= sources.items()
    assert len(study.expected_artifacts()) == 8
    recipe = study.recipe()
    assert recipe["arms"] == ["affine_preconditioned"] and recipe["lambda"] == .001
    assert recipe["fit_windows"] == list(range(1024))
    assert recipe["sampled_training_windows"] == list(range(0, 1024, 16))
    assert recipe["development_windows"] == list(range(88, 152))
    assert recipe["new_backbone_forwards"] == recipe["backbone_gradient_queries"] == recipe["backbone_optimizer_steps"] == 0
    assert recipe["producer_training_gram_passes"] == 4 and recipe["independent_training_gram_passes"] == 1


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Real new binding/Gram/solve/audit using a real tiny old audit capsule."""
    previous_tests = importlib.import_module("test_fpga_lm_readout_oracle")
    inherited = previous_tests.synthetic.__wrapped__(tmp_path_factory)
    value = next(inherited)
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_readout_preconditioned")
    patch = pytest.MonkeyPatch()
    temporary = tmp_path_factory.mktemp("readout_preconditioned")
    limiter = None
    try:
        prior_audit = SimpleNamespace(**vars(value.config), audit_output=temporary / "prior_audit",
                                      audit_record=temporary / "prior_audit.json")
        value.study.audit(prior_audit)
        limiter = pytest.importorskip("threadpoolctl").threadpool_limits(limits=1)
        plan, source = temporary / "PLAN.md", temporary / "source.txt"
        plan.write_text("New synthetic cached-only protocol, distinct from historical declaration.\n")
        source.write_text("Synthetic new source closure.\n")
        for name, setting in {"TRAIN_WINDOWS": (0, 1, 2, 3), "SCORE_TRAIN_WINDOWS": (0, 2),
                              "DEV_WINDOWS": (88, 89), "SEQUENCE": 4, "WIDTH": 4, "VOCAB": 4,
                              "PLAN": plan}.items():
            patch.setattr(study, name, setting)
        for name, path in {"PRODUCER_SHA256": value.record, "MANIFEST_SHA256": value.root / "manifest.json",
                           "SEAL_SHA256": value.root / "artifact_manifest.json", "DECLARATION_SHA256": value.plan,
                           "AUDIT_SHA256": prior_audit.audit_record,
                           "AUDIT_MANIFEST_SHA256": prior_audit.audit_output / "audit_manifest.json"}.items():
            patch.setattr(study, name, study.old.digest(path.read_bytes()))
        patch.setattr(study, "source_identity", lambda: {**value.study.source_identity(), "new_source": study.old.digest(source.read_bytes())})
        original_prior_bind = study.previous.bind_inputs

        def prior_bind(config):
            binding, raw, token = original_prior_bind(config)
            # The old fixture's lightweight binder delegates its immutable
            # synthetic ancestry checks to frozen(); real admission does both.
            study.previous.frozen(binding, token)
            return binding, raw, token

        patch.setattr(study.previous, "bind_inputs", prior_bind)

        def forbidden(*args, **kwargs):
            pytest.fail("new cached-only run must not execute a backbone or refit the historical solver")

        for module, names in ((study.previous, ("run", "audit", "execute", "collect_cache", "frozen_model")),
                              (study.previous.oracle, ("extract_features", "solve"))):
            for name in names:
                patch.setattr(module, name, forbidden)
        calls = {"solve": 0, "gram": 0, "independent_gram": 0, "independent_objective": 0}
        gram_inputs = []
        original_gram, original_solve, original_objective = study.helper._gram, study.helper.solve, study.helper.objective

        def gram(h, chunk_size, checkpoint, independent=False):
            calls["independent_gram" if independent else "gram"] += 1
            gram_inputs.append(h.copy())
            return original_gram(h, chunk_size, checkpoint, independent)

        def solve(*args, **kwargs):
            calls["solve"] += 1
            return original_solve(*args, **kwargs)

        def objective(*args, **kwargs):
            if kwargs.get("independent", False):
                calls["independent_objective"] += 1
            return original_objective(*args, **kwargs)

        patch.setattr(study.helper, "_gram", gram)
        patch.setattr(study.helper, "solve", solve)
        patch.setattr(study.helper, "objective", objective)
        config = SimpleNamespace(prior_run=value.root, prior_record=value.record, prior_audit=prior_audit.audit_record,
                                 corpus=value.corpus, output=temporary / "producer", record=temporary / "public.json")
        prior_tree = tree_bytes(value.root)
        binding, raw, token = study.bind_inputs(config)
        result = study.run(config)
        producer_calls = dict(calls)
        audit_config = SimpleNamespace(**vars(config), audit_output=temporary / "audit", audit_record=temporary / "audit.json")
        receipt = study.audit(audit_config)
        yield SimpleNamespace(study=study, previous=value, prior_audit=prior_audit, temporary=temporary,
                              source=source, plan=plan, config=config, root=config.output, record=config.record,
                              binding=binding, raw=raw, token=token, result=result, audit_config=audit_config,
                              receipt=receipt, calls=calls, producer_calls=producer_calls, gram_inputs=gram_inputs,
                              prior_tree=prior_tree)
    finally:
        patch.undo()
        if limiter is not None:
            limiter.restore_original_limits()
        inherited.close()


def test_real_cached_only_producer_and_independent_audit_complete_without_backbone_or_refit(synthetic):
    value, study = synthetic, synthetic.study
    assert value.producer_calls == {"solve": 1, "gram": 4, "independent_gram": 0, "independent_objective": 0}
    assert value.calls == {"solve": 1, "gram": 4, "independent_gram": 1, "independent_objective": 2}
    assert tree_bytes(value.previous.root) == value.prior_tree
    assert len(study.inventory(value.root)) == len(study.expected_artifacts()) == 8
    assert value.result["complete"] is True and value.result["inherited_audited_cache_windows"] == 6
    assert value.result["initial_scores"] == value.previous.result["arms"]["affine"]["initial_scores"]
    assert value.result["solver_complete_labeled_position_evaluations"] == value.result["solver"]["summary"]["complete_function_evaluations"] * 16
    assert value.result["new_distinct_training_pairs"] == value.result["new_backbone_forwards"] == value.result["backbone_optimizer_steps"] == 0
    assert value.receipt["independent_dense_feature_windows"] == value.receipt["supplemental_guarded_feature_windows"] == 0
    assert value.receipt["independent_training_gram_checks"] == 1
    assert value.receipt["independent_coordinate_endpoint_checks"] == value.receipt["independent_readout_full_gradients"] == 2
    assert value.receipt["independent_initial_readout_score_windows"] == value.receipt["independent_final_readout_score_windows"] == 4
    assert value.receipt["optimizer_trajectory_replayed"] is False and value.receipt["formal_optimization_certificate"] is False
    assert {p.name for p in value.audit_config.audit_output.iterdir()} == {"audit_manifest.json", "declaration.md", "verification.json"}
    assert (value.audit_config.audit_output / "verification.json").read_bytes() == value.audit_config.audit_record.read_bytes()
    assert study.old.digest(value.previous.plan.read_bytes()) == study.DECLARATION_SHA256 != study.old.digest(value.plan.read_bytes())


def test_every_gram_pass_uses_only_actual_training_cache_and_positive_zero_start(synthetic):
    value, study = synthetic, synthetic.study
    expected = np.concatenate([study.load(study.previous.cache_path(value.previous.root, "train", i))["h"].numpy() for i in range(4)])
    assert len(value.gram_inputs) == 5
    assert all(array.tobytes() == expected.tobytes() for array in value.gram_inputs)
    initial = study.load(value.root / "initial.npz")
    for name in ("U", "theta"):
        assert initial[name].numpy().tobytes() == np.zeros_like(initial[name].numpy()).tobytes()
    assert [r["window"] for r in value.result["after"]["train"]["per_window"]] == [0, 2]
    assert [r["window"] for r in value.result["after"]["development"]["per_window"]] == [88, 89]


@pytest.mark.parametrize("case", ["U", "theta", "gradient", "transformed_gradient", "initial_signed_zero", "factor", "factor_summary"])
def test_saved_endpoint_and_factor_tampering_fail_independent_checks(synthetic, case):
    value, study = synthetic, synthetic.study
    name = "initial.npz" if case == "initial_signed_zero" else "final.npz"
    if case.startswith("factor"):
        name = "transform.json" if case == "factor_summary" else "transform.npz"
    path = value.root / name
    original = path.read_bytes()
    try:
        if case == "factor_summary":
            record = study.read(path)
            record["summary"]["orientation"] = "wrong coordinate map"
            path.write_text(json.dumps(record))
        else:
            arrays = study.load(path)
            if case == "initial_signed_zero":
                arrays["U"].reshape(-1)[0] = -0.
            else:
                arrays["S" if case == "factor" else case].reshape(-1)[0] += .1
            path.unlink()
            study.parent.parent.save(path, arrays)
        with pytest.raises(ValueError):
            study.execute(value.root, value.binding, value.raw, independent=True, clock=SimpleNamespace(check=lambda: None))
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("case", ["physical_stop", "call_bool", "call_cap", "accepted_hash", "trial_as_final", "false_convergence", "ridge_changed"])
def test_solver_provenance_is_bound_to_actual_accepted_arrays_and_physical_stopping(synthetic, case):
    value, study = synthetic, synthetic.study
    solver = study.read(value.root / "solver.json")
    if case == "physical_stop":
        solver["summary"]["stopping_gradient"] = "transformed"
    elif case == "call_bool":
        solver["summary"]["function_calls"] = True
    elif case == "call_cap":
        solver["summary"]["options"]["maxfun"] = 401
    elif case == "accepted_hash":
        solver["iteration_history"][-1]["U_sha256"] = "0" * 64
    elif case == "trial_as_final":
        solver["iteration_history"][-1]["function_call"] = 1
    elif case == "false_convergence":
        solver["summary"]["numerically_converged"] = not solver["summary"]["numerically_converged"]
    else:
        solver["summary"]["regularization"] = .002
    transform = {**{key: tensor.numpy() for key, tensor in study.load(value.root / "transform.npz").items()},
                 "summary": study.read(value.root / "transform.json")["summary"]}
    arrays = {name: study.load(value.root / (name + ".npz")) for name in ("initial", "final")}
    data = study.collect_cache(value.binding, value.raw, SimpleNamespace(check=lambda: None))["train"]
    with pytest.raises(ValueError):
        study.validator.validate_solver_record(solver, value.result["endpoints"], arrays, data, transform)


@pytest.mark.parametrize("case", ["source", "historical_source", "cache", "public", "private_verification", "checkpoint", "corpus"])
def test_actual_canonical_binding_and_frozen_proof_reject_changed_prior_bytes(synthetic, case):
    value, study = synthetic, synthetic.study
    path = {"source": value.source, "historical_source": value.previous.source,
            "cache": study.previous.cache_path(value.previous.root, "train", 1),
            "public": value.previous.record,
            "private_verification": value.prior_audit.audit_output / "verification.json",
            "checkpoint": Path(value.previous.binding["checkpoint"]["path"]),
            "corpus": value.previous.corpus}[case]
    original, before = path.read_bytes(), dict(value.calls)
    path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError):
            study.frozen(value.binding, value.token)
        if case != "source":  # A fresh standalone binder cannot know an earlier new-source snapshot.
            with pytest.raises(ValueError):
                study.bind_inputs(value.config)
        assert value.calls == before
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("case", ["extra_capsule_file", "extra_capsule_directory", "symlink_verification"])
def test_exact_predecessor_capsule_is_rechecked_after_admission(synthetic, case):
    value, study = synthetic, synthetic.study
    root = value.prior_audit.audit_output
    path = root / ("verification.json" if case == "symlink_verification" else "extra")
    original = path.read_bytes() if path.is_file() else None
    if case == "extra_capsule_file":
        path.write_text("unbound")
    elif case == "extra_capsule_directory":
        path.mkdir()
    else:
        path.unlink()
        path.symlink_to(value.prior_audit.audit_record)
    try:
        with pytest.raises(ValueError):
            study.frozen(value.binding, value.token)
    finally:
        path.rmdir() if path.is_dir() else path.unlink()
        if original is not None:
            path.write_bytes(original)


def test_invocation_validation_deepcopies_both_new_and_predecessor_identity(synthetic):
    value, study = synthetic, synthetic.study
    binding, sources, inherited = {"x": [1]}, {"source": "original"}, {"p": [2]}
    proof = study.Validation(binding, sources, inherited, value.token.predecessor_validation)
    binding["x"][0], sources["source"], inherited["p"][0] = 4, "changed", 5
    assert proof.binding == {"x": [1]} and proof.sources == {"source": "original"}
    assert proof.predecessor_binding == {"p": [2]}


def invocation(value, tmp_path, mode):
    fields = dict(vars(value.config))
    if mode == "producer":
        fields.update(output=tmp_path / "producer", record=tmp_path / "public.json")
        return value.study.run, SimpleNamespace(**fields), fields["output"], fields["record"]
    fields.update(audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    return value.study.audit, SimpleNamespace(**fields), fields["audit_output"], fields["audit_record"]


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("error", [TimeoutError, KeyboardInterrupt])
def test_failure_is_retained_no_success_receipt_and_alarm_restored(synthetic, tmp_path, monkeypatch, mode, error):
    value, study = synthetic, synthetic.study
    method, config, output, record = invocation(value, tmp_path, mode)
    original_handler = signal.getsignal(signal.SIGALRM)

    def fail(*args, **kwargs):
        raise error("synthetic bounded interruption")

    monkeypatch.setattr(study, "execute", fail)
    with pytest.raises(error):
        method(config)
    assert not record.exists() and study.read(output / "failure.json")["complete"] is False
    assert signal.getsignal(signal.SIGALRM) == original_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("replacement", ["same_inode", "foreign_inode"])
def test_failed_publication_preserves_foreign_receipts_and_retracts_owned_bytes(synthetic, tmp_path, monkeypatch, mode, replacement):
    value, study = synthetic, synthetic.study
    method, config, output, record = invocation(value, tmp_path, mode)
    original = study.parent._Publication.publish
    payload = b'{"complete":true,"changed":"synthetic publication"}\n'

    def publish(self, result):
        original(self, result)
        if replacement == "foreign_inode":
            record.unlink()
        record.write_bytes(payload)
        raise RuntimeError("synthetic publication interruption")

    monkeypatch.setattr(study.parent._Publication, "publish", publish)
    with pytest.raises(RuntimeError):
        method(config)
    failure = study.read(output / "failure.json")
    assert failure["complete"] is False
    if replacement == "same_inode":
        assert not record.exists()
        assert (output / "rejected_public.json").read_bytes() == payload
        assert failure["retained_rejected_public"]["sha256"] == study.old.digest(payload)
    else:
        assert record.read_bytes() == payload and not (output / "rejected_public.json").exists()


@pytest.mark.parametrize("target", ["producer_output", "producer_public", "audit_output"])
@pytest.mark.parametrize("marker", ["artifact_manifest.json", "audit_manifest.json"])
def test_no_new_path_can_be_nested_inside_an_unrelated_sealed_tree(synthetic, tmp_path, target, marker):
    value = synthetic
    method, config, _, _ = invocation(value, tmp_path, "audit" if target == "audit_output" else "producer")
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / marker).write_text("immutable unrelated capsule")
    before, calls = tree_bytes(sealed), dict(value.calls)
    setattr(config, {"producer_output": "output", "producer_public": "record", "audit_output": "audit_output"}[target], sealed / "new")
    with pytest.raises(ValueError):
        method(config)
    assert tree_bytes(sealed) == before and value.calls == calls


def test_audit_rejects_changed_new_declaration_before_endpoint_or_gram_calls(synthetic, tmp_path):
    value = synthetic
    method, config, output, record = invocation(value, tmp_path, "audit")
    original, before = value.plan.read_bytes(), dict(value.calls)
    value.plan.write_bytes(original + b"changed\n")
    try:
        with pytest.raises(ValueError):
            method(config)
        assert value.calls == before and not record.exists()
        assert (output / "failure.json").exists()
    finally:
        value.plan.write_bytes(original)


def test_changed_source_during_binding_is_not_adopted(synthetic, tmp_path, monkeypatch):
    value, study = synthetic, synthetic.study
    method, config, output, record = invocation(value, tmp_path, "producer")
    original, before = value.source.read_bytes(), dict(value.calls)
    real_bind = study.bind_inputs

    def bind(settings):
        value.source.write_bytes(original + b"changed during admission\n")
        return real_bind(settings)

    monkeypatch.setattr(study, "bind_inputs", bind)
    try:
        with pytest.raises(ValueError):
            method(config)
        assert value.calls == before and not record.exists() and (output / "failure.json").exists()
    finally:
        value.source.write_bytes(original)
