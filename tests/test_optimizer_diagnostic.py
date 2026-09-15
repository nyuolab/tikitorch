import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.route_certificates import optimizer_diagnostic as D


def test_cache_reset_supports_pytorch_20_without_compiler_namespace(monkeypatch):
    import torch._dynamo
    calls = []
    monkeypatch.setattr(D, "torch", SimpleNamespace(), raising=False)
    monkeypatch.setattr(torch._dynamo, "reset", lambda: calls.append("dynamo"))
    D._reset_compile_cache()
    assert calls == ["dynamo"]


def test_cache_reset_prefers_public_compiler_namespace(monkeypatch):
    calls = []
    compiler = SimpleNamespace(reset=lambda: calls.append("compiler"))
    monkeypatch.setattr(D, "torch", SimpleNamespace(compiler=compiler), raising=False)
    D._reset_compile_cache()
    assert calls == ["compiler"]


def test_adamw_amplifies_null_gradient_cancellation_without_an_update_bug():
    # First-step AdamW has delta = -lr*g/(abs(g)+eps); tiny opposite-signed
    # cancellation residues can therefore create a near-learning-rate update.
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    gradients = (-2.5858753360807896e-8, 1.7025740817189217e-8)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = torch.tensor([gradient])
        torch.optim.AdamW([parameter], lr=1e-3, eps=1e-8, weight_decay=0.01).step()
        expected = -1e-3 * gradient / (abs(gradient) + 1e-8)
        assert parameter.item() == pytest.approx(expected, rel=1e-6)
    assert abs(parameters[0].item() - parameters[1].item()) > 1e-3


def test_matched_state_control_has_structural_not_thresholded_null_coordinate():
    torch.set_num_threads(1)
    result = D.primitive_controls({**D.default_config(), "compile_backend": "aot_eager"})
    assert result["same_gradient_adamw_max_parameter_error"] == 0
    assert not any(result["initial_route_differences"].values())
    nulls = result["structural_null_coordinates"]
    assert nulls
    for row in nulls:
        assert set(row["classes_selecting_feature_per_row"]) <= {0, 11}
        assert abs(row["fp64_gradient"]) < 1e-12


def test_trajectory_records_gradients_moments_updates_and_batches():
    torch.set_num_threads(1)
    result = D.compare_trajectory({**D.default_config(), "compile_backend": "aot_eager"},
                                  eps=1e-4, steps=2, arms=("packed",))
    assert result["status"] == "pass"
    assert result["optimizer"]["eps"] == 1e-4
    assert result["atol"] == result["rtol"] == 1e-4
    assert result["rows"][0]["batch_sha256"] != result["rows"][1]["batch_sha256"]
    parameter = result["rows"][0]["parameters"][0]
    assert {"gradient", "parameter", "update", "exp_avg", "exp_avg_sq",
            "gradient_at_max_parameter_error"} <= parameter.keys()


def test_existing_evidence_is_never_overwritten(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("original")
    with pytest.raises(SystemExit):
        D.main(["--output", str(path)])
    assert path.read_text() == "original"


def test_integrity_checker_needs_no_site_packages():
    result = subprocess.run([sys.executable, "-S", str(D.__file__), "--help"],
                            check=True, text=True, capture_output=True)
    assert "--check" in result.stdout


@pytest.mark.parametrize("dimension", ["width", "layers", "vocab", "batch", "sequence"])
def test_cli_rejects_invalid_shape_before_compilation(dimension, monkeypatch):
    def unexpected_runtime():
        pytest.fail("invalid dimensions must fail before loading PyTorch")
    monkeypatch.setattr(D, "_load_runtime", unexpected_runtime)
    with pytest.raises(SystemExit):
        D.main([f"--{dimension}", "0"])


def test_cli_records_explicit_shape(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(D, "_load_runtime", lambda: None)
    monkeypatch.setattr(D, "torch", torch, raising=False)
    def compare(config, **kwargs):
        seen.append(config)
        return dict(status="pass")
    monkeypatch.setattr(D, "compare_trajectory", compare)
    output = tmp_path / "shape.json"
    assert D.main(["--width", "32", "--layers", "2", "--sequence", "24",
                   "--batch", "4", "--vocab", "17", "--steps", "1",
                   "--eps", "1e-4", "--seeds", "0", "--skip-controls",
                   "--output", str(output)]) == 0
    config = json.loads(output.read_text())["protocol"]["config"]
    assert (config["width"], config["layers"], config["sequence"],
            config["batch"], config["vocab"]) == (32, 2, 24, 4, 17)
    assert seen == [config]


def test_source_manifest_covers_executable_dependency_set():
    from scripts.route_certificates import training_benchmark as B
    assert set(B.protocol_source_files()) | {D.Path(D.__file__)} == set(D.source_paths())


def test_diagnostic_content_digest_detects_row_mutation():
    record = {"rows": [{"status": "fail"}]}
    record["record_sha256"] = D.record_digest(record)
    assert D.record_digest(record) == record["record_sha256"]
    record["rows"][0]["status"] = "pass"
    assert D.record_digest(record) != record["record_sha256"]


def test_check_preserves_negative_outcomes_and_rejects_stale_source(tmp_path):
    measurement = dict(max_abs=0, differing_elements=0, flat_index=0, expected=0, actual=0, allclose=True)
    parameter = {quantity: dict(measurement) for quantity in (
        "gradient", "parameter", "update", "same_state_gradient", "exp_avg", "exp_avg_sq")}
    parameter.update(name="test.weight", gradient_allclose=True, parameter_allclose=True)
    row = dict(update=1, arm="compiled", status="fail", parameters=[parameter], loss_allclose=False,
               loss={**measurement, "max_abs": 1, "actual": 1, "differing_elements": 1, "allclose": False},
               independent_state_route_changes=0, same_state_compiled_route_changes=0)
    config = D.default_config()
    record = dict(schema="tikitorch.optimizer_diagnostic.v1", source=D.source_identity(),
                  protocol=dict(config=config, seeds=[0], epsilons=[1e-8], steps=1, arms=["compiled"],
                                fresh_batches=True, controls=False), primitive_controls=None,
                  trajectories=[dict(config=config, fresh_batches=True, steps=1, arms=["compiled"], rows=[row], atol=1e-4, rtol=1e-4,
                                     status="fail", first_failure=dict(update=1, arm="compiled"),
                                     optimizer=dict(name="torch.optim.AdamW", compiled=False, eps=1e-8,
                                                    lr=0.001, betas=[0.9, 0.999], weight_decay=0.01))])
    record["record_sha256"] = D.record_digest(record)
    path = tmp_path / "diagnostic.json"
    path.write_text(json.dumps(record))
    assert D.check_record(path)["trajectories"][0]["status"] == "fail"
    subprocess.run([sys.executable, "-S", str(D.__file__), "--check", "--record", str(path)],
                   check=True, capture_output=True)
    record["source"]["files"]["scripts/route_certificates/optimizer_diagnostic.py"] = "stale"
    record["record_sha256"] = D.record_digest(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="stale"):
        D.check_record(path)


@pytest.mark.parametrize("key,value", [("max_abs", -1), ("flat_index", -1), ("actual", float("nan"))])
def test_invalid_measurements_fail_closed(key, value):
    with pytest.raises(ValueError, match="invalid|nonfinite"):
        D._check_finite({key: value})


@pytest.mark.skipif(os.environ.get("TIKITORCH_REQUIRE_COMPILE") != "1",
                    reason="native toolchain validation is explicit")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_native_adamw_explicit_epsilon_fresh_batch_recipe(seed):
    torch.set_num_threads(1)
    config = {**D.default_config(), "seed": seed}
    if os.environ.get("TIKITORCH_REQUIRE_CUDA") == "1":
        assert torch.cuda.is_available(), "required CUDA is unavailable"
        config["device"] = "cuda"
    result = D.compare_trajectory(config, eps=1e-4, steps=20)
    assert result["status"] == "pass", result["first_failure"]
