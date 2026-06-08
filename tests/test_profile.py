"""Tests for tikitorch.profile — the one-call closure / hardware report."""
import torch
import torch.nn as tnn

import tikitorch as tt
from tikitorch.profile import profile_model, format_report


def test_tropical_cnn_reports_zero_multiplies_and_closed():
    cnn = tt.models.TropicalCNN(in_channels=1, num_classes=10)
    report = profile_model(cnn, torch.rand(1, 1, 28, 28))

    assert report["multiplications"] == 0
    assert report["closed"] is True
    assert report["structural_violations"] == []
    assert report["runtime_multiplies"] == []           # traced, multiplier-free
    assert report["param_count"] > 0

    text = format_report(report)
    # The headline line the paper quotes, exactly.
    assert "multiplications: 0" in text
    assert "closed: True" in text


def test_morphssm_lm_is_closed():
    mssm = tt.models.MorphSSM(d_model=64, n_layers=2, task_type="lm",
                              vocab_size=128, use_soft=False)
    report = profile_model(mssm, torch.randint(0, 128, (1, 16)))
    assert report["multiplications"] == 0
    assert report["closed"] is True
    assert "multiplications: 0" in format_report(report)


def test_nn_linear_reports_dense_multiply():
    # A model wrapping a standard nn.Linear is NOT closed — both the structural
    # scan and the runtime trace must surface the dense multiply.
    model = tnn.Sequential(tnn.Linear(8, 8))
    report = profile_model(model, torch.randn(2, 8))

    assert report["multiplications"] > 0
    assert report["closed"] is False
    # structural scan names the Linear module
    assert any(typ == "Linear" for _, typ in report["structural_violations"])
    # runtime trace flags the dense matmul op
    assert any(cat == "dense-multiply" for cat, _ in report["runtime_multiplies"])

    text = format_report(report)
    assert "multiplications: 0" not in text
    assert "Linear" in text


def test_structural_only_when_no_example_inputs():
    cnn = tt.models.TropicalCNN(in_channels=1, num_classes=10)
    report = profile_model(cnn)            # no example_inputs -> no runtime trace
    assert report["runtime_multiplies"] is None
    assert report["runtime_transcendental"] is None
    assert report["multiplications"] == 0  # structural alone is clean -> closed
    assert report["closed"] is True
    assert "runtime trace: skipped" in format_report(report)


def test_hardware_estimate_folded_in_when_exposed():
    # MorphSSM exposes hardware_cost(); the report should carry its summary dict.
    mssm = tt.models.MorphSSM(d_model=64, n_layers=2, task_type="lm",
                              vocab_size=128, use_soft=False)
    report = profile_model(mssm, torch.randint(0, 128, (1, 16)))
    assert report["hardware"] is not None
    assert "dsp_blocks" in report["hardware"]
    assert "hardware estimate" in format_report(report)


def test_hardware_estimate_absent_when_not_exposed():
    model = tnn.Sequential(tnn.ReLU())     # plain module, no hardware_cost()
    report = profile_model(model)
    assert report["hardware"] is None
    assert "not exposed" in format_report(report)


def test_runtime_multiply_counts_toward_headline_even_if_structural_clean():
    # A raw non-pow2 scalar multiply in a custom forward is invisible to the
    # structural scan but must be caught by the runtime trace and counted.
    class RawMul(tnn.Module):
        def forward(self, x):
            return x * 1.3
    report = profile_model(RawMul(), torch.randn(4))
    assert report["structural_violations"] == []
    assert report["multiplications"] >= 1
    assert report["closed"] is False


def test_cli_main_runs_for_builtin(capsys):
    from tikitorch.__main__ import main
    rc = main(["TropicalCNN"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "multiplications: 0" in out
    assert "TropicalCNN" in out


def test_cli_main_unknown_model_errors(capsys):
    from tikitorch.__main__ import main
    rc = main(["NotARealModel"])
    assert rc == 2
    assert "unknown model" in capsys.readouterr().err
