"""The structural-audit CLI preserves its gate status across loading and reports."""

import sys
from types import ModuleType

import pytest
from torch import nn

from tikitorch.cli import _load, main


@pytest.fixture
def targets(monkeypatch):
    module = ModuleType("tikitorch_test_cli_targets")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _spec(module, name):
    return f"{module.__name__}:{name}"


def test_load_module_instance_does_not_call_forward(targets):
    class ExistingModel(nn.Module):
        def forward(self, x):
            raise AssertionError("loading a model must not execute forward")

    targets.instance = ExistingModel()
    assert _load(_spec(targets, "instance")) is targets.instance


@pytest.mark.parametrize("kind", ["class", "factory", "instance"])
def test_closed_target_passes_for_each_supported_form(targets, capsys, kind):
    targets.model = {
        "class": nn.Identity,
        "factory": lambda: nn.Identity(),
        "instance": nn.Identity(),
    }[kind]
    assert main([_spec(targets, "model")]) == 0
    assert "SOFTWARE-CLOSED (STRUCTURAL)" in capsys.readouterr().out


@pytest.mark.parametrize("report", [False, True])
def test_report_does_not_bypass_nonclosed_gate(targets, capsys, report):
    targets.model = nn.Sequential(nn.Linear(2, 2))
    args = [_spec(targets, "model")] + (["--report"] if report else [])
    assert main(args) == 1
    captured = capsys.readouterr()
    assert "NOT CLOSED" in captured.err
    assert "SOFTWARE-CLOSED" not in captured.out
    if report:
        assert "Linear" in captured.out


@pytest.mark.parametrize("report", [False, True])
def test_declared_boundary_has_same_status_with_report(targets, capsys, report):
    targets.model = nn.Sequential(nn.Linear(2, 2))
    args = [_spec(targets, "model"), "--allow", "0"]
    if report:
        args.append("--report")
    assert main(args) == 0
    output = capsys.readouterr().out
    assert "SOFTWARE-CLOSED (STRUCTURAL)" in output
    assert "except declared boundaries: 0" in output


@pytest.mark.parametrize("target", [
    "no_colon",
    ":missing_module",
    "tikitorch_test_cli_targets:",
    "tikitorch_test_cli_nonexistent:model",
    "tikitorch_test_cli_targets:missing_attribute",
])
def test_invalid_target_is_usage_error(targets, capsys, target):
    with pytest.raises(SystemExit) as exc:
        main([target])
    assert exc.value.code == 2
    assert "could not prepare target" in capsys.readouterr().err


@pytest.mark.parametrize("target, message", [
    (42, "must name an nn.Module"),
    (lambda: object(), "expected nn.Module"),
    (nn.Linear, "required positional arguments"),
])
def test_invalid_target_type_or_factory_is_usage_error(targets, capsys, target, message):
    targets.model = target
    with pytest.raises(SystemExit) as exc:
        main([_spec(targets, "model")])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_factory_failure_is_reported_as_usage_error(targets, capsys):
    def broken_factory():
        raise RuntimeError("missing model configuration")

    targets.model = broken_factory
    with pytest.raises(SystemExit) as exc:
        main([_spec(targets, "model")])
    assert exc.value.code == 2
    assert "missing model configuration" in capsys.readouterr().err


def test_unexpected_auditor_failure_is_not_hidden(targets, monkeypatch):
    import tikitorch.audit

    targets.model = nn.Identity()

    def broken_audit(*args, **kwargs):
        raise RuntimeError("unexpected audit implementation failure")

    monkeypatch.setattr(tikitorch.audit, "assert_closed", broken_audit)
    with pytest.raises(RuntimeError, match="unexpected audit implementation failure"):
        main([_spec(targets, "model")])
