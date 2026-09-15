"""Pure mocked lifecycle tests: model/native/trajectory execution is forbidden."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from manuscript.tools import diagnose_fpga_lm_persistent_trajectory as study


@pytest.fixture(autouse=True)
def forbid_science(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("lifecycle tests must never execute a model, native engine, or trajectory")
    for name in ("trajectories", "later_probes", "bind_inputs"):
        monkeypatch.setattr(study, name, forbidden)
    monkeypatch.setattr(study.native, "EFSGD", forbidden)
    for module, names in ((study.extension, ("forward", "gradient", "raw_legacy_vjp")),
                          (study.reference, ("forward", "manual_vjp"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(study.torch, "set_num_threads", lambda value: None)


@pytest.fixture
def pure_run(tmp_path, monkeypatch):
    calls, timer_calls = [], []
    original_handler = object()
    alarm = {"handler": original_handler}
    source = {"version": "a" * 64}
    plan = tmp_path / "PLAN.md"
    plan.write_text("Declared synthetic lifecycle fixture. No numerical execution.\n")
    input_file = tmp_path / "input.json"
    input_file.write_text('{"fixture":true}\n')
    config = SimpleNamespace(output=tmp_path / "raw", record=tmp_path / "public.json",
                             corpus=tmp_path / "unused-corpus", prior_run=tmp_path / "old",
                             prior_record=tmp_path / "old-public.json")
    binding = {"inputs": [study.file_identity(input_file)]}
    monkeypatch.setattr(study, "PLAN", plan)
    monkeypatch.setattr(study, "source_identity", lambda: {"fixture.py": source["version"]})

    def bind(config):
        calls.append("bind")
        return binding, b"no corpus", {}

    def paired(*args):
        calls.append("trajectories")
        return []

    def probes(*args):
        calls.append("probes")
        return []

    def validate(*args):
        # This file tests orchestration; full numerical validation has its own suite.
        calls.append("validate")

    def verify(manifest):
        # The sibling complete-fixture suite owns real historical binding/schema
        # validation. Keep real synthetic identity checks for lifecycle ordering.
        calls.append("verify")
        study.equal(manifest["schema"], study.SCHEMA, "schema")
        study.equal(manifest["recipe"], study.recipe(), "recipe")
        study.equal(manifest["source_hashes"], study.source_identity(), "source identity")
        for item in manifest["binding"]["inputs"]:
            study.equal(study.file_identity(item["path"]), item, "bound input identity")

    def signal(signum, handler):
        assert signum == study.signal.SIGALRM
        previous, alarm["handler"] = alarm["handler"], handler
        return previous

    def timer(which, seconds):
        assert which == study.signal.ITIMER_REAL
        timer_calls.append(seconds)
        return (0.0, 0.0)

    monkeypatch.setattr(study, "bind_inputs", bind)
    monkeypatch.setattr(study, "trajectories", paired)
    monkeypatch.setattr(study, "later_probes", probes)
    monkeypatch.setattr(study, "validate_summary", validate)
    monkeypatch.setattr(study, "verify_inputs", verify)
    monkeypatch.setattr(study.signal, "signal", signal)
    monkeypatch.setattr(study.signal, "getsignal", lambda signum: alarm["handler"])
    monkeypatch.setattr(study.signal, "setitimer", timer)
    return SimpleNamespace(config=config, calls=calls, timers=timer_calls, alarm=alarm,
                           original_handler=original_handler, source=source, input=input_file,
                           binding=binding, bind=bind, paired=paired, probes=probes)


def assert_failure(run, kind):
    raw = run.config.output
    assert json.loads((raw / "failure.json").read_text())["type"] == kind
    assert not run.config.record.exists()
    assert run.alarm["handler"] is run.original_handler
    assert run.timers[-1] == 0


def test_pure_run_publishes_only_complete_bound_outputs_and_rechecks(pure_run):
    run = pure_run
    result = study.run(run.config)
    assert result["complete"] is True
    assert run.calls.count("trajectories") == run.calls.count("probes") == 1
    assert run.calls.count("bind") == 1
    assert run.alarm["handler"] is run.original_handler
    assert run.timers == [study.DEADLINE_S, 0]
    assert not (run.config.output / "failure.json").exists()
    assert study.check(run.config.output, run.config.record) == result
    assert result["heldout_corpus_evaluated"] is False
    assert result["physical_fpga_execution"] is False and result["paid_execution_authorized"] is False


@pytest.mark.parametrize("target", ["raw", "public"])
def test_existing_outputs_are_never_overwritten_or_executed(pure_run, target):
    run = pure_run
    if target == "raw":
        run.config.output.mkdir()
        marker = run.config.output / "user-data"
    else:
        marker = run.config.record
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        study.run(run.config)
    assert marker.read_text() == "preserve"
    assert not run.calls and not run.timers


@pytest.mark.parametrize("placement", ["same", "inside", "history"])
def test_public_and_raw_paths_cannot_overlap_immutable_history(pure_run, placement):
    run = pure_run
    if placement == "same":
        run.config.record = run.config.output
    elif placement == "inside":
        run.config.record = run.config.output / "public.json"
    else:
        history = run.config.output.parent / "history"
        history.mkdir()
        (history / "manifest.json").write_text(json.dumps({"schema": study.SCHEMA}))
        run.config.output = history / "new-run"
    with pytest.raises(ValueError):
        study.run(run.config)
    assert not run.calls and not run.timers
    assert not run.config.output.exists()


@pytest.mark.parametrize("stage", ["trajectories", "later_probes"])
@pytest.mark.parametrize("error", [ValueError("stage failure"), KeyboardInterrupt("stop"), TimeoutError("deadline")])
def test_stage_failures_are_retained_without_retry_or_publication(pure_run, monkeypatch, stage, error):
    run = pure_run

    def fail(*args):
        run.calls.append(stage)
        raise error

    monkeypatch.setattr(study, stage, fail)
    with pytest.raises(type(error), match=str(error)):
        study.run(run.config)
    assert_failure(run, type(error).__name__)
    assert run.calls.count(stage) == 1
    if stage == "trajectories":
        assert "probes" not in run.calls
    assert not (run.config.output / "summary.json").exists()


def test_actual_installed_timeout_callback_is_retained_and_restored(pure_run, monkeypatch):
    run = pure_run

    def expire(*args):
        run.calls.append("timeout callback")
        run.alarm["handler"](study.signal.SIGALRM, None)

    monkeypatch.setattr(study, "trajectories", expire)
    with pytest.raises(TimeoutError, match="deadline"):
        study.run(run.config)
    assert_failure(run, "TimeoutError")
    assert run.calls.count("timeout callback") == 1 and "probes" not in run.calls


def test_source_changed_during_binding_fails_before_any_science(pure_run, monkeypatch):
    run = pure_run

    def changed(config):
        result = run.bind(config)
        run.source["version"] = "b" * 64
        return result

    monkeypatch.setattr(study, "bind_inputs", changed)
    with pytest.raises(ValueError, match="source"):
        study.run(run.config)
    assert "trajectories" not in run.calls and "probes" not in run.calls
    assert not run.config.record.exists()
    if run.config.output.exists():
        assert (run.config.output / "failure.json").exists()


@pytest.mark.parametrize("mutation", ["source", "input"])
def test_after_stage_input_or_source_changes_cannot_publish(pure_run, monkeypatch, mutation):
    run = pure_run

    def changed(*args):
        result = run.probes(*args)
        if mutation == "source":
            run.source["version"] = "c" * 64
        else:
            run.input.write_text("changed bound input")
        return result

    monkeypatch.setattr(study, "later_probes", changed)
    with pytest.raises(ValueError):
        study.run(run.config)
    assert_failure(run, "ValueError")
    assert run.calls.count("trajectories") == run.calls.count("probes") == 1


def test_summary_validation_failure_prevents_success_artifacts(pure_run, monkeypatch):
    run = pure_run

    def reject(*args):
        raise ValueError("invalid saved summary")

    monkeypatch.setattr(study, "validate_summary", reject)
    with pytest.raises(ValueError, match="invalid saved summary"):
        study.run(run.config)
    assert_failure(run, "ValueError")
    assert not (run.config.output / "summary.json").exists()


def test_final_check_failure_cannot_leave_a_success_public_record(pure_run, monkeypatch):
    run = pure_run

    def reject(*args):
        raise ValueError("final independent check rejected")

    monkeypatch.setattr(study, "check", reject)
    with pytest.raises(ValueError, match="final independent check rejected"):
        study.run(run.config)
    assert_failure(run, "ValueError")
    assert json.loads((run.config.output / "rejected_public.json").read_text())["complete"] is True


def test_real_final_check_catches_source_change_and_retracts_only_owned_receipt(pure_run, monkeypatch):
    run, actual_check = pure_run, study.check

    def changed(root, record):
        run.source["version"] = "d" * 64
        return actual_check(root, record)

    monkeypatch.setattr(study, "check", changed)
    with pytest.raises(ValueError, match="source"):
        study.run(run.config)
    assert_failure(run, "ValueError")
    assert (run.config.output / "rejected_public.json").is_file()


def test_failed_final_check_does_not_delete_a_concurrently_changed_receipt(pure_run, monkeypatch):
    run = pure_run

    def changed(root, record):
        Path(record).write_text("a concurrent user edit; preserve these exact bytes")
        raise ValueError("public changed concurrently")

    monkeypatch.setattr(study, "check", changed)
    with pytest.raises(ValueError, match="public changed concurrently"):
        study.run(run.config)
    assert run.config.record.read_text() == "a concurrent user edit; preserve these exact bytes"
    assert not (run.config.output / "rejected_public.json").exists()
    assert json.loads((run.config.output / "failure.json").read_text())["type"] == "ValueError"
    assert run.alarm["handler"] is run.original_handler


def test_alarm_arming_failure_restores_handler_and_retains_owned_run_failure(pure_run, monkeypatch):
    run = pure_run

    def fail_arm(which, seconds):
        run.timers.append(seconds)
        if seconds:
            raise OSError("cannot arm timer")
        return (0, 0)

    monkeypatch.setattr(study.signal, "setitimer", fail_arm)
    with pytest.raises(OSError, match="cannot arm timer"):
        study.run(run.config)
    assert_failure(run, "OSError")
    assert "trajectories" not in run.calls and "probes" not in run.calls


@pytest.mark.parametrize("operation", ["disarm", "restore"])
def test_one_shot_cleanup_failure_retracts_success_and_finally_restores(pure_run, monkeypatch, operation):
    run = pure_run
    original_timer, original_signal = study.signal.setitimer, study.signal.signal
    failures = []

    def timer(which, seconds):
        if operation == "disarm" and seconds == 0 and not failures:
            failures.append(operation)
            raise OSError("one-shot cleanup failure")
        return original_timer(which, seconds)

    def signal(signum, handler):
        if operation == "restore" and handler is run.original_handler and not failures:
            failures.append(operation)
            raise OSError("one-shot cleanup failure")
        return original_signal(signum, handler)

    monkeypatch.setattr(study.signal, "setitimer", timer)
    monkeypatch.setattr(study.signal, "signal", signal)
    with pytest.raises(OSError, match="one-shot cleanup failure"):
        study.run(run.config)
    assert failures == [operation]
    assert_failure(run, "OSError")
    assert (run.config.output / "rejected_public.json").is_file()
    assert run.calls.count("trajectories") == run.calls.count("probes") == 1


def test_check_rejects_failure_marker_without_running_models(pure_run):
    run = pure_run
    study.run(run.config)
    (run.config.output / "failure.json").write_text('{"complete":false}\n')
    # Reseal to ensure this is the explicit failure guard, not only changed membership.
    (run.config.output / "artifact_manifest.json").write_text(json.dumps(study.prior.artifact_manifest(run.config.output)))
    with pytest.raises(ValueError, match="failed execution"):
        study.check(run.config.output, run.config.record)


def test_cli_check_dispatch_is_read_only_and_does_not_require_corpus(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(study, "check", lambda output, record: calls.append((output, record)) or {"complete": True})
    monkeypatch.setattr(study, "run", lambda *args: pytest.fail("check mode cannot run a study"))
    output, record = tmp_path / "raw", tmp_path / "public.json"
    assert study.main(["--check", "--output", str(output), "--record", str(record)]) == 0
    assert calls == [(output, record)]
    assert json.loads(capsys.readouterr().out)["check_only"] is True


@pytest.mark.parametrize("argv", [[], ["--output", "raw", "--record", "record.json"]])
def test_cli_rejects_missing_run_inputs_without_execution(argv):
    with pytest.raises(SystemExit) as caught:
        study.main(argv)
    assert caught.value.code == 2


def test_cli_help_is_offline(capsys):
    with pytest.raises(SystemExit) as caught:
        study.main(["--help"])
    assert caught.value.code == 0 and "CPU" in capsys.readouterr().out
