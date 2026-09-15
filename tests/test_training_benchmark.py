import json
import os
import sys
import time

import pytest
import torch

from scripts.route_certificates import training_benchmark as B


def config():
    return dict(seed=0, width=16, layers=1, vocab=11, batch=2, sequence=12,
                device="cpu", dtype="float32", lr=0.001, warmup=0, reps=1,
                compile_backend="aot_eager")


def test_missing_or_failed_arm_never_becomes_a_win():
    rows = [dict(arm=arm, status="ok", median_ms=2, checkpoint_policy="layer",
                 init_sha256="same", batch_sha256="same", update_count=2) for arm in B.ARMS]
    assert B.summarize(rows)["status"] == "complete"
    rows[1]["status"] = "parity_failed"
    result = B.summarize(rows)
    assert result["status"] == "inconclusive"
    assert result["packed_speedup_vs_best_baseline"] is None
    rows[1]["status"] = "ok"
    rows[1]["batch_sha256"] = "different"
    assert B.summarize(rows)["status"] == "inconclusive"


def test_checkpoint_policy_uses_measured_budget():
    rows = [dict(arm=a, status="ok", median_ms=2, cuda_peak_allocated_bytes=100,
                 checkpoint_policy="layer", init_sha256="same", batch_sha256="same", update_count=2) for a in B.ARMS]
    rows.append(dict(arm="checkpoint", status="ok", median_ms=1,
                     cuda_peak_allocated_bytes=200, checkpoint_policy="selective",
                     init_sha256="same", batch_sha256="same", update_count=2,
                     selective_policy_coverage=dict(status="observed")))
    assert B.summarize(rows, 150)["selected_checkpoint_policy"] == "layer"
    assert B.summarize(rows, 250)["selected_checkpoint_policy"] == "selective"


def test_heldout_learning_regression():
    results = B.learning_check(config(), steps=25)
    assert results["status"] == "pass"
    for metrics in results["models"].values():
        assert metrics["after"]["loss"] < metrics["before"]["loss"] * 0.75
        assert metrics["after"]["accuracy"] > 0.8


def test_untrained_synthetic_metrics_do_not_pass_learning_gate():
    results = B.learning_check(config(), steps=0)
    assert results["status"] == "fail"


def test_learning_control_records_actual_optimizer_separately():
    result = B.learning_check({**config(), "adamw_eps": 1e-4, "lr": 1e-5, "optimizer": "sgd"}, steps=0)
    optimizer = result["optimizer"]
    assert optimizer["name"] == "AdamW" and optimizer["lr"] == 0.01
    assert optimizer["eps"] == 1e-4 and optimizer["betas"] == [0.9, 0.999]
    assert optimizer["weight_decay"] == 0.01 and optimizer["execution"] == "eager"
    assert "separate" in optimizer["scope"]


@pytest.mark.parametrize("layers,span", [(1, 3), (2, 7), (4, 31)])
def test_model_context_metadata_reads_constructed_layers(layers, span):
    model = B.make_model({**config(), "layers": layers})
    metadata = B.model_context_metadata(model)
    assert metadata["causal_receptive_field"] == span
    assert metadata["convolution_layers"] == [dict(kernel_size=3, dilation=2 ** i) for i in range(layers)]
    model.layers[0].morph_conv.dilation = 3
    assert B.model_context_metadata(model)["causal_receptive_field"] == span + 4


def test_protocol_hashes_executable_helpers_and_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    paths = B.protocol_source_files(corpus)
    names = {p.name for p in paths}
    assert {"training_benchmark.py", "actual_model_benchmark.py", "provenance.py",
            "morphssm_tape.py", "device_tape.py"} <= names
    assert corpus.resolve() in paths


@pytest.mark.parametrize("outcome", [dict(returncode=1, stop_reason=None),
                                     dict(returncode=0, stop_reason="CELL_TIMEOUT")])
def test_failed_worker_cannot_supply_successful_json(outcome):
    row = B.validated_worker_row(dict(status="ok", measurement_status="fit"), outcome)
    assert row["status"] == "worker_failed"
    assert row["measurement_status"] != "fit"


def test_measured_final_state_is_checked_after_all_updates():
    c = config()
    model = B.make_model(c)
    ids, targets = B.make_batch(c, c["seed"] + 100)
    step = B.build_step(model, ids, targets, c, "eager")
    first_loss = step()
    first = (first_loss.cpu(), B.parameter_state_cpu(model))
    result = B.verify_arm(c, "eager", "layer", observed=first)
    assert result["status"] == "fail"  # one update cannot stand in for two
    last_loss = step()
    result = B.verify_arm(c, "eager", "layer", observed=(last_loss.cpu(), B.parameter_state_cpu(model)))
    assert result["status"] == "pass" and result["updates_checked"] == 2
    assert result["compared_measured_state"]


def test_corpus_split_and_unknown_vocab(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abcde" * 34 + "z" * 30)
    train, heldout, vocab = B.corpus_tokens(str(path))
    assert len(train) == 170 and len(heldout) == 30 and vocab == 6
    assert heldout.eq(0).all()  # no held-out vocabulary leakage
    c = {**config(), "corpus": str(path), "vocab": vocab, "sequence": 4}
    x, y = B.make_batch(c, 0)
    ex, ey = B.make_batch({**c, "split": "eval"}, 0)
    assert x.gt(0).all() and y.gt(0).all() and ex.eq(0).all() and ey.eq(0).all()


def test_adamw_epsilon_is_explicit_shared_protocol_metadata(capsys):
    import json
    assert B.adamw_epsilon(config()) == 1e-8
    assert B.main(["--dry-run", "--adamw-eps", "1e-4"]) == 0
    protocol = json.loads(capsys.readouterr().out)
    cells = protocol["cells"]
    assert all(cell["adamw_eps"] == 1e-4 for cell in cells)
    assert "compiled_checkpoint" in protocol["arms"]
    assert protocol["checkpoint_arms"] == ["checkpoint", "compiled_checkpoint"]
    assert protocol["checkpoint_policies"] == ["layer", "grouped", "selective"]


@pytest.mark.parametrize("eps", [0, -1e-8, float("nan"), float("inf")])
def test_adamw_epsilon_rejects_invalid_values(eps):
    with pytest.raises(ValueError, match="finite and positive"):
        B.adamw_epsilon({**config(), "adamw_eps": eps})
    with pytest.raises(SystemExit):
        B.main(["--dry-run", f"--adamw-eps={eps}"])


@pytest.mark.parametrize("arm", B.ARMS)
def test_every_arm_receives_same_adamw_epsilon(monkeypatch, arm):
    seen = []
    original = torch.optim.AdamW

    def capture(*args, **kwargs):
        seen.append(kwargs["eps"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", capture)
    c = {**config(), "adamw_eps": 1e-4}
    model = B.make_model(c)
    ids, targets = B.make_batch(c, 100)
    B.build_step(model, ids, targets, c, arm)
    assert seen == [1e-4]


@pytest.mark.parametrize("arm", ["packed", "unpacked", "compiled", "checkpoint", "compiled_checkpoint"])
def test_fullgraph_and_checkpoint_update_parity(arm):
    torch.set_num_threads(1)
    result = B.verify_arm(config(), arm, "layer")
    assert result["status"] == "pass", result


def test_compiled_checkpoint_is_required_and_competes_for_best_baseline():
    rows = [dict(arm=arm, status="ok", median_ms=2, checkpoint_policy="layer",
                 init_sha256="same", batch_sha256="same", update_count=2) for arm in B.ARMS]
    assert B.summarize([row for row in rows if row["arm"] != "compiled_checkpoint"])["status"] == "inconclusive"
    compiled = next(row for row in rows if row["arm"] == "compiled_checkpoint")
    compiled.update(median_ms=0.5, checkpoint_policy="grouped")
    result = B.summarize(rows)
    assert result["best_baseline_arm"] == "compiled_checkpoint"
    assert result["selected_compiled_checkpoint_policy"] == "grouped"
    assert result["packed_speedup_vs_best_baseline"] == 0.25


@pytest.mark.parametrize("coverage", [None, "unavailable", "unobserved"])
def test_unobserved_selective_policy_cannot_win(coverage):
    rows = [dict(arm=arm, status="ok", median_ms=2, checkpoint_policy="layer",
                 init_sha256="same", batch_sha256="same", update_count=2) for arm in B.ARMS]
    rows.append(dict(arm="compiled_checkpoint", status="ok", median_ms=0.01,
                     checkpoint_policy="selective", init_sha256="same", batch_sha256="same",
                     update_count=2, selective_policy_coverage={} if coverage is None else dict(status=coverage)))
    assert B.summarize(rows)["selected_compiled_checkpoint_policy"] == "layer"


@pytest.mark.parametrize("arm", B.CHECKPOINT_ARMS)
def test_selective_coverage_observes_ops_in_explicit_separate_scope(arm):
    import torch.utils.checkpoint as checkpoint
    if not hasattr(checkpoint, "create_selective_checkpoint_contexts"):
        pytest.skip("selective checkpointing unavailable in this PyTorch")
    torch.set_num_threads(1)
    result = B.selective_policy_coverage(config(), arm)
    assert result["status"] == "observed", result
    assert result["separate_untimed_forward_backward"]
    assert not result["physical_retention_or_recompute_savings_measured"]
    assert result["observed_save_ops"]
    assert "aten.amax.default" in result["observed_save_ops"]
    assert "aten.max.dim" in result["observed_save_ops"]
    assert result["visited_op_policy_calls"]["aten.amax.default"]["decision"] == "MUST_SAVE"
    assert any(op["decision"] == "PREFER_RECOMPUTE" for op in result["visited_op_policy_calls"].values())
    assert set(result["observed_save_ops"]) <= set(result["requested_save_ops"])
    assert all(result["target_policy_calls"][op]["forward"] > 0 for op in result["observed_save_ops"])
    assert result["scope"] == ("compile_trace_policy_calls" if arm == "compiled_checkpoint"
                               else "eager_runtime_policy_calls")


def test_missing_selective_api_is_not_replaced_by_layer_checkpoint(monkeypatch):
    import torch.utils.checkpoint as checkpoint
    monkeypatch.delattr(checkpoint, "create_selective_checkpoint_contexts", raising=False)
    model = B.make_model(config())
    ids, _ = B.make_batch(config(), 100)
    with pytest.raises(ImportError):
        B.checkpoint_forward(model, ids, "selective")


def test_requested_but_absent_selective_target_is_not_coverage(monkeypatch):
    import torch.utils.checkpoint as checkpoint
    if not hasattr(checkpoint, "create_selective_checkpoint_contexts"):
        pytest.skip("selective checkpointing unavailable in this PyTorch")
    monkeypatch.setattr(B, "selective_checkpoint_ops", lambda: (torch.ops.aten.mm.default,))
    result = B.selective_policy_coverage(config(), "checkpoint")
    assert result["status"] == "unobserved"
    assert result["observed_save_ops"] == []
    assert result["unobserved_save_ops"] == ["aten.mm.default"]


def test_compiled_checkpoint_compile_failure_never_falls_back(monkeypatch):
    def unavailable(*args, **kwargs):
        assert kwargs["fullgraph"] is True
        raise RuntimeError("test compiler failure")
    monkeypatch.setattr(torch, "compile", unavailable)
    model = B.make_model(config())
    ids, targets = B.make_batch(config(), 100)
    with pytest.raises(RuntimeError, match="test compiler failure"):
        B.build_step(model, ids, targets, config(), "compiled_checkpoint")


def test_compiled_checkpoint_worker_metadata():
    result = B.measure_cell(config(), "compiled_checkpoint", "layer")
    assert result["status"] == "ok", result
    assert result["checkpoint_policy"] == "layer"
    assert result["compile_fullgraph"] and result["compile_backend"] == "aot_eager"
    assert result["optimizer_execution"] == "eager"
    assert result["model_context"]["causal_receptive_field"] == 3


@pytest.mark.skipif(os.environ.get("TIKITORCH_REQUIRE_COMPILE") != "1",
                    reason="native toolchain validation is explicit")
def test_native_fullgraph_route_update():
    torch.set_num_threads(1)
    c = {**config(), "compile_backend": "inductor", "optimizer": "sgd"}
    if os.environ.get("TIKITORCH_REQUIRE_CUDA") == "1":
        assert torch.cuda.is_available(), "required CUDA is unavailable"
        c["device"] = "cuda"
    for arm in ("compiled", "compiled_checkpoint", "unpacked", "packed"):
        result = B.verify_arm(c, arm, "layer")
        assert result["status"] == "pass", (arm, result)


def _check_optional_native_selective(row, outcome, output):
    """A crashing/unsupported optional policy is not a successful baseline."""
    checked = B.validated_worker_row(row, outcome)
    if outcome.get("returncode") != 0 or outcome.get("stop_reason") is not None:
        assert checked["status"] != "ok"
        assert "compiled_checkpoint" in B.summarize([checked])["missing_arms"]
        pytest.skip("optional native selective checkpoint worker failed; no fallback or support claim: "
                    + json.dumps(dict(outcome=outcome, validated_row=checked, output=output)))
    # Normal completion still requires correctness; do not skip parity failures.
    assert checked["status"] == "ok", dict(row=checked, outcome=outcome, output=output)
    assert checked["parity"]["status"] == "pass", checked
    assert checked["selective_policy_coverage"]["status"] == "observed", checked


@pytest.mark.parametrize("outcome", [dict(returncode=-11, stop_reason=None),
                                     dict(returncode=1, stop_reason=None),
                                     dict(returncode=-15, stop_reason="CELL_TIMEOUT")])
def test_optional_native_selective_failure_is_explicit_and_excluded(outcome):
    # Even a success-looking partial JSON result cannot survive a failed worker.
    row = dict(arm="compiled_checkpoint", checkpoint_policy="selective", status="ok",
               median_ms=0.01, init_sha256="same", batch_sha256="same", update_count=2,
               parity=dict(status="pass"), selective_policy_coverage=dict(status="observed"))
    with pytest.raises(pytest.skip.Exception) as caught:
        _check_optional_native_selective(row, outcome, "complete worker output")
    assert json.dumps(outcome) in str(caught.value)
    assert "complete worker output" in str(caught.value)
    assert "no fallback or support claim" in str(caught.value)


def test_optional_native_selective_normal_completion_requires_parity():
    row = dict(arm="compiled_checkpoint", checkpoint_policy="selective", status="parity_failed",
               parity=dict(status="fail"), selective_policy_coverage=dict(status="observed"))
    with pytest.raises(AssertionError):
        _check_optional_native_selective(row, dict(returncode=0, stop_reason=None), "")


@pytest.mark.skipif(os.environ.get("TIKITORCH_REQUIRE_COMPILE") != "1",
                    reason="native toolchain validation is explicit")
@pytest.mark.parametrize("policy", B.POLICIES)
def test_native_compiled_checkpoint_policies(policy, tmp_path):
    torch.set_num_threads(1)
    import torch.utils.checkpoint as checkpoint
    if policy == "selective" and not hasattr(checkpoint, "create_selective_checkpoint_contexts"):
        pytest.skip("selective checkpointing unavailable in this PyTorch")
    c = {**config(), "layers": 2, "compile_backend": "inductor", "optimizer": "sgd"}
    if os.environ.get("TIKITORCH_REQUIRE_CUDA") == "1":
        assert torch.cuda.is_available(), "required CUDA is unavailable"
        c["device"] = "cuda"
    if policy != "selective":
        # Layer/grouped remain required, not optional skips.
        assert B.verify_arm(c, "compiled_checkpoint", policy)["status"] == "pass"
        return

    # Some native SAC/compiler combinations fail below Python, so both the
    # parity run and the auxiliary coverage run belong in a fresh worker. Reuse
    # the benchmark's bounded process-group/descendant cleanup on timeout.
    from scripts.route_certificates.actual_model_benchmark import run_subprocess_cell
    marker = "TIKITORCH_NATIVE_SELECTIVE_RESULT="
    code = """
import json
import sys
import torch
from scripts.route_certificates import training_benchmark as B
torch.set_num_threads(1)
c = json.loads(sys.argv[1])
print("native selective phase: parity begin", flush=True)
parity = B.verify_arm(c, "compiled_checkpoint", "selective")
print("native selective phase: parity end " + parity["status"], flush=True)
print("native selective phase: coverage begin", flush=True)
coverage = B.selective_policy_coverage(c, "compiled_checkpoint")
print("native selective phase: coverage end " + coverage["status"], flush=True)
row = dict(arm="compiled_checkpoint", checkpoint_policy="selective", parity=parity,
           selective_policy_coverage=coverage,
           status="ok" if parity["status"] == "pass" and coverage["status"] == "observed" else "parity_failed")
print("TIKITORCH_NATIVE_SELECTIVE_RESULT=" + json.dumps(row), flush=True)
"""
    log = tmp_path / "native-selective.log"
    outcome = run_subprocess_cell(
        [sys.executable, "-c", code, json.dumps(c)], log, tmp_path / "heartbeat.json",
        cell_id="native-selective", timeout_s=60, hard_deadline=time.monotonic() + 70,
        heartbeat_s=5)
    outcome["log_path"] = str(log)
    output = log.read_text(errors="replace")
    completed = [line[len(marker):] for line in output.splitlines() if line.startswith(marker)]
    row = json.loads(completed[-1]) if completed else dict(
        arm="compiled_checkpoint", checkpoint_policy="selective", status="unavailable",
        error="worker did not return a completed diagnostic result")
    _check_optional_native_selective(row, outcome, output)
