import os

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
                     init_sha256="same", batch_sha256="same", update_count=2))
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


@pytest.mark.parametrize("arm", ["packed", "unpacked", "compiled", "checkpoint"])
def test_fullgraph_and_checkpoint_update_parity(arm):
    torch.set_num_threads(1)
    result = B.verify_arm(config(), arm, "layer")
    assert result["status"] == "pass", result


@pytest.mark.skipif(os.environ.get("TIKITORCH_REQUIRE_COMPILE") != "1",
                    reason="native toolchain validation is explicit")
def test_native_fullgraph_route_update():
    c = {**config(), "compile_backend": "inductor", "optimizer": "sgd"}
    if os.environ.get("TIKITORCH_REQUIRE_CUDA") == "1":
        assert torch.cuda.is_available(), "required CUDA is unavailable"
        c["device"] = "cuda"
    for arm in ("compiled", "unpacked", "packed"):
        result = B.verify_arm(c, arm, "layer")
        assert result["status"] == "pass", (arm, result)
