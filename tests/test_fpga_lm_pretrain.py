"""Fresh-window hardware-pilot contracts; native checks run in subprocesses."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from scripts import fpga_lm_pretrain as pilot


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes(bytes(range(128)) * 40)
    return path


def config(corpus, output, **overrides):
    args = pilot.parser().parse_args(["--corpus", str(corpus), "--output", str(output),
                                     "--library", "explicit.so", "--steps", "2",
                                     "--eval-every", "1", "--eval-windows", "1"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeBridge:
    last = None

    def __init__(self, path, backend, shape):
        self.shape = shape
        self.values = [0] * shape.ML_BUFFER_WORDS
        self.identity = {"path": str(path), "sha256": "test-fake-not-silicon"}
        self.training_inputs = []
        self.training_weight_values = []
        self.closed = False
        type(self).last = self

    def write(self, offset, values):
        self.values[offset:offset + len(values)] = values

    def read(self, offset, count):
        return self.values[offset:offset + count]

    def step(self, lr, threshold):
        shape = self.shape
        tokens = self.read(shape.OFF_TOKENS, shape.ML_L_SEQ)
        targets = self.read(shape.OFF_TARGETS, shape.ML_L_SEQ)
        result = pilot.integer_forward(pilot.read_weights(self, shape), tokens, targets, shape)
        self.write(shape.OFF_PRED, result["predictions"])
        self.write(shape.OFF_GRAD_LOGIT, result["gradients"])
        if lr:
            self.training_inputs.append(tokens)
            self.training_weight_values.append(self.values[shape.OFF_EMBED])
            self.values[shape.OFF_EMBED] += lr

    def close(self):
        self.closed = True


def test_shared_layout_ignores_environment_and_restores_it(monkeypatch):
    monkeypatch.setenv("ML_D", "512")
    monkeypatch.setenv("ML_BATCH", "8")
    shape = pilot.layout(16)
    assert shape.ML_D == 16 and shape.ML_BATCH == 1
    assert os.environ["ML_D"] == "512" and os.environ["ML_BATCH"] == "8"
    assert shape.OFF_CE_LUT % 16 == 0
    assert shape.ML_BUFFER_WORDS == shape.OFF_CE_LUT + 256
    assert len(pilot.layout_values(shape)) == 40


@pytest.mark.parametrize("width", [0, 15, 17, 1024])
def test_invalid_width_rejected(width):
    with pytest.raises(ValueError, match="width"):
        pilot.layout(width)


def test_corpus_windows_do_not_cross_partitions_or_repeat(corpus):
    raw, plan = pilot.corpus_plan(corpus, 2, 1)
    train = plan["splits"]["train"]
    for split in plan["splits"]:
        part = plan["splits"][split]
        index = part["windows"] - 1
        tokens, targets = pilot.window(raw, plan, split, index)
        start = part["start"] + index * 129
        assert start + 128 < part["stop"]
        assert tokens == list(raw[start:start + 128])
        assert targets == list(raw[start + 1:start + 129])
    assert train["stop"] == plan["splits"]["validation"]["start"]
    assert plan["test_policy"] == "reserved, not evaluated"
    with pytest.raises(ValueError, match="outside"):
        pilot.window(raw, plan, "train", train["windows"])


@pytest.mark.parametrize("payload", [b"", b"bad\xff", "é".encode()])
def test_non_ascii_empty_corpus_rejected(tmp_path, payload):
    path = tmp_path / "bad.txt"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="strict ASCII"):
        pilot.corpus_plan(path, 1, 1)


def test_corpus_capacity_never_recycles(corpus):
    with pytest.raises(ValueError, match="never repeats"):
        pilot.corpus_plan(corpus, 1000, 1)
    with pytest.raises(ValueError, match="validation/test"):
        pilot.corpus_plan(corpus, 1, 1000)


def test_plan_has_no_native_or_numpy_requirement(corpus, tmp_path):
    result = subprocess.run([sys.executable, "-S", str(pilot.REPO / "scripts/fpga_lm_pretrain.py"),
                             "--corpus", str(corpus), "--steps", "2", "--eval-windows", "1",
                             "--plan"], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["backend"] == "csim"
    assert record["dataset"]["sha256"] == pilot.digest(corpus.read_bytes())
    assert not (tmp_path / "run").exists()


def test_persistent_updates_frozen_validation_append_only_checkpoints(corpus, tmp_path):
    output = tmp_path / "run"
    result = pilot.run(config(corpus, output), FakeBridge)
    fake = FakeBridge.last
    assert fake.closed
    assert fake.training_weight_values[1] == fake.training_weight_values[0] + 1
    assert fake.training_inputs[0] != fake.training_inputs[1]
    assert result["training_token_exposures"] == 256
    assert result["distinct_training_windows"] == 2
    assert result["heldout_test_evaluated"] is False
    assert result["quality_advantage_established"] is False
    checkpoints = sorted(output.glob("checkpoint_*.i32"))
    assert len(checkpoints) == 3
    for path in checkpoints:
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["sha256"] == pilot.digest(path.read_bytes())
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records if record["kind"] == "train"] == [1, 2]
    assert all(record["weights_unchanged"] for record in records if record["kind"] == "validation")
    with pytest.raises(FileExistsError):
        pilot.run(config(corpus, output), FakeBridge)


def test_token_readback_mismatch_is_failure(corpus, tmp_path):
    class Corrupt(FakeBridge):
        def read(self, offset, count):
            values = super().read(offset, count)
            if offset == self.shape.OFF_TOKENS:
                values[0] ^= 1
            return values

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="TOKENS transport"):
        pilot.run(config(corpus, output), Corrupt)
    assert not (output / "summary.json").exists()
    assert json.loads((output / "failure.json").read_text())["complete"] is False
    assert Corrupt.last.closed


def test_native_failure_is_recorded(corpus, tmp_path):
    class Failing(FakeBridge):
        def step(self, lr, threshold):
            raise RuntimeError("native status -7")

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="status -7"):
        pilot.run(config(corpus, output), Failing)
    assert not (output / "summary.json").exists()
    assert Failing.last.closed


def test_cleanup_failure_cannot_publish_success(corpus, tmp_path):
    class FailingClose(FakeBridge):
        def close(self):
            raise RuntimeError("native close failed")

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="close failed"):
        pilot.run(config(corpus, output, steps=1), FailingClose)
    assert not (output / "summary.json").exists()
    assert json.loads((output / "failure.json").read_text())["complete"] is False


def test_frozen_eval_must_not_update_weights(corpus, tmp_path):
    class Wrong(FakeBridge):
        def step(self, lr, threshold):
            super().step(lr, threshold)
            self.values[self.shape.OFF_EMBED] += 1

    with pytest.raises(RuntimeError, match="evaluation changed weights"):
        pilot.run(config(corpus, tmp_path / "bad"), Wrong)


def test_independent_controlled_logits_and_balanced_gradient():
    shape = pilot.layout(16)
    weights = {name: [0] * getattr(shape, "WORDS_" + name) for name in pilot.WEIGHTS}
    weights["W_HEAD"] = [row for row in range(128) for _ in range(16)]
    result = pilot.integer_forward(weights, [0] * 128, [127] * 128, shape)
    assert result["predictions"] == [127] * 128
    assert all(sum(result["gradients"][i:i + 128]) == 0 for i in range(0, 128 * 128, 128))
    assert 0 < result["lut_normalized_nll"] < 7
    with pytest.raises(ValueError, match="ASCII"):
        pilot.integer_forward(weights, [128] * 128, [0] * 128, shape)


@pytest.mark.parametrize("key,value", [("steps", 0), ("eval_every", 0), ("eval_windows", 0),
                                       ("lr", 0), ("threshold", -1), ("width", 17)])
def test_bad_run_options_fail_before_native(corpus, tmp_path, key, value):
    with pytest.raises(ValueError):
        pilot.run(config(corpus, tmp_path / "bad", **{key: value}), FakeBridge)
    assert not (tmp_path / "bad").exists()


def test_pci_requires_image_and_existing_native_library(corpus, tmp_path):
    with pytest.raises(ValueError, match="image-id"):
        pilot.run(config(corpus, tmp_path / "bad", backend="pci"), FakeBridge)
    with pytest.raises(FileNotFoundError):
        pilot.NativeBridge(tmp_path / "missing.so", "csim", pilot.layout(16))


def test_native_bounds_without_loading_library():
    bridge = object.__new__(pilot.NativeBridge)
    bridge.shape = pilot.layout(16)
    for offset, count in ((-1, 1), (0, -1), (bridge.shape.ML_BUFFER_WORDS, 1)):
        with pytest.raises(ValueError, match="bounds"):
            bridge._bounds(offset, count)


def test_native_checked_transport_failure():
    bridge = object.__new__(pilot.NativeBridge)
    bridge.last_error = lambda handle: -42
    bridge.handle = object()
    with pytest.raises(RuntimeError, match="-42"):
        bridge._check()


def test_external_checkpoint_payloads_are_explicit(corpus, tmp_path):
    output, payloads = tmp_path / "records", tmp_path / "large-payloads"
    result = pilot.run(config(corpus, output, checkpoint_dir=payloads), FakeBridge)
    assert not list(output.glob("*.i32"))
    assert len(list(payloads.glob("*.i32"))) == 3
    final = result["final_checkpoint"]
    assert final["location_portable"] is False
    assert str(payloads) in final["file"]
    assert json.loads((output / "manifest.json").read_text())["numpy_version"]


@pytest.fixture
def paired_runs(corpus, tmp_path):
    left, right = tmp_path / "csim", tmp_path / "pci-test-fixture"
    pilot.run(config(corpus, left, steps=1), FakeBridge)
    shutil.copytree(left, right)
    # Test fixture only: no file here is physical hardware evidence.
    for name in ("manifest.json", "summary.json"):
        record = json.loads((right / name).read_text())
        record["backend"] = "pci"
        (right / name).write_text(json.dumps(record))
    return left, right


def test_paired_comparison_reads_every_full_checkpoint(paired_runs):
    result = pilot.compare_runs(*paired_runs)
    assert result["all_checkpoint_payloads_equal"] is True
    assert [row["step"] for row in result["checkpoints"]] == [0, 1]
    assert result["quality_or_speed_advantage_established"] is False


def test_paired_comparison_rejects_mixed_failure_success(paired_runs):
    left, right = paired_runs
    pilot.write_json(right / "failure.json", {"complete": False})
    with pytest.raises(ValueError, match="failed runs"):
        pilot.compare_runs(left, right)


def test_paired_comparison_rejects_changed_protocol(paired_runs):
    left, right = paired_runs
    path = right / "manifest.json"
    record = json.loads(path.read_text())
    record["seed"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="protocol mismatch: seed"):
        pilot.compare_runs(left, right)


def test_paired_comparison_rejects_changed_payload(paired_runs):
    left, right = paired_runs
    path = right / "checkpoint_000001.i32"
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="payload does not match"):
        pilot.compare_runs(left, right)


def test_paired_comparison_rejects_missing_trajectory(paired_runs):
    left, right = paired_runs
    path = right / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(row) for row in records if row["kind"] != "train") + "\n")
    with pytest.raises(ValueError, match="trajectory coverage"):
        pilot.compare_runs(left, right)


def test_paired_comparison_rejects_disconnected_summary(paired_runs):
    left, right = paired_runs
    path = right / "summary.json"
    record = json.loads(path.read_text())
    record["final_checkpoint"]["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="summary final checkpoint"):
        pilot.compare_runs(left, right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_paired_comparison_rejects_nonfinite_records(paired_runs, value):
    left, right = paired_runs
    path = right / "summary.json"
    record = json.loads(path.read_text())
    record["end_to_end_wall_s"] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="nonfinite"):
        pilot.compare_runs(left, right)


def test_paired_comparison_rejects_zero_step_protocol(paired_runs):
    for root in paired_runs:
        path = root / "manifest.json"
        record = json.loads(path.read_text())
        record["steps"] = 0
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="positive integers"):
        pilot.compare_runs(*paired_runs)


def test_paired_comparison_rejects_empty_but_matching_payloads(paired_runs):
    # Update every cross-link consistently: shape-derived completeness still wins.
    for root in paired_runs:
        records = []
        for path in sorted(root.glob("checkpoint_*.json")):
            record = json.loads(path.read_text())
            record.update(bytes=0, sha256=pilot.digest(b""), regions=[])
            path.write_text(json.dumps(record))
            path.with_suffix(".i32").write_bytes(b"")
            records.append(record)
        metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        for row in metrics:
            if row["kind"] == "validation":
                row["checkpoint"] = records[row["step"]]
        (root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n")
        summary = json.loads((root / "summary.json").read_text())
        summary["final_checkpoint"] = records[-1]
        (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="complete model weight payload"):
        pilot.compare_runs(*paired_runs)


def test_csim_manifest_rejected_before_loading_native(tmp_path):
    path = tmp_path / "not-a-real-library.so"
    path.write_bytes(b"invalid native file")
    pilot.write_json(tmp_path / "build.json", {"library_sha256": "wrong"})
    with pytest.raises(RuntimeError, match="build/source/layout mismatch"):
        pilot.NativeBridge(path, "csim", pilot.layout(16))


@pytest.mark.kernel
def test_csim_fresh_window_pilot_and_wrong_layout_are_process_isolated(corpus, tmp_path):
    if not (shutil.which("clang++") or shutil.which("g++")):
        pytest.skip("no C++17 compiler")
    script = """
import sys
from pathlib import Path
from scripts import fpga_lm_pretrain as p
root = Path(sys.argv[1])
library = p.build_csim(root / 'build', 16)
try:
    p.NativeBridge(library, 'csim', p.layout(32))
except RuntimeError as error:
    assert 'layout mismatch' in str(error)
else:
    raise AssertionError('wrong layout accepted')
try:
    p.NativeBridge(library, 'pci', p.layout(16))
except RuntimeError as error:
    assert 'backend capability' in str(error)
else:
    raise AssertionError('CSIM labeled PCI')
args = p.parser().parse_args(['--corpus', sys.argv[2], '--library', str(library),
    '--output', str(root / 'run'), '--steps', '2', '--eval-every', '1', '--eval-windows', '1'])
result = p.run(args)
assert result['complete'] and result['training_token_exposures'] == 256
assert result['initial_validation']['lut_normalized_nll'] > 0
"""
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(corpus)],
                               cwd=pilot.REPO, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
