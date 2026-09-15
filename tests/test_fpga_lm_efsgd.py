"""Unit-only tests for the new isolated CPU optimizer; no corpus or AWS use."""
import json
import random
import shutil
import subprocess

import pytest

from manuscript.tools import fpga_lm_efsgd as ef
from manuscript.tools import screen_fpga_lm_heads as screen
from manuscript.tools.observe_fpga_lm_routes import observe
from scripts import fpga_lm_pretrain as pilot


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if not (shutil.which("clang++") or shutil.which("g++")):
        pytest.skip("C++ compiler unavailable")
    return ef.build(tmp_path_factory.mktemp("efsgd") / "build", 16)


def initial(width=16, seed=0):
    shape = pilot.layout(width)
    if width == 512:
        return shape, screen.initialize_buffer(shape, seed, "anchors")
    buffer = shape.build_init(seed)
    buffer[shape.OFF_CFG] = 1
    buffer[shape.OFF_CFG + 1] = 4
    for vocabulary in range(shape.ML_V):
        for feature in range(shape.ML_D):
            if feature != (5 * vocabulary + seed) % shape.ML_D:
                buffer[shape.OFF_W_HEAD + vocabulary * shape.ML_D + feature] -= 8192
    return shape, buffer


def window(index=0):
    return ([(position * 7 + index) % 65 for position in range(128)],
            [(position * 11 + 3 + index) % 65 for position in range(128)])


def session(native, tmp_path, name="native", k=20, buffer=None, shape=None):
    default_shape, default_buffer = initial()
    return ef.EFSGD(native, shape or default_shape, default_buffer if buffer is None else buffer,
                    k, timeout_s=30, log_path=tmp_path / (name + ".log"))


@pytest.mark.parametrize("k", [0, 1, 8, 16, 20, 30])
def test_scalar_exact_signed_recurrence(k):
    rng = random.Random(137)
    for _ in range(100):
        weight = rng.randint(-100, 100)
        remainder = rng.randint(-(1 << k) + 1, (1 << k) - 1)
        gradient = rng.randint(-10000, 10000)
        accumulator = remainder + gradient
        expected_q = abs(accumulator) // (1 << k)
        if accumulator < 0:
            expected_q = -expected_q
        value, residual, update = ef.scalar_update(weight, remainder, gradient, k)
        assert (value, residual, update) == (weight - expected_q, accumulator - expected_q * (1 << k), expected_q)
        assert abs(residual) < (1 << k)


def test_scalar_preserves_cumulative_magnitude_and_negative_remainders():
    weight, remainder = 0, 0
    gradients = [3, -1, -7, 2, -6, 4, 9, -13]
    for gradient in gradients:
        weight, remainder, _ = ef.scalar_update(weight, remainder, gradient, 3)
    assert -weight * 8 + remainder == sum(gradients)
    assert ef.scalar_update(0, 0, -9, 3) == (1, -1, -1)
    assert ef.scalar_update(0, 0, -(1 << 31), 30) == (2, 0, -2)


@pytest.mark.parametrize("args,error", [((0, 0, 1, True), ValueError), ((0, 0, 1, -1), ValueError),
    ((0, 0, 1, 31), ValueError), ((0, 0, 1, 1.0), ValueError), ((0, 2, 1, 1), ValueError),
    ((0, -2, 1, 1), ValueError), ((False, 0, 1, 1), ValueError),
    ((0, (1 << 30) - 1, (1 << 31) - 1, 30), OverflowError),
    ((0, -(1 << 30) + 1, -(1 << 31), 30), OverflowError),
    ((32767, 0, -1, 0), OverflowError), ((-32768, 0, 1, 0), OverflowError)])
def test_scalar_rejects_invalid_or_unrepresentable(args, error):
    with pytest.raises(error):
        ef.scalar_update(*args)


def test_native_full_gradient_recurrence_and_evaluation_invariance(native, tmp_path):
    shape, buffer = initial()
    with session(native, tmp_path, buffer=buffer) as engine:
        before = engine.export_state()
        old = ef.decode_state(before, shape)
        tokens, targets = window()
        reference = observe(old["weights"], tokens, targets, shape, zshift=4, include_parameter_gradients=True)
        evaluated = engine.evaluate(tokens, targets)
        assert evaluated["step"] == 0
        for name in ("predictions", "gradients", "native_routes"):
            assert evaluated[name] == reference[name]
        assert engine.export_state() == before
        trained = engine.train(tokens, targets, routes=True)
        assert trained["step"] == 1
        for name in ("predictions", "gradients", "native_routes"):
            assert trained[name] == reference[name]
        state = ef.decode_state(engine.export_state(), shape)
        for name in pilot.WEIGHTS:
            for index, gradient in enumerate(reference["parameter_gradients"][name]):
                expected = ef.scalar_update(old["weights"][name][index], 0, gradient, 20)
                assert (state["weights"][name][index], state["remainders"][name][index]) == expected[:2]
        checkpoint = engine.export_state()
        engine.evaluate(*window(1))
        assert engine.export_state() == checkpoint


def test_resume_reproduces_further_training_and_reset(native, tmp_path):
    shape, buffer = initial()
    with session(native, tmp_path, "first") as first:
        original = first.export_state()
        first.train(*window())
        checkpoint = first.export_state()
        output = first.train(*window(1), routes=True)
        final = first.export_state()
        first.reset(buffer, 20)
        assert first.export_state() == original
    with session(native, tmp_path, "resumed") as resumed:
        resumed.import_state(checkpoint)
        assert resumed.train(*window(1), routes=True) == output
        assert resumed.export_state() == final
        assert ef.decode_state(final, shape)["step"] == 2


def test_multiple_native_processes_are_independent(native, tmp_path):
    with session(native, tmp_path, "left") as left, session(native, tmp_path, "right") as right:
        unchanged = right.export_state()
        left.train(*window())
        assert right.export_state() == unchanged


@pytest.mark.parametrize("k", [0, 1, 16, 30])
def test_configuration_roundtrip(native, tmp_path, k):
    shape, _ = initial()
    with session(native, tmp_path, k=k) as engine:
        payload = engine.export_state()
        state = ef.decode_state(payload, shape)
        assert state["k"] == k and state["step"] == 0
        assert all(not value for row in state["remainders"].values() for value in row)
        engine.import_state(payload)
        assert engine.export_state() == payload


@pytest.mark.parametrize("edit", [lambda x: x.pop(), lambda x: x.__setitem__(1, 2),
    lambda x: x.__setitem__(5, 31), lambda x: x.__setitem__(12, x[12] ^ 1),
    lambda x: x.__setitem__(ef.STATE_HEADER, 0),
    lambda x: x.__setitem__(ef.STATE_HEADER + 256, 32768),
    lambda x: x.__setitem__(-1, 1 << 20)])
def test_rejects_malformed_complete_state(native, tmp_path, edit):
    shape, _ = initial()
    with session(native, tmp_path) as engine:
        before = engine.export_state()
        values = ef._unpack(before)
        edit(values)
        with pytest.raises(ValueError):
            engine.import_state(ef._pack(values))
        assert engine.export_state() == before
        assert ef.decode_state(before, shape)["step"] == 0


def test_checkpoints_complete_exclusive_and_source_bound(native, tmp_path):
    shape, _ = initial()
    with session(native, tmp_path) as engine:
        engine.train(*window())
        record = engine.save_checkpoint(tmp_path, "state_000001")
        payload = (tmp_path / record["file"]).read_bytes()
        assert len(payload) == record["bytes"]
        assert pilot.digest(payload) == record["sha256"]
        assert record["identity"]["source_hashes"] == ef.source_identity()
        assert record["step"] == 1 and ef.decode_state(payload, shape)["step"] == 1
        assert json.loads((tmp_path / "state_000001.json").read_text()) == record
        with pytest.raises(FileExistsError):
            engine.save_checkpoint(tmp_path, "state_000001")
        with pytest.raises(ValueError):
            engine.save_checkpoint(tmp_path, "../escape")


@pytest.mark.parametrize("mode", ["forward", "update"])
def test_native_clipping_poison_cannot_publish_checkpoint(native, tmp_path, mode):
    shape, buffer = initial()
    if mode == "forward":
        buffer[shape.OFF_W_HEAD:shape.OFF_W_HEAD + shape.WORDS_W_HEAD] = [32767] * shape.WORDS_W_HEAD
        k = 20
    else:
        buffer = [0] * shape.ML_BUFFER_WORDS
        buffer[shape.OFF_CFG] = 1
        buffer[shape.OFF_CFG + 1] = 4
        buffer[shape.OFF_CE_LUT:shape.OFF_CE_LUT + 256] = [round(65536 * 2 ** (-i / 16)) for i in range(256)]
        k = 0
    with session(native, tmp_path, buffer=buffer, k=k) as engine:
        with pytest.raises(RuntimeError, match="failed closed"):
            engine.train([0] * 128, [1] * 128)
        assert engine.poisoned and engine.process.poll() is not None
        with pytest.raises(RuntimeError, match="poisoned"):
            engine.save_checkpoint(tmp_path, "invalid")
        assert not (tmp_path / "invalid.i32").exists()


def test_native_invalid_request_and_early_exit_poison(native, tmp_path):
    with session(native, tmp_path) as engine:
        with pytest.raises(RuntimeError, match="failed closed"):
            engine._exchange(99, b"")
        assert engine.poisoned
    with session(native, tmp_path, "killed") as engine:
        engine.process.kill()
        engine.process.wait(timeout=10)
        with pytest.raises((RuntimeError, BrokenPipeError)):
            engine.export_state()
        assert engine.poisoned


def test_source_binding_rejects_unsanitized_or_tampered_build(native, tmp_path):
    directory = tmp_path / "copy"
    shutil.copytree(native.parent, directory)
    record = json.loads((directory / "build.json").read_text())
    record["sanitizer"] = "disabled"
    (directory / "build.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="binding"):
        session(directory / native.name, tmp_path)


def test_timeout_reaps_only_owned_child(native, tmp_path):
    with session(native, tmp_path) as engine:
        import os
        import signal
        assert os.getpgid(engine.process.pid) == os.getpgrp()
        os.kill(engine.process.pid, signal.SIGSTOP)
        engine.timeout_s = 0.03
        with pytest.raises(TimeoutError):
            engine.export_state()
        assert engine.poisoned and engine.process.poll() is not None


def compile_probe(tmp_path, body):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    path = tmp_path / "probe.cpp"
    path.write_text('#define main ef_protocol_main\n#include "' + str(ef.CPP) + '"\n#undef main\nint main(){' + body + '}\n')
    binary = tmp_path / "probe"
    command = [compiler, *ef._flags(16, ef.source_identity()), "-I" + str(pilot.HLS), str(path), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return subprocess.run([str(binary)], capture_output=True, timeout=20)


def test_actual_ubsan_overflow_is_contained_in_native_child(tmp_path):
    result = compile_probe(tmp_path, 'g_embed_oc[0]=2147483647; volatile int one=1; GEMBED(0)+=one; return 0;')
    assert result.returncode != 0
    assert b"signed integer overflow" in result.stderr


def test_native_update_intmin_negative_truncation_and_transactional_clip(tmp_path):
    result = compile_probe(tmp_path, '''
        shift=30; g_embed_oc[0]=(-2147483647-1); if(apply_updates()!=0) return 1;
        if(embed_oc[0]!=2 || g_embed_oc[0]!=0) return 2;
        shift=3; g_embed_oc[0]=-9; if(apply_updates()!=0) return 3;
        if(embed_oc[0]!=3 || g_embed_oc[0]!=-1) return 4;
        shift=0; g_embed_oc[0]=1; wh_buf[WORDS_W_HEAD-1]=-32768;
        g_whead_oc[WORDS_W_HEAD-1]=1;
        if(apply_updates()!=60 || embed_oc[0]!=3 || g_embed_oc[0]!=1) return 5;
        return 0;''')
    assert result.returncode == 0, result.stderr


def test_d512_two_synthetic_windows_with_retained_remainder_native_parity(tmp_path):
    if not (shutil.which("clang++") or shutil.which("g++")):
        pytest.skip("C++ compiler unavailable")
    executable = ef.build(tmp_path / "d512", 512)
    shape, buffer = initial(512)
    with session(executable, tmp_path, "d512_native", shape=shape, buffer=buffer, k=20) as engine:
        before = ef.decode_state(engine.export_state(), shape)
        reference = observe(before["weights"], *window(), shape, zshift=4, include_parameter_gradients=True)
        result = engine.train(*window(), routes=True)
        for name in ("predictions", "gradients", "native_routes"):
            assert result[name] == reference[name]
        after = ef.decode_state(engine.export_state(), shape)
        for name in pilot.WEIGHTS:
            for i, gradient in enumerate(reference["parameter_gradients"][name]):
                assert (after["weights"][name][i], after["remainders"][name][i]) == ef.scalar_update(
                    before["weights"][name][i], 0, gradient, 20)[:2]
        checkpoint = engine.export_state()
        engine.evaluate(*window(1))
        assert engine.export_state() == checkpoint
        assert any(value for row in after["remainders"].values() for value in row)
        reference_next = observe(after["weights"], *window(1), shape, zshift=4,
                                 include_parameter_gradients=True)
        next_result = engine.train(*window(1), routes=True)
        assert next_result["step"] == 2
        for name in ("predictions", "gradients", "native_routes"):
            assert next_result[name] == reference_next[name]
        next_state = ef.decode_state(engine.export_state(), shape)
        for name in pilot.WEIGHTS:
            for i, gradient in enumerate(reference_next["parameter_gradients"][name]):
                assert (next_state["weights"][name][i], next_state["remainders"][name][i]) == ef.scalar_update(
                    after["weights"][name][i], after["remainders"][name][i], gradient, 20)[:2]
