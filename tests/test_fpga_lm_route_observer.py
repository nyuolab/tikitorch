"""Independent controlled VJPs and frozen-native checks for the host observer."""
import copy
import math
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from manuscript.tools.observe_fpga_lm_routes import observe
from scripts import fpga_lm_pretrain as pilot


@pytest.fixture(scope="module")
def shape():
    return pilot.layout(16)


def zero_weights(shape):
    d, n, k, v = shape.ML_D, shape.ML_N_LAYERS, shape.ML_K, shape.ML_V
    return {"EMBED": [0] * (v * d), "MH": [0] * (n * d * k),
            "MP": [0] * (n * d * d), "MB": [0] * (n * d),
            "W_HEAD": [0] * (v * d)}


def anchor_weights(shape):
    weights = zero_weights(shape)
    weights["W_HEAD"] = [0 if j == v % shape.ML_D else -8192
                         for v in range(shape.ML_V) for j in range(shape.ML_D)]
    return weights


def initialized_weights(shape):
    buffer = shape.build_init(7)
    weights = {name: buffer[getattr(shape, "OFF_" + name):
                            getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
               for name in pilot.WEIGHTS}
    # A deterministic wider head exercises routes beyond the initialization's
    # single dominant feature; the backbone remains the frozen initializer's.
    weights["W_HEAD"] = [(v * 37 + j * 19) % 1025 - 512
                         for v in range(shape.ML_V) for j in range(shape.ML_D)]
    return weights


def observe_zeros(weights, shape, **kwargs):
    return observe(weights, [0] * shape.ML_L_SEQ, [0] * shape.ML_L_SEQ,
                   shape, include_parameter_gradients=True, **kwargs)


def test_balanced_head_scatter_cancels_without_erasing_head_parameter_gradients(shape):
    result = observe_zeros(zero_weights(shape), shape)
    length, d, vocab = shape.ML_L_SEQ, shape.ML_D, shape.ML_V
    per_logit = [-65024] + [512] * (vocab - 1)
    assert result["predictions"] == [0] * length
    assert result["gradients"] == per_logit * length
    assert result["lut_normalized_nll"] == pytest.approx(math.log(vocab))
    assert result["accuracy"] == 1.0
    assert result["saturation_events"] == result["rounded_zero_target_mass"] == 0
    assert all(not any(values) for values in result["native_routes"].values())
    diagnostic = result["route_diagnostics"]
    assert diagnostic["nonzero_logit_gradient_positions"] == length
    assert diagnostic["exact_zero_head_input_vjp_on_nonzero_gradient_positions"] == length
    assert diagnostic["single_shared_head_winner_positions"] == length
    assert diagnostic["head_winner_counts"] == [1] * length
    assert diagnostic["head_winners_ge16_positions"] == 0
    assert diagnostic["nonzero_embedding_input_vjp_positions"] == 0
    assert diagnostic["backward_int32_overflow_detected"] is False
    grads = result["parameter_gradients"]
    assert all(not any(grads[name]) for name in ("EMBED", "MH", "MP", "MB"))
    assert grads["W_HEAD"] == [length * per_logit[v] if j == 0 else 0
                                for v in range(vocab) for j in range(d)]
    for row in diagnostic["projection_layers"]:
        assert row["single_projection_winner_positions"] == length
        assert row["nonzero_incoming_vjp_positions"] == 0
        # Conditional cancellation must not count a branch with no input VJP.
        assert row["zero_projection_input_vjp_on_nonzero_incoming_positions"] == 0
        assert row["nonzero_temporal_carry_positions"] == 0
        assert row["nonzero_centering_remainder_positions"] == 0


@pytest.mark.parametrize("saturate_residual", [False, True])
def test_projection_cancellation_preserves_residual_and_row_parameter_gradients(
        shape, saturate_residual):
    weights = anchor_weights(shape)
    if saturate_residual:
        weights["EMBED"] = [32767] * len(weights["EMBED"])
        weights["MB"] = [32767] * len(weights["MB"])
    result = observe_zeros(weights, shape)
    length, d, n = shape.ML_L_SEQ, shape.ML_D, shape.ML_N_LAYERS
    # Eight vocabulary rows route to each feature. Their sum is zero globally,
    # but not componentwise; a subsequent common projection winner cancels it.
    head_vjp = [-61440] + [4096] * (d - 1)
    diagnostic = result["route_diagnostics"]
    assert diagnostic["head_winner_counts"] == [16] * length
    assert diagnostic["head_winners_ge16_positions"] == length
    assert diagnostic["exact_zero_head_input_vjp_on_nonzero_gradient_positions"] == 0
    assert diagnostic["max_abs_head_input_vjp"] == 61440
    assert diagnostic["nonzero_embedding_input_vjp_positions"] == length
    assert diagnostic["max_abs_embedding_input_vjp"] == 61440
    assert result["saturation_events"] == (length * n * d if saturate_residual else 0)
    assert result["lut_normalized_nll"] == pytest.approx(math.log(128))
    assert result["native_routes"]["AD_HEAD"] == list(range(16)) * (128 * 8)
    assert not any(result["native_routes"]["PROJ_IDX"])
    grads = result["parameter_gradients"]
    assert not any(grads["MH"])
    assert grads["MB"] == [length * value for value in head_vjp] * n
    assert grads["MP"] == [length * head_vjp[j] if k == 0 else 0
                            for _ in range(n) for j in range(d) for k in range(d)]
    assert grads["EMBED"] == [length * value for value in head_vjp] + [0] * (127 * d)
    for row in diagnostic["projection_layers"]:
        assert row["single_projection_winner_positions"] == length
        assert row["nonzero_incoming_vjp_positions"] == length
        assert row["zero_projection_input_vjp_on_nonzero_incoming_positions"] == length
        assert row["projection_incoming_l1"] == length * 122880
        assert row["projection_scattered_l1"] == 0
        assert row["max_abs_projection_input_vjp"] == 0
        assert row["max_abs_layer_input_vjp"] == 61440
        assert row["nonzero_temporal_carry_positions"] == 0
        assert row["nonzero_centering_remainder_positions"] == 0


def test_head_route_is_selected_before_logit_clipping(shape):
    weights = zero_weights(shape)
    weights["EMBED"] = [32767] * len(weights["EMBED"])
    weights["W_HEAD"] = [32766, 32767] + [0] * (shape.ML_D - 2)
    weights["W_HEAD"] *= shape.ML_V
    result = observe_zeros(weights, shape)
    # Both candidates saturate to 32767, but the raw larger candidate is j=1.
    assert result["native_routes"]["AD_HEAD"] == [1] * (128 * 128)
    assert result["predictions"] == [0] * 128
    assert result["saturation_events"] == 128 * 128
    assert result["parameter_gradients"]["W_HEAD"][1] == -65024 * 128
    assert result["route_diagnostics"]["exact_zero_head_input_vjp_on_nonzero_gradient_positions"] == 128


@pytest.mark.parametrize("zshift", [0, 4])
@pytest.mark.parametrize("use_context", [False, True])
def test_forward_exactly_matches_frozen_reference_and_context_off_has_no_temporal_carry(
        shape, zshift, use_context):
    weights = initialized_weights(shape)
    tokens = [(position * 7) % 128 for position in range(128)]
    targets = [(position * 11 + 3) % 128 for position in range(128)]
    result = observe(weights, tokens, targets, shape, zshift=zshift, use_context=use_context)
    expected = pilot.integer_forward(weights, tokens, targets, shape,
                                     zshift=zshift, use_context=use_context)
    assert {key: result[key] for key in expected} == expected
    assert "parameter_gradients" not in result
    layers = result["route_diagnostics"]["projection_layers"]
    if not use_context:
        assert not any(result["native_routes"]["TAP_IDX"])
        assert all(row["nonzero_temporal_carry_positions"] == 0 for row in layers)
    else:
        # Ensure the native comparison's randomized fixture is not vacuous.
        assert any(row["nonzero_temporal_carry_positions"] > 0 for row in layers)
        assert any(row["nonzero_centering_remainder_positions"] > 0 for row in layers)


def test_int32_storage_low16_sign_extension_and_inputs_not_mutated(shape):
    weights = anchor_weights(shape)
    wrapped = {name: [value + 65536 for value in values] for name, values in weights.items()}
    original = copy.deepcopy(wrapped)
    assert observe_zeros(wrapped, shape) == observe_zeros(weights, shape)
    assert wrapped == original


@pytest.mark.parametrize("value", [0.5, float("nan"), True, -(1 << 31) - 1, 1 << 31,
                                   np.uint64(1 << 63)])
def test_noninteger_or_outside_int32_weights_fail_before_coercion(shape, value):
    weights = zero_weights(shape)
    # A bool-only array remains distinguishable from an ordinary integer list.
    weights["MB"] = [value] * len(weights["MB"])
    with pytest.raises(ValueError, match="exact int32 integers"):
        observe_zeros(weights, shape)


@pytest.mark.parametrize("zshift", [True, False, -1, 31, 4.0, "4"])
def test_invalid_gradient_scale_is_rejected(shape, zshift):
    with pytest.raises(ValueError, match="zshift"):
        observe_zeros(zero_weights(shape), shape, zshift=zshift)


@pytest.mark.parametrize("value", [-1, 128, 0.5])
@pytest.mark.parametrize("region", ["tokens", "targets"])
def test_invalid_token_rejected(shape, region, value):
    sequence = {"tokens": [0] * 128, "targets": [0] * 128}
    sequence[region][0] = value
    with pytest.raises(ValueError, match="token"):
        observe(zero_weights(shape), shape=shape, **sequence)


def test_partial_window_rejected(shape):
    with pytest.raises(ValueError, match="complete window"):
        observe(zero_weights(shape), [0] * 127, [0] * 128, shape)


def test_checked_backward_rejects_overflow_instead_of_publishing_diagnostics():
    # A deliberately deeper host-only toy (not a hardware configuration): every
    # projection is identity, so centered balanced cotangents double per layer.
    shape = SimpleNamespace(ML_D=16, ML_LOG2_D=4, ML_N_LAYERS=20, ML_K=1,
                            ML_V=128, ML_L_SEQ=1)
    weights = anchor_weights(shape)
    weights["MP"] = [0 if j == k else -8192
                     for _ in range(20) for j in range(16) for k in range(16)]
    with pytest.raises(OverflowError, match="int32 overflow"):
        observe(weights, [0], [0], shape)


@pytest.mark.kernel
def test_all_saved_routes_and_full_parameter_gradients_match_frozen_native(tmp_path):
    if not (shutil.which("clang++") or shutil.which("g++")):
        pytest.skip("no C++17 compiler")
    script = """
import ctypes
import sys
from pathlib import Path
from manuscript.tools.observe_fpga_lm_routes import observe
from scripts import fpga_lm_pretrain as p
shape = p.layout(16)
library = p.build_csim(Path(sys.argv[1]) / 'build', 16)
native = ctypes.CDLL(str(library)).forge_train_step_morph_lm
native.argtypes = [ctypes.POINTER(ctypes.c_int32)] + [ctypes.c_int] * 4
native.restype = ctypes.c_int
for recipe in ('initializer', 'wider', 'anchors'):
    for zshift in (0, 4):
        initial = shape.build_init(7)
        if recipe != 'initializer':
            head = [((v * 37 + j * 19) % 1025 - 512) if recipe == 'wider'
                    else (0 if j == v % 16 else -8192)
                    for v in range(128) for j in range(16)]
            initial[shape.OFF_W_HEAD:shape.OFF_W_HEAD + shape.WORDS_W_HEAD] = head
        initial[shape.OFF_CFG + 1] = zshift
        weights = {name: initial[getattr(shape, 'OFF_' + name):
                                getattr(shape, 'OFF_' + name) + getattr(shape, 'WORDS_' + name)]
                   for name in p.WEIGHTS}
        tokens = [(position * 7) % 128 for position in range(128)]
        targets = [(position * 11 + 3) % 128 for position in range(128)]
        expected = observe(weights, tokens, targets, shape, zshift=zshift,
                           include_parameter_gradients=True)
        buffer = (ctypes.c_int32 * len(initial))(*initial)
        # Modes 0/1 invoke the exact forward/backward used by mode 3, but let
        # us inspect G_* before mode 3's optimizer clears them even with lr=0.
        for position in range(128):
            assert native(buffer, 0, position, tokens[position], targets[position]) == 0
        outputs = dict(expected['native_routes'])
        outputs.update(PRED=expected['predictions'], GRAD_LOGIT=expected['gradients'])
        for name, values in outputs.items():
            offset, count = getattr(shape, 'OFF_' + name), getattr(shape, 'WORDS_' + name)
            assert list(buffer[offset:offset + count]) == values, (recipe, zshift, name)
        for name in ('BS_STREAM',) + tuple('G_' + name for name in p.WEIGHTS):
            offset, count = getattr(shape, 'OFF_' + name), getattr(shape, 'WORDS_' + name)
            for index in range(offset, offset + count):
                buffer[index] = 0
        for position in range(127, -1, -1):
            assert native(buffer, 1, position, tokens[position], 0) == 0
        for name, values in expected['parameter_gradients'].items():
            offset, count = getattr(shape, 'OFF_G_' + name), getattr(shape, 'WORDS_G_' + name)
            assert count == len(values) > 0
            assert list(buffer[offset:offset + count]) == values, (recipe, zshift, 'G_' + name)
        for name, values in weights.items():
            offset = getattr(shape, 'OFF_' + name)
            assert list(buffer[offset:offset + len(values)]) == values, 'diagnostic changed weights'
print('All six recipes/scales: complete routes, predictions, logit and parameter gradients match.')
"""
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                               cwd=pilot.REPO, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
