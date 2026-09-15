"""Independent integer route/gradient observer for the v1 head-repair screen.

This is a host diagnostic, not a new training kernel. It mirrors the frozen
kernel's selected-route, straight-through saturation surrogate and arithmetic
right shifts, NOT derivatives of integer rounding or clipping. Int32 overflow
is rejected rather than assigning portable semantics to C++ signed overflow.
"""
from __future__ import annotations

import math

import numpy as np


def observe(weights, tokens, targets, shape, zshift=4, use_context=True,
            include_parameter_gradients=False):
    """Observe a complete window, including reverse-time temporal cotangents.

    ``native_routes`` uses the frozen ABI's layer/position/feature ordering.
    Parameter gradients are optional because the D512 ABI keeps them on-chip;
    D16 tests can independently compare every gradient with C-simulation.
    """
    d, n, k, vocab, length = (shape.ML_D, shape.ML_N_LAYERS, shape.ML_K,
                             shape.ML_V, shape.ML_L_SEQ)
    if len(tokens) != length or len(targets) != length:
        raise ValueError("observer requires a complete window")
    if type(zshift) is not int or not 0 <= zshift <= 30:
        raise ValueError("zshift must be an integer in 0..30")
    if any(not isinstance(x, (int, np.integer)) or not 0 <= x < vocab
           for x in list(tokens) + list(targets)):
        raise ValueError("token outside vocabulary")
    dims = {"EMBED": (vocab, d), "MH": (n, d, k), "MP": (n, d, d),
            "MB": (n, d), "W_HEAD": (vocab, d)}
    arrays = {}
    for name, dimensions in dims.items():
        values = np.asarray(weights[name])
        if (values.dtype.kind not in "iu" or np.any(values < -(1 << 31))
                or np.any(values >= (1 << 31))):
            raise ValueError("observer weights must be exact int32 integers")
        values = values.astype(np.int64)
        arrays[name] = (((values + 32768) % 65536) - 32768).reshape(dimensions)
    history = np.zeros((n, length, d), dtype=np.int64)
    taps = np.zeros((n, length, d), dtype=np.int64)
    projections = np.zeros_like(taps)
    heads = np.zeros((length, vocab), dtype=np.int64)
    gradients = np.zeros((length, vocab), dtype=np.int64)
    head_vjps = np.zeros((length, d), dtype=np.int64)
    lut = np.asarray([round(65536 * 2.0 ** (-i / 16.0)) for i in range(256)], dtype=np.int64)
    predictions, losses, winner_counts = [], [], []
    saturation_events = rounded_zero_targets = 0

    def sat(values):
        nonlocal saturation_events
        saturation_events += int(np.count_nonzero((values < -32768) | (values > 32767)))
        return np.clip(values, -32768, 32767)

    def checked(values):
        if np.any(values < -(1 << 31)) or np.any(values >= (1 << 31)):
            raise OverflowError("observer cannot establish native equivalence with int32 overflow")
        return values

    def checked_sum(values):
        # Bound every intermediate accumulation, not only a cancelling total.
        positive = int(values[values > 0].sum())
        negative = int(values[values < 0].sum())
        checked(np.asarray([positive, negative]))
        return positive + negative

    def scatter(target, indices, values, unique=False):
        if unique:
            # Callers only use this path when an arange coordinate proves
            # indices distinct; there are then no hidden partial reductions.
            target[indices] = checked(target[indices] + values)
            return
        # Every prefix is bounded by positive/negative contributions. This may
        # conservatively reject a cancelling near-limit sequence; never bless it.
        positive = np.zeros_like(target)
        negative = np.zeros_like(target)
        np.add.at(positive, indices, np.maximum(values, 0))
        np.add.at(negative, indices, np.minimum(values, 0))
        checked(target + positive)
        checked(target + negative)
        target += positive + negative

    for position, token in enumerate(tokens):
        residual = arrays["EMBED"][token].copy()
        for layer in range(n):
            history[layer, position] = sat(residual - (int(residual.sum()) >> shape.ML_LOG2_D))
            candidates = np.stack([
                history[layer, position - (tap << min(layer, 6))] + arrays["MH"][layer, :, tap]
                for tap in range(k if use_context else 1)
                if position >= (tap << min(layer, 6))])
            selected = candidates.argmax(axis=0) if layer % 2 == 0 else candidates.argmin(axis=0)
            taps[layer, position] = selected
            conv = sat(candidates[selected, np.arange(d)])
            candidates = arrays["MP"][layer] + conv
            selected = candidates.argmax(axis=1)
            projections[layer, position] = selected
            projected = sat(candidates[np.arange(d), selected] + arrays["MB"][layer])
            residual = sat(residual + projected)
        scores = arrays["W_HEAD"] + residual
        selected = scores.argmax(axis=1)
        heads[position] = selected
        logits = sat(scores[np.arange(vocab), selected])
        prediction = int(logits.argmax())
        predictions.append(prediction)
        energies = lut[np.minimum((int(logits[prediction]) - logits) >> zshift, 255)]
        total = int(energies.sum())
        losses.append(math.log(total) - math.log(int(energies[targets[position]])))
        shift = 0
        while total >> shift > 65536:
            shift += 1
        gradient = energies >> shift
        rounded_zero_targets += int(gradient[targets[position]] == 0)
        gradient[targets[position]] -= gradient.sum()
        gradients[position] = gradient
        scatter(head_vjps[position], selected, gradient)
        winner_counts.append(int(np.unique(selected).size))

    parameter_grads = {name: np.zeros(dimensions, dtype=np.int64)
                       for name, dimensions in dims.items()}
    stream = np.zeros_like(history)
    embedding_vjps = np.zeros((length, d), dtype=np.int64)
    layer_rows = [{"layer": layer, "positions": length,
                   "single_projection_winner_positions": 0,
                   "nonzero_incoming_vjp_positions": 0,
                   "zero_projection_input_vjp_on_nonzero_incoming_positions": 0,
                   "nonzero_temporal_carry_positions": 0,
                   "nonzero_centering_remainder_positions": 0,
                   "max_abs_projection_input_vjp": 0,
                   "max_abs_layer_input_vjp": 0,
                   "projection_incoming_l1": 0, "projection_scattered_l1": 0}
                  for layer in range(n)]
    for position in range(length - 1, -1, -1):
        g = head_vjps[position].copy()
        scatter(parameter_grads["W_HEAD"], (np.arange(vocab), heads[position]), gradients[position], unique=True)
        for layer in range(n - 1, -1, -1):
            row = layer_rows[layer]
            selected = projections[layer, position]
            row["single_projection_winner_positions"] += int(np.unique(selected).size == 1)
            incoming_nonzero = bool(np.any(g))
            row["nonzero_incoming_vjp_positions"] += int(incoming_nonzero)
            parameter_grads["MB"][layer] = checked(parameter_grads["MB"][layer] + g)
            scatter(parameter_grads["MP"][layer], (np.arange(d), selected), g, unique=True)
            conv_gradient = np.zeros(d, dtype=np.int64)
            scatter(conv_gradient, selected, g)
            row["zero_projection_input_vjp_on_nonzero_incoming_positions"] += int(
                incoming_nonzero and not np.any(conv_gradient))
            row["projection_incoming_l1"] += int(np.abs(g).sum())
            row["projection_scattered_l1"] += int(np.abs(conv_gradient).sum())
            row["max_abs_projection_input_vjp"] = max(
                row["max_abs_projection_input_vjp"], int(np.abs(conv_gradient).max()))
            row["nonzero_temporal_carry_positions"] += int(np.any(stream[layer, position]))
            selected_taps = taps[layer, position]
            scatter(parameter_grads["MH"][layer], (np.arange(d), selected_taps), conv_gradient, unique=True)
            destination = position - (selected_taps << min(layer, 6))
            scatter(stream[layer], (destination, np.arange(d)), conv_gradient, unique=True)
            own = stream[layer, position]
            total = checked_sum(own)
            mean = total >> shape.ML_LOG2_D
            row["nonzero_centering_remainder_positions"] += int(total != mean * d)
            # Parenthesized HLS expression is gR += gbs - mean.
            g = checked(g + checked(own - mean))
            row["max_abs_layer_input_vjp"] = max(row["max_abs_layer_input_vjp"], int(np.abs(g).max()))
        embedding_vjps[position] = g
        parameter_grads["EMBED"][tokens[position]] = checked(parameter_grads["EMBED"][tokens[position]] + g)
    nonzero = np.any(gradients != 0, axis=1)
    zero_head = ~np.any(head_vjps != 0, axis=1)
    result = {
        "predictions": predictions, "gradients": gradients.ravel().tolist(),
        "lut_normalized_nll": sum(losses) / length,
        "accuracy": sum(a == b for a, b in zip(predictions, targets)) / length,
        "saturation_events": saturation_events, "rounded_zero_target_mass": rounded_zero_targets,
        "native_routes": {"AD_HEAD": heads.ravel().tolist(), "PROJ_IDX": projections.ravel().tolist(),
                          "TAP_IDX": taps.ravel().tolist()},
        "route_diagnostics": {
            "positions": length, "nonzero_logit_gradient_positions": int(nonzero.sum()),
            "exact_zero_head_input_vjp_on_nonzero_gradient_positions": int((nonzero & zero_head).sum()),
            "head_winners_ge16_positions": sum(count >= 16 for count in winner_counts),
            "single_shared_head_winner_positions": winner_counts.count(1),
            "head_winner_counts": winner_counts,
            "max_abs_head_input_vjp": int(np.abs(head_vjps).max()),
            "projection_layers": layer_rows,
            "nonzero_embedding_input_vjp_positions": int(np.any(embedding_vjps != 0, axis=1).sum()),
            "max_abs_embedding_input_vjp": int(np.abs(embedding_vjps).max()),
            "backward_int32_overflow_detected": False,
            "scope": "host selected-route surrogate; reverse-time carry and residual skips included; "
                     "forward clipping has straight-through backward, as in frozen kernel"}}
    if include_parameter_gradients:
        result["parameter_gradients"] = {name: values.ravel().tolist() for name, values in parameter_grads.items()}
    return result
