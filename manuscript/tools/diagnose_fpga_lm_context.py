"""Frozen-checkpoint input sensitivity; no training or physical FPGA execution.

Target windows remain fixed. Reversing/replacing inputs is a counterfactual
sensitivity probe, not a valid held-out language-quality benchmark. Equality of
predictions or rounded loss gradients does not establish equality of logits.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts import fpga_lm_pretrain as pilot  # noqa: E402


def transformed(tokens, variant):
    if variant in ("original", "context_disabled"):
        return list(tokens)
    if variant == "all_space":
        return [32] * len(tokens)
    if variant == "reversed_tokens":
        return list(reversed(tokens))
    raise ValueError("unknown input transformation")


def array_digest(values):
    return pilot.digest(struct.pack(f"<{len(values)}i", *values))


def compare(original, candidate):
    pred_a, pred_b = original["predictions"], candidate["predictions"]
    grad_a, grad_b = original["gradients"], candidate["gradients"]
    if not pred_a or not grad_a or len(pred_a) != len(pred_b) or len(grad_a) != len(grad_b):
        raise ValueError("comparison requires complete nonempty matching output shapes")
    return {"prediction_mismatches": sum(a != b for a, b in zip(pred_a, pred_b)),
            "rounded_gradient_mismatches": sum(a != b for a, b in zip(grad_a, grad_b)),
            "rounded_gradient_max_abs_difference": max(abs(a - b) for a, b in zip(grad_a, grad_b)),
            "lut_nll_difference": candidate["lut_normalized_nll"] - original["lut_normalized_nll"]}


def describe_weights(weights, shape):
    import numpy as np

    result = {}
    for name, values in weights.items():
        array = np.asarray(values, dtype=np.int32)
        slices = {name: array}
        if name in ("MH", "MP", "MB"):
            slices.update({f"{name}_layer_{i}": row for i, row in
                           enumerate(array.reshape(shape.ML_N_LAYERS, -1))})
        for label, row in slices.items():
            result[label] = {"count": int(row.size), "min": int(row.min()), "max": int(row.max()),
                             "mean": float(row.mean()), "unique_values": int(np.unique(row).size),
                             "negative": int((row < 0).sum()), "zero": int((row == 0).sum()),
                             "positive": int((row > 0).sum()),
                             "at_int16_limit": int(((row == -32768) | (row == 32767)).sum())}
    return result


def head_route_summary(weights, tokens, targets, shape, original):
    """Independent integer forward observer; frozen runner remains unchanged."""
    import numpy as np

    d, n, k = shape.ML_D, shape.ML_N_LAYERS, shape.ML_K
    dims = {"EMBED": (128, d), "MH": (n, d, k), "MP": (n, d, d),
            "MB": (n, d), "W_HEAD": (128, d)}
    arrays = {name: (((np.asarray(values, dtype=np.int64) + 32768) % 65536) - 32768).reshape(dims[name])
              for name, values in weights.items()}
    history = np.zeros((n, len(tokens), d), dtype=np.int64)
    unique_routes, zero_vjps = [], 0
    gradients = np.asarray(original["gradients"], dtype=np.int64).reshape(len(tokens), 128)
    lut = np.asarray([round(65536 * 2.0 ** (-i / 16.0)) for i in range(256)], dtype=np.int64)
    maximum_vjp = 0
    for position, token in enumerate(tokens):
        residual = arrays["EMBED"][token].copy()
        for layer in range(n):
            history[layer, position] = np.clip(residual - (int(residual.sum()) >> shape.ML_LOG2_D), -32768, 32767)
            candidates = np.stack([history[layer, position - (tap << min(layer, 6))] + arrays["MH"][layer, :, tap]
                                   for tap in range(k) if position >= (tap << min(layer, 6))])
            conv = np.clip(candidates.max(axis=0) if layer % 2 == 0 else candidates.min(axis=0), -32768, 32767)
            projection = np.clip((arrays["MP"][layer] + conv).max(axis=1) + arrays["MB"][layer], -32768, 32767)
            residual = np.clip(residual + projection, -32768, 32767)
        scores = arrays["W_HEAD"] + residual
        winners = scores.argmax(axis=1)  # first-index ties, matching HLS strict '>'
        logits = np.clip(scores.max(axis=1), -32768, 32767)
        if int(logits.argmax()) != original["predictions"][position]:
            raise ValueError("independent head observer prediction mismatch")
        energies = lut[np.minimum((int(logits.max()) - logits) >> 4, 255)]
        shift, total = 0, int(energies.sum())
        while total >> shift > 65536:
            shift += 1
        observer_gradient = energies >> shift
        observer_gradient[targets[position]] -= observer_gradient.sum()
        if not np.array_equal(observer_gradient, gradients[position]):
            raise ValueError("independent head observer normalized LUT gradient mismatch")
        unique_routes.append(int(np.unique(winners).size))
        hidden_vjp = np.zeros(d, dtype=np.int64)
        np.add.at(hidden_vjp, winners, gradients[position])
        zero_vjps += int(not np.any(hidden_vjp))
        maximum_vjp = max(maximum_vjp, int(np.abs(hidden_vjp).max()))
    return {"positions": len(tokens), "single_shared_head_winner_positions": unique_routes.count(1),
            "unique_head_winners_min": min(unique_routes), "unique_head_winners_max": max(unique_routes),
            "exact_zero_head_input_vjp_positions": zero_vjps, "max_abs_head_input_vjp": maximum_vjp,
            "scope": "host integer observer, original validation inputs, final frozen checkpoint only"}


def diagnose(run_directory):
    root = Path(run_directory)
    if any((root / name).exists() for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("cannot diagnose a failed run as a complete checkpoint")
    manifest = pilot._finite_json((root / "manifest.json").read_text())
    summary = pilot._finite_json((root / "summary.json").read_text())
    if (manifest.get("schema") != "tikitorch.fpga_lm_pretrain.v1"
            or summary.get("complete") is not True
            or manifest["source_hashes"] != pilot.source_identity()):
        raise ValueError("requires a complete run with matching frozen sources")
    shape = pilot.layout(manifest["layout"][1])
    if pilot.layout_values(shape) != manifest["layout"]:
        raise ValueError("unsupported checkpoint layout")
    checkpoint = summary["final_checkpoint"]
    path = Path(checkpoint["file"])
    if not path.is_absolute():
        path = root / path
    payload = path.read_bytes()
    if len(payload) != checkpoint["bytes"] or pilot.digest(payload) != checkpoint["sha256"]:
        raise ValueError("checkpoint payload digest mismatch")
    flat = struct.unpack(f"<{len(payload) // 4}i", payload)
    weights, offset = {}, 0
    for name in pilot.WEIGHTS:
        count = getattr(shape, "WORDS_" + name)
        weights[name] = flat[offset:offset + count]
        offset += count
    if offset != len(flat):
        raise ValueError("checkpoint size does not match weight layout")
    count = manifest["evaluation"]["windows"]
    raw, dataset = pilot.corpus_plan(manifest["dataset"]["path"], manifest["steps"], count)
    if dataset != manifest["dataset"]:
        raise ValueError("corpus or partitions changed")
    variants = ("original", "context_disabled", "all_space", "reversed_tokens")
    rows, head_rows = [], []
    for index in range(count):
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        original = None
        for variant in variants:
            result = pilot.integer_forward(weights, transformed(tokens, variant), targets, shape,
                                           use_context=variant != "context_disabled")
            if original is None:
                original = result
                head_rows.append({"window": index, **head_route_summary(weights, tokens, targets, shape, original)})
            rows.append({"window": index, "variant": variant,
                         "input_sha256": pilot.digest(bytes(transformed(tokens, variant))),
                         "target_sha256": pilot.digest(bytes(targets)),
                         "prediction_sha256": array_digest(result["predictions"]),
                         "rounded_gradient_sha256": array_digest(result["gradients"]),
                         "accuracy": result["accuracy"],
                         "lut_normalized_nll": result["lut_normalized_nll"],
                         "comparison_to_original": compare(original, result)})
    totals = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        totals[variant] = {
            "prediction_mismatches": sum(row["comparison_to_original"]["prediction_mismatches"] for row in selected),
            "rounded_gradient_mismatches": sum(row["comparison_to_original"]["rounded_gradient_mismatches"] for row in selected),
            "lut_normalized_nll": sum(row["lut_normalized_nll"] for row in selected) / len(selected),
            "accuracy": sum(row["accuracy"] for row in selected) / len(selected)}
    return {"schema": "tikitorch.fpga_lm_context_diagnostic.v1", "training_performed": False,
            "physical_fpga_execution": False, "test_partition_used": False,
            "checkpoint": checkpoint, "dataset_sha256": dataset["sha256"],
            "validation_windows": count, "positions_per_variant": count * shape.ML_L_SEQ,
            "targets": "unchanged first frozen validation windows for every variant",
            "source_hashes": {**manifest["source_hashes"],
                              Path(__file__).resolve().relative_to(REPO).as_posix():
                              pilot.digest(Path(__file__).read_bytes())},
            "limitations": ["Only tested predictions, rounded gradients and LUT-normalized NLL are compared.",
                            "Equal rounded outputs do not prove equal raw logits or zero input dependence.",
                            "Reversed/all-space inputs are sensitivity controls, not a language benchmark."],
            "totals": totals, "windows": rows, "head_route_observations": head_rows,
            "weight_statistics": describe_weights(weights, shape)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = diagnose(args.run)
    pilot.write_json(args.output, result)
    print(json.dumps(result["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
