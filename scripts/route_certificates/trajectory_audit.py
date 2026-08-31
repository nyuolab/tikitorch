"""Small, matched first-divergence audit for the legacy 128-layer comparison.

This does not rerun the eight-epoch MNIST quality experiment. It holds the
DyadicPCNet forward graph fixed at the reported depth and asks where framework
autograd, an exact cached-route VJP, and the legacy quantized cascade first cease
to be the same computation.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from tikitorch.nn import dyadic_quantize
from tikitorch.nn.predictive_coding import DyadicPCNet


SCHEMA = "route-sufficiency.trajectory-audit.v1"


def _git(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _digest_tensors(items: Iterable[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for tensor in items:
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _forward(net: DyadicPCNet, x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of DyadicPCNet's no-grad feedforward."""
    h = x
    for layer in range(net.depth):
        h = net.f(layer, h)
    return h


@torch.no_grad()
def _route_grads(
    net: DyadicPCNet,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    reduction: str,
    quantized_outer: bool,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor | None, bool]]]:
    """Set cached-route gradients with an explicit reduction/outer-product rule."""
    acts, cache = net._forward_cache(x)  # audited legacy API
    target = F.one_hot(y, num_classes=net.dims[-1]).to(dtype=x.dtype)
    error = torch.softmax(acts[-1], dim=-1) - target
    if reduction == "mean":
        error = error / x.shape[0]
    elif reduction != "sum":
        raise ValueError(reduction)

    quantize = dyadic_quantize if quantized_outer else (lambda value: value)
    for layer in reversed(range(net.depth)):
        relu_mask, skip_gate, has_skip = cache[layer]
        relu_error = error * skip_gate if has_skip else error
        delta = relu_error * relu_mask
        net.layers[layer].weight.grad = (
            quantize(delta).transpose(0, 1) @ quantize(acts[layer])
        )
        net.layers[layer].bias.grad = quantize(delta).sum(0)
        if layer > 0:
            lower = delta @ dyadic_quantize(net.layers[layer].weight)
            if has_skip:
                lower = lower + error * (1.0 - skip_gate)
            error = lower
    return acts[-1], cache


def _grads(net: DyadicPCNet) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in net.named_parameters()
        if parameter.grad is not None
    }


def _flat(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([values[name].reshape(-1) for name in sorted(values)])


def _comparison(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict:
    a, b = _flat(reference), _flat(candidate)
    denom = max(float(torch.linalg.vector_norm(a)), torch.finfo(a.dtype).tiny)
    rel = float(torch.linalg.vector_norm(b - a)) / denom
    if float(torch.linalg.vector_norm(a)) == 0.0 or float(torch.linalg.vector_norm(b)) == 0.0:
        cosine = 1.0 if torch.equal(a, b) else 0.0
    else:
        cosine = float(F.cosine_similarity(a, b, dim=0))
    first = None
    layerwise = []
    for name in sorted(reference):
        ra, rb = reference[name], candidate[name]
        scale = max(float(torch.linalg.vector_norm(ra)), torch.finfo(ra.dtype).tiny)
        error = float(torch.linalg.vector_norm(rb - ra)) / scale
        layerwise.append({"parameter": name, "relative_l2": error})
        tolerance = 1e-10 if ra.dtype == torch.float64 else 1e-5
        if first is None and error > tolerance:
            first = name
    return {
        "relative_l2": rel,
        "cosine": cosine,
        "first_parameter_over_tolerance": first,
        "max_layer_relative_l2": max(item["relative_l2"] for item in layerwise),
    }


def _route_signature(cache) -> list[torch.Tensor]:
    signature: list[torch.Tensor] = []
    for relu_mask, skip_gate, _ in cache:
        signature.append(relu_mask.to(torch.uint8))
        if skip_gate is not None:
            signature.append(skip_gate.to(torch.uint8))
    return signature


def _route_agreement(left, right) -> float:
    a, b = _route_signature(left), _route_signature(right)
    total = sum(t.numel() for t in a)
    same = sum(int((x == y).sum()) for x, y in zip(a, b))
    return same / total if total else 1.0


def _parameter_distance(left: DyadicPCNet, right: DyadicPCNet) -> float:
    a = torch.cat([p.detach().reshape(-1) for p in left.parameters()])
    b = torch.cat([p.detach().reshape(-1) for p in right.parameters()])
    return float(torch.linalg.vector_norm(a - b))


def run_configuration(args, seed: int, dtype: torch.dtype, common: dict) -> list[dict]:
    torch.manual_seed(seed)
    dims = [args.input] + [args.width] * (args.depth - 1) + [args.classes]
    base = DyadicPCNet(dims, residual=True).to(dtype=dtype)
    initial_digest = _digest_tensors(base.state_dict().values())
    auto = copy.deepcopy(base)
    direct = copy.deepcopy(base)
    legacy = copy.deepcopy(base)
    optimizers = {
        "autograd": torch.optim.Adam(auto.parameters(), lr=args.lr),
        "direct": torch.optim.Adam(direct.parameters(), lr=args.lr),
        "legacy": torch.optim.Adam(legacy.parameters(), lr=args.lr),
    }
    generator = torch.Generator().manual_seed(seed + 1009)
    batches = [
        (
            torch.randn(args.batch, args.input, generator=generator, dtype=dtype),
            torch.randint(args.classes, (args.batch,), generator=generator),
        )
        for _ in range(args.steps)
    ]
    data_digest = _digest_tensors(t for pair in batches for t in pair)
    rows: list[dict] = []

    for step, (x, y) in enumerate(batches):
        optimizers["autograd"].zero_grad(set_to_none=True)
        logits_auto = _forward(auto, x)
        loss_auto = F.cross_entropy(logits_auto, y, reduction="mean")
        loss_auto.backward()
        grad_auto = _grads(auto)
        with torch.no_grad():
            _, cache_auto = auto._forward_cache(x)

        optimizers["direct"].zero_grad(set_to_none=True)
        logits_direct, cache_direct = _route_grads(
            direct, x, y, reduction="mean", quantized_outer=False
        )
        grad_direct = _grads(direct)

        optimizers["legacy"].zero_grad(set_to_none=True)
        logits_legacy, cache_legacy = _route_grads(
            legacy, x, y, reduction="sum", quantized_outer=True
        )
        grad_legacy = _grads(legacy)

        # Decompose the legacy discrepancy on an unchanged clone: CE reduction
        # alone is a factor of batch; quantization is the remaining distortion.
        decomposition = copy.deepcopy(auto)
        _route_grads(decomposition, x, y, reduction="sum", quantized_outer=False)
        grad_exact_sum = _grads(decomposition)
        scaled_mean = {name: value * args.batch for name, value in grad_auto.items()}

        record = {
            **common,
            "record_type": "step",
            "seed": seed,
            "dtype": str(dtype).replace("torch.", ""),
            "configuration": {
                "depth": args.depth,
                "width": args.width,
                "input": args.input,
                "classes": args.classes,
                "batch": args.batch,
                "lr": args.lr,
                "optimizer": "Adam",
                "residual": "tropical_max",
                "step": step,
            },
            "data_digest": data_digest,
            "initial_state_digest": initial_digest,
            "metrics": {
                "loss_autograd_mean": float(loss_auto.detach()),
                "logit_max_abs_direct_vs_autograd": float((logits_direct - logits_auto.detach()).abs().max()),
                "logit_max_abs_legacy_vs_autograd": float((logits_legacy - logits_auto.detach()).abs().max()),
                "route_agreement_direct_vs_autograd": _route_agreement(cache_direct, cache_auto),
                "route_agreement_legacy_vs_autograd": _route_agreement(cache_legacy, cache_auto),
                "direct_vs_autograd_grad": _comparison(grad_auto, grad_direct),
                "exact_sum_vs_batch_times_mean_grad": _comparison(scaled_mean, grad_exact_sum),
                "legacy_quantized_sum_vs_exact_sum_grad": _comparison(grad_exact_sum, grad_legacy),
                "legacy_vs_autograd_grad": _comparison(grad_auto, grad_legacy),
            },
            "incident_status": "none",
        }
        rows.append(record)

        for optimizer in optimizers.values():
            optimizer.step()
        record["metrics"]["post_update_parameter_l2_direct_vs_autograd"] = _parameter_distance(direct, auto)
        record["metrics"]["post_update_parameter_l2_legacy_vs_autograd"] = _parameter_distance(legacy, auto)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--input", type=int, default=16)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dtypes", choices=["float32", "float64"], nargs="+", default=["float32", "float64"])
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/route_certificates/reference/trajectory_audit.jsonl"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_id = str(uuid.uuid4())
    root = Path(__file__).resolve().parents[2]
    common = {
        "schema": SCHEMA,
        "experiment_id": "RS-008",
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tikitorch_commit": _git(root),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tikitorch_path": str(Path(sys.modules["tikitorch"].__file__).resolve()),
        "device": "cpu",
        "accelerator": "none",
        "checkpoint_sha256": "not_applicable_synthetic_problem",
        "deterministic": True,
    }
    rows: list[dict] = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for seed in args.seeds:
            rows.extend(run_configuration(args, seed, dtype, common))
    rows.append(
        {
            **common,
            "record_type": "source_audit",
            "legacy_result": {"cascade": 0.9382, "bp_dyadic": 0.8660, "bp_float": 0.2412},
            "findings": [
                "cascade and bp_dyadic share a dyadic forward family and tropical-max residual, but use quantized-surrogate versus exact outer-product parameter gradients",
                "cascade uses an unreduced softmax-minus-target adjoint whereas framework cross_entropy defaults to a batch mean",
                "bp_float uses ordinary float Linear layers and an additive residual, so its forward graph differs before differentiation",
            ],
            "incident_status": "legacy accuracy logs contain final metrics only; this audit uses deterministic synthetic batches at the same depth",
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"run_id": run_id, "rows": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
