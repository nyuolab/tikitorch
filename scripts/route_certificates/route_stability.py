"""Empirical active-set stability audit for fixed max-plus route traces."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import torch


def _git(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _routes_and_margins(x: torch.Tensor, weight: torch.Tensor):
    scores = x[:, None, :] + weight[None, :, :]
    ordered, indices = torch.sort(scores, dim=-1, descending=True, stable=True)
    return indices[..., 0], ordered[..., 0] - ordered[..., 1]


def _vjp(route: torch.Tensor, adjoint: torch.Tensor, input_width: int) -> torch.Tensor:
    result = torch.zeros(route.shape[0], input_width, dtype=adjoint.dtype)
    result.scatter_add_(1, route, adjoint)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--inputs", type=int, default=64)
    parser.add_argument("--outputs", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 0.5])
    parser.add_argument("--dtypes", choices=["float32", "float64"], nargs="+", default=["float32", "float64"])
    parser.add_argument(
        "--out", type=Path,
        default=Path("manuscript/results/route_certificates/reference/route_stability.jsonl"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    run_id = str(uuid.uuid4())
    base = {
        "schema": "route-sufficiency.route-stability.v1",
        "experiment_id": "RS-009",
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tikitorch_commit": _git(root),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "accelerator": "none",
        "checkpoint_sha256": "not_applicable_synthetic_problem",
        "configuration": vars(args) | {"out": str(args.out)},
    }
    rows = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for seed in args.seeds:
            generator = torch.Generator().manual_seed(seed + 2047)
            inputs = torch.randn(args.samples, args.batch, args.inputs, generator=generator, dtype=dtype)
            weights = torch.randn(args.samples, args.outputs, args.inputs, generator=generator, dtype=dtype)
            adjoints = torch.randn(args.samples, args.batch, args.outputs, generator=generator, dtype=dtype)
            for sigma in args.sigmas:
                perturbations = sigma * torch.randn(
                    args.samples, args.batch, args.inputs, generator=generator, dtype=dtype
                )
                flips = 0
                positions = 0
                no_flip_error_max = 0.0
                error_sum = 0.0
                certified_samples = 0
                certified_violations = 0
                margin_values = []
                for sample in range(args.samples):
                    route0, margin = _routes_and_margins(inputs[sample], weights[sample])
                    route1, _ = _routes_and_margins(
                        inputs[sample] + perturbations[sample], weights[sample]
                    )
                    changed = route0 != route1
                    flips += int(changed.sum())
                    positions += changed.numel()
                    margin_values.append(margin.reshape(-1))
                    frozen = _vjp(route0, adjoints[sample], args.inputs)
                    changed_vjp = _vjp(route1, adjoints[sample], args.inputs)
                    denom = max(float(torch.linalg.vector_norm(changed_vjp)), torch.finfo(dtype).tiny)
                    relative = float(torch.linalg.vector_norm(frozen - changed_vjp)) / denom
                    error_sum += relative
                    if not bool(changed.any()):
                        no_flip_error_max = max(no_flip_error_max, relative)
                    epsilon = float(perturbations[sample].abs().max())
                    certified = bool((2.0 * epsilon < margin).all())
                    if certified:
                        certified_samples += 1
                        certified_violations += int(bool(changed.any()))
                margins = torch.cat(margin_values)
                rows.append(
                    {
                        **base,
                        "record_type": "aggregate",
                        "seed": seed,
                        "dtype": dtype_name,
                        "sigma": sigma,
                        "initial_problem_sha256": _hash(inputs, weights, adjoints),
                        "data_digest": _hash(inputs, weights, adjoints, perturbations),
                        "metrics": {
                            "route_flip_fraction": flips / positions,
                            "mean_frozen_vs_changed_vjp_relative_l2": error_sum / args.samples,
                            "max_vjp_error_when_no_routes_flip": no_flip_error_max,
                            "certified_samples_2eps_below_all_margins": certified_samples,
                            "certified_route_change_violations": certified_violations,
                            "margin_min": float(margins.min()),
                            "margin_median": float(margins.median()),
                            "margin_p05": float(torch.quantile(margins, 0.05)),
                        },
                        "incident_status": "none",
                    }
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"run_id": run_id, "rows": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
