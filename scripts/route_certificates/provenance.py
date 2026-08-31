"""Provenance writer for the TikiTorch route-certificate experiments.

Every runner records the exact TikiTorch revision and dirty-patch digest,
resolved modules, environment, inputs and command in an append-only run
directory before publishing any claim-bearing summary.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "route_certificates" / "runs"


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return "unavailable"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=_sha256(b))


def detect_device(declared: str | None = None) -> str:
    """The DETECTED or DECLARED compute device (OA-CYC20 contract Phase 1:
    never hard-code 'cpu'). A declared value wins — detection cannot know
    which device the caller actually placed tensors on."""
    if declared is not None:
        return str(declared)
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return f"cuda:{torch.cuda.current_device()}"
        if getattr(torch.backends, "mps", None) is not None and \
                torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _module_record(mod, repos: dict[str, Path]) -> dict[str, Any]:
    """Record the resolved module path and owning repository revision."""
    f = Path(getattr(mod, "__file__", "") or "unknown").resolve()
    rec: dict[str, Any] = dict(module=mod.__name__, file=str(f))
    for name, repo in repos.items():
        try:
            inside = f.is_relative_to(repo.resolve())
        except (OSError, ValueError):
            inside = False
        if inside:
            rec["repo"] = name
            rec["commit"] = _git(["rev-parse", "HEAD"], repo)
            return rec
    top = mod.__name__.split(".")[0]
    if "site-packages" in str(f) and top == "tikitorch" and REPO.exists():
        raise RuntimeError(
            f"provenance: module {mod.__name__!r} imported from "
            f"site-packages ({f}) while a sibling checkout exists for it "
            f"— the repository SHA would misrepresent the code that ran. "
            f"Fix PYTHONPATH to this checkout (or remove the "
            f"site-packages install) before a scientific run.")
    rec["repo"] = None
    rec["commit"] = None
    return rec


def run_record(experiment: str, command: str,
               config: Mapping[str, Any],
               inputs: list[str | Path] | None = None,
               device: str | None = None,
               modules: list[Any] | None = None) -> dict[str, Any]:
    """Create the run directory + manifest skeleton; returns the record
    (caller appends metrics and calls finish()).

    device: declared device string; None ⇒ detect (never hard-coded).
    modules: imported modules to record (resolved path + repo SHA); None uses
    the imported ``tikitorch`` package when present."""
    run_id = str(uuid.uuid4())
    out_dir = RESULTS / experiment / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = "unavailable"
    try:
        import numpy
        np_v = numpy.__version__
    except Exception:
        np_v = "unavailable"
    repos = {"tikitorch": REPO}
    if modules is None:
        modules = [m for name in ("tikitorch",)
                   if (m := sys.modules.get(name)) is not None]
    module_recs = [_module_record(m, repos) for m in modules]
    rec: dict[str, Any] = dict(
        schema="tikitorch.route-certificates.run.v1",
        experiment=experiment,
        run_id=run_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        command=command,
        config=dict(config),
        tikitorch_commit=_git(["rev-parse", "HEAD"], REPO),
        tikitorch_dirty_patch_sha256=_sha256(
            _git(["diff", "HEAD"], REPO).encode()),
        host=platform.node(),
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch=torch_v,
        numpy=np_v,
        device=detect_device(device),
        modules=module_recs,
        inputs=[file_digest(p) for p in (inputs or [])],
        incident_status="running",
        _out_dir=str(out_dir),
        _t0=time.perf_counter(),
    )
    return rec


def finish(rec: dict[str, Any], metrics: Mapping[str, Any],
           status: str = "completed") -> Path:
    rec = dict(rec)
    t0 = rec.pop("_t0")
    out_dir = Path(rec.pop("_out_dir"))
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    rec["incident_status"] = status
    rec["metrics"] = dict(metrics)
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(rec, indent=1))
    return out
