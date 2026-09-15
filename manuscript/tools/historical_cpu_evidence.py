"""Verify five selected CPU records against a pinned historical source snapshot.

No experiment, Torch import, subprocess, network access or current-source waiver.
Only the three pinned original stdlib checkers are loaded from a verified private
source tree. Their strict source checks remain enabled against that historical
tree. This is historical integrity, NOT freshness of the current library.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import runpy
import tarfile
import tempfile


REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "manuscript/research/source_snapshots/cpu_2026-09-10"
SCHEMA = "tikitorch.historical_cpu_sources.v1"
MANIFEST_SHA256 = "85dbc27660bb51eb09f1e17fdea0240b9d8e5efc97b1e41934f972d3dd42aba4"
RETAINED = "manuscript/evidence/retained_state_layout_2026-09-09.json"
ADAMW = "manuscript/evidence/adamw_diagnostic_benchmarks_2026-09-09.json"
WIDER = "manuscript/evidence/adamw_diagnostic_wider_2026-09-09.json"
PILOT = "manuscript/results/route_certificates/runs/device_route_training/172834f0-c1cf-42c1-ab87-ec0cdf6f0da9/manifest.json"
FAILURE = "manuscript/results/route_certificates/runs/compiled_selective_followup/907579f2-5b1c-42a5-93b0-0d1867e8577e/manifest.json"
RECORDS = {
    RETAINED: "2e536dd65bc89ac5d59046cb25fe27ccd4fcca1d2b7af50675a3629cc5f8cc4c",
    ADAMW: "4dacb4274f27058bec8cedd3ef929369b6d5fe9d4791c93f2f76a5bd0145d70a",
    WIDER: "f5b11559aefc3be4cfebcd3d312f522f5920f6ee3ddeacb53e71f8147e60f693",
    PILOT: "aad9d1c28ccd3d6c4fc9bf1326cf701ac38671d4af0a4afb6dbfb429cc279c1f",
    FAILURE: "cbadf238193a0c3e46f714e18d359b0c78c4d49f7e146506c55c877f0d768783",
}
TABLES = {
    "manuscript/source/generated/current_retained_state.tex": "07e50bfc1a0c0fecbbce3338f8e59752881c3ab93f99cfc034baef59e0bff99f",
    "manuscript/source/generated/training_pilot.tex": "f9b887220492cca4470178a7596a1f5ac251577bc5b6df1b87cb7694223f5c2e",
}
MEASUREMENT = "manuscript/tools/measure_retained_state.py"
OPTIMIZER = "scripts/route_certificates/optimizer_diagnostic.py"
SUMMARIZER = "manuscript/tools/summarize_training_pilot.py"
SUMMARIZER_SHA256 = "b0b4123641853f48e0e69649244a6acbe66865d8ef8cc11372f2dac634c85531"
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(data):
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, parse_constant=invalid)


def canonical_path(name):
    require(isinstance(name, str) and bool(name), "invalid snapshot path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and ".." not in path.parts
            and "\\" not in name and ":" not in name, "unsafe snapshot path")
    return path


def pinned_artifacts(root=REPO):
    result = {}
    for name, expected in {**RECORDS, **TABLES}.items():
        path = root / name
        require(not path.is_symlink() and path.is_file(), "selected artifact must be regular")
        data = path.read_bytes()
        require(len(data) <= 16 * 1024 * 1024 and digest(data) == expected,
                "selected historical artifact changed: " + name)
        result[name] = data
    return result


def required_sources(artifacts):
    """Recover the recorded source union, not the mutable current library glob."""
    result = {}
    def merge(mapping):
        for name, value in mapping.items():
            canonical_path(name)
            require(name.endswith(".py") and isinstance(value, str) and len(value) == 64,
                    "invalid recorded Python source")
            require(name not in result or result[name] == value, "historical source versions disagree")
            result[name] = value
    retained = read_json(artifacts[RETAINED])
    require(retained["sources_before_sha256"] == retained["sources_after_sha256"], "retained source changed during run")
    merge(retained["sources_before_sha256"])
    for name in (ADAMW, WIDER):
        merge(read_json(artifacts[name])["source"]["files"])
    for name in (PILOT, FAILURE):
        rows = read_json(artifacts[name])["inputs"]
        suffix = "scripts/route_certificates/training_benchmark.py"
        runners = [PurePosixPath(row["path"]) for row in rows if row["path"].endswith("/" + suffix)]
        require(len(runners) == 1, "ambiguous historical checkout")
        original = runners[0].parents[2]
        mapping = {}
        for row in rows:
            path = PurePosixPath(row["path"])
            require(path.is_absolute() and ".." not in path.parts and "\\" not in row["path"], "unsafe original source path")
            relative = path.relative_to(original).as_posix()
            require(relative not in mapping, "duplicate historical source")
            mapping[relative] = row["sha256"]
        merge(mapping)
    merge({SUMMARIZER: SUMMARIZER_SHA256})
    require(len(result) == 71 and len([name for name in result if name.startswith("tikitorch/")]) == 65,
            "selected source union changed")
    require("tikitorch/optim/fixed_point.py" not in result, "new implementation is not historical evidence")
    return result


def _capture():
    """One-time maintainer capture; never overwrite or select a new record."""
    require(not SNAPSHOT.exists(), "snapshot destination must be new")
    artifacts = pinned_artifacts()
    expected = required_sources(artifacts)
    sources = {}
    for name, value in expected.items():
        path = REPO / name
        require(not path.is_symlink(), "source symlink")
        data = path.read_bytes()
        require(digest(data) == value and len(data) <= MAX_SOURCE_BYTES, "capture source differs: " + name)
        sources[name] = data
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in sorted(sources.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    compressed = gzip.compress(raw.getvalue(), mtime=0)
    require(len(compressed) <= MAX_ARCHIVE_BYTES, "snapshot archive too large")
    manifest = {"schema": SCHEMA, "scope": "historical CPU source integrity; not current-library freshness or scientific reexecution",
                "archive": {"file": "sources.tar.gz", "bytes": len(compressed), "sha256": digest(compressed)},
                "sources": {name: {"bytes": len(data), "sha256": digest(data)} for name, data in sorted(sources.items())},
                "records": RECORDS, "tables": TABLES}
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    # Recheck every input immediately before preserving the new immutable archive.
    require(pinned_artifacts() == artifacts, "artifacts changed during capture")
    require(all((REPO / name).read_bytes() == data for name, data in sources.items()), "source changed during capture")
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    with (SNAPSHOT / "sources.tar.gz").open("xb") as handle:
        handle.write(compressed)
    with (SNAPSHOT / "manifest.json").open("xb") as handle:
        handle.write(encoded)
    return {"snapshot_manifest_sha256": digest(encoded), "source_files": len(sources), "archive_bytes": len(compressed)}


def snapshot_sources():
    """Validate the pinned archive without extraction or code execution."""
    manifest_path = SNAPSHOT / "manifest.json"
    require(not manifest_path.is_symlink(), "snapshot manifest symlink")
    encoded = manifest_path.read_bytes()
    require(digest(encoded) == MANIFEST_SHA256, "snapshot manifest digest mismatch")
    manifest = read_json(encoded)
    require(manifest["schema"] == SCHEMA and manifest["records"] == RECORDS and manifest["tables"] == TABLES,
            "snapshot selection changed")
    artifacts = pinned_artifacts()
    required = required_sources(artifacts)
    require(set(manifest["sources"]) == set(required), "snapshot source membership differs")
    item = manifest["archive"]
    require(item["file"] == "sources.tar.gz", "unexpected archive filename")
    path = SNAPSHOT / item["file"]
    require(not path.is_symlink() and path.stat().st_size <= MAX_ARCHIVE_BYTES, "unsafe/oversized archive")
    blob = path.read_bytes()
    require(type(item["bytes"]) is int and len(blob) == item["bytes"] and digest(blob) == item["sha256"], "archive digest/size mismatch")
    result, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive:
            name = canonical_path(member.name).as_posix()
            require(name in required and name not in result, "unknown/duplicate archive member")
            require(member.isreg() and not member.pax_headers and 0 <= member.size <= MAX_SOURCE_BYTES,
                    "only bounded regular source files are allowed")
            total += member.size
            require(total <= MAX_TOTAL_BYTES, "source archive payload too large")
            handle = archive.extractfile(member)
            require(handle is not None, "missing source payload")
            data = handle.read(MAX_SOURCE_BYTES + 1)
            require(len(data) == member.size, "incomplete source payload")
            require(manifest["sources"][name] == {"bytes": len(data), "sha256": digest(data)}
                    and digest(data) == required[name], "source payload digest differs")
            result[name] = data
    require(set(result) == set(required), "incomplete source archive")
    return result, artifacts


@contextmanager
def verified_workspace():
    """Materialize only verified regular files; never use tar.extract/extractall."""
    sources, artifacts = snapshot_sources()
    with tempfile.TemporaryDirectory(prefix="tikitorch-historical-cpu-") as temporary:
        root = Path(temporary)
        for name, data in {**sources, **artifacts}.items():
            path = root.joinpath(*canonical_path(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
        yield root


def check():
    # These exact original scripts import only the standard library at load time.
    # No arbitrary checker path or archive may be provided through the CLI.
    with verified_workspace() as root:
        measurement = runpy.run_path(str(root / MEASUREMENT))
        record = read_json((root / RETAINED).read_bytes())
        measurement["validate_record"](record)  # strict source check stays ON
        require(measurement["render_table"](record).encode() == (root / next(iter(TABLES))).read_bytes(),
                "historical retained-state table differs")
        optimizer = runpy.run_path(str(root / OPTIMIZER))
        optimizer["check_record"](root / ADAMW)
        optimizer["check_record"](root / WIDER)
        summarizer = runpy.run_path(str(root / SUMMARIZER))
        table = summarizer["generate"]()
        require(table.encode() == (root / "manuscript/source/generated/training_pilot.tex").read_bytes(),
                "historical training-pilot table differs")
    # A concurrent repository edit must not turn a completed check into a claim
    # about different selected bytes. No numerical work is repeated here.
    snapshot_sources()
    return {"verified": True, "verification_scope": "pinned historical CPU source and artifact integrity",
            "current_library_freshness_claimed": False, "scientific_reexecution": False,
            "selected_records_verified": len(RECORDS), "source_files_verified": 71,
            "snapshot_manifest_sha256": MANIFEST_SHA256}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--capture", action="store_true", help="one-time new pinned snapshot; refuses existing destination")
    config = parser.parse_args(argv)
    result = _capture() if config.capture else check()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
