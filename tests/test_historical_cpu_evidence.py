"""Pinned historical integrity, never execution or a current-source waiver."""
import builtins
import copy
import gzip
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile

import pytest

from manuscript.tools import historical_cpu_evidence as historical


def test_snapshot_covers_exact_selected_old_source_union_not_new_optimizer():
    sources, artifacts = historical.snapshot_sources()
    assert len(sources) == 71 and len(artifacts) == 7
    assert sum(name.startswith("tikitorch/") for name in sources) == 65
    assert "tikitorch/optim/__init__.py" in sources
    assert "tikitorch/optim/fixed_point.py" not in sources
    assert historical.MEASUREMENT in sources and historical.OPTIMIZER in sources
    assert historical.SUMMARIZER in sources
    assert set(historical.required_sources(artifacts)) == set(sources)


def test_all_five_historical_records_check_without_torch_process_network_or_tar_extraction(monkeypatch):
    imported = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "tikitorch"):
            pytest.fail("historical verification must not import the model runtime")
        return imported(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("historical verification attempted subprocess or unsafe extraction")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    result = historical.check()
    assert result["verified"] is True and result["selected_records_verified"] == 5
    assert result["current_library_freshness_claimed"] is False
    assert result["scientific_reexecution"] is False


def test_cli_is_stdlib_only_and_explicitly_historical():
    result = subprocess.run([sys.executable, "-S", str(Path(historical.__file__)), "--check"],
                            capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    assert record["verification_scope"] == "pinned historical CPU source and artifact integrity"
    assert record["current_library_freshness_claimed"] is False


@pytest.mark.parametrize("change", ["addition", "edit"])
def test_original_current_source_checks_remain_strict_against_added_or_changed_library(change):
    with historical.verified_workspace() as root:
        retained = runpy.run_path(str(root / historical.MEASUREMENT))
        optimizer = runpy.run_path(str(root / historical.OPTIMIZER))
        pilot = runpy.run_path(str(root / historical.SUMMARIZER))
        record = json.loads((root / historical.RETAINED).read_text())
        retained["validate_record"](record)
        optimizer["check_record"](root / historical.ADAMW)
        pilot["generate"]()
        path = root / ("tikitorch/optim/fixed_point.py" if change == "addition" else "tikitorch/optim/__init__.py")
        path.write_text("# New code is not old evidence.\n")
        for check in (lambda: retained["validate_record"](record),
                      lambda: optimizer["check_record"](root / historical.ADAMW),
                      lambda: pilot["generate"]()):
            with pytest.raises(ValueError):
                check()


def test_only_three_pinned_checkers_are_loaded_and_temporary_tree_is_removed(monkeypatch):
    original = runpy.run_path
    loaded = []

    def record(path, *args, **kwargs):
        loaded.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runpy, "run_path", record)
    historical.check()
    assert len(loaded) == 3
    assert [path.name for path in loaded] == ["measure_retained_state.py", "optimizer_diagnostic.py", "summarize_training_pilot.py"]
    assert all(not path.exists() for path in loaded)


def test_capture_refuses_existing_snapshot_without_modifying_any_bytes():
    before = {path.name: path.read_bytes() for path in historical.SNAPSHOT.iterdir()}
    with pytest.raises(ValueError, match="must be new"):
        historical._capture()
    assert {path.name: path.read_bytes() for path in historical.SNAPSHOT.iterdir()} == before


@pytest.fixture
def snapshot_copy(tmp_path, monkeypatch):
    directory = tmp_path / "snapshot"
    directory.mkdir()
    for path in historical.SNAPSHOT.iterdir():
        (directory / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(historical, "SNAPSHOT", directory)
    return directory


@pytest.mark.parametrize("target", ["manifest", "archive"])
def test_snapshot_byte_mutation_is_rejected_before_any_checker_execution(snapshot_copy, monkeypatch, target):
    path = snapshot_copy / ("manifest.json" if target == "manifest" else "sources.tar.gz")
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(runpy, "run_path", lambda *a, **k: pytest.fail("loaded changed source"))
    with pytest.raises(ValueError):
        historical.check()


def test_selected_record_and_table_mutation_is_rejected_before_loading_sources(tmp_path, monkeypatch):
    for name in historical.RECORDS | historical.TABLES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((historical.REPO / name).read_bytes())
    record = tmp_path / historical.ADAMW
    record.write_bytes(record.read_bytes() + b" ")
    with pytest.raises(ValueError, match="historical artifact changed"):
        historical.pinned_artifacts(tmp_path)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown", "traversal", "absolute", "symlink", "hardlink", "directory", "oversized", "payload"])
def test_archive_members_are_complete_regular_bounded_and_hash_checked(snapshot_copy, monkeypatch, mutation):
    archive_path = snapshot_copy / "sources.tar.gz"
    original = archive_path.read_bytes()
    raw = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(original), mode="r:gz") as old:
        members = [(copy.copy(member), old.extractfile(member).read()) for member in old]
    member, data = members[0]
    if mutation == "duplicate":
        members.append((copy.copy(member), data))
    elif mutation == "missing":
        members.pop()
    elif mutation in ("unknown", "traversal", "absolute"):
        member.name = {"unknown": "unlisted.py", "traversal": "../escaped.py", "absolute": "/escaped.py"}[mutation]
    elif mutation in ("symlink", "hardlink", "directory"):
        member.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "directory": tarfile.DIRTYPE}[mutation]
        member.linkname = "outside"
        member.size = 0
        members[0] = (member, b"")
    elif mutation == "oversized":
        data = b"x" * (historical.MAX_SOURCE_BYTES + 1)
        member.size = len(data)
        members[0] = (member, data)
    else:
        changed = b"x" + data[1:]
        members[0] = (member, changed)
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for member, data in members:
            archive.addfile(member, io.BytesIO(data))
    blob = gzip.compress(raw.getvalue(), mtime=0)
    archive_path.write_bytes(blob)
    manifest_path = snapshot_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["archive"].update(bytes=len(blob), sha256=historical.digest(blob))
    encoded = json.dumps(manifest).encode()
    manifest_path.write_bytes(encoded)
    # Bypass only the outer fixture pin to exercise the archive parser's deeper checks.
    monkeypatch.setattr(historical, "MANIFEST_SHA256", historical.digest(encoded))
    with pytest.raises(ValueError):
        historical.snapshot_sources()


def test_archive_and_manifest_symlinks_are_rejected(snapshot_copy):
    path = snapshot_copy / "sources.tar.gz"
    moved = snapshot_copy / "moved.gz"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="unsafe"):
        historical.snapshot_sources()


def test_workspace_is_removed_when_a_verifier_fails(monkeypatch):
    staged = []

    def reject(path, *args, **kwargs):
        staged.append(Path(path))
        raise ValueError("checker rejected historical record")

    monkeypatch.setattr(runpy, "run_path", reject)
    with pytest.raises(ValueError, match="checker rejected"):
        historical.check()
    assert len(staged) == 1 and not staged[0].exists()


def test_selected_bytes_are_rechecked_after_validators_finish(monkeypatch):
    original = historical.snapshot_sources
    calls = []

    def changed():
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("selected input changed during check")
        return original()

    monkeypatch.setattr(historical, "snapshot_sources", changed)
    with pytest.raises(ValueError, match="changed during check"):
        historical.check()
    assert len(calls) == 2
