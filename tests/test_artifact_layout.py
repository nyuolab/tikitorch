"""Frozen evidence moves intact while new research remains append-only."""

import copy
from pathlib import Path

import pytest

from scripts import check_artifact_layout as layout


@pytest.fixture
def frozen(tmp_path):
    path = tmp_path / "manuscript/results/study/raw.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"metric": 1}\n')
    path.chmod(0o644)
    record = dict(original_path="results/study/raw.json",
                  path="manuscript/results/study/raw.json", mode="100644",
                  sha256=layout.file_digest(path), original_sha256=layout.file_digest(path),
                  change="byte_preserved")
    manifest = dict(schema=layout.SCHEMA, source_commit=layout.SOURCE_COMMIT,
                    source_root="results", destination_root="manuscript/results",
                    tracked_files=1, navigation_changes=0, files=[record])
    return tmp_path, path, manifest


def test_frozen_records_validate_without_git_and_allow_new_artifacts(frozen, monkeypatch):
    root, path, manifest = frozen
    def no_git(*args, **kwargs):
        raise AssertionError("default validation must not call Git")
    monkeypatch.setattr(layout.subprocess, "check_output", no_git)
    (path.parent / "new_run.json").write_text("new append-only evidence")
    report = layout.inspect_inventory(root, manifest)
    assert report["tracked_files"] == report["byte_preserved"] == 1
    assert report["new_artifacts_permitted"] is True


@pytest.mark.parametrize("kind", ["directory", "broken_symlink"])
def test_legacy_root_directory_or_symlink_is_rejected(frozen, kind):
    root, _, manifest = frozen
    if kind == "directory":
        (root / "results").mkdir()
    else:
        (root / "results").symlink_to(root / "nonexistent")
    with pytest.raises(ValueError, match="legacy root results"):
        layout.inspect_inventory(root, manifest)


@pytest.mark.parametrize("change,match", [
    ("bytes", "digest mismatch"), ("mode", "mode changed"),
    ("missing", "missing frozen"), ("symlink", "symlink in frozen"),
])
def test_changed_or_missing_frozen_files_fail(frozen, change, match):
    root, path, manifest = frozen
    if change == "bytes":
        path.write_text("changed")
    elif change == "mode":
        path.chmod(0o755)
    elif change == "missing":
        path.unlink()
    else:
        target = path.parent / "target.json"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError, match=match):
        layout.inspect_inventory(root, manifest)


@pytest.mark.parametrize("change,match", [
    ("schema", "schema"), ("unknown_field", "inventory fields"),
    ("duplicate", "duplicate"), ("count", "count mismatch"),
    ("escape", "escaping path"), ("absolute", "invalid relocation"),
    ("original_digest", "raw artifact digest"), ("bad_digest", "invalid SHA256"),
    ("raw_navigation", "unapproved navigation"), ("unknown_change", "classification"),
    ("source_commit", "full Git SHA"), ("empty", "nonempty list"),
])
def test_malformed_or_incomplete_inventory_fails_closed(frozen, change, match):
    root, _, original = frozen
    manifest = copy.deepcopy(original)
    record = manifest["files"][0]
    if change == "schema":
        manifest["schema"] = "guess"
    elif change == "unknown_field":
        manifest["approved"] = True
    elif change == "duplicate":
        manifest["files"].append(copy.deepcopy(record))
        manifest["tracked_files"] = 2
    elif change == "count":
        manifest["tracked_files"] = 2
    elif change == "escape":
        record.update(original_path="results/../../outside", path="manuscript/results/../../outside")
    elif change == "absolute":
        record["path"] = "/outside"
    elif change == "original_digest":
        record["original_sha256"] = "0" * 64
    elif change == "bad_digest":
        record["sha256"] = "no digest"
    elif change == "raw_navigation":
        record["change"] = "navigation_links_only"
    elif change == "unknown_change":
        record["change"] = "probably_same"
    elif change == "source_commit":
        manifest["source_commit"] = "HEAD"
    elif change == "empty":
        manifest["files"] = []
    with pytest.raises(ValueError, match=match):
        layout.inspect_inventory(root, manifest)


def test_missing_frozen_entries_cannot_be_hidden_by_adjusting_the_count(frozen):
    root, _, manifest = frozen
    with pytest.raises(ValueError, match="544-file source snapshot"):
        layout.inspect_inventory(root, manifest, require_frozen_snapshot=True)


def test_inventory_creation_never_overwrites_existing_record(frozen):
    root, _, _ = frozen
    output = root / "inventory.json"
    output.write_text("original inventory")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        layout.record_inventory(root, layout.SOURCE_COMMIT, output)
    assert output.read_text() == "original inventory"


@pytest.mark.parametrize("mutation", [None, "metric", "old_target", "extra_link"])
def test_recording_allows_only_the_exact_paper_archive_link(tmp_path, monkeypatch, mutation):
    original = "results/route_certificates/README.md"
    old_target = "../../papers/route_certificates/README.md"
    new_target = "../../archive/route_certificates/README.md"
    before = f"Ratio: 1.69x. See [paper README]({old_target}).\n".encode()
    after = before.replace(old_target.encode(), new_target.encode())
    if mutation == "metric":
        after = after.replace(b"1.69x", b"9.69x")
    elif mutation == "old_target":
        after = before
    elif mutation == "extra_link":
        after += f"[extra]({new_target})\n".encode()
    path = tmp_path / "manuscript" / original
    path.parent.mkdir(parents=True)
    path.write_bytes(after)
    path.chmod(0o644)
    output = tmp_path / "inventory.json"
    oid = "f" * 40

    def source_snapshot(command, **kwargs):
        if command[1] == "rev-parse":
            return layout.SOURCE_COMMIT + "\n"
        if command[1] == "ls-tree":
            return f"100644 blob {oid}\t{original}\0".encode()
        assert command[1:] == ["cat-file", "--batch"]
        assert kwargs["input"] == f"{oid}\n".encode()
        return f"{oid} blob {len(before)}\n".encode() + before + b"\n"

    monkeypatch.setattr(layout.subprocess, "check_output", source_snapshot)
    monkeypatch.setattr(layout, "TRACKED_FILES", 1)
    monkeypatch.setattr(layout, "NAVIGATION_EDITS", {
        original: layout.NAVIGATION_EDITS[original],
    })
    if mutation:
        with pytest.raises(ValueError, match="unapproved artifact edit"):
            layout.record_inventory(tmp_path, layout.SOURCE_COMMIT, output)
        assert not output.exists()
    else:
        manifest = layout.record_inventory(tmp_path, layout.SOURCE_COMMIT, output)
        assert manifest["source_commit"] == layout.SOURCE_COMMIT
        assert manifest["files"] == [dict(
            original_path=original, path="manuscript/" + original, mode="100644",
            sha256=layout.digest(after), original_sha256=layout.digest(before),
            change="navigation_links_only",
        )]
        assert layout.read_inventory(output) == manifest


@pytest.mark.parametrize("source", [
    '{"schema": "one", "schema": "two"}',
    '{"files": [{"path": "one", "path": "two"}]}',
])
def test_duplicate_json_fields_are_rejected_before_validation(tmp_path, source):
    output = tmp_path / "inventory.json"
    output.write_text(source)
    with pytest.raises(ValueError, match="duplicate JSON field"):
        layout.read_inventory(output)


def test_checked_in_relocation_preserves_544_tracked_files():
    root = Path(__file__).resolve().parents[1]
    manifest = layout.read_inventory(root / layout.INVENTORY)
    report = layout.inspect_inventory(root, manifest, require_frozen_snapshot=True)
    assert report["tracked_files"] == 544
    assert report["byte_preserved"] == 538
    assert report["navigation_links_only"] == 6
