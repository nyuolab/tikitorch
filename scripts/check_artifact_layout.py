#!/usr/bin/env python3
"""Validate the frozen research-artifact relocation without Git or dependencies.

The default check is read-only and permits new append-only artifacts. Historical
files may not disappear or change. ``--record-from`` is a one-time migration
operation requiring Git; it refuses to overwrite an existing inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = Path("manuscript/artifact_relocation.json")
SOURCE_COMMIT = "197b2644441877a9662d5a33b84b118bef169bc5"
TRACKED_FILES = 544
SCHEMA = "tikitorch.artifact_relocation.v1"
NAVIGATION_EDITS = {
    "results/README.md": (
        "../docs/research/CLAIM_LEDGER.md", "../scripts/README.md",
    ),
    "results/board_scaling/diagnostics/SCALED_RUN_DESIGN.md": (
        "../../../scripts/muzero_scale_bench.py", "../../../scripts/muzero_scale_cost.py",
    ),
    "results/board_scaling/diagnostics/ROOT_CAUSE_TRAINING.md": (
        "../../../scripts/muzero_train_diagnostics.py",
    ),
    "results/route_certificates/README.md": (
        "../../papers/route_certificates/README.md",
    ),
    "results/fpga_silicon/ddr_arch/README.md": ("../../../THIRD_PARTY_NOTICES.md",),
    "results/fpga_silicon/s1_afibuild/ARTIFACTS.md": ("../../../scripts/run_s1_afi_build.sh",),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_inventory(path):
    """Reject repeated JSON keys instead of accepting an ambiguous inventory."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def regular_file(root, relative):
    """Reject traversal, symlink aliases and non-files before reading bytes."""
    require(isinstance(relative, str) and relative, "path must be a nonempty string")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and path.as_posix() == relative
            and ".." not in path.parts and "\\" not in relative,
            f"noncanonical or escaping path: {relative}")
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        require(not candidate.is_symlink(), f"symlink in frozen artifact path: {relative}")
    require(candidate.is_file(), f"missing frozen artifact: {relative}")
    require(candidate.resolve().is_relative_to(root.resolve()),
            f"artifact path escapes checkout: {relative}")
    require(stat.S_ISREG(candidate.stat().st_mode), f"artifact is not a regular file: {relative}")
    return candidate


def git_mode(path):
    """Git preserves executable intent, not per-user filesystem permission bits."""
    return "100755" if path.stat().st_mode & 0o111 else "100644"


def inspect_inventory(root, manifest, *, require_frozen_snapshot=False):
    """Validate an inventory and its frozen files; new artifact files are allowed."""
    root = Path(root)
    legacy = root / "results"
    require(not legacy.exists() and not legacy.is_symlink(),
            "legacy root results path exists (including symlinks)")
    require(isinstance(manifest, dict), "inventory must be an object")
    keys = {"schema", "source_commit", "source_root", "destination_root",
            "tracked_files", "navigation_changes", "files"}
    require(set(manifest) == keys, "unsupported or incomplete inventory fields")
    require(manifest["schema"] == SCHEMA, "unsupported inventory schema")
    require(isinstance(manifest["source_commit"], str)
            and re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]),
            "source_commit must be a full Git SHA")
    require(manifest["source_root"] == "results"
            and manifest["destination_root"] == "manuscript/results",
            "unsupported artifact relocation roots")
    records = manifest["files"]
    require(isinstance(records, list) and records, "files must be a nonempty list")
    require(type(manifest["tracked_files"]) is int
            and manifest["tracked_files"] == len(records), "tracked file count mismatch")
    require(type(manifest["navigation_changes"]) is int, "invalid navigation change count")
    if require_frozen_snapshot:
        require(manifest["source_commit"] == SOURCE_COMMIT
                and manifest["tracked_files"] == TRACKED_FILES,
                "inventory does not describe the frozen 544-file source snapshot")
    seen = set()
    navigation = set()
    for record in records:
        require(isinstance(record, dict), "file record must be an object")
        require(set(record) == {"original_path", "path", "mode", "sha256",
                                "original_sha256", "change"}, "unsupported file record fields")
        original, relocated = record["original_path"], record["path"]
        require(isinstance(original, str) and original.startswith("results/")
                and relocated == "manuscript/" + original, "invalid relocation mapping")
        require(relocated not in seen, "duplicate artifact path")
        seen.add(relocated)
        for field in ("sha256", "original_sha256"):
            require(isinstance(record[field], str)
                    and re.fullmatch(r"[0-9a-f]{64}", record[field]), "invalid SHA256 digest")
        change = record["change"]
        if change == "navigation_links_only":
            require(original in NAVIGATION_EDITS, "unapproved navigation-edit path")
            require(record["sha256"] != record["original_sha256"], "navigation edit has equal digests")
            navigation.add(original)
        else:
            require(change == "byte_preserved", "unsupported artifact change classification")
            require(record["sha256"] == record["original_sha256"], "raw artifact digest changed")
        path = regular_file(root, relocated)
        require(record["mode"] in {"100644", "100755"}, "unsupported Git file mode")
        require(git_mode(path) == record["mode"], f"executable mode changed: {relocated}")
        require(file_digest(path) == record["sha256"], f"artifact digest mismatch: {relocated}")
    require(len(navigation) == manifest["navigation_changes"], "navigation change count mismatch")
    if require_frozen_snapshot:
        require(navigation == set(NAVIGATION_EDITS), "frozen navigation-edit inventory is incomplete")
    return {"inventory_valid": True, "tracked_files": len(records),
            "byte_preserved": len(records) - len(navigation),
            "navigation_links_only": len(navigation), "new_artifacts_permitted": True}


def record_inventory(root, revision, output):
    """Bind old Git blobs to relocated bytes, allowing exactly declared link edits."""
    root, output = Path(root), Path(output)
    require(not output.exists() and not output.is_symlink(), "refusing to overwrite inventory")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=root, text=True,
    ).strip()
    require(commit == SOURCE_COMMIT, "recording is restricted to the approved source snapshot")
    entries = subprocess.check_output(
        ["git", "ls-tree", "-rz", commit, "--", "results/"], cwd=root,
    ).split(b"\0")
    tree = []
    for entry in filter(None, entries):
        header, path = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        require(kind == "blob" and mode in {"100644", "100755"}, "unsupported source Git object")
        tree.append((mode, oid, path.decode("utf-8")))
    require(len(tree) == TRACKED_FILES, "source snapshot does not contain 544 tracked artifacts")
    blobs = io.BytesIO(subprocess.check_output(
        ["git", "cat-file", "--batch"], cwd=root,
        input="".join(f"{oid}\n" for _, oid, _ in tree).encode(),
    ))
    records = []
    for mode, oid, original in tree:
        actual_oid, kind, size = blobs.readline().decode().strip().split()
        require(actual_oid == oid and kind == "blob", "Git blob response mismatch")
        before = blobs.read(int(size))
        require(blobs.read(1) == b"\n", "truncated Git blob response")
        relocated = "manuscript/" + original
        path = regular_file(root, relocated)
        after = path.read_bytes()
        require(git_mode(path) == mode, f"source executable mode changed: {original}")
        expected = before
        if original in NAVIGATION_EDITS:
            for target in NAVIGATION_EDITS[original]:
                old = f"]({target})".encode()
                require(expected.count(old) == 1, f"ambiguous historical navigation link: {original}")
                expected = expected.replace(old, f"](../{target})".encode())
        require(after == expected, f"unapproved artifact edit: {original}")
        records.append(dict(original_path=original, path=relocated, mode=mode,
                            sha256=digest(after), original_sha256=digest(before),
                            change="byte_preserved" if before == after else "navigation_links_only"))
    manifest = dict(schema=SCHEMA, source_commit=commit, source_root="results",
                    destination_root="manuscript/results", tracked_files=len(records),
                    navigation_changes=len(NAVIGATION_EDITS), files=records)
    inspect_inventory(root, manifest, require_frozen_snapshot=True)
    # Exclusive creation also protects against a concurrent writer after validation.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-from", metavar="COMMIT",
                        help="create the one-time relocation inventory from the approved Git commit")
    args = parser.parse_args(argv)
    try:
        if args.record_from:
            manifest = record_inventory(ROOT, args.record_from, ROOT / INVENTORY)
        else:
            manifest = read_inventory(ROOT / INVENTORY)
        report = inspect_inventory(ROOT, manifest, require_frozen_snapshot=True)
    except (ValueError, OSError, TypeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"inventory_valid": False, "error": str(exc)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
