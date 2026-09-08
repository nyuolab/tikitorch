#!/usr/bin/env python3
"""Check the library/research boundary in built wheels and sdists, without extraction."""

import argparse
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile


CHECKOUT_ONLY = {"manuscript", "papers", "scripts", "tests", "results"}
GENERATED_ROOTS = {"build", "dist", "site"}
CACHES = {"__pycache__", ".cache", ".pytest_cache", ".ruff_cache", ".mypy_cache",
          ".ipynb_checkpoints", ".tox", ".nox", ".venv", ".DS_Store", ".git"}
WHEEL_DATA_DESTINATIONS = {"purelib", "platlib", "data", "headers", "scripts"}


def _zip_member_kind(member):
    # ZIP entries from non-Unix builders need not have a POSIX file-type field.
    mode = stat.S_IFMT(member.external_attr >> 16)
    if mode == stat.S_IFLNK:
        return "linked"
    if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
        return "special"
    return "directory" if member.is_dir() or mode == stat.S_IFDIR else "file"


def _tar_member_kind(member):
    if member.issym() or member.islnk():
        return "linked"
    if member.isdir():
        return "directory"
    return "file" if member.isfile() else "special"


def inspect_archive(archive):
    """Reject research trees, local generated state and unsafe member paths."""
    archive = Path(archive)
    if archive.name.endswith(".whl"):
        with zipfile.ZipFile(archive) as handle:
            entries = [(item.filename, _zip_member_kind(item)) for item in handle.infolist()]
        prefix = None
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as handle:
            entries = [(item.name, _tar_member_kind(item)) for item in handle]
        prefix = archive.name[:-7]
    else:
        raise ValueError("expected a .whl or .tar.gz distribution")
    if not entries:
        raise ValueError("empty distribution")
    seen = set()
    for name, kind in entries:
        path = PurePosixPath(name)
        if (not name or not path.parts or path.is_absolute() or ".." in path.parts
                or "\\" in name or ":" in name
                or name not in {path.as_posix(), path.as_posix() + "/"}):
            raise ValueError(f"unsafe or noncanonical archive path: {name}")
        if kind == "linked":
            raise ValueError(f"linked archive member: {name}")
        if kind == "special":
            raise ValueError(f"non-regular archive member: {name}")
        canonical = path.as_posix()
        if canonical in seen:
            raise ValueError(f"duplicate archive member: {name}")
        seen.add(canonical)
        parts = path.parts
        if prefix:
            if not parts or parts[0] != prefix:
                raise ValueError(f"member outside the sdist root: {name}")
            parts = parts[1:]
        # Wheels can relocate content through their .data install scheme; checking
        # only the ZIP root would let a research tree bypass this boundary there.
        installed_parts = parts
        if not prefix and len(parts) >= 2 and parts[0].endswith(".data"):
            if parts[1] not in WHEEL_DATA_DESTINATIONS:
                raise ValueError(f"unknown wheel data destination: {name}")
            installed_parts = parts[2:]
        if installed_parts and installed_parts[0] in CHECKOUT_ONLY:
            raise ValueError(f"checkout-only tree in distribution: {name}")
        if installed_parts and installed_parts[0] in GENERATED_ROOTS:
            raise ValueError(f"generated root tree in distribution: {name}")
        if any(part in CACHES or part == ".env" or part.startswith(".env.") for part in parts):
            raise ValueError(f"local environment/cache artifact in distribution: {name}")
        if path.suffix in {".pyc", ".pyo"}:
            raise ValueError(f"bytecode in distribution: {name}")
    return {"archive": archive.name, "layout_valid": True, "members": len(entries)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        reports = [inspect_archive(path) for path in args.archives]
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(json.dumps({"layout_valid": False, "error": str(exc)}))
        return 1
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
