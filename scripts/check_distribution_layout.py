#!/usr/bin/env python3
"""Check the library/research boundary in built wheels and sdists, without extraction."""

import argparse
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile


CHECKOUT_ONLY = {"manuscript", "papers", "scripts", "tests", "results"}
CACHES = {"__pycache__", ".cache", ".pytest_cache", ".ruff_cache", ".mypy_cache",
          ".ipynb_checkpoints", ".tox", ".nox"}


def inspect_archive(archive):
    """Reject research trees, local generated state and unsafe member paths."""
    archive = Path(archive)
    if archive.name.endswith(".whl"):
        with zipfile.ZipFile(archive) as handle:
            entries = [(item.filename,
                        stat.S_ISLNK(item.external_attr >> 16)) for item in handle.infolist()]
        prefix = None
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as handle:
            entries = [(item.name, item.issym() or item.islnk()) for item in handle]
        prefix = archive.name[:-7]
    else:
        raise ValueError("expected a .whl or .tar.gz distribution")
    if not entries:
        raise ValueError("empty distribution")
    for name, linked in entries:
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
                or ":" in name or path.as_posix() != name.rstrip("/")):
            raise ValueError(f"unsafe or noncanonical archive path: {name}")
        if linked:
            raise ValueError(f"linked archive member: {name}")
        parts = path.parts
        if prefix:
            if not parts or parts[0] != prefix:
                raise ValueError(f"member outside the sdist root: {name}")
            parts = parts[1:]
        if parts and parts[0] in CHECKOUT_ONLY:
            raise ValueError(f"checkout-only tree in distribution: {name}")
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
