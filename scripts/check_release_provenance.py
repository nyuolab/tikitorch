#!/usr/bin/env python3
"""Fail-closed release gate for recorded provenance decisions (stdlib only).

This validates a maintainer-owned inventory; it does not determine licenses,
grant permissions, inspect remote refs, or replace the documented human review.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_IDS = {
    "aws_ddr_muzero", "aws_ddr_pc", "aws_ddr_pc_train", "aws_ddr_train_s1",
    "retained_git_history",
}
RESOLVED_STATUSES = {"independently_replaced", "approved_for_distribution", "excluded_from_release"}


def inspect_inventory(root, manifest):
    """Return unresolved IDs; reject incomplete, stale or unsupported records."""
    if not isinstance(manifest, dict):
        raise ValueError("inventory must be an object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported or missing schema_version")
    if not isinstance(manifest.get("scope"), str) or not manifest["scope"].strip():
        raise ValueError("scope must describe the intended release")
    records = manifest.get("records")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("records must be a list of objects")
    ids = [r.get("id") for r in records]
    if not all(isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("record IDs must be unique strings")
    if not REQUIRED_IDS.issubset(ids):
        raise ValueError(f"required records missing: {sorted(REQUIRED_IDS.difference(ids))}")
    unresolved = []
    for record in records:
        name, status = record["id"], record.get("status")
        if status not in RESOLVED_STATUSES | {"unresolved"}:
            raise ValueError(f"{name}: unsupported or missing status")
        path = record.get("path")
        if name.startswith("aws_ddr_") and not path:
            raise ValueError(f"{name}: source path is required")
        if path:
            source = (root / path).resolve()
            try:
                source.relative_to(root.resolve())
            except ValueError:
                raise ValueError(f"{name}: path escapes checkout") from None
            if status == "excluded_from_release":
                if source.exists():
                    raise ValueError(f"{name}: excluded file is still in the release tree")
            elif not source.is_file():
                raise ValueError(f"{name}: inventoried source is missing")
            elif hashlib.sha256(source.read_bytes()).hexdigest() != record.get("sha256"):
                raise ValueError(f"{name}: source changed; update its digest and review record")
        if status == "unresolved":
            unresolved.append(name)
        else:
            evidence = record.get("evidence")
            if not isinstance(evidence, list) or not evidence or not all(
                isinstance(item, str) and item.strip() for item in evidence
            ):
                raise ValueError(f"{name}: resolved record needs review evidence references")
            if not isinstance(record.get("reviewed_by"), str) or not record["reviewed_by"].strip():
                raise ValueError(f"{name}: resolved record needs a named reviewer")
    return unresolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-only", action="store_true",
        help="validate records without granting release clearance (ordinary CI)",
    )
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((ROOT / "docs/release_provenance.json").read_text())
        unresolved = inspect_inventory(ROOT, manifest)
    except (ValueError, OSError, TypeError) as exc:
        print(json.dumps({"inventory_valid": False, "release_ready": False, "error": str(exc)}))
        return 2
    print(json.dumps({
        "inventory_valid": True,
        "release_ready": not unresolved,
        "unresolved": unresolved,
        "scope": manifest["scope"],
        "mode": "inventory_only" if args.inventory_only else "release_gate",
    }, indent=2))
    if unresolved and not args.inventory_only:
        print("Release blocked: see docs/release_validation.md and THIRD_PARTY_NOTICES.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
