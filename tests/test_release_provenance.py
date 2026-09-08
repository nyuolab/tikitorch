"""The provenance gate cannot silently turn incomplete records into clearance."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.check_release_provenance import REQUIRED_IDS, inspect_inventory


ROOT = Path(__file__).resolve().parents[1]


def _inventory():
    return json.loads((ROOT / "docs/release_provenance.json").read_text())


def test_checked_in_inventory_is_complete_but_not_release_clearance():
    unresolved = inspect_inventory(ROOT, _inventory())
    assert set(unresolved) == REQUIRED_IDS


@pytest.mark.parametrize("inventory_only,expected_rc", [(True, 0), (False, 1)])
def test_provenance_cli_distinguishes_integrity_from_release(inventory_only, expected_rc):
    command = [sys.executable, str(ROOT / "scripts/check_release_provenance.py")]
    if inventory_only:
        command.append("--inventory-only")
    result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert result.returncode == expected_rc
    report = json.loads(result.stdout)
    assert report["inventory_valid"] is True
    assert report["release_ready"] is False


@pytest.mark.parametrize("corruption,match", [
    ("missing", "required records missing"),
    ("duplicate", "unique strings"),
    ("status", "unsupported or missing status"),
    ("digest", "source changed"),
    ("traversal", "path escapes checkout"),
    ("empty_approval", "review evidence"),
    ("no_reviewer", "named reviewer"),
    ("false_exclusion", "still in the release tree"),
])
def test_incomplete_or_stale_records_fail_closed(corruption, match):
    manifest = _inventory()
    first = manifest["records"][0]
    if corruption == "missing":
        manifest["records"].pop()
    elif corruption == "duplicate":
        manifest["records"].append(copy.deepcopy(first))
    elif corruption == "status":
        first["status"] = "probably_ok"
    elif corruption == "digest":
        first["sha256"] = "0" * 64
    elif corruption == "traversal":
        first["path"] = "../outside.sv"
    elif corruption == "empty_approval":
        first["status"] = "approved_for_distribution"
    elif corruption == "no_reviewer":
        first.update(status="approved_for_distribution", evidence=["review-reference"])
    elif corruption == "false_exclusion":
        first["status"] = "excluded_from_release"
    with pytest.raises(ValueError, match=match):
        inspect_inventory(ROOT, manifest)


def test_complete_declared_approvals_are_structurally_accepted():
    # Synthetic data only: this tests schema behavior, not actual legal approval.
    manifest = _inventory()
    for record in manifest["records"]:
        record.update(
            status="approved_for_distribution",
            evidence=["synthetic-test-review"], reviewed_by="test-fixture",
        )
    assert inspect_inventory(ROOT, manifest) == []
