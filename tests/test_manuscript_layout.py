"""One research workspace, with byte-preserved superseded scientific sources."""

import hashlib
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript"
ARCHIVE = PAPER / "archive"
SOURCE_NAMES = {
    "multiplier_free_fpga/.latexmkrc", "multiplier_free_fpga/main.tex",
    "multiplier_free_fpga/refs.bib", "route_certificates/.latexmkrc",
    "route_certificates/main.tex", "route_certificates/methods.tex",
    "route_certificates/refs.bib", "route_certificates/supplement.tex",
}


def test_research_has_one_workspace_and_one_shared_handoff():
    assert not (ROOT / "papers").exists()
    assert not (ROOT / "papers").is_symlink()
    for name in ("main.tex", "RESEARCH_PLAN.md", "HANDOFF.md"):
        assert (PAPER / name).is_file()
    for name in ("STUDENT_HANDOFF.md", "XIANG_HANDOFF.md"):
        for parent in (PAPER, ARCHIVE):
            assert not (parent / name).exists()
            assert not (parent / name).is_symlink()
    assert (PAPER / "results").is_dir()
    assert set(PAPER.glob("*HANDOFF*.md")) == {PAPER / "HANDOFF.md"}


def test_archived_scientific_sources_match_the_pre_merge_snapshot():
    manifest = json.loads((ARCHIVE / "source_manifest.json").read_text())
    assert manifest["schema"] == "tikitorch.archived_manuscript_sources.v1"
    assert manifest["source_commit"] == "1b4b00308a53919dee6afbc8fdca86011659d348"
    records = manifest["files"]
    assert len(records) == len(SOURCE_NAMES)
    assert {row["original_path"] for row in records} == {
        "papers/" + name for name in SOURCE_NAMES
    }
    for row in records:
        assert row["path"] == row["original_path"].replace("papers/", "manuscript/archive/", 1)
        path = ROOT / row["path"]
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        assert mode == row["mode"]


@pytest.mark.parametrize("relative", [
    "README.md", "HANDOFF.md", "ARTIFACTS.md",
    "archive/README.md", "archive/route_certificates/README.md",
    "archive/multiplier_free_fpga/README.md", "results/route_certificates/README.md",
])
def test_consolidated_manuscript_navigation_resolves(relative):
    source = PAPER / relative
    # These navigation pages use ordinary inline Markdown links, not generated URLs.
    for target in re.findall(r"\]\(([^\s)]+)\)", source.read_text()):
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        assert (source.parent / target).exists(), f"{relative}: {target}"


def test_archived_build_outputs_stay_ignored_and_out_of_distributions():
    rules = (ROOT / ".gitignore").read_text().splitlines()
    assert "/manuscript/archive/*/build/" in rules
    assert "/papers/*/build/" not in rules
    assert "prune manuscript" in (ROOT / "MANIFEST.in").read_text().splitlines()
