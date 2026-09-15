"""Offline manuscript integrity checks; no LaTeX, network or GPU required."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import zipfile

import pytest

PAPER = Path(__file__).resolve().parents[1] / "manuscript"
SOURCE = PAPER / "source"


def load_script(name):
    directory = PAPER if name == "build" else PAPER / "tools"
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_citations_are_defined_once():
    bibliography = (SOURCE / "refs.bib").read_text()
    keys = re.findall(r"@\w+\{([^,]+),", bibliography)
    assert len(keys) == len(set(keys))
    used = set()
    for filename in ("main.tex", "theory_appendix.tex", "methods_appendix.tex"):
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", (SOURCE / filename).read_text()):
            used.update(k.strip() for k in group.split(","))
    assert used <= set(keys), used - set(keys)
    assert {"groenendijk2023geometric", "kosson2023multiplicationfree",
            "jain2018gist", "novikov2023fewbit", "bolte2020model"} <= used


def test_historical_cpu_evidence_and_table_are_consistent():
    historical = load_script("historical_cpu_evidence")
    assert historical.check()["current_library_freshness_claimed"] is False
    module = load_script("measure_retained_state")
    record = json.loads(module.DEFAULT_RECORD.read_text())
    # Original strict source validation has already run against the pinned tree.
    module.validate_record(record, check_source=False)
    assert module.DEFAULT_TABLE.read_text() == module.render_table(record)


@pytest.mark.parametrize("change", ["gradient", "route", "bytes", "cells", "sources"])
def test_historical_retained_evidence_rejects_inconsistent_claims(change):
    module = load_script("measure_retained_state")
    record = deepcopy(json.loads(module.DEFAULT_RECORD.read_text()))
    if change == "gradient":
        record["rows"][0]["gradient_errors"][0]["supplied_allclose"] = False
    elif change == "route":
        record["rows"][0]["eager_route_checks"][0]["integer_equal"] = False
    elif change == "bytes":
        record["rows"][0]["packed_device_bytes"] += 1
    elif change == "cells":
        record["rows"][1] = record["rows"][0]
    else:
        record["sources_after_sha256"]["invalid"] = "changed"
    record["results_sha256"] = module.digest(module.canonical(record["rows"]))
    with pytest.raises(ValueError):
        module.validate_record(record, check_source=False)


def archive_with(name, data):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, data)
    blob = stream.getvalue()
    return blob, dict(archive_sha256=hashlib.sha256(blob).hexdigest(),
                      members={name: hashlib.sha256(data).hexdigest()})


def test_style_install_preserves_bytes_and_checks_integrity(tmp_path):
    module = load_script("build")
    blob, manifest = archive_with("example.sty", b"% upstream bytes\n")
    module.install_style(blob, manifest, tmp_path / "style")
    assert (tmp_path / "style/example.sty").read_bytes() == b"% upstream bytes\n"
    with pytest.raises(ValueError, match="archive changed"):
        module.install_style(blob + b"changed", manifest, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_nested_style_install_flattens_only_pinned_members(tmp_path):
    module = load_script("build")
    blob, manifest = archive_with("iclr2027/example.sty", b"% exact upstream bytes\n")
    module.install_style(blob, manifest, tmp_path / "style")
    assert (tmp_path / "style/example.sty").read_bytes() == b"% exact upstream bytes\n"
    assert sorted(p.name for p in (tmp_path / "style").iterdir()) == ["example.sty"]


def test_style_member_digest_is_checked_before_any_install(tmp_path):
    module = load_script("build")
    blob, manifest = archive_with("iclr2027/example.sty", b"upstream")
    manifest["members"]["iclr2027/example.sty"] = "0" * 64
    with pytest.raises(ValueError, match="invalid template member"):
        module.install_style(blob, manifest, tmp_path / "style")
    assert not (tmp_path / "style").exists()


@pytest.mark.parametrize("name", [
    "../escape.sty", "nested/../../escape.sty", "/escape.sty",
    "nested\\escape.sty", "C:/escape.sty", "./escape.sty", "nested//escape.sty",
    "nested/example.tex",
])
def test_style_members_cannot_escape_destination(tmp_path, name):
    module = load_script("build")
    blob, manifest = archive_with(name, b"test")
    with pytest.raises(ValueError, match="invalid template member"):
        module.install_style(blob, manifest, tmp_path / "style")
    assert not (tmp_path / "escape.sty").exists()
    assert not (tmp_path / "style").exists()


def test_style_install_rejects_ambiguous_flattened_filenames(tmp_path):
    module = load_script("build")
    blob, manifest = archive_with("one/example.sty", b"test")
    manifest["members"]["two/example.sty"] = manifest["members"]["one/example.sty"]
    with pytest.raises(ValueError, match="invalid template member"):
        module.install_style(blob, manifest, tmp_path / "style")
    assert not (tmp_path / "style").exists()


def test_style_install_rejects_symbolic_archive_member(tmp_path):
    module = load_script("build")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        member = zipfile.ZipInfo("iclr2027/example.sty")
        member.create_system = 3
        member.external_attr = 0o120777 << 16
        archive.writestr(member, b"outside.sty")
    blob = stream.getvalue()
    manifest = dict(archive_sha256=hashlib.sha256(blob).hexdigest(),
                    members={"iclr2027/example.sty": hashlib.sha256(b"outside.sty").hexdigest()})
    with pytest.raises(ValueError, match="invalid template member"):
        module.install_style(blob, manifest, tmp_path / "style")
    assert not (tmp_path / "style").exists()


def test_style_install_rejects_symbolic_output_member(tmp_path):
    module = load_script("build")
    blob, manifest = archive_with("iclr2027/example.sty", b"upstream")
    outside = tmp_path / "outside.sty"
    outside.write_bytes(b"preserved")
    destination = tmp_path / "style"
    destination.mkdir()
    (destination / "example.sty").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        module.install_style(blob, manifest, destination)
    assert outside.read_bytes() == b"preserved"


def test_verified_template_declares_initial_submission_page_limit():
    manifest = json.loads((SOURCE / "template_manifest.json").read_text())
    assert manifest["venue"] == "ICLR" and manifest["template_year"] == 2027
    assert manifest["main_body_page_limit"] == 9
    assert manifest["author_instructions"] == "https://iclr.cc/Conferences/2027/AuthorGuidelines"
    assert set(manifest["members"]) == {
        "iclr2027/iclr2027_conference.sty", "iclr2027/iclr2027_conference.bst",
        "iclr2027/fancyhdr.sty", "iclr2027/natbib.sty",
    }


def test_review_surface_is_single_anonymous_manuscript():
    source = (SOURCE / "main.tex").read_text()
    assert "\\usepackage{iclr2027_conference,times}" in source
    assert "\\maketitle" in source
    assert "\\bibliographystyle{iclr2027_conference}" in source
    assert "\\twocolumn" not in source
    assert "\\label{main:end}" in source
    assert "../papers/" not in source
    assert "nyuolab" not in source


def test_discussion_copy_does_not_claim_submission_or_acceptance():
    source = (SOURCE / "main.tex").read_text()
    assert "Discussion draft. Not submitted." in source
    assert "pdfsubject={TikiTorch discussion draft; not submitted}" in source
    active_source = "\n".join(line.split("%", 1)[0] for line in source.splitlines())
    assert r"\iclrfinalcopy" not in active_source
