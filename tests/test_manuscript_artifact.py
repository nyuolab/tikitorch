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


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PAPER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_citations_are_defined_once():
    bibliography = (PAPER / "refs.bib").read_text()
    keys = re.findall(r"@\w+\{([^,]+),", bibliography)
    assert len(keys) == len(set(keys))
    used = set()
    for filename in ("main.tex", "theory_appendix.tex", "methods_appendix.tex"):
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", (PAPER / filename).read_text()):
            used.update(k.strip() for k in group.split(","))
    assert used <= set(keys), used - set(keys)
    assert {"groenendijk2023geometric", "kosson2023multiplicationfree",
            "jain2018gist", "novikov2023fewbit", "bolte2020model"} <= used


def test_current_evidence_and_table_are_consistent():
    module = load_script("measure_retained_state")
    record = json.loads(module.DEFAULT_RECORD.read_text())
    module.validate_record(record)
    assert module.DEFAULT_TABLE.read_text() == module.render_table(record)


@pytest.mark.parametrize("change", ["gradient", "route", "bytes", "cells", "sources"])
def test_current_evidence_rejects_inconsistent_claims(change):
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
        module.validate_record(record)


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


def test_style_members_cannot_escape_destination(tmp_path):
    module = load_script("build")
    blob, manifest = archive_with("../escape.sty", b"test")
    with pytest.raises(ValueError, match="invalid template member"):
        module.install_style(blob, manifest, tmp_path / "style")
    assert not (tmp_path / "escape.sty").exists()


def test_review_surface_is_single_anonymous_manuscript():
    source = (PAPER / "main.tex").read_text()
    assert "\\usepackage{icml2026}" in source
    assert "\\icmlcorrespondingauthor" in source
    assert "\\label{main:end}" in source
    assert "../papers/" not in source
    assert "nyuolab" not in source
