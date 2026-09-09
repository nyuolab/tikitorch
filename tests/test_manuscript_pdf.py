"""Sharing a PDF must not silently share stale or failed-build output."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from manuscript import build as builder


@pytest.fixture
def review_workspace(tmp_path, monkeypatch):
    paper = tmp_path / "manuscript"
    source = paper / "source"
    build = paper / "build"
    source.mkdir(parents=True)
    (build / "style").mkdir(parents=True)
    (paper / "evidence").mkdir()
    style = b"% pinned upstream style\n"
    (build / "style/example.sty").write_bytes(style)
    manifest = {"members": {"example.sty": hashlib.sha256(style).hexdigest()}}
    (source / "template_manifest.json").write_text(json.dumps(manifest))
    (build / "main.pdf").write_bytes(b"%PDF-1.4\nnew build\n")
    (build / "main.log").write_text("No layout warnings.\n")
    (build / "main.aux").write_text(r"\newlabel{main:end}{{7}{6}{Conclusion}{section.7}{}}")
    (paper / "TikiTorch.pdf").write_bytes(b"%PDF-1.4\nprevious checked copy\n")
    (paper / "evidence/review_pdf.json").write_text("previous record\n")
    for key, value in {
        "PAPER": paper, "SOURCE": source, "TOOLS": paper / "tools",
        "REVIEW_PDF": paper / "TikiTorch.pdf",
        "REVIEW_RECORD": paper / "evidence/review_pdf.json",
    }.items():
        monkeypatch.setattr(builder, key, value)
    monkeypatch.setattr(builder, "check_evidence", lambda: None)
    monkeypatch.setattr(builder, "review_inputs", lambda: {"manuscript/source/main.tex": "0" * 64})
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/test/bin/latexmk")
    monkeypatch.setattr(builder.subprocess, "run", lambda *args, **kwargs: None)
    return paper


def test_successful_build_publishes_pdf_and_bound_record(review_workspace):
    assert builder.main([]) == 0
    assert builder.REVIEW_PDF.read_bytes() == (review_workspace / "build/main.pdf").read_bytes()
    record = builder.check_review_pdf()
    assert record["main_body_pages"] == 6
    assert record["inputs_sha256"] == builder.review_inputs()
    assert not list((review_workspace / "build").glob("review-*"))


@pytest.mark.parametrize("failure", [
    "style", "evidence", "late_evidence", "latex", "citation", "overfull",
    "page_missing", "page_zero", "page_over", "size", "not_pdf", "source_changed",
])
def test_failed_build_preserves_previous_review_copy(review_workspace, monkeypatch, failure):
    before_pdf = builder.REVIEW_PDF.read_bytes()
    before_record = builder.REVIEW_RECORD.read_bytes()
    build = review_workspace / "build"

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate failed build check")

    if failure == "style":
        (build / "style/example.sty").write_bytes(b"modified")
    elif failure == "evidence":
        monkeypatch.setattr(builder, "check_evidence", fail)
    elif failure == "late_evidence":
        calls = []

        def evidence():
            calls.append(True)
            if len(calls) == 2:
                fail()
        monkeypatch.setattr(builder, "check_evidence", evidence)
    elif failure == "latex":
        monkeypatch.setattr(builder.subprocess, "run", fail)
    elif failure == "citation":
        (build / "main.log").write_text("LaTeX Warning: Citation `missing' on page 1 undefined")
    elif failure == "overfull":
        (build / "main.log").write_text(r"Overfull \hbox (2.0pt too wide)")
    elif failure.startswith("page_"):
        page = {"page_missing": "", "page_zero": "0", "page_over": "9"}[failure]
        (build / "main.aux").write_text(r"\newlabel{main:end}{{7}{" + page + "}{Conclusion}{section.7}{}}")
    elif failure == "size":
        with (build / "main.pdf").open("r+b") as pdf:
            pdf.truncate(20 * 1024 * 1024 + 1)
    elif failure == "not_pdf":
        (build / "main.pdf").write_bytes(b"not a PDF")
    elif failure == "source_changed":
        inputs = iter(({"before": "0" * 64}, {"after": "1" * 64}))
        monkeypatch.setattr(builder, "review_inputs", lambda: next(inputs))
    with pytest.raises((ValueError, RuntimeError)):
        builder.main([])
    assert builder.REVIEW_PDF.read_bytes() == before_pdf
    assert builder.REVIEW_RECORD.read_bytes() == before_record


def test_readonly_check_needs_neither_latex_nor_network(review_workspace, monkeypatch):
    builder.main([])
    before = {p.relative_to(review_workspace): p.read_bytes()
              for p in review_workspace.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("read-only check attempted LaTeX, a download or style installation")

    monkeypatch.setattr(builder.shutil, "which", forbidden)
    monkeypatch.setattr(builder, "urlopen", forbidden)
    monkeypatch.setattr(builder, "install_style", forbidden)
    monkeypatch.setattr(builder.subprocess, "run", forbidden)
    assert builder.main(["--check"]) == 0
    after = {p.relative_to(review_workspace): p.read_bytes()
             for p in review_workspace.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("option", ["--fetch-style", "--template-zip"])
def test_readonly_check_rejects_style_install_options(review_workspace, option):
    arguments = ["--check", option]
    if option == "--template-zip":
        arguments.append("template.zip")
    with pytest.raises(SystemExit) as error:
        builder.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("failure", ["pdf", "input", "schema", "pages", "missing_record"])
def test_review_check_rejects_stale_or_corrupt_snapshot(review_workspace, monkeypatch, failure):
    builder.main([])
    record = json.loads(builder.REVIEW_RECORD.read_text())
    if failure == "pdf":
        builder.REVIEW_PDF.write_bytes(b"%PDF-1.4\nchanged\n")
    elif failure == "input":
        monkeypatch.setattr(builder, "review_inputs", lambda: {"changed": "1" * 64})
    elif failure == "schema":
        record["schema"] = "unsupported"
        builder.REVIEW_RECORD.write_text(json.dumps(record))
    elif failure == "pages":
        record["main_body_pages"] = True
        builder.REVIEW_RECORD.write_text(json.dumps(record))
    else:
        builder.REVIEW_RECORD.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.check_review_pdf()


def test_checked_in_review_pdf_is_current():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "manuscript/build.py", "--check"],
                   cwd=root, check=True, capture_output=True, text=True)


def test_review_input_record_covers_sources_generators_and_selected_evidence():
    inputs = builder.review_inputs()
    expected = {
        "manuscript/build.py", "manuscript/source/.latexmkrc",
        "manuscript/source/main.tex", "manuscript/source/theory_appendix.tex",
        "manuscript/source/methods_appendix.tex", "manuscript/source/refs.bib",
        "manuscript/source/template_manifest.json",
        "manuscript/source/generated/current_retained_state.tex",
        "manuscript/source/generated/table_provenance.json",
        "manuscript/tools/generate_tables.py", "manuscript/tools/measure_retained_state.py",
        "manuscript/evidence/retained_state_layout_2026-09-09.json",
    }
    assert expected <= set(inputs)
    assert any(name.startswith("manuscript/results/") for name in inputs)
