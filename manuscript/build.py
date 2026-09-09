#!/usr/bin/env python3
"""Build the review PDF, or check its source freshness without LaTeX or network access.

Style integrity, source freshness and page checks are not scientific clearance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "source"
TOOLS = PAPER / "tools"
REVIEW_PDF = PAPER / "TikiTorch.pdf"
REVIEW_RECORD = PAPER / "evidence" / "review_pdf.json"
REVIEW_SCHEMA = "tikitorch.review_pdf.v1"


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_evidence():
    for name in ("generate_tables", "measure_retained_state"):
        subprocess.run([sys.executable, str(TOOLS / f"{name}.py"), "--check"], check=True)


def review_inputs():
    """Bind repository PDF inputs, not a claim of a hermetic TeX installation.

    The template manifest pins upstream styles; local TeX/font installations
    remain build dependencies. Read-only checking does not require those tools.
    The selected evidence checker separately binds every library Python source.
    """
    measurement = runpy.run_path(str(TOOLS / "measure_retained_state.py"))
    tables = runpy.run_path(str(TOOLS / "generate_tables.py"))
    paths = [PAPER / "build.py", TOOLS / "generate_tables.py",
             TOOLS / "measure_retained_state.py", measurement["DEFAULT_RECORD"]]
    paths.extend(path for path in SOURCE.rglob("*") if path.is_file())
    paths.extend(PAPER.parent / name for name in tables["SOURCE_SHA256"])
    if any(path.is_symlink() for path in paths):
        raise ValueError("review inputs must be regular files, not symbolic links")
    return {path.relative_to(PAPER.parent).as_posix(): file_digest(path)
            for path in sorted(set(paths))}


def check_review_pdf():
    """Verify the checked-in review copy without fetching styles or running TeX."""
    check_evidence()
    record = json.loads(REVIEW_RECORD.read_text(encoding="utf-8"))
    if record.get("schema") != REVIEW_SCHEMA or record.get("pdf") != "TikiTorch.pdf":
        raise ValueError("unsupported review PDF record")
    if record.get("inputs_sha256") != review_inputs():
        raise ValueError("review PDF source inputs changed; rebuild with python manuscript/build.py")
    if not REVIEW_PDF.read_bytes().startswith(b"%PDF-") or file_digest(REVIEW_PDF) != record.get("pdf_sha256"):
        raise ValueError("review PDF digest mismatch; rebuild with python manuscript/build.py")
    pages = record.get("main_body_pages")
    if type(pages) is not int or not 1 <= pages <= 8:
        raise ValueError("invalid recorded main-body page check")
    return record


def publish_review_pdf(pdf, inputs, pages):
    """Replace the visible copy only after every build/evidence check succeeds."""
    data = pdf.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise ValueError("build output is not a PDF")
    record = dict(schema=REVIEW_SCHEMA, pdf="TikiTorch.pdf",
                  pdf_sha256=hashlib.sha256(data).hexdigest(),
                  inputs_sha256=inputs, main_body_pages=pages,
                  scope="repository input freshness and checked build; not scientific, release or venue clearance")
    REVIEW_RECORD.parent.mkdir(parents=True, exist_ok=True)
    # Prepare both files before replacing either; each individual replacement
    # is atomic. A process interruption between replacements fails --check.
    with tempfile.TemporaryDirectory(prefix="review-", dir=PAPER / "build") as staging:
        staged_pdf = Path(staging) / "TikiTorch.pdf"
        staged_record = Path(staging) / "review_pdf.json"
        staged_pdf.write_bytes(data)
        staged_record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staged_pdf, REVIEW_PDF)
        os.replace(staged_record, REVIEW_RECORD)


def install_style(blob, manifest, destination):
    if hashlib.sha256(blob).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("official template archive changed; review before updating its pinned digest")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        contents = {name: archive.read(name) for name in manifest["members"]}
    for name, data in contents.items():
        if Path(name).name != name or hashlib.sha256(data).hexdigest() != manifest["members"][name]:
            raise ValueError(f"invalid template member: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (destination / name).write_bytes(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the review PDF and repository inputs without LaTeX or network access")
    parser.add_argument("--fetch-style", action="store_true", help="download the pinned official template")
    parser.add_argument("--template-zip", type=Path, help="use an already-downloaded official archive")
    args = parser.parse_args(argv)
    if args.fetch_style and args.template_zip:
        parser.error("choose --fetch-style or --template-zip")
    if args.check:
        if args.fetch_style or args.template_zip:
            parser.error("--check cannot fetch or install styles")
        check_review_pdf()
        print(f"Review PDF and source-bound evidence are current: {REVIEW_PDF}")
        print("This is a freshness/artifact check, not scientific or publication clearance.")
        return 0
    manifest = json.loads((SOURCE / "template_manifest.json").read_text())
    destination = PAPER / "build" / "style"
    if args.fetch_style:
        with urlopen(manifest["archive_url"], timeout=30) as response:
            blob = response.read(2 * 1024 * 1024 + 1)
        if len(blob) > 2 * 1024 * 1024:
            raise ValueError("unexpected template size")
        install_style(blob, manifest, destination)
    elif args.template_zip:
        install_style(args.template_zip.read_bytes(), manifest, destination)
    for name, expected in manifest["members"].items():
        path = destination / name
        if not path.exists():
            parser.error("style missing; run with --fetch-style or --template-zip")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"modified venue style: {name}")
    if not shutil.which("latexmk"):
        parser.error("latexmk/LaTeX is required (MacTeX or TeX Live)")
    check_evidence()
    inputs = review_inputs()
    subprocess.run(["latexmk", "-pdf", "main.tex"], cwd=SOURCE, check=True)
    build = PAPER / "build"
    log = (build / "main.log").read_text(errors="replace")
    if re.search(r"(?:Citation|Reference) .* undefined|There were undefined|multiply defined", log):
        raise RuntimeError("unresolved or duplicate manuscript references")
    if re.search(r"Overfull \\[hv]box", log):
        raise RuntimeError("overfull manuscript boxes; inspect layout before sharing")
    aux = (build / "main.aux").read_text()
    match = re.search(r"\\newlabel\{main:end\}\{\{[^}]*\}\{(\d+)\}", aux)
    if match is None or not 1 <= int(match.group(1)) <= 8:
        raise RuntimeError("main-body page marker missing or exceeds the provisional eight-page limit")
    pdf = build / "main.pdf"
    if pdf.stat().st_size > 20 * 1024 * 1024:
        raise RuntimeError("PDF exceeds conservative 20 MiB packaging limit")
    check_evidence()
    if review_inputs() != inputs:
        raise RuntimeError("manuscript inputs changed during the build; review PDF not replaced")
    publish_review_pdf(pdf, inputs, int(match.group(1)))
    print(f"Review PDF: {REVIEW_PDF}\nMain body ends on page {match.group(1)}; references resolved; styles unmodified.")
    print("This is a format/artifact check, not a passed scientific or publication-readiness gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
