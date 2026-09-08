#!/usr/bin/env python3
"""Build the unified manuscript; style integrity and page checks are not scientific clearance."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import urlopen
import zipfile

PAPER = Path(__file__).resolve().parent


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
    parser.add_argument("--fetch-style", action="store_true", help="download the pinned official template")
    parser.add_argument("--template-zip", type=Path, help="use an already-downloaded official archive")
    args = parser.parse_args(argv)
    if args.fetch_style and args.template_zip:
        parser.error("choose --fetch-style or --template-zip")
    manifest = json.loads((PAPER / "template_manifest.json").read_text())
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
    subprocess.run([sys.executable, str(PAPER / "generate_tables.py"), "--check"], check=True)
    subprocess.run([sys.executable, str(PAPER / "measure_retained_state.py"), "--check"], check=True)
    subprocess.run(["latexmk", "-pdf", "main.tex"], cwd=PAPER, check=True)
    build = PAPER / "build"
    log = (build / "main.log").read_text(errors="replace")
    if re.search(r"(?:Citation|Reference) .* undefined|There were undefined|multiply defined", log):
        raise RuntimeError("unresolved or duplicate manuscript references")
    if re.search(r"Overfull \\[hv]box", log):
        raise RuntimeError("overfull manuscript boxes; inspect layout before sharing")
    aux = (build / "main.aux").read_text()
    match = re.search(r"\\newlabel\{main:end\}\{\{[^}]*\}\{(\d+)\}", aux)
    if match is None or int(match.group(1)) > 8:
        raise RuntimeError("main-body page marker missing or exceeds the provisional eight-page limit")
    pdf = build / "main.pdf"
    if pdf.stat().st_size > 20 * 1024 * 1024:
        raise RuntimeError("PDF exceeds conservative 20 MiB packaging limit")
    print(f"PDF: {pdf}\nMain body ends on page {match.group(1)}; references resolved; styles unmodified.")
    print("This is a format/artifact check, not a passed scientific or publication-readiness gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
