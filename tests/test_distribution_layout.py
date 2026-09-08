"""Explicit library/research packaging boundaries, without rebuilding per test."""

import io
from pathlib import Path
import re
import stat
import tarfile
import zipfile

import pytest

from scripts import check_distribution_layout as distributions


ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_ONLY = ("manuscript", "papers", "scripts", "tests")


@pytest.mark.parametrize("directory", CHECKOUT_ONLY)
def test_source_manifest_excludes_checkout_only_trees(directory):
    commands = [line.split("#", 1)[0].strip()
                for line in (ROOT / "MANIFEST.in").read_text().splitlines()]
    assert f"prune {directory}" in commands


def test_wheel_discovery_is_limited_to_the_library_namespace():
    # Read the small configuration section directly so Python 3.9/3.10 tests do
    # not need a TOML dependency just for this packaging-intent regression.
    config = (ROOT / "pyproject.toml").read_text()
    discovery = re.search(
        r"(?ms)^\[tool\.setuptools\.packages\.find\]\s*\n(.*?)(?=^\[|\Z)", config,
    )
    assert discovery is not None
    assert re.search(r'^include\s*=\s*\["tikitorch\*"\]\s*$',
                     discovery.group(1), re.MULTILINE)


def test_manifest_documents_the_source_checkout_test_requirement():
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "Git source checkout" in manifest


def make_archive(tmp_path, kind, names, *, linked=False):
    prefix = "tikitorch-0.2.0"
    if kind == "wheel":
        archive = tmp_path / "tikitorch-0.2.0-py3-none-any.whl"
        with zipfile.ZipFile(archive, "w") as handle:
            for name in names:
                member = zipfile.ZipInfo(name)
                member.create_system = 3
                member.external_attr = ((stat.S_IFLNK if linked else stat.S_IFREG) | 0o644) << 16
                handle.writestr(member, "fixture")
    else:
        archive = tmp_path / f"{prefix}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for name in names:
                member = tarfile.TarInfo(f"{prefix}/{name}")
                member.size = 7
                if linked:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/outside"
                handle.addfile(member, io.BytesIO(b"fixture"))
    return archive


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_built_archive_allows_library_and_nested_hardware_build_sources(tmp_path, kind):
    archive = make_archive(tmp_path, kind, [
        "tikitorch/__init__.py", "tikitorch/hardware/hdk/core/build/scripts/synth.tcl",
    ])
    assert distributions.inspect_archive(archive)["members"] == 2


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("directory", ["manuscript", "papers", "scripts", "tests", "results"])
def test_built_archive_rejects_checkout_only_trees(tmp_path, kind, directory):
    archive = make_archive(tmp_path, kind, [f"{directory}/artifact.txt"])
    with pytest.raises(ValueError, match="checkout-only tree"):
        distributions.inspect_archive(archive)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("name,match", [
    ("tikitorch/.env", "environment/cache"),
    ("tikitorch/.env.local", "environment/cache"),
    ("tikitorch/__pycache__/module.pyc", "environment/cache"),
    ("tikitorch/.pytest_cache/data", "environment/cache"),
    ("tikitorch/module.pyc", "bytecode"),
    ("../outside", "unsafe"),
    ("tikitorch/../../outside", "unsafe"),
    ("tikitorch\\outside", "unsafe"),
])
def test_built_archive_rejects_local_state_and_traversal(tmp_path, kind, name, match):
    archive = make_archive(tmp_path, kind, [name])
    with pytest.raises(ValueError, match=match):
        distributions.inspect_archive(archive)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_built_archive_rejects_symlinks_without_extracting(tmp_path, kind):
    archive = make_archive(tmp_path, kind, ["tikitorch/link"], linked=True)
    with pytest.raises(ValueError, match="linked archive member"):
        distributions.inspect_archive(archive)
    assert not (tmp_path / "tikitorch").exists()


def test_distribution_cli_fails_on_invalid_archive(tmp_path, capsys):
    archive = tmp_path / "broken.whl"
    archive.write_bytes(b"not a ZIP archive")
    assert distributions.main([str(archive)]) == 1
    assert '"layout_valid": false' in capsys.readouterr().out
