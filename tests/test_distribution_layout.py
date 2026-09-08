"""Explicit library/research packaging boundaries, without rebuilding per test."""

import io
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import zipfile

import pytest

from scripts import check_distribution_layout as distributions


ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_ONLY = ("manuscript", "papers", "scripts", "tests", "results")


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


def test_source_manifest_excludes_environment_variants_and_generated_roots():
    commands = [line.split("#", 1)[0].split()
                for line in (ROOT / "MANIFEST.in").read_text().splitlines()]
    excluded = {pattern for command in commands if command and command[0] == "global-exclude"
                for pattern in command[1:]}
    assert {".env", ".env.*", ".DS_Store", "*.py[cod]"} <= excluded
    for directory in ("build", "dist", "site", ".venv"):
        assert ["prune", directory] in commands


def test_git_ignores_environment_variants_but_allows_reviewed_templates():
    # Synthetic filenames only: no credentials or local environment files are read.
    ignored = {".env", ".env.local", "tikitorch/example/.env.production"}
    templates = {".env.example", "tikitorch/example/.env.template"}
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"], cwd=ROOT,
        input="\n".join(sorted(ignored | templates)) + "\n", text=True,
        capture_output=True, check=True,
    )
    assert set(result.stdout.splitlines()) == ignored


def make_archive(tmp_path, kind, names, *, linked=False, member_kind="file"):
    prefix = "tikitorch-0.2.0"
    if linked:
        member_kind = "symlink"
    if kind == "wheel":
        archive = tmp_path / "tikitorch-0.2.0-py3-none-any.whl"
        with zipfile.ZipFile(archive, "w") as handle:
            for name in names:
                member = zipfile.ZipInfo(name)
                member.create_system = 3
                mode = {"file": stat.S_IFREG, "directory": stat.S_IFDIR,
                        "symlink": stat.S_IFLNK, "fifo": stat.S_IFIFO,
                        "character": stat.S_IFCHR, "block": stat.S_IFBLK,
                        "socket": stat.S_IFSOCK, "untyped": 0}[member_kind]
                member.external_attr = (mode | 0o644) << 16
                handle.writestr(member, "fixture")
    else:
        archive = tmp_path / f"{prefix}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for name in names:
                member = tarfile.TarInfo(f"{prefix}/{name}")
                member.type = {"file": tarfile.REGTYPE, "directory": tarfile.DIRTYPE,
                               "symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
                               "fifo": tarfile.FIFOTYPE, "character": tarfile.CHRTYPE,
                               "block": tarfile.BLKTYPE, "socket": b"s"}[member_kind]
                member.size = 7 if member_kind == "file" else 0
                if member_kind in {"symlink", "hardlink"}:
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
def test_built_archive_allows_explicit_directories(tmp_path, kind):
    archive = make_archive(tmp_path, kind, ["tikitorch/"], member_kind="directory")
    assert distributions.inspect_archive(archive)["members"] == 1


def test_built_wheel_allows_regular_members_without_a_unix_type(tmp_path):
    archive = make_archive(tmp_path, "wheel", ["tikitorch/__init__.py"], member_kind="untyped")
    assert distributions.inspect_archive(archive)["members"] == 1


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("directory", ["manuscript", "papers", "scripts", "tests", "results"])
def test_built_archive_rejects_checkout_only_trees(tmp_path, kind, directory):
    archive = make_archive(tmp_path, kind, [f"{directory}/artifact.txt"])
    with pytest.raises(ValueError, match="checkout-only tree"):
        distributions.inspect_archive(archive)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("directory", ["build", "dist", "site"])
def test_built_archive_rejects_generated_root_trees(tmp_path, kind, directory):
    archive = make_archive(tmp_path, kind, [f"{directory}/local.txt"])
    with pytest.raises(ValueError, match="generated root tree"):
        distributions.inspect_archive(archive)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("name,match", [
    ("tikitorch/.env", "environment/cache"),
    ("tikitorch/.env.local", "environment/cache"),
    ("tikitorch/.env.example", "environment/cache"),
    (".venv/bin/python", "environment/cache"),
    ("tikitorch/nested/.venv/pyvenv.cfg", "environment/cache"),
    ("tikitorch/.DS_Store", "environment/cache"),
    (".git/config", "environment/cache"),
    ("tikitorch/__pycache__/module.pyc", "environment/cache"),
    ("tikitorch/.pytest_cache/data", "environment/cache"),
    ("tikitorch/module.pyc", "bytecode"),
    ("../outside", "unsafe"),
    ("tikitorch/../../outside", "unsafe"),
    ("tikitorch\\outside", "unsafe"),
    ("tikitorch//module.py", "unsafe"),
    ("./tikitorch/module.py", "unsafe"),
    ("tikitorch//", "unsafe"),
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


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("member_kind", ["fifo", "character", "block", "socket"])
def test_built_archive_rejects_special_members(tmp_path, kind, member_kind):
    archive = make_archive(tmp_path, kind, ["tikitorch/special"], member_kind=member_kind)
    with pytest.raises(ValueError, match="non-regular archive member"):
        distributions.inspect_archive(archive)
    assert not (tmp_path / "tikitorch").exists()


def test_built_sdist_rejects_hardlinks(tmp_path):
    archive = make_archive(tmp_path, "sdist", ["tikitorch/link"], member_kind="hardlink")
    with pytest.raises(ValueError, match="linked archive member"):
        distributions.inspect_archive(archive)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_built_archive_rejects_duplicate_members(tmp_path, kind):
    names = ["tikitorch/__init__.py"] * 2
    if kind == "wheel":
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive = make_archive(tmp_path, kind, names)
    else:
        archive = make_archive(tmp_path, kind, names)
    with pytest.raises(ValueError, match="duplicate archive member"):
        distributions.inspect_archive(archive)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_built_archive_rejects_duplicate_directory_aliases(tmp_path, kind):
    archive = make_archive(tmp_path, kind, ["tikitorch", "tikitorch/"], member_kind="directory")
    with pytest.raises(ValueError, match="duplicate archive member"):
        distributions.inspect_archive(archive)


@pytest.mark.parametrize("destination", ["purelib", "platlib", "data", "headers", "scripts"])
@pytest.mark.parametrize("path,match", [
    ("manuscript/results/run.json", "checkout-only tree"),
    ("build/generated.py", "generated root tree"),
    ("tikitorch/.env.local", "environment/cache"),
])
def test_wheel_data_destinations_cannot_hide_disallowed_files(tmp_path, destination, path, match):
    archive = make_archive(tmp_path, "wheel", [f"tikitorch-0.2.0.data/{destination}/{path}"])
    with pytest.raises(ValueError, match=match):
        distributions.inspect_archive(archive)


def test_built_wheel_allows_library_data_and_installed_scripts(tmp_path):
    archive = make_archive(tmp_path, "wheel", [
        "tikitorch-0.2.0.data/purelib/tikitorch/__init__.py",
        "tikitorch-0.2.0.data/scripts/tt-closure",
    ])
    assert distributions.inspect_archive(archive)["members"] == 2


def test_built_wheel_rejects_unknown_data_destination(tmp_path):
    archive = make_archive(tmp_path, "wheel", ["tikitorch-0.2.0.data/unknown/file"])
    with pytest.raises(ValueError, match="unknown wheel data destination"):
        distributions.inspect_archive(archive)


def test_distribution_cli_fails_on_invalid_archive(tmp_path, capsys):
    archive = tmp_path / "broken.whl"
    archive.write_bytes(b"not a ZIP archive")
    assert distributions.main([str(archive)]) == 1
    assert '"layout_valid": false' in capsys.readouterr().out
