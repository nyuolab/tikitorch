"""Offline-only tests of the first paired FPGA pilot's generated guest contract."""
from datetime import datetime, timedelta, timezone
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import tarfile
import zipfile

import pytest

from scripts import fpga_lm_aws as cloud
from scripts import fpga_lm_guest as G


@pytest.fixture
def config():
    return {
        "profile": "research", "account": "123456789012", "region": "us-east-1",
        "ami": "ami-0123456789abcdef0", "ami_owner": "210987654321",
        "agfi": "agfi-0123456789abcdef0", "bucket": "tiki-private-research",
        "run_prefix": "tikitorch-fpga-lm/pilot-20260909-test",
        "subnet": "subnet-0123456789abcdef0", "security_group": "sg-0123456789abcdef0",
        "instance_profile": "arn:aws:iam::123456789012:instance-profile/tiki-pilot",
        "sdk_root": "/opt/aws-fpga", "sdk_revision": "a" * 40, "sdk_sha256": "d" * 64,
        "sdk_s3_key": "tikitorch-fpga-lm/pilot-20260909-test/sdk.tar",
        "numpy_s3_key": "tikitorch-fpga-lm/pilot-20260909-test/numpy-wheels.tar",
        "numpy_archive_sha256": "e" * 64,
        "numpy_wheels": [{"python_tag": tag,
                          "filename": f"numpy-{G.numpy_version_for_tag(tag)}-{tag}-{tag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                          "sha256": "f" * 64} for tag in ("cp38", "cp310", "cp311", "cp312")],
        "source_sha256": "b" * 64, "corpus_sha256": "c" * 64,
        "corpus_key": "datasets/approved.txt", "corpus_license": "Public domain; approved corpus",
        "instance_type": "f2.6xlarge", "max_minutes": 90, "root_volume_gib": 50,
        "hourly_usd": "1.98", "storage_ip_artifact_allowance_usd": "7.03",
        "price_checked_on": datetime.now(timezone.utc).date().isoformat(),
        "price_source_url": "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json",
        "steps": 64, "eval_every": 32, "eval_windows": 2,
        "deadline_utc": (datetime.now(timezone.utc) + timedelta(minutes=90)).isoformat(),
        "watchdog_execution_arn": "arn:aws:states:us-east-1:123456789012:execution:tiki-watchdog:pilot-test",
    }


@pytest.fixture
def ledger(config):
    return {"account": config["account"], "currency": "USD", "ceiling_usd": "1000", "runs": []}


def gate_fixture(config, ledger):
    expected = {key: config[key] for key in ("account", "region", "run_prefix", "source_sha256", "deadline_utc", "watchdog_execution_arn")}
    expected["client_token"] = cloud.plan(config, ledger)["proposed_run_instances"]["ClientToken"]
    identity = {"instanceId": "i-0123456789abcdef0", "accountId": config["account"], "region": config["region"]}
    gate = {**expected, "authorized": True, "instance_id": identity["instanceId"]}
    return gate, expected, identity


def test_current_gate_is_bound_to_identity_and_controller(config, ledger):
    gate, expected, identity = gate_fixture(config, ledger)
    assert G.verify_start_gate(gate, expected, identity, datetime.now(timezone.utc)) == gate
    with pytest.raises(ValueError, match="schema"):
        G.verify_start_gate({**gate, "unknown": True}, expected, identity, datetime.now(timezone.utc))


@pytest.mark.parametrize("field", ["authorized", "account", "region", "run_prefix", "client_token", "source_sha256", "deadline_utc", "watchdog_execution_arn", "instance_id"])
def test_start_gate_rejects_every_foreign_or_missing_binding(config, ledger, field):
    gate, expected, identity = gate_fixture(config, ledger)
    del gate[field]
    with pytest.raises(ValueError):
        G.verify_start_gate(gate, expected, identity, datetime.now(timezone.utc))


@pytest.mark.parametrize("field", ["accountId", "region", "instanceId"])
def test_start_gate_rejects_wrong_imds_identity(config, ledger, field):
    gate, expected, identity = gate_fixture(config, ledger)
    identity[field] = "other"
    with pytest.raises(ValueError):
        G.verify_start_gate(gate, expected, identity, datetime.now(timezone.utc))


def test_start_gate_rejects_expired_and_naive_deadline(config, ledger):
    gate, expected, identity = gate_fixture(config, ledger)
    for stamp in ("2000-01-01T00:00:00Z", "2026-09-09T00:00:00"):
        gate["deadline_utc"] = expected["deadline_utc"] = stamp
        with pytest.raises(ValueError):
            G.verify_start_gate(gate, expected, identity, datetime.now(timezone.utc))


AFI = "AFI 0 agfi-0123456789abcdef0 loaded 0 ok 0 0x10212415"


def test_exact_loaded_image():
    assert G.verify_image_text("header\n" + AFI + "\nAFIDEVICE ...", "agfi-0123456789abcdef0")["status"] == "loaded"


@pytest.mark.parametrize("text", ["", "log says loaded agfi-0123456789abcdef0", AFI + "\n" + AFI,
    AFI.replace("AFI 0", "AFI 1"), AFI.replace("loaded", "cleared"),
    AFI.replace("abcdef0", "abcdef1"), AFI.replace("loaded 0", "loaded 1"),
    AFI.replace("ok 0", "ok 1"), AFI.replace("ok", "error"), AFI.replace("0x10212415", "0x10212414")])
def test_image_gate_rejects_ambiguous_or_wrong_state(text):
    with pytest.raises(ValueError):
        G.verify_image_text(text, "agfi-0123456789abcdef0")


def write_tar(path, payloads, *, special=None):
    with tarfile.open(path, "w") as archive:
        for name, data in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(member, io.BytesIO(data))
        if special is not None:
            archive.addfile(special)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sdk_payloads():
    return {"aws-fpga/sdk/userspace/mkall_fpga_mgmt_tools.sh": b"#!/bin/bash\nexit 0\n",
            "aws-fpga/shared/bin/set_common_functions.sh": b"# pinned functions\n",
            "aws-fpga/release_version.txt": b"2.3.4\n", "aws-fpga/LICENSE.txt": b"license\n"}


def test_sdk_extract_is_hash_bound_allowlisted_and_preserves_executable_bits(tmp_path):
    archive = tmp_path / "sdk.tar"
    digest = write_tar(archive, sdk_payloads())
    sdk = Path(G.extract_sdk_archive(archive, tmp_path / "sdk", digest))
    assert (sdk / "release_version.txt").read_text() == "2.3.4\n"
    assert (sdk / "sdk/userspace/mkall_fpga_mgmt_tools.sh").stat().st_mode & 0o111
    assert not (sdk / "LICENSE.txt").stat().st_mode & 0o111
    with pytest.raises(FileExistsError):
        G.extract_sdk_archive(archive, tmp_path / "sdk", digest)


@pytest.mark.parametrize("name", ["/tmp/evil", "aws-fpga/../evil", "aws-fpga/hdk/evil", "other/sdk/evil", "aws-fpga/release_version.txt/evil", "aws-fpga/LICENSE.txt/evil"])
def test_sdk_extract_rejects_paths(tmp_path, name):
    archive = tmp_path / "sdk.tar"
    digest = write_tar(archive, {**sdk_payloads(), name: b"bad"})
    with pytest.raises(ValueError):
        G.extract_sdk_archive(archive, tmp_path / "sdk", digest)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_sdk_extract_rejects_links_and_special_files(tmp_path, kind):
    archive = tmp_path / "sdk.tar"
    member = tarfile.TarInfo("aws-fpga/sdk/unsafe")
    member.type = kind
    member.linkname = "/etc/passwd"
    digest = write_tar(archive, sdk_payloads(), special=member)
    with pytest.raises(ValueError):
        G.extract_sdk_archive(archive, tmp_path / "sdk", digest)


def test_sdk_rejects_wrong_archive_hash_and_missing_license(tmp_path):
    archive = tmp_path / "sdk.tar"
    payloads = sdk_payloads()
    del payloads["aws-fpga/LICENSE.txt"]
    digest = write_tar(archive, payloads)
    with pytest.raises(ValueError, match="SHA256"):
        G.extract_sdk_archive(archive, tmp_path / "first", "0" * 64)
    with pytest.raises(ValueError, match="required"):
        G.extract_sdk_archive(archive, tmp_path / "second", digest)


def test_source_extract_binds_manifest_and_exact_allowlist(tmp_path):
    names = ["LICENSE", "scripts/run.py"]
    payloads = {name: name.encode() for name in names}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    payloads["SOURCE_MANIFEST.json"] = json.dumps(manifest).encode()
    archive = tmp_path / "source.tar"
    digest = write_tar(archive, payloads)
    assert G.extract_verified_archive(archive, tmp_path / "source", names, digest) == manifest
    with pytest.raises(ValueError, match="allowlist"):
        G.extract_verified_archive(archive, tmp_path / "foreign", ["LICENSE"], digest)
    payloads["LICENSE"] = b"changed"
    digest = write_tar(archive, payloads)
    with pytest.raises(ValueError, match="member SHA256"):
        G.extract_verified_archive(archive, tmp_path / "changed", names, digest)


def test_bundle_is_portable_complete_and_readback_verified(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    for backend in ("csim", "pci"):
        (root / backend).mkdir()
        (root / backend / "checkpoint.i32").write_bytes(b"\0\1\2\3")
    (root / "guest.log").write_text("mutable log excluded")
    bundle = tmp_path / "results.tar.gz"
    receipt = G.bundle_artifacts(root, bundle, tmp_path / "receipt.json")
    assert G.verify_bundle_readback(bundle, receipt)
    with tarfile.open(bundle) as archive:
        assert set(archive.getnames()) == {"csim/checkpoint.i32", "pci/checkpoint.i32", "artifact_manifest.json"}
        assert all(member.isfile() for member in archive.getmembers())
        manifest_data = archive.extractfile("artifact_manifest.json").read()
        assert hashlib.sha256(manifest_data).hexdigest() == receipt["artifact_manifest_sha256"]
        assert set(json.loads(manifest_data)) == {"csim/checkpoint.i32", "pci/checkpoint.i32"}
    with pytest.raises(FileExistsError):
        G.bundle_artifacts(root, bundle, tmp_path / "receipt2.json")
    bundle.write_bytes(bundle.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="readback"):
        G.verify_bundle_readback(bundle, receipt)


def test_artifacts_reject_symlinks_and_empty_output(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        G.artifact_manifest(tmp_path)
    (tmp_path / "outside").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        G.artifact_manifest(tmp_path)


def test_conditional_s3_write_capability_has_no_legacy_fallback():
    assert G.verify_put_object_schema({"Bucket": "", "IfNoneMatch": ""})
    for schema in ({"Bucket": ""}, None, []):
        with pytest.raises(ValueError, match="failed setup.*no overwrite fallback"):
            G.verify_put_object_schema(schema)


def runtime_identity(minor=11, **overrides):
    return {"implementation": "cpython", "cache_tag": f"cpython-3{minor}",
            "version": (3, minor), "platform": "linux", "machine": "x86_64", **overrides}


@pytest.mark.parametrize("minor", [8, 10, 11, 12])
def test_numpy_selects_exact_supported_python_abi(config, minor):
    selected = G.select_numpy_wheel(config["numpy_wheels"], runtime_identity(minor))
    assert selected["python_tag"] == f"cp3{minor}"


@pytest.mark.parametrize("overrides", [{"implementation": "pypy"}, {"cache_tag": "cpython-310"},
    {"version": (3, 13)}, {"platform": "darwin"}, {"machine": "aarch64"}, {"abiflags": "d"}])
def test_numpy_rejects_unsupported_runtime_before_extract(config, overrides):
    with pytest.raises(ValueError, match="unsupported NumPy runtime"):
        G.select_numpy_wheel(config["numpy_wheels"], runtime_identity(**overrides))


def fake_numpy_archive(tmp_path, *, extra=None, metadata=None):
    records, payloads = [], {}
    for tag in ("cp38", "cp310", "cp311", "cp312"):
        version = G.numpy_version_for_tag(tag)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("numpy/__init__.py", f"__version__ = '{version}'\n")
            archive.writestr(f"numpy-{version}.dist-info/METADATA", metadata or f"Name: numpy\nVersion: {version}\n".encode())
            archive.writestr(f"numpy-{version}.dist-info/WHEEL", "Wheel-Version: 1.0\n")
            archive.writestr("numpy.libs/libexample.so", b"preserve bundled-library path")
            if extra is not None:
                archive.writestr(extra, b"invalid member")
        data = buffer.getvalue()
        filename = f"numpy-{version}-{tag}-{tag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        records.append({"python_tag": tag, "filename": filename, "sha256": hashlib.sha256(data).hexdigest()})
        payloads[filename] = data
    path = tmp_path / "numpy-wheels.tar"
    digest = write_tar(path, payloads)
    return path, digest, records


def test_numpy_verified_extraction_preserves_bundled_libraries_and_never_overwrites(tmp_path):
    path, digest, records = fake_numpy_archive(tmp_path)
    selected = G.select_numpy_wheel(records, runtime_identity())
    target = tmp_path / "runtime"
    assert G.extract_numpy_runtime(path, target, digest, records, selected) == selected
    assert (target / "numpy.libs/libexample.so").read_bytes() == b"preserve bundled-library path"
    with pytest.raises(FileExistsError):
        G.extract_numpy_runtime(path, target, digest, records, selected)


@pytest.mark.parametrize("name", ["/tmp/evil", "numpy/../evil", "numpy//evil", "numpy/./evil",
    "numpy\\evil", "other/evil"])
def test_numpy_rejects_unsafe_zip_paths_before_writing(tmp_path, name):
    path, digest, records = fake_numpy_archive(tmp_path, extra=name)
    with pytest.raises(ValueError, match="unsafe path"):
        G.extract_numpy_runtime(path, tmp_path / "runtime", digest, records, records[0])
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("kind", [0o120777, 0o010600, 0o020600])
def test_numpy_rejects_zip_symlinks_and_special_files(tmp_path, kind):
    member = zipfile.ZipInfo("numpy/unsafe")
    member.create_system = 3
    member.external_attr = kind << 16
    path, digest, records = fake_numpy_archive(tmp_path, extra=member)
    with pytest.raises(ValueError, match="unsafe path"):
        G.extract_numpy_runtime(path, tmp_path / "runtime", digest, records, records[0])


def test_numpy_rejects_duplicate_zip_members_and_wrong_version(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        path, digest, records = fake_numpy_archive(tmp_path, extra="numpy/__init__.py")
    with pytest.raises(ValueError, match="duplicate"):
        G.extract_numpy_runtime(path, tmp_path / "duplicate", digest, records, records[0])
    path, digest, records = fake_numpy_archive(tmp_path, metadata=b"Name: numpy\nVersion: 1.26.4\n")
    with pytest.raises(ValueError, match="metadata version"):
        G.extract_numpy_runtime(path, tmp_path / "old", digest, records, records[0])


def test_numpy_checks_archive_and_every_wheel_even_when_not_selected(tmp_path):
    path, digest, records = fake_numpy_archive(tmp_path)
    with pytest.raises(ValueError, match="archive SHA256"):
        G.extract_numpy_runtime(path, tmp_path / "bad_archive", "0" * 64, records, records[0])
    records[2]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="wheel SHA256"):
        G.extract_numpy_runtime(path, tmp_path / "bad_unselected", digest, records, records[0])


def test_numpy_archive_rejects_unexpected_and_link_members(tmp_path):
    path, digest, records = fake_numpy_archive(tmp_path)
    with tarfile.open(path) as archive:
        payloads = {member.name: archive.extractfile(member).read() for member in archive.getmembers()}
    digest = write_tar(path, {**payloads, "unexpected.txt": b"extra"})
    with pytest.raises(ValueError, match="allowlist"):
        G.extract_numpy_runtime(path, tmp_path / "extra", digest, records, records[0])
    member = tarfile.TarInfo(records[0]["filename"])
    member.type = tarfile.SYMTYPE
    member.linkname = records[1]["filename"]
    del payloads[member.name]
    digest = write_tar(path, payloads, special=member)
    with pytest.raises(ValueError, match="type/size"):
        G.extract_numpy_runtime(path, tmp_path / "link", digest, records, records[0])


def test_numpy_rejects_excessive_expanded_zip_size_before_writes(tmp_path, monkeypatch):
    path, digest, records = fake_numpy_archive(tmp_path)
    original = zipfile.ZipFile.infolist
    def inflated_members(archive):
        members = original(archive)
        members[0].file_size = 257 * 1024 * 1024
        return members
    monkeypatch.setattr(zipfile.ZipFile, "infolist", inflated_members)
    with pytest.raises(ValueError, match="expanded size"):
        G.extract_numpy_runtime(path, tmp_path / "runtime", digest, records, records[0])
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("mutation", ["tag", "filename", "hash", "extra", "missing"])
def test_numpy_wheel_config_fails_closed(config, mutation):
    wheels = config["numpy_wheels"]
    if mutation == "tag":
        wheels[0]["python_tag"] = "cp311"
    elif mutation == "filename":
        wheels[0]["filename"] = wheels[0]["filename"].replace("1.24.4", "1.24.3")
    elif mutation == "hash":
        wheels[0]["sha256"] = "unknown"
    elif mutation == "extra":
        wheels[0]["url"] = "https://unapproved.invalid"
    else:
        wheels.pop()
    with pytest.raises(ValueError):
        G.validate_numpy_wheels(wheels)


def test_render_has_valid_shell_and_every_embedded_python_block_compiles(config, ledger):
    script = G.render(config, ledger)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True, capture_output=True)
    blocks = re.findall(r"<<'(PY_[A-Z_]+)'\n(.*?)\n\1", script, flags=re.S)
    assert len(blocks) >= 10
    for name, source in blocks:
        ast.parse(source, filename=name, feature_version=(3, 8))
    assert "_FUNCTION" not in script and "EXPECTED_GATE" not in script
    # EC2 permits 16 KiB raw user data. The host must gzip this larger script;
    # cloud-init recognizes gzip and the compressed bytes stay within the cap.
    assert len(gzip.compress(script.encode(), mtime=0)) < 16 * 1024


def test_guest_and_frozen_scientific_python_sources_parse_for_python38(config, ledger):
    root = Path(__file__).resolve().parents[1]
    for name in ("scripts/fpga_lm_guest.py", "scripts/fpga_lm_pretrain.py",
                 "tikitorch/fpga/silicon/gen_buffer_init.py"):
        ast.parse((root / name).read_text(), filename=name, feature_version=(3, 8))
    script = G.render(config, ledger)
    assert "is_relative_to(" not in script
    assert script.index('"artifacts/preflight_runtime.json"') < script.index("selected = select_numpy_wheel(")


def test_all_pretrainer_commands_use_absolute_script_path(config, ledger):
    script = G.render(config, ledger)
    commands = [line for line in script.splitlines() if line.startswith("run_for ")
                and "fpga_lm_pretrain.py" in line]
    assert len(commands) == 4
    assert all('python3 "$WORK/source/scripts/fpga_lm_pretrain.py"' in line for line in commands)
    assert {next(flag for flag in ("--build-csim", "--backend csim", "--backend pci", "--compare-runs")
                 if flag in line) for line in commands} == {
                     "--build-csim", "--backend csim", "--backend pci", "--compare-runs"}
    assert script.index('cd "$WORK/source"') < script.index(commands[0])


def test_pci_cpp_build_does_not_shadow_standard_cstdint(config, ledger):
    script = G.render(config, ledger)
    common = re.search(r"COMMON=\((.*?)\)\n", script, flags=re.S).group(1)
    compile_command = re.search(r"run_for 120 g\+\+ (.*?)\nsha256sum", script, flags=re.S).group(1)
    assert "cstdint_shim" not in common + compile_command
    assert "-DTT_FPGA_PCI" in compile_command
    assert "tikitorch/fpga/bridge.cpp scripts/fpga_lm_layout.cpp" in compile_command
    assert '-I"$SDK_ROOT/sdk/userspace/include"' in compile_command
    assert all(flag in common for flag in ("-std=c++17", "-DML_D=512", "-DTT_RESIDENT_W",
                                           "-DTT_ONCHIP_GRAD", "-DTT_URAM_W"))


def test_python38_relative_file_failure_and_absolute_csim_source_identity(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    # run_path preserves the supplied __file__ on modern Python too, reproducing
    # Python3.8's script-launch behavior without changing the frozen runner.
    relative = runpy.run_path("scripts/fpga_lm_pretrain.py", run_name="_relative_path_regression")
    assert relative["__file__"] == "scripts/fpga_lm_pretrain.py"
    with pytest.raises(ValueError):
        relative["source_identity"]()
    absolute = runpy.run_path(str(root / "scripts/fpga_lm_pretrain.py"), run_name="_absolute_path_regression")
    expected = absolute["source_identity"]()
    assert expected["scripts/fpga_lm_pretrain.py"] == hashlib.sha256(
        (root / "scripts/fpga_lm_pretrain.py").read_bytes()).hexdigest()
    assert len(expected) == 8
    if not (shutil.which("clang++") or shutil.which("g++")):
        pytest.skip("absolute source identity verified; C++ compiler unavailable for bounded CSIM build")
    # Compile the exact frozen D16 source only; do not execute training or open
    # a device. The sidecar binds the same eight source files and their bytes.
    library = absolute["build_csim"](tmp_path / "absolute-csim", width=16)
    build = json.loads((library.parent / "build.json").read_text())
    assert library.stat().st_size > 0
    assert build["source_hashes"] == expected
    assert build["library_sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert absolute["source_identity"]() == expected


def test_python38_selects_legacy_version_and_dist_info(tmp_path):
    path, digest, records = fake_numpy_archive(tmp_path)
    selected = G.select_numpy_wheel(records, runtime_identity(8))
    target = tmp_path / "runtime"
    G.extract_numpy_runtime(path, target, digest, records, selected)
    assert G.numpy_version_for_tag(selected["python_tag"]) == "1.24.4"
    assert (target / "numpy-1.24.4.dist-info/METADATA").is_file()
    assert not (target / "numpy-2.2.6.dist-info").exists()


def test_render_orders_gates_checked_backend_and_durable_terminal_upload(config, ledger):
    script = G.render(config, ledger)
    assert script.index("PY_START_GATE") < script.index('aws s3 cp "$S3_SDK"')
    assert script.index("--generate-cli-skeleton input") < script.index('aws s3 cp "$S3_SDK"')
    assert script.index("PY_NUMPY_SELECT") < script.index('aws s3 cp "$S3_NUMPY"')
    assert script.index("PY_NUMPY_VERIFY") < script.index("mkall_fpga_mgmt_tools.sh 0 64")
    assert 'export PYTHONPATH="$WORK/runtime" PYTHONNOUSERSITE=1' in script
    assert "numpy_runtime.json" in script and "numpy_archive_sha256" in script
    assert script.index('shutdown -h +"$MAX_MINUTES"') < script.index("PY_START_GATE")
    assert script.index("--backend csim") < script.index('"$FPGA_CMD" LoadFpgaImage -S 0 -I "$AGFI"')
    assert script.index("image_before.txt") < script.index("--backend pci") < script.index("image_after.txt")
    assert script.index("--backend pci") < script.index("--compare-runs")
    assert "tt_fpga_last_error" in script and "tt_morph_lm_backend() != 2" in script
    assert "TT_KICK_SPIN=20000000" in script and "run_for 300 python3" in script
    assert "--steps 64 --eval-every 32 --eval-windows 2" in script
    assert '--output "$WORK/artifacts/pci"' in script
    assert '"$SDK_LIB/libfpga_mgmt.a" -lrt -lpthread' in script
    assert '"$FPGA_CMD" DescribeFpgaImage -S 0' in script
    assert "ln -s" not in script
    assert "git -" not in script and "apt-get" not in script and "pip install" not in script
    assert " -F" not in script and "libfpga_mgmt.so" not in script
    assert "sdk_sha256" in script and "sdk_revision" in script
    assert "AWS_SHARED_CREDENTIALS_FILE=/dev/null" in script
    assert "unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID" in script
    assert "unset AWS_EC2_METADATA_SERVICE_ENDPOINT" in script and "${!AWS_ENDPOINT_URL@}" in script
    assert script.count("--if-none-match '*'") == 2
    assert script.index('"$RESULTS_PREFIX/results.tar.gz"') < script.index('"$RESULTS_PREFIX/completion.json"') < script.index("shutdown -h now")
    assert script.index("verify_bundle_readback(sys.argv") < script.index("'PY_TERMINAL'")
    assert " - 300" in script and 'timeout --signal=TERM --kill-after=10 "$WORKLOAD_SECONDS"' in script


@pytest.mark.parametrize("key,value", [("sdk_sha256", "bad"), ("sdk_s3_key", "other/sdk.tar"),
    ("deadline_utc", "2026-09-09T00:00:00"), ("watchdog_execution_arn", "arn:other"),
    ("steps", 63), ("eval_every", 16), ("eval_windows", 1), ("max_minutes", 7)])
def test_render_fails_closed_for_unapproved_inputs(config, ledger, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        G.render(config, ledger)
