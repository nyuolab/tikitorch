"""Render a bounded, paired CSIM/PCI guest workload; never launch an instance.

The external controller must reserve budget, bind instance ownership and enforce
termination independently. Guest shutdown is a second backstop, not a billing
cap. Only approved S3 objects are downloaded; no package installation or
unpinned SDK checkout is rendered. The rendered script must be gzip-compressed
before submission as EC2 user data (the uncompressed form exceeds 16 KiB).
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shlex
import sys

try:
    from scripts import fpga_lm_aws as cloud
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import fpga_lm_aws as cloud


def verify_image_text(text, expected_agfi, expected_shell="0x10212415"):
    """Parse the SDK's documented AFI row, not a loose 'loaded' substring."""
    rows = [line.split() for line in text.splitlines() if line.split()[:1] == ["AFI"]]
    if len(rows) != 1:
        raise ValueError("expected exactly one AFI status row")
    row = rows[0]
    if len(row) < 8 or row[1:7] != ["0", expected_agfi, "loaded", "0", "ok", "0"]:
        raise ValueError("slot/image/status/error does not match the approved loaded AFI")
    if row[7] != expected_shell:
        raise ValueError("FPGA shell version does not match the approved image gate")
    return {"slot": 0, "agfi": row[2], "status": row[3], "status_code": 0,
            "error": "ok", "error_code": 0, "shell_version": row[7]}


def extract_verified_archive(archive_path, destination, expected, expected_sha256):
    """Allowlisted regular files only, verified before any source is executed."""
    import hashlib
    import json
    from pathlib import Path
    import tarfile

    path = Path(archive_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("source archive SHA256 mismatch")
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected) | {"SOURCE_MANIFEST.json"}:
            raise ValueError("source archive allowlist mismatch")
        if any(not member.isfile() or member.size > 16 * 1024 * 1024 for member in members):
            raise ValueError("invalid source archive member type/size")
        manifest = json.load(archive.extractfile("SOURCE_MANIFEST.json"))
        if set(manifest) != set(expected):
            raise ValueError("source manifest allowlist mismatch")
        payloads = {}
        for name in expected:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe source member name")
            data = archive.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != manifest[name]:
                raise ValueError("source member SHA256 mismatch")
            payloads[name] = data
        for name, data in payloads.items():
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(data)
        return manifest


def artifact_manifest(directory):
    """Hash scientific artifacts; the live guest log is uploaded separately."""
    import hashlib
    from pathlib import Path

    root = Path(directory)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlinks are forbidden")
        if path.is_file() and path.name not in ("artifact_manifest.json", "guest.log"):
            data = path.read_bytes()
            result[path.relative_to(root).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if not result:
        raise ValueError("empty artifact set")
    return result


def extract_sdk_archive(archive_path, destination, expected_sha256):
    import hashlib
    from pathlib import Path
    import tarfile

    source = Path(archive_path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("SDK archive SHA256 mismatch")
    root = Path(destination)
    root.mkdir(exist_ok=False)
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        names = [member.name.rstrip("/") for member in members]
        if len(names) != len(set(names)) or sum(member.size for member in members) > 128 * 1024 * 1024:
            raise ValueError("invalid SDK archive coverage/size")
        for member in members:
            path = Path(member.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "aws-fpga"
                    or (len(path.parts) > 1 and path.parts[1] not in ("sdk", "shared", "release_version.txt", "LICENSE.txt"))
                    or (len(path.parts) == 1 and not member.isdir())
                    or (len(path.parts) > 1 and path.parts[1] in ("release_version.txt", "LICENSE.txt")
                        and (len(path.parts) != 2 or not member.isfile()))
                    or not (member.isfile() or member.isdir()) or member.size > 16 * 1024 * 1024):
                raise ValueError("SDK archive has a link, unsafe path, or unexpected member")
        for member in members:
            output = root / member.name
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream:
                    stream.write(archive.extractfile(member).read())
                output.chmod(0o755 if member.mode & 0o111 else 0o644)
    sdk = root / "aws-fpga"
    for required in ("sdk/userspace/mkall_fpga_mgmt_tools.sh", "shared/bin/set_common_functions.sh", "release_version.txt", "LICENSE.txt"):
        if not (sdk / required).is_file():
            raise ValueError("SDK archive lacks required pinned build input")
    return str(sdk)


def verify_start_gate(gate, expected, identity, now):
    from datetime import datetime
    import re

    if set(gate) != set(expected) | {"authorized", "instance_id"}:
        raise ValueError("start gate schema does not match the controller permit")
    if gate.get("authorized") is not True:
        raise ValueError("work is not authorized")
    for field in ("account", "region", "run_prefix", "client_token", "source_sha256", "deadline_utc", "watchdog_execution_arn"):
        if gate.get(field) != expected[field]:
            raise ValueError("start gate mismatch: " + field)
    if (not re.fullmatch(r"i-[0-9a-f]{17}", str(identity.get("instanceId", "")))
            or identity.get("accountId") != expected["account"] or identity.get("region") != expected["region"]
            or gate.get("instance_id") != identity.get("instanceId")):
        raise ValueError("start gate does not authorize this IMDS instance identity")
    deadline = datetime.fromisoformat(expected["deadline_utc"].replace("Z", "+00:00"))
    if deadline.tzinfo is None or now >= deadline:
        raise ValueError("start gate deadline has passed or is not timezone-aware")
    return gate


def bundle_artifacts(directory, bundle_path, receipt_path):
    """Freeze scientific records into one portable bundle with an inner manifest."""
    import hashlib
    import json
    from pathlib import Path
    import tarfile

    root = Path(directory)
    records = artifact_manifest(root)
    manifest_data = (json.dumps(records, sort_keys=True, allow_nan=False) + "\n").encode()
    with (root / "artifact_manifest.json").open("xb") as stream:
        stream.write(manifest_data)
    bundle = Path(bundle_path)
    with bundle.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz", dereference=True) as archive:
        for name in sorted([*records, "artifact_manifest.json"]):
            archive.add(root / name, arcname=name, recursive=False)
    data = bundle.read_bytes()
    receipt = {"bundle_name": "results.tar.gz", "bundle_bytes": len(data),
               "bundle_sha256": hashlib.sha256(data).hexdigest(),
               "artifact_manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
               "artifact_count": len(records)}
    with Path(receipt_path).open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
    return receipt


def verify_bundle_readback(bundle_path, receipt):
    import hashlib
    from pathlib import Path

    data = Path(bundle_path).read_bytes()
    if len(data) != receipt["bundle_bytes"] or hashlib.sha256(data).hexdigest() != receipt["bundle_sha256"]:
        raise ValueError("durable bundle readback mismatch")
    return True


def verify_put_object_schema(schema):
    """Check the installed CLI model offline; never install or downgrade safety."""
    if not isinstance(schema, dict) or "IfNoneMatch" not in schema:
        raise ValueError("failed setup: AWS CLI put-object lacks --if-none-match; no overwrite fallback is allowed")
    return True


def numpy_version_for_tag(tag):
    versions = {"cp38": "1.24.4", "cp310": "2.2.6", "cp311": "2.2.6", "cp312": "2.2.6"}
    if tag not in versions:
        raise ValueError("unsupported NumPy Python tag")
    return versions[tag]


def validate_numpy_wheels(wheels):
    import re

    if not isinstance(wheels, list) or len(wheels) != 4:
        raise ValueError("exactly four pinned CPython NumPy wheels are required")
    by_tag = {}
    for wheel in wheels:
        if not isinstance(wheel, dict) or set(wheel) != {"python_tag", "filename", "sha256"}:
            raise ValueError("invalid NumPy wheel record schema")
        tag = wheel["python_tag"]
        if tag not in {"cp38", "cp310", "cp311", "cp312"} or tag in by_tag:
            raise ValueError("unsupported or duplicate NumPy Python tag")
        version = re.escape(numpy_version_for_tag(tag))
        pattern = rf"numpy-{version}-{tag}-{tag}-manylinux_2_17_x86_64\.manylinux2014_x86_64\.whl"
        if not re.fullmatch(pattern, str(wheel["filename"])) or not re.fullmatch(r"[0-9a-f]{64}", str(wheel["sha256"])):
            raise ValueError("NumPy wheel filename/version/ABI/hash is not pinned")
        by_tag[tag] = wheel
    return by_tag


def select_numpy_wheel(wheels, runtime=None):
    import platform
    import sys

    by_tag = validate_numpy_wheels(wheels)
    if runtime is None:
        runtime = {"implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag,
                   "version": tuple(sys.version_info[:2]), "platform": sys.platform, "machine": platform.machine(),
                   "abiflags": getattr(sys, "abiflags", "")}
    major, minor = runtime["version"]
    tag = f"cp{major}{minor}"
    if (runtime["implementation"] != "cpython" or runtime["cache_tag"] != f"cpython-{major}{minor}"
            or runtime.get("abiflags", "") != ""
            or runtime["platform"] != "linux" or runtime["machine"] != "x86_64" or tag not in by_tag):
        raise ValueError("unsupported NumPy runtime: require Linux x86_64 CPython 3.8, 3.10, 3.11 or 3.12 with matching ABI")
    return by_tag[tag]


def extract_numpy_runtime(archive_path, destination, expected_sha256, wheels, selected):
    """Verify every wheel, then safely unpack the selected official wheel only."""
    import hashlib
    import io
    from pathlib import Path, PurePosixPath
    import stat
    import tarfile
    import zipfile

    validate_numpy_wheels(wheels)
    if selected not in wheels:
        raise ValueError("selected NumPy wheel is not approved")
    version = numpy_version_for_tag(selected["python_tag"])
    dist_info = "numpy-" + version + ".dist-info"
    archive_path = Path(archive_path)
    if archive_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("NumPy archive exceeds bounded size")
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("NumPy archive SHA256 mismatch")
    expected = {wheel["filename"]: wheel for wheel in wheels}
    selected_bytes = None
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("NumPy archive exact wheel allowlist mismatch")
        for member in members:
            if not member.isfile() or not 0 < member.size <= 32 * 1024 * 1024:
                raise ValueError("invalid NumPy archive member type/size")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]["sha256"]:
                raise ValueError("NumPy wheel SHA256 mismatch")
            if member.name == selected["filename"]:
                selected_bytes = data
    with zipfile.ZipFile(io.BytesIO(selected_bytes)) as wheel:
        members = wheel.infolist()
        names = [member.filename.rstrip("/") for member in members]
        if (len(names) > 10000 or len(names) != len(set(names))
                or sum(member.file_size for member in members) > 256 * 1024 * 1024):
            raise ValueError("NumPy wheel duplicate members or excessive expanded size")
        for member in members:
            relative = PurePosixPath(member.filename)
            kind = stat.S_IFMT(member.external_attr >> 16)
            if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                    or relative.parts[0] not in {"numpy", "numpy.libs", dist_info}
                    or str(relative) != member.filename.rstrip("/") or "\\" in member.filename
                    or kind not in (0, stat.S_IFREG, stat.S_IFDIR) or member.flag_bits & 1
                    or member.file_size > 128 * 1024 * 1024
                    or (kind == stat.S_IFDIR and not member.is_dir())
                    or (kind == stat.S_IFREG and member.is_dir())):
                raise ValueError("NumPy wheel has unsafe path, link, encrypted or special member")
        required = {"numpy/__init__.py", dist_info + "/METADATA", dist_info + "/WHEEL"}
        if not required.issubset(names):
            raise ValueError("NumPy wheel lacks required package metadata")
        from email.parser import BytesParser
        metadata = BytesParser().parsebytes(wheel.read(dist_info + "/METADATA"))
        if metadata.get("Name") != "numpy" or metadata.get("Version") != version:
            raise ValueError("NumPy package metadata version mismatch")
        root = Path(destination)
        root.mkdir(exist_ok=False)
        for member in members:
            output = root / member.filename
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream:
                    stream.write(wheel.read(member))
                output.chmod(0o755 if member.external_attr >> 16 & 0o111 else 0o644)
    return selected


def render(config, ledger, *, today=None):
    proposed = cloud.plan(config, ledger, today=today)
    if config["steps"] != 64 or config["eval_every"] != 32 or config["eval_windows"] != 2:
        raise ValueError("production first pilot is fixed to 64 updates, eval32/windows2")
    if config["max_minutes"] < 8:
        raise ValueError("guest needs at least eight minutes including bounded upload reserve")
    cloud.matched(config, "sdk_sha256", r"[0-9a-f]{64}")
    sdk_key = cloud.matched(config, "sdk_s3_key", r"[A-Za-z0-9_./-]+")
    if sdk_key != config["run_prefix"] + "/sdk.tar":
        raise ValueError("SDK archive must use this run's sdk.tar object")
    validate_numpy_wheels(config.get("numpy_wheels"))
    cloud.matched(config, "numpy_archive_sha256", r"[0-9a-f]{64}")
    numpy_key = cloud.matched(config, "numpy_s3_key", r"[A-Za-z0-9_./-]+")
    if numpy_key != config["run_prefix"] + "/numpy-wheels.tar":
        raise ValueError("NumPy archive must use this run's numpy-wheels.tar object")
    from datetime import datetime
    deadline = datetime.fromisoformat(config["deadline_utc"].replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("deadline_utc must be timezone-aware")
    cloud.matched(config, "watchdog_execution_arn",
                  rf"arn:aws:states:{config['region']}:{config['account']}:execution:[A-Za-z0-9_-]+:[A-Za-z0-9_-]+")
    expected_gate = {key: config[key] for key in ("account", "region", "run_prefix", "source_sha256", "deadline_utc", "watchdog_execution_arn")}
    expected_gate["client_token"] = proposed["proposed_run_instances"]["ClientToken"]
    assignments = {
        "AWS_DEFAULT_REGION": config["region"], "AWS_REGION": config["region"], "AWS_PAGER": "",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null", "AWS_CONFIG_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "false", "BOTO_CONFIG": "/dev/null",
        "EXPECTED_ACCOUNT": config["account"], "AGFI": config["agfi"],
        "SDK_ROOT": config["sdk_root"], "SDK_REVISION": config["sdk_revision"],
        "S3_SDK": f"s3://{config['bucket']}/{sdk_key}", "SDK_SHA256": config["sdk_sha256"],
        "S3_NUMPY": f"s3://{config['bucket']}/{numpy_key}", "NUMPY_ARCHIVE_SHA256": config["numpy_archive_sha256"],
        "S3_START_GATE": f"s3://{config['bucket']}/{config['run_prefix']}/control/start.json",
        "S3_SOURCE": proposed["source_s3_uri"], "S3_RESULTS": proposed["results_s3_uri"],
        "ARTIFACT_BUCKET": config["bucket"], "RESULTS_PREFIX": config["run_prefix"] + "/results",
        "SOURCE_SHA256": config["source_sha256"],
        "S3_CORPUS": f"s3://{config['bucket']}/{config['corpus_key']}",
        "CORPUS_SHA256": config["corpus_sha256"],
        "MAX_MINUTES": str(config["max_minutes"]),
        "DEADLINE_UTC": config["deadline_utc"],
        "WORKLOAD_SECONDS": str(config["max_minutes"] * 60 - 300),
    }
    header = "\n".join("export " + key + "=" + shlex.quote(value) for key, value in assignments.items())
    body = r'''
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME
# Use only this instance's attached role, never credentials inherited from an
# AMI environment, root's profile files, or a container/web-identity provider.
unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
unset AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONTAINER_CREDENTIALS_FULL_URI AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN
unset AWS_EC2_METADATA_SERVICE_ENDPOINT AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE AWS_CREDENTIAL_FILE
unset AWS_CONTAINER_AUTHORIZATION_TOKEN AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE
for endpoint_override in "${!AWS_ENDPOINT_URL@}"; do unset "$endpoint_override"; done
for tool in timeout aws python3 g++ sha256sum mktemp shutdown bash; do command -v "$tool" >/dev/null; done
WORK=$(mktemp -d /tmp/tikitorch-fpga-lm.XXXXXXXX)
export WORK
mkdir "$WORK/artifacts"
exec > "$WORK/artifacts/guest.log" 2>&1
HEARTBEAT_PID=
RESULT=FAILED
finish(){
  rc=$?
  trap - EXIT TERM INT HUP
  set +e
  if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null; wait "$HEARTBEAT_PID" 2>/dev/null; fi
  [ "$rc" -eq 0 ] && [ "$RESULT" = SUCCESS ] || RESULT=FAILED
  # Preserve a bounded immutable log snapshot even when setup failed before a
  # scientific manifest existed. Upload/cleanup messages remain in guest.log.
  timeout --kill-after=5 5 tail -c 4194304 "$WORK/artifacts/guest.log" > "$WORK/artifacts/terminal_log.txt"
  # No success marker until uploaded scientific files have been downloaded and
  # re-hashed. Failed uploads never keep this paid instance alive indefinitely.
  timeout --kill-after=5 30 python3 - "$WORK/artifacts" "$WORK/results.tar.gz" "$WORK/receipt.json" <<'PY_ARTIFACTS'
import sys
ARTIFACT_MANIFEST_FUNCTION
BUNDLE_ARTIFACTS_FUNCTION
bundle_artifacts(*sys.argv[1:])
PY_ARTIFACTS
  manifest_rc=$?
  if [ "$manifest_rc" -eq 0 ]; then
    timeout --kill-after=5 55 aws s3api put-object --bucket "$ARTIFACT_BUCKET" \
        --key "$RESULTS_PREFIX/results.tar.gz" --body "$WORK/results.tar.gz" --if-none-match '*'
    upload_rc=$?
    timeout --kill-after=5 55 aws s3 cp "$S3_RESULTS/results.tar.gz" "$WORK/results.readback.tar.gz" --only-show-errors
    download_rc=$?
    timeout --kill-after=5 10 python3 - "$WORK/receipt.json" "$WORK/results.readback.tar.gz" <<'PY_RECEIPT'
import json, pathlib, sys
VERIFY_BUNDLE_FUNCTION
verify_bundle_readback(sys.argv[2], json.loads(pathlib.Path(sys.argv[1]).read_text()))
PY_RECEIPT
    verify_rc=$?
  else
    upload_rc=1; download_rc=1; verify_rc=1
  fi
  if [ "$upload_rc" -ne 0 ] || [ "$download_rc" -ne 0 ] || [ "$verify_rc" -ne 0 ]; then
    RESULT=FAILED; [ "$rc" -ne 0 ] || rc=74
  fi
  timeout --kill-after=5 10 python3 - "$WORK/completion.json" "$RESULT" "$rc" "$verify_rc" <<'PY_TERMINAL'
import json, os, pathlib, sys
work = pathlib.Path(os.environ["WORK"])
identity_path = work / "artifacts/instance_identity.json"
identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
receipt_path = work / "receipt.json"
receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
with pathlib.Path(sys.argv[1]).open("x") as stream:
    json.dump({"schema": "tikitorch.fpga_lm_guest_completion.v1", **EXPECTED_GATE,
               "instance_id": identity.get("instanceId"), **receipt,
               "status":sys.argv[2], "exit_code":int(sys.argv[3]),
               "durable_artifacts_verified":int(sys.argv[4]) == 0,
               "scope":"paired correctness only; no quality, speed or DSP claim"}, stream, sort_keys=True)
PY_TERMINAL
  timeout --kill-after=5 15 aws s3api put-object --bucket "$ARTIFACT_BUCKET" \
      --key "$RESULTS_PREFIX/completion.json" --body "$WORK/completion.json" --if-none-match '*'
  terminal_rc=$?
  # The diagnostic log is deliberately mutable, not a scientific success
  # marker. Preserve failed setup details even on a CLI too old for safe PUT.
  timeout --kill-after=5 10 aws s3 cp "$WORK/artifacts/guest.log" "$S3_RESULTS/guest.log" --only-show-errors
  timeout --kill-after=5 10 shutdown -h now
  [ "$terminal_rc" -eq 0 ] || exit 74
  exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM INT HUP
# Must be armed before any SDK build, download, or FPGA work; an independently
# supervised EC2 termination deadline is also mandatory in the host controller.
timeout --kill-after=5 10 shutdown -h +"$MAX_MINUTES" "TikiTorch FPGA LM deadline"
WORKLOAD_SECONDS=$(timeout --kill-after=5 10 python3 - <<'PY_DEADLINE'
import os
from datetime import datetime, timezone
deadline = datetime.fromisoformat(os.environ["DEADLINE_UTC"].replace("Z", "+00:00"))
seconds = min(int(os.environ["WORKLOAD_SECONDS"]), int((deadline - datetime.now(timezone.utc)).total_seconds()) - 300)
if seconds <= 0:
    raise SystemExit("insufficient time before supervised deadline to reserve durable artifact upload")
print(seconds)
PY_DEADLINE
)
( while sleep 30; do timeout --kill-after=5 10 aws s3 cp "$WORK/artifacts/guest.log" "$S3_RESULTS/guest.log" --only-show-errors || true; done ) &
HEARTBEAT_PID=$!
cat > "$WORK/workload.sh" <<'WORKLOAD'
#!/bin/bash
set -euo pipefail
run_for(){ local seconds="$1"; shift; timeout --signal=TERM --kill-after=10 "$seconds" "$@"; }
# The permit is created only after the host controller reconciles this exact
# instance against the pre-armed watchdog. IMDSv2 is read without any proxy.
run_for 315 python3 - <<'PY_START_GATE'
import json, os, pathlib, subprocess, time, urllib.request
from datetime import datetime, timezone
VERIFY_START_GATE_FUNCTION
expected = EXPECTED_GATE
work = pathlib.Path(os.environ["WORK"])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
request = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                 headers={"X-aws-ec2-metadata-token-ttl-seconds": "600"})
with opener.open(request, timeout=5) as response:
    token = response.read().decode()
request = urllib.request.Request("http://169.254.169.254/latest/dynamic/instance-identity/document",
                                 headers={"X-aws-ec2-metadata-token": token})
with opener.open(request, timeout=5) as response:
    identity = json.load(response)
del token
with (work / "artifacts/instance_identity.json").open("x") as stream:
    json.dump(identity, stream, sort_keys=True)
end = time.monotonic() + 300
deadline = datetime.fromisoformat(expected["deadline_utc"].replace("Z", "+00:00"))
while time.monotonic() < end and datetime.now(timezone.utc) < deadline:
    try:
        result = subprocess.run(["aws", "s3", "cp", os.environ["S3_START_GATE"], "-", "--only-show-errors"],
                                capture_output=True, timeout=5, check=False)
    except subprocess.TimeoutExpired:
        result = None
    if result is not None and result.returncode == 0:
        gate = verify_start_gate(json.loads(result.stdout), expected, identity, datetime.now(timezone.utc))
        with (work / "artifacts/start_gate.json").open("x") as stream:
            json.dump(gate, stream, sort_keys=True)
        break
    time.sleep(2)
else:
    raise SystemExit("no current controller start authorization within bounded wait")
PY_START_GATE
test "$(run_for 20 aws sts get-caller-identity --query Account --output text)" = "$EXPECTED_ACCOUNT"
# This skeleton command is local model introspection, not an S3 request. Detect
# unsupported conditional writes before source/SDK downloads or native builds.
run_for 15 aws --version > "$WORK/artifacts/aws_cli.txt"
run_for 15 env AWS_EC2_METADATA_DISABLED=true aws s3api put-object --generate-cli-skeleton input \
    > "$WORK/artifacts/aws_cli_put_object_schema.json"
python3 - "$WORK/artifacts/aws_cli_put_object_schema.json" <<'PY_CLI_CAPABILITY'
import json, pathlib, sys
VERIFY_PUT_OBJECT_FUNCTION
verify_put_object_schema(json.loads(pathlib.Path(sys.argv[1]).read_text()))
PY_CLI_CAPABILITY
# Select the exact ABI before downloading or compiling anything. The runtime
# bundle contains approved official wheels; no pip, package manager or fallback.
run_for 15 python3 - <<'PY_NUMPY_SELECT'
import json, os, pathlib, platform, sys
NUMPY_VERSION_FUNCTION
VALIDATE_NUMPY_FUNCTION
SELECT_NUMPY_FUNCTION
facts = {"implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag,
         "python_version": sys.version, "version": list(sys.version_info[:3]),
         "platform": sys.platform, "machine": platform.machine(), "abiflags": getattr(sys, "abiflags", ""),
         "executable": sys.executable}
os_release = pathlib.Path("/etc/os-release")
facts["os_release"] = os_release.read_text() if os_release.is_file() else None
with (pathlib.Path(os.environ["WORK"]) / "artifacts/preflight_runtime.json").open("x") as stream:
    json.dump(facts, stream, sort_keys=True, allow_nan=False)
selected = select_numpy_wheel(NUMPY_WHEELS)
record = {"version": numpy_version_for_tag(selected["python_tag"]), "python_tag": selected["python_tag"],
          "cache_tag": sys.implementation.cache_tag, "python_version": sys.version,
          "platform": sys.platform, "machine": platform.machine(),
          "archive_sha256": os.environ["NUMPY_ARCHIVE_SHA256"], "wheel": selected}
with (pathlib.Path(os.environ["WORK"]) / "numpy_selection.json").open("x") as stream:
    json.dump(record, stream, sort_keys=True, allow_nan=False)
PY_NUMPY_SELECT
run_for 60 aws s3 cp "$S3_NUMPY" "$WORK/numpy-wheels.tar" --only-show-errors
run_for 60 python3 - <<'PY_NUMPY_EXTRACT'
import json, os, pathlib
NUMPY_VERSION_FUNCTION
VALIDATE_NUMPY_FUNCTION
EXTRACT_NUMPY_FUNCTION
work = pathlib.Path(os.environ["WORK"])
selection = json.loads((work / "numpy_selection.json").read_text())
extract_numpy_runtime(work / "numpy-wheels.tar", work / "runtime", os.environ["NUMPY_ARCHIVE_SHA256"],
                      NUMPY_WHEELS, selection["wheel"])
PY_NUMPY_EXTRACT
export PYTHONPATH="$WORK/runtime" PYTHONNOUSERSITE=1
run_for 30 python3 - <<'PY_NUMPY_VERIFY'
import json, os, pathlib
import numpy
work = pathlib.Path(os.environ["WORK"])
record = json.loads((work / "numpy_selection.json").read_text())
if numpy.__version__ != record["version"] or (work / "runtime").resolve() not in pathlib.Path(numpy.__file__).resolve().parents:
    raise SystemExit("NumPy import did not use the exact verified private runtime")
if int(numpy.arange(4, dtype=numpy.int32).sum()) != 6:
    raise SystemExit("NumPy native runtime smoke check failed")
record["import_verified"] = True
with (work / "artifacts/numpy_runtime.json").open("x") as stream:
    json.dump(record, stream, sort_keys=True, allow_nan=False)
PY_NUMPY_VERIFY
run_for 60 aws s3 cp "$S3_SDK" "$WORK/sdk.tar" --only-show-errors
python3 - "$WORK/sdk.tar" "$WORK/sdk_source" "$SDK_SHA256" <<'PY_SDK_EXTRACT'
import sys
EXTRACT_SDK_FUNCTION
extract_sdk_archive(*sys.argv[1:])
PY_SDK_EXTRACT
SDK_ROOT="$WORK/sdk_source/aws-fpga"
export SDK_ROOT
export AWS_FPGA_REPO_DIR="$SDK_ROOT" SDK_DIR="$SDK_ROOT/sdk"
# Rebuild only the pinned SDK userspace tools/libraries; no global installer,
# repository fetch, package manager, or fallback to a stale system binary.
( cd "$SDK_ROOT/sdk/userspace"; run_for 180 bash ./mkall_fpga_mgmt_tools.sh 0 64 )
SDK_LIB="$SDK_ROOT/sdk/userspace/lib"
FPGA_CMD="$SDK_ROOT/sdk/userspace/fpga_mgmt_tools/src/static-fpga-local-cmd"
test -x "$FPGA_CMD"
test -s "$SDK_LIB/libfpga_mgmt.a"
run_for 15 python3 - <<'PY_ENVIRONMENT'
import json, os, pathlib, platform, sys
import numpy
runtime = json.loads((pathlib.Path(os.environ["WORK"]) / "artifacts/numpy_runtime.json").read_text())
record = {"sdk_revision": os.environ["SDK_REVISION"], "sdk_sha256": os.environ["SDK_SHA256"],
          "source_sha256": os.environ["SOURCE_SHA256"], "corpus_sha256": os.environ["CORPUS_SHA256"],
          "python": sys.version, "numpy": numpy.__version__, "platform": platform.platform(),
          "numpy_archive_sha256": runtime["archive_sha256"], "numpy_wheel": runtime["wheel"],
          "numpy_version_expected": runtime["version"], "python_tag": runtime["python_tag"],
          "scope": "paired correctness only; no quality, speed or DSP claim"}
with (pathlib.Path(os.environ["WORK"]) / "artifacts/environment.json").open("x") as stream:
    json.dump(record, stream, sort_keys=True, allow_nan=False)
PY_ENVIRONMENT
g++ --version > "$WORK/artifacts/compiler.txt"
sha256sum "$FPGA_CMD" "$SDK_LIB/libfpga_mgmt.a" > "$WORK/artifacts/sdk_binaries.sha256"
run_for 60 aws s3 cp "$S3_SOURCE" "$WORK/source.tar.gz" --only-show-errors
python3 - "$WORK/source.tar.gz" "$WORK/source" "$SOURCE_SHA256" <<'PY_EXTRACT'
import sys
EXTRACT_FUNCTION
extract_verified_archive(sys.argv[1], sys.argv[2], EXPECTED_MEMBERS, sys.argv[3])
PY_EXTRACT
run_for 60 aws s3 cp "$S3_CORPUS" "$WORK/corpus.txt" --only-show-errors
printf '%s  %s\n' "$CORPUS_SHA256" "$WORK/corpus.txt" | sha256sum -c -
cd "$WORK/source"
# Python 3.8 preserves a relative __file__ for relative script invocation.
# The frozen runner's source-identity contract therefore needs an absolute
# script path; keep cwd here for the native compiler's relative source paths.
run_for 150 python3 "$WORK/source/scripts/fpga_lm_pretrain.py" --build-csim "$WORK/native_csim" --width 512
cp "$WORK/native_csim/build.json" "$WORK/artifacts/csim_build.json"
COMMON=(-O2 -std=c++17 -fPIC -shared -DML_D=512 -DTT_RESIDENT_W -DTT_ONCHIP_GRAD -DTT_URAM_W
        -Itikitorch/hardware/hls/forge_kernels)
# Both translation units are C++: use the real standard <cstdint>. The archived
# C-only shim is not needed by SDK headers, which include <stdint.h> directly.
run_for 120 g++ "${COMMON[@]}" -DTT_FPGA_PCI -I"$SDK_ROOT/sdk/userspace/include" \
    tikitorch/fpga/bridge.cpp scripts/fpga_lm_layout.cpp \
    "$SDK_LIB/libfpga_mgmt.a" -lrt -lpthread -o "$WORK/libpci.so"
sha256sum "$WORK/native_csim/libmorphlm.so" "$WORK/libpci.so" > "$WORK/artifacts/native_libraries.sha256"
python3 - "$WORK/libpci.so" <<'PY_CAPABILITY'
import ctypes, sys
library = ctypes.CDLL(sys.argv[1])
library.tt_fpga_last_error.argtypes = [ctypes.c_void_p]
library.tt_fpga_last_error.restype = ctypes.c_int
library.tt_morph_lm_backend.restype = ctypes.c_int
if library.tt_morph_lm_backend() != 2:
    raise SystemExit("checked PCI backend capability missing")
PY_CAPABILITY
# Identical data path, optimizer and evaluation schedule on CPU reference and
# silicon. Ordinary training defaults do not control these explicit recipes.
RECIPE=(--width 512 --corpus "$WORK/corpus.txt" --steps 64 --eval-every 32 --eval-windows 2
        --lr 1 --threshold 2048 --seed 0)
run_for 180 python3 "$WORK/source/scripts/fpga_lm_pretrain.py" --backend csim \
    --library "$WORK/native_csim/libmorphlm.so" --output "$WORK/artifacts/csim" "${RECIPE[@]}"
run_for 120 "$FPGA_CMD" LoadFpgaImage -S 0 -I "$AGFI"
run_for 30 "$FPGA_CMD" DescribeFpgaImage -S 0 > "$WORK/artifacts/image_before.txt"
python3 - "$WORK/artifacts/image_before.txt" "$AGFI" <<'PY_IMAGE'
import json, pathlib, sys
VERIFY_IMAGE_FUNCTION
path = pathlib.Path(sys.argv[1])
record = verify_image_text(path.read_text(), sys.argv[2])
with path.with_suffix(".json").open("x") as stream:
    json.dump(record, stream, sort_keys=True)
PY_IMAGE
# Native busy-poll loop is finite; its enclosing process has a wall timeout too.
export ML_D=512 ML_BATCH=1 TT_CLKGEN_BRINGUP=1 TT_KICK_SPIN=20000000
run_for 300 python3 "$WORK/source/scripts/fpga_lm_pretrain.py" --backend pci --image-id "$AGFI" \
    --library "$WORK/libpci.so" --output "$WORK/artifacts/pci" "${RECIPE[@]}"
run_for 30 "$FPGA_CMD" DescribeFpgaImage -S 0 > "$WORK/artifacts/image_after.txt"
python3 - "$WORK/artifacts/image_after.txt" "$AGFI" <<'PY_IMAGE_AFTER'
import json, pathlib, sys
VERIFY_IMAGE_FUNCTION
path = pathlib.Path(sys.argv[1])
record = verify_image_text(path.read_text(), sys.argv[2])
with path.with_suffix(".json").open("x") as stream:
    json.dump(record, stream, sort_keys=True)
PY_IMAGE_AFTER
run_for 60 python3 "$WORK/source/scripts/fpga_lm_pretrain.py" --compare-runs \
    "$WORK/artifacts/csim" "$WORK/artifacts/pci" > "$WORK/artifacts/paired_comparison.json"
python3 - "$WORK/artifacts/paired_comparison.json" <<'PY_PASSED'
import json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
if record.get("complete") is not True or record.get("all_checkpoint_payloads_equal") is not True:
    raise SystemExit("paired checkpoint equality gate did not pass")
PY_PASSED
WORKLOAD
timeout --signal=TERM --kill-after=10 "$WORKLOAD_SECONDS" bash "$WORK/workload.sh"
RESULT=SUCCESS
'''
    body = body.replace("ARTIFACT_MANIFEST_FUNCTION", inspect.getsource(artifact_manifest))
    body = body.replace("BUNDLE_ARTIFACTS_FUNCTION", inspect.getsource(bundle_artifacts))
    body = body.replace("VERIFY_BUNDLE_FUNCTION", inspect.getsource(verify_bundle_readback))
    body = body.replace("VERIFY_PUT_OBJECT_FUNCTION", inspect.getsource(verify_put_object_schema))
    body = body.replace("VALIDATE_NUMPY_FUNCTION", inspect.getsource(validate_numpy_wheels))
    body = body.replace("NUMPY_VERSION_FUNCTION", inspect.getsource(numpy_version_for_tag))
    body = body.replace("SELECT_NUMPY_FUNCTION", inspect.getsource(select_numpy_wheel))
    body = body.replace("EXTRACT_NUMPY_FUNCTION", inspect.getsource(extract_numpy_runtime))
    body = body.replace("NUMPY_WHEELS", repr(config["numpy_wheels"]))
    body = body.replace("VERIFY_START_GATE_FUNCTION", inspect.getsource(verify_start_gate))
    body = body.replace("EXTRACT_SDK_FUNCTION", inspect.getsource(extract_sdk_archive))
    body = body.replace("EXTRACT_FUNCTION", inspect.getsource(extract_verified_archive))
    body = body.replace("VERIFY_IMAGE_FUNCTION", inspect.getsource(verify_image_text))
    body = body.replace("EXPECTED_MEMBERS", repr(list(cloud.SOURCE_FILES)))
    body = body.replace("EXPECTED_GATE", repr(expected_gate))
    return "#!/bin/bash\n# No launch API: external ownership/budget/termination supervision is required.\n" + header + body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    print(render(json.loads(args.config.read_text()), json.loads(args.ledger.read_text())), end="")


if __name__ == "__main__":
    main()
