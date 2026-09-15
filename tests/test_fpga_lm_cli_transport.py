"""Actual AWS CLI serialization regression; never allows a network request.

Source-installed CLI v2 distributions expose their Python interpreter through
the aws executable's shebang. Bundled executables without one are skipped.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest


def installed_cli_python():
    executable = shutil.which("aws")
    if executable is None:
        pytest.skip("AWS CLI is not installed")
    with Path(executable).resolve().open("rb") as stream:
        shebang = stream.readline(4096)
    if not shebang.startswith(b"#!/"):
        pytest.skip("installed AWS CLI does not expose a source Python interpreter")
    try:
        interpreter = shebang[2:].decode("utf-8").strip()
    except UnicodeDecodeError:
        pytest.skip("AWS CLI interpreter shebang is not UTF-8")
    if "python" not in Path(interpreter).name or not Path(interpreter).is_file():
        pytest.skip("AWS CLI interpreter cannot be located directly")
    return interpreter


COMMON_PROBE = r'''
import base64, gzip, io, json, os, socket, sys, tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# Block networking before importing any CLI code. Audit hooks also catch paths
# that bypass patched Python socket methods; no endpoint is contacted.
def deny_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        raise AssertionError("offline CLI serialization test forbids networking")
sys.addaudithook(deny_network)
def forbidden_network(*args, **kwargs):
    raise AssertionError("offline CLI serialization test forbids networking")
socket.socket.connect = forbidden_network
socket.socket.connect_ex = forbidden_network
socket.create_connection = forbidden_network
socket.getaddrinfo = forbidden_network

try:
    import awscli
    from awscli.clidriver import create_clidriver
    from botocore.awsrequest import AWSResponse
except ImportError as exc:
    print(json.dumps({"skip": "installed CLI Python modules unavailable: " + str(exc)}))
    raise SystemExit(77)
if not awscli.__version__.startswith("2."):
    print(json.dumps({"skip": "serialization regression targets AWS CLI v2"}))
    raise SystemExit(77)
'''

PROBE = COMMON_PROBE + r'''
script = b"#!/bin/bash\necho offline-encoding-regression\n"
compressed = gzip.compress(script, mtime=0)
encoded = base64.b64encode(compressed).decode()

def wire_value(parameters, extra_arguments):
    driver = create_clidriver()
    driver.session.set_credentials("offline-key", "offline-secret")
    model = driver.session.get_service_model("ec2").operation_model("RunInstances")
    assert model.input_shape.members["UserData"].type_name == "string"
    captured = []
    def capture(params, **kwargs):
        captured.append(params["body"]["UserData"])
        return AWSResponse("https://offline.invalid", 200, {}, None), {"Instances": []}
    driver.session.register("before-call.ec2.RunInstances", capture)
    with redirect_stdout(io.StringIO()):
        rc = driver.main(["ec2", "run-instances", "--region", "us-east-1",
                          "--cli-input-json", json.dumps(parameters), *extra_arguments])
    assert rc == 0 and len(captured) == 1, (rc, len(captured))
    return captured[0]

parameters = {"ImageId": "ami-0123456789abcdef0", "MinCount": 1, "MaxCount": 1}
wrong = wire_value({**parameters, "UserData": encoded}, [])
assert wrong == base64.b64encode(encoded.encode()).decode()
assert not base64.b64decode(wrong).startswith(b"\x1f\x8b")
with tempfile.TemporaryDirectory(prefix="tiki-cli-userdata-", dir=sys.argv[1]) as directory:
    payload = Path(directory) / "userdata.gz"
    with payload.open("xb") as stream:
        stream.write(compressed)
    correct = wire_value(parameters, ["--user-data", "fileb://" + str(payload)])
assert correct == encoded
assert gzip.decompress(base64.b64decode(correct)) == script
print(json.dumps({"cli_version": awscli.__version__, "network_forbidden": True,
                  "requests_intercepted_before_http": 2,
                  "json_base64_doubleencodes": True,
                  "fileb_gzip_restores_exact_script": True}, sort_keys=True))
'''


GET_OBJECT_PROBE = COMMON_PROBE + r'''
parameters = {"Bucket": "tiki-offline-probe", "Key": "run/control/start.json",
              "ExpectedBucketOwner": "123456789012"}
expected = b'{"offline_permit_only":true}'
with tempfile.TemporaryDirectory(prefix="tiki-cli-getobject-", dir=sys.argv[1]) as directory:
    output = Path(directory) / "object"
    for mode in ("json", "explicit"):
        driver = create_clidriver()
        driver.session.set_credentials("offline-key", "offline-secret")
        captured = []
        def capture(params, **kwargs):
            captured.append(params)
            return AWSResponse(params["url"], 200, {}, None), {"Body": io.BytesIO(expected)}
        driver.session.register("before-call.s3.GetObject", capture)
        args = ["--region", "us-east-1", "--no-cli-pager", "s3api", "get-object"]
        if mode == "json":
            args += ["--cli-input-json", json.dumps(parameters)]
        else:
            args += ["--bucket", parameters["Bucket"], "--key", parameters["Key"],
                     "--expected-bucket-owner", parameters["ExpectedBucketOwner"]]
        args += [str(output)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = driver.main(args)
        if mode == "json":
            assert rc == 252 and not captured, (rc, stderr.getvalue())
            assert "--bucket, --key" in stderr.getvalue()
            assert not output.exists()
        else:
            assert rc == 0 and len(captured) == 1, (rc, stderr.getvalue())
            assert captured[0]["headers"]["x-amz-expected-bucket-owner"] == parameters["ExpectedBucketOwner"]
            assert captured[0]["url"].split("?")[0].endswith("/" + parameters["Key"])
            assert output.read_bytes() == expected
print(json.dumps({"cli_version": awscli.__version__, "network_forbidden": True,
                  "requests_intercepted_before_http": 1,
                  "json_required_flags_rejected_before_dispatch": True,
                  "explicit_flags_restore_exact_object": True}, sort_keys=True))
'''


def run_probe(probe, tmp_path):
    interpreter = installed_cli_python()
    # Do not inherit profiles, credentials, endpoint overrides or import hooks.
    environment = {
        "PATH": os.defpath, "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_CONFIG_FILE": os.devnull, "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "BOTO_CONFIG": os.devnull, "AWS_PAGER": "", "AWS_CLI_AUTO_PROMPT": "off",
        "AWS_DEFAULT_REGION": "us-east-1", "AWS_REGION": "us-east-1",
    }
    result = subprocess.run([interpreter, "-I", "-c", probe, str(tmp_path)],
                            env=environment, capture_output=True, text=True,
                            timeout=20, check=False)
    if result.returncode == 77:
        pytest.skip(json.loads(result.stdout)["skip"])
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_installed_cli_runinstances_json_and_fileb_serialization(tmp_path):
    record = run_probe(PROBE, tmp_path)
    assert record["network_forbidden"] is True
    assert record["requests_intercepted_before_http"] == 2
    assert record["json_base64_doubleencodes"] is True
    assert record["fileb_gzip_restores_exact_script"] is True


def test_installed_cli_get_object_requires_explicit_flags_before_streaming(tmp_path):
    record = run_probe(GET_OBJECT_PROBE, tmp_path)
    assert record["network_forbidden"] is True
    assert record["requests_intercepted_before_http"] == 1
    assert record["json_required_flags_rejected_before_dispatch"] is True
    assert record["explicit_flags_restore_exact_object"] is True
