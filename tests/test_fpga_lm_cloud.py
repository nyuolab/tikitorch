"""AWS CLI adapter contracts exercised exclusively through a fake subprocess."""

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from scripts import fpga_lm_cloud as A
from scripts import fpga_lm_control as C


@pytest.fixture
def config():
    value = A.preparation.example()["config"]
    value.update(profile="offline", account="123456789012", ami="ami-0123456789abcdef0",
                 ami_owner="210987654321", agfi="agfi-0123456789abcdef0", bucket="tiki-private-test",
                 subnet="subnet-0123456789abcdef0", security_group="sg-0123456789abcdef0",
                 instance_profile="arn:aws:iam::123456789012:instance-profile/tiki-test",
                 sdk_revision="a" * 40, sdk_sha256="b" * 64, source_sha256="c" * 64,
                 corpus_sha256="d" * 64, price_checked_on=datetime.now(timezone.utc).date().isoformat(),
                 guest_user_data="#!/bin/bash\n# Offline test only.\n" + "# compressed fixture\n" * 2000,
                 max_minutes=30, storage_ip_artifact_allowance_usd="5")
    value["source_file_hashes"] = {name: "e" * 64 for name in A.preparation.SOURCE_FILES}
    token = hashlib.sha256(f"{value['account']}:{value['region']}:{value['run_prefix']}".encode()).hexdigest()
    value["watchdog_execution_arn"] = f"arn:aws:states:{value['region']}:{value['account']}:execution:tt-fpga-lm-wd-{token[:32]}:run-{token[:32]}"
    value["deadline_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=29)).isoformat()
    return value


class Clock:
    def __init__(self):
        self.elapsed = 0
        self.origin = datetime.now(timezone.utc)

    def time(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds

    def now(self):
        return self.origin + timedelta(seconds=self.elapsed)


class CLI:
    def __init__(self, config):
        self.config = config
        self.calls, self.role, self.policy, self.machine = [], None, None, None
        self.execution, self.instances, self.objects = None, [], {}
        self.failures = {}
        self.invalid_definition = False
        self.bad_history = False
        self.broad_policy = False
        self.user_data_payloads = []
        self.discovery_pages = None
        self.discovery_delay = None

    def error(self, command, code):
        return subprocess.CompletedProcess(command, 255, "", f"An error occurred ({code}): PRIVATE_PROVIDER_DETAIL")

    def instance(self, token):
        return {"InstanceId": "i-0123456789abcdef0", "InstanceType": "f2.6xlarge", "ClientToken": token,
                "Placement": {"AvailabilityZone": self.config["region"] + "a"}, "State": {"Name": "running"},
                "Tags": [{"Key": "Project", "Value": A.preparation.PROJECT_TAG},
                         {"Key": "Run", "Value": self.config["run_prefix"].split("/")[-1]},
                         {"Key": "OwnerAccount", "Value": self.config["account"]}]}

    def __call__(self, command, **kwargs):
        assert command[0] == "aws"
        assert command[command.index("--profile") + 1] == self.config["profile"]
        assert command[command.index("--region") + 1] == self.config["region"]
        assert 0 < kwargs["timeout"] <= 40
        if "get-object" in command:
            assert "--cli-input-json" not in command
            index = command.index("get-object")
            service, operation = command[index - 1:index + 1]
            payload = {key: command[command.index(flag) + 1] for key, flag in (
                ("Bucket", "--bucket"), ("Key", "--key"), ("ExpectedBucketOwner", "--expected-bucket-owner"))}
            assert len(command[index + 1:]) == 7
            extra = [command[-1]]
        else:
            index = command.index("--cli-input-json")
            service, operation = command[index - 2:index]
            payload = json.loads(command[index + 1])
            extra = command[index + 2:]
        self.calls.append((service, operation, payload, extra))
        key = (service, operation)
        if key in self.failures:
            return self.error(command, self.failures[key])
        if key == ("sts", "get-caller-identity"):
            result = {"Account": self.config["account"]}
        elif key == ("iam", "get-role"):
            if self.role is None:
                return self.error(command, "NoSuchEntity")
            result = {"Role": self.role}
        elif key == ("iam", "create-role"):
            assert self.role is None
            self.role = {"Arn": f"arn:aws:iam::{self.config['account']}:role/{payload['RoleName']}",
                         "AssumeRolePolicyDocument": json.loads(payload["AssumeRolePolicyDocument"]), "Tags": payload["Tags"]}
            result = {"Role": self.role}
        elif key == ("iam", "put-role-policy"):
            self.policy = json.loads(payload["PolicyDocument"]); result = {}
        elif key == ("iam", "get-role-policy"):
            result = {"PolicyDocument": self.policy}
        elif key == ("iam", "list-role-policies"):
            result = {"PolicyNames": ["terminate-owned-pilot"]}
        elif key == ("iam", "list-attached-role-policies"):
            result = {"AttachedPolicies": [{"PolicyName": "Admin"}] if self.broad_policy else []}
        elif key == ("stepfunctions", "validate-state-machine-definition"):
            result = {"result": "FAIL" if self.invalid_definition else "OK", "truncated": False}
        elif key == ("stepfunctions", "describe-state-machine"):
            if self.machine is None:
                return self.error(command, "StateMachineDoesNotExist")
            result = self.machine
        elif key == ("stepfunctions", "create-state-machine"):
            assert self.machine is None
            self.machine = {"stateMachineArn": f"arn:aws:states:{self.config['region']}:{self.config['account']}:stateMachine:{payload['name']}",
                            "type": payload["type"], "roleArn": payload["roleArn"], "definition": payload["definition"]}
            result = {"stateMachineArn": self.machine["stateMachineArn"]}
        elif key == ("stepfunctions", "start-execution"):
            self.execution = {"executionArn": self.config["watchdog_execution_arn"], "status": "RUNNING",
                              "stateMachineArn": payload["stateMachineArn"], "input": payload["input"]}
            result = {"executionArn": self.execution["executionArn"]}
        elif key == ("stepfunctions", "describe-execution"):
            result = self.execution
        elif key == ("stepfunctions", "get-execution-history"):
            result = {"events": [
                {"id": 2, "type": "TaskStateEntered", "stateEnteredEventDetails": {"name": "StartupHealthCheck"}},
                {"id": 3, "type": "TaskSucceeded"},
                {"id": 4, "type": "TaskStateExited", "stateExitedEventDetails": {"name": "StartupHealthCheck"}},
                {"id": 5, "type": "WaitStateEntered", "stateEnteredEventDetails": {"name": "WaitUntilDeadline"}},
            ] if not self.bad_history else []}
        elif key == ("ec2", "describe-instances"):
            if "Filters" in payload:
                assert extra == ["--no-paginate"] and payload["MaxResults"] == 1000
                if self.discovery_delay is not None:
                    self.discovery_delay()
            if "Filters" in payload and self.discovery_pages is not None:
                result = self.discovery_pages.pop(0)
            else:
                result = {"Reservations": [{"OwnerId": self.config["account"], "Instances": self.instances}]} if self.instances else {"Reservations": []}
        elif key == ("ec2", "run-instances"):
            assert "UserData" not in payload
            assert len(extra) == 2 and extra[0] == "--user-data"
            assert extra[1].startswith("fileb://")
            path = Path(extra[1][len("fileb://"):])
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_mode & 0o777 == 0o600
            self.user_data_payloads.append(path.read_bytes())
            self.instances = [self.instance(payload["ClientToken"])]
            result = {"Instances": self.instances}
        elif key == ("ec2", "terminate-instances"):
            assert payload["InstanceIds"] == [self.instances[0]["InstanceId"]]
            self.instances[0]["State"]["Name"] = "terminated"; result = {}
        elif key == ("s3api", "head-object"):
            if payload["Key"] not in self.objects:
                return self.error(command, "404")
            result = {"ContentLength": len(self.objects[payload["Key"]])}
        elif key == ("s3api", "get-object"):
            Path(extra[-1]).write_bytes(self.objects[payload["Key"]]); result = {}
        elif key == ("s3api", "put-object"):
            assert payload["IfNoneMatch"] == "*"
            assert payload["Key"] not in self.objects
            self.objects[payload["Key"]] = Path(extra[extra.index("--body") + 1]).read_bytes(); result = {}
        else:
            raise AssertionError(f"unapproved fake CLI operation {key}")
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")


@pytest.fixture
def setup(config, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    cli, clock = CLI(config), Clock()
    adapter = A.Adapter(config, state, tmp_path / "retrieved", runner=cli, monotonic=clock.time, sleep=clock.sleep, now=clock.now)
    payload = {key: config[key] for key in ("account", "region", "run_prefix", "deadline_utc")}
    payload.update(client_token=adapter.token, started_at_utc=datetime.now(timezone.utc).isoformat(),
                   execution_arn=config["watchdog_execution_arn"])
    return adapter, cli, clock, payload


def test_watchdog_created_only_after_server_validation_and_independent_readback(setup):
    adapter, cli, _, payload = setup
    receipt = adapter("arm_watchdog", payload)
    assert receipt["verified"] and receipt["state"] == "WaitUntilDeadline"
    operations = [operation for _, operation, _, _ in cli.calls]
    assert operations.index("validate-state-machine-definition") < operations.index("create-role")
    assert operations.index("get-role-policy") < operations.index("start-execution")
    assert operations.index("create-state-machine") < operations.index("start-execution")
    assert operations[-1] == "get-execution-history"
    events = (adapter.state / "adapter-events.jsonl").read_text()
    assert "PRIVATE_PROVIDER_DETAIL" not in events and "guest_user_data" not in events
    assert (adapter.state / f"{adapter.token}-watchdog-artifacts.json").exists()
    assert (adapter.state / f"{adapter.token}-watchdog-armed.json").exists()


def test_distinct_runs_share_ledger_directory_without_overwriting_receipts(setup):
    first, cli, _, _ = setup
    legacy = first.state / "watchdog-artifacts.json"
    legacy.write_bytes(b"preserve-original-pilot-history")
    later_config = {**first.config, "run_prefix": first.config["run_prefix"] + "-retry"}
    second = A.Adapter(later_config, first.state, runner=cli)
    names = ("watchdog-artifacts.json", "watchdog-armed.json", "retrieval-verified.json")
    for adapter in (first, second):
        for name in names:
            adapter._record_new(name, {"run_prefix": adapter.config["run_prefix"]})
        adapter._event(operation="offline-regression", outcome="success")
    assert first.token != second.token
    for adapter in (first, second):
        for name in names:
            assert json.loads((adapter.state / f"{adapter.token}-{name}").read_bytes()) == {
                "run_prefix": adapter.config["run_prefix"]}
        with pytest.raises(FileExistsError):
            adapter._record_new(names[0], {"should_not_overwrite": True})
    assert legacy.read_bytes() == b"preserve-original-pilot-history"
    events = [json.loads(line) for line in (first.state / "adapter-events.jsonl").read_text().splitlines()]
    assert [(event["run_prefix"], event["client_token"]) for event in events] == [
        (adapter.config["run_prefix"], adapter.token) for adapter in (first, second)]
    assert not cli.calls


@pytest.mark.parametrize("case", ["invalid_definition", "existing_role", "existing_machine", "broad_policy", "bad_history"])
def test_unverified_watchdog_cannot_arm_or_launch(setup, case):
    adapter, cli, clock, payload = setup
    if case == "existing_role":
        cli.role = {"Arn": "not-owned"}
    elif case == "existing_machine":
        cli.machine = {"stateMachineArn": "not-owned"}
    else:
        setattr(cli, case, True)
    with pytest.raises(C.LifecycleError):
        adapter("arm_watchdog", payload)
    assert not any(operation == "run-instances" for _, operation, _, _ in cli.calls)
    if case == "bad_history":
        assert clock.elapsed == 120


def test_complete_launch_request_passes_raw_gzip_file_to_cli_without_json_user_data(setup):
    adapter, cli, _, payload = setup
    with C.Ledger(adapter.state, adapter.config["account"]) as ledger:
        record = ledger.reserve(adapter.config)
        payload.update(deadline_utc=record["deadline_utc"], started_at_utc=record["created_utc"])
        receipt = adapter("arm_watchdog", payload)
        ledger.event(record["run_prefix"], "watchdog_verified", watchdog=receipt)
        ledger.event(record["run_prefix"], "launch_requested")
        result = adapter("launch_instance", {**payload, "request": record["request"]})
        wire = next(value[2] for value in cli.calls if value[1] == "run-instances")
        assert "UserData" not in wire
        assert wire == {key: value for key, value in record["request"].items() if key != "UserData"}
        assert len(cli.user_data_payloads) == 1
        compressed = cli.user_data_payloads[0]
        assert len(compressed) <= 16384
        assert gzip.decompress(compressed).decode() == record["request"]["UserData"]
        assert compressed == gzip.compress(record["request"]["UserData"].encode(), mtime=0)
        assert (adapter.state / f"user-data-{adapter.token}.gz").read_bytes() == compressed
        transport_record = json.loads((adapter.state / f"{adapter.token}-user-data.json").read_bytes())
        assert transport_record == {
            "transport": "aws-cli-fileb-gzip", "gzip_bytes": len(compressed),
            "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
            "raw_script_sha256": hashlib.sha256(record["request"]["UserData"].encode()).hexdigest(),
            "client_token": adapter.token,
        }
        assert record["request"]["UserData"] == adapter.config["guest_user_data"]
        assert record["user_data_encoding"] == "gzip"
        assert result["client_token"] == adapter.token


@pytest.mark.parametrize("field,value", [("account", "999999999999"), ("region", "us-west-2"), ("client_token", "wrong")])
def test_payload_identity_mismatch_makes_no_aws_request(setup, field, value):
    adapter, cli, _, payload = setup
    with pytest.raises(C.LifecycleError, match="configured run identity"):
        adapter("identity", {**payload, field: value})
    assert not cli.calls


def test_existing_foreign_or_notfound_instance_is_never_terminated(setup):
    adapter, cli, _, payload = setup
    payload["instance_id"] = "i-02f1a1ba582558949"
    cli.instances = [cli.instance(adapter.token)]
    cli.instances[0]["InstanceId"] = payload["instance_id"]
    cli.instances[0]["Tags"][1]["Value"] = "protected-existing-sentinel"
    with pytest.raises(C.LifecycleError, match="tags"):
        adapter("terminate_instance", payload)
    cli.instances = []
    with pytest.raises(C.LifecycleError, match="independently observed"):
        adapter("terminate_instance", payload)
    assert not any(operation == "terminate-instances" for _, operation, _, _ in cli.calls)


def test_gate_is_conditional_write_and_exact_readback(setup):
    adapter, cli, _, payload = setup
    cli.instances = [cli.instance(adapter.token)]
    permit = {key: payload[key] for key in ("account", "region", "run_prefix", "client_token", "deadline_utc")}
    permit.update(instance_id=cli.instances[0]["InstanceId"], source_sha256=adapter.config["source_sha256"],
                  watchdog_execution_arn=adapter.config["watchdog_execution_arn"])
    returned = adapter("authorize_work", permit)
    assert returned == {**permit, "authorized": True}
    assert json.loads(cli.objects[adapter.config["run_prefix"] + "/control/start.json"]) == returned
    with pytest.raises(C.LifecycleError, match="already exists"):
        adapter("authorize_work", permit)


def test_get_object_uses_explicit_required_flags_and_positional_destination(setup):
    adapter, cli, _, _ = setup
    key = adapter.config["run_prefix"] + "/control/start.json"
    cli.objects[key] = b'{"unit_test":true}'
    assert adapter._object(key) == cli.objects[key]
    calls = [call for call in cli.calls if call[1] == "get-object"]
    assert len(calls) == 1
    assert calls[0][2] == {"Bucket": adapter.config["bucket"], "Key": key,
                           "ExpectedBucketOwner": adapter.config["account"]}
    assert len(calls[0][3]) == 1


def test_get_object_rejects_unbounded_extra_arguments_before_cli(setup):
    adapter, cli, _, _ = setup
    with pytest.raises(C.LifecycleError, match="exact bounded streaming arguments"):
        adapter._aws("s3api", "get-object", {"Bucket": "other"}, extra=("outside",))
    assert not cli.calls


def build_artifacts(tmp_path, adapter, payload, monkeypatch, mutate=None):
    root = tmp_path / "guest-artifacts"
    root.mkdir()
    hashes = {name: adapter.config["source_file_hashes"][name] for name in A.RUN_SOURCES}
    for backend in ("csim", "pci"):
        (root / backend).mkdir()
        manifest = {"backend": backend, "source_hashes": hashes,
                    "dataset": {"sha256": adapter.config["corpus_sha256"]}, "steps": 64, "seed": 0,
                    "layout": ["mocked-in-unit-test"], "evaluation": {"every_updates": 32, "windows": 2},
                    "optimizer": {"lr": 1, "threshold": 2048}, "image_id_declared": adapter.config["agfi"]}
        (root / backend / "manifest.json").write_text(json.dumps(manifest))
        (root / backend / "checkpoint_000000.json").write_text(json.dumps({
            "step": 0, "file": "checkpoint_000000.i32", "location_portable": True}))
        (root / backend / "checkpoint_000000.i32").write_bytes(b"not-native-evidence; unit fixture")
    expected_gate = {key: payload[key] for key in ("account", "region", "run_prefix", "client_token", "instance_id")}
    expected_gate.update({key: adapter.config[key] for key in ("source_sha256", "deadline_utc", "watchdog_execution_arn")})
    (root / "start_gate.json").write_text(json.dumps({**expected_gate, "authorized": True}))
    (root / "environment.json").write_text(json.dumps({key: adapter.config[key] for key in ("sdk_revision", "sdk_sha256", "source_sha256", "corpus_sha256")}))
    (root / "instance_identity.json").write_text(json.dumps({"accountId": payload["account"], "region": payload["region"], "instanceId": payload["instance_id"]}))
    for name in ("image_before", "image_after"):
        text = f"AFI 0 {adapter.config['agfi']} loaded 0 ok 0 0x10212415\n"
        (root / f"{name}.txt").write_text(text)
        (root / f"{name}.json").write_text(json.dumps(A.guest.verify_image_text(text, adapter.config["agfi"])))
    if mutate:
        mutate(root)
    bundle = tmp_path / "results.tar.gz"
    receipt = A.guest.bundle_artifacts(root, bundle, tmp_path / "receipt.json")
    completion = {"schema": "tikitorch.fpga_lm_guest_completion.v1", "status": "SUCCESS", "exit_code": 0,
                  "durable_artifacts_verified": True, **expected_gate, **receipt}
    monkeypatch.setattr(A.pilot, "layout_values", lambda shape: ["mocked-in-unit-test"])
    calls = []
    def compare(left, right):
        assert left == adapter.output / "csim" and right == adapter.output / "pci"
        calls.append((left, right))
        return {"all_checkpoint_payloads_equal": True}
    monkeypatch.setattr(A.pilot, "compare_runs", compare)
    return bundle, completion, calls


def test_retrieve_verifies_bytes_provenance_and_reruns_local_pair_comparison(setup, tmp_path, monkeypatch):
    adapter, cli, _, payload = setup
    payload["instance_id"] = "i-0123456789abcdef0"
    bundle, completion, compared = build_artifacts(tmp_path, adapter, payload, monkeypatch)
    prefix = adapter.config["run_prefix"] + "/results/"
    cli.objects[prefix + "completion.json"] = json.dumps(completion).encode()
    cli.objects[prefix + "results.tar.gz"] = bundle.read_bytes()
    assert adapter("poll_result", payload) == {"status": "success"}
    result = adapter("retrieve_results", payload)
    assert result["artifact_checks_passed"] is True and result["fresh"] is True
    assert len(compared) == 1
    with pytest.raises(FileExistsError):
        adapter("retrieve_results", payload)


@pytest.mark.parametrize("mutation", [None, "archive", "wheel", "version", "abi", "receipt", "missing_receipt"])
@pytest.mark.parametrize("tag,legacy_config", [("cp38", False), ("cp310", False), ("cp311", False), ("cp312", False),
                                             ("cp310", True), ("cp311", True), ("cp312", True)])
def test_pinned_numpy_runtime_must_match_manifest_covered_import_receipt(setup, tmp_path, monkeypatch, mutation, tag, legacy_config):
    adapter, cli, _, payload = setup
    payload["instance_id"] = "i-0123456789abcdef0"
    versions = {"cp38": "1.24.4", "cp310": "2.2.6", "cp311": "2.2.6", "cp312": "2.2.6"}
    wheels = [{"python_tag": wheel_tag,
               "filename": f"numpy-{versions[wheel_tag]}-{wheel_tag}-{wheel_tag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
               "sha256": str(index + 1) * 64} for index, wheel_tag in enumerate(versions)
              if not legacy_config or wheel_tag != "cp38"]
    selected = next(wheel for wheel in wheels if wheel["python_tag"] == tag)
    version = versions[tag]
    adapter.config.update(numpy_s3_key=adapter.config["run_prefix"] + "/numpy-wheels.tar",
                          numpy_archive_sha256="f" * 64, numpy_wheels=wheels)
    def mutate(root):
        path = root / "environment.json"
        environment = json.loads(path.read_bytes())
        environment.update(numpy=version, numpy_version_expected=version, python_tag=tag,
                           python=f"3.{tag[3:]}.15 (offline fixture)", numpy_archive_sha256="f" * 64,
                           numpy_wheel=selected)
        runtime = {"version": version, "python_tag": tag, "cache_tag": "cpython-" + tag[2:],
                   "python_version": environment["python"], "platform": "linux", "machine": "x86_64",
                   "archive_sha256": "f" * 64, "wheel": selected, "import_verified": True}
        if mutation == "archive":
            environment["numpy_archive_sha256"] = "0" * 64
        elif mutation == "wheel":
            environment["numpy_wheel"] = next(wheel for wheel in wheels if wheel != selected)
        elif mutation == "version":
            environment["numpy"] = "2.3.0"
        elif mutation == "abi":
            environment["python"] = "3.13.10 (offline fixture)"
        elif mutation == "receipt":
            runtime["import_verified"] = False
        if mutation != "missing_receipt":
            (root / "numpy_runtime.json").write_text(json.dumps(runtime))
        path.write_text(json.dumps(environment))
    bundle, completion, compared = build_artifacts(tmp_path, adapter, payload, monkeypatch, mutate)
    prefix = adapter.config["run_prefix"] + "/results/"
    cli.objects[prefix + "completion.json"] = json.dumps(completion).encode()
    cli.objects[prefix + "results.tar.gz"] = bundle.read_bytes()
    if mutation is None:
        assert adapter("retrieve_results", payload)["artifact_checks_passed"] is True
        assert len(compared) == 1
    else:
        with pytest.raises(ValueError, match="NumPy"):
            adapter("retrieve_results", payload)
        assert not compared


@pytest.mark.parametrize("field", ["source_hashes", "dataset", "optimizer"])
def test_matching_hash_bundle_with_wrong_protocol_is_not_certified(setup, tmp_path, monkeypatch, field):
    adapter, cli, _, payload = setup
    payload["instance_id"] = "i-0123456789abcdef0"
    def mutate(root):
        path = root / "pci" / "manifest.json"
        manifest = json.loads(path.read_text()); manifest[field] = {}; path.write_text(json.dumps(manifest))
    bundle, completion, compared = build_artifacts(tmp_path, adapter, payload, monkeypatch, mutate)
    prefix = adapter.config["run_prefix"] + "/results/"
    cli.objects[prefix + "completion.json"] = json.dumps(completion).encode()
    cli.objects[prefix + "results.tar.gz"] = bundle.read_bytes()
    with pytest.raises(ValueError, match="approved pilot"):
        adapter("retrieve_results", payload)
    assert not compared
    assert not (adapter.state / f"{adapter.token}-retrieval-verified.json").exists()


@pytest.mark.parametrize("target", ["/etc/passwd", "../csim/checkpoint_000000.i32", "../../private.txt", "different.i32"])
def test_remote_checkpoint_json_cannot_redirect_local_comparison(setup, tmp_path, monkeypatch, target):
    adapter, cli, _, payload = setup
    payload["instance_id"] = "i-0123456789abcdef0"
    def mutate(root):
        path = root / "pci" / "checkpoint_000000.json"
        record = json.loads(path.read_text()); record["file"] = target
        path.write_text(json.dumps(record))
    bundle, completion, compared = build_artifacts(tmp_path, adapter, payload, monkeypatch, mutate)
    prefix = adapter.config["run_prefix"] + "/results/"
    cli.objects[prefix + "completion.json"] = json.dumps(completion).encode()
    cli.objects[prefix + "results.tar.gz"] = bundle.read_bytes()
    with pytest.raises(ValueError, match="same-backend"):
        adapter("retrieve_results", payload)
    assert not compared


@pytest.mark.parametrize("stage", ["health", "mutation_identity"])
def test_health_or_final_identity_delay_cannot_launch_after_deadline(setup, monkeypatch, stage):
    adapter, cli, clock, payload = setup
    with C.Ledger(adapter.state, adapter.config["account"], now=clock.now) as ledger:
        record = ledger.reserve(adapter.config)
        ledger.event(record["run_prefix"], "launch_requested")
        def health(ignored):
            if stage == "health":
                clock.sleep(3600)
            else:
                original = adapter._account
                def delayed_account():
                    result = original(); clock.sleep(3600); return result
                monkeypatch.setattr(adapter, "_account", delayed_account)
            return {"verified": True}
        monkeypatch.setattr(adapter, "_health", health)
        with pytest.raises(TimeoutError, match="immediately before RunInstances"):
            adapter("launch_instance", {**payload, "request": record["request"]})
    assert not any(operation == "run-instances" for _, operation, _, _ in cli.calls)


@pytest.mark.parametrize("name,kind", [("../outside", "file"), ("/outside", "file"), ("linked", "symlink"), ("duplicate", "duplicate")])
def test_archive_rejects_traversal_links_duplicates_before_extraction(tmp_path, name, kind):
    bundle = tmp_path / "bad.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        item = tarfile.TarInfo(name)
        if kind == "symlink":
            item.type = tarfile.SYMTYPE; item.linkname = "/etc/passwd"
        archive.addfile(item, io.BytesIO(b""))
        if kind == "duplicate":
            archive.addfile(item, io.BytesIO(b""))
    receipt = {"bundle_bytes": bundle.stat().st_size, "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="unsafe"):
        A.extract_results(bundle, tmp_path / "output", receipt)
    assert not (tmp_path / "output").exists()


def test_cli_status_and_unapproved_run_are_offline(tmp_path):
    script = Path(A.__file__)
    status = subprocess.run([sys.executable, "-S", str(script), "status"], cwd=tmp_path,
                            text=True, capture_output=True, timeout=10)
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["offline"] is True
    run = subprocess.run([sys.executable, "-S", str(script), "run"], cwd=tmp_path,
                         text=True, capture_output=True, timeout=10)
    assert run.returncode == 2 and "explicit --execute" in run.stderr
    cancel = subprocess.run([sys.executable, "-S", str(script), "cancel-unlaunched"], cwd=tmp_path,
                            text=True, capture_output=True, timeout=10)
    assert cancel.returncode == 2 and "explicit --execute" in cancel.stderr
    assert not list(tmp_path.iterdir())


def test_cancel_cli_uses_only_authenticated_read_only_discovery(setup, monkeypatch, capsys):
    adapter, cli, _, _ = setup
    with C.Ledger(adapter.state, adapter.config["account"]) as ledger:
        ledger.reserve(adapter.config)
        ledger.event(adapter.config["run_prefix"], "controller_failed")
    config_path = adapter.state.parent / "cancel-config.json"
    config_path.write_text(json.dumps(adapter.config))
    monkeypatch.setattr(A, "Adapter", lambda config, state, output: adapter)
    assert A.main(["cancel-unlaunched", "--execute", "--config", str(config_path), "--state", str(adapter.state)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == C.CANCELLED and "termination_confirmed" not in result
    assert {(service, operation) for service, operation, _, _ in cli.calls} <= {
        ("sts", "get-caller-identity"), ("ec2", "describe-instances")}


@pytest.mark.parametrize("changed", ["tags", "type"])
def test_cancel_cannot_hide_same_token_instance_with_changed_ownership(setup, changed):
    adapter, cli, _, _ = setup
    with C.Ledger(adapter.state, adapter.config["account"]) as ledger:
        ledger.reserve(adapter.config)
    cli.instances = [cli.instance(adapter.token)]
    if changed == "tags":
        cli.instances[0]["Tags"] = []
    else:
        cli.instances[0]["InstanceType"] = "f2.12xlarge"
    before = (adapter.state / "ledger.json").read_bytes()
    with pytest.raises(C.LifecycleError):
        C.cancel_before_launch(adapter.state, adapter.config["account"], adapter.config["run_prefix"], adapter)
    request = next(payload for service, operation, payload, _ in cli.calls if operation == "describe-instances")
    assert request["Filters"] == [{"Name": "client-token", "Values": [adapter.token]}]
    assert (adapter.state / "ledger.json").read_bytes() == before


@pytest.mark.parametrize("response", [{}, {"Reservations": None}, {"Reservations": [{}]},
                                      {"Reservations": [{"OwnerId": "123456789012", "Instances": None}]},
                                      {"Reservations": [], "NextToken": None},
                                      {"Reservations": [], "NextToken": ""},
                                      {"Reservations": [], "NextToken": 123}])
def test_cancel_requires_explicit_complete_well_formed_instance_absence(setup, monkeypatch, response):
    adapter, _, _, _ = setup
    with C.Ledger(adapter.state, adapter.config["account"]) as ledger:
        ledger.reserve(adapter.config)
    original = adapter._aws
    def malformed(service, operation, payload, **kwargs):
        if (service, operation) == ("ec2", "describe-instances"):
            return response
        return original(service, operation, payload, **kwargs)
    monkeypatch.setattr(adapter, "_aws", malformed)
    before = (adapter.state / "ledger.json").read_bytes()
    with pytest.raises(C.LifecycleError):
        C.cancel_before_launch(adapter.state, adapter.config["account"], adapter.config["run_prefix"], adapter)
    assert (adapter.state / "ledger.json").read_bytes() == before


@pytest.mark.parametrize("case", ["all_empty", "later_match", "repeated_token", "page_bound", "time_bound", "malformed_later"])
def test_cancel_exhausts_bounded_manual_pages_before_certifying_absence(setup, case):
    adapter, cli, clock, _ = setup
    with C.Ledger(adapter.state, adapter.config["account"]) as ledger:
        ledger.reserve(adapter.config)
    cli.discovery_pages = [{"Reservations": [], "NextToken": "first-page-token"}, {"Reservations": []}]
    if case == "later_match":
        cli.discovery_pages[1] = {"Reservations": [{"OwnerId": adapter.config["account"],
                                                     "Instances": [cli.instance(adapter.token)]}]}
    elif case == "repeated_token":
        cli.discovery_pages[1]["NextToken"] = "first-page-token"
    elif case == "page_bound":
        cli.discovery_pages = [{"Reservations": [], "NextToken": f"page-{index}"} for index in range(11)]
    elif case == "time_bound":
        cli.discovery_delay = lambda: clock.sleep(31)
    elif case == "malformed_later":
        cli.discovery_pages[1] = {}
    before = (adapter.state / "ledger.json").read_bytes()
    if case == "all_empty":
        result = C.cancel_before_launch(adapter.state, adapter.config["account"], adapter.config["run_prefix"], adapter)
        assert result["status"] == C.CANCELLED
    else:
        with pytest.raises(C.LifecycleError):
            C.cancel_before_launch(adapter.state, adapter.config["account"], adapter.config["run_prefix"], adapter)
        assert (adapter.state / "ledger.json").read_bytes() == before
    discovery = [row for row in cli.calls if row[1] == "describe-instances"]
    assert len(discovery) == (10 if case == "page_bound" else 2)
    for _, _, request, extra in discovery:
        assert request["Filters"] == [{"Name": "client-token", "Values": [adapter.token]}]
        assert request["MaxResults"] == 1000 and extra == ["--no-paginate"]
    assert "NextToken" not in discovery[0][2]
    assert discovery[1][2]["NextToken"] == ("page-0" if case == "page_bound" else "first-page-token")
    assert adapter.api_deadline is None


def test_unknown_actions_and_custom_endpoints_are_forbidden(setup, monkeypatch):
    adapter, cli, _, payload = setup
    with pytest.raises(ValueError, match="unapproved"):
        adapter("delete_bucket", payload)
    assert not cli.calls
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://unapproved.example")
    with pytest.raises(ValueError, match="endpoint"):
        A.Adapter(adapter.config, adapter.state)


def test_nonfinite_json_is_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        A.strict_json('{"bad":NaN}')
