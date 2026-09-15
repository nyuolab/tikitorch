#!/usr/bin/env python3
"""Explicit, narrow AWS CLI adapter for the reviewed FPGA LM pilot controller.

Help/status are offline. Run/recover require ``--execute``; the explicit
``cancel-unlaunched --execute`` uses read-only AWS checks and local ledger writes.
No shared IAM/network/bootstrap policy is changed. The only IAM
role and state machine this adapter creates are new deterministic per-run
watchdog resources. Resource identity, policies, execution history and artifact
bytes are independently queried; request echoes are never arming evidence.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import unquote

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fpga_lm_aws as preparation
from scripts import fpga_lm_control as control
from scripts import fpga_lm_guest as guest
from scripts import fpga_lm_pretrain as pilot
from scripts import fpga_lm_watchdog as watchdog


RUN_SOURCES = (
    "scripts/fpga_lm_pretrain.py", "scripts/fpga_lm_layout.cpp",
    "tikitorch/fpga/silicon/gen_buffer_init.py", "tikitorch/fpga/bridge.cpp",
    "tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp",
    "tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp",
    "tikitorch/hardware/hls/forge_kernels/forge_ce_lut.hpp",
    "tikitorch/hardware/hls/forge_hls_types.hpp",
)
MUTATIONS = {("iam", "create-role"), ("iam", "put-role-policy"),
             ("stepfunctions", "create-state-machine"), ("stepfunctions", "start-execution"),
             ("ec2", "run-instances"), ("ec2", "terminate-instances"), ("s3api", "put-object")}


class AWSFailure(control.LifecycleError):
    def __init__(self, service, operation, code):
        self.code = code
        super().__init__(f"AWS {service}/{operation} failed ({code}); details intentionally omitted")


def document(value):
    return json.loads(unquote(value)) if isinstance(value, str) else value


def strict_json(data):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(data, parse_constant=invalid)


def verify_numpy_runtime(config, environment, runtime):
    """Bind the optional offline runtime to its approved wheel and Python ABI."""
    versions = {"cp38": "1.24.4", "cp310": "2.2.6", "cp311": "2.2.6", "cp312": "2.2.6"}
    fields = {"numpy_s3_key", "numpy_archive_sha256", "numpy_wheels"}
    if not fields <= set(config):
        raise ValueError("incomplete approved NumPy runtime configuration")
    digest = config["numpy_archive_sha256"]
    wheels = config["numpy_wheels"]
    if (config["numpy_s3_key"] != config["run_prefix"] + "/numpy-wheels.tar"
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(wheels, list) or len(wheels) not in (3, 4)):
        raise ValueError("invalid approved NumPy runtime artifact configuration")
    by_tag = {}
    for wheel in wheels:
        if not isinstance(wheel, dict) or set(wheel) != {"python_tag", "filename", "sha256"}:
            raise ValueError("invalid approved NumPy wheel schema")
        tag = wheel["python_tag"]
        if (not isinstance(tag, str) or tag not in versions or tag in by_tag
                or wheel["filename"] != f"numpy-{versions[tag]}-{tag}-{tag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
                or not isinstance(wheel["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"]) is None):
            raise ValueError("NumPy wheel is outside the approved version/platform/ABI allowlist")
        by_tag[tag] = wheel
    # Historical three-wheel configurations remain independently verifiable;
    # new recipes add exactly CPython 3.8, without silently dropping an ABI.
    if set(by_tag) not in ({"cp310", "cp311", "cp312"}, set(versions)):
        raise ValueError("NumPy wheel set does not match a supported explicit runtime recipe")
    if not isinstance(environment, dict) or not isinstance(runtime, dict):
        raise ValueError("NumPy runtime provenance records must be objects")
    tag = environment.get("python_tag")
    if not isinstance(tag, str) or tag not in by_tag:
        raise ValueError("retrieved NumPy runtime Python ABI is not approved")
    python_version = environment.get("python")
    if (not isinstance(python_version, str)
            or re.match(r"3\." + tag[3:] + r"\.[0-9]+(?:[+ ]|$)", python_version) is None):
        raise ValueError("retrieved NumPy runtime Python version/ABI mismatch")
    version = versions[tag]
    expected_environment = {"numpy": version, "numpy_version_expected": version,
                            "numpy_archive_sha256": digest, "numpy_wheel": by_tag[tag]}
    if any(environment.get(key) != value for key, value in expected_environment.items()):
        raise ValueError("retrieved NumPy environment differs from the approved runtime")
    expected_runtime = {"version": version, "python_tag": tag, "cache_tag": "cpython-" + tag[2:],
                        "python_version": python_version, "platform": "linux", "machine": "x86_64",
                        "archive_sha256": digest, "wheel": by_tag[tag], "import_verified": True}
    if runtime != expected_runtime or runtime.get("import_verified") is not True:
        raise ValueError("retrieved NumPy bootstrap receipt differs from the approved runtime")


def extract_results(bundle, output, receipt):
    """Verify the receipt and complete manifest before exclusive safe extraction."""
    bundle = Path(bundle)
    if (bundle.stat().st_size != receipt["bundle_bytes"]
            or hashlib.sha256(bundle.read_bytes()).hexdigest() != receipt["bundle_sha256"]):
        raise ValueError("downloaded result bundle does not match completion SHA256/size")
    output = Path(output)
    if output.exists():
        raise FileExistsError("result extraction requires a fresh directory")
    with tarfile.open(bundle) as archive:
        members, names, total = [], set(), 0
        for member in archive:
            name = member.name
            relative = PurePosixPath(name)
            total += member.size
            if (len(members) >= 200 or total > 128 * 1024 * 1024
                    or not member.isfile() or member.size < 0 or member.size > 64 * 1024 * 1024
                    or name in names or relative.is_absolute() or ".." in relative.parts
                    or not relative.parts or relative.as_posix() != name or "\\" in name):
                raise ValueError("unsafe, duplicate or oversized result archive member")
            names.add(name); members.append(member)
        if "artifact_manifest.json" not in names:
            raise ValueError("result archive lacks its full artifact manifest")
        raw_manifest = archive.extractfile("artifact_manifest.json").read()
        if hashlib.sha256(raw_manifest).hexdigest() != receipt["artifact_manifest_sha256"]:
            raise ValueError("artifact manifest digest mismatch")
        manifest = strict_json(raw_manifest)
        if (not isinstance(manifest, dict) or len(manifest) != receipt["artifact_count"]
                or set(manifest) != names - {"artifact_manifest.json"}):
            raise ValueError("artifact manifest coverage mismatch")
        payloads = {"artifact_manifest.json": raw_manifest}
        for name, record in manifest.items():
            data = archive.extractfile(name).read()
            if (type(record.get("bytes")) is not int or record["bytes"] != len(data)
                    or record.get("sha256") != hashlib.sha256(data).hexdigest()):
                raise ValueError("artifact payload hash/size mismatch")
            payloads[name] = data
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, data in payloads.items():
        path = output / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return manifest


class Adapter:
    def __init__(self, config, state, output=None, *, runner=subprocess.run,
                 monotonic=time.monotonic, sleep=time.sleep, now=control.utc_now):
        self.config, self.state = dict(config), Path(state)
        self.output = Path(output) if output is not None else None
        self.runner, self.monotonic, self.sleep = runner, monotonic, sleep
        self.now = now
        for key, pattern in (("account", r"[0-9]{12}"), ("profile", r"[A-Za-z0-9_.@-]{1,128}"),
                             ("region", r"us-(?:east-1|west-2)"),
                             ("run_prefix", r"tikitorch-fpga-lm/[a-z0-9][a-z0-9-]{7,79}")):
            preparation.matched(config, key, pattern)
        self.token = hashlib.sha256(f"{config['account']}:{config['region']}:{config['run_prefix']}".encode()).hexdigest()
        self.artifacts = None
        self.completion = None
        self.api_deadline = None
        self.launch_deadline_monotonic = None
        if self.output is not None:
            hashes = config.get("source_file_hashes", {})
            if (not isinstance(hashes, dict) or not set(RUN_SOURCES) <= set(hashes)
                    or not set(hashes) <= set(preparation.SOURCE_FILES)
                    or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes.values())):
                raise ValueError("approved source_file_hashes must cover every native source with SHA256s")
        if any(key.startswith("AWS_ENDPOINT_URL") for key in os.environ):
            raise ValueError("custom AWS endpoint overrides are forbidden for this adapter")

    def _event(self, **facts):
        # The controller creates and locks this directory before adapter calls.
        if not self.state.is_dir() or self.state.is_symlink():
            raise control.LifecycleError("adapter event directory must be the owned controller state")
        event = {"at": datetime.now(timezone.utc).isoformat(), **facts,
                 "run_prefix": self.config["run_prefix"], "client_token": self.token}
        fd = os.open(self.state / "adapter-events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "ab") as stream:
            stream.write(control.encoded(event).replace(b"\n", b" ").rstrip() + b"\n")
            stream.flush(); os.fsync(stream.fileno())

    def _aws(self, service, operation, payload, *, extra=()):
        if (service, operation) in MUTATIONS:
            self._account()
        if (service, operation) == ("ec2", "run-instances"):
            absolute = datetime.fromisoformat(self.config["deadline_utc"].replace("Z", "+00:00"))
            if (absolute.tzinfo is None or self.now() >= absolute
                    or self.launch_deadline_monotonic is None
                    or self.monotonic() >= self.launch_deadline_monotonic):
                raise TimeoutError("pilot deadline elapsed immediately before RunInstances")
        command = ["aws", "--profile", self.config["profile"], "--region", self.config["region"],
                   "--no-cli-pager", "--output", "json", "--cli-connect-timeout", "10",
                   "--cli-read-timeout", "30", service, operation]
        if (service, operation) == ("s3api", "get-object"):
            # AWS CLI v2's customized streaming GetObject validates required
            # command-line flags before merging --cli-input-json. Supply its
            # three exact allowlisted inputs as flags, with a positional file.
            if set(payload) != {"Bucket", "Key", "ExpectedBucketOwner"} or len(extra) != 1:
                raise control.LifecycleError("GetObject requires exact bounded streaming arguments")
            command += ["--bucket", payload["Bucket"], "--key", payload["Key"],
                        "--expected-bucket-owner", payload["ExpectedBucketOwner"], str(extra[0])]
        else:
            command += ["--cli-input-json", json.dumps(payload, allow_nan=False), *map(str, extra)]
        timeout = 40 if self.api_deadline is None else min(40, self.api_deadline - self.monotonic())
        if timeout <= 0:
            raise AWSFailure(service, operation, "Deadline")
        try:
            result = self.runner(command, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            self._event(service=service, operation=operation, outcome="timeout")
            raise AWSFailure(service, operation, "Timeout") from None
        if result.returncode:
            match = re.search(r"An error occurred \(([A-Za-z0-9_.-]{1,80})\)", result.stderr)
            code = match.group(1) if match else "CLIError"
            self._event(service=service, operation=operation, outcome="failed", code=code)
            raise AWSFailure(service, operation, code)
        self._event(service=service, operation=operation, outcome="success")
        return strict_json(result.stdout) if result.stdout.strip() else {}

    def _account(self):
        identity = self._aws("sts", "get-caller-identity", {})
        if identity.get("Account") != self.config["account"]:
            raise control.LifecycleError("authenticated AWS account differs from explicit configuration")
        return identity

    def _guard(self, payload):
        expected = {key: self.config[key] for key in ("account", "region", "run_prefix")}
        expected["client_token"] = self.token
        if any(payload.get(key) != value for key, value in expected.items()):
            raise control.LifecycleError("adapter action does not match the exact configured run identity")
        self._account()

    def _missing(self, service, operation, payload, allowed):
        try:
            self._aws(service, operation, payload)
        except AWSFailure as error:
            if error.code in allowed:
                return
            raise
        raise control.LifecycleError("a proposed new run-scoped resource already exists; no overwrite permitted")

    def _record_new(self, name, value):
        # The ledger/event log span the cumulative campaign, but immutable
        # receipts belong to one client token. Never collide with earlier runs
        # or migrate/overwrite their historical top-level receipt filenames.
        with (self.state / f"{self.token}-{name}").open("xb") as stream:
            stream.write(control.encoded(value)); stream.flush(); os.fsync(stream.fileno())

    def _observed(self, reservation, instance):
        zone = instance.get("Placement", {}).get("AvailabilityZone", "")
        if reservation.get("OwnerId") != self.config["account"] or not zone.startswith(self.config["region"]):
            raise control.LifecycleError("EC2 response owner/placement region mismatch")
        if instance.get("InstanceType") != "f2.6xlarge":
            raise control.LifecycleError("observed instance is outside the approved f2.6xlarge scope")
        observed = {"instance_id": instance["InstanceId"], "account": reservation["OwnerId"],
                    "region": self.config["region"], "client_token": instance.get("ClientToken"),
                    "state": instance.get("State", {}).get("Name"),
                    "tags": {item["Key"]: item["Value"] for item in instance.get("Tags", [])}}
        control.owned_instance(observed, {**self.config, "client_token": self.token})
        return observed

    def _describe(self, instance_id):
        response = self._aws("ec2", "describe-instances", {"InstanceIds": [instance_id]})
        observed = [self._observed(reservation, instance)
                    for reservation in response.get("Reservations", []) for instance in reservation.get("Instances", [])]
        if len(observed) != 1 or observed[0]["instance_id"] != instance_id:
            raise control.LifecycleError("exact instance could not be independently observed")
        return observed[0]

    def _find_instances(self):
        """Exhaust exact-token discovery before certifying absence (10 pages/60s)."""
        previous = self.api_deadline
        until = self.monotonic() + 60
        self.api_deadline = min(previous, until) if previous is not None else until
        until = self.api_deadline
        observed, tokens, instance_ids = [], set(), set()
        request = {"Filters": [{"Name": "client-token", "Values": [self.token]}], "MaxResults": 1000}
        try:
            for _ in range(10):
                if self.monotonic() >= until:
                    raise control.LifecycleError("run discovery exhausted its 60-second deadline")
                response = self._aws("ec2", "describe-instances", request, extra=("--no-paginate",))
                if self.monotonic() >= until:
                    raise control.LifecycleError("run discovery exhausted its 60-second deadline")
                if not isinstance(response, dict) or not isinstance(response.get("Reservations"), list):
                    raise control.LifecycleError("run discovery lacks an explicit reservation list")
                for reservation in response["Reservations"]:
                    if (not isinstance(reservation, dict) or reservation.get("OwnerId") != self.config["account"]
                            or not isinstance(reservation.get("Instances"), list)):
                        raise control.LifecycleError("run discovery reservation owner/instance list is malformed")
                    for instance in reservation["Instances"]:
                        if not isinstance(instance, dict):
                            raise control.LifecycleError("run discovery instance is malformed")
                        found = self._observed(reservation, instance)
                        if found["instance_id"] in instance_ids:
                            raise control.LifecycleError("run discovery repeated an instance across pages")
                        instance_ids.add(found["instance_id"])
                        observed.append(found)
                if "NextToken" not in response:
                    return observed
                token = response["NextToken"]
                if not isinstance(token, str) or not token.strip() or token in tokens:
                    raise control.LifecycleError("run discovery pagination token is malformed or repeated")
                tokens.add(token)
                request = {**request, "NextToken": token}
            raise control.LifecycleError("run discovery exceeded its ten-page bound")
        finally:
            self.api_deadline = previous

    def _role_matches(self, artifacts):
        role = self._aws("iam", "get-role", {"RoleName": artifacts["role_name"]})["Role"]
        if role.get("Arn") != artifacts["role_arn"] or document(role.get("AssumeRolePolicyDocument")) != artifacts["trust_policy"]:
            raise control.LifecycleError("watchdog role ARN/trust policy mismatch")
        expected_tags = {"Project": preparation.PROJECT_TAG, "Run": self.config["run_prefix"].split("/")[-1],
                         "OwnerAccount": self.config["account"]}
        if {item["Key"]: item["Value"] for item in role.get("Tags", [])} != expected_tags:
            raise control.LifecycleError("watchdog role ownership tags mismatch")
        inline = self._aws("iam", "get-role-policy", {"RoleName": artifacts["role_name"], "PolicyName": "terminate-owned-pilot"})
        names = self._aws("iam", "list-role-policies", {"RoleName": artifacts["role_name"]})
        attached = self._aws("iam", "list-attached-role-policies", {"RoleName": artifacts["role_name"]})
        if (document(inline.get("PolicyDocument")) != artifacts["permissions_policy"]
                or names.get("PolicyNames") != ["terminate-owned-pilot"] or names.get("IsTruncated")
                or attached.get("AttachedPolicies") != [] or attached.get("IsTruncated")):
            raise control.LifecycleError("watchdog role permissions differ from the exact reviewed policy")

    def _machine_matches(self, artifacts):
        machine = self._aws("stepfunctions", "describe-state-machine", {"stateMachineArn": artifacts["state_machine_arn"]})
        if (machine.get("stateMachineArn") != artifacts["state_machine_arn"] or machine.get("type") != "STANDARD"
                or machine.get("roleArn") != artifacts["role_arn"] or document(machine.get("definition")) != artifacts["definition"]):
            raise control.LifecycleError("watchdog state machine definition/type/role mismatch")
        self._role_matches(artifacts)

    def _health_checks(self, payload):
        artifacts = self.artifacts or watchdog.build_artifacts(payload)
        self._machine_matches(artifacts)
        execution = self._aws("stepfunctions", "describe-execution", {"executionArn": artifacts["execution_arn"]})
        if (execution.get("executionArn") != artifacts["execution_arn"]
                or execution.get("stateMachineArn") != artifacts["state_machine_arn"]
                or execution.get("status") != "RUNNING" or document(execution.get("input")) != {}):
            raise control.LifecycleError("watchdog execution identity/input/status mismatch")
        history = self._aws("stepfunctions", "get-execution-history", {
            "executionArn": artifacts["execution_arn"], "maxResults": 1000,
            "reverseOrder": False, "includeExecutionData": True})
        if history.get("nextToken"):
            raise control.LifecycleError("unexpected paginated startup history")
        events = history.get("events", [])
        entered = [event.get("stateEnteredEventDetails", {}).get("name")
                   for event in events if event.get("type", "").endswith("StateEntered")]
        exits = [event for event in events if event.get("type") == "TaskStateExited"
                 and event.get("stateExitedEventDetails", {}).get("name") == "StartupHealthCheck"]
        succeeded = [event for event in events if event.get("type") == "TaskSucceeded"]
        if not entered or entered[-1] != "WaitUntilDeadline" or not exits or not succeeded:
            raise control.LifecycleError("watchdog startup health has not reached its verified wait state")
        if not any(event.get("id", 0) < exits[0].get("id", 0) for event in succeeded):
            raise control.LifecycleError("StartupHealthCheck lacks a preceding successful task event")
        return {"verified": True, "execution_arn": execution["executionArn"],
                **{key: artifacts[key] for key in ("account", "region", "run_prefix", "client_token", "deadline_utc")},
                "startup_health_check_passed": True, "status": execution["status"], "state": entered[-1]}

    def _health(self, payload):
        previous = self.api_deadline
        self.api_deadline = min(previous, self.monotonic() + 120) if previous is not None else self.monotonic() + 120
        try:
            return self._health_checks(payload)
        finally:
            self.api_deadline = previous

    def _arm(self, payload):
        artifacts = watchdog.build_artifacts(payload)
        if artifacts["execution_arn"] != self.config["watchdog_execution_arn"] or payload["execution_arn"] != artifacts["execution_arn"]:
            raise control.LifecycleError("deterministic watchdog execution ARN mismatch")
        validation = self._aws("stepfunctions", "validate-state-machine-definition", {
            "definition": json.dumps(artifacts["definition"]), "type": "STANDARD", "severity": "ERROR", "maxResults": 100})
        if validation.get("result") != "OK" or validation.get("truncated"):
            raise control.LifecycleError("AWS rejected or truncated validation of the exact watchdog definition")
        self._missing("iam", "get-role", {"RoleName": artifacts["role_name"]}, {"NoSuchEntity"})
        self._missing("stepfunctions", "describe-state-machine", {"stateMachineArn": artifacts["state_machine_arn"]}, {"StateMachineDoesNotExist"})
        self._record_new("watchdog-artifacts.json", artifacts)
        tags = [{"Key": "Project", "Value": preparation.PROJECT_TAG},
                {"Key": "Run", "Value": self.config["run_prefix"].split("/")[-1]},
                {"Key": "OwnerAccount", "Value": self.config["account"]}]
        self._aws("iam", "create-role", {"RoleName": artifacts["role_name"],
                  "AssumeRolePolicyDocument": json.dumps(artifacts["trust_policy"]), "Tags": tags})
        self._aws("iam", "put-role-policy", {"RoleName": artifacts["role_name"], "PolicyName": "terminate-owned-pilot",
                  "PolicyDocument": json.dumps(artifacts["permissions_policy"])})
        self._role_matches(artifacts)
        self._aws("stepfunctions", "create-state-machine", {
            "name": artifacts["state_machine_name"], "definition": json.dumps(artifacts["definition"]),
            "roleArn": artifacts["role_arn"], "type": "STANDARD",
            "tags": [{"key": tag["Key"], "value": tag["Value"]} for tag in tags]})
        self.artifacts = artifacts
        self._machine_matches(artifacts)
        self._aws("stepfunctions", "start-execution", {"stateMachineArn": artifacts["state_machine_arn"],
                  "name": artifacts["execution_name"], "input": "{}"})
        until = self.monotonic() + 120
        previous, self.api_deadline = self.api_deadline, until
        try:
            while True:
                try:
                    receipt = self._health(payload)
                    self._record_new("watchdog-armed.json", receipt)
                    return receipt
                except control.LifecycleError:
                    if self.monotonic() >= until:
                        raise
                    self.sleep(min(3, max(0, until - self.monotonic())))
        finally:
            self.api_deadline = previous

    def _object(self, key, *, limit=65536):
        args = {"Bucket": self.config["bucket"], "Key": key, "ExpectedBucketOwner": self.config["account"]}
        try:
            head = self._aws("s3api", "head-object", args)
        except AWSFailure as error:
            if error.code in {"NoSuchKey", "NotFound", "404"}:
                return None
            raise
        if type(head.get("ContentLength")) is not int or not 0 < head["ContentLength"] <= limit:
            raise ValueError("S3 object exceeds its reviewed size bound")
        with tempfile.TemporaryDirectory(prefix="tiki-object-", dir=self.state) as directory:
            destination = Path(directory) / "object"
            self._aws("s3api", "get-object", args, extra=(destination,))
            data = destination.read_bytes()
            if len(data) != head["ContentLength"]:
                raise ValueError("S3 object size changed during retrieval")
            return data

    def _completion(self, payload):
        raw = self._object(self.config["run_prefix"] + "/results/completion.json")
        if raw is None:
            return None
        record = strict_json(raw)
        expected = {**{key: payload[key] for key in ("account", "region", "run_prefix", "client_token", "instance_id")},
                    **{key: self.config[key] for key in ("source_sha256", "deadline_utc", "watchdog_execution_arn")}}
        if record.get("schema") != "tikitorch.fpga_lm_guest_completion.v1" or any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("guest completion is not bound to the approved instance/run/source/deadline")
        return record

    def _retrieve(self, payload):
        receipt = self._completion(payload)
        if receipt is None or receipt.get("status") != "SUCCESS" or receipt.get("exit_code") != 0 or receipt.get("durable_artifacts_verified") is not True:
            raise ValueError("complete successful guest artifacts are required for retrieval certification")
        if (receipt.get("bundle_name") != "results.tar.gz" or type(receipt.get("bundle_bytes")) is not int
                or not 0 < receipt["bundle_bytes"] <= 64 * 1024 * 1024):
            raise ValueError("invalid bounded result bundle receipt")
        if self.output is None or self.output.exists():
            raise FileExistsError("retrieval requires a new explicit output directory")
        data = self._object(self.config["run_prefix"] + "/results/results.tar.gz", limit=64 * 1024 * 1024)
        if data is None:
            raise ValueError("declared result bundle is missing")
        with tempfile.TemporaryDirectory(prefix="tiki-result-", dir=self.state) as directory:
            archive = Path(directory) / "results.tar.gz"
            archive.write_bytes(data)
            artifact_files = extract_results(archive, self.output, receipt)
        expected_sources = {name: self.config["source_file_hashes"][name] for name in RUN_SOURCES}
        for backend in ("csim", "pci"):
            # Remote checkpoint JSON must never redirect the local comparator to
            # an absolute, parent, other-backend or unmanifested local file.
            for path in (self.output / backend).glob("checkpoint_*.json"):
                checkpoint = strict_json(path.read_bytes())
                step = checkpoint.get("step")
                if type(step) is not int or not 0 <= step <= 64:
                    raise ValueError("invalid remote checkpoint step")
                expected_file = f"checkpoint_{step:06d}.i32"
                expected_record = f"checkpoint_{step:06d}.json"
                target = self.output / backend / expected_file
                if (checkpoint.get("location_portable") is not True
                        or checkpoint.get("file") != expected_file or path.name != expected_record
                        or f"{backend}/{expected_file}" not in artifact_files
                        or f"{backend}/{expected_record}" not in artifact_files
                        or not target.is_file() or target.is_symlink()):
                    raise ValueError("remote checkpoint must name its same-backend manifest-covered portable payload")
            manifest = strict_json((self.output / backend / "manifest.json").read_bytes())
            if (manifest.get("backend") != backend or manifest.get("source_hashes") != expected_sources
                    or manifest.get("dataset", {}).get("sha256") != self.config["corpus_sha256"]
                    or manifest.get("steps") != 64 or manifest.get("seed") != 0
                    or manifest.get("layout") != pilot.layout_values(pilot.layout(512))
                    or manifest.get("evaluation", {}).get("every_updates") != 32
                    or manifest.get("evaluation", {}).get("windows") != 2
                    or manifest.get("optimizer", {}).get("lr") != 1 or manifest.get("optimizer", {}).get("threshold") != 2048):
                raise ValueError("retrieved native protocol/source/corpus does not match the approved pilot")
            if backend == "pci" and manifest.get("image_id_declared") != self.config["agfi"]:
                raise ValueError("retrieved PCI image declaration mismatch")
        environment = strict_json((self.output / "environment.json").read_bytes())
        for key in ("sdk_revision", "sdk_sha256", "source_sha256", "corpus_sha256"):
            if environment.get(key) != self.config[key]:
                raise ValueError("retrieved environment provenance mismatch")
        if {"numpy_s3_key", "numpy_archive_sha256", "numpy_wheels"} & set(self.config):
            if "numpy_runtime.json" not in artifact_files:
                raise ValueError("retrieved NumPy bootstrap receipt is not manifest-covered")
            verify_numpy_runtime(self.config, environment,
                                 strict_json((self.output / "numpy_runtime.json").read_bytes()))
        expected_gate = {**{key: payload[key] for key in ("account", "region", "run_prefix", "client_token", "instance_id")},
                         **{key: self.config[key] for key in ("source_sha256", "deadline_utc", "watchdog_execution_arn")}, "authorized": True}
        if strict_json((self.output / "start_gate.json").read_bytes()) != expected_gate:
            raise ValueError("retrieved guest start permit mismatch")
        identity = strict_json((self.output / "instance_identity.json").read_bytes())
        if any(identity.get(key) != value for key, value in {"accountId": payload["account"], "region": payload["region"], "instanceId": payload["instance_id"]}.items()):
            raise ValueError("retrieved IMDS instance identity mismatch")
        for name in ("image_before", "image_after"):
            image = guest.verify_image_text((self.output / f"{name}.txt").read_text(), self.config["agfi"])
            if strict_json((self.output / f"{name}.json").read_bytes()) != image:
                raise ValueError("retrieved loaded-image proof mismatch")
        comparison = pilot.compare_runs(self.output / "csim", self.output / "pci")
        if comparison.get("all_checkpoint_payloads_equal") is not True:
            raise ValueError("independent local full-checkpoint comparison failed")
        self._record_new("retrieval-verified.json", {"completion": receipt, "comparison": comparison,
                                                   "directory": str(self.output.resolve())})
        return {"fresh": True, "run_prefix": payload["run_prefix"],
                "directory": str(self.output.resolve()), "artifact_checks_passed": True}

    def __call__(self, action, payload):
        if action not in {"identity", "find_by_client_token", "launch_instance", "describe_instance", "arm_watchdog",
                          "watchdog_health", "authorize_work", "poll_result", "retrieve_results", "terminate_instance"}:
            raise ValueError("unknown/unapproved adapter action")
        self._guard(payload)
        if action == "identity":
            return {"account": self._account()["Account"], "region": self.config["region"]}
        if action == "find_by_client_token":
            # Discover every exact-token match first: filtering on mutable tags
            # or type could hide an unexpected instance and falsely certify
            # absence. Every observed match must still pass ownership checks.
            return self._find_instances()
        if action == "arm_watchdog":
            return self._arm(payload)
        if action == "watchdog_health":
            return self._health(payload)
        if action == "launch_instance":
            empty = {"account": self.config["account"], "currency": "USD", "ceiling_usd": "1000", "runs": []}
            expected = preparation.plan(self.config, empty)["proposed_run_instances"]
            expected["UserData"] = self.config["guest_user_data"]
            if payload["request"] != expected:
                raise control.LifecycleError("launch request differs from the exact reviewed single-instance plan")
            record = strict_json((self.state / "ledger.json").read_bytes())["runs"][-1]
            if record.get("status") != "launch_requested" or record.get("request") != expected:
                raise control.LifecycleError("durable launch intent is missing")
            if record.get("deadline_utc") != self.config["deadline_utc"]:
                raise control.LifecycleError("guest/config/durable launch deadline mismatch")
            absolute = datetime.fromisoformat(record["deadline_utc"].replace("Z", "+00:00"))
            self.launch_deadline_monotonic = self.monotonic() + max(0, (absolute - self.now()).total_seconds())
            self._health({**payload, "started_at_utc": record["created_utc"], "deadline_utc": record["deadline_utc"]})
            transport = {key: value for key, value in expected.items() if key != "UserData"}
            raw_script = expected["UserData"].encode()
            compressed = gzip.compress(raw_script, mtime=0)
            if len(compressed) > 16384 or record.get("user_data_encoding") != "gzip":
                raise control.LifecycleError("persisted gzip user-data contract/size mismatch")
            # EC2's AWS CLI handler base64-encodes UserData unconditionally,
            # including strings supplied through --cli-input-json. Pass raw
            # gzip bytes through fileb:// so that handler performs the one and
            # only base64 encoding. Keep the private exact payload for audit.
            script_path = self.state / f"user-data-{self.token}.gz"
            fd = os.open(script_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(compressed); stream.flush(); os.fsync(stream.fileno())
            self._record_new("user-data.json", {
                "transport": "aws-cli-fileb-gzip", "gzip_bytes": len(compressed),
                "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                "raw_script_sha256": hashlib.sha256(raw_script).hexdigest(),
                "client_token": self.token,
            })
            response = self._aws("ec2", "run-instances", transport,
                                 extra=("--user-data", f"fileb://{script_path.resolve()}"))
            instances = response.get("Instances", [])
            if len(instances) != 1:
                raise control.LifecycleError("RunInstances did not return exactly one instance")
            return self._describe(instances[0]["InstanceId"])
        if action == "describe_instance":
            return self._describe(payload["instance_id"])
        if action == "terminate_instance":
            self._describe(payload["instance_id"])
            return self._aws("ec2", "terminate-instances", {"InstanceIds": [payload["instance_id"]]})
        if action == "authorize_work":
            self._describe(payload["instance_id"])
            key = self.config["run_prefix"] + "/control/start.json"
            self._missing("s3api", "head-object", {"Bucket": self.config["bucket"], "Key": key,
                          "ExpectedBucketOwner": self.config["account"]}, {"NoSuchKey", "NotFound", "404"})
            permit = {**payload, "authorized": True}
            with tempfile.TemporaryDirectory(prefix="tiki-permit-", dir=self.state) as directory:
                path = Path(directory) / "permit.json"; path.write_bytes(control.encoded(permit))
                self._aws("s3api", "put-object", {"Bucket": self.config["bucket"], "Key": key,
                          "ExpectedBucketOwner": self.config["account"], "ContentType": "application/json",
                          "ServerSideEncryption": "AES256", "IfNoneMatch": "*"}, extra=("--body", path))
            observed = strict_json(self._object(key))
            if observed != permit:
                raise ValueError("S3 guest permit readback mismatch")
            return observed
        if action == "poll_result":
            record = self._completion(payload)
            if record is None:
                return {"status": "running"}
            return {"status": "success" if record.get("status") == "SUCCESS" and record.get("exit_code") == 0
                    and record.get("durable_artifacts_verified") is True else "failed"}
        return self._retrieve(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="status", choices=("status", "run", "recover", "cancel-unlaunched"))
    parser.add_argument("--config", type=Path); parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps({"offline": True, "explicit_execute_required": True, "automatic_launch": False}))
        return 0
    if not args.execute or args.config is None or args.state is None or (args.command == "run" and args.output is None):
        parser.error("explicit --execute, --config, --state and (for run) new --output are required")
    config = strict_json(args.config.read_bytes())
    adapter = Adapter(config, args.state, args.output)
    if args.command == "run":
        if args.output.exists():
            parser.error("--output must name a fresh directory")
        result = control.supervise(config, args.state, adapter)
    elif args.command == "recover":
        result = {"termination_confirmed": control.recover(args.state, config["account"], config["run_prefix"], adapter)}
    else:
        result = control.cancel_before_launch(args.state, config["account"], config["run_prefix"], adapter)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
