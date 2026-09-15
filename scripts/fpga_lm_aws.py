#!/usr/bin/env python3
"""Offline, fail-closed preparation for the first FPGA language-model pilot.

This is NOT an operational cloud launcher or an AWS billing cap. ``plan`` and
``render`` read explicit local inputs only; ``bundle`` writes one requested new
archive. ``preflight`` makes read-only AWS queries with the named profile.
``launch`` is deliberately disabled until ownership-checked supervision and
termination have been integration-tested. Never use a legacy launcher to bypass
this gate. Prices, resources, dataset and IAM permission decisions stay explicit.

Example configuration: ``python scripts/fpga_lm_aws.py example``.
"""

import argparse
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROJECT_TAG = "TikiTorchFPGALM"
CEILING = Decimal("1000")
CONTINGENCY = Decimal("100")
PILOT_RESERVATION = Decimal("10.00")
LAUNCH_DISABLED = (
    "launch is disabled: an ownership-checked external supervisor, durable "
    "budget reservation, and confirmed termination/retrieval still require "
    "integration validation; plan/render/preflight are preparation, not a billing cap"
)
SOURCE_FILES = (
    "LICENSE", "THIRD_PARTY_NOTICES.md", "scripts/fpga_lm_pretrain.py",
    "scripts/fpga_lm_layout.cpp", "tikitorch/fpga/bridge.cpp",
    "tikitorch/fpga/silicon/gen_buffer_init.py",
    "tikitorch/fpga/silicon/morph_lm_silicon.py",
    "tikitorch/fpga/silicon/cstdint_shim/cstdint",
    "tikitorch/hardware/hls/forge_hls_types.hpp",
    "tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp",
    "tikitorch/hardware/hls/forge_kernels/forge_ce_lut.hpp",
    "tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp",
)


def money(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return amount


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def matched(config, name, pattern):
    value = config.get(name)
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"invalid or missing {name}")
    return value


def validate_config(config, *, today=None):
    """Reject ambiguous identities, shell inputs, unsupported shapes and stale prices."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    matched(config, "profile", r"[A-Za-z0-9_.@-]{1,128}")
    account = matched(config, "account", r"[0-9]{12}")
    matched(config, "region", r"us-(?:east-1|west-2)")
    matched(config, "ami", r"ami-[0-9a-f]{8,17}")
    matched(config, "ami_owner", r"[0-9]{12}")
    matched(config, "agfi", r"agfi-[0-9a-f]{17}")
    matched(config, "bucket", r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
    prefix = matched(config, "run_prefix", r"tikitorch-fpga-lm/[a-z0-9][a-z0-9-]{7,79}")
    matched(config, "subnet", r"subnet-[0-9a-f]{8,17}")
    matched(config, "security_group", r"sg-[0-9a-f]{8,17}")
    matched(config, "instance_profile", rf"arn:aws:iam::{account}:instance-profile/[A-Za-z0-9_+=,.@/-]+")
    matched(config, "sdk_root", r"/[A-Za-z0-9_./-]+")
    if ".." in Path(config["sdk_root"]).parts:
        raise ValueError("sdk_root must not contain parent traversal")
    matched(config, "sdk_revision", r"[0-9a-f]{40}")
    matched(config, "source_sha256", r"[0-9a-f]{64}")
    matched(config, "corpus_sha256", r"[0-9a-f]{64}")
    key = matched(config, "corpus_key", r"[A-Za-z0-9_./-]+")
    if ".." in key.split("/") or key.startswith("/") or key.startswith(prefix + "/"):
        raise ValueError("corpus_key must be a separate existing object, without traversal")
    if not isinstance(config.get("corpus_license"), str) or not config["corpus_license"].strip():
        raise ValueError("corpus_license must explicitly identify the permitted dataset use")
    if config.get("instance_type") != "f2.6xlarge":
        raise ValueError("only one on-demand f2.6xlarge is supported")
    minutes = integer(config.get("max_minutes"), "max_minutes", 1, 90)
    integer(config.get("steps"), "steps", 1, 256)
    integer(config.get("eval_every"), "eval_every", 1, 256)
    integer(config.get("eval_windows"), "eval_windows", 1, 8)
    integer(config.get("root_volume_gib"), "root_volume_gib", 30, 150)
    integer(config.get("secondary_volume_gib", 0), "secondary_volume_gib", 0, 20)
    if type(config.get("public_ipv4", False)) is not bool:
        raise ValueError("public_ipv4 must be an explicit boolean")
    if config.get("public_ipv4"):
        reason = config.get("network_exception_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("public IPv4 requires an explicit no-ingress network justification")
    rate = money(config.get("hourly_usd"), "hourly_usd", positive=True)
    allowance = money(config.get("storage_ip_artifact_allowance_usd"), "storage_ip_artifact_allowance_usd")
    if allowance < Decimal("5"):
        raise ValueError("reserve at least $5 for EBS, public IP, S3 and transfer uncertainty")
    price_url = config.get("price_source_url", "")
    if not isinstance(price_url, str) or re.fullmatch(
        r"https://(?:pricing\.us-east-1\.amazonaws\.com|aws\.amazon\.com|b0\.p\.awsstatic\.com)/[^\s]+", price_url
    ) is None:
        raise ValueError("price_source_url must identify an official AWS price source")
    try:
        checked = date.fromisoformat(config["price_checked_on"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("price_checked_on must be an ISO date") from exc
    age = ((today or datetime.now(timezone.utc).date()) - checked).days
    if not 0 <= age <= 7:
        raise ValueError("explicit price verification must be no more than seven days old")
    estimated = (rate * Decimal(minutes) / 60 + allowance).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    if estimated > PILOT_RESERVATION:
        raise ValueError("first pilot reservation must not exceed $10")
    # Reserve the full reviewed envelope, not merely today's lower estimate.
    return PILOT_RESERVATION


def validate_unlaunched_record(record, *, require_absence=False):
    """Validate durable prelaunch history, never infer absence after a launch."""
    allowed = {"launch_intent", "watchdog_verified", "controller_failed"}
    if require_absence:
        allowed.add("cancelled_before_launch")
    events = record.get("events")
    forbidden = {"instance_id", "launch_error_type", "guest_authorized", "retrieval",
                 "termination_confirmed", "termination_verified_at", "termination_error_type"}
    if (not isinstance(events, list) or not events or not isinstance(events[0], dict)
            or events[0].get("status") != "launch_intent"
            or record.get("status") not in allowed
            or any(not isinstance(event, dict) or event.get("status") not in allowed for event in events)
            or events[-1].get("status") != record["status"]
            or any(forbidden & set(value) for value in [record, *events])):
        raise ValueError("cancellation requires unambiguous prelaunch-only durable history and no instance")
    matched(record, "account", r"[0-9]{12}")
    matched(record, "region", r"us-(?:east-1|west-2)")
    matched(record, "run_prefix", r"tikitorch-fpga-lm/[a-z0-9][a-z0-9-]{7,79}")
    expected_token = hashlib.sha256(f"{record['account']}:{record['region']}:{record['run_prefix']}".encode()).hexdigest()
    if record.get("client_token") != expected_token:
        raise ValueError("cancelled run client token does not match its exact identity")
    if money(record.get("reserved_usd"), "reserved_usd", positive=True) != PILOT_RESERVATION:
        raise ValueError("cancelled unlaunched run must retain its full $10 reservation")
    if require_absence:
        expected = {key: record[key] for key in ("account", "region", "run_prefix", "client_token")}
        expected["matches"] = []
        stamp = record.get("absence_verified_at")
        try:
            moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise ValueError("cancelled run lacks a valid absence-verification timestamp") from error
        if (moment.tzinfo is None or moment.utcoffset().total_seconds() != 0
                or record.get("status") != "cancelled_before_launch"
                or record.get("absence_evidence") != expected
                or events[-1].get("absence_evidence") != expected
                or events[-1].get("absence_verified_at") != stamp
                or sum(event["status"] == "cancelled_before_launch" for event in events) != 1):
            raise ValueError("cancelled run lacks exact authenticated absence evidence")


def validate_ledger(ledger, config, reservation):
    """Conservatively count prior reservations; never infer money was refunded.

    This only validates a proposed ledger. It neither reserves AWS funds nor
    releases earlier reservations. An unreconciled run blocks subsequent work.
    """
    if not isinstance(ledger, dict) or ledger.get("account") != config["account"]:
        raise ValueError("ledger must be bound to the explicit account")
    if ledger.get("currency") != "USD" or money(ledger.get("ceiling_usd"), "ceiling_usd") != CEILING:
        raise ValueError("ledger must declare a $1000 USD cumulative ceiling")
    records = ledger.get("runs")
    if not isinstance(records, list):
        raise ValueError("ledger runs must be an explicit list")
    seen, spent = set(), Decimal("0")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid ledger record")
        run_id = record.get("run_prefix")
        if not isinstance(run_id, str) or run_id in seen:
            raise ValueError("ledger run prefixes must be unique")
        seen.add(run_id)
        if record.get("status") not in {"terminated_reconciled", "cancelled_before_launch"}:
            raise ValueError("an active or unreconciled run blocks further work")
        matched(record, "region", r"us-(east|west)-[12]")
        if record["status"] == "cancelled_before_launch":
            if record.get("account") != config["account"]:
                raise ValueError("cancelled run account differs from the bound ledger")
            validate_unlaunched_record(record, require_absence=True)
        else:
            matched(record, "instance_id", r"i-[0-9a-f]{8,17}")
            if not record.get("termination_verified_at"):
                raise ValueError("ledger needs explicit confirmed-termination evidence")
        reserved = money(record.get("reserved_usd"), "reserved_usd", positive=True)
        billed = money(record.get("billed_usd"), "billed_usd")
        spent += max(reserved, billed)
    if config["run_prefix"] in seen:
        raise ValueError("run_prefix has already been used; never overwrite prior state")
    if spent + reservation > CEILING - CONTINGENCY:
        raise ValueError("reservation would cross $900, excluding the $100 contingency")
    return spent


def plan(config, ledger, *, today=None):
    reservation = validate_config(config, today=today)
    previous = validate_ledger(ledger, config, reservation)
    run_id = config["run_prefix"].split("/")[-1]
    tags = [{"Key": "Project", "Value": PROJECT_TAG},
            {"Key": "Run", "Value": run_id},
            {"Key": "OwnerAccount", "Value": config["account"]}]
    estimate = (Decimal(str(config["hourly_usd"])) * Decimal(config["max_minutes"]) / 60
                + Decimal(str(config["storage_ip_artifact_allowance_usd"]))).quantize(
                    Decimal("0.01"), rounding=ROUND_CEILING)
    proposed = {
        "schema_version": 1, "launch_enabled": False,
        "scope": "offline proposal only; no AWS spending cap or live reservation",
        "reservation_usd": str(reservation), "previous_accounted_usd": str(previous),
        "estimated_compute_and_allowance_usd": str(estimate),
        "proposed_accounted_usd": str(previous + reservation),
        "contingency_usd": str(CONTINGENCY), "ceiling_usd": str(CEILING),
        "account": config["account"], "region": config["region"],
        "run_prefix": config["run_prefix"], "source_files": list(SOURCE_FILES),
        "source_s3_uri": f"s3://{config['bucket']}/{config['run_prefix']}/source.tar.gz",
        "results_s3_uri": f"s3://{config['bucket']}/{config['run_prefix']}/results",
        "proposed_run_instances": {
            "ImageId": config["ami"], "InstanceType": "f2.6xlarge",
            "MinCount": 1, "MaxCount": 1,
            "DisableApiTermination": False,
            "ClientToken": hashlib.sha256(
                f"{config['account']}:{config['region']}:{config['run_prefix']}".encode()
            ).hexdigest(),
            "IamInstanceProfile": {"Arn": config["instance_profile"]},
            "InstanceInitiatedShutdownBehavior": "terminate",
            "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled",
                                "HttpPutResponseHopLimit": 1},
            "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": config["subnet"],
                                   "Groups": [config["security_group"]],
                                   "AssociatePublicIpAddress": config.get("public_ipv4", False),
                                   "DeleteOnTermination": True}],
            "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {
                "VolumeSize": config["root_volume_gib"], "VolumeType": "gp3",
                "Encrypted": True, "DeleteOnTermination": True}}],
            "TagSpecifications": [{"ResourceType": kind, "Tags": tags}
                                  for kind in ("instance", "volume")],
        },
        "remaining_launch_gates": [
            "Validated external watchdog that terminates only this account/region/run-tagged instance",
            "Atomic durable reservation and launch reconciliation before any EC2 write",
            "Verified AMI root snapshot minimum and all block mappings fit the requested encrypted root volume",
            "Verified private subnet routes/DNS/endpoints permit S3 and required runtime dependencies; not checked by preflight (no NAT or package-install mutation authorized)",
            "Verified instance-profile permissions and pinned AMI SDK compatibility",
            "Paired identical fresh-window CSIM and PCI recipes with full weight-checkpoint equality at every checkpoint; the old 30-step CRC/prediction KAT does not validate streaming updates",
            "Fresh artifact retrieval and confirmed termination before reservation reconciliation",
        ],
    }
    if config.get("secondary_volume_gib", 0):
        proposed["proposed_run_instances"]["BlockDeviceMappings"].append({
            "DeviceName": "/dev/sdm", "Ebs": {
                "VolumeSize": config["secondary_volume_gib"], "VolumeType": "gp3",
                "Encrypted": True, "DeleteOnTermination": True}})
    return proposed


def build_bundle(output, root=ROOT):
    """Create a new deterministic allowlisted archive, including current untracked sources."""
    root = Path(root).resolve()
    payloads = {}
    for name in SOURCE_FILES:
        path = root / name
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"required source is missing or symlinked: {name}")
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f"source member exceeds the guest's 16 MiB limit: {name}")
        payloads[name] = path.read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())}
    payloads["SOURCE_MANIFEST.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    output = Path(output)
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted(payloads.items()):
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
                    archive.addfile(entry, io.BytesIO(data))
    return {"archive": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "files": manifest, "bytes": output.stat().st_size}


def render(config, ledger, *, today=None):
    """Render a reviewable guest script; no EC2 controller is provided."""
    proposed = plan(config, ledger, today=today)
    q = shlex.quote
    assignments = {
        "AWS_DEFAULT_REGION": config["region"], "S3_SOURCE": proposed["source_s3_uri"],
        "S3_RESULTS": proposed["results_s3_uri"], "SOURCE_SHA256": config["source_sha256"],
        "S3_CORPUS": f"s3://{config['bucket']}/{config['corpus_key']}",
        "CORPUS_SHA256": config["corpus_sha256"], "AGFI": config["agfi"],
        "SDK_ROOT": config["sdk_root"], "SDK_REVISION": config["sdk_revision"],
    }
    header = "\n".join(f"export {key}={q(value)}" for key, value in assignments.items())
    trainer = shlex.join([
        "python3", "scripts/fpga_lm_pretrain.py", "--backend", "pci",
        "--library", "/tmp/tiki-fpga-lm/libpci.so", "--image-id", config["agfi"],
        "--width", "512", "--corpus", "/tmp/tiki-fpga-lm/corpus.txt",
        "--output", "/tmp/tiki-fpga-lm/training",
        "--steps", str(config["steps"]), "--eval-every", str(config["eval_every"]),
        "--eval-windows", str(config["eval_windows"]), "--lr", "1",
        "--threshold", "2048", "--seed", "0",
    ])
    body = r'''
set -euo pipefail
umask 077
WORK=/tmp/tiki-fpga-lm
mkdir "$WORK"
exec > "$WORK/guest.log" 2>&1
HEARTBEAT_PID=
RESULT=FAILED
upload(){
  timeout 20 aws s3 cp "$WORK/guest.log" "$S3_RESULTS/guest.log" --only-show-errors || true
  if [ -d "$WORK/training" ]; then
    timeout 20 aws s3 cp "$WORK/training/" "$S3_RESULTS/training/" --recursive --only-show-errors || true
  fi
}
finish(){
  rc=$?
  trap - EXIT
  set +e
  [ -z "$HEARTBEAT_PID" ] || kill "$HEARTBEAT_PID" 2>/dev/null
  upload
  printf '{"status":"%s","exit_code":%d}\n' "$RESULT" "$rc" > "$WORK/completion.json"
  timeout 20 aws s3 cp "$WORK/completion.json" "$S3_RESULTS/completion.json" --only-show-errors
  shutdown -h now
  exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM INT HUP
command -v timeout >/dev/null
command -v aws >/dev/null
# This must succeed before any download/build. The missing external watchdog is
# why the local CLI refuses to launch even this guest script today.
shutdown -h +MAX_MINUTES "TikiTorch FPGA LM pilot deadline"
( while sleep 30; do upload; done ) & HEARTBEAT_PID=$!
test "$(git -C "$SDK_ROOT" rev-parse HEAD)" = "$SDK_REVISION"
test -f "$SDK_ROOT/sdk_setup.sh"
# The pinned SDK setup historically assumes unset variables; restore strict mode immediately.
set +u
source "$SDK_ROOT/sdk_setup.sh"
set -u
command -v fpga-load-local-image >/dev/null
command -v fpga-describe-local-image >/dev/null
python3 -c 'import numpy'
aws s3 cp "$S3_SOURCE" "$WORK/source.tar.gz" --only-show-errors
printf '%s  %s\n' "$SOURCE_SHA256" "$WORK/source.tar.gz" | sha256sum -c -
mkdir "$WORK/source"
python3 - "$WORK/source.tar.gz" "$WORK/source" <<'PY'
import hashlib, json, pathlib, sys, tarfile
archive_path, destination = sys.argv[1:]
expected = EXPECTED_SOURCE_FILES
with tarfile.open(archive_path) as archive:
    members = archive.getmembers()
    names = [m.name for m in members]
    if len(set(names)) != len(names) or set(names) != set(expected) | {"SOURCE_MANIFEST.json"}:
        raise SystemExit("source archive allowlist mismatch")
    if any(not m.isfile() or m.size > 16 * 1024 * 1024 for m in members):
        raise SystemExit("invalid source archive member")
    manifest = json.load(archive.extractfile("SOURCE_MANIFEST.json"))
    if set(manifest) != set(expected):
        raise SystemExit("source manifest allowlist mismatch")
    for name in expected:
        data = archive.extractfile(name).read()
        if hashlib.sha256(data).hexdigest() != manifest[name]:
            raise SystemExit("source member hash mismatch")
        path = pathlib.Path(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
PY
aws s3 cp "$S3_CORPUS" "$WORK/corpus.txt" --only-show-errors
printf '%s  %s\n' "$CORPUS_SHA256" "$WORK/corpus.txt" | sha256sum -c -
cd "$WORK/source"
COMMON=(-O2 -std=c++17 -fPIC -shared -DML_D=512 -DTT_RESIDENT_W -DTT_ONCHIP_GRAD -DTT_URAM_W
        -Itikitorch/hardware/hls/forge_kernels -Itikitorch/fpga/silicon/cstdint_shim)
g++ "${COMMON[@]}" -DCSIM_ONLY -DTT_KERNEL_MORPH_LM tikitorch/fpga/bridge.cpp \
    tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp -o "$WORK/libcsim.so"
g++ "${COMMON[@]}" -DTT_FPGA_PCI -I"$SDK_ROOT/sdk/userspace/include" \
    tikitorch/fpga/bridge.cpp scripts/fpga_lm_layout.cpp -lfpga_mgmt -o "$WORK/libpci.so"
export ML_D=512 ML_BATCH=1 TT_FUSED_STEPS=30 TT_CLKGEN_BRINGUP=1 TT_KICK_SPIN=2000000000
python3 tikitorch/fpga/silicon/gen_buffer_init.py "$WORK/buffer_init.bin"
TIKITORCH_FPGA_LIB="$WORK/libcsim.so" python3 tikitorch/fpga/silicon/morph_lm_silicon.py \
    "$WORK/buffer_init.bin" > "$WORK/golden.jsonl" 2> "$WORK/golden.stderr"
fpga-load-local-image -S 0 -I "$AGFI" -F
fpga-describe-local-image -S 0 > "$WORK/image.txt"
TIKITORCH_FPGA_LIB="$WORK/libpci.so" python3 tikitorch/fpga/silicon/morph_lm_silicon.py \
    "$WORK/buffer_init.bin" > "$WORK/silicon.jsonl" 2> "$WORK/silicon.stderr"
for file in golden.jsonl golden.stderr silicon.jsonl silicon.stderr image.txt; do
  aws s3 cp "$WORK/$file" "$S3_RESULTS/$file" --only-show-errors
done
test -s "$WORK/golden.jsonl"
cmp "$WORK/golden.jsonl" "$WORK/silicon.jsonl"
# This observable KAT checks CRC/predictions, not every signal or final weight.
# A paired fresh-window CSIM/PCI run with full weight-checkpoint equality is a
# separate unresolved launch gate, not established by this rendered pilot.
TRAINER_COMMAND
# Required artifacts must upload successfully before SUCCESS, including the terminal record.
test -s "$WORK/training/summary.json"
aws s3 cp "$WORK/training/" "$S3_RESULTS/training/" --recursive --only-show-errors
RESULT=SUCCESS
'''
    body = body.replace("MAX_MINUTES", str(config["max_minutes"]))
    body = body.replace("EXPECTED_SOURCE_FILES", repr(list(SOURCE_FILES)))
    body = body.replace("TRAINER_COMMAND", trainer)
    return "#!/bin/bash\n# REVIEW ONLY: the CLI does not launch instances.\n" + header + body


def aws_json(config, *arguments):
    command = ["aws", "--profile", config["profile"], "--region", config["region"],
               "--no-cli-pager", "--output", "json", *arguments]
    result = subprocess.run(command, text=True, capture_output=True, timeout=40, check=False)
    if result.returncode:
        # Do not echo arbitrary CLI output that may contain credential-provider details.
        raise ValueError(f"read-only AWS preflight failed at {' '.join(arguments[:2])}; check profile authentication/permissions")
    return json.loads(result.stdout) if result.stdout.strip() else {}


MAX_STAGED_BYTES = 128 * 1024 * 1024


def staged_input_hashes(config):
    """Only the three approved immutable pilot inputs may occupy a resumed prefix."""
    prefix = matched(config, "run_prefix", r"tikitorch-fpga-lm/[a-z0-9][a-z0-9-]{7,79}")
    if (config.get("sdk_s3_key") != prefix + "/sdk.tar"
            or config.get("numpy_s3_key") != prefix + "/numpy-wheels.tar"):
        raise ValueError("staged input keys must match this exact run prefix")
    return {prefix + "/" + name: matched(config, field, r"[0-9a-f]{64}") for name, field in (
        ("source.tar.gz", "source_sha256"), ("sdk.tar", "sdk_sha256"),
        ("numpy-wheels.tar", "numpy_archive_sha256"))}


def aws_object_bytes(config, key, expected_bytes):
    """Read one bounded S3 byte range; verify the total object size, not an ETag."""
    if type(expected_bytes) is not int or not 0 < expected_bytes <= MAX_STAGED_BYTES:
        raise ValueError("staged object exceeds the 128 MiB byte bound")
    with tempfile.TemporaryDirectory(prefix="tikitorch-staged-read-") as directory:
        destination = Path(directory) / "object"
        result = aws_json(config, "s3api", "get-object", "--bucket", config["bucket"],
                          "--key", key, "--expected-bucket-owner", config["account"],
                          "--range", f"bytes=0-{expected_bytes - 1}", str(destination))
        if (result.get("ContentRange") != f"bytes 0-{expected_bytes - 1}/{expected_bytes}"
                or type(result.get("ContentLength")) is not int or result["ContentLength"] != expected_bytes):
            raise ValueError("staged object changed size or ranged readback was incomplete")
        if destination.stat().st_size != expected_bytes:
            raise ValueError("staged byte readback length mismatch")
        return destination.read_bytes()


def verify_staged_prefix(config, query, read_object):
    expected = staged_input_hashes(config)
    listing = query(config, "s3api", "list-objects-v2", "--bucket", config["bucket"],
                    "--prefix", config["run_prefix"] + "/", "--max-keys", "4", "--no-paginate",
                    "--expected-bucket-owner", config["account"])
    if (not isinstance(listing, dict) or listing.get("IsTruncated") is not False
            or "NextContinuationToken" in listing or "NextToken" in listing
            or listing.get("CommonPrefixes") not in (None, [])
            or type(listing.get("KeyCount")) is not int or listing["KeyCount"] != 3
            or not isinstance(listing.get("Contents"), list) or len(listing["Contents"]) != 3):
        raise ValueError("staged prefix needs one complete explicit three-object listing")
    entries = listing["Contents"]
    if (any(not isinstance(item, dict) or not isinstance(item.get("Key"), str) for item in entries)
            or len({item["Key"] for item in entries}) != 3
            or {item["Key"] for item in entries} != set(expected)):
        raise ValueError("staged prefix contains missing, duplicate or unexpected keys")
    verified = {}
    for item in entries:
        key, size = item["Key"], item.get("Size")
        if type(size) is not int or not 0 < size <= MAX_STAGED_BYTES:
            raise ValueError("staged input listing size exceeds its 128 MiB bound")
        head = query(config, "s3api", "head-object", "--bucket", config["bucket"],
                     "--key", key, "--expected-bucket-owner", config["account"])
        if not isinstance(head, dict) or type(head.get("ContentLength")) is not int or head["ContentLength"] != size:
            raise ValueError("staged object listing/head size mismatch")
        payload = read_object(config, key, size)
        if not isinstance(payload, bytes) or len(payload) != size or hashlib.sha256(payload).hexdigest() != expected[key]:
            raise ValueError("staged object byte readback SHA256/size mismatch")
        verified[key] = {"bytes": size, "sha256": expected[key]}
    return verified


def preflight(config, ledger, query=aws_json):
    """Original empty-prefix preflight; staged resumes use a separate explicit path."""
    return _preflight(config, ledger, query)


def preflight_staged(config, ledger, query=aws_json, read_object=aws_object_bytes):
    """Read-only resource/input checks for an unreserved staged run, never a launch gate override.

    The caller must separately prove exact-token instance/watchdog absence and
    regenerate fresh timing into new files. Existing ledger identities remain
    forbidden; historical empty-prefix evidence is not replaced or fabricated.
    """
    staged_input_hashes(config)
    return _preflight(config, ledger, query, staged_reader=read_object)


def _preflight(config, ledger, query, *, staged_reader=None):
    """Explicit read-only resource checks, not a guarantee of launch permission/network reachability."""
    proposed = plan(config, ledger)
    identity = query(config, "sts", "get-caller-identity")
    if identity.get("Account") != config["account"]:
        raise ValueError("authenticated AWS account does not match explicit account")
    images = query(config, "ec2", "describe-images", "--image-ids", config["ami"]).get("Images", [])
    if len(images) != 1 or images[0].get("OwnerId") != config["ami_owner"] or images[0].get("State") != "available":
        raise ValueError("AMI identity, owner or availability mismatch")
    if images[0].get("Architecture") != "x86_64" or images[0].get("RootDeviceName") != "/dev/sda1":
        raise ValueError("AMI architecture/root-device contract differs from reviewed plan")
    root_mappings = [mapping for mapping in images[0].get("BlockDeviceMappings", [])
                     if mapping.get("DeviceName") == "/dev/sda1"]
    if len(root_mappings) > 1:
        raise ValueError("ambiguous AMI root block-device mappings")
    declared_root_size = root_mappings[0].get("Ebs", {}).get("VolumeSize") if root_mappings else None
    if declared_root_size is not None:
        if type(declared_root_size) is not int or declared_root_size < 1:
            raise ValueError("invalid AMI root volume size")
        if declared_root_size > config["root_volume_gib"]:
            raise ValueError("requested root volume is smaller than the AMI root mapping")
    planned_volumes = {entry["DeviceName"]: entry["Ebs"] for entry in
                       proposed["proposed_run_instances"]["BlockDeviceMappings"]}
    for mapping in images[0].get("BlockDeviceMappings", []):
        if "Ebs" not in mapping:
            continue
        volume = planned_volumes.get(mapping.get("DeviceName"))
        declared = mapping["Ebs"].get("VolumeSize")
        if volume is None or type(declared) is not int or declared < 1:
            raise ValueError("every AMI EBS mapping needs an explicit validated encrypted disposable volume")
        if volume["VolumeSize"] < declared:
            raise ValueError("requested volume is smaller than the AMI block mapping")
    afis = query(config, "ec2", "describe-fpga-images", "--filters",
                 f"Name=fpga-image-global-id,Values={config['agfi']}").get("FpgaImages", [])
    if len(afis) != 1 or afis[0].get("OwnerId") != config["account"] or afis[0].get("State", {}).get("Code") != "available":
        raise ValueError("AGFI must be available and owned by the explicit account")
    subnets = query(config, "ec2", "describe-subnets", "--subnet-ids", config["subnet"]).get("Subnets", [])
    groups = query(config, "ec2", "describe-security-groups", "--group-ids", config["security_group"]).get("SecurityGroups", [])
    if len(subnets) != 1 or len(groups) != 1:
        raise ValueError("subnet/security-group identity mismatch")
    subnet, group = subnets[0], groups[0]
    if subnet.get("OwnerId") != config["account"] or group.get("OwnerId") != config["account"] or subnet.get("VpcId") != group.get("VpcId"):
        raise ValueError("subnet/security group must share this account and VPC")
    if subnet.get("State") != "available" or group.get("IpPermissions") != []:
        raise ValueError("an available subnet and security group with no inbound rules are required")
    profile_name = config["instance_profile"].rsplit("/", 1)[-1]
    profile = query(config, "iam", "get-instance-profile", "--instance-profile-name", profile_name).get("InstanceProfile", {})
    if profile.get("Arn") != config["instance_profile"] or len(profile.get("Roles", [])) != 1:
        raise ValueError("instance-profile identity/role mismatch")
    query(config, "s3api", "head-bucket", "--bucket", config["bucket"],
          "--expected-bucket-owner", config["account"])
    location = query(config, "s3api", "get-bucket-location", "--bucket", config["bucket"],
                     "--expected-bucket-owner", config["account"]).get("LocationConstraint") or "us-east-1"
    if location != config["region"]:
        raise ValueError("artifact bucket must be in the explicitly selected region")
    public = query(config, "s3api", "get-public-access-block", "--bucket", config["bucket"],
                   "--expected-bucket-owner", config["account"]).get("PublicAccessBlockConfiguration", {})
    if any(public.get(key) is not True for key in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")):
        raise ValueError("artifact bucket must have all public-access blocks enabled")
    verified_inputs = None
    if staged_reader is None:
        objects = query(config, "s3api", "list-objects-v2", "--bucket", config["bucket"],
                        "--prefix", config["run_prefix"] + "/", "--max-keys", "1",
                        "--expected-bucket-owner", config["account"])
        if objects.get("KeyCount", 0) or objects.get("Contents"):
            raise ValueError("S3 run prefix already contains artifacts; choose a new run identity")
    else:
        verified_inputs = verify_staged_prefix(config, query, staged_reader)
    corpus = query(config, "s3api", "head-object", "--bucket", config["bucket"],
                   "--key", config["corpus_key"], "--expected-bucket-owner", config["account"])
    if not 1024 <= corpus.get("ContentLength", 0) <= 128 * 1024 * 1024:
        raise ValueError("first-pilot corpus must be between 1 KiB and 128 MiB")
    active = query(config, "ec2", "describe-instances", "--filters",
                   f"Name=tag:Project,Values={PROJECT_TAG}",
                   "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down")
    if any(reservation.get("Instances") for reservation in active.get("Reservations", [])):
        raise ValueError("existing project instances need reconciliation before another run")
    result = {"status": "read_only_preflight_passed", "launch_enabled": False,
            "account": config["account"], "region": config["region"],
            "reservation_usd": proposed["reservation_usd"],
            "ami_declared_root_volume_gib": declared_root_size,
            "ami_root_snapshot_minimum_verified": False,
            "private_subnet_runtime_reachability_verified": False,
            "paired_fresh_window_full_checkpoint_equality_verified": False,
            "remaining_launch_gates": proposed["remaining_launch_gates"]}
    if verified_inputs is not None:
        result.update(status="read_only_staged_preflight_passed", s3_prefix_state="approved_staged_inputs",
                      verified_staged_inputs=verified_inputs,
                      exact_token_instance_and_watchdog_absence_verified=False)
    return result


def example():
    """Deliberately invalid placeholders prevent this example being an accidental launch recipe."""
    return {
        "config": {
            "profile": "YOUR_PROFILE", "account": "YOUR_12_DIGIT_ACCOUNT",
            "region": "us-east-1", "ami": "VERIFIED_FPGA_AMI", "ami_owner": "VERIFIED_AMI_OWNER",
            "agfi": "VERIFIED_ACCOUNT_OWNED_AGFI", "bucket": "YOUR_PRIVATE_BUCKET",
            "run_prefix": "tikitorch-fpga-lm/pilot-20260909-unique",
            "subnet": "EXISTING_SUBNET", "security_group": "EXISTING_NO_INGRESS_GROUP",
            "instance_profile": "EXPLICIT_INSTANCE_PROFILE_ARN",
            "sdk_root": "/opt/aws-fpga", "sdk_revision": "PINNED_40_CHARACTER_COMMIT",
            "instance_type": "f2.6xlarge", "max_minutes": 90, "root_volume_gib": 50,
            "hourly_usd": "1.98", "storage_ip_artifact_allowance_usd": "7.03",
            "price_checked_on": "2026-09-09",
            "price_source_url": "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json",
            "source_sha256": "SHA256_FROM_BUNDLE", "corpus_key": "datasets/approved-corpus.txt",
            "corpus_sha256": "VERIFIED_CORPUS_SHA256", "corpus_license": "EXPLICIT_LICENSE_AND_PROVENANCE",
            "steps": 64, "eval_every": 32, "eval_windows": 2,
        },
        "ledger": {"account": "YOUR_12_DIGIT_ACCOUNT", "currency": "USD",
                   "ceiling_usd": "1000", "runs": []},
        "warning": LAUNCH_DISABLED,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="example",
                        choices=("example", "plan", "render", "bundle", "preflight", "preflight-staged", "launch"))
    parser.add_argument("--config", type=Path, help="explicit JSON config, never inferred from credentials")
    parser.add_argument("--ledger", type=Path, help="read-only cumulative reservation ledger JSON")
    parser.add_argument("--output", type=Path, help="new local source archive, bundle only")
    args = parser.parse_args(argv)
    try:
        if args.command == "launch":
            raise ValueError(LAUNCH_DISABLED)
        if args.command == "example":
            result = example()
        elif args.command == "bundle":
            if args.output is None:
                raise ValueError("bundle requires --output naming a new file")
            result = build_bundle(args.output)
        else:
            if args.config is None or args.ledger is None:
                raise ValueError("explicit --config and --ledger are required")
            config = json.loads(args.config.read_text())
            ledger = json.loads(args.ledger.read_text())
            if args.command == "render":
                print(render(config, ledger), end="")
                return 0
            if args.command == "preflight":
                result = preflight(config, ledger)
            elif args.command == "preflight-staged":
                result = preflight_staged(config, ledger)
            else:
                result = plan(config, ledger)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
