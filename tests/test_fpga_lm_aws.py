"""Offline cloud preparation contracts. No test contacts AWS or starts a workload."""

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import fpga_lm_aws as A


@pytest.fixture
def config():
    return {
        "profile": "research", "account": "123456789012", "region": "us-east-1",
        "ami": "ami-0123456789abcdef0", "ami_owner": "210987654321",
        "agfi": "agfi-0123456789abcdef0", "bucket": "tiki-private-research",
        "run_prefix": "tikitorch-fpga-lm/pilot-20260909-test",
        "subnet": "subnet-0123456789abcdef0", "security_group": "sg-0123456789abcdef0",
        "instance_profile": "arn:aws:iam::123456789012:instance-profile/tiki-pilot",
        "sdk_root": "/opt/aws-fpga", "sdk_revision": "a" * 40,
        "source_sha256": "b" * 64, "corpus_sha256": "c" * 64,
        "corpus_key": "datasets/approved.txt", "corpus_license": "Public domain; approved corpus",
        "instance_type": "f2.6xlarge", "max_minutes": 90, "root_volume_gib": 50,
        "hourly_usd": "1.98", "storage_ip_artifact_allowance_usd": "7.03",
        "price_checked_on": datetime.now(timezone.utc).date().isoformat(),
        "price_source_url": "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json",
        "steps": 64, "eval_every": 32, "eval_windows": 2,
    }


@pytest.fixture
def ledger(config):
    return {"account": config["account"], "currency": "USD", "ceiling_usd": "1000", "runs": []}


def old_record(**overrides):
    return {
        "run_prefix": "tikitorch-fpga-lm/old-unique-run", "status": "terminated_reconciled",
        "instance_id": "i-0123456789abcdef0", "region": "us-east-1",
        "termination_verified_at": "2026-09-09T13:00:00Z",
        "reserved_usd": "10", "billed_usd": "2.50", **overrides,
    }


def test_plan_is_explicit_single_instance_private_and_bounded(config, ledger):
    result = A.plan(config, ledger)
    assert result["launch_enabled"] is False
    assert result["reservation_usd"] == "10.00"
    assert result["contingency_usd"] == "100"
    launch = result["proposed_run_instances"]
    assert launch["MinCount"] == launch["MaxCount"] == 1
    assert launch["InstanceType"] == "f2.6xlarge"
    assert launch["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert launch["MetadataOptions"]["HttpTokens"] == "required"
    assert launch["BlockDeviceMappings"][0]["Ebs"]["Encrypted"] is True
    assert launch["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"] is True
    assert launch["NetworkInterfaces"][0]["AssociatePublicIpAddress"] is False
    assert "KeyName" not in launch and "InstanceMarketOptions" not in launch
    assert len(launch["ClientToken"]) == 64
    assert launch["ClientToken"] == A.plan(config, ledger)["proposed_run_instances"]["ClientToken"]
    assert {tag["ResourceType"] for tag in launch["TagSpecifications"]} == {"instance", "volume"}


def test_explicit_no_ingress_public_network_and_developer_ami_disks(config, ledger):
    config.update(public_ipv4=True, network_exception_reason="No private endpoints; no inbound rules",
                  root_volume_gib=120, secondary_volume_gib=10)
    launch = A.plan(config, ledger)["proposed_run_instances"]
    assert launch["NetworkInterfaces"][0]["AssociatePublicIpAddress"] is True
    assert launch["DisableApiTermination"] is False
    assert [(item["DeviceName"], item["Ebs"]["VolumeSize"]) for item in
            launch["BlockDeviceMappings"]] == [("/dev/sda1", 120), ("/dev/sdm", 10)]
    assert all(item["Ebs"]["Encrypted"] and item["Ebs"]["DeleteOnTermination"]
               for item in launch["BlockDeviceMappings"])


def test_preflight_rejects_inherited_unreviewed_extra_disk(config, ledger):
    def query(c, *args):
        result = fake_aws(c, *args)
        if args[:2] == ("ec2", "describe-images"):
            result["Images"][0]["BlockDeviceMappings"] = [
                {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 50}},
                {"DeviceName": "/dev/sdm", "Ebs": {"VolumeSize": 10}}]
        return result
    with pytest.raises(ValueError, match="every AMI EBS mapping"):
        A.preflight(config, ledger, query=query)
    config["secondary_volume_gib"] = 10
    assert A.preflight(config, ledger, query=query)["status"] == "read_only_preflight_passed"


@pytest.mark.parametrize("key,value", [
    ("profile", "$(touch stolen)"), ("account", "not-an-account"),
    ("agfi", "agfi-bad"), ("region", "eu-west-1"),
    ("region", "us-east-2"), ("region", "us-west-1"),
    ("run_prefix", "tikitorch-fpga-lm/../../other"), ("sdk_root", "/opt/../etc"),
    ("sdk_revision", "main"), ("source_sha256", "sha256-missing"),
    ("corpus_key", "../../private"), ("corpus_key", "$(env)"),
    ("corpus_license", ""), ("instance_type", "f2.48xlarge"),
    ("max_minutes", 91), ("max_minutes", 0), ("max_minutes", True),
    ("max_minutes", 1.5), ("steps", 257), ("eval_windows", 9),
    ("root_volume_gib", 151), ("secondary_volume_gib", 21),
    ("public_ipv4", "true"), ("public_ipv4", True), ("hourly_usd", "NaN"),
    ("hourly_usd", "Infinity"), ("hourly_usd", 0),
    ("storage_ip_artifact_allowance_usd", "4.99"),
    ("storage_ip_artifact_allowance_usd", "20"),
    ("price_source_url", "https://example.org/prices"),
    ("price_checked_on", "2000-01-01"),
    ("instance_profile", "arn:aws:iam::999999999999:instance-profile/other"),
])
def test_invalid_inputs_fail_before_any_external_work(config, ledger, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        A.plan(config, ledger)


def test_estimate_rounds_up_but_full_envelope_is_reserved_and_prior_runs_not_refunded(config, ledger):
    config.update(max_minutes=1, hourly_usd="1.981", storage_ip_artifact_allowance_usd="5")
    ledger["runs"] = [old_record()]
    result = A.plan(config, ledger)
    assert result["reservation_usd"] == "10.00"
    assert result["estimated_compute_and_allowance_usd"] == "5.04"
    assert result["previous_accounted_usd"] == "10"
    assert result["proposed_accounted_usd"] == "20.00"


@pytest.mark.parametrize("change", [None, "missing_absence", "launch_attempt", "known_instance", "wrong_account", "reduced_reservation"])
def test_cancelled_prelaunch_records_preserve_budget_without_termination_claim(config, ledger, change):
    prefix = "tikitorch-fpga-lm/cancelled-old-pilot"
    token = hashlib.sha256(f"{config['account']}:{config['region']}:{prefix}".encode()).hexdigest()
    absence = {"account": config["account"], "region": config["region"], "run_prefix": prefix,
               "client_token": token, "matches": []}
    stamp = "2026-09-09T13:00:00Z"
    record = {**{key: absence[key] for key in ("account", "region", "run_prefix", "client_token")},
              "status": "cancelled_before_launch", "reserved_usd": "10.00", "billed_usd": "0",
              "absence_verified_at": stamp, "absence_evidence": absence,
              "events": [{"status": "launch_intent", "at": stamp},
                         {"status": "cancelled_before_launch", "at": stamp,
                          "absence_verified_at": stamp, "absence_evidence": absence}]}
    if change == "missing_absence":
        del record["absence_evidence"]
    elif change == "launch_attempt":
        record["events"].insert(1, {"status": "launch_requested", "at": stamp})
    elif change == "known_instance":
        record["instance_id"] = "i-0123456789abcdef0"
    elif change == "wrong_account":
        record["account"] = "999999999999"
    elif change == "reduced_reservation":
        record["reserved_usd"] = "0.01"
    ledger["runs"] = [record]
    if change is None:
        result = A.plan(config, ledger)
        assert result["previous_accounted_usd"] == "10.00"
        assert result["proposed_accounted_usd"] == "20.00"
        assert "instance_id" not in record and "termination_verified_at" not in record
    else:
        with pytest.raises(ValueError):
            A.plan(config, ledger)


@pytest.mark.parametrize("region", ["us-east-1", "us-west-2"])
def test_official_verified_pricing_host_and_region_are_supported(config, ledger, region):
    config.update(region=region, price_source_url="https://b0.p.awsstatic.com/pricing/verified-price.json")
    assert A.plan(config, ledger)["reservation_usd"] == "10.00"


@pytest.mark.parametrize("host", ["b0.p.awsstatic.com.example.org", "b0.p.awsstatic.com@evil.example", "unverified.awsstatic.com"])
def test_pricing_allowlist_rejects_lookalike_hosts(config, ledger, host):
    config["price_source_url"] = f"https://{host}/pricing.json"
    with pytest.raises(ValueError, match="official AWS"):
        A.plan(config, ledger)


def test_budget_ceiling_accounts_for_larger_billed_cost(config, ledger):
    ledger["runs"] = [old_record(reserved_usd="1", billed_usd="890")]
    assert A.plan(config, ledger)["proposed_accounted_usd"] == "900.00"
    ledger["runs"][0]["billed_usd"] = "890.01"
    with pytest.raises(ValueError, match=r"cross \$900"):
        A.plan(config, ledger)


@pytest.mark.parametrize("status", ["reserved", "running", "stopped", "shutting-down", "terminated", "failed", None])
def test_unreconciled_runs_block_further_work(config, ledger, status):
    ledger["runs"] = [old_record(status=status)]
    with pytest.raises(ValueError, match="unreconciled"):
        A.plan(config, ledger)


def test_duplicate_identity_and_wrong_account_rejected(config, ledger):
    ledger["runs"] = [old_record(), old_record()]
    with pytest.raises(ValueError, match="unique"):
        A.plan(config, ledger)
    ledger["runs"] = [old_record(run_prefix=config["run_prefix"])]
    with pytest.raises(ValueError, match="already been used"):
        A.plan(config, ledger)
    ledger["account"] = "999999999999"
    with pytest.raises(ValueError, match="bound"):
        A.plan(config, ledger)


def synthetic_sources(root):
    for name in A.SOURCE_FILES:
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("current untracked source: " + name)


def test_archive_is_exact_current_source_deterministic_and_never_overwrites(tmp_path):
    repo = tmp_path / "repo"
    synthetic_sources(repo)
    (repo / "secret.env").write_text("do not upload")
    (repo / "manuscript").mkdir()
    (repo / "manuscript" / "private.txt").write_text("do not upload research")
    first, second = tmp_path / "first.tgz", tmp_path / "second.tgz"
    record = A.build_bundle(first, root=repo)
    assert record["sha256"] == A.build_bundle(second, root=repo)["sha256"]
    with tarfile.open(first) as archive:
        assert set(archive.getnames()) == set(A.SOURCE_FILES) | {"SOURCE_MANIFEST.json"}
        manifest = json.load(archive.extractfile("SOURCE_MANIFEST.json"))
        for name in A.SOURCE_FILES:
            assert manifest[name] == hashlib.sha256((repo / name).read_bytes()).hexdigest()
    preserved = first.read_bytes()
    with pytest.raises(FileExistsError):
        A.build_bundle(first, root=repo)
    assert first.read_bytes() == preserved


def test_missing_or_symlinked_sources_fail_before_archive_creation(tmp_path):
    repo = tmp_path / "repo"
    synthetic_sources(repo)
    target = repo / A.SOURCE_FILES[0]
    target.unlink()
    output = tmp_path / "source.tgz"
    with pytest.raises(ValueError, match="missing or symlinked"):
        A.build_bundle(output, root=repo)
    target.symlink_to(repo / A.SOURCE_FILES[1])
    with pytest.raises(ValueError, match="missing or symlinked"):
        A.build_bundle(output, root=repo)
    assert not output.exists()


def test_render_has_fixed_archive_integrity_kat_deadline_and_no_credentials(config, ledger):
    script = A.render(config, ledger)
    checked = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr
    assert "set -euo pipefail" in script and "set -x" not in script
    assert 'shutdown -h +90 "TikiTorch FPGA LM pilot deadline"\n' in script
    assert "sha256sum -c -" in script and "source archive allowlist mismatch" in script
    assert "git clone" not in script and "pip install" not in script
    assert "export HOME=" not in script and "AWS_ACCESS_KEY_ID" not in script
    assert '--backend pci --library' in script
    assert '-DTT_FPGA_PCI' in script and 'scripts/fpga_lm_layout.cpp' in script
    assert script.index('cmp "$WORK/golden.jsonl" "$WORK/silicon.jsonl"') < script.index('python3 scripts/fpga_lm_pretrain.py')
    assert script.index('sha256sum -c -') < script.index('tarfile.open(')
    assert script.index('shutdown -h +90') < script.index('aws s3 cp "$S3_SOURCE"')
    assert 'RESULT=SUCCESS' in script and 'trap finish EXIT' in script


@pytest.mark.parametrize("same", [False, True])
def test_rendered_kat_gate_blocks_training_on_mismatch(config, ledger, tmp_path, same):
    script = A.render(config, ledger)
    start = script.index('test -s "$WORK/golden.jsonl"')
    end = script.index('python3 scripts/fpga_lm_pretrain.py', start)
    gate = script[start:end]
    (tmp_path / "golden.jsonl").write_text('{"wcrc":"test"}\n')
    (tmp_path / "silicon.jsonl").write_text('{"wcrc":"test"}\n' if same else '{"wcrc":"wrong"}\n')
    result = subprocess.run(["bash", "-e", "-c", gate + 'printf "TRAINING_ALLOWED"'],
                            env={"WORK": str(tmp_path)}, text=True, capture_output=True)
    assert (result.returncode == 0) == same
    assert ("TRAINING_ALLOWED" in result.stdout) == same


def fake_aws(config, *args):
    key = tuple(args[:2])
    replies = {
        ("sts", "get-caller-identity"): {"Account": config["account"]},
        ("ec2", "describe-images"): {"Images": [{"OwnerId": config["ami_owner"], "State": "available", "Architecture": "x86_64", "RootDeviceName": "/dev/sda1"}]},
        ("ec2", "describe-fpga-images"): {"FpgaImages": [{"OwnerId": config["account"], "State": {"Code": "available"}}]},
        ("ec2", "describe-subnets"): {"Subnets": [{"OwnerId": config["account"], "VpcId": "vpc-test", "State": "available"}]},
        ("ec2", "describe-security-groups"): {"SecurityGroups": [{"OwnerId": config["account"], "VpcId": "vpc-test", "IpPermissions": []}]},
        ("iam", "get-instance-profile"): {"InstanceProfile": {"Arn": config["instance_profile"], "Roles": [{}]}},
        ("s3api", "head-bucket"): {},
        ("s3api", "get-bucket-location"): {"LocationConstraint": None},
        ("s3api", "get-public-access-block"): {"PublicAccessBlockConfiguration": dict.fromkeys(
            ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"), True)},
        ("s3api", "list-objects-v2"): {"KeyCount": 0},
        ("s3api", "head-object"): {"ContentLength": 1048576},
        ("ec2", "describe-instances"): {"Reservations": []},
    }
    assert key in replies, "unexpected or mutating AWS operation"
    return replies[key]


@pytest.fixture
def staged(config):
    prefix = config["run_prefix"]
    payloads = {prefix + "/source.tar.gz": b"approved source unit fixture",
                prefix + "/sdk.tar": b"approved SDK unit fixture",
                prefix + "/numpy-wheels.tar": b"approved runtime unit fixture"}
    config.update(sdk_s3_key=prefix + "/sdk.tar", numpy_s3_key=prefix + "/numpy-wheels.tar",
                  source_sha256=hashlib.sha256(payloads[prefix + "/source.tar.gz"]).hexdigest(),
                  sdk_sha256=hashlib.sha256(payloads[prefix + "/sdk.tar"]).hexdigest(),
                  numpy_archive_sha256=hashlib.sha256(payloads[prefix + "/numpy-wheels.tar"]).hexdigest())
    listing = {"IsTruncated": False, "KeyCount": 3,
               "Contents": [{"Key": key, "Size": len(data)} for key, data in payloads.items()]}
    calls, reads = [], []
    def query(c, *args):
        calls.append(args)
        if args[:2] == ("s3api", "list-objects-v2"):
            return listing
        if args[:2] == ("s3api", "head-object") and args[args.index("--key") + 1] in payloads:
            return {"ContentLength": len(payloads[args[args.index("--key") + 1]])}
        return fake_aws(c, *args)
    def read_object(c, key, expected_bytes):
        assert c == config and expected_bytes <= A.MAX_STAGED_BYTES
        reads.append(key)
        return payloads[key]
    return payloads, listing, query, read_object, calls, reads


def test_staged_preflight_honestly_verifies_all_three_inputs_without_relaxing_empty_mode(config, ledger, staged):
    payloads, _, query, reader, calls, reads = staged
    result = A.preflight_staged(config, ledger, query=query, read_object=reader)
    assert result["status"] == "read_only_staged_preflight_passed" and result["launch_enabled"] is False
    assert result["s3_prefix_state"] == "approved_staged_inputs"
    assert result["exact_token_instance_and_watchdog_absence_verified"] is False
    assert result["verified_staged_inputs"] == {
        key: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for key, data in payloads.items()}
    assert set(reads) == set(payloads)
    listing_call = next(args for args in calls if args[:2] == ("s3api", "list-objects-v2"))
    assert listing_call[listing_call.index("--max-keys") + 1] == "4" and "--no-paginate" in listing_call
    for args in calls:
        if args[0] == "s3api":
            assert args[args.index("--expected-bucket-owner") + 1] == config["account"]
    with pytest.raises(ValueError, match="already contains"):
        A.preflight(config, ledger, query=query)


@pytest.mark.parametrize("change", ["missing_flag", "truncated", "continuation", "token", "missing_contents",
    "wrong_count", "bool_count", "duplicate", "unexpected", "missing_source", "bad_size", "oversized"])
def test_staged_preflight_rejects_incomplete_malformed_or_unexpected_listing_before_download(config, ledger, staged, change):
    _, listing, query, reader, _, reads = staged
    if change == "missing_flag":
        del listing["IsTruncated"]
    elif change == "truncated":
        listing["IsTruncated"] = True
    elif change == "continuation":
        listing["NextContinuationToken"] = None
    elif change == "token":
        listing["NextToken"] = "additional"
    elif change == "missing_contents":
        del listing["Contents"]
    elif change == "wrong_count":
        listing["KeyCount"] = 4
    elif change == "bool_count":
        listing["KeyCount"] = True
    elif change == "duplicate":
        listing["Contents"][2] = listing["Contents"][0]
    elif change == "unexpected":
        listing["Contents"][0]["Key"] = config["run_prefix"] + "/control/start.json"
    elif change == "missing_source":
        listing["Contents"].pop(0); listing["KeyCount"] = 2
    else:
        listing["Contents"][0]["Size"] = 0 if change == "bad_size" else A.MAX_STAGED_BYTES + 1
    with pytest.raises(ValueError, match="staged"):
        A.preflight_staged(config, ledger, query=query, read_object=reader)
    assert not reads


@pytest.mark.parametrize("change", ["head", "hash", "length", "type"])
def test_staged_preflight_requires_real_bounded_matching_bytes(config, ledger, staged, change):
    _, _, query, reader, _, _ = staged
    def changed_query(c, *args):
        result = query(c, *args)
        if change == "head" and args[:2] == ("s3api", "head-object"):
            return {"ContentLength": 0}
        return result
    def changed_reader(c, key, size):
        payload = reader(c, key, size)
        return payload[:-1] if change == "length" else "not bytes" if change == "type" else b"x" * size
    with pytest.raises(ValueError, match="staged"):
        A.preflight_staged(config, ledger, query=changed_query, read_object=changed_reader)


def test_staged_preflight_rejects_any_existing_run_identity_before_aws(config, ledger, staged):
    _, _, query, reader, calls, _ = staged
    ledger["runs"] = [old_record(run_prefix=config["run_prefix"])]
    with pytest.raises(ValueError, match="already been used"):
        A.preflight_staged(config, ledger, query=query, read_object=reader)
    assert not calls


@pytest.mark.parametrize("field", ["sdk_sha256", "numpy_archive_sha256", "sdk_s3_key", "numpy_s3_key"])
def test_staged_config_requires_exact_input_hashes_and_keys_before_aws(config, ledger, staged, field):
    _, _, query, reader, calls, _ = staged
    config[field] = "not-approved"
    with pytest.raises(ValueError):
        A.preflight_staged(config, ledger, query=query, read_object=reader)
    assert not calls


@pytest.mark.parametrize("bad", [None, "total_size", "length", "payload"])
def test_default_staged_reader_uses_bounded_expected_owner_explicit_get_object(config, monkeypatch, bad):
    data = b"bounded readback"
    calls = []
    def query(c, *args):
        calls.append(args)
        assert args[:2] == ("s3api", "get-object")
        assert args[args.index("--expected-bucket-owner") + 1] == config["account"]
        assert args[args.index("--range") + 1] == f"bytes=0-{len(data) - 1}"
        assert "--cli-input-json" not in args
        Path(args[-1]).write_bytes(data[:-1] if bad == "payload" else data)
        return {"ContentRange": f"bytes 0-{len(data) - 1}/{len(data) + (1 if bad == 'total_size' else 0)}",
                "ContentLength": len(data) + (1 if bad == "length" else 0)}
    monkeypatch.setattr(A, "aws_json", query)
    if bad is None:
        assert A.aws_object_bytes(config, config["run_prefix"] + "/source.tar.gz", len(data)) == data
    else:
        with pytest.raises(ValueError, match="staged"):
            A.aws_object_bytes(config, config["run_prefix"] + "/source.tar.gz", len(data))
    assert len(calls) == 1
    with pytest.raises(ValueError, match="128 MiB"):
        A.aws_object_bytes(config, "unread", A.MAX_STAGED_BYTES + 1)
    assert len(calls) == 1


def test_staged_cli_dispatch_is_explicit_and_preserves_records(config, ledger, tmp_path, monkeypatch, capsys):
    config_path, ledger_path = tmp_path / "config.json", tmp_path / "ledger.json"
    config_path.write_text(json.dumps(config)); ledger_path.write_text(json.dumps(ledger))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    def checked(c, l):
        assert c == config and l == ledger
        return {"status": "offline_staged_dispatch_test", "launch_enabled": False}
    monkeypatch.setattr(A, "preflight_staged", checked)
    assert A.main(["preflight-staged", "--config", str(config_path), "--ledger", str(ledger_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "offline_staged_dispatch_test"
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_preflight_is_read_only_and_still_does_not_enable_launch(config, ledger):
    calls = []
    def query(c, *args):
        calls.append(args)
        return fake_aws(c, *args)
    result = A.preflight(config, ledger, query=query)
    assert result["status"] == "read_only_preflight_passed"
    assert result["launch_enabled"] is False
    assert result["ami_root_snapshot_minimum_verified"] is False
    assert result["private_subnet_runtime_reachability_verified"] is False
    assert result["paired_fresh_window_full_checkpoint_equality_verified"] is False
    assert any("full weight-checkpoint equality" in gate for gate in result["remaining_launch_gates"])
    assert len(calls) == 12


@pytest.mark.parametrize("size,passes", [(30, True), (100, False)])
def test_preflight_rejects_known_oversized_ami_root_mapping(config, ledger, size, passes):
    def query(c, *args):
        result = fake_aws(c, *args)
        if args[1] == "describe-images":
            result["Images"][0]["BlockDeviceMappings"] = [
                {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": size, "SnapshotId": "snap-unknown"}}]
        return result
    if passes:
        result = A.preflight(config, ledger, query=query)
        assert result["ami_declared_root_volume_gib"] == size
        assert result["ami_root_snapshot_minimum_verified"] is False
    else:
        with pytest.raises(ValueError, match="smaller than the AMI"):
            A.preflight(config, ledger, query=query)


@pytest.mark.parametrize("operation,replacement,match", [
    ("get-caller-identity", {"Account": "999999999999"}, "account"),
    ("describe-fpga-images", {"FpgaImages": []}, "AGFI"),
    ("get-bucket-location", {"LocationConstraint": "us-west-2"}, "region"),
    ("get-public-access-block", {}, "public-access"),
    ("list-objects-v2", {"KeyCount": 1}, "already contains"),
    ("head-object", {"ContentLength": 1024 ** 3}, "corpus"),
    ("describe-instances", {"Reservations": [{"Instances": [{}]}]}, "reconciliation"),
])
def test_preflight_rejects_unsafe_resource_state(config, ledger, operation, replacement, match):
    def query(c, *args):
        return replacement if args[1] == operation else fake_aws(c, *args)
    with pytest.raises(ValueError, match=match):
        A.preflight(config, ledger, query=query)


def test_cli_default_and_disabled_launch_never_call_aws(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("external command must not run")
    monkeypatch.setattr(subprocess, "run", forbidden)
    assert A.main([]) == 0
    assert "config" in json.loads(capsys.readouterr().out)
    assert A.main(["launch"]) == 2
    assert "launch is disabled" in capsys.readouterr().err


def test_cli_plan_render_read_only_and_missing_inputs_fail(tmp_path, config, ledger, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("external command must not run")
    monkeypatch.setattr(subprocess, "run", forbidden)
    config_path, ledger_path = tmp_path / "config.json", tmp_path / "ledger.json"
    config_path.write_text(json.dumps(config)); ledger_path.write_text(json.dumps(ledger))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    args = ["--config", str(config_path), "--ledger", str(ledger_path)]
    assert A.main(["plan", *args]) == 0
    assert json.loads(capsys.readouterr().out)["reservation_usd"] == "10.00"
    assert A.main(["render", *args]) == 0
    assert capsys.readouterr().out.startswith("#!/bin/bash")
    assert A.main(["plan"]) == 2
    assert "explicit --config and --ledger" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_money_rejects_boolean_and_nonfinite():
    assert A.money("1.98", "rate") == Decimal("1.98")
    with pytest.raises(ValueError):
        A.money(True, "rate")


def test_aws_errors_do_not_echo_credential_provider_details(config, monkeypatch):
    def failed(command, **kwargs):
        assert command[:6] == ["aws", "--profile", config["profile"], "--region", config["region"], "--no-cli-pager"]
        return subprocess.CompletedProcess(command, 1, "private token", "private provider details")
    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(ValueError, match="authentication/permissions") as error:
        A.aws_json(config, "sts", "get-caller-identity")
    assert "private" not in str(error.value)
