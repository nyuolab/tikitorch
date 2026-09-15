"""Injected-adapter lifecycle tests: no AWS access, credentials or paid resources."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import fpga_lm_control as C


@pytest.fixture
def config():
    return {
        "profile": "offline", "account": "123456789012", "region": "us-east-1",
        "ami": "ami-0123456789abcdef0", "ami_owner": "210987654321",
        "agfi": "agfi-0123456789abcdef0", "bucket": "tiki-private-research",
        "run_prefix": "tikitorch-fpga-lm/pilot-20260909-control",
        "subnet": "subnet-0123456789abcdef0", "security_group": "sg-0123456789abcdef0",
        "instance_profile": "arn:aws:iam::123456789012:instance-profile/tiki-pilot",
        "sdk_root": "/opt/aws-fpga", "sdk_revision": "a" * 40,
        "source_sha256": "b" * 64, "corpus_sha256": "c" * 64,
        "corpus_key": "datasets/approved.txt", "corpus_license": "Public domain; approved corpus",
        "instance_type": "f2.6xlarge", "max_minutes": 1, "root_volume_gib": 50,
        "hourly_usd": "1.98", "storage_ip_artifact_allowance_usd": "5",
        "price_checked_on": datetime.now(timezone.utc).date().isoformat(),
        "price_source_url": "https://b0.p.awsstatic.com/pricing/verified-test.json",
        "steps": 64, "eval_every": 32, "eval_windows": 2,
        "guest_user_data": "#!/bin/bash\n# Offline fixture; no executable workload.\n",
        "watchdog_execution_arn": "arn:aws:states:us-east-1:123456789012:execution:tiki-test:pilot-test",
    }


class Clock:
    def __init__(self):
        self.elapsed = 0.0
        self.origin = datetime.now(timezone.utc)

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        assert 0 <= seconds <= 60
        self.elapsed += seconds

    def now(self):
        return self.origin + timedelta(seconds=self.elapsed)


class Cloud:
    """Canonical adapter responses, not an AWS mock that could reach the network."""

    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config
        self.calls = []
        self.instance = None
        self.watchdog = None
        self.ambiguous_launch = None
        self.healthy = True
        self.poll = "success"
        self.terminate = True
        self.retrieval = True
        self.bad_identity = False

    def record(self):
        return json.loads((self.directory / "ledger.json").read_text())["runs"][-1]

    def observed(self, payload):
        return {"instance_id": "i-0123456789abcdef0", "account": payload["account"],
                "region": payload["region"], "client_token": payload["client_token"],
                "tags": {"Project": C.PROJECT_TAG, "Run": payload["run_prefix"].split("/")[-1],
                         "OwnerAccount": payload["account"]}, "state": "running"}

    def __call__(self, action, payload):
        self.calls.append(action)
        record = self.record()
        # Every call sees the durable intent/reservation; mutations never precede it.
        assert record["reserved_usd"] == "10.00"
        assert record["client_token"] == payload["client_token"]
        if action == "identity":
            return {"account": "999999999999" if self.bad_identity else payload["account"],
                    "region": payload["region"]}
        if action == "find_by_client_token":
            return [] if self.instance is None else [self.instance.copy()]
        if action == "arm_watchdog":
            assert self.instance is None
            self.watchdog = {"verified": self.healthy, "execution_arn": payload["execution_arn"],
                             "account": payload["account"], "region": payload["region"],
                             "client_token": payload["client_token"], "run_prefix": payload["run_prefix"],
                             "deadline_utc": payload["deadline_utc"], "startup_health_check_passed": self.healthy,
                             "status": "RUNNING", "state": "WaitUntilDeadline"}
            return self.watchdog.copy()
        if action == "watchdog_health":
            assert record["status"] == "watchdog_verified"
            return self.watchdog.copy()
        if action == "launch_instance":
            assert record["status"] == "launch_requested"
            assert record["request"] == payload["request"]
            assert hashlib.sha256(C.encoded(payload["request"])).hexdigest() == record["request_sha256"]
            assert "watchdog_health" in self.calls
            if self.ambiguous_launch != "without_instance":
                self.instance = self.observed(payload)
            if self.ambiguous_launch:
                raise TimeoutError("simulated lost launch response")
            return self.instance.copy()
        if action == "describe_instance":
            assert payload["instance_id"] == self.instance["instance_id"]
            return self.instance.copy()
        if action == "authorize_work":
            assert record["instance_id"] == payload["instance_id"]
            assert record["watchdog"]["execution_arn"] == payload["watchdog_execution_arn"]
            return {**payload, "authorized": True}
        if action == "poll_result":
            if isinstance(self.poll, BaseException):
                raise self.poll
            return {"status": self.poll}
        if action == "retrieve_results":
            return {"fresh": self.retrieval, "run_prefix": payload["run_prefix"],
                    "directory": str(self.directory / "fresh-artifacts"),
                    "artifact_checks_passed": self.retrieval}
        if action == "terminate_instance":
            assert record["status"] == "termination_requested"
            if isinstance(self.terminate, Exception):
                raise self.terminate
            if self.terminate:
                self.instance["state"] = "terminated"
            return {}
        raise AssertionError(f"unknown adapter action {action}")


def supervise(config, directory, cloud, clock):
    return C.supervise(config, directory, cloud, now=clock.now, monotonic=clock.monotonic,
                       sleep=clock.sleep, poll_seconds=2, termination_wait_seconds=4)


def test_success_is_prearmed_durable_retrieved_terminated_but_not_refunded(config, tmp_path):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    result = supervise(config, tmp_path, cloud, clock)
    assert result["termination_confirmed"] is True
    assert result["status"] == "terminated_unreconciled"
    assert result["reservation_usd"] == "10.00"
    assert cloud.calls.index("arm_watchdog") < cloud.calls.index("watchdog_health") < cloud.calls.index("launch_instance")
    assert cloud.calls.index("launch_instance") < cloud.calls.index("authorize_work") < cloud.calls.index("poll_result")
    assert cloud.calls.index("retrieve_results") < cloud.calls.index("terminate_instance")
    record = cloud.record()
    assert record["termination_confirmed"] is True
    assert record["request"]["UserData"] == config["guest_user_data"]
    assert [item["status"] for item in record["events"]][0] == "launch_intent"
    with C.Ledger(tmp_path, config["account"]) as ledger:
        with pytest.raises(C.LifecycleError, match="unreconciled"):
            ledger.reserve({**config, "run_prefix": "tikitorch-fpga-lm/another-pilot"})
        ledger.reconcile(config["run_prefix"], "0.01", "Offline test confirms termination and retrieves artifacts")
        assert ledger.get(config["run_prefix"])["reserved_usd"] == "10.00"
        assert ledger.get(config["run_prefix"])["status"] == C.TERMINAL
        with pytest.raises(ValueError, match="already been used"):
            ledger.reserve(config)


@pytest.mark.parametrize("healthy", [False, "changes_before_launch"])
def test_unhealthy_prearmed_watchdog_blocks_all_ec2_writes(config, tmp_path, healthy):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    cloud.healthy = healthy is not False
    def adapter(action, payload):
        response = cloud(action, payload)
        if action == "watchdog_health" and healthy == "changes_before_launch":
            response["status"] = "FAILED"
        return response
    with pytest.raises(C.LifecycleError, match="watchdog"):
        supervise(config, tmp_path, adapter, clock)
    assert "launch_instance" not in cloud.calls
    assert "authorize_work" not in cloud.calls
    assert cloud.record()["reserved_usd"] == "10.00"


@pytest.mark.parametrize("kind", ["with_instance", "without_instance"])
def test_ambiguous_launch_reconciles_token_and_never_blindly_retries(config, tmp_path, kind):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    cloud.ambiguous_launch = kind
    if kind == "with_instance":
        assert supervise(config, tmp_path, cloud, clock)["termination_confirmed"] is True
    else:
        with pytest.raises(C.LifecycleError, match="ambiguous"):
            supervise(config, tmp_path, cloud, clock)
        assert cloud.record()["status"] == "launch_ambiguous"
        assert "terminate_instance" not in cloud.calls
    assert cloud.calls.count("launch_instance") == 1


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("guest transport failed"), "failed"])
def test_failure_or_control_c_terminates_exact_owned_instance(config, tmp_path, failure):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    cloud.poll = failure
    expected = type(failure) if isinstance(failure, BaseException) else C.LifecycleError
    with pytest.raises(expected):
        supervise(config, tmp_path, cloud, clock)
    assert cloud.calls.count("terminate_instance") == 1
    assert cloud.record()["status"] == "terminated_unreconciled"


def test_controller_monotonic_deadline_does_not_run_indefinitely(config, tmp_path):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    cloud.poll = "running"
    with pytest.raises(TimeoutError, match="monotonic deadline"):
        supervise(config, tmp_path, cloud, clock)
    assert clock.elapsed == 60
    assert cloud.record()["termination_confirmed"] is True


@pytest.mark.parametrize("termination", [False, RuntimeError("API unavailable")])
def test_unconfirmed_termination_preserves_reservation_state_and_blocks_next(config, tmp_path, termination):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    cloud.terminate = termination
    with pytest.raises(C.LifecycleError, match="termination remains unconfirmed"):
        supervise(config, tmp_path, cloud, clock)
    assert cloud.record()["status"] == "termination_unconfirmed"
    assert cloud.record()["reserved_usd"] == "10.00"
    with C.Ledger(tmp_path, config["account"]) as ledger:
        with pytest.raises(C.LifecycleError, match="unreconciled"):
            ledger.reserve(config)
        with pytest.raises(C.LifecycleError, match="confirmed terminated"):
            ledger.reconcile(config["run_prefix"], "0", "cannot yet prove termination")


def test_wrong_account_blocks_launch(config, tmp_path):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    cloud.bad_identity = True
    with pytest.raises(C.LifecycleError, match="account/region"):
        supervise(config, tmp_path, cloud, clock)
    assert "launch_instance" not in cloud.calls


def test_foreign_instance_is_never_adopted_or_terminated(config, tmp_path):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    def adapter(action, payload):
        if action == "find_by_client_token":
            foreign = cloud.observed(payload)
            foreign["instance_id"] = "i-02f1a1ba582558949"  # protected pre-existing sentinel
            foreign["tags"]["Run"] = "unrelated-existing-work"
            return [foreign]
        return cloud(action, payload)
    with pytest.raises(C.LifecycleError, match="tags"):
        supervise(config, tmp_path, adapter, clock)
    assert "launch_instance" not in cloud.calls and "terminate_instance" not in cloud.calls


@pytest.mark.parametrize("field,value", [("client_token", "other"), ("region", "us-west-2"),
                                         ("account", "999999999999"), ("instance_id", "i-00000000000000000")])
def test_ownership_change_prevents_termination_of_wrong_target(config, tmp_path, field, value):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    changed = False
    def adapter(action, payload):
        nonlocal changed
        response = cloud(action, payload)
        if action == "poll_result":
            changed = True
            raise RuntimeError("stop after ownership changes")
        if action == "describe_instance" and changed:
            response[field] = value
        return response
    with pytest.raises(RuntimeError, match="ownership changes"):
        supervise(config, tmp_path, adapter, clock)
    assert "terminate_instance" not in cloud.calls
    assert cloud.record()["status"] == "termination_unconfirmed"


def test_retrieval_must_be_fresh_and_validated(config, tmp_path):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    cloud.retrieval = False
    with pytest.raises(C.LifecycleError, match="fresh complete artifact"):
        supervise(config, tmp_path, cloud, clock)
    assert cloud.record()["status"] == "terminated_unreconciled"


def test_local_lock_blocks_second_controller_and_atomic_replacement_preserves_disk(config, tmp_path, monkeypatch):
    with C.Ledger(tmp_path, config["account"]) as ledger:
        with pytest.raises(C.LifecycleError, match="holds.*lock"):
            with C.Ledger(tmp_path, config["account"]):
                pass
        before = (tmp_path / "ledger.json").read_bytes()
        def fail_replace(*args):
            raise OSError("simulated crash before atomic replacement")
        monkeypatch.setattr(C.os, "replace", fail_replace)
        with pytest.raises(OSError, match="atomic replacement"):
            ledger.reserve(config)
        assert (tmp_path / "ledger.json").read_bytes() == before
        assert not list(tmp_path.glob(".ledger-*.tmp"))


def test_mutations_fsync_file_and_directory(config, tmp_path, monkeypatch):
    original = C.os.fsync
    calls = []
    def tracked(fd):
        calls.append(fd)
        return original(fd)
    monkeypatch.setattr(C.os, "fsync", tracked)
    with C.Ledger(tmp_path, config["account"]) as ledger:
        assert len(calls) == 2
        ledger.reserve(config)
        assert len(calls) == 4


def test_recovery_reconciles_and_terminates_without_relaunching(config, tmp_path):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    with C.Ledger(tmp_path, config["account"], now=clock.now) as ledger:
        record = ledger.reserve(config)
        ledger.event(record["run_prefix"], "launch_requested")
        cloud.instance = cloud.observed(record)
    assert C.recover(tmp_path, config["account"], config["run_prefix"], cloud,
                     monotonic=clock.monotonic, sleep=clock.sleep, wait_seconds=4) is True
    assert "launch_instance" not in cloud.calls and "authorize_work" not in cloud.calls
    assert cloud.record()["status"] == "terminated_unreconciled"


def test_cancel_unlaunched_retains_reservation_and_allows_a_distinct_next_run(config, tmp_path):
    cloud = Cloud(tmp_path, config)
    with C.Ledger(tmp_path, config["account"]) as ledger:
        ledger.reserve(config)
        ledger.event(config["run_prefix"], "controller_failed", controller_error_type="LifecycleError")
    before = cloud.record()
    result = C.cancel_before_launch(tmp_path, config["account"], config["run_prefix"], cloud)
    assert result["status"] == C.CANCELLED and result["reservation_usd"] == "10.00"
    assert cloud.calls == ["identity", "find_by_client_token"]
    cancelled = cloud.record()
    assert "instance_id" not in cancelled and "termination_confirmed" not in cancelled
    assert cancelled["events"][:-1] == before["events"]
    assert cancelled["absence_evidence"] == {**C.identity_payload(cancelled), "matches": []}
    with pytest.raises(C.LifecycleError, match="historical"):
        C.recover(tmp_path, config["account"], config["run_prefix"], cloud)
    with pytest.raises(C.LifecycleError, match="historical"):
        C.cancel_before_launch(tmp_path, config["account"], config["run_prefix"], cloud)
    assert cloud.calls == ["identity", "find_by_client_token"]
    with C.Ledger(tmp_path, config["account"]) as ledger:
        ledger.reserve({**config, "run_prefix": "tikitorch-fpga-lm/next-after-cancel"})
        assert sum(C.validate_amount(row["reserved_usd"]) for row in ledger.data["runs"]) == 20
        assert ledger.data["runs"][0] == cancelled


@pytest.mark.parametrize("status,facts", [
    ("launch_requested", {}), ("launch_ambiguous", {}),
    ("instance_reconciled", {"instance_id": "i-0123456789abcdef0"}),
    ("controller_failed", {"instance_id": "i-0123456789abcdef0"}),
    ("controller_failed", {"guest_authorized": True}),
    ("termination_unconfirmed", {"termination_confirmed": False}),
])
def test_cancel_refuses_any_launch_or_instance_history_without_querying_aws(config, tmp_path, status, facts):
    cloud = Cloud(tmp_path, config)
    with C.Ledger(tmp_path, config["account"]) as ledger:
        ledger.reserve(config)
        ledger.event(config["run_prefix"], status, **facts)
        if status == "launch_requested":
            ledger.event(config["run_prefix"], "controller_failed")
    before = (tmp_path / "ledger.json").read_bytes()
    with pytest.raises(ValueError, match="prelaunch-only"):
        C.cancel_before_launch(tmp_path, config["account"], config["run_prefix"], cloud)
    assert not cloud.calls
    assert (tmp_path / "ledger.json").read_bytes() == before


@pytest.mark.parametrize("case", ["matching_instance", "account_mismatch", "region_mismatch", "discovery_error"])
def test_cancel_requires_authenticated_exact_run_absence(config, tmp_path, case):
    cloud = Cloud(tmp_path, config)
    with C.Ledger(tmp_path, config["account"]) as ledger:
        record = ledger.reserve(config)
    if case == "matching_instance":
        cloud.instance = cloud.observed(record)
    cloud.bad_identity = case == "account_mismatch"
    def invoke(action, payload):
        result = cloud(action, payload)
        if action == "identity" and case == "region_mismatch":
            result["region"] = "us-west-2"
        if action == "find_by_client_token" and case == "discovery_error":
            raise C.LifecycleError("read-only discovery failed")
        return result
    before = (tmp_path / "ledger.json").read_bytes()
    with pytest.raises(C.LifecycleError):
        C.cancel_before_launch(tmp_path, config["account"], config["run_prefix"], invoke)
    assert set(cloud.calls) <= {"identity", "find_by_client_token"}
    assert (tmp_path / "ledger.json").read_bytes() == before


@pytest.mark.parametrize("change", ["missing_absence", "launch_event", "lower_reservation"])
def test_cancelled_historical_record_is_revalidated_on_load(config, tmp_path, change):
    cloud = Cloud(tmp_path, config)
    with C.Ledger(tmp_path, config["account"]) as ledger:
        ledger.reserve(config)
    C.cancel_before_launch(tmp_path, config["account"], config["run_prefix"], cloud)
    path = tmp_path / "ledger.json"
    data = json.loads(path.read_bytes())
    if change == "missing_absence":
        del data["runs"][0]["absence_evidence"]
    elif change == "launch_event":
        data["runs"][0]["events"].insert(1, {"status": "launch_requested", "at": C.iso(C.utc_now())})
    else:
        data["runs"][0]["reserved_usd"] = "0.01"
    path.write_bytes(C.encoded(data))
    with pytest.raises((ValueError, C.LifecycleError)):
        with C.Ledger(tmp_path, config["account"]):
            pass


def test_exact_guest_deadline_is_preserved_and_limits_controller(config, tmp_path):
    clock, cloud = Clock(), Cloud(tmp_path, config)
    config["deadline_utc"] = (clock.origin + timedelta(seconds=20)).isoformat().replace("+00:00", "Z")
    cloud.poll = "running"
    with pytest.raises(TimeoutError):
        supervise(config, tmp_path, cloud, clock)
    assert clock.elapsed == 20
    assert cloud.record()["deadline_utc"] == config["deadline_utc"]


@pytest.mark.parametrize("field,value", [("guest_user_data", ""), ("guest_user_data", "#!/bin/bash\n" + "x" * (256 * 1024)),
                                         ("watchdog_execution_arn", "arn:aws:states:us-west-2:999999999999:execution:other:other")])
def test_unreviewed_guest_or_watchdog_identity_rejected_before_adapter(config, tmp_path, field, value):
    config[field] = value
    def forbidden(*args):
        raise AssertionError("no adapter call is permitted")
    with pytest.raises(ValueError):
        C.supervise(config, tmp_path, forbidden)


def test_prior_reservations_count_toward_900_and_never_use_contingency(config, tmp_path):
    with C.Ledger(tmp_path, config["account"]) as ledger:
        first = ledger.reserve(config)
        ledger.event(first["run_prefix"], "terminated_unreconciled", instance_id="i-0123456789abcdef0",
                     termination_verified_at=C.iso(C.utc_now()), termination_confirmed=True)
        ledger.reconcile(first["run_prefix"], "890.01", "large prior independently reconciled spend")
        with pytest.raises(ValueError, match=r"cross \$900"):
            ledger.reserve({**config, "run_prefix": "tikitorch-fpga-lm/next-unique-pilot"})


def test_tampered_immutable_request_or_wrong_account_cannot_load(config, tmp_path):
    with C.Ledger(tmp_path, config["account"]) as ledger:
        ledger.reserve(config)
    with pytest.raises(C.LifecycleError, match="identity mismatch"):
        with C.Ledger(tmp_path, "999999999999"):
            pass
    path = tmp_path / "ledger.json"
    data = json.loads(path.read_text())
    data["runs"][0]["request"]["InstanceType"] = "f2.48xlarge"
    path.write_text(json.dumps(data))
    with pytest.raises(C.LifecycleError, match="digest mismatch"):
        with C.Ledger(tmp_path, config["account"]):
            pass


def test_direct_stdlib_cli_cannot_launch(tmp_path):
    script = Path(C.__file__)
    status = subprocess.run([sys.executable, "-S", str(script)], cwd=tmp_path,
                            text=True, capture_output=True, timeout=10)
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["launch_enabled"] is False
    launch = subprocess.run([sys.executable, "-S", str(script), "launch"], cwd=tmp_path,
                            text=True, capture_output=True, timeout=10)
    assert launch.returncode == 2
    assert "launch disabled" in launch.stderr
    assert not list(tmp_path.iterdir())
