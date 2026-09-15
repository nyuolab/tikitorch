#!/usr/bin/env python3
"""Durable, injected-adapter lifecycle contracts for one FPGA LM pilot.

The separate ``scripts.fpga_lm_cloud`` adapter provides explicitly authorized
live AWS operations; this controller's direct CLI still cannot launch. The
adapter contract is deliberately small and explicit; integration must verify all returned facts
against AWS, not merely echo request fields. In particular ``arm_watchdog`` must
start AND verify a Standard Step Functions watchdog, whose filtered discovery
uses the exact run tags and client token. The watchdog's StartupHealthCheck must
pass and its state must be WaitUntilDeadline BEFORE RunInstances. Only after
that receipt is durable may launch proceed. ``authorize_work`` then publishes a
run-specific S3 start permit; the guest must wait for it before training. A local
process cannot itself guarantee a spending cap if it or the host disappears.

All calls here are injected. Tests use an in-memory fake and never contact AWS.
Reservations remain charged conservatively after termination; explicit local
reconciliation is required before another job can reserve budget. An attempt
that never requested launch can instead be closed by authenticated absence
verification, also retaining its full reservation and history.
"""

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

if __package__:
    from scripts.fpga_lm_aws import PROJECT_TAG, plan, validate_unlaunched_record
else:
    from fpga_lm_aws import PROJECT_TAG, plan, validate_unlaunched_record


RESERVATION = Decimal("10.00")
SPEND_LIMIT = Decimal("900.00")
TERMINAL = "terminated_reconciled"
CANCELLED = "cancelled_before_launch"
TERMINAL_STATUSES = {TERMINAL, CANCELLED}


class LifecycleError(RuntimeError):
    """A required lifecycle fact could not be established safely."""


def utc_now():
    return datetime.now(timezone.utc)


def iso(moment):
    if moment.tzinfo is None:
        raise ValueError("lifecycle timestamps require timezone-aware UTC")
    return moment.astimezone(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def validate_amount(value):
    if isinstance(value, bool):
        raise ValueError("cost must be a nonnegative finite decimal")
    try:
        amount = Decimal(str(value))
    except ArithmeticError as error:
        raise ValueError("cost must be a nonnegative finite decimal") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("cost must be a nonnegative finite decimal")
    return amount


class Ledger(AbstractContextManager):
    """One exclusive local writer, atomic replacement, fsynced file and directory.

    Keep this context open throughout supervision. Historical run records and
    their event history are retained; a used run identity cannot be replaced.
    This is a local single-host lock, not a distributed/account-wide AWS lock.
    """

    def __init__(self, directory, account, now=utc_now):
        if re.fullmatch(r"[0-9]{12}", account) is None:
            raise ValueError("explicit 12-digit account is required")
        self.directory = Path(directory)
        self.account, self.now = account, now
        self.path = self.directory / "ledger.json"
        self.lock = None
        self.data = None

    def __enter__(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise LifecycleError("state directory must not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.directory / ".controller.lock", flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise LifecycleError("another controller holds the local ledger lock") from None
        self.lock = fd
        try:
            if self.path.exists() or self.path.is_symlink():
                read_fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(read_fd, "r") as stream:
                    self.data = json.load(stream)
                self._validate()
            else:
                self.data = {"schema": "tikitorch.fpga_lm_control.v1", "account": self.account,
                             "currency": "USD", "ceiling_usd": "1000", "contingency_usd": "100",
                             "generation": 0, "runs": []}
                self._save()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        if self.lock is not None:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            os.close(self.lock)
            self.lock = None
        return False

    def _validate(self):
        if (self.data.get("schema") != "tikitorch.fpga_lm_control.v1"
                or self.data.get("account") != self.account
                or self.data.get("currency") != "USD"
                or self.data.get("ceiling_usd") != "1000"
                or self.data.get("contingency_usd") != "100"
                or type(self.data.get("generation")) is not int
                or not isinstance(self.data.get("runs"), list)):
            raise LifecycleError("ledger schema/account/budget identity mismatch")
        seen = set()
        for record in self.data["runs"]:
            prefix = record.get("run_prefix")
            if not isinstance(prefix, str) or prefix in seen:
                raise LifecycleError("ledger contains duplicate/invalid run identities")
            seen.add(prefix)
            if validate_amount(record.get("reserved_usd")) != RESERVATION:
                raise LifecycleError("each pilot must preserve its full $10 reservation")
            validate_amount(record.get("billed_usd", "0"))
            if record.get("account") != self.account or not isinstance(record.get("events"), list):
                raise LifecycleError("run ownership/event history mismatch")
            if (record.get("request_sha256") != hashlib.sha256(encoded(record.get("request"))).hexdigest()
                    or record.get("request", {}).get("ClientToken") != record.get("client_token")):
                raise LifecycleError("immutable launch request/token digest mismatch")
            if record.get("status") == TERMINAL and not record.get("termination_verified_at"):
                raise LifecycleError("reconciled run lacks confirmed termination")
            if record.get("status") == CANCELLED:
                validate_unlaunched_record(record, require_absence=True)

    def _save(self):
        if self.lock is None:
            raise LifecycleError("ledger mutation requires its exclusive lock")
        self._validate()
        self.data["generation"] += 1
        payload = encoded(self.data)
        fd, temporary = tempfile.mkstemp(prefix=".ledger-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, prefix):
        matches = [record for record in self.data["runs"] if record["run_prefix"] == prefix]
        if len(matches) != 1:
            raise LifecycleError("run identity is not uniquely recorded")
        return matches[0]

    def event(self, prefix, status, **facts):
        record = self.get(prefix)
        if record["status"] in TERMINAL_STATUSES:
            raise LifecycleError("historical terminal runs cannot be mutated")
        protected = {"account", "region", "run_prefix", "client_token", "request", "request_sha256",
                     "reserved_usd", "deadline_utc", "created_utc", "events", "max_minutes",
                     "source_sha256", "watchdog_execution_arn", "user_data_encoding"}
        if protected.intersection(facts):
            raise LifecycleError("immutable launch/reservation identity cannot be changed")
        record.update(facts)
        record["status"] = status
        record["events"].append({"status": status, "at": iso(self.now()), **facts})
        self._save()

    def reserve(self, config):
        """Fsync the full reservation AND immutable launch intent before any AWS write."""
        if any(record["status"] not in TERMINAL_STATUSES for record in self.data["runs"]):
            raise LifecycleError("an active or unreconciled run blocks a new reservation")
        proposal = plan(config, self.data)
        request = proposal["proposed_run_instances"]
        user_data = config.get("guest_user_data")
        if not isinstance(user_data, str) or not user_data.startswith("#!/bin/bash\n"):
            raise ValueError("control requires the exact reviewed guest_user_data bash script")
        if len(user_data.encode()) > 256 * 1024:
            raise ValueError("guest user data exceeds the reviewed 256 KiB raw limit")
        if len(gzip.compress(user_data.encode(), mtime=0)) > 16384:
            raise ValueError("gzip guest user data exceeds EC2's 16 KiB decoded limit")
        request["UserData"] = user_data
        execution_arn = config.get("watchdog_execution_arn", "")
        if re.fullmatch(rf"arn:aws:states:{config['region']}:{self.account}:execution:[A-Za-z0-9_-]+:[A-Za-z0-9_-]+", execution_arn) is None:
            raise ValueError("exact account/region-bound watchdog_execution_arn is required")
        prefix = config["run_prefix"]
        if any(record["run_prefix"] == prefix for record in self.data["runs"]):
            raise LifecycleError("historical run identity cannot be reused")
        accounted = sum(max(validate_amount(row["reserved_usd"]),
                            validate_amount(row.get("billed_usd", "0"))) for row in self.data["runs"])
        if accounted + RESERVATION > SPEND_LIMIT:
            raise LifecycleError("reservation crosses $900; the $100 contingency is not available")
        created = self.now()
        requested_deadline = config.get("deadline_utc")
        if requested_deadline is None:
            deadline_text = iso(created + timedelta(minutes=config["max_minutes"]))
        else:
            try:
                parsed_deadline = datetime.fromisoformat(requested_deadline.replace("Z", "+00:00"))
            except (AttributeError, ValueError) as error:
                raise ValueError("deadline_utc must be an explicit UTC ISO timestamp") from error
            if (parsed_deadline.tzinfo is None or parsed_deadline.utcoffset() != timedelta(0)
                    or not created < parsed_deadline <= created + timedelta(minutes=config["max_minutes"])):
                raise ValueError("deadline_utc must be future UTC within the reviewed max_minutes")
            deadline_text = requested_deadline
        record = {
            "account": self.account, "region": config["region"], "run_prefix": prefix,
            "client_token": request["ClientToken"], "request": request,
            "request_sha256": hashlib.sha256(encoded(request)).hexdigest(),
            "source_sha256": config["source_sha256"], "reserved_usd": "10.00", "billed_usd": "0",
            "watchdog_execution_arn": execution_arn,
            "user_data_encoding": "gzip",
            "created_utc": iso(created), "deadline_utc": deadline_text,
            "max_minutes": config["max_minutes"], "status": "launch_intent",
            "events": [{"status": "launch_intent", "at": iso(created)}],
        }
        self.data["runs"].append(record)
        self._save()
        return record

    def reconcile(self, prefix, billed_usd, note):
        """Explicit local acknowledgement; never lowers the conservative reservation."""
        record = self.get(prefix)
        if record.get("status") != "terminated_unreconciled" or not record.get("termination_verified_at"):
            raise LifecycleError("only a confirmed terminated run can be reconciled")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("an explicit reconciliation note is required")
        billed = max(validate_amount(record.get("billed_usd", "0")), validate_amount(billed_usd))
        self.event(prefix, TERMINAL, billed_usd=str(billed), reconciliation_note=note)


def identity_payload(record):
    return {key: record[key] for key in ("account", "region", "run_prefix", "client_token")}


def check_identity(invoke, record):
    observed = invoke("identity", identity_payload(record))
    if observed.get("account") != record["account"] or observed.get("region") != record["region"]:
        raise LifecycleError("AWS adapter account/region does not match the durable run")


def owned_instance(instance, record, expected_id=None):
    """Require independent observed identity, not a name-only ownership guess."""
    required = {"Project": PROJECT_TAG, "Run": record["run_prefix"].split("/")[-1],
                "OwnerAccount": record["account"]}
    instance_id = instance.get("instance_id", "")
    if re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id) is None:
        raise LifecycleError("invalid observed instance identity")
    if expected_id is not None and instance_id != expected_id:
        raise LifecycleError("observed instance does not match the exact recorded instance")
    for key in ("account", "region", "client_token"):
        if instance.get(key) != record[key]:
            raise LifecycleError(f"instance {key} ownership mismatch")
    if any(instance.get("tags", {}).get(key) != value for key, value in required.items()):
        raise LifecycleError("instance run/owner tags do not match the durable intent")
    return instance_id


def discover(invoke, record):
    check_identity(invoke, record)
    matches = invoke("find_by_client_token", identity_payload(record))
    if not isinstance(matches, list) or len(matches) > 1:
        raise LifecycleError("launch reconciliation did not resolve exactly one owned instance")
    if not matches:
        return None
    owned_instance(matches[0], record, record.get("instance_id"))
    return matches[0]


def verify_watchdog(receipt, record):
    """Require the expected prearmed execution, successful health check and initial wait."""
    if (receipt.get("verified") is not True
            or receipt.get("execution_arn") != record["watchdog_execution_arn"]
            or receipt.get("account") != record["account"] or receipt.get("region") != record["region"]
            or receipt.get("client_token") != record["client_token"]
            or receipt.get("run_prefix") != record["run_prefix"]
            or receipt.get("deadline_utc") != record["deadline_utc"]
            or receipt.get("startup_health_check_passed") is not True
            or receipt.get("status") != "RUNNING" or receipt.get("state") != "WaitUntilDeadline"):
        raise LifecycleError("prearmed watchdog identity/health/deadline could not be verified")
    return receipt


def terminate_owned(ledger, record, invoke, *, monotonic=time.monotonic, sleep=time.sleep,
                    wait_seconds=60, poll_seconds=2):
    """Never remove state or release money on an unconfirmed termination."""
    if not math.isfinite(wait_seconds) or not 0 <= wait_seconds <= 60:
        raise ValueError("termination confirmation wait must be 0..60 seconds")
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("poll interval must be positive and no more than 60 seconds")
    prefix = record["run_prefix"]
    until = monotonic() + wait_seconds
    try:
        check_identity(invoke, record)
        if not record.get("instance_id"):
            found = discover(invoke, record)
            if found is None:
                ledger.event(prefix, "launch_ambiguous", termination_confirmed=False)
                return False
            ledger.event(prefix, "instance_reconciled", instance_id=found["instance_id"])
        payload = {**identity_payload(record), "instance_id": record["instance_id"]}
        observed = invoke("describe_instance", payload)
        owned_instance(observed, record, record["instance_id"])
        if observed.get("state") != "terminated":
            ledger.event(prefix, "termination_requested")
            invoke("terminate_instance", payload)
        while True:
            observed = invoke("describe_instance", payload)
            owned_instance(observed, record, record["instance_id"])
            if observed.get("state") == "terminated":
                ledger.event(prefix, "terminated_unreconciled", termination_confirmed=True,
                             termination_verified_at=iso(ledger.now()))
                return True
            if monotonic() >= until:
                break
            sleep(min(poll_seconds, max(0, until - monotonic())))
        ledger.event(prefix, "termination_unconfirmed", termination_confirmed=False)
        return False
    except BaseException as error:
        ledger.event(prefix, "termination_unconfirmed", termination_confirmed=False,
                     termination_error_type=type(error).__name__)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return False


def supervise(config, state_directory, invoke, *, now=utc_now, monotonic=time.monotonic,
              sleep=time.sleep, poll_seconds=5, termination_wait_seconds=60):
    """Supervise one proposed pilot through injected adapter calls only.

    Mutation order: fsynced intent -> prearmed verified independent watchdog ->
    durable receipt -> fresh watchdog health -> launch once -> reconcile/verify
    -> S3 guest start permit -> poll.
    Every path after an attempted launch finally requests exact-owned-instance
    termination. An ambiguous launch is never blindly retried. On success a
    fresh, validated artifact retrieval is required; termination still leaves a
    $10 reservation and blocks new work until explicit reconciliation.
    """
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("poll interval must be positive and no more than 60 seconds")
    if not math.isfinite(termination_wait_seconds) or not 0 <= termination_wait_seconds <= 60:
        raise ValueError("termination confirmation wait must be 0..60 seconds")
    with Ledger(state_directory, config["account"], now=now) as ledger:
        record = ledger.reserve(config)
        prefix = record["run_prefix"]
        launch_attempted = False
        result = None
        began = monotonic()
        absolute_deadline = datetime.fromisoformat(record["deadline_utc"].replace("Z", "+00:00"))
        remaining = (absolute_deadline - now()).total_seconds()
        deadline = began + max(0, remaining)
        try:
            check_identity(invoke, record)
            existing = discover(invoke, record)
            if existing is not None:
                raise LifecycleError("run identity already exists in AWS; recover it explicitly")
            watchdog_payload = {**identity_payload(record), "deadline_utc": record["deadline_utc"],
                                "started_at_utc": record["created_utc"],
                                "execution_arn": record["watchdog_execution_arn"]}
            receipt = invoke("arm_watchdog", watchdog_payload)
            verify_watchdog(receipt, record)
            ledger.event(prefix, "watchdog_verified", watchdog=receipt)
            # Recheck immediately before the EC2 write, not only when StartExecution returned.
            verify_watchdog(invoke("watchdog_health", watchdog_payload), record)
            if monotonic() >= deadline or now() >= absolute_deadline:
                raise TimeoutError("pilot deadline elapsed before instance launch")
            launch_attempted = True
            ledger.event(prefix, "launch_requested")
            try:
                # Transport adapters may encode UserData; never hand them mutable ledger storage.
                request_copy = json.loads(json.dumps(record["request"], allow_nan=False))
                launched = invoke("launch_instance", {**identity_payload(record), "request": request_copy})
            except Exception as error:
                ledger.event(prefix, "launch_ambiguous", launch_error_type=type(error).__name__)
                launched = discover(invoke, record)
                if launched is None:
                    raise LifecycleError("launch outcome ambiguous; intent retained, no blind retry") from error
            instance_id = owned_instance(launched, record)
            ledger.event(prefix, "instance_reconciled", instance_id=instance_id)
            # Query after launch; launch responses alone are insufficient ownership evidence.
            payload = {**identity_payload(record), "instance_id": instance_id}
            owned_instance(invoke("describe_instance", payload), record, instance_id)
            permit = {**payload, "source_sha256": config["source_sha256"],
                      "deadline_utc": record["deadline_utc"], "watchdog_execution_arn": receipt["execution_arn"]}
            acknowledged = invoke("authorize_work", permit)
            if acknowledged != {**permit, "authorized": True}:
                raise LifecycleError("guest start permit was not independently confirmed")
            ledger.event(prefix, "running", guest_authorized=True)
            while monotonic() < deadline:
                status = invoke("poll_result", payload)
                if status.get("status") == "success":
                    retrieval = invoke("retrieve_results", payload)
                    if (retrieval.get("fresh") is not True or retrieval.get("run_prefix") != prefix
                            or retrieval.get("artifact_checks_passed") is not True
                            or not retrieval.get("directory")):
                        raise LifecycleError("fresh complete artifact retrieval was not verified")
                    ledger.event(prefix, "workload_complete", retrieval=retrieval)
                    result = retrieval
                    break
                if status.get("status") != "running":
                    raise LifecycleError("guest workload failed or returned an invalid terminal state")
                sleep(min(poll_seconds, max(0, deadline - monotonic())))
            else:
                raise TimeoutError("controller monotonic deadline reached")
        except BaseException as error:
            ledger.event(prefix, "controller_failed", controller_error_type=type(error).__name__)
            raise
        finally:
            if launch_attempted:
                confirmed = terminate_owned(ledger, record, invoke, monotonic=monotonic, sleep=sleep,
                                            wait_seconds=termination_wait_seconds, poll_seconds=poll_seconds)
                if not confirmed and result is not None:
                    raise LifecycleError("work completed but instance termination remains unconfirmed")
        return {"retrieval": result, "termination_confirmed": True,
                "reservation_usd": record["reserved_usd"], "status": record["status"]}


def recover(state_directory, account, run_prefix, invoke, **options):
    """Recover by reconciliation and termination only; never relaunch or authorize work."""
    with Ledger(state_directory, account) as ledger:
        record = ledger.get(run_prefix)
        if record["status"] in TERMINAL_STATUSES:
            raise LifecycleError("historical reconciled runs cannot be recovered/mutated")
        return terminate_owned(ledger, record, invoke, **options)


def cancel_before_launch(state_directory, account, run_prefix, invoke, *, now=utc_now):
    """Close only a provably unattempted launch; retain its full reservation.

    This uses read-only authenticated discovery, never termination or an AWS
    mutation. Absence alone is insufficient: the fsynced event history must also
    prove that RunInstances was never requested or became ambiguous.
    """
    with Ledger(state_directory, account, now=now) as ledger:
        record = ledger.get(run_prefix)
        if record["status"] in TERMINAL_STATUSES:
            raise LifecycleError("historical terminal runs cannot be cancelled/mutated")
        validate_unlaunched_record(record)
        if discover(invoke, record) is not None:
            raise LifecycleError("a matching instance exists; cancellation cannot replace recovery")
        absence = {**identity_payload(record), "matches": []}
        stamp = iso(now())
        ledger.event(run_prefix, CANCELLED, absence_verified_at=stamp, absence_evidence=absence)
        return {"status": CANCELLED, "run_prefix": run_prefix, "absence_verified_at": stamp,
                "reservation_usd": record["reserved_usd"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("status", "launch"), default="status")
    arguments = parser.parse_args(argv)
    if arguments.command == "launch":
        parser.error("launch disabled: live adapter, guest permit gate and prearmed watchdog require integration review")
    print(json.dumps({"launch_enabled": False, "live_adapter_module": "scripts.fpga_lm_cloud",
                      "requires_explicit_execute": True,
                      "requires": ["verified prearmed Step Functions termination watchdog", "guest S3 start gate",
                                   "reviewed injected AWS adapter", "integration lifecycle validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
