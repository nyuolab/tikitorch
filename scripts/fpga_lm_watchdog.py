#!/usr/bin/env python3
"""Build an offline, pre-armed Standard Step Functions EC2 watchdog.

This module never contacts AWS, launches instances, or writes IAM policies. A
separately authorized adapter must validate/deploy the artifacts, start the
execution, and observe StartupHealthCheck succeeding and WaitUntilDeadline
being entered while the execution is RUNNING BEFORE calling RunInstances.
Read back the exact definition, execution role, input and account/region too;
neither successful deployment nor an echoed receipt proves the guard is armed.

The immutable definition, not execution input, binds ownership and deadlines.
Only an explicit observed ``terminated`` state counts as cleanup success.
Empty discovery, NotFound, and exhausted retries remain failures. This is an
operational fail-safe, not an AWS spending cap or protection from service outage.
Each discovery scan follows at most ten pages and validates a single candidate
before acting. A partial scan never proves absence or authorizes termination.

Official contracts used:
https://docs.aws.amazon.com/step-functions/latest/dg/supported-services-awssdk.html
https://docs.aws.amazon.com/step-functions/latest/dg/intrinsic-functions.html
https://docs.aws.amazon.com/step-functions/latest/dg/concepts-error-handling.html
https://docs.aws.amazon.com/ec2/latest/devguide/eventual-consistency.html
https://docs.aws.amazon.com/ec2/latest/devguide/ec2-api-pagination.html
https://docs.aws.amazon.com/service-authorization/latest/reference/list_ec2.html
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re


PROJECT_TAG = "TikiTorchFPGALM"
INSTANCE_TYPE = "f2.6xlarge"
STARTUP_STATE = "StartupHealthCheck"
WAIT_STATE = "WaitUntilDeadline"
MAX_SCAN_PAGES = 10
EC2 = "arn:aws:states:::aws-sdk:ec2:"


def _text(config, key, pattern):
    value = config.get(key)
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"invalid or missing {key}")
    return value


def _integer(config, key, default, low, high):
    value = config.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return value


def _time(config, key):
    value = _text(config, key, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid {key}") from error
    return parsed.replace(tzinfo=timezone.utc)


def _iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _choice(choices, default):
    return {"Type": "Choice", "Choices": choices, "Default": default}


def _test(variable, operator, value):
    return {"Variable": variable, operator: value}


def _task(action, parameters, next_state, failure_state, result_path):
    return {
        "Type": "Task", "Resource": EC2 + action, "Parameters": parameters,
        "TimeoutSeconds": 10, "ResultPath": result_path, "Next": next_state,
        "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"],
                   "IntervalSeconds": 2, "BackoffRate": 2, "MaxAttempts": 2}],
        "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.last_error",
                   "Next": failure_state}],
    }


def build_artifacts(config):
    """Return deterministic ASL/IAM JSON objects; no SDK, network or file writes.

    Required: account, region, run_prefix, client_token, started_at_utc,
    deadline_utc. The token must match the existing pilot planner's SHA256.
    Optional bounds: grace_seconds (7200), max_attempts (120), poll_seconds (60).
    started_at_utc must be the proposed execution start, at most 90 minutes
    before the deadline. The live adapter must reject elapsed launch deadlines.
    """
    if not isinstance(config, dict):
        raise ValueError("watchdog config must be an object")
    account = _text(config, "account", r"[0-9]{12}")
    region = _text(config, "region", r"us-(?:east-1|west-2)")
    prefix = _text(config, "run_prefix", r"tikitorch-fpga-lm/[a-z0-9][a-z0-9-]{7,79}")
    token = _text(config, "client_token", r"[0-9a-f]{64}")
    expected_token = hashlib.sha256(f"{account}:{region}:{prefix}".encode()).hexdigest()
    if token != expected_token:
        raise ValueError("client_token does not match the exact account/region/run_prefix")
    if config.get("instance_type", INSTANCE_TYPE) != INSTANCE_TYPE:
        raise ValueError("watchdog supports only f2.6xlarge")
    if config.get("project", PROJECT_TAG) != PROJECT_TAG:
        raise ValueError("watchdog project must be TikiTorchFPGALM")
    started, deadline = _time(config, "started_at_utc"), _time(config, "deadline_utc")
    launch_window = (deadline - started).total_seconds()
    if not 0 < launch_window <= 90 * 60:
        raise ValueError("deadline must follow started_at_utc by at most 90 minutes")
    grace = _integer(config, "grace_seconds", 7200, 1, 7200)
    attempts = _integer(config, "max_attempts", 120, 1, 120)
    poll = _integer(config, "poll_seconds", 60, 1, 60)
    cleanup_deadline = _iso(deadline + timedelta(seconds=grace))
    run = prefix.split("/")[-1]
    name = "tt-fpga-lm-wd-" + token[:32]
    state_machine_arn = f"arn:aws:states:{region}:{account}:stateMachine:{name}"
    role_arn = f"arn:aws:iam::{account}:role/{name}-role"
    execution_name = "run-" + token[:32]
    tags = {"Project": PROJECT_TAG, "Run": run, "OwnerAccount": account}
    expected_tags = {key: {"Key": key, "Value": value} for key, value in tags.items()}
    filters = [{"Name": f"tag:{key}", "Values": [value]} for key, value in tags.items()]
    filters += [{"Name": "client-token", "Values": [token]},
                {"Name": "instance-type", "Values": [INSTANCE_TYPE]}]
    discovery = {"Filters": filters, "MaxResults": 1000}
    instance = "$.found.Instances[0]"

    # Input is discarded, so StartExecution input cannot replace bound identity,
    # counters, expected tags, an instance ID, or the absolute deadline.
    states = {
        "Initialize": {"Type": "Pass", "Result": {
            "attempt": {"value": 0}, "phase": "startup", "locked": {"InstanceId": ""},
            "expected_tags": expected_tags,
        }, "Next": "InitializeScan"},
        "InitializeScan": {"Type": "Pass", "Result": {
            "page": 1, "tokens": [], "candidate": {"Reservations": []},
        }, "ResultPath": "$.scan", "Next": "ChooseScanStart"},
        "ChooseScanStart": _choice([
            {**_test("$.phase", "StringEquals", "startup"), "Next": STARTUP_STATE},
        ], "DiscoverOwnedInstance"),
        STARTUP_STATE: {
            **_task("describeInstances", discovery, "FlattenInstances", "StartupFailed", "$.probe"),
            # A just-created role may not yet be assumable. TaskFailed also
            # matches States.Permissions; Timeout shares the same retry budget.
            # Four retries wait 5+10+20+20=55 seconds. Five calls at <=10 seconds
            # add at most 50 seconds: 105 total, leaving 15 seconds inside the
            # adapter's 120-second arming window for overhead and further pages.
            # That outer deadline still fails closed if AWS takes longer.
            "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"],
                       "IntervalSeconds": 5, "BackoffRate": 2, "MaxAttempts": 4,
                       "MaxDelaySeconds": 20}],
        },
        "FlattenInstances": {"Type": "Pass", "Parameters": {
            "Instances.$": "$.probe.Reservations[*].Instances[*]",
        }, "ResultPath": "$.found", "Next": "CountInstances"},
        "CountInstances": {"Type": "Pass", "Parameters": {
            "value.$": "States.ArrayLength($.found.Instances)",
        }, "ResultPath": "$.count", "Next": "CheckPageMatches"},
        "CheckPageMatches": _choice([
            {**_test("$.count.value", "NumericEquals", 0), "Next": "CheckPagination"},
            {**_test("$.count.value", "NumericEquals", 1), "Next": "RequireNoEarlierCandidate"},
        ], "AmbiguousDiscovery"),
        "RequireNoEarlierCandidate": _choice([
            {**_test("$.scan.candidate.Reservations[0]", "IsPresent", False),
             "Next": "CheckOwnershipFields"},
        ], "AmbiguousDiscovery"),
        # Keep only one fully validated candidate, not an unbounded page list.
        # Even a candidate on page one cannot be acted on before the scan ends.
        "SaveCandidate": {"Type": "Pass", "Parameters": {
            "page.$": "$.scan.page", "tokens.$": "$.scan.tokens", "candidate.$": "$.probe",
        }, "ResultPath": "$.scan", "Next": "CheckPagination"},
        "CheckPagination": _choice([
            {**_test("$.probe.NextToken", "IsPresent", False), "Next": "CompleteScan"},
            {**_test("$.probe.NextToken", "IsNull", True), "Next": "CompleteScan"},
            {"And": [_test("$.probe.NextToken", "IsString", True),
                     {"Not": _test("$.probe.NextToken", "StringEquals", "")}],
             "Next": "CheckSeenToken"},
        ], "PaginationFailed"),
        "CheckSeenToken": {"Type": "Pass", "Parameters": {
            "repeated.$": "States.ArrayContains($.scan.tokens, $.probe.NextToken)",
        }, "ResultPath": "$.pagination", "Next": "CheckPageBounds"},
        "CheckPageBounds": _choice([
            {"And": [_test("$.pagination.repeated", "BooleanEquals", False),
                     _test("$.scan.page", "NumericLessThan", MAX_SCAN_PAGES)],
             "Next": "AppendPageToken"},
        ], "PaginationFailed"),
        "AppendPageToken": {"Type": "Pass", "Parameters": {
            "groups.$": "States.Array($.scan.tokens, States.Array($.probe.NextToken))",
        }, "ResultPath": "$.pending_tokens", "Next": "AdvancePage"},
        "AdvancePage": {"Type": "Pass", "Parameters": {
            "page.$": "States.MathAdd($.scan.page, 1)",
            "tokens.$": "$.pending_tokens.groups[*][*]", "candidate.$": "$.scan.candidate",
        }, "ResultPath": "$.scan", "Next": "CheckPageDeadline"},
        "CheckPageDeadline": _choice([
            # The adapter's independent 120-second arming deadline still blocks
            # launch if startup pagination is slow; no receipt bypass is added.
            {**_test("$.phase", "StringEquals", "startup"), "Next": "DescribeNextPage"},
            {**_test("$$.State.EnteredTime", "TimestampLessThan", cleanup_deadline),
             "Next": "DescribeNextPage"},
        ], "CleanupUnconfirmed"),
        "DescribeNextPage": _task("describeInstances", {
            **discovery, "NextToken.$": "$.probe.NextToken",
        }, "FlattenInstances", "PaginationFailed", "$.probe"),
        "PaginationFailed": _choice([
            {**_test("$.phase", "StringEquals", "startup"), "Next": "StartupFailed"},
        ], "RetryWait"),
        "CompleteScan": {"Type": "Pass", "Parameters": {
            "Reservations.$": "$.scan.candidate.Reservations",
        }, "ResultPath": "$.probe", "Next": "FlattenCompletedScan"},
        "FlattenCompletedScan": {"Type": "Pass", "Parameters": {
            "Instances.$": "$.probe.Reservations[*].Instances[*]",
        }, "ResultPath": "$.found", "Next": "CountCompletedScan"},
        "CountCompletedScan": {"Type": "Pass", "Parameters": {
            "value.$": "States.ArrayLength($.found.Instances)",
        }, "ResultPath": "$.count", "Next": "RoutePhase"},
        "RoutePhase": _choice([
            {**_test("$.phase", "StringEquals", "startup"), "Next": "StartupRunMustBeAbsent"},
        ], "RequireSingleInstance"),
        "StartupRunMustBeAbsent": _choice([
            {**_test("$.count.value", "NumericEquals", 0), "Next": WAIT_STATE},
        ], "ExistingRun"),
        WAIT_STATE: {"Type": "Wait", "Timestamp": _iso(deadline), "Next": "BeginCleanup"},
        "BeginCleanup": {"Type": "Pass", "Result": "cleanup", "ResultPath": "$.phase",
                         "Next": "CheckCleanupBounds"},
        "CheckCleanupBounds": _choice([
            {"And": [_test("$$.State.EnteredTime", "TimestampLessThan", cleanup_deadline),
                     _test("$.attempt.value", "NumericLessThan", attempts)],
             "Next": "CountAttempt"},
        ], "CleanupUnconfirmed"),
        "CountAttempt": {"Type": "Pass", "Parameters": {
            "value.$": "States.MathAdd($.attempt.value, 1)",
        }, "ResultPath": "$.attempt", "Next": "InitializeScan"},
        "DiscoverOwnedInstance": _task("describeInstances", discovery, "FlattenInstances",
                                        "RetryWait", "$.probe"),
        "RequireSingleInstance": _choice([
            {**_test("$.count.value", "NumericEquals", 0), "Next": "RetryWait"},
            {**_test("$.count.value", "NumericEquals", 1), "Next": "RequireExactLockedId"},
        ], "AmbiguousDiscovery"),
        "CheckOwnershipFields": _choice([
            {"And": [
                _test("$.probe.Reservations[0].OwnerId", "StringEquals", account),
                {"Not": _test("$.probe.Reservations[1]", "IsPresent", True)},
                _test(instance + ".ClientToken", "StringEquals", token),
                _test(instance + ".InstanceType", "StringEquals", INSTANCE_TYPE),
                _test(instance + ".InstanceId", "StringMatches", "i-*"),
                _test(instance + ".Tags", "IsPresent", True),
            ], "Next": "CheckOwnershipTags"},
        ], "OwnershipMismatch"),
        "CheckOwnershipTags": {"Type": "Pass", "Parameters": {
            key + ".$": f"States.ArrayContains({instance}.Tags, $.expected_tags.{key})"
            for key in tags
        }, "ResultPath": "$.tag_match", "Next": "RequireOwnershipTags"},
        "RequireOwnershipTags": _choice([
            {"And": [_test(f"$.tag_match.{key}", "BooleanEquals", True) for key in tags],
             "Next": "SaveCandidate"},
        ], "OwnershipMismatch"),
        "RequireExactLockedId": _choice([
            {"Or": [_test("$.locked.InstanceId", "StringEquals", ""),
                    _test("$.locked.InstanceId", "StringEqualsPath", instance + ".InstanceId")],
             "Next": "LockExactInstanceId"},
        ], "OwnershipMismatch"),
        "LockExactInstanceId": {"Type": "Pass", "Parameters": {
            "InstanceId.$": instance + ".InstanceId",
        }, "ResultPath": "$.locked", "Next": "CheckInstanceState"},
        "CheckInstanceState": _choice([
            {**_test(instance + ".State.Name", "StringEquals", "terminated"),
             "Next": "ConfirmedTerminated"},
            {**_test(instance + ".State.Name", "StringEquals", "shutting-down"),
             "Next": "RetryWait"},
            {"Or": [_test(instance + ".State.Name", "StringEquals", value)
                    for value in ("pending", "running", "stopping", "stopped")],
             "Next": "CheckTerminationTime"},
        ], "OwnershipMismatch"),
        "CheckTerminationTime": _choice([
            {**_test("$$.State.EnteredTime", "TimestampLessThan", cleanup_deadline),
             "Next": "TerminateExactOwnedInstance"},
        ], "CleanupUnconfirmed"),
        "TerminateExactOwnedInstance": _task("terminateInstances", {
            "InstanceIds.$": "States.Array($.locked.InstanceId)",
        }, "RetryWait", "RetryWait", "$.termination_response"),
        "RetryWait": {"Type": "Wait", "Seconds": poll, "Next": "CheckCleanupBounds"},
        "ConfirmedTerminated": {"Type": "Succeed"},
    }
    for failure in ("StartupFailed", "ExistingRun", "AmbiguousDiscovery", "OwnershipMismatch",
                    "CleanupUnconfirmed"):
        states[failure] = {"Type": "Fail", "Error": "TikiTorchWatchdog." + failure,
                           "Cause": "No confirmed cleanup; retain budget reservation and investigate."}
    definition = {
        "Comment": "Pre-armed, unique-run EC2 cleanup; no launch authority; not a spending cap.",
        "StartAt": "Initialize", "TimeoutSeconds": math.ceil(launch_window) + grace + 60,
        "States": states,
    }
    permissions = {"Version": "2012-10-17", "Statement": [
        {"Sid": "ReadInstancesInOneRegion", "Effect": "Allow", "Action": "ec2:DescribeInstances",
         "Resource": "*", "Condition": {"StringEquals": {"ec2:Region": region}}},
        {"Sid": "TerminateOnlyThisRun", "Effect": "Allow", "Action": "ec2:TerminateInstances",
         "Resource": f"arn:aws:ec2:{region}:{account}:instance/*",
         "Condition": {"StringEquals": {
             **{f"ec2:ResourceTag/{key}": value for key, value in tags.items()},
             "ec2:InstanceType": INSTANCE_TYPE,
         }}},
    ]}
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "states.amazonaws.com"},
        "Action": "sts:AssumeRole", "Condition": {
            "StringEquals": {"aws:SourceAccount": account},
            "ArnEquals": {"aws:SourceArn": state_machine_arn},
        },
    }]}
    return {
        "schema_version": 1, "deployment_status": "offline_only", "type": "STANDARD",
        "account": account, "region": region, "run_prefix": prefix, "client_token": token,
        "deadline_utc": config["deadline_utc"], "cleanup_deadline_utc": cleanup_deadline,
        "started_at_utc": config["started_at_utc"], "state_machine_name": name,
        "state_machine_arn": state_machine_arn, "role_name": name + "-role", "role_arn": role_arn,
        "execution_name": execution_name,
        "execution_arn": f"arn:aws:states:{region}:{account}:execution:{name}:{execution_name}",
        "startup_health_state": STARTUP_STATE, "startup_wait_state": WAIT_STATE,
        "definition": definition, "permissions_policy": permissions, "trust_policy": trust,
        "definition_sha256": _sha(definition), "permissions_policy_sha256": _sha(permissions),
        "trust_policy_sha256": _sha(trust),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="explicit local JSON watchdog configuration")
    parser.add_argument("--emit", choices=("all", "definition", "permissions_policy", "trust_policy"),
                        default="all")
    args = parser.parse_args(argv)
    try:
        artifacts = build_artifacts(json.loads(args.config.read_text()))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(artifacts if args.emit == "all" else artifacts[args.emit], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
