"""Offline ASL control-flow and least-privilege watchdog regression tests.

The tiny interpreter exercises our emitted JSONPath subset against scripted EC2
responses. It is not AWS integration validation, an IAM simulator, or a claim
that deployment/termination has been exercised against a real account.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fnmatch
import hashlib
import json
import re
import subprocess
import sys

import pytest

from scripts import fpga_lm_watchdog as W


@pytest.fixture
def config():
    config = {
        "account": "123456789012", "region": "us-east-1",
        "run_prefix": "tikitorch-fpga-lm/20260909-unique-pilot",
        "started_at_utc": "2026-09-09T12:00:00Z", "deadline_utc": "2026-09-09T12:30:00Z",
        "max_attempts": 4, "poll_seconds": 30, "grace_seconds": 7200,
    }
    config["client_token"] = hashlib.sha256(
        f"{config['account']}:{config['region']}:{config['run_prefix']}".encode()
    ).hexdigest()
    return config


def response(config, state="running", instance_id="i-0123456789abcdef0", **overrides):
    instance = {
        "InstanceId": instance_id, "InstanceType": W.INSTANCE_TYPE,
        "ClientToken": config["client_token"], "State": {"Name": state},
        "Tags": [{"Key": "Run", "Value": config["run_prefix"].split("/")[-1]},
                 {"Key": "OwnerAccount", "Value": config["account"]},
                 {"Key": "Project", "Value": W.PROJECT_TAG}],
        **overrides,
    }
    return {"Reservations": [{"OwnerId": config["account"], "Instances": [instance]}]}


EMPTY = {"Reservations": []}


class MissingPath(Exception):
    pass


def select(path, data, now):
    if path == "$$.State.EnteredTime":
        return W._iso(now)
    if path == "$":
        return data
    values = [data]
    wildcard = False
    for field, index in re.findall(r"\.([A-Za-z_]\w*)|\[(\d+|\*)\]", path):
        try:
            if index == "*":
                values = [item for value in values for item in value]
                wildcard = True
            else:
                key = field if field else int(index)
                values = [value[key] for value in values]
        except (KeyError, IndexError, TypeError) as error:
            raise MissingPath(path) from error
    if wildcard:
        return values
    if not values:
        raise MissingPath(path)
    return values[0]


def split_args(text):
    depth, start, args = 0, 0, []
    for index, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    return args + [text[start:].strip()]


def expression(text, data, now):
    if text.startswith("$"):
        return deepcopy(select(text, data, now))
    match = re.fullmatch(r"States\.(\w+)\((.*)\)", text)
    if match:
        name, arguments = match.groups()
        values = [expression(value, data, now) for value in split_args(arguments)]
        operations = {
            "Array": lambda *args: list(args), "ArrayLength": len,
            "ArrayContains": lambda array, item: item in array,
            "MathAdd": lambda first, second: first + second,
        }
        return operations[name](*values)
    return json.loads(text)


def parameters(template, data, now):
    if isinstance(template, dict):
        return {
            key[:-2] if key.endswith(".$") else key:
                expression(value, data, now) if key.endswith(".$")
                else parameters(value, data, now)
            for key, value in template.items()
        }
    if isinstance(template, list):
        return [parameters(value, data, now) for value in template]
    return deepcopy(template)


def choice(rule, data, now):
    if "And" in rule:
        return all(choice(item, data, now) for item in rule["And"])
    if "Or" in rule:
        return any(choice(item, data, now) for item in rule["Or"])
    if "Not" in rule:
        return not choice(rule["Not"], data, now)
    try:
        value = select(rule["Variable"], data, now)
    except MissingPath:
        return rule.get("IsPresent") is False
    if "IsPresent" in rule:
        return rule["IsPresent"] is True
    if "IsNull" in rule:
        return (value is None) == rule["IsNull"]
    if "IsString" in rule:
        return isinstance(value, str) == rule["IsString"]
    operator = next(key for key in rule if key not in ("Variable", "Next"))
    expected = rule[operator]
    if operator.endswith("Path"):
        expected = select(expected, data, now)
        operator = operator[:-4]
    if operator in ("StringEquals", "NumericEquals", "BooleanEquals"):
        return value == expected
    if operator in ("NumericLessThan", "TimestampLessThan"):
        return value < expected
    if operator == "StringMatches":
        return fnmatch.fnmatchcase(value, expected)
    raise AssertionError(operator)


def retry_delays(state):
    """The emitted tasks have one retrier with a shared retry counter."""
    assert len(state["Retry"]) == 1
    retry = state["Retry"][0]
    return [min(retry["IntervalSeconds"] * retry["BackoffRate"] ** index,
                retry.get("MaxDelaySeconds", float("inf")))
            for index in range(retry["MaxAttempts"])]


def simulate(config, responses, *, termination_error=False, startup_error=False,
             execution_input=None, startup_response=EMPTY, startup_failures=(),
             startup_pages=(), requests=None):
    """Run the emitted states, including bounded SDK retries and failures."""
    states = W.build_artifacts(config)["definition"]["States"]
    name, data = "Initialize", deepcopy(execution_input or {})
    now = datetime.strptime(config["started_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    queue = iter(responses)
    startup_queue = iter(startup_pages)
    last = EMPTY
    visited, terminations = [], []
    for _ in range(10000):
        state = states[name]
        visited.append(name)
        if state["Type"] in ("Fail", "Succeed"):
            return name, terminations, visited
        if state["Type"] == "Choice":
            name = next((rule["Next"] for rule in state["Choices"] if choice(rule, data, now)),
                        state["Default"])
            continue
        if state["Type"] == "Wait":
            if "Seconds" in state:
                now += timedelta(seconds=state["Seconds"])
            else:
                now = max(now, datetime.strptime(state["Timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                          .replace(tzinfo=timezone.utc))
            name = state["Next"]
            continue
        try:
            if state["Type"] == "Pass":
                result = (parameters(state["Parameters"], data, now) if "Parameters" in state
                          else deepcopy(state["Result"]))
            elif state["Type"] == "Task":
                request = parameters(state["Parameters"], data, now)
                if requests is not None:
                    requests.append((name, deepcopy(request)))
                if name == W.STARTUP_STATE:
                    delays = retry_delays(state)
                    for attempt in range(len(delays) + 1):
                        error = ("States.TaskFailed" if startup_error else
                                 startup_failures[attempt] if attempt < len(startup_failures) else None)
                        result = RuntimeError(error) if error else startup_response
                        if not isinstance(result, Exception):
                            break
                        now += timedelta(seconds=state["TimeoutSeconds"])
                        retry_errors = state["Retry"][0]["ErrorEquals"]
                        matches = error in retry_errors or ("States.TaskFailed" in retry_errors
                                                            and error != "States.Timeout")
                        if not matches or attempt == len(delays):
                            break
                        now += timedelta(seconds=delays[attempt])
                elif name == "DescribeNextPage" and data["phase"] == "startup":
                    result = next(startup_queue, EMPTY)
                elif state["Resource"].endswith(":describeInstances"):
                    last = next(queue, last)
                    result = last
                else:
                    terminations.append(request)
                    result = RuntimeError("termination denied") if termination_error else {}
                if isinstance(result, Exception):
                    # Scripted errors persist for each retry of the same call.
                    if name != W.STARTUP_STATE:
                        delays = retry_delays(state)
                        now += timedelta(seconds=state["TimeoutSeconds"] * (len(delays) + 1)
                                         + sum(delays))
                    data["last_error"] = {"Error": str(result)}
                    name = state["Catch"][0]["Next"]
                    continue
            else:
                raise AssertionError(state)
            path = state.get("ResultPath", "$")
            if path == "$":
                data = deepcopy(result)
            elif path is not None:
                assert re.fullmatch(r"\$\.\w+", path)
                data[path[2:]] = deepcopy(result)
            name = state["Next"]
        except MissingPath:
            return "RuntimeFailure", terminations, visited
    raise AssertionError("watchdog failed to terminate within bounded offline simulation")


def test_definition_and_permissions_are_deterministic_and_narrow(config):
    artifacts = W.build_artifacts(config)
    assert artifacts == W.build_artifacts(config)
    assert artifacts["type"] == "STANDARD"
    assert artifacts["definition"]["TimeoutSeconds"] == 1800 + 7200 + 60
    assert artifacts["definition_sha256"] == W._sha(artifacts["definition"])
    trust = artifacts["trust_policy"]["Statement"][0]
    assert trust["Principal"] == {"Service": "states.amazonaws.com"}
    assert trust["Condition"]["ArnEquals"] == {"aws:SourceArn": artifacts["state_machine_arn"]}
    permissions = artifacts["permissions_policy"]["Statement"]
    assert {item["Action"] for item in permissions} == {"ec2:DescribeInstances", "ec2:TerminateInstances"}
    assert permissions[0]["Resource"] == "*"
    assert permissions[0]["Condition"] == {"StringEquals": {"ec2:Region": config["region"]}}
    assert permissions[1]["Resource"] == f"arn:aws:ec2:{config['region']}:{config['account']}:instance/*"
    conditions = permissions[1]["Condition"]["StringEquals"]
    assert conditions == {"ec2:ResourceTag/Project": W.PROJECT_TAG,
                          "ec2:ResourceTag/Run": config["run_prefix"].split("/")[-1],
                          "ec2:ResourceTag/OwnerAccount": config["account"],
                          "ec2:InstanceType": "f2.6xlarge"}
    text = json.dumps(artifacts)
    for forbidden in ("RunInstances", "runInstances", "CreateTags", "DeleteTags", "iam:*"):
        assert forbidden not in text


def test_discovery_filters_and_single_termination_task(config):
    assert W.MAX_SCAN_PAGES == 10
    states = W.build_artifacts(config)["definition"]["States"]
    tasks = {key: value for key, value in states.items() if value["Type"] == "Task"}
    assert set(tasks) == {W.STARTUP_STATE, "DiscoverOwnedInstance", "DescribeNextPage",
                          "TerminateExactOwnedInstance"}
    for key in (W.STARTUP_STATE, "DiscoverOwnedInstance", "DescribeNextPage"):
        filters = {item["Name"]: item["Values"] for item in tasks[key]["Parameters"]["Filters"]}
        assert filters == {"tag:Project": [W.PROJECT_TAG],
                           "tag:Run": [config["run_prefix"].split("/")[-1]],
                           "tag:OwnerAccount": [config["account"]],
                           "client-token": [config["client_token"]],
                           "instance-type": [W.INSTANCE_TYPE]}
        assert tasks[key]["Parameters"]["MaxResults"] == 1000
    assert tasks["DescribeNextPage"]["Parameters"]["NextToken.$"] == "$.probe.NextToken"
    terminate = tasks["TerminateExactOwnedInstance"]
    assert terminate["Parameters"] == {"InstanceIds.$": "States.Array($.locked.InstanceId)"}
    assert terminate["Next"] == terminate["Catch"][0]["Next"] == "RetryWait"
    assert all(task["TimeoutSeconds"] == 10 for task in tasks.values())


def test_healthy_start_waits_then_terminates_and_confirms(config):
    result, calls, states = simulate(config, [response(config), response(config, "shutting-down"),
                                            response(config, "terminated")])
    assert result == "ConfirmedTerminated"
    assert calls == [{"InstanceIds": ["i-0123456789abcdef0"]}]
    assert states.index(W.STARTUP_STATE) < states.index(W.WAIT_STATE) < states.index("DiscoverOwnedInstance")


def test_empty_discovery_and_not_found_are_never_success(config):
    for replies in ([EMPTY], [RuntimeError("Ec2.InvalidInstanceID.NotFoundException")]):
        result, calls, states = simulate(config, replies)
        assert result == "CleanupUnconfirmed"
        assert calls == []
        assert states.count("DiscoverOwnedInstance") == config["max_attempts"]


def test_transient_empty_discovery_is_reconciled(config):
    result, calls, _ = simulate(config, [EMPTY, response(config), response(config, "terminated")])
    assert result == "ConfirmedTerminated" and len(calls) == 1


def test_lost_instance_after_termination_request_is_unconfirmed(config):
    result, calls, _ = simulate(config, [response(config), EMPTY])
    assert result == "CleanupUnconfirmed" and len(calls) == 1


def test_failed_termination_exhausts_bounded_attempts(config):
    result, calls, _ = simulate(config, [response(config)], termination_error=True)
    assert result == "CleanupUnconfirmed"
    assert len(calls) == config["max_attempts"]


def test_startup_failure_never_reaches_launch_health_gate(config):
    result, calls, states = simulate(config, [], startup_error=True)
    assert result == "StartupFailed" and calls == [] and W.WAIT_STATE not in states


def test_role_propagation_retry_is_startup_only_and_fits_arming_window(config):
    artifacts = W.build_artifacts(config)
    states = artifacts["definition"]["States"]
    startup = states[W.STARTUP_STATE]
    assert startup["Retry"] == [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"],
                                 "IntervalSeconds": 5, "BackoffRate": 2, "MaxAttempts": 4,
                                 "MaxDelaySeconds": 20}]
    delays = retry_delays(startup)
    assert delays == [5, 10, 20, 20]
    bounded_seconds = sum(delays) + (len(delays) + 1) * startup["TimeoutSeconds"]
    assert bounded_seconds == 105 <= 120 - 15
    assert artifacts["definition"]["TimeoutSeconds"] == 1800 + 7200 + 60
    for name in ("DiscoverOwnedInstance", "DescribeNextPage", "TerminateExactOwnedInstance"):
        assert states[name]["Retry"] == [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"],
                                        "IntervalSeconds": 2, "BackoffRate": 2, "MaxAttempts": 2}]
    assert startup["Catch"] == [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.last_error",
                                 "Next": "StartupFailed"}]


@pytest.mark.parametrize("errors", [("States.TaskFailed",) * 4, ("States.Permissions",) * 4,
                                    ("States.Timeout",) * 4,
                                    ("States.TaskFailed", "States.Permissions", "States.Timeout", "States.TaskFailed")])
def test_transient_startup_failures_can_recover_only_after_a_healthy_empty_read(config, errors):
    result, calls, states = simulate(config, [response(config, "terminated")], startup_failures=errors)
    assert result == "ConfirmedTerminated" and calls == []
    assert states.index(W.STARTUP_STATE) < states.index(W.WAIT_STATE)
    result, calls, states = simulate(config, [], startup_failures=errors,
                                    startup_response=response(config))
    assert result == "ExistingRun" and calls == [] and W.WAIT_STATE not in states


@pytest.mark.parametrize("error", ["States.TaskFailed", "States.Permissions", "States.Timeout"])
def test_startup_fifth_failure_exhausts_shared_budget_without_arming(config, error):
    result, calls, states = simulate(config, [], startup_failures=(error,) * 5)
    assert result == "StartupFailed" and calls == [] and W.WAIT_STATE not in states


def test_existing_run_or_invalid_startup_token_never_arms(config):
    for reply in (response(config), {**EMPTY, "NextToken": ""}):
        result, calls, states = simulate(config, [], startup_response=reply)
        assert result in ("ExistingRun", "StartupFailed")
        assert calls == [] and W.WAIT_STATE not in states


@pytest.mark.parametrize("token", [None])
def test_empty_optional_pagination_token_does_not_invent_more_matches(config, token):
    result, calls, _ = simulate(config, [response(config, "terminated")],
                               startup_response={**EMPTY, "NextToken": token})
    assert result == "ConfirmedTerminated" and calls == []


def test_startup_empty_pages_are_followed_before_absence_is_accepted(config):
    requests = []
    result, calls, visited = simulate(config, [response(config, "terminated")],
        startup_response={**EMPTY, "NextToken": "first-page"},
        startup_pages=[{**EMPTY, "NextToken": "second-page"}, EMPTY], requests=requests)
    assert result == "ConfirmedTerminated" and calls == []
    assert visited[:visited.index(W.WAIT_STATE)].count("DescribeNextPage") == 2
    next_pages = [request for state, request in requests if state == "DescribeNextPage"]
    assert [request["NextToken"] for request in next_pages] == ["first-page", "second-page"]


def test_startup_instance_on_later_page_blocks_launch(config):
    result, calls, visited = simulate(config, [],
        startup_response={**EMPTY, "NextToken": "next"}, startup_pages=[response(config)])
    assert result == "ExistingRun" and calls == [] and W.WAIT_STATE not in visited


@pytest.mark.parametrize("terminal_state", ["running", "terminated"])
def test_cleanup_does_not_act_on_candidate_until_final_page(config, terminal_state):
    requests = []
    result, calls, visited = simulate(config,
        [{**response(config, terminal_state), "NextToken": "next"}, EMPTY,
         response(config, "terminated")], requests=requests)
    assert result == "ConfirmedTerminated"
    expected_calls = [] if terminal_state == "terminated" else [{"InstanceIds": ["i-0123456789abcdef0"]}]
    assert calls == expected_calls
    decision = "ConfirmedTerminated" if terminal_state == "terminated" else "TerminateExactOwnedInstance"
    assert visited.index("DescribeNextPage") < visited.index(decision)
    assert [request["NextToken"] for state, request in requests if state == "DescribeNextPage"] == ["next"]


def test_cleanup_follows_empty_page_to_owned_instance(config):
    requests = []
    result, calls, _ = simulate(config,
        [{**EMPTY, "NextToken": "next"}, response(config), response(config, "terminated")],
        requests=requests)
    assert result == "ConfirmedTerminated" and calls == [{"InstanceIds": ["i-0123456789abcdef0"]}]
    assert [request["NextToken"] for state, request in requests if state == "DescribeNextPage"] == ["next"]


@pytest.mark.parametrize("change", [{"InstanceType": "f2.12xlarge"}, {"Tags": []},
                                   {"ClientToken": "foreign"}])
def test_later_page_candidate_must_independently_pass_ownership(config, change):
    result, calls, _ = simulate(config,
        [{**EMPTY, "NextToken": "next"}, response(config, **change)])
    assert result == "OwnershipMismatch" and calls == []


@pytest.mark.parametrize("instance_id", ["i-0123456789abcdef0", "i-9876543210abcdef0"])
def test_second_match_on_later_page_is_rejected_before_any_termination(config, instance_id):
    result, calls, _ = simulate(config,
        [{**response(config), "NextToken": "next"}, response(config, instance_id=instance_id)])
    assert result == "AmbiguousDiscovery" and calls == []


@pytest.mark.parametrize("phase", ["startup", "cleanup"])
@pytest.mark.parametrize("tokens", [["same", "same"], ["first", "second", "first"],
                                    [""], [False], [1], [[]], [{}]])
def test_invalid_or_repeated_tokens_never_prove_absence_or_termination(config, phase, tokens):
    pages = [{**EMPTY, "NextToken": token} for token in tokens]
    if phase == "startup":
        result, calls, visited = simulate(config, [], startup_response=pages[0], startup_pages=pages[1:])
        assert result == "StartupFailed" and W.WAIT_STATE not in visited
    else:
        config["max_attempts"] = 1
        result, calls, visited = simulate(config, pages)
        assert result == "CleanupUnconfirmed" and "RetryWait" in visited
    assert calls == [] and "ConfirmedTerminated" not in visited


@pytest.mark.parametrize("phase", ["startup", "cleanup"])
def test_scan_is_limited_to_ten_pages_even_when_all_pages_are_empty(config, phase):
    pages = [{**EMPTY, "NextToken": f"page-{index}"} for index in range(W.MAX_SCAN_PAGES)]
    requests = []
    if phase == "startup":
        result, calls, visited = simulate(config, [], startup_response=pages[0], startup_pages=pages[1:], requests=requests)
        assert result == "StartupFailed" and W.WAIT_STATE not in visited
    else:
        config["max_attempts"] = 1
        result, calls, visited = simulate(config, pages, requests=requests)
        assert result == "CleanupUnconfirmed"
    assert calls == []
    assert sum(state == "DescribeNextPage" for state, _ in requests) == W.MAX_SCAN_PAGES - 1


def test_cleanup_scan_exhaustion_discards_candidate_and_retries_with_fresh_scan(config):
    pages = [{**response(config), "NextToken": "page-0"}]
    pages += [{**EMPTY, "NextToken": f"page-{index}"} for index in range(1, W.MAX_SCAN_PAGES)]
    result, calls, visited = simulate(config, [*pages, EMPTY, response(config, "terminated")])
    assert result == "ConfirmedTerminated" and calls == []
    assert visited.count("DiscoverOwnedInstance") == 3
    assert visited.count("SaveCandidate") == 2  # Exhausted candidate, then confirmed terminated candidate.


def test_cleanup_next_page_service_failure_retries_instead_of_permanently_disabling_reaper(config):
    result, calls, visited = simulate(config,
        [{**EMPTY, "NextToken": "next"}, RuntimeError("States.TaskFailed"),
         response(config), response(config, "terminated")])
    assert result == "ConfirmedTerminated" and calls == [{"InstanceIds": ["i-0123456789abcdef0"]}]
    assert "PaginationFailed" in visited and "StartupFailed" not in visited


def test_pagination_retries_remain_bounded_by_absolute_cleanup_deadline(config):
    config.update(grace_seconds=1, max_attempts=120)
    result, calls, visited = simulate(config,
        [{**EMPTY, "NextToken": "next"}, RuntimeError("States.Timeout")])
    assert result == "CleanupUnconfirmed" and calls == []
    assert visited.count("DiscoverOwnedInstance") == 1


def test_explicit_utc_fractional_inputs_match_controller_and_normalize_asl(config):
    config["started_at_utc"] = "2026-09-09T12:00:00.123456+00:00"
    config["deadline_utc"] = "2026-09-09T12:30:00.654321+00:00"
    result = W.build_artifacts(config)
    assert result["started_at_utc"] == config["started_at_utc"]
    assert result["deadline_utc"] == config["deadline_utc"]
    assert result["definition"]["States"][W.WAIT_STATE]["Timestamp"] == "2026-09-09T12:30:00Z"
    assert result["definition"]["TimeoutSeconds"] == 1801 + 7200 + 60


@pytest.mark.parametrize("field,value", [("ClientToken", "wrong"), ("InstanceType", "f2.48xlarge"),
                                        ("InstanceId", "wrong"), ("Tags", [])])
def test_foreign_instance_never_terminated(config, field, value):
    result, calls, _ = simulate(config, [response(config, **{field: value})])
    assert result == "OwnershipMismatch" and calls == []


@pytest.mark.parametrize("tag", ["Project", "Run", "OwnerAccount"])
def test_each_required_tag_is_verified_without_order_assumptions(config, tag):
    reply = response(config)
    tags = reply["Reservations"][0]["Instances"][0]["Tags"]
    next(item for item in tags if item["Key"] == tag)["Value"] = "foreign"
    result, calls, _ = simulate(config, [reply])
    assert result == "OwnershipMismatch" and calls == []


def test_reservation_owner_and_duplicate_matches_fail_closed(config):
    reply = response(config)
    reply["Reservations"][0]["OwnerId"] = "210987654321"
    assert simulate(config, [reply])[:2] == ("OwnershipMismatch", [])
    reply = response(config)
    reply["Reservations"][0]["Instances"] *= 2
    assert simulate(config, [reply])[:2] == ("AmbiguousDiscovery", [])


def test_discovered_id_cannot_change_after_lock(config):
    result, calls, _ = simulate(config, [response(config), response(config, instance_id="i-9876543210abcdef0")])
    assert result == "OwnershipMismatch" and calls == [{"InstanceIds": ["i-0123456789abcdef0"]}]


def test_absolute_grace_limit_bounds_work_independent_of_attempts(config):
    config.update(grace_seconds=1, max_attempts=120)
    result, calls, states = simulate(config, [EMPTY])
    assert result == "CleanupUnconfirmed" and not calls
    assert states.count("DiscoverOwnedInstance") == 1


def test_execution_input_cannot_override_scope_or_termination_target(config):
    result, calls, _ = simulate(config, [response(config), response(config, "terminated")],
                               execution_input={"attempt": {"value": -100}, "phase": "cleanup",
                                                "locked": {"InstanceId": "i-foreign"},
                                                "expected_tags": {}})
    assert result == "ConfirmedTerminated" and calls == [{"InstanceIds": ["i-0123456789abcdef0"]}]


@pytest.mark.parametrize("key,value", [
    ("account", "*"), ("region", "eu-west-1"), ("run_prefix", "tikitorch-fpga-lm/*"),
    ("client_token", "f" * 64), ("instance_type", "f2.48xlarge"), ("project", "other"),
    ("grace_seconds", True), ("grace_seconds", 7201), ("max_attempts", 121),
    ("max_attempts", 0), ("poll_seconds", 0), ("poll_seconds", "60"),
    ("deadline_utc", "2026-09-09T12:00:00Z"), ("deadline_utc", "2026-09-09T13:31:00Z"),
    ("deadline_utc", "2026-09-09T12:30:00-00:00"), ("started_at_utc", "2026-02-30T12:00:00Z"),
])
def test_invalid_or_unbounded_config_rejected(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        W.build_artifacts(config)


def test_cli_is_stdlib_only_and_emits_json_without_output_files(config, tmp_path):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config))
    before = set(tmp_path.iterdir())
    completed = subprocess.run([sys.executable, "-S", W.__file__, str(source), "--emit", "definition"],
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == W.build_artifacts(config)["definition"]
    assert set(tmp_path.iterdir()) == before
