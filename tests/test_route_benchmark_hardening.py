import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.route_certificates import actual_model_benchmark as B  # noqa: E402


def test_arm_order_is_deterministic_and_recordable():
    a = B.deterministic_arm_order(128, 4, 96, 0)
    assert a == B.deterministic_arm_order(128, 4, 96, 0)
    assert set(a) == set(B.ARMS)
    assert "compiled" not in B.deterministic_arm_order(64, 2, 96, 0)
    assert a != B.deterministic_arm_order(128, 4, 96, 1)


def test_durable_jsonl_and_atomic_heartbeat(tmp_path):
    rows = tmp_path / "rows.jsonl"
    B.append_jsonl_durable(rows, {"cell": 1})
    B.append_jsonl_durable(rows, {"cell": 2})
    assert [json.loads(line)["cell"] for line in rows.read_text().splitlines()] == [
        1,
        2,
    ]
    heartbeat = tmp_path / "heartbeat.json"
    B.atomic_write_json(heartbeat, {"state": "running", "cell": 2})
    assert json.loads(heartbeat.read_text())["cell"] == 2
    assert not heartbeat.with_name("heartbeat.json.tmp").exists()


def test_cell_timeout_is_enforced_and_heartbeated(tmp_path):
    heartbeat = tmp_path / "heartbeat.json"
    result = B.run_subprocess_cell(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path / "worker.log",
        heartbeat,
        cell_id="hang",
        timeout_s=0.15,
        hard_deadline=time.monotonic() + 5,
        heartbeat_s=0.02,
    )
    assert result["stop_reason"] == "CELL_TIMEOUT"
    assert result["elapsed_s"] < 2
    assert json.loads(heartbeat.read_text())["cell_id"] == "hang"


def test_timeout_terminates_the_entire_worker_process_group(tmp_path):
    child_pid = tmp_path / "child.pid"
    survived = tmp_path / "child-survived"
    child_code = (
        "import pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(1); "
        f"pathlib.Path({str(survived)!r}).write_text('alive')"
    )
    parent_code = (
        "import subprocess,sys,time,pathlib; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); "
        "time.sleep(30)"
    )
    result = B.run_subprocess_cell(
        [sys.executable, "-c", parent_code],
        tmp_path / "tree.log",
        tmp_path / "tree-heartbeat.json",
        cell_id="tree",
        timeout_s=0.3,
        hard_deadline=time.monotonic() + 5,
        heartbeat_s=0.02,
    )
    assert result["stop_reason"] == "CELL_TIMEOUT"
    assert child_pid.exists()
    time.sleep(1.1)
    assert not survived.exists(), "timed-out worker descendant survived"


def test_packed_boundary_includes_logits_and_targets():
    result = B._packed_arm(16, 1, 12, 7, warm=0, reps=1)
    state = result["packed_state"]
    assert state["boundary_logits_bytes"] == 1 * 12 * B.V * 4
    assert state["boundary_targets_bytes"] == 1 * 12 * 8
    assert state["boundary_total_bytes"] == (
        state["boundary_logits_bytes"] + state["boundary_targets_bytes"]
    )
    assert state["packed_reverse_retained_bytes"] == (
        state["tape_serialized_bytes"] + state["boundary_total_bytes"]
    )


def test_cascade_and_packed_match_at_5_and_100_updates():
    result = B.trajectory_parity(16, 1, 12, 11, steps=100, checkpoints=(5, 100))
    assert result["status"] == "PASS", result
    assert [row["step"] for row in result["records"]] == [5, 100]
    assert all(row["bit_exact"] for row in result["records"])
    assert all(
        row["cascade_digest"] == row["packed_digest"] for row in result["records"]
    )
