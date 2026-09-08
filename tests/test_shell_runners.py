"""Offline runner contracts: these tests never call AWS or start training."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(script, *args, env=None, cwd=None):
    clean_env = dict(os.environ)
    for name in ("AWS_PROFILE", "REGION", "PC_TASK", "PC_ARGS", "MAX_MINUTES", "REPO"):
        clean_env.pop(name, None)
    clean_env.update(env or {})
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *args],
        env=clean_env, cwd=cwd, text=True, capture_output=True, timeout=10,
    )


@pytest.fixture
def offline_env(tmp_path):
    """An accidental network command is a visible failure, never an AWS request."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sentinel = tmp_path / "external-command"
    for name in ("aws", "curl", "shutdown", "apt-get", "nvidia-smi"):
        stub = bin_dir / name
        stub.write_text(f"#!/bin/bash\ntouch {shlex.quote(str(sentinel))}\nexit 99\n")
        stub.chmod(0o755)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}, sentinel


@pytest.mark.parametrize("task,script,matrix", [
    ("mnist", "pc_scale_mnist.py", True),
    ("cifar", "pc_scale_cifar.py", True),
    ("convlm", "pc_charlm_conv.py", False),
])
def test_scale_print_uses_exact_safe_arguments(task, script, matrix, offline_env, tmp_path):
    env, sentinel = offline_env
    injected = tmp_path / "injected"
    literal = f"$(touch {injected})"
    extra = ["--tag", "two words", "--label", literal, "--empty", ""]
    env.update(PC_TASK=task, PYBIN=sys.executable, PC_ARGS=shlex.join(extra))
    result = _run("_pc_scale_onbox.sh", "--print-commands", env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    expected = [sys.executable, f"scripts/{script}"]
    if matrix:
        expected += ["--matrix"]
    assert command[:len(expected)] == expected
    assert command[len(expected):-2] == extra
    assert command[-2] == "--out"
    assert not sentinel.exists()
    assert not injected.exists()
    assert not (tmp_path / "results").exists()


def test_scale_rejects_invalid_quoting_before_any_side_effect(offline_env, tmp_path):
    env, sentinel = offline_env
    env.update(PC_ARGS="--tag 'unclosed", PYBIN=sys.executable)
    result = _run("_pc_scale_onbox.sh", "--print-commands", env=env, cwd=tmp_path)
    assert result.returncode == 2
    assert "invalid shell-style quoting" in result.stderr
    assert not sentinel.exists()
    assert not (tmp_path / "results").exists()


def test_g5_prints_audit_gate_then_three_array_safe_quality_jobs(offline_env, tmp_path):
    env, sentinel = offline_env
    python_path = str(tmp_path / "python with spaces;literal")
    env["PY"] = python_path
    result = _run("_pc_g5_onbox.sh", "--print-commands", env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 4
    assert all(command[0] == python_path for command in commands)
    assert "--strict-audit" in commands[0]
    assert all("--strict-audit" not in command for command in commands[1:])
    assert commands[2][commands[2].index("--steps") + 1] == "6000"
    assert "--no-anneal" in commands[3]
    assert not sentinel.exists()
    assert not (tmp_path / "results").exists()


def test_launcher_render_is_offline_and_preserves_literal_inputs(offline_env, tmp_path):
    env, sentinel = offline_env
    payload = f"$(touch {tmp_path / 'injected'})"
    args = [f"s3://source/{payload}", "s3://results/with spaces", "commit'quote", "sha\\quote"]
    env.update(PC_ARGS=f"--tag '{payload}'", PC_TASK="convlm", MAX_MINUTES="12")
    result = _run("run_pc_scale_gpu.sh", "render", *args, env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert subprocess.run(["bash", "-n"], input=result.stdout, text=True).returncode == 0
    assert "shutdown -h +12" in result.stdout
    assert 'exit "$RUN_RC"' in result.stdout
    # Execute only rendered variable assignments, never the generated workload.
    prefixes = ("S3_SRC=", "SOURCE_COMMIT=", "EXPECTED_SOURCE_ARCHIVE_SHA256=", "export PC_TASK=")
    assignments = [line for line in result.stdout.splitlines() if line.startswith(prefixes)]
    names = ["S3_SRC", "S3_RESULTS", "SOURCE_COMMIT", "EXPECTED_SOURCE_ARCHIVE_SHA256", "PC_ARGS"]
    probe = "import os,json; print(json.dumps([os.environ[x] for x in " + repr(names) + "]))"
    checked = subprocess.run(
        ["bash", "-c", "set -a\n" + "\n".join(assignments) + "\n"
         + shlex.join([sys.executable, "-c", probe])],
        text=True, capture_output=True, timeout=10,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == args + [env["PC_ARGS"]]
    assert not sentinel.exists()
    assert not (tmp_path / "injected").exists()


@pytest.mark.parametrize("setting,value", [("PC_TASK", "typo"), ("MAX_MINUTES", "0"), ("MAX_MINUTES", "1;exit")])
def test_launcher_invalid_inputs_fail_offline(setting, value, offline_env, tmp_path):
    env, sentinel = offline_env
    env[setting] = value
    result = _run("run_pc_scale_gpu.sh", "render", "src", "dst", "commit", "sha", env=env, cwd=tmp_path)
    assert result.returncode == 2
    assert not sentinel.exists()


def test_all_repository_shell_scripts_parse():
    paths = subprocess.check_output(["git", "ls-files", "-z", "*.sh"], cwd=ROOT).decode().split("\0")
    assert len(paths) > 3
    for path in filter(None, paths):
        result = subprocess.run(["bash", "-n", str(ROOT / path)], text=True, capture_output=True)
        assert result.returncode == 0, f"{path}: {result.stderr}"
