"""Offline runner contracts: these tests never call AWS or start training."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(script, *args, env=None, cwd=None):
    clean_env = dict(os.environ)
    for name in ("AWS_PROFILE", "REGION", "PC_TASK", "PC_ARGS", "MAX_MINUTES", "REPO",
                 "KEY_NAME", "SG", "AGFI", "PROJECT_DIR", "PEM"):
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
    for name in ("aws", "curl", "shutdown", "apt-get", "nvidia-smi", "ssh", "scp",
                 "rsync", "sudo", "tmux"):
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
    assert not (tmp_path / "manuscript" / "results").exists()


def test_scale_rejects_invalid_quoting_before_any_side_effect(offline_env, tmp_path):
    env, sentinel = offline_env
    env.update(PC_ARGS="--tag 'unclosed", PYBIN=sys.executable)
    result = _run("_pc_scale_onbox.sh", "--print-commands", env=env, cwd=tmp_path)
    assert result.returncode == 2
    assert "invalid shell-style quoting" in result.stderr
    assert not sentinel.exists()
    assert not (tmp_path / "manuscript" / "results").exists()


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
    assert not (tmp_path / "manuscript" / "results").exists()


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


_EXPLICIT_LAUNCHERS = (
    ("run_tikitorch_gpu_benchmarks.sh", ".tiki_gpu_state"),
    ("run_tikitorch_f2_silicon.sh", ".tiki_f2_state"),
)


@pytest.mark.parametrize("script,state_dir", _EXPLICIT_LAUNCHERS)
@pytest.mark.parametrize("args,code", [
    ((), 0), (("help",), 0), (("--help",), 0), (("-h",), 0),
    (("typo",), 2), (("launch", "extra"), 2),
])
@pytest.mark.parametrize("configured", [False, True])
def test_cloud_launchers_help_and_invalid_commands_have_no_side_effects(
        script, state_dir, args, code, configured, offline_env, tmp_path):
    env, sentinel = offline_env
    env["PROJECT_DIR"] = str(tmp_path)
    if configured:
        env.update(AWS_PROFILE="offline", KEY_NAME="offline", SG="sg-offline",
                   AGFI="agfi-offline")
    result = _run(script, *args, env=env, cwd=tmp_path)
    assert result.returncode == code, result.stderr
    assert "Usage:" in result.stdout + result.stderr
    assert not sentinel.exists(), "help/invalid input must never contact external tools"
    assert not (tmp_path / state_dir).exists()
    assert not (tmp_path / "manuscript").exists()


@pytest.mark.parametrize("script,state_dir", _EXPLICIT_LAUNCHERS)
def test_cloud_launchers_require_credentials_before_creating_launch_state(
        script, state_dir, offline_env, tmp_path):
    env, sentinel = offline_env
    env["PROJECT_DIR"] = str(tmp_path)
    result = _run(script, "launch", env=env, cwd=tmp_path)
    assert result.returncode != 0
    assert "set AWS_PROFILE" in result.stderr
    assert not sentinel.exists()
    assert not (tmp_path / state_dir).exists()


@pytest.mark.parametrize("script,state_dir", _EXPLICIT_LAUNCHERS)
def test_cloud_upload_filters_include_runners_and_exclude_research_and_caches(
        script, state_dir, tmp_path):
    source = (ROOT / "scripts" / script).read_text()
    lines = source.replace("\\\n", " ").splitlines()
    upload = next(line for line in lines if line.lstrip().startswith("rsync -avz"))
    filters = [arg for arg in shlex.split(upload)
               if arg.startswith(("--include=", "--exclude="))]
    assert "--include=/tikitorch/***" in filters
    assert "--include=/scripts/***" in filters
    assert filters.index("--exclude=__pycache__/") < filters.index("--include=/tikitorch/***")
    assert filters.index("--exclude=zoo/") < filters.index("--include=/tikitorch/***")
    assert "tikitorch/manuscript/results" not in upload
    assert filters[-1] == "--exclude=*"
    rsync = shutil.which("rsync")
    if rsync is None:
        pytest.skip("local rsync needed to exercise upload filter semantics")

    # Exercise the actual filter arguments on a local synthetic tree. This never
    # calls the launcher, starts a workload, or supplies a remote rsync endpoint.
    included = {
        "tikitorch/__init__.py", "tikitorch/fpga/bridge.cpp",
        "tikitorch/hardware/hls/forge_kernels/kernel.cpp",
        "scripts/run_closed_benchmarks.py", "scripts/run_fpga_f2_verify.sh",
        "scripts/nested/helper.py",
    }
    excluded = {
        "tikitorch/zoo/weights.pt", "tikitorch/nested/__pycache__/stale.pyc",
        "tikitorch/orphan.pyc", "scripts/__pycache__/helper.pyc",
        "scripts/.cache/value", "tikitorch/.pytest_cache/value",
        "tikitorch/.ruff_cache/value", "manuscript/results/run/raw.txt",
        "manuscript/main.tex", "results/legacy.txt", "tikitorch/results/stale.txt",
        "README.md", ".git/config",
    }
    src, dst = tmp_path / "source", tmp_path / "destination"
    src.mkdir()
    dst.mkdir()
    for name in included | excluded:
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    result = subprocess.run(
        [rsync, "-a", *filters, f"{src}/", f"{dst}/"],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    copied = {str(path.relative_to(dst)) for path in dst.rglob("*") if path.is_file()}
    assert copied == included


@pytest.mark.parametrize("script,state_dir", _EXPLICIT_LAUNCHERS)
@pytest.mark.parametrize("aws_status", [0, 99])
def test_cloud_termination_keeps_state_until_request_succeeds(
        script, state_dir, aws_status, offline_env, tmp_path):
    env, sentinel = offline_env
    env.update(PROJECT_DIR=str(tmp_path), AWS_PROFILE="offline", KEY_NAME="offline",
               SG="sg-offline", AGFI="agfi-offline")
    state = tmp_path / state_dir / "instance.state"
    state.parent.mkdir()
    record = "i-offline|192.0.2.1|agfi-offline\n"
    state.write_text(record)
    # Fake CLI only: assert the exact intended operation before returning the
    # selected status. No credentials or real AWS executable are used.
    aws_stub = tmp_path / "bin" / "aws"
    aws_stub.write_text(
        '#!/bin/bash\n[ "$1 $2" = "ec2 terminate-instances" ] || exit 98\n'
        + f"touch {shlex.quote(str(sentinel))}\nexit {aws_status}\n"
    )
    aws_stub.chmod(0o755)
    result = _run(script, "terminate", env=env, cwd=tmp_path)
    assert result.returncode == aws_status, result.stderr
    assert sentinel.exists(), "the local fake CLI should receive the termination request"
    if aws_status:
        assert state.read_text() == record
    else:
        assert not state.exists()


def _remote_job_command(script):
    """Render only the tmux SSH argument using a local printf stub, then parse it."""
    source = (ROOT / "scripts" / script).read_text()
    call = re.search(r'ssh_to "\$IP" "cd ~/tikitorch[\s\S]*?tmux list-sessions"', source)
    assert call is not None
    rendered = subprocess.run(
        ["bash", "-c", 'ssh_to(){ printf "%s" "$2"; }; IP=offline; AGFI=agfi-offline;\n'
         + call.group()], text=True, capture_output=True, timeout=10,
    )
    assert rendered.returncode == 0, rendered.stderr
    command = re.search(r"tmux new-session[^']*'([^']*)'", rendered.stdout)
    assert command is not None
    args = shlex.split(command.group(1))
    assert args[:5] == ["bash", "-e", "-o", "pipefail", "-c"]
    return args


@pytest.mark.parametrize("failed_task", ["none", "smnist", "scifar"])
def test_gpu_done_marker_requires_both_jobs_to_succeed(failed_task, offline_env, tmp_path):
    env, sentinel = offline_env
    job = _remote_job_command("run_tikitorch_gpu_benchmarks.sh")
    repo = tmp_path / "checkout"
    (repo / "logs").mkdir(parents=True)
    (repo / "bench_DONE").touch()  # A previous success must not mask this run's failure.
    calls = tmp_path / "job-calls"
    python_stub = tmp_path / "bin" / "python3"
    python_stub.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_JOB_CALLS"\n'
        'case "$*" in *"--task $TEST_FAILED_TASK "*) exit 17;; esac\n'
        'printf "fake successful workload\\n"\n'
    )
    python_stub.chmod(0o755)
    job[-1] = job[-1].replace("~/tikitorch", shlex.quote(str(repo)))
    run_env = dict(os.environ, **env, TEST_JOB_CALLS=str(calls), TEST_FAILED_TASK=failed_task)
    result = subprocess.run(job, env=run_env, text=True, capture_output=True, timeout=10)
    assert result.returncode == (0 if failed_task == "none" else 17), result.stderr
    assert (repo / "bench_DONE").exists() == (failed_task == "none")
    assert len(calls.read_text().splitlines()) == (1 if failed_task == "smnist" else 2)
    assert not sentinel.exists()


@pytest.mark.parametrize("workload_status", [0, 17])
def test_fpga_done_marker_requires_verifier_to_succeed(workload_status, offline_env, tmp_path):
    env, sentinel = offline_env
    job = _remote_job_command("run_tikitorch_f2_silicon.sh")
    repo = tmp_path / "checkout"
    (repo / "logs").mkdir(parents=True)
    (repo / "tiki_f2_DONE").touch()
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_fpga_f2_verify.sh").write_text(
        f"#!/bin/bash\nprintf 'fake verifier\\n'\nexit {workload_status}\n"
    )
    job[-1] = job[-1].replace("~/tikitorch", shlex.quote(str(repo)))
    result = subprocess.run(job, env=dict(os.environ, **env), text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == workload_status, result.stderr
    assert (repo / "tiki_f2_DONE").exists() == (workload_status == 0)
    assert not sentinel.exists()


def test_all_repository_shell_scripts_parse():
    paths = subprocess.check_output(["git", "ls-files", "-z", "*.sh"], cwd=ROOT).decode().split("\0")
    assert len(paths) > 3
    for path in filter(None, paths):
        result = subprocess.run(["bash", "-n", str(ROOT / path)], text=True, capture_output=True)
        assert result.returncode == 0, f"{path}: {result.stderr}"
