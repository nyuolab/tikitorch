"""Local CLI for a deployed organism — S3-only, no SSH.
  python -m ontogenic_learner.organism.cli status
  python -m ontogenic_learner.organism.cli sample [--prompt "..."] [--n 300] [--temp 0.8]
  python -m ontogenic_learner.organism.cli report
  python -m ontogenic_learner.organism.cli feed <file.txt> [--steps 2000]
  python -m ontogenic_learner.organism.cli terminate
Reads .tiki_organism_state/run.env (written by the launcher) for S3_PREFIX/INSTANCE_ID/PROFILE."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STATE = os.path.join(REPO, ".tiki_organism_state", "run.env")


def env():
    d = {}
    for line in open(STATE):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            d[k] = v.strip('"')
    return d


def aws(e, *args, capture=True):
    cmd = ["aws", "--profile", e.get("PROFILE", "g1b"), "--region",
           e.get("REGION", "us-east-1"), *args]
    return subprocess.run(cmd, capture_output=capture, text=True)


def cmd_status(e):
    r = aws(e, "s3", "cp", f"{e['S3_PREFIX']}/status.json", "-")
    if r.returncode != 0:
        print("no status.json yet"); return
    st = json.loads(r.stdout)
    ii = aws(e, "ec2", "describe-instances", "--instance-ids", e["INSTANCE_ID"],
             "--query", "Reservations[0].Instances[0].[State.Name,LaunchTime]",
             "--output", "text")
    state, launch = (ii.stdout.split() + ["?", "?"])[:2] if ii.returncode == 0 else ("?", "?")
    hrs = st.get("uptime_h", 0)
    rate = float(e.get("HOURLY_RATE", "0.357"))
    print(f"instance {e['INSTANCE_ID']} [{state}] · life {st['life_day']:.2f} days "
          f"(step {st['life_step']:,}) · status={st['status']} · domain={st.get('domain')}")
    print(f"gate: {st['gate']} · counters: {st['counters']}")
    print(f"est. spend: ~${hrs*rate:.0f} ({hrs:.1f} h × ${rate}/h)")


def cmd_sample(e, prompt, n, temp, render, timeout=180):
    name = f"sample-{uuid.uuid4().hex[:8]}.json"
    req = {"kind": "sample", "prompt": prompt, "n": n, "temp": temp, "render": render}
    p = f"/tmp/{name}"
    json.dump(req, open(p, "w"))
    aws(e, "s3", "cp", p, f"{e['S3_PREFIX']}/inbox/{name}")
    print(f"asked; waiting up to {timeout}s for the organism's answer...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = aws(e, "s3", "cp", f"{e['S3_PREFIX']}/outbox/{name}", "-")
        if r.returncode == 0:
            print("\n" + json.loads(r.stdout)["text"])
            aws(e, "s3", "rm", f"{e['S3_PREFIX']}/outbox/{name}")
            return
        time.sleep(10)
    print("no answer (daemon busy or probing) — try again")


def cmd_report(e):
    loc = os.path.join(REPO, "organism_report_sync")
    for sub in ("probes", "ledger"):
        aws(e, "s3", "sync", f"{e['S3_PREFIX']}/{sub}", os.path.join(loc, sub),
            "--quiet", capture=False)
    subprocess.run([sys.executable, "-m", "ontogenic_learner.organism.report", loc],
                   cwd=REPO, env={**os.environ, "PYTHONPATH": REPO})
    print(f"open {loc}/report/report.md")


def cmd_feed(e, path, steps):
    text = open(path, encoding="utf-8", errors="ignore").read()
    name = f"feed-{int(time.time())}.json"
    req = {"kind": "feed", "text": text, "steps": steps,
           "sha256": hashlib.sha256(text.encode()).hexdigest()}
    p = f"/tmp/{name}"
    json.dump(req, open(p, "w"))
    aws(e, "s3", "cp", p, f"{e['S3_PREFIX']}/inbox/{name}")
    print(f"fed {len(text)} chars as guest experience ({steps} steps), ledgered by hash")


def cmd_terminate(e):
    name = f"terminate-{int(time.time())}.json"
    p = f"/tmp/{name}"
    json.dump({"kind": "terminate"}, open(p, "w"))
    aws(e, "s3", "cp", p, f"{e['S3_PREFIX']}/inbox/{name}")
    print("termination requested; waiting up to 5 min for final checkpoint ack...")
    t0 = time.time()
    while time.time() - t0 < 300:
        r = aws(e, "s3", "cp", f"{e['S3_PREFIX']}/status.json", "-")
        if r.returncode == 0 and json.loads(r.stdout).get("status") == "terminate-acked":
            print("acked — terminating instance")
            break
        time.sleep(15)
    aws(e, "ec2", "terminate-instances", "--instance-ids", e["INSTANCE_ID"],
        capture=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sp = sub.add_parser("sample")
    sp.add_argument("--prompt", default="")
    sp.add_argument("--n", type=int, default=300)
    sp.add_argument("--temp", type=float, default=0.8)
    sp.add_argument("--render", action="store_true")
    sub.add_parser("report")
    fp = sub.add_parser("feed")
    fp.add_argument("file")
    fp.add_argument("--steps", type=int, default=2000)
    sub.add_parser("terminate")
    a = ap.parse_args()
    e = env()
    if a.cmd == "status":
        cmd_status(e)
    elif a.cmd == "sample":
        cmd_sample(e, a.prompt, a.n, a.temp, a.render)
    elif a.cmd == "report":
        cmd_report(e)
    elif a.cmd == "feed":
        cmd_feed(e, a.file, a.steps)
    elif a.cmd == "terminate":
        cmd_terminate(e)


if __name__ == "__main__":
    main()
