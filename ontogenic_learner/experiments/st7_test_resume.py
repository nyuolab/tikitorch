"""E5 Stage-0 drill harness (E5_SPEC pilot gates, local, free).

Drills:
  1. kill-and-resume twin: a life interrupted at S1 and resumed in a FRESH
     process must produce telemetry identical to an uninterrupted twin at S2
     (covers model, opt, both env states, ctx buffers, EMAs, RNG).
  2. probe isolation: the state-hash witness is idempotent and unchanged
     across a full probe battery; post-probe continuation identical to a
     probe-free twin.
  3. env roundtrips: bit-exact get/set_state for every world in the trio.

Run: python ontogenic_learner/experiments/st7_test_resume.py [workdir]
(Uses a subprocess for the resume leg so restore crosses a real process
boundary — the organism test_bit_resume re-exec pattern.)"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))

SMOKE_CFG = dict(era_steps=800, biome_steps=200, cycles=2, prelife_frames=300,
                 prelife_steps=40, ckpt_every_steps=650, ckpt_every_secs=3600,
                 d_model=64, n_layers=2)


def run_life(workdir, max_steps, resume=False, cfg_path=None):
    cmd = [sys.executable, str(HERE / "st7_life.py"), "--workdir", str(workdir),
           "--max-steps", str(max_steps), "--config", str(cfg_path)]
    if resume:
        cmd.append("--resume")
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def tele(workdir):
    p = Path(workdir) / "tele.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def drill_1_kill_resume(base, cfg_path):
    a, b = base / "twinA", base / "twinB"
    run_life(a, 2600, cfg_path=cfg_path)
    run_life(b, 1300, cfg_path=cfg_path)          # ...process exits (the kill)
    run_life(b, 1300, resume=True, cfg_path=cfg_path)
    ta, tb = tele(a), tele(b)
    assert ta and ta == tb, "kill-resume twin diverged"
    print(f"DRILL 1 OK: interrupted+resumed ≡ uninterrupted ({len(ta)} rows)")


def drill_2_probe_isolation(base, cfg_path):
    import torch  # noqa: F401
    from st7_life import Life, DEFAULTS
    import st7_probes as P
    cfg = dict(DEFAULTS); cfg.update(json.loads(Path(cfg_path).read_text()))
    w1, w2 = base / "probeA", base / "probeB"
    life = Life(str(w1), cfg)
    life.birth(); life.live(max_steps=700)
    h0 = P.state_hash(life)
    assert h0 == P.state_hash(life), "witness not idempotent"
    windows = P.generate_windows(life.codec, cfg, n_windows=4, window=96)
    P.suite_a(life.agent.model, life.codec, windows)
    P.suite_b(life.agent, cfg, steps=200)
    acts, _ = P.suite_c(None, life.agent, windows)
    P.suite_c(None, life.agent, windows, snapshot=acts)
    assert h0 == P.state_hash(life), "probe battery mutated the life"
    life.live(max_steps=500)
    life2 = Life(str(w2), cfg)
    life2.birth(); life2.live(max_steps=1200)
    assert tele(w1) == tele(w2), "post-probe continuation diverged"
    print("DRILL 2 OK: witness idempotent; battery isolated; continuation ≡ twin")


def drill_3_env_roundtrips():
    import torch, hashlib
    from st5_envs import make_env
    for name, steps in (("foraging", 300), ("minatar", 300), ("babyai", 300)):
        torch.manual_seed(0)
        a = make_env(name, world_seed=0, biome_steps=100)
        for t in range(50):
            a.step(t % a.n_actions)
        blob = a.get_state()
        seq_a = [(round(float(r), 6),
                  hashlib.sha256(bytes(o.tolist())).hexdigest()[:8], d)
                 for r, o, d in (a.step(t % a.n_actions) for t in range(steps))]
        b = make_env(name, world_seed=999, biome_steps=7)
        b.set_state(blob)
        seq_b = [(round(float(r), 6),
                  hashlib.sha256(bytes(o.tolist())).hexdigest()[:8], d)
                 for r, o, d in (b.step(t % b.n_actions) for t in range(steps))]
        assert seq_a == seq_b, f"{name} roundtrip diverged"
    print("DRILL 3 OK: all three worlds bit-exact after restore")


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(tempfile.mkdtemp(prefix="e5drills-"))
    base.mkdir(parents=True, exist_ok=True)
    cfg_path = base / "smoke.json"
    cfg_path.write_text(json.dumps(SMOKE_CFG))
    try:
        drill_3_env_roundtrips()
        drill_1_kill_resume(base, cfg_path)
        drill_2_probe_isolation(base, cfg_path)
        print("ALL STAGE-0 DRILLS PASS")
    finally:
        if len(sys.argv) <= 1:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
