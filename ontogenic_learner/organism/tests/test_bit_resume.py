"""Gate-2 test: checkpoint + fresh-process resume must be BIT-IDENTICAL to an uninterrupted
run — params, reservoir, counters, and the RNG-driven trajectory.

Run (repo root): PYTHONPATH=. python3 ontogenic_learner/organism/tests/test_bit_resume.py
It re-execs itself as subprocesses to guarantee genuinely fresh process state."""
import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

STEPS_A, STEPS_TOTAL = 150, 300


def build(seed=0):
    import torch
    from tikitorch.nn.predictive_coding import DyadicPCNet
    from ontogenic_learner.organism.config import make_config
    from ontogenic_learner.organism.vocab import Vocab
    from ontogenic_learner.organism.corpus import World
    from ontogenic_learner.organism.policy import TwoSpeedPolicy
    from ontogenic_learner.organism.schedule import Schedule
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    cfg = make_config(smoke=True)
    vocab = Vocab.build()
    gens = {"schedule": torch.Generator().manual_seed(seed + 1),
            "mnist": torch.Generator().manual_seed(seed + 2)}
    world = World(vocab, cfg.ctx, cfg.batch, gens["mnist"], val_windows=256)
    dims = [cfg.ctx * vocab.V] + [cfg.hidden] * (cfg.depth - 1) + [vocab.V]
    net = DyadicPCNet(dims, residual=True, gamma=cfg.gamma)
    policy = TwoSpeedPolicy(net, vocab.V, cfg, seed)
    sched = Schedule(cfg, gens["schedule"])
    return cfg, vocab, world, net, policy, sched, gens


def run_steps(cfg, world, net, policy, sched, start, end):
    for step in range(start, end):
        dom = sched.domain(step)
        xi, yi = world.cursors[dom].next_windows()
        policy.step(xi, yi, dom, step, "cpu")
        if step > 0 and step % cfg.sleep_every == 0:
            policy.sleep_block("cpu")


def fingerprint(net, policy, world, sched):
    import torch
    h = hashlib.sha256()
    for p in net.parameters():
        h.update(p.detach().numpy().tobytes())
    h.update(str(policy.n_fast).encode())
    h.update(str(policy.n_sleep).encode())
    h.update(str(len(policy.res.x)).encode())
    if policy.res.x:
        h.update(torch.stack(policy.res.x).numpy().tobytes())
    h.update(str(world.cursor_states()).encode())
    h.update(str(sched.state()).encode())
    return h.hexdigest()


def main_child(mode, ckpt_dir, out_path):
    from ontogenic_learner.organism import state as S
    cfg, vocab, world, net, policy, sched, gens = build()
    if mode == "straight":
        run_steps(cfg, world, net, policy, sched, 0, STEPS_TOTAL)
    elif mode == "part1":
        run_steps(cfg, world, net, policy, sched, 0, STEPS_A)
        blob = S.capture(net, policy, world, sched, gens, STEPS_A, {}, {})
        S.save_atomic(blob, ckpt_dir, STEPS_A)
        os._exit(9)                              # simulated kill -9 (no cleanup)
    elif mode == "part2":
        blob = S.load_latest(ckpt_dir)
        life_step, _, _ = S.restore(blob, net, policy, world, sched, gens)
        assert life_step == STEPS_A
        run_steps(cfg, world, net, policy, sched, life_step, STEPS_TOTAL)
    open(out_path, "w").write(fingerprint(net, policy, world, sched))


if __name__ == "__main__":
    if len(sys.argv) > 1:                        # child mode
        main_child(sys.argv[1], sys.argv[2], sys.argv[3])
        sys.exit(0)
    tmp = tempfile.mkdtemp(prefix="orgtest_")
    env = {**os.environ, "PYTHONPATH": REPO, "PYTHONHASHSEED": "0"}
    fA = os.path.join(tmp, "a.txt")
    fB = os.path.join(tmp, "b.txt")
    ck = os.path.join(tmp, "ckpt")
    print("run A (uninterrupted 300 steps)...", flush=True)
    subprocess.run([sys.executable, __file__, "straight", ck, fA], env=env, check=True)
    print("run B part 1 (150 steps -> checkpoint -> hard exit)...", flush=True)
    r = subprocess.run([sys.executable, __file__, "part1", ck, fB], env=env)
    assert r.returncode == 9, f"expected simulated kill, got {r.returncode}"
    print("run B part 2 (fresh process resume -> 300)...", flush=True)
    subprocess.run([sys.executable, __file__, "part2", ck, fB], env=env, check=True)
    a, b = open(fA).read(), open(fB).read()
    print(f"A={a[:16]}… B={b[:16]}…")
    assert a == b, "BIT-RESUME FAILED: fingerprints differ"
    print("GATE 2 PASS: kill -9 resume is bit-identical to uninterrupted.")
