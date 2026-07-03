"""Gate-4 test: a full probe suite must leave the organism's learnable/behavioral state
bit-identical (params, optimizer, reservoir, EMA, cursors — everything except the dedicated
probe RNG, which is probes' own stream by design)."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

import torch
from tikitorch.nn.predictive_coding import DyadicPCNet
from ontogenic_learner.organism.config import make_config
from ontogenic_learner.organism.corpus import World
from ontogenic_learner.organism.policy import TwoSpeedPolicy
from ontogenic_learner.organism.probes import run_suite
from ontogenic_learner.organism.schedule import Schedule
from ontogenic_learner.organism.vocab import Vocab


def hash_state(net, policy, world, sched, gens):
    h = hashlib.sha256()
    for p in net.parameters():
        h.update(p.detach().numpy().tobytes())
    import json
    h.update(json.dumps(policy.state(), default=lambda t: t.tolist()
                        if isinstance(t, torch.Tensor) else str(t),
                        sort_keys=True).encode())
    h.update(str(world.cursor_states()).encode())
    h.update(str(sched.state()).encode())
    h.update(torch.get_rng_state().numpy().tobytes())
    for k in ("schedule", "mnist", "sampler"):          # probe gen excluded by design
        h.update(gens[k].get_state().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    cfg = make_config(smoke=True)
    vocab = Vocab.build()
    gens = {"schedule": torch.Generator().manual_seed(1),
            "mnist": torch.Generator().manual_seed(2),
            "probe": torch.Generator().manual_seed(3),
            "sampler": torch.Generator().manual_seed(4)}
    world = World(vocab, cfg.ctx, cfg.batch, gens["mnist"], val_windows=128)
    dims = [cfg.ctx * vocab.V] + [cfg.hidden] * (cfg.depth - 1) + [vocab.V]
    net = DyadicPCNet(dims, residual=True, gamma=cfg.gamma)
    policy = TwoSpeedPolicy(net, vocab.V, cfg, 0)
    sched = Schedule(cfg, gens["schedule"])
    # give it a little life so the state is non-trivial
    for step in range(40):
        dom = sched.domain(step)
        xi, yi = world.cursors[dom].next_windows()
        policy.step(xi, yi, dom, step, "cpu")

    before = hash_state(net, policy, world, sched, gens)
    out = "/tmp/org_probe_iso"
    os.makedirs(out, exist_ok=True)
    run_suite(net, policy, vocab, world, cfg, gens, 40, out, "cpu")
    after = hash_state(net, policy, world, sched, gens)
    assert before == after, "PROBE CONTAMINATION: organism state changed"
    print("GATE 4 (isolation) PASS: full probe suite leaves organism state bit-identical.")


if __name__ == "__main__":
    main()
