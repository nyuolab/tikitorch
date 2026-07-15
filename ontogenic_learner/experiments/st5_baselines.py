"""E1 baselines — the actor/learner-split incumbents on the same worlds.

Two shapes the manuscript's §6 contrasts with the fused loop:
  streaming_ac — streaming actor-critic in the spirit of Elsayed et al. 2024:
      no replay, online TD(lambda) with accumulating eligibility traces and an
      overshoot-bounded (ObGD-style) step. Still a DISTINCT learning
      computation per step (backward graph + trace memory) — that is the
      point of the comparison.
  replay_dqn   — small DQN with replay buffer + target network (the deployed
      actor/learner shape).

Same World, same seeds, same return accounting as st5_world. Featurization:
one-hot 3x3 tiles (27) + last-action one-hot (5). Verdict rows append to the
same e1_world.jsonl; per-arm parameter counts and per-step update flops are
recorded for the matched-compute Pareto.

Usage::

    python ontogenic_learner/experiments/st5_baselines.py --arm streaming_ac \\
        --steps 250000 --world-seed 0 --seed 0
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from st5_world import ACTIONS, OUT, World                    # noqa: E402

NTILE = 3            # empty/wall/food
FDIM = 9 * NTILE + 5


def feats(obs, last_a):
    x = torch.zeros(FDIM)
    for i, t in enumerate(obs):
        x[i * NTILE + int(t)] = 1.0
    x[27 + last_a] = 1.0
    return x


class MLP(nn.Module):
    def __init__(self, out_dim, h=128, two_head=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(FDIM, h), nn.LayerNorm(h), nn.LeakyReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.LeakyReLU())
        self.head = nn.Linear(h, out_dim)
        self.vhead = nn.Linear(h, 1) if two_head else None

    def forward(self, x):
        z = self.body(x)
        return (self.head(z), self.vhead(z).squeeze(-1)) if self.vhead \
            else self.head(z)


def run_streaming_ac(args, world, tele):
    """Online AC with accumulating traces; ObGD-style bounded step."""
    torch.manual_seed(args.seed)
    net = MLP(5, two_head=True)
    params = [p for p in net.parameters() if p.requires_grad]
    z = [torch.zeros_like(p) for p in params]                # traces
    gamma, lam, alpha, ent = 0.99, 0.8, args.lr, 0.01
    last_a = 4
    obs = world.obs()
    total_r = 0.0
    ret_window = []
    t0 = time.time()
    for t in range(args.steps):
        x = feats(obs, last_a)
        logits, v = net(x)
        v_detached = float(v.detach())
        pi = torch.softmax(logits, -1)
        a = int(torch.multinomial(pi, 1))
        r, obs2 = world.step(a)
        with torch.no_grad():
            _, v2 = net(feats(obs2, a))
        delta = r + gamma * float(v2) - v_detached
        # grad of (log pi(a) + v + entropy bonus) -> traces
        loss = torch.log(pi[a] + 1e-8) + v + ent * (-(pi * (pi + 1e-8).log()).sum())
        net.zero_grad(set_to_none=True)
        loss.backward()
        znorm = 0.0
        for zi, p in zip(z, params):
            zi.mul_(gamma * lam).add_(p.grad if p.grad is not None
                                      else torch.zeros_like(p))
            znorm += float(zi.abs().sum())
        # ObGD-style overshoot bound: effective step <= alpha
        step = alpha / max(1.0, alpha * abs(delta) * znorm)
        with torch.no_grad():
            for zi, p in zip(z, params):
                p.add_(zi, alpha=step * delta)
        obs, last_a = obs2, a
        total_r += r
        ret_window.append(r)
        if (t + 1) % 1000 == 0:
            tele.write(json.dumps(dict(t=t + 1,
                                       ret1k=sum(ret_window[-1000:]),
                                       biome=world.biome)) + "\n")
            tele.flush()
    return total_r, sum(p.numel() for p in params), time.time() - t0


def run_replay_dqn(args, world, tele):
    torch.manual_seed(args.seed)
    q = MLP(5)
    tgt = MLP(5)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr)
    buf = deque(maxlen=20_000)
    gamma = 0.99
    last_a = 4
    obs = world.obs()
    total_r = 0.0
    ret_window = []
    t0 = time.time()
    for t in range(args.steps):
        eps = max(0.05, 1.0 - t / 50_000)
        x = feats(obs, last_a)
        if float(torch.rand(())) < eps:
            a = int(torch.randint(0, 5, (1,)))
        else:
            with torch.no_grad():
                a = int(q(x).argmax())
        r, obs2 = world.step(a)
        buf.append((x, a, r, feats(obs2, a)))
        obs, last_a = obs2, a
        total_r += r
        ret_window.append(r)
        if len(buf) >= 512 and t % 4 == 0:                   # 1 update / 4 steps
            idx = torch.randint(0, len(buf), (32,))
            xb = torch.stack([buf[i][0] for i in idx.tolist()])
            ab = torch.tensor([buf[i][1] for i in idx.tolist()])
            rb = torch.tensor([buf[i][2] for i in idx.tolist()])
            x2 = torch.stack([buf[i][3] for i in idx.tolist()])
            with torch.no_grad():
                y = rb + gamma * tgt(x2).max(-1).values
            qsa = q(xb).gather(1, ab.unsqueeze(1)).squeeze(1)
            opt.zero_grad(set_to_none=True)
            F.smooth_l1_loss(qsa, y).backward()
            opt.step()
        if t % 1000 == 0:
            tgt.load_state_dict(q.state_dict())
            if t:
                tele.write(json.dumps(dict(t=t, ret1k=sum(ret_window[-1000:]),
                                           biome=world.biome)) + "\n")
                tele.flush()
    return total_r, sum(p.numel() for p in q.parameters()), time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["streaming_ac", "replay_dqn"],
                   required=True)
    p.add_argument("--steps", type=int, default=250_000)
    p.add_argument("--world-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--biome-steps", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    world = World(args.world_seed, biome_steps=args.biome_steps)
    tele = (OUT / f"e1_tele_{args.arm}_w{args.world_seed}_s{args.seed}.jsonl"
            ).open("w")
    fn = run_streaming_ac if args.arm == "streaming_ac" else run_replay_dqn
    total_r, n_params, wall = fn(args, world, tele)
    tele.close()
    row = dict(gate="E1", arm=args.arm, world_seed=args.world_seed,
               seed=args.seed, steps=args.steps, total_return=total_r,
               ret_per_1k=round(total_r / max(args.steps, 1) * 1000, 3),
               n_params=n_params, wall_s=round(wall, 1))
    with (OUT / "e1_world.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[E1] {args.arm}/w{args.world_seed}/s{args.seed}: "
          f"return/1k {row['ret_per_1k']} params {n_params}")


if __name__ == "__main__":
    main()
