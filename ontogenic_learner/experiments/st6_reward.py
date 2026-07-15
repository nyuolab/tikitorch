"""E2 — reward in the loop: the mechanism-screening factorial (E2_SPEC.md).

One process, one MorphSSM world model. The 2x2:
  GATE      S (surprise-only margin gate) | N (neuromod: surprise AND realized
            reward-salience above baseline; reward enters the SCHEDULE only)
  OBJECTIVE O (obs-only frames, REWARD-FREE objective -> random policy on both
            envs; the clean no-reward-in-objective baseline)
          | R (frames carry a REWARD TOKEN; model predicts it; MPC preference
            = P(next reward = + | history, a_cand) -- env-agnostic)

Cells: SO/NO (reward-free objective = random policy, S|N-gated world-model
learning -- the negative controls) | SR (b) | NR (a+b). Plus frozen-R / random
floors. GATE fires on OBS-ONLY surprise (orthogonal to the objective axis).
Primary metric = reward-return/1k (E1: prediction != control).

Frame codec (action-first, causal). Token id layout per env:
  0=[ACT] 1=[OBS] 2=[RWD]  |  3..3+K-1 = action tokens  |  next 2 = reward
  {none,+}  |  next n_obs_sym = obs cell-symbols.
  O frame: [ACT] a [OBS] <obs cells>
  R frame: [ACT] a [RWD] r [OBS] <obs cells>

Usage::
    python ontogenic_learner/experiments/st6_reward.py --env minatar \\
        --gate N --objective R --steps 120000 --world-seed 0 --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pc_convrnn_charlm import MorphPCEngine, build_model    # noqa: E402
from st2_live import MarginGate                             # noqa: E402
from st5_envs import make_env                               # noqa: E402
from nanogpt_shakespeare import tau_at                      # noqa: E402

OUT = REPO / "ontogenic_learner" / "results" / "st_reward"


class Codec:
    """Per-env token layout + frame builders (see module docstring)."""

    def __init__(self, env, objective):
        self.K = env.n_actions
        self.objective = objective                          # "O" | "R"
        self.ACT, self.OBS, self.RWD = 0, 1, 2
        self.ACT0 = 3
        self.RWD0 = self.ACT0 + self.K                      # 2 reward tokens
        self.SYM0 = self.RWD0 + 2                           # obs cell-symbols
        self.vocab = self.SYM0 + env.n_obs_sym
        self.obs_len = env.obs_len
        # frame length differs by objective
        self.frame_len = (2 + 1 + self.obs_len if objective == "O"
                          else 2 + 2 + 1 + self.obs_len)    # +[RWD]+r

    def frame(self, obs_sym, action, reward):
        a = torch.tensor([self.ACT, self.ACT0 + action], dtype=torch.long)
        obs = torch.tensor([self.OBS], dtype=torch.long)
        cells = obs_sym + self.SYM0
        if self.objective == "O":
            return torch.cat([a, obs, cells])
        r = torch.tensor([self.RWD, self.RWD0 + (1 if reward > 0 else 0)],
                         dtype=torch.long)
        return torch.cat([a, r, obs, cells])

    def learn_mask(self, y):
        """Learn ONLY meaningful targets: obs cell-symbols (always) + the
        reward token (R objective). Excludes separators + action tokens
        (policy tokens carry no loss)."""
        m = y >= self.SYM0                                  # obs symbols
        if self.objective == "R":
            m = m | ((y >= self.RWD0) & (y < self.SYM0))    # reward token
        return m

    def obs_mask(self, y):
        """Obs-symbol positions ONLY — the gate's surprise signal, so the
        GATE axis stays orthogonal to the OBJECTIVE axis (the gate never sees
        reward-prediction surprise; reported obs_ce is obs-only in both O/R)."""
        return y >= self.SYM0


class RewardSalience:
    """Realized-reward-salience for the neuromod gate: is |reward| on this
    window above its running baseline? EMA of |reward|; salient if the window's
    reward-magnitude exceeds the EMA (dense reward -> permissive; sparse -> fires
    only on the rare rewarding moment)."""

    def __init__(self, ema=0.99):
        self.ema = ema
        self.mu = 0.0

    def salient(self, reward_mag: float) -> bool:
        out = reward_mag > self.mu
        self.mu = self.ema * self.mu + (1 - self.ema) * reward_mag
        return out


class Agent:
    def __init__(self, env, args):
        self.env, self.args = env, args
        self.codec = Codec(env, args.objective)
        ns = argparse.Namespace(d_model=args.d_model, n_layers=args.n_layers,
                                kernel=7, dropout=0.0, tau=0.5, lam_min=2 ** -6)
        self.model = build_model("morphssm", self.codec.vocab, ns)
        self.model.set_tau(0.0)
        self.model.eval()
        self.engine = MorphPCEngine(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                     weight_decay=0.01)
        self.gate = MarginGate(delta=args.delta)
        self.sal = RewardSalience() if args.gate == "N" else None
        self.tau_act = args.tau_act
        self.ctx = torch.zeros(args.window, dtype=torch.long)

    def push(self, frame):
        self.ctx = torch.cat([self.ctx[len(frame):], frame])

    @torch.no_grad()
    def score_actions(self):
        """Batched (K, L) forward. Objective R: P(reward=+) at the [RWD]
        position -> the MPC preference. Objective O: None (reward-free objective
        -> random policy)."""
        c = self.codec
        if c.objective == "R":
            probes = torch.stack([
                torch.cat([self.ctx[3:], torch.tensor([c.ACT, c.ACT0 + a, c.RWD],
                                                      dtype=torch.long)])
                for a in range(c.K)])
            logits = self.model(probes)                     # (K, L, V)
            p = torch.softmax(logits[:, -1], dim=-1)[:, c.RWD0 + 1]   # P(reward=+)
            return [float(x) for x in p]
        # objective O = NO reward in the objective -> no reward-directed
        # controller -> random policy (uniform on both envs; the clean
        # "reward-free objective" baseline. The E1 obs-heuristic P(food) is a
        # reward proxy and is intentionally NOT used here.)
        return None

    def act(self):
        scores = self.score_actions()
        if scores is None:                                  # no reward signal -> explore
            return int(torch.randint(0, self.codec.K, (1,)))
        t = torch.tensor(scores)
        probs = torch.softmax((t - t.mean()) / self.tau_act, dim=0)
        return int(torch.multinomial(probs, 1))

    def step(self, learn=True):
        action = self.act()
        reward, obs_sym, done = self.env.step(action)
        self.push(self.codec.frame(obs_sym, action, reward))
        x = self.ctx[:-1].unsqueeze(0)
        y = self.ctx[1:].unsqueeze(0)
        self.engine.loss_mask = self.codec.learn_mask(y)      # train obs(+reward)
        self.engine.gate_mask = self.codec.obs_mask(y)        # gate on obs-only
        fired = False
        if learn:
            def fire_fn(ce):
                salient = (self.sal.salient(abs(reward))      # EVAL FIRST so the
                           if self.sal is not None else True)  # baseline always
                surprise = self.gate(ce)                       # updates
                return surprise and salient                    # neuromod AND
            _, ce, fired = self.engine.serve_and_learn(
                x, y, opt=self.opt, fire=fire_fn)
        else:
            with torch.no_grad():
                lg = self.model(x).reshape(-1, self.codec.vocab).float()
                keep = self.engine.gate_mask.reshape(-1)      # obs-only, comparable
                ce = float(F.cross_entropy(lg[keep], y.reshape(-1)[keep]))
            if self.sal is not None:                          # keep baseline live
                self.sal.salient(abs(reward))
        self.engine.loss_mask = None
        self.engine.gate_mask = None
        return reward, fired, ce


def run(args):
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env(args.env, world_seed=args.world_seed,
                   biome_steps=args.biome_steps)
    arm = f"{args.gate}{args.objective}" if args.mode == "agent" else args.mode
    # disambiguate tele by delta+horizon so parallel sweep cells never collide
    # (V2b lesson): a d/h tag keeps every cell's telemetry in its own file.
    dh = f"_d{args.delta}h{getattr(args, 'horizon', 1)}"
    tele = (OUT / f"e2_tele_{args.env}_{arm}{dh}_w{args.world_seed}_s{args.seed}.jsonl"
            ).open("w")
    t0 = time.time()
    total_r = fires = 0
    ce_sum = 0.0
    win = []

    if args.mode == "random":
        for t in range(args.steps):
            r, _, _ = env.step(int(torch.randint(0, env.n_actions, (1,))))
            total_r += r
            win.append(r)
            if (t + 1) % 1000 == 0:
                tele.write(json.dumps(dict(t=t + 1, ret1k=sum(win[-1000:]),
                                           biome=env.biome)) + "\n")
    else:
        agent = Agent(env, args)
        # tau-annealed birth: a short pre-life on the FIRST biome under a random
        # policy (world-model shaping; identical across the 4 cells via seed)
        birth = []
        for _ in range(args.prelife_frames):
            a = int(torch.randint(0, env.n_actions, (1,)))
            r, o, _ = env.step(a)
            birth.append(agent.codec.frame(o, a, r))
        ids = torch.cat(birth)
        agent.model.train()
        gb = torch.Generator().manual_seed(args.seed + 777)
        for step in range(args.prelife_steps):
            agent.model.set_tau(tau_at(step, args.prelife_steps, 0.5, True, 0.0))
            st = int(torch.randint(0, ids.numel() - args.window - 1, (1,),
                                   generator=gb))
            xb = ids[st:st + args.window].unsqueeze(0)
            yb = ids[st + 1:st + args.window + 1].unsqueeze(0)
            agent.engine.loss_mask = agent.codec.learn_mask(yb)
            agent.engine.train_step(xb, yb, agent.opt, credit="cascade")
        agent.engine.loss_mask = None
        agent.model.set_tau(0.0)
        agent.model.eval()
        agent.opt = torch.optim.AdamW(agent.model.parameters(), lr=args.lr,
                                      weight_decay=0.01)
        agent.gate = MarginGate(delta=args.delta)
        # warm the ctx with real frames
        for _ in range(args.window // agent.codec.frame_len + 1):
            a = int(torch.randint(0, env.n_actions, (1,)))
            r, o, _ = env.step(a)
            agent.push(agent.codec.frame(o, a, r))
        learn = args.mode == "agent"
        for t in range(args.steps):
            r, fired, ce = agent.step(learn=learn)
            total_r += r
            fires += int(fired)
            ce_sum += ce
            win.append(r)
            if (t + 1) % 1000 == 0:
                tele.write(json.dumps(dict(
                    t=t + 1, ret1k=sum(win[-1000:]), ce=round(ce_sum / (t + 1), 4),
                    fire_rate=round(fires / (t + 1), 4), biome=env.biome)) + "\n")
                tele.flush()
    tele.close()

    row = dict(gate="E2", env=args.env, arm=arm, delta=args.delta,
               horizon=getattr(args, "horizon", 1),
               world_seed=args.world_seed,
               seed=args.seed, steps=args.steps, total_return=total_r,
               ret_per_1k=round(total_r / max(args.steps, 1) * 1000, 3),
               fire_rate=round(fires / max(args.steps, 1), 4),
               obs_ce=round(ce_sum / max(args.steps, 1), 4),
               wall_s=round(time.time() - t0, 1))
    with (OUT / "e2_screen.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[E2] {args.env}/{arm}/w{args.world_seed}/s{args.seed}: "
          f"ret/1k {row['ret_per_1k']} fire {row['fire_rate']} ce {row['obs_ce']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=["foraging", "minatar"], required=True)
    p.add_argument("--mode", choices=["agent", "frozen", "random"],
                   default="agent")
    p.add_argument("--gate", choices=["S", "N"], default="S")
    p.add_argument("--objective", choices=["O", "R"], default="O")
    p.add_argument("--steps", type=int, default=120_000)
    p.add_argument("--world-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--biome-steps", type=int, default=20_000)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--tau-act", type=float, default=0.02)
    p.add_argument("--prelife-frames", type=int, default=8_000)
    p.add_argument("--prelife-steps", type=int, default=1_500)
    args = p.parse_args()
    run(args)
