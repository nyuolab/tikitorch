"""E1 — the embodied gate: model-predictive acting over a live world model.

Pre-registration: ontogenic_learner/E1_SPEC.md (frozen before any run).
One process, one model: acting = K=5 candidate-action forwards through the
MorphSSM world model, choosing by a FIXED preference functional over the
predicted next observation; learning = the fused, gated cascade kick on the
realized observation's prediction error. Prediction is the only teacher —
no value function, no replay buffer, no separate learner.

World: procedural foraging gridworld rendered as a token stream. Frame =
[OBS] t1..t9 [ACT] a  (12 tokens; 3x3 local view row-major). Biomes differ
in wall layout and food statistics and switch on a schedule with revisits.

Arms: live | frozen_mpc | random | oracle_mpc (pilot: MPC with the TRUE
simulator as the model — upper-bounds the preference functional).
(streaming-RL and replay actor-learner baselines land in st5_baselines.py.)

Usage::

    python ontogenic_learner/experiments/st5_world.py --arm oracle_mpc \\
        --steps 20000 --world-seed 0 --seed 0        # the pilot cell
    python ontogenic_learner/experiments/st5_world.py --arm live \\
        --steps 1000000 --world-seed 0 --seed 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pc_convrnn_charlm import MorphPCEngine, build_model    # noqa: E402
from st2_live import MarginGate, UnigramCE, _NoStep         # noqa: E402

OUT = REPO / "ontogenic_learner" / "results" / "st_world"

# ---- vocabulary -------------------------------------------------------------
# tiles
EMPTY, WALL, FOOD = 0, 1, 2
# stream tokens
TOK_OBS, TOK_ACT = 3, 4                      # separators
TOK_TILE = {EMPTY: 5, WALL: 6, FOOD: 7}
TOK_ACTION0 = 8                              # 8..12 = N,S,E,W,stay
ACTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
VOCAB = 13
FRAME = 12                                   # [OBS] + 9 tiles + [ACT] + a


class World:
    """Procedural foraging gridworld with biome switches (seeded, pure)."""

    SIZE = 16

    def __init__(self, world_seed: int, biome_steps: int = 50_000,
                 biomes=(0, 1, 2, 0, 1, 2)):
        self.gen = torch.Generator().manual_seed(world_seed)
        self.biomes = biomes
        self.biome_steps = biome_steps
        self.t = 0
        self._enter_biome(0)

    # biome b: wall density and food placement statistics differ
    def _enter_biome(self, idx):
        self.biome_idx = idx
        b = self.biomes[idx % len(self.biomes)]
        self.biome = b
        S = self.SIZE
        self.grid = torch.zeros(S, S, dtype=torch.long)
        wall_p = (0.05, 0.18, 0.10)[b]
        walls = torch.rand(S, S, generator=self.gen) < wall_p
        self.grid[walls] = WALL
        self.grid[0, :] = self.grid[-1, :] = WALL
        self.grid[:, 0] = self.grid[:, -1] = WALL
        # food distribution: biome 0 uniform; 1 clustered NE; 2 clustered SW
        self.food_center = ((S // 2, S // 2), (3, S - 4), (S - 4, 3))[b]
        self.food_spread = (6.0, 2.5, 2.5)[b]
        for _ in range(24):
            self._spawn_food()
        # agent start: any empty tile
        while True:
            r = int(torch.randint(1, S - 1, (1,), generator=self.gen))
            c = int(torch.randint(1, S - 1, (1,), generator=self.gen))
            if self.grid[r, c] == EMPTY:
                self.pos = (r, c)
                break

    def _spawn_food(self):
        S = self.SIZE
        for _ in range(64):
            r = int(torch.randn((), generator=self.gen) * self.food_spread
                    + self.food_center[0])
            c = int(torch.randn((), generator=self.gen) * self.food_spread
                    + self.food_center[1])
            if 0 < r < S - 1 and 0 < c < S - 1 and self.grid[r, c] == EMPTY:
                self.grid[r, c] = FOOD
                return

    def obs(self):
        r, c = self.pos
        return self.grid[r - 1:r + 2, c - 1:c + 2].reshape(-1).clone()

    def step(self, action: int):
        """Apply action; returns (reward, next_obs). Food regrows."""
        self.t += 1
        if self.t % self.biome_steps == 0:
            self._enter_biome(self.biome_idx + 1)
            return 0.0, self.obs()
        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        reward = 0.0
        if self.grid[r, c] != WALL:
            self.pos = (r, c)
            if self.grid[r, c] == FOOD:
                reward = 1.0
                self.grid[r, c] = EMPTY
                self._spawn_food()
        return reward, self.obs()

    def predict_true(self, action: int):
        """Oracle next-obs (for the oracle_mpc pilot arm): simulate without
        mutating state."""
        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if self.grid[r, c] == WALL:
            r, c = self.pos
        return self.grid[r - 1:r + 2, c - 1:c + 2].reshape(-1)


# CENTER-FIRST tile order: the first OBS token after [OBS] is the tile the
# agent now stands on. One forward per candidate action then yields exactly
# P(center = food | history, action) at the final position — the greedy
# foraging signal — without autoregressive rollout (E1_SPEC amendment A1).
CENTER_FIRST = [4, 0, 1, 2, 3, 5, 6, 7, 8]


def encode_frame(obs, action):
    tiles = [TOK_TILE[int(obs[i])] for i in CENTER_FIRST]
    return torch.tensor([TOK_OBS] + tiles
                        + [TOK_ACT, TOK_ACTION0 + action], dtype=torch.long)


def pick(scores, gen=None):
    """Argmax with UNIFORM RANDOM tie-breaking over near-max scores
    (E1_SPEC amendment A2, from the oracle pilot smoke: strict argmax
    degenerates to a fixed direction when all scores are flat-zero — the
    agent parks against a wall and loses to random walk. Ties broken
    randomly = random-walk exploration when the signal is flat, greedy
    foraging when it is not)."""
    t = torch.tensor(scores)
    mx = float(t.max())
    cand = (t >= mx - 1e-9).nonzero().reshape(-1)
    j = int(torch.randint(0, len(cand), (1,))) if gen is None else \
        int(torch.randint(0, len(cand), (1,), generator=gen))
    return int(cand[j])


def preference(center_probs):
    """FIXED preference functional (E1_SPEC amendment A1): the probability
    that the CENTER tile of the next view — the tile this action lands on —
    is food. center_probs: (V,) next-token distribution at the [OBS]+1
    position."""
    return float(center_probs[TOK_TILE[FOOD]])


class Agent:
    """The live loop, embodied: a rolling token context; acting = K candidate
    forwards; learning = gated fused kick on the realized OBS positions."""

    def __init__(self, args):
        ns = argparse.Namespace(d_model=args.d_model, n_layers=args.n_layers,
                                kernel=7, dropout=0.0, tau=0.5,
                                lam_min=2 ** -6)
        self.model = build_model("morphssm", VOCAB, ns)
        self.model.set_tau(0.0)
        self.model.eval()
        self.engine = MorphPCEngine(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                     weight_decay=0.01)
        self.gate = MarginGate(delta=args.delta)
        self.ug = UnigramCE(VOCAB) if args.gate == "excess" else None
        self.ctx = torch.zeros(args.window, dtype=torch.long)
        self.window = args.window

    def push(self, frame):
        self.ctx = torch.cat([self.ctx[len(frame):], frame])

    @torch.no_grad()
    def predict_obs(self, action):
        """One forward with the candidate action appended; returns the (V,)
        next-token distribution at the final position — semantically
        P(next frame's CENTER tile | history, action) under the center-first
        codec."""
        probe = torch.cat([self.ctx[FRAME:],
                           torch.tensor([TOK_ACT, TOK_ACTION0 + action,
                                         TOK_OBS], dtype=torch.long)])
        logits = self.model(probe.unsqueeze(0))[0]
        return torch.softmax(logits[-1], dim=-1)

    def act_and_learn(self, world, learn=True):
        """One embodied step. Returns (reward, fired, obs_ce)."""
        scores = [preference(self.predict_obs(a)) for a in range(5)]
        eps = torch.rand(())
        action = (int(torch.randint(0, 5, (1,))) if float(eps) < 0.05
                  else pick(scores))
        reward, next_obs = world.step(action)
        frame = encode_frame(next_obs, action)
        self.push(frame)
        # prequential obs-CE + gated kick on the realized frame
        x = self.ctx[:-1].unsqueeze(0)
        y = self.ctx[1:].unsqueeze(0)
        fired = False
        if learn:
            if self.ug is not None:
                ug_ce = self.ug.ce(y)
                fire_fn = (lambda ce, _u=ug_ce: self.gate(ce - _u))
            else:
                fire_fn = (lambda ce: self.gate(ce))
            logits, ce, fired = self.engine.serve_and_learn(
                x, y, opt=self.opt, fire=fire_fn)
            if self.ug is not None:
                self.ug.update(y)
        else:
            with torch.no_grad():
                lg = self.model(x)
                ce = float(torch.nn.functional.cross_entropy(
                    lg.reshape(-1, VOCAB).float(), y.reshape(-1)))
        return reward, fired, float(ce)


def run(args):
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    world = World(args.world_seed, biome_steps=args.biome_steps)
    tele = (OUT / f"e1_tele_{args.arm}_w{args.world_seed}_s{args.seed}.jsonl"
            ).open("w")
    t0 = time.time()
    total_r = fires = 0
    ce_sum = 0.0
    ret_window = []

    if args.arm == "random":
        for t in range(args.steps):
            r, _ = world.step(int(torch.randint(0, 5, (1,))))
            total_r += r
            ret_window.append(r)
            if (t + 1) % 1000 == 0:
                tele.write(json.dumps(dict(
                    t=t + 1, ret1k=sum(ret_window[-1000:]),
                    biome=world.biome)) + "\n")
    elif args.arm == "oracle_mpc":
        for t in range(args.steps):
            scores = []
            for a in range(5):
                nxt = world.predict_true(a)
                onehot = torch.zeros(VOCAB)
                onehot[TOK_TILE[int(nxt[4])]] = 1.0   # true landing (center)
                scores.append(preference(onehot))
            eps = float(torch.rand(()))
            a = (int(torch.randint(0, 5, (1,))) if eps < 0.05
                 else pick(scores))
            r, _ = world.step(a)
            total_r += r
            ret_window.append(r)
            if (t + 1) % 1000 == 0:
                tele.write(json.dumps(dict(
                    t=t + 1, ret1k=sum(ret_window[-1000:]),
                    biome=world.biome)) + "\n")
    else:                                    # live | frozen_mpc
        agent = Agent(args)
        learn = args.arm == "live"
        for t in range(args.steps):
            r, fired, ce = agent.act_and_learn(world, learn=learn)
            total_r += r
            fires += int(fired)
            ce_sum += ce
            ret_window.append(r)
            if (t + 1) % 1000 == 0:
                tele.write(json.dumps(dict(
                    t=t + 1, ret1k=sum(ret_window[-1000:]),
                    ce=round(ce_sum / (t + 1), 4),
                    fire_rate=round(fires / (t + 1), 4),
                    biome=world.biome)) + "\n")
                tele.flush()
    tele.close()

    row = dict(gate="E1", arm=args.arm, world_seed=args.world_seed,
               seed=args.seed, steps=args.steps,
               total_return=total_r,
               ret_per_1k=round(total_r / max(args.steps, 1) * 1000, 3),
               fire_rate=round(fires / max(args.steps, 1), 4),
               obs_ce=round(ce_sum / max(args.steps, 1), 4),
               wall_s=round(time.time() - t0, 1))
    with (OUT / "e1_world.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[E1] {args.arm}/w{args.world_seed}/s{args.seed}: "
          f"return/1k {row['ret_per_1k']} fire {row['fire_rate']} "
          f"obs_ce {row['obs_ce']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["live", "frozen_mpc", "random",
                                     "oracle_mpc"], required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--world-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--biome-steps", type=int, default=50_000)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gate", choices=["margin", "excess"], default="margin")
    p.add_argument("--delta", type=float, default=0.25)
    args = p.parse_args()
    run(args)
