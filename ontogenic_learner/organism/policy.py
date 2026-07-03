"""TwoSpeedPolicy — the G1/G3-validated learner, lifted verbatim (gate z>0.25 on a z-scored
batch-CE EMA; fast cascade kicks at lr 1e-3; sleep blocks on a surprise-weighted reservoir at
lr 3e-4). The ONE lifetime adaptation: the per-domain fast budget becomes a rolling cap
(fast kicks <= fast_rate_cap of the trailing fast_rate_window steps).

No new learning mathematics lives here."""
import math
import os
import sys
from collections import deque

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
from two_speed_learner import Reservoir


def onehot(xi, V):
    return F.one_hot(xi, V).float().reshape(xi.size(0), -1)


class TwoSpeedPolicy:
    def __init__(self, net, V, cfg, seed):
        self.net, self.V, self.cfg = net, V, cfg
        self.opt_fast = torch.optim.Adam(net.parameters(), lr=cfg.lr_fast)
        self.opt_slow = torch.optim.Adam(net.parameters(), lr=cfg.lr_slow)
        self.res = Reservoir(cfg.reservoir_cap, seed + 101)
        self.mu = self.var = None
        self.window = deque(maxlen=cfg.fast_rate_window)   # 1=fast kick, 0=skip
        self.n_fast = self.n_sleep = self.fwd_eq = 0

    # ---------------------------------------------------------------- surprise
    def surprise(self, loss: float) -> float:
        if self.mu is None:
            self.mu, self.var = loss, 1e-2
            return 0.0
        z = (loss - self.mu) / math.sqrt(self.var + 1e-8)
        self.mu = 0.99 * self.mu + 0.01 * loss
        self.var = 0.99 * self.var + 0.01 * (loss - self.mu) ** 2
        return z

    def rolling_rate(self) -> float:
        return (sum(self.window) / len(self.window)) if self.window else 0.0

    # ---------------------------------------------------------------- kicks
    def _kick(self, xb, yb, slow=False):
        self.net.train_step_cascade(xb, yb, self.opt_slow if slow else self.opt_fast,
                                    mode="dyadic")
        self.fwd_eq += 3

    def step(self, xi, yi, dom: str, life_step: int, device: str):
        """One stream step. Returns an event dict for the ledger (or None if pure skip)."""
        xb = onehot(xi, self.V).to(device)
        yb = yi.to(device)
        with torch.no_grad():
            loss = F.cross_entropy(self.net.predict_logits(xb), yb).item()
        self.fwd_eq += 1
        z = self.surprise(loss)
        fire = (z > self.cfg.gate_z or life_step < self.cfg.warm_steps) \
            and self.rolling_rate() < self.cfg.fast_rate_cap
        self.window.append(1 if fire else 0)
        if fire:
            self._kick(xb, yb)
            self.res.insert(xi, yi, prio=max(z, 0.1), dom=dom)
            self.n_fast += 1
            return {"kind": "fast", "dom": dom, "loss": round(loss, 4), "z": round(z, 3)}
        return {"kind": "skip", "dom": dom, "loss": round(loss, 4), "z": round(z, 3)} \
            if life_step % 200 == 0 else None            # skips ledgered sparsely

    def sleep_block(self, device: str):
        """One consolidation block (cfg.sleep_updates kicks on the reservoir)."""
        done = 0
        for _ in range(self.cfg.sleep_updates):
            rx, ry = self.res.sample(self.cfg.batch, weighted=True)
            if rx is None:
                break
            self._kick(onehot(rx, self.V).to(device), ry.to(device), slow=True)
            self.n_sleep += 1
            done += 1
        return done

    # ---------------------------------------------------------------- health
    def finite(self) -> bool:
        return all(torch.isfinite(p).all() for p in self.net.parameters())

    # ---------------------------------------------------------------- state
    def state(self):
        return {
            "opt_fast": self.opt_fast.state_dict(),
            "opt_slow": self.opt_slow.state_dict(),
            "res": {"x": torch.stack(self.res.x) if self.res.x else None,
                    "y": torch.stack(self.res.y) if self.res.y else None,
                    "p": list(self.res.p), "d": list(self.res.d),
                    "seen": self.res.seen,
                    "gen": self.res.gen.get_state()},
            "mu": self.mu, "var": self.var,
            "window": list(self.window),
            "n_fast": self.n_fast, "n_sleep": self.n_sleep, "fwd_eq": self.fwd_eq,
        }

    def restore(self, st):
        self.opt_fast.load_state_dict(st["opt_fast"])
        self.opt_slow.load_state_dict(st["opt_slow"])
        r = st["res"]
        self.res.x = list(r["x"]) if r["x"] is not None else []
        self.res.y = list(r["y"]) if r["y"] is not None else []
        self.res.p, self.res.d, self.res.seen = list(r["p"]), list(r["d"]), r["seen"]
        self.res.gen.set_state(r["gen"])
        self.mu, self.var = st["mu"], st["var"]
        self.window = deque(st["window"], maxlen=self.cfg.fast_rate_window)
        self.n_fast, self.n_sleep = st["n_fast"], st["n_sleep"]
        self.fwd_eq = st["fwd_eq"]
