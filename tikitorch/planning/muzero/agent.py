"""tikitorch.planning.muzero.agent — the full Tropical MuZero training loop.

Ties the tropical networks, the validated TropicalMCTS, self-play, and the replay
buffer into the standard MuZero loop: alternate (self-play -> replay) with
(sample -> K-step unrolled train). The backup operator is a swept hyperparameter
(`topk` k=2 default here; `mean` the published-MuZero safe option, `max` the pure
(max,+) bet). Losses: value and reward (MSE for a ScalarHead, categorical
cross-entropy for a CategoricalHead), policy (cross-entropy to the MCTS visits).

This is the CPU-scale vertical slice that validates the loop end-to-end; absolute
performance scales on EC2 (the learned-model fidelity is the ceiling, see Stage 4).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tikitorch.planning.mcts import MCTSConfig, TropicalMCTS
from tikitorch.planning.muzero.networks import TropicalMuZeroNet
from tikitorch.planning.muzero.networks_dense import DenseMuZeroNet
from tikitorch.planning.muzero.replay import ReplayBuffer
from tikitorch.planning.muzero.selfplay import play_game

__all__ = ["MuZeroConfig", "TropicalMuZero"]


@dataclass
class MuZeroConfig:
    net_kind: str = "tropical"    # "tropical" (multiplier-free) | "dense" (matched control)
    optimizer: str = "tropicalsgd"  # "tropicalsgd"/"signsgd" (closed, multiply-free) | "adamw" (open baseline)
    latent_dim: int = 64
    tau: float = 1.0
    tau_start: float = None       # if set with tau_end, anneal tau over training
    tau_end: float = None
    value_support: tuple = None   # e.g. (-20, 0) -> categorical value head (MuZero-style)
    reward_support: tuple = None  # e.g. (-1, 0)  -> categorical reward head
    # search
    num_simulations: int = 50
    c_puct: float = 1.25
    gamma: float = 1.0
    backup: str = "topk"          # "mean" | "topk" | "max"  (swept hyperparameter)
    topk_k: int = 2
    root_exploration_frac: float = 0.25
    dirichlet_alpha: float = 0.3
    # training
    unroll: int = 3
    td_steps: int = 5
    value_loss_weight: float = 0.25   # MuZero downweights value so it doesn't swamp policy
    batch: int = 128
    lr: float = 3e-3
    capacity: int = 2000
    # loop
    iters: int = 40
    games_per_iter: int = 8
    train_steps_per_iter: int = 20
    temp_start: float = 1.0
    temp_end: float = 0.1
    eval_every: int = 5
    random_starts: bool = True
    seed: int = 0


class TropicalMuZero:
    def __init__(self, env, config: MuZeroConfig = None):
        self.env = env
        self.cfg = config or MuZeroConfig()
        self.obs_dim = 3 * env.H * env.W
        net_cls = DenseMuZeroNet if self.cfg.net_kind == "dense" else TropicalMuZeroNet
        self.net = net_cls(self.obs_dim, env.n_actions,
                           latent_dim=self.cfg.latent_dim, tau=self.cfg.tau,
                           value_support=self.cfg.value_support,
                           reward_support=self.cfg.reward_support)
        self.buffer = ReplayBuffer(capacity=self.cfg.capacity, n_actions=env.n_actions)
        self.opt = _make_optimizer(self.cfg.optimizer, self.net.parameters(), self.cfg.lr)
        self.gen = torch.Generator().manual_seed(self.cfg.seed)

    # ---------------------------------------------------------------- training
    def train_step(self):
        c = self.cfg
        obs0, acts, vt, rt, pt = self.buffer.sample_batch(
            c.batch, c.unroll, c.td_steps, c.gamma, gen=self.gen)
        w = c.value_loss_weight
        vh, rh = self.net.pred.value_head, self.net.dyn.reward_head   # scalar or categorical
        h, p, v = self.net.initial(obs0)
        loss = w * vh.loss(v, vt[:, 0]) + _ce(p, pt[:, 0])
        for k in range(c.unroll):
            h, r, p, v = self.net.recurrent(h, acts[:, k])
            loss = (loss + w * vh.loss(v, vt[:, k + 1])
                    + rh.loss(r, rt[:, k]) + _ce(p, pt[:, k + 1]))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())

    # ---------------------------------------------------------------- self-play
    def _start_state(self, i):
        if not self.cfg.random_starts:
            return None
        # a random non-terminal start for data diversity
        while True:
            s = int(torch.randint(0, self.env.n_states, (1,), generator=self.gen).item())
            if not self.env.is_terminal_state(s):
                self.env.reset(s)
                return s

    def self_play(self, temperature, base_seed):
        c = self.cfg
        for i in range(c.games_per_iter):
            self._start_state(i)
            g = play_game(self.net, self.env, num_simulations=c.num_simulations,
                          c_puct=c.c_puct, gamma=c.gamma, backup=c.backup, topk_k=c.topk_k,
                          root_exploration_frac=c.root_exploration_frac,
                          dirichlet_alpha=c.dirichlet_alpha, temperature=temperature,
                          seed=base_seed + i)
            self.buffer.save(g)

    def _ensure_oracle(self):
        if getattr(self, "_oracle", None) is None:
            import tikitorch.planning as _P
            R, succ, _ = self.env.as_mdp()
            V, _ = _P.TropicalValueIteration(self.env.n_states, self.env.n_actions,
                                             gamma=1.0)(torch.zeros(self.env.n_states), R, succ)
            self._oracle = (V, R, succ)
        return self._oracle

    @torch.no_grad()
    def evaluate(self):
        """Greedy episode from the fixed start, plus robust learning metrics:
        optimal-action agreement vs value iteration over all states, and the
        fraction of non-terminal starts the greedy policy solves."""
        cfg = MCTSConfig(num_simulations=self.cfg.num_simulations, c_puct=self.cfg.c_puct,
                         gamma=self.cfg.gamma, backup=self.cfg.backup, topk_k=self.cfg.topk_k,
                         root_exploration_frac=0.0)
        mcts = TropicalMCTS(self.net, cfg)
        V, R, succ = self._ensure_oracle()

        def greedy_from(start):
            s, ret, steps, done = self.env.reset(start), 0.0, 0, False
            while not done and steps < 4 * self.env.n_states:
                vp, _ = mcts.run(self.env.obs(s).flatten())
                s, r, done = self.env.step_state(s, int(vp.argmax()))
                ret += r
                steps += 1
            return ret, done, steps

        hits = tot = solved_starts = 0
        for s in range(self.env.n_states):
            if self.env.is_terminal_state(s):
                continue
            tot += 1
            vp, _ = mcts.run(self.env.obs(s).flatten())
            best = V[s].item()
            opt = {a for a in range(self.env.n_actions)
                   if abs((R[s, a] + V[succ[s, a]]).item() - best) < 1e-5}
            if int(vp.argmax()) in opt:
                hits += 1
            if greedy_from(s)[1]:
                solved_starts += 1
        ret, done, steps = greedy_from(None)
        return dict(return_=ret, solved=done, steps=steps,
                    agree=hits / tot, solve_frac=solved_starts / tot)

    # ---------------------------------------------------------------- main loop
    def run(self, log=print):
        c = self.cfg
        history = []
        for it in range(c.iters):
            frac = it / max(c.iters - 1, 1)
            if c.tau_start is not None and c.tau_end is not None:
                self.net.set_tau(c.tau_start + (c.tau_end - c.tau_start) * frac)  # soft->hard anneal
            temp = c.temp_start + (c.temp_end - c.temp_start) * frac
            self.self_play(temperature=temp, base_seed=c.seed + it * 1000 + 1)
            losses = [self.train_step() for _ in range(c.train_steps_per_iter)] \
                if len(self.buffer) else []
            if it % c.eval_every == 0 or it == c.iters - 1:
                ev = self.evaluate()
                rec = dict(iter=it, games=len(self.buffer), buffer_steps=self.buffer.n_steps,
                           loss=(sum(losses) / len(losses) if losses else None), **ev)
                history.append(rec)
                log(f"iter {it:3d} | games {rec['games']:4d} | loss "
                    f"{(rec['loss'] if rec['loss'] is not None else float('nan')):.3f} | "
                    f"agree {ev['agree']:.0%} | solve_frac {ev['solve_frac']:.0%} | "
                    f"start {'SOLVED' if ev['solved'] else 'fail'}")
        return history


def _make_optimizer(name, params, lr):
    """TikiTorch closed (multiply-free) optimizer by default; AdamW for the open arm."""
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    import tikitorch.optim as ttoptim
    if name == "signsgd":
        return ttoptim.SignSGD(params, lr=lr)           # W -= lr*sign(grad)
    return ttoptim.TropicalSGD(params, lr=lr)            # closed: compare + add, zero DSP


def _ce(logits, target):
    """Cross-entropy to a soft (visit-distribution) target."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
