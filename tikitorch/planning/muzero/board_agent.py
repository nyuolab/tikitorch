"""tikitorch.planning.muzero.board_agent — learned two-player MuZero for board games.

The Connect4 (and TicTacToe / chess-later) training loop: two-player self-play via
negamax MCTS over the learned net -> a board replay buffer (positions labelled with
the game OUTCOME) -> K-step unrolled training (value->outcome, policy->MCTS visits,
reward->0). Skill is measured as win/draw/loss vs a fixed depth-limited negamax
opponent. Fully TikiTorch-native: tropical nets + closed optimizer (TropicalSGD);
``mean`` backup (the verified adversarial-correct backup); ``net_kind="dense"`` is
the matched control for the head-to-head.
"""

from dataclasses import dataclass

import torch

from tikitorch.planning.mcts import MCTSConfig, TropicalMCTS
from tikitorch.planning.muzero.networks import TropicalMuZeroNet
from tikitorch.planning.muzero.networks_dense import DenseMuZeroNet
from tikitorch.planning.muzero.games import BoardGameModel, negamax_agent, play_match, random_opening
from tikitorch.planning.muzero.selfplay import play_board_game
from tikitorch.planning.muzero.agent import _make_optimizer, _ce

__all__ = ["BoardConfig", "BoardReplay", "BoardMuZero", "build_muzero_net"]


def build_muzero_net(net_kind, obs_dim, n_actions, latent_dim, tau,
                     value_support, reward_support, dense_hidden_mult, core_kind="maxplus"):
    kw = dict(latent_dim=latent_dim, tau=tau, value_support=value_support,
              reward_support=reward_support)
    if net_kind == "dense":
        return DenseMuZeroNet(obs_dim, n_actions, hidden_mult=dense_hidden_mult, **kw)
    return TropicalMuZeroNet(obs_dim, n_actions, core_kind=core_kind, **kw)


# --- parallel self-play workers (the MuZero "actor" pattern; module-level for pickling) ---
_W = {}


def _init_worker(blueprint, state_dict, game, sp_kwargs):
    import torch as _t
    _t.set_num_threads(1)                       # one thread/worker; parallelism is across games
    net = build_muzero_net(**blueprint)
    net.load_state_dict(state_dict)
    net.eval()
    _W.update(net=net, game=game, sp=sp_kwargs)


def _run_worker(seed):
    records, _ = play_board_game(_W["net"], _W["game"], seed=seed, **_W["sp"])
    return records


@dataclass
class BoardConfig:
    net_kind: str = "tropical"
    core_kind: str = "maxplus"          # tropical cores: "maxplus" | "dyadic" (multiplier-free oblique)
    optimizer: str = "tropicalsgd"      # closed (multiply-free) by default
    latent_dim: int = 64
    dense_hidden_mult: int = 3          # widen dense cores to param-match the tropical net
    tau: float = 1.0
    tau_start: float = 1.0              # soft->hard anneal (verified recipe)
    tau_end: float = 0.1
    value_support: tuple = None
    reward_support: tuple = None
    # search
    num_simulations: int = 50
    c_puct: float = 1.25
    backup: str = "mean"               # adversarial-correct (max/topk over-optimistic)
    root_exploration_frac: float = 0.25
    dirichlet_alpha: float = 0.3
    # training
    unroll: int = 3
    value_loss_weight: float = 0.25
    batch: int = 128
    lr: float = 3e-3
    capacity: int = 3000
    num_workers: int = 1               # >1 = parallel self-play across this many processes
    # loop
    iters: int = 40
    games_per_iter: int = 16
    train_steps_per_iter: int = 20
    temp_start: float = 1.0
    temp_end: float = 0.15
    eval_every: int = 5
    eval_games: int = 20
    eval_depth: int = 2                # depth of the negamax skill opponent
    eval_open_plies: int = 4           # random opening plies per eval game (0 = fixed start; see below)
    seed: int = 0


class BoardReplay:
    """Stores self-play games (lists of ``(obs, action, policy, value_target)``)."""

    def __init__(self, capacity, n_actions):
        self.capacity = capacity
        self.A = n_actions
        self.games = []

    def save(self, records):
        if records:
            self.games.append(records)
        if len(self.games) > self.capacity:
            self.games.pop(0)

    def __len__(self):
        return len(self.games)

    def sample_batch(self, batch, unroll, gen):
        """obs0 (B,obs); acts (B,K) long; value_targets (B,K+1); policy_targets
        (B,K+1,A). Reward target is 0 throughout (board games). Past game end is
        padded absorbing (value 0, action 0, uniform policy)."""
        A = self.A
        lengths = [len(g) for g in self.games]
        tot = sum(lengths)
        obs0, acts, vts, pts = [], [], [], []
        for _ in range(batch):
            r = int(torch.randint(0, max(tot, 1), (1,), generator=gen).item())
            gi = 0
            while r >= lengths[gi]:
                r -= lengths[gi]
                gi += 1
            g, pos = self.games[gi], r
            obs0.append(g[pos][0])
            a_k, v_k, p_k = [], [g[pos][3]], [g[pos][2]]
            for k in range(1, unroll + 1):
                p = pos + k
                a_k.append(g[p - 1][1] if p - 1 < len(g) else 0)
                v_k.append(g[p][3] if p < len(g) else 0.0)
                p_k.append(g[p][2] if p < len(g) else torch.full((A,), 1.0 / A))
            acts.append(torch.tensor(a_k, dtype=torch.long))
            vts.append(torch.tensor(v_k))
            pts.append(torch.stack(p_k))
        return torch.stack(obs0), torch.stack(acts), torch.stack(vts), torch.stack(pts)


class BoardMuZero:
    def __init__(self, game, config: BoardConfig = None):
        self.game = game
        self.cfg = config or BoardConfig()
        self.current_tau = self.cfg.tau
        self._bp = dict(net_kind=self.cfg.net_kind, obs_dim=game.obs_dim, n_actions=game.n_actions,
                        latent_dim=self.cfg.latent_dim, tau=self.cfg.tau,
                        value_support=self.cfg.value_support, reward_support=self.cfg.reward_support,
                        dense_hidden_mult=self.cfg.dense_hidden_mult, core_kind=self.cfg.core_kind)
        self.net = build_muzero_net(**self._bp)
        self.opt = _make_optimizer(self.cfg.optimizer, self.net.parameters(), self.cfg.lr)
        self.buffer = BoardReplay(self.cfg.capacity, game.n_actions)
        self.gen = torch.Generator().manual_seed(self.cfg.seed)

    # ---------------------------------------------------------------- self-play
    def self_play(self, temperature, base_seed):
        c = self.cfg
        sp = dict(num_simulations=c.num_simulations, c_puct=c.c_puct, backup=c.backup,
                  root_exploration_frac=c.root_exploration_frac,
                  dirichlet_alpha=c.dirichlet_alpha, temperature=temperature)
        if c.num_workers <= 1:
            for i in range(c.games_per_iter):
                records, _ = play_board_game(self.net, self.game, seed=base_seed + i, **sp)
                self.buffer.save(records)
            return
        # parallel actors: rebuild the current net once per worker, fan games across them
        import multiprocessing as mp
        bp = dict(self._bp)
        bp["tau"] = self.current_tau
        sd = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        seeds = range(base_seed, base_seed + c.games_per_iter)
        with mp.Pool(c.num_workers, initializer=_init_worker, initargs=(bp, sd, self.game, sp)) as pool:
            for records in pool.imap_unordered(_run_worker, seeds):
                self.buffer.save(records)

    # ---------------------------------------------------------------- training
    def train_step(self):
        c = self.cfg
        obs0, acts, vt, pt = self.buffer.sample_batch(c.batch, c.unroll, self.gen)
        vh = self.net.pred.value_head
        h, p, v = self.net.initial(obs0)
        loss = c.value_loss_weight * vh.loss(v, vt[:, 0]) + _ce(p, pt[:, 0])
        for k in range(c.unroll):
            h, r, p, v = self.net.recurrent(h, acts[:, k])
            loss = loss + c.value_loss_weight * vh.loss(v, vt[:, k + 1]) + _ce(p, pt[:, k + 1])
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())

    # ---------------------------------------------------------------- evaluation
    @torch.no_grad()
    def _mcts_agent(self):
        cfg = MCTSConfig(num_simulations=self.cfg.num_simulations, c_puct=self.cfg.c_puct,
                         gamma=1.0, backup=self.cfg.backup, players=2, root_exploration_frac=0.0)
        mcts = TropicalMCTS(BoardGameModel(self.net, self.game), cfg)
        return lambda state: int(mcts.run(state)[0].argmax())

    @torch.no_grad()
    def evaluate(self):
        """Win/draw/loss vs a depth-limited negamax opponent, from RANDOM OPENINGS so the games are
        distinct. Both the MCTS agent (root noise off) and negamax are deterministic, so playing from
        the single fixed start collapses to ~2 games (one per color) replayed eval_games times — a
        silent artifact that quantizes the score to {0,0.5,1} and is blind to gradual skill. Starting
        each game from eval_open_plies random plies (alternating which color the agent plays) gives
        eval_games genuinely distinct games and a smooth skill estimate."""
        me = self._mcts_agent()
        opp = negamax_agent(self.game, self.cfg.eval_depth)
        rng = torch.Generator().manual_seed(self.cfg.seed + 99991)
        w = d = losses = 0
        for g in range(self.cfg.eval_games):
            start = (random_opening(self.game, rng, self.cfg.eval_open_plies)
                     if self.cfg.eval_open_plies else None)
            my_id = 1 if g % 2 == 0 else 2                 # alternate the agent's color for fairness
            agents = {my_id: me, 3 - my_id: opp}
            res = play_match(agents[1], agents[2], self.game, start_state=start)
            w += res == my_id
            d += res == 0
            losses += (res != 0 and res != my_id)
        n = self.cfg.eval_games
        return dict(win=w / n, draw=d / n, loss=losses / n, score=(w + 0.5 * d) / n)

    # ---------------------------------------------------------------- main loop
    def run(self, log=print):
        c = self.cfg
        history = []
        self.best_score = -1.0
        self.best_state = None
        for it in range(c.iters):
            frac = it / max(c.iters - 1, 1)
            if c.tau_start is not None and c.tau_end is not None:
                self.current_tau = c.tau_start + (c.tau_end - c.tau_start) * frac
                self.net.set_tau(self.current_tau)
            temp = c.temp_start + (c.temp_end - c.temp_start) * frac
            self.self_play(temperature=temp, base_seed=c.seed + it * 1000 + 1)
            losses = [self.train_step() for _ in range(c.train_steps_per_iter)] if len(self.buffer) else []
            if it % c.eval_every == 0 or it == c.iters - 1:
                ev = self.evaluate()
                # MuZero training oscillates -> keep the BEST-scoring snapshot (the
                # deployable agent), not the possibly-collapsed final one.
                if ev["score"] > self.best_score:
                    self.best_score = ev["score"]
                    self.best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                tl = (sum(losses) / len(losses)) if losses else None
                rec = dict(iter=it, games=len(self.buffer), train_loss=tl, best=self.best_score, **ev)
                history.append(rec)
                log(f"iter {it:3d} | games {rec['games']:4d} | loss "
                    f"{(tl if tl is not None else float('nan')):.3f} | "
                    f"vs negamax(d{c.eval_depth}): win {ev['win']:.0%} draw {ev['draw']:.0%} "
                    f"loss {ev['loss']:.0%} | score {ev['score']:.2f} | best {self.best_score:.2f}")
        if self.best_state is not None:            # deploy the best snapshot, not the final
            self.net.load_state_dict(self.best_state)
        return history
