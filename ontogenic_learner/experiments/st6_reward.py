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
        self.use_head = getattr(args, "action_head", False)
        if self.use_head:
            # E4d A4: second readout on the shared trunk — same head class as
            # the LM head so the engine's scatter/adjoint path applies verbatim
            from tikitorch.nn import TropicalTopKModule
            self.model.action_head = TropicalTopKModule(
                args.d_model, self.codec.K, k=min(8, args.d_model))
        self.model.set_tau(0.0)
        self.model.eval()
        self.engine = MorphPCEngine(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                     weight_decay=0.01)
        self.gate = MarginGate(delta=args.delta)
        self.sal = RewardSalience() if args.gate == "N" else None
        self.tau_act = args.tau_act
        self.horizon = getattr(args, "horizon", 1)
        self.gamma = getattr(args, "gamma", 0.9)
        self.window = args.window
        self.ctx = torch.zeros(args.window, dtype=torch.long)
        # E3 token-policy machinery
        self.policy = getattr(args, "policy", "mpc")
        self.tau_pi = getattr(args, "tau_pi", 1.0)
        self.ctx_r = torch.zeros(args.window)      # per-token frame reward
        self.ctx_d = torch.zeros(args.window)      # per-token frame done-flag
        self.adv_baseline = 0.0                     # EMA of reward-to-go
        self.fire_always = getattr(args, "fire_always", False)
        # E4 ignition levers (E4_SPEC; all default OFF = the E3 polA path)
        self.intr_beta = getattr(args, "intr_beta", 0.0)   # (a) surprise reward
        self.ce_ema = None                                  #     slow CE baseline
        self.intr_sum = 0.0                                 #     shaped-mass audit
        self.adv_norm = getattr(args, "adv_norm", False)   # (b) A / sqrt(var)
        self.var_ema = 1.0                                  #     EMA of A^2
        self.mb_warmup = getattr(args, "mb_warmup", 0)     # (c) MPC birth actor
        self.t_step = 0                                     #     lifetime step
        self.ent_k = getattr(args, "ent_k", None)          # (d) 2^k uniform pull
        # E4d levers (E4D_SPEC; all default OFF = the E4b recipe path)
        self.obs_k = getattr(args, "obs_k", None)          # A1/A2 obs down-weight
        self.obs_frac = getattr(args, "obs_frac", None)    # A3 obs subsample
        self.obs_gen = torch.Generator().manual_seed(
            getattr(args, "seed", 0) + 4242)               #    subsample stream
        self.fire_anneal = getattr(args, "fire_anneal", None)  # B1 critical period
        self.adv_fire = getattr(args, "adv_fire", False)   # B2 advantage gate
        self.life_steps = getattr(args, "steps", 0)        #    for the anneal clock

    def push(self, frame, frame_reward=0.0, frame_done=False):
        self.ctx = torch.cat([self.ctx[len(frame):], frame])
        self.ctx_r = torch.cat([self.ctx_r[len(frame):],
                                torch.full((len(frame),), float(frame_reward))])
        self.ctx_d = torch.cat([self.ctx_d[len(frame):],
                                torch.full((len(frame),), float(frame_done))])

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

    @torch.no_grad()
    def _imagine(self, base, a1):
        """Greedy imagined rollout of the frame after action a1: append
        [ACT a1 RWD], read P(reward1) + argmax reward token, then autoregress
        the [OBS] + obs_len cells by argmax. Returns (imagined_ctx, P(reward1)).
        Cost = obs_len+1 forwards (the token-scale planning tax)."""
        c = self.codec
        seq = torch.cat([base, torch.tensor([c.ACT, c.ACT0 + a1, c.RWD])])
        lg = self.model(seq[-self.window:].unsqueeze(0))[0][-1]
        p_r1 = float(torch.softmax(lg, -1)[c.RWD0 + 1])
        # slice-constrained argmax: imagined tokens must stay in their slot's
        # valid range, else a stray argmax builds a malformed frame silently
        r_tok = c.RWD0 + int(lg[c.RWD0:c.RWD0 + 2].argmax())
        seq = torch.cat([seq, torch.tensor([r_tok, c.OBS])])
        for _ in range(c.obs_len):
            lo = self.model(seq[-self.window:].unsqueeze(0))[0][-1]
            o = c.SYM0 + int(lo[c.SYM0:].argmax())
            seq = torch.cat([seq, torch.tensor([o])])
        return seq, p_r1

    @torch.no_grad()
    def score_actions_h2(self):
        """2-step lookahead: score(a1) = P(reward1|a1) + gamma * max_a2
        P(reward2 | imagined-frame(a1), a2). Faithful model rollout (imagines
        the next obs frame); the token-scale cost is K*(obs_len+1) + K^2
        forwards/step (~15x foraging, ~100x minatar vs greedy)."""
        c = self.codec
        out = []
        for a1 in range(c.K):
            imag, p_r1 = self._imagine(self.ctx, a1)
            # window-crop the imagined context so the a2 probe length equals
            # the H1 probe length — else H1-vs-H2 is confounded by context
            # length (OOD 110 vs 96), not horizon
            base2 = imag[-(self.window - 3):]
            probes = torch.stack([
                torch.cat([base2, torch.tensor([c.ACT, c.ACT0 + a2, c.RWD])])
                for a2 in range(c.K)])
            lg2 = self.model(probes)[:, -1]
            p_r2 = torch.softmax(lg2, -1)[:, c.RWD0 + 1].max()
            out.append(p_r1 + self.gamma * float(p_r2))
        return out

    @torch.no_grad()
    def act_token(self):
        """E3: the model's own next-action-token distribution IS the policy.
        One forward: probe = ctx[1:] + [ACT]; sample the action token at
        temperature tau_pi (restricted to the K action ids)."""
        c = self.codec
        probe = torch.cat([self.ctx[1:], torch.tensor([c.ACT])])
        if self.use_head:
            lg = self.engine.act_probe_logits(probe.unsqueeze(0))[0][-1]
            pi = torch.softmax(lg.float() / self.tau_pi, dim=0)
        else:
            lg = self.model(probe.unsqueeze(0))[0][-1]
            pi = torch.softmax(lg[c.ACT0:c.ACT0 + c.K] / self.tau_pi, dim=0)
        return int(torch.multinomial(pi, 1))

    @torch.no_grad()
    def pol_entropy(self):
        """Entropy of the current action marginal (attractor-state telemetry)."""
        c = self.codec
        probe = torch.cat([self.ctx[1:], torch.tensor([c.ACT])])
        if self.use_head:
            lg = self.engine.act_probe_logits(probe.unsqueeze(0))[0][-1]
            pi = torch.softmax(lg.float() / self.tau_pi, dim=0)
        else:
            lg = self.model(probe.unsqueeze(0))[0][-1]
            pi = torch.softmax(lg[c.ACT0:c.ACT0 + c.K] / self.tau_pi, dim=0)
        return float(-(pi * pi.clamp_min(1e-9).log()).sum())

    def act(self):
        if self.policy == "token":
            # E4 mbwarm: decaying mixture with the model-based (MPC) actor —
            # the enumerative P(a|reward+) posterior seeds reward-directed
            # exploration before REINFORCE has anything to compound
            if (self.mb_warmup and self.t_step < self.mb_warmup
                    and float(torch.rand(1)) < 1.0 - self.t_step / self.mb_warmup):
                scores = self.score_actions()
                if scores is not None:
                    t = torch.tensor(scores)
                    probs = torch.softmax((t - t.mean()) / self.tau_act, dim=0)
                    return int(torch.multinomial(probs, 1))
            return self.act_token()
        if self.codec.objective != "R":                     # reward-free -> random
            return int(torch.randint(0, self.codec.K, (1,)))
        scores = (self.score_actions_h2() if self.horizon >= 2
                  else self.score_actions())
        t = torch.tensor(scores)
        probs = torch.softmax((t - t.mean()) / self.tau_act, dim=0)
        return int(torch.multinomial(probs, 1))

    def step(self, learn=True):
        action = self.act()
        self.t_step += 1
        reward, obs_sym, done = self.env.step(action)
        self.push(self.codec.frame(obs_sym, action, reward),
                  frame_reward=reward, frame_done=done)
        x = self.ctx[:-1].unsqueeze(0)
        y = self.ctx[1:].unsqueeze(0)
        c = self.codec
        mask = c.learn_mask(y)
        if self.policy == "token":
            # E3: action tokens join the loss, weighted by dyadic advantage
            act_pos = (y >= c.ACT0) & (y < c.ACT0 + c.K)
            mask = mask | act_pos
            sgn = torch.ones_like(y, dtype=torch.int8)
            kex = torch.zeros_like(y, dtype=torch.int32)
            idx = act_pos[0].nonzero().reshape(-1)
            if len(idx):
                r = self.ctx_r[1:][idx]                       # frame rewards, in order
                d = self.ctx_d[1:][idx]                       # frame done-flags
                G = torch.zeros_like(r)
                acc = 0.0
                for i in range(len(r) - 1, -1, -1):           # reward-to-go; done
                    acc = float(r[i]) + self.gamma * acc * (1.0 - float(d[i]))
                    G[i] = acc                                # zeroes cross-episode credit
                A = G - self.adv_baseline
                self.adv_baseline = 0.99 * self.adv_baseline + 0.01 * float(G.mean())
                if self.adv_norm:
                    # E4 norm: unit-scale the advantage BEFORE quantizing so
                    # small consistent signals reach weight ~1 (normalize by
                    # the pre-update estimate, then update from the same
                    # pre-normalization A the kick uses)
                    a_raw2 = float(A.pow(2).mean())
                    A = A / (self.var_ema + 1e-4) ** 0.5
                    self.var_ema = 0.99 * self.var_ema + 0.01 * a_raw2
                mag = A.abs()
                # clamp INSIDE the live branch: clamping the -99 drop-sentinel
                # would resurrect |A|<2^-5 positions at k=-4 with sign(A<=0)=-1
                # — a systematic negative push on every neutral action
                k = torch.where(mag > 2 ** -5,
                                torch.log2(mag.clamp_min(2 ** -5))
                                .round().clamp(-4, 2),
                                torch.full_like(mag, -99.0))
                for j, p_ in enumerate(idx.tolist()):
                    if k[j] <= -99:
                        sgn[0, p_] = 0                        # |A| too small: drop
                    else:
                        sgn[0, p_] = 1 if A[j] > 0 else -1
                        kex[0, p_] = int(k[j])
            if self.use_head:
                # E4d A4: ALL policy gradient goes through the action head;
                # the LM head's loss drops action positions (E1 rule restored
                # — the whole point is blocking the shared-head path)
                self.engine.action_loss = (
                    act_pos, (y - c.ACT0).clamp(0, c.K - 1), sgn, kex)
                mask = c.learn_mask(y)
                sgn = torch.ones_like(y, dtype=torch.int8)
                kex = torch.zeros_like(y, dtype=torch.int32)
            if self.obs_k is not None:
                # E4d A1/A2: dyadic down-weight of obs-position errors — the
                # direct gradient-volume rebalance (~200:2 -> ~200/2^|k|:2)
                obs_m = c.obs_mask(y)
                kex = torch.where(obs_m,
                                  torch.full_like(kex, self.obs_k), kex)
            self.engine.loss_dyadic = (sgn, kex)
            has_adv = bool(((sgn != 0) & act_pos).any()) if not self.use_head \
                else bool(((self.engine.action_loss[2] != 0) & act_pos).any())
            if self.ent_k is not None:
                # E4 ent: always-on 2^k uniform pull at action positions
                # (mask ⊆ loss_mask holds: act_pos was OR-ed into mask above)
                self.engine.loss_uniform_pull = (act_pos, self.ent_k,
                                                 c.ACT0, c.ACT0 + c.K)
        else:
            has_adv = False
        if self.obs_frac is not None:
            # E4d A3: keep a seeded random fraction of obs positions in the
            # LOSS only (gate_mask untouched -> gate CE stays comparable)
            obs_m = c.obs_mask(y)
            keep = torch.rand(y.shape, generator=self.obs_gen) < self.obs_frac
            mask = mask & (~obs_m | keep)
        self.engine.loss_mask = mask
        self.engine.gate_mask = c.obs_mask(y)                 # gate on obs-only
        fired = False
        if learn:
            def fire_fn(ce):
                # E4d B1: critical period — fire-always for the first
                # fire_anneal fraction of life, gated after
                always = self.fire_always or (
                    self.fire_anneal is not None
                    and self.t_step <= self.fire_anneal * self.life_steps)
                if always:
                    self.gate(ce)                             # keep EMAs live
                    if self.sal is not None:
                        self.sal.salient(abs(reward))
                    return True
                salient = (self.sal.salient(abs(reward))      # EVAL FIRST so the
                           if self.sal is not None else True)  # baseline always
                surprise = self.gate(ce)                       # updates
                if self.adv_fire and has_adv:
                    # E4d B2: advantage pulses fire the kick regardless of
                    # obs surprise — the policy-aligned gate signal (E4c:
                    # obs surprise self-quiets exactly when policy learning
                    # still needs kicks). Gate EMAs updated above either way.
                    return True
                return surprise and salient                    # neuromod AND
            lg_serve, ce, fired = self.engine.serve_and_learn(
                x, y, opt=self.opt, fire=fire_fn)
            if self.intr_beta and self.policy == "token":
                # E4 intr: the newest frame's obs surprise (excess over a slow
                # EMA, clipped at 0) is back-filled as intrinsic CREDIT on that
                # frame — ctx_r only; the [RWD] token, salience EMA, and all
                # reported returns stay extrinsic (E4_SPEC)
                flen = self.codec.frame_len
                yl = y[0, -flen:]
                om = yl >= self.codec.SYM0
                with torch.no_grad():
                    lf = lg_serve[0, -flen:].float()[om]
                    ce_new = float(F.cross_entropy(lf, yl[om]))
                if self.ce_ema is None:
                    self.ce_ema = ce_new
                excess = max(ce_new - self.ce_ema, 0.0)
                self.ce_ema = 0.999 * self.ce_ema + 0.001 * ce_new
                r_int = self.intr_beta * excess
                if r_int > 0:
                    self.ctx_r[-flen:] = self.ctx_r[-flen:] + r_int
                    self.intr_sum += r_int
        else:
            with torch.no_grad():
                lg = self.model(x).reshape(-1, self.codec.vocab).float()
                keep = self.engine.gate_mask.reshape(-1)      # obs-only, comparable
                ce = float(F.cross_entropy(lg[keep], y.reshape(-1)[keep]))
            if self.sal is not None:                          # keep baseline live
                self.sal.salient(abs(reward))
        self.engine.loss_mask = None
        self.engine.gate_mask = None
        self.engine.loss_dyadic = None
        self.engine.loss_uniform_pull = None
        self.engine.action_loss = None
        return reward, fired, ce


def run(args):
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env(args.env, world_seed=args.world_seed,
                   biome_steps=args.biome_steps)
    arm = f"{args.gate}{args.objective}" if args.mode == "agent" else args.mode
    if getattr(args, "policy", "mpc") == "token":
        arm = ("polA" if getattr(args, "fire_always", False) else "polG") \
            if args.mode == "agent" else f"{args.mode}Tok"
        # E4 lever suffix (bare polA == the E3 control/sentinel)
        if getattr(args, "intr_beta", 0.0):
            arm += f"_intr{args.intr_beta}"
        if getattr(args, "adv_norm", False):
            arm += "_norm"
        if getattr(args, "mb_warmup", 0):
            arm += f"_mb{args.mb_warmup}"
        if getattr(args, "ent_k", None) is not None:
            arm += f"_entk{args.ent_k}"
        if getattr(args, "obs_k", None) is not None:
            arm += f"_obsk{args.obs_k}"
        if getattr(args, "obs_frac", None) is not None:
            arm += f"_sub{args.obs_frac}"
        if getattr(args, "fire_anneal", None) is not None:
            arm += f"_fa{args.fire_anneal}"
        if getattr(args, "adv_fire", False):
            arm += "_advf"
        if getattr(args, "action_head", False):
            arm += "_head"
    # disambiguate tele by delta+horizon so parallel sweep cells never collide
    # (V2b lesson): a d/h tag keeps every cell's telemetry in its own file.
    dh = f"_d{args.delta}h{getattr(args, 'horizon', 1)}w{args.window}"
    tele = (OUT / f"e2_tele_{args.env}_{arm}{dh}_w{args.world_seed}_s{args.seed}.jsonl"
            ).open("w")
    t0 = time.time()
    total_r = fires = 0
    ce_sum = 0.0
    win = []
    pi0_ent = None

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
        token_pol = getattr(args, "policy", "mpc") == "token"
        for step in range(args.prelife_steps):
            agent.model.set_tau(tau_at(step, args.prelife_steps, 0.5, True, 0.0))
            st = int(torch.randint(0, ids.numel() - args.window - 1, (1,),
                                   generator=gb))
            xb = ids[st:st + args.window].unsqueeze(0)
            yb = ids[st + 1:st + args.window + 1].unsqueeze(0)
            m = agent.codec.learn_mask(yb)
            if token_pol:
                # E3 amendment A1: behaviour-clone the random birth policy
                # (action tokens at weight 1) so the tau_pi action marginal
                # starts calibrated ~uniform instead of an init lottery; the
                # frozen floor shares this birth, so comparisons stay fair
                c0 = agent.codec
                ap = (yb >= c0.ACT0) & (yb < c0.ACT0 + c0.K)
                if getattr(agent, "use_head", False):
                    # E4d head arm: the policy IS the head — BC the random
                    # birth policy THROUGH the head (review launch-blocker:
                    # cloning into the LM head would leave the acting head
                    # at random init, the exact A1 lottery)
                    agent.engine.action_loss = (
                        ap, (yb - c0.ACT0).clamp(0, c0.K - 1),
                        torch.ones_like(yb, dtype=torch.int8),
                        torch.zeros_like(yb, dtype=torch.int32))
                else:
                    m = m | ap
            agent.engine.loss_mask = m
            agent.engine.train_step(xb, yb, agent.opt, credit="cascade")
            agent.engine.action_loss = None
        agent.engine.loss_mask = None
        agent.model.set_tau(0.0)
        agent.model.eval()
        agent.opt = torch.optim.AdamW(agent.model.parameters(), lr=args.lr,
                                      weight_decay=0.01)
        agent.gate = MarginGate(delta=args.delta)
        # warm the ctx with real frames (reward/done flow into ctx_r/ctx_d so
        # the first windows' reward-to-go and the [RWD] tokens agree)
        for _ in range(args.window // agent.codec.frame_len + 1):
            a = int(torch.randint(0, env.n_actions, (1,)))
            r, o, dn = env.step(a)
            agent.push(agent.codec.frame(o, a, r), frame_reward=r, frame_done=dn)
        if token_pol:
            pi0_ent = agent.pol_entropy()           # birth-policy audit
                                                    # (head-aware)
        else:
            pi0_ent = None
        learn = args.mode == "agent"
        for t in range(args.steps):
            r, fired, ce = agent.step(learn=learn)
            total_r += r
            fires += int(fired)
            ce_sum += ce
            win.append(r)
            if (t + 1) % 1000 == 0:
                rec = dict(
                    t=t + 1, ret1k=sum(win[-1000:]), ce=round(ce_sum / (t + 1), 4),
                    fire_rate=round(fires / (t + 1), 4), biome=env.biome)
                if agent.policy == "token":
                    rec["piH"] = round(agent.pol_entropy(), 4)   # attractor state
                if agent.intr_beta:
                    rec["intr_sum"] = round(agent.intr_sum, 2)   # shaped-mass audit
                tele.write(json.dumps(rec) + "\n")
                tele.flush()
    tele.close()

    row = dict(gate="E2", env=args.env, arm=arm, delta=args.delta,
               window=args.window,
               policy=getattr(args, "policy", "mpc"),
               horizon=getattr(args, "horizon", 1),
               world_seed=args.world_seed,
               seed=args.seed, steps=args.steps, total_return=total_r,
               ret_per_1k=round(total_r / max(args.steps, 1) * 1000, 3),
               fire_rate=round(fires / max(args.steps, 1), 4),
               obs_ce=round(ce_sum / max(args.steps, 1), 4),
               pi0_ent=None if pi0_ent is None else round(pi0_ent, 4),
               intr_sum=None if args.mode == "random" else round(
                   getattr(locals().get("agent"), "intr_sum", 0.0), 2),
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
    p.add_argument("--horizon", type=int, default=1, choices=[1, 2])
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--policy", choices=["mpc", "token"], default="mpc")
    p.add_argument("--tau-pi", type=float, default=1.0)
    p.add_argument("--fire-always", action="store_true")
    # E4 ignition levers (E4_SPEC; defaults = E3 polA exactly)
    p.add_argument("--intr-beta", type=float, default=0.0)
    p.add_argument("--adv-norm", action="store_true")
    p.add_argument("--mb-warmup", type=int, default=0)
    p.add_argument("--ent-k", type=int, default=None)
    # E4d levers (E4D_SPEC; defaults = the E4b recipe exactly)
    p.add_argument("--obs-k", type=int, default=None)
    p.add_argument("--obs-frac", type=float, default=None)
    p.add_argument("--fire-anneal", type=float, default=None)
    p.add_argument("--adv-fire", action="store_true")
    p.add_argument("--action-head", action="store_true")
    p.add_argument("--prelife-frames", type=int, default=8_000)
    p.add_argument("--prelife-steps", type=int, default=1_500)
    args = p.parse_args()
    run(args)
