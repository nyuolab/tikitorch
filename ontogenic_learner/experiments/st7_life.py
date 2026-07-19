"""E5 — the second life (E5_SPEC.md): a kill-9-resumable always-on embodied
agent living alternating foraging/minatar eras with revisits, on the E4b
recipe (+E4d winners via config). One birth; envs never reset; retention,
transfer and interference measured by clone-isolated probes (st7_probes).

Design authority: the approved plan + the Plan-agent architecture report
(both in E5_SPEC.md). Scaffolding imported verbatim from organism/
(save_atomic, load_latest, Ledger); loop/guard patterns ported (not
imported) from organism/daemon.py; era rotation + config-identity resume
ported from st4_life.py. st6_reward.py stays frozen as the E2-E4 record —
LifeAgent subclasses it.

Usage::
    python ontogenic_learner/experiments/st7_life.py --workdir /tmp/e5 --smoke
    python ontogenic_learner/experiments/st7_life.py --workdir /opt/e5 \
        --config pilot.json [--resume]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pc_convrnn_charlm import MorphPCEngine, build_model      # noqa: E402
from st2_live import MarginGate                               # noqa: E402
from st5_envs import make_env                                 # noqa: E402
from st6_reward import Agent                                  # noqa: E402
from nanogpt_shakespeare import tau_at                        # noqa: E402
from ontogenic_learner.organism.state import (                # noqa: E402
    save_atomic, load_latest)
from ontogenic_learner.organism.ledger import Ledger          # noqa: E402
from st7_probes import env_digest as _env_digest              # noqa: E402


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULTS = dict(
    # eras (ratios frozen in E5_SPEC; absolute scale rescaled once by pilot)
    era_steps=500_000, biome_steps=125_000, cycles=10,
    env_order=["foraging", "minatar"],
    # agent (the recipe; E4d winners set fire_schedule / mech_lever)
    d_model=128, n_layers=4, window=96, lr=1e-3,
    delta_foraging=0.10, delta_minatar=0.05,
    tau_pi=1.0, gamma=0.9, intr_beta=0.5, adv_norm=True,
    fire_schedule="always",        # always | anneal | advfire (E4d winner)
    fire_anneal_frac=0.25,         # used when fire_schedule == "anneal"
    mech_lever="none",             # none | obsk-2 | obsk-3 | sub25 | head
    # birth
    prelife_frames=8_000, prelife_steps=1_500, world_seed=0, seed=0,
    # lr decay: halve per N completed cycles, floored
    lr_halve_every_cycles=2, lr_floor_div=16,
    # ops
    ckpt_every_steps=50_000, ckpt_every_secs=900, nan_check_every=1_000,
    heartbeat_secs=60, s3_sync_secs=300, probe_every_hours=6.0,
    budget_wall_hours=0.0,         # 0 = uncapped; >0 = graceful idle at cap
    # reserved third world (capacity allocated at birth even if unused)
    babyai_syms=20, babyai_from_cycle=6, babyai_enabled=False,
)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def env_code_hash() -> str:
    h = hashlib.sha256()
    for f in ("ontogenic_learner/experiments/st5_envs.py",
              "ontogenic_learner/experiments/st5_world.py"):
        h.update((REPO / f).read_bytes())
    import minatar
    h.update(getattr(minatar, "__version__", "?").encode())
    h.update(torch.__version__.encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# UnifiedCodec — one token space for every world the life may ever touch
# --------------------------------------------------------------------------

class UnifiedCodec:
    """Shared [ACT][OBS][RWD] + shared action block (union K) + shared reward
    block + DISJOINT per-env obs-symbol blocks (transfer must flow through
    shared structure; interference is measurable at disjoint output rows).
    The vocab — INCLUDING the reserved BabyAI block — is fixed at birth.

    st6's learn_mask / obs_mask expressions hold verbatim because every obs
    block sits contiguously at the top of the vocab (>= SYM0)."""

    K = 7                                   # union: MiniGrid's 7 (E5_SPEC)
    ENV_K = {"foraging": 5, "minatar": 6, "babyai": 7}

    def __init__(self, envs: dict, babyai_syms: int = 20):
        self.objective = "R"
        self.ACT, self.OBS, self.RWD = 0, 1, 2
        self.ACT0 = 3
        self.RWD0 = self.ACT0 + self.K
        self.SYM0 = self.RWD0 + 2
        self.sym_base, base = {}, self.SYM0
        order = list(envs.keys()) + (["babyai"] if "babyai" not in envs else [])
        for name in order:
            n = envs[name].n_obs_sym if name in envs else babyai_syms
            self.sym_base[name] = base
            base += n
        self.vocab = base
        self.obs_len = {n: e.obs_len for n, e in envs.items()}
        self.frame_len_by = {n: 2 + 2 + 1 + e.obs_len for n, e in envs.items()}
        self.cur = list(envs.keys())[0]

    def set_env(self, name: str):
        self.cur = name

    @property
    def frame_len(self):                    # st6 Agent reads codec.frame_len
        return self.frame_len_by[self.cur]

    def frame(self, obs_sym, action, reward):
        a = torch.tensor([self.ACT, self.ACT0 + action], dtype=torch.long)
        r = torch.tensor([self.RWD, self.RWD0 + (1 if reward > 0 else 0)],
                         dtype=torch.long)
        obs = torch.tensor([self.OBS], dtype=torch.long)
        cells = obs_sym + self.sym_base[self.cur]
        return torch.cat([a, r, obs, cells])

    def learn_mask(self, y):
        m = y >= self.SYM0
        return m | ((y >= self.RWD0) & (y < self.SYM0))

    def obs_mask(self, y):
        return y >= self.SYM0


# --------------------------------------------------------------------------
# LifeAgent — st6 Agent living in several worlds with full state capture
# --------------------------------------------------------------------------

class LifeAgent(Agent):
    def __init__(self, envs: dict, args, codec: UnifiedCodec):
        self.envs = envs
        self._cur = list(envs.keys())[0]
        super().__init__(envs[self._cur], args)
        self.codec = codec                  # replace the per-env codec
        # rebuild model/opt for the UNIFIED vocab (Agent.__init__ built for
        # the single-env codec; the life needs the full shared space)
        ns = argparse.Namespace(d_model=args.d_model, n_layers=args.n_layers,
                                kernel=7, dropout=0.0, tau=0.5,
                                lam_min=2 ** -6)
        self.model = build_model("morphssm", codec.vocab, ns)
        if self.use_head:
            from tikitorch.nn import TropicalTopKModule
            self.model.action_head = TropicalTopKModule(
                args.d_model, codec.K, k=min(8, args.d_model))
        self.model.set_tau(0.0)
        self.model.eval()
        self.engine = MorphPCEngine(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                     weight_decay=0.01)

    # -- world switching ---------------------------------------------------
    @property
    def cur(self):
        return self._cur

    def switch_env(self, name: str):
        """Era boundary: mark the outgoing frame done (zeroes cross-era
        credit via the existing ctx_d machinery), swap the resident world.
        ctx ROLLS ACROSS the boundary by design (E5_SPEC)."""
        if len(self.ctx_d) > 0:
            self.ctx_d[-1] = 1.0
        self._cur = name
        self.env = self.envs[name]
        self.codec.set_env(name)
        self.gate = MarginGate(delta=self.args.delta_by[name])

    # -- acting: renormalize over the CURRENT world's action set -----------
    @torch.no_grad()
    def act_token(self):
        c = self.codec
        k_env = c.ENV_K[self._cur]
        probe = torch.cat([self.ctx[1:], torch.tensor([c.ACT])])
        if self.use_head:
            lg = self.engine.act_probe_logits(probe.unsqueeze(0))[0][-1]
            pi = torch.softmax(lg[:k_env].float() / self.tau_pi, dim=0)
        else:
            lg = self.model(probe.unsqueeze(0))[0][-1]
            pi = torch.softmax(lg[c.ACT0:c.ACT0 + k_env] / self.tau_pi, dim=0)
        return int(torch.multinomial(pi, 1))

    @torch.no_grad()
    def pol_entropy(self):
        c = self.codec
        k_env = c.ENV_K[self._cur]
        probe = torch.cat([self.ctx[1:], torch.tensor([c.ACT])])
        if self.use_head:
            lg = self.engine.act_probe_logits(probe.unsqueeze(0))[0][-1]
            pi = torch.softmax(lg[:k_env].float() / self.tau_pi, dim=0)
        else:
            lg = self.model(probe.unsqueeze(0))[0][-1]
            pi = torch.softmax(lg[c.ACT0:c.ACT0 + k_env] / self.tau_pi, dim=0)
        return float(-(pi * pi.clamp_min(1e-9).log()).sum())

    # -- full state capture (E5_SPEC 15-item blob, agent part) -------------
    def state(self) -> dict:
        return dict(
            model=self.model.state_dict(),
            opt=self.opt.state_dict(),
            ctx=self.ctx.clone(), ctx_r=self.ctx_r.clone(),
            ctx_d=self.ctx_d.clone(),
            gate=(self.gate.mu_f, self.gate.mu_s),
            sal_mu=None if self.sal is None else self.sal.mu,
            adv_baseline=self.adv_baseline, var_ema=self.var_ema,
            ce_ema=self.ce_ema, intr_sum=self.intr_sum,
            t_step=self.t_step, cur=self._cur,
            env_states={n: e.get_state() for n, e in self.envs.items()},
            env_digests={n: _env_digest(e)
                         for n, e in self.envs.items()},
            obs_gen=self.obs_gen.get_state(),
        )

    def restore(self, blob: dict):
        self.model.load_state_dict(blob["model"])
        self.opt.load_state_dict(blob["opt"])
        self.ctx = blob["ctx"].clone()
        self.ctx_r = blob["ctx_r"].clone()
        self.ctx_d = blob["ctx_d"].clone()
        self._cur = blob["cur"]
        self.env = self.envs[self._cur]
        self.codec.set_env(self._cur)
        self.gate = MarginGate(delta=self.args.delta_by[self._cur])
        self.gate.mu_f, self.gate.mu_s = blob["gate"]
        if self.sal is not None and blob["sal_mu"] is not None:
            self.sal.mu = blob["sal_mu"]
        self.adv_baseline = blob["adv_baseline"]
        self.var_ema = blob["var_ema"]
        self.ce_ema = blob["ce_ema"]
        self.intr_sum = blob["intr_sum"]
        self.t_step = blob["t_step"]
        for n, e in self.envs.items():
            e.set_state(blob["env_states"][n])
        self.obs_gen.set_state(blob["obs_gen"])


# --------------------------------------------------------------------------
# Life — the era loop with organism-grade ops
# --------------------------------------------------------------------------

def agent_args(cfg: dict):
    lever = cfg["mech_lever"]
    fire = cfg["fire_schedule"]
    return argparse.Namespace(
        env="foraging", mode="agent", gate="S", objective="R",
        steps=cfg["era_steps"] * cfg["cycles"] * len(cfg["env_order"]),
        world_seed=cfg["world_seed"], seed=cfg["seed"],
        biome_steps=cfg["biome_steps"], d_model=cfg["d_model"],
        n_layers=cfg["n_layers"], window=cfg["window"], lr=cfg["lr"],
        delta=cfg["delta_foraging"], tau_act=0.02, horizon=1,
        gamma=cfg["gamma"], policy="token", tau_pi=cfg["tau_pi"],
        fire_always=(fire == "always"),
        prelife_frames=cfg["prelife_frames"],
        prelife_steps=cfg["prelife_steps"],
        intr_beta=cfg["intr_beta"], adv_norm=cfg["adv_norm"], mb_warmup=0,
        ent_k=None,
        obs_k=(-2 if lever in ("obsk-2", "headobsk") else
               -3 if lever == "obsk-3" else None),
        obs_frac=(0.25 if lever == "sub25" else None),
        fire_anneal=(cfg["fire_anneal_frac"] if fire == "anneal" else None),
        adv_fire=(fire == "advfire"),
        action_head=(lever in ("head", "headobsk")),
        wm_value=False,
        delta_by={"foraging": cfg["delta_foraging"],
                  "minatar": cfg["delta_minatar"],
                  "babyai": cfg["delta_minatar"]},
    )


class Life:
    def __init__(self, workdir: str, cfg: dict):
        self.cfg = cfg
        torch.manual_seed(cfg["seed"])      # BEFORE env+model construction —
                                            # the life must be deterministic
                                            # from (seed, ckpt chain)
        self.wd = Path(workdir)
        (self.wd / "ckpt").mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(str(self.wd / "ledger"))
        self.tele = (self.wd / "tele.jsonl")
        self.hashes = dict(config=config_hash(cfg), env_code=env_code_hash())
        envs = {n: make_env(n, world_seed=cfg["world_seed"],
                            biome_steps=cfg["biome_steps"])
                for n in cfg["env_order"]}
        self.codec = UnifiedCodec(envs, babyai_syms=cfg["babyai_syms"])
        self.args = agent_args(cfg)
        self.agent = LifeAgent(envs, self.args, self.codec)
        self.life_step = 0
        self.era_idx = 0
        self.halvings = 0
        self.fires = 0
        self.total_return = 0.0
        self.ce_sum = 0.0
        self.win = []
        self.counters = dict(resumes=0, incidents=0, probes=0)
        self.probe_gen = torch.Generator().manual_seed(cfg["seed"] + 999)
        self.probe_windows = None            # lazy-frozen at first suite
        self.snap_actions = None             # end-of-cycle-1 action snapshot

    # -- era math ----------------------------------------------------------
    def era_env(self, era: int) -> str:
        order = self.cfg["env_order"]
        cyc, off = divmod(era, len(order))
        rot = order[cyc % len(order):] + order[:cyc % len(order)]
        return rot[off]

    @property
    def total_eras(self):
        return self.cfg["cycles"] * len(self.cfg["env_order"])

    # -- checkpointing -----------------------------------------------------
    def capture(self) -> dict:
        return dict(
            agent=self.agent.state(),
            life_step=self.life_step, era_idx=self.era_idx,
            halvings=self.halvings, fires=self.fires,
            total_return=self.total_return, ce_sum=self.ce_sum,
            win_tail=self.win[-1000:],
            rng=dict(torch=torch.get_rng_state(),
                     probe=self.probe_gen.get_state()),
            ledger_seq=self.ledger.seq,
            counters=dict(self.counters), hashes=dict(self.hashes),
            config=dict(self.cfg),
            snap_actions=self.snap_actions,
        )

    def save(self, tag="periodic"):
        path = save_atomic(self.capture(), str(self.wd / "ckpt"),
                           self.life_step)
        self.ledger.append("CHECKPOINT", step=self.life_step, tag=tag,
                           path=os.path.basename(path),
                           env_digests=self.agent.state()["env_digests"])
        return path

    def resume(self) -> bool:
        blob = load_latest(str(self.wd / "ckpt"))
        if blob is None:
            return False
        assert blob["hashes"]["config"] == self.hashes["config"], \
            "config identity mismatch — refuse resume"
        assert blob["hashes"]["env_code"] == self.hashes["env_code"], \
            "env-code hash mismatch — refuse resume (declare RESUME_NONEXACT)"
        self.agent.restore(blob["agent"])
        self.life_step = blob["life_step"]
        self.era_idx = blob["era_idx"]
        self.halvings = blob["halvings"]
        self.fires = blob["fires"]
        self.total_return = blob["total_return"]
        self.ce_sum = blob["ce_sum"]
        self.win = list(blob["win_tail"])
        torch.set_rng_state(blob["rng"]["torch"])
        self.probe_gen.set_state(blob["rng"]["probe"])
        self.ledger.seq = max(self.ledger.seq, blob["ledger_seq"])
        self.counters = dict(blob["counters"])
        self.counters["resumes"] = self.counters.get("resumes", 0) + 1
        self.snap_actions = blob.get("snap_actions")
        # telemetry dedupe (st4 pattern): drop rows logged past the ckpt
        if self.tele.exists():
            rows = [json.loads(l) for l in self.tele.read_text().splitlines()]
            rows = [r for r in rows if r["t"] <= self.life_step]
            self.tele.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.ledger.append("RESUME", step=self.life_step,
                           resumes=self.counters["resumes"])
        return True

    # -- birth (one, on era-0's env; ports the st6 prelife verbatim) -------
    def birth(self):
        a, cfg = self.agent, self.cfg
        env = a.env
        torch.manual_seed(cfg["seed"])
        birth_frames = []
        for _ in range(cfg["prelife_frames"]):
            act = int(torch.randint(0, env.n_actions, (1,)))
            r, o, _ = env.step(act)
            birth_frames.append(a.codec.frame(o, act, r))
        ids = torch.cat(birth_frames)
        a.model.train()
        gb = torch.Generator().manual_seed(cfg["seed"] + 777)
        c0 = a.codec
        for step in range(cfg["prelife_steps"]):
            a.model.set_tau(tau_at(step, cfg["prelife_steps"], 0.5, True, 0.0))
            st = int(torch.randint(0, ids.numel() - cfg["window"] - 1, (1,),
                                   generator=gb))
            xb = ids[st:st + cfg["window"]].unsqueeze(0)
            yb = ids[st + 1:st + cfg["window"] + 1].unsqueeze(0)
            m = c0.learn_mask(yb)
            ap = (yb >= c0.ACT0) & (yb < c0.ACT0 + c0.K)
            if a.use_head:
                a.engine.action_loss = (
                    ap, (yb - c0.ACT0).clamp(0, c0.K - 1),
                    torch.ones_like(yb, dtype=torch.int8),
                    torch.zeros_like(yb, dtype=torch.int32))
            else:
                m = m | ap
            a.engine.loss_mask = m
            a.engine.train_step(xb, yb, a.opt, credit="cascade")
            a.engine.action_loss = None
        a.engine.loss_mask = None
        a.model.set_tau(0.0)
        a.model.eval()
        a.opt = torch.optim.AdamW(a.model.parameters(), lr=cfg["lr"],
                                  weight_decay=0.01)
        a.gate = MarginGate(delta=self.args.delta_by[a.cur])
        for _ in range(cfg["window"] // a.codec.frame_len + 1):
            act = int(torch.randint(0, env.n_actions, (1,)))
            r, o, dn = env.step(act)
            a.push(a.codec.frame(o, act, r), frame_reward=r, frame_done=dn)
        self.ledger.append("BIRTH", pi0_ent=round(a.pol_entropy(), 4),
                           vocab=a.codec.vocab)
        self.save(tag="birth")

    # -- guards ------------------------------------------------------------
    def nan_guard(self) -> bool:
        finite = all(torch.isfinite(p).all() for p in
                     self.agent.model.parameters())
        emas = [self.agent.gate.mu_f, self.agent.gate.mu_s,
                self.agent.var_ema, self.agent.adv_baseline]
        finite = finite and all(x is None or (x == x and abs(x) < 1e12)
                                for x in emas)
        if not finite:
            self.counters["incidents"] += 1
            self.ledger.append("INCIDENT", kind="nonfinite",
                               step=self.life_step)
            blob = load_latest(str(self.wd / "ckpt"))
            if blob is not None:
                self.agent.restore(blob["agent"])
            return False
        return True

    def apply_lr(self):
        cyc_done = self.era_idx // len(self.cfg["env_order"])
        halvings = min(cyc_done // self.cfg["lr_halve_every_cycles"],
                       (self.cfg["lr_floor_div"] - 1).bit_length())
        if halvings != self.halvings:
            self.halvings = halvings
            lr = self.cfg["lr"] / (2 ** halvings)
            lr = max(lr, self.cfg["lr"] / self.cfg["lr_floor_div"])
            for g in self.agent.opt.param_groups:
                g["lr"] = lr
            self.ledger.append("LR", step=self.life_step, lr=lr,
                               halvings=halvings)

    def run_probes(self):
        """The 6h retention/transfer suite (E5_SPEC): clone-isolated, life-
        clock frozen, rng-guarded. Rows -> probes.jsonl; snapshot at the
        first suite after cycle 1 completes."""
        import st7_probes as P
        t0 = time.time()
        h0 = P.state_hash(self)
        if self.probe_windows is None:
            pw_path = self.wd / "probe_windows.pt"
            if pw_path.exists():
                self.probe_windows = torch.load(pw_path, weights_only=False)
            else:
                self.probe_windows = P.generate_windows(
                    self.codec, self.cfg, n_windows=16, window=self.cfg["window"])
                torch.save(self.probe_windows, pw_path)
        a = P.suite_a(self.agent.model, self.codec, self.probe_windows)
        b = P.suite_b(self.agent, self.cfg, steps=1_000)
        acts, agree = P.suite_c(None, self.agent, self.probe_windows,
                                snapshot=self.snap_actions)
        cyc_done = self.era_idx // len(self.cfg["env_order"])
        if self.snap_actions is None and cyc_done >= 1:
            self.snap_actions = acts                   # end-of-cycle-1 snapshot
        assert h0 == P.state_hash(self), "probe suite mutated the life"
        self.counters["probes"] += 1
        row = dict(t=self.life_step, era=self.era_idx, env=self.agent.cur,
                   wall_s=round(time.time() - t0, 1),
                   a=a, b=b, agree=agree)
        with (self.wd / "probes.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.ledger.append("PROBE", step=self.life_step,
                           n=self.counters["probes"],
                           wall_s=row["wall_s"])

    # -- the loop ----------------------------------------------------------
    def live(self, max_steps=None):
        t_ckpt = t_beat = t_probe = time.time()
        t0 = time.time()
        cfg = self.cfg
        total = self.total_eras * cfg["era_steps"]
        if max_steps is not None:
            total = min(total, self.life_step + max_steps)
        # enter the correct era env on boot/resume
        want = self.era_env(min(self.era_idx, self.total_eras - 1))
        if self.agent.cur != want:
            self.agent.switch_env(want)
        while self.life_step < total:
            era_for_step = min(self.life_step // cfg["era_steps"],
                               self.total_eras - 1)
            if era_for_step != self.era_idx:
                self.save(tag="era_end")
                self.era_idx = era_for_step
                nxt = self.era_env(self.era_idx)
                self.ledger.append("ERA", idx=self.era_idx, env=nxt,
                                   step=self.life_step)
                if nxt != self.agent.cur:
                    self.agent.switch_env(nxt)
                self.apply_lr()
            try:
                r, fired, ce = self.agent.step(learn=True)
            except Exception as ex:                       # env incident
                self.counters["incidents"] += 1
                self.ledger.append("INCIDENT", kind="env_exception",
                                   err=repr(ex)[:200], step=self.life_step)
                blob = load_latest(str(self.wd / "ckpt"))
                if blob is not None:
                    self.agent.restore(blob["agent"])
                    torch.rand(1)                          # POISON_SKIP splice
                    self.ledger.append("POISON_SKIP", step=self.life_step)
                continue
            self.life_step += 1
            self.total_return += r
            self.fires += int(fired)
            self.ce_sum += ce
            self.win.append(r)
            if len(self.win) > 1000:
                self.win = self.win[-1000:]
            if self.life_step % 1000 == 0:
                with self.tele.open("a") as f:
                    f.write(json.dumps(dict(
                        t=self.life_step, era=self.era_idx,
                        env=self.agent.cur, ret1k=sum(self.win),
                        ce=round(self.ce_sum / self.life_step, 4),
                        fire=round(self.fires / self.life_step, 4),
                        piH=round(self.agent.pol_entropy(), 4),
                        intr=round(self.agent.intr_sum, 1))) + "\n")
            if self.life_step % cfg["nan_check_every"] == 0:
                self.nan_guard()
            now = time.time()
            if (self.life_step % cfg["ckpt_every_steps"] == 0
                    or now - t_ckpt > cfg["ckpt_every_secs"]):
                self.save()
                t_ckpt = now
            if (cfg["probe_every_hours"]
                    and now - t_probe > cfg["probe_every_hours"] * 3600):
                self.run_probes()
                t_probe = time.time()
            if now - t_beat > cfg["heartbeat_secs"]:
                (self.wd / "status.json").write_text(json.dumps(dict(
                    ts=int(now), step=self.life_step, era=self.era_idx,
                    env=self.agent.cur, wall_h=round((now - t0) / 3600, 2),
                    sps=round(self.life_step / max(now - t0, 1), 1),
                    probes=self.counters["probes"],
                    resumes=self.counters["resumes"])))
                t_beat = now
            if (cfg["budget_wall_hours"]
                    and (now - t0) / 3600 > cfg["budget_wall_hours"]):
                self.ledger.append("END_OF_LIFE", reason="budget_cap",
                                   step=self.life_step)
                break
        self.save(tag="era_end" if self.life_step >= total else "budget")
        self.ledger.append("LIFE_COMPLETE" if self.life_step >= total
                           else "LIFE_PAUSED", step=self.life_step)
        self.ledger.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    a = p.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text()))
    if a.smoke:
        cfg.update(era_steps=1_000, biome_steps=250, cycles=2,
                   prelife_frames=400, prelife_steps=60,
                   ckpt_every_steps=500, ckpt_every_secs=3600,
                   d_model=64, n_layers=2)
    life = Life(a.workdir, cfg)
    if not (a.resume and life.resume()):
        life.birth()
    life.live(max_steps=a.max_steps)
    print(f"[E5] life at step {life.life_step}/"
          f"{life.total_eras * cfg['era_steps']} "
          f"(era {life.era_idx}, {life.agent.cur}); "
          f"resumes={life.counters['resumes']}")


if __name__ == "__main__":
    main()
