"""E5 probe suites (E5_SPEC): clone-isolated retention/transfer measurement.

Cells = {foraging biome 0/1/2, each minatar game, [babyai levels on arrival]}.
Suite A: world-model obs-CE (+reward-CE) on FROZEN teacher-forced windows.
Suite B: policy return of a CLONED frozen model on fresh fixed-seed envs
         (2k steps at the life's tau_pi — NOT greedy; E1 argmax-parking).
Suite C: action-agreement vs the end-of-cycle-1 snapshot on Suite A windows.

Isolation contract (organism/probes.py): probes touch the live agent only via
model forward under no_grad; never the gate/salience; never the live envs;
dedicated generators. st7_test_resume asserts the life state-hash is
unchanged across a full suite."""
import contextlib
import copy
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def rng_guard():
    """Probes must not advance the life's global RNG stream (st5_world.pick
    falls back to the global generator, so probe env stepping would otherwise
    shift the live trajectory — the isolation assert caught this). Snapshot +
    restore around every suite; matches the life-clock-frozen semantics."""
    st = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(st)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from st5_envs import make_env                                 # noqa: E402

PROBE_SEED = 424242


def probe_cells(cfg):
    cells = [("foraging", b) for b in range(3)]
    cells += [("minatar", g) for g in range(5)]
    if cfg.get("babyai_enabled"):
        cells += [("babyai", l) for l in range(2)]
    return cells


def _cell_env(env_name, idx, biome_steps=10 ** 9):
    """A fresh probe env pinned to one biome/game/level (never the live env)."""
    e = make_env(env_name, world_seed=PROBE_SEED, biome_steps=biome_steps)
    if env_name == "foraging":
        e.world._enter_biome(idx)
    elif env_name == "minatar":
        e._enter(idx)                     # game index == biome order start
        while e._game != idx:             # rotate to the requested game
            e._enter(e._bi + 1)
    else:                                 # babyai: level index
        e._enter(idx)
        while e._lvl != idx:
            e._enter(e._bi + 1)
    return e


def generate_windows(codec, cfg, n_windows=32, window=96):
    """Frozen per-cell (x, y) teacher-forcing windows from fixed random
    rollouts. Generated ONCE (pilot time), saved + sha256'd."""
    gen = torch.Generator().manual_seed(PROBE_SEED)
    out = {}
    with rng_guard():
      for env_name, idx in probe_cells(cfg):
        e = _cell_env(env_name, idx)
        codec.set_env(env_name)
        frames = []
        need = (n_windows + 2) * window
        toks = 0
        while toks < need:
            a = int(torch.randint(0, e.n_actions, (1,), generator=gen))
            r, o, dn = e.step(a)
            f = codec.frame(o, a, r)
            frames.append(f)
            toks += len(f)
        ids = torch.cat(frames)
        xs, ys = [], []
        for w in range(n_windows):
            st = int(torch.randint(0, ids.numel() - window - 1, (1,),
                                   generator=gen))
            xs.append(ids[st:st + window])
            ys.append(ids[st + 1:st + window + 1])
        out[f"{env_name}:{idx}"] = (torch.stack(xs), torch.stack(ys))
    return out


def windows_digest(windows) -> str:
    h = hashlib.sha256()
    for k in sorted(windows):
        h.update(k.encode())
        h.update(bytes(windows[k][0].reshape(-1).tolist()[:512]))
    return h.hexdigest()[:16]


@torch.no_grad()
def suite_a(model, codec, windows):
    """Per-cell obs-CE and reward-CE, teacher-forced on the frozen windows."""
    rows = {}
    for cell, (xs, ys) in windows.items():
        lg = model(xs).float()
        V = lg.shape[-1]
        flat_lg, flat_y = lg.reshape(-1, V), ys.reshape(-1)
        obs_m = (flat_y >= codec.SYM0)
        rwd_m = (flat_y >= codec.RWD0) & (flat_y < codec.SYM0)
        rows[cell] = dict(
            obs_ce=round(float(F.cross_entropy(flat_lg[obs_m],
                                               flat_y[obs_m])), 4),
            rwd_ce=round(float(F.cross_entropy(flat_lg[rwd_m],
                                               flat_y[rwd_m])), 4)
            if int(rwd_m.sum()) else None)
    return rows


@torch.no_grad()
def suite_b(agent, cfg, steps=2_000):
    """Clone-return probes: deepcopy the model, fresh fixed-seed env per
    cell, private ctx, act at the life's tau_pi. The live agent, its envs,
    gate, and EMAs are never touched."""
    model = copy.deepcopy(agent.model)
    model.eval()
    engine = None
    if agent.use_head:
        from pc_convrnn_charlm import MorphPCEngine
        engine = MorphPCEngine(model)
    codec = agent.codec
    gen = torch.Generator().manual_seed(PROBE_SEED + 1)
    rows = {}
    with rng_guard():
      for env_name, idx in probe_cells(cfg):
        e = _cell_env(env_name, idx)
        codec.set_env(env_name)
        k_env = codec.ENV_K[env_name]
        window = agent.window
        ctx = torch.zeros(window, dtype=torch.long)
        for _ in range(window // codec.frame_len + 1):    # private warm
            a = int(torch.randint(0, e.n_actions, (1,), generator=gen))
            r, o, dn = e.step(a)
            f = codec.frame(o, a, r)
            ctx = torch.cat([ctx[len(f):], f])
        ret = 0.0
        for _ in range(steps):
            probe = torch.cat([ctx[1:], torch.tensor([codec.ACT])])
            if engine is not None:
                lg = engine.act_probe_logits(probe.unsqueeze(0))[0][-1]
                pi = torch.softmax(lg[:k_env].float() / agent.tau_pi, dim=0)
            else:
                lg = model(probe.unsqueeze(0))[0][-1]
                pi = torch.softmax(
                    lg[codec.ACT0:codec.ACT0 + k_env] / agent.tau_pi, dim=0)
            a = int(torch.multinomial(pi, 1, generator=gen))
            r, o, dn = e.step(a)
            ret += r
            f = codec.frame(o, a, r)
            ctx = torch.cat([ctx[len(f):], f])
        rows[f"{env_name}:{idx}"] = dict(ret1k=round(ret / steps * 1000, 2))
    codec.set_env(agent.cur)                              # restore view
    return rows


@torch.no_grad()
def suite_c(model_or_engine, agent, windows, snapshot=None):
    """Argmax action at every [ACT]-successor position of the frozen windows;
    agreement vs the end-of-cycle-1 snapshot (st4 snap_top1 analog)."""
    codec = agent.codec
    acts = {}
    for cell, (xs, ys) in windows.items():
        pos = (ys >= codec.ACT0) & (ys < codec.ACT0 + codec.K)
        if agent.use_head:
            lg = agent.engine.act_probe_logits(xs)        # (N, T, K)
            a = lg.argmax(-1)
        else:
            lg = agent.model(xs)
            a = lg[:, :, codec.ACT0:codec.ACT0 + codec.K].argmax(-1)
        acts[cell] = a[pos]
    if snapshot is None:
        return acts, None
    agree = {cell: round(float((acts[cell] == snapshot[cell]).float().mean()),
                         4) if cell in snapshot else None
             for cell in acts}
    return acts, agree


def canonical_bytes(obj, depth=0) -> bytes:
    """Value-canonical serialization for digests/witnesses. Plain pickle
    embeds torch STORAGE MEMORY ADDRESSES as keys (verified: two dumps of an
    unchanged env differ only in the storage-id string), so byte-stable
    hashing must walk values instead."""
    if depth > 6:
        return b"<deep>"
    if torch.is_tensor(obj):
        return obj.detach().contiguous().numpy().tobytes()
    if isinstance(obj, torch.Generator):
        return obj.get_state().numpy().tobytes()
    if isinstance(obj, (str, int, float, bool, bytes, type(None))):
        return repr(obj).encode()
    if isinstance(obj, dict):
        return b"{" + b",".join(
            repr(k).encode() + b":" + canonical_bytes(v, depth + 1)
            for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))) + b"}"
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(canonical_bytes(v, depth + 1)
                                for v in obj) + b"]"
    if hasattr(obj, "get_state") and "RandomState" in type(obj).__name__:
        return repr(obj.get_state()).encode()
    if hasattr(obj, "__dict__"):
        return canonical_bytes(vars(obj), depth + 1)
    return repr(obj).encode()


def env_digest(env) -> str:
    return hashlib.sha256(canonical_bytes(env)).hexdigest()[:16]


def state_hash(life) -> str:
    """The isolation witness: everything a probe must NOT change."""
    h = hashlib.sha256()
    for p in life.agent.model.parameters():
        h.update(p.detach().numpy().tobytes())
    st = life.agent.state()
    for k in ("ctx", "ctx_r", "ctx_d"):
        h.update(st[k].numpy().tobytes())
    h.update(repr((st["gate"], st["sal_mu"], st["adv_baseline"],
                   st["var_ema"], st["ce_ema"], st["t_step"])).encode())
    for n in sorted(life.agent.envs):
        h.update(env_digest(life.agent.envs[n]).encode())
    h.update(torch.get_rng_state().numpy().tobytes())
    return h.hexdigest()[:16]
