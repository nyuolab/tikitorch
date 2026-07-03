"""Organism v0 daemon — the life loop (plan §daemon).

Single-threaded: stream -> gate -> (fast kick | skip) -> sleep blocks -> checkpoints ->
probes -> inbox -> heartbeat. Probes and inbox handling pause the stream (pauses are
ledgered; life_step does not advance). Restart-safe: boot resumes from the newest local or
S3 checkpoint (Gate-2-tested bit-exact).

Run: PYTHONPATH=<repo> python -u ontogenic_learner/organism/daemon.py --config organism.json
     [--smoke] [--workdir DIR]
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import torch

from tikitorch.nn.predictive_coding import DyadicPCNet
from ontogenic_learner.organism import state as S
from ontogenic_learner.organism.config import OrganismConfig, make_config
from ontogenic_learner.organism.corpus import World
from ontogenic_learner.organism.ledger import Ledger
from ontogenic_learner.organism.policy import TwoSpeedPolicy, onehot
from ontogenic_learner.organism.probes import run_suite
from ontogenic_learner.organism.sampler import sample
from ontogenic_learner.organism.schedule import Schedule
from ontogenic_learner.organism.vocab import Vocab


def s3(cmd, *args):
    """aws-cli wrapper; on the box the instance role supplies creds. Never raises."""
    try:
        return subprocess.run(["aws", "s3", cmd, *args], capture_output=True,
                              text=True, timeout=120)
    except Exception:
        return None


class Organism:
    def __init__(self, cfg: OrganismConfig, workdir: str):
        self.cfg = cfg
        self.wd = workdir
        os.makedirs(workdir, exist_ok=True)
        self.ckpt_dir = os.path.join(workdir, "checkpoints")
        self.probe_dir = os.path.join(workdir, "probes")
        self.dev = cfg.device
        torch.set_num_threads(cfg.threads or max(1, (os.cpu_count() or 2) - 1))

        # deterministic streams of the life
        self.gens = {"schedule": torch.Generator().manual_seed(cfg.seed + 1),
                     "mnist": torch.Generator().manual_seed(cfg.seed + 2),
                     "probe": torch.Generator().manual_seed(cfg.seed + 3),
                     "sampler": torch.Generator().manual_seed(cfg.seed + 4)}

        vpath = os.path.join(workdir, "vocab.json")
        self.vocab = Vocab.load(vpath) if os.path.exists(vpath) else Vocab.build()
        if not os.path.exists(vpath):
            self.vocab.save(vpath)
        self.world = World(self.vocab, cfg.ctx, cfg.batch, self.gens["mnist"],
                           val_windows=256 if cfg.smoke else 4096)
        dims = [cfg.ctx * self.vocab.V] + [cfg.hidden] * (cfg.depth - 1) + [self.vocab.V]
        torch.manual_seed(cfg.seed)
        self.net = DyadicPCNet(dims, residual=True, gamma=cfg.gamma).to(self.dev)
        self.policy = TwoSpeedPolicy(self.net, self.vocab.V, cfg, cfg.seed)
        self.sched = Schedule(cfg, self.gens["schedule"])
        self.life_step = 0
        self.counters = {"incidents": 0, "resumes": 0, "probes": 0, "feeds": 0}
        self.hashes = {"config": cfg.config_hash(), "vocab": self.vocab.vocab_hash()}
        self.ledger = Ledger(os.path.join(workdir, "ledger"))
        self.t_ckpt = self.t_probe = self.t_beat = self.t_inbox = self.t_sync = 0.0
        self.status = "booting"
        self.born = time.time()

    # ------------------------------------------------------------- lifecycle
    def boot(self):
        blob = S.load_latest(self.ckpt_dir)
        if blob is None and self.cfg.s3_prefix:      # try S3 latest
            r = s3("cp", f"{self.cfg.s3_prefix}/checkpoints/latest.pt",
                   os.path.join(self.ckpt_dir, "ckpt-s3.pt"))
            if r and r.returncode == 0:
                blob = S.load_latest(self.ckpt_dir)
        if blob is not None:
            self.life_step, self.counters, saved_hashes = S.restore(
                blob, self.net, self.policy, self.world, self.sched, self.gens)
            assert saved_hashes.get("vocab") == self.hashes["vocab"], "vocab hash mismatch"
            self.counters["resumes"] = self.counters.get("resumes", 0) + 1
            # never move seq backwards: disk may hold events newer than the checkpoint
            self.ledger.seq = max(self.ledger.seq,
                                  blob.get("counters", {}).get("ledger_seq", 0))
            self.ledger.append("RESUME", life_step=self.life_step)
            print(f"[organism] RESUMED at life_step={self.life_step}", flush=True)
        else:
            self.birth()

    def birth(self):
        cfg = self.cfg
        print(f"[organism] BIRTH — phylogeny: {cfg.birth_steps} wiki steps", flush=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr_fast)
        cur = self.world.cursors["wiki"]
        for k in range(cfg.birth_steps):
            xi, yi = cur.next_windows()
            self.net.train_step_cascade(onehot(xi, self.vocab.V).to(self.dev),
                                        yi.to(self.dev), opt, mode="dyadic")
        cur.pos, cur.epoch = 0, 0                    # ontogeny restarts the stream
        # warm the surprise EMA + seed prior-life memories (the G1 recipe)
        import torch.nn.functional as F
        ids = self.world.ids["wiki"]
        for w in range(20):
            off = len(ids) - (w + 2) * cfg.batch * 3
            xs = torch.stack([ids[off+j: off+j+cfg.ctx] for j in range(cfg.batch)])
            ys = ids[off+cfg.ctx: off+cfg.ctx+cfg.batch]
            with torch.no_grad():
                l = F.cross_entropy(self.net.predict_logits(
                    onehot(xs, self.vocab.V).to(self.dev)), ys.to(self.dev)).item()
            self.policy.surprise(l)
            if w < 8:
                self.policy.res.insert(xs, ys, prio=1.0, dom="wiki")
        self.ledger.append("BIRTH", config=self.hashes["config"],
                           vocab=self.hashes["vocab"], birth_steps=cfg.birth_steps)
        self.checkpoint(tag="birth")

    # ------------------------------------------------------------- machinery
    def checkpoint(self, tag=""):
        self.counters["ledger_seq"] = self.ledger.seq
        blob = S.capture(self.net, self.policy, self.world, self.sched, self.gens,
                         self.life_step, self.counters, self.hashes)
        path = S.save_atomic(blob, self.ckpt_dir, self.life_step,
                             keep=self.cfg.ckpt_keep_local)
        self.ledger.append("CHECKPOINT", life_step=self.life_step, tag=tag,
                           path=os.path.basename(path))
        self.t_ckpt = time.time()
        if self.cfg.s3_prefix:
            s3("cp", path, f"{self.cfg.s3_prefix}/checkpoints/latest.pt")

    def heartbeat(self):
        st = {"status": self.status, "life_step": self.life_step,
              "life_day": round(self.life_step / self.cfg.steps_per_day, 3),
              "domain": self.sched.block_dom,
              "gate": {"n_fast": self.policy.n_fast, "n_sleep": self.policy.n_sleep,
                       "rolling_rate": round(self.policy.rolling_rate(), 4)},
              "counters": self.counters, "uptime_h": round((time.time()-self.born)/3600, 2),
              "ts": round(time.time(), 1)}
        with open(os.path.join(self.wd, "status.json"), "w") as f:
            json.dump(st, f, indent=1)
        self.t_beat = time.time()

    def sync_s3(self):
        if not self.cfg.s3_prefix:
            return
        for sub in ("ledger", "probes"):
            p = os.path.join(self.wd, sub)
            if os.path.isdir(p):
                s3("sync", p, f"{self.cfg.s3_prefix}/{sub}", "--quiet")
        s3("cp", os.path.join(self.wd, "status.json"),
           f"{self.cfg.s3_prefix}/status.json")
        self.t_sync = time.time()

    def poll_inbox(self):
        self.t_inbox = time.time()
        if not self.cfg.s3_prefix:
            return
        r = s3("ls", f"{self.cfg.s3_prefix}/inbox/")
        if not r or r.returncode != 0:
            return
        for line in r.stdout.splitlines():
            name = line.split()[-1]
            if not name.endswith(".json"):
                continue
            loc = os.path.join(self.wd, "inbox_" + name)
            if s3("cp", f"{self.cfg.s3_prefix}/inbox/{name}", loc).returncode != 0:
                continue
            try:
                req = json.load(open(loc))
            except Exception:
                req = {}
            if req.get("kind") == "sample":
                txt = sample(self.net, self.vocab, self.cfg.ctx,
                             req.get("prompt", ""), int(req.get("n", 300)),
                             float(req.get("temp", 0.8)), self.gens["sampler"], self.dev)
                out = {"req": name, "life_step": self.life_step,
                       "text": self.vocab.render_human(self.vocab.encode_mixed(txt))
                       if req.get("render") else txt}
                op = os.path.join(self.wd, "out_" + name)
                json.dump(out, open(op, "w"))
                s3("cp", op, f"{self.cfg.s3_prefix}/outbox/{name}")
                self.ledger.append("SAMPLE", req=name)
            elif req.get("kind") == "feed":
                text = req.get("text", "")
                gname = f"feed_{self.counters['feeds']}"
                self.world.add_guest(gname, text)
                self.sched.inject(gname, int(req.get("steps", 2000)))
                self.counters["feeds"] += 1
                self.ledger.append("FEED", name=gname, chars=len(text),
                                   sha=req.get("sha256", ""))
            elif req.get("kind") == "terminate":
                self.checkpoint(tag="terminate-request")
                self.status = "terminate-acked"
                self.heartbeat(); self.sync_s3()
                self.ledger.append("TERMINATE_ACK"); self.ledger.flush()
            s3("rm", f"{self.cfg.s3_prefix}/inbox/{name}")

    def nan_guard(self):
        if self.policy.finite():
            return
        self.counters["incidents"] += 1
        self.ledger.append("INCIDENT", kind="nonfinite", life_step=self.life_step)
        print(f"[organism] INCIDENT nonfinite @ {self.life_step} — restoring", flush=True)
        blob = S.load_latest(self.ckpt_dir)
        self.life_step, self.counters, _ = S.restore(
            blob, self.net, self.policy, self.world, self.sched, self.gens)
        # skip one batch on the current cursor to escape the poison window
        dom = self.sched.domain(self.life_step)
        self.world.cursors[dom].next_windows()

    # ------------------------------------------------------------- the life
    def live(self):
        cfg = self.cfg
        self.status = "alive"
        self.heartbeat()
        while True:
            now = time.time()
            if self.life_step >= cfg.max_life_steps:
                if self.status != "life_complete":
                    self.checkpoint(tag="end-of-life")
                    self.status = "life_complete"
                    self.ledger.append("LIFE_COMPLETE", life_step=self.life_step)
                    self.ledger.flush()
                    print("[organism] life complete — idling", flush=True)
                self.heartbeat(); self.sync_s3(); self.poll_inbox()
                time.sleep(30)
                continue

            dom = self.sched.domain(self.life_step)
            xi, yi = self.world.cursors[dom].next_windows()
            ev = self.policy.step(xi, yi, dom, self.life_step, self.dev)
            if ev:
                self.ledger.append(ev.pop("kind").upper(), step=self.life_step,
                                   pos=self.world.cursors[dom].pos, **ev)
            if self.life_step > 0 and self.life_step % cfg.sleep_every == 0:
                n = self.policy.sleep_block(self.dev)
                if n:
                    self.ledger.append("SLEEP", step=self.life_step, kicks=n)
            if self.life_step % cfg.nan_check_every == 0:
                self.nan_guard()
            if (self.life_step - getattr(self, "_ckpt_step", 0) >= cfg.ckpt_every_steps
                    or now - self.t_ckpt > cfg.ckpt_every_secs):
                self.checkpoint()
                self._ckpt_step = self.life_step
            if now - self.t_beat > cfg.heartbeat_secs:
                self.heartbeat()
            if now - self.t_inbox > cfg.inbox_poll_secs:
                self.poll_inbox()
            if now - self.t_sync > cfg.s3_sync_secs:
                self.sync_s3()
            if now - self.t_probe > cfg.probe_every_hours * 3600:
                self.status = "probing"; self.heartbeat()
                row = run_suite(self.net, self.policy, self.vocab, self.world, cfg,
                                self.gens, self.life_step, self.probe_dir, self.dev)
                self.counters["probes"] += 1
                self.ledger.append("PROBE", life_step=self.life_step,
                                   wall_s=row["wall_s"])
                self.t_probe = time.time()
                self.status = "alive"; self.heartbeat()
            self.life_step += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--workdir", default=os.path.join(REPO, "organism_life"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.config and os.path.exists(a.config):
        cfg = OrganismConfig.load(a.config)
    else:
        cfg = make_config(smoke=a.smoke)
    org = Organism(cfg, a.workdir)
    org.boot()
    org.live()


if __name__ == "__main__":
    main()
