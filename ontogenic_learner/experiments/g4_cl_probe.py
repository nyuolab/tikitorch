"""G4 — the continual-learning gate (see ../G4_SPEC.md §5).

Compressed organism life: wiki 30k -> shak 15k -> wiki-revisit 15k, val probes every 2k.
Cumulative arms isolate each mechanism's marginal:
  base | m1 (weight prior) | m12 (+satiating gate) | m123 (+precision consolidation)
       | m1234 (+balanced/strong replay)

Run: PYTHONPATH=. python ontogenic_learner/experiments/g4_cl_probe.py [--arms base m1 ...]
"""
import argparse
import copy
import json
import math
import os
import sys
import time

sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from tikitorch.nn.predictive_coding import DyadicPCNet
from ontogenic_learner.experiments.two_speed_learner import Reservoir
from ontogenic_learner.organism.vocab import Vocab
from ontogenic_learner.organism.corpus import raw_text

OUT = "ontogenic_learner/results/g4_cl/probe.jsonl"
CTX, HIDDEN, DEPTH, BATCH = 12, 256, 4, 256
ERA = [("wiki", 30_000), ("shakespeare", 15_000), ("wiki", 15_000)]
PROBE_EVERY = 2_000
VAL_N = 4_096


def onehot(X, V):
    return F.one_hot(X, V).float().flatten(1)


class Domain:
    def __init__(self, name, vocab, gen):
        text = raw_text(name)
        ids = torch.tensor(vocab.encode_text(text), dtype=torch.long)
        n_val = 50_000
        self.train, self.val = ids[:-n_val], ids[-n_val:]
        self.gen = gen

    def batch(self, n=BATCH):
        i = torch.randint(0, self.train.size(0) - CTX - 1, (n,), generator=self.gen)
        idx = i[:, None] + torch.arange(CTX + 1)
        w = self.train[idx]
        return w[:, :CTX], w[:, CTX]

    def val_windows(self, n=VAL_N):
        i = torch.arange(0, self.val.size(0) - CTX - 1, max(1, (self.val.size(0) - CTX - 1) // n))[:n]
        idx = i[:, None] + torch.arange(CTX + 1)
        w = self.val[idx]
        return w[:, :CTX], w[:, CTX]


@torch.no_grad()
def val_bpc(net, Xw, yw, V):
    tot, n = 0.0, 0
    for i in range(0, Xw.size(0), 1024):
        logits = net.predict_logits(onehot(Xw[i:i + 1024], V))
        tot += F.cross_entropy(logits, yw[i:i + 1024], reduction="sum").item()
        n += logits.size(0)
    return tot / n / math.log(2)


@torch.no_grad()
def top1_agree(net, ref_net, Xw, V):
    """Scale-invariant retention: top-1 agreement with the era-1 snapshot (spec §5)."""
    agree, n = 0, 0
    for i in range(0, Xw.size(0), 1024):
        x = onehot(Xw[i:i + 1024], V)
        agree += (net.predict_logits(x).argmax(1) == ref_net.predict_logits(x).argmax(1)).sum().item()
        n += x.size(0)
    return agree / n


class G4Policy:
    """Two-speed policy with the G4 mechanism toggles. Constants match the organism.

    Round-3 toggles (from round-1/2 nulls): anchor = EWC-style precision anchor
    (changes the fixed point — survives Adam, unlike grad damping); parity = sleep
    at fast-lr mass (tests the 14x replay-imbalance arithmetic causally); tau =
    fire less (z>1.0, ~16%% duty — the one lever that moved retention in round 1).
    """

    def __init__(self, net, seed, m1=False, m2=False, m3=False, m4=False,
                 anchor=False, parity=False, tau=False):
        self.net, self.m1, self.m2, self.m3, self.m4 = net, m1, m2, m3, m4
        self.anchor, self.parity, self.tau = anchor, parity, tau
        wd = 1e-2 if m1 else 0.0
        Opt = torch.optim.AdamW if m1 else torch.optim.Adam
        self.opt_fast = Opt(net.parameters(), lr=1e-3, weight_decay=wd)
        self.opt_slow = Opt(net.parameters(), lr=1e-3 if parity else 3e-4,
                            weight_decay=wd)
        self.mu, self.var = 0.0, 1.0
        self.res = Reservoir(4096, seed)
        self.n_fast = 0
        self.sleep_updates = 45 if parity else (30 if m4 else 15)
        self.epi_floor = 0.20  # bits, M2 satiation floor
        self.z_thresh = 1.0 if tau else 0.25
        self.lam = 1.0        # anchor strength
        self.theta_ema = [p.detach().clone() for p in net.parameters()] if anchor else None

    def _anchor_pull(self):
        """EWC-style: grad += lam * v_hat * (theta - theta_ema); v_hat from the slow
        Adam's exp_avg_sq (weight-posterior precision). Fixed-point change, Adam-proof."""
        st = self.opt_slow.state
        params = list(self.net.parameters())
        vs = [st[p]["exp_avg_sq"] for p in params if p in st and "exp_avg_sq" in st[p]]
        vmean = (torch.cat([v.flatten() for v in vs]).mean().clamp_min(1e-12)
                 if vs else torch.tensor(1.0))
        for p, ema in zip(params, self.theta_ema):
            if p.grad is None:
                continue
            vhat = ((st[p]["exp_avg_sq"] / vmean).clamp(max=100.0)
                    if p in st and "exp_avg_sq" in st[p] else torch.zeros_like(p))
            p.grad.add_(self.lam * vhat * (p.detach() - ema))

    def _anchor_track(self):
        with torch.no_grad():
            for p, ema in zip(self.net.parameters(), self.theta_ema):
                ema.mul_(0.9999).add_(p.detach(), alpha=1e-4)

    def _grads(self, xb, yb):
        self.opt_fast.zero_grad(set_to_none=True)
        target = self.net._make_target(yb, ref=xb)
        self.net.cascade_grads(xb, target)

    def _consolidation_damp(self):
        """M3: damp fast-kick grads by slow-Adam weight-posterior precision."""
        st = self.opt_slow.state
        vs = [st[p]["exp_avg_sq"] for p in self.net.parameters()
              if p in st and "exp_avg_sq" in st[p]]
        if not vs:
            return
        vmean = torch.cat([v.flatten() for v in vs]).mean().clamp_min(1e-12)
        for p in self.net.parameters():
            if p.grad is None or p not in st or "exp_avg_sq" not in st[p]:
                continue
            vhat = st[p]["exp_avg_sq"] / vmean
            p.grad.div_(1.0 + 4.0 * vhat)

    def step(self, xi, yi, V, step_i):
        xb = onehot(xi, V)
        with torch.no_grad():
            logits = self.net.predict_logits(xb)
            ce = F.cross_entropy(logits, yi).item()
            ent = -(F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(1).mean().item()
        z = (ce - self.mu) / max(math.sqrt(self.var), 1e-6)
        self.mu = 0.99 * self.mu + 0.01 * ce
        self.var = 0.99 * self.var + 0.01 * (ce - self.mu) ** 2
        fire = ((ce - ent) / math.log(2) > self.epi_floor) if self.m2 else (z > self.z_thresh)
        if fire:
            self.n_fast += 1
            self._grads(xb, yi)
            if self.m3:
                self._consolidation_damp()
            if self.anchor:
                self._anchor_pull()
            self.opt_fast.step()
            self.res.insert(xi, yi, prio=max(z, 0.1), dom="cur")
        else:
            self.res.insert(xi, yi, prio=0.1, dom="cur", k=4)
        if self.anchor:
            self._anchor_track()
        balanced_sleep = self.m4 or self.parity
        if step_i % 150 == 149:
            for _ in range(self.sleep_updates):
                rx, ry = self.res.sample(BATCH, weighted=not balanced_sleep, balanced=balanced_sleep)
                if rx is None:
                    break
                self.opt_slow.zero_grad(set_to_none=True)
                t = self.net._make_target(ry, ref=onehot(rx, V))
                self.net.cascade_grads(onehot(rx, V), t)
                self.opt_slow.step()
        return fire


def run_arm(arm, seed, vocab, V, out_f):
    torch.manual_seed(seed)
    dims = [CTX * V] + [HIDDEN] * (DEPTH - 1) + [V]
    net = DyadicPCNet(dims, residual=True)
    suffix = arm[1:] if arm != "base" else ""
    flags = dict(m1="1" in suffix, m2="2" in suffix, m3="3" in suffix,
                 m4="4" in suffix, anchor="a" in suffix, parity="p" in suffix,
                 tau="t" in suffix)
    pol = G4Policy(net, seed, **flags)
    doms = {n: Domain(n, vocab, torch.Generator().manual_seed(seed + hash(n) % 1000))
            for n in ("wiki", "shakespeare")}
    vals = {n: d.val_windows() for n, d in doms.items()}
    # domain tags for the reservoir need the real name; wrap insert
    era1_snapshot = None
    g_step = 0
    t0 = time.time()
    for era_i, (dom_name, n_steps) in enumerate(ERA):
        d = doms[dom_name]
        for s in range(n_steps):
            xi, yi = d.batch()
            # tag reservoir inserts with the true domain
            orig_insert = pol.res.insert
            pol.res.insert = lambda x, y, prio=None, k=32, dom="": orig_insert(x, y, prio=prio, k=k, dom=dom_name)
            pol.step(xi, yi, V, g_step)
            pol.res.insert = orig_insert
            g_step += 1
            if g_step % PROBE_EVERY == 0:
                norm = sum(p.norm().item() for p in net.parameters()) / sum(1 for _ in net.parameters())
                row = {"arm": arm, "seed": seed, "step": g_step, "era": era_i, "dom": dom_name,
                       "wiki": round(val_bpc(net, *vals["wiki"], V), 4),
                       "shak": round(val_bpc(net, *vals["shakespeare"], V), 4),
                       "norm": round(norm, 2),
                       "fast_rate": round(pol.n_fast / g_step, 4)}
                if era1_snapshot is not None:
                    row["wiki_top1_vs_era1"] = round(
                        top1_agree(net, era1_snapshot, vals["wiki"][0], V), 4)
                out_f.write(json.dumps(row) + "\n"); out_f.flush()
                print(f"[{arm} s{seed}] {g_step:>6} {dom_name:<12} wiki={row['wiki']:.3f} "
                      f"shak={row['shak']:.3f} norm={row['norm']:.1f} fast={row['fast_rate']:.2f}"
                      + (f" ret={row.get('wiki_top1_vs_era1', 0):.3f}" if era1_snapshot else ""),
                      flush=True)
        if era_i == 0:
            era1_snapshot = copy.deepcopy(net).eval()
    print(f"[{arm} s{seed}] done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["base", "m1", "m12", "m123", "m1234"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    vocab = Vocab.build()
    V = vocab.V
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as f:
        for seed in args.seeds:
            for arm in args.arms:
                run_arm(arm, seed, vocab, V, f)


if __name__ == "__main__":
    main()
