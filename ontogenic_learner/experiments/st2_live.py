"""V2 — does-it-help: the live serve+learn loop vs frozen, on calibrated streams.

North-star gate V2 (NORTHSTAR_VALIDATION.md). ONE process walks a stream
window-by-window: serve (prequential CE with theta(t-1), BEFORE any learning),
surprise-gate, and — if fired — a FUSED cascade kick on the very caches the
serve produced (V0's serve_and_learn). A frozen twin is scored on identical
windows. Telemetry rows are the D2 strip-chart's data.

Arms: frozen | gated | gated+anchor.  Regimes: hard_shift | drift |
novel_vocab (+ stationary, gated arm only — the self-quieting calibration).
Warm start: the M1 S1 BPE checkpoint (V1-verified tokenizer identity).
tau=0 throughout (continued adaptation of a trained model — the deployed regime).

Metrics per run: prequential CE, regret vs frozen, HEADROOM-CAPTURED
(1 - R_oracle/R_oracle^frozen, in [0,1]), adaptation latency per boundary,
fire-rate, amortized cost multiplier (V0-measured kick cost 0.38 fwd-eq).

Usage::

    python ontogenic_learner/experiments/st2_live.py --arm gated --regime hard_shift --seed 0
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pc_convrnn_charlm import MorphPCEngine                # noqa: E402
from st1_streams import (                                  # noqa: E402
    DOMAINS, OUT, load_assets, load_ckpt, regime_schedule)

LN2 = math.log(2)
KICK_COST = 0.38          # V0-measured: fused op-volume ratio 1.38x => kick = 0.38 fwd-eq


class SurpriseGate:
    """z-scored surprise on window CE (G1 constants: EMA 0.99, z* = 0.25).

    V2b verdict: this family is a fixed-quantile RATE LIMITER, not a novelty
    detector — self-normalization makes the steady-state fire-rate
    regime-independent at every z*. Kept for reproducibility; the margin gate
    below is the V4 revision."""

    def __init__(self, ema=0.99, gate_z=0.25):
        self.ema, self.gate_z = ema, gate_z
        self.mu = None
        self.var = 1.0

    def __call__(self, ce: float) -> bool:
        if self.mu is None:
            self.mu = ce
            return True                                    # first window: learn
        z = (ce - self.mu) / max(self.var ** 0.5, 1e-6)
        fire = z > self.gate_z
        self.mu = self.ema * self.mu + (1 - self.ema) * ce
        self.var = self.ema * self.var + (1 - self.ema) * (ce - self.mu) ** 2
        return fire


class MarginGate:
    """Two-timescale ABSOLUTE-margin gate (the V2b revision): fire when the
    fast EMA of window CE exceeds the slow EMA by delta NATS. No variance
    normalization, so on stationary data fast==slow -> gap -> 0 and the gate
    self-quiets by construction; a shift jumps the gap by the CE step (~4
    nats >> delta) and storms until the learner closes it."""

    def __init__(self, delta=0.5, fast=0.9, slow=0.999):
        self.delta, self.a_f, self.a_s = delta, fast, slow
        self.mu_f = None
        self.mu_s = None

    def __call__(self, ce: float) -> bool:
        if self.mu_f is None:
            self.mu_f = self.mu_s = ce
            return True                                    # first window: learn
        self.mu_f = self.a_f * self.mu_f + (1 - self.a_f) * ce
        self.mu_s = self.a_s * self.mu_s + (1 - self.a_s) * ce
        return (self.mu_f - self.mu_s) > self.delta


class Anchor:
    """G4 precision anchor, ported verbatim in mechanism: after grads are set,
    pull toward the slow EMA of weights, weighted by the optimizer's own
    second-moment precision (vhat = exp_avg_sq / mean, clamped)."""

    def __init__(self, model, lam=1.0, ema=0.9999, clamp=100.0):
        self.lam, self.ema, self.clamp = lam, ema, clamp
        self.theta = {n: p.detach().clone() for n, p in model.named_parameters()}

    def pull(self, model, opt):
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            st = opt.state.get(p, {})
            v = st.get("exp_avg_sq")
            if v is None:
                continue
            vhat = (v / v.mean().clamp_min(1e-12)).clamp(max=self.clamp)
            p.grad.add_(self.lam * vhat * (p.detach() - self.theta[n]))

    def track(self, model):
        for n, p in model.named_parameters():
            self.theta[n].mul_(self.ema).add_(p.detach(), alpha=1 - self.ema)


def oracle_ces():
    o = {}
    for line in (OUT / "v1_streams.jsonl").open():
        r = json.loads(line)
        if r.get("gate") == "V1_oracle":
            o[r["domain"]] = r["oracle_ce"]
    return o


def stream_windows(assets, schedule, T, gen):
    """Yield (domain, x, y) windows following the regime schedule (train splits)."""
    for domain, n_tokens in schedule:
        ids = assets[domain]["train"]
        for _ in range(max(1, n_tokens // T)):
            s = int(torch.randint(0, ids.numel() - T - 1, (1,), generator=gen))
            yield domain, ids[s:s + T].unsqueeze(0), ids[s + 1:s + T + 1].unsqueeze(0)


def run(args):
    assets = load_assets()
    oc = oracle_ces()
    ckpt = REPO / "results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt"
    learner = load_ckpt(ckpt)
    learner.train()                                        # dropout active for kicks
    frozen = load_ckpt(ckpt)
    frozen.eval()
    engine = MorphPCEngine(learner)
    opt = torch.optim.AdamW(learner.parameters(), lr=args.lr, weight_decay=0.01)
    gate = (MarginGate(delta=args.delta) if args.gate == "margin"
            else SurpriseGate(gate_z=args.gate_z))
    anchor = (Anchor(learner, lam=args.anchor_lam)
              if args.arm == "gated_anchor" else None)
    learn = args.arm != "frozen"

    sched = regime_schedule(args.regime, assets, era_tokens=args.era_tokens)
    gen = torch.Generator().manual_seed(args.seed)
    # gate tag in the filename: parallel sweep cells must never share a tele
    # file (V2b lesson — three z* jobs interleaved one file)
    gtag = (f"m{args.delta}" if args.gate == "margin" else f"z{args.gate_z}")
    if anchor is not None:
        gtag += f"_lam{args.anchor_lam}"
    if args.tag:
        gtag += f"_{args.tag}"
    tele_path = OUT / f"v2_tele_{args.regime}_{args.arm}_{gtag}_s{args.seed}.jsonl"
    tele = tele_path.open("w")
    n = fires = 0
    sum_l = sum_f = sum_ro = sum_rof = fwd_eq = 0.0
    boundaries = []
    last_domain = None
    lat = {}                                               # boundary idx -> latency
    t0 = time.time()

    for domain, x, y in stream_windows(assets, sched, args.window, gen):
        if domain != last_domain and last_domain is not None:
            boundaries.append((n, domain))
        last_domain = domain
        with torch.no_grad():
            frozen.eval()
            lf = frozen(x)
            ce_f = F.cross_entropy(
                lf.reshape(-1, lf.shape[-1]).float(), y.reshape(-1)).item()
        if learn:
            def fire_fn(ce):
                return gate(ce)
            if anchor is not None:
                # two-phase: grads via fused serve (no step), anchor edit, step
                logits, ce_l, fired = engine.serve_and_learn(
                    x, y, opt=_NoStep(opt), fire=fire_fn)
                if fired:
                    anchor.pull(learner, opt)
                    opt.step()
                    anchor.track(learner)
            else:
                logits, ce_l, fired = engine.serve_and_learn(
                    x, y, opt=opt, fire=fire_fn)
        else:
            with torch.no_grad():
                ll = learner(x)
                ce_l = F.cross_entropy(
                    ll.reshape(-1, ll.shape[-1]).float(), y.reshape(-1)).item()
            fired = False
        n += 1
        fires += int(fired)
        if args.lr_decay_every and fired and fires % args.lr_decay_every == 0:
            # bit-shift lr decay (nanogpt lines 509-516): halve on a fired-kick
            # schedule — the long-horizon sign-drift fix, smoke-tested here
            # ahead of the V4 life
            for grp in opt.param_groups:
                grp["lr"] = grp["lr"] / 2.0
        fwd_eq += 1.0 + (KICK_COST if fired else 0.0)
        sum_l += ce_l
        sum_f += ce_f
        sum_ro += ce_l - oc[domain]
        sum_rof += ce_f - oc[domain]
        # adaptation latency: windows from the last boundary until
        # ce_l <= oracle + 0.1 bits
        if boundaries and boundaries[-1][0] not in lat and \
                ce_l <= oc[domain] + 0.1 * LN2:
            lat[boundaries[-1][0]] = n - boundaries[-1][0]
        if n % args.tele_every == 0 or fired:
            norm = sum(float(p.detach().norm()) for p in learner.parameters())
            tele.write(json.dumps(dict(
                t=n * args.window, domain=domain, ce_l=round(ce_l, 4),
                ce_f=round(ce_f, 4), fired=fired,
                fire_rate=round(fires / n, 4), norm=round(norm, 2))) + "\n")
    tele.close()

    # HC undefined when frozen is already near-oracle (stationary control)
    hc = (1.0 - sum_ro / sum_rof) if (learn and sum_rof / n > 0.02) else None
    row = dict(gate="V2", arm=args.arm, regime=args.regime, seed=args.seed,
               gate_kind=args.gate,
               gate_z=(args.gate_z if args.gate == "z" else None),
               delta=(args.delta if args.gate == "margin" else None),
               anchor_lam=(args.anchor_lam if anchor is not None else None),
               lr_decay_every=args.lr_decay_every or None,
               tag=args.tag or None,
               n_windows=n, preq_ce=round(sum_l / n, 4),
               preq_ce_frozen=round(sum_f / n, 4),
               regret_vs_frozen=round((sum_l - sum_f) / n, 4),
               headroom_captured=(round(hc, 4) if hc is not None else None),
               fire_rate=round(fires / n, 4),
               amortized_mult=round(fwd_eq / n, 4),
               adapt_latency_windows={str(k): v for k, v in lat.items()},
               wall_s=round(time.time() - t0, 1))
    with (OUT / "v2_live.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[V2] {args.regime}/{args.arm}/s{args.seed}: preq {row['preq_ce']} "
          f"(frozen {row['preq_ce_frozen']}) regret {row['regret_vs_frozen']} "
          f"HC {row['headroom_captured']} fire {row['fire_rate']} "
          f"mult {row['amortized_mult']}")


class _NoStep:
    """Optimizer shim: zero_grad passes through, step is deferred to the caller
    (the anchor edits grads between the sweep and the real step)."""

    def __init__(self, opt):
        self.opt = opt

    def zero_grad(self, **kw):
        self.opt.zero_grad(**kw)

    def step(self):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["frozen", "gated", "gated_anchor"],
                   required=True)
    p.add_argument("--regime", choices=["hard_shift", "drift", "novel_vocab",
                                        "stationary"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gate", choices=["z", "margin"], default="z")
    p.add_argument("--gate-z", type=float, default=0.25)
    p.add_argument("--delta", type=float, default=0.5,
                   help="margin gate threshold, nats")
    p.add_argument("--anchor-lam", type=float, default=1.0)
    p.add_argument("--lr-decay-every", type=int, default=0,
                   help="halve lr every N FIRED kicks (0 = off)")
    p.add_argument("--tag", default="",
                   help="extra tele-filename tag for parallel sweep cells")
    p.add_argument("--era-tokens", type=int, default=200_000)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--tele-every", type=int, default=10)
    args = p.parse_args()
    run(args)
