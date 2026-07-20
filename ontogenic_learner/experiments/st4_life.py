"""V4 — the lifetime gate: 10M served tokens, 5 era-cycles with revisits.

North-star gate V4 (NORTHSTAR_VALIDATION.md, Goal B). ONE process lives on the
cycling stream (wiki->shak->docs->code) x 5, serving every window and learning
through the V4cal-locked machinery:

    margin gate delta=0.25 (fast 0.9 / slow 0.999)   [V4cal: self-quiets]
    precision anchor lambda=0.01 (EMA 0.9999)        [V4cal: mild HC tax]
    bit-shift lr decay: halve per 2000 fired kicks   [long-horizon drift fix]
    serve dropout OFF (model.eval() throughout)      [kills the +0.11 artifact]

Goal-B instrumentation:
  * PROBES at life start + every era boundary: per-domain fixed tail windows
    (isolated: no_grad, own generator, no state touched) -> CE transfer
    matrix + top1 predictions; top1_agree is measured against the SNAPSHOT
    predictions taken at the end of the first full cycle (prereg: "vs era-1
    snapshot").
  * FORGETTING per domain: probe CE at each revisit vs the best end-of-own-era
    CE seen so far; the pass criterion is a non-increasing trend after the
    anchor engages.
  * NORM trajectory (equilibrium = plateau; kill = unbounded growth).
  * SAMPLED ZERO-MULTIPLY AUDIT: ~5% of windows run inside the dispatch
    catcher (audited_fired tracks kick coverage separately; anchor/AdamW
    arithmetic is outside the audited region — the V0 A4 declared boundary).
    Full-life ATTRIBUTABILITY is separate and total: the life is
    deterministic from (seed, ckpt chain), so every update is replayable.
  * CKPT/RESUME: full state (model, opt, anchor, gate, stream position, RNG)
    every --ckpt-every windows; --resume continues bit-continuously, so a box
    death costs at most one ckpt interval.

V4b rider: --domains wiki,docs (shard A) / shakespeare,code (shard B) runs the
same life on a sub-stream; the final ckpt keeps Adam exp_avg_sq so the fleet
Fisher-merge (G3-F2 recipe) can run weights-only afterwards.

Usage::

    python ontogenic_learner/experiments/st4_life.py --tag main --seed 0
    python ontogenic_learner/experiments/st4_life.py --tag riderA \
        --domains wiki,docs --era-tokens 250000
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pc_convrnn_charlm import MorphPCEngine                 # noqa: E402
from st1_streams import DOMAINS, OUT, load_assets, load_ckpt  # noqa: E402
from st2_live import (Anchor, DyadicAnchor, PrecisionAnchor,  # noqa: E402
                      MarginGate, _NoStep, oracle_ces)

LN2 = math.log(2)
KICK_COST = 0.38


def life_schedule(domains, era_tokens, cycles):
    """Domain order ROTATES per cycle (cycle c starts at domain c mod D).
    Pre-launch review finding: with a fixed monotone-easier order the
    absolute-margin gate is structurally silent at every easier-entry
    boundary after cycle 1 (CE drops with domain entropy even when the
    learner forgot) — rotation gives every domain ascending-CE entries
    across the life. The gate's asymmetric-shift blindness is documented
    for V4.1 (excess-surprise gate)."""
    sched = []
    for c in range(cycles):
        k = c % len(domains)
        order = domains[k:] + domains[:k]
        sched.extend((d, era_tokens) for d in order)
    return sched


def probe_windows(assets, domains, T, n=32, seed=104729):
    """Fixed per-domain probe windows from the TAIL splits (never streamed)."""
    probes = {}
    for d in domains:
        ids = assets[d]["tail"]
        gen = torch.Generator().manual_seed(seed)
        xs, ys = [], []
        for _ in range(n):
            s = int(torch.randint(0, ids.numel() - T - 1, (1,), generator=gen))
            xs.append(ids[s:s + T])
            ys.append(ids[s + 1:s + T + 1])
        probes[d] = (torch.stack(xs), torch.stack(ys))
    return probes


@torch.no_grad()
def run_probes(model, probes):
    """Isolated probe pass: CE + top1 predictions per domain."""
    out = {}
    for d, (x, y) in probes.items():
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             y.reshape(-1)).item()
        out[d] = (ce, logits.argmax(-1).reshape(-1))
    return out


def run(args):
    assets = load_assets()
    oc = oracle_ces()
    domains = tuple(args.domains.split(","))
    for d in domains:
        assert d in DOMAINS, d
    sched = life_schedule(domains, args.era_tokens, args.cycles)
    seg_windows = [(d, max(1, n // args.window)) for d, n in sched]
    total_windows = sum(n for _, n in seg_windows)

    learner = load_ckpt(REPO / "results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt")
    frozen = load_ckpt(REPO / "results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt")
    learner.eval()                          # serve dropout OFF (V4cal-locked)
    frozen.eval()
    engine = MorphPCEngine(learner)
    import tikitorch.optim as ttoptim
    if args.opt == "puretropical":
        # CLOSED shift-based: lr = 2^-lr_shift; decay = shift+1. sign + int8 counter.
        opt = ttoptim.PureTropicalSGD(learner.parameters(),
                                      lr_shift=args.lr_shift, momentum_bits=4)
    elif args.opt == "eftropical":
        # CLOSED error-feedback (cures the fire-always collapse).
        opt = ttoptim.EFTropicalSGD(learner.parameters(), lr_shift=args.lr_shift,
                                    ef=not args.no_ef, bound=args.ef_bound)
    elif args.opt == "hybrid":
        # CLOSED parity-hunt base: error feedback + frexp adaptive step
        # (Adam's 1/sqrt(v) surrogate) + optional temporal counter momentum.
        opt = ttoptim.EFTropicalSGD(learner.parameters(), lr_shift=args.lr_shift,
                                    ef=not args.no_ef, bound=args.ef_bound,
                                    adapt=not args.no_adapt, momentum=args.momentum)
    else:
        opt = torch.optim.AdamW(learner.parameters(), lr=args.lr, weight_decay=0.01)

    # --- consolidation anchor (the retention/Goal-B leg) ---
    kind = args.anchor
    if kind == "auto":                              # preserve prior defaults
        kind = ("adam" if args.opt == "adamw" and args.anchor_lam > 0
                else "dyadic" if args.opt == "eftropical" and args.anchor_lam > 0
                else "none")
    if kind == "adam":
        assert args.opt == "adamw", "adam anchor needs AdamW exp_avg_sq"
        anchor = Anchor(learner, lam=args.anchor_lam)
    elif kind == "dyadic":
        assert args.opt in ("eftropical", "hybrid"), \
            "dyadic anchor needs the optimizer's 'c' counter"
        anchor = DyadicAnchor(learner, k0=args.anchor_k0)
    elif kind == "precision":                       # base-agnostic: reads p.grad
        anchor = PrecisionAnchor(learner, k0=args.anchor_k0)
    else:
        anchor = None
    gate = MarginGate(delta=args.delta)
    probes = probe_windows(assets, domains, args.window)

    from tikitorch.audit import _make_catcher
    AUDIT_WAIVED = {"elementwise-divide", "transcendental"}

    def life_config(a):
        return dict(domains=domains, era_tokens=a.era_tokens,
                    cycles=a.cycles, window=a.window, delta=a.delta,
                    anchor_lam=a.anchor_lam, lr=a.lr, opt=a.opt,
                    lr_shift=a.lr_shift,
                    lr_decay_every=a.lr_decay_every, seed=a.seed,
                    fire_always=a.fire_always, anchor=a.anchor,
                    ef_bound=a.ef_bound, no_ef=a.no_ef, no_adapt=a.no_adapt,
                    momentum=a.momentum, anchor_k0=a.anchor_k0)

    ck_dir = OUT / "ckpts"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / f"v4_life_{args.tag}.pt"
    tele_path = OUT / f"v4_tele_{args.tag}.jsonl"
    probe_path = OUT / f"v4_probes_{args.tag}.jsonl"

    # ---- state (fresh or resumed) ------------------------------------------
    start_w = 0
    n = fires = halvings = 0
    sum_l = sum_f = sum_ro = sum_rof = fwd_eq = 0.0
    audited = audit_viol = audited_fired = 0
    snap_top1 = None                        # end-of-cycle-1 probe predictions
    gen = torch.Generator().manual_seed(args.seed)
    audit_gen = torch.Generator().manual_seed(args.seed + 31337)
    era_of_window, era_starts = [], []
    w = 0
    for i, (d, k) in enumerate(seg_windows):
        era_starts.append(w)
        era_of_window.extend([i] * k)
        w += k
    windows_per_cycle = total_windows // args.cycles

    if args.resume and ck_path.exists():
        st = torch.load(ck_path, map_location="cpu")
        assert st["tag"] == args.tag, (st["tag"], args.tag)
        assert st["config"] == life_config(args), \
            f"resume config mismatch: {st['config']} vs {life_config(args)}"
        learner.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if anchor is not None:
            anchor.theta = st["anchor_theta"]
        gate.mu_f, gate.mu_s = st["gate"]
        gen.set_state(st["gen_state"])
        audit_gen.set_state(st["audit_state"])
        (start_w, n, fires, halvings, sum_l, sum_f, sum_ro, sum_rof, fwd_eq,
         audited, audit_viol, audited_fired) = st["counters"]
        snap_top1 = st["snap_top1"]
        # drop rows the dead run logged past this ckpt (dedupe the replay)
        for path in (tele_path, probe_path):
            if path.exists():
                keep = [l for l in path.open()
                        if json.loads(l)["t"] <= start_w * args.window]
                path.write_text("".join(keep))
        print(f"[V4/{args.tag}] RESUMED at window {start_w}/{total_windows}",
              flush=True)
    tele = tele_path.open("a" if start_w else "w")
    probe_log = probe_path.open("a" if start_w else "w")

    def log_probes(where):
        res = run_probes(learner, probes)
        row = dict(t=n * args.window, where=where,
                   **{f"ce_{d}": round(res[d][0], 4) for d in domains})
        if snap_top1 is not None:
            row.update({f"agree_{d}":
                        round((res[d][1] == snap_top1[d]).float().mean().item(), 4)
                        for d in domains})
        norm = sum(float(p.detach().norm()) for p in learner.parameters())
        row["norm"] = round(norm, 2)
        probe_log.write(json.dumps(row) + "\n")
        probe_log.flush()
        return res

    def save_ckpt(w_idx):
        tmp = ck_path.with_suffix(".pt.tmp")
        torch.save(dict(
            model=learner.state_dict(), opt=opt.state_dict(),
            anchor_theta=(anchor.theta if anchor is not None else None),
            gate=(gate.mu_f, gate.mu_s),
            gen_state=gen.get_state(), audit_state=audit_gen.get_state(),
            counters=(w_idx, n, fires, halvings, sum_l, sum_f, sum_ro,
                      sum_rof, fwd_eq, audited, audit_viol, audited_fired),
            snap_top1=snap_top1, tag=args.tag,
            config=life_config(args),
        ), tmp)
        import os
        os.replace(tmp, ck_path)            # atomic: sync/kill never sees a
                                            # half-written ckpt

    if start_w == 0:
        log_probes("birth")
    t0 = time.time()

    for w_idx in range(start_w, total_windows):
        seg = era_of_window[w_idx]
        domain = seg_windows[seg][0]
        ids = assets[domain]["train"]
        s = int(torch.randint(0, ids.numel() - args.window - 1, (1,),
                              generator=gen))
        x = ids[s:s + args.window].unsqueeze(0)
        y = ids[s + 1:s + args.window + 1].unsqueeze(0)

        with torch.no_grad():
            lf = frozen(x)
            ce_f = F.cross_entropy(
                lf.reshape(-1, lf.shape[-1]).float(), y.reshape(-1)).item()

        fire_fn = ((lambda ce: (gate(ce), True)[1]) if args.fire_always
                   else (lambda ce: gate(ce)))
        do_audit = float(torch.rand((), generator=audit_gen)) < args.audit_frac
        hits = []
        if do_audit:
            with _make_catcher(hits):
                logits, ce_l, fired = engine.serve_and_learn(
                    x, y, opt=_NoStep(opt), fire=fire_fn)
            audited += 1
            audited_fired += int(fired)
            bad = Counter(c for c, _ in hits
                          if c not in AUDIT_WAIVED)
            audit_viol += sum(bad.values())
        else:
            logits, ce_l, fired = engine.serve_and_learn(
                x, y, opt=_NoStep(opt), fire=fire_fn)
        if fired:
            if anchor is not None:
                anchor.pull(learner, opt)
            opt.step()
            fires += 1
            if args.lr_decay_every and fires % args.lr_decay_every == 0:
                halvings += 1
                for grp in opt.param_groups:
                    if "lr_shift" in grp:
                        grp["lr_shift"] += 1      # bit-shift decay, closed form
                    grp["lr"] = grp["lr"] / 2.0

        # anchor EMA on a per-WINDOW clock (review finding: per-fire tracking
        # at ~3k fires leaves theta ~= birth weights — pull-to-birth, not
        # consolidation; per-window half-life ~6.9k windows ~= 0.66M tokens)
        if anchor is not None:
            anchor.track(learner)
        n += 1
        # cost boundary: learner serve+kick only (V0 constant); anchor pull
        # ~O(P) adds per fire and the frozen twin/probes are instrumentation
        fwd_eq += 1.0 + (KICK_COST if fired else 0.0)
        sum_l += ce_l
        sum_f += ce_f
        sum_ro += ce_l - oc[domain]
        sum_rof += ce_f - oc[domain]

        if n % args.tele_every == 0 or fired:
            norm = sum(float(p.detach().norm()) for p in learner.parameters())
            tele.write(json.dumps(dict(
                t=n * args.window, domain=domain, ce_l=round(ce_l, 4),
                ce_f=round(ce_f, 4), fired=fired, fires=fires,
                halvings=halvings,
                fire_rate=round(fires / n, 4), norm=round(norm, 2),
                lr=opt.param_groups[0]["lr"])) + "\n")
            tele.flush()

        nxt = w_idx + 1
        if nxt in era_starts[1:] or nxt == total_windows:
            res = log_probes(f"era_end_{era_of_window[w_idx]}")
            if snap_top1 is None and nxt >= windows_per_cycle:
                snap_top1 = {d: res[d][1] for d in domains}
                probe_log.write(json.dumps(dict(
                    t=n * args.window, where="cycle1_snapshot")) + "\n")
                probe_log.flush()
                print(f"[V4/{args.tag}] cycle-1 snapshot taken at window "
                      f"{nxt}", flush=True)
        if nxt % args.ckpt_every == 0 or nxt == total_windows:
            save_ckpt(nxt)
            print(f"[V4/{args.tag}] {nxt}/{total_windows} windows "
                  f"({time.time()-t0:.0f}s) preq {sum_l/n:.4f} "
                  f"fire {fires/n:.4f} lr {opt.param_groups[0]['lr']:.2e}",
                  flush=True)
        if args.max_windows and n >= args.max_windows:
            print(f"[V4/{args.tag}] --max-windows {args.max_windows} reached; "
                  f"stopping (pre-cut mode)", flush=True)
            break
    tele.close()
    probe_log.close()

    # ---- verdict row --------------------------------------------------------
    hc = (1.0 - sum_ro / sum_rof) if sum_rof / n > 0.02 else None
    row = dict(gate="V4", tag=args.tag, domains=args.domains,
               era_tokens=args.era_tokens, cycles=args.cycles,
               seed=args.seed, delta=args.delta, anchor_lam=args.anchor_lam,
               opt=args.opt,
               lr_decay_every=args.lr_decay_every, lr_halvings=halvings,
               n_windows=n, tokens=n * args.window,
               preq_ce=round(sum_l / n, 4),
               preq_ce_frozen=round(sum_f / n, 4),
               regret_vs_frozen=round((sum_l - sum_f) / n, 4),
               headroom_captured=(round(hc, 4) if hc is not None else None),
               fire_rate=round(fires / n, 4),
               amortized_mult=round(fwd_eq / n, 4),
               audit_windows=audited, audited_fired=audited_fired,
               audit_violations=audit_viol,
               audit_frac=round(audited / n, 4),
               fire_always=args.fire_always,
               wall_s=round(time.time() - t0, 1))
    with (OUT / "v4_life.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[V4/{args.tag}] LIFE COMPLETE: preq {row['preq_ce']} "
          f"(frozen {row['preq_ce_frozen']}) fire {row['fire_rate']} "
          f"mult {row['amortized_mult']} audit {audited}w/{audit_viol}viol")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--era-tokens", type=int, default=500_000)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--opt", choices=["adamw", "puretropical", "eftropical",
                                     "hybrid"], default="adamw")
    p.add_argument("--anchor", choices=["auto", "none", "adam", "dyadic",
                                        "precision"], default="auto",
                   help="consolidation anchor; 'auto' = prior per-opt default")
    p.add_argument("--lr-shift", type=int, default=10)
    p.add_argument("--ef-bound", type=float, default=0.0,
                   help="EFTropicalSGD soft-bound magnitude (0 = off)")
    p.add_argument("--no-ef", action="store_true",
                   help="ablate error feedback (EFTropicalSGD -> naked sign-SGD)")
    p.add_argument("--no-adapt", action="store_true",
                   help="hybrid: ablate the frexp adaptive step")
    p.add_argument("--momentum", action="store_true",
                   help="hybrid: enable temporal counter momentum")
    p.add_argument("--anchor-k0", type=int, default=7,
                   help="Dyadic/Precision anchor base pull shift (larger = weaker)")
    p.add_argument("--max-windows", type=int, default=0,
                   help="cap served windows (pre-cut mode; 0 = full life)")
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--anchor-lam", type=float, default=0.01)
    p.add_argument("--lr-decay-every", type=int, default=2000)
    p.add_argument("--tele-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2500)
    p.add_argument("--audit-frac", type=float, default=0.05)
    p.add_argument("--fire-always", action="store_true",
                   help="bypass the gate (the sign-drift adversary rider)")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args)
