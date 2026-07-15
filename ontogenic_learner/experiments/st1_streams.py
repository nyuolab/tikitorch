"""V1 — stream calibration + oracle bounds (NORTHSTAR_VALIDATION.md, gate V1).

Builds the BPE-1024 nonstationary streams, verifies tokenizer identity against
the M1 S1 checkpoint, trains per-domain oracle specialists, and computes the
ADAPTATION HEADROOM that decides whether each stream regime is admissible
(headroom >= 0.3 bits/token) — i.e., whether a frozen model provably needs
adaptation there. The frozen baseline and oracle numbers become V2's regret
denominators and D2's reference bands.

Regimes: hard_shift (wiki->shakespeare->docs->code) | drift (rotating mixture
blocks) | novel_vocab (code-symbol injection into wiki) | stationary (wiki
only; the gate-calibration negative control).

Usage::

    python ontogenic_learner/experiments/st1_streams.py --build-assets   # tokenize (cheap)
    python ontogenic_learner/experiments/st1_streams.py --identity-check --ckpt <p>
    python ontogenic_learner/experiments/st1_streams.py --oracle wiki --ckpt <p>
    python ontogenic_learner/experiments/st1_streams.py --headroom
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

from pc_convrnn_charlm import MorphPCEngine, build_model   # noqa: E402
from nanogpt_shakespeare import BPEData                    # noqa: E402

DOMAINS = ("wiki", "shakespeare", "docs", "code")
STREAM_DIR = REPO / "data" / "experience_stream"
ASSET = REPO / "data" / "experience_stream_bpe.pt"
OUT = REPO / "ontogenic_learner" / "results" / "st_stream"
LN2 = math.log(2)


def tokenizer():
    """The M1 text8 BPE-1024 tokenizer (deterministic rebuild, cached)."""
    return BPEData(REPO / "data" / "text8_20m.txt", 1024)


def build_assets():
    tok = tokenizer()
    assets = {}
    for d in DOMAINS:
        text = (STREAM_DIR / f"{d}.txt").read_text(encoding="utf-8",
                                                   errors="replace")
        ids = torch.tensor(tok.tok.encode(text).ids, dtype=torch.long)
        n = int(len(ids) * 0.9)
        assets[d] = dict(train=ids[:n], tail=ids[n:])
        print(f"[V1] {d}: {len(ids)} tokens (train {n}, tail {len(ids)-n})")
    torch.save(assets, ASSET)
    print(f"[V1] assets -> {ASSET}")


def load_assets():
    return torch.load(ASSET)


def batch_from(ids, batch, T, gen):
    starts = torch.randint(0, ids.numel() - T - 1, (batch,), generator=gen)
    x = torch.stack([ids[s:s + T] for s in starts])
    y = torch.stack([ids[s + 1:s + T + 1] for s in starts])
    return x, y


@torch.no_grad()
def eval_ce(model, ids, batch=24, T=96, batches=8, seed=1):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = n = 0
    for _ in range(batches):
        x, y = batch_from(ids, batch, T, gen)
        logits = model(x).reshape(-1, model.output_head.out_features).float()
        tot += F.cross_entropy(logits, y.reshape(-1), reduction="sum").item()
        n += y.numel()
    model.train()
    return tot / n


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu")
    ns = argparse.Namespace(d_model=ck["config"]["d_model"],
                            n_layers=ck["config"]["n_layers"],
                            kernel=ck["config"]["kernel"], dropout=0.1,
                            tau=0.5, lam_min=ck["config"]["lam_min"])
    model = build_model(ck["arch"], ck["config"]["vocab"], ns)
    model.load_state_dict(ck["state_dict"])
    model.set_tau(0.0)
    return model


def identity_check(args):
    """The rebuilt tokenizer must reproduce the ckpt's banked text8 bpb (~2.027).

    BPE training is deterministic on fixed input/config; this check catches any
    drift between the box-trained tokenizer the ckpt learned under and the
    rebuild. tokens/byte on text8-val converts CE/token to bpb."""
    tok = tokenizer()
    model = load_ckpt(args.ckpt)
    ce = eval_ce(model, tok.val_ids)
    bpb = ce * tok.val_tokens_per_byte / LN2
    ok = abs(bpb - 2.0273) < 0.05
    row = dict(gate="V1_identity", ckpt=str(args.ckpt), val_ce=ce,
               val_bpb=round(bpb, 4), banked_bpb=2.0273, ok=ok)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "v1_streams.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[V1] tokenizer identity: bpb={bpb:.4f} vs banked 2.0273 "
          f"({'PASS' if ok else 'FAIL — tokenizer drift, STOP'})")
    return ok


def train_oracle(args):
    """Per-domain oracle: warm-start the S1 ckpt, cascade-train to convergence
    on the domain, record best tail CE. This is the ceiling the stream's
    headroom is measured against."""
    assets = load_assets()
    ids_tr, ids_tail = assets[args.oracle]["train"], assets[args.oracle]["tail"]
    model = load_ckpt(args.ckpt)
    frozen_ce = eval_ce(model, ids_tail)
    engine = MorphPCEngine(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)
    best, since = frozen_ce, 0
    t0 = time.time()
    model.train()
    for step in range(args.oracle_steps):
        x, y = batch_from(ids_tr, 24, 96, gen)
        engine.train_step(x, y, opt, credit="cascade")
        if (step + 1) % 500 == 0:
            ce = eval_ce(model, ids_tail)
            print(f"[V1/{args.oracle}] step {step+1} tail CE {ce:.4f} "
                  f"(frozen {frozen_ce:.4f})", flush=True)
            if ce < best - 0.005:
                best, since = ce, 0
            else:
                since += 1
                if since >= 2:                      # converged: 1k steps flat
                    break
    row = dict(gate="V1_oracle", domain=args.oracle, frozen_ce=frozen_ce,
               oracle_ce=best, headroom_nats=frozen_ce - best,
               headroom_bits=round((frozen_ce - best) / LN2, 4),
               steps=step + 1, wall_s=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "v1_streams.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    ck_dir = OUT / "ckpts"
    ck_dir.mkdir(exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "domain": args.oracle, "oracle_ce": best},
               ck_dir / f"oracle_{args.oracle}.pt")
    print(f"[V1] oracle {args.oracle}: frozen {frozen_ce:.4f} -> oracle "
          f"{best:.4f} (headroom {(frozen_ce-best)/LN2:.3f} bits/tok)")


# regime schedules: list of (domain, n_tokens) segments over the TRAIN splits
def regime_schedule(regime, assets, era_tokens=200_000):
    if regime == "hard_shift":
        return [(d, era_tokens) for d in DOMAINS]
    if regime == "stationary":
        return [("wiki", 4 * era_tokens)]
    if regime == "two_cycle":               # FIXED order, twice: the margin
        # gate's re-entry deadband case (E0-1) — cycle 2 revisits every domain
        # in the same monotone-easier order that silences absolute-margin gates
        return [(d, era_tokens) for _ in range(2) for d in DOMAINS]
    if regime == "drift":                      # rotating mixture, 2k blocks
        sched = []
        blocks = 4 * era_tokens // 2000
        for b in range(blocks):
            phase = b / blocks * len(DOMAINS)
            d = DOMAINS[int(phase) % len(DOMAINS)]
            sched.append((d, 2000))
        return sched
    if regime == "novel_vocab":                # wiki with rising code density
        sched = []
        for i in range(8):
            sched.append(("wiki", era_tokens // 4))
            sched.append(("code", (i + 1) * era_tokens // 32))
        return sched
    raise ValueError(regime)


def headroom(args):
    """Regime admissibility: schedule-weighted (frozen - oracle) in bits/tok."""
    rows = [json.loads(l) for l in (OUT / "v1_streams.jsonl").open()]
    oracle = {r["domain"]: r for r in rows if r.get("gate") == "V1_oracle"}
    missing = [d for d in DOMAINS if d not in oracle]
    assert not missing, f"oracles missing: {missing}"
    assets = load_assets()
    verdicts = {}
    for regime in ("hard_shift", "drift", "novel_vocab", "stationary"):
        sched = regime_schedule(regime, assets)
        tot = sum(n for _, n in sched)
        h = sum(n * oracle[d]["headroom_bits"] for d, n in sched) / tot
        admissible = h >= 0.3 if regime != "stationary" else h < 0.15
        verdicts[regime] = (h, admissible)
        row = dict(gate="V1_headroom", regime=regime,
                   headroom_bits_per_tok=round(h, 4),
                   bar=(">=0.3" if regime != "stationary" else "<0.15 (control)"),
                   admissible=admissible)
        with (OUT / "v1_streams.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[V1] {regime:>12}: headroom {h:.3f} bits/tok -> "
              f"{'ADMISSIBLE' if admissible else 'NOT ADMISSIBLE'}")
    ok = all(v for _, v in verdicts.values())
    print(f"[V1] overall: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-assets", action="store_true")
    p.add_argument("--identity-check", action="store_true")
    p.add_argument("--oracle", choices=DOMAINS, default=None)
    p.add_argument("--headroom", action="store_true")
    p.add_argument("--ckpt", type=Path,
                   default=REPO / "results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt")
    p.add_argument("--oracle-steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.build_assets:
        build_assets()
    if args.identity_check:
        ok = identity_check(args)
        raise SystemExit(0 if ok else 1)
    if args.oracle:
        train_oracle(args)
    if args.headroom:
        ok = headroom(args)
        raise SystemExit(0 if ok else 1)
