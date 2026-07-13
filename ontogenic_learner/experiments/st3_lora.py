"""V3 — honest incumbents: the adapting dense GPT on the same streams (gate V3).

North-star gate V3 (NORTHSTAR_VALIDATION.md). The incumbent is NOT a frozen
GPU — it is a dense GPT that adapts, the way GPUs actually adapt in the field:
buffer the stream, periodically run K backprop steps (LoRA-adapters or full
FT), merge, keep serving. This script builds that baseline (none exists) and
prices it on the SAME streams, windows, and prequential accounting as V2.

Fairness by construction:
  * ISO-QUALITY WARM START — the dense GPT is trained on text8 BPE-1024 and
    early-stopped at the M1 S1 checkpoint's banked val bpb (2.0273), so both
    substrates enter the stream at equal capability. (The dense model could go
    lower — capability ceiling is reported, not hidden.)
  * SAME WINDOWS — st2_live.stream_windows with the same seed reproduces the
    identical window sequence V2 served.
  * LoRA TUNED TO WIN — rank x cadence x inner-K x lr sweep, rehearsal buffer
    allowed, plus full-FT-periodic and per-window online-BP arms.
  * FULL-CYCLE TAX COUNTED — update MACs (fwd+bwd = 3x serve on every
    buffered token it revisits) and the activation-graph peak-memory class
    are measured per arm, beside serve MACs.

LoRA placement: FFN (mlp up/down) + head via module swap. Attention
projections are excluded because nn.MultiheadAttention reads out_proj.weight
directly (module swap never runs); the rank sweep spans the capacity range.

Usage::

    python ontogenic_learner/experiments/st3_lora.py --warm-start [--bpb-target 2.0273]
    python ontogenic_learner/experiments/st3_lora.py --arm lora --rank 8 \
        --cadence 50000 --inner-k 32 --regime hard_shift --seed 0
"""
import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nanogpt_shakespeare import BPEData, DenseGPT                # noqa: E402
from lm_energy_projection import transformer_macs                # noqa: E402
from st1_streams import OUT, load_assets                         # noqa: E402
from st2_live import stream_windows, oracle_ces                  # noqa: E402
from st1_streams import regime_schedule                          # noqa: E402

LN2 = math.log(2)
WARM_CKPT = OUT / "ckpts" / "v3_dense_warm.pt"


# ---------------------------------------------------------------------------
# Iso-quality warm start
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_bpb(model, tok, dev, batches=16, batch=24, T=96, seed=1):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = n = 0
    for _ in range(batches):
        x, y = tok.batch("val", batch, T, gen)
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1), reduction="sum").item()
        n += y.numel()
    model.train()
    return (tot / n) * tok.val_tokens_per_byte / LN2


def warm_start(args, dev):
    """Train the dense GPT on text8 to the M1 S1 checkpoint's bpb, then STOP.
    Both substrates enter the streams at the same measured capability."""
    torch.manual_seed(0)
    tok = BPEData(REPO / "data" / "text8_20m.txt", 1024)
    cfg = dict(vocab=1024, d=args.d_model, n_heads=args.n_heads,
               n_layers=args.n_layers, d_ff=args.d_ff, max_seq_len=args.window)
    model = DenseGPT(cfg["vocab"], cfg["d"], cfg["n_heads"], cfg["n_layers"],
                     cfg["d_ff"], cfg["max_seq_len"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.warm_lr,
                            weight_decay=0.01)
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    best, hit_step = float("inf"), None
    model.train()
    eval_every = 250
    for step in range(args.warm_steps):
        x, y = tok.batch("train", 24, args.window, gen)
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        model.compute_loss(x, y).backward()
        opt.step()
        if (step + 1) % eval_every == 0:
            bpb = val_bpb(model, tok, dev)
            best = min(best, bpb)
            print(f"[V3/warm] step {step+1} val bpb {bpb:.4f} "
                  f"(target {args.bpb_target})", flush=True)
            if bpb <= args.bpb_target + 0.1:
                # near the target: tighten the eval cadence so the early-stop
                # cannot overshoot far BELOW target (a too-strong incumbent is
                # as unfair as a too-weak one)
                eval_every = 50
            if bpb <= args.bpb_target:
                hit_step = step + 1
                break
    bpb = val_bpb(model, tok, dev)
    iso = abs(bpb - args.bpb_target) <= 0.02          # TWO-sided
    WARM_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": cfg, "val_bpb": bpb, "bpb_target": args.bpb_target,
                "iso_quality": iso,
                "steps": hit_step or args.warm_steps}, WARM_CKPT)
    row = dict(gate="V3_warm", val_bpb=round(bpb, 4), target=args.bpb_target,
               start_bpb_delta=round(bpb - args.bpb_target, 4),
               iso_quality=iso,
               steps=hit_step or args.warm_steps,
               wall_s=round(time.time() - t0, 1), **{k: v for k, v in cfg.items()
                                                     if k != "vocab"})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "v3_lora.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[V3/warm] saved {WARM_CKPT} at bpb {bpb:.4f} "
          f"({'ISO-QUALITY' if row['iso_quality'] else 'MISSED TARGET — flag'})")


def load_warm(dev):
    ck = torch.load(WARM_CKPT, map_location="cpu")
    c = ck["config"]
    model = DenseGPT(c["vocab"], c["d"], c["n_heads"], c["n_layers"],
                     c["d_ff"], c["max_seq_len"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    return model, c


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


def inject_lora(model, r):
    """FFN up/down + head. Returns the trainable LoRA parameters."""
    for blk in model.blocks:
        blk.mlp[0] = LoRALinear(blk.mlp[0], r)
        blk.mlp[2] = LoRALinear(blk.mlp[2], r)
    model.head = LoRALinear(model.head, r)
    for n, p in model.named_parameters():
        if not (n.endswith(".A") or n.endswith(".B")):
            p.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# The adapting-incumbent stream run
# ---------------------------------------------------------------------------

def macs_per_tok(cfg):
    return transformer_macs(cfg["d"], cfg["n_layers"], cfg["n_heads"],
                            cfg["max_seq_len"], cfg["vocab"],
                            d_ff_mult=cfg["d_ff"] / cfg["d"])["total_mac"]


def run(args, dev):
    torch.manual_seed(args.seed)            # LoRA A-init + update dropout
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    assets = load_assets()
    oc = oracle_ces()                       # MorphSSM oracles: reference only
    learner, cfg = load_warm(dev)
    frozen, _ = load_warm(dev)
    frozen.eval()
    mpt = macs_per_tok(cfg)

    trainable = None
    if args.arm == "lora":
        trainable = inject_lora(learner, args.rank)
        learner.to(dev)
    elif args.arm in ("full_ft", "online_bp"):
        trainable = list(learner.parameters())
    opt = (torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
           if trainable else None)
    # serve ALWAYS in eval (dropout off): a train-mode serving artifact here
    # would bias the comparison in OUR favor (V2's ran the other way — against
    # us — and was left in place as conservative; this one would not be)
    learner.eval()

    sched = regime_schedule(args.regime, assets, era_tokens=args.era_tokens)
    gen = torch.Generator().manual_seed(args.seed)
    buffer = deque(maxlen=args.buffer)
    n = 0
    sum_l = sum_f = sum_ro = sum_rof = 0.0
    upd_events = upd_steps = upd_tokens = 0
    tokens_since = 0
    mu_fast = mu_slow = None                # 'trig' cadence detector
    peak_serve = peak_update = 0
    cuda = dev.type == "cuda"
    t0 = time.time()

    def update_from_buffer():
        nonlocal upd_events, upd_steps, upd_tokens, peak_update
        if not buffer or opt is None:
            return
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        idx_gen = torch.Generator().manual_seed(args.seed * 7919 + upd_events)
        for _ in range(args.inner_k):
            pick = torch.randint(0, len(buffer), (min(8, len(buffer)),),
                                 generator=idx_gen)
            xb = torch.cat([buffer[i][0] for i in pick.tolist()]).to(dev)
            yb = torch.cat([buffer[i][1] for i in pick.tolist()]).to(dev)
            opt.zero_grad(set_to_none=True)
            learner.compute_loss(xb, yb).backward()
            opt.step()
            upd_steps += 1
            upd_tokens += yb.numel()
        upd_events += 1
        if cuda:
            peak_update = max(peak_update, torch.cuda.max_memory_allocated())

    serve_base = 0
    for domain, x, y in stream_windows(assets, sched, args.window, gen):
        x, y = x.to(dev), y.to(dev)
        with torch.no_grad():
            lf = frozen(x)
            ce_f = F.cross_entropy(
                lf.reshape(-1, lf.shape[-1]).float(), y.reshape(-1)).item()
            if cuda and n == 0:
                # serve memory probe: learner forward ALONE (the frozen twin
                # is measurement scaffolding, not deployment)
                serve_base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            ll = learner(x)                 # prequential: theta(t-1) serves
            ce_l = F.cross_entropy(
                ll.reshape(-1, ll.shape[-1]).float(), y.reshape(-1)).item()
            if cuda and n == 0:
                peak_serve = torch.cuda.max_memory_allocated()
        n += 1
        sum_l += ce_l
        sum_f += ce_f
        sum_ro += ce_l - oc[domain]
        sum_rof += ce_f - oc[domain]
        buffer.append((x.cpu(), y.cpu()))
        tokens_since += y.numel()
        if args.arm == "online_bp":
            if cuda and upd_events == 0:
                torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True)
            learner.train()
            learner.compute_loss(x, y).backward()
            opt.step()
            learner.eval()
            if cuda and upd_events == 0:
                peak_update = max(peak_update, torch.cuda.max_memory_allocated())
            upd_events += 1
            upd_steps += 1
            upd_tokens += y.numel()
        elif args.arm in ("lora", "full_ft"):
            fire = False
            if args.cadence == "trig":
                if mu_fast is None:
                    mu_fast = mu_slow = ce_l
                mu_fast = 0.9 * mu_fast + 0.1 * ce_l
                mu_slow = 0.999 * mu_slow + 0.001 * ce_l
                fire = (mu_fast - mu_slow) > 0.5 and tokens_since >= 2000
            else:
                fire = tokens_since >= int(args.cadence)
            if fire:
                learner.train()
                update_from_buffer()
                learner.eval()
                tokens_since = 0

    # update cycle cost: full_ft/online_bp = fwd + full bwd (grad-input +
    # grad-weight) = 3x fwd; LoRA skips the grad-WEIGHT matmuls for every
    # frozen base weight but still traverses the graph for grad-input = 2x
    # fwd, plus the (small) adapter terms — overcharging LoRA 3x would bias
    # the joules comparison in OUR favor
    upd_mult = 2.0 if args.arm == "lora" else 3.0
    update_macs = upd_mult * upd_tokens * mpt
    hc = (1.0 - sum_ro / sum_rof) if sum_rof / n > 0.02 else None
    row = dict(gate="V3", arm=args.arm, rank=args.rank,
               cadence=str(args.cadence), inner_k=args.inner_k, lr=args.lr,
               regime=args.regime, seed=args.seed, n_windows=n,
               preq_ce=round(sum_l / n, 4), preq_ce_frozen=round(sum_f / n, 4),
               regret_vs_frozen=round((sum_l - sum_f) / n, 4),
               # NOT V2's headroom_captured: dense arms vs the MorphSSM
               # oracles with a dense-frozen denominator (unbounded, reference
               # only) — the cross-substrate comparison is absolute preq_ce
               hc_vs_morph_oracles=(round(hc, 4) if hc is not None else None),
               upd_events=upd_events, upd_steps=upd_steps,
               upd_tokens=upd_tokens, upd_mult=upd_mult,
               macs_per_tok=mpt,
               update_macs_over_serve=round(update_macs / (n * args.window * mpt), 4),
               peak_mem_serve_mb=round(peak_serve / 2**20, 1),
               serve_base_mb=round(serve_base / 2**20, 1),
               peak_mem_update_mb=round(peak_update / 2**20, 1),
               n_trainable=sum(p.numel() for p in (trainable or [])),
               wall_s=round(time.time() - t0, 1))
    with (OUT / "v3_lora.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[V3] {args.regime}/{args.arm}(r{args.rank},c{args.cadence},"
          f"k{args.inner_k},lr{args.lr})/s{args.seed}: preq {row['preq_ce']} "
          f"(frozen {row['preq_ce_frozen']}) regret {row['regret_vs_frozen']} "
          f"upd_macs/serve {row['update_macs_over_serve']} "
          f"mem {row['peak_mem_serve_mb']}/{row['peak_mem_update_mb']}MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warm-start", action="store_true")
    p.add_argument("--bpb-target", type=float, default=2.0273)
    p.add_argument("--warm-steps", type=int, default=20000)
    p.add_argument("--warm-lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--arm", choices=["frozen", "lora", "full_ft", "online_bp"],
                   default=None)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--cadence", default="50000",
                   help="tokens between update events, or 'trig'")
    p.add_argument("--inner-k", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer", type=int, default=64,
                   help="rehearsal ring buffer, in windows")
    p.add_argument("--regime", choices=["hard_shift", "drift", "novel_vocab",
                                        "stationary"], default="hard_shift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--era-tokens", type=int, default=200_000)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    if args.warm_start:
        warm_start(args, dev)
    if args.arm:
        run(args, dev)
