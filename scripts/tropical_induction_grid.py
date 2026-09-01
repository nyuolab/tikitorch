"""tropical_induction_grid.py — disentangle SCORE vs AGGREGATION in the
induction/copy probe, with a parameter-free, scale-fair design.

Background: the 6-arm probe (scripts/tropical_induction.py) found that a fully
multiplier-free attention — L1 score (-sum_d|Q-K|) + hard max-plus value pull
max_j(S+V) — hits 1.000 induction accuracy (= softmax dot-product ceiling),
while the SAME L1 score with soft softmax aggregation at a FIXED tau=0.25 hit
only 0.027. Two confounds:
  (1) the winning cell differs from the maxplus baseline in BOTH score and agg;
  (2) tau=0.25 is misscaled for an L1 distance, so the soft-agg failure may be a
      temperature artifact.

A LEARNABLE tau does NOT fix (2): it is cold-start-trapped for routing — at init
the routing is random, sharpening random attention is worse than averaging it, so
the loss gradient drives tau UP (flatter) and routing never forms (measured:
dot_soft with learnable tau = chance, even warm-started at 0.25). So we use FIXED
tau and SWEEP it, which keeps the model in the sharp regime while it learns.

Design (parameter-free, scale-fair via empirical tau-sweep):
  scores      : dot (/sqrt d) | maxplus (max_d(Q+K)) | l1 (-sum_d|Q-K|) | hint (sum_d min)
  HARD cells  : agg = max_j(S+V), parameter-free       <- isolates SCORE under tropical agg
  SOFT cells  : agg = softmax(S/tau)@V, tau swept       <- best fixed tau per score (fair)

Targeted config list (not a full grid):
  - all 4 scores x HARD                         (the score effect, robust/parameter-free)
  - dot  x SOFT @ tau=0.25                       (reproduce the softmax ceiling = 1.000)
  - l1   x SOFT @ tau in {1,.25,.0625,.0156}     (can ANY fixed tau rescue soft L1? kills confound 2)
  - maxplus x SOFT @ tau=0.25                    (the original baseline reference)
  - conv  (negative control)

Each config over N seeds; report mean +/- std and per-seed.

Usage: PYTHONPATH=. python scripts/tropical_induction_grid.py
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def induction_batch(B, L, vocab, gen, device):
    """seq = [r ; r], r = L//2 distinct tokens. Returns x,(B,L-1) y,(B,L-1),
    and a mask of the second-half (induction-solvable) target positions."""
    assert L % 2 == 0 and vocab >= L // 2
    half = L // 2
    seq = torch.empty(B, L, dtype=torch.long, device=device)
    for b in range(B):
        r = torch.randperm(vocab, generator=gen)[:half]
        seq[b, :half] = r
        seq[b, half:] = r
    x, y = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros(B, L - 1, dtype=torch.bool, device=device)
    mask[:, half:] = True
    return x, y, mask


def _causal_neg(scores):
    L = scores.shape[-1]
    m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=scores.device), 1)
    return scores.masked_fill(m, float("-inf"))


class Attn(nn.Module):
    """Multi-head attention; pluggable SCORE x AGGREGATION, causal, FIXED tau.

    score in {dot, maxplus, l1, hint}; agg in {soft, hard}."""
    def __init__(self, d, heads, score, agg, tau=0.25):
        super().__init__()
        self.h, self.dh, self.score_kind, self.agg, self.tau = heads, d // heads, score, agg, tau
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)

    def score(self, q, k):  # q,k: (B,H,L,dh) -> (B,H,L,L)
        if self.score_kind == "dot":
            return torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.dh)
        if self.score_kind == "maxplus":
            return (q.unsqueeze(-2) + k.unsqueeze(-3)).max(dim=-1).values
        if self.score_kind == "l1":
            return -(q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(dim=-1)
        if self.score_kind == "hint":
            return torch.minimum(q.unsqueeze(-2), k.unsqueeze(-3)).sum(dim=-1)
        raise ValueError(self.score_kind)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        s = _causal_neg(self.score(q, k))
        if self.agg == "hard":                  # tropical max-plus value pull
            out = (s.unsqueeze(-1) + v.unsqueeze(-3)).max(dim=-2).values
        else:                                   # soft, FIXED tau
            out = torch.einsum("bhij,bhjd->bhid", F.softmax(s / self.tau, dim=-1), v)
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class Conv(nn.Module):
    def __init__(self, d, ksize=7):
        super().__init__()
        self.dw = nn.Conv1d(d, d, ksize, groups=d, padding=ksize - 1)
        self.pw = nn.Linear(d, d); self.ks = ksize

    def forward(self, x):
        h = self.dw(x.transpose(1, 2))[..., : -(self.ks - 1)].transpose(1, 2)
        return self.pw(F.gelu(h))


class TinyLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, cfg, L):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            if cfg["mixer"] == "conv":
                mix = Conv(d)
            else:
                mix = Attn(d, heads, cfg["score"], cfg["agg"], cfg.get("tau", 0.25))
            self.blocks.append(nn.ModuleList([nn.LayerNorm(d), mix, nn.LayerNorm(d),
                                              nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                                            nn.Linear(4 * d, d))]))
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for ln1, mix, ln2, mlp in self.blocks:
            h = h + mix(ln1(h)); h = h + mlp(ln2(h))
        return self.head(self.lnf(h))


def run(cfg, seed, args, device):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = TinyLM(args.vocab, args.d, args.layers, args.heads, cfg, args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(args.steps):
        x, y, _ = induction_batch(args.batch, args.L, args.vocab, gen, device)
        loss = F.cross_entropy(model(x).reshape(-1, args.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        accs = []
        for _ in range(args.eval_batches):
            x, y, mask = induction_batch(args.batch, args.L, args.vocab, gen, device)
            pred = model(x).argmax(-1)
            accs.append(((pred == y) & mask).sum().item() / mask.sum().item())
    return sum(accs) / len(accs)


# targeted config list: label -> cfg
def build_configs():
    cfgs = []
    for sc in ("dot", "maxplus", "l1", "hint"):                 # SCORE effect (hard agg)
        cfgs.append((f"{sc}_hard", {"mixer": "attn", "score": sc, "agg": "hard"}))
    cfgs.append(("dot_soft_t0.25", {"mixer": "attn", "score": "dot", "agg": "soft", "tau": 0.25}))
    cfgs.append(("maxplus_soft_t0.25", {"mixer": "attn", "score": "maxplus", "agg": "soft", "tau": 0.25}))
    for t in (1.0, 0.25, 0.0625, 0.015625):                     # can ANY fixed tau rescue soft L1?
        cfgs.append((f"l1_soft_t{t:g}", {"mixer": "attn", "score": "l1", "agg": "soft", "tau": t}))
    cfgs.append(("conv", {"mixer": "conv"}))                    # negative control
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"induction GRID: L={args.L} vocab={args.vocab} d={args.d} "
          f"layers={args.layers} heads={args.heads} steps={args.steps} "
          f"seeds={args.seeds} device={device}", flush=True)
    print(f"chance (1/vocab) = {1.0/args.vocab:.3f}\n", flush=True)
    t0 = time.time()
    results = {}

    for label, cfg in build_configs():
        vals = []
        for s in args.seeds:
            a = run(cfg, s, args, device)
            vals.append(a)
            print(f"  {label:18s} seed{s} {a:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        results[label] = vals
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:18s} MEAN {mean:.3f} +/- {sd:.3f}\n", flush=True)

    print("\n=== INDUCTION GRID (mean 2nd-half acc over seeds) ===", flush=True)
    order = sorted(results.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    for label, vals in order:
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:18s} {mean:.3f} +/- {sd:.3f}   {['%.3f'%v for v in vals]}", flush=True)

    def m(lbl):
        return sum(results[lbl]) / len(results[lbl])
    l1_soft_best = max(m(l) for l in results if l.startswith("l1_soft"))
    print("\n--- disentangling reads ---", flush=True)
    print(f"  SCORE effect (HARD agg fixed): dot {m('dot_hard'):.3f} | maxplus {m('maxplus_hard'):.3f} "
          f"| l1 {m('l1_hard'):.3f} | hint {m('hint_hard'):.3f}", flush=True)
    print(f"  AGG effect (L1 score fixed):  l1_soft(best tau) {l1_soft_best:.3f} -> l1_hard {m('l1_hard'):.3f}", flush=True)
    print(f"  tau-confound: l1_soft over tau {{1,.25,.0625,.0156}} = "
          f"{[round(m(l),3) for l in ['l1_soft_t1','l1_soft_t0.25','l1_soft_t0.0625','l1_soft_t0.015625']]} "
          f"(was 0.027 @ tau=0.25 in 6-arm probe)", flush=True)
    print(f"  ceiling check: dot_soft_t0.25 {m('dot_soft_t0.25'):.3f} (probe standard=1.000)", flush=True)


if __name__ == "__main__":
    main()
