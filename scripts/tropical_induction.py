"""tropical_induction.py — does multiplier-free MANY-TO-MANY message passing
approximate self-attention?

The induction/copy probe (the canonical content-routing test): each sequence is a
random run of DISTINCT tokens repeated once, seq = [r ; r]. In the SECOND copy the
correct next token is the successor of the current token's PRIOR occurrence — only
recoverable by content-addressed all-pairs routing ("find where this token appeared
before, copy what followed"). A fixed-receptive-field conv or a bigram table CANNOT
do it (the match offset is data-dependent); a real induction head solves it.

We ABLATE THE SCORE while holding the aggregation fixed, to test the hypothesis that
attention's power is the many-to-many DOT PRODUCT (similarity accumulated over ALL D
dims), and that the tropical attention floor is the max-plus score using only ONE dim:

  standard : softmax(QK^T / sqrt(d)) @ V            real dot product  (ceiling)
  maxplus  : S=max_d(Q+K),  soft agg               one-dim tropical score (baseline)
  l1       : S=-sum_d|Q-K|, soft agg               MULTIPLIER-FREE sum-over-dims score
  hint     : S=sum_d min(Q,K), soft agg            histogram-intersection (mult-free)
  l1_hard  : S=-sum_d|Q-K|, HARD max_j(S+V)         fully multiplier-free (deploy-pure)
  conv     : depthwise causal conv mixer            negative control (no routing)

"soft agg" = softmax(S/tau) @ V (real, to isolate the SCORE's effect). l1_hard swaps
in the pure tropical value pull to show the fully-multiplier-free version.

Metric: next-token accuracy on the SECOND-half (induction-solvable) positions.

Usage: PYTHONPATH=. python scripts/tropical_induction.py
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
    mask[:, half:] = True            # second-copy targets are induction-recoverable
    return x, y, mask


# ----------------------------- mixers ---------------------------------------
def _causal_neg(scores):
    L = scores.shape[-1]
    m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=scores.device), 1)
    return scores.masked_fill(m, float("-inf"))


class Attn(nn.Module):
    """Multi-head attention with a pluggable SCORE and aggregation, causal."""
    def __init__(self, d, heads, variant, tau=0.25):
        super().__init__()
        self.h, self.dh, self.variant, self.tau = heads, d // heads, variant, tau
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)

    def score(self, q, k):  # q,k: (B,H,L,dh) -> (B,H,L,L)
        if self.variant == "standard":
            return torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.dh)
        if self.variant == "maxplus":          # one-dim max-plus inner product
            return (q.unsqueeze(-2) + k.unsqueeze(-3)).max(dim=-1).values
        if self.variant in ("l1", "l1_hard"):  # -sum_d |Q-K|  (sum-over-dims, mult-free)
            return -(q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(dim=-1)
        if self.variant == "hint":             # sum_d min(Q,K) histogram intersection
            return torch.minimum(q.unsqueeze(-2), k.unsqueeze(-3)).sum(dim=-1)
        raise ValueError(self.variant)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        s = _causal_neg(self.score(q, k))
        if self.variant == "l1_hard":          # fully multiplier-free: max_j(S + V)
            out = (s.unsqueeze(-1) + v.unsqueeze(-3)).max(dim=-2).values
        else:                                   # soft agg (isolates the score)
            out = torch.einsum("bhij,bhjd->bhid", F.softmax(s / self.tau, dim=-1), v)
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class Conv(nn.Module):
    """Depthwise causal conv mixer (negative control — fixed receptive field)."""
    def __init__(self, d, ksize=7):
        super().__init__()
        self.dw = nn.Conv1d(d, d, ksize, groups=d, padding=ksize - 1)
        self.pw = nn.Linear(d, d); self.ks = ksize

    def forward(self, x):
        h = self.dw(x.transpose(1, 2))[..., : -(self.ks - 1)].transpose(1, 2)
        return self.pw(F.gelu(h))


class TinyLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, variant, L):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            mix = Conv(d) if variant == "conv" else Attn(d, heads, variant)
            self.blocks.append(nn.ModuleList([nn.LayerNorm(d), mix, nn.LayerNorm(d),
                                              nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                                            nn.Linear(4 * d, d))]))
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for ln1, mix, ln2, mlp in self.blocks:
            h = h + mix(ln1(h)); h = h + mlp(ln2(h))
        return self.head(self.lnf(h))


def run(variant, args, device):
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    model = TinyLM(args.vocab, args.d, args.layers, args.heads, variant, args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    t0 = time.time()
    for step in range(args.steps):
        x, y, _ = induction_batch(args.batch, args.L, args.vocab, gen, device)
        loss = F.cross_entropy(model(x).reshape(-1, args.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    # eval induction-position accuracy
    model.eval()
    with torch.no_grad():
        accs = []
        for _ in range(args.eval_batches):
            x, y, mask = induction_batch(args.batch, args.L, args.vocab, gen, device)
            pred = model(x).argmax(-1)
            accs.append(((pred == y) & mask).sum().item() / mask.sum().item())
    return sum(accs) / len(accs), time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+",
                   default=["standard", "maxplus", "l1", "hint", "l1_hard", "conv"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"induction probe: L={args.L} vocab={args.vocab} d={args.d} "
          f"layers={args.layers} heads={args.heads} steps={args.steps} device={device}")
    print(f"chance (1/vocab) = {1.0/args.vocab:.3f}")
    rows = []
    for v in args.variants:
        acc, secs = run(v, args, device)
        rows.append((v, acc, secs))
        print(f"  {v:10s} induction-acc {acc:.3f}   ({secs:.0f}s)")
    print("\n=== INDUCTION RESULT (2nd-half next-token acc) ===")
    for v, acc, _ in sorted(rows, key=lambda r: -r[1]):
        print(f"  {v:10s} {acc:.3f}")


if __name__ == "__main__":
    main()
