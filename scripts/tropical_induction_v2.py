"""tropical_induction_v2.py — the CORRECTED induction probe.

v1 (scripts/tropical_induction.py) and the grid used seq=[r;r] with r a distinct-
token permutation. An adversarial audit proved that task is DEGENERATE: a token's
two occurrences are ALWAYS exactly +L/2 apart, so it is solvable by a FIXED
positional offset, not content routing. (Controls: shuffling the 2nd half or
zeroing position embeddings collapsed every variant to chance; the trained score
peaked at the matching token 0.000 of the time.) So v1's "l1_hard=1.000" measured
"learned the fixed offset", NOT "tropical attention does content routing".

This is the proper test: a VARIABLE-OFFSET in-context copy task.
  - Each sequence: a length-m block of m DISTINCT random tokens is placed TWICE at
    two RANDOM non-overlapping positions p1<p2; the rest is random filler drawn
    from the complement of the block tokens (so each block token appears ONLY in
    the two copies -> unambiguous prior occurrence).
  - The block content (and the in-context successor chain block[t]->block[t+1]) is
    resampled every sequence, so there is NO fixed token->token map to memorize.
  - Targets/mask: the second copy's positions p2 .. p2+m-2 (predict block[1..m-1]).
    Because p2-p1 varies per sequence, POSITION cannot solve it; only finding the
    first copy by CONTENT and copying its successor works. A small causal conv
    (fixed receptive field) cannot reach the variable offset -> negative control.

Decisive reads (per audit):
  - dot_soft (real softmax attention) is the CEILING and a task-validity check:
    if even standard attention can't solve it, the task is broken.
  - conv must fail (offset is variable, beyond a fixed kernel).
  - no-pos ablation: on a variable-offset task position is useless, so a true
    content-router should be ~unaffected by zeroing position embeddings.
  - selectivity: does the trained score actually PEAK at the matching first-copy
    position? (the mechanistic proof of content routing, not just accuracy.)
  - >=5 seeds, mean+/-std everywhere (v1's single-seed numbers were uninterpretable;
    grid std was 0.11-0.44).

Usage: PYTHONPATH=. python scripts/tropical_induction_v2.py
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def block_copy_batch(B, L, vocab, m, gen, device):
    """Variable-offset in-context copy. Returns seq,(B,L) x,(B,L-1) y,(B,L-1)
    mask,(B,L-1) and match_pos,(B,L-1) = the first-copy position p1+t a query at
    second-copy position p2+t should attend to (for selectivity; -1 elsewhere)."""
    assert 2 * m + 2 <= L and m <= vocab
    seq = torch.empty(B, L, dtype=torch.long, device=device)
    match = torch.full((B, L), -1, dtype=torch.long, device=device)
    mask_full = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        block = torch.randperm(vocab, generator=gen)[:m]            # m distinct block tokens
        block_set = set(block.tolist())
        # filler from the complement of block tokens -> block appears only in copies
        comp = torch.tensor([t for t in range(vocab) if t not in block_set])
        fill = comp[torch.randint(len(comp), (L,), generator=gen)]
        seq[b] = fill.to(device)
        # two random non-overlapping placements p1 < p2
        # pick p1 in [0, L-2m], gap in [0, L-2m-p1], p2 = p1+m+gap
        p1 = int(torch.randint(0, L - 2 * m + 1, (1,), generator=gen).item())
        p2 = int(torch.randint(p1 + m, L - m + 1, (1,), generator=gen).item())
        seq[b, p1:p1 + m] = block.to(device)
        seq[b, p2:p2 + m] = block.to(device)
        # second-copy interior queries: positions p2 .. p2+m-2 predict block[1..m-1]
        for t in range(0, m - 1):
            mask_full[b, p2 + t] = True
            match[b, p2 + t] = p1 + t          # the matching first-copy token position
    x, y = seq[:, :-1], seq[:, 1:]
    return seq, x, y, mask_full[:, :-1], match[:, :-1]


def _causal_neg(s):
    L = s.shape[-1]
    mm = torch.triu(torch.ones(L, L, dtype=torch.bool, device=s.device), 1)
    return s.masked_fill(mm, float("-inf"))


def kbit_quantize(x, mbits):
    """Quantize to a K-bit-MANTISSA dyadic float: sign * m * 2^e, m a (mbits)-bit value
    in [1,2). mbits=0 == pure ±2^k (1-bit, = dyadic_quantize). On fabric, multiplying by
    a K-bit-mantissa value is a sum of <=mbits+1 SHIFTS, so products of two such operands
    are sums of shifts -- 'nearly multiplier-free' (more adds, no multiplier). STE-free
    here (used inside a custom backward; gradients don't flow through the quantizer)."""
    sign = x.sign()
    a = x.abs().clamp_min(1e-30)
    e = torch.floor(torch.log2(a))
    m = a / torch.exp2(e)
    step = 2.0 ** mbits
    return sign * (torch.round(m * step) / step) * torch.exp2(e)


class GatherDyadicGradFn(torch.autograd.Function):
    """Hard argmax-GATHER forward + MAGNITUDE-AWARE, MULTIPLIER-FREE backward (lever d,
    the PC-faithful attempt). Forward = V[argmax_j s] (compare+gather, deploy-identical).
    Backward (all products use ±2^k DYADIC-quantized operands, so every multiply is a
    bit-SHIFT on fabric -- mult-free by the quantized-operand property, like the PC
    set_mf_grads weight-grads, verified by value not op-trace):
      grad_V[j] = scatter grad_out to the gathered key j* (mult-free).
      grad_s[i,j] = <q(grad_out_i), q(V_j) - q(out_i)>  for j in top-k(s_i) (causal),
                    else 0.  This is the value-ALIGNMENT advantage of key j (how much
                    attending there would move the output along -grad), the magnitude
                    signal the uniform top-k scatter discarded. q(.)=dyadic_quantize."""

    @staticmethod
    def forward(ctx, s, v, k, mbits, soft_rank):
        jstar = s.argmax(dim=-1)                                   # (B,H,Lq)
        idx = jstar.unsqueeze(-1).expand(*jstar.shape, v.shape[-1])
        out = torch.gather(v, -2, idx)                            # (B,H,Lq,dh)
        ctx.save_for_backward(s, v, jstar, out)
        ctx.k, ctx.mbits, ctx.soft_rank = k, mbits, soft_rank
        return out

    @staticmethod
    def backward(ctx, grad_out):
        s, v, jstar, out = ctx.saved_tensors
        k = min(ctx.k, s.shape[-1]); mb = ctx.mbits
        dh = v.shape[-1]
        # grad_V: scatter the upstream grad to the gathered key (standard gather bwd, mult-free)
        grad_v = torch.zeros_like(v)
        grad_v.scatter_add_(-2, jstar.unsqueeze(-1).expand(*jstar.shape, dh), grad_out)
        # grad_s: K-bit-mantissa-operand alignment advantage, top-k + causal masked
        qg = kbit_quantize(grad_out, mb)                          # (B,H,Lq,dh)
        qv = kbit_quantize(v, mb)                                 # (B,H,Lv,dh)
        align = torch.einsum("bhid,bhjd->bhij", qg, qv)          # q(g)·q(v): sums of shifts
        baseline = torch.einsum("bhid,bhid->bhi",
                                qg, kbit_quantize(out, mb)).unsqueeze(-1)
        adv = align - baseline
        valid = s > float("-inf")
        topv, topi = s.topk(k, dim=-1)                            # descending score order
        if ctx.soft_rank:
            # DYADIC-RANK SOFT WEIGHT: a multiplier-free non-degenerate "p" -- rank r in
            # the top-k gets weight 2^-r (1, 1/2, 1/4, ...; a shift). Gives the COMPETING
            # keys a non-zero gradient (the soft p that hard-attention's one-hot p kills).
            rw = (2.0 ** (-torch.arange(k, device=s.device, dtype=s.dtype)))  # (k,)
            weight = torch.zeros_like(s).scatter_(-1, topi, rw.expand(*topi.shape))
            grad_s = torch.where(valid, weight * adv, torch.zeros_like(adv))
        else:
            mask = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, topi, True) & valid
            grad_s = torch.where(mask, adv, torch.zeros_like(adv))
        return grad_s, grad_v, None, None, None


def gather_dyadic_grad(s, v, k=4, mbits=0, soft_rank=False):
    return GatherDyadicGradFn.apply(s, v, k, mbits, soft_rank)


class Attn(nn.Module):
    def __init__(self, d, heads, score, agg, tau=0.25, k_bwd=4, grad_mbits=0):
        super().__init__()
        self.h, self.dh, self.sk, self.agg, self.tau = heads, d // heads, score, agg, tau
        self.k_bwd = k_bwd                       # iter-topk density for gather_ste_mf backward
        self.grad_mbits = grad_mbits             # K-bit mantissa for gather_dyadicgrad backward
        self.soft_rank = False                   # dyadic-rank soft weighting in gather_dyadicgrad
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.last_s = None

    def score(self, q, k):
        if self.sk == "dot":
            return torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.dh)
        if self.sk == "maxplus":
            return (q.unsqueeze(-2) + k.unsqueeze(-3)).max(dim=-1).values
        if self.sk == "l1":
            return -(q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(dim=-1)
        raise ValueError(self.sk)

    def forward(self, x, record=False):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        s = _causal_neg(self.score(q, k))
        if record:
            self.last_s = s.detach()
            self.last_s_live = s            # live (grad-carrying) scores for distillation losses
        if self.agg == "hard":                  # max-plus MATMUL: per-dim argmax of S+V
            out = (s.unsqueeze(-1) + v.unsqueeze(-3)).max(dim=-2).values
        elif self.agg in ("gather", "gather_ste", "gather_ste_mf", "gather_dyadicgrad"):  # argmax-GATHER deploy
            jstar = s.argmax(dim=-1)            # (B,H,Lq) single best key per query
            hard = torch.gather(v, -2, jstar.unsqueeze(-1).expand(*jstar.shape, self.dh))
            if self.agg == "gather_ste":        # STE, SOFT surrogate (has a multiply): fwd hard,
                soft = torch.einsum("bhij,bhjd->bhid", F.softmax(s / self.tau, dim=-1), v)
                out = soft + (hard - soft).detach()
            elif self.agg == "gather_ste_mf":   # STE, MULTIPLIER-FREE surrogate (lever d): fwd =
                # hard gather (deploy-identical); backward = closed iter-topk SCATTER (top-k of
                # V+scores, unscaled scatter_add grad -- no softmax/divide/multiply). k_bwd =
                # the gradient-density knob ((b): moderately-dense top-k, not single argmax).
                from tikitorch.functional import tropical_value_pull_iter_topk
                surrogate = tropical_value_pull_iter_topk(v, s, k=self.k_bwd)
                out = surrogate + (hard - surrogate).detach()
            elif self.agg == "gather_dyadicgrad":  # PC-faithful: hard gather fwd + MAGNITUDE-
                # AWARE dyadic-operand backward (q(grad)·q(V) alignment). Mult-free by the
                # quantized-operand property. Tests if magnitude-weighting (which uniform
                # top-k scatter lacked) trained mult-free fixes the router.
                out = gather_dyadic_grad(s, v, k=self.k_bwd, mbits=self.grad_mbits,
                                         soft_rank=self.soft_rank)
            else:
                out = hard
        else:
            out = torch.einsum("bhij,bhjd->bhid", F.softmax(s / self.tau, dim=-1), v)
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class Conv(nn.Module):
    def __init__(self, d, ksize=7):
        super().__init__()
        self.dw = nn.Conv1d(d, d, ksize, groups=d, padding=ksize - 1)
        self.pw = nn.Linear(d, d); self.ks = ksize

    def forward(self, x, record=False):
        h = self.dw(x.transpose(1, 2))[..., : -(self.ks - 1)].transpose(1, 2)
        return self.pw(F.gelu(h))


class TinyLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, cfg, L, use_pos=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(L, d)
        self.use_pos = use_pos
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            if cfg["mixer"] == "conv":
                mix = Conv(d)
            else:
                mix = Attn(d, heads, cfg["score"], cfg["agg"], cfg.get("tau", 0.25),
                           k_bwd=cfg.get("k_bwd", 4), grad_mbits=cfg.get("grad_mbits", 0))
                mix.soft_rank = cfg.get("soft_rank", False)
            self.blocks.append(nn.ModuleList([nn.LayerNorm(d), mix, nn.LayerNorm(d),
                nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))]))
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)

    def forward(self, x, record=False):
        h = self.tok(x)
        if self.use_pos:
            h = h + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for ln1, mix, ln2, mlp in self.blocks:
            h = h + mix(ln1(h), record=record); h = h + mlp(ln2(h))
        return self.head(self.lnf(h))


def run(cfg, seed, args, device, use_pos=True, want_model=False):
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    model = TinyLM(args.vocab, args.d, args.layers, args.heads, cfg, args.L, use_pos).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(args.steps):
        _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, gen, device)
        logits = model(x)
        loss = F.cross_entropy(logits[mask], y[mask])   # only predictable (induction) positions
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    egen = torch.Generator().manual_seed(seed + 9999)        # held-out sequences
    accs = []
    with torch.no_grad():
        for _ in range(args.eval_batches):
            _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, egen, device)
            pred = model(x).argmax(-1)
            accs.append(((pred == y) & mask).sum().item() / mask.sum().item())
    acc = sum(accs) / len(accs)
    return (acc, model) if want_model else (acc, None)


def selectivity(model, args, seed, device):
    """Does the trained score PEAK at the matching first-copy position? Averaged
    over attention layers; the mechanistic proof of content routing."""
    egen = torch.Generator().manual_seed(seed + 5)
    seq, x, y, mask, match = block_copy_batch(args.batch, args.L, args.vocab, args.m, egen, device)
    with torch.no_grad():
        _ = model(x, record=True)
    hits, tot = 0, 0
    near_hits = 0
    for ln1, mix, ln2, mlp in model.blocks:
        if not isinstance(mix, Attn):
            continue
        s = mix.last_s                       # (B,H,Lq,Lk)
        srow = s.mean(1)                     # head-avg (B,Lq,Lk)
        B, Lq, Lk = srow.shape
        for b in range(B):
            qs = mask[b].nonzero(as_tuple=True)[0]
            for i in qs.tolist():
                j = srow[b, i, :i + 1].argmax().item()
                jstar = match[b, i].item()
                tot += 1
                if j == jstar:
                    hits += 1
                if abs(j - jstar) <= 1:       # matching token or its successor (the value)
                    near_hits += 1
    return (hits / tot if tot else 0.0, near_hits / tot if tot else 0.0, tot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--L", type=int, default=48)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--m", type=int, default=8)           # block length
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"INDUCTION v2 (variable-offset copy): L={args.L} vocab={args.vocab} m={args.m} "
          f"d={args.d} layers={args.layers} heads={args.heads} steps={args.steps} "
          f"seeds={args.seeds} dev={device}", flush=True)
    print(f"chance (1/vocab) = {1.0/args.vocab:.3f}\n", flush=True)
    t0 = time.time()

    cells = [
        ("dot_soft",     {"mixer": "attn", "score": "dot", "agg": "soft", "tau": 0.25}),   # ceiling + validity
        ("dot_hard",     {"mixer": "attn", "score": "dot", "agg": "hard"}),                # mandatory baseline
        ("l1_hard",      {"mixer": "attn", "score": "l1", "agg": "hard"}),                 # the candidate
        ("maxplus_hard", {"mixer": "attn", "score": "maxplus", "agg": "hard"}),
        ("l1_soft_t1",   {"mixer": "attn", "score": "l1", "agg": "soft", "tau": 1.0}),
        ("conv",         {"mixer": "conv"}),                                               # neg control
    ]
    results, sel = {}, {}
    for label, cfg in cells:
        vals = []
        for s in args.seeds:
            want = (s == args.seeds[0] and cfg["mixer"] == "attn")
            acc, model = run(cfg, s, args, device, want_model=want)
            vals.append(acc)
            tag = ""
            if want:
                f_exact, f_near, n = selectivity(model, args, s, device)
                sel[label] = (f_exact, f_near, n)
                tag = f"  [sel exact {f_exact:.3f} near±1 {f_near:.3f} n={n}]"
            print(f"  {label:14s} seed{s} {acc:.3f}{tag}  ({time.time()-t0:.0f}s)", flush=True)
        results[label] = vals
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:14s} MEAN {mean:.3f} +/- {sd:.3f}\n", flush=True)

    # no-pos controls (variable offset => position useless => true router unaffected)
    print("--- no-position controls ---", flush=True)
    for label, cfg in [("dot_soft", cells[0][1]), ("l1_hard", cells[2][1])]:
        vals = []
        for s in args.seeds:
            acc, _ = run(cfg, s, args, device, use_pos=False)
            vals.append(acc)
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        results[f"{label}_nopos"] = vals
        print(f"  {label:14s} NO-POS MEAN {mean:.3f} +/- {sd:.3f}", flush=True)

    print("\n=== INDUCTION v2 RESULT (variable-offset, mean over seeds) ===", flush=True)
    for label, vals in sorted(results.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        extra = ""
        if label in sel:
            extra = f"   sel_exact={sel[label][0]:.3f} sel_near={sel[label][1]:.3f}"
        print(f"  {label:16s} {mean:.3f} +/- {sd:.3f}{extra}", flush=True)
    print("\nPASS condition: a tropical variant must (a) beat conv, (b) survive no-pos, "
          "(c) show sel_exact/near >> 0 -> genuine content routing.", flush=True)


if __name__ == "__main__":
    main()
