"""charlm_mixer.py — char-LM head-to-head that isolates the ATTENTION MIXER.

Question (post-induction): the prior bakeoff found "tropical attention floors at
bigram (~2.5 bpb)". The induction probe showed that was an OPERATOR-MISMATCH:
soft-trained attention deployed with the per-dim max-plus MATMUL collapses, while
argmax-GATHER (the true hard limit of softmax) is lossless. Does fixing the deploy
op let a multiplier-free attention MIXER reach dense-attention quality on a real
char-LM (tiny-shakespeare)?

This uses a STANDARD transformer (nn.Linear Q/K/V/O + nn.Linear FFN + LayerNorm)
and swaps ONLY the attention MIXER (score + aggregation) -- exactly the induction
setup. So "multiplier-free" is scoped to the mixer (the honest scope; the full
tropical projections/FFN are a separate, slower question). This also runs ~100x
faster than the full TropicalTransformer (whose max-plus linears materialize
(B,L,out,in)), so we can actually train to convergence.

Mixer arms (causal):
  dense       score=dot,  agg=soft               real softmax attention (nanoGPT mixer) = ceiling
  l1_gather   score=l1,   train soft -> deploy GATHER   fully mult-free mixer (the candidate)
  l1_matmul   score=l1,   train soft -> deploy max-plus MATMUL   the collapse control (old op)
  dot_gather  score=dot,  train soft -> deploy GATHER   does gather cost vs soft dot?

Train soft with tau annealed (north-star); eval BOTH soft (final tau) and the hard
deploy op. Metric: val bits-per-byte (char => bpb = nats/ln2). Multiple seeds.

Usage: PYTHONPATH=. python scripts/charlm_mixer.py
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import induction_selfsup                          # for _SHIFT_WBITS global (finer blend mantissa)
from induction_selfsup import _hinge_grad, ShiftSoftmaxBlend, MarginBandedRankBlend, ResidualTopKBlend, delta_at   # hinge + mult-free differentiable blends
from tikitorch.nn import dyadic_quantize       # mult-free magnitude optimizer step

REPO = Path(__file__).resolve().parents[1]

_MLP_ACT = "gelu"          # gelu (multiplies) | relu (mult-free, for the deploy-pure stack)


def _act():
    return nn.ReLU() if _MLP_ACT == "relu" else nn.GELU()

# Linear factory: nn.Linear (standard) or DyadicLinear (±2^k shift-add, mult-free
# deploy). Set by --linear; lets us isolate the dyadic-linear cost with everything
# else (nn.LayerNorm, structure, mixer) held identical.
_LINEAR = "standard"


def Lin(i, o, bias=True):
    if _LINEAR == "dyadic":
        from tikitorch.nn import DyadicLinear
        return DyadicLinear(i, o, bias=bias)
    return nn.Linear(i, o, bias=bias)


# Norm factory: nn.LayerNorm (multiplies) vs the multiplier-free tropical norms.
# Lets us isolate whether the full-tropical gap is the NORM (TropicalLayerNorm) vs
# the linears/mixer.
_NORM = "layernorm"


class _VarNoGamma(nn.Module):
    """True-variance LayerNorm with the gamma SCALE dropped (additive beta only).
    Isolates the cost of dropping the learnable per-feature scale (vs nn.LayerNorm)
    from the cost of the pow2/mean-abs rescale (vs TropicalLayerNorm)."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps; self.beta = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        c = x - x.mean(-1, keepdim=True)
        return c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + self.eps) + self.beta


_NORM_MBITS = 2          # mantissa bits for fine_dyadic (0 = pure pow2; higher = finer)
_NORM_SCALE = "meanabs"  # scale estimator: meanabs/maxabs/proxy (mult-free) | rms (has a square!)


def _norm_scale(c, eps):
    """Per-row scale used as the normalizer. MULTIPLIER-FREE options:
      meanabs = mean|c|        (abs + add)            -- the TropicalLayerNorm scale
      maxabs  = max|c|         (abs + max)
      proxy   = alpha-max-beta-min L2 estimate: 0.5*max|c| + mean|c| (shifts + add) --
                a closer mult-free estimate of RMS than mean-abs alone.
    rms = sqrt(mean c^2) is the IDEAL but its square is a MULTIPLY (reference only)."""
    a = c.abs()
    if _NORM_SCALE == "rms":
        return (c.pow(2).mean(-1, keepdim=True) + eps).sqrt()      # HAS A MULTIPLY (ref)
    if _NORM_SCALE == "maxabs":
        return a.max(-1, keepdim=True).values + eps
    if _NORM_SCALE == "proxy":                                     # mult-free L2 proxy
        return 0.5 * a.max(-1, keepdim=True).values + a.mean(-1, keepdim=True) + eps
    return a.mean(-1, keepdim=True) + eps                          # meanabs (default)


def _fine_dyadic_rescale(c, mbits, eps):
    """Multiplier-free rescale by 1/scale, with 1/scale quantized to a (mbits)-bit
    MANTISSA dyadic value m*2^e (m in [1,2)). On hardware m*2^e * c is a sum of
    <=mbits+1 shifts + adds (e.g. 1.5x = x + x>>1) -- zero-multiply, but far finer
    than the single-power-of-2 rescale (which rounds the whole scale to the nearest
    2^k, up to +-41% error). STE: forward quantized, backward smooth."""
    inv = 1.0 / _norm_scale(c, eps)                            # 1/scale (smooth target)
    e = torch.floor(torch.log2(inv.clamp_min(1e-12)))
    m = inv / torch.exp2(e)                                    # mantissa in [1,2)
    step = 2.0 ** mbits
    mq = torch.round(m * step) / step                         # mbits-bit mantissa
    invq = mq * torch.exp2(e)
    invq = inv + (invq - inv).detach()                        # straight-through
    return c * invq


class _FineDyadicNorm(nn.Module):
    """subtract-mean + (mbits)-bit-mantissa dyadic 1/RMS rescale + additive beta.
    mbits=0 == the single-power-of-2 TropicalLayerNorm rescale; mbits->inf -> true
    LayerNorm-without-gamma. Multiplier-free (shifts + adds) at any mbits."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps; self.beta = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        c = x - x.mean(-1, keepdim=True)
        return _fine_dyadic_rescale(c, _NORM_MBITS, self.eps) + self.beta


class _DyadicGammaNorm(nn.Module):
    """TropicalLayerNorm (subtract-mean + pow2 mean-abs rescale) PLUS a learnable
    DYADIC gamma: per-feature scale constrained to +-2^k (a bit-shift, STE). Restores
    the affine scale that TropicalLayerNorm drops, while staying multiplier-free at
    deploy. Tests whether the dropped gamma is the norm's quality cost."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d)); self.beta = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        from tikitorch.nn import _pow2_normalize, dyadic_quantize
        c = x - x.mean(-1, keepdim=True)
        y = _pow2_normalize(c, dim=-1, eps=self.eps)
        return y * dyadic_quantize(self.gamma) + self.beta


def Norm(d):
    if _NORM == "tropical":
        from tikitorch.nn import TropicalLayerNorm
        return TropicalLayerNorm(d)
    if _NORM == "rms":
        from tikitorch.nn import TropicalRMSNorm
        return TropicalRMSNorm(d)
    if _NORM == "var_nogamma":
        return _VarNoGamma(d)
    if _NORM == "dyadic_gamma":
        return _DyadicGammaNorm(d)
    if _NORM == "fine_dyadic":
        return _FineDyadicNorm(d)
    return nn.LayerNorm(d)


# ----------------------------- model ----------------------------------------
def _causal_neg(s):
    L = s.shape[-1]
    m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=s.device), 1)
    return s.masked_fill(m, float("-inf"))


class Attn(nn.Module):
    """Standard nn.Linear projections; pluggable tropical SCORE x AGG mixer."""
    def __init__(self, d, heads, score, agg, tau=0.5, dropout=0.0):
        super().__init__()
        self.h, self.dh, self.sk, self.agg, self.tau = heads, d // heads, score, agg, tau
        self.q = Lin(d, d, bias=False); self.k = Lin(d, d, bias=False)
        self.v = Lin(d, d, bias=False); self.o = Lin(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.last_s = None         # live causal scores (B,H,L,L), exposed for the teacher-free hinge
        self.Delta = 1.0           # shift-softmax temperature (agg="shift"; annealed)

    def score(self, q, k):
        if self.sk == "dot":
            return torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.dh)
        if self.sk == "l1":                       # -sum_d|Q-K| (MPS-native; cdist bwd unimpl on MPS)
            return -(q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(-1)
        raise ValueError(self.sk)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
        s = _causal_neg(self.score(q, k))
        if self.agg == "gather":                  # deploy-pure: copy one key's value vec
            self.last_s = s                        # keep graph -> the hinge backprops to Q/K
            jstar = s.argmax(dim=-1)
            out = torch.gather(v, -2, jstar.unsqueeze(-1).expand(*jstar.shape, self.dh))
        elif self.agg == "matmul":                # the WRONG op: per-dim max-plus matmul
            out = (s.unsqueeze(-1) + v.unsqueeze(-3)).max(dim=-2).values
        elif self.agg == "shift":                 # mult-free DIFFERENTIABLE blend (train relaxation)
            self.last_s = s                        # expose scores for the pshift peaking penalty
            out = ShiftSoftmaxBlend.apply(s, v, self.Delta, 12)
        elif self.agg == "mbr":                   # IDEA 5: margin-banded rank blend (mult-free, smooth+peaked)
            out = MarginBandedRankBlend.apply(s, v, self.Delta, induction_selfsup._MBR_K)
        elif self.agg == "rtk":                   # IDEA 2: residual top-k blend (finer top weight, lossless)
            out = ResidualTopKBlend.apply(s, v, self.Delta, induction_selfsup._RTK_K, 12)
        else:                                     # soft (train relaxation): softmax@V
            out = torch.einsum("bhij,bhjd->bhid", F.softmax(s / self.tau, dim=-1), v)
        return self.o(self.drop(out.transpose(1, 2).reshape(B, L, D)))


class Block(nn.Module):
    def __init__(self, d, heads, d_ff, score, agg, dropout):
        super().__init__()
        self.ln1 = Norm(d); self.attn = Attn(d, heads, score, agg, dropout=dropout)
        self.ln2 = Norm(d)
        self.mlp = nn.Sequential(Lin(d, d_ff), _act(), Lin(d_ff, d),
                                 nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x)); return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d, heads, layers, d_ff, L, score, agg, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(L, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, heads, d_ff, score, agg, dropout)
                                     for _ in range(layers)])
        self.lnf = Norm(d); self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.drop(self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None])
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))


# ----------------------------- data -----------------------------------------
def load_chars(path, device):
    text = Path(path).read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n].to(device), ids[n:].to(device), len(chars)


def batch(ids, B, L, gen, device):
    s = torch.randint(0, ids.numel() - L - 1, (B,), generator=gen)
    x = torch.stack([ids[i:i + L] for i in s])
    y = torch.stack([ids[i + 1:i + L + 1] for i in s])
    return x.to(device), y.to(device)


# ----------------------------- train / eval ---------------------------------
def set_tau(model, tau):
    for b in model.blocks:
        b.attn.tau = tau


def set_delta(model, d):
    for b in model.blocks:
        b.attn.Delta = d


def set_agg(model, agg):
    saved = [b.attn.agg for b in model.blocks]
    for b in model.blocks:
        b.attn.agg = agg
    return saved


@torch.no_grad()
def eval_bpb(model, val, args, device, agg, tau):
    saved = set_agg(model, agg); set_tau(model, max(tau, 1e-4))  # soft needs tau>0; gather/matmul ignore tau
    if agg in ("shift", "mbr", "rtk"):
        set_delta(model, tau)                          # shift/mbr deploy: reuse the tau slot as Delta (the band/floor)
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    ces = []
    for _ in range(args.eval_batches):
        x, y = batch(val, args.batch, args.seq, gen, device)
        ce = F.cross_entropy(model(x).reshape(-1, model.head.out_features), y.reshape(-1))
        ces.append(ce.item())
    for b, a in zip(model.blocks, saved):
        b.attn.agg = a
    ce = sum(ces) / len(ces)
    return ce, ce / math.log(2)        # nats, bpb (char => 1 tok/byte)


def tau_at(step, steps, tau0, floor):
    # hold tau0 to 50%, cosine-ish linear decay to floor by 90%.
    if step < 0.5 * steps:
        return tau0
    if step > 0.9 * steps:
        return floor
    frac = (step - 0.5 * steps) / (0.4 * steps)
    return floor + (tau0 - floor) * (1 - frac)


def peak_margin_grad(s, margin):
    """pshift PEAKING penalty (mult-free, the _hinge_grad pattern): where the top-1/top-2 score gap is
    below `margin`, push s[top1] UP and s[top2] DOWN ({-1,0,+1} scatter). Backprop through s=-Σ|Q-K|
    gives sign(Q-K) -> drives the LEARNED attention to be genuinely one-hot-able, so deploy=argmax-gather
    is lossless WITHOUT the blend itself having to collapse (decouples smoothness from peakedness)."""
    topv, topi = s.topk(2, dim=-1)                    # (B,H,L,2)
    active = ((topv[..., 0] - topv[..., 1]) < margin) & torch.isfinite(topv[..., 1])
    a = active.to(s.dtype)
    g = torch.zeros_like(s)
    g.scatter_(-1, topi[..., 0:1], (-a).unsqueeze(-1))   # descent raises s[top1]
    g.scatter_add_(-1, topi[..., 1:2], (a).unsqueeze(-1))  # ... and lowers s[top2]
    return g


def train_arm(name, score, train_agg, anneal, deploy_aggs, seed, args, train, val, vocab, device):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = GPT(vocab, args.d, args.heads, args.layers, args.d_ff, args.seq,
                score, train_agg, args.dropout).to(device)
    use_dadam = (args.soft_opt == "dadam")            # fully mult-free optimizer (else AdamW)
    is_pshift = (name == "pshift")                     # peaking-regularized shift blend
    opt = (torch.optim.SGD(model.parameters(), lr=args.lr) if use_dadam
           else torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1))
    m_buf, v_buf = {}, {}
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * (min(1.0, (step + 1) / args.warmup)
                        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))))
        for g in opt.param_groups:
            g["lr"] = lr
        # dense = fixed-tau softmax (the nanoGPT mixer); tropical arms anneal
        # tau -> tau_floor (small +ve so the soft train path stays valid; the GATHER
        # deploy is tau-free and is the true hard limit).
        set_tau(model, tau_at(step, args.steps, args.tau, args.tau_floor) if anneal else args.tau)
        if train_agg in ("shift", "mbr", "rtk"):              # anneal the mult-free blend temperature/band
            set_delta(model, delta_at(step, args))
        model.train()
        x, y = batch(train, args.batch, args.seq, gen, device)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        if is_pshift:                                  # PEAKING penalty as a scalar relu-hinge on the
            plam = args.peak_lambda * min(1.0, step / (0.5 * args.steps))  # top1-top2 gap (annealed in
            pk = 0.0                                    # over first 50%). Its subgradient is the {-1,0,+1}
            for b in model.blocks:                      # peak_margin_grad -> sign(Q-K), single backward
                topv = b.attn.last_s.topk(2, dim=-1).values        # (B,H,L,2), grad-carrying
                gap = topv[..., 0] - topv[..., 1]
                valid = torch.isfinite(topv[..., 1])
                pk = pk + torch.where(valid, F.relu(args.peak_margin - gap),
                                      torch.zeros_like(gap)).mean()
            loss = loss + plam * pk
        opt.zero_grad(); loss.backward()
        if use_dadam:                                  # dyadic-Adam: mult-free per-param adaptive step
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad
                    mm = m_buf.get(p, torch.zeros_like(g)); vv = v_buf.get(p, torch.zeros_like(g))
                    mm = (mm - mm * 0.125) + g * 0.125; vv = (vv - vv * 0.125) + g.abs() * 0.125
                    m_buf[p], v_buf[p] = mm, vv
                    p.grad = mm * torch.exp2(-torch.round(torch.log2(vv.clamp_min(1e-12))))
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    res = {}
    for dname, dagg, dtau in deploy_aggs:
        ce, bpb = eval_bpb(model, val, args, device, dagg, dtau)
        res[dname] = bpb
    secs = time.time() - t0
    summary = "  ".join(f"{k} {v:.3f}" for k, v in res.items())
    print(f"  {name:11s} seed{seed}  {summary}  ({secs:.0f}s)", flush=True)
    return res


# =================== TEACHER-FREE MULT-FREE ROUTER (ff2) ====================
# Port of the induction-probe breakthrough (scripts/induction_selfsup.py `ff_succ`,
# 1.000 teacher-free) to the char-LM. Train Q/K with a per-head STRUCTURAL TARGET via
# the proven L1-margin hinge (-> sign(Q-K), mult-free); V/O/MLP/head/embeds learn from
# the LM loss through the gather scatter; optimizer = dyadic-magnitude. NO teacher, NO
# train-time softmax/exp. Targets are all from input x (equality compares + index only).
# Empirical basis (workflow measurement, tinyshakespeare): single-char induction is a
# SUB-UNIGRAM target (copyacc 0.149 < unigram 0.153) -> FORBIDDEN; the content-bearing
# targets are skip-2/skip-3 CONTEXT-induction (copyacc ~0.30 / ~0.46).


def _most_recent(ok):
    """ok:(B,Lq,Lk) bool, already causal-masked. Return (jpos,valid): jpos = the LARGEST
    legal key index (MOST-RECENT match; recency-biased like a real induction head), valid =
    any legal key. Pure index arithmetic + masked-max -> no multiply."""
    B, Lq, Lk = ok.shape
    cand = torch.arange(Lk, device=ok.device)[None, None, :].expand(B, Lq, Lk).clone()
    cand = cand.masked_fill(~ok, -1)
    has = ok.any(-1)
    tgt = cand.argmax(-1)                                   # largest legal index (cand[j]=j)
    return torch.where(has, tgt, torch.full_like(tgt, -1)), has


def tgt_prev(x):                                           # previous-token (i-1); the induction feeder
    B, L = x.shape; ar = torch.arange(L, device=x.device)
    return (ar - 1).clamp_min(0)[None, :].expand(B, L), (ar >= 1)[None, :].expand(B, L)


def tgt_local(x, w):                                       # fixed local n-gram offset (i-w)
    B, L = x.shape; ar = torch.arange(L, device=x.device)
    return (ar - w).clamp_min(0)[None, :].expand(B, L), (ar >= w)[None, :].expand(B, L)


def tgt_skip(x, kctx):
    """CONTEXT-induction: key j right after the most-recent earlier kctx-context that matches
    the query's current kctx-context (x[i..i-kctx+1] == x[j-1..j-kctx]); V[j]=proj(x[j]) is the
    predicted next char. kctx=1 is the sub-unigram single-char induction (NOT used); kctx>=2 is
    content-bearing. Equality compares + roll + most-recent -> mult-free."""
    B, L = x.shape; ar = torch.arange(L, device=x.device)
    ok = (ar[None, :, None] >= ar[None, None, :]) & (ar[None, None, :] >= kctx)   # causal & j>=kctx
    for d in range(1, kctx + 1):                           # x[i-(d-1)] == x[j-d]
        ok = ok & (torch.roll(x, d - 1, 1).unsqueeze(2) == torch.roll(x, d, 1).unsqueeze(1))
    return _most_recent(ok)


def tgt_dup(x):                                            # most-recent earlier duplicate of x[i]
    B, L = x.shape; ar = torch.arange(L, device=x.device)
    ok = (ar[None, :, None] > ar[None, None, :]) & (x.unsqueeze(2) == x.unsqueeze(1))
    return _most_recent(ok)


# head -> (kind, arg) per layer; 4 layers x 8 heads. By depth (prev-token feeds induction),
# spread within layer for >=6 distinct measured-useful targets. ZERO heads on skip-1.
HEAD_MAP = [
    [("prev", 0), ("prev", 0), ("prev", 0), ("local", 2), ("local", 3), ("local", 4), ("local", 5), ("dup", 0)],
    [("skip", 2), ("skip", 2), ("skip", 3), ("dup", 0), ("prev", 0), ("prev", 0), ("local", 2), ("local", 3)],
    [("skip", 2), ("skip", 2), ("skip", 3), ("skip", 3), ("skip", 4), ("dup", 0), ("prev", 0), ("local", 2)],
    [("skip", 2), ("skip", 2), ("skip", 3), ("skip", 3), ("skip", 4), ("skip", 4), ("dup", 0), ("prev", 0)],
]


def _target_for(kind, arg, x, cache):
    key = (kind, arg)
    if key not in cache:
        if kind == "prev":    cache[key] = tgt_prev(x)
        elif kind == "local": cache[key] = tgt_local(x, arg)
        elif kind == "skip":  cache[key] = tgt_skip(x, arg)
        elif kind == "dup":   cache[key] = tgt_dup(x)
        else: raise ValueError(kind)
    return cache[key]


def charlm_router_grads(model, x, margin, shuffle, phaseA):
    """Per-layer grad-on-S list (feeds torch.autograd.backward(S_live, gss)). Each head gets its
    assigned structural target's L1-margin hinge via _hinge_grad. phaseA: curriculum -> ALL heads
    use prev-token (install predecessors first). shuffle: random causal targets (content control)."""
    B, L = x.shape
    ar = torch.arange(L, device=x.device)
    causal = (ar[None, :, None] >= ar[None, None, :])
    cache = {}
    out = []
    for li, blk in enumerate(model.blocks):
        s = blk.attn.last_s.detach()                        # (B,H,L,L)
        g = torch.zeros_like(s)
        for h in range(s.shape[1]):
            kind, arg = ("prev", 0) if phaseA else HEAD_MAP[li][h]
            jpos, valid = _target_for(kind, arg, x, cache)
            if shuffle:
                jpos = torch.rand(B, L, L, device=x.device).masked_fill(~causal, -1.0).argmax(-1)
                valid = torch.ones(B, L, dtype=torch.bool, device=x.device)
            g[:, h:h + 1] = _hinge_grad(s[:, h:h + 1], jpos, valid, 1, margin)
        out.append(g)
    return out


def charlm_gradtok_circuit_grads(model, x, glogits, layers, heads, margin, shuffle):
    """ATTEMPT 6 on char-LM: induction CIRCUIT with a CONTENT-ADAPTIVE grad-token reader.
    LAYER 0 = previous-token feeder (universal structural primitive: makes key j encode x[j-1]).
    LAYERS 1+ = GRAD-TOKEN reader: target = argmax over causal j of (-dL/dlogits_i)[token x_j] -- route
    to the key whose TOKEN the LM loss most wants emitted at i. On NATURAL TEXT this is genuinely
    content-adaptive (the wanted token varies with context, unlike the REFUTED fixed structural library);
    teacher-free (only the model's own logit gradient); mult-free (gather + argmax + sign hinge)."""
    B, L = x.shape
    ar = torch.arange(L, device=x.device)
    causal = (ar[None, :, None] >= ar[None, None, :])
    gl = -glogits                                              # (B,L,V) >0 = token the loss wants
    keytok = torch.gather(gl, 2, x[:, None, :].expand(B, L, L))   # (B,Li,Lj) = gl[b,i, x[b,j]]
    if shuffle:                                               # CONTROL: random causal reader target
        jtok = torch.rand(B, L, L, device=x.device).masked_fill(~causal, -1.0).argmax(-1)
    else:
        jtok = keytok.masked_fill(~causal, torch.finfo(gl.dtype).min).argmax(-1)   # (B,L) reader target
    jprev = (ar - 1).clamp_min(0)[None, :].expand(B, L)       # prev-token feeder
    valid_prev = (ar >= 1)[None, :].expand(B, L)
    valid_tok = (ar >= 1)[None, :].expand(B, L)
    out = []
    for li, blk in enumerate(model.blocks):
        s = blk.attn.last_s.detach()
        jpos, valid = (jprev, valid_prev) if li == 0 else (jtok, valid_tok)
        g = torch.zeros_like(s)
        for h in range(s.shape[1]):
            g[:, h:h + 1] = _hinge_grad(s[:, h:h + 1], jpos, valid, 1, margin)
        out.append(g)
    return out


def train_ff_arm(name, seed, args, train, val, vocab, device):
    """Teacher-free router training: hard-gather forward + per-head structural hinge. Optimizer set by
    --ff-opt: 'dyadic' = dyadic-magnitude (fully mult-free TRAIN) | 'adamw' = AdamW (NOT mult-free;
    DIAGNOSTIC to isolate router quality from the mult-free-optimizer tax). name 'lm_only_hard' = no
    hinge (Q/K frozen floor); '*_shuffle' = random-target content control."""
    is_shuffle = name.endswith("_shuffle")
    is_floor = name.startswith("lm_only_hard")
    is_gradtok = name.startswith("gradtok")                   # content-adaptive grad-token reader
    if name.startswith("ff2"):
        assert args.layers == 4 and args.heads == 8, "HEAD_MAP is fixed for 4 layers x 8 heads"
    use_adamw = (args.ff_opt == "adamw") or name.endswith("_adamw")   # per-arm opt override
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = GPT(vocab, args.d, args.heads, args.layers, args.d_ff, args.seq,
                "l1", "gather", args.dropout).to(device)
    opt = (torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
           if use_adamw else torch.optim.SGD(model.parameters(), lr=args.sign_lr))
    t0 = time.time()
    for step in range(args.steps):
        if use_adamw:
            lr = args.lr * (min(1.0, (step + 1) / args.warmup)
                            * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))))
            for g in opt.param_groups:
                g["lr"] = lr
        model.train()
        x, y = batch(train, args.batch, args.seq, gen, device)
        logits = model(x)
        if is_gradtok and not is_floor:
            logits.retain_grad()                            # expose logit grad for the grad-token reader
        lm = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))   # -> V/O/MLP/head/embed
        opt.zero_grad()
        lm.backward(retain_graph=not is_floor)
        if not is_floor:
            S_live = [b.attn.last_s for b in model.blocks]
            if is_gradtok:
                gss = charlm_gradtok_circuit_grads(model, x, logits.grad, args.layers,
                                                   args.heads, args.margin, is_shuffle)
            else:
                phaseA = step < args.curric_frac * args.steps
                gss = charlm_router_grads(model, x, args.margin, is_shuffle, phaseA)
            torch.autograd.backward(S_live, gss)            # hinge -> Q/K (sign(Q-K), mult-free)
        if use_adamw:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        else:
            for p in model.parameters():                    # dyadic-magnitude mult-free step
                if p.grad is not None:
                    g = dyadic_quantize(p.grad)
                    p.grad = g.sign() * g.abs().clamp(2.0 ** -12, 2.0 ** -1)
        opt.step()
    ce, bpb = eval_bpb(model, val, args, device, "gather", 0.0)
    print(f"  {name:13s} seed{seed}  gather {bpb:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"       routing: {router_diag(model, val, args, device)}", flush=True)
    return {"gather": bpb}


@torch.no_grad()
def target_diag(train, args, device):
    """Pre-train: per-target copyacc (x[tgt]==y over valid) + coverage. Reproduces the workflow's
    measured numbers (skip2 ~0.30, skip3 ~0.46, prev ~0.02) and predicts any underfit."""
    gen = torch.Generator().manual_seed(99)
    x, y = batch(train, args.batch, args.seq, gen, device)
    cache = {}
    parts = []
    for kind, arg, nm in [("prev", 0, "prev"), ("local", 2, "local2"), ("skip", 1, "skip1"),
                          ("skip", 2, "skip2"), ("skip", 3, "skip3"), ("skip", 4, "skip4"), ("dup", 0, "dup")]:
        jpos, valid = _target_for(kind, arg, x, cache)
        xt = torch.gather(x, 1, jpos.clamp_min(0))
        acc = ((xt == y) & valid).sum().item() / max(1, valid.sum().item())
        parts.append(f"{nm} acc{acc:.3f}/cov{valid.float().mean().item():.2f}")
    return "  ".join(parts)


@torch.no_grad()
def router_diag(model, val, args, device):
    """Post-train: per-layer prev_sel (argmax==i-1) + per-head assigned-target selectivity."""
    saved = set_agg(model, "gather"); model.eval()
    gen = torch.Generator().manual_seed(4242)
    x, _ = batch(val, args.batch, args.seq, gen, device)
    model(x)
    B, L = x.shape; ar = torch.arange(L, device=device)
    cache = {}
    lines = []
    for li, blk in enumerate(model.blocks):
        am = blk.attn.last_s.argmax(-1)                     # (B,H,L)
        prev_sel = ((am == (ar - 1).clamp_min(0)[None, None, :]) & (ar >= 1)[None, None, :]).float().mean().item()
        hsel = []
        for h in range(am.shape[1]):
            kind, arg = HEAD_MAP[li][h]
            jpos, valid = _target_for(kind, arg, x, cache)
            hsel.append(((am[:, h] == jpos) & valid).sum().item() / max(1, valid.sum().item()))
        lines.append(f"L{li} prev{prev_sel:.2f} tgt[" + "/".join(f"{v:.2f}" for v in hsel) + "]")
    for b, a in zip(model.blocks, saved):
        b.attn.agg = a
    return " | ".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=Path, default=REPO / "data" / "tinyshakespeare.txt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--tau-floor", type=float, default=0.05)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--linear", choices=["standard", "dyadic"], default="standard",
                   help="proj+MLP linear type: nn.Linear or DyadicLinear (±2^k, mult-free)")
    p.add_argument("--norm", choices=["layernorm", "tropical", "rms", "var_nogamma",
                                      "dyadic_gamma", "fine_dyadic"], default="layernorm",
                   help="normalization: nn.LayerNorm | mult-free tropical/rms | var_nogamma "
                        "(isolate gamma) | dyadic_gamma (pow2+±2^k scale) | fine_dyadic "
                        "(mbits-mantissa dyadic 1/RMS rescale, --norm-mbits)")
    p.add_argument("--norm-mbits", type=int, default=2,
                   help="mantissa bits for --norm fine_dyadic (0=pow2; higher=finer, mult-free)")
    p.add_argument("--norm-scale", choices=["meanabs", "maxabs", "proxy", "rms"], default="meanabs",
                   help="fine_dyadic scale estimator (rms has a multiply; others mult-free)")
    p.add_argument("--mlp-act", choices=["gelu", "relu"], default="gelu",
                   help="MLP activation; relu for the mult-free deploy stack (gelu has multiplies)")
    p.add_argument("--sign-lr", type=float, default=2e-3, help="dyadic-magnitude optimizer lr (ff arms)")
    p.add_argument("--margin", type=float, default=8.0, help="L1-margin hinge margin (ff arms)")
    p.add_argument("--curric-frac", type=float, default=0.4,
                   help="fraction of steps in curriculum phase A (all heads = prev-token)")
    p.add_argument("--delta0", type=float, default=8.0)       # shift-blend temperature anneal start
    p.add_argument("--delta-floor", type=float, default=0.25)  # ... and floor
    p.add_argument("--shift-wbits", type=int, default=0)      # shift-blend weight mantissa bits (0=pow2; 3=finer)
    p.add_argument("--soft-opt", choices=["adamw", "dadam"], default="adamw",  # dadam = fully mult-free optimizer
                   help="optimizer for the soft/shift/mbr/pshift arms: AdamW (isolates blend) | dyadic-Adam (mult-free)")
    p.add_argument("--peak-margin", type=float, default=8.0)   # pshift: peaking penalty band (L1-dist units)
    p.add_argument("--peak-lambda", type=float, default=2.0)   # pshift: peaking penalty weight (pow2 -> shift)
    p.add_argument("--mbr-k", type=int, default=4)            # margin-banded rank blend: top-k support (k shifts+adds)
    p.add_argument("--rtk-k", type=int, default=2)            # residual top-k blend: top-k support (2 = convex)
    p.add_argument("--mbr-deploy-band", type=float, default=0.5)  # top-k dyadic DEPLOY band (>0 = top-k blend; 0 = exact gather)
    p.add_argument("--ff-opt", choices=["dyadic", "adamw"], default="dyadic",
                   help="ff-arm optimizer: dyadic-magnitude (fully mult-free) | adamw (diagnostic, "
                        "isolates router quality from the mult-free-optimizer tax). Per-arm '_adamw' suffix overrides.")
    p.add_argument("--arms", type=str, default=None,
                   help="comma-list to run only some arms (e.g. dense,l1_gather,ff2) to save time")
    args = p.parse_args()
    global _LINEAR, _NORM, _NORM_MBITS, _NORM_SCALE, _MLP_ACT
    _LINEAR = args.linear
    _NORM = args.norm
    _NORM_MBITS = args.norm_mbits
    _NORM_SCALE = args.norm_scale
    _MLP_ACT = args.mlp_act
    induction_selfsup._SHIFT_WBITS = args.shift_wbits        # finer mult-free blend weights
    induction_selfsup._MBR_K = args.mbr_k                     # margin-banded rank blend top-k
    induction_selfsup._RTK_K = args.rtk_k                     # residual top-k blend top-k
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    train, val, vocab = load_chars(args.text, device)
    print(f"charlm mixer head-to-head: vocab={vocab} d={args.d} L={args.layers} h={args.heads} "
          f"seq={args.seq} steps={args.steps} seeds={args.seeds} dev={device}", flush=True)
    print("bigram floor ~2.5 bpb; dense-conv ~2.4; target = clear floor + approach dense\n", flush=True)

    # (name, score, train_agg, anneal, [(deploy_name, deploy_agg, deploy_tau), ...])
    soft_arms = [
        ("dense",      "dot", "soft", False, [("soft", "soft", args.tau)]),  # fixed-tau softmax = nanoGPT ceiling
        ("l1_gather",  "l1",  "soft", True,  [("soft", "soft", args.tau_floor), ("gather", "gather", 0.0)]),
        ("l1_matmul",  "l1",  "soft", True,  [("soft", "soft", args.tau_floor), ("matmul", "matmul", 0.0)]),
        ("dot_gather", "dot", "soft", True,  [("soft", "soft", args.tau_floor), ("gather", "gather", 0.0)]),
        ("shift",      "l1",  "shift", True, [("soft_shift", "shift", args.delta_floor), ("gather", "gather", 0.0)]),  # mult-free DIFFERENTIABLE routing (AdamW); deploy gather
        ("mbr",        "l1",  "mbr",   True, [("soft_mbr", "mbr", args.delta_floor), ("gather", "gather", 0.0), ("topk_dyadic", "mbr", args.mbr_deploy_band)]),  # IDEA 5: margin-banded rank blend; deploy gather (Δ=0) or top-k dyadic blend (Δ>0)
        ("pshift",     "l1",  "shift", True, [("soft_pshift", "shift", args.delta_floor), ("gather", "gather", 0.0)]),  # WINNER: peaking-regularized finer-mantissa shift blend; deploy gather
        ("rtk",        "l1",  "rtk",   True, [("soft_rtk", "rtk", args.delta_floor), ("gather", "gather", 0.0)]),  # RUNNER-UP: residual top-k blend (finer top weight); deploy gather
    ]
    ff_arms = ["ff2", "ff2_shuffle", "lm_only_hard",    # teacher-free mult-free router + controls
               "ff2_adamw", "lm_only_hard_adamw",       # AdamW diagnostics (isolate router vs optimizer tax)
               "gradtok_circuit", "gradtok_circuit_shuffle",          # content-adaptive grad-token reader (the real test)
               "gradtok_circuit_adamw"]                 # + AdamW (isolate router from optimizer tax)
    if args.arms:
        keep = [a.strip() for a in args.arms.split(",")]
    else:
        keep = [a[0] for a in soft_arms]                # default: soft arms only (ff on request)
    run_soft = [a for a in soft_arms if a[0] in keep]
    run_ff = [n for n in keep if n in ff_arms]

    if run_ff:                                          # teacher-free target diagnostic (predicts underfit)
        print(f"target copyacc/cov (pre-train): {target_diag(train, args, device)}\n", flush=True)
    order, agg = [], {}
    for name, score, tagg, anneal, daggs in run_soft:
        vals = {}
        for seed in args.seeds:
            r = train_arm(name, score, tagg, anneal, daggs, seed, args, train, val, vocab, device)
            for k, v in r.items():
                vals.setdefault(k, []).append(v)
        agg[name] = vals; order.append(name)
    for name in run_ff:
        vals = {}
        for seed in args.seeds:
            r = train_ff_arm(name, seed, args, train, val, vocab, device)
            for k, v in r.items():
                vals.setdefault(k, []).append(v)
        agg[name] = vals; order.append(name)

    print("\n=== CHAR-LM MIXER RESULT (val bpb, mean +/- std over seeds) ===", flush=True)
    for name in order:
        for k, vlist in agg[name].items():
            mn = sum(vlist) / len(vlist)
            sd = (sum((x - mn) ** 2 for x in vlist) / len(vlist)) ** 0.5
            print(f"  {name:13s} {k:7s}  {mn:.3f} +/- {sd:.3f}   {['%.3f'%x for x in vlist]}", flush=True)
    print("\nReads: ff2(gather) vs l1_gather(gather)@same-stack = teacher-free-router tax; "
          "ff2 vs ff2_shuffle/lm_only_hard = content routing exists; "
          "l1_gather hard~soft = lossless deploy.", flush=True)


if __name__ == "__main__":
    main()
