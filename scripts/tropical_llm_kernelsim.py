#!/usr/bin/env python3
"""Kernel-faithful QAT for the S1 tropical transformer LM.

The earlier "faithful" trainer matched the kernel in FORM (MADLN, dyad weights,
argmax-gather, int16) but NOT in the kernel's exact INTEGER arithmetic: it used a
TRUE float divide in LayerNorm, a fixed activation quant scale (round(x*256)), and
float residuals.  Measured on the REAL kernel forward (s1_deploy_eval) that gap is
~1-2 bpc: a clean model at 2.5 bpc deploys at 4.3-4.8 (worse than bigram).  emu_decomp
localized it: weight round-log2 (+0.27, tolerable) + LN 2-bit-mantissa reciprocal +
int-round-to-0 of sub-1 activations + the integer SCALES.

This trainer runs the forward through a BIT-FAITHFUL sim of the kernel's integer
pipeline (forge_train_step_s1.cpp do_fwd_loss), with straight-through estimators on
every non-differentiable op, so the model LEARNS weights/scales robust to all of it.
Because the deploy kernel == this sim, the on-silicon bpc == the trained bpc by
construction (validate against s1_deploy_eval after export).

Faithful ops replicated VERBATIM from the kernel:
  - embed+pos gather at scale SA (int16)
  - LN: mean=sum>>6, c=sat16(x-mean), meanabs=sum|c|>>6, fdyad 2-bit reciprocal
        r=c/meanabs (mantissa nq in [4,8), exponent e), gamma=dyad(round(g*SGAM))
        applied as r<<ge, +round(b*SGAM), sat16
  - GEMM (qx=0): out[t,o]=sum_i x_int[t,i] * dyad(round(W[o,i]*SW)) ; sat16   (DSP=0)
  - attn: L1 score s[i,j]=-sum_d|Q-K|, causal, argmax-GATHER out=V[jstar]      (DSP=0)
  - residual: sat16(R + branch)
  - head (HEAD_MODE=1): dyadic GEMM logits=Rn @ dyad(round(Whead*SH))
Model keys match tropical_llm_export_s1_faithful.py so export is unchanged.
"""
import argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D, H, DH, FFN, T, VREAL = 64, 2, 32, 256, 64, 39
LOG2_D, MBITS = 6, 2
I16 = 32767
ABL_LN_EXACT = False   # ablation: true float /meanabs (skip 2-bit mantissa)
ABL_W_EXACT  = False   # ablation: full-precision weights (skip dyad round-log2)
ACT_Q = 1.0            # activation quant grid (1.0=integer=kernel; >1 = finer)
ACT_NOCLIP = False     # ablation: disable sat16 clip
DYAD_K = 1             # weight = sum of K signed powers of 2 (1=current single-pow2; >1 = CSD, still shift-add)
GEMM_RESCALE = 0       # if >0: right-shift GEMM output by this (pow2 rescale; lets SW be large w/o overflow)
LN_MBITS = 2           # LN reciprocal mantissa bits (kernel=2; 4 = finer)

# ----------------------------- straight-through helpers -----------------------------
def ste_round(x):                      # round to the ACT_Q grid, identity gradient
    return x + (torch.round(x * ACT_Q) / ACT_Q - x).detach()

def sat16(x):                          # clip to int16, identity gradient in-range
    if ACT_NOCLIP: return x
    return x + (x.clamp(-32768, 32767) - x).detach()

def afloor(x):                         # arithmetic floor (matches >> for two's complement), no grad needed
    return torch.floor(x)

def dyad_scaled(w, scale):
    """kernel weight value used in the GEMM: sign * 2^round(log2(|round(w*scale)|)), with the
    nonzero mask of round(w*scale).  forward = exact decoded value (scaled-int units);
    backward flows to w as if value == w*scale (smooth surrogate)."""
    ws = w * scale
    if ABL_W_EXACT: return ws
    target = torch.round(ws)            # integer weight to approximate as sum of DYAD_K powers of 2
    hard = torch.zeros_like(target)
    resid = target
    for _ in range(DYAD_K):
        a = resid.abs()
        m = (a >= 0.5).float()          # add a term only while the residual still rounds to >=1
        e = torch.round(torch.log2(a.clamp_min(1.0)))
        term = torch.sign(resid) * torch.pow(2.0, e) * m
        hard = hard + term
        resid = resid - term
    soft = ws                           # smooth surrogate (grad path)
    return soft + (hard - soft).detach()

# ----------------------------- faithful LayerNorm -----------------------------
def kernel_ln(x, g, b, SGAM):
    """x: [...,D] integer-valued float. returns int16 LN output, faithful to layernorm_tok."""
    mean = afloor(x.sum(-1, keepdim=True) / float(1 << LOG2_D))
    c = sat16(x - mean)
    meanabs = afloor(c.abs().sum(-1, keepdim=True) / float(1 << LOG2_D)).clamp_min(1.0)

    # fdyad(meanabs) -> e, nq  (exact integer, detached: it's a fixed nonlinearity). mb = LN mantissa bits.
    with torch.no_grad():
        mb = LN_MBITS
        ma = meanabs
        p = afloor(torch.log2(ma))                       # highest set bit
        exact = (ma == torch.pow(2.0, p)).float()
        e = torch.where(exact > 0, -p, -(p + 1.0))
        nege = -e
        numer = torch.pow(2.0, mb + nege)
        thresh = numer + afloor(ma / 2.0)
        nbase = float(1 << mb)
        nq = torch.full_like(ma, nbase)
        for step in range(1, (1 << mb) + 1):
            acc = (nbase + step) * ma
            nq = torch.where(acc <= thresh, torch.full_like(ma, nbase + step), nq)
        over = (nq >= float(1 << (mb + 1)))
        nq = torch.where(over, torch.full_like(nq, nbase), nq)
        e = torch.where(over, e + 1.0, e)
        nq = nq.clamp_min(nbase)
        nqm4 = nq - nbase
        add = nqm4.clamp(0, (1 << mb) - 1)
        # exact RESC: r = ((c*(nbase+add))>>mb) then <<e / >>-e   (arithmetic floor shifts)
        cn = c * (nbase + add)
        r1 = afloor(cn / float(1 << mb))
        pe = torch.pow(2.0, e.abs())
        r_hard = torch.where(e >= 0, r1 * pe, afloor(r1 / pe))

    r_soft = c / meanabs                                  # smooth /meanabs (grad path)
    r = r_soft if ABL_LN_EXACT else (r_soft + (r_hard - r_soft).detach())

    gg = dyad_scaled(g, SGAM)                             # gamma value (=sign*2^ge), scaled-int units
    out = r * gg + ste_round(b * SGAM)
    return sat16(ste_round(out))

# ----------------------------- faithful GEMM (qx=0) -----------------------------
def kernel_gemm(x_int, w, scale):
    """x_int:[...,I] int-valued, w:[O,I].  out[...,o]=sum_i x_int[i]*dyad(round(w*scale)); optional
    pow2 rescale (>>GEMM_RESCALE) so SW can be large (precise weights) without sat16 overflow; sat16."""
    wv = dyad_scaled(w, scale)                            # [O,I] scaled-int weight values
    out = x_int @ wv.t()
    rs = 0
    if GEMM_RESCALE < 0:                                  # AUTO: rescale by this matrix's own (pow2) weight scale
        rs = int(round(math.log2(scale))) if scale > 1 else 0
    elif GEMM_RESCALE > 0:
        rs = GEMM_RESCALE
    if rs > 0:
        sm = out / float(1 << rs)                         # smooth (grad path)
        out = sm + (afloor(sm) - sm).detach()             # STE floor: fwd = arithmetic >> , bwd = divide
    return sat16(ste_round(out))

# ----------------------------- model -----------------------------
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1g = nn.Parameter(torch.ones(D));  s.ln1b = nn.Parameter(torch.zeros(D))
        s.ln2g = nn.Parameter(torch.ones(D));  s.ln2b = nn.Parameter(torch.zeros(D))
        s.q = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        s.k = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        s.v = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        s.o = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        s.up = nn.Parameter(torch.randn(FFN, D) / math.sqrt(D))
        s.dn = nn.Parameter(torch.randn(D, FFN) / math.sqrt(FFN))

# parameter-name shim so export (which reads blocks.L.q.w etc.) works unchanged
class _W(nn.Module):
    def __init__(s, p): super().__init__(); s.w = p

def kernel_attn(Qm, Km, Vm, invtau, mask, soft_deploy=False):
    """Qm/Km/Vm: [B,T,D] int.  per head L1 score s[i,j]=-sum_d|Q-K|, then SOFT-annealed blend:
      FORWARD = sat16(round(softmax(score*invtau) @ V)).  invtau is annealed LOW (genuinely soft;
      L1 scores are ~hundreds so invtau~0.002 is the soft regime) -> HIGH over training.  At eval
      invtau=1e6 the softmax IS one-hot argmax = the kernel's argmax-GATHER deploy (S1_ATTN_HARD=1, DSP=0).
      Training forward=soft avoids the from-scratch argmax stall; deploy=hard gather (mult-free)."""
    outs = []
    for h in range(H):
        sl = slice(h * DH, (h + 1) * DH)
        Q, K, V = Qm[..., sl], Km[..., sl], Vm[..., sl]              # [B,T,DH]
        score = -(Q.unsqueeze(2) - K.unsqueeze(1)).abs().sum(-1)     # [B,T,T]
        score = score.masked_fill(mask, float("-inf"))
        p = torch.softmax(score * invtau, dim=-1)
        outs.append(sat16(ste_round(p @ V)))
    return torch.cat(outs, -1)                                       # [B,T,D]

class KSimLM(nn.Module):
    def __init__(s, nl, SA, SW, SGAM, SH, soft=False):
        super().__init__()
        s.nl, s.SA, s.SW, s.SGAM, s.SH, s.soft = nl, SA, SW, SGAM, SH, soft
        s.emb = nn.Embedding(VREAL, D)
        s.pos = nn.Parameter(torch.zeros(1, T, D))
        s.blocks = nn.ModuleList([Block() for _ in range(nl)])
        s.lnfg = nn.Parameter(torch.ones(D));  s.lnfb = nn.Parameter(torch.zeros(D))
        s.head = nn.Parameter(torch.randn(VREAL, D) / math.sqrt(D))
        nn.init.normal_(s.emb.weight, std=0.5)

    def export_state(s):
        """remap to the key layout tropical_llm_export_s1_faithful.py expects."""
        sd = {}
        sd['emb.weight'] = s.emb.weight.data
        sd['pos'] = s.pos.data
        for L, blk in enumerate(s.blocks):
            for nm, p in [('q', blk.q), ('k', blk.k), ('v', blk.v), ('o', blk.o), ('up', blk.up), ('dn', blk.dn)]:
                sd[f'blocks.{L}.{nm}.w'] = p.data
            sd[f'blocks.{L}.ln1.g'] = blk.ln1g.data; sd[f'blocks.{L}.ln1.b'] = blk.ln1b.data
            sd[f'blocks.{L}.ln2.g'] = blk.ln2g.data; sd[f'blocks.{L}.ln2.b'] = blk.ln2b.data
        sd['lnf.g'] = s.lnfg.data; sd['lnf.b'] = s.lnfb.data
        sd['head.w'] = s.head.data
        return sd

    def forward(s, idx, invtau):
        B, Tt = idx.shape
        mask = torch.triu(torch.ones(Tt, Tt, device=idx.device), 1).bool()
        # embed + pos at scale SA (int16)
        emb_i = ste_round(s.emb(idx) * s.SA)
        pos_i = ste_round(s.pos[:, :Tt] * s.SA)
        R = sat16(emb_i + pos_i)
        SW = s.SW
        for blk in s.blocks:
            X1 = kernel_ln(R, blk.ln1g, blk.ln1b, s.SGAM)
            Qm = kernel_gemm(X1, blk.q, SW['q']); Km = kernel_gemm(X1, blk.k, SW['k']); Vm = kernel_gemm(X1, blk.v, SW['v'])
            A = kernel_attn(Qm, Km, Vm, invtau, mask, soft_deploy=s.soft)
            AO = kernel_gemm(A, blk.o, SW['o'])
            R = sat16(R + AO)
            X2 = kernel_ln(R, blk.ln2g, blk.ln2b, s.SGAM)
            U = kernel_gemm(X2, blk.up, SW['up'])
            Hd = F.relu(U)
            Dn = kernel_gemm(Hd, blk.dn, SW['dn'])
            R = sat16(R + Dn)
        Rn = kernel_ln(R, s.lnfg, s.lnfb, s.SGAM)
        logits = kernel_gemm(Rn, s.head, s.SH)            # [B,T,V] int
        return logits

# ----------------------------- data -----------------------------
def load():
    t = open(REPO / "data" / "tinyshakespeare.txt").read().lower()
    ch = sorted(set(t)); st = {c: i for i, c in enumerate(ch)}
    ids = np.array([st[c] for c in t], dtype=np.int64); n = len(ids)
    return ids[:int(.9 * n)], ids[int(.9 * n):], len(ch)

def batch(data, B, dev):
    ix = np.random.randint(0, len(data) - T - 1, B)
    x = np.stack([data[i:i + T] for i in ix]); y = np.stack([data[i + 1:i + 1 + T] for i in ix])
    return torch.tensor(x, device=dev), torch.tensor(y, device=dev)

def bpc(model, data, dev, invtau, temp, iters=40, B=64):
    model.eval(); tot = 0.; n = 0
    with torch.no_grad():
        for _ in range(iters):
            x, y = batch(data, B, dev)
            lg = model(x, invtau).float() / temp
            tot += F.cross_entropy(lg.reshape(-1, VREAL), y.reshape(-1)).item() * y.numel(); n += y.numel()
    model.train(); return tot / n / math.log(2)

def bpc_sweep(model, data, dev, invtau, iters=40, B=64):
    """sweep the deploy-time logit temperature (like s1_deploy_eval) -> best bpc + its temp."""
    best, bt = 1e9, 0
    for temp in [16, 32, 64, 96, 128, 192, 256, 384, 512, 768]:
        v = bpc(model, data, dev, invtau, temp, iters=iters, B=B)
        if v < best: best, bt = v, temp
    return best, bt

# ----------------------------- train -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--B', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--SA', type=int, default=64)
    ap.add_argument('--SGAM', type=int, default=32)
    ap.add_argument('--SH', type=int, default=8)
    ap.add_argument('--SW', type=str, default='q:2,k:2,v:2,o:2,up:2,dn:2')
    ap.add_argument('--temp', type=float, default=8.0, help='logit temperature for CE (matches head scale)')
    ap.add_argument('--tau0', type=float, default=0.002, help='initial attn inv-temp (SOFT: L1 scores~hundreds)')
    ap.add_argument('--tau1', type=float, default=0.2, help='final attn inv-temp (anneal up -> near-hard)')
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--soft', type=int, default=0, help='1=soft p*V blend attention (S1_ATTN_HARD=0, ONE multiply)')
    ap.add_argument('--ln_exact', type=int, default=0, help='ablation: exact /meanabs LN')
    ap.add_argument('--w_exact', type=int, default=0, help='ablation: full-precision weights')
    ap.add_argument('--fine_act', type=int, default=0, help='ablation: fine (1/16) unclipped activations ~= float')
    ap.add_argument('--dyad_k', type=int, default=1, help='weight = sum of K signed powers of 2 (CSD; 1=current)')
    ap.add_argument('--mbits', type=int, default=2, help='LN reciprocal mantissa bits (kernel=2; 4=finer)')
    ap.add_argument('--D', type=int, default=64, help='model width (power of 2)')
    ap.add_argument('--H', type=int, default=2, help='attention heads')
    ap.add_argument('--ffn', type=int, default=256, help='FFN inner dim')
    ap.add_argument('--gemm_rescale', type=int, default=0, help='pow2 right-shift of GEMM output (rescale)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    global ABL_LN_EXACT, ABL_W_EXACT, ACT_Q, ACT_NOCLIP, D, H, DH, FFN, LOG2_D
    D = a.D; H = a.H; DH = a.D // a.H; FFN = a.ffn; LOG2_D = int(round(math.log2(a.D)))
    ABL_LN_EXACT = bool(a.ln_exact); ABL_W_EXACT = bool(a.w_exact)
    if a.fine_act: ACT_Q = 16.0; ACT_NOCLIP = True
    global DYAD_K, LN_MBITS
    global GEMM_RESCALE
    DYAD_K = a.dyad_k; LN_MBITS = a.mbits; GEMM_RESCALE = a.gemm_rescale
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    SW = {kv.split(':')[0]: float(kv.split(':')[1]) for kv in a.SW.split(',')}

    tr, va, vocab = load()
    assert vocab == VREAL, f'vocab {vocab} != {VREAL}'
    model = KSimLM(a.L, a.SA, SW, a.SGAM, a.SH, soft=a.soft).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)

    for step in range(a.steps):
        frac = step / a.steps
        invtau = a.tau0 + (a.tau1 - a.tau0) * frac          # anneal attention -> hard
        x, y = batch(tr, a.B, dev)
        lg = model(x, invtau).float() / a.temp
        loss = F.cross_entropy(lg.reshape(-1, VREAL), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f'[{step}] loss {loss.item():.3f}  invtau {invtau:.2f}', flush=True)

    soft, st = bpc_sweep(model, va, dev, a.tau1)
    hard, ht = bpc_sweep(model, va, dev, 1e6)                # invtau huge -> argmax (== kernel deploy)
    bg = 3.472
    print(f'[RESULT] kernel-sim QAT  soft_bpc={soft:.4f}(t{st})  HARD_bpc={hard:.4f}(t{ht})  bigram={bg}  '
          f'beats_bigram={"YES" if hard < bg else "NO"}')
    if a.out:
        torch.save({'model': model.export_state(), 'cfg': {'L': a.L, 'SA': a.SA, 'SW': SW, 'SGAM': a.SGAM, 'SH': a.SH, 'temp': a.temp, 'dyad_k': a.dyad_k, 'mbits': a.mbits, 'D': a.D, 'H': a.H, 'ffn': a.ffn}}, a.out)
        print(f'[saved] {a.out}')

if __name__ == '__main__':
    main()
