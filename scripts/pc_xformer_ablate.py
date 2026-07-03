"""ABLATION: the de-risk (pc_xformer_dyadic_ref.py) hits ~2.4 bpb with {dot-attn, float-LN, float-act, AdamW,
dyadic-weights}. The KERNEL (forge_pc_xformer_block) additionally swaps in {L1-attn, fine-dyadic-LN, per-op
int16-act, dyadic-Adam}. The on-fabric recipe trains WEAKLY. Which of those four kernel-components costs the
accuracy? This harness reproduces the de-risk and toggles each kernel-component independently so the per-axis
bpb cost is isolated (the integer cascade backward is NOT toggled: per the project's Z-IL finding it just
*computes* backprop, so float autograd is a faithful stand-in).

  ATTN={dot,l1} LN={float,fdyad} ACT={float,i16} OPT={adamw,dyadic} WQUANT={dyad,float} \
    PYTHONPATH=. python scripts/pc_xformer_ablate.py
"""
import os, sys, math
sys.path.insert(0, ".")
import torch, torch.nn as nn, torch.nn.functional as F

DATA  = os.environ.get("SHAKESPEARE", os.path.expanduser("~/.cache/tiur/tinyshakespeare.txt"))
D     = int(os.environ.get("XF_D", 64));     NH    = int(os.environ.get("XF_HEADS", 2))
NBLK  = int(os.environ.get("XF_BLOCKS", 8));  SEQ   = int(os.environ.get("XF_SEQ", 64))
STEPS = int(os.environ.get("XF_STEPS", 2000)); BATCH = int(os.environ.get("XF_BATCH", 32))
MBITS = int(os.environ.get("XF_MBITS", 2))    # fine-dyadic LN mantissa bits
ABITS = int(os.environ.get("XF_ABITS", 9))    # int16 activation band

ATTN   = os.environ.get("ATTN", "dot")        # dot | l1
LN     = os.environ.get("LN", "float")        # float | fdyad
ACT    = os.environ.get("ACT", "float")       # float | i16
OPT    = os.environ.get("OPT", "adamw")       # adamw | dyadic
WQUANT = os.environ.get("WQUANT", "dyad")     # dyad | float
TAG    = os.environ.get("TAG", "")

# HIPREC_ATTN: keep the ENTIRE attention block (Q/K/V/O projections, the score, the
# softmax, and the p*V blend) in FLOAT, while everything else (FFN, head, embedding,
# LN, the cross-chip boundary) stays the tropical multiplier-free recipe. Measures how
# much of the float->tropical gap is specifically the integer ATTENTION (the suspected
# wall) vs the rest of the network. NOT fully multiplier-free (the warm-train attention
# / "one allowed multiply" path).
HIPREC_ATTN = int(os.environ.get("HIPREC_ATTN", 0))

# HIPREC_MF (adversarial control): a MULTIPLIER-FREE analog of HIPREC_ATTN. Keeps the
# tropical recipe everywhere (dyadic weights, L1 mult-free score, mult-free softmax via
# do_softmax) but RAISES the activation precision inside attention by skipping the int16
# activation quant on q/k/v and on the attn-out projection input. Raising activation
# precision is multiplier-free-legal (you just don't quantize). Isolates whether the
# HIPREC_ATTN lift comes from precision-in-attention (free) or from the float multiply
# (forbidden). No nn.Linear, no dot-product, no float softmax.
HIPREC_MF = int(os.environ.get("HIPREC_MF", 0))


# ---- dyadic weight (the kernel's mult-free shift-add matmul), straight-through ----
WBITS2 = int(os.environ.get("WBITS2", 0))   # 1: represent each weight as a SUM of 2 signed powers of two
def _pow2_round(w):
    """Quantize each entry of w to its nearest single signed power of two (pure shift). Zeros stay zero."""
    a = w.abs(); nz = a > 0
    e = torch.where(nz, torch.round(torch.log2(a.clamp_min(1e-12))), torch.zeros_like(a))
    return torch.where(nz, torch.sign(w) * torch.pow(2.0, e), torch.zeros_like(w))
class DyadSTE(torch.autograd.Function):
    """Straight-through dyadic weight quant. Default = nearest single power of two (one shift per weight).
    WBITS2=1 = 2-term residual: w ~ q1 + q2, each q a signed power of two (two shifts + an add per weight,
    still multiplier-free). Strictly more weight precision (~one extra mantissa bit's worth)."""
    @staticmethod
    def forward(ctx, w):
        q1 = _pow2_round(w)
        if WBITS2 < 1:
            return q1
        q2 = _pow2_round(w - q1)       # quantize the leftover to a second signed power of two
        return q1 + q2
    @staticmethod
    def backward(ctx, g): return g
def dyq(w): return DyadSTE.apply(w) if WQUANT == "dyad" else w
QMASTER = int(os.environ.get("QMASTER", 0))   # if >0: int master weight at scale 2^QMASTER (golden's int16 master); STE forward
class QMasterSTE(torch.autograd.Function):     # forward: round(w*2^Q)/2^Q clamped to int16 (the golden's master); grad straight-through
    @staticmethod
    def forward(ctx, w): s=float(1<<QMASTER); return torch.clamp(torch.round(w*s), -32767, 32767)/s
    @staticmethod
    def backward(ctx, g): return g
def qmaster(w): return QMasterSTE.apply(w) if QMASTER>0 else w
class DyLinear(nn.Linear):
    def forward(self, x): return F.linear(x, dyq(qmaster(self.weight)), self.bias)


# ---- Hadamard rotation for outlier-spreading (mult-free: ±1 fast Walsh-Hadamard transform + a fixed 1/sqrt(n) shift) ----
HADAMARD = int(os.environ.get("HADAMARD", 0))     # 1: rotate activations into the Hadamard basis before int16-quant; fold H^T into the weight
_HCACHE = {}
def hadamard_matrix(n):
    """Normalized n x n Hadamard (orthonormal: H @ H.T = I). n must be a power of 2.
    Entries are +/- 1/sqrt(n); the +/-1 part is a pure sign/add transform (fast WHT = mult-free), the
    1/sqrt(n) is a single fixed per-tensor scale (folded into the existing max-shift on the fabric)."""
    if n in _HCACHE: return _HCACHE[n]
    assert (n & (n - 1)) == 0, f"Hadamard needs power-of-2 dim, got {n}"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / math.sqrt(n)
    _HCACHE[n] = H
    return H

class HadDyLinear(nn.Linear):
    """Hadamard-rotated mult-free linear. Computes  y = (qact(x @ H)) @ (H^T W^T) + b.
    In float this equals x W^T + b exactly (H H^T = I); under int16 activation quant the rotation spreads
    per-channel outliers so the per-tensor max-shift wastes less dynamic range -> a more accurate matmul.
    The activation quant (qact) is applied IN the rotated basis (that's the whole point). H is a fixed
    sign/shift-add transform; H^T folds into the (already-dyadic) weight, so no extra runtime multiply."""
    def forward(self, x):
        n = x.shape[-1]
        H = hadamard_matrix(n).to(x.dtype).to(x.device)
        xr = x @ H                                  # rotate activation into Hadamard basis (sign/add transform)
        xr = qact(xr)                               # quantize the ROTATED activation (outliers spread)
        Wf = qmaster(self.weight) @ H               # fold H into the weight side (offline; W is float master here)
        return F.linear(xr, dyq(Wf), self.bias)     # dyadic weight matmul (H^T W^T absorbed, result == x W^T in float)


# ---- per-op int16 activation max-shift (straight-through) ----
def maxshift_q(x, bits):
    m = x.abs().max().clamp_min(1e-9)
    sh = torch.floor(torch.log2(m)).item() - bits
    return torch.round(x / 2.0**sh) * 2.0**sh
class QAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return maxshift_q(x, ABITS)
    @staticmethod
    def backward(ctx, g): return g
def qact(x): return QAct.apply(x) if ACT == "i16" else x

class QBound(torch.autograd.Function):     # int16 neighbour message between chips (always on — the shard boundary)
    @staticmethod
    def forward(ctx, x): return maxshift_q(x, ABITS)
    @staticmethod
    def backward(ctx, g): return maxshift_q(g, 12)
def qbound(x): return QBound.apply(x)


# ---- fine-dyadic LayerNorm (the kernel's g_layernorm: mean-center, scale by a MBITS-mantissa 1/mean-abs-dev) ----
class FDyadLN(nn.Module):
    def __init__(self, d):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.b = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True); c = x - mean
        if os.environ.get("LNSCALE", "mad") == "rms":
            scale = (c*c).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()     # RMS (like float LayerNorm)
        else:
            scale = c.abs().mean(-1, keepdim=True).clamp_min(1e-9)           # mean-abs-deviation (kernel's choice)
        e    = torch.floor(torch.log2(scale))
        mant = torch.round((scale / 2.0**e - 1.0) * (2**MBITS)) / (2**MBITS) # fine-dyadic mantissa (MBITS bits)
        r    = c / ((1.0 + mant) * 2.0**e)                          # invq (straight-through via the divide)
        y_q  = r * self.g + self.b
        grad = os.environ.get("LNGRAD", "straight")
        if grad == "exact":          # forward = quantized fdyad, backward = exact differentiable LayerNorm gradient
            y_x = F.layer_norm(x, (x.shape[-1],), self.g, self.b)
            return y_x + (y_q - y_x).detach()
        if grad == "center":         # tropical: forward fdyad, backward = mean-centered g_y only (no variance-coupling)
            xn = (c / scale).detach()                              # x_hat (detached: only used to shape the value path)
            y_c = (c / scale.detach()) * self.g + self.b          # mean-centering flows (c has the mean op); scale detached
            return y_c + (y_q - y_c).detach()
        return y_q                   # straight-through (kernel's current: detached quantized scale)
def mkln(d): return FDyadLN(d) if LN == "fdyad" else nn.LayerNorm(d)


SMBITS  = int(os.environ.get("XF_SMBITS", 6))
LUT_RES = int(os.environ.get("LUT_RES", 8))      # gap-steps per probability-octave (8 = current coarse LUT)
LUT_N2  = int(os.environ.get("LUT_N2", 256))     # LUT entries (must cover gap_max = 2^SMBITS)
_LUTQ = torch.tensor([round(65536 * 2.0**(-g/LUT_RES)) for g in range(LUT_N2)], dtype=torch.float)
class IntSoftmax(torch.autograd.Function):
    """The golden's exp2-LUT softmax: adaptive shift (spread->SMBITS bits) + gap=(max-s)>>shift + LUT[gap] + Q16 norm.
    Forward = the coarse LUT probabilities; backward = the standard softmax-of-those-probs gradient (the golden's)."""
    @staticmethod
    def forward(ctx, att):
        finite = torch.isfinite(att)
        mx = torch.where(finite, att, torch.full_like(att, -1e30)).max(-1, keepdim=True).values
        mn = torch.where(finite, att, torch.full_like(att, 1e30)).min(-1, keepdim=True).values
        spread = (mx - mn).clamp_min(1.0)
        shift = (torch.floor(torch.log2(spread)) + 1 - SMBITS).clamp_min(0)
        gap = ((mx - att) / torch.pow(2.0, shift)).floor().clamp(0, LUT_N2-1)
        gap = torch.where(finite, gap, torch.full_like(gap, float(LUT_N2-1)))
        p = _LUTQ.to(att.device)[gap.long()]
        p = torch.where(finite, p, torch.zeros_like(p))
        probs = p / p.sum(-1, keepdim=True).clamp_min(1.0)
        ctx.save_for_backward(probs); return probs
    @staticmethod
    def backward(ctx, g):
        (probs,) = ctx.saved_tensors
        return probs * (g - (probs * g).sum(-1, keepdim=True))
EXPFRAC = int(os.environ.get("EXPFRAC", 8))    # fractional bits of the mantissa exp LUT
TEMP    = int(os.environ.get("TEMP", 0))       # temperature shift on the raw gap (0 = temp 1.0, matching float)
_LUTEXP = torch.tensor([round(65536 * 2.0**(-f/(1<<EXPFRAC))) for f in range(1<<EXPFRAC)], dtype=torch.float)
class PreciseSoftmax(torch.autograd.Function):
    """Tropical PRECISE int softmax: exp(-d) = LUT[frac(u)] >> int(u), u = d*log2(e) via shift-add (x1.4375).
    Right temperature (raw gap d, like float) + fine mantissa LUT -> no coarse bucketing. Mult-free (ROM+shift+add)."""
    @staticmethod
    def forward(ctx, att):
        finite = torch.isfinite(att)
        mx = torch.where(finite, att, torch.full_like(att, -1e30)).max(-1, keepdim=True).values
        d = (mx - att).clamp_min(0)
        di = torch.round(d / (2.0**TEMP))                       # integer gap (optional temperature)
        base = di * float(1 << EXPFRAC)
        u = base + torch.floor(base/4) + torch.floor(base/8) + torch.floor(base/16)   # base * 1.4375 ~ d/ln2
        intp = torch.floor(u / (1 << EXPFRAC))
        frac = (u - intp*(1 << EXPFRAC)).long().clamp(0, (1<<EXPFRAC)-1)
        p = _LUTEXP.to(att.device)[frac] / torch.pow(2.0, intp.clamp(0, 40))
        p = torch.where(finite & (intp <= 31), p, torch.zeros_like(p))
        probs = p / p.sum(-1, keepdim=True).clamp_min(1.0)
        ctx.save_for_backward(probs); return probs
    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors; return p*(g - (p*g).sum(-1, keepdim=True))
class Q16Softmax(torch.autograd.Function):     # float softmax then quantize probs to Q16 (1/65536) — isolates "is it the prob precision or the LUT/shift structure?"
    @staticmethod
    def forward(ctx, att):
        p = att.softmax(-1); pq = torch.round(p*65536)/65536; ctx.save_for_backward(pq); return pq
    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors; return p*(g - (p*g).sum(-1, keepdim=True))
def do_softmax(att):
    mode = os.environ.get("SOFTMAX","float")
    if mode=="int":     return IntSoftmax.apply(att)
    if mode=="q16":     return Q16Softmax.apply(att)
    if mode=="precise": return PreciseSoftmax.apply(att)
    return att.softmax(-1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = mkln(D); self.ln2 = mkln(D)
        # HIPREC_ATTN: the attention Q/K/V (qkv) and output (proj) projections run in
        # plain float (no dyadic weight quant, no qmaster, no Hadamard) — the whole
        # attention block is the "one allowed multiply" warm path. FFN stays tropical.
        if HIPREC_ATTN:
            self.qkv = nn.Linear(D, 3*D)
            self.proj = nn.Linear(D, D)
        else:
            self.qkv = DyLinear(D, 3*D)
            ProjLin = HadDyLinear if HADAMARD else DyLinear   # rotate outlier-prone attn-out proj input
            self.proj = ProjLin(D, D)
        Lin = HadDyLinear if HADAMARD else DyLinear     # rotate the MLP down-proj input
        self.fc1 = DyLinear(D, 4*D); self.fc2 = Lin(4*D, D)
    def forward(self, x):
        B, T, _ = x.shape
        hp = HIPREC_ATTN
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        # In the hi-precision attention block the activations are NOT int16-quantized.
        # HIPREC_MF: same precision relief (skip int16 on q/k/v) but STAYS mult-free
        # (dyadic qkv weights, L1 score, mult-free softmax all unchanged below).
        aqk = (lambda t: t) if (hp or HIPREC_MF) else qact
        q = aqk(q).view(B, T, NH, D//NH).transpose(1, 2)
        k = aqk(k).view(B, T, NH, D//NH).transpose(1, 2)
        v = aqk(v).view(B, T, NH, D//NH).transpose(1, 2)
        if ATTN == "l1" and not hp:
            att = -(q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(-1)   # -L1(q_i,k_j): the kernel's mult-free score
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D//NH)        # dot-product score (float in hp; warm-train multiply otherwise)
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf"))
        # hi-precision: exact float softmax + float p*V blend.
        smx = att.softmax(-1) if hp else do_softmax(att)
        y = (smx @ v).transpose(1, 2).reshape(B, T, D)               # soft p*V blend
        # HadDyLinear quantizes its input internally (in the rotated basis); plain DyLinear needs the outer qact.
        # In the hi-precision block the attn-out projection sees the un-quantized float blend.
        if hp or HIPREC_MF:
            aq = (lambda t: t)
        else:
            aq = (lambda t: t) if HADAMARD else qact
        x = x + self.proj(aq(y))
        x = x + self.fc2(aq(F.relu(self.fc1(self.ln2(x)))))
        return x


class CharXformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D); self.pos = nn.Embedding(SEQ, D)
        self.blocks = nn.ModuleList([Block() for _ in range(NBLK)])
        self.lnf = mkln(D); self.head = DyLinear(D, vocab)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i < len(self.blocks) - 1: x = qbound(x)                # int16 cross-chip boundary
        return self.head(self.lnf(x))


# ---- dyadic-Adam: per-param pow2-adaptive step (mult-free on fabric), float proxy here ----
def dyadic_adam_step(params, lr):
    for p in params:
        if p.grad is None: continue
        st = p._st
        st['m'].mul_(0.875).add_(p.grad, alpha=0.125)
        st['v'].mul_(0.875).add_(p.grad.abs(), alpha=0.125)
        scale = torch.pow(2.0, torch.floor(torch.log2((st['m'].abs() / (st['v'] + 1e-9)).clamp_min(1e-9))))
        p.data.add_(-lr * torch.sign(st['m']) * scale)


def get_batch(data, n):
    ix = torch.randint(0, len(data) - SEQ - 1, (n,))
    return torch.stack([data[i:i+SEQ] for i in ix]), torch.stack([data[i+1:i+1+SEQ] for i in ix])


def train(data, vocab, seed=0):
    torch.manual_seed(seed)
    model = CharXformer(vocab)
    LR = float(os.environ.get("XF_LR", "3e-3"))
    if OPT == "dyadic":
        for p in model.parameters(): p._st = {'m': torch.zeros_like(p), 'v': torch.zeros_like(p)}
        opt = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
    last = 0.0
    for step in range(STEPS):
        x, y = get_batch(data, BATCH)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        if opt: opt.zero_grad(); loss.backward(); opt.step()
        else:
            model.zero_grad(); loss.backward(); dyadic_adam_step(list(model.parameters()), LR)
        last = loss.item() / math.log(2)
        if step % 400 == 0 or step == STEPS - 1:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"   [{TAG}] step {step:5d}  bpb={last:.3f}  acc={acc:.3f}", flush=True)
    acc = (logits.argmax(-1) == y).float().mean().item()
    return last, acc


if __name__ == "__main__":
    text = open(DATA).read()
    chars = sorted(set(text)); vocab = len(chars); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"ABLATE [{TAG}] ATTN={ATTN} LN={LN} ACT={ACT} OPT={OPT} WQUANT={WQUANT} "
          f"HIPREC_ATTN={HIPREC_ATTN} | "
          f"vocab={vocab} D={D} blk={NBLK} seq={SEQ} steps={STEPS}", flush=True)
    bpb, acc = train(data, vocab, seed=int(os.environ.get("XF_SEED", 0)))
    print(f"RESULT [{TAG}] bpb={bpb:.3f} acc={acc:.3f}", flush=True)
