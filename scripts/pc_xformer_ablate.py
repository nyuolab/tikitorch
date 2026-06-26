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


# ---- dyadic weight (the kernel's mult-free shift-add matmul), straight-through ----
class DyadSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        a = w.abs(); nz = a > 0
        e = torch.where(nz, torch.round(torch.log2(a.clamp_min(1e-12))), torch.zeros_like(a))
        return torch.where(nz, torch.sign(w) * torch.pow(2.0, e), torch.zeros_like(w))
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
        self.qkv = DyLinear(D, 3*D); self.proj = DyLinear(D, D)
        self.fc1 = DyLinear(D, 4*D); self.fc2 = DyLinear(4*D, D)
    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = qact(q).view(B, T, NH, D//NH).transpose(1, 2)
        k = qact(k).view(B, T, NH, D//NH).transpose(1, 2)
        v = qact(v).view(B, T, NH, D//NH).transpose(1, 2)
        if ATTN == "l1":
            att = -(q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(-1)   # -L1(q_i,k_j): the kernel's mult-free score
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D//NH)        # dot-product score (warm-train multiply)
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf"))
        y = (do_softmax(att) @ v).transpose(1, 2).reshape(B, T, D)    # soft p*V blend (the one fabric multiply)
        x = x + self.proj(qact(y))
        x = x + self.fc2(qact(F.relu(self.fc1(self.ln2(x)))))
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
    if OPT == "dyadic":
        for p in model.parameters(): p._st = {'m': torch.zeros_like(p), 'v': torch.zeros_like(p)}
        opt = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    last = 0.0
    for step in range(STEPS):
        x, y = get_batch(data, BATCH)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        if opt: opt.zero_grad(); loss.backward(); opt.step()
        else:
            model.zero_grad(); loss.backward(); dyadic_adam_step(list(model.parameters()), 3e-3)
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
    print(f"ABLATE [{TAG}] ATTN={ATTN} LN={LN} ACT={ACT} OPT={OPT} WQUANT={WQUANT} | "
          f"vocab={vocab} D={D} blk={NBLK} seq={SEQ} steps={STEPS}", flush=True)
    bpb, acc = train(data, vocab)
    print(f"RESULT [{TAG}] bpb={bpb:.3f} acc={acc:.3f}", flush=True)
