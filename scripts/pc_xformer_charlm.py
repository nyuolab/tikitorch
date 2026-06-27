"""Phase A1 reference for the PC-fabric transformer LM (BUILD_PLAN.md, recipe "#3"): a char-LM trained with
WARM dot-product attention (the one allowed multiply) + a MULTIPLIER-FREE PC-trained bulk (dyadic +-2^k weights,
fine-dyadic LayerNorm, int16 activations, dyadic-Adam). Trains + SAMPLES coherent TinyShakespeare text. The same
script scales from a quick local proof up to ~350M on a GPU (set XF_D/XF_LAYER/... ; uses cuda if available).

  PYTHONPATH=. XF_D=384 XF_LAYER=6 XF_HEAD=6 XF_BLOCK=256 XF_STEPS=5000 python scripts/pc_xformer_charlm.py
"""
import os, sys, math, time
sys.path.insert(0, ".")
import torch, torch.nn as nn, torch.nn.functional as F

DATA   = os.environ.get("SHAKESPEARE", os.path.expanduser("~/.cache/tiur/tinyshakespeare.txt"))
D      = int(os.environ.get("XF_D", 256));    NL = int(os.environ.get("XF_LAYER", 4))
NH     = int(os.environ.get("XF_HEAD", 4));   BLOCK = int(os.environ.get("XF_BLOCK", 128))
STEPS  = int(os.environ.get("XF_STEPS", 3000)); BATCH = int(os.environ.get("XF_BATCH", 48))
LR     = float(os.environ.get("XF_LR", 3e-3)); WARMUP = int(os.environ.get("XF_WARMUP", 200))
MBITS  = int(os.environ.get("XF_MBITS", 2));  ABITS = int(os.environ.get("XF_ABITS", 9))
MULTFREE = int(os.environ.get("MULTFREE", 1))  # 1 = #3 mult-free bulk; 0 = full float (NanoGPT-style control)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---- multiplier-free bulk primitives (the on-fabric ops; straight-through gradients) ----
class DyadSTE(torch.autograd.Function):           # nearest signed power of two (== the kernel's dyad)
    @staticmethod
    def forward(ctx, w):
        a = w.abs(); nz = a > 0
        e = torch.where(nz, torch.round(torch.log2(a.clamp_min(1e-12))), torch.zeros_like(a))
        return torch.where(nz, torch.sign(w) * torch.pow(2.0, e), torch.zeros_like(w))
    @staticmethod
    def backward(ctx, g): return g

class DyLinear(nn.Linear):                         # dyadic-weight matmul = shift-add on fabric (DSP_MULT=0)
    def forward(self, x):
        w = DyadSTE.apply(self.weight) if MULTFREE else self.weight
        return F.linear(x, w, self.bias)

class QAct(torch.autograd.Function):               # per-tensor int16 activation max-shift (the cross-op band)
    @staticmethod
    def forward(ctx, x):
        m = x.abs().max().clamp_min(1e-9); sh = torch.floor(torch.log2(m)) - ABITS
        return torch.round(x / 2.0**sh) * 2.0**sh
    @staticmethod
    def backward(ctx, g): return g
def qact(x): return QAct.apply(x) if MULTFREE else x

class FDyadLN(nn.Module):                           # fine-dyadic LayerNorm (mult-free) | float LN if not MULTFREE
    def __init__(self, d):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.b = nn.Parameter(torch.zeros(d)); self.d = d
    def forward(self, x):
        if not MULTFREE: return F.layer_norm(x, (self.d,), self.g, self.b)
        mean = x.mean(-1, keepdim=True); c = x - mean
        scale = c.abs().mean(-1, keepdim=True).clamp_min(1e-9)
        e = torch.floor(torch.log2(scale))
        mant = torch.round((scale / 2.0**e - 1.0) * (2**MBITS)) / (2**MBITS)
        return (c / ((1.0 + mant) * 2.0**e)) * self.g + self.b


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = FDyadLN(D); self.ln2 = FDyadLN(D)
        self.qkv = DyLinear(D, 3*D); self.proj = DyLinear(D, D)
        self.fc1 = DyLinear(D, 4*D); self.fc2 = DyLinear(4*D, D)
    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = qact(q).view(B, T, NH, D//NH).transpose(1, 2)   # int16 acts feed the matmuls...
        k = qact(k).view(B, T, NH, D//NH).transpose(1, 2)
        v = qact(v).view(B, T, NH, D//NH).transpose(1, 2)
        # WARM attention = the one allowed multiply (dot-product score + softmax + p*V blend), in float
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(qact(y))
        x = x + self.fc2(qact(F.relu(self.fc1(self.ln2(x)))))
        return x


class CharLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D); self.pos = nn.Embedding(BLOCK, D)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.lnf = FDyadLN(D); self.head = DyLinear(D, vocab)
        self.vocab = vocab
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks: x = blk(x)
        return self.head(self.lnf(x))
    @torch.no_grad()
    def generate(self, idx, n):
        for _ in range(n):
            logits = self(idx[:, -BLOCK:])[:, -1, :]
            probs = F.softmax(logits, -1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], 1)
        return idx


# ---- dyadic-Adam: per-param pow2-adaptive step (mult-free on fabric), float here ----
def dyadic_adam(params, lr):
    for p in params:
        if p.grad is None: continue
        st = p._st; st['m'].mul_(0.875).add_(p.grad, alpha=0.125); st['v'].mul_(0.875).add_(p.grad.abs(), alpha=0.125)
        scale = torch.pow(2.0, torch.floor(torch.log2((st['m'].abs()/(st['v']+1e-9)).clamp_min(1e-12))))
        p.data.add_(-lr * torch.sign(st['m']) * scale)


def main():
    text = open(DATA).read(); chars = sorted(set(text)); vocab = len(chars)
    stoi = {c:i for i,c in enumerate(chars)}; itos = {i:c for c,i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    nparams = sum(p.numel() for p in CharLM(vocab).parameters())
    print(f"[charlm] recipe={'#3 mult-free bulk + warm attn' if MULTFREE else 'FULL FLOAT (control)'} | "
          f"D={D} L={NL} H={NH} block={BLOCK} vocab={vocab} params={nparams/1e6:.1f}M dev={DEV} steps={STEPS}", flush=True)
    torch.manual_seed(0)
    model = CharLM(vocab).to(DEV)
    if MULTFREE:
        for p in model.parameters(): p._st = {'m': torch.zeros_like(p), 'v': torch.zeros_like(p)}
        opt = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
    def get_batch():
        ix = torch.randint(0, len(data)-BLOCK-1, (BATCH,))
        x = torch.stack([data[i:i+BLOCK] for i in ix]); y = torch.stack([data[i+1:i+1+BLOCK] for i in ix])
        return x.to(DEV), y.to(DEV)
    # STALE_DELAY simulates the no-global-sync bubble-free pipeline: each layer's gradient is applied D steps
    # late (the pipeline round-trip), so every update uses a D-step-stale gradient. Backprop forbids this (one
    # global backward); PC tolerates it (local credit assignment). STALE_PERLAYER makes the delay grow with
    # block depth (block l lags ~ l+1 steps), the faithful pipeline schedule. 0 = synchronous baseline.
    STALE = int(os.environ.get("STALE_DELAY", 0))
    PERLAYER = int(os.environ.get("STALE_PERLAYER", 0))
    params = list(model.parameters())
    # per-param delay: uniform STALE, or (block-index+1)-scaled if per-layer
    def pdelay(p):
        if not PERLAYER: return STALE
        nm = pnames.get(id(p), "")
        import re as _re; m = _re.search(r"blocks\.(\d+)\.", nm)
        return (int(m.group(1))+1) * STALE if m else STALE
    pnames = {id(p): n for n, p in model.named_parameters()}
    gqueues = {id(p): [] for p in params}            # per-param FIFO of stale grad snapshots
    t0 = time.time()
    for step in range(STEPS):
        lr = LR * min(1.0, (step+1)/max(1,WARMUP))       # linear warmup
        x, y = get_batch()
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        model.zero_grad(set_to_none=True); loss.backward()
        if STALE > 0:                                     # delay each grad by its pipeline depth
            ready = True
            for p in params:
                if p.grad is None: continue
                q = gqueues[id(p)]; q.append(p.grad.detach().clone())
                d = pdelay(p)
                if len(q) > d: p.grad = q.pop(0)          # apply the d-step-stale grad
                else: p.grad = None; ready = False        # still filling the pipe
            if ready or step >= STALE:
                if opt:
                    for g in opt.param_groups: g['lr']=lr
                    opt.step()
                else: dyadic_adam(params, lr)
        elif opt:
            for g in opt.param_groups: g['lr']=lr
            opt.step()
        else: dyadic_adam(params, lr)
        if step % 250 == 0 or step == STEPS-1:
            print(f"  step {step:5d}  bpb={loss.item()/math.log(2):.3f}  lr={lr:.1e}  {(time.time()-t0):.0f}s", flush=True)
    # sample
    print("\n===== SAMPLE =====", flush=True)
    ctx = torch.tensor([[stoi.get('\n',0)]], device=DEV)
    out = model.generate(ctx, 400)[0].tolist()
    print(''.join(itos[i] for i in out), flush=True)


if __name__ == "__main__": main()
