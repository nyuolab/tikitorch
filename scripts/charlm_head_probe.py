"""Head-comparison probe: a D=64/1-layer causal char-transformer (mirrors the s1 kernel structure:
embed -> [LN->attn->res->LN->FFN->res] -> LN_F -> HEAD) on lowercased tinyshakespeare, with a switchable
LM head. Question: does a LINEAR (dot-product) head beat the bigram baseline where the MAX-PLUS head does not,
at D=64/1L? And does a DYADIC (mult-free, weights snapped to +-2^k) linear head also work?
"""
import math, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

txt = open("data/tinyshakespeare.txt").read().lower()
chars = sorted(set(txt)); V = len(chars); stoi = {c:i for i,c in enumerate(chars)}
ids = np.array([stoi[c] for c in txt], dtype=np.int64)
n = len(ids); tr, va = ids[:int(.9*n)], ids[int(.9*n):]
D, H, DH, FFN, L = 64, 2, 32, 256, 64
print(f"V={V} D={D} H={H} L_SEQ={L} | train={len(tr)} val={len(va)}")

# baselines (bits/char) on val
from collections import Counter
uni = Counter(va); Ntot=len(va)
unig = -sum((c/Ntot)*math.log2(c/Ntot) for c in uni.values())
big = Counter(zip(va[:-1], va[1:])); c0 = Counter(va[:-1])
bigH = -sum((cnt/(len(va)-1))*math.log2(cnt/c0[a]) for (a,b),cnt in big.items())
print(f"BASELINES: unigram={unig:.4f}  bigram(cond)={bigH:.4f} bits/char")

def batch(split, bs=64):
    d = tr if split=="tr" else va
    ix = np.random.randint(0, len(d)-L-1, size=bs)
    x = torch.from_numpy(np.stack([d[i:i+L] for i in ix])).to(dev)
    y = torch.from_numpy(np.stack([d[i+1:i+1+L] for i in ix])).to(dev)
    return x, y

class DyadicLinear(nn.Module):
    """linear head whose weights are snapped to +-2^k (mult-free deploy) via STE."""
    def __init__(self, d, v):
        super().__init__(); self.w = nn.Parameter(torch.randn(v, d)*0.02)
    def forward(self, h):
        s = torch.sign(self.w); a = self.w.abs().clamp_min(1e-9)
        k = torch.round(torch.log2(a)); wq = s * torch.pow(2.0, k)        # +-2^k
        wq = self.w + (wq - self.w).detach()                              # STE
        return h @ wq.t()

class TinyXfmr(nn.Module):
    def __init__(self, head="linear"):
        super().__init__()
        self.emb = nn.Embedding(V, D); self.pos = nn.Parameter(torch.zeros(1, L, D))
        self.ln1=nn.LayerNorm(D); self.ln2=nn.LayerNorm(D); self.lnf=nn.LayerNorm(D)
        self.q=nn.Linear(D,D,bias=False); self.k=nn.Linear(D,D,bias=False)
        self.v=nn.Linear(D,D,bias=False); self.o=nn.Linear(D,D,bias=False)
        self.up=nn.Linear(D,FFN); self.dn=nn.Linear(FFN,D)
        self.head_kind = head
        if head=="linear":   self.head = nn.Linear(D, V, bias=False)
        elif head=="dyadic": self.head = DyadicLinear(D, V)
        elif head=="maxplus":self.W = nn.Parameter(torch.randn(V, D)*0.02)
        mask = torch.tril(torch.ones(L,L)); self.register_buffer("m", mask==0)
    def forward(self, x):
        B,T = x.shape
        h = self.emb(x) + self.pos[:,:T]
        z = self.ln1(h)
        q=self.q(z).view(B,T,H,DH).transpose(1,2); k=self.k(z).view(B,T,H,DH).transpose(1,2)
        v=self.v(z).view(B,T,H,DH).transpose(1,2)
        att=(q@k.transpose(-2,-1))/math.sqrt(DH); att=att.masked_fill(self.m[:T,:T], float("-inf"))
        att=att.softmax(-1); y=(att@v).transpose(1,2).reshape(B,T,D); h=h+self.o(y)
        z=self.ln2(h); h=h+self.dn(F.relu(self.up(z)))
        hf=self.lnf(h)
        if self.head_kind=="maxplus":
            # logits[v] = max_d (W[v,d] + hf[d])  (the s1 kernel head)
            logits = (hf.unsqueeze(2) + self.W.view(1,1,V,D)).max(-1).values
        else:
            logits = self.head(hf)
        return logits

def train_eval(head, steps=3000):
    m = TinyXfmr(head).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    t0=time.time()
    for s in range(steps):
        x,y = batch("tr"); logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1,V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    # val bpc
    m.eval(); tot=0.0; npos=0
    with torch.no_grad():
        for _ in range(40):
            x,y=batch("va"); lp = F.log_softmax(m(x),-1)
            nll = F.nll_loss(lp.reshape(-1,V), y.reshape(-1), reduction="sum")
            tot += nll.item()/math.log(2); npos += y.numel()
    bpc = tot/npos
    print(f"  HEAD={head:8s}  val_bpc={bpc:.4f}  (train_loss={loss.item()/math.log(2):.3f} bits)  {time.time()-t0:.0f}s")
    return bpc

print("\n=== D=64 / 1-layer, AdamW, 3000 steps ===")
r = {h: train_eval(h) for h in ["maxplus","linear","dyadic"]}
print(f"\nSUMMARY (val bits/char; bigram bar={bigH:.3f}):")
for h,b in r.items():
    print(f"  {h:8s}: {b:.4f}  {'BEATS bigram' if b<bigH else ('beats unigram' if b<unig else 'NO (>= unigram)')}")
