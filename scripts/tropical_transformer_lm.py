"""Fully MULTIPLIER-FREE tropical transformer char-LM, measured vs a dense transformer.

Attention = MAX-PLUS multi-head: score_ij = -L1(Q_i, K_j) (mult-free), causal. Trained with a tau-ANNEALED
soft relaxation (softmax(score/tau), tau high->low) so the argmax is learnable; DEPLOYED HARD as argmax-GATHER
(copy the winning position's value -- the coherent op, NOT per-channel max-plus-matmul). All linears (Q/K/V/O,
FFN, embed, head) are DYADIC (+-2^k via STE -> shift-add, DSP=0). The only multiply is the soft-blend at
TRAIN time (warm-train + mult-free-deploy); the reported HARD bpc is the fully mult-free deploy number.

Reports: dense transformer (softmax, float) reference; tropical (soft train-time bpc) and tropical (HARD deploy
bpc) -- the latter is the multiplier-free result. Char-LM on tinyshakespeare (lowercased). bigram ~3.47.
"""
import math, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

dev = "mps" if torch.backends.mps.is_available() else "cpu"
REPO = Path(__file__).resolve().parents[1]

def load():
    txt = open(REPO/"data"/"tinyshakespeare.txt").read().lower()
    chars = sorted(set(txt)); stoi = {c:i for i,c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], dtype=np.int64)
    n = len(ids); return ids[:int(.9*n)], ids[int(.9*n):], len(chars)

def dyad(w):  # snap to +-2^k via STE (mult-free deploy weights)
    s = torch.sign(w); a = w.abs().clamp_min(1e-9); k = torch.round(torch.log2(a))
    return w + (s*torch.pow(2.0,k) - w).detach()

class DLin(nn.Module):  # dyadic linear (mult-free at deploy)
    def __init__(s,i,o,bias=False):
        super().__init__(); s.w=nn.Parameter(torch.randn(o,i)*(1/math.sqrt(i)))
        s.b=nn.Parameter(torch.zeros(o)) if bias else None
    def forward(s,x):
        y=x@dyad(s.w).t(); return y if s.b is None else y+s.b
class FLin(nn.Module):  # float linear (dense reference)
    def __init__(s,i,o,bias=False): super().__init__(); s.l=nn.Linear(i,o,bias=bias)
    def forward(s,x): return s.l(x)

class Block(nn.Module):
    def __init__(s, D, H, T, tropical):
        super().__init__(); s.H=H; s.DH=D//H; s.T=T; s.trop=tropical
        Lin = DLin if tropical else FLin
        s.ln1=nn.LayerNorm(D); s.ln2=nn.LayerNorm(D)
        s.q=Lin(D,D); s.k=Lin(D,D); s.v=Lin(D,D); s.o=Lin(D,D)
        s.up=Lin(D,4*D); s.dn=Lin(4*D,D)
        s.register_buffer("cmask", torch.triu(torch.ones(T,T),1).bool())  # j>i masked
    def attn(s, x, tau, hard):
        B,T,D = x.shape; H,DH = s.H, s.DH
        q=s.q(s.ln1(x)).view(B,T,H,DH).transpose(1,2)   # B,H,T,DH
        k=s.k(s.ln1(x)).view(B,T,H,DH).transpose(1,2)
        v=s.v(s.ln1(x)).view(B,T,H,DH).transpose(1,2)
        if s.trop:
            # -L1 score (mult-free): per-head to bound memory (cdist backward is unimplemented on MPS).
            sc=[]
            for h in range(H):
                sc.append(-(q[:,h].unsqueeze(2)-k[:,h].unsqueeze(1)).abs().sum(-1))  # B,T,T
            score = torch.stack(sc,1)                                 # B,H,T,T
        else:
            score = (q@k.transpose(-2,-1))/math.sqrt(DH)             # dense dot-product
        score = score.masked_fill(s.cmask[:T,:T], float("-inf"))
        if s.trop and hard:   # mult-free deploy: argmax-GATHER (copy winning position's whole V vector)
            j = score.argmax(-1)                                     # B,H,T
            y = torch.gather(v, 2, j.unsqueeze(-1).expand(-1,-1,-1,DH))
        elif s.trop:          # tropical warm-train: tau-annealed soft blend (the one train-time multiply)
            w = (score/tau).softmax(-1); y = w@v
        else:                 # dense softmax attention (tau does NOT apply — score already /sqrt(DH))
            w = score.softmax(-1); y = w@v
        y = y.transpose(1,2).reshape(B,T,D)
        return s.o(y)
    def forward(s, x, tau, hard):
        x = x + s.attn(x, tau, hard)
        x = x + s.dn(F.relu(s.up(s.ln2(x))))
        return x

class LM(nn.Module):
    def __init__(s, V, D=128, H=4, L=3, T=128, tropical=False):
        super().__init__(); s.trop=tropical
        Lin = DLin if tropical else FLin
        s.emb=nn.Embedding(V,D); s.pos=nn.Parameter(torch.zeros(1,T,D)); s.lnf=nn.LayerNorm(D)
        s.blocks=nn.ModuleList([Block(D,H,T,tropical) for _ in range(L)])
        s.head=Lin(D,V)
    def forward(s, x, tau=1.0, hard=False):
        h=s.emb(x)+s.pos[:,:x.shape[1]]
        for b in s.blocks: h=b(h, tau, hard)
        return s.head(s.lnf(h))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--D",type=int,default=128); ap.add_argument("--H",type=int,default=4)
    ap.add_argument("--L",type=int,default=3); ap.add_argument("--T",type=int,default=128)
    ap.add_argument("--steps",type=int,default=5000); ap.add_argument("--bs",type=int,default=64)
    args=ap.parse_args()
    torch.manual_seed(0)
    tr,va,V=load()
    from collections import Counter
    cN=len(va); bg=Counter(zip(va[:-1],va[1:])); c0=Counter(va[:-1])
    bgH=-sum((cnt/(cN-1))*math.log2(cnt/c0[a]) for (a,b),cnt in bg.items())
    print(f"V={V} T={args.T} D={args.D} H={args.H} L={args.L} bigram={bgH:.3f}")
    def batch(split):
        d=tr if split=="tr" else va; ix=np.random.randint(0,len(d)-args.T-1,size=args.bs)
        x=torch.from_numpy(np.stack([d[i:i+args.T] for i in ix])).to(dev)
        y=torch.from_numpy(np.stack([d[i+1:i+1+args.T] for i in ix])).to(dev); return x,y
    def bpc(m,hard):
        m.eval(); tot=0.; npos=0
        with torch.no_grad():
            for _ in range(30):
                x,y=batch("va"); lp=F.log_softmax(m(x,tau=0.1,hard=hard),-1)
                tot+=F.nll_loss(lp.reshape(-1,V),y.reshape(-1),reduction="sum").item()/math.log(2); npos+=y.numel()
        m.train(); return tot/npos
    def train(tropical):
        m=LM(V,args.D,args.H,args.L,args.T,tropical).to(dev)
        opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4)
        warm=int(0.05*args.steps)
        def lr_at(i):  # linear warmup -> cosine decay (standard; prevents late transformer divergence)
            if i<warm: return (i+1)/warm
            p=(i-warm)/max(1,args.steps-warm); return 0.5*(1+math.cos(math.pi*p))
        np_=sum(p.numel() for p in m.parameters()); t0=time.time()
        for i in range(args.steps):
            for g in opt.param_groups: g["lr"]=3e-3*lr_at(i)
            # tau anneal 2.0 -> 0.2 over the first 70% (tropical only; dense ignores tau)
            tau=max(0.2, 2.0*(1-i/(0.7*args.steps))) if tropical else 1.0
            x,y=batch("tr"); loss=F.cross_entropy(m(x,tau=tau,hard=False).reshape(-1,V),y.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        soft=bpc(m,hard=False); hard=bpc(m,hard=True) if tropical else soft
        print(f"  {'tropical' if tropical else 'dense':9s}: soft_bpc={soft:.4f}"
              + (f"  HARD_deploy_bpc={hard:.4f}" if tropical else "")
              + f"  ({np_/1e3:.0f}k params, {time.time()-t0:.0f}s)")
        return soft,hard
    print(f"=== char-LM, {args.steps} steps (bigram {bgH:.3f}) ===")
    ds,_=train(False)
    ts,th=train(True)
    print(f"\nRESULT: dense {ds:.3f} | tropical soft {ts:.3f} | tropical HARD(mult-free) {th:.3f}"
          f" | gap(hard-dense) {th-ds:+.3f}")

if __name__=="__main__": main()
