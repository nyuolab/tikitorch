"""Build + train + GENERATE with the fully multiplier-free TROPICAL LLM (the validated recipe).

Model: a tropical transformer — max-plus multi-head argmax-GATHER attention (mult-free) + dyadic ±2^k linears
(Q/K/V/O, FFN, embed, head) + positional embeddings. Trained τ-annealed (soft warm-train, the one train-time
multiply) → DEPLOYED HARD (argmax-gather), fully multiplier-free (DSP=0 datapath). This script trains a solid
char-LM, saves a checkpoint, GENERATES text with the mult-free hard deploy (proving it's a real working LLM),
and reports held-out bits/char.

Usage:  python scripts/tropical_llm.py --D 192 --L 4 --T 128 --steps 9000
"""
import math, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

dev = "mps" if torch.backends.mps.is_available() else "cpu"
REPO = Path(__file__).resolve().parents[1]

def load_corpus():
    txt = open(REPO/"data"/"tinyshakespeare.txt").read().lower()
    chars = sorted(set(txt)); stoi={c:i for i,c in enumerate(chars)}; itos={i:c for c,i in stoi.items()}
    ids = np.array([stoi[c] for c in txt], dtype=np.int64); n=len(ids)
    return ids[:int(.9*n)], ids[int(.9*n):], stoi, itos

def dyad(w):  # ±2^k via STE — the mult-free weight
    s=torch.sign(w); a=w.abs().clamp_min(1e-9); k=torch.round(torch.log2(a)); return w+(s*torch.pow(2.,k)-w).detach()
class DLin(nn.Module):
    def __init__(s,i,o): super().__init__(); s.w=nn.Parameter(torch.randn(o,i)*(1/math.sqrt(i)))
    def forward(s,x): return x@dyad(s.w).t()

class Block(nn.Module):
    def __init__(s,D,H,T):
        super().__init__(); s.H=H; s.DH=D//H
        s.ln1=nn.LayerNorm(D); s.ln2=nn.LayerNorm(D)
        s.q=DLin(D,D); s.k=DLin(D,D); s.v=DLin(D,D); s.o=DLin(D,D); s.up=DLin(D,4*D); s.dn=DLin(4*D,D)
        s.register_buffer("cmask", torch.triu(torch.ones(T,T),1).bool())
    def attn(s,x,tau,hard):
        B,T,D=x.shape; H,DH=s.H,s.DH
        q=s.q(s.ln1(x)).view(B,T,H,DH).transpose(1,2); k=s.k(s.ln1(x)).view(B,T,H,DH).transpose(1,2); v=s.v(s.ln1(x)).view(B,T,H,DH).transpose(1,2)
        sc=[ -(q[:,h].unsqueeze(2)-k[:,h].unsqueeze(1)).abs().sum(-1) for h in range(H) ]  # -L1, mult-free
        score=torch.stack(sc,1).masked_fill(s.cmask[:T,:T], float("-inf"))
        if hard:  # mult-free deploy: argmax-GATHER
            j=score.argmax(-1); y=torch.gather(v,2,j.unsqueeze(-1).expand(-1,-1,-1,DH))
        else:     # warm-train soft blend (the one train-time multiply)
            y=(score/tau).softmax(-1)@v
        return s.o(y.transpose(1,2).reshape(B,T,D))
    def forward(s,x,tau,hard):
        x=x+s.attn(x,tau,hard); return x+s.dn(F.relu(s.up(s.ln2(x))))

class TropicalLLM(nn.Module):
    def __init__(s,V,D=192,H=6,L=4,T=128):
        super().__init__(); s.T=T
        s.emb=nn.Embedding(V,D); s.pos=nn.Parameter(torch.zeros(1,T,D)); s.lnf=nn.LayerNorm(D)
        s.blocks=nn.ModuleList([Block(D,H,T) for _ in range(L)]); s.head=DLin(D,V)
    def forward(s,x,tau=1.0,hard=False):
        h=s.emb(x)+s.pos[:,:x.shape[1]]
        for b in s.blocks: h=b(h,tau,hard)
        return s.head(s.lnf(h))
    @torch.no_grad()
    def generate(s,prompt_ids,n,itos,temp=0.8):
        s.eval(); ids=list(prompt_ids)
        for _ in range(n):
            ctx=torch.tensor([ids[-s.T:]],device=next(s.parameters()).device)
            logits=s(ctx,hard=True)[0,-1]/temp      # HARD = fully multiplier-free deploy
            p=torch.softmax(logits,-1); nxt=int(torch.multinomial(p,1))
            ids.append(nxt)
        return "".join(itos[i] for i in ids)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--D",type=int,default=192); ap.add_argument("--H",type=int,default=6)
    ap.add_argument("--L",type=int,default=4); ap.add_argument("--T",type=int,default=128)
    ap.add_argument("--taufloor",type=float,default=0.05)
    ap.add_argument("--steps",type=int,default=9000); ap.add_argument("--bs",type=int,default=48)
    ap.add_argument("--out",type=str,default=str(REPO/"results"/"lm_shakespeare"/"tropical_llm.pt"))
    a=ap.parse_args(); torch.manual_seed(0)
    tr,va,stoi,itos=load_corpus(); V=len(stoi)
    from collections import Counter
    cN=len(va); bg=Counter(zip(va[:-1],va[1:])); c0=Counter(va[:-1])
    bgH=-sum((cnt/(cN-1))*math.log2(cnt/c0[x]) for (x,y),cnt in bg.items())
    print(f"V={V} D={a.D} H={a.H} L={a.L} T={a.T} steps={a.steps} | bigram={bgH:.3f}")
    def batch(d):
        ix=np.random.randint(0,len(d)-a.T-1,size=a.bs)
        x=torch.from_numpy(np.stack([d[i:i+a.T] for i in ix])).to(dev); y=torch.from_numpy(np.stack([d[i+1:i+1+a.T] for i in ix])).to(dev); return x,y
    def bpc(m,hard):
        m.eval(); t=0.;n=0
        with torch.no_grad():
            for _ in range(30):
                x,y=batch(va); lp=F.log_softmax(m(x,tau=a.taufloor,hard=hard),-1); t+=F.nll_loss(lp.reshape(-1,V),y.reshape(-1),reduction="sum").item()/math.log(2); n+=y.numel()
        m.train(); return t/n
    m=TropicalLLM(V,a.D,a.H,a.L,a.T).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4); warm=int(0.05*a.steps)
    nparam=sum(p.numel() for p in m.parameters()); t0=time.time()
    for i in range(a.steps):
        lr=3e-3*((i+1)/warm if i<warm else 0.5*(1+math.cos(math.pi*(i-warm)/(a.steps-warm))))
        for g in opt.param_groups: g["lr"]=lr
        tau=max(a.taufloor,2.0*(1-i/(0.7*a.steps)))
        x,y=batch(tr); loss=F.cross_entropy(m(x,tau=tau,hard=False).reshape(-1,V),y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if (i+1)%2000==0: print(f"  step {i+1}/{a.steps}: train_loss={loss.item()/math.log(2):.3f} bits  lr={lr:.1e}  ({time.time()-t0:.0f}s)")
    soft=bpc(m,False); hard=bpc(m,True)
    print(f"\n[RESULT] {nparam/1e3:.0f}k params | soft_bpc={soft:.4f} | HARD(mult-free)_bpc={hard:.4f} | bigram={bgH:.3f}")
    torch.save({"model":m.state_dict(),"cfg":vars(a),"stoi":stoi}, a.out); print(f"[saved] {a.out}")
    print("\n[GENERATION — fully multiplier-free hard deploy, prompt='the ']")
    print(m.generate([stoi.get(c,0) for c in "the "], 400, itos, temp=0.8))

if __name__=="__main__": main()
