"""Kernel-FAITHFUL tropical LLM: does training WITH the s1 kernel's ops recover a useful bpc?
The deploy gate showed the std-LN-trained model deploys at 4.05 (doesn't beat bigram) through the s1 kernel
because the kernel uses MEAN-ABS-DEVIATION LayerNorm + pow2 gamma (not std-LN + float gamma). Fix = train
faithful. This trains from scratch with MAD-LN + pow2 gamma (the dominant mismatch factors) + dyadic weights +
argmax-gather attention + int16 activations, and reports HARD (mult-free) bpc vs bigram.
"""
import math, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
dev="mps" if torch.backends.mps.is_available() else "cpu"; REPO=Path(__file__).resolve().parents[1]
def load():
    t=open(REPO/"data"/"tinyshakespeare.txt").read().lower(); ch=sorted(set(t)); st={c:i for i,c in enumerate(ch)}
    ids=np.array([st[c] for c in t],dtype=np.int64); n=len(ids); return ids[:int(.9*n)],ids[int(.9*n):],len(ch)
def dyad(w): s=torch.sign(w);a=w.abs().clamp_min(1e-9);k=torch.round(torch.log2(a));return w+(s*torch.pow(2.,k)-w).detach()
AQ=0.0
def fq(x):
    if AQ<=0: return x
    return x+(torch.clamp(torch.round(x*AQ),-32768,32767)/AQ - x).detach()
QX_POW2=False
class DLin(nn.Module):
    def __init__(s,i,o): super().__init__(); s.w=nn.Parameter(torch.randn(o,i)*(1/math.sqrt(i)))
    def forward(s,x): return (dyad(x) if QX_POW2 else x)@dyad(s.w).t()   # QX_POW2 -> q(x)*q(W) (the kernel dyadic-MAC)
class MADLN(nn.Module):   # kernel-faithful: normalize by mean-ABS-deviation; gamma snapped to pow2 (STE)
    def __init__(s,D): super().__init__(); s.g=nn.Parameter(torch.ones(D)); s.b=nn.Parameter(torch.zeros(D))
    def forward(s,x):
        mu=x.mean(-1,keepdim=True); c=x-mu; mad=c.abs().mean(-1,keepdim=True).clamp_min(1e-5)
        return c/mad*dyad(s.g)+s.b
def mk_ln(D,madln): return MADLN(D) if madln else nn.LayerNorm(D)
class Block(nn.Module):
    def __init__(s,D,H,T,madln):
        super().__init__(); s.H=H; s.DH=D//H; s.ln1=mk_ln(D,madln); s.ln2=mk_ln(D,madln)
        s.q=DLin(D,D);s.k=DLin(D,D);s.v=DLin(D,D);s.o=DLin(D,D);s.up=DLin(D,4*D);s.dn=DLin(4*D,D)
        s.register_buffer("m",torch.triu(torch.ones(T,T),1).bool())
    def attn(s,x,tau,hard):
        B,T,D=x.shape;H,DH=s.H,s.DH; z=fq(s.ln1(x))
        q=fq(s.q(z)).view(B,T,H,DH).transpose(1,2);k=fq(s.k(z)).view(B,T,H,DH).transpose(1,2);v=fq(s.v(z)).view(B,T,H,DH).transpose(1,2)
        sc=[-(q[:,h].unsqueeze(2)-k[:,h].unsqueeze(1)).abs().sum(-1) for h in range(H)]
        score=torch.stack(sc,1).masked_fill(s.m[:T,:T],float("-inf"))
        if hard: j=score.argmax(-1); y=torch.gather(v,2,j.unsqueeze(-1).expand(-1,-1,-1,DH))
        else: y=(score/tau).softmax(-1)@v
        return s.o(y.transpose(1,2).reshape(B,T,D))
    def forward(s,x,tau,hard): x=fq(x+s.attn(x,tau,hard)); return fq(x+s.dn(F.relu(fq(s.up(fq(s.ln2(x)))))))
class LM(nn.Module):
    def __init__(s,V,D,H,L,T,madln):
        super().__init__(); s.T=T; s.emb=nn.Embedding(V,D); s.pos=nn.Parameter(torch.zeros(1,T,D))
        s.lnf=mk_ln(D,madln); s.blocks=nn.ModuleList([Block(D,H,T,madln) for _ in range(L)]); s.head=DLin(D,V)
    def forward(s,x,tau=1.0,hard=False):
        h=fq(s.emb(x)+s.pos[:,:x.shape[1]])
        for b in s.blocks: h=b(h,tau,hard)
        return s.head(s.lnf(h))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--D",type=int,default=64); ap.add_argument("--H",type=int,default=2)
    ap.add_argument("--L",type=int,default=2); ap.add_argument("--T",type=int,default=64); ap.add_argument("--steps",type=int,default=7000)
    ap.add_argument("--bs",type=int,default=64); ap.add_argument("--taufloor",type=float,default=0.04)
    ap.add_argument("--madln",type=int,default=1); ap.add_argument("--aq",type=float,default=256.0)
    ap.add_argument("--out",type=str,default=""); ap.add_argument("--qx",type=int,default=0); a=ap.parse_args()
    global AQ,QX_POW2; QX_POW2=bool(a.qx); torch.manual_seed(0); tr,va,V=load()
    from collections import Counter
    cN=len(va); bg=Counter(zip(va[:-1],va[1:])); c0=Counter(va[:-1]); bgH=-sum((c/(cN-1))*math.log2(c/c0[x]) for (x,y),c in bg.items())
    print(f"V={V} D={a.D} L={a.L} T={a.T} madln={a.madln} aq={a.aq} | bigram={bgH:.3f}")
    def batch(d):
        ix=np.random.randint(0,len(d)-a.T-1,size=a.bs)
        return (torch.from_numpy(np.stack([d[i:i+a.T] for i in ix])).to(dev),torch.from_numpy(np.stack([d[i+1:i+1+a.T] for i in ix])).to(dev))
    def bpc(m,hard):
        m.eval(); t=0.;n=0
        with torch.no_grad():
            for _ in range(30):
                x,y=batch(va); lp=F.log_softmax(m(x,tau=a.taufloor,hard=hard),-1); t+=F.nll_loss(lp.reshape(-1,V),y.reshape(-1),reduction="sum").item()/math.log(2); n+=y.numel()
        m.train(); return t/n
    m=LM(V,a.D,a.H,a.L,a.T,a.madln).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4); warm=int(.05*a.steps)
    AQ=a.aq
    for i in range(a.steps):
        for g in opt.param_groups: g["lr"]=3e-3*((i+1)/warm if i<warm else .5*(1+math.cos(math.pi*(i-warm)/(a.steps-warm))))
        tau=max(a.taufloor,2.0*(1-i/(0.7*a.steps)))
        x,y=batch(tr); loss=F.cross_entropy(m(x,tau=tau,hard=False).reshape(-1,V),y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    print(f"[RESULT] madln={a.madln} aq={a.aq}: soft_bpc={bpc(m,False):.4f} HARD_bpc={bpc(m,True):.4f} bigram={bgH:.3f}")
    if a.out: torch.save({"model":m.state_dict(),"cfg":vars(a)}, a.out); print(f"[saved] {a.out}")
if __name__=="__main__": main()
