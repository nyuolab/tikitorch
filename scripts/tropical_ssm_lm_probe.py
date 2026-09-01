"""Can a multiplier-free SSM match a transformer (char-LM), and how much can it be pushed? Compare at matched
size on tinyshakespeare-char: (1) dense transformer (softmax attn = the reference), (2) dense gated SSM
(Mamba-lite linear recurrence, no softmax), (3) MULT-FREE tropical SSM (dyadic recurrence + dyadic GEMMs +
dyadic head, the morph-LM substrate), (4) mult-free tropical SSM + SELECTIVITY (input-dependent gate, Mamba's
key trick, made mult-free). Reports val bits/char + the tropical-vs-dense gap + whether selectivity closes it.
"""
import math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
dev="mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(0)
txt=open("data/tinyshakespeare.txt").read().lower(); chars=sorted(set(txt)); V=len(chars); stoi={c:i for i,c in enumerate(chars)}
ids=np.array([stoi[c] for c in txt],dtype=np.int64); n=len(ids); tr,va=ids[:int(.9*n)],ids[int(.9*n):]
L=128
from collections import Counter
c=Counter(va);N=len(va);bg=Counter(zip(va[:-1],va[1:]));c0=Counter(va[:-1]);bgH=-sum((cnt/(N-1))*math.log2(cnt/c0[a]) for (a,b),cnt in bg.items());print(f"V={V} bigram={bgH:.3f}")
def batch(s,bs=64):
    d=tr if s=="tr" else va; ix=np.random.randint(0,len(d)-L-1,size=bs)
    return (torch.from_numpy(np.stack([d[i:i+L] for i in ix])).to(dev),torch.from_numpy(np.stack([d[i+1:i+1+L] for i in ix])).to(dev))
def dyad(w): s=torch.sign(w);a=w.abs().clamp_min(1e-9);k=torch.round(torch.log2(a));return w+(s*torch.pow(2.,k)-w).detach()
class DLin(nn.Module):  # dyadic (mult-free) linear via STE
    def __init__(s,i,o): super().__init__(); s.w=nn.Parameter(torch.randn(o,i)*(1/math.sqrt(i)))
    def forward(s,x): return x@dyad(s.w).t()
class FLin(nn.Module):  # float linear
    def __init__(s,i,o): super().__init__(); s.l=nn.Linear(i,o,bias=False)
    def forward(s,x): return s.l(x)

def lin(dy,i,o): return DLin(i,o) if dy else FLin(i,o)

class XfmrBlock(nn.Module):
    def __init__(s,D,H):
        super().__init__(); s.H=H; s.DH=D//H; s.ln1=nn.LayerNorm(D); s.ln2=nn.LayerNorm(D)
        s.q=nn.Linear(D,D,bias=False);s.k=nn.Linear(D,D,bias=False);s.v=nn.Linear(D,D,bias=False);s.o=nn.Linear(D,D,bias=False)
        s.up=nn.Linear(D,4*D);s.dn=nn.Linear(4*D,D)
        m=torch.tril(torch.ones(L,L)); s.register_buffer("m",m==0)
    def forward(s,x):
        B,T,D=x.shape; z=s.ln1(x)
        q=s.q(z).view(B,T,s.H,s.DH).transpose(1,2);k=s.k(z).view(B,T,s.H,s.DH).transpose(1,2);v=s.v(z).view(B,T,s.H,s.DH).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(s.DH); a=a.masked_fill(s.m[:T,:T],float("-inf")).softmax(-1)
        x=x+s.o((a@v).transpose(1,2).reshape(B,T,D)); return x+s.dn(F.relu(s.up(s.ln2(x))))

class SSMBlock(nn.Module):
    """gated linear-recurrence time mixer (Mamba-lite): h_t = a*h_{t-1} + b*x_t (per-channel a in (0,1)),
    optional input-dependent (selective) a,b; then a channel MLP. dy=True -> dyadic mult-free weights."""
    def __init__(s,D,dy,selective):
        super().__init__(); s.D=D; s.dy=dy; s.sel=selective; s.ln1=nn.LayerNorm(D); s.ln2=nn.LayerNorm(D)
        s.a_logit=nn.Parameter(torch.zeros(D))            # base per-channel decay
        s.bproj=lin(dy,D,D); s.gate=lin(dy,D,D); s.out=lin(dy,D,D)
        s.up=lin(dy,D,4*D); s.dn=lin(dy,4*D,D)
        if selective: s.a_sel=lin(dy,D,D)                 # input-dependent decay (selectivity)
    def scan(s,a,bx):  # h_t = a_t*h_{t-1}+bx_t  (sequential causal scan), a:(B,T,D) or (D,), bx:(B,T,D)
        B,T,D=bx.shape; h=torch.zeros(B,D,device=bx.device); ys=[]
        for t in range(T):
            at = a[:,t] if a.dim()==3 else a
            h = at*h + bx[:,t]; ys.append(h)
        return torch.stack(ys,1)
    def forward(s,x):
        z=s.ln1(x); b=s.bproj(z)
        if s.sel: a=torch.sigmoid(s.a_logit + s.a_sel(z))      # selective per-(t,channel) decay
        else:     a=torch.sigmoid(s.a_logit)
        h=s.scan(a,b)
        y=s.out(h*torch.sigmoid(s.gate(z)))                    # gated output
        x=x+y; return x+s.dn(F.relu(s.up(s.ln2(x))))

class LM(nn.Module):
    def __init__(s,kind,D=128,nl=2,H=4):
        super().__init__(); s.kind=kind; dy=("trop" in kind)
        s.emb=nn.Embedding(V,D); s.pos=nn.Parameter(torch.zeros(1,L,D)); s.lnf=nn.LayerNorm(D)
        if kind=="xfmr":          s.blocks=nn.ModuleList([XfmrBlock(D,H) for _ in range(nl)])
        else:                     s.blocks=nn.ModuleList([SSMBlock(D,dy,("sel" in kind)) for _ in range(nl)])
        s.head=lin(dy,D,V) if dy else nn.Linear(D,V,bias=False)
    def forward(s,x):
        h=s.emb(x)+s.pos[:,:x.shape[1]]
        for blk in s.blocks: h=blk(h)
        return s.head(s.lnf(h))
def run(kind,steps=2500,D=128,nl=2):
    m=LM(kind,D,nl).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
    nparam=sum(p.numel() for p in m.parameters()); t0=time.time()
    for i in range(steps):
        x,y=batch("tr");loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1));opt.zero_grad();loss.backward();opt.step()
    m.eval();tot=0.;np_=0
    with torch.no_grad():
        for _ in range(30):
            x,y=batch("va");lp=F.log_softmax(m(x),-1);tot+=F.nll_loss(lp.reshape(-1,V),y.reshape(-1),reduction="sum").item()/math.log(2);np_+=y.numel()
    print(f"  {kind:22s} val_bpc={tot/np_:.4f}  ({nparam/1e3:.0f}k params, {time.time()-t0:.0f}s)")
    return tot/np_
print("\n=== char-LM @ D=128, 2 layers, 2500 steps (bigram bar 3.47) ===")
r={}
for kind in ["xfmr","ssm","trop_ssm","trop_ssm_sel"]:
    r[kind]=run(kind)
print(f"\nGAPS: dense SSM vs xfmr = {r['ssm']-r['xfmr']:+.3f} | tropical SSM vs dense SSM = {r['trop_ssm']-r['ssm']:+.3f} | selectivity lift = {r['trop_ssm']-r['trop_ssm_sel']:+.3f}")
print(f"tropical-sel vs dense xfmr = {r['trop_ssm_sel']-r['xfmr']:+.3f}")
