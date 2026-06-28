"""torch_check.py — cross-check the C++ BF16 STE backward vs TORCH AUTOGRAD (ground truth).

Rebuilds bf_forward (the char-LM-faithful wide-float forward) in torch from the dumped EMB/POS + deployed
dyadic weights + LN gamma/beta, runs autograd of CE, and compares the autograd grads to the C++ bf_backward
grads. Everything float64. qact = STE (forward rounds to the SAME pow2 grid, backward = identity). The
deployed dyadic weights are CONSTANTS w.r.t the float master, but we differentiate CE w.r.t. them to get
dW (== the master grad under STE: grad flows through the deployed value). The forward is a clean recompute
(no cached-value bridges) so torch IS the unambiguous ground truth.
"""
import sys, math, torch

D = {}
for line in open("/tmp/bf16bwd/dump.txt"):
    p = line.split()
    if not p: continue
    k = p[0]
    if k == "DIMS": D["DIMS"] = list(map(int, p[1:])); continue
    if k in ("TOK","TGT"): D[k] = list(map(int, p[2:])); continue
    n = int(p[1]); D[k] = torch.tensor(list(map(float, p[2:2+n])), dtype=torch.float64)

Dm,H,DH,FFN,SEQ,VOCAB,ABITS = D["DIMS"]
tok = torch.tensor(D["TOK"]); tgt = torch.tensor(D["TGT"], dtype=torch.long)
def mat(name,r,c): return D[name].view(r,c)

WQ=mat("WQd",Dm,Dm).clone().requires_grad_(True); WK=mat("WKd",Dm,Dm).clone().requires_grad_(True)
WV=mat("WVd",Dm,Dm).clone().requires_grad_(True); WO=mat("WOd",Dm,Dm).clone().requires_grad_(True)
WUP=mat("WUPd",FFN,Dm).clone().requires_grad_(True); WDN=mat("WDNd",Dm,FFN).clone().requires_grad_(True)
WH=mat("WHEADd",VOCAB,Dm).clone().requires_grad_(True)
LN1G=D["LN1G"]; LN1B=D["LN1B"]; LN2G=D["LN2G"]; LN2B=D["LN2B"]; LNFG=D["LNFG"]; LNFB=D["LNFB"]
EMB=mat("EMB",VOCAB,Dm); POS=mat("POS",SEQ,Dm)

class QAct(torch.autograd.Function):       # char-LM qact: round to pow2 grid, keep magnitude; STE identity bwd
    @staticmethod
    def forward(ctx,x):
        m=x.abs().max().clamp_min(1e-9); sh=torch.floor(torch.log2(m))-ABITS
        return torch.round(x/2.0**sh)*2.0**sh
    @staticmethod
    def backward(ctx,g): return g
def qact(x): return QAct.apply(x)

def fdyad_ln(x, gamma, beta):              # char-LM FDyadLN (float), gamma/beta constants here
    mean=x.mean(-1,keepdim=True); c=x-mean
    scale=c.abs().mean(-1,keepdim=True).clamp_min(1e-9)
    e=torch.floor(torch.log2(scale))
    MBITS=2; mant=torch.round((scale/2.0**e-1.0)*(2**MBITS))/(2**MBITS)
    return c/((1.0+mant)*2.0**e)*gamma+beta

# block input R = EMB[tok] + POS  (leaf for gR)
R = (EMB[tok] + POS).clone().requires_grad_(True)
X1 = fdyad_ln(R, LN1G, LN1B)
Q = qact(X1 @ WQ.t()); K = qact(X1 @ WK.t()); V = qact(X1 @ WV.t())
Q.retain_grad(); K.retain_grad(); V.retain_grad()
scale = 1.0/math.sqrt(DH)
ATTN = torch.zeros(SEQ, Dm, dtype=torch.float64)
parts=[]
for h in range(H):
    qs=Q[:,h*DH:(h+1)*DH]; ks=K[:,h*DH:(h+1)*DH]; vs=V[:,h*DH:(h+1)*DH]
    s=(qs@ks.t())*scale
    mask=torch.tril(torch.ones(SEQ,SEQ,dtype=torch.bool))
    s=s.masked_fill(~mask,float("-inf"))
    Pr=torch.softmax(s,dim=-1)
    parts.append(Pr@vs)
ATTN=torch.cat(parts,dim=-1)
ATTN.retain_grad()
ATTNq = qact(ATTN)                          # char-LM proj(qact(y))
AO = ATTNq @ WO.t()
M = R + AO
X2 = fdyad_ln(M, LN2G, LN2B)
HD = qact(torch.relu(X2 @ WUP.t()))         # fc2(qact(relu(fc1)))
FO = HD @ WDN.t()
RP = M + FO
XF = fdyad_ln(RP, LNFG, LNFB)
LG = XF @ WH.t()
loss = torch.nn.functional.cross_entropy(LG, tgt, reduction="mean")
loss.backward()

# fwd-fidelity: torch LG vs C++ LG (must match -> proves the twin forward == bf_forward)
lg_cpp = D["LG"].view(SEQ,VOCAB)
fwd_relerr = ((LG.detach()-lg_cpp).norm()/(lg_cpp.norm()+1e-12)).item()

def cmp(name, autograd, cpp_key):
    a=autograd.reshape(-1).double(); c=D[cpp_key].double()
    thr=1e-9; nz=(a.abs()>thr)&(c.abs()>thr)
    sign=(torch.sign(a[nz])==torch.sign(c[nz])).double().mean().item() if nz.sum()>0 else float("nan")
    cos=torch.nn.functional.cosine_similarity(a.unsqueeze(0),c.unsqueeze(0)).item()
    rel=((a-c).norm()/(a.norm()+1e-12)).item()
    print(f"  {name:7s} sign_agree={sign:.4f} cos={cos:.4f} relL2={rel:.4f}  |ag|={a.norm():.4g} |cpp|={c.norm():.4g} nz={int(nz.sum())}")
    return sign,cos

print(f"FWD fidelity: LG relerr(torch vs C++)={fwd_relerr:.3e}  loss={loss.item():.4f}  chance={math.log(VOCAB):.4f}")
print("torch-autograd vs C++ bf_backward:")
ss=[]
for nm,ag,ck in [("WHEAD",WH.grad,"gWHEAD"),("WDN",WDN.grad,"gWDN"),("WUP",WUP.grad,"gWUP"),
                 ("WO",WO.grad,"gWO"),("WV",WV.grad,"gWV"),("WK",WK.grad,"gWK"),("WQ",WQ.grad,"gWQ"),
                 ("gR",R.grad,"gR")]:
    s,c=cmp(nm,ag,ck); ss.append(s)
import statistics
print(f"  MEAN sign_agree over all params = {statistics.mean([x for x in ss if x==x]):.4f}")

# intermediate attention-grad localization (dump these from C++ too via gQi/gVi/gATTNi if present)
import os
for nm,grad,ck in [("gATTN",ATTN.grad,"gATTNi"),("gV",V.grad,"gVi"),("gQ",Q.grad,"gQi"),("gK",K.grad,"gKi")]:
    if ck in D:
        cmp(nm,grad,ck)

