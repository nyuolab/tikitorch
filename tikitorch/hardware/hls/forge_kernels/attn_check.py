import math, torch
D={}
for l in open("/tmp/bf16bwd/attn_dump.txt"):
    p=l.split()
    if not p: continue
    if p[0]=="DIMS": D["DIMS"]=list(map(int,p[1:])); continue
    n=int(p[1]); D[p[0]]=torch.tensor(list(map(float,p[2:2+n])),dtype=torch.float64)
Dm,H,DH,SEQ=D["DIMS"]
Q=D["Q"].view(SEQ,Dm).clone().requires_grad_(True)
K=D["K"].view(SEQ,Dm).clone().requires_grad_(True)
V=D["V"].view(SEQ,Dm).clone().requires_grad_(True)
gATTN=D["gATTN"].view(SEQ,Dm)
scale=1.0/math.sqrt(DH)
parts=[]
mask=torch.tril(torch.ones(SEQ,SEQ,dtype=torch.bool))
for h in range(H):
    qs=Q[:,h*DH:(h+1)*DH]; ks=K[:,h*DH:(h+1)*DH]; vs=V[:,h*DH:(h+1)*DH]
    s=(qs@ks.t())*scale; s=s.masked_fill(~mask,float("-inf"))
    Pr=torch.softmax(s,dim=-1); parts.append(Pr@vs)
ATTN=torch.cat(parts,-1)
(ATTN*gATTN).sum().backward()
def cmp(nm,ag,ck):
    a=ag.reshape(-1); c=D[ck]
    nz=(a.abs()>1e-12)&(c.abs()>1e-12)
    sign=(torch.sign(a[nz])==torch.sign(c[nz])).double().mean().item()
    cos=torch.nn.functional.cosine_similarity(a.unsqueeze(0),c.unsqueeze(0)).item()
    rel=((a-c).norm()/a.norm()).item()
    print(f"  {nm}: sign_agree={sign:.4f} cos={cos:.6f} relL2={rel:.2e} |ag|={a.norm():.4g} nz={int(nz.sum())}")
print("attention backward vs torch autograd (SOFT attention):")
cmp("gQ",Q.grad,"gQ"); cmp("gK",K.grad,"gK"); cmp("gV",V.grad,"gV")
