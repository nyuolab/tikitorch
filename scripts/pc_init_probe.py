"""Fast probe: which init lets a DEEP distributed dyadic net (forge_pc_block recipe, MSE+SignSGD) train without
the depth collapse? The project's finding: identity-biased init solves deep COLLAPSE. Reduced-data, 1-epoch.
  PYTHONPATH=. python scripts/pc_init_probe.py [NSAMP]
"""
import sys, time, torch
sys.path.insert(0, ".")
from scripts.pc_shard_train_ref import step_L, accuracy_L, D, C
from scripts.pc_shard_screen_init import load_bins

NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
Xtr, Ytr, Xte, Yte = load_bins("/tmp/pc_fashion")
Xtr, Ytr = Xtr[:NSAMP], Ytr[:NSAMP]
N = 32


def mk_init(L, mode, diag, sigma_off, seed):
    g = torch.Generator().manual_seed(seed)
    Ws = []
    for l in range(L):
        if l == L - 1 or mode == "random":               # output layer (or all-random baseline)
            W = (torch.randn(D, D, generator=g) * 4).round().double()
        else:                                            # hidden: identity-biased (diag dominant + small noise)
            W = (torch.randn(D, D, generator=g) * sigma_off).round().double()
            W.fill_diagonal_(float(diag))
        Ws.append(W)
    return Ws


def run(L, mode, diag, sigma_off, seed, epochs=1):
    Ws = mk_init(L, mode, diag, sigma_off, seed)
    ms = [torch.zeros(D, D, dtype=torch.float64) for _ in range(L)]
    g = torch.Generator().manual_seed(1000 + seed)
    perm = torch.randperm(Xtr.shape[0], generator=g)
    nb = Xtr.shape[0] // N
    Ws0 = [w.clone() for w in Ws]
    for ep in range(epochs):
        for b in range(nb):
            idx = perm[b*N:(b+1)*N]
            Ws, ms, _ = step_L(Xtr[idx], Ytr[idx], Ws, ms)
    acc = accuracy_L(Xte, Yte, Ws)
    chg = [int((Ws[l] != Ws0[l]).sum()) for l in range(L)]   # did each chip's weights move?
    return acc, chg


for (L, mode, diag, sig) in [(8,"random",0,0), (8,"identity",1,1), (8,"identity",4,2),
                             (8,"identity",2,2), (4,"random",0,0), (4,"identity",2,2)]:
    t = time.time()
    acc, chg = run(L, mode, diag, sig, seed=0)
    tag = f"L={L} {mode} diag={diag} sig={sig}"
    print(f"{tag:34s} acc={acc:.4f}  W-changed/chip={chg}  ({time.time()-t:.0f}s)", flush=True)
