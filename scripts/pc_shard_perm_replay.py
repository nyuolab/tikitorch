"""Replay the FULL silicon epoch in Python: silicon init + silicon's exact shuffle order. If Python then
also collapses to ~0.098, the data ORDER causes it (and we check the perm is a valid permutation). If Python
trains, there is a non-per-step / non-order silicon bug."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from scripts.pc_shard_train_ref import step, accuracy, D, C, load_mnist_16

def ld(p, shape):
    return torch.from_numpy(np.fromfile(p, dtype=np.int32).astype(np.int64)).double().reshape(shape)

perm = np.fromfile("/tmp/s1/s1_perm.i32", dtype=np.int32)
print(f"perm: len={len(perm)}, unique={len(np.unique(perm))}, min={perm.min()}, max={perm.max()}")
u, c = np.unique(perm, return_counts=True)
print(f"  valid permutation? {'YES' if len(u)==len(perm) and perm.min()==0 and perm.max()==len(perm)-1 else 'NO -- DEGENERATE'}"
      f"  (max repeat count={c.max()}, #missing={len(perm)-len(u)})")

iW0 = ld("/tmp/s1/s1_iW0.i32", (D, D)); iW1 = ld("/tmp/s1/s1_iW1.i32", (D, D))
(Xtr, Ytr), (Xte, Yte) = load_mnist_16(); Xtr = Xtr.double(); Xte = Xte.double()
W0, W1 = iW0.clone(), iW1.clone()
m0 = torch.zeros(D, D, dtype=torch.float64); m1 = torch.zeros(D, D, dtype=torch.float64)
N = 32; nb = len(perm) // N
pt = torch.from_numpy(perm.astype(np.int64))
for b in range(nb):
    idx = pt[b*N:(b+1)*N]
    W0, W1, m0, m1, _ = step(Xtr[idx], Ytr[idx], W0, W1, m0, m1)
acc = accuracy(Xte, Yte, W0, W1)
print(f"PYTHON with SILICON order+init: epoch-1 acc={acc:.4f}  |W0|max={int(W0.abs().max())}")
# compare to silicon's dumped final weights if present
import os
if os.path.exists("/tmp/s1/shardW0.i32"):
    sW0 = ld("/tmp/s1/shardW0.i32", (D, D)); sW1 = ld("/tmp/s1/shardW1.i32", (D, D))
    print(f"  vs SILICON final weights: W0 max|diff|={int((W0-sW0).abs().max())}  W1 max|diff|={int((W1-sW1).abs().max())}")
print("  -> " + ("ORDER REPRODUCES COLLAPSE (perm is the lever; check degeneracy above)" if acc < 0.2
                 else "Python trains on the silicon order -> the bug is NOT per-step and NOT order (deeper)"))
