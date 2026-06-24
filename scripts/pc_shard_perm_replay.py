"""Bit-exact verification of the SIGNATURE 0.91 run. Replays the Python reference for the full 4 epochs with
the EXACT init + data order the silicon loaded (committed in results/predictive_coding/shard_train/sig_*),
and diffs against the silicon's committed final weights. A clean, independent, end-to-end check that the
distributed FPGA training is bit-exact to the reference over all 7500 steps.

Run: PYTHONPATH=. DATA_ROOT=/tmp/_tiki_data python scripts/pc_shard_perm_replay.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, ".")
from scripts.pc_shard_train_ref import step, accuracy, D, C, load_mnist_16

A = "results/predictive_coding/shard_train"
def ld(p): return torch.from_numpy(np.fromfile(p, dtype=np.int32).astype(np.int64)).double().reshape(D, D)

W0 = ld(f"{A}/sig_W0_init.i32"); W1 = ld(f"{A}/sig_W1_init.i32")          # exact init the silicon loaded
perm = torch.from_numpy(np.fromfile(f"{A}/sig_perm.i32", dtype=np.int32).astype(np.int64))  # exact order
u = np.unique(perm.numpy())
print(f"order: len={len(perm)} unique={len(u)} valid_permutation={len(u)==len(perm)}")

(Xtr, Ytr), (Xte, Yte) = load_mnist_16(); Xtr = Xtr.double(); Xte = Xte.double()
N = 32; nb = Xtr.shape[0] // N
m0 = torch.zeros(D, D, dtype=torch.float64); m1 = torch.zeros(D, D, dtype=torch.float64)
EPOCHS = 4
for ep in range(EPOCHS):                       # reuse the verified order each epoch, exactly as the driver does
    for b in range(nb):
        idx = perm[b*N:(b+1)*N]
        W0, W1, m0, m1, _ = step(Xtr[idx], Ytr[idx], W0, W1, m0, m1)

sW0 = ld(f"{A}/sig_W0_final.i32"); sW1 = ld(f"{A}/sig_W1_final.i32")      # silicon's final weights (0.91 run)
d0 = int((W0 - sW0).abs().max()); d1 = int((W1 - sW1).abs().max())
print(f"PYTHON {EPOCHS}-epoch ({EPOCHS*nb} steps) vs SILICON 0.91-run final weights:")
print(f"  W0 max|diff|={d0}  W1 max|diff|={d1}   python acc={accuracy(Xte, Yte, W0, W1):.4f}")
print("  -> " + ("BIT-EXACT: the distributed FPGA training == the reference over all 7500 steps."
                 if d0 == 0 and d1 == 0 else "DIVERGES -- investigate."))
