"""Offline eval + bit-exact check for the 2-INSTANCE cross-box run (forge_pc_train_shard_net.c). The head dumps
netW0..W{L-1}.i32, the tail dumps netW{L}..W{2L-1}.i32 — together ONE 2L-layer residual net trained across two
physically separate f2 boxes. This loads all 2L weight matrices, scores test accuracy (the combined net), and
replays the Python reference from the verified init+order to confirm the cross-box training is bit-exact.

  PYTHONPATH=. python scripts/pc_shard_net_eval.py PC_BINS=/tmp/pc_fashion PC_GLOBAL_L=16 \
      PC_WDIR=results/predictive_coding/shard_net PC_INITPERM=/tmp/ip16 PC_EPOCHS=4
"""
import os, sys, numpy as np, torch
sys.path.insert(0, ".")
from scripts.pc_shard_train_ref import step_L, accuracy_L, D
from scripts.pc_shard_screen_init import load_bins

ENV = dict(os.environ)
for a in sys.argv[1:]:
    if "=" in a: k, v = a.split("=", 1); ENV[k] = v
GL    = int(ENV.get("PC_GLOBAL_L", 16))
BINS  = ENV.get("PC_BINS", "/tmp/pc_fashion")
WDIR  = ENV.get("PC_WDIR", "results/predictive_coding/shard_net")
IP    = ENV.get("PC_INITPERM", "")
EPOCHS= int(ENV.get("PC_EPOCHS", 4))
N = 32

def ld(p): return torch.from_numpy(np.fromfile(p, dtype=np.int32).astype(np.int64)).double().reshape(D, D)

Xtr, Ytr, Xte, Yte = load_bins(BINS)
W = [ld(f"{WDIR}/netW{l}.i32") for l in range(GL)]
print(f"cross-box net: {GL} layers across 2 instances -> test acc = {accuracy_L(Xte, Yte, W, res=True):.4f}", flush=True)

if IP:
    Ws = [ld(f"{IP}/W{l}_init.i32") for l in range(GL)]
    ms = [torch.zeros(D, D, dtype=torch.float64) for _ in range(GL)]
    perm = torch.from_numpy(np.fromfile(f"{IP}/perm.i32", dtype=np.int32).astype(np.int64))
    nb = Xtr.shape[0] // N
    for ep in range(EPOCHS):
        for b in range(nb):
            Ws, ms, _ = step_L(Xtr[perm[b*N:(b+1)*N]], Ytr[perm[b*N:(b+1)*N]], Ws, ms, res=True)
    d = max(int((Ws[l] - W[l]).abs().max()) for l in range(GL))
    print(f"bit-exact replay (single-process {GL}-layer ref vs 2-instance silicon): max|diff|={d}  "
          f"{'BIT-EXACT across two boxes' if d == 0 else 'DIVERGES'}", flush=True)
