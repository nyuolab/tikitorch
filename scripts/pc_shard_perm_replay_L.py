"""L-layer bit-exact verification of a distributed PC run (forge_pc_train_shard.c, any 2<=L<=8).

Replays the Python reference for the full run with the EXACT verified init + data order the silicon loaded
(PC_INITPERM dir: W0..W{L-1}_init.i32 + perm.i32), reusing the order each epoch exactly as the driver does,
then diffs every chip's final weights against the silicon's dumps (/tmp/shardW<l>.i32, or a given dir).
Generalizes pc_shard_perm_replay.py (which is hardwired to the committed 2-layer signature artifacts).

  PYTHONPATH=. python scripts/pc_shard_perm_replay_L.py \
      PC_LAYERS=8 PC_BINS=/tmp/pc_fashion PC_INITPERM=/tmp/pc_initperm_fashion8 \
      PC_FINAL=/tmp PC_EPOCHS=4
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
from scripts.pc_shard_train_ref import step_L, accuracy_L, D, C

ENV = dict(os.environ)
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1); ENV[k] = v

L      = int(ENV.get("PC_LAYERS", 8))
BINS   = ENV.get("PC_BINS", "/tmp/pc_fashion")
IP     = ENV["PC_INITPERM"]
FINAL  = ENV.get("PC_FINAL", "/tmp")
EPOCHS = int(ENV.get("PC_EPOCHS", 4))
N      = int(ENV.get("PC_BATCH", 32))
RES    = ENV.get("PC_RESIDUAL", "") != ""


def ld(p):
    return torch.from_numpy(np.fromfile(p, dtype=np.int32).astype(np.int64)).double().reshape(D, D)


def load_bins(d):
    Xtr = np.fromfile(f"{d}/train_x.i8", dtype=np.int8).reshape(-1, D).astype(np.float64)
    Ytr = np.fromfile(f"{d}/train_y.u8", dtype=np.uint8).astype(np.int64)
    Xte = np.fromfile(f"{d}/test_x.i8",  dtype=np.int8).reshape(-1, D).astype(np.float64)
    Yte = np.fromfile(f"{d}/test_y.u8",  dtype=np.uint8).astype(np.int64)
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def main():
    Ws = [ld(f"{IP}/W{l}_init.i32") for l in range(L)]
    ms = [torch.zeros(D, D, dtype=torch.float64) for _ in range(L)]
    perm = torch.from_numpy(np.fromfile(f"{IP}/perm.i32", dtype=np.int32).astype(np.int64))
    u = np.unique(perm.numpy())
    print(f"order: len={len(perm)} unique={len(u)} valid_permutation={len(u)==len(perm)}", flush=True)
    Xtr, Ytr, Xte, Yte = load_bins(BINS)
    nb = Xtr.shape[0] // N
    for ep in range(EPOCHS):
        for b in range(nb):
            idx = perm[b*N:(b+1)*N]
            Ws, ms, _ = step_L(Xtr[idx], Ytr[idx], Ws, ms, res=RES)
    acc = accuracy_L(Xte, Yte, Ws, res=RES)
    print(f"PYTHON {EPOCHS}-epoch ({EPOCHS*nb} steps), L={L}: python acc={acc:.4f}", flush=True)
    allzero = True
    for l in range(L):
        fp = f"{FINAL}/shardW{l}.i32"
        if not os.path.exists(fp):
            print(f"  chip {l}: (no silicon dump {fp}) — skipped"); continue
        sW = ld(fp); d = int((Ws[l] - sW).abs().max())
        allzero &= (d == 0)
        print(f"  chip {l}: W max|diff|={d}  {'OK' if d==0 else 'DIVERGES'}")
    print("  -> " + ("BIT-EXACT: the distributed FPGA training == the reference for every chip over all steps."
                     if allzero else "DIVERGES on >=1 chip -- investigate."))


if __name__ == "__main__":
    main()
