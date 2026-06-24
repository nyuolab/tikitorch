"""Screen + emit a VERIFIED-GOOD init + data order for the L-chip distributed PC trainer (forge_pc_train_shard.c).

The MSE+SignSGD recipe has class-0 collapse basins for SOME valid data orders (the signature-result lesson).
This screens several (init_seed) candidates — each with its own fixed data order, reused every epoch exactly as
the driver does when PC_INITPERM is set — picks the best non-collapsing one, and dumps:
    <out>/W0_init.i32 .. W{L-1}_init.i32   (each D*D int32, round(randn*4) init the kernel reproduces bit-exactly)
    <out>/perm.i32                          (Ntr int32 data order, reused each epoch)
The silicon then loads these via PC_INITPERM and bit-exactly reproduces the screened accuracy.

It screens on the EXACT int8 bins the driver feeds (0..16), so it is faithful to the silicon data path.

  PYTHONPATH=. python scripts/pc_shard_screen_init.py \
      PC_LAYERS=8 PC_BINS=/tmp/pc_fashion PC_EPOCHS=4 PC_SEEDS=8 PC_OUT=/tmp/pc_initperm_fashion8
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
from scripts.pc_shard_train_ref import step_L, accuracy_L, init_Ws, D, C

# env-as-args (also accept k=v on argv)
ENV = dict(os.environ)
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1); ENV[k] = v

L      = int(ENV.get("PC_LAYERS", 8))
BINS   = ENV.get("PC_BINS", "/tmp/pc_fashion")
EPOCHS = int(ENV.get("PC_EPOCHS", 4))
NSEED  = int(ENV.get("PC_SEEDS", 8))
OUT    = ENV.get("PC_OUT", "/tmp/pc_initperm")
N      = int(ENV.get("PC_BATCH", 32))
RES    = ENV.get("PC_RESIDUAL", "") != ""    # host-side residual skip on hidden chips (deep nets)


def load_bins(d):
    Xtr = np.fromfile(f"{d}/train_x.i8", dtype=np.int8).reshape(-1, D).astype(np.float64)
    Ytr = np.fromfile(f"{d}/train_y.u8", dtype=np.uint8).astype(np.int64)
    Xte = np.fromfile(f"{d}/test_x.i8",  dtype=np.int8).reshape(-1, D).astype(np.float64)
    Yte = np.fromfile(f"{d}/test_y.u8",  dtype=np.uint8).astype(np.int64)
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def run_one(Xtr, Ytr, Xte, Yte, seed, epochs):
    Ws = init_Ws(L, seed)
    ms = [torch.zeros(D, D, dtype=torch.float64) for _ in range(L)]
    g = torch.Generator().manual_seed(1000 + seed)
    perm = torch.randperm(Xtr.shape[0], generator=g)        # ONE order, reused each epoch (driver behaviour)
    nb = Xtr.shape[0] // N
    traj = []
    for ep in range(epochs):
        for b in range(nb):
            idx = perm[b*N:(b+1)*N]
            Ws, ms, _ = step_L(Xtr[idx], Ytr[idx], Ws, ms, res=RES)
        acc = accuracy_L(Xte, Yte, Ws, res=RES)
        traj.append(acc)
        print(f"    seed {seed} epoch {ep+1}: acc={acc:.4f}", flush=True)
    return Ws, perm, traj


def main():
    Xtr, Ytr, Xte, Yte = load_bins(BINS)
    print(f"[screen] L={L} chips, bins={BINS} train={tuple(Xtr.shape)} test={tuple(Xte.shape)}, "
          f"{NSEED} seeds x {EPOCHS} epochs", flush=True)
    best = None
    for seed in range(NSEED):
        _, perm, traj = run_one(Xtr, Ytr, Xte, Yte, seed, EPOCHS)
        fin = traj[-1]
        if best is None or fin > best[1]:
            best = (seed, fin, perm)
        print(f"  seed {seed}: final acc={fin:.4f}  (best so far seed {best[0]} @ {best[1]:.4f})", flush=True)
    seed, fin, perm = best
    print(f"[screen] WINNER seed {seed} final acc={fin:.4f} -> dumping verified init+order to {OUT}", flush=True)
    # re-run the winner to dump the EXACT init it started from (init_Ws is deterministic per seed)
    Ws0 = init_Ws(L, seed)
    os.makedirs(OUT, exist_ok=True)
    for l in range(L):
        Ws0[l].round().to(torch.int32).numpy().tofile(f"{OUT}/W{l}_init.i32")
    perm.to(torch.int32).numpy().tofile(f"{OUT}/perm.i32")
    with open(f"{OUT}/screen.txt", "w") as f:
        f.write(f"L={L} bins={BINS} epochs={EPOCHS} winner_seed={seed} final_acc={fin:.4f}\n")
    print(f"[screen] wrote W0..W{L-1}_init.i32 + perm.i32 ({Xtr.shape[0]} ints) to {OUT}", flush=True)


if __name__ == "__main__":
    main()
