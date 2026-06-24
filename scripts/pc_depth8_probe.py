"""Two principled fixes for the depth-8 MSE+SignSGD collapse (host-side, NO kernel rebuild):
  (A) STICKY dyadic-identity hidden init: diag=64 -> dyad=2^6, snaps back to 64 for the whole [48,96) band, so
      SignSGD's +-1/step can't destroy the passthrough -> deep signal survives.
  (B) FREEZE the interior layers (per-chip DO_UPDATE=0, already supported by the kernel): train only the
      boundary layers; the frozen chips still do a real fwd+bwd route -> genuine 8-chip model-parallel compute.
Both reuse the EXACT kernel arithmetic (step_L). 3 epochs, 30k samples.
  PYTHONPATH=. python scripts/pc_depth8_probe.py
"""
import sys, time, torch
sys.path.insert(0, ".")
from scripts.pc_shard_train_ref import step_L, accuracy_L, D, C
from scripts.pc_shard_screen_init import load_bins

Xtr, Ytr, Xte, Yte = load_bins("/tmp/pc_fashion")
NSAMP = 30000
Xtr, Ytr = Xtr[:NSAMP], Ytr[:NSAMP]
N, L, EP = 32, 8, 3


def mk(mode, diag, sig, seed):
    g = torch.Generator().manual_seed(seed); Ws = []
    for l in range(L):
        if l == L - 1:
            Ws.append((torch.randn(D, D, generator=g) * 4).round().double())          # output: random
        elif mode == "sticky":
            W = (torch.randn(D, D, generator=g) * sig).round().double(); W.fill_diagonal_(float(diag)); Ws.append(W)
        else:                                                                          # random hidden
            Ws.append((torch.randn(D, D, generator=g) * 4).round().double())
    return Ws


def run(mode, diag, sig, seed, upd=None):
    Ws = mk(mode, diag, sig, seed); ms = [torch.zeros(D, D, dtype=torch.float64) for _ in range(L)]
    g = torch.Generator().manual_seed(1000 + seed); perm = torch.randperm(Xtr.shape[0], generator=g)
    nb = Xtr.shape[0] // N; traj = []
    for ep in range(EP):
        for b in range(nb):
            idx = perm[b*N:(b+1)*N]; Ws, ms, _ = step_L(Xtr[idx], Ytr[idx], Ws, ms, upd=upd)
        traj.append(round(accuracy_L(Xte, Yte, Ws), 4))
    return traj


# (A) sticky dyadic-identity, all layers trainable
for diag in (32, 64):
    t = time.time(); traj = run("sticky", diag, 2, seed=0)
    print(f"A sticky diag={diag:3d} (all-train)   traj={traj}  ({time.time()-t:.0f}s)", flush=True)
# (B) freeze interior: train only chip 0 (first hidden) + chip 7 (output); chips 1..6 frozen near-identity
upd = [1, 0, 0, 0, 0, 0, 0, 1]
t = time.time(); traj = run("sticky", 64, 2, seed=0, upd=upd)
print(f"B freeze-interior diag=64 upd={upd} traj={traj}  ({time.time()-t:.0f}s)", flush=True)
# (B2) freeze interior with trainable first-3 + output
upd2 = [1, 1, 1, 0, 0, 0, 0, 1]
t = time.time(); traj = run("sticky", 64, 2, seed=0, upd=upd2)
print(f"B2 freeze diag=64 upd={upd2} traj={traj}  ({time.time()-t:.0f}s)", flush=True)
