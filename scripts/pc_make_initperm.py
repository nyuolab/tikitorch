"""Emit a verified-good init + data order for the L-chip distributed PC trainer WITHOUT a slow training screen.
With the residual skip the recipe is robust (no class-collapse), so a fixed seed's round(randn*4) init + a single
reused data order trains; the post-run bit-exact replay (pc_shard_perm_replay_L.py) confirms the silicon == this
reference exactly. Dumps W0..W{L-1}_init.i32 (each D*D int32) + perm.i32 (Ntr int32) in the driver's format.

  PYTHONPATH=. python scripts/pc_make_initperm.py PC_LAYERS=8 PC_BINS=/tmp/pc_fashion PC_SEED=0 PC_OUT=/tmp/ip8
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np, torch
from scripts.pc_shard_train_ref import init_Ws, D

ENV = dict(os.environ)
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1); ENV[k] = v
L    = int(ENV.get("PC_LAYERS", 8))
BINS = ENV.get("PC_BINS", "/tmp/pc_fashion")
SEED = int(ENV.get("PC_SEED", 0))
OUT  = ENV.get("PC_OUT", "/tmp/pc_initperm")

Ntr = np.fromfile(f"{BINS}/train_y.u8", dtype=np.uint8).shape[0]
os.makedirs(OUT, exist_ok=True)
Ws = init_Ws(L, SEED)
for l in range(L):
    Ws[l].round().to(torch.int32).numpy().tofile(f"{OUT}/W{l}_init.i32")
g = torch.Generator().manual_seed(1000 + SEED)
perm = torch.randperm(Ntr, generator=g).to(torch.int32)
perm.numpy().tofile(f"{OUT}/perm.i32")
print(f"[make] L={L} seed={SEED} Ntr={Ntr} -> {OUT}/W0..W{L-1}_init.i32 + perm.i32 "
      f"(reused each epoch; residual recipe; replay verifies bit-exactness post-run)", flush=True)
