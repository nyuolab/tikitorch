"""Replay the first K silicon shard steps in Python and find where they diverge. Loads silicon-dumped init
weights + the first K batches + the post-step-K weights, runs pc_shard_train_ref.step() K times, diffs."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from scripts.pc_shard_train_ref import step, D, C

def ld(p, shape):
    return torch.from_numpy(np.fromfile(p, dtype=np.int32).astype(np.int64)).double().reshape(shape)

K = 4
iW0 = ld("/tmp/s1/s1_iW0.i32", (D, D)); iW1 = ld("/tmp/s1/s1_iW1.i32", (D, D))
X = ld("/tmp/s1/s1_x.i32", (K, -1, D)); Y = ld("/tmp/s1/s1_y.i32", (K, -1)).long()
sW0 = ld("/tmp/s1/s1_W0.i32", (D, D)); sW1 = ld("/tmp/s1/s1_W1.i32", (D, D))
print(f"replaying {K} steps; batch {X.shape[1]}x{D}; labels step0={Y[0,:8].tolist()}")

W0, W1 = iW0.clone(), iW1.clone()
m0 = torch.zeros(D, D, dtype=torch.float64); m1 = torch.zeros(D, D, dtype=torch.float64)
for k in range(K):
    W0, W1, m0, m1, _ = step(X[k], Y[k], W0, W1, m0, m1)
d0 = (W0 - sW0).abs(); d1 = (W1 - sW1).abs()
print(f"PYTHON({K} steps) vs SILICON: W0 max|diff|={int(d0.max())} ({int((d0>0).sum())} differ) | "
      f"W1 max|diff|={int(d1.max())} ({int((d1>0).sum())} differ)")
print("  -> " + ("BIT-EXACT through 4 steps (bug is even later / data-order)" if int(d0.max())==0 and int(d1.max())==0
                 else "DIVERGES within 4 steps -> inspect which layer/rows below"))
if int(d1.max())>0:
    print("  W1 per-row diffs (first 10 = output classes):", (d1[:C]>0).sum(1).tolist())
if int(d0.max())>0:
    print("  W0 rows with diffs:", int(((d0>0).sum(1)>0).sum()), "/", D)
