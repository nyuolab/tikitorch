"""Independent check on the FULLY-ON-FABRIC trained weights.

The forge_pc_train_step kernel trains W0/W1 entirely on the FPGA (fwd+bwd+update in DDR); the driver
dumps the final integer weights (trainedW0.i32 / trainedW1.i32). This script loads those exact bytes
and runs the *recipe* forward (scripts/pc_onfabric_train_ref.py, kernel-shift convention) on 16x16
MNIST -> an accuracy computed by a SEPARATE codebase. If it matches the on-fabric driver's reported
accuracy, the fabric-trained weights are genuinely good (not a predict() artifact) and the host
forward == the kernel forward. Pure cross-check; no training here.

Run: PYTHONPATH=. DATA_ROOT=/tmp/_tiki_data \
     python scripts/pc_eval_onfabric_weights.py results/.../trainedW0.i32 results/.../trainedW1.i32
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
os.environ["PC_KERNEL_SHIFT"] = "1"   # eval with the FABRIC kernel's norm_shift (matches the trained weights)
from scripts.pc_onfabric_train_ref import D, C, load_mnist_16, forward, accuracy  # noqa: E402


def load_i32(path):
    w = np.fromfile(path, dtype=np.int32)
    assert w.size == D * D, f"{path}: expected {D*D} int32, got {w.size}"
    return torch.from_numpy(w.astype(np.int64)).double().reshape(D, D)


if __name__ == "__main__":
    w0_path, w1_path = sys.argv[1], sys.argv[2]
    W0, W1 = load_i32(w0_path), load_i32(w1_path)
    print(f"loaded on-fabric weights: |W0|max={W0.abs().max():.0f} |W1|max={W1.abs().max():.0f} "
          f"(nz W0={int((W0!=0).sum())}/{D*D})", flush=True)
    (_, _), (Xte, Yte) = load_mnist_16()
    acc = accuracy(Xte, Yte, W0, W1)
    # per-class prediction histogram (catches a degenerate all-one-class predictor)
    hist = torch.zeros(C, dtype=torch.long)
    for i in range(0, Xte.shape[0], 4000):
        lg, _, _ = forward(Xte[i:i+4000], W0, W1)
        for c in lg.argmax(1):
            hist[c] += 1
    print(f"RECIPE-EVAL of ON-FABRIC weights: test acc = {acc:.4f}", flush=True)
    print(f"  pred histogram (per class 0..9): {hist.tolist()}", flush=True)
    print("  -> if this matches the driver's on-fabric accuracy, the fabric-trained weights are\n"
          "     genuinely good and the host predict() == kernel forward (independent confirmation).")
