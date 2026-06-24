"""Preprocess Fashion-MNIST -> flat int8 bins for the distributed PC training driver (forge_pc_train_shard.c).

Identical pipeline to pc_mnist_to_bin.py (28x28 -> 16x16 area-downsample, pixels scaled 0..16 int8), but on
Fashion-MNIST — a harder 10-class drop-in (same D=256, same C=10) so the 8-FPGA scaling demo runs with NO
kernel change. Run on the F2 box before the training driver (the runner invokes this).
  PYTHONPATH=. DATA_ROOT=/tmp/_tiki_data python scripts/pc_fashion_to_bin.py <out_dir>
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch


def main(out_dir):
    from torchvision import datasets
    root = os.environ.get("DATA_ROOT", "/tmp/_tiki_data")
    os.makedirs(out_dir, exist_ok=True)
    for split, train in (("train", True), ("test", False)):
        ds = datasets.FashionMNIST(root=root, train=train, download=True)
        X = torch.nn.functional.interpolate(ds.data.float().view(-1, 1, 28, 28), size=16, mode="area")
        X = (X.reshape(X.shape[0], -1) / 255.0 * 16).round().clamp(0, 127).to(torch.int8).numpy()   # 0..16, int8
        y = ds.targets.to(torch.uint8).numpy()
        X.tofile(os.path.join(out_dir, f"{split}_x.i8"))
        y.tofile(os.path.join(out_dir, f"{split}_y.u8"))
        print(f"{split}: x {X.shape} {X.dtype} -> {split}_x.i8 ; y {y.shape} -> {split}_y.u8", flush=True)
    print(f"wrote Fashion-MNIST bins to {out_dir} (D=256, C=10, scale 0..16)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pc_fashion")
