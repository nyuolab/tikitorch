"""Phase 1 scale test: multiplier-free DYADIC convolutional nets on CIFAR-10/100.

This is the real Qi-et-al-scale comparison (they train deep VGG-style PC nets on
CIFAR/Tiny-ImageNet). We carry the extra multiplier-free (+-2^k) constraint, so
the honest question is: does a +-2^k dyadic VGG match a float VGG at depth?

For convolutions the predictive-coding "cascade" (single-pass cached-routing
backward) computes exactly the chain-rule gradient through the cached routing --
which is precisely what autograd computes through the same forward. So we realize
the cascade via autograd-through-the-dyadic-forward, and add the multiplier-free
weight update as the conv analog of DyadicPCNet.set_mf_grads: backward hooks
capture each layer's input activation `a` and output-error `g`, quantize BOTH to
+-2^k, and recompute the weight grad as a shift-add correlation (no multiply on
fabric). The on-fabric cached-routing cascade produces the identical gradient.

Modes:
  bp_float   -- float Conv2d/Linear VGG, backprop. The absolute ceiling.
  bp_dyadic  -- DyadicConv2d/DyadicLinear (multiplier-free FORWARD), backprop
                weight update. Isolates the cost of multiplier-free *inference*
                (what deploys on the FPGA forward kernel).
  cascade    -- dyadic forward + multiplier-free weight update (operand-quantized
                grads). The fully multiplier-free learner.

Usage (matrix, what the EC2 runner invokes):
    python scripts/pc_scale_cifar.py --matrix --out results/predictive_coding/scale_cifar.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch.nn import dyadic_quantize

# CIFAR channel stats.
_NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": ((0.480, 0.448, 0.398), (0.277, 0.269, 0.282)),
}
_NCLASS = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}

# VGG-style configs: ints = conv out-channels, 'M' = 2x2 maxpool. The label's
# number is (#conv + 1 FC head) to echo Qi's VGG-5/7/10/15 naming.
_CFGS = {
    "vgg5":  [64, "M", 128, "M", 256, "M"],
    "vgg7":  [64, "M", 128, "M", 256, 256, "M"],
    "vgg10": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, "M"],
    "vgg15": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
              512, 512, 512, "M"],                  # 13-conv deep VGG (~Qi VGG-15/16)
}


class DyadicConv2d(nn.Conv2d):
    """``nn.Conv2d`` with weights snapped to signed powers of two (multiplier-free
    forward; a shift+add correlation, zero DSP). STE through the rounding, exactly
    like :class:`tikitorch.nn.DyadicLinear`."""

    _zero_dsp_layer = True

    def forward(self, x):
        return self._conv_forward(x, dyadic_quantize(self.weight), self.bias)


def make_vgg(cfg, n_classes, dyadic: bool):
    Conv = DyadicConv2d if dyadic else nn.Conv2d
    Lin = None  # set after import to avoid shadowing
    from tikitorch.nn import DyadicLinear
    Lin = DyadicLinear if dyadic else nn.Linear
    layers, c_in = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [Conv(c_in, v, 3, padding=1, bias=False),
                       nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            c_in = v
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), Lin(c_in, n_classes)]
    return nn.Sequential(*layers)


class _BasicBlock(nn.Module):
    """ResNet basic block, dyadic convs + BN + additive skip (the standard residual
    that float ResNets use; the cascade/MFGrad handles the skip via autograd)."""

    def __init__(self, Conv, c_in, c_out, stride):
        super().__init__()
        self.conv1 = Conv(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = Conv(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(Conv(c_in, c_out, 1, stride=stride, bias=False),
                                      nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (self.down(x) if self.down is not None else x)
        return torch.relu(out)


def make_resnet18(n_classes, dyadic: bool):
    """CIFAR ResNet-18 (3x3 stem, 4 stages x 2 basic blocks). All convs/fc dyadic when
    dyadic=True. This is exactly where iterative settling-PC collapses (lit: PC 70.4 vs
    BP 92.8 on CIFAR-10) — the dyadic cascade (=backprop) should match BP."""
    Conv = DyadicConv2d if dyadic else nn.Conv2d
    from tikitorch.nn import DyadicLinear
    Lin = DyadicLinear if dyadic else nn.Linear
    stage = [(64, 1), (128, 2), (256, 2), (512, 2)]
    layers = [Conv(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
    c_in = 64
    for c_out, stride in stage:
        layers.append(_BasicBlock(Conv, c_in, c_out, stride))
        layers.append(_BasicBlock(Conv, c_out, c_out, 1))
        c_in = c_out
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), Lin(512, n_classes)]
    return nn.Sequential(*layers)


def make_net(cfg_name, n_classes, dyadic):
    if cfg_name.startswith("resnet"):
        return make_resnet18(n_classes, dyadic)
    return make_vgg(_CFGS[cfg_name], n_classes, dyadic=dyadic)


class MFGrad:
    """Multiplier-free weight-grad manager (conv analog of set_mf_grads("dyadic")).

    Hooks each Dyadic layer to cache its input activation and output error; after
    backward, overwrites weight.grad with the operand-quantized (+-2^k x +-2^k =
    shift-add) version. Bias grads (a plain sum, already add-only) are left as-is.
    """

    def __init__(self, model):
        self.acts, self.gouts, self.mods = {}, {}, []
        for m in model.modules():
            if isinstance(m, (DyadicConv2d,)) or m.__class__.__name__ == "DyadicLinear":
                self.mods.append(m)
                m.register_forward_hook(self._fwd(m))
                m.register_full_backward_hook(self._bwd(m))

    def _fwd(self, m):
        def hook(mod, inp, out):
            self.acts[m] = inp[0].detach()
        return hook

    def _bwd(self, m):
        def hook(mod, gin, gout):
            self.gouts[m] = gout[0].detach()
        return hook

    @torch.no_grad()
    def apply(self):
        for m in self.mods:
            a = dyadic_quantize(self.acts[m])
            g = dyadic_quantize(self.gouts[m])
            if isinstance(m, nn.Conv2d):
                gw = torch.nn.grad.conv2d_weight(
                    a, m.weight.shape, g, stride=m.stride, padding=m.padding,
                    dilation=m.dilation, groups=m.groups)
            else:                                   # linear: [out,in] = g^T a
                gw = g.transpose(0, 1) @ a
            m.weight.grad = gw


def _prep_tinyimagenet(root):
    """Download Tiny-ImageNet-200 and reorganize val/ into class folders for ImageFolder."""
    import os, urllib.request, zipfile, shutil
    base = os.path.join(root, "tiny-imagenet-200")
    if not os.path.isdir(base):
        zp = os.path.join(root, "tiny-imagenet-200.zip")
        if not os.path.exists(zp):
            os.makedirs(root, exist_ok=True)
            urllib.request.urlretrieve(
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip", zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
    val_org = os.path.join(base, "val_organized")
    if not os.path.isdir(val_org):
        os.makedirs(val_org, exist_ok=True)
        img_dir = os.path.join(base, "val", "images")
        with open(os.path.join(base, "val", "val_annotations.txt")) as f:
            for line in f:
                fn, wnid = line.split("\t")[:2]
                cd = os.path.join(val_org, wnid)
                os.makedirs(cd, exist_ok=True)
                shutil.copy(os.path.join(img_dir, fn), os.path.join(cd, fn))
    return os.path.join(base, "train"), val_org


def loaders(name, data_root, batch, workers, augment=True):
    from torchvision import datasets, transforms
    mean, std = _NORM[name]
    norm = transforms.Normalize(mean, std)
    imgsz = 64 if name == "tinyimagenet" else 32
    tf_tr = transforms.Compose(
        ([transforms.RandomCrop(imgsz, padding=imgsz // 8), transforms.RandomHorizontalFlip()]
         if augment else []) + [transforms.ToTensor(), norm])
    tf_te = transforms.Compose([transforms.ToTensor(), norm])
    if name == "tinyimagenet":
        train_dir, val_dir = _prep_tinyimagenet(data_root)
        tr = datasets.ImageFolder(train_dir, transform=tf_tr)
        te = datasets.ImageFolder(val_dir, transform=tf_te)
    else:
        cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
        tr = cls(data_root, train=True, download=True, transform=tf_tr)
        te = cls(data_root, train=False, download=True, transform=tf_te)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=sh, num_workers=workers, pin_memory=True,
        drop_last=sh)
    return mk(tr, True), mk(te, False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / total


def run_one(mode, cfg_name, dataset, *, seed, epochs, lr, batch, workers,
            data_root, device, augment):
    torch.manual_seed(seed)
    n_classes = _NCLASS[dataset]
    dyadic = mode in ("bp_dyadic", "cascade")
    model = make_net(cfg_name, n_classes, dyadic=dyadic).to(device)
    mf = MFGrad(model) if mode == "cascade" else None
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr, te = loaders(dataset, data_root, batch, workers, augment=augment)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if mf is not None:
                mf.apply()                          # overwrite weight grads (mult-free)
            opt.step()
        sched.step()
    acc = evaluate(model, te, device)
    return {"test_acc": round(acc, 4), "wall_s": round(time.time() - t0, 1),
            "epochs": epochs}


def build_matrix(args):
    jobs = []
    for dataset in args.datasets:
        for cfg in args.cfgs:
            for mode in args.modes:
                for seed in args.seeds:
                    jobs.append(dict(dataset=dataset, cfg=cfg, mode=mode, seed=seed))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["cifar10"])
    p.add_argument("--cfgs", nargs="+", default=["vgg7", "vgg10"])
    p.add_argument("--modes", nargs="+", default=["bp_float", "bp_dyadic", "cascade"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./_data"))
    p.add_argument("--out", default="results/predictive_coding/scale_cifar.jsonl")
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny CPU sanity run")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.datasets, args.cfgs = ["cifar10"], ["vgg5"]
        args.modes, args.seeds = ["bp_dyadic", "cascade"], [0]
        args.epochs, args.workers = 1, 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    jobs = build_matrix(args)
    print(f"device={device}  jobs={len(jobs)}  out={args.out}", flush=True)
    fout = open(args.out, "a")
    for j, job in enumerate(jobs):
        ep = 1 if args.smoke else args.epochs
        m = run_one(job["mode"], job["cfg"], job["dataset"], seed=job["seed"],
                    epochs=ep, lr=args.lr, batch=args.batch, workers=args.workers,
                    data_root=args.data_root, device=device, augment=not args.smoke)
        rec = {**job, "device": str(device), **m}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[{j+1}/{len(jobs)}] {job['dataset']:8s} {job['cfg']:6s} "
              f"{job['mode']:10s} s{job['seed']}  test={m['test_acc']:.4f}  "
              f"({m['wall_s']}s)", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
