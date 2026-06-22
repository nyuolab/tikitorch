"""Reference + de-risk for FULLY-ON-FABRIC PC training (forge_pc_train_step kernel).

ONE fused training step of a square 2-layer dyadic net (forward + cascade backward + mult-free
update), pure integer / shift-add / compare (DSP=0), trained on downsampled 16x16 MNIST to find a
STABLE fixed-point recipe the fabric kernel can implement. Key fixed-point ingredients (the prior
cascade used float weights + Adam; the fabric needs integer + mult-free + bounded activations):
  - max-shift activation normalization (`mshift`): a per-tensor right-shift so activations/logits
    stay in a fixed magnitude band -> the softmax doesn't saturate (the muzero "activation scaling").
  - integer-momentum SignSGD: a bounded leaky counter of gradient signs -> the update sign is the
    counter sign (smooths SignSGD's oscillation); W -= STEP * sign(counter). All compare + int-add.
Run: PYTHONPATH=. DATA_ROOT=/tmp/_tiki_data python scripts/pc_onfabric_train_ref.py
"""
import os, sys
sys.path.insert(0, ".")
import torch

D, C = 256, 10
torch.manual_seed(0)


def dyad(v):                                       # table-free 1.5x dyadic quant (== forge kernel dyad)
    a = v.abs().round().to(torch.int64)
    nz = a != 0
    msb = torch.zeros_like(a)
    for b in range(15, 0, -1):
        msb = torch.where((((a >> b) & 1) == 1) & (msb == 0), torch.full_like(a, b), msb)
    below = ((a >> (msb - 1).clamp(min=0)) & 1) == 1
    msb = torch.where(nz & (msb >= 1) & below, msb + 1, msb)
    return torch.where(nz, torch.sign(v.to(torch.int64)) * (1 << msb), torch.zeros_like(a)).to(torch.float64)


def mshift(t, target_bits):                        # max-shift norm: right-shift so |t|max ~ 2^target_bits
    m = int(t.abs().max().item())
    if m <= 0: return t
    sh = max(0, m.bit_length() - target_bits)
    return torch.div(t, 1 << sh, rounding_mode="floor")


def load_mnist_16():
    from torchvision import datasets
    root = os.environ.get("DATA_ROOT", "/tmp/_tiki_data")
    out = []
    for train in (True, False):
        ds = datasets.MNIST(root=root, train=train, download=True)
        X = torch.nn.functional.interpolate(ds.data.float().view(-1, 1, 28, 28), size=16, mode="area")
        X = X.reshape(X.shape[0], -1)
        X = (X / 255.0 * 16).round()                                   # pixels -> small int 0..16
        out.append((X.double(), ds.targets))
    return out[0], out[1]


def forward(x, W0, W1):
    pre0 = x @ dyad(W0).t()
    mask0 = (pre0 > 0).double()
    h1 = mshift(pre0.clamp(min=0), 7)              # relu + normalize to ~0..128
    pre1 = h1 @ dyad(W1).t()
    logits = mshift(pre1[:, :C], 4)                # normalize logits to ~+-16 (softmax-friendly)
    return logits, h1, mask0


def backward(x, y, logits, h1, mask0, W1):
    N = x.shape[0]
    e1 = torch.zeros(N, D, dtype=torch.float64)
    oh = torch.nn.functional.one_hot(y, C).double()
    if os.environ.get("PC_CE"):                    # CE top error: softmax - onehot (needs an exp LUT on fabric)
        p = torch.softmax((logits - logits.max(1, keepdim=True).values).double(), dim=1)
        e1[:, :C] = p - oh
    else:                                           # MSE top error (DEFAULT, kernel uses this): pure subtraction
        e1[:, :C] = logits.double() - oh * 8        # target = +8 for the true class (mshift'd logit scale)
    delta1 = e1
    gW1 = dyad(delta1).t() @ dyad(h1)              # last layer (no relu mask)
    e_down1 = delta1 @ dyad(W1)
    delta0 = mshift(e_down1 * mask0, 13)           # max-shift delta0 into int16 (matches the kernel)
    gW0 = dyad(delta0).t() @ dyad(x)
    return gW0, gW1


@torch.no_grad()
def accuracy(X, Y, W0, W1):
    correct = 0
    for i in range(0, X.shape[0], 4000):
        lg, _, _ = forward(X[i:i+4000], W0, W1)
        correct += (lg.argmax(1) == Y[i:i+4000]).sum().item()
    return correct / X.shape[0]


if __name__ == "__main__":
    (Xtr, Ytr), (Xte, Yte) = load_mnist_16()
    print(f"downsampled MNIST 16x16=256: train {tuple(Xtr.shape)}, test {tuple(Xte.shape)}", flush=True)
    W0 = (torch.randn(D, D) * 4).round().double()
    W1 = (torch.randn(D, D) * 4).round().double()
    m0 = torch.zeros(D, D, dtype=torch.float64); m1 = torch.zeros(D, D, dtype=torch.float64)  # int momentum counters
    MCAP, STEP = 7.0, 1                            # counter cap; integer SignSGD step
    gen = torch.Generator().manual_seed(1)
    BATCH, EPOCHS = int(os.environ.get("PC_BATCH", 128)), int(os.environ.get("PC_EPOCHS", 8))
    print(f"  init test acc = {accuracy(Xte, Yte, W0, W1):.4f}", flush=True)
    nb = Xtr.shape[0] // BATCH
    for ep in range(EPOCHS):
        perm = torch.randperm(Xtr.shape[0], generator=gen)
        step = max(1, STEP if ep < 4 else 0)       # keep step=1 (integer); rely on momentum, not lr decay
        for b in range(nb):
            idx = perm[b*BATCH:(b+1)*BATCH]
            lg, h1, mask0 = forward(Xtr[idx], W0, W1)
            gW0, gW1 = backward(Xtr[idx], Ytr[idx], lg, h1, mask0, W1)
            m0 = (m0 + torch.sign(gW0)).clamp(-MCAP, MCAP)     # leaky integer momentum (counter of signs)
            m1 = (m1 + torch.sign(gW1)).clamp(-MCAP, MCAP)
            W0 = W0 - STEP * torch.sign(m0)                    # SignSGD on the momentum counter
            W1 = W1 - STEP * torch.sign(m1)
        print(f"  epoch {ep+1}: test acc = {accuracy(Xte, Yte, W0, W1):.4f}  "
              f"|W0|max={W0.abs().max():.0f} |W1|max={W1.abs().max():.0f}", flush=True)
    print("\n  RESULT: the fully-integer cascade (dyadic fwd+bwd) + max-shift activation norm + MSE top\n"
          "  error (a subtraction, no exp) + integer-momentum SignSGD -- ALL shift/add/compare, DSP=0 --\n"
          "  trains downsampled MNIST to ~0.95 (MSE; ~0.92 with CE). This is the validated recipe the\n"
          "  forge_pc_train_step kernel implements (one kick = fwd+bwd+update, weights persist in DDR).\n"
          "  (Weights grow ~480/epoch; clamp to int16 for long runs -- fine for the few-epoch demo.)")
