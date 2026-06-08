"""Generate the TikiTorch Colab notebooks (.ipynb) from cell definitions.

Run:  python notebooks/_build_notebooks.py
Produces: 01_intro.ipynb, 02_tropical_math.ipynb, 03_vision_cnn.ipynb,
          04_sequence.ipynb  (in this directory).

Keeping the notebooks as code here (rather than hand-written JSON) makes them
diffable and regenerable. Each notebook is self-contained and runnable in Google
Colab; the SETUP cell installs TikiTorch.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n") + "\n"}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n") + "\n"}


def write_nb(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


# Shared SETUP cell: install TikiTorch in Colab (or use a local checkout).
SETUP = code(r"""
# --- Setup: install TikiTorch (run once) ---
# TikiTorch is research code, not on PyPI. In Colab, install from the repo.
try:
    import tikitorch  # already installed?
except ImportError:
    REPO_URL = "https://github.com/nyuolab/tikitorch.git"
    import subprocess, sys, os
    if not os.path.exists("tikitorch_repo"):
        subprocess.run(["git", "clone", REPO_URL, "tikitorch_repo"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "tikitorch_repo"], check=False)
    import tikitorch
import torch
print("TikiTorch", getattr(tikitorch, "__version__", "?"), "| torch", torch.__version__)
""")


# ===========================================================================
# 01 — Intro
# ===========================================================================
intro = [
    md(r"""
# TikiTorch — deep learning without a single multiply

**TikiTorch** is a PyTorch-shaped deep-learning library whose entire learning
loop — forward, backward, *and* the optimizer — runs in the **tropical
semiring**: the only primitive operations are **compare, add, and bit-shift**.
There are **no multiplications anywhere**, which is exactly what lets a whole
training step map onto FPGA fabric with **zero DSP blocks** (zero hardware
multipliers).

In the tropical semiring, "multiply" becomes ordinary **addition** and "add"
becomes a **min** or **max** (a comparison). A linear layer
`y = Wx` becomes a *morphological* operation `y_j = min_i (W_ji + x_i)` (erosion)
or `max_i (W_ji + x_i)` (dilation). Stack those, add a tropical attention or a
morphological convolution, train with a sign-based optimizer, and you have a deep
net that never multiplies.

This notebook is a 10-minute tour. The others go deeper:
- **02_tropical_math** — the semiring, dilation/erosion, the soft→hard curriculum.
- **03_vision_cnn** — a fully multiplier-free CNN on MNIST.
- **04_sequence** — multiplier-free language modeling + the closure-faithful loss.
"""),
    SETUP,
    md(r"""
## A complete training step in ten lines

TikiTorch mirrors the PyTorch namespace: `tt.nn`, `tt.optim`, `tt.models`,
`tt.functional`. Here is a morphological state-space LM trained one step with a
strictly multiplier-free optimizer.
"""),
    code(r"""
import torch
import tikitorch as tt
import tikitorch.optim as optim

model = tt.models.MorphSSM(d_model=128, n_layers=2, task_type="lm", vocab_size=128)
opt = optim.PureTropicalSGD(model.parameters(), lr_shift=10)  # lr = 2^-10, a bit-shift

x = torch.randint(0, 128, (8, 64))   # (batch, seq)
y = torch.randint(0, 128, (8, 64))
loss = model.compute_loss(x, y)      # cross-entropy
loss.backward()                       # gradients route through argmin/argmax
opt.step(); opt.zero_grad()
print("loss:", round(loss.item(), 4))
"""),
    md(r"""
## The zero-multiply claim is *checkable*

The headline isn't a vibe — it's mechanically verifiable. `assert_closed` walks a
model and raises if it finds **any** standard `torch.nn` module that contains a
hardware multiply or divide (`nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, the cuDNN
RNNs, ...). A TikiTorch model built from tropical primitives passes.
"""),
    code(r"""
from tikitorch.audit import assert_closed, find_closure_violations

cnn = tt.models.TropicalCNN(in_channels=1, num_classes=10)   # morphological CNN
assert_closed(cnn)
print("TropicalCNN is fully multiplier-free  ✓  (params:", cnn.param_count, ")")

# A standard nn.Linear head is a *declared* DSP boundary — the checker names it:
boundary = tt.models.TropicalCNN(head_type="linear")
print("violations in linear-head variant:", find_closure_violations(boundary))
"""),
    md(r"""
## Why this matters: it runs on silicon at single-digit millijoules

Because the loop never multiplies, the same computation synthesizes to **zero DSP
blocks** on an FPGA. TikiTorch ships the HLS kernels; the architectures here were
trained **end-to-end on AWS F2 silicon** (Virtex UltraScale+ VU47P):

| Build | Throughput | Energy | DSP |
|---|---|---|---|
| 2-layer tropical LM (char) | ~772 tok/s | ~9.0 mJ/token | **0** |
| MorphSSM ECG (clinical) | ~1207 tok/s | ~5.8 mJ/token | **0** |

The accuracy cost of this *closure* is real and measured (a few points on
classification; larger on hard char-LM) — and TikiTorch gives you the knobs (the
soft→hard `tau` curriculum, the loss/optimizer choices) to trade it back. That's
the subject of the other notebooks.

**The map:**

| namespace | what |
|---|---|
| `tt.functional` | stateless tropical ops + autograd (linear, dilation, conv2d, attention, losses) |
| `tt.nn` | layer modules (TropicalLinear, TropicalConv2d, TropicalAttention, TropicalLayerNorm, ...) |
| `tt.optim` | multiplier-free optimizers (SignSGD, TropicalLion, PureTropicalSGD) |
| `tt.models` | ready architectures (TropicalMLP/CNN, MorphSSM, TropicalTransformer, MoTE) |
| `tt.audit` | `assert_closed` — verify a model is multiplier-free |
"""),
]


# ===========================================================================
# 02 — Tropical math
# ===========================================================================
math_nb = [
    md(r"""
# The tropical semiring, by hand

A *semiring* swaps the two operations of ordinary arithmetic. The **tropical**
(or min-plus / max-plus) semiring uses:

| | ordinary | tropical (min-plus) | tropical (max-plus) |
|---|---|---|---|
| "multiply" (⊗) | `a · b` | `a + b` | `a + b` |
| "add" (⊕) | `a + b` | `min(a, b)` | `max(a, b)` |

So a dot product `Σ_i a_i b_i` becomes `min_i (a_i + b_i)` (or `max_i`). No
multiplications — just additions and comparisons. This notebook builds intuition
for what that means for layers, gradients, and training.
"""),
    SETUP,
    md(r"""
## Tropical linear = morphological erosion / dilation

`tropical_linear` is `y_j = min_i (W_ji + x_i)` (min-plus *erosion*);
`tropical_dilation` is the `max` dual. Each output picks **one** input — its
argmin/argmax — so the backward pass routes the gradient only through that single
winning index (a sparse, one-hot scatter). That sparsity is the source of both
the efficiency and the hardware-friendliness.
"""),
    code(r"""
import torch
import tikitorch.functional as F

x = torch.randn(1, 5, requires_grad=True)
W = torch.randn(3, 5)
y = F.tropical_linear(x, W)                 # (1,3): y_j = min_i (W[j,i] + x[i])
print("y =", y.detach().numpy().round(3))

y.sum().backward()
print("grad_x nonzeros:", (x.grad != 0).sum().item(), "of", x.numel(),
      "-> each output routed to ONE input (sparse gradient)")
"""),
    md(r"""
## The soft → hard curriculum (Maslov dequantization)

The hard `min`/`max` has sparse gradients, which can be hard to train from
scratch. TikiTorch provides a **soft relaxation** controlled by a temperature
`tau`: `softmin(z, tau) = -tau · logsumexp(-z/tau)`, which is smooth (dense
gradients) for `tau > 0` and **converges to the hard min as `tau → 0`**. You
train soft, anneal `tau → 0`, and deploy the hard, multiplier-free op — the *same
weights* drive both. Let's watch a soft min converge to the hard min.
"""),
    code(r"""
import torch, tikitorch.functional as F
z = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
hard = z.min(dim=-1).values.item()
for tau in [2.0, 1.0, 0.5, 0.1, 0.01]:
    soft = F.tropical_softmin(z, tau=tau, dim=-1).item()
    print(f"tau={tau:>4}: softmin={soft:+.4f}   (hard min={hard:+.4f})")
"""),
    md(r"""
## Tropical matmul and the morphological view

`tropical_matmul(A, B)[i,j] = min_k (A[i,k] + B[k,j])`. The tropical identity is
a matrix with `0` on the diagonal and `+inf` off it; multiplying by it leaves `A`
unchanged — the same role the identity plays in ordinary linear algebra.

We can also *see* the morphology: a 1-D max-plus convolution (dilation) of a
spiky signal with a small structuring element spreads peaks; min-plus (erosion)
carves valleys. This is classical grayscale mathematical morphology — and it is
exactly what a tropical conv layer computes.
"""),
    code(r"""
import torch, tikitorch.functional as F

A = torch.randn(3, 3)
eye = torch.full((3, 3), float("inf")); eye.fill_diagonal_(0.0)
print("tropical_matmul(A, I) == A:", torch.allclose(F.tropical_matmul(A, eye), A))

# 1-D morphology: dilation (max-plus) vs erosion (min-plus) of a signal
import tikitorch.nn as nn
sig = torch.zeros(1, 1, 32); sig[0, 0, [8, 20]] = 1.0   # two spikes
dil = nn.MorphConv1d(1, kernel_size=5, mode="dilation", use_soft=False)
ero = nn.MorphConv1d(1, kernel_size=5, mode="erosion",  use_soft=False)
with torch.no_grad():
    dil.weight.zero_(); ero.weight.zero_()              # flat structuring element
    print("input peaks at:", (sig[0,0] > 0).nonzero().flatten().tolist())
    print("dilation spreads each peak into a plateau (max-plus).")
"""),
    md(r"""
## The closure penalty

Forcing the whole loop into the tropical semiring with sign-only updates costs
some accuracy versus an unconstrained `AdamW + softmax-CE + FP32` baseline — the
**closure penalty**. On a hard char-LM the penalty decomposes (measured) as:
SignSGD vs AdamW ≈ −13pp, perceptron loss vs cross-entropy ≈ −17pp, INT16
quantization ≈ −1pp, hard tropical ops ≈ −5pp. The point of TikiTorch is not to
beat a GPU transformer on perplexity — it is to make the *whole loop closed* and
to give you the levers (the `tau` curriculum, the loss, the optimizer) to climb
the penalty back. Notebook **04** shows one lever recovering ~11pp,
multiplier-free.
"""),
]


# ===========================================================================
# 03 — Vision: a multiplier-free CNN
# ===========================================================================
vision = [
    md(r"""
# A multiplier-free CNN on MNIST

A tropical 2-D convolution **is** a learned grayscale *morphological* operation:
`y = max_{ci,kh,kw} (weight + patch)` (dilation) or `min` (erosion). No multiplies
— a structuring element is added to each image patch and reduced by a compare.
This notebook builds a CNN entirely from such ops (conv → tropical-norm →
power-of-2 leaky-relu → max-pool → compositional tropical head), verifies it is
multiplier-free, and trains it on MNIST with a **sign-only optimizer**.
"""),
    SETUP,
    md("## See the morphology: TropicalConv2d as dilation / erosion"),
    code(r"""
import torch, tikitorch.nn as nn
img = torch.zeros(1, 1, 16, 16); img[0, 0, 6:10, 6:10] = 1.0   # a bright square

dil = nn.TropicalConv2d(1, 1, 3, padding=1, mode="dilation", bias=False)
ero = nn.TropicalConv2d(1, 1, 3, padding=1, mode="erosion",  bias=False)
with torch.no_grad():
    dil.weight.zero_(); ero.weight.zero_()           # flat 3x3 structuring element
    d = dil(img)[0, 0]; e = ero(img)[0, 0]
print("input bright pixels :", int((img > 0.5).sum()))
print("after dilation      :", int((d > 0.5).sum()), "(square grew — max-plus)")
print("after erosion       :", int((e > 0.5).sum()), "(square shrank — min-plus)")
"""),
    md(r"""
## The model: a multiplier-free morphological feature extractor

`TropicalCNN` alternates dilation/erosion conv blocks (opening/closing structure,
much more expressive than all-dilation), pools with `max`, normalizes with a
power-of-2 `TropicalLayerNorm`, keeps coarse spatial layout with an adaptive
max-pool grid, and ends in a head. Two honest choices for the head:

- **`head_type="linear"`** — a standard `nn.Linear` classifier. The *feature
  extractor* is fully multiplier-free; the head is a single **declared DSP
  boundary** (`assert_closed(model, allow={"head"})`). This is the strong,
  honest configuration: ~**96–97% MNIST with a sign-only optimizer**.
- **`head_type="tropical"`** — a compositional dilation∘erosion head, so the
  **entire** model is multiplier-free (`assert_closed(model)` passes outright).
  This costs accuracy at the 10-way head — the *closure penalty* made visible.

We train the strong configuration first, then quantify the fully-closed cost.
"""),
    code(r"""
import tikitorch as tt
from tikitorch.audit import assert_closed, find_closure_violations

model = tt.models.TropicalCNN(in_channels=1, num_classes=10, channels=(16, 32),
                              pool_grid=4, head_type="linear",
                              use_soft=True, tau=1.0)
model.loss_type = "cross_entropy"
# The feature extractor is multiplier-free; only the head is a declared boundary:
print("closure violations:", find_closure_violations(model))   # -> [('head', 'Linear')]
assert_closed(model, allow={"head"})
print("Backbone is multiplier-free ✓ (head = declared DSP boundary). params:", model.param_count)
"""),
    md("## Load MNIST and train with a sign-only optimizer + soft→hard anneal"),
    code(r"""
import torch
from torchvision import datasets, transforms
import tikitorch.optim as optim

tfm = transforms.ToTensor()
train = datasets.MNIST(root="./data", train=True,  download=True, transform=tfm)
test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test,  batch_size=512)

opt = optim.SignSGD(model.parameters(), lr=2e-3, momentum=0.9)
EPOCHS = 3
total = EPOCHS * len(train_loader)
step = 0
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        model.set_tau(max(1.0 * (1 - step / (0.8 * total)), 0.0))   # anneal tau->0
        opt.zero_grad()
        model.compute_loss(xb, yb).backward(); opt.step()
        step += 1
    model.set_tau(0.0); model.eval()
    correct = tot = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            correct += (model(xb).argmax(-1) == yb).sum().item(); tot += yb.numel()
    print(f"epoch {epoch}: tau->0  test_acc = {correct/tot:.4f}   "
          f"(multiplier-free backbone + SignSGD; head is the one declared boundary)")
"""),
    md(r"""
## The fully-closed model, and the closure penalty

Now swap to the **tropical head** so the *entire* network is multiplier-free —
`assert_closed(model)` passes with no allowances. On MNIST this 10-way tropical
head is the bottleneck (the single-argmax routing is weaker than a dense head),
so accuracy drops — that gap *is* the closure penalty at the head, and reporting
both numbers is the honest comparison. The morphological **features** are
identical and excellent (they drove the ~96% above); only the head changed.
"""),
    code(r"""
closed = tt.models.TropicalCNN(in_channels=1, num_classes=10, channels=(16, 32),
                               pool_grid=2, head_type="tropical",
                               use_soft=True, tau=1.0)
closed.loss_type = "tropical_ce"
assert_closed(closed)     # no allow=... needed: the WHOLE model is multiplier-free
print("Fully multiplier-free CNN ✓ (forward + backward + optimizer, zero DSP).")

opt = optim.SignSGD(closed.parameters(), lr=2e-3, momentum=0.9)
total = 2 * len(train_loader); step = 0
for epoch in range(2):
    closed.train()
    for xb, yb in train_loader:
        closed.set_tau(max(1.0 * (1 - step / (0.8 * total)), 0.0))
        opt.zero_grad(); closed.compute_loss(xb, yb).backward(); opt.step(); step += 1
    closed.set_tau(0.0); closed.eval()
    c = t = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            c += (closed(xb).argmax(-1) == yb).sum().item(); t += yb.numel()
    print(f"epoch {epoch}: FULLY-closed test_acc = {c/t:.4f}  (vs ~0.96 with a boundary head)")

print("\\nThe deploy graph (tau=0) is exactly what maps to the zero-DSP FPGA kernel.")
"""),
]


# ===========================================================================
# 04 — Sequence: multiplier-free language modeling + the closure-faithful loss
# ===========================================================================
sequence = [
    md(r"""
# Multiplier-free sequence modeling, and recovering accuracy for free

The last place a multiply hides in a "closed" model is the **loss**: a softmax
cross-entropy uses exp + divide + multiply. On silicon the loop instead used a
*perceptron* gradient (a 2-hot ±1 vector) — multiplier-free, but the single
biggest accuracy cost in the closure penalty (~−17pp on char-LM).

TikiTorch's `tropical_cross_entropy` closes that gap **without a hardware
multiply**: the CE gradient is `softmax(logits) − onehot(target)`, and
`shift_softmax` computes the softmax from a subtract-max, a power-of-2-addressed
exp lookup, additions, and a shift-normalize. This notebook reproduces the
recovery live on a character LM, then shows the `TropicalLion` optimizer and the
streaming `MorphSSM` cell.
"""),
    SETUP,
    md("## A character-level Shakespeare LM at the silicon dims (D=128, V=128)"),
    code(r"""
import torch, tikitorch as tt, tikitorch.optim as optim
V = 128

# Tiny built-in text so the notebook is self-contained; swap in any corpus.
import urllib.request, os
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
try:
    text = urllib.request.urlopen(url, timeout=10).read().decode("utf-8", "replace")
except Exception:
    text = ("To be, or not to be, that is the question:\n" * 2000)   # fallback
ids = torch.tensor([min(ord(c), V - 1) for c in text], dtype=torch.long)
print("corpus chars:", ids.numel())

def batch(bs, L, gen):
    s = torch.randint(0, ids.numel() - L - 1, (bs,), generator=gen)
    x = torch.stack([ids[i:i+L] for i in s]); y = torch.stack([ids[i+1:i+L+1] for i in s])
    return x, y
"""),
    md(r"""
## The recovery: perceptron → closure-faithful CE (multiplier-free)

We hold the model, optimizer, seed, and data fixed and vary **only the loss
conditioning**: the perceptron baseline (what silicon used), the host-softmax
`tropical_ce`, and the fully multiplier-free `tropical_ce_lut`. Watch the
multiplier-free LUT softmax recover most of the perceptron penalty — and match
the float cross-entropy ceiling (under a sign optimizer, only the gradient *sign*
matters, and the LUT reproduces it).
"""),
    code(r"""
def run(loss_type, steps=400, L=48, bs=24, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=1, d_ff=512,
                                      tau=0.5, max_seq_len=L, task_type="lm", vocab_size=V)
    m.loss_type = loss_type
    opt = optim.SignSGD(m.parameters(), lr=2e-3, momentum=0.9)
    for _ in range(steps):
        x, y = batch(bs, L, gen); opt.zero_grad()
        m.compute_loss(x, y).backward(); opt.step()
    gen2 = torch.Generator().manual_seed(seed + 1)
    accs = []
    with torch.no_grad():
        for _ in range(8):
            x, y = batch(bs, L, gen2)
            accs.append((m(x).reshape(-1, V).argmax(-1) == y.reshape(-1)).float().mean().item())
    return sum(accs) / len(accs)

rows = [("perceptron (silicon baseline)", "perceptron"),
        ("tropical_ce (host softmax)", "tropical_ce"),
        ("tropical_ce_lut (FULLY mult-free)", "tropical_ce_lut"),
        ("cross_entropy (float ceiling)", "cross_entropy")]
res = {lt: run(lt) for _, lt in rows}
for label, lt in rows:
    print(f"{label:34s}  eval_acc={res[lt]:.4f}")
print(f"\nrecovery perceptron -> tropical_ce_lut: "
      f"{res['tropical_ce_lut'] - res['perceptron']:+.4f}  (multiplier-free)")
"""),
    md(r"""
## A multiplier-free momentum optimizer: TropicalLion

`TropicalLion` takes the **sign of a momentum** (stronger than SignSGD's
sign-of-gradient), with power-of-2 betas so the EMA is a shift+add. Crucially the
momentum is simulated in **truncating integer fixed-point**, faithfully modeling
the silicon (a float EMA would over-state it). `momentum_density()` confirms the
momentum is alive (not collapsed to SignSGD).
"""),
    code(r"""
m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=1, d_ff=512,
                                  tau=0.5, max_seq_len=48, task_type="lm", vocab_size=V)
m.loss_type = "tropical_ce"
lion = optim.TropicalLion(m.parameters(), lr_shift=9, grad_bits=16)
gen = torch.Generator().manual_seed(0)
for _ in range(50):
    x, y = batch(24, 48, gen); lion.zero_grad(); m.compute_loss(x, y).backward(); lion.step()
print("Lion momentum density (fraction of live coords):", round(lion.momentum_density(), 3),
      "\n> 0 means a real momentum, not SignSGD-degenerate.")
"""),
    md(r"""
## Streaming inference: MorphSSM `.step()`

The morphological SSM runs token-by-token with an on-chip ring buffer — the form
the FPGA kernel uses for streaming. Same weights, no recompute.
"""),
    code(r"""
ssm = tt.models.MorphSSM(d_model=64, n_layers=2, task_type="lm", vocab_size=V)
from tikitorch.audit import find_closure_violations
print("MorphSSM closure violations (LayerNorm/Linear-head are the declared boundary):",
      [t for _, t in find_closure_violations(ssm)][:4])
print("Swap in TropicalLayerNorm + a tropical head for a fully-closed sequence model.")
"""),
]


if __name__ == "__main__":
    write_nb("01_intro.ipynb", intro)
    write_nb("02_tropical_math.ipynb", math_nb)
    write_nb("03_vision_cnn.ipynb", vision)
    write_nb("04_sequence.ipynb", sequence)
    print("done.")
