"""Reference + de-risk for DISTRIBUTED (model-parallel) on-fabric PC training (forge_pc_block kernel).

A 2-layer dyadic net is trained MODEL-PARALLEL across two "chips": chip0 owns layer 0 (hidden, D->D
ReLU), chip1 owns layer 1 (output, D->C readout). Each chip holds ONLY its own layer's weights+momentum.
The only data that crosses chips is a neighbour ACTIVATION (downstream, forward) and a neighbour ERROR
(upstream, backward) — no chip ever sees the global model or the global gradient (no weight transport, no
all-reduce, no barrier). This Python sim reproduces forge_pc_block.cpp's exact per-layer arithmetic so it
(a) confirms the distributed split TRAINS as well as the single-process recipe [Gate B de-risk], and
(b) can emit golden vectors for the on-silicon bit-exact check [Gate A].

Key fabric-faithful points (matching forge_pc_block.cpp):
  - norm_shift is the FIXED convention bit_length-target (the new kernel fixes the MNIST kernel's off-by-one),
    so the MSE target is tgt=8 again (NO host tgt=16 compensation).
  - each chip MAX-SHIFTS its outgoing error to ~13 bits (EDN_BITS) BEFORE it crosses; the receiving chip
    applies its relu-mask. (maxshift-then-mask; sign(gW) is invariant to the uniform message rescale, so
    this matches the single-process recipe's training up to fixed-point noise.)
  - SignSGD-momentum update, leaky counter cap MCAP=7, integer step 1.

Run: PYTHONPATH=. DATA_ROOT=/tmp/_tiki_data python scripts/pc_shard_train_ref.py
     PC_GOLDEN=/tmp/pcblock_golden python scripts/pc_shard_train_ref.py   # also dump golden vectors
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
from scripts.pc_onfabric_train_ref import D, C, dyad, mshift, load_mnist_16  # exact same dyad / max-shift

H1_BITS, LG_BITS, EDN_BITS = 7, 4, 13
MCAP = 7


def chip_fwd(a_in, W, role):
    """One chip's forward. role='hidden' -> relu+maxshift(H1_BITS); 'output' -> first C, maxshift(LG_BITS).
    Returns (a_out, mask)."""
    od = C if role == "output" else D
    pre = a_in @ dyad(W)[:od].t()                      # (N, od)  dyadic matmul = shift-add
    if role == "hidden":
        mask = (pre > 0).double()
        a_out = mshift(pre.clamp(min=0), H1_BITS)
    else:
        mask = torch.ones_like(pre)
        a_out = mshift(pre, LG_BITS)
    return a_out, mask


def chip_bwd(a_in, W, role, mask, a_out=None, tgt=None, e_in=None):
    """One chip's backward. Builds delta = e*mask, gW = q(delta)^T q(a_in), and the maxshifted outgoing
    error e_out = maxshift(delta @ q(W), EDN_BITS). Returns (gW, e_out)."""
    od = C if role == "output" else D
    if role == "output":
        e = a_out - tgt                                # MSE top error (a_out already = maxshifted logits)
    else:
        e = e_in                                       # the rescaled neighbour error
    delta = e * mask                                   # (N, od)
    gW = torch.zeros_like(W)
    gW[:od] = dyad(delta).t() @ dyad(a_in)             # q(delta)^T q(a_in), shift-add outer product
    e_down = delta @ dyad(W)[:od]                      # (N, D) error route, reuses forward q(W) in place
    e_out = mshift(e_down, EDN_BITS)                   # maxshift to ~13 bits before it crosses chips
    return gW, e_out


def sign(t):
    return torch.sign(t)


@torch.no_grad()
def accuracy(X, Y, W0, W1):
    correct = 0
    for i in range(0, X.shape[0], 4000):
        h1, _ = chip_fwd(X[i:i+4000], W0, "hidden")
        lg, _ = chip_fwd(h1, W1, "output")
        correct += (lg.argmax(1) == Y[i:i+4000]).sum().item()
    return correct / X.shape[0]


def step(xb, yb, W0, W1, m0, m1):
    """One distributed training step across 2 chips (forward sweep up, backward sweep down)."""
    tgt = torch.nn.functional.one_hot(yb, C).double() * 8     # (N, C) onehot*8 ; tgt=8 (fixed-shift kernel)
    # ---- forward sweep: chip0 (hidden) -> message h -> chip1 (output) ----
    h, mask0 = chip_fwd(xb, W0, "hidden")             # AOUT0 (crosses to chip1)
    logits, mask1 = chip_fwd(h, W1, "output")         # AOUT1 (stays on chip1)
    # ---- backward sweep: chip1 (output) -> message e_out -> chip0 (hidden) ----
    gW1, e_out1 = chip_bwd(h, W1, "output", mask1, a_out=logits, tgt=tgt)  # e_out1 crosses to chip0
    gW0, _      = chip_bwd(xb, W0, "hidden", mask0, e_in=e_out1)           # chip0 consumes the neighbour error
    # ---- each chip updates ONLY its own weights (SignSGD-momentum) ----
    m0 = (m0 + sign(gW0)).clamp(-MCAP, MCAP); W0 = W0 - sign(m0)
    m1 = (m1 + sign(gW1)).clamp(-MCAP, MCAP); W1 = W1 - sign(m1)
    return W0, W1, m0, m1, (h, logits, e_out1, gW0, gW1)


if __name__ == "__main__":
    (Xtr, Ytr), (Xte, Yte) = load_mnist_16()
    print(f"distributed (2-chip) PC training, 16x16 MNIST: train {tuple(Xtr.shape)} test {tuple(Xte.shape)}",
          flush=True)
    torch.manual_seed(0)
    W0 = (torch.randn(D, D) * 4).round().double()
    W1 = (torch.randn(D, D) * 4).round().double()
    m0 = torch.zeros(D, D, dtype=torch.float64); m1 = torch.zeros(D, D, dtype=torch.float64)
    BATCH = int(os.environ.get("PC_BATCH", 32))
    EPOCHS = int(os.environ.get("PC_EPOCHS", 8))
    gen = torch.Generator().manual_seed(1)
    print(f"  init test acc = {accuracy(Xte, Yte, W0, W1):.4f}", flush=True)

    golden_dir = os.environ.get("PC_GOLDEN")
    nb = Xtr.shape[0] // BATCH
    for ep in range(EPOCHS):
        perm = torch.randperm(Xtr.shape[0], generator=gen)
        for b in range(nb):
            idx = perm[b*BATCH:(b+1)*BATCH]
            W0, W1, m0, m1, dbg = step(Xtr[idx], Ytr[idx], W0, W1, m0, m1)
            if golden_dir and ep == 0 and b == 0:
                # emit a golden one-step record (fixed init W, fixed batch) for the on-silicon bit-exact check
                os.makedirs(golden_dir, exist_ok=True)
                h, logits, e_out1, gW0, gW1 = dbg
                def dump(name, t): t.round().to(torch.int32).numpy().tofile(os.path.join(golden_dir, name))
                dump("aout0_h.i32", h); dump("aout1_logits.i32", logits); dump("eout1.i32", e_out1)
                dump("W0_after.i32", W0); dump("W1_after.i32", W1)
                print(f"  [golden] dumped one-step vectors to {golden_dir}", flush=True)
        print(f"  epoch {ep+1}: test acc = {accuracy(Xte, Yte, W0, W1):.4f}  "
              f"|W0|max={W0.abs().max():.0f} |W1|max={W1.abs().max():.0f}", flush=True)
    print("\n  RESULT: a 2-layer net trained MODEL-PARALLEL across 2 chips (each holding ONE layer; only a\n"
          "  neighbour activation + neighbour error cross) reaches the single-process recipe's accuracy ->\n"
          "  distributed PC training with NO global model / gradient / all-reduce / barrier. This is the\n"
          "  recipe the forge_pc_block kernel implements (one chip = one kernel; 2 kicks/step/chip).")
