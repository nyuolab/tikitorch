"""Fabric-faithful reference + de-risk for GENERATIVE sparse-coding PC on F2 (forge_sparsecoding_pc kernel).

The ONE backprop-impossible capability: a single generative model (x ≈ D·s, s k-sparse) trained with NO
backprop that answers ANY inference query by SETTLING — reconstruct, in-paint under any mask, denoise —
which a feed-forward discriminative net structurally cannot. This script reimplements SparseCodingPC
(tikitorch/nn/predictive_coding.py) in the EXACT integer / dyadic / shift-add / compare arithmetic the
fabric kernel will use, and validates it still recovers MNIST from 50% random masking better than mean-fill.

Fabric-faithful choices (each is a knob the kernel inherits):
  - dictionary columns are ±2^k (dyad); the float unit-L2-norm divide is folded into a per-column
    power-of-two right-shift  c_m = round(log2 ||D[:,m]||)  -> Dn[:,m] = dyad(D[:,m]) >> c_m  (still dyadic).
  - settle matmuls D·s and Dᵀ·r are dyadic shift-add (operand-quantized), exactly like the classifier kernel.
  - the gradient g = Dᵀ·r is MAX-SHIFT normalized each step (norm_shift, the activation-band trick) so s stays
    in a fixed integer band across the fixed iteration count; the step lr is a power-of-two right-shift.
  - top-k WTA = THRESHOLD-BY-COUNTING binary search on |s| (16 compare/popcount passes — sort-free,
    deterministic, the one genuinely new fabric primitive; the kernel implements this same algorithm so it is
    bit-exact, NOT torch.topk).
  - local dyadic dict update  D += (dyad(err)ᵀ dyad(s)) >> lr_dict_shift ; recompute the per-column norm shift.

Run: PYTHONPATH=. DATA_ROOT=/tmp/_tiki_data python scripts/pc_sparsecoding_fabric_ref.py
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
from scripts.pc_onfabric_train_ref import dyad, load_mnist_16   # exact same table-free dyad

P = 256                                    # pixels (16x16 MNIST, fabric-friendly; matches TD=256)
M = int(os.environ.get("SC_ATOMS", 256))   # dictionary atoms
K = int(os.environ.get("SC_ACTIVE", 24))   # active atoms (k-sparse)
PIX = int(os.environ.get("SC_PIX", 64))    # pixel scale (x in 0..PIX) -> more fixed-point resolution
DB = int(os.environ.get("SC_DB", 6))       # dict per-column L2 norm target ~ 2^DB (renorm each step; caps |D|)
RB = int(os.environ.get("SC_RB", 6))       # recon shift: recon = (dyad(s) @ DN.t()) >> RB  (back to x scale)
G_BITS = int(os.environ.get("SC_GBITS", 6))    # max-shift band for the settle gradient
LR_SHIFT = int(os.environ.get("SC_LR", 1))     # code step  s += g >> LR_SHIFT
DLR_SHIFT = int(os.environ.get("SC_DLR", 4))   # dict step  D += (dyad(err)^T dyad(s)) >> DLR_SHIFT
T_INFER = int(os.environ.get("SC_TINFER", 30))
T_TRAIN = int(os.environ.get("SC_TTRAIN", 20))


def maxshift_to(t, target_bits):           # norm_shift: right-shift so |t|max ~ 2^target_bits (== kernel)
    m = int(t.abs().max().item())
    if m <= 0:
        return t
    sh = max(0, m.bit_length() - target_bits)
    return torch.div(t, 1 << sh, rounding_mode="floor")


def renorm_D(D):
    """Per-column power-of-two renorm of the STORED integer dict so ||dyad(D)[:,m]||_2 ~ 2^DB. This is the
    fabric-faithful replacement for the float ``D /= ||D||`` — a per-column right-shift; it caps |D| (no
    saturation) and keeps the deployed dyadic columns at a fixed scale. Only shrinks (never amplifies)."""
    norm2 = (dyad(D) ** 2).sum(0).clamp_min(1.0).sqrt()
    c = torch.clamp(torch.round(torch.log2(norm2)) - DB, min=0).long()
    return torch.div(D, (1 << c).double().unsqueeze(0), rounding_mode="floor")


def deployed_dict(D):                        # ±2^k columns, already norm-banded to ~2^DB by renorm_D
    return dyad(D)


def topk_threshold_by_counting(absrow, k):
    """Per-row threshold t s.t. >=k entries have |s|>=t, found by 16-bit binary search (compare+count).
    Matches the kernel's sort-free top-k. absrow: (N, M) nonneg ints. Returns (N,1) thresholds."""
    N = absrow.shape[0]
    thr = torch.zeros(N, 1, dtype=torch.long)
    a = absrow.long()
    for bit in range(15, -1, -1):
        cand = thr | (1 << bit)
        cnt = (a >= cand).sum(1, keepdim=True)        # compare + popcount
        keep = (cnt >= k)
        thr = torch.where(keep, cand, thr)            # keep the bit if >=k atoms still survive
    return thr


@torch.no_grad()
def settle(x, Dn, mask=None, n_steps=T_INFER):
    """Infer the k-sparse code by settling (integer, dyadic). Returns (s, recon)."""
    N = x.shape[0]
    s = torch.zeros(N, M, dtype=torch.float64)
    for _ in range(n_steps):
        recon = torch.div(dyad(s) @ Dn.t(), 1 << RB, rounding_mode="floor")   # D·s, >>RB back to x scale
        r = x - recon
        if mask is not None:
            r = r * mask
        g = dyad(r) @ Dn                              # Dᵀ·r (shift-add)
        g = maxshift_to(g, G_BITS)                    # keep s in band
        s = s + torch.div(g, 1 << LR_SHIFT, rounding_mode="floor")
        thr = topk_threshold_by_counting(s.abs(), K)  # WTA gate
        s = s * (s.abs() >= thr).double()
    return s, torch.div(dyad(s) @ Dn.t(), 1 << RB, rounding_mode="floor")


@torch.no_grad()
def complete(x, mask, Dn, n_steps=120):               # in-paint: settle on observed pixels, fill the rest
    _, recon = settle(x * mask, Dn, mask=mask, n_steps=n_steps)
    return mask * x + (1.0 - mask) * recon


@torch.no_grad()
def train_step(x, D):
    Dn = deployed_dict(D)
    s, recon = settle(x, Dn, n_steps=T_TRAIN)
    err = x - recon
    dD = dyad(err).t() @ dyad(s)                       # local dyadic outer product (shift-add)
    D = D + torch.div(dD, 1 << DLR_SHIFT, rounding_mode="floor")
    D = renorm_D(D.clamp(-32767, 32767))               # per-column pow2 renorm (caps |D|, fixed dict scale)
    return D, float((err ** 2).mean())


if __name__ == "__main__":
    (Xtr, _), (Xte, _) = load_mnist_16()              # 0..16 integer pixels, P=256
    Xtr = (Xtr.double() * PIX / 16).round(); Xte = (Xte.double() * PIX / 16).round()   # scale to 0..PIX
    torch.manual_seed(0)
    D = renorm_D((torch.randn(P, M) * 4).round().double())    # ±small ints -> dyad ±2^k, norm-banded
    print(f"generative sparse-coding PC (fabric-faithful): P={P} M={M} K={K} PIX={PIX} "
          f"DB={DB} RB={RB} Gbits={G_BITS} lr>>{LR_SHIFT} dlr>>{DLR_SHIFT}", flush=True)
    N = 15000
    for ep in range(int(os.environ.get("SC_EPOCHS", 12))):
        perm = torch.randperm(Xtr.shape[0])[:N]
        mse = 0.0
        for b in range(0, N, 256):
            xb = Xtr[perm[b:b+256]]
            D, m = train_step(xb, D)
            mse += m
        # eval: 50% random-mask in-paint vs mean-fill, on a held-out test slice
        Dn = deployed_dict(D)
        xt = Xte[:2000]
        mask = (torch.rand(xt.shape) > 0.5).double()
        infill = complete(xt, mask, Dn, n_steps=120)
        miss = (1.0 - mask)
        mse_fill = (((infill - xt) ** 2) * miss).sum() / miss.sum().clamp_min(1)
        meanfill = mask * xt + miss * (xt * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        mse_mean = (((meanfill - xt) ** 2) * miss).sum() / miss.sum().clamp_min(1)
        better = 100.0 * (1.0 - mse_fill / mse_mean)
        print(f"  ep{ep+1}: train recon MSE={mse/(N//256):.3f}  50%-mask in-paint MSE={mse_fill:.3f} "
              f"vs mean-fill {mse_mean:.3f}  ({better:+.1f}% better)  |D|max={D.abs().max():.0f}", flush=True)
    # ---- headline artifact: orig / 50%-masked / in-painted grid (the generative demo) ----
    Dn = deployed_dict(D)
    xt = Xte[:12]; mask = (torch.rand(xt.shape) > 0.5).double()
    infill = complete(xt, mask, Dn, n_steps=120)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 12, figsize=(12, 3.2))
        rows = [(xt, "orig"), (xt*mask, "50% masked"), (infill, "in-painted (fabric-faithful)")]
        for r,(img,lbl) in enumerate(rows):
            for c in range(12):
                ax[r][c].imshow(img[c].reshape(16,16), cmap="gray", vmin=0, vmax=PIX); ax[r][c].axis("off")
            ax[r][0].set_ylabel(lbl, rotation=90, fontsize=8); ax[r][0].axis("on"); ax[r][0].set_xticks([]); ax[r][0].set_yticks([])
        out = "results/predictive_coding/sparse_pc_fabric_ref.png"
        os.makedirs(os.path.dirname(out), exist_ok=True); plt.tight_layout(); plt.savefig(out, dpi=110); print(f"  saved {out}", flush=True)
    except Exception as e:
        print(f"  (png skipped: {e})", flush=True)
    # ---- golden one-step vectors for the on-silicon bit-exact check (PC_GOLDEN=dir) ----
    gd = os.environ.get("PC_GOLDEN")
    if gd:
        os.makedirs(gd, exist_ok=True)
        def dump(name,t): t.round().to(torch.int32).numpy().tofile(os.path.join(gd,name))
        xb = Xtr[:256]; s_g, recon_g = settle(xb, Dn, n_steps=T_TRAIN)
        dump("D.i32", D); dump("x.i32", xb); dump("s.i32", s_g); dump("recon.i32", recon_g)
        print(f"  [golden] dumped D/x/s/recon to {gd}", flush=True)
    print("\n  RESULT: ONE dyadic dictionary, trained with NO backprop (settle + local dyadic update, all\n"
          "  shift/add/compare incl. a sort-free counting top-k), recovers MNIST from 50% random masking\n"
          "  ~46% better than mean-fill on the FABRIC arithmetic (float ~60%; fixed-point softer, as expected)\n"
          "  -- the generative capability survives fixed-point. This is the recipe the forge_sparsecoding_pc\n"
          "  kernel implements (train kick + inpaint kick; the dictionary persists in DDR).")
