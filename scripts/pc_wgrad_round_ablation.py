"""Does the forge_pc_wgrad kernel's cheap hardware dyadic round move the RESULT?

The kernel rounds at 1.5x (bump on mantissa MSB — the standard cheap HW rule, exact in
mult-free logic); tikitorch's `dyadic_quantize` rounds at sqrt(2) (round-half-in-log, which
needs a multiply, so silicon can't use it). pc_wgrad_kernel_check showed the per-step
gradient differs ~13% in L1 but stays cosine ~0.97 / sign ~96%. This asks the decisive
question: trained END-TO-END at depth on real MNIST, does the silicon's round change test
accuracy beyond seed noise? Patches dyadic_quantize to the kernel's 1.5x rule globally (so
BOTH the DyadicLinear forward AND the cascade backward use the silicon round, exactly as the
fabric would) and compares cascade test accuracy, sqrt(2) vs 1.5x, across depths/seeds.

Run: PYTHONPATH=. python scripts/pc_wgrad_round_ablation.py
"""
import sys, math
sys.path.insert(0, ".")
import torch
import tikitorch.nn as ttnn
from scripts.pc_scale_mnist import load_dataset, make_dims, run_one

_orig_dq = ttnn.dyadic_quantize


def dq_15x(w: torch.Tensor) -> torch.Tensor:
    """dyadic_quantize with the KERNEL's 1.5x round: q(v)=sign(v)*2^(e+[m>=1.5])."""
    a = w.abs()
    e = torch.floor(torch.log2(a.clamp(min=1e-30)))
    m = a / torch.pow(2.0, e)                       # mantissa in [1,2)
    e = e + (m >= 1.5).to(e.dtype)                  # bump at 1.5x (the HW rule)
    out = torch.sign(w) * torch.pow(2.0, e)
    return torch.where(a > 0, out, torch.zeros_like(out))


def patch(fn):
    ttnn.dyadic_quantize = fn
    # predictive_coding imports it at call-scope from tikitorch.nn, so the module attr is enough;
    # patch the package-level name everything resolves through.
    import tikitorch.nn.predictive_coding as pc
    if hasattr(pc, "dyadic_quantize"):
        pc.dyadic_quantize = fn


if __name__ == "__main__":
    device = torch.device("cpu")
    DATA = "/tmp/_tiki_data"
    train, test = load_dataset("mnist", DATA, device)
    DEPTHS, SEEDS, EPOCHS = [8, 16], [0, 1], 3

    # sanity: the two rounds really do differ, and dq_15x matches the kernel's integer rule
    t = torch.tensor([1.45, 2.83, 5.7, 11.4])
    print("round check  v=[1.45,2.83,5.7,11.4]")
    print(f"  sqrt(2) dyadic_quantize : {_orig_dq(t).tolist()}")
    print(f"  1.5x kernel round       : {dq_15x(t).tolist()}\n")

    rows = []
    for depth in DEPTHS:
        dims = make_dims(train[0].shape[1], 256, depth)
        for seed in SEEDS:
            patch(_orig_dq)
            a = run_one("cascade", dims, train, test, seed=seed, epochs=EPOCHS,
                        batch=128, lr=2e-3, n_sweeps=0, residual=True, device=device)
            patch(dq_15x)
            b = run_one("cascade", dims, train, test, seed=seed, epochs=EPOCHS,
                        batch=128, lr=2e-3, n_sweeps=0, residual=True, device=device)
            patch(_orig_dq)
            rows.append((depth, seed, a["test_acc"], b["test_acc"]))
            print(f"  d{depth:<3d} s{seed}  sqrt2 {a['test_acc']:.4f}  |  1.5x {b['test_acc']:.4f}  "
                  f"|  delta {b['test_acc']-a['test_acc']:+.4f}", flush=True)

    deltas = [r[3] - r[2] for r in rows]
    md = sum(deltas) / len(deltas)
    mx = max(abs(d) for d in deltas)
    print(f"\n  mean delta (1.5x - sqrt2) = {md:+.4f}   max|delta| = {mx:.4f}")
    print("  Reading: if |delta| is within seed noise (~<=0.01), the silicon's cheap round does")
    print("  NOT move the measured cascade result -> the kernel faithfully trains the carrier.")
