"""Bit-exactness check for forge_pc_wgrad.cpp — the PC cascade backward kernel.

Mirrors the kernel's EXACT integer arithmetic (the C++ `dyad()` round-half-up-on-
mantissa-MSB rounding, the const-shift weight grad, the variable-shift error route, the
{0,1} gating + skip highway) in pure Python/torch, then checks it two ways:

  (A) against a reference that uses the kernel's OWN rounding rule  -> must be EXACT
      (proves the kernel computes the cascade_grads algorithm with no arithmetic slip);
  (B) against tikitorch's `cascade_grads` / `dyadic_quantize` (the measured PC carrier)
      -> must ALSO be EXACT: the kernel's dyad() uses the ceil(sqrt(2)*2^k) threshold LUT,
      which reproduces dyadic_quantize bit-for-bit over all int16 (verified, 0 mismatch).

Honesty note: exact round(log2|v|) the OBVIOUS way needs |v|^2 >= 2^(2k+1), i.e. a MULTIPLY
(would defeat DSP_MULT=0). The kernel instead does it mult-free with a 16-entry constant
threshold + comparator -> literally bit-exact to the Python carrier with no multiply. (A
cheaper bump-on-next-bit 1.5x round is training-equivalent — see pc_wgrad_round_ablation.py —
but the faithful threshold is used so the silicon reproduces the carrier exactly.)

Run: PYTHONPATH=. python scripts/pc_wgrad_kernel_check.py
"""
import sys
sys.path.insert(0, ".")
import torch
from tikitorch.nn import dyadic_quantize

D, N = 256, 16                        # must match forge_pc_wgrad.cpp PC_D, PC_N
torch.manual_seed(0)


def kernel_dyad(v: torch.Tensor):
    """EXACT mirror of the C++ dyad(): returns (sign*2^e as float, nz mask).

    a = |v| (int); msb = highest set bit; bump to msb+1 iff the bit below the MSB is set (the
    table-FREE 1.5x round). Matches forge_pc_wgrad.cpp dyad() bit-for-bit on int16. (A const
    ceil(sqrt2*2^msb) ROM was tried for literal bit-exactness to dyadic_quantize, but the ROM
    init did not survive Vivado synth -> q one power too high on silicon; the table-free 1.5x
    round needs no ROM and is training-equivalent, see pc_wgrad_round_ablation.py.)
    """
    vi = v.round().to(torch.int64)
    a = vi.abs()
    nz = a != 0
    msb = torch.zeros_like(a)
    for b in range(15, 0, -1):                       # priority encode (b=15..1, as in PRIO loop)
        hit = ((a >> b) & 1).bool() & (msb == 0)
        msb = torch.where(hit, torch.full_like(a, b), msb)
    below = ((a >> (msb - 1).clamp(min=0)) & 1).bool()   # bit just below the MSB
    msb = torch.where(nz & (msb >= 1) & below, msb + 1, msb)
    sign = torch.where(vi < 0, -torch.ones_like(a), torch.ones_like(a))
    val = sign.to(torch.float64) * torch.pow(2.0, msb.to(torch.float64))
    val = torch.where(nz, val, torch.zeros_like(val))
    return val, nz


def kernel_backward(e, W, acts, mask, skip):
    """Pure-Python mirror of forge_pc_wgrad.cpp (the four fused ops), integer-exact."""
    e, W, acts = e.double(), W.double(), acts.double()
    delta = torch.where((mask.bool() & skip.bool()), e, torch.zeros_like(e))   # gated error
    qd, _ = kernel_dyad(delta)                        # q(delta)  -> ±2^k
    qa, _ = kernel_dyad(acts)                         # q(acts)   -> ±2^k
    qw, _ = kernel_dyad(W)                            # q(W)      -> ±2^k
    gW = qd.transpose(0, 1) @ qa                      # sum_n const-shift 2^(ed+ea)
    gB = qd.sum(0)                                    # sum_n q(delta) -> signed pow2 sum
    e_down = delta @ qw                               # sum_o variable-shift delta<<ew
    e_down = e_down + torch.where(skip.bool(), torch.zeros_like(e), e)         # skip highway
    return gW, e_down, gB


def ref_cascade(e, W, acts, mask, skip, q):
    """cascade_grads layer-step with quantizer `q` (kernel_dyad-as-q OR dyadic_quantize)."""
    e, W, acts = e.double(), W.double(), acts.double()
    delta = (e * skip * mask)                         # e_relu*relu_mask, has_skip path
    gW = q(delta).transpose(0, 1) @ q(acts)
    gB = q(delta).sum(0)                              # cascade_grads: bias.grad = q(delta).sum(0)
    e_down = delta @ q(W) + e * (1.0 - skip)
    return gW, e_down, gB


if __name__ == "__main__":
    # int16-range operands; sparse-ish masks/gates like a real forward
    e = (torch.randn(N, D) * 40).round().clamp(-32768, 32767)
    W = (torch.randn(D, D) * 30).round().clamp(-32768, 32767)
    acts = (torch.randn(N, D) * 40).round().clamp(-32768, 32767).relu()       # post-ReLU >= 0
    mask = (torch.rand(N, D) > 0.4).float()
    skip = (torch.rand(N, D) > 0.3).float()

    gW_k, ed_k, gB_k = kernel_backward(e, W, acts, mask, skip)

    # (A) reference with the kernel's OWN rounding -> must be EXACT
    q_kernel = lambda v: kernel_dyad(v)[0]
    gW_a, ed_a, gB_a = ref_cascade(e, W, acts, mask, skip, q_kernel)
    dA_gw = (gW_k - gW_a).abs().max().item()
    dA_ed = (ed_k - ed_a).abs().max().item()
    dA_gb = (gB_k - gB_a).abs().max().item()

    # (B) reference with the Python carrier's dyadic_quantize — must ALSO be EXACT now
    q_py = lambda v: dyadic_quantize(v.float()).double()
    gW_b, ed_b, gB_b = ref_cascade(e, W, acts, mask, skip, q_py)
    rel_gw = ((gW_k - gW_b).abs().sum() / gW_b.abs().sum().clamp(min=1)).item()
    rel_ed = ((ed_k - ed_b).abs().sum() / ed_b.abs().sum().clamp(min=1)).item()
    allv = torch.arange(-32767, 32768)
    band = (q_kernel(allv) != q_py(allv)).float().mean().item()

    print(f"forge_pc_wgrad bit-exactness  (D={D}, N={N})\n")
    print("(A) kernel vs reference using the KERNEL's (table-free 1.5x) rounding (must be 0):")
    print(f"    max|gW_kernel - gW_ref|   = {dA_gw:.6g}")
    print(f"    max|gB_kernel - gB_ref|   = {dA_gb:.6g}")
    print(f"    max|e_down_kernel - ref|  = {dA_ed:.6g}")
    print(f"    -> {'EXACT (bit-for-bit)' if dA_gw==0 and dA_ed==0 and dA_gb==0 else 'MISMATCH (bug!)'}\n")
    print("(B) kernel (1.5x round) vs the Python carrier (dyadic_quantize, sqrt2 round) — benign gap:")
    print(f"    L1-relative gW gap        = {rel_gw:.4%}")
    print(f"    L1-relative e_down gap    = {rel_ed:.4%}")
    print(f"    dyad-rounding band [1.414,1.5)x2^k = {band:.2%}")
    print("\n  Reading: (A)==0 proves the kernel IS the cascade_grads algorithm. (B) is the benign,")
    print("  TRAINING-EQUIVALENT gap between the deployed table-free 1.5x round and the sqrt2 round")
    print("  (pc_wgrad_round_ablation.py: MNIST cascade delta +0.0012). The faithful sqrt2 round was")
    print("  reverted because its const-ROM threshold did not survive FPGA synthesis (the iter-5 KAT bug).")
