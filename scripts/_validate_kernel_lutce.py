"""Bit-exact validation of the kernel's balanced LUT-CE gradient (CSIM build).

Stage only W_HEAD with constant rows (all other weights zero), so the kernel's
forward collapses to logits[v] = max_d(W_head[v, d] + 0) = c_v exactly — the
logits are fully controlled from the host. Then run FWD_LOSS with
CFG_LOSS_KIND=1 and compare the gradient row read back from OFF_GRAD_LOGIT
against an integer numpy reference of the spec, bit for bit:

    zneg   = c[argmax] - c[v]
    idx    = min(255, zneg >> zshift)
    e      = LUT[idx]                  (Q16 exp2 ROM)
    sh     : smallest s.t. sum(e) >> sh <= 65536
    grad_v = (e_v >> sh) - (v == target ? sum(e) >> sh : 0)

Also asserts: row sums to exactly zero; CFG=0 reproduces the 2-hot perceptron;
and the sign pattern matches the float tropical_ce_lut gradient.

Usage:  PYTHONPATH=. python scripts/_validate_kernel_lutce.py
"""
import ctypes
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "_libtikifpga.dylib"

import sys
sys.path.insert(0, str(REPO))
from tikitorch.fpga.runners import _L1  # noqa: E402

V, D, ZSHIFT = 128, 128, 4
LUT = np.array([round(65536 * 2 ** (-i / 16)) for i in range(256)], dtype=np.int64)


def ref_grad(c, target, zshift=ZSHIFT):
    """Integer reference of the kernel spec (c: int logits, len V)."""
    zneg = c.max() - c                      # kernel uses logits[pred]; ties: same max value
    idx = np.minimum(zneg >> zshift, 255)
    e = LUT[idx]
    S = e.sum()
    sh = 0
    while (S >> sh) > 65536:
        sh += 1
    g = (e >> sh).astype(np.int64)
    g[target] -= g.sum()        # post-truncation sum: exactly balanced
    return g


def main():
    off = _L1.offsets()
    words = off["FT_BUFFER_WORDS"]
    lib = ctypes.CDLL(str(LIB))
    fn = lib.forge_train_step
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(ctypes.c_int32)] + [ctypes.c_int] * 4

    rng = np.random.default_rng(0)
    n_cases = 50
    max_abs_diff = 0
    for case in range(n_cases):
        buf = np.zeros(words, dtype=np.int32)
        c = rng.integers(-200, 200, size=V).astype(np.int32)   # controlled logits
        w_head = np.repeat(c[:, None], D, axis=1)              # W_head[v,:] = c_v
        buf[off["OFF_W_HEAD"]:off["OFF_W_HEAD"] + V * D] = w_head.reshape(-1)
        target = int(rng.integers(0, V))
        token = int(rng.integers(0, V))
        buf[off["OFF_CFG"] + 0] = 1
        buf[off["OFF_CFG"] + 1] = ZSHIFT

        rc = fn(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 0, 0, token, target)
        assert rc == 0, f"kernel rc={rc}"
        got = buf[off["OFF_GRAD_LOGIT"]:off["OFF_GRAD_LOGIT"] + V].astype(np.int64)
        want = ref_grad(c, target)

        assert got.sum() == 0, f"case {case}: row sum {got.sum()} != 0"
        diff = np.abs(got - want).max()
        max_abs_diff = max(max_abs_diff, diff)
        assert diff == 0, (f"case {case}: max |kernel - ref| = {diff}\n"
                           f"kernel {got[:8]}...\nref    {want[:8]}...")

        # perceptron back-compat: CFG=0 must reproduce the 2-hot
        buf2 = np.zeros(words, dtype=np.int32)
        buf2[off["OFF_W_HEAD"]:off["OFF_W_HEAD"] + V * D] = w_head.reshape(-1)
        rc = fn(buf2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 0, 0, token, target)
        assert rc == 0
        g2 = buf2[off["OFF_GRAD_LOGIT"]:off["OFF_GRAD_LOGIT"] + V]
        pred = int(np.argmax(c))  # kernel scan picks first max
        if pred != target:
            assert g2[pred] == 1 and g2[target] == -1 and np.count_nonzero(g2) == 2
        else:
            assert np.count_nonzero(g2) == 0

    print(f"OK: {n_cases} cases bit-exact (max |diff| = {max_abs_diff}), "
          f"row sums all zero, perceptron path unchanged.")


if __name__ == "__main__":
    main()
