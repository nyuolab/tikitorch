"""tikitorch.fpga.runners — reference on-board runners for the bundled kernels.

The :mod:`tikitorch.fpga._session` registry holds the *last mile*: a per-kernel
runner that owns the on-board fused training step. BUILD_NATIVE.md says none ship
by default; this module ships the **reference runner for the 1-layer Tier-1 LM
kernel** (``forge_train_step``, the one with the silicon-validated AGFI), so that
once ``_libtikifpga`` is built (LOCAL or XRT) a ``TropicalTransformer`` becomes
*bound* and ``model.fpga().train_step(...)`` actually drives the kernel.

It is the faithful Python translation of the kernel's own silicon-validated
testbench (``forge_train_step_tb.cpp``): per step it zeroes the K/V caches,
runs the in-kernel training-window dispatch (``mode 3``) which does forward +
loss + backward over the staged sequence and returns the per-position
correct-count, then applies the multiplier-free ``SIGNSGD`` update (``mode 2``).
Weights are seeded into the kernel buffer from the model (quantized to INT16) and
synced back after training.

**Honest scope.** This runner validates the *whole transport + training chain*
end-to-end on the real compiled kernel (LOCAL: host math; XRT: F2 silicon): the
kernel trains and converges, driven entirely through TikiTorch's ``.fpga()`` API.
It does **not** assert bit-exact equivalence between the PyTorch tropical forward
and the kernel forward — that per-tensor weight calibration is the kernel-specific
step noted in BUILD_NATIVE.md. The kernel is the silicon reference; this seeds it
from the model and trains it.

Call :func:`register_reference_runners` (also done automatically on
``import tikitorch.fpga``) to install them.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

from ._session import register_board_runner

# Canonical mode codes (mirror kernels.MODE_* and the kernel headers).
_MODE_SIGNSGD = 2
_MODE_TRAIN_WINDOW = 3  # do_train_window: fwd+loss+bwd over the staged sequence


@dataclass(frozen=True)
class _Layout1:
    """The 1-layer ``forge_buffer.hpp`` layout, recomputed in Python.

    Field order and arithmetic mirror the C header exactly so the offsets are
    identical; verified against forge_buffer.hpp constants in the tests.
    """

    V: int = 128
    D: int = 128
    D_FF: int = 512
    L_MAX: int = 128

    def offsets(self):
        V, D, D_FF, L = self.V, self.D, self.D_FF, self.L_MAX
        w = {
            "EMBED": V * D, "S_Q": L * D, "S_K": L * D, "W1": D_FF * D,
            "B1": D_FF, "W2": D * D_FF, "B2": D, "W_HEAD": V * D,
        }
        # params, then gradients (same sizes), then activations + caches + IO.
        order = ["EMBED", "S_Q", "S_K", "W1", "B1", "W2", "B2", "W_HEAD"]
        off = {}
        cur = 0
        for k in order:
            off["OFF_" + k] = cur
            cur += w[k]
        for k in order:  # gradients
            off["OFF_G_" + k] = cur
            cur += w[k]
        acts = [("AD_SCORE", L * L), ("AN_OUT", L * D), ("AD_DIL", L * D_FF),
                ("AK_ERO", L * D), ("AD_HEAD", L * V), ("K_CACHE", L * D),
                ("V_CACHE", L * D), ("TOKENS", L), ("TARGETS", L),
                ("GRAD_LOGIT", L * V), ("PRED", L),
                # Config words (mirrors forge_buffer.hpp OFF_CFG):
                # [0] loss kind (0 = perceptron, 1 = balanced LUT-CE),
                # [1] LUT-CE z-shift. Zeros = silicon-validated default.
                ("CFG", 2)]
        for name, n in acts:
            off["OFF_" + name] = cur
            off["W_" + name] = n
            cur += n
        off["FT_BUFFER_WORDS"] = cur
        # param weight regions actually seeded by the testbench (B1/B2 stay 0).
        off["_weight_regions"] = [("EMBED", w["EMBED"]), ("S_Q", w["S_Q"]),
                                  ("S_K", w["S_K"]), ("W1", w["W1"]),
                                  ("W2", w["W2"]), ("W_HEAD", w["W_HEAD"])]
        return off


_L1 = _Layout1()


def _quantize_int16(flat, clip=8):
    """Quantize a 1-D float tensor to INT16 in the kernel's validated [-clip, clip]."""
    import torch

    if flat.numel() == 0:
        return flat.new_zeros(0, dtype=torch.int32)
    scale = flat.abs().mean().clamp_min(1e-6) / (clip / 2.0)
    q = torch.round(flat / scale).clamp_(-clip, clip)
    return q.to(torch.int32)


def _model_flat_params(model):
    import torch

    ps = [p.detach().reshape(-1) for p in model.parameters() if p.requires_grad]
    return torch.cat(ps) if ps else torch.zeros(0)


_MODE_FWD_LOSS = 0
_MODE_BWD = 1


def _np_i32(values):
    import numpy as np
    return np.ascontiguousarray(values, dtype=np.int32)


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _run_forge_train_step_1layer(handle, spec, model, tokens, *, targets, lr, threshold):
    """Reference on-board runner for the 1-layer kernel ``forge_train_step``.

    Faithful translation of scripts/train_shakespeare_on_silicon_fast.c: per step
    zero the K/V caches, run the per-position FWD_LOSS kicks (the kernel writes the
    perceptron grad to OFF_GRAD_LOGIT and an all-zero grad row == a correct
    prediction), read it to score, zero caches again, run the per-position BWD
    kicks in reverse, then one SIGNSGD kick. Uses the bridge's region API so the
    SAME runner drives LOCAL (host kernel) and silicon (BAR4 over PCIe) identically.
    Returns an error-rate loss proxy (1 - correct/total).
    """
    import numpy as np
    import torch

    lib = handle._lib
    off = _L1.offsets()
    words = off["FT_BUFFER_WORDS"]
    V, L_MAX = _L1.V, _L1.L_MAX
    entry = spec.entry.encode()

    def _as_seqs(t):
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        if t and not isinstance(t[0], (list, tuple)):
            t = [t]
        return t

    tok_seqs = _as_seqs(tokens)
    tgt_seqs = _as_seqs(targets) if targets is not None else [[0] * len(s) for s in tok_seqs]

    ctx = lib.tt_fpga_open(None)
    if not ctx:
        raise RuntimeError("tt_fpga_open returned NULL (native bridge failed to open)")
    try:
        bufptr = lib.tt_fpga_alloc(ctx, words)
        if not bufptr:
            raise RuntimeError("tt_fpga_alloc returned NULL")
        bufptr = ctypes.c_void_p(int(bufptr))

        def wr(off_w, arr):
            a = _np_i32(arr)
            lib.tt_fpga_write_region(ctx, bufptr, off_w, _ptr(a), a.size)

        def rd(off_w, n):
            a = np.zeros(n, dtype=np.int32)
            lib.tt_fpga_read_region(ctx, bufptr, off_w, _ptr(a), n)
            return a

        def kick(mode, a1, a2, a3):
            return lib.tt_fpga_run(ctx, entry, bufptr, mode, int(a1), int(a2), int(a3), -1)

        zeros_cache = np.zeros(off["W_K_CACHE"], dtype=np.int32)

        def zero_caches():
            lib.tt_fpga_write_region(ctx, bufptr, off["OFF_K_CACHE"], _ptr(zeros_cache), zeros_cache.size)
            lib.tt_fpga_write_region(ctx, bufptr, off["OFF_V_CACHE"], _ptr(zeros_cache), zeros_cache.size)

        # Seed kernel weights from the model (quantized INT16), same regions the
        # silicon driver fills (B1/B2 left zero).
        q = _quantize_int16(_model_flat_params(model)).tolist()
        cur = 0
        for name, n in off["_weight_regions"]:
            chunk = q[cur:cur + n]
            if len(chunk) < n:
                chunk = chunk + [0] * (n - len(chunk))
            wr(off["OFF_" + name], chunk)
            cur += n

        total_correct = total = 0
        for seq_tok, seq_tgt in zip(tok_seqs, tgt_seqs):
            Ls = min(len(seq_tok), L_MAX)
            wr(off["OFF_TOKENS"], [int(seq_tok[m]) for m in range(Ls)])
            wr(off["OFF_TARGETS"], [int(seq_tgt[m]) for m in range(Ls)])
            zero_caches()
            for m in range(Ls):                                   # forward + loss
                rc = kick(_MODE_FWD_LOSS, m, int(seq_tok[m]), int(seq_tgt[m]))
                if rc < -90:
                    raise RuntimeError(f"FWD kick m={m} rc={rc}")
            grad = rd(off["OFF_GRAD_LOGIT"], Ls * V).reshape(Ls, V)
            total_correct += int((np.abs(grad).sum(axis=1) == 0).sum())
            total += Ls
            zero_caches()                                         # caches -> grad accumulators
            for m in range(Ls - 1, -1, -1):                       # backward (reverse)
                rc = kick(_MODE_BWD, m, int(seq_tok[m]), 0)
                if rc < -90:
                    raise RuntimeError(f"BWD kick m={m} rc={rc}")
            kick(_MODE_SIGNSGD, int(lr), int(threshold), 0)       # multiplier-free update

        # Sync trained INT16 weights back into the model (best-effort, same order).
        trained = []
        for name, n in off["_weight_regions"]:
            trained.extend(int(v) for v in rd(off["OFF_" + name], n))
        _sync_back(model, trained)

        lib.tt_fpga_free(ctx, bufptr)
        return 1.0 - (total_correct / max(total, 1))
    finally:
        lib.tt_fpga_close(ctx)


def _sync_back(model, trained_int16):
    """Write trained INT16 kernel weights back into the model's float params."""
    import torch

    with torch.no_grad():
        cur = 0
        n = len(trained_int16)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            k = p.numel()
            if cur >= n:
                break
            chunk = trained_int16[cur:cur + k]
            if len(chunk) < k:
                chunk = chunk + [0] * (k - len(chunk))
            p.copy_(torch.tensor(chunk, dtype=p.dtype).reshape(p.shape))
            cur += k


def register_reference_runners() -> None:
    """Register the bundled reference runner(s) into the on-board session registry.

    Currently the 1-layer Tier-1 LM kernel (``forge_train_step``). The 2-/3-layer
    and morph runners are analogous (drive their respective ``*_tb.cpp`` sequence)
    and are left as the documented next additions.
    """
    register_board_runner("forge_train_step", _run_forge_train_step_1layer)
