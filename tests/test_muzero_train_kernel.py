"""Bit-exact host harness for the dyadic MuZero ON-CHIP TRAINING kernel
(tikitorch/hardware/hls/forge_kernels/forge_train_step_muzero_train.cpp).

Compiles the training kernel to a shared lib with g++ (CSIM shim, no Vitis), runs the
actual C training step via ctypes for several SignSGD steps, and compares the ENTIRE
buffer (every weight, bias, output, and the loss) BIT-FOR-BIT against an independent
Python integer mirror of the same multiplier-free rule. This pins the zero-DSP backward
(shift grad-input, sign-product weight grads, SignSGD on int16 shadows) to a precise spec
— any indexing/shift/saturation/ordering bug shows up as a mismatch. Skipped if g++ is
unavailable. (Convergence is covered by the CSIM testbench; this is exactness.)
"""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pytest

# Builds + ctypes-loads a native HLS kernel in-process: a kernel bug can segfault the
# whole pytest run, and the kernel math is already gated by the dedicated `csim` CI job.
# So this is deselected from the default `pytest -q` (run explicitly with `-m kernel`).
pytestmark = pytest.mark.kernel

HLS = os.path.join(os.path.dirname(__file__), "..", "tikitorch", "hardware", "hls")
KERNELS = os.path.join(HLS, "forge_kernels")


_ISOLATED_CHILD = "TIKITORCH_MUZERO_TRAIN_KERNEL_CHILD"


def _run_isolated(test_name):
    if shutil.which("g++") is None:
        pytest.skip("g++ not available for CSIM build")
    env = os.environ.copy()
    env[_ISOLATED_CHILD] = "1"
    result = subprocess.run(
        [sys.executable, __file__, test_name],
        capture_output=True,
        text=True,
        close_fds=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _build_kernel():
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ not available for CSIM build")
    so = os.path.join(tempfile.mkdtemp(), "libmuzero_train.so")
    src = os.path.join(KERNELS, "forge_train_step_muzero_train.cpp")
    r = subprocess.run([compiler, "-O2", "-std=c++17", "-DCSIM_ONLY", "-shared", "-fPIC",
                        f"-I{KERNELS}", f"-I{HLS}", src, "-o", so],
                       capture_output=True, text=True, close_fds=False)
    if r.returncode != 0:
        pytest.fail(f"training kernel CSIM build failed:\n{r.stderr}")
    lib = ctypes.CDLL(so)
    lib.forge_train_step_muzero_train.argtypes = [ctypes.POINTER(ctypes.c_int32)] + [ctypes.c_int] * 4
    lib.forge_muzero_train_buffer_words.restype = ctypes.c_int
    lib.forge_muzero_train_layout.argtypes = [ctypes.POINTER(ctypes.c_int)]
    return lib


def _layout(lib):
    arr = (ctypes.c_int * 23)()
    lib.forge_muzero_train_layout(arr)
    keys = ["OBS", "L", "A", "DR", "DP",
            "W_R1", "W_R2", "W_P1", "W_P2", "W_PH", "W_VH",
            "B_R1", "B_R2", "B_P1", "B_P2", "B_PH", "B_VH",
            "OBS_OFF", "PTGT", "VTGT", "POL", "VAL", "LOSS"]
    return dict(zip(keys, list(arr)))


# ---- exact Python mirror of forge_train_step_muzero_train.cpp (integer arithmetic) ----

def _sat16(a):
    return max(-32768, min(32767, int(a)))


def _trunc_div(s, n):                              # C integer division truncates toward zero
    q = abs(int(s)) // n
    return -q if s < 0 else q


def _isign(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _sgnprod(a, b):
    sa, sb = _isign(a), _isign(b)
    return 0 if (sa == 0 or sb == 0) else (1 if sa == sb else -1)


def _lrstep(lr, g):
    return lr if g > 0 else (-lr if g < 0 else 0)


def _nearest_pow2(w):
    if w == 0:
        return 0, 0
    s = 1 if w > 0 else -1
    a = w if w > 0 else -w
    hi, t = 0, a
    while t > 1:
        t >>= 1
        hi += 1
    lo = 1 << hi
    if ((a - lo) << 1) >= lo:                      # round to nearest pow2 (shift-compare)
        hi += 1
    return hi, s


def _dyq(e, s, v):
    if s == 0:
        return 0
    r = (v << e) if e >= 0 else (v >> (-e))        # arithmetic shift (Py >> floors == C arith)
    return r if s > 0 else -r


def _fwd_linear(buf, woff, boff, x, OUT, IN):
    y = []
    for o in range(OUT):
        acc = int(buf[boff + o])
        for i in range(IN):
            e, s = _nearest_pow2(int(buf[woff + o * IN + i]))
            acc += _dyq(e, s, int(x[i]))
        y.append(_sat16(acc))
    return y


def _bwd_input(buf, woff, g_out, OUT, IN):
    g_in = []
    for i in range(IN):
        g = 0
        for o in range(OUT):
            e, s = _nearest_pow2(int(buf[woff + o * IN + i]))
            g += _dyq(e, s, int(g_out[o]))
        g_in.append(g)
    return g_in


def _sgd_layer(buf, woff, boff, g_out, x, OUT, IN, lr):
    for o in range(OUT):
        go = _isign(g_out[o])
        for i in range(IN):
            idx = woff + o * IN + i
            buf[idx] = _sat16(int(buf[idx]) - _lrstep(lr, _sgnprod(go, int(x[i]))))
        buf[boff + o] = _sat16(int(buf[boff + o]) - _lrstep(lr, go))


def _reference_train_step(buf, L, lr):
    """Mutate buf in place exactly as forge_train_step_muzero_train.cpp mode 2 does."""
    OBS, LAT, A, DR, DP = L["OBS"], L["L"], L["A"], L["DR"], L["DP"]
    obs = [int(buf[L["OBS_OFF"] + i]) for i in range(OBS)]

    # forward (save activations)
    pre = _fwd_linear(buf, L["W_R1"], L["B_R1"], obs, DR, OBS)
    zr1 = [v if v > 0 else 0 for v in pre]
    hraw = _fwd_linear(buf, L["W_R2"], L["B_R2"], zr1, LAT, DR)
    mean = _trunc_div(sum(hraw), LAT)
    h = [_sat16(v - mean) for v in hraw]
    pre = _fwd_linear(buf, L["W_P1"], L["B_P1"], h, DP, LAT)
    zp1 = [v if v > 0 else 0 for v in pre]
    z = _fwd_linear(buf, L["W_P2"], L["B_P2"], zp1, LAT, DP)
    pol = _fwd_linear(buf, L["W_PH"], L["B_PH"], z, A, LAT)
    val = _fwd_linear(buf, L["W_VH"], L["B_VH"], z, 1, LAT)
    for o in range(A):
        buf[L["POL"] + o] = pol[o]
    buf[L["VAL"]] = val[0]

    # L1 loss + output grads
    loss = 0
    gpol = []
    for o in range(A):
        d = pol[o] - int(buf[L["PTGT"] + o])
        loss += abs(d)
        gpol.append(_isign(d))
    dv = val[0] - int(buf[L["VTGT"]])
    loss += abs(dv)
    gval = _isign(dv)
    buf[L["LOSS"]] = loss

    # backward: grad-input with CURRENT weights, then SignSGD-update (exact C order)
    gz = []
    for i in range(LAT):
        g = 0
        for o in range(A):
            e, s = _nearest_pow2(int(buf[L["W_PH"] + o * LAT + i]))
            g += _dyq(e, s, gpol[o])
        e, s = _nearest_pow2(int(buf[L["W_VH"] + i]))
        g += _dyq(e, s, gval)
        gz.append(g)
    _sgd_layer(buf, L["W_PH"], L["B_PH"], gpol, z, A, LAT, lr)
    _sgd_layer(buf, L["W_VH"], L["B_VH"], [gval], z, 1, LAT, lr)

    gzp1 = _bwd_input(buf, L["W_P2"], gz, LAT, DP)
    _sgd_layer(buf, L["W_P2"], L["B_P2"], gz, zp1, LAT, DP, lr)
    for j in range(DP):
        if zp1[j] <= 0:
            gzp1[j] = 0

    gh = _bwd_input(buf, L["W_P1"], gzp1, DP, LAT)
    _sgd_layer(buf, L["W_P1"], L["B_P1"], gzp1, h, DP, LAT, lr)

    gmean = _trunc_div(sum(gh), LAT)
    ghraw = [gh[i] - gmean for i in range(LAT)]

    gzr1 = _bwd_input(buf, L["W_R2"], ghraw, LAT, DR)
    _sgd_layer(buf, L["W_R2"], L["B_R2"], ghraw, zr1, LAT, DR, lr)
    for m in range(DR):
        if zr1[m] <= 0:
            gzr1[m] = 0

    _sgd_layer(buf, L["W_R1"], L["B_R1"], gzr1, obs, DR, OBS, lr)


def test_muzero_train_kernel_bit_exact_vs_python_reference():
    if os.environ.get(_ISOLATED_CHILD) != "1":
        _run_isolated("test_muzero_train_kernel_bit_exact_vs_python_reference")
        return
    lib = _build_kernel()
    n = lib.forge_muzero_train_buffer_words()
    L = _layout(lib)
    rng = np.random.default_rng(7)
    lr = 2

    buf = np.zeros(n, dtype=np.int32)
    buf[L["W_R1"]:L["B_R1"]] = rng.integers(-4, 5, L["B_R1"] - L["W_R1"])  # shadow weights+biases
    buf[L["OBS_OFF"]:L["OBS_OFF"] + L["OBS"]] = rng.integers(0, 2, L["OBS"])
    buf[L["PTGT"]:L["PTGT"] + L["A"]] = [40, -24, 16, -8]
    buf[L["VTGT"]] = 32
    ref = buf.tolist()

    for step in range(6):
        _reference_train_step(ref, L, lr)
        rc = lib.forge_train_step_muzero_train(
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 2, lr, 0, 0)
        assert rc == 0
        mism = np.nonzero(buf != np.array(ref, dtype=np.int32))[0]
        assert mism.size == 0, (
            f"step {step}: {mism.size} buffer words differ, first at index {int(mism[0])} "
            f"C={int(buf[mism[0]])} ref={ref[mism[0]]}")


def test_muzero_train_kernel_unknown_mode_returns_error():
    if os.environ.get(_ISOLATED_CHILD) != "1":
        _run_isolated("test_muzero_train_kernel_unknown_mode_returns_error")
        return
    lib = _build_kernel()
    n = lib.forge_muzero_train_buffer_words()
    buf = np.zeros(n, dtype=np.int32)
    # only mode 2 (train step) is defined for this kernel
    rc = lib.forge_train_step_muzero_train(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 0, 1, 0, 0)
    assert rc == -1


if __name__ == "__main__":
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
