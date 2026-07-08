"""Bit-exact host harness for the dyadic MuZero HLS forward kernel
(tikitorch/hardware/hls/forge_kernels/forge_train_step_muzero.cpp).

Compiles the kernel to a shared lib with g++ (the CSIM shim — no Vitis), runs the actual
C kernel via ctypes, and compares it BIT-FOR-BIT to an independent Python reference of the
exact same int16 fixed-point shift+add forward. This pins the C kernel to a precise spec
(catching any indexing/saturation/shift bug) and is the harness the on-fabric weights will
flow through (export a trained dyadic net's shift schedule into this buffer). Skipped if g++
is unavailable.
"""
import ctypes
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

# Builds + ctypes-loads a native HLS kernel in-process: a kernel bug can segfault the
# whole pytest run, and the kernel math is already gated by the dedicated `csim` CI job.
# So this is deselected from the default `pytest -q` (run explicitly with `-m kernel`).
pytestmark = pytest.mark.kernel

HLS = os.path.join(os.path.dirname(__file__), "..", "tikitorch", "hardware", "hls")
KERNELS = os.path.join(HLS, "forge_kernels")


@pytest.fixture(scope="module")
def kernel():
    if shutil.which("g++") is None:
        pytest.skip("g++ not available for CSIM build")
    so = os.path.join(tempfile.mkdtemp(), "libmuzero.so")
    src = os.path.join(KERNELS, "forge_train_step_muzero.cpp")
    r = subprocess.run(["g++", "-O2", "-std=c++17", "-DCSIM_ONLY", "-shared", "-fPIC",
                        f"-I{KERNELS}", f"-I{HLS}", src, "-o", so],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"kernel CSIM build failed:\n{r.stderr}")
    lib = ctypes.CDLL(so)
    lib.forge_train_step_muzero.argtypes = [ctypes.POINTER(ctypes.c_int32)] + [ctypes.c_int] * 4
    lib.forge_muzero_buffer_words.restype = ctypes.c_int
    lib.forge_muzero_layout.argtypes = [ctypes.POINTER(ctypes.c_int)]
    return lib


def _layout(lib):
    arr = (ctypes.c_int * 28)()
    lib.forge_muzero_layout(arr)
    keys = ["OBS", "L", "A", "DR", "DP", "OBS_OFF", "H", "TMP_L", "POLICY", "VALUE",
            "R1E", "R1S", "R2E", "R2S", "P1E", "P1S", "P2E", "P2S", "PHE", "PHS", "VHE", "VHS",
            "BR1", "BR2", "BP1", "BP2", "BPH", "BVH"]
    return dict(zip(keys, list(arr)))


def _sat16(a):
    return max(-32768, min(32767, int(a)))


def _trunc_div(s, n):                       # C integer division truncates toward zero
    q = abs(int(s)) // n
    return -q if s < 0 else q


def _dyadic_linear(buf, off_e, off_s, off_b, x, OUT, IN):
    y = []
    for o in range(OUT):
        acc = int(buf[off_b + o])
        for i in range(IN):
            idx = o * IN + i
            e, sgn = int(buf[off_e + idx]), int(buf[off_s + idx])
            v = (int(x[i]) << e) if e >= 0 else (int(x[i]) >> (-e))   # arithmetic shift (both Py & g++)
            acc += v if sgn >= 0 else -v
        y.append(_sat16(acc))
    return y


def _reference_forward(buf, L):
    """Exact Python mirror of do_fwd in forge_train_step_muzero.cpp."""
    obs = [int(buf[L["OBS_OFF"] + i]) for i in range(L["OBS"])]
    tdr = _dyadic_linear(buf, L["R1E"], L["R1S"], L["BR1"], obs, L["DR"], L["OBS"])
    tdr = [max(0, v) for v in tdr]                                   # relu
    h = _dyadic_linear(buf, L["R2E"], L["R2S"], L["BR2"], tdr, L["L"], L["DR"])
    s = sum(h); m = _trunc_div(s, L["L"])                            # submean (pow2 /L = shift)
    h = [_sat16(v - m) for v in h]
    tdp = _dyadic_linear(buf, L["P1E"], L["P1S"], L["BP1"], h, L["DP"], L["L"])
    tdp = [max(0, v) for v in tdp]
    z = _dyadic_linear(buf, L["P2E"], L["P2S"], L["BP2"], tdp, L["L"], L["DP"])
    pol = _dyadic_linear(buf, L["PHE"], L["PHS"], L["BPH"], z, L["A"], L["L"])
    val = _dyadic_linear(buf, L["VHE"], L["VHS"], L["BVH"], z, 1, L["L"])
    return pol, val[0]


def test_muzero_kernel_bit_exact_vs_python_reference(kernel):
    lib = kernel
    n = lib.forge_muzero_buffer_words()
    L = _layout(lib)
    rng = np.random.default_rng(0)
    for trial in range(5):
        buf = np.zeros(n, dtype=np.int32)
        # small dyadic weights (exp in [-3,1]) + signs + small biases + binary obs -> exercises
        # the real shift/add/relu/submean path without everything saturating
        for (oe, os_, ob, OUT, IN) in [(L["R1E"], L["R1S"], L["BR1"], L["DR"], L["OBS"]),
                                       (L["R2E"], L["R2S"], L["BR2"], L["L"], L["DR"]),
                                       (L["P1E"], L["P1S"], L["BP1"], L["DP"], L["L"]),
                                       (L["P2E"], L["P2S"], L["BP2"], L["L"], L["DP"]),
                                       (L["PHE"], L["PHS"], L["BPH"], L["A"], L["L"]),
                                       (L["VHE"], L["VHS"], L["BVH"], 1, L["L"])]:
            buf[oe:oe + OUT * IN] = rng.integers(-3, 2, OUT * IN)
            buf[os_:os_ + OUT * IN] = rng.integers(0, 2, OUT * IN) * 2 - 1
            buf[ob:ob + OUT] = rng.integers(-2, 3, OUT)
        buf[L["OBS_OFF"]:L["OBS_OFF"] + L["OBS"]] = rng.integers(0, 2, L["OBS"])

        ref_pol, ref_val = _reference_forward(buf, L)
        rc = lib.forge_train_step_muzero(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 0, 0, 0, 0)
        assert rc == 0
        got_pol = [int(buf[L["POLICY"] + i]) for i in range(L["A"])]
        got_val = int(buf[L["VALUE"]])
        assert got_pol == ref_pol, f"trial {trial}: policy C={got_pol} ref={ref_pol}"
        assert got_val == ref_val, f"trial {trial}: value C={got_val} ref={ref_val}"
