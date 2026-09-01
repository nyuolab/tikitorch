"""muzero_fabric_integration — end-to-end bit-exact validation of the MuZero fabric kernel against a
real exported dyadic net (forward AND dynamics).

The C++ CSIM testbench cross-checks the kernel against an independent recompute on RANDOM buffer values.
This goes further: it builds a dyadic ``TropicalMuZeroNet``, runs the real EXPORT -> PACK path
(net weights -> (sign,exp) schedule -> the MZ buffer layout) — the path the MuZero AFI never exercised
(it was power-tested with zero weights) — invokes the compiled kernel via ctypes, and checks the kernel
reproduces an INDEPENDENT NumPy integer reference of the fabric spec, BIT-EXACTLY, for mode 0 (forward)
and mode 1 (dynamics). This validates export + packing offsets + the kernel arithmetic together.
The scope is only these fixed-point forward/dynamics records; it excludes policy softmax, MCTS,
losses, optimizer behavior, and the rest of the MuZero loop.

The "fabric net" is the deployed spec: dyadic-quantized weights, int16 activations, int32 accumulate,
submean normalization (replacing TropicalLayerNorm/_minmax_unit, as the kernel does), gather embedding.

    PYTHONPATH=. python3 scripts/muzero_fabric_integration.py            # gridworld + chess scale
"""
import ctypes
import os
import subprocess
import sys

import numpy as np
import torch

from tikitorch.fpga.dyadic_export import export_layer
from tikitorch.planning.muzero.networks import TropicalMuZeroNet
from tikitorch.nn import DyadicLinear

HLS = os.path.join(os.path.dirname(__file__), "..", "tikitorch", "hardware", "hls", "forge_kernels")

# layout index -> name (must match forge_muzero_layout's v[] order in the kernel)
LAY = ["OBS", "L", "A", "DR", "DP", "OFF_OBS", "OFF_H", "OFF_TMP_L", "OFF_POLICY", "OFF_VALUE",
       "R1_EXP", "R1_SGN", "R2_EXP", "R2_SGN", "P1_EXP", "P1_SGN", "P2_EXP", "P2_SGN",
       "PH_EXP", "PH_SGN", "VH_EXP", "VH_SGN", "B_R1", "B_R2", "B_P1", "B_P2", "B_PH", "B_VH",
       "DD", "OFF_AEMB", "OFF_REWARD", "D0A_EXP", "D0A_SGN", "D0B_EXP", "D0B_SGN",
       "D1A_EXP", "D1A_SGN", "D1B_EXP", "D1B_SGN", "RH_EXP", "RH_SGN",
       "B_D0A", "B_D0B", "B_D1A", "B_D1B", "B_RH"]


def compile_lib(obs, lat, a, tag):
    so = f"/tmp/libmuzero_{tag}.so"
    cmd = ["g++", "-O2", "-std=c++17", "-DCSIM_ONLY", "-shared", "-fPIC",
           f"-DMZ_OBS_OVERRIDE={obs}", f"-DMZ_L_OVERRIDE={lat}", f"-DMZ_A_OVERRIDE={a}",
           "-I..", "-I.", "forge_train_step_muzero.cpp", "-o", so]
    subprocess.run(cmd, cwd=HLS, check=True)
    return so


# ---- independent NumPy integer reference (mirrors the kernel arithmetic exactly) ----
def _sat16(a):
    return np.clip(a, -32768, 32767).astype(np.int64)


def _lin(buf, oe, os_, ob, x, OUT, IN):
    exp = buf[oe:oe + OUT * IN].reshape(OUT, IN).astype(np.int64)
    sgn = buf[os_:os_ + OUT * IN].reshape(OUT, IN).astype(np.int64)
    xx = x.astype(np.int64)[None, :]                              # (1, IN)
    v = np.where(exp >= 0, xx << np.clip(exp, 0, None), xx >> np.clip(-exp, 0, None))  # arithmetic shift
    v = np.where(sgn == 0, 0, np.where(sgn > 0, v, -v))           # sgn=0 -> 0 (zero weight)
    acc = buf[ob:ob + OUT].astype(np.int64) + v.sum(axis=1)
    assert np.all(np.abs(acc) < 2**31), "int32 overflow in reference — kernel uses int32 acc"
    return _sat16(acc)


def _relu(a):
    return np.maximum(a, 0)


def _submean(a, L):
    s = int(a.sum()); m = abs(s) // L; m = m if s >= 0 else -m        # C-style truncate toward zero
    return _sat16(a - m)


def _pred(buf, o, h):
    L, A = o["L"], o["A"]
    tdp = _relu(_lin(buf, o["P1_EXP"], o["P1_SGN"], o["B_P1"], h, o["DP"], L))
    z = _lin(buf, o["P2_EXP"], o["P2_SGN"], o["B_P2"], tdp, L, o["DP"])
    pol = _lin(buf, o["PH_EXP"], o["PH_SGN"], o["B_PH"], z, A, L)
    val = _lin(buf, o["VH_EXP"], o["VH_SGN"], o["B_VH"], z, 1, L)
    return pol, val[0]


def ref_forward(buf, o):
    L = o["L"]
    obs = buf[o["OFF_OBS"]:o["OFF_OBS"] + o["OBS"]].astype(np.int64)
    tdr = _relu(_lin(buf, o["R1_EXP"], o["R1_SGN"], o["B_R1"], obs, o["DR"], o["OBS"]))
    h = _submean(_lin(buf, o["R2_EXP"], o["R2_SGN"], o["B_R2"], tdr, L, o["DR"]), L)
    pol, val = _pred(buf, o, h)
    return h, pol, val


def ref_dyn(buf, o, h_in, action):
    L, DD = o["L"], o["DD"]
    emb = buf[o["OFF_AEMB"] + action * L: o["OFF_AEMB"] + action * L + L].astype(np.int64)
    z = _sat16(h_in.astype(np.int64) + emb)
    for ax, sx, ba, bx, sb in [("D0A_EXP", "D0A_SGN", "B_D0A", "D0B", "B_D0B"),
                               ("D1A_EXP", "D1A_SGN", "B_D1A", "D1B", "B_D1B")]:
        u = _relu(_lin(buf, o[ax], o[sx], o[ba], z, DD, L))
        bxe, bxs = bx + "_EXP", bx + "_SGN"
        c = _lin(buf, o[bxe], o[bxs], o[sb], u, L, DD)
        z = _submean(_sat16(z + c), L)
    rew = _lin(buf, o["RH_EXP"], o["RH_SGN"], o["B_RH"], z, 1, L)
    pol, val = _pred(buf, o, z)
    return z, rew[0], pol, val


def pack(buf, o, net, obs_int, action):
    mods = dict(net.named_modules())
    M = {"repr.enc.lin1": "R1", "repr.enc.lin2": "R2", "pred.mix.lin1": "P1", "pred.mix.lin2": "P2",
         "pred.policy_head": "PH", "pred.value_head.head": "VH",
         "dyn.core.0.lin1": "D0A", "dyn.core.0.lin2": "D0B", "dyn.core.1.lin1": "D1A",
         "dyn.core.1.lin2": "D1B", "dyn.reward_head.head": "RH"}
    for path, tag in M.items():
        lin = mods[path]
        assert isinstance(lin, DyadicLinear), f"{path} is {type(lin).__name__}, expected DyadicLinear"
        s = export_layer(lin)
        OUT, IN = s["shape"]
        buf[o[tag + "_EXP"]:o[tag + "_EXP"] + OUT * IN] = s["exponent"].to(torch.int32).reshape(-1).numpy()
        buf[o[tag + "_SGN"]:o[tag + "_SGN"] + OUT * IN] = s["sign"].to(torch.int32).reshape(-1).numpy()
        if s["bias"] is not None:
            buf[o["B_" + tag]:o["B_" + tag] + OUT] = torch.round(s["bias"]).to(torch.int32).numpy()
    emb = torch.round(net.dyn.action_embed.weight.detach()).to(torch.int32).numpy()   # (A, L) gather table
    buf[o["OFF_AEMB"]:o["OFF_AEMB"] + emb.size] = emb.reshape(-1)
    buf[o["OFF_OBS"]:o["OFF_OBS"] + o["OBS"]] = obs_int


def run_scale(obs_dim, lat, n_act, tag, seed=0):
    print(f"\n=== {tag}: OBS={obs_dim} L={lat} A={n_act} ===")
    so = compile_lib(obs_dim, lat, n_act, tag)
    lib = ctypes.CDLL(so)
    lib.forge_muzero_buffer_words.restype = ctypes.c_int
    lib.forge_train_step_muzero.restype = ctypes.c_int
    lib.forge_train_step_muzero.argtypes = [ctypes.POINTER(ctypes.c_int32)] + [ctypes.c_int] * 4
    lib.forge_muzero_layout.argtypes = [ctypes.POINTER(ctypes.c_int)]
    N = lib.forge_muzero_buffer_words()
    lay = (ctypes.c_int * 64)(); lib.forge_muzero_layout(lay)
    o = {LAY[i]: lay[i] for i in range(len(LAY))}
    assert o["OBS"] == obs_dim and o["L"] == lat and o["A"] == n_act

    torch.manual_seed(seed)
    net = TropicalMuZeroNet(obs_dim, n_act, latent_dim=lat, core_kind="dyadic",
                            dyn_kind="dyadic", dyadic_heads=True)
    net.eval()
    rng = np.random.default_rng(seed)
    # FIXED-POINT activation scaling: naive 0/1 obs collapse to all-zero outputs because int16 dyadic
    # RIGHT-shifts annihilate tiny activations (1>>2 == 0). Deploy needs obs in a fixed-point unit so the
    # shifts retain bits — here a "1" is FXP. (A real deploy tunes FXP per layer to avoid saturation.)
    FXP = 256
    obs_int = ((rng.random(obs_dim) < 0.3).astype(np.int32)) * FXP   # sparse {0, FXP} board-like obs
    action = 1 % n_act

    buf = np.zeros(N, dtype=np.int32)
    pack(buf, o, net, obs_int, action)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # ---- forward (mode 0) ----
    assert lib.forge_train_step_muzero(ptr, 0, 0, 0, 0) == 0
    k_h = buf[o["OFF_H"]:o["OFF_H"] + lat].copy()
    k_pol = buf[o["OFF_POLICY"]:o["OFF_POLICY"] + n_act].copy()
    k_val = int(buf[o["OFF_VALUE"]])
    r_h, r_pol, r_val = ref_forward(buf, o)
    fwd_bad = (int(np.abs(k_h - r_h).max()) + int(np.abs(k_pol - r_pol).max()) + abs(k_val - int(r_val)))

    # ---- dynamics (mode 1) from the forward h ----
    h_in = buf[o["OFF_H"]:o["OFF_H"] + lat].copy()
    assert lib.forge_train_step_muzero(ptr, 1, action, 0, 0) == 0
    k_h2 = buf[o["OFF_H"]:o["OFF_H"] + lat].copy()
    k_rew = int(buf[o["OFF_REWARD"]])
    k_pol2 = buf[o["OFF_POLICY"]:o["OFF_POLICY"] + n_act].copy()
    k_val2 = int(buf[o["OFF_VALUE"]])
    r_h2, r_rew, r_pol2, r_val2 = ref_dyn(buf, o, h_in, action)
    dyn_bad = (int(np.abs(k_h2 - r_h2).max()) + abs(k_rew - int(r_rew))
               + int(np.abs(k_pol2 - r_pol2).max()) + abs(k_val2 - int(r_val2)))

    nz_pol = int(np.count_nonzero(k_pol))
    changed = int(np.abs(k_h2.astype(np.int64) - h_in.astype(np.int64)).sum())
    # bit-exactness is the correctness gate; non-degenerate outputs confirm the check exercised real values
    bitexact = (fwd_bad == 0 and dyn_bad == 0)
    nondegen = (nz_pol > 0 and changed > 0)
    print(f"  forward : max|Δ|={fwd_bad}  (policy nonzero {nz_pol}/{n_act}, value={k_val})")
    print(f"  dynamics: max|Δ|={dyn_bad}  (h' changed={changed > 0}, reward={k_rew}, value={k_val2})")
    print(f"  -> {'PASS' if (bitexact and nondegen) else 'FAIL'}: bit-exact={bitexact} (kernel == exported-net "
          f"integer ref, fwd+dyn), non-degenerate={nondegen}")
    return bitexact and nondegen


if __name__ == "__main__":
    results = []
    results.append(run_scale(48, 32, 4, "gridworld"))
    results.append(run_scale(769, 64, 4096, "chess"))
    print("\n" + ("ALL PASS" if all(results) else "FAILURES PRESENT"))
    sys.exit(0 if all(results) else 1)
