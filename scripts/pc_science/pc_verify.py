"""pc_verify.py -- THE THEOREM-DERIVED CORRECTNESS GATE (mandatory, gates everything).

Two checks:

  (A) THEOREM 1 GATE (the critical one, empirical anchor of the manuscript):
      On a PURELY PIECEWISE-LINEAR net (SMOOTH_OFF=1: hard argmax-gather attention +
      fixed-scale identity-Jacobian LN => dyadic-linear + ReLU + gather only), assert
          relax_sync  ==  cascade   BIT-EXACT.
      If they do not match bit-exactly, the relaxation implementation is WRONG.
      Also print relax_async vs cascade (should differ only by selection-flip events).

  (B) BACKPROP FINITE-DIFFERENCE CHECK:
      On a tiny SMOOTH (softmax/LN ON) FLOAT net (MULTFREE=0), verify mode='backprop'
      matches a central finite-difference gradient of the CE loss to < 1e-5.

Run:  python3 pc_verify.py
Exit code 0 iff BOTH pass. Prints numbers-first.
"""
import sys, math
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pc_transformer import Transformer, make_config


def _grads_equal_bitexact(ga, gb):
    """Compare two per-block grad dict lists for exact byte equality."""
    maxabs = 0.0
    allkeys_ok = True
    for da, db in zip(ga, gb):
        if set(da.keys()) != set(db.keys()):
            allkeys_ok = False
        for k in da:
            a = np.asarray(da[k]); b = np.asarray(db[k])
            if a.shape != b.shape:
                return False, math.inf
            d = np.max(np.abs(a - b)) if a.size else 0.0
            maxabs = max(maxabs, float(d))
            if not np.array_equal(a, b):
                pass  # tracked via maxabs
    exact = allkeys_ok and maxabs == 0.0
    return exact, maxabs


def _eps_equal_bitexact(ea, eb):
    maxabs = 0.0
    for k in ea:
        a = np.asarray(ea[k]); b = np.asarray(eb[k])
        d = np.max(np.abs(a - b)) if a.size else 0.0
        maxabs = max(maxabs, float(d))
    return maxabs == 0.0, maxabs


def theorem_gate(verbose=True):
    """Theorem 1: PWL + sync => relax_sync == cascade BIT-EXACT."""
    cf = make_config(D=32, H=2, DH=16, FFN=64, SEQ=24, VOCAB=16, NL=4,
                     MULTFREE=1, BF16=1, SMOOTH_OFF=1, SEED=3,
                     RELAX_T=60, RELAX_GAMMA=0.4, RELAX_TOL=1e-12)
    m = Transformer(cf)
    rng = np.random.RandomState(0)
    tok = rng.randint(0, cf["VOCAB"], size=cf["SEQ"])
    tgt = rng.randint(0, cf["VOCAB"], size=cf["SEQ"])

    logits, cache = m.forward(tok)
    cas = m.backward(cache, tgt, mode="cascade")
    rsy = m.backward(cache, tgt, mode="relaxation", variant="sync")
    ras = m.backward(cache, tgt, mode="relaxation", variant="async")

    g_ok, g_max = _grads_equal_bitexact(cas["grads"], rsy["grads"])
    e_ok, e_max = _eps_equal_bitexact(cas["eps"], rsy["eps"])

    # async vs cascade (expected to differ only by flip events)
    ga_ok, ga_max = _grads_equal_bitexact(cas["grads"], ras["grads"])
    ea_ok, ea_max = _eps_equal_bitexact(cas["eps"], ras["eps"])
    async_flips = int(np.sum(ras["flips"]))

    if verbose:
        print("=" * 74)
        print("THEOREM 1 GATE  (PWL net: SMOOTH_OFF=1, hard-gather attn + identity-J LN)")
        print(f"  config: D={cf['D']} H={cf['H']} FFN={cf['FFN']} SEQ={cf['SEQ']} "
              f"VOCAB={cf['VOCAB']} NL={cf['NL']} MULTFREE={cf['MULTFREE']}")
        print(f"  relax_sync vs cascade : grads bit-exact={g_ok} (max|d|={g_max:.3e})  "
              f"eps bit-exact={e_ok} (max|d|={e_max:.3e})")
        print(f"  relax_sync iters={rsy['iters']}  final ||dF/dv||={rsy['trace'][-1]:.3e}  "
              f"sync flips(total)={int(np.sum(rsy['flips']))}")
        print(f"  relax_async vs cascade: grads max|d|={ga_max:.3e} eps max|d|={ea_max:.3e}  "
              f"async selection-flips(total)={async_flips}")
        verdict = "PASS (bit-exact)" if (g_ok and e_ok) else "FAIL"
        print(f"  >>> THEOREM GATE: {verdict}")
        print("=" * 74)
    return (g_ok and e_ok), dict(g_max=g_max, e_max=e_max, ga_max=ga_max,
                                 ea_max=ea_max, async_flips=async_flips,
                                 sync_iters=rsy["iters"])


def _flat_params(m):
    """Flatten all trainable params (block-ordered) into a single vector + a setter."""
    slices = []
    vals = []
    for l, blk in enumerate(m.blocks):
        names = ["WQ", "WK", "WV", "WO", "WUP", "WDN"]
        if blk.role in ("out", "both"): names.append("WHEAD")
        if blk.role in ("in", "both"): names += ["EMB", "POS"]
        for nm in names:
            arr = getattr(blk, nm)
            slices.append((l, nm, arr.shape, arr.size))
            vals.append(arr.ravel().copy())
    theta = np.concatenate(vals)
    return theta, slices


def _set_params(m, theta, slices):
    o = 0
    for (l, nm, shape, size) in slices:
        setattr(m.blocks[l], nm, theta[o:o + size].reshape(shape).copy())
        o += size


def finite_diff_check(verbose=True, n_probe=40, eps=1e-6):
    """Central finite-difference check of mode='backprop' on a tiny FLOAT smooth net."""
    cf = make_config(D=16, H=2, DH=8, FFN=32, SEQ=8, VOCAB=10, NL=2,
                     MULTFREE=0, BF16=0, F32=0, SMOOTH_OFF=0, SEED=5)
    m = Transformer(cf)
    rng = np.random.RandomState(1)
    tok = rng.randint(0, cf["VOCAB"], size=cf["SEQ"])
    tgt = rng.randint(0, cf["VOCAB"], size=cf["SEQ"])

    def loss_of(theta, slices):
        _set_params(m, theta, slices)
        m.clear_dyad_cache()
        lg, _ = m.forward(tok)
        return m.loss_bpb(lg, tgt) * math.log(2.0)   # nats CE (matches grad scale 1/SEQ)

    theta0, slices = _flat_params(m)
    # analytic grad from backprop mode
    _set_params(m, theta0, slices)
    lg, cache = m.forward(tok)
    out = m.backward(cache, tgt, mode="backprop")
    ganalytic, _ = _flat_params_grads(m, out["grads"])

    # probe a random subset of coordinates
    idx = rng.choice(theta0.size, size=min(n_probe, theta0.size), replace=False)
    max_rel = 0.0; worst = None
    for i in idx:
        tp = theta0.copy(); tp[i] += eps
        tm = theta0.copy(); tm[i] -= eps
        gfd = (loss_of(tp, slices) - loss_of(tm, slices)) / (2 * eps)
        ga = ganalytic[i]
        denom = max(1.0, abs(ga), abs(gfd))
        rel = abs(gfd - ga) / denom
        if rel > max_rel:
            max_rel = rel; worst = (int(i), float(ga), float(gfd))
    _set_params(m, theta0, slices)   # restore
    ok = max_rel < 1e-5
    if verbose:
        print("BACKPROP FINITE-DIFF CHECK (tiny float smooth net, central diff)")
        print(f"  config: D={cf['D']} NL={cf['NL']} SEQ={cf['SEQ']} probes={len(idx)} eps={eps}")
        print(f"  max rel err = {max_rel:.3e}  (worst i={worst[0]} analytic={worst[1]:.4e} fd={worst[2]:.4e})")
        print(f"  >>> FINITE-DIFF: {'PASS' if ok else 'FAIL'} (< 1e-5)")
    return ok, max_rel


def _flat_params_grads(m, grads):
    """Flatten grads in the SAME order as _flat_params."""
    vals = []; slices = []
    for l, blk in enumerate(m.blocks):
        names = ["WQ", "WK", "WV", "WO", "WUP", "WDN"]
        if blk.role in ("out", "both"): names.append("WHEAD")
        if blk.role in ("in", "both"): names += ["EMB", "POS"]
        for nm in names:
            g = grads[l][nm]
            slices.append((l, nm, g.shape, g.size))
            vals.append(g.ravel().copy())
    return np.concatenate(vals), slices


def main():
    print("#" * 74)
    print("# pc_verify.py -- PC theorem gate + backprop finite-diff check")
    print("#" * 74)
    gate_ok, gate_stats = theorem_gate()
    print()
    fd_ok, fd_rel = finite_diff_check()
    print()
    print("SUMMARY:")
    print(f"  THEOREM GATE (relax_sync==cascade bit-exact on PWL): {'PASS' if gate_ok else 'FAIL'}")
    print(f"  BACKPROP FINITE-DIFF (<1e-5): {'PASS' if fd_ok else 'FAIL'}  (max rel err {fd_rel:.3e})")
    ok = gate_ok and fd_ok
    print(f"  OVERALL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
