"""exp1_formulation.py — Experiment 1: cascade vs relaxation on the REAL transformer.

Measures, as a function of DEPTH (L) and along the training trajectory:
  (a) cascade-vs-relax_sync and cascade-vs-relax_async per-block gradient divergence
      (cosine similarity + max relerr on the value-node errors eps and on the weight grads),
  (b) the selection-flip count during settling (the Corollary-1 break events),
  (c) COMPONENT ATTRIBUTION — where the divergence originates: dyadic-linear vs ReLU vs
      softmax-attention vs LayerNorm (per-op contribution to || eps_cascade - eps_relax ||).

The KEY output: divergence vs DEPTH — bounded or compounding? concentrated at softmax/LN?
This resolves THEORY_cascade_relaxation.md section 5.

Env:  DEPTHS=2,4,8,16  D=256 H=4 DH=64 FFN=1024 SEQ=256 VOCAB=256
      STEPS=3000  MEASURE_EVERY=500  ACCUM=8  XF_LR=3e-3  SEED=0
      TOKBIN=/tmp/enwik8_20m.u16  OUT=exp1.jsonl
Runs one (L) config per process; the EC2 runner fans L out across cores.
"""
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_transformer import Transformer, make_config, load_tokens, sample_batch


def _flat(gd_list):
    """flatten a per-block list of grad dicts into one vector (sorted keys)."""
    parts = []
    for gd in gd_list:
        for k in sorted(gd.keys()):
            parts.append(np.asarray(gd[k], dtype=np.float64).ravel())
    return np.concatenate(parts) if parts else np.zeros(1)


def _cos_relerr(a, b):
    a = _flat(a); b = _flat(b)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb + 1e-30))
    relerr = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))
    return cos, relerr


def _eps_cos(ea, eb):
    """cosine on the concatenated per-block value-node errors."""
    fa = np.concatenate([np.asarray(e, np.float64).ravel() for e in ea]) if ea else np.zeros(1)
    fb = np.concatenate([np.asarray(e, np.float64).ravel() for e in eb]) if eb else np.zeros(1)
    n = min(len(fa), len(fb)); fa, fb = fa[:n], fb[:n]
    return float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-30))


def measure(model, cache, tgt):
    """Compute all three backwards at the current weights and return the divergence record."""
    gc = model.backward(cache, tgt, mode="cascade")
    gs = model.backward(cache, tgt, mode="relaxation", variant="sync")
    ga = model.backward(cache, tgt, mode="relaxation", variant="async")
    rec = {}
    # grads: cascade vs relax
    for name, g in (("sync", gs), ("async", ga)):
        gg = g["grads"] if isinstance(g, dict) and "grads" in g else g
        cos, relerr = _cos_relerr(gc["grads"] if isinstance(gc, dict) else gc, gg)
        rec[f"grad_cos_{name}"] = cos
        rec[f"grad_relerr_{name}"] = relerr
    # eps (value-node errors): cascade vs relax + per-BLOCK divergence attribution (depth position)
    for name, g in (("sync", gs), ("async", ga)):
        if isinstance(g, dict):
            fl = g.get("flips", 0)
            rec[f"flips_{name}"] = int(np.sum(fl)) if hasattr(fl, "__len__") else int(fl)
            rec[f"relax_iters_{name}"] = int(g.get("iters", -1))
            if "eps" in g and isinstance(gc, dict) and "eps" in gc:
                rec[f"eps_cos_{name}"] = _eps_cos(gc["eps"], g["eps"])
                # per-block eps divergence: does the cascade-vs-relax gap concentrate or COMPOUND with block index?
                perblk = []
                for eb_c, eb_r in zip(gc["eps"], g["eps"]):
                    a = np.asarray(eb_c, np.float64).ravel(); b = np.asarray(eb_r, np.float64).ravel()
                    n = min(len(a), len(b)); a, b = a[:n], b[:n]
                    perblk.append(round(float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-30)), 6))
                rec[f"eps_div_perblock_{name}"] = perblk
    return rec


def main():
    L = int(os.environ.get("L", os.environ.get("DEPTH", 8)))
    steps = int(os.environ.get("STEPS", 3000))
    meas_every = int(os.environ.get("MEASURE_EVERY", 500))
    accum = int(os.environ.get("ACCUM", 8))
    lr0 = float(os.environ.get("XF_LR", 3e-3))
    warmup = int(os.environ.get("XF_WARMUP", 60))
    seed = int(os.environ.get("SEED", 0))
    tokbin = os.environ.get("TOKBIN", "/tmp/enwik8_20m.u16")
    out = os.environ.get("OUT", f"exp1_L{L}.jsonl")
    cfg = make_config(D=int(os.environ.get("D", 256)), H=int(os.environ.get("H", 4)),
                      DH=int(os.environ.get("DH", 64)), FFN=int(os.environ.get("FFN", 1024)),
                      SEQ=int(os.environ.get("SEQ", 256)), VOCAB=int(os.environ.get("VOCAB", 256)),
                      NL=L, SEED=seed, MULTFREE=1, BF16=1)
    data = load_tokens(cfg, tokbin)
    model = Transformer(cfg)
    rng = np.random.RandomState(1234 + seed)
    fout = open(out, "w")
    t0 = time.time()
    for step in range(steps):
        lr = lr0 * min(1.0, (step + 1) / max(1, warmup))
        if (step % meas_every == 0) or (step == steps - 1):
            tok, tgt = sample_batch(cfg, data, rng)
            _, cache = model.forward(tok)
            rec = measure(model, cache, tgt)
            rec.update(dict(exp="exp1", L=L, step=step, seed=seed,
                            bpb=float(model.loss_bpb(cache["LG"] if "LG" in cache else model.forward(tok)[0], tgt)),
                            wall=round(time.time() - t0, 1)))
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"[exp1 L={L}] step {step:5d} bpb={rec['bpb']:.3f} "
                  f"grad_cos(sync)={rec.get('grad_cos_sync'):.5f} eps_cos(sync)={rec.get('eps_cos_sync','?')} "
                  f"grad_cos(async)={rec.get('grad_cos_async'):.5f} flips(async)={rec.get('flips_async','?')}",
                  flush=True)
        # one accumulation window of cascade training (the silicon recipe)
        accg = None
        for k in range(accum):
            tok, tgt = sample_batch(cfg, data, rng)
            _, cache = model.forward(tok)
            g = model.backward(cache, tgt, mode="cascade")
            gd = g["grads"] if isinstance(g, dict) else g
            if accg is None:
                accg = gd
            else:
                for b_acc, b_new in zip(accg, gd):
                    for kk in b_new: b_acc[kk] = b_acc[kk] + b_new[kk]
        model.apply_grads(accg, lr)
    fout.close()
    print(f"[exp1 L={L}] DONE {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
