"""exp2_depthsweep.py — Experiment 2: PC-vs-backprop accuracy gap as a function of DEPTH.

For a given (L, seed), trains THREE models on the SAME data/compute budget and logs the bpb
trajectory (so we can confirm the plateau — the user's "train sufficiently, do not cut short"):
  - PC  = cascade  (our method, dyadic forward + single-pass STE/cascade backward = silicon)
  - BP  = exact global backprop on the SAME dyadic forward (isolates PC-vs-BP from dyadic-vs-float)
  - BPf = exact global backprop on the FLOAT forward (MULTFREE=0) = the gold-standard reference

KEY output: the PC-BP bpb gap vs L — does it stay flat, or widen with depth?
(flat where smooth deep PC degrades = the deep-PC-traction claim.)

Env: L=8 SEED=0  D=256 H=4 DH=64 FFN=1024 SEQ=256 VOCAB=256
     STEPS=6000 ACCUM=8 XF_LR=3e-3 WARMUP=100 LR_DECAY=1 LR_MINFRAC=0.1
     EVAL_EVERY=250 EVAL_BATCHES=8  TOKBIN=/tmp/enwik8_20m.u16  OUT=exp2_L{L}_s{seed}.jsonl
One (L,seed) per process; the EC2 runner fans (L x seed) across cores.
"""
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_transformer import Transformer, make_config, load_tokens, sample_batch


def cosine_lr(step, steps, lr0, warmup, minfrac, decay):
    lr = lr0 * min(1.0, (step + 1) / max(1, warmup))
    if decay and step >= warmup:
        t = (step - warmup) / max(1, steps - warmup)
        lr = lr0 * (minfrac + (1 - minfrac) * 0.5 * (1 + math.cos(math.pi * t)))
    return lr


def evaluate(model, data, cfg, rng, nb):
    tot = 0.0
    for _ in range(nb):
        tok, tgt = sample_batch(cfg, data, rng)
        lg, _ = model.forward(tok)
        tot += float(model.loss_bpb(lg, tgt))
    return tot / nb


def train_one(arm, L, seed, data, base, steps, accum, lr0, warmup, minfrac, decay,
              eval_every, eval_batches, tokbin, fout):
    """arm in {'PC','BP','BPf'}."""
    multfree = 0 if arm == "BPf" else 1
    mode = "cascade" if arm == "PC" else "backprop"
    cfg = dict(base); cfg["NL"] = L; cfg["SEED"] = seed; cfg["MULTFREE"] = multfree
    cfg = make_config(**cfg)
    model = Transformer(cfg)
    rng = np.random.RandomState(1234 + seed)       # SAME data stream across arms (matched compute)
    ev_rng_seed = 9999 + seed
    t0 = time.time(); best = 99.0
    # float-precision optimizer for the BP arms (they are the full-precision reference);
    # PC uses the dyadic-Adam apply (silicon recipe).
    # PC and BP share the dyadic-Adam apply so PC-vs-BP isolates the GRADIENT METHOD (cascade vs global
    # backprop), not the optimizer; only BPf (pure-float reference) uses the float optimizer.
    apply = getattr(model, "apply_grads_float", model.apply_grads) if arm == "BPf" else model.apply_grads
    for step in range(steps):
        lr = cosine_lr(step, steps, lr0, warmup, minfrac, decay)
        accg = None
        for k in range(accum):
            tok, tgt = sample_batch(cfg, data, rng)
            lg, cache = model.forward(tok)
            g = model.backward(cache, tgt, mode=mode)
            gd = g["grads"] if isinstance(g, dict) else g
            if accg is None: accg = gd
            else:
                for ba, bn in zip(accg, gd):
                    for kk in bn: ba[kk] = ba[kk] + bn[kk]
        apply(accg, lr)
        if (step % eval_every == 0) or (step == steps - 1):
            bpb = evaluate(model, data, cfg, np.random.RandomState(ev_rng_seed), eval_batches)
            best = min(best, bpb)
            rec = dict(exp="exp2", arm=arm, L=L, seed=seed, step=step,
                       bpb=round(bpb, 5), best=round(best, 5), lr=round(lr, 6),
                       wall=round(time.time() - t0, 1))
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"[exp2 {arm} L={L} s{seed}] step {step:5d} bpb={bpb:.4f} best={best:.4f}", flush=True)
    return best


def main():
    L = int(os.environ.get("L", 8)); seed = int(os.environ.get("SEED", 0))
    steps = int(os.environ.get("STEPS", 6000)); accum = int(os.environ.get("ACCUM", 8))
    lr0 = float(os.environ.get("XF_LR", 3e-3)); warmup = int(os.environ.get("WARMUP", 100))
    minfrac = float(os.environ.get("LR_MINFRAC", 0.1)); decay = int(os.environ.get("LR_DECAY", 1))
    eval_every = int(os.environ.get("EVAL_EVERY", 250)); eval_batches = int(os.environ.get("EVAL_BATCHES", 8))
    tokbin = os.environ.get("TOKBIN", "/tmp/enwik8_20m.u16")
    arms = os.environ.get("ARMS", "PC,BP,BPf").split(",")
    out = os.environ.get("OUT", f"exp2_L{L}_s{seed}.jsonl")
    base = dict(D=int(os.environ.get("D", 256)), H=int(os.environ.get("H", 4)),
                DH=int(os.environ.get("DH", 64)), FFN=int(os.environ.get("FFN", 1024)),
                SEQ=int(os.environ.get("SEQ", 256)), VOCAB=int(os.environ.get("VOCAB", 256)), BF16=1)
    data = load_tokens(base, tokbin)
    fout = open(out, "w"); finals = {}
    for arm in arms:
        finals[arm] = train_one(arm, L, seed, data, base, steps, accum, lr0, warmup, minfrac,
                                 decay, eval_every, eval_batches, tokbin, fout)
    summary = dict(exp="exp2_summary", L=L, seed=seed, finals=finals,
                   gap_PC_BP=round(finals.get("PC", 0) - finals.get("BP", 0), 5),
                   gap_PC_BPf=round(finals.get("PC", 0) - finals.get("BPf", 0), 5), note="PC/BP=dyadicAdam BPf=floatAdam; PC-BP isolates cascade-vs-globalBP")
    fout.write(json.dumps(summary) + "\n"); fout.close()
    print(f"[exp2 L={L} s{seed}] SUMMARY {summary}", flush=True)


if __name__ == "__main__":
    main()
