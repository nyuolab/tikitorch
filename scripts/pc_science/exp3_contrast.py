"""exp3_contrast.py -- Experiment 2 (CORRECTED): the deep-PC-traction CONTRAST figure.

THE CLAIM (the manuscript's central figure): the PC-to-backprop accuracy gap stays
FLAT with depth for our TROPICAL/cascade PC, while it GROWS with depth for the SMOOTH
finite-inference PC (the known deep-PC degradation).

This is a 2x2 CONTRAST -- {tropical net, smooth net} x {backprop ceiling, PC method} --
NOT a cascade-vs-exact-BP comparison on one net (that ~=0 by construction = Experiment 1).
For a given (L, seed) it trains FOUR arms on byte-level enwik8 to a comparable point,
logs the bpb trajectory (confirm plateau), reports final bpb per arm, and emits the two
gap curves.

  TROPICAL net (MULTFREE=1, our dyadic transformer):
    (1) BP_trop     = global backprop with STE THROUGH THE DYAD. mode="backprop" already
                      does this: the weight grad of y = x @ dyad(W).T is gy.T @ x, i.e.
                      the dyad-quant step is treated identity/STE (d dyad(W)/dW ~= I), so
                      gradients FLOW (verified nonzero, ~= cascade grad norm). This is the
                      achievable ceiling for the dyadic net.  Optimizer: dyadic-Adam (the
                      silicon recipe) -- shared with PC_trop so the contrast isolates the
                      GRADIENT METHOD (global BP vs cascade), not the optimizer.
    (2) PC_trop     = cascade  (mode="cascade", our method / silicon recipe).
      -> claim: bpb(PC_trop) ~= bpb(BP_trop) at ALL depths (gap flat ~0), because on the
         (piecewise-linear) dyadic net cascade == exact relaxation == backprop (Exp 1).

  SMOOTH net (MULTFREE=0, float transformer, softmax + real LN -- where deep PC degrades):
    (3) BP_smooth   = global backprop (float, mode="backprop", float-Adam). Ceiling.
    (4) PC_smooth_T = iterative PC inference with a FIXED FINITE budget T, TRAINED with
                      those finite-T (unconverged) gradients each step. This is the
                      literature's deep-PC.
      -> claim: bpb(PC_smooth_T) - bpb(BP_smooth) GROWS with depth: a fixed T can only
         propagate the top loss error T layers down, so deep layers receive a degraded
         gradient -> worse accuracy the deeper the net.

FINITE-T PC INFERENCE (the load-bearing correctness piece).
The residual-stream PC energy F = 1/2 sum_l ||eps_l||^2 + CE(top) has its EQUILIBRIUM
(dF/dv = 0) at exactly the backprop gradient (Whittington & Bogacz 2017). With the PWL
selections held at the forward pass (the exact local Jacobian at the forward activations),
the equilibrium condition is the linear back-substitution
        eps_L := g_top   (loss force at the top node),
        eps_l  = J_{l+1}^T eps_{l+1}   for l = L-1 .. 1.
Running this recursion is EXACTLY the cascade / backprop reverse sweep -> at convergence
grad == backprop (verified grad_cos = 1.000000, relerr = 0 on the smooth float net).

A single Jacobi sweep of this system propagates the error exactly ONE layer deeper. So
"T inference steps" = T Jacobi sweeps from eps=0 => after T sweeps the loss error has
reached min(T, depth) layers. This is the literal finite-inference PC:
        eps^(0) = 0
        eps^(t)_L = g_top,   eps^(t)_l = J_{l+1}^T eps^(t-1)_{l+1}   (l < L)
    * T >= depth  -> eps == equilibrium == backprop  (grad_cos -> 1.0)   [validates it]
    * T <  depth  -> deep layers still 0 / partially-filled -> degraded gradient, and the
                     degradation is WORSE the deeper the net (fixed T reaches a smaller
                     FRACTION of the layers)                              [the degradation]
Plain fixed-step value-node GD is unusable here: J ~ I + delta is stiff (||J|| > 1) so GD
diverges (verified: grad_cos gets WORSE with more steps). The truncated back-substitution
is the numerically-exact, monotone realisation of the same finite-T inference.

Env: L=8 SEED=0
     D=192 H=4 DH=48 FFN=768 SEQ=128 VOCAB=256   (smaller than exp1's D=256 so the
        4-arm x depth x seed sweep + the T-sweep cost of PC_smooth is affordable)
     STEPS=4000 ACCUM=8 XF_LR=3e-3 WARMUP=100 LR_DECAY=1 LR_MINFRAC=0.1
     RELAX_T=5,20 (comma list; PC_smooth_T runs one arm per value)
     EVAL_EVERY=250 EVAL_BATCHES=16  TOKBIN=/tmp/enwik8.u16  OUT=exp3_L{L}_s{seed}.jsonl
One (L, seed) per process; the EC2 runner fans (L x seed) across cores.
"""
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_transformer import Transformer, make_config, load_tokens, sample_batch


# ------------------------------------------------------------------ finite-T PC --
def relax_finiteT_grads(model, cache, tgt, T):
    """Finite-T iterative PC inference gradient (truncated back-substitution).

    T Jacobi sweeps of the equilibrium recursion from eps=0, selections frozen at the
    forward pass (exact local Jacobian at the forward activations):
        eps_L := g_top ;  eps_l = J_{l+1}^T eps_{l+1}  (l < L) ;  1 sweep = 1 layer deeper.
    T -> infinity reproduces backprop exactly (verified); finite T truncates the depth the
    loss error can reach -> the deep-PC degradation. Weight grads are then read out by the
    exact per-block reverse sweep at the settled {eps_l}.
    """
    cf = model.cf; NL = cf["NL"]; tokens = cache["tokens"]; caches0 = cache["caches"]
    lc = [None] + [caches0[l] for l in range(NL)]           # frozen forward caches
    gLG = model.logit_grad_exact(lc[NL]["LG"], tgt)
    g_top = model._head_bwd_exact(model.blocks[NL - 1], lc[NL], gLG)   # loss force at top
    eps = [None] + [np.zeros_like(caches0[l]["RP"]) for l in range(NL)]  # eps[1..NL] = 0
    for _ in range(max(1, T)):
        new = [None] * (NL + 1)
        new[NL] = g_top                                     # top node clamped by the loss
        for l in range(NL - 1, 0, -1):
            new[l] = model._Jt_eps(model.blocks[l], lc[l + 1], eps[l + 1], tokens, tgt)
        eps = new
    grads = [None] * NL
    for l in range(NL):
        if model.blocks[l].role in ("out", "both"):
            _, gd = model.bwd_block_exact(model.blocks[l], lc[l + 1], eps[l + 1],
                                          tokens, tgt, gLG=gLG)
        else:
            _, gd = model.bwd_block_exact(model.blocks[l], lc[l + 1], eps[l + 1],
                                          tokens, tgt)
        grads[l] = gd
    return grads


# --------------------------------------------------------------- training helpers --
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


ARM_SPEC = {
    #  arm         : (MULTFREE, how-to-get-grads)
    "BP_trop":     (1, "backprop"),   # dyadic net, global BP (STE-through-dyad) -- ceiling
    "PC_trop":     (1, "cascade"),    # dyadic net, cascade -- our method
    "BP_smooth":   (0, "backprop"),   # float net, global BP -- ceiling
    "PC_smooth_T": (0, "relaxT"),     # float net, finite-T iterative PC -- deep-PC
}


def arm_grads(model, cache, tgt, how, T):
    if how == "cascade":
        return model.backward(cache, tgt, mode="cascade")["grads"]
    if how == "backprop":
        return model.backward(cache, tgt, mode="backprop")["grads"]
    if how == "relaxT":
        return relax_finiteT_grads(model, cache, tgt, T)
    raise ValueError(how)


def train_one(arm, L, seed, data, base, steps, accum, lr0, warmup, minfrac, decay,
              eval_every, eval_batches, T, fout):
    multfree, how = ARM_SPEC[arm]
    cfg = dict(base); cfg["NL"] = L; cfg["SEED"] = seed; cfg["MULTFREE"] = multfree
    cfg["BF16"] = 1 if multfree else 0
    cfg = make_config(**cfg)
    model = Transformer(cfg)
    rng = np.random.RandomState(1234 + seed)          # SAME data stream across arms
    ev_rng_seed = 9999 + seed
    # dyadic-Adam for the tropical arms (silicon recipe, shared BP_trop<->PC_trop so the
    # contrast is gradient-method only); float-Adam for the smooth arms.
    apply = model.apply_grads if multfree else model.apply_grads_float
    t0 = time.time(); best = 99.0
    label = f"{arm}" + (f"(T={T})" if how == "relaxT" else "")
    for step in range(steps):
        lr = cosine_lr(step, steps, lr0, warmup, minfrac, decay)
        accg = None
        for _ in range(accum):
            tok, tgt = sample_batch(cfg, data, rng)
            _, cache = model.forward(tok)
            gd = arm_grads(model, cache, tgt, how, T)
            if accg is None:
                accg = gd
            else:
                for ba, bn in zip(accg, gd):
                    for kk in bn:
                        ba[kk] = ba[kk] + bn[kk]
        apply(accg, lr)
        if (step % eval_every == 0) or (step == steps - 1):
            bpb = evaluate(model, data, cfg, np.random.RandomState(ev_rng_seed), eval_batches)
            best = min(best, bpb)
            rec = dict(exp="exp3", arm=arm, L=L, seed=seed, step=step, T=(T if how == "relaxT" else None),
                       bpb=round(bpb, 5), best=round(best, 5), lr=round(lr, 6),
                       wall=round(time.time() - t0, 1))
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"[exp3 {label} L={L} s{seed}] step {step:5d} bpb={bpb:.4f} best={best:.4f}",
                  flush=True)
    return best


def _parse_arms(spec, relaxTs):
    """Expand an ARMS spec into a list of (arm_name, arm_key, T) tuples.
    'all' -> the 4-arm design (5 training runs: +one PC_smooth_T per finite budget).
    A comma list may name any of: BP_trop PC_trop BP_smooth PC_smooth_T (=> one per T)."""
    if spec.strip().lower() in ("all", ""):
        names = ["BP_trop", "PC_trop", "BP_smooth", "PC_smooth_T"]
    else:
        names = [x.strip() for x in spec.split(",") if x.strip()]
    out = []
    for nm in names:
        if nm == "PC_smooth_T":
            for T in relaxTs:
                out.append(("PC_smooth_T", f"PC_smooth_T{T}", T))
        else:
            out.append((nm, nm, None))
    return out


def main():
    L = int(os.environ.get("L", 8)); seed = int(os.environ.get("SEED", 0))
    steps = int(os.environ.get("STEPS", 2500)); accum = int(os.environ.get("ACCUM", 8))
    lr0 = float(os.environ.get("XF_LR", 3e-3)); warmup = int(os.environ.get("WARMUP", 100))
    minfrac = float(os.environ.get("LR_MINFRAC", 0.1)); decay = int(os.environ.get("LR_DECAY", 1))
    eval_every = int(os.environ.get("EVAL_EVERY", 250)); eval_batches = int(os.environ.get("EVAL_BATCHES", 16))
    tokbin = os.environ.get("TOKBIN", "/tmp/enwik8.u16")
    # RELAX_T is a comma list: one PC_smooth_T arm per finite budget (e.g. "5,20").
    relaxTs = [int(x) for x in os.environ.get("RELAX_T", "5,20").split(",") if x.strip()]
    # ARMS selects which arms this process trains ('all' = the full 4-arm design). The EC2
    # runner dispatches ONE arm per job so the 4L x 2seed x 5arm grid fans across cores;
    # a single 'all' process reproduces the whole (L,seed) contrast locally.
    arm_spec = os.environ.get("ARMS", "all")
    arms = _parse_arms(arm_spec, relaxTs)
    out = os.environ.get("OUT", f"exp3_L{L}_s{seed}.jsonl")
    base = dict(D=int(os.environ.get("D", 192)), H=int(os.environ.get("H", 4)),
                DH=int(os.environ.get("DH", 48)), FFN=int(os.environ.get("FFN", 768)),
                SEQ=int(os.environ.get("SEQ", 128)), VOCAB=int(os.environ.get("VOCAB", 256)))
    data = load_tokens(base, tokbin)
    fout = open(out, "w"); finals = {}

    for arm_name, arm_key, T in arms:
        finals[arm_key] = train_one(arm_name, L, seed, data, base, steps, accum, lr0,
                                    warmup, minfrac, decay, eval_every, eval_batches, T, fout)

    # per-arm summary (always emitted). The two gap curves are computed here when the
    # needed ceilings + PC arms were both trained IN THIS PROCESS; otherwise the EC2
    # SUMMARY step recombines per-arm jsonls into the gap-vs-depth curves.
    summary = dict(exp="exp3_summary", L=L, seed=seed, D=base["D"], finals=finals,
                   relaxTs=relaxTs, arms=[a[1] for a in arms],
                   note="gap_trop=PC_trop-BP_trop (claim: ~0 flat vs depth); "
                        "gap_smooth_T*=PC_smooth_T-BP_smooth (claim: grows with depth)")
    if "PC_trop" in finals and "BP_trop" in finals:
        summary["gap_trop_PC_BP"] = round(finals["PC_trop"] - finals["BP_trop"], 5)
    if "BP_smooth" in finals:
        for T in relaxTs:
            k = f"PC_smooth_T{T}"
            if k in finals:
                summary[f"gap_smooth_T{T}"] = round(finals[k] - finals["BP_smooth"], 5)
    fout.write(json.dumps(summary) + "\n"); fout.close()
    print(f"[exp3 L={L} s{seed}] SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
