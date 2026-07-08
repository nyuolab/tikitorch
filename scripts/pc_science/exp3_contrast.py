"""exp3_contrast.py -- Experiment 2 (CORRECTED): the deep-PC-traction CONTRAST figure.

THE CLAIM (the manuscript's central figure): the PC-to-backprop accuracy gap stays
FLAT with depth for our TROPICAL/cascade PC, while it GROWS with depth for the SMOOTH
finite-inference PC (the known deep-PC degradation).

This is a CONTRAST across THREE net regimes x {backprop ceiling, PC method} -- NOT a
cascade-vs-exact-BP comparison on one net (that ~=0 by construction = Experiment 1).
For a given (L, seed) it trains the arms on byte-level enwik8 to a comparable point,
logs the bpb trajectory (train + held-out val; confirm plateau), reports final bpb per
arm, runs the built-in correctness checks, and emits the gap-vs-depth curves.

THE THREE REGIMES (the manuscript spectrum):
  (A) PURE-PWL tropical  (SMOOTH_OFF=1, MULTFREE=1): argmax-gather attn + identity-J LN
      + dyadic-linear + ReLU only. cascade == relaxation == BP EXACTLY (Thm 1, Exp1).
      Arms BP_pwl (STE-through-dyad backprop ceiling) + PC_pwl (cascade). -> gap ~0, FLAT.
  (B) TROPICAL-DOMINATED  (SMOOTH_OFF=0, MULTFREE=1): dyadic linear + softmax attn +
      dyadic LN -- the two smooth ops cost a BOUNDED linearisation error. Arms BP_trop +
      PC_trop (cascade). -> gap bounded (~0.5 nat), grows slowly.
  (C) SMOOTH well-conditioned  (MULTFREE=0, RESID_ALPHA<1): float softmax + real LN, with
      per-block residual SCALING alpha on the sublayer contributions (M=R+alpha*AO,
      RP=M+alpha*FO) so the residual-stream block Jacobian J=I+alpha*delta is well-
      conditioned and the finite-T PC inference converges to BP even deep. Arms
      BP_smooth (float BP ceiling) + PC_smooth_T{5,20} (finite-T deep-PC) +
      PC_smooth_Tconv (T=2*depth, the CONVERGENCE-CHECK arm). -> finite-T grows with
      depth (truncation); Tconv ~= BP (the built-in correctness check).

  NOTE (adversarial honesty): the finite-T PC used here (relax_finiteT_grads) is the
  numerically-exact TRUNCATED BACK-SUBSTITUTION (eps init 0, top clamped by the loss,
  one Jacobi sweep = one layer deeper). It is structurally triangular and top-clamped, so
  it is UNCONDITIONALLY STABLE and reproduces BP bit-exactly once T>=depth -- there is NO
  gradient-level divergence at large T (the historical bpb->8 divergence was the OLD
  fixed-step value-node GD in backward_relaxation, which this path replaces). RESID_ALPHA
  is therefore NOT required to make the convergence check pass; its measured effect is to
  make the SMALL-T (truncated) arm track BP better at a given T (fewer sweeps needed on a
  better-conditioned stream), sharpening the finite-T-vs-Tconv contrast. Applied ONLY to
  the smooth net; the tropical nets always run alpha=1 (realistic).

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
        multi-arm x depth x seed sweep + the T-sweep cost of PC_smooth is affordable)
     STEPS=5000 ACCUM=8 XF_LR=3e-3 WARMUP=100 LR_DECAY=1 LR_MINFRAC=0.1
     RELAX_T=5,20 (comma list; PC_smooth_T runs one arm per finite budget; PC_smooth_Tconv
        additionally runs at T=2*depth as the convergence check)
     RESID_ALPHA=0.25 (smooth-net conditioner; tropical nets ignore it / run alpha=1)
     EVAL_EVERY=250 EVAL_BATCHES=16  TOKBIN=/tmp/enwik8.u16  OUT=exp3_L{L}_s{seed}.jsonl
     Each per-eval jsonl record logs BOTH train_bpb (current batch) and bpb (held-out val).
One (L, seed) per process; the EC2 runner fans (L x seed x arm) across cores.
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
    #  arm            : (MULTFREE, SMOOTH_OFF, how-to-get-grads)
    # -- regime A: PURE-PWL tropical (argmax-gather attn + identity-J LN): gap ~0 flat --
    "BP_pwl":         (1, 1, "backprop"),   # PWL dyadic net, global BP (STE-thru-dyad) ceiling
    "PC_pwl":         (1, 1, "cascade"),    # PWL dyadic net, cascade -- ours (== BP exactly)
    # -- regime B: TROPICAL-DOMINATED (softmax attn + dyadic LN): gap bounded ~0.5 --
    "BP_trop":        (1, 0, "backprop"),   # dyadic net, global BP (STE-thru-dyad) -- ceiling
    "PC_trop":        (1, 0, "cascade"),    # dyadic net, cascade -- our method
    # -- regime C: SMOOTH well-conditioned (float; RESID_ALPHA<1): finite-T grows w/ depth --
    "BP_smooth":      (0, 0, "backprop"),   # float net, global BP -- ceiling
    "PC_smooth_T":    (0, 0, "relaxT"),     # float net, finite-T deep-PC (one arm per budget)
    "PC_smooth_Tconv":(0, 0, "relaxT"),     # float net, T=2*depth -- the CONVERGENCE CHECK
}


def arm_grads(model, cache, tgt, how, T):
    if how == "cascade":
        return model.backward(cache, tgt, mode="cascade")["grads"]
    if how == "backprop":
        return model.backward(cache, tgt, mode="backprop")["grads"]
    if how == "relaxT":
        return relax_finiteT_grads(model, cache, tgt, T)
    raise ValueError(how)


def train_one(arm, arm_key, L, seed, data, base, steps, accum, lr0, warmup, minfrac, decay,
              eval_every, eval_batches, T, resid_alpha, fout):
    multfree, smooth_off, how = ARM_SPEC[arm]
    cfg = dict(base); cfg["NL"] = L; cfg["SEED"] = seed; cfg["MULTFREE"] = multfree
    cfg["BF16"] = 1 if multfree else 0
    cfg["SMOOTH_OFF"] = smooth_off
    # RESID_ALPHA conditions ONLY the smooth net (MULTFREE=0); tropical nets stay realistic.
    cfg["RESID_ALPHA"] = resid_alpha if (multfree == 0) else 1.0
    cfg = make_config(**cfg)
    model = Transformer(cfg)
    rng = np.random.RandomState(1234 + seed)          # SAME data stream across arms
    ev_rng_seed = 9999 + seed
    # dyadic-Adam for the tropical arms (silicon recipe, shared BP<->PC so the contrast is
    # gradient-method only); float-Adam for the smooth arms.
    apply = model.apply_grads if multfree else model.apply_grads_float
    t0 = time.time(); best = 99.0; init_val = None; final_val = None
    label = f"{arm_key}"
    for step in range(steps):
        lr = cosine_lr(step, steps, lr0, warmup, minfrac, decay)
        accg = None; train_bpb_acc = 0.0
        for _ in range(accum):
            tok, tgt = sample_batch(cfg, data, rng)
            logits, cache = model.forward(tok)
            train_bpb_acc += float(model.loss_bpb(logits, tgt))
            gd = arm_grads(model, cache, tgt, how, T)
            if accg is None:
                accg = gd
            else:
                for ba, bn in zip(accg, gd):
                    for kk in bn:
                        ba[kk] = ba[kk] + bn[kk]
        apply(accg, lr)
        if (step % eval_every == 0) or (step == steps - 1):
            train_bpb = train_bpb_acc / accum      # mean train-batch bpb over this step
            bpb = evaluate(model, data, cfg, np.random.RandomState(ev_rng_seed), eval_batches)
            best = min(best, bpb)
            if init_val is None:
                init_val = bpb
            final_val = bpb
            rec = dict(exp="exp3", arm=arm, arm_key=arm_key, L=L, seed=seed, step=step,
                       T=(T if how == "relaxT" else None), alpha=cfg["RESID_ALPHA"],
                       train_bpb=round(train_bpb, 5), bpb=round(bpb, 5),
                       best=round(best, 5), lr=round(lr, 6), wall=round(time.time() - t0, 1))
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"[exp3 {label} L={L} s{seed}] step {step:5d} train={train_bpb:.4f} "
                  f"val={bpb:.4f} best={best:.4f}", flush=True)
    return dict(best=best, init_val=init_val, final_val=final_val)


def _parse_arms(spec, relaxTs, L):
    """Expand an ARMS spec into a list of (arm_name, arm_key, T) tuples.
    'all' -> the full THREE-regime design:
        PWL:     BP_pwl, PC_pwl
        TROP:    BP_trop, PC_trop
        SMOOTH:  BP_smooth, PC_smooth_T{t} (one per finite budget), PC_smooth_Tconv (T=2L)
    A comma list may name any ARM_SPEC key (PC_smooth_T => one arm per RELAX_T budget;
    PC_smooth_Tconv => T=2*depth, the convergence-check arm)."""
    if spec.strip().lower() in ("all", ""):
        names = ["BP_pwl", "PC_pwl", "BP_trop", "PC_trop",
                 "BP_smooth", "PC_smooth_T", "PC_smooth_Tconv"]
    else:
        names = [x.strip() for x in spec.split(",") if x.strip()]
    out = []
    for nm in names:
        if nm == "PC_smooth_T":
            for T in relaxTs:
                out.append(("PC_smooth_T", f"PC_smooth_T{T}", T))
        elif nm == "PC_smooth_Tconv":
            out.append(("PC_smooth_Tconv", f"PC_smooth_Tconv{2 * L}", 2 * L))
        else:
            out.append((nm, nm, None))
    return out


def main():
    L = int(os.environ.get("L", 8)); seed = int(os.environ.get("SEED", 0))
    steps = int(os.environ.get("STEPS", 5000)); accum = int(os.environ.get("ACCUM", 8))
    lr0 = float(os.environ.get("XF_LR", 3e-3)); warmup = int(os.environ.get("WARMUP", 100))
    minfrac = float(os.environ.get("LR_MINFRAC", 0.1)); decay = int(os.environ.get("LR_DECAY", 1))
    eval_every = int(os.environ.get("EVAL_EVERY", 250)); eval_batches = int(os.environ.get("EVAL_BATCHES", 16))
    tokbin = os.environ.get("TOKBIN", "/tmp/enwik8.u16")
    # RESID_ALPHA: the smooth-net residual conditioner (< 1). Tropical arms ignore it.
    resid_alpha = float(os.environ.get("RESID_ALPHA", 0.25))
    # convergence-check tolerance (nats): |bpb(PC_smooth_Tconv) - bpb(BP_smooth)| must be < TOL.
    conv_tol = float(os.environ.get("CONV_TOL", 0.15))
    # RELAX_T is a comma list: one PC_smooth_T arm per finite budget (e.g. "5,20").
    relaxTs = [int(x) for x in os.environ.get("RELAX_T", "5,20").split(",") if x.strip()]
    # ARMS selects which arms this process trains ('all' = the full three-regime design).
    # The EC2 runner dispatches ONE ARM per job so the L x seed x arm grid fans across cores;
    # a single 'all' process reproduces the whole (L,seed) contrast locally.
    arm_spec = os.environ.get("ARMS", "all")
    arms = _parse_arms(arm_spec, relaxTs, L)
    out = os.environ.get("OUT", f"exp3_L{L}_s{seed}.jsonl")
    base = dict(D=int(os.environ.get("D", 192)), H=int(os.environ.get("H", 4)),
                DH=int(os.environ.get("DH", 48)), FFN=int(os.environ.get("FFN", 768)),
                SEQ=int(os.environ.get("SEQ", 128)), VOCAB=int(os.environ.get("VOCAB", 256)))
    data = load_tokens(base, tokbin)
    fout = open(out, "w")
    finals = {}   # arm_key -> best (kept for back-compat with the S3 SUMMARY recombiner)
    stats = {}    # arm_key -> dict(best, init_val, final_val)

    for arm_name, arm_key, T in arms:
        r = train_one(arm_name, arm_key, L, seed, data, base, steps, accum, lr0,
                       warmup, minfrac, decay, eval_every, eval_batches, T, resid_alpha, fout)
        finals[arm_key] = r["best"]; stats[arm_key] = r

    # -------------------------------------------------------- CORRECTNESS CHECKS --
    # (a) BP-STE arms actually TRAIN: final val bpb < initial val bpb.
    # (b) The smooth relaxation-to-CONVERGENCE (Tconv=2L) ~= BP_smooth within conv_tol.
    # Failing checks are logged + flagged (VISIBLE, not silent).
    checks = []
    for arm_key, r in stats.items():
        if arm_key.startswith("BP_"):
            trains = (r["final_val"] is not None and r["init_val"] is not None
                      and r["final_val"] < r["init_val"] - 1e-6)
            checks.append(dict(check="BP_trains", arm=arm_key,
                               init=round(r["init_val"], 5), final=round(r["final_val"], 5),
                               passed=bool(trains)))
    # convergence check: PC_smooth_Tconv{2L} vs BP_smooth
    tconv_key = f"PC_smooth_Tconv{2 * L}"
    if tconv_key in finals and "BP_smooth" in finals:
        gap = finals[tconv_key] - finals["BP_smooth"]
        checks.append(dict(check="smooth_Tconv_matches_BP", arm=tconv_key,
                           Tconv=2 * L, tol=conv_tol,
                           bpb_PC=round(finals[tconv_key], 5),
                           bpb_BP=round(finals["BP_smooth"], 5),
                           gap=round(gap, 5), passed=bool(abs(gap) < conv_tol)))

    # ------------------------------------------------------------------ SUMMARY ---
    summary = dict(exp="exp3_summary", L=L, seed=seed, D=base["D"],
                   resid_alpha=resid_alpha, finals=finals, relaxTs=relaxTs,
                   arms=[a[1] for a in arms], checks=checks,
                   note="gap_pwl=PC_pwl-BP_pwl (~0 flat); gap_trop=PC_trop-BP_trop (bounded); "
                        "gap_smooth_T*=PC_smooth_T-BP_smooth (grows w/ depth); "
                        "gap_smooth_Tconv=PC_smooth_Tconv-BP_smooth (CHECK: ~0)")
    if "PC_pwl" in finals and "BP_pwl" in finals:
        summary["gap_pwl_PC_BP"] = round(finals["PC_pwl"] - finals["BP_pwl"], 5)
    if "PC_trop" in finals and "BP_trop" in finals:
        summary["gap_trop_PC_BP"] = round(finals["PC_trop"] - finals["BP_trop"], 5)
    if "BP_smooth" in finals:
        for T in relaxTs:
            k = f"PC_smooth_T{T}"
            if k in finals:
                summary[f"gap_smooth_T{T}"] = round(finals[k] - finals["BP_smooth"], 5)
        if tconv_key in finals:
            summary[f"gap_smooth_Tconv{2 * L}"] = round(finals[tconv_key] - finals["BP_smooth"], 5)
    fout.write(json.dumps(summary) + "\n"); fout.close()
    print(f"[exp3 L={L} s{seed}] SUMMARY {json.dumps(summary)}", flush=True)
    # print CHECK PASS/FAIL prominently (a failing check must be visible)
    print(f"[exp3 L={L} s{seed}] CHECKS:", flush=True)
    for c in checks:
        print(f"    {'PASS' if c['passed'] else 'FAIL'}  {c['check']}({c['arm']}): {json.dumps(c)}",
              flush=True)


if __name__ == "__main__":
    main()
