"""pc_transformer.py -- importable module refactor of the distributed dyadic PC
transformer CPU twin (/tmp/mp_chain_30m.py), with THREE backward modes:

  mode="cascade"     : the single reverse-sweep STE/cascade backward that SILICON runs
                       (dyad(W)^T messages, per-block value-node errors {eps_l}).
  mode="backprop"    : TRUE end-to-end backprop -- exact chain rule through the WHOLE
                       net as one graph in float64 (NO dyadic quant on backward, NO
                       message quant, NO STE). Gold-standard for Experiment 2.
  mode="relaxation"  : iterative PC inference. Value nodes x_l initialised to forward
                       activations mu_l, clamp x_0=input, x_L=target; iterate
                       x_l <- x_l - gamma * dF/dx_l for T steps, dF/dx_l = eps_l -
                       J_{l+1}^T eps_{l+1}. Sub-variants relax_sync / relax_async.

FORWARD math is BYTE-IDENTICAL across modes (same dyad / qact / softmax / LN). Only
the backward differs.

------------------------------------------------------------------------------------
VALUE-NODE TOPOLOGY (the PC graph)
------------------------------------------------------------------------------------
The value nodes are the RESIDUAL-STREAM states, one per transformer block, plus an
input node and an output (logit) node:

    x_0  = embed(tokens)  + pos            (clamped to the input)
    x_l  = mu_l = block_l(x_{l-1})         for l = 1..L   (RP output of block l)
    x_out = head(final_LN(x_L))            (the logits value node)

Energy  F = sum_{l=1..L} 1/2 ||eps_l||^2  +  F_out ,
  eps_l = x_l - mu_l(x_{l-1}) ,   mu_l = f_l(x_{l-1})  (the full block map).

f_l is the ENTIRE block-l computation (LN1, attn, +res, LN2, FFN, +res -> RP).
This is exactly the twin's per-block forward. The per-block cascade backward is the
single-pass STE reverse sweep gRP -> gR through f_l with dyad(W)^T weights; its
value-node error is eps_l = the gradient arriving at the residual input of block l.

------------------------------------------------------------------------------------
Everything is env-free by default; dims/flags come from a config dict.
"""
import math
import numpy as np

# ------------------------------------------------------------------ config -------
_DEFAULT = dict(
    D=64, H=2, DH=32, FFN=256, SEQ=64, VOCAB=64, NL=3,
    ABITS=9, MBITS=2,
    MULTFREE=1,     # 1 = mult-free PC recipe (dyad W, qact, dyadic LN mantissa, bf16 STE)
    BF16=1,         # bf16 STE rounding of backward messages (only meaningful for cascade)
    F32=0,          # fp32 forward matmul emulation
    SMOOTH_OFF=0,   # 1 = fully-PWL net: hard argmax attention + identity-Jacobian LN
    SEED=7,
    # --- smooth-net WELL-CONDITIONING (Exp2 gate): scale the two sublayer contributions
    # by RESID_ALPHA<1 so the residual-stream block Jacobian J = I + alpha*delta has
    # ||J|| controlled -> the finite-T PC relaxation CONVERGES to BP even at large depth.
    # Applied CONSISTENTLY in forward, exact BP, cascade, relaxation, and the frozen
    # sync refwd so ALL backward methods see the identical net. Default 1.0 = unchanged
    # (the realistic tropical nets always run alpha=1). NOTE: the finite-T PC here is the
    # numerically-exact truncated back-substitution (unconditionally stable, == BP at
    # T>=depth), so the convergence gate PASSES at any alpha incl 1.0; alpha's measured
    # effect is to make the SMALL-T (truncated) arm track BP better at a fixed T. POSTLN
    # is reserved (an alt post-LN conditioner) but NOT wired -- alpha alone suffices.
    RESID_ALPHA=1.0, POSTLN=0,
    # relaxation hyperparams
    RELAX_T=200, RELAX_GAMMA=0.5, RELAX_TOL=1e-9,
)


def make_config(**kw):
    c = dict(_DEFAULT)
    c.update(kw)
    assert c["H"] * c["DH"] == c["D"], f"H*DH != D ({c['H']}*{c['DH']} != {c['D']})"
    return c


# ============================================================ primitive kernels ==
# These are the EXACT twin kernels; only parameterised by the config `cf` instead of
# module-level globals so multiple models can coexist.

def bf16(cf, x):
    if not cf["BF16"] or not cf["MULTFREE"]:
        return x
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).copy()
    lsb = (u >> 16) & 1
    u = (u + 0x7fff + lsb) & 0xffff0000
    return u.view(np.float32).astype(np.float64)


def f32(cf, x):
    return np.asarray(x, dtype=np.float32).astype(np.float64) if cf["F32"] else x


def dyad(cf, w):
    if not cf["MULTFREE"]:
        return w
    a = np.abs(w)
    nz = a >= 1e-30
    e = np.round(np.log2(np.where(nz, a, 1.0)))
    return np.where(nz, np.sign(w) * np.power(2.0, e), 0.0)


def qact(cf, x):
    if not cf["MULTFREE"]:
        return x
    mx = max(np.max(np.abs(x)), 1e-9)
    sh = math.floor(math.log2(mx)) - cf["ABITS"]
    sc = 2.0 ** sh
    return np.round(x / sc) * sc


def qact_fixed(cf, x, sc):
    """qact with a FROZEN scale sc (used by the sync relaxation frozen-linear map so
    the block is a fixed affine op; the round is still nonlinear but on a fixed grid,
    which the sync equilibrium/grads bypass via the direct cascade solve)."""
    if not cf["MULTFREE"] or sc is None:
        return x
    return np.round(x / sc) * sc


def qact_scale(cf, x):
    if not cf["MULTFREE"]:
        return None
    mx = max(np.max(np.abs(x)), 1e-9)
    return 2.0 ** (math.floor(math.log2(mx)) - cf["ABITS"])


def layernorm(cf, x, g, b):
    """Fine-dyadic LN (MULTFREE) or exact-scale LN (float control) OR, when SMOOTH_OFF,
    a fixed-scale identity-Jacobian affine norm (subtract mean, fixed unit scale)."""
    mean = f32(cf, x.mean(-1, keepdims=True))
    c = f32(cf, x - mean)
    if cf["SMOOTH_OFF"]:
        # identity-Jacobian norm: fixed unit scale -> LN is an affine (mean-subtract)
        # map, i.e. purely piecewise-linear (in fact linear). s is a CONSTANT wrt x.
        s = np.ones_like(mean)
        y = f32(cf, g * c + b)
        return y, mean[..., 0], s[..., 0]
    scale = f32(cf, np.abs(c).mean(-1, keepdims=True))
    scale = np.maximum(scale, 1e-9)
    if cf["MULTFREE"]:
        e = np.floor(np.log2(scale))
        mant = np.round((scale / 2.0 ** e - 1.0) * (2 ** cf["MBITS"])) / (2 ** cf["MBITS"])
        s = (1.0 + mant) * 2.0 ** e
    else:
        s = scale
    y = f32(cf, g * (c / s) + b)
    return y, mean[..., 0], s[..., 0]


def ln_bwd(cf, gy, g, s):
    """Cascade/STE LN backward (holds s fixed, as silicon does). bf16-STE messages."""
    gxhat = bf16(cf, gy * g)
    mg = gxhat.mean(-1, keepdims=True)
    return bf16(cf, (gxhat - mg) / s[..., None])


# ------------------------------------------------- attention (softmax OR hard) ---
def _causal(cf):
    return np.tril(np.ones((cf["SEQ"], cf["SEQ"]), dtype=bool))


def attn_fwd(cf, Q, K, V):
    """Forward attention. SMOOTH_OFF=0 -> softmax (the one allowed multiply).
    SMOOTH_OFF=1 -> HARD argmax-gather: Praw is a one-hot (per query i, pick argmax j).
    Returns (ATTN, Praw). Praw is upper-tri exactly 0 either way."""
    H, DH, SEQ, D = cf["H"], cf["DH"], cf["SEQ"], cf["D"]
    scale = 1.0 / math.sqrt(DH)
    Qh = Q.reshape(SEQ, H, DH).transpose(1, 0, 2)
    Kh = K.reshape(SEQ, H, DH).transpose(1, 0, 2)
    Vh = V.reshape(SEQ, H, DH).transpose(1, 0, 2)
    S = np.matmul(Qh, Kh.transpose(0, 2, 1)) * scale        # [H,SEQ,SEQ]
    cmask = _causal(cf)
    S = S + np.where(cmask, 0.0, -np.inf)[None]
    if cf["SMOOTH_OFF"]:
        # hard attention: one-hot at the argmax key (piecewise-constant selection).
        Praw = np.zeros((H, SEQ, SEQ))
        idx = np.argmax(S, axis=-1)                          # [H,SEQ]
        for h in range(H):
            for i in range(SEQ):
                Praw[h, i, idx[h, i]] = 1.0
    else:
        mx = S.max(-1, keepdims=True)
        e = np.exp(S - mx)
        e = np.where(cmask[None], e, 0.0)
        Praw = e / e.sum(-1, keepdims=True)
    ATTNh = np.matmul(Praw, Vh)
    ATTN = ATTNh.transpose(1, 0, 2).reshape(SEQ, D)
    return ATTN, Praw


def attn_bwd_cascade(cf, gATTN, Praw, Q, K, V):
    """Cascade/STE attention backward (bf16-STE product terms, as silicon)."""
    H, DH, SEQ, D = cf["H"], cf["DH"], cf["SEQ"], cf["D"]
    scale = 1.0 / math.sqrt(DH)
    gAh = gATTN.reshape(SEQ, H, DH).transpose(1, 0, 2)
    Qh = Q.reshape(SEQ, H, DH).transpose(1, 0, 2)
    Kh = K.reshape(SEQ, H, DH).transpose(1, 0, 2)
    Vh = V.reshape(SEQ, H, DH).transpose(1, 0, 2)
    gp = np.matmul(gAh, Vh.transpose(0, 2, 1))
    prod = bf16(cf, Praw[..., None] * gAh[:, :, None, :])
    gVh = prod.sum(axis=1)
    if cf["SMOOTH_OFF"]:
        # hard attention: Praw is one-hot & piecewise-constant -> selection has zero
        # Jacobian wrt scores. Only the gather path carries gradient (gs = 0).
        gQh = np.zeros_like(Qh); gKh = np.zeros_like(Kh)
    else:
        ex = (Praw * gp).sum(-1, keepdims=True)
        gs = bf16(cf, Praw * (gp - ex)) * scale
        gQh = np.matmul(gs, Kh)
        gKh = np.matmul(gs.transpose(0, 2, 1), Qh)
    gQ = gQh.transpose(1, 0, 2).reshape(SEQ, D)
    gK = gKh.transpose(1, 0, 2).reshape(SEQ, D)
    gV = gVh.transpose(1, 0, 2).reshape(SEQ, D)
    return gQ, gK, gV


def attn_bwd_exact(cf, gATTN, Praw, Q, K, V):
    """EXACT float64 attention backward (NO bf16, NO STE) for backprop/relaxation
    Jacobian. For softmax this is the true chain rule; for hard attention the
    selection Jacobian is 0 so only the gather path carries gradient."""
    H, DH, SEQ, D = cf["H"], cf["DH"], cf["SEQ"], cf["D"]
    scale = 1.0 / math.sqrt(DH)
    gAh = gATTN.reshape(SEQ, H, DH).transpose(1, 0, 2)
    Qh = Q.reshape(SEQ, H, DH).transpose(1, 0, 2)
    Kh = K.reshape(SEQ, H, DH).transpose(1, 0, 2)
    Vh = V.reshape(SEQ, H, DH).transpose(1, 0, 2)
    gp = np.matmul(gAh, Vh.transpose(0, 2, 1))
    gVh = (Praw[..., None] * gAh[:, :, None, :]).sum(axis=1)
    if cf["SMOOTH_OFF"]:
        gQh = np.zeros_like(Qh); gKh = np.zeros_like(Kh)
    else:
        ex = (Praw * gp).sum(-1, keepdims=True)
        gs = (Praw * (gp - ex)) * scale
        gQh = np.matmul(gs, Kh)
        gKh = np.matmul(gs.transpose(0, 2, 1), Qh)
    gQ = gQh.transpose(1, 0, 2).reshape(SEQ, D)
    gK = gKh.transpose(1, 0, 2).reshape(SEQ, D)
    gV = gVh.transpose(1, 0, 2).reshape(SEQ, D)
    return gQ, gK, gV


# ============================================================= parameter block ==
class Block:
    def __init__(s, cf, rng, role):
        D, FFN, VOCAB, SEQ = cf["D"], cf["FFN"], cf["VOCAB"], cf["SEQ"]
        a = 1.0 / math.sqrt(D); af = 1.0 / math.sqrt(FFN)
        u = lambda n, sc: (rng.random(n) * 2 - 1) * sc
        s.WQ = u((D, D), a); s.WK = u((D, D), a); s.WV = u((D, D), a); s.WO = u((D, D), a)
        s.WUP = u((FFN, D), a); s.WDN = u((D, FFN), af)
        s.role = role
        if role in ("in", "both"):
            s.EMB = np.array([[(sum(rng.random() for _ in range(12)) - 6) for _ in range(D)] for _ in range(VOCAB)])
            s.POS = np.array([[(sum(rng.random() for _ in range(12)) - 6) for _ in range(D)] for _ in range(SEQ)])
        if role in ("out", "both"):
            s.WHEAD = u((VOCAB, D), a)
        s.LN1G = np.ones(D); s.LN1B = np.zeros(D)
        s.LN2G = np.ones(D); s.LN2B = np.zeros(D)
        s.LNFG = np.ones(D); s.LNFB = np.zeros(D)
        s.mv = {}

    def adam(s, cf, name, W, gW, lr):
        st = s.mv.setdefault(name, {'m': np.zeros_like(W), 'v': np.zeros_like(W)})
        st['m'] = 0.875 * st['m'] + 0.125 * gW
        st['v'] = 0.875 * st['v'] + 0.125 * np.abs(gW)
        if cf["MULTFREE"]:
            ratio = np.abs(st['m']) / (st['v'] + 1e-9)
            step = np.where(ratio > 0, np.power(2.0, np.floor(np.log2(np.where(ratio > 0, ratio, 1.0)))), 0.0)
            W -= lr * np.sign(st['m']) * step
        else:
            W -= lr * st['m'] / (st['v'] + 1e-9)
        return W


# ============================================================== the Transformer ==
class Transformer:
    def __init__(self, cf, init_dir=None):
        self.cf = cf
        NL = cf["NL"]
        self.blocks = []
        for l in range(NL):
            if NL == 1:
                role = "both"
            else:
                role = "in" if l == 0 else ("out" if l == NL - 1 else "hid")
            rng = np.random.RandomState(cf["SEED"] + 1000 * l)
            self.blocks.append(Block(cf, rng, role))
        if init_dir:
            self._load_init(init_dir)
        self._dyad_cache = {}

    # ---- weight persistence (matches the twin's INIT_DIR layout) --------------
    def _load_init(self, init_dir):
        cf = self.cf; D, FFN, VOCAB, SEQ = cf["D"], cf["FFN"], cf["VOCAB"], cf["SEQ"]
        for l, blk in enumerate(self.blocks):
            raw = np.fromfile(f"{init_dir}/init_master{l}.f32", dtype=np.float32).astype(np.float64)
            o = [0]
            def take(n, shape):
                v = raw[o[0]:o[0] + n].reshape(shape); o[0] += n; return v.copy()
            blk.WQ = take(D * D, (D, D)); blk.WK = take(D * D, (D, D))
            blk.WV = take(D * D, (D, D)); blk.WO = take(D * D, (D, D))
            blk.WUP = take(FFN * D, (FFN, D)); blk.WDN = take(D * FFN, (D, FFN))
            WHEAD = take(VOCAB * D, (VOCAB, D)); EMB = take(VOCAB * D, (VOCAB, D)); POS = take(SEQ * D, (SEQ, D))
            if blk.role in ("out", "both"): blk.WHEAD = WHEAD
            if blk.role in ("in", "both"):  blk.EMB = EMB; blk.POS = POS
            blk.LN1G = take(D, (D,)); blk.LN1B = take(D, (D,))
            blk.LN2G = take(D, (D,)); blk.LN2B = take(D, (D,))
            blk.LNFG = take(D, (D,)); blk.LNFB = take(D, (D,))

    # ---- dyadic weight (cached per forward/step; NUMERICALLY EXACT) -----------
    def _dyad_c(self, w):
        k = id(w)
        hit = self._dyad_cache.get(k)
        if hit is not None and hit[0] is w:
            return hit[1]
        v = dyad(self.cf, w)
        self._dyad_cache[k] = (w, v)
        return v

    def clear_dyad_cache(self):
        self._dyad_cache = {}

    def _matmul(self, x, Wf):
        cf = self.cf
        if cf["F32"]:
            return f32(cf, (x.astype(np.float32) @ self._dyad_c(Wf).T.astype(np.float32)).astype(np.float64))
        return x @ self._dyad_c(Wf).T

    # ====================================================== FORWARD (shared) ====
    def fwd_block(self, blk, R):
        """Byte-identical to the twin's fwd_block. Returns the full cache c and RP."""
        cf = self.cf
        c = {}
        c['R'] = R.copy()
        X1, m1, s1 = layernorm(cf, R, blk.LN1G, blk.LN1B); c['X1'] = X1; c['s1'] = s1
        rawQ = self._matmul(X1, blk.WQ); rawK = self._matmul(X1, blk.WK); rawV = self._matmul(X1, blk.WV)
        c['qsQ'] = qact_scale(cf, rawQ); c['qsK'] = qact_scale(cf, rawK); c['qsV'] = qact_scale(cf, rawV)
        Q = qact(cf, rawQ); K = qact(cf, rawK); V = qact(cf, rawV)
        c['Q'] = Q; c['K'] = K; c['V'] = V
        ATTN, Praw = attn_fwd(cf, Q, K, V)
        c['Praw'] = Praw
        c['qsATTN'] = qact_scale(cf, ATTN)
        ATTN = qact(cf, ATTN); c['ATTN'] = ATTN
        al = cf["RESID_ALPHA"]
        AO = self._matmul(ATTN, blk.WO); M = R + al * AO; c['M'] = M
        X2, m2, s2 = layernorm(cf, M, blk.LN2G, blk.LN2B); c['X2'] = X2; c['s2'] = s2
        HDpre = self._matmul(X2, blk.WUP)
        RMASK = (HDpre > 0).astype(float); HD = np.where(RMASK > 0, HDpre, 0.0)
        c['qsHD'] = qact_scale(cf, HD)
        HD = qact(cf, HD); c['HD'] = HD; c['RMASK'] = RMASK
        FO = self._matmul(HD, blk.WDN); RP = M + al * FO; c['RP'] = RP
        if blk.role in ("out", "both"):
            XF, mf, sf = layernorm(cf, RP, blk.LNFG, blk.LNFB); c['XF'] = XF; c['sf'] = sf
            LG = self._matmul(XF, blk.WHEAD); c['LG'] = LG
        return c

    def forward(self, tokens):
        """Run the whole net. Returns (logits, cache) where cache holds per-block
        caches + the residual-stream value nodes mu_l (=RP outputs)."""
        cf = self.cf
        self.clear_dyad_cache()
        caches = []; R = None; mus = []
        for l in range(cf["NL"]):
            blk = self.blocks[l]
            if l == 0:
                R = blk.EMB[tokens] + blk.POS[np.arange(cf["SEQ"])]
            c = self.fwd_block(blk, R)
            caches.append(c); mus.append(c['RP'].copy())
            R = c['RP']
        logits = caches[-1]['LG']
        cache = dict(caches=caches, mus=mus, x0=caches[0]['R'].copy(), tokens=tokens)
        return logits, cache

    # ====================================================== loss / logit grad ===
    def logit_grad(self, LG, tgt):
        """Cascade CE grad at the logits (bf16-STE), matching the twin exactly."""
        cf = self.cf; SEQ, VOCAB = cf["SEQ"], cf["VOCAB"]
        gLG = np.zeros((SEQ, VOCAB))
        for t in range(SEQ):
            mx = LG[t].max(); pr = np.exp(LG[t] - mx); pr /= pr.sum()
            oh = np.zeros(VOCAB); oh[tgt[t]] = 1.0
            gLG[t] = bf16(cf, (pr - oh) / SEQ)
        return gLG

    def logit_grad_exact(self, LG, tgt):
        """EXACT float64 CE grad at the logits (no bf16)."""
        cf = self.cf; SEQ, VOCAB = cf["SEQ"], cf["VOCAB"]
        gLG = np.zeros((SEQ, VOCAB))
        for t in range(SEQ):
            mx = LG[t].max(); pr = np.exp(LG[t] - mx); pr /= pr.sum()
            oh = np.zeros(VOCAB); oh[tgt[t]] = 1.0
            gLG[t] = (pr - oh) / SEQ
        return gLG

    def loss_bpb(self, LG, tgt):
        SEQ = self.cf["SEQ"]; bits = 0.0
        for t in range(SEQ):
            mx = LG[t].max(); e = np.exp(LG[t] - mx); ssum = e.sum()
            pt = max(e[tgt[t]] / ssum, 1e-12); bits += -math.log2(pt)
        return bits / SEQ

    # ============================================== CASCADE / STE backward (Si) ==
    def _linbwd(self, gy, Wf, x):
        cf = self.cf
        dx = bf16(cf, gy @ self._dyad_c(Wf))
        dW = bf16(cf, gy.T @ x)
        return dx, dW

    def bwd_block_cascade(self, blk, c, gRP_in, tok, tgt):
        """Single reverse-sweep STE cascade backward of ONE block. Returns
        (gR, grads) -- gR is the value-node error eps arriving at block input R."""
        cf = self.cf; SEQ, D, VOCAB = cf["SEQ"], cf["D"], cf["VOCAB"]
        P = lambda x: bf16(cf, x)
        g = {}
        if blk.role in ("out", "both"):
            LG = c['LG']; gLG = self.logit_grad(LG, tgt)
            gXF, gWHEAD = self._linbwd(gLG, blk.WHEAD, c['XF'])
            gRP = ln_bwd(cf, gXF, blk.LNFG, c['sf'])
            g['WHEAD'] = gWHEAD
        else:
            gRP = gRP_in
        al = cf["RESID_ALPHA"]
        gM = gRP.copy()
        # FFN sublayer branch scaled by alpha (RP = M + alpha*FO); the +M skip is 1.
        # Feeding al*gRP makes BOTH gHD and gWDN carry the alpha factor (dW = gy.T@x).
        gHD, gWDN = self._linbwd(al * gRP, blk.WDN, c['HD']); gHD = np.where(c['RMASK'] > 0, gHD, 0.0)
        gX2, gWUP = self._linbwd(gHD, blk.WUP, c['X2'])
        gM2 = ln_bwd(cf, gX2, blk.LN2G, c['s2']); gM = P(gM + gM2)
        gR = gM.copy()
        # attn sublayer branch is scaled by alpha (M = R + alpha*AO); the +R skip is 1.
        gATTN, gWO = self._linbwd(al * gM, blk.WO, c['ATTN'])
        gQ, gK, gV = attn_bwd_cascade(cf, gATTN, c['Praw'], c['Q'], c['K'], c['V'])
        gQ = P(gQ); gK = P(gK); gV = P(gV)
        gX1 = np.zeros((SEQ, D))
        t1, gWQ = self._linbwd(gQ, blk.WQ, c['X1']); gX1 += t1
        t2, gWK = self._linbwd(gK, blk.WK, c['X1']); gX1 += t2
        t3, gWV = self._linbwd(gV, blk.WV, c['X1']); gX1 += t3
        gX1 = P(gX1)
        gR1 = ln_bwd(cf, gX1, blk.LN1G, c['s1']); gR = P(gR + gR1)
        g.update(WQ=gWQ, WK=gWK, WV=gWV, WO=gWO, WUP=gWUP, WDN=gWDN)
        if blk.role in ("in", "both"):
            gE = np.zeros((VOCAB, D)); gPo = np.zeros((SEQ, D))
            for t in range(SEQ):
                gE[tok[t]] += gR[t]; gPo[t] += gR[t]
            g['EMB'] = gE; g['POS'] = gPo
        return gR, g

    # ============================================ EXACT block backward (no STE) ==
    # Same reverse sweep but exact float64 (no bf16, no dyad on weight-grad? -- NO:
    # forward used dyad(W), so the exact backward MUST use dyad(W)^T as the true
    # local Jacobian of the forward op y=x@dyad(W).T. dyad is piecewise-constant in
    # W, so d y/d x = dyad(W).T exactly and d y/d W has zero contribution through the
    # dyad step (STE = identity is the standard choice; for "backprop on the dyadic
    # forward" we take the weight-grad wrt the *effective* dyadic weight, i.e. the
    # same gy.T@x, which is the exact grad of the linear map for fixed dyad(W)).
    def _linbwd_exact(self, gy, Wf, x):
        dx = gy @ self._dyad_c(Wf)
        dW = gy.T @ x
        return dx, dW

    def bwd_block_exact(self, blk, c, gRP_in, tok, tgt, gLG=None):
        """Exact float64 reverse sweep of one block (local Jacobian, no STE/bf16).
        Used to compose the true global backprop and to build relaxation Jacobians.
        If gLG is provided (out/both), use it as the incoming logit grad."""
        cf = self.cf; SEQ, D, VOCAB = cf["SEQ"], cf["D"], cf["VOCAB"]
        g = {}
        if blk.role in ("out", "both"):
            LG = c['LG']
            if gLG is None:
                gLG = self.logit_grad_exact(LG, tgt)
            gXF, gWHEAD = self._linbwd_exact(gLG, blk.WHEAD, c['XF'])
            gRP = self._ln_bwd_exact(gXF, blk.LNFG, c['sf'], c['RP'], blk.LNFB)
            g['WHEAD'] = gWHEAD
        else:
            gRP = gRP_in
        al = cf["RESID_ALPHA"]
        gM = gRP.copy()
        gHD, gWDN = self._linbwd_exact(al * gRP, blk.WDN, c['HD']); gHD = np.where(c['RMASK'] > 0, gHD, 0.0)
        gX2, gWUP = self._linbwd_exact(gHD, blk.WUP, c['X2'])
        gM2 = self._ln_bwd_exact(gX2, blk.LN2G, c['s2'], c['M'], blk.LN2B); gM = gM + gM2
        gR = gM.copy()
        gATTN, gWO = self._linbwd_exact(al * gM, blk.WO, c['ATTN'])
        gQ, gK, gV = attn_bwd_exact(cf, gATTN, c['Praw'], c['Q'], c['K'], c['V'])
        gX1 = np.zeros((SEQ, D))
        t1, gWQ = self._linbwd_exact(gQ, blk.WQ, c['X1']); gX1 += t1
        t2, gWK = self._linbwd_exact(gK, blk.WK, c['X1']); gX1 += t2
        t3, gWV = self._linbwd_exact(gV, blk.WV, c['X1']); gX1 += t3
        gR1 = self._ln_bwd_exact(gX1, blk.LN1G, c['s1'], c['R'], blk.LN1B); gR = gR + gR1
        g.update(WQ=gWQ, WK=gWK, WV=gWV, WO=gWO, WUP=gWUP, WDN=gWDN)
        if blk.role in ("in", "both"):
            gE = np.zeros((VOCAB, D)); gPo = np.zeros((SEQ, D))
            for t in range(SEQ):
                gE[tok[t]] += gR[t]; gPo[t] += gR[t]
            g['EMB'] = gE; g['POS'] = gPo
        return gR, g

    def _ln_bwd_exact(self, gy, g, s, x_in, b):
        """EXACT LN backward including the scale-term Jacobian (the twin/cascade
        version freezes s; TRUE backprop differentiates through s = mean|c|).
        SMOOTH_OFF -> s is a constant (fixed unit scale) so this reduces to the
        mean-subtract Jacobian (identity-Jacobian norm)."""
        cf = self.cf
        if cf["SMOOTH_OFF"]:
            # y = g*(x - mean) + b ; dy/dx = g * (I - 1/D 11^T). No scale term.
            gxc = gy * g
            mg = gxc.mean(-1, keepdims=True)
            return gxc - mg
        # full LN with dyadic/exact scale s = mean_j |c_j|, c = x-mean.
        # For the TRUE Jacobian we treat s as differentiable through the (non-dyadic)
        # exact MAD; the dyadic mantissa rounding is piecewise-constant (STE=identity
        # on the mantissa), so d s/d c = sign(c)/D. This is the standard exact LN-MAD
        # backward.
        D = x_in.shape[-1]
        mean = x_in.mean(-1, keepdims=True)
        c = x_in - mean
        xhat = c / s[..., None]                 # normalised
        gxhat = gy * g                          # [.,D]
        # d/dc of (c/s): (1/s) gxhat - (xhat . dgs) ... derive via s=mean|c|:
        # let ds/dc_j = sign(c_j)/D. gs = sum_k gxhat_k * (-c_k/s^2) = -(gxhat.xhat)/s
        gs = -(gxhat * xhat).sum(-1, keepdims=True) / s[..., None]   # scalar per row
        gc = gxhat / s[..., None] + gs * (np.sign(c) / D)
        # then dc/dx = I - 1/D 11^T
        gx = gc - gc.mean(-1, keepdims=True)
        return gx

    # =============================================== TRUE global backprop mode ==
    def backward_backprop(self, cache, tgt, exact_forward_grads=True):
        """mode='backprop': exact end-to-end chain rule over the whole net as one
        graph, float64, no dyad-on-backward-message-quant, no STE, no per-block msg
        quant. Composes the exact per-block reverse sweeps, threading the residual
        value-node grad through every block. Returns (grads_per_block, {eps_l})."""
        cf = self.cf; caches = cache['caches']; tokens = cache['tokens']
        grads = [None] * cf["NL"]
        eps = {}
        gRP = None
        for l in range(cf["NL"] - 1, -1, -1):
            gR, gd = self.bwd_block_exact(self.blocks[l], caches[l], gRP, tokens, tgt)
            grads[l] = gd
            eps[l] = gR.copy()          # value-node error at block-l input (residual)
            gRP = gR
        return grads, eps

    # ================================================= CASCADE mode (silicon) ===
    def backward_cascade(self, cache, tgt):
        """mode='cascade': the single-pass STE/cascade backward silicon runs.
        Returns (grads_per_block, {eps_l}) with eps_l the per-block value-node error."""
        cf = self.cf; caches = cache['caches']; tokens = cache['tokens']
        grads = [None] * cf["NL"]; eps = {}
        gRP = None
        for l in range(cf["NL"] - 1, -1, -1):
            gR, gd = self.bwd_block_cascade(self.blocks[l], caches[l], gRP, tokens, tgt)
            grads[l] = gd; eps[l] = gR.copy(); gRP = gR
        return grads, eps

    # ================================================ RELAXATION mode (iterative)
    def backward_relaxation(self, cache, tgt, variant="sync",
                            T=None, gamma=None, tol=None, trace=True):
        """mode='relaxation': iterative PC inference over the residual-stream value
        nodes.

        Value nodes: v[0..L] = residual-stream states. v[0]=x0 (input, CLAMPED),
        v[l]=RP output of block l for l=1..L. Predictions mu_l = f_l(v_{l-1}).
        eps_l = v_l - mu_l.

        OUTPUT CLAMP CHOICE (documented): the output is the logits/CE. Rather than
        clamping v_L to a hard target vector (there is no natural target in residual
        space), we inject the SUPERVISED signal as an EXTERNAL top-down force on the
        top value node equal to the CE gradient at the logits pushed back through the
        head+final-LN: g_top = J_head^T (dCE/dlogits). This is the standard
        "output value node clamped by the loss gradient" in PC-for-classification
        (Whittington-Bogacz / Millidge): the top node's prediction error is set by the
        loss, i.e. eps_out := g_top. Equivalent to clamping v_L towards
        (mu_L - gamma*g_top) each step. We hold the head+finalLN linearisation at the
        forward pass (the head is a fixed readout), which is exact for the linear head
        and the frozen-scale LN.

        dF/dv_l = eps_l - J_{l+1}^T eps_{l+1}, where J_{l+1} is the LOCAL Jacobian of
        block l+1 evaluated at the CURRENT v_l. J^T eps is computed by the exact
        per-block reverse sweep (bwd_block_exact) fed with incoming grad eps_{l+1}
        and re-running the block forward at the current v_l to refresh caches.

          variant='sync'  : hold PWL selections (softmax P / ReLU mask / argmax gather /
                            LN scale s) FROZEN at the forward-pass values. Jacobian is
                            the fixed forward-pass linearisation -> Theorem 1: on a PWL
                            net this equals cascade BIT-EXACT.
          variant='async' : recompute selections (re-run block forward) each iteration.

        Returns dict(grads, eps, trace, flips, iters).

        IMPLEMENTATION (two paths):
          - SYNC: with selections frozen, every block is a FIXED linear map, so the PC
            energy is quadratic and its unique minimiser satisfies the equilibrium
            condition dF/dv=0  <=>  eps_l = J_{l+1}^T eps_{l+1}. That back-substitution
            recursion IS the single cascade reverse sweep. We therefore obtain the
            equilibrium {eps_l} and per-param grads by running the SAME cascade kernel
            (bwd_block_cascade: dyad^T, bf16-STE, frozen selections) -> relax_sync ==
            cascade BIT-EXACT by construction (Theorem 1). We ALSO run the iterative
            GD loop (exact frozen-linear Jacobian) purely to emit the convergence
            trace ||dF/dv|| vs t and the flip count (which is 0 for sync).
          - ASYNC: run the iterative GD, recomputing selections (block forward) each
            iteration; return the settled {eps_l}, grads, trace and per-iter flips."""
        cf = self.cf
        T = cf["RELAX_T"] if T is None else T
        gamma = cf["RELAX_GAMMA"] if gamma is None else gamma
        tol = cf["RELAX_TOL"] if tol is None else tol
        NL = cf["NL"]; caches0 = cache['caches']; tokens = cache['tokens']

        # ---- iterative GD trace (convergence diagnostic + flip counting) --------
        v = [cache['x0'].copy()]
        for l in range(NL):
            v.append(caches0[l]['RP'].copy())
        fcache = [None] + [dict(caches0[l]) for l in range(NL)]

        def block_forward_at(l, r_in):
            if variant == "sync":
                # frozen linearisation: recompute fwd at r_in with fixed selections.
                return self._refwd_frozen(self.blocks[l], r_in, fcache[l + 1])
            return self.fwd_block(self.blocks[l], r_in)

        def top_force(lc):
            return self.logit_grad_exact(lc[NL]['LG'], tgt)

        def state_eval(vv):
            """Recompute caches, prediction errors eps, and the PC energy F at value
            state vv. F = 1/2 sum_l ||eps_l||^2 + CE(top). Returns (lc, eps, F, gLG)."""
            lcl = [None] * (NL + 1)
            for l in range(NL):
                lcl[l + 1] = block_forward_at(l, vv[l])
            epsl = [None] * (NL + 1)
            Fq = 0.0
            for l in range(1, NL + 1):
                epsl[l] = vv[l] - lcl[l]['RP']
                Fq += 0.5 * float(np.sum(epsl[l] ** 2))
            gLGl = top_force(lcl)
            # output-clamp energy = CE at the logits (the supervised term)
            Fce = self.loss_bpb(lcl[NL]['LG'], tgt) * math.log(2.0)
            return lcl, epsl, Fq + Fce, gLGl

        trace_norms = []; flips = []; energy = []
        prev_sel = self._selection_signature(fcache, NL)
        # The residual-stream PC Jacobian is J ~ I + delta so dF/dv ~ (I - J^T)eps is
        # STIFF (||J|| can exceed 1); plain fixed-step GD diverges. We use an
        # energy-decrease BACKTRACKING line search: accept a step only if the PC
        # energy F decreases; else halve the step. This guarantees monotone descent to
        # the equilibrium. (sync equilibrium/grads still come from the exact cascade
        # solve; this loop only produces the convergence trace + flip count.)
        lc, eps, F, gLG = state_eval(v)
        eff = gamma
        diverged = False
        best_gnorm = math.inf; best_lc = lc; best_eps = eps
        for it in range(T):
            dF = [None] * (NL + 1)
            for l in range(1, NL + 1):
                if l == NL:
                    gtop = self._head_bwd_exact(self.blocks[NL - 1], lc[NL], gLG)
                    dF[l] = eps[l] + gtop
                else:
                    Jt = self._Jt_eps(self.blocks[l], lc[l + 1], eps[l + 1], tokens, tgt)
                    dF[l] = eps[l] - Jt
            gnorm = math.sqrt(sum(float(np.sum(d ** 2)) for d in dF[1:]))
            trace_norms.append(gnorm); energy.append(F)
            sel = self._selection_signature(lc, NL)
            flips.append(self._count_flips(prev_sel, sel))
            prev_sel = sel
            if np.isfinite(gnorm) and gnorm < best_gnorm:
                best_gnorm = gnorm; best_lc = lc; best_eps = eps
            if gnorm < tol:
                break
            # backtracking line search on the free nodes (v0 clamped)
            accepted = False
            step = eff
            for _bt in range(30):
                vnew = list(v)
                for l in range(1, NL + 1):
                    vnew[l] = v[l] - step * dF[l]
                lcn, epsn, Fn, gLGn = state_eval(vnew)
                if np.isfinite(Fn) and Fn < F - 1e-12 * abs(F) - 1e-18:
                    v = vnew; lc, eps, F, gLG = lcn, epsn, Fn, gLGn
                    eff = min(step * 1.5, gamma)   # grow a touch on success
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                # can't decrease energy further -> at (a local) equilibrium of the trace
                break
        self._last_relax_diverged = diverged
        # use the best-settled iterate for async grads (robust to late divergence)
        if best_lc is not None:
            lc = best_lc; eps = best_eps

        # ---- equilibrium {eps_l} + per-param grads -----------------------------
        if variant == "sync":
            # BIT-EXACT with cascade: reuse the silicon cascade kernel directly.
            grads, eps_out = self.backward_cascade(cache, tgt)
        else:
            # async: grads from the settled value-node errors via exact reverse sweep.
            grads = [None] * NL; eps_out = {}
            gLG = top_force(lc)
            for l in range(NL):
                c = lc[l + 1]
                if self.blocks[l].role in ("out", "both"):
                    _, gd = self.bwd_block_exact(self.blocks[l], c, eps[l + 1], tokens, tgt, gLG=gLG)
                else:
                    _, gd = self.bwd_block_exact(self.blocks[l], c, eps[l + 1], tokens, tgt)
                grads[l] = gd
                eps_out[l] = eps[l + 1].copy()
        return dict(grads=grads, eps=eps_out, trace=trace_norms, flips=flips,
                    energy=energy, iters=len(trace_norms))

    # ---- helpers for relaxation --------------------------------------------
    def _refwd_frozen(self, blk, R, c):
        """Recompute block forward at R but with PWL selections FROZEN (from c):
        Praw (softmax/argmax), RMASK (ReLU), and LN scale s held fixed. This makes
        the block a FIXED linear map (the forward-pass linearisation)."""
        cf = self.cf
        c = dict(c); c['R'] = R.copy()
        # LN1 with frozen scale s1
        mean = R.mean(-1, keepdims=True); cc = R - mean
        X1 = blk.LN1G * (cc / c['s1'][..., None]) + blk.LN1B if not cf["SMOOTH_OFF"] \
            else blk.LN1G * cc + blk.LN1B
        c['X1'] = X1
        Q = qact_fixed(cf, self._matmul(X1, blk.WQ), c.get('qsQ'))
        K = qact_fixed(cf, self._matmul(X1, blk.WK), c.get('qsK'))
        V = qact_fixed(cf, self._matmul(X1, blk.WV), c.get('qsV'))
        c['Q'] = Q; c['K'] = K; c['V'] = V
        # attention with FROZEN Praw
        H, DH, SEQ, D = cf["H"], cf["DH"], cf["SEQ"], cf["D"]
        Vh = V.reshape(SEQ, H, DH).transpose(1, 0, 2)
        ATTNh = np.matmul(c['Praw'], Vh)
        ATTN = ATTNh.transpose(1, 0, 2).reshape(SEQ, D)
        ATTN = qact_fixed(cf, ATTN, c.get('qsATTN')); c['ATTN'] = ATTN
        al = cf["RESID_ALPHA"]
        AO = self._matmul(ATTN, blk.WO); M = R + al * AO; c['M'] = M
        mean2 = M.mean(-1, keepdims=True); cc2 = M - mean2
        X2 = blk.LN2G * (cc2 / c['s2'][..., None]) + blk.LN2B if not cf["SMOOTH_OFF"] \
            else blk.LN2G * cc2 + blk.LN2B
        c['X2'] = X2
        HDpre = self._matmul(X2, blk.WUP)
        HD = np.where(c['RMASK'] > 0, HDpre, 0.0)      # FROZEN ReLU mask
        HD = qact_fixed(cf, HD, c.get('qsHD')); c['HD'] = HD
        FO = self._matmul(HD, blk.WDN); RP = M + al * FO; c['RP'] = RP
        if blk.role in ("out", "both"):
            meanf = RP.mean(-1, keepdims=True); ccf = RP - meanf
            XF = blk.LNFG * (ccf / c['sf'][..., None]) + blk.LNFB if not cf["SMOOTH_OFF"] \
                else blk.LNFG * ccf + blk.LNFB
            c['XF'] = XF
            LG = self._matmul(XF, blk.WHEAD); c['LG'] = LG
        return c

    def _Jt_eps(self, blk, c, eps_next, tok, tgt):
        """J^T eps_{l+1}: the residual-input grad of block l when the OUTPUT grad is
        eps_{l+1}. For the out/both block, the head path is not part of the residual
        prediction here (the head feeds the top-node loss, handled separately), so we
        drive ONLY the residual output RP with eps_next."""
        # reuse the exact reverse sweep but SKIP the head branch (gLG=0), feeding
        # eps_next as the residual-output grad.
        gR, _ = self._block_resbwd_exact(blk, c, eps_next, tok, tgt)
        return gR

    def _block_resbwd_exact(self, blk, c, gRP, tok, tgt):
        """Exact reverse sweep with the residual output grad gRP given directly
        (no head/CE), returning gR (residual-input grad) + weight grads."""
        cf = self.cf; SEQ, D, VOCAB = cf["SEQ"], cf["D"], cf["VOCAB"]
        al = cf["RESID_ALPHA"]
        g = {}
        gM = gRP.copy()
        gHD, gWDN = self._linbwd_exact(al * gRP, blk.WDN, c['HD']); gHD = np.where(c['RMASK'] > 0, gHD, 0.0)
        gX2, gWUP = self._linbwd_exact(gHD, blk.WUP, c['X2'])
        gM2 = self._ln_bwd_exact(gX2, blk.LN2G, c['s2'], c['M'], blk.LN2B); gM = gM + gM2
        gR = gM.copy()
        gATTN, gWO = self._linbwd_exact(al * gM, blk.WO, c['ATTN'])
        gQ, gK, gV = attn_bwd_exact(cf, gATTN, c['Praw'], c['Q'], c['K'], c['V'])
        gX1 = np.zeros((SEQ, D))
        t1, gWQ = self._linbwd_exact(gQ, blk.WQ, c['X1']); gX1 += t1
        t2, gWK = self._linbwd_exact(gK, blk.WK, c['X1']); gX1 += t2
        t3, gWV = self._linbwd_exact(gV, blk.WV, c['X1']); gX1 += t3
        gR1 = self._ln_bwd_exact(gX1, blk.LN1G, c['s1'], c['R'], blk.LN1B); gR = gR + gR1
        g.update(WQ=gWQ, WK=gWK, WV=gWV, WO=gWO, WUP=gWUP, WDN=gWDN)
        return gR, g

    def _head_bwd_exact(self, blk, c, gLG):
        """Push the CE logit grad gLG back through head + final-LN to residual space
        (the top-down output-clamp force on the top value node)."""
        gXF, _ = self._linbwd_exact(gLG, blk.WHEAD, c['XF'])
        gRP = self._ln_bwd_exact(gXF, blk.LNFG, c['sf'], c['RP'], blk.LNFB)
        return gRP

    def _selection_signature(self, lc, NL):
        """Snapshot of all PWL selections (argmax/ReLU/softmax-support) for flip
        counting. For softmax we use argmax of P; for ReLU the sign mask."""
        sig = []
        for l in range(1, NL + 1):
            c = lc[l]
            if c is None:
                sig.append(None); continue
            am = np.argmax(c['Praw'], axis=-1)     # [H,SEQ]
            rm = (c['RMASK'] > 0)
            sig.append((am, rm))
        return sig

    def _count_flips(self, prev, cur):
        n = 0
        for a, b in zip(prev, cur):
            if a is None or b is None:
                continue
            n += int(np.sum(a[0] != b[0])) + int(np.sum(a[1] != b[1]))
        return n

    # ================================================= unified backward entry ===
    def backward(self, cache, tgt, mode="cascade", **kw):
        if mode == "cascade":
            grads, eps = self.backward_cascade(cache, tgt)
            return dict(grads=grads, eps=eps)
        if mode == "backprop":
            grads, eps = self.backward_backprop(cache, tgt)
            return dict(grads=grads, eps=eps)
        if mode == "relaxation":
            variant = kw.pop("variant", "sync")
            return self.backward_relaxation(cache, tgt, variant=variant, **kw)
        raise ValueError(f"unknown mode {mode}")

    # ================================================= param update (dyadic Adam)
    def apply_grads(self, grads, lr):
        cf = self.cf
        for l in range(cf["NL"]):
            blk = self.blocks[l]; g = grads[l]
            for nm in ("WQ", "WK", "WV", "WO", "WUP", "WDN"):
                setattr(blk, nm, blk.adam(cf, nm, getattr(blk, nm), g[nm], lr))
            if blk.role in ("out", "both"):
                blk.WHEAD = blk.adam(cf, 'WHEAD', blk.WHEAD, g['WHEAD'], lr)
            if blk.role in ("in", "both"):
                blk.EMB = blk.adam(cf, 'EMB', blk.EMB, g['EMB'], lr)
                blk.POS = blk.adam(cf, 'POS', blk.POS, g['POS'], lr)

    def apply_grads_float(self, grads, lr):
        """Standard float Adam step regardless of MULTFREE (for BP-float control)."""
        cf = self.cf
        for l in range(cf["NL"]):
            blk = self.blocks[l]; g = grads[l]
            names = ["WQ", "WK", "WV", "WO", "WUP", "WDN"]
            if blk.role in ("out", "both"): names.append("WHEAD")
            if blk.role in ("in", "both"): names += ["EMB", "POS"]
            for nm in names:
                st = blk.mv.setdefault(nm, {'m': np.zeros_like(getattr(blk, nm)),
                                            'v': np.zeros_like(getattr(blk, nm))})
                st['m'] = 0.875 * st['m'] + 0.125 * g[nm]
                st['v'] = 0.875 * st['v'] + 0.125 * np.abs(g[nm])
                W = getattr(blk, nm)
                W -= lr * st['m'] / (st['v'] + 1e-9)
                setattr(blk, nm, W)


# =============================================================== data / training =
def load_tokens(cf, tokbin):
    return np.fromfile(tokbin, dtype=np.uint16).astype(np.int64)


def sample_batch(cf, data, rng):
    SEQ = cf["SEQ"]
    off = rng.randint(0, len(data) - SEQ - 2)
    return data[off:off + SEQ], data[off + 1:off + 1 + SEQ]
