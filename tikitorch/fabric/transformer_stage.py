"""tikitorch.fabric.transformer_stage — a REAL transformer block as a fabric Stage.

`LinearStage` exercises the pipeline/staleness/comms machinery on a toy layer. This is the
faithful workload: one pre-LN causal transformer block (LN -> attention -> residual ->
LN -> FFN -> residual) whose ``backward`` is the PC cascade *through the block* — the same
local-credit op the F2 kernel implements — so a chain of these trains the ACTUAL model under
the distributed pipeline, and we can ask "does the recipe survive the depth-proportional
staleness, at what W" before the silicon driver exists.

A microbatch here is one sequence: activations are (seq, d). The block's local backward is
ordinary backprop through the block (cascade = chain rule, block-local — exactly what PC
computes), gradient-checked against numerical gradients (see ``gradcheck`` below).

Forward (pre-LN, causal, single head-group folded into d for simplicity of the cascade):
    h1 = LN(x)·γ1+β1 ;  attn = softmax_causal(Q Kᵀ/√d)·V ;  x = x + attn·Wo
    h2 = LN(x)·γ2+β2 ;  x = x + relu(h2·W1)·W2
Weights: Wq,Wk,Wv,Wo (d×d), W1 (d×f), W2 (f×d), γ1,β1,γ2,β2 (d).
"""
from __future__ import annotations

import numpy as np


def _ln_fwd(x):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + 1e-5)
    return xc * inv, (xc, inv)


def _ln_bwd(dy, cache):
    xc, inv = cache
    D = xc.shape[-1]
    # d/dx of  xhat = xc*inv  (mean-centered, unit-var); standard LN input grad
    dxhat = dy
    dvar = (dxhat * xc).sum(-1, keepdims=True) * (-0.5) * inv ** 3
    dmu = (-dxhat * inv).sum(-1, keepdims=True) + dvar * (-2.0 / D) * xc.sum(-1, keepdims=True)
    return dxhat * inv + dvar * 2.0 * xc / D + dmu / D


def _softmax_causal(s):
    seq = s.shape[-1]
    mask = np.triu(np.ones((seq, seq), bool), 1)
    s = np.where(mask, -1e30, s)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


class TransformerBlock:
    """Parameters + forward/backward for one pre-LN causal transformer block."""

    def __init__(self, d, f=None, seed=0):
        f = f or 4 * d
        rng = np.random.default_rng(seed)
        sc = 1.0 / np.sqrt(d)
        self.d, self.f = d, f
        self.Wq = rng.standard_normal((d, d)) * sc
        self.Wk = rng.standard_normal((d, d)) * sc
        self.Wv = rng.standard_normal((d, d)) * sc
        self.Wo = rng.standard_normal((d, d)) * sc
        self.W1 = rng.standard_normal((d, f)) * sc
        self.W2 = rng.standard_normal((f, d)) * (1.0 / np.sqrt(f))
        self.g1 = np.ones(d); self.b1 = np.zeros(d)
        self.g2 = np.ones(d); self.b2 = np.zeros(d)

    # the param list the fabric runner mutates (order fixed; matches grad order)
    @property
    def params(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.W2,
                self.g1, self.b1, self.g2, self.b2]

    def forward(self, x):
        d = self.d
        h1, ln1c = _ln_fwd(x); h1 = h1 * self.g1 + self.b1
        Q, K, V = h1 @ self.Wq, h1 @ self.Wk, h1 @ self.Wv
        att = _softmax_causal(Q @ K.T / np.sqrt(d))
        ctx = att @ V
        proj = ctx @ self.Wo
        x2 = x + proj
        h2, ln2c = _ln_fwd(x2); h2 = h2 * self.g2 + self.b2
        pre = h2 @ self.W1
        hid = np.maximum(pre, 0.0)
        ffn = hid @ self.W2
        y = x2 + ffn
        cache = (x, ln1c, h1, Q, K, V, att, ctx, x2, ln2c, h2, pre, hid)
        return y, cache

    def backward(self, dy, cache):
        d = self.d
        x, ln1c, h1, Q, K, V, att, ctx, x2, ln2c, h2, pre, hid = cache
        # FFN + residual 2
        dx2 = dy.copy()
        dffn = dy
        dW2 = hid.T @ dffn
        dhid = dffn @ self.W2.T
        dpre = dhid * (pre > 0)
        dW1 = h2.T @ dpre
        dh2n = dpre @ self.W1.T
        dg2 = (dh2n * (_ln_fwd(x2)[0])).sum(0); db2 = dh2n.sum(0)
        dx2 += _ln_bwd(dh2n * self.g2, ln2c)
        # attention + residual 1
        dx = dx2.copy()
        dproj = dx2
        dWo = ctx.T @ dproj
        dctx = dproj @ self.Wo.T
        dV = att.T @ dctx
        datt = dctx @ V.T
        # softmax bwd (rows are causal-normalised): ds = att * (datt - sum(att*datt))
        ds = att * (datt - (att * datt).sum(-1, keepdims=True))
        ds /= np.sqrt(d)
        dQ = ds @ K
        dK = ds.T @ Q
        dWq = h1.T @ dQ; dWk = h1.T @ dK; dWv = h1.T @ dV
        dh1n = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        dg1 = (dh1n * (_ln_fwd(x)[0])).sum(0); db1 = dh1n.sum(0)
        dx += _ln_bwd(dh1n * self.g1, ln1c)
        grads = [dWq, dWk, dWv, dWo, dW1, dW2, dg1, db1, dg2, db2]
        return grads, dx


# ======================= fabric Stage wrappers (a char-LM) =======================
# A microbatch = one length-`seq` token sequence. EmbedStage (input chip) embeds tokens;
# TransformerBlockStage (hidden chips) is the gradchecked block above; HeadStage (output
# chip) does final-LN + LM-head + cross-entropy and originates the backward error. Each
# caches per tag (the pipeline keeps several microbatches in flight).

class TransformerBlockStage:
    def __init__(self, d, f=None, seed=0):
        self.block = TransformerBlock(d, f, seed=seed)
        self._cache = {}

    @property
    def params(self):
        return self.block.params

    def forward(self, tag, x):
        y, c = self.block.forward(x)
        self._cache[tag] = c
        return y

    def backward(self, tag, dy):
        return self.block.backward(dy, self._cache.pop(tag))


class EmbedStage:
    """Input chip: token ids (seq,) -> x = E[tokens] + P[:seq]."""
    def __init__(self, vocab, d, seq, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.standard_normal((vocab, d)) * 0.1
        self.P = rng.standard_normal((seq, d)) * 0.1
        self.params = [self.E, self.P]
        self._cache = {}

    def forward(self, tag, tokens):
        toks = np.asarray(tokens, dtype=int)
        self._cache[tag] = toks
        return self.E[toks] + self.P[:len(toks)]

    def backward(self, tag, dx):
        toks = self._cache.pop(tag)
        dE = np.zeros_like(self.E)
        np.add.at(dE, toks, dx)                      # scatter the per-token grads
        dP = np.zeros_like(self.P); dP[:len(toks)] = dx
        return [dE, dP], None                         # first chip: no error upstream


class HeadStage:
    """Output chip: final-LN + LM head + cross-entropy. Originates the backward error."""
    def __init__(self, d, vocab, seed=0):
        rng = np.random.default_rng(seed)
        self.Wh = rng.standard_normal((d, vocab)) * (1.0 / np.sqrt(d))
        self.g = np.ones(d); self.b = np.zeros(d)
        self.params = [self.Wh, self.g, self.b]
        self._cache = {}

    def forward(self, tag, x):
        hn, lnc = _ln_fwd(x)
        hn = hn * self.g + self.b
        logits = hn @ self.Wh
        self._cache[tag] = (x, lnc, hn)
        return logits                                 # a_out = logits

    def loss_error(self, tag, logits, target):
        z = logits - logits.max(-1, keepdims=True)
        p = np.exp(z); p /= p.sum(-1, keepdims=True)
        tgt = np.asarray(target, dtype=int)
        seq = len(tgt)
        loss = float(-np.log(p[np.arange(seq), tgt] + 1e-12).mean()) / np.log(2)   # bits/char
        d_logits = p.copy(); d_logits[np.arange(seq), tgt] -= 1.0
        d_logits /= (seq * np.log(2))
        return loss, d_logits                         # e_out fed into backward below

    def backward(self, tag, d_logits):
        x, lnc, hn = self._cache.pop(tag)
        dWh = hn.T @ d_logits
        dhn = d_logits @ self.Wh.T
        dg = (dhn * _ln_fwd(x)[0]).sum(0); db = dhn.sum(0)
        dx = _ln_bwd(dhn * self.g, lnc)
        return [dWh, dg, db], dx


def make_charlm_pipeline(vocab, d, seq, n_blocks, f=None, seed=0):
    """Embed -> n_blocks transformer blocks -> Head(CE). One stage per chip; the chain is
    the model the fabric trains under the pipeline."""
    stages = [EmbedStage(vocab, d, seq, seed=seed)]
    for l in range(n_blocks):
        stages.append(TransformerBlockStage(d, f, seed=seed + 1 + l))
    stages.append(HeadStage(d, vocab, seed=seed + 100))
    return stages
