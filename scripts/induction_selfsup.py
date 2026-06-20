"""induction_selfsup.py — PATH B: teacher-free, multiplier-free attention-router
training via COUNTERFACTUAL REINFORCE (design wf_34c97c18).

The wall: the LM-loss gradient through the hard gather is degenerate; distillation
fixes it but needs a softmax TEACHER (GPU, multiplies). Path B uses NO teacher.

The rule (non-degenerate per-key signal, teacher-free, mult-free):
  forward (deploy) : s = -Σ|Q-K| (l1), j* = argmax_j s, out = V[j*]  (compare+gather)
  signal           : per masked position i, baseline B_i = per-position LM loss at the
                     argmax route (tropical_cross_entropy, LUT, mult-free). For each
                     competing key j in top-k(s_i): COUNTERFACTUALLY substitute V[j] at
                     that layer's output, recompute L_i^(j); advantage A_ij = B_i - L_i^(j)
                     (>0 => routing to j would have LOWERED loss). The LM loss is used as a
                     scalar REWARD, never backpropped through the gather -> sidesteps the
                     degeneracy; no teacher.
  score gradient   : grad_s[i,j] = -sign(A_ij_batchavg) * sign(q(V_j)-q(V_j*))  (top-k),
                     i.e. push s up toward higher-advantage keys. sign-product = mult-free.
  optimizer        : dyadic-magnitude (step ∝ dyadic_quantize(grad), a shift), exp_clamp.
  V/O/head         : learn from the LM loss via the gather's mult-free scatter (unchanged).
  variance         : (Polyak dyadic baseline optional), batch-AVERAGE A_ij before sign.

Everything is add/compare/shift/gather/scatter -> train == deploy, on fabric, no GPU.

Arms (induction variable-offset copy; chance 0.031; deploy eval = hard gather):
  lm_only       hard gather + LM loss only, dyadic-mag         FLOOR (degenerate, ~0.06)
  reinforce_cf  counterfactual-REINFORCE score grad, dyadic-mag  THE TEST (teacher-free)
  reinforce_sign  same signal, pure SignSGD                     ablation (optimizer)
Compare to the measured teacher-based anchor distill_dyadicmag ~0.55 (induction_distill.py).

Usage: PYTHONPATH=. python scripts/induction_selfsup.py
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tropical_induction_v2 import block_copy_batch
from tikitorch.nn import dyadic_quantize


def _causal_neg(s):
    L = s.shape[-1]
    return s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=s.device), 1),
                         float("-inf"))


_SHIFT_CONTRAST = True       # backward keeps the weighted -<g,out> partition contrast (control: False)
_SHIFT_WBITS = 0             # weight mantissa bits: 0 = pure pow2 (coarse); >0 = K-bit dyadic (finer, mult-free)
_MBR_K = 4                   # margin-banded rank blend: top-k support (k shifts+adds per query)
_RTK_K = 2                   # residual top-k blend: top-k support (k=2 = convex regime)


class ShiftSoftmaxBlend(torch.autograd.Function):
    """ATTEMPT 7 (workflow winner): MULTIPLIER-FREE DIFFERENTIABLE attention blend.
    Forward (deploy can still hard-gather; this is the TRAIN-TIME relaxation):
      sp = s - max_j s        (max-subtract, mult-free)
      e  = round(sp/Delta) clamped to [-W,0]   (integer exponent; STE round)
      w  = 2^e                (dyadic weight = right-SHIFT)
      out = (sum_j w_j V_j) / pow2_ceil(sum_j w_j)   (shift-weighted sum / pow2 = SHIFT; NO exp, NO divide)
    Backward (explicit dyadic-operand STE -- the load-bearing WEIGHTED contrast):
      adv_ij = <q(g),q(V_j)> - <q(g),q(out_i)>     (the -<g,out> is softmax's partition contrast, WEIGHTED
                                                    -- heterogeneous cross-key coupling, unlike mean-subtract)
      ds_ij  = (w_ij / Zhat) * adv_ij              (dyadic weight x advantage = shifts+adds)
    All products use +-2^k operands -> shifts (project convention). Delta annealed high->low so the blend
    collapses toward one-hot => hard argmax-gather deploy is lossless (the l1_gather tau-anneal mechanism)."""

    @staticmethod
    def forward(ctx, s, v, Delta, W):
        finite = torch.isfinite(s)
        sp = s - s.amax(-1, keepdim=True)                   # <=0; max-subtract
        if _SHIFT_WBITS > 0:                                # K-bit-mantissa dyadic weight (finer, mult-free)
            we = (sp / Delta).clamp(-float(W), 0.0)
            wf = torch.exp2(we).clamp_min(1e-30)
            e2 = torch.floor(torch.log2(wf)); m = wf / torch.exp2(e2)
            step = 2.0 ** _SHIFT_WBITS
            w = (torch.round(m * step) / step) * torch.exp2(e2)   # m*2^e, m a K-bit value -> sum of shifts
            w = torch.where(finite, w, torch.zeros_like(w))
        else:                                              # pure power-of-2 weight (coarse)
            e = (sp / Delta).round().clamp(-float(W), 0.0)
            w = torch.where(finite, torch.exp2(e), torch.zeros_like(e))
        num = torch.einsum("bhij,bhjd->bhid", w, v)
        Z = w.sum(-1, keepdim=True).clamp_min(1e-12)
        Zhat = torch.exp2(torch.ceil(torch.log2(Z)))        # pow2-ceil -> 1/Zhat is an exact shift
        out = num / Zhat
        ctx.save_for_backward(w, v, out, Zhat, finite)
        return out

    @staticmethod
    def backward(ctx, g_up):
        w, v, out, Zhat, finite = ctx.saved_tensors
        gq, vq, oq = dyadic_quantize(g_up), dyadic_quantize(v), dyadic_quantize(out)
        gV = torch.einsum("bhid,bhjd->bhij", gq, vq)        # <g,V_j>  (B,H,L,L)
        adv = gV
        if _SHIFT_CONTRAST:
            gout = torch.einsum("bhid,bhid->bhi", gq, oq).unsqueeze(-1)   # <g,out_i>
            adv = gV - gout                                 # WEIGHTED partition contrast
        wn = w / Zhat
        ds = torch.where(finite, wn * adv, torch.zeros_like(adv))
        dv = torch.einsum("bhij,bhid->bhjd", wn, g_up)
        return ds, dv, None, None


class MarginBandedRankBlend(torch.autograd.Function):
    """IDEA 5 WILDCARD — a mult-free SMOOTH aggregation that SIDESTEPS the fine-w*V multiply.

    The core tension: out_d = Σ_j w_ij V_jd with a fine continuous w_ij is a MULTIPLY, so w must
    be a shift (dyadic). ShiftSoftmaxBlend made w = 2^round((s-max)/Δ): score-MAGNITUDE drives the
    weight VALUE, which (a) is coarse and (b) couples smoothness to peakedness (finer = flatter =
    lossy gather). This Function DECOUPLES the two:

      * The blend WEIGHTS are FIXED dyadic RANK weights w_r = 2^(-r) over the top-k keys (r=rank,
        0=winner). They are constants {1, 1/2, 1/4, ...} -> a right-SHIFT by r. NEVER a multiply,
        NEVER a function of the (continuous) score magnitude, so they introduce no fine product.
      * WHICH keys participate is controlled by a MARGIN BAND: rank r is INCLUDED iff its score gap
        to the winner  m_r = s_(0) - s_(r)  is within the band  Δ  (a COMPARE). gate_r = 1[m_r <= Δ].
      * out = (Σ_r gate_r · 2^-r · V_(r)) / pow2_ceil(Σ_r gate_r · 2^-r).  Numerator = k shifts+adds;
        denominator divide = a shift (pow2-ceil). ZERO multiply (top-k=k adds, vs the dense Σ_j w V).

    WHY THIS RESOLVES THE TENSION (not "just finer weights"):
      - SMOOTH for the LM: the gradient does NOT come from a finer weight VALUE; it comes from the
        soft GATE. gate_r = sigmoid_STE of (Δ - m_r): forward is the hard {0,1} compare, backward is
        a smooth band derivative. Moving a key's score across the band edge smoothly grows/shrinks
        its (fixed 2^-r) contribution -> the LM controls HOW MANY keys blend and HOW PEAKED, by
        moving scores RELATIVE TO THE BAND, without ever needing a fine weight. Smoothness lives in
        the BAND, not the weight.
      - PEAKED for deploy: as Δ -> 0 (annealed) the band closes, every runner-up's gate -> 0, the
        blend collapses to rank-0 only = EXACT top-1 argmax-gather. Hardening is LOSSLESS BY
        CONSTRUCTION: at Δ_floor=0 the train op IS the deploy op (out = V_(0)). No "flatter blend
        => lossy" failure mode, because peakedness is the band width, set to 0 at deploy, NOT a
        weight-mantissa knob fighting smoothness.
      - The two pulls are now ORTHOGONAL: weight FINENESS (fixed at the dyadic rank ladder, never
        touched) vs PEAKEDNESS (the band Δ, annealed to 0). ShiftSoftmaxBlend tied them to one knob.

    Backward (mult-free, dyadic-operand like ShiftSoftmaxBlend):
      - to V: scatter g_up by the (fixed) normalized rank weights wn_r (shifts) -> dv.
      - to s: the gate is the ONLY score-dependent term. d gate_r / d m_r is a band bump; m_r =
        s_(0)-s_(r) so it pushes s_(r) toward/away from the winner. The contribution magnitude is
        the dyadic-operand advantage adv_r = <q(g), q(V_(r))> - <q(g), q(out)> (= ShiftSoftmaxBlend's
        weighted partition contrast, the load-bearing cross-key coupling), scaled by the FIXED wn_r.
        All products are ±2^k operands -> shifts. The band bump is a constant (1/bandwidth, a shift).
    """

    @staticmethod
    def forward(ctx, s, v, Delta, K):
        B, H, L, Lk = s.shape
        kk = min(K, Lk)
        finite = torch.isfinite(s)
        # rank by score (desc). topv = sorted scores, topi = their key indices.
        topv, topi = s.topk(kk, dim=-1)                       # (B,H,L,kk)
        win = topv[..., :1]                                   # s_(0), the winner score
        margin = win - topv                                   # m_r = s_(0)-s_(r) >= 0  (COMPARE input)
        valid = torch.isfinite(topv)                          # masked (causal) keys are -inf
        # FIXED dyadic rank weights 2^-r (a right-shift by r); NEVER score-magnitude.
        ranks = torch.arange(kk, device=s.device, dtype=s.dtype)
        wr = torch.exp2(-ranks)                               # (kk,) = {1, 1/2, 1/4, ...}
        wr = wr.view(1, 1, 1, kk).expand(B, H, L, kk)
        # MARGIN-BAND gate: include rank r iff m_r <= Delta. Forward hard {0,1}; backward smooth band.
        # band derivative width = Delta (so the soft edge scales with the annealed band).
        bw = max(Delta, 1e-6)
        soft_gate = torch.sigmoid((Delta - margin) / bw)      # smooth (backward only)
        hard_gate = (margin <= Delta).to(s.dtype)             # compare (forward)
        gate = hard_gate + (soft_gate - soft_gate.detach())   # STE: hard forward, soft grad
        gate = gate * valid                                   # never gate in an -inf (masked) key
        gate = torch.where(torch.isnan(gate), torch.zeros_like(gate), gate)
        w = wr * gate                                         # (B,H,L,kk) shift * {0,1-ish}
        # gather the top-k value vectors, blend, normalize by pow2-ceil (a shift).
        dh = v.shape[-1]
        v_exp = v.unsqueeze(2).expand(B, H, L, Lk, dh)        # (B,H,Lq,Lk,dh) value per query
        idx = topi.unsqueeze(-1).expand(B, H, L, kk, dh)      # (B,H,L,kk,dh)
        vtop = torch.gather(v_exp, 3, idx)                    # (B,H,L,kk,dh) top-k value vectors
        num = torch.einsum("bhik,bhikd->bhid", w, vtop)       # Σ_r w_r V_(r) -- k shifts+adds (k small)
        Z = w.sum(-1, keepdim=True).clamp_min(1e-12)
        Zhat = torch.exp2(torch.ceil(torch.log2(Z)))          # pow2-ceil -> 1/Zhat is an exact shift
        out = num / Zhat
        wn = (w / Zhat)                                       # normalized fixed-rank weights (shifts)
        ctx.save_for_backward(wn, vtop, out, topi, valid, soft_gate, wr)
        ctx.bw = bw; ctx.dh = v.shape[-1]; ctx.Lk = Lk
        return out

    @staticmethod
    def backward(ctx, g_up):
        wn, vtop, out, topi, valid, soft_gate, wr = ctx.saved_tensors
        B, H, L, kk, dh = vtop.shape
        gq = dyadic_quantize(g_up)                            # (B,H,L,dh) ±2^k
        vq = dyadic_quantize(vtop)                            # (B,H,L,kk,dh) ±2^k
        oq = dyadic_quantize(out)                             # (B,H,L,dh) ±2^k
        gV = torch.einsum("bhid,bhikd->bhik", gq, vq)         # <g,V_(r)>  (dyadic-operand dot = shifts)
        gout = torch.einsum("bhid,bhid->bhi", gq, oq).unsqueeze(-1)
        adv = gV - gout                                       # weighted partition contrast (cross-key)
        # TWO score-gradient channels (both mult-free; the SUM is the load-bearing signal):
        #  (1) DENSE in-blend value-alignment (the TRUE softmax score-gradient form, matching
        #      ShiftSoftmaxBlend): dL/ds_r = wn_r * (<g,V_r> - <g,out>). A key whose value moves the
        #      output DOWNHILL (advantage <0 vs the uphill g=dL/dout) gets dL/ds<0 -> descent RAISES
        #      its score. Non-sparse: flows to every key in the blend, regardless of band position.
        ds_align = (wn * adv) * valid                         # (B,H,L,kk)  TRUE gradient sign (no extra negation)
        #  (2) BAND-EDGE participation control: gate_r = sigmoid((Δ-margin_r)/bw), gate'= soft_gate*
        #      (1-soft_gate)/bw. Chain: dL/ds_r = (wn_r·adv_r)·gate_d·(d gate_r/d s_r). margin_r=s_0-s_r
        #      => d gate_r/d s_r = +gate_d (r>=1); the WINNER couples to ALL ranks (d gate_r/d s_0 =
        #      -gate_d) -> winner += -Σ_r. Lets the LM widen/narrow the blend by moving scores vs the band.
        gate_d = (soft_gate * (1.0 - soft_gate)) / ctx.bw
        band = wn * adv * gate_d * valid                      # (B,H,L,kk)  per-rank band sensitivity
        ds_band = band.clone()                                # r>=1: +band ; r=0 handled below
        ds_rank = ds_align + ds_band                          # (B,H,L,kk) grad wrt sorted score s_(r)
        ds_rank[..., 0:1] = ds_rank[..., 0:1] - band.sum(-1, keepdim=True)  # winner couples to all ranks
        # scatter the per-rank score grad back to the (B,H,L,Lk) score tensor by key index.
        ds = torch.zeros(B, H, L, ctx.Lk, device=g_up.device, dtype=g_up.dtype)
        ds.scatter_(-1, topi, ds_rank)
        # grad to V: scatter g_up by the fixed normalized rank weight (shifts).
        dv_top = torch.einsum("bhik,bhid->bhikd", wn, g_up)  # (B,H,L,kk,dh)
        return ds, _scatter_dv(topi, dv_top, ctx.Lk, dh), None, None


def _scatter_dv(topi, dv_top, Lk, dh):
    """Scatter per-rank value grads (B,H,L,kk,dh) back to (B,H,L_key,dh) value grad.
    Each query's top-k keys are distinct indices into the L_key axis; accumulate."""
    B, H, L, kk = topi.shape
    dv = torch.zeros(B, H, Lk, dh, device=dv_top.device, dtype=dv_top.dtype)
    idx = topi.reshape(B, H, L * kk)                          # key indices (B,H,L*kk)
    src = dv_top.reshape(B, H, L * kk, dh)
    dv.scatter_add_(-2, idx.unsqueeze(-1).expand(B, H, L * kk, dh), src)
    return dv


class ResidualTopKBlend(torch.autograd.Function):
    """RUNNER-UP (idea 2, RTK): residual top-k blend, finer top weight than MBR, no pow2-ceil divide.
    out = V_(0) + Σ_{r>=1} 2^-c_r · (V_(r) - V_(0)),  c_r = round((s_(0) - s_(r))/Δ) clamped [0,W].
    Each term = vector-sub -> right-SHIFT by c_r -> add (mult-free, self-normalizing). k=2 = the convex
    regime (top weight 1-2^-c_1 ∈ {0,.5,.75,.875,...} -- finer than MBR's fixed 0.5). Δ annealed high->low:
    high Δ => small c => runners weighted ~1 => smooth blend (dense grad); Δ->0 => c->inf => out=V_(0) =
    EXACT argmax-gather (lossless by construction)."""

    @staticmethod
    def forward(ctx, s, v, Delta, k, W):
        B, H, L, Lk = s.shape; dh = v.shape[-1]
        kk = min(k, Lk)
        topv, topi = s.topk(kk, dim=-1)                      # (B,H,L,kk) compare only
        idx = topi.unsqueeze(-1).expand(B, H, L, kk, dh)
        Vk = torch.gather(v.unsqueeze(2).expand(B, H, L, Lk, dh), 3, idx)  # (B,H,L,kk,dh)
        Vtop = Vk[..., 0, :]                                 # (B,H,L,dh) = argmax-gather anchor
        gap = topv[..., :1] - topv                           # (B,H,L,kk) >=0; gap[...,0]=0
        finite = torch.isfinite(topv)
        c = (gap / Delta).round().clamp(0.0, float(W))
        w = torch.where(finite, torch.exp2(-c), torch.zeros_like(c))   # (B,H,L,kk) dyadic; w[...,0]=1
        out = Vtop.clone()
        for r in range(1, kk):                               # residual corrections: sub -> shift -> add
            out = out + w[..., r:r + 1] * (Vk[..., r, :] - Vtop)
        ctx.save_for_backward(w, Vk, Vtop, out, topi, finite)
        ctx.Lk = Lk
        return out

    @staticmethod
    def backward(ctx, g_up):
        w, Vk, Vtop, out, topi, finite = ctx.saved_tensors
        B, H, L, kk, dh = Vk.shape
        gq, Vq, oq = dyadic_quantize(g_up), dyadic_quantize(Vk), dyadic_quantize(out)
        gVk = torch.einsum("bhid,bhikd->bhik", gq, Vq)       # <g,V_(r)>
        gout = torch.einsum("bhid,bhid->bhi", gq, oq).unsqueeze(-1)
        adv = (gVk - gout) * finite.to(gVk.dtype)            # weighted partition contrast
        ds_rank = w * adv                                    # per-rank score grad (dyadic-operand)
        ds = torch.zeros(B, H, L, ctx.Lk, device=g_up.device, dtype=ds_rank.dtype)
        ds.scatter_(-1, topi, ds_rank)
        coeff = w.clone()                                    # value-grad coeff: runner r -> w_r ...
        if kk > 1:
            coeff[..., 0] = 1.0 - w[..., 1:].sum(-1)         # ... anchor -> 1 - Σ_{r>=1} w_r
        coeff = coeff * finite.to(coeff.dtype)
        dv_top = torch.einsum("bhik,bhid->bhikd", coeff, g_up)
        return ds, _scatter_dv(topi, dv_top, ctx.Lk, dh), None, None, None


class Block(nn.Module):
    def __init__(self, d, heads, agg="gather", skip=False, k=4):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.agg, self.skip, self.topk = agg, skip, k
        self._g_last = None                # gathered attention output (B,H,L,dh), grad exposed for VALIGN
        self.Delta = 1.0                   # shift-softmax temperature (annealed for agg="shift")
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attn_out(self, x, subst=None):
        """Attention sublayer output. agg selects the routing op:
          gather  : out=V[argmax_j s]   (score DECOUPLED from output -> no score grad; the floor)
          matmul  : out_d=max_j(s+V)     (PC-style: score ADDITIVE in the max -> winning key's
                    score gets a mult-free scatter gradient from the LM loss; no teacher)
          itertopk: top-k version of matmul (denser, iterative-argmax)
        subst (counterfactual value) overrides for the reinforce path (gather only)."""
        from tikitorch.functional import tropical_value_pull, tropical_value_pull_iter_topk
        B, L, D = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(h).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(h).view(B, L, self.h, self.dh).transpose(1, 2)
        s = _causal_neg(-(q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(-1))   # (B,H,L,L) l1
        if subst is not None:                              # counterfactual (gather path)
            g = subst
        elif self.agg == "matmul":
            g = tropical_value_pull(v, s)                  # max_j(s+V) -- score additive, mult-free bwd
        elif self.agg == "itertopk":
            g = tropical_value_pull_iter_topk(v, s, self.topk)
        elif self.agg == "shift":                          # mult-free DIFFERENTIABLE blend (train relaxation)
            g = ShiftSoftmaxBlend.apply(s, v, self.Delta, 12)
        elif self.agg == "mbr":                             # IDEA 5: margin-banded rank blend (mult-free)
            g = MarginBandedRankBlend.apply(s, v, self.Delta, _MBR_K)
        else:                                              # gather (deploy/floor)
            jstar = s.argmax(-1)
            g = torch.gather(v, -2, jstar.unsqueeze(-1).expand(*jstar.shape, self.dh))
        if self.training and g.requires_grad:              # expose gathered-output grad (g_i) for VALIGN
            g.retain_grad(); self._g_last = g
        a = self.o(g.transpose(1, 2).reshape(B, L, D))
        if self.skip:                                      # PC tropical residual highway (unit gain)
            a = torch.maximum(a, x)
        return a, s, v

    def forward(self, x, subst=None):
        a, s, v = self.attn_out(x, subst)
        x = x + a
        return x + self.mlp(self.ln2(x)), s, v


class GPT(nn.Module):
    def __init__(self, vocab, d, heads, layers, L, agg="gather", skip=False, k=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(L, d)
        self.blocks = nn.ModuleList([Block(d, heads, agg, skip, k) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)

    def set_agg(self, agg):                       # switch routing op (e.g. for eval)
        for b in self.blocks:
            b.agg = agg

    def embed(self, x):
        return self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]

    def forward(self, x, record=False, subst_layer=None, subst_val=None):
        """subst_layer/subst_val: at that layer, use subst_val as the gathered value
        (counterfactual). record=True stashes the LIVE per-layer (s, v) on self._s/_v
        for the reinforce signal; record=False leaves them untouched (so counterfactual
        re-forwards don't clobber the cached training scores)."""
        h = self.embed(x)
        if record:
            self._s, self._v = [], []
        for li, blk in enumerate(self.blocks):
            sub = subst_val if (subst_layer == li) else None
            h, s, v = blk(h, subst=sub)
            if record:
                self._s.append(s); self._v.append(v)
        logits = self.head(self.lnf(h))
        if record and logits.requires_grad:           # expose logits grad for the grad-token warmup
            self._logits = logits; logits.retain_grad()
        return logits


def gather_rank(s, v, r):
    """Gathered value using the rank-r key (0=argmax) per query, from cached s/v -> (B,H,L,dh)."""
    idx = s.topk(min(r + 1, s.shape[-1]), dim=-1).indices[..., r:r + 1]
    return torch.gather(v, -2, idx.expand(*idx.shape[:-1], v.shape[-1]))


def per_pos_ce(logits, y):
    """Per-position cross-entropy (no reduction) -> (B,L). Mult-free-equivalent reward
    value (we only use its VALUE, not its gradient through the gather)."""
    B, L, Vc = logits.shape
    return F.cross_entropy(logits.reshape(-1, Vc), y.reshape(-1), reduction="none").view(B, L)


def reinforce_score_grads(model, x, y, mask, S, V, B_i, k, layers):
    """Counterfactual-REINFORCE per-layer score gradients (teacher-free, mult-free).
    S/V = cached (detached) training scores/values; B_i = baseline per-pos loss (argmax
    route). For each top-k key r: advantage A = B_i - L_i^(r) (counterfactual loss diff,
    value-only reward); score grad at that key = -sign(A)*sign(q(V_r)-q(V*)) (sign-product
    scatter). Counterfactual re-forwards use record=False so they don't clobber S/V.
    Returned grads are backpropped into the LIVE scores -> Q/K; optimizer DESCENDS them
    (=> pushes scores UP toward higher-advantage keys)."""
    with torch.no_grad():
        out = [torch.zeros_like(S[li]) for li in range(layers)]
        for li in range(layers):
            s, v = S[li], V[li]
            jstar = s.argmax(-1)
            vstar = torch.gather(v, -2, jstar.unsqueeze(-1).expand(*jstar.shape, v.shape[-1]))
            kk = min(k, s.shape[-1])
            topi = s.topk(kk, dim=-1).indices                  # (B,H,L,kk) keys, desc score
            for r in range(kk):
                gval = gather_rank(s, v, r)                    # (B,H,L,dh) counterfactual values
                L_j = per_pos_ce(model(x, subst_layer=li, subst_val=gval), y)  # (B,L)
                A = (B_i - L_j).unsqueeze(1)                    # (B,1,L) advantage (>0 => key r better)
                key_r = topi[..., r]                           # (B,H,L)
                vr = torch.gather(v, -2, key_r.unsqueeze(-1).expand(*key_r.shape, v.shape[-1]))
                vsign = torch.sign((dyadic_quantize(vr) - dyadic_quantize(vstar)).sum(-1))  # (B,H,L)
                contrib = torch.sign(A) * vsign                # sign-product (mult-free)
                contrib = torch.where(mask.unsqueeze(1), contrib, torch.zeros_like(contrib))
                out[li].scatter_(-1, key_r.unsqueeze(-1), (-contrib).unsqueeze(-1))
        return out


def succ_target(x):
    """ATTEMPT 4 helper (teacher-free, mult-free): per query i, the target KEY is the unique
    causal j (1<=j<=i) with x[j-1]==x[i]. In the block-copy generator that key sits at
    match_pos+1 (the SUCCESSOR of the same-token match), and its value V[j] encodes y[i]=block[t+1]
    -- the answer-carrying key (verified vs block_copy_batch). Pure equality-compare + index ops."""
    B, L = x.shape
    ar = torch.arange(L, device=x.device)
    xc = x.unsqueeze(2)                                  # (B,Lq,1) current token at query i
    xprevk = torch.roll(x, 1, dims=1).unsqueeze(1)       # (B,1,Lk) predecessor x[j-1] of key j
    eq = (xc == xprevk)                                  # (B,Lq,Lk) equality compare
    jge1 = (ar >= 1)[None, None, :]                      # key has a predecessor
    causal = (ar[None, :, None] >= ar[None, None, :])    # j<=i
    ok = eq & jge1 & causal                              # candidate successor keys (unique here)
    has = ok.any(-1)
    tgt = torch.where(has, ok.float().argmax(-1),
                      torch.full((B, L), -1, dtype=torch.long, device=x.device))
    return tgt, has                                      # (B,L) key idx or -1 ; (B,L) bool


def _hinge_grad(s, jpos, valid_q, heads, margin):
    """Triplet-hinge grad-on-S for ONE layer given a positive target key per query.
    s:(B,H,L,L)=-l1 (detached); jpos:(B,L) target key idx; valid_q:(B,L) which queries get a signal.
    NEGATIVE = the model's current hardest causal distractor != target. Subgradient of the {-1,0,+1}
    coefficients through s=-(q-k).abs().sum is sign(Q-K) -- dense, contrastive, mult-free."""
    B, H, L, _ = s.shape
    ar = torch.arange(L, device=s.device)
    causal_j = (ar[None, :, None] >= ar[None, None, :])          # (1,Lq,Lk) i>=j
    NEG_INF = torch.finfo(s.dtype).min
    jp = jpos.clamp_min(0)[:, None, :].expand(B, H, L)           # (B,H,L)
    not_tgt = torch.ones(B, H, L, L, dtype=torch.bool, device=s.device)
    not_tgt.scatter_(-1, jp.unsqueeze(-1), False)
    legal = causal_j.unsqueeze(1) & not_tgt                      # causal keys != target
    jneg = s.masked_fill(~legal, NEG_INF).argmax(-1)            # (B,H,L) hardest wrong winner
    dpos = -torch.gather(s, -1, jp.unsqueeze(-1)).squeeze(-1)    # L1 dist to positive
    dneg = -torch.gather(s, -1, jneg.unsqueeze(-1)).squeeze(-1)  # L1 dist to negative
    active = ((margin + dpos - dneg) > 0) & valid_q[:, None, :]  # triplet hinge (compare)
    a = active.to(s.dtype)
    out = torch.zeros_like(s)
    # optimizer DESCENDS: -1 at pos => raise s_pos => Q,K[pos] L1-near; +1 at neg => push far.
    out.scatter_(-1, jp.unsqueeze(-1), (-a).unsqueeze(-1))
    out.scatter_add_(-1, jneg.unsqueeze(-1), (a).unsqueeze(-1))
    return out


def ff_succ_score_grads(x, mask, S, layers, heads, margin=8.0, shuffle=False):
    """ATTEMPT 4 (workflow winner, induction-circuit form): L1-MARGIN FORWARD-FORWARD router.
    Trains Q/K on a SEPARATE, non-degenerate L1-distance path (never through the argmax), with a
    TEACHER-FREE per-layer structural target -- the canonical 2-head induction circuit:
      * EARLY layers  -> PREVIOUS-TOKEN head: query i attends to key i-1 (positional, makes key j
                         encode seq[j-1] so the successor-match below becomes learnable).
      * LAST layer    -> SUCCESSOR head: query i attends to the unique causal j with x[j-1]==x[i]
                         (=match+1), whose value carries the answer y[i].
    Each is a triplet hinge whose subgradient is sign(Q-K): dense, contrastive over competitors,
    mult-free, no exp/teacher/counterfactual. V/O/head learn from the LM loss via the gather scatter.
    shuffle=True scrambles BOTH targets to random causal keys (decisive control: must collapse to
    floor -> proves CONTENT routing, not load-balancing). NB induction is a 2-layer circuit; a
    1-layer model cannot encode predecessors so layers>=2 is required for this to train."""
    with torch.no_grad():
        B, L = x.shape
        ar = torch.arange(L, device=x.device)
        causal_j = (ar[None, :, None] >= ar[None, None, :])
        tgt_succ, has = succ_target(x)                          # last-layer target (content)
        jprev = (ar - 1).clamp_min(0)[None, :].expand(B, L)     # early-layer target (i-1, positional)
        valid_prev = (ar >= 1)[None, :].expand(B, L)
        if shuffle:                                             # control: random causal targets
            r1 = torch.rand(B, L, L, device=x.device).masked_fill(~causal_j, -1.0)
            r2 = torch.rand(B, L, L, device=x.device).masked_fill(~causal_j, -1.0)
            tgt_succ, has = r1.argmax(-1), mask.clone()
            jprev = r2.argmax(-1)
        out = []
        for li in range(layers):
            if li == layers - 1:                               # successor / induction head
                out.append(_hinge_grad(S[li], tgt_succ, has & mask, heads, margin))
            else:                                              # previous-token head (all positions)
                out.append(_hinge_grad(S[li], jprev, valid_prev, heads, margin))
        return out


@torch.no_grad()
def successor_selectivity(model, args, seed, device):
    """Per-layer routing diagnostics (independent of accuracy):
      succ_sel = frac of masked queries whose argmax lands on the SUCCESSOR key (match+1); chance~1/L
      prev_sel = frac of all positions (i>=1) whose argmax lands on the PREVIOUS key (i-1)
    Expected for a trained 2-layer induction circuit: layer0 high prev_sel, last layer high succ_sel."""
    model.eval(); model.set_agg("gather")
    g = torch.Generator().manual_seed(seed + 7777)
    _, x, y, m, match = block_copy_batch(args.batch, args.L, args.vocab, args.m, g, device)
    model(x, record=True)
    B, Lm1 = x.shape
    ar = torch.arange(Lm1, device=device)
    tgt = (match + 1)[:, None, :]                          # (B,1,L) successor key
    prev = (ar - 1)[None, None, :]                         # (1,1,L) previous key
    pmask = (ar >= 1)[None, None, :]
    succ, prv = [], []
    for s in model._s:
        am = s.argmax(-1)
        hs = (am == tgt) & m[:, None, :]
        succ.append(hs.sum().item() / max(1, (m[:, None, :].expand_as(hs)).sum().item()))
        hp = (am == prev) & pmask
        prv.append(hp.sum().item() / max(1, pmask.expand_as(hp).sum().item()))
    return succ, prv


def correctness_score_grads(model, x, y, mask, S, V, k, layers):
    """ATTEMPT 3 (creative): SELF-SUPERVISED routing target from counterfactual CORRECTNESS.
    For each top-k key r, counterfactually substitute its value and ask a 1-BIT question:
    does the prediction become CORRECT (argmax==y)? Push s UP toward correct keys, DOWN
    toward incorrect ones (perceptron sign-scatter). This is the WORKING distillation
    machinery (perceptron-to-target + dyadic-mag, hit 0.918) but with a TEACHER-FREE target,
    and a 1-bit reward (low variance — the fix for attempt-1's continuous-advantage noise).
    Mult-free: counterfactual = gather/compare; correctness = compare; grad = sign scatter."""
    with torch.no_grad():
        out = [torch.zeros_like(S[li]) for li in range(layers)]
        for li in range(layers):
            s, v = S[li], V[li]
            kk = min(k, s.shape[-1])
            topi = s.topk(kk, dim=-1).indices                  # (B,H,L,kk)
            for r in range(kk):
                gval = gather_rank(s, v, r)
                cf_correct = (model(x, subst_layer=li, subst_val=gval).argmax(-1) == y)  # (B,L) bool
                gate = (2.0 * cf_correct.float() - 1.0)                # (B,L) +1 correct / -1 wrong
                gate = torch.where(mask, gate, torch.zeros_like(gate))
                gate = gate.unsqueeze(1).expand(-1, s.shape[1], -1)    # (B,H,L) broadcast to heads
                key_r = topi[..., r]                                   # (B,H,L)
                out[li].scatter_(-1, key_r.unsqueeze(-1), (-gate).unsqueeze(-1))  # correct -> s UP
        return out


def valign_score_grads(x, mask, S, V, G_grads, layers, heads, margin, shuffle=False, gfloor=0.0):
    """ATTEMPT 5 (workflow winner): VALUE-ALIGNMENT TARGET — teacher-free, mult-free, CONTENT-ADAPTIVE.
    The content signal is the model's OWN LM-loss gradient at the attention output: g_i = -dL/d(out_i)
    (read from G_grads = blocks[li]._g_last.grad after lm.backward). The target key = argmax over causal
    j of <q(g_i), q(V_j)> (the key whose value points most in the loss-reducing direction = exactly the
    soft gradient's value-alignment term, since out_i is constant across j). q = dyadic_quantize -> the
    dot is a SUM OF SHIFTS (mult-free, the project's quantized-operand convention). The TARGET is then fed
    to the proven _hinge_grad (which supplies the per-query CONTRAST every prior surrogate dropped). NO
    structural target, NO teacher, NO exp: the routing is DISCOVERED from the LM gradient. shuffle: random
    causal target (content control, must collapse). gfloor: skip queries with tiny ||g_i|| (route solved)."""
    with torch.no_grad():
        B, L = x.shape
        ar = torch.arange(L, device=x.device)
        causal = (ar[None, :, None] >= ar[None, None, :])             # (1,Lq,Lk)
        NEG = torch.finfo(S[0].dtype).min
        out, jtgts = [], []
        for li in range(layers):
            gi = dyadic_quantize(-G_grads[li])                        # (B,H,L,dh) loss-decrease dir, ±2^k
            vq = dyadic_quantize(V[li])                               # (B,H,L,dh) ±2^k value
            align = torch.einsum("bhid,bhjd->bhij", gi, vq)          # dyadic-operand dot = sum of shifts
            align = align.masked_fill(~causal.unsqueeze(1), NEG)
            if shuffle:
                j_tgt = torch.rand_like(align).masked_fill(~causal.unsqueeze(1), -1.0).argmax(-1)
            else:
                j_tgt = align.argmax(-1)                              # (B,H,L) value-alignment target
            vq_h = (G_grads[li].abs().sum(-1) > gfloor)               # (B,H,L) drop near-zero-grad queries
            g_layer = torch.zeros_like(S[li])
            for h in range(heads):
                g_layer[:, h:h + 1] = _hinge_grad(S[li][:, h:h + 1], j_tgt[:, h], mask & vq_h[:, h], 1, margin)
            out.append(g_layer); jtgts.append(j_tgt)
        return out, jtgts


def gradtoken_target(model, x, mask, S, layers, heads, margin):
    """NON-CIRCULAR bootstrap (runner-up): target key = argmax over causal j of (-dL/dlogits_i)[token x_j]
    -- route to the key whose TOKEN the LM loss most wants emitted at i. Uses ONLY the logit gradient (no
    V/O dependence -> informative at step 0, fixes VALIGN cold-start). Mult-free: CE backward (floor's
    cost) + gather-by-token + argmax. NB on the induction probe this == the structural x[j]==y target, so
    it is a probe FALSE-POSITIVE alone -> used only as VALIGN's warmup, never as the headline."""
    with torch.no_grad():
        gl = -model._logits.grad                                     # (B,L,V) >0 = token the loss wants
        B, L = x.shape
        ar = torch.arange(L, device=x.device)
        causal = (ar[None, :, None] >= ar[None, None, :])
        keytok = torch.gather(gl, 2, x[:, None, :].expand(B, L, L))  # (B,Li,Lj) = gl[b,i, x[b,j]]
        j_tgt = keytok.masked_fill(~causal, torch.finfo(gl.dtype).min).argmax(-1)   # (B,L) shared over heads
        out = []
        for li in range(layers):
            g_layer = torch.zeros_like(S[li])
            for h in range(heads):
                g_layer[:, h:h + 1] = _hinge_grad(S[li][:, h:h + 1], j_tgt, mask, 1, margin)
            out.append(g_layer)
        return out


def gradtok_circuit_grads(model, x, mask, S, layers, heads, margin, shuffle=False):
    """ATTEMPT 6 (the real test): induction CIRCUIT with a grad-token CONTENT-ADAPTIVE reader.
    EARLY layers = PREVIOUS-TOKEN feeder (universal structural primitive: makes key j encode x[j-1]).
    LAST layer = GRAD-TOKEN reader: target = argmax over causal j of (-dL/dlogits_i)[token x_j] -- route
    to the key whose TOKEN the LM loss most wants emitted (content-adaptive on natural text; uses only the
    model's own logit gradient, teacher-free; gather+argmax, mult-free). Fixes the bare 'gradtok' failure
    (which forced the content target on layer 0 too, breaking the prev-token feeder)."""
    with torch.no_grad():
        gl = -model._logits.grad                                # (B,L,V) >0 = token loss wants emitted
        B, L = x.shape
        ar = torch.arange(L, device=x.device)
        causal = (ar[None, :, None] >= ar[None, None, :])
        keytok = torch.gather(gl, 2, x[:, None, :].expand(B, L, L))   # (B,Li,Lj) = gl[b,i, x[b,j]]
        if shuffle:
            jtok = torch.rand(B, L, L, device=x.device).masked_fill(~causal, -1.0).argmax(-1)
        else:
            jtok = keytok.masked_fill(~causal, torch.finfo(gl.dtype).min).argmax(-1)   # (B,L) reader target
        jprev = (ar - 1).clamp_min(0)[None, :].expand(B, L)     # prev-token feeder target
        valid_prev = (ar >= 1)[None, :].expand(B, L)
        out = []
        for li in range(layers):
            jpos, valid = (jtok, mask) if li == layers - 1 else (jprev, valid_prev)
            g = torch.zeros_like(S[li])
            for h in range(heads):
                g[:, h:h + 1] = _hinge_grad(S[li][:, h:h + 1], jpos, valid, 1, margin)
            out.append(g)
        return out


def lr_sched(step, args):
    warm = min(1.0, (step + 1) / args.warmup)
    cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
    return args.lr * warm * cos


def delta_at(step, args):
    """Anneal the shift-softmax temperature Delta high->low (mirror tau_at): hold to 50%, decay to
    floor by 90%. High Delta = wide soft blend (dense grad to s); low Delta -> one-hot -> lossless hard."""
    if step < 0.5 * args.steps:
        return args.delta0
    if step > 0.9 * args.steps:
        return args.delta_floor
    frac = (step - 0.5 * args.steps) / (0.4 * args.steps)
    return args.delta_floor + (args.delta0 - args.delta_floor) * (1 - frac)


# arm -> (routing op, tropical residual skip, self-supervised score signal)
ARMS = {
    "lm_only":     dict(agg="gather",   skip=False, signal=None),         # score decoupled -> FLOOR
    "matmul":      dict(agg="matmul",   skip=False, signal=None),         # PC additive-max (failed)
    "reinforce_cf": dict(agg="gather",  skip=False, signal="reinforce"),  # counterfactual LOSS (failed: variance)
    "ccorrect":    dict(agg="gather",   skip=False, signal="correct"),    # ATTEMPT 3: counterfactual CORRECTNESS (1-bit, teacher-free) -> failed
    "ff_succ":     dict(agg="gather",   skip=False, signal="ff_succ"),         # ATTEMPT 4: L1-margin FF, STRUCTURAL successor target
    "ff_succ_shuffle": dict(agg="gather", skip=False, signal="ff_succ_shuffle"),  # CONTROL: random target -> must collapse to floor
    "valign":      dict(agg="gather",   skip=False, signal="valign"),          # ATTEMPT 5 (winner): value-alignment target (content-adaptive, teacher-free)
    "valign_shuffle": dict(agg="gather", skip=False, signal="valign_shuffle"), # CONTROL: random target -> must collapse
    "valign_warmup": dict(agg="gather", skip=False, signal="valign", warmup_struct=600),  # grad-token bootstrap then VALIGN
    "gradtok":     dict(agg="gather",   skip=False, signal="gradtok"),         # grad-token target on ALL layers (breaks the circuit -> chance)
    "gradtok_circuit": dict(agg="gather", skip=False, signal="gradtok_circuit"),         # ATTEMPT 6: prev-token feeder + grad-token reader
    "gradtok_circuit_shuffle": dict(agg="gather", skip=False, signal="gradtok_circuit_shuffle"),  # CONTROL
    "shift":          dict(agg="shift", skip=False, signal=None),   # ATTEMPT 7: mult-free DIFFERENTIABLE blend, trained by LM loss alone
    "shift_nocontrast": dict(agg="shift", skip=False, signal=None), # CONTROL: drop the -<g,out> contrast -> must collapse
    "shift_adamw":    dict(agg="shift", skip=False, signal=None),   # DIAGNOSTIC: shift blend + AdamW (isolate mechanism from dyadic-mag optimizer)
    "shift_nocontrast_adamw": dict(agg="shift", skip=False, signal=None),  # CONTROL + AdamW
    "shift_ef":       dict(agg="shift", skip=False, signal=None),   # ATTEMPT 8: shift blend + ERROR-FEEDBACK dyadic optimizer (fully mult-free)
    "shift_mom":      dict(agg="shift", skip=False, signal=None),   # ATTEMPT 9: shift blend + MOMENTUM dyadic optimizer (mult-free EMA)
    "shift_sgd":      dict(agg="shift", skip=False, signal=None),   # DIAGNOSTIC: shift + plain SGD+momentum on TRUE grad (is Adam's adaptivity essential, or was dyadic-quant crude?)
    "shift_dadam":    dict(agg="shift", skip=False, signal=None),   # ATTEMPT 10: shift + DYADIC-ADAM (mult-free per-param adaptive scale: 1/|g|_ema as a pow2 SHIFT)
    "mbr":            dict(agg="mbr", skip=False, signal=None),     # IDEA 5: margin-banded rank blend + dyadic-mag
    "mbr_adamw":      dict(agg="mbr", skip=False, signal=None),     # IDEA 5 + AdamW (isolate mechanism)
    "mbr_dadam":      dict(agg="mbr", skip=False, signal=None),     # IDEA 5 + dyadic-Adam (fully mult-free)
}


@torch.no_grad()
def eval_acc(model, args, seed, device, agg):
    saved = [b.agg for b in model.blocks]
    model.set_agg(agg); model.eval()
    egen = torch.Generator().manual_seed(seed + 9999)
    accs = []
    for _ in range(args.eval_batches):
        _, x, y, m, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, egen, device)
        pred = model(x).argmax(-1)
        accs.append(((pred == y) & m).sum().item() / m.sum().item())
    for b, a in zip(model.blocks, saved):
        b.agg = a
    return sum(accs) / len(accs)


def train(arm, seed, args, device):
    cfg = ARMS[arm]
    global _SHIFT_CONTRAST
    _SHIFT_CONTRAST = ("nocontrast" not in arm)            # control drops the weighted contrast
    use_adamw = arm.endswith("_adamw")                     # diagnostic: isolate mechanism from dyadic-mag
    use_ef = arm.endswith("_ef")                           # ATTEMPT 8: error-feedback dyadic optimizer
    use_mom = arm.endswith("_mom")                         # ATTEMPT 9: momentum dyadic optimizer
    use_sgd = arm.endswith("_sgd")                         # DIAGNOSTIC: plain SGD+momentum, true grad
    use_dadam = arm.endswith("_dadam")                     # ATTEMPT 10: dyadic-Adam (mult-free adaptive)
    use_sched = use_adamw or use_sgd or use_dadam          # lr-scheduled optimizers
    ef_buf, mom_buf, m_buf, v_buf = {}, {}, {}, {}         # optimizer state buffers
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = GPT(args.vocab, args.d, args.heads, args.layers, args.L,
                agg=cfg["agg"], skip=cfg["skip"], k=args.k).to(device)
    if use_adamw:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    elif use_sgd:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif use_dadam:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)   # the adaptive step is built in p.grad below
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.sign_lr)
    for step in range(args.steps):
        model.train()
        if use_sched:
            for g in opt.param_groups:
                g["lr"] = lr_sched(step, args)
        if cfg["agg"] in ("shift", "mbr"):                 # anneal the band/temperature high->low
            d = delta_at(step, args)
            for b in model.blocks:
                b.Delta = d
        _, x, y, mask, match = block_copy_batch(args.batch, args.L, args.vocab, args.m, gen, device)
        opt.zero_grad()
        use_sig = cfg["signal"] is not None
        logits = model(x, record=use_sig)
        lm = F.cross_entropy(logits[mask], y[mask])    # matmul: -> Q/K via additive-max; gather: only V/O/head
        lm.backward(retain_graph=use_sig)
        if use_sig:
            S_live = list(model._s); S = [s.detach() for s in model._s]; V = [v.detach() for v in model._v]
            sig = cfg["signal"]
            if sig == "reinforce":
                with torch.no_grad():
                    B_i = per_pos_ce(logits, y)
                gss = reinforce_score_grads(model, x, y, mask, S, V, B_i, args.k, args.layers)
            elif sig == "correct":                      # teacher-free 1-bit counterfactual
                gss = correctness_score_grads(model, x, y, mask, S, V, args.k, args.layers)
            elif sig in ("ff_succ", "ff_succ_shuffle"):
                gss = ff_succ_score_grads(x, mask, S, args.layers, args.heads,
                                          margin=args.margin, shuffle=(sig == "ff_succ_shuffle"))
            elif sig == "gradtok":                      # logit-grad token-preference target (all layers)
                gss = gradtoken_target(model, x, mask, S, args.layers, args.heads, args.margin)
            elif sig in ("gradtok_circuit", "gradtok_circuit_shuffle"):
                gss = gradtok_circuit_grads(model, x, mask, S, args.layers, args.heads,
                                            margin=args.margin, shuffle=sig.endswith("_shuffle"))
            else:                                       # "valign" / "valign_shuffle"
                if step < cfg.get("warmup_struct", 0):  # non-circular grad-token bootstrap
                    gss = gradtoken_target(model, x, mask, S, args.layers, args.heads, args.margin)
                else:
                    G_grads = [b._g_last.grad for b in model.blocks]
                    gss, jt = valign_score_grads(x, mask, S, V, G_grads, args.layers, args.heads,
                                                 margin=args.margin, shuffle=(sig == "valign_shuffle"),
                                                 gfloor=args.gfloor)
                    if step % max(1, args.steps // 6) == 0:   # diagnostic: does the target FIND the successor?
                        jt_last = jt[-1]
                        hit = ((jt_last == (match + 1)[:, None, :]) & mask[:, None, :]).sum().item()
                        den = max(1, (mask[:, None, :].expand_as(jt_last)).sum().item())
                        print(f"    [{arm} s{seed}] step{step} j_tgt==succ {hit/den:.3f}", flush=True)
            torch.autograd.backward(S_live, gss)        # self-supervised signal -> Q/K via score grad
        if use_adamw or use_sgd:                       # diagnostic optimizers on TRUE grad (NOT mult-free)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        elif use_dadam:                                # DYADIC-ADAM: per-param adaptive scale, mult-free
            for p in model.parameters():               # m,v = dyadic EMAs (b=0.875=1-2^-3); step = m * pow2(1/v)
                if p.grad is not None:
                    g = p.grad
                    m = m_buf.get(p, torch.zeros_like(g)); v = v_buf.get(p, torch.zeros_like(g))
                    m = (m - m * 0.125) + g * 0.125            # 0.125 = 2^-3 -> shifts (mult-free EMA)
                    v = (v - v * 0.125) + g.abs() * 0.125      # EMA of |grad| (L1, no square)
                    m_buf[p], v_buf[p] = m, v
                    scale = torch.exp2(-torch.round(torch.log2(v.clamp_min(1e-12))))  # 1/|g|_ema as a pow2 SHIFT
                    p.grad = m * scale                        # adaptive step = m shifted (mult-free)
        elif use_ef:                                   # error-feedback dyadic (mult-free): accumulate
            for p in model.parameters():               # the rounding residual so coarse steps de-bias
                if p.grad is not None:
                    c = p.grad + ef_buf.get(p, torch.zeros_like(p.grad))
                    q = dyadic_quantize(c)
                    q = q.sign() * q.abs().clamp(max=2.0 ** -1)   # upper cap only (no floor -> fine grads accumulate)
                    ef_buf[p] = c - q                  # residual fed forward
                    p.grad = q
        elif use_mom:                                  # momentum dyadic (mult-free EMA b=0.875=1-2^-3)
            for p in model.parameters():
                if p.grad is not None:
                    m = mom_buf.get(p, torch.zeros_like(p.grad))
                    m = m - m * 0.125 + p.grad         # 0.125 = 2^-3 -> m>>3 on fabric (mult-free)
                    mom_buf[p] = m
                    g = dyadic_quantize(m)
                    p.grad = g.sign() * g.abs().clamp(2.0 ** -12, 2.0 ** -1)
        else:
            for p in model.parameters():               # dyadic-magnitude mult-free optimizer
                if p.grad is not None:
                    g = dyadic_quantize(p.grad)
                    p.grad = g.sign() * g.abs().clamp(2.0 ** -12, 2.0 ** -1)
        opt.step()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=32); p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--m", type=int, default=5); p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=64); p.add_argument("--k", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--sign-lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=int, default=300); p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--eval-batches", type=int, default=8); p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--margin", type=float, default=8.0)   # triplet hinge margin (dyadic-friendly)
    p.add_argument("--gfloor", type=float, default=0.0)    # skip VALIGN queries with ||g_i|| below this
    p.add_argument("--delta0", type=float, default=2.0)    # shift-softmax temperature: anneal start
    p.add_argument("--delta-floor", type=float, default=0.25)  # ... and floor
    p.add_argument("--mbr-k", type=int, default=4)         # margin-banded rank blend: top-k support
    p.add_argument("--arms", type=str, default="lm_only,ff_succ,ff_succ_shuffle")
    args = p.parse_args()
    global _MBR_K
    _MBR_K = args.mbr_k
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SELFSUP test (ATTEMPT 4: L1-margin FF, successor target): L={args.L} vocab={args.vocab} m={args.m} "
          f"d={args.d} k={args.k} steps={args.steps} seeds={args.seeds} dev={device} "
          f"chance={1/args.vocab:.3f}", flush=True)
    res = {}; t0 = time.time()
    for arm in args.arms.split(","):
        train_agg = ARMS[arm]["agg"]
        for seed in args.seeds:
            model = train(arm, seed, args, device)
            # eval with the training op (consistent) AND with gather (clean-copy using learned score)
            a_train = eval_acc(model, args, seed, device, train_agg)
            a_gather = eval_acc(model, args, seed, device, "gather")
            succ, prv = successor_selectivity(model, args, seed, device)
            res.setdefault(arm, []).append((a_train, a_gather))
            sstr = "/".join(f"{x:.2f}" for x in succ); pstr = "/".join(f"{x:.2f}" for x in prv)
            print(f"  {arm:16s} seed{seed} {train_agg}={a_train:.3f} gather={a_gather:.3f} "
                  f"succ_sel[{sstr}] prev_sel[{pstr}]  ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== SELFSUP RESULT (teacher-free; deploy = gather; mean +/- std) ===", flush=True)
    for arm, v in res.items():
        tr = [a for a, _ in v]; ga = [g for _, g in v]
        def ms(z): mn = sum(z) / len(z); return mn, (sum((x - mn) ** 2 for x in z) / len(z)) ** 0.5
        mt, st = ms(tr); mg, sg = ms(ga)
        print(f"  {arm:12s} train_op {mt:.3f}+/-{st:.3f}   gather {mg:.3f}+/-{sg:.3f}", flush=True)
    print("\nPASS: ff_succ gather >> lm_only floor (>=0.55 strong; [0.10,0.55] partial) AND succ_sel >> chance "
          "AND ff_succ_shuffle collapses to floor => teacher-free, mult-free CONTENT routing works.", flush=True)


if __name__ == "__main__":
    main()
