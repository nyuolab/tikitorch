# Cascade PC on sequence models: the unrolled-graph corollary

Extends `results/pc_depth_traction/THEORY_CASCADE_EQUIVALENCE.md` (conditions C1–C3)
from feedforward MLP/conv stacks to (a) the full MorphSSM LM stack and (b) recurrent
tropical networks via unrolling. Branch `pc-morphssm`.

## 1. The corollary

**Claim.** For a sequence model whose per-step computation graph, unrolled over the
T positions of a training window, is acyclic — this includes MorphSSM (causal convs
are already position-parallel) and any RNN unrolled over the window — the one-pass
**cascade** (forward, inject the CE top error at every position, sweep once in
reverse-topological order over the (layer, time) lattice) equals **iterative PC
settled to its energy minimum** on that unrolled graph, exactly.

*Why.* C1: the unrolled graph is a DAG over (layer ℓ, time t) — a recurrent state
h_t depends only on h_{t−1} and the drive u_t, so unrolling removes the cycle
(contrast: PC on the *cyclic* recurrent graph, Salvatori-style, genuinely requires
relaxation — that case is NOT covered here and we do not claim it). C2: the
all-position CE readout injects ε_T = ∂CE/∂logits per position — the chain-rule
boundary. C3 is vacuous for the cascade: on a DAG, stationarity ∂E/∂x = 0 is a
single back-substitution, and the cascade IS that back-substitution in closed form.

**Honesty note (pre-registered against overclaim).** On the unrolled graph with tied
weights, the settled-PC/cascade gradient is *identical to backprop-through-time*.
We do not claim a different descent direction, and we do not cite the depth-traction
negative rule tax (−0.2 bpc, `results/pc_depth_traction/EXP_RESULTS.md`) as PC
superiority — that effect is quantization/settling regularization and flipped sign
on conv (`results/predictive_coding/CHARLM_CONV_LOCAL_PC.md`, +0.17). The claims
this branch runs on are the four below.

## 1b. The two-mode statement (THE PROGRAM GOAL)

The point of proving cascade ≡ settled-PC is the **asymmetry it licenses**:

> **Pay one forward-pass-equivalent for training credit (the cascade), and pay
> iteration only where you want the generative behavior — the equivalence
> theorem guarantees it is the same model in both modes.**

Training by cascade forecloses nothing: the weights it produces are exactly the
weights the iterative-PC view describes, and the energy function E(states; W)
ships with them for free. At inference the SAME checkpoint runs in two modes:

- **Feedforward mode** (the default, fully closed at τ=0): the ordinary O(T)
  forward. Note the precision — clamping the *input* and iteratively settling
  converges to exactly this forward pass (E → 0 at x_l = f_l(x_{l−1})); naive
  "iterative inference" is a no-op, only slower. This is C1 read in reverse.
- **Generative/relaxation mode** (the PC-unique payoff): choose a different
  clamping pattern and descend the energy over the *states*:
  * **infilling** — clamp prefix AND suffix, relax the span between (a causal
    LM has no native bidirectional-constraint mode; PC does, on the same weights);
  * **goal-conditioned steering** — clamp desired future logits, relax latents;
  * **denoising / completion** — clamp the trusted fragment of a corrupted
    context, relax the rest (the SparseCodingPC associative-memory mode, never
    yet demonstrated on a deep tropical sequence model).

Honest boundaries of the claim (pre-registered): (i) generative mode needs
*state* relaxation (routing re-evaluated each step) — a different loop from the
frozen-routing *error* relaxation used for training credit; (ii) state settling
has measured stability failure modes in this repo (settle_nudge energy
divergence as weights grow; fixed-step descent divergence at width 768 — the
U-gate) — C3 is trivial for the cascade but returns as a real problem here;
(iii) at hard τ=0 the energy is piecewise-linear (kinks at winner changes), so
relaxation runs at soft τ>0 and is evaluated hard — a disclosed inference-time
closure boundary for this mode only; the feedforward deploy mode stays fully
closed. Gate G6 (PREREG) measures all of this.

## 2. What is actually new here

1. **Settling is deleted where it would be catastrophic.** Iterative PC on the
   unrolled lattice has diameter O(L + chain-length); at L=7, T=96 that is
   hundreds of relaxation rounds (the measured 10–20× settling wall, worsened by
   T). The cascade computes the identical fixed point in ONE sweep at O(forward)
   cost. For sequence models this is the difference between PC being unusable and
   PC being free.

2. **Zero-multiply credit assignment at τ=0.** Every op in the MorphSSM stack has a
   discrete, cacheable routing: morph-conv argmax slot, max-plus out_proj argmax,
   top-k head selection, dropout gate, residual copy, and (for the conv-RNN arm)
   the recurrence carry gate. The frozen-routing adjoint of each is a gather/
   scatter/add — **no multiplies of any kind**, stronger than the dyadic-conv PC
   backward (which still needed ±2^k shifts for `delta @ W`). The only boundary op
   is the CE softmax at the top (the same sanctioned boundary as every closed run
   in the repo). The stop-grad pow-2 LayerNorm scale is the one by-design deviation
   from autograd, exactly as in the DyadicPCNet gradchecks.

3. **Unit-gain error highways through time.** For the leaky-max recurrence
   h_t[c] = max(u_t[c], h_{t−1}[c] − λ[c]), the frozen temporal Jacobian is the
   binary carry gate g_t[c] = 1[h_{t−1}[c] − λ[c] ≥ u_t[c]]; the reverse-time
   recursion

       tot_t = δ_t + c_t;   ε^u_t = (1−g_t)⊙tot_t;   c_{t−1} = g_t⊙tot_t;
       ∇λ[c] = −Σ_t g_t[c]·tot_t[c]   (the −λ edge has ∂/∂λ = −1: a signed count)

   carries error k steps back along a carry run with gain exactly 1 — no vanishing
   or exploding through time, structurally (the temporal analogue of the ReLU
   frozen-routing / max-gate highways that give DyadicPCNet depth-100 traction).
   Caveat, pre-registered: *bounded ≠ delivered* — winner sparsity can starve long
   lags; G1 measures **delivery** (fraction of positions receiving nonzero error at
   lag k), not just magnitude.

4. **λ bounds the settling diameter, not T.** A carry run ends when a fresh input
   wins; its expected length is ≈ range(u)/λ. So async/iterative relaxation on the
   lattice needs O(L + range/λ) rounds, not O(L + T). G1 reports the chain-length
   histogram; if the histogram is T-bounded instead, the fabric story weakens (the
   cascade claim is unaffected) and we say so.

## 3. Soft-τ exactness (why the anneal costs the equivalence nothing)

For τ>0 the morphological ops compute y = τ·logsumexp(z/τ) with z_k = w_k + x_k.
The first-order Jacobian is *exactly* the softmax weight vector p_k = softmax(z/τ)_k
(no second-order correction), and since z = w + x, ∂y/∂w_k = ∂y/∂x_k = p_k.
Freezing p from the forward therefore makes each layer a fixed linear map in the
error; the PC error system stays **linear-triangular at every τ**, so:

- cascade ≡ settled relax at every τ (C1/C2 are τ-independent);
- the frozen-soft cascade gradient equals autograd of the soft forward, up to the
  stop-grad pow-2 norm scale;
- the τ-anneal (`tau_at`: hold 0.5 → linear to 0 by 90% → hard tail) needs no PC
  special-casing — the sweep freezes whatever routing the current-τ forward made
  (soft p above τ=0, argmax indices at τ=0).

Closure ladder at train time (disclosed per run): the soft-phase sweep contains
ε·p products (same sanction as the BP arms' soft forward); the τ=0 sweep is pure
gather/scatter/add. The recurrence gate (conv-RNN arm) is **hard at all τ** by
design: a soft 2-way scan overshoots the hard max by up to τ·ln2 *per step*
(compounding to ≈ T·τ·ln2 ≈ 33 at τ=0.5, T=96 — it would reintroduce the drift the
cell exists to prevent) and would create a train/deploy scan mismatch (the l1-attention
operator-mismatch trap, temporal edition).

## 4. The recurrence cell (conv-RNN arm) and why it cannot drift

The known-dead min-plus RNN (`results/predictive_coding/CHARLM_LOCAL_PC.md`: state
→ running extremum, mean −81, 16.5 bpc) is a theorem instance, not a bug: for
h_t = W_h ⊗ h_{t−1} ⊕ ..., the long-run growth rate is the tropical spectral radius
λ(W_h) = max cycle mean of W_h's weighted digraph — nothing pins it to 0, so learned
weights generically drift linearly in t. The per-channel leaky-max cell

    h_t[c] = max(u_t[c], h_{t−1}[c] − λ[c]),   λ[c] ≥ λ_min = 2⁻⁶  (λ = λ_min + |raw|)

excludes this structurally: the only cycle is the self-loop with weight −λ[c] < 0,
and the closed form h_t = max_{k≥0}(u_{t−k} − k·λ) gives the two-sided bound
u_t ≤ h_t ≤ running-max(u). With u post-TropicalLayerNorm (mean-abs ≈ 1) the state
is bounded for all T; the effective per-channel memory horizon is ≈ range(u)/λ[c],
so λ is a learned, per-channel horizon (init log-spaced, horizons ≈ 8–256 — the
S4/Mamba multi-timescale init, tropicalized). λ=0 is kept as the drift positive
control in G2. The scan is add/compare only — closed — and is a semiring-linear
recurrence (associative-scannable later; v1 uses the time loop).

Equivalently: the cell is a morphological dilation with an infinite linear-ramp
structuring element and O(1) state — an unbounded receptive field the finite-K
MorphSSM lacks, at O(d) streaming state per layer vs Σ(K−1)·dilation·d ≈ 762·d for
the L7 conv buffers. That state compression and beyond-RF behavior (G2 selective
copy, G3 streaming eval at T=512/1024) — not seq-96 bpc — is where the recurrence
is expected to pay (MorphSSM L3≈L7 saturation; L7 RF 763 ≫ 96; predicted seq-96
null is pre-registered as not-a-failure).
