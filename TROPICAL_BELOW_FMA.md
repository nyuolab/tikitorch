# Tropical, below the dense FMA: where TikiTorch sits

A positioning note. The "below-dense-FMA" neighborhood is the set of methods that
replace the dense multiply-accumulate (the fused multiply-add the whole GPU/TPU
stack is built around) with something cheaper than a multiplier. Several families
live here, and they are all good ideas. TikiTorch is one of them — but it is the
only one that ships a *mechanical check* that the multiplier is actually gone, and
the only one with that check confirmed against **zero-DSP silicon training**.

This doc places TikiTorch among its neighbors honestly: what each family does,
where it still depends on a multiply (or a learned scale, or a batch-norm, or an
FP optimizer that the marketing leaves out), and what specifically TikiTorch
brings that the others do not. The differentiator is not "no multiplies" — most
of these claim that. It is **algebraic closure that is verifiable**
(`tikitorch.audit.assert_closed` / `assert_closed_runtime`) and **end-to-end
training validated on FPGA fabric that synthesizes to zero DSP blocks.**

---

## The dense FMA and why people leave it

A standard layer computes `y = Σ_k W[o,k]·x[k] + b` — for each output, `in_features`
multiply-accumulates. On silicon those multiplies are the expensive part: a DSP
slice (the FPGA hard multiplier) or the bulk of a systolic MAC array, and on the
energy budget the multiplier dominates an add by roughly an order of magnitude in
the same precision. So a whole literature asks: *can we keep the accumulation but
kill the multiply?* The families below are the serious answers.

A subtlety that matters for the taxonomy: "multiply-free **inference**" is common
and comparatively easy; "multiply-free **training**" is rare, because the backward
pass, the normalization layers, and the optimizer all tend to smuggle a multiply
back in. The honest axis is therefore not one bit but two — *is the forward
multiply-free?* and *is the entire learning loop multiply-free?* — and a third,
sharper one: *can you prove it, mechanically, rather than assert it in the
abstract?*

---

## The families

### 1. AdderNet — ℓ1-distance convolution
Chen et al., *AdderNet: Do We Really Need Multiplications in Deep Learning?*
(CVPR 2020). Replace the dot product with the negative ℓ1 distance between the
filter and the patch:

    y[o] = − Σ_i |x_i − W[o,i]|   (+ bias)

No multiply in the forward similarity. The connection to the tropical world is
direct: `|a| = max(a, −a)` is a max-plus expression, so AdderNet is a
comparison/addition layer in disguise. TikiTorch ships exactly this as
`tikitorch.nn.AdderLinear` (`y[o] = −Σ_i |x_i − W[o,i]| + b_o`), explicitly so the
library spans the taxonomy and carries an AdderNet baseline rather than just
citing one.

The honest caveat AdderNet papers carry: the ℓ1 forward changes the gradient
magnitude scale, so the published recipe needs an **adaptive per-layer learning
rate** and **batch normalization** to train at all — and BN is a multiply
(scale·x) plus a running-variance division. So AdderNet is multiply-free in the
*convolution* but not, as published, in the *whole loop*. Nobody on the AdderNet
side claims otherwise; it just means "multiply-free" there is scoped to the
similarity op, not the optimizer or the norm.

### 2. ShiftAddNet — shift + add
You et al., *ShiftAddNet: A Hardware-Inspired Deep Network* (NeurIPS 2020).
Decompose the layer into two cheap primitives: a **bit-shift** (a coarse,
power-of-2 multiply that is free in fabric — it is just rewiring) followed by an
**add**. The shift handles the dynamic range; the add handles the fine structure.
This is the most hardware-literate framing in the neighborhood: it names the two
operations a multiplier-free datapath actually wants.

TikiTorch agrees with the premise and reuses it deliberately — the entire
optimizer story is shift+add. `PureTropicalSGD` uses a **power-of-2 learning rate
as a bit-shift** (`lr_shift`) and **integer-counter momentum**, and `TropicalScale`
is a learnable **power-of-2 per-channel α** (the closed analogue of the morph-net
scale that breaks the deep gradient collapse), precisely because a pow-2 scale is
a shift and stays closed. Where ShiftAddNet stops short for our purposes: the
shift+add backbone still trains with standard FP optimizers and norms, and "is a
given tensor op actually a shift, or did a real `×` sneak back in?" is left to the
reader. That is the gap the runtime audit closes.

### 3. Log-Number-System (LNS) nets
Represent values in the log domain, where **multiply becomes add** (`log(ab) =
log a + log b`) and the genuinely hard operation moves to *addition*, handled by a
LNS adder (a lookup/approx of `log(1 + e^Δ)`). A long DSP/hardware lineage (Arnold,
Coleman et al.; revived for DNNs as, e.g., LNS-Madam and related log-domain
accelerators). LNS is the most interesting cousin because it is, structurally, a
**deformation of the tropical semiring**: as the base → ∞ (Maslov dequantization),
`log(e^a + e^b) → max(a, b)`, and LNS addition *is* TikiTorch's soft-to-hard
`logsumexp` relaxation. TikiTorch's `tau`-anneal (train soft `log(Σ e^·/τ)`,
deploy hard `max`) is literally the LNS→tropical limit, with the same weights
driving both paths.

The honest difference: LNS keeps the **multiply** (cheaply, as an add) and pays at
the **add** (the LNS-add table / approximation, which is the non-trivial hardware).
It is multiply-cheap, not comparison-only. The tropical limit drops the add cost
entirely — the "addition" of the semiring is a bare comparison — at the price of
the expressivity you lose by hardening `logsumexp` to `max`. That price is exactly
TikiTorch's documented closure cost.

### 4. Binary / ternary nets
BNN (Courbariaux & Bengio, 2016), XNOR-Net (Rastegari et al., 2016), and the
ternary/low-bit line through to BitNet b1.58 (Wang et al., 2023, weights in
{−1, 0, +1}). Quantize weights (and often activations) so the multiply degenerates:
±1 weights turn `W·x` into a sign flip / XNOR-popcount, ternary adds a zero. This
is the most deployed family by far and the most aggressive on energy.

Two honest caveats, both well known in that literature: (a) the *forward* is
multiply-free but a **scaling factor** (often a per-channel FP `α` and a
batch-norm) is reintroduced to recover accuracy — XNOR-Net's α, BitNet's
activation scale — so a multiply lives at the layer boundary; and (b) **training**
uses full-precision latent ("shadow") weights, a straight-through estimator, and a
standard FP optimizer (Adam) — the *learning loop* is firmly FP. Binary nets are
multiply-free at inference, not in the loop. That is the right tradeoff for their
goal (ship a small fast model); it is a different goal from TikiTorch's
(a learning loop that is closed end to end, including the backward pass and the
optimizer, all the way onto silicon).

### 5. TikiTorch — the tropical semiring, closed and audited
TikiTorch replaces `(+, ×)` with the tropical semiring `(min, +)` / `(max, +)`:
the "multiply" of the network *is ordinary addition*, the "addition" *is a
comparison* (min/max). Because that holds in the **forward, the backward (gradients
route through argmin/argmax), the normalization (`BatchShift` = subtract-mean,
`TropicalLayerNorm`), and the optimizer (`SignSGD` / `PureTropicalSGD`, shift-lr +
integer momentum)**, the entire learning loop is comparisons and additions. The
only standard-arithmetic survivors are the optional embedding gather and output
head, and `assert_closed` reports them.

This is the same neighborhood as the four families above — and TikiTorch borrows
freely from each (AdderNet's ℓ1 layer; ShiftAddNet's shift-lr/α; LNS's log-domain
relaxation as the soft training path; quantization for INT weights). What it adds
is not a sixth trick. It is two things none of the neighbors ship:

1. **A mechanical closure check.**
   `tikitorch.audit.assert_closed(model)` is the *structural* contract: it scans the
   module tree and fails if any `nn.Linear` / `nn.Conv` / `nn.LayerNorm` / standard
   norm is present — the layers that imply a dense multiply.
   `tikitorch.audit.assert_closed_runtime(model, x)` is the stronger, *sufficient*
   check: it traces a real forward under a `TorchDispatchMode` and flags any genuine
   `×` / `÷` / `exp` that a custom `forward` could hide, while treating `x * 2**k`
   and `x / 8` as bit-shifts (so legitimate pow-2 scales pass and a real multiplier
   is caught). This is the check none of AdderNet, ShiftAddNet, LNS, or BNN ship —
   each of those leaves "is the multiply *actually* gone, here, in this run?" to a
   careful reading of the paper and the BN/optimizer footnotes. TikiTorch makes it
   an assertion you can put in a test. It has already caught a real bug in our own
   code: the structural scan missed a `rope` ramp built with a true multiply;
   `assert_closed_runtime` flagged it, and it was rewritten as cumulative addition.
   The audit is the differentiator precisely because it is adversarial against
   *our own* claims.

2. **Verified zero-DSP silicon training.**
   The closure claim has a physical witness: the bundled HLS kernels implement a
   complete tropical LM **training step** (forward + loss + backward + sign-SGD
   update) that synthesizes to **zero DSP blocks** on AWS F2 (Virtex UltraScale+),
   validated end to end — a 2-layer tropical LM trained on silicon at ~772 tok/s and
   ~9.0 mJ/token, the first multilayer NN *trained* on FPGA with no multiplier on
   the chip. "Zero DSPs" is the place-and-route tool agreeing with `assert_closed`:
   there is literally no multiplier in the fabric. None of the four neighbor
   families has demonstrated multiplier-free *training* on zero-DSP silicon — they
   target multiply-*cheap* inference (BNN/LNS), or a multiply-free op inside an
   otherwise-FP training loop (AdderNet/ShiftAddNet).

The price is honest and stated plainly in the README and roadmap: hardening
`logsumexp` to `max` and replacing AdamW+CE with sign-SGD costs accuracy
(~2.5–5 pp on classification/sequence tasks; a larger, decomposed gap on hard
char-LM). TikiTorch is not claiming to beat a GPU transformer on perplexity. It is
claiming, and letting you reproduce, the one thing this neighborhood mostly only
gestures at: a learning loop that is *provably* closed, multiplier-free end to end,
on real silicon.

---

## The taxonomy, in one table

| family | core op | closed? (`assert_closed`) | trains on zero-DSP silicon? |
|---|---|---|---|
| **AdderNet** | ℓ1 distance `−Σ\|x−W\|` (a max-plus op) | no — needs BN (×) + adaptive-LR optimizer; *forward* op only | no (multiply-free conv inside an FP training loop) |
| **ShiftAddNet** | bit-shift then add | partial — shift+add datapath, but FP norm/optimizer; no audit | no (multiplier-free op, FP training) |
| **LNS nets** | multiply→add in log domain; LNS-adder for `+` | no — multiply survives (as add); pays at the add table; FP loop | no (multiply-cheap inference target) |
| **Binary / ternary** (BNN, XNOR, BitNet 1.58) | ±1/{−1,0,1} weights → XNOR/popcount/sign | no — per-channel scale (×) + BN; FP latent weights + Adam in training | no (multiply-free *inference*; FP training loop) |
| **TikiTorch (tropical)** | `(min,+)` / `(max,+)` — add *is* the multiply, compare *is* the add | **yes — structural `assert_closed` + runtime `assert_closed_runtime`**, whole loop | **yes — HLS training-step kernel, 0 DSPs, validated on AWS F2** |

The column that distinguishes TikiTorch is the third — the *mechanical, runtime*
closure check — and the fifth row's fourth column is the consequence: closure you
can audit is closure you can put on silicon, and confirm with place-and-route.
Everything else in the neighborhood is a genuinely good way to make a multiply
cheaper; TikiTorch is a way to make the whole learning loop *not contain one*, and
to prove it.

---

*See also:* [`TROPICAL_DL_ROADMAP.md`](TROPICAL_DL_ROADMAP.md) for the full
landscape and gap analysis, [`README.md`](README.md) for the closure contract,
benchmarks, and the F2 silicon results, and `tikitorch.nn.AdderLinear` for the
in-library AdderNet baseline.
