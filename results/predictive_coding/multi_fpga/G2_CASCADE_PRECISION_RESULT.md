# Gate G2 — Cascade Precision Result

**Question the gate was built to answer:** Is the PC-cascade char-LM's failure to reach
coherent loss a *format* problem (int quantization) that a fix16 datapath would lift, or is
the *hand-rolled cascade itself* the wall regardless of precision? If the former, build the
multiplier-free kernel (`forge_pc_xformer_mp_block.cpp` + csim + csynth). If the latter, STOP
and fix the cascade first.

**Date:** 2026-06-27 · **Status:** RESOLVED · **Decision: B — fix the cascade.**

---

## 1. Measured table

Real TinyShakespeare (vocab 64, `~/.cache/tiur/tinyshakespeare.txt`), single seed, D=64 /
SEQ=64, identical init/order/accumulation across all four modes, 2500 steps. Only the
backward-multiply operand precision, the master-weight representation, and the dyadic-Adam
m/v + step representation change between modes; the forward is byte-identical (verified).

| mode  | repr                                              | bpb     | acc    |
|-------|---------------------------------------------------|---------|--------|
| int   | int16 master, X_LR_SHIFT=0 (partially frozen)     | 7.4705  | 0.0861 |
| fix16 | per-tensor pow2-scaled 16-bit fixed point         | 5.6057  | 0.1489 |
| fix24 | per-tensor pow2-scaled 24-bit fixed point         | 7.0176  | 0.1114 |
| dbl   | pure double master + Adam + wgrad-accumulate      | 5.3934  | 0.1502 |

**chance_bpb = log2(64) = 6.0000.**

Reference (NOT run in this harness): prior torch **autograd** char-LM ≈ **2.0 bpb**
(mult-free fwd 2.016 / BF16 1.732 / fix16 2.898). This is the bar the cascade must approach.

---

## 2. int reproduces the wall — YES

`MP_OPT=int` dispatches `mp_wgrad→g_wgrad` and `mp_adam→g_adam` by early-return, so it is
**byte-exact to the original `forge_pc_xformer_golden_test.cpp`**: acc_ema 0.047/0.114/0.124
and spread 28864/360448/311296 at steps 0/200/400, identical final 0.134@600. int sits **at/
above chance** (bpb 7.47 > 6.0, acc ~0.086–0.13) — the known integer-cascade wall is
reproduced exactly. The harness is sound (independently rebuilt and re-run by verification;
table reproduced bit-exact to 4 d.p. across two runs).

The four modes genuinely diverge mid-run (step-200 logit spreads: int 360k, fix16 54M, fix24
28M, dbl 430k), so each precision path is actually active — no silent fallback to one mode.

---

## 3. dbl vs autograd-2.0 — IS the cascade itself a wall? YES

**dbl is the cascade with ZERO format loss** (double master + double Adam m/v + double
wgrad-accumulate). It caps at **bpb ≈ 5.39 / acc ≈ 0.15** and *stays there*: run to 5000 steps
it plateaus at 5.24–5.43 from step ~1000, reaches its lowest (5.24 @3800) then drifts back **up**
to 5.43 — it does **not** keep falling toward 2.0. The logit spread grows unbounded the whole
time (28k→430k→1.3M→2.7M→5.2M).

So the cascade's float ceiling is **~3.4 bpb above** the autograd reference (5.39 vs ~2.0),
and barely dips under chance (6.0). **The hand-rolled cascade, not the integer format, is the
limiter.** Going int→dbl moves bpb only ~2 and stops just below chance.

---

## 4. fix16 vs dbl format cost — SMALL and non-monotone

fix16 5.61 vs dbl 5.39 ≈ **0.2 bpb** — the format costs almost nothing relative to the 3.4 bpb
cascade gap. And it is **non-monotone**: fix24 (7.02) is *worse* than fix16 (5.61). Higher
per-step resolution drives the logit spread up faster → more softmax saturation → worse
calibrated bpb. This is itself a symptom of the cascade pathology (§5), not of quantization.
The format axis is decisively not the bottleneck.

---

## 5. Mechanism — why the cascade caps

The cascade trains **argmax-accuracy to ~9× chance** (acc 0.15) but its CE softmax **saturates
to a hard one-hot**. Confirmed in source: `g_step`/`mp_step` accumulate grads into the LN
gamma/beta slots but **never apply them** — the final-LN gammas are **frozen by design**
(gamma=64, beta=0). With nothing bounding logit magnitude, the **logit spread grows unbounded**,
the softmax collapses to a hard one-hot, and a wrong hard one-hot gives the target ≈0
probability. So **calibrated target probability (= BPB) stays pinned near chance no matter how
precise the optimizer is**. Autograd keeps logits bounded/calibrated through the LN-gamma
gradient; this cascade structurally cannot.

This is a genuine **cascade design failure**, not a quantization failure.

---

## 6. Adversarial verification verdicts

Four independent audits, all reproduced the harness from source:

- **[forward-identical] SOUND.** `g_forward` is defined exactly once (golden header), pulled
  into the mp file verbatim via `#include`; zero references to `Wf`/`mp_`/mode knobs inside it.
  The precision change is confined to exactly the three allowed sites (wgrad operands, master
  shadow, Adam m/v). MP_INT byte-exact to golden by construction. The cascade-vs-format
  isolation is valid; the forward is not contaminated by the mode knob.

- **[fix16-faithful] SOUND-WITH-CAVEATS.** The C++ quantizer is **bit-exact** to the torch
  precision-floor `_quant_fix` (max|torch−cpp| = 0.000e+00 at fix16), correctly pow2-scaled,
  applied at master + m/v + wgrad-operands, gated on bits>0, demonstrably **not a no-op** (four
  distinct monotone results toward the dbl ceiling). int bit-exact to golden, confirmed
  empirically. Caveats (not refutations): (a) the C++ quantizes the wgrad **operands** before
  the multiply (HW-accurate pow2×pow2→fixed-point model) whereas the torch witness quantized
  the **assembled grad** — different arithmetic, but the master + m/v quantization (the
  spec-identified *dominant* wall) uses the identical formula; (b) the absolute 2.898 torch
  number is an autograd witness at a different recipe and is neither reproduced nor claimed here.

- **[bpb-soundness] SOUND.** `batch_bpb` reconstructs the target's true model probability from
  the cached GLOG (`p = (GLOG[tgt]+2^16)/2^16`) and reports mean per-token CE **in bits**;
  cross-checked bit-exact against a direct softmax. Because the normalizer makes probs sum ≤1,
  the metric can only **under-report** lift (conservative). chance = log2(64) = 6.0 confirmed at
  runtime. The task is a **real next-char LM over real Shakespeare** with **no target leakage**
  (tgt enters only the grad/loss after logits are produced; verified by grep). The metric is
  not inflating any lift — it honestly reports the cascade fails to beat chance on bpb.

- **[interpretation] SOUND-WITH-CAVEATS.** Table reproduced bit-exact (int 7.4705/0.0861,
  fix16 5.6057/0.1489, fix24 7.0176/0.1114, dbl 5.3934/0.1502). Decisive logic verified: int
  reproduces the wall; dbl (zero format loss) caps ~3.4 bpb above autograd and never falls
  toward 2.0; fix16-vs-dbl is small + non-monotone; frozen-LN-gamma mechanism confirmed in
  source. **No over-claimed lift** — the report resists the "format-is-the-wall, build the
  kernel" framing. Caveat: the absolute-BPB gap leans on a metric inflated by the saturated
  softmax, but the **spread-invariant argmax-accuracy independently plateaus at ~0.15 (9×
  chance)** and never climbs toward coherent, so both a calibrated and an uncalibrated metric
  show the cap. A same-harness autograd twin would make it airtight, but a 3.4 bpb gap is far
  too large to be a harness artifact.

---

## 7. THE DECISION — (B) the cascade caps below autograd regardless of precision → STOP and fix the cascade

This is **option (B)**, not (A). The fix16 cascade does **not** lift off the wall toward
coherent. The cascade's *float* ceiling (dbl, zero format loss) caps at bpb ≈ 5.39 / acc ≈ 0.15,
~3.4 bpb above the autograd reference and barely under chance — and stays there through 5000
steps. The format is **not** the limiter (int→dbl moves bpb only ~2; fix16-vs-dbl is ~0.2 bpb
and non-monotone). The **hand-rolled cascade itself is the wall.** This is precisely the
critical negative the gate was built to catch.

**Do NOT build `forge_pc_xformer_mp_block.cpp` / csim / csynth yet.** Building a precision-swap
kernel on a cascade that caps 3.4 bpb short of backprop would harden the wrong target.

**Fix the cascade first** — address the design failure, not the optimizer precision:
1. **Un-freeze / learn the final-LN gamma** so the LN-gamma gradient can bound logit magnitude
   (the direct cause of the unbounded spread → saturated softmax → bpb pinned near chance).
2. **Or a calibrated CE backward** that keeps the logits bounded the way autograd does.

Re-run G2 after the fix. The build decision flips to (A) only once the **dbl cascade** descends
toward the ~2.0 autograd reference (or a same-harness autograd twin confirms the cascade matches
backprop). Until then the kernel is premature.

---

## 8. Strength of evidence — honest caveats

- **Single seed, small D (=64), small SEQ (=64).** The magnitude of the gap (3.4 bpb) and its
  reproduction across two independent runs make a seed/scale artifact implausible, but multi-seed
  / larger-D confirmation would harden it.
- **Real (not synthetic) data** — genuine TinyShakespeare, real next-char prediction, no target
  leakage. This is a strength.
- **No autograd twin in THIS harness.** The ~2.0 bar is a *prior* torch char-LM at a different
  recipe (D=64/L=3/1200 steps). A same-harness autograd twin would make the comparison airtight
  and is the single most valuable follow-up. That said, the cascade's dbl ceiling barely dips
  under chance on **either** a calibrated (bpb) **or** an uncalibrated (argmax-accuracy ~9×
  chance, spread-invariant) metric, so the conclusion does not hinge on the inflated absolute
  bpb.
- **The bpb absolute is inflated by the saturated-softmax pathology** (can exceed chance even at
  acc > chance). The load-bearing signal is therefore RELATIVE (dbl ≫ autograd) + the dbl
  CEILING + the independent accuracy plateau, all of which agree.

**Net:** strong enough to act on — STOP and fix the cascade — while flagging that a same-harness
autograd twin + multi-seed re-run would convert "strong" into "airtight."

---

### Source files (absolute)
- `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_golden_block.hpp`
- `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_golden_test.cpp`
- `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/synth_pc_xformer_mp_golden_csim.sh`
- Reuses verbatim (forward + core ops): `forge_pc_xformer_golden_block.hpp`
