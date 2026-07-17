# ★ S2 30M PC-Transformer TRAINS ENTIRELY ON F2 SILICON — forward bit-exact, loss descends

**Date:** 2026-07-08 · **AGFI:** `agfi-0c375fcd404810e17` (production: laned/fast + bit-exact dyadic) ·
**Config:** D=512, H=8, DH=64, FFN=2048, SEQ=256, VOCAB=256 (byte-level enwik8), L=8 blocks **one per FPGA chip**
(f2.48xlarge, 8× VU47P), K=32 MAC lanes, 250 MHz, ~30.6M params ·
**Kernel:** `tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_s2_block.cpp` (DDR-streaming; commits
`a1168ad`→`0af45b5`) · **Driver:** `tikitorch/fpga/silicon/forge_pc_xformer_s2_shard.cpp` (serial PIPE=0 for this run) ·
**Harness:** `tikitorch/fpga/silicon/s2_shim/` (the bit-faithful diff tool that found the root cause)

Every number below is **[MEASURED]** on real F2 silicon unless tagged otherwise.

---

## 1. The results

### 1.1 Forward is BYTE-FOR-BYTE bit-exact on silicon

2-chip matched-init `FWD_DUMP` (chip0 = HID + host-embed, chip1 = OUT; byte-identical init + data window vs the
local shim compiled from the same source at the same flags — `s2_shim/SILICON_CMD.sh`):

| stage (both chips) | relerr | bitmatch |
|---|---|---|
| R, X1(LN1), Q, K, V, ATTN, M, X2, HD, RP, XF, LG — **all 22 dumps** | **0.00e+00** | **100.00%** |

The same check on the pre-fix AGFI measured X1 relerr **5.36e-2** (cosine 0.9992, 47% bitmatch) — the divergence
source, now eliminated at the arithmetic level (not tuned around).

### 1.2 Training descends on-fabric (8 chips, 30 steps)

`STEPS=30 ACCUM=8 XF_LR=2.6e-4 XF_WARMUP=20 LR_DECAY=0 PIPE=0`, byte-level enwik8:

| metric | step 0 | step 29 |
|---|---|---|
| bpb (raw) | 8.417 | **7.671** |
| bpb_ema (0.99) | 8.417 | **8.321** (EMA of a monotone-declining series) |
| next-token acc | 0.000 | **0.168** (byte-level chance ≈ 0.004) |

240 backward kicks, **no divergence, no hang, no KICK HUNG**. This is the first multi-step loss descent of the
D=512/30M model on silicon — every forward, backward (cascade), and dyadic-Adam update ran on the FPGAs
(embedding trained host-side via the established HOST_EMBED path; LN γ/β frozen as in the recipe).

### 1.3 Throughput (the next front, not a correctness issue)

| | per-chip kick | note |
|---|---|---|
| FWD | **818 ms** (OUTPUT 836 ms) | 6 GEMMs + attention + LN (+head on OUTPUT) |
| BWD | **1776 ms** (OUTPUT 1817 ms) | dx + wgrad ≈ 2.2× FWD MACs |
| serial step (8 chips × ACCUM=8) | **~167 s → 0.006 steps/s** | PCIe relay < 2% — kick-dominated |

Levers (see `THROUGHPUT_ROADMAP.md`): PIPE=1 slot pipeline (exists, bit-exact, ~2.95× measured on the D=64
kernel), the reverted-innocent `S2_AGGRESSIVE_SCHED` Bcache hoist, and the `-DS2_K` MAC-width sweep.

---

## 2. Root cause of the (weeks-long) divergence — the reusable finding

**The LayerNorm dyadic quantizer used `hls::log2` + `hls::round`, which Vitis HLS silicon APPROXIMATES.**
Per-token closed-form fit of silicon LN vs intended math (relerr **2.6e-7**, 100% of 256 tokens):

```
s_silicon = min( quant(scale), scale )        # instead of quant(scale)
```

— on round-UP tokens (53%) silicon fails to round the 2-bit mantissa up and keeps the raw scale. At
`S2_MBITS=2` that is a per-token LN-scale error up to **11%** → ~5% X1 error → ~20% of downstream gradient
signs flip → training diverges. **Why every check missed it:** csim and the shim map `hls::log2/round` to
*exact libm*; the approximation exists only in the generated hardware. s1 (D=64) was bit-exact because it used
an explicit integer CSD reciprocal; s2 had switched to `hls::log2` quantization.

**Fix** (commits `a1168ad`, `da7da96`): derive dyadic exponent/mantissa **directly from the IEEE-754 bits with
integer ops** — `s2_ilog2_floor` (= IEEE exponent), `s2_ilog2_round` (threshold `frac ≥ ceil((√2−1)·2²³)`),
`s2_dyad_quant` (mantissa round-half-up by integer add; scaling via `ldexp` = exponent add → `DSP_MULT=0`
kept), `s2_round_ne` (qact round-to-int via exact float→int truncation). Replaced **all** `hls::log2/round/floor`
sites in the forward/optimizer paths: LN normalizer, `bf_dyad_f` (forward GEMM operands), qact scale + round,
dyadic-Adam step. Validated 0-diff vs libm on dense sweeps; full shim forward byte-identical before rebuild;
silicon bit-exact after (§1.1). DSP went **down** 945→404 (bit-ops are cheaper than the intrinsic LUTs).

### 2.1 The false-trail ledger (each ruled out by a disproving experiment)

| hypothesis | disproving experiment |
|---|---|
| residual bf16 swamping | residual is fp32 end-to-end; CPU twin descends in every precision config |
| driver PIPE event loop | PIPE=0 serial diverged identically |
| gradient accumulation | ACCUM=1 == ACCUM=8 |
| backward GEMMs broken | HOST_BACKWARD (verified-correct host backward) still diverged; step-1 weights bit-identical to in-kernel backward → the shared *forward activations* were the bug |
| DDR-streamed reduction / pipelined-accumulate hazard | full de-pipeline rebuild (MAC IterLat 2421, csynth-confirmed) left X1 **identically** 5.36e-2 wrong |

The de-pipeline was a fix for a non-bug: it cost ~5× backward throughput (tripping the driver's 60 s kick
timeout, `KICK HUNG`) and changed nothing numerically — reverted (`b837dd1`); the laned II=1 reductions were
always faithful (8-way lane interleave hides the fadd latency). Reverting exposed a latent laned-path csynth
bug (`DEPENDENCE variable=YACC` named the macro parameter, not the array; invisible to g++/csim which ignore
pragmas) — fixed (`0af45b5`).

### 2.2 The localization method (reusable for any silicon-vs-source numeric bug)

1. Byte-identical init + fixed data window on silicon and shim (S3 `s2_locinit{,_data}`).
2. `FWD_DUMP=1` per-stage dumps on both → `s2_shim/compare_fwd.py` → first diverging stage.
3. **Per-token ratio analysis** on that stage: a per-ROW scalar fit with ~zero residual ⇒ a per-token
   normalizer (LN 1/std class). Reverse-engineer the closed form against the *intended math* — the shim is
   structurally blind to `hls::` intrinsic approximations; only silicon-vs-math can see them.
4. Verdict thresholds: order-noise floor X1 ~5e-8 / ATTN ~5e-4; the bug was 1000× above floor.

---

## 3. The SECOND divergence — training-stability, root-caused to the softmax `hls::exp` (2026-07-10)

§1's 30-step descent was **warmup only**. Held at peak LR past ~step 100, the D=512/L=8 model (moved to
BPE-8192, VOCAB=8192, K=64, prod AGFI `agfi-0b7691e71f2bbf079`) **descends then DIVERGES** — bpb 13.5→11.3
(warmup) then climbs **15.8@125 → 38@450** (confidently-wrong: logits saturate on the wrong class). Onset
delays as LR drops but appears at every productive LR. **The trap:** every prior "trains on F2" result was
≤40 steps — inside warmup — so this was never seen. *A "trains on fabric" claim needs >150 updates held at
peak LR.* The forward is bit-exact (§2), so this is a distinct, **biased-gradient** instability — §3.3 traces
it to the one silicon-approximate op §4 below had dismissed as "bounded, does not affect": the softmax `hls::exp`.

### 3.1 The false-trail ledger (each ruled out by a disproving experiment)

| hypothesis | disproving experiment |
|---|---|
| dyadic-Adam runaway | **proof**: `m`=EMA(g), `v`=EMA(\|g\|), *same* 0.875/0.125 coeffs ⇒ \|m\|≤v invariantly ⇒ step = lr·sign·2^floor(log2(\|m\|/(v+ε))) ≤ lr, always. The optimizer is bounded — cannot explode. |
| cross-step pipeline staleness | serial PIPE=0 diverged identically |
| small-batch gradient noise | EFF_BATCH=32 (4× effective batch) diverged identically ⇒ a *biased*, not variance, error (averaging can't remove it) |
| host embedding gradient | EMB_FREEZE=1 (embeddings frozen) still diverged (bpb→27); the CE-loss gradient *seed* is computed on-chip |
| a depth-amplified recipe bug (detached-scale LN backward) | **the L=8 exact-libm twin is stable** (§3.2) — refuted at the true depth |

### 3.2 The deconfounder — L=2 AND L=8 exact-libm CPU twins (the decisive method)

The shim `s2_twochip_shim.cpp` compiles the REAL kernel with `_mpshim` mapping every `hls::` intrinsic to
**exact libm**, then runs the full training loop on CPU; generalized to N chips (`NL` env, commit `14ae1c4`)
to run the true L=8 recipe. V=8192, rolling BPE data, held LR 2.6e-4, ACCUM=8, 160 steps:

| config | arithmetic | post-warmup CE (bits/tok) |
|---|---|---|
| L=2 | exact-libm twin | **stable** — mean 10.14, 0 steps >15 |
| L=8 | exact-libm twin | **stable** — mean 10.09, descends to 7.08, 0 steps >15 |
| L=8 | **F2 silicon** | **diverges → 38** |

Both depths train stably in *exact* arithmetic; silicon diverges at both. This **deconfounds depth** (the L=2
twin alone was ambiguous — it changed exp-exactness AND depth at once) and isolates the divergence to a
**silicon-approximated intrinsic** the shim computes exactly. Per `_mpshim` those are `hls::exp/sqrt/log2`;
the dyadic paths are already bit-exact integer (§2) and `hls::sqrt` is a shared compile-time 1/√DH constant, so
the **softmax `hls::exp`** is the sole variable-input suspect — the CE-loss seed (highest risk: no forward
master keeps it consistent) plus the attention softmax.

### 3.3 The fix — `s2_exp_dyad` (bit-exact mult-free exp), commit `0be81fe`

Port the golden kernel's proven precise exp (`g_exp_q16`) into float using §2's integer-bit template: for x≤0,
`d=−x`; `u = d·log2(e)` via a 6-term shift-add (`1+¼+⅛+1/16+1/256+1/1024`, exact `ldexp` — **no multiply**);
split `u` into octave + 8-bit fraction; `exp = 2^−u = S2_GEXP[frac] >> octave` with a **compile-time 256-entry**
`2^(−f/256)` mantissa ROM. Every op is deterministic int / `ldexp` / ROM ⇒ **BIT-IDENTICAL on silicon and
shim** (the property that makes the shim a valid oracle, exactly as the §2 LN fix). 0.09–0.18% vs `expf` in the
softmax band; args < −21.5 floor to 0 (true mass ~0). All 5 `hls::exp` sites swapped (fwd attn, bwd PRAW, CE seed).

- **Twin validation** — L=8 twin REBUILT with `s2_exp_dyad` tracks the exact-libm twin bit-for-bit and stays
  stable (mean **10.09**, 0 divergence, matching exact-libm through the full run).
- **CSYNTH gate** — clean: **0 new DSP multiplies** (`ldexp`→`generic_ldexp_float_s` DSP=0; the 1349 DSP are the
  pre-existing sanctioned attention/bf16 mults), timing closes 250 MHz, +3 BRAM (the ROM) / +6k LUT (fits SLR).
- **P&R** — TIMING MET (~2.3h), AGFI **`agfi-0bb58e4b8eee1e199`**.
- **Silicon gate** — _[PENDING]_ held-LR 2.6e-4, 300 steps on the fixed AGFI (run `s2shard-20260711T024736Z`):
  must stay bounded/descending past step 125 where the old kernel hit 38.

## 4. Honest scope

- **Is:** a 30-step MEASURED loss descent (bpb −0.75, acc 0→16.8%) of a 30.6M-param, 8-block,
  model-parallel PC-transformer running its forward + cascade backward + dyadic-Adam on 8 physical FPGAs,
  with a byte-for-byte bit-exact forward vs the source reference.
- **Is not:** convergence (that requires the throughput work — 0.006 steps/s serial ⇒ ~2.4 days for one
  enwik8 epoch), a tokenizer-competitive LM claim (byte-level V=256), or an energy win.
- Embedding trains host-side (HOST_EMBED — the established workaround for the FEMB silicon DDR-read bug);
  LN γ/β frozen per recipe. **The softmax `hls::exp` — flagged here originally as "bounded, does not affect"
  — was the §3 training-divergence cause; fixed by `s2_exp_dyad` (§3.3). It is now the last silicon-approximate
  intrinsic eliminated: every `hls::log2/round/floor/exp` in the fwd+bwd is bit-exact integer/ROM.**
- The §3 convergence claim is pending the silicon gate; the §1 window was 30 steps (warmup, pre-LR-decay).

## 5. Reproduce

```bash
# §3 deconfounder — L=8 exact-libm CPU twin (no F2), then the fix twin (must both stay stable):
cd tikitorch/fpga/silicon/s2_shim
SEED=7 CHIPS=8 OUTDIR=data_v8k ./build/gen_init                    # V=8192 matched init, 8 chips
INIT_DIR=data_v8k ENWIK=data_v8k/train.u16 NL=8 STEPS=160 ACCUM=8 \
  XF_LR=2.6e-4 XF_WARMUP=10 ROLL=1 ./build/s2twochip               # exact-libm twin (rebuild with s2_exp_dyad = fix twin)

# §3.3 silicon gate on the exp-fix AGFI (held LR — the divergence condition):
AGFI=agfi-0bb58e4b8eee1e199 S2_DEFS="-DS2_K=64 -DS2_VOCAB=8192 -DS2_SEQ_TILE=256" \
ENWIK_S3=s3://tiki-sweep-739966866637/s2_bpe8192_bins/train.u16 \
NSLOTS=8 STEPS=300 ACCUM=8 XF_LR=2.6e-4 XF_WARMUP=10 LR_DECAY=0 PIPE=1 \
  bash scripts/run_s2_shard_f2.sh launch

# forward bit-exactness (2-chip, ~20 min + ~1 min local):
SHIM_DEFS="" FWD_DUMP=1 OUTDIR=fwd_ref bash tikitorch/fpga/silicon/s2_shim/build_and_run.sh
AGFI=agfi-0c375fcd404810e17 bash tikitorch/fpga/silicon/s2_shim/SILICON_CMD.sh   # then fetch + compare_fwd.py

# 30-step training (8 chips):
AGFI=agfi-0c375fcd404810e17 ENWIK_S3=s3://tiki-sweep-739966866637/s2_bins/train.u16 \
NSLOTS=8 STEPS=30 ACCUM=8 XF_LR=2.6e-4 XF_WARMUP=20 LR_DECAY=0 PIPE=0 \
  bash scripts/run_s2_shard_f2.sh launch
```

Run logs: S3 `s2shard-20260707T200404Z` (bit-exact forward), `s2shard-20260708T104531Z` (30-step training).

---

## 6. ★ THE ACTUAL divergence root cause — the softmax exp ROM read as ZERO on silicon (RESOLVED 2026-07-15/16)

**Date:** 2026-07-15 · **AGFI:** `agfi-0e562920e6ccbe3ab` (ROM-free-exp fix, VOCAB=8192) · **Fix commit:** `848504a`

§3.3 believed the softmax was fixed by replacing `hls::exp` with the dyadic `s2_exp_dyad` (`0be81fe`), and §4
called it "the last silicon-approximate intrinsic eliminated." **That was wrong.** `s2_exp_dyad` was *itself*
silently broken on F2: its 256-entry `S2_GEXP` mantissa ROM — `static const float S2_GEXP[256] = {…}` — was
synthesised by Vitis HLS as **UNINITIALISED BRAM that reads ZERO** on silicon (an initialised ROM in csim/shim).
So `return s2_ldexp_i(S2_GEXP[fi], -octave)` returned 0 for every `x<0`, making the exp a **step function**
(1 at the max, 0 elsewhere) → the attention softmax collapsed to **one-hot (hard argmax)** → the softmax score
gradient is *correctly* zero → `gQ/gK` never train → Q/K frozen → the imbalanced model diverged to bpb 38.

### 6.1 The measured proof chain (DDR-snapshot instrumentation, not inference)
Every step was *measured* on silicon by snapshotting on-chip scratch to DDR (`scrGSV`/`scrPALL`/`scrSC`) and
reading it back — not reasoned about:
1. Forward `Q/K/V` are **bit-exact** vs the exact-math shim (cos 1.0000).
2. Raw pre-exp scores (`scrSC`) are **correct**: ratio 1.000 vs numpy `dot*0.125`, span ~2-3 → the dot and the
   `1/sqrt(DH)=0.125` scale are both right.
3. For the 2nd-best key at `x=-0.489`, silicon `exp = 0` vs the correct `0.613` → step-function exp.
4. Forward `ATTN == V[argmax]` at cos **1.0000** (hard attention); `scrPALL` one-hot (1/101 nonzero per row).

**After the fix** (silicon, `agfi-0e562920e6ccbe3ab`): fwd `ATTN` cos **1.0000** to the *smooth* softmax blend,
`scrPALL` **101/101** nonzero, `gQ/gK/gV` **nonzero** (were bit-exact 0). The exact held-LR 2.6e-4 config that
blew up to bpb 38 now stays **bpb_ema ~13.4 bounded, monotonically descending, past the old update-125 blow-up
point** — divergence gone.

### 6.2 The fix
Drop the ROM; compute `2^(-frac)` with a degree-4 Taylor poly of `exp(-frac*ln2)` (verified vs libm: max relerr
0.24% on [-8,0], `exp(-0.489)=0.6135`). Attention is the allowed-multiply front, so the mults are fine; every op
is now a plain float mul/add or integer bit-op — **no ROM, no silicon-approx intrinsic → faithful by construction.**

### 6.3 The false-trail ledger — EIGHT prior AFI builds, all downstream of this
`gsv=0` is the *correct* gradient of a one-hot softmax, so every gradient-side "fix" was fixing correct code:
gp[] write-drop (`d333bce`), V-deviation reformulation, phase-1 de-pipeline (`22da155`), pre-transpose revert
(`a6a7e69`); plus two **byte-identical no-ops** — bit-exact-ldexp (`2b170e5`; `hls::ldexp` is actually fine) and
scale-literal (`a1dba0a`; `1.0f/hls::sqrt(64)` already const-folds to 0.125). What finally cracked it was
**instrumentation over inference** (the `scrSC` score dump). **META-LESSON:** when a gradient is exactly zero,
first check whether zero is the *correct* gradient for the (possibly degenerate) forward — trace UPSTREAM before
touching gradient code; and **dump-verify any large `static const` LUT on F2** (it can synthesise to zeroed BRAM).

### 6.4 Honest scope of THIS result
- **Is:** the weeks-long train-then-diverge failure ROOT-CAUSED and FIXED, silicon-verified (smooth softmax +
  nonzero attention gradients), and confirmed non-diverging under sustained training at the exact reproducing config.
- **Is NOT (still open, separate problem):** *convergence* to low bpb. An overnight run (~1600 updates ≈ 3.3M
  tokens) descends only bpb 13.47→13.40 — the model learns *correctly but glacially*, a property of the mult-free
  **dyadic Adam (bounded sign-SGD)**, plus a throughput wall (~26-32 s/step). Note: `S2_K=128` is NOT the
  throughput lever (measured 1.26×; the walls are DDR re-stream + attention MAC + driver host-tail per
  THROUGHPUT_ROADMAP). Convergence is the next phase: optimizer strength within the mult-free constraint + the
  correct throughput fixes.

Run logs (S3): score-dump proof `s2shard-20260715T101523Z`; fix verify `s2shard-20260715T124833Z`; held-LR
non-divergence `s2shard-20260715T130638Z`; overnight K=128 `s2shard-20260716T133033Z`.
