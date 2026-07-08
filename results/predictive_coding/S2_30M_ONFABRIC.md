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

## 3. Honest scope

- **Is:** a 30-step MEASURED loss descent (bpb −0.75, acc 0→16.8%) of a 30.6M-param, 8-block,
  model-parallel PC-transformer running its forward + cascade backward + dyadic-Adam on 8 physical FPGAs,
  with a byte-for-byte bit-exact forward vs the source reference.
- **Is not:** convergence (that requires the throughput work — 0.006 steps/s serial ⇒ ~2.4 days for one
  enwik8 epoch), a tokenizer-competitive LM claim (byte-level V=256), or an energy win.
- Embedding trains host-side (HOST_EMBED — the established workaround for the FEMB silicon DDR-read bug);
  LN γ/β frozen per recipe. Attention softmax `hls::exp` remains the one silicon-approximate op in the
  forward (bounded, does not affect the dyadic paths; it is inside the allowed-multiply attention front).
- Single seed, single data order; the 30-step window is before LR decay matters.

## 4. Reproduce

```bash
# forward bit-exactness (2-chip, ~20 min + ~1 min local):
SHIM_DEFS="" FWD_DUMP=1 OUTDIR=fwd_ref bash tikitorch/fpga/silicon/s2_shim/build_and_run.sh
AGFI=agfi-0c375fcd404810e17 bash tikitorch/fpga/silicon/s2_shim/SILICON_CMD.sh   # then fetch + compare_fwd.py

# 30-step training (8 chips):
AGFI=agfi-0c375fcd404810e17 ENWIK_S3=s3://tiki-sweep-739966866637/s2_bins/train.u16 \
NSLOTS=8 STEPS=30 ACCUM=8 XF_LR=2.6e-4 XF_WARMUP=20 LR_DECAY=0 PIPE=0 \
  bash scripts/run_s2_shard_f2.sh launch
```

Run logs: S3 `s2shard-20260707T200404Z` (bit-exact forward), `s2shard-20260708T104531Z` (30-step training).
