// forge_buffer_morph_lm.hpp — shared flat int32 buffer layout for the
// "config B" MorphSSM LANGUAGE-MODEL training kernel
// (forge_train_step_morph_lm).
//
// Mirrors forge_buffer.hpp / forge_buffer_morph.hpp style exactly: a single
// flat int32 buffer in DDR with constexpr per-tensor word counts (WORDS_*)
// and cumulative word offsets (OFF_*). One monolithic kernel reads only the
// regions it needs and writes only the regions it produces, dispatched by
// `mode`.
//
// ON-CHIP design (this is "config B"): UNLIKE forge_buffer_morph.hpp (which is
// hybrid — host owns embed + classification head + cross-entropy), this kernel
// owns the WHOLE LM training step on-chip: embed gather, the 4-layer morph
// core, the tropical LM head, AND the balanced LUT cross-entropy (or 2-hot
// perceptron) loss-gradient. Only compare / add / shift ops on the data path —
// ZERO multiplies (mean over D is >>ML_LOG2_D, a shift because D is a power of
// two; index arithmetic c*K etc. is small loop-counter address math, not a
// data-path multiply).
//
// BUILD-TIME WIDTH: the model dimension ML_D defaults to 128 (the deployed
// sentinel AFI agfi-0433370d93cb58a78) but is overridable via -DML_D=<pow2>
// (e.g. -DML_D=512). ML_LOG2_D is DERIVED from ML_D (ml_ilog2) and a
// static_assert((1<<ML_LOG2_D)==ML_D) makes any mismatch a compile error. Every
// WORDS_*/OFF_* count, on-chip array dim, the shift-normalize, and ML_BUFFER_WORDS
// is expressed in terms of ML_D/ML_LOG2_D, so a single -DML_D=512 rederives the
// whole layout; gen_buffer_init.py reads the same width from env ML_D so the
// driver-staged buffer matches byte-for-byte. NOTE: ML_V (vocab) and ML_L_SEQ
// (window) are SEPARATE shape constants and stay 128 under a width override —
// only the channel dimension D scales.
//
// =============================================================================
// POSITION-BY-POSITION execution model (this is the key difference from
// forge_buffer_morph.hpp, which runs the whole sequence in one mode-0 call).
// The LM is causal and trained token-by-token, exactly like the 1-layer LM
// kernel forge_train_step.cpp:
//
//   mode 0 (FWD+LOSS) at position m, token, target:
//       embed gather R0[m]=EMBED[token]
//         -> for l in 0..N-1:  BS_l[m] = batch_shift_D(R_l[m])      (norm over D)
//                              MC_l[m][c] = (max|min)_k h_l[c][k] + BS_l[m-k*dil][c]
//                              PJ_l[m][j] = max_i (P_l[j][i] + MC_l[m][i]) + MB_l[j]
//                              R_{l+1}[m] = sat(R_l[m] + PJ_l[m])   (residual)
//         -> head:  logits[v] = max_d (W_HEAD[v][d] + y[m][d]),  y[m]=R_N[m]
//         -> grad_logit[m] = balanced LUT-CE (CFG=1) | 2-hot perceptron (CFG=0)
//       The morph conv at position m reads BS_l at EARLIER positions
//       (m-k*dil >= 0), so the post-norm BS stream is a ROLLING CACHE that
//       PERSISTS across the per-position mode-0 calls — the morph analogue of
//       K_CACHE/V_CACHE in forge_buffer.hpp. Host walks m = 0..L_SEQ-1.
//
//   mode 1 (BWD) at position m: head bwd -> morph layers bwd (sparse scatter
//       through saved TAP_IDX/PROJ_IDX) -> embed grad. The conv backward
//       scatters grad_MC_l[m][c] into grad_BS_l[m-k*dil][c] — an EARLIER
//       position. So the BS_STREAM region is REUSED as the grad_BS
//       accumulator: host zeroes it between the fwd loop and the bwd loop
//       (exactly as forge_train_step.cpp reuses K_CACHE/V_CACHE as grad
//       accumulators). Host walks m = L_SEQ-1 .. 0 (REVERSE), so by the time
//       position m is processed every later position has already scattered its
//       conv-tap grad into grad_BS_l[m] — the complete-by-read invariant that
//       makes the per-position decomposition bit-identical to a full-sequence
//       backward.
//
//   mode 2 (SIGNSGD): region-wide SignSGD over every weight region vs its grad
//       region, zeroing the grads (verbatim helper from forge_train_step.cpp).
//
// Per-layer core block (alternating by parity, identical math to
// forge_train_step_morph.cpp):
//   even layer l -> DILATION (max-plus) depthwise morph conv
//   odd  layer l -> EROSION  (min-plus) depthwise morph conv
//   dilation d_l = min(2^l, ML_DIL_CAP); tropical projection is ALWAYS max-plus
//   (+ an additive tropical out-proj bias MB, post-max, like b1/b2 in
//   forge_train_step.cpp's FFN — the only extension over forge_*_morph.cpp).
//
// Saved for backward (sparse-scatter argmax routing, exactly like the existing
// kernels' ad_head / proj_idx / tap_idx):
//   TAP_IDX [l][m][c]  — argmax/argmin tap index k* per (layer,position,channel)
//   PROJ_IDX[l][m][j]  — argmax input index i* per (layer,position,out)
//   AD_HEAD [m][v]     — argmax feature index d* per (position,vocab)
//
// ABI matches the v6 silicon-validated wide-AXI pattern: int32 buffer pointer
// + small int args via s_axilite, max_widen_bitwidth=512.

#ifndef FORGE_BUFFER_MORPH_LM_HPP
#define FORGE_BUFFER_MORPH_LM_HPP

#include "../forge_hls_types.hpp"
#include <cstdint>

// =============================================================================
// Shape constants — "config B" MorphSSM char-LM at V=L_SEQ=128, N_LAYERS=4, and
// build-time channel dimension ML_D (default 128; -DML_D=512 for the larger
// deployed artifact). All sizes are powers of two (or multiples of 16) so every
// WORDS_* divides 16 — required for the 512-bit (16-word) wide-AXI burst helpers
// — and the divide-by-N normalizations collapse to arithmetic shifts (no
// divider, no multiply on the data path).
// =============================================================================
// ML_D: model / channel dimension. Build-time selectable — default 128 (the
// deployed sentinel AFI), override with -DML_D=512 (any power of two). Defined as
// a macro (not a constexpr int) so the -D flag can override it; every derived
// constant below expands it as an integer literal, so all constexpr/array/loop
// contexts behave exactly as before at the default.
#ifndef ML_D
#define ML_D 128
#endif
constexpr int ML_V        = 128;   // vocabulary (ASCII chars, clipped to 128)
constexpr int ML_L_SEQ    = 128;   // sequence length / training window
constexpr int ML_N_LAYERS = 4;     // number of morph SSM layers
constexpr int ML_K        = 7;     // morph conv kernel taps
constexpr int ML_DIL_CAP  = 64;    // cap on per-layer dilation d_l = min(2^l, cap)

// =============================================================================
// BATCH parallelism (TT_BATCH). ML_BATCH = number of INDEPENDENT sequences a
// single fused training step processes data-parallel: the weights (and resident
// weight arrays + the on-chip grad accumulators) are SHARED; each sequence keeps
// its OWN activation scratch (BS stream / saved argmax routes / grad_logit /
// pred) and its OWN tokens+targets; the per-step gradients SUM across the B
// sequences into the shared accumulators; ONE SignSGD update per step. ML_BATCH
// is 1 in non-batch builds so the shipped buffer layout (and AFI) is UNCHANGED;
// under TT_BATCH it grows every per-sequence region by ML_BATCH (the shared
// weight/grad regions stay one copy). See forge_train_step_morph_lm.cpp's
// do_train_run TT_BATCH path and the off_*(b, ...) helpers below.
#ifdef TT_BATCH
#ifndef ML_BATCH
#define ML_BATCH 8
#endif
constexpr int ML_BATCH_N = ML_BATCH;
#else
constexpr int ML_BATCH_N = 1;
#endif
static_assert(ML_BATCH_N >= 1, "ML_BATCH must be >= 1");

// log2 helpers for the mean-subtract (>>) normalization. D is a power of two so
// the per-token divide-by-D collapses to an arithmetic shift. ML_LOG2_D is
// DERIVED from ML_D (compile-time recursion), never a standalone literal, so a
// -DML_D=512 build automatically shifts by 9. The static_assert turns a
// non-power-of-two ML_D (or a hand-mismatched ML_LOG2_D) into a compile error.
constexpr int ml_ilog2(int n) { return (n <= 1) ? 0 : (1 + ml_ilog2(n >> 1)); }
constexpr int ML_LOG2_D   = ml_ilog2(ML_D);   // 128 -> 7, 512 -> 9
static_assert((1 << ML_LOG2_D) == ML_D,
              "ML_LOG2_D must equal log2(ML_D): ML_D must be a power of two "
              "(-DML_D=512 -> ML_LOG2_D=9)");

// Mode codes (match forge_train_step.cpp dispatcher style).
constexpr int ML_MODE_FWD       = 0;
constexpr int ML_MODE_BWD       = 1;
constexpr int ML_MODE_SIGNSGD   = 2;
// Mode 3: fused whole-run on-fabric trainer — runs arg1 optimizer steps
// (each: zero-BS, fwd 0..L-1, zero-grads+BS, bwd L-1..0 reverse, signsgd
// arg2=lr/arg3=threshold) entirely in-kernel, tokens/targets read from the
// resident buffer, weights reloaded once per sweep (resident across positions).
// One host kick trains the whole window; arithmetic identical to looping 0/1/2.
constexpr int ML_MODE_TRAIN_RUN = 3;

// =============================================================================
// Per-tensor word counts
// =============================================================================
// Weights:
constexpr int WORDS_EMBED    = ML_V * ML_D;                 // embedding table   [v][d]
constexpr int WORDS_MH       = ML_N_LAYERS * ML_D * ML_K;   // morph conv kernels [l][c][k]
constexpr int WORDS_MP       = ML_N_LAYERS * ML_D * ML_D;   // out-proj matrices  [l][j][i]
constexpr int WORDS_MB       = ML_N_LAYERS * ML_D;          // out-proj bias      [l][j]
constexpr int WORDS_W_HEAD   = ML_V * ML_D;                 // LM head            [v][d]

// Gradient accumulators (one per weight region, same sizes):
#if ML_D >= 512
// D>=512: the buffer G sections are CULLED (zero words). Grads live ONLY in
// the on-chip g_*_oc arrays: under TT_RESIDENT_W+TT_ONCHIP_GRAD (mandatory at
// this width, enforced below) NO code path reads or writes buffer grads —
// do_signsgd uses signsgd_region_oc, the grad-zero runs on-chip, and the
// grad flush at the end of do_train_run is guarded !TT_RESIDENT_W. Culling
// drops 4.58 MB so the D=512 buffer (7.69 MB) fits the 8 MB staging window
// (axi_bram_ctrl caps at 65536 lines = 4 MB/bank; the CL banks two of them).
// The D=128 layout below is UNTOUCHED (deployed-AFI ABI, sentinel-exact).
#if !defined(TT_ONCHIP_GRAD) || !defined(TT_RESIDENT_W)
#error "ML_D>=512 requires -DTT_RESIDENT_W -DTT_ONCHIP_GRAD (buffer G sections are culled to fit the 8MB staging window; non-resident builds would silently not train)"
#endif
constexpr int WORDS_G_EMBED  = 0;
constexpr int WORDS_G_MH     = 0;
constexpr int WORDS_G_MP     = 0;
constexpr int WORDS_G_MB     = 0;
constexpr int WORDS_G_W_HEAD = 0;
#else
constexpr int WORDS_G_EMBED  = WORDS_EMBED;
constexpr int WORDS_G_MH     = WORDS_MH;
constexpr int WORDS_G_MP     = WORDS_MP;
constexpr int WORDS_G_MB     = WORDS_MB;
constexpr int WORDS_G_W_HEAD = WORDS_W_HEAD;
#endif

// Per-sequence region STRIDES (one sequence's worth of words). Under TT_BATCH
// the buffer holds ML_BATCH_N of each of these back-to-back; sequence b lives at
// OFF_<region> + b * <region>_STRIDE. In non-batch builds ML_BATCH_N==1 so the
// region == its stride and the layout is byte-identical to the shipped buffer.
constexpr int STRIDE_TAP_IDX   = ML_N_LAYERS * ML_L_SEQ * ML_D;   // [l][m][c]
constexpr int STRIDE_PROJ_IDX  = ML_N_LAYERS * ML_L_SEQ * ML_D;   // [l][m][j]
constexpr int STRIDE_AD_HEAD   = ML_L_SEQ * ML_V;                 // [m][v]
constexpr int STRIDE_BS_STREAM = ML_N_LAYERS * ML_L_SEQ * ML_D;   // [l][m][d]
constexpr int STRIDE_TOKENS     = ML_L_SEQ;                       // [m]
constexpr int STRIDE_TARGETS    = ML_L_SEQ;                       // [m]
constexpr int STRIDE_GRAD_LOGIT = ML_L_SEQ * ML_V;               // [m][v]
constexpr int STRIDE_PRED       = ML_L_SEQ;                       // [m]

// Saved argmax index arrays — POSITION-MAJOR ([l][m][.]) so each per-position
// row is D (or V) contiguous words, matching the position-by-position kernel.
// PER-SEQUENCE under TT_BATCH (grown by ML_BATCH_N).
constexpr int WORDS_TAP_IDX  = ML_BATCH_N * STRIDE_TAP_IDX;    // [b][l][m][c]
constexpr int WORDS_PROJ_IDX = ML_BATCH_N * STRIDE_PROJ_IDX;   // [b][l][m][j]
constexpr int WORDS_AD_HEAD  = ML_BATCH_N * STRIDE_AD_HEAD;    // [b][m][v]

// Per-layer post-norm activation stream BS_l[m][d]. DOUBLES as the backward
// grad_BS accumulator (host zeroes it between the fwd and bwd loops — exactly
// like forge_train_step.cpp reuses K_CACHE/V_CACHE). Forward writes BS_l[m] and
// reads BS_l[m-k*dil] (the rolling causal-conv cache); backward scatters
// grad_MC_l[m] into grad_BS_l[m-k*dil] and consumes grad_BS_l[m] in the
// batch_shift backward.
// PER-SEQUENCE under TT_BATCH (grown by ML_BATCH_N).
constexpr int WORDS_BS_STREAM = ML_BATCH_N * STRIDE_BS_STREAM;  // [b][l][m][d]

// Host <-> kernel I/O — PER-SEQUENCE under TT_BATCH (grown by ML_BATCH_N).
// The shipped buffer holds only ONE sequence's TOKENS/TARGETS; staging B streams
// naively would let seq1's tokens overwrite seq0's targets — so each of these
// grows to ML_BATCH_N * stride and sequence b lives at OFF_* + b*stride.
constexpr int WORDS_TOKENS     = ML_BATCH_N * STRIDE_TOKENS;     // [b][m]
constexpr int WORDS_TARGETS    = ML_BATCH_N * STRIDE_TARGETS;    // [b][m]
constexpr int WORDS_GRAD_LOGIT = ML_BATCH_N * STRIDE_GRAD_LOGIT; // [b][m][v]
constexpr int WORDS_PRED       = ML_BATCH_N * STRIDE_PRED;       // [b][m]

// Host-written config words, read by do_fwd_loss (mirrors forge_buffer.hpp's
// OFF_CFG convention exactly):
//   buffer[OFF_CFG + 0] = CFG_LOSS_KIND : 0 = 2-hot perceptron (default, the
//                         zero-init behavior); 1 = balanced LUT cross-entropy
//                         (exp2 ROM + shift-normalize; grad rows sum to zero —
//                         an unbalanced row bias collapses sign-trained models;
//                         see forge_ce_lut.hpp).
//   buffer[OFF_CFG + 1] = CFG_CE_ZSHIFT : LUT index = (zmax - logit) >> this.
// NOTE: LUT-CE gradients are Q16 (<= 65536), not +/-1; pass a proportionally
// larger threshold to do_signsgd (mode 2, arg2).
constexpr int WORDS_CFG        = 2;
// exp2 LUT for the balanced CE gradient, STAGED BY THE HOST into the buffer
// (256 words = FT_CE_EXP2_Q16). The kernel reads the table from here instead of
// a `static const` ROM: on the F2 AFI the synthesized ROM's init contents did
// NOT survive (CE grads read all-zero on silicon — see fpga/silicon/
// SILICON_FINDINGS.md); host-staging the table guarantees it is present.
constexpr int WORDS_CE_LUT     = 256;

// =============================================================================
// Cumulative offsets (word indices into the flat int32 buffer).
// Order: weights -> weight grads -> saved indices -> activation stream ->
// host I/O -> CFG (last), matching forge_buffer.hpp's grouping.
// =============================================================================
constexpr int OFF_EMBED      = 0;
constexpr int OFF_MH         = OFF_EMBED      + WORDS_EMBED;
constexpr int OFF_MP         = OFF_MH         + WORDS_MH;
constexpr int OFF_MB         = OFF_MP         + WORDS_MP;
constexpr int OFF_W_HEAD     = OFF_MB         + WORDS_MB;

constexpr int OFF_G_EMBED    = OFF_W_HEAD     + WORDS_W_HEAD;
constexpr int OFF_G_MH       = OFF_G_EMBED    + WORDS_G_EMBED;
constexpr int OFF_G_MP       = OFF_G_MH       + WORDS_G_MH;
constexpr int OFF_G_MB       = OFF_G_MP       + WORDS_G_MP;
constexpr int OFF_G_W_HEAD   = OFF_G_MB       + WORDS_G_MB;

constexpr int OFF_TAP_IDX    = OFF_G_W_HEAD   + WORDS_G_W_HEAD;
constexpr int OFF_PROJ_IDX   = OFF_TAP_IDX    + WORDS_TAP_IDX;
constexpr int OFF_AD_HEAD    = OFF_PROJ_IDX   + WORDS_PROJ_IDX;

constexpr int OFF_BS_STREAM  = OFF_AD_HEAD    + WORDS_AD_HEAD;

constexpr int OFF_TOKENS     = OFF_BS_STREAM  + WORDS_BS_STREAM;
constexpr int OFF_TARGETS    = OFF_TOKENS     + WORDS_TOKENS;
constexpr int OFF_GRAD_LOGIT = OFF_TARGETS    + WORDS_TARGETS;
constexpr int OFF_PRED       = OFF_GRAD_LOGIT + WORDS_GRAD_LOGIT;

constexpr int OFF_CFG        = OFF_PRED       + WORDS_PRED;
// OFF_CE_LUT MUST be 16-word aligned: the kernel reads the table with the wide
// (512-bit = 16x int32) burst_load, whose offset is (word_off >> 4) — a
// non-aligned base silently reads the table shifted (it read 2 words early when
// it sat at OFF_CFG+2, corrupting the CE gradient on silicon while the host
// scalar path stayed correct). Round up past the 2-word CFG to the next 16.
constexpr int OFF_CE_LUT     = ((OFF_CFG + WORDS_CFG + 15) / 16) * 16;  // host-staged exp2 table

constexpr int ML_BUFFER_WORDS = OFF_CE_LUT    + WORDS_CE_LUT;
constexpr int ML_BUFFER_BYTES = ML_BUFFER_WORDS * 4;
#if ML_D >= 512
// D>=512 staging window: two 4 MB axi_bram_ctrl banks = 8 MB total (see
// cl_forge_morph_lm_d512.sv). The G-culled 7.69 MB layout MUST fit it.
// (#if'd so the D=128 sentinel's preprocessed source stays byte-identical.)
static_assert(ML_BUFFER_BYTES <= (8 << 20),
              "D>=512 buffer must fit the 8 MB (two-bank) staging window");
#endif

// =============================================================================
// 16-word (512-bit AXI burst) alignment guard — FAILS THE BUILD, not the bitstream.
// burst_load_words/burst_store_words truncate the byte offset to (word_off >> 4)
// on the SYNTHESIS wide_t path, so any burst-reached base that is NOT a multiple
// of 16 silently reads/writes SHIFTED on silicon while the host scalar (#else)
// path stays correct — so the bit-exact CSIM golden does NOT catch it. This is
// exactly bug (a): OFF_CE_LUT once sat at OFF_CFG+2 and read the exp2 LUT 2 words
// early on the F2 AFI (see fpga/silicon/SILICON_FINDINGS.md). These asserts would
// have caught it. CFG and PRED are scalar-only (buffer[OFF_CFG+i], buffer[OFF_PRED+m])
// and are intentionally NOT required to align (WORDS_CFG=2 is deliberately odd).
static_assert(ML_D     % 16 == 0, "ML_D must be a multiple of 16: per-row bursts are D words");
static_assert(ML_V     % 16 == 0, "ML_V must be a multiple of 16: head/grad_logit rows are V words");
static_assert(ML_L_SEQ % 16 == 0, "ML_L_SEQ must be a multiple of 16: TOKENS/TARGETS bursts are L words");
#define ML_ALIGN16(off, words) \
    static_assert((off) % 16 == 0, #off " must be 16-word aligned for the 512-bit burst"); \
    static_assert((words) % 16 == 0, #words " must be a multiple of 16 (whole-region/row burst)")
ML_ALIGN16(OFF_EMBED,    WORDS_EMBED);
ML_ALIGN16(OFF_MH,       WORDS_MH);
ML_ALIGN16(OFF_MP,       WORDS_MP);
ML_ALIGN16(OFF_MB,       WORDS_MB);
ML_ALIGN16(OFF_W_HEAD,   WORDS_W_HEAD);
ML_ALIGN16(OFF_G_EMBED,  WORDS_G_EMBED);
ML_ALIGN16(OFF_G_MH,     WORDS_G_MH);
ML_ALIGN16(OFF_G_MP,     WORDS_G_MP);
ML_ALIGN16(OFF_G_MB,     WORDS_G_MB);
ML_ALIGN16(OFF_G_W_HEAD, WORDS_G_W_HEAD);
ML_ALIGN16(OFF_TAP_IDX,  WORDS_TAP_IDX);
ML_ALIGN16(OFF_PROJ_IDX, WORDS_PROJ_IDX);
ML_ALIGN16(OFF_AD_HEAD,  WORDS_AD_HEAD);
ML_ALIGN16(OFF_BS_STREAM, WORDS_BS_STREAM);
ML_ALIGN16(OFF_TOKENS,   WORDS_TOKENS);
ML_ALIGN16(OFF_TARGETS,  WORDS_TARGETS);
ML_ALIGN16(OFF_GRAD_LOGIT, WORDS_GRAD_LOGIT);
ML_ALIGN16(OFF_CE_LUT,   WORDS_CE_LUT);   // bug-(a) region: aligned base REQUIRED
#undef ML_ALIGN16

// -----------------------------------------------------------------------------
// Per-layer / per-position slice helpers — GLOBAL (buffer) word offsets.
//
// IMPORTANT (same convention as forge_buffer_morph.hpp): off_mh/off_mp/...
// return GLOBAL addresses (they include the region's OFF_*) for burst
// load/store of a whole region. Once a region is loaded into a LOCAL array
// (mh_buf, mp_buf, g_mh, g_mp, ...) index that local array with the LOCAL
// per-layer base `l*(ML_D*ML_K)` / `l*(ML_D*ML_D)` / `l*ML_D` — NOT these
// helpers — since the local array already starts at the region base.
// -----------------------------------------------------------------------------
// morph kernel h[l][c][k]  -> OFF_MH + l*(D*K) + c*K + k
static inline int off_mh(int l)        { return OFF_MH    + l * (ML_D * ML_K); }
// out-proj P[l][j][i]      -> OFF_MP + l*(D*D) + j*D + i
static inline int off_mp(int l)        { return OFF_MP    + l * (ML_D * ML_D); }
// out-proj bias b[l][j]    -> OFF_MB + l*D + j
static inline int off_mb(int l)        { return OFF_MB    + l * ML_D; }
// embedding row for token  -> OFF_EMBED + token*D
static inline int off_embed(int tok)   { return OFF_EMBED + tok * ML_D; }
static inline int off_g_embed(int tok) { return OFF_G_EMBED + tok * ML_D; }
// saved indices, position-major [l][m][.]
static inline int off_tap(int l, int m)  { return OFF_TAP_IDX  + l * (ML_L_SEQ * ML_D) + m * ML_D; }
static inline int off_proj(int l, int m) { return OFF_PROJ_IDX + l * (ML_L_SEQ * ML_D) + m * ML_D; }
static inline int off_head(int m)        { return OFF_AD_HEAD   + m * ML_V; }
// post-norm / grad_BS stream slice BS_l[m] -> OFF_BS_STREAM + l*(L_SEQ*D) + m*D
static inline int off_bs(int l, int m)   { return OFF_BS_STREAM + l * (ML_L_SEQ * ML_D) + m * ML_D; }
// per-position logit grad row
static inline int off_grad_logit(int m)  { return OFF_GRAD_LOGIT + m * ML_V; }

// -----------------------------------------------------------------------------
// PER-SEQUENCE (batch-b) offset helpers. Sequence b's per-seq region begins at
// OFF_<region> + b * STRIDE_<region>; within it the layout is IDENTICAL to the
// single-sequence layout above (so passing b=0 reproduces the non-batch helper
// exactly, and the per-sequence arithmetic in do_fwd_loss/do_bwd is unchanged
// modulo the constant b-stride => bit-exact to running that sequence alone).
// -----------------------------------------------------------------------------
static inline int off_tap_b(int b, int l, int m)  { return OFF_TAP_IDX   + b * STRIDE_TAP_IDX   + l * (ML_L_SEQ * ML_D) + m * ML_D; }
static inline int off_proj_b(int b, int l, int m) { return OFF_PROJ_IDX  + b * STRIDE_PROJ_IDX  + l * (ML_L_SEQ * ML_D) + m * ML_D; }
static inline int off_head_b(int b, int m)        { return OFF_AD_HEAD   + b * STRIDE_AD_HEAD   + m * ML_V; }
static inline int off_bs_b(int b, int l, int m)   { return OFF_BS_STREAM + b * STRIDE_BS_STREAM + l * (ML_L_SEQ * ML_D) + m * ML_D; }
static inline int off_grad_logit_b(int b, int m)  { return OFF_GRAD_LOGIT + b * STRIDE_GRAD_LOGIT + m * ML_V; }
static inline int off_tokens_b(int b)             { return OFF_TOKENS  + b * STRIDE_TOKENS; }
static inline int off_targets_b(int b)            { return OFF_TARGETS + b * STRIDE_TARGETS; }
static inline int off_pred_b(int b)               { return OFF_PRED    + b * STRIDE_PRED; }

// dilation for layer l (capped power-of-two). Compile/host-time helper; the
// kernel multiplies the small loop-counter k by this (address arithmetic on
// counters, NOT a data-path multiply).
static inline int dilation_of_layer(int l) {
    int d = 1;
    for (int i = 0; i < l; i++) {
        d <<= 1;
        if (d >= ML_DIL_CAP) { d = ML_DIL_CAP; break; }
    }
    return d;
}

#endif  // FORGE_BUFFER_MORPH_LM_HPP
