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
// ZERO multiplies (mean over D=128 is >>7; index arithmetic c*K etc. is small
// loop-counter address math, not a data-path multiply).
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
// Shape constants — "config B" MorphSSM char-LM at D=V=L_SEQ=128, N_LAYERS=4.
// All sizes are powers of two (or multiples of 16) so every WORDS_* divides 16
// — required for the 512-bit (16-word) wide-AXI burst helpers — and the
// divide-by-N normalizations collapse to arithmetic shifts (no divider, no
// multiply on the data path).
// =============================================================================
constexpr int ML_D        = 128;   // model / channel dimension
constexpr int ML_V        = 128;   // vocabulary (ASCII chars, clipped to 128)
constexpr int ML_L_SEQ    = 128;   // sequence length / training window
constexpr int ML_N_LAYERS = 4;     // number of morph SSM layers
constexpr int ML_K        = 7;     // morph conv kernel taps
constexpr int ML_DIL_CAP  = 64;    // cap on per-layer dilation d_l = min(2^l, cap)

// log2 helpers for the mean-subtract (>>) normalization. D is a power of two so
// the per-token divide-by-D collapses to an arithmetic shift.
constexpr int ML_LOG2_D   = 7;     // log2(ML_D) -> 128

// Mode codes (match forge_train_step.cpp dispatcher style).
constexpr int ML_MODE_FWD     = 0;
constexpr int ML_MODE_BWD     = 1;
constexpr int ML_MODE_SIGNSGD = 2;

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
constexpr int WORDS_G_EMBED  = WORDS_EMBED;
constexpr int WORDS_G_MH     = WORDS_MH;
constexpr int WORDS_G_MP     = WORDS_MP;
constexpr int WORDS_G_MB     = WORDS_MB;
constexpr int WORDS_G_W_HEAD = WORDS_W_HEAD;

// Saved argmax index arrays — POSITION-MAJOR ([l][m][.]) so each per-position
// row is D (or V) contiguous words, matching the position-by-position kernel.
constexpr int WORDS_TAP_IDX  = ML_N_LAYERS * ML_L_SEQ * ML_D;   // [l][m][c]
constexpr int WORDS_PROJ_IDX = ML_N_LAYERS * ML_L_SEQ * ML_D;   // [l][m][j]
constexpr int WORDS_AD_HEAD  = ML_L_SEQ * ML_V;                 // [m][v]

// Per-layer post-norm activation stream BS_l[m][d]. DOUBLES as the backward
// grad_BS accumulator (host zeroes it between the fwd and bwd loops — exactly
// like forge_train_step.cpp reuses K_CACHE/V_CACHE). Forward writes BS_l[m] and
// reads BS_l[m-k*dil] (the rolling causal-conv cache); backward scatters
// grad_MC_l[m] into grad_BS_l[m-k*dil] and consumes grad_BS_l[m] in the
// batch_shift backward.
constexpr int WORDS_BS_STREAM = ML_N_LAYERS * ML_L_SEQ * ML_D;  // [l][m][d]

// Host <-> kernel I/O:
constexpr int WORDS_TOKENS     = ML_L_SEQ;            // input token id per position
constexpr int WORDS_TARGETS    = ML_L_SEQ;            // target token id per position
constexpr int WORDS_GRAD_LOGIT = ML_L_SEQ * ML_V;     // loss grad wrt logits [m][v]
constexpr int WORDS_PRED       = ML_L_SEQ;            // argmax-predicted token per position

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

constexpr int ML_BUFFER_WORDS = OFF_CFG       + WORDS_CFG;
constexpr int ML_BUFFER_BYTES = ML_BUFFER_WORDS * 4;

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
