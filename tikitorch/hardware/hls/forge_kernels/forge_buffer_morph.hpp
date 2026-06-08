// forge_buffer_morph.hpp — shared flat int32 buffer layout for the
// MorphSSM-core training kernel (forge_train_step_morph).
//
// Mirrors forge_buffer.hpp's style exactly: a single flat int32 buffer in
// DDR with constexpr per-tensor word counts (WORDS_*) and cumulative byte
// (word) offsets (OFF_*). One monolithic kernel reads only the regions it
// needs and writes only the regions it produces, dispatched by `mode`.
//
// HYBRID design: the morphological CORE runs on-chip (this kernel); the
// host owns embed + classification head + cross-entropy. The host writes
// the already-embedded input sequence X[L_SEQ][D] into the buffer, the
// kernel produces the mean-pooled feature POOLED[D], the host computes the
// head/softmax/CE and writes GRAD_POOLED[D] back, then the kernel
// backpropagates through the core.
//
// Per-layer core block (alternating by parity):
//   even layer l -> DILATION  (max-plus) depthwise morph conv
//   odd  layer l -> EROSION   (min-plus) depthwise morph conv
//   dilation d_l = min(2^l, MD_DIL_CAP)
//
// Forward per layer:
//   R  -> batch_shift(over D, per t)  = BS
//      -> depthwise morph conv (causal, dilation d_l, per-channel h[D][K]) = MC
//      -> tropical projection  y[t][j] = max_i(P[j][i] + MC[t][i])         = PROJ
//      -> residual add  Rout = sat(R + PROJ)
//   After L_LAYERS: POOLED[j] = (1/L_SEQ) * sum_t Rout[t][j]   (>> LOG2_L)
//
// Saved for backward (sparse-scatter argmax routing, exactly like
// ad_dil_row / ak_ero_row / an_out_row in forge_train_step.cpp):
//   TAP_IDX [l][D][L_SEQ]  — argmax/argmin tap index k* per (layer,c,t)
//   PROJ_IDX[l][L_SEQ][D]  — argmax input index i* per (layer,t,j)

#ifndef FORGE_BUFFER_MORPH_HPP
#define FORGE_BUFFER_MORPH_HPP

#include "../forge_hls_types.hpp"
#include <cstdint>

// =============================================================================
// Shape constants — small classification config (fast C-sim).
// Scale D / L_LAYERS later for the ECG config (D=128, L_LAYERS=8).
// =============================================================================
constexpr int MD_D          = 64;   // model / channel dimension
constexpr int MD_L_LAYERS   = 4;    // number of morph SSM layers
constexpr int MD_K          = 4;    // morph conv kernel taps
constexpr int MD_L_SEQ      = 64;   // sequence length
constexpr int MD_DIL_CAP    = 16;   // cap on per-layer dilation d_l = min(2^l, cap)

// log2 helpers for the mean-subtract (>>) normalization. D and L_SEQ are
// powers of two so the divide-by-N collapses to an arithmetic shift — no
// multiplications / no dividers on the data path.
constexpr int MD_LOG2_D     = 6;    // log2(MD_D)     -> 64
constexpr int MD_LOG2_L     = 6;    // log2(MD_L_SEQ) -> 64

// Mode codes (match forge_train_step.cpp dispatcher style).
constexpr int MD_MODE_FWD     = 0;
constexpr int MD_MODE_BWD     = 1;
constexpr int MD_MODE_SIGNSGD = 2;

// =============================================================================
// Per-tensor word counts
// =============================================================================
// Weights (per layer):
constexpr int WORDS_MH       = MD_L_LAYERS * MD_D * MD_K;    // morph conv kernels
constexpr int WORDS_MP       = MD_L_LAYERS * MD_D * MD_D;    // tropical proj matrices

// Gradient accumulators (per layer):
constexpr int WORDS_G_MH     = WORDS_MH;
constexpr int WORDS_G_MP     = WORDS_MP;

// Saved argmax index arrays (per layer):
constexpr int WORDS_TAP_IDX  = MD_L_LAYERS * MD_D * MD_L_SEQ;   // [l][c][t]
constexpr int WORDS_PROJ_IDX = MD_L_LAYERS * MD_L_SEQ * MD_D;   // [l][t][j]

// Residual-stream activations needed for backward: the residual-stream input
// R entering EACH layer (R[l] for l in 0..L_LAYERS-1) plus the final output
// Rout[L_LAYERS-1]. Stored as one contiguous (L_LAYERS+1) x L_SEQ x D region;
// slot l is the input to layer l, slot L_LAYERS is the final pooled-from
// activation. Forward writes slots 0..L_LAYERS; backward only needs the
// argmax indices + the incoming grad, but we keep the residual stream so the
// CPU reference and the kernel agree on the exact activations to scatter into.
constexpr int WORDS_RSTREAM  = (MD_L_LAYERS + 1) * MD_L_SEQ * MD_D;

// Host <-> kernel I/O:
constexpr int WORDS_X           = MD_L_SEQ * MD_D;   // embedded input sequence
constexpr int WORDS_POOLED      = MD_D;              // mean-pooled feature out
constexpr int WORDS_GRAD_POOLED = MD_D;              // upstream grad wrt POOLED

// =============================================================================
// Cumulative offsets (word indices into the flat int32 buffer)
// =============================================================================
constexpr int OFF_MH          = 0;
constexpr int OFF_MP          = OFF_MH          + WORDS_MH;

constexpr int OFF_G_MH        = OFF_MP          + WORDS_MP;
constexpr int OFF_G_MP        = OFF_G_MH        + WORDS_G_MH;

constexpr int OFF_TAP_IDX     = OFF_G_MP        + WORDS_G_MP;
constexpr int OFF_PROJ_IDX    = OFF_TAP_IDX     + WORDS_TAP_IDX;

constexpr int OFF_RSTREAM     = OFF_PROJ_IDX    + WORDS_PROJ_IDX;

constexpr int OFF_X           = OFF_RSTREAM     + WORDS_RSTREAM;
constexpr int OFF_POOLED      = OFF_X           + WORDS_X;
constexpr int OFF_GRAD_POOLED = OFF_POOLED      + WORDS_POOLED;

constexpr int MD_BUFFER_WORDS = OFF_GRAD_POOLED + WORDS_GRAD_POOLED;
constexpr int MD_BUFFER_BYTES = MD_BUFFER_WORDS * 4;

// -----------------------------------------------------------------------------
// Per-layer slice helpers — DDR (buffer) word offsets into each region.
//
// IMPORTANT: off_mh/off_mp/off_g_mh/off_g_mp return GLOBAL buffer addresses
// (they include the region's OFF_*). They are for burst_load/store of a whole
// region. Once a region is loaded into a LOCAL array (e.g. mh_buf, mp_buf,
// g_mh, g_mp), index that local array with the LOCAL per-layer base
// `l*(MD_D*MD_K)` / `l*(MD_D*MD_D)` — NOT these helpers — since the local
// array already starts at the region base.
// -----------------------------------------------------------------------------
// morph kernel h[l][c][k]  -> OFF_MH + l*(D*K) + c*K + k
static inline int off_mh(int l)        { return OFF_MH    + l * (MD_D * MD_K); }
static inline int off_mp(int l)        { return OFF_MP    + l * (MD_D * MD_D); }
static inline int off_g_mh(int l)      { return OFF_G_MH  + l * (MD_D * MD_K); }
static inline int off_g_mp(int l)      { return OFF_G_MP  + l * (MD_D * MD_D); }
// tap_idx[l][c][t]  -> OFF_TAP_IDX + l*(D*L_SEQ) + c*L_SEQ + t
static inline int off_tap(int l)       { return OFF_TAP_IDX  + l * (MD_D * MD_L_SEQ); }
// proj_idx[l][t][j] -> OFF_PROJ_IDX + l*(L_SEQ*D) + t*D + j
static inline int off_proj(int l)      { return OFF_PROJ_IDX + l * (MD_L_SEQ * MD_D); }
// residual stream slot s (s in 0..L_LAYERS) -> [s][t][j]
static inline int off_rstream(int s)   { return OFF_RSTREAM  + s * (MD_L_SEQ * MD_D); }

// dilation for layer l (capped power-of-two). Compile/host-time helper; the
// kernel computes this with shifts (no multiply) inside the forward loop.
static inline int dilation_of_layer(int l) {
    int d = 1;
    for (int i = 0; i < l; i++) {
        d <<= 1;
        if (d >= MD_DIL_CAP) { d = MD_DIL_CAP; break; }
    }
    return d;
}

#endif  // FORGE_BUFFER_MORPH_HPP
