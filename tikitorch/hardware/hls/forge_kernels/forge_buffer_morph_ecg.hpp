// forge_buffer_morph_ecg.hpp — flat int32 buffer layout for the LARGE
// (PTB-XL ECG) MorphSSM-core training kernel (forge_train_step_morph_ecg).
//
// Same monolithic-single-kernel / flat-DDR-buffer philosophy as
// forge_buffer_morph.hpp and forge_buffer.hpp, but redesigned so the kernel
// SCALES to L_SEQ=5000 without exceeding on-chip URAM+BRAM (~33 MB on the
// VU47P).  The blocker in the small kernel is that it held per-(layer,t)
// saved argmax indices AND the residual-stream activations ON-CHIP.  At
// L_SEQ=5000 each of TAP_IDX / PROJ_IDX is L_LAYERS*L_SEQ*D ~= 5.12M int32
// ~= 20 MB — far over budget.
//
// Redesign:
//   * The BIG arrays — X[L_SEQ][D], saved TAP_IDX[l][t][c], saved
//     PROJ_IDX[l][t][j], POOLED[D], GRAD_POOLED[D], weights MH/MP and grad
//     accumulators G_MH/G_MP — live in DDR at fixed constexpr OFF_* offsets,
//     accessed by burst_load_words / burst_store_words exactly like the
//     per-token DDR-row stores in forge_train_step.cpp.
//   * Forward STREAMS over t with only an O(L_LAYERS * D * WPOW) on-chip
//     sliding window of batch-shifted inputs per layer (so the dilated conv
//     can read x[c][t-k*dil]).  No O(L_SEQ*D) on-chip activation arrays.
//   * Backward STREAMS in reverse t with a symmetric O(L_LAYERS * D * WPOW)
//     on-chip grad-window per layer (the conv routes a grad at output (c,t)
//     to input (c, t-k*dil) — a FUTURE timestep in reverse order).
//   * NOTE: the residual-stream activations are NOT stored — the morph core
//     is piecewise-linear (max-plus), so the backward gradient routes purely
//     through the saved argmax INDICES with coefficient 1; the activation
//     VALUES are never needed for g_MH / g_MP (the small kernel's do_bwd
//     likewise reads only tap_idx / proj_idx, never the rstream region).
//
// HYBRID design (unchanged): host owns embed + classification head + CE;
// the kernel owns the morph core fwd/bwd/update.  Host writes X[L_SEQ][D],
// kernel produces POOLED[D], host computes head/softmax/CE and writes
// GRAD_POOLED[D] back, kernel backpropagates into G_MH / G_MP.
//
// Per-layer core block (alternating by parity), identical math to the small
// kernel:
//   even layer l -> DILATION (max-plus) depthwise morph conv
//   odd  layer l -> EROSION  (min-plus) depthwise morph conv
//   dilation d_l = min(2^l, MDE_DIL_CAP)
//
//   R  -> batch_shift(over D, per t)                         = BS
//      -> depthwise morph conv (causal, dilation d_l)        = MC  (save k*)
//      -> tropical projection y[j]=max_i(P[j][i]+MC[i])      = PJ  (save i*)
//      -> residual add Rout = sat(R + PJ)
//   After L_LAYERS: POOLED[j] = (sum_t Rout[t][j]) >> LOG2_L

#ifndef FORGE_BUFFER_MORPH_ECG_HPP
#define FORGE_BUFFER_MORPH_ECG_HPP

#include "../forge_hls_types.hpp"
#include <cstdint>

// =============================================================================
// Shape constants.
//
// SHIPPED at the PTB-XL ECG target (manuscript dilation cap 2^7 = 128):
//   D=128, L_LAYERS=8, K=7, L_SEQ=5000, DIL_CAP=128.
// The TB overrides these (via -D macros) to a small L_SEQ for fast bit-exact
// C-sim; nothing else in the kernel hard-codes a shape.
// =============================================================================
// Config is set via -D macros (so the TB can override to a small L_SEQ for a
// fast bit-exact C-sim). Defaults below are the SHIPPED ECG target. The macro
// names carry a _CFG suffix so the clean constexpr identifiers used everywhere
// else are not themselves macro-expanded.
#ifndef MDE_D_CFG
#define MDE_D_CFG 128
#endif
#ifndef MDE_L_LAYERS_CFG
#define MDE_L_LAYERS_CFG 8
#endif
#ifndef MDE_K_CFG
#define MDE_K_CFG 7
#endif
#ifndef MDE_L_SEQ_CFG
#define MDE_L_SEQ_CFG 5000
#endif
// DIL_CAP=32 (was 128): caps the on-chip sliding-window/grad rings at WPOW=256
// so all URAM fits in a single SLR (~136 URAM < 320/SLR), eliminating the
// cross-SLR URAM->FF path that missed timing by -0.777 ns at DIL_CAP=128.
// Receptive field over 8 layers is still ~760 samples (~1.5 s of 500 Hz ECG).
#ifndef MDE_DIL_CAP_CFG
#define MDE_DIL_CAP_CFG 32
#endif

constexpr int MDE_D        = MDE_D_CFG;
constexpr int MDE_L_LAYERS = MDE_L_LAYERS_CFG;
constexpr int MDE_K        = MDE_K_CFG;
constexpr int MDE_L_SEQ    = MDE_L_SEQ_CFG;
constexpr int MDE_DIL_CAP  = MDE_DIL_CAP_CFG;

// log2 of D used by the per-token batch_shift mean-subtract (>>). D MUST be a
// power of two so the divide-by-D collapses to an arithmetic shift (no
// divider, no multiply on the data path).
constexpr int mde_log2_ceil(int n) { return (n <= 1) ? 0 : 1 + mde_log2_ceil((n + 1) / 2); }
constexpr int mde_is_pow2(int n)    { return (n > 0) && ((n & (n - 1)) == 0); }
static_assert(mde_is_pow2(MDE_D), "MDE_D must be a power of two (batch_shift uses >> log2(D))");
constexpr int MDE_LOG2_D = mde_log2_ceil(MDE_D);

// L_SEQ need NOT be a power of two for the mean-pool: we accumulate the exact
// integer sum over t and shift right by LOG2_L = floor(log2(L_SEQ)). This is
// the same >> normalization the small kernel uses; it is an approximate
// (power-of-two) mean and is reproduced bit-exactly by the CPU reference, so
// correctness is defined by self-consistency with the reference, not by the
// true 1/L_SEQ divide.
constexpr int mde_log2_floor(int n) { return (n <= 1) ? 0 : 1 + mde_log2_floor(n / 2); }
constexpr int MDE_LOG2_L = mde_log2_floor(MDE_L_SEQ);

// Sliding-window capacity. The dilated causal conv at time t reads taps at
// t - k*dil for k in [0,K-1], dil <= DIL_CAP, so the live span is
// (K-1)*DIL_CAP + 1 timesteps. We round this UP to a power of two (WPOW) so
// the ring-buffer slot index is a pure bitmask (t & WMASK) — no modulo /
// divide on the address path.
constexpr int MDE_WSPAN = (MDE_K - 1) * MDE_DIL_CAP + 1;
constexpr int mde_next_pow2(int n) {
    // constexpr-friendly: smallest power of two >= n.
    return (n <= 1) ? 1 : (mde_next_pow2((n + 1) / 2) * 2);
}
constexpr int MDE_WPOW  = mde_next_pow2(MDE_WSPAN);
constexpr int MDE_WMASK = MDE_WPOW - 1;

// Mode codes (match forge_train_step.cpp dispatcher style).
constexpr int MDE_MODE_FWD     = 0;
constexpr int MDE_MODE_BWD     = 1;
constexpr int MDE_MODE_SIGNSGD = 2;

// =============================================================================
// Per-tensor word counts (flat int32 buffer)
// =============================================================================
// Weights (per layer):
constexpr int WORDS_MHE = MDE_L_LAYERS * MDE_D * MDE_K;   // morph conv kernels [l][c][k]
constexpr int WORDS_MPE = MDE_L_LAYERS * MDE_D * MDE_D;   // tropical proj      [l][j][i]

// Gradient accumulators (per layer):
constexpr int WORDS_G_MHE = WORDS_MHE;
constexpr int WORDS_G_MPE = WORDS_MPE;

// Saved argmax index arrays — BIG, live in DDR. Stored t-major so each
// timestep's row is a contiguous burst (matches the per-token DDR-row store
// pattern of forge_train_step.cpp).
//   TAP_IDX [l][t][c]  -> OFF_TAP_IDX_E + l*(L_SEQ*D) + t*D + c
//   PROJ_IDX[l][t][j]  -> OFF_PROJ_IDX_E + l*(L_SEQ*D) + t*D + j
constexpr int WORDS_TAP_IDX_E  = MDE_L_LAYERS * MDE_L_SEQ * MDE_D;
constexpr int WORDS_PROJ_IDX_E = MDE_L_LAYERS * MDE_L_SEQ * MDE_D;

// Host <-> kernel I/O — BIG X lives in DDR, streamed one row at a time.
constexpr int WORDS_XE           = MDE_L_SEQ * MDE_D;   // embedded input sequence
constexpr int WORDS_POOLED_E     = MDE_D;               // mean-pooled feature out
constexpr int WORDS_GRAD_POOLED_E = MDE_D;              // upstream grad wrt POOLED
// grad wrt the EMBEDDED INPUT X[t][d] (= layer-0 input gradient gR after the
// last bwd layer closes for timestep t). Lets the host train the float input
// embedding exactly instead of with a surrogate. Stored t-major (D-contiguous
// per timestep, same row convention as TAP_IDX / X), so each per-t store is a
// clean 16-word-aligned burst (requires D % 16 == 0, already required by the
// per-(layer,t) tap/proj row stores).
constexpr int WORDS_GRAD_X_E     = MDE_L_SEQ * MDE_D;   // grad wrt embedded input

// =============================================================================
// Cumulative word offsets into the flat int32 buffer.
// =============================================================================
constexpr int OFF_MHE          = 0;
constexpr int OFF_MPE          = OFF_MHE          + WORDS_MHE;

constexpr int OFF_G_MHE        = OFF_MPE          + WORDS_MPE;
constexpr int OFF_G_MPE        = OFF_G_MHE        + WORDS_G_MHE;

constexpr int OFF_TAP_IDX_E    = OFF_G_MPE        + WORDS_G_MPE;
constexpr int OFF_PROJ_IDX_E   = OFF_TAP_IDX_E    + WORDS_TAP_IDX_E;

constexpr int OFF_XE           = OFF_PROJ_IDX_E   + WORDS_PROJ_IDX_E;
constexpr int OFF_POOLED_E     = OFF_XE           + WORDS_XE;
constexpr int OFF_GRAD_POOLED_E = OFF_POOLED_E    + WORDS_POOLED_E;
constexpr int OFF_GRAD_X_E     = OFF_GRAD_POOLED_E + WORDS_GRAD_POOLED_E;

constexpr int FT_BUFFER_WORDS_E = OFF_GRAD_X_E    + WORDS_GRAD_X_E;
constexpr long long FT_BUFFER_BYTES = (long long)FT_BUFFER_WORDS_E * 4;

// 16-word (512-bit AXI burst) alignment guard for every region that is touched
// by burst_load_words / burst_store_words. WORDS_POOLED_E = WORDS_GRAD_POOLED_E
// = MDE_D, so OFF_GRAD_X_E inherits 16-alignment iff D % 16 == 0; all other
// region word-counts are multiples of D (and D*K / D*D), so D % 16 == 0 keeps
// every OFF_* 16-aligned. (D=128 shipped; the TB uses D=64; both % 16 == 0.)
static_assert(MDE_D % 16 == 0, "MDE_D must be a multiple of 16 for 512-bit AXI burst alignment");
static_assert(OFF_GRAD_X_E % 16 == 0, "OFF_GRAD_X_E must be 16-word aligned for burst_store_words");
static_assert(WORDS_GRAD_X_E % 16 == 0, "WORDS_GRAD_X_E must be a multiple of 16 (whole-region bursts)");

// -----------------------------------------------------------------------------
// DDR (buffer) word-offset helpers. These return GLOBAL buffer addresses
// (they include the region's OFF_*) for burst_load/store of a whole region or
// a per-(layer,t) row. Once a weight region is loaded into a LOCAL array,
// index that local array with the LOCAL per-layer base (l*D*K / l*D*D), NOT
// these helpers.
// -----------------------------------------------------------------------------
static inline int off_mhe(int l)   { return OFF_MHE   + l * (MDE_D * MDE_K); }
static inline int off_mpe(int l)   { return OFF_MPE   + l * (MDE_D * MDE_D); }
static inline int off_g_mhe(int l) { return OFF_G_MHE + l * (MDE_D * MDE_K); }
static inline int off_g_mpe(int l) { return OFF_G_MPE + l * (MDE_D * MDE_D); }
// per-(layer,t) DDR row for the saved indices (t-major, D-contiguous):
static inline int off_tap_row(int l, int t)  { return OFF_TAP_IDX_E  + l * (MDE_L_SEQ * MDE_D) + t * MDE_D; }
static inline int off_proj_row(int l, int t) { return OFF_PROJ_IDX_E + l * (MDE_L_SEQ * MDE_D) + t * MDE_D; }
// per-t DDR row of the embedded input X[t][d]:
static inline int off_x_row(int t)           { return OFF_XE + t * MDE_D; }
// per-t DDR row of grad wrt the embedded input grad_X[t][d] (t-major, same
// layout as off_x_row so the host reads it back with the X-row stride):
static inline int off_grad_x_row(int t)      { return OFF_GRAD_X_E + t * MDE_D; }

// dilation for layer l (capped power-of-two). Compile/host-time helper; the
// kernel computes the same value with shifts (no multiply) in its loops.
static inline int dilation_of_layer_e(int l) {
    int d = 1;
    for (int i = 0; i < l; i++) {
        d <<= 1;
        if (d >= MDE_DIL_CAP) { d = MDE_DIL_CAP; break; }
    }
    return d;
}

// log2 of the dilation for layer l. dil = 2^min(l, log2(DIL_CAP)), so the
// kernel can compute k*dil as (k << log2dil) — a shift, not a multiply, which
// keeps DSP usage at zero (a runtime k*dil otherwise infers 1 DSP).
static inline int dilation_log2_of_layer_e(int l) {
    int lg = 0;
    for (int i = 0; i < l; i++) {
        if ((1 << (lg + 1)) > MDE_DIL_CAP) break;
        lg++;
    }
    return lg;
}

#endif  // FORGE_BUFFER_MORPH_ECG_HPP
