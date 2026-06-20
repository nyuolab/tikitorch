// forge_buffer_gpt2.hpp — DDR memory layout for the 124M GPT-2-small training rig (flagship S4).
//
// Forces the 124M memory arithmetic to be exact. 124M params >> ~34 MB on-chip (270 Mb URAM+BRAM),
// so weights + dyadic-Adam state are DDR-RESIDENT and streamed per layer (the cl_forge_muzero_ddr
// pattern). On-chip holds only a per-layer/per-tile working set. Layout is LAYER-MAJOR so each block's
// weights stream as one contiguous burst.
//
// Storage (multiplier-free training):
//   W : dyadic ±2^k weight -> a 1-byte code (1 sign bit + ~5-bit exponent + slack)  = INT8.
//   m : dyadic-Adam 1st moment EMA  = INT8 (log/linear; the silicon optimizer keeps int state).
//   v : dyadic-Adam 2nd-moment (|g| EMA)  = INT8.
//   => 3 bytes / param resident. (grad is TILED output-stationary and flushed, not full-resident;
//      activation checkpoints are per-layer inputs only, recompute the rest in the backward.)
//
// This header is config + arithmetic (constexpr). The per-tensor counts below sum to ~124M (GPT-2-124M
// with tied token-embedding / LM-head). The kernels (forge_fwd_gemm / forge_wgrad / forge_dyadic_adam /
// forge_attn) operate on one tensor tile at a time, addressed by OFF_<region> + layer*stride.

#pragma once
#include <cstdint>

// ----- GPT-2-small (124M) config -----
constexpr int G_D    = 768;        // d_model
constexpr int G_L    = 12;         // layers
constexpr int G_H    = 12;         // heads
constexpr int G_DH   = G_D / G_H;  // 64 head dim
constexpr int G_FFN  = 4 * G_D;    // 3072
constexpr int G_V    = 50257;      // BPE vocab
constexpr int G_SEQ  = 1024;       // context

// ----- per-layer weight param counts -----
constexpr long P_QKV    = 3L * G_D * G_D;     // Wq,Wk,Wv          1,769,472
constexpr long P_O      = (long)G_D * G_D;    // Wo                  589,824
constexpr long P_FFN    = 2L * G_D * G_FFN;   // up + down         4,718,592
constexpr long P_LN     = 4L * G_D;           // 2 norms (gamma+beta) — tiny
constexpr long P_LAYER  = P_QKV + P_O + P_FFN + P_LN;          // ~7,080,960
constexpr long P_LAYERS = (long)G_L * P_LAYER;                 // ~84,971,520

// ----- embedding / head (tied) + final norm -----
constexpr long P_EMBED  = (long)G_V * G_D;    // token embedding = LM head (tied)  38,597,376
constexpr long P_POS    = (long)G_SEQ * G_D;  // learned positional                   786,432
constexpr long P_LNF    = 2L * G_D;

constexpr long G_PARAMS = P_LAYERS + P_EMBED + P_POS + P_LNF;  // ~124,355,328
static_assert(G_PARAMS > 124000000L && G_PARAMS < 125000000L, "GPT-2-124M param count");

// ----- DDR footprint (bytes) -----
constexpr long BYTES_W   = G_PARAMS * 1;      // int8 dyadic weights        ~124.4 MB
constexpr long BYTES_M   = G_PARAMS * 1;      // dyadic-Adam m              ~124.4 MB
constexpr long BYTES_V   = G_PARAMS * 1;      // dyadic-Adam v              ~124.4 MB
constexpr long BYTES_OPT = BYTES_W + BYTES_M + BYTES_V;        // ~373 MB W+m+v resident

// activation checkpoints: per-layer block INPUT only (recompute within-block on bwd). int16.
constexpr long BYTES_ACT_CKPT = (long)G_L * G_SEQ * G_D * 2;   // ~18.9 MB @ batch 1
// grad: TILED (output-stationary), a single matrix-tile working region, NOT full-resident.
constexpr long BYTES_GRAD_TILE = 64L * G_FFN * 4;             // one OT=64 x FFN row-tile, int32 ~0.79 MB
// host I/O
constexpr long BYTES_TOKENS  = (long)G_SEQ * 4;
constexpr long BYTES_CE_LUT  = 4096L * 4;                      // host-staged exp2 table (16-word aligned)

constexpr long BYTES_TOTAL = BYTES_OPT + BYTES_ACT_CKPT + BYTES_GRAD_TILE + BYTES_TOKENS + BYTES_CE_LUT;
// ~392 MB resident on the F2 DDR4 (channels have GBs) — fits comfortably; the constraint is BANDWIDTH/step.

// ----- per-step DDR traffic (the throughput-relevant arithmetic) -----
// fwd: read W (124).  bwd: read W again (124) + read act-ckpt.  opt: read (W,m,v)=373 + write (W,m,v)=373.
// Weight+optimizer traffic per step ~= 124 + 124 + 746 = ~994 MB  (batch-independent; amortized over a
// real microbatch). At even ~16 GB/s 1-channel DDR this is ~62 ms/step of pure DDR -> >=16k tok/s at
// batch>=1k; so the rig is COMPUTE-bound (the ~0.74 Gadd/token), NOT bandwidth-bound (10-90x headroom).
constexpr long BYTES_PER_STEP = BYTES_W + BYTES_W + BYTES_OPT * 2;   // ~994 MB

// ----- region offsets (BYTE offsets into the DDR base; layer-major within W/m/v) -----
// W block per layer: [QKV | O | FFN | LN] at OFF_W + l*P_LAYER (bytes, int8). m,v mirror W.
constexpr long OFF_W       = 0;
constexpr long OFF_M       = OFF_W + BYTES_W;
constexpr long OFF_V       = OFF_M + BYTES_M;
constexpr long OFF_ACTCK   = OFF_V + BYTES_V;
constexpr long OFF_GRADT   = OFF_ACTCK + BYTES_ACT_CKPT;
constexpr long OFF_TOKENS  = OFF_GRADT + BYTES_GRAD_TILE;
constexpr long OFF_CE_LUT  = ((OFF_TOKENS + BYTES_TOKENS + 63) / 64) * 64;   // 16-word(64B) aligned (the
                                                                            // morph OFF_CE_LUT bug-fix)
// per-layer weight sub-offsets (bytes from a layer's W base)
constexpr long LW_QKV = 0, LW_O = P_QKV, LW_FFN = P_QKV + P_O, LW_LN = P_QKV + P_O + P_FFN;
constexpr long layer_W(int l) { return OFF_W + (long)l * P_LAYER; }
constexpr long layer_M(int l) { return OFF_M + (long)l * P_LAYER; }
constexpr long layer_V(int l) { return OFF_V + (long)l * P_LAYER; }
