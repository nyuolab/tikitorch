// forge_buffer.hpp — shared buffer layout + accessors for the per-op
// training kernels (forge_fwd_step, forge_loss_step, forge_bwd_step,
// forge_signsgd_step).
//
// All four kernels share ONE flat int32 buffer in DDR and the same byte
// offsets for each tensor. Each kernel reads only the regions it needs
// and writes only the regions it produces. Host orchestrates the per-token
// training loop by sequencing kernel launches.
//
// ABI matches v6 silicon-validated wide-AXI pattern: int32 buffer pointer
// + small int args via s_axilite, max_widen_bitwidth=512 forces a 512-bit
// AXI master that the auto-generated CL's hardcoded 512-bit nets need.

#ifndef FORGE_BUFFER_HPP
#define FORGE_BUFFER_HPP

#include "../forge_hls_types.hpp"
#include <cstdint>

// =============================================================================
// Tier 1 scaling: 1-layer tropical attn LM at D=128, V=128 (ASCII chars),
// L_MAX=128. ~2x bigger than the original D=64/V=64 toy. Used for the FPGA
// silicon training demo on character-level Shakespeare in the manuscript.
// =============================================================================
constexpr int FT_V       = 128;
constexpr int FT_D       = 128;
constexpr int FT_D_FF    = 512;
constexpr int FT_L_MAX   = 128;
constexpr int FT_LOG2_D  = 7;

// =============================================================================
// Buffer layout (int32 word counts + cumulative byte offsets)
// =============================================================================
constexpr int WORDS_EMBED      = FT_V * FT_D;
constexpr int WORDS_S_Q        = FT_L_MAX * FT_D;
constexpr int WORDS_S_K        = FT_L_MAX * FT_D;
constexpr int WORDS_W1         = FT_D_FF * FT_D;
constexpr int WORDS_B1         = FT_D_FF;
constexpr int WORDS_W2         = FT_D * FT_D_FF;
constexpr int WORDS_B2         = FT_D;
constexpr int WORDS_W_HEAD     = FT_V * FT_D;

constexpr int WORDS_AD_SCORE   = FT_L_MAX * FT_L_MAX;
constexpr int WORDS_AN_OUT     = FT_L_MAX * FT_D;
constexpr int WORDS_AD_DIL     = FT_L_MAX * FT_D_FF;
constexpr int WORDS_AK_ERO     = FT_L_MAX * FT_D;
constexpr int WORDS_AD_HEAD    = FT_L_MAX * FT_V;
constexpr int WORDS_K_CACHE    = FT_L_MAX * FT_D;
constexpr int WORDS_V_CACHE    = FT_L_MAX * FT_D;

constexpr int WORDS_TOKENS     = FT_L_MAX;
constexpr int WORDS_TARGETS    = FT_L_MAX;
constexpr int WORDS_GRAD_LOGIT = FT_L_MAX * FT_V;
// One predicted token id per position. Written by do_fwd_loss; read by
// do_train_window for in-kernel accuracy counting (a throughput
// path that amortises PCIe/host scheduling over a whole training window).
constexpr int WORDS_PRED       = FT_L_MAX;

constexpr int WORDS_G_EMBED    = WORDS_EMBED;
constexpr int WORDS_G_S_Q      = WORDS_S_Q;
constexpr int WORDS_G_S_K      = WORDS_S_K;
constexpr int WORDS_G_W1       = WORDS_W1;
constexpr int WORDS_G_B1       = WORDS_B1;
constexpr int WORDS_G_W2       = WORDS_W2;
constexpr int WORDS_G_B2       = WORDS_B2;
constexpr int WORDS_G_W_HEAD   = WORDS_W_HEAD;

constexpr int OFF_EMBED      = 0;
constexpr int OFF_S_Q        = OFF_EMBED      + WORDS_EMBED;
constexpr int OFF_S_K        = OFF_S_Q        + WORDS_S_Q;
constexpr int OFF_W1         = OFF_S_K        + WORDS_S_K;
constexpr int OFF_B1         = OFF_W1         + WORDS_W1;
constexpr int OFF_W2         = OFF_B1         + WORDS_B1;
constexpr int OFF_B2         = OFF_W2         + WORDS_W2;
constexpr int OFF_W_HEAD     = OFF_B2         + WORDS_B2;

constexpr int OFF_G_EMBED    = OFF_W_HEAD     + WORDS_W_HEAD;
constexpr int OFF_G_S_Q      = OFF_G_EMBED    + WORDS_G_EMBED;
constexpr int OFF_G_S_K      = OFF_G_S_Q      + WORDS_G_S_Q;
constexpr int OFF_G_W1       = OFF_G_S_K      + WORDS_G_S_K;
constexpr int OFF_G_B1       = OFF_G_W1       + WORDS_G_W1;
constexpr int OFF_G_W2       = OFF_G_B1       + WORDS_G_B1;
constexpr int OFF_G_B2       = OFF_G_W2       + WORDS_G_W2;
constexpr int OFF_G_W_HEAD   = OFF_G_B2       + WORDS_G_B2;

constexpr int OFF_AD_SCORE   = OFF_G_W_HEAD   + WORDS_G_W_HEAD;
constexpr int OFF_AN_OUT     = OFF_AD_SCORE   + WORDS_AD_SCORE;
constexpr int OFF_AD_DIL     = OFF_AN_OUT     + WORDS_AN_OUT;
constexpr int OFF_AK_ERO     = OFF_AD_DIL     + WORDS_AD_DIL;
constexpr int OFF_AD_HEAD    = OFF_AK_ERO     + WORDS_AK_ERO;
constexpr int OFF_K_CACHE    = OFF_AD_HEAD    + WORDS_AD_HEAD;
constexpr int OFF_V_CACHE    = OFF_K_CACHE    + WORDS_K_CACHE;
constexpr int OFF_TOKENS     = OFF_V_CACHE    + WORDS_V_CACHE;
constexpr int OFF_TARGETS    = OFF_TOKENS     + WORDS_TOKENS;
constexpr int OFF_GRAD_LOGIT = OFF_TARGETS    + WORDS_TARGETS;
constexpr int OFF_PRED       = OFF_GRAD_LOGIT + WORDS_GRAD_LOGIT;

constexpr int FT_BUFFER_WORDS = OFF_PRED      + WORDS_PRED;
constexpr int FT_BUFFER_BYTES = FT_BUFFER_WORDS * 4;

#endif  // FORGE_BUFFER_HPP
