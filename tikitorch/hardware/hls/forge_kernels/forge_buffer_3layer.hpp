// forge_buffer_3layer.hpp — buffer layout for the 3-layer extension of
// the Tier-1 tropical attention LM (Silicon Experiment 7).
//
// Same per-layer structure as forge_buffer_2layer.hpp; adds layer 2 and a
// second HIDDEN region (between layers 1 and 2).
//
// Buffer growth:
//   1-layer total = ~3.0 MB
//   2-layer total = ~4.1 MB
//   3-layer total = ~5.6 MB  (still well under VU47P's ~33 MB on-chip budget)
//
// ABI: same dispatcher as 2-layer; layer_id now takes values {0,1,2}.
//   mode=FWD_LOSS, layer_id=0, m=*, token=*, target=ignored   -> layer 0 fwd
//   mode=FWD_LOSS, layer_id=1, m=*, token=ignored, target=ig  -> layer 1 fwd
//   mode=FWD_LOSS, layer_id=2, m=*, token=ignored, target=*   -> layer 2 fwd + loss
//   mode=BWD,      layer_id=2, m=*, token=ignored             -> layer 2 bwd
//   mode=BWD,      layer_id=1, m=*, token=ignored             -> layer 1 bwd
//   mode=BWD,      layer_id=0, m=*, token=*                   -> layer 0 bwd
//   mode=SIGNSGD,  layer_id=-1 (ignored), lr=*, threshold=*   -> updates all params

#ifndef FORGE_BUFFER_3LAYER_HPP
#define FORGE_BUFFER_3LAYER_HPP

#include "../forge_hls_types.hpp"
#include <cstdint>

// Shape constants — same as 1-/2-layer build.
constexpr int FT3_V       = 128;
constexpr int FT3_D       = 128;
constexpr int FT3_D_FF    = 512;
constexpr int FT3_L_MAX   = 128;
constexpr int FT3_LOG2_D  = 7;
constexpr int FT3_N_LAYERS = 3;

// Per-layer word counts (identical to 2-layer).
constexpr int W3_EMBED      = FT3_V * FT3_D;
constexpr int W3_S_Q        = FT3_L_MAX * FT3_D;
constexpr int W3_S_K        = FT3_L_MAX * FT3_D;
constexpr int W3_W1         = FT3_D_FF * FT3_D;
constexpr int W3_B1         = FT3_D_FF;
constexpr int W3_W2         = FT3_D * FT3_D_FF;
constexpr int W3_B2         = FT3_D;
constexpr int W3_W_HEAD     = FT3_V * FT3_D;

// Scratch / activation buffers (per-layer).
constexpr int W3_AD_SCORE   = FT3_L_MAX * FT3_L_MAX;
constexpr int W3_AN_OUT     = FT3_L_MAX * FT3_D;
constexpr int W3_AD_DIL     = FT3_L_MAX * FT3_D_FF;
constexpr int W3_AK_ERO     = FT3_L_MAX * FT3_D;
constexpr int W3_AD_HEAD    = FT3_L_MAX * FT3_V;
constexpr int W3_K_CACHE    = FT3_L_MAX * FT3_D;
constexpr int W3_V_CACHE    = FT3_L_MAX * FT3_D;

constexpr int W3_TOKENS     = FT3_L_MAX;
constexpr int W3_TARGETS    = FT3_L_MAX;
constexpr int W3_GRAD_LOGIT = FT3_L_MAX * FT3_V;

// One HIDDEN region per inter-layer link (L0->L1, L1->L2).
constexpr int W3_HIDDEN     = FT3_L_MAX * FT3_D;

// =============================================================================
// Per-layer parameter offsets.  L0, L1, L2 in sequence.
// =============================================================================
constexpr int OFF3_EMBED         = 0;
constexpr int OFF3_W_HEAD        = OFF3_EMBED + W3_EMBED;

constexpr int OFF3_L0_S_Q        = OFF3_W_HEAD       + W3_W_HEAD;
constexpr int OFF3_L0_S_K        = OFF3_L0_S_Q       + W3_S_Q;
constexpr int OFF3_L0_W1         = OFF3_L0_S_K       + W3_S_K;
constexpr int OFF3_L0_B1         = OFF3_L0_W1        + W3_W1;
constexpr int OFF3_L0_W2         = OFF3_L0_B1        + W3_B1;
constexpr int OFF3_L0_B2         = OFF3_L0_W2        + W3_W2;

constexpr int OFF3_L1_S_Q        = OFF3_L0_B2        + W3_B2;
constexpr int OFF3_L1_S_K        = OFF3_L1_S_Q       + W3_S_Q;
constexpr int OFF3_L1_W1         = OFF3_L1_S_K       + W3_S_K;
constexpr int OFF3_L1_B1         = OFF3_L1_W1        + W3_W1;
constexpr int OFF3_L1_W2         = OFF3_L1_B1        + W3_B1;
constexpr int OFF3_L1_B2         = OFF3_L1_W2        + W3_W2;

constexpr int OFF3_L2_S_Q        = OFF3_L1_B2        + W3_B2;
constexpr int OFF3_L2_S_K        = OFF3_L2_S_Q       + W3_S_Q;
constexpr int OFF3_L2_W1         = OFF3_L2_S_K       + W3_S_K;
constexpr int OFF3_L2_B1         = OFF3_L2_W1        + W3_W1;
constexpr int OFF3_L2_W2         = OFF3_L2_B1        + W3_B1;
constexpr int OFF3_L2_B2         = OFF3_L2_W2        + W3_W2;

// Gradients (same shape as params, per layer).
constexpr int OFF3_G_EMBED       = OFF3_L2_B2        + W3_B2;
constexpr int OFF3_G_W_HEAD      = OFF3_G_EMBED      + W3_EMBED;

constexpr int OFF3_G_L0_S_Q      = OFF3_G_W_HEAD     + W3_W_HEAD;
constexpr int OFF3_G_L0_S_K      = OFF3_G_L0_S_Q     + W3_S_Q;
constexpr int OFF3_G_L0_W1       = OFF3_G_L0_S_K     + W3_S_K;
constexpr int OFF3_G_L0_B1       = OFF3_G_L0_W1      + W3_W1;
constexpr int OFF3_G_L0_W2       = OFF3_G_L0_B1      + W3_B1;
constexpr int OFF3_G_L0_B2       = OFF3_G_L0_W2      + W3_W2;

constexpr int OFF3_G_L1_S_Q      = OFF3_G_L0_B2      + W3_B2;
constexpr int OFF3_G_L1_S_K      = OFF3_G_L1_S_Q     + W3_S_Q;
constexpr int OFF3_G_L1_W1       = OFF3_G_L1_S_K     + W3_S_K;
constexpr int OFF3_G_L1_B1       = OFF3_G_L1_W1      + W3_W1;
constexpr int OFF3_G_L1_W2       = OFF3_G_L1_B1      + W3_B1;
constexpr int OFF3_G_L1_B2       = OFF3_G_L1_W2      + W3_W2;

constexpr int OFF3_G_L2_S_Q      = OFF3_G_L1_B2      + W3_B2;
constexpr int OFF3_G_L2_S_K      = OFF3_G_L2_S_Q     + W3_S_Q;
constexpr int OFF3_G_L2_W1       = OFF3_G_L2_S_K     + W3_S_K;
constexpr int OFF3_G_L2_B1       = OFF3_G_L2_W1      + W3_W1;
constexpr int OFF3_G_L2_W2       = OFF3_G_L2_B1      + W3_B1;
constexpr int OFF3_G_L2_B2       = OFF3_G_L2_W2      + W3_W2;

// Per-layer activations (forward) + caches for backward.
constexpr int OFF3_L0_AD_SCORE   = OFF3_G_L2_B2      + W3_B2;
constexpr int OFF3_L0_AN_OUT     = OFF3_L0_AD_SCORE  + W3_AD_SCORE;
constexpr int OFF3_L0_AD_DIL     = OFF3_L0_AN_OUT    + W3_AN_OUT;
constexpr int OFF3_L0_AK_ERO     = OFF3_L0_AD_DIL    + W3_AD_DIL;
constexpr int OFF3_L0_K_CACHE    = OFF3_L0_AK_ERO    + W3_AK_ERO;
constexpr int OFF3_L0_V_CACHE    = OFF3_L0_K_CACHE   + W3_K_CACHE;

constexpr int OFF3_L1_AD_SCORE   = OFF3_L0_V_CACHE   + W3_V_CACHE;
constexpr int OFF3_L1_AN_OUT     = OFF3_L1_AD_SCORE  + W3_AD_SCORE;
constexpr int OFF3_L1_AD_DIL     = OFF3_L1_AN_OUT    + W3_AN_OUT;
constexpr int OFF3_L1_AK_ERO     = OFF3_L1_AD_DIL    + W3_AD_DIL;
constexpr int OFF3_L1_K_CACHE    = OFF3_L1_AK_ERO    + W3_AK_ERO;
constexpr int OFF3_L1_V_CACHE    = OFF3_L1_K_CACHE   + W3_K_CACHE;

constexpr int OFF3_L2_AD_SCORE   = OFF3_L1_V_CACHE   + W3_V_CACHE;
constexpr int OFF3_L2_AN_OUT     = OFF3_L2_AD_SCORE  + W3_AD_SCORE;
constexpr int OFF3_L2_AD_DIL     = OFF3_L2_AN_OUT    + W3_AN_OUT;
constexpr int OFF3_L2_AK_ERO     = OFF3_L2_AD_DIL    + W3_AD_DIL;
constexpr int OFF3_L2_K_CACHE    = OFF3_L2_AK_ERO    + W3_AK_ERO;
constexpr int OFF3_L2_V_CACHE    = OFF3_L2_K_CACHE   + W3_K_CACHE;

// HIDDEN regions: HIDDEN_0 carries L0 output -> L1 input; HIDDEN_1 carries L1 output -> L2 input.
constexpr int OFF3_HIDDEN_0      = OFF3_L2_V_CACHE   + W3_V_CACHE;
constexpr int OFF3_HIDDEN_1      = OFF3_HIDDEN_0     + W3_HIDDEN;

// Shared head activations + IO tensors.
constexpr int OFF3_AD_HEAD       = OFF3_HIDDEN_1     + W3_HIDDEN;
constexpr int OFF3_TOKENS        = OFF3_AD_HEAD      + W3_AD_HEAD;
constexpr int OFF3_TARGETS       = OFF3_TOKENS       + W3_TOKENS;
constexpr int OFF3_GRAD_LOGIT    = OFF3_TARGETS      + W3_TARGETS;

constexpr int FT3_BUFFER_WORDS   = OFF3_GRAD_LOGIT   + W3_GRAD_LOGIT;
constexpr int FT3_BUFFER_BYTES   = FT3_BUFFER_WORDS * 4;

// Mode constants (mirror 2-layer kernel).
constexpr int MODE3_FWD_LOSS = 0;
constexpr int MODE3_BWD      = 1;
constexpr int MODE3_SIGNSGD  = 2;

#endif  // FORGE_BUFFER_3LAYER_HPP
