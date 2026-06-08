// forge_buffer_2layer.hpp — buffer layout for the 2-layer extension of
// the Tier-1 tropical attention LM (Silicon Experiment 5).
//
// Mirrors forge_buffer.hpp's single-layer layout but doubles every
// per-layer region (S_Q, S_K, W1, B1, W2, B2 and their gradient slots)
// and adds a hidden-state region carrying the residual stream between
// layers.
//
// Buffer growth:
//   1-layer total = ~5.1 MB  (well within VU47P URAM/BRAM budget).
//   2-layer total = ~9.7 MB  (still under the ~270 Mb on-chip budget).
//
// ABI extension: a new `layer_id` arg is added to the dispatcher
// (passed via arg4) so the same kernel can serve both layers.  Host
// runs the train step as:
//   mode=FWD_LOSS, layer_id=0, m=*, token=*, target=ignored   -> layer 0 fwd
//   mode=FWD_LOSS, layer_id=1, m=*, token=ignored, target=*   -> layer 1 fwd + loss
//   mode=BWD,      layer_id=1, m=*, token=ignored             -> layer 1 bwd
//   mode=BWD,      layer_id=0, m=*, token=*                   -> layer 0 bwd
//   mode=SIGNSGD,  layer_id=-1 (ignored), lr=*, threshold=*   -> updates all params

#ifndef FORGE_BUFFER_2LAYER_HPP
#define FORGE_BUFFER_2LAYER_HPP

#include "../forge_hls_types.hpp"
#include <cstdint>

// Shape constants — same as Tier-1 single-layer build.
constexpr int FT2_V       = 128;
constexpr int FT2_D       = 128;
constexpr int FT2_D_FF    = 512;
constexpr int FT2_L_MAX   = 128;
constexpr int FT2_LOG2_D  = 7;
constexpr int FT2_N_LAYERS = 2;

// Per-layer word counts (identical to single-layer).
constexpr int W2_EMBED      = FT2_V * FT2_D;
constexpr int W2_S_Q        = FT2_L_MAX * FT2_D;
constexpr int W2_S_K        = FT2_L_MAX * FT2_D;
constexpr int W2_W1         = FT2_D_FF * FT2_D;
constexpr int W2_B1         = FT2_D_FF;
constexpr int W2_W2         = FT2_D * FT2_D_FF;
constexpr int W2_B2         = FT2_D;
constexpr int W2_W_HEAD     = FT2_V * FT2_D;

// Scratch / activation buffers (per-layer).
constexpr int W2_AD_SCORE   = FT2_L_MAX * FT2_L_MAX;
constexpr int W2_AN_OUT     = FT2_L_MAX * FT2_D;
constexpr int W2_AD_DIL     = FT2_L_MAX * FT2_D_FF;
constexpr int W2_AK_ERO     = FT2_L_MAX * FT2_D;
constexpr int W2_AD_HEAD    = FT2_L_MAX * FT2_V;
constexpr int W2_K_CACHE    = FT2_L_MAX * FT2_D;
constexpr int W2_V_CACHE    = FT2_L_MAX * FT2_D;

constexpr int W2_TOKENS     = FT2_L_MAX;
constexpr int W2_TARGETS    = FT2_L_MAX;
constexpr int W2_GRAD_LOGIT = FT2_L_MAX * FT2_V;

// New for 2-layer: the residual stream output of layer 0 = input to layer 1.
// Stored as (L_MAX, D) int32 — same shape as AN_OUT but carries the post-FFN
// hidden state.  In the 1-layer build this was implicit (layer 0's AN_OUT
// fed directly into the LM head); now we need an explicit slot.
constexpr int W2_HIDDEN     = FT2_L_MAX * FT2_D;

// =============================================================================
// Per-layer parameter offsets.  Layer 0 first, then layer 1.
// =============================================================================
constexpr int OFF2_EMBED         = 0;
constexpr int OFF2_W_HEAD        = OFF2_EMBED + W2_EMBED;

// Layer 0 parameters
constexpr int OFF2_L0_S_Q        = OFF2_W_HEAD       + W2_W_HEAD;
constexpr int OFF2_L0_S_K        = OFF2_L0_S_Q       + W2_S_Q;
constexpr int OFF2_L0_W1         = OFF2_L0_S_K       + W2_S_K;
constexpr int OFF2_L0_B1         = OFF2_L0_W1        + W2_W1;
constexpr int OFF2_L0_W2         = OFF2_L0_B1        + W2_B1;
constexpr int OFF2_L0_B2         = OFF2_L0_W2        + W2_W2;

// Layer 1 parameters
constexpr int OFF2_L1_S_Q        = OFF2_L0_B2        + W2_B2;
constexpr int OFF2_L1_S_K        = OFF2_L1_S_Q       + W2_S_Q;
constexpr int OFF2_L1_W1         = OFF2_L1_S_K       + W2_S_K;
constexpr int OFF2_L1_B1         = OFF2_L1_W1        + W2_W1;
constexpr int OFF2_L1_W2         = OFF2_L1_B1        + W2_B1;
constexpr int OFF2_L1_B2         = OFF2_L1_W2        + W2_W2;

// Gradients (same shape as params, per layer).
constexpr int OFF2_G_EMBED       = OFF2_L1_B2        + W2_B2;
constexpr int OFF2_G_W_HEAD      = OFF2_G_EMBED      + W2_EMBED;
constexpr int OFF2_G_L0_S_Q      = OFF2_G_W_HEAD     + W2_W_HEAD;
constexpr int OFF2_G_L0_S_K      = OFF2_G_L0_S_Q     + W2_S_Q;
constexpr int OFF2_G_L0_W1       = OFF2_G_L0_S_K     + W2_S_K;
constexpr int OFF2_G_L0_B1       = OFF2_G_L0_W1      + W2_W1;
constexpr int OFF2_G_L0_W2       = OFF2_G_L0_B1      + W2_B1;
constexpr int OFF2_G_L0_B2       = OFF2_G_L0_W2      + W2_W2;
constexpr int OFF2_G_L1_S_Q      = OFF2_G_L0_B2      + W2_B2;
constexpr int OFF2_G_L1_S_K      = OFF2_G_L1_S_Q     + W2_S_Q;
constexpr int OFF2_G_L1_W1       = OFF2_G_L1_S_K     + W2_S_K;
constexpr int OFF2_G_L1_B1       = OFF2_G_L1_W1      + W2_W1;
constexpr int OFF2_G_L1_W2       = OFF2_G_L1_B1      + W2_B1;
constexpr int OFF2_G_L1_B2       = OFF2_G_L1_W2      + W2_W2;

// Per-layer activations (forward) + caches for backward.  Indexed by layer.
constexpr int OFF2_L0_AD_SCORE   = OFF2_G_L1_B2      + W2_B2;
constexpr int OFF2_L0_AN_OUT     = OFF2_L0_AD_SCORE  + W2_AD_SCORE;
constexpr int OFF2_L0_AD_DIL     = OFF2_L0_AN_OUT    + W2_AN_OUT;
constexpr int OFF2_L0_AK_ERO     = OFF2_L0_AD_DIL    + W2_AD_DIL;
constexpr int OFF2_L0_K_CACHE    = OFF2_L0_AK_ERO    + W2_AK_ERO;
constexpr int OFF2_L0_V_CACHE    = OFF2_L0_K_CACHE   + W2_K_CACHE;

constexpr int OFF2_L1_AD_SCORE   = OFF2_L0_V_CACHE   + W2_V_CACHE;
constexpr int OFF2_L1_AN_OUT     = OFF2_L1_AD_SCORE  + W2_AD_SCORE;
constexpr int OFF2_L1_AD_DIL     = OFF2_L1_AN_OUT    + W2_AN_OUT;
constexpr int OFF2_L1_AK_ERO     = OFF2_L1_AD_DIL    + W2_AD_DIL;
constexpr int OFF2_L1_K_CACHE    = OFF2_L1_AK_ERO    + W2_AK_ERO;
constexpr int OFF2_L1_V_CACHE    = OFF2_L1_K_CACHE   + W2_K_CACHE;

// Residual hidden state between layers (output of layer 0 -> input of layer 1).
constexpr int OFF2_HIDDEN        = OFF2_L1_V_CACHE   + W2_V_CACHE;

// Shared head activations + IO tensors.
constexpr int OFF2_AD_HEAD       = OFF2_HIDDEN       + W2_HIDDEN;
constexpr int OFF2_TOKENS        = OFF2_AD_HEAD      + W2_AD_HEAD;
constexpr int OFF2_TARGETS       = OFF2_TOKENS       + W2_TOKENS;
constexpr int OFF2_GRAD_LOGIT    = OFF2_TARGETS      + W2_TARGETS;

// Dedicated logits scratchpad (one row of V int16-in-int32 per position).  The
// L1 forward writes raw logits HERE; the host reads them and writes the loss
// gradient to OFF2_GRAD_LOGIT.  Separating logits-out from grad-in (previously
// both aliased OFF2_GRAD_LOGIT) lets host-side cross-entropy be validated
// without the kernel clobbering its own logits, and matches the closure-faithful
// CE in tikitorch.functional.tropical_cross_entropy.  Append-only: every offset
// above is unchanged, only the buffer grows by W2_LOGITS words.
constexpr int W2_LOGITS          = FT2_L_MAX * FT2_V;
constexpr int OFF2_LOGITS        = OFF2_GRAD_LOGIT   + W2_GRAD_LOGIT;

constexpr int FT2_BUFFER_WORDS   = OFF2_LOGITS       + W2_LOGITS;
constexpr int FT2_BUFFER_BYTES   = FT2_BUFFER_WORDS * 4;

// Mode constants (mirror 1-layer kernel).
constexpr int MODE2_FWD_LOSS = 0;
constexpr int MODE2_BWD      = 1;
constexpr int MODE2_SIGNSGD  = 2;

#endif  // FORGE_BUFFER_2LAYER_HPP
