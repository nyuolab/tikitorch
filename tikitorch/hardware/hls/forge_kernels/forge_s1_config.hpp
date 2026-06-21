// forge_s1_config.hpp — S1 transformer-block dimensions + the COMPLETE monolithic-buffer offset map
// for the on-chip end-to-end trainer forge_train_step_s1. Shared by the kernel and the testbench so
// the two NEVER disagree on a layout word. Every region base is 16-word (512-bit) aligned (the wide
// burst contract; avoids the morph OFF_CE_LUT unaligned-read class of silicon bug).
//
// S1 is the SMALL, on-chip, no-DDR config (the d=768 kernels shrunk to d=128 so the whole block +
// activation checkpoints fit in URAM/BRAM for the first integration CSIM). The dims are chosen to
// REUSE the verified kernels with zero arithmetic change:
//   D=128  (LOG2_D=7)  -> LayerNorm mean is sum>>7 (no divider); matches morph ML_D / forge_layernorm LN_D
//   H=2, DH=64         -> head_dim == forge_attn AT_DH=64 EXACTLY (attn reused per-head, the ONE multiply)
//   L_SEQ=128          -> == forge_attn AT_L=128 EXACTLY (whole-sequence, causal)
//   FFN=512 (=4*D)     -> the FFN inner width (largest tensor through ReLU); ÷16
//   V=128              -> tied embed/LM-head; small CSIM vocab; matches morph ML_V (CE-LUT ports verbatim)
//   N_LAYERS=1         -> one block first; offsets are LAYER-MAJOR so bumping to 2 is a define change
//
// WEIGHT ENCODINGS (each region documents its contract; identical to the source kernels):
//   EMBED   : plain int16 activation values, one per int32 lane (morph embed gather: R[d]=EMBED[tok*D+d])
//   Wq/Wk/Wv/Wo/W_up/W_down : PRE-DYADIC int8 byte-codes, one per int32 lane (forge_fwd_wide wdecode():
//             bit7=sign, bit5=nonzero, bits4..0=exponent). The GEMM decodes inline. ROW-MAJOR [O][I].
//   LN1/LN2 gamma : PRE-DYADIC int8 byte-codes (forge_layernorm wdecode), one per channel.
//   LN1/LN2 beta  : plain int16 additive, one per channel.
//   W_HEAD  : plain int16, ROW-MAJOR [V][D]; the max-plus head adds W_HEAD[v][d]+R[d] (morph LM head).
//   CE_LUT  : host-staged exp2 ROM (256 Q16; forge_ce_lut.hpp FT_CE_EXP2_Q16). Read at runtime (the
//             BRAM static-init-reads-zero F2 AFI bug => MUST be host-staged).
//   ATTN_LUT: host-staged exp2 ROM for the softmax (256 Q16; forge_attn exp2_lut). Same staging rule.
//   TOKENS/TARGETS : one token id per position (int32), L_SEQ entries each.
//
// ACTIVATION-SAVE regions (allocated NOW for the future mode-1 backward; written by mode 0):
//   R       : per-layer block-output residual stream R_l[t][d], (N_LAYERS+1)*L_SEQ*D  (R[0]=embed in,
//             R[l+1]=block-l output; the backward needs each layer's input residual)
//   X1/X2   : LN1 / LN2 outputs per layer, N_LAYERS*L_SEQ*D each (LN-bwd needs them via saved stats too)
//   LN1_STATS/LN2_STATS : per-(layer,token) fine-dyadic (e,nq), N_LAYERS*L_SEQ*2 each (forge_layernorm
//             OFF_STATS contract; the LN backward reconstructs invq from these)
//   ATTN_P  : soft attention probabilities p[h][i][j] (Q16), N_LAYERS*H*L_SEQ*L_SEQ (the blend bwd needs p)
//   ATTN_JSTAR : argmax j* per (layer,head,query) for the deploy/gather adjoint, N_LAYERS*H*L_SEQ
//   RELU_MASK  : FFN ReLU active mask, N_LAYERS*L_SEQ*FFN (1 word/elt; backward gate)
//   ARGMAX_HEAD: LM-head argmax d* per (position,vocab), L_SEQ*V (head bwd sparse scatter route)
//
// OUTPUTS:
//   GRAD_LOGIT : grad at logits, L_SEQ*V (rows sum to zero; the CE adjoint, mode-1 entry point)
//   LOGITS     : the LM-head logits per position, L_SEQ*V (saved for the loss + bit-exact check)
//   LOSS       : scalar accumulated loss proxy region (16 words; [0]=sum of -log p(target) proxy)

#ifndef FORGE_S1_CONFIG_HPP
#define FORGE_S1_CONFIG_HPP

#include <cstdint>

// ---- S1 model config ----
#define S1_D        128     // d_model
#define S1_LOG2_D   7       // log2(D): LayerNorm mean = sum >> 7 (no divider); D MUST == 1<<LOG2_D
#define S1_H        2       // attention heads
#define S1_DH       64      // head dim (== forge_attn AT_DH); H*DH == D
#define S1_FFN      512     // FFN inner width (4*D)
#define S1_L_SEQ    128     // sequence length (== forge_attn AT_L)
#define S1_V        128     // vocab (tied embed / LM-head)
#define S1_N_LAYERS 1       // transformer blocks (offsets are layer-major; bump to 2 once bit-exact)
#define S1_MBITS    2       // fine-dyadic LN mantissa bits (== forge_layernorm LN_MBITS)
#define S1_LUT_N    256     // exp2 LUT entries (CE + attn)

// attention softmax constants (== forge_attn)
#define S1_AT_DELTA_SHIFT 2 // (rowmax - s) >> 2  (Delta = 4)
#define S1_AT_P_FRAC      16 // probabilities in Q16

// CE constants (== morph LM CE path)
#define S1_CE_ZSHIFT 4      // (logit[pred]-logit[v]) >> zshift  index into ce_lut

// 16-word (512-bit) alignment helper: round a count up to a multiple of 16.
#define S1_AL16(x) ( (((x) + 15) / 16) * 16 )

// ----------------------------------------------------------------------------
// Weight / static-input regions (read-only in mode 0).
// ----------------------------------------------------------------------------
static const int S1_OFF_EMBED   = 0;                                            // [V*D] int16 embed table
static const int S1_OFF_WQ      = S1_OFF_EMBED  + S1_AL16(S1_V * S1_D);         // [N_L][D][D] dyadic codes
static const int S1_OFF_WK      = S1_OFF_WQ     + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_WV      = S1_OFF_WK     + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_WO      = S1_OFF_WV     + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_W_UP    = S1_OFF_WO     + S1_AL16(S1_N_LAYERS * S1_D * S1_D);   // [N_L][FFN][D]
static const int S1_OFF_W_DOWN  = S1_OFF_W_UP   + S1_AL16(S1_N_LAYERS * S1_FFN * S1_D); // [N_L][D][FFN]
static const int S1_OFF_LN1_G   = S1_OFF_W_DOWN + S1_AL16(S1_N_LAYERS * S1_D * S1_FFN); // [N_L][D] gamma codes
static const int S1_OFF_LN1_B   = S1_OFF_LN1_G  + S1_AL16(S1_N_LAYERS * S1_D);          // [N_L][D] beta int16
static const int S1_OFF_LN2_G   = S1_OFF_LN1_B  + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_LN2_B   = S1_OFF_LN2_G  + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_W_HEAD  = S1_OFF_LN2_B  + S1_AL16(S1_N_LAYERS * S1_D);          // [V][D] int16
static const int S1_OFF_CE_LUT  = S1_OFF_W_HEAD + S1_AL16(S1_V * S1_D);                 // [256] Q16 exp2
static const int S1_OFF_ATTN_LUT= S1_OFF_CE_LUT + S1_AL16(S1_LUT_N);                    // [256] Q16 exp2
static const int S1_OFF_TOKENS  = S1_OFF_ATTN_LUT + S1_AL16(S1_LUT_N);                  // [L_SEQ] token ids
static const int S1_OFF_TARGETS = S1_OFF_TOKENS + S1_AL16(S1_L_SEQ);                    // [L_SEQ] target ids

// ----------------------------------------------------------------------------
// Activation-save / checkpoint regions (written in mode 0; read by the future mode 1).
// ----------------------------------------------------------------------------
static const int S1_OFF_R         = S1_OFF_TARGETS + S1_AL16(S1_L_SEQ);                 // [(N_L+1)*L_SEQ*D]
static const int S1_OFF_X1        = S1_OFF_R       + S1_AL16((S1_N_LAYERS + 1) * S1_L_SEQ * S1_D);
static const int S1_OFF_X2        = S1_OFF_X1      + S1_AL16(S1_N_LAYERS * S1_L_SEQ * S1_D);
static const int S1_OFF_LN1_STATS = S1_OFF_X2      + S1_AL16(S1_N_LAYERS * S1_L_SEQ * S1_D);
static const int S1_OFF_LN2_STATS = S1_OFF_LN1_STATS + S1_AL16(S1_N_LAYERS * S1_L_SEQ * 2);
static const int S1_OFF_ATTN_P    = S1_OFF_LN2_STATS + S1_AL16(S1_N_LAYERS * S1_L_SEQ * 2);
                                                                                        // [N_L*H*L_SEQ*L_SEQ]
static const int S1_OFF_ATTN_JSTAR= S1_OFF_ATTN_P  + S1_AL16(S1_N_LAYERS * S1_H * S1_L_SEQ * S1_L_SEQ);
static const int S1_OFF_RELU_MASK = S1_OFF_ATTN_JSTAR + S1_AL16(S1_N_LAYERS * S1_H * S1_L_SEQ);
                                                                                        // [N_L*L_SEQ*FFN]
static const int S1_OFF_ARGMAX_HEAD = S1_OFF_RELU_MASK + S1_AL16(S1_N_LAYERS * S1_L_SEQ * S1_FFN);
                                                                                        // [L_SEQ*V]
// ----------------------------------------------------------------------------
// Output regions.
// ----------------------------------------------------------------------------
static const int S1_OFF_GRAD_LOGIT = S1_OFF_ARGMAX_HEAD + S1_AL16(S1_L_SEQ * S1_V);     // [L_SEQ*V]
static const int S1_OFF_LOGITS     = S1_OFF_GRAD_LOGIT  + S1_AL16(S1_L_SEQ * S1_V);     // [L_SEQ*V]
static const int S1_OFF_LOSS       = S1_OFF_LOGITS      + S1_AL16(S1_L_SEQ * S1_V);     // [16] scalar(s)

static const int S1_BUFLEN         = S1_OFF_LOSS + 16;   // total int32 words; already a multiple of 16

// ---- per-region addressing helpers (layer-major; collapse to single-layer at l=0) ----
// weight rows are ROW-MAJOR [O][I]; a layer's slab is l*(O*I).
static inline int s1_off_w(int base, int l, int O, int I) { return base + l * (O * I); }   // [l][o][i]
static inline int s1_off_ln(int base, int l)            { return base + l * S1_D; }         // [l][d]
static inline int s1_off_lnstats(int base, int l)       { return base + l * S1_L_SEQ * 2; } // [l][t][2]
static inline int s1_off_act(int base, int l)           { return base + l * S1_L_SEQ * S1_D; } // [l][t][d]
static inline int s1_off_R(int l)                       { return S1_OFF_R + l * S1_L_SEQ * S1_D; }
static inline int s1_off_attn_p(int l, int h)           { return S1_OFF_ATTN_P + (l * S1_H + h) * S1_L_SEQ * S1_L_SEQ; }
static inline int s1_off_attn_j(int l, int h)           { return S1_OFF_ATTN_JSTAR + (l * S1_H + h) * S1_L_SEQ; }
static inline int s1_off_relu(int l)                    { return S1_OFF_RELU_MASK + l * S1_L_SEQ * S1_FFN; }

#endif  // FORGE_S1_CONFIG_HPP
