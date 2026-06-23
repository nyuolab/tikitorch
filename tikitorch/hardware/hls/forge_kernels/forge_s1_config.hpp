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

// ----------------------------------------------------------------------------
// GRADIENT regions (written by mode 1 = backward; consumed by mode 2 = dyadic-Adam).
// APPENDED after the mode-0 map (every mode-0 offset above is UNCHANGED). Each weight
// gradient region is the SAME logical size as its weight region; gradients accumulate as
// int32 FULL precision (summed over the L_SEQ positions), so one int32 word per weight.
// (The weights themselves are dyadic byte-codes / int16; their gradients are int32.)
//   G_EMBED   : [V*D]                 — embed-table grad (sparse scatter, embed bwd)
//   G_WQ/WK/WV/WO : [N_L*D*D] each     — Q/K/V/O projection weight grads (wgrad GEMM)
//   G_W_UP    : [N_L*FFN*D]            — FFN up weight grad
//   G_W_DOWN  : [N_L*D*FFN]            — FFN down weight grad
//   G_LN1_G/B, G_LN2_G/B : [N_L*D] each — LN gamma-grad + beta-grad (per channel)
//   G_W_HEAD  : [V*D]                  — tied LM-head weight grad (sparse scatter)
// ----------------------------------------------------------------------------
static const int S1_OFF_G_EMBED   = S1_OFF_LOSS      + 16;                              // [V*D]
static const int S1_OFF_G_WQ      = S1_OFF_G_EMBED   + S1_AL16(S1_V * S1_D);            // [N_L][D][D]
static const int S1_OFF_G_WK      = S1_OFF_G_WQ      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_G_WV      = S1_OFF_G_WK      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_G_WO      = S1_OFF_G_WV      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_G_W_UP    = S1_OFF_G_WO      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);   // [N_L][FFN][D]
static const int S1_OFF_G_W_DOWN  = S1_OFF_G_W_UP    + S1_AL16(S1_N_LAYERS * S1_FFN * S1_D); // [N_L][D][FFN]
static const int S1_OFF_G_LN1_G   = S1_OFF_G_W_DOWN  + S1_AL16(S1_N_LAYERS * S1_D * S1_FFN); // [N_L][D]
static const int S1_OFF_G_LN1_B   = S1_OFF_G_LN1_G   + S1_AL16(S1_N_LAYERS * S1_D);          // [N_L][D]
static const int S1_OFF_G_LN2_G   = S1_OFF_G_LN1_B   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_G_LN2_B   = S1_OFF_G_LN2_G   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_G_W_HEAD  = S1_OFF_G_LN2_B   + S1_AL16(S1_N_LAYERS * S1_D);          // [V*D]

// ----------------------------------------------------------------------------
// SCRATCH grad regions (mode-1 intermediate activation grads, exposed to the buffer so
// the testbench can bit-check them; the kernel holds them on-chip during compute and
// burst-stores at the end, like the mode-0 activation saves). Each int32, FULL precision.
//   GR    : residual-stream grad gR[t][d]   (L_SEQ*D) — the running backward residual
//   GX1   : LN1-output grad gX1[t][d]       (L_SEQ*D)
//   GX2   : LN2-output grad gX2[t][d]       (L_SEQ*D)
//   GHD   : FFN-hidden (post-ReLU) grad gHd[t][k] (L_SEQ*FFN)
//   GU    : FFN-up (pre-ReLU) grad gU[t][k]      (L_SEQ*FFN)
//   GACAT : concatenated-attn-output grad gAcat[t][d] (L_SEQ*D)
//   GQ/GK/GV : per-head-concatenated Q/K/V grads (L_SEQ*D each)
// ----------------------------------------------------------------------------
static const int S1_OFF_GR      = S1_OFF_G_W_HEAD + S1_AL16(S1_V * S1_D);               // [L_SEQ*D]
static const int S1_OFF_GX1     = S1_OFF_GR      + S1_AL16(S1_L_SEQ * S1_D);
static const int S1_OFF_GX2     = S1_OFF_GX1     + S1_AL16(S1_L_SEQ * S1_D);
static const int S1_OFF_GHD     = S1_OFF_GX2     + S1_AL16(S1_L_SEQ * S1_D);            // [L_SEQ*FFN]
static const int S1_OFF_GU      = S1_OFF_GHD     + S1_AL16(S1_L_SEQ * S1_FFN);          // [L_SEQ*FFN]
static const int S1_OFF_GACAT   = S1_OFF_GU      + S1_AL16(S1_L_SEQ * S1_FFN);          // [L_SEQ*D]
static const int S1_OFF_GQ      = S1_OFF_GACAT   + S1_AL16(S1_L_SEQ * S1_D);            // [L_SEQ*D]
static const int S1_OFF_GK      = S1_OFF_GQ      + S1_AL16(S1_L_SEQ * S1_D);
static const int S1_OFF_GV      = S1_OFF_GK      + S1_AL16(S1_L_SEQ * S1_D);

// ----------------------------------------------------------------------------
// OPTIMIZER regions (mode 2 = dyadic-Adam; mode 3 = fused trainer). APPENDED after the mode-0/1 map —
// EVERY offset above is UNCHANGED (mode 0/1 are bit-exact-frozen). The QAT master-weight scheme:
//
//   W_fp  : FULL-PRECISION master weight (int16, ONE per int32 lane, like EMBED/W_HEAD). One W_fp region
//           per learnable weight. The optimizer accumulates the (sub-ulp) Adam step into W_fp; the forward
//           never reads W_fp — it reads the int8 dyadic CODE region (WQ.. / LN*_G) or the int16 VALUE
//           region (EMBED / W_HEAD / LN*_B). After each Adam step mode 2 RE-QUANTIZES the master:
//             - CODE   weights (WQ/WK/WV/WO/W_UP/W_DOWN/LN1_G/LN2_G): code = s1_encode(dyad(W_fp))
//               (the SAME dyad() rule the forward decodes — self-consistent, deploy-faithful).
//             - VALUE  weights (EMBED/W_HEAD/LN1_B/LN2_B): the int16 master IS the value; mode 2 copies
//               W_fp straight into the value region (no re-encode).
//   M / V : dyadic-Adam EMA state (int32 each, ONE per int32 lane), same element count as the weight.
//           m = m - (m>>B1) + (g>>B1) ; v = v - (v>>B2) + (|g|>>B2)  (forge_dyadic_adam recipe, B1=B2=3).
//
// CODE-INIT contract: the host stages the INITIAL W_fp (full precision). The matching CODE / VALUE region
// the forward reads is produced by RE-QUANTIZING W_fp:
//   (a) mode 3 (do_train_run) re-quantizes ALL masters into codes/values ONCE at entry (before step 0's
//       forward), so the first forward reads valid codes; OR
//   (b) the host calls mode 2 with a1=S1_OPT_INIT (grad treated as zero -> Adam is a no-op, only the
//       re-quantize runs), which populates the codes/values from the staged W_fp without moving weights.
// The one-step bit-exact TB stages W_fp AND the matching code/value (== s1_encode(dyad(W_fp)) / W_fp) so
// mode-0 reads valid codes BEFORE any optimizer step; mode 2 then updates W_fp + re-quantizes.
//
// Region ORDER mirrors the weight map (EMBED, WQ, WK, WV, WO, W_UP, W_DOWN, LN1_G, LN1_B, LN2_G, LN2_B,
// W_HEAD), tripled (W_fp, then M, then V), each 16-aligned.
// ----------------------------------------------------------------------------
// --- W_fp master-weight regions (int16 value, one per int32 lane) ---
static const int S1_OFF_WFP_EMBED  = S1_OFF_GV       + S1_AL16(S1_L_SEQ * S1_D);            // [V*D]
static const int S1_OFF_WFP_WQ     = S1_OFF_WFP_EMBED + S1_AL16(S1_V * S1_D);               // [N_L][D][D]
static const int S1_OFF_WFP_WK     = S1_OFF_WFP_WQ    + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_WFP_WV     = S1_OFF_WFP_WK    + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_WFP_WO     = S1_OFF_WFP_WV    + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_WFP_W_UP   = S1_OFF_WFP_WO    + S1_AL16(S1_N_LAYERS * S1_D * S1_D);   // [N_L][FFN][D]
static const int S1_OFF_WFP_W_DOWN = S1_OFF_WFP_W_UP  + S1_AL16(S1_N_LAYERS * S1_FFN * S1_D); // [N_L][D][FFN]
static const int S1_OFF_WFP_LN1_G  = S1_OFF_WFP_W_DOWN+ S1_AL16(S1_N_LAYERS * S1_D * S1_FFN); // [N_L][D]
static const int S1_OFF_WFP_LN1_B  = S1_OFF_WFP_LN1_G + S1_AL16(S1_N_LAYERS * S1_D);          // [N_L][D]
static const int S1_OFF_WFP_LN2_G  = S1_OFF_WFP_LN1_B + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_WFP_LN2_B  = S1_OFF_WFP_LN2_G + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_WFP_W_HEAD = S1_OFF_WFP_LN2_B + S1_AL16(S1_N_LAYERS * S1_D);          // [V*D]

// --- M (dyadic-Adam first moment) regions (int32) ---
static const int S1_OFF_M_EMBED    = S1_OFF_WFP_W_HEAD+ S1_AL16(S1_V * S1_D);
static const int S1_OFF_M_WQ       = S1_OFF_M_EMBED   + S1_AL16(S1_V * S1_D);
static const int S1_OFF_M_WK       = S1_OFF_M_WQ      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_M_WV       = S1_OFF_M_WK      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_M_WO       = S1_OFF_M_WV      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_M_W_UP     = S1_OFF_M_WO      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_M_W_DOWN   = S1_OFF_M_W_UP    + S1_AL16(S1_N_LAYERS * S1_FFN * S1_D);
static const int S1_OFF_M_LN1_G    = S1_OFF_M_W_DOWN  + S1_AL16(S1_N_LAYERS * S1_D * S1_FFN);
static const int S1_OFF_M_LN1_B    = S1_OFF_M_LN1_G   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_M_LN2_G    = S1_OFF_M_LN1_B   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_M_LN2_B    = S1_OFF_M_LN2_G   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_M_W_HEAD   = S1_OFF_M_LN2_B   + S1_AL16(S1_N_LAYERS * S1_D);

// --- V (dyadic-Adam second moment = EMA of |g|) regions (int32) ---
static const int S1_OFF_V_EMBED    = S1_OFF_M_W_HEAD  + S1_AL16(S1_V * S1_D);
static const int S1_OFF_V_WQ       = S1_OFF_V_EMBED   + S1_AL16(S1_V * S1_D);
static const int S1_OFF_V_WK       = S1_OFF_V_WQ      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_V_WV       = S1_OFF_V_WK      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_V_WO       = S1_OFF_V_WV      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_V_W_UP     = S1_OFF_V_WO      + S1_AL16(S1_N_LAYERS * S1_D * S1_D);
static const int S1_OFF_V_W_DOWN   = S1_OFF_V_W_UP    + S1_AL16(S1_N_LAYERS * S1_FFN * S1_D);
static const int S1_OFF_V_LN1_G    = S1_OFF_V_W_DOWN  + S1_AL16(S1_N_LAYERS * S1_D * S1_FFN);
static const int S1_OFF_V_LN1_B    = S1_OFF_V_LN1_G   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_V_LN2_G    = S1_OFF_V_LN1_B   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_V_LN2_B    = S1_OFF_V_LN2_G   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_V_W_HEAD   = S1_OFF_V_LN2_B   + S1_AL16(S1_N_LAYERS * S1_D);

// dyadic-Adam constants (== forge_dyadic_adam.cpp; == charlm dyadic-Adam betas 0.875 = 1 - 2^-3).
#define S1_ADAM_B1     3      // first-moment EMA shift: m += (g - m) >> B1   (beta1 = 1 - 2^-3)
#define S1_ADAM_B2     3      // second-moment EMA shift: v += (|g| - v) >> B2 (beta2 = 1 - 2^-3)
// learning rate = 2^-LR_SHIFT (a shift; W_fp -= step >> LR_SHIFT). NOTE the step is ALREADY normalized by
// the dyadic-Adam adaptive scale (step = m >> round(log2 v) ≈ sign(m)·O(1) — the dyadic analog of Adam's
// m/√v ≈ ±1), so on the int16 MASTER the natural integer learning rate is 2^0=1: each step nudges W_fp by
// the O(1) normalized amount, accumulating to cross dyad() power-of-2 boundaries (code weights) and to
// move the int16 values (EMBED/W_HEAD/beta). forge_dyadic_adam.cpp uses LR_SHIFT=12 because its weights
// are int32 at a ~2^12 fixed-point scale; the S1 int16 master is at unit scale -> LR_SHIFT=0. Tune per-rig.
#ifndef S1_LR_SHIFT
#define S1_LR_SHIFT    0
#endif

// mode-2 a1 selector: a normal Adam step (default) vs init-only re-quantize (grad ignored, Adam no-op).
#define S1_OPT_STEP    0      // a1=0 -> full dyadic-Adam step + re-quantize + grad-zero
#define S1_OPT_INIT    1      // a1=1 -> re-quantize codes/values from the staged W_fp ONLY (no weight move)

// mode-2/3 a2 selector: which weights the optimizer MOVES.
//   S1_OPT_TRAIN_ALL    : every learnable region steps (the dense PRE-DYADIC CODE backbone INCLUDED).
//   S1_OPT_FREEZE_CODES : the dense code weights (is_code=1: Wq/Wk/Wv/Wo/W_up/W_down/LN1_g/LN2_g) are
//        FROZEN — their grad is dropped, master+code untouched — and only the SCATTER/VALUE weights
//        (EMBED, W_HEAD, LN1_b/LN2_b) train. This is the QAT "frozen-quantized-backbone, train the
//        embeddings + head" regime, and is the configuration in which the dyadic-Adam trainer is STABLE:
//        with the full dense stack trainable, the dyadic re-quantize RATCHETS the code magnitudes upward
//        (a ±1 int16-master nudge crosses a dyad() power-of-2 boundary, doubling the code), which drives
//        the residual stream into the int16 saturation ceiling and collapses the max-plus head to uniform
//        logits (no learning — the all-trainable "convergence" was a saturation artifact). FREEZE_CODES
//        keeps the verified dyadic backbone fixed and lets the scatter weights genuinely reduce the CE.
//        (init re-quantizes EVERY code regardless, so the first forward reads valid codes for all regions.)
#define S1_OPT_TRAIN_ALL    0
#define S1_OPT_FREEZE_CODES 1

static const int S1_BUFLEN         = S1_OFF_V_W_HEAD + S1_AL16(S1_V * S1_D);   // total int32 words; multiple of 16

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

// gradient-region layer-major helpers (mirror s1_off_w / s1_off_ln; weight grads are ROW-MAJOR [O][I]).
static inline int s1_off_gw(int base, int l, int O, int I) { return base + l * (O * I); }   // [l][o][i]
static inline int s1_off_gln(int base, int l)              { return base + l * S1_D; }       // [l][d]

// ----------------------------------------------------------------------------
// OPTIMIZER region descriptor table — mode 2 (do_opt) walks this once. Each learnable weight has a
// 5-tuple { grad off, W_fp off, M off, V off, code/value off, element count, is_code }. is_code=1 means
// the forward reads an int8 dyadic CODE produced by s1_encode(dyad(W_fp)); is_code=0 means the forward
// reads the int16 VALUE which is just W_fp copied through (EMBED / W_HEAD / LN beta). The element count is
// the LOGICAL extent (the kernel iterates this many params; the S1_AL16 padding between regions is unread).
// ----------------------------------------------------------------------------
#define S1_NUM_OPT_REGIONS 12
struct s1_opt_region_t { int g_off, wfp_off, m_off, v_off, dst_off, n, is_code; };
// dst_off = the CODE region (is_code=1) or VALUE region (is_code=0) the forward reads.
static const s1_opt_region_t S1_OPT_REGIONS[S1_NUM_OPT_REGIONS] = {
    // grad,            W_fp,               M,                V,                dst (code/value),  n,                              is_code
    { S1_OFF_G_EMBED,   S1_OFF_WFP_EMBED,   S1_OFF_M_EMBED,   S1_OFF_V_EMBED,   S1_OFF_EMBED,   S1_V * S1_D,                 0 }, // EMBED  value
    { S1_OFF_G_WQ,      S1_OFF_WFP_WQ,      S1_OFF_M_WQ,      S1_OFF_V_WQ,      S1_OFF_WQ,      S1_N_LAYERS * S1_D * S1_D,   1 }, // Wq     code
    { S1_OFF_G_WK,      S1_OFF_WFP_WK,      S1_OFF_M_WK,      S1_OFF_V_WK,      S1_OFF_WK,      S1_N_LAYERS * S1_D * S1_D,   1 }, // Wk     code
    { S1_OFF_G_WV,      S1_OFF_WFP_WV,      S1_OFF_M_WV,      S1_OFF_V_WV,      S1_OFF_WV,      S1_N_LAYERS * S1_D * S1_D,   1 }, // Wv     code
    { S1_OFF_G_WO,      S1_OFF_WFP_WO,      S1_OFF_M_WO,      S1_OFF_V_WO,      S1_OFF_WO,      S1_N_LAYERS * S1_D * S1_D,   1 }, // Wo     code
    { S1_OFF_G_W_UP,    S1_OFF_WFP_W_UP,    S1_OFF_M_W_UP,    S1_OFF_V_W_UP,    S1_OFF_W_UP,    S1_N_LAYERS * S1_FFN * S1_D, 1 }, // W_up   code
    { S1_OFF_G_W_DOWN,  S1_OFF_WFP_W_DOWN,  S1_OFF_M_W_DOWN,  S1_OFF_V_W_DOWN,  S1_OFF_W_DOWN,  S1_N_LAYERS * S1_D * S1_FFN, 1 }, // W_down code
    { S1_OFF_G_LN1_G,   S1_OFF_WFP_LN1_G,   S1_OFF_M_LN1_G,   S1_OFF_V_LN1_G,   S1_OFF_LN1_G,   S1_N_LAYERS * S1_D,          1 }, // LN1 g  code
    { S1_OFF_G_LN1_B,   S1_OFF_WFP_LN1_B,   S1_OFF_M_LN1_B,   S1_OFF_V_LN1_B,   S1_OFF_LN1_B,   S1_N_LAYERS * S1_D,          0 }, // LN1 b  value
    { S1_OFF_G_LN2_G,   S1_OFF_WFP_LN2_G,   S1_OFF_M_LN2_G,   S1_OFF_V_LN2_G,   S1_OFF_LN2_G,   S1_N_LAYERS * S1_D,          1 }, // LN2 g  code
    { S1_OFF_G_LN2_B,   S1_OFF_WFP_LN2_B,   S1_OFF_M_LN2_B,   S1_OFF_V_LN2_B,   S1_OFF_LN2_B,   S1_N_LAYERS * S1_D,          0 }, // LN2 b  value
    { S1_OFF_G_W_HEAD,  S1_OFF_WFP_W_HEAD,  S1_OFF_M_W_HEAD,  S1_OFF_V_W_HEAD,  S1_OFF_W_HEAD,  S1_V * S1_D,                 0 }, // W_HEAD value
};

#endif  // FORGE_S1_CONFIG_HPP
