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
//   W_HEAD  : S1_HEAD_MODE=0 -> plain int16, ROW-MAJOR [V][D]; the max-plus head adds W_HEAD[v][d]+R[d].
//             S1_HEAD_MODE=1 -> PRE-DYADIC int8 byte-codes, ROW-MAJOR [V][D]; the dyadic-GEMM head decodes
//             inline (wdecode) and computes logits[v]=sum_d wdecode(code)*hf[d] (== the backbone GEMM).
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

// ---- S1 model config (S1-small: D=64, halved from D=128 so the AFI fits+closes on F2) ----
// D=128 monolithic was in an area<->timing bind (ALLOCATION-sharing muxes -> WNS -4.6ns post-route across 5
// AFI builds). Halving every dim shrinks the design ~4x -> ~18% LUT / ~16% BRAM -> the sharing muxes shrink and
// it fits+closes with margin (keeps all 4 modes incl the fused trainer). All dims stay 16-divisible (wide bursts).
#define S1_D        64      // d_model (was 128)
#define S1_LOG2_D   6       // log2(D)=6: LayerNorm mean = sum >> 6 (no divider); D MUST == 1<<LOG2_D
#define S1_H        2       // attention heads
#define S1_DH       32      // head dim; H*DH == D (2*32=64)  (was 64)
#define S1_FFN      256     // FFN inner width (4*D)  (was 512)
#define S1_L_SEQ    64      // sequence length  (was 128)
#define S1_V        64      // vocab (tied embed / LM-head)  (was 128; >= 8-symbol memorization task)
#define S1_N_LAYERS 1       // transformer blocks (offsets are layer-major; bump to 2 once bit-exact)
#define S1_MBITS    2       // fine-dyadic LN mantissa bits (== forge_layernorm LN_MBITS)
#define S1_LUT_N    256     // exp2 LUT entries (CE + attn)

// ----------------------------------------------------------------------------
// FINAL LAYERNORM before the LM head (S1_USE_LNF) — the TRAIN_ALL fix (hypothesis 3 = scale control).
// ----------------------------------------------------------------------------
// THE ROOT CAUSE the prior (frozen-only) verdict diagnosed: the max-plus LM head reads the residual stream
// R DIRECTLY (logits[v] = max_d W_HEAD[v][d] + R[t][d]) with NO normalization. When the dense PRE-DYADIC
// CODE backbone is trainable (TRAIN_ALL), the codes densify, the additive/max-plus GEMM chain INFLATES |R|,
// and R hits the int16 saturation ceiling (32767) -> every logit pins to 32767 -> uniform distribution ->
// CE pinned at log V -> NO learning. (Verified: Q_CODE=6 moves 150k code bytes but peak|R|=32768 and CE
// drops only 0.40 nats < FREEZE's 2.25.) This is an ARCHITECTURE gap, NOT the per-weight ratchet (the fine
// master + exp-clamp already removed the ratchet): the reference float trainer charlm_mixer.py reaches 2.481
// at this config and its GPT has `return self.head(self.lnf(h))` — a FINAL LayerNorm BEFORE the head that
// reins R in. The S1 kernel was missing it.
//
// THE FIX: add a final fine-dyadic LayerNorm LN_F (the SAME multiplier-free layernorm_tok the blocks use)
// between the last block's residual output and the LM head. Now the head reads NORMALIZED Rn (bounded to
// ~meanabs*gamma*mbits regardless of how large the raw R grows), so densifying the dense backbone reshapes
// R's DIRECTION (which the max-plus argmax-over-d head exploits) WITHOUT saturating the head input. Pure
// shift/add/compare (DSP=0); one extra weight region pair (gamma code + beta value) + its grad/master/mv.
// S1_USE_LNF=1 (default ON) inserts LN_F; =0 reproduces the old head-on-raw-R path (the frozen-only regime).
#ifndef S1_USE_LNF
#define S1_USE_LNF  1
#endif

// ----------------------------------------------------------------------------
// LM-HEAD MODE (S1_HEAD_MODE) — the max-plus-head COLLAPSE fix.
// ----------------------------------------------------------------------------
// THE ROOT CAUSE (proven): the legacy LM head is MAX-PLUS — logits[v]=max_d(W_HEAD[v][d]+R_final[d]), an
// argmax-over-d MORPHOLOGICAL readout. That op collapses every input row to (essentially) the unigram
// marginal: NO optimizer (even a perfect FP-master Adam) and NO capacity (D=64/1L .. D=256/4L) beats the
// unigram entropy (4.43 bpc) — confirmed empirically. A torch probe settled the FIX: a LINEAR (dot-product)
// head gets 2.59 bpc and a DYADIC (mult-free, weights snapped to ±2^k) head gets 2.63 bpc — both BEAT the
// bigram baseline (3.47) by ~0.9 bits, while the max-plus head (3.59) does not. So the fix = replace the
// max-plus head with a MULTIPLIER-FREE DYADIC-GEMM head (the SAME shift-add GEMM the backbone uses).
//
//   S1_HEAD_MODE=0 (default, LEGACY) : max-plus head logits[v]=max_d(W_HEAD[v][d]+hf[d]) + argmax d* save.
//        W_HEAD is an int16 VALUE region (is_code=0). EVERY mode/test/AFI path is BYTE-IDENTICAL to before.
//   S1_HEAD_MODE=1 (DYADIC-GEMM)     : logits[v]=sum_d wdecode(WHEAD_code[v][d])·hf[d], a dyadic GEMM
//        [V][D]x[D] (== fwd_gemm, output dim V; DSP=0). hf = the LN_F output (Rn). W_HEAD becomes a DYADIC
//        CODE region (int8 codes, like Wq), UNTIED from the embedding; S1_OPT_REGIONS sets W_HEAD is_code=1
//        so the optimizer re-quantizes it as a code automatically. No argmax_head save (the head is a GEMM,
//        not a max). The backward mirrors the backbone GEMM bwd: grad_hf = q(W)^T@gO (xgrad, into LN_F bwd),
//        grad_WHEAD = q(gO)^T@q(hf) (wgrad, operand-quantized). The CE is UNCHANGED.
#ifndef S1_HEAD_MODE
#define S1_HEAD_MODE  0
#endif

// attention softmax constants (== forge_attn)
#define S1_AT_DELTA_SHIFT 2 // (rowmax - s) >> 2  (Delta = 4)
#define S1_AT_P_FRAC      16 // probabilities in Q16

// CE constants (== morph LM CE path).
// S1_CE_ZSHIFT sets the softmax TEMPERATURE: the LUT index is (logit[pred]-logit[v]) >> zshift, so a
// LARGER zshift = flatter (hotter) distribution. Default 4 for BOTH head modes (HEAD_MODE=0 is then
// byte-identical to before). NOTE the dyadic head's logits are SMALLER (O(10-40)) than the max-plus
// head's (O(100)), so the softmax it sees at zshift=4 is hotter; an empirical sweep at a short horizon
// found zshift=4 still best (a sharper zshift over-commits early), so 4 is kept. Overridable via
// -DS1_CE_ZSHIFT for tuning per rig / horizon.
#ifndef S1_CE_ZSHIFT
#define S1_CE_ZSHIFT 4      // (logit[pred]-logit[v]) >> zshift  index into ce_lut
#endif

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
// FINAL LayerNorm (LN_F) weights — ONE per channel (not layer-major; there is a single final LN). gamma is
// a dyadic code (forge_layernorm wdecode), beta is plain int16, exactly like LN1/LN2.
static const int S1_OFF_LNF_G   = S1_OFF_LN2_B  + S1_AL16(S1_D);                        // [D] gamma codes
static const int S1_OFF_LNF_B   = S1_OFF_LNF_G  + S1_AL16(S1_D);                        // [D] beta int16
static const int S1_OFF_W_HEAD  = S1_OFF_LNF_B  + S1_AL16(S1_D);                        // [V][D] int16(HEAD_MODE0)/int8 code(HEAD_MODE1)
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
// LN_F fine-dyadic (e,nq) per token (single final LN): the head-bwd reconstructs invq from these.
static const int S1_OFF_LNF_STATS = S1_OFF_LN2_STATS + S1_AL16(S1_N_LAYERS * S1_L_SEQ * 2);
static const int S1_OFF_ATTN_P    = S1_OFF_LNF_STATS + S1_AL16(S1_L_SEQ * 2);
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
static const int S1_OFF_G_LNF_G   = S1_OFF_G_LN2_B   + S1_AL16(S1_N_LAYERS * S1_D);          // [D] LN_F gamma grad
static const int S1_OFF_G_LNF_B   = S1_OFF_G_LNF_G   + S1_AL16(S1_D);                        // [D] LN_F beta  grad
static const int S1_OFF_G_W_HEAD  = S1_OFF_G_LNF_B   + S1_AL16(S1_D);                        // [V*D]

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
static const int S1_OFF_WFP_LNF_G  = S1_OFF_WFP_LN2_B + S1_AL16(S1_N_LAYERS * S1_D);          // [D] LN_F gamma master (code, fine fixed-pt)
static const int S1_OFF_WFP_LNF_B  = S1_OFF_WFP_LNF_G + S1_AL16(S1_D);                        // [D] LN_F beta  master (value, int16)
static const int S1_OFF_WFP_W_HEAD = S1_OFF_WFP_LNF_B + S1_AL16(S1_D);                        // [V*D]

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
static const int S1_OFF_M_LNF_G    = S1_OFF_M_LN2_B   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_M_LNF_B    = S1_OFF_M_LNF_G   + S1_AL16(S1_D);
static const int S1_OFF_M_W_HEAD   = S1_OFF_M_LNF_B   + S1_AL16(S1_D);

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
static const int S1_OFF_V_LNF_G    = S1_OFF_V_LN2_B   + S1_AL16(S1_N_LAYERS * S1_D);
static const int S1_OFF_V_LNF_B    = S1_OFF_V_LNF_G   + S1_AL16(S1_D);
static const int S1_OFF_V_W_HEAD   = S1_OFF_V_LNF_B   + S1_AL16(S1_D);

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

// ----------------------------------------------------------------------------
// FINE FIXED-POINT MASTER for the dense PRE-DYADIC CODE weights (the TRAIN_ALL fix).
// ----------------------------------------------------------------------------
// THE PROBLEM the int16/unit-scale master (LR_SHIFT=0) has on the CODE backbone:
//   A code weight's deploy value is a dyadic ±2^k. Its int16 master at UNIT scale means the smallest
//   representable weight is 1 (=2^0) and the next is 2 (=2^1) — a 2x jump. A dyadic-Adam step on the unit
//   master is dw = m>>round(log2 v) >> 0 = sign(m)·O(1) ∈ {-1,0,+1}. So a single step either does NOTHING
//   (rounds to 0 — the weight never leaves its current exponent, and from a ZERO init it is stuck at 0
//   FOREVER: the code backbone is silently un-trained) OR flips the exponent by a full power of two (1->2,
//   2->4 — a runaway DOUBLE that compounds through the additive/max-plus chain into the int16 sat ceiling
//   -> uniform-logit collapse). Coarse-master = either dead or exploding; there is no NUDGE.
//
// THE FIX (forge_dyadic_adam.cpp's scheme, applied to the code masters): hold the code-weight master as a
// FULL int32 FIXED-POINT value W_fp_q = weight << Q_CODE (Q_CODE~12), do the dyadic-Adam step at a FINE
// LR (LR_SHIFT_CODE~12 so dw is a SMALL fraction of a code unit), and re-quantize the deploy code from the
// ROUNDED master value (W_fp_q + half) >> Q_CODE via dyad(). Now the master moves SMOOTHLY (accumulating
// many sub-unit nudges), and the deploy CODE only changes when the master GENUINELY crosses a dyad()
// power-of-2 boundary — no premature exponent ratchet, and a zero-init code can climb off zero gradually.
// VALUE weights (EMBED/W_HEAD/LN beta) keep the int16 unit master (they SCATTER, no fan-in blow-up, and
// already train STABLY in FREEZE_CODES) — only the is_code regions get the fine fixed-point master.
//
// Q_CODE: the fixed-point fractional bits of the code master. weight w -> stored w<<Q_CODE; deploy exp is
//   dyad( round(master / 2^Q_CODE) ). The dyadic-Adam step is ALREADY adaptive-NORMALIZED
//   (step = m>>round(log2 v) ≈ sign(m)·O(1) — the dyadic analog of m/√v), so one step nudges the master by
//   ~±1-2 master-LSBs = ±2^-Q_CODE weight units (LR_SHIFT_CODE=0 below — the normalized step IS the nudge;
//   shifting it further would round it to 0 and re-freeze the codes). Q_CODE then sets HOW MANY sub-unit
//   nudges it takes to cross a dyad() boundary: the first nonzero code appears when |master| rounds to >=1,
//   i.e. |master| >= 2^(Q_CODE-1), so a zero-init code climbs off zero in ~2^(Q_CODE-1)/step ≈ 2^(Q_CODE-1)
//   steps. Q_CODE=6 -> ~32 steps to the first 2^0 code, ~32 more per subsequent dyad doubling — fine enough
//   to be a NUDGE (no premature ratchet), coarse enough to actually move within an O(100)-step run. (Q=12
//   like forge_dyadic_adam would need ~2048 steps/boundary — dead at this horizon; the S1 grads are at a
//   single Q16-CE scale, not the multi-region accumulation forge_dyadic_adam's DA_LR_SHIFT=12 assumes.)
// LR_SHIFT_CODE: extra shift on the normalized step before it nudges the fixed-point master. 0 = the
//   normalized ±O(1) step adds directly (one master-LSB nudge); >0 re-rounds the small step toward 0 and
//   re-freezes (the unit-master LR_SHIFT=0 pathology, one fixed-point bit up). Keep 0.
// EXP_CLAMP (hypothesis 2): cap the per-step change of the deploy code EXPONENT to +/-1, so even if the
//   master jumps a boundary no single step can runaway-DOUBLE a weight (a damping safety on top of the
//   smooth master). 1 = clamp to +/-1 per step; 0 = no clamp.
// DEFAULT Q_CODE=6. UPDATED VERDICT (was "frozen-only / OPEN"): with the FINAL LayerNorm (S1_USE_LNF) +
// the code-master WEIGHT DECAY (S1_CODE_WD_SHIFT) BOTH on, the dense CODE backbone GENUINELY trains and
// BEATS FREEZE. The earlier negative was caused by TWO missing pieces, both now supplied:
//   (1) NO final LN -> the head read raw R, which the densifying GEMM drove into the int16 sat ceiling ->
//       uniform-logit collapse. FIX = S1_USE_LNF (head reads LN_F(R), scale-bounded). (charlm has lnf.)
//   (2) the TRAINING DYNAMICS ratcheted the code magnitudes up with no counter-force, so even with a
//       bounded head the RAW residual still saturated (~72% of R clipped at Q=6, peak|R|=32768). FIX =
//       S1_CODE_WD_SHIFT (multiplier-free weight decay on the code master) -> peak|R| stays ~1.5k, satR=0.
// HONEST MEASURED at the defaults (Q=6, WD_SHIFT=4, USE_LNF=1; forge_train_step_s1_tb TRAIN_ALL block):
//   TRAIN_ALL CE 4.848 -> 2.067 (drop 2.78 nats), peak|R|=1516 (NO saturation), 13.7k code bytes moved,
//   and the DECISIVE ablation TRAIN_ALL beats FREEZE(same init) by -0.038 nats MEAN over 6 seeds (6/6 win,
//   margin GROWS with horizon). Removing EITHER fix reverts to the old failure: USE_LNF=0 -> TA-FRZ=+0.75
//   (codes HURT, saturated); WD_SHIFT=0 -> codes move but saturate and only MATCH freeze (TA-FRZ≈0).
//   The benefit is MODEST (the dense block on this small memorization task adds ~0.04 nats over a strong
//   embed/head baseline) but it is REAL, consistent, and un-fakeable (ablation, not a saturating proxy).
// Q_CODE choice: Q=6 -> ~32 master-steps to cross the first dyad() boundary (codes move within an O(100-400)
//   step run). Q=10 -> codes barely move in 120 steps (TRAIN_ALL ~= FREEZE, safe-but-inert). With WD+LNF
//   the Q=6 saturation that previously hurt is GONE, so Q=6 is now the productive default.
#ifndef S1_Q_CODE
#define S1_Q_CODE         6
#endif
#ifndef S1_LR_SHIFT_CODE
#define S1_LR_SHIFT_CODE  0
#endif
#ifndef S1_CODE_EXP_CLAMP
#define S1_CODE_EXP_CLAMP 1
#endif
// CODE-MASTER WEIGHT DECAY (hypothesis 3 = scale control on the TRAINING DYNAMICS, not just the head).
// MEASURED root cause of the residual saturation in TRAIN_ALL: it is NOT the dense codes per se (a FROZEN
// dense init does NOT saturate — peak|R| grows smoothly with density and stays < the int16 ceiling). It is
// the TRAINING DYNAMICS: dyadic-Adam's ±O(1) normalized step + the dyad() round-up-at-1.5*2^k bias is an
// UPWARD-biased random walk on the code exponents with NO counter-force, so over many steps the magnitudes
// ratchet up and drive R into sat16 (~72% of R clipped at Q=6). The fine master + exp-clamp slow a SINGLE
// step but do not remove the cumulative drift. A multiplier-free WEIGHT DECAY supplies the missing
// counter-force: each step, pull the fixed-point code master toward zero by a shift fraction
//   wq -= wq >> S1_CODE_WD_SHIFT     (a right-shift -> shift+sub only, DSP=0, bit-exact)
// so a code only STAYS large if the gradient keeps pushing it there. 0 = OFF (no decay). Larger = weaker
// decay (e.g. 6 ≈ 1.5%/step). Applies to is_code masters ONLY (the value weights scatter, no fan-in drift).
#ifndef S1_CODE_WD_SHIFT
#define S1_CODE_WD_SHIFT  4
#endif

// mode-2 a1 selector: a normal Adam step (default) vs init-only re-quantize (grad ignored, Adam no-op).
#define S1_OPT_STEP    0      // a1=0 -> full dyadic-Adam step + re-quantize + grad-zero
#define S1_OPT_INIT    1      // a1=1 -> re-quantize codes/values from the staged W_fp ONLY (no weight move)

// mode-2/3 a2 selector: which weights the optimizer MOVES.
//   S1_OPT_TRAIN_ALL    : every learnable region steps (the dense PRE-DYADIC CODE backbone INCLUDED).
//   S1_OPT_FREEZE_CODES : the dense code weights (is_code=1: Wq/Wk/Wv/Wo/W_up/W_down/LN1_g/LN2_g/LN_F_g)
//        are FROZEN — grad dropped, master+code untouched — and only the SCATTER/VALUE weights
//        (EMBED, W_HEAD, LN1_b/LN2_b/LN_F_b) train. The QAT "frozen-quantized-backbone, train the
//        embeddings + head" regime; kept as the conservative baseline (always stable).
//   HISTORY: TRAIN_ALL originally COLLAPSED — the all-trainable dyadic re-quantize ratcheted the code
//        magnitudes up and, with NO final LN, drove the head-input residual into the int16 sat ceiling ->
//        uniform logits (the old "all-trainable convergence was a saturation artifact" note). That is now
//        FIXED by S1_USE_LNF (bounded head input) + S1_CODE_WD_SHIFT (weight decay counters the ratchet):
//        TRAIN_ALL now trains WITHOUT collapse/saturation AND beats FREEZE in the ablation (see S1_Q_CODE).
//        FREEZE_CODES remains a valid, always-safe mode; TRAIN_ALL is the now-working full-stack trainer.
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
#define S1_NUM_OPT_REGIONS 14
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
    { S1_OFF_G_LNF_G,   S1_OFF_WFP_LNF_G,   S1_OFF_M_LNF_G,   S1_OFF_V_LNF_G,   S1_OFF_LNF_G,   S1_D,                        1 }, // LN_F g code
    { S1_OFF_G_LNF_B,   S1_OFF_WFP_LNF_B,   S1_OFF_M_LNF_B,   S1_OFF_V_LNF_B,   S1_OFF_LNF_B,   S1_D,                        0 }, // LN_F b value
    // W_HEAD: VALUE (is_code=0) under the legacy max-plus head (S1_HEAD_MODE=0); a DYADIC CODE (is_code=1)
    // under the dyadic-GEMM head (S1_HEAD_MODE=1) — flipping is_code makes the optimizer re-quantize it as a
    // code (the master is held at scale 2^Q_CODE, deploy = s1_encode_fine(dyad(W_fp))), exactly like Wq.
    { S1_OFF_G_W_HEAD,  S1_OFF_WFP_W_HEAD,  S1_OFF_M_W_HEAD,  S1_OFF_V_W_HEAD,  S1_OFF_W_HEAD,  S1_V * S1_D,                 S1_HEAD_MODE }, // W_HEAD value(0)/code(1)
};

#endif  // FORGE_S1_CONFIG_HPP
