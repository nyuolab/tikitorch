// forge_buffer_muzero_train.hpp — buffer layout for the ON-CHIP TRAINING of the
// dyadic Tropical MuZero net (the backward + SignSGD step).
//
// WHY a second layout. The inference kernel (forge_buffer_muzero.hpp) stores each
// weight as a FROZEN (sign, exponent) shift schedule — the deploy artifact. Training
// needs a *trainable* representation, and the dyadic constraint forbids the obvious one
// (nudge a float weight, re-quantize) because on-chip re-quantization would need a log2.
//
// THE TRAINABLE REPRESENTATION. Each weight is a plain int16 SHADOW value `w`. The
// forward quantizes it on the fly to the nearest signed power of two q(w) = sgn(w)*2^e
// (e = rounded-MSB position — a leading-bit search + compare, NOT a multiply), so the
// matmul stays shift+add (zero DSP). The update is SignSGD on the shadow:
//     w <- clamp16( w - lr * sign(dL/dw) ),   sign(dL/dw) = sign(g_out) (xor) sign(x)
// a 1-bit sign-product + an integer add — multiplier-free. Weights cross zero (and thus
// flip sign) naturally as the shadow passes through 0. This is the standard
// straight-through pow2 recipe (what tikitorch.nn.dyadic_quantize does in float),
// re-expressed in integers so forward AND backward are zero-DSP.
//
// DEPLOY BRIDGE. After training, q(shadow) IS the (sign, exponent) schedule — export it
// into the inference buffer (forge_buffer_muzero.hpp) with the SAME nearest-pow2 rounding
// (tikitorch.fpga.dyadic_export / nearest_pow2 below) and run the frozen forward kernel.
//
// One flat int32 buffer (int16-in-int32). Shapes are shared with the inference layout.

#ifndef FORGE_BUFFER_MUZERO_TRAIN_HPP
#define FORGE_BUFFER_MUZERO_TRAIN_HPP

#include "forge_buffer_muzero.hpp"   // MZ_OBS, MZ_L, MZ_A, MZ_DR, MZ_DP + forge_hls_types
#include <cstdint>

// --- trainable int16 shadow weight matrices [out,in] (one int32 word each) ---
constexpr int TW_R1 = MZ_DR * MZ_OBS;   // repr.enc.lin1  (OBS->DR)
constexpr int TW_R2 = MZ_L  * MZ_DR;    // repr.enc.lin2  (DR->L)
constexpr int TW_P1 = MZ_DP * MZ_L;     // pred.mix.lin1  (L->DP)
constexpr int TW_P2 = MZ_L  * MZ_DP;    // pred.mix.lin2  (DP->L)
constexpr int TW_PH = MZ_A  * MZ_L;     // policy head    (L->A)
constexpr int TW_VH = 1     * MZ_L;     // value head     (L->1)

// --- cumulative offsets: weights, biases, io, targets, outputs ---
constexpr int TOFF_W_R1 = 0;
constexpr int TOFF_W_R2 = TOFF_W_R1 + TW_R1;
constexpr int TOFF_W_P1 = TOFF_W_R2 + TW_R2;
constexpr int TOFF_W_P2 = TOFF_W_P1 + TW_P1;
constexpr int TOFF_W_PH = TOFF_W_P2 + TW_P2;
constexpr int TOFF_W_VH = TOFF_W_PH + TW_PH;
constexpr int TOFF_B_R1 = TOFF_W_VH + TW_VH;
constexpr int TOFF_B_R2 = TOFF_B_R1 + MZ_DR;
constexpr int TOFF_B_P1 = TOFF_B_R2 + MZ_L;
constexpr int TOFF_B_P2 = TOFF_B_P1 + MZ_DP;
constexpr int TOFF_B_PH = TOFF_B_P2 + MZ_L;
constexpr int TOFF_B_VH = TOFF_B_PH + MZ_A;
constexpr int TOFF_OBS  = TOFF_B_VH + 1;        // input observation  [OBS]
constexpr int TOFF_PTGT = TOFF_OBS  + MZ_OBS;   // policy target       [A]
constexpr int TOFF_VTGT = TOFF_PTGT + MZ_A;     // value  target       [1]
constexpr int TOFF_POL  = TOFF_VTGT + 1;        // forward policy out  [A]
constexpr int TOFF_VAL  = TOFF_POL  + MZ_A;     // forward value  out  [1]
constexpr int TOFF_LOSS = TOFF_VAL  + 1;        // scalar L1 loss      [1]

constexpr int MZT_BUFFER_WORDS = TOFF_LOSS + 1;
constexpr int MZT_BUFFER_BYTES = MZT_BUFFER_WORDS * 4;

#endif  // FORGE_BUFFER_MUZERO_TRAIN_HPP
