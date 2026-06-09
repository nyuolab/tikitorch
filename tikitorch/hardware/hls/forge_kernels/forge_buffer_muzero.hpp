// forge_buffer_muzero.hpp — buffer layout for the DYADIC Tropical MuZero kernel.
//
// The MuZero initial-inference path obs -> repr -> pred -> (policy, value), with the
// repr encoder and pred mixer as DYADIC projections (tikitorch.nn.DyadicTropicalProjection):
// two dyadic-linear stages (lin1, lin2) with a ReLU between. A dyadic weight is a SIGNED
// POWER OF TWO, so every "multiply" is a sign + bit-SHIFT fed to an int32 add-tree —
// ZERO DSP. Weights are stored as the (sign, exponent) SHIFT SCHEDULE produced by
// tikitorch.fpga.dyadic_export.export_net (verified bit-exact against the trained net).
//
// One flat int32 buffer (v6 wide-AXI ABI); int16 activations, int32 accumulators (the
// forge_hls_types.hpp shim maps these to ap_int<16>/<32> under Vitis, plain ints under g++).
//
// FORWARD-INFERENCE kernel (mode 0). Training (bwd + signsgd, dynamics unroll) is the
// documented next stage; the heads (policy/value) are dyadic here too so the WHOLE path is
// zero-DSP (the deploy choice — the float net's nn.Linear heads are dyadicized at export).

#ifndef FORGE_BUFFER_MUZERO_HPP
#define FORGE_BUFFER_MUZERO_HPP

#include "../forge_hls_types.hpp"
#include <cstdint>

// --- shape (matches a TropicalMuZeroNet(obs_dim=MZ_OBS, n_actions=MZ_A, latent_dim=MZ_L,
//     core_kind="dyadic") on the 4x4 GridWorld example; d_intermediate = 2*max(in,out)) ---
constexpr int MZ_OBS  = 48;                 // 3 * 4 * 4 one-hot gridworld obs
constexpr int MZ_L    = 32;                 // latent_dim
constexpr int MZ_A    = 4;                  // n_actions
constexpr int MZ_DR   = 2 * (MZ_OBS > MZ_L ? MZ_OBS : MZ_L);   // repr.enc d_intermediate = 96
constexpr int MZ_DP   = 2 * MZ_L;                              // pred.mix d_intermediate = 64

// A dyadic weight matrix [out,in] is stored as TWO int32 arrays: EXP (the shift amount,
// signed: >=0 left-shift, <0 right-shift) and SGN (+1/-1; 0 => weight is exactly 0).
#define MZ_WMAT(name, OUT, IN) \
    constexpr int WORDS_##name##_EXP = (OUT) * (IN); \
    constexpr int WORDS_##name##_SGN = (OUT) * (IN);

// repr: enc.lin1 (OBS->DR), enc.lin2 (DR->L); pred: mix.lin1 (L->DP), mix.lin2 (DP->L)
MZ_WMAT(R1, MZ_DR, MZ_OBS)
MZ_WMAT(R2, MZ_L,  MZ_DR)
MZ_WMAT(P1, MZ_DP, MZ_L)
MZ_WMAT(P2, MZ_L,  MZ_DP)
// heads (dyadic too): policy (A->... actually L->A) and value (L->1)
MZ_WMAT(PH, MZ_A, MZ_L)
MZ_WMAT(VH, 1,    MZ_L)

constexpr int WORDS_B_R1 = MZ_DR, WORDS_B_R2 = MZ_L;     // biases (int16-in-int32)
constexpr int WORDS_B_P1 = MZ_DP, WORDS_B_P2 = MZ_L;
constexpr int WORDS_B_PH = MZ_A,  WORDS_B_VH = 1;

constexpr int WORDS_OBS    = MZ_OBS;     // input obs
constexpr int WORDS_H      = MZ_L;       // latent after repr
constexpr int WORDS_POLICY = MZ_A;       // output policy logits
constexpr int WORDS_VALUE  = 1;          // output value
constexpr int WORDS_TMP_DR = MZ_DR;      // intermediate (repr lin1 -> relu)
constexpr int WORDS_TMP_DP = MZ_DP;      // intermediate (pred lin1 -> relu)
constexpr int WORDS_TMP_L  = MZ_L;       // intermediate (pred mix output)

// --- cumulative offsets ---
constexpr int OFF_R1_EXP = 0;
constexpr int OFF_R1_SGN = OFF_R1_EXP + WORDS_R1_EXP;
constexpr int OFF_R2_EXP = OFF_R1_SGN + WORDS_R1_SGN;
constexpr int OFF_R2_SGN = OFF_R2_EXP + WORDS_R2_EXP;
constexpr int OFF_P1_EXP = OFF_R2_SGN + WORDS_R2_SGN;
constexpr int OFF_P1_SGN = OFF_P1_EXP + WORDS_P1_EXP;
constexpr int OFF_P2_EXP = OFF_P1_SGN + WORDS_P1_SGN;
constexpr int OFF_P2_SGN = OFF_P2_EXP + WORDS_P2_EXP;
constexpr int OFF_PH_EXP = OFF_P2_SGN + WORDS_P2_SGN;
constexpr int OFF_PH_SGN = OFF_PH_EXP + WORDS_PH_EXP;
constexpr int OFF_VH_EXP = OFF_PH_SGN + WORDS_PH_SGN;
constexpr int OFF_VH_SGN = OFF_VH_EXP + WORDS_VH_EXP;
constexpr int OFF_B_R1   = OFF_VH_SGN + WORDS_VH_SGN;
constexpr int OFF_B_R2   = OFF_B_R1   + WORDS_B_R1;
constexpr int OFF_B_P1   = OFF_B_R2   + WORDS_B_R2;
constexpr int OFF_B_P2   = OFF_B_P1   + WORDS_B_P1;
constexpr int OFF_B_PH   = OFF_B_P2   + WORDS_B_P2;
constexpr int OFF_B_VH   = OFF_B_PH   + WORDS_B_PH;
constexpr int OFF_OBS    = OFF_B_VH   + WORDS_B_VH;
constexpr int OFF_H      = OFF_OBS    + WORDS_OBS;
constexpr int OFF_TMP_DR = OFF_H      + WORDS_H;
constexpr int OFF_TMP_DP = OFF_TMP_DR + WORDS_TMP_DR;
constexpr int OFF_TMP_L  = OFF_TMP_DP + WORDS_TMP_DP;
constexpr int OFF_POLICY = OFF_TMP_L  + WORDS_TMP_L;
constexpr int OFF_VALUE  = OFF_POLICY + WORDS_POLICY;

constexpr int MZ_BUFFER_WORDS = OFF_VALUE + WORDS_VALUE;
constexpr int MZ_BUFFER_BYTES = MZ_BUFFER_WORDS * 4;

#endif  // FORGE_BUFFER_MUZERO_HPP
