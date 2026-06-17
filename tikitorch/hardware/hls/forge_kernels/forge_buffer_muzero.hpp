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
// Base dims are overridable at compile time so the SAME kernel synthesizes at gridworld scale (default)
// OR chess scale: -DMZ_OBS_OVERRIDE=769 -DMZ_L_OVERRIDE=64 -DMZ_A_OVERRIDE=4096 (matches
// tikitorch.planning.muzero.chess_game: 769-d obs, latent 64, 4096 from-to actions).
#ifdef MZ_OBS_OVERRIDE
constexpr int MZ_OBS  = MZ_OBS_OVERRIDE;
#else
constexpr int MZ_OBS  = 48;                 // 3 * 4 * 4 one-hot gridworld obs
#endif
#ifdef MZ_L_OVERRIDE
constexpr int MZ_L    = MZ_L_OVERRIDE;
#else
constexpr int MZ_L    = 32;                 // latent_dim
#endif
#ifdef MZ_A_OVERRIDE
constexpr int MZ_A    = MZ_A_OVERRIDE;
#else
constexpr int MZ_A    = 4;                  // n_actions
#endif
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

// --- dynamics (mode 1): g(h,a) -> (h', r). action gather + 2 dyadic residual blocks + reward head.
// Each core[d] is a DyadicTropicalProjection(L,L) = lin1(L->DD) -> relu -> lin2(DD->L); the residual
// block is z = submean(z + core(z)) (submean replaces TropicalLayerNorm, exactly as repr/pred substitute
// it for the norm + _minmax_unit on fabric). Reward head is a dyadic L->1 readout. ZERO DSP. ---
constexpr int MZ_DD = 2 * MZ_L;             // dynamics core d_intermediate = 2*max(L,L)
MZ_WMAT(D0A, MZ_DD, MZ_L)                   // core[0].lin1 (L -> DD)
MZ_WMAT(D0B, MZ_L,  MZ_DD)                  // core[0].lin2 (DD -> L)
MZ_WMAT(D1A, MZ_DD, MZ_L)                   // core[1].lin1
MZ_WMAT(D1B, MZ_L,  MZ_DD)                  // core[1].lin2
MZ_WMAT(RH,  1,     MZ_L)                   // reward head (L -> 1)

constexpr int WORDS_B_R1 = MZ_DR, WORDS_B_R2 = MZ_L;     // biases (int16-in-int32)
constexpr int WORDS_B_P1 = MZ_DP, WORDS_B_P2 = MZ_L;
constexpr int WORDS_B_PH = MZ_A,  WORDS_B_VH = 1;
constexpr int WORDS_AEMB = MZ_A * MZ_L;                  // action-embedding table (gather, int16-in-int32)
constexpr int WORDS_B_D0A = MZ_DD, WORDS_B_D0B = MZ_L;
constexpr int WORDS_B_D1A = MZ_DD, WORDS_B_D1B = MZ_L, WORDS_B_RH = 1;
constexpr int WORDS_REWARD = 1;                          // dynamics reward output

constexpr int WORDS_OBS    = MZ_OBS;     // input obs
constexpr int WORDS_H      = MZ_L;       // latent after repr
constexpr int WORDS_POLICY = MZ_A;       // output policy logits
constexpr int WORDS_VALUE  = 1;          // output value
constexpr int WORDS_TMP_DR = MZ_DR;      // intermediate (repr lin1 -> relu)
constexpr int WORDS_TMP_DP = MZ_DP;      // intermediate (pred lin1 -> relu)
constexpr int WORDS_TMP_L  = MZ_L;       // intermediate (pred mix output)

// --- cumulative offsets ---
// Every region base is rounded UP to a 16-word (512-bit) boundary so the kernel can read each
// region with aligned 512-bit AXI bursts from HBM (chess weights are ~12 MB -> can't be resident;
// they stream from HBM in 16-word beats). R16 leaves small inter-region gaps (allocated padding the
// last beat of a region may harmlessly overread into). The host export reads OFF_* via the kernel's
// layout introspection, so it tracks these automatically.
constexpr int R16(int x) { return (x + 15) & ~15; }
constexpr int OFF_R1_EXP = 0;
constexpr int OFF_R1_SGN = R16(OFF_R1_EXP + WORDS_R1_EXP);
constexpr int OFF_R2_EXP = R16(OFF_R1_SGN + WORDS_R1_SGN);
constexpr int OFF_R2_SGN = R16(OFF_R2_EXP + WORDS_R2_EXP);
constexpr int OFF_P1_EXP = R16(OFF_R2_SGN + WORDS_R2_SGN);
constexpr int OFF_P1_SGN = R16(OFF_P1_EXP + WORDS_P1_EXP);
constexpr int OFF_P2_EXP = R16(OFF_P1_SGN + WORDS_P1_SGN);
constexpr int OFF_P2_SGN = R16(OFF_P2_EXP + WORDS_P2_EXP);
constexpr int OFF_PH_EXP = R16(OFF_P2_SGN + WORDS_P2_SGN);
constexpr int OFF_PH_SGN = R16(OFF_PH_EXP + WORDS_PH_EXP);
constexpr int OFF_VH_EXP = R16(OFF_PH_SGN + WORDS_PH_SGN);
constexpr int OFF_VH_SGN = R16(OFF_VH_EXP + WORDS_VH_EXP);
constexpr int OFF_B_R1   = R16(OFF_VH_SGN + WORDS_VH_SGN);
constexpr int OFF_B_R2   = R16(OFF_B_R1   + WORDS_B_R1);
constexpr int OFF_B_P1   = R16(OFF_B_R2   + WORDS_B_R2);
constexpr int OFF_B_P2   = R16(OFF_B_P1   + WORDS_B_P1);
constexpr int OFF_B_PH   = R16(OFF_B_P2   + WORDS_B_P2);
constexpr int OFF_B_VH   = R16(OFF_B_PH   + WORDS_B_PH);
constexpr int OFF_OBS    = R16(OFF_B_VH   + WORDS_B_VH);
constexpr int OFF_H      = R16(OFF_OBS    + WORDS_OBS);
constexpr int OFF_TMP_DR = R16(OFF_H      + WORDS_H);
constexpr int OFF_TMP_DP = R16(OFF_TMP_DR + WORDS_TMP_DR);
constexpr int OFF_TMP_L  = R16(OFF_TMP_DP + WORDS_TMP_DP);
constexpr int OFF_POLICY = R16(OFF_TMP_L  + WORDS_TMP_L);
constexpr int OFF_VALUE  = R16(OFF_POLICY + WORDS_POLICY);

// --- dynamics regions (APPENDED after the forward IO so OFF_R1..OFF_VALUE are an unchanged prefix:
// the already-routed FORWARD AFI's buffer is exactly this buffer's head). ---
constexpr int OFF_AEMB    = R16(OFF_VALUE   + WORDS_VALUE);
constexpr int OFF_D0A_EXP = R16(OFF_AEMB    + WORDS_AEMB);
constexpr int OFF_D0A_SGN = R16(OFF_D0A_EXP + WORDS_D0A_EXP);
constexpr int OFF_D0B_EXP = R16(OFF_D0A_SGN + WORDS_D0A_SGN);
constexpr int OFF_D0B_SGN = R16(OFF_D0B_EXP + WORDS_D0B_EXP);
constexpr int OFF_D1A_EXP = R16(OFF_D0B_SGN + WORDS_D0B_SGN);
constexpr int OFF_D1A_SGN = R16(OFF_D1A_EXP + WORDS_D1A_EXP);
constexpr int OFF_D1B_EXP = R16(OFF_D1A_SGN + WORDS_D1A_SGN);
constexpr int OFF_D1B_SGN = R16(OFF_D1B_EXP + WORDS_D1B_EXP);
constexpr int OFF_RH_EXP  = R16(OFF_D1B_SGN + WORDS_D1B_SGN);
constexpr int OFF_RH_SGN  = R16(OFF_RH_EXP  + WORDS_RH_EXP);
constexpr int OFF_B_D0A   = R16(OFF_RH_SGN  + WORDS_RH_SGN);
constexpr int OFF_B_D0B   = R16(OFF_B_D0A   + WORDS_B_D0A);
constexpr int OFF_B_D1A   = R16(OFF_B_D0B   + WORDS_B_D0B);
constexpr int OFF_B_D1B   = R16(OFF_B_D1A   + WORDS_B_D1A);
constexpr int OFF_B_RH    = R16(OFF_B_D1B   + WORDS_B_D1B);
constexpr int OFF_REWARD  = R16(OFF_B_RH    + WORDS_B_RH);

constexpr int MZ_BUFFER_WORDS = R16(OFF_REWARD + WORDS_REWARD);  // total padded to a whole beat
constexpr int MZ_BUFFER_BYTES = MZ_BUFFER_WORDS * 4;

// Silicon bug-class guard: every weight/IO region base MUST be 16-word aligned for the 512-bit burst
// (a misaligned base reads the wrong words on the AFI — the silicon-only bug the morph-LM CE-LUT hit).
static_assert(OFF_R1_EXP % 16 == 0 && OFF_R1_SGN % 16 == 0 && OFF_R2_EXP % 16 == 0 && OFF_R2_SGN % 16 == 0, "R-region bases must be 16-aligned");
static_assert(OFF_P1_EXP % 16 == 0 && OFF_P1_SGN % 16 == 0 && OFF_P2_EXP % 16 == 0 && OFF_P2_SGN % 16 == 0, "P-region bases must be 16-aligned");
static_assert(OFF_PH_EXP % 16 == 0 && OFF_PH_SGN % 16 == 0 && OFF_VH_EXP % 16 == 0 && OFF_VH_SGN % 16 == 0, "head bases must be 16-aligned");
static_assert(OFF_OBS % 16 == 0 && OFF_POLICY % 16 == 0 && OFF_VALUE % 16 == 0, "IO bases must be 16-aligned");
static_assert(OFF_AEMB % 16 == 0 && OFF_D0A_EXP % 16 == 0 && OFF_D0A_SGN % 16 == 0 && OFF_D0B_EXP % 16 == 0 && OFF_D0B_SGN % 16 == 0, "dyn core0 bases must be 16-aligned");
static_assert(OFF_D1A_EXP % 16 == 0 && OFF_D1A_SGN % 16 == 0 && OFF_D1B_EXP % 16 == 0 && OFF_D1B_SGN % 16 == 0, "dyn core1 bases must be 16-aligned");
static_assert(OFF_RH_EXP % 16 == 0 && OFF_RH_SGN % 16 == 0 && OFF_REWARD % 16 == 0, "reward head/output bases must be 16-aligned");

#endif  // FORGE_BUFFER_MUZERO_HPP
