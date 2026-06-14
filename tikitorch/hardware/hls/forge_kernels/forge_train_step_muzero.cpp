// forge_train_step_muzero.cpp — DYADIC Tropical MuZero kernel (Tier-1, FORWARD inference).
//
// mode 0 -> do_fwd: obs -> repr -> pred -> (policy, value), every "multiply" a SIGN + bit-SHIFT
//                   on an int32 add-tree (dyadic weights = signed powers of two). ZERO DSP.
// This is the FROZEN forward-inference kernel (the deploy artifact): weights are a fixed
// (sign, exponent) shift schedule. The on-chip TRAINING step (forward + backward + SignSGD,
// zero-DSP in the backward too) is the sibling kernel forge_train_step_muzero_train.cpp, which
// trains int16 shadow weights and exports their nearest-pow2 schedule into THIS buffer to deploy.
//
// CSIM:  g++ -O2 -std=c++17 -DCSIM_ONLY -I.. -I. forge_train_step_muzero.cpp \
//            forge_train_step_muzero_tb.cpp -o tb_muzero && ./tb_muzero
#include "forge_buffer_muzero.hpp"
#include <cstdint>

// sgn * (x << exp); exp<0 => arithmetic right shift. A bit-shift + sign — NO multiplier.
static inline acc_t dyadic_mac(data_t x, int exp, int sgn) {
    acc_t v = (exp >= 0) ? ((acc_t)x << exp) : ((acc_t)x >> (-exp));
    return sgn >= 0 ? v : (acc_t)(-v);   // explicit cast: ap_int negation widens to <33>, so the ?: needs it
}

static inline data_t sat16(acc_t a) {
    if (a > 32767) a = 32767;
    if (a < -32768) a = -32768;
    return (data_t)a;
}

// 512-bit (16-word) aligned burst read of n_words from buffer[word_off]. word_off MUST be 16-aligned.
// Under Vitis this is one wide AXI beat per 16 words (HBM streaming); under g++ a plain copy (CSIM).
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
typedef ap_uint<512> wide_t;
#endif
static inline void burst_load16(int32_t* buffer, int word_off, int32_t* dst) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wb = reinterpret_cast<wide_t*>(buffer);
    wide_t beat = wb[word_off >> 4];
LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
        dst[lane] = (int32_t)(int)(ap_int<32>)beat.range(lane * 32 + 31, lane * 32);
    }
#else
    for (int lane = 0; lane < 16; lane++) dst[lane] = buffer[word_off + lane];
#endif
}

// y[o] = sat16( bias[o] + sum_i dyadic_mac(x[i], exp[o,i], sgn[o,i]) ) — a shift+add-tree matmul.
// STREAMING: the weights (~MB) live in HBM and are read as flat 16-word aligned bursts (off_exp/off_sgn
// must be 16-aligned); only the small activations x/y stay on-chip. Bit-exact to the scalar matmul:
// the flat index idx == o*IN+i is read in the same order, just 16 words per AXI beat.
static void dyadic_linear(int32_t* buffer, int off_exp, int off_sgn, const int32_t* bias,
                          const data_t* x, data_t* y, int OUT, int IN) {
    int32_t eb[16], sb[16];
    int lane = 16;                              // force a beat load on the first iteration
    int o = 0, i = 0, total = OUT * IN;
    acc_t acc = (acc_t)bias[0];
    DL: for (int idx = 0; idx < total; idx++) {
#pragma HLS PIPELINE II=1
        if (lane == 16) { burst_load16(buffer, off_exp + idx, eb); burst_load16(buffer, off_sgn + idx, sb); lane = 0; }
        acc += dyadic_mac(x[i], eb[lane], sb[lane]);
        lane++; i++;
        if (i == IN) { y[o] = sat16(acc); o++; i = 0; if (o < OUT) acc = (acc_t)bias[o]; }
    }
}

static inline void relu(data_t* a, int n) {              // tropical max(.,0)
    for (int i = 0; i < n; i++) if (a[i] < 0) a[i] = 0;
}

// subtract-mean normalization (closure-clean BatchShift). n=MZ_L=32 is a power of two, so the
// /n is a right-shift on fabric (a constant-pow2 divide), not a DSP divide.
static void submean(data_t* a, int n) {
    acc_t s = 0;
    for (int i = 0; i < n; i++) s += a[i];
    data_t m = (data_t)(s / n);
    for (int i = 0; i < n; i++) a[i] = sat16((acc_t)a[i] - m);
}

static int do_fwd(int32_t* buffer) {
    data_t obs[MZ_OBS], tdr[MZ_DR], h[MZ_L], tdp[MZ_DP], z[MZ_L], pol[MZ_A], val[1];
    for (int i = 0; i < MZ_OBS; i++) obs[i] = (data_t)buffer[OFF_OBS + i];

    // representation: DyadicLin1 (OBS->DR) -> relu -> DyadicLin2 (DR->L) -> subtract-mean
    dyadic_linear(buffer, OFF_R1_EXP, OFF_R1_SGN, &buffer[OFF_B_R1], obs, tdr, MZ_DR, MZ_OBS);
    relu(tdr, MZ_DR);
    dyadic_linear(buffer, OFF_R2_EXP, OFF_R2_SGN, &buffer[OFF_B_R2], tdr, h, MZ_L, MZ_DR);
    submean(h, MZ_L);
    for (int i = 0; i < MZ_L; i++) buffer[OFF_H + i] = h[i];

    // prediction: DyadicLin1 (L->DP) -> relu -> DyadicLin2 (DP->L) -> dyadic heads
    dyadic_linear(buffer, OFF_P1_EXP, OFF_P1_SGN, &buffer[OFF_B_P1], h, tdp, MZ_DP, MZ_L);
    relu(tdp, MZ_DP);
    dyadic_linear(buffer, OFF_P2_EXP, OFF_P2_SGN, &buffer[OFF_B_P2], tdp, z, MZ_L, MZ_DP);
    for (int i = 0; i < MZ_L; i++) buffer[OFF_TMP_L + i] = z[i];

    dyadic_linear(buffer, OFF_PH_EXP, OFF_PH_SGN, &buffer[OFF_B_PH], z, pol, MZ_A, MZ_L);
    dyadic_linear(buffer, OFF_VH_EXP, OFF_VH_SGN, &buffer[OFF_B_VH], z, val, 1, MZ_L);
    for (int i = 0; i < MZ_A; i++) buffer[OFF_POLICY + i] = pol[i];
    buffer[OFF_VALUE] = val[0];
    return 0;
}

extern "C" int forge_train_step_muzero(int32_t* buffer, int mode, int arg1, int arg2, int arg3) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=mode
#pragma HLS INTERFACE s_axilite port=arg1
#pragma HLS INTERFACE s_axilite port=arg2
#pragma HLS INTERFACE s_axilite port=arg3
#pragma HLS INTERFACE s_axilite port=return
    (void)arg1; (void)arg2; (void)arg3;
    if (mode == 0) return do_fwd(buffer);
    return -1;   // bwd / signsgd / dynamics-unroll: next stage
}

// Layout introspection for the host bit-exact harness (so a Python reference cannot
// silently drift from this buffer layout). Not part of the silicon ABI.
extern "C" int forge_muzero_buffer_words() { return MZ_BUFFER_WORDS; }
extern "C" void forge_muzero_layout(int* out) {
    // [OBS, L, A, DR, DP, OFF_OBS, OFF_H, OFF_TMP_L, OFF_POLICY, OFF_VALUE,
    //  OFF_R1_EXP, OFF_R1_SGN, OFF_R2_EXP, OFF_R2_SGN, OFF_P1_EXP, OFF_P1_SGN,
    //  OFF_P2_EXP, OFF_P2_SGN, OFF_PH_EXP, OFF_PH_SGN, OFF_VH_EXP, OFF_VH_SGN,
    //  OFF_B_R1, OFF_B_R2, OFF_B_P1, OFF_B_P2, OFF_B_PH, OFF_B_VH]
    int v[] = {MZ_OBS, MZ_L, MZ_A, MZ_DR, MZ_DP, OFF_OBS, OFF_H, OFF_TMP_L, OFF_POLICY, OFF_VALUE,
               OFF_R1_EXP, OFF_R1_SGN, OFF_R2_EXP, OFF_R2_SGN, OFF_P1_EXP, OFF_P1_SGN,
               OFF_P2_EXP, OFF_P2_SGN, OFF_PH_EXP, OFF_PH_SGN, OFF_VH_EXP, OFF_VH_SGN,
               OFF_B_R1, OFF_B_R2, OFF_B_P1, OFF_B_P2, OFF_B_PH, OFF_B_VH};
    for (int i = 0; i < (int)(sizeof(v) / sizeof(int)); i++) out[i] = v[i];
}
