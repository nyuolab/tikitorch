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

// 512-bit (16-word) aligned burst read of n_words (a multiple of 16) from buffer[word_off] (16-aligned)
// into an on-chip tile. One wide AXI beat per 16 words under Vitis (HBM streaming); plain copy under g++.
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
typedef ap_uint<512> wide_t;
#endif
static void burst_load_words(int32_t* buffer, int word_off, int32_t* dst, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wb = reinterpret_cast<wide_t*>(buffer);
    int n_wide = n_words >> 4, woff = word_off >> 4;
LOAD_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
        wide_t beat = wb[woff + w];
        for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            dst[w * 16 + lane] = (int32_t)(int)(ap_int<32>)beat.range(lane * 32 + 31, lane * 32);
        }
    }
#else
    for (int i = 0; i < n_words; i++) dst[i] = buffer[word_off + i];
#endif
}

// y[o] = sat16( bias[o] + sum_i dyadic_mac(x[i], exp[o,i], sgn[o,i]) ) — a shift+add-tree matmul.
// STREAMING (HBM): weights (~MB) don't fit on-chip, so read them in 16-word-aligned CHUNK tiles and run a
// CLEANLY-PIPELINED compute pass over each tile (II=1; the burst is its own pipelined pass). This is the
// fast form — the per-element conditional-reload version stalled the pipeline (~28 cyc/MAC). Bit-exact to
// the scalar matmul: flat idx == o*IN+i, same order. Only the small activations x/y stay on-chip.
static void dyadic_linear(int32_t* buffer, int off_exp, int off_sgn, const int32_t* bias,
                          const data_t* x, data_t* y, int OUT, int IN) {
    const int CH = 4096;                        // 16-word-multiple tile (16 KB/array on-chip)
    int32_t et[CH], st[CH];
    int o = 0, i = 0, total = OUT * IN;
    acc_t acc = (acc_t)bias[0];
    CHUNK: for (int base = 0; base < total; base += CH) {
        int n = (total - base < CH) ? (total - base) : CH;
        int nrd = (n + 15) & ~15;               // whole beats (last beat overreads into the 16-aligned gap)
        burst_load_words(buffer, off_exp + base, et, nrd);
        burst_load_words(buffer, off_sgn + base, st, nrd);
        COMPUTE: for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
            acc += dyadic_mac(x[i], et[k], st[k]);
            i++;
            if (i == IN) { y[o] = sat16(acc); o++; i = 0; if (o < OUT) acc = (acc_t)bias[o]; }
        }
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

// prediction half: latent h -> P1 -> relu -> P2 -> dyadic heads -> (policy, value) into the buffer.
// Shared by do_fwd (h from representation) and do_dyn (h' from dynamics) so a recurrent step also
// emits its (policy, value) — i.e. one kick yields the full MuZero recurrent_inference (h', r, p, v).
static void pred_from_h(int32_t* buffer, const data_t* h) {
    data_t tdp[MZ_DP], z[MZ_L], pol[MZ_A], val[1];
    dyadic_linear(buffer, OFF_P1_EXP, OFF_P1_SGN, &buffer[OFF_B_P1], h, tdp, MZ_DP, MZ_L);
    relu(tdp, MZ_DP);
    dyadic_linear(buffer, OFF_P2_EXP, OFF_P2_SGN, &buffer[OFF_B_P2], tdp, z, MZ_L, MZ_DP);
    for (int i = 0; i < MZ_L; i++) buffer[OFF_TMP_L + i] = z[i];
    dyadic_linear(buffer, OFF_PH_EXP, OFF_PH_SGN, &buffer[OFF_B_PH], z, pol, MZ_A, MZ_L);
    dyadic_linear(buffer, OFF_VH_EXP, OFF_VH_SGN, &buffer[OFF_B_VH], z, val, 1, MZ_L);
    for (int i = 0; i < MZ_A; i++) buffer[OFF_POLICY + i] = pol[i];
    buffer[OFF_VALUE] = val[0];
}

static int do_fwd(int32_t* buffer) {
    data_t obs[MZ_OBS], tdr[MZ_DR], h[MZ_L];
    for (int i = 0; i < MZ_OBS; i++) obs[i] = (data_t)buffer[OFF_OBS + i];

    // representation: DyadicLin1 (OBS->DR) -> relu -> DyadicLin2 (DR->L) -> subtract-mean
    dyadic_linear(buffer, OFF_R1_EXP, OFF_R1_SGN, &buffer[OFF_B_R1], obs, tdr, MZ_DR, MZ_OBS);
    relu(tdr, MZ_DR);
    dyadic_linear(buffer, OFF_R2_EXP, OFF_R2_SGN, &buffer[OFF_B_R2], tdr, h, MZ_L, MZ_DR);
    submean(h, MZ_L);
    for (int i = 0; i < MZ_L; i++) buffer[OFF_H + i] = h[i];
    pred_from_h(buffer, h);
    return 0;
}

// one tropical dyadic residual block: c = lin2(relu(lin1(z))); z = submean(z + c).
// (submean replaces TropicalLayerNorm on fabric, mirroring repr/pred — zero DSP throughout.)
static void dyn_block(int32_t* buffer, int off_a_exp, int off_a_sgn, const int32_t* bias_a,
                      int off_b_exp, int off_b_sgn, const int32_t* bias_b, data_t* z) {
    data_t u[MZ_DD], c[MZ_L];
    dyadic_linear(buffer, off_a_exp, off_a_sgn, bias_a, z, u, MZ_DD, MZ_L);
    relu(u, MZ_DD);
    dyadic_linear(buffer, off_b_exp, off_b_sgn, bias_b, u, c, MZ_L, MZ_DD);
    for (int i = 0; i < MZ_L; i++) z[i] = sat16((acc_t)z[i] + (acc_t)c[i]);   // residual add
    submean(z, MZ_L);
}

// dynamics g(h, a) -> (h', r): action-embedding gather + residual add + 2 dyadic residual blocks +
// reward head. Reads the running latent from OFF_H, writes h' back to OFF_H (so repeated mode-1 kicks
// chain the K-step unroll), reward to OFF_REWARD, and (via pred_from_h) policy/value of h'. ZERO DSP:
// the gather is a table lookup, the rest is shift+add+compare.
static int do_dyn(int32_t* buffer, int action) {
    data_t z[MZ_L], emb[MZ_L];
    if (action < 0) action = 0;
    if (action >= MZ_A) action = MZ_A - 1;                              // gather bounds safety
    for (int i = 0; i < MZ_L; i++) {
        emb[i] = (data_t)buffer[OFF_AEMB + action * MZ_L + i];
        z[i] = sat16((acc_t)buffer[OFF_H + i] + (acc_t)emb[i]);         // z = h + action_embed(a)
    }
    dyn_block(buffer, OFF_D0A_EXP, OFF_D0A_SGN, &buffer[OFF_B_D0A], OFF_D0B_EXP, OFF_D0B_SGN, &buffer[OFF_B_D0B], z);
    dyn_block(buffer, OFF_D1A_EXP, OFF_D1A_SGN, &buffer[OFF_B_D1A], OFF_D1B_EXP, OFF_D1B_SGN, &buffer[OFF_B_D1B], z);
    data_t rew[1];
    dyadic_linear(buffer, OFF_RH_EXP, OFF_RH_SGN, &buffer[OFF_B_RH], z, rew, 1, MZ_L);
    buffer[OFF_REWARD] = rew[0];
    for (int i = 0; i < MZ_L; i++) buffer[OFF_H + i] = z[i];            // h' replaces h (chained unroll)
    pred_from_h(buffer, z);
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
    (void)arg2; (void)arg3;
    if (mode == 0) return do_fwd(buffer);
    if (mode == 1) return do_dyn(buffer, arg1);   // arg1 = action index for the recurrent step
    return -1;   // bwd / signsgd: next stage
}

// Layout introspection for the host bit-exact harness (so a Python reference cannot
// silently drift from this buffer layout). Not part of the silicon ABI.
extern "C" int forge_muzero_buffer_words() { return MZ_BUFFER_WORDS; }
extern "C" void forge_muzero_layout(int* out) {
    // [OBS, L, A, DR, DP, OFF_OBS, OFF_H, OFF_TMP_L, OFF_POLICY, OFF_VALUE,
    //  OFF_R1_EXP, OFF_R1_SGN, OFF_R2_EXP, OFF_R2_SGN, OFF_P1_EXP, OFF_P1_SGN,
    //  OFF_P2_EXP, OFF_P2_SGN, OFF_PH_EXP, OFF_PH_SGN, OFF_VH_EXP, OFF_VH_SGN,
    //  OFF_B_R1, OFF_B_R2, OFF_B_P1, OFF_B_P2, OFF_B_PH, OFF_B_VH,   <- forward (unchanged prefix)
    //  DD, OFF_AEMB, OFF_REWARD, OFF_D0A_EXP, OFF_D0A_SGN, OFF_D0B_EXP, OFF_D0B_SGN,
    //  OFF_D1A_EXP, OFF_D1A_SGN, OFF_D1B_EXP, OFF_D1B_SGN, OFF_RH_EXP, OFF_RH_SGN,
    //  OFF_B_D0A, OFF_B_D0B, OFF_B_D1A, OFF_B_D1B, OFF_B_RH]            <- dynamics (appended)
    int v[] = {MZ_OBS, MZ_L, MZ_A, MZ_DR, MZ_DP, OFF_OBS, OFF_H, OFF_TMP_L, OFF_POLICY, OFF_VALUE,
               OFF_R1_EXP, OFF_R1_SGN, OFF_R2_EXP, OFF_R2_SGN, OFF_P1_EXP, OFF_P1_SGN,
               OFF_P2_EXP, OFF_P2_SGN, OFF_PH_EXP, OFF_PH_SGN, OFF_VH_EXP, OFF_VH_SGN,
               OFF_B_R1, OFF_B_R2, OFF_B_P1, OFF_B_P2, OFF_B_PH, OFF_B_VH,
               MZ_DD, OFF_AEMB, OFF_REWARD, OFF_D0A_EXP, OFF_D0A_SGN, OFF_D0B_EXP, OFF_D0B_SGN,
               OFF_D1A_EXP, OFF_D1A_SGN, OFF_D1B_EXP, OFF_D1B_SGN, OFF_RH_EXP, OFF_RH_SGN,
               OFF_B_D0A, OFF_B_D0B, OFF_B_D1A, OFF_B_D1B, OFF_B_RH};
    for (int i = 0; i < (int)(sizeof(v) / sizeof(int)); i++) out[i] = v[i];
}
