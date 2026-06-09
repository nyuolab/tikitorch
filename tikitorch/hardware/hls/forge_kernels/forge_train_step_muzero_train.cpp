// forge_train_step_muzero_train.cpp — ON-CHIP TRAINING of the dyadic Tropical MuZero net.
//
// The companion to forge_train_step_muzero.cpp (frozen forward inference). This kernel runs a
// full supervised initial-inference TRAINING step — forward + L1 loss + backward + weight
// update — entirely in comparisons, additions, and bit-shifts. ZERO DSP, in the backward too.
//
// HOW THE BACKWARD STAYS MULTIPLIER-FREE (the crux):
//   * forward:        weights are int16 SHADOWS quantized on the fly to nearest signed pow2
//                     q(w)=s*2^e (nearest_pow2 = leading-bit search + a shift-compare round),
//                     so every product is a sign+shift.                       [shift]
//   * grad wrt input: dL/dx = q(W)^T . g_out — multiplies by the *pow2* weights -> shift+add.
//   * grad wrt weight: SignSGD needs only sign(dL/dw)=sign(g_out) (xor) sign(x), a 1-bit
//                     sign-product — NO multiply (the exact dL/dw = g_out (x) x is the one
//                     genuine multiply, and SignSGD never forms it).          [sign-product]
//   * update:         w <- clamp16(w - lr*sign(dL/dw)); a select/negate + int add. Weights
//                     pass through 0, so signs flip naturally.                [int add]
//   * relu backward:  mask by (activation>0).  submean backward: g - mean(g) (/L = shift).
//
// This is straight-through pow2 SignSGD (what tikitorch.nn.dyadic_quantize does in float),
// re-expressed in integers so forward AND backward are zero-DSP. assert_training_closed on the
// Python mirror of this rule is the moat check that the *training* is multiplier-free.
//
// CSIM:  g++ -O2 -std=c++17 -DCSIM_ONLY -I.. -I. forge_train_step_muzero_train.cpp \
//            forge_train_step_muzero_train_tb.cpp -o tb_muzero_train && ./tb_muzero_train
#include "forge_buffer_muzero_train.hpp"
#include <cstdint>

static inline int isign(acc_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

// sign(dL/dw) = sign(g_out) (xor) sign(x) in {-1,0,+1} — a 1-bit product, no multiplier.
static inline int sgnprod(acc_t a, acc_t b) {
    int sa = isign(a), sb = isign(b);
    if (sa == 0 || sb == 0) return 0;
    return (sa == sb) ? 1 : -1;
}

// lr * gsign as a select/negate (gsign in {-1,0,+1}) — never a hardware multiply.
static inline int lrstep(int lr, int gsign) { return gsign > 0 ? lr : (gsign < 0 ? -lr : 0); }

// nearest signed power of two of an int16 shadow: q(w)=s*2^e. e = rounded floor(log2|w|)
// found by a leading-bit search + a shift-compare round-up (|w| >= 1.5*2^e). NO multiply.
static inline void nearest_pow2(int w, int& e, int& s) {
    if (w == 0) { e = 0; s = 0; return; }
    s = (w > 0) ? 1 : -1;
    int a = (w > 0) ? w : -w;            // |w|
    int hi = 0, t = a;
    while (t > 1) { t >>= 1; hi++; }     // hi = floor(log2|w|)
    int lo = 1 << hi, rem = a - lo;
    if ((rem << 1) >= lo) hi++;          // round to nearest pow2 by value-midpoint (shift+compare)
    e = hi;
}

// apply a quantized weight q=(e,s) to a value: s*(v<<e) / s*(v>>-e); s==0 => 0. Arithmetic shift.
static inline acc_t dyq_apply(int e, int s, acc_t v) {
    if (s == 0) return 0;
    acc_t r = (e >= 0) ? (v << e) : (v >> (-e));
    return (s > 0) ? r : -r;
}

static inline data_t sat16(acc_t a) {
    if (a > 32767) a = 32767;
    if (a < -32768) a = -32768;
    return (data_t)a;
}

static inline void sgd_update(int32_t* w, int delta) {   // SignSGD: w <- clamp16(w - delta)
    acc_t nw = (acc_t)(*w) - (acc_t)delta;
    if (nw > 32767) nw = 32767;
    if (nw < -32768) nw = -32768;
    *w = (int32_t)nw;
}

// y[o] = sat16( bias[o] + sum_i q(W[o,i]) . x[i] ) — shadow-weight dyadic forward.
static void fwd_linear(const int32_t* W, const int32_t* bias, const data_t* x,
                       data_t* y, int OUT, int IN) {
    for (int o = 0; o < OUT; o++) {
        acc_t acc = (acc_t)bias[o];
        for (int i = 0; i < IN; i++) {
            int e, s; nearest_pow2((int)W[o * IN + i], e, s);
            acc += dyq_apply(e, s, (acc_t)x[i]);
        }
        y[o] = sat16(acc);
    }
}

// g_in[i] = sum_o q(W[o,i]) . g_out[o] — backprop of input grad through the pow2 weights.
static void bwd_input(const int32_t* W, const acc_t* g_out, acc_t* g_in, int OUT, int IN) {
    for (int i = 0; i < IN; i++) {
        acc_t g = 0;
        for (int o = 0; o < OUT; o++) {
            int e, s; nearest_pow2((int)W[o * IN + i], e, s);
            g += dyq_apply(e, s, g_out[o]);
        }
        g_in[i] = g;
    }
}

// SignSGD weight+bias update: W[o,i] -= lr*sign(g_out[o])*sign(x[i]); bias[o] -= lr*sign(g_out[o]).
static void sgd_layer(int32_t* W, int32_t* bias, const acc_t* g_out, const data_t* x,
                      int OUT, int IN, int lr) {
    for (int o = 0; o < OUT; o++) {
        int go = isign(g_out[o]);
        for (int i = 0; i < IN; i++)
            sgd_update(&W[o * IN + i], lrstep(lr, sgnprod((acc_t)go, (acc_t)x[i])));
        sgd_update(&bias[o], lrstep(lr, go));
    }
}

static int train_step(int32_t* buf, int lr) {
    int32_t* WR1 = &buf[TOFF_W_R1]; int32_t* WR2 = &buf[TOFF_W_R2];
    int32_t* WP1 = &buf[TOFF_W_P1]; int32_t* WP2 = &buf[TOFF_W_P2];
    int32_t* WPH = &buf[TOFF_W_PH]; int32_t* WVH = &buf[TOFF_W_VH];
    int32_t* BR1 = &buf[TOFF_B_R1]; int32_t* BR2 = &buf[TOFF_B_R2];
    int32_t* BP1 = &buf[TOFF_B_P1]; int32_t* BP2 = &buf[TOFF_B_P2];
    int32_t* BPH = &buf[TOFF_B_PH]; int32_t* BVH = &buf[TOFF_B_VH];

    data_t obs[MZ_OBS];
    for (int i = 0; i < MZ_OBS; i++) obs[i] = (data_t)buf[TOFF_OBS + i];

    // ---- FORWARD (save the activations the backward needs) ----
    data_t zr1[MZ_DR], h[MZ_L], zp1[MZ_DP], z[MZ_L], pol[MZ_A], val[1];
    data_t hraw[MZ_L], pre[MZ_DR > MZ_DP ? MZ_DR : MZ_DP];

    fwd_linear(WR1, BR1, obs, pre, MZ_DR, MZ_OBS);                  // repr lin1
    for (int i = 0; i < MZ_DR; i++) zr1[i] = pre[i] > 0 ? pre[i] : 0;   // relu
    fwd_linear(WR2, BR2, zr1, hraw, MZ_L, MZ_DR);                  // repr lin2
    acc_t sm = 0; for (int i = 0; i < MZ_L; i++) sm += hraw[i];
    data_t mean = (data_t)(sm / MZ_L);                            // /L = shift (L pow2)
    for (int i = 0; i < MZ_L; i++) h[i] = sat16((acc_t)hraw[i] - mean);  // submean -> latent

    fwd_linear(WP1, BP1, h, pre, MZ_DP, MZ_L);                    // pred lin1
    for (int i = 0; i < MZ_DP; i++) zp1[i] = pre[i] > 0 ? pre[i] : 0;   // relu
    fwd_linear(WP2, BP2, zp1, z, MZ_L, MZ_DP);                    // pred lin2
    fwd_linear(WPH, BPH, z, pol, MZ_A, MZ_L);                     // policy head
    fwd_linear(WVH, BVH, z, val, 1, MZ_L);                        // value head

    for (int o = 0; o < MZ_A; o++) buf[TOFF_POL + o] = pol[o];
    buf[TOFF_VAL] = val[0];

    // ---- L1 LOSS + output grads (sign of residual, in {-1,0,+1}) ----
    acc_t loss = 0, gpol[MZ_A], gval;
    for (int o = 0; o < MZ_A; o++) {
        acc_t d = (acc_t)pol[o] - (acc_t)buf[TOFF_PTGT + o];
        loss += (d > 0 ? d : -d); gpol[o] = isign(d);
    }
    { acc_t d = (acc_t)val[0] - (acc_t)buf[TOFF_VTGT];
      loss += (d > 0 ? d : -d); gval = isign(d); }
    buf[TOFF_LOSS] = (int32_t)loss;

    // ---- BACKWARD: per layer compute grad-input with CURRENT weights, then SignSGD-update ----
    acc_t gz[MZ_L];                                               // grad wrt z (heads' input)
    for (int i = 0; i < MZ_L; i++) {
        int e, s; acc_t g = 0;
        for (int o = 0; o < MZ_A; o++) { nearest_pow2((int)WPH[o * MZ_L + i], e, s); g += dyq_apply(e, s, gpol[o]); }
        nearest_pow2((int)WVH[i], e, s); g += dyq_apply(e, s, gval);
        gz[i] = g;
    }
    sgd_layer(WPH, BPH, gpol, z, MZ_A, MZ_L, lr);
    { acc_t gv1[1] = {gval}; sgd_layer(WVH, BVH, gv1, z, 1, MZ_L, lr); }

    acc_t gzp1[MZ_DP];                                            // through pred lin2 (P2)
    bwd_input(WP2, gz, gzp1, MZ_L, MZ_DP);
    sgd_layer(WP2, BP2, gz, zp1, MZ_L, MZ_DP, lr);
    for (int j = 0; j < MZ_DP; j++) if (zp1[j] <= 0) gzp1[j] = 0;       // relu mask

    acc_t gh[MZ_L];                                              // through pred lin1 (P1)
    bwd_input(WP1, gzp1, gh, MZ_DP, MZ_L);
    sgd_layer(WP1, BP1, gzp1, h, MZ_DP, MZ_L, lr);

    acc_t gsum = 0; for (int i = 0; i < MZ_L; i++) gsum += gh[i];       // submean backward
    acc_t gmean = gsum / MZ_L;                                          // /L = shift
    acc_t ghraw[MZ_L]; for (int i = 0; i < MZ_L; i++) ghraw[i] = gh[i] - gmean;

    acc_t gzr1[MZ_DR];                                          // through repr lin2 (R2)
    bwd_input(WR2, ghraw, gzr1, MZ_L, MZ_DR);
    sgd_layer(WR2, BR2, ghraw, zr1, MZ_L, MZ_DR, lr);
    for (int m = 0; m < MZ_DR; m++) if (zr1[m] <= 0) gzr1[m] = 0;       // relu mask

    sgd_layer(WR1, BR1, gzr1, obs, MZ_DR, MZ_OBS, lr);          // repr lin1 (R1); no grad past obs
    return 0;
}

// mode 2 = one supervised SignSGD training step (arg1 = integer learning rate, default 1).
extern "C" int forge_train_step_muzero_train(int32_t* buffer, int mode, int arg1, int arg2, int arg3) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=mode
#pragma HLS INTERFACE s_axilite port=arg1
#pragma HLS INTERFACE s_axilite port=arg2
#pragma HLS INTERFACE s_axilite port=arg3
#pragma HLS INTERFACE s_axilite port=return
    (void)arg2; (void)arg3;
    if (mode == 2) return train_step(buffer, arg1 > 0 ? arg1 : 1);
    return -1;
}

// Layout introspection for the host harness (keeps a Python mirror from drifting).
extern "C" int forge_muzero_train_buffer_words() { return MZT_BUFFER_WORDS; }
extern "C" void forge_muzero_train_layout(int* out) {
    int v[] = {MZ_OBS, MZ_L, MZ_A, MZ_DR, MZ_DP,
               TOFF_W_R1, TOFF_W_R2, TOFF_W_P1, TOFF_W_P2, TOFF_W_PH, TOFF_W_VH,
               TOFF_B_R1, TOFF_B_R2, TOFF_B_P1, TOFF_B_P2, TOFF_B_PH, TOFF_B_VH,
               TOFF_OBS, TOFF_PTGT, TOFF_VTGT, TOFF_POL, TOFF_VAL, TOFF_LOSS};
    for (int i = 0; i < (int)(sizeof(v) / sizeof(int)); i++) out[i] = v[i];
}
