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
    // floor(log2|w|). BOUNDED loop (was `while (t>1)`, data-dependent -> HLS left the train-kernel
    // latency UNRESOLVED '?'): the shadow weight is int16, so |w| < 2^15 => at most 15 right-shifts.
    // A fixed 15-trip loop is functionally identical (t pins at 1) and lets HLS resolve the latency.
    for (int _b = 0; _b < 15; _b++) {
        if (t > 1) { t >>= 1; hi++; }
    }
    int lo = 1 << hi, rem = a - lo;
    if ((rem << 1) >= lo) hi++;          // round to nearest pow2 by value-midpoint (shift+compare)
    e = hi;
}

// apply a quantized weight q=(e,s) to a value: s*(v<<e) / s*(v>>-e); s==0 => 0. Arithmetic shift.
static inline acc_t dyq_apply(int e, int s, acc_t v) {
    if (s == 0) return 0;
    acc_t r = (e >= 0) ? (v << e) : (v >> (-e));
    return (s > 0) ? r : (acc_t)(-r);   // explicit cast: ap_int negation widens to <33>, so the ?: needs it
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
// OUT/IN are TEMPLATE (compile-time) params so o*IN strength-reduces to a shift (zero DSP).
// At chess scale runtime IN=769/1538 are non-pow2; passing them as int args made HLS build a
// var*var multiplier farm under the unroll (the chess csynth DSP=195). All call sites pass
// constexpr MZ_* dims, so this is a pure addressing change — bit-exact.
template <int OUT, int IN>
static void fwd_linear(const int32_t* W, const int32_t* bias, const data_t* x, data_t* y) {
    const int32_t* Wrow = W;                 // pointer-walk: Wrow += IN (add), NOT W[o*IN+i]. HLS maps even a
    for (int o = 0; o < OUT; o++) {          // CONSTANT mul (o*IN, IN compile-time) to a DSP48 by default, so
        acc_t acc = (acc_t)bias[o];          // templating alone left DSP=195 — the mul must be GONE from source.
        for (int i = 0; i < IN; i++) {
            int e, s; nearest_pow2((int)Wrow[i], e, s);
            acc += dyq_apply(e, s, (acc_t)x[i]);
        }
        y[o] = sat16(acc);
        Wrow += IN;
    }
}

// g_in[i] = sum_o q(W[o,i]) . g_out[o] — backprop of input grad through the pow2 weights.
// OUT/IN template params (see fwd_linear): the transposed read W[o*IN+i] was the largest DSP
// cluster at chess scale; compile-time IN makes o*IN a shift. Bit-exact.
template <int OUT, int IN>
static void bwd_input(const int32_t* W, const acc_t* g_out, acc_t* g_in) {
    for (int i = 0; i < IN; i++) {
        acc_t g = 0;
        const int32_t* Wp = W + i;           // transposed read: Wp += IN per o (add), NOT W[o*IN+i] (DSP mul)
        for (int o = 0; o < OUT; o++) {
            int e, s; nearest_pow2((int)*Wp, e, s);
            g += dyq_apply(e, s, g_out[o]);
            Wp += IN;
        }
        g_in[i] = g;
    }
}

// SignSGD weight+bias update: W[o,i] -= lr*sign(g_out[o])*sign(x[i]); bias[o] -= lr*sign(g_out[o]).
// OUT/IN template params (see fwd_linear) so the W[o*IN+i] update index is a shift. Bit-exact.
template <int OUT, int IN>
static void sgd_layer(int32_t* W, int32_t* bias, const acc_t* g_out, const data_t* x, int lr) {
    int32_t* Wrow = W;                        // pointer-walk: Wrow += IN (add), NOT W[o*IN+i] (DSP mul)
    for (int o = 0; o < OUT; o++) {
        int go = isign(g_out[o]);
        for (int i = 0; i < IN; i++)
            sgd_update(&Wrow[i], lrstep(lr, sgnprod((acc_t)go, (acc_t)x[i])));
        sgd_update(&bias[o], lrstep(lr, go));
        Wrow += IN;
    }
}

// ---- HBM streaming for WR1 (~4.7 MB, the only matrix too big to keep resident) ----
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
typedef ap_uint<512> wide_t;
#endif
static void burst_load_words(int32_t* buffer, int word_off, int32_t* dst, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wb = reinterpret_cast<wide_t*>(buffer); int nw = n_words >> 4, wo = word_off >> 4;
    int32_t* dp = dst;                       // pointer-walk: dst[w*16+l] -> dp[l], dp+=16 (add, not w*16 mul)
    for (int w = 0; w < nw; w++) {
#pragma HLS PIPELINE II=1
        wide_t b = wb[wo + w];
        for (int l = 0; l < 16; l++) { dp[l] = (int32_t)(int)(ap_int<32>)b.range((l<<5)+31, (l<<5)); }
        dp += 16;
    }
#else
    for (int i = 0; i < n_words; i++) dst[i] = buffer[word_off + i];
#endif
}
static void burst_store_words(int32_t* src, int32_t* buffer, int word_off, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wb = reinterpret_cast<wide_t*>(buffer); int nw = n_words >> 4, wo = word_off >> 4;
    int32_t* sp = src;                       // pointer-walk: src[w*16+l] -> sp[l], sp+=16 (add, not w*16 mul)
    for (int w = 0; w < nw; w++) {
#pragma HLS PIPELINE II=1
        wide_t b = 0;
        for (int l = 0; l < 16; l++) { b.range((l<<5)+31, (l<<5)) = (ap_uint<32>)(ap_int<32>)sp[l]; }
        wb[wo + w] = b;
        sp += 16;
    }
#else
    for (int i = 0; i < n_words; i++) buffer[word_off + i] = src[i];
#endif
}

// R1 forward, streamed: q(shadow) . x with WR1 read in 16-word-aligned CHUNK tiles (chunked+pipelined,
// same fast pattern as the forward kernel). Bit-exact to fwd_linear on WR1.
// OUT/IN template params so total=OUT*IN folds to a compile-time constant (no var*var DSP).
template <int OUT, int IN>
static void fwd_linear_stream(int32_t* buffer, int off_W, const int32_t* bias,
                              const data_t* x, data_t* y) {
    const int CH = 4096; int32_t wt[CH];
    int o = 0, i = 0, total = OUT * IN; acc_t acc = (acc_t)bias[0];
    for (int base = 0; base < total; base += CH) {
        int n = (total - base < CH) ? (total - base) : CH, nrd = (n + 15) & ~15;
        burst_load_words(buffer, off_W + base, wt, nrd);
        for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
            int e, s; nearest_pow2(wt[k], e, s);
            acc += dyq_apply(e, s, (acc_t)x[i]);
            i++; if (i == IN) { y[o] = sat16(acc); o++; i = 0; if (o < OUT) acc = (acc_t)bias[o]; }
        }
    }
}

// R1 SignSGD, streamed read-modify-write: chunk-load WR1 tile, sign-update in place, write the tile back.
// OUT/IN template params so total=OUT*IN folds to a compile-time constant (no var*var DSP).
template <int OUT, int IN>
static void sgd_layer_stream(int32_t* buffer, int off_W, int32_t* bias,
                             const acc_t* g_out, const data_t* x, int lr) {
    const int CH = 4096; int32_t wt[CH];
    int o = 0, i = 0, total = OUT * IN, go = isign(g_out[0]);
    for (int base = 0; base < total; base += CH) {
        int n = (total - base < CH) ? (total - base) : CH, nrd = (n + 15) & ~15;
        burst_load_words(buffer, off_W + base, wt, nrd);
        for (int k = 0; k < n; k++) {
#pragma HLS PIPELINE II=1
            sgd_update(&wt[k], lrstep(lr, sgnprod((acc_t)go, (acc_t)x[i])));
            i++; if (i == IN) { sgd_update(&bias[o], lrstep(lr, go)); o++; i = 0; if (o < OUT) go = isign(g_out[o]); }
        }
        burst_store_words(wt, buffer, off_W + base, nrd);
    }
}

static int train_step(int32_t* buf, int lr) {
    // WR1 (~4.7 MB) stays in HBM and is STREAMED (fwd + SignSGD-RMW); the smaller matrices fit on-chip
    // (~1.46 MB) and are loaded RESIDENT so their transposed backward reads (bwd_input / the gz loop) are
    // on-chip random access. Static (not stack) so CSIM doesn't blow the stack on the 1 MB WPH array.
    static int32_t WR2[TW_R2], WP1[TW_P1], WP2[TW_P2], WPH[TW_PH], WVH[TW_VH];
    static int32_t BR1[MZ_DR], BR2[MZ_L], BP1[MZ_DP], BP2[MZ_L], BPH[MZ_A], BVH[1];
    burst_load_words(buf, TOFF_W_R2, WR2, TW_R2); burst_load_words(buf, TOFF_W_P1, WP1, TW_P1);
    burst_load_words(buf, TOFF_W_P2, WP2, TW_P2); burst_load_words(buf, TOFF_W_PH, WPH, TW_PH);
    burst_load_words(buf, TOFF_W_VH, WVH, TW_VH);
    for (int i = 0; i < MZ_DR; i++) BR1[i] = buf[TOFF_B_R1 + i];
    for (int i = 0; i < MZ_L;  i++) BR2[i] = buf[TOFF_B_R2 + i];
    for (int i = 0; i < MZ_DP; i++) BP1[i] = buf[TOFF_B_P1 + i];
    for (int i = 0; i < MZ_L;  i++) BP2[i] = buf[TOFF_B_P2 + i];
    for (int i = 0; i < MZ_A;  i++) BPH[i] = buf[TOFF_B_PH + i];
    BVH[0] = buf[TOFF_B_VH];

    data_t obs[MZ_OBS];
    for (int i = 0; i < MZ_OBS; i++) obs[i] = (data_t)buf[TOFF_OBS + i];

    // ---- FORWARD (save the activations the backward needs) ----
    data_t zr1[MZ_DR], h[MZ_L], zp1[MZ_DP], z[MZ_L], pol[MZ_A], val[1];
    data_t hraw[MZ_L], pre[MZ_DR > MZ_DP ? MZ_DR : MZ_DP];

    fwd_linear_stream<MZ_DR, MZ_OBS>(buf, TOFF_W_R1, BR1, obs, pre);  // repr lin1 (R1 streamed from HBM)
    for (int i = 0; i < MZ_DR; i++) zr1[i] = pre[i] > 0 ? pre[i] : (data_t)0;   // relu
    fwd_linear<MZ_L, MZ_DR>(WR2, BR2, zr1, hraw);                  // repr lin2
    acc_t sm = 0; for (int i = 0; i < MZ_L; i++) sm += hraw[i];
    data_t mean = (data_t)(sm / MZ_L);                            // /L = shift (L pow2)
    for (int i = 0; i < MZ_L; i++) h[i] = sat16((acc_t)hraw[i] - mean);  // submean -> latent

    fwd_linear<MZ_DP, MZ_L>(WP1, BP1, h, pre);                    // pred lin1
    for (int i = 0; i < MZ_DP; i++) zp1[i] = pre[i] > 0 ? pre[i] : (data_t)0;   // relu
    fwd_linear<MZ_L, MZ_DP>(WP2, BP2, zp1, z);                    // pred lin2
    fwd_linear<MZ_A, MZ_L>(WPH, BPH, z, pol);                     // policy head
    fwd_linear<1, MZ_L>(WVH, BVH, z, val);                        // value head

    for (int o = 0; o < MZ_A; o++) buf[TOFF_POL + o] = pol[o];
    buf[TOFF_VAL] = val[0];

    // ---- L1 LOSS + output grads (sign of residual, in {-1,0,+1}) ----
    acc_t loss = 0, gpol[MZ_A], gval;
    for (int o = 0; o < MZ_A; o++) {
        acc_t d = (acc_t)pol[o] - (acc_t)buf[TOFF_PTGT + o];
        loss += (d > 0 ? d : (acc_t)(-d)); gpol[o] = isign(d);
    }
    { acc_t d = (acc_t)val[0] - (acc_t)buf[TOFF_VTGT];
      loss += (d > 0 ? d : (acc_t)(-d)); gval = isign(d); }
    buf[TOFF_LOSS] = (int32_t)loss;

    // ---- BACKWARD: per layer compute grad-input with CURRENT weights, then SignSGD-update ----
    acc_t gz[MZ_L];                                               // grad wrt z (heads' input)
    for (int i = 0; i < MZ_L; i++) {
        int e, s; acc_t g = 0;
        const int32_t* wp = WPH + i;        // pointer-walk: WPH[o*MZ_L+i] -> *wp, wp+=MZ_L (add, not o*MZ_L)
        for (int o = 0; o < MZ_A; o++) { nearest_pow2((int)*wp, e, s); g += dyq_apply(e, s, gpol[o]); wp += MZ_L; }
        nearest_pow2((int)WVH[i], e, s); g += dyq_apply(e, s, gval);
        gz[i] = g;
    }
    sgd_layer<MZ_A, MZ_L>(WPH, BPH, gpol, z, lr);
    { acc_t gv1[1] = {gval}; sgd_layer<1, MZ_L>(WVH, BVH, gv1, z, lr); }

    acc_t gzp1[MZ_DP];                                            // through pred lin2 (P2)
    bwd_input<MZ_L, MZ_DP>(WP2, gz, gzp1);
    sgd_layer<MZ_L, MZ_DP>(WP2, BP2, gz, zp1, lr);
    for (int j = 0; j < MZ_DP; j++) if (zp1[j] <= 0) gzp1[j] = 0;       // relu mask

    acc_t gh[MZ_L];                                              // through pred lin1 (P1)
    bwd_input<MZ_DP, MZ_L>(WP1, gzp1, gh);
    sgd_layer<MZ_DP, MZ_L>(WP1, BP1, gzp1, h, lr);

    acc_t gsum = 0; for (int i = 0; i < MZ_L; i++) gsum += gh[i];       // submean backward
    acc_t gmean = gsum / MZ_L;                                          // /L = shift
    acc_t ghraw[MZ_L]; for (int i = 0; i < MZ_L; i++) ghraw[i] = gh[i] - gmean;

    acc_t gzr1[MZ_DR];                                          // through repr lin2 (R2)
    bwd_input<MZ_L, MZ_DR>(WR2, ghraw, gzr1);
    sgd_layer<MZ_L, MZ_DR>(WR2, BR2, ghraw, zr1, lr);
    for (int m = 0; m < MZ_DR; m++) if (zr1[m] <= 0) gzr1[m] = 0;       // relu mask

    sgd_layer_stream<MZ_DR, MZ_OBS>(buf, TOFF_W_R1, BR1, gzr1, obs, lr);  // R1 streamed RMW (writes WR1 back)

    // ---- write the updated RESIDENT weights + all biases back to HBM (WR1 already written by the stream) ----
    burst_store_words(WR2, buf, TOFF_W_R2, TW_R2); burst_store_words(WP1, buf, TOFF_W_P1, TW_P1);
    burst_store_words(WP2, buf, TOFF_W_P2, TW_P2); burst_store_words(WPH, buf, TOFF_W_PH, TW_PH);
    burst_store_words(WVH, buf, TOFF_W_VH, TW_VH);
    for (int i = 0; i < MZ_DR; i++) buf[TOFF_B_R1 + i] = BR1[i];
    for (int i = 0; i < MZ_L;  i++) buf[TOFF_B_R2 + i] = BR2[i];
    for (int i = 0; i < MZ_DP; i++) buf[TOFF_B_P1 + i] = BP1[i];
    for (int i = 0; i < MZ_L;  i++) buf[TOFF_B_P2 + i] = BP2[i];
    for (int i = 0; i < MZ_A;  i++) buf[TOFF_B_PH + i] = BPH[i];
    buf[TOFF_B_VH] = BVH[0];
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
