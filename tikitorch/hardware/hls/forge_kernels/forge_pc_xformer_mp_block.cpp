// forge_pc_xformer_mp_block.cpp — BF16 MIXED-PRECISION distributed transformer-block training shard.
// ONE physical FPGA = ONE block; a whole transformer trained MODEL-PARALLEL across L chips, exchanging ONLY a
// neighbour hidden-state (down, forward) + a neighbour error (up, backward). No chip holds the global model or
// global gradient. Role (INPUT/HID/OUTPUT) + phase (FWD/BWD) from a CFG word -> ONE AFI on every slot.
//
// THIS kernel is the BF16 backward variant of forge_pc_xformer_block.cpp. The no-global-sync architecture, the
// mult-free dyadic FORWARD, the cacheable cascade, and the DDR scaffolding are UNCHANGED; only the BACKWARD
// arithmetic precision changes (int cascade -> textbook transformer backprop with STE, in bf16).
//
//   FWD (DEPLOY, int, byte-identical to forge_pc_xformer_block): mult-free dyadic proj + warm DOT attention +
//        precise mult-free exp softmax -> int16 acts + Q16 probs cached at the BASE XO_* offsets.
//   FWD (WIDE, float-master): a parallel char-LM-faithful forward (bf_forward) caches the WIDE pre-quantization
//        activations the STE backward differentiates, in the appended MP_* region.
//   BWD (BF16): textbook transformer backprop (analytic softmax-Jacobian, mean-centering LN backward, ReLU/
//        residual masks, dyad(W)^T@gy route + gy^T@x wgrad) on the float master, every backward intermediate
//        rounded through bf16 (1s8e7m round-to-even) == the de-risked golden BF16_BWD=2. Inter-chip error
//        message (gRin/gRout) is bf16-wide.
//   OPT: float/bf16 dyadic-Adam apply on the float master (narrow-master is a later step; matches the golden).
//
// THE FORWARD (int) is a verbatim transcription of forge_pc_xformer_block.cpp's forward (same dyad/norm_shift/
// burst/attention/layernorm). It is reproduced here so this kernel is self-contained and the int kernel file is
// untouched. PRECISION CHOICE: the bf16 backward uses `float` working storage with an explicit bf16 round of
// every intermediate (bf16_round below) == golden to_bf16(); real bf16 multiplies live in route/wgrad/Jacobian
// (the new DSP). Production can swap `float`+bf16_round for hls::bfloat16/ap_fixed; the rounding is identical.
// Part xcvu47p-fsvh2892-2L-e, 4.0 ns.
#include <stdint.h>
#include <ap_int.h>
#include <hls_math.h>
#include "forge_pc_xformer_mp_config.hpp"

typedef ap_int<16> data_t;     // activations / int16 weights (DEPLOY forward)
typedef ap_int<6>  exp_t;      // dyadic exponent
typedef ap_int<64> acc_t;      // int accumulators (forward only)
typedef ap_int<32> prob_t;     // Q16 probabilities

// =====================================================================================================
// bf16: float32 truncated to 7 mantissa bits, round-to-even — BIT-IDENTICAL to the golden to_bf16 / P().
// HLS-synthesizable: bit ops on the 32-bit IEEE pattern (no library bfloat needed). The mp backward stores
// activations/grads as `float` and rounds every intermediate through this, so the kernel reproduces the
// golden's BF16_BWD=2 arithmetic exactly while Vitis binds the products to DSP.
// =====================================================================================================
static inline float bf16_round(float f) {
#pragma HLS INLINE
    union { float f; uint32_t u; } c; c.f = f;
    uint32_t u = c.u;
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;        // round-to-nearest-even into bit 16
    u &= 0xffff0000u;          // drop the low 16 mantissa bits
    c.u = u; return c.f;
}
static inline float P(float x) {
#pragma HLS INLINE
    return bf16_round(x);
}
// float<->int32 DDR lane bit-reinterpret (one float per int32 lane; reuses the 512-bit burst path)
static inline float i2f(int32_t i) {
#pragma HLS INLINE
    union { float f; int32_t i; } c; c.i = i; return c.f;
}
static inline int32_t f2i(float f) {
#pragma HLS INLINE
    union { float f; int32_t i; } c; c.f = f; return c.i;
}
// the DEPLOYED dyadic weight = nearest signed power of two of the FLOAT master (== char-LM DyadSTE / bf_dyad_f).
// route/wgrad multiply by THIS; the STE means the round is identity, grad flows to the float master.
static inline float bf_dyad_f(float wf) {
#pragma HLS INLINE
    float a = wf < 0 ? -wf : wf; if (a < 1e-30f) return 0.0f;
    int e = (int)hls::round(hls::log2(a));
    float p = hls::ldexp(1.0f, e);
    return wf < 0 ? -p : p;
}

// ============================== DEPLOY FORWARD primitives (int, verbatim) ==============================
static inline void dyad(data_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<16> a = (v < 0) ? (ap_uint<16>)(-v) : (ap_uint<16>)v;
    sgn = (v < 0); nz = (a != 0);
    exp_t msb = 0;
PRIO: for (int b = 15; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { msb = b; break; }
    }
    if (msb >= 1 && a[msb - 1]) msb = msb + 1;
    e = msb;
}
static inline int norm_shift(acc_t mx, int target) {
#pragma HLS INLINE
    acc_t a = (mx < 0) ? (acc_t)(-mx) : mx;
    int hb = 0;
NS: for (int b = 62; b >= 1; b--) {
#pragma HLS UNROLL
        if ((a >> b) & 1) { hb = b; break; }
    }
    int s = (hb + 1) - target;
    return s > 0 ? s : 0;
}
static inline void fdyad(acc_t scale, int &e_out, int &nq_out) {
#pragma HLS INLINE
    if (scale < 1) scale = 1;
    int p = 0;
FP: for (int b = 46; b >= 1; b--) {
#pragma HLS UNROLL
        if (scale[b]) { p = b; break; }
    }
    bool exact = (scale == ((acc_t)1 << p));
    int e = exact ? (-p) : (-(p + 1));
    int nege = -e;
    acc_t thresh = ((acc_t)1 << (X_MBITS + nege)) + (scale >> 1);
    int nbase = (1 << X_MBITS), nq = nbase;
    acc_t acc = scale << X_MBITS;
CAND: for (int st = 1; st <= (1 << X_MBITS); st++) {
#pragma HLS UNROLL
        acc += scale;
        if (acc <= thresh) nq = nbase + st;
    }
    if (nq >= (1 << (X_MBITS + 1))) { nq = (1 << X_MBITS); e += 1; }
    if (nq < (1 << X_MBITS)) nq = (1 << X_MBITS);
    e_out = e; nq_out = nq;
}

// ---- 512-bit burst I/O (data_t / int32 / float lanes) ----
static inline void bload(ap_uint<512> *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int w = 0; w < n / X_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = buf[off / X_WL + w];
        for (int l = 0; l < X_WL; l++) {
#pragma HLS UNROLL
            dst[w * X_WL + l] = (data_t)(ap_int<32>)word.range(32 * l + 31, 32 * l);
        }
    }
}
static inline void bstore(ap_uint<512> *buf, int off, const data_t *src, int n) {
#pragma HLS INLINE off
ST: for (int w = 0; w < n / X_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = 0;
        for (int l = 0; l < X_WL; l++) {
#pragma HLS UNROLL
            word.range(32 * l + 31, 32 * l) = (ap_uint<32>)(ap_int<32>)(int)src[w * X_WL + l];
        }
        buf[off / X_WL + w] = word;
    }
}
static inline void bload32(ap_uint<512> *buf, int off, int32_t *dst, int n) {
#pragma HLS INLINE off
LDW: for (int w = 0; w < n / X_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = buf[off / X_WL + w];
        for (int l = 0; l < X_WL; l++) {
#pragma HLS UNROLL
            dst[w * X_WL + l] = (int32_t)(int)(ap_int<32>)word.range(32 * l + 31, 32 * l);
        }
    }
}
static inline void bstore32(ap_uint<512> *buf, int off, const int32_t *src, int n) {
#pragma HLS INLINE off
STW: for (int w = 0; w < n / X_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = 0;
        for (int l = 0; l < X_WL; l++) {
#pragma HLS UNROLL
            word.range(32 * l + 31, 32 * l) = (ap_uint<32>)(ap_int<32>)src[w * X_WL + l];
        }
        buf[off / X_WL + w] = word;
    }
}
// FLOAT lane burst I/O (one float per int32 lane, bit-reinterpret) — for the float master + wide cache + bf16 msg
static inline void bload_f(ap_uint<512> *buf, int off, float *dst, int n) {
#pragma HLS INLINE off
LDF: for (int w = 0; w < n / X_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = buf[off / X_WL + w];
        for (int l = 0; l < X_WL; l++) {
#pragma HLS UNROLL
            dst[w * X_WL + l] = i2f((int32_t)(int)(ap_int<32>)word.range(32 * l + 31, 32 * l));
        }
    }
}
static inline void bstore_f(ap_uint<512> *buf, int off, const float *src, int n) {
#pragma HLS INLINE off
STF: for (int w = 0; w < n / X_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = 0;
        for (int l = 0; l < X_WL; l++) {
#pragma HLS UNROLL
            word.range(32 * l + 31, 32 * l) = (ap_uint<32>)(ap_int<32>)f2i(src[w * X_WL + l]);
        }
        buf[off / X_WL + w] = word;
    }
}

static void rescale_band2(const data_t *src, data_t *dst, int n, int bits) {
#pragma HLS INLINE off
    acc_t mx = 0;
RC1: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        acc_t v = (src[i] < 0) ? (acc_t)(-src[i]) : (acc_t)src[i]; if (v > mx) mx = v;
    }
    int sh = norm_shift(mx, bits);
RC2: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)((acc_t)src[i] >> sh);
    }
}
static void dyad_matmul(const data_t *x, const data_t *W, acc_t *y, int M, int I, int O) {
#pragma HLS INLINE off
MM_O: for (int o = 0; o < O; o++) {
    MM_T: for (int t = 0; t < M; t++) {
            acc_t acc = 0;
        MM_I: for (int i = 0; i < I; i++) {
#pragma HLS PIPELINE II=1
                bool s, nz; exp_t e;
                dyad(W[o * I + i], s, e, nz);
                acc_t term = ((acc_t)x[t * I + i]) << e;
                if (nz) acc += s ? (acc_t)(-term) : term;
            }
            y[t * O + o] = acc;
        }
    }
}
static void rescale_to(const acc_t *src, data_t *dst, int n, int bits) {
#pragma HLS INLINE off
    acc_t mx = 0;
RTo1: for (int i = 0; i < n; i++) { acc_t v = src[i] < 0 ? (acc_t)(-src[i]) : src[i]; if (v > mx) mx = v; }
    int sh = norm_shift(mx, bits);
RTo2: for (int i = 0; i < n; i++) dst[i] = (data_t)(src[i] >> sh);
}
static void layernorm(const data_t *x, const data_t *gcode, const data_t *beta, data_t *y,
                      int32_t *stats, int M) {
#pragma HLS INLINE off
LN_T: for (int t = 0; t < M; t++) {
        acc_t sum = 0;
    LN_S: for (int d = 0; d < X_D; d++) {
#pragma HLS PIPELINE II=1
            sum += (acc_t)x[t * X_D + d];
        }
        acc_t mean = sum >> X_LOG2_D;
        data_t c[X_D]; acc_t asum = 0;
    LN_C: for (int d = 0; d < X_D; d++) {
#pragma HLS PIPELINE II=1
            acc_t cv = (acc_t)x[t * X_D + d] - mean;
            c[d] = (data_t)cv;
            asum += (cv < 0) ? (acc_t)(-cv) : cv;
        }
        acc_t meanabs = asum >> X_LOG2_D;
        int e, nq; fdyad(meanabs, e, nq);
        stats[2 * t] = e; stats[2 * t + 1] = nq;
        int nqm = nq - (1 << X_MBITS);
    LN_Y: for (int d = 0; d < X_D; d++) {
#pragma HLS PIPELINE II=1
            acc_t cv = (acc_t)c[d];
            acc_t cn = cv << X_MBITS;
            if (nqm >= 1) cn += cv;
            if (nqm >= 2) cn += cv;
            if (nqm >= 3) cn += cv;
            acc_t r = cn >> X_MBITS;
            if (e >= 0) r = r << e; else r = r >> (-e);
            bool gs, gnz; exp_t ge; dyad(gcode[d], gs, ge, gnz);
            acc_t g = gnz ? (acc_t)(r << (int)ge) : (acc_t)0;
            if (gnz && gs) g = -g;
            y[t * X_D + d] = (data_t)(g + (acc_t)beta[d]);
        }
    }
}

// =====================================================================================================
// WIDE-FLOAT forward-for-backward (== bf16 golden bf_forward). Mirrors the char-LM EXACTLY; fills the MP_*
// wide cache the textbook STE backward differentiates. Operates on the FLOAT master (char-LM scale).
// =====================================================================================================
static void bf_qact(float *x, int n) {
#pragma HLS INLINE off
    float mx = 0;
QA1: for (int i = 0; i < n; i++) { float a = x[i] < 0 ? -x[i] : x[i]; if (a > mx) mx = a; }
    if (mx < 1e-9f) mx = 1e-9f;
    int sh = (int)hls::floor(hls::log2(mx)) - X_ABITS;
    float sc = hls::ldexp(1.0f, sh);
QA2: for (int i = 0; i < n; i++) x[i] = hls::round(x[i] / sc) * sc;
}
static void bf_layernorm(const float *x, const float *gamma, const float *beta,
                         float *y, float *mean_out, float *s_out, int M) {
#pragma HLS INLINE off
BL_T: for (int t = 0; t < M; t++) {
        // NOTE: outer PIPELINE removed — the body has 3 sequential factor-8 inner reductions over X_D
        // plus transcendental log2/round, so an achievable single II is not cleanly computable. HLS now
        // schedules the 8-wide inners naturally (fast synth); inner UNROLL factor=8 kept for area.
        float sum = 0;
    BL_S: for (int d = 0; d < X_D; d++) {
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
            sum += x[t*X_D+d];
        }
        float mean = sum / X_D; float asum = 0;
    BL_A: for (int d = 0; d < X_D; d++) {
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
            float c = x[t*X_D+d]-mean; asum += c < 0 ? -c : c;
        }
        float scale = asum / X_D; if (scale < 1e-9f) scale = 1e-9f;
        int e = (int)hls::floor(hls::log2(scale));
        float mant = hls::round((scale/hls::ldexp(1.0f,e) - 1.0f)*(1<<X_MBITS)) / (float)(1<<X_MBITS);
        float s = (1.0f + mant) * hls::ldexp(1.0f,e);
        mean_out[t] = mean; s_out[t] = s;
    BL_Y: for (int d = 0; d < X_D; d++) {
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
            y[t*X_D+d] = gamma[d]*((x[t*X_D+d]-mean)/s) + beta[d];
        }
    }
}
static void bf_matmul(const float *x, const float *Wf, float *y, int M, int I, int O) {
#pragma HLS INLINE off
    // HOIST the dyadic weight out of the row-redundant inner MAC: dyad(W) depends only on (o,i), so precompute
    // it ONCE (O*I calls) instead of recomputing per row t (M*O*I). At SEQ=64 that is 64x fewer log2/round/ldexp
    // AND takes the transcendental chain off the MAC critical path. Bit-identical (same value, computed once).
    static float Wd[X_FFN*X_D];
    BM_D: for (int j = 0; j < O*I; j++) {
#pragma HLS PIPELINE II=1
                Wd[j] = bf_dyad_f(Wf[j]); }
BM_T: for (int t = 0; t < M; t++)
    BM_O: for (int o = 0; o < O; o++) {
#pragma HLS LOOP_FLATTEN off

            // partial-sum reduction: 16 INDEPENDENT lanes break the loop-carried fp-add recurrence so the inner
            // reduction pipelines at II=1 (vs II=80 when 8 unrolled adds chain into one acc). Tree-reduce at the end.
            float ps[16];
#pragma HLS ARRAY_PARTITION variable=ps complete dim=1
            for (int k=0;k<16;k++) ps[k]=0;
        BM_I: for (int i = 0; i < I; i++) {
#pragma HLS PIPELINE II=1
                ps[i&15] += Wd[o*I+i] * x[t*I+i];     // dyad(master) * x (wide) — dyad hoisted, lanes independent
            }
            float a0=(ps[0]+ps[1])+(ps[2]+ps[3]),   a1=(ps[4]+ps[5])+(ps[6]+ps[7]);
            float a2=(ps[8]+ps[9])+(ps[10]+ps[11]), a3=(ps[12]+ps[13])+(ps[14]+ps[15]);
            y[t*O+o] = (a0+a1)+(a2+a3);
        }
}

// ============================== BF16 STE BACKWARD primitives (== golden) ==============================
// dx = dyad(W)^T @ gy ; dW += gy^T @ x  (real bf16 multiplies = the new DSP). STE on the dyad round.
static void bf_linear_bwd(const float *gy, const float *Wf, const float *x,
                          float *dx, float *dW, int M, int I, int O) {
#pragma HLS INLINE off
    // HOIST dyad(W) out of the row-redundant dx MAC (see bf_matmul): precompute once (O*I) not per row (M*O*I).
    static float Wd[X_FFN*X_D];
    LR_D: for (int j = 0; j < O*I; j++) {
#pragma HLS PIPELINE II=1
                Wd[j] = bf_dyad_f(Wf[j]); }
LR_T: for (int t = 0; t < M; t++)
    LR_I: for (int i = 0; i < I; i++) {
#pragma HLS LOOP_FLATTEN off

            // partial-sum lanes -> II=1 reduction (was II=80 from 8 chained fp-adds); tree-reduce at the end
            float ps[16];
#pragma HLS ARRAY_PARTITION variable=ps complete dim=1
            for (int k=0;k<16;k++) ps[k]=0;
        LR_O: for (int o = 0; o < O; o++) {
#pragma HLS PIPELINE II=1
                ps[o&15] += Wd[o*I+i] * gy[t*O+o];     // dyad hoisted, lanes independent
            }
            float a0=(ps[0]+ps[1])+(ps[2]+ps[3]),   a1=(ps[4]+ps[5])+(ps[6]+ps[7]);
            float a2=(ps[8]+ps[9])+(ps[10]+ps[11]), a3=(ps[12]+ps[13])+(ps[14]+ps[15]);
            dx[t*I+i] = P((a0+a1)+(a2+a3));
        }
WG_O: for (int o = 0; o < O; o++)
    WG_I: for (int i = 0; i < I; i++) {
#pragma HLS LOOP_FLATTEN off

            // partial-sum lanes -> II=1 wgrad reduction (was II=80)
            float ps[16];
#pragma HLS ARRAY_PARTITION variable=ps complete dim=1
            for (int k=0;k<16;k++) ps[k]=0;
        WG_T: for (int t = 0; t < M; t++) {
#pragma HLS PIPELINE II=1
                ps[t&15] += gy[t*O+o] * x[t*I+i];
            }
            { float a0=(ps[0]+ps[1])+(ps[2]+ps[3]),   a1=(ps[4]+ps[5])+(ps[6]+ps[7]);
              float a2=(ps[8]+ps[9])+(ps[10]+ps[11]), a3=(ps[12]+ps[13])+(ps[14]+ps[15]);
              dW[o*I+i] += P((a0+a1)+(a2+a3)); }
        }
}
// LN backward EXACT to torch autograd of FDyadLN: s is stop-grad (round/floor block it) -> only mean coupling.
//   d_x = (1/s)(gxhat - mean(gxhat)),  gxhat = gy*gamma.
static void bf_ln_bwd(const float *gy, const float *gamma, const float *s, float *dx, int M) {
#pragma HLS INLINE off
LB_T: for (int t = 0; t < M; t++) {
        // NOTE: outer PIPELINE removed — body has 2 sequential factor-8 inner loops over X_D plus a
        // scalar divide, so a single achievable II isn't clean. HLS schedules the 8-wide inners naturally.
        float gxhat[X_D], mg = 0;
    LB_C: for (int d = 0; d < X_D; d++) {
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
            gxhat[d] = P(gy[t*X_D+d]*gamma[d]); mg += gxhat[d];
        }
        mg /= X_D;
    LB_D: for (int d = 0; d < X_D; d++) {
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
            dx[t*X_D+d] = P((gxhat[d]-mg)/s[t]);
        }
    }
}
// float/bf16 dyadic-Adam (== golden bf_adam_one): m=.875m+.125g, v=.875v+.125|g|, step=sign(m)2^floor(log2(|m|/v)).
static void bf_adam_one(float *Wf, const float *gW, float *m, float *v, int n, float lr) {
#pragma HLS INLINE off
AD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        float gv = gW[i], ag = gv < 0 ? -gv : gv;
        m[i] = 0.875f*m[i] + 0.125f*gv; v[i] = 0.875f*v[i] + 0.125f*ag;
        float am = m[i] < 0 ? -m[i] : m[i], ratio = am/(v[i]+1e-9f);
        float step = ratio > 0 ? hls::ldexp(1.0f, (int)hls::floor(hls::log2(ratio))) : 0.0f;
        float sgn = m[i] > 0 ? 1.0f : (m[i] < 0 ? -1.0f : 0.0f);
        Wf[i] -= lr*sgn*step;
    }
}

// ===================================================================================================
extern "C" void forge_pc_xformer_mp_block(ap_uint<512> *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    ap_uint<512> cfgw = buffer[XO_CFG / X_WL];
    int PHASE = (int)(ap_int<32>)cfgw.range(31, 0);
    int ROLE  = (int)(ap_int<32>)cfgw.range(63, 32);
    int DO_UP = (int)(ap_int<32>)cfgw.range(95, 64);
    int LRBITS = (int)(ap_int<32>)cfgw.range(127, 96);   // lr code: lr = LRBITS * 2^-20 (host stages a fixed-pt lr)
    float LR = (float)LRBITS * hls::ldexp(1.0f, -20);
    int GZERO = (int)(ap_int<32>)cfgw.range(159, 128);   // ACCUM>1: 1 = first sub-batch of a window -> zero the grad
                                                         //   accumulators; else add to them. dyadic-Adam (DO_UP) on the sum.
    int TRAIN = (int)(ap_int<32>)cfgw.range(191, 160);   // 1 = training kick: SKIP the deploy-int forward (dead weight in
                                                         //   training; bpb/backward use only the WIDE path). 0 = full/deploy.
    int PARITY = (int)(ap_int<32>)cfgw.range(223, 192);  // cross-sub-batch PIPELINING: which slot (0/1 = g&1) of the
                                                         //   per-sub-batch volatile state (wide cache + msgs + tok/tgt).
    // slot offset for EVERY per-sub-batch wide-cache / neighbour-message region (0 = slot0, MP_PAR_OFF = slot1).
    // Applied at every bstore_f/bload_f/bstore/bload of an MP_W*/MP_GRIN/MP_GROUT region below; tok/tgt use
    // MP_TOK/MP_TGT + PARITY*X_SEQ. Masters/GACC/adam-moments are NOT per-sub-batch -> never offset. PARITY is
    // always 0 in deploy (TRAIN=0) mode, so the deploy-int forward + a default-0 cfg lane stay BIT-COMPATIBLE.
    const int po = PARITY ? MP_PAR_OFF : 0;
    const int pto = PARITY * X_SEQ;                      // slot offset for the int32 tok/tgt lanes

    // ---- DEPLOY int16 master weights (drive the byte-identical forward) ----
    static data_t WQ[X_D*X_D], WK[X_D*X_D], WV[X_D*X_D], WO[X_D*X_D];
    static data_t WUP[X_FFN*X_D], WDN[X_D*X_FFN];
    static data_t LN1G[X_D], LN1B[X_D], LN2G[X_D], LN2B[X_D];
    bload(buffer, XO_WQ, WQ, X_D*X_D);  bload(buffer, XO_WK, WK, X_D*X_D);
    bload(buffer, XO_WV, WV, X_D*X_D);  bload(buffer, XO_WO, WO, X_D*X_D);
    bload(buffer, XO_WUP, WUP, X_FFN*X_D); bload(buffer, XO_WDN, WDN, X_D*X_FFN);
    bload(buffer, XO_LN1G, LN1G, X_D); bload(buffer, XO_LN1B, LN1B, X_D);
    bload(buffer, XO_LN2G, LN2G, X_D); bload(buffer, XO_LN2B, LN2B, X_D);

    // ---- FLOAT master (trainable; the STE backward + dyadic-Adam operate on these) ----
    static float FWQ[X_D*X_D], FWK[X_D*X_D], FWV[X_D*X_D], FWO[X_D*X_D];
    static float FWUP[X_FFN*X_D], FWDN[X_D*X_FFN];
    static float FLN1G[X_D], FLN1B[X_D], FLN2G[X_D], FLN2B[X_D];
    bload_f(buffer, MP_FWQ, FWQ, X_D*X_D); bload_f(buffer, MP_FWK, FWK, X_D*X_D);
    bload_f(buffer, MP_FWV, FWV, X_D*X_D); bload_f(buffer, MP_FWO, FWO, X_D*X_D);
    bload_f(buffer, MP_FWUP, FWUP, X_FFN*X_D); bload_f(buffer, MP_FWDN, FWDN, X_D*X_FFN);
    bload_f(buffer, MP_FLN1G, FLN1G, X_D); bload_f(buffer, MP_FLN1B, FLN1B, X_D);
    bload_f(buffer, MP_FLN2G, FLN2G, X_D); bload_f(buffer, MP_FLN2B, FLN2B, X_D);

    static data_t R[X_SEQ*X_D];          // int residual stream (deploy forward)

    // =========================================================================================
    // FORWARD PHASE
    // =========================================================================================
    if (PHASE == X_PH_FWD) {
        // ---- DEPLOY (int) forward: byte-identical to forge_pc_xformer_block ----
        // (skipped when TRAIN=1: it feeds neither the wide backward nor the bpb -> ~8 ms/fwd-kick saved)
        if (!TRAIN) {
        if (ROLE == X_RL_INPUT) {
            static int32_t tok[X_SEQ];
            bload32(buffer, XO_TOK, tok, X_SEQ);
            static data_t EMB[X_VOCAB*X_D], POS[X_SEQ*X_D];
            bload(buffer, XO_EMBED, EMB, X_VOCAB*X_D);
            bload(buffer, XO_POS, POS, X_SEQ*X_D);
        EMBED: for (int t = 0; t < X_SEQ; t++)
                for (int d = 0; d < X_D; d++) {
#pragma HLS PIPELINE II=1
                    int v = tok[t]; if (v < 0) v = 0; if (v >= X_VOCAB) v = X_VOCAB - 1;
                    R[t*X_D + d] = (data_t)((acc_t)EMB[v*X_D + d] + (acc_t)POS[t*X_D + d]);
                }
        } else {
            bload(buffer, XO_HIN, R, X_SEQ*X_D);
        }
        bstore(buffer, XO_RIN, R, X_SEQ*X_D);

        static data_t X1[X_SEQ*X_D]; static int32_t ln1s[X_SEQ*2];
        layernorm(R, LN1G, LN1B, X1, ln1s, X_SEQ);
        bstore(buffer, XO_X1, X1, X_SEQ*X_D);
        bstore32(buffer, XO_LN1S, ln1s, X_SEQ*2);

        static data_t Q[X_SEQ*X_D], K[X_SEQ*X_D], V[X_SEQ*X_D];
        { static acc_t Qa[X_SEQ*X_D], Ka[X_SEQ*X_D], Va[X_SEQ*X_D];
          dyad_matmul(X1, WQ, Qa, X_SEQ, X_D, X_D); rescale_to(Qa, Q, X_SEQ*X_D, X_ABITS);
          dyad_matmul(X1, WK, Ka, X_SEQ, X_D, X_D); rescale_to(Ka, K, X_SEQ*X_D, X_ABITS);
          dyad_matmul(X1, WV, Va, X_SEQ, X_D, X_D); rescale_to(Va, V, X_SEQ*X_D, X_ABITS); }
        static prob_t LUT[X_LUT_N]; { static int32_t t32[X_LUT_N]; bload32(buffer, XO_LUT, t32, X_LUT_N);
            for (int i = 0; i < X_LUT_N; i++) LUT[i] = (prob_t)t32[i]; }
        static data_t ATTN[X_SEQ*X_D];
        static int32_t Pcache[X_H*X_SEQ*X_SEQ];
    HEAD: for (int h = 0; h < X_H; h++) {
        AQ: for (int i = 0; i < X_SEQ; i++) {
                acc_t s[X_SEQ]; acc_t rowmax = (acc_t)-1 << 40; acc_t rowmin = (acc_t)1 << 40;
            SC: for (int j = 0; j < X_SEQ; j++) {
#pragma HLS PIPELINE II=1
                    if (j > i) { s[j] = (acc_t)-1 << 40; continue; }
                    acc_t l1 = 0;
                    for (int d = 0; d < X_DH; d++) {
#pragma HLS UNROLL
                        data_t diff = Q[i*X_D + h*X_DH + d] - K[j*X_D + h*X_DH + d];
                        l1 += (diff < 0) ? (acc_t)(-diff) : (acc_t)diff;
                    }
                    s[j] = -l1; if (s[j] > rowmax) rowmax = s[j]; if (s[j] < rowmin) rowmin = s[j];
                }
                int smsh = norm_shift(rowmax - rowmin, X_SMBITS);
                acc_t rowsum = 0; prob_t p[X_SEQ];
            EX: for (int j = 0; j < X_SEQ; j++) {
#pragma HLS PIPELINE II=1
                    if (j > i) { p[j] = 0; continue; }
                    int gap = (int)((rowmax - s[j]) >> smsh);
                    if (gap < 0) gap = 0; if (gap >= X_LUT_N) gap = X_LUT_N - 1;
                    p[j] = LUT[gap]; rowsum += (acc_t)p[j];
                    Pcache[(h*X_SEQ + i)*X_SEQ + j] = (int32_t)p[j];
                }
                int shn = 0; for (int b = 31; b >= 0; b--) if ((rowsum >> b) & 1) { shn = b; break; }
                shn = shn + 1;
            BL: for (int d = 0; d < X_DH; d++) {
#pragma HLS PIPELINE II=1
                    acc_t blend = 0;
                    for (int j = 0; j <= i; j++) {
                        blend += (acc_t)p[j] * (acc_t)V[j*X_D + h*X_DH + d];
                    }
                    ATTN[i*X_D + h*X_DH + d] = (data_t)(blend >> shn);
                }
            }
        }
        bstore32(buffer, XO_P, Pcache, X_H*X_SEQ*X_SEQ);
        bstore(buffer, XO_ATTNO, ATTN, X_SEQ*X_D);

        static data_t AO[X_SEQ*X_D]; { static acc_t AOa[X_SEQ*X_D]; dyad_matmul(ATTN, WO, AOa, X_SEQ, X_D, X_D); rescale_to(AOa, AO, X_SEQ*X_D, X_ABITS); }
        static data_t M[X_SEQ*X_D];
    RES1: for (int i = 0; i < X_SEQ*X_D; i++) {
#pragma HLS PIPELINE II=1
            M[i] = (data_t)((acc_t)R[i] + (acc_t)AO[i]);
        }
        bstore(buffer, XO_RMID, M, X_SEQ*X_D);

        static data_t X2[X_SEQ*X_D]; static int32_t ln2s[X_SEQ*2];
        layernorm(M, LN2G, LN2B, X2, ln2s, X_SEQ);
        bstore(buffer, XO_X2, X2, X_SEQ*X_D);
        bstore32(buffer, XO_LN2S, ln2s, X_SEQ*2);
        static data_t HD[X_SEQ*X_FFN]; static data_t RMASK[X_SEQ*X_FFN];
        { static acc_t HDa[X_SEQ*X_FFN]; dyad_matmul(X2, WUP, HDa, X_SEQ, X_D, X_FFN);
        RELU: for (int i = 0; i < X_SEQ*X_FFN; i++) {
#pragma HLS PIPELINE II=1
                bool pos = (HDa[i] > 0); RMASK[i] = pos ? (data_t)1 : (data_t)0;
                if (!pos) HDa[i] = 0;
            }
            rescale_to(HDa, HD, X_SEQ*X_FFN, X_ABITS); }
        bstore(buffer, XO_HUP, HD, X_SEQ*X_FFN);
        bstore(buffer, XO_RMASK, RMASK, X_SEQ*X_FFN);
        static data_t FO[X_SEQ*X_D]; { static acc_t FOa[X_SEQ*X_D]; dyad_matmul(HD, WDN, FOa, X_SEQ, X_FFN, X_D); rescale_to(FOa, FO, X_SEQ*X_D, X_ABITS); }
        static data_t RP[X_SEQ*X_D];
    RES2: for (int i = 0; i < X_SEQ*X_D; i++) {
#pragma HLS PIPELINE II=1
            RP[i] = (data_t)((acc_t)M[i] + (acc_t)FO[i]);
        }
        bstore(buffer, XO_RP, RP, X_SEQ*X_D);

        if (ROLE == X_RL_OUT) {
            static data_t LNFG[X_D], LNFB[X_D]; bload(buffer, XO_LNFG, LNFG, X_D); bload(buffer, XO_LNFB, LNFB, X_D);
            static data_t WHEAD[X_VOCAB*X_D]; bload(buffer, XO_WHEAD, WHEAD, X_VOCAB*X_D);
            static data_t XF[X_SEQ*X_D]; static int32_t lnfs[X_SEQ*2];
            layernorm(RP, LNFG, LNFB, XF, lnfs, X_SEQ);
            bstore32(buffer, XO_LNFS, lnfs, X_SEQ*2);
            static acc_t LG[X_SEQ*X_VOCAB]; dyad_matmul(XF, WHEAD, LG, X_SEQ, X_D, X_VOCAB);
            { static int32_t lg32[X_SEQ*X_VOCAB]; for (int i=0;i<X_SEQ*X_VOCAB;i++) lg32[i]=(int32_t)LG[i]; bstore32(buffer, XO_LOGIT, lg32, X_SEQ*X_VOCAB); }
        } else {
            static data_t HOUT[X_SEQ*X_D];
            rescale_band2(RP, HOUT, X_SEQ*X_D, X_ABITS);
            bstore(buffer, XO_HOUT, HOUT, X_SEQ*X_D);
        }
        }   // !TRAIN (deploy-int forward gate)

        // ---- WIDE (float-master) forward-for-backward: fill the MP_* cache the STE backward differentiates ----
        // == bf16 golden bf_forward, on the FLOAT master loaded above. The deployed int forward above is the one
        // the model predicts with; this wide recompute exists only to feed the textbook backward.
        static float FEMB_[1]; (void)FEMB_;
        {
            static float WR[X_SEQ*X_D], WX1[X_SEQ*X_D], WQf[X_SEQ*X_D], WKf[X_SEQ*X_D], WVf[X_SEQ*X_D];
            static float WATTN[X_SEQ*X_D], WM[X_SEQ*X_D], WX2[X_SEQ*X_D], WHD[X_SEQ*X_FFN];
            static float WRP[X_SEQ*X_D], WRMASK[X_SEQ*X_FFN];
            static float ln1m[X_SEQ], ln1ss[X_SEQ], ln2m[X_SEQ], ln2ss[X_SEQ];
            // block input R (float): INPUT embeds via the FLOAT embed/pos masters; else from the wide down-msg.
            if (ROLE == X_RL_INPUT) {
                static int32_t tok[X_SEQ]; bload32(buffer, MP_TOK + pto, tok, X_SEQ);
                static float FEMB[X_VOCAB*X_D], FPOS[X_SEQ*X_D];
                bload_f(buffer, MP_FEMB, FEMB, X_VOCAB*X_D); bload_f(buffer, MP_FPOS, FPOS, X_SEQ*X_D);
                for (int t=0;t<X_SEQ;t++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1;
                    for (int d=0;d<X_D;d++) WR[t*X_D+d]=FEMB[v*X_D+d]+FPOS[t*X_D+d]; }
            } else {
                bload_f(buffer, MP_GRIN + po, WR, X_SEQ*X_D);   // reuse: forward wide down-msg staged here by host
                // (host stages the upstream chip's wide R' into MP_GRIN before the FWD kick for HID/OUTPUT roles)
            }
            bstore_f(buffer, MP_WR + po, WR, X_SEQ*X_D);
            bf_layernorm(WR, FLN1G, FLN1B, WX1, ln1m, ln1ss, X_SEQ);
            bstore_f(buffer, MP_WX1 + po, WX1, X_SEQ*X_D);
            bf_matmul(WX1, FWQ, WQf, X_SEQ, X_D, X_D); bf_qact(WQf, X_SEQ*X_D);
            bf_matmul(WX1, FWK, WKf, X_SEQ, X_D, X_D); bf_qact(WKf, X_SEQ*X_D);
            bf_matmul(WX1, FWV, WVf, X_SEQ, X_D, X_D); bf_qact(WVf, X_SEQ*X_D);
            bstore_f(buffer, MP_WQ + po, WQf, X_SEQ*X_D); bstore_f(buffer, MP_WK + po, WKf, X_SEQ*X_D); bstore_f(buffer, MP_WV + po, WVf, X_SEQ*X_D);
            float scale = 1.0f/hls::sqrt((float)X_DH);
            static float Praw[X_H*X_SEQ*X_SEQ];
            for (int h=0;h<X_H;h++) for (int i=0;i<X_SEQ;i++) {
                float sc[X_SEQ], mx=-1e30f;
            FA_SC: for (int j=0;j<=i;j++){
#pragma HLS PIPELINE II=4
                    // II=4 = ceil(X_DH(32)/8): cycles the 8-wide inner reduction needs -> achievable, fast synth.
                    float dot=0;
                FA_SD: for(int d=0;d<X_DH;d++){
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
                        dot += WQf[i*X_D+h*X_DH+d]*WKf[j*X_D+h*X_DH+d];
                    }
                    sc[j]=dot*scale; if(sc[j]>mx)mx=sc[j]; }
                float sum=0;
            FA_EX: for(int j=0;j<=i;j++){
#pragma HLS PIPELINE II=1
                    float e=hls::exp(sc[j]-mx); Praw[(h*X_SEQ+i)*X_SEQ+j]=e; sum+=e; }
            FA_DV: for(int j=0;j<=i;j++){
#pragma HLS PIPELINE II=1
                    Praw[(h*X_SEQ+i)*X_SEQ+j] /= sum; }
            FA_BD: for(int d=0;d<X_DH;d++){
#pragma HLS PIPELINE II=8
                    // II=8 = ceil(X_SEQ(64)/8): max cycles the 8-wide inner (j<=i, bound X_SEQ) needs -> achievable.
                    float blend=0;
                FA_BJ: for(int j=0;j<=i;j++){
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
                        blend+=Praw[(h*X_SEQ+i)*X_SEQ+j]*WVf[j*X_D+h*X_DH+d];
                    }
                    WATTN[i*X_D+h*X_DH+d]=blend; }
            }
            bstore_f(buffer, MP_WPRAW + po, Praw, X_H*X_SEQ*X_SEQ);
            bf_qact(WATTN, X_SEQ*X_D);                          // char-LM: proj(qact(y))
            bstore_f(buffer, MP_WATTN + po, WATTN, X_SEQ*X_D);
            static float AOf[X_SEQ*X_D]; bf_matmul(WATTN, FWO, AOf, X_SEQ, X_D, X_D);
            for(int i=0;i<X_SEQ*X_D;i++) WM[i]=WR[i]+AOf[i];
            bstore_f(buffer, MP_WM + po, WM, X_SEQ*X_D);
            bf_layernorm(WM, FLN2G, FLN2B, WX2, ln2m, ln2ss, X_SEQ);
            bstore_f(buffer, MP_WX2 + po, WX2, X_SEQ*X_D);
            static float HDpre[X_SEQ*X_FFN]; bf_matmul(WX2, FWUP, HDpre, X_SEQ, X_D, X_FFN);
            for(int i=0;i<X_SEQ*X_FFN;i++){ WRMASK[i]=HDpre[i]>0?1.0f:0.0f; WHD[i]=WRMASK[i]>0?HDpre[i]:0.0f; }
            bf_qact(WHD, X_SEQ*X_FFN);
            bstore_f(buffer, MP_WHD + po, WHD, X_SEQ*X_FFN);
            bstore_f(buffer, MP_WRMASK + po, WRMASK, X_SEQ*X_FFN);
            static float FOf[X_SEQ*X_D]; bf_matmul(WHD, FWDN, FOf, X_SEQ, X_FFN, X_D);
            for(int i=0;i<X_SEQ*X_D;i++) WRP[i]=WM[i]+FOf[i];
            bstore_f(buffer, MP_WRP + po, WRP, X_SEQ*X_D);
            bstore_f(buffer, MP_WLN1S + po, ln1ss, X_SEQ); bstore_f(buffer, MP_WLN2S + po, ln2ss, X_SEQ);
            bstore_f(buffer, MP_WLN1M + po, ln1m, X_SEQ);  bstore_f(buffer, MP_WLN2M + po, ln2m, X_SEQ);
            if (ROLE == X_RL_OUT) {
                static float FLNFG[X_D], FLNFB[X_D]; bload_f(buffer, MP_FLNFG, FLNFG, X_D); bload_f(buffer, MP_FLNFB, FLNFB, X_D);
                static float FWHEAD[X_VOCAB*X_D]; bload_f(buffer, MP_FWHEAD, FWHEAD, X_VOCAB*X_D);
                static float WXF[X_SEQ*X_D], WLG[X_SEQ*X_VOCAB], lnfm[X_SEQ], lnfss[X_SEQ];
                bf_layernorm(WRP, FLNFG, FLNFB, WXF, lnfm, lnfss, X_SEQ);
                bf_matmul(WXF, FWHEAD, WLG, X_SEQ, X_D, X_VOCAB);
                bstore_f(buffer, MP_WXF + po, WXF, X_SEQ*X_D); bstore_f(buffer, MP_WLG + po, WLG, X_SEQ*X_VOCAB);
                bstore_f(buffer, MP_WLNFS + po, lnfss, X_SEQ); bstore_f(buffer, MP_WLNFM + po, lnfm, X_SEQ);
            } else {
                // HID/OUTPUT-as-non-final not used; but for non-OUTPUT also emit the wide R' as the down-msg
                bstore_f(buffer, MP_GROUT + po, WRP, X_SEQ*X_D);   // host relays MP_GROUT -> downstream chip's MP_GRIN
            }
        }
        return;
    }

    // =========================================================================================
    // BACKWARD PHASE — BF16 STE textbook transformer backprop (== golden bf_backward), on the float master.
    // =========================================================================================
    // load the wide forward cache
    static float WR[X_SEQ*X_D], WX1[X_SEQ*X_D], WQf[X_SEQ*X_D], WKf[X_SEQ*X_D], WVf[X_SEQ*X_D];
    static float WATTN[X_SEQ*X_D], WM[X_SEQ*X_D], WX2[X_SEQ*X_D], WHD[X_SEQ*X_FFN];
    static float WRP[X_SEQ*X_D], WRMASK[X_SEQ*X_FFN], Praw[X_H*X_SEQ*X_SEQ];
    static float ln1ss[X_SEQ], ln2ss[X_SEQ];
    bload_f(buffer, MP_WR + po, WR, X_SEQ*X_D);   bload_f(buffer, MP_WX1 + po, WX1, X_SEQ*X_D);
    bload_f(buffer, MP_WQ + po, WQf, X_SEQ*X_D);  bload_f(buffer, MP_WK + po, WKf, X_SEQ*X_D); bload_f(buffer, MP_WV + po, WVf, X_SEQ*X_D);
    bload_f(buffer, MP_WATTN + po, WATTN, X_SEQ*X_D); bload_f(buffer, MP_WM + po, WM, X_SEQ*X_D);
    bload_f(buffer, MP_WX2 + po, WX2, X_SEQ*X_D);  bload_f(buffer, MP_WHD + po, WHD, X_SEQ*X_FFN);
    bload_f(buffer, MP_WRP + po, WRP, X_SEQ*X_D);  bload_f(buffer, MP_WRMASK + po, WRMASK, X_SEQ*X_FFN);
    bload_f(buffer, MP_WPRAW + po, Praw, X_H*X_SEQ*X_SEQ);
    bload_f(buffer, MP_WLN1S + po, ln1ss, X_SEQ); bload_f(buffer, MP_WLN2S + po, ln2ss, X_SEQ);

    // float grad accumulators (zeroed; one kick = one accumulation pass, matching ACCUM=1 of the de-risk)
    static float gWQ[X_D*X_D], gWK[X_D*X_D], gWV[X_D*X_D], gWO[X_D*X_D], gWUP[X_FFN*X_D], gWDN[X_D*X_FFN];
    for (int i=0;i<X_D*X_D;i++){ gWQ[i]=0;gWK[i]=0;gWV[i]=0;gWO[i]=0; }
    for (int i=0;i<X_FFN*X_D;i++) gWUP[i]=0; for (int i=0;i<X_D*X_FFN;i++) gWDN[i]=0;

    static float gRP[X_SEQ*X_D];
    if (ROLE == X_RL_OUT) {
        // (0) CE on the WIDE logits: grad_logit = softmax(wide LG) - onehot, mean over tokens; head + LNF bwd.
        static float WXF[X_SEQ*X_D], WLG[X_SEQ*X_VOCAB], lnfss[X_SEQ];
        bload_f(buffer, MP_WXF + po, WXF, X_SEQ*X_D); bload_f(buffer, MP_WLG + po, WLG, X_SEQ*X_VOCAB);
        bload_f(buffer, MP_WLNFS + po, lnfss, X_SEQ);
        static int32_t tgt[X_SEQ]; bload32(buffer, MP_TGT + pto, tgt, X_SEQ);
        static float FWHEAD[X_VOCAB*X_D], FLNFG[X_D]; bload_f(buffer, MP_FWHEAD, FWHEAD, X_VOCAB*X_D); bload_f(buffer, MP_FLNFG, FLNFG, X_D);
        static float gLG[X_SEQ*X_VOCAB];
    CE: for (int t=0;t<X_SEQ;t++){
            // NOTE: outer PIPELINE removed — body has 3 sequential factor-8 inner loops over X_VOCAB plus
            // hls::exp, so a single achievable II isn't clean. HLS schedules the 8-wide inners naturally.
            float mx=-1e30f;
        CE_MX: for(int v=0;v<X_VOCAB;v++){
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
                float l=WLG[t*X_VOCAB+v]; if(l>mx)mx=l;
            }
            float sum=0,pr[X_VOCAB];
        CE_EX: for(int v=0;v<X_VOCAB;v++){
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
                pr[v]=hls::exp(WLG[t*X_VOCAB+v]-mx); sum+=pr[v];
            }
        CE_DV: for(int v=0;v<X_VOCAB;v++){
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
                float p=pr[v]/sum, oh=(v==tgt[t])?1.0f:0.0f; gLG[t*X_VOCAB+v]=P((p-oh)/X_SEQ);
            }
        }
        static float gWHEAD[X_VOCAB*X_D]; for (int i=0;i<X_VOCAB*X_D;i++) gWHEAD[i]=0;
        static float gXF[X_SEQ*X_D]; bf_linear_bwd(gLG, FWHEAD, WXF, gXF, gWHEAD, X_SEQ, X_D, X_VOCAB);
        bf_ln_bwd(gXF, FLNFG, lnfss, gRP, X_SEQ);
#ifdef MP_DEBUG_GRADS
        bstore_f(buffer, MP_DG_WH, gWHEAD, X_VOCAB*X_D);   // DEBUG: kernel gWHEAD vs golden bg.WHEAD
#endif
        // gradient accumulation (effective batch = ACCUM): fold gWHEAD into its DDR accumulator
        { static float aH[X_VOCAB*X_D];
          if (!GZERO) { bload_f(buffer, MP_GACC_WH, aH, X_VOCAB*X_D); for(int i=0;i<X_VOCAB*X_D;i++) gWHEAD[i]+=aH[i]; }
          bstore_f(buffer, MP_GACC_WH, gWHEAD, X_VOCAB*X_D); }
        // head dyadic-Adam (OUTPUT-only weight)
        if (DO_UP) {
            static float FWHEADm[X_VOCAB*X_D]; bload_f(buffer, MP_FWHEAD, FWHEADm, X_VOCAB*X_D);
            static float mH[X_VOCAB*X_D], vH[X_VOCAB*X_D];
            bload_f(buffer, MP_MOM+MP_OFF_FHEAD, mH, X_VOCAB*X_D); bload_f(buffer, MP_VAR+MP_OFF_FHEAD, vH, X_VOCAB*X_D);
            bf_adam_one(FWHEADm, gWHEAD, mH, vH, X_VOCAB*X_D, LR);
            bstore_f(buffer, MP_FWHEAD, FWHEADm, X_VOCAB*X_D);
            bstore_f(buffer, MP_MOM+MP_OFF_FHEAD, mH, X_VOCAB*X_D); bstore_f(buffer, MP_VAR+MP_OFF_FHEAD, vH, X_VOCAB*X_D);
        }
    } else {
        // seed gRP from the incoming WIDE (bf16) error message
        static float GRIN[X_SEQ*X_D]; bload_f(buffer, MP_GRIN + po, GRIN, X_SEQ*X_D);
        for (int i=0;i<X_SEQ*X_D;i++) gRP[i]=GRIN[i];
    }

    // (1) FFN: RP=M+FO ; FO=WDN(HD) ; HD=qact(relu(WUP(X2)))
    static float gM[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gM[i]=gRP[i];
    static float gHD[X_SEQ*X_FFN]; bf_linear_bwd(gRP, FWDN, WHD, gHD, gWDN, X_SEQ, X_FFN, X_D);
    for(int i=0;i<X_SEQ*X_FFN;i++) if(WRMASK[i]==0.0f) gHD[i]=0;
    static float gX2[X_SEQ*X_D]; bf_linear_bwd(gHD, FWUP, WX2, gX2, gWUP, X_SEQ, X_D, X_FFN);
    // (2) LN2 over M -> add into gM
    static float gM2[X_SEQ*X_D]; bf_ln_bwd(gX2, FLN2G, ln2ss, gM2, X_SEQ);
    for(int i=0;i<X_SEQ*X_D;i++) gM[i]=P(gM[i]+gM2[i]);
    // (3) attention: M=R+AO ; AO=WO(ATTN) ; ATTN=qact(P@V) ; P=softmax(QK/sqrt(DH))
    static float gR[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gR[i]=gM[i];
    static float gATTN[X_SEQ*X_D]; bf_linear_bwd(gM, FWO, WATTN, gATTN, gWO, X_SEQ, X_D, X_D);
    static float gQ[X_SEQ*X_D], gK[X_SEQ*X_D], gV[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++){gQ[i]=0;gK[i]=0;gV[i]=0;}
    float ascale = 1.0f/hls::sqrt((float)X_DH);
ABWD: for (int h=0;h<X_H;h++) for (int i=0;i<X_SEQ;i++) {
        const float* p=&Praw[(h*X_SEQ+i)*X_SEQ];
        float gp[X_SEQ];
    AB_GP: for (int j=0;j<=i;j++){
            // NOTE: outer PIPELINE removed — body has 2 sequential factor-8 inner loops over X_DH (gp dot +
            // gV scatter), so a single achievable II isn't clean. HLS schedules the 8-wide inners naturally.
            float s=0;
        AB_GPD: for(int d=0;d<X_DH;d++){
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
                s+=gATTN[i*X_D+h*X_DH+d]*WVf[j*X_D+h*X_DH+d];   // MUL (bf16)
            }
            gp[j]=s;
        AB_GVD: for(int d=0;d<X_DH;d++){
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
                gV[j*X_D+h*X_DH+d]+=P(p[j]*gATTN[i*X_D+h*X_DH+d]);   // MUL
            }
        }
        float ex=0;
    AB_EX: for(int j=0;j<=i;j++){
#pragma HLS PIPELINE II=1
            ex+=p[j]*gp[j];
        }
    AB_JAC: for (int j=0;j<=i;j++){
#pragma HLS PIPELINE II=4
            // II=4 = ceil(X_DH(32)/8): cycles the 8-wide inner gQ/gK scatter needs -> achievable, fast synth.
            float gs=P(p[j]*(gp[j]-ex))*ascale;                                                          // softmax Jacobian
        AB_JACD: for(int d=0;d<X_DH;d++){
#pragma HLS UNROLL factor=8  // restored from 2: the 150MHz CDC has the timing margin back -> ~4x faster kick
                gQ[i*X_D+h*X_DH+d]+=gs*WKf[j*X_D+h*X_DH+d]; gK[j*X_D+h*X_DH+d]+=gs*WQf[i*X_D+h*X_DH+d];
            }
        }
    }
    for(int i=0;i<X_SEQ*X_D;i++){ gQ[i]=P(gQ[i]); gK[i]=P(gK[i]); gV[i]=P(gV[i]); }
    static float gX1[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gX1[i]=0; static float tmp[X_SEQ*X_D];
    bf_linear_bwd(gQ, FWQ, WX1, tmp, gWQ, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i];
    bf_linear_bwd(gK, FWK, WX1, tmp, gWK, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i];
    bf_linear_bwd(gV, FWV, WX1, tmp, gWV, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i];
    for(int i=0;i<X_SEQ*X_D;i++) gX1[i]=P(gX1[i]);
    // (4) LN1 over R -> add into gR
    static float gR1[X_SEQ*X_D]; bf_ln_bwd(gX1, FLN1G, ln1ss, gR1, X_SEQ);
    for(int i=0;i<X_SEQ*X_D;i++) gR[i]=P(gR[i]+gR1[i]);

#ifdef MP_DEBUG_GRADS
    // DEBUG: dump the kernel's RAW bf16 gradients (pre-optimizer) so the gradcheck tb can compare them DIRECTLY
    // to the golden bf_backward (bg.*). gWHEAD/gE/gPo are dumped at their own sites (head/INPUT branches).
    bstore_f(buffer, MP_DG_WQ, gWQ, X_D*X_D);   bstore_f(buffer, MP_DG_WK, gWK, X_D*X_D);
    bstore_f(buffer, MP_DG_WV, gWV, X_D*X_D);   bstore_f(buffer, MP_DG_WO, gWO, X_D*X_D);
    bstore_f(buffer, MP_DG_WUP, gWUP, X_FFN*X_D); bstore_f(buffer, MP_DG_WDN, gWDN, X_D*X_FFN);
    bstore_f(buffer, MP_DG_GR, gR, X_SEQ*X_D);
    bstore_f(buffer, MP_DG_GQ, gQ, X_SEQ*X_D);  bstore_f(buffer, MP_DG_GK, gK, X_SEQ*X_D);
    bstore_f(buffer, MP_DG_GV, gV, X_SEQ*X_D);
#endif
    // (4b) GRADIENT ACCUMULATION (effective batch = ACCUM): fold this sub-batch's matmul-weight grads into the
    //      persistent DDR accumulators. Zero on the first sub-batch (GZERO), add otherwise; gWQ..gWDN then hold the
    //      running sum, so the dyadic-Adam below (only on the last sub-batch, DO_UP) steps on the accumulated grad.
    {
        static float aB[X_FFN*X_D];
        if (!GZERO) {
            bload_f(buffer, MP_GACC_WQ,  aB, X_D*X_D);   for(int i=0;i<X_D*X_D;i++)   gWQ[i]+=aB[i];
            bload_f(buffer, MP_GACC_WK,  aB, X_D*X_D);   for(int i=0;i<X_D*X_D;i++)   gWK[i]+=aB[i];
            bload_f(buffer, MP_GACC_WV,  aB, X_D*X_D);   for(int i=0;i<X_D*X_D;i++)   gWV[i]+=aB[i];
            bload_f(buffer, MP_GACC_WO,  aB, X_D*X_D);   for(int i=0;i<X_D*X_D;i++)   gWO[i]+=aB[i];
            bload_f(buffer, MP_GACC_WUP, aB, X_FFN*X_D); for(int i=0;i<X_FFN*X_D;i++) gWUP[i]+=aB[i];
            bload_f(buffer, MP_GACC_WDN, aB, X_D*X_FFN); for(int i=0;i<X_D*X_FFN;i++) gWDN[i]+=aB[i];
        }
        bstore_f(buffer, MP_GACC_WQ,  gWQ,  X_D*X_D);   bstore_f(buffer, MP_GACC_WK,  gWK,  X_D*X_D);
        bstore_f(buffer, MP_GACC_WV,  gWV,  X_D*X_D);   bstore_f(buffer, MP_GACC_WO,  gWO,  X_D*X_D);
        bstore_f(buffer, MP_GACC_WUP, gWUP, X_FFN*X_D); bstore_f(buffer, MP_GACC_WDN, gWDN, X_D*X_FFN);
    }

    // (5) dyadic-Adam on the float master (matmul weights; LN gammas/betas frozen)
    if (DO_UP) {
        static float mB[X_FFN*X_D], vB[X_FFN*X_D];
        bload_f(buffer, MP_MOM+MP_OFF_FWQ, mB, X_D*X_D); bload_f(buffer, MP_VAR+MP_OFF_FWQ, vB, X_D*X_D); bf_adam_one(FWQ, gWQ, mB, vB, X_D*X_D, LR); bstore_f(buffer, MP_FWQ, FWQ, X_D*X_D); bstore_f(buffer, MP_MOM+MP_OFF_FWQ, mB, X_D*X_D); bstore_f(buffer, MP_VAR+MP_OFF_FWQ, vB, X_D*X_D);
        bload_f(buffer, MP_MOM+MP_OFF_FWK, mB, X_D*X_D); bload_f(buffer, MP_VAR+MP_OFF_FWK, vB, X_D*X_D); bf_adam_one(FWK, gWK, mB, vB, X_D*X_D, LR); bstore_f(buffer, MP_FWK, FWK, X_D*X_D); bstore_f(buffer, MP_MOM+MP_OFF_FWK, mB, X_D*X_D); bstore_f(buffer, MP_VAR+MP_OFF_FWK, vB, X_D*X_D);
        bload_f(buffer, MP_MOM+MP_OFF_FWV, mB, X_D*X_D); bload_f(buffer, MP_VAR+MP_OFF_FWV, vB, X_D*X_D); bf_adam_one(FWV, gWV, mB, vB, X_D*X_D, LR); bstore_f(buffer, MP_FWV, FWV, X_D*X_D); bstore_f(buffer, MP_MOM+MP_OFF_FWV, mB, X_D*X_D); bstore_f(buffer, MP_VAR+MP_OFF_FWV, vB, X_D*X_D);
        bload_f(buffer, MP_MOM+MP_OFF_FWO, mB, X_D*X_D); bload_f(buffer, MP_VAR+MP_OFF_FWO, vB, X_D*X_D); bf_adam_one(FWO, gWO, mB, vB, X_D*X_D, LR); bstore_f(buffer, MP_FWO, FWO, X_D*X_D); bstore_f(buffer, MP_MOM+MP_OFF_FWO, mB, X_D*X_D); bstore_f(buffer, MP_VAR+MP_OFF_FWO, vB, X_D*X_D);
        bload_f(buffer, MP_MOM+MP_OFF_FWUP, mB, X_FFN*X_D); bload_f(buffer, MP_VAR+MP_OFF_FWUP, vB, X_FFN*X_D); bf_adam_one(FWUP, gWUP, mB, vB, X_FFN*X_D, LR); bstore_f(buffer, MP_FWUP, FWUP, X_FFN*X_D); bstore_f(buffer, MP_MOM+MP_OFF_FWUP, mB, X_FFN*X_D); bstore_f(buffer, MP_VAR+MP_OFF_FWUP, vB, X_FFN*X_D);
        bload_f(buffer, MP_MOM+MP_OFF_FWDN, mB, X_D*X_FFN); bload_f(buffer, MP_VAR+MP_OFF_FWDN, vB, X_D*X_FFN); bf_adam_one(FWDN, gWDN, mB, vB, X_D*X_FFN, LR); bstore_f(buffer, MP_FWDN, FWDN, X_D*X_FFN); bstore_f(buffer, MP_MOM+MP_OFF_FWDN, mB, X_D*X_FFN); bstore_f(buffer, MP_VAR+MP_OFF_FWDN, vB, X_D*X_FFN);
    }

    // (6) emit: INPUT scatters gR -> float EMB/POS (+ their adam) ; else the WIDE up-message gRout = bf16(gR).
    if (ROLE == X_RL_INPUT) {
        static int32_t tok[X_SEQ]; bload32(buffer, MP_TOK + pto, tok, X_SEQ);
        static float gE[X_VOCAB*X_D], gPo[X_SEQ*X_D]; for(int i=0;i<X_VOCAB*X_D;i++) gE[i]=0; for(int i=0;i<X_SEQ*X_D;i++) gPo[i]=0;
        for(int t=0;t<X_SEQ;t++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1;
            for(int d=0;d<X_D;d++){ gE[v*X_D+d]+=gR[t*X_D+d]; gPo[t*X_D+d]+=gR[t*X_D+d]; } }
        // gradient accumulation (effective batch = ACCUM): fold gE/gPo into their DDR accumulators
        { static float aE[X_VOCAB*X_D];
          if (!GZERO) { bload_f(buffer, MP_GACC_EMB, aE, X_VOCAB*X_D); for(int i=0;i<X_VOCAB*X_D;i++) gE[i]+=aE[i];
                        bload_f(buffer, MP_GACC_POS, aE, X_SEQ*X_D);   for(int i=0;i<X_SEQ*X_D;i++)   gPo[i]+=aE[i]; }
          bstore_f(buffer, MP_GACC_EMB, gE, X_VOCAB*X_D); bstore_f(buffer, MP_GACC_POS, gPo, X_SEQ*X_D); }
        if (DO_UP) {
            static float FEMB[X_VOCAB*X_D], FPOS[X_SEQ*X_D]; bload_f(buffer, MP_FEMB, FEMB, X_VOCAB*X_D); bload_f(buffer, MP_FPOS, FPOS, X_SEQ*X_D);
            static float mE[X_VOCAB*X_D], vE[X_VOCAB*X_D], mPo[X_SEQ*X_D], vPo[X_SEQ*X_D];
            bload_f(buffer, MP_MEMB, mE, X_VOCAB*X_D); bload_f(buffer, MP_VEMB, vE, X_VOCAB*X_D);
            bload_f(buffer, MP_MPOS, mPo, X_SEQ*X_D); bload_f(buffer, MP_VPOS, vPo, X_SEQ*X_D);
            bf_adam_one(FEMB, gE, mE, vE, X_VOCAB*X_D, LR); bf_adam_one(FPOS, gPo, mPo, vPo, X_SEQ*X_D, LR);
            bstore_f(buffer, MP_FEMB, FEMB, X_VOCAB*X_D); bstore_f(buffer, MP_FPOS, FPOS, X_SEQ*X_D);
            bstore_f(buffer, MP_MEMB, mE, X_VOCAB*X_D); bstore_f(buffer, MP_VEMB, vE, X_VOCAB*X_D);
            bstore_f(buffer, MP_MPOS, mPo, X_SEQ*X_D); bstore_f(buffer, MP_VPOS, vPo, X_SEQ*X_D);
        }
    } else {
        static float GROUT[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) GROUT[i]=P(gR[i]);   // bf16-wide up-message
        bstore_f(buffer, MP_GROUT + po, GROUT, X_SEQ*X_D);
    }
    (void)WR;
}
