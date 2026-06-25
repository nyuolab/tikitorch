// forge_pc_xformer_block.cpp — ONE transformer block as a DISTRIBUTED training shard: one physical FPGA =
// one block. A whole transformer is trained MODEL-PARALLEL across L chips, each running THIS kernel on its own
// block, exchanging ONLY a neighbour hidden-state (downstream, forward) + a neighbour error (upstream,
// backward). No chip holds the global model or the global gradient. Role (INPUT/HIDDEN/OUTPUT) + phase
// (FWD/BWD) read from a CFG word so ONE AFI loads on every slot. Weights + dyadic-Adam state PERSIST in each
// chip's DDR across all kicks.
//
//   FWD: R = embed(tok)+pos (INPUT) | HIN (else)
//        X1=LN1(R); A=attn(X1) [the ONE allowed multiply: soft sum_j p*V]; M=R+WO*A; X2=LN2(M)
//        Hd=relu(WUP*X2); R'=M+WDN*Hd
//        OUTPUT: Xf=LNF(R'); logits=WHEAD*Xf; CE -> grad_logit (cached)    else: HOUT=maxshift(R',ABITS)
//   BWD: top error gR' = (OUTPUT: head+LNF backward of grad_logit) | EIN (else)
//        PC cascade backward through FFN/LN2/attn/LN1 (FFN+proj mult-free q(gO)^T q(x); attn bwd uses the
//        multiply); accumulate weight grads; dyadic-Adam update; EOUT=maxshift(gR,EBITS) (INPUT: scatter gEmbed)
//
// Reuses forge_pc_block's dyad / norm_shift / burst I/O and the S1/attn/layernorm/dyadic-adam patterns
// (copied here, originals untouched). Every op is shift/add/compare/mux EXCEPT the attention soft-blend.
// DSP_MULT must be 0 outside attention. Part xcvu47p-fsvh2892-2L-e, 4.0 ns.
#include <stdint.h>
#include <ap_int.h>
#include "forge_pc_xformer_config.hpp"

typedef ap_int<16> data_t;     // activations / int16 weights
typedef ap_int<6>  exp_t;      // dyadic exponent (0..31 + round bump)
typedef ap_int<64> acc_t;      // accumulators + grads — int64 to match the golden (attention-bwd gp/pn*gp reach
                               // ~2^63; int48 overflowed, per the verification workflow). DSP_MULT still 0.
typedef ap_int<32> prob_t;     // Q16 probabilities / int32 grads

// ---- dyadic quantizer: v -> ±2^e (round-to-nearest pow2), table-free (forge_pc_block verbatim) ----
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
// wide variant for int32 gradients (scan 30 bits)
static inline void dyad_wide(acc_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<32> a = (v < 0) ? (ap_uint<32>)(acc_t)(-v) : (ap_uint<32>)(acc_t)v;
    sgn = (v < 0); nz = (a != 0);
    exp_t msb = 0;
PRW: for (int b = 30; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { msb = b; break; }
    }
    if (msb >= 1 && ((a >> (int)(msb - 1)) & 1)) msb = msb + 1;
    e = msb;
}
// max-shift: s = bit_length(mx) - target (forge_pc_block's FIXED convention)
static inline int norm_shift(acc_t mx, int target) {
#pragma HLS INLINE
    ap_uint<48> a = (mx < 0) ? (ap_uint<48>)(-mx) : (ap_uint<48>)mx;
    int hb = 0;
NS: for (int b = 47; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { hb = b; break; }
    }
    int s = (hb + 1) - target;
    return s > 0 ? s : 0;
}
// fine-dyadic 1/scale -> (e, nq): invq ~ (nq/4)*2^e, nq in {4..7}; NO divide, NO multiply (forge_layernorm)
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

// ---- 512-bit burst I/O between the int32 DDR buffer and on-chip data_t / int32 arrays ----
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
static inline void bload_acc(ap_uint<512> *buf, int off, acc_t *dst, int n) {
#pragma HLS INLINE off
LDA: for (int w = 0; w < n / X_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = buf[off / X_WL + w];
        for (int l = 0; l < X_WL; l++) {
#pragma HLS UNROLL
            dst[w * X_WL + l] = (acc_t)(ap_int<32>)word.range(32 * l + 31, 32 * l);
        }
    }
}
static inline void bstore_acc(ap_uint<512> *buf, int off, const acc_t *src, int n) {
#pragma HLS INLINE off
STA: for (int w = 0; w < n / X_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = 0;
        for (int l = 0; l < X_WL; l++) {
#pragma HLS UNROLL
            word.range(32 * l + 31, 32 * l) = (ap_uint<32>)(ap_int<32>)(int)src[w * X_WL + l];
        }
        buf[off / X_WL + w] = word;
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
// in-place max-shift rescale of an activation array to ~`bits` band (forge_pc_block's norm_shift)
static void rescale_band(data_t *a, int n, int bits) {
#pragma HLS INLINE off
    acc_t mx = 0;
RB1: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        acc_t v = (a[i] < 0) ? (acc_t)(-a[i]) : (acc_t)a[i]; if (v > mx) mx = v;
    }
    int sh = norm_shift(mx, bits);
RB2: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        a[i] = (data_t)((acc_t)a[i] >> sh);
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

// ---- dyadic matmul y[t][o] = sum_i q(W[o][i]) * x[t][i] (shift-add, DSP_MULT=0). M rows, I in, O out ----
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
            y[t * O + o] = acc;     // keep FULL acc (int64) — caller rescales then narrows to int16 (matches golden)
        }
    }
}
// max-shift an acc_t array to ~bits, then narrow to data_t (int16). Replaces dyad_matmul-narrow-then-rescale.
static void rescale_to(const acc_t *src, data_t *dst, int n, int bits) {
#pragma HLS INLINE off
    acc_t mx = 0;
RTo1: for (int i = 0; i < n; i++) { acc_t v = src[i] < 0 ? (acc_t)(-src[i]) : src[i]; if (v > mx) mx = v; }
    int sh = norm_shift(mx, bits);
RTo2: for (int i = 0; i < n; i++) dst[i] = (data_t)(src[i] >> sh);
}

// ---- per-token fine-dyadic LayerNorm: y = dyadic_gamma(c * 1/meanabs) + beta ; saves (e,nq) (forge_layernorm)
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
            acc_t g = gnz ? (r << (int)ge) : (acc_t)0;
            if (gnz && gs) g = -g;
            y[t * X_D + d] = (data_t)(g + (acc_t)beta[d]);
        }
    }
}

// ---- gdv: saturate |v| to int16 then round-to-nearest-pow2 (== golden g_dyad / g_dyadval, for grads/acts) ----
static inline void gdv(acc_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    acc_t a = v < 0 ? (acc_t)(-v) : v; sgn = (v < 0); nz = (a != 0); if (!nz) { e = 0; return; }
    if (a > 32767) a = 32767;
    exp_t m = 0; for (int b = 15; b >= 1; b--) { if ((a >> b) & 1) { m = b; break; } }
    if (m >= 1 && ((a >> (int)(m - 1)) & 1)) m = m + 1; e = m;
}
// gin[t][i] = sum_o dyadval(W[o][i]) * gout[t][o]   (transpose-free route, W dyadic -> shift-add, DSP_MULT=0)
static void dyad_route(const acc_t *gout, const data_t *W, acc_t *gin, int M, int I, int O) {
#pragma HLS INLINE off
RT_T: for (int t = 0; t < M; t++)
        for (int i = 0; i < I; i++) {
            acc_t acc = 0;
        RT_O: for (int o = 0; o < O; o++) {
#pragma HLS PIPELINE II=1
                bool s, nz; exp_t e; dyad(W[o*I+i], s, e, nz);
                if (nz) { acc_t term = gout[t*O+o] << e; acc += s ? (acc_t)(-term) : term; }
            }
            gin[t*I+i] = acc;
        }
}
// gW[o][i] += sum_t dyadval(go[t][o]) * dyadval(x[t][i])  (both dyadic -> sign-xor + exp-add + shift, DSP_MULT=0)
static void dyad_wgrad(const acc_t *go, const data_t *x, acc_t *gW, int M, int I, int O) {
#pragma HLS INLINE off
WG_O: for (int o = 0; o < O; o++)
        for (int i = 0; i < I; i++) {
            acc_t acc = 0;
        WG_T: for (int t = 0; t < M; t++) {
#pragma HLS PIPELINE II=1
                bool gsign, gnz; exp_t ge; gdv(go[t*O+o], gsign, ge, gnz);
                bool xs, xnz; exp_t xe; gdv((acc_t)x[t*I+i], xs, xe, xnz);
                if (gnz && xnz) { acc_t term = (acc_t)1 << ((int)ge + (int)xe); acc += (gsign ^ xs) ? (acc_t)(-term) : term; }
            }
            gW[o*I+i] += acc;
        }
}
// LN backward (straight-through-invq + gamma-STE + mean-subtract); gin += result; gG/gB accumulate
static void ln_bwd(const acc_t *gy, const data_t *xin, const data_t *gcode, const int32_t *stats,
                   acc_t *gin, acc_t *gG, acc_t *gB, int M) {
#pragma HLS INLINE off
LB_T: for (int t = 0; t < M; t++) {
        int e = stats[2*t], nq = stats[2*t+1], nqm = nq - (1 << X_MBITS);
        acc_t gc[X_D]; acc_t msum = 0;
    LB_C: for (int d = 0; d < X_D; d++) {
#pragma HLS PIPELINE II=1
            bool gs, gnz; exp_t ge; dyad(gcode[d], gs, ge, gnz);
            acc_t gch = gnz ? (gy[t*X_D+d] << (int)ge) : (acc_t)0; if (gnz && gs) gch = -gch;
            acc_t cn = gch << X_MBITS; if (nqm>=1) cn+=gch; if (nqm>=2) cn+=gch; if (nqm>=3) cn+=gch;
            acc_t g = cn >> X_MBITS; if (e>=0) g = g << e; else g = g >> (-e);
            gc[d] = g; msum += g; gB[d] += gy[t*X_D+d];
        }
        acc_t mean = msum >> X_LOG2_D;
        acc_t xs = 0; for (int d=0; d<X_D; d++) xs += xin[t*X_D+d]; acc_t xmean = xs >> X_LOG2_D;
    LB_G: for (int d = 0; d < X_D; d++) {
#pragma HLS PIPELINE II=1
            gin[t*X_D+d] += gc[d] - mean;
            acc_t cv = (acc_t)xin[t*X_D+d] - xmean;
            acc_t cn = cv << X_MBITS; if (nqm>=1) cn+=cv; if (nqm>=2) cn+=cv; if (nqm>=3) cn+=cv;
            acc_t chat = cn >> X_MBITS; if (e>=0) chat = chat << e; else chat = chat >> (-e);
            // gG += dyadval(gy)*dyadval(chat)  (saturate-then-dyad both)
            bool gysn, genz; exp_t gee; gdv(gy[t*X_D+d], gysn, gee, genz);
            bool cs, cnz; exp_t cee; gdv(chat, cs, cee, cnz);
            if (genz && cnz) { acc_t term = (acc_t)1 << ((int)gee+(int)cee); gG[d] += (gysn^cs)?(acc_t)(-term):term; }
        }
    }
}
// dyadic-Adam (one matrix): m,v EMA + pow2 step + gentle decay + int16 clamp (== golden g_adam)
static inline exp_t dyad_exp32(acc_t v) {
#pragma HLS INLINE
    ap_uint<32> a = (v<0)?(ap_uint<32>)(acc_t)(-v):(ap_uint<32>)(acc_t)v; exp_t m=0;
    for (int b=30;b>=1;b--){ if(a[b]){m=b;break;} } if (m>=1 && a[m-1] && m<30) m=m+1; return m;
}
static void dyadic_adam(data_t *W, const acc_t *gW, acc_t *m, acc_t *v, int n) {
#pragma HLS INLINE off
AD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        acc_t g = gW[i], ag = g<0?(acc_t)(-g):g;
        m[i] = m[i] - (m[i]>>X_ADAM_BSH) + (g>>X_ADAM_BSH);
        v[i] = v[i] - (v[i]>>X_ADAM_BSH) + (ag>>X_ADAM_BSH);
        int se = (int)dyad_exp32(v[i]); acc_t step = m[i] >> se; acc_t dw = step >> X_LR_SHIFT;
        acc_t nw = (acc_t)W[i] - dw - ((acc_t)W[i] >> X_WD_SHIFT);
        if (nw>32767) nw=32767; if (nw<-32767) nw=-32767; W[i] = (data_t)nw;
    }
}

// ===================================================================================================
void forge_pc_xformer_block(ap_uint<512> *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    ap_uint<512> cfgw = buffer[XO_CFG / X_WL];
    int PHASE = (int)(ap_int<32>)cfgw.range(31, 0);
    int ROLE  = (int)(ap_int<32>)cfgw.range(63, 32);
    int DO_UP = (int)(ap_int<32>)cfgw.range(95, 64);
    (void)DO_UP;

    // ---- resident weights (int16 master; dyad() at use) ----
    static data_t WQ[X_D*X_D], WK[X_D*X_D], WV[X_D*X_D], WO[X_D*X_D];
    static data_t WUP[X_FFN*X_D], WDN[X_D*X_FFN];
    static data_t LN1G[X_D], LN1B[X_D], LN2G[X_D], LN2B[X_D];
    bload(buffer, XO_WQ, WQ, X_D*X_D);  bload(buffer, XO_WK, WK, X_D*X_D);
    bload(buffer, XO_WV, WV, X_D*X_D);  bload(buffer, XO_WO, WO, X_D*X_D);
    bload(buffer, XO_WUP, WUP, X_FFN*X_D); bload(buffer, XO_WDN, WDN, X_D*X_FFN);
    bload(buffer, XO_LN1G, LN1G, X_D); bload(buffer, XO_LN1B, LN1B, X_D);
    bload(buffer, XO_LN2G, LN2G, X_D); bload(buffer, XO_LN2B, LN2B, X_D);

    static data_t R[X_SEQ*X_D];          // residual stream (block input)

    if (PHASE == X_PH_FWD) {
        // ---- block input: INPUT role embeds tokens + pos; else take the neighbour hidden-state ----
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
        bstore(buffer, XO_RIN, R, X_SEQ*X_D);   // cache block input for backward

        // ---- LN1 -> X1 ----
        static data_t X1[X_SEQ*X_D]; static int32_t ln1s[X_SEQ*2];
        layernorm(R, LN1G, LN1B, X1, ln1s, X_SEQ);
        bstore(buffer, XO_X1, X1, X_SEQ*X_D);
        bstore32(buffer, XO_LN1S, ln1s, X_SEQ*2);

        // ---- attention: Q/K/V = dyadic proj of X1; per head L1-score softmax; soft blend (THE multiply) ----
        // NOTE: this is the project's mult-free-DEPLOY attention (L1 score -Sum|Q-K| + exp2-LUT softmax). The
        // CPU recipe de-risk (pc_xformer_dyadic_ref.py) uses float DOT-PRODUCT softmax, so it does NOT validate
        // THIS attention math — the int64 golden testbench (which mirrors this kernel) is what validates it.
        static data_t Q[X_SEQ*X_D], K[X_SEQ*X_D], V[X_SEQ*X_D];
        { static acc_t Qa[X_SEQ*X_D], Ka[X_SEQ*X_D], Va[X_SEQ*X_D];
          dyad_matmul(X1, WQ, Qa, X_SEQ, X_D, X_D); rescale_to(Qa, Q, X_SEQ*X_D, X_ABITS);   // full acc -> rescale -> int16
          dyad_matmul(X1, WK, Ka, X_SEQ, X_D, X_D); rescale_to(Ka, K, X_SEQ*X_D, X_ABITS);
          dyad_matmul(X1, WV, Va, X_SEQ, X_D, X_D); rescale_to(Va, V, X_SEQ*X_D, X_ABITS); }
        static prob_t LUT[X_LUT_N]; { static int32_t t32[X_LUT_N]; bload32(buffer, XO_LUT, t32, X_LUT_N);
            for (int i = 0; i < X_LUT_N; i++) LUT[i] = (prob_t)t32[i]; }
        static data_t ATTN[X_SEQ*X_D];      // attention output (pre WO), concat over heads
        static int32_t Pcache[X_H*X_SEQ*X_SEQ];
    HEAD: for (int h = 0; h < X_H; h++) {
        AQ: for (int i = 0; i < X_SEQ; i++) {
                acc_t s[X_SEQ]; acc_t rowmax = (acc_t)-1 << 40;
            SC: for (int j = 0; j < X_SEQ; j++) {
#pragma HLS PIPELINE II=1
                    if (j > i) { s[j] = (acc_t)-1 << 40; continue; }
                    acc_t l1 = 0;
                    for (int d = 0; d < X_DH; d++) {
#pragma HLS UNROLL
                        data_t diff = Q[i*X_D + h*X_DH + d] - K[j*X_D + h*X_DH + d];
                        l1 += (diff < 0) ? (acc_t)(-diff) : (acc_t)diff;
                    }
                    s[j] = -l1; if (s[j] > rowmax) rowmax = s[j];
                }
                acc_t rowsum = 0; prob_t p[X_SEQ];
            EX: for (int j = 0; j < X_SEQ; j++) {
#pragma HLS PIPELINE II=1
                    if (j > i) { p[j] = 0; continue; }
                    int gap = (int)((rowmax - s[j]) >> X_AT_DELTA);
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
                        blend += (acc_t)p[j] * (acc_t)V[j*X_D + h*X_DH + d];   // *** THE ONE MULTIPLY *** (p Q16, V int16)
                    }
                    // normalize ONCE by the row's pow2-rounded softmax sum: p_j/rowsum cancels the Q16 scale,
                    // so >>shn (shn=bit_length(rowsum)) is the only de-scale (do NOT also >>X_AT_PFRAC — that
                    // over-divides by 2^16 and collapses the output to zero).
                    ATTN[i*X_D + h*X_DH + d] = (data_t)(blend >> shn);
                }
            }
        }
        bstore32(buffer, XO_P, Pcache, X_H*X_SEQ*X_SEQ);   // attn probs cached (int32) for backward
        bstore(buffer, XO_ATTNO, ATTN, X_SEQ*X_D);

        // ---- WO proj + residual: M = R + WO*ATTN ----
        static data_t AO[X_SEQ*X_D]; { static acc_t AOa[X_SEQ*X_D]; dyad_matmul(ATTN, WO, AOa, X_SEQ, X_D, X_D); rescale_to(AOa, AO, X_SEQ*X_D, X_ABITS); }
        static data_t M[X_SEQ*X_D];
    RES1: for (int i = 0; i < X_SEQ*X_D; i++) {
#pragma HLS PIPELINE II=1
            M[i] = (data_t)((acc_t)R[i] + (acc_t)AO[i]);
        }
        bstore(buffer, XO_RMID, M, X_SEQ*X_D);

        // ---- LN2 -> X2 ; FFN: Hd=relu(WUP*X2) ; R' = M + WDN*Hd ----
        static data_t X2[X_SEQ*X_D]; static int32_t ln2s[X_SEQ*2];
        layernorm(M, LN2G, LN2B, X2, ln2s, X_SEQ);
        bstore(buffer, XO_X2, X2, X_SEQ*X_D);
        bstore32(buffer, XO_LN2S, ln2s, X_SEQ*2);
        static data_t HD[X_SEQ*X_FFN]; static data_t RMASK[X_SEQ*X_FFN];
        { static acc_t HDa[X_SEQ*X_FFN]; dyad_matmul(X2, WUP, HDa, X_SEQ, X_D, X_FFN);
        RELU: for (int i = 0; i < X_SEQ*X_FFN; i++) {
#pragma HLS PIPELINE II=1
                bool pos = (HDa[i] > 0); RMASK[i] = pos ? (data_t)1 : (data_t)0;   // mask on the full acc (pre-rescale)
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
        bstore(buffer, XO_RP, RP, X_SEQ*X_D);   // cache R' for the backward (LNF seed on OUTPUT; gR/EOUT seed elsewhere)

        // ---- OUTPUT role: final LN + LM head + CE grad ; else: downstream hidden-state message ----
        if (ROLE == X_RL_OUT) {
            static data_t LNFG[X_D], LNFB[X_D]; bload(buffer, XO_LNFG, LNFG, X_D); bload(buffer, XO_LNFB, LNFB, X_D);
            static data_t WHEAD[X_VOCAB*X_D]; bload(buffer, XO_WHEAD, WHEAD, X_VOCAB*X_D);
            static data_t XF[X_SEQ*X_D]; static int32_t lnfs[X_SEQ*2];
            layernorm(RP, LNFG, LNFB, XF, lnfs, X_SEQ);
            bstore32(buffer, XO_LNFS, lnfs, X_SEQ*2);
            static acc_t LG[X_SEQ*X_VOCAB]; dyad_matmul(XF, WHEAD, LG, X_SEQ, X_D, X_VOCAB);   // full acc (CE uses it)
            { static int32_t lg32[X_SEQ*X_VOCAB]; for (int i=0;i<X_SEQ*X_VOCAB;i++) lg32[i]=(int32_t)LG[i]; bstore32(buffer, XO_LOGIT, lg32, X_SEQ*X_VOCAB); }
            // CE grad via exp2-LUT softmax: grad_logit = softmax(LG) - onehot(target); cached in EOUT seed below
            static prob_t LUT2[X_LUT_N]; { static int32_t t32[X_LUT_N]; bload32(buffer, XO_LUT, t32, X_LUT_N);
                for (int i = 0; i < X_LUT_N; i++) LUT2[i] = (prob_t)t32[i]; }
            static int32_t tgt[X_SEQ]; bload32(buffer, XO_TGT, tgt, X_SEQ);
            static int32_t GLOG[X_SEQ*X_VOCAB];     // grad_logit is Q16 (up to 2^16) -> MUST be int32, not int16 (wraps)
        CE: for (int t = 0; t < X_SEQ; t++) {
                acc_t mx = (acc_t)-1 << 40; for (int v = 0; v < X_VOCAB; v++) if ((acc_t)LG[t*X_VOCAB+v] > mx) mx = LG[t*X_VOCAB+v];
                acc_t sum = 0; prob_t pr[X_VOCAB];
                for (int v = 0; v < X_VOCAB; v++) {
                    int gap = (int)((mx - (acc_t)LG[t*X_VOCAB+v]) >> X_AT_DELTA);
                    if (gap < 0) gap = 0; if (gap >= X_LUT_N) gap = X_LUT_N-1;
                    pr[v] = LUT2[gap]; sum += (acc_t)pr[v];
                }
                int shn = 0; for (int b = 31; b >= 0; b--) if ((sum >> b)&1){ shn=b; break; } shn += 1;
                for (int v = 0; v < X_VOCAB; v++) {
                    acc_t pq = ((acc_t)pr[v] << X_AT_PFRAC) >> shn;        // softmax prob in Q16
                    acc_t oh = (v == tgt[t]) ? ((acc_t)1 << X_AT_PFRAC) : 0;
                    GLOG[t*X_VOCAB+v] = (int32_t)(pq - oh);                // grad_logit (Q16-scaled, int32)
                }
            }
            bstore32(buffer, XO_HOUT, GLOG, X_SEQ*X_VOCAB);  // cache grad_logit (VOCAB==D) for BWD seed; OUTPUT sends no down msg
        } else {
            static data_t HOUT[X_SEQ*X_D];
            rescale_band2(RP, HOUT, X_SEQ*X_D, X_ABITS);
            bstore(buffer, XO_HOUT, HOUT, X_SEQ*X_D);
        }
        return;
    }

    // ================================================================================================
    // BWD phase — transcription of the locally-VALIDATED int64 golden (forge_pc_xformer_golden_block.hpp
    // g_backward + g_step). Mirrors the forward in reverse. The three attention multiplies (gV/gp/gs) are the
    // only DSP-multiplies; everything else is dyadic shift-add / sign-subgradient. LN gammas/betas are FROZEN
    // (the golden found SignSGD on them explodes the logits). bit-exactness is validated by the int64 tb.
    // ---- load cached forward activations + the weights the backward needs ----
    static data_t X1[X_SEQ*X_D], X2[X_SEQ*X_D], ATTN[X_SEQ*X_D], Mmid[X_SEQ*X_D], RP[X_SEQ*X_D];  // R reuses the fn-scope decl
    static data_t HD[X_SEQ*X_FFN], RMASK[X_SEQ*X_FFN];
    bload(buffer, XO_RIN, R, X_SEQ*X_D);   bload(buffer, XO_X1, X1, X_SEQ*X_D);
    bload(buffer, XO_X2, X2, X_SEQ*X_D);   bload(buffer, XO_ATTNO, ATTN, X_SEQ*X_D);
    bload(buffer, XO_RMID, Mmid, X_SEQ*X_D); bload(buffer, XO_RP, RP, X_SEQ*X_D);
    bload(buffer, XO_HUP, HD, X_SEQ*X_FFN); bload(buffer, XO_RMASK, RMASK, X_SEQ*X_FFN);
    static int32_t ln1s[X_SEQ*2], ln2s[X_SEQ*2], lnfs[X_SEQ*2];
    bload32(buffer, XO_LN1S, ln1s, X_SEQ*2); bload32(buffer, XO_LN2S, ln2s, X_SEQ*2);
    static int32_t Pc[X_H*X_SEQ*X_SEQ]; bload32(buffer, XO_P, Pc, X_H*X_SEQ*X_SEQ);
    // recompute Q,K,V (not cached) bit-identically to the forward (X1, WQ/WK/WV, same rescale band)
    static data_t Q[X_SEQ*X_D], K[X_SEQ*X_D], V[X_SEQ*X_D];
    { static acc_t Qa[X_SEQ*X_D], Ka[X_SEQ*X_D], Va[X_SEQ*X_D];
      dyad_matmul(X1, WQ, Qa, X_SEQ, X_D, X_D); rescale_to(Qa, Q, X_SEQ*X_D, X_ABITS);
      dyad_matmul(X1, WK, Ka, X_SEQ, X_D, X_D); rescale_to(Ka, K, X_SEQ*X_D, X_ABITS);
      dyad_matmul(X1, WV, Va, X_SEQ, X_D, X_D); rescale_to(Va, V, X_SEQ*X_D, X_ABITS); }

    // grad accumulators (int32/acc) for the weights this kernel updates
    static acc_t gWQ[X_D*X_D], gWK[X_D*X_D], gWV[X_D*X_D], gWO[X_D*X_D], gWUP[X_FFN*X_D], gWDN[X_D*X_FFN];
    static acc_t gLN1G[X_D], gLN1B[X_D], gLN2G[X_D], gLN2B[X_D];
    for (int i=0;i<X_D*X_D;i++){ gWQ[i]=0;gWK[i]=0;gWV[i]=0;gWO[i]=0; }
    for (int i=0;i<X_FFN*X_D;i++) gWUP[i]=0; for (int i=0;i<X_D*X_FFN;i++) gWDN[i]=0;
    for (int d=0;d<X_D;d++){ gLN1G[d]=0;gLN1B[d]=0;gLN2G[d]=0;gLN2B[d]=0; }

    static acc_t gRP[X_SEQ*X_D]; for (int i=0;i<X_SEQ*X_D;i++) gRP[i]=0;
    if (ROLE == X_RL_OUT) {
        // (0) seed: head bwd of cached grad_logit (XO_HOUT, int32 Q16) -> gXF, gWHEAD ; LNF bwd -> gRP
        static int32_t GLOG[X_SEQ*X_VOCAB]; bload32(buffer, XO_HOUT, GLOG, X_SEQ*X_VOCAB);
        static acc_t gGLOG[X_SEQ*X_VOCAB]; for (int i=0;i<X_SEQ*X_VOCAB;i++) gGLOG[i]=(acc_t)GLOG[i];
        static data_t WHEAD[X_VOCAB*X_D]; bload(buffer, XO_WHEAD, WHEAD, X_VOCAB*X_D);
        static data_t LNFG[X_D], LNFB[X_D]; bload(buffer, XO_LNFG, LNFG, X_D); bload(buffer, XO_LNFB, LNFB, X_D);
        // XF = LNF(RP) recomputed bit-identically (not separately cached; this also fills lnfs for the LNF bwd)
        static data_t XF[X_SEQ*X_D]; layernorm(RP, LNFG, LNFB, XF, lnfs, X_SEQ);
        static acc_t gXF[X_SEQ*X_D]; dyad_route(gGLOG, WHEAD, gXF, X_SEQ, X_D, X_VOCAB);
        static acc_t gWHEAD[X_VOCAB*X_D]; for (int i=0;i<X_VOCAB*X_D;i++) gWHEAD[i]=0;
        dyad_wgrad(gGLOG, XF, gWHEAD, X_SEQ, X_D, X_VOCAB);
        static acc_t gLNFG[X_D], gLNFB[X_D]; for (int d=0;d<X_D;d++){ gLNFG[d]=0; gLNFB[d]=0; }
        ln_bwd(gXF, RP, LNFG, lnfs, gRP, gLNFG, gLNFB, X_SEQ);   // lnfs filled by the XF recompute above
        static acc_t mHD[X_VOCAB*X_D], vHD[X_VOCAB*X_D];      // head Adam state (region XO_VAR..; tb stages)
        bload_acc(buffer, XO_MOM + X_OFF_MHEAD, mHD, X_VOCAB*X_D); bload_acc(buffer, XO_VAR + X_OFF_MHEAD, vHD, X_VOCAB*X_D);
        if (DO_UP) { dyadic_adam(WHEAD, gWHEAD, mHD, vHD, X_VOCAB*X_D);
            bstore(buffer, XO_WHEAD, WHEAD, X_VOCAB*X_D);
            bstore_acc(buffer, XO_MOM + X_OFF_MHEAD, mHD, X_VOCAB*X_D); bstore_acc(buffer, XO_VAR + X_OFF_MHEAD, vHD, X_VOCAB*X_D); }
    } else {
        static data_t EIN[X_SEQ*X_D]; bload(buffer, XO_EIN, EIN, X_SEQ*X_D);
        for (int i=0;i<X_SEQ*X_D;i++) gRP[i]=(acc_t)EIN[i];
    }
    // (1) FFN bwd
    static acc_t gM[X_SEQ*X_D]; for (int i=0;i<X_SEQ*X_D;i++) gM[i]=gRP[i];
    static acc_t gHD[X_SEQ*X_FFN]; dyad_route(gRP, WDN, gHD, X_SEQ, X_FFN, X_D);
    for (int i=0;i<X_SEQ*X_FFN;i++) if (!RMASK[i]) gHD[i]=0;
    dyad_wgrad(gRP, HD, gWDN, X_SEQ, X_FFN, X_D);
    static acc_t gX2[X_SEQ*X_D]; dyad_route(gHD, WUP, gX2, X_SEQ, X_D, X_FFN);
    dyad_wgrad(gHD, X2, gWUP, X_SEQ, X_D, X_FFN);
    // (2) LN2 bwd over M
    ln_bwd(gX2, Mmid, LN2G, ln2s, gM, gLN2G, gLN2B, X_SEQ);
    // (3) attention bwd
    static acc_t gR[X_SEQ*X_D]; for (int i=0;i<X_SEQ*X_D;i++) gR[i]=gM[i];
    static acc_t gATTN[X_SEQ*X_D]; dyad_route(gM, WO, gATTN, X_SEQ, X_D, X_D);
    dyad_wgrad(gM, ATTN, gWO, X_SEQ, X_D, X_D);   // gWO declared+zeroed in the grad-accumulator block above
    static acc_t gQ[X_SEQ*X_D], gK[X_SEQ*X_D], gV[X_SEQ*X_D];
    for (int i=0;i<X_SEQ*X_D;i++){ gQ[i]=0;gK[i]=0;gV[i]=0; }
ABWD: for (int h=0; h<X_H; h++)
        for (int i=0; i<X_SEQ; i++) {
            prob_t rowsum_pn[X_SEQ]; acc_t pn[X_SEQ];
            int shn = 0; { acc_t rs=0; for (int j=0;j<=i;j++) rs += (acc_t)Pc[(h*X_SEQ+i)*X_SEQ+j]; shn = 0; for (int b=31;b>=0;b--) if ((rs>>b)&1){shn=b;break;} shn+=1; }
            for (int j=0;j<=i;j++) pn[j] = ((acc_t)Pc[(h*X_SEQ+i)*X_SEQ+j] << X_AT_PFRAC) >> shn;   // Q16 normalized prob
            (void)rowsum_pn;
            acc_t gp[X_SEQ]; for (int j=0;j<=i;j++){ acc_t s=0; for (int d=0;d<X_DH;d++) s += gATTN[i*X_D+h*X_DH+d]*(acc_t)V[j*X_D+h*X_DH+d]; gp[j]=s; }   // MUL
            acc_t ex=0; for (int j=0;j<=i;j++) ex += (pn[j]*gp[j]) >> X_AT_PFRAC;   // MUL
            for (int j=0;j<=i;j++){
                acc_t gs = (pn[j]*(gp[j]-ex)) >> X_AT_PFRAC;   // MUL (softmax bwd)
                for (int d=0; d<X_DH; d++){ acc_t df=(acc_t)Q[i*X_D+h*X_DH+d]-(acc_t)K[j*X_D+h*X_DH+d];
                    // L1 subgradient = conditional add/sub (NOT a multiply -> stays off DSP); == golden -gs*sg / gs*sg
                    if (df>0){ gQ[i*X_D+h*X_DH+d]-=gs; gK[j*X_D+h*X_DH+d]+=gs; }
                    else if (df<0){ gQ[i*X_D+h*X_DH+d]+=gs; gK[j*X_D+h*X_DH+d]-=gs; } }
                for (int d=0; d<X_DH; d++) gV[j*X_D+h*X_DH+d] += (pn[j]*gATTN[i*X_D+h*X_DH+d]) >> X_AT_PFRAC;   // MUL
            }
        }
    static acc_t gX1[X_SEQ*X_D]; for (int i=0;i<X_SEQ*X_D;i++) gX1[i]=0;
    static acc_t tmp[X_SEQ*X_D];
    dyad_route(gQ, WQ, tmp, X_SEQ, X_D, X_D); for (int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; dyad_wgrad(gQ, X1, gWQ, X_SEQ, X_D, X_D);
    dyad_route(gK, WK, tmp, X_SEQ, X_D, X_D); for (int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; dyad_wgrad(gK, X1, gWK, X_SEQ, X_D, X_D);
    dyad_route(gV, WV, tmp, X_SEQ, X_D, X_D); for (int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; dyad_wgrad(gV, X1, gWV, X_SEQ, X_D, X_D);
    // (4) LN1 bwd over R
    ln_bwd(gX1, R, LN1G, ln1s, gR, gLN1G, gLN1B, X_SEQ);
    // (5) dyadic-Adam over the matmul weights (LN gammas/betas FROZEN; embed/pos handled by INPUT role host)
    if (DO_UP) {
        static acc_t mB[X_FFN*X_D], vB[X_FFN*X_D];
        bload_acc(buffer, XO_MOM+X_OFF_WQ, mB, X_D*X_D); bload_acc(buffer, XO_VAR+X_OFF_WQ, vB, X_D*X_D); dyadic_adam(WQ, gWQ, mB, vB, X_D*X_D); bstore(buffer, XO_WQ, WQ, X_D*X_D); bstore_acc(buffer, XO_MOM+X_OFF_WQ, mB, X_D*X_D); bstore_acc(buffer, XO_VAR+X_OFF_WQ, vB, X_D*X_D);
        bload_acc(buffer, XO_MOM+X_OFF_WK, mB, X_D*X_D); bload_acc(buffer, XO_VAR+X_OFF_WK, vB, X_D*X_D); dyadic_adam(WK, gWK, mB, vB, X_D*X_D); bstore(buffer, XO_WK, WK, X_D*X_D); bstore_acc(buffer, XO_MOM+X_OFF_WK, mB, X_D*X_D); bstore_acc(buffer, XO_VAR+X_OFF_WK, vB, X_D*X_D);
        bload_acc(buffer, XO_MOM+X_OFF_WV, mB, X_D*X_D); bload_acc(buffer, XO_VAR+X_OFF_WV, vB, X_D*X_D); dyadic_adam(WV, gWV, mB, vB, X_D*X_D); bstore(buffer, XO_WV, WV, X_D*X_D); bstore_acc(buffer, XO_MOM+X_OFF_WV, mB, X_D*X_D); bstore_acc(buffer, XO_VAR+X_OFF_WV, vB, X_D*X_D);
        bload_acc(buffer, XO_MOM+X_OFF_WO, mB, X_D*X_D); bload_acc(buffer, XO_VAR+X_OFF_WO, vB, X_D*X_D); dyadic_adam(WO, gWO, mB, vB, X_D*X_D); bstore(buffer, XO_WO, WO, X_D*X_D); bstore_acc(buffer, XO_MOM+X_OFF_WO, mB, X_D*X_D); bstore_acc(buffer, XO_VAR+X_OFF_WO, vB, X_D*X_D);
        bload_acc(buffer, XO_MOM+X_OFF_WUP, mB, X_FFN*X_D); bload_acc(buffer, XO_VAR+X_OFF_WUP, vB, X_FFN*X_D); dyadic_adam(WUP, gWUP, mB, vB, X_FFN*X_D); bstore(buffer, XO_WUP, WUP, X_FFN*X_D); bstore_acc(buffer, XO_MOM+X_OFF_WUP, mB, X_FFN*X_D); bstore_acc(buffer, XO_VAR+X_OFF_WUP, vB, X_FFN*X_D);
        bload_acc(buffer, XO_MOM+X_OFF_WDN, mB, X_D*X_FFN); bload_acc(buffer, XO_VAR+X_OFF_WDN, vB, X_D*X_FFN); dyadic_adam(WDN, gWDN, mB, vB, X_D*X_FFN); bstore(buffer, XO_WDN, WDN, X_D*X_FFN); bstore_acc(buffer, XO_MOM+X_OFF_WDN, mB, X_D*X_FFN); bstore_acc(buffer, XO_VAR+X_OFF_WDN, vB, X_D*X_FFN);
    }
    // (6) emit: OUTPUT/HIDDEN -> EOUT = maxshift(gR, EBITS) up-message ; INPUT -> scatter gR into gEmbed by token
    if (ROLE == X_RL_INPUT) {
        static int32_t tok[X_SEQ]; bload32(buffer, XO_TOK, tok, X_SEQ);
        static acc_t gE[X_VOCAB*X_D]; for (int i=0;i<X_VOCAB*X_D;i++) gE[i]=0;
        for (int t=0;t<X_SEQ;t++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1; for (int d=0;d<X_D;d++) gE[v*X_D+d]+=gR[t*X_D+d]; }
        // (host applies the embedding SignSGD update from gEmbed; emit it via EOUT region as int32)
        static int32_t gE32[X_VOCAB*X_D]; for (int i=0;i<X_VOCAB*X_D;i++) gE32[i]=(int32_t)gE[i];
        bstore32(buffer, XO_EOUT, gE32, X_VOCAB*X_D <= X_SEQ*X_D ? X_VOCAB*X_D : X_SEQ*X_D);
    } else {
        static data_t EOUT[X_SEQ*X_D];
        acc_t mx=0; for (int i=0;i<X_SEQ*X_D;i++){ acc_t a=gR[i]<0?(acc_t)(-gR[i]):gR[i]; if(a>mx)mx=a; }
        int sh = norm_shift(mx, X_EBITS);
        for (int i=0;i<X_SEQ*X_D;i++) EOUT[i]=(data_t)(gR[i]>>sh);
        bstore(buffer, XO_EOUT, EOUT, X_SEQ*X_D);
    }
    (void)DO_UP;
}
