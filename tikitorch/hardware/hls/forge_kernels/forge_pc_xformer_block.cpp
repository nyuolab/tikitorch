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
typedef ap_int<48> acc_t;      // accumulators
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
static void dyad_matmul(const data_t *x, const data_t *W, data_t *y, int M, int I, int O) {
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
            y[t * O + o] = (data_t)acc;     // caller rescales (max-shift) as needed
        }
    }
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
        static data_t Q[X_SEQ*X_D], K[X_SEQ*X_D], V[X_SEQ*X_D];
        dyad_matmul(X1, WQ, Q, X_SEQ, X_D, X_D);
        dyad_matmul(X1, WK, K, X_SEQ, X_D, X_D);
        dyad_matmul(X1, WV, V, X_SEQ, X_D, X_D);
        // rescale Q,K,V to a fixed band so L1 scores + the blend stay in range
        rescale_band(Q, X_SEQ*X_D, X_ABITS); rescale_band(K, X_SEQ*X_D, X_ABITS);
        rescale_band(V, X_SEQ*X_D, X_ABITS);
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
                        blend += ((acc_t)p[j] * (acc_t)V[j*X_D + h*X_DH + d]) >> X_AT_PFRAC;  // *** MULTIPLY ***
                    }
                    ATTN[i*X_D + h*X_DH + d] = (data_t)(blend >> shn);
                }
            }
        }
        bstore32(buffer, XO_P, Pcache, X_H*X_SEQ*X_SEQ);   // attn probs cached (int32) for backward
        bstore(buffer, XO_ATTNO, ATTN, X_SEQ*X_D);

        // ---- WO proj + residual: M = R + WO*ATTN ----
        static data_t AO[X_SEQ*X_D]; dyad_matmul(ATTN, WO, AO, X_SEQ, X_D, X_D);
        rescale_band(AO, X_SEQ*X_D, X_ABITS);
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
        static data_t HD[X_SEQ*X_FFN]; dyad_matmul(X2, WUP, HD, X_SEQ, X_D, X_FFN);
        static data_t RMASK[X_SEQ*X_FFN];
    RELU: for (int i = 0; i < X_SEQ*X_FFN; i++) {
#pragma HLS PIPELINE II=1
            bool pos = (HD[i] > 0); RMASK[i] = pos ? (data_t)1 : (data_t)0;
            HD[i] = pos ? HD[i] : (data_t)0;
        }
        rescale_band(HD, X_SEQ*X_FFN, X_ABITS);
        bstore(buffer, XO_HUP, HD, X_SEQ*X_FFN);
        bstore(buffer, XO_RMASK, RMASK, X_SEQ*X_FFN);
        static data_t FO[X_SEQ*X_D]; dyad_matmul(HD, WDN, FO, X_SEQ, X_FFN, X_D);
        rescale_band(FO, X_SEQ*X_D, X_ABITS);
        static data_t RP[X_SEQ*X_D];
    RES2: for (int i = 0; i < X_SEQ*X_D; i++) {
#pragma HLS PIPELINE II=1
            RP[i] = (data_t)((acc_t)M[i] + (acc_t)FO[i]);
        }

        // ---- OUTPUT role: final LN + LM head + CE grad ; else: downstream hidden-state message ----
        if (ROLE == X_RL_OUT) {
            static data_t LNFG[X_D], LNFB[X_D]; bload(buffer, XO_LNFG, LNFG, X_D); bload(buffer, XO_LNFB, LNFB, X_D);
            static data_t WHEAD[X_VOCAB*X_D]; bload(buffer, XO_WHEAD, WHEAD, X_VOCAB*X_D);
            static data_t XF[X_SEQ*X_D]; static int32_t lnfs[X_SEQ*2];
            layernorm(RP, LNFG, LNFB, XF, lnfs, X_SEQ);
            bstore32(buffer, XO_LNFS, lnfs, X_SEQ*2);
            static data_t LG[X_SEQ*X_VOCAB]; dyad_matmul(XF, WHEAD, LG, X_SEQ, X_D, X_VOCAB);
            bstore(buffer, XO_LOGIT, LG, X_SEQ*X_VOCAB);
            // CE grad via exp2-LUT softmax: grad_logit = softmax(LG) - onehot(target); cached in EOUT seed below
            static prob_t LUT2[X_LUT_N]; { static int32_t t32[X_LUT_N]; bload32(buffer, XO_LUT, t32, X_LUT_N);
                for (int i = 0; i < X_LUT_N; i++) LUT2[i] = (prob_t)t32[i]; }
            static int32_t tgt[X_SEQ]; bload32(buffer, XO_TGT, tgt, X_SEQ);
            static data_t GLOG[X_SEQ*X_VOCAB];
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
                    GLOG[t*X_VOCAB+v] = (data_t)(pq - oh);                 // grad_logit (Q16-scaled)
                }
            }
            bstore(buffer, XO_HOUT, GLOG, X_SEQ*X_VOCAB);  // cache grad_logit (VOCAB==D) for BWD seed; OUTPUT sends no down msg
        } else {
            static data_t HOUT[X_SEQ*X_D];
            rescale_band2(RP, HOUT, X_SEQ*X_D, X_ABITS);
            bstore(buffer, XO_HOUT, HOUT, X_SEQ*X_D);
        }
        return;
    }

    // ================================================================================================
    // BWD phase — PC cascade backward through the block + dyadic-Adam update. The backward mirrors the
    // forward stages in reverse: seed the top error (OUTPUT: head+LNF backward of the cached grad_logit;
    // else EIN), then FFN bwd (mult-free q(gO)^T q(x) weight-grads per forge_wgrad), LN2 bwd, attention bwd
    // (the multiplies, per forge_attn_bwd), LN1 bwd; accumulate weight grads; dyadic-Adam (forge_dyadic_adam)
    // update of all weights; emit EOUT=maxshift(gR, EBITS) (INPUT: scatter into gEmbed).
    //
    // This forward path is csim/csynth-ready and is the inference datapath of the distributed block; the
    // backward is transcribed from the validated fixed-point recipe (pc_xformer_dyadic_ref.py) — written as
    // the next build step so it is bit-exact to the int golden in forge_pc_xformer_block_tb.cpp. Kept as a
    // marked stub here so the FORWARD csynths cleanly first (forward-first, the project's silicon discipline).
    (void)DO_UP;
}
