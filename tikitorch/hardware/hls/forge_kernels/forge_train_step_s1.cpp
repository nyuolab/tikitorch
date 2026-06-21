// forge_train_step_s1.cpp — the S1 integrated FORWARD training-step kernel: ONE transformer block
// (embed -> [LN1 -> attn -> res -> LN2 -> FFN -> res] -> tied max-plus LM-head -> balanced-LUT
// cross-entropy loss + grad-at-logits) for the end-to-end on-F2 transformer trainer.
//
// This kernel COMPOSES the four verified, csynth-closing kernels by INLINING their EXACT dyadic math
// as static helper functions (HLS cannot call other top-level kernels, so their loop bodies are
// reproduced verbatim, only re-parameterized to the S1 dims in forge_s1_config.hpp):
//   - forge_fwd_wide.cpp : dyad() quantizer + wdecode() + the dyadic-MAC GEMM y=q(x)·q(W)  (DSP=0)
//                          -> Q/K/V/O projections, FFN up/down, all reuse fwd_gemm()
//   - forge_attn.cpp     : L1 score + exp2-LUT softmax + soft p*V blend (the ONE multiply -> DSP)
//   - forge_layernorm.cpp: fine-dyadic LN (mean=sum>>7, meanabs rescale mbits=2, dyadic gamma, beta)
//   - forge_relu.cpp     : y=max(x,0), mask saved  (DSP=0)
//   - forge_train_step_morph_lm.cpp : max-plus LM-head (logits[v]=max_d(W_HEAD[v][d]+R[d]), argmax d*)
//                          + balanced LUT cross-entropy (grad_logit[v]=p[v]-(v==tgt?sum_p:0))
//                          + embed gather (R[d]=EMBED[tok*D+d])
//
// MULTIPLIER-FREE except the single attention soft p*V blend (forge_attn's one irreducible multiply,
// kept as a real multiply -> DSP). Everything else is sub/abs/add-tree/compare/shift only -> DSP=0.
//
// dyad() RULE is forge_fwd_wide.cpp dyad() VERBATIM (priority-encode MSB; bump exponent if the bit
// below MSB is set; sign separate; zero-guard). The CPU golden in the testbench uses the IDENTICAL
// rule, so the integrated golden IS the deploy-faithful reference (self-consistent + authoritative).
//
// ABI mirrors forge_train_step_morph_lm.cpp: extern "C" int forge_train_step_s1(buffer, mode, a1,a2,a3).
//   mode 0 -> do_fwd_loss (the whole forward + CE, this task). modes 1/2/3 stubbed (return 0).
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csim+csynth via synth_train_step_s1.tcl.

#include <stdint.h>
#include "forge_s1_config.hpp"

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_int<16>   data_t;    // activation values (int16, the morph data_t range)
typedef ap_int<6>    exp_t;     // dyadic exponent: holds W-code 0..31 + dyad() round-bump 16 (6 bits)
typedef ap_int<48>   acc_t;     // accumulators (DSP48-width signed)
typedef ap_int<32>   prob_t;    // Q16 attention probability
#else
#include <cstdint>
typedef int16_t      data_t;
typedef int32_t      exp_t;
typedef int64_t      acc_t;     // int64 outside Vitis so the g++ standalone path matches the golden
typedef int32_t      prob_t;
#endif

static const acc_t S1_NEG_INF = (acc_t)(-((acc_t)1 << 40));   // attn causal-mask sentinel
static const acc_t ML_NEG_INF = (acc_t)(-2147483647LL);       // max-plus head -inf (morph value)

// =============================================================================
// dyadic primitives — VERBATIM from forge_fwd_wide.cpp / forge_layernorm.cpp.
// =============================================================================

// dyadic-quantize a FRESH activation to (sign, exponent): round(log2|v|) via MSB + next bit.
static inline void dyad(data_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    ap_uint<16> a = (v < 0) ? (ap_uint<16>)(-v) : (ap_uint<16>)v;
#else
    uint16_t a = (v < 0) ? (uint16_t)(-v) : (uint16_t)v;
#endif
    sgn = (v < 0);
    nz = (a != 0);
    exp_t msb = 0;
PRIO: for (int b = 15; b >= 1; b--) {
#pragma HLS UNROLL
        if ((a >> b) & 1) { msb = (exp_t)b; break; }
    }
    if (msb >= 1 && ((a >> ((int)msb - 1)) & 1)) msb = msb + 1;   // bump on bit below MSB
    e = msb;
}

// decode a PRE-DYADIC weight byte code: bit7=sign, bit5=nonzero, bits4..0=exponent.
static inline void wdecode(int32_t code, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    uint32_t c = (uint32_t)(code & 0xFF);
    sgn = ((c >> 7) & 1) != 0;
    nz  = ((c >> 5) & 1) != 0;
    e   = (exp_t)(c & 0x1F);
}

static inline data_t sat16(acc_t v) {
#pragma HLS INLINE
    if (v >  32767) return  (data_t)32767;
    if (v < -32768) return (data_t)(-32768);
    return (data_t)v;
}

// fine-dyadic 1/scale -> (e, nq): VERBATIM forge_layernorm.cpp fdyad() (mbits=2; NO divide, NO mult).
static inline void fdyad(acc_t scale, int &e_out, int &nq_out) {
#pragma HLS INLINE
    if (scale < 1) scale = 1;
    int p = 0;
PRIO2: for (int b = 47; b >= 1; b--) {
#pragma HLS UNROLL
        if ((scale >> b) & 1) { p = b; break; }
    }
    bool exact_pow2 = (scale == ((acc_t)1 << p));
    int e = exact_pow2 ? (-p) : (-(p + 1));
    int nege = -e;
    acc_t numer  = (acc_t)1 << (S1_MBITS + nege);
    acc_t thresh = numer + (scale >> 1);
    int   nbase = (1 << S1_MBITS);                 // 4
    int   nq    = nbase;
    acc_t acc   = scale << S1_MBITS;               // 4*scale
CAND: for (int step = 1; step <= (1 << S1_MBITS); step++) {
#pragma HLS UNROLL
        acc += scale;
        if (acc <= thresh) nq = nbase + step;
    }
    if (nq >= (1 << (S1_MBITS + 1))) { nq = (1 << S1_MBITS); e += 1; }
    if (nq < (1 << S1_MBITS))         nq = (1 << S1_MBITS);
    e_out = e;
    nq_out = nq;
}

// =============================================================================
// fwd_gemm — the dyadic-MAC GEMM y[t][o] = sum_i q(x[t][i]) MAC q(W[o][i]) + bias(=0 in S1).
// Reproduces forge_fwd_wide.cpp's inner: x dyad-quantized once, W PRE-DYADIC (wdecode inline), the
// MAC is sign-xor + exponent-ADD + barrel-SHIFT + signed accumulate (NO multiply, DSP=0).
// Generic over (T rows, I contraction, O outputs); buffer-resident W codes ROW-MAJOR [O][I].
// x is held on-chip in xin[T][I]; y written to yout[T][O]. Bounded loops, II=1 inner.
// =============================================================================
template <int T, int I, int O>
static void fwd_gemm(const data_t xin[T][I], int32_t *buffer, int w_off, data_t yout[T][O]) {
#pragma HLS INLINE off
    // dyad-quantize x once (q(x) = ±2^ex).
    // Partition the T (row/token) dim COMPLETE so the IRED inner loop's T-way UNROLL gets T parallel
    // reads of xE/xS/xNZ[*][i] per cycle (the forge_fwd_wide dim=1 partition recipe). Without this the
    // T-way-unrolled MAC can't hit II=1 on a single BRAM port (II would balloon -> schedule blowup).
    static exp_t xE[T][I]; static bool xS[T][I]; static bool xNZ[T][I];
#pragma HLS ARRAY_PARTITION variable=xE  complete dim=1
#pragma HLS ARRAY_PARTITION variable=xS  complete dim=1
#pragma HLS ARRAY_PARTITION variable=xNZ complete dim=1
QX: for (int t = 0; t < T; t++)
        for (int i = 0; i < I; i++) {
#pragma HLS PIPELINE II=1
            dyad(xin[t][i], xS[t][i], xE[t][i], xNZ[t][i]);
        }
    // output-stationary: for each output o, stream the i-reduction over all T rows.
GO: for (int o = 0; o < O; o++) {
        acc_t acc[T];
#pragma HLS ARRAY_PARTITION variable=acc complete
        for (int t = 0; t < T; t++) {
#pragma HLS UNROLL
            acc[t] = 0;
        }
IRED:   for (int i = 0; i < I; i++) {
#pragma HLS PIPELINE II=1
            bool sw, nzw; exp_t ew;
            wdecode(buffer[w_off + o * I + i], sw, ew, nzw);
            for (int t = 0; t < T; t++) {
#pragma HLS UNROLL
                exp_t ex = xE[t][i];
                int e = (int)ew + (int)ex;
                acc_t term = (acc_t)1 << e;
                bool neg = sw ^ xS[t][i];
                bool nz  = nzw & xNZ[t][i];
                if (nz) acc[t] += neg ? (acc_t)(-term) : term;
            }
        }
        for (int t = 0; t < T; t++) {
#pragma HLS UNROLL
            yout[t][o] = sat16(acc[t]);   // cast to int16 activation range (LN/attn consume int16)
        }
    }
}

// =============================================================================
// layernorm_tok — fine-dyadic LayerNorm over one token row x[D] -> y[D], saving (e,nq).
// VERBATIM math from forge_layernorm.cpp (mean=sum>>LOG2_D, meanabs rescale, dyadic gamma, beta add).
// =============================================================================
static void layernorm_tok(const data_t x[S1_D], int32_t *buffer, int gamma_off, int beta_off,
                          data_t y[S1_D], int &e_out, int &nq_out) {
#pragma HLS INLINE off
    acc_t sum = 0;
SUMX: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
        sum += (acc_t)x[d];
    }
    acc_t mean = sum >> S1_LOG2_D;

    static data_t c[S1_D];
    acc_t asum = 0;
CEN: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t cv = (acc_t)x[d] - mean;
        c[d] = sat16(cv);
        acc_t a = (cv < 0) ? (acc_t)(-cv) : (acc_t)cv;
        asum += a;
    }
    acc_t meanabs = asum >> S1_LOG2_D;

    int e, nq;
    fdyad(meanabs, e, nq);
    e_out = e; nq_out = nq;
    int nqm4 = nq - (1 << S1_MBITS);

RESC: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t cv = (acc_t)c[d];
        acc_t cn = cv << S1_MBITS;
        if (nqm4 >= 1) cn += cv;
        if (nqm4 >= 2) cn += cv;
        if (nqm4 >= 3) cn += cv;
        acc_t r = cn >> S1_MBITS;
        if (e >= 0)  r = r << e;
        else         r = r >> (-e);
        bool gs, gnz; exp_t ge;
        wdecode(buffer[gamma_off + d], gs, ge, gnz);
        acc_t g = gnz ? (r << (int)ge) : (acc_t)0;
        if (gnz && gs) g = -g;
        data_t bd = (data_t)(int16_t)(buffer[beta_off + d] & 0xFFFF);
        acc_t yv = g + (acc_t)bd;
        y[d] = sat16(yv);
    }
}

// =============================================================================
// attn_head — one head of attention over the whole causal sequence, soft p*V blend.
// VERBATIM math from forge_attn.cpp: L1 score s[i][j]=-sum_d|Q-K|, exp2-LUT softmax, soft blend
// out[i][d]=sum_j p[i][j]*V[j][d] (the ONE real multiply -> DSP), de-scaled by >>P_FRAC. Saves the
// Q16 probs p and the argmax j* per query (for the future backward).
// Q/K/V are [L_SEQ][DH] on-chip; out written to [L_SEQ][DH]; p saved to buffer at p_off ([L][L]).
// =============================================================================
static void attn_head(const data_t Q[S1_L_SEQ][S1_DH], const data_t K[S1_L_SEQ][S1_DH],
                      const data_t V[S1_L_SEQ][S1_DH], const int32_t *attn_lut,
                      data_t out[S1_L_SEQ][S1_DH], int32_t *buffer, int p_off, int j_off) {
#pragma HLS INLINE off
    static acc_t  s[S1_L_SEQ];
    static prob_t p[S1_L_SEQ];
    static acc_t  blend[S1_DH];
#pragma HLS ARRAY_PARTITION variable=blend complete

ROW: for (int i = 0; i < S1_L_SEQ; i++) {
        acc_t rowmax = S1_NEG_INF;
        int   jstar  = 0;
        acc_t bestS  = S1_NEG_INF;
    SCORE: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { s[j] = S1_NEG_INF; continue; }
            acc_t l1 = 0;
        DABS: for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL
                data_t diff = Q[i][d] - K[j][d];
                acc_t a = (diff < 0) ? (acc_t)(-diff) : (acc_t)diff;
                l1 += a;
            }
            acc_t sc = -l1;
            s[j] = sc;
            if (sc > rowmax) rowmax = sc;
            if (sc > bestS) { bestS = sc; jstar = j; }
        }

        acc_t rowsum = 0;
    EXP: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { p[j] = 0; continue; }
            acc_t gap = (rowmax - s[j]) >> S1_AT_DELTA_SHIFT;
            int idx = (int)gap;
            if (idx < 0) idx = 0;
            if (idx > S1_LUT_N - 1) idx = S1_LUT_N - 1;
            prob_t pv = (prob_t)attn_lut[idx];
            p[j] = pv;
            rowsum += (acc_t)pv;
        }
        int sh = 0;
    NSH: for (int b = 0; b < 16; b++) {
#pragma HLS UNROLL
            if ((rowsum >> sh) > ((acc_t)1 << S1_AT_P_FRAC)) sh++;
        }
    NRM: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            p[j] = p[j] >> sh;
        }

    BINIT: for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL
            blend[d] = 0;
        }
    BLEND: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            prob_t pj = p[j];
            for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL
                blend[d] += (acc_t)((acc_t)pj * (acc_t)V[j][d]);   // THE ONE MULTIPLY -> DSP
            }
        }

    WOUT: for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
            out[i][d] = sat16(blend[d] >> S1_AT_P_FRAC);
        }
        // save probs (Q16, post-normalize) + argmax j* for the future backward.
    SAVEP: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            buffer[p_off + i * S1_L_SEQ + j] = (int32_t)p[j];
        }
        buffer[j_off + i] = jstar;
    }
}

// =============================================================================
// Mode 0: do_fwd_loss — the whole forward + CE over the whole causal sequence.
// =============================================================================
static int do_fwd_loss(int32_t *buffer) {
#pragma HLS INLINE off

    // host-staged exp2 LUTs (read at runtime; static-const ROM init reads zero on the F2 AFI).
    static int32_t ce_lut[S1_LUT_N];
    static int32_t at_lut[S1_LUT_N];
LDLUT: for (int i = 0; i < S1_LUT_N; i++) {
#pragma HLS PIPELINE II=1
        ce_lut[i] = buffer[S1_OFF_CE_LUT + i];
        at_lut[i] = buffer[S1_OFF_ATTN_LUT + i];
    }

    static int32_t toks[S1_L_SEQ];
LDTOK: for (int t = 0; t < S1_L_SEQ; t++) {
#pragma HLS PIPELINE II=1
        toks[t] = buffer[S1_OFF_TOKENS + t];
    }

    // R[t][d] = EMBED[tok[t]*D + d]  (embed gather, morph LM head idiom). Saved as R layer-0 input.
    static data_t R[S1_L_SEQ][S1_D];
EMB: for (int t = 0; t < S1_L_SEQ; t++) {
        int tok = toks[t];
        if (tok < 0) tok = 0; if (tok >= S1_V) tok = S1_V - 1;
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            data_t rv = (data_t)(int16_t)(buffer[S1_OFF_EMBED + tok * S1_D + d] & 0xFFFF);
            R[t][d] = rv;
            buffer[s1_off_R(0) + t * S1_D + d] = (int32_t)rv;   // save R_0 (block-0 input residual)
        }
    }

    // on-chip block scratch
    static data_t X1[S1_L_SEQ][S1_D];
    static data_t Qm[S1_L_SEQ][S1_D];
    static data_t Km[S1_L_SEQ][S1_D];
    static data_t Vm[S1_L_SEQ][S1_D];
    static data_t Qh[S1_L_SEQ][S1_DH];
    static data_t Kh[S1_L_SEQ][S1_DH];
    static data_t Vh[S1_L_SEQ][S1_DH];
    static data_t Ah[S1_L_SEQ][S1_DH];
    static data_t Acat[S1_L_SEQ][S1_D];
    static data_t AO[S1_L_SEQ][S1_D];
    static data_t X2[S1_L_SEQ][S1_D];
    static data_t U[S1_L_SEQ][S1_FFN];
    static data_t Hd[S1_L_SEQ][S1_FFN];
    static data_t Dn[S1_L_SEQ][S1_D];

LAYER: for (int l = 0; l < S1_N_LAYERS; l++) {
        // ---- LN1 ----
    LN1: for (int t = 0; t < S1_L_SEQ; t++) {
            data_t row[S1_D], yrow[S1_D];
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                row[d] = R[t][d];
            }
            int e, nq;
            layernorm_tok(row, buffer, s1_off_ln(S1_OFF_LN1_G, l), s1_off_ln(S1_OFF_LN1_B, l),
                          yrow, e, nq);
            buffer[s1_off_lnstats(S1_OFF_LN1_STATS, l) + 2 * t]     = e;
            buffer[s1_off_lnstats(S1_OFF_LN1_STATS, l) + 2 * t + 1] = nq;
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                X1[t][d] = yrow[d];
                buffer[s1_off_act(S1_OFF_X1, l) + t * S1_D + d] = (int32_t)yrow[d];   // save LN1 out
            }
        }

        // ---- Q/K/V projections (each [L][D] -> [L][D]) ----
        fwd_gemm<S1_L_SEQ, S1_D, S1_D>(X1, buffer, s1_off_w(S1_OFF_WQ, l, S1_D, S1_D), Qm);
        fwd_gemm<S1_L_SEQ, S1_D, S1_D>(X1, buffer, s1_off_w(S1_OFF_WK, l, S1_D, S1_D), Km);
        fwd_gemm<S1_L_SEQ, S1_D, S1_D>(X1, buffer, s1_off_w(S1_OFF_WV, l, S1_D, S1_D), Vm);

        // ---- per head: split D into H heads of DH, run soft causal attention, concat ----
    HEAD: for (int h = 0; h < S1_H; h++) {
            for (int t = 0; t < S1_L_SEQ; t++) {
                for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
                    Qh[t][d] = Qm[t][h * S1_DH + d];
                    Kh[t][d] = Km[t][h * S1_DH + d];
                    Vh[t][d] = Vm[t][h * S1_DH + d];
                }
            }
            attn_head(Qh, Kh, Vh, at_lut, Ah, buffer, s1_off_attn_p(l, h), s1_off_attn_j(l, h));
            for (int t = 0; t < S1_L_SEQ; t++) {
                for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
                    Acat[t][h * S1_DH + d] = Ah[t][d];
                }
            }
        }

        // ---- output projection [L][D] -> [L][D] then residual add ----
        fwd_gemm<S1_L_SEQ, S1_D, S1_D>(Acat, buffer, s1_off_w(S1_OFF_WO, l, S1_D, S1_D), AO);
    RES1: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                R[t][d] = sat16((acc_t)R[t][d] + (acc_t)AO[t][d]);
            }

        // ---- LN2 ----
    LN2: for (int t = 0; t < S1_L_SEQ; t++) {
            data_t row[S1_D], yrow[S1_D];
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                row[d] = R[t][d];
            }
            int e, nq;
            layernorm_tok(row, buffer, s1_off_ln(S1_OFF_LN2_G, l), s1_off_ln(S1_OFF_LN2_B, l),
                          yrow, e, nq);
            buffer[s1_off_lnstats(S1_OFF_LN2_STATS, l) + 2 * t]     = e;
            buffer[s1_off_lnstats(S1_OFF_LN2_STATS, l) + 2 * t + 1] = nq;
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                X2[t][d] = yrow[d];
                buffer[s1_off_act(S1_OFF_X2, l) + t * S1_D + d] = (int32_t)yrow[d];
            }
        }

        // ---- FFN up [L][D]->[L][FFN], ReLU (save mask), down [L][FFN]->[L][D], residual add ----
        fwd_gemm<S1_L_SEQ, S1_D, S1_FFN>(X2, buffer, s1_off_w(S1_OFF_W_UP, l, S1_FFN, S1_D), U);
    RELU: for (int t = 0; t < S1_L_SEQ; t++)
            for (int k = 0; k < S1_FFN; k++) {
#pragma HLS PIPELINE II=1
                bool act = (U[t][k] > 0);
                Hd[t][k] = act ? U[t][k] : (data_t)0;
                buffer[s1_off_relu(l) + t * S1_FFN + k] = act ? 1 : 0;
            }
        fwd_gemm<S1_L_SEQ, S1_FFN, S1_D>(Hd, buffer, s1_off_w(S1_OFF_W_DOWN, l, S1_D, S1_FFN), Dn);
    RES2: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                R[t][d] = sat16((acc_t)R[t][d] + (acc_t)Dn[t][d]);
            }

        // save the block-l output residual (block-(l+1) input).
    SAVER: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                buffer[s1_off_R(l + 1) + t * S1_D + d] = (int32_t)R[t][d];
            }
    }

    // ---- LM head + CE loss/grad per position ----
    acc_t loss_sum = 0;
HEADLOOP: for (int t = 0; t < S1_L_SEQ; t++) {
        int target = buffer[S1_OFF_TARGETS + t];
        if (target < 0) target = 0; if (target >= S1_V) target = S1_V - 1;

        // logits[v] = max_d (W_HEAD[v][d] + R[t][d]); save argmax d*.
        static data_t  logits[S1_V];
        static int32_t ad_head[S1_V];
    HEAD_V: for (int v = 0; v < S1_V; v++) {
            acc_t mx = ML_NEG_INF;
            int   idx_d = 0;
        HEAD_D: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                data_t Wvd = (data_t)(int16_t)(buffer[S1_OFF_W_HEAD + v * S1_D + d] & 0xFFFF);
                acc_t cand = (acc_t)Wvd + (acc_t)R[t][d];
                if (cand > mx) { mx = cand; idx_d = d; }
            }
            logits[v] = sat16(mx);
            ad_head[v] = idx_d;
        }
        // pred = argmax_v logits[v]
        int pred = 0;
        {
            acc_t mx = ML_NEG_INF;
        ARGV: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
                if ((acc_t)logits[v] > mx) { mx = (acc_t)logits[v]; pred = v; }
            }
        }

        // balanced LUT cross-entropy: p = ce_lut[(logit[pred]-logit[v])>>zshift] shift-normalized to
        // Q16; grad_logit[v] = p[v] - (v==target ? sum_p : 0)  (rows sum to zero). morph CE verbatim.
        static int32_t e_row[S1_V];
        int32_t S = 0;
    CE_E: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t zneg = (int32_t)logits[pred] - (int32_t)logits[v];   // >= 0
            int idx = (int)(zneg >> S1_CE_ZSHIFT);
            if (idx > 255) idx = 255;
            if (idx < 0) idx = 0;
            e_row[v] = ce_lut[idx];
            S += e_row[v];
        }
        int sh = 0;
    CE_SH: for (int b = 0; b < 8; b++) {
#pragma HLS UNROLL
            if ((S >> sh) > 65536) sh++;
        }
        int32_t psum = 0;
    CE_P: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t pq = e_row[v] >> sh;
            e_row[v] = pq;
            psum += pq;
        }
    CE_G: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t g = e_row[v] - (v == target ? psum : 0);
            buffer[S1_OFF_GRAD_LOGIT + t * S1_V + v] = g;
            buffer[S1_OFF_LOGITS     + t * S1_V + v] = (int32_t)logits[v];
        }
        // loss proxy: the normalized prob mass NOT on the target (Q16) — a monotone CE proxy that is
        // bit-exact integer (no log); the algorithm golden's float CE is the convergence authority.
        loss_sum += (acc_t)(psum - e_row[target]);

        // save head argmax route (for the future backward sparse scatter).
    SAVEH: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            buffer[S1_OFF_ARGMAX_HEAD + t * S1_V + v] = ad_head[v];
        }
    }
    buffer[S1_OFF_LOSS + 0] = (int32_t)loss_sum;

    return 0;
}

// =============================================================================
// Top-level dispatcher — mirror forge_train_step_morph_lm.cpp ABI/pragmas.
// =============================================================================
extern "C" int forge_train_step_s1(
    int32_t  buffer[S1_BUFLEN],
    int      mode,
    int      a1,
    int      a2,
    int      a3
) {
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=S1_BUFLEN \
        num_write_outstanding=16 num_read_outstanding=16 \
        max_write_burst_length=16 max_read_burst_length=16 \
        max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite  port=mode    bundle=control
#pragma HLS INTERFACE s_axilite  port=a1      bundle=control
#pragma HLS INTERFACE s_axilite  port=a2      bundle=control
#pragma HLS INTERFACE s_axilite  port=a3      bundle=control
#pragma HLS INTERFACE s_axilite  port=buffer  bundle=control
#pragma HLS INTERFACE s_axilite  port=return  bundle=control

    (void)a1; (void)a2; (void)a3;
    if (mode == 0) return do_fwd_loss(buffer);
    // modes 1 (BWD), 2 (OPT), 3 (TRAIN_RUN) stubbed for S1 phase-A; offsets are allocated.
    if (mode == 1) return 0;
    if (mode == 2) return 0;
    if (mode == 3) return 0;
    return 99;
}
