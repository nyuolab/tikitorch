// forge_train_step_s1_tb.cpp — Vitis HLS C-simulation testbench for the S1 integrated forward
// training-step kernel. Runs the ACTUAL kernel C++ (forge_train_step_s1 mode 0) against an INT64 CPU
// GOLDEN of the WHOLE forward+CE chain whose dyad / wdecode / fdyad / GEMM / LN / attn-soft-blend /
// max-plus-head / LUT-CE arithmetic is BIT-IDENTICAL to the kernel, and asserts max|kernel-ref| == 0
// (BIT-EXACT) over logits, grad_logit, and the saved activations (R, X1, X2, attn p, attn jstar, ReLU
// mask, head argmax). Clone of forge_pc_wgrad_tb.cpp / forge_layernorm_tb.cpp.
//
// The soft p*V blend is integer-exact (Q16 prob * int16 value summed in int64, de-scaled by >>16
// identically in kernel and golden), so the whole forward is bit-exact — no epsilon needed.
//
// Deployed-scaling operands: |embed|<600, weights as dyadic byte-codes (|gain| up to 2^4), tokens in
// [0,V), the exp2 LUTs host-staged from forge_ce_lut.hpp. The int48 headroom check confirms peak |acc|
// stays inside 2^31 at this range.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>   // only needed under Vitis; the standalone g++ path uses long/int32_t only
#endif

#include "forge_s1_config.hpp"
#include "forge_ce_lut.hpp"

extern "C" int forge_train_step_s1(int32_t *buffer, int mode, int a1, int a2, int a3);   // DUT

// ---------------- golden helpers (BIT-IDENTICAL to the kernel) ----------------

static inline long sat16(long v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return v;
}

// dyad: forge_fwd_wide dyad() (priority MSB + bump on bit below MSB). returns (sgn,e,nz) like kernel.
static void ref_dyad(long v, int &sgn, int &e, int &nz) {
    long a = v < 0 ? -v : v;
    a &= 0xFFFFL;                 // int16 magnitude (matches kernel dyad()'s ap_uint<16> truncation)
    sgn = (v < 0) ? 1 : 0;
    nz  = (a != 0) ? 1 : 0;
    int msb = 0;
    for (int b = 15; b >= 1; b--) { if ((a >> b) & 1) { msb = b; break; } }
    if (msb >= 1 && ((a >> (msb - 1)) & 1)) msb++;
    e = msb;
}

// FULL-precision dyad for the wgrad gO operand (matches the kernel dyad_wide: scan the whole 31-bit
// magnitude, NO int16 truncation). Used ONLY for the gO side of ref_wgrad.
static void ref_dyad_wide(long v, int &sgn, int &e, int &nz) {
    long a = v < 0 ? -v : v;
    a &= 0x7FFFFFFFL;            // 31-bit magnitude (matches kernel dyad_wide's ap_uint<32> of an int32)
    sgn = (v < 0) ? 1 : 0;
    nz  = (a != 0) ? 1 : 0;
    int msb = 0;
    for (int b = 30; b >= 1; b--) { if ((a >> b) & 1) { msb = b; break; } }
    if (msb >= 1 && ((a >> (msb - 1)) & 1)) msb++;
    e = msb;
}

static void ref_wdecode(int32_t code, int &sgn, int &e, int &nz) {
    unsigned c = (unsigned)(code & 0xFF);
    sgn = (c >> 7) & 1;
    nz  = (c >> 5) & 1;
    e   = c & 0x1F;
}

// fdyad: forge_layernorm fdyad() (no divide; bounded shift-add candidate compare).
static void ref_fdyad(long scale, int &e_out, int &nq_out) {
    if (scale < 1) scale = 1;
    int p = 0;
    for (int b = 47; b >= 1; b--) { if ((scale >> b) & 1L) { p = b; break; } }
    bool exact_pow2 = (scale == (1L << p));
    int e = exact_pow2 ? (-p) : (-(p + 1));
    int nege = -e;
    long numer  = 1L << (S1_MBITS + nege);
    long thresh = numer + (scale >> 1);
    int  nbase  = (1 << S1_MBITS);
    int  nq     = nbase;
    long acc    = scale << S1_MBITS;
    for (int step = 1; step <= (1 << S1_MBITS); step++) { acc += scale; if (acc <= thresh) nq = nbase + step; }
    if (nq >= (1 << (S1_MBITS + 1))) { nq = (1 << S1_MBITS); e += 1; }
    if (nq <  (1 << S1_MBITS))         nq = (1 << S1_MBITS);
    e_out = e; nq_out = nq;
}

// GEMM golden: y[t][o] = sum_i q(x[t][i]) MAC q(W[o][i]); W ROW-MAJOR [O][I]; sat16 to int16.
template <int T, int I, int O>
static void ref_gemm(const long x[T][I], const int32_t *buf, int w_off, long y[T][O]) {
    for (int o = 0; o < O; o++) {
        for (int t = 0; t < T; t++) {
            long acc = 0;
            for (int i = 0; i < I; i++) {
                int sw, ew, nzw; ref_wdecode(buf[w_off + o * I + i], sw, ew, nzw);
                int sx, ex, nzx; ref_dyad(x[t][i], sx, ex, nzx);
                if (nzw && nzx) {
                    long term = 1L << (ew + ex);
                    acc += (sw ^ sx) ? -term : term;
                }
            }
            y[t][o] = sat16(acc);
        }
    }
}

// LN golden: fine-dyadic per token; writes y, returns (e,nq).
static void ref_ln(const long x[S1_D], const int32_t *buf, int gamma_off, int beta_off,
                   long y[S1_D], int &e_out, int &nq_out) {
    long sum = 0; for (int d = 0; d < S1_D; d++) sum += x[d];
    long mean = sum >> S1_LOG2_D;
    long c[S1_D]; long asum = 0;
    for (int d = 0; d < S1_D; d++) { long cv = x[d] - mean; c[d] = sat16(cv); asum += cv < 0 ? -cv : cv; }
    long meanabs = asum >> S1_LOG2_D;
    int e, nq; ref_fdyad(meanabs, e, nq); e_out = e; nq_out = nq;
    int nqm4 = nq - (1 << S1_MBITS);
    for (int d = 0; d < S1_D; d++) {
        long cv = c[d];
        long cn = cv << S1_MBITS;
        if (nqm4 >= 1) cn += cv; if (nqm4 >= 2) cn += cv; if (nqm4 >= 3) cn += cv;
        long r = cn >> S1_MBITS;
        if (e >= 0) r = r << e; else r = r >> (-e);
        int gs, ge, gnz; ref_wdecode(buf[gamma_off + d], gs, ge, gnz);
        long g = gnz ? (r << ge) : 0;
        if (gnz && gs) g = -g;
        long bd = (int16_t)(buf[beta_off + d] & 0xFFFF);
        y[d] = sat16(g + bd);
    }
}

// attn head golden: soft p*V blend; writes out[L][DH], p[L][L], jstar[L].
static void ref_attn(const long Q[S1_L_SEQ][S1_DH], const long K[S1_L_SEQ][S1_DH],
                     const long V[S1_L_SEQ][S1_DH], const int32_t *at_lut,
                     long out[S1_L_SEQ][S1_DH], long p_save[S1_L_SEQ][S1_L_SEQ], int jstar_save[S1_L_SEQ]) {
    const long NEG_INF = -(1L << 40);
    for (int i = 0; i < S1_L_SEQ; i++) {
        long s[S1_L_SEQ];
        long rowmax = NEG_INF; int jstar = 0; long bestS = NEG_INF;
        for (int j = 0; j < S1_L_SEQ; j++) {
            if (j > i) { s[j] = NEG_INF; continue; }
            long l1 = 0;
            for (int d = 0; d < S1_DH; d++) { long diff = Q[i][d] - K[j][d]; l1 += diff < 0 ? -diff : diff; }
            long sc = -l1; s[j] = sc;
            if (sc > rowmax) rowmax = sc;
            if (sc > bestS) { bestS = sc; jstar = j; }
        }
        long pp[S1_L_SEQ]; long rowsum = 0;
        for (int j = 0; j < S1_L_SEQ; j++) {
            if (j > i) { pp[j] = 0; continue; }
            long gap = (rowmax - s[j]) >> S1_AT_DELTA_SHIFT;
            int idx = (int)gap; if (idx < 0) idx = 0; if (idx > S1_LUT_N - 1) idx = S1_LUT_N - 1;
            pp[j] = at_lut[idx]; rowsum += pp[j];
        }
        int sh = 0; for (int b = 0; b < 16; b++) { if ((rowsum >> sh) > (1L << S1_AT_P_FRAC)) sh++; }
        for (int j = 0; j < S1_L_SEQ; j++) { pp[j] = pp[j] >> sh; p_save[i][j] = pp[j]; }
        long blend[S1_DH]; for (int d = 0; d < S1_DH; d++) blend[d] = 0;
        for (int j = 0; j <= i; j++)
            for (int d = 0; d < S1_DH; d++) blend[d] += (long)pp[j] * (long)V[j][d];   // the one multiply
        for (int d = 0; d < S1_DH; d++) out[i][d] = sat16(blend[d] >> S1_AT_P_FRAC);
        jstar_save[i] = jstar;
    }
}

static int32_t make_code(int sign, int nonzero, int exp) {   // dyadic byte code (sign,nz,exp)
    int c = (exp & 0x1F);
    if (nonzero) c |= (1 << 5);
    if (sign)    c |= (1 << 7);
    return c;
}

static inline long sat32g(long v) {
    if (v >  2147483647L)  return  2147483647L;
    if (v < -2147483648L)  return -2147483648L;
    return v;
}

// ---------------- BACKWARD golden helpers (BIT-IDENTICAL to do_bwd) ----------------

// xgrad: gx[t][i] = sum_o gO[t][o] * q(W[o][i]); gO FULL, W dyadic-shift (sign folded). sat32.
template <int T, int I, int O>
static void ref_xgrad(const long gO[T][O], const int32_t *buf, int w_off, long gx[T][I]) {
    for (int i = 0; i < I; i++)
        for (int t = 0; t < T; t++) {
            long acc = 0;
            for (int o = 0; o < O; o++) {
                int sw, ew, nzw; ref_wdecode(buf[w_off + o * I + i], sw, ew, nzw);
                if (nzw) { long sh = gO[t][o] << ew; acc += sw ? -sh : sh; }
            }
            gx[t][i] = sat32g(acc);
        }
}

// wgrad: gW[o][i] += sum_t q(gO[t][o]) * q(x[t][i]); BOTH dyadic -> ±2^(eo+ei). x is int16 activation.
template <int T, int I, int O>
static void ref_wgrad(const long gO[T][O], const long xin[T][I], long *gW, int gw_off) {
    for (int o = 0; o < O; o++)
        for (int i = 0; i < I; i++) {
            long acc = 0;
            for (int t = 0; t < T; t++) {
                int so, eo, nzo; ref_dyad_wide(gO[t][o], so, eo, nzo);          // FULL int32 grad (no wrap)
                int sx, ex, nzx; ref_dyad((int16_t)xin[t][i], sx, ex, nzx);     // x is int16 activation
                if (nzo && nzx) { long term = 1L << (eo + ex); acc += (so ^ sx) ? -term : term; }
            }
            gW[gw_off + o * I + i] = sat32g(gW[gw_off + o * I + i] + sat32g(acc));
        }
}

// attn bwd golden: re-derive p from Q,K (== ref_attn) then gQ/gK/gV. gOut FULL. Accumulates into gQ/gK/gV.
static void ref_attn_bwd(const long Q[S1_L_SEQ][S1_DH], const long K[S1_L_SEQ][S1_DH],
                         const long V[S1_L_SEQ][S1_DH], const long gOut[S1_L_SEQ][S1_DH],
                         const int32_t *at_lut,
                         long gQ[S1_L_SEQ][S1_DH], long gK[S1_L_SEQ][S1_DH], long gV[S1_L_SEQ][S1_DH]) {
    const long NEG_INF = -(1L << 40);
    for (int i = 0; i < S1_L_SEQ; i++) {
        long s[S1_L_SEQ]; long rowmax = NEG_INF;
        for (int j = 0; j < S1_L_SEQ; j++) {
            if (j > i) { s[j] = NEG_INF; continue; }
            long l1 = 0;
            for (int d = 0; d < S1_DH; d++) { long diff = Q[i][d] - K[j][d]; l1 += diff < 0 ? -diff : diff; }
            long sc = -l1; s[j] = sc; if (sc > rowmax) rowmax = sc;
        }
        long p[S1_L_SEQ]; long rowsum = 0;
        for (int j = 0; j < S1_L_SEQ; j++) {
            if (j > i) { p[j] = 0; continue; }
            long gap = (rowmax - s[j]) >> S1_AT_DELTA_SHIFT;
            int idx = (int)gap; if (idx < 0) idx = 0; if (idx > S1_LUT_N - 1) idx = S1_LUT_N - 1;
            p[j] = at_lut[idx]; rowsum += p[j];
        }
        int sh = 0; for (int b = 0; b < 16; b++) if ((rowsum >> sh) > (1L << S1_AT_P_FRAC)) sh++;
        for (int j = 0; j < S1_L_SEQ; j++) p[j] = p[j] >> sh;
        // (b) gp + expnum
        long gp[S1_L_SEQ]; long expnum = 0;
        for (int j = 0; j <= i; j++) {
            long dot = 0; for (int d = 0; d < S1_DH; d++) dot += gOut[i][d] * V[j][d];   // multiply
            gp[j] = dot; expnum += p[j] * dot;                                            // multiply
        }
        long ex = expnum >> S1_AT_P_FRAC;
        // (c) gs
        long gs[S1_L_SEQ];
        for (int j = 0; j < S1_L_SEQ; j++) {
            if (j > i) { gs[j] = 0; continue; }
            long centered = gp[j] - ex; long prod = p[j] * centered;                      // multiply
            gs[j] = prod >> S1_AT_P_FRAC;
        }
        // (a) gV
        for (int j = 0; j <= i; j++)
            for (int d = 0; d < S1_DH; d++) {
                long t = p[j] * gOut[i][d];                                               // multiply
                gV[j][d] = sat32g(gV[j][d] + (t >> S1_AT_P_FRAC));
            }
        // (d) L1 subgradient (mult-free)
        for (int j = 0; j <= i; j++) {
            long gsj = gs[j];
            for (int d = 0; d < S1_DH; d++) {
                long diff = Q[i][d] - K[j][d];
                if (diff > 0)      { gQ[i][d] = sat32g(gQ[i][d] - gsj); gK[j][d] = sat32g(gK[j][d] + gsj); }
                else if (diff < 0) { gQ[i][d] = sat32g(gQ[i][d] + gsj); gK[j][d] = sat32g(gK[j][d] - gsj); }
            }
        }
    }
}

// LN backward (matches do_bwd LN block + ln_gamma_grad). gY -> gR_add (returned), g_gamma, g_beta.
// Rin = the PRE-NORM residual row (for the gamma-grad's dyad(r)); (e,nq) = saved stats per token.
static void ref_ln_bwd(const long Rin[S1_L_SEQ][S1_D], const int32_t *gcode_buf, int gamma_off,
                       const long gY[S1_L_SEQ][S1_D], const int en[S1_L_SEQ], const int nqn[S1_L_SEQ],
                       long gR_add[S1_L_SEQ][S1_D], long *gGamma, long *gBeta, int gln_off) {
    for (int t = 0; t < S1_L_SEQ; t++) {
        int e = en[t], nq = nqn[t]; int nqm4 = nq - (1 << S1_MBITS);
        // (gx path) g_chat = gY << gamma_exp (sign-folded); g_c = apply_invq; gx = g_c - mean(g_c)
        long g_c[S1_D]; long gsum = 0;
        for (int d = 0; d < S1_D; d++) {
            int gs, ge, gnz; ref_wdecode(gcode_buf[gamma_off + d], gs, ge, gnz);
            long gchat = gnz ? (gY[t][d] << ge) : 0; if (gnz && gs) gchat = -gchat;
            long cn = gchat << S1_MBITS;
            if (nqm4 >= 1) cn += gchat; if (nqm4 >= 2) cn += gchat; if (nqm4 >= 3) cn += gchat;
            long r = cn >> S1_MBITS;
            if (e >= 0) r = r << e; else r = r >> (-e);
            g_c[d] = r; gsum += r;
        }
        long mean = gsum >> S1_LOG2_D;
        for (int d = 0; d < S1_D; d++) {
            gR_add[t][d] = g_c[d] - mean;
            gBeta[gln_off + d] = sat32g(gBeta[gln_off + d] + gY[t][d]);
        }
        // (g_gamma) recompute r = pre-gamma rescaled c from Rin; g_gamma += gY << dyad(sat16(r)).exp
        long sum = 0; for (int d = 0; d < S1_D; d++) sum += Rin[t][d];
        long mn = sum >> S1_LOG2_D;
        for (int d = 0; d < S1_D; d++) {
            long cv = sat16(Rin[t][d] - mn);
            long cn = cv << S1_MBITS;
            if (nqm4 >= 1) cn += cv; if (nqm4 >= 2) cn += cv; if (nqm4 >= 3) cn += cv;
            long r = cn >> S1_MBITS;
            if (e >= 0) r = r << e; else r = r >> (-e);
            int rs, re, rnz; ref_dyad(sat16(r), rs, re, rnz);
            if (rnz) { long term = gY[t][d] << re; gGamma[gln_off + d] = sat32g(gGamma[gln_off + d] + (rs ? -term : term)); }
        }
    }
}

// ---------------- DYADIC-ADAM (mode 2) golden helpers (BIT-IDENTICAL to do_opt) ----------------

// round(log2|v|) via MSB + round-to-nearest bump; clamp to 31. == kernel s1_dyad_exp / forge_dyadic_adam.
static int ref_dyad_exp(long v) {
    unsigned long a = v < 0 ? (unsigned long)(-(long long)v) : (unsigned long)v;
    a &= 0xFFFFFFFFUL;                                  // 32-bit magnitude (matches ap_uint<32> of int32)
    int msb = 0;
    for (int b = 31; b >= 1; b--) { if ((a >> b) & 1UL) { msb = b; break; } }
    if (msb >= 1 && ((a >> (msb - 1)) & 1UL) && msb < 31) msb++;
    return msb;
}

// re-quantize an int16 master into the PRE-DYADIC byte code (== kernel s1_encode). dyad(W_fp)->pack.
static int32_t ref_s1_encode(long wfp) {
    int sgn, e, nz; ref_dyad((int16_t)wfp, sgn, e, nz);   // dyad on the int16 master (matches kernel dyad)
    int ee = e; if (ee > 31) ee = 31;
    int c = (ee & 0x1F);
    if (nz)  c |= (1 << 5);
    if (sgn) c |= (1 << 7);
    return (int32_t)(int8_t)(unsigned char)c;
}

static inline long sat32g_fwd(long v) {   // == kernel sat32 (int32 clamp); local copy for the encode-fine
    if (v >  2147483647L)  return  2147483647L;
    if (v < -2147483648L)  return -2147483648L;
    return v;
}

// FINE FIXED-POINT code re-quantize (== kernel s1_encode_fine). wq = int32 master at scale 2^Q_CODE;
// round-half-away-from-zero to the integer code value, dyad() it, +/-1 exponent clamp vs prev_code
// (skipped when allow_clamp=false, i.e. init). BIT-IDENTICAL integer arithmetic to the kernel.
static int32_t ref_s1_encode_fine(long wq, int32_t prev_code, bool allow_clamp) {
    long aw = (wq < 0) ? -wq : wq;
    long half = 1L << (S1_Q_CODE - 1);
    long mag = (aw + half) >> S1_Q_CODE;
    long wr = (wq < 0) ? -(long)sat16(mag) : (long)sat16(mag);
    int sgn, e, nz; ref_dyad((int16_t)wr, sgn, e, nz);
    int ee = e;
#if S1_CODE_EXP_CLAMP
    if (allow_clamp) {
        unsigned pc = (unsigned)(prev_code & 0xFF); int pnz = (pc >> 5) & 1; int pe = (int)(pc & 0x1F);
        if (nz) {
            if (pnz) { if (ee > pe + 1) ee = pe + 1; if (ee < pe - 1) ee = pe - 1; if (ee < 0) ee = 0; }
            else     { ee = 0; }
        }
    }
#endif
    if (ee > 31) ee = 31;
    int c = (ee & 0x1F);
    if (nz)  c |= (1 << 5);
    if (sgn) c |= (1 << 7);
    return (int32_t)(int8_t)(unsigned char)c;
}

// CPU dyadic-Adam over one region. Mirrors the kernel dyadic_adam_region #else path EXACTLY: reads g/m/v
// from the snapshots, writes the updated W_fp/m/v/code-or-value into the *_out arrays (parallel to the
// kernel's buffer writes). is_code regions use the FINE FIXED-POINT master (int32 at scale 2^Q_CODE) +
// exp-clamp; value regions use the int16 unit master. dst_in = the deploy code/value BEFORE this step
// (for the code exp-clamp). init_only=1: g treated as 0, m/v/W_fp unchanged, only the re-quantize runs.
struct OptRegRef { int g_off, wfp_off, m_off, v_off, dst_off, n, is_code; };
static void ref_dyadic_adam_region(const long *g_in, const long *m_in, const long *v_in,
                                   const long *wfp_in, const long *dst_in, int n, int is_code, bool init_only,
                                   long *m_out, long *v_out, long *wfp_out, long *dst_out, long *g_out) {
    for (int p = 0; p < n; p++) {
        long g = init_only ? 0 : g_in[p];
        long m = m_in[p];
        long v = v_in[p];
        long absg = g < 0 ? -g : g;
        long mn = m - (m >> S1_ADAM_B1) + (g    >> S1_ADAM_B1);
        long vn = v - (v >> S1_ADAM_B2) + (absg >> S1_ADAM_B2);
        int se = ref_dyad_exp(vn);
        long step = mn >> se;
        if (is_code) {
            long wq = (int32_t)wfp_in[p];                       // full int32 fixed-point master
            long dw = step >> S1_LR_SHIFT_CODE;
#if S1_CODE_WD_SHIFT
            long wd = (long)wq >> S1_CODE_WD_SHIFT;             // multiplier-free weight decay
#else
            long wd = 0;
#endif
            long wn = init_only ? wq : sat32g_fwd((long)wq - dw - wd);
            if (!init_only) { m_out[p] = mn; v_out[p] = vn; wfp_out[p] = (int32_t)wn; g_out[p] = 0; }
            else            { m_out[p] = m;  v_out[p] = v;  wfp_out[p] = wq;           g_out[p] = g_in[p]; }
            dst_out[p] = ref_s1_encode_fine(wn, (int32_t)dst_in[p], !init_only);
        } else {
            int16_t wfp = (int16_t)(wfp_in[p] & 0xFFFF);
            long dw   = step >> S1_LR_SHIFT;
            int16_t wn = init_only ? wfp : (int16_t)sat16((long)wfp - dw);
            if (!init_only) { m_out[p] = mn; v_out[p] = vn; wfp_out[p] = (int16_t)wn; g_out[p] = 0; }
            else            { m_out[p] = m;  v_out[p] = v;  wfp_out[p] = wfp;          g_out[p] = g_in[p]; }
            dst_out[p] = (int32_t)(int16_t)wn;
        }
    }
}

// the kernel's optimizer region table, mirrored for the golden (same offsets/order/sizes/is_code).
static const OptRegRef OPT_REGS_REF[S1_NUM_OPT_REGIONS] = {
    { S1_OFF_G_EMBED,   S1_OFF_WFP_EMBED,   S1_OFF_M_EMBED,   S1_OFF_V_EMBED,   S1_OFF_EMBED,   S1_V * S1_D,                 0 },
    { S1_OFF_G_WQ,      S1_OFF_WFP_WQ,      S1_OFF_M_WQ,      S1_OFF_V_WQ,      S1_OFF_WQ,      S1_N_LAYERS * S1_D * S1_D,   1 },
    { S1_OFF_G_WK,      S1_OFF_WFP_WK,      S1_OFF_M_WK,      S1_OFF_V_WK,      S1_OFF_WK,      S1_N_LAYERS * S1_D * S1_D,   1 },
    { S1_OFF_G_WV,      S1_OFF_WFP_WV,      S1_OFF_M_WV,      S1_OFF_V_WV,      S1_OFF_WV,      S1_N_LAYERS * S1_D * S1_D,   1 },
    { S1_OFF_G_WO,      S1_OFF_WFP_WO,      S1_OFF_M_WO,      S1_OFF_V_WO,      S1_OFF_WO,      S1_N_LAYERS * S1_D * S1_D,   1 },
    { S1_OFF_G_W_UP,    S1_OFF_WFP_W_UP,    S1_OFF_M_W_UP,    S1_OFF_V_W_UP,    S1_OFF_W_UP,    S1_N_LAYERS * S1_FFN * S1_D, 1 },
    { S1_OFF_G_W_DOWN,  S1_OFF_WFP_W_DOWN,  S1_OFF_M_W_DOWN,  S1_OFF_V_W_DOWN,  S1_OFF_W_DOWN,  S1_N_LAYERS * S1_D * S1_FFN, 1 },
    { S1_OFF_G_LN1_G,   S1_OFF_WFP_LN1_G,   S1_OFF_M_LN1_G,   S1_OFF_V_LN1_G,   S1_OFF_LN1_G,   S1_N_LAYERS * S1_D,          1 },
    { S1_OFF_G_LN1_B,   S1_OFF_WFP_LN1_B,   S1_OFF_M_LN1_B,   S1_OFF_V_LN1_B,   S1_OFF_LN1_B,   S1_N_LAYERS * S1_D,          0 },
    { S1_OFF_G_LN2_G,   S1_OFF_WFP_LN2_G,   S1_OFF_M_LN2_G,   S1_OFF_V_LN2_G,   S1_OFF_LN2_G,   S1_N_LAYERS * S1_D,          1 },
    { S1_OFF_G_LN2_B,   S1_OFF_WFP_LN2_B,   S1_OFF_M_LN2_B,   S1_OFF_V_LN2_B,   S1_OFF_LN2_B,   S1_N_LAYERS * S1_D,          0 },
    { S1_OFF_G_LNF_G,   S1_OFF_WFP_LNF_G,   S1_OFF_M_LNF_G,   S1_OFF_V_LNF_G,   S1_OFF_LNF_G,   S1_D,                        1 },
    { S1_OFF_G_LNF_B,   S1_OFF_WFP_LNF_B,   S1_OFF_M_LNF_B,   S1_OFF_V_LNF_B,   S1_OFF_LNF_B,   S1_D,                        0 },
    { S1_OFF_G_W_HEAD,  S1_OFF_WFP_W_HEAD,  S1_OFF_M_W_HEAD,  S1_OFF_V_W_HEAD,  S1_OFF_W_HEAD,  S1_V * S1_D,                 0 },
};
static const char *OPT_REG_NAMES[S1_NUM_OPT_REGIONS] = {
    "EMBED","Wq","Wk","Wv","Wo","W_up","W_down","LN1_g","LN1_b","LN2_g","LN2_b","LN_F_g","LN_F_b","W_HEAD" };

// Stage a CONSISTENT optimizer buffer: small int16 W_fp masters, zeroed m/v/grads, tokens/targets/LUTs.
// The forward-read code/value regions are LEFT to mode-2 init (a1=S1_OPT_INIT) so the codes are produced
// by the SAME dyad rule the kernel uses — proving the init path AND giving mode 0 valid codes.
//
// TWO STAGING REGIMES:
//   memorize=false (the MODE-2 ONE-STEP BIT-EXACT check): RANDOM dense dyadic code weights (sparse-ish,
//     |W| in {1,2,4}) + random embed/head in [-100,100]. This produces a NON-TRIVIAL gradient through
//     every region so the one-step dyadic-Adam update is exercised on real (large) grads — what the
//     bit-exact-vs-golden test needs.
//   memorize=true (the MODE-3 CONVERGENCE check): the transformer block is RESIDUAL-PASSTHROUGH-INIT —
//     the residual-branch projections (Wo, W_down) AND Wq/Wk/Wv/W_up are ZERO, LN gamma = +1, beta = 0 —
//     so the block starts as the IDENTITY (R_out == R_in == embed) and the residual stream stays in the
//     deployed int16 range instead of being inflated by a random dense GEMM into the max-plus head's
//     SATURATION regime (which collapses every logit to the int16 ceiling -> uniform -> no learning; that
//     was the all-trainable / random-dense divergence). This is the standard transformer "zero-init the
//     residual branch so the net starts as identity" trick. With FREEZE_CODES the block stays the
//     identity and the SCATTER weights (embed/head, small init |.|<20 for headroom) genuinely fit the
//     memorization target — true CE drops ~2.2 nats to a near-perfect fit (see the convergence check).
static void stage_opt_buffer(int32_t *buf, unsigned seed, bool memorize) {
    for (int i = 0; i < S1_BUFLEN; i++) buf[i] = 0;
    srand(seed);
    // --- dense PRE-DYADIC code-weight masters (Wq/Wk/Wv/Wo/W_up/W_down). FINE FIXED-POINT: the host stages
    //     the master as (weight << S1_Q_CODE); the optimizer re-quantizes the deploy code from it via dyad().
    const long QC = (1L << S1_Q_CODE);
    auto fill_wfp_code = [&](int base, int O, int I) {
        for (int l = 0; l < S1_N_LAYERS; l++)
            for (int k = 0; k < O * I; k++) {
                if (memorize) { buf[base + l * (O * I) + k] = 0; continue; }  // ZERO dense block (identity)
                int nz = (rand() % 100) < 35;
                int mag = nz ? (1 << (rand() % 3)) : 0;     // {1,2,4} -> dyad exp 0..2
                int sgn = (rand() & 1) ? -1 : 1;
                buf[base + l * (O * I) + k] = (int32_t)(nz ? (long)sgn * mag * QC : 0);  // << Q_CODE
            }
    };
    fill_wfp_code(S1_OFF_WFP_WQ,     S1_D,   S1_D);
    fill_wfp_code(S1_OFF_WFP_WK,     S1_D,   S1_D);
    fill_wfp_code(S1_OFF_WFP_WV,     S1_D,   S1_D);
    fill_wfp_code(S1_OFF_WFP_WO,     S1_D,   S1_D);
    fill_wfp_code(S1_OFF_WFP_W_UP,   S1_FFN, S1_D);
    fill_wfp_code(S1_OFF_WFP_W_DOWN, S1_D,   S1_FFN);
    // LN gamma (code, fine fixed-point << Q_CODE) / beta (value, int16). memorize: gamma=+1, beta=0.
    for (int l = 0; l < S1_N_LAYERS; l++)
        for (int d = 0; d < S1_D; d++) {
            if (memorize) {
                buf[S1_OFF_WFP_LN1_G + l * S1_D + d] = (int32_t)QC;  buf[S1_OFF_WFP_LN2_G + l * S1_D + d] = (int32_t)QC;
                buf[S1_OFF_WFP_LN1_B + l * S1_D + d] = 0;  buf[S1_OFF_WFP_LN2_B + l * S1_D + d] = 0;
            } else {
                int nz1 = (rand() % 100) < 70, nz2 = (rand() % 100) < 70;
                buf[S1_OFF_WFP_LN1_G + l * S1_D + d] = (int32_t)(nz1 ? (long)((rand() & 1) ? -1 : 1) * (1 << (rand() % 2)) * QC : 0);
                buf[S1_OFF_WFP_LN2_G + l * S1_D + d] = (int32_t)(nz2 ? (long)((rand() & 1) ? -1 : 1) * (1 << (rand() % 2)) * QC : 0);
                buf[S1_OFF_WFP_LN1_B + l * S1_D + d] = (rand() % 41) - 20;   // beta int16
                buf[S1_OFF_WFP_LN2_B + l * S1_D + d] = (rand() % 41) - 20;
            }
        }
    // final-LN (LN_F) masters: memorize -> gamma=+1 (normalize), beta=0; random -> small dyadic gamma + beta.
    for (int d = 0; d < S1_D; d++) {
        if (memorize) {
            buf[S1_OFF_WFP_LNF_G + d] = (int32_t)QC; buf[S1_OFF_WFP_LNF_B + d] = 0;
        } else {
            int nz = (rand() % 100) < 70;
            buf[S1_OFF_WFP_LNF_G + d] = (int32_t)(nz ? (long)((rand() & 1) ? -1 : 1) * (1 << (rand() % 2)) * QC : 0);
            buf[S1_OFF_WFP_LNF_B + d] = (rand() % 41) - 20;
        }
    }
    // EMBED / W_HEAD masters (int16 values). memorize: small |.|<20 (keeps R in range so the head has room
    // to sharpen before saturating); else: deployed range |.|<100 (the mode-0/1 headroom discipline).
    int emag = memorize ? 41 : 201, eoff = memorize ? 20 : 100;
    for (int i = 0; i < S1_V * S1_D; i++) {
        buf[S1_OFF_WFP_EMBED  + i] = (rand() % emag) - eoff;
        buf[S1_OFF_WFP_W_HEAD + i] = (rand() % emag) - eoff;
    }
    // m, v, grads: ZERO (fresh optimizer state). (buf already zeroed above.)
    // host-staged exp2 LUTs (CE + attn share the Q16 exp2 ROM).
    for (int i = 0; i < S1_LUT_N; i++) { buf[S1_OFF_CE_LUT + i] = FT_CE_EXP2_Q16[i];
                                         buf[S1_OFF_ATTN_LUT + i] = FT_CE_EXP2_Q16[i]; }
    // tokens / targets.
    if (memorize) {
        // a FIXED short pattern repeated to fill L_SEQ; target = next token in the pattern (memorization).
        const int PAT = 8;                                   // 8-symbol pattern over a small vocab slice
        for (int t = 0; t < S1_L_SEQ; t++) {
            int sym = t % PAT;
            buf[S1_OFF_TOKENS  + t] = sym;
            buf[S1_OFF_TARGETS + t] = (sym + 1) % PAT;        // predict the successor (learnable by head/embed)
        }
    } else {
        for (int t = 0; t < S1_L_SEQ; t++) { buf[S1_OFF_TOKENS + t]  = rand() % S1_V;
                                             buf[S1_OFF_TARGETS + t] = rand() % S1_V; }
    }
}

// Stage a buffer for the TRAIN_ALL convergence probe: the memorization task (8-symbol successor) with a
// SMALL-NONZERO sparse dyadic CODE init (density% nonzero, |w|=1 at scale 2^Q_CODE), so the dense backbone
// receives a NON-ZERO gradient (the zero/identity init of stage_opt_buffer(memorize=true) gives the dense
// code weights EXACTLY-ZERO gradient — W_up=0 -> Hd=relu(0)=0 -> g_Wdn uses Hd=0; W_down=0 -> gHd=0 ->
// g_Wup uses gU=0; Wo=0 -> gAcat=0 -> attn-bwd gOut=0 -> gV=0 and gQ/gK=sign(Q-K=0)=0 — so under the
// identity init TRAIN_ALL CANNOT move the dense codes at all; it is silently == FREEZE_CODES). A small
// nonzero init breaks that degeneracy so the dense-backbone training can be HONESTLY exercised.
// embed/head small (|.|<embmag) keeps R in the deployed range at init.
static void stage_train_all(int32_t *buf, unsigned seed, int densepct, int embmag) {
    for (int i = 0; i < S1_BUFLEN; i++) buf[i] = 0;
    srand(seed);
    const long QC = (1L << S1_Q_CODE);
    auto fc = [&](int base, int O, int I) {
        for (int l = 0; l < S1_N_LAYERS; l++)
            for (int k = 0; k < O * I; k++) {
                int nz = (rand() % 100) < densepct; int sg = (rand() & 1) ? -1 : 1;
                buf[base + l * (O * I) + k] = (int32_t)(nz ? (long)sg * QC : 0);   // |w|=1 << Q_CODE
            }
    };
    fc(S1_OFF_WFP_WQ, S1_D, S1_D); fc(S1_OFF_WFP_WK, S1_D, S1_D); fc(S1_OFF_WFP_WV, S1_D, S1_D);
    fc(S1_OFF_WFP_WO, S1_D, S1_D); fc(S1_OFF_WFP_W_UP, S1_FFN, S1_D); fc(S1_OFF_WFP_W_DOWN, S1_D, S1_FFN);
    for (int l = 0; l < S1_N_LAYERS; l++)
        for (int d = 0; d < S1_D; d++) {
            buf[S1_OFF_WFP_LN1_G + l * S1_D + d] = (int32_t)QC; buf[S1_OFF_WFP_LN2_G + l * S1_D + d] = (int32_t)QC;
            buf[S1_OFF_WFP_LN1_B + l * S1_D + d] = 0;            buf[S1_OFF_WFP_LN2_B + l * S1_D + d] = 0;
        }
    for (int d = 0; d < S1_D; d++) { buf[S1_OFF_WFP_LNF_G + d] = (int32_t)QC; buf[S1_OFF_WFP_LNF_B + d] = 0; }
    for (int i = 0; i < S1_V * S1_D; i++) {
        buf[S1_OFF_WFP_EMBED  + i] = (rand() % (2 * embmag + 1)) - embmag;
        buf[S1_OFF_WFP_W_HEAD + i] = (rand() % (2 * embmag + 1)) - embmag;
    }
    for (int i = 0; i < S1_LUT_N; i++) { buf[S1_OFF_CE_LUT + i] = FT_CE_EXP2_Q16[i];
                                         buf[S1_OFF_ATTN_LUT + i] = FT_CE_EXP2_Q16[i]; }
    const int PAT = 8;
    for (int t = 0; t < S1_L_SEQ; t++) { int sym = t % PAT; buf[S1_OFF_TOKENS + t] = sym; buf[S1_OFF_TARGETS + t] = (sym + 1) % PAT; }
}

// count how many deploy code BYTES (is_code regions) differ from a snapshot — the honest discriminator of
// "did the dense backbone actually TRAIN" vs "codes never moved (silently == FREEZE)".
static int count_code_changes(const int32_t *bb, const int8_t *snap_codes) {
    int dsts[] = { S1_OFF_WQ, S1_OFF_WK, S1_OFF_WV, S1_OFF_WO, S1_OFF_W_UP, S1_OFF_W_DOWN, S1_OFF_LN1_G, S1_OFF_LN2_G };
    int ns[]   = { S1_D*S1_D, S1_D*S1_D, S1_D*S1_D, S1_D*S1_D, S1_FFN*S1_D, S1_D*S1_FFN, S1_D, S1_D };
    int off = 0, ch = 0;
    for (int r = 0; r < 8; r++) { for (int p = 0; p < ns[r]; p++) { int8_t c = (int8_t)(bb[dsts[r] + p] & 0xFF); if (c != snap_codes[off + p]) ch++; } off += ns[r]; }
    return ch;
}
static void snapshot_codes(const int32_t *bb, int8_t *out) {
    int dsts[] = { S1_OFF_WQ, S1_OFF_WK, S1_OFF_WV, S1_OFF_WO, S1_OFF_W_UP, S1_OFF_W_DOWN, S1_OFF_LN1_G, S1_OFF_LN2_G };
    int ns[]   = { S1_D*S1_D, S1_D*S1_D, S1_D*S1_D, S1_D*S1_D, S1_FFN*S1_D, S1_D*S1_FFN, S1_D, S1_D };
    int off = 0;
    for (int r = 0; r < 8; r++) { for (int p = 0; p < ns[r]; p++) out[off + p] = (int8_t)(bb[dsts[r] + p] & 0xFF); off += ns[r]; }
}

// snapshot n int32 words from buf[off..] into a long[] (sign-extended).
static void snap(const int32_t *buf, int off, long *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = (int32_t)buf[off + i];
}
// max|buf[off+i] - ref[i]| over n words.
static long cmp_buf(const int32_t *buf, int off, const long *ref, int n) {
    long mx = 0;
    for (int i = 0; i < n; i++) { long d = labs((long)(int32_t)buf[off + i] - ref[i]); if (d > mx) mx = d; }
    return mx;
}

// ============================================================================

int main() {
    int32_t *buf = (int32_t *)aligned_alloc(64, sizeof(int32_t) * S1_BUFLEN);
    for (int i = 0; i < S1_BUFLEN; i++) buf[i] = 0;
    srand(1234);

    // ---- embed table: deployed-range int16 values (|.| < 100; keeps the dense-GEMM backward grads,
    //      which the Q16 CE grad amplifies by the fan-in, inside int32 across seeds — see the BWD
    //      headroom gate). The morph trainer needs no such bound (it SCATTERS, no fan-in sum); the S1
    //      dense GEMMs do, so the test stays in the verified fixed-point range like forge_pc_wgrad_tb. ----
    for (int i = 0; i < S1_V * S1_D; i++) buf[S1_OFF_EMBED + i] = (rand() % 201) - 100;

    // ---- projection weights: dyadic byte-codes at a realistic DEPLOYED-SCALING init. The dense GEMM
    //      backward amplifies the Q16 CE grad by ~(density * gain * fan-in), so a hot init (50% @ 2^2
    //      over fan-in 512) blows the weight grad past int32. A realistic dyadic init is SPARSE (~25%
    //      nonzero) at unit-ish gain ({1,2} = exp 0..1) — the regime where the int16/int32 fixed-point
    //      kernels are valid (mirrors forge_pc_wgrad_tb's |.|<600 deployed-range discipline). ----
    auto fill_w = [&](int base, int O, int I) {
        for (int l = 0; l < S1_N_LAYERS; l++)
            for (int k = 0; k < O * I; k++) {
                int nz = (rand() % 100) < 35 ? 1 : 0;       // ~35% nonzero (sparse-ish dyadic init)
                int sgn = rand() & 1;
                int exp = rand() % 2;                       // |W| in {1,2}  (exp 0..1; unit-ish gain)
                buf[base + l * (O * I) + k] = make_code(sgn, nz, exp);
            }
    };
    fill_w(S1_OFF_WQ,     S1_D,   S1_D);
    fill_w(S1_OFF_WK,     S1_D,   S1_D);
    fill_w(S1_OFF_WV,     S1_D,   S1_D);
    fill_w(S1_OFF_WO,     S1_D,   S1_D);
    fill_w(S1_OFF_W_UP,   S1_FFN, S1_D);
    fill_w(S1_OFF_W_DOWN, S1_D,   S1_FFN);

    // ---- LN gamma (dyadic codes, ~70% nonzero, |gamma| up to 2^1) + beta (int16) ----
    auto fill_ln = [&](int gbase, int bbase) {
        for (int l = 0; l < S1_N_LAYERS; l++)
            for (int d = 0; d < S1_D; d++) {
                int nz = (rand() % 100) < 70 ? 1 : 0;
                buf[gbase + l * S1_D + d] = make_code(rand() & 1, nz, rand() % 2);
                buf[bbase + l * S1_D + d] = (rand() % 41) - 20;
            }
    };
    fill_ln(S1_OFF_LN1_G, S1_OFF_LN1_B);
    fill_ln(S1_OFF_LN2_G, S1_OFF_LN2_B);
    // ---- final-LN (LN_F) gamma (dyadic code) + beta (int16) — single LN, one entry per channel ----
    for (int d = 0; d < S1_D; d++) {
        int nz = (rand() % 100) < 70 ? 1 : 0;
        buf[S1_OFF_LNF_G + d] = make_code(rand() & 1, nz, rand() % 2);
        buf[S1_OFF_LNF_B + d] = (rand() % 41) - 20;
    }

    // ---- tied LM-head weights (plain int16, |.| < 100; same deployed-range bound as embed) ----
    for (int i = 0; i < S1_V * S1_D; i++) buf[S1_OFF_W_HEAD + i] = (rand() % 201) - 100;

    // ---- host-staged exp2 LUTs (CE + attn share the same Q16 exp2 ROM) ----
    for (int i = 0; i < S1_LUT_N; i++) { buf[S1_OFF_CE_LUT + i] = FT_CE_EXP2_Q16[i];
                                         buf[S1_OFF_ATTN_LUT + i] = FT_CE_EXP2_Q16[i]; }

    // ---- tokens / targets in [0,V) ----
    for (int t = 0; t < S1_L_SEQ; t++) { buf[S1_OFF_TOKENS + t]  = rand() % S1_V;
                                         buf[S1_OFF_TARGETS + t] = rand() % S1_V; }

    // ============================ run the kernel ============================
    forge_train_step_s1(buf, 0, 0, 0, 0);

    // ============================ int64 golden ============================
    static long R[S1_L_SEQ][S1_D];
    static long X1[S1_L_SEQ][S1_D];
    static long X2[S1_L_SEQ][S1_D];
    static long Qm[S1_L_SEQ][S1_D], Km[S1_L_SEQ][S1_D], Vm[S1_L_SEQ][S1_D];
    static long Qh[S1_L_SEQ][S1_DH], Kh[S1_L_SEQ][S1_DH], Vh[S1_L_SEQ][S1_DH], Ah[S1_L_SEQ][S1_DH];
    static long Acat[S1_L_SEQ][S1_D], AO[S1_L_SEQ][S1_D];
    static long U[S1_L_SEQ][S1_FFN], Hd[S1_L_SEQ][S1_FFN], Dn[S1_L_SEQ][S1_D];
    static long P_g[S1_H][S1_L_SEQ][S1_L_SEQ]; static int J_g[S1_H][S1_L_SEQ];
    static int RM_g[S1_L_SEQ][S1_FFN];
    static int LN1_e[S1_L_SEQ], LN1_q[S1_L_SEQ], LN2_e[S1_L_SEQ], LN2_q[S1_L_SEQ];
    static int LNF_e[S1_L_SEQ], LNF_q[S1_L_SEQ];
    static long Rn[S1_L_SEQ][S1_D];   // LN_F(R_fin) — the head input (== R_fin when S1_USE_LNF==0)

    // embed gather
    for (int t = 0; t < S1_L_SEQ; t++) {
        int tok = buf[S1_OFF_TOKENS + t]; if (tok < 0) tok = 0; if (tok >= S1_V) tok = S1_V - 1;
        for (int d = 0; d < S1_D; d++) R[t][d] = (int16_t)(buf[S1_OFF_EMBED + tok * S1_D + d] & 0xFFFF);
    }

    long peak = 0;   // track peak |accum| for the int32 headroom check
    for (int l = 0; l < S1_N_LAYERS; l++) {
        // LN1
        for (int t = 0; t < S1_L_SEQ; t++) {
            long row[S1_D], yr[S1_D];
            for (int d = 0; d < S1_D; d++) row[d] = R[t][d];
            ref_ln(row, buf, S1_OFF_LN1_G + l * S1_D, S1_OFF_LN1_B + l * S1_D, yr, LN1_e[t], LN1_q[t]);
            for (int d = 0; d < S1_D; d++) X1[t][d] = yr[d];
        }
        // QKV
        ref_gemm<S1_L_SEQ, S1_D, S1_D>(X1, buf, S1_OFF_WQ + l * S1_D * S1_D, Qm);
        ref_gemm<S1_L_SEQ, S1_D, S1_D>(X1, buf, S1_OFF_WK + l * S1_D * S1_D, Km);
        ref_gemm<S1_L_SEQ, S1_D, S1_D>(X1, buf, S1_OFF_WV + l * S1_D * S1_D, Vm);
        // heads
        for (int h = 0; h < S1_H; h++) {
            for (int t = 0; t < S1_L_SEQ; t++)
                for (int d = 0; d < S1_DH; d++) {
                    Qh[t][d] = Qm[t][h * S1_DH + d];
                    Kh[t][d] = Km[t][h * S1_DH + d];
                    Vh[t][d] = Vm[t][h * S1_DH + d];
                }
            ref_attn(Qh, Kh, Vh, FT_CE_EXP2_Q16, Ah, P_g[h], J_g[h]);
            for (int t = 0; t < S1_L_SEQ; t++)
                for (int d = 0; d < S1_DH; d++) Acat[t][h * S1_DH + d] = Ah[t][d];
        }
        // O proj + res
        ref_gemm<S1_L_SEQ, S1_D, S1_D>(Acat, buf, S1_OFF_WO + l * S1_D * S1_D, AO);
        for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) R[t][d] = sat16(R[t][d] + AO[t][d]);
        // LN2
        for (int t = 0; t < S1_L_SEQ; t++) {
            long row[S1_D], yr[S1_D];
            for (int d = 0; d < S1_D; d++) row[d] = R[t][d];
            ref_ln(row, buf, S1_OFF_LN2_G + l * S1_D, S1_OFF_LN2_B + l * S1_D, yr, LN2_e[t], LN2_q[t]);
            for (int d = 0; d < S1_D; d++) X2[t][d] = yr[d];
        }
        // FFN up, relu, down, res
        ref_gemm<S1_L_SEQ, S1_D, S1_FFN>(X2, buf, S1_OFF_W_UP + l * S1_FFN * S1_D, U);
        for (int t = 0; t < S1_L_SEQ; t++)
            for (int k = 0; k < S1_FFN; k++) {
                int act = (U[t][k] > 0) ? 1 : 0; RM_g[t][k] = act; Hd[t][k] = act ? U[t][k] : 0;
            }
        ref_gemm<S1_L_SEQ, S1_FFN, S1_D>(Hd, buf, S1_OFF_W_DOWN + l * S1_D * S1_FFN, Dn);
        for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) { R[t][d] = sat16(R[t][d] + Dn[t][d]);
                                             if (labs(R[t][d]) > peak) peak = labs(R[t][d]); }
    }

    // ---- final LayerNorm (LN_F) -> Rn (head input). When S1_USE_LNF==0, Rn==R (no LN_F). ----
#if S1_USE_LNF
    for (int t = 0; t < S1_L_SEQ; t++) {
        long row[S1_D], yr[S1_D];
        for (int d = 0; d < S1_D; d++) row[d] = R[t][d];
        ref_ln(row, buf, S1_OFF_LNF_G, S1_OFF_LNF_B, yr, LNF_e[t], LNF_q[t]);
        for (int d = 0; d < S1_D; d++) Rn[t][d] = yr[d];
    }
#else
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) Rn[t][d] = R[t][d];
#endif

    // ---- compare saved activations ----
    long max_R = 0, max_X1 = 0, max_X2 = 0, max_P = 0, max_J = 0, max_RM = 0, max_LNST = 0;
    int bad = 0;
    // LN fine-dyadic stats (e,nq) per token, both layers' worth at l=0 — the mode-1 backward's OFF_RSC
    // contract. (Transitively covered by X1/X2 bit-exactness, but checked explicitly per the plan's
    // "save R, x1, x2, ..., LN stats" activation-save list.)
    for (int t = 0; t < S1_L_SEQ; t++) {
        long k1e = (int32_t)buf[s1_off_lnstats(S1_OFF_LN1_STATS, 0) + 2 * t];
        long k1q = (int32_t)buf[s1_off_lnstats(S1_OFF_LN1_STATS, 0) + 2 * t + 1];
        long k2e = (int32_t)buf[s1_off_lnstats(S1_OFF_LN2_STATS, 0) + 2 * t];
        long k2q = (int32_t)buf[s1_off_lnstats(S1_OFF_LN2_STATS, 0) + 2 * t + 1];
        long e1 = labs(k1e - LN1_e[t]), q1 = labs(k1q - LN1_q[t]);
        long e2 = labs(k2e - LN2_e[t]), q2 = labs(k2q - LN2_q[t]);
        if (e1 > max_LNST) max_LNST = e1; if (q1 > max_LNST) max_LNST = q1;
        if (e2 > max_LNST) max_LNST = e2; if (q2 > max_LNST) max_LNST = q2;
#if S1_USE_LNF
        long kfe = (int32_t)buf[S1_OFF_LNF_STATS + 2 * t];
        long kfq = (int32_t)buf[S1_OFF_LNF_STATS + 2 * t + 1];
        long ef = labs(kfe - LNF_e[t]), qf = labs(kfq - LNF_q[t]);
        if (ef > max_LNST) max_LNST = ef; if (qf > max_LNST) max_LNST = qf;
#endif
    }
    for (int t = 0; t < S1_L_SEQ; t++)
        for (int d = 0; d < S1_D; d++) {
            long g = (int32_t)buf[s1_off_R(S1_N_LAYERS) + t * S1_D + d];
            long dd = labs(g - R[t][d]); if (dd > max_R) max_R = dd;
            long g1 = (int32_t)buf[s1_off_act(S1_OFF_X1, 0) + t * S1_D + d];
            long d1 = labs(g1 - X1[t][d]); if (d1 > max_X1) max_X1 = d1;
            long g2 = (int32_t)buf[s1_off_act(S1_OFF_X2, 0) + t * S1_D + d];
            long d2 = labs(g2 - X2[t][d]); if (d2 > max_X2) max_X2 = d2;
            if (dd != 0 && bad < 20) { printf("  R[%d][%d] ref=%ld got=%ld\n", t, d, R[t][d], g); bad++; }
        }
    for (int h = 0; h < S1_H; h++) {
        for (int i = 0; i < S1_L_SEQ; i++) {
            long gj = (int32_t)buf[s1_off_attn_j(0, h) + i];
            long dj = labs(gj - J_g[h][i]); if (dj > max_J) max_J = dj;
            for (int j = 0; j < S1_L_SEQ; j++) {
                long gp = (int32_t)buf[s1_off_attn_p(0, h) + i * S1_L_SEQ + j];
                long dp = labs(gp - P_g[h][i][j]); if (dp > max_P) max_P = dp;
            }
        }
    }
    for (int t = 0; t < S1_L_SEQ; t++)
        for (int k = 0; k < S1_FFN; k++) {
            long gm = (int32_t)buf[s1_off_relu(0) + t * S1_FFN + k];
            long dm = labs(gm - RM_g[t][k]); if (dm > max_RM) max_RM = dm;
        }

    // ---- LM head + CE golden, compared to kernel logits/grad_logit ----
    long max_logit = 0, max_grad = 0;
    long peak_logit = 0, peak_grad = 0;
    for (int t = 0; t < S1_L_SEQ; t++) {
        int target = buf[S1_OFF_TARGETS + t]; if (target < 0) target = 0; if (target >= S1_V) target = S1_V - 1;
        long logits[S1_V];
        for (int v = 0; v < S1_V; v++) {
            long mx = -2147483647L;
            for (int d = 0; d < S1_D; d++) {
                long Wvd = (int16_t)(buf[S1_OFF_W_HEAD + v * S1_D + d] & 0xFFFF);
                long cand = Wvd + Rn[t][d];
                if (cand > mx) mx = cand;
            }
            logits[v] = sat16(mx);
            if (labs(logits[v]) > peak_logit) peak_logit = labs(logits[v]);
        }
        int pred = 0; { long mx = -2147483647L; for (int v = 0; v < S1_V; v++) if (logits[v] > mx) { mx = logits[v]; pred = v; } }
        long e_row[S1_V]; long Ssum = 0;
        for (int v = 0; v < S1_V; v++) {
            long zneg = logits[pred] - logits[v];
            int idx = (int)(zneg >> S1_CE_ZSHIFT); if (idx > 255) idx = 255; if (idx < 0) idx = 0;
            e_row[v] = FT_CE_EXP2_Q16[idx]; Ssum += e_row[v];
        }
        int sh = 0; for (int b = 0; b < 8; b++) if ((Ssum >> sh) > 65536) sh++;
        long psum = 0; for (int v = 0; v < S1_V; v++) { e_row[v] = e_row[v] >> sh; psum += e_row[v]; }
        for (int v = 0; v < S1_V; v++) {
            long g = e_row[v] - (v == target ? psum : 0);
            if (labs(g) > peak_grad) peak_grad = labs(g);
            long kg = (int32_t)buf[S1_OFF_GRAD_LOGIT + t * S1_V + v];
            long kl = (int32_t)buf[S1_OFF_LOGITS     + t * S1_V + v];
            long dg = labs(kg - g);       if (dg > max_grad)  max_grad  = dg;
            long dl = labs(kl - logits[v]); if (dl > max_logit) max_logit = dl;
            if (dg != 0 && bad < 40) { printf("  grad_logit[%d][%d] ref=%ld got=%ld\n", t, v, g, kg); bad++; }
        }
    }

    // ============================================================================
    // ======================  MODE 1 (BACKWARD) bit-exact check  =================
    // ============================================================================
    // Run the kernel backward, then an int64 CPU reverse chain (bit-identical math) and assert
    // max|kernel_grad - ref_grad| == 0 over EVERY gradient region + the intermediate gR/gX1/gX2.
    // (S1_N_LAYERS==1; the reverse chain reconstructs the per-layer forward intermediates it needs.)

    forge_train_step_s1(buf, 1, 0, 0, 0);

    // --- reconstruct the forward intermediates the backward consumes (single block, l=0) ---
    static long Rin1[S1_L_SEQ][S1_D];     // block input (embed) = pre-LN1 residual
    for (int t = 0; t < S1_L_SEQ; t++) {
        int tok = buf[S1_OFF_TOKENS + t]; if (tok < 0) tok = 0; if (tok >= S1_V) tok = S1_V - 1;
        for (int d = 0; d < S1_D; d++) Rin1[t][d] = (int16_t)(buf[S1_OFF_EMBED + tok * S1_D + d] & 0xFFFF);
    }
    static long bX1[S1_L_SEQ][S1_D]; static int bLN1e[S1_L_SEQ], bLN1q[S1_L_SEQ];
    for (int t = 0; t < S1_L_SEQ; t++) {
        long row[S1_D], yr[S1_D]; for (int d = 0; d < S1_D; d++) row[d] = Rin1[t][d];
        ref_ln(row, buf, S1_OFF_LN1_G, S1_OFF_LN1_B, yr, bLN1e[t], bLN1q[t]);
        for (int d = 0; d < S1_D; d++) bX1[t][d] = yr[d];
    }
    static long bQ[S1_L_SEQ][S1_D], bK[S1_L_SEQ][S1_D], bV[S1_L_SEQ][S1_D];
    ref_gemm<S1_L_SEQ, S1_D, S1_D>(bX1, buf, S1_OFF_WQ, bQ);
    ref_gemm<S1_L_SEQ, S1_D, S1_D>(bX1, buf, S1_OFF_WK, bK);
    ref_gemm<S1_L_SEQ, S1_D, S1_D>(bX1, buf, S1_OFF_WV, bV);
    static long bAcat[S1_L_SEQ][S1_D];
    static long Qh2[S1_L_SEQ][S1_DH], Kh2[S1_L_SEQ][S1_DH], Vh2[S1_L_SEQ][S1_DH], Ah2[S1_L_SEQ][S1_DH];
    static long Pdum[S1_L_SEQ][S1_L_SEQ]; static int Jdum[S1_L_SEQ];
    for (int h = 0; h < S1_H; h++) {
        for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_DH; d++) { Qh2[t][d]=bQ[t][h*S1_DH+d]; Kh2[t][d]=bK[t][h*S1_DH+d]; Vh2[t][d]=bV[t][h*S1_DH+d]; }
        ref_attn(Qh2, Kh2, Vh2, FT_CE_EXP2_Q16, Ah2, Pdum, Jdum);
        for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_DH; d++) bAcat[t][h*S1_DH+d] = Ah2[t][d];
    }
    static long bAO[S1_L_SEQ][S1_D]; ref_gemm<S1_L_SEQ, S1_D, S1_D>(bAcat, buf, S1_OFF_WO, bAO);
    static long Rattn[S1_L_SEQ][S1_D];
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) Rattn[t][d] = sat16(Rin1[t][d] + bAO[t][d]);
    static long bX2[S1_L_SEQ][S1_D]; static int bLN2e[S1_L_SEQ], bLN2q[S1_L_SEQ];
    for (int t = 0; t < S1_L_SEQ; t++) {
        long row[S1_D], yr[S1_D]; for (int d = 0; d < S1_D; d++) row[d] = Rattn[t][d];
        ref_ln(row, buf, S1_OFF_LN2_G, S1_OFF_LN2_B, yr, bLN2e[t], bLN2q[t]);
        for (int d = 0; d < S1_D; d++) bX2[t][d] = yr[d];
    }
    static long bU[S1_L_SEQ][S1_FFN], bHd[S1_L_SEQ][S1_FFN]; static int bRM[S1_L_SEQ][S1_FFN];
    ref_gemm<S1_L_SEQ, S1_D, S1_FFN>(bX2, buf, S1_OFF_W_UP, bU);
    for (int t = 0; t < S1_L_SEQ; t++) for (int k = 0; k < S1_FFN; k++) { bRM[t][k] = (bU[t][k] > 0) ? 1 : 0; bHd[t][k] = bRM[t][k] ? bU[t][k] : 0; }
    static long bDn[S1_L_SEQ][S1_D]; ref_gemm<S1_L_SEQ, S1_FFN, S1_D>(bHd, buf, S1_OFF_W_DOWN, bDn);
    static long Rfin[S1_L_SEQ][S1_D];
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) Rfin[t][d] = sat16(Rattn[t][d] + bDn[t][d]);
    // LN_F(Rfin) -> Rnf (the head input) + its (e,nq) stats. Rnf==Rfin when S1_USE_LNF==0.
    static long Rnf[S1_L_SEQ][S1_D]; static int bLNFe[S1_L_SEQ], bLNFq[S1_L_SEQ];
#if S1_USE_LNF
    for (int t = 0; t < S1_L_SEQ; t++) {
        long row[S1_D], yr[S1_D]; for (int d = 0; d < S1_D; d++) row[d] = Rfin[t][d];
        ref_ln(row, buf, S1_OFF_LNF_G, S1_OFF_LNF_B, yr, bLNFe[t], bLNFq[t]);
        for (int d = 0; d < S1_D; d++) Rnf[t][d] = yr[d];
    }
#else
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) Rnf[t][d] = Rfin[t][d];
#endif

    // --- LM head argmax d* + grad_logit (golden; == forward, recomputed for the head-scatter) ---
    // The head reads Rnf (LN_F output), so the scatter accumulates into gRn_ref (head-input grad); LN_F bwd
    // then maps gRn_ref -> gR_ref (+ g_LNF gamma/beta).
    static long gR_ref[S1_L_SEQ][S1_D];
    static long gRn_ref[S1_L_SEQ][S1_D];
    static long gLNFG_ref[S1_D], gLNFB_ref[S1_D];
    static long gWHEAD_ref[S1_V * S1_D];
    for (int i = 0; i < S1_V * S1_D; i++) gWHEAD_ref[i] = 0;
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) { gR_ref[t][d] = 0; gRn_ref[t][d] = 0; }
    for (int d = 0; d < S1_D; d++) { gLNFG_ref[d] = 0; gLNFB_ref[d] = 0; }
    for (int t = 0; t < S1_L_SEQ; t++) {
        int target = buf[S1_OFF_TARGETS + t]; if (target < 0) target = 0; if (target >= S1_V) target = S1_V - 1;
        long logits[S1_V]; int adh[S1_V];
        for (int v = 0; v < S1_V; v++) {
            long mx = -2147483647L; int idd = 0;
            for (int d = 0; d < S1_D; d++) {
                long Wvd = (int16_t)(buf[S1_OFF_W_HEAD + v * S1_D + d] & 0xFFFF);
                long cand = Wvd + Rnf[t][d];
                if (cand > mx) { mx = cand; idd = d; }
            }
            logits[v] = sat16(mx); adh[v] = idd;
        }
        int pred = 0; { long mx = -2147483647L; for (int v = 0; v < S1_V; v++) if (logits[v] > mx) { mx = logits[v]; pred = v; } }
        long e_row[S1_V]; long Ssum = 0;
        for (int v = 0; v < S1_V; v++) { long zneg = logits[pred] - logits[v]; int idx = (int)(zneg >> S1_CE_ZSHIFT); if (idx > 255) idx = 255; if (idx < 0) idx = 0; e_row[v] = FT_CE_EXP2_Q16[idx]; Ssum += e_row[v]; }
        int sh2 = 0; for (int b = 0; b < 8; b++) if ((Ssum >> sh2) > 65536) sh2++;
        long psum = 0; for (int v = 0; v < S1_V; v++) { e_row[v] = e_row[v] >> sh2; psum += e_row[v]; }
        // head scatter: gRn[t][d*] += g ; gWHEAD[v][d*] += g
        for (int v = 0; v < S1_V; v++) {
            long g = e_row[v] - (v == target ? psum : 0);
            if (g == 0) continue;
            int dstar = adh[v];
            gWHEAD_ref[v * S1_D + dstar] += g;
            gRn_ref[t][dstar] = sat32g(gRn_ref[t][dstar] + g);
        }
    }
#if S1_USE_LNF
    // --- LN_F backward: gY=gRn_ref -> gR_ref + g_LNF_gamma/beta (== ref_ln_bwd against the LN_F gamma codes
    //     and Rfin pre-norm). LN_F is a single LN, so its stats are bLNFe/bLNFq. ---
    {
        static long gRf_add[S1_L_SEQ][S1_D];
        ref_ln_bwd(Rfin, buf, S1_OFF_LNF_G, gRn_ref, bLNFe, bLNFq, gRf_add, gLNFG_ref, gLNFB_ref, 0);
        for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) gR_ref[t][d] = sat32g(gR_ref[t][d] + gRf_add[t][d]);
    }
#else
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) gR_ref[t][d] = gRn_ref[t][d];
#endif

    // --- gradient accumulators (golden) ---
    static long gEMBED_ref[S1_V * S1_D];
    static long gWQ_ref[S1_D*S1_D], gWK_ref[S1_D*S1_D], gWV_ref[S1_D*S1_D], gWO_ref[S1_D*S1_D];
    static long gWUP_ref[S1_FFN*S1_D], gWDN_ref[S1_D*S1_FFN];
    static long gLN1G_ref[S1_D], gLN1B_ref[S1_D], gLN2G_ref[S1_D], gLN2B_ref[S1_D];
    for (int i = 0; i < S1_V*S1_D; i++) gEMBED_ref[i] = 0;
    for (int i = 0; i < S1_D*S1_D; i++) { gWQ_ref[i]=0; gWK_ref[i]=0; gWV_ref[i]=0; gWO_ref[i]=0; }
    for (int i = 0; i < S1_FFN*S1_D; i++) gWUP_ref[i] = 0;
    for (int i = 0; i < S1_D*S1_FFN; i++) gWDN_ref[i] = 0;
    for (int i = 0; i < S1_D; i++) { gLN1G_ref[i]=0; gLN1B_ref[i]=0; gLN2G_ref[i]=0; gLN2B_ref[i]=0; }

    static long gX1_ref[S1_L_SEQ][S1_D], gX2_ref[S1_L_SEQ][S1_D];

    // === FFN backward ===
    static long gDn_ref[S1_L_SEQ][S1_D];
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) gDn_ref[t][d] = gR_ref[t][d];
    static long gHd_ref[S1_L_SEQ][S1_FFN];
    ref_xgrad<S1_L_SEQ, S1_FFN, S1_D>(gDn_ref, buf, S1_OFF_W_DOWN, gHd_ref);
    ref_wgrad<S1_L_SEQ, S1_FFN, S1_D>(gDn_ref, bHd, gWDN_ref, 0);
    static long gU_ref[S1_L_SEQ][S1_FFN];
    for (int t = 0; t < S1_L_SEQ; t++) for (int k = 0; k < S1_FFN; k++) gU_ref[t][k] = bRM[t][k] ? gHd_ref[t][k] : 0;
    ref_xgrad<S1_L_SEQ, S1_D, S1_FFN>(gU_ref, buf, S1_OFF_W_UP, gX2_ref);
    ref_wgrad<S1_L_SEQ, S1_D, S1_FFN>(gU_ref, bX2, gWUP_ref, 0);

    // === LN2 backward (gR += gR_pre; g_LN2_gamma/beta) ===
    static long gR2add[S1_L_SEQ][S1_D];
    ref_ln_bwd(Rattn, buf, S1_OFF_LN2_G, gX2_ref, bLN2e, bLN2q, gR2add, gLN2G_ref, gLN2B_ref, 0);
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) gR_ref[t][d] = sat32g(gR_ref[t][d] + gR2add[t][d]);

    // === residual1 split + Wo backward ===
    static long gAO_ref[S1_L_SEQ][S1_D];
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) gAO_ref[t][d] = gR_ref[t][d];
    static long gAcat_ref[S1_L_SEQ][S1_D];
    ref_xgrad<S1_L_SEQ, S1_D, S1_D>(gAO_ref, buf, S1_OFF_WO, gAcat_ref);
    ref_wgrad<S1_L_SEQ, S1_D, S1_D>(gAO_ref, bAcat, gWO_ref, 0);

    // === attention backward per head -> gQ/gK/gV ===
    static long gQv_ref[S1_L_SEQ][S1_D], gKv_ref[S1_L_SEQ][S1_D], gVv_ref[S1_L_SEQ][S1_D];
    for (int h = 0; h < S1_H; h++) {
        static long Qh3[S1_L_SEQ][S1_DH], Kh3[S1_L_SEQ][S1_DH], Vh3[S1_L_SEQ][S1_DH], gO3[S1_L_SEQ][S1_DH];
        static long gQ3[S1_L_SEQ][S1_DH], gK3[S1_L_SEQ][S1_DH], gV3[S1_L_SEQ][S1_DH];
        for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_DH; d++) {
                Qh3[t][d]=bQ[t][h*S1_DH+d]; Kh3[t][d]=bK[t][h*S1_DH+d]; Vh3[t][d]=bV[t][h*S1_DH+d];
                gO3[t][d]=gAcat_ref[t][h*S1_DH+d]; gQ3[t][d]=0; gK3[t][d]=0; gV3[t][d]=0;
            }
        ref_attn_bwd(Qh3, Kh3, Vh3, gO3, FT_CE_EXP2_Q16, gQ3, gK3, gV3);
        for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_DH; d++) { gQv_ref[t][h*S1_DH+d]=gQ3[t][d]; gKv_ref[t][h*S1_DH+d]=gK3[t][d]; gVv_ref[t][h*S1_DH+d]=gV3[t][d]; }
    }

    // === Q/K/V backward -> gX1 (accumulate 3 paths) + g_Wq/g_Wk/g_Wv ===
    static long gX1q[S1_L_SEQ][S1_D], gX1k[S1_L_SEQ][S1_D], gX1v2[S1_L_SEQ][S1_D];
    ref_xgrad<S1_L_SEQ, S1_D, S1_D>(gQv_ref, buf, S1_OFF_WQ, gX1q);
    ref_xgrad<S1_L_SEQ, S1_D, S1_D>(gKv_ref, buf, S1_OFF_WK, gX1k);
    ref_xgrad<S1_L_SEQ, S1_D, S1_D>(gVv_ref, buf, S1_OFF_WV, gX1v2);
    ref_wgrad<S1_L_SEQ, S1_D, S1_D>(gQv_ref, bX1, gWQ_ref, 0);
    ref_wgrad<S1_L_SEQ, S1_D, S1_D>(gKv_ref, bX1, gWK_ref, 0);
    ref_wgrad<S1_L_SEQ, S1_D, S1_D>(gVv_ref, bX1, gWV_ref, 0);
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) gX1_ref[t][d] = sat32g(gX1q[t][d] + gX1k[t][d] + gX1v2[t][d]);

    // === LN1 backward ===
    static long gR1add[S1_L_SEQ][S1_D];
    ref_ln_bwd(Rin1, buf, S1_OFF_LN1_G, gX1_ref, bLN1e, bLN1q, gR1add, gLN1G_ref, gLN1B_ref, 0);
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) gR_ref[t][d] = sat32g(gR_ref[t][d] + gR1add[t][d]);

    // === embed backward ===
    for (int t = 0; t < S1_L_SEQ; t++) {
        int tok = buf[S1_OFF_TOKENS + t]; if (tok < 0) tok = 0; if (tok >= S1_V) tok = S1_V - 1;
        for (int d = 0; d < S1_D; d++) if (gR_ref[t][d] != 0) gEMBED_ref[tok * S1_D + d] = sat32g(gEMBED_ref[tok * S1_D + d] + gR_ref[t][d]);
    }

    // ---- compare every gradient region kernel-vs-ref ----
    auto cmp_region = [&](const char *name, int off, const long *ref, int n) -> long {
        long mx = 0;
        for (int i = 0; i < n; i++) {
            long k = (int32_t)buf[off + i];
            long dd = labs(k - ref[i]);
            if (dd > mx) mx = dd;
        }
        return mx;
    };
    long m_gemb  = cmp_region("g_EMBED",   S1_OFF_G_EMBED,  gEMBED_ref, S1_V*S1_D);
    long m_gwq   = cmp_region("g_Wq",      S1_OFF_G_WQ,     gWQ_ref,    S1_D*S1_D);
    long m_gwk   = cmp_region("g_Wk",      S1_OFF_G_WK,     gWK_ref,    S1_D*S1_D);
    long m_gwv   = cmp_region("g_Wv",      S1_OFF_G_WV,     gWV_ref,    S1_D*S1_D);
    long m_gwo   = cmp_region("g_Wo",      S1_OFF_G_WO,     gWO_ref,    S1_D*S1_D);
    long m_gwup  = cmp_region("g_W_up",    S1_OFF_G_W_UP,   gWUP_ref,   S1_FFN*S1_D);
    long m_gwdn  = cmp_region("g_W_down",  S1_OFF_G_W_DOWN, gWDN_ref,   S1_D*S1_FFN);
    long m_gl1g  = cmp_region("g_LN1_g",   S1_OFF_G_LN1_G,  gLN1G_ref,  S1_D);
    long m_gl1b  = cmp_region("g_LN1_b",   S1_OFF_G_LN1_B,  gLN1B_ref,  S1_D);
    long m_gl2g  = cmp_region("g_LN2_g",   S1_OFF_G_LN2_G,  gLN2G_ref,  S1_D);
    long m_gl2b  = cmp_region("g_LN2_b",   S1_OFF_G_LN2_B,  gLN2B_ref,  S1_D);
#if S1_USE_LNF
    long m_glfg  = cmp_region("g_LN_F_g",  S1_OFF_G_LNF_G,  gLNFG_ref,  S1_D);
    long m_glfb  = cmp_region("g_LN_F_b",  S1_OFF_G_LNF_B,  gLNFB_ref,  S1_D);
#else
    long m_glfg = 0, m_glfb = 0;
#endif
    long m_gwh   = cmp_region("g_W_HEAD",  S1_OFF_G_W_HEAD, gWHEAD_ref, S1_V*S1_D);
    long m_gr = 0, m_gx1 = 0, m_gx2 = 0;
    for (int t = 0; t < S1_L_SEQ; t++)
        for (int d = 0; d < S1_D; d++) {
            long kr = (int32_t)buf[S1_OFF_GR  + t*S1_D + d]; if (labs(kr - gR_ref[t][d])  > m_gr)  m_gr  = labs(kr - gR_ref[t][d]);
            long k1 = (int32_t)buf[S1_OFF_GX1 + t*S1_D + d]; if (labs(k1 - gX1_ref[t][d]) > m_gx1) m_gx1 = labs(k1 - gX1_ref[t][d]);
            long k2 = (int32_t)buf[S1_OFF_GX2 + t*S1_D + d]; if (labs(k2 - gX2_ref[t][d]) > m_gx2) m_gx2 = labs(k2 - gX2_ref[t][d]);
        }

    // ---- BACKWARD headroom check (the deployed-scaling discipline of forge_pc_wgrad_tb). ----
    // The wgrad kernel now dyad-quantizes its gO operand FULL-PRECISION (dyad_wide, scans 31 bits),
    // matching charlm's dyadic_quantize(gf) on the full gradient. So gO must fit int32 (dyad_wide's
    // 31-bit scan exact) and every grad accumulator must fit int32 (xgrad/wgrad store with sat32 — a
    // grad acc >= 2^31 SATURATES, which would be a silent magnitude clamp). Both must hold for the
    // bit-exact pass to also certify the grads are faithful (not silently wrapped/saturated).
    long peak_wgrad_go = 0;   // every operand that enters a wgrad as gO (now dyad_wide -> int32 range)
    for (int t = 0; t < S1_L_SEQ; t++)
        for (int d = 0; d < S1_D; d++) {
            long a;
            a = labs(gDn_ref[t][d]);   if (a > peak_wgrad_go) peak_wgrad_go = a;  // W_down wgrad gO
            a = labs(gAO_ref[t][d]);   if (a > peak_wgrad_go) peak_wgrad_go = a;  // W_o    wgrad gO
            a = labs(gQv_ref[t][d]);   if (a > peak_wgrad_go) peak_wgrad_go = a;  // W_q    wgrad gO
            a = labs(gKv_ref[t][d]);   if (a > peak_wgrad_go) peak_wgrad_go = a;  // W_k    wgrad gO
            a = labs(gVv_ref[t][d]);   if (a > peak_wgrad_go) peak_wgrad_go = a;  // W_v    wgrad gO
        }
    for (int t = 0; t < S1_L_SEQ; t++)
        for (int k = 0; k < S1_FFN; k++) {
            long a = labs(gU_ref[t][k]); if (a > peak_wgrad_go) peak_wgrad_go = a; // W_up wgrad gO
        }
    long peak_grad_acc = 0;   // every full-precision grad accumulator (must stay < 2^31)
    for (int t = 0; t < S1_L_SEQ; t++) {
        for (int d = 0; d < S1_D; d++) {
            long a;
            a = labs(gR_ref[t][d]);  if (a > peak_grad_acc) peak_grad_acc = a;
            a = labs(gX1_ref[t][d]); if (a > peak_grad_acc) peak_grad_acc = a;
            a = labs(gX2_ref[t][d]); if (a > peak_grad_acc) peak_grad_acc = a;
        }
        for (int k = 0; k < S1_FFN; k++) {
            long a = labs(gHd_ref[t][k]); if (a > peak_grad_acc) peak_grad_acc = a;
        }
    }
    // peak weight-grad: must fit int32 (wgrad stores with sat32) AND the wgrad inner acc must fit the
    // ap_int<48> kernel accumulator (peak |gW| << 2^47 leaves margin; checked explicitly).
    long peak_gw = 0;
    auto trk = [&](const long *g, int n){ for (int i = 0; i < n; i++) { long a = labs(g[i]); if (a > peak_gw) peak_gw = a; } };
    trk(gWQ_ref, S1_D*S1_D); trk(gWK_ref, S1_D*S1_D); trk(gWV_ref, S1_D*S1_D); trk(gWO_ref, S1_D*S1_D);
    trk(gWUP_ref, S1_FFN*S1_D); trk(gWDN_ref, S1_D*S1_FFN);
    trk(gWHEAD_ref, S1_V*S1_D); trk(gEMBED_ref, S1_V*S1_D);
    trk(gLN1G_ref, S1_D); trk(gLN2G_ref, S1_D); trk(gLN1B_ref, S1_D); trk(gLN2B_ref, S1_D);

    printf("forge_train_step_s1 CSIM  (D=%d H=%d DH=%d L=%d FFN=%d V=%d LAYERS=%d)\n",
           S1_D, S1_H, S1_DH, S1_L_SEQ, S1_FFN, S1_V, S1_N_LAYERS);
    printf("  [FWD]  max|R_kernel       - ref| = %ld\n", max_R);
    printf("  [FWD]  max|X1_kernel      - ref| = %ld\n", max_X1);
    printf("  [FWD]  max|X2_kernel      - ref| = %ld\n", max_X2);
    printf("  [FWD]  max|attn_p_kernel  - ref| = %ld\n", max_P);
    printf("  [FWD]  max|attn_j_kernel  - ref| = %ld\n", max_J);
    printf("  [FWD]  max|relu_mask_kernel-ref| = %ld\n", max_RM);
    printf("  [FWD]  max|ln_stats_kernel- ref| = %ld\n", max_LNST);
    printf("  [FWD]  max|logits_kernel  - ref| = %ld   (peak |logit|=%ld)\n", max_logit, peak_logit);
    printf("  [FWD]  max|grad_logit_kern- ref| = %ld   (peak |grad|=%ld, peak |R|=%ld)\n", max_grad, peak_grad, peak);
    printf("  ----- BACKWARD (mode 1) -----\n");
    printf("  [BWD]  max|g_EMBED   - ref| = %ld\n", m_gemb);
    printf("  [BWD]  max|g_Wq      - ref| = %ld\n", m_gwq);
    printf("  [BWD]  max|g_Wk      - ref| = %ld\n", m_gwk);
    printf("  [BWD]  max|g_Wv      - ref| = %ld\n", m_gwv);
    printf("  [BWD]  max|g_Wo      - ref| = %ld\n", m_gwo);
    printf("  [BWD]  max|g_W_up    - ref| = %ld\n", m_gwup);
    printf("  [BWD]  max|g_W_down  - ref| = %ld\n", m_gwdn);
    printf("  [BWD]  max|g_LN1_g   - ref| = %ld\n", m_gl1g);
    printf("  [BWD]  max|g_LN1_b   - ref| = %ld\n", m_gl1b);
    printf("  [BWD]  max|g_LN2_g   - ref| = %ld\n", m_gl2g);
    printf("  [BWD]  max|g_LN2_b   - ref| = %ld\n", m_gl2b);
#if S1_USE_LNF
    printf("  [BWD]  max|g_LN_F_g  - ref| = %ld\n", m_glfg);
    printf("  [BWD]  max|g_LN_F_b  - ref| = %ld\n", m_glfb);
#endif
    printf("  [BWD]  max|g_W_HEAD  - ref| = %ld\n", m_gwh);
    printf("  [BWD]  max|gR        - ref| = %ld\n", m_gr);
    printf("  [BWD]  max|gX1       - ref| = %ld\n", m_gx1);
    printf("  [BWD]  max|gX2       - ref| = %ld\n", m_gx2);
    printf("  int32 headroom: peak values %s 2^31 (%ld)\n",
           (peak < 2147483647L && peak_logit < 2147483647L && peak_grad < 2147483647L) ? "<" : ">=",
           2147483647L);
    // backward faithfulness headroom: gO-into-wgrad must fit int16 (else the dyad cast wraps) and
    // every grad acc must fit int32 (else it saturates). Below these the bit-exact pass also == faithful.
    bool go_ok  = (peak_wgrad_go < 2147483647L);   // dyad_wide scans 31 bits -> gO must fit int32
    bool acc_ok = (peak_grad_acc < 2147483647L);   // grad accumulators stored with sat32
    bool gw_ok  = (peak_gw < 2147483647L);         // weight grads stored with sat32
    printf("  [BWD]  peak |gO into wgrad| = %ld  (%s int32 range 2^31; dyad_wide full-precision)\n",
           peak_wgrad_go, go_ok ? "<" : ">=");
    printf("  [BWD]  peak |grad accum|    = %ld  (%s int32 range 2^31)\n",
           peak_grad_acc, acc_ok ? "<" : ">=");
    printf("  [BWD]  peak |weight grad|   = %ld  (%s int32 range 2^31; wgrad acc in int48)\n",
           peak_gw, gw_ok ? "<" : ">=");
    int fwd_ok = (max_R == 0 && max_X1 == 0 && max_X2 == 0 && max_P == 0 && max_J == 0 &&
                  max_RM == 0 && max_LNST == 0 && max_logit == 0 && max_grad == 0);
    int bwd_ok = (m_gemb == 0 && m_gwq == 0 && m_gwk == 0 && m_gwv == 0 && m_gwo == 0 &&
                  m_gwup == 0 && m_gwdn == 0 && m_gl1g == 0 && m_gl1b == 0 && m_gl2g == 0 &&
                  m_gl2b == 0 && m_glfg == 0 && m_glfb == 0 && m_gwh == 0 && m_gr == 0 &&
                  m_gx1 == 0 && m_gx2 == 0 &&
                  go_ok && acc_ok && gw_ok);   // bit-exact AND no silent int32 saturation anywhere
    printf("  -> FWD %s ; BWD %s\n",
           fwd_ok ? "BIT-EXACT" : "MISMATCH", bwd_ok ? "BIT-EXACT (in-range)" : "MISMATCH/OUT-OF-RANGE");

    // ============================================================================
    // ==============  MODE 2 (DYADIC-ADAM) ONE-STEP BIT-EXACT check  ==============
    // ============================================================================
    // Run mode 0 -> 1 -> 2 ONCE on a consistent QAT buffer (small int16 masters; codes produced by the
    // mode-2 INIT path). Snapshot g/m/v/W_fp BEFORE mode 2, run mode 2, then a CPU int64 dyadic-Adam
    // reference of the SAME recipe (per region) and assert max|kernel - ref| == 0 over the updated
    // W_fp + re-quantized code/value + m + v for EVERY learnable region.
    printf("  ----- MODE 2 (dyadic-Adam) one-step bit-exact -----\n");
    int32_t *ob = (int32_t *)aligned_alloc(64, sizeof(int32_t) * S1_BUFLEN);
    stage_opt_buffer(ob, 7777, /*memorize=*/false);

    // (init) re-quantize the staged W_fp into the forward-read codes/values; verify INIT is bit-exact too.
    forge_train_step_s1(ob, 2, S1_OPT_INIT, 0, 0);
    long max_init = 0;
    for (int r = 0; r < S1_NUM_OPT_REGIONS; r++) {
        const OptRegRef &R = OPT_REGS_REF[r];
        for (int p = 0; p < R.n; p++) {
            long want;
            if (R.is_code) {
                // FINE FIXED-POINT code master: init re-quantizes via dyad(round(master/2^Q)), no clamp.
                want = ref_s1_encode_fine((int32_t)ob[R.wfp_off + p], /*prev_code=*/0, /*allow_clamp=*/false);
            } else {
                long wfp = (int16_t)(ob[R.wfp_off + p] & 0xFFFF);
                want = (int32_t)(int16_t)wfp;
            }
            long got  = (int32_t)ob[R.dst_off + p];
            long dd = labs(got - want); if (dd > max_init) max_init = dd;
        }
    }
    printf("  [OPT-INIT] max|code/value - dyad(W_fp)| = %ld\n", max_init);

    // forward + backward -> grads (the SAME bit-exact path proven above; here it just FEEDS the optimizer).
    forge_train_step_s1(ob, 0, 0, 0, 0);
    forge_train_step_s1(ob, 1, 0, 0, 0);

    // snapshot pre-mode-2 state (g/m/v/W_fp) for the CPU reference.
    static long g_snap[S1_FFN * S1_D], m_snap[S1_FFN * S1_D], v_snap[S1_FFN * S1_D], wfp_snap[S1_FFN * S1_D];
    static long m_ref[S1_FFN * S1_D], v_ref[S1_FFN * S1_D], wfp_ref[S1_FFN * S1_D], dst_ref[S1_FFN * S1_D], g_ref2[S1_FFN * S1_D];
    // we snapshot region-by-region (the largest region is W_up = FFN*D, sizing the scratch above).

    // run the kernel optimizer step.
    long pre_g_peak = 0;
    {   // track the largest |grad| entering the optimizer (sanity: the step is non-trivial).
        for (int r = 0; r < S1_NUM_OPT_REGIONS; r++) {
            const OptRegRef &R = OPT_REGS_REF[r];
            for (int p = 0; p < R.n; p++) { long a = labs((long)(int32_t)ob[R.g_off + p]); if (a > pre_g_peak) pre_g_peak = a; }
        }
    }
    // snapshot ALL regions' g/m/v/W_fp + the deploy code (for the exp-clamp prev_code) BEFORE the step.
    static long g_all[S1_NUM_OPT_REGIONS][S1_FFN * S1_D];
    static long m_all[S1_NUM_OPT_REGIONS][S1_FFN * S1_D];
    static long v_all[S1_NUM_OPT_REGIONS][S1_FFN * S1_D];
    static long w_all[S1_NUM_OPT_REGIONS][S1_FFN * S1_D];
    static long d_all[S1_NUM_OPT_REGIONS][S1_FFN * S1_D];
    for (int r = 0; r < S1_NUM_OPT_REGIONS; r++) {
        const OptRegRef &R = OPT_REGS_REF[r];
        snap(ob, R.g_off,   g_all[r], R.n);
        snap(ob, R.m_off,   m_all[r], R.n);
        snap(ob, R.v_off,   v_all[r], R.n);
        snap(ob, R.wfp_off, w_all[r], R.n);
        snap(ob, R.dst_off, d_all[r], R.n);
    }

    forge_train_step_s1(ob, 2, S1_OPT_STEP, 0, 0);     // the dyadic-Adam step (DUT)

    long max_opt_wfp = 0, max_opt_dst = 0, max_opt_m = 0, max_opt_v = 0, max_opt_gz = 0;
    for (int r = 0; r < S1_NUM_OPT_REGIONS; r++) {
        const OptRegRef &R = OPT_REGS_REF[r];
        ref_dyadic_adam_region(g_all[r], m_all[r], v_all[r], w_all[r], d_all[r], R.n, R.is_code, /*init_only=*/false,
                               m_ref, v_ref, wfp_ref, dst_ref, g_ref2);
        long dw = cmp_buf(ob, R.wfp_off, wfp_ref, R.n); if (dw > max_opt_wfp) max_opt_wfp = dw;
        long dd = cmp_buf(ob, R.dst_off, dst_ref, R.n); if (dd > max_opt_dst) max_opt_dst = dd;
        long dm = cmp_buf(ob, R.m_off,   m_ref,   R.n); if (dm > max_opt_m)   max_opt_m   = dm;
        long dv = cmp_buf(ob, R.v_off,   v_ref,   R.n); if (dv > max_opt_v)   max_opt_v   = dv;
        long dg = cmp_buf(ob, R.g_off,   g_ref2,  R.n); if (dg > max_opt_gz)  max_opt_gz  = dg;  // grads zeroed
    }
    printf("  [OPT]  peak |grad into Adam| = %ld\n", pre_g_peak);
    printf("  [OPT]  max|W_fp_kernel  - ref| = %ld\n", max_opt_wfp);
    printf("  [OPT]  max|code/value   - ref| = %ld\n", max_opt_dst);
    printf("  [OPT]  max|m_kernel     - ref| = %ld\n", max_opt_m);
    printf("  [OPT]  max|v_kernel     - ref| = %ld\n", max_opt_v);
    printf("  [OPT]  max|grad (zeroed)- ref| = %ld\n", max_opt_gz);
    int opt_ok = (max_init == 0 && max_opt_wfp == 0 && max_opt_dst == 0 &&
                  max_opt_m == 0 && max_opt_v == 0 && max_opt_gz == 0);
    printf("  -> MODE 2 (dyadic-Adam) %s\n", opt_ok ? "BIT-EXACT" : "MISMATCH");
    free(ob);

#ifndef __VITIS_HLS__   // heavy multi-step convergence runs in native g++ ONLY. Vitis csim does the one-step
                        // bit-exact checks above (ap_int == int64 for one step) then proceeds to csynth; the
                        // N-step convergence is proven in g++ and is transitively valid (deterministic, one
                        // bit-exact step => N bit-exact steps). Running ~1440 train steps in csim_design would
                        // exceed the build backstop.
    // ============================================================================
    // =========  MODE 3 (FUSED TRAINER) N-STEP CONVERGENCE check  =================
    // ============================================================================
    // (1) HOST-DRIVEN trajectory: mode-2-INIT once, then loop N steps of mode 0 (read LOGITS) -> 1 -> 2,
    //     in the S1_OPT_FREEZE_CODES regime (the dense PRE-DYADIC backbone is frozen; only EMBED/W_HEAD +
    //     LN beta train). This is the CPU int64 reference TRAJECTORY of the recipe (each mode bit-exact-
    //     verified above) and lets us PRINT a TRUE, NON-FAKEABLE cross-entropy per step.
    // (2) FUSED: run mode 3 (do_train_run, ONE host kick = N steps) from the SAME fresh init in the SAME
    //     FREEZE_CODES regime and assert the FINAL W_fp + codes/values + m + v match the host-driven loop
    //     BIT-EXACT — the fused on-fabric trainer == the sequenced per-mode trajectory, per step (no drift).
    //
    // WHY FREEZE_CODES: with the full dense stack trainable, the dyadic re-quantize RATCHETS the int8 code
    // magnitudes upward each step (a ±1 int16-master nudge crosses a dyad() power-of-2 boundary, doubling
    // the code), the GEMM blows the residual stream into the int16 sat ceiling, and the max-plus head
    // collapses to UNIFORM logits (true CE pinned at log V) — the all-trainable "convergence" was a
    // saturation artifact (the OLD test asserted on a target-margin that reads 0 under uniform saturation,
    // and an acc_final>=acc0 that is vacuous when acc0==0). Freezing the verified dyadic backbone and
    // training the scatter weights is the QAT regime in which the dyadic-Adam trainer GENUINELY reduces CE.
    //
    // THE CONVERGENCE METRIC IS A TRUE FLOAT CROSS-ENTROPY computed from the saved LOGITS — it CANNOT be
    // faked by saturation (uniform logits => CE == log V == the FLOOR, which FAILS the test). Convergence
    // requires (a) CE drops by a substantial margin, (b) the final distribution is NOT the uniform-collapse
    // floor, and (c) argmax accuracy strictly improves. The loss must move because the WEIGHTS move.
    const int N_STEPS = 120;
    const int FRZ = S1_OPT_FREEZE_CODES;   // a2 for modes 2 & 3 in the convergence run
    printf("  ----- MODE 3 (fused trainer) %d-step convergence (memorization, FREEZE_CODES) -----\n", N_STEPS);

    // TRUE cross-entropy from the saved LOGITS: -mean_t log softmax(logit[t])[target]. The logit scale is
    // O(1e3-1e4) so we evaluate the softmax at a small temperature (0.01) purely to keep exp() in range —
    // this is a strictly-monotone reparam of CE, so "CE decreases" is invariant to it. UNIFORM logits give
    // CE == log V exactly (the floor), which is what saturation collapse produces — and that FAILS.
    const double CE_TEMP = 0.01;
    const double CE_FLOOR = log((double)S1_V);    // = uniform-distribution CE (the saturation-collapse value)
    auto true_ce = [&](const int32_t *bb) -> double {
        double tot = 0.0;
        for (int t = 0; t < S1_L_SEQ; t++) {
            int tgt = bb[S1_OFF_TARGETS + t]; if (tgt < 0) tgt = 0; if (tgt >= S1_V) tgt = S1_V - 1;
            double mx = -1e300;
            for (int v = 0; v < S1_V; v++) { double lv = (int32_t)bb[S1_OFF_LOGITS + t * S1_V + v]; if (lv > mx) mx = lv; }
            double Z = 0.0;
            for (int v = 0; v < S1_V; v++) Z += exp(((double)(int32_t)bb[S1_OFF_LOGITS + t * S1_V + v] - mx) * CE_TEMP);
            double lt = ((double)(int32_t)bb[S1_OFF_LOGITS + t * S1_V + tgt] - mx) * CE_TEMP;
            tot += -(lt - log(Z));
        }
        return tot / (double)S1_L_SEQ;
    };

    // accuracy over the saved LOGITS region (argmax_v logits[t][v] == target[t]); a corroborating signal.
    auto count_correct = [&](const int32_t *bb) -> int {
        int correct = 0;
        for (int t = 0; t < S1_L_SEQ; t++) {
            int tgt = bb[S1_OFF_TARGETS + t]; if (tgt < 0) tgt = 0; if (tgt >= S1_V) tgt = S1_V - 1;
            long lmax = -2147483647L; int pred = 0;
            for (int v = 0; v < S1_V; v++) { long lv = (int32_t)bb[S1_OFF_LOGITS + t * S1_V + v]; if (lv > lmax) { lmax = lv; pred = v; } }
            if (pred == tgt) correct++;
        }
        return correct;
    };

    int32_t *hb = (int32_t *)aligned_alloc(64, sizeof(int32_t) * S1_BUFLEN);
    stage_opt_buffer(hb, 4242, /*memorize=*/true);
    forge_train_step_s1(hb, 2, S1_OPT_INIT, FRZ, 0);   // populate codes/values from staged W_fp
    double ce_traj[256]; int acc_traj[256];
    for (int s = 0; s < N_STEPS; s++) {
        forge_train_step_s1(hb, 0, 0, 0, 0);                       // forward + CE  (writes LOGITS)
        ce_traj[s]  = true_ce(hb);                                 // TRUE float CE (un-fakeable)
        acc_traj[s] = count_correct(hb);
        forge_train_step_s1(hb, 1, 0, 0, 0);                       // backward
        forge_train_step_s1(hb, 2, S1_OPT_STEP, FRZ, 0);          // dyadic-Adam (freeze codes)
    }
    // final CE/acc after the last update.
    forge_train_step_s1(hb, 0, 0, 0, 0);
    double ce_final = true_ce(hb);
    int    acc_final = count_correct(hb);

    printf("  TRUE cross-entropy trajectory (-mean_t log softmax(logits)[target]; uniform/collapse = %.4f):\n", CE_FLOOR);
    for (int s = 0; s < N_STEPS; s++) {
        if (s < 6 || s % 20 == 0 || s == N_STEPS - 1)
            printf("    step %3d  CE = %.4f   acc = %3d/%d\n", s, ce_traj[s], acc_traj[s], S1_L_SEQ);
    }
    printf("    final     CE = %.4f   acc = %3d/%d\n", ce_final, acc_final, S1_L_SEQ);
    double ce0 = ce_traj[0]; int acc0 = acc_traj[0];
    // CONVERGENCE (all three required; none satisfiable by saturation collapse):
    //   (a) CE drops by a substantial absolute margin (>= 0.5 nats) — REAL learning, not noise.
    //   (b) the final CE is meaningfully BELOW the uniform/collapse floor (so it is NOT the saturation
    //       artifact: a collapsed model sits AT log V and would FAIL this).
    //   (c) argmax accuracy strictly improves (corroborating; collapse freezes acc at the tie-break value).
    bool ce_drops   = (ce0 - ce_final) >= 0.5;
    bool not_collapsed = ce_final < (CE_FLOOR - 0.25);
    bool acc_improves  = (acc_final > acc0);
    bool conv_ok = ce_drops && not_collapsed && acc_improves;
    printf("  -> CE %.4f -> %.4f  (drop %.4f nats; floor %.4f)  acc %d -> %d/%d  %s\n",
           ce0, ce_final, ce0 - ce_final, CE_FLOOR, acc0, acc_final, S1_L_SEQ,
           conv_ok ? "CONVERGES" : "NO-CONVERGENCE");
    printf("     [conv gates] CE-drops>=0.5: %s  below-floor: %s  acc-improves: %s\n",
           ce_drops ? "yes" : "NO", not_collapsed ? "yes" : "NO", acc_improves ? "yes" : "NO");

    // (2) fused mode-3 == host-driven loop, bit-exact final state (SAME FREEZE_CODES regime + seed/init).
    int32_t *fb = (int32_t *)aligned_alloc(64, sizeof(int32_t) * S1_BUFLEN);
    stage_opt_buffer(fb, 4242, /*memorize=*/true);     // SAME seed/init as the host-driven run
    forge_train_step_s1(fb, 3, N_STEPS, FRZ, 0);        // ONE host kick = N full optimizer steps (freeze codes)

    long max_fused = 0;
    const char *worst = "";
    for (int r = 0; r < S1_NUM_OPT_REGIONS; r++) {
        const OptRegRef &R = OPT_REGS_REF[r];
        long mw = 0, md = 0, mm = 0, mv = 0;
        for (int p = 0; p < R.n; p++) {
            long a;
            a = labs((long)(int32_t)fb[R.wfp_off + p] - (int32_t)hb[R.wfp_off + p]); if (a > mw) mw = a;
            a = labs((long)(int32_t)fb[R.dst_off + p] - (int32_t)hb[R.dst_off + p]); if (a > md) md = a;
            a = labs((long)(int32_t)fb[R.m_off   + p] - (int32_t)hb[R.m_off   + p]); if (a > mm) mm = a;
            a = labs((long)(int32_t)fb[R.v_off   + p] - (int32_t)hb[R.v_off   + p]); if (a > mv) mv = a;
        }
        long rmax = mw; if (md > rmax) rmax = md; if (mm > rmax) rmax = mm; if (mv > rmax) rmax = mv;
        if (rmax > max_fused) { max_fused = rmax; worst = OPT_REG_NAMES[r]; }
    }
    // also compare the fused final TRUE CE to the host-driven final CE (read mode 0 on fb): the fused
    // on-fabric trainer must reach the IDENTICAL learned state (so the same CE), not just match weights.
    forge_train_step_s1(fb, 0, 0, 0, 0);
    double fused_ce_final = true_ce(fb);
    printf("  [FUSED] max|mode3 - host-loop| over W_fp/code/m/v = %ld  (worst region: %s)\n",
           max_fused, max_fused ? worst : "none");
    printf("  [FUSED] mode-3 final CE = %.4f   (host-loop final CE = %.4f)\n", fused_ce_final, ce_final);
    bool fused_ok = (max_fused == 0 && fused_ce_final == ce_final);
    printf("  -> MODE 3 fused == host-loop %s\n", fused_ok ? "BIT-EXACT" : "MISMATCH");
    free(hb); free(fb);

    // ============================================================================
    // ====  TRAIN_ALL (DENSE-CODE BACKBONE TRAINABLE) convergence + negative control  ====
    // ============================================================================
    // The OPEN research question: with the dense PRE-DYADIC CODE backbone ALSO trainable (a2 =
    // S1_OPT_TRAIN_ALL=0; Wq/Wk/Wv/Wo/W_up/W_down + LN gamma step too), does the model still converge —
    // or does the all-trainable dyadic re-quantize drive saturation/collapse? This block runs the SAME
    // memorization task with a SMALL-NONZERO sparse dyadic code init (so the dense backbone gets a real
    // gradient — the identity init gives the dense codes exactly-zero gradient, see stage_train_all), at
    // a2=TRAIN_ALL, and reports the TRUE float CE trajectory + the honest discriminator
    // "code-bytes-changed" (did the dense codes actually move, or is TRAIN_ALL silently == FREEZE).
    //
    // NEGATIVE CONTROL: from the SAME init, run only forward (NO backward, NO optimizer step) for the same
    // horizon. CE MUST stay flat (no weight moves => no learning). If the "convergence" above moved the CE,
    // it must be the WEIGHTS moving, not an artifact of re-reading logits.
    printf("  ----- TRAIN_ALL (dense-code backbone trainable) convergence + negative control -----\n");
    printf("        [fine fixed-pt code master: Q_CODE=%d, LR_SHIFT_CODE=%d, EXP_CLAMP=%d, CODE_WD_SHIFT=%d, USE_LNF=%d]\n",
           S1_Q_CODE, S1_LR_SHIFT_CODE, S1_CODE_EXP_CLAMP, S1_CODE_WD_SHIFT, S1_USE_LNF);
    static int8_t ta_init_codes[262144];
    const int TA_DENSE = 5;    // 5% nonzero dyadic code init (gives the dense backbone gradient signal)
    const int TA_EMB   = 20;   // small embed/head init -> R in deployed range at start
    // The dense-backbone benefit over FREEZE GROWS with the horizon (measured: ~0.04 nats @120 -> ~0.05 @400
    // steps, all-seeds-consistent). Use a longer horizon for the TRAIN_ALL block so the ablation has a clear,
    // un-fakeable margin. (The FREEZE convergence + fused bit-exact checks above keep the shorter N_STEPS.)
    const int TA_STEPS = 400;

    // (A) TRAIN_ALL trajectory.
    int32_t *tb = (int32_t *)aligned_alloc(64, sizeof(int32_t) * S1_BUFLEN);
    stage_train_all(tb, 4242, TA_DENSE, TA_EMB);
    forge_train_step_s1(tb, 2, S1_OPT_INIT, S1_OPT_TRAIN_ALL, 0);
    snapshot_codes(tb, ta_init_codes);
    forge_train_step_s1(tb, 0, 0, 0, 0);
    double ta_ce0 = true_ce(tb); int ta_acc0 = count_correct(tb);
    for (int s = 0; s < TA_STEPS; s++) {
        forge_train_step_s1(tb, 0, 0, 0, 0);
        forge_train_step_s1(tb, 1, 0, 0, 0);
        forge_train_step_s1(tb, 2, S1_OPT_STEP, S1_OPT_TRAIN_ALL, 0);
    }
    forge_train_step_s1(tb, 0, 0, 0, 0);
    double ta_ce_final = true_ce(tb); int ta_acc_final = count_correct(tb);
    int ta_code_changes = count_code_changes(tb, ta_init_codes);
    long ta_peakR = 0;
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) { long a = labs((long)(int32_t)tb[s1_off_R(S1_N_LAYERS) + t*S1_D + d]); if (a > ta_peakR) ta_peakR = a; }

    // (A2) FUSED mode-3 TRAIN_ALL == host-driven TRAIN_ALL loop, bit-exact final state (proves the
    //      on-fabric fused trainer matches the per-mode loop even with the dense CODE backbone trainable —
    //      i.e. the fine fixed-point master + exp-clamp re-quantize is fused-faithful, not just freeze-only).
    int32_t *tfb = (int32_t *)aligned_alloc(64, sizeof(int32_t) * S1_BUFLEN);
    stage_train_all(tfb, 4242, TA_DENSE, TA_EMB);
    forge_train_step_s1(tfb, 3, TA_STEPS, S1_OPT_TRAIN_ALL, 0);
    long ta_max_fused = 0; const char *ta_worst = "";
    for (int r = 0; r < S1_NUM_OPT_REGIONS; r++) {
        const OptRegRef &R = OPT_REGS_REF[r];
        long rm = 0;
        for (int p = 0; p < R.n; p++) {
            long a;
            a = labs((long)(int32_t)tfb[R.wfp_off + p] - (int32_t)tb[R.wfp_off + p]); if (a > rm) rm = a;
            a = labs((long)(int32_t)tfb[R.dst_off + p] - (int32_t)tb[R.dst_off + p]); if (a > rm) rm = a;
            a = labs((long)(int32_t)tfb[R.m_off   + p] - (int32_t)tb[R.m_off   + p]); if (a > rm) rm = a;
            a = labs((long)(int32_t)tfb[R.v_off   + p] - (int32_t)tb[R.v_off   + p]); if (a > rm) rm = a;
        }
        if (rm > ta_max_fused) { ta_max_fused = rm; ta_worst = OPT_REG_NAMES[r]; }
    }
    bool ta_fused_ok = (ta_max_fused == 0);
    free(tfb);

    // (A3) DECISIVE DISCRIMINATOR — does the DENSE BACKBONE genuinely HELP, or is the CE drop entirely the
    //      SCATTER/VALUE weights (embed/head/LN beta) that train under TRAIN_ALL anyway? Run FREEZE_CODES
    //      from the IDENTICAL init/seed/horizon (only the dense codes differ: TRAIN_ALL steps them, FREEZE
    //      holds them) and compare the final TRUE CE. If TRAIN_ALL < FREEZE, the trainable dense codes are
    //      genuinely contributing; if TRAIN_ALL >= FREEZE, the codes move but add nothing (the "codes-moved"
    //      gate alone is NOT proof of benefit — Q=10/no-WD showed codes that move yet match FREEZE exactly).
    //      This is the un-fakeable "the dense backbone helps" test: it ablates ONLY the code trainability.
    int32_t *zb = (int32_t *)aligned_alloc(64, sizeof(int32_t) * S1_BUFLEN);
    stage_train_all(zb, 4242, TA_DENSE, TA_EMB);
    forge_train_step_s1(zb, 2, S1_OPT_INIT, S1_OPT_FREEZE_CODES, 0);
    for (int s = 0; s < TA_STEPS; s++) {
        forge_train_step_s1(zb, 0, 0, 0, 0);
        forge_train_step_s1(zb, 1, 0, 0, 0);
        forge_train_step_s1(zb, 2, S1_OPT_STEP, S1_OPT_FREEZE_CODES, 0);
    }
    forge_train_step_s1(zb, 0, 0, 0, 0);
    double frz_ce_final = true_ce(zb);
    free(zb);

    // (B) NEGATIVE CONTROL: same init, forward-only (no bwd, no opt) — CE must NOT move.
    int32_t *nb = (int32_t *)aligned_alloc(64, sizeof(int32_t) * S1_BUFLEN);
    stage_train_all(nb, 4242, TA_DENSE, TA_EMB);
    forge_train_step_s1(nb, 2, S1_OPT_INIT, S1_OPT_TRAIN_ALL, 0);
    forge_train_step_s1(nb, 0, 0, 0, 0);
    double nc_ce0 = true_ce(nb);
    for (int s = 0; s < TA_STEPS; s++) forge_train_step_s1(nb, 0, 0, 0, 0);   // forward only
    double nc_ce_final = true_ce(nb);

    printf("  TRAIN_ALL:  CE %.4f -> %.4f  (drop %.4f nats; floor %.4f)  acc %d -> %d/%d  peak|R|=%ld\n",
           ta_ce0, ta_ce_final, ta_ce0 - ta_ce_final, CE_FLOOR, ta_acc0, ta_acc_final, S1_L_SEQ, ta_peakR);
    printf("  TRAIN_ALL:  dense code bytes changed from init = %d  (0 => codes never moved == FREEZE; >0 => dense backbone genuinely stepped)\n", ta_code_changes);
    printf("  ABLATION :  TRAIN_ALL final CE %.4f  vs  FREEZE(same init) final CE %.4f  (TA - FRZ = %+.4f; <0 => dense codes HELP)\n",
           ta_ce_final, frz_ce_final, ta_ce_final - frz_ce_final);
    printf("  NEG-CTRL :  forward-only CE %.4f -> %.4f  (drift %.4f; must be ~0)\n",
           nc_ce0, nc_ce_final, nc_ce0 - nc_ce_final);
    bool ta_ce_drops    = (ta_ce0 - ta_ce_final) >= 0.5;
    bool ta_not_collapsed = ta_ce_final < (CE_FLOOR - 0.25);
    bool ta_acc_improves  = (ta_acc_final > ta_acc0);
    bool ta_codes_moved   = (ta_code_changes > 0);          // the dense backbone GENUINELY stepped
    bool ta_no_saturation = (ta_peakR < 32767);             // residual not pinned at the int16 ceiling
    bool ta_beats_freeze  = (ta_ce_final < frz_ce_final - 1e-9);  // the DECISIVE gate: trainable codes HELP
    bool nc_holds         = (nc_ce_final >= nc_ce0 - 1e-9); // forward-only does NOT improve (drift <= 0)
    // GENUINE = converges WITHOUT collapse/saturation AND the dense codes ACTUALLY HELP (beat FREEZE). The
    // codes-moved + no-saturation + below-floor gates rule out the two failure modes (frozen / saturated);
    // ta_beats_freeze rules out the third (codes move but contribute nothing).
    bool train_all_genuine = ta_ce_drops && ta_not_collapsed && ta_acc_improves && ta_codes_moved &&
                             ta_no_saturation && ta_beats_freeze;
    printf("     [TRAIN_ALL gates] CE-drops>=0.5: %s  below-floor: %s  acc-improves: %s  codes-moved: %s  no-saturation: %s  beats-FREEZE: %s\n",
           ta_ce_drops ? "yes" : "NO", ta_not_collapsed ? "yes" : "NO", ta_acc_improves ? "yes" : "NO",
           ta_codes_moved ? "yes" : "NO", ta_no_saturation ? "yes" : "NO", ta_beats_freeze ? "yes" : "NO");
    printf("     [neg control] forward-only does NOT improve: %s\n", nc_holds ? "yes" : "NO");
    printf("  [FUSED-TA] max|mode3(TRAIN_ALL) - host-loop| over W_fp/code/m/v = %ld  (worst: %s) -> %s\n",
           ta_max_fused, ta_max_fused ? ta_worst : "none", ta_fused_ok ? "BIT-EXACT" : "MISMATCH");
    printf("  -> TRAIN_ALL (dense backbone genuinely trains) %s ; negative control %s ; fused %s\n",
           train_all_genuine ? "YES" : "NO (see honest report)", nc_holds ? "HOLDS" : "VIOLATED",
           ta_fused_ok ? "BIT-EXACT" : "MISMATCH");
    free(tb); free(nb);

    int ok = fwd_ok && bwd_ok && opt_ok && conv_ok && fused_ok;
    printf("  ============================================================\n");
    printf("  S1 TRAINER: FWD %s ; BWD %s ; OPT %s ; CONVERGE(FREEZE) %s ; FUSED %s ; TRAIN_ALL %s\n",
           fwd_ok ? "OK" : "FAIL", bwd_ok ? "OK" : "FAIL", opt_ok ? "OK" : "FAIL",
           conv_ok ? "OK" : "FAIL", fused_ok ? "OK" : "FAIL",
           train_all_genuine ? "GENUINE-TRAIN" : "DOES-NOT-GENUINELY-TRAIN");
    free(buf);
    return ok ? 0 : 1;
#else   // __VITIS_HLS__ : Vitis csim path — one-step bit-exact only (N-step convergence is the g++ path)
    int ok = fwd_ok && bwd_ok && opt_ok;
    printf("  ============================================================\n");
    printf("  S1 TRAINER (Vitis csim, one-step bit-exact): FWD %s ; BWD %s ; OPT %s\n",
           fwd_ok ? "OK" : "FAIL", bwd_ok ? "OK" : "FAIL", opt_ok ? "OK" : "FAIL");
    free(buf);
    return ok ? 0 : 1;
#endif
}
