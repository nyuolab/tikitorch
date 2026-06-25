// forge_pc_xformer_golden.hpp — plain-int64 reference arithmetic for the distributed PC transformer block.
// This is BOTH (a) the bit-exact golden the testbench compares the HLS kernel against, and (b) a standalone,
// g++-compilable, locally-RUNNABLE reference I validate by training a tiny char-LM (loss must fall) + a
// finite-difference gradient check on the exact-gradient (matmul-weight) parts. The HLS kernel
// (forge_pc_xformer_block.cpp) transcribes these exact integer ops into ap_int. NO floats, NO multiplies in
// the datapath except the attention soft-blend + its backward (gV/gp/gs) — the one allowed multiply.
//
// Dims/offsets come from forge_pc_xformer_config.hpp. Arithmetic mirrors the forward kernel verbatim; the
// backward follows the audit-workflow design (LN-bwd = straight-through-invq + gamma-STE + mean-subtract, as
// in the LLM-instance forge_layernorm_bwd; attention-bwd = the three multiplies + L1 sign-subgradient).
#ifndef FORGE_PC_XFORMER_GOLDEN_HPP
#define FORGE_PC_XFORMER_GOLDEN_HPP
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "forge_pc_xformer_config.hpp"

typedef int64_t i64;

// ---------- core integer ops (match the kernel's dyad / norm_shift / fdyad exactly) ----------
static inline i64 g_dyadval(i64 w) {           // round-to-nearest signed power of two (== kernel dyad())
    i64 a = w < 0 ? -w : w; if (a == 0) return 0;
    if (a > 32767) a = 32767;                  // int16 master clamp (kernel data_t = ap_int<16>)
    int m = 0; for (int b = 15; b >= 1; b--) if ((a >> b) & 1) { m = b; break; }
    if (m >= 1 && ((a >> (m - 1)) & 1)) m++;    // round-to-nearest bump
    i64 p = (i64)1 << m; return w < 0 ? -p : p;
}
static inline void g_dyad(i64 w, int &sgn, int &e, int &nz) {   // sign/exp/nz form (for q(.)^T q(.) wgrad)
    i64 a = w < 0 ? -w : w; sgn = w < 0; nz = (a != 0); if (!nz) { e = 0; return; }
    if (a > 32767) a = 32767;
    int m = 0; for (int b = 15; b >= 1; b--) if ((a >> b) & 1) { m = b; break; }
    if (m >= 1 && ((a >> (m - 1)) & 1)) m++; e = m;
}
static inline int g_bitlen(i64 v) { i64 a = v < 0 ? -v : v; int h = 0; for (int b = 47; b >= 1; b--) if ((a >> b) & 1) { h = b; break; } return h + 1; }
static inline int g_normshift(i64 mx, int target) { int s = g_bitlen(mx) - target; return s > 0 ? s : 0; }
static inline i64 g_floorshr(i64 v, int s) { return v >> s; }   // arithmetic >> = floor-div (gcc/x86)

// fine-dyadic 1/scale -> (e,nq): invq ~ (nq/4)*2^e, nq in {4..7} (== kernel fdyad())
static inline void g_fdyad(i64 scale, int &e_out, int &nq_out) {
    if (scale < 1) scale = 1;
    int p = 0; for (int b = 46; b >= 1; b--) if ((scale >> b) & 1) { p = b; break; }
    int exact = (scale == ((i64)1 << p));
    int e = exact ? -p : -(p + 1); int nege = -e;
    i64 thresh = ((i64)1 << (X_MBITS + nege)) + (scale >> 1);
    int nbase = (1 << X_MBITS), nq = nbase; i64 acc = scale << X_MBITS;
    for (int st = 1; st <= (1 << X_MBITS); st++) { acc += scale; if (acc <= thresh) nq = nbase + st; }
    if (nq >= (1 << (X_MBITS + 1))) { nq = (1 << X_MBITS); e += 1; }
    if (nq < (1 << X_MBITS)) nq = (1 << X_MBITS);
    e_out = e; nq_out = nq;
}
// apply invq=(nq/4)*2^e to a value (shift-add of nq copies then the e shift) — used fwd (on c) and bwd (STE)
static inline i64 g_applyinvq(i64 cv, int e, int nq) {
    int nqm = nq - (1 << X_MBITS);
    i64 cn = cv << X_MBITS; if (nqm >= 1) cn += cv; if (nqm >= 2) cn += cv; if (nqm >= 3) cn += cv;
    i64 r = cn >> X_MBITS;
    if (e >= 0) r = r << e; else r = r >> (-e);
    return r;
}

// ---------- forward: dyadic matmul y[M*O] = sum_i dyadval(W[o*I+i]) * x[t*I+i] ----------
static void g_matmul(const i64 *x, const i64 *W, i64 *y, int M, int I, int O) {
    for (int t = 0; t < M; t++) for (int o = 0; o < O; o++) {
        i64 acc = 0; for (int i = 0; i < I; i++) acc += g_dyadval(W[o * I + i]) * x[t * I + i];
        y[t * O + o] = acc;
    }
}
static void g_rescale(i64 *a, int n, int bits) {        // in-place max-shift to ~bits
    i64 mx = 0; for (int i = 0; i < n; i++) { i64 v = a[i] < 0 ? -a[i] : a[i]; if (v > mx) mx = v; }
    int sh = g_normshift(mx, bits); for (int i = 0; i < n; i++) a[i] = g_floorshr(a[i], sh);
}
// per-token LayerNorm; saves (e,nq) stats; y = dyadic_gamma(c*invq) + beta
static void g_layernorm(const i64 *x, const i64 *gcode, const i64 *beta, i64 *y, int *stats, int M) {
    for (int t = 0; t < M; t++) {
        i64 sum = 0; for (int d = 0; d < X_D; d++) sum += x[t * X_D + d];
        i64 mean = sum >> X_LOG2_D;
        i64 c[X_D]; i64 asum = 0;
        for (int d = 0; d < X_D; d++) { i64 cv = x[t * X_D + d] - mean; c[d] = cv; asum += cv < 0 ? -cv : cv; }
        i64 meanabs = asum >> X_LOG2_D; int e, nq; g_fdyad(meanabs, e, nq);
        stats[2 * t] = e; stats[2 * t + 1] = nq;
        for (int d = 0; d < X_D; d++) {
            i64 r = g_applyinvq(c[d], e, nq);
            int gs, ge, gnz; g_dyad(gcode[d], gs, ge, gnz);
            i64 g = gnz ? (r << ge) : 0; if (gnz && gs) g = -g;
            y[t * X_D + d] = g + beta[d];
        }
    }
}

#endif
