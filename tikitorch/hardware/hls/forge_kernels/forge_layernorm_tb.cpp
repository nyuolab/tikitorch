// forge_layernorm_tb.cpp — Vitis HLS C-simulation testbench for the fine-dyadic LayerNorm forward.
// Runs the ACTUAL kernel C++ against an int64 reference whose mean/meanabs/fdyad/gamma math is
// BIT-IDENTICAL to the kernel, over DEPLOYED-RANGE random operands. Asserts max|kernel-ref| == 0
// (BIT-EXACT) over y and the saved stats. Build: csim adds this as the testbench, forge_layernorm.cpp
// as the design. (Clone of forge_pc_wgrad_tb.cpp.)

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ap_int.h>

#define TB_D      128
#define TB_LOG2_D 7
#define TB_N      16
#define TB_MBITS  2

static const int O_X     = 0;
static const int O_GAMMA = O_X     + TB_N * TB_D;
static const int O_BETA  = O_GAMMA + TB_D;
static const int O_Y     = O_BETA  + TB_D;
static const int O_STATS = O_Y     + TB_N * TB_D;
static const int BUFLEN  = O_STATS + TB_N * 2;

void forge_layernorm(int32_t *buffer);                 // design under test

static inline long sat16(long v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return v;
}

// reference fine-dyadic 1/scale -> (e, nq), BIT-IDENTICAL to the kernel's fdyad().
// NO divide: same bounded shift-add candidate compare over n in {4..8} the kernel uses.
static void ref_fdyad(long scale, int &e_out, int &nq_out) {
    if (scale < 1) scale = 1;
    int p = 0;
    for (int b = 47; b >= 1; b--) { if ((scale >> b) & 1L) { p = b; break; } }
    bool exact_pow2 = (scale == (1L << p));
    int e = exact_pow2 ? (-p) : (-(p + 1));
    int nege = -e;
    long numer  = 1L << (TB_MBITS + nege);             // 4 << nege
    long thresh = numer + (scale >> 1);                // round-half-up threshold
    int  nbase  = (1 << TB_MBITS);                     // 4
    int  nq     = nbase;
    long acc    = scale << TB_MBITS;                   // 4*scale
    for (int step = 1; step <= (1 << TB_MBITS); step++) {   // n = 5,6,7,8
        acc += scale;
        if (acc <= thresh) nq = nbase + step;
    }
    if (nq >= (1 << (TB_MBITS + 1))) { nq = (1 << TB_MBITS); e += 1; }
    if (nq <  (1 << TB_MBITS))         nq = (1 << TB_MBITS);
    e_out = e; nq_out = nq;
}

// build a PRE-DYADIC ±2^k gamma byte code (bit7=sign, bit5=nonzero, bits4..0=exp), like forge_fwd_wide.
static int32_t make_gamma_code(int sign, int nonzero, int exp) {
    int c = (exp & 0x1F);
    if (nonzero) c |= (1 << 5);
    if (sign)    c |= (1 << 7);
    return c;
}

int main() {
    int32_t *buf = (int32_t *)aligned_alloc(64, sizeof(int32_t) * ((BUFLEN + 15) / 16 * 16));
    srand(1234);

    // deployed-range x (|x| < 600 to keep sums/rescale inside int48), per-channel dyadic gamma
    // (exp 0..4 -> gain 1..16, ~30% zeros, random sign) and additive beta.
    for (int i = 0; i < TB_N * TB_D; i++) buf[O_X + i] = (rand() % 1201) - 600;
    for (int d = 0; d < TB_D; d++) {
        int nz   = (rand() % 100) < 70 ? 1 : 0;        // ~70% nonzero
        int sgn  = rand() & 1;
        int exp  = rand() % 5;                          // |gamma| = 2^0..2^4
        buf[O_GAMMA + d] = make_gamma_code(sgn, nz, exp);
        buf[O_BETA  + d] = (rand() % 201) - 100;        // beta in [-100,100]
    }
    for (int i = 0; i < TB_N * TB_D; i++) buf[O_Y + i] = 0;
    for (int s = 0; s < TB_N * 2; s++)    buf[O_STATS + s] = 0;

    forge_layernorm(buf);                               // run the kernel

    // ---- int64 reference of the SAME per-token fine-dyadic LayerNorm ----
    long max_y = 0, max_e = 0, max_nq = 0;
    int bad = 0;
    for (int n = 0; n < TB_N; n++) {
        // mean
        long sum = 0;
        for (int d = 0; d < TB_D; d++) sum += (long)(int32_t)buf[O_X + n*TB_D + d];
        long mean = sum >> TB_LOG2_D;
        // c, meanabs
        long c[TB_D]; long asum = 0;
        for (int d = 0; d < TB_D; d++) {
            long cv = (long)(int32_t)buf[O_X + n*TB_D + d] - mean;
            c[d] = sat16(cv);
            asum += cv < 0 ? -cv : cv;
        }
        long meanabs = asum >> TB_LOG2_D;
        int e, nq; ref_fdyad(meanabs, e, nq);
        int nqm4 = nq - (1 << TB_MBITS);

        // check saved stats (e, nq) — the backward's OFF_RSC contract
        long got_e  = (int32_t)buf[O_STATS + 2*n];
        long got_nq = (int32_t)buf[O_STATS + 2*n + 1];
        long de = labs(got_e - (long)e);   if (de > max_e)  max_e  = de;
        long dq = labs(got_nq - (long)nq); if (dq > max_nq) max_nq = dq;
        if (de != 0 && bad < 20) { printf("  e[%d] ref=%d got=%ld\n", n, e, got_e); bad++; }
        if (dq != 0 && bad < 20) { printf("  nq[%d] ref=%d got=%ld\n", n, nq, got_nq); bad++; }

        // y
        for (int d = 0; d < TB_D; d++) {
            long cv = c[d];
            long cn = cv << TB_MBITS;
            if (nqm4 >= 1) cn += cv;
            if (nqm4 >= 2) cn += cv;
            if (nqm4 >= 3) cn += cv;
            long r = cn >> TB_MBITS;
            if (e >= 0) r = r << e; else r = r >> (-e);
            // decode gamma
            int code = (int32_t)buf[O_GAMMA + d];
            int gs  = (code >> 7) & 1;
            int gnz = (code >> 5) & 1;
            int ge  = code & 0x1F;
            long g = gnz ? (r << ge) : 0;
            if (gnz && gs) g = -g;
            long yv = sat16(g + (long)(int32_t)buf[O_BETA + d]);
            long got = (int32_t)buf[O_Y + n*TB_D + d];
            long dy = labs(got - yv); if (dy > max_y) max_y = dy;
            if (dy != 0 && bad < 40) { printf("  y[%d][%d] ref=%ld got=%ld (c=%ld nq=%d e=%d)\n",
                                              n, d, yv, got, cv, nq, e); bad++; }
        }
    }

    printf("forge_layernorm CSIM  (D=%d N=%d MBITS=%d)\n", TB_D, TB_N, TB_MBITS);
    printf("  max|y_kernel - ref|     = %ld\n", max_y);
    printf("  max|scale_e_kernel-ref| = %ld\n", max_e);
    printf("  max|nq_kernel - ref|    = %ld\n", max_nq);
    int ok = (max_y == 0 && max_e == 0 && max_nq == 0);
    printf("  -> %s\n", ok ? "BIT-EXACT (kernel C++ == int64 reference)" : "MISMATCH");
    free(buf);
    return ok ? 0 : 1;
}
