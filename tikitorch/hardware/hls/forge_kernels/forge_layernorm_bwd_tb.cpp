// forge_layernorm_bwd_tb.cpp — Vitis HLS C-simulation testbench for the dyadic LayerNorm backward.
// Runs the ACTUAL kernel C++ against an INT64 reference of the SAME backward (STE gamma pass-through
// + fine-dyadic RESCALE STE (invq, the _fine_dyadic_rescale 1/scale) + mean-subtract bwd + dyadic
// gamma-grad + beta-grad), so it catches any fixed-width truncation the float CPU mirror cannot see.
// Operands are DEPLOYED-RANGE random (|gY| < 1200; per-token invq = (nq/4)*2^e with nq in {4..7},
// e in {-3..0}) so every accumulator stays well inside int32 -> the int32 buffer comparison is
// meaningful. Asserts max|kernel-ref| == 0 (BIT-EXACT) over gx, g_gamma, g_beta. Clone of
// forge_pc_wgrad_tb.cpp.
//
// The saved s[d] and c_hat[n][d] are PRE-DYADIC byte codes (sign bit7 / nonzero bit5 / exp bits4..0),
// exactly the contract forge_fwd_wide.cpp::wdecode decodes. The golden builds those codes from random
// integers with the SAME dyad() (priority-encode MSB, round-to-nearest by bumping on the bit below the
// MSB, sign separate, zero-guarded) the forward kernel would have used -> the decode is bit-identical.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ap_int.h>

#define TB_D       128
#define TB_LOG2_D  7
#define TB_N       16
#define TB_MBITS   2                                  // fine-dyadic mantissa bits (charlm _NORM_MBITS=2)
static const int O_GY    = 0;
static const int O_S     = O_GY    + TB_N * TB_D;
static const int O_CHAT  = O_S     + TB_D;
static const int O_RSC   = O_CHAT  + TB_N * TB_D;      // per-token (e,nq) rescale factor
static const int O_GX    = O_RSC   + TB_N * 2;
static const int O_GGAM  = O_GX    + TB_N * TB_D;
static const int O_GBETA = O_GGAM  + TB_D;
static const int BUFLEN  = O_GBETA + TB_D;

void forge_layernorm_bwd(int32_t *buffer);          // the design under test

// reference dyad: returns the (sign,nz,exp) of sign*2^e with round-to-nearest (bump on the bit
// below the MSB) — IDENTICAL to forge_fwd_wide.cpp::dyad. Then ENCODE into the byte code that the
// kernel's wdecode() will decode (bit7=sign, bit5=nonzero, bits4..0=exp).
static int32_t encode_dyad(long v) {
    long a = v < 0 ? -v : v;
    int sgn = (v < 0) ? 1 : 0;
    int nz  = (a != 0) ? 1 : 0;
    int msb = 0;
    for (int b = 15; b >= 1; b--) { if ((a >> b) & 1) { msb = b; break; } }
    if (msb >= 1 && ((a >> (msb - 1)) & 1)) msb++;          // round-to-nearest pow2
    int e = nz ? msb : 0;
    if (e > 31) e = 31;                                      // bits4..0 hold 0..31
    return (sgn << 7) | (nz << 5) | (e & 0x1F);
}

// reference decode (mirror of the kernel's wdecode) -> the value 2^e folded by sign, or 0.
static long decode_term(int32_t code, long g) {
    int sgn = (code >> 7) & 1;
    int nz  = (code >> 5) & 1;
    int e   = code & 0x1F;
    if (!nz) return 0;
    long t = g << e;
    return sgn ? -t : t;
}

// reference fine-dyadic rescale (mirror of the kernel's apply_invq) -> v * (nq/4) * 2^e, SHIFT-ADD only,
// BIT-IDENTICAL to the kernel and to forge_layernorm.cpp's forward rescale (floor right-shifts).
static long apply_invq_ref(long v, int e, int nq) {
    int nqm4 = nq - (1 << TB_MBITS);
    long cn = v << TB_MBITS;
    if (nqm4 >= 1) cn += v;
    if (nqm4 >= 2) cn += v;
    if (nqm4 >= 3) cn += v;
    long r = cn >> TB_MBITS;
    if (e >= 0) r = r << e;
    else        r = r >> (-e);
    return r;
}

int main() {
    // 64-byte aligned so the int32 view matches the kernel's 512b widening (BUFLEN % 16 == 0).
    int32_t *buf = (int32_t *)aligned_alloc(64, sizeof(int32_t) * BUFLEN);
    srand(4321);

    // DEPLOYED-RANGE operands. gY kept full-precision (|.| < 1200). The saved s/c_hat are stored as
    // BYTE CODES: build them from random integers via encode_dyad (what the forward kernel emits).
    for (int i = 0; i < TB_N * TB_D; i++) {
        buf[O_GY + i]   = (rand() % 2401) - 1200;                  // |gY| <= 1200
        buf[O_CHAT + i] = encode_dyad((rand() % 1201) - 600);      // saved c_hat dyadic code
    }
    for (int d = 0; d < TB_D; d++) {
        // gamma scale s[d] in a SMALL ±2^k range (|exp| small so g = gY<<s_exp stays in int32).
        long sv = (rand() % 9) - 4;                                // -4..4  -> exp 0..2 after dyad
        if (sv == 0) sv = 1;                                        // keep most channels active
        buf[O_S + d] = encode_dyad(sv);
    }
    // Per-token fine-dyadic rescale (e, nq): DEPLOYED range. e = floor(log2(1/scale)) <= 0 for scale>=1
    // (a RIGHT shift), nq the 2-bit mantissa in {4..7}. Mirror what forge_layernorm.cpp emits.
    for (int n = 0; n < TB_N; n++) {
        buf[O_RSC + 2 * n]     = -(rand() % 4);                     // e in {-3..0} (right shift)
        buf[O_RSC + 2 * n + 1] = 4 + (rand() % 4);                  // nq in {4,5,6,7}
    }
    for (int i = 0; i < TB_N * TB_D; i++) buf[O_GX + i] = 0;
    for (int d = 0; d < TB_D; d++) { buf[O_GGAM + d] = 0; buf[O_GBETA + d] = 0; }

    forge_layernorm_bwd(buf);                                      // run the kernel

    // ---- int64 reference of the SAME backward ----
    long max_gx = 0, max_gg = 0, max_gb = 0;
    long worst_gx = 0, worst_gg = 0, worst_gb = 0;
    int bad = 0;

    // (1a)+(1b)+(2): g_chat = gY << s_exp (sign-folded); g_c = g_chat * invq_n (fine-dyadic rescale STE,
    // shift-add); gx[n][d] = g_c[d] - (sum_d g_c >> LOG2_D).
    for (int n = 0; n < TB_N; n++) {
        int e_n  = (int)buf[O_RSC + 2 * n];
        int nq_n = (int)buf[O_RSC + 2 * n + 1];
        long g[TB_D];
        long gsum = 0;
        for (int d = 0; d < TB_D; d++) {
            long gchat = decode_term(buf[O_S + d], (long)buf[O_GY + n * TB_D + d]);
            g[d] = apply_invq_ref(gchat, e_n, nq_n);                // g_c = g_chat * invq
            gsum += g[d];
        }
        long mean = gsum >> TB_LOG2_D;
        for (int d = 0; d < TB_D; d++) {
            long ref = g[d] - mean;
            if (labs(ref) > worst_gx) worst_gx = labs(ref);
            long got = (int32_t)buf[O_GX + n * TB_D + d];
            long dd = labs(got - ref);
            if (dd > max_gx) max_gx = dd;
            if (dd != 0 && bad < 20) { printf("  gx[%d][%d] ref=%ld got=%ld\n", n, d, ref, got); bad++; }
        }
    }

    // (3) g_gamma[d] = sum_n gY[n][d] << chat_exp[n][d] (sign-folded);  (4) g_beta[d] = sum_n gY[n][d]
    for (int d = 0; d < TB_D; d++) {
        long accg = 0, accb = 0;
        for (int n = 0; n < TB_N; n++) {
            long gv = (long)buf[O_GY + n * TB_D + d];
            accg += decode_term(buf[O_CHAT + n * TB_D + d], gv);
            accb += gv;
        }
        if (labs(accg) > worst_gg) worst_gg = labs(accg);
        if (labs(accb) > worst_gb) worst_gb = labs(accb);
        long gotg = (int32_t)buf[O_GGAM + d];
        long gotb = (int32_t)buf[O_GBETA + d];
        long dg = labs(gotg - accg), db = labs(gotb - accb);
        if (dg > max_gg) max_gg = dg;
        if (db > max_gb) max_gb = db;
        if (dg != 0 && bad < 40) { printf("  g_gamma[%d] ref=%ld got=%ld\n", d, accg, gotg); bad++; }
        if (db != 0 && bad < 60) { printf("  g_beta[%d]  ref=%ld got=%ld\n", d, accb, gotb); bad++; }
    }

    printf("forge_layernorm_bwd CSIM  (D=%d N=%d)\n", TB_D, TB_N);
    printf("  max|gx_kernel - ref|      = %ld   (peak |gx|=%ld)\n", max_gx, worst_gx);
    printf("  max|g_gamma_kernel - ref| = %ld   (peak |g_gamma|=%ld)\n", max_gg, worst_gg);
    printf("  max|g_beta_kernel - ref|  = %ld   (peak |g_beta|=%ld)\n", max_gb, worst_gb);
    printf("  int32 headroom: peak values %s 2^31 (%ld)\n",
           (worst_gx < 2147483647L && worst_gg < 2147483647L && worst_gb < 2147483647L) ? "<" : ">=",
           2147483647L);
    int ok = (max_gx == 0 && max_gg == 0 && max_gb == 0);
    printf("  -> %s\n", ok ? "BIT-EXACT (kernel C++ == int64 reference)" : "MISMATCH");
    free(buf);
    return ok ? 0 : 1;
}
