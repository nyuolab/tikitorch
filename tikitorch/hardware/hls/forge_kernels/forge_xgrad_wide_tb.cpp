// forge_xgrad_wide_tb.cpp — Vitis HLS C-simulation testbench for the input/activation-grad GEMM kernel.
// Runs the ACTUAL kernel C++ (not a Python mirror) against an INT64 golden reference of the SAME
//   gx[n][i] = sum_o gO[n][o] * q(W[o][i])   contraction, with wdecode()/shift math BIT-IDENTICAL to the
// kernel, then asserts max|kernel - ref| == 0 (BIT-EXACT) over the gx output region. Catches fixed-width
// truncation/overflow the float CPU mirror cannot see, and separately checks the worst-case acc bound.
// Deployed-range random operands (|gO| <= 1200 so int48 never saturates over the o=768 contraction).
// Clone of forge_pc_wgrad_tb.cpp / forge_relu_tb.cpp.
//
// Buffer layout (IDENTICAL to forge_xgrad_wide):  [ gO (N*O) | Wcode (O*I) | gx (N*I) ]
//   gO     : full int16 gradient (NOT quantized)
//   Wcode  : one int8 (sign,nz,exp) code per int32 lane (the pre-dyadic weight code)
//   gx     : the kernel's wide-written output region
// The single int32 allocation is passed as `buffer` (int32 view) AND as the two ap_uint<512> wide views
// (w_wide reads the Wcode region, gx_wide writes the gx region) — all three alias the same memory.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ap_int.h>

#define TB_O   768     // == XG_O  (contraction, rows of W)
#define TB_I   768     // == XG_I  (gx output feature, cols of W)
#define TB_N   16      // == XG_N  (microbatch)

static const int OFF_GO = 0;
static const int OFF_WC = OFF_GO + TB_N * TB_O;        // = 12288  (divisible by 16)
static const int OFF_GX = OFF_WC + TB_O * TB_I;        // = 602112 (divisible by 16)
static const int BUFLEN = OFF_GX + TB_N * TB_I;        // % 16 == 0

void forge_xgrad_wide(int32_t *buffer, ap_uint<512> *w_wide, ap_uint<512> *gx_wide);  // design under test

// ---- golden reference: decode a pre-dyadic W code EXACTLY like the kernel's wdecode(). ----
// returns the signed power-of-two value of the weight as an int64 (0 if the nonzero bit is clear).
static long ref_wval(int32_t code) {
    unsigned c   = (unsigned)(code & 0xFF);
    int sgn      = (c >> 7) & 1;                  // bit7 = sign
    int nz       = (c >> 5) & 1;                  // bit5 = nonzero flag
    int e        = (int)(c & 0x1F);               // bits4..0 = exponent
    if (!nz) return 0;
    long m = 1L << e;                             // |W| = 2^e
    return sgn ? -m : m;
}

// build a random pre-dyadic W code with the kernel's bit layout (sign,nz,exp), exponent kept modest so
// gO<<e stays inside the int48 accumulator over the o=768 contraction.
static int32_t rand_wcode() {
    int nz  = (rand() % 100) < 80 ? 1 : 0;        // ~80% of weights nonzero
    int sgn = rand() & 1;
    int e   = rand() % 12;                         // exponent 0..11 (|W| <= 2^11); deployed dyadic range
    int32_t c = 0;
    if (sgn) c |= (1 << 7);
    if (nz)  c |= (1 << 5);
    c |= (e & 0x1F);
    return c;
}

int main() {
    // 64-byte aligned so the int32 view reinterprets as ap_uint<512> words (BUFLEN % 16 == 0).
    int32_t *buf = (int32_t *)aligned_alloc(64, sizeof(int32_t) * BUFLEN);
    srand(1234);

    // gO: FULL int16 gradient, deployed range |gO| <= 1200 (NOT quantized). Include exact zeros.
    for (int i = 0; i < TB_N * TB_O; i++) {
        int g = (rand() % 2401) - 1200;
        if ((rand() % 20) == 0) g = 0;            // ~5% exact-zero grads (boundary)
        buf[OFF_GO + i] = g;
    }
    // Wcode: pre-dyadic ±2^e codes (one per int32 lane).
    for (int i = 0; i < TB_O * TB_I; i++) buf[OFF_WC + i] = rand_wcode();
    // clear the gx output region.
    for (int i = 0; i < TB_N * TB_I; i++) buf[OFF_GX + i] = 0;

    // run the kernel (int32 view + the two 512-bit wide views of the SAME allocation).
    forge_xgrad_wide(buf, (ap_uint<512> *)buf, (ap_uint<512> *)buf);

    // ---- int64 golden reference of the SAME contraction, compared to the kernel's int32 gx output. ----
    //   gx[n][i] = sum_o gO[n][o] * ref_wval(W[o][i])     (full-int gO shifted by the dyadic weight)
    long max_gx = 0, worst_gx = 0;
    int  bad = 0;
    for (int n = 0; n < TB_N && bad < 40; n++) {
        for (int i = 0; i < TB_I && bad < 40; i++) {
            long acc = 0;
            for (int o = 0; o < TB_O; o++) {
                long g  = buf[OFF_GO + n * TB_O + o];          // full int16 grad
                long wv = ref_wval(buf[OFF_WC + o * TB_I + i]); // signed power-of-two weight (or 0)
                acc += g * wv;                                  // == g << e with W's sign folded in (golden)
            }
            if (labs(acc) > worst_gx) worst_gx = labs(acc);
            long got = (int32_t)buf[OFF_GX + n * TB_I + i];
            long d = labs(got - acc);
            if (d > max_gx) max_gx = d;
            if (d != 0 && bad < 40) { printf("  gx[%d][%d] ref=%ld got=%ld\n", n, i, acc, got); bad++; }
        }
    }

    printf("forge_xgrad_wide CSIM  (O=%d I=%d N=%d)\n", TB_O, TB_I, TB_N);
    printf("  max|gx_kernel - ref| = %ld   (peak |gx|=%ld)\n", max_gx, worst_gx);
    printf("  int48 headroom: peak value %s 2^47 (%lld)\n",
           (worst_gx < (1LL << 47)) ? "<" : ">=", (long long)(1LL << 47));
    int ok = (max_gx == 0);
    printf("  -> %s\n", ok ? "BIT-EXACT (kernel C++ == int64 reference)" : "MISMATCH");
    free(buf);
    return ok ? 0 : 1;
}
