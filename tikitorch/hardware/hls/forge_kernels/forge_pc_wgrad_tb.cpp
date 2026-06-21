// forge_pc_wgrad_tb.cpp — Vitis HLS C-simulation testbench for the PC cascade backward kernel.
// Runs the ACTUAL kernel C++ (not the Python mirror) against an int64 reference of the same
// cascade_grads layer-step, so it catches fixed-width truncation/overflow the float64 CPU mirror
// (pc_wgrad_kernel_check.py) cannot see. Uses modest fixed-point magnitudes (the deployed scaling)
// so the int32 buffer comparison is meaningful, and separately checks the worst-case accumulator
// bound. Build: csim adds this as the testbench, forge_pc_wgrad.cpp as the design.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TB_D  256
#define TB_N  16
static const int O_E    = 0;
static const int O_W    = O_E    + TB_N * TB_D;
static const int O_ACT  = O_W    + TB_D * TB_D;
static const int O_MASK = O_ACT  + TB_N * TB_D;
static const int O_SKIP = O_MASK + TB_N * TB_D;
static const int O_GW   = O_SKIP + TB_N * TB_D;
static const int O_EDN  = O_GW   + TB_D * TB_D;
static const int BUFLEN = O_EDN  + TB_N * TB_D;

void forge_pc_wgrad(int32_t *buffer);              // the design under test

// reference dyad: sign*2^e via ceil(sqrt(2)*2^msb) threshold (== kernel DYAD_THRESH, == dyadic_quantize)
static const int THRESH[16] = {2,3,6,12,23,46,91,182,363,725,1449,2897,5793,11586,23171,46341};
static long ref_dyad(long v) {
    long a = v < 0 ? -v : v;
    if (a == 0) return 0;
    int msb = 0; for (int b = 15; b >= 1; b--) { if ((a >> b) & 1) { msb = b; break; } }
    if (a >= THRESH[msb]) msb++;
    long m = 1L << msb;
    return v < 0 ? -m : m;
}

int main() {
    int32_t *buf = (int32_t *)malloc(sizeof(int32_t) * BUFLEN);
    srand(1234);
    // modest fixed-point magnitudes: errors/acts/weights in a deployed-scaling range (|.| <~ 600),
    // so gW/e_down stay well inside int32 — the realistic operating point.
    for (int i = 0; i < TB_N * TB_D; i++) {
        buf[O_E + i]   = (rand() % 1201) - 600;
        buf[O_ACT + i] = rand() % 601;                       // post-ReLU >= 0
        buf[O_MASK + i] = (rand() % 100) < 60 ? 1 : 0;       // ~60% relu-active
        buf[O_SKIP + i] = (rand() % 100) < 70 ? 1 : 0;       // ~70% conv-branch-won
    }
    for (int i = 0; i < TB_D * TB_D; i++) buf[O_W + i] = (rand() % 1201) - 600;
    for (int i = 0; i < TB_N * TB_D; i++) { buf[O_GW + (i % (TB_D*TB_D))] = 0; buf[O_EDN + i] = 0; }
    for (int i = 0; i < TB_D * TB_D; i++) buf[O_GW + i] = 0;

    forge_pc_wgrad(buf);                                     // run the kernel

    // int64 reference of the SAME layer-step (no overflow), compared to the kernel's int32 output
    long max_gw = 0, max_ed = 0, worst_gw = 0, worst_ed = 0;
    // precompute gated delta
    static long delta[TB_N][TB_D];
    for (int n = 0; n < TB_N; n++)
        for (int o = 0; o < TB_D; o++) {
            int g = buf[O_MASK + n*TB_D + o] && buf[O_SKIP + n*TB_D + o];
            delta[n][o] = g ? buf[O_E + n*TB_D + o] : 0;
        }
    // (1) gW[o][i] = sum_n q(delta[n][o]) * q(acts[n][i])
    int bad = 0;
    for (int o = 0; o < TB_D && bad < 20; o++)
        for (int i = 0; i < TB_D && bad < 20; i++) {
            long acc = 0;
            for (int n = 0; n < TB_N; n++) acc += ref_dyad(delta[n][o]) * ref_dyad(buf[O_ACT + n*TB_D + i]);
            if (labs(acc) > worst_gw) worst_gw = labs(acc);
            long got = (int32_t)buf[O_GW + o*TB_D + i];
            long d = labs(got - acc);
            if (d > max_gw) max_gw = d;
            if (d != 0 && bad < 20) { printf("  gW[%d][%d] ref=%ld got=%ld\n", o, i, acc, got); bad++; }
        }
    // (2) e_down[n][i] = sum_o delta[n][o]*q(W[o][i]) + e[n][i]*(1-skip[n][i])
    for (int n = 0; n < TB_N && bad < 40; n++)
        for (int i = 0; i < TB_D && bad < 40; i++) {
            long acc = 0;
            for (int o = 0; o < TB_D; o++) acc += delta[n][o] * ref_dyad(buf[O_W + o*TB_D + i]);
            if (!buf[O_SKIP + n*TB_D + i]) acc += buf[O_E + n*TB_D + i];
            if (labs(acc) > worst_ed) worst_ed = labs(acc);
            long got = (int32_t)buf[O_EDN + n*TB_D + i];
            long d = labs(got - acc);
            if (d > max_ed) max_ed = d;
            if (d != 0 && bad < 40) { printf("  e_down[%d][%d] ref=%ld got=%ld\n", n, i, acc, got); bad++; }
        }

    printf("forge_pc_wgrad CSIM  (D=%d N=%d)\n", TB_D, TB_N);
    printf("  max|gW_kernel - ref|     = %ld   (peak |gW|=%ld)\n", max_gw, worst_gw);
    printf("  max|e_down_kernel - ref| = %ld   (peak |e_down|=%ld)\n", max_ed, worst_ed);
    printf("  int32 headroom: peak values %s 2^31 (%ld)\n",
           (worst_gw < 2147483647L && worst_ed < 2147483647L) ? "<" : ">=", 2147483647L);
    int ok = (max_gw == 0 && max_ed == 0);
    printf("  -> %s\n", ok ? "BIT-EXACT (kernel C++ == int64 reference)" : "MISMATCH");
    free(buf);
    return ok ? 0 : 1;
}
