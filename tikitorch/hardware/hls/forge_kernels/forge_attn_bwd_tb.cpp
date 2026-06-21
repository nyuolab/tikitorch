// forge_attn_bwd_tb.cpp — Vitis HLS C-simulation testbench for the attention-backward kernel.
// Runs the ACTUAL kernel C++ against an int64 reference whose every integer op (L1 fwd re-derive,
// exp2 ROM lookup, pow2-ceil normalize, the gV/gp/gs DSP multiplies, and the MULT-FREE sign(Q-K)
// subgradient) is BIT-IDENTICAL to the kernel. Because the whole path is integer fixed-point, ALL
// three grad outputs (gQ, gK, gV) are asserted BIT-EXACT (max|kernel-ref|==0) — both the DSP-multiply
// softmax/blend backward AND the mult-free subgradient. Operands are in the DEPLOYED range
// (|x|<600, |g|<1200) so the int48 accumulators don't saturate. Build: csim adds this as the
// testbench, forge_attn_bwd.cpp as the design.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ap_int.h>

#define AT_L   128
#define AT_DH  64
#define AT_DELTA_SHIFT 2
#define AT_LUT_N 256
#define AT_P_FRAC 16

static const int OFF_Q    = 0;
static const int OFF_K    = OFF_Q   + AT_L * AT_DH;
static const int OFF_V    = OFF_K   + AT_L * AT_DH;
static const int OFF_GO   = OFF_V   + AT_L * AT_DH;
static const int OFF_CFG  = OFF_GO  + AT_L * AT_DH;
static const int OFF_LUT  = OFF_CFG + 16;
static const int OFF_GQ   = OFF_LUT + AT_LUT_N;
static const int OFF_GK   = OFF_GQ  + AT_L * AT_DH;
static const int OFF_GV   = OFF_GK  + AT_L * AT_DH;
static const int BUFLEN   = OFF_GV  + AT_L * AT_DH;

static const long NEG_INF = -(1L << 40);   // == the kernel's NEG_INF (acc_t 48-bit sentinel)

void forge_attn_bwd(int32_t *buffer);      // the design under test

int main() {
    int32_t *buf = (int32_t *)aligned_alloc(64, sizeof(int32_t) * ((BUFLEN + 15) & ~15));
    srand(20260621);

    // DEPLOYED-RANGE random operands: |Q,K,V| < 600, |gOut| < 1200 (keeps int48 accumulators safe).
    for (int i = 0; i < AT_L * AT_DH; i++) {
        buf[OFF_Q  + i] = (rand() % 1201) - 600;
        buf[OFF_K  + i] = (rand() % 1201) - 600;
        buf[OFF_V  + i] = (rand() % 1201) - 600;
        buf[OFF_GO + i] = (rand() % 2401) - 1200;
    }
    for (int i = 0; i < 16; i++) buf[OFF_CFG + i] = 0;
    // exp2 ROM (Q16): lut[g] = round(2^(-g/8) * 65536), the host-staged decay table (matches forge_attn:
    // lut[0]=65536, monotone-decreasing). The scale (8 here) is arbitrary for the test — kernel & ref
    // read the SAME table, so bit-exactness holds for any monotone Q16 ROM.
    for (int g = 0; g < AT_LUT_N; g++) {
        double val = pow(2.0, -((double)g) / 8.0) * 65536.0;
        long q = (long)(val + 0.5);
        if (q < 1) q = 1;                       // never exactly 0 (matches a real exp2 ROM tail)
        buf[OFF_LUT + g] = (int32_t)q;
    }
    // zero the output regions
    for (int i = 0; i < AT_L * AT_DH; i++) { buf[OFF_GQ + i] = 0; buf[OFF_GK + i] = 0; buf[OFF_GV + i] = 0; }

    // -------------------- run the kernel --------------------
    forge_attn_bwd(buf);

    // -------------------- int64 BIT-IDENTICAL reference --------------------
    static long rgQ[AT_L][AT_DH];
    static long rgK[AT_L][AT_DH];
    static long rgV[AT_L][AT_DH];
    for (int a = 0; a < AT_L; a++)
        for (int d = 0; d < AT_DH; d++) { rgQ[a][d] = 0; rgK[a][d] = 0; rgV[a][d] = 0; }

    static long s[AT_L], p[AT_L], gp[AT_L], gs[AT_L];

    for (int i = 0; i < AT_L; i++) {
        // (1) L1 scores (causal), rowmax  — identical to the kernel
        long rowmax = NEG_INF;
        for (int j = 0; j < AT_L; j++) {
            if (j > i) { s[j] = NEG_INF; continue; }
            long l1 = 0;
            for (int d = 0; d < AT_DH; d++) {
                long diff = (long)buf[OFF_Q + i*AT_DH + d] - (long)buf[OFF_K + j*AT_DH + d];
                l1 += (diff < 0) ? -diff : diff;
            }
            long sc = -l1;
            s[j] = sc;
            if (sc > rowmax) rowmax = sc;
        }
        // (2) softmax via exp2 ROM (Q16) + pow2-ceil normalize SHIFT
        long rowsum = 0;
        for (int j = 0; j < AT_L; j++) {
            if (j > i) { p[j] = 0; continue; }
            long gap = (rowmax - s[j]) >> AT_DELTA_SHIFT;
            long idx = gap;
            if (idx < 0) idx = 0;
            if (idx > AT_LUT_N - 1) idx = AT_LUT_N - 1;
            long pv = (long)buf[OFF_LUT + idx];
            p[j] = pv;
            rowsum += pv;
        }
        int sh = 0;
        for (int b = 0; b < 16; b++) {
            if ((rowsum >> sh) > (1L << AT_P_FRAC)) sh++;
        }
        for (int j = 0; j < AT_L; j++) p[j] = p[j] >> sh;

        // (b) gp[i][j] = sum_d gOut[i][d]*V[j][d]  + partition-contrast weighted sum
        long expnum = 0;
        for (int j = 0; j <= i; j++) {
            long dot = 0;
            for (int d = 0; d < AT_DH; d++)
                dot += (long)buf[OFF_GO + i*AT_DH + d] * (long)buf[OFF_V + j*AT_DH + d];
            long gpj = (long)(int32_t)dot;       // (grad_t) truncation, matches kernel
            gp[j] = gpj;
            expnum += p[j] * gpj;
        }
        long ex = (long)(int32_t)(expnum >> AT_P_FRAC);

        // (c) gs[i][j] = ( p[i][j]*(gp[i][j]-ex) ) >> AT_P_FRAC
        for (int j = 0; j < AT_L; j++) {
            if (j > i) { gs[j] = 0; continue; }
            long centered = (long)(int32_t)(gp[j] - ex);
            long prod = p[j] * centered;
            gs[j] = (long)(int32_t)(prod >> AT_P_FRAC);
        }

        // (a) gV[j][d] += (p[i][j]*gOut[i][d]) >> AT_P_FRAC
        for (int j = 0; j <= i; j++) {
            long pj = p[j];
            for (int d = 0; d < AT_DH; d++) {
                long t = pj * (long)buf[OFF_GO + i*AT_DH + d];
                rgV[j][d] += (t >> AT_P_FRAC);
            }
        }

        // (d) MULT-FREE sign(Q-K) subgradient
        for (int j = 0; j <= i; j++) {
            long gsj = gs[j];
            for (int d = 0; d < AT_DH; d++) {
                long diff = (long)buf[OFF_Q + i*AT_DH + d] - (long)buf[OFF_K + j*AT_DH + d];
                if (diff > 0) { rgQ[i][d] -= gsj; rgK[j][d] += gsj; }
                else if (diff < 0) { rgQ[i][d] += gsj; rgK[j][d] -= gsj; }
            }
        }
    }

    // -------------------- compare (BIT-EXACT over all three grad outputs) --------------------
    long max_gq = 0, max_gk = 0, max_gv = 0;
    long peak_gq = 0, peak_gk = 0, peak_gv = 0;
    int bad = 0;
    for (int a = 0; a < AT_L; a++)
        for (int d = 0; d < AT_DH; d++) {
            long refq = (long)(int32_t)rgQ[a][d];   // int32 writeback truncation (kernel & ref alike)
            long refk = (long)(int32_t)rgK[a][d];
            long refv = (long)(int32_t)rgV[a][d];
            if (labs(rgQ[a][d]) > peak_gq) peak_gq = labs(rgQ[a][d]);
            if (labs(rgK[a][d]) > peak_gk) peak_gk = labs(rgK[a][d]);
            if (labs(rgV[a][d]) > peak_gv) peak_gv = labs(rgV[a][d]);
            long gq = (long)(int32_t)buf[OFF_GQ + a*AT_DH + d];
            long gk = (long)(int32_t)buf[OFF_GK + a*AT_DH + d];
            long gv = (long)(int32_t)buf[OFF_GV + a*AT_DH + d];
            long dq = labs(gq - refq), dk = labs(gk - refk), dv = labs(gv - refv);
            if (dq > max_gq) max_gq = dq;
            if (dk > max_gk) max_gk = dk;
            if (dv > max_gv) max_gv = dv;
            if ((dq || dk || dv) && bad < 20) {
                printf("  [%d][%d] gQ ref=%ld got=%ld | gK ref=%ld got=%ld | gV ref=%ld got=%ld\n",
                       a, d, refq, gq, refk, gk, refv, gv);
                bad++;
            }
        }

    printf("forge_attn_bwd CSIM  (L=%d DH=%d)\n", AT_L, AT_DH);
    printf("  max|gQ_kernel - ref| = %ld   (peak |gQ|=%ld)\n", max_gq, peak_gq);
    printf("  max|gK_kernel - ref| = %ld   (peak |gK|=%ld)\n", max_gk, peak_gk);
    printf("  max|gV_kernel - ref| = %ld   (peak |gV|=%ld)\n", max_gv, peak_gv);
    printf("  int48 headroom: peak grads %s 2^47 (%ld)\n",
           (peak_gq < (1L<<47) && peak_gk < (1L<<47) && peak_gv < (1L<<47)) ? "<" : ">=", (1L<<47));
    int ok = (max_gq == 0 && max_gk == 0 && max_gv == 0);
    printf("  -> %s\n", ok ? "BIT-EXACT (kernel C++ == int64 reference)" : "MISMATCH");
    free(buf);
    return ok ? 0 : 1;
}
