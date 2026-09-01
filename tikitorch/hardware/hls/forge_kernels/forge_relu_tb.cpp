// forge_relu_tb.cpp — Vitis HLS C-simulation testbench for the mult-free ReLU fwd/bwd kernel.
// Runs the ACTUAL kernel C++ (both modes) against an int64 golden reference of the SAME
// max(x,0)/mask-gate math, and asserts max|kernel - ref| == 0 (BIT-EXACT) over every output region.
// Deployed-range random operands (|x|<600, |g|<1200) so no value escapes int32. Clone of
// forge_pc_wgrad_tb.cpp. Build: csim adds this as the testbench, forge_relu.cpp as the design.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ap_int.h>

#define TB_N    16
#define TB_FFN  512
#define TB_LEN  (TB_N * TB_FFN)          // = 8192 (== RL_LEN)

static const int OFF_MODE = 0;
static const int OFF_X    = 16;
static const int OFF_GO   = OFF_X   + TB_LEN;
static const int OFF_MASK = OFF_GO  + TB_LEN;
static const int OFF_Y    = OFF_MASK + TB_LEN;
static const int OFF_GX   = OFF_Y   + TB_LEN;
static const int BUFLEN   = OFF_GX  + TB_LEN;   // % 16 == 0 (16 header + 5*8192)

void forge_relu(int32_t *buffer, ap_uint<512> *io_wide);   // design under test (int32 + 512b view)

int main() {
    // 64-byte aligned so the int32 view reinterprets as ap_uint<512> words (BUFLEN % 16 == 0).
    int32_t *buf = (int32_t *)aligned_alloc(64, sizeof(int32_t) * BUFLEN);
    srand(1234);

    // deployed-scaling random operands; include exact-zero x to exercise the x==0 boundary.
    for (int i = 0; i < TB_LEN; i++) {
        int xv = (rand() % 1201) - 600;          // |x| <= 600
        if ((rand() % 20) == 0) xv = 0;          // ~5% exact zeros (boundary: must be INACTIVE)
        buf[OFF_X + i]  = xv;
        buf[OFF_GO + i] = (rand() % 2401) - 1200; // |g| <= 1200 (upstream grad range)
    }
    for (int i = 0; i < TB_LEN; i++) { buf[OFF_MASK + i] = 0; buf[OFF_Y + i] = 0; buf[OFF_GX + i] = 0; }

    // ---- run FORWARD ----
    buf[OFF_MODE] = 0;
    forge_relu(buf, (ap_uint<512> *)buf);

    long max_y = 0, max_m = 0;
    int  bad = 0;
    for (int i = 0; i < TB_LEN; i++) {
        long x        = buf[OFF_X + i];
        long ref_y    = x > 0 ? x : 0;           // max(x,0); x==0 -> 0
        long ref_mask = x > 0 ? 1 : 0;           // (x>0); x==0 -> INACTIVE
        long got_y    = (int32_t)buf[OFF_Y + i];
        long got_mask = (int32_t)buf[OFF_MASK + i];
        long dy = labs(got_y - ref_y), dm = labs(got_mask - ref_mask);
        if (dy > max_y) max_y = dy;
        if (dm > max_m) max_m = dm;
        if ((dy != 0 || dm != 0) && bad < 20) {
            printf("  FWD[%d] x=%ld  y ref=%ld got=%ld  mask ref=%ld got=%ld\n",
                   i, x, ref_y, got_y, ref_mask, got_mask); bad++;
        }
    }

    // ---- run BACKWARD (reuses the mask the forward just wrote) ----
    buf[OFF_MODE] = 1;
    forge_relu(buf, (ap_uint<512> *)buf);

    long max_gx = 0;
    for (int i = 0; i < TB_LEN; i++) {
        long mask    = (int32_t)buf[OFF_MASK + i];
        long go      = buf[OFF_GO + i];
        long ref_gx  = mask ? go : 0;            // gate: pass gO where active, 0 where clamped
        long got_gx  = (int32_t)buf[OFF_GX + i];
        long dg = labs(got_gx - ref_gx);
        if (dg > max_gx) max_gx = dg;
        if (dg != 0 && bad < 40) {
            printf("  BWD[%d] mask=%ld gO=%ld  gx ref=%ld got=%ld\n", i, mask, go, ref_gx, got_gx); bad++;
        }
    }

    printf("forge_relu CSIM  (N=%d FFN=%d LEN=%d)\n", TB_N, TB_FFN, TB_LEN);
    printf("  max|y_kernel    - ref| = %ld\n", max_y);
    printf("  max|mask_kernel - ref| = %ld\n", max_m);
    printf("  max|gx_kernel   - ref| = %ld\n", max_gx);
    int ok = (max_y == 0 && max_m == 0 && max_gx == 0);
    printf("  -> %s\n", ok ? "BIT-EXACT (kernel C++ == int64 reference)" : "MISMATCH");
    free(buf);
    return ok ? 0 : 1;
}
