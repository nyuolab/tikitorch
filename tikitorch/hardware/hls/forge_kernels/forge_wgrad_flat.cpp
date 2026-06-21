// forge_wgrad_flat.cpp — THROUGHPUT-optimized operand-quantized weight-grad GEMM (training rig).
//
// Same math + SAME buffer layout as forge_wgrad_probe (the PC set_mf_grads recipe):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])     with q(v) = sign(v) * 2^round(log2|v|)
// Because BOTH operands are dyadic (±2^k), each product is ±2^(eo+ei): a sign-xor + exponent-ADD
// + a barrel SHIFT ((acc_t)1<<e) + a signed ACCUMULATE. NO multiply, NO DSP multiplier.
//
// WHY THIS VERSION (strategy FLATTEN+PIPELINE):
// forge_wgrad_probe CLOSED timing (2.92 ns, 49% LUT-of-SLR, DSP=0) but ran at 2.66 ms / 665k cycles
// because its OTILE(96) x ITILE(12) tile-traversal loops were NOT pipelined: every one of the 1152
// tiles paid (zero-init + NRED pipeline fill/drain + writeback) latency SERIALLY. The N=16 reduction
// was the only II=1 loop and it was dwarfed by per-tile fixed overhead — no tile-to-tile overlap.
//
// FIX: hold the FULL gW row (all PR_I columns) for a block of PR_OT output rows in an on-chip
// register accumulator tile, partitioned COMPLETE so all PR_OT*PR_I lanes update in parallel. The
// reduction over the microbatch N is ONE perfectly-pipelined II=1 loop: a new PR_OT*PR_I-wide MAC
// vector issues every cycle, with NO per-tile bubble. Accumulator-init happens ONCE per row-block
// (before the inner pipeline); writeback is a separate AXI-widened burst AFTER it. The 1152 fill/
// drain bubbles of the baseline collapse to PR_NOB (= PR_O/PR_OT) tiny block boundaries.
//
// Lane count = PR_OT * PR_I. With PR_OT=1 that is 768 lanes; compute roofline = O*I*N/lanes =
// 768*768*16/768 = 12,288 cycles. Writeback (O*I = 590k int32) AXI-widened to 512-bit (16 int32/beat)
// is the DDR floor ~36,864 beats; it overlaps the next block's compute across the row-block loop.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_flat.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      1       // output rows held on-chip per block (full PR_I cols each)
#define PR_NOB     (PR_O / PR_OT)   // 768 output-row blocks

typedef ap_int<16>  data_t;    // gO / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (drop-in, identical to forge_wgrad_probe):
//   [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO = 0;
static const int OFF_X  = OFF_GO + PR_N * PR_O;
static const int OFF_GW = OFF_X  + PR_N * PR_I;

// dyadic-quantize a value to (sign, exponent): round(log2|v|) via MSB + the next bit (round-to-nearest).
// (copied verbatim from forge_wgrad_probe — the proven priority-encoder.)
static inline void dyad(data_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<16> a = (v < 0) ? (ap_uint<16>)(-v) : (ap_uint<16>)v;
    sgn = (v < 0);
    nz = (a != 0);
    exp_t msb = 0;
PRIO: for (int b = 15; b >= 1; b--) {           // priority encoder: highest set bit
#pragma HLS UNROLL
        if (a[b]) { msb = b; break; }
    }
    // round-to-nearest power of two: bump if the bit below the MSB is set
    if (msb >= 1 && a[msb - 1]) msb = msb + 1;
    e = msb;
}

static inline void burst_load(int32_t *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)buf[off + i];
    }
}

void forge_wgrad_flat(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // ---- Stage operands on-chip, then prequantize their dyadic (sign,exp,nz) ONCE. -----------
    // x is the per-cycle broadcast operand: partition it COMPLETE on the I axis so every one of the
    // PR_I lanes can read its column in the same cycle (the II=1 enabler for the inner pipeline).
    static data_t gO[PR_N][PR_O];
    static data_t xx[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=xx complete dim=2
    burst_load(buffer, OFF_GO, &gO[0][0], PR_N * PR_O);
    burst_load(buffer, OFF_X,  &xx[0][0], PR_N * PR_I);

    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
    // x fields fully partitioned on I (all lanes read in parallel each cycle).
#pragma HLS ARRAY_PARTITION variable=xE complete dim=2
#pragma HLS ARRAY_PARTITION variable=xS complete dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ complete dim=2

QO: for (int n = 0; n < PR_N; n++)
        for (int o = 0; o < PR_O; o++) {
#pragma HLS PIPELINE II=1
            dyad(gO[n][o], goS[n][o], goE[n][o], goNZ[n][o]);
        }
QX: for (int n = 0; n < PR_N; n++)
        for (int i = 0; i < PR_I; i++) {
#pragma HLS PIPELINE II=1
            dyad(xx[n][i], xS[n][i], xE[n][i], xNZ[n][i]);
        }

    // ---- FLATTEN+PIPELINE GEMM: the ENTIRE (output-row x N) traversal is ONE II=1 pipeline. -----
    // FIX (throughput): the prior per-OBLK structure paid, for EACH of the 768 row blocks, an NRED
    // pipeline fill/drain PLUS a 768-cycle writeback that did NOT overlap compute (no DATAFLOW; the WB
    // loop is a sequential sibling of NRED inside OBLK, so HLS runs INIT->NRED->WB serially per block).
    // That is ~768 blocks * (~16 reduction + ~768 writeback) ~= 600k cycles — barely better than the
    // 665k baseline; the un-overlapped per-block writeback re-creates exactly the baseline's tile tax.
    //
    // The cure: flatten OBLK x N into ONE PIPELINE II=1 loop over PR_O*PR_N trips. A single PR_I-wide
    // accumulator row `acc` is SEEDED at n==0 (zero-init folded into the first add) and committed to an
    // on-chip gW staging array at n==N-1; the pipeline NEVER drains between row blocks, so the 768
    // fill/drain bubbles collapse to ONE. Writeback is a SEPARATE AXI-widened (512-bit) burst over the
    // whole O*I result AFTER the loop — one streaming region, not 768 short non-overlapped bursts.
    // Lanes = PR_I = 768; compute floor = O*I*N/lanes = 768*768*16/768 = 12,288 trips (+ fill + burst).

    // on-chip grad staging (committed row-by-row from the hot loop, then one wide burst to DDR).
    // Partition COMPLETE on the column axis so the n==N-1 row commit writes all PR_I cols in one
    // cycle (one write port per column) without colliding with the II=1 pipeline.
    static acc_t gW[PR_O][PR_I];
#pragma HLS ARRAY_PARTITION variable=gW complete dim=2

    acc_t acc[PR_I];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

FLAT: for (int t = 0; t < PR_O * PR_N; t++) {
#pragma HLS PIPELINE II=1
        int ob = t / PR_N;     // current output row (0..PR_O-1); bounded -> small counter in HW
        int n  = t % PR_N;     // current microbatch step (0..PR_N-1)

        exp_t eo  = goE[n][ob];
        bool  so  = goS[n][ob];
        bool  nzo = goNZ[n][ob];

    COL: for (int c = 0; c < PR_I; c++) {
#pragma HLS UNROLL
            // dyadic MAC: product = ±2^(eo+ei) — barrel shift + signed accumulate, NO multiply
            exp_t ei = xE[n][c];
            ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;
            acc_t term = (acc_t)1 << e;
            bool neg = so ^ xS[n][c];
            bool nz  = nzo & xNZ[n][c];
            acc_t add = nz ? (neg ? (acc_t)(-term) : term) : (acc_t)0;
            // SEED at n==0 (folds the zero-init into the first add) else accumulate
            acc[c] = (n == 0) ? add : (acc_t)(acc[c] + add);
        }

        // commit the finished row to on-chip staging at the last reduction step
        if (n == PR_N - 1) {
        EM: for (int c = 0; c < PR_I; c++) {
#pragma HLS UNROLL
                gW[ob][c] = acc[c];
            }
        }
    }

    // single AXI-widenable (512-bit) burst writeback of the whole O*I grad — one pipelined region
WB: for (int o = 0; o < PR_O; o++)
    WBC: for (int c = 0; c < PR_I; c++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_GW + o * PR_I + c] = (int32_t)gW[o][c];
        }
}
