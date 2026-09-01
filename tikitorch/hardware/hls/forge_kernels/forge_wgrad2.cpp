// forge_wgrad2.cpp — THROUGHPUT-OPTIMIZED operand-quantized weight-grad GEMM (training rig).
//
// Same math as forge_wgrad_probe (the PC set_mf_grads recipe):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])     with q(v) = sign(v) * 2^round(log2|v|)
// Because BOTH operands are dyadic (±2^k), each product is ±2^(eo+ei): a sign-xor + exponent-ADD
// + a barrel SHIFT ((acc_t)1<<e) + a signed ACCUMULATE. NO multiply, NO DSP multiplier.
//
// WHY THIS VERSION: forge_wgrad_probe CLOSED timing (2.92 ns, 49% LUT-of-SLR, DSP=0) but ran at
// 2.66 ms for the 768x768 / N16 grad because its OTILE x ITILE tile-traversal loops were NOT
// pipelined: every one of the 96*12 = 1152 tiles paid (zero-init + NRED pipeline-depth fill/drain
// + store) latency serially. The N-reduction was the only II=1 loop and it is tiny (N=16), so the
// per-tile fixed overhead dominated and there was no tile-to-tile overlap.
//
// FIX (throughput): the per-tile NRED reduction is restructured into a SINGLE PIPELINE II=1 loop
// whose iteration space is FLATTENED over (tile x N). One accumulator BANK per IT lane is reused
// across N; the bank is zeroed on n==0 and committed to the store FIFO on n==N-1. Because the loop
// pipeline never drains between tiles (flattened trip count = NUM_TILES*N), the IT-lane PE array
// stays continuously busy and the 1152 separate fill/drain bubbles collapse to ONE. Load and store
// are split into their own functions and overlapped with compute via pragma HLS DATAFLOW, so DDR
// traffic hides under the arithmetic. Same d768 / N16 sizing, same dyadic PE, timing target 4.0 ns.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad2.tcl.

#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= the unrolled PE lane count)

#define PR_NOT     (PR_O / PR_OT)          // 96  output-row tiles
#define PR_NIT     (PR_I / PR_IT)          // 12  output-col tiles
#define PR_TILES   (PR_NOT * PR_NIT)       // 1152 tiles total
#define PR_TILEN   (PR_TILES * PR_N)       // 18432 flattened (tile x N) trips

typedef ap_int<16>  data_t;    // gO / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (probe-local): [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO = 0;
static const int OFF_X  = OFF_GO + PR_N * PR_O;
static const int OFF_GW = OFF_X  + PR_N * PR_I;

// dyadic-quantize a value to (sign, exponent): round(log2|v|) via MSB + the next bit (round-to-nearest).
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

// ---- stage + dyadic-prequantize all operands once (load region) -------------------------
// Output: per-element (sign, exp, nz) for gO and x. Done once; reused across all tiles.
static void stage_and_quant(int32_t *buffer,
                            exp_t goE[PR_N][PR_O], bool goS[PR_N][PR_O], bool goNZ[PR_N][PR_O],
                            exp_t xE[PR_N][PR_I],  bool xS[PR_N][PR_I],  bool xNZ[PR_N][PR_I]) {
#pragma HLS INLINE off
    static data_t gO[PR_N][PR_O];
    static data_t xx[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=gO cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xx cyclic factor=PR_IT dim=2
    burst_load(buffer, OFF_GO, &gO[0][0], PR_N * PR_O);
    burst_load(buffer, OFF_X,  &xx[0][0], PR_N * PR_I);
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
}

// ---- compute: ONE flattened, fully-pipelined (tile x N) reduction loop ------------------
// The whole tile traversal is a single PIPELINE II=1 loop over PR_TILEN trips. An OT x IT
// accumulator BANK is held in regs (partitioned complete), zeroed at n==0 and pushed into the
// store FIFO at n==N-1. No per-tile pipeline drain: the IT-lane PE array stays busy across tiles.
static void compute(exp_t goE[PR_N][PR_O], bool goS[PR_N][PR_O], bool goNZ[PR_N][PR_O],
                    exp_t xE[PR_N][PR_I],  bool xS[PR_N][PR_I],  bool xNZ[PR_N][PR_I],
                    hls::stream<acc_t> &out) {
#pragma HLS INLINE off
    acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    int ot = 0;   // current output-row tile base (0,8,16,..,760)
    int it = 0;   // current output-col tile base (0,64,..,704)

FLAT: for (int t = 0; t < PR_TILEN; t++) {
#pragma HLS PIPELINE II=1
        int n = t % PR_N;                       // bounded; folds to a small counter in HW

        // zero the bank at the start of each tile
        if (n == 0) {
        Z0: for (int r = 0; r < PR_OT; r++)
                for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                    acc[r][c] = 0;
                }
        }

        // dyadic MAC across the OT x IT PE lanes for this (tile, n)
    MR: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
            exp_t eo = goE[n][ot + r]; bool so = goS[n][ot + r]; bool nzo = goNZ[n][ot + r];
        MC: for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                exp_t ei = xE[n][it + c];
                ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;
                acc_t term = (acc_t)1 << e;          // barrel shift, NO multiply
                bool neg = so ^ xS[n][it + c];
                bool nz  = nzo & xNZ[n][it + c];
                if (nz) acc[r][c] += neg ? (acc_t)(-term) : term;
            }
        }

        // commit the completed tile to the store FIFO at the last reduction step
        if (n == PR_N - 1) {
        EM: for (int r = 0; r < PR_OT; r++)
                for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                    out.write(acc[r][c]);
                }
            // advance to the next tile (row-major over IT then OT)
            it += PR_IT;
            if (it == PR_I) { it = 0; ot += PR_OT; }
        }
    }
}

// ---- store: drain the FIFO to DDR in tile order (matches compute emit order) ------------
static void store(int32_t *buffer, hls::stream<acc_t> &in) {
#pragma HLS INLINE off
SOT: for (int ot = 0; ot < PR_O; ot += PR_OT) {
SIT:    for (int it = 0; it < PR_I; it += PR_IT) {
        SR: for (int r = 0; r < PR_OT; r++)
            SC: for (int c = 0; c < PR_IT; c++) {
#pragma HLS PIPELINE II=1
                    buffer[OFF_GW + (ot + r) * PR_I + (it + c)] = (int32_t)in.read();
                }
        }
    }
}

void forge_wgrad2(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW

    // prequantized operand fields (live across the whole compute pass)
    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=goE cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goS cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xE cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xS cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ cyclic factor=PR_IT dim=2

    // store FIFO: holds completed grad tiles (OT*IT = 512 deep covers a couple tiles of slack)
    hls::stream<acc_t> gwfifo;
#pragma HLS STREAM variable=gwfifo depth=1024

    stage_and_quant(buffer, goE, goS, goNZ, xE, xS, xNZ);
    compute(goE, goS, goNZ, xE, xS, xNZ, gwfifo);
    store(buffer, gwfifo);
}
