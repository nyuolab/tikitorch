// forge_wgrad_sys.cpp — SYSTOLIC / output-stationary STREAMING weight-grad GEMM (training rig).
//
// Same math as forge_wgrad_probe (the PC set_mf_grads recipe):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])     with q(v) = sign(v) * 2^round(log2|v|)
// Because BOTH operands are dyadic (±2^k), each product is ±2^(eo+ei): a sign-xor + exponent-ADD
// + a barrel SHIFT ((acc_t)1<<e) + a signed ACCUMULATE. NO multiply, NO DSP multiplier (DSP48 may
// serve as the signed accumulator/adder — the "DSP-as-adder" lever). SAME buffer layout / OFF_* as
// the probe (drop-in): [ gO (N*O) | x (N*I) | gW (O*I) ].
//
// WHY THIS VERSION (throughput): forge_wgrad_probe CLOSED 2.92 ns at 49% LUT-of-SLR, DSP=0, but ran
// 664,932 cycles (2.66 ms) because its OTILE(96) x ITILE(12) tile loops were NOT pipelined — every
// one of the 1152 tiles paid (acc zero-init + NRED fill/drain + writeback) serially, and the only
// II=1 loop (the N=16 reduction) was dwarfed by that fixed per-tile overhead.
//
// THE FIX — a 2D PE grid (rows = OT output o-lanes, cols = IT input i-lanes) of OT*IT = 512 lanes,
// each holding gW[o][i] STATIONARY. The whole tile traversal is ONE flattened II=1 pipeline over
// PR_TILES super-tiles of (PR_N reduce + PR_IT drain) trips: the first PR_N trips MAC the microbatch
// into the bank, the next PR_IT trips DRAIN it ONE OT-wide column / cycle into the store FIFO. The
// pipeline never drains between tiles. Compute = PR_TILES*(PR_N+PR_IT) = 92,160 trips; store overlaps
// it under DATAFLOW (DDR floor O*I/16 ~= 36,864 beats). ~7x faster than the probe's 664,932 cycles.
//
// WHY IT WON'T REPEAT forge_wgrad2's FAILURE (81% LUT, 2.7-40 ms unreliable): wgrad2 put a CONDITIONAL
// 512-WIDE FIFO emit (all OT*IT accs) INSIDE the II=1 hot loop, which CANNOT pipeline at II=1. The
// FIRST cut of THIS file made the same mistake at smaller width — 64 out.write() (one per IT column)
// in a single n==N-1 iteration, still 64 writes to a single-port hls::stream / cycle => II>=64. FIXED:
// the drain is now spread over PR_IT separate trips, EXACTLY ONE out.write() per cycle, so the FLAT
// loop genuinely holds II=1. The bank is stable during its own drain phase (next tile's reduce only
// starts at the next super-tile), so no ping-pong / RAW hazard. Load / quantize / compute / store are
// split into stages joined by HLS DATAFLOW with a bounded FIFO; compute and store overlap.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_sys.tcl.

#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // PE-grid rows: output o-lanes held stationary
#define PR_IT      64      // PE-grid cols: input  i-lanes held stationary

#define PR_NOT     (PR_O / PR_OT)          // 96  output-row tiles
#define PR_NIT     (PR_I / PR_IT)          // 12  output-col tiles
#define PR_TILES   (PR_NOT * PR_NIT)       // 1152 tiles total
// FIX: a super-tile is PR_N reduce trips THEN PR_IT drain trips. Draining one OT-wide column / cycle
// (not all IT in one cycle) is what keeps the FIFO to ONE write/cycle so the FLAT loop holds II=1.
#define PR_PHASE   (PR_N + PR_IT)          // 80  trips per tile: N reduce + IT drain
#define PR_TRIPS   (PR_TILES * PR_PHASE)   // 92160 flattened super-tile trips (the latency floor)

typedef ap_int<16>  data_t;    // gO / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..16 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (SAME as forge_wgrad_probe — drop-in): [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO = 0;
static const int OFF_X  = OFF_GO + PR_N * PR_O;
static const int OFF_GW = OFF_X  + PR_N * PR_I;

// One OT-wide column of grad accumulators — the unit drained per store cycle (a packed struct so the
// store FIFO carries OT accs/transfer; IT such columns per tile => tile drains in IT cycles).
struct accvec_t { acc_t v[PR_OT]; };

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

// ============================================================================================
// STAGE 1 (load + quantize): stage gO/x on-chip and pre-quantize to dyadic (sign,exp,nz) ONCE.
// Done before the compute pass; the (sign,exp,nz) fields are the operands the PE grid streams.
// ============================================================================================
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

// ============================================================================================
// STAGE 2 (compute): the OT x IT output-stationary PE grid. ONE flattened, fully-pipelined loop over
// PR_TRIPS = PR_TILES super-tiles of (PR_N reduce + PR_IT drain) trips, at II=1. The 512-lane bank
// `acc` is the stationary grid.
//   * ph in [0,PR_N)            REDUCE: 512 dyadic MACs into `acc` for token ph (ph==0 SEEDS, folding
//                               zero-init into the first add — no separate init pass).
//   * ph in [PR_N, PR_N+PR_IT)  DRAIN : emit ONE OT-wide column (col ph-PR_N) of the finished `acc` to
//                               the FIFO. EXACTLY ONE out.write() per cycle.
// FIX (was the II BLOCKER): the previous version emitted all PR_IT(=64) columns in ONE iteration
// (64 out.write() to a single-port hls::stream) inside the II=1 loop — which CANNOT pipeline at II=1
// (forces II>=64, the same wide-FIFO-emit trap that gave forge_wgrad2 its 81% LUT / 2.7-40 ms). Now
// the drain is spread over PR_IT trips, one write/cycle, so the loop genuinely holds II=1. The bank
// is stable during its own drain phase (reduce of the NEXT tile only begins at the next super-tile),
// so no ping-pong / RAW hazard. The pipeline never drains between tiles; store overlaps via DATAFLOW.
// ============================================================================================
static void compute(exp_t goE[PR_N][PR_O], bool goS[PR_N][PR_O], bool goNZ[PR_N][PR_O],
                    exp_t xE[PR_N][PR_I],  bool xS[PR_N][PR_I],  bool xNZ[PR_N][PR_I],
                    hls::stream<accvec_t> &out) {
#pragma HLS INLINE off
    // stationary grid: OT rows x IT cols of grad accumulators (fully in registers)
    acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    int ot = 0;   // current output-row tile base (0,8,..,760)
    int it = 0;   // current output-col tile base (0,64,..,704)

FLAT: for (int t = 0; t < PR_TRIPS; t++) {
#pragma HLS PIPELINE II=1
        int ph = t % PR_PHASE;                  // position within the super-tile [0..PR_PHASE)

        if (ph < PR_N) {
            // ---- REDUCE: dyadic MAC across the OT x IT PE lanes for token ph of this tile. ----
            int n = ph;
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
                    acc_t add = nz ? (neg ? (acc_t)(-term) : term) : (acc_t)0;
                    acc[r][c] = (n == 0) ? add : (acc_t)(acc[r][c] + add);
                }
            }
        } else {
            // ---- DRAIN: emit ONE OT-wide column (col ph-PR_N) of the finished bank. ----
            int dc = ph - PR_N;                 // drain column 0..PR_IT-1
            accvec_t col;
        EMIT_R: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                col.v[r] = acc[r][dc];
            }
            out.write(col);                     // EXACTLY ONE FIFO write this cycle => II=1 holds
            // on the last drain trip, advance to the next tile (row-major over IT then OT)
            if (dc == PR_IT - 1) {
                it += PR_IT;
                if (it == PR_I) { it = 0; ot += PR_OT; }
            }
        }
    }
}

// ============================================================================================
// STAGE 3 (store): drain the FIFO to DDR in the SAME tile order compute emits (row-major IT then
// OT). Each FIFO transfer is an OT-wide column; write its OT elements to their gW rows. Pipelined
// II=1; overlaps with compute under DATAFLOW so DDR write-back hides under the arithmetic.
// ============================================================================================
static void store(int32_t *buffer, hls::stream<accvec_t> &in) {
#pragma HLS INLINE off
SOT: for (int ot = 0; ot < PR_O; ot += PR_OT) {
SIT:    for (int it = 0; it < PR_I; it += PR_IT) {
        SC: for (int c = 0; c < PR_IT; c++) {
#pragma HLS PIPELINE II=1
                accvec_t col = in.read();
            SR: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                    buffer[OFF_GW + (ot + r) * PR_I + (it + c)] = (int32_t)col.v[r];
                }
            }
        }
    }
}

// ============================================================================================
// TOP: load+quantize -> systolic compute -> store, joined by HLS DATAFLOW with a bounded FIFO.
// ============================================================================================
void forge_wgrad_sys(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW

    // prequantized operand fields (the streams the PE grid consumes; partitioned for parallel lane
    // reads: gO on the OT (o) lanes, x on the IT (i) lanes — exactly the probe's banking).
    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=goE  cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goS  cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xE   cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xS   cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ  cyclic factor=PR_IT dim=2

    // store FIFO: carries OT-wide grad columns. A tile = IT (=64) columns; depth 256 = ~4 tiles of
    // slack so the store stage never backpressures the compute pipeline (verified-drain: store
    // consumes IT cols/tile at II=1 while compute produces them in one n==N-1 burst per N=16 trips,
    // so the long-run consume rate >= produce rate and the FIFO stays bounded).
    hls::stream<accvec_t> gwfifo;
#pragma HLS STREAM variable=gwfifo depth=256

    stage_and_quant(buffer, goE, goS, goNZ, xE, xS, xNZ);
    compute(goE, goS, goNZ, xE, xS, xNZ, gwfifo);
    store(buffer, gwfifo);
}
