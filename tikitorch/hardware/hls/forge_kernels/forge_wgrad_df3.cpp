// forge_wgrad_df3.cpp — DATAFLOW (2-stage) weight-grad GEMM, PER-ROW token granularity.
//
// Math + buffer layout IDENTICAL to forge_wgrad_wide2 (the current winner, 76,640 cyc):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])     with q(v) = sign(v) * 2^round(log2|v|)
// Both operands dyadic (±2^k) -> each product = ±2^(eo+ei): sign-xor + exponent-ADD + barrel SHIFT
// ((acc_t)1<<e) + signed ACCUMULATE. NO multiply, NO DSP multiplier (DSP48 may serve as the adder).
//
// ---- WHY THIS VERSION (strategy: 2-STAGE DATAFLOW, OVERLAP WRITEBACK WITH COMPUTE) ---------------
// forge_wgrad_wide2 is OUTPUT-WRITE-BOUND. It must write the whole O*I = 589,824-element gW = 36,864
// wide (512-bit) words. That write floor (36,864) EXCEEDS the 512-lane compute floor
// (768*768*16/512 = 18,432). In wide2 each ITILE costs NRED(~20) + WB(~41) SERIALIZED -> the DDR
// burst sits idle-blocking after every reduction, ~47k of the 75k cycles are write.
//
// forge_wgrad_flat2 tried to overlap compute(k+1) with writeback(k) inside ONE ping-pong loop and
// FAILED to schedule at II=1 (projected 36,864, measured 73,841) — a single pipeline cannot express
// two independent rates. The mechanism flat2 could not express in one loop, DATAFLOW (two concurrent
// processes + a FIFO between them) CAN: a producer that computes tiles and a consumer that drains
// them to DDR run as separate hardware, automatically double-buffered through the FIFO.
//
// ---- df3 GRANULARITY HEDGE (vs a hypothetical whole-tile-token df) ------------------------------
// Instead of streaming a whole tile (PR_OT*PR_IT = 512 accumulators = a 16384-bit token) we stream
// PER-ROW: each token = one gW row-slice of the tile (PR_IT=64 int32 = 4 wide 512-bit words) plus the
// flat wide DDR base index for that row. Small tokens (a tight struct of 4 ap_uint<512> + an int)
// keep the FIFO cheap, schedule cleanly at II=1, and STILL overlap every DDR burst with downstream
// compute. FIFO depth 16 = 2 full tiles of rows of slack so the consumer never stalls the producer.
//
// The producer emits a tile's PR_OT=8 rows as soon as that tile's NRED finishes (one row per inner
// trip). The consumer pops rows and writes each as 4 contiguous wide words to gW.
//
// SYNTHESIS-TRACTABILITY (the v1 df variants fast/turbo/sys blew the 75-min backstop by OVER-UNROLLING):
//   - unroll ONLY the 512 PE lanes (PR_OT x PR_IT) and the 16-wide pack. NEVER the tile count, NEVER N.
//   - two DATAFLOW processes, each a simple bounded loop; the FIFO is a small struct stream.
//   - DSP stays 0 (dyad = priority encoder; MAC = shift + signed accumulate).
//
// gW is 512-bit aligned: OFF_GW = 24576 int32 -> /16 = 1536 wide words; every (ot+r)*PR_I + it start
// is divisible by 16 (768/16=48; it in {0,64,...} -> /16 integral) -> exact wide indexing.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_df3.tcl.

#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= part of the 512 unrolled PE lanes)

#define PR_WBW     (PR_IT / 16)          // wide (512-bit) words per gW row-slice = 4

typedef ap_int<16>   data_t;   // gO / x values (int16, the morph data_t range)
typedef ap_int<5>    exp_t;    // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>   acc_t;    // grad accumulator (DSP48-width signed)
typedef ap_uint<512> wide_t;   // 16 x int32 packed AXI word

// flat int32 DDR buffer layout (probe-local, IDENTICAL to forge_wgrad_wide2):
//   [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO   = 0;
static const int OFF_X    = OFF_GO + PR_N * PR_O;        // = 12288 int32
static const int OFF_GW   = OFF_X  + PR_N * PR_I;        // = 24576 int32
static const int OFF_GO_W = OFF_GO / 16;                 // = 0    wide (512-bit) words
static const int OFF_X_W  = OFF_X  / 16;                 // = 768  wide words
static const int OFF_GW_W = OFF_GW / 16;                 // = 1536 wide words
static const int N_GO_W   = (PR_N * PR_O) / 16;          // = 768  wide words to read for gO
static const int N_X_W    = (PR_N * PR_I) / 16;          // = 768  wide words to read for x

// total gW row-slices streamed = (PR_O rows) * (PR_I / PR_IT col-tiles) = 768 * 12 = 9216
static const int N_ROWS   = PR_O * (PR_I / PR_IT);

// ---- the per-ROW token: 4 wide words (a full gW row-slice) + its flat wide DDR base index. -------
// Small (4*512 + 32 bits) so the FIFO is cheap and schedules at II=1, yet each token still carries a
// complete contiguous 4-word DDR burst that the consumer overlaps with the producer's next compute.
struct row_t {
    wide_t   w[PR_WBW];   // 4 x 512-bit = the 64 int32 of one gW row-slice
    int      widx;        // wide-word DDR index of w[0] (subsequent words are widx+1..widx+3)
};

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

// WIDE LOAD + FUSED QUANTIZE (byte-for-byte forge_wgrad_wide2): read `nwords` 512-bit words, unpack
// the 16 lanes, dyad() each straight into the (sign,exp,nz) banks. element index ee = wbase implied by
// srcw; n = ee/feat, f = ee%feat. feat = 768 = 16*48 so each wide word lies WITHIN one row n -> n
// constant per word, f contiguous; 16 lanes hit 16 distinct cyclic banks -> II=1.
static inline void wide_quant(wide_t *src, int srcw, int nwords, int feat,
                              exp_t eb[][PR_O], bool sb[][PR_O], bool nzb[][PR_O]) {
#pragma HLS INLINE off
LDQ: for (int w = 0; w < nwords; w++) {
#pragma HLS PIPELINE II=1
        wide_t word = src[srcw + w];
        int base = 16 * w;              // flat element index of lane 0 of this word
        int n0   = base / feat;         // row index (constant across the 16 lanes — 16 | feat)
        int f0   = base - n0 * feat;    // column index of lane 0
        for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
            data_t v = (data_t)(int32_t)word.range(32 * k + 31, 32 * k);
            bool s, nz; exp_t e;
            dyad(v, s, e, nz);
            int f = f0 + k;
            sb[n0][f] = s; eb[n0][f] = e; nzb[n0][f] = nz;
        }
    }
}

// ---- STAGE 1: compute_tiles — front-matter + the proven 512-lane PE; emit each finished row. -----
// Output-stationary tiled GEMM identical to wide2's OTILE/ITILE/NRED, but instead of writing gW to DDR
// it PACKS each completed row-slice into a row_t and PUSHES it to the FIFO. The producer's next ITILE
// compute proceeds while the consumer drains the just-pushed rows -> writeback overlaps compute.
static void compute_tiles(wide_t *go_x, hls::stream<row_t> &fifo) {
#pragma HLS INLINE off
    // Dyadic operand banks ONLY (quantize fused into the load; raw int16 arrays never materialized).
    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=goE  cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=goS  cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=xE   cyclic factor=PR_IT  dim=2
#pragma HLS ARRAY_PARTITION variable=xS   cyclic factor=PR_IT  dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ  cyclic factor=PR_IT  dim=2

    // wide-load + fused-quantize gO then x (~768 cyc each).
    wide_quant(go_x, OFF_GO_W, N_GO_W, PR_O, goE, goS, goNZ);
    wide_quant(go_x, OFF_X_W,  N_X_W,  PR_I, xE,  xS,  xNZ);

OTILE: for (int ot = 0; ot < PR_O; ot += PR_OT) {
ITILE: for (int it = 0; it < PR_I; it += PR_IT) {
        acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
        for (int r = 0; r < PR_OT; r++)
            for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                acc[r][c] = 0;
            }
NRED:   for (int n = 0; n < PR_N; n++) {
#pragma HLS PIPELINE II=1
            for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                exp_t eo = goE[n][ot + r]; bool so = goS[n][ot + r]; bool nzo = goNZ[n][ot + r];
                for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                    // dyadic MAC: product = ±2^(eo+ei); a barrel shift + signed accumulate, NO multiply
                    exp_t ei = xE[n][it + c];
                    ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;
                    acc_t term = (acc_t)1 << e;
                    bool neg = so ^ xS[n][it + c];
                    bool nz = nzo & xNZ[n][it + c];
                    if (nz) acc[r][c] += neg ? (acc_t)(-term) : term;
                }
            }
        }
        // EMIT this tile's PR_OT rows as soon as NRED finished. Each row = PR_WBW=4 wide words packed
        // from PR_IT=64 contiguous accumulators; widx = base wide-word index of gW[ot+r][it].
EMIT:   for (int r = 0; r < PR_OT; r++) {
#pragma HLS PIPELINE II=1
            row_t tok;
            for (int w = 0; w < PR_WBW; w++) {
#pragma HLS UNROLL
                wide_t word;
                for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                    word.range(32 * k + 31, 32 * k) = (ap_uint<32>)acc[r][w * 16 + k];
                }
                tok.w[w] = word;
            }
            tok.widx = OFF_GW_W + ((ot + r) * PR_I + it) / 16;
            fifo.write(tok);
        }
    }
    }
}

// ---- STAGE 2: write_tiles — drain the FIFO, burst each row-slice (4 wide words) to gW DDR. --------
// Runs concurrently with compute_tiles; its DDR bursts overlap the producer's next-tile compute.
static void write_tiles(hls::stream<row_t> &fifo, wide_t *gw_wide) {
#pragma HLS INLINE off
WROW: for (int rr = 0; rr < N_ROWS; rr++) {
        row_t tok = fifo.read();
WBW:    for (int w = 0; w < PR_WBW; w++) {
#pragma HLS PIPELINE II=1
            gw_wide[tok.widx + w] = tok.w[w];
        }
    }
}

void forge_wgrad_df3(wide_t *go_x, wide_t *gw_wide) {
#pragma HLS INTERFACE m_axi port=go_x    offset=slave bundle=gmem  max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=gw_wide offset=slave bundle=gmemw max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=go_x
#pragma HLS INTERFACE s_axilite port=gw_wide
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW
    // per-ROW token FIFO; depth 16 = 2 full tiles (8 rows each) of slack so the consumer's DDR latency
    // never back-pressures the producer's compute.
    hls::stream<row_t> fifo;
#pragma HLS STREAM variable=fifo depth=16

    compute_tiles(go_x, fifo);
    write_tiles(fifo, gw_wide);
}
