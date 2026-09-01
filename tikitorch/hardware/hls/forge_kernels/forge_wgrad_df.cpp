// forge_wgrad_df.cpp — DATAFLOW weight-grad GEMM: overlap DDR writeback with compute via a FIFO.
//
// Math + buffer layout IDENTICAL to forge_wgrad_wide2 (the current winner, 76,640 cycles, DSP=0, closes
// 2.920 ns). The PE is byte-for-byte wide2's proven 512-lane dyadic-MAC (OT8 x IT64), and the writeback
// is wide2's exact 32-word 512-bit burst with the SAME wide index. The only structural change: the NRED
// compute and the DDR writeback are split into TWO concurrent dataflow processes linked by one FIFO.
//
// ---- WHY DATAFLOW (the thing flat2 could NOT express) ------------------------------------------------
// wide2 is OUTPUT-WRITE-BOUND. It writes the whole O*I = 589,824-element gW = 36,864 wide (512-bit) words
// — a write-floor that EXCEEDS the 512-lane compute floor (768*768*16/512 = 18,432). In wide2 each tile
// costs ~65 cyc = NRED(20) + WB(41) + ~4, with WB serialized AFTER NRED (no overlap) -> ~47k of the 75k
// cycles are DDR bursts that idle the PE. flat2 tried to overlap them in ONE II=1 ping-pong loop and the
// scheduler refused (projected 36,864, MEASURED 73,841 — do not copy it).
//
// DATAFLOW expresses the overlap that a single loop cannot: two functions running CONCURRENTLY, the FIFO
// decoupling them. compute_tiles emits a packed tile then immediately starts the next tile's NRED while
// write_tiles drains the previous tile's 32 DDR words. With FIFO depth>=4 the writeback of tile k fully
// overlaps the NRED of tile k+1, so total -> max(sum_compute ~23-25k, sum_write ~37-38k) + fill, the
// ~36,864 write-floor + a small fill. Projected ~40k cyc (~1.9x over wide2).
//
// ---- SYNTHESIS-TRACTABILITY (the v1 sweep's fast/turbo/sys blew the 75-min backstop — NOT repeated) ---
//   - UNROLL ONLY the 512 PE lanes (OT8 x IT64) and the 16/32-wide pack. NEVER the tile count, NEVER N.
//   - exactly TWO dataflow processes, each with ONE internal II=1 pipeline (NRED; per-word WB).
//   - the operand banks are populated by the SEQUENTIAL wide_quant front-matter BEFORE the dataflow region
//     (compute random-accesses them, so they must be fully resident — partitioned static arrays, NOT
//     streamed). This is wide2's proven front-matter (~1,563 cyc), unchanged.
//   - one FIFO carrying a whole-tile struct (32 wide words) — single producer / single consumer.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. DSP=0 (dyad is a priority encoder; the MAC is shift+add).
// csynth via synth_wgrad_df.tcl.

#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= part of the 512 unrolled PE lanes)

#define PR_NOT     (PR_O / PR_OT)        // 96 output-row tiles
#define PR_NIT     (PR_I / PR_IT)        // 12 input-col tiles
#define PR_WBW     (PR_IT / 16)          // wide words per tile-row = 4
#define PR_WBT     (PR_OT * PR_WBW)      // wide writeback words per tile = 8*4 = 32

typedef ap_int<16>   data_t;   // gO / x values (int16, the morph data_t range)
typedef ap_int<5>    exp_t;    // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>   acc_t;    // grad accumulator (DSP48-width signed)
typedef ap_uint<512> wide_t;   // 16 x int32 packed AXI word

// one 8x64 grad tile packed as 32 wide (512-bit) words: w[r*PR_WBW + ws] holds
// acc[r][ws*16 .. ws*16+15] (r in 0..PR_OT-1, ws in 0..PR_WBW-1). The writer reconstructs the
// SAME wide index wide2 uses, so packing order MUST stay r-major / ws-minor.
struct tile_t { wide_t w[PR_WBT]; };

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

// dyadic-quantize a value to (sign, exponent): round(log2|v|) via MSB + the next bit (round-to-nearest).
// (copied verbatim from forge_wgrad_wide2.)
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

// WIDE LOAD + FUSED QUANTIZE (verbatim from forge_wgrad_wide2): read `nwords` 512-bit words from
// src[srcw .. srcw+nwords), unpack the 16 lanes, dyad() each straight into the (sign,exp,nz) banks.
// flat element index = 16*w + k; feat (PR_O or PR_I) = 768 = 16*48, so each wide word lies WITHIN one
// row n (16 | 768) and the 16 contiguous columns hit 16 distinct cyclic banks (factor multiple of 16).
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

// ---- DATAFLOW STAGE 1: COMPUTE. For each tile, run the proven 512-lane PE NRED over N, pack the
// resulting 8x64 accumulator into a tile_t (32 wide words), and write it ONCE to the FIFO. The pack+write
// is the only handoff, so per tile this stage is ~NRED(20) + ~1. Banks are read-only here. ---------------
static void compute_tiles(const exp_t goE[PR_N][PR_O], const bool goS[PR_N][PR_O], const bool goNZ[PR_N][PR_O],
                          const exp_t xE[PR_N][PR_I],  const bool xS[PR_N][PR_I],  const bool xNZ[PR_N][PR_I],
                          hls::stream<tile_t> &fifo) {
CT_OT: for (int ot = 0; ot < PR_O; ot += PR_OT) {
CT_IT: for (int it = 0; it < PR_I; it += PR_IT) {
        acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
        for (int r = 0; r < PR_OT; r++)
            for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                acc[r][c] = 0;
            }
        // 512-lane dyadic-MAC reduction over the microbatch (byte-for-byte wide2's PE).
NRED:   for (int n = 0; n < PR_N; n++) {
#pragma HLS PIPELINE II=1
            for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                exp_t eo = goE[n][ot + r]; bool so = goS[n][ot + r]; bool nzo = goNZ[n][ot + r];
                for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                    // dyadic MAC: product = ±2^(eo+ei) — a barrel shift + signed accumulate, NO multiply
                    exp_t ei = xE[n][it + c];
                    ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;
                    acc_t term = (acc_t)1 << e;
                    bool neg = so ^ xS[n][it + c];
                    bool nz = nzo & xNZ[n][it + c];
                    if (nz) acc[r][c] += neg ? (acc_t)(-term) : term;
                }
            }
        }
        // PACK the whole tile into one token: 32 wide words, r-major / ws-minor (matches the writer's
        // wide index). The 16-lane pack per word is unrolled; the 32 words are unrolled too (pure register
        // rewires, no DDR here) so the handoff is a single fifo.write.
        tile_t tile;
PACK:   for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
            for (int ws = 0; ws < PR_WBW; ws++) {
#pragma HLS UNROLL
                wide_t word;
                for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                    word.range(32 * k + 31, 32 * k) = (ap_uint<32>)acc[r][ws * 16 + k];
                }
                tile.w[r * PR_WBW + ws] = word;
            }
        }
        fifo.write(tile);
    }
    }
}

// ---- DATAFLOW STAGE 2: WRITE. Drain one tile_t per tile and stream its 32 wide words to DDR with the
// EXACT wide index wide2 uses. Iteration order MUST match the producer (ot outer, it inner) so the FIFO
// reads line up with the tiles produced. Per word: row r = wi / PR_WBW, sub-word ws = wi % PR_WBW. ------
static void write_tiles(hls::stream<tile_t> &fifo, wide_t *gw_wide) {
WT_OT: for (int ot = 0; ot < PR_O; ot += PR_OT) {
WT_IT: for (int it = 0; it < PR_I; it += PR_IT) {
        tile_t tile = fifo.read();
WB:     for (int wi = 0; wi < PR_WBT; wi++) {
#pragma HLS PIPELINE II=1
            int r  = wi / PR_WBW;       // 0..PR_OT-1
            int ws = wi - r * PR_WBW;   // 0..PR_WBW-1
            // SAME wide word index as wide2: gW row (ot+r), col block (it + ws*16), in 512-bit units.
            int widx = OFF_GW_W + ((ot + r) * PR_I + it) / 16 + ws;
            gw_wide[widx] = tile.w[wi];
        }
    }
    }
}

void forge_wgrad_df(wide_t *go_x, wide_t *gw_wide) {
#pragma HLS INTERFACE m_axi port=go_x    offset=slave bundle=gmem  max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=gw_wide offset=slave bundle=gmemw max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=go_x
#pragma HLS INTERFACE s_axilite port=gw_wide
#pragma HLS INTERFACE s_axilite port=return

    // Dyadic operand banks (the raw int16 gO/xx arrays are gone — quantize is fused into the load).
    // Partition factor 16 (gO) / 64 (x) on the feature dim makes the 16-lane wide unpack conflict-free
    // AND feeds the PE: for gO, ot (multiple of 8) -> {ot..ot+7} hit 8 distinct banks mod 16; for x,
    // it (multiple of 64) -> {it..it+63} hit all 64 banks. (Identical to wide2.)
    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=goE  cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=goS  cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=xE   cyclic factor=PR_IT  dim=2
#pragma HLS ARRAY_PARTITION variable=xS   cyclic factor=PR_IT  dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ  cyclic factor=PR_IT  dim=2

    // ---- SEQUENTIAL FRONT-MATTER (outside the dataflow region): wide-load + fused-quantize gO then x.
    // The banks are random-accessed by the compute stage, so they must be FULLY resident before the
    // dataflow region — populate them here (~768 cyc each, II=1, 16-lane unrolled). (wide2's front-matter.)
    wide_quant(go_x, OFF_GO_W, N_GO_W, PR_O, goE, goS, goNZ);
    wide_quant(go_x, OFF_X_W,  N_X_W,  PR_I, xE,  xS,  xNZ);

    // ---- DATAFLOW REGION: exactly TWO concurrent processes linked by ONE FIFO. compute_tiles produces
    // packed tiles; write_tiles drains them to DDR. depth>=4 lets the DDR bursts of tile k overlap the
    // NRED of tile k+1 -> total ~= the write-floor (~37k) + fill, not compute+write serialized.
    //
    // The pragma MUST scope ONLY these two process calls (an explicit nested { }), so the wide_quant
    // front-matter above stays SEQUENTIAL pre-region code: the dyadic banks are then fully resident
    // before the region begins and cross the boundary as a read-only shared array (the one allowed
    // non-stream dependency). A function-scope DATAFLOW would pull wide_quant into the region, making
    // the banks an inter-process random-access array (un-ping-pongable, static) -> canonical-form
    // violation + serialization. Scoping the region fixes that.
    {
#pragma HLS DATAFLOW
        hls::stream<tile_t> fifo;
#pragma HLS STREAM variable=fifo depth=4

        compute_tiles(goE, goS, goNZ, xE, xS, xNZ, fifo);
        write_tiles(fifo, gw_wide);
    }
}
