// forge_wgrad_flat2.cpp — ROOFLINE weight-grad GEMM: ping-pong accumulator, single flat II=1 pipeline.
//
// Math + buffer layout IDENTICAL to forge_wgrad_probe / forge_wgrad_wide (PC set_mf_grads recipe):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])     with q(v) = sign(v) * 2^round(log2|v|)
// Both operands dyadic (±2^k) -> each product is ±2^(eo+ei): sign-xor + exponent-ADD + barrel SHIFT
// ((acc_t)1<<e) + signed ACCUMULATE. NO multiply, NO DSP multiplier (DSP48 may serve as the adder).
//
// ---- WHY THIS VERSION (strategy: FLAT II=1 + PING-PONG WRITEBACK OVERLAP) ----------------------
// forge_wgrad_wide already widened the writeback (512-bit, 32 wide words/tile vs 512 scalar) and got
// to ~112k cycles / 0.448 ms. Its residual cost is STRUCTURAL: OTILE(96) x ITILE(12) = 1152 tiles,
// each a SEPARATE NRED pipeline (trip count only PR_N=16) followed by a SERIALIZED WB. The deep
// 512-lane MAC pipeline FILL is therefore paid 1152 times, and the 32-word WB never overlaps the next
// tile's compute -> per-tile fixed overhead dominates (OTILE region ~75k of the ~112k).
//
// ROOFLINE: the kernel writes the whole O*I = 589,824 int32 grad. 512-bit-wide that is 589,824/16 =
// 36,864 wide words — an UNAVOIDABLE writeback floor. The 512-lane compute roofline is only
// O*I*N/512 = 768*768*16/512 = 18,432 cycles, BELOW the writeback floor. So this kernel is
// WRITEBACK-bound, and the best achievable is ~36,864 cycles IF compute hides fully under writeback.
//
// FIX (this file): ONE flat loop over (tile, phase) with exactly ONE PIPELINE II=1 pragma, and a
// DOUBLE-BUFFERED (ping-pong) accumulator acc[2][OT][IT] selected by (tile&1). Each tile gets
// STEPS=32 phases. Within a tile's phases the loop body does, every cycle, BOTH:
//   (a) one dyadic-MAC reduction step into the CURRENT bank   (active for phases p < PR_N = 16), and
//   (b) one 512-bit WIDE writeback word from the PREVIOUS bank (active for phases p < 32 = WB words).
// Because (a) and (b) touch OPPOSITE ping-pong banks, tile k's writeback OVERLAPS tile k+1's compute
// inside the single pipeline — the 1152 fill/drain bubbles collapse to ONE. Cost ~= NTILES*32 + fill
// = 1152*32 + fill ≈ 36,864 cycles ≈ the writeback floor (~3x faster than wide, ~18x vs the probe).
//
// SYNTHESIS-TRACTABILITY (the v1 flatten/dataflow variants blew the 75-min backstop — NOT repeated):
//   - the PROVEN PE is byte-for-byte the probe's: OT=8 x IT=64 = 512 lanes, the ONLY thing unrolled.
//   - NO unroll of the tile count, NO unroll of N, NO full-row (768-wide) accumulator (that was the
//     forge_wgrad_flat blowup: gW[768][768] partitioned complete = ~590k registers, 1.83 SLR).
//   - exactly ONE PIPELINE pragma (on the flat loop). NO DATAFLOW (the dataflow variants timed out).
//   - bounded loops only, no dynamic alloc/recursion. The accumulator doubles 512->1024 acc_t regs.
//
// gW is 512-bit aligned: OFF_GW = 24576 int32 -> /16 = 1536 wide words; every (ot+r)*PR_I + it start
// is divisible by 16 (768/16=48, it in {0,64,...}/16 integral) -> exact wide indexing.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_flat2.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= part of the 512 unrolled PE lanes)

#define PR_NOT     (PR_O / PR_OT)        // 96 output-row tiles
#define PR_NIT     (PR_I / PR_IT)        // 12 input-col tiles
#define PR_NTILES  (PR_NOT * PR_NIT)     // 1152 tiles
#define PR_WBW     (PR_IT / 16)          // wide words per tile-row = 4
#define PR_WBT     (PR_OT * PR_WBW)      // wide writeback words per tile = 8*4 = 32
// phases/tile = max(reduction trips, writeback words) so BOTH fit one phase budget; here WB(32) > N(16)
#define PR_STEPS   PR_WBT                // 32

typedef ap_int<16>   data_t;   // gO / x values (int16, the morph data_t range)
typedef ap_int<5>    exp_t;    // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>   acc_t;    // grad accumulator (DSP48-width signed)
typedef ap_uint<512> wide_t;   // 16 x int32 packed AXI word

// flat int32 DDR buffer layout (probe-local, identical to wide): [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO   = 0;
static const int OFF_X    = OFF_GO + PR_N * PR_O;
static const int OFF_GW   = OFF_X  + PR_N * PR_I;          // = 24576 int32
static const int OFF_GW_W = OFF_GW / 16;                   // = 1536 wide (512-bit) words

// dyadic-quantize a value to (sign, exponent): round(log2|v|) via MSB + the next bit (round-to-nearest).
// (copied verbatim from the proven probe.)
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

void forge_wgrad_flat2(int32_t *buffer, wide_t *gw_wide) {
#pragma HLS INTERFACE m_axi port=buffer  offset=slave bundle=gmem  max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=gw_wide offset=slave bundle=gmemw
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=gw_wide
#pragma HLS INTERFACE s_axilite port=return

    // ---- Stage operands on-chip, then precompute their dyadic (sign,exp,nz) once. ----------------
    static data_t gO[PR_N][PR_O];
    static data_t xx[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=gO cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xx cyclic factor=PR_IT dim=2
    burst_load(buffer, OFF_GO, &gO[0][0], PR_N * PR_O);
    burst_load(buffer, OFF_X,  &xx[0][0], PR_N * PR_I);

    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=goE  cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goS  cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xE   cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xS   cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ  cyclic factor=PR_IT dim=2
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

    // ---- PING-PONG accumulator (double-buffered): bank 0/1 selected by (tile & 1). Two banks so the
    // wide writeback of the just-finished tile (PREV bank) overlaps the compute of the current tile
    // (CUR bank) inside one pipeline. 2 * PR_OT * PR_IT = 1024 acc_t regs, partitioned complete so all
    // 512 lanes update + a 16-wide pack reads in the same cycle.
    static acc_t acc[2][PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // ---- ONE flat II=1 pipeline over (tile, phase). The loop is offset by +1 tile of phases so the
    // FINAL tile's writeback drains: trips = (PR_NTILES + 1) * PR_STEPS. For each trip we derive the
    // compute tile `ct` (does the MAC into CUR bank) and the writeback tile `wt = ct-1` (drains PREV
    // bank). The first tile-of-phases has no PREV (skip WB); the last has no CUR (skip compute).
    const int TOTAL = (PR_NTILES + 1) * PR_STEPS;
FLAT: for (int t = 0; t < TOTAL; t++) {
#pragma HLS PIPELINE II=1
        int ct = t / PR_STEPS;       // compute-tile index 0..PR_NTILES   (== PR_NTILES => compute idle)
        int p  = t % PR_STEPS;       // phase 0..PR_STEPS-1
        int wt = ct - 1;             // writeback-tile index (the tile finished in the PREVIOUS group)

        int cur  = ct & 1;           // ping-pong bank being computed this group
        int prev = wt & 1;           // ping-pong bank being written back this group

        // ----- (a) COMPUTE one reduction step (n = p) into the CUR bank, for the active compute tile.
        // Phase p < PR_N (=16) carries the contraction; p==0 SEEDS (folds zero-init into the first add).
        if (ct < PR_NTILES && p < PR_N) {
            int ot = (ct / PR_NIT) * PR_OT;     // output-row base of this tile
            int it = (ct % PR_NIT) * PR_IT;     // input-col base of this tile
            int n  = p;
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
                    bool nz  = nzo & xNZ[n][it + c];
                    acc_t add = nz ? (neg ? (acc_t)(-term) : term) : (acc_t)0;
                    // SEED at n==0 else accumulate (collapses per-tile zero-init into the pipeline)
                    acc[cur][r][c] = (n == 0) ? add : (acc_t)(acc[cur][r][c] + add);
                }
            }
        }

        // ----- (b) WRITE BACK one 512-bit wide word (phase p in 0..PR_WBT-1) from the PREV bank, for
        // the tile finished last group. PR_WBT=32 words/tile: row r = p / PR_WBW, col-block w = p % PR_WBW.
        if (wt >= 0 && p < PR_WBT) {
            int wot = (wt / PR_NIT) * PR_OT;    // output-row base of the writeback tile
            int wit = (wt % PR_NIT) * PR_IT;    // input-col base of the writeback tile
            int r   = p / PR_WBW;               // 0..PR_OT-1
            int w   = p % PR_WBW;               // 0..PR_WBW-1
            wide_t word;
            for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                word.range(32 * k + 31, 32 * k) = (ap_uint<32>)acc[prev][r][w * 16 + k];
            }
            int widx = OFF_GW_W + ((wot + r) * PR_I + wit) / 16 + w;
            gw_wide[widx] = word;
        }
    }
}
