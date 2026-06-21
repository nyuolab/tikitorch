// forge_wgrad_turbo.cpp — THROUGHPUT-optimized operand-quantized weight-grad GEMM (training rig).
//
// Same computation + SAME DDR buffer layout as forge_wgrad_probe (drop-in):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])      with q(v) = sign(v) * 2^round(log2|v|)
// Because BOTH operands are dyadic (±2^k), each product is ±2^(eo+ei): a sign-XOR + exponent-ADD
// + a barrel SHIFT ((acc_t)1<<e) + a signed ACCUMULATE. NO multiply, NO DSP multiplier.
//
// WHY THIS VERSION (throughput is THE flagship lever — the rig is compute-bound):
//   forge_wgrad_probe CLOSED timing (2.920 ns, 49% LUT-of-SLR, DSP=0) but ran 664,932 cycles
//   (2.66 ms for the 768x768 / N16 grad) because its OTILE(96) x ITILE(12) tile-traversal loops were
//   NOT pipelined: every one of the 1152 tiles paid (zero-init + NRED pipeline fill/drain + writeback)
//   latency SERIALLY, and the only II=1 loop (the tiny N=16 reduction) was dwarfed by that per-tile
//   fixed overhead. A DATAFLOW/FIFO rewrite (forge_wgrad2) instead blew LUT to 81% / 2.7-40 ms.
//
// STRATEGY — FLATTEN the whole traversal into ONE II=1 pipeline; DECOUPLE the writeback.
//   1. Keep the probe's PROVEN PE tile geometry: OT=8 x IT=64 = 512 dyadic-MAC lanes (the exact array
//      that closed 2.920 ns at 49% LUT, DSP=0). The change is LOOP STRUCTURE, not lane count, so
//      timing/area should track the probe.
//   2. Collapse OTILE x ITILE x NRED into a SINGLE PIPELINE II=1 loop over PR_TILES*PR_N = 18,432
//      trips. The OT x IT accumulator bank (partitioned COMPLETE = 512 independent registers) is
//      SEEDED at n==0 (zero-init folded into the first add, so no separate init pass competes for the
//      II=1 slot) and committed to an ON-CHIP gW staging array at n==N-1. The pipeline NEVER drains
//      between tiles -> the baseline's 1152 fill/drain bubbles collapse to ONE.
//   3. The grad writeback is a SEPARATE pipelined region AFTER the hot loop: a contiguous,
//      AXI-widenable (512-bit) burst over the staged O*I result. This is the proven commit-to-on-chip
//      -then-burst pattern (forge_wgrad_dsp). It avoids the two known traps: (a) a wide CONDITIONAL
//      FIFO/stream emit inside the II=1 loop (one stream has one write port/cycle -> forces II>=width;
//      sank forge_wgrad2/sys with 81% LUT), and (b) a wide m_axi store burst inside the II=1 loop (the
//      m_axi port likewise cannot take 512 writes/cycle -> forces II>1 on the hot loop). Here the hot
//      loop touches ONLY on-chip banks, so it holds II=1.
//
//   Compute floor = total_MACs / lanes = 768*768*16 / 512 = 18,432 cycles. Plus load+quant (~few k)
//   and the widened drain (O*I = 590k int32 -> ~36,864 beats at 16 int32 / 512-bit beat). Total
//   ~60k cycles (~240 us @ 4 ns) vs the baseline's 664,932 — an ~11x cut, at the probe's proven
//   512-lane area/timing. (More lanes -> fewer compute cycles but more area; see forge_wgrad_bigpe.)
//
//   AREA: same 512-lane PE tile as the probe (49% LUT, DSP=0) + an on-chip O*I acc_t staging array
//   (banked cyclic by PR_IT for the 512-wide tile commit and the contiguous drain). Within one SLR.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_turbo.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= the unrolled PE lane count)

#define PR_NOT     (PR_O / PR_OT)          // 96   output-row tiles
#define PR_NIT     (PR_I / PR_IT)          // 12   output-col tiles
#define PR_TILES   (PR_NOT * PR_NIT)       // 1152 tiles total
#define PR_TILEN   (PR_TILES * PR_N)       // 18432 flattened (tile x N) trips — the hot loop & floor

typedef ap_int<16>  data_t;    // gO / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..16 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (drop-in, identical to forge_wgrad_probe):
//   [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO = 0;
static const int OFF_X  = OFF_GO + PR_N * PR_O;
static const int OFF_GW = OFF_X  + PR_N * PR_I;

// dyadic-quantize a value to (sign, exponent): round(log2|v|) via MSB + the next bit (round-to-nearest).
// COPIED verbatim from forge_wgrad_probe (the proven priority-encoder).
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

void forge_wgrad_turbo(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // ---- Stage operands on-chip, then precompute their dyadic (sign,exp,nz) ONCE. ----------------
    static data_t gO[PR_N][PR_O];
    static data_t xx[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=gO cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xx cyclic factor=PR_IT dim=2
    burst_load(buffer, OFF_GO, &gO[0][0], PR_N * PR_O);
    burst_load(buffer, OFF_X,  &xx[0][0], PR_N * PR_I);

    // Pre-quantized operand fields (live across the whole hot loop). Partition on the feature dim so
    // the OT/IT unrolled PE lanes each read their (sign,exp,nz) from a distinct bank — required for II=1.
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

    // On-chip grad staging: the whole O x I result, committed tile-by-tile from the hot loop, then
    // burst-stored to DDR. The n==N-1 commit writes all OT*IT = 512 tile elements in ONE cycle, so
    // gW must be banked on BOTH axes — cyclic factor PR_OT on the row (o) axis AND PR_IT on the col
    // (i) axis — giving PR_OT*PR_IT = 512 distinct banks, one write port each. (Banking on i alone
    // would put PR_OT rows into the same column bank => 8 writes to one 2-port memory => II>1.)
    static acc_t gW[PR_O][PR_I];
#pragma HLS ARRAY_PARTITION variable=gW cyclic factor=PR_OT dim=1
#pragma HLS ARRAY_PARTITION variable=gW cyclic factor=PR_IT dim=2

    // ---- THE HOT LOOP: one flattened, fully-pipelined (tile x N) reduction over the 512-lane PE -----
    // The OT x IT accumulator bank lives in registers (partitioned complete dim=0 => 512 independent
    // accumulators). It is SEEDED at n==0 and committed to gW[][] at n==N-1. The ONLY loop-carried
    // dependency is each lane's own acc += add (distance 1, a 48-bit add => 1 cycle @ 4 ns), and every
    // operand read hits a distinct on-chip bank, so the loop holds II=1 and never drains between tiles.
    acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    int ot = 0;   // current output-row tile base (0,8,..,760)
    int it = 0;   // current output-col tile base (0,64,..,704)

FLAT: for (int t = 0; t < PR_TILEN; t++) {
#pragma HLS PIPELINE II=1
        int n = t % PR_N;                       // bounded; folds to a small mod-16 counter in HW

        // dyadic MAC across the OT x IT PE lanes for this (tile, n) — 512 MACs this cycle.
    MR: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
            exp_t eo = goE[n][ot + r]; bool so = goS[n][ot + r]; bool nzo = goNZ[n][ot + r];
        MC: for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                // product = ±2^(eo+ei): barrel shift of a constant 1, NO multiply
                exp_t ei = xE[n][it + c];
                ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;     // exponent ADD
                acc_t term = (acc_t)1 << e;                         // barrel SHIFT
                bool neg = so ^ xS[n][it + c];                      // sign XOR
                bool nz  = nzo & xNZ[n][it + c];
                acc_t add = nz ? (neg ? (acc_t)(-term) : term) : (acc_t)0;
                // SEED at n==0 (folds the zero-init into the first add) else signed-accumulate.
                acc[r][c] = (n == 0) ? add : (acc_t)(acc[r][c] + add);
            }
        }

        // commit the completed tile to on-chip gW at the last reduction step, then advance the tile.
        // This touches only on-chip banks (distinct addresses, one write port each) — no m_axi/FIFO in
        // the hot loop, so the FLAT pipeline stays at II=1.
        if (n == PR_N - 1) {
        EM: for (int r = 0; r < PR_OT; r++)
                for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                    gW[ot + r][it + c] = acc[r][c];
                }
            // advance to the next tile (row-major over IT then OT)
            it += PR_IT;
            if (it == PR_I) { it = 0; ot += PR_OT; }
        }
    }

    // ---- single AXI-widenable (512-bit) burst writeback of the whole O*I grad to DDR ---------------
    // Own pipelined region (no per-tile drain); contiguous addressing -> HLS packs 16 int32 / beat.
ST: for (int o = 0; o < PR_O; o++)
        for (int i = 0; i < PR_I; i++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_GW + o * PR_I + i] = (int32_t)gW[o][i];
        }
}
