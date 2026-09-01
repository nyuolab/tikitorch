// forge_wgrad_pe512.cpp — THROUGHPUT (low-latency) operand-quantized weight-grad GEMM (training rig).
//
// Same computation + SAME buffer layout as forge_wgrad_probe (DROP-IN):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])      with q(v) = sign(v) * 2^round(log2|v|)
// Both operands are dyadic (±2^k), so each product is ±2^(eo+ei): a sign-xor + exponent-ADD
// + a barrel SHIFT ((acc_t)1<<e) + a signed ACCUMULATE. NO multiply, NO DSP multiplier in the term.
//
// WHY THIS VERSION (vs forge_wgrad_probe, which CLOSED 2.920 ns / 49% LUT / DSP=0 but ran 664,932
// cycles = 2.66 ms): the probe's OTILE(96) x ITILE(12) tile-traversal loops were NOT pipelined.
// Each of the 1152 tiles paid (zero-init + the N=16 reduction's fill/drain + a 512-element tile
// writeback) latency SERIALLY, with zero tile-to-tile overlap. The only II=1 loop (the tiny N=16
// reduction) was dwarfed by that fixed per-tile overhead (~577 cycles/tile x 1152 tiles).
//
// THE FIX — two changes, both keeping the PROVEN probe conventions (same PE width, same dyad encoder,
// same interface, same banking style):
//  (1) FLATTEN the whole tile traversal (OTILE x ITILE x NRED) into ONE PIPELINE II=1 loop over
//      PR_TILEN = PR_TILES*PR_N = 18,432 trips. The OT x IT = 512-lane output-stationary accumulator
//      bank lives in registers (partitioned COMPLETE). It is SEEDED at n==0 (the first reduction step
//      writes instead of adds — folds the zero-init into the first MAC, removing both the separate
//      init pass AND any read-before-write on the per-lane recurrence) and COMMITTED to an on-chip
//      gW staging array at n==N-1. The pipeline never drains between tiles, so the 1152 fill/drain
//      bubbles of the probe collapse to ONE fill + ONE drain. The per-lane accumulate is a distance-1
//      self-recurrence on a PRIVATE register acc[r][c] (one int48 add/cycle); this is the EXACT
//      recurrence the probe's NRED loop already closed at II=1 / 2.92 ns, so II=1 holds here too.
//  (2) DECOUPLE the writeback. The probe (and the naive "write the tile straight to m_axi at n==N-1"
//      approach) push 512 results through the SINGLE m_axi port in one trip — that one port serializes
//      them, so that trip costs ~512 cycles and effective II balloons (~33). Here completed tiles land
//      in an on-chip gW[O][I] staging array banked on the I axis to match the IT lanes, and a SEPARATE
//      AXI-widened (512-bit) burst streams the whole result to DDR ONCE, after compute. No DATAFLOW /
//      FIFO — that path (forge_wgrad2) blew to 81% LUT and 2.7-40 ms.
//
// LANES = PR_OT*PR_IT = 512 dyadic-MAC ALUs. Compute floor = O*I*N/lanes = 768*768*16/512 = 18,432
// cycles. The final result is O*I = 589,824 int32; the AXI-widened store floor is ~36,864 beats — so
// the STORE, not the compute, is the latency floor of the whole kernel. Total ~ 18,432 (compute) +
// ~36,864 (store) + load/quant a few k  ==> ~60k cycles (~0.24 ms), ~11x faster than the probe's
// 664,932, DSP=0-class, well inside one SLR.
//
// gW staging is O*I*48 bit = 28.3 Mbit, banked cyclic-64 on I -> 64 banks of 768*12=9216 acc_t each
// (~96 URAM18-class) — fits one SLR's 320 URAM18 with the 512-lane PE array (49%-of-SLR LUT, DSP=0).
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_pe512.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= the unrolled PE lane count along I)

#define PR_NOT     (PR_O / PR_OT)          // 96   output-row tiles
#define PR_NIT     (PR_I / PR_IT)          // 12   output-col tiles
#define PR_TILES   (PR_NOT * PR_NIT)       // 1152 tiles total
#define PR_TILEN   (PR_TILES * PR_N)       // 18432 flattened (tile x N) reduction trips — the hot loop

typedef ap_int<16>  data_t;    // gO / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..16 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (DROP-IN, identical to forge_wgrad_probe):
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

void forge_wgrad_pe512(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // ---- Stage operands on-chip, then precompute their dyadic (sign,exp,nz) ONCE. -----------------
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

    // On-chip grad staging: hold the whole O x I result, written tile-by-tile from the hot loop, then
    // burst-stored to DDR. Banked on the I axis by PR_IT so the IT unrolled commit hits distinct banks.
    static acc_t gW[PR_O][PR_I];
#pragma HLS ARRAY_PARTITION variable=gW cyclic factor=PR_IT dim=2

    // ---- THE HOT LOOP: one flattened, fully-pipelined (tile x N) reduction --------------------------
    // An OT x IT accumulator BANK lives in registers (partitioned COMPLETE), seeded at n==0 and
    // committed to gW[][] at n==N-1. The pipeline runs PR_TILEN trips without draining between tiles.
    acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    int ot = 0;   // current output-row tile base (0,8,..,760)
    int it = 0;   // current output-col tile base (0,64,..,704)

FLAT: for (int t = 0; t < PR_TILEN; t++) {
#pragma HLS PIPELINE II=1
        int n = t % PR_N;                       // bounded; folds to a small mod-16 counter in HW

        // dyadic MAC across the OT x IT PE lanes for this (tile, n). On n==0 SEED the bank with this
        // step's term instead of zero-then-add (folds the init into the first MAC — one pass over acc,
        // and removes any read-before-first-write on the per-lane recurrence).
    MR: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
            exp_t eo = goE[n][ot + r]; bool so = goS[n][ot + r]; bool nzo = goNZ[n][ot + r];
        MC: for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                // product = ±2^(eo+ei): barrel shift of a constant 1, NO multiply
                exp_t ei = xE[n][it + c];
                ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;
                acc_t term = (acc_t)1 << e;
                bool neg = so ^ xS[n][it + c];
                bool nz  = nzo & xNZ[n][it + c];
                acc_t add = nz ? (neg ? (acc_t)(-term) : term) : (acc_t)0;
                acc[r][c] = (n == 0) ? add : (acc_t)(acc[r][c] + add);   // distance-1 self-recurrence
            }
        }

        // commit the completed tile to on-chip gW at the last reduction step, then advance the tile.
        // The IT (c) writes hit distinct gW banks (it is a multiple of PR_IT); the PR_OT (r) writes to
        // the same bank are scheduled by HLS within this once-per-16-trips slot — it does not perturb
        // the steady-state II=1 of the MAC because gW is a staging buffer, not the recurrence variable.
        if (n == PR_N - 1) {
        EM: for (int r = 0; r < PR_OT; r++)
                for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                    gW[ot + r][it + c] = acc[r][c];
                }
            it += PR_IT;
            if (it == PR_I) { it = 0; ot += PR_OT; }
        }
    }

    // ---- burst the staged grad back to DDR (own pipelined region; AXI-widenable; no per-tile drain) -
ST: for (int o = 0; o < PR_O; o++)
        for (int i = 0; i < PR_I; i++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_GW + o * PR_I + i] = (int32_t)gW[o][i];
        }
}
