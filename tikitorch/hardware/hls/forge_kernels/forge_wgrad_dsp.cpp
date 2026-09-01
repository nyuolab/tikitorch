// forge_wgrad_dsp.cpp — THROUGHPUT (DSP-as-adder) operand-quantized weight-grad GEMM (training rig).
//
// Same computation + same buffer layout as forge_wgrad_probe (drop-in):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])      with q(v) = sign(v) * 2^round(log2|v|)
// Both operands are dyadic (±2^k), so each product is ±2^(eo+ei): a sign-xor + exponent-ADD
// + a barrel SHIFT ((acc_t)1<<e) + a signed ACCUMULATE. NO multiply, NO DSP multiplier in the term.
//
// WHY THIS VERSION (vs forge_wgrad_probe, which CLOSED 2.920 ns / 49% LUT / DSP=0 but ran 664,932
// cycles = 2.66 ms): the probe's OTILE(96) x ITILE(12) tile-traversal loops were NOT pipelined —
// every one of the 1152 tiles paid (zero-init + NRED fill/drain + writeback) latency serially, and
// NRED is tiny (N=16), so per-tile fixed overhead dominated. THIS version flattens the WHOLE tile
// traversal into ONE PIPELINE II=1 loop over (tile x N) = 18,432 trips: an OT x IT accumulator bank
// is zeroed at n==0, MAC'd across the OT x IT PE lanes every trip, and committed to an on-chip grad
// staging array at n==N-1. The pipeline never drains between tiles -> the 1152 fill/drain bubbles
// collapse to ONE. (No DATAFLOW / FIFO — that path, forge_wgrad2.cpp, blew up to 81% LUT and gave
// unreliable 2.7-40 ms. Here load/quant and store are plain pipelined regions around the one hot loop.)
//
// STRATEGY — DSP-AS-ADDER (gate 1): the probe is DSP=0 (LUT adders), leaving ~9000 DSP48 idle. Here
// each of the OT*IT = 512 accumulator lanes does one int48 signed add/cycle in the hot loop; we BIND
// those adds onto DSP48 ALUs (#pragma HLS BIND_OP variable=acc op=add impl=dsp). That moves the wide
// adders off the LUT fabric onto the dedicated DSP48 adder (which also carries the accumulate register),
// FREEING LUT (-> more PE lanes at II=1, higher tok/s) and helping the carry-chain timing. ~512 DSPs as
// adders is well within one SLR's 3008. Aim: DSP in the hundreds, LUT DOWN vs probe, <= 4 ns, one SLR.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_dsp.tcl.

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
#define PR_TILEN   (PR_TILES * PR_N)       // 18432 flattened (tile x N) trips — the hot loop

typedef ap_int<16>  data_t;    // gO / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (drop-in, same as forge_wgrad_probe): [ gO (N*O) | x (N*I) | gW (O*I) ]
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

void forge_wgrad_dsp(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // Stage operands on-chip (URAM-class), then precompute their dyadic (sign,exp) once.
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
    // burst-stored to DDR. Partition on i by PR_IT so the OT*IT writeback at n==N-1 hits distinct banks.
    static acc_t gW[PR_O][PR_I];
#pragma HLS ARRAY_PARTITION variable=gW cyclic factor=PR_IT dim=2

    // ---- THE HOT LOOP: one flattened, fully-pipelined (tile x N) reduction ------------------------
    // An OT x IT accumulator BANK lives in registers (partitioned complete), zeroed at n==0 and
    // committed to gW[][] at n==N-1. The pipeline runs PR_TILEN trips without draining between tiles.
    acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
    // DSP-AS-ADDER: bind the 512 wide accumulate adds onto DSP48 ALUs (frees LUT, helps carry timing).
#pragma HLS BIND_OP variable=acc op=add impl=dsp

    int ot = 0;   // current output-row tile base (0,8,..,760)
    int it = 0;   // current output-col tile base (0,64,..,704)

FLAT: for (int t = 0; t < PR_TILEN; t++) {
#pragma HLS PIPELINE II=1
        int n = t % PR_N;                       // bounded; folds to a small mod-16 counter in HW

        // dyadic MAC across the OT x IT PE lanes for this (tile, n). On n==0 SEED the bank with this
        // step's term instead of zero-then-add (one fewer pass over acc, same result).
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
                acc[r][c] = (n == 0) ? add : (acc_t)(acc[r][c] + add);   // <- DSP48 add-accumulate
            }
        }

        // commit the completed tile to on-chip gW at the last reduction step, then advance the tile.
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

    // ---- burst the staged grad back to DDR (own pipelined region; no per-tile drain) --------------
ST: for (int o = 0; o < PR_O; o++)
        for (int i = 0; i < PR_I; i++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_GW + o * PR_I + i] = (int32_t)gW[o][i];
        }
}
