// forge_wgrad_wide.cpp — THROUGHPUT fix for the rig's weight-grad GEMM (gate-2 backward kernel).
//
// Diagnosis (from forge_wgrad_probe csynth, 664,932 cycles / 2.66 ms): the 512-lane dyadic-MAC PE is
// NOT the bottleneck (NRED ~35k cycles total). The bottleneck is the WRITEBACK: the probe writes the
// OT*IT = 8*64 = 512-element grad tile back as 512 SCALAR int32 stores at II=1 (one per cycle), so
// 512 * 1152 tiles ≈ 590k cycles = ~89% of the runtime. bigpe (scale lanes) and the flatten variants
// attacked compute — the wrong thing — and either won nothing (bigpe: 600k cycles, 1.83 SLR) or blew
// past the 75-min synthesis backstop (turbo/dsp/fast/pe512).
//
// FIX: keep the PROVEN PE byte-for-byte (same dyadic-MAC, same OT8xIT64 partitioning that closes
// 2.920 ns at 49% LUT/SLR, DSP=0), and replace the scalar writeback with an explicit 512-bit WIDE
// burst: pack 16 consecutive int32 accumulators into one ap_uint<512> word and stream them. IT=64 cols
// = 4 wide words/row * 8 rows = 32 wide writes/tile (II=1) instead of 512 scalar writes -> ~16x on the
// dominant term. Projected ~72k cycles (~0.29 ms), ~9x faster overall, DSP still 0, synthesis-safe
// (this is the probe + a pack loop, NOT a flatten/unroll explosion).
//
// gW region is 512-bit aligned: OFF_GW = 24576 int32 -> /16 = 1536 wide words; every row start
// (ot+r)*768 + it is divisible by 16 (768/16=48, it in {0,64,...}/16=4) -> exact wide indexing.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_wide.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; probe the FFN-class proj 768x768) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= the unrolled PE lane count)

typedef ap_int<16>   data_t;   // gO / x values (int16, the morph data_t range)
typedef ap_int<5>    exp_t;    // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>   acc_t;    // grad accumulator (DSP48-width signed)
typedef ap_uint<512> wide_t;   // 16 x int32 packed AXI word

// flat int32 DDR buffer layout (probe-local): [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO = 0;
static const int OFF_X  = OFF_GO + PR_N * PR_O;
static const int OFF_GW = OFF_X  + PR_N * PR_I;          // = 24576 int32
static const int OFF_GW_W = OFF_GW / 16;                 // = 1536 wide (512-bit) words

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

void forge_wgrad_wide(int32_t *buffer, wide_t *gw_wide) {
#pragma HLS INTERFACE m_axi port=buffer  offset=slave bundle=gmem  max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=gw_wide offset=slave bundle=gmemw
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=gw_wide
#pragma HLS INTERFACE s_axilite port=return

    // Stage operands on-chip (URAM-class), then precompute their dyadic (sign,exp) once.
    static data_t gO[PR_N][PR_O];
    static data_t xx[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=gO cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xx cyclic factor=PR_IT dim=2
    burst_load(buffer, OFF_GO, &gO[0][0], PR_N * PR_O);
    burst_load(buffer, OFF_X,  &xx[0][0], PR_N * PR_I);

    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=goE cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goS cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xE cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xS cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ cyclic factor=PR_IT dim=2
QO: for (int n = 0; n < PR_N; n++)
        for (int o = 0; o < PR_O; o++) { dyad(gO[n][o], goS[n][o], goE[n][o], goNZ[n][o]); }
QX: for (int n = 0; n < PR_N; n++)
        for (int i = 0; i < PR_I; i++) { dyad(xx[n][i], xS[n][i], xE[n][i], xNZ[n][i]); }

    // Output-stationary tiled GEMM: tile (OT x IT) of gW, reduce over the N microbatch.
    // Inner IT cols unrolled = the PE lane array (identical to the proven probe).
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
        // WIDE-BURST writeback: pack 16 contiguous int32 accumulators per 512-bit word.
        // IT=64 cols -> PR_IT/16 = 4 wide words per row; PR_OT=8 rows -> 32 wide writes/tile (II=1).
WB:     for (int r = 0; r < PR_OT; r++) {
            for (int w = 0; w < PR_IT / 16; w++) {
#pragma HLS PIPELINE II=1
                wide_t word;
                for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                    word.range(32 * k + 31, 32 * k) = (ap_uint<32>)acc[r][w * 16 + k];
                }
                // wide word index: gW row (ot+r), col block (it + w*16), in 512-bit units
                int widx = OFF_GW_W + ((ot + r) * PR_I + it) / 16 + w;
                gw_wide[widx] = word;
            }
        }
    }
    }
}
