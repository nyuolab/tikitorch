// forge_fwd_gemm.cpp — DYADIC FORWARD GEMM for the FPGA transformer-training rig.
//
// The deploy/forward projection of the rig (the companion to forge_wgrad_probe, the backward):
//   y[n][o] = sum_i q(W[o][i]) * x[n][i] + b[o]        (d=768 FFN-class proj 768x768)
// Here ONLY the WEIGHT is dyadic: q(W) = sign(W)*2^round(log2|W|) = ±2^ew. The activation x is a
// full int16. So each product q(W[o][i]) * x[n][i] is NOT a multiply — it is one VARIABLE SHIFT of the
// full activation by the weight exponent + a signed ACCUMULATE:
//     term = ±( x[n][i] << ew_oi )
// (Contrast the backward wgrad, where BOTH operands are dyadic so the term is a constant 1<<(eo+ei).)
// W is pre-quantized to (sign,exp) once via dyad(); x is streamed in full int16. NO DSP multiplier
// (DSP48 may serve as the signed accumulator/adder — the "DSP-as-adder" lever).
//
// Output-stationary tiling: hold a y-tile (OT rows x N microbatch cols) in on-chip int48 accumulators
// and STREAM the i-reduction (the contraction d=768). The inner OT output-lanes are unrolled = the PE
// lane array; the i-reduction loop is PIPELINE II=1. Mirrors forge_wgrad_probe tiling/pragmas.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_fwd_gemm.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; forward FFN-class proj 768x768) ----
#define FG_O       768     // output features (rows of W) — d_model
#define FG_I       768     // input  features (cols of W) — d_model
#define FG_N       16      // microbatch tokens forward'd together
#define FG_OT      8       // output-tile rows held on-chip (= the unrolled PE output-lane count)

typedef ap_int<16>  data_t;    // W / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic weight exponent 0..16 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // y accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (probe-local): [ W (O*I) | x (N*I) | bias (O) | y (N*O) ]
static const int OFF_W    = 0;
static const int OFF_X    = OFF_W    + FG_O * FG_I;
static const int OFF_BIAS = OFF_X    + FG_N * FG_I;
static const int OFF_Y    = OFF_BIAS + FG_O;

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

void forge_fwd_gemm(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // Stage the activations on-chip (URAM-class) — streamed in full int16; partition on the i (input)
    // dim so the i-reduction reads N lanes per cycle.
    static data_t xx[FG_N][FG_I];
#pragma HLS ARRAY_PARTITION variable=xx complete dim=1
    burst_load(buffer, OFF_X, &xx[0][0], FG_N * FG_I);

    // Stage + pre-quantize the weights to dyadic (sign,exp) once. W[o][i]; partition on i so the
    // i-reduction reads the OT output-lanes' weight for column i in parallel.
    static exp_t wE[FG_O][FG_I];
    static bool  wS[FG_O][FG_I];
    static bool  wNZ[FG_O][FG_I];
    // Partition the o (output) dim by FG_OT so the IRED inner loop's OT unrolled output-lanes each
    // read their weight (wE/wS/wNZ[ot+r][i], r unrolled) from a distinct bank — required for II=1.
#pragma HLS ARRAY_PARTITION variable=wE  cyclic factor=FG_OT dim=1
#pragma HLS ARRAY_PARTITION variable=wS  cyclic factor=FG_OT dim=1
#pragma HLS ARRAY_PARTITION variable=wNZ cyclic factor=FG_OT dim=1
QW: for (int o = 0; o < FG_O; o++) {
        for (int i = 0; i < FG_I; i++) {
#pragma HLS PIPELINE II=1
            data_t w = (data_t)buffer[OFF_W + o * FG_I + i];
            bool s, nz; exp_t e;
            dyad(w, s, e, nz);
            wS[o][i] = s; wE[o][i] = e; wNZ[o][i] = nz;
        }
    }

    // Output-stationary tiled GEMM: hold a y-tile (OT rows x N microbatch) on-chip, STREAM i (the
    // contraction). The OT output-lanes are unrolled = the PE lane array (area+timing live here).
OTILE: for (int ot = 0; ot < FG_O; ot += FG_OT) {
        acc_t acc[FG_OT][FG_N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
        // seed with bias (per output row, broadcast across the N microbatch)
        for (int r = 0; r < FG_OT; r++) {
#pragma HLS UNROLL
            acc_t b = (acc_t)(int32_t)buffer[OFF_BIAS + ot + r];
            for (int n = 0; n < FG_N; n++) {
#pragma HLS UNROLL
                acc[r][n] = b;
            }
        }
IRED:   for (int i = 0; i < FG_I; i++) {
#pragma HLS PIPELINE II=1
            for (int r = 0; r < FG_OT; r++) {
#pragma HLS UNROLL
                exp_t ew = wE[ot + r][i]; bool sw = wS[ot + r][i]; bool nzw = wNZ[ot + r][i];
                for (int n = 0; n < FG_N; n++) {
#pragma HLS UNROLL
                    // dyadic-weight * full activation: term = ±( x << ew ) — one variable shift +
                    // signed accumulate, NO multiply.
                    if (nzw) {
                        // ew is a non-negative magnitude exponent (0..15 for int16). Cast the shift
                        // amount to unsigned (mirrors forge_wgrad_probe) so HLS never sees a signed
                        // shift operand.
                        acc_t term = (acc_t)(int)xx[n][i] << (ap_uint<5>)ew;
                        acc[r][n] += sw ? (acc_t)(-term) : term;
                    }
                }
            }
        }
        // write the y-tile back: y[n][ot+r]
        for (int r = 0; r < FG_OT; r++) {
            for (int n = 0; n < FG_N; n++) {
#pragma HLS PIPELINE II=1
                buffer[OFF_Y + n * FG_O + (ot + r)] = (int32_t)acc[r][n];
            }
        }
    }
}
