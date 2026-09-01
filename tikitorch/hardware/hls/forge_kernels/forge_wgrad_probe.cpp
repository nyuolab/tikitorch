// forge_wgrad_probe.cpp — GATE-2 csynth PROBE for the FPGA training rig.
//
// The rig's biggest synthesis risk: the operand-quantized weight-grad GEMM at transformer width
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])        (the PC set_mf_grads recipe)
// where q(v) = sign(v)*2^round(log2|v|) (dyadic). Because BOTH operands are ±2^k, each product
// q(gO)*q(x) = ±2^(eo+ei) — NOT a multiply: it is a sign-xor + exponent-ADD + a barrel-SHIFT
// (1<<(eo+ei)) + a signed ACCUMULATE. So the PE uses NO DSP multiplier (DSP48 may serve as the
// signed accumulator/adder — the "DSP-as-adder" lever, gate 1). This probe measures LUT/DSP/timing
// at d768, against the chess-TRAIN precedent that did NOT close (184% of one SLR, 5.427 ns vs 4.0).
//
// Algorithm validated (no quality cost on a transformer char-LM, 2.481 bpb): scripts/charlm_mixer.py
// --linear mfgrad. This kernel is the silicon form of that backward. Output-stationary tiling holds an
// OT x IT grad tile in on-chip int32 accumulators; the microbatch N is the reduction (add-tree).
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_probe.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; probe the FFN-class proj 768x768) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= the unrolled PE lane count)

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

void forge_wgrad_probe(int32_t *buffer) {
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
    // Inner IT cols unrolled = the PE lane array (gate 1/2 area+timing live here).
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
        // write the grad tile back
        for (int r = 0; r < PR_OT; r++)
            for (int c = 0; c < PR_IT; c++) {
#pragma HLS PIPELINE II=1
                buffer[OFF_GW + (ot + r) * PR_I + (it + c)] = (int32_t)acc[r][c];
            }
    }
    }
}
