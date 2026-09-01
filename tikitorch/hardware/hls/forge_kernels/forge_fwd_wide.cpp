// forge_fwd_wide.cpp — REALISTIC dyadic FORWARD GEMM for the FPGA transformer-training rig.
//
// The deploy/forward projection (companion to forge_wgrad_wide, the backward weight-grad):
//   y[n][o] = sum_i ( q(x[n][i])  dyadic-MAC  W_dyadic[o][i] ) + b[o]     (d=768 FFN-class proj 768x768)
//
// CRUCIAL DIFFERENCE vs forge_fwd_gemm: W ARRIVES PRE-DYADIC. In the real 124M rig (forge_buffer_gpt2.hpp)
// the DDR holds every weight as a 1-byte (sign,exp) code — NOT a full int16 to be quantized on-chip. So the
// 589,828-cycle on-chip QW pass that dominated forge_fwd_gemm (a PROBE ARTIFACT: it dyad()'d the full
// 768x768 W with a priority encoder per element) is GONE. Here W is brought on-chip by a 512-bit WIDE BURST
// (16 codes/cycle, II=1 -> ~37k cycles) and DECODED INLINE in the reduction (a few mask/shift logic ops per
// lane, zero extra passes, NO priority encoder).
//
// W byte encoding (this kernel's contract; documented + self-contained):
//   int8 code, per weight (stored one-code-per-int32 lane in the probe buffer; the real rig packs 4/word —
//   the on-chip decode is identical):
//     bit 7      = sign     (1 = negative)
//     bit 5      = nonzero  (1 = weight is a real ±2^e ; 0 = weight is exactly 0, term skipped)
//     bits 4..0  = exponent e in 0..31 (the dyadic magnitude shift; |W| = 2^e)
//     bit 6      = reserved (0)
//   This is the "1 sign bit + ~5-bit exponent + slack = INT8" of forge_buffer_gpt2.hpp made concrete.
//
// x is a FRESH activation, so it IS dyadic-quantized on-chip once (q(x) = ±2^ex via the proven dyad()).
// With both operands dyadic the MAC is the proven constant-shift PE (identical to forge_wgrad_wide):
//     term = ±2^(ex+ew)   — a sign-xor + exponent-ADD + barrel-SHIFT + signed ACCUMULATE, NO multiply.
// (The single soft-blend multiply of the rig lives in forge_attn, NOT here. DSP=0 expected.)
//
// Output-stationary tiling: hold a y-tile (OT output rows x N microbatch) in on-chip int48 accumulators;
// STREAM the i-reduction (the contraction d=768, PIPELINE II=1). The inner OT*N tile is the unrolled PE
// lane array (OT=32 x N=16 = 512 lanes — the proven ~512-lane budget; we unroll ONLY the lane tile, never
// the tile-count loop nor the i-reduction). WIDE-BURST writeback of y: pack 16 contiguous int32 outputs
// (along o) into one ap_uint<512> word -> 2 wide words/row -> 32 wide writes/tile vs 512 scalar.
//
// Expected cycles (front-matter is the W+x WIDE loads + x quantize; compute is the i-reduction):
//   W wide load  ~36,864 (589,824 codes / 16 per 512-bit word, II=1)
//   x burst load ~12,288 (N*I) + x quantize QX ~12,288
//   IRED compute  24 otiles x ~768 (+ depth) ~= 18,500 ; WB 24 x 32 wide writes ~= 768
//   => ~80-85k cycles total, DSP=0, 4ns close, small LUT/SLR.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_fwd_wide.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; forward FFN-class proj 768x768) ----
#define FW_O       768     // output features (rows of W) — d_model
#define FW_I       768     // input  features (cols of W) — d_model  (the contraction)
#define FW_N       16      // microbatch tokens forward'd together
#define FW_OT      32      // output-tile rows held on-chip (= unrolled PE output-lane count)
                           // OT(32) * N(16) = 512 PE lanes = the proven lane budget

typedef ap_int<16>   data_t;   // x values (int16, the morph data_t range)
typedef ap_int<6>    exp_t;    // dyadic exponent: MUST hold the documented W code range 0..31 AND dyad()'s
                               // round-bump max of 16. ap_int<5> (-16..15) would WRAP a W-code exponent of
                               // 16..31 to negative -> garbage barrel-shift -> silent corruption. 6 bits
                               // (-32..31) covers both. (DSP=0 unaffected; one extra exp bit only.)
typedef ap_int<48>   acc_t;    // y accumulator (DSP48-width signed)
typedef ap_uint<512> wide_t;   // 16 x int32 packed AXI word

// flat int32 DDR buffer layout (probe-local): [ Wcode (O*I) | x (N*I) | bias (O) | y (N*O) ]
// Wcode is one int8 byte code per weight, staged one-code-per-int32 lane for a clean self-contained probe.
static const int OFF_WC   = 0;
static const int OFF_X    = OFF_WC   + FW_O * FW_I;     // = 589824 (divisible by 16)
static const int OFF_BIAS = OFF_X    + FW_N * FW_I;     // = 602112
static const int OFF_Y    = OFF_BIAS + FW_O;            // y region base (int32 units)
static const int OFF_WC_W = OFF_WC / 16;                // Wcode base in 512-bit (16-int32) words = 0
static const int OFF_Y_W  = OFF_Y / 16;                 // y region base in 512-bit words

// dyadic-quantize a FRESH activation to (sign, exponent): round(log2|v|) via MSB + next bit
// (round-to-nearest). Identical to the proven forge_wgrad_wide quantizer.
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

// decode the PRE-DYADIC weight byte code (no priority encoder, no quantize): sign / nonzero / exponent.
static inline void wdecode(int32_t code, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<8> c = (ap_uint<8>)(code & 0xFF);
    sgn = c[7];                                  // bit7 = sign
    nz  = c[5];                                  // bit5 = nonzero flag
    e   = (exp_t)(c & (ap_uint<8>)0x1F);         // bits4..0 = exponent
}

static inline void burst_load(int32_t *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)buf[off + i];
    }
}

void forge_fwd_wide(int32_t *buffer, wide_t *w_wide, wide_t *y_wide) {
#pragma HLS INTERFACE m_axi port=buffer max_widen_bitwidth=512 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=w_wide offset=slave bundle=gmemw
#pragma HLS INTERFACE m_axi port=y_wide offset=slave bundle=gmemy
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=w_wide
#pragma HLS INTERFACE s_axilite port=y_wide
#pragma HLS INTERFACE s_axilite port=return

    // --- Stage the fresh activations on-chip and dyadic-quantize them ONCE (q(x) = ±2^ex). ---
    // Partition on the i (contraction) dim by FW_N so the i-reduction reads N microbatch lanes/cycle.
    static data_t xx[FW_N][FW_I];
#pragma HLS ARRAY_PARTITION variable=xx complete dim=1
    burst_load(buffer, OFF_X, &xx[0][0], FW_N * FW_I);

    static exp_t xE[FW_N][FW_I]; static bool xS[FW_N][FW_I]; static bool xNZ[FW_N][FW_I];
#pragma HLS ARRAY_PARTITION variable=xE  complete dim=1
#pragma HLS ARRAY_PARTITION variable=xS  complete dim=1
#pragma HLS ARRAY_PARTITION variable=xNZ complete dim=1
QX: for (int n = 0; n < FW_N; n++)
        for (int i = 0; i < FW_I; i++) {
#pragma HLS PIPELINE II=1
            dyad(xx[n][i], xS[n][i], xE[n][i], xNZ[n][i]);
        }

    // --- WIDE-BURST load the PRE-DYADIC weight byte codes on-chip (NO quantize, NO QW pass). ---
    // 16 int32 codes per 512-bit word, II=1 -> 589,824/16 = 36,864 wide reads. Stored as int32 codes;
    // decoded INLINE in the reduction. Partition on the o (output) dim by FW_OT so the i-reduction's OT
    // unrolled output-lanes each read W[ot+r][i] from a distinct bank (required for II=1).
    static int32_t wc[FW_O][FW_I];
#pragma HLS ARRAY_PARTITION variable=wc cyclic factor=FW_OT dim=1
LDW: for (int o = 0; o < FW_O; o++) {
        for (int wblk = 0; wblk < FW_I / 16; wblk++) {
#pragma HLS PIPELINE II=1
            wide_t word = w_wide[OFF_WC_W + (o * FW_I) / 16 + wblk];
            for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                wc[o][wblk * 16 + k] = (int32_t)(ap_int<32>)word.range(32 * k + 31, 32 * k);
            }
        }
    }

    // --- Output-stationary tiled GEMM: hold a y-tile (OT rows x N microbatch) on-chip, STREAM i. ---
    // The OT*N inner tile is the unrolled PE lane array (area+timing live here). Tile-count + i-reduction
    // are NEVER unrolled (the throughput-sweep lesson). W decoded inline per lane (cheap logic).
OTILE: for (int ot = 0; ot < FW_O; ot += FW_OT) {
        acc_t acc[FW_OT][FW_N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
        // seed with bias (per output row, broadcast across the N microbatch)
        for (int r = 0; r < FW_OT; r++) {
#pragma HLS UNROLL
            acc_t b = (acc_t)(int32_t)buffer[OFF_BIAS + ot + r];
            for (int n = 0; n < FW_N; n++) {
#pragma HLS UNROLL
                acc[r][n] = b;
            }
        }
IRED:   for (int i = 0; i < FW_I; i++) {
#pragma HLS PIPELINE II=1
            for (int r = 0; r < FW_OT; r++) {
#pragma HLS UNROLL
                bool sw, nzw; exp_t ew;
                wdecode(wc[ot + r][i], sw, ew, nzw);      // inline decode of the pre-dyadic code
                for (int n = 0; n < FW_N; n++) {
#pragma HLS UNROLL
                    // dyadic MAC: both operands dyadic -> product = ±2^(ex+ew); one constant barrel shift
                    // + signed accumulate, NO multiply (identical PE to forge_wgrad_wide).
                    exp_t ex = xE[n][i];
                    ap_uint<6> e = (ap_uint<6>)ew + (ap_uint<6>)ex;
                    acc_t term = (acc_t)1 << e;
                    bool neg = sw ^ xS[n][i];
                    bool nz = nzw & xNZ[n][i];
                    if (nz) acc[r][n] += neg ? (acc_t)(-term) : term;
                }
            }
        }
        // --- WIDE-BURST writeback: pack 16 contiguous int32 outputs (along o) per 512-bit word. ---
        // For a fixed token n, OT=32 output rows = FW_OT/16 = 2 wide words. N(16) tokens x 2 = 32 wide
        // writes/tile (II=1) vs 512 scalar. y index for (n, o=ot+r): n*O + o; a wide word covers a run of
        // 16 consecutive o, so we pack acc[w*16+k][n] (16 output rows, fixed token) into one word.
        // n*O (=n*768) is divisible by 16 and ot is a multiple of 32 -> exact wide indexing.
WB:     for (int n = 0; n < FW_N; n++) {
            for (int w = 0; w < FW_OT / 16; w++) {
#pragma HLS PIPELINE II=1
                wide_t word;
                for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                    word.range(32 * k + 31, 32 * k) = (ap_uint<32>)acc[w * 16 + k][n];
                }
                int widx = OFF_Y_W + (n * FW_O + ot + w * 16) / 16;
                y_wide[widx] = word;
            }
        }
    }
}
