// forge_xgrad_wide.cpp — REALISTIC dyadic INPUT/ACTIVATION-GRAD GEMM for the FPGA transformer-training rig.
//
// The third GEMM of the mult-free backward pass (the critical-path blocker that completes the trio with the
// proven forward forge_fwd_wide and weight-grad forge_wgrad_wide). Computes the activation gradient that is
// propagated to the previous layer:
//   gx[n][i] = sum_o gO[n][o] * q(W[o][i])     (the charlm_mixer authority:  gx = gO @ q(W) ,  W^T gO)
//     n in N=16 microbatch tokens, o,i in D=768 d_model (general; S1 shrinks D=128 via the defines below).
//
// CRUCIAL ASYMMETRY vs forge_wgrad_wide (and the reason this is its OWN kernel, NOT a transpose of fwd):
//   - gO is a FRESH activation grad and is held FULL int16 — it is NOT dyadic-quantized (matches charlm_mixer
//     line 76: `gx = (gO @ Wq)` — gO untouched, only W is q()'d). So the gradient keeps its full mantissa.
//   - W ARRIVES PRE-DYADIC, same int8 (sign,nz,exp) code + wdecode() as forge_fwd_wide (the real rig stores
//     every weight as a 1-byte code in DDR; brought on-chip by a 512-bit WIDE BURST, decoded INLINE — no QW
//     pass, no priority encoder on W).
//   Therefore each MAC term is a product of a FULL int (gO) by a dyadic (±2^e):  a SINGLE BARREL SHIFT of the
//   full gradient by the weight exponent, sign = sign_W xor sign_gO, signed-accumulate.  NO multiply, DSP=0.
//     term = (gO[n][o] << exp_W[o][i]) with the sign folded in     <-- the ONE structural diff from wgrad,
//   where wgrad's term was 2^(eo+ei) (both operands dyadic). Here ONE operand is dyadic so we shift the FULL
//   int instead of shifting a 1. (The single soft-blend multiply of the rig lives in forge_attn, NOT here.)
//
// REDUCTION DIRECTION (transposed-W access vs the forward):
//   forward  y[n][o] = sum_i x[n][i] * W[o][i]   reduces over i, W read row-major as W[o][i].
//   xgrad    gx[n][i] = sum_o gO[n][o] * W[o][i] reduces over o, so for a fixed output column i we walk W
//   DOWN the o (row) dimension — a transposed read of the SAME W[o][i] store. We tile output-stationary on
//   (n,i): hold a gx-tile (IT input-cols x N microbatch) in on-chip int48 accumulators and STREAM the o
//   reduction (the contraction d=768, PIPELINE II=1). The IT*N inner tile is the unrolled PE lane array
//   (IT=32 x N=16 = 512 lanes — the proven ~512-lane budget); we unroll ONLY the lane tile, never the
//   tile-count loop nor the o-reduction. gO is held FULL int16 on-chip (partitioned on n so all N lanes read
//   in parallel). WIDE ap_uint<512> writeback of gx[N][I] (16 contiguous int32 along i per 512-bit word).
//
// W byte encoding (IDENTICAL to forge_fwd_wide; documented + self-contained):
//   int8 code, per weight (one code per int32 lane in the probe buffer; the real rig packs 4/word):
//     bit 7      = sign     (1 = negative)
//     bit 5      = nonzero  (1 = weight is a real ±2^e ; 0 = weight is exactly 0, term skipped)
//     bits 4..0  = exponent e in 0..31 (the dyadic magnitude shift; |W| = 2^e)
//     bit 6      = reserved (0)
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csim+csynth via synth_xgrad_wide.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768 general; S1 uses a subset via the same defines) ----
#define XG_O       768     // output features (rows of W) — d_model  (the contraction, reduced over)
#define XG_I       768     // input  features (cols of W) — d_model  (the gx output feature)
#define XG_N       16      // microbatch tokens back-prop'd together
#define XG_IT      32      // input-tile cols held on-chip (= unrolled PE input-lane count)
                           // IT(32) * N(16) = 512 PE lanes = the proven lane budget

typedef ap_int<16>   data_t;   // gO values (FULL int16 — NOT quantized; the morph data_t range)
typedef ap_int<6>    exp_t;    // dyadic exponent: MUST hold the W code range 0..31. ap_int<5> (-16..15)
                               // would WRAP a code exponent of 16..31 to negative -> garbage barrel shift.
                               // 6 bits (-32..31) covers it. (DSP=0 unaffected; one extra exp bit only.)
typedef ap_int<48>   acc_t;    // gx accumulator (DSP48-width signed). gO int16 << up to 2^31 then summed
                               // over o=768 stays well inside 48 bits at deployed range; bounds-checked in TB.
typedef ap_uint<512> wide_t;   // 16 x int32 packed AXI word

// flat int32 DDR buffer layout (probe-local): [ gO (N*O) | Wcode (O*I) | gx (N*I) ]
// gO is read scalar (full int16); Wcode is one int8 code per int32 lane (wide-burst); gx is wide-written.
static const int OFF_GO   = 0;
static const int OFF_WC   = OFF_GO + XG_N * XG_O;       // = 12288 int32  (divisible by 16)
static const int OFF_GX   = OFF_WC + XG_O * XG_I;       // = 602112 int32 (gx region base, divisible by 16)
static const int OFF_WC_W = OFF_WC / 16;               // Wcode base in 512-bit (16-int32) words = 768
static const int OFF_GX_W = OFF_GX / 16;               // gx region base in 512-bit words

// decode the PRE-DYADIC weight byte code (no priority encoder, no quantize): sign / nonzero / exponent.
// IDENTICAL to forge_fwd_wide::wdecode — must be bit-for-bit the same as the golden reference.
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

void forge_xgrad_wide(int32_t *buffer, wide_t *w_wide, wide_t *gx_wide) {
#pragma HLS INTERFACE m_axi port=buffer  max_widen_bitwidth=512 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=w_wide  offset=slave bundle=gmemw
#pragma HLS INTERFACE m_axi port=gx_wide offset=slave bundle=gmemx
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=w_wide
#pragma HLS INTERFACE s_axilite port=gx_wide
#pragma HLS INTERFACE s_axilite port=return

    // --- Stage the FRESH activation grads gO on-chip as FULL int16 (NO quantize — gO keeps its mantissa). ---
    // Partition on the o (contraction) dim by XG_IT so the o-reduction's IT unrolled input-lanes can each
    // read the SAME gO[n][o] (broadcast across i) — gO is indexed [n][o], so partition on n (dim 1) so all
    // N microbatch lanes read in parallel; o is streamed scalar in the reduction.
    static data_t go[XG_N][XG_O];
#pragma HLS ARRAY_PARTITION variable=go complete dim=1
    burst_load(buffer, OFF_GO, &go[0][0], XG_N * XG_O);

    // --- WIDE-BURST load the PRE-DYADIC weight byte codes on-chip (NO quantize, NO QW pass). ---
    // 16 int32 codes per 512-bit word, II=1 -> 589,824/16 = 36,864 wide reads. Decoded INLINE in the
    // reduction. We reduce over o (rows of W) for a fixed i (col of W), so the IT unrolled input-lanes read
    // W[o][it+c] for contiguous c at a fixed o -> partition W on the i (col) dim by XG_IT so {it..it+IT-1}
    // hit IT distinct banks (required for II=1 on the unrolled lane array).
    static int32_t wc[XG_O][XG_I];
#pragma HLS ARRAY_PARTITION variable=wc cyclic factor=XG_IT dim=2
LDW: for (int o = 0; o < XG_O; o++) {
        for (int wblk = 0; wblk < XG_I / 16; wblk++) {
#pragma HLS PIPELINE II=1
            wide_t word = w_wide[OFF_WC_W + (o * XG_I) / 16 + wblk];
            for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                wc[o][wblk * 16 + k] = (int32_t)(ap_int<32>)word.range(32 * k + 31, 32 * k);
            }
        }
    }

    // --- Output-stationary tiled GEMM: hold a gx-tile (IT input-cols x N microbatch) on-chip, STREAM o. ---
    // The IT*N inner tile is the unrolled PE lane array (area+timing live here). Tile-count + o-reduction are
    // NEVER unrolled (the throughput-sweep lesson). W decoded inline per lane (cheap logic).
ITILE: for (int it = 0; it < XG_I; it += XG_IT) {
        acc_t acc[XG_IT][XG_N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
        // seed at zero (gx has no bias — it is a pure gradient contraction)
        for (int c = 0; c < XG_IT; c++) {
#pragma HLS UNROLL
            for (int n = 0; n < XG_N; n++) {
#pragma HLS UNROLL
                acc[c][n] = 0;
            }
        }
ORED:   for (int o = 0; o < XG_O; o++) {
#pragma HLS PIPELINE II=1
            for (int c = 0; c < XG_IT; c++) {
#pragma HLS UNROLL
                bool sw, nzw; exp_t ew;
                wdecode(wc[o][it + c], sw, ew, nzw);      // inline decode of the pre-dyadic code W[o][it+c]
                for (int n = 0; n < XG_N; n++) {
#pragma HLS UNROLL
                    // ASYMMETRIC dyadic MAC: gO is FULL int16, W is dyadic ±2^ew -> the term is a SINGLE
                    // barrel SHIFT of the full gradient by the weight exponent, sign folded in. NO multiply.
                    //   term = (sign) * ( gO[n][o] << ew )
                    acc_t g = (acc_t)go[n][o];           // full int16 grad, sign-extended into the accumulator
                    acc_t shifted = g << (int)(ap_uint<6>)ew;   // barrel shift by the dyadic exponent
                    bool neg = sw;                       // W's sign flips the term; gO's sign is already in g
                    if (nzw) acc[c][n] += neg ? (acc_t)(-shifted) : shifted;
                }
            }
        }
        // --- WIDE-BURST writeback: pack 16 contiguous int32 outputs (along i) per 512-bit word. ---
        // gx index for (n, i=it+c): n*I + i; a wide word covers a run of 16 consecutive i, so we pack
        // acc[w*16+k][n] (16 input cols, fixed token) into one word. n*I (=n*768) is divisible by 16 and it
        // is a multiple of IT=32 -> exact wide indexing. IT=32 -> XG_IT/16 = 2 wide words per token row;
        // N(16) tokens x 2 = 32 wide writes/tile (II=1) vs 512 scalar.
WB:     for (int n = 0; n < XG_N; n++) {
            for (int w = 0; w < XG_IT / 16; w++) {
#pragma HLS PIPELINE II=1
                wide_t word;
                for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                    word.range(32 * k + 31, 32 * k) = (ap_uint<32>)acc[w * 16 + k][n];
                }
                int widx = OFF_GX_W + (n * XG_I + it + w * 16) / 16;
                gx_wide[widx] = word;
            }
        }
    }
}
