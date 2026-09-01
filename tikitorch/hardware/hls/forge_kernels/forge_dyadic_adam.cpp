// forge_dyadic_adam.cpp — DYADIC-ADAM optimizer update for the FPGA training rig.
//
// The multiplier-free adaptive optimizer (measured to beat AdamW). For each weight param p we do a
// read-modify-write over a contiguous region of P = 768*768 params, holding int32 state in DDR:
//
//   g    = grad[p]
//   absg = |g|
//   m    = m - (m>>3) + (g>>3)        // EMA, beta = 0.875 = 1 - 2^-3   (first moment)
//   v    = v - (v>>3) + (absg>>3)     // EMA of |g|, beta = 0.875        (adaptive scale)
//   se   = round(log2(v))             // priority-encoder MSB + round-to-nearest pow2 (dyad)
//   step = m >> se                    // arithmetic barrel shift -> per-param 1/|g|_EMA as a pow2
//   w[p] = w[p] - (step >> LR_SHIFT)  // lr = 2^-LR_SHIFT (a shift)
//
// EVERYTHING is shift + add + compare. NO multiply, NO DSP multiplier (DSP48 may serve only as a
// signed adder/accumulator — the "DSP-as-adder" lever). This mirrors the morph signsgd_region_wide
// region-RMW idiom: wide 512-bit burst load of a region tile -> per-lane update -> wide store, with
// the region loop PIPELINEd II=1.
//
// Buffer layout (flat int32 DDR): [ W (P) | grad (P) | m (P) | v (P) ].
// m, v are int32 optimizer state in the buffer (read-modify-written in place); grad is read-only.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_dyadic_adam.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config: one 768x768 weight region ----
#define DA_O        768                  // output features (rows of W)
#define DA_I        768                  // input  features (cols of W)
#define DA_P        (DA_O * DA_I)        // params in the region = 589824 (a multiple of 16)
#define DA_LR_SHIFT 12                   // learning rate = 2^-12 (a shift; tune per-rig)

typedef ap_int<512> wide_t;              // 512-bit DDR word = 16 lanes of int32
typedef ap_uint<6>  exp_t;               // dyadic exponent 0..32 (msb position of a 32-bit magnitude,
                                         // UNSIGNED so the round-to-nearest bump to 32 and the
                                         // resulting right-shift amount are always non-negative)

// flat int32 DDR buffer layout (probe-local): [ W (P) | grad (P) | m (P) | v (P) ]
static const int OFF_W = 0;
static const int OFF_G = OFF_W + DA_P;
static const int OFF_M = OFF_G + DA_P;
static const int OFF_V = OFF_M + DA_P;

// dyadic exponent of an int32 magnitude: round(log2|v|) via MSB + the next bit (round-to-nearest).
// Returns 0 for v==0 (a zero-scale param contributes no step beyond the raw m).
static inline exp_t dyad_exp(ap_int<32> v) {
#pragma HLS INLINE
    ap_uint<32> a = (v < 0) ? (ap_uint<32>)(-v) : (ap_uint<32>)v;
    exp_t msb = 0;
PRIO: for (int b = 31; b >= 1; b--) {            // priority encoder: highest set bit
#pragma HLS UNROLL
        if (a[b]) { msb = b; break; }
    }
    // round-to-nearest power of two: bump if the bit below the MSB is set.
    // Clamp the bumped exponent to 31 so the subsequent (mn >> se) shift amount stays
    // strictly < 32 — a >> by the full word width is UB on the scalar int32 #else path and
    // adds nothing for the optimizer step (dividing m by 2^31 already drives step to 0/-1).
    if (msb >= 1 && a[msb - 1] && msb < 31) msb = msb + 1;
    return msb;
}

void forge_dyadic_adam(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t *wide_buf = reinterpret_cast<wide_t *>(buffer);
    const int n_wide  = DA_P >> 4;          // 512-bit words spanning one P-region
    const int w_woff  = OFF_W >> 4;
    const int g_woff  = OFF_G >> 4;
    const int m_woff  = OFF_M >> 4;
    const int v_woff  = OFF_V >> 4;

REGION: for (int wi = 0; wi < n_wide; wi++) {
#pragma HLS PIPELINE II=1
        wide_t w_word = wide_buf[w_woff + wi];
        wide_t g_word = wide_buf[g_woff + wi];
        wide_t m_word = wide_buf[m_woff + wi];
        wide_t v_word = wide_buf[v_woff + wi];

        wide_t w_new = w_word;
        wide_t m_new = m_word;
        wide_t v_new = v_word;

LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int lo = lane * 32, hi = lo + 31;
            ap_int<32> w = (ap_int<32>) w_word.range(hi, lo);
            ap_int<32> g = (ap_int<32>) g_word.range(hi, lo);
            ap_int<32> m = (ap_int<32>) m_word.range(hi, lo);
            ap_int<32> v = (ap_int<32>) v_word.range(hi, lo);
            ap_int<32> absg = (g < 0) ? (ap_int<32>)(-g) : g;

            // EMA updates (beta = 0.875 = 1 - 2^-3): m,v stay int32 (shifts keep them bounded).
            ap_int<32> mn = m - (m >> 3) + (g    >> 3);
            ap_int<32> vn = v - (v >> 3) + (absg >> 3);

            // adaptive per-param scale: divide m by 2^round(log2 v) == arithmetic right shift by se.
            exp_t se = dyad_exp(vn);
            ap_int<32> step = mn >> se;                 // arithmetic shift preserves m's sign

            // weight update: w <- w - (step >> LR_SHIFT); lr = 2^-LR_SHIFT.
            ap_int<32> dw = step >> DA_LR_SHIFT;
            ap_int<32> wn = w - dw;

            w_new.range(hi, lo) = (ap_uint<32>) wn;
            m_new.range(hi, lo) = (ap_uint<32>) mn;
            v_new.range(hi, lo) = (ap_uint<32>) vn;
        }

        wide_buf[w_woff + wi] = w_new;
        wide_buf[m_woff + wi] = m_new;
        wide_buf[v_woff + wi] = v_new;
    }
#else
    // ---- C-sim fallback (host build): scalar reference of the same recipe ----
    for (int p = 0; p < DA_P; p++) {
        int32_t w = buffer[OFF_W + p];
        int32_t g = buffer[OFF_G + p];
        int32_t m = buffer[OFF_M + p];
        int32_t v = buffer[OFF_V + p];
        int32_t absg = (g < 0) ? -g : g;
        int32_t mn = m - (m >> 3) + (g >> 3);
        int32_t vn = v - (v >> 3) + (absg >> 3);
        exp_t se = dyad_exp((ap_int<32>)vn);
        int32_t step = mn >> (int)se;
        int32_t dw = step >> DA_LR_SHIFT;
        buffer[OFF_W + p] = w - dw;
        buffer[OFF_M + p] = mn;
        buffer[OFF_V + p] = vn;
    }
#endif
}
