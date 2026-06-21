// forge_layernorm_bwd.cpp — BACKWARD of the multiplier-free (dyadic) LayerNorm for the S1
// FPGA transformer-training rig. Companion to the forward _FineDyadicNorm / _fine_dyadic_rescale
// of scripts/charlm_mixer.py (the validated-to-2.481 ALGORITHM AUTHORITY). Matches that backward
// EXACTLY and the morph batch_shift backward idiom (forge_train_step_morph_lm.cpp:516).
//
// FORWARD recap (what this is the gradient of), per token n, channel d:
//     c[n][d]     = x[n][d] - mean_d(x[n])                    (mean = sum >> LOG2_D, NO divider)
//     c_hat[n][d] = c[n][d] * invq[n]                         (invq = dyadic 1/scale; c_hat is DYADIC)
//     y[n][d]     = gamma_dyadic[d] * c_hat[n][d] + beta[d]   (gamma constrained to ±2^k = a SHIFT)
//   ("gamma_dyadic" is the per-channel learnable scale, constrained to ±2^k like _DyadicGammaNorm;
//    _FineDyadicNorm itself drops it -> pass a unit code s={+,2^0} and the gamma path is a no-op.)
//
// BACKWARD contract (the pieces the rig needs; ALL shift-add, NO multiply, DSP=0):
//   Given gY[n][d] (the incoming output grad — a FRESH grad, kept FULL-precision / NOT quantized,
//   matching charlm's gx=gO@Wq where only W is dyadic), the SAVED per-channel dyadic gamma scale
//   s[d] (= gamma_dyadic[d], a ±2^k code), the SAVED normalized c_hat[n][d] (a ±2^k code), and the
//   SAVED per-token fine-dyadic rescale factor invq_n = (nq_n/4)*2^e_n (the _fine_dyadic_rescale
//   1/scale, mbits=2; OFF_RSC, emitted by forge_layernorm.cpp):
//
//   (1a) gamma SCALE is a STRAIGHT-THROUGH estimator -> the grad FLOWS as gY shifted by the SAME
//        dyadic scale s (pass-through, NOT re-derived):
//             g_chat[n][d] = gY[n][d] * gamma_dyadic[d]  =  ± ( gY[n][d] << s_exp[d] )
//   (1b) RESCALE backward (STE: the 1/scale invq is detached/constant w.r.t. c in _fine_dyadic_rescale)
//        -> the grad is multiplied by the SAME per-token invq_n, applied as the SAME shift-add the
//        forward used (apply_invq; NO multiply):
//             g_c[n][d] = g_chat[n][d] * invq_n  =  ( g_chat * nq_n / 4 ) << e_n (e<0 => right shift)
//   (2)  mean-subtract backward (morph:516 idiom): subtract the per-token mean of g_c:
//             gx[n][d] = g_c[n][d] - ( sum_d g_c[n][:] >> LOG2_D )
//   (3)  gamma-grad (per channel): g_gamma[d] = sum_n gY[n][d] * c_hat[n][d]. c_hat is DYADIC ->
//        this is gY shifted by c_hat's exponent (sign-folded), ACCUMULATED over n. NO multiply:
//             g_gamma[d] = sum_n  ± ( gY[n][d] << chat_exp[n][d] )
//   (4)  beta-grad (per channel): the additive bias grad is just the column sum:
//             g_beta[d] = sum_n gY[n][d]
//
//   NOTE: dropping (1b) — the rescale scale in the gx path — was the original bug; gx then matched a
//   gamma-only norm, NOT _fine_dyadic_rescale. The per-token invq is the dominant magnitude of gx, so
//   omitting it is not a small error. (g_gamma already includes the scale via the saved c_hat.)
//
// Every operation is sign / shift / signed-add / compare. DSP=0 expected (no DSP MULTIPLIER; the
// only allowed multiply in the whole rig is the attention soft-blend, which lives in forge_attn_bwd,
// NOT here).
//
// dyad() / wdecode() are IDENTICAL to forge_fwd_wide.cpp (priority-encode MSB, round-to-nearest by
// bumping when the bit below the MSB is set, sign separate, zero-guarded) so the int64 golden in the
// testbench is BIT-EXACT to the kernel. The saved s[d] and c_hat[n][d] arrive PRE-DYADIC as the same
// 1-byte (sign,nonzero,exp) code that forge_fwd_wide's W uses (the forward kernel emits them).
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csim + csynth via synth_layernorm_bwd.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- S1 LayerNorm config (d_model=128; keeps the d768-general defines, S1 shrinks them) ----
#define LN_D       128     // d_model (channels). LOG2_D = 7 -> mean = sum>>7, NO divider.
#define LN_LOG2_D  7       // log2(LN_D); 1<<LN_LOG2_D == LN_D
#define LN_N       16      // microbatch tokens (= N*L flattened rows handed to the norm)
#define LN_DT      16      // channel-tile width (= unrolled lane count for the column reductions);
                           // divides LN_D and is 16-wide (512-bit burst friendly)

typedef ap_int<32>   data_t;   // gY values: FULL-precision grad (NOT dyadic-quantized), int32 range
typedef ap_int<6>    exp_t;    // dyadic exponent: holds the documented code range 0..31 AND dyad()'s
                               // round-bump max (matches forge_fwd_wide's exp_t rationale).
typedef ap_int<48>   acc_t;    // accumulator (DSP48-width signed): column sums of shifted grads.
typedef ap_uint<512> wide_t;   // 16 x int32 packed AXI word

// ---- flat int32 DDR buffer layout (probe-local; self-contained, documented) ----
//   [ gY (N*D) | s_code (D) | chat_code (N*D) | rsc (N*2) | gx (N*D) | g_gamma (D) | g_beta (D) ]
// gY      : incoming output grad, one full int32 per element (NOT quantized).
// s_code  : per-channel SAVED dyadic gamma scale, one byte code per channel (sign/nz/exp).
// chat_code: per-element SAVED normalized c_hat, one byte code per element (sign/nz/exp; c_hat dyadic).
// rsc     : per-token SAVED fine-dyadic RESCALE factor invq = (nq/4) * 2^e  (the _fine_dyadic_rescale
//           1/scale, mbits=2). [2*n]=e (signed scale exponent, the forward's stats[2n+1]),
//           [2*n+1]=nq (2-bit mantissa in {4..7}, the forward's nq). The forward kernel emits these;
//           the input-grad path MUST apply this scale (STE: scale is constant w.r.t. c) — see (1b).
// gx      : output input-grad  (N*D int32).
// g_gamma : output gamma-grad  (D   int32).
// g_beta  : output beta-grad   (D   int32).
static const int OFF_GY    = 0;
static const int OFF_S     = OFF_GY    + LN_N * LN_D;          // 2048
static const int OFF_CHAT  = OFF_S     + LN_D;                 // 2176
static const int OFF_RSC   = OFF_CHAT  + LN_N * LN_D;          // 4224  (per-token e,nq)
static const int OFF_GX    = OFF_RSC   + LN_N * 2;             // 4256
static const int OFF_GGAM  = OFF_GX    + LN_N * LN_D;          // 6304
static const int OFF_GBETA = OFF_GGAM  + LN_D;                 // 6432
static const int BUFLEN    = OFF_GBETA + LN_D;                 // 6560 (divisible by 16)

#define LN_MBITS  2       // fine-dyadic mantissa bits (charlm _NORM_MBITS=2); matches forge_layernorm.cpp

// decode a PRE-DYADIC byte code (no priority encoder, no quantize): sign / nonzero / exponent.
// IDENTICAL to forge_fwd_wide.cpp::wdecode (the saved-scale / saved-c_hat contract).
static inline void wdecode(int32_t code, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<8> c = (ap_uint<8>)(code & 0xFF);
    sgn = c[7];                                  // bit7 = sign
    nz  = c[5];                                  // bit5 = nonzero flag
    e   = (exp_t)(c & (ap_uint<8>)0x1F);         // bits4..0 = exponent
}

// signed barrel shift of a FULL-precision grad by a dyadic exponent, with sign fold. This is the
// "× ±2^e" of a dyadic-by-full product == ONE constant barrel shift + a sign select. NO multiply.
static inline acc_t shift_term(data_t g, bool neg, exp_t e, bool nz) {
#pragma HLS INLINE
    if (!nz) return (acc_t)0;
    acc_t t = (acc_t)g << (int)e;                 // g * 2^e (one barrel shift; e cast to int = unambiguous)
    return neg ? (acc_t)(-t) : t;
}

// Apply the per-token fine-dyadic RESCALE factor invq = (nq/4)*2^e to a grad value, as SHIFT-ADD only
// (NO multiply) — BIT-IDENTICAL to forge_layernorm.cpp's forward rescale (cn = v<<MBITS + (nq-4) copies;
// r = cn>>MBITS; then <<e if e>=0 else >>-e). Under the _fine_dyadic_rescale STE the scale is constant
// w.r.t. c, so the input-grad flows through it as this same multiply-free scalar product.
static inline acc_t apply_invq(acc_t v, int e, int nq) {
#pragma HLS INLINE
    int nqm4 = nq - (1 << LN_MBITS);                  // 0..3 extra copies of v beyond the v<<MBITS base
    acc_t cn = v << LN_MBITS;
    if (nqm4 >= 1) cn += v;
    if (nqm4 >= 2) cn += v;
    if (nqm4 >= 3) cn += v;
    acc_t r = cn >> LN_MBITS;                          // = v * (nq/4)
    if (e >= 0) r = r << e;
    else        r = r >> (-e);
    return r;
}

static inline int32_t sat32(acc_t v) {
#pragma HLS INLINE
    if (v >  (acc_t)2147483647LL) return  2147483647;
    if (v < (acc_t)(-2147483648LL)) return (int32_t)(-2147483648LL);
    return (int32_t)v;
}

static inline void burst_load_words(int32_t *buf, int off, int32_t *dst, int n) {
#pragma HLS INLINE off
LDW: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = buf[off + i];
    }
}

void forge_layernorm_bwd(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // --- Stage gY and the two saved dyadic-code regions on-chip. Partition on the channel (d)
    //     dim by LN_DT so the LN_DT unrolled lanes of every reduction read distinct banks. ---
    static data_t  gY[LN_N][LN_D];
    static int32_t sc[LN_D];               // per-channel gamma scale code
    static int32_t cc[LN_N][LN_D];         // per-element c_hat code
#pragma HLS ARRAY_PARTITION variable=gY cyclic factor=LN_DT dim=2
#pragma HLS ARRAY_PARTITION variable=cc cyclic factor=LN_DT dim=2
#pragma HLS ARRAY_PARTITION variable=sc cyclic factor=LN_DT dim=1

LD_GY: for (int n = 0; n < LN_N; n++) {
        for (int d = 0; d < LN_D; d++) {
#pragma HLS PIPELINE II=1
            gY[n][d] = (data_t)buffer[OFF_GY + n * LN_D + d];
            cc[n][d] = buffer[OFF_CHAT + n * LN_D + d];
        }
    }
    burst_load_words(buffer, OFF_S, sc, LN_D);

    // Per-token fine-dyadic rescale factors (e, nq) saved by the forward; needed by the gx path (1b).
    static int32_t rsc[LN_N * 2];
    burst_load_words(buffer, OFF_RSC, rsc, LN_N * 2);

    // Decode the per-channel gamma scale code once (s_exp[d], s_sgn[d], s_nz[d]).
    static exp_t s_exp[LN_D]; static bool s_sgn[LN_D]; static bool s_nz[LN_D];
#pragma HLS ARRAY_PARTITION variable=s_exp cyclic factor=LN_DT dim=1
#pragma HLS ARRAY_PARTITION variable=s_sgn cyclic factor=LN_DT dim=1
#pragma HLS ARRAY_PARTITION variable=s_nz  cyclic factor=LN_DT dim=1
DEC_S: for (int d = 0; d < LN_D; d++) {
#pragma HLS PIPELINE II=1
        wdecode(sc[d], s_sgn[d], s_exp[d], s_nz[d]);
    }

    // =========================================================================
    // (1a) STE gamma pass-through + (1b) fine-dyadic rescale STE + (2) mean-subtract backward -> gx,
    //      ROW (token) at a time. Matches _fine_dyadic_rescale + subtract-mean backward EXACTLY:
    //        g_chat[d] = gY[d] shifted by the SAVED dyadic gamma scale s[d]   (STE pass-through)
    //        g_c[d]    = g_chat[d] * invq_n                                    (rescale STE, scale const)
    //        gx[d]     = g_c[d] - (sum_d g_c >> LOG2_D)                        (subtract-mean backward)
    //      invq_n = (nq_n/4)*2^e_n is a per-token CONSTANT (saved by the forward in OFF_RSC). We form
    //      g_c[d] = apply_invq(g_chat[d]) (shift-add, NO multiply) in the lane stage, accumulate the
    //      mean OVER g_c, then subtract — BIT-FAITHFUL to "g_c = g_chat*invq then subtract-mean"
    //      (the rescale rounding happens per-value exactly as the forward's, before the reduction).
    // =========================================================================
GX_ROW: for (int n = 0; n < LN_N; n++) {
        int e_n  = (int)rsc[2 * n];                  // saved scale exponent (signed)
        int nq_n = (int)rsc[2 * n + 1];              // saved 2-bit mantissa nq in {4..7}
        static acc_t g_row[LN_D];
#pragma HLS ARRAY_PARTITION variable=g_row cyclic factor=LN_DT dim=1
        acc_t gsum = 0;
        // post-STE grad g_c = invq * (gamma-shifted gY) + running row sum (carried add over tiles).
    G_STE: for (int d = 0; d < LN_D; d += LN_DT) {
#pragma HLS PIPELINE II=1
            acc_t tt[LN_DT];
#pragma HLS ARRAY_PARTITION variable=tt complete
        G_LANE: for (int u = 0; u < LN_DT; u++) {
#pragma HLS UNROLL
                acc_t gchat = shift_term(gY[n][d + u], s_sgn[d + u], s_exp[d + u], s_nz[d + u]);
                acc_t gd    = apply_invq(gchat, e_n, nq_n);   // g_c = g_chat * invq (shift-add)
                g_row[d + u] = gd;
                tt[u] = gd;
            }
            // balanced add-tree over the LN_DT lanes, then a single carried add into gsum.
        G_TREE: for (int st = 1; st < LN_DT; st <<= 1) {
#pragma HLS UNROLL
                for (int u = 0; u < LN_DT; u += (st << 1)) {
#pragma HLS UNROLL
                    tt[u] += tt[u + st];
                }
            }
            gsum += tt[0];
        }
        acc_t mean = gsum >> LN_LOG2_D;          // per-token mean of g_c (NO divider)
    G_SUB: for (int d = 0; d < LN_D; d++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_GX + n * LN_D + d] = sat32(g_row[d] - mean);
        }
    }

    // =========================================================================
    // (3) gamma-grad + (4) beta-grad — per-channel COLUMN reductions over the N tokens.
    //     g_gamma[d] = sum_n gY[n][d] * c_hat[n][d]   (c_hat dyadic -> gY << chat_exp, sign-folded)
    //     g_beta[d]  = sum_n gY[n][d]
    //     Output-stationary over a tile of LN_DT channels; STREAM the n-reduction (II=1).
    // =========================================================================
GAM_TILE: for (int dt = 0; dt < LN_D; dt += LN_DT) {
        acc_t accg[LN_DT];   // gamma-grad accumulators
        acc_t accb[LN_DT];   // beta-grad  accumulators
#pragma HLS ARRAY_PARTITION variable=accg complete
#pragma HLS ARRAY_PARTITION variable=accb complete
    GINIT: for (int u = 0; u < LN_DT; u++) {
#pragma HLS UNROLL
            accg[u] = 0; accb[u] = 0;
        }
    NRED: for (int n = 0; n < LN_N; n++) {
#pragma HLS PIPELINE II=1
        NLANE: for (int u = 0; u < LN_DT; u++) {
#pragma HLS UNROLL
                int d = dt + u;
                bool cs, cnz; exp_t ce;
                wdecode(cc[n][d], cs, ce, cnz);                 // inline decode of saved c_hat code
                data_t gv = gY[n][d];
                accg[u] += shift_term(gv, cs, ce, cnz);        // gY << chat_exp (sign-folded)
                accb[u] += (acc_t)gv;                          // plain column sum
            }
        }
    GWB: for (int u = 0; u < LN_DT; u++) {
#pragma HLS UNROLL
            buffer[OFF_GGAM  + dt + u] = sat32(accg[u]);
            buffer[OFF_GBETA + dt + u] = sat32(accb[u]);
        }
    }
}
