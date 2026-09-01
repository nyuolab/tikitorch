// forge_layernorm.cpp — multiplier-free FORWARD fine-dyadic LayerNorm for the S1 FPGA transformer block.
//
// Matches scripts/charlm_mixer.py _fine_dyadic_rescale (mbits=2) + the morph batch_shift_row
// (>>LOG2_D mean-subtract, NO divider), reproduced as an INTEGER-EXACT shift-add pipeline. ZERO
// multiplies on the datapath (DSP=0 expected): mean is a sum>>LOG2_D, meanabs is a sum>>LOG2_D, the
// 1/scale rescale is a 2-bit-mantissa dyadic (mq in {4,5,6,7}/4) applied as a SHIFT-ADD, the
// per-channel gamma is a PRE-DYADIC ±2^k code (a barrel shift), beta is an additive int16.
//
//   per token n over d = LN_D (=128, LOG2_D=7):
//     mean      = ( sum_d x[n][d] ) >> LOG2_D                          (no divider)
//     c[d]      = x[n][d] - mean
//     meanabs   = ( sum_d |c[d]| ) >> LOG2_D       (== scale, the meanabs normalizer, >=1 floored)
//     (e, nq)   = fine-dyadic 1/meanabs : 2-bit mantissa nq in {4..7}, exponent e  (see fdyad())
//     r[d]      = c[d] rescaled by (nq * 2^e) / 4  -> shift-add of c[d]<<2 .. plus (nq-4) copies
//     y[d]      = ( r[d] dyadic-* gamma[d] )  + beta[d]                (gamma = a shift; DSP=0)
//
// The math authority is charlm_mixer.py _fine_dyadic_rescale (validated to bpb 2.481). The float
// reference there does inv=1/scale; e=floor(log2(inv)); m=inv/2^e; mq=round(m*4)/4; invq=mq*2^e;
// y=c*invq. We reproduce invq EXACTLY in integers (no float, no divide on the datapath) so the kernel
// and the testbench's int64 golden are BIT-IDENTICAL — see fdyad() for the derivation.
//
// The per-token fine-dyadic rescale factor (e, nq) is SAVED to the buffer (OFF_STATS) because the
// backward (forge_layernorm_bwd) needs the EXACT rounded invq=(nq/4)*2^e (it cannot recompute the
// dyadic rounding cheaply). The backward computes its own mean-subtract internally, so mean is NOT saved.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csim+csynth via synth_layernorm.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- S1 config (d768-general defines; S1 shrinks D to 128) ----
#define LN_D      128     // model dim (per-token normalization length)
#define LN_LOG2_D 7       // log2(LN_D): mean = sum >> 7, NO divider  (LN_D MUST == 1<<LN_LOG2_D)
#define LN_N      16      // microbatch tokens normalized together
#define LN_MBITS  2       // fine-dyadic mantissa bits (charlm _NORM_MBITS=2): step = 1<<MBITS = 4

typedef ap_int<16>  data_t;   // x / y values (int16, the morph data_t range)
typedef ap_int<6>   exp_t;    // dyadic exponent (gamma code 0..31 + slack; matches forge_fwd_wide)
typedef ap_int<48>  acc_t;    // sums / rescale accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (probe-local; self-contained):
//   [ X (N*D) | GAMMA (D codes) | BETA (D) | Y (N*D) | STATS (N*2: scale_e,nq per token) ]
// STATS is the per-token fine-dyadic RESCALE factor invq=(nq/4)*2^e — EXACTLY the contract the
// backward consumes (forge_layernorm_bwd.cpp OFF_RSC: [2n]=e, [2n+1]=nq). The backward reconstructs
// invq from (e,nq) for its STE input-grad path; it does NOT need the forward's mean (it computes the
// backward mean-subtract of g_c internally). So we emit (e, nq), NOT (mean, e).
static const int OFF_X     = 0;
static const int OFF_GAMMA = OFF_X     + LN_N * LN_D;     // per-CHANNEL dyadic gamma byte-codes
static const int OFF_BETA  = OFF_GAMMA + LN_D;            // per-CHANNEL additive int16 beta
static const int OFF_Y     = OFF_BETA  + LN_D;            // normalized output (N*D)
static const int OFF_STATS = OFF_Y     + LN_N * LN_D;     // per token: [2*n]=scale exp e, [2*n+1]=nq

// decode a PRE-DYADIC ±2^k code (sign/nonzero/exponent), identical to forge_fwd_wide.wdecode():
//   bit7=sign, bit5=nonzero, bits4..0=exponent. Used for the per-channel gamma scale.
static inline void wdecode(int32_t code, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<8> c = (ap_uint<8>)(code & 0xFF);
    sgn = c[7];
    nz  = c[5];
    e   = (exp_t)(c & (ap_uint<8>)0x1F);
}

// fine-dyadic 1/scale -> (e, nq): a 2-bit-mantissa dyadic value (nq/4) * 2^e == invq.
// EXACT integer reproduction of charlm _fine_dyadic_rescale (mbits=2), no float / no datapath divide:
//   scale (>=1) has MSB at position p  ->  2^p <= scale < 2^(p+1)  ->  inv=1/scale in (2^-(p+1), 2^-p].
//   e = floor(log2(inv)) = (scale == 2^p) ? -p : -(p+1)
//   nq = round(m*4) = round( inv * 4 * 2^-e ) = round( (4<<(-e)) / scale )      (round-half-up: +scale/2)
//   nq is clamped to [4,7] (the m in [1,2) -> 4/4..7/4 mantissa; an 8 would mean m rounded to 2.0,
//   folded back to nq=4 with e bumped +1, exactly as a binary mantissa carry).
// Returns e (the LEFT shift amount applied later) and nq. Caller forms r = (c*nq) shifted by e.
//
// NO DATAPATH DIVIDE: after the priority encode, 2^p <= scale < 2^(p+1) so the raw quotient
// (4<<nege)/scale lands in {4,5,6,7,8} ALWAYS (proven exhaustively). So nq (round-half-up) is the
// largest n in {4..8} with n*scale <= numer + scale/2. We find it by a BOUNDED, fully-unrolled
// compare over the 4 candidate increments (n*scale built by shift-add: 4*scale = scale<<2, then +scale
// per step). This is bit-IDENTICAL to round((4<<nege)/scale) but uses ZERO multiplies AND zero divides
// (a runtime integer divide is the one op no measured forge kernel uses; the candidate-compare keeps
// DSP=0 guaranteed and timing clean).
static inline void fdyad(acc_t scale, int &e_out, int &nq_out) {
#pragma HLS INLINE
    if (scale < 1) scale = 1;                       // floor: meanabs>=1 (matches +eps; avoids /0)
    // priority-encode MSB position p of scale
    int p = 0;
PRIO: for (int b = 47; b >= 1; b--) {
#pragma HLS UNROLL
        if (scale[b]) { p = b; break; }
    }
    bool exact_pow2 = (scale == ((acc_t)1 << p));   // scale is an exact power of two?
    int e = exact_pow2 ? (-p) : (-(p + 1));         // floor(log2(1/scale))
    int nege = -e;                                  // nege = p or p+1 (>=0)
    // numer = 4<<nege ; round-half-up threshold = numer + scale/2
    acc_t numer  = (LN_MBITS == 0) ? (acc_t)1 : ((acc_t)1 << (LN_MBITS + nege));
    acc_t thresh = numer + (scale >> 1);
    // nq = largest n in {4..8} with n*scale <= thresh, via shift-add candidate compares (NO divide).
    int   nbase = (1 << LN_MBITS);                  // 4
    int   nq    = nbase;
    acc_t acc   = scale << LN_MBITS;                // 4*scale = scale<<2
CAND: for (int step = 1; step <= (1 << LN_MBITS); step++) {   // n = 5,6,7,8 (bounded, unrolled)
#pragma HLS UNROLL
        acc += scale;                               // n*scale built incrementally (shift-add)
        if (acc <= thresh) nq = nbase + step;
    }
    // fold a rounded-up-to-2.0 mantissa carry: nq==8 -> nq=4, e+1
    if (nq >= (1 << (LN_MBITS + 1))) { nq = (1 << LN_MBITS); e += 1; }
    if (nq < (1 << LN_MBITS))         nq = (1 << LN_MBITS);  // guard low (>=4)
    e_out = e;
    nq_out = nq;
}

static inline data_t sat16(acc_t v) {
#pragma HLS INLINE
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return (data_t)v;
}

static inline void burst_load(int32_t *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)buf[off + i];
    }
}

void forge_layernorm(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // --- stage x on-chip; partition the d (channel) dim so the per-token sum trees and the
    //     rescale read a whole row of D lanes in parallel. ---
    static data_t xx[LN_N][LN_D];
#pragma HLS ARRAY_PARTITION variable=xx cyclic factor=16 dim=2
    burst_load(buffer, OFF_X, &xx[0][0], LN_N * LN_D);

    // --- stage the per-channel gamma codes + beta. ---
    static int32_t gcode[LN_D];
    static data_t  beta[LN_D];
#pragma HLS ARRAY_PARTITION variable=gcode cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=beta  cyclic factor=16 dim=1
    burst_load(buffer, OFF_BETA, beta, LN_D);
LDG: for (int d = 0; d < LN_D; d++) {
#pragma HLS PIPELINE II=1
        gcode[d] = buffer[OFF_GAMMA + d];
    }

    static data_t  yy[LN_N][LN_D];
#pragma HLS ARRAY_PARTITION variable=yy cyclic factor=16 dim=2
    static int32_t stats[LN_N * 2];

    // --- per-token fine-dyadic LayerNorm ---
TOK: for (int n = 0; n < LN_N; n++) {
        // (1) mean = sum_d x >> LOG2_D
        acc_t sum = 0;
    SUMX: for (int d = 0; d < LN_D; d++) {
#pragma HLS PIPELINE II=1
            sum += (acc_t)xx[n][d];
        }
        acc_t mean = sum >> LN_LOG2_D;

        // (2) c[d] = x - mean ; meanabs = sum_d |c| >> LOG2_D
        static data_t c[LN_D];
#pragma HLS ARRAY_PARTITION variable=c cyclic factor=16 dim=1
        acc_t asum = 0;
    CEN: for (int d = 0; d < LN_D; d++) {
#pragma HLS PIPELINE II=1
            acc_t cv = (acc_t)xx[n][d] - mean;
            c[d] = sat16(cv);
            acc_t a = (cv < 0) ? (acc_t)(-cv) : (acc_t)cv;   // abs, no multiply
            asum += a;
        }
        acc_t meanabs = asum >> LN_LOG2_D;          // == scale (the meanabs normalizer)

        // (3) fine-dyadic 1/scale -> (e, nq); save (e, nq) per token = the backward's OFF_RSC contract.
        int e, nq;
        fdyad(meanabs, e, nq);
        stats[2 * n]     = (int32_t)e;     // scale exponent (signed)  -> bwd OFF_RSC[2n]
        stats[2 * n + 1] = (int32_t)nq;    // 2-bit mantissa nq in {4..7} -> bwd OFF_RSC[2n+1]

        // (4) y[d] = ( c[d] * (nq/4) * 2^e ) dyadic-* gamma[d] + beta[d]   — all shift-add.
        //     r = (c*nq) / 4 * 2^e.  c*nq with nq in {4..7} = (c<<2) + (nq-4) copies of c -> shift-add.
        //     Then >>2 (the /4 mantissa denom) and apply e (LEFT shift if e>=0 else RIGHT shift),
        //     then the gamma barrel shift. NO multiply anywhere.
        int nqm4 = nq - (1 << LN_MBITS);            // 0..3 extra copies of c beyond the c<<2 base
    RESC: for (int d = 0; d < LN_D; d++) {
#pragma HLS PIPELINE II=1
            acc_t cv = (acc_t)c[d];
            // c * nq  (nq in [4,7]): base (c<<MBITS) plus nqm4 additional c's, fully unrolled adds
            acc_t cn = cv << LN_MBITS;
            if (nqm4 >= 1) cn += cv;
            if (nqm4 >= 2) cn += cv;
            if (nqm4 >= 3) cn += cv;
            // divide by the mantissa denom (>>MBITS), then apply the scale exponent e
            acc_t r = cn >> LN_MBITS;               // = c * (nq/4)
            if (e >= 0)  r = r << e;
            else         r = r >> (-e);
            // per-channel dyadic gamma: a sign + ±2^k barrel shift (DSP=0)
            bool gs, gnz; exp_t ge;
            wdecode(gcode[d], gs, ge, gnz);
            acc_t g = gnz ? (r << (int)ge) : (acc_t)0;
            if (gnz && gs) g = -g;
            acc_t yv = g + (acc_t)beta[d];
            yy[n][d] = sat16(yv);
        }
    }

    // --- writeback y + stats ---
WBY: for (int n = 0; n < LN_N; n++)
        for (int d = 0; d < LN_D; d++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_Y + n * LN_D + d] = (int32_t)yy[n][d];
        }
WBS: for (int s = 0; s < LN_N * 2; s++) {
#pragma HLS PIPELINE II=1
        buffer[OFF_STATS + s] = stats[s];
    }
}
