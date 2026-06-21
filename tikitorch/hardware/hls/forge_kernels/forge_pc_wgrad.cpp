// forge_pc_wgrad.cpp — the PC (predictive-coding) cascade / error-relaxation BACKWARD layer-step,
// fully multiplier-free. The one op the rig had not yet fused: the *routed adjoint*. It composes the
// two already-DSP=0 PEs (forge_wgrad_probe = both-dyadic weight grad; forge_fwd_gemm = real×dyadic
// route) with the cached {0,1} routing gates, exactly as predictive_coding.py cascade_grads does for a
// square interior layer (O == I == D, where the tropical max-residual skip lives):
//
//   delta[n][o]  = e[n][o] * relu_mask[n][o] * skip_gate[n][o]            (cached {0,1} gates -> SELECT)
//   gW[o][i]     = sum_n q(delta[n][o]) * q(acts[n][i])                   (BOTH dyadic -> const 1<<(ed+ea))
//   e_down[n][i] = sum_o delta[n][o] * q(W[o][i]) + e[n][i]*(1-skip[n][i])(W dyadic -> var delta<<ew; + skip)
//
// Every product has a ±2^k operand => a SHIFT, not a multiply; the gates are {0,1} => muxes; the skip
// highway is a mux/add on the cached gate. So DSP_MULT must be 0 (DSP48 may serve only as signed
// accumulators — the allowed "DSP-as-adder" lever). This is the GATE that turns Rung-5's structural
// mapping ("the PC backward IS the DSP=0 kernels") into a synthesized fact for the fused PC dataflow.
// Validated on CPU bit-for-bit by cascade_grads(mode="dyadic"). Part xcvu47p-fsvh2892-2L-e, 4.0 ns.

#include <stdint.h>
#include <ap_int.h>

// ---- square interior layer config (O == I == D) ----
#define PC_D   256     // layer width (rows of W == cols of W)
#define PC_N   16      // microbatch tokens reduced per weight-grad
#define PC_OT  8       // gW output-tile rows held on-chip
#define PC_IT  64      // gW output-tile cols held on-chip (= unrolled PE lane count)

typedef ap_int<16>  data_t;    // e / W / acts values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..16 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad / error accumulator (DSP48-width signed)

// flat int32 DDR buffer: [ e | W | acts | relu_mask | skip_gate | gW | e_down ]
static const int OFF_E    = 0;
static const int OFF_W    = OFF_E    + PC_N * PC_D;
static const int OFF_ACT  = OFF_W    + PC_D * PC_D;
static const int OFF_MASK = OFF_ACT  + PC_N * PC_D;
static const int OFF_SKIP = OFF_MASK + PC_N * PC_D;
static const int OFF_GW   = OFF_SKIP + PC_N * PC_D;
static const int OFF_EDN  = OFF_GW   + PC_D * PC_D;

// dyadic-quantize a value to (sign, exponent): round-half-in-LOG -> sign*2^round(log2|v|).
// Bit-exact to tikitorch.nn.dyadic_quantize (the measured PC carrier's quantizer) via a 16-entry
// constant threshold: bump to 2^(msb+1) iff |v| >= ceil(sqrt(2)*2^msb). Verified 0 mismatch over all
// int16 (pc_wgrad_kernel_check.py). Mult-FREE: just a priority-encode, a constant LUT read, and a
// compare -- exact round(log2|v|) the cheap-1.5x way would only approximate (the cheaper bump-on-next-
// bit round is training-equivalent, see pc_wgrad_round_ablation.py; faithful is used for rigor).
static const ap_uint<16> DYAD_THRESH[16] = {
    2, 3, 6, 12, 23, 46, 91, 182, 363, 725, 1449, 2897, 5793, 11586, 23171, 46341};

static inline void dyad(data_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<16> a = (v < 0) ? (ap_uint<16>)(-v) : (ap_uint<16>)v;
    sgn = (v < 0); nz = (a != 0);
    exp_t msb = 0;
PRIO: for (int b = 15; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { msb = b; break; }
    }
    if (nz && a >= DYAD_THRESH[msb]) msb = msb + 1;   // round half in log (== dyadic_quantize)
    e = msb;
}

static inline void burst_load(int32_t *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)buf[off + i];
    }
}

void forge_pc_wgrad(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // ---- stage operands on-chip ----
    static data_t ee[PC_N][PC_D];     // incoming error e[n][o]
    static data_t aa[PC_N][PC_D];     // input activation acts[n][i]
    static ap_uint<1> mk[PC_N][PC_D]; // relu mask {0,1}
    static ap_uint<1> sk[PC_N][PC_D]; // residual skip gate {0,1}
#pragma HLS ARRAY_PARTITION variable=ee cyclic factor=PC_OT dim=2
#pragma HLS ARRAY_PARTITION variable=aa cyclic factor=PC_IT dim=2
#pragma HLS ARRAY_PARTITION variable=mk cyclic factor=PC_OT dim=2
#pragma HLS ARRAY_PARTITION variable=sk cyclic factor=PC_IT dim=2
    burst_load(buffer, OFF_E,   &ee[0][0], PC_N * PC_D);
    burst_load(buffer, OFF_ACT, &aa[0][0], PC_N * PC_D);
    {
        static data_t tmpm[PC_N * PC_D], tmps[PC_N * PC_D];
        burst_load(buffer, OFF_MASK, tmpm, PC_N * PC_D);
        burst_load(buffer, OFF_SKIP, tmps, PC_N * PC_D);
    GATE0: for (int n = 0; n < PC_N; n++)
            for (int o = 0; o < PC_D; o++) {
#pragma HLS PIPELINE II=1
                mk[n][o] = (tmpm[n * PC_D + o] != 0);
                sk[n][o] = (tmps[n * PC_D + o] != 0);
            }
    }

    // delta[n][o] = e[n][o] gated by (relu_mask AND skip_gate)  -> a SELECT, not a multiply
    static data_t dd[PC_N][PC_D];
#pragma HLS ARRAY_PARTITION variable=dd cyclic factor=PC_OT dim=2
    static exp_t ddE[PC_N][PC_D]; static bool ddS[PC_N][PC_D]; static bool ddNZ[PC_N][PC_D];
#pragma HLS ARRAY_PARTITION variable=ddE cyclic factor=PC_OT dim=2
#pragma HLS ARRAY_PARTITION variable=ddS cyclic factor=PC_OT dim=2
#pragma HLS ARRAY_PARTITION variable=ddNZ cyclic factor=PC_OT dim=2
DELTA: for (int n = 0; n < PC_N; n++)
        for (int o = 0; o < PC_D; o++) {
#pragma HLS PIPELINE II=1
            data_t d = (mk[n][o] & sk[n][o]) ? ee[n][o] : (data_t)0;   // gated error
            dd[n][o] = d;
            dyad(d, ddS[n][o], ddE[n][o], ddNZ[n][o]);
        }

    // pre-quantize the activations (for gW) and the weights (for e_down) to dyadic once
    static exp_t aE[PC_N][PC_D]; static bool aS[PC_N][PC_D]; static bool aNZ[PC_N][PC_D];
#pragma HLS ARRAY_PARTITION variable=aE cyclic factor=PC_IT dim=2
#pragma HLS ARRAY_PARTITION variable=aS cyclic factor=PC_IT dim=2
#pragma HLS ARRAY_PARTITION variable=aNZ cyclic factor=PC_IT dim=2
QA: for (int n = 0; n < PC_N; n++)
        for (int i = 0; i < PC_D; i++) { dyad(aa[n][i], aS[n][i], aE[n][i], aNZ[n][i]); }

    static exp_t wE[PC_D][PC_D]; static bool wS[PC_D][PC_D]; static bool wNZ[PC_D][PC_D];
#pragma HLS ARRAY_PARTITION variable=wE cyclic factor=PC_OT dim=1
#pragma HLS ARRAY_PARTITION variable=wS cyclic factor=PC_OT dim=1
#pragma HLS ARRAY_PARTITION variable=wNZ cyclic factor=PC_OT dim=1
    {
        static data_t wbuf[PC_D * PC_D];
        burst_load(buffer, OFF_W, wbuf, PC_D * PC_D);
    QW: for (int o = 0; o < PC_D; o++)
            for (int i = 0; i < PC_D; i++) { dyad(wbuf[o * PC_D + i], wS[o][i], wE[o][i], wNZ[o][i]); }
    }

    // ---- (1) WEIGHT GRAD: gW[o][i] = sum_n q(delta[n][o]) * q(acts[n][i]) (both dyadic -> const shift) ----
GW_OTILE: for (int ot = 0; ot < PC_D; ot += PC_OT) {
GW_ITILE:  for (int it = 0; it < PC_D; it += PC_IT) {
        acc_t acc[PC_OT][PC_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
        for (int r = 0; r < PC_OT; r++)
            for (int c = 0; c < PC_IT; c++) {
#pragma HLS UNROLL
                acc[r][c] = 0;
            }
GW_NRED: for (int n = 0; n < PC_N; n++) {
#pragma HLS PIPELINE II=1
            for (int r = 0; r < PC_OT; r++) {
#pragma HLS UNROLL
                exp_t eo = ddE[n][ot + r]; bool so = ddS[n][ot + r]; bool nzo = ddNZ[n][ot + r];
                for (int c = 0; c < PC_IT; c++) {
#pragma HLS UNROLL
                    ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)aE[n][it + c];
                    acc_t term = (acc_t)1 << e;                       // ±2^(ed+ea): a constant shift
                    bool neg = so ^ aS[n][it + c];
                    if (nzo & aNZ[n][it + c]) acc[r][c] += neg ? (acc_t)(-term) : term;
                }
            }
        }
        for (int r = 0; r < PC_OT; r++)
            for (int c = 0; c < PC_IT; c++) {
#pragma HLS PIPELINE II=1
                buffer[OFF_GW + (ot + r) * PC_D + (it + c)] = (int32_t)acc[r][c];
            }
    }
}

    // ---- (2) ERROR ROUTE: e_down[n][i] = sum_o delta[n][o]*q(W[o][i]) + e[n][i]*(1-skip[n][i]) ----
    // delta is a FULL int -> each term is a VARIABLE SHIFT of delta by the weight exponent (forge_fwd_gemm),
    // then the unit-gain residual skip highway adds the raw error where the skip won (a mux/add).
ER_NTILE: for (int n = 0; n < PC_N; n++) {
ER_ITILE:  for (int it = 0; it < PC_D; it += PC_IT) {
        acc_t er[PC_IT];
#pragma HLS ARRAY_PARTITION variable=er complete dim=0
        for (int c = 0; c < PC_IT; c++) {
#pragma HLS UNROLL
            er[c] = 0;
        }
ER_ORED: for (int o = 0; o < PC_D; o++) {
#pragma HLS PIPELINE II=1
            data_t dv = dd[n][o];                                    // the gated error (full int)
            for (int c = 0; c < PC_IT; c++) {
#pragma HLS UNROLL
                acc_t term = ((acc_t)dv) << wE[o][it + c];           // delta << ew: a variable shift
                if (wNZ[o][it + c]) er[c] += wS[o][it + c] ? (acc_t)(-term) : term;
            }
        }
        for (int c = 0; c < PC_IT; c++) {
#pragma HLS PIPELINE II=1
            // residual skip highway: where the skip won (skip_gate==0) the raw error passes through
            acc_t skip_term = (sk[n][it + c] == 0) ? (acc_t)ee[n][it + c] : (acc_t)0;
            buffer[OFF_EDN + n * PC_D + (it + c)] = (int32_t)(er[c] + skip_term);
        }
    }
}
}
