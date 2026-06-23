// forge_pc_block.cpp — ONE layer of a net as a DISTRIBUTED training block: one physical FPGA = one
// layer. A whole network is trained MODEL-PARALLEL across N chips, each running THIS kernel on its own
// layer, with ONLY a neighbour activation (downstream, forward) and a neighbour error (upstream, backward)
// crossing between chips. No chip ever holds the global model or the global gradient — no weight transport,
// no all-reduce, no barrier. Each chip's weights + momentum PERSIST in its own on-card DDR across all kicks.
//
// One training step = TWO kicks per chip (FWD then BWD; the BWD reads the FWD's cached relu-mask/output
// from DDR). Role (HIDDEN vs OUTPUT) + phase (FWD vs BWD) + MSE target + update-enable are read from a CFG
// word at the front of the DDR buffer, so the AXI-Lite control map is byte-identical to forge_pc_train_step
// (a single buffer pointer + kick/poll) and ONE AFI loads on every slot.
//
//   FWD  HIDDEN: pre=q(W)*a_in ; mask=(pre>0) ; a_out = maxshift(relu(pre), H1_BITS)     [a_out -> next chip]
//   FWD  OUTPUT: logits=q(W)*a_in (first C) ; a_out = maxshift(logits, LG_BITS)
//   BWD  OUTPUT: e = a_out - tgt   (MSE top error, a subtraction) ; below
//   BWD  HIDDEN: e = e_in          (the rescaled neighbour error staged by the host)
//        both:   delta = e*mask ; gW = q(delta)^T q(a_in) ; e_down = delta*q(W) ; e_out = maxshift(e_down,13)
//                mom += sign(gW) (clamped) ; W -= sign(mom)    [e_out -> previous chip; W,mom persist]
//
// Reuses forge_pc_train_step.cpp's dyad / norm_shift / burst I/O / SignSGD verbatim. Every op is a
// shift / add / compare / mux: DSP_MULT must be 0. norm_shift here is the OFF-BY-ONE-CORRECT bit_length-
// target (one more shift than the original MNIST kernel; built right from the start so tgt=8 works).
// Part xcvu47p-fsvh2892-2L-e, 4.0 ns.
#include <stdint.h>
#include <ap_int.h>

#define TD   256     // layer width (square: hidden D->D; output uses first TC rows as the readout)
#define TN   32      // microbatch
#define TC   10      // classes (OUTPUT role: first TC outputs are the logits)
#define TOT  8       // weight-grad output-tile rows
#define TIT  64      // weight-grad output-tile cols (PE lanes)
#define H1_BITS  7   // max-shift target for hidden activations
#define LG_BITS  4   // max-shift target for logits
#define EDN_BITS 13  // max-shift target for the outgoing neighbour error (fits int16, ~13 bits)
#define MCAP 7       // momentum counter cap

// roles / phases (in the CFG word)
#define PH_FWD 0
#define PH_BWD 1
#define RL_HIDDEN 0
#define RL_OUTPUT 1

typedef ap_int<16> data_t;
typedef ap_int<5>  exp_t;
typedef ap_int<48> acc_t;

#define WL 16        // int32 lanes per 512b word

// int32 DDR buffer, all regions 16-aligned. Persistent per chip: [CFG|W|MOM]. Staged per kick by host:
// AIN (fwd input / persists into bwd), TGT (output role), EIN (hidden role, bwd). Written by kernel:
// AOUT + MASK (fwd, persist into bwd), EOUT (bwd, the upstream neighbour message).
static const int OFF_CFG  = 0;                    // [PHASE, ROLE, TGT, DO_UPDATE, ...] one 512b word
static const int OFF_W    = OFF_CFG  + WL;        // TD*TD persistent weights
static const int OFF_MOM  = OFF_W    + TD*TD;     // TD*TD persistent momentum
static const int OFF_AIN  = OFF_MOM  + TD*TD;     // TN*TD this layer's input activation
static const int OFF_TGT  = OFF_AIN  + TN*TD;     // TN*TD MSE target (OUTPUT role): onehot*TGT in first C
static const int OFF_EIN  = OFF_TGT  + TN*TD;     // TN*TD incoming neighbour error (HIDDEN role, BWD)
static const int OFF_AOUT = OFF_EIN  + TN*TD;     // TN*TD forward output (downstream message)
static const int OFF_MASK = OFF_AOUT + TN*TD;     // TN*TD relu mask cached fwd->bwd
static const int OFF_EOUT = OFF_MASK + TN*TD;     // TN*TD outgoing neighbour error (upstream message)
static const int BUFLEN   = OFF_EOUT + TN*TD;

static inline void dyad(data_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<16> a = (v < 0) ? (ap_uint<16>)(-v) : (ap_uint<16>)v;
    sgn = (v < 0); nz = (a != 0);
    exp_t msb = 0;
PRIO: for (int b = 15; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { msb = b; break; }
    }
    if (msb >= 1 && a[msb - 1]) msb = msb + 1;     // table-free 1.5x round (no ROM)
    e = msb;
}

// shift s.t. |max| >> s ~ 2^target. FIXED convention: s = bit_length - target = (hb+1) - target
// (one more shift than the original MNIST kernel's hb-target; so tgt=8, no host compensation needed).
static inline int norm_shift(acc_t mx, int target) {
#pragma HLS INLINE
    ap_uint<48> a = (mx < 0) ? (ap_uint<48>)(-mx) : (ap_uint<48>)mx;
    int hb = 0;
NS: for (int b = 47; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { hb = b; break; }
    }
    int s = (hb + 1) - target;
    return s > 0 ? s : 0;
}

static inline void burst_load(ap_uint<512> *buf, int off_i32, data_t *dst, int n_i32) {
#pragma HLS INLINE off
LD: for (int w = 0; w < n_i32/WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = buf[off_i32/WL + w];
        for (int l = 0; l < WL; l++) {
#pragma HLS UNROLL
            dst[w*WL + l] = (data_t)(ap_int<32>)word.range(32*l+31, 32*l);
        }
    }
}
static inline void burst_store(ap_uint<512> *buf, int off_i32, const data_t *src, int n_i32) {
#pragma HLS INLINE off
ST: for (int w = 0; w < n_i32/WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = 0;
        for (int l = 0; l < WL; l++) {
#pragma HLS UNROLL
            word.range(32*l+31, 32*l) = (ap_uint<32>)(ap_int<32>)(int)src[w*WL + l];
        }
        buf[off_i32/WL + w] = word;
    }
}

void forge_pc_block(ap_uint<512> *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // ---- read the CFG word ----
    ap_uint<512> cfgw = buffer[OFF_CFG/WL];
    int PHASE     = (int)(ap_int<32>)cfgw.range(31, 0);
    int ROLE      = (int)(ap_int<32>)cfgw.range(63, 32);
    int TGTVAL    = (int)(ap_int<32>)cfgw.range(95, 64);   // (unused on fabric; host bakes it into OFF_TGT)
    int DO_UPDATE = (int)(ap_int<32>)cfgw.range(127, 96);
    (void)TGTVAL;
    int OUT_DIM = (ROLE == RL_OUTPUT) ? TC : TD;           // rows produced by this layer

    static data_t W[TD*TD], mom[TD*TD];
    static data_t ain[TN][TD];
#pragma HLS ARRAY_PARTITION variable=ain cyclic factor=TIT dim=2
    burst_load(buffer, OFF_W, W, TD*TD);

    // dyadic-quantize the weights (used in fwd matmul and the bwd error route, in place — transpose-free)
    static exp_t wE[TD][TD]; static bool wS[TD][TD], wNZ[TD][TD];
#pragma HLS ARRAY_PARTITION variable=wE cyclic factor=TOT dim=1
#pragma HLS ARRAY_PARTITION variable=wS cyclic factor=TOT dim=1
#pragma HLS ARRAY_PARTITION variable=wNZ cyclic factor=TOT dim=1
QW: for (int o = 0; o < TD; o++)
        for (int i = 0; i < TD; i++)
            dyad(W[o*TD+i], wS[o][i], wE[o][i], wNZ[o][i]);

    burst_load(buffer, OFF_AIN, &ain[0][0], TN*TD);

    if (PHASE == PH_FWD) {
        // ---- FORWARD: pre = q(W)*a_in ; (HIDDEN) relu+mask+maxshift ; (OUTPUT) logits+maxshift ----
        static acc_t pre[TN][TD];
        static ap_uint<1> mask[TN][TD];
        acc_t mx = 0;
    F: for (int n = 0; n < TN; n++) {
            for (int o = 0; o < TD; o++) {
                if (o >= OUT_DIM) { pre[n][o] = 0; mask[n][o] = 0; continue; }
                acc_t acc = 0;
            FI: for (int i = 0; i < TD; i++) {
#pragma HLS PIPELINE II=1
                    acc_t term = ((acc_t)ain[n][i]) << wE[o][i];
                    if (wNZ[o][i]) acc += wS[o][i] ? (acc_t)(-term) : term;
                }
                pre[n][o] = acc;
                if (ROLE == RL_HIDDEN) {
                    mask[n][o] = (acc > 0);
                    acc_t r = acc > 0 ? acc : (acc_t)0;
                    if (r > mx) mx = r;
                } else {
                    mask[n][o] = 1;                                  // readout: no relu gate
                    acc_t al = acc < 0 ? (acc_t)(-acc) : acc;
                    if (al > mx) mx = al;
                }
            }
        }
        int sh = norm_shift(mx, (ROLE == RL_HIDDEN) ? H1_BITS : LG_BITS);
        static data_t aout[TN][TD]; static data_t mout[TN][TD];
    FO: for (int n = 0; n < TN; n++)
            for (int o = 0; o < TD; o++) {
#pragma HLS PIPELINE II=1
                acc_t v = (ROLE == RL_HIDDEN) ? (pre[n][o] > 0 ? pre[n][o] : (acc_t)0) : pre[n][o];
                aout[n][o] = (o < OUT_DIM) ? (data_t)(v >> sh) : (data_t)0;
                mout[n][o] = (data_t)mask[n][o];
            }
        burst_store(buffer, OFF_AOUT, &aout[0][0], TN*TD);
        burst_store(buffer, OFF_MASK, &mout[0][0], TN*TD);
        return;
    }

    // ---- BACKWARD ----
    burst_load(buffer, OFF_MOM, mom, TD*TD);
    static data_t aout[TN][TD], mask[TN][TD];
    burst_load(buffer, OFF_AOUT, &aout[0][0], TN*TD);
    burst_load(buffer, OFF_MASK, &mask[0][0], TN*TD);

    // build delta[n][o] = e * relu_mask  (e: OUTPUT = a_out - tgt ; HIDDEN = e_in)
    static data_t delta[TN][TD];
    static exp_t dE[TN][TD]; static bool dS[TN][TD], dNZ[TN][TD];
    if (ROLE == RL_OUTPUT) {
        static data_t tgt[TN][TD];
        burst_load(buffer, OFF_TGT, &tgt[0][0], TN*TD);
    DO_OUT: for (int n = 0; n < TN; n++)
            for (int o = 0; o < TD; o++) {
#pragma HLS PIPELINE II=1
                data_t e = (o < TC) ? (data_t)(aout[n][o] - tgt[n][o]) : (data_t)0;  // mask=1 on readout
                delta[n][o] = e;
                dyad(e, dS[n][o], dE[n][o], dNZ[n][o]);
            }
    } else {
        static data_t ein[TN][TD];
        burst_load(buffer, OFF_EIN, &ein[0][0], TN*TD);
    DO_HID: for (int n = 0; n < TN; n++)
            for (int o = 0; o < TD; o++) {
#pragma HLS PIPELINE II=1
                data_t e = (mask[n][o] != 0) ? ein[n][o] : (data_t)0;                // delta = e_in * mask
                delta[n][o] = e;
                dyad(e, dS[n][o], dE[n][o], dNZ[n][o]);
            }
    }

    // dyadic-quantize a_in (operand for gW)
    static exp_t aE[TN][TD]; static bool aS[TN][TD], aNZ[TN][TD];
#pragma HLS ARRAY_PARTITION variable=aE cyclic factor=TIT dim=2
    for (int n = 0; n < TN; n++) for (int i = 0; i < TD; i++) dyad(ain[n][i], aS[n][i], aE[n][i], aNZ[n][i]);

    // ---- e_down[n][i] = sum_o delta[n][o] * q(W)[o][i] ; maxshift -> EOUT (the upstream neighbour msg) ----
    static acc_t edn[TN][TD];
    acc_t mxd = 0;
    ED: for (int n = 0; n < TN; n++)
        for (int i = 0; i < TD; i++) {
#pragma HLS PIPELINE II=1
            acc_t acc = 0;
            for (int o = 0; o < TD; o++) {
                if (o >= OUT_DIM) continue;
                acc_t term = ((acc_t)delta[n][o]) << wE[o][i];     // real delta << ew (transpose-free route)
                if (wNZ[o][i]) acc += wS[o][i] ? (acc_t)(-term) : term;
            }
            edn[n][i] = acc;
            acc_t ad = acc < 0 ? (acc_t)(-acc) : acc;
            if (ad > mxd) mxd = ad;
        }
    int she = norm_shift(mxd, EDN_BITS);
    static data_t eout[TN][TD];
    for (int n = 0; n < TN; n++)
        for (int i = 0; i < TD; i++) {
#pragma HLS PIPELINE II=1
            eout[n][i] = (data_t)(edn[n][i] >> she);
        }
    burst_store(buffer, OFF_EOUT, &eout[0][0], TN*TD);

    // ---- gW[o][i] = sum_n q(delta[n][o]) * q(a_in[n][i]) ; SignSGD-momentum update of W/mom in place ----
    GW: for (int ot = 0; ot < TD; ot += TOT) {
        for (int it = 0; it < TD; it += TIT) {
            acc_t acc[TOT][TIT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
            for (int r = 0; r < TOT; r++) for (int c2 = 0; c2 < TIT; c2++) {
#pragma HLS UNROLL
                acc[r][c2] = 0;
            }
            for (int n = 0; n < TN; n++) {
#pragma HLS PIPELINE II=1
                for (int r = 0; r < TOT; r++) {
#pragma HLS UNROLL
                    exp_t eo = dE[n][ot+r]; bool so = dS[n][ot+r]; bool nzo = dNZ[n][ot+r];
                    for (int c2 = 0; c2 < TIT; c2++) {
#pragma HLS UNROLL
                        ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)aE[n][it+c2];
                        acc_t term = (acc_t)1 << e;
                        bool neg = so ^ aS[n][it+c2];
                        if (nzo & aNZ[n][it+c2]) acc[r][c2] += neg ? (acc_t)(-term) : term;
                    }
                }
            }
            for (int r = 0; r < TOT; r++)
                for (int c2 = 0; c2 < TIT; c2++) {
#pragma HLS PIPELINE II=1
                    int row = ot + r;
                    int idx = row*TD + (it+c2);
                    int g = acc[r][c2] > 0 ? 1 : (acc[r][c2] < 0 ? -1 : 0);
                    if (row < OUT_DIM && DO_UPDATE) {
                        int m = (int)mom[idx] + g;
                        m = m > MCAP ? MCAP : (m < -MCAP ? -MCAP : m);
                        mom[idx] = (data_t)m;
                        int sm = m > 0 ? 1 : (m < 0 ? -1 : 0);
                        W[idx] = (data_t)((int)W[idx] - sm);
                    }
                }
        }
    }

    if (DO_UPDATE) {
        burst_store(buffer, OFF_W,   W,   TD*TD);
        burst_store(buffer, OFF_MOM, mom, TD*TD);
    }
}
