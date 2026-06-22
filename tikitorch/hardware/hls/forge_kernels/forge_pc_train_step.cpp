// forge_pc_train_step.cpp — ONE fused PC TRAINING STEP entirely on the FPGA (forward + cascade backward
// + multiplier-free update), no host in the inner loop. A square 2-layer dyadic net (layer 0: D->D ReLU;
// layer 1: D->C linear readout) trained by the validated DSP=0 recipe (pc_onfabric_train_ref.py, ~95% on
// 16x16 MNIST). One kick = one SGD step on a microbatch; weights + momentum PERSIST in on-card DDR across
// kicks (the host only stages a new batch and kicks). Every op is a shift / add / compare / mux:
//   forward:  pre0 = q(W0)*x (shift-add); relu+mask; h1 = maxshift(relu)  (fixed-point activation norm)
//             logits = q(W1)*h1 (first C outs); logits = maxshift(logits)
//   top err:  e1[:, :C] = logits - target          (MSE: a SUBTRACTION -- no exp/softmax/ROM)
//   backward: gW1 = q(e1)^T q(h1); e_down1 = e1*q(W1); delta0 = e_down1*relu_mask; gW0 = q(delta0)^T q(x)
//   update:   mom += sign(gW) (clamped leaky counter); W -= sign(mom)   (compare + int add; SignSGD)
// DSP must be 0. Part xcvu47p-fsvh2892-2L-e, 4.0 ns.
#include <stdint.h>
#include <ap_int.h>

#define TD   256     // square width
#define TN   16      // microbatch
#define TC   10      // classes (first C outputs of layer 1 are the logits)
#define TOT  8       // weight-grad output-tile rows
#define TIT  64      // weight-grad output-tile cols (PE lanes)
#define H1_BITS 7    // max-shift target for hidden activations (~0..128)
#define LG_BITS 4    // max-shift target for logits (~+-16)
#define MCAP   7     // momentum counter cap (leaky integrator of grad signs)

typedef ap_int<16> data_t;   // weights / acts / errors (int16 fixed-point)
typedef ap_int<5>  exp_t;    // dyadic exponent 0..16
typedef ap_int<48> acc_t;    // accumulators

// int32 DDR buffer, all regions 16-aligned (512b words): persistent [W0|W1|mom0|mom1] + staged [x|tgt]
static const int OFF_W0   = 0;
static const int OFF_W1   = OFF_W0   + TD*TD;
static const int OFF_MOM0 = OFF_W1   + TD*TD;
static const int OFF_MOM1 = OFF_MOM0 + TD*TD;
static const int OFF_X    = OFF_MOM1 + TD*TD;
static const int OFF_TGT  = OFF_X    + TN*TD;     // MSE target: onehot*8 in first C, 0 else (length TN*TD)

#define WL 16        // int32 lanes per 512b word

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

// shift count s.t. |max| >> s < 2^target (max-shift activation norm); mult-free priority encode
static inline int norm_shift(acc_t mx, int target) {
#pragma HLS INLINE
    ap_uint<48> a = (mx < 0) ? (ap_uint<48>)(-mx) : (ap_uint<48>)mx;
    int hb = 0;
NS: for (int b = 47; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { hb = b; break; }
    }
    int s = hb - target;
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

void forge_pc_train_step(ap_uint<512> *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    static data_t W0[TD*TD], W1[TD*TD], mom0[TD*TD], mom1[TD*TD];
    static data_t xx[TN][TD];
    static data_t tgt[TN][TD];
#pragma HLS ARRAY_PARTITION variable=xx cyclic factor=TIT dim=2
    burst_load(buffer, OFF_W0,   W0,   TD*TD);
    burst_load(buffer, OFF_W1,   W1,   TD*TD);
    burst_load(buffer, OFF_MOM0, mom0, TD*TD);
    burst_load(buffer, OFF_MOM1, mom1, TD*TD);
    burst_load(buffer, OFF_X,   &xx[0][0],  TN*TD);
    burst_load(buffer, OFF_TGT, &tgt[0][0], TN*TD);

    // pre-quantize the weights (both layers) to dyadic
    static exp_t w0E[TD][TD]; static bool w0S[TD][TD], w0NZ[TD][TD];
    static exp_t w1E[TD][TD]; static bool w1S[TD][TD], w1NZ[TD][TD];
#pragma HLS ARRAY_PARTITION variable=w0E cyclic factor=TOT dim=1
#pragma HLS ARRAY_PARTITION variable=w0S cyclic factor=TOT dim=1
#pragma HLS ARRAY_PARTITION variable=w0NZ cyclic factor=TOT dim=1
QW: for (int o = 0; o < TD; o++)
        for (int i = 0; i < TD; i++) {
            dyad(W0[o*TD+i], w0S[o][i], w0E[o][i], w0NZ[o][i]);
            dyad(W1[o*TD+i], w1S[o][i], w1E[o][i], w1NZ[o][i]);
        }

    // quantize x once (used in fwd as the real operand AND in gW0 as dyadic)
    static exp_t xE[TN][TD]; static bool xS[TN][TD], xNZ[TN][TD];
#pragma HLS ARRAY_PARTITION variable=xE cyclic factor=TIT dim=2
    for (int n = 0; n < TN; n++) for (int i = 0; i < TD; i++) dyad(xx[n][i], xS[n][i], xE[n][i], xNZ[n][i]);

    // ---- FORWARD layer 0: pre0 = q(W0)*x ; relu + mask ; h1 = maxshift(relu) ----
    static acc_t pre0[TN][TD];
    static ap_uint<1> mask0[TN][TD];
#pragma HLS ARRAY_PARTITION variable=mask0 cyclic factor=TOT dim=2
    acc_t mxh = 0;
F0: for (int n = 0; n < TN; n++) {
        for (int o = 0; o < TD; o++) {
#pragma HLS PIPELINE II=1
            acc_t acc = 0;
            for (int i = 0; i < TD; i++) {
                acc_t term = ((acc_t)xx[n][i]) << w0E[o][i];     // real x << ew (shift)
                if (w0NZ[o][i]) acc += w0S[o][i] ? (acc_t)(-term) : term;
            }
            pre0[n][o] = acc;
            mask0[n][o] = (acc > 0);
            acc_t r = acc > 0 ? acc : (acc_t)0;
            acc_t ar = r;                                         // track max |relu|
            if (ar > mxh) mxh = ar;
        }
    }
    int sh0 = norm_shift(mxh, H1_BITS);
    static data_t h1[TN][TD];
    static exp_t h1E[TN][TD]; static bool h1S[TN][TD], h1NZ[TN][TD];
#pragma HLS ARRAY_PARTITION variable=h1 cyclic factor=TIT dim=2
H1N: for (int n = 0; n < TN; n++)
        for (int o = 0; o < TD; o++) {
#pragma HLS PIPELINE II=1
            acc_t r = pre0[n][o] > 0 ? pre0[n][o] : (acc_t)0;
            data_t hv = (data_t)(r >> sh0);
            h1[n][o] = hv;
            dyad(hv, h1S[n][o], h1E[n][o], h1NZ[n][o]);
        }

    // ---- FORWARD layer 1 (readout, first C outputs): logits = maxshift(q(W1)*h1) ----
    static acc_t lg[TN][TC];
    acc_t mxl = 0;
F1: for (int n = 0; n < TN; n++)
        for (int c = 0; c < TC; c++) {
#pragma HLS PIPELINE II=1
            acc_t acc = 0;
            for (int i = 0; i < TD; i++) {
                acc_t term = ((acc_t)h1[n][i]) << w1E[c*TD+i];
                if (w1NZ[c*TD+i]) acc += w1S[c*TD+i] ? (acc_t)(-term) : term;
            }
            lg[n][c] = acc;
            acc_t al = acc < 0 ? (acc_t)(-acc) : acc;
            if (al > mxl) mxl = al;
        }
    int sh1 = norm_shift(mxl, LG_BITS);

    // ---- TOP ERROR (MSE): e1[n][c] = (logits>>sh1) - tgt[n][c] ; e1 = 0 for c>=C ----
    static data_t e1[TN][TC];
    static exp_t e1E[TN][TC]; static bool e1S[TN][TC], e1NZ[TN][TC];
ERR: for (int n = 0; n < TN; n++)
        for (int c = 0; c < TC; c++) {
#pragma HLS PIPELINE II=1
            data_t ev = (data_t)(lg[n][c] >> sh1) - tgt[n][c];
            e1[n][c] = ev;
            dyad(ev, e1S[n][c], e1E[n][c], e1NZ[n][c]);
        }

    // ---- BACKWARD: gW1[c][i] = sum_n q(e1)*q(h1) ; e_down1 = sum_c e1*q(W1) ; delta0 = e_down1*mask0 ----
    static data_t gW1[TC][TD];
GW1: for (int c = 0; c < TC; c++)
        for (int i = 0; i < TD; i++) {
#pragma HLS PIPELINE II=1
            acc_t acc = 0;
            for (int n = 0; n < TN; n++) {
                ap_uint<6> e = (ap_uint<6>)e1E[n][c] + (ap_uint<6>)h1E[n][i];
                acc_t term = (acc_t)1 << e;
                bool neg = e1S[n][c] ^ h1S[n][i];
                if (e1NZ[n][c] & h1NZ[n][i]) acc += neg ? (acc_t)(-term) : term;
            }
            gW1[c][i] = (data_t)(acc > 32767 ? 32767 : (acc < -32768 ? -32768 : (int)acc)); // sign is what matters
        }
    static acc_t edn[TN][TD];
    acc_t mxd = 0;
ED: for (int n = 0; n < TN; n++)
        for (int i = 0; i < TD; i++) {
#pragma HLS PIPELINE II=1
            acc_t acc = 0;
            for (int c = 0; c < TC; c++) {
                acc_t term = ((acc_t)e1[n][c]) << w1E[c*TD+i];   // real e1 << ew
                if (w1NZ[c*TD+i]) acc += w1S[c*TD+i] ? (acc_t)(-term) : term;
            }
            acc_t d = mask0[n][i] ? acc : (acc_t)0;             // delta0 = e_down1 * relu_mask
            edn[n][i] = d;
            acc_t ad = d < 0 ? (acc_t)(-d) : d;
            if (ad > mxd) mxd = ad;
        }
    int shd = norm_shift(mxd, 13);                              // max-shift delta0 into int16 (sign-preserving)
    static exp_t d0E[TN][TD]; static bool d0S[TN][TD], d0NZ[TN][TD];
#pragma HLS ARRAY_PARTITION variable=d0E cyclic factor=TOT dim=2
    for (int n = 0; n < TN; n++)
        for (int o = 0; o < TD; o++) {
            data_t dv = (data_t)(edn[n][o] >> shd);
            dyad(dv, d0S[n][o], d0E[n][o], d0NZ[n][o]);
        }

    // gW0[o][i] = sum_n q(delta0[n][o]) * q(x[n][i])   (D x D), then UPDATE W0/mom0 in place
GW0: for (int ot = 0; ot < TD; ot += TOT) {
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
                    exp_t eo = d0E[n][ot+r]; bool so = d0S[n][ot+r]; bool nzo = d0NZ[n][ot+r];
                    for (int c2 = 0; c2 < TIT; c2++) {
#pragma HLS UNROLL
                        ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)xE[n][it+c2];
                        acc_t term = (acc_t)1 << e;
                        bool neg = so ^ xS[n][it+c2];
                        if (nzo & xNZ[n][it+c2]) acc[r][c2] += neg ? (acc_t)(-term) : term;
                    }
                }
            }
            for (int r = 0; r < TOT; r++)
                for (int c2 = 0; c2 < TIT; c2++) {
#pragma HLS PIPELINE II=1
                    int idx = (ot+r)*TD + (it+c2);
                    int g = acc[r][c2] > 0 ? 1 : (acc[r][c2] < 0 ? -1 : 0);   // sign(gW0)
                    int m = (int)mom0[idx] + g;
                    m = m > MCAP ? MCAP : (m < -MCAP ? -MCAP : m);
                    mom0[idx] = (data_t)m;
                    int sm = m > 0 ? 1 : (m < 0 ? -1 : 0);
                    W0[idx] = (data_t)((int)W0[idx] - sm);                    // W0 -= sign(mom0)
                }
        }
    }

    // ---- UPDATE W1/mom1 (first C rows) from gW1 ----
UW1: for (int c = 0; c < TC; c++)
        for (int i = 0; i < TD; i++) {
#pragma HLS PIPELINE II=1
            int idx = c*TD + i;
            int g = gW1[c][i] > 0 ? 1 : (gW1[c][i] < 0 ? -1 : 0);
            int m = (int)mom1[idx] + g;
            m = m > MCAP ? MCAP : (m < -MCAP ? -MCAP : m);
            mom1[idx] = (data_t)m;
            int sm = m > 0 ? 1 : (m < 0 ? -1 : 0);
            W1[idx] = (data_t)((int)W1[idx] - sm);
        }

    // ---- write updated weights + momentum back to DDR ----
    burst_store(buffer, OFF_W0,   W0,   TD*TD);
    burst_store(buffer, OFF_W1,   W1,   TD*TD);
    burst_store(buffer, OFF_MOM0, mom0, TD*TD);
    burst_store(buffer, OFF_MOM1, mom1, TD*TD);
}
