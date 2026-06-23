// forge_sparsecoding_pc.cpp — GENERATIVE sparse-coding predictive coding ENTIRELY on the FPGA. ONE dyadic
// dictionary D (x ≈ D·s, s K-sparse), trained with NO backprop (settle the code + a LOCAL dyadic dictionary
// update), answers ANY inference query by SETTLING — reconstruct, in-paint under any mask, denoise. The one
// backprop-IMPOSSIBLE capability (a feed-forward net needs a separate decoder per mask). Multiplier-free:
// D is ±2^k (shift-add), the code update is shift-add, the top-k winner-take-all is a sort-free
// threshold-by-counting (compare + popcount). Validated fabric-faithful in scripts/pc_sparsecoding_fabric_ref.py
// (recovers MNIST from 50% masking ~46% better than mean-fill). Dictionary persists in on-card DDR across kicks.
//   mode=0 TRAIN  : settle the code on a clean batch -> local dyadic dict update -> per-column pow2 renorm.
//   mode=1 INFER  : settle the code on a (masked) batch -> write s + reconstruction.
// DSP_MULT must be 0. Part xcvu47p-fsvh2892-2L-e, 4.0 ns.
#include <stdint.h>
#include <ap_int.h>

#define P  256       // pixels (16x16)
#define M  256       // dictionary atoms
#define NB 32        // microbatch (images per kick)
#define K  24        // active atoms (k-sparse)
#define RB 6         // recon shift: recon = (dyad(s)·DN) >> RB
#define GBITS 6      // max-shift band for the settle gradient
#define LRS 1        // code step: s += g >> LRS
#define DLRS 4       // dict step: D += (dyad(err)^T dyad(s)) >> DLRS
#define DB 6         // dict per-column L2-norm target ~ 2^DB (renorm caps |D|)
#define MODE_TRAIN 0
#define MODE_INFER 1
#define WL 16

typedef ap_int<16> data_t;
typedef ap_int<6>  exp_t;     // dyadic exponent 0..16 (+headroom)
typedef ap_int<48> acc_t;

static const int OFF_CTRL  = 0;                  // [mode, n_steps, ...] one 512b word
static const int OFF_D     = OFF_CTRL + WL;      // P*M dictionary (persists)
static const int OFF_X     = OFF_D    + P*M;     // NB*P staged batch
static const int OFF_MASK  = OFF_X    + NB*P;    // NB*P mask (1=observed; all-ones for train)
static const int OFF_S     = OFF_MASK + NB*P;    // NB*M OUT settled codes
static const int OFF_RECON = OFF_S    + NB*M;    // NB*P OUT reconstructions
static const int BUFLEN    = OFF_RECON+ NB*P;

static inline void dyad(data_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<16> a = (v < 0) ? (ap_uint<16>)(-v) : (ap_uint<16>)v;
    sgn = (v < 0); nz = (a != 0);
    exp_t msb = 0;
PRIO: for (int b = 15; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { msb = b; break; }
    }
    if (msb >= 1 && a[msb - 1]) msb = msb + 1;     // table-free 1.5x round
    e = msb;
}
static inline int norm_shift(acc_t mx, int target) {
#pragma HLS INLINE
    ap_uint<48> a = (mx < 0) ? (ap_uint<48>)(-mx) : (ap_uint<48>)mx;
    int hb = 0;
NS: for (int b = 47; b >= 1; b--) {
#pragma HLS UNROLL
        if (a[b]) { hb = b; break; }
    }
    int s = (hb + 1) - target; return s > 0 ? s : 0;
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

// top-k by THRESHOLD-BY-COUNTING: 16-bit binary search for the threshold with >=K atoms |s|>=thr.
static inline int topk_threshold(data_t s[M]) {
#pragma HLS INLINE off
    int thr = 0;
TKB: for (int bit = 15; bit >= 0; bit--) {
        int cand = thr | (1 << bit);
        int cnt = 0;
    TKC: for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE II=1
            int a = s[m] < 0 ? -(int)s[m] : (int)s[m];
            if (a >= cand) cnt++;
        }
        if (cnt >= K) thr = cand;
    }
    return thr;
}

void forge_sparsecoding_pc(ap_uint<512> *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    ap_uint<512> ctrl = buffer[OFF_CTRL/WL];
    int mode    = (int)(ap_int<32>)ctrl.range(31, 0);
    int n_steps = (int)(ap_int<32>)ctrl.range(63, 32);
    if (n_steps <= 0) n_steps = (mode == MODE_TRAIN) ? 20 : 30;

    static data_t D[P*M];
    static data_t X[NB][P], MASK[NB][P];
    burst_load(buffer, OFF_D, D, P*M);
    burst_load(buffer, OFF_X, &X[0][0], NB*P);
    burst_load(buffer, OFF_MASK, &MASK[0][0], NB*P);

    // dyadic-quantize the dictionary (DN = dyad(D); columns are already norm-banded to ~2^DB by the last
    // train kick's renorm). Used in both D·s (recon) and Dᵀ·r (gradient) — multiplier-free.
    static exp_t DE[P][M]; static bool DS[P][M], DNZ[P][M];
#pragma HLS ARRAY_PARTITION variable=DE cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=DS cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=DNZ cyclic factor=8 dim=1
QD: for (int p = 0; p < P; p++)
        for (int m = 0; m < M; m++)
            dyad(D[p*M+m], DS[p][m], DE[p][m], DNZ[p][m]);

    static data_t Sall[NB][M], Recon[NB][P];

    // ---- SETTLE the K-sparse code for each image (fixed iteration count) ----
SET_N: for (int n = 0; n < NB; n++) {
        static data_t s[M];
        for (int m = 0; m < M; m++) s[m] = 0;
    STEP: for (int t = 0; t < n_steps; t++) {
            // recon[p] = (sum_m dyad(s)[m] * DN[p][m]) >> RB
            exp_t sE[M]; bool sS[M], sNZ[M];
            for (int m = 0; m < M; m++) dyad(s[m], sS[m], sE[m], sNZ[m]);
            static data_t r[P];
            exp_t rE[P]; bool rS[P], rNZ[P];
        RC: for (int p = 0; p < P; p++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int m = 0; m < M; m++) {
                    ap_uint<7> e = (ap_uint<7>)sE[m] + (ap_uint<7>)DE[p][m];
                    acc_t term = (acc_t)1 << e;
                    bool neg = sS[m] ^ DS[p][m];
                    if (sNZ[m] & DNZ[p][m]) acc += neg ? (acc_t)(-term) : term;
                }
                acc_t recon = acc >> RB;
                if (t == n_steps - 1) Recon[n][p] = (data_t)recon;
                acc_t rv = (acc_t)X[n][p] - recon;
                if (MASK[n][p] == 0) rv = 0;            // fit only observed pixels (mux)
                data_t rd = (data_t)rv;
                r[p] = rd;
                dyad(rd, rS[p], rE[p], rNZ[p]);
            }
            // g[m] = sum_p dyad(r)[p] * DN[p][m] ; track max|g|
            static acc_t g[M]; acc_t mxg = 0;
        GG: for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int p = 0; p < P; p++) {
                    ap_uint<7> e = (ap_uint<7>)rE[p] + (ap_uint<7>)DE[p][m];
                    acc_t term = (acc_t)1 << e;
                    bool neg = rS[p] ^ DS[p][m];
                    if (rNZ[p] & DNZ[p][m]) acc += neg ? (acc_t)(-term) : term;
                }
                g[m] = acc;
                acc_t ag = acc < 0 ? (acc_t)(-acc) : acc; if (ag > mxg) mxg = ag;
            }
            int shg = norm_shift(mxg, GBITS);            // max-shift the gradient into band
            // s += (g >> shg) >> LRS
        UP: for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE II=1
                acc_t gv = g[m] >> shg;
                s[m] = (data_t)((int)s[m] + (int)(gv >> LRS));
            }
            // top-k WTA gate
            int thr = topk_threshold(s);
        GATE: for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE II=1
                int a = s[m] < 0 ? -(int)s[m] : (int)s[m];
                if (a < thr) s[m] = 0;
            }
        }
        for (int m = 0; m < M; m++) Sall[n][m] = s[m];
    }

    // write settled codes + reconstructions (used directly in INFER; also read by host)
    burst_store(buffer, OFF_S,     &Sall[0][0],  NB*M);
    burst_store(buffer, OFF_RECON, &Recon[0][0], NB*P);
    if (mode == MODE_INFER) return;

    // ---- TRAIN: local dyadic dict update  D += (dyad(err)^T dyad(s)) >> DLRS ; per-column pow2 renorm ----
    static exp_t eE[NB][P]; static bool eS[NB][P], eNZ[NB][P];
    static exp_t SE[NB][M]; static bool SS2[NB][M], SNZ[NB][M];
    for (int n = 0; n < NB; n++) {
        for (int p = 0; p < P; p++) { data_t e = (data_t)((int)X[n][p] - (int)Recon[n][p]); dyad(e, eS[n][p], eE[n][p], eNZ[n][p]); }
        for (int m = 0; m < M; m++) dyad(Sall[n][m], SS2[n][m], SE[n][m], SNZ[n][m]);
    }
    // dD[p][m] = sum_n q(err[n][p]) * q(s[n][m]) ; D[p][m] += dD >> DLRS
DD: for (int p = 0; p < P; p++)
        for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE II=1
            acc_t acc = 0;
            for (int n = 0; n < NB; n++) {
                ap_uint<7> e = (ap_uint<7>)eE[n][p] + (ap_uint<7>)SE[n][m];
                acc_t term = (acc_t)1 << e;
                bool neg = eS[n][p] ^ SS2[n][m];
                if (eNZ[n][p] & SNZ[n][m]) acc += neg ? (acc_t)(-term) : term;
            }
            int nv = (int)D[p*M+m] + (int)(acc >> DLRS);
            D[p*M+m] = (data_t)(nv > 32767 ? 32767 : (nv < -32768 ? -32768 : nv));
        }
    // per-column pow2 renorm: c = max(0, (bitlen(sum_p dyad(D)[p][m]^2)/2) - DB) ; D[:,m] >>= c
RN: for (int m = 0; m < M; m++) {
        acc_t ss = 0;
        for (int p = 0; p < P; p++) {
            bool s_; exp_t e_; bool nz_; dyad(D[p*M+m], s_, e_, nz_);
            if (nz_) ss += (acc_t)1 << (ap_uint<7>)(e_ + e_);     // sum of (2^e)^2 = 2^(2e)
        }
        int hb = 0; ap_uint<48> a = (ap_uint<48>)ss;
        for (int b = 47; b >= 1; b--) { if (a[b]) { hb = b; break; } }
        int c = (hb / 2) - DB; if (c < 0) c = 0;
        if (c > 0) for (int p = 0; p < P; p++) D[p*M+m] = (data_t)((int)D[p*M+m] >> c);
    }
    burst_store(buffer, OFF_D, D, P*M);
}
