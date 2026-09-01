// forge_attn_bwd.cpp — ATTENTION BACKWARD for the FPGA transformer-training rig (one head).
//
// The backward companion to forge_attn (the minimal-multiply attention mixer). Given the upstream
// gradient gOut[i][d] this kernel produces gQ, gK, gV. It re-derives the forward softmax p[i][j]
// from Q,K (BIT-IDENTICAL to forge_attn: L1 score -> exp2 ROM -> pow2-ceil normalize) so no extra
// host state is needed, then runs the standard soft-attention backward:
//
//   forward (forge_attn, soft p*V blend):
//     s[i][j]   = - sum_d |Q[i][d] - K[j][d]|                 (causal; sub+abs+add-tree, NO mult)
//     gap       = (rowmax - s[i][j]) >> AT_DELTA_SHIFT
//     p[i][j]   = exp2_lut[gap]  (Q16, unnormalized)  >> sh   (sh = pow2-ceil normalize SHIFT)
//     out[i][d] = (sum_j p[i][j] * V[j][d]) >> AT_P_FRAC       (THE one fwd multiply -> DSP)
//
//   backward (given gOut[i][d]):
//     (a) gV[j][d] = sum_i p[i][j] * gOut[i][d]                <-- MULTIPLY (DSP, allowed; mirrors fwd p*V)
//     (b) gp[i][j] = sum_d gOut[i][d] * V[j][d]                <-- MULTIPLY (DSP, allowed)
//     (c) softmax backward (the partition contrast, == ShiftSoftmaxBlend's weighted -<g,out>):
//           ex[i]    = ( sum_k p[i][k] * gp[i][k] ) >> AT_P_FRAC      (weighted mean of gp; p is Q16~1.0)
//           gs[i][j] = ( p[i][j] * (gp[i][j] - ex[i]) ) >> AT_P_FRAC  <-- MULTIPLY (DSP, allowed)
//     (d) L1-score subgradient (MULT-FREE — the project invariant):
//           sgn = sign(Q[i][d] - K[j][d])  in {-1,0,+1}
//           gQ[i][d] += -sgn * gs[i][j]    ;   gK[j][d] += +sgn * gs[i][j]
//         sign is ±1 so this is a CONDITIONAL ADD of gs (NO multiply): the dscore/dQ = -sign(Q-K)
//         and dscore/dK = +sign(Q-K) of the L1 distance, exactly _hinge_grad's sign(Q-K) subgradient.
//
// The ONLY multiplies (gV, gp, gs) are the irreducible softmax/blend backward — both operands
// data-dependent — kept as REAL multiplies so csynth maps them to DSP48 (mirrors forge_attn fwd
// DSP~128). The L1-score subgradient (gQ/gK), the L1 forward re-derivation, the exp2 ROM lookup
// and the pow2-ceil normalize are all sub/abs/add/compare/shift -> NO DSP multiplier.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csim+csynth via synth_attn_bwd.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- attention config (one head) — IDENTICAL to forge_attn (S1: AT_L=L_SEQ=128, AT_DH=DH=64) ----
#define AT_L   128     // sequence length
#define AT_DH  64      // head dimension (d_head)
#define AT_DELTA_SHIFT 2   // Delta divisor as a right-shift: (rowmax-s) >> AT_DELTA_SHIFT (Delta=4)
#define AT_LUT_N 256   // exp2 LUT entries (indexed by the negated, shifted score gap)
#define AT_P_FRAC 16   // probabilities held in Q16 fixed point (exp2 LUT scaled to 65536 at gap 0)

typedef ap_int<16>  data_t;    // Q / K / V values (int16, the morph data_t range)
typedef ap_int<32>  prob_t;    // Q16 probability (post shift-normalize)
typedef ap_int<32>  grad_t;    // gOut / gp / gs values (int32 deployed-grad range)
typedef ap_int<48>  acc_t;     // L1 / blend / dot accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (probe-local). Inputs first, then the three grad outputs:
//   [ Q (L*DH) | K (L*DH) | V (L*DH) | gOut (L*DH) | cfg (16) | exp2_lut (256)
//     | gQ (L*DH) | gK (L*DH) | gV (L*DH) ]
static const int OFF_Q    = 0;
static const int OFF_K    = OFF_Q   + AT_L * AT_DH;
static const int OFF_V    = OFF_K   + AT_L * AT_DH;
static const int OFF_GO   = OFF_V   + AT_L * AT_DH;     // gOut (upstream gradient)
static const int OFF_CFG  = OFF_GO  + AT_L * AT_DH;     // [0]=mode (reserved; soft path always computed)
static const int OFF_LUT  = OFF_CFG + 16;               // host-staged exp2 ROM (Q16; lut[g]=2^(-g/scale)*65536)
static const int OFF_GQ   = OFF_LUT + AT_LUT_N;
static const int OFF_GK   = OFF_GQ  + AT_L * AT_DH;
static const int OFF_GV   = OFF_GK  + AT_L * AT_DH;

static const acc_t NEG_INF = (acc_t)(-((acc_t)1 << 40));   // causal-mask sentinel (very negative)

static inline void burst_load(int32_t *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)buf[off + i];
    }
}

static inline void burst_load_g(int32_t *buf, int off, grad_t *dst, int n) {
#pragma HLS INLINE off
LDG: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (grad_t)buf[off + i];
    }
}

static inline void burst_load_words(int32_t *buf, int off, int32_t *dst, int n) {
#pragma HLS INLINE off
LDW: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = buf[off + i];
    }
}

void forge_attn_bwd(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // Stage Q/K/V/gOut on-chip. Partition the d_head dim so the L1 score add-tree, the dot products
    // and the subgradient scatter can touch a whole row/column of DH lanes in parallel.
    static data_t Qm[AT_L][AT_DH];
    static data_t Km[AT_L][AT_DH];
    static data_t Vm[AT_L][AT_DH];
    static grad_t Gm[AT_L][AT_DH];      // gOut
#pragma HLS ARRAY_PARTITION variable=Qm cyclic factor=AT_DH dim=2
#pragma HLS ARRAY_PARTITION variable=Km cyclic factor=AT_DH dim=2
#pragma HLS ARRAY_PARTITION variable=Vm cyclic factor=AT_DH dim=2
#pragma HLS ARRAY_PARTITION variable=Gm cyclic factor=AT_DH dim=2
    burst_load(buffer, OFF_Q,  &Qm[0][0], AT_L * AT_DH);
    burst_load(buffer, OFF_K,  &Km[0][0], AT_L * AT_DH);
    burst_load(buffer, OFF_V,  &Vm[0][0], AT_L * AT_DH);
    burst_load_g(buffer, OFF_GO, &Gm[0][0], AT_L * AT_DH);

    // host-staged exp2 ROM (Q16). lut[0] == 65536 (== 2^0); decays for larger score gaps.
    static int32_t exp2_lut[AT_LUT_N];
    burst_load_words(buffer, OFF_LUT, exp2_lut, AT_LUT_N);

    // gQ / gK accumulators — accumulated by the subgradient over BOTH loops (gQ over j, gK over i),
    // so they live for the whole kernel and are zeroed up front. Partition on d so the DH inner
    // conditional-add tile is conflict-free.
    static acc_t gQ[AT_L][AT_DH];
    static acc_t gK[AT_L][AT_DH];
#pragma HLS ARRAY_PARTITION variable=gQ cyclic factor=AT_DH dim=2
#pragma HLS ARRAY_PARTITION variable=gK cyclic factor=AT_DH dim=2
GZERO: for (int i = 0; i < AT_L; i++) {
#pragma HLS PIPELINE II=1
        for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
            gQ[i][d] = 0;
            gK[i][d] = 0;
        }
    }

    // gV accumulators — gV[j][d] = sum_i p[i][j]*gOut[i][d], accumulated across the query rows i.
    static acc_t gV[AT_L][AT_DH];
#pragma HLS ARRAY_PARTITION variable=gV cyclic factor=AT_DH dim=2
VZERO: for (int j = 0; j < AT_L; j++) {
#pragma HLS PIPELINE II=1
        for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
            gV[j][d] = 0;
        }
    }

    // Per-row scratch (function scope; fully overwritten each ROW iteration — the proven convention).
    static acc_t  s[AT_L];      // L1 scores
    static prob_t p[AT_L];      // Q16 normalized probabilities
    static grad_t gp[AT_L];     // gp[i][j] = sum_d gOut[i][d]*V[j][d]
    static grad_t gs[AT_L];     // softmax-backward score grad
#pragma HLS ARRAY_PARTITION variable=s  cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=p  cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=gp cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=gs cyclic factor=8

    // ============================ process one query row i at a time ============================
ROW: for (int i = 0; i < AT_L; i++) {

        // ---- (fwd re-derive) (1) L1 scores for j in [0..i] (causal); j>i masked to NEG_INF. ----
        acc_t rowmax = NEG_INF;
    SCORE: for (int j = 0; j < AT_L; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { s[j] = NEG_INF; continue; }
            acc_t l1 = 0;
        DABS: for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
                data_t diff = Qm[i][d] - Km[j][d];
                acc_t a = (diff < 0) ? (acc_t)(-diff) : (acc_t)diff;   // abs, no multiply
                l1 += a;
            }
            acc_t sc = -l1;          // L1 score = -sum_d |Q-K|
            s[j] = sc;
            if (sc > rowmax) rowmax = sc;
        }

        // ---- (fwd re-derive) (2) softmax via exp2 ROM (Q16), then pow2-ceil normalize (a SHIFT). ----
        acc_t rowsum = 0;
    EXP: for (int j = 0; j < AT_L; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { p[j] = 0; continue; }
            acc_t gap = (rowmax - s[j]) >> AT_DELTA_SHIFT;   // >= 0; Delta scaling is a shift
            int idx = (int)gap;
            if (idx < 0) idx = 0;
            if (idx > AT_LUT_N - 1) idx = AT_LUT_N - 1;      // clamp tail to last ROM entry
            prob_t pv = (prob_t)exp2_lut[idx];               // Q16 unnormalized prob
            p[j] = pv;
            rowsum += (acc_t)pv;
        }
        int sh = 0;
    NSH: for (int b = 0; b < 16; b++) {
#pragma HLS UNROLL
            if ((rowsum >> sh) > ((acc_t)1 << AT_P_FRAC)) sh++;
        }
    NRM: for (int j = 0; j < AT_L; j++) {
#pragma HLS PIPELINE II=1
            p[j] = p[j] >> sh;       // p[j] now ~Q16 normalized (sum ~= 65536)
        }

        // ---- (b) gp[i][j] = sum_d gOut[i][d] * V[j][d]   (REAL multiply -> DSP). ----
        //      Also accumulate the partition-contrast weighted sum  expnum = sum_j p[i][j]*gp[i][j].
        acc_t expnum = 0;          // Q16-scaled weighted sum of gp (== p's Q16 * gp)
    GPLOOP: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            acc_t dot = 0;
            for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
                // REAL multiply (a DSP user): int32 gOut * int16 value.
                dot += (acc_t)((acc_t)Gm[i][d] * (acc_t)Vm[j][d]);
            }
            grad_t gpj = (grad_t)dot;
            gp[j] = gpj;
            // partition contrast accumulator: p (Q16) * gp  (REAL multiply -> DSP)
            expnum += (acc_t)((acc_t)p[j] * (acc_t)gpj);
        }
        // weighted mean of gp (de-scale the Q16 prob weighting): ex = (sum_j p*gp) >> AT_P_FRAC
        grad_t ex = (grad_t)(expnum >> AT_P_FRAC);

        // ---- (c) softmax backward: gs[i][j] = ( p[i][j] * (gp[i][j] - ex) ) >> AT_P_FRAC. ----
    GSLOOP: for (int j = 0; j < AT_L; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { gs[j] = 0; continue; }
            grad_t centered = (grad_t)(gp[j] - ex);
            // REAL multiply (a DSP user): Q16 prob * centered grad.
            acc_t prod = (acc_t)((acc_t)p[j] * (acc_t)centered);
            gs[j] = (grad_t)(prod >> AT_P_FRAC);
        }

        // ---- (a) gV[j][d] += p[i][j] * gOut[i][d]   (REAL multiply -> DSP, mirrors fwd p*V). ----
    GVLOOP: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            prob_t pj = p[j];
            for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
                // REAL multiply: Q16 prob * int32 gOut. De-scale by >>AT_P_FRAC to gOut's scale.
                acc_t t = (acc_t)((acc_t)pj * (acc_t)Gm[i][d]);
                gV[j][d] += (t >> AT_P_FRAC);
            }
        }

        // ---- (d) L1-score subgradient (MULT-FREE conditional add). ----
        //   sgn = sign(Q[i][d]-K[j][d]); gQ[i][d] += -sgn*gs[j]; gK[j][d] += +sgn*gs[j].
        //   sign is ±1 -> add/subtract gs (NO multiply). diff==0 -> no contribution.
    SUBG: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            acc_t gsj = (acc_t)gs[j];
            for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
                data_t diff = Qm[i][d] - Km[j][d];
                if (diff > 0) {          // sgn=+1: gQ -= gs ; gK += gs
                    gQ[i][d] -= gsj;
                    gK[j][d] += gsj;
                } else if (diff < 0) {   // sgn=-1: gQ += gs ; gK -= gs
                    gQ[i][d] += gsj;
                    gK[j][d] -= gsj;
                }
            }
        }
    }

    // ============================ writeback gQ / gK / gV ============================
WGQ: for (int i = 0; i < AT_L; i++) {
        for (int d = 0; d < AT_DH; d++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_GQ + i * AT_DH + d] = (int32_t)gQ[i][d];
        }
    }
WGK: for (int j = 0; j < AT_L; j++) {
        for (int d = 0; d < AT_DH; d++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_GK + j * AT_DH + d] = (int32_t)gK[j][d];
        }
    }
WGV: for (int j = 0; j < AT_L; j++) {
        for (int d = 0; d < AT_DH; d++) {
#pragma HLS PIPELINE II=1
            buffer[OFF_GV + j * AT_DH + d] = (int32_t)gV[j][d];
        }
    }
}
