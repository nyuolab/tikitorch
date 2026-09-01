// forge_attn.cpp — csynth PROBE for the FPGA training rig's ATTENTION block (one head).
//
// The minimal-multiply attention mixer. This probe MEASURES that the soft value-blend is the
// ONLY DSP user in the whole attention path. Everything else is sub/abs/add-tree, compares,
// shifts and a host-staged exp2 ROM — NO multiply. The single irreducible multiply is the
// p*V blend (both operands data-dependent), kept as a REAL multiply so csynth maps it to DSP48.
//
//   (1) L1 score   s[i][j] = - sum_d |Q[i][d] - K[j][d]|        (sub + abs + add-tree)
//                  causal mask j<=i, else NEG_INF.
//   (2) softmax    p[i][j] = exp2((s[i][j]-rowmax)/Delta) / pow2_ceil(rowsum)
//                  exp2 via a small static const LUT (host-staged on silicon like ce_lut),
//                  the row-sum normalize is a SHIFT (divide by ceil-pow2 of the sum). Delta const.
//   (3) blend      out[i][d] = sum_j p[i][j] * V[j][d]          <-- THE ONE IRREDUCIBLE MULTIPLY
//                  (p and V both data-dependent) -> DSP48. DSP>0 expected/correct here.
//
// Deploy path (no multiply): jstar = argmax_j s[i][j]; out[i][d] = V[jstar][d]  (compare + gather).
// Both paths are computed; OFF_CFG selects which is written to the out region.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_attn.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- attention config (one head) ----
#define AT_L   128     // sequence length
#define AT_DH  64      // head dimension (d_head)
#define AT_DELTA_SHIFT 2   // Delta divisor as a right-shift: (s-rowmax) >> AT_DELTA_SHIFT  (Delta = 4)
#define AT_LUT_N 256   // exp2 LUT entries (indexed by the negated, shifted score gap)
#define AT_P_FRAC 16   // probabilities held in Q16 fixed point (exp2 LUT scaled to 65536 at gap 0)

typedef ap_int<16>  data_t;    // Q / K / V values (int16, the morph data_t range)
typedef ap_int<32>  prob_t;    // Q16 probability (post shift-normalize), fits in int32
typedef ap_int<48>  acc_t;     // L1 / blend accumulator (DSP48-width signed)

// flat int32 DDR buffer layout (probe-local):
//   [ Q (L*DH) | K (L*DH) | V (L*DH) | out (L*DH) | cfg (16) | exp2_lut (256) ]
static const int OFF_Q      = 0;
static const int OFF_K      = OFF_Q   + AT_L * AT_DH;
static const int OFF_V      = OFF_K   + AT_L * AT_DH;
static const int OFF_OUT    = OFF_V   + AT_L * AT_DH;
static const int OFF_CFG    = OFF_OUT + AT_L * AT_DH;   // [0]=mode (0=soft p*V blend, 1=argmax gather)
static const int OFF_LUT    = OFF_CFG + 16;             // host-staged exp2 ROM (Q16; lut[g]=2^(-g/scale)*65536)

static const acc_t NEG_INF = (acc_t)(-((acc_t)1 << 40));   // causal-mask sentinel (very negative)

static inline void burst_load(int32_t *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)buf[off + i];
    }
}

static inline void burst_load_words(int32_t *buf, int off, int32_t *dst, int n) {
#pragma HLS INLINE off
LDW: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = buf[off + i];
    }
}

void forge_attn(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // Stage Q/K/V on-chip. Partition the d_head dim so the L1 score add-tree and the blend
    // can read a whole row/column of DH lanes in parallel.
    static data_t Qm[AT_L][AT_DH];
    static data_t Km[AT_L][AT_DH];
    static data_t Vm[AT_L][AT_DH];
#pragma HLS ARRAY_PARTITION variable=Qm cyclic factor=AT_DH dim=2
#pragma HLS ARRAY_PARTITION variable=Km cyclic factor=AT_DH dim=2
#pragma HLS ARRAY_PARTITION variable=Vm cyclic factor=AT_DH dim=2
    burst_load(buffer, OFF_Q, &Qm[0][0], AT_L * AT_DH);
    burst_load(buffer, OFF_K, &Km[0][0], AT_L * AT_DH);
    burst_load(buffer, OFF_V, &Vm[0][0], AT_L * AT_DH);

    // host-staged exp2 ROM (Q16). lut[0] == 65536 (== 2^0); decays for larger score gaps.
    static int32_t exp2_lut[AT_LUT_N];
    burst_load_words(buffer, OFF_LUT, exp2_lut, AT_LUT_N);

    const int mode = buffer[OFF_CFG + 0];   // 0 = soft p*V blend (DSP), 1 = argmax gather (no mult)

    // Per-row scratch (function scope, matching the proven probe convention: partitioned arrays are
    // declared once at top level, not inside a loop body). Fully overwritten each ROW iteration.
    static acc_t  s[AT_L];      // L1 scores
    static prob_t p[AT_L];      // Q16 probabilities
    static acc_t  blend[AT_DH]; // soft p*V accumulators
#pragma HLS ARRAY_PARTITION variable=s     cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=p     cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=blend complete

    // ---- process one query row i at a time ----
ROW: for (int i = 0; i < AT_L; i++) {
        // (1) L1 scores for j in [0..i] (causal); j>i masked to NEG_INF.
        acc_t rowmax = NEG_INF;
        int   jstar  = 0;       // argmax index (deploy gather path)
        acc_t bestS  = NEG_INF;
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
            if (sc > bestS) { bestS = sc; jstar = j; }   // argmax for deploy path
        }

        // (2) softmax via exp2 ROM. gap = (rowmax - s[j]) >> Delta-shift; p = lut[gap] (Q16).
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
        // normalize by pow2-ceil of the row sum: a SHIFT, not a divide. Find sh s.t. (sum>>sh)<=Q16.
        int sh = 0;
    NSH: for (int b = 0; b < 16; b++) {
#pragma HLS UNROLL
            if ((rowsum >> sh) > ((acc_t)1 << AT_P_FRAC)) sh++;
        }
    NRM: for (int j = 0; j < AT_L; j++) {
#pragma HLS PIPELINE II=1
            p[j] = p[j] >> sh;       // p[j] now ~Q16 normalized (sum ~= 65536)
        }

        // (3a) SOFT blend: out[i][d] = sum_j p[i][j] * V[j][d]  -- THE irreducible REAL multiply.
        //      p (Q16) and V both data-dependent -> csynth maps to DSP48. Output-stationary over d.
    BINIT: for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
            blend[d] = 0;
        }
    BLEND: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            prob_t pj = p[j];
            for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
                // REAL multiply (the one DSP user): Q16 prob * int16 value.
                blend[d] += (acc_t)((acc_t)pj * (acc_t)Vm[j][d]);
            }
        }

        // (3b) DEPLOY gather: out[i][d] = V[jstar][d]  -- compare/gather, NO multiply.
        // (jstar already found above.)

        // write the selected path back (de-scale the soft blend out of Q16 by >>AT_P_FRAC).
    WOUT: for (int d = 0; d < AT_DH; d++) {
#pragma HLS PIPELINE II=1
            int32_t o;
            if (mode == 1) {
                o = (int32_t)Vm[jstar][d];               // argmax gather (deploy)
            } else {
                o = (int32_t)(blend[d] >> AT_P_FRAC);     // soft p*V blend, back to data scale
            }
            buffer[OFF_OUT + i * AT_DH + d] = o;
        }
    }
}
