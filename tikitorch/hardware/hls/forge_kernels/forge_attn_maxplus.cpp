// forge_attn_maxplus.cpp — the VALIDATED fully-multiplier-free attention: max-plus L1 score + argmax-GATHER.
// This is forge_attn.cpp's deploy path ONLY (mode=1), with the soft-blend/softmax/exp2-LUT removed entirely,
// so csynth measures the true DSP=0 cost of the attention used by the tropical transformer LM (scripts/
// tropical_transformer_lm.py: 2.276 bpc, +0.056 vs dense — the mult-free recipe). Per query i:
//   score s[i][j] = -sum_d |Q[i][d] - K[j][d]|   (causal j<=i)   -- abs+add, NO multiply
//   jstar = argmax_j s[i][j]                                     -- compare tree
//   out[i][d] = V[jstar][d]                                      -- gather (one read)
// No exp, no normalize, no p*V blend -> DSP=0, and far cheaper / more parallel than softmax on fabric.
#include <stdint.h>
#include <ap_int.h>

#define AT_L   128     // sequence length
#define AT_DH  64      // head dimension (d_head)

typedef ap_int<16> data_t;   // Q/K/V values (int16)
typedef ap_int<48> acc_t;    // L1 accumulator (DSP48-width signed, used as an ADDER here)

static const int OFF_Q   = 0;
static const int OFF_K   = OFF_Q  + AT_L*AT_DH;
static const int OFF_V   = OFF_K  + AT_L*AT_DH;
static const int OFF_OUT = OFF_V  + AT_L*AT_DH;          // [ Q | K | V | out ]
static const acc_t NEG_INF = (acc_t)(-((acc_t)1 << 40));

static inline void burst_load(int32_t *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)buf[off + i];
    }
}

void forge_attn_maxplus(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    static data_t Qm[AT_L][AT_DH], Km[AT_L][AT_DH], Vm[AT_L][AT_DH];
#pragma HLS ARRAY_PARTITION variable=Qm complete dim=2
#pragma HLS ARRAY_PARTITION variable=Km complete dim=2
#pragma HLS ARRAY_PARTITION variable=Vm complete dim=2
    burst_load(buffer, OFF_Q, &Qm[0][0], AT_L*AT_DH);
    burst_load(buffer, OFF_K, &Km[0][0], AT_L*AT_DH);
    burst_load(buffer, OFF_V, &Vm[0][0], AT_L*AT_DH);

ROW: for (int i = 0; i < AT_L; i++) {
        int   jstar = 0;
        acc_t bestS = NEG_INF;
    SCORE: for (int j = 0; j < AT_L; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) continue;                                // causal mask
            acc_t l1 = 0;
        DABS: for (int d = 0; d < AT_DH; d++) {
#pragma HLS UNROLL
                data_t diff = Qm[i][d] - Km[j][d];
                acc_t a = (diff < 0) ? (acc_t)(-diff) : (acc_t)diff;   // abs, NO multiply
                l1 += a;
            }
            acc_t sc = -l1;                                     // max-plus L1 score
            if (sc > bestS) { bestS = sc; jstar = j; }          // argmax (compare tree)
        }
    WOUT: for (int d = 0; d < AT_DH; d++) {                     // argmax-GATHER: out = V[jstar], NO multiply
#pragma HLS PIPELINE II=1
            buffer[OFF_OUT + i*AT_DH + d] = (int32_t)Vm[jstar][d];
        }
    }
}
