// forge_train_step_morph_lm.cpp — "config B": monolithic MorphSSM
// LANGUAGE-MODEL training kernel, full LM on chip (embed gather, morph core,
// max-plus LM head, loss gradient), dispatched by `mode`. Position-by-position
// like forge_train_step.cpp; the post-norm BS stream is the causal-conv
// rolling cache and doubles as the grad_BS accumulator in backward (host
// zeroes it between the fwd and bwd loops; host walks bwd in REVERSE position
// order — see forge_buffer_morph_lm.hpp for the complete-by-read invariant).
//
// Layer math is IDENTICAL to forge_train_step_morph.cpp (batch_shift,
// causal depthwise morph conv with out-of-range taps EXCLUDED, alternating
// max/min by parity, max-plus projection + additive bias, residual).
// Loss is the CFG-selected 2-hot perceptron (zero-init default) or the
// balanced LUT cross-entropy lifted verbatim from forge_train_step.cpp
// (bit-exact validated; rows sum to zero against the POST-truncation sum).
//
// EVERYTHING on the data path is compare / add / shift only — ZERO
// multiplies. (k*dil / c*K etc. are loop-counter address arithmetic.)
//
//   mode 0 -> do_fwd_loss (buffer, m=arg1, token=arg2, target=arg3)
//   mode 1 -> do_bwd      (buffer, m=arg1, token=arg2)
//   mode 2 -> do_signsgd  (buffer, lr=arg1, threshold=arg2)
//
// Returns 0 on success, mode-specific error code otherwise.

#include "forge_buffer_morph_lm.hpp"
#include "forge_ce_lut.hpp"

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_uint<512> wide_t;
#endif

static const acc_t ML_NEG_INF = acc_t(-2147483647LL);
static const acc_t ML_POS_INF = acc_t( 2147483647LL);

// =============================================================================
// Common helpers — verbatim from forge_train_step_morph.cpp.
// =============================================================================

static inline void burst_load_words(int32_t* buffer, int word_off, int32_t* dst, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wide_buf = reinterpret_cast<wide_t*>(buffer);
    int n_wide = n_words >> 4;
    int wide_off = word_off >> 4;
LOAD_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
LOAD_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int hi = lane * 32 + 31, lo = lane * 32;
            ap_int<32> v = (ap_int<32>) wide_buf[wide_off + w].range(hi, lo);
            dst[w * 16 + lane] = (int32_t)(int)v;
        }
    }
#else
    for (int i = 0; i < n_words; i++) dst[i] = buffer[word_off + i];
#endif
}

static inline void burst_store_words(int32_t* src, int32_t* buffer, int word_off, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wide_buf = reinterpret_cast<wide_t*>(buffer);
    int n_wide = n_words >> 4;
    int wide_off = word_off >> 4;
STORE_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
        wide_t word = 0;
STORE_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int hi = lane * 32 + 31, lo = lane * 32;
            int32_t v = src[w * 16 + lane];
            word.range(hi, lo) = (ap_uint<32>)(ap_int<32>)v;
        }
        wide_buf[wide_off + w] = word;
    }
#else
    for (int i = 0; i < n_words; i++) buffer[word_off + i] = src[i];
#endif
}

static inline data_t sat16(acc_t v) {
#pragma HLS INLINE
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return (data_t)v;
}

// batch_shift: subtract the per-token mean over D (sum >> LOG2_D; no divider).
static void batch_shift_row(const data_t x[ML_D], data_t y[ML_D]) {
#pragma HLS INLINE
    acc_t sum = 0;
BS_SUM: for (int d = 0; d < ML_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        sum += (acc_t)x[d];
    }
    acc_t mean = sum >> ML_LOG2_D;
BS_SUB: for (int d = 0; d < ML_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        y[d] = sat16((acc_t)x[d] - mean);
    }
}

// =============================================================================
// Mode 0: do_fwd_loss — embed -> morph layers -> head -> loss grad, position m.
// =============================================================================
static int do_fwd_loss(int32_t* buffer, int m, int token, int target) {
#pragma HLS INLINE off
    if (m < 0 || m >= ML_L_SEQ)          return 1;
    if (token < 0 || token >= ML_V)      return 2;
    if (target < 0 || target >= ML_V)    return 3;

    // Weights (loaded per call, like forge_train_step.cpp's per-position loads).
    static int32_t mh_buf[WORDS_MH];
    static int32_t mb_buf[WORDS_MB];
    burst_load_words(buffer, OFF_MH, mh_buf, WORDS_MH);
    burst_load_words(buffer, OFF_MB, mb_buf, WORDS_MB);

    // Embed gather: R = EMBED[token].
    static data_t R[ML_D];
    static int32_t row32[ML_D];
    burst_load_words(buffer, off_embed(token), row32, ML_D);
EMB: for (int d = 0; d < ML_D; d++) {
#pragma HLS PIPELINE II=1
        R[d] = (data_t)(int16_t)(row32[d] & 0xFFFF);
    }

    static data_t  BS_row[ML_D];
    static data_t  MC[ML_D];
    static data_t  PJ[ML_D];
    static int32_t tap_row [ML_D];
    static int32_t proj_row[ML_D];
    static int32_t bs32[ML_D];
    static int32_t taps32[ML_K][ML_D];   // earlier BS rows for the conv taps
    static int32_t mp_row[ML_D];         // one projection row P[l][j][:]

LAYER_LOOP: for (int l = 0; l < ML_N_LAYERS; l++) {
        const int dil    = dilation_of_layer(l);
        const bool is_dil = ((l & 1) == 0);   // even -> dilation(max), odd -> erosion(min)
        const int hbase  = l * (ML_D * ML_K);

        // ---- batch_shift, then persist to the rolling BS stream ----
        batch_shift_row(R, BS_row);
BS32: for (int d = 0; d < ML_D; d++) {
#pragma HLS PIPELINE II=1
            bs32[d] = (int32_t)(int16_t)BS_row[d];
        }
        burst_store_words(bs32, buffer, off_bs(l, m), ML_D);

        // ---- causal depthwise morph conv over the BS stream ----
        // Tap k reads BS_l[m - k*dil]; out-of-range taps (t_in < 0) EXCLUDED
        // (matches forge_train_step_morph.cpp; deviates from the Python
        // zero-pad — see the buffer header note). Tap rows are burst-loaded
        // from the stream; k = 0 is the row just computed.
TAPS: for (int k = 1; k < ML_K; k++) {
            int t_in = m - k * dil;
            if (t_in >= 0) {
                burst_load_words(buffer, off_bs(l, t_in), taps32[k], ML_D);
            }
        }
MC_C: for (int c = 0; c < ML_D; c++) {
#pragma HLS PIPELINE II=1
            acc_t best = is_dil ? ML_NEG_INF : ML_POS_INF;
            int   kstar = 0;
MC_K: for (int k = 0; k < ML_K; k++) {
#pragma HLS UNROLL
                int t_in = m - k * dil;
                if (t_in >= 0) {
                    data_t hck = (data_t)(int16_t)(mh_buf[hbase + c * ML_K + k] & 0xFFFF);
                    data_t bsv = (k == 0) ? BS_row[c]
                                          : (data_t)(int16_t)(taps32[k][c] & 0xFFFF);
                    acc_t s = (acc_t)hck + (acc_t)bsv;
                    if (is_dil) { if (s > best) { best = s; kstar = k; } }
                    else        { if (s < best) { best = s; kstar = k; } }
                }
            }
            MC[c]      = sat16(best);
            tap_row[c] = kstar;
        }

        // ---- max-plus projection + additive bias ----
        // PJ[j] = max_i(P[l][j][i] + MC[i]) + MB[l][j]; argmax i* saved.
PJ_J: for (int j = 0; j < ML_D; j++) {
            burst_load_words(buffer, off_mp(l) + j * ML_D, mp_row, ML_D);
            acc_t best = ML_NEG_INF;
            int   istar = 0;
PJ_I: for (int i = 0; i < ML_D; i++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
                data_t pji = (data_t)(int16_t)(mp_row[i] & 0xFFFF);
                acc_t s = (acc_t)pji + (acc_t)MC[i];
                if (s > best) { best = s; istar = i; }
            }
            data_t bj = (data_t)(int16_t)(mb_buf[l * ML_D + j] & 0xFFFF);
            PJ[j]       = sat16(best + (acc_t)bj);
            proj_row[j] = istar;
        }

        // ---- residual add -> next layer input ----
RES: for (int j = 0; j < ML_D; j++) {
#pragma HLS PIPELINE II=1
            R[j] = sat16((acc_t)R[j] + (acc_t)PJ[j]);
        }

        burst_store_words(tap_row,  buffer, off_tap(l, m),  ML_D);
        burst_store_words(proj_row, buffer, off_proj(l, m), ML_D);
    }

    // ---- LM head: logits[v] = max_d(W_HEAD[v][d] + R[d]) ----
    static data_t  logits[ML_V];
    static int32_t ad_head_row[ML_V];
    static int32_t wh_row[ML_D];
HEAD_V: for (int v = 0; v < ML_V; v++) {
        burst_load_words(buffer, OFF_W_HEAD + v * ML_D, wh_row, ML_D);
        acc_t mx = ML_NEG_INF;
        int   idx_d = 0;
HEAD_D: for (int d = 0; d < ML_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            data_t W_vd = (data_t)(int16_t)(wh_row[d] & 0xFFFF);
            acc_t s = (acc_t)W_vd + (acc_t)R[d];
            if (s > mx) { mx = s; idx_d = d; }
        }
        logits[v] = sat16(mx);
        ad_head_row[v] = idx_d;
    }

    int pred = 0;
    {
        acc_t mx = ML_NEG_INF;
ARGMAX_V: for (int v = 0; v < ML_V; v++) {
#pragma HLS PIPELINE II=1
            if ((acc_t)logits[v] > mx) { mx = (acc_t)logits[v]; pred = v; }
        }
    }

    // ---- loss gradient (CFG-selected) — verbatim from forge_train_step.cpp ----
    static int32_t grad_logit_row[ML_V];
GINIT: for (int v = 0; v < ML_V; v++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_logit_row[v] = 0;
    }
    const int loss_kind = buffer[OFF_CFG + 0];
    if (loss_kind == 1) {
        // Balanced LUT cross-entropy: p = exp2-ROM(z >> zshift) shift-normalized
        // to Q16; grad balanced against the POST-truncation sum so every row
        // sums to exactly zero (see forge_ce_lut.hpp). Adds/compares/shifts +
        // a 256-word ROM — no multiply, no DSP.
        const int zshift = buffer[OFF_CFG + 1];
        static int32_t e_row[ML_V];
        int32_t S = 0;                       // <= ML_V * 65536 = 2^23: fits int32
CE_E:   for (int v = 0; v < ML_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t zneg = (int32_t)logits[pred] - (int32_t)logits[v];   // >= 0
            int idx = (int)(zneg >> zshift);
            if (idx > 255) idx = 255;
            e_row[v] = FT_CE_EXP2_Q16[idx];
            S += e_row[v];
        }
        int sh = 0;                          // ceil_log2(S) - 16, bounded by 7
CE_SH:  for (int b = 0; b < 8; b++) {
#pragma HLS UNROLL
            if ((S >> sh) > 65536) sh++;
        }
        int32_t psum = 0;
CE_P:   for (int v = 0; v < ML_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t p = e_row[v] >> sh;
            e_row[v] = p;
            psum += p;
        }
CE_G:   for (int v = 0; v < ML_V; v++) {
#pragma HLS PIPELINE II=1
            grad_logit_row[v] = e_row[v] - (v == target ? psum : 0);
        }
    } else if (pred != target) {
        grad_logit_row[pred]   = +1;
        grad_logit_row[target] = -1;
    }

    burst_store_words(ad_head_row,    buffer, off_head(m),       ML_V);
    burst_store_words(grad_logit_row, buffer, off_grad_logit(m), ML_V);
    buffer[OFF_PRED + m] = pred;

    return 0;
}

// =============================================================================
// Mode 1: do_bwd — head/layers/embed backward at position m, sparse scatter
// through saved indices. Host walks m in REVERSE so grad_BS stream slots are
// complete when read (see buffer header). Scalar read-modify-write into the
// gradient regions (the backward is sparse: <= V + 3*L*D touched words).
// =============================================================================
static int do_bwd(int32_t* buffer, int m, int token) {
#pragma HLS INLINE off
    if (m < 0 || m >= ML_L_SEQ)          return 1;
    if (token < 0 || token >= ML_V)      return 2;

    static int32_t grad_logit_row[ML_V];
    static int32_t ad_head_row[ML_V];
    burst_load_words(buffer, off_grad_logit(m), grad_logit_row, ML_V);
    burst_load_words(buffer, off_head(m),       ad_head_row,    ML_V);

    // ---- head backward: scatter grad_logit through ad_head ----
    static acc_t gR[ML_D];
GR_INIT: for (int d = 0; d < ML_D; d++) {
#pragma HLS PIPELINE II=1
        gR[d] = 0;
    }
HEAD_BWD: for (int v = 0; v < ML_V; v++) {
#pragma HLS PIPELINE II=4
        acc_t g = (acc_t)grad_logit_row[v];
        if (g == 0) continue;
        int dstar = ad_head_row[v];
        if (dstar < 0 || dstar >= ML_D) continue;
        buffer[OFF_G_W_HEAD + v * ML_D + dstar] += (int32_t)g;
        gR[dstar] += g;
    }

    static int32_t tap_row [ML_D];
    static int32_t proj_row[ML_D];
    static int32_t gbs32[ML_D];
    static acc_t   grad_MC[ML_D];

BWD_LAYER: for (int lr = 0; lr < ML_N_LAYERS; lr++) {
        const int l   = ML_N_LAYERS - 1 - lr;
        const int dil = dilation_of_layer(l);
        const int pbase = l * (ML_D * ML_D);
        const int hbase = l * (ML_D * ML_K);

        burst_load_words(buffer, off_tap(l, m),  tap_row,  ML_D);
        burst_load_words(buffer, off_proj(l, m), proj_row, ML_D);

        // ---- residual split: grad_PJ = gR; the skip path keeps gR ----
        // ---- projection backward (incl. additive bias grad) ----
GMC_INIT: for (int i = 0; i < ML_D; i++) {
#pragma HLS PIPELINE II=1
            grad_MC[i] = 0;
        }
PJ_BWD: for (int j = 0; j < ML_D; j++) {
#pragma HLS PIPELINE II=4
            acc_t g = gR[j];
            if (g == 0) continue;
            buffer[OFF_G_MB + l * ML_D + j] += (int32_t)g;   // bias: post-max add
            int istar = proj_row[j];
            if (istar < 0 || istar >= ML_D) continue;
            buffer[OFF_G_MP + pbase + j * ML_D + istar] += (int32_t)g;
            grad_MC[istar] += g;
        }

        // ---- conv backward: scatter to k*'s tap weight and BS stream slot ----
        // Own-slot (k*=0) contributions land in off_bs(l, m) BEFORE the
        // batch-shift backward below reads it — the in-call ordering that,
        // with the host's reverse walk, completes the slot exactly.
MC_BWD: for (int c = 0; c < ML_D; c++) {
#pragma HLS PIPELINE II=4
            acc_t g = grad_MC[c];
            if (g == 0) continue;
            int kstar = tap_row[c];
            if (kstar < 0 || kstar >= ML_K) continue;
            int t_in = m - kstar * dil;
            if (t_in < 0) continue;
            buffer[OFF_G_MH + hbase + c * ML_K + kstar] += (int32_t)g;
            buffer[off_bs(l, t_in) + c]                 += (int32_t)g;
        }

        // ---- batch_shift backward over the (complete) own grad_BS slot ----
        // grad_R[c] += grad_BS[c] - (sum_c grad_BS) >> LOG2_D, on top of the
        // residual skip already in gR.
        burst_load_words(buffer, off_bs(l, m), gbs32, ML_D);
        acc_t sum = 0;
BS_BWD_SUM: for (int c = 0; c < ML_D; c++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            sum += (acc_t)gbs32[c];
        }
        acc_t mean = sum >> ML_LOG2_D;
BS_BWD_ADD: for (int c = 0; c < ML_D; c++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            gR[c] += (acc_t)gbs32[c] - mean;
        }
    }

    // ---- embed backward: scatter gR into the token's embedding-grad row ----
EMB_BWD: for (int d = 0; d < ML_D; d++) {
#pragma HLS PIPELINE II=2
        if (gR[d] != 0) {
            buffer[off_g_embed(token) + d] += (int32_t)gR[d];
        }
    }

    return 0;
}

// =============================================================================
// Mode 2: do_signsgd — region-wide SignSGD (verbatim helper), all 5 weights.
// =============================================================================
static void signsgd_region_wide(int32_t* buffer,
                                int      w_off_words,
                                int      g_off_words,
                                int      n_words,
                                int      lr,
                                int      threshold) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wide_buf = reinterpret_cast<wide_t*>(buffer);
    int n_wide = n_words >> 4;
    int w_wide_off = w_off_words >> 4;
    int g_wide_off = g_off_words >> 4;
    ap_int<16> lr16 = (ap_int<16>)lr;
    ap_int<16> th16 = (ap_int<16>)threshold;

WORD_LOOP: for (int wi = 0; wi < n_wide; wi++) {
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=2 max=2
        wide_t w_word = wide_buf[w_wide_off + wi];
        wide_t g_word = wide_buf[g_wide_off + wi];

        ap_int<16> step_arr[16];
        ap_int<16> wcur_arr[16];
#pragma HLS ARRAY_PARTITION variable=step_arr complete
#pragma HLS ARRAY_PARTITION variable=wcur_arr complete
LANE_PREP: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int hi = lane * 32 + 31, lo = lane * 32;
            ap_int<16> w16 = (ap_int<16>) w_word.range(lo + 15, lo);
            ap_int<32> g32 = (ap_int<32>) g_word.range(hi, lo);
            ap_int<32> abs_g = (g32 < 0) ? (ap_int<32>)(-g32) : g32;
            ap_int<16> step = 0;
            if (abs_g > (ap_int<32>)th16) {
                if      (g32 > 0) step =  lr16;
                else if (g32 < 0) step = -lr16;
            }
            step_arr[lane] = step;
            wcur_arr[lane] = w16;
        }

        wide_t w_new = w_word;
LANE_SAT: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            ap_int<16> step = step_arr[lane];
            ap_int<16> wcur = wcur_arr[lane];
            if (step != 0) {
                ap_int<17> wn17 = (ap_int<17>)wcur - (ap_int<17>)step;
                ap_int<16> wn_sat;
                if      (wn17 > (ap_int<17>) 32767) wn_sat = (ap_int<16>) 32767;
                else if (wn17 < (ap_int<17>)-32768) wn_sat = (ap_int<16>)-32768;
                else                                wn_sat = (ap_int<16>)wn17;
                int hi = lane * 32 + 31, lo = lane * 32;
                w_new.range(hi, lo) = (ap_uint<32>)(ap_int<32>)wn_sat;
            }
        }

        wide_buf[w_wide_off + wi] = w_new;
        wide_buf[g_wide_off + wi] = 0;
    }
#else
    for (int i = 0; i < n_words; i++) {
        int32_t g = buffer[g_off_words + i];
        int32_t abs_g = (g < 0) ? -g : g;
        int32_t step = 0;
        if (abs_g > threshold) {
            if      (g > 0) step =  lr;
            else if (g < 0) step = -lr;
        }
        if (step != 0) {
            int16_t w = (int16_t)(buffer[w_off_words + i] & 0xFFFF);
            int32_t wn = (int32_t)w - step;
            if (wn >  32767) wn =  32767;
            if (wn < -32768) wn = -32768;
            buffer[w_off_words + i] = (int32_t)(int16_t)wn;
        }
        buffer[g_off_words + i] = 0;
    }
#endif
}

static int do_signsgd(int32_t* buffer, int lr, int threshold) {
#pragma HLS INLINE off
    signsgd_region_wide(buffer, OFF_EMBED,  OFF_G_EMBED,  WORDS_EMBED,  lr, threshold);
    signsgd_region_wide(buffer, OFF_MH,     OFF_G_MH,     WORDS_MH,     lr, threshold);
    signsgd_region_wide(buffer, OFF_MP,     OFF_G_MP,     WORDS_MP,     lr, threshold);
    signsgd_region_wide(buffer, OFF_MB,     OFF_G_MB,     WORDS_MB,     lr, threshold);
    signsgd_region_wide(buffer, OFF_W_HEAD, OFF_G_W_HEAD, WORDS_W_HEAD, lr, threshold);
    return 0;
}

// =============================================================================
// Top-level dispatcher — same ABI / pragmas as forge_train_step.cpp.
// =============================================================================
extern "C" int forge_train_step_morph_lm(
    int32_t  buffer[ML_BUFFER_WORDS],
    int      mode,
    int      arg1,
    int      arg2,
    int      arg3
) {
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=ML_BUFFER_WORDS \
        num_write_outstanding=16 num_read_outstanding=16 \
        max_write_burst_length=16 max_read_burst_length=16 \
        max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite  port=mode    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg1    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg2    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg3    bundle=control
#pragma HLS INTERFACE s_axilite  port=buffer  bundle=control
#pragma HLS INTERFACE s_axilite  port=return  bundle=control

    if (mode == ML_MODE_FWD)     return do_fwd_loss(buffer, arg1, arg2, arg3);
    if (mode == ML_MODE_BWD)     return do_bwd(buffer, arg1, arg2);
    if (mode == ML_MODE_SIGNSGD) return do_signsgd(buffer, arg1, arg2);
    return 99;
}
