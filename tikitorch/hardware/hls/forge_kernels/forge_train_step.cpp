// forge_train_step.cpp — monolithic single-kernel design that combines
// fwd_loss + bwd + signsgd into one HLS kernel dispatched by a `mode` arg.
//
// Why one kernel: F2 silicon testing showed BOTH the custom Nto1 mux AND
// chained PoC `axi_2to1_ddr_bridge` instances drop kernel writes (kernels
// AP_DONE but DDR untouched). Only single-kernel→PCIM (the v6 signsgd path)
// works on silicon. To preserve that path while still implementing the
// closure-clean training loop, we collapse the three ops into a single
// HLS function with a dispatcher.
//
// Args:
//   mode = 0 -> do_fwd_loss   (buffer, m=arg1, token=arg2, target=arg3)
//   mode = 1 -> do_bwd        (buffer, m=arg1, token=arg2)
//   mode = 2 -> do_signsgd    (buffer, lr=arg1, threshold=arg2)
//   mode = 3 -> do_train_window(buffer, /*unused*/arg1, /*unused*/arg2, L=arg3)
//                              tokens/targets pre-staged in OFF_TOKENS/OFF_TARGETS.
//                              Runs L forward + L backward internally; host must
//                              follow with a mode-2 signsgd kick. Returns
//                              correct-prediction count (0..L).
//
// Returns 0 on success, mode-specific error code otherwise.
// Mode 3 returns the number of correct predictions in the window.

#include "forge_buffer.hpp"
#include "forge_ce_lut.hpp"

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_uint<512> wide_t;
#endif

// =============================================================================
// Common helpers (shared across all 3 modes)
// =============================================================================

static inline void burst_load_words(int32_t* buffer, int word_off, int32_t* dst, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wide_buf = reinterpret_cast<wide_t*>(buffer);
    int n_wide = n_words >> 4;
    int wide_off = word_off >> 4;
LOAD_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
        wide_t word = wide_buf[wide_off + w];
LOAD_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int hi = lane * 32 + 31, lo = lane * 32;
            ap_int<32> v = (ap_int<32>) word.range(hi, lo);
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

static void batch_shift(const data_t x[FT_D], data_t y[FT_D]) {
#pragma HLS INLINE
    acc_t sum = 0;
BS_SUM: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        sum += (acc_t)x[d];
    }
    acc_t mean = sum >> FT_LOG2_D;
BS_SUB: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        y[d] = sat16((acc_t)x[d] - mean);
    }
}

// =============================================================================
// Mode 0: do_fwd_loss — forward pass + LM head + perceptron loss
// (body lifted from forge_fwd_loss_step.cpp)
// =============================================================================
static int do_fwd_loss(int32_t* buffer, int m, int token, int target) {
#pragma HLS INLINE off
    if (m < 0 || m >= FT_L_MAX)              return 1;
    if (token < 0 || token >= FT_V)          return 2;
    if (target < 0 || target >= FT_V)        return 3;

    static int32_t W1_buf[WORDS_W1];
    static int32_t b1_buf[WORDS_B1];
    static int32_t W2_buf[WORDS_W2];
    static int32_t b2_buf[WORDS_B2];
    static int32_t W_head_buf[WORDS_W_HEAD];
    static int32_t s_q_row[FT_D];
    static int32_t s_k_row[FT_D];
    static int32_t embed_row[FT_D];
    static int32_t k_cache_buf[WORDS_K_CACHE];
    static int32_t v_cache_buf[WORDS_V_CACHE];

    burst_load_words(buffer, OFF_W1,            W1_buf,     WORDS_W1);
    burst_load_words(buffer, OFF_B1,            b1_buf,     WORDS_B1);
    burst_load_words(buffer, OFF_W2,            W2_buf,     WORDS_W2);
    burst_load_words(buffer, OFF_B2,            b2_buf,     WORDS_B2);
    burst_load_words(buffer, OFF_W_HEAD,        W_head_buf, WORDS_W_HEAD);
    burst_load_words(buffer, OFF_S_Q   + m * FT_D,  s_q_row,    FT_D);
    burst_load_words(buffer, OFF_S_K   + m * FT_D,  s_k_row,    FT_D);
    burst_load_words(buffer, OFF_EMBED + token*FT_D, embed_row, FT_D);
    burst_load_words(buffer, OFF_K_CACHE,       k_cache_buf, WORDS_K_CACHE);
    burst_load_words(buffer, OFF_V_CACHE,       v_cache_buf, WORDS_V_CACHE);

    data_t x_t[FT_D];
EMBED_CAST: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        x_t[d] = (data_t)(int16_t)(embed_row[d] & 0xFFFF);
    }

    data_t h1[FT_D];
    batch_shift(x_t, h1);

    data_t Q_t[FT_D];
QKV: for (int d = 0; d < FT_D; d++) {
#pragma HLS PIPELINE II=1
        data_t s_q_d = (data_t)(int16_t)(s_q_row[d] & 0xFFFF);
        data_t s_k_d = (data_t)(int16_t)(s_k_row[d] & 0xFFFF);
        acc_t  q     = (acc_t)h1[d] + (acc_t)s_q_d;
        acc_t  k     = (acc_t)h1[d] + (acc_t)s_k_d;
        Q_t[d] = sat16(q);
        k_cache_buf[m * FT_D + d] = (int32_t)(int16_t)sat16(k);
        v_cache_buf[m * FT_D + d] = (int32_t)(int16_t)h1[d];
    }

    static acc_t scores[FT_L_MAX];
    static int32_t ad_score_row[FT_L_MAX];
SCORE_INIT: for (int n = 0; n < FT_L_MAX; n++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        ad_score_row[n] = -1;
    }
SCORE_N: for (int n = 0; n <= m; n++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
SCORE_D: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
            data_t k_v = (data_t)(int16_t)(k_cache_buf[n * FT_D + d] & 0xFFFF);
            acc_t s = (acc_t)Q_t[d] + (acc_t)k_v;
            if (s > mx) { mx = s; idx_d = d; }
        }
        scores[n] = mx;
        ad_score_row[n] = idx_d;
    }

    data_t a[FT_D];
    static int32_t an_out_row[FT_D];
PULL_D: for (int d = 0; d < FT_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_n = 0;
PULL_N: for (int n = 0; n <= m; n++) {
#pragma HLS UNROLL factor=8
            data_t v_v = (data_t)(int16_t)(v_cache_buf[n * FT_D + d] & 0xFFFF);
            acc_t s = (acc_t)v_v + scores[n];
            if (s > mx) { mx = s; idx_n = n; }
        }
        a[d] = sat16(mx);
        an_out_row[d] = idx_n;
    }

    data_t x_after[FT_D];
RES1: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        x_after[d] = sat16((acc_t)x_t[d] + (acc_t)a[d]);
    }

    data_t h2[FT_D];
    batch_shift(x_after, h2);

    static data_t ffn_h[FT_D_FF];
    static int32_t ad_dil_row[FT_D_FF];
FFN_DIL_K: for (int k = 0; k < FT_D_FF; k++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
FFN_DIL_D: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
            data_t W1_kd = (data_t)(int16_t)(W1_buf[k * FT_D + d] & 0xFFFF);
            acc_t s = (acc_t)W1_kd + (acc_t)h2[d];
            if (s > mx) { mx = s; idx_d = d; }
        }
        data_t b1_k = (data_t)(int16_t)(b1_buf[k] & 0xFFFF);
        ffn_h[k] = sat16(mx + (acc_t)b1_k);
        ad_dil_row[k] = idx_d;
    }

    data_t f[FT_D];
    static int32_t ak_ero_row[FT_D];
FFN_ERO_D: for (int d = 0; d < FT_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t  mn = (acc_t) 2147483647LL;
        int    idx_k = 0;
FFN_ERO_K: for (int k = 0; k < FT_D_FF; k++) {
#pragma HLS UNROLL factor=8
            data_t W2_dk = (data_t)(int16_t)(W2_buf[d * FT_D_FF + k] & 0xFFFF);
            acc_t s = (acc_t)W2_dk + (acc_t)ffn_h[k];
            if (s < mn) { mn = s; idx_k = k; }
        }
        data_t b2_d = (data_t)(int16_t)(b2_buf[d] & 0xFFFF);
        f[d] = sat16(mn + (acc_t)b2_d);
        ak_ero_row[d] = idx_k;
    }

    data_t y_t[FT_D];
RES2: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        y_t[d] = sat16((acc_t)x_after[d] + (acc_t)f[d]);
    }

    static data_t  logits[FT_V];
    static int32_t ad_head_row[FT_V];
HEAD_V: for (int v = 0; v < FT_V; v++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
HEAD_D: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
            data_t W_vd = (data_t)(int16_t)(W_head_buf[v * FT_D + d] & 0xFFFF);
            acc_t s = (acc_t)W_vd + (acc_t)y_t[d];
            if (s > mx) { mx = s; idx_d = d; }
        }
        logits[v] = sat16(mx);
        ad_head_row[v] = idx_d;
    }

    int pred = 0;
    {
        acc_t mx = (acc_t)-2147483648LL;
ARGMAX_V: for (int v = 0; v < FT_V; v++) {
#pragma HLS PIPELINE II=1
            if ((acc_t)logits[v] > mx) { mx = (acc_t)logits[v]; pred = v; }
        }
    }

    static int32_t grad_logit_row[FT_V];
GINIT: for (int v = 0; v < FT_V; v++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_logit_row[v] = 0;
    }
    const int loss_kind = buffer[OFF_CFG + 0];
    if (loss_kind == 1) {
        // Balanced LUT cross-entropy: p = exp2-ROM(z >> zshift) shift-normalized
        // to Q16 (sum in (2^15, 2^16]); grad = p - s*onehot(target) with
        // s = sum(p) so every row sums to zero exactly. Adds, compares, shifts,
        // and a 256-word ROM -- no multiply, no DSP. The unbalanced (p - 1@target)
        // variant collapses sign-trained models (see forge_ce_lut.hpp).
        const int zshift = buffer[OFF_CFG + 1];
        static int32_t e_row[FT_V];
        int32_t S = 0;                       // <= FT_V * 65536 = 2^23: fits int32
CE_E:   for (int v = 0; v < FT_V; v++) {
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
        // Balance against the POST-truncation sum: shifting each e_v
        // individually drops fractional bits, so sum(e_v >> sh) != S >> sh
        // (short by up to V-1). Scattering the sum of the p values actually
        // emitted makes the row sum exactly zero by construction.
        int32_t psum = 0;
CE_P:   for (int v = 0; v < FT_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t p = e_row[v] >> sh;
            e_row[v] = p;
            psum += p;
        }
CE_G:   for (int v = 0; v < FT_V; v++) {
#pragma HLS PIPELINE II=1
            grad_logit_row[v] = e_row[v] - (v == target ? psum : 0);
        }
    } else if (pred != target) {
        grad_logit_row[pred]   = +1;
        grad_logit_row[target] = -1;
    }

    burst_store_words(ad_score_row,   buffer, OFF_AD_SCORE   + m * FT_L_MAX, FT_L_MAX);
    burst_store_words(an_out_row,     buffer, OFF_AN_OUT     + m * FT_D,     FT_D);
    burst_store_words(ad_dil_row,     buffer, OFF_AD_DIL     + m * FT_D_FF,  FT_D_FF);
    burst_store_words(ak_ero_row,     buffer, OFF_AK_ERO     + m * FT_D,     FT_D);
    burst_store_words(ad_head_row,    buffer, OFF_AD_HEAD    + m * FT_V,     FT_V);
    burst_store_words(grad_logit_row, buffer, OFF_GRAD_LOGIT + m * FT_V,     FT_V);
    burst_store_words(k_cache_buf + m * FT_D, buffer, OFF_K_CACHE + m * FT_D, FT_D);
    burst_store_words(v_cache_buf + m * FT_D, buffer, OFF_V_CACHE + m * FT_D, FT_D);
    buffer[OFF_PRED + m] = pred;

    return 0;
}

// =============================================================================
// Mode 1: do_bwd — backward pass (sparse scatter through saved indices)
// (body lifted from forge_bwd_step.cpp)
// =============================================================================
static int do_bwd(int32_t* buffer, int m, int token) {
#pragma HLS INLINE off
    if (m < 0 || m >= FT_L_MAX)            return 1;
    if (token < 0 || token >= FT_V)        return 2;

    static int32_t ad_head_row [FT_V];
    static int32_t ad_dil_row  [FT_D_FF];
    static int32_t ak_ero_row  [FT_D];
    static int32_t ad_score_row[FT_L_MAX];
    static int32_t an_out_row  [FT_D];
    static int32_t grad_logit_row[FT_V];

    burst_load_words(buffer, OFF_AD_HEAD    + m * FT_V,     ad_head_row,    FT_V);
    burst_load_words(buffer, OFF_AD_DIL     + m * FT_D_FF,  ad_dil_row,     FT_D_FF);
    burst_load_words(buffer, OFF_AK_ERO     + m * FT_D,     ak_ero_row,     FT_D);
    burst_load_words(buffer, OFF_AD_SCORE   + m * FT_L_MAX, ad_score_row,   FT_L_MAX);
    burst_load_words(buffer, OFF_AN_OUT     + m * FT_D,     an_out_row,     FT_D);
    burst_load_words(buffer, OFF_GRAD_LOGIT + m * FT_V,     grad_logit_row, FT_V);

    static int32_t g_W_head[WORDS_G_W_HEAD];
    static int32_t g_W1    [WORDS_G_W1];
    static int32_t g_W2    [WORDS_G_W2];
    static int32_t g_b1    [WORDS_G_B1];
    static int32_t g_b2    [WORDS_G_B2];
    static int32_t g_s_q_row[FT_D];
    static int32_t g_s_k_row[FT_D];
    static int32_t g_embed_row[FT_D];
#pragma HLS BIND_STORAGE variable=g_W_head type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=g_W1     type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=g_W2     type=ram_t2p impl=uram

    burst_load_words(buffer, OFF_G_W_HEAD,                    g_W_head, WORDS_G_W_HEAD);
    burst_load_words(buffer, OFF_G_W1,                        g_W1,     WORDS_G_W1);
    burst_load_words(buffer, OFF_G_W2,                        g_W2,     WORDS_G_W2);
    burst_load_words(buffer, OFF_G_B1,                        g_b1,     WORDS_G_B1);
    burst_load_words(buffer, OFF_G_B2,                        g_b2,     WORDS_G_B2);
    burst_load_words(buffer, OFF_G_S_Q   + m * FT_D,          g_s_q_row,  FT_D);
    burst_load_words(buffer, OFF_G_S_K   + m * FT_D,          g_s_k_row,  FT_D);
    burst_load_words(buffer, OFF_G_EMBED + token * FT_D,      g_embed_row, FT_D);

    static int32_t grad_k_cache[WORDS_K_CACHE];
    static int32_t grad_v_cache[WORDS_V_CACHE];
#pragma HLS BIND_STORAGE variable=grad_k_cache type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=grad_v_cache type=ram_t2p impl=uram
    burst_load_words(buffer, OFF_K_CACHE, grad_k_cache, WORDS_K_CACHE);
    burst_load_words(buffer, OFF_V_CACHE, grad_v_cache, WORDS_V_CACHE);

    static acc_t grad_x_layer[FT_D];
GXL_INIT: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_layer[d] = 0;
    }
HEAD_BWD_V: for (int v = 0; v < FT_V; v++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int32_t g = grad_logit_row[v];
        if (g == 0) continue;
        int d_star = ad_head_row[v];
        if (d_star < 0 || d_star >= FT_D) continue;
        g_W_head[v * FT_D + d_star] += g;
        grad_x_layer[d_star] += (acc_t)g;
    }

    static acc_t grad_x_after[FT_D];
    static acc_t grad_f      [FT_D];
RB_R2: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_after[d] = grad_x_layer[d];
        grad_f      [d] = grad_x_layer[d];
    }

    static acc_t grad_ffn_h[FT_D_FF];
GFH_INIT: for (int k = 0; k < FT_D_FF; k++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_ffn_h[k] = 0;
    }
EROSION_BWD: for (int d = 0; d < FT_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int k_star = ak_ero_row[d];
        if (k_star < 0 || k_star >= FT_D_FF) continue;
        acc_t g = grad_f[d];
        g_W2[d * FT_D_FF + k_star] += (int32_t)g;
        grad_ffn_h[k_star]         += g;
        g_b2[d]                    += (int32_t)g;
    }
    static acc_t grad_h2[FT_D];
GH2_INIT: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_h2[d] = 0;
    }
DILATION_BWD: for (int k = 0; k < FT_D_FF; k++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int d_star = ad_dil_row[k];
        if (d_star < 0 || d_star >= FT_D) continue;
        acc_t g = grad_ffn_h[k];
        g_W1[k * FT_D + d_star] += (int32_t)g;
        grad_h2[d_star]         += g;
        g_b1[k]                 += (int32_t)g;
    }

    {
        acc_t sum = 0;
BS2_SUM: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            sum += grad_h2[d];
        }
        acc_t mean = sum >> FT_LOG2_D;
BS2_ADD: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            grad_x_after[d] += grad_h2[d] - mean;
        }
    }

    static acc_t grad_x_t[FT_D];
    static acc_t grad_a  [FT_D];
RB_R1: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_t[d] = grad_x_after[d];
        grad_a  [d] = grad_x_after[d];
    }

    static acc_t grad_score[FT_L_MAX];
GS_INIT: for (int n = 0; n < FT_L_MAX; n++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_score[n] = 0;
    }
ATTN_VAL: for (int d = 0; d < FT_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int n_star = an_out_row[d];
        if (n_star < 0 || n_star > m) continue;
        acc_t g = grad_a[d];
        grad_v_cache[n_star * FT_D + d] += (int32_t)g;
        grad_score[n_star]              += g;
    }
    static acc_t grad_Q[FT_D];
GQ_INIT: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_Q[d] = 0;
    }
ATTN_SCORE: for (int n = 0; n <= m; n++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int d_star = ad_score_row[n];
        if (d_star < 0 || d_star >= FT_D) continue;
        acc_t gs = grad_score[n];
        grad_Q[d_star]                       += gs;
        grad_k_cache[n * FT_D + d_star]      += (int32_t)gs;
    }
    static acc_t grad_h1[FT_D];
ATTN_OUT: for (int d = 0; d < FT_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        acc_t gV = (acc_t)grad_v_cache[m * FT_D + d];
        acc_t gK = (acc_t)grad_k_cache[m * FT_D + d];
        acc_t gQ = grad_Q[d];
        grad_h1[d] = gV + gQ + gK;
        g_s_q_row[d] += (int32_t)gQ;
        g_s_k_row[d] += (int32_t)gK;
    }

    {
        acc_t sum = 0;
BS1_SUM: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            sum += grad_h1[d];
        }
        acc_t mean = sum >> FT_LOG2_D;
BS1_ADD: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            grad_x_t[d] += grad_h1[d] - mean;
        }
    }

EMBED_BWD: for (int d = 0; d < FT_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        g_embed_row[d] += (int32_t)grad_x_t[d];
    }

    burst_store_words(g_W_head,    buffer, OFF_G_W_HEAD,                  WORDS_G_W_HEAD);
    burst_store_words(g_W1,        buffer, OFF_G_W1,                      WORDS_G_W1);
    burst_store_words(g_W2,        buffer, OFF_G_W2,                      WORDS_G_W2);
    burst_store_words(g_b1,        buffer, OFF_G_B1,                      WORDS_G_B1);
    burst_store_words(g_b2,        buffer, OFF_G_B2,                      WORDS_G_B2);
    burst_store_words(g_s_q_row,   buffer, OFF_G_S_Q   + m * FT_D,        FT_D);
    burst_store_words(g_s_k_row,   buffer, OFF_G_S_K   + m * FT_D,        FT_D);
    burst_store_words(g_embed_row, buffer, OFF_G_EMBED + token * FT_D,    FT_D);
    burst_store_words(grad_k_cache, buffer, OFF_K_CACHE,                  WORDS_K_CACHE);
    burst_store_words(grad_v_cache, buffer, OFF_V_CACHE,                  WORDS_V_CACHE);

    return 0;
}

// =============================================================================
// Mode 2: do_signsgd — apply SignSGD to all 8 weight tensors
// (body lifted from forge_signsgd_step.cpp; staged sub+saturate version)
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
    signsgd_region_wide  (buffer, OFF_EMBED,  OFF_G_EMBED,  WORDS_EMBED,  lr, threshold);
    signsgd_region_wide  (buffer, OFF_S_Q,    OFF_G_S_Q,    WORDS_S_Q,    lr, threshold);
    signsgd_region_wide  (buffer, OFF_S_K,    OFF_G_S_K,    WORDS_S_K,    lr, threshold);
    signsgd_region_wide  (buffer, OFF_W1,     OFF_G_W1,     WORDS_W1,     lr, threshold);
    signsgd_region_wide  (buffer, OFF_W2,     OFF_G_W2,     WORDS_W2,     lr, threshold);
    signsgd_region_wide  (buffer, OFF_W_HEAD, OFF_G_W_HEAD, WORDS_W_HEAD, lr, threshold);
    signsgd_region_wide  (buffer, OFF_B1,     OFF_G_B1,     WORDS_B1,     lr, threshold);
    signsgd_region_wide  (buffer, OFF_B2,     OFF_G_B2,     WORDS_B2,     lr, threshold);
    return 0;
}

// =============================================================================
// Mode 3: do_train_window — fwd + bwd over an L-token window in one kernel call.
//
// Replaces 256 per-mode host invocations (L forward + L backward) with a single
// kernel call; host still issues a separate mode-2 do_signsgd kick after each
// window to apply the weight update.  This keeps signsgd's placement identical
// to the working 1-layer AGFI (signsgd had marginal timing closure at WNS
// -0.201 ns; bundling it inside do_train_window pushed congestion past Vivado's
// route window and produced a -1.706 ns VIOLATED build).
//
// Host calls per training step:
//   1. mode 3 (this) — fwd + bwd + accuracy count   [1 kick]
//   2. mode 2 (do_signsgd) — weight update          [1 kick]
// vs the original 257 kicks: still ~99% PCIe reduction.
// =============================================================================
static int do_train_window(int32_t* buffer, int /*lr*/, int /*threshold*/, int L) {
#pragma HLS INLINE off
    if (L <= 0 || L > FT_L_MAX) return 1;

    static int32_t tokens [FT_L_MAX];
    static int32_t targets[FT_L_MAX];
    burst_load_words(buffer, OFF_TOKENS,  tokens,  FT_L_MAX);
    burst_load_words(buffer, OFF_TARGETS, targets, FT_L_MAX);

    // Zero K/V cache before fwd starts — these are used as the rolling
    // activation cache during the forward loop.  Doing it in-kernel saves
    // ~16K BAR4 pokes per training step on the host.
KCLR0: for (int i = 0; i < (int)WORDS_K_CACHE; i++) {
#pragma HLS PIPELINE II=1
        buffer[OFF_K_CACHE + i] = 0;
    }
VCLR0: for (int i = 0; i < (int)WORDS_V_CACHE; i++) {
#pragma HLS PIPELINE II=1
        buffer[OFF_V_CACHE + i] = 0;
    }

FWD_LOOP: for (int m = 0; m < L; m++) {
        int tok = tokens [m];
        int tgt = targets[m];
        // Clamp to in-range ids defensively; out-of-range ids signal "skip".
        if (tok < 0 || tok >= FT_V || tgt < 0 || tgt >= FT_V) continue;
        do_fwd_loss(buffer, m, tok, tgt);
    }

    // OFF_K_CACHE / OFF_V_CACHE are reused as gradient accumulators by do_bwd.
    // They held K/V activations during the forward loop above; clear them
    // before backward starts so the accumulation begins at 0.
KCLR: for (int i = 0; i < (int)WORDS_K_CACHE; i++) {
#pragma HLS PIPELINE II=1
        buffer[OFF_K_CACHE + i] = 0;
    }
VCLR: for (int i = 0; i < (int)WORDS_V_CACHE; i++) {
#pragma HLS PIPELINE II=1
        buffer[OFF_V_CACHE + i] = 0;
    }

BWD_LOOP: for (int m_rev = 0; m_rev < L; m_rev++) {
        int m   = L - 1 - m_rev;
        int tok = tokens[m];
        if (tok < 0 || tok >= FT_V) continue;
        do_bwd(buffer, m, tok);
    }

    // NOTE: do_signsgd is intentionally NOT called here.  Host must issue a
    // separate mode-2 kick after this returns to apply the weight update.
    // Keeping signsgd as its own placement region is what makes the 250 MHz
    // timing close on F2 silicon (see VIOLATED-build forensics 2026-05-18).

    int correct = 0;
ACC_COUNT: for (int m = 0; m < L; m++) {
#pragma HLS PIPELINE II=1
        if (buffer[OFF_PRED + m] == targets[m]) correct++;
    }
    return correct;
}

// =============================================================================
// Top-level dispatcher
// =============================================================================
extern "C" int forge_train_step(
    int32_t  buffer[FT_BUFFER_WORDS],
    int      mode,
    int      arg1,
    int      arg2,
    int      arg3
) {
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=FT_BUFFER_WORDS \
        num_write_outstanding=16 num_read_outstanding=16 \
        max_write_burst_length=16 max_read_burst_length=16 \
        max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite  port=mode    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg1    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg2    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg3    bundle=control
#pragma HLS INTERFACE s_axilite  port=buffer  bundle=control
#pragma HLS INTERFACE s_axilite  port=return  bundle=control

    if (mode == 0) return do_fwd_loss   (buffer, arg1, arg2, arg3);
    if (mode == 1) return do_bwd        (buffer, arg1, arg2);
    if (mode == 2) return do_signsgd    (buffer, arg1, arg2);
    // mode 3: in-kernel multi-step window. arg1=lr, arg2=threshold,
    //         arg3=window length L (use FT_L_MAX for full window).
    if (mode == 3) return do_train_window(buffer, arg1, arg2, arg3);
    return 99;
}
