// forge_train_step_3layer.cpp — 3-layer extension of the Tier-1
// monolithic train_step kernel.  Built by extending forge_train_step_2layer.cpp
// with one additional per-layer dispatch case (layer_id == 2) and a second
// HIDDEN region between layers 1 and 2.
//
// Forward control flow:
//
//   layer 0 fwd:  input <- embed[token_id, :]
//                 output -> HIDDEN_0[m, :]
//   layer 1 fwd:  input <- HIDDEN_0[m, :]    (NEW: middle layer)
//                 output -> HIDDEN_1[m, :]
//   layer 2 fwd:  input <- HIDDEN_1[m, :]
//                 output (y_t) feeds LM head -> ad_head, logits (host-side loss)
//
// Backward control flow (reverse):
//
//   layer 2 bwd:  upstream <- LM head adjoint
//                 grad_x_t -> HIDDEN_1[m, :]
//   layer 1 bwd:  upstream <- HIDDEN_1[m, :]
//                 grad_x_t -> HIDDEN_0[m, :]
//   layer 0 bwd:  upstream <- HIDDEN_0[m, :]
//                 grad_x_t -> g_embed[token_id, :]
//
// Reuse policy:
//   - K_CACHE / V_CACHE store K/V activations during fwd; reused as
//     gradient accumulators during bwd (host clears between fwd/bwd).
//   - HIDDEN_0 / HIDDEN_1 each hold one layer's fwd output, then the same
//     slot is reused as the upstream-gradient slot during bwd (overwritten).

#include "forge_buffer_3layer.hpp"
#include "../forge_hls_types.hpp"
#include <cstdint>

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_uint<512> wide_t;
#endif

// =============================================================================
// Per-layer offset selectors (template-free; emitted as straight muxes).
// =============================================================================
static inline int LAYER_OFF(int layer_id, int l0_off, int l1_off, int l2_off) {
#pragma HLS INLINE
    if (layer_id == 0) return l0_off;
    if (layer_id == 1) return l1_off;
    return l2_off;
}

#define L_OFF(LAYER, TENSOR)  LAYER_OFF((LAYER), OFF3_L0_##TENSOR,   OFF3_L1_##TENSOR,   OFF3_L2_##TENSOR)
#define L_GOFF(LAYER, TENSOR) LAYER_OFF((LAYER), OFF3_G_L0_##TENSOR, OFF3_G_L1_##TENSOR, OFF3_G_L2_##TENSOR)

// HIDDEN region selectors.  layer L's fwd input comes from HIDDEN_{L-1} when L>0;
// its fwd output goes to HIDDEN_L when L<N-1.  For bwd, layer L reads upstream
// grad from HIDDEN_L (its own fwd output slot, now overwritten with grad) when
// L<N-1, and writes its downstream grad to HIDDEN_{L-1} when L>0.
static inline int hidden_out_off(int layer_id) {
#pragma HLS INLINE
    // Only valid for layer_id in {0, 1}.  layer_id == 0 -> HIDDEN_0, 1 -> HIDDEN_1.
    return (layer_id == 0) ? OFF3_HIDDEN_0 : OFF3_HIDDEN_1;
}
static inline int hidden_in_off(int layer_id) {
#pragma HLS INLINE
    // Only valid for layer_id in {1, 2}.  layer_id == 1 -> HIDDEN_0, 2 -> HIDDEN_1.
    return (layer_id == 1) ? OFF3_HIDDEN_0 : OFF3_HIDDEN_1;
}

// =============================================================================
// Per-layer word counts (alias the single-layer constants for brevity).
// All identical to forge_buffer.hpp's WORDS_* because the per-layer
// shape (D, D_FF, V, L_MAX) is the same.
// =============================================================================
constexpr int WORDS3_EMBED      = W3_EMBED;
constexpr int WORDS3_W_HEAD     = W3_W_HEAD;
constexpr int WORDS3_W1         = W3_W1;
constexpr int WORDS3_B1         = W3_B1;
constexpr int WORDS3_W2         = W3_W2;
constexpr int WORDS3_B2         = W3_B2;
constexpr int WORDS3_K_CACHE    = W3_K_CACHE;
constexpr int WORDS3_V_CACHE    = W3_V_CACHE;
constexpr int WORDS3_S_Q        = W3_S_Q;
constexpr int WORDS3_S_K        = W3_S_K;
constexpr int WORDS3_AD_SCORE   = W3_AD_SCORE;
constexpr int WORDS3_AN_OUT     = W3_AN_OUT;
constexpr int WORDS3_AD_DIL     = W3_AD_DIL;
constexpr int WORDS3_AK_ERO     = W3_AK_ERO;
constexpr int WORDS3_AD_HEAD    = W3_AD_HEAD;
constexpr int WORDS3_GRAD_LOGIT = W3_GRAD_LOGIT;

// =============================================================================
// Common helpers (verbatim from forge_train_step.cpp).
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

static void batch_shift(const data_t x[FT3_D], data_t y[FT3_D]) {
#pragma HLS INLINE
    acc_t sum = 0;
BS_SUM: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        sum += (acc_t)x[d];
    }
    acc_t mean = sum >> FT3_LOG2_D;
BS_SUB: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        y[d] = sat16((acc_t)x[d] - mean);
    }
}

// =============================================================================
// Forward + loss for a single position m at one layer.
// Body lifted from do_fwd_loss() in forge_train_step.cpp and parameterised
// on layer_id.
// =============================================================================
static int do_fwd_loss_layer(
    int32_t buffer[FT3_BUFFER_WORDS],
    int     layer_id,
    int     m,
    int     token_id,
    int     target_id_or_neg1  // -1 means no loss (used for layer-0 calls; we
                               // also accept >=0 from the host for both layers
                               // — only layer 1 consumes the target).
) {
#pragma HLS INLINE off
    if (layer_id < 0 || layer_id >= FT3_N_LAYERS) return 4;
    if (m < 0 || m >= FT3_L_MAX)                  return 1;
    if (layer_id == 0) {
        if (token_id < 0 || token_id >= FT3_V)    return 2;
    }
    const bool is_last = (layer_id == FT3_N_LAYERS - 1);
    if (is_last && target_id_or_neg1 >= 0) {
        if (target_id_or_neg1 >= FT3_V)           return 3;
    }

    const int sq_off    = L_OFF(layer_id, S_Q);
    const int sk_off    = L_OFF(layer_id, S_K);
    const int w1_off    = L_OFF(layer_id, W1);
    const int b1_off    = L_OFF(layer_id, B1);
    const int w2_off    = L_OFF(layer_id, W2);
    const int b2_off    = L_OFF(layer_id, B2);
    const int score_off = L_OFF(layer_id, AD_SCORE);
    const int an_off    = L_OFF(layer_id, AN_OUT);
    const int dil_off   = L_OFF(layer_id, AD_DIL);
    const int ero_off   = L_OFF(layer_id, AK_ERO);
    const int kc_off    = L_OFF(layer_id, K_CACHE);
    const int vc_off    = L_OFF(layer_id, V_CACHE);

    static int32_t W1_buf[WORDS3_W1];
    static int32_t b1_buf[WORDS3_B1];
    static int32_t W2_buf[WORDS3_W2];
    static int32_t b2_buf[WORDS3_B2];
    static int32_t W_head_buf[WORDS3_W_HEAD];
    static int32_t s_q_row[FT3_D];
    static int32_t s_k_row[FT3_D];
    static int32_t input_row[FT3_D];
    static int32_t k_cache_buf[WORDS3_K_CACHE];
    static int32_t v_cache_buf[WORDS3_V_CACHE];

    burst_load_words(buffer, w1_off,                W1_buf,     WORDS3_W1);
    burst_load_words(buffer, b1_off,                b1_buf,     WORDS3_B1);
    burst_load_words(buffer, w2_off,                W2_buf,     WORDS3_W2);
    burst_load_words(buffer, b2_off,                b2_buf,     WORDS3_B2);
    if (is_last) {
        burst_load_words(buffer, OFF3_W_HEAD,       W_head_buf, WORDS3_W_HEAD);
    }
    burst_load_words(buffer, sq_off  + m * FT3_D,   s_q_row,    FT3_D);
    burst_load_words(buffer, sk_off  + m * FT3_D,   s_k_row,    FT3_D);
    // Input row: layer 0 = embedding lookup; layer > 0 = HIDDEN_{layer-1}[m, :].
    if (layer_id == 0) {
        burst_load_words(buffer, OFF3_EMBED + token_id * FT3_D, input_row, FT3_D);
    } else {
        burst_load_words(buffer, hidden_in_off(layer_id) + m * FT3_D, input_row, FT3_D);
    }
    burst_load_words(buffer, kc_off,                k_cache_buf, WORDS3_K_CACHE);
    burst_load_words(buffer, vc_off,                v_cache_buf, WORDS3_V_CACHE);

    data_t x_t[FT3_D];
EMBED_CAST: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        x_t[d] = (data_t)(int16_t)(input_row[d] & 0xFFFF);
    }

    data_t h1[FT3_D];
    batch_shift(x_t, h1);

    data_t Q_t[FT3_D];
QKV: for (int d = 0; d < FT3_D; d++) {
#pragma HLS PIPELINE II=1
        data_t s_q_d = (data_t)(int16_t)(s_q_row[d] & 0xFFFF);
        data_t s_k_d = (data_t)(int16_t)(s_k_row[d] & 0xFFFF);
        acc_t  q     = (acc_t)h1[d] + (acc_t)s_q_d;
        acc_t  k     = (acc_t)h1[d] + (acc_t)s_k_d;
        Q_t[d] = sat16(q);
        k_cache_buf[m * FT3_D + d] = (int32_t)(int16_t)sat16(k);
        v_cache_buf[m * FT3_D + d] = (int32_t)(int16_t)h1[d];
    }

    static acc_t   scores[FT3_L_MAX];
    static int32_t ad_score_row[FT3_L_MAX];
SCORE_INIT: for (int n = 0; n < FT3_L_MAX; n++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        ad_score_row[n] = -1;
    }
SCORE_N: for (int n = 0; n <= m; n++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
SCORE_D: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
            data_t k_v = (data_t)(int16_t)(k_cache_buf[n * FT3_D + d] & 0xFFFF);
            acc_t s = (acc_t)Q_t[d] + (acc_t)k_v;
            if (s > mx) { mx = s; idx_d = d; }
        }
        scores[n] = mx;
        ad_score_row[n] = idx_d;
    }

    data_t a[FT3_D];
    static int32_t an_out_row[FT3_D];
PULL_D: for (int d = 0; d < FT3_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_n = 0;
PULL_N: for (int n = 0; n <= m; n++) {
#pragma HLS UNROLL factor=8
            data_t v_v = (data_t)(int16_t)(v_cache_buf[n * FT3_D + d] & 0xFFFF);
            acc_t s = (acc_t)v_v + scores[n];
            if (s > mx) { mx = s; idx_n = n; }
        }
        a[d] = sat16(mx);
        an_out_row[d] = idx_n;
    }

    data_t x_after[FT3_D];
RES1: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        x_after[d] = sat16((acc_t)x_t[d] + (acc_t)a[d]);
    }

    data_t h2[FT3_D];
    batch_shift(x_after, h2);

    static data_t  ffn_h[FT3_D_FF];
    static int32_t ad_dil_row[FT3_D_FF];
FFN_DIL_K: for (int k = 0; k < FT3_D_FF; k++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
FFN_DIL_D: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
            data_t W1_kd = (data_t)(int16_t)(W1_buf[k * FT3_D + d] & 0xFFFF);
            acc_t s = (acc_t)W1_kd + (acc_t)h2[d];
            if (s > mx) { mx = s; idx_d = d; }
        }
        data_t b1_k = (data_t)(int16_t)(b1_buf[k] & 0xFFFF);
        ffn_h[k] = sat16(mx + (acc_t)b1_k);
        ad_dil_row[k] = idx_d;
    }

    data_t f[FT3_D];
    static int32_t ak_ero_row[FT3_D];
FFN_ERO_D: for (int d = 0; d < FT3_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t  mn = (acc_t) 2147483647LL;
        int    idx_k = 0;
FFN_ERO_K: for (int k = 0; k < FT3_D_FF; k++) {
#pragma HLS UNROLL factor=8
            data_t W2_dk = (data_t)(int16_t)(W2_buf[d * FT3_D_FF + k] & 0xFFFF);
            acc_t s = (acc_t)W2_dk + (acc_t)ffn_h[k];
            if (s < mn) { mn = s; idx_k = k; }
        }
        data_t b2_d = (data_t)(int16_t)(b2_buf[d] & 0xFFFF);
        f[d] = sat16(mn + (acc_t)b2_d);
        ak_ero_row[d] = idx_k;
    }

    data_t y_t[FT3_D];
RES2: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        y_t[d] = sat16((acc_t)x_after[d] + (acc_t)f[d]);
    }

    // Store per-layer activations / caches (shared by both layers).
    burst_store_words(ad_score_row,             buffer, score_off + m * FT3_L_MAX, FT3_L_MAX);
    burst_store_words(an_out_row,               buffer, an_off    + m * FT3_D,     FT3_D);
    burst_store_words(ad_dil_row,               buffer, dil_off   + m * FT3_D_FF,  FT3_D_FF);
    burst_store_words(ak_ero_row,               buffer, ero_off   + m * FT3_D,     FT3_D);
    burst_store_words(k_cache_buf + m * FT3_D,  buffer, kc_off    + m * FT3_D,     FT3_D);
    burst_store_words(v_cache_buf + m * FT3_D,  buffer, vc_off    + m * FT3_D,     FT3_D);

    if (!is_last) {
        // Non-last layer: block output -> HIDDEN_{layer_id}[m, :].  No LM head.
        static int32_t y_out_row[FT3_D];
HIDDEN_CAST: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            y_out_row[d] = (int32_t)(int16_t)y_t[d];
        }
        burst_store_words(y_out_row, buffer, hidden_out_off(layer_id) + m * FT3_D, FT3_D);
        return 0;
    }

    // ===== last layer: LM head + (host-side) loss =====
    static data_t  logits[FT3_V];
    static int32_t ad_head_row[FT3_V];
HEAD_V: for (int v = 0; v < FT3_V; v++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
HEAD_D: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
            data_t W_vd = (data_t)(int16_t)(W_head_buf[v * FT3_D + d] & 0xFFFF);
            acc_t s = (acc_t)W_vd + (acc_t)y_t[d];
            if (s > mx) { mx = s; idx_d = d; }
        }
        logits[v] = sat16(mx);
        ad_head_row[v] = idx_d;
    }

    // Kernel writes raw int16 logits to OFF3_GRAD_LOGIT (reusing the region as a
    // logits scratchpad).  Host computes the loss gradient (perceptron, CE, or
    // any other) and writes the actual gradient back to the same region before
    // backward.  This removes pred/argmax/perceptron-grad logic from the kernel
    // (less code, easier timing) and enables host-side cross-entropy without a
    // buffer layout change.
    (void)target_id_or_neg1;  // unused: loss now computed host-side
    static int32_t logits_i32_row[FT3_V];
LOGITS_OUT: for (int v = 0; v < FT3_V; v++) {
#pragma HLS PIPELINE II=1
        logits_i32_row[v] = (int32_t)(int16_t)logits[v];
    }

    burst_store_words(ad_head_row,    buffer, OFF3_AD_HEAD    + m * FT3_V, FT3_V);
    burst_store_words(logits_i32_row, buffer, OFF3_GRAD_LOGIT + m * FT3_V, FT3_V);

    return 0;
}

// =============================================================================
// Backward through one layer at one position m.
// Body lifted from do_bwd() in forge_train_step.cpp and parameterised
// on layer_id.  Layer 1's head-adjoint seeds grad_x_layer; layer 0's
// seeds it from HIDDEN[m, :] (written by layer 1's backward pass).
// =============================================================================
static int do_bwd_layer(
    int32_t buffer[FT3_BUFFER_WORDS],
    int     layer_id,
    int     m,
    int     token_id_or_neg1
) {
#pragma HLS INLINE off
    if (layer_id < 0 || layer_id >= FT3_N_LAYERS) return 4;
    if (m < 0 || m >= FT3_L_MAX)                  return 1;
    if (layer_id == 0) {
        if (token_id_or_neg1 < 0 || token_id_or_neg1 >= FT3_V) return 2;
    }
    const bool is_last = (layer_id == FT3_N_LAYERS - 1);

    const int sq_off    = L_OFF (layer_id, S_Q);
    const int sk_off    = L_OFF (layer_id, S_K);
    const int score_off = L_OFF (layer_id, AD_SCORE);
    const int an_off    = L_OFF (layer_id, AN_OUT);
    const int dil_off   = L_OFF (layer_id, AD_DIL);
    const int ero_off   = L_OFF (layer_id, AK_ERO);
    const int kc_off    = L_OFF (layer_id, K_CACHE);
    const int vc_off    = L_OFF (layer_id, V_CACHE);
    const int g_sq_off  = L_GOFF(layer_id, S_Q);
    const int g_sk_off  = L_GOFF(layer_id, S_K);
    const int g_w1_off  = L_GOFF(layer_id, W1);
    const int g_b1_off  = L_GOFF(layer_id, B1);
    const int g_w2_off  = L_GOFF(layer_id, W2);
    const int g_b2_off  = L_GOFF(layer_id, B2);

    static int32_t ad_head_row [FT3_V];
    static int32_t ad_dil_row  [FT3_D_FF];
    static int32_t ak_ero_row  [FT3_D];
    static int32_t ad_score_row[FT3_L_MAX];
    static int32_t an_out_row  [FT3_D];
    static int32_t grad_logit_row[FT3_V];

    burst_load_words(buffer, dil_off   + m * FT3_D_FF,  ad_dil_row,     FT3_D_FF);
    burst_load_words(buffer, ero_off   + m * FT3_D,     ak_ero_row,     FT3_D);
    burst_load_words(buffer, score_off + m * FT3_L_MAX, ad_score_row,   FT3_L_MAX);
    burst_load_words(buffer, an_off    + m * FT3_D,     an_out_row,     FT3_D);
    if (is_last) {
        burst_load_words(buffer, OFF3_AD_HEAD    + m * FT3_V, ad_head_row,    FT3_V);
        burst_load_words(buffer, OFF3_GRAD_LOGIT + m * FT3_V, grad_logit_row, FT3_V);
    }

    static int32_t g_W_head[W3_W_HEAD];
    static int32_t g_W1    [W3_W1];
    static int32_t g_W2    [W3_W2];
    static int32_t g_b1    [W3_B1];
    static int32_t g_b2    [W3_B2];
    static int32_t g_s_q_row[FT3_D];
    static int32_t g_s_k_row[FT3_D];
    static int32_t g_embed_row[FT3_D];
#pragma HLS BIND_STORAGE variable=g_W_head type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=g_W1     type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=g_W2     type=ram_t2p impl=uram

    if (is_last) {
        burst_load_words(buffer, OFF3_G_W_HEAD, g_W_head, W3_W_HEAD);
    }
    burst_load_words(buffer, g_w1_off,                    g_W1,       W3_W1);
    burst_load_words(buffer, g_w2_off,                    g_W2,       W3_W2);
    burst_load_words(buffer, g_b1_off,                    g_b1,       W3_B1);
    burst_load_words(buffer, g_b2_off,                    g_b2,       W3_B2);
    burst_load_words(buffer, g_sq_off + m * FT3_D,        g_s_q_row,  FT3_D);
    burst_load_words(buffer, g_sk_off + m * FT3_D,        g_s_k_row,  FT3_D);
    if (layer_id == 0) {
        burst_load_words(buffer, OFF3_G_EMBED + token_id_or_neg1 * FT3_D,
                         g_embed_row, FT3_D);
    }

    static int32_t grad_k_cache[W3_K_CACHE];
    static int32_t grad_v_cache[W3_V_CACHE];
#pragma HLS BIND_STORAGE variable=grad_k_cache type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=grad_v_cache type=ram_t2p impl=uram
    burst_load_words(buffer, kc_off, grad_k_cache, W3_K_CACHE);
    burst_load_words(buffer, vc_off, grad_v_cache, W3_V_CACHE);

    // -------------------------------------------------------------------------
    // Seed grad_x_layer with the upstream gradient.
    //   last layer (2): from LM head adjoint (head writes to W_head and grad_x_layer)
    //   layer < N-1:    from HIDDEN_{layer}[m, :] (previously written by the
    //                   bwd of layer+1; this is the same buffer region that the
    //                   forward of layer wrote its output to).
    // -------------------------------------------------------------------------
    static acc_t grad_x_layer[FT3_D];
GXL_INIT: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_layer[d] = 0;
    }

    if (is_last) {
HEAD_BWD_V: for (int v = 0; v < FT3_V; v++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
            int32_t g = grad_logit_row[v];
            if (g == 0) continue;
            int d_star = ad_head_row[v];
            if (d_star < 0 || d_star >= FT3_D) continue;
            g_W_head[v * FT3_D + d_star] += g;
            grad_x_layer[d_star]         += (acc_t)g;
        }
    } else {
        // Non-last layer: upstream comes from HIDDEN_{layer}[m, :] (written
        // by the bwd of the layer above us).
        static int32_t hidden_grad_row[FT3_D];
        burst_load_words(buffer, hidden_out_off(layer_id) + m * FT3_D,
                         hidden_grad_row, FT3_D);
GXL_LOAD0: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            grad_x_layer[d] = (acc_t)hidden_grad_row[d];
        }
    }

    // -------------------------------------------------------------------------
    // Backward through the residual2 split (post-FFN).
    // -------------------------------------------------------------------------
    static acc_t grad_x_after[FT3_D];
    static acc_t grad_f      [FT3_D];
RB_R2: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_after[d] = grad_x_layer[d];
        grad_f      [d] = grad_x_layer[d];
    }

    static acc_t grad_ffn_h[FT3_D_FF];
GFH_INIT: for (int k = 0; k < FT3_D_FF; k++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_ffn_h[k] = 0;
    }
EROSION_BWD: for (int d = 0; d < FT3_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int k_star = ak_ero_row[d];
        if (k_star < 0 || k_star >= FT3_D_FF) continue;
        acc_t g = grad_f[d];
        g_W2[d * FT3_D_FF + k_star] += (int32_t)g;
        grad_ffn_h[k_star]          += g;
        g_b2[d]                     += (int32_t)g;
    }
    static acc_t grad_h2[FT3_D];
GH2_INIT: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_h2[d] = 0;
    }
DILATION_BWD: for (int k = 0; k < FT3_D_FF; k++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int d_star = ad_dil_row[k];
        if (d_star < 0 || d_star >= FT3_D) continue;
        acc_t g = grad_ffn_h[k];
        g_W1[k * FT3_D + d_star] += (int32_t)g;
        grad_h2[d_star]          += g;
        g_b1[k]                  += (int32_t)g;
    }

    {
        acc_t sum = 0;
BS2_SUM: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            sum += grad_h2[d];
        }
        acc_t mean = sum >> FT3_LOG2_D;
BS2_ADD: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            grad_x_after[d] += grad_h2[d] - mean;
        }
    }

    // -------------------------------------------------------------------------
    // Backward through residual1 split (post-attention).
    // -------------------------------------------------------------------------
    static acc_t grad_x_t[FT3_D];
    static acc_t grad_a  [FT3_D];
RB_R1: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_t[d] = grad_x_after[d];
        grad_a  [d] = grad_x_after[d];
    }

    static acc_t grad_score[FT3_L_MAX];
GS_INIT: for (int n = 0; n < FT3_L_MAX; n++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_score[n] = 0;
    }
ATTN_VAL: for (int d = 0; d < FT3_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int n_star = an_out_row[d];
        if (n_star < 0 || n_star > m) continue;
        acc_t g = grad_a[d];
        grad_v_cache[n_star * FT3_D + d] += (int32_t)g;
        grad_score[n_star]               += g;
    }
    static acc_t grad_Q[FT3_D];
GQ_INIT: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_Q[d] = 0;
    }
ATTN_SCORE: for (int n = 0; n <= m; n++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int d_star = ad_score_row[n];
        if (d_star < 0 || d_star >= FT3_D) continue;
        acc_t gs = grad_score[n];
        grad_Q[d_star]                       += gs;
        grad_k_cache[n * FT3_D + d_star]     += (int32_t)gs;
    }
    static acc_t grad_h1[FT3_D];
ATTN_OUT: for (int d = 0; d < FT3_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        acc_t gV = (acc_t)grad_v_cache[m * FT3_D + d];
        acc_t gK = (acc_t)grad_k_cache[m * FT3_D + d];
        acc_t gQ = grad_Q[d];
        grad_h1[d] = gV + gQ + gK;
        g_s_q_row[d] += (int32_t)gQ;
        g_s_k_row[d] += (int32_t)gK;
    }

    {
        acc_t sum = 0;
BS1_SUM: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            sum += grad_h1[d];
        }
        acc_t mean = sum >> FT3_LOG2_D;
BS1_ADD: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            grad_x_t[d] += grad_h1[d] - mean;
        }
    }

    // -------------------------------------------------------------------------
    // Write upstream gradient for the next-deepest layer.
    //   layer > 0: write to HIDDEN_{layer-1}[m, :] (becomes that layer's input grad).
    //   layer 0:   accumulate into G_EMBED[token, :].
    // -------------------------------------------------------------------------
    if (layer_id > 0) {
        static int32_t hidden_grad_out[FT3_D];
HIDGRAD_CAST: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            hidden_grad_out[d] = (int32_t)grad_x_t[d];
        }
        burst_store_words(hidden_grad_out, buffer,
                          hidden_in_off(layer_id) + m * FT3_D, FT3_D);
    } else {
EMBED_BWD: for (int d = 0; d < FT3_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            g_embed_row[d] += (int32_t)grad_x_t[d];
        }
        burst_store_words(g_embed_row, buffer,
                          OFF3_G_EMBED + token_id_or_neg1 * FT3_D, FT3_D);
    }

    // Flush parameter / cache gradient stores.
    if (is_last) {
        burst_store_words(g_W_head, buffer, OFF3_G_W_HEAD, W3_W_HEAD);
    }
    burst_store_words(g_W1,        buffer, g_w1_off,                W3_W1);
    burst_store_words(g_W2,        buffer, g_w2_off,                W3_W2);
    burst_store_words(g_b1,        buffer, g_b1_off,                W3_B1);
    burst_store_words(g_b2,        buffer, g_b2_off,                W3_B2);
    burst_store_words(g_s_q_row,   buffer, g_sq_off + m * FT3_D,    FT3_D);
    burst_store_words(g_s_k_row,   buffer, g_sk_off + m * FT3_D,    FT3_D);
    burst_store_words(grad_k_cache, buffer, kc_off,                 W3_K_CACHE);
    burst_store_words(grad_v_cache, buffer, vc_off,                 W3_V_CACHE);

    return 0;
}

// =============================================================================
// SignSGD: 14 (param, grad) pairs across both layers + shared embed/head.
// signsgd_region_wide() lifted verbatim from forge_train_step.cpp.
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

static int do_signsgd_3layer(
    int32_t buffer[FT3_BUFFER_WORDS],
    int     lr,
    int     threshold
) {
#pragma HLS INLINE off
    // Shared embedding + LM head.
    signsgd_region_wide(buffer, OFF3_EMBED,    OFF3_G_EMBED,    W3_EMBED,    lr, threshold);
    signsgd_region_wide(buffer, OFF3_W_HEAD,   OFF3_G_W_HEAD,   W3_W_HEAD,   lr, threshold);

    // Layer 0 parameters.
    signsgd_region_wide(buffer, OFF3_L0_S_Q,   OFF3_G_L0_S_Q,   W3_S_Q,      lr, threshold);
    signsgd_region_wide(buffer, OFF3_L0_S_K,   OFF3_G_L0_S_K,   W3_S_K,      lr, threshold);
    signsgd_region_wide(buffer, OFF3_L0_W1,    OFF3_G_L0_W1,    W3_W1,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L0_B1,    OFF3_G_L0_B1,    W3_B1,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L0_W2,    OFF3_G_L0_W2,    W3_W2,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L0_B2,    OFF3_G_L0_B2,    W3_B2,       lr, threshold);

    // Layer 1 parameters.
    signsgd_region_wide(buffer, OFF3_L1_S_Q,   OFF3_G_L1_S_Q,   W3_S_Q,      lr, threshold);
    signsgd_region_wide(buffer, OFF3_L1_S_K,   OFF3_G_L1_S_K,   W3_S_K,      lr, threshold);
    signsgd_region_wide(buffer, OFF3_L1_W1,    OFF3_G_L1_W1,    W3_W1,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L1_B1,    OFF3_G_L1_B1,    W3_B1,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L1_W2,    OFF3_G_L1_W2,    W3_W2,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L1_B2,    OFF3_G_L1_B2,    W3_B2,       lr, threshold);

    // Layer 2 parameters.
    signsgd_region_wide(buffer, OFF3_L2_S_Q,   OFF3_G_L2_S_Q,   W3_S_Q,      lr, threshold);
    signsgd_region_wide(buffer, OFF3_L2_S_K,   OFF3_G_L2_S_K,   W3_S_K,      lr, threshold);
    signsgd_region_wide(buffer, OFF3_L2_W1,    OFF3_G_L2_W1,    W3_W1,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L2_B1,    OFF3_G_L2_B1,    W3_B1,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L2_W2,    OFF3_G_L2_W2,    W3_W2,       lr, threshold);
    signsgd_region_wide(buffer, OFF3_L2_B2,    OFF3_G_L2_B2,    W3_B2,       lr, threshold);
    return 0;
}

// =============================================================================
// Top-level dispatcher.  `layer_id` is arg4 (was unused in 1-layer kernel).
// =============================================================================
extern "C" int forge_train_step_3layer(
    int32_t  buffer[FT3_BUFFER_WORDS],
    int      mode,
    int      arg1,        // m (FWD/BWD) or lr (SIGNSGD)
    int      arg2,        // token (FWD/BWD) or threshold (SIGNSGD)
    int      arg3,        // target (FWD only); ignored otherwise
    int      layer_id     // 0, 1, or 2; ignored for SIGNSGD
) {
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=FT3_BUFFER_WORDS \
        num_write_outstanding=16 num_read_outstanding=16 \
        max_write_burst_length=16 max_read_burst_length=16 \
        max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite  port=mode      bundle=control
#pragma HLS INTERFACE s_axilite  port=arg1      bundle=control
#pragma HLS INTERFACE s_axilite  port=arg2      bundle=control
#pragma HLS INTERFACE s_axilite  port=arg3      bundle=control
#pragma HLS INTERFACE s_axilite  port=layer_id  bundle=control
#pragma HLS INTERFACE s_axilite  port=buffer    bundle=control
#pragma HLS INTERFACE s_axilite  port=return    bundle=control

    if (mode == MODE3_FWD_LOSS) return do_fwd_loss_layer(buffer, layer_id, arg1, arg2, arg3);
    if (mode == MODE3_BWD)      return do_bwd_layer    (buffer, layer_id, arg1, arg2);
    if (mode == MODE3_SIGNSGD)  return do_signsgd_3layer(buffer, arg1, arg2);
    return 99;
}
