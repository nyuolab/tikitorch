// forge_train_step_2layer.cpp — 2-layer extension of the Tier-1
// single-kernel train_step kernel (forge_train_step.cpp).  Same monolithic
// single-kernel design dispatched by mode + layer_id args (avoids the
// multi-kernel AXI mux issue documented in
// .claude/projects/.../memory/project_fpga_silicon_training.md).
//
// Per-helper structure mirrors the single-layer reference exactly; the only
// change is that all per-layer offsets are selected via L_OFF / L_GOFF
// macros at function entry.  Forward control flow:
//
//   layer 0 fwd:  input <- embed[token_id, :]
//                 ... transformer block (attn + FFN, both batch-shifted) ...
//                 output -> hidden[m, :]
//   layer 1 fwd:  input <- hidden[m, :]
//                 ... transformer block ...
//                 output (y_t) feeds LM head -> ad_head, grad_logit, pred
//
// Backward control flow (reverse):
//
//   layer 1 bwd:  upstream <- LM head adjoint
//                 ... block backward ...
//                 grad_x_t -> hidden[m, :]   (reused as gradient slot)
//   layer 0 bwd:  upstream <- hidden[m, :]   (gradient written by layer 1)
//                 ... block backward ...
//                 grad_x_t -> g_embed[token_id, :]
//
// Reuse policy mirrors single-layer kernel:
//   - K_CACHE / V_CACHE store K/V activations during fwd; reused as
//     gradient accumulators during bwd (host clears between fwd/bwd).
//   - HIDDEN stores layer 0's fwd output, then layer 1's fwd input, then
//     layer 1's bwd grad output, then layer 0's bwd grad input.  Host
//     does NOT need to clear it because each phase fully overwrites.

#include "forge_buffer_2layer.hpp"
#include "../forge_hls_types.hpp"
#include <cstdint>

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_uint<512> wide_t;
#endif

// =============================================================================
// Per-layer offset selectors (template-free; emitted as straight muxes).
// =============================================================================
static inline int LAYER_OFF(int layer_id, int l0_off, int l1_off) {
#pragma HLS INLINE
    return (layer_id == 0) ? l0_off : l1_off;
}

#define L_OFF(LAYER, TENSOR)  LAYER_OFF((LAYER), OFF2_L0_##TENSOR,   OFF2_L1_##TENSOR)
#define L_GOFF(LAYER, TENSOR) LAYER_OFF((LAYER), OFF2_G_L0_##TENSOR, OFF2_G_L1_##TENSOR)

// =============================================================================
// Per-layer word counts (alias the single-layer constants for brevity).
// All identical to forge_buffer.hpp's WORDS_* because the per-layer
// shape (D, D_FF, V, L_MAX) is the same.
// =============================================================================
constexpr int WORDS2_EMBED      = W2_EMBED;
constexpr int WORDS2_W_HEAD     = W2_W_HEAD;
constexpr int WORDS2_W1         = W2_W1;
constexpr int WORDS2_B1         = W2_B1;
constexpr int WORDS2_W2         = W2_W2;
constexpr int WORDS2_B2         = W2_B2;
constexpr int WORDS2_K_CACHE    = W2_K_CACHE;
constexpr int WORDS2_V_CACHE    = W2_V_CACHE;
constexpr int WORDS2_S_Q        = W2_S_Q;
constexpr int WORDS2_S_K        = W2_S_K;
constexpr int WORDS2_AD_SCORE   = W2_AD_SCORE;
constexpr int WORDS2_AN_OUT     = W2_AN_OUT;
constexpr int WORDS2_AD_DIL     = W2_AD_DIL;
constexpr int WORDS2_AK_ERO     = W2_AK_ERO;
constexpr int WORDS2_AD_HEAD    = W2_AD_HEAD;
constexpr int WORDS2_GRAD_LOGIT = W2_GRAD_LOGIT;

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

static void batch_shift(const data_t x[FT2_D], data_t y[FT2_D]) {
#pragma HLS INLINE
    acc_t sum = 0;
BS_SUM: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        sum += (acc_t)x[d];
    }
    acc_t mean = sum >> FT2_LOG2_D;
BS_SUB: for (int d = 0; d < FT2_D; d++) {
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
    int32_t buffer[FT2_BUFFER_WORDS],
    int     layer_id,
    int     m,
    int     token_id,
    int     target_id_or_neg1  // -1 means no loss (used for layer-0 calls; we
                               // also accept >=0 from the host for both layers
                               // — only layer 1 consumes the target).
) {
#pragma HLS INLINE off
    if (layer_id < 0 || layer_id >= FT2_N_LAYERS) return 4;
    if (m < 0 || m >= FT2_L_MAX)                  return 1;
    if (layer_id == 0) {
        if (token_id < 0 || token_id >= FT2_V)    return 2;
    }
    if (layer_id == 1 && target_id_or_neg1 >= 0) {
        if (target_id_or_neg1 >= FT2_V)           return 3;
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

    static int32_t W1_buf[WORDS2_W1];
    static int32_t b1_buf[WORDS2_B1];
    static int32_t W2_buf[WORDS2_W2];
    static int32_t b2_buf[WORDS2_B2];
    static int32_t W_head_buf[WORDS2_W_HEAD];
    static int32_t s_q_row[FT2_D];
    static int32_t s_k_row[FT2_D];
    static int32_t input_row[FT2_D];
    static int32_t k_cache_buf[WORDS2_K_CACHE];
    static int32_t v_cache_buf[WORDS2_V_CACHE];

    burst_load_words(buffer, w1_off,                W1_buf,     WORDS2_W1);
    burst_load_words(buffer, b1_off,                b1_buf,     WORDS2_B1);
    burst_load_words(buffer, w2_off,                W2_buf,     WORDS2_W2);
    burst_load_words(buffer, b2_off,                b2_buf,     WORDS2_B2);
    if (layer_id == 1) {
        burst_load_words(buffer, OFF2_W_HEAD,       W_head_buf, WORDS2_W_HEAD);
    }
    burst_load_words(buffer, sq_off  + m * FT2_D,   s_q_row,    FT2_D);
    burst_load_words(buffer, sk_off  + m * FT2_D,   s_k_row,    FT2_D);
    // Input row: layer 0 = embedding lookup; layer 1 = hidden[m,:].
    if (layer_id == 0) {
        burst_load_words(buffer, OFF2_EMBED + token_id * FT2_D, input_row, FT2_D);
    } else {
        burst_load_words(buffer, OFF2_HIDDEN + m * FT2_D,       input_row, FT2_D);
    }
    burst_load_words(buffer, kc_off,                k_cache_buf, WORDS2_K_CACHE);
    burst_load_words(buffer, vc_off,                v_cache_buf, WORDS2_V_CACHE);

    data_t x_t[FT2_D];
EMBED_CAST: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        x_t[d] = (data_t)(int16_t)(input_row[d] & 0xFFFF);
    }

    data_t h1[FT2_D];
    batch_shift(x_t, h1);

    data_t Q_t[FT2_D];
QKV: for (int d = 0; d < FT2_D; d++) {
#pragma HLS PIPELINE II=1
        data_t s_q_d = (data_t)(int16_t)(s_q_row[d] & 0xFFFF);
        data_t s_k_d = (data_t)(int16_t)(s_k_row[d] & 0xFFFF);
        acc_t  q     = (acc_t)h1[d] + (acc_t)s_q_d;
        acc_t  k     = (acc_t)h1[d] + (acc_t)s_k_d;
        Q_t[d] = sat16(q);
        k_cache_buf[m * FT2_D + d] = (int32_t)(int16_t)sat16(k);
        v_cache_buf[m * FT2_D + d] = (int32_t)(int16_t)h1[d];
    }

    static acc_t   scores[FT2_L_MAX];
    static int32_t ad_score_row[FT2_L_MAX];
SCORE_INIT: for (int n = 0; n < FT2_L_MAX; n++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        ad_score_row[n] = -1;
    }
SCORE_N: for (int n = 0; n <= m; n++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
SCORE_D: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
            data_t k_v = (data_t)(int16_t)(k_cache_buf[n * FT2_D + d] & 0xFFFF);
            acc_t s = (acc_t)Q_t[d] + (acc_t)k_v;
            if (s > mx) { mx = s; idx_d = d; }
        }
        scores[n] = mx;
        ad_score_row[n] = idx_d;
    }

    data_t a[FT2_D];
    static int32_t an_out_row[FT2_D];
PULL_D: for (int d = 0; d < FT2_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_n = 0;
PULL_N: for (int n = 0; n <= m; n++) {
#pragma HLS UNROLL factor=8
            data_t v_v = (data_t)(int16_t)(v_cache_buf[n * FT2_D + d] & 0xFFFF);
            acc_t s = (acc_t)v_v + scores[n];
            if (s > mx) { mx = s; idx_n = n; }
        }
        a[d] = sat16(mx);
        an_out_row[d] = idx_n;
    }

    data_t x_after[FT2_D];
RES1: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        x_after[d] = sat16((acc_t)x_t[d] + (acc_t)a[d]);
    }

    data_t h2[FT2_D];
    batch_shift(x_after, h2);

    static data_t  ffn_h[FT2_D_FF];
    static int32_t ad_dil_row[FT2_D_FF];
FFN_DIL_K: for (int k = 0; k < FT2_D_FF; k++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
FFN_DIL_D: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
            data_t W1_kd = (data_t)(int16_t)(W1_buf[k * FT2_D + d] & 0xFFFF);
            acc_t s = (acc_t)W1_kd + (acc_t)h2[d];
            if (s > mx) { mx = s; idx_d = d; }
        }
        data_t b1_k = (data_t)(int16_t)(b1_buf[k] & 0xFFFF);
        ffn_h[k] = sat16(mx + (acc_t)b1_k);
        ad_dil_row[k] = idx_d;
    }

    data_t f[FT2_D];
    static int32_t ak_ero_row[FT2_D];
FFN_ERO_D: for (int d = 0; d < FT2_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t  mn = (acc_t) 2147483647LL;
        int    idx_k = 0;
FFN_ERO_K: for (int k = 0; k < FT2_D_FF; k++) {
#pragma HLS UNROLL factor=8
            data_t W2_dk = (data_t)(int16_t)(W2_buf[d * FT2_D_FF + k] & 0xFFFF);
            acc_t s = (acc_t)W2_dk + (acc_t)ffn_h[k];
            if (s < mn) { mn = s; idx_k = k; }
        }
        data_t b2_d = (data_t)(int16_t)(b2_buf[d] & 0xFFFF);
        f[d] = sat16(mn + (acc_t)b2_d);
        ak_ero_row[d] = idx_k;
    }

    data_t y_t[FT2_D];
RES2: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        y_t[d] = sat16((acc_t)x_after[d] + (acc_t)f[d]);
    }

    // Store per-layer activations / caches (shared by both layers).
    burst_store_words(ad_score_row,             buffer, score_off + m * FT2_L_MAX, FT2_L_MAX);
    burst_store_words(an_out_row,               buffer, an_off    + m * FT2_D,     FT2_D);
    burst_store_words(ad_dil_row,               buffer, dil_off   + m * FT2_D_FF,  FT2_D_FF);
    burst_store_words(ak_ero_row,               buffer, ero_off   + m * FT2_D,     FT2_D);
    burst_store_words(k_cache_buf + m * FT2_D,  buffer, kc_off    + m * FT2_D,     FT2_D);
    burst_store_words(v_cache_buf + m * FT2_D,  buffer, vc_off    + m * FT2_D,     FT2_D);

    if (layer_id == 0) {
        // Layer 0 block output -> HIDDEN[m, :].  No LM head, no loss.
        static int32_t y_out_row[FT2_D];
HIDDEN_CAST: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            y_out_row[d] = (int32_t)(int16_t)y_t[d];
        }
        burst_store_words(y_out_row, buffer, OFF2_HIDDEN + m * FT2_D, FT2_D);
        return 0;
    }

    // ===== layer_id == 1: LM head + perceptron loss =====
    static data_t  logits[FT2_V];
    static int32_t ad_head_row[FT2_V];
HEAD_V: for (int v = 0; v < FT2_V; v++) {
#pragma HLS PIPELINE II=1
        acc_t  mx = (acc_t)-2147483648LL;
        int    idx_d = 0;
HEAD_D: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
            data_t W_vd = (data_t)(int16_t)(W_head_buf[v * FT2_D + d] & 0xFFFF);
            acc_t s = (acc_t)W_vd + (acc_t)y_t[d];
            if (s > mx) { mx = s; idx_d = d; }
        }
        logits[v] = sat16(mx);
        ad_head_row[v] = idx_d;
    }

    // Kernel writes raw int16 logits to the dedicated OFF2_LOGITS scratchpad.
    // The host reads them and writes the loss gradient (perceptron, or the
    // closure-faithful softmax cross-entropy of tikitorch.tropical_cross_entropy)
    // to OFF2_GRAD_LOGIT, which do_bwd_layer then consumes.  Keeping logits-out
    // (OFF2_LOGITS) separate from grad-in (OFF2_GRAD_LOGIT) removes the previous
    // aliasing so the host CE can be validated bit-exactly against the library.
    // This keeps pred/argmax/perceptron-grad logic off the kernel (less code,
    // easier timing — the path that closed at WNS -0.155 ns).
    (void)target_id_or_neg1;  // unused: loss now computed host-side
    static int32_t logits_i32_row[FT2_V];
LOGITS_OUT: for (int v = 0; v < FT2_V; v++) {
#pragma HLS PIPELINE II=1
        logits_i32_row[v] = (int32_t)(int16_t)logits[v];
    }

    burst_store_words(ad_head_row,    buffer, OFF2_AD_HEAD + m * FT2_V, FT2_V);
    burst_store_words(logits_i32_row, buffer, OFF2_LOGITS  + m * FT2_V, FT2_V);

    return 0;
}

// =============================================================================
// Backward through one layer at one position m.
// Body lifted from do_bwd() in forge_train_step.cpp and parameterised
// on layer_id.  Layer 1's head-adjoint seeds grad_x_layer; layer 0's
// seeds it from HIDDEN[m, :] (written by layer 1's backward pass).
// =============================================================================
static int do_bwd_layer(
    int32_t buffer[FT2_BUFFER_WORDS],
    int     layer_id,
    int     m,
    int     token_id_or_neg1
) {
#pragma HLS INLINE off
    if (layer_id < 0 || layer_id >= FT2_N_LAYERS) return 4;
    if (m < 0 || m >= FT2_L_MAX)                  return 1;
    if (layer_id == 0) {
        if (token_id_or_neg1 < 0 || token_id_or_neg1 >= FT2_V) return 2;
    }

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

    static int32_t ad_head_row [FT2_V];
    static int32_t ad_dil_row  [FT2_D_FF];
    static int32_t ak_ero_row  [FT2_D];
    static int32_t ad_score_row[FT2_L_MAX];
    static int32_t an_out_row  [FT2_D];
    static int32_t grad_logit_row[FT2_V];

    burst_load_words(buffer, dil_off   + m * FT2_D_FF,  ad_dil_row,     FT2_D_FF);
    burst_load_words(buffer, ero_off   + m * FT2_D,     ak_ero_row,     FT2_D);
    burst_load_words(buffer, score_off + m * FT2_L_MAX, ad_score_row,   FT2_L_MAX);
    burst_load_words(buffer, an_off    + m * FT2_D,     an_out_row,     FT2_D);
    if (layer_id == 1) {
        burst_load_words(buffer, OFF2_AD_HEAD    + m * FT2_V, ad_head_row,    FT2_V);
        burst_load_words(buffer, OFF2_GRAD_LOGIT + m * FT2_V, grad_logit_row, FT2_V);
    }

    static int32_t g_W_head[W2_W_HEAD];
    static int32_t g_W1    [W2_W1];
    static int32_t g_W2    [W2_W2];
    static int32_t g_b1    [W2_B1];
    static int32_t g_b2    [W2_B2];
    static int32_t g_s_q_row[FT2_D];
    static int32_t g_s_k_row[FT2_D];
    static int32_t g_embed_row[FT2_D];
#pragma HLS BIND_STORAGE variable=g_W_head type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=g_W1     type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=g_W2     type=ram_t2p impl=uram

    if (layer_id == 1) {
        burst_load_words(buffer, OFF2_G_W_HEAD, g_W_head, W2_W_HEAD);
    }
    burst_load_words(buffer, g_w1_off,                    g_W1,       W2_W1);
    burst_load_words(buffer, g_w2_off,                    g_W2,       W2_W2);
    burst_load_words(buffer, g_b1_off,                    g_b1,       W2_B1);
    burst_load_words(buffer, g_b2_off,                    g_b2,       W2_B2);
    burst_load_words(buffer, g_sq_off + m * FT2_D,        g_s_q_row,  FT2_D);
    burst_load_words(buffer, g_sk_off + m * FT2_D,        g_s_k_row,  FT2_D);
    if (layer_id == 0) {
        burst_load_words(buffer, OFF2_G_EMBED + token_id_or_neg1 * FT2_D,
                         g_embed_row, FT2_D);
    }

    static int32_t grad_k_cache[W2_K_CACHE];
    static int32_t grad_v_cache[W2_V_CACHE];
#pragma HLS BIND_STORAGE variable=grad_k_cache type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=grad_v_cache type=ram_t2p impl=uram
    burst_load_words(buffer, kc_off, grad_k_cache, W2_K_CACHE);
    burst_load_words(buffer, vc_off, grad_v_cache, W2_V_CACHE);

    // -------------------------------------------------------------------------
    // Seed grad_x_layer with the upstream gradient.
    //   layer 1: from LM head adjoint (head writes to W_head and grad_x_layer)
    //   layer 0: from HIDDEN[m, :] which layer 1's bwd previously wrote.
    // -------------------------------------------------------------------------
    static acc_t grad_x_layer[FT2_D];
GXL_INIT: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_layer[d] = 0;
    }

    if (layer_id == 1) {
HEAD_BWD_V: for (int v = 0; v < FT2_V; v++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
            int32_t g = grad_logit_row[v];
            if (g == 0) continue;
            int d_star = ad_head_row[v];
            if (d_star < 0 || d_star >= FT2_D) continue;
            g_W_head[v * FT2_D + d_star] += g;
            grad_x_layer[d_star]         += (acc_t)g;
        }
    } else {
        // Layer 0: upstream comes from HIDDEN[m, :] (written by layer 1 bwd).
        static int32_t hidden_grad_row[FT2_D];
        burst_load_words(buffer, OFF2_HIDDEN + m * FT2_D, hidden_grad_row, FT2_D);
GXL_LOAD0: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            grad_x_layer[d] = (acc_t)hidden_grad_row[d];
        }
    }

    // -------------------------------------------------------------------------
    // Backward through the residual2 split (post-FFN).
    // -------------------------------------------------------------------------
    static acc_t grad_x_after[FT2_D];
    static acc_t grad_f      [FT2_D];
RB_R2: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_after[d] = grad_x_layer[d];
        grad_f      [d] = grad_x_layer[d];
    }

    static acc_t grad_ffn_h[FT2_D_FF];
GFH_INIT: for (int k = 0; k < FT2_D_FF; k++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_ffn_h[k] = 0;
    }
EROSION_BWD: for (int d = 0; d < FT2_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int k_star = ak_ero_row[d];
        if (k_star < 0 || k_star >= FT2_D_FF) continue;
        acc_t g = grad_f[d];
        g_W2[d * FT2_D_FF + k_star] += (int32_t)g;
        grad_ffn_h[k_star]          += g;
        g_b2[d]                     += (int32_t)g;
    }
    static acc_t grad_h2[FT2_D];
GH2_INIT: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_h2[d] = 0;
    }
DILATION_BWD: for (int k = 0; k < FT2_D_FF; k++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int d_star = ad_dil_row[k];
        if (d_star < 0 || d_star >= FT2_D) continue;
        acc_t g = grad_ffn_h[k];
        g_W1[k * FT2_D + d_star] += (int32_t)g;
        grad_h2[d_star]          += g;
        g_b1[k]                  += (int32_t)g;
    }

    {
        acc_t sum = 0;
BS2_SUM: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            sum += grad_h2[d];
        }
        acc_t mean = sum >> FT2_LOG2_D;
BS2_ADD: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            grad_x_after[d] += grad_h2[d] - mean;
        }
    }

    // -------------------------------------------------------------------------
    // Backward through residual1 split (post-attention).
    // -------------------------------------------------------------------------
    static acc_t grad_x_t[FT2_D];
    static acc_t grad_a  [FT2_D];
RB_R1: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_x_t[d] = grad_x_after[d];
        grad_a  [d] = grad_x_after[d];
    }

    static acc_t grad_score[FT2_L_MAX];
GS_INIT: for (int n = 0; n < FT2_L_MAX; n++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_score[n] = 0;
    }
ATTN_VAL: for (int d = 0; d < FT2_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int n_star = an_out_row[d];
        if (n_star < 0 || n_star > m) continue;
        acc_t g = grad_a[d];
        grad_v_cache[n_star * FT2_D + d] += (int32_t)g;
        grad_score[n_star]               += g;
    }
    static acc_t grad_Q[FT2_D];
GQ_INIT: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        grad_Q[d] = 0;
    }
ATTN_SCORE: for (int n = 0; n <= m; n++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        int d_star = ad_score_row[n];
        if (d_star < 0 || d_star >= FT2_D) continue;
        acc_t gs = grad_score[n];
        grad_Q[d_star]                       += gs;
        grad_k_cache[n * FT2_D + d_star]     += (int32_t)gs;
    }
    static acc_t grad_h1[FT2_D];
ATTN_OUT: for (int d = 0; d < FT2_D; d++) {
#pragma HLS PIPELINE II=16
#pragma HLS LATENCY min=4 max=8
        acc_t gV = (acc_t)grad_v_cache[m * FT2_D + d];
        acc_t gK = (acc_t)grad_k_cache[m * FT2_D + d];
        acc_t gQ = grad_Q[d];
        grad_h1[d] = gV + gQ + gK;
        g_s_q_row[d] += (int32_t)gQ;
        g_s_k_row[d] += (int32_t)gK;
    }

    {
        acc_t sum = 0;
BS1_SUM: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            sum += grad_h1[d];
        }
        acc_t mean = sum >> FT2_LOG2_D;
BS1_ADD: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            grad_x_t[d] += grad_h1[d] - mean;
        }
    }

    // -------------------------------------------------------------------------
    // Write upstream gradient for the next-deepest layer.
    //   layer 1: write to HIDDEN[m, :] (becomes layer 0's input grad).
    //   layer 0: accumulate into G_EMBED[token, :].
    // -------------------------------------------------------------------------
    if (layer_id == 1) {
        static int32_t hidden_grad_out[FT2_D];
HIDGRAD_CAST: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            hidden_grad_out[d] = (int32_t)grad_x_t[d];
        }
        burst_store_words(hidden_grad_out, buffer, OFF2_HIDDEN + m * FT2_D, FT2_D);
    } else {
EMBED_BWD: for (int d = 0; d < FT2_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            g_embed_row[d] += (int32_t)grad_x_t[d];
        }
        burst_store_words(g_embed_row, buffer,
                          OFF2_G_EMBED + token_id_or_neg1 * FT2_D, FT2_D);
    }

    // Flush parameter / cache gradient stores.
    if (layer_id == 1) {
        burst_store_words(g_W_head, buffer, OFF2_G_W_HEAD, W2_W_HEAD);
    }
    burst_store_words(g_W1,        buffer, g_w1_off,                W2_W1);
    burst_store_words(g_W2,        buffer, g_w2_off,                W2_W2);
    burst_store_words(g_b1,        buffer, g_b1_off,                W2_B1);
    burst_store_words(g_b2,        buffer, g_b2_off,                W2_B2);
    burst_store_words(g_s_q_row,   buffer, g_sq_off + m * FT2_D,    FT2_D);
    burst_store_words(g_s_k_row,   buffer, g_sk_off + m * FT2_D,    FT2_D);
    burst_store_words(grad_k_cache, buffer, kc_off,                 W2_K_CACHE);
    burst_store_words(grad_v_cache, buffer, vc_off,                 W2_V_CACHE);

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

static int do_signsgd_2layer(
    int32_t buffer[FT2_BUFFER_WORDS],
    int     lr,
    int     threshold
) {
#pragma HLS INLINE off
    // Shared embedding + LM head.
    signsgd_region_wide(buffer, OFF2_EMBED,    OFF2_G_EMBED,    W2_EMBED,    lr, threshold);
    signsgd_region_wide(buffer, OFF2_W_HEAD,   OFF2_G_W_HEAD,   W2_W_HEAD,   lr, threshold);

    // Layer 0 parameters.
    signsgd_region_wide(buffer, OFF2_L0_S_Q,   OFF2_G_L0_S_Q,   W2_S_Q,      lr, threshold);
    signsgd_region_wide(buffer, OFF2_L0_S_K,   OFF2_G_L0_S_K,   W2_S_K,      lr, threshold);
    signsgd_region_wide(buffer, OFF2_L0_W1,    OFF2_G_L0_W1,    W2_W1,       lr, threshold);
    signsgd_region_wide(buffer, OFF2_L0_B1,    OFF2_G_L0_B1,    W2_B1,       lr, threshold);
    signsgd_region_wide(buffer, OFF2_L0_W2,    OFF2_G_L0_W2,    W2_W2,       lr, threshold);
    signsgd_region_wide(buffer, OFF2_L0_B2,    OFF2_G_L0_B2,    W2_B2,       lr, threshold);

    // Layer 1 parameters.
    signsgd_region_wide(buffer, OFF2_L1_S_Q,   OFF2_G_L1_S_Q,   W2_S_Q,      lr, threshold);
    signsgd_region_wide(buffer, OFF2_L1_S_K,   OFF2_G_L1_S_K,   W2_S_K,      lr, threshold);
    signsgd_region_wide(buffer, OFF2_L1_W1,    OFF2_G_L1_W1,    W2_W1,       lr, threshold);
    signsgd_region_wide(buffer, OFF2_L1_B1,    OFF2_G_L1_B1,    W2_B1,       lr, threshold);
    signsgd_region_wide(buffer, OFF2_L1_W2,    OFF2_G_L1_W2,    W2_W2,       lr, threshold);
    signsgd_region_wide(buffer, OFF2_L1_B2,    OFF2_G_L1_B2,    W2_B2,       lr, threshold);
    return 0;
}

// =============================================================================
// Top-level dispatcher.  `layer_id` is arg4 (was unused in 1-layer kernel).
// =============================================================================
extern "C" int forge_train_step_2layer(
    int32_t  buffer[FT2_BUFFER_WORDS],
    int      mode,
    int      arg1,        // m (FWD/BWD) or lr (SIGNSGD)
    int      arg2,        // token (FWD/BWD) or threshold (SIGNSGD)
    int      arg3,        // target (FWD only); ignored otherwise
    int      layer_id     // 0 or 1; ignored for SIGNSGD
) {
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=FT2_BUFFER_WORDS \
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

    if (mode == MODE2_FWD_LOSS) return do_fwd_loss_layer(buffer, layer_id, arg1, arg2, arg3);
    if (mode == MODE2_BWD)      return do_bwd_layer    (buffer, layer_id, arg1, arg2);
    if (mode == MODE2_SIGNSGD)  return do_signsgd_2layer(buffer, arg1, arg2);
    return 99;
}
