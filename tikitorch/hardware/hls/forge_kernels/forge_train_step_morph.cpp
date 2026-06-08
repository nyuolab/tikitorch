// forge_train_step_morph.cpp — monolithic single-kernel MorphSSM-core
// training kernel, dispatched by `mode`. Mirrors the silicon-validated
// structure of forge_train_step.cpp (wide-AXI burst load/store, sat16,
// batch_shift mean-subtract normalization, sparse-scatter backward through
// saved argmax indices, signsgd_region_wide) but implements the
// morphological core (alternating max-plus dilation / min-plus erosion
// depthwise convs + tropical projection + residual) instead of tropical
// attention.
//
// HYBRID: host owns embed + classification head + cross-entropy; this
// kernel owns the morph core fwd/bwd/update.
//
//   mode 0 (FWD):     read X[L_SEQ][D] -> run L_LAYERS -> mean-pool ->
//                     write POOLED[D]; also save TAP_IDX / PROJ_IDX /
//                     residual-stream activations for backward.
//   mode 1 (BWD):     read GRAD_POOLED[D] + saved indices -> backprop
//                     through mean-pool + all layers (reverse) -> accumulate
//                     into g_MH / g_MP.
//   mode 2 (SIGNSGD): apply signsgd to MH and MP weight regions vs their
//                     grad regions; zero the grads.
//
// EVERYTHING on the core data path is compare / add / shift only — ZERO
// multiplications. (Index arithmetic like c*K uses `*` but is address
// computation on small loop counters, not a data-path multiply; the actual
// op math is add/compare/shift.)
//
// Returns 0 on success, mode-specific error code otherwise.

#include "forge_buffer_morph.hpp"

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_uint<512> wide_t;
#endif

// Tropical sentinels (int32 extremes), reused for max/min init.
static const acc_t MD_NEG_INF = acc_t(-2147483647LL);
static const acc_t MD_POS_INF = acc_t( 2147483647LL);

// =============================================================================
// Common helpers (shared across all modes) — mirrors forge_train_step.cpp.
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

// Store an int16 activation row (t-major [L_SEQ][D]) into the int32 buffer,
// sign-extending to 32 bits (matches the k_cache write idiom in the
// forge_train_step.cpp template). Used for the residual-stream activations.
static void store_rstream(const data_t R[MD_L_SEQ][MD_D], int32_t* buffer, int word_off) {
#pragma HLS INLINE off
RS_T: for (int t = 0; t < MD_L_SEQ; t++) {
RS_D: for (int d = 0; d < MD_D; d++) {
#pragma HLS PIPELINE II=1
            buffer[word_off + t * MD_D + d] = (int32_t)(int16_t)R[t][d];
        }
    }
}

// batch_shift: subtract the per-token mean over the D feature axis. Matches
// BS_SUM/BS_SUB in forge_train_step.cpp (mean = sum >> LOG2_D; pure shift,
// no divider, no multiply).
static void batch_shift_row(const data_t x[MD_D], data_t y[MD_D]) {
#pragma HLS INLINE
    acc_t sum = 0;
BS_SUM: for (int d = 0; d < MD_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        sum += (acc_t)x[d];
    }
    acc_t mean = sum >> MD_LOG2_D;
BS_SUB: for (int d = 0; d < MD_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        y[d] = sat16((acc_t)x[d] - mean);
    }
}

// =============================================================================
// Mode 0: do_fwd — morph core forward + mean-pool.
// =============================================================================
static int do_fwd(int32_t* buffer) {
#pragma HLS INLINE off

    // Weights for all layers (small; fit on-chip). Loaded once.
    static int32_t mh_buf[WORDS_MH];   // morph conv kernels [l][c][k]
    static int32_t mp_buf[WORDS_MP];   // tropical proj      [l][j][i]
    burst_load_words(buffer, OFF_MH, mh_buf, WORDS_MH);
    burst_load_words(buffer, OFF_MP, mp_buf, WORDS_MP);

    // Residual stream. R holds the current layer input (t-major [t][d]).
    static data_t R[MD_L_SEQ][MD_D];
LOAD_X: for (int t = 0; t < MD_L_SEQ; t++) {
        for (int d = 0; d < MD_D; d++) {
#pragma HLS PIPELINE II=1
            R[t][d] = (data_t)(int16_t)(buffer[OFF_X + t * MD_D + d] & 0xFFFF);
        }
    }

    // Per-layer scratch.
    static data_t BS [MD_L_SEQ][MD_D];   // batch-shifted input  [t][d]
    static data_t MC [MD_L_SEQ][MD_D];   // morph conv output    [t][c]
    static data_t PJ [MD_L_SEQ][MD_D];   // tropical proj output [t][j]
    static int32_t tap_idx [MD_D][MD_L_SEQ];     // [c][t]
    static int32_t proj_idx[MD_L_SEQ][MD_D];     // [t][j]

    // Save slot 0 of the residual stream = the embedded input X (layer-0 in).
    store_rstream(R, buffer, off_rstream(0));

LAYER_LOOP: for (int l = 0; l < MD_L_LAYERS; l++) {
        const int dil    = dilation_of_layer(l);
        const bool is_dil = ((l & 1) == 0);   // even -> dilation(max), odd -> erosion(min)

        // ---- batch_shift over D, per time step ----
BS_T: for (int t = 0; t < MD_L_SEQ; t++) {
            batch_shift_row(R[t], BS[t]);
        }

        // ---- depthwise morph conv (causal, dilation dil) ----
        // MC[t][c] = (max|min)_k ( h[l][c][k] + BS[t - k*dil][c] ),  t-k*dil>=0
        // NOTE: mh_buf / mp_buf are LOCAL arrays already holding the whole
        // weight region, so the per-layer base is l*(...) only (NOT off_mh()
        // which adds the global OFF_MH offset used for the DDR address).
        const int hbase = l * (MD_D * MD_K);
MC_C: for (int c = 0; c < MD_D; c++) {
MC_T: for (int t = 0; t < MD_L_SEQ; t++) {
#pragma HLS PIPELINE II=1
                acc_t best = is_dil ? MD_NEG_INF : MD_POS_INF;
                int   kstar = 0;
MC_K: for (int k = 0; k < MD_K; k++) {
#pragma HLS UNROLL
                    int t_in = t - k * dil;   // tap offset (index arith on loop counters)
                    if (t_in >= 0) {
                        data_t hck = (data_t)(int16_t)(mh_buf[hbase + c * MD_K + k] & 0xFFFF);
                        acc_t s = (acc_t)hck + (acc_t)BS[t_in][c];
                        if (is_dil) { if (s > best) { best = s; kstar = k; } }
                        else        { if (s < best) { best = s; kstar = k; } }
                    }
                }
                MC[t][c]      = sat16(best);
                tap_idx[c][t] = kstar;
            }
        }

        // ---- tropical projection (max-plus), per time step ----
        // PJ[t][j] = max_i ( P[l][j][i] + MC[t][i] ),  argmax index i*
        const int pbase = l * (MD_D * MD_D);   // local base into mp_buf
PJ_T: for (int t = 0; t < MD_L_SEQ; t++) {
PJ_J: for (int j = 0; j < MD_D; j++) {
#pragma HLS PIPELINE II=1
                acc_t best = MD_NEG_INF;
                int   istar = 0;
PJ_I: for (int i = 0; i < MD_D; i++) {
#pragma HLS UNROLL factor=8
                    data_t pji = (data_t)(int16_t)(mp_buf[pbase + j * MD_D + i] & 0xFFFF);
                    acc_t s = (acc_t)pji + (acc_t)MC[t][i];
                    if (s > best) { best = s; istar = i; }
                }
                PJ[t][j]       = sat16(best);
                proj_idx[t][j] = istar;
            }
        }

        // ---- residual add -> next layer input (overwrite R in place) ----
RES: for (int t = 0; t < MD_L_SEQ; t++) {
            for (int j = 0; j < MD_D; j++) {
#pragma HLS PIPELINE II=1
                R[t][j] = sat16((acc_t)R[t][j] + (acc_t)PJ[t][j]);
            }
        }

        // ---- persist saved tensors for backward ----
        burst_store_words(&tap_idx[0][0],  buffer, off_tap(l),  MD_D * MD_L_SEQ);
        burst_store_words(&proj_idx[0][0], buffer, off_proj(l), MD_L_SEQ * MD_D);
        store_rstream(R, buffer, off_rstream(l + 1));
    }

    // ---- mean-pool over time -> POOLED[j] = (sum_t Rout[t][j]) >> LOG2_L ----
    static int32_t pooled[MD_D];
POOL_J: for (int j = 0; j < MD_D; j++) {
#pragma HLS PIPELINE II=1
        acc_t sum = 0;
POOL_T: for (int t = 0; t < MD_L_SEQ; t++) {
#pragma HLS UNROLL factor=8
            sum += (acc_t)R[t][j];
        }
        pooled[j] = (int32_t)(int16_t)sat16(sum >> MD_LOG2_L);
    }
    burst_store_words(pooled, buffer, OFF_POOLED, MD_D);

    return 0;
}

// =============================================================================
// Mode 1: do_bwd — backprop GRAD_POOLED through mean-pool + all layers.
// Sparse-scatter through saved argmax indices; batch_shift bwd = subtract
// the per-token mean of the incoming grad (mirror of BS1/BS2 in template).
// Accumulates into g_MH / g_MP.
// =============================================================================
static int do_bwd(int32_t* buffer) {
#pragma HLS INLINE off

    // Gradient accumulators (read-modify-write across all layers).
    static int32_t g_mh[WORDS_G_MH];
    static int32_t g_mp[WORDS_G_MP];
    burst_load_words(buffer, OFF_G_MH, g_mh, WORDS_G_MH);
    burst_load_words(buffer, OFF_G_MP, g_mp, WORDS_G_MP);

    // ---- mean-pool backward: grad_R_out[t][j] = GRAD_POOLED[j] >> LOG2_L ----
    static int32_t grad_pooled[MD_D];
    burst_load_words(buffer, OFF_GRAD_POOLED, grad_pooled, MD_D);

    static acc_t gR[MD_L_SEQ][MD_D];   // grad wrt current layer output (Rout)
GP_T: for (int t = 0; t < MD_L_SEQ; t++) {
        for (int j = 0; j < MD_D; j++) {
#pragma HLS PIPELINE II=1
            gR[t][j] = (acc_t)grad_pooled[j] >> MD_LOG2_L;
        }
    }

    // Per-layer scratch.
    static int32_t tap_idx [MD_D][MD_L_SEQ];
    static int32_t proj_idx[MD_L_SEQ][MD_D];
    static acc_t   grad_PJ [MD_L_SEQ][MD_D];
    static acc_t   grad_MC [MD_L_SEQ][MD_D];
    static acc_t   grad_BS [MD_L_SEQ][MD_D];
    static acc_t   grad_R  [MD_L_SEQ][MD_D];   // grad wrt layer input R

BWD_LAYER: for (int lr = 0; lr < MD_L_LAYERS; lr++) {
        const int l = MD_L_LAYERS - 1 - lr;
        const int dil = dilation_of_layer(l);

        burst_load_words(buffer, off_tap(l),  &tap_idx[0][0],  MD_D * MD_L_SEQ);
        burst_load_words(buffer, off_proj(l), &proj_idx[0][0], MD_L_SEQ * MD_D);

        // g_mp / g_mh are LOCAL accumulator arrays -> per-layer base is the
        // local l*(...) offset, not off_mp()/off_mh() (those are DDR offsets).
        const int pbase = l * (MD_D * MD_D);
        const int hbase = l * (MD_D * MD_K);

        // ---- residual add backward: split grad ----
        // Rout = R + PJ  ->  grad_PJ = gR ; grad_R starts as gR (residual path)
RB_SPLIT: for (int t = 0; t < MD_L_SEQ; t++) {
            for (int j = 0; j < MD_D; j++) {
#pragma HLS PIPELINE II=1
                grad_PJ[t][j] = gR[t][j];
                grad_R [t][j] = gR[t][j];   // residual skip path
            }
        }

        // ---- tropical projection backward (sparse scatter to i*) ----
        // PJ[t][j] = max_i(P[j][i] + MC[t][i]); only i* receives grad.
GMC_INIT: for (int t = 0; t < MD_L_SEQ; t++) {
            for (int i = 0; i < MD_D; i++) {
#pragma HLS PIPELINE II=1
                grad_MC[t][i] = 0;
            }
        }
PJ_BWD_T: for (int t = 0; t < MD_L_SEQ; t++) {
PJ_BWD_J: for (int j = 0; j < MD_D; j++) {
#pragma HLS PIPELINE II=4
                acc_t g = grad_PJ[t][j];
                if (g == 0) continue;
                int istar = proj_idx[t][j];
                if (istar < 0 || istar >= MD_D) continue;
                g_mp[pbase + j * MD_D + istar] += (int32_t)g;
                grad_MC[t][istar]              += g;
            }
        }

        // ---- depthwise morph conv backward (sparse scatter to k*/t_in) ----
        // MC[t][c] = (max|min)_k(h[c][k] + BS[t-k*dil][c]); only k* receives.
GBS_INIT: for (int t = 0; t < MD_L_SEQ; t++) {
            for (int c = 0; c < MD_D; c++) {
#pragma HLS PIPELINE II=1
                grad_BS[t][c] = 0;
            }
        }
MC_BWD_C: for (int c = 0; c < MD_D; c++) {
MC_BWD_T: for (int t = 0; t < MD_L_SEQ; t++) {
#pragma HLS PIPELINE II=4
                acc_t g = grad_MC[t][c];
                if (g == 0) continue;
                int kstar = tap_idx[c][t];
                if (kstar < 0 || kstar >= MD_K) continue;
                int t_in = t - kstar * dil;
                if (t_in < 0) continue;
                g_mh[hbase + c * MD_K + kstar] += (int32_t)g;
                grad_BS[t_in][c]               += g;
            }
        }

        // ---- batch_shift backward, per time step over D ----
        // BS[t][c] = R[t][c] - mean_c R[t][:]
        //   grad_R[t][c] += grad_BS[t][c] - (sum_c grad_BS[t][:]) >> LOG2_D
BS_BWD_T: for (int t = 0; t < MD_L_SEQ; t++) {
            acc_t sum = 0;
BS_BWD_SUM: for (int c = 0; c < MD_D; c++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
                sum += grad_BS[t][c];
            }
            acc_t mean = sum >> MD_LOG2_D;
BS_BWD_ADD: for (int c = 0; c < MD_D; c++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
                grad_R[t][c] += grad_BS[t][c] - mean;
            }
        }

        // grad_R becomes the next (earlier) layer's grad_R_out.
NEXT_GR: for (int t = 0; t < MD_L_SEQ; t++) {
            for (int j = 0; j < MD_D; j++) {
#pragma HLS PIPELINE II=1
                gR[t][j] = grad_R[t][j];
            }
        }
    }

    burst_store_words(g_mh, buffer, OFF_G_MH, WORDS_G_MH);
    burst_store_words(g_mp, buffer, OFF_G_MP, WORDS_G_MP);
    return 0;
}

// =============================================================================
// Mode 2: do_signsgd — SignSGD on MH and MP weight regions (verbatim
// region-wide helper from forge_train_step.cpp), zeroing grads.
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
    signsgd_region_wide(buffer, OFF_MH, OFF_G_MH, WORDS_MH, lr, threshold);
    signsgd_region_wide(buffer, OFF_MP, OFF_G_MP, WORDS_MP, lr, threshold);
    return 0;
}

// =============================================================================
// Top-level dispatcher — same ABI / pragmas as forge_train_step.cpp.
// =============================================================================
extern "C" int forge_train_step_morph(
    int32_t  buffer[MD_BUFFER_WORDS],
    int      mode,
    int      arg1,
    int      arg2,
    int      arg3
) {
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=MD_BUFFER_WORDS \
        num_write_outstanding=16 num_read_outstanding=16 \
        max_write_burst_length=16 max_read_burst_length=16 \
        max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite  port=mode    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg1    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg2    bundle=control
#pragma HLS INTERFACE s_axilite  port=arg3    bundle=control
#pragma HLS INTERFACE s_axilite  port=buffer  bundle=control
#pragma HLS INTERFACE s_axilite  port=return  bundle=control

    (void)arg3;
    if (mode == MD_MODE_FWD)     return do_fwd(buffer);
    if (mode == MD_MODE_BWD)     return do_bwd(buffer);
    if (mode == MD_MODE_SIGNSGD) return do_signsgd(buffer, arg1, arg2);
    return 99;
}
