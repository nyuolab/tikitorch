// forge_train_step_morph_ecg.cpp — STREAMING MorphSSM-core training kernel
// scaled to the PTB-XL ECG config (D=128, L_LAYERS=8, K=7, L_SEQ=5000,
// DIL_CAP=128). Dispatched by `mode`, same ABI / pragmas as
// forge_train_step.cpp.
//
// Identical morph-core MATH to forge_train_step_morph.cpp, but the big
// per-(layer,t) tensors live in DDR and are streamed so the on-chip working
// set stays O(L_LAYERS * D * WPOW) instead of O(L_SEQ * D).
//
//   mode 0 (FWD):     stream X[t] over t=0..L_SEQ-1 through all layers using a
//                     per-layer on-chip sliding window of batch-shifted
//                     inputs; save TAP_IDX[l][t][c] / PROJ_IDX[l][t][j] to DDR
//                     per timestep; accumulate the mean-pool; write POOLED[D].
//   mode 1 (BWD):     stream in REVERSE t=L_SEQ-1..0, re-loading saved indices
//                     from DDR, sparse-scattering into g_MH/g_MP; the conv
//                     backward routes a grad at output (c,t) to input
//                     (c,t-k*dil) — a FUTURE timestep in reverse — via a
//                     per-layer on-chip grad-window (symmetric to fwd).
//   mode 2 (SIGNSGD): signsgd_region_wide on MH and MP regions; zero grads.
//
// EVERYTHING on the core data path is compare / add / shift only — ZERO
// multiplications. `*` appears only in address/index arithmetic (e.g. c*K,
// l*D*D, the ring-buffer base) and the time-index `t - k*dil` / `t - kstar*dil`.
//
// Returns 0 on success, mode-specific error code otherwise.

#include "forge_buffer_morph_ecg.hpp"

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_uint<512> wide_t;
#endif

// Tropical sentinels (int32 extremes), reused for max/min init.
static const acc_t MDE_NEG_INF = acc_t(-2147483647LL);
static const acc_t MDE_POS_INF = acc_t( 2147483647LL);

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

// batch_shift: subtract the per-token mean over the D feature axis (mean =
// sum >> LOG2_D; pure shift). Operates on a single timestep row.
static void batch_shift_row(const data_t x[MDE_D], data_t y[MDE_D]) {
#pragma HLS INLINE
    acc_t sum = 0;
BS_SUM: for (int d = 0; d < MDE_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        sum += (acc_t)x[d];
    }
    acc_t mean = sum >> MDE_LOG2_D;
BS_SUB: for (int d = 0; d < MDE_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        y[d] = sat16((acc_t)x[d] - mean);
    }
}

// =============================================================================
// Mode 0: do_fwd — streaming morph core forward + mean-pool.
//
// Streams over t (outer) and layers (inner, bottom-up). Each layer keeps an
// on-chip sliding window WIN[l][c][slot] of its last WPOW batch-shifted input
// values, so the dilated conv at time t can read BS[t-k*dil][c]. Within a
// timestep, layer l's residual output feeds layer l+1 at the SAME t (kept in
// the small row buffer cur[]).
// =============================================================================
static int do_fwd(int32_t* buffer) {
#pragma HLS INLINE off

    // Weights for all layers (small; fit on-chip). Loaded once.
    static int32_t mh_buf[WORDS_MHE];   // morph conv kernels [l][c][k]
    static int32_t mp_buf[WORDS_MPE];   // tropical proj      [l][j][i]
    burst_load_words(buffer, OFF_MHE, mh_buf, WORDS_MHE);
    burst_load_words(buffer, OFF_MPE, mp_buf, WORDS_MPE);

    // Per-layer sliding window of batch-shifted inputs: WIN[l][c][t & WMASK].
    // This is the only O(L_LAYERS*D*window) on-chip state.
    static data_t WIN[MDE_L_LAYERS][MDE_D][MDE_WPOW];
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#pragma HLS BIND_STORAGE variable=WIN type=ram_t2p impl=uram
#pragma HLS ARRAY_PARTITION variable=WIN cyclic factor=8 dim=2
#endif

    // Running mean-pool accumulator over the FINAL layer's residual output.
    static acc_t pool_acc[MDE_D];
PINIT: for (int d = 0; d < MDE_D; d++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
        pool_acc[d] = 0;
    }

    // Per-(layer,t) saved index rows, staged then burst-stored to DDR.
    static int32_t tap_row [MDE_D];   // k* for layer l at time t, indexed by c
    static int32_t proj_row[MDE_D];   // i* for layer l at time t, indexed by j

    // Small per-timestep row buffers: input to current layer, conv out,
    // residual out (= input to next layer).
    static data_t cur[MDE_D];    // residual-stream value entering current layer
    static data_t bs [MDE_D];    // batch-shifted current input
    static data_t mc [MDE_D];    // morph-conv output
    static data_t pj [MDE_D];    // tropical-proj output

FWD_T: for (int t = 0; t < MDE_L_SEQ; t++) {
        // Load X[t] from DDR into cur[] (layer-0 input).
        int32_t xrow[MDE_D];
        burst_load_words(buffer, off_x_row(t), xrow, MDE_D);
LOADX: for (int d = 0; d < MDE_D; d++) {
#pragma HLS PIPELINE II=1
            cur[d] = (data_t)(int16_t)(xrow[d] & 0xFFFF);
        }

        int dil = 1;
FWD_L: for (int l = 0; l < MDE_L_LAYERS; l++) {
            const bool is_dil = ((l & 1) == 0);   // even->dilation(max), odd->erosion(min)
            const int  hbase  = l * (MDE_D * MDE_K);   // local base into mh_buf
            const int  pbase  = l * (MDE_D * MDE_D);   // local base into mp_buf
            const int  slot   = t & MDE_WMASK;
            const int  log2dil = dilation_log2_of_layer_e(l);   // dil = 2^log2dil

            // ---- batch_shift over D for the current input row ----
            batch_shift_row(cur, bs);

            // ---- push batch-shifted row into this layer's sliding window ----
PUSHW: for (int c = 0; c < MDE_D; c++) {
#pragma HLS PIPELINE II=1
                WIN[l][c][slot] = bs[c];
            }

            // ---- depthwise morph conv (causal, dilation dil) ----
            // MC[c] = (max|min)_k ( h[l][c][k] + WIN[l][c][t-k*dil] ), t-k*dil>=0
MC_C: for (int c = 0; c < MDE_D; c++) {
#pragma HLS PIPELINE II=1
                acc_t best  = is_dil ? MDE_NEG_INF : MDE_POS_INF;
                int   kstar = 0;
MC_K: for (int k = 0; k < MDE_K; k++) {
#pragma HLS UNROLL
                    int t_in = t - (k << log2dil);   // shift (dil=2^l); no multiply -> no DSP
                    if (t_in >= 0) {
                        data_t hck = (data_t)(int16_t)(mh_buf[hbase + c * MDE_K + k] & 0xFFFF);
                        acc_t s = (acc_t)hck + (acc_t)WIN[l][c][t_in & MDE_WMASK];
                        if (is_dil) { if (s > best) { best = s; kstar = k; } }
                        else        { if (s < best) { best = s; kstar = k; } }
                    }
                }
                mc[c]      = sat16(best);
                tap_row[c] = kstar;
            }

            // ---- tropical projection (max-plus) ----
            // PJ[j] = max_i ( P[l][j][i] + MC[i] ), argmax index i*
PJ_J: for (int j = 0; j < MDE_D; j++) {
#pragma HLS PIPELINE II=1
                acc_t best  = MDE_NEG_INF;
                int   istar = 0;
PJ_I: for (int i = 0; i < MDE_D; i++) {
#pragma HLS UNROLL factor=8
                    data_t pji = (data_t)(int16_t)(mp_buf[pbase + j * MDE_D + i] & 0xFFFF);
                    acc_t s = (acc_t)pji + (acc_t)mc[i];
                    if (s > best) { best = s; istar = i; }
                }
                pj[j]       = sat16(best);
                proj_row[j] = istar;
            }

            // ---- residual add -> input to next layer (overwrite cur) ----
RES: for (int j = 0; j < MDE_D; j++) {
#pragma HLS PIPELINE II=1
                cur[j] = sat16((acc_t)cur[j] + (acc_t)pj[j]);
            }

            // ---- persist saved indices for this (layer,t) row to DDR ----
            burst_store_words(tap_row,  buffer, off_tap_row(l, t),  MDE_D);
            burst_store_words(proj_row, buffer, off_proj_row(l, t), MDE_D);

            // dilation for next layer (shift-doubled, capped). No multiply.
            int nd = dil << 1;
            dil = (nd >= MDE_DIL_CAP) ? MDE_DIL_CAP : nd;
        }

        // ---- accumulate mean-pool over the FINAL layer's residual out ----
POOL_ACC: for (int j = 0; j < MDE_D; j++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
            pool_acc[j] += (acc_t)cur[j];
        }
    }

    // POOLED[j] = (sum_t Rout[t][j]) >> LOG2_L
    static int32_t pooled[MDE_D];
POOL_OUT: for (int j = 0; j < MDE_D; j++) {
#pragma HLS PIPELINE II=1
        pooled[j] = (int32_t)(int16_t)sat16(pool_acc[j] >> MDE_LOG2_L);
    }
    burst_store_words(pooled, buffer, OFF_POOLED_E, MDE_D);

    return 0;
}

// =============================================================================
// Mode 1: do_bwd — streaming backprop in reverse t.
//
// gR (grad wrt the current layer's residual OUTPUT) flows TOP-DOWN within a
// single timestep (layer l produces the gR consumed by layer l-1 at the same
// t). The only CROSS-timestep state is GBW[l][c][slot]: the conv backward
// scatters a grad at output (c,t) to input (c, t-k*dil) — a future (smaller)
// timestep in reverse — so it is accumulated in a per-layer ring window and
// read out (then cleared) when the reverse stream reaches that timestep.
//
// All contributions to GBW[l][.][t] come from output timesteps in
// [t, t+(K-1)*dil], every one of which is processed BEFORE t in reverse, so
// GBW[l][.][t] is complete by the time t is read. The window is zeroed at
// init and re-cleared on read, maintaining the cleared-before-scatter
// invariant under ring reuse.
// =============================================================================
static int do_bwd(int32_t* buffer) {
#pragma HLS INLINE off

    // Gradient accumulators (read-modify-write across all layers/timesteps).
    static int32_t g_mh[WORDS_G_MHE];
    static int32_t g_mp[WORDS_G_MPE];
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#pragma HLS BIND_STORAGE variable=g_mp type=ram_t2p impl=uram
#endif
    burst_load_words(buffer, OFF_G_MHE, g_mh, WORDS_G_MHE);
    burst_load_words(buffer, OFF_G_MPE, g_mp, WORDS_G_MPE);

    // Mean-pool backward: grad wrt EACH final-layer Rout[t][j] is the same
    // constant grad_pooled[j] >> LOG2_L for all t.
    static int32_t grad_pooled[MDE_D];
    burst_load_words(buffer, OFF_GRAD_POOLED_E, grad_pooled, MDE_D);
    static acc_t gR_top[MDE_D];
GRT: for (int j = 0; j < MDE_D; j++) {
#pragma HLS PIPELINE II=1
        gR_top[j] = (acc_t)grad_pooled[j] >> MDE_LOG2_L;
    }

    // Per-layer cross-timestep grad window (the conv's gradient routing).
    static acc_t GBW[MDE_L_LAYERS][MDE_D][MDE_WPOW];
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#pragma HLS BIND_STORAGE variable=GBW type=ram_t2p impl=uram
#pragma HLS ARRAY_PARTITION variable=GBW cyclic factor=8 dim=2
#endif
GBW_INIT_L: for (int l = 0; l < MDE_L_LAYERS; l++) {
GBW_INIT_C: for (int c = 0; c < MDE_D; c++) {
GBW_INIT_S: for (int s = 0; s < MDE_WPOW; s++) {
#pragma HLS PIPELINE II=1
                GBW[l][c][s] = 0;
            }
        }
    }

    // Per-(layer,t) saved index rows, re-loaded from DDR.
    static int32_t tap_row [MDE_D];
    static int32_t proj_row[MDE_D];

    // Per-t staging row for grad_X[t] (the layer-0 input gradient), burst-stored
    // to DDR at OFF_GRAD_X_E once the lowest layer closes for this timestep.
    static int32_t gx_row[MDE_D];

    // Per-timestep working rows.
    static acc_t gR      [MDE_D];   // grad wrt current layer's Rout at t
    static acc_t grad_PJ [MDE_D];
    static acc_t grad_R  [MDE_D];   // residual-skip + batch_shift-bwd accumulation
    static acc_t grad_MC [MDE_D];

    // Precompute per-layer dilations (so the reverse layer loop can index
    // without recomputing; pure shift table, no multiply on the data path).
    int dil_tab[MDE_L_LAYERS];
DTAB: for (int l = 0; l < MDE_L_LAYERS; l++) {
        dil_tab[l] = dilation_of_layer_e(l);
    }

BWD_T: for (int t = MDE_L_SEQ - 1; t >= 0; t--) {
        const int slot = t & MDE_WMASK;

        // Seed top layer's output grad from the (constant) mean-pool grad.
SEED: for (int j = 0; j < MDE_D; j++) {
#pragma HLS PIPELINE II=1
            gR[j] = gR_top[j];
        }

BWD_L: for (int lr = 0; lr < MDE_L_LAYERS; lr++) {
            const int  l      = MDE_L_LAYERS - 1 - lr;
            const int  dil    = dil_tab[l];
            const int  log2dil = dilation_log2_of_layer_e(l);   // dil = 2^log2dil
            const int  pbase  = l * (MDE_D * MDE_D);
            const int  hbase  = l * (MDE_D * MDE_K);

            // Re-load this (layer,t) row's saved indices from DDR.
            burst_load_words(buffer, off_tap_row(l, t),  tap_row,  MDE_D);
            burst_load_words(buffer, off_proj_row(l, t), proj_row, MDE_D);

            // ---- residual add backward: split grad ----
            // Rout = R + PJ -> grad_PJ = gR ; grad_R starts as gR (skip path)
RB_SPLIT: for (int j = 0; j < MDE_D; j++) {
#pragma HLS PIPELINE II=1
                grad_PJ[j] = gR[j];
                grad_R [j] = gR[j];
                grad_MC[j] = 0;
            }

            // ---- tropical projection backward (sparse scatter to i*) ----
PJ_BWD_J: for (int j = 0; j < MDE_D; j++) {
#pragma HLS PIPELINE II=4
                acc_t g = grad_PJ[j];
                if (g == 0) continue;
                int istar = proj_row[j];
                if (istar < 0 || istar >= MDE_D) continue;
                g_mp[pbase + j * MDE_D + istar] += (int32_t)g;
                grad_MC[istar]                  += g;
            }

            // ---- depthwise morph conv backward ----
            // MC[c] = (max|min)_k(h[c][k] + BS[t-k*dil][c]); only k* receives.
            // g_MH gets the local update; the grad to BS[t-k*dil][c] is a
            // FUTURE (smaller) timestep -> scatter into the per-layer window.
MC_BWD_C: for (int c = 0; c < MDE_D; c++) {
#pragma HLS PIPELINE II=4
                acc_t g = grad_MC[c];
                if (g == 0) continue;
                int kstar = tap_row[c];
                if (kstar < 0 || kstar >= MDE_K) continue;
                int t_in = t - (kstar << log2dil);   // shift (dil=2^l); no multiply -> no DSP
                if (t_in < 0) continue;
                g_mh[hbase + c * MDE_K + kstar] += (int32_t)g;
                GBW[l][c][t_in & MDE_WMASK]     += g;
            }

            // ---- read accumulated grad_BS for THIS timestep (now complete) ----
            // batch_shift backward: grad_R[c] += grad_BS[c] - (sum>>LOG2_D).
            // grad_BS[c] = GBW[l][c][slot]; clear slot after reading.
            acc_t sumg = 0;
BS_BWD_SUM: for (int c = 0; c < MDE_D; c++) {
#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
                sumg += GBW[l][c][slot];
            }
            acc_t meang = sumg >> MDE_LOG2_D;
BS_BWD_ADD: for (int c = 0; c < MDE_D; c++) {
#pragma HLS PIPELINE II=1
                acc_t gbs = GBW[l][c][slot];
                grad_R[c] += gbs - meang;
                GBW[l][c][slot] = 0;   // clear for ring reuse (logical t-WPOW)
            }

            // grad_R becomes the next (lower) layer's output grad at this t.
NEXT_GR: for (int j = 0; j < MDE_D; j++) {
#pragma HLS PIPELINE II=1
                gR[j] = grad_R[j];
            }
        }

        // After the lowest layer (l=0) closes, gR == grad wrt this layer's
        // INPUT = grad_X[t] (the embedded input X[t]). Persist it to DDR so the
        // host can train the float input embedding exactly. Pure copy + index
        // arithmetic (t*D is an address, no data-path multiply). The region is
        // overwritten in full (every t writes its own row), so no pre-zeroing
        // is needed across backward calls.
GX_COPY: for (int d = 0; d < MDE_D; d++) {
#pragma HLS PIPELINE II=1
            gx_row[d] = (int32_t)gR[d];
        }
        burst_store_words(gx_row, buffer, off_grad_x_row(t), MDE_D);
    }

    burst_store_words(g_mh, buffer, OFF_G_MHE, WORDS_G_MHE);
    burst_store_words(g_mp, buffer, OFF_G_MPE, WORDS_G_MPE);
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
    signsgd_region_wide(buffer, OFF_MHE, OFF_G_MHE, WORDS_MHE, lr, threshold);
    signsgd_region_wide(buffer, OFF_MPE, OFF_G_MPE, WORDS_MPE, lr, threshold);
    return 0;
}

// =============================================================================
// Top-level dispatcher — same ABI / pragmas as forge_train_step.cpp.
// =============================================================================
extern "C" int forge_train_step_morph_ecg(
    int32_t  buffer[FT_BUFFER_WORDS_E],
    int      mode,
    int      arg1,
    int      arg2,
    int      arg3
) {
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=FT_BUFFER_WORDS_E \
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
    if (mode == MDE_MODE_FWD)     return do_fwd(buffer);
    if (mode == MDE_MODE_BWD)     return do_bwd(buffer);
    if (mode == MDE_MODE_SIGNSGD) return do_signsgd(buffer, arg1, arg2);
    return 99;
}
