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

// --- throughput knobs (compile-time; default = the validated baseline) ---
// TT_WIDE_LANES: lanes per cycle in the max-plus projection (PJ) and LM head
// reductions. 8 = baseline. 16/32/64 widen the add+compare tree (more LUT, needs
// mp_buf/wh_buf banked). MUST divide ML_D (128 by default; 512 under -DML_D=512
// — 8 divides both). Bit-exact for any value: the
// balanced strict-'>' tree always keeps the lowest index among equal maxima.
#ifndef TT_WIDE_LANES
#define TT_WIDE_LANES 8
#endif

// --- #3 on-chip weight-grad accumulators (TT_ONCHIP_GRAD) ---
// do_bwd's grad scatters are buffer (m_axi) read-modify-writes with data-dependent
// indices -> HLS serializes them at II=4. Accumulating into on-chip static arrays
// (II=1 RMW) removes that. do_train_run zeros them each step and flushes them to
// the buffer once (a wide burst) before do_signsgd, so signsgd stays unchanged.
// Bit-exact: same per-position adds, same order (integer adds associative); the
// CSIM fused golden validates. Default OFF -> shipped buffer-backed path.
#ifdef TT_ONCHIP_GRAD
static int32_t g_embed_oc[WORDS_G_EMBED];
static int32_t g_mh_oc[WORDS_G_MH];
static int32_t g_mp_oc[WORDS_G_MP];
static int32_t g_mb_oc[WORDS_G_MB];
static int32_t g_whead_oc[WORDS_G_W_HEAD];
#define GEMBED(i) g_embed_oc[i]
#define GMH(i)    g_mh_oc[i]
#define GMP(i)    g_mp_oc[i]
#define GMB(i)    g_mb_oc[i]
#define GWHEAD(i) g_whead_oc[i]
#else
#define GEMBED(i) buffer[OFF_G_EMBED + (i)]
#define GMH(i)    buffer[OFF_G_MH + (i)]
#define GMP(i)    buffer[OFF_G_MP + (i)]
#define GMB(i)    buffer[OFF_G_MB + (i)]
#define GWHEAD(i) buffer[OFF_G_W_HEAD + (i)]
#endif

// --- resident weights + on-chip signsgd (TT_RESIDENT_W; requires TT_ONCHIP_GRAD) ---
// signsgd's m_axi weight RMW is the recurring critical path (-0.252/-0.592 ns) AND
// ~13% of the step. Make ALL weights resident on-chip: do_train_run loads them once
// before the step loop and writes them back once after; signsgd updates them on-chip
// (reading g_*_oc); do_fwd_loss reads them directly (no per-step reload). Removes the
// m_axi signsgd path (timing) and the per-step weight reload. Bit-exact: same values,
// same update; CSIM fused golden validates. Default OFF.
#ifdef TT_RESIDENT_W
#ifndef TT_ONCHIP_GRAD
#error "TT_RESIDENT_W requires TT_ONCHIP_GRAD (signsgd reads g_*_oc on-chip)"
#endif
static int32_t embed_oc[WORDS_EMBED];
static int32_t mh_buf[WORDS_MH];
static int32_t mb_buf[WORDS_MB];
static int32_t mp_buf[WORDS_MP];
static int32_t wh_buf[WORDS_W_HEAD];
static int32_t ce_lut[WORDS_CE_LUT];
#endif

// --- BATCH parallelism (TT_BATCH; requires TT_ONCHIP_GRAD + TT_RESIDENT_W) ---
// do_train_run loops the fwd/bwd over b in 0..ML_BATCH-1. The B sequences share
// the resident weights (read-only in fwd) and SUM their gradients into the
// shared g_*_oc accumulators; ONE signsgd per step. do_fwd_loss/do_bwd take a
// `b` argument selecting that sequence's per-seq buffer scratch (BS stream /
// saved argmax routes / grad_logit / pred) via the off_*_b(b,...) helpers; the
// arithmetic is otherwise IDENTICAL to single-sequence (a constant b-stride on
// the per-seq addresses), so each sequence's grad contribution is bit-exact to
// running that sequence alone => summed-batched == sum-of-per-sequence. The
// per-seq region OFFSETS below collapse to the single-seq ones at b=0 (non-batch
// builds pass b=0 and the helpers reduce to off_bs/off_tap/... unchanged).
#ifdef TT_BATCH
#if !defined(TT_ONCHIP_GRAD) || !defined(TT_RESIDENT_W)
#error "TT_BATCH requires TT_ONCHIP_GRAD + TT_RESIDENT_W (shared on-chip grads + resident shared weights)"
#endif
#define ML_OFF_BS(l, m)    off_bs_b(b, l, m)
#define ML_OFF_TAP(l, m)   off_tap_b(b, l, m)
#define ML_OFF_PROJ(l, m)  off_proj_b(b, l, m)
#define ML_OFF_HEAD(m)     off_head_b(b, m)
#define ML_OFF_GLOGIT(m)   off_grad_logit_b(b, m)
#define ML_OFF_PRED        off_pred_b(b)
#else
#define ML_OFF_BS(l, m)    off_bs(l, m)
#define ML_OFF_TAP(l, m)   off_tap(l, m)
#define ML_OFF_PROJ(l, m)  off_proj(l, m)
#define ML_OFF_HEAD(m)     off_head(m)
#define ML_OFF_GLOGIT(m)   off_grad_logit(m)
#define ML_OFF_PRED        OFF_PRED
#endif

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
BS_SUM: for (int d = 0; d < ML_D; d += 8) {
#pragma HLS PIPELINE II=1
        acc_t t0 = (acc_t)x[d]     + (acc_t)x[d + 1];
        acc_t t1 = (acc_t)x[d + 2] + (acc_t)x[d + 3];
        acc_t t2 = (acc_t)x[d + 4] + (acc_t)x[d + 5];
        acc_t t3 = (acc_t)x[d + 6] + (acc_t)x[d + 7];
        sum += (t0 + t1) + (t2 + t3);   // carried dep: one add
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
static int do_fwd_loss(int32_t* buffer, int b, int m, int token, int target, int reload) {
#pragma HLS INLINE off
    (void)b;   // non-batch: b==0; per-seq offsets via ML_OFF_* (b-aware under TT_BATCH)
    if (m < 0 || m >= ML_L_SEQ)          return 1;
    if (token < 0 || token >= ML_V)      return 2;
    if (target < 0 || target >= ML_V)    return 3;

    // Weights live in static (BRAM/URAM-resident) arrays; reload from the buffer
    // only when `reload` is set — once per forward sweep in the fused trainer
    // (mode 3), every call in the per-position mode (mode 0). do_signsgd writes
    // updated weights back to the buffer, so each sweep's first call refreshes
    // them. ce_lut is the host-staged exp2 table (OFF_CE_LUT), read here instead
    // of the static-const ROM (whose init read all-zero on the F2 AFI).
#ifndef TT_RESIDENT_W
    static int32_t mh_buf[WORDS_MH];
    static int32_t mb_buf[WORDS_MB];
    static int32_t mp_buf[WORDS_MP];      // URAM-class
    static int32_t wh_buf[WORDS_W_HEAD];
    static int32_t ce_lut[WORDS_CE_LUT];
#endif
#if TT_WIDE_LANES > 8
    // Bank the projection/head weight arrays so the widened reductions get
    // TT_WIDE_LANES contiguous reads/cycle (else the wide tree bottlenecks on
    // a single memory port). Default (8) leaves the baseline mapping untouched.
#pragma HLS ARRAY_PARTITION variable=mp_buf cyclic factor=TT_WIDE_LANES dim=1
#pragma HLS ARRAY_PARTITION variable=wh_buf cyclic factor=TT_WIDE_LANES dim=1
#endif
#ifndef TT_RESIDENT_W
    if (reload) {
        burst_load_words(buffer, OFF_MH, mh_buf, WORDS_MH);
        burst_load_words(buffer, OFF_MB, mb_buf, WORDS_MB);
        burst_load_words(buffer, OFF_MP, mp_buf, WORDS_MP);
        burst_load_words(buffer, OFF_W_HEAD, wh_buf, WORDS_W_HEAD);
        burst_load_words(buffer, OFF_CE_LUT, ce_lut, WORDS_CE_LUT);
    }
#else
    (void)reload;   // resident: weights+LUT loaded once by do_train_run
#endif

    // Embed gather: R = EMBED[token].
    static data_t R[ML_D];
    static int32_t row32[ML_D];
#ifdef TT_RESIDENT_W
    for (int d = 0; d < ML_D; d++) row32[d] = embed_oc[token * ML_D + d];
#else
    burst_load_words(buffer, off_embed(token), row32, ML_D);
#endif
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

LAYER_LOOP: for (int l = 0; l < ML_N_LAYERS; l++) {
        // dil = 1 << dsh (power of two by construction): tap offsets are
        // SHIFTS. k*dil as a multiply binds DSP48s (measured: 3) - forbidden.
        const int dsh    = (l < 6) ? l : 6;   // log2(min(2^l, ML_DIL_CAP))
        const int dil    = 1 << dsh; (void)dil;
        const bool is_dil = ((l & 1) == 0);   // even -> dilation(max), odd -> erosion(min)
        const int hbase  = l * (ML_D * ML_K);

        // ---- batch_shift, then persist to the rolling BS stream ----
        batch_shift_row(R, BS_row);
BS32: for (int d = 0; d < ML_D; d++) {
#pragma HLS PIPELINE II=1
            bs32[d] = (int32_t)(int16_t)BS_row[d];
        }
        burst_store_words(bs32, buffer, ML_OFF_BS(l, m), ML_D);

        // ---- causal depthwise morph conv over the BS stream ----
        // Tap k reads BS_l[m - k*dil]; out-of-range taps (t_in < 0) EXCLUDED
        // (matches forge_train_step_morph.cpp; deviates from the Python
        // zero-pad — see the buffer header note). Tap rows are burst-loaded
        // from the stream; k = 0 is the row just computed.
TAPS: for (int k = 1; k < ML_K; k++) {
            int t_in = m - (k << dsh);
            if (t_in >= 0) {
                burst_load_words(buffer, ML_OFF_BS(l, t_in), taps32[k], ML_D);
            }
        }
MC_C: for (int c = 0; c < ML_D; c++) {
#pragma HLS PIPELINE II=1
            acc_t best = is_dil ? ML_NEG_INF : ML_POS_INF;
            int   kstar = 0;
MC_K: for (int k = 0; k < ML_K; k++) {
#pragma HLS UNROLL
                int t_in = m - (k << dsh);
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
        const int pbase = l * (ML_D * ML_D);
PJ_J: for (int j = 0; j < ML_D; j++) {
            acc_t best = ML_NEG_INF;
            int   istar = 0;
PJ_I: for (int i = 0; i < ML_D; i += TT_WIDE_LANES) {
#pragma HLS PIPELINE II=1
                acc_t cand[TT_WIDE_LANES]; int cidx[TT_WIDE_LANES];
#pragma HLS ARRAY_PARTITION variable=cand complete
#pragma HLS ARRAY_PARTITION variable=cidx complete
PJ_LANE: for (int u = 0; u < TT_WIDE_LANES; u++) {
#pragma HLS UNROLL
                    data_t pji = (data_t)(int16_t)(mp_buf[pbase + j * ML_D + i + u] & 0xFFFF);
                    cand[u] = (acc_t)pji + (acc_t)MC[i + u];
                    cidx[u] = i + u;
                }
PJ_TREE: for (int st = 1; st < TT_WIDE_LANES; st <<= 1) {
#pragma HLS UNROLL
                    for (int u = 0; u < TT_WIDE_LANES; u += (st << 1)) {
#pragma HLS UNROLL
                        if (cand[u + st] > cand[u]) { cand[u] = cand[u + st]; cidx[u] = cidx[u + st]; }
                    }
                }
                if (cand[0] > best) { best = cand[0]; istar = cidx[0]; }
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

        burst_store_words(tap_row,  buffer, ML_OFF_TAP(l, m),  ML_D);
        burst_store_words(proj_row, buffer, ML_OFF_PROJ(l, m), ML_D);
    }

    // ---- LM head: logits[v] = max_d(W_HEAD[v][d] + R[d]) ----
    static data_t  logits[ML_V];
    static int32_t ad_head_row[ML_V];
HEAD_V: for (int v = 0; v < ML_V; v++) {
        acc_t mx = ML_NEG_INF;
        int   idx_d = 0;
HEAD_D: for (int d = 0; d < ML_D; d += TT_WIDE_LANES) {
#pragma HLS PIPELINE II=1
            acc_t cand[TT_WIDE_LANES]; int cidx[TT_WIDE_LANES];
#pragma HLS ARRAY_PARTITION variable=cand complete
#pragma HLS ARRAY_PARTITION variable=cidx complete
HD_LANE: for (int u = 0; u < TT_WIDE_LANES; u++) {
#pragma HLS UNROLL
                data_t W_vd = (data_t)(int16_t)(wh_buf[v * ML_D + d + u] & 0xFFFF);
                cand[u] = (acc_t)W_vd + (acc_t)R[d + u];
                cidx[u] = d + u;
            }
HD_TREE: for (int st = 1; st < TT_WIDE_LANES; st <<= 1) {
#pragma HLS UNROLL
                for (int u = 0; u < TT_WIDE_LANES; u += (st << 1)) {
#pragma HLS UNROLL
                    if (cand[u + st] > cand[u]) { cand[u] = cand[u + st]; cidx[u] = cidx[u + st]; }
                }
            }
            if (cand[0] > mx) { mx = cand[0]; idx_d = cidx[0]; }
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
            e_row[v] = ce_lut[idx];      // host-staged exp2 table (see OFF_CE_LUT)
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

    burst_store_words(ad_head_row,    buffer, ML_OFF_HEAD(m),    ML_V);
    burst_store_words(grad_logit_row, buffer, ML_OFF_GLOGIT(m),  ML_V);
    buffer[ML_OFF_PRED + m] = pred;

    return 0;
}

// =============================================================================
// Mode 1: do_bwd — head/layers/embed backward at position m, sparse scatter
// through saved indices. Host walks m in REVERSE so grad_BS stream slots are
// complete when read (see buffer header). Scalar read-modify-write into the
// gradient regions (the backward is sparse: <= V + 3*L*D touched words).
// =============================================================================
static int do_bwd(int32_t* buffer, int b, int m, int token) {
#pragma HLS INLINE off
    (void)b;   // non-batch: b==0; per-seq offsets via ML_OFF_* (b-aware under TT_BATCH)
    if (m < 0 || m >= ML_L_SEQ)          return 1;
    if (token < 0 || token >= ML_V)      return 2;

    static int32_t grad_logit_row[ML_V];
    static int32_t ad_head_row[ML_V];
    burst_load_words(buffer, ML_OFF_GLOGIT(m), grad_logit_row, ML_V);
    burst_load_words(buffer, ML_OFF_HEAD(m),   ad_head_row,    ML_V);

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
        GWHEAD(v * ML_D + dstar) += (int32_t)g;
        gR[dstar] += g;
    }

    static int32_t tap_row [ML_D];
    static int32_t proj_row[ML_D];
    static int32_t gbs32[ML_D];
    static acc_t   grad_MC[ML_D];

BWD_LAYER: for (int lr = 0; lr < ML_N_LAYERS; lr++) {
        const int l   = ML_N_LAYERS - 1 - lr;
        const int dsh = (l < 6) ? l : 6;      // log2 dilation (see fwd note)
        const int dil = 1 << dsh; (void)dil;
        const int pbase = l * (ML_D * ML_D);
        const int hbase = l * (ML_D * ML_K);

        burst_load_words(buffer, ML_OFF_TAP(l, m),  tap_row,  ML_D);
        burst_load_words(buffer, ML_OFF_PROJ(l, m), proj_row, ML_D);

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
            GMB(l * ML_D + j) += (int32_t)g;   // bias: post-max add
            int istar = proj_row[j];
            if (istar < 0 || istar >= ML_D) continue;
            GMP(pbase + j * ML_D + istar) += (int32_t)g;
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
            int t_in = m - (kstar << dsh);
            if (t_in < 0) continue;
            GMH(hbase + c * ML_K + kstar) += (int32_t)g;
            buffer[ML_OFF_BS(l, t_in) + c]              += (int32_t)g;
        }

        // ---- batch_shift backward over the (complete) own grad_BS slot ----
        // grad_R[c] += grad_BS[c] - (sum_c grad_BS) >> LOG2_D, on top of the
        // residual skip already in gR.
        burst_load_words(buffer, ML_OFF_BS(l, m), gbs32, ML_D);
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
            GEMBED(token * ML_D + d) += (int32_t)gR[d];
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
    ap_int<32> th32 = (ap_int<32>)threshold;   // full int32: CE Q16 grads exceed 32767

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
            if (abs_g > th32) {
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

#ifdef TT_RESIDENT_W
// On-chip signsgd: weight and grad both resident -> no m_axi (removes the signsgd
// critical path). Scalar int (matches the host #else path of signsgd_region_wide
// exactly -> bit-exact). Zeros the grad as it consumes it (so no separate per-step
// grad-zero is needed; do_train_run zeros once before the first step).
static void signsgd_region_oc(int32_t* w_oc, int32_t* g_oc, int n, int lr, int threshold) {
#pragma HLS INLINE off
SGOC: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        int32_t g = g_oc[i];
        int32_t abs_g = (g < 0) ? -g : g;
        int32_t step = 0;
        if (abs_g > threshold) { step = (g > 0) ? lr : -lr; }
        if (step != 0) {
            int16_t w = (int16_t)(w_oc[i] & 0xFFFF);
            int32_t wn = (int32_t)w - step;
            if (wn >  32767) wn =  32767;
            if (wn < -32768) wn = -32768;
            w_oc[i] = (int32_t)(int16_t)wn;
        }
        g_oc[i] = 0;
    }
}
#endif

static int do_signsgd(int32_t* buffer, int lr, int threshold) {
#pragma HLS INLINE off
#ifdef TT_RESIDENT_W
    signsgd_region_oc(embed_oc, g_embed_oc,  WORDS_EMBED,  lr, threshold);
    signsgd_region_oc(mh_buf,   g_mh_oc,     WORDS_MH,     lr, threshold);
    signsgd_region_oc(mp_buf,   g_mp_oc,     WORDS_MP,     lr, threshold);
    signsgd_region_oc(mb_buf,   g_mb_oc,     WORDS_MB,     lr, threshold);
    signsgd_region_oc(wh_buf,   g_whead_oc,  WORDS_W_HEAD, lr, threshold);
    (void)buffer;
#else
    signsgd_region_wide(buffer, OFF_EMBED,  OFF_G_EMBED,  WORDS_EMBED,  lr, threshold);
    signsgd_region_wide(buffer, OFF_MH,     OFF_G_MH,     WORDS_MH,     lr, threshold);
    signsgd_region_wide(buffer, OFF_MP,     OFF_G_MP,     WORDS_MP,     lr, threshold);
    signsgd_region_wide(buffer, OFF_MB,     OFF_G_MB,     WORDS_MB,     lr, threshold);
    signsgd_region_wide(buffer, OFF_W_HEAD, OFF_G_W_HEAD, WORDS_W_HEAD, lr, threshold);
#endif
    return 0;
}

// =============================================================================
// In-kernel region zero (wide stores). All grad/BS regions are 16-word aligned
// in both offset and length, so the wide path covers them exactly.
// =============================================================================
static void zero_region_wide(int32_t* buffer, int off_words, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t* wide_buf = reinterpret_cast<wide_t*>(buffer);
    int n_wide = n_words >> 4;
    int wide_off = off_words >> 4;
ZERO_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
        wide_buf[wide_off + w] = (wide_t)0;
    }
#else
    for (int i = 0; i < n_words; i++) buffer[off_words + i] = 0;
#endif
}

// =============================================================================
// Mode 3: do_train_run — the WHOLE training run on fabric. One host kick runs
// `n_steps` optimizer steps over the resident window; tokens/targets/weights/
// LUT all live in the buffer (no host round-trips per position). Weights are
// reloaded into the resident static arrays once per forward sweep (m==0).
// Arithmetic is IDENTICAL to host-looping modes 0/1/2, so it is bit-exact to
// the per-position golden — the host CSIM build verifies this.
// =============================================================================
static int do_train_run(int32_t* buffer, int n_steps, int lr, int threshold) {
#pragma HLS INLINE off
#if defined(TT_RESIDENT_W) && defined(TT_URAM_W)
    // GPU-parity replication enabler: move the read-mostly resident weights off
    // BRAM into URAM so per-kernel BRAM shrinks enough for 2 kernels per SLR
    // (BRAM-bound at ~354 tiles/kernel; URAM is ~0% used otherwise). These four
    // are NOT wide-partitioned (mp_buf/wh_buf stay BRAM for the TT_WIDE_LANES
    // reads) and NOT in the bwd RMW path (grads stay BRAM for II=1 scatter), so
    // URAM's fixed latency costs nothing here. Bit-exact: same int32 storage.
#pragma HLS bind_storage variable=embed_oc type=RAM_T2P impl=URAM
#pragma HLS bind_storage variable=mh_buf   type=RAM_T2P impl=URAM
#pragma HLS bind_storage variable=mb_buf   type=RAM_T2P impl=URAM
#pragma HLS bind_storage variable=ce_lut   type=RAM_T2P impl=URAM
    // The grads dominate BRAM (V*D g_embed/g_whead). Move them to URAM too; they
    // are scatter-RMW'd in the bwd, so verify the bwd II + tok/s on silicon.
#pragma HLS bind_storage variable=g_embed_oc type=RAM_T2P impl=URAM
#pragma HLS bind_storage variable=g_mh_oc    type=RAM_T2P impl=URAM
#pragma HLS bind_storage variable=g_mp_oc    type=RAM_T2P impl=URAM
#pragma HLS bind_storage variable=g_mb_oc    type=RAM_T2P impl=URAM
#pragma HLS bind_storage variable=g_whead_oc type=RAM_T2P impl=URAM
#endif
#ifdef TT_BATCH
    // BATCH: ML_BATCH independent sequences, each with its own tokens/targets.
    static int32_t toks_b[ML_BATCH_N][ML_L_SEQ];
    static int32_t tgts_b[ML_BATCH_N][ML_L_SEQ];
    for (int b = 0; b < ML_BATCH_N; b++) {
        burst_load_words(buffer, off_tokens_b(b),  toks_b[b], ML_L_SEQ);
        burst_load_words(buffer, off_targets_b(b), tgts_b[b], ML_L_SEQ);
    }
#else
    static int32_t toks[ML_L_SEQ];
    static int32_t tgts[ML_L_SEQ];
    burst_load_words(buffer, OFF_TOKENS,  toks, ML_L_SEQ);   // ML_L_SEQ=128 (16-aligned)
    burst_load_words(buffer, OFF_TARGETS, tgts, ML_L_SEQ);
#endif

#ifdef TT_RESIDENT_W
    // Load ALL weights + the exp2 LUT into resident on-chip arrays ONCE; zero the
    // on-chip grad accumulators once (signsgd zeros them each step thereafter).
    burst_load_words(buffer, OFF_EMBED,  embed_oc, WORDS_EMBED);
    burst_load_words(buffer, OFF_MH,     mh_buf,   WORDS_MH);
    burst_load_words(buffer, OFF_MB,     mb_buf,   WORDS_MB);
    burst_load_words(buffer, OFF_MP,     mp_buf,   WORDS_MP);
    burst_load_words(buffer, OFF_W_HEAD, wh_buf,   WORDS_W_HEAD);
    burst_load_words(buffer, OFF_CE_LUT, ce_lut,   WORDS_CE_LUT);
    for (int i = 0; i < WORDS_G_EMBED;  i++) g_embed_oc[i] = 0;
    for (int i = 0; i < WORDS_G_MH;     i++) g_mh_oc[i]    = 0;
    for (int i = 0; i < WORDS_G_MP;     i++) g_mp_oc[i]    = 0;
    for (int i = 0; i < WORDS_G_MB;     i++) g_mb_oc[i]    = 0;
    for (int i = 0; i < WORDS_G_W_HEAD; i++) g_whead_oc[i] = 0;
#endif

STEP_LOOP: for (int s = 0; s < n_steps; s++) {
        // (1) zero the rolling BS stream(s) before the forward sweep. Under
        // TT_BATCH WORDS_BS_STREAM already spans ALL B per-seq streams, so this
        // one wide zero clears every sequence's rolling cache.
        zero_region_wide(buffer, OFF_BS_STREAM, WORDS_BS_STREAM);
#ifdef TT_BATCH
        // (2) FORWARD over the B sequences (shared resident weights, per-seq scratch).
        // reload=0: TT_RESIDENT_W keeps weights+LUT resident (loaded once below).
        FWD_SWEEP_B: for (int b = 0; b < ML_BATCH_N; b++) {
            for (int m = 0; m < ML_L_SEQ; m++) {
                do_fwd_loss(buffer, b, m, (int)toks_b[b][m], (int)tgts_b[b][m], 0);
            }
        }
#else
        // (2) FORWARD m = 0..L-1 ; reload weights+LUT once (m==0), resident after.
        FWD_SWEEP: for (int m = 0; m < ML_L_SEQ; m++) {
            do_fwd_loss(buffer, 0, m, (int)toks[m], (int)tgts[m], (m == 0) ? 1 : 0);
        }
#endif
        // (3) zero the 5 grad regions AND re-zero BS (now the grad_BS accumulator).
#if defined(TT_RESIDENT_W)
        // grads zeroed once before the loop; signsgd zeros them on-chip each step.
#elif defined(TT_ONCHIP_GRAD)
        // grads live on-chip; zero the accumulators (signsgd zeros only the buffer copy).
GZ_E: for (int i = 0; i < WORDS_G_EMBED;  i++) { GEMBED(i) = 0; }
GZ_H: for (int i = 0; i < WORDS_G_MH;     i++) { GMH(i)    = 0; }
GZ_P: for (int i = 0; i < WORDS_G_MP;     i++) { GMP(i)    = 0; }
GZ_B: for (int i = 0; i < WORDS_G_MB;     i++) { GMB(i)    = 0; }
GZ_W: for (int i = 0; i < WORDS_G_W_HEAD; i++) { GWHEAD(i) = 0; }
#elif !defined(TT_DROP_GRAD_ZERO)
        // These grad-zeros are redundant: do_signsgd zeros all 5 grad regions at
        // the end of every step, and the init buffer zeros them for step 0. The
        // BS re-zero below is NOT redundant (it clears the fwd activations so BS
        // can serve as the grad_BS accumulator). TT_DROP_GRAD_ZERO drops the 5.
        zero_region_wide(buffer, OFF_G_EMBED,  WORDS_G_EMBED);
        zero_region_wide(buffer, OFF_G_MH,     WORDS_G_MH);
        zero_region_wide(buffer, OFF_G_MP,     WORDS_G_MP);
        zero_region_wide(buffer, OFF_G_MB,     WORDS_G_MB);
        zero_region_wide(buffer, OFF_G_W_HEAD, WORDS_G_W_HEAD);
#endif
        zero_region_wide(buffer, OFF_BS_STREAM, WORDS_BS_STREAM);
#ifdef TT_BATCH
        // (4) BACKWARD over B sequences, REVERSE position per sequence; each
        // sequence's weight grads SUM into the shared g_*_oc accumulators (the
        // GMH/GMP/GMB/GWHEAD/GEMBED macros), and its conv grad scatters into its
        // OWN per-seq grad_BS slot (off_bs_b(b,...)). The accumulation order
        // (b-outer, m-reverse-inner) is a fixed permutation of integer adds =>
        // bit-exact to summing the per-sequence runs.
        BWD_SWEEP_B: for (int b = 0; b < ML_BATCH_N; b++) {
            for (int m = ML_L_SEQ - 1; m >= 0; m--) {
                do_bwd(buffer, b, m, (int)toks_b[b][m]);
            }
        }
#else
        // (4) BACKWARD m = L-1..0 (REVERSE).
        BWD_SWEEP: for (int m = ML_L_SEQ - 1; m >= 0; m--) {
            do_bwd(buffer, 0, m, (int)toks[m]);
        }
#endif
#if defined(TT_ONCHIP_GRAD) && !defined(TT_RESIDENT_W)
        // flush on-chip grads -> buffer (one wide burst each) so signsgd is unchanged.
        burst_store_words(g_embed_oc, buffer, OFF_G_EMBED,  WORDS_G_EMBED);
        burst_store_words(g_mh_oc,    buffer, OFF_G_MH,     WORDS_G_MH);
        burst_store_words(g_mp_oc,    buffer, OFF_G_MP,     WORDS_G_MP);
        burst_store_words(g_mb_oc,    buffer, OFF_G_MB,     WORDS_G_MB);
        burst_store_words(g_whead_oc, buffer, OFF_G_W_HEAD, WORDS_G_W_HEAD);
#endif
        // (5) SignSGD weight update (resident: on-chip; else writes weights to buffer).
        do_signsgd(buffer, lr, threshold);
    }
#ifdef TT_RESIDENT_W
    // write resident weights back to the buffer for host read-back / next AFI run.
    burst_store_words(embed_oc, buffer, OFF_EMBED,  WORDS_EMBED);
    burst_store_words(mh_buf,   buffer, OFF_MH,     WORDS_MH);
    burst_store_words(mb_buf,   buffer, OFF_MB,     WORDS_MB);
    burst_store_words(mp_buf,   buffer, OFF_MP,     WORDS_MP);
    burst_store_words(wh_buf,   buffer, OFF_W_HEAD, WORDS_W_HEAD);
#endif
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

    // On SILICON only the fused whole-run trainer (mode 3) is dispatched. The
    // per-position modes 0/1/2 are kept for the HOST CSIM golden (which validates
    // fused == per-position) but excluded from synthesis: dispatching them
    // standalone makes HLS instantiate a SECOND copy of do_fwd_loss/do_bwd/
    // do_signsgd (~2x CL area), which congested the signsgd path to -0.346 ns.
    // do_train_run still calls these functions, so the logic is synthesized once.
#if !(defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__))
    // Per-position host-CSIM modes operate on sequence 0 (b=0). Under TT_BATCH the
    // golden drives a per-sequence reference by re-staging each sequence into the
    // b=0 slot — see golden_morph_lm.cpp's per-seq schedule.
    if (mode == ML_MODE_FWD)       return do_fwd_loss(buffer, 0, arg1, arg2, arg3, 1);
    if (mode == ML_MODE_BWD)       return do_bwd(buffer, 0, arg1, arg2);
    if (mode == ML_MODE_SIGNSGD)   return do_signsgd(buffer, arg1, arg2);
#endif
    if (mode == ML_MODE_TRAIN_RUN) return do_train_run(buffer, arg1, arg2, arg3);
    return 99;
}
