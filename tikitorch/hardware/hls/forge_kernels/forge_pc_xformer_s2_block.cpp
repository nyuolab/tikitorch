// forge_pc_xformer_s2_block.cpp — SCALE-2 (30M, D=512) DDR-STREAMING + K-WIDE-MAC transformer-block training
// shard. PHASE-1 of results/predictive_coding/SCALE_30M_124M_DESIGN.md.
//
// WHAT CHANGED vs forge_pc_xformer_mp_block (the D=64 base whose ARITHMETIC this preserves — bf_dyad_f, qact,
// dyadic LN, softmax, STE backward, dyadic-Adam, gradient accumulation, TRAIN gate, SLOT addressing):
//   (1) THE FOOTPRINT WALL (§2): weights / adam m,v / gradient accumulators are NO LONGER bload'd resident.
//       They are STREAMED from gmem in double-buffered [TILE_O x TILE_I] tiles (dyad'd on arrival for the
//       mult-free forward path) — see s2_gemm() / s2_adam_stream(). Only the per-stage ACTIVATIONS stay
//       resident (§2.4: at D=512 the post-checkpoint working set fits one SLR → NO sequence tiling needed;
//       the SEQ_TILE row-block loop is present so the SAME kernel scales to D=768 by DDR-staging activations).
//   (2) THE K-WIDE MAC ENGINE (§3): the contraction runs S2_K products/cycle into a balanced adder tree
//       (feed-forward, pipelines at inner II=1 for any K), ONE accumulator add per output per k-block.
//       cyc/output = ceil(I/K) (kills the mp kernel's cyc/output ∝ I² wall). s2_gemm(mode=...) is the reusable
//       engine, used by the forward matmul, the backward dx route, and the backward weight-grad.
//   (3) PRAW RECOMPUTE (§2.4): the backward does NOT load a cached H*SEQ*SEQ softmax-prob tensor; it recomputes
//       scores+softmax from the DDR-cached Q,K (S2_WQ/S2_WK) in the exact forward order (FlashAttention-style).
//   (4) TRAIN-ONLY: the deploy-int forward of the mp kernel is OMITTED (deploy comes later). The one sanctioned
//       multiply is the attention P@V blend + the bf16 backward route/wgrad/Jacobian (DSP; forward matmul stays
//       mult-free dyad-shift → DSP_MULT=0 outside attention, same invariant as the base kernel).
//
// NUMERICS: the wide forward is full-float (dyad weights + qact, no bf16 round); the backward rounds every
// intermediate through P() (== golden to_bf16). The K-wide TREE reduction reorders the sums vs the mp kernel's
// 16-lane [i&15] split, so gradients are NOT bit-exact to the mp kernel — they match a same-tree fp64 reference
// within <=1 bf16 ULP (the design's G3 within-1-ULP reconcile). Part xcvu47p-fsvh2892-2L-e, 4.0 ns.
#include <stdint.h>
#include <ap_int.h>
#include <hls_math.h>
#include "forge_pc_xformer_s2_config.hpp"

// ======================================================================================================
// bit-reinterpret + bf16 round + dyad — IDENTICAL to the mp kernel (the arithmetic to preserve).
// ======================================================================================================
static inline float bf16_round(float f) {
#pragma HLS INLINE
    union { float f; uint32_t u; } c; c.f = f;
    uint32_t u = c.u;
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    u &= 0xffff0000u;
    c.u = u; return c.f;
}
static inline float P(float x) { return bf16_round(x); }
static inline float i2f(int32_t i) { union { float f; int32_t i; } c; c.i = i; return c.f; }
static inline int32_t f2i(float f) { union { float f; int32_t i; } c; c.f = f; return c.i; }
static inline float bf_dyad_f(float wf) {
#pragma HLS INLINE
    float a = wf < 0 ? -wf : wf; if (a < 1e-30f) return 0.0f;
    int e = (int)hls::round(hls::log2(a));
    float p = hls::ldexp(1.0f, e);
    return wf < 0 ? -p : p;
}

// ---- 512-bit burst I/O (float lane + int32 lane) — one float/int per 32-bit lane, reused lane-for-lane ----
static void bload_f(ap_uint<512> *buf, int off, float *dst, int n) {
#pragma HLS INLINE off
LDF: for (int w = 0; w < n / S2_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = buf[off / S2_WL + w];
        for (int l = 0; l < S2_WL; l++) {
#pragma HLS UNROLL
            dst[w * S2_WL + l] = i2f((int32_t)(int)(ap_int<32>)word.range(32 * l + 31, 32 * l));
        }
    }
}
static void bstore_f(ap_uint<512> *buf, int off, const float *src, int n) {
#pragma HLS INLINE off
STF: for (int w = 0; w < n / S2_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = 0;
        for (int l = 0; l < S2_WL; l++) {
#pragma HLS UNROLL
            word.range(32 * l + 31, 32 * l) = (ap_uint<32>)(ap_int<32>)f2i(src[w * S2_WL + l]);
        }
        buf[off / S2_WL + w] = word;
    }
}
static void bload32(ap_uint<512> *buf, int off, int32_t *dst, int n) {
#pragma HLS INLINE off
LDW: for (int w = 0; w < n / S2_WL; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<512> word = buf[off / S2_WL + w];
        for (int l = 0; l < S2_WL; l++) {
#pragma HLS UNROLL
            dst[w * S2_WL + l] = (int32_t)(int)(ap_int<32>)word.range(32 * l + 31, 32 * l);
        }
    }
}

// ======================================================================================================
// WIDE-FLOAT primitives (== mp bf_qact / bf_layernorm / bf_ln_bwd; operate on the resident float activations).
// ======================================================================================================
static void s2_qact(float *x, int n) {
#pragma HLS INLINE off
    float mx = 0;
QA1: for (int i = 0; i < n; i++) { float a = x[i] < 0 ? -x[i] : x[i]; if (a > mx) mx = a; }
    if (mx < 1e-9f) mx = 1e-9f;
    int sh = (int)hls::floor(hls::log2(mx)) - S2_ABITS;
    float sc = hls::ldexp(1.0f, sh);
QA2: for (int i = 0; i < n; i++) x[i] = hls::round(x[i] / sc) * sc;
}
static void s2_layernorm(const float *x, const float *gamma, const float *beta,
                         float *y, float *s_out, int M) {
#pragma HLS INLINE off
LN_T: for (int t = 0; t < M; t++) {
        float sum = 0;
    LN_S: for (int d = 0; d < S2_D; d++) sum += x[t*S2_D+d];
        float mean = sum / S2_D; float asum = 0;
    LN_A: for (int d = 0; d < S2_D; d++) { float c = x[t*S2_D+d]-mean; asum += c < 0 ? -c : c; }
        float scale = asum / S2_D; if (scale < 1e-9f) scale = 1e-9f;
        int e = (int)hls::floor(hls::log2(scale));
        float mant = hls::round((scale/hls::ldexp(1.0f,e) - 1.0f)*(1<<S2_MBITS)) / (float)(1<<S2_MBITS);
        float s = (1.0f + mant) * hls::ldexp(1.0f,e);
        s_out[t] = s;
    LN_Y: for (int d = 0; d < S2_D; d++) y[t*S2_D+d] = gamma[d]*((x[t*S2_D+d]-mean)/s) + beta[d];
    }
}
static void s2_ln_bwd(const float *gy, const float *gamma, const float *s, float *dx, int M) {
#pragma HLS INLINE off
LB_T: for (int t = 0; t < M; t++) {
        float gxhat[S2_D], mg = 0;
    LB_C: for (int d = 0; d < S2_D; d++) { gxhat[d] = P(gy[t*S2_D+d]*gamma[d]); mg += gxhat[d]; }
        mg /= S2_D;
    LB_D: for (int d = 0; d < S2_D; d++) dx[t*S2_D+d] = P((gxhat[d]-mg)/s[t]);
    }
}

// ======================================================================================================
// THE K-WIDE MAC CORE — the reusable arithmetic (§3): K products into a balanced feed-forward adder tree.
// This is the single primitive s2_gemm's three modes (FWD/DX/WGRAD) share. K compile-time; K MUST be a power
// of two (asserted). Depth ceil(log2 K); pipelines at II=1 (no loop-carry inside the tree).
// ======================================================================================================
static_assert((S2_K & (S2_K-1)) == 0, "S2_K must be a power of two (balanced adder tree)");
static inline float s2_treeK(const float *a, const float *b) {
#pragma HLS INLINE
    float p[S2_K];
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
    for (int j = 0; j < S2_K; j++) {
#pragma HLS UNROLL
        p[j] = a[j] * b[j];                        // K parallel mult (dyad-shift fwd / bf16 fmul bwd)
    }
    for (int s = 1; s < S2_K; s <<= 1) {           // balanced pairwise tree
        for (int j = 0; j < S2_K; j += (s << 1)) {
#pragma HLS UNROLL
            p[j] = p[j] + p[j + s];
        }
    }
    return p[0];
}

// modes
#define S2M_FWD 0      // out[SEQ x O] = x[SEQ x I] @ dyad(W[O x I])^T ; weight streamed [O x I] ; raw write
#define S2M_DX  1      // dx[SEQ x I]  = gy[SEQ x O] @ dyad(W[O x I]) ; weight streamed transposed ; P write
#define S2M_WG  2      // dW[O x I]    = gyT[O x SEQ]^T-contraction @ xT[I x SEQ] ; resident ; P + GACC-RMW write

// ------------------------------------------------------------------------------------------------------
// s2_gemm — the reusable K-wide streamed contraction engine (used by fwd matmul, bwd dx, bwd wgrad).
//   out[r][c] = [P]( sum_{k<KC} lhs[r*KC+k] * B(c,k) ),  reduced K-at-a-time in GLOBAL k order (so the result
//   is BIT-IDENTICAL across TILE_I/SEQ_TILE choices — each output's contraction accumulates in the same order
//   regardless of tiling; tiles only gate weight (re)loads and row-block residency, never the arithmetic).
//   B(c,k):  FWD  = dyad(Wmaster[Woff + c*KC + k])                    (per-c contiguous-k gmem burst, ping-pong)
//            DX   = dyad(Wmaster[Woff + k*cols + c])                  (per-k contiguous-c gmem burst, transposed)
//            WG   = Bres[c*KC + k]   (Bres = xT, resident; no gmem, no dyad)
//   The inner pipelined loop iterates (r,c) per k-block → each cycle touches a DISTINCT y[r][c] accumulator, so
//   the fp-add into y is not a consecutive-cycle recurrence → inner II=1; cyc/output = ceil(KC/K).
//   lhs is resident contiguous-k. For WG the caller passes gyT/xT (transposed so k=SEQ is contiguous).
//   FWD/DX write the resident `out`. WG writes the gmem GACC region (buffer+gacc_off) read-modify-write.
// ------------------------------------------------------------------------------------------------------
static void s2_gemm(int mode, ap_uint<512> *buffer, int Woff,
                    const float *lhs, const float *Bres, float *out,
                    int rows, int cols, int KC,
                    int seq_tile, int tile_o, int tile_i,
                    int gacc_off, int gzero) {
#pragma HLS INLINE off
    // resident double-buffered weight/operand tiles [TILE_O x TILE_I] (0.13 MB @128x128, per §2.3)
    static float Btile0[S2_TILE_O * S2_TILE_I];
    static float Btile1[S2_TILE_O * S2_TILE_I];
#pragma HLS ARRAY_PARTITION variable=Btile0 cyclic factor=S2_K dim=1
#pragma HLS ARRAY_PARTITION variable=Btile1 cyclic factor=S2_K dim=1
    static float y[S2_SEQ_TILE * S2_TILE_O];       // per row-block output accumulators
    static float row[S2_TILE_O > S2_TILE_I ? S2_TILE_O : S2_TILE_I];
    static float gaccrow[S2_TILE_O];

    if (tile_o <= 0 || tile_o > S2_TILE_O) tile_o = S2_TILE_O;
    if (tile_i <= 0 || tile_i > S2_TILE_I) tile_i = S2_TILE_I;
    if (seq_tile <= 0 || seq_tile > S2_SEQ_TILE) seq_tile = S2_SEQ_TILE;

    for (int c0 = 0; c0 < cols; c0 += tile_o) {
        int TC = (cols - c0 < tile_o) ? (cols - c0) : tile_o;
        for (int r0 = 0; r0 < rows; r0 += seq_tile) {
            int TR = (rows - r0 < seq_tile) ? (rows - r0) : seq_tile;
            for (int i = 0; i < TR * tile_o; i++) y[i] = 0.0f;   // zero over the ACCESS stride (tile_o), not TC:
                                                                 // y is indexed y[r*tile_o+c], so a contiguous
                                                                 // TR*TC zero would leave stale gaps when tile_o>TC.

            int NI = (KC + tile_i - 1) / tile_i;              // number of contraction (I) tiles
            // -------- acquire i-tile 0 into Btile0 (double-buffer ping-pong across NI) --------
            float *bufA = Btile0, *bufB = Btile1;
            // (lambda-free: acquire inline via a goto-less helper pattern; replicate for prefetch)
            #define S2_ACQUIRE(DST, IT) do {                                                              \
                int _i0 = (IT) * tile_i; int _TI = (KC - _i0 < tile_i) ? (KC - _i0) : tile_i;             \
                if (mode == S2M_DX) {                                                                     \
                    for (int _k = 0; _k < _TI; _k++) {                                                    \
                        bload_f(buffer, Woff + (_i0 + _k) * cols + c0, row, TC);                          \
                        for (int _c = 0; _c < TC; _c++) (DST)[_c * tile_i + _k] = bf_dyad_f(row[_c]);     \
                    }                                                                                     \
                } else if (mode == S2M_FWD) {                                                             \
                    for (int _c = 0; _c < TC; _c++) {                                                     \
                        bload_f(buffer, Woff + (c0 + _c) * KC + _i0, row, _TI);                           \
                        for (int _k = 0; _k < _TI; _k++) (DST)[_c * tile_i + _k] = bf_dyad_f(row[_k]);    \
                    }                                                                                     \
                } else { /* WG: copy from resident Bres[c*KC + k] (xT) */                                 \
                    for (int _c = 0; _c < TC; _c++)                                                       \
                        for (int _k = 0; _k < _TI; _k++) (DST)[_c * tile_i + _k] = Bres[(c0 + _c) * KC + _i0 + _k]; \
                }                                                                                         \
            } while (0)

            S2_ACQUIRE(bufA, 0);
            for (int it = 0; it < NI; it++) {
                int i0 = it * tile_i;
                int TI = (KC - i0 < tile_i) ? (KC - i0) : tile_i;
                float *cur  = (it & 1) ? bufB : bufA;
                float *next = (it & 1) ? bufA : bufB;
                if (it + 1 < NI) S2_ACQUIRE(next, it + 1);    // prefetch tile it+1 while consuming it (ping-pong)
                // -------- consume i-tile `it`: K-wide MAC, inner II=1 over (r,c) per k-block --------
                for (int kb = 0; kb < TI; kb += S2_K) {       // k-blocks in GLOBAL k order (tile-independent)
                    const float *aptr_base = lhs + (r0) * KC + i0 + kb;
                    for (int r = 0; r < TR; r++) {
                        const float *aptr = aptr_base + r * KC;
                        for (int c = 0; c < TC; c++) {
#pragma HLS PIPELINE II=1
                            y[r * tile_o + c] += s2_treeK(aptr, cur + c * tile_i + kb);
                        }
                    }
                }
            }
            #undef S2_ACQUIRE
            // -------- finalize + write this row-block's outputs --------
            if (mode == S2M_WG) {
                // dW[o][i] : P-round, then GACC read-modify-write (zero on the first sub-batch of the window)
                for (int r = 0; r < TR; r++) {
                    if (!gzero) bload_f(buffer, gacc_off + (r0 + r) * cols + c0, gaccrow, TC);
                    for (int c = 0; c < TC; c++) {
                        float v = P(y[r * tile_o + c]);
                        if (!gzero) v += gaccrow[c];
                        gaccrow[c] = v;
                    }
                    bstore_f(buffer, gacc_off + (r0 + r) * cols + c0, gaccrow, TC);
                }
            } else {
                for (int r = 0; r < TR; r++)
                    for (int c = 0; c < TC; c++) {
                        float v = y[r * tile_o + c];
                        out[(r0 + r) * cols + c0 + c] = (mode == S2M_DX) ? P(v) : v;
                    }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------
// s2_adam_stream — dyadic-Adam apply as a DDR read-modify-write stream (§2.3.3). Reads [master, m, v, gacc]
// tiles, updates in place (II=1), writes back. m/v fp32 in DDR this version (bf16-in-DDR is a later size opt).
//   m = .875m + .125g ; v = .875v + .125|g| ; step = sign(m)*2^floor(log2(|m|/v)) ; W -= lr*step   (== golden).
// ------------------------------------------------------------------------------------------------------
static void s2_adam_stream(ap_uint<512> *buffer, int Woff, int mom_off, int var_off, int gacc_off,
                           int n, float lr) {
#pragma HLS INLINE off
    static float Wt[4096], mt[4096], vt[4096], gt[4096];
    const int TILE = 4096;                          // element tile (16-word aligned; n is a multiple)
    for (int b = 0; b < n; b += TILE) {
        int T = (n - b < TILE) ? (n - b) : TILE;
        bload_f(buffer, Woff + b, Wt, T);
        bload_f(buffer, mom_off + b, mt, T);
        bload_f(buffer, var_off + b, vt, T);
        bload_f(buffer, gacc_off + b, gt, T);
    AD: for (int i = 0; i < T; i++) {
#pragma HLS PIPELINE II=1
            float gv = gt[i], ag = gv < 0 ? -gv : gv;
            mt[i] = 0.875f*mt[i] + 0.125f*gv; vt[i] = 0.875f*vt[i] + 0.125f*ag;
            float am = mt[i] < 0 ? -mt[i] : mt[i], ratio = am/(vt[i]+1e-9f);
            float step = ratio > 0 ? hls::ldexp(1.0f, (int)hls::floor(hls::log2(ratio))) : 0.0f;
            float sgn = mt[i] > 0 ? 1.0f : (mt[i] < 0 ? -1.0f : 0.0f);
            Wt[i] -= lr*sgn*step;
        }
        bstore_f(buffer, Woff + b, Wt, T);
        bstore_f(buffer, mom_off + b, mt, T);
        bstore_f(buffer, var_off + b, vt, T);
    }
}

// transpose src[R x C] -> dst[C x R] (contiguous-k layout for the WGRAD engine)
static void s2_transpose(const float *src, float *dst, int R, int C) {
#pragma HLS INLINE off
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) dst[c * R + r] = src[r * C + c];
}

// convenience: run a full weight-grad (dW = gy^T @ x) via the WG engine + GACC-RMW. gyT/xT scratch resident.
static float s2_gyT[S2_FFN * S2_SEQ];   // gy transposed [O x SEQ]  (O up to FFN)
static float s2_xT [S2_FFN * S2_SEQ];   // x  transposed [I x SEQ]  (I up to FFN)
static void s2_wgrad(ap_uint<512> *buffer, const float *gy, const float *x, int O, int I, int M,
                     int gacc_off, int gzero, int seq_tile, int tile_o, int tile_i) {
#pragma HLS INLINE off
    s2_transpose(gy, s2_gyT, M, O);     // [SEQ x O] -> [O x SEQ]  (lhs, contiguous k=SEQ)
    s2_transpose(x,  s2_xT,  M, I);     // [SEQ x I] -> [I x SEQ]  (Bres, contiguous k=SEQ)
    // out=dW[O x I]: rows=O, cols=I, KC=M(SEQ). Note tile_i must divide/relate to KC=SEQ; SEQ is a mult of K.
    s2_gemm(S2M_WG, buffer, 0, s2_gyT, s2_xT, 0, O, I, M, seq_tile, tile_o, tile_i, gacc_off, gzero);
}

// ======================================================================================================
extern "C" void forge_pc_xformer_s2_block(ap_uint<512> *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    ap_uint<512> cfgw = buffer[S2O_CFG / S2_WL];
    int PHASE = (int)(ap_int<32>)cfgw.range(31, 0);
    int ROLE  = (int)(ap_int<32>)cfgw.range(63, 32);
    int DO_UP = (int)(ap_int<32>)cfgw.range(95, 64);
    int LRBITS= (int)(ap_int<32>)cfgw.range(127, 96);
    float LR  = (float)LRBITS * hls::ldexp(1.0f, -20);
    int GZERO = (int)(ap_int<32>)cfgw.range(159, 128);
    int TRAIN = (int)(ap_int<32>)cfgw.range(191, 160);   (void)TRAIN;   // train-only kernel (deploy omitted)
    int SLOT  = (int)(ap_int<32>)cfgw.range(223, 192);
    int TO    = (int)(ap_int<32>)cfgw.range(255, 224);   // runtime TILE_O (0 => default) — lets one binary sweep tiles
    int TI    = (int)(ap_int<32>)cfgw.range(287, 256);   // runtime TILE_I
    int ST    = (int)(ap_int<32>)cfgw.range(319, 288);   // runtime SEQ_TILE
    // slot offset for the per-sub-batch wide-cache / message regions (S=2 mirror scheme, DDR-only)
    const int po  = SLOT ? (S2_PAR_OFF + (SLOT - 1) * S2_PAR_SPAN) : 0;
    const int pto = SLOT * S2_SEQ;

    // ---- LN params + head/lnf (small; resident) — streamed matmul weights stay in gmem ----
    static float FLN1G[S2_D], FLN1B[S2_D], FLN2G[S2_D], FLN2B[S2_D];
    bload_f(buffer, S2_FLN1G, FLN1G, S2_D); bload_f(buffer, S2_FLN1B, FLN1B, S2_D);
    bload_f(buffer, S2_FLN2G, FLN2G, S2_D); bload_f(buffer, S2_FLN2B, FLN2B, S2_D);

    // =========================================================================================
    // FORWARD (WIDE float-master; fills the DDR cache the STE backward differentiates)
    // =========================================================================================
    if (PHASE == S2_PH_FWD) {
        static float WR[S2_SEQ*S2_D], WX1[S2_SEQ*S2_D], WQf[S2_SEQ*S2_D], WKf[S2_SEQ*S2_D], WVf[S2_SEQ*S2_D];
        static float WATTN[S2_SEQ*S2_D], WM[S2_SEQ*S2_D], WX2[S2_SEQ*S2_D], WHD[S2_SEQ*S2_FFN];
        static float WRP[S2_SEQ*S2_D], WRMASK[S2_SEQ*S2_FFN];
        static float ln1s[S2_SEQ], ln2s[S2_SEQ];

        // block input R: INPUT embeds via streamed float embed/pos; else the wide down-msg staged in S2_GRIN.
        if (ROLE == S2_RL_INPUT) {
            static int32_t tok[S2_SEQ]; bload32(buffer, S2_TOK + pto, tok, S2_SEQ);
            static float femb[S2_D], fpos[S2_D];
            for (int t = 0; t < S2_SEQ; t++) {
                int v = tok[t]; if (v < 0) v = 0; if (v >= S2_VOCAB) v = S2_VOCAB - 1;
                bload_f(buffer, S2_FEMB + v*S2_D, femb, S2_D);
                bload_f(buffer, S2_FPOS + t*S2_D, fpos, S2_D);
                for (int d = 0; d < S2_D; d++) WR[t*S2_D+d] = femb[d] + fpos[d];
            }
        } else {
            bload_f(buffer, S2_GRIN + po, WR, S2_SEQ*S2_D);
        }
        bstore_f(buffer, S2_WR + po, WR, S2_SEQ*S2_D);

        s2_layernorm(WR, FLN1G, FLN1B, WX1, ln1s, S2_SEQ);
        bstore_f(buffer, S2_WX1 + po, WX1, S2_SEQ*S2_D);

        // Q,K,V = qact( X1 @ dyad(W{Q,K,V})^T )  — streamed weight, K-wide, mult-free
        s2_gemm(S2M_FWD, buffer, S2_FWQ, WX1, 0, WQf, S2_SEQ, S2_D, S2_D, ST, TO, TI, 0, 0); s2_qact(WQf, S2_SEQ*S2_D);
        s2_gemm(S2M_FWD, buffer, S2_FWK, WX1, 0, WKf, S2_SEQ, S2_D, S2_D, ST, TO, TI, 0, 0); s2_qact(WKf, S2_SEQ*S2_D);
        s2_gemm(S2M_FWD, buffer, S2_FWV, WX1, 0, WVf, S2_SEQ, S2_D, S2_D, ST, TO, TI, 0, 0); s2_qact(WVf, S2_SEQ*S2_D);
        bstore_f(buffer, S2_WQ + po, WQf, S2_SEQ*S2_D); bstore_f(buffer, S2_WK + po, WKf, S2_SEQ*S2_D); bstore_f(buffer, S2_WV + po, WVf, S2_SEQ*S2_D);

        // attention (handrolled; minority of MACs). scores = Q.K/sqrt(DH), causal softmax, ATTN = P@V.
        float scale = 1.0f/hls::sqrt((float)S2_DH);
        for (int h = 0; h < S2_H; h++) for (int i = 0; i < S2_SEQ; i++) {
            float sc[S2_SEQ], mx = -1e30f;
            for (int j = 0; j <= i; j++) {
                float dot = 0;
                for (int d = 0; d < S2_DH; d++) dot += WQf[i*S2_D+h*S2_DH+d]*WKf[j*S2_D+h*S2_DH+d];
                sc[j] = dot*scale; if (sc[j] > mx) mx = sc[j];
            }
            float sum = 0, pr[S2_SEQ];
            for (int j = 0; j <= i; j++) { float e = hls::exp(sc[j]-mx); pr[j] = e; sum += e; }
            for (int j = 0; j <= i; j++) pr[j] /= sum;
            for (int d = 0; d < S2_DH; d++) {
                float blend = 0;
                for (int j = 0; j <= i; j++) blend += pr[j]*WVf[j*S2_D+h*S2_DH+d];
                WATTN[i*S2_D+h*S2_DH+d] = blend;
            }
        }
        s2_qact(WATTN, S2_SEQ*S2_D);
        bstore_f(buffer, S2_WATTN + po, WATTN, S2_SEQ*S2_D);

        // AO = ATTN @ dyad(WO)^T ; M = R + AO
        static float AOf[S2_SEQ*S2_D];
        s2_gemm(S2M_FWD, buffer, S2_FWO, WATTN, 0, AOf, S2_SEQ, S2_D, S2_D, ST, TO, TI, 0, 0);
        for (int i = 0; i < S2_SEQ*S2_D; i++) WM[i] = WR[i] + AOf[i];
        bstore_f(buffer, S2_WM + po, WM, S2_SEQ*S2_D);

        s2_layernorm(WM, FLN2G, FLN2B, WX2, ln2s, S2_SEQ);
        bstore_f(buffer, S2_WX2 + po, WX2, S2_SEQ*S2_D);

        // HDpre = X2 @ dyad(WUP)^T ; relu (+ mask) ; qact
        static float HDpre[S2_SEQ*S2_FFN];
        s2_gemm(S2M_FWD, buffer, S2_FWUP, WX2, 0, HDpre, S2_SEQ, S2_FFN, S2_D, ST, TO, TI, 0, 0);
        for (int i = 0; i < S2_SEQ*S2_FFN; i++) { WRMASK[i] = HDpre[i] > 0 ? 1.0f : 0.0f; WHD[i] = WRMASK[i] > 0 ? HDpre[i] : 0.0f; }
        s2_qact(WHD, S2_SEQ*S2_FFN);
        bstore_f(buffer, S2_WHD + po, WHD, S2_SEQ*S2_FFN);
        bstore_f(buffer, S2_WRMASK + po, WRMASK, S2_SEQ*S2_FFN);

        // FO = HD @ dyad(WDN)^T ; RP = M + FO
        static float FOf[S2_SEQ*S2_D];
        s2_gemm(S2M_FWD, buffer, S2_FWDN, WHD, 0, FOf, S2_SEQ, S2_D, S2_FFN, ST, TO, TI, 0, 0);
        for (int i = 0; i < S2_SEQ*S2_D; i++) WRP[i] = WM[i] + FOf[i];
        bstore_f(buffer, S2_WRP + po, WRP, S2_SEQ*S2_D);
        bstore_f(buffer, S2_WLN1S + po, ln1s, S2_SEQ); bstore_f(buffer, S2_WLN2S + po, ln2s, S2_SEQ);

        if (ROLE == S2_RL_OUT) {
            static float FLNFG[S2_D], FLNFB[S2_D]; bload_f(buffer, S2_FLNFG, FLNFG, S2_D); bload_f(buffer, S2_FLNFB, FLNFB, S2_D);
            static float WXF[S2_SEQ*S2_D], WLG[S2_SEQ*S2_VOCAB], lnfs[S2_SEQ];
            s2_layernorm(WRP, FLNFG, FLNFB, WXF, lnfs, S2_SEQ);
            s2_gemm(S2M_FWD, buffer, S2_FWHEAD, WXF, 0, WLG, S2_SEQ, S2_VOCAB, S2_D, ST, TO, TI, 0, 0);
            bstore_f(buffer, S2_WXF + po, WXF, S2_SEQ*S2_D); bstore_f(buffer, S2_WLG + po, WLG, S2_SEQ*S2_VOCAB);
            bstore_f(buffer, S2_WLNFS + po, lnfs, S2_SEQ);
        } else {
            bstore_f(buffer, S2_GROUT + po, WRP, S2_SEQ*S2_D);   // wide down-msg to the next chip
        }
        return;
    }

    // =========================================================================================
    // BACKWARD (BF16 STE textbook backprop; PRAW recompute; streamed wgrad/adam)
    // =========================================================================================
    static float WX1[S2_SEQ*S2_D], WQf[S2_SEQ*S2_D], WKf[S2_SEQ*S2_D], WVf[S2_SEQ*S2_D];
    static float WATTN[S2_SEQ*S2_D], WX2[S2_SEQ*S2_D], WHD[S2_SEQ*S2_FFN], WRMASK[S2_SEQ*S2_FFN];
    static float ln1s[S2_SEQ], ln2s[S2_SEQ];
    bload_f(buffer, S2_WX1 + po, WX1, S2_SEQ*S2_D);
    bload_f(buffer, S2_WQ + po, WQf, S2_SEQ*S2_D); bload_f(buffer, S2_WK + po, WKf, S2_SEQ*S2_D); bload_f(buffer, S2_WV + po, WVf, S2_SEQ*S2_D);
    bload_f(buffer, S2_WATTN + po, WATTN, S2_SEQ*S2_D);
    bload_f(buffer, S2_WX2 + po, WX2, S2_SEQ*S2_D); bload_f(buffer, S2_WHD + po, WHD, S2_SEQ*S2_FFN);
    bload_f(buffer, S2_WRMASK + po, WRMASK, S2_SEQ*S2_FFN);
    bload_f(buffer, S2_WLN1S + po, ln1s, S2_SEQ); bload_f(buffer, S2_WLN2S + po, ln2s, S2_SEQ);

    static float gRP[S2_SEQ*S2_D];
    if (ROLE == S2_RL_OUT) {
        // CE on the wide logits: grad = softmax(WLG) - onehot, mean over tokens ; head + LNF backward
        static float WXF[S2_SEQ*S2_D], WLG[S2_SEQ*S2_VOCAB], lnfs[S2_SEQ];
        bload_f(buffer, S2_WXF + po, WXF, S2_SEQ*S2_D); bload_f(buffer, S2_WLG + po, WLG, S2_SEQ*S2_VOCAB);
        bload_f(buffer, S2_WLNFS + po, lnfs, S2_SEQ);
        static int32_t tgt[S2_SEQ]; bload32(buffer, S2_TGT + pto, tgt, S2_SEQ);
        static float FLNFG[S2_D]; bload_f(buffer, S2_FLNFG, FLNFG, S2_D);
        static float gLG[S2_SEQ*S2_VOCAB];
        for (int t = 0; t < S2_SEQ; t++) {
            float mx = -1e30f;
            for (int v = 0; v < S2_VOCAB; v++) { float l = WLG[t*S2_VOCAB+v]; if (l > mx) mx = l; }
            float sum = 0, pr[S2_VOCAB];
            for (int v = 0; v < S2_VOCAB; v++) { pr[v] = hls::exp(WLG[t*S2_VOCAB+v]-mx); sum += pr[v]; }
            for (int v = 0; v < S2_VOCAB; v++) { float p = pr[v]/sum, oh = (v==tgt[t])?1.0f:0.0f; gLG[t*S2_VOCAB+v] = P((p-oh)/S2_SEQ); }
        }
        static float gXF[S2_SEQ*S2_D];
        s2_gemm(S2M_DX, buffer, S2_FWHEAD, gLG, 0, gXF, S2_SEQ, S2_D, S2_VOCAB, ST, TO, TI, 0, 0);   // dx = gLG@dyad(WHEAD)
        s2_wgrad(buffer, gLG, WXF, S2_VOCAB, S2_D, S2_SEQ, S2_GACC_WH, GZERO, ST, TO, TI);            // dW head + GACC
        s2_ln_bwd(gXF, FLNFG, lnfs, gRP, S2_SEQ);
#ifdef S2_DEBUG_GRADS
        bload_f(buffer, S2_GACC_WH, gLG, S2_VOCAB*S2_D);   // reuse gLG buffer to dump the accumulated head grad
        bstore_f(buffer, S2_DG_WH, gLG, S2_VOCAB*S2_D);
#endif
        if (DO_UP) s2_adam_stream(buffer, S2_FWHEAD, S2_MOM+S2_OFF_FHEAD, S2_VAR+S2_OFF_FHEAD, S2_GACC_WH, S2_VOCAB*S2_D, LR);
    } else {
        static float GRIN[S2_SEQ*S2_D]; bload_f(buffer, S2_GRIN + po, GRIN, S2_SEQ*S2_D);
        for (int i = 0; i < S2_SEQ*S2_D; i++) gRP[i] = GRIN[i];
    }

    // (1) FFN backward: RP=M+FO, FO=WDN(HD), HD=qact(relu(WUP(X2)))
    static float gM[S2_SEQ*S2_D]; for (int i = 0; i < S2_SEQ*S2_D; i++) gM[i] = gRP[i];
    static float gHD[S2_SEQ*S2_FFN];
    s2_gemm(S2M_DX, buffer, S2_FWDN, gRP, 0, gHD, S2_SEQ, S2_FFN, S2_D, ST, TO, TI, 0, 0);       // gHD = gRP@dyad(WDN)
    s2_wgrad(buffer, gRP, WHD, S2_D, S2_FFN, S2_SEQ, S2_GACC_WDN, GZERO, ST, TO, TI);            // gWDN + GACC
    for (int i = 0; i < S2_SEQ*S2_FFN; i++) if (WRMASK[i] == 0.0f) gHD[i] = 0;
    static float gX2[S2_SEQ*S2_D];
    s2_gemm(S2M_DX, buffer, S2_FWUP, gHD, 0, gX2, S2_SEQ, S2_D, S2_FFN, ST, TO, TI, 0, 0);       // gX2 = gHD@dyad(WUP)
    s2_wgrad(buffer, gHD, WX2, S2_FFN, S2_D, S2_SEQ, S2_GACC_WUP, GZERO, ST, TO, TI);            // gWUP + GACC
    // (2) LN2 over M
    static float gM2[S2_SEQ*S2_D]; s2_ln_bwd(gX2, FLN2G, ln2s, gM2, S2_SEQ);
    for (int i = 0; i < S2_SEQ*S2_D; i++) gM[i] = P(gM[i] + gM2[i]);
    // (3) attention backward: M=R+AO, AO=WO(ATTN), ATTN=qact(P@V), P=softmax(QK/sqrt(DH)) [PRAW RECOMPUTE]
    static float gR[S2_SEQ*S2_D]; for (int i = 0; i < S2_SEQ*S2_D; i++) gR[i] = gM[i];
    static float gATTN[S2_SEQ*S2_D];
    s2_gemm(S2M_DX, buffer, S2_FWO, gM, 0, gATTN, S2_SEQ, S2_D, S2_D, ST, TO, TI, 0, 0);         // gATTN = gM@dyad(WO)
    s2_wgrad(buffer, gM, WATTN, S2_D, S2_D, S2_SEQ, S2_GACC_WO, GZERO, ST, TO, TI);              // gWO + GACC
    static float gQ[S2_SEQ*S2_D], gK[S2_SEQ*S2_D], gV[S2_SEQ*S2_D];
    for (int i = 0; i < S2_SEQ*S2_D; i++) { gQ[i]=0; gK[i]=0; gV[i]=0; }
    float ascale = 1.0f/hls::sqrt((float)S2_DH);
    for (int h = 0; h < S2_H; h++) for (int i = 0; i < S2_SEQ; i++) {
        // ---- PRAW RECOMPUTE: rebuild this (h,i) softmax row from cached Q,K in the exact forward order ----
        float p[S2_SEQ], mx = -1e30f;
        for (int j = 0; j <= i; j++) {
            float dot = 0;
            for (int d = 0; d < S2_DH; d++) dot += WQf[i*S2_D+h*S2_DH+d]*WKf[j*S2_D+h*S2_DH+d];
            p[j] = dot*ascale; if (p[j] > mx) mx = p[j];
        }
        float sum = 0;
        for (int j = 0; j <= i; j++) { float e = hls::exp(p[j]-mx); p[j] = e; sum += e; }
        for (int j = 0; j <= i; j++) p[j] /= sum;
        // ---- attention backward on the recomputed probs (== mp ABWD / twin scalar) ----
        float gp[S2_SEQ];
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int d = 0; d < S2_DH; d++) s += gATTN[i*S2_D+h*S2_DH+d]*WVf[j*S2_D+h*S2_DH+d];
            gp[j] = s;
            for (int d = 0; d < S2_DH; d++) gV[j*S2_D+h*S2_DH+d] += P(p[j]*gATTN[i*S2_D+h*S2_DH+d]);
        }
        float ex = 0; for (int j = 0; j <= i; j++) ex += p[j]*gp[j];
        for (int j = 0; j <= i; j++) {
            float gs = P(p[j]*(gp[j]-ex))*ascale;
            for (int d = 0; d < S2_DH; d++) { gQ[i*S2_D+h*S2_DH+d] += gs*WKf[j*S2_D+h*S2_DH+d]; gK[j*S2_D+h*S2_DH+d] += gs*WQf[i*S2_D+h*S2_DH+d]; }
        }
    }
    for (int i = 0; i < S2_SEQ*S2_D; i++) { gQ[i]=P(gQ[i]); gK[i]=P(gK[i]); gV[i]=P(gV[i]); }
    // gX1 += dyad(W{Q,K,V})^T-route ; wgrad each ; then LN1
    static float gX1[S2_SEQ*S2_D]; for (int i = 0; i < S2_SEQ*S2_D; i++) gX1[i] = 0; static float tmp[S2_SEQ*S2_D];
    // (FLN1G already resident from the top-of-kernel load)
    s2_gemm(S2M_DX, buffer, S2_FWQ, gQ, 0, tmp, S2_SEQ, S2_D, S2_D, ST, TO, TI, 0, 0); for (int i=0;i<S2_SEQ*S2_D;i++) gX1[i]+=tmp[i];
    s2_wgrad(buffer, gQ, WX1, S2_D, S2_D, S2_SEQ, S2_GACC_WQ, GZERO, ST, TO, TI);
    s2_gemm(S2M_DX, buffer, S2_FWK, gK, 0, tmp, S2_SEQ, S2_D, S2_D, ST, TO, TI, 0, 0); for (int i=0;i<S2_SEQ*S2_D;i++) gX1[i]+=tmp[i];
    s2_wgrad(buffer, gK, WX1, S2_D, S2_D, S2_SEQ, S2_GACC_WK, GZERO, ST, TO, TI);
    s2_gemm(S2M_DX, buffer, S2_FWV, gV, 0, tmp, S2_SEQ, S2_D, S2_D, ST, TO, TI, 0, 0); for (int i=0;i<S2_SEQ*S2_D;i++) gX1[i]+=tmp[i];
    s2_wgrad(buffer, gV, WX1, S2_D, S2_D, S2_SEQ, S2_GACC_WV, GZERO, ST, TO, TI);
    for (int i = 0; i < S2_SEQ*S2_D; i++) gX1[i] = P(gX1[i]);
    static float gR1[S2_SEQ*S2_D]; s2_ln_bwd(gX1, FLN1G, ln1s, gR1, S2_SEQ);
    for (int i = 0; i < S2_SEQ*S2_D; i++) gR[i] = P(gR[i] + gR1[i]);

#ifdef S2_DEBUG_GRADS
    // dump the ACCUMULATED weight grads (post GACC-RMW, this sub-batch) + the raw activation grads, so the TB
    // can compare directly to the fp64 reference. (GACC holds the running sum; at GZERO it == this sub-batch.)
    { static float t[S2_FFN*S2_D];
      bload_f(buffer, S2_GACC_WQ,  t, S2_D*S2_D);   bstore_f(buffer, S2_DG_WQ,  t, S2_D*S2_D);
      bload_f(buffer, S2_GACC_WK,  t, S2_D*S2_D);   bstore_f(buffer, S2_DG_WK,  t, S2_D*S2_D);
      bload_f(buffer, S2_GACC_WV,  t, S2_D*S2_D);   bstore_f(buffer, S2_DG_WV,  t, S2_D*S2_D);
      bload_f(buffer, S2_GACC_WO,  t, S2_D*S2_D);   bstore_f(buffer, S2_DG_WO,  t, S2_D*S2_D);
      bload_f(buffer, S2_GACC_WUP, t, S2_FFN*S2_D); bstore_f(buffer, S2_DG_WUP, t, S2_FFN*S2_D);
      bload_f(buffer, S2_GACC_WDN, t, S2_D*S2_FFN); bstore_f(buffer, S2_DG_WDN, t, S2_D*S2_FFN); }
    bstore_f(buffer, S2_DG_GR, gR, S2_SEQ*S2_D);
    bstore_f(buffer, S2_DG_GQ, gQ, S2_SEQ*S2_D); bstore_f(buffer, S2_DG_GK, gK, S2_SEQ*S2_D); bstore_f(buffer, S2_DG_GV, gV, S2_SEQ*S2_D);
#endif

    // (5) dyadic-Adam apply (streamed RMW) on the matmul masters — only on the last sub-batch (DO_UP)
    if (DO_UP) {
        s2_adam_stream(buffer, S2_FWQ,  S2_MOM+S2_OFF_FWQ,  S2_VAR+S2_OFF_FWQ,  S2_GACC_WQ,  S2_D*S2_D,   LR);
        s2_adam_stream(buffer, S2_FWK,  S2_MOM+S2_OFF_FWK,  S2_VAR+S2_OFF_FWK,  S2_GACC_WK,  S2_D*S2_D,   LR);
        s2_adam_stream(buffer, S2_FWV,  S2_MOM+S2_OFF_FWV,  S2_VAR+S2_OFF_FWV,  S2_GACC_WV,  S2_D*S2_D,   LR);
        s2_adam_stream(buffer, S2_FWO,  S2_MOM+S2_OFF_FWO,  S2_VAR+S2_OFF_FWO,  S2_GACC_WO,  S2_D*S2_D,   LR);
        s2_adam_stream(buffer, S2_FWUP, S2_MOM+S2_OFF_FWUP, S2_VAR+S2_OFF_FWUP, S2_GACC_WUP, S2_FFN*S2_D, LR);
        s2_adam_stream(buffer, S2_FWDN, S2_MOM+S2_OFF_FWDN, S2_VAR+S2_OFF_FWDN, S2_GACC_WDN, S2_D*S2_FFN, LR);
    }

    // (6) emit: INPUT scatters gR -> float EMB/POS (+ GACC + adam) ; else the wide (bf16) up-message
    if (ROLE == S2_RL_INPUT) {
        static int32_t tok[S2_SEQ]; bload32(buffer, S2_TOK + pto, tok, S2_SEQ);
        static float gE[S2_VOCAB*S2_D], gPo[S2_SEQ*S2_D];
        for (int i = 0; i < S2_VOCAB*S2_D; i++) gE[i] = 0; for (int i = 0; i < S2_SEQ*S2_D; i++) gPo[i] = 0;
        for (int t = 0; t < S2_SEQ; t++) { int v = tok[t]; if (v<0)v=0; if (v>=S2_VOCAB)v=S2_VOCAB-1;
            for (int d = 0; d < S2_D; d++) { gE[v*S2_D+d] += gR[t*S2_D+d]; gPo[t*S2_D+d] += gR[t*S2_D+d]; } }
        // GACC fold (streamed via resident buffers here since gE/gPo already resident) + adam
        { static float aE[S2_VOCAB*S2_D];
          if (!GZERO) { bload_f(buffer, S2_GACC_EMB, aE, S2_VOCAB*S2_D); for (int i=0;i<S2_VOCAB*S2_D;i++) gE[i]+=aE[i];
                        bload_f(buffer, S2_GACC_POS, aE, S2_SEQ*S2_D);   for (int i=0;i<S2_SEQ*S2_D;i++)   gPo[i]+=aE[i]; }
          bstore_f(buffer, S2_GACC_EMB, gE, S2_VOCAB*S2_D); bstore_f(buffer, S2_GACC_POS, gPo, S2_SEQ*S2_D); }
        if (DO_UP) {
            s2_adam_stream(buffer, S2_FEMB, S2_MEMB, S2_VEMB, S2_GACC_EMB, S2_VOCAB*S2_D, LR);
            s2_adam_stream(buffer, S2_FPOS, S2_MPOS, S2_VPOS, S2_GACC_POS, S2_SEQ*S2_D,   LR);
        }
    } else {
        static float GROUT[S2_SEQ*S2_D]; for (int i = 0; i < S2_SEQ*S2_D; i++) GROUT[i] = P(gR[i]);
        bstore_f(buffer, S2_GROUT + po, GROUT, S2_SEQ*S2_D);
    }
}
