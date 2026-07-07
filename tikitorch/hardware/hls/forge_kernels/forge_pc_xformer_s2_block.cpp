// forge_pc_xformer_s2_block.cpp — SCALE-2 (30M, D=512) DDR-STREAMING + K-WIDE-MAC transformer-block training
// shard. PHASE-1 of results/predictive_coding/SCALE_30M_124M_DESIGN.md.
//
// WHAT CHANGED vs forge_pc_xformer_mp_block (the D=64 base whose ARITHMETIC this preserves — bf_dyad_f, qact,
// dyadic LN, softmax, STE backward, dyadic-Adam, gradient accumulation, TRAIN gate, SLOT addressing):
//   (1) THE FOOTPRINT WALL (§2): weights / adam m,v / gradient accumulators are NO LONGER bload'd resident.
//       They are STREAMED from gmem in double-buffered [TILE_O x TILE_I] tiles (dyad'd on arrival for the
//       mult-free forward path) — see s2_gemm() / s2_adam_stream().
//   (1b) THE G0 MEMORY FIX (§2.3/§2.4): the fwd/bwd ACTIVATION tensors (R,X1,Q,K,V,ATTN,M,X2,HD,mask,RP and the
//       bwd gradient counterparts, each SEQ*D or SEQ*FFN) are ALSO no longer full-SEQ resident. They live in the
//       DDR wide-cache / S2_BG_* scratch regions; every producer/consumer streams [SEQ_TILE x D]/[SEQ_TILE x FFN]
//       row-block STRIPS. The on-chip working set is now weight-tile ping-pong + a few strips + LN scalars +
//       per-head attention slices (~0.8 MB @ D=512, D/SEQ-independent) instead of ~34 MB → fits one SLR.
//   (2) THE K-WIDE MAC ENGINE (§3): the contraction runs S2_K products/cycle into a balanced adder tree
//       (feed-forward, pipelines at inner II=1 for any K), ONE accumulator add per output per k-block.
//       cyc/output = ceil(I/K). s2_gemm(mode=...) is the reusable engine (forward matmul, backward dx route,
//       backward weight-grad). lhs/out/Bres are now DDR OFFSETS streamed as tiles (no resident full tensors).
//   (3) PRAW RECOMPUTE (§2.4): the backward does NOT load a cached H*SEQ*SEQ softmax-prob tensor; it recomputes
//       scores+softmax from the DDR-cached Q,K (S2_WQ/S2_WK) in the exact forward order (streamed per head).
//   (4) TRAIN-ONLY: the deploy-int forward of the mp kernel is OMITTED (deploy comes later). The one sanctioned
//       multiply is the attention P@V blend + the bf16 backward route/wgrad/Jacobian (forward matmul stays
//       mult-free dyad-shift → DSP_MULT=0 outside attention, same invariant as the base kernel).
//
// NUMERICS: the wide forward is full-float (dyad weights + qact, no bf16 round); the backward rounds every
// intermediate through P() (== golden to_bf16). Streaming to DDR does NOT touch the arithmetic: every output's
// contraction is reduced K-at-a-time in GLOBAL k order regardless of TILE/SEQ_TILE, qact still reduces the max
// over the whole tensor, LN/softmax are per-row — so results are BIT-IDENTICAL across tile choices (TB CHECK 2)
// and match a same-tree fp64 reference within <=1 bf16 ULP (TB CHECK 1). Part xcvu47p-fsvh2892-2L-e, 4.0 ns.
//
// ================================ RESIDENT-ARRAY (BRAM) CENSUS @ D=512 defaults ================================
// The BUDGET is < ~5.4 MB (80% of the 6.75 MB one-SLR BRAM). Every persistent on-chip array, in bytes:
//   s2_gemm     Bcache          1 x [128*2048] fp32         =     1048576 = 1.05 MB  (FIX C: whole-c0 dyad'd weight
//               (single engine)                                                        cache, hoisted above r0)
//               lhstile         1 x [64*128]  fp32          =       32768 =  32 KB
//               y               1 x [64*128]  fp32          =       32768 =  32 KB
//               row/gaccrow     1 x [128*2]   fp32          =        1024 =   1 KB
//   s2_adam     Wt/mt/vt/gt     4 x [4096]    fp32          =     65536  =  64 KB   (adam RMW tiles)
//   s2_lnstrip  LN row-block    1 x [64*512]  fp32          =    131072  = 128 KB   (LN fwd/bwd, in place)
//   s2_tA/s2_tB elementwise     2 x [2048]    fp32          =     16384  =  16 KB   (copy/add/zero/qact/relu)
//   attn slices s2_Qh..s2_Dh    7 x [256*64]  fp32          = 7 x 65536  = 448 KB   (per-head Q/K/V/gATTN/gQ/gK/gV)
//   LN params   FLN1G/B,2G/B    4 x [512]     fp32          =      8192  =   8 KB
//   LN scalars  ln1s/ln2s/lnfs  3 x [256]     fp32          =      3072  =   3 KB
//   misc rows   femb/fpos/tok.. ~6 x [512]    fp32          =    ~12288  = ~12 KB
//   -----------------------------------------------------------------------------------------------------
//   TOTAL persistent on-chip                                                         ~= 1.9 MB  (< 5.4 MB budget)
//   (Bcache is TILE_O*KC_MAX = the per-c0 hoisted weight tile; KC_MAX=FFN=2048 at D=512. The single-engine 1.05 MB
//    is the FIX C cost — comfortably inside the one-SLR budget; DSP stays 1-2% as the MACs are dyad-shift.)
//   NOTE: the DEFAULT build is CONSERVATIVE (S2_AGGRESSIVE_SCHED undefined) — the s2_gemm engine uses Btile
//   [128*128]=64KB re-streamed per r0 instead of the 1.05MB hoisted Bcache, so persistent on-chip is ~0.9 MB.
//   Define -DS2_AGGRESSIVE_SCHED to restore the Bcache-hoist fast path (the 1.9 MB census above).
// Nothing above is a full-SEQ tensor; the ~34 MB of full-SEQ statics (WR..gV, gyT/xT, the 4 MB debug temp)
// are DELETED. The only sizes that scale with the model are the attn head slices (SEQ*DH) and the LN strip
// (SEQ_TILE*D); both stay small and D-tileable. (See the BEFORE census in the accompanying report.)
// ==============================================================================================================
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

// ======================================================================================================
// BIT-EXACT dyadic extraction — silicon == csim == CPU by construction.
// ------------------------------------------------------------------------------------------------------
// ROOT CAUSE of the D=512 s2 F2 divergence (proven by per-token silicon-vs-shim LN diff, relerr 2.6e-7 fit):
// the LayerNorm mantissa quantizer used hls::log2 + hls::round, which the Vitis HLS HARDWARE APPROXIMATES.
// On silicon the mantissa fails to round UP, so s = min(quant(scale), scale) instead of quant(scale) — a
// per-token LN-scale error up to 11% at S2_MBITS=2 that cascades to divergence. The ap_int/hls_math csim
// shim maps hls::log2/round to EXACT libm, so the bug was invisible to every csim/shim check. Fix: derive
// the dyadic exponent/mantissa straight from the IEEE-754 bits with INTEGER ops (identical on every target;
// validated bit-identical to libm over dense sweeps). These replace hls::log2/hls::round/hls::floor in the
// dyadic paths (LN normaliser, bf_dyad_f gemm operands, qact scale, dyadic-Adam step). ldexp is EXACT
// (it only adds to the exponent field) so it is retained.
static inline int s2_ilog2_floor(float x) {           // == (int)hls::floor(hls::log2(x)), x>0 normal
#pragma HLS INLINE
    union { float f; uint32_t u; } c; c.f = x;
    return (int)((c.u >> 23) & 0xFFu) - 127;
}
static inline int s2_ilog2_round(float x) {           // == (int)hls::round(hls::log2(x)), x>0 normal
#pragma HLS INLINE
    union { float f; uint32_t u; } c; c.f = x;
    int e = (int)((c.u >> 23) & 0xFFu) - 127;
    uint32_t frac = c.u & 0x7FFFFFu;                  // 3474676 = ceil((sqrt(2)-1)*2^23): mantissa>=sqrt(2) -> round up
    return e + (frac >= 3474676u ? 1 : 0);
}
static inline float s2_dyad_quant(float x, int MB) {  // == (1+round_MB((x/2^e)-1))*2^e, e=floor(log2 x); x>0 normal
#pragma HLS INLINE
    union { float f; uint32_t u; } c; c.f = x;
    int e = (int)((c.u >> 23) & 0xFFu) - 127;
    uint32_t frac = c.u & 0x7FFFFFu;
    int drop = 23 - MB;
    uint32_t r = (frac + (1u << (drop - 1))) >> drop; // MB-bit mantissa, round half-up; carry-safe: r in 0..2^MB
    float s = 1.0f + hls::ldexp((float)r, -MB);       // r*2^-MB via ldexp (exponent add) — NO mult/div => DSP_MULT=0
    return hls::ldexp(s, e);
}
static inline float s2_round_ne(float y) {            // == hls::round(y) (round half away from zero); |y|<2^23.
#pragma HLS INLINE                                    // float->int truncation is EXACT on HW, unlike hls::round which
    return (y >= 0.0f) ? (float)(int)(y + 0.5f)       // silicon APPROXIMATES (same class as the LN bug — qact grid).
                       : (float)(int)(y - 0.5f);
}

static inline float bf_dyad_f(float wf) {
#pragma HLS INLINE
    float a = wf < 0 ? -wf : wf; if (a < 1e-30f) return 0.0f;
    int e = s2_ilog2_round(a);                        // was (int)hls::round(hls::log2(a)) — silicon-approx; now bit-exact
    float p = hls::ldexp(1.0f, e);
    return wf < 0 ? -p : p;
}

// ======================================================================================================
// DE-PIPELINE GATE (DEFAULT OFF — the laned/pipelined reductions are the correct FAST path)
// ------------------------------------------------------------------------------------------------------
// HISTORY (corrected): this gate was added on the theory that the F2 divergence was a silicon
// pipelined-fp-accumulate HAZARD in the laned II=1 reductions. That theory was DISPROVEN: an AFI built with
// every reduction de-pipelined produced silicon LN1 (X1) IDENTICALLY 5.36e-2 wrong (same maxabs 6.3954) —
// de-pipelining changed NOTHING. The real bug was the LayerNorm dyadic quantiser using hls::log2 + hls::round,
// which the Vitis HLS hardware APPROXIMATES (s = min(quant,scale); see the bit-exact helpers above). With that
// fixed bit-exactly, silicon LN1 went to relerr 0.00 (100% bitmatch) — with the reductions STILL laned. So the
// laned II=1 reductions were always FAITHFUL on silicon (the 8-way lane interleave hides the fadd latency, so
// there is no read-during-write hazard); de-pipelining only cost ~5x throughput (sequential MAC IterLat 2421)
// and tripped the driver's 60s kick timeout on the OUTPUT chip's backward (KICK HUNG slot 7). DEFAULT is now
// OFF = laned/fast. The sequential path is kept behind the gate for reference only.
#ifndef S2_DEPIPELINE_REDUCE
#define S2_DEPIPELINE_REDUCE 0
#endif

#define S2_LANES 8
#if S2_DEPIPELINE_REDUCE
// De-pipelined path: the "lane" array is still filled by the caller's SEQUENTIAL loop, but as a defensive
// fold this is a plain sequential sum (no tree). Callers below no longer feed it laned — they accumulate a
// scalar directly — so this is only reached by the (now sequential) attention helpers that still stage a lane[].
static inline float s2_lanesum(float *lane) {      // sequential fold lane[0..L) -> scalar (csim-exact order)
#pragma HLS INLINE
    float acc = lane[0];
    for (int j = 1; j < S2_LANES; j++) acc += lane[j];   // NO PIPELINE: loop-carried fadd recurrence honoured
    return acc;
}
#else
// PERF (pre-fix) path: L independent accumulators indexed [i & (L-1)] break the loop-carried fp-add recurrence
// so a reduction loop pipelines at achieved-II=1; a final balanced tree folds the L lanes. THIS IS THE SILICON
// HAZARD — kept only for reference / A-B, never the default build.
static inline float s2_lanesum(float *lane) {      // S2_LANES power of two; folds lane[0..L) -> scalar (in place)
#pragma HLS INLINE
    for (int s = 1; s < S2_LANES; s <<= 1)
        for (int j = 0; j < S2_LANES; j += (s << 1)) {
#pragma HLS UNROLL
            lane[j] += lane[j + s];
        }
    return lane[0];
}
#endif

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
// STREAMING DDR ELEMENTWISE HELPERS (§2.3): read a DDR tensor tile-by-tile, transform, write back. Small
// resident element tiles (S2_ELT_TILE = 2048 = 8 KB each). All ops are pointwise / per-row so bit-identical
// to the pre-streaming resident versions regardless of tiling.
// ======================================================================================================
#define S2_ELT_TILE 2048
static float s2_tA[S2_ELT_TILE];
static float s2_tB[S2_ELT_TILE];

static void s2_zero_ddr(ap_uint<512> *buffer, int off, int n) {
#pragma HLS INLINE off
    for (int b = 0; b < n; b += S2_ELT_TILE) {
        int T = (n - b < S2_ELT_TILE) ? (n - b) : S2_ELT_TILE;
        for (int i = 0; i < T; i++) s2_tA[i] = 0.0f;
        bstore_f(buffer, off + b, s2_tA, T);
    }
}
// dst = doP? P(src) : src  (src==dst allowed — read tile before writing)
static void s2_copy_ddr(ap_uint<512> *buffer, int src, int dst, int n, int doP) {
#pragma HLS INLINE off
    for (int b = 0; b < n; b += S2_ELT_TILE) {
        int T = (n - b < S2_ELT_TILE) ? (n - b) : S2_ELT_TILE;
        bload_f(buffer, src + b, s2_tA, T);
        if (doP) for (int i = 0; i < T; i++) s2_tA[i] = P(s2_tA[i]);
        bstore_f(buffer, dst + b, s2_tA, T);
    }
}
// dst = doP? P(a+b) : (a+b)   (b==dst or a==dst allowed — both tiles read before writing)
static void s2_add_ddr(ap_uint<512> *buffer, int aoff, int boff, int dst, int n, int doP) {
#pragma HLS INLINE off
    for (int b = 0; b < n; b += S2_ELT_TILE) {
        int T = (n - b < S2_ELT_TILE) ? (n - b) : S2_ELT_TILE;
        bload_f(buffer, aoff + b, s2_tA, T);
        bload_f(buffer, boff + b, s2_tB, T);
        for (int i = 0; i < T; i++) { float v = s2_tA[i] + s2_tB[i]; s2_tA[i] = doP ? P(v) : v; }
        bstore_f(buffer, dst + b, s2_tA, T);
    }
}
// qact IN PLACE on a DDR tensor: two streaming passes (global max abs, then quantize). Bit-identical to the
// resident s2_qact (same global max set + same round math) regardless of how the tensor was tiled to write it.
static void s2_qact_ddr(ap_uint<512> *buffer, int off, int n) {
#pragma HLS INLINE off
    float mx = 0;
    for (int b = 0; b < n; b += S2_ELT_TILE) {
        int T = (n - b < S2_ELT_TILE) ? (n - b) : S2_ELT_TILE;
        bload_f(buffer, off + b, s2_tA, T);
        for (int i = 0; i < T; i++) { float a = s2_tA[i] < 0 ? -s2_tA[i] : s2_tA[i]; if (a > mx) mx = a; }
    }
    if (mx < 1e-9f) mx = 1e-9f;
    int sh = s2_ilog2_floor(mx) - S2_ABITS;   // was (int)hls::floor(hls::log2(mx)) — silicon-approx; now bit-exact
    float sc = hls::ldexp(1.0f, sh);
    for (int b = 0; b < n; b += S2_ELT_TILE) {
        int T = (n - b < S2_ELT_TILE) ? (n - b) : S2_ELT_TILE;
        bload_f(buffer, off + b, s2_tA, T);
        for (int i = 0; i < T; i++) s2_tA[i] = s2_round_ne(s2_tA[i] / sc) * sc;   // was hls::round — silicon-approx; now bit-exact
        bstore_f(buffer, off + b, s2_tA, T);
    }
}
// relu forward on FFN hidden: mask[i]=hd[i]>0?1:0 ; hd[i]=relu(hd[i]).  Streamed [SEQ_TILE-ish] element tiles.
static void s2_relu_mask_ddr(ap_uint<512> *buffer, int hd_off, int mask_off, int n) {
#pragma HLS INLINE off
    for (int b = 0; b < n; b += S2_ELT_TILE) {
        int T = (n - b < S2_ELT_TILE) ? (n - b) : S2_ELT_TILE;
        bload_f(buffer, hd_off + b, s2_tA, T);
        for (int i = 0; i < T; i++) { float m = s2_tA[i] > 0 ? 1.0f : 0.0f; s2_tB[i] = m; s2_tA[i] = m > 0 ? s2_tA[i] : 0.0f; }
        bstore_f(buffer, mask_off + b, s2_tB, T);
        bstore_f(buffer, hd_off + b, s2_tA, T);
    }
}
// relu backward mask apply: ghd[i] = mask[i]==0 ? 0 : ghd[i].
static void s2_relu_apply_ddr(ap_uint<512> *buffer, int ghd_off, int mask_off, int n) {
#pragma HLS INLINE off
    for (int b = 0; b < n; b += S2_ELT_TILE) {
        int T = (n - b < S2_ELT_TILE) ? (n - b) : S2_ELT_TILE;
        bload_f(buffer, ghd_off + b, s2_tA, T);
        bload_f(buffer, mask_off + b, s2_tB, T);
        for (int i = 0; i < T; i++) if (s2_tB[i] == 0.0f) s2_tA[i] = 0.0f;
        bstore_f(buffer, ghd_off + b, s2_tA, T);
    }
}

// ======================================================================================================
// STREAMING DDR LAYERNORM (fwd + bwd) — per-row, so a [SEQ_TILE x D] strip (whole rows) is bit-identical to
// the resident version. Rows of a [M x D] tensor are contiguous in DDR → one burst per strip. In place.
// ======================================================================================================
static float s2_lnstrip[S2_SEQ_TILE * S2_D];      // one row-block strip (whole D rows), reused fwd/bwd
static void s2_layernorm_ddr(ap_uint<512> *buffer, int x_off, const float *gamma, const float *beta,
                             int y_off, float *s_out, int M) {
#pragma HLS INLINE off
LNRB: for (int r0 = 0; r0 < M; r0 += S2_SEQ_TILE) {
        int TR = (M - r0 < S2_SEQ_TILE) ? (M - r0) : S2_SEQ_TILE;
        bload_f(buffer, x_off + r0 * S2_D, s2_lnstrip, TR * S2_D);
    LN_T: for (int t = 0; t < TR; t++) {
            float *xr = s2_lnstrip + t * S2_D;
#if S2_DEPIPELINE_REDUCE
            // DE-PIPELINE: sequential scalar mean/abs-dev sums. NO PIPELINE pragma → the tool honours the
            // loop-carried fadd recurrence and schedules LN_S/LN_A at II = fadd latency (>1). csim-exact.
            float msum = 0.0f;
        LN_S: for (int d = 0; d < S2_D; d++) msum += xr[d];
            float mean = msum / S2_D;
            float asum = 0.0f;
        LN_A: for (int d = 0; d < S2_D; d++) { float c = xr[d] - mean; asum += c < 0 ? -c : c; }
            float scale = asum / S2_D; if (scale < 1e-9f) scale = 1e-9f;
#else
            float slane[S2_LANES];                                   // FIX E: laned mean-sum (LN_S was II=6)
#pragma HLS ARRAY_PARTITION variable=slane complete dim=1
            for (int l = 0; l < S2_LANES; l++) slane[l] = 0;
        LN_S: for (int d = 0; d < S2_D; d++) {
#pragma HLS PIPELINE II=1
                slane[d & (S2_LANES-1)] += xr[d];
            }
            float mean = s2_lanesum(slane) / S2_D;
            float alane[S2_LANES];                                   // FIX E: laned abs-dev-sum (LN_A was II=6)
#pragma HLS ARRAY_PARTITION variable=alane complete dim=1
            for (int l = 0; l < S2_LANES; l++) alane[l] = 0;
        LN_A: for (int d = 0; d < S2_D; d++) {
#pragma HLS PIPELINE II=1
                float c = xr[d] - mean; alane[d & (S2_LANES-1)] += c < 0 ? -c : c;
            }
            float scale = s2_lanesum(alane) / S2_D; if (scale < 1e-9f) scale = 1e-9f;
#endif
            // was: e=floor(log2(scale)); mant=round((scale/2^e-1)*2^MBITS)/2^MBITS; s=(1+mant)*2^e.
            // hls::log2/hls::round are APPROXIMATE on F2 silicon -> s = min(quant,scale) (fails to round up)
            // -> up to 11% per-token LN error at MBITS=2 -> divergence. Now bit-exact (silicon == csim).
            float s = s2_dyad_quant(scale, S2_MBITS);
            s_out[r0 + t] = s;
        LN_Y: for (int d = 0; d < S2_D; d++) xr[d] = gamma[d] * ((xr[d] - mean) / s) + beta[d];
        }
        bstore_f(buffer, y_off + r0 * S2_D, s2_lnstrip, TR * S2_D);
    }
}
static void s2_ln_bwd_ddr(ap_uint<512> *buffer, int gy_off, const float *gamma, const float *s,
                          int dx_off, int M) {
#pragma HLS INLINE off
LBRB: for (int r0 = 0; r0 < M; r0 += S2_SEQ_TILE) {
        int TR = (M - r0 < S2_SEQ_TILE) ? (M - r0) : S2_SEQ_TILE;
        bload_f(buffer, gy_off + r0 * S2_D, s2_lnstrip, TR * S2_D);
    LB_T: for (int t = 0; t < TR; t++) {
            float *gr = s2_lnstrip + t * S2_D;
            float gxhat[S2_D];
#if S2_DEPIPELINE_REDUCE
            // DE-PIPELINE: sequential mean-of-gxhat. NO PIPELINE → loop-carried fadd on msum honoured, II>1. csim-exact.
            float msum = 0.0f;
        LB_C: for (int d = 0; d < S2_D; d++) { gxhat[d] = P(gr[d] * gamma[d]); msum += gxhat[d]; }
            float mg = msum / S2_D;
#else
            float mlane[S2_LANES];                                   // FIX E: laned mean-of-gxhat (LB_C was II=6)
#pragma HLS ARRAY_PARTITION variable=mlane complete dim=1
            for (int l = 0; l < S2_LANES; l++) mlane[l] = 0;
        LB_C: for (int d = 0; d < S2_D; d++) {
#pragma HLS PIPELINE II=1
                gxhat[d] = P(gr[d] * gamma[d]); mlane[d & (S2_LANES-1)] += gxhat[d];
            }
            float mg = s2_lanesum(mlane) / S2_D;
#endif
        LB_D: for (int d = 0; d < S2_D; d++) gr[d] = P((gxhat[d] - mg) / s[r0 + t]);
        }
        bstore_f(buffer, dx_off + r0 * S2_D, s2_lnstrip, TR * S2_D);
    }
}

// ======================================================================================================
// THE K-WIDE MAC CORE — the reusable arithmetic (§3): K products into a balanced feed-forward adder tree.
// K compile-time; K MUST be a power of two (asserted). Depth ceil(log2 K); pipelines at II=1.
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

// ------------------------------------------------------------------------------------------------------
// _S2_MAC_ITILE_BASE — the per-i-tile MAC over (r,c), shared by both engines (they differ only in the B
// operand base/stride: conservative Btile[c*tile_i+kb], aggressive Bcache[c*KC + i0 + kb]).
//   YACC   = the output accumulator array (y), accessed [r*TILE_O + c]
//   LHS    = lhstile, accessed [r*tile_i + kb]
//   BB     = B operand base pointer already offset to i-tile start (Btile, or Bcache+i0)
//   BSTR   = per-column stride of BB (conservative tile_i, aggressive KC)
//   TO/TI_ = tile_o / tile_i (compile-time strides)     TR/TC = live row/col counts     TIn = live k count
//
// DE-PIPELINE (S2_DEPIPELINE_REDUCE=1, DEFAULT): for each output cell (r,c) accumulate the WHOLE i-tile
// k-contraction into a SCALAR `acc` over the k-blocks (kb loop) with NO PIPELINE pragma, then y[r][c] += acc.
// The kb loop is a genuine loop-carried fp-add recurrence on `acc` → the tool CANNOT schedule it below II =
// the fadd latency (~4-8 cyc). No `y`-address is read-during-write across k-blocks anymore (the recurrence is
// the scalar `acc`, which the schedule serializes) → the silicon pipelined-accumulate hazard is gone. The
// k-blocks are visited in ASCENDING GLOBAL-k order and each block still reduces via s2_treeK, so the summation
// order is IDENTICAL to the previous laned path's per-cell order minus the lane reorder — csim BIT-EXACT vs a
// sequential reference (and the previous path was only within-1-ULP of that). Throughput: II≈fadd-latency per
// (r,c) k-block step instead of 1 — correctness-first, slower (see synth II note).
//
// PERF path (gate off): the old inner (r,c) PIPELINE II=1 with the same-address k-block recurrence — the
// SILICON HAZARD. Kept for A-B only.
#if S2_DEPIPELINE_REDUCE
#define _S2_MAC_ITILE_BASE(YACC, LHS, BB, BSTR, TO, TI_, TR, TC, TIn)                                          \
    for (int r = 0; r < (TR); r++) {                                                                           \
        for (int c = 0; c < (TC); c++) {                                                                       \
            float acc = 0.0f;                                                                                  \
            _Pragma("HLS PIPELINE off")                                                                        \
            for (int kb = 0; kb < (TIn); kb += S2_K) {   /* GLOBAL-k order; loop-carried fadd on acc -> II>1 */\
                acc += s2_treeK((LHS) + r * (TI_) + kb, (BB) + c * (BSTR) + kb);                               \
            }                                                                                                  \
            (YACC)[r * (TO) + c] += acc;                                                                       \
        }                                                                                                      \
    }
#else
#define _S2_MAC_ITILE_BASE(YACC, LHS, BB, BSTR, TO, TI_, TR, TC, TIn)                                          \
    for (int kb = 0; kb < (TIn); kb += S2_K) {                                                                 \
        for (int r = 0; r < (TR); r++) {                                                                       \
            const float *aptr = (LHS) + r * (TI_) + kb;                                                        \
            for (int c = 0; c < (TC); c++) {                                                                   \
                _Pragma("HLS PIPELINE II=1")                                                                   \
                _Pragma("HLS DEPENDENCE variable=y inter false") /* accumulator is always `y` (see 389); the  \
                    macro-param token YACC is NOT expanded inside a _Pragma string -> must name y directly */  \
                (YACC)[r * (TO) + c] += s2_treeK(aptr, (BB) + c * (BSTR) + kb);                                \
            }                                                                                                  \
        }                                                                                                      \
    }
#endif

// modes
#define S2M_FWD 0      // out[SEQ x O] = x[SEQ x I] @ dyad(W[O x I])^T ; weight streamed [O x I] ; raw write
#define S2M_DX  1      // dx[SEQ x I]  = gy[SEQ x O] @ dyad(W[O x I]) ; weight streamed transposed ; P write
#define S2M_WG  2      // dW[O x I]    = gy[SEQ x O]^T-contraction @ x[SEQ x I] ; both streamed ; P + GACC-RMW

// ------------------------------------------------------------------------------------------------------
// s2_gemm — the reusable K-wide streamed contraction engine (fwd matmul, bwd dx, bwd wgrad). EVERYTHING is
// a DDR offset now (lhs_off/Bres_off/out_off) — no resident full tensors. On-chip it holds only:
//   Bcache        [TILE_O x KC_MAX]  FIX C: ALL i-tiles for the current c0, weight-dyad'd (FWD/DX) or x (WG) ONCE
//   lhstile       [SEQ_TILE x TILE_I] the activation tile for the current (row-block, i-tile)
//   y             [SEQ_TILE x TILE_O] the output-strip accumulator
//
//   out[r][c] = [P]( sum_{k<KC} lhs[r][k] * B(c,k) ),  reduced K-at-a-time in GLOBAL k order → the result is
//   BIT-IDENTICAL across TILE_O/TILE_I/SEQ_TILE choices (each output's contraction accumulates in the same
//   order regardless of tiling; tiles only gate DMA + residency, never the arithmetic).
//   lhs[r][k]:  FWD/DX = gmem[lhs_off + (r0+r)*KC + k]                      (contiguous-k burst)
//               WG     = gmem[lhs_off + k*rows + (r0+r)]  (= gy^T, transposed on load; k=SEQ is the contraction)
//   B(c,k):     FWD    = dyad(gmem[Woff + c*KC + k])                        (per-c contiguous-k burst)
//               DX     = dyad(gmem[Woff + k*cols + c])                      (per-k contiguous-c burst, transposed)
//               WG     = gmem[Bres_off + k*cols + c]   (= x, transposed on load; no dyad)
//   FWD/DX stream the output strip to out_off; WG does the gmem GACC read-modify-write.
// ------------------------------------------------------------------------------------------------------
// FIX A: tile sizes are COMPILE-TIME (S2_TILE_O/S2_TILE_I/S2_SEQ_TILE constants; the runtime CFG-lane clamps are
//   gone) so the y[r*tile_o+c] access stride is a compile-time constant -> HLS can prove the MAC's reuse distance
//   (TR*TC) >> the ~49-cyc tree depth and pipeline the inner (r,c) MAC at achieved-II=1 (was II=3 from a false
//   inter-dependence on y). A belt-and-braces `#pragma HLS DEPENDENCE variable=y inter false` is added too.
// FIX C: the per-(c0) weight/x operand i-tiles are streamed + dyad'd ONCE into Bcache[TC x KC] ABOVE the r0
//   row-block loop (was re-acquired per row-block => 4x redundant DMA+dyad at SEQ=256). The MAC reads Bcache
//   in the SAME global-k order, so the per-output accumulation sequence is byte-identical (CHECK 2 bit-exact).
// The engine body is a macro over the function NAME only. The on-chip arrays keep FIXED source names (Bcache/
// lhstile/y/row/gaccrow) — each expansion is a distinct function, so its `static` locals are DISTINCT variables
// (a separate HW instance) even though they share the same source identifier; the HLS pragmas are function-scoped
// and refer to that function's own locals, so the _Pragma strings can be constant literals.
// ------------------------------------------------------------------------------------------------------
// S2_AGGRESSIVE_SCHED — GATE for the d86d8fd perf scheduling opts (DEFAULTS OFF = conservative). When OFF, the
// engine uses the PRE-OPT (72d427c) DDR access pattern + scheduling: the B operand is RE-STREAMED per r0
// row-block via a Btile0/Btile1 ping-pong (NO acquire-hoist), and the inner MAC pipelines WITHOUT the
// `DEPENDENCE variable=y inter false` assertion (so the tool honours the real y-accumulator recurrence and
// schedules a safe II). This targets the SILICON-ONLY backward divergence: FIX C's hoist (Bcache loaded once
// above r0) and the false-dependence II=1 forcing are csim-invisible scheduling/DMA-pattern changes — the
// two silicon-hazard suspects — while being ARITHMETICALLY identical (the MAC consumes k-blocks in the SAME
// GLOBAL-k order in both paths, so y accumulates byte-identically; csim stays bit-exact). Define
// -DS2_AGGRESSIVE_SCHED to restore the fast d86d8fd path.
// ------------------------------------------------------------------------------------------------------
#ifdef S2_AGGRESSIVE_SCHED
#define S2_GEMM_ENGINE(NAME)                                                                                   \
static void NAME(int mode, ap_uint<512> *buffer, int Woff,                                                    \
                 int lhs_off, int Bres_off, int out_off,                                                       \
                 int rows, int cols, int KC, int gacc_off, int gzero) {                                       \
    _Pragma("HLS INLINE off")                                                                                  \
    static float Bcache[S2_TILE_O * S2_KC_MAX];    /* FIX C: all i-tiles for the current c0, dyad'd ONCE */    \
    _Pragma("HLS ARRAY_PARTITION variable=Bcache cyclic factor=S2_K dim=1")                                    \
    static float lhstile[S2_SEQ_TILE * S2_TILE_I];                                                             \
    _Pragma("HLS ARRAY_PARTITION variable=lhstile cyclic factor=S2_K dim=1")                                   \
    static float y[S2_SEQ_TILE * S2_TILE_O];                                                                   \
    static float row[S2_TILE_O > S2_TILE_I ? S2_TILE_O : S2_TILE_I];                                           \
    static float gaccrow[S2_TILE_O];                                                                           \
    const int tile_o = S2_TILE_O, tile_i = S2_TILE_I, seq_tile = S2_SEQ_TILE;  /* FIX A: compile-time */       \
    for (int c0 = 0; c0 < cols; c0 += tile_o) {                                                                \
        int TC = (cols - c0 < tile_o) ? (cols - c0) : tile_o;                                                  \
        int NI = (KC + tile_i - 1) / tile_i;              /* number of contraction (I) tiles */                \
        /* ---- FIX C: HOIST the operand acquire above r0 — stream + dyad every i-tile ONCE per (c0). Bcache  */\
        /*      is laid out [c * KC + kglobal]; bursts stay row-major (per-row 128f = 512B min; only when     */\
        /*      NI==1 i.e. tile_i>=KC are rows contiguous enough to coalesce — never at D=512 dims, noted). */ \
        for (int it = 0; it < NI; it++) {                                                                      \
            int _i0 = it * tile_i; int _TI = (KC - _i0 < tile_i) ? (KC - _i0) : tile_i;                        \
            if (mode == S2M_DX) {                                                                             \
                for (int _k = 0; _k < _TI; _k++) {                                                            \
                    bload_f(buffer, Woff + (_i0 + _k) * cols + c0, row, TC);                                  \
                    for (int _c = 0; _c < TC; _c++) Bcache[_c * KC + _i0 + _k] = bf_dyad_f(row[_c]);          \
                }                                                                                             \
            } else if (mode == S2M_FWD) {                                                                     \
                for (int _c = 0; _c < TC; _c++) {                                                             \
                    bload_f(buffer, Woff + (c0 + _c) * KC + _i0, row, _TI);                                   \
                    for (int _k = 0; _k < _TI; _k++) Bcache[_c * KC + _i0 + _k] = bf_dyad_f(row[_k]);         \
                }                                                                                             \
            } else { /* WG: B = x[SEQ x I] transposed on load, no dyad */                                     \
                for (int _k = 0; _k < _TI; _k++) {                                                            \
                    bload_f(buffer, Bres_off + (_i0 + _k) * cols + c0, row, TC);                              \
                    for (int _c = 0; _c < TC; _c++) Bcache[_c * KC + _i0 + _k] = row[_c];                     \
                }                                                                                             \
            }                                                                                                 \
        }                                                                                                     \
        for (int r0 = 0; r0 < rows; r0 += seq_tile) {                                                          \
            int TR = (rows - r0 < seq_tile) ? (rows - r0) : seq_tile;                                          \
            for (int i = 0; i < TR * tile_o; i++) y[i] = 0.0f;   /* zero over the ACCESS stride tile_o */      \
            for (int it = 0; it < NI; it++) {                                                                  \
                int i0 = it * tile_i;                                                                          \
                int TI = (KC - i0 < tile_i) ? (KC - i0) : tile_i;                                              \
                /* -------- load the activation lhs i-tile (contiguous-k, or WG-transposed) -------- */       \
                if (mode == S2M_WG) {                                                                          \
                    for (int _k = 0; _k < TI; _k++) {                                                          \
                        bload_f(buffer, lhs_off + (i0 + _k) * rows + r0, row, TR);                             \
                        for (int _r = 0; _r < TR; _r++) lhstile[_r * tile_i + _k] = row[_r];                   \
                    }                                                                                          \
                } else {                                                                                       \
                    for (int _r = 0; _r < TR; _r++)                                                            \
                        bload_f(buffer, lhs_off + (r0 + _r) * KC + i0, lhstile + _r * tile_i, TI);             \
                }                                                                                              \
                /* -------- consume i-tile `it`: K-wide MAC. DE-PIPELINED accumulation -------- */             \
                _S2_MAC_ITILE_BASE(y, lhstile, (Bcache + i0), KC, tile_o, tile_i, TR, TC, TI)                  \
            }                                                                                                 \
            /* -------- finalize + write this row-block's outputs -------- */                                  \
            if (mode == S2M_WG) {                                                                              \
                for (int r = 0; r < TR; r++) {                                                                 \
                    if (!gzero) bload_f(buffer, gacc_off + (r0 + r) * cols + c0, gaccrow, TC);                 \
                    for (int c = 0; c < TC; c++) {                                                             \
                        float v = P(y[r * tile_o + c]);                                                        \
                        if (!gzero) v += gaccrow[c];                                                           \
                        gaccrow[c] = v;                                                                        \
                    }                                                                                          \
                    bstore_f(buffer, gacc_off + (r0 + r) * cols + c0, gaccrow, TC);                            \
                }                                                                                              \
            } else {                                                                                           \
                for (int r = 0; r < TR; r++) {                                                                 \
                    for (int c = 0; c < TC; c++) {                                                             \
                        float v = y[r * tile_o + c];                                                           \
                        row[c] = (mode == S2M_DX) ? P(v) : v;                                                  \
                    }                                                                                          \
                    bstore_f(buffer, out_off + (r0 + r) * cols + c0, row, TC);                                 \
                }                                                                                              \
            }                                                                                                 \
        }                                                                                                     \
    }                                                                                                         \
}
#else  /* CONSERVATIVE (default): pre-opt 72d427c engine — re-stream B per r0 (no hoist), no DEPENDENCE-false */
// REVERT of d86d8fd's two SILICON-hazard-suspect scheduling opts, LOGIC-IDENTICAL to the aggressive path:
//   * FIX C hoist OFF: the B operand i-tile is RE-STREAMED (+dyad'd) inside the r0 row-block loop into a single
//     Btile[TILE_O x TILE_I] (tile-local layout [c*tile_i+klocal]), instead of the whole-c0 Bcache hoisted once
//     above r0. The MAC consumes k-blocks in the SAME GLOBAL-k order, so y[r*tile_o+c] accumulates BYTE-IDENTICALLY
//     (csim bit-exact). This restores the conservative DDR access pattern (re-DMA per r0; no ping-pong prefetch —
//     the most conservative overlap, plain re-stream) and drops the resident weight cache 1.05MB -> 64KB.
//   * DEPENDENCE-false OFF: the inner MAC keeps `#pragma HLS PIPELINE II=1` (a REQUEST, not a forcing) but DROPS
//     `#pragma HLS DEPENDENCE variable=y inter false`, so the tool HONORS the real y-accumulator recurrence and
//     schedules a safe (higher) II rather than asserting no-dependence — removing the csim-invisible RAW-hazard risk.
// The macro takes NAME (== HEAD's ABI) but is instantiated ONCE (s2_gemm); locals keep 72d427c source names.
#define S2_GEMM_ENGINE(NAME)                                                                                   \
static void NAME(int mode, ap_uint<512> *buffer, int Woff,                                                    \
                 int lhs_off, int Bres_off, int out_off,                                                       \
                 int rows, int cols, int KC, int gacc_off, int gzero) {                                       \
    _Pragma("HLS INLINE off")                                                                                  \
    static float Btile[S2_TILE_O * S2_TILE_I];     /* REVERT FIX C: per-i-tile B, re-streamed inside r0 loop */ \
    _Pragma("HLS ARRAY_PARTITION variable=Btile cyclic factor=S2_K dim=1")                                     \
    static float lhstile[S2_SEQ_TILE * S2_TILE_I];                                                             \
    _Pragma("HLS ARRAY_PARTITION variable=lhstile cyclic factor=S2_K dim=1")                                   \
    static float y[S2_SEQ_TILE * S2_TILE_O];                                                                   \
    static float row[S2_TILE_O > S2_TILE_I ? S2_TILE_O : S2_TILE_I];                                           \
    static float gaccrow[S2_TILE_O];                                                                           \
    const int tile_o = S2_TILE_O, tile_i = S2_TILE_I, seq_tile = S2_SEQ_TILE;  /* FIX A tiles kept compile-time */\
    for (int c0 = 0; c0 < cols; c0 += tile_o) {                                                                \
        int TC = (cols - c0 < tile_o) ? (cols - c0) : tile_o;                                                  \
        int NI = (KC + tile_i - 1) / tile_i;              /* number of contraction (I) tiles */                \
        for (int r0 = 0; r0 < rows; r0 += seq_tile) {                                                          \
            int TR = (rows - r0 < seq_tile) ? (rows - r0) : seq_tile;                                          \
            for (int i = 0; i < TR * tile_o; i++) y[i] = 0.0f;   /* zero over the ACCESS stride tile_o */      \
            for (int it = 0; it < NI; it++) {                                                                  \
                int i0 = it * tile_i;                                                                          \
                int TI = (KC - i0 < tile_i) ? (KC - i0) : tile_i;                                              \
                /* -- acquire B i-tile `it` (weight FWD/DX dyad'd, or transposed-x WG) into Btile[c*tile_i+k] */\
                /*    RE-STREAMED per (c0,r0,it): conservative DDR pattern, no acquire-hoist above r0.        */\
                if (mode == S2M_DX) {                                                                          \
                    for (int _k = 0; _k < TI; _k++) {                                                          \
                        bload_f(buffer, Woff + (i0 + _k) * cols + c0, row, TC);                                \
                        for (int _c = 0; _c < TC; _c++) Btile[_c * tile_i + _k] = bf_dyad_f(row[_c]);          \
                    }                                                                                          \
                } else if (mode == S2M_FWD) {                                                                  \
                    for (int _c = 0; _c < TC; _c++) {                                                          \
                        bload_f(buffer, Woff + (c0 + _c) * KC + i0, row, TI);                                  \
                        for (int _k = 0; _k < TI; _k++) Btile[_c * tile_i + _k] = bf_dyad_f(row[_k]);          \
                    }                                                                                          \
                } else { /* WG: B = x[SEQ x I] transposed on load, no dyad */                                  \
                    for (int _k = 0; _k < TI; _k++) {                                                          \
                        bload_f(buffer, Bres_off + (i0 + _k) * cols + c0, row, TC);                            \
                        for (int _c = 0; _c < TC; _c++) Btile[_c * tile_i + _k] = row[_c];                     \
                    }                                                                                          \
                }                                                                                              \
                /* -------- load the activation lhs i-tile (contiguous-k, or WG-transposed) -------- */       \
                if (mode == S2M_WG) {                                                                          \
                    for (int _k = 0; _k < TI; _k++) {                                                          \
                        bload_f(buffer, lhs_off + (i0 + _k) * rows + r0, row, TR);                             \
                        for (int _r = 0; _r < TR; _r++) lhstile[_r * tile_i + _k] = row[_r];                   \
                    }                                                                                          \
                } else {                                                                                       \
                    for (int _r = 0; _r < TR; _r++)                                                            \
                        bload_f(buffer, lhs_off + (r0 + _r) * KC + i0, lhstile + _r * tile_i, TI);             \
                }                                                                                              \
                /* -------- consume i-tile `it`: K-wide MAC. DE-PIPELINED accumulation -------- */             \
                _S2_MAC_ITILE_BASE(y, lhstile, Btile, tile_i, tile_o, tile_i, TR, TC, TI)                      \
            }                                                                                                 \
            /* -------- finalize + write this row-block's outputs -------- */                                  \
            if (mode == S2M_WG) {                                                                              \
                for (int r = 0; r < TR; r++) {                                                                 \
                    if (!gzero) bload_f(buffer, gacc_off + (r0 + r) * cols + c0, gaccrow, TC);                 \
                    for (int c = 0; c < TC; c++) {                                                             \
                        float v = P(y[r * tile_o + c]);                                                        \
                        if (!gzero) v += gaccrow[c];                                                           \
                        gaccrow[c] = v;                                                                        \
                    }                                                                                          \
                    bstore_f(buffer, gacc_off + (r0 + r) * cols + c0, gaccrow, TC);                            \
                }                                                                                              \
            } else {                                                                                           \
                for (int r = 0; r < TR; r++) {                                                                 \
                    for (int c = 0; c < TC; c++) {                                                             \
                        float v = y[r * tile_o + c];                                                           \
                        row[c] = (mode == S2M_DX) ? P(v) : v;                                                  \
                    }                                                                                          \
                    bstore_f(buffer, out_off + (r0 + r) * cols + c0, row, TC);                                 \
                }                                                                                              \
            }                                                                                                 \
        }                                                                                                     \
    }                                                                                                         \
}
#endif  /* S2_AGGRESSIVE_SCHED */
// The engine is instantiated ONCE. (An earlier FIX G duplicated it as s2_gemm_b so the FFN backward's DX-route
// and WG for the SAME linear could overlap in a DATAFLOW region. csynth REJECTED that: two processes both read
// the single `gmem` m_axi bundle — "Bundled bus interface 'gmem' failed dataflow checking: cannot read data in
// multiple processes — s2_gemm and s2_gemm_b". Reverted: the FFN DX and WG run SEQUENTIALLY like every other
// DX/WG pair in this kernel — which is how the pre-FIX-G baseline already computed them, so numerics are
// unchanged. The throughput that FIX G chased now comes from the S=8 SLOT pipeline event loop in the driver.)
S2_GEMM_ENGINE(s2_gemm)

// ------------------------------------------------------------------------------------------------------
// s2_adam_stream — dyadic-Adam apply as a DDR read-modify-write stream (§2.3.3). (== golden; unchanged.)
// ------------------------------------------------------------------------------------------------------
static void s2_adam_stream(ap_uint<512> *buffer, int Woff, int mom_off, int var_off, int gacc_off,
                           int n, float lr) {
#pragma HLS INLINE off
    static float Wt[4096], mt[4096], vt[4096], gt[4096];
    const int TILE = 4096;
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
            float step = ratio > 0 ? hls::ldexp(1.0f, s2_ilog2_floor(ratio)) : 0.0f;   // bit-exact floor(log2)
            float sgn = mt[i] > 0 ? 1.0f : (mt[i] < 0 ? -1.0f : 0.0f);
            Wt[i] -= lr*sgn*step;
        }
        bstore_f(buffer, Woff + b, Wt, T);
        bstore_f(buffer, mom_off + b, mt, T);
        bstore_f(buffer, var_off + b, vt, T);
    }
}

// convenience: run a full weight-grad (dW = gy^T @ x) via the WG engine + GACC-RMW. gy/x stream from DDR;
// the SEQ-contiguous transpose the K-wide tree needs happens inside s2_gemm's tile loads (no resident gyT/xT).
static void s2_wgrad(ap_uint<512> *buffer, int gy_off, int x_off, int O, int I, int M,
                     int gacc_off, int gzero) {
#pragma HLS INLINE off
    s2_gemm(S2M_WG, buffer, 0, gy_off, x_off, 0, O, I, M, gacc_off, gzero);
}

// ======================================================================================================
// STREAMED ATTENTION (fwd + PRAW-recompute bwd) — per-head slices [SEQ x DH] streamed from the DDR Q/K/V
// caches. The math is IDENTICAL to the resident version (non-flash: full score row → max → exp → normalize),
// so it is bit-identical; only residency changed (a head slice, not the whole SEQ*D tensors).
// ======================================================================================================
static float s2_Qh[S2_SEQ * S2_DH];   // Q head slice
static float s2_Kh[S2_SEQ * S2_DH];   // K head slice
static float s2_Vh[S2_SEQ * S2_DH];   // V head slice
static float s2_Ah[S2_SEQ * S2_DH];   // ATTN out (fwd) / gATTN in (bwd)
static float s2_Bh[S2_SEQ * S2_DH];   // gQ head slice (bwd)
static float s2_Ch[S2_SEQ * S2_DH];   // gK head slice (bwd)
static float s2_Dh[S2_SEQ * S2_DH];   // gV head slice (bwd)

static void s2_attn_fwd(ap_uint<512> *buffer, int wq_off, int wk_off, int wv_off, int wattn_off) {
#pragma HLS INLINE off
    float scale = 1.0f / hls::sqrt((float)S2_DH);
    for (int h = 0; h < S2_H; h++) {
        for (int t = 0; t < S2_SEQ; t++) {
            bload_f(buffer, wq_off + t*S2_D + h*S2_DH, s2_Qh + t*S2_DH, S2_DH);
            bload_f(buffer, wk_off + t*S2_D + h*S2_DH, s2_Kh + t*S2_DH, S2_DH);
            bload_f(buffer, wv_off + t*S2_D + h*S2_DH, s2_Vh + t*S2_DH, S2_DH);
        }
        for (int i = 0; i < S2_SEQ; i++) {
            float sc[S2_SEQ], pr[S2_SEQ], mx = -1e30f;
            for (int j = 0; j <= i; j++) {
#if S2_DEPIPELINE_REDUCE
                float dot = 0.0f;                                        // DE-PIPELINE: sequential score-dot, II>1
                for (int d = 0; d < S2_DH; d++) dot += s2_Qh[i*S2_DH+d] * s2_Kh[j*S2_DH+d];
#else
                float lane[S2_LANES];                                    // FIX B: laned score-dot (was II=2)
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
                for (int l = 0; l < S2_LANES; l++) lane[l] = 0;
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS PIPELINE II=1
                    lane[d & (S2_LANES-1)] += s2_Qh[i*S2_DH+d] * s2_Kh[j*S2_DH+d];
                }
                float dot = s2_lanesum(lane);
#endif
                sc[j] = dot * scale; if (sc[j] > mx) mx = sc[j];
            }
#if S2_DEPIPELINE_REDUCE
            float esum = 0.0f;                                           // DE-PIPELINE: sequential exp-sum, II>1
            for (int j = 0; j <= i; j++) { float e = hls::exp(sc[j]-mx); pr[j] = e; esum += e; }
            float sum = esum;
#else
            float slane[S2_LANES];                                       // FIX B: laned exp-sum (was II=6)
#pragma HLS ARRAY_PARTITION variable=slane complete dim=1
            for (int l = 0; l < S2_LANES; l++) slane[l] = 0;
            for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                float e = hls::exp(sc[j]-mx); pr[j] = e; slane[j & (S2_LANES-1)] += e;
            }
            float sum = s2_lanesum(slane);
#endif
            for (int j = 0; j <= i; j++) pr[j] /= sum;
            for (int d = 0; d < S2_DH; d++) {
#if S2_DEPIPELINE_REDUCE
                float blend = 0.0f;                                      // DE-PIPELINE: sequential P@V blend, II>1
                for (int j = 0; j <= i; j++) blend += pr[j] * s2_Vh[j*S2_DH+d];
                s2_Ah[i*S2_DH+d] = blend;
#else
                float blane[S2_LANES];                                   // FIX B: laned P@V blend (was II=7)
#pragma HLS ARRAY_PARTITION variable=blane complete dim=1
                for (int l = 0; l < S2_LANES; l++) blane[l] = 0;
                for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                    blane[j & (S2_LANES-1)] += pr[j] * s2_Vh[j*S2_DH+d];
                }
                s2_Ah[i*S2_DH+d] = s2_lanesum(blane);
#endif
            }
        }
        for (int t = 0; t < S2_SEQ; t++) bstore_f(buffer, wattn_off + t*S2_D + h*S2_DH, s2_Ah + t*S2_DH, S2_DH);
    }
}

static void s2_attn_bwd(ap_uint<512> *buffer, int wq_off, int wk_off, int wv_off,
                        int gattn_off, int gq_off, int gk_off, int gv_off) {
#pragma HLS INLINE off
    float ascale = 1.0f / hls::sqrt((float)S2_DH);
    for (int h = 0; h < S2_H; h++) {
        for (int t = 0; t < S2_SEQ; t++) {
            bload_f(buffer, wq_off    + t*S2_D + h*S2_DH, s2_Qh + t*S2_DH, S2_DH);
            bload_f(buffer, wk_off    + t*S2_D + h*S2_DH, s2_Kh + t*S2_DH, S2_DH);
            bload_f(buffer, wv_off    + t*S2_D + h*S2_DH, s2_Vh + t*S2_DH, S2_DH);
            bload_f(buffer, gattn_off + t*S2_D + h*S2_DH, s2_Ah + t*S2_DH, S2_DH);
        }
        for (int i = 0; i < S2_SEQ*S2_DH; i++) { s2_Bh[i] = 0; s2_Ch[i] = 0; s2_Dh[i] = 0; }
        for (int i = 0; i < S2_SEQ; i++) {
            // ---- PRAW RECOMPUTE: rebuild this (h,i) softmax row from cached Q,K (SAME laned order as fwd) ----
            float p[S2_SEQ], mx = -1e30f;
            for (int j = 0; j <= i; j++) {
#if S2_DEPIPELINE_REDUCE
                float dot = 0.0f;                                        // DE-PIPELINE: sequential score-dot, II>1
                for (int d = 0; d < S2_DH; d++) dot += s2_Qh[i*S2_DH+d] * s2_Kh[j*S2_DH+d];
                p[j] = dot * ascale; if (p[j] > mx) mx = p[j];
#else
                float lane[S2_LANES];                                    // FIX B: laned score-dot
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
                for (int l = 0; l < S2_LANES; l++) lane[l] = 0;
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS PIPELINE II=1
                    lane[d & (S2_LANES-1)] += s2_Qh[i*S2_DH+d] * s2_Kh[j*S2_DH+d];
                }
                p[j] = s2_lanesum(lane) * ascale; if (p[j] > mx) mx = p[j];
#endif
            }
#if S2_DEPIPELINE_REDUCE
            float sum = 0.0f;                                            // DE-PIPELINE: sequential exp-sum, II>1
            for (int j = 0; j <= i; j++) { float e = hls::exp(p[j]-mx); p[j] = e; sum += e; }
#else
            float slane[S2_LANES];                                       // FIX B: laned exp-sum
#pragma HLS ARRAY_PARTITION variable=slane complete dim=1
            for (int l = 0; l < S2_LANES; l++) slane[l] = 0;
            for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                float e = hls::exp(p[j]-mx); p[j] = e; slane[j & (S2_LANES-1)] += e;
            }
            float sum = s2_lanesum(slane);
#endif
            for (int j = 0; j <= i; j++) p[j] /= sum;
            // ---- attention backward on the recomputed probs (== mp ABWD / twin scalar) ----
            float gp[S2_SEQ];
            for (int j = 0; j <= i; j++) {
#if S2_DEPIPELINE_REDUCE
                float gpv = 0.0f;                                        // DE-PIPELINE: sequential gp=<gATTN_i,V_j>, II>1
                for (int d = 0; d < S2_DH; d++) gpv += s2_Ah[i*S2_DH+d] * s2_Vh[j*S2_DH+d];
                gp[j] = gpv;
#else
                float glane[S2_LANES];                                   // FIX B: laned gp = <gATTN_i, V_j>
#pragma HLS ARRAY_PARTITION variable=glane complete dim=1
                for (int l = 0; l < S2_LANES; l++) glane[l] = 0;
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS PIPELINE II=1
                    glane[d & (S2_LANES-1)] += s2_Ah[i*S2_DH+d] * s2_Vh[j*S2_DH+d];
                }
                gp[j] = s2_lanesum(glane);
#endif
                for (int d = 0; d < S2_DH; d++) s2_Dh[j*S2_DH+d] += P(p[j] * s2_Ah[i*S2_DH+d]);   // gV: i-reduction, order kept
            }
#if S2_DEPIPELINE_REDUCE
            float ex = 0.0f;                                             // DE-PIPELINE: sequential ex=sum_j p*gp, II>1
            for (int j = 0; j <= i; j++) ex += p[j]*gp[j];
#else
            float exlane[S2_LANES];                                      // FIX B: laned ex = sum_j p*gp
#pragma HLS ARRAY_PARTITION variable=exlane complete dim=1
            for (int l = 0; l < S2_LANES; l++) exlane[l] = 0;
            for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                exlane[j & (S2_LANES-1)] += p[j]*gp[j];
            }
            float ex = s2_lanesum(exlane);
#endif
            // precompute gs[j] so the gQ (Bh, j-reduction) can be laned; gK (Ch) has no j-recurrence -> direct.
            float gsv[S2_SEQ];
            for (int j = 0; j <= i; j++) gsv[j] = P(p[j]*(gp[j]-ex)) * ascale;
            for (int j = 0; j <= i; j++)                                 // gK: Ch[j] accumulates over i (outer) -> order kept
                for (int d = 0; d < S2_DH; d++) s2_Ch[j*S2_DH+d] += gsv[j]*s2_Qh[i*S2_DH+d];
            for (int d = 0; d < S2_DH; d++) {
#if S2_DEPIPELINE_REDUCE
                float qacc = 0.0f;                                       // DE-PIPELINE: sequential gQ blend over j, II>1
                for (int j = 0; j <= i; j++) qacc += gsv[j]*s2_Kh[j*S2_DH+d];
                s2_Bh[i*S2_DH+d] += qacc;
#else
                float qlane[S2_LANES];                                   // FIX B: gQ (Bh) laned over j (was II=2)
#pragma HLS ARRAY_PARTITION variable=qlane complete dim=1
                for (int l = 0; l < S2_LANES; l++) qlane[l] = 0;
                for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                    qlane[j & (S2_LANES-1)] += gsv[j]*s2_Kh[j*S2_DH+d];
                }
                s2_Bh[i*S2_DH+d] += s2_lanesum(qlane);
#endif
            }
        }
        // heads write disjoint columns of gQ/gK/gV → the per-element accumulation is complete within this head,
        // so the final P() (== resident version's post-loop P) can be applied on the head-slice write.
        for (int t = 0; t < S2_SEQ; t++) {
            for (int d = 0; d < S2_DH; d++) { s2_Bh[t*S2_DH+d] = P(s2_Bh[t*S2_DH+d]); s2_Ch[t*S2_DH+d] = P(s2_Ch[t*S2_DH+d]); s2_Dh[t*S2_DH+d] = P(s2_Dh[t*S2_DH+d]); }
            bstore_f(buffer, gq_off + t*S2_D + h*S2_DH, s2_Bh + t*S2_DH, S2_DH);
            bstore_f(buffer, gk_off + t*S2_D + h*S2_DH, s2_Ch + t*S2_DH, S2_DH);
            bstore_f(buffer, gv_off + t*S2_D + h*S2_DH, s2_Dh + t*S2_DH, S2_DH);
        }
    }
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
    // FIX A: CFG lanes 7/8/9 (tile_o/tile_i/seq_tile) are no longer read — the tiles are compile-time now.
    const int po  = SLOT ? (S2_PAR_OFF + (SLOT - 1) * S2_PAR_SPAN) : 0;
    const int pto = SLOT * S2_SEQ;

    // ---- LN params (small; resident) — streamed matmul weights stay in gmem ----
    static float FLN1G[S2_D], FLN1B[S2_D], FLN2G[S2_D], FLN2B[S2_D];
    bload_f(buffer, S2_FLN1G, FLN1G, S2_D); bload_f(buffer, S2_FLN1B, FLN1B, S2_D);
    bload_f(buffer, S2_FLN2G, FLN2G, S2_D); bload_f(buffer, S2_FLN2B, FLN2B, S2_D);
    static float ln1s[S2_SEQ], ln2s[S2_SEQ], lnfs[S2_SEQ];

    // =========================================================================================
    // FORWARD (WIDE float-master; fills the DDR cache the STE backward differentiates). ALL activation
    // tensors live in the DDR wide-cache regions and are streamed row-blocked — nothing full-SEQ resident.
    // =========================================================================================
    if (PHASE == S2_PH_FWD) {
        // block input R -> S2_WR
        if (ROLE == S2_RL_INPUT) {
            static int32_t tok[S2_SEQ]; bload32(buffer, S2_TOK + pto, tok, S2_SEQ);
            static float femb[S2_D], fpos[S2_D], rrow[S2_D];
            for (int t = 0; t < S2_SEQ; t++) {
                int v = tok[t]; if (v < 0) v = 0; if (v >= S2_VOCAB) v = S2_VOCAB - 1;
                bload_f(buffer, S2_FEMB + v*S2_D, femb, S2_D);
                bload_f(buffer, S2_FPOS + t*S2_D, fpos, S2_D);
                for (int d = 0; d < S2_D; d++) rrow[d] = femb[d] + fpos[d];
                bstore_f(buffer, S2_WR + po + t*S2_D, rrow, S2_D);
            }
        } else {
            s2_copy_ddr(buffer, S2_GRIN + po, S2_WR + po, S2_SEQ*S2_D, 0);
        }

        s2_layernorm_ddr(buffer, S2_WR + po, FLN1G, FLN1B, S2_WX1 + po, ln1s, S2_SEQ);

        // Q,K,V = qact( X1 @ dyad(W{Q,K,V})^T ) — streamed weight, K-wide, mult-free
        s2_gemm(S2M_FWD, buffer, S2_FWQ, S2_WX1 + po, 0, S2_WQ + po, S2_SEQ, S2_D, S2_D, 0, 0); s2_qact_ddr(buffer, S2_WQ + po, S2_SEQ*S2_D);
        s2_gemm(S2M_FWD, buffer, S2_FWK, S2_WX1 + po, 0, S2_WK + po, S2_SEQ, S2_D, S2_D, 0, 0); s2_qact_ddr(buffer, S2_WK + po, S2_SEQ*S2_D);
        s2_gemm(S2M_FWD, buffer, S2_FWV, S2_WX1 + po, 0, S2_WV + po, S2_SEQ, S2_D, S2_D, 0, 0); s2_qact_ddr(buffer, S2_WV + po, S2_SEQ*S2_D);

        // attention (streamed per head): scores = Q.K/sqrt(DH), causal softmax, ATTN = P@V, then qact.
        s2_attn_fwd(buffer, S2_WQ + po, S2_WK + po, S2_WV + po, S2_WATTN + po);
        s2_qact_ddr(buffer, S2_WATTN + po, S2_SEQ*S2_D);

        // AO = ATTN @ dyad(WO)^T ; M = R + AO   (WM first holds AO, then the residual add)
        s2_gemm(S2M_FWD, buffer, S2_FWO, S2_WATTN + po, 0, S2_WM + po, S2_SEQ, S2_D, S2_D, 0, 0);
        s2_add_ddr(buffer, S2_WR + po, S2_WM + po, S2_WM + po, S2_SEQ*S2_D, 0);

        s2_layernorm_ddr(buffer, S2_WM + po, FLN2G, FLN2B, S2_WX2 + po, ln2s, S2_SEQ);

        // HDpre = X2 @ dyad(WUP)^T ; relu (+ mask) ; qact   (WHD first holds HDpre raw)
        s2_gemm(S2M_FWD, buffer, S2_FWUP, S2_WX2 + po, 0, S2_WHD + po, S2_SEQ, S2_FFN, S2_D, 0, 0);
        s2_relu_mask_ddr(buffer, S2_WHD + po, S2_WRMASK + po, S2_SEQ*S2_FFN);
        s2_qact_ddr(buffer, S2_WHD + po, S2_SEQ*S2_FFN);

        // FO = HD @ dyad(WDN)^T ; RP = M + FO   (WRP first holds FO)
        s2_gemm(S2M_FWD, buffer, S2_FWDN, S2_WHD + po, 0, S2_WRP + po, S2_SEQ, S2_D, S2_FFN, 0, 0);
        s2_add_ddr(buffer, S2_WM + po, S2_WRP + po, S2_WRP + po, S2_SEQ*S2_D, 0);

        bstore_f(buffer, S2_WLN1S + po, ln1s, S2_SEQ); bstore_f(buffer, S2_WLN2S + po, ln2s, S2_SEQ);

        if (ROLE == S2_RL_OUT) {
            static float FLNFG[S2_D], FLNFB[S2_D]; bload_f(buffer, S2_FLNFG, FLNFG, S2_D); bload_f(buffer, S2_FLNFB, FLNFB, S2_D);
            s2_layernorm_ddr(buffer, S2_WRP + po, FLNFG, FLNFB, S2_WXF + po, lnfs, S2_SEQ);
            s2_gemm(S2M_FWD, buffer, S2_FWHEAD, S2_WXF + po, 0, S2_WLG + po, S2_SEQ, S2_VOCAB, S2_D, 0, 0);
            bstore_f(buffer, S2_WLNFS + po, lnfs, S2_SEQ);
        } else {
            s2_copy_ddr(buffer, S2_WRP + po, S2_GROUT + po, S2_SEQ*S2_D, 0);   // wide down-msg to the next chip
        }
        return;
    }

    // =========================================================================================
    // BACKWARD (BF16 STE textbook backprop; PRAW recompute; streamed wgrad/adam). ALL gradient tensors live
    // in the DDR S2_BG_* scratch and are streamed row-blocked — nothing full-SEQ resident.
    // =========================================================================================
    bload_f(buffer, S2_WLN1S + po, ln1s, S2_SEQ); bload_f(buffer, S2_WLN2S + po, ln2s, S2_SEQ);

    if (ROLE == S2_RL_OUT) {
        // CE on the wide logits: grad = softmax(WLG) - onehot, mean over tokens ; head + LNF backward
        bload_f(buffer, S2_WLNFS + po, lnfs, S2_SEQ);
        static float FLNFG[S2_D]; bload_f(buffer, S2_FLNFG, FLNFG, S2_D);
        static int32_t tgt[S2_SEQ]; bload32(buffer, S2_TGT + pto, tgt, S2_SEQ);
        // per-row (per-token) softmax-onehot, streamed [SEQ_TILE x VOCAB] strips
        static float lgrow[S2_VOCAB], pr[S2_VOCAB];
        for (int t = 0; t < S2_SEQ; t++) {
            bload_f(buffer, S2_WLG + po + t*S2_VOCAB, lgrow, S2_VOCAB);
            float mx = -1e30f;
            for (int v = 0; v < S2_VOCAB; v++) { if (lgrow[v] > mx) mx = lgrow[v]; }
            float sum = 0;
            for (int v = 0; v < S2_VOCAB; v++) { pr[v] = hls::exp(lgrow[v]-mx); sum += pr[v]; }
            for (int v = 0; v < S2_VOCAB; v++) { float p = pr[v]/sum, oh = (v==tgt[t])?1.0f:0.0f; lgrow[v] = P((p-oh)/S2_SEQ); }
            bstore_f(buffer, S2_BG_LG + t*S2_VOCAB, lgrow, S2_VOCAB);
        }
        s2_gemm(S2M_DX, buffer, S2_FWHEAD, S2_BG_LG, 0, S2_BG_XF, S2_SEQ, S2_D, S2_VOCAB, 0, 0);   // dx = gLG@dyad(WHEAD)
        s2_wgrad(buffer, S2_BG_LG, S2_WXF + po, S2_VOCAB, S2_D, S2_SEQ, S2_GACC_WH, GZERO);         // dW head + GACC
        s2_ln_bwd_ddr(buffer, S2_BG_XF, FLNFG, lnfs, S2_BG_RP, S2_SEQ);                                         // gRP
        if (DO_UP) s2_adam_stream(buffer, S2_FWHEAD, S2_MOM+S2_OFF_FHEAD, S2_VAR+S2_OFF_FHEAD, S2_GACC_WH, S2_VOCAB*S2_D, LR);
    } else {
        s2_copy_ddr(buffer, S2_GRIN + po, S2_BG_RP, S2_SEQ*S2_D, 0);   // gRP = incoming wide error
    }

    // (1) FFN backward: RP=M+FO, FO=WDN(HD), HD=qact(relu(WUP(X2)))
    s2_copy_ddr(buffer, S2_BG_RP, S2_BG_M, S2_SEQ*S2_D, 0);                                                     // gM = gRP
    s2_gemm(S2M_DX, buffer, S2_FWDN, S2_BG_RP, 0, S2_BG_HD, S2_SEQ, S2_FFN, S2_D, 0, 0);                        // gHD = gRP@dyad(WDN)
    s2_wgrad(buffer, S2_BG_RP, S2_WHD + po, S2_D, S2_FFN, S2_SEQ, S2_GACC_WDN, GZERO);                          // gWDN + GACC
    s2_relu_apply_ddr(buffer, S2_BG_HD, S2_WRMASK + po, S2_SEQ*S2_FFN);                                         // mask gHD
    s2_gemm(S2M_DX, buffer, S2_FWUP, S2_BG_HD, 0, S2_BG_X2, S2_SEQ, S2_D, S2_FFN, 0, 0);   // gX2 = gHD@dyad(WUP)
    s2_wgrad(buffer, S2_BG_HD, S2_WX2 + po, S2_FFN, S2_D, S2_SEQ, S2_GACC_WUP, GZERO);      // gWUP + GACC
    // (2) LN2 over M
    s2_ln_bwd_ddr(buffer, S2_BG_X2, FLN2G, ln2s, S2_BG_TMP, S2_SEQ);                                            // gM2 -> TMP
    s2_add_ddr(buffer, S2_BG_M, S2_BG_TMP, S2_BG_M, S2_SEQ*S2_D, 1);                                            // gM = P(gM + gM2)
    // (3) attention backward: M=R+AO, AO=WO(ATTN), ATTN=qact(P@V), P=softmax(QK/sqrt(DH)) [PRAW RECOMPUTE]
    s2_copy_ddr(buffer, S2_BG_M, S2_BG_R, S2_SEQ*S2_D, 0);                                                      // gR = gM
    s2_gemm(S2M_DX, buffer, S2_FWO, S2_BG_M, 0, S2_BG_ATTN, S2_SEQ, S2_D, S2_D, 0, 0);                          // gATTN = gM@dyad(WO)
    s2_wgrad(buffer, S2_BG_M, S2_WATTN + po, S2_D, S2_D, S2_SEQ, S2_GACC_WO, GZERO);                            // gWO + GACC
    s2_attn_bwd(buffer, S2_WQ + po, S2_WK + po, S2_WV + po, S2_BG_ATTN, S2_BG_Q, S2_BG_K, S2_BG_V);            // gQ,gK,gV (P'd)
    // (4) gX1 += dyad(W{Q,K,V})^T-route ; wgrad each ; then LN1
    s2_zero_ddr(buffer, S2_BG_X1, S2_SEQ*S2_D);
    s2_gemm(S2M_DX, buffer, S2_FWQ, S2_BG_Q, 0, S2_BG_TMP, S2_SEQ, S2_D, S2_D, 0, 0); s2_add_ddr(buffer, S2_BG_X1, S2_BG_TMP, S2_BG_X1, S2_SEQ*S2_D, 0);
    s2_wgrad(buffer, S2_BG_Q, S2_WX1 + po, S2_D, S2_D, S2_SEQ, S2_GACC_WQ, GZERO);
    s2_gemm(S2M_DX, buffer, S2_FWK, S2_BG_K, 0, S2_BG_TMP, S2_SEQ, S2_D, S2_D, 0, 0); s2_add_ddr(buffer, S2_BG_X1, S2_BG_TMP, S2_BG_X1, S2_SEQ*S2_D, 0);
    s2_wgrad(buffer, S2_BG_K, S2_WX1 + po, S2_D, S2_D, S2_SEQ, S2_GACC_WK, GZERO);
    s2_gemm(S2M_DX, buffer, S2_FWV, S2_BG_V, 0, S2_BG_TMP, S2_SEQ, S2_D, S2_D, 0, 0); s2_add_ddr(buffer, S2_BG_X1, S2_BG_TMP, S2_BG_X1, S2_SEQ*S2_D, 0);
    s2_wgrad(buffer, S2_BG_V, S2_WX1 + po, S2_D, S2_D, S2_SEQ, S2_GACC_WV, GZERO);
    s2_copy_ddr(buffer, S2_BG_X1, S2_BG_X1, S2_SEQ*S2_D, 1);                                                    // gX1 = P(gX1)
    s2_ln_bwd_ddr(buffer, S2_BG_X1, FLN1G, ln1s, S2_BG_TMP, S2_SEQ);                                            // gR1 -> TMP
    s2_add_ddr(buffer, S2_BG_R, S2_BG_TMP, S2_BG_R, S2_SEQ*S2_D, 1);                                            // gR = P(gR + gR1)

#ifdef S2_DEBUG_GRADS
    // dump the ACCUMULATED weight grads (post GACC-RMW) + the raw activation grads for the gradcheck TB.
    // GACC holds the running sum; at GZERO it == this sub-batch. All DDR->DDR streamed copies (no resident temp).
    s2_copy_ddr(buffer, S2_GACC_WQ,  S2_DG_WQ,  S2_D*S2_D,   0);
    s2_copy_ddr(buffer, S2_GACC_WK,  S2_DG_WK,  S2_D*S2_D,   0);
    s2_copy_ddr(buffer, S2_GACC_WV,  S2_DG_WV,  S2_D*S2_D,   0);
    s2_copy_ddr(buffer, S2_GACC_WO,  S2_DG_WO,  S2_D*S2_D,   0);
    s2_copy_ddr(buffer, S2_GACC_WUP, S2_DG_WUP, S2_FFN*S2_D, 0);
    s2_copy_ddr(buffer, S2_GACC_WDN, S2_DG_WDN, S2_D*S2_FFN, 0);
    if (ROLE == S2_RL_OUT) s2_copy_ddr(buffer, S2_GACC_WH, S2_DG_WH, S2_VOCAB*S2_D, 0);
    s2_copy_ddr(buffer, S2_BG_R, S2_DG_GR, S2_SEQ*S2_D, 0);
    s2_copy_ddr(buffer, S2_BG_Q, S2_DG_GQ, S2_SEQ*S2_D, 0);
    s2_copy_ddr(buffer, S2_BG_K, S2_DG_GK, S2_SEQ*S2_D, 0);
    s2_copy_ddr(buffer, S2_BG_V, S2_DG_GV, S2_SEQ*S2_D, 0);
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

    // (6) emit: INPUT scatters gR -> float EMB/POS GACC (+ adam) ; else the wide (bf16) up-message
    if (ROLE == S2_RL_INPUT) {
        static int32_t tok[S2_SEQ]; bload32(buffer, S2_TOK + pto, tok, S2_SEQ);
        // POS grad accumulates gR per position (each position once) -> GACC_POS = [gR | GACC_POS+gR]
        if (GZERO) s2_copy_ddr(buffer, S2_BG_R, S2_GACC_POS, S2_SEQ*S2_D, 0);
        else       s2_add_ddr(buffer, S2_GACC_POS, S2_BG_R, S2_GACC_POS, S2_SEQ*S2_D, 0);
        // EMB grad scatters gR by token id into GACC_EMB (streamed RMW per token; no resident VOCAB*D)
        if (GZERO) s2_zero_ddr(buffer, S2_GACC_EMB, S2_VOCAB*S2_D);
        static float grrow[S2_D], embrow[S2_D];
        for (int t = 0; t < S2_SEQ; t++) {
            int v = tok[t]; if (v < 0) v = 0; if (v >= S2_VOCAB) v = S2_VOCAB - 1;
            bload_f(buffer, S2_BG_R + t*S2_D, grrow, S2_D);
            bload_f(buffer, S2_GACC_EMB + v*S2_D, embrow, S2_D);
            for (int d = 0; d < S2_D; d++) embrow[d] += grrow[d];
            bstore_f(buffer, S2_GACC_EMB + v*S2_D, embrow, S2_D);
        }
        if (DO_UP) {
            s2_adam_stream(buffer, S2_FEMB, S2_MEMB, S2_VEMB, S2_GACC_EMB, S2_VOCAB*S2_D, LR);
            s2_adam_stream(buffer, S2_FPOS, S2_MPOS, S2_VPOS, S2_GACC_POS, S2_SEQ*S2_D,   LR);
        }
    } else {
        s2_copy_ddr(buffer, S2_BG_R, S2_GROUT + po, S2_SEQ*S2_D, 1);   // wide up-message = P(gR)
    }
}
