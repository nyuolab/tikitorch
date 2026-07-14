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
// PRECISE MULT-FREE exp — the fix for the SILICON hls::exp SOFTMAX BIAS (the last silicon-approximated
// intrinsic). Silicon APPROXIMATES hls::exp; the shim/_mpshim maps it to EXACT libm, so a deterministic-but-
// biased silicon exp was invisible to every csim/twin — the SAME intrinsic-approximation class as the forward
// LN log2 bug (fixed above with s2_ilog2_*). Measured 2026-07: L=2 AND L=8 exact-libm CPU twins TRAIN STABLY
// on the exact recipe (V=8192, held-peak-LR, >150 steps) while F2 silicon DIVERGES to bpb 38 — deconfounding
// depth and isolating the divergence to this one intrinsic. s2_exp_dyad reproduces the golden kernel's proven
// precise exp (forge_pc_xformer_golden_block.hpp g_exp_q16) in FLOAT using this file's integer-bit template:
//   exp(x), x<=0 -> d=-x; u = d*log2(e) via a 6-term shift-add (1+1/4+1/8+1/16+1/256+1/1024, EXACT hls::ldexp,
//   NO multiply -> DSP_MULT=0, err ~0.02% vs true log2(e)); split u into octave(int)+8-bit fraction; then
//   exp = 2^-u = S2_GEXP[frac] >> octave via a compile-time 256-entry mantissa ROM (2^(-f/256), plain float
//   literals). Every op is deterministic int/ldexp/ROM -> BIT-IDENTICAL on F2 silicon and the libm shim.
//   Accuracy vs expf in the softmax-relevant band |x|<8: 0.09-0.18% rel; args < -21.5 floor to 0 (true mass ~0).
static const float S2_GEXP[256] = {
    1.0f, 0.997296035f, 0.994599402f, 0.9919101f, 0.98922801f, 0.986553192f,
    0.983885586f, 0.981225252f, 0.978572071f, 0.975926042f, 0.973287225f, 0.970655501f,
    0.96803087f, 0.965413392f, 0.962802947f, 0.960199594f, 0.957603276f, 0.95501399f,
    0.952431679f, 0.949856341f, 0.947287977f, 0.944726586f, 0.94217211f, 0.939624488f,
    0.93708384f, 0.934549987f, 0.932023048f, 0.929502904f, 0.926989555f, 0.924483061f,
    0.921983302f, 0.919490278f, 0.917004049f, 0.914524496f, 0.912051678f, 0.909585536f,
    0.907126069f, 0.904673278f, 0.902227104f, 0.899787486f, 0.897354543f, 0.894928157f,
    0.892508328f, 0.890094995f, 0.88768822f, 0.885288f, 0.882894218f, 0.880506933f,
    0.878126085f, 0.875751674f, 0.873383701f, 0.871022105f, 0.868666947f, 0.866318107f,
    0.863975644f, 0.8616395f, 0.859309673f, 0.856986105f, 0.854668856f, 0.852357924f,
    0.850053191f, 0.847754657f, 0.845462382f, 0.843176305f, 0.840896428f, 0.838622689f,
    0.83635509f, 0.83409363f, 0.83183831f, 0.829589069f, 0.827345908f, 0.825108767f,
    0.822877765f, 0.820652723f, 0.818433702f, 0.816220701f, 0.81401372f, 0.811812639f,
    0.809617579f, 0.80742842f, 0.805245161f, 0.803067803f, 0.800896406f, 0.798730791f,
    0.796571076f, 0.794417202f, 0.792269111f, 0.79012686f, 0.787990451f, 0.785859764f,
    0.783734798f, 0.781615674f, 0.779502213f, 0.777394474f, 0.775292456f, 0.773196101f,
    0.771105409f, 0.769020379f, 0.766941011f, 0.764867246f, 0.762799084f, 0.760736525f,
    0.758679509f, 0.756628096f, 0.754582226f, 0.75254184f, 0.750507057f, 0.748477697f,
    0.746453881f, 0.744435489f, 0.742422581f, 0.740415096f, 0.738413095f, 0.736416459f,
    0.734425187f, 0.732439399f, 0.730458915f, 0.728483796f, 0.726513982f, 0.724549532f,
    0.722590387f, 0.720636547f, 0.718688011f, 0.716744721f, 0.714806676f, 0.712873876f,
    0.710946321f, 0.709023952f, 0.707106769f, 0.705194831f, 0.703288019f, 0.701386333f,
    0.699489832f, 0.697598457f, 0.695712209f, 0.693831027f, 0.69195497f, 0.690083921f,
    0.688217998f, 0.686357081f, 0.684501231f, 0.682650387f, 0.680804491f, 0.678963661f,
    0.677127779f, 0.675296843f, 0.673470914f, 0.671649873f, 0.669833779f, 0.668022573f,
    0.666216254f, 0.664414883f, 0.662618339f, 0.660826623f, 0.659039795f, 0.657257795f,
    0.655480623f, 0.65370822f, 0.651940644f, 0.650177836f, 0.648419797f, 0.646666467f,
    0.644917965f, 0.643174112f, 0.641435027f, 0.639700592f, 0.637970865f, 0.636245847f,
    0.634525478f, 0.632809758f, 0.631098688f, 0.629392207f, 0.627690375f, 0.625993133f,
    0.62430048f, 0.622612417f, 0.620928884f, 0.61924994f, 0.617575526f, 0.615905643f,
    0.614240289f, 0.612579405f, 0.610922992f, 0.609271109f, 0.607623696f, 0.605980694f,
    0.604342163f, 0.602708042f, 0.601078391f, 0.599453092f, 0.597832203f, 0.596215665f,
    0.594603539f, 0.592995763f, 0.591392338f, 0.589793265f, 0.588198483f, 0.586608052f,
    0.585021913f, 0.583440006f, 0.58186245f, 0.580289125f, 0.578720033f, 0.577155232f,
    0.575594604f, 0.574038267f, 0.572486043f, 0.57093811f, 0.56939429f, 0.567854702f,
    0.566319287f, 0.564787984f, 0.563260794f, 0.561737776f, 0.560218871f, 0.558704078f,
    0.557193398f, 0.555686772f, 0.554184198f, 0.552685738f, 0.55119127f, 0.549700916f,
    0.548214555f, 0.546732187f, 0.545253873f, 0.543779552f, 0.542309165f, 0.540842831f,
    0.539380372f, 0.537921965f, 0.536467433f, 0.535016835f, 0.53357023f, 0.53212744f,
    0.530688584f, 0.529253662f, 0.527822614f, 0.52639538f, 0.524972022f, 0.523552537f,
    0.522136867f, 0.520725071f, 0.519317031f, 0.517912865f, 0.516512454f, 0.515115798f,
    0.513722956f, 0.51233387f, 0.510948598f, 0.509567022f, 0.508189142f, 0.506815016f,
    0.505444646f, 0.504077971f, 0.502714932f, 0.501355648f
};
static inline float s2_exp_dyad(float x) {            // == hls::exp(x) for x<=0, BIT-EXACT on silicon and shim
#pragma HLS INLINE
    if (x >= 0.0f) return 1.0f;                        // s2 softmax always passes x = score - rowmax <= 0
    float d = -x;                                       // d >= 0
    float u = d + hls::ldexp(d,-2) + hls::ldexp(d,-3) + hls::ldexp(d,-4)
                + hls::ldexp(d,-8) + hls::ldexp(d,-10); // d * log2(e), shift-add, mult-free (err ~0.02%)
    int octave = (int)u; if (octave >= 31) return 0.0f; // 2^-u underflow floor (arg < ~-21.5): true mass ~0
    float frac = u - (float)octave;                     // [0,1)
    int fi = (int)(hls::ldexp(frac, 8) + 0.5f);         // round frac*256 (exponent add, no multiply)
    if (fi > 255) fi = 255;
    return hls::ldexp(S2_GEXP[fi], -octave);            // 2^(-fi/256) * 2^(-octave), EXACT ldexp
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
// s2_dotDH — Σ_{d<S2_DH} a[d]*b[d] as a SINGLE combinational DH-wide MAC (DH parallel mults + balanced adder
// tree). Attention is the one allowed-multiply front, so DH=64 mults/cycle here is fine. Purpose: let the
// caller's OUTER (j) loop pipeline at II=1 instead of re-entering an inner reduction pipeline O(N) times.
// The tree association differs from the old 8-way s2_lanesum fold → within-1-ULP (attention stages only).
// FWD score-dot and BWD PRAW score-dot both route through this helper, so PRAW reproduces the fwd softmax
// row BIT-IDENTICALLY (required).
static_assert((S2_DH & (S2_DH-1)) == 0, "S2_DH must be a power of two (s2_dotDH balanced adder tree)");
static inline float s2_dotDH(const float *a, const float *b) {
#pragma HLS INLINE
    float p[S2_DH];
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
    for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
        p[d] = a[d] * b[d];                        // DH parallel mult (attention: allowed-multiply front)
    }
    for (int s = 1; s < S2_DH; s <<= 1) {          // balanced pairwise tree
        for (int d = 0; d < S2_DH; d += (s << 1)) {
#pragma HLS UNROLL
            p[d] = p[d] + p[d + s];
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
static float s2_PALL[S2_SEQ * S2_SEQ];// bwd Phase-1: per-query softmax rows p_i[j] (WRITE-ONCE; Phase-2 reads)
static float s2_GSV [S2_SEQ * S2_SEQ];// bwd Phase-1: per-query score gradients gsv_i[j] (WRITE-ONCE; Phase-2 reads)

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
#if S2_DEPIPELINE_REDUCE
            for (int j = 0; j <= i; j++) {
                float dot = 0.0f;                                        // DE-PIPELINE: sequential score-dot, II>1
                for (int d = 0; d < S2_DH; d++) dot += s2_Qh[i*S2_DH+d] * s2_Kh[j*S2_DH+d];
                sc[j] = dot * scale; if (sc[j] > mx) mx = sc[j];
            }
#else
            // FIX: outer-j pipelines at II=1; the DH reduction is one combinational DH-wide MAC (s2_dotDH).
            // mx stays a cheap loop-carried max.
            for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                float dot = s2_dotDH(&s2_Qh[i*S2_DH], &s2_Kh[j*S2_DH]);
                sc[j] = dot * scale; if (sc[j] > mx) mx = sc[j];
            }
#endif
#if S2_DEPIPELINE_REDUCE
            float esum = 0.0f;                                           // DE-PIPELINE: sequential exp-sum, II>1
            for (int j = 0; j <= i; j++) { float e = s2_exp_dyad(sc[j]-mx); pr[j] = e; esum += e; }
            float sum = esum;
#else
            float slane[S2_LANES];                                       // FIX B: laned exp-sum (was II=6)
#pragma HLS ARRAY_PARTITION variable=slane complete dim=1
            for (int l = 0; l < S2_LANES; l++) slane[l] = 0;
            for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                float e = s2_exp_dyad(sc[j]-mx); pr[j] = e; slane[j & (S2_LANES-1)] += e;
            }
            float sum = s2_lanesum(slane);
#endif
            for (int j = 0; j <= i; j++) pr[j] /= sum;
#if S2_DEPIPELINE_REDUCE
            for (int d = 0; d < S2_DH; d++) {
                float blend = 0.0f;                                      // DE-PIPELINE: sequential P@V blend, II>1
                for (int j = 0; j <= i; j++) blend += pr[j] * s2_Vh[j*S2_DH+d];
                s2_Ah[i*S2_DH+d] = blend;
            }
#else
            // FIX: ONE j-loop with S2_DH independent accumulators (disjoint per d → no cross-lane hazard).
            // The per-(i,d) sum is now ascending-j sequential (was the 8-way lane fold) → within-1-ULP.
            // acc[d] is an fp-add recurrence: do NOT force II=1 (let the tool pick II=fadd-latency); the win
            // is collapsing the DH pipeline RE-ENTRIES into a single j-loop.
            {
                static float acc[S2_DH];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                    acc[d] = 0;
                }
                for (int j = 0; j <= i; j++) {
                    for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                        acc[d] += pr[j] * s2_Vh[j*S2_DH+d];
                    }
                }
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                    s2_Ah[i*S2_DH+d] = acc[d];
                }
            }
#endif
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
        // ============================================================================================
        // ATTENTION BACKWARD — synthesis-robust rewrite (2026-07-11, root-caused via BWD_DUMP silicon-vs-shim).
        // The prior loop computed gp[j] AND the gV cross-iteration RMW (s2_Dh += over the OUTER i) FUSED in one
        // pipelined loop; on F2 that scheduling hazard corrupted gp -> gsv==0 -> gQ/gK read back EXACTLY ZERO
        // and gV wrong (cosine 0.31). Invisible to g++ csim / the twin (no __SYNTHESIS__ guard — an implicit
        // synth != csim divergence), so it masqueraded as a training-instability. FIX: PHASE 1 computes p_i and
        // gsv_i in SEPARATE clean loops (gp is NOT fused with any accumulation) and STORES them write-once
        // (s2_PALL/s2_GSV, no RMW); PHASE 2 forms gQ/gK/gV as fresh-accumulator reductions mirroring the PROVEN
        // forward P@V (acc[d] per output, no cross-iteration RMW on a shared array). Accumulation order is kept
        // identical to the old sweeps, so the result is bit-identical to the reference in exact arithmetic.
        // ---- PHASE 1: per query i, recompute softmax row p_i[j] and score gradient gsv_i[j]; store write-once.
        for (int i = 0; i < S2_SEQ; i++) {
            float p[S2_SEQ], mx = -1e30f;
            for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                float dot = s2_dotDH(&s2_Qh[i*S2_DH], &s2_Kh[j*S2_DH]);   // == fwd score, bit-identical PRAW
                p[j] = dot * ascale; if (p[j] > mx) mx = p[j];
            }
            float slane[S2_LANES];
#pragma HLS ARRAY_PARTITION variable=slane complete dim=1
            for (int l = 0; l < S2_LANES; l++) slane[l] = 0;
            for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                float e = s2_exp_dyad(p[j]-mx); p[j] = e; slane[j & (S2_LANES-1)] += e;
            }
            float sum = s2_lanesum(slane);
            for (int j = 0; j <= i; j++) p[j] /= sum;
            // gsv_i[j] = P( p_i[j] * <gATTN_i, V_j - Vbar_i> ) * ascale,  Vbar_i[d] = sum_{j<=i} p_i[j]*V_j[d].
            // This is the softmax VJP  p_ij*(gp_ij - ex_i)  [gp_ij=<gA_i,V_j>, ex_i=<gA_i,Vbar_i>] rewritten to
            // KILL the fused gp[]-write + exlane-RMW II=1 pipeline that F2 miscompiles: on silicon the scheduler
            // dropped the gp[] BRAM writes so gp read back its zero-init -> ex=0 -> gsv=P(0)=0 EXACTLY -> gQ/gK=0
            // (bit-exact 0, root-caused 2026-07-13 via dumped-operand reconstruction + adversarial verify; the
            // 2026-07-11 two-phase rewrite un-fused gV but left THIS loop fused, so it stayed the live hazard).
            // Vbar is the PROVEN-on-silicon forward P@V blend pattern (DH independent unrolled accumulators, outer
            // loop NOT II=1); gsv is then a combinational per-(i,j) balanced-tree dot of the DEVIATION (no scalar
            // cancellation) stored WRITE-ONCE — no gp[] array, no loop-carried RMW fused with a dot.
            float vbar[S2_DH];
#pragma HLS ARRAY_PARTITION variable=vbar complete dim=1
            for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                vbar[d] = 0;
            }
            for (int j = 0; j <= i; j++) {
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                    vbar[d] += p[j] * s2_Vh[j*S2_DH+d];
                }
            }
            for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
                float prod[S2_DH];                        // gA_i[d] * (V_j[d] - Vbar_i[d]) ; then balanced tree
#pragma HLS ARRAY_PARTITION variable=prod complete dim=1
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                    prod[d] = s2_Ah[i*S2_DH+d] * (s2_Vh[j*S2_DH+d] - vbar[d]);
                }
                for (int s = 1; s < S2_DH; s <<= 1) {     // same association as s2_dotDH (within-1-ULP)
                    for (int d = 0; d < S2_DH; d += (s << 1)) {
#pragma HLS UNROLL
                        prod[d] = prod[d] + prod[d + s];
                    }
                }
                s2_PALL[i*S2_SEQ+j] = p[j];
                s2_GSV[i*S2_SEQ+j]  = P(p[j] * prod[0]) * ascale;
            }
        }
        // DIAGNOSTIC (head 0 only): snapshot the phase-1 scratch to DDR via a CLEAN sequential bstore, BEFORE
        // phase-2 reads it. Read-back ZERO => phase-1 s2_GSV/s2_PALL writes were dropped (write-drop hazard);
        // read-back NONZERO while gQ/gK come back 0 => the phase-2 STRIDED read s2_GSV[i*SEQ+j] is the hazard.
        if (h == 0) {
            bstore_f(buffer, S2_SCR_GSV,  s2_GSV,  S2_SEQ*S2_SEQ);
            bstore_f(buffer, S2_SCR_PALL, s2_PALL, S2_SEQ*S2_SEQ);
        }
        // ---- PHASE 2: fresh-accumulator reductions (== forward P@V pattern; NO cross-iteration RMW). ----
        // gV[j][d] = sum_{i>=j} P(p_i[j]*gATTN_i[d]) ; gK[j][d] = sum_{i>=j} gsv_i[j]*Q_i[d]  (ascending-i order).
        for (int j = 0; j < S2_SEQ; j++) {
            float accV[S2_DH], accK[S2_DH];
#pragma HLS ARRAY_PARTITION variable=accV complete dim=1
#pragma HLS ARRAY_PARTITION variable=accK complete dim=1
            for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                accV[d] = 0; accK[d] = 0;
            }
            for (int i = j; i < S2_SEQ; i++) {
                float pij = s2_PALL[i*S2_SEQ+j], gij = s2_GSV[i*S2_SEQ+j];
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                    // accV: RAW accumulate then P once post-loop (line ~1000), IDENTICAL to accK and the proven
                    // forward P@V blend. The old per-term P(pij*gA) fused a bf16-round INTO the loop-carried RMW,
                    // which F2 miscompiled -> gV double-counted (~2x) / dropped (~58% exact-zero, rising with the
                    // i>=j depth). Removing the in-loop P makes accV structurally the proven accumulator.
                    accV[d] += pij * s2_Ah[i*S2_DH+d];
                    accK[d] += gij * s2_Qh[i*S2_DH+d];
                }
            }
            for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                s2_Dh[j*S2_DH+d] = accV[d]; s2_Ch[j*S2_DH+d] = accK[d];
            }
        }
        // gQ[i][d] = sum_{j<=i} gsv_i[j]*K_j[d]  (ascending-j order).
        for (int i = 0; i < S2_SEQ; i++) {
            float accQ[S2_DH];
#pragma HLS ARRAY_PARTITION variable=accQ complete dim=1
            for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                accQ[d] = 0;
            }
            for (int j = 0; j <= i; j++) {
                float gij = s2_GSV[i*S2_SEQ+j];
                for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                    accQ[d] += gij * s2_Kh[j*S2_DH+d];
                }
            }
            for (int d = 0; d < S2_DH; d++) {
#pragma HLS UNROLL
                s2_Bh[i*S2_DH+d] = accQ[d];
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
            for (int v = 0; v < S2_VOCAB; v++) { pr[v] = s2_exp_dyad(lgrow[v]-mx); sum += pr[v]; }
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
