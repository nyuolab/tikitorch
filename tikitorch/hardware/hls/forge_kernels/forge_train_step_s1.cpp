// forge_train_step_s1.cpp — the S1 integrated FORWARD training-step kernel: ONE transformer block
// (embed -> [LN1 -> attn -> res -> LN2 -> FFN -> res] -> tied max-plus LM-head -> balanced-LUT
// cross-entropy loss + grad-at-logits) for the end-to-end on-F2 transformer trainer.
//
// This kernel COMPOSES the four verified, csynth-closing kernels by INLINING their EXACT dyadic math
// as static helper functions (HLS cannot call other top-level kernels, so their loop bodies are
// reproduced verbatim, only re-parameterized to the S1 dims in forge_s1_config.hpp):
//   - forge_fwd_wide.cpp : dyad() quantizer + wdecode() + the dyadic-MAC GEMM y=q(x)·q(W)  (DSP=0)
//                          -> Q/K/V/O projections, FFN up/down, all reuse fwd_gemm()
//   - forge_attn.cpp     : L1 score + exp2-LUT softmax + soft p*V blend (the ONE multiply -> DSP)
//   - forge_layernorm.cpp: fine-dyadic LN (mean=sum>>7, meanabs rescale mbits=2, dyadic gamma, beta)
//   - forge_relu.cpp     : y=max(x,0), mask saved  (DSP=0)
//   - forge_train_step_morph_lm.cpp : max-plus LM-head (logits[v]=max_d(W_HEAD[v][d]+R[d]), argmax d*)
//                          + balanced LUT cross-entropy (grad_logit[v]=p[v]-(v==tgt?sum_p:0))
//                          + embed gather (R[d]=EMBED[tok*D+d])
//
// MULTIPLIER-FREE except the single attention soft p*V blend (forge_attn's one irreducible multiply,
// kept as a real multiply -> DSP). Everything else is sub/abs/add-tree/compare/shift only -> DSP=0.
//
// dyad() RULE is forge_fwd_wide.cpp dyad() VERBATIM (priority-encode MSB; bump exponent if the bit
// below MSB is set; sign separate; zero-guard). The CPU golden in the testbench uses the IDENTICAL
// rule, so the integrated golden IS the deploy-faithful reference (self-consistent + authoritative).
//
// ABI mirrors forge_train_step_morph_lm.cpp: extern "C" int forge_train_step_s1(buffer, mode, a1,a2,a3).
//   mode 0 -> do_fwd_loss (the whole forward + CE, this task). modes 1/2/3 stubbed (return 0).
//
// TIMING ARCHITECTURE — RESIDENT ON-CHIP STAGING (the forge_train_step_morph_lm TT_RESIDENT_W pattern):
// the v1 of this kernel made DIRECT m_axi (gmem) reads INSIDE the compute loops — the data-dependent
// embed gather (R=buffer[OFF_EMBED+tok*D+d]) and per-element weight/LUT reads in the GEMM / LN / head —
// which put the long AXI read path on the critical path (5.840 ns > 4.0 ns target) and stalled II=1
// (limited AXI ports + carried bus deps). The standalone kernels (forge_fwd_wide etc.) close at 2.920 ns
// because they BURST-LOAD into on-chip arrays and compute locally. So here:
//   (1) At kernel entry, BURST-LOAD every weight/LUT/token region from gmem ONCE into static on-chip
//       arrays (sequential II=1 pipelined staging loops -> HLS infers wide bursts). Dyadic byte-codes are
//       kept in int8 arrays, int16 tables (EMBED/W_HEAD/beta) in int16 arrays -> minimal BRAM; the
//       reconstruction (& 0xFF / (int16_t)(.&0xFFFF)) is BIT-IDENTICAL to the v1 gmem read, so the
//       golden is unchanged.
//   (2) Compute the ENTIRE forward from the on-chip arrays. NO gmem access in any compute loop: the embed
//       gather hits the on-chip EMBED array, every GEMM reads on-chip W, LN/attn/head/CE all on-chip.
//   (3) Keep activations (R, X1, X2, attn p/jstar, relu mask, ln stats, argmax_head, logits, grad_logit)
//       in on-chip arrays during compute, then BURST-STORE the buffer-exposed regions at the END
//       (sequential II=1 pipelined store loops -> wide bursts).
//   (4) ARRAY_PARTITION the GEMM operand banks (dim=1 complete, the proven forge_fwd_wide recipe) so the
//       inner MACs/reductions pipeline II=1.
// The math / loop bodies are otherwise VERBATIM from v1: the CPU golden in the testbench still reports
// max|kernel-ref|==0 over all regions. Staging is purely internal; the buffer ABI + offset map UNCHANGED.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csim+csynth via synth_train_step_s1.tcl.

#include <stdint.h>
#include "forge_s1_config.hpp"

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_int<16>   data_t;    // activation values (int16, the morph data_t range)
typedef ap_int<6>    exp_t;     // dyadic exponent: holds W-code 0..31 + dyad() round-bump 16 (6 bits)
typedef ap_int<48>   acc_t;     // accumulators (DSP48-width signed)
typedef ap_int<32>   prob_t;    // Q16 attention probability
typedef ap_uint<512> wide_t;    // 512-bit wide bus word (16 int32 lanes) for burst staging
#else
#include <cstdint>
typedef int16_t      data_t;
typedef int32_t      exp_t;
typedef int64_t      acc_t;     // int64 outside Vitis so the g++ standalone path matches the golden
typedef int32_t      prob_t;
#endif

// on-chip resident weight encodings: byte-codes in int8 (wdecode masks &0xFF), int16 tables in int16,
// LUTs / tokens / targets in int32 — minimal BRAM, reconstruction BIT-IDENTICAL to the v1 gmem read.
#if S1_KV2
typedef int32_t  code_t;        // kernel-v2: int32 holds 4 packed byte-codes (4-term weight CSD)
#else
typedef int8_t   code_t;        // pre-dyadic weight byte code (sign|nz|exp, 8 bits)
#endif
typedef int16_t  half_t;        // int16 table value (EMBED / W_HEAD / LN beta)

static const acc_t S1_NEG_INF = (acc_t)(-((acc_t)1 << 40));   // attn causal-mask sentinel
static const acc_t ML_NEG_INF = (acc_t)(-2147483647LL);       // max-plus head -inf (morph value)

// =============================================================================
// burst staging helpers (forge_train_step_morph_lm burst_load_words pattern). Move a gmem region into /
// out of an on-chip array ONCE per kernel call via 512-bit wide pipelined bursts (II=1). The decode to
// the resident type happens HERE (at entry), NOT in the compute loops -> no AXI on the critical path.
// All staged regions are 16-word (512-bit) aligned in the offset map (S1_AL16), so the wide path covers
// them exactly; lengths rounded down to a wide multiple, with a scalar tail for any remainder.
// =============================================================================

// load n_words int32 from buffer[word_off..] into an int32 on-chip array.
static void burst_load_i32(int32_t *buffer, int word_off, int32_t *dst, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t *wide_buf = reinterpret_cast<wide_t *>(buffer);
    int n_wide = n_words >> 4;
    int wide_off = word_off >> 4;
LL_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
        wide_t v = wide_buf[wide_off + w];
LL_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int hi = lane * 32 + 31, lo = lane * 32;
            ap_int<32> x = (ap_int<32>)v.range(hi, lo);
            dst[w * 16 + lane] = (int32_t)(int)x;
        }
    }
    for (int i = (n_wide << 4); i < n_words; i++) dst[i] = buffer[word_off + i];
#else
    for (int i = 0; i < n_words; i++) dst[i] = buffer[word_off + i];
#endif
}

// load n_words int32 codes from buffer and narrow each to its low byte into an int8 on-chip array.
// (& 0xFF, sign-stored in int8: wdecode((int32_t)int8) re-masks &0xFF -> bit-identical to v1 buffer read.)
static void burst_load_codes(int32_t *buffer, int word_off, code_t *dst, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t *wide_buf = reinterpret_cast<wide_t *>(buffer);
    int n_wide = n_words >> 4;
    int wide_off = word_off >> 4;
LC_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
        wide_t v = wide_buf[wide_off + w];
LC_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int lo = lane * 32;
#if S1_KV2
            dst[w * 16 + lane] = (code_t)(int32_t)v.range(lo + 31, lo);   // full int32 (4 packed CSD byte-codes)
#else
            ap_uint<8> b = (ap_uint<8>)v.range(lo + 7, lo);
            dst[w * 16 + lane] = (code_t)(int8_t)(unsigned char)b;
#endif
        }
    }
#if S1_KV2
    for (int i = (n_wide << 4); i < n_words; i++) dst[i] = (code_t)buffer[word_off + i];
#else
    for (int i = (n_wide << 4); i < n_words; i++) dst[i] = (code_t)(int8_t)(buffer[word_off + i] & 0xFF);
#endif
#else
#if S1_KV2
    for (int i = 0; i < n_words; i++) dst[i] = (code_t)buffer[word_off + i];
#else
    for (int i = 0; i < n_words; i++) dst[i] = (code_t)(int8_t)(buffer[word_off + i] & 0xFF);
#endif
#endif
}

// load n_words int32 values from buffer and narrow each to int16 into an int16 on-chip array.
// ((int16_t)(. & 0xFFFF) -> bit-identical to the v1 buffer read for EMBED / W_HEAD / LN beta.)
static void burst_load_half(int32_t *buffer, int word_off, half_t *dst, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t *wide_buf = reinterpret_cast<wide_t *>(buffer);
    int n_wide = n_words >> 4;
    int wide_off = word_off >> 4;
LH_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
        wide_t v = wide_buf[wide_off + w];
LH_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int lo = lane * 32;
            ap_uint<16> h = (ap_uint<16>)v.range(lo + 15, lo);
            dst[w * 16 + lane] = (half_t)(int16_t)(unsigned short)h;
        }
    }
    for (int i = (n_wide << 4); i < n_words; i++) dst[i] = (half_t)(int16_t)(buffer[word_off + i] & 0xFFFF);
#else
    for (int i = 0; i < n_words; i++) dst[i] = (half_t)(int16_t)(buffer[word_off + i] & 0xFFFF);
#endif
}

// store n_words int32 from an on-chip array to buffer[word_off..] via wide bursts.
static void burst_store_i32(int32_t *src, int32_t *buffer, int word_off, int n_words) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    wide_t *wide_buf = reinterpret_cast<wide_t *>(buffer);
    int n_wide = n_words >> 4;
    int wide_off = word_off >> 4;
SS_W: for (int w = 0; w < n_wide; w++) {
#pragma HLS PIPELINE II=1
        wide_t word = 0;
SS_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int hi = lane * 32 + 31, lo = lane * 32;
            int32_t x = src[w * 16 + lane];
            word.range(hi, lo) = (ap_uint<32>)(ap_int<32>)x;
        }
        wide_buf[wide_off + w] = word;
    }
    for (int i = (n_wide << 4); i < n_words; i++) buffer[word_off + i] = src[i];
#else
    for (int i = 0; i < n_words; i++) buffer[word_off + i] = src[i];
#endif
}

// =============================================================================
// dyadic primitives — VERBATIM from forge_fwd_wide.cpp / forge_layernorm.cpp.
// =============================================================================

// dyadic-quantize a FRESH activation to (sign, exponent): round(log2|v|) via MSB + next bit.
static inline void dyad(data_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    ap_uint<16> a = (v < 0) ? (ap_uint<16>)(-v) : (ap_uint<16>)v;
#else
    uint16_t a = (v < 0) ? (uint16_t)(-v) : (uint16_t)v;
#endif
    sgn = (v < 0);
    nz = (a != 0);
    exp_t msb = 0;
PRIO: for (int b = 15; b >= 1; b--) {
#pragma HLS UNROLL
        if ((a >> b) & 1) { msb = (exp_t)b; break; }
    }
    if (msb >= 1 && ((a >> ((int)msb - 1)) & 1)) msb = msb + 1;   // bump on bit below MSB
    e = msb;
}

// dyadic-quantize a FULL-PRECISION (int32-range) gradient to (sign, exponent). Same round-to-nearest
// rule as dyad(), but scans the WHOLE 31-bit magnitude (NOT truncated to int16). The S1 backward's
// upstream grads are int32 (the Q16 CE-grad scale alone is ~2^16, amplified by the GEMM chain), so the
// wgrad gO operand MUST be quantized from its full magnitude — charlm does dyadic_quantize(gf) on the
// full-precision gradient, NOT on an int16-clamped one. (The standalone forge_wgrad_df3 int16-limits gO
// because its CSIM TB stays in |g|<600; the integrated trainer at Q16 grad scale needs this wide form.)
// exp can reach ~30 (+1 round-bump => 31); exp_t (ap_int<6>, -32..31) holds it. The x operand stays
// int16 (forward activations ARE sat16'd), so only the gO side widens.
static inline void dyad_wide(acc_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    ap_uint<32> a = (v < 0) ? (ap_uint<32>)(acc_t)(-v) : (ap_uint<32>)(acc_t)v;
#else
    uint32_t a = (v < 0) ? (uint32_t)(-v) : (uint32_t)v;
#endif
    sgn = (v < 0);
    nz = (a != 0);
    exp_t msb = 0;
PRIOW: for (int b = 30; b >= 1; b--) {
#pragma HLS UNROLL
        if ((a >> b) & 1) { msb = (exp_t)b; break; }
    }
    if (msb >= 1 && ((a >> ((int)msb - 1)) & 1)) msb = msb + 1;   // bump on bit below MSB
    e = msb;
}

// decode a PRE-DYADIC weight byte code: bit7=sign, bit5=nonzero, bits4..0=exponent.
static inline void wdecode(int32_t code, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    uint32_t c = (uint32_t)(code & 0xFF);
    sgn = ((c >> 7) & 1) != 0;
    nz  = ((c >> 5) & 1) != 0;
    e   = (exp_t)(c & 0x1F);
}

#if S1_KV2
// kernel-v2 weight MAC: the int32 code packs S1_WCSD_K byte-codes (one CSD term each, same wdecode
// format). Returns sum_k (+/- xv<<e_k) — the weight*activation as K shift-adds (DSP=0). The per-term
// exponents are non-negative (weights at SW=64 are >=1 in scaled-int); the GEMM rescales >> later.
static inline acc_t wmac4(int32_t code, acc_t xv) {
#pragma HLS INLINE
    acc_t s = 0;
WCSD: for (int k = 0; k < S1_WCSD_K; k++) {
#pragma HLS UNROLL
        uint32_t c = (uint32_t)((code >> (8 * k)) & 0xFF);
        if ((c >> 5) & 1) {                                   // nz
            acc_t term = xv << (int)(c & 0x1F);               // xv << exp  (exp >= 0)
            s += ((c >> 7) & 1) ? (acc_t)(-term) : term;      // apply term sign
        }
    }
    return s;
}

// kernel-v2 LayerNorm reciprocal: 1/meanabs as S1_RECIP_K signed powers of 2 (CSD), returned as
// exponents e_out[k] (negative; r = sum_k c<<e_k via >> ). recip computed in Q-RFRAC fixed point
// (a divide — restoring shift/subtract, NO multiply), then the top-K set bits are the CSD terms.
#define S1_RFRAC 24
static inline void recip_csd(acc_t meanabs, int e_out[S1_RECIP_K]) {
#pragma HLS INLINE
    if (meanabs < 1) meanabs = 1;
    acc_t recip = ((acc_t)1 << S1_RFRAC) / meanabs;           // Q24 reciprocal of meanabs (<= 2^24)
RCSD: for (int k = 0; k < S1_RECIP_K; k++) {
#pragma HLS UNROLL
        int p = -1;
    RTOP: for (int b = S1_RFRAC + 1; b >= 0; b--) {           // leading set bit of the residual
#pragma HLS UNROLL
            if ((recip >> b) & 1) { p = b; break; }
        }
        if (p < 0) { e_out[k] = -64; }                        // residual 0 -> harmless huge negative shift
        else { e_out[k] = p - S1_RFRAC; recip -= ((acc_t)1 << p); }   // subtract this power; e relative to Q24
    }
}
#endif

static inline data_t sat16(acc_t v) {
#pragma HLS INLINE
    if (v >  32767) return  (data_t)32767;
    if (v < -32768) return (data_t)(-32768);
    return (data_t)v;
}

// fine-dyadic 1/scale -> (e, nq): VERBATIM forge_layernorm.cpp fdyad() (mbits=2; NO divide, NO mult).
static inline void fdyad(acc_t scale, int &e_out, int &nq_out) {
#pragma HLS INLINE
    if (scale < 1) scale = 1;
    int p = 0;
PRIO2: for (int b = 47; b >= 1; b--) {
#pragma HLS UNROLL
        if ((scale >> b) & 1) { p = b; break; }
    }
    bool exact_pow2 = (scale == ((acc_t)1 << p));
    int e = exact_pow2 ? (-p) : (-(p + 1));
    int nege = -e;
    acc_t numer  = (acc_t)1 << (S1_MBITS + nege);
    acc_t thresh = numer + (scale >> 1);
    int   nbase = (1 << S1_MBITS);                 // 4
    int   nq    = nbase;
    acc_t acc   = scale << S1_MBITS;               // 4*scale
CAND: for (int step = 1; step <= (1 << S1_MBITS); step++) {
#pragma HLS UNROLL
        acc += scale;
        if (acc <= thresh) nq = nbase + step;
    }
    if (nq >= (1 << (S1_MBITS + 1))) { nq = (1 << S1_MBITS); e += 1; }
    if (nq < (1 << S1_MBITS))         nq = (1 << S1_MBITS);
    e_out = e;
    nq_out = nq;
}

// =============================================================================
// MAX operand dims — one physical engine per GEMM class is sized to the LARGEST shape any call site uses
// (T<=L_SEQ=128 rows, I<=FFN=512 contraction, O<=FFN=512 outputs). The de-templatized runtime-dim
// functions below size their on-chip operand banks to these maxima and bound their loops by the runtime
// args, so a smaller call (e.g. D x D) simply iterates fewer times on the SAME hardware. With ALLOCATION
// function instances=...,limit=1 in do_fwd_loss/do_bwd, HLS builds ONE engine of each class and
// time-multiplexes every call site (trading latency for area — the S1 area fix).
// =============================================================================
#define S1_GEMM_TMAX  S1_L_SEQ    // 128
#define S1_GEMM_IMAX  S1_FFN      // 512
#define S1_GEMM_OMAX  S1_FFN      // 512

// =============================================================================
// AREA FIX — NARROW LANE WIDTH. The engines inherited the throughput-optimized wide tiling from the
// standalone kernels: complete dim=1 partition of the [TMAX][*] operand banks + a TMAX(=128)-way inner
// reduction UNROLL. That makes each shared engine huge (~128 parallel MAC lanes) -> 1.35M LUT = 103% of
// the whole device. S1 is a CORRECTNESS / CONVERGENCE proof, NOT a throughput kernel (the tok/s story is
// the standalone optimized kernels + projection, already done), so we narrow the parallelism to a SMALL
// lane width: partition the T (token) dim CYCLIC factor=LANE, UNROLL the inner T-reduction factor=LANE,
// and accumulate over T in LANE-wide chunks. Latency grows ~TMAX/LANE x (irrelevant for S1); area drops
// ~TMAX/LANE x. BIT-EXACT-PRESERVING: an integer reduction is order-independent (no overflow — int48/int32
// headroom already verified < 2^31), so a LANE-wide chunked accumulation produces the IDENTICAL total as
// the full 128-way unroll — only the SCHEDULE changes, never the result. PIPELINE II=1 still holds because
// the cyclic-by-LANE partition gives LANE conflict-free reads/cycle for the LANE-wide inner unroll.
// LANE=16 (at most 32). S1_GEMM_TMAX MUST be a multiple of LANE (128 % 16 == 0) so the cyclic banks and
// the chunked loop tile evenly.
// =============================================================================
#ifndef S1_LANE
#define S1_LANE 4     // was 16 -> 8 -> 4. LUT is set by the parallel-lane count (NOT the model dims), so cutting
                      // lanes is what shrinks compute LUT. v10 (LANE=8) post-route MISSED at WNS -0.409: the worst
                      // path is the shell sh_ddr(SLR1) async-FIFO wready -> XBAR_DDR crossbar -> high-fanout(577)
                      // net crossing to the DMA-PCIS side in SLR0. That SLR0/SLR1-boundary crossbar<->DMA-PCIS path
                      // is structural (DMA-PCIS is shell-fixed in SLR0, DDR in SLR1) and CLOSED in the proven
                      // muzero_ddr build with no help -- S1 misses only because its bigger trainer kernel + clkgen +
                      // 2 CDCs CROWD the boundary region, degrading that net's placement/replication. LANE=4 halves
                      // the dominant logic consumer to de-crowd the boundary (restore muzero-like routing). LANE=4
                      // divides D=64/DH=32/FFN=256/seq=64/V=64 and S1_GEMM_TMAX=64/IMAX=OMAX=256 (all %4==0).
                      // Bit-exact-preserving (integer reduction is order-independent); ~2x more cycles than LANE=8
                      // (irrelevant for the single-shot S1 proof). Re-csim-verified bit-exact before the AFI build.
#endif

// =============================================================================
// OPT AREA FIX — the optimizer (mode 2 do_opt) is a per-element streaming RMW (dyadic-Adam EMA + pow2
// step + re-quantize + weight decay), NOT a throughput GEMM. The old version inherited the GEMM's wide
// S1_LANE(=16) UNROLL on the RMW loop, which REPLICATED the whole per-element datapath — and crucially the
// s1_dyad_exp / dyad / s1_encode_fine PRIORITY ENCODERS (32-/16-deep) — 16x per region. That is why do_opt
// was 515k LUT (16x the encoder trees). The fix: a SEPARATE, SMALL optimizer lane width S1_OPT_LANE
// (default 1 = ONE shared dyadic-Adam-element datapath, II=1) so the heavy priority-encoders are
// instantiated ONCE. Latency grows ~S1_LANE/S1_OPT_LANE x (irrelevant for the S1 correctness/convergence
// proof — the throughput story is the standalone optimized kernels). BIT-EXACT-PRESERVING: the optimizer
// is a per-element map (no contraction, no cross-element carry), so any lane width produces the IDENTICAL
// result — only the SCHEDULE changes. The #else (g++) reference path is a plain scalar loop already, so
// the int64 golden is untouched. S1_OPT_LANE must divide nothing (the p<n guard handles any tail).
// =============================================================================
#ifndef S1_OPT_LANE
#define S1_OPT_LANE 1
#endif

// =============================================================================
// OPT WRITEBACK FIX — FUSED WIDE READ+WRITE in do_opt (dyadic_adam_region). HISTORY: the optimizer first
// streamed the RMW directly on the m_axi `buffer` (per-scalar variable-offset access -> a 512-bit AXI read
// PLUS a VARIABLE 512-bit barrel shift -> the 4.32 ns timing pole). The next version fixed TIMING with
// BOUNDED-TILE on-chip staging (burst-load g/m/v/W_fp into static tile arrays, compute with fixed indexing,
// burst-STORE back) — but those load->static->store ROUND-TRIPS on the SAME gmem offsets that do_fwd_loss
// reads back through a DIFFERENT limit=1 cast were a writeback the Vitis RTL scheduler could NOT prove live:
// it DROPPED the stores on silicon (csim+csynth pass; do_opt no-ops; the live regions stay ZERO; mode-3
// training is FLAT). THE FIX (this version): mirror the morph signsgd_region_wide pattern that COMMITS on
// the identical F2 shell — a SINGLE FUSED wide read+write per 512-bit word (read wide_buf[wi], compute 16
// lanes with FIXED per-lane .range() shifts, write wide_buf[wi] in the SAME body). NO static staging arrays,
// NO separate dead store; do_opt OWNS its writes (the nested ALLOCATION limit=1 share is also removed). The
// timing argument still holds: the per-lane shifts are FIXED (lane*32) constants — no variable barrel shift —
// and the per-word work is bounded (16 lanes over one wide word, S1_OPT_LANE-class small), so the fused
// datapath is the morph datapath that closed at 2.920 ns. S1_OPT_TILE survives ONLY for zero_grad_region's
// frozen-code grad-zero tile (a separate, already-committing store; must be a multiple of 16).
// =============================================================================
#ifndef S1_OPT_TILE
#define S1_OPT_TILE 4096
#endif

// =============================================================================
// DDR-STREAMED ACTIVATION-SAVE TILE granularities (the BRAM-relief rearchitecture). The big activation
// SAVES (relu mask / R snapshots / X1 / X2 / attn probs / head trio) are no longer held RESIDENT for the
// whole block — they are streamed per-tile to/from their S1_OFF_* DDR regions (fwd burst-store the tile as
// it is computed; bwd burst-load the tile right before the consuming op), following the do_opt template.
// Each tile is one op's natural slice; every tile BASE = region_base (S1_AL16-aligned) + k*STRIDE where
// STRIDE is one of these — all multiples of 16 — so every wide burst stays 512-bit aligned (the
// OFF_CE_LUT-class corruption guard). The compute LOOP BODY is unchanged: only the destination of the
// already-computed value changes from a resident array slot to a per-tile burst (bit-exact: a pure write /
// pure reload, identical offsets + arithmetic; the int64 #else golden sees the same buffer contents because
// burst_load_i32/burst_store_i32 reduce to a plain scalar copy there).
#define S1_TILE_RELU   S1_FFN      // relu mask:  per-token row  [FFN]
#define S1_TILE_RD     S1_D        // R snapshot / X1 / X2: per-token row [D]
#define S1_TILE_ATTNP  S1_L_SEQ    // attn probs: per-query row  [L_SEQ]
#define S1_TILE_V      S1_V        // head trio (argmax/grad-logit/logits): per-position row [V]
static_assert(S1_TILE_RELU  % 16 == 0, "relu tile stride must be 16-word aligned");
static_assert(S1_TILE_RD    % 16 == 0, "R/X1/X2 tile stride must be 16-word aligned");
static_assert(S1_TILE_ATTNP % 16 == 0, "attn-p tile stride must be 16-word aligned");
static_assert(S1_TILE_V     % 16 == 0, "head tile stride must be 16-word aligned");

// =============================================================================
// fwd_gemm — the dyadic-MAC GEMM y[t][o] = sum_i q(x[t][i]) MAC q(W[o][i]) + bias(=0 in S1).
// Reproduces forge_fwd_wide.cpp's inner: x dyad-quantized once, W PRE-DYADIC (wdecode inline), the
// MAC is sign-xor + exponent-ADD + barrel-SHIFT + signed accumulate (NO multiply, DSP=0).
// RUNTIME-DIM (M rows, I contraction, O outputs) — ONE physical engine, banks sized to the MAX dims,
// loops bounded by the runtime args. ON-CHIP-resident W byte-codes ROW-MAJOR [O][I] (wcode[w_off+o*I+i],
// a code_t int8 staged at entry; NO gmem in the inner loop -> II=1, no AXI on the critical path). x is
// held FLAT on-chip ROW-MAJOR [M][I] (xin[t*I + i]); y written FLAT ROW-MAJOR [M][O] (yout[t*O + o]).
// The math/order is VERBATIM the templated version (just runtime-bounded) -> the int64 golden is unchanged.
// =============================================================================
static void fwd_gemm(const data_t *xin, const code_t *wcode, int w_off, data_t *yout,
                     int M, int I, int O, int rshift = S1_GEMM_RSHIFT) {
#pragma HLS INLINE off
    (void)rshift;
    // dyad-quantize x once (q(x) = ±2^ex).
    // Partition the T (row/token) dim COMPLETE so the IRED inner loop's T-way UNROLL gets T parallel
    // reads of xE/xS/xNZ[*][i] per cycle (the forge_fwd_wide dim=1 partition recipe). Without this the
    // T-way-unrolled MAC can't hit II=1 on a single BRAM port (II would balloon -> schedule blowup).
#if S1_GEMM_QX
    static exp_t xE[S1_GEMM_TMAX][S1_GEMM_IMAX]; static bool xS[S1_GEMM_TMAX][S1_GEMM_IMAX]; static bool xNZ[S1_GEMM_TMAX][S1_GEMM_IMAX];
#pragma HLS ARRAY_PARTITION variable=xE  cyclic factor=S1_LANE dim=1
#pragma HLS ARRAY_PARTITION variable=xS  cyclic factor=S1_LANE dim=1
#pragma HLS ARRAY_PARTITION variable=xNZ cyclic factor=S1_LANE dim=1
QX: for (int t = 0; t < M; t++)
        for (int i = 0; i < I; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=128 max=512
            dyad(xin[t * I + i], xS[t][i], xE[t][i], xNZ[t][i]);
        }
#endif
    // output-stationary: for each output o, stream the i-reduction over all T rows. The T (token) reduction
    // is NARROWED to LANE-wide chunks: cyclic-by-LANE banks give LANE conflict-free reads/cycle so the inner
    // LANE-way UNROLL pipelines II=1; the outer tt chunk loop walks the FULL T range in TMAX/LANE steps.
    // BIT-EXACT: integer adds, same total as the old full-T unroll (order-independent, no overflow).
GO: for (int o = 0; o < O; o++) {
        acc_t acc[S1_GEMM_TMAX];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=S1_LANE
        for (int t = 0; t < M; t++) {
#pragma HLS UNROLL factor=S1_LANE
            acc[t] = 0;
        }
IRED:   for (int i = 0; i < I; i++) {
            bool sw, nzw; exp_t ew;
            wdecode((int32_t)wcode[w_off + o * I + i], sw, ew, nzw);
TT:         for (int tt = 0; tt < S1_GEMM_TMAX; tt += S1_LANE) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=128 max=512
                for (int tl = 0; tl < S1_LANE; tl++) {
#pragma HLS UNROLL
                    int t = tt + tl;
                    if (t < M) {
#if S1_KV2
                        // kernel-v2: 4-term CSD weight (int32 code) MAC'd with the full int16 activation.
                        acc[t] += wmac4((int32_t)wcode[w_off + o * I + i], (acc_t)xin[t * I + i]);
#elif S1_GEMM_QX
                        exp_t ex = xE[t][i];
                        int e = (int)ew + (int)ex;
                        acc_t term = (acc_t)1 << e;
                        bool neg = sw ^ xS[t][i];
                        bool nz  = nzw & xNZ[t][i];
                        if (nz) acc[t] += neg ? (acc_t)(-term) : term;
#else
                        // q(W)*x_full: keep the FULL int16 activation, shift by the weight's exponent (DSP=0).
                        acc_t xv = (acc_t)xin[t * I + i];
                        acc_t term = (ew >= 0) ? (xv << (int)ew) : (xv >> (int)(-ew));
                        if (nzw) acc[t] += sw ? (acc_t)(-term) : term;
#endif
                    }
                }
            }
        }
        for (int t = 0; t < M; t++) {
#pragma HLS UNROLL factor=S1_LANE
#if S1_KV2
            yout[t * O + o] = sat16(acc[t] >> rshift);   // arithmetic >> = floor toward -inf (matches sim afloor)
#else
            yout[t * O + o] = sat16(acc[t]);   // cast to int16 activation range (LN/attn consume int16)
#endif
        }
    }
}

// =============================================================================
// layernorm_tok — fine-dyadic LayerNorm over one token row x[D] -> y[D], saving (e,nq).
// VERBATIM math from forge_layernorm.cpp (mean=sum>>LOG2_D, meanabs rescale, dyadic gamma, beta add).
// gamma (int8 byte-codes) + beta (int16) read from ON-CHIP arrays staged at entry -> no gmem in RESC.
// =============================================================================
static void layernorm_tok(const data_t x[S1_D], const code_t *gamma, int gamma_off,
                          const half_t *beta, int beta_off,
                          data_t y[S1_D], int &e_out, int &nq_out) {
#pragma HLS INLINE off
    acc_t sum = 0;
SUMX: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
        sum += (acc_t)x[d];
    }
    acc_t mean = sum >> S1_LOG2_D;

    static data_t c[S1_D];
    acc_t asum = 0;
CEN: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t cv = (acc_t)x[d] - mean;
        c[d] = sat16(cv);
        acc_t a = (cv < 0) ? (acc_t)(-cv) : (acc_t)cv;
        asum += a;
    }
    acc_t meanabs = asum >> S1_LOG2_D;

#if S1_KV2
    // kernel-v2: y = c * (1/meanabs as K'-term CSD) * (gamma as K-term CSD) + beta. The double-CSD product
    // is the K'*K cross-terms of c shifted by (recip_exp[k] + gamma_exp[j]) — all shift-adds, DSP=0. The
    // recip exps are negative, the gamma exps positive (gamma~SGAM>=1), so the shifts land near 0 (no
    // fractional-precision loss). Matches scripts/tropical_llm_kernelsim.py kernel_ln (LN_RECIP_K path).
    int erec[S1_RECIP_K];
    recip_csd(meanabs, erec);
    e_out = erec[0]; nq_out = erec[1];
RESC: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t cv = (acc_t)c[d];
        int32_t gcode = (int32_t)gamma[gamma_off + d];
        acc_t hi = 0;                                          // c*recip*gamma at +S1_LN_GUARD fractional bits
    GK: for (int k = 0; k < S1_RECIP_K; k++) {
#pragma HLS UNROLL
        GJ: for (int j = 0; j < S1_WCSD_K; j++) {
#pragma HLS UNROLL
                uint32_t gc = (uint32_t)((gcode >> (8 * j)) & 0xFF);
                if ((gc >> 5) & 1) {                           // gamma CSD term nonzero
                    int sh = erec[k] + (int)(gc & 0x1F) + S1_LN_GUARD;   // recip(neg)+gamma(pos)+guard
                    acc_t term = (sh >= 0) ? (cv << sh) : (cv >> (-sh));
                    hi += ((gc >> 7) & 1) ? (acc_t)(-term) : term;       // gamma term sign (recip all +)
                }
            }
        }
        // round the high-precision product once (round-half-up), then add beta (int16, SGAM scale).
        acc_t r = (hi + ((acc_t)1 << (S1_LN_GUARD - 1))) >> S1_LN_GUARD;
        y[d] = sat16(r + (acc_t)beta[beta_off + d]);
    }
#else
    int e, nq;
    fdyad(meanabs, e, nq);
    e_out = e; nq_out = nq;
    int nqm4 = nq - (1 << S1_MBITS);

RESC: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
        acc_t cv = (acc_t)c[d];
        acc_t cn = cv << S1_MBITS;
        if (nqm4 >= 1) cn += cv;
        if (nqm4 >= 2) cn += cv;
        if (nqm4 >= 3) cn += cv;
        acc_t r = cn >> S1_MBITS;
        if (e >= 0)  r = r << e;
        else         r = r >> (-e);
        bool gs, gnz; exp_t ge;
        wdecode((int32_t)gamma[gamma_off + d], gs, ge, gnz);
        acc_t g = gnz ? (r << (int)ge) : (acc_t)0;
        if (gnz && gs) g = -g;
        data_t bd = (data_t)beta[beta_off + d];
        acc_t yv = g + (acc_t)bd;
        y[d] = sat16(yv);
    }
#endif
}

// =============================================================================
// attn_head — one head of attention over the whole causal sequence, soft p*V blend.
// VERBATIM math from forge_attn.cpp: L1 score s[i][j]=-sum_d|Q-K|, exp2-LUT softmax, soft blend
// out[i][d]=sum_j p[i][j]*V[j][d] (the ONE real multiply -> DSP), de-scaled by >>P_FRAC. Saves the
// Q16 probs p and the argmax j* per query (for the future backward) into ON-CHIP arrays (burst-stored
// to gmem at the END of do_fwd_loss) -> no gmem write in the attn loops.
// Q/K/V are [L_SEQ][DH] on-chip; out written to [L_SEQ][DH]; p_save is [L][L], jstar_save is [L].
// =============================================================================
// p_off: DDR word-offset of THIS head's attn-prob region (s1_off_attn_p(l,h)); the per-query prob row is
// burst-stored there (the dead-row reuse — query i's row is final once i+1 starts). p_off < 0 => skip the
// save entirely (the mode-1 forward RECOMPUTE does not re-save probs). jstar stays resident (1 word/query).
static void attn_head(const data_t Q[S1_L_SEQ][S1_DH], const data_t K[S1_L_SEQ][S1_DH],
                      const data_t V[S1_L_SEQ][S1_DH], const int32_t *attn_lut,
                      data_t out[S1_L_SEQ][S1_DH], int32_t *buffer, int p_off, int32_t *jstar_save) {
#pragma HLS INLINE off
    static acc_t  s[S1_L_SEQ];
    static prob_t p[S1_L_SEQ];
    static int32_t p_row[S1_L_SEQ];     // per-query prob row -> DDR (replaces the resident p_save array)
#pragma HLS BIND_STORAGE variable=p_row type=ram_2p impl=bram
    static acc_t  blend[S1_DH];
#pragma HLS ARRAY_PARTITION variable=blend cyclic factor=S1_LANE

ROW: for (int i = 0; i < S1_L_SEQ; i++) {
        acc_t rowmax = S1_NEG_INF;
        int   jstar  = 0;
        acc_t bestS  = S1_NEG_INF;
    SCORE: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { s[j] = S1_NEG_INF; continue; }
            acc_t l1 = 0;
        DABS: for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL factor=S1_LANE
                data_t diff = Q[i][d] - K[j][d];
                acc_t a = (diff < 0) ? (acc_t)(-diff) : (acc_t)diff;
                l1 += a;
            }
            acc_t sc = -l1;
            s[j] = sc;
            if (sc > rowmax) rowmax = sc;
            if (sc > bestS) { bestS = sc; jstar = j; }
        }

#if S1_ATTN_HARD
        // MULT-FREE deploy: argmax-GATHER out=V[jstar] (skip softmax + p*V blend -> DSP=0). jstar from SCORE.
    WOUTH: for (int dh = 0; dh < S1_DH; dh++) {
#pragma HLS PIPELINE II=1
            out[i][dh] = sat16((acc_t)V[jstar][dh]);
        }
        jstar_save[i] = jstar;
#else
        acc_t rowsum = 0;
    EXP: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { p[j] = 0; continue; }
            acc_t gap = (rowmax - s[j]) >> S1_AT_DELTA_SHIFT;
            int idx = (int)gap;
            if (idx < 0) idx = 0;
            if (idx > S1_LUT_N - 1) idx = S1_LUT_N - 1;
            prob_t pv = (prob_t)attn_lut[idx];
            p[j] = pv;
            rowsum += (acc_t)pv;
        }
        int sh = 0;
    NSH: for (int b = 0; b < 16; b++) {
#pragma HLS UNROLL
            if ((rowsum >> sh) > ((acc_t)1 << S1_AT_P_FRAC)) sh++;
        }
    NRM: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            p[j] = p[j] >> sh;
        }

    BINIT: for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL factor=S1_LANE
            blend[d] = 0;
        }
    BLEND: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            prob_t pj = p[j];
            for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL factor=S1_LANE
                blend[d] += (acc_t)((acc_t)pj * (acc_t)V[j][d]);   // THE ONE MULTIPLY -> DSP
            }
        }

    WOUT: for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
            out[i][d] = sat16(blend[d] >> S1_AT_P_FRAC);
        }
        // save probs (Q16, post-normalize) + argmax j* for the backward. The prob row streams PER QUERY to
        // DDR (p_off>=0); jstar stays resident (1 word/query). p_off<0 (bwd recompute) skips the save.
    SAVEP: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            p_row[j] = (int32_t)p[j];
        }
        if (p_off >= 0)
            burst_store_i32(p_row, buffer, p_off + i * S1_TILE_ATTNP, S1_TILE_ATTNP);
        jstar_save[i] = jstar;
#endif
    }
}

// =============================================================================
// On-chip RESIDENT region word-counts (logical extents; the offset map's S1_AL16 padding between
// regions is irrelevant on-chip — we hold each region densely and burst-store its logical extent back).
// =============================================================================
#define S1_W_EMBED   (S1_V * S1_D)
#define S1_W_PROJ    (S1_N_LAYERS * S1_D * S1_D)          // each of WQ/WK/WV/WO
#define S1_W_WUP     (S1_N_LAYERS * S1_FFN * S1_D)
#define S1_W_WDOWN   (S1_N_LAYERS * S1_D * S1_FFN)
#define S1_W_LN      (S1_N_LAYERS * S1_D)                 // each gamma/beta
#define S1_W_WHEAD   (S1_V * S1_D)
#define S1_W_RALL    ((S1_N_LAYERS + 1) * S1_L_SEQ * S1_D)
#define S1_W_ACT     (S1_N_LAYERS * S1_L_SEQ * S1_D)      // X1 / X2
#define S1_W_LNST    (S1_N_LAYERS * S1_L_SEQ * 2)         // LN1/LN2 stats
#define S1_W_ATTNP   (S1_N_LAYERS * S1_H * S1_L_SEQ * S1_L_SEQ)
#define S1_W_ATTNJ   (S1_N_LAYERS * S1_H * S1_L_SEQ)
#define S1_W_RELU    (S1_N_LAYERS * S1_L_SEQ * S1_FFN)
#define S1_W_ARGH    (S1_L_SEQ * S1_V)
#define S1_W_GLOG    (S1_L_SEQ * S1_V)                    // grad_logit / logits

// =============================================================================
// Mode 0: do_fwd_loss — the whole forward + CE over the whole causal sequence.
// RESIDENT on-chip staging: burst-load every weight/LUT/token region ONCE, compute entirely on-chip
// (no gmem in any compute loop), then burst-store every buffer-exposed region at the END.
// =============================================================================
static int do_fwd_loss(int32_t *buffer) {
#pragma HLS INLINE off
    // AREA FIX — force ONE physical engine of each helper, time-multiplexed across all call sites.
    // Without this HLS builds a separate PE per call site (7x fwd_gemm + per-call attn/LN) and the LUT
    // count blows past the device. ALLOCATION limit=1 serializes the calls (latency tradeoff, OK for S1).
#pragma HLS ALLOCATION function instances=fwd_gemm limit=1
#pragma HLS ALLOCATION function instances=attn_head limit=1
#pragma HLS ALLOCATION function instances=layernorm_tok limit=1

    // ---------------------------------------------------------------------
    // (A) RESIDENT weights / LUTs / tokens — burst-loaded ONCE from gmem.
    //     Byte-codes in int8, int16 tables in int16, LUTs/tokens in int32.
    // ---------------------------------------------------------------------
    static half_t  embed_oc[S1_W_EMBED];     // [V*D] int16 embed table
#if S1_POS_EMBED
    static half_t  pos_oc[S1_POS_N];         // [L_SEQ*D] int16 positional embed (added at the gather)
#endif
    static code_t  wq_oc[S1_W_PROJ];         // [N_L][D][D] dyadic codes
    static code_t  wk_oc[S1_W_PROJ];
    static code_t  wv_oc[S1_W_PROJ];
    static code_t  wo_oc[S1_W_PROJ];
    static code_t  wup_oc[S1_W_WUP];         // [N_L][FFN][D]
    static code_t  wdn_oc[S1_W_WDOWN];       // [N_L][D][FFN]
    static code_t  ln1g_oc[S1_W_LN];         // gamma codes
    static half_t  ln1b_oc[S1_W_LN];         // beta int16
    static code_t  ln2g_oc[S1_W_LN];
    static half_t  ln2b_oc[S1_W_LN];
#if S1_USE_LNF
    static code_t  lnfg_oc[S1_D];            // final-LN gamma codes
    static half_t  lnfb_oc[S1_D];            // final-LN beta int16
#endif
#if S1_HEAD_MODE
    static code_t  whead_oc[S1_W_WHEAD];     // [V][D] dyadic codes (the dyadic-GEMM head, untied from embed)
#else
    static half_t  whead_oc[S1_W_WHEAD];     // [V][D] int16 (the legacy max-plus head)
#endif
    static int32_t ce_lut[S1_LUT_N];         // host-staged exp2 ROMs (BRAM static-init reads zero on AFI)
    static int32_t at_lut[S1_LUT_N];
    static int32_t toks[S1_L_SEQ];
    static int32_t tgts[S1_L_SEQ];

    burst_load_half (buffer, S1_OFF_EMBED,    embed_oc, S1_W_EMBED);
#if S1_POS_EMBED
    burst_load_half (buffer, S1_OFF_POS,      pos_oc,   S1_POS_N);
#endif
    burst_load_codes(buffer, S1_OFF_WQ,       wq_oc,    S1_W_PROJ);
    burst_load_codes(buffer, S1_OFF_WK,       wk_oc,    S1_W_PROJ);
    burst_load_codes(buffer, S1_OFF_WV,       wv_oc,    S1_W_PROJ);
    burst_load_codes(buffer, S1_OFF_WO,       wo_oc,    S1_W_PROJ);
    burst_load_codes(buffer, S1_OFF_W_UP,     wup_oc,   S1_W_WUP);
    burst_load_codes(buffer, S1_OFF_W_DOWN,   wdn_oc,   S1_W_WDOWN);
    burst_load_codes(buffer, S1_OFF_LN1_G,    ln1g_oc,  S1_W_LN);
    burst_load_half (buffer, S1_OFF_LN1_B,    ln1b_oc,  S1_W_LN);
    burst_load_codes(buffer, S1_OFF_LN2_G,    ln2g_oc,  S1_W_LN);
    burst_load_half (buffer, S1_OFF_LN2_B,    ln2b_oc,  S1_W_LN);
#if S1_USE_LNF
    burst_load_codes(buffer, S1_OFF_LNF_G,    lnfg_oc,  S1_D);
    burst_load_half (buffer, S1_OFF_LNF_B,    lnfb_oc,  S1_D);
#endif
#if S1_HEAD_MODE
    burst_load_codes(buffer, S1_OFF_W_HEAD,   whead_oc, S1_W_WHEAD);   // dyadic-GEMM head: codes
#else
    burst_load_half (buffer, S1_OFF_W_HEAD,   whead_oc, S1_W_WHEAD);   // max-plus head: int16 values
#endif
    burst_load_i32  (buffer, S1_OFF_CE_LUT,   ce_lut,   S1_LUT_N);
    burst_load_i32  (buffer, S1_OFF_ATTN_LUT, at_lut,   S1_LUT_N);
    burst_load_i32  (buffer, S1_OFF_TOKENS,   toks,     S1_L_SEQ);
    burst_load_i32  (buffer, S1_OFF_TARGETS,  tgts,     S1_L_SEQ);

    // ---------------------------------------------------------------------
    // (B) Activation-save staging.
    //   RESIDENT (small per-(layer,token) metadata — kept on-chip, burst-stored once at END):
    //     ln1st_oc / ln2st_oc / lnfst_oc (LN stats) + attnj_oc (argmax j*).
    //   DDR-STREAMED (the BRAM hogs — NOT resident; each is written per-tile to its S1_OFF_* region as it
    //     is computed via burst_store_i32, mirrored by a per-tile reload in mode 1):
    //     R snapshots / X1 / X2 / relu mask / attn probs / argmax-head / grad-logit / logits.
    //     The small per-tile arrays below replace the whole-block resident arrays.
    // ---------------------------------------------------------------------
    static int32_t ln1st_oc[S1_W_LNST];
    static int32_t ln2st_oc[S1_W_LNST];
#if S1_USE_LNF
    static int32_t lnfst_oc[S1_L_SEQ * 2];   // final-LN (e,nq) per token
#endif
    static int32_t attnj_oc[S1_W_ATTNJ];
    // per-tile DDR-stream scratch (ram_2p/BRAM, one op-slice each — the do_opt tile pattern).
    static int32_t rsnap_t[S1_TILE_RD];      // R snapshot row  [D]  -> S1_OFF_R
#pragma HLS BIND_STORAGE variable=rsnap_t type=ram_2p impl=bram
    static int32_t x1_t[S1_TILE_RD];         // LN1 out row     [D]  -> S1_OFF_X1
#pragma HLS BIND_STORAGE variable=x1_t   type=ram_2p impl=bram
    static int32_t x2_t[S1_TILE_RD];         // LN2 out row     [D]  -> S1_OFF_X2
#pragma HLS BIND_STORAGE variable=x2_t   type=ram_2p impl=bram
    static int32_t relu_t[S1_TILE_RELU];     // relu mask row   [FFN]-> S1_OFF_RELU_MASK
#pragma HLS BIND_STORAGE variable=relu_t type=ram_2p impl=bram
    // (attn prob row streams to S1_OFF_ATTN_P from attn_head's own resident p_row[L_SEQ]; no tile here.)
    static int32_t argh_t[S1_TILE_V];        // head argmax row [V]  -> S1_OFF_ARGMAX_HEAD
#pragma HLS BIND_STORAGE variable=argh_t type=ram_2p impl=bram
    static int32_t glog_t[S1_TILE_V];        // grad-logit row  [V]  -> S1_OFF_GRAD_LOGIT
#pragma HLS BIND_STORAGE variable=glog_t type=ram_2p impl=bram
    static int32_t logit_t[S1_TILE_V];       // logits row      [V]  -> S1_OFF_LOGITS
#pragma HLS BIND_STORAGE variable=logit_t type=ram_2p impl=bram

    // R[t][d] = EMBED[tok[t]*D + d] (+ POS[t*D+d] when S1_POS_EMBED). Embed gather, morph LM head idiom.
    // Saved as R layer-0 input. The POS add is a PURE INT ADD (multiplier-free) that localizes the content
    // routing the L1 attention does (the no-pos kernel is bigram-class; pos lets it USE CONTEXT).
    static data_t R[S1_L_SEQ][S1_D];
EMB: for (int t = 0; t < S1_L_SEQ; t++) {
        int tok = toks[t];
        if (tok < 0) tok = 0; if (tok >= S1_V) tok = S1_V - 1;
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            data_t rv = (data_t)embed_oc[tok * S1_D + d];   // on-chip embed gather (was gmem)
#if S1_POS_EMBED
            rv = (data_t)(rv + (data_t)pos_oc[t * S1_D + d]);   // + positional embed (int add; multiplier-free)
#endif
            R[t][d] = rv;
            rsnap_t[d] = (int32_t)rv;                        // R_0 snapshot row (DDR-streamed)
        }
        // stream-save R_0 (block-0 input) row t -> S1_OFF_R (was resident R_all_oc).
        burst_store_i32(rsnap_t, buffer, s1_off_R(0) + t * S1_TILE_RD, S1_TILE_RD);
    }

    // on-chip block scratch
    static data_t X1[S1_L_SEQ][S1_D];
    static data_t Qm[S1_L_SEQ][S1_D];
    static data_t Km[S1_L_SEQ][S1_D];
    static data_t Vm[S1_L_SEQ][S1_D];
    static data_t Qh[S1_L_SEQ][S1_DH];
    static data_t Kh[S1_L_SEQ][S1_DH];
    static data_t Vh[S1_L_SEQ][S1_DH];
    static data_t Ah[S1_L_SEQ][S1_DH];
    static data_t Acat[S1_L_SEQ][S1_D];
    static data_t AO[S1_L_SEQ][S1_D];
    static data_t X2[S1_L_SEQ][S1_D];
    static data_t U[S1_L_SEQ][S1_FFN];
    static data_t Hd[S1_L_SEQ][S1_FFN];
    static data_t Dn[S1_L_SEQ][S1_D];

LAYER: for (int l = 0; l < S1_N_LAYERS; l++) {
        // ---- LN1 ----
    LN1: for (int t = 0; t < S1_L_SEQ; t++) {
            data_t row[S1_D], yrow[S1_D];
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                row[d] = R[t][d];
            }
            int e, nq;
            layernorm_tok(row, ln1g_oc, l * S1_D, ln1b_oc, l * S1_D, yrow, e, nq);
            ln1st_oc[l * S1_L_SEQ * 2 + 2 * t]     = e;
            ln1st_oc[l * S1_L_SEQ * 2 + 2 * t + 1] = nq;
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                X1[t][d] = yrow[d];
                x1_t[d]  = (int32_t)yrow[d];                  // LN1 out row (DDR-streamed)
            }
            // stream-save LN1 out row t -> S1_OFF_X1 (was resident X1_oc).
            burst_store_i32(x1_t, buffer, s1_off_act(S1_OFF_X1, l) + t * S1_TILE_RD, S1_TILE_RD);
        }

        // ---- Q/K/V projections (each [L][D] -> [L][D]) ----
        fwd_gemm(&X1[0][0], wq_oc, l * S1_D * S1_D, &Qm[0][0], S1_L_SEQ, S1_D, S1_D);
        fwd_gemm(&X1[0][0], wk_oc, l * S1_D * S1_D, &Km[0][0], S1_L_SEQ, S1_D, S1_D);
        fwd_gemm(&X1[0][0], wv_oc, l * S1_D * S1_D, &Vm[0][0], S1_L_SEQ, S1_D, S1_D);

        // ---- per head: split D into H heads of DH, run soft causal attention, concat ----
    HEAD: for (int h = 0; h < S1_H; h++) {
            for (int t = 0; t < S1_L_SEQ; t++) {
                for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
                    Qh[t][d] = Qm[t][h * S1_DH + d];
                    Kh[t][d] = Km[t][h * S1_DH + d];
                    Vh[t][d] = Vm[t][h * S1_DH + d];
                }
            }
            int j_base = (l * S1_H + h) * S1_L_SEQ;
            attn_head(Qh, Kh, Vh, at_lut, Ah, buffer, s1_off_attn_p(l, h), &attnj_oc[j_base]);  // p->DDR per query row
            for (int t = 0; t < S1_L_SEQ; t++) {
                for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
                    Acat[t][h * S1_DH + d] = Ah[t][d];
                }
            }
        }

        // ---- output projection [L][D] -> [L][D] then residual add ----
        fwd_gemm(&Acat[0][0], wo_oc, l * S1_D * S1_D, &AO[0][0], S1_L_SEQ, S1_D, S1_D);
    RES1: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                R[t][d] = sat16((acc_t)R[t][d] + (acc_t)AO[t][d]);
            }

        // ---- LN2 ----
    LN2: for (int t = 0; t < S1_L_SEQ; t++) {
            data_t row[S1_D], yrow[S1_D];
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                row[d] = R[t][d];
            }
            int e, nq;
            layernorm_tok(row, ln2g_oc, l * S1_D, ln2b_oc, l * S1_D, yrow, e, nq);
            ln2st_oc[l * S1_L_SEQ * 2 + 2 * t]     = e;
            ln2st_oc[l * S1_L_SEQ * 2 + 2 * t + 1] = nq;
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                X2[t][d] = yrow[d];
                x2_t[d]  = (int32_t)yrow[d];                  // LN2 out row (DDR-streamed)
            }
            // stream-save LN2 out row t -> S1_OFF_X2 (was resident X2_oc).
            burst_store_i32(x2_t, buffer, s1_off_act(S1_OFF_X2, l) + t * S1_TILE_RD, S1_TILE_RD);
        }

        // ---- FFN up [L][D]->[L][FFN], ReLU (save mask), down [L][FFN]->[L][D], residual add ----
        fwd_gemm(&X2[0][0], wup_oc, l * S1_FFN * S1_D, &U[0][0], S1_L_SEQ, S1_D, S1_FFN);
    RELU: for (int t = 0; t < S1_L_SEQ; t++) {
            for (int k = 0; k < S1_FFN; k++) {
#pragma HLS PIPELINE II=1
                bool act = (U[t][k] > 0);
                Hd[t][k] = act ? U[t][k] : (data_t)0;
                relu_t[k] = act ? 1 : 0;                      // relu mask row (DDR-streamed)
            }
            // stream-save the FFN ReLU mask row t -> S1_OFF_RELU_MASK (was resident relu_oc, the 114-BRAM hog).
            burst_store_i32(relu_t, buffer, s1_off_relu(l) + t * S1_TILE_RELU, S1_TILE_RELU);
        }
        fwd_gemm(&Hd[0][0], wdn_oc, l * S1_D * S1_FFN, &Dn[0][0], S1_L_SEQ, S1_FFN, S1_D);
    RES2: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                R[t][d] = sat16((acc_t)R[t][d] + (acc_t)Dn[t][d]);
            }

        // save the block-l output residual (block-(l+1) input) — per-token snapshot row, DDR-streamed.
    SAVER: for (int t = 0; t < S1_L_SEQ; t++) {
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                rsnap_t[d] = (int32_t)R[t][d];
            }
            // stream-save R_{l+1} snapshot row t -> S1_OFF_R (was resident R_all_oc).
            burst_store_i32(rsnap_t, buffer, s1_off_R(l + 1) + t * S1_TILE_RD, S1_TILE_RD);
        }
    }

    // ---- FINAL LayerNorm (LN_F) before the LM head (S1_USE_LNF). Normalizes the head INPUT so the
    //      max-plus head reads a bounded Rn[t][d] regardless of how large the raw residual R grew when the
    //      dense backbone densifies (the TRAIN_ALL saturation fix; mirrors charlm `head(lnf(h))`). Pure
    //      shift/add/compare (the same layernorm_tok the blocks use). Save LN_F (e,nq) stats for the bwd. ----
    static data_t Rn[S1_L_SEQ][S1_D];
#if S1_USE_LNF
LNF: for (int t = 0; t < S1_L_SEQ; t++) {
        data_t row[S1_D], yrow[S1_D];
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            row[d] = R[t][d];
        }
        int e, nq;
        layernorm_tok(row, lnfg_oc, 0, lnfb_oc, 0, yrow, e, nq);
        lnfst_oc[2 * t]     = e;
        lnfst_oc[2 * t + 1] = nq;
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            Rn[t][d] = yrow[d];
        }
    }
#else
LNF_ID: for (int t = 0; t < S1_L_SEQ; t++)
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            Rn[t][d] = R[t][d];
        }
#endif

    // ---- LM head + CE loss/grad per position ----
    // loss_margin (LOSS[0], the CONVERGENCE signal): sum_t (logit[pred] - logit[target]) — the target
    //   classification margin. It is a TRUE monotone CE surrogate that is bit-exact integer (no log) and,
    //   unlike the Q16 prob-mass proxy, does NOT saturate as the logits sharpen: it is exactly the LUT's
    //   z-argument at v==target, and it strictly -> 0 as the model maximizes the target logit. THIS is the
    //   honest "CE decreases" signal the convergence check asserts on. (The Q16-mass proxy saturates to
    //   2^16-per-position once the distribution peaks, which made it a BROKEN convergence metric.)
    // loss_proxy (LOSS[1], kept for reference): the normalized prob mass NOT on the target (Q16).
    acc_t loss_margin = 0;
    acc_t loss_sum = 0;
#if S1_HEAD_MODE
    // ---- DYADIC-GEMM head (S1_HEAD_MODE=1): logits[t][v] = sum_d wdecode(WHEAD_code[v][d]) * Rn[t][d].
    //      This is EXACTLY the backbone GEMM (fwd_gemm) with x=Rn (=hf, the LN_F output), W=WHEAD codes,
    //      output dim V — a dyadic shift-add MAC, DSP=0. Computed for ALL positions at once into Logits_all;
    //      the per-position loop below reads it (no max, no argmax_head save). ----
    static data_t Logits_all[S1_L_SEQ][S1_D >= S1_V ? S1_D : S1_V];   // [L_SEQ][V] (oversized to V<=col)
    fwd_gemm(&Rn[0][0], whead_oc, 0, &Logits_all[0][0], S1_L_SEQ, S1_D, S1_V, S1_HEAD_RSHIFT);
#endif
HEADLOOP: for (int t = 0; t < S1_L_SEQ; t++) {
        int target = tgts[t];
        if (target < 0) target = 0; if (target >= S1_V) target = S1_V - 1;

        // logits[v]: max-plus head (HEAD_MODE=0) saves argmax d*; dyadic-GEMM head (HEAD_MODE=1) reads the
        // pre-computed Logits_all (no argmax). Rn = LN_F(R) when S1_USE_LNF.
        static data_t  logits[S1_V];
        static int32_t ad_head[S1_V];
#if S1_HEAD_MODE
    HEAD_V: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            logits[v] = Logits_all[t][v];
        }
#else
    HEAD_V: for (int v = 0; v < S1_V; v++) {
            acc_t mx = ML_NEG_INF;
            int   idx_d = 0;
        HEAD_D: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                data_t Wvd = (data_t)whead_oc[v * S1_D + d];   // on-chip head weight (was gmem)
                acc_t cand = (acc_t)Wvd + (acc_t)Rn[t][d];
                if (cand > mx) { mx = cand; idx_d = d; }
            }
            logits[v] = sat16(mx);
            ad_head[v] = idx_d;
        }
#endif
        // pred = argmax_v logits[v]
        int pred = 0;
        {
            acc_t mx = ML_NEG_INF;
        ARGV: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
                if ((acc_t)logits[v] > mx) { mx = (acc_t)logits[v]; pred = v; }
            }
        }

        // balanced LUT cross-entropy: p = ce_lut[(logit[pred]-logit[v])>>zshift] shift-normalized to
        // Q16; grad_logit[v] = p[v] - (v==target ? sum_p : 0)  (rows sum to zero). morph CE verbatim.
        static int32_t e_row[S1_V];
        int32_t S = 0;
    CE_E: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t zneg = (int32_t)logits[pred] - (int32_t)logits[v];   // >= 0
            int idx = (int)(zneg >> S1_CE_ZSHIFT);
            if (idx > 255) idx = 255;
            if (idx < 0) idx = 0;
            e_row[v] = ce_lut[idx];
            S += e_row[v];
        }
        int sh = 0;
    CE_SH: for (int b = 0; b < 8; b++) {
#pragma HLS UNROLL
            if ((S >> sh) > 65536) sh++;
        }
        int32_t psum = 0;
    CE_P: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t pq = e_row[v] >> sh;
            e_row[v] = pq;
            psum += pq;
        }
    CE_G: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            int32_t g = e_row[v] - (v == target ? psum : 0);
            glog_t[v]  = g;                              // grad-logit row (DDR-streamed)
            logit_t[v] = (int32_t)logits[v];            // logits row     (DDR-streamed)
        }
        // stream-save grad-logit + logits rows for position t (was resident glog_oc/logit_oc).
        burst_store_i32(glog_t,  buffer, S1_OFF_GRAD_LOGIT + t * S1_TILE_V, S1_TILE_V);
        burst_store_i32(logit_t, buffer, S1_OFF_LOGITS     + t * S1_TILE_V, S1_TILE_V);
        // CONVERGENCE signal: the target classification margin (logit[pred]-logit[target]) >= 0, summed
        // over positions. Strictly monotone with the model getting the target right; -> 0 at fit. (No log,
        // bit-exact integer, no saturation — the honest CE surrogate. == zneg at v==target.)
        loss_margin += (acc_t)((acc_t)logits[pred] - (acc_t)logits[target]);
        // reference proxy: the normalized prob mass NOT on the target (Q16). Kept in LOSS[1]; this one
        // SATURATES once the distribution peaks (2^16/position), so it is NOT used as the convergence test.
        loss_sum += (acc_t)(psum - e_row[target]);

#if !S1_HEAD_MODE
        // save head argmax route (for the backward sparse scatter) — per-position row, DDR-streamed.
        // (HEAD_MODE=1 has no argmax: the head is a GEMM, its bwd is the dyadic input/weight-grad GEMM.)
    SAVEH: for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            argh_t[v] = ad_head[v];
        }
        burst_store_i32(argh_t, buffer, S1_OFF_ARGMAX_HEAD + t * S1_TILE_V, S1_TILE_V);
#else
        (void)ad_head; (void)argh_t;
#endif
    }

    // ---------------------------------------------------------------------
    // (C) BURST-STORE the RESIDENT metadata back to gmem (wide II=1 writes). The big activation SAVES
    //     (R snapshots / X1 / X2 / attn probs / relu mask / argmax-head / grad-logit / logits) were ALREADY
    //     streamed per-tile to their S1_OFF_* regions during compute above — they are NOT held resident.
    // ---------------------------------------------------------------------
    burst_store_i32(ln1st_oc, buffer, S1_OFF_LN1_STATS,   S1_W_LNST);
    burst_store_i32(ln2st_oc, buffer, S1_OFF_LN2_STATS,   S1_W_LNST);
#if S1_USE_LNF
    burst_store_i32(lnfst_oc, buffer, S1_OFF_LNF_STATS,   S1_L_SEQ * 2);
#endif
    burst_store_i32(attnj_oc, buffer, S1_OFF_ATTN_JSTAR,  S1_W_ATTNJ);
    buffer[S1_OFF_LOSS + 0] = (int32_t)loss_margin;   // CONVERGENCE signal (target margin; monotone)
    buffer[S1_OFF_LOSS + 1] = (int32_t)loss_sum;      // reference Q16-mass proxy (saturating; not asserted)

    return 0;
}

#if !S1_FWD_ONLY   // ---- training-only (do_bwd / do_opt / do_train_run); forward-only deploy gates these out ----
// =============================================================================
// ======================  MODE 1 (BACKWARD) — do_bwd  =========================
// =============================================================================
// Reverses the mode-0 forward chain to compute EVERY gradient for one transformer block, reading the
// activations mode 0 saved (R_all / X1 / X2 / LN stats / relu mask / argmax_head / grad_logit) and the
// resident weights. COMPOSES the CSIM-VERIFIED backward kernels as inlined helpers, math BIT-IDENTICAL:
//   - forge_xgrad_wide.cpp : gx[t][i] = sum_o gO[t][o] * q(W[o][i])  (gO FULL, W dyadic-shift; DSP=0)
//   - forge_wgrad_df3.cpp  : gW[o][i] = sum_t q(gO[t][o]) * q(x[t][i]) (BOTH dyadic -> shift; DSP=0)
//   - forge_attn_bwd.cpp   : gQ/gK/gV (soft-blend + softmax multiplies on DSP; L1 sign(Q-K) mult-free)
//   - forge_layernorm_bwd  : gx + g_gamma + g_beta (STE gamma pass-through + invq rescale; DSP=0)
//   - forge_relu.cpp       : gx = mask ? gO : 0
// MULTIPLIER-FREE except the attn-bwd soft-blend/softmax multiplies (forge_attn_bwd's irreducible DSPs).
// RESIDENT ON-CHIP: burst-load weights + saved acts once, compute the reverse chain on-chip, burst-store
// every g_* region at the END (the morph_lm TT_RESIDENT_W pattern that closed mode 0).
//
// dyad()/wdecode()/fdyad()/apply_invq are the SAME helpers as the forward, so the int64 golden is bit-exact.

// apply the per-token fine-dyadic RESCALE invq=(nq/4)*2^e to a grad/value via shift-add (forge_layernorm_bwd
// apply_invq, BIT-IDENTICAL to the forward rescale chain). NO multiply.
static inline acc_t s1_apply_invq(acc_t v, int e, int nq) {
#pragma HLS INLINE
    int nqm4 = nq - (1 << S1_MBITS);
    acc_t cn = v << S1_MBITS;
    if (nqm4 >= 1) cn += v;
    if (nqm4 >= 2) cn += v;
    if (nqm4 >= 3) cn += v;
    acc_t r = cn >> S1_MBITS;
    if (e >= 0) r = r << e;
    else        r = r >> (-e);
    return r;
}

static inline int32_t sat32(acc_t v) {
#pragma HLS INLINE
    if (v >  (acc_t)2147483647LL)    return  2147483647;
    if (v < (acc_t)(-2147483648LL))  return (int32_t)(-2147483648LL);
    return (int32_t)v;
}

// -----------------------------------------------------------------------------
// xgrad_gemm — gx[t][i] = sum_o gO[t][o] * q(W[o][i]).  gO held FULL int32 on-chip; W decoded inline.
// Reduces over o (transposed-W access vs the forward), output-stationary on (t,i). VERBATIM forge_xgrad_wide
// asymmetric dyadic MAC: term = (gO << ew), sign folded, NO multiply (DSP=0).  W ROW-MAJOR [O][I].
// RUNTIME-DIM (M rows, I, O) — ONE engine, loops runtime-bounded. gO FLAT ROW-MAJOR [M][O] (gO[t*O+o]);
// gx FLAT ROW-MAJOR [M][I] (gx[t*I+i]). Math/order VERBATIM the templated version -> golden unchanged.
// -----------------------------------------------------------------------------
static void xgrad_gemm(const int32_t *gO, const code_t *wcode, int w_off, int32_t *gx,
                       int M, int I, int O) {
#pragma HLS INLINE off
XG_I: for (int i = 0; i < I; i++) {
        acc_t acc[S1_GEMM_TMAX];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=S1_LANE
        for (int t = 0; t < M; t++) {
#pragma HLS UNROLL factor=S1_LANE
            acc[t] = 0;
        }
        // NARROWED T-reduction: LANE-wide chunks (cyclic acc banks -> LANE adds/cycle, II=1). The full-T
        // reduction is the same integer total (order-independent) -> BIT-EXACT, just more cycles.
XG_O:   for (int o = 0; o < O; o++) {
            bool sw, nzw; exp_t ew;
            wdecode((int32_t)wcode[w_off + o * I + i], sw, ew, nzw);
XG_TT:      for (int tt = 0; tt < S1_GEMM_TMAX; tt += S1_LANE) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=128 max=512
                for (int tl = 0; tl < S1_LANE; tl++) {
#pragma HLS UNROLL
                    int t = tt + tl;
                    if (t < M) {
                        acc_t g = (acc_t)gO[t * O + o];
                        acc_t shifted = g << (int)ew;      // gO * 2^ew (barrel shift; gO keeps its sign)
                        if (nzw) acc[t] += sw ? (acc_t)(-shifted) : shifted;
                    }
                }
            }
        }
        for (int t = 0; t < M; t++) {
#pragma HLS UNROLL factor=S1_LANE
            gx[t * I + i] = sat32(acc[t]);
        }
    }
}

// -----------------------------------------------------------------------------
// wgrad_gemm — gW[o][i] = sum_t q(gO[t][o]) * q(x[t][i]).  BOTH operands dyadic-quantized (forge_wgrad_df3):
// each product = ±2^(eo+ei) (sign-xor + exp-ADD + barrel shift + signed accumulate), NO multiply (DSP=0).
// The contraction is over t (the L_SEQ positions). gW ROW-MAJOR [O][I], int32 full precision, ACCUMULATED.
// RUNTIME-DIM (M=T rows, I, O) — ONE engine, banks sized to MAX, loops runtime-bounded. gO FLAT
// ROW-MAJOR [M][O] (gO[t*O+o]); xin FLAT ROW-MAJOR [M][I] (xin[t*I+i]). Math/order VERBATIM -> golden same.
// -----------------------------------------------------------------------------
static void wgrad_gemm(const int32_t *gO, const data_t *xin, int32_t *gW, int gw_off,
                       int M, int I, int O) {
#pragma HLS INLINE off
    // dyad-quantize gO and x once (q(.) = ±2^e). Partition T complete so the t-reduction lanes read in parallel.
    static exp_t goE[S1_GEMM_TMAX][S1_GEMM_OMAX]; static bool goS[S1_GEMM_TMAX][S1_GEMM_OMAX]; static bool goNZ[S1_GEMM_TMAX][S1_GEMM_OMAX];
    static exp_t xE[S1_GEMM_TMAX][S1_GEMM_IMAX];  static bool xS[S1_GEMM_TMAX][S1_GEMM_IMAX];  static bool xNZ[S1_GEMM_TMAX][S1_GEMM_IMAX];
#pragma HLS ARRAY_PARTITION variable=goE  cyclic factor=S1_LANE dim=1
#pragma HLS ARRAY_PARTITION variable=goS  cyclic factor=S1_LANE dim=1
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=S1_LANE dim=1
#pragma HLS ARRAY_PARTITION variable=xE   cyclic factor=S1_LANE dim=1
#pragma HLS ARRAY_PARTITION variable=xS   cyclic factor=S1_LANE dim=1
#pragma HLS ARRAY_PARTITION variable=xNZ  cyclic factor=S1_LANE dim=1
QGO: for (int t = 0; t < M; t++)
        for (int o = 0; o < O; o++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=128 max=512
            dyad_wide((acc_t)gO[t * O + o], goS[t][o], goE[t][o], goNZ[t][o]);  // FULL int32 grad (no int16 wrap)
        }
QXW: for (int t = 0; t < M; t++)
        for (int i = 0; i < I; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=128 max=512
            dyad((data_t)xin[t * I + i], xS[t][i], xE[t][i], xNZ[t][i]);
        }
    // NARROWED T-reduction: instead of a single full-TMAX unrolled accumulate, keep LANE partial sums
    // (cyclic banks -> LANE conflict-free reads/cycle, inner LANE UNROLL pipelines the tt chunk loop II=1),
    // then sum the LANE partials. BIT-EXACT: integer adds, same total (order-independent, int48 headroom).
WG_O: for (int o = 0; o < O; o++) {
WG_I:   for (int i = 0; i < I; i++) {
            acc_t psum[S1_LANE];
#pragma HLS ARRAY_PARTITION variable=psum complete
            for (int tl = 0; tl < S1_LANE; tl++) {
#pragma HLS UNROLL
                psum[tl] = 0;
            }
WG_TT:      for (int tt = 0; tt < S1_GEMM_TMAX; tt += S1_LANE) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=128 max=512
                for (int tl = 0; tl < S1_LANE; tl++) {
#pragma HLS UNROLL
                    int t = tt + tl;
                    if (t < M) {
                        exp_t eo = goE[t][o]; exp_t ei = xE[t][i];
                        int e = (int)eo + (int)ei;
                        acc_t term = (acc_t)1 << e;
                        bool neg = goS[t][o] ^ xS[t][i];
                        bool nz  = goNZ[t][o] & xNZ[t][i];
                        if (nz) psum[tl] += neg ? (acc_t)(-term) : term;
                    }
                }
            }
            acc_t acc = 0;
            for (int tl = 0; tl < S1_LANE; tl++) {
#pragma HLS UNROLL
                acc += psum[tl];
            }
            gW[gw_off + o * I + i] += sat32(acc);
        }
    }
}

// -----------------------------------------------------------------------------
// attn_head_bwd — VERBATIM forge_attn_bwd math (one head). Re-derives p from Q,K (BIT-IDENTICAL to the
// forward attn_head: L1 score -> exp2 ROM -> pow2-ceil normalize), then the soft-attention backward:
//   gV[j][d] += p[i][j]*gOut[i][d]        (multiply -> DSP)
//   gp[i][j]  = sum_d gOut[i][d]*V[j][d]  (multiply -> DSP)
//   ex[i]     = (sum_j p*gp) >> P_FRAC ; gs[i][j] = (p*(gp-ex)) >> P_FRAC   (multiply -> DSP)
//   gQ[i][d] += -sgn(Q-K)*gs ; gK[j][d] += +sgn(Q-K)*gs                     (MULT-FREE conditional add)
// gQ/gK/gV are int32 on-chip accumulators (zeroed by the caller per head). gOut is FULL int32.
// -----------------------------------------------------------------------------
static void attn_head_bwd(const data_t Q[S1_L_SEQ][S1_DH], const data_t K[S1_L_SEQ][S1_DH],
                          const data_t V[S1_L_SEQ][S1_DH], const int32_t gOut[S1_L_SEQ][S1_DH],
                          const int32_t *attn_lut,
                          int32_t gQ[S1_L_SEQ][S1_DH], int32_t gK[S1_L_SEQ][S1_DH],
                          int32_t gV[S1_L_SEQ][S1_DH]) {
#pragma HLS INLINE off
    static acc_t  s[S1_L_SEQ];
    static prob_t p[S1_L_SEQ];
    static int32_t gp[S1_L_SEQ];
    static int32_t gs[S1_L_SEQ];

AB_ROW: for (int i = 0; i < S1_L_SEQ; i++) {
        // --- re-derive L1 scores (causal) ---
        acc_t rowmax = S1_NEG_INF;
    AB_SCORE: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { s[j] = S1_NEG_INF; continue; }
            acc_t l1 = 0;
        AB_DABS: for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL factor=S1_LANE
                data_t diff = Q[i][d] - K[j][d];
                acc_t a = (diff < 0) ? (acc_t)(-diff) : (acc_t)diff;
                l1 += a;
            }
            acc_t sc = -l1;
            s[j] = sc;
            if (sc > rowmax) rowmax = sc;
        }
        // --- softmax via exp2 ROM (Q16) + pow2-ceil normalize ---
        acc_t rowsum = 0;
    AB_EXP: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { p[j] = 0; continue; }
            acc_t gap = (rowmax - s[j]) >> S1_AT_DELTA_SHIFT;
            int idx = (int)gap;
            if (idx < 0) idx = 0;
            if (idx > S1_LUT_N - 1) idx = S1_LUT_N - 1;
            prob_t pv = (prob_t)attn_lut[idx];
            p[j] = pv;
            rowsum += (acc_t)pv;
        }
        int sh = 0;
    AB_NSH: for (int b = 0; b < 16; b++) {
#pragma HLS UNROLL
            if ((rowsum >> sh) > ((acc_t)1 << S1_AT_P_FRAC)) sh++;
        }
    AB_NRM: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            p[j] = p[j] >> sh;
        }
        // --- (b) gp[j] = sum_d gOut[i][d]*V[j][d] ; expnum = sum_j p*gp  (multiplies -> DSP) ---
        acc_t expnum = 0;
    AB_GP: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            acc_t dot = 0;
            for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL factor=S1_LANE
                dot += (acc_t)((acc_t)gOut[i][d] * (acc_t)V[j][d]);   // multiply
            }
            int32_t gpj = (int32_t)dot;
            gp[j] = gpj;
            expnum += (acc_t)((acc_t)p[j] * (acc_t)gpj);              // multiply
        }
        int32_t ex = (int32_t)(expnum >> S1_AT_P_FRAC);
        // --- (c) gs[j] = (p[j]*(gp[j]-ex)) >> P_FRAC  (multiply -> DSP) ---
    AB_GS: for (int j = 0; j < S1_L_SEQ; j++) {
#pragma HLS PIPELINE II=1
            if (j > i) { gs[j] = 0; continue; }
            int32_t centered = (int32_t)(gp[j] - ex);
            acc_t prod = (acc_t)((acc_t)p[j] * (acc_t)centered);      // multiply
            gs[j] = (int32_t)(prod >> S1_AT_P_FRAC);
        }
        // --- (a) gV[j][d] += p[i][j]*gOut[i][d]  (multiply -> DSP, mirrors fwd p*V) ---
    AB_GV: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            prob_t pj = p[j];
            for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL factor=S1_LANE
                acc_t t = (acc_t)((acc_t)pj * (acc_t)gOut[i][d]);     // multiply
                gV[j][d] = sat32((acc_t)gV[j][d] + (t >> S1_AT_P_FRAC));
            }
        }
        // --- (d) L1-score subgradient (MULT-FREE conditional add) ---
    AB_SUBG: for (int j = 0; j <= i; j++) {
#pragma HLS PIPELINE II=1
            acc_t gsj = (acc_t)gs[j];
            for (int d = 0; d < S1_DH; d++) {
#pragma HLS UNROLL factor=S1_LANE
                data_t diff = Q[i][d] - K[j][d];
                if (diff > 0) {           // sgn=+1: gQ -= gs ; gK += gs
                    gQ[i][d] = sat32((acc_t)gQ[i][d] - gsj);
                    gK[j][d] = sat32((acc_t)gK[j][d] + gsj);
                } else if (diff < 0) {    // sgn=-1: gQ += gs ; gK -= gs
                    gQ[i][d] = sat32((acc_t)gQ[i][d] + gsj);
                    gK[j][d] = sat32((acc_t)gK[j][d] - gsj);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// ln_gamma_grad — g_gamma[d] += sum_t gY[t][d] * q(r[t][d]), where r[t][d] is the pre-gamma rescaled
// centered value of the LayerNorm (r = c*(nq/4)*2^e, c = x-mean), recomputed from the PRE-NORM residual
// row Rin[t][:] using the SAVED per-token (e,nq) — BIT-IDENTICAL to the forward layernorm_tok rescale.
// r is dyadic-quantized (forge_layernorm_bwd c_hat-code contract) so the product is gY << dyad(r).exp
// (sign-folded) — NO multiply (DSP=0). The gamma scale is a per-channel learnable ±2^k; this is its grad.
// -----------------------------------------------------------------------------
static void ln_gamma_grad(const data_t Rin[S1_L_SEQ][S1_D], const int32_t *lnst, int l,
                          const int32_t gY[S1_L_SEQ][S1_D], int32_t *gGamma, int gg_off) {
#pragma HLS INLINE off
LNG_T: for (int t = 0; t < S1_L_SEQ; t++) {
        // recompute mean + c (sat16) over the pre-norm row, exactly as layernorm_tok.
        acc_t sum = 0;
    LNG_SUM: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            sum += (acc_t)Rin[t][d];
        }
        acc_t mean = sum >> S1_LOG2_D;
        int e  = lnst[l * S1_L_SEQ * 2 + 2 * t];
        int nq = lnst[l * S1_L_SEQ * 2 + 2 * t + 1];
        int nqm4 = nq - (1 << S1_MBITS);
    LNG_D: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            acc_t cv = (acc_t)sat16((acc_t)Rin[t][d] - mean);
            acc_t cn = cv << S1_MBITS;
            if (nqm4 >= 1) cn += cv;
            if (nqm4 >= 2) cn += cv;
            if (nqm4 >= 3) cn += cv;
            acc_t r = cn >> S1_MBITS;
            if (e >= 0) r = r << e;
            else        r = r >> (-e);
            // dyadic-quantize r, then g_gamma += gY << dyad(r).exp (sign-folded). NO multiply.
            bool rs, rnz; exp_t re;
            dyad(sat16(r), rs, re, rnz);
            if (rnz) {
                acc_t term = (acc_t)gY[t][d] << (int)re;
                bool neg = rs;   // r's sign flips; gY's sign already in term
                gGamma[gg_off + d] = sat32((acc_t)gGamma[gg_off + d] + (neg ? (acc_t)(-term) : term));
            }
        }
    }
}

// =============================================================================
// Mode 1: do_bwd — the whole backward chain over the causal sequence (reverse of do_fwd_loss).
// RESIDENT on-chip: burst-load every weight + saved activation ONCE, compute the reverse chain entirely
// on-chip, then burst-store every gradient region at the END.
// =============================================================================
static int do_bwd(int32_t *buffer) {
#pragma HLS INLINE off
    // AREA FIX — force ONE physical engine of each helper, time-multiplexed across all call sites.
    // do_bwd is the dominant area consumer: it INLINES xgrad_gemm + wgrad_gemm per linear (~14 GEMM PEs)
    // plus fwd_gemm recompute, attn_head/attn_head_bwd and ln_gamma_grad. ALLOCATION limit=1 builds ONE
    // engine of each class and serializes the call sites (latency tradeoff, acceptable for the S1 proof).
#pragma HLS ALLOCATION function instances=fwd_gemm limit=1
#pragma HLS ALLOCATION function instances=xgrad_gemm limit=1
#pragma HLS ALLOCATION function instances=wgrad_gemm limit=1
#pragma HLS ALLOCATION function instances=attn_head limit=1
#pragma HLS ALLOCATION function instances=attn_head_bwd limit=1
#pragma HLS ALLOCATION function instances=ln_gamma_grad limit=1

    // ---------------------------------------------------------------------
    // (A) RESIDENT weights / LUTs / tokens — burst-loaded ONCE (same encodings as mode 0).
    // ---------------------------------------------------------------------
    static half_t  embed_oc[S1_W_EMBED];
    static code_t  wq_oc[S1_W_PROJ];
    static code_t  wk_oc[S1_W_PROJ];
    static code_t  wv_oc[S1_W_PROJ];
    static code_t  wo_oc[S1_W_PROJ];
    static code_t  wup_oc[S1_W_WUP];
    static code_t  wdn_oc[S1_W_WDOWN];
    static code_t  ln1g_oc[S1_W_LN];
    static half_t  ln1b_oc[S1_W_LN];   (void)ln1b_oc;   // beta not needed in bwd (its grad is a column sum)
    static code_t  ln2g_oc[S1_W_LN];
    static half_t  ln2b_oc[S1_W_LN];   (void)ln2b_oc;
#if S1_USE_LNF
    static code_t  lnfg_oc[S1_D];      // final-LN gamma codes (needed for LN_F bwd)
    static half_t  lnfb_oc[S1_D];   (void)lnfb_oc;   // beta not needed in bwd (its grad is a column sum)
#endif
#if S1_HEAD_MODE
    static code_t  whead_oc[S1_W_WHEAD];   // dyadic-GEMM head: codes (needed for the head input-grad GEMM)
#else
    static half_t  whead_oc[S1_W_WHEAD];   (void)whead_oc;   // max-plus head bwd does not read W_HEAD (sparse scatter routes via argmax)
#endif
    static int32_t at_lut[S1_LUT_N];
    static int32_t toks[S1_L_SEQ];

    burst_load_half (buffer, S1_OFF_EMBED,    embed_oc, S1_W_EMBED);   (void)embed_oc;
    burst_load_codes(buffer, S1_OFF_WQ,       wq_oc,    S1_W_PROJ);
    burst_load_codes(buffer, S1_OFF_WK,       wk_oc,    S1_W_PROJ);
    burst_load_codes(buffer, S1_OFF_WV,       wv_oc,    S1_W_PROJ);
    burst_load_codes(buffer, S1_OFF_WO,       wo_oc,    S1_W_PROJ);
    burst_load_codes(buffer, S1_OFF_W_UP,     wup_oc,   S1_W_WUP);
    burst_load_codes(buffer, S1_OFF_W_DOWN,   wdn_oc,   S1_W_WDOWN);
    burst_load_codes(buffer, S1_OFF_LN1_G,    ln1g_oc,  S1_W_LN);
    burst_load_half (buffer, S1_OFF_LN1_B,    ln1b_oc,  S1_W_LN);
    burst_load_codes(buffer, S1_OFF_LN2_G,    ln2g_oc,  S1_W_LN);
    burst_load_half (buffer, S1_OFF_LN2_B,    ln2b_oc,  S1_W_LN);
#if S1_USE_LNF
    burst_load_codes(buffer, S1_OFF_LNF_G,    lnfg_oc,  S1_D);
    burst_load_half (buffer, S1_OFF_LNF_B,    lnfb_oc,  S1_D);
#endif
#if S1_HEAD_MODE
    burst_load_codes(buffer, S1_OFF_W_HEAD,   whead_oc, S1_W_WHEAD);   // dyadic-GEMM head: codes
#else
    burst_load_half (buffer, S1_OFF_W_HEAD,   whead_oc, S1_W_WHEAD);   // max-plus head: int16 (unused in bwd)
#endif
    burst_load_i32  (buffer, S1_OFF_ATTN_LUT, at_lut,   S1_LUT_N);
    burst_load_i32  (buffer, S1_OFF_TOKENS,   toks,     S1_L_SEQ);

    // ---------------------------------------------------------------------
    // (B) Saved activations from mode 0.
    //   RESIDENT (small per-(layer,token) metadata — burst-loaded once): LN stats.
    //   DDR-STREAMED (the BRAM hogs — reloaded PER-TILE right before each consuming op, mirroring the fwd
    //     per-tile saves): R snapshots / X1 / X2 / relu mask / argmax-head / grad-logit. The per-tile reload
    //     scratch below replaces the whole-block resident arrays. Every reload reads the IDENTICAL S1_OFF_*
    //     offset the fwd wrote (bit-exact; the int64 #else path makes burst_load_i32 a plain scalar copy).
    // ---------------------------------------------------------------------
    static int32_t ln1st_oc[S1_W_LNST];
    static int32_t ln2st_oc[S1_W_LNST];
#if S1_USE_LNF
    static int32_t lnfst_oc[S1_L_SEQ * 2];
#endif
    // per-tile DDR-reload scratch (ram_2p/BRAM, one op-slice each — the do_opt tile pattern).
    static int32_t rrow_t[S1_TILE_RD];       // R snapshot row  [D]  <- S1_OFF_R
#pragma HLS BIND_STORAGE variable=rrow_t   type=ram_2p impl=bram
    static int32_t x1row_t[S1_TILE_RD];      // LN1 out row     [D]  <- S1_OFF_X1
#pragma HLS BIND_STORAGE variable=x1row_t  type=ram_2p impl=bram
    static int32_t x2row_t[S1_TILE_RD];      // LN2 out row     [D]  <- S1_OFF_X2
#pragma HLS BIND_STORAGE variable=x2row_t  type=ram_2p impl=bram
    static int32_t relurow_t[S1_TILE_RELU];  // relu mask row   [FFN]<- S1_OFF_RELU_MASK
#pragma HLS BIND_STORAGE variable=relurow_t type=ram_2p impl=bram
    static int32_t arghrow_t[S1_TILE_V];     // head argmax row [V]  <- S1_OFF_ARGMAX_HEAD
#pragma HLS BIND_STORAGE variable=arghrow_t type=ram_2p impl=bram
    static int32_t glogrow_t[S1_TILE_V];     // grad-logit row  [V]  <- S1_OFF_GRAD_LOGIT
#pragma HLS BIND_STORAGE variable=glogrow_t type=ram_2p impl=bram

    burst_load_i32(buffer, S1_OFF_LN1_STATS,   ln1st_oc, S1_W_LNST);
    burst_load_i32(buffer, S1_OFF_LN2_STATS,   ln2st_oc, S1_W_LNST);
#if S1_USE_LNF
    burst_load_i32(buffer, S1_OFF_LNF_STATS,   lnfst_oc, S1_L_SEQ * 2);
#endif

    // ---------------------------------------------------------------------
    // (C) RESIDENT gradient accumulators (zeroed) + intermediate grad scratch.
    // ---------------------------------------------------------------------
    static int32_t gEMBED_oc[S1_W_EMBED];
    static int32_t gWQ_oc[S1_W_PROJ];
    static int32_t gWK_oc[S1_W_PROJ];
    static int32_t gWV_oc[S1_W_PROJ];
    static int32_t gWO_oc[S1_W_PROJ];
    static int32_t gWUP_oc[S1_W_WUP];
    static int32_t gWDN_oc[S1_W_WDOWN];
    static int32_t gLN1G_oc[S1_W_LN];
    static int32_t gLN1B_oc[S1_W_LN];
    static int32_t gLN2G_oc[S1_W_LN];
    static int32_t gLN2B_oc[S1_W_LN];
#if S1_USE_LNF
    static int32_t gLNFG_oc[S1_D];
    static int32_t gLNFB_oc[S1_D];
#endif
    static int32_t gWHEAD_oc[S1_W_WHEAD];
#if S1_POS_EMBED
    static int32_t gPOS_oc[S1_POS_N];        // [L_SEQ*D] positional-embed grad (= residual-input grad, per row t)
#endif

ZG_E: for (int i = 0; i < S1_W_EMBED; i++) { gEMBED_oc[i] = 0; gWHEAD_oc[i] = 0; }
ZG_P: for (int i = 0; i < S1_W_PROJ; i++) { gWQ_oc[i]=0; gWK_oc[i]=0; gWV_oc[i]=0; gWO_oc[i]=0; }
ZG_U: for (int i = 0; i < S1_W_WUP; i++)   gWUP_oc[i] = 0;
ZG_D: for (int i = 0; i < S1_W_WDOWN; i++)  gWDN_oc[i] = 0;
ZG_L: for (int i = 0; i < S1_W_LN; i++) { gLN1G_oc[i]=0; gLN1B_oc[i]=0; gLN2G_oc[i]=0; gLN2B_oc[i]=0; }
#if S1_USE_LNF
ZG_F: for (int i = 0; i < S1_D; i++) { gLNFG_oc[i]=0; gLNFB_oc[i]=0; }
#endif
#if S1_POS_EMBED
ZG_POS: for (int i = 0; i < S1_POS_N; i++) gPOS_oc[i] = 0;
#endif

    // gR — the running backward residual stream (output-grad of the block; built by the head bwd).
    static int32_t gR[S1_L_SEQ][S1_D];
    // intermediate grad scratch (buffer-exposed at the END for the bit-check).
    static int32_t gX1_oc[S1_L_SEQ][S1_D];
    static int32_t gX2_oc[S1_L_SEQ][S1_D];

    // ---------------------------------------------------------------------
    // (1) LM-head backward. The head reads Rn = LN_F(R_fin) (the head INPUT); the head bwd produces gRn (grad
    //     w.r.t. Rn). Then LN_F backward maps gRn -> gR (+ g_LNF_gamma/beta). When S1_USE_LNF=0, Rn==R_fin.
    //
    //   HEAD_MODE=0 (max-plus, SPARSE): route grad_logit[t][v] through the saved argmax d*:
    //       gRn[t][d*] += grad_logit[t][v]  AND  g_W_HEAD[v][d*] += grad_logit[t][v].   (morph head-scatter)
    //   HEAD_MODE=1 (dyadic-GEMM): mirror the backbone GEMM backward EXACTLY (gO = grad_logit [L_SEQ][V]):
    //       gRn[t][d]      = sum_v grad_logit[t][v] * wdecode(WHEAD[v][d])     (= q(W)^T @ gO, the xgrad GEMM)
    //       g_WHEAD[v][d] += sum_t q(grad_logit[t][v]) * q(hf[t][d])           (operand-quantized wgrad GEMM)
    //     where hf = Rn = LN_F(R_fin) (the SAME head input the forward used), recomputed here.
    // ---------------------------------------------------------------------
    static int32_t gRn[S1_L_SEQ][S1_D];
GRN_ZERO: for (int t = 0; t < S1_L_SEQ; t++)
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            gRn[t][d] = 0;
            gR[t][d]  = 0;
        }
#if S1_HEAD_MODE
    // --- recompute hf = Rn (the head input): load R_fin, apply LN_F when S1_USE_LNF (bit-identical to fwd). ---
    static data_t hf_m[S1_L_SEQ][S1_D];
    static int32_t glogit_all[S1_L_SEQ][S1_V];
HF_RECOMP: for (int t = 0; t < S1_L_SEQ; t++) {
        burst_load_i32(buffer, s1_off_R(S1_N_LAYERS) + t * S1_TILE_RD, rrow_t, S1_TILE_RD);
#if S1_USE_LNF
        data_t row[S1_D], yrow[S1_D];
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            row[d] = (data_t)rrow_t[d];
        }
        int e_d, nq_d;
        layernorm_tok(row, lnfg_oc, 0, lnfb_oc, 0, yrow, e_d, nq_d);   // hf = LN_F(R_fin)
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            hf_m[t][d] = yrow[d];
        }
#else
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            hf_m[t][d] = (data_t)rrow_t[d];        // no LN_F: head input is R_fin directly
        }
#endif
        // load this position's grad-logit row into the resident [L_SEQ][V] gO operand (for the GEMM bwd).
        burst_load_i32(buffer, S1_OFF_GRAD_LOGIT + t * S1_TILE_V, glogrow_t, S1_TILE_V);
        for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=1
            glogit_all[t][v] = glogrow_t[v];
        }
    }
    // gRn[t][d] = sum_v grad_logit[t][v] * q(WHEAD[v][d])  (the dyadic input-grad GEMM; gO=[L_SEQ][V], W=[V][D]).
    xgrad_gemm(&glogit_all[0][0], whead_oc, 0, &gRn[0][0], S1_L_SEQ, S1_D, S1_V);
    // g_WHEAD[v][d] += sum_t q(grad_logit[t][v]) * q(hf[t][d])  (the dyadic weight-grad GEMM; gW=[V][D]).
    wgrad_gemm(&glogit_all[0][0], &hf_m[0][0], gWHEAD_oc, 0, S1_L_SEQ, S1_D, S1_V);
    (void)arghrow_t;
#else
HEAD_BWD: for (int t = 0; t < S1_L_SEQ; t++) {
        // reload position t's grad-logit + argmax-head rows from DDR (was resident glog_oc/argh_oc).
        burst_load_i32(buffer, S1_OFF_GRAD_LOGIT  + t * S1_TILE_V, glogrow_t, S1_TILE_V);
        burst_load_i32(buffer, S1_OFF_ARGMAX_HEAD + t * S1_TILE_V, arghrow_t, S1_TILE_V);
        for (int v = 0; v < S1_V; v++) {
#pragma HLS PIPELINE II=4
            int32_t g = glogrow_t[v];
            if (g == 0) continue;
            int dstar = arghrow_t[v];
            if (dstar < 0 || dstar >= S1_D) continue;
            gWHEAD_oc[v * S1_D + dstar] += g;
            gRn[t][dstar] = sat32((acc_t)gRn[t][dstar] + (acc_t)g);
        }
    }
#endif

#if S1_USE_LNF
    // ---- LN_F backward: gY = gRn -> gR (residual-stream grad) + g_LNF_gamma + g_LNF_beta. The pre-LN_F
    //      residual is R_fin = R_all[N_LAYERS] (saved by mode 0). Same math as LN1/LN2 bwd. ----
    static data_t Rfin_m[S1_L_SEQ][S1_D];
LNF_LDRFIN: for (int t = 0; t < S1_L_SEQ; t++) {
        // reload the final R snapshot row t from DDR (was resident R_all_oc[N_LAYERS]).
        burst_load_i32(buffer, s1_off_R(S1_N_LAYERS) + t * S1_TILE_RD, rrow_t, S1_TILE_RD);
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            Rfin_m[t][d] = (data_t)rrow_t[d];
        }
    }
LNF_BWD: for (int t = 0; t < S1_L_SEQ; t++) {
        int e_n  = lnfst_oc[2 * t];
        int nq_n = lnfst_oc[2 * t + 1];
        acc_t g_c[S1_D];
        acc_t gsum = 0;
    LNF_GSTE: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            bool gsn, gnz; exp_t ge;
            wdecode((int32_t)lnfg_oc[d], gsn, ge, gnz);
            acc_t gchat = gnz ? ((acc_t)gRn[t][d] << (int)ge) : (acc_t)0;
            if (gnz && gsn) gchat = -gchat;
            acc_t gd = s1_apply_invq(gchat, e_n, nq_n);
            g_c[d] = gd;
            gsum += gd;
        }
        acc_t mean = gsum >> S1_LOG2_D;
    LNF_GSUB: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            gR[t][d] = sat32((acc_t)gR[t][d] + (g_c[d] - mean));
            gLNFB_oc[d] = sat32((acc_t)gLNFB_oc[d] + (acc_t)gRn[t][d]);
        }
    }
    // g_LNF_gamma[d] += sum_t gRn[t][d] * q(rf[t][d])  (rf = pre-gamma rescaled c of LN_F). ln_gamma_grad
    // reads its stats as lnst[l*L_SEQ*2 + ...]; LN_F is a single LN so pass l=0 against the LN_F stats array.
    ln_gamma_grad(Rfin_m, lnfst_oc, 0, gRn, gLNFG_oc, 0);
#else
LNF_BWD_ID: for (int t = 0; t < S1_L_SEQ; t++)
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            gR[t][d] = gRn[t][d];
        }
#endif

    // ---------------------------------------------------------------------
    // (2) per layer (REVERSE).
    // ---------------------------------------------------------------------
    static int32_t gDn[S1_L_SEQ][S1_D];
    static int32_t gHd[S1_L_SEQ][S1_FFN];
    static int32_t gU[S1_L_SEQ][S1_FFN];
    static int32_t gX2v[S1_L_SEQ][S1_D];
    static int32_t gAcat[S1_L_SEQ][S1_D];
    static int32_t gQv[S1_L_SEQ][S1_D];
    static int32_t gKv[S1_L_SEQ][S1_D];
    static int32_t gVv[S1_L_SEQ][S1_D];
    static int32_t gX1v[S1_L_SEQ][S1_D];

    // recomputed forward intermediates needed by the wgrad/attn backward (X1 saved; Acat/Hd recomputed).
    static data_t X1m[S1_L_SEQ][S1_D];
    static data_t X2m[S1_L_SEQ][S1_D];
    static data_t Qm[S1_L_SEQ][S1_D];
    static data_t Km[S1_L_SEQ][S1_D];
    static data_t Vm[S1_L_SEQ][S1_D];
    static data_t Qh[S1_L_SEQ][S1_DH];
    static data_t Kh[S1_L_SEQ][S1_DH];
    static data_t Vh[S1_L_SEQ][S1_DH];
    static data_t Acatm[S1_L_SEQ][S1_D];
    static data_t Hdm[S1_L_SEQ][S1_FFN];
    static int32_t gOuth[S1_L_SEQ][S1_DH];
    static int32_t gQh[S1_L_SEQ][S1_DH];
    static int32_t gKh[S1_L_SEQ][S1_DH];
    static int32_t gVh[S1_L_SEQ][S1_DH];

BWD_LAYER: for (int lr = 0; lr < S1_N_LAYERS; lr++) {
        const int l = S1_N_LAYERS - 1 - lr;

        // === recompute the forward intermediates of layer l from its saved input residual R_l. ===
        // X1 = LN1(R_l) was saved; X2 = LN2(R_after_attn) was saved. Q/K/V = GEMM(X1); Acat from attn;
        // Hd = relu(GEMM(X2)). We need X1m/X2m (wgrad x-operand), Qm/Km/Vm + Acatm (attn + Wo wgrad),
        // Hdm (W_down wgrad). All BIT-IDENTICAL to the forward.
    LD_X1: for (int t = 0; t < S1_L_SEQ; t++) {
            // reload LN1/LN2 out rows for token t from DDR (was resident X1_oc/X2_oc).
            burst_load_i32(buffer, s1_off_act(S1_OFF_X1, l) + t * S1_TILE_RD, x1row_t, S1_TILE_RD);
            burst_load_i32(buffer, s1_off_act(S1_OFF_X2, l) + t * S1_TILE_RD, x2row_t, S1_TILE_RD);
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                X1m[t][d] = (data_t)x1row_t[d];
                X2m[t][d] = (data_t)x2row_t[d];
            }
        }
        fwd_gemm(&X1m[0][0], wq_oc, l * S1_D * S1_D, &Qm[0][0], S1_L_SEQ, S1_D, S1_D);
        fwd_gemm(&X1m[0][0], wk_oc, l * S1_D * S1_D, &Km[0][0], S1_L_SEQ, S1_D, S1_D);
        fwd_gemm(&X1m[0][0], wv_oc, l * S1_D * S1_D, &Vm[0][0], S1_L_SEQ, S1_D, S1_D);
    RC_HEAD: for (int h = 0; h < S1_H; h++) {
            static data_t Ahh[S1_L_SEQ][S1_DH];
            static int32_t j_dummy[S1_L_SEQ];
            for (int t = 0; t < S1_L_SEQ; t++)
                for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
                    Qh[t][d] = Qm[t][h * S1_DH + d];
                    Kh[t][d] = Km[t][h * S1_DH + d];
                    Vh[t][d] = Vm[t][h * S1_DH + d];
                }
            attn_head(Qh, Kh, Vh, at_lut, Ahh, buffer, -1, j_dummy);  // recompute only; p_off=-1 skips the prob save
            for (int t = 0; t < S1_L_SEQ; t++)
                for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
                    Acatm[t][h * S1_DH + d] = Ahh[t][d];
                }
        }
        // Hd = relu(GEMM(X2, W_up)) — recompute U then gate by the SAVED relu mask (bit-identical to fwd).
        static data_t Um[S1_L_SEQ][S1_FFN];
        fwd_gemm(&X2m[0][0], wup_oc, l * S1_FFN * S1_D, &Um[0][0], S1_L_SEQ, S1_D, S1_FFN);
    RC_RELU: for (int t = 0; t < S1_L_SEQ; t++) {
            // reload the relu mask row for token t from DDR (was resident relu_oc, the 114-BRAM hog).
            burst_load_i32(buffer, s1_off_relu(l) + t * S1_TILE_RELU, relurow_t, S1_TILE_RELU);
            for (int k = 0; k < S1_FFN; k++) {
#pragma HLS PIPELINE II=1
                Hdm[t][k] = (relurow_t[k] != 0) ? Um[t][k] : (data_t)0;
            }
        }

        // === residual2 split: gR feeds BOTH the FFN-down output (gDn) AND the skip (gR kept). ===
    RES2_BWD: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                gDn[t][d] = gR[t][d];          // gradient into the FFN branch output
            }

        // === FFN down-GEMM bwd: gHd = xgrad(gDn, W_down)  ; g_W_down += wgrad(gDn, Hd). ===
        xgrad_gemm(&gDn[0][0], wdn_oc, l * S1_D * S1_FFN, &gHd[0][0], S1_L_SEQ, S1_FFN, S1_D);
        wgrad_gemm(&gDn[0][0], &Hdm[0][0], gWDN_oc, l * S1_D * S1_FFN, S1_L_SEQ, S1_FFN, S1_D);

        // === ReLU bwd: gU = mask ? gHd : 0. ===
    RELU_BWD: for (int t = 0; t < S1_L_SEQ; t++) {
            // reload the relu mask row for token t from DDR (was resident relu_oc).
            burst_load_i32(buffer, s1_off_relu(l) + t * S1_TILE_RELU, relurow_t, S1_TILE_RELU);
            for (int k = 0; k < S1_FFN; k++) {
#pragma HLS PIPELINE II=1
                gU[t][k] = (relurow_t[k] != 0) ? gHd[t][k] : 0;
            }
        }

        // === FFN up-GEMM bwd: gX2 = xgrad(gU, W_up)  ; g_W_up += wgrad(gU, X2). ===
        xgrad_gemm(&gU[0][0], wup_oc, l * S1_FFN * S1_D, &gX2v[0][0], S1_L_SEQ, S1_D, S1_FFN);
        wgrad_gemm(&gU[0][0], &X2m[0][0], gWUP_oc, l * S1_FFN * S1_D, S1_L_SEQ, S1_D, S1_FFN);
    SV_GX2: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                gX2_oc[t][d] = gX2v[t][d];
            }

        // === LN2 backward: gY=gX2 -> gR_pre + g_LN2_gamma + g_LN2_beta. The gradient ADDS to gR (the
        //     residual skip already in gR), matching the forward R = R + Dn ; X2 = LN2(R). ===
    LN2_BWD: for (int t = 0; t < S1_L_SEQ; t++) {
            int e_n  = ln2st_oc[l * S1_L_SEQ * 2 + 2 * t];
            int nq_n = ln2st_oc[l * S1_L_SEQ * 2 + 2 * t + 1];
            // recompute r[d] (pre-gamma rescaled c) from the layer's pre-LN2 residual = R_after_attn,
            // which equals R_l-output-of-attn-residual. The forward fed LN2 from R post-RES1; that exact
            // value is X2's pre-norm source. We recompute c -> r from R_attn = (R_l + AO). But we only
            // need r for g_gamma's dyad(r) magnitude; r is BIT-IDENTICAL to the forward rescale of c.
            // Source residual for LN2 = the saved R after attn-residual. We reconstruct it below.
            // (handled by ln_backward_row which recomputes c->r from the given pre-norm row.)
            acc_t g_c[S1_D];
            acc_t gsum = 0;
        LN2_GSTE: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                bool gsn, gnz; exp_t ge;
                wdecode((int32_t)ln2g_oc[l * S1_D + d], gsn, ge, gnz);
                acc_t gchat = gnz ? ((acc_t)gX2v[t][d] << (int)ge) : (acc_t)0;
                if (gnz && gsn) gchat = -gchat;
                acc_t gd = s1_apply_invq(gchat, e_n, nq_n);
                g_c[d] = gd;
                gsum += gd;
            }
            acc_t mean = gsum >> S1_LOG2_D;
        LN2_GSUB: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                gR[t][d] = sat32((acc_t)gR[t][d] + (g_c[d] - mean));
                gLN2B_oc[l * S1_D + d] = sat32((acc_t)gLN2B_oc[l * S1_D + d] + (acc_t)gX2v[t][d]);
            }
        }
        // g_LN2_gamma[d] += sum_t gX2[t][d] * q(r2[t][d])  (r2 = pre-gamma rescaled c of LN2; dyadic -> shift).
        // recompute r2 from the pre-LN2 residual R_attn = R_l + AO. R_l is R_all[l]; AO = Acat@Wo (=R_attn-R_l).
        // We reconstruct R_attn as R_all[l] residual stream BEFORE the FFN residual add but AFTER attn:
        //   R_attn[t][d] = sat16(R_all[l][t][d] + AO[t][d]).  AO = fwd_gemm(Acat, Wo).
        static data_t AOm[S1_L_SEQ][S1_D];
        static data_t Rattn[S1_L_SEQ][S1_D];
        fwd_gemm(&Acatm[0][0], wo_oc, l * S1_D * S1_D, &AOm[0][0], S1_L_SEQ, S1_D, S1_D);
    RC_RATTN: for (int t = 0; t < S1_L_SEQ; t++) {
            // reload R_l snapshot row t from DDR (was resident R_all_oc[l]).
            burst_load_i32(buffer, s1_off_R(l) + t * S1_TILE_RD, rrow_t, S1_TILE_RD);
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                Rattn[t][d] = sat16((acc_t)rrow_t[d] + (acc_t)AOm[t][d]);
            }
        }
        ln_gamma_grad(Rattn, ln2st_oc, l, gX2v, gLN2G_oc, l * S1_D);

        // === residual1 split: gR feeds BOTH the attn-output (Wo) branch AND the skip. ===
        static int32_t gAO[S1_L_SEQ][S1_D];
    RES1_BWD: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                gAO[t][d] = gR[t][d];
            }

        // === Wo bwd: gAcat = xgrad(gAO, Wo) ; g_Wo += wgrad(gAO, Acat). ===
        xgrad_gemm(&gAO[0][0], wo_oc, l * S1_D * S1_D, &gAcat[0][0], S1_L_SEQ, S1_D, S1_D);
        wgrad_gemm(&gAO[0][0], &Acatm[0][0], gWO_oc, l * S1_D * S1_D, S1_L_SEQ, S1_D, S1_D);

        // === attention backward per head -> gQ/gK/gV (assembled into gQv/gKv/gVv over the heads). ===
    ATTN_BWD_H: for (int h = 0; h < S1_H; h++) {
            for (int t = 0; t < S1_L_SEQ; t++)
                for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
                    Qh[t][d] = Qm[t][h * S1_DH + d];
                    Kh[t][d] = Km[t][h * S1_DH + d];
                    Vh[t][d] = Vm[t][h * S1_DH + d];
                    gOuth[t][d] = gAcat[t][h * S1_DH + d];
                    gQh[t][d] = 0; gKh[t][d] = 0; gVh[t][d] = 0;
                }
            attn_head_bwd(Qh, Kh, Vh, gOuth, at_lut, gQh, gKh, gVh);
            for (int t = 0; t < S1_L_SEQ; t++)
                for (int d = 0; d < S1_DH; d++) {
#pragma HLS PIPELINE II=1
                    gQv[t][h * S1_DH + d] = gQh[t][d];
                    gKv[t][h * S1_DH + d] = gKh[t][d];
                    gVv[t][h * S1_DH + d] = gVh[t][d];
                }
        }

        // === Q/K/V bwd: gX1 = xgrad(gQ,Wq) + xgrad(gK,Wk) + xgrad(gV,Wv) (ACCUMULATE the 3 paths);
        //     g_Wq/g_Wk/g_Wv += wgrad(gQ/gK/gV, X1). ===
        static int32_t gX1q[S1_L_SEQ][S1_D];
        static int32_t gX1k[S1_L_SEQ][S1_D];
        static int32_t gX1vv[S1_L_SEQ][S1_D];
        xgrad_gemm(&gQv[0][0], wq_oc, l * S1_D * S1_D, &gX1q[0][0],  S1_L_SEQ, S1_D, S1_D);
        xgrad_gemm(&gKv[0][0], wk_oc, l * S1_D * S1_D, &gX1k[0][0],  S1_L_SEQ, S1_D, S1_D);
        xgrad_gemm(&gVv[0][0], wv_oc, l * S1_D * S1_D, &gX1vv[0][0], S1_L_SEQ, S1_D, S1_D);
        wgrad_gemm(&gQv[0][0], &X1m[0][0], gWQ_oc, l * S1_D * S1_D, S1_L_SEQ, S1_D, S1_D);
        wgrad_gemm(&gKv[0][0], &X1m[0][0], gWK_oc, l * S1_D * S1_D, S1_L_SEQ, S1_D, S1_D);
        wgrad_gemm(&gVv[0][0], &X1m[0][0], gWV_oc, l * S1_D * S1_D, S1_L_SEQ, S1_D, S1_D);
    ACC_GX1: for (int t = 0; t < S1_L_SEQ; t++)
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                gX1v[t][d] = sat32((acc_t)gX1q[t][d] + (acc_t)gX1k[t][d] + (acc_t)gX1vv[t][d]);
                gX1_oc[t][d] = gX1v[t][d];
            }

        // === LN1 backward: gY=gX1 -> gR + g_LN1_gamma + g_LN1_beta. ADDS to gR (the residual1 skip). ===
    LN1_BWD: for (int t = 0; t < S1_L_SEQ; t++) {
            int e_n  = ln1st_oc[l * S1_L_SEQ * 2 + 2 * t];
            int nq_n = ln1st_oc[l * S1_L_SEQ * 2 + 2 * t + 1];
            acc_t g_c[S1_D];
            acc_t gsum = 0;
        LN1_GSTE: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                bool gsn, gnz; exp_t ge;
                wdecode((int32_t)ln1g_oc[l * S1_D + d], gsn, ge, gnz);
                acc_t gchat = gnz ? ((acc_t)gX1v[t][d] << (int)ge) : (acc_t)0;
                if (gnz && gsn) gchat = -gchat;
                acc_t gd = s1_apply_invq(gchat, e_n, nq_n);
                g_c[d] = gd;
                gsum += gd;
            }
            acc_t mean = gsum >> S1_LOG2_D;
        LN1_GSUB: for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                gR[t][d] = sat32((acc_t)gR[t][d] + (g_c[d] - mean));
                gLN1B_oc[l * S1_D + d] = sat32((acc_t)gLN1B_oc[l * S1_D + d] + (acc_t)gX1v[t][d]);
            }
        }
        // g_LN1_gamma[d] += sum_t gX1[t][d] * q(r1[t][d]); r1 from the pre-LN1 residual = R_all[l] (block input).
        static data_t Rin1[S1_L_SEQ][S1_D];
    RC_RIN1: for (int t = 0; t < S1_L_SEQ; t++) {
            // reload R_l snapshot row t from DDR (was resident R_all_oc[l]).
            burst_load_i32(buffer, s1_off_R(l) + t * S1_TILE_RD, rrow_t, S1_TILE_RD);
            for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
                Rin1[t][d] = (data_t)rrow_t[d];
            }
        }
        ln_gamma_grad(Rin1, ln1st_oc, l, gX1v, gLN1G_oc, l * S1_D);
    }

    // ---------------------------------------------------------------------
    // (3) embed backward (SPARSE): g_EMBED[tok[t]*D + d] += gR[t][d].
    //     POS backward (S1_POS_EMBED): g_POS[t*D + d] = gR[t][d] — the SAME residual-input grad. Unlike EMBED
    //     (which scatters into the shared token rows, so it accumulates), each (t,d) maps to a UNIQUE POS row,
    //     so it is a direct write (no accumulation across positions). Operand-free (a copy of the residual grad).
    // ---------------------------------------------------------------------
EMB_BWD: for (int t = 0; t < S1_L_SEQ; t++) {
        int tok = toks[t];
        if (tok < 0) tok = 0; if (tok >= S1_V) tok = S1_V - 1;
        for (int d = 0; d < S1_D; d++) {
#pragma HLS PIPELINE II=1
            if (gR[t][d] != 0) gEMBED_oc[tok * S1_D + d] = sat32((acc_t)gEMBED_oc[tok * S1_D + d] + (acc_t)gR[t][d]);
#if S1_POS_EMBED
            gPOS_oc[t * S1_D + d] = gR[t][d];
#endif
        }
    }

    // ---------------------------------------------------------------------
    // (D) BURST-STORE every gradient region + the intermediate grads (for the bit-check).
    // ---------------------------------------------------------------------
    burst_store_i32(gEMBED_oc, buffer, S1_OFF_G_EMBED,  S1_W_EMBED);
    burst_store_i32(gWQ_oc,    buffer, S1_OFF_G_WQ,     S1_W_PROJ);
    burst_store_i32(gWK_oc,    buffer, S1_OFF_G_WK,     S1_W_PROJ);
    burst_store_i32(gWV_oc,    buffer, S1_OFF_G_WV,     S1_W_PROJ);
    burst_store_i32(gWO_oc,    buffer, S1_OFF_G_WO,     S1_W_PROJ);
    burst_store_i32(gWUP_oc,   buffer, S1_OFF_G_W_UP,   S1_W_WUP);
    burst_store_i32(gWDN_oc,   buffer, S1_OFF_G_W_DOWN, S1_W_WDOWN);
    burst_store_i32(gLN1G_oc,  buffer, S1_OFF_G_LN1_G,  S1_W_LN);
    burst_store_i32(gLN1B_oc,  buffer, S1_OFF_G_LN1_B,  S1_W_LN);
    burst_store_i32(gLN2G_oc,  buffer, S1_OFF_G_LN2_G,  S1_W_LN);
    burst_store_i32(gLN2B_oc,  buffer, S1_OFF_G_LN2_B,  S1_W_LN);
#if S1_USE_LNF
    burst_store_i32(gLNFG_oc,  buffer, S1_OFF_G_LNF_G,  S1_D);
    burst_store_i32(gLNFB_oc,  buffer, S1_OFF_G_LNF_B,  S1_D);
#endif
    burst_store_i32(gWHEAD_oc, buffer, S1_OFF_G_W_HEAD, S1_W_WHEAD);
#if S1_POS_EMBED
    burst_store_i32(gPOS_oc,   buffer, S1_OFF_G_POS,    S1_POS_N);
#endif
    burst_store_i32(&gR[0][0],     buffer, S1_OFF_GR,    S1_L_SEQ * S1_D);
    burst_store_i32(&gX1_oc[0][0], buffer, S1_OFF_GX1,   S1_L_SEQ * S1_D);
    burst_store_i32(&gX2_oc[0][0], buffer, S1_OFF_GX2,   S1_L_SEQ * S1_D);

    return 0;
}

// =============================================================================
// ======================  MODE 2 (DYADIC-ADAM OPT) — do_opt  ==================
// =============================================================================
// The multiplier-free ADAPTIVE optimizer (forge_dyadic_adam.cpp recipe, == charlm dyadic-Adam that
// reaches 2.481 at this config). QAT master-weight scheme: the optimizer accumulates the Adam step into
// a FULL-PRECISION int16 master W_fp, then RE-QUANTIZES it into the int8 dyadic CODE (or copies it to the
// int16 VALUE) the forward reads. EVERYTHING is shift + add + compare — NO multiply, NO DSP (DSP=0).
//
//   g    = grad[p]                                  (int32, from mode 1)
//   m    = m - (m>>B1) + (g>>B1)                     (EMA, beta1 = 1 - 2^-B1)   first moment
//   v    = v - (v>>B2) + (|g|>>B2)                   (EMA, beta2 = 1 - 2^-B2)   adaptive scale
//   se   = round(log2 v)                             (priority-encoder MSB + round-to-nearest pow2)
//   step = m >> se                                   (arithmetic barrel shift -> per-param 1/|g|_EMA pow2)
//   W_fp = sat16( W_fp - (step >> LR_SHIFT) )        (lr = 2^-LR_SHIFT; a shift — no divide, no multiply)
//   code = s1_encode(dyad(W_fp))  [is_code]   |   value = W_fp  [!is_code]      (re-quantize the master)
//   g    = 0                                         (zero the grad region after applying)
//
// STREAMING REGION-RMW (the forge_dyadic_adam / morph signsgd_region_wide idiom): mode 2 walks each
// learnable region ONCE in LANE-wide chunks, read-modify-writing W_fp/m/v in place + writing the
// re-quantized code/value + zeroing g. This is a per-element map (no contraction, no fan-in) so it needs
// only a tiny on-chip tile — near-ZERO BRAM (does NOT add to the mode-1 68% BRAM baseline) and a narrow
// engine. Bit-exact to the int64 CPU dyadic-Adam reference (same shifts/betas/round rule).
// =============================================================================

// dyadic exponent of an int32 magnitude: round(log2|v|) via MSB + the next bit (round-to-nearest).
// VERBATIM forge_dyadic_adam.cpp dyad_exp(). Returns 0 for v==0 (a zero-scale param -> step = m>>0 = m).
// Clamp the bumped exponent to 31 so the (m >> se) shift amount stays < 32 (a >> by the full word width
// is UB on the scalar int32 #else path and adds nothing — dividing m by 2^31 already drives step to 0/-1).
static inline int s1_dyad_exp(acc_t v) {
#pragma HLS INLINE
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    ap_uint<32> a = (v < 0) ? (ap_uint<32>)(acc_t)(-v) : (ap_uint<32>)(acc_t)v;
#else
    uint32_t a = (v < 0) ? (uint32_t)(-(int64_t)v) : (uint32_t)v;
#endif
    int msb = 0;
DA_PRIO: for (int b = 31; b >= 1; b--) {
#pragma HLS UNROLL
        if ((a >> b) & 1) { msb = b; break; }
    }
    if (msb >= 1 && ((a >> (msb - 1)) & 1) && msb < 31) msb = msb + 1;   // round-to-nearest pow2
    return msb;
}

// re-quantize an int16 master weight to the PRE-DYADIC byte CODE the forward decodes (wdecode inverse):
// dyad(W_fp) -> (sign, exp, nz); pack bit7=sign, bit5=nonzero, bits4..0=exp. The exponent is the SAME
// round(log2|W_fp|) the GEMM consumes (self-consistent: wdecode(s1_encode(dyad(w))) == dyad(w)). Returns
// the byte code as a sign-extended int32 lane (forge writes the code regions one-per-lane).
static inline int32_t s1_encode(data_t wfp) {
#pragma HLS INLINE
    bool sgn, nz; exp_t e;
    dyad(wfp, sgn, e, nz);
    int ee = (int)e;
    if (ee > 31) ee = 31;            // 5-bit exponent field clamp (int16 dyad exp <= 16 in practice)
    int c = (ee & 0x1F);
    if (nz)  c |= (1 << 5);
    if (sgn) c |= (1 << 7);
    return (int32_t)(int8_t)(unsigned char)c;   // store as a sign-extended byte (== burst_load_codes read)
}

// re-quantize a FINE FIXED-POINT code master (the TRAIN_ALL fix). wq is the int32 master at scale 2^Q_CODE
// (weight value = wq / 2^Q_CODE). Round-to-nearest to the integer code value, dyad() it, then optionally
// CLAMP the per-step exponent change to +/-1 relative to the previous deploy code (S1_CODE_EXP_CLAMP) so no
// single step can runaway-double a weight. prev_code is the byte code the forward read BEFORE this step
// (decoded for its sign/nz/exp). The result is BIT-IDENTICAL integer arithmetic in kernel + golden.
// rwq  = round(wq / 2^Q_CODE) with round-half-away-from-zero (symmetric, sign-stable):
//        |rwq| = (|wq| + 2^(Q-1)) >> Q ; sign from wq. (Q_CODE>=1 so the half-bias 2^(Q-1) is exact.)
static inline int32_t s1_encode_fine(acc_t wq, int32_t prev_code, bool allow_clamp) {
#pragma HLS INLINE
    acc_t aw = (wq < 0) ? (acc_t)(-wq) : wq;
    acc_t half = (acc_t)1 << (S1_Q_CODE - 1);
    acc_t mag = (aw + half) >> S1_Q_CODE;            // rounded integer code magnitude
    data_t wr = (wq < 0) ? (data_t)(-(acc_t)sat16(mag)) : (data_t)sat16(mag);
    bool sgn, nz; exp_t e;
    dyad(wr, sgn, e, nz);                            // dyad() the rounded integer code value
    int ee = (int)e;
#if S1_CODE_EXP_CLAMP
    // damping safety: limit |Δexp| <= 1 per step (and entering-from-zero starts at the floor exp 0).
    // SKIPPED on init (allow_clamp=false): init must produce the true dyad(round(master)) — there is no
    // meaningful "previous" deploy code yet (the host only staged the master).
    if (allow_clamp) {
        int pe; int pnz;
        { unsigned pc = (unsigned)(prev_code & 0xFF); pnz = (pc >> 5) & 1; pe = (int)(pc & 0x1F); }
        if (nz) {
            if (pnz) {                               // prev nonzero: stay within +/-1 of its exponent
                if (ee > pe + 1) ee = pe + 1;
                if (ee < pe - 1) ee = pe - 1;
                if (ee < 0) ee = 0;
            } else {                                 // climbing off zero: enter at the smallest dyadic (2^0)
                ee = 0;
            }
        }
    }
#endif
    if (ee > 31) ee = 31;
    int c = (ee & 0x1F);
    if (nz)  c |= (1 << 5);
    if (sgn) c |= (1 << 7);
    return (int32_t)(int8_t)(unsigned char)c;
}

// one dyadic-Adam step over ONE region [n params, contiguous]. STREAMING RMW in LANE-wide chunks:
//   load W_fp/g/m/v tile -> per-lane Adam update -> store W_fp/m/v + the re-quantized code/value + g=0.
// init_only=1: grad is treated as ZERO (Adam is a no-op: m,v,W_fp unchanged) and ONLY the re-quantize
// runs — the code-init path (populate codes/values from the staged W_fp without moving the weights).
//
// TIMING SPLIT (csynth named the critical path as two priority encoders in series in ONE comb stage:
// s1_dyad_exp(vn) THEN dyad() inside s1_encode_fine — 4.266 ns > 4.0 ns target). The element work is split
// into TWO sequential II=1 passes so neither pass holds both encoders:
//   PASS A (ADAM_R): load g/m/v/W_fp -> m,v EMA -> se=s1_dyad_exp(vn) [encoder #1] -> step -> dw -> W_fp_new
//                    (+weight decay) -> STORE W_fp_new, m, v, zero grad. For VALUE regions the deploy slot
//                    IS W_fp_new (no re-quantize), so it is written here too.
//   PASS B (ADAM_Q): CODE regions ONLY — load W_fp_new -> s1_encode_fine(dyad(W_fp_new)) [encoder #2] with
//                    the |Δexp|<=1 clamp vs the prev deploy code -> STORE the deploy code.
// The two passes are BIT-EXACT to the original single pass: PASS A writes exactly the int32 W_fp value the
// single pass formerly fed straight into s1_encode_fine, and PASS B reads it back. The elements are
// independent (each p touches distinct buffer offsets; PASS B reads R.wfp_off written by PASS A and the
// untouched R.dst_off prev code), so II=1 holds for both passes. On init_only PASS A leaves W_fp/m/v/g
// untouched (Adam no-op) and PASS B re-quantizes from the staged W_fp — identical to the old init path.
static void dyadic_adam_region(int32_t *buffer, const s1_opt_region_t &R, bool init_only) {
#pragma HLS INLINE off
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    int n = R.n;
    // FUSED WIDE READ+WRITE (the morph signsgd_region_wide pattern, proven to COMMIT on the F2 shell). The
    // prior TILED-STATIC version delegated every write to load->static-tile->store ROUND-TRIPS on the same
    // gmem offsets (the wfp/m/v/dst regions read back by do_fwd_loss through a DIFFERENT limit=1 cast); the
    // Vitis scheduler could not prove the cross-instance RAW dependence and DROPPED the writeback on silicon
    // (csim/csynth pass, mode-2 no-ops -> live regions left ZERO). The fix mirrors morph EXACTLY: per 512-bit
    // wide word, READ the needed wide words via reinterpret_cast<wide_t*>, compute all 16 lanes with FIXED
    // per-lane .range() shifts (NO variable barrel shift, NO static staging arrays), and WRITE the updated
    // wide words back IN THE SAME LOOP BODY. The two-pass split is kept ONLY to separate the two priority
    // encoders (s1_dyad_exp in PASS A, dyad-in-s1_encode_fine in PASS B); EACH pass is itself a fused wide
    // read+write with no dead separate store. PASS B reads R.wfp_off — the SAME region PASS A just wrote in
    // its own fused body — so it sees W_fp_new (the in-region RAW the scheduler CAN prove: same offset, same
    // wide cast, same function instance). BIT-EXACT to the #else per-element golden: the per-lane integer
    // arithmetic and the FIXED [i] -> lane mapping are verbatim; only the staging is removed.
    wide_t *wide_buf = reinterpret_cast<wide_t *>(buffer);
    int n_wide      = n >> 4;
    int wfp_wide    = R.wfp_off  >> 4;
    int g_wide      = R.g_off    >> 4;
    int m_wide      = R.m_off    >> 4;
    int v_wide      = R.v_off    >> 4;
    int dst_wide    = R.dst_off  >> 4;

    // ---- PASS A (fused): Adam update -> W_fp_new (one priority encoder: s1_dyad_exp). Reads wfp (+g/m/v on
    //      a real step), writes ONLY the optimizer STATE (wfp/m/v/g) in this body. ----
    // STEP WRITEBACK FIX: the VALUE-region deploy slot (dst) is NO LONGER written here on a STEP. The old
    // STEP body wrote FIVE wide_buf slots per iteration (wfp/m/v/g/dst); the INIT body that COMMITS on
    // silicon writes ONLY ONE (dst), and the morph signsgd_region_wide that COMMITS writes TWO (w/grad-zero).
    // The 5th write (dst) — whose consumer is a FUTURE kick's do_fwd_loss reading the deploy region back
    // through a DIFFERENT reinterpret_cast (limit=1) — is the one the RTL scheduler could not prove live and
    // DROPPED under the 5-way per-body write fan-out, so the STEP was a silicon NO-OP (weights never moved)
    // while csim/csynth/INIT passed. THE FIX: the wfp/m/v/g writes here are in-instance RAW (do_opt itself
    // reads m/v/wfp back next kick -> the scheduler proves them live, so PASS A's state writes already
    // commit), and the VALUE deploy slot is moved to PASS A2 below — a SEPARATE fused pass with the EXACT
    // single-read(wfp)+single-write(dst) shape that the committing INIT path and PASS B use. Bit-exact: PASS
    // A2 copies the SAME (int16)wn that PASS A wrote into wfp_new (==the deploy value), so the deploy slot is
    // identical to the old in-body write — only the SCHEDULE/structure changes to the proven-committing form.
ADAM_RW: for (int wi = 0; wi < n_wide; wi++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        wide_t wfp_word = wide_buf[wfp_wide + wi];
        wide_t g_word   = init_only ? (wide_t)0 : wide_buf[g_wide + wi];
        wide_t m_word   = init_only ? (wide_t)0 : wide_buf[m_wide + wi];
        wide_t v_word   = init_only ? (wide_t)0 : wide_buf[v_wide + wi];
        wide_t wfp_new  = wfp_word;
        wide_t m_new    = m_word;
        wide_t v_new    = v_word;
        wide_t g_new    = g_word;
        wide_t dst_new  = 0;   // VALUE deploy slot — written here on INIT; on STEP it is written by PASS A2
                               // (the single-write committing shape); CODE dst is set by PASS B.
RWA_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
            int hi = lane * 32 + 31, lo = lane * 32;
            if (init_only) {
                // Adam no-op on init: W_fp/m/v/g untouched (the unread words above stayed = the source word).
                // CODE regions re-quantize the staged W_fp in PASS B; VALUE regions populate the deploy slot
                // here from the staged master (== old init: dst = (int16)wfp, unchanged weights).
                if (!R.is_code) {
                    data_t wfp = (data_t)(int16_t)(int32_t)(ap_int<32>)wfp_word.range(hi, lo);
                    dst_new.range(hi, lo) = (ap_uint<32>)(ap_int<32>)(int32_t)(int16_t)wfp;
                }
                continue;
            }
            acc_t g = (acc_t)(int32_t)(ap_int<32>)g_word.range(hi, lo);
            acc_t m = (acc_t)(int32_t)(ap_int<32>)m_word.range(hi, lo);
            acc_t v = (acc_t)(int32_t)(ap_int<32>)v_word.range(hi, lo);
            acc_t absg = (g < 0) ? (acc_t)(-g) : g;
            acc_t mn = m - (m >> S1_ADAM_B1) + (g    >> S1_ADAM_B1);
            acc_t vn = v - (v >> S1_ADAM_B2) + (absg >> S1_ADAM_B2);
            int se = s1_dyad_exp(vn);                    // PRIORITY ENCODER #1 (alone in this stage)
            acc_t step = mn >> se;                       // m / 2^round(log2 v)  (arith shift, keeps sign)
            if (R.is_code) {
                // FINE FIXED-POINT code master (int32 at scale 2^Q_CODE): nudge by a small fraction of a code
                // unit; the deploy code is re-quantized from W_fp_new in PASS B (encoder #2).
                acc_t wq  = (acc_t)(int32_t)(ap_int<32>)wfp_word.range(hi, lo);
                acc_t dw  = step >> S1_LR_SHIFT_CODE;    // fine lr = 2^-LR_SHIFT_CODE of a code unit
#if S1_CODE_WD_SHIFT
                acc_t wd  = wq >> S1_CODE_WD_SHIFT;      // multiplier-free weight decay (shift toward 0)
#else
                acc_t wd  = 0;
#endif
                acc_t wn  = (acc_t)sat32((acc_t)wq - dw - wd);
                m_new.range(hi, lo)   = (ap_uint<32>)(ap_int<32>)(int32_t)mn;
                v_new.range(hi, lo)   = (ap_uint<32>)(ap_int<32>)(int32_t)vn;
                wfp_new.range(hi, lo) = (ap_uint<32>)(ap_int<32>)(int32_t)wn;   // full int32 fixed-point master
                g_new.range(hi, lo)   = 0;
            } else {
                // VALUE master (EMBED/W_HEAD/LN beta): int16 unit scale, coarse LR_SHIFT. No re-quantize: the
                // deploy slot IS W_fp_new — but it is written by PASS A2 (the single-write committing shape),
                // NOT here (see the STEP WRITEBACK FIX comment above; the in-body dst write was dropped on RTL).
                data_t wfp = (data_t)(int16_t)(int32_t)(ap_int<32>)wfp_word.range(hi, lo);
                acc_t dw   = step >> S1_LR_SHIFT;        // lr = 2^-LR_SHIFT
                data_t wn  = sat16((acc_t)wfp - dw);
                m_new.range(hi, lo)   = (ap_uint<32>)(ap_int<32>)(int32_t)mn;
                v_new.range(hi, lo)   = (ap_uint<32>)(ap_int<32>)(int32_t)vn;
                wfp_new.range(hi, lo) = (ap_uint<32>)(ap_int<32>)(int32_t)(int16_t)wn;
                g_new.range(hi, lo)   = 0;
            }
        }
        // WRITE-BACK in the SAME body (the morph fused commit).
        if (init_only) {
            // INIT: the VALUE deploy slot is the SOLE write here (the proven-committing single-write shape,
            // same as morph/PASS B). CODE deploy is re-quantized by PASS B. Optimizer state untouched (no-op).
            if (!R.is_code) wide_buf[dst_wide + wi] = dst_new;
        } else {
            // STEP: write ONLY the optimizer STATE (in-instance RAW: do_opt reads m/v/wfp back next kick, so
            // the scheduler proves these live and they commit). The VALUE deploy slot is written by PASS A2
            // (single-write shape) and the CODE deploy by PASS B — NOT here (the STEP WRITEBACK FIX).
            wide_buf[wfp_wide + wi] = wfp_new;
            wide_buf[m_wide   + wi] = m_new;
            wide_buf[v_wide   + wi] = v_new;
            wide_buf[g_wide   + wi] = g_new;             // grads zeroed per lane above
        }
    }
    // ---- PASS A2 (fused, STEP + VALUE regions only): write the VALUE deploy slot from W_fp_new in the EXACT
    //      single-read(wfp)+single-write(dst) shape that the committing INIT path and PASS B use. This is the
    //      STEP WRITEBACK FIX: the deploy write — whose consumer is a future kick's forward read-back — is now
    //      structurally identical to the INIT dst write that COMMITS on silicon (one read, one write per body),
    //      instead of being the 5th write in PASS A's body (which the RTL scheduler dropped). Skipped on init
    //      (PASS A already wrote the VALUE deploy slot above) and on CODE regions (PASS B handles their dst).
    //      BIT-EXACT: dst = (int16)W_fp_new == the value PASS A formerly wrote in-body. ----
    if (!init_only && !R.is_code) {
ADAM_VW: for (int wi = 0; wi < n_wide; wi++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
            wide_t wfp_word = wide_buf[wfp_wide + wi];   // W_fp_new from PASS A (same offset/cast/instance)
            wide_t dst_new  = 0;
VWA_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
                int hi = lane * 32 + 31, lo = lane * 32;
                data_t wn = (data_t)(int16_t)(int32_t)(ap_int<32>)wfp_word.range(hi, lo);
                dst_new.range(hi, lo) = (ap_uint<32>)(ap_int<32>)(int32_t)(int16_t)wn;
            }
            wide_buf[dst_wide + wi] = dst_new;           // fused single-write commit (the morph/INIT shape)
        }
    }
    // ---- PASS B (fused): re-quantize CODE regions W_fp_new -> deploy code (one priority encoder: dyad in
    //      s1_encode_fine). Reads R.wfp_off (the SAME region PASS A wrote -> sees W_fp_new) + the prev deploy
    //      code (for the exp-clamp), writes the deploy code IN THIS BODY. ----
    if (R.is_code) {
ADAM_QW: for (int wi = 0; wi < n_wide; wi++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
            wide_t wfp_word  = wide_buf[wfp_wide + wi];   // W_fp_new from PASS A (or staged W_fp on init)
            wide_t prev_word = wide_buf[dst_wide + wi];   // prev deploy code (exp-clamp reference)
            wide_t dst_new   = 0;
RWB_LANE: for (int lane = 0; lane < 16; lane++) {
#pragma HLS UNROLL
                int hi = lane * 32 + 31, lo = lane * 32;
                acc_t wn = (acc_t)(int32_t)(ap_int<32>)wfp_word.range(hi, lo);
                int32_t prev = (int32_t)(ap_int<32>)prev_word.range(hi, lo);
                dst_new.range(hi, lo) =
                    (ap_uint<32>)(ap_int<32>)s1_encode_fine(wn, prev, !init_only);  // PRIORITY ENCODER #2
            }
            wide_buf[dst_wide + wi] = dst_new;            // fused commit
        }
    }
    // scalar tail for any region whose logical extent is not a wide multiple (all S1 regions ARE 16-aligned,
    // so this is a no-op in the shipped config; kept for safety, BIT-IDENTICAL per-element math to PASS A/B).
    for (int p = (n_wide << 4); p < n; p++) {
        int32_t g = init_only ? 0 : buffer[R.g_off + p];
        int32_t m = buffer[R.m_off + p];
        int32_t v = buffer[R.v_off + p];
        acc_t absg = g < 0 ? (acc_t)(-(acc_t)g) : (acc_t)g;
        acc_t mn = (acc_t)m - ((acc_t)m >> S1_ADAM_B1) + ((acc_t)g    >> S1_ADAM_B1);
        acc_t vn = (acc_t)v - ((acc_t)v >> S1_ADAM_B2) + (absg        >> S1_ADAM_B2);
        int se = s1_dyad_exp(vn);
        acc_t step = mn >> se;
        if (R.is_code) {
            int32_t wq = buffer[R.wfp_off + p];
            acc_t dw = step >> S1_LR_SHIFT_CODE;
#if S1_CODE_WD_SHIFT
            acc_t wd = (acc_t)wq >> S1_CODE_WD_SHIFT;
#else
            acc_t wd = 0;
#endif
            int32_t wn = init_only ? wq : (int32_t)sat32((acc_t)wq - dw - wd);
            if (!init_only) {
                buffer[R.m_off + p]   = (int32_t)mn;
                buffer[R.v_off + p]   = (int32_t)vn;
                buffer[R.wfp_off + p] = wn;
                buffer[R.g_off + p]   = 0;
            }
            buffer[R.dst_off + p] = s1_encode_fine((acc_t)wn, buffer[R.dst_off + p], !init_only);
        } else {
            int16_t wfp = (int16_t)(buffer[R.wfp_off + p] & 0xFFFF);
            acc_t dw  = step >> S1_LR_SHIFT;
            int16_t wn = init_only ? wfp : (int16_t)sat16((acc_t)wfp - dw);
            if (!init_only) {
                buffer[R.m_off + p]   = (int32_t)mn;
                buffer[R.v_off + p]   = (int32_t)vn;
                buffer[R.wfp_off + p] = (int32_t)(int16_t)wn;
                buffer[R.g_off + p]   = 0;
            }
            buffer[R.dst_off + p] = (int32_t)(int16_t)wn;
        }
    }
#else
    int n = R.n;
    for (int p = 0; p < n; p++) {
        int32_t g = init_only ? 0 : buffer[R.g_off + p];
        int32_t m = buffer[R.m_off + p];
        int32_t v = buffer[R.v_off + p];
        int64_t absg = g < 0 ? -(int64_t)g : g;
        int64_t mn = (int64_t)m - ((int64_t)m >> S1_ADAM_B1) + ((int64_t)g    >> S1_ADAM_B1);
        int64_t vn = (int64_t)v - ((int64_t)v >> S1_ADAM_B2) + (absg          >> S1_ADAM_B2);
        int se = s1_dyad_exp((acc_t)vn);
        int64_t step = mn >> se;
        if (R.is_code) {
            int32_t wq = buffer[R.wfp_off + p];                  // full int32 fixed-point master (scale 2^Q)
            int64_t dw = step >> S1_LR_SHIFT_CODE;
#if S1_CODE_WD_SHIFT
            int64_t wd = (int64_t)wq >> S1_CODE_WD_SHIFT;        // multiplier-free weight decay (shift toward 0)
#else
            int64_t wd = 0;
#endif
            int32_t wn = init_only ? wq : (int32_t)sat32((acc_t)((int64_t)wq - dw - wd));
            if (!init_only) {
                buffer[R.m_off + p]   = (int32_t)mn;
                buffer[R.v_off + p]   = (int32_t)vn;
                buffer[R.wfp_off + p] = wn;
                buffer[R.g_off + p]   = 0;
            }
            buffer[R.dst_off + p] = s1_encode_fine((acc_t)wn, buffer[R.dst_off + p], !init_only);
        } else {
            int16_t wfp = (int16_t)(buffer[R.wfp_off + p] & 0xFFFF);
            int64_t dw  = step >> S1_LR_SHIFT;
            int16_t wn  = init_only ? wfp : (int16_t)sat16((acc_t)((int64_t)wfp - dw));
            if (!init_only) {
                buffer[R.m_off + p]   = (int32_t)mn;
                buffer[R.v_off + p]   = (int32_t)vn;
                buffer[R.wfp_off + p] = (int32_t)(int16_t)wn;
                buffer[R.g_off + p]   = 0;
            }
            buffer[R.dst_off + p] = (int32_t)(int16_t)wn;
        }
    }
#endif
}

// zero a region's grad in LANE-wide chunks (used when a region is FROZEN: we still drop its grad so the
// next step starts clean, but we do NOT move its master / re-quantize its code). Streaming RMW, II=1.
static void zero_grad_region(int32_t *buffer, const s1_opt_region_t &R) {
#pragma HLS INLINE off
    int n = R.n;
#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    // TILED ON-CHIP ZERO — same anti-512-bit-variable-shift pattern: build a zero TILE on-chip (sequential
    // II=1 writes, no variable-offset gmem), then burst-store it to the grad region in wide AXI chunks. A
    // direct buffer[R.g_off + p] = 0 would be a 512-bit RMW + variable barrel shift (the pole we removed).
    static int32_t zero_t[S1_OPT_TILE];
#pragma HLS BIND_STORAGE variable=zero_t type=ram_2p impl=bram
ZGR_INIT: for (int i = 0; i < S1_OPT_TILE; i++) {
#pragma HLS PIPELINE II=1
        zero_t[i] = 0;
    }
ZGR_TILE: for (int tbase = 0; tbase < n; tbase += S1_OPT_TILE) {
        int tn = n - tbase; if (tn > S1_OPT_TILE) tn = S1_OPT_TILE;
        burst_store_i32(zero_t, buffer, R.g_off + tbase, tn);
    }
#else
ZGR: for (int base = 0; base < n; base += S1_OPT_LANE) {
        for (int lane = 0; lane < S1_OPT_LANE; lane++) {
            int p = base + lane;
            if (p < n) buffer[R.g_off + p] = 0;
        }
    }
#endif
}

// =============================================================================
// Mode 2: do_opt — dyadic-Adam over the learnable regions (the S1_OPT_REGIONS table) + re-quantize the
// forward-read codes/values + zero the grads.
//   a1 = S1_OPT_STEP (full step) | S1_OPT_INIT (re-quantize the codes/values from the staged W_fp ONLY —
//        the code-init path; Adam is a no-op).
//   freeze_codes (mode-2 a2; S1_OPT_TRAIN_ALL=0 | S1_OPT_FREEZE_CODES=1): when set, the dense PRE-DYADIC
//        CODE weights (is_code=1: Wq/Wk/Wv/Wo/W_up/W_down/LN1_g/LN2_g/LN_F_g) are NOT updated and NOT
//        re-quantized — grad dropped — so only the SCATTER/VALUE weights (EMBED, W_HEAD, LN beta) train.
//        The conservative always-stable QAT baseline. TRAIN_ALL (full stack) NOW WORKS too: with the final
//        LayerNorm (S1_USE_LNF) bounding the head input + the code-master weight decay (S1_CODE_WD_SHIFT)
//        countering the upward ratchet, the dense backbone trains without the old residual-saturation
//        collapse and BEATS freeze in the TB ablation (see forge_s1_config.hpp S1_Q_CODE / the TRAIN_ALL TB
//        block). init_only ignores freeze (the init must re-quantize EVERY code from staged W_fp
//        so the first forward reads valid codes for all regions).
// =============================================================================
static int do_opt(int32_t *buffer, int a1, int freeze_codes) {
#pragma HLS INLINE off
    // do_opt OWNS its writes: dyadic_adam_region is now a FUSED wide read+write (it commits in-body, the
    // morph signsgd_region_wide pattern), so the prior nested `#pragma HLS ALLOCATION instances=
    // dyadic_adam_region limit=1` is REMOVED. That nested limit=1 forced a single shared sub-instance whose
    // round-trip stores the RTL scheduler could not prove live (the mode-2 silicon no-op). The region loop
    // below is NOT unrolled and S1_OPT_LANE is 1 (the fused per-lane datapath is bounded — 16 lanes over a
    // single 512-bit word, not the GEMM's wide unroll), so HLS still builds ONE bounded datapath time-shared
    // across the 14 regions WITHOUT the nested share that broke the writeback. zero_grad_region keeps its
    // limit=1 (frozen-code grad-zero; a separate, already-committing tile store).
#pragma HLS ALLOCATION function instances=zero_grad_region   limit=1
    bool init_only = (a1 == S1_OPT_INIT);
    bool freeze = (freeze_codes == S1_OPT_FREEZE_CODES) && !init_only;
OPT_REGIONS: for (int r = 0; r < S1_NUM_OPT_REGIONS; r++) {
        const s1_opt_region_t &R = S1_OPT_REGIONS[r];
        if (freeze && R.is_code) {
            zero_grad_region(buffer, R);          // frozen code weight: drop its grad, keep master+code
        } else {
            dyadic_adam_region(buffer, R, init_only);
        }
    }
    return 0;
}

// =============================================================================
// ======================  MODE 3 (FUSED TRAINER) — do_train_run  ==============
// =============================================================================
// The WHOLE training run on fabric (the morph do_train_run structure, S1-adapted). ONE host kick runs
// `n_steps` full dyadic-Adam optimizer steps over the resident window; tokens/targets/master-weights/
// state all live in the buffer (no host round-trips per step). Each step is the EXACT same sequence the
// host would issue mode-by-mode:
//   (init, once)  do_opt(S1_OPT_INIT) -> re-quantize the staged W_fp into the codes/values the FIRST
//                 forward reads (so the host only needs to stage W_fp, not the codes).
//   per step:     mode-0 forward (whole sequence) -> grad_logit + activation saves
//                 mode-1 backward (whole sequence) -> g_* (the backward ZEROS its own grad accumulators
//                                                           on-chip at entry, so no separate grad-zero)
//                 mode-2 dyadic-Adam -> update W_fp + re-quantize codes/values + zero g_*
// Arithmetic is BIT-IDENTICAL to host-looping modes 0/1/2 (do_train_run just SEQUENCES the same three
// functions), so the host CSIM build verifies it is bit-exact to the per-mode golden.
//
// NOTE on residency: do_fwd_loss / do_bwd already burst-load every weight/state region into on-chip
// resident arrays at entry and burst-store at exit (the TT_RESIDENT_W pattern that made S1 FIT). The S1
// master-weight + Adam-state set is ~1000 BRAM18 tiles — too large to ALSO hold resident ACROSS steps on
// top of mode 1's 68% BRAM baseline — so mode 3 keeps weights/state in gmem BETWEEN phases (each phase
// re-stages its own resident window, exactly as the standalone modes do) and only the heavy fwd/bwd
// compute is on-chip-resident. This is the BRAM-correct S1 adaptation of morph's fully-resident loop;
// it is bit-exact (the values flowing through gmem are identical to the per-mode runs) and the optimizer
// streaming-RMW touches gmem with near-zero BRAM. n_steps full optimizer steps, one host kick.
// =============================================================================
static int do_train_run(int32_t *buffer, int n_steps, int freeze_codes) {
#pragma HLS INLINE off
    do_opt(buffer, S1_OPT_INIT, freeze_codes);  // populate the forward-read codes/values from staged W_fp
S1_STEP_LOOP: for (int s = 0; s < n_steps; s++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        do_fwd_loss(buffer);                // mode 0: forward + CE -> grad_logit + activation saves
        do_bwd(buffer);                     // mode 1: backward -> g_* (zeros its own accumulators)
        do_opt(buffer, S1_OPT_STEP, freeze_codes);  // mode 2: dyadic-Adam -> W_fp + re-quantize + grad-zero
    }
    return 0;
}

#endif  // !S1_FWD_ONLY

// =============================================================================
// Top-level dispatcher — mirror forge_train_step_morph_lm.cpp ABI/pragmas.
// =============================================================================
extern "C" int forge_train_step_s1(
    int32_t  buffer[S1_BUFLEN],
    int      mode,
    int      a1,
    int      a2,
    int      a3
) {
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=S1_BUFLEN \
        num_write_outstanding=16 num_read_outstanding=16 \
        max_write_burst_length=16 max_read_burst_length=16 \
        max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite  port=mode    bundle=control
#pragma HLS INTERFACE s_axilite  port=a1      bundle=control
#pragma HLS INTERFACE s_axilite  port=a2      bundle=control
#pragma HLS INTERFACE s_axilite  port=a3      bundle=control
#pragma HLS INTERFACE s_axilite  port=buffer  bundle=control
#pragma HLS INTERFACE s_axilite  port=return  bundle=control

    // AREA FIX (mode-3 sharing) — force ONE physical instance of each sub-function, time-multiplexed by the
    // dispatcher's direct mode-0/1/2 calls AND by mode 3's do_train_run (which calls the SAME three). Without
    // this, HLS builds the sub-functions once for the dispatch path and AGAIN inside do_train_run — a full
    // duplicate copy of do_fwd_loss + do_bwd + do_opt (~904k LUT, ~half the kernel). limit=1 collapses the
    // duplicate to a single shared instance (mode 3 just re-sequences the same engines; bit-exact unchanged).
#pragma HLS ALLOCATION function instances=do_fwd_loss limit=1
#if !S1_FWD_ONLY
#pragma HLS ALLOCATION function instances=do_bwd      limit=1
#pragma HLS ALLOCATION function instances=do_opt      limit=1
#endif

    (void)a3;
    if (mode == 0) return do_fwd_loss(buffer);
#if !S1_FWD_ONLY
    if (mode == 1) return do_bwd(buffer);
    // mode 2: dyadic-Adam. a1 = S1_OPT_STEP | S1_OPT_INIT ; a2 = S1_OPT_TRAIN_ALL | S1_OPT_FREEZE_CODES.
    if (mode == 2) return do_opt(buffer, a1, a2);
    // mode 3: fused trainer. a1 = n_steps (ONE host kick) ; a2 = S1_OPT_TRAIN_ALL | S1_OPT_FREEZE_CODES.
    if (mode == 3) return do_train_run(buffer, a1, a2);
#else
    (void)a1; (void)a2;
#endif
    return 99;
}
