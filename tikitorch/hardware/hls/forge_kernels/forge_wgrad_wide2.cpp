// forge_wgrad_wide2.cpp — FRONT-MATTER fix for the rig's weight-grad GEMM (gate-2 backward kernel).
//
// Starts from forge_wgrad_wide.cpp (the current best wgrad: 111,972 cycles, closes 2.920 ns, DSP=0,
// 50% LUT/SLR). That kernel's compute is the proven 512-lane dyadic-MAC PE (OT8xIT64) + a 512-bit
// wide-burst writeback. Its REMAINING waste is the ~49k-cycle FRONT-MATTER:
//     burst_load gO  ~12,300  (scalar int32 reads, 1/cycle)
//     burst_load x   ~12,300
//     QO  quantize   ~12,293  (re-scan gO, dyad each element)
//     QX  quantize   ~12,292  (re-scan x,  dyad each element)
// = ~49k of the 112k cycles is just getting operands in and dyadic-encoding them.
//
// FIX (two moves, both front-matter only — the PE + writeback are byte-for-byte the wide kernel):
//   (1) WIDE LOADS. gO and x are read through an ap_uint<512> port (16 int32 / cycle) instead of the
//       scalar burst_load. gO = N*O = 12,288 int32 = 768 wide words; x = N*I = 12,288 = 768 wide words.
//       Both regions are 512-bit aligned (OFF_GO=0 -> /16=0; OFF_X=12288 -> /16=768). So one ap_uint<512>
//       port indexed in 16-int32 units gives an exact, contiguous wide burst: ~768 cycles each vs 12,300.
//   (2) FUSE QUANTIZE INTO THE UNPACK. As each wide word's 16 lanes are unpacked we dyad() each lane
//       straight into the dyadic (sign,exp,nz) banks. The separate QO/QX scans (~24,600 cycles) vanish,
//       AND we never materialize the raw int16 gO/xx arrays (dropped — only the dyadic banks remain).
//
//   Front-matter ~49k -> ~1.6k cycles (768 + 768 wide-unpack-quantize passes, II=1, 16-lane unrolled).
//   Projected total ~64k cycles (~0.26 ms) vs the wide kernel's 112k — the PE/writeback (~63k) now
//   dominate. DSP still 0 (dyad is a priority-encoder; the MAC is shift+accumulate). Synthesis-safe:
//   this is the proven kernel minus two scan passes plus a 16-lane unpack, NOT a flatten/unroll blowup.
//
// gO/x wide ports are SEPARATE m_axi bundles from gw_wide so the two reads + the write don't serialize
// on one AXI master. gO and x share a read bundle (they're read sequentially, never concurrently).
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_wide2.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; probe the FFN-class proj 768x768) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= the unrolled PE lane count)

typedef ap_int<16>   data_t;   // gO / x values (int16, the morph data_t range)
typedef ap_int<5>    exp_t;    // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>   acc_t;    // grad accumulator (DSP48-width signed)
typedef ap_uint<512> wide_t;   // 16 x int32 packed AXI word

// flat int32 DDR buffer layout (probe-local, IDENTICAL to forge_wgrad_wide):
//   [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO   = 0;
static const int OFF_X    = OFF_GO + PR_N * PR_O;        // = 12288 int32
static const int OFF_GW   = OFF_X  + PR_N * PR_I;        // = 24576 int32
static const int OFF_GO_W = OFF_GO / 16;                 // = 0    wide (512-bit) words
static const int OFF_X_W  = OFF_X  / 16;                 // = 768  wide words
static const int OFF_GW_W = OFF_GW / 16;                 // = 1536 wide words
static const int N_GO_W   = (PR_N * PR_O) / 16;          // = 768  wide words to read for gO
static const int N_X_W    = (PR_N * PR_I) / 16;          // = 768  wide words to read for x

// dyadic-quantize a value to (sign, exponent): round(log2|v|) via MSB + the next bit (round-to-nearest).
static inline void dyad(data_t v, bool &sgn, exp_t &e, bool &nz) {
#pragma HLS INLINE
    ap_uint<16> a = (v < 0) ? (ap_uint<16>)(-v) : (ap_uint<16>)v;
    sgn = (v < 0);
    nz = (a != 0);
    exp_t msb = 0;
PRIO: for (int b = 15; b >= 1; b--) {           // priority encoder: highest set bit
#pragma HLS UNROLL
        if (a[b]) { msb = b; break; }
    }
    // round-to-nearest power of two: bump if the bit below the MSB is set
    if (msb >= 1 && a[msb - 1]) msb = msb + 1;
    e = msb;
}

// WIDE LOAD + FUSED QUANTIZE: read `nwords` 512-bit words from src[srcw .. srcw+nwords) (each = 16 int32),
// unpack the 16 lanes, dyad() each straight into the (sign,exp,nz) banks. flat element index = wbase + 16w+k.
// The dyadic banks are 2-D [PR_N][feat]; element index ee = n*feat + f, so n = ee/feat, f = ee%feat. feat is
// PR_O or PR_I = 768 = 16*48, so each 16-lane wide word lies WITHIN one row n (16 | 768) -> n constant per
// word, f = (16*w+k) % 768 contiguous. Contiguous f across 16 lanes hit 16 distinct cyclic banks when the
// partition factor is a multiple of 16 (both banks here are: goE factor 16, xE factor 64) -> II=1 writes.
static inline void wide_quant(wide_t *src, int srcw, int nwords, int feat,
                              exp_t eb[][PR_O], bool sb[][PR_O], bool nzb[][PR_O]) {
#pragma HLS INLINE off
LDQ: for (int w = 0; w < nwords; w++) {
#pragma HLS PIPELINE II=1
        wide_t word = src[srcw + w];
        int base = 16 * w;              // flat element index of lane 0 of this word
        int n0   = base / feat;         // row index (constant across the 16 lanes — 16 | feat)
        int f0   = base - n0 * feat;    // column index of lane 0
        for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
            data_t v = (data_t)(int32_t)word.range(32 * k + 31, 32 * k);
            bool s, nz; exp_t e;
            dyad(v, s, e, nz);
            int f = f0 + k;
            sb[n0][f] = s; eb[n0][f] = e; nzb[n0][f] = nz;
        }
    }
}

void forge_wgrad_wide2(wide_t *go_x, wide_t *gw_wide) {
#pragma HLS INTERFACE m_axi port=go_x    offset=slave bundle=gmem  max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=gw_wide offset=slave bundle=gmemw max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=go_x
#pragma HLS INTERFACE s_axilite port=gw_wide
#pragma HLS INTERFACE s_axilite port=return

    // Dyadic operand banks ONLY (the raw int16 gO/xx arrays are gone — quantize is fused into the load).
    // Partition factor 16 (gO) / 64 (x) on the feature dim: (a) makes the 16-lane wide unpack conflict-free
    // and (b) feeds the PE — for gO, ot is a multiple of PR_OT=8 so {ot..ot+7} hit 8 distinct banks mod 16;
    // for x, it is a multiple of PR_IT=64 so {it..it+63} hit all 64 banks.
    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=goE  cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=goS  cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=16     dim=2
#pragma HLS ARRAY_PARTITION variable=xE   cyclic factor=PR_IT  dim=2
#pragma HLS ARRAY_PARTITION variable=xS   cyclic factor=PR_IT  dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ  cyclic factor=PR_IT  dim=2

    // (1)+(2) wide-load + fused-quantize gO then x. ~768 cycles each (vs 12,300 load + 12,300 quantize).
    wide_quant(go_x, OFF_GO_W, N_GO_W, PR_O, goE, goS, goNZ);
    wide_quant(go_x, OFF_X_W,  N_X_W,  PR_I, xE,  xS,  xNZ);

    // ---- BELOW THIS LINE IS BYTE-FOR-BYTE forge_wgrad_wide: the proven PE + wide writeback. ----

    // Output-stationary tiled GEMM: tile (OT x IT) of gW, reduce over the N microbatch.
    // Inner IT cols unrolled = the PE lane array (identical to the proven probe).
OTILE: for (int ot = 0; ot < PR_O; ot += PR_OT) {
ITILE: for (int it = 0; it < PR_I; it += PR_IT) {
        acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
        for (int r = 0; r < PR_OT; r++)
            for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                acc[r][c] = 0;
            }
NRED:   for (int n = 0; n < PR_N; n++) {
#pragma HLS PIPELINE II=1
            for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                exp_t eo = goE[n][ot + r]; bool so = goS[n][ot + r]; bool nzo = goNZ[n][ot + r];
                for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                    // dyadic MAC: product = ±2^(eo+ei); a barrel shift + signed accumulate, NO multiply
                    exp_t ei = xE[n][it + c];
                    ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;
                    acc_t term = (acc_t)1 << e;
                    bool neg = so ^ xS[n][it + c];
                    bool nz = nzo & xNZ[n][it + c];
                    if (nz) acc[r][c] += neg ? (acc_t)(-term) : term;
                }
            }
        }
        // WIDE-BURST writeback: pack 16 contiguous int32 accumulators per 512-bit word.
        // IT=64 cols -> PR_IT/16 = 4 wide words per row; PR_OT=8 rows -> 32 wide writes/tile (II=1).
WB:     for (int r = 0; r < PR_OT; r++) {
            for (int w = 0; w < PR_IT / 16; w++) {
#pragma HLS PIPELINE II=1
                wide_t word;
                for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                    word.range(32 * k + 31, 32 * k) = (ap_uint<32>)acc[r][w * 16 + k];
                }
                // wide word index: gW row (ot+r), col block (it + w*16), in 512-bit units
                int widx = OFF_GW_W + ((ot + r) * PR_I + it) / 16 + w;
                gw_wide[widx] = word;
            }
        }
    }
    }
}
