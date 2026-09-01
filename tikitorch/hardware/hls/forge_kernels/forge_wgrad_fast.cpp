// forge_wgrad_fast.cpp — THROUGHPUT rebuild of the operand-quantized weight-grad GEMM (training rig).
//
// Same math + SAME buffer layout as forge_wgrad_probe (the PC set_mf_grads recipe):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])     with q(v) = sign(v) * 2^round(log2|v|)
// Because BOTH operands are dyadic (±2^k), each product is ±2^(eo+ei): a sign-xor + exponent-ADD
// + a barrel SHIFT ((acc_t)1<<e) + a signed ACCUMULATE. NO multiply, NO DSP multiplier (a DSP48 may
// still serve as the signed accumulator/adder — the "DSP-as-adder" lever, not a true multiply).
//
// WHY THIS VERSION (throughput is THE flagship lever — the rig is compute-bound):
//   forge_wgrad_probe CLOSED timing (2.92 ns, 49% LUT-of-SLR, DSP=0) but ran 664,932 cycles
//   (2.66 ms for the 768x768 / N16 grad): its OTILE(96) x ITILE(12) tile-traversal loops were NOT
//   pipelined, so each of the 1152 tiles paid (acc zero-init + the NRED pipeline fill/drain +
//   writeback) latency SERIALLY, and the only II=1 loop (the tiny N=16 reduction) was dwarfed.
//
// THE FIX — FLATTEN the whole tile traversal into ONE PIPELINE II=1 loop over PR_TILES "super-tiles"
//   of (PR_N reduce + PR_IT drain) trips each. The SAME 8x64 = 512-lane PE tile (the array that
//   already closed at 2.92 ns / 49% LUT, DSP=0) is held STATIONARY in registers. Within a super-tile
//   the first PR_N trips MAC the microbatch into the bank (ph==0 SEEDS, folding zero-init into the
//   first add); the next PR_IT trips DRAIN the finished bank, emitting ONE OT-wide column / cycle to
//   the store FIFO. Reduce-then-drain is sequential WITHIN a tile (so the bank is stable while drained
//   — no ping-pong, no RAW hazard), but the WHOLE traversal is one continuous pipeline: the per-tile
//   fill/drain bubble of the baseline is paid ONCE for the matrix, not 1152 times.
//
// WHY a 1-write/cycle FIFO drain, NOT an inline wide emit (the trap that sank wgrad2 AND wgrad_sys):
//   Both put a wide CONDITIONAL multi-element emit (512 / 64 stream writes, or an unrolled 512-wide
//   m_axi store burst) INSIDE the II=1 hot loop. A single hls::stream — and a single m_axi port — has
//   ONE write port per cycle, so N writes in one iteration force II>=N on that iteration (or fail to
//   pipeline) and explode mux/area (wgrad2: 81% LUT, 2.7-40 ms). Here the hot loop writes EXACTLY ONE
//   accvec_t per drain cycle; the decoupled store() stage reads one per cycle and bursts it to DDR
//   under DATAFLOW. The store FIFO carries a couple tiles of slack so neither stage backpressures.
//
// THROUGHPUT ROOFLINE: per tile = PR_N (reduce) + PR_IT (drain) = 16 + 64 = 80 trips, x 1152 tiles
//   = 92,160 trips for compute, + pipeline depth + load/quant. The store DDR floor is O*I = 589,824
//   int32 = 36,864 beats @ 512-bit (16 int32/beat) and overlaps compute under DATAFLOW. So latency
//   ~= ~92k + small overhead (~370 us @ 4 ns) — ~7x faster than the baseline's 665k, at the SAME PE
//   width / area. (The drain trips are the price of a clean single-write FIFO that actually holds
//   II=1; that beats the unreliable wide-emit attempts, which did not pipeline at all.)
//
// AREA: identical 512-lane PE to forge_wgrad_probe (closed 49% LUT-of-SLR, DSP=0) + one small store
//   FIFO — comfortably under one SLR (434560 LUT, 3008 DSP, 869120 FF, 320 URAM18, 1344 BRAM18).
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_fast.tcl.

#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) — d_model
#define PR_I       768     // input  features (cols of W) — d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      8       // output-tile rows held on-chip
#define PR_IT      64      // output-tile cols held on-chip (= unrolled PE lane width) -> 512 lanes

#define PR_NOT     (PR_O / PR_OT)          // 96  output-row tiles
#define PR_NIT     (PR_I / PR_IT)          // 12  output-col tiles
#define PR_TILES   (PR_NOT * PR_NIT)       // 1152 tiles total
// One flattened super-tile per gW tile: PR_N reduction trips THEN PR_IT drain trips. Both run inside
// the SAME II=1 pipeline, so the per-tile fill/drain bubble of the baseline is paid ONCE for the whole
// matrix instead of 1152 times. Per-tile cost = PR_N + PR_IT = 16 + 64 = 80 trips.
#define PR_PHASE   (PR_N + PR_IT)          // 80  trips per tile: N reduce + IT drain
#define PR_TRIPS   (PR_TILES * PR_PHASE)   // 92160 flattened super-tile trips (the latency floor)

typedef ap_int<16>  data_t;    // gO / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..16 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad accumulator (DSP48-width signed)

// One OT-wide column of grad accumulators — the unit drained per store cycle (a packed struct so the
// store FIFO carries OT accs/transfer; IT such columns per tile => a tile drains in IT cycles).
struct accvec_t { acc_t v[PR_OT]; };

// flat int32 DDR buffer layout (drop-in, identical to forge_wgrad_probe):
//   [ gO (N*O) | x (N*I) | gW (O*I) ]
static const int OFF_GO = 0;
static const int OFF_X  = OFF_GO + PR_N * PR_O;
static const int OFF_GW = OFF_X  + PR_N * PR_I;

// dyadic-quantize a value to (sign, exponent): round(log2|v|) via MSB + the next bit (round-to-nearest).
// COPIED verbatim from forge_wgrad_probe (the proven priority-encoder).
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

static inline void burst_load(int32_t *buf, int off, data_t *dst, int n) {
#pragma HLS INLINE off
LD: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        dst[i] = (data_t)buf[off + i];
    }
}

// ============================================================================================
// STAGE 1 (load + quantize): stage gO/x on-chip and pre-quantize to dyadic (sign,exp,nz) ONCE.
// ============================================================================================
static void stage_and_quant(int32_t *buffer,
                            exp_t goE[PR_N][PR_O], bool goS[PR_N][PR_O], bool goNZ[PR_N][PR_O],
                            exp_t xE[PR_N][PR_I],  bool xS[PR_N][PR_I],  bool xNZ[PR_N][PR_I]) {
#pragma HLS INLINE off
    static data_t gO[PR_N][PR_O];
    static data_t xx[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=gO cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xx cyclic factor=PR_IT dim=2
    burst_load(buffer, OFF_GO, &gO[0][0], PR_N * PR_O);
    burst_load(buffer, OFF_X,  &xx[0][0], PR_N * PR_I);
QO: for (int n = 0; n < PR_N; n++)
        for (int o = 0; o < PR_O; o++) {
#pragma HLS PIPELINE II=1
            dyad(gO[n][o], goS[n][o], goE[n][o], goNZ[n][o]);
        }
QX: for (int n = 0; n < PR_N; n++)
        for (int i = 0; i < PR_I; i++) {
#pragma HLS PIPELINE II=1
            dyad(xx[n][i], xS[n][i], xE[n][i], xNZ[n][i]);
        }
}

// ============================================================================================
// STAGE 2 (compute): the 512-lane output-stationary PE grid. ONE flattened II=1 pipeline over
// PR_TILES super-tiles of PR_PHASE(=PR_N+PR_IT=80) trips each. Within a super-tile:
//   * ph in [0,PR_N)            REDUCE: 512 dyadic-MACs into the `acc` bank for token ph of this tile.
//                               ph==0 SEEDS (zero-init folded into the first add).
//   * ph in [PR_N, PR_N+PR_IT)  DRAIN : emit ONE OT-wide column (col ph-PR_N) of the completed `acc`
//                               bank to the store FIFO — exactly ONE FIFO write per cycle, so the loop
//                               holds II=1 (NO wide conditional emit, the trap that sank wgrad2/sys).
// Reduce then drain are sequential WITHIN a tile (so `acc` is stable while drained — no ping-pong, no
// RAW hazard), but the WHOLE traversal is one continuous pipeline: the per-tile fill/drain bubble is
// paid ONCE for the matrix, not 1152 times. The next tile's coordinates are advanced on the last drain
// trip. Store overlaps this under DATAFLOW, hiding the DDR write-back under the arithmetic.
// ============================================================================================
static void compute(exp_t goE[PR_N][PR_O], bool goS[PR_N][PR_O], bool goNZ[PR_N][PR_O],
                    exp_t xE[PR_N][PR_I],  bool xS[PR_N][PR_I],  bool xNZ[PR_N][PR_I],
                    hls::stream<accvec_t> &out) {
#pragma HLS INLINE off
    acc_t acc[PR_OT][PR_IT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    int ot = 0;       // current tile's output-row base
    int it = 0;       // current tile's output-col base

FLAT: for (int t = 0; t < PR_TRIPS; t++) {
#pragma HLS PIPELINE II=1
        int ph = t % PR_PHASE;              // position within the super-tile [0..PR_PHASE)

        if (ph < PR_N) {
            // ---- REDUCE: 512 dyadic MACs into the bank for token ph of this tile. ----
            int n = ph;
        MR: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                exp_t eo = goE[n][ot + r]; bool so = goS[n][ot + r]; bool nzo = goNZ[n][ot + r];
            MC: for (int c = 0; c < PR_IT; c++) {
#pragma HLS UNROLL
                    exp_t ei = xE[n][it + c];
                    ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;   // exponent ADD
                    acc_t term = (acc_t)1 << e;                       // barrel SHIFT, NO multiply
                    bool neg = so ^ xS[n][it + c];                    // sign XOR
                    bool nz  = nzo & xNZ[n][it + c];
                    acc_t add = nz ? (neg ? (acc_t)(-term) : term) : (acc_t)0;
                    // SEED on n==0 (folds zero-init into first add); accumulate otherwise.
                    acc[r][c] = (n == 0) ? add : (acc_t)(acc[r][c] + add);
                }
            }
        } else {
            // ---- DRAIN: emit ONE OT-wide column (col ph-PR_N) of the finished bank. ----
            int dc = ph - PR_N;             // drain column 0..PR_IT-1
            accvec_t col;
        DR: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                col.v[r] = acc[r][dc];
            }
            out.write(col);                 // exactly ONE FIFO write this cycle => II=1 holds
            // on the last drain trip, advance to the next tile (row-major over IT then OT)
            if (dc == PR_IT - 1) {
                it += PR_IT;
                if (it == PR_I) { it = 0; ot += PR_OT; }
            }
        }
    }
}

// ============================================================================================
// STAGE 3 (store): drain the FIFO to DDR in the SAME tile order compute emits (row-major IT then
// OT). Each FIFO transfer is an OT-wide column; write its OT elements to their gW rows. Pipelined
// II=1; overlaps with compute under DATAFLOW so DDR write-back hides under the arithmetic.
// ============================================================================================
static void store(int32_t *buffer, hls::stream<accvec_t> &in) {
#pragma HLS INLINE off
SOT: for (int ot = 0; ot < PR_O; ot += PR_OT) {
SIT:    for (int it = 0; it < PR_I; it += PR_IT) {
        SC: for (int c = 0; c < PR_IT; c++) {
#pragma HLS PIPELINE II=1
                accvec_t col = in.read();
            SR: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                    buffer[OFF_GW + (ot + r) * PR_I + (it + c)] = (int32_t)col.v[r];
                }
            }
        }
    }
}

// ============================================================================================
// TOP: load+quantize -> compute (512-lane PE, phased reduce/drain) -> store, joined by HLS DATAFLOW.
// ============================================================================================
void forge_wgrad_fast(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW

    // pre-quantized operand fields — partitioned so all 512 lanes read in parallel each cycle:
    // gO banked on its O (row) axis by PR_OT, x banked on its I (col) axis by PR_IT.
    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=goE  cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goS  cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=goNZ cyclic factor=PR_OT dim=2
#pragma HLS ARRAY_PARTITION variable=xE   cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xS   cyclic factor=PR_IT dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ  cyclic factor=PR_IT dim=2

    // store FIFO: OT-wide grad columns; depth 256 = ~4 tiles of slack so store never backpressures
    // compute (store consumes IT cols/tile at II=1; compute produces them one/cycle in the drain
    // phase, so the long-run consume rate >= produce rate and the FIFO stays bounded).
    hls::stream<accvec_t> gwfifo;
#pragma HLS STREAM variable=gwfifo depth=256

    stage_and_quant(buffer, goE, goS, goNZ, xE, xS, xNZ);
    compute(goE, goS, goNZ, xE, xS, xNZ, gwfifo);
    store(buffer, gwfifo);
}
