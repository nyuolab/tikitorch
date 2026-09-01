// forge_wgrad_bigpe.cpp — BIG-PE-ARRAY throughput rebuild of the operand-quantized weight-grad GEMM.
//
// Same math + same DDR buffer layout as forge_wgrad_probe (the PC set_mf_grads recipe):
//   gW[o][i] = sum_n q(gO[n][o]) * q(x[n][i])     with q(v) = sign(v) * 2^round(log2|v|)
// Because BOTH operands are dyadic (+/-2^k), each product is +/-2^(eo+ei): a sign-xor + exponent-ADD
// + a barrel SHIFT ((acc_t)1<<e) + a signed ACCUMULATE. NO multiply, NO DSP multiplier.
//
// WHY THIS VERSION (throughput is THE flagship lever -- the rig is compute-bound):
//   forge_wgrad_probe CLOSED timing (2.92 ns, 49% LUT-of-SLR, DSP=0) but ran 664,932 cycles
//   (2.66 ms for the 768x768 / N16 grad) because its OTILE x ITILE tile loops were NOT pipelined:
//   every one of the 96*12 = 1152 tiles paid (zero-init + NRED fill/drain + writeback) latency
//   serially, and the only II=1 loop (the N=16 reduction) was tiny, so the per-tile fixed overhead
//   dominated. A DATAFLOW rewrite (forge_wgrad2) instead blew LUT to 81% and timing went unreliable.
//
// STRATEGY -- BIG PE ARRAY, ROW-BLOCK / OUTPUT-STATIONARY-IN-REGISTERS (no dataflow, no FIFOs):
//   Hold a block of PR_OT output ROWS x the FULL PR_I columns in an on-chip REGISTER accumulator
//   (acc[PR_OT][PR_I], partitioned COMPLETE). Lane count = PR_OT*PR_I -- a wide PE array. The
//   microbatch reduction over N is ONE perfectly-pipelined II=1 loop: every cycle issues a full
//   PR_OT*PR_I-wide vector of dyadic MACs, accumulating into the registers. There is NO per-tile
//   drain inside the reduction (acc is seeded at n==0), so the baseline's 1152 fill/drain bubbles
//   collapse to one tiny boundary per row-block.
//
//   WHY ROW-BLOCK INSTEAD OF A FLATTENED-TILE + ON-CHIP gW STAGING: a single flattened (tile x N)
//   loop that commits each OT x IT tile into a full on-chip gW[O][I] array needs gW banked OT*IT-wide
//   to commit without a bank conflict (PR_OT rows colliding on each col-bank otherwise forces II>1) --
//   and an OT*IT-bank acc_t[O][I] array overflows one SLR's BRAM. Keeping acc in REGISTERS and writing
//   each finished row-block STRAIGHT to DDR as a CONTIGUOUS, AXI-widenable burst avoids BOTH the
//   bank-conflict-driven II>1 AND the BRAM blowup. (Scattering OT*IT m_axi writes inside the II=1 hot
//   loop -- the prior bigpe attempt -- serializes on the single AXI write port; this does not.)
//
//   Wide fan-out: x is the per-cycle broadcast operand, so its pre-quantized (sign,exp,nz) banks are
//   partitioned COMPLETE on the I axis -- every one of the PR_I columns reads its operand in parallel.
//
//   Lanes = PR_OT*PR_I. With PR_OT=2 that is 1536 lanes; compute roofline = O*I*N/lanes =
//   768*768*16/1536 = 6144 cycles. The O*I=590k-word grad writeback AXI-widened to 512-bit
//   (16 int32/beat) is ~36,864 beats -- the DDR floor -- and overlaps the next block's reduction
//   across the row-block loop. End-to-end ~ tens of k cycles vs the baseline's 664,932 -- a ~18x+ cut,
//   while acc stays in registers (one-SLR-safe) and timing stays the probe's clean DSP=0 LUT-adder
//   form. If area/timing allow, raise PR_OT for more lanes (PR_OT=4 -> 3072 lanes, floor 3072 cycles).
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csynth via synth_wgrad_bigpe.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- transformer-width config (d=768; FFN-class proj 768x768, microbatch N=16) ----
#define PR_O       768     // output features (rows of W) -- d_model
#define PR_I       768     // input  features (cols of W) -- d_model
#define PR_N       16      // microbatch tokens reduced per weight-grad (the contraction)
#define PR_OT      2       // output ROWS held on-chip per block (full PR_I cols each) -> 1536 lanes
#define PR_NOB     (PR_O / PR_OT)   // 384 output-row blocks

typedef ap_int<16>  data_t;    // gO / x values (int16, the morph data_t range)
typedef ap_int<5>   exp_t;     // dyadic exponent 0..15 (msb position of a 16-bit magnitude)
typedef ap_int<48>  acc_t;     // grad accumulator (DSP48-width signed)

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

void forge_wgrad_bigpe(int32_t *buffer) {
#pragma HLS INTERFACE m_axi port=buffer offset=slave bundle=gmem max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=return

    // ---- Stage operands on-chip, then prequantize their dyadic (sign,exp,nz) ONCE. -----------
    // x is the per-cycle broadcast operand: partition it COMPLETE on the I axis so every one of the
    // PR_I lanes can read its column in the same cycle (the II=1 enabler for the inner pipeline).
    static data_t gO[PR_N][PR_O];
    static data_t xx[PR_N][PR_I];
#pragma HLS ARRAY_PARTITION variable=xx complete dim=2
    burst_load(buffer, OFF_GO, &gO[0][0], PR_N * PR_O);
    burst_load(buffer, OFF_X,  &xx[0][0], PR_N * PR_I);

    static exp_t goE[PR_N][PR_O]; static bool goS[PR_N][PR_O]; static bool goNZ[PR_N][PR_O];
    static exp_t xE[PR_N][PR_I];  static bool xS[PR_N][PR_I];  static bool xNZ[PR_N][PR_I];
    // x fields fully partitioned on I (all PR_I lanes read in parallel each cycle).
#pragma HLS ARRAY_PARTITION variable=xE  complete dim=2
#pragma HLS ARRAY_PARTITION variable=xS  complete dim=2
#pragma HLS ARRAY_PARTITION variable=xNZ complete dim=2

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

    // ---- ROW-BLOCK BIG-PE GEMM: PR_OT rows x ALL PR_I cols on-chip, in registers. -------------
    // acc holds PR_OT output rows x ALL PR_I columns, partitioned COMPLETE so the PR_OT*PR_I lanes
    // update simultaneously. The reduction over the microbatch N is ONE II=1 loop that issues a full
    // PR_OT*PR_I MAC vector every cycle; acc is SEEDED at n==0 (no separate zero pass). After the
    // reduction, the finished row-block is bursted STRAIGHT to DDR (contiguous, AXI-widenable) -- no
    // on-chip gW staging array, no scattered m_axi writes inside the hot loop. The writeback overlaps
    // the next block's reduction across the OBLK loop.
OBLK: for (int ob = 0; ob < PR_O; ob += PR_OT) {

        acc_t acc[PR_OT][PR_I];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

        // single perfectly-pipelined reduction over the microbatch N
    NRED: for (int n = 0; n < PR_N; n++) {
#pragma HLS PIPELINE II=1
        ROW: for (int r = 0; r < PR_OT; r++) {
#pragma HLS UNROLL
                exp_t eo  = goE[n][ob + r];
                bool  so  = goS[n][ob + r];
                bool  nzo = goNZ[n][ob + r];
            COL: for (int c = 0; c < PR_I; c++) {
#pragma HLS UNROLL
                    // dyadic MAC: product = +/-2^(eo+ei) -- barrel shift + signed accumulate, NO multiply
                    exp_t ei = xE[n][c];
                    ap_uint<6> e = (ap_uint<6>)eo + (ap_uint<6>)ei;        // exponent ADD
                    acc_t term = (acc_t)1 << e;                            // barrel SHIFT, NO multiply
                    bool neg = so ^ xS[n][c];                              // sign XOR
                    bool nz  = nzo & xNZ[n][c];
                    acc_t add = nz ? (neg ? (acc_t)(-term) : term) : (acc_t)0;
                    acc[r][c] = (n == 0) ? add : (acc_t)(acc[r][c] + add); // seed | signed ACCUMULATE
                }
            }
        }

        // writeback the completed row-block STRAIGHT to DDR (contiguous, AXI-widenable burst, II=1)
    WB: for (int r = 0; r < PR_OT; r++)
        WBC: for (int c = 0; c < PR_I; c++) {
#pragma HLS PIPELINE II=1
                buffer[OFF_GW + (ob + r) * PR_I + c] = (int32_t)acc[r][c];
            }
    }
}
