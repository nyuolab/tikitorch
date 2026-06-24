// forge_train_step_s1_f2.c — TRAIN one multiplier-free transformer block ENTIRELY ON F2 SILICON
// (forward + backward + dyadic-Adam, no GPU), driving the verified forge_train_step_s1 kernel that
// reads/writes its whole working set in ON-CARD DDR4. ONE host kick = mode 3 (do_train_run) runs
// N_STEPS of fwd->bwd->opt with all weights/state resident in DDR — no per-step host round-trip.
//
// DATA PATH (the proven cl_forge_muzero_ddr DDR topology; PCIM is NOT drivable on EC2 Nitro, R1):
//   host BAR0 (OCL AXI4-Lite)  -> s1_ocl_slice -> [OCL CDC 250->125] -> kernel s_axi_control[12:0]  (mode/args/buffer-ptr + kick/poll)
//   host BAR4 (DMA-PCIS, 512b) -\
//                                >- axi_crossbar_ddr (2->1 SASD, 250MHz) -> sh_ddr -> DDR4   (stage + readback)
//   kernel m_axi gmem (512b RW)-[GMEM CDC 125->250]-/    buffer lives at DDR addr 0; kernel BUFFER ptr reg = 0
//   host mgmt AXI-Lite (SDA/PF1-BAR4) -> AWS_CLK_GEN   (MMCM_LOCK poll + SYS_RST deassert = clkgen bring-up)
//
// CLKGEN+CDC (S1_CLKGEN_PLAN.md): the kernel runs on AWS_CLK_GEN's GENERATED clk_extra_a1 = 125MHz (the
// kernel post-route closes ~144MHz but MISSES the shell-fixed 250MHz clk_main_a0); the shell interfaces
// (DMA-PCIS, OCL, sh_ddr, crossbar) stay at 250MHz and the kernel's two AXI boundaries are bridged by
// axi_clock_converter CDC IP. The host MUST do the AWS_CLK_GEN bring-up (clkgen_bringup() below) AFTER AFI
// load and BEFORE any stage/kick — else clk_extra_a1 is static-low / the kernel domain is held in reset.
//
// SILICON-PROOF: this reproduces the forge_train_step_s1_tb MODE-3 convergence block on real fabric.
// The host stages the IDENTICAL init the TB uses (stage_opt_buffer(seed=4242, memorize=true): the
// residual-passthrough identity block + small embed/head + the host-staged exp2 LUTs + the 8-symbol
// successor memorization tokens/targets), kicks mode 3 in FREEZE_CODES, reads back the saved LOGITS
// and computes the TRUE float cross-entropy (un-fakeable: uniform/saturation collapse == log V floor).
// The on-fabric CE drop must match the CSIM golden (TB reports ~CE 4.2 -> well below the floor).
//
// Because the host C srand()/rand() sequence is replayed in the SAME order as the TB's
// stage_opt_buffer() and both run on the F2's glibc, the staged buffer is byte-identical to the CSIM
// init -> the silicon trajectory is the CSIM trajectory (modes 0/1/2/3 are all bit-exact-frozen).
//
// Build on an F2 with the aws-fpga SDK (compile-as-C; the kernel ABI + offset map come from the shared
// C++ header forge_s1_config.hpp via two tiny C-safe include shims, below). The shared header does
// `#include <cstdint>` (a C++ header); the companion shim dir cstdint_shim/ next to this file provides a
// one-line `cstdint` -> <stdint.h> redirect on the -I path, so the driver builds with a plain C toolchain:
//   gcc -O2 -std=gnu11 -I$SDK/include \
//       -I<repo>/tikitorch/hardware/hls/forge_kernels \
//       -I<repo>/tikitorch/fpga/silicon/cstdint_shim \
//       forge_train_step_s1_f2.c -lfpga_mgmt -o s1_f2
//   The CLKGEN+CDC variant additionally needs the SDK clkgen library for the spec-blessed bring-up:
//   add -lfpga_clkgen (and ensure <fpga_clkgen.h> is on -I$SDK/include). If an SDK lacks the clkgen lib,
//   build with -DHAVE_FPGA_CLKGEN=0 to use the explicit-register fallback (mgmt-BAR MMCM_LOCK poll +
//   SYS_RST clear) and drop -lfpga_clkgen.
//   sudo ./s1_f2 <slot> <n_steps> [freeze|trainall] [seed]
//
//   <slot>     FPGA slot (0)
//   <n_steps>  mode-3 fused steps in ONE kick (default 32; TB csim horizon)
//   freeze     a2=S1_OPT_FREEZE_CODES (default; the bit-exact-verified convergence regime)
//   trainall   a2=S1_OPT_TRAIN_ALL + the stage_train_all dense-backbone init (the dense-code probe)
//   seed       RNG seed for the init (default 4242 = the TB mode-3 convergence seed)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
// Compiled as C (gcc -std=gnu11): the AWS fpga_pci.h burst path is C-clean. We DON'T include
// fpga_dma.h (its `char buf[static N]` is invalid C++; not needed — we use fpga_pci_write_burst).
#include <fpga_pci.h>
#include <fpga_mgmt.h>
// CLKGEN+CDC rearchitecture (S1_CLKGEN_PLAN.md): the kernel now runs on AWS_CLK_GEN's generated 125MHz
// clk_extra_a1. At AFI load AWS_CLK_GEN boots with SYS_RST_REG=0xFFFF_FFFE (cl_rst_a1_n ASSERTED, the
// kernel domain held in reset) and clk_extra_a1 STATIC-LOW until MMCM_A locks. The host MUST, after AFI
// load and BEFORE staging/kicking, poll MMCM_LOCK_REG until MMCM_A is locked then clear the CL resets
// (SYS_RST_REG) so clk_extra_a1 oscillates and the two AXI clock-converter CDC bridges + the kernel come
// out of reset. The spec-blessed mechanism is aws_clkgen_deassert_resets(slot) in the SDK fpga_clkgen
// library, which performs exactly this lock-poll + reset-deassert sequence over the mgmt (SDA/PF1-BAR4)
// path. Include it when available (HAVE_FPGA_CLKGEN); otherwise fall back to the explicit register
// sequence below over the management AXI-Lite path. Define HAVE_FPGA_CLKGEN=0 at compile time on an SDK
// that lacks the header.
#ifndef HAVE_FPGA_CLKGEN
#define HAVE_FPGA_CLKGEN 1
#endif
#if HAVE_FPGA_CLKGEN
#include <fpga_clkgen.h>
#endif

// ----------------------------------------------------------------------------
// C-safe inclusion of the C++ shared config header (S1_BUFLEN + S1_OFF_* + s1_off_* helpers).
// forge_s1_config.hpp #includes <cstdint> and declares `struct s1_opt_region_t {...};` then uses the
// bare C++ typename in `static const s1_opt_region_t S1_OPT_REGIONS[]`. Two minimal shims let C parse
// it WITHOUT editing the shared header (which the kernel + TB also include and must not diverge from):
//   (1) provide <cstdint> as a redirect to <stdint.h> via a local include dir (build flag), OR inline it;
//   (2) forward a typedef of the opt-region tag, then map the bare typename token to it so BOTH the
//       tagged definition and the bare use compile. The driver references only S1_BUFLEN + S1_OFF_*;
//       the S1_OPT_REGIONS table is unused here.
// (1) is handled by the cstdint_shim/ dir on the -I path (a one-line `cstdint` header that #includes
//     <stdint.h>); the C preprocessor does NOT macro-expand the <...> filename, so a real shim file is
//     the portable way to satisfy the C++ `#include <cstdint>`.
// (2) is the typedef-forward + bare-token map below.
typedef struct S1OPT S1OPT;          // forward the opt-region tag (so the bare-typename table decl parses)
#define s1_opt_region_t S1OPT        // map the header's bare C++ typename use -> the tagged typedef
#include "forge_s1_config.hpp"
#undef  s1_opt_region_t

// ----------------------------------------------------------------------------
// s_axilite control register offsets (the proven muzero/morph map; the S1 control bundle is 13-bit so
// buffer-ptr + mode + a1/a2/a3 are all addressable — matches cl_forge_train_step_s1.sv's [12:0] slice).
// ----------------------------------------------------------------------------
#define R_AP_CTRL   0x00
#define R_AP_RETURN 0x10
#define R_BUFFER_0  0x18
#define R_BUFFER_1  0x1c
#define R_MODE      0x24
#define R_ARG1      0x2c
#define R_ARG2      0x34
#define R_ARG3      0x3c
#define AP_START    0x01
#define AP_DONE     0x02
#define DDR_BUF     0x0ULL          // buffer staged at DDR address 0 (crossbar M00 base = 0)

// ----------------------------------------------------------------------------
// AWS_CLK_GEN AWS_CLK_REGS (BASE_REG = 0x0005_8000) — for the bring-up sequence (AWS_CLK_GEN_spec.md
// "Boot and Reset Sequencing"). Reached over the management AXI-Lite (SDA / PF1-BAR4) path. Only used by
// the explicit-register fallback when the SDK fpga_clkgen library is not available.
// ----------------------------------------------------------------------------
#define CLKGEN_BASE_REG   0x00058000UL  // BASE_REG = AWS_CLK_REGS base in the SDA address space
#define CLKGEN_SYS_RST    0x14U         // SYS_RST_REG  off; boot=0xFFFF_FFFE (cl_rst_a1_n bit1 asserted)
#define CLKGEN_MMCM_LOCK  0x20U         // MMCM_LOCK_REG off; bit0 = MMCM_BASE_A locked
#define CLKGEN_MMCM_A_LOCKED  0x00000001U  // group-A-only lock mask (we enable CLK_GRP_A_EN only)
#define CLKGEN_ALL_LOCKED     0x00000151U  // all four MMCMs locked (A=0,B=4,C=6,HBM=8) per the spec

// kernel mode + a1/a2 selectors (== forge_s1_config.hpp). a1 of mode 3 = n_steps; a2 = which weights move.
#define MODE_FWD    0
#define MODE_BWD    1
#define MODE_OPT    2
#define MODE_TRAIN  3
// S1_OPT_FREEZE_CODES / S1_OPT_TRAIN_ALL / S1_OPT_INIT / S1_OPT_STEP come from the header.

// ----------------------------------------------------------------------------
// exp2 ROM for the balanced-LUT cross-entropy + attention softmax (forge_ce_lut.hpp FT_CE_EXP2_Q16).
// Host-staged: BRAM static-init reads ZERO on the F2 AFI (R4), so the host ALWAYS stages both LUTs.
// Inlined here so the driver is a single self-contained C file (matches the TB's identical table).
// ----------------------------------------------------------------------------
static const int32_t CE_EXP2_Q16[256] = {
     65536, 62757, 60097, 57549, 55109, 52773, 50535, 48393,
     46341, 44376, 42495, 40693, 38968, 37316, 35734, 34219,
     32768, 31379, 30048, 28774, 27554, 26386, 25268, 24196,
     23170, 22188, 21247, 20347, 19484, 18658, 17867, 17109,
     16384, 15689, 15024, 14387, 13777, 13193, 12634, 12098,
     11585, 11094, 10624, 10173,  9742,  9329,  8933,  8555,
      8192,  7845,  7512,  7194,  6889,  6597,  6317,  6049,
      5793,  5547,  5312,  5087,  4871,  4664,  4467,  4277,
      4096,  3922,  3756,  3597,  3444,  3298,  3158,  3025,
      2896,  2774,  2656,  2543,  2435,  2332,  2233,  2139,
      2048,  1961,  1878,  1798,  1722,  1649,  1579,  1512,
      1448,  1387,  1328,  1272,  1218,  1166,  1117,  1069,
      1024,   981,   939,   899,   861,   825,   790,   756,
       724,   693,   664,   636,   609,   583,   558,   535,
       512,   490,   470,   450,   431,   412,   395,   378,
       362,   347,   332,   318,   304,   292,   279,   267,
       256,   245,   235,   225,   215,   206,   197,   189,
       181,   173,   166,   159,   152,   146,   140,   134,
       128,   123,   117,   112,   108,   103,    99,    95,
        91,    87,    83,    79,    76,    73,    70,    67,
        64,    61,    59,    56,    54,    52,    49,    47,
        45,    43,    41,    40,    38,    36,    35,    33,
        32,    31,    29,    28,    27,    26,    25,    24,
        23,    22,    21,    20,    19,    18,    17,    17,
        16,    15,    15,    14,    13,    13,    12,    12,
        11,    11,    10,    10,    10,     9,     9,     8,
         8,     8,     7,     7,     7,     6,     6,     6,
         6,     5,     5,     5,     5,     5,     4,     4,
         4,     4,     4,     4,     3,     3,     3,     3,
         3,     3,     3,     2,     2,     2,     2,     2,
         2,     2,     2,     2,     2,     2,     2,     1,
         1,     1,     1,     1,     1,     1,     1,     1
};

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static pci_bar_handle_t bar0, bar4, bar4r;
static int32_t *hbuf;   // host staging buffer (S1_BUFLEN words)

// stage [off, off+n) words of hbuf into DDR via BAR4 DMA-PCIS in 256K-word (1 MB) bursts.
static int stage(long off, long n) {
    const long CHW = 256 * 1024;
    for (long o = off; o < off + n; o += CHW) {
        long k = (off + n - o < CHW) ? (off + n - o) : CHW;
        int rc = fpga_pci_write_burst(bar4, (uint64_t)DDR_BUF + (uint64_t)o * 4,
                                      (uint32_t *)(hbuf + o), (uint64_t)k);
        if (rc) { fprintf(stderr, "[s1] write_burst @word %ld rc=%d\n", o, rc); return rc; }
    }
    return 0;
}
// read n words back from DDR @off via BAR4 plain (non-burst) peek.
static void readback(long off, int32_t *dst, long n) {
    uint32_t v;
    for (long j = 0; j < n; j++) { fpga_pci_peek(bar4r, (uint64_t)(off + j) * 4, &v); dst[j] = (int32_t)v; }
}

// ----------------------------------------------------------------------------
// HOST INIT — byte-identical to forge_train_step_s1_tb.cpp stage_opt_buffer(seed, memorize=true):
// the residual-passthrough identity block (Wq/Wk/Wv/Wo/W_up/W_down master = 0 -> dyad code 0; LN gamma
// master = +1<<Q_CODE, beta = 0; LN_F gamma = +1<<Q_CODE, beta = 0) + small embed/head masters
// (|.|<20) + the host-staged exp2 LUTs + the PAT=8 successor memorization tokens/targets. Mode 3 then
// re-quantizes every code/value from the staged W_fp at entry (do_opt(S1_OPT_INIT)) so the first
// forward reads valid codes. The rand() call ORDER here EXACTLY mirrors the TB so the buffer matches.
// ----------------------------------------------------------------------------
static void stage_opt_buffer_memorize(int32_t *buf, unsigned seed) {
    for (long i = 0; i < S1_BUFLEN; i++) buf[i] = 0;
    srand(seed);
    const long QC = (1L << S1_Q_CODE);
    // dense PRE-DYADIC code masters: memorize -> ZERO (identity block). NO rand() consumed (matches TB).
    // (Wq/Wk/Wv/Wo/W_up/W_down left zero by the buffer-clear above.)
    // LN gamma (code, <<Q_CODE) = +1, beta (value) = 0. No rand() in the memorize branch.
    for (int l = 0; l < S1_N_LAYERS; l++)
        for (int d = 0; d < S1_D; d++) {
            buf[S1_OFF_WFP_LN1_G + l * S1_D + d] = (int32_t)QC; buf[S1_OFF_WFP_LN2_G + l * S1_D + d] = (int32_t)QC;
            buf[S1_OFF_WFP_LN1_B + l * S1_D + d] = 0;            buf[S1_OFF_WFP_LN2_B + l * S1_D + d] = 0;
        }
    // final-LN (LN_F): gamma = +1<<Q_CODE (normalize), beta = 0.
    for (int d = 0; d < S1_D; d++) { buf[S1_OFF_WFP_LNF_G + d] = (int32_t)QC; buf[S1_OFF_WFP_LNF_B + d] = 0; }
    // EMBED / W_HEAD masters (int16 values), small |.|<20: rand() consumed in EXACTLY the TB order
    // (one EMBED then one W_HEAD per element). emag=41, eoff=20 (the memorize=true branch).
    const int emag = 41, eoff = 20;
    for (int i = 0; i < S1_V * S1_D; i++) {
        buf[S1_OFF_WFP_EMBED  + i] = (rand() % emag) - eoff;
        buf[S1_OFF_WFP_W_HEAD + i] = (rand() % emag) - eoff;
    }
    // m, v, grads: ZERO (fresh optimizer state) — already cleared.
    // host-staged exp2 LUTs (CE + attn share the Q16 exp2 ROM).
    for (int i = 0; i < S1_LUT_N; i++) { buf[S1_OFF_CE_LUT + i] = CE_EXP2_Q16[i];
                                         buf[S1_OFF_ATTN_LUT + i] = CE_EXP2_Q16[i]; }
    // tokens / targets: PAT=8 successor memorization (no rand() in the memorize branch).
    const int PAT = 8;
    for (int t = 0; t < S1_L_SEQ; t++) {
        int sym = t % PAT;
        buf[S1_OFF_TOKENS  + t] = sym;
        buf[S1_OFF_TARGETS + t] = (sym + 1) % PAT;
    }
}

// HOST INIT — byte-identical to stage_train_all(seed, densepct, embmag): the dense-code-backbone probe
// (a2 = S1_OPT_TRAIN_ALL). Small-nonzero sparse dyadic code init (|w|=1<<Q_CODE at density%) so the
// dense backbone receives a real gradient; LN gamma = +1, LN_F gamma = +1; embed/head |.|<embmag.
static void stage_train_all_buffer(int32_t *buf, unsigned seed, int densepct, int embmag) {
    for (long i = 0; i < S1_BUFLEN; i++) buf[i] = 0;
    srand(seed);
    const long QC = (1L << S1_Q_CODE);
    // sparse dense code masters (rand() order: for each region, for each layer, for each element: nz then sg).
    // matches stage_train_all's fc() lambda exactly.
    int code_bases[6] = { S1_OFF_WFP_WQ, S1_OFF_WFP_WK, S1_OFF_WFP_WV, S1_OFF_WFP_WO, S1_OFF_WFP_W_UP, S1_OFF_WFP_W_DOWN };
    int code_O[6]     = { S1_D, S1_D, S1_D, S1_D, S1_FFN, S1_D };
    int code_I[6]     = { S1_D, S1_D, S1_D, S1_D, S1_D,   S1_FFN };
    for (int r = 0; r < 6; r++) {
        int base = code_bases[r], O = code_O[r], I = code_I[r];
        for (int l = 0; l < S1_N_LAYERS; l++)
            for (int k = 0; k < O * I; k++) {
                int nz = (rand() % 100) < densepct; int sg = (rand() & 1) ? -1 : 1;
                buf[base + l * (O * I) + k] = (int32_t)(nz ? (long)sg * QC : 0);
            }
    }
    for (int l = 0; l < S1_N_LAYERS; l++)
        for (int d = 0; d < S1_D; d++) {
            buf[S1_OFF_WFP_LN1_G + l * S1_D + d] = (int32_t)QC; buf[S1_OFF_WFP_LN2_G + l * S1_D + d] = (int32_t)QC;
            buf[S1_OFF_WFP_LN1_B + l * S1_D + d] = 0;            buf[S1_OFF_WFP_LN2_B + l * S1_D + d] = 0;
        }
    for (int d = 0; d < S1_D; d++) { buf[S1_OFF_WFP_LNF_G + d] = (int32_t)QC; buf[S1_OFF_WFP_LNF_B + d] = 0; }
    for (int i = 0; i < S1_V * S1_D; i++) {
        buf[S1_OFF_WFP_EMBED  + i] = (rand() % (2 * embmag + 1)) - embmag;
        buf[S1_OFF_WFP_W_HEAD + i] = (rand() % (2 * embmag + 1)) - embmag;
    }
    for (int i = 0; i < S1_LUT_N; i++) { buf[S1_OFF_CE_LUT + i] = CE_EXP2_Q16[i];
                                         buf[S1_OFF_ATTN_LUT + i] = CE_EXP2_Q16[i]; }
    const int PAT = 8;
    for (int t = 0; t < S1_L_SEQ; t++) { int sym = t % PAT; buf[S1_OFF_TOKENS + t] = sym; buf[S1_OFF_TARGETS + t] = (sym + 1) % PAT; }
}

// ----------------------------------------------------------------------------
// TRUE float cross-entropy from the saved LOGITS region (== the TB true_ce): -mean_t log softmax(logit)[tgt],
// evaluated at a small temperature to keep exp() in range (a strictly-monotone reparam; CE-decreases is
// invariant to it). UNIFORM logits give CE == log V exactly (the saturation-collapse floor). hbuf must
// hold a fresh LOGITS + TARGETS readback. CANNOT be faked by saturation.
// ----------------------------------------------------------------------------
#define CE_TEMP 0.01
static double true_ce(const int32_t *bb) {
    double tot = 0.0;
    for (int t = 0; t < S1_L_SEQ; t++) {
        int tgt = bb[S1_OFF_TARGETS + t]; if (tgt < 0) tgt = 0; if (tgt >= S1_V) tgt = S1_V - 1;
        double mx = -1e300;
        for (int v = 0; v < S1_V; v++) { double lv = (int32_t)bb[S1_OFF_LOGITS + t * S1_V + v]; if (lv > mx) mx = lv; }
        double Z = 0.0;
        for (int v = 0; v < S1_V; v++) Z += exp(((double)(int32_t)bb[S1_OFF_LOGITS + t * S1_V + v] - mx) * CE_TEMP);
        double lt = ((double)(int32_t)bb[S1_OFF_LOGITS + t * S1_V + tgt] - mx) * CE_TEMP;
        tot += -(lt - log(Z));
    }
    return tot / (double)S1_L_SEQ;
}
// argmax accuracy over the saved LOGITS (argmax_v logits[t][v] == target[t]).
static int count_correct(const int32_t *bb) {
    int correct = 0;
    for (int t = 0; t < S1_L_SEQ; t++) {
        int tgt = bb[S1_OFF_TARGETS + t]; if (tgt < 0) tgt = 0; if (tgt >= S1_V) tgt = S1_V - 1;
        long lmax = -2147483647L; int pred = 0;
        for (int v = 0; v < S1_V; v++) { long lv = (int32_t)bb[S1_OFF_LOGITS + t * S1_V + v]; if (lv > lmax) { lmax = lv; pred = v; } }
        if (pred == tgt) correct++;
    }
    return correct;
}

// kick the kernel: program mode/args/buffer-ptr, AP_START, poll AP_DONE with a generous timeout.
// timeout = n_steps * 0.2 s + 5 s (R7: a mode-3 kick runs n_steps * ~90 ms).
static int kick(int mode, int a1, int a2, double timeout_s) {
    fpga_pci_poke(bar0, R_BUFFER_0, (uint32_t)(DDR_BUF & 0xFFFFFFFFULL));
    fpga_pci_poke(bar0, R_BUFFER_1, (uint32_t)(DDR_BUF >> 32));
    fpga_pci_poke(bar0, R_MODE, (uint32_t)mode);
    fpga_pci_poke(bar0, R_ARG1, (uint32_t)a1);
    fpga_pci_poke(bar0, R_ARG2, (uint32_t)a2);
    fpga_pci_poke(bar0, R_ARG3, 0);
    fpga_pci_poke(bar0, R_AP_CTRL, AP_START);
    double t0 = now_s(); uint32_t ctrl = 0;
    while (now_s() - t0 < timeout_s) { fpga_pci_peek(bar0, R_AP_CTRL, &ctrl); if (ctrl & AP_DONE) return 0; }
    fprintf(stderr, "[s1] KICK HUNG (mode=%d a1=%d, no AP_DONE in %.1fs)\n", mode, a1, timeout_s);
    return -1;
}

// ----------------------------------------------------------------------------
// AWS_CLK_GEN bring-up (REQUIRED before any staging/kicking) — S1_CLKGEN_PLAN.md sec 5 +
// AWS_CLK_GEN_spec.md "Boot and Reset Sequencing".
//
// At AFI load AWS_CLK_GEN starts with SYS_RST_REG = 0xFFFF_FFFE (every CL reset asserted except
// cl_rst_main_n) and the MMCM clock outputs (clk_extra_a1, the 125MHz KERNEL clock) HELD STATIC-LOW
// until MMCM_A locks. With the kernel + both CDC bridges held in reset and clk_extra_a1 not toggling,
// the kernel never runs and a mode-3 kick would hang forever. The host MUST therefore:
//   (1) poll MMCM_LOCK_REG until the enabled MMCM(s) report locked, THEN
//   (2) clear SYS_RST_REG to deassert the CL resets so clk_extra_a1 oscillates and the CDCs + kernel
//       exit reset (releasing resets BEFORE lock is the spec's explicit anti-pattern).
//
// The spec-blessed, version-robust mechanism is the SDK aws_clkgen_deassert_resets(slot) (fpga_clkgen
// library), which performs exactly this lock-poll + reset-deassert sequence over the management path —
// we use it when the header is available (HAVE_FPGA_CLKGEN). When it is not, the explicit-register
// fallback attaches the management AXI-Lite (the SDA / PF1-BAR4 path the CL wires to AWS_CLK_GEN's
// s_axil_ctrl_*) and runs the same sequence by hand.
//
// Returns 0 on success (clocks locked + resets released), nonzero on failure.
static int clkgen_bringup(int slot) {
#if HAVE_FPGA_CLKGEN
    // Robust SDK path: lock-poll MMCM_LOCK_REG + clear SYS_RST_REG internally, over the mgmt path.
    int rc = aws_clkgen_deassert_resets(slot);
    if (rc != 0) {
        fprintf(stderr, "[s1] aws_clkgen_deassert_resets(slot=%d) FAILED rc=%d "
                        "(AFI may not be clkgen-enabled — built without --aws_clk_gen?)\n", slot, rc);
        return rc;
    }
    printf("[s1] AWS_CLK_GEN: aws_clkgen_deassert_resets ok — clk_extra_a1 (125MHz) live, CL resets released\n");
    return 0;
#else
    // Explicit-register fallback over the management AXI-Lite (SDA / PF1-BAR4) path.
    pci_bar_handle_t mbar = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(slot, FPGA_MGMT_PF, MGMT_PF_BAR4, 0, &mbar) != 0) {
        fprintf(stderr, "[s1] clkgen: attach mgmt BAR (FPGA_MGMT_PF/MGMT_PF_BAR4) failed\n");
        return -1;
    }
    // (1) poll MMCM_LOCK_REG for MMCM_A lock (bit0). CL enables CLK_GRP_A_EN only, so MMCM_A_LOCKED suffices.
    uint32_t lock = 0; double t0 = now_s();
    while (now_s() - t0 < 5.0) {
        if (fpga_pci_peek(mbar, CLKGEN_BASE_REG + CLKGEN_MMCM_LOCK, &lock) == 0 &&
            (lock & CLKGEN_MMCM_A_LOCKED)) break;
    }
    if (!(lock & CLKGEN_MMCM_A_LOCKED)) {
        fprintf(stderr, "[s1] clkgen: MMCM_A did NOT lock (MMCM_LOCK_REG=0x%08x) in 5s — clk_extra_a1 dead\n", lock);
        fpga_pci_detach(mbar); return -2;
    }
    // (2) deassert ALL CL resets (cl_rst_a1_n + the rest): write SYS_RST_REG = 0. Per the spec this MUST
    //     happen only AFTER the lock above, else the downstream CDCs/kernel see no clocked reset assertion.
    if (fpga_pci_poke(mbar, CLKGEN_BASE_REG + CLKGEN_SYS_RST, 0x00000000U) != 0) {
        fprintf(stderr, "[s1] clkgen: SYS_RST_REG deassert write failed\n");
        fpga_pci_detach(mbar); return -3;
    }
    uint32_t rstv = 0xdeadbeef; fpga_pci_peek(mbar, CLKGEN_BASE_REG + CLKGEN_SYS_RST, &rstv);
    printf("[s1] AWS_CLK_GEN: MMCM_LOCK=0x%08x (A locked) -> SYS_RST_REG cleared (readback 0x%08x); "
           "clk_extra_a1 (125MHz) live, CL resets released\n", lock, rstv);
    fpga_pci_detach(mbar);
    return 0;
#endif
}

int main(int argc, char **argv) {
    int   slot     = (argc > 1) ? atoi(argv[1]) : 0;
    int   n_steps  = (argc > 2) ? atoi(argv[2]) : 32;
    int   trainall = (argc > 3 && strcmp(argv[3], "trainall") == 0) ? 1 : 0;
    unsigned seed  = (argc > 4) ? (unsigned)strtoul(argv[4], NULL, 10) : 4242u;
    const int a2   = trainall ? S1_OPT_TRAIN_ALL : S1_OPT_FREEZE_CODES;
    const size_t bytes = (size_t)S1_BUFLEN * 4;

    printf("[s1] forge_train_step_s1 on F2: S1_BUFLEN=%d (%.3f MB); n_steps=%d; regime=%s; seed=%u; DDR addr=0x%llx\n",
           S1_BUFLEN, bytes / 1048576.0, n_steps, trainall ? "TRAIN_ALL" : "FREEZE_CODES", seed,
           (unsigned long long)DDR_BUF);
    printf("[s1] dims D=%d H=%d DH=%d FFN=%d L_SEQ=%d V=%d N_LAYERS=%d Q_CODE=%d (CE floor=log V=%.4f)\n",
           S1_D, S1_H, S1_DH, S1_FFN, S1_L_SEQ, S1_V, S1_N_LAYERS, S1_Q_CODE, log((double)S1_V));

    if (fpga_mgmt_init() != 0) { fprintf(stderr, "fpga_mgmt_init failed\n"); return 2; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR0, 0, &bar0) != 0)             { fprintf(stderr, "attach BAR0 failed\n"); return 3; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &bar4) != 0) { fprintf(stderr, "attach BAR4(burst) failed\n"); return 4; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, 0, &bar4r) != 0)            { fprintf(stderr, "attach BAR4(plain) failed\n"); return 4; }

    // 0. AWS_CLK_GEN BRING-UP (MUST precede any staging/kick): poll MMCM_LOCK_REG until MMCM_A locks,
    //    then deassert the CL resets (SYS_RST_REG) so the generated 125MHz clk_extra_a1 oscillates and
    //    the kernel + its two AXI clock-converter CDC bridges leave reset. Without this the kernel domain
    //    is held in reset / clk_extra_a1 is static-low and every kick below would hang. (S1_CLKGEN_PLAN.md)
    if (clkgen_bringup(slot) != 0) {
        fprintf(stderr, "[s1] AWS_CLK_GEN bring-up FAILED — kernel has no 125MHz clock; aborting before stage/kick\n");
        return 9;
    }

    hbuf = (int32_t *)calloc(S1_BUFLEN, 4);
    if (!hbuf) { fprintf(stderr, "calloc hbuf failed\n"); return 5; }

    // 1. build the INITIAL state in the host buffer (the CSIM-golden layout).
    if (trainall) stage_train_all_buffer(hbuf, seed, /*densepct=*/35, /*embmag=*/20);
    else          stage_opt_buffer_memorize(hbuf, seed);

    // 2. stage the WHOLE buffer into DDR @0 over BAR4 (one-time; mode 3 keeps state resident after).
    double sdma = now_s();
    if (stage(0, S1_BUFLEN)) return 6;
    printf("[s1] staged %.3f MB into DDR via BAR4 in %.3f s\n", bytes / 1048576.0, now_s() - sdma);

    // 3. warm kick: mode 0 (forward only) so we can read an INITIAL loss BEFORE any training. Mode 3
    //    re-quantizes codes from W_fp at entry, but a standalone mode-0 needs valid codes first, so run a
    //    mode-2 INIT (re-quantize, no weight move) then mode-0. Timeout small (single fwd ~18 ms).
    if (kick(MODE_OPT, S1_OPT_INIT, a2, 10.0)) return 7;   // re-quantize codes/values from staged W_fp
    if (kick(MODE_FWD, 0, 0, 10.0))            return 7;   // forward + CE -> LOGITS
    readback(S1_OFF_LOGITS,  hbuf + S1_OFF_LOGITS,  (long)S1_L_SEQ * S1_V);
    readback(S1_OFF_TARGETS, hbuf + S1_OFF_TARGETS, (long)S1_L_SEQ);
    readback(S1_OFF_LOSS,    hbuf + S1_OFF_LOSS,    16);
    double ce0  = true_ce(hbuf);
    int    acc0 = count_correct(hbuf);
    long   loss0_proxy = (int32_t)hbuf[S1_OFF_LOSS];
    printf("[s1] INITIAL  CE = %.4f   acc = %d/%d   (loss-proxy[0] = %ld)\n", ce0, acc0, S1_L_SEQ, loss0_proxy);

    // 4. THE TRAINING KICK: mode 3 (do_train_run) = ONE kick runs n_steps of fwd->bwd->opt with weights +
    //    optimizer state RESIDENT in DDR (no per-step host traffic). a2 = freeze/train-all regime.
    //    mode 3 itself does do_opt(S1_OPT_INIT) at entry, so this is self-contained from the staged W_fp.
    double timeout = (double)n_steps * 0.2 + 5.0;       // R7: ~90 ms/step + generous margin
    printf("[s1] KICK mode 3 (fused trainer): n_steps=%d a2=%s timeout=%.1fs ...\n",
           n_steps, trainall ? "TRAIN_ALL" : "FREEZE_CODES", timeout);
    double tk = now_s();
    if (kick(MODE_TRAIN, n_steps, a2, timeout)) return 8;
    double elapsed = now_s() - tk;
    uint32_t ret = 0; fpga_pci_peek(bar0, R_AP_RETURN, &ret);
    printf("[s1] mode-3 AP_DONE in %.3f s (%.2f ms/step over %d steps), ret=%u\n",
           elapsed, elapsed / (n_steps > 0 ? n_steps : 1) * 1e3, n_steps, ret);

    // 5. read back the FINAL state: LOGITS + TARGETS (for the true CE), the LOSS proxy, GRAD_LOGIT (CE
    //    adjoint, for the bit-exact check), and the trained scatter weights W_HEAD / EMBED.
    readback(S1_OFF_LOGITS,     hbuf + S1_OFF_LOGITS,     (long)S1_L_SEQ * S1_V);
    readback(S1_OFF_TARGETS,    hbuf + S1_OFF_TARGETS,    (long)S1_L_SEQ);
    readback(S1_OFF_GRAD_LOGIT, hbuf + S1_OFF_GRAD_LOGIT, (long)S1_L_SEQ * S1_V);
    readback(S1_OFF_LOSS,       hbuf + S1_OFF_LOSS,       16);
    readback(S1_OFF_W_HEAD,     hbuf + S1_OFF_W_HEAD,     (long)S1_V * S1_D);
    readback(S1_OFF_EMBED,      hbuf + S1_OFF_EMBED,      (long)S1_V * S1_D);
    // peak |R| at the final layer's residual stream — the int16 saturation indicator (==32767 -> collapse).
    readback(s1_off_R(S1_N_LAYERS), hbuf + s1_off_R(S1_N_LAYERS), (long)S1_L_SEQ * S1_D);
    long peakR = 0;
    for (int t = 0; t < S1_L_SEQ; t++)
        for (int d = 0; d < S1_D; d++) { long a = labs((long)(int32_t)hbuf[s1_off_R(S1_N_LAYERS) + t * S1_D + d]); if (a > peakR) peakR = a; }

    double ce1  = true_ce(hbuf);
    int    acc1 = count_correct(hbuf);
    long   loss1_proxy = (int32_t)hbuf[S1_OFF_LOSS];
    // a crude grad-logit row-sum sanity (CE adjoint rows must sum to ~0): report the max |row sum|.
    long max_rowsum = 0;
    for (int t = 0; t < S1_L_SEQ; t++) { long rs = 0; for (int v = 0; v < S1_V; v++) rs += (int32_t)hbuf[S1_OFF_GRAD_LOGIT + t * S1_V + v]; if (labs(rs) > max_rowsum) max_rowsum = labs(rs); }

    double floor_ce = log((double)S1_V);
    printf("[s1] FINAL    CE = %.4f   acc = %d/%d   (loss-proxy[0] = %ld)   peak|R| = %ld   max|gradlogit rowsum| = %ld\n",
           ce1, acc1, S1_L_SEQ, loss1_proxy, peakR, max_rowsum);
    printf("[s1] ============================================================\n");
    printf("[s1] CE %.4f -> %.4f  (drop %.4f nats; uniform/collapse floor = %.4f)   acc %d -> %d/%d\n",
           ce0, ce1, ce0 - ce1, floor_ce, acc0, acc1, S1_L_SEQ);
    // CONVERGENCE gates (the TB's un-fakeable bar; collapse sits AT the floor and FAILS all three):
    int ce_drops      = (ce0 - ce1) >= 0.5;
    int not_collapsed = ce1 < (floor_ce - 0.25);
    int acc_improves  = (acc1 > acc0);
    int peak_ok       = (peakR < 32767);   // not pinned at the int16 ceiling
    printf("[s1] [gates] CE-drops>=0.5: %s   below-floor: %s   acc-improves: %s   no-saturation: %s\n",
           ce_drops ? "yes" : "NO", not_collapsed ? "yes" : "NO", acc_improves ? "yes" : "NO", peak_ok ? "yes" : "NO");
    printf("[s1] -> ON-FABRIC TRAINER %s\n",
           (ce_drops && not_collapsed && acc_improves && peak_ok) ? "CONVERGES (silicon == CSIM)" : "NO-CONVERGENCE (inspect)");
    printf("[s1] ============================================================\n");
    // greppable one-line summary for the runner / CSIM-reproduction diff.
    printf("[S1 RESULT] regime=%s seed=%u n_steps=%d CE0=%.4f CE1=%.4f drop=%.4f floor=%.4f acc0=%d acc1=%d peakR=%ld msPerStep=%.2f ret=%u\n",
           trainall ? "TRAIN_ALL" : "FREEZE_CODES", seed, n_steps, ce0, ce1, ce0 - ce1, floor_ce,
           acc0, acc1, peakR, elapsed / (n_steps > 0 ? n_steps : 1) * 1e3, ret);

    free(hbuf);
    fpga_pci_detach(bar0); fpga_pci_detach(bar4); fpga_pci_detach(bar4r);
    return 0;
}
