// forge_s1_infer_f2.c — INFERENCE / GENERATE the multiplier-free tropical LLM ENTIRELY ON F2 SILICON
// (forward-only, no GPU, no training), driving the forge_train_step_s1 FORWARD kernel (mode 0 = do_fwd_loss)
// built with the DEPLOY defines:
//   -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1 -DS1_N_LAYERS=2
// (csynth-confirmed DSP=0 / 2.920ns / fits one SLR). This is the inference twin of the trainer
// forge_train_step_s1_f2.c: it REUSES that driver's proven infrastructure (AWS_CLK_GEN bring-up, BAR4
// DMA-PCIS staging, the kick(MODE,a1,a2)+readback helpers, the C-safe shared-header inclusion shims) but
// runs ONLY mode-0 forwards — NO do_bwd / do_opt / do_train_run.
//
// WHAT IT DOES (two forward-only passes over the SAME staged weights):
//   (1) GENERATE: autoregressive sampling. Keep a rolling L_SEQ-length context of token ids; each step write
//       the context to S1_OFF_TOKENS, kick(MODE_FWD), read S1_OFF_LOGITS, host-argmax/sample the LAST valid
//       position's V logits, append the chosen char, slide the window, repeat. Prints the generated text.
//   (2) HELD-OUT BPC: slide L_SEQ windows over a val-id stream (int32 .bin), write TOKENS+TARGETS, kick
//       (MODE_FWD), read the per-position logits, accumulate the TRUE held-out bits/char
//       (mean -log2 softmax(logit)[target] over the REAL vocab columns). The same metric the g++
//       s1_deploy_eval.cpp computes off-fabric — so the on-silicon bpc is directly comparable to the csim
//       re-gate (and to bigram 3.47 / torch HARD 2.4-2.6).
//
// DATA PATH (the proven cl_forge_muzero_ddr / forge_train_step_s1_f2 DDR topology; PCIM is not drivable on
// EC2 Nitro): host BAR0 (OCL AXI4-Lite) -> kernel s_axi_control (mode/args/buffer-ptr + kick/poll); host
// BAR4 (DMA-PCIS, 512b) -> axi_crossbar_ddr -> sh_ddr -> DDR4 (stage + readback); the kernel reads/writes
// its whole working set in on-card DDR at addr 0. The host stages the EXPORTED deploy buffer (the int32
// .npy from scripts/tropical_llm_export_s1.py --faithful) once, then issues only mode-0 kicks.
//
// CLKGEN: identical to the trainer — AWS_CLK_GEN bring-up (lock-poll MMCM_A + deassert CL resets) MUST run
// after AFI load and before any stage/kick, else the kernel's 125MHz clk_extra_a1 is static-low and a kick
// hangs. Use the SDK fpga_clkgen lib when available, else -DHAVE_FPGA_CLKGEN=0 for the explicit-register
// fallback (the deploy-runner default; compile-checks against the SDK stubs with that define).
//
// Build on an F2 with the aws-fpga SDK (compile-as-C; ABI + offset map from forge_s1_config.hpp via the two
// C-safe include shims, same as forge_train_step_s1_f2.c):
//   gcc -O2 -std=gnu11 -DHAVE_FPGA_CLKGEN=0 \
//       -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1 -DS1_N_LAYERS=2 \
//       -I$SDK/include \
//       -I<repo>/tikitorch/hardware/hls/forge_kernels \
//       -I<repo>/tikitorch/fpga/silicon/cstdint_shim \
//       forge_s1_infer_f2.c -lfpga_mgmt -o s1_infer
//   (drop -DHAVE_FPGA_CLKGEN=0 and add -lfpga_clkgen if the SDK has <fpga_clkgen.h>.)
//   The deploy defines MUST match the AFI / the export buffer (S1_BUFLEN is checked against the .npy length).
//
// Run:
//   sudo ./s1_infer <slot> <buf.npy> [gen=200] [valbin] [nwin=63] [temp=256] [seed=1234] [vocab.txt]
//     <slot>     FPGA slot (0)
//     <buf.npy>  the exported int32 deploy buffer (scripts/tropical_llm_export_s1.py --faithful --out ...)
//     gen        number of chars to autoregressively generate (0 = skip generation)
//     valbin     int32 held-out char ids (the bpc pass; omit / "-" to skip)
//     nwin       held-out windows to score
//     temp       sampling/softmax logit temperature (logit/temp; argmax if temp<=0)
//     seed       RNG seed for sampling
//     vocab.txt  optional: a file whose first line's chars map id->char for human-readable text
//                (else prints raw ids). For tinyshakespeare-lowercase the natural sorted-unique-char order.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>
#ifndef HAVE_FPGA_CLKGEN
#define HAVE_FPGA_CLKGEN 1
#endif
#if HAVE_FPGA_CLKGEN
#include <fpga_clkgen.h>
#endif

// ----------------------------------------------------------------------------
// C-safe inclusion of the C++ shared config header (S1_BUFLEN + S1_OFF_* offsets) — IDENTICAL shim pair to
// forge_train_step_s1_f2.c: a local cstdint redirect (on the -I path) + a typedef-forward of the bare opt-
// region typename so the C preprocessor can parse the header without editing it. We reference only S1_BUFLEN
// + S1_OFF_*; the S1_OPT_REGIONS table is unused here (inference does not touch the optimizer regions).
// ----------------------------------------------------------------------------
typedef struct S1OPT S1OPT;
#define s1_opt_region_t S1OPT
#include "forge_s1_config.hpp"
#undef  s1_opt_region_t

// ----------------------------------------------------------------------------
// s_axilite control register offsets (== forge_train_step_s1_f2.c / cl_forge_train_step_s1.sv [12:0]).
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

// AWS_CLK_GEN regs (== forge_train_step_s1_f2.c) — explicit-register fallback bring-up path.
#define CLKGEN_BASE_REG   0x00058000UL
#define CLKGEN_SYS_RST    0x14U
#define CLKGEN_MMCM_LOCK  0x20U
#define CLKGEN_MMCM_A_LOCKED  0x00000001U

#define MODE_FWD    0              // do_fwd_loss (the ONLY mode this driver issues)

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static pci_bar_handle_t bar0, bar4, bar4r;
static int32_t *hbuf;   // host staging buffer (S1_BUFLEN words)

// stage [off, off+n) words of hbuf into DDR via BAR4 DMA-PCIS in 256K-word (1 MB) bursts. (== trainer stage())
static int stage(long off, long n) {
    const long CHW = 256 * 1024;
    for (long o = off; o < off + n; o += CHW) {
        long k = (off + n - o < CHW) ? (off + n - o) : CHW;
        int rc = fpga_pci_write_burst(bar4, (uint64_t)DDR_BUF + (uint64_t)o * 4,
                                      (uint32_t *)(hbuf + o), (uint64_t)k);
        if (rc) { fprintf(stderr, "[infer] write_burst @word %ld rc=%d\n", o, rc); return rc; }
    }
    return 0;
}
// read n words back from DDR @off via BAR4 plain (non-burst) peek. (== trainer readback())
static void readback(long off, int32_t *dst, long n) {
    uint32_t v;
    for (long j = 0; j < n; j++) { fpga_pci_peek(bar4r, (uint64_t)(off + j) * 4, &v); dst[j] = (int32_t)v; }
}

// kick the kernel: program mode/args/buffer-ptr, AP_START, poll AP_DONE. (== trainer kick())
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
    fprintf(stderr, "[infer] KICK HUNG (mode=%d a1=%d, no AP_DONE in %.1fs)\n", mode, a1, timeout_s);
    return -1;
}

// AWS_CLK_GEN bring-up (REQUIRED before any stage/kick) — IDENTICAL to forge_train_step_s1_f2.c clkgen_bringup.
static int clkgen_bringup(int slot) {
#if HAVE_FPGA_CLKGEN
    int rc = aws_clkgen_deassert_resets(slot);
    if (rc != 0) {
        fprintf(stderr, "[infer] aws_clkgen_deassert_resets(slot=%d) FAILED rc=%d "
                        "(AFI may not be clkgen-enabled — built without --aws_clk_gen?)\n", slot, rc);
        return rc;
    }
    printf("[infer] AWS_CLK_GEN: aws_clkgen_deassert_resets ok — clk_extra_a1 (125MHz) live, CL resets released\n");
    return 0;
#else
    pci_bar_handle_t mbar = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(slot, FPGA_MGMT_PF, MGMT_PF_BAR4, 0, &mbar) != 0) {
        fprintf(stderr, "[infer] clkgen: attach mgmt BAR (FPGA_MGMT_PF/MGMT_PF_BAR4) failed\n");
        return -1;
    }
    uint32_t lock = 0; double t0 = now_s();
    while (now_s() - t0 < 5.0) {
        if (fpga_pci_peek(mbar, CLKGEN_BASE_REG + CLKGEN_MMCM_LOCK, &lock) == 0 &&
            (lock & CLKGEN_MMCM_A_LOCKED)) break;
    }
    if (!(lock & CLKGEN_MMCM_A_LOCKED)) {
        fprintf(stderr, "[infer] clkgen: MMCM_A did NOT lock (MMCM_LOCK_REG=0x%08x) in 5s — clk_extra_a1 dead\n", lock);
        fpga_pci_detach(mbar); return -2;
    }
    if (fpga_pci_poke(mbar, CLKGEN_BASE_REG + CLKGEN_SYS_RST, 0x00000000U) != 0) {
        fprintf(stderr, "[infer] clkgen: SYS_RST_REG deassert write failed\n");
        fpga_pci_detach(mbar); return -3;
    }
    uint32_t rstv = 0xdeadbeef; fpga_pci_peek(mbar, CLKGEN_BASE_REG + CLKGEN_SYS_RST, &rstv);
    printf("[infer] AWS_CLK_GEN: MMCM_LOCK=0x%08x (A locked) -> SYS_RST_REG cleared "
           "(raw 0x%08x; writable[9:0]=0x%03x, expect 0x000); clk_extra_a1 (kernel clk) live, resets released\n",
           lock, rstv, rstv & 0x3FFu);
    fpga_pci_detach(mbar);
    return 0;
#endif
}

// ----------------------------------------------------------------------------
// Minimal .npy (int32, 1-D, little-endian, C order) loader — reads the export buffer into a heap array.
// Mirrors s1_deploy_eval.cpp load_npy_i32 in plain C. Returns malloc'd int32 array; sets *n_out.
// ----------------------------------------------------------------------------
static int32_t *load_npy_i32(const char *path, long *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[infer] cannot open %s\n", path); return NULL; }
    unsigned char magic[6]; size_t r = fread(magic, 1, 6, f);
    if (r != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) { fprintf(stderr, "[infer] not a .npy: %s\n", path); fclose(f); return NULL; }
    unsigned char ver[2]; r = fread(ver, 1, 2, f); (void)r;
    uint16_t hlen = 0; r = fread(&hlen, 2, 1, f); (void)r;          // v1.0 header length (uint16 LE)
    char *hdr = (char *)calloc(hlen + 1, 1);
    r = fread(hdr, 1, hlen, f); (void)r;
    long n = 0;
    char *sp = strstr(hdr, "'shape':");
    if (sp) { char *lp = strchr(sp, '('); if (lp) sscanf(lp + 1, "%ld", &n); }
    if (!strstr(hdr, "<i4")) fprintf(stderr, "[infer][warn] .npy dtype not <i4: %s\n", hdr);
    free(hdr);
    if (n <= 0) { fprintf(stderr, "[infer] bad .npy shape in %s\n", path); fclose(f); return NULL; }
    int32_t *v = (int32_t *)malloc((size_t)n * 4);
    r = fread(v, 4, n, f);
    if ((long)r != n) { fprintf(stderr, "[infer] short read %s: %zu != %ld\n", path, r, n); free(v); fclose(f); return NULL; }
    fclose(f);
    *n_out = n;
    return v;
}

// raw int32 binary loader (held-out val ids).
static int32_t *load_bin_i32(const char *path, long *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[infer] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    long n = bytes / 4;
    int32_t *v = (int32_t *)malloc((size_t)(n > 0 ? n : 1) * 4);
    size_t r = fread(v, 4, n, f); (void)r;
    fclose(f);
    *n_out = n;
    return v;
}

// ----------------------------------------------------------------------------
// VREAL — the REAL vocab size (the model's embed/head rows; padded rows V..VREAL are never valid targets).
// The deploy export pads to S1_V=64 but the tinyshakespeare-lowercase vocab is 39. Sampling/bpc restrict to
// [0, VREAL) so a padded logit can never be chosen / scored. Overridable at compile time.
// ----------------------------------------------------------------------------
#ifndef S1_VREAL
#define S1_VREAL 39
#endif

// sample (or argmax) one token from a V-logit row over the REAL vocab [0,VREAL), at logit temperature `temp`.
// temp<=0 -> deterministic argmax. Numerically-stable softmax then inverse-CDF draw with a uniform [0,1).
static int sample_logits(const int32_t *logits, int vreal, double temp, unsigned *rngstate) {
    if (temp <= 0.0) {                                   // argmax
        long best = -2147483647L; int bi = 0;
        for (int v = 0; v < vreal; v++) { long lv = (int32_t)logits[v]; if (lv > best) { best = lv; bi = v; } }
        return bi;
    }
    double mx = -1e300;
    for (int v = 0; v < vreal; v++) { double z = (double)(int32_t)logits[v] / temp; if (z > mx) mx = z; }
    double sum = 0.0;
    for (int v = 0; v < vreal; v++) sum += exp((double)(int32_t)logits[v] / temp - mx);
    // rand_r-style LCG for a reproducible uniform (avoids global rand() state; portable, self-contained).
    *rngstate = (*rngstate) * 1103515245u + 12345u;
    double u = ((*rngstate >> 16) & 0x7fff) / 32768.0;
    double acc = 0.0;
    for (int v = 0; v < vreal; v++) {
        acc += exp((double)(int32_t)logits[v] / temp - mx) / sum;
        if (u <= acc) return v;
    }
    return vreal - 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <slot> <buf.npy> [gen=200] [valbin|-] [nwin=63] [temp=256] [seed=1234] [vocab.txt]\n", argv[0]);
        return 1;
    }
    int          slot    = atoi(argv[1]);
    const char  *bufpath = argv[2];
    int          gen     = (argc > 3) ? atoi(argv[3]) : 200;
    const char  *valpath = (argc > 4 && strcmp(argv[4], "-") != 0) ? argv[4] : NULL;
    int          nwin    = (argc > 5) ? atoi(argv[5]) : 63;
    double       temp    = (argc > 6) ? atof(argv[6]) : 256.0;
    unsigned     seed    = (argc > 7) ? (unsigned)strtoul(argv[7], NULL, 10) : 1234u;
    const char  *vocpath = (argc > 8) ? argv[8] : NULL;
    const int    vreal   = S1_VREAL;
    const size_t bytes   = (size_t)S1_BUFLEN * 4;

    printf("[infer] forge_train_step_s1 FORWARD on F2: S1_BUFLEN=%d (%.3f MB) D=%d H=%d FFN=%d L_SEQ=%d V=%d(real=%d) N_LAYERS=%d\n",
           S1_BUFLEN, bytes / 1048576.0, S1_D, S1_H, S1_FFN, S1_L_SEQ, S1_V, vreal, S1_N_LAYERS);
    printf("[infer] mode: FORWARD/INFERENCE ONLY (no bwd/opt/train); gen=%d temp=%.1f seed=%u; DDR addr=0x%llx\n",
           gen, temp, seed, (unsigned long long)DDR_BUF);

    // optional vocab table for human-readable output. The vocab file is the LITERAL id->char map: byte k is
    // the char for token id k (the tinyshakespeare-lowercase sorted-unique set, whose id 0 is '\n'). Read up
    // to 256 raw bytes — do NOT stop at a newline (the char set legitimately contains '\n' as id 0).
    char id2ch[256]; int have_voc = 0, nvoc = 0;
    if (vocpath) {
        FILE *vf = fopen(vocpath, "rb");
        if (vf) { nvoc = (int)fread(id2ch, 1, 256, vf); fclose(vf); have_voc = (nvoc > 0); }
        if (!have_voc) fprintf(stderr, "[infer][warn] could not read vocab from %s — printing raw ids\n", vocpath);
        else printf("[infer] vocab: %d chars from %s (id 0 = 0x%02x)\n", nvoc, vocpath, (unsigned char)id2ch[0]);
    }

    // load the exported deploy buffer
    long blen = 0; int32_t *buf0 = load_npy_i32(bufpath, &blen);
    if (!buf0) return 2;
    if (blen != S1_BUFLEN) {
        fprintf(stderr, "[infer][FATAL] buffer length %ld != S1_BUFLEN %d — rebuild the export at matching deploy defines "
                        "(-DS1_N_LAYERS=2 -DS1_POS_EMBED=1 -DS1_HEAD_MODE=1 -DS1_ATTN_HARD=1 -DS1_FWD_ONLY=1)\n", blen, S1_BUFLEN);
        free(buf0); return 3;
    }

    // ---- FPGA bring-up (mgmt init, attach BAR0/BAR4 burst+plain, clkgen) ----
    if (fpga_mgmt_init() != 0) { fprintf(stderr, "[infer] fpga_mgmt_init failed\n"); return 4; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR0, 0, &bar0) != 0)             { fprintf(stderr, "[infer] attach BAR0 failed\n"); return 5; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &bar4) != 0) { fprintf(stderr, "[infer] attach BAR4(burst) failed\n"); return 5; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, 0, &bar4r) != 0)            { fprintf(stderr, "[infer] attach BAR4(plain) failed\n"); return 5; }
    if (clkgen_bringup(slot) != 0) {
        fprintf(stderr, "[infer] AWS_CLK_GEN bring-up FAILED — kernel has no 125MHz clock; aborting before stage/kick\n");
        return 6;
    }

    hbuf = (int32_t *)calloc(S1_BUFLEN, 4);
    if (!hbuf) { fprintf(stderr, "[infer] calloc hbuf failed\n"); free(buf0); return 7; }
    memcpy(hbuf, buf0, bytes);

    // ---- stage the WHOLE deploy buffer into DDR @0 once (weights resident; all later kicks are mode-0) ----
    double sdma = now_s();
    if (stage(0, S1_BUFLEN)) { free(buf0); free(hbuf); return 8; }
    printf("[infer] staged %.3f MB into DDR via BAR4 in %.3f s\n", bytes / 1048576.0, now_s() - sdma);

    // POST-STAGE SANITY READBACK + FLUSH: the BAR4 burst writes are posted; a read-after-write (bar4r, same
    // physical BAR4) forces them to land before the first kick and localizes a staging-path fault (the v1
    // silicon symptom was peak-logits 0 / uniform because the kick raced ahead of the staged weights). Probe
    // representative nonzero regions of the deploy buffer (EMBED, W_HEAD, CE_LUT, POS).
    {
        long pr[4]; pr[0] = S1_OFF_EMBED; pr[1] = S1_OFF_W_HEAD; pr[2] = S1_OFF_CE_LUT;
#if S1_POS_EMBED
        pr[3] = S1_OFF_POS;
#else
        pr[3] = S1_OFF_TOKENS;
#endif
        const char *pn[4] = { "EMBED", "W_HEAD", "CE_LUT", "POS/TOK" };
        int total_mism = 0;
        for (int p = 0; p < 4; p++) {
            int32_t rb[8]; readback(pr[p], rb, 8);
            int m = 0; for (int j = 0; j < 8; j++) if (rb[j] != hbuf[pr[p] + j]) m++;
            total_mism += m;
            printf("[infer] stage-check %-7s @%ld: ddr={%d,%d,%d,%d} host={%d,%d,%d,%d} mism=%d/8\n",
                   pn[p], pr[p], rb[0], rb[1], rb[2], rb[3],
                   hbuf[pr[p]], hbuf[pr[p]+1], hbuf[pr[p]+2], hbuf[pr[p]+3], m);
        }
        printf("[infer] POST-STAGE READBACK: %s (%d mismatches / 32 probed words)\n",
               total_mism == 0 ? "MATCH (weights landed)" : "MISMATCH (staging did NOT land)", total_mism);
    }

    // ========================================================================
    // (1) GENERATE — autoregressive sampling on F2. A rolling L_SEQ-length context of token ids drives
    //     each mode-0 forward; the LAST valid position's logits are sampled, the chosen char appended, the
    //     window slid. The kernel processes the WHOLE L_SEQ; positions past `ctxlen` are filled with the
    //     last real token (a benign pad — only the position at index `last` is read for the next token).
    // ========================================================================
    if (gen > 0) {
        int32_t ctx[S1_L_SEQ];
        int     ctxlen = 1;
        unsigned rng = seed ? seed : 1u;
        // seed prompt: a single token id 0 (vocab char 0). (A richer prompt could be loaded; id 0 is a clean
        // deterministic start. The model has no BOS — position 0 with pos-embed[0] is a valid context start.)
        ctx[0] = 0;
        printf("[infer] ============================================================\n");
        printf("[infer] GENERATE %d chars (rolling ctx<=%d, temp=%.1f):\n", gen, S1_L_SEQ, temp);
        char *out = (char *)malloc((size_t)gen + 1); long oi = 0;
        double tg = now_s();
        for (int g = 0; g < gen; g++) {
            int last = ctxlen - 1;                  // the position whose next-token logits we sample
            // write TOKENS for this forward: real context [0..last], pad the rest with the last token id.
            for (int t = 0; t < S1_L_SEQ; t++) hbuf[S1_OFF_TOKENS + t] = (t <= last) ? ctx[t] : ctx[last];
            if (stage(S1_OFF_TOKENS, S1_L_SEQ)) { fprintf(stderr, "[infer] gen: TOKENS stage failed\n"); break; }
            { int32_t fl[1]; readback(S1_OFF_TOKENS, fl, 1); (void)fl; }   // flush posted write before the kick
            if (kick(MODE_FWD, 0, 0, 10.0)) { fprintf(stderr, "[infer] gen: forward kick hung at char %d\n", g); break; }
            readback(S1_OFF_LOGITS + (long)last * S1_V, hbuf + S1_OFF_LOGITS + (long)last * S1_V, S1_V);
            int nx = sample_logits(hbuf + S1_OFF_LOGITS + (long)last * S1_V, vreal, temp, &rng);
            out[oi++] = (have_voc && nx < nvoc) ? id2ch[nx] : '?';
            // append nx; slide the window when full (drop the oldest token, keep the most recent L_SEQ-1).
            if (ctxlen < S1_L_SEQ) { ctx[ctxlen++] = nx; }
            else { for (int t = 1; t < S1_L_SEQ; t++) ctx[t - 1] = ctx[t]; ctx[S1_L_SEQ - 1] = nx; }
            if (!have_voc) printf("%d ", nx);
        }
        out[oi] = '\0';
        double eg = now_s() - tg;
        if (have_voc) { printf("[infer] >>> "); fwrite(out, 1, (size_t)oi, stdout); printf("\n"); }
        else printf("\n");
        printf("[infer] generated %ld chars in %.3f s (%.1f ms/char, forward-bound)\n",
               oi, eg, eg / (oi > 0 ? oi : 1) * 1e3);
        free(out);
    }

    // ========================================================================
    // (2) HELD-OUT BPC — slide L_SEQ windows over a val-id stream; mode-0 forward per window; accumulate the
    //     TRUE held-out bits/char (mean -log2 softmax(logit)[target] over [0,VREAL)). The metric is identical
    //     to s1_deploy_eval.cpp so the silicon bpc is directly comparable to the off-fabric csim re-gate.
    // ========================================================================
    if (valpath) {
        long vn = 0; int32_t *val = load_bin_i32(valpath, &vn);
        if (!val) { fprintf(stderr, "[infer] could not load val ids %s — skipping bpc\n", valpath); }
        else {
            printf("[infer] ============================================================\n");
            printf("[infer] HELD-OUT BPC: %d windows of %d chars (val stream %ld chars, temp=%.1f, vreal=%d)\n",
                   nwin, S1_L_SEQ, vn, temp <= 0 ? 256.0 : temp, vreal);
            double tot_bits = 0.0; long cnt = 0; int used = 0;
            double bpc_temp = (temp <= 0.0) ? 256.0 : temp;   // bpc needs a real temperature (argmax has no prob)
            double tb = now_s();
            for (int w = 0; w < nwin; w++) {
                long s = (long)w * S1_L_SEQ;
                if (s + S1_L_SEQ + 1 > vn) break;
                for (int t = 0; t < S1_L_SEQ; t++) {
                    hbuf[S1_OFF_TOKENS + t]  = val[s + t];
                    hbuf[S1_OFF_TARGETS + t] = val[s + t + 1];
                }
                if (stage(S1_OFF_TOKENS, S1_L_SEQ))  { fprintf(stderr, "[infer] bpc: TOKENS stage failed\n"); break; }
                if (stage(S1_OFF_TARGETS, S1_L_SEQ)) { fprintf(stderr, "[infer] bpc: TARGETS stage failed\n"); break; }
                { int32_t fl[1]; readback(S1_OFF_TOKENS, fl, 1); (void)fl; }   // flush before the kick
                if (kick(MODE_FWD, 0, 0, 10.0)) { fprintf(stderr, "[infer] bpc: forward kick hung at window %d\n", w); break; }
                readback(S1_OFF_LOGITS, hbuf + S1_OFF_LOGITS, (long)S1_L_SEQ * S1_V);
                for (int t = 0; t < S1_L_SEQ; t++) {
                    const int32_t *lg = hbuf + S1_OFF_LOGITS + (long)t * S1_V;
                    int tgt = val[s + t + 1];
                    if (tgt < 0) tgt = 0; if (tgt >= vreal) tgt = vreal - 1;
                    double mx = -1e300;
                    for (int v = 0; v < vreal; v++) { double z = (double)(int32_t)lg[v] / bpc_temp; if (z > mx) mx = z; }
                    double sum = 0.0;
                    for (int v = 0; v < vreal; v++) sum += exp((double)(int32_t)lg[v] / bpc_temp - mx);
                    double pt = exp((double)(int32_t)lg[tgt] / bpc_temp - mx) / sum;
                    if (pt < 1e-30) pt = 1e-30;
                    tot_bits += -log2(pt);
                    cnt++;
                }
                used = w + 1;
            }
            double bpc = (cnt > 0) ? tot_bits / (double)cnt : 0.0;
            double eb = now_s() - tb;
            printf("[infer] BPC windows=%d positions=%ld in %.3f s (%.1f ms/window)\n",
                   used, cnt, eb, eb / (used > 0 ? used : 1) * 1e3);
            printf("[infer] HELD-OUT bpc = %.4f   | bigram = 3.47 | torch HARD ref = 2.40-2.65\n", bpc);
            printf("[infer] beats_bigram = %s\n", (bpc < 3.47) ? "YES" : "NO");
            printf("[S1 INFER RESULT] bpc=%.4f windows=%d temp=%.1f gen=%d seed=%u\n", bpc, used, bpc_temp, gen, seed);
            free(val);
        }
    }

    printf("[infer] ============================================================\n");
    printf("[infer] done (forward-only inference; weights untouched on DDR)\n");
    free(buf0); free(hbuf);
    fpga_pci_detach(bar0); fpga_pci_detach(bar4); fpga_pci_detach(bar4r);
    return 0;
}
