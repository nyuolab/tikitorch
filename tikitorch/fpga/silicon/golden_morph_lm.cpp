// golden_morph_lm.cpp — HOST CSIM golden harness for the config-B MorphSSM LM
// training kernel (forge_train_step_morph_lm). Compiled with a plain clang++/
// g++ (NO __VITIS_HLS__/__SDSVHLS__/__SYNTHESIS__), so the kernel takes its
// portable int32 path (forge_hls_types.hpp -> int16_t/int32_t; burst helpers
// take the plain-loop #else branch). This is a faithful bit-exact reference
// for the FPGA silicon driver (morph_lm_silicon.py): both load the SAME
// buffer_init.bin and run the SAME schedule, so identical JSONL <=> correct HW.
//
// Schedule (one optimizer step over an L_SEQ=128 window), verified against the
// kernel source and forge_buffer_morph_lm.hpp's complete-by-read invariant:
//   1. zero OFF_BS_STREAM (rolling causal-conv cache) before the fwd sweep
//   2. FORWARD m = 0..L-1 : run(mode=0, m, token[m], target[m])
//   3. zero all 5 OFF_G_* regions AND re-zero OFF_BS_STREAM (now grad_BS accum)
//   4. BACKWARD m = L-1..0 (REVERSE) : run(mode=1, m, token[m], 0)
//   5. SignSGD : run(mode=2, lr, threshold, 0)  -> updates weights, zeroes grads
//
// Per-step JSONL: {"step":k,"acc":...,"wcrc":...,"preds16":[...16...]}
//
// Build (macOS clang or linux g++):
//   REPO=/Users/ekoermann/Documents/tikitorch
//   HLS="$REPO/tikitorch/hardware/hls/forge_kernels"
//   clang++ -O2 -std=c++17 -Wno-unknown-pragmas -Wno-unused-label \
//       -I"$HLS" -I"$REPO/tikitorch/hardware/hls" \
//       golden_morph_lm.cpp "$HLS/forge_train_step_morph_lm.cpp" \
//       -o /tmp/golden
//   /tmp/golden /tmp/buffer_init.bin > /tmp/golden_traj.jsonl

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "forge_buffer_morph_lm.hpp"   // ML_* dims, OFF_*, WORDS_*

// The kernel entry. golden links forge_train_step_morph_lm.cpp directly; we
// declare the C ABI here rather than pulling the .cpp in via #include so there
// is exactly one definition (the separately-compiled translation unit).
extern "C" int forge_train_step_morph_lm(int32_t* buffer, int mode,
                                         int a1, int a2, int a3);

// ---- fixed run parameters (must match morph_lm_silicon.py) ----
// SGD_THR via env TT_SGD_THR (default 2048): CE grads are Q16 (thr=2048); the
// perceptron loss (CFG loss_kind=0) emits +/-1 grads, so it needs thr=0 to step.
static const int   STEPS     = 30;     // optimizer steps over the same window
static const int   SGD_LR    = 1;      // signsgd learning rate (arg1)

// ---- deterministic weight-region checksum (FNV-1a over int32 words) ----
// Must be byte-identical to the Python implementation in morph_lm_silicon.py.
// We checksum, in order, all five weight regions (EMBED, MH, MP, MB, W_HEAD)
// — the quantities SignSGD updates. 64-bit FNV-1a; emitted as unsigned.
static uint64_t fnv1a_words(const int32_t* p, long n, uint64_t h) {
    for (long i = 0; i < n; i++) {
        uint32_t w = (uint32_t)p[i];
        // feed 4 bytes little-endian
        for (int b = 0; b < 4; b++) {
            uint8_t byte = (uint8_t)((w >> (b * 8)) & 0xFF);
            h ^= (uint64_t)byte;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

static uint64_t weight_crc(const int32_t* buf) {
    uint64_t h = 1469598103934665603ULL;   // FNV offset basis
    h = fnv1a_words(buf + OFF_EMBED,  WORDS_EMBED,  h);
    h = fnv1a_words(buf + OFF_MH,     WORDS_MH,     h);
    h = fnv1a_words(buf + OFF_MP,     WORDS_MP,     h);
    h = fnv1a_words(buf + OFF_MB,     WORDS_MB,     h);
    h = fnv1a_words(buf + OFF_W_HEAD, WORDS_W_HEAD, h);
    return h;
}

static void zero_region(int32_t* buf, int off, int n) {
    std::memset(buf + off, 0, (size_t)n * sizeof(int32_t));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <buffer_init.bin>\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const int SGD_THR = std::getenv("TT_SGD_THR") ? std::atoi(std::getenv("TT_SGD_THR")) : 2048;

    // ---- load the SHARED ground-truth buffer ----
    std::vector<int32_t> buf(ML_BUFFER_WORDS);
    {
        FILE* f = std::fopen(path, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 3; }
        size_t got = std::fread(buf.data(), sizeof(int32_t), ML_BUFFER_WORDS, f);
        std::fclose(f);
        if (got != (size_t)ML_BUFFER_WORDS) {
            std::fprintf(stderr, "short read: got %zu words, want %d\n",
                         got, ML_BUFFER_WORDS);
            return 4;
        }
    }
    int32_t* B = buf.data();

#ifdef TT_BATCH
    // =====================================================================
    // BATCH path (binary built with -DTT_BATCH): one mode-3 call runs the
    // data-parallel batched trainer over ML_BATCH_N sequences for N steps,
    // SUMMING per-sequence grads into shared accumulators and applying ONE
    // SignSGD per step. We emit, per step, the per-sequence accuracy and the
    // post-update weight-CRC. The companion reference binary (built WITHOUT
    // TT_BATCH, run with TT_BATCH_REF=ML_BATCH_N) reproduces the per-step
    // CRC by summing the SAME B per-sequence grads through the SAME SignSGD
    // — equal CRCs at every step == batched is bit-exact to summed-per-seq.
    //
    // We loop N=1..STEPS, re-running the whole batch trainer from the SAME
    // init each iteration, so each line reports the CRC AFTER exactly N
    // SignSGD updates (mode 3 internally zeros its on-chip grads at start,
    // so re-running from the fresh buffer each N is well-defined). Per-seq
    // tokens/targets live at off_tokens_b(b)/off_targets_b(b).
    {
        // capture per-seq targets up front (read from the staged buffer).
        std::vector<std::vector<int>> tg(ML_BATCH_N, std::vector<int>(ML_L_SEQ));
        for (int b = 0; b < ML_BATCH_N; b++)
            for (int m = 0; m < ML_L_SEQ; m++)
                tg[b][m] = B[off_targets_b(b) + m];
        std::vector<int32_t> init(buf);   // pristine init for re-runs
        for (int n = 1; n <= STEPS; n++) {
            std::memcpy(B, init.data(), (size_t)ML_BUFFER_WORDS * sizeof(int32_t));
            int rc = forge_train_step_morph_lm(B, ML_MODE_TRAIN_RUN, n, SGD_LR, SGD_THR);
            if (rc != 0) { std::fprintf(stderr, "batch train_run rc=%d\n", rc); return 13; }
            uint64_t wcrc = weight_crc(B);
            std::printf("{\"step\":%d,\"wcrc\":%llu,\"batch\":%d,\"accs\":[",
                        n - 1, (unsigned long long)wcrc, ML_BATCH_N);
            for (int b = 0; b < ML_BATCH_N; b++) {
                int correct = 0;
                for (int m = 0; m < ML_L_SEQ; m++)
                    if (B[off_pred_b(b) + m] == tg[b][m]) correct++;
                std::printf("%.6f%s", (double)correct / ML_L_SEQ, (b == ML_BATCH_N - 1 ? "" : ","));
            }
            std::printf("],\"preds16_b0\":[");
            for (int m = 0; m < 16; m++)
                std::printf("%d%s", (int)B[off_pred_b(0) + m], (m == 15 ? "" : ","));
            std::printf("]}\n");
            std::fflush(stdout);
        }
        return 0;
    }
#endif

    // =====================================================================
    // PER-SEQUENCE-SUMMED REFERENCE (binary built WITHOUT TT_BATCH; activated
    // by env TT_BATCH_REF=B). Reads B distinct sequences from a BATCH-staged
    // buffer file (the SAME file the batched binary consumes) using the
    // per-seq strides, but drives the PLAIN buffer-backed kernel (modes
    // 0/1/2) over its OWN non-batch buffer. Per step: zero the shared grad
    // regions ONCE, then for each b {stage b's tokens into the b=0 slot,
    // zero BS, fwd 0..L-1, zero BS, bwd L-1..0} accumulating (NOT zeroing)
    // grads across b, then ONE SignSGD. This is exactly "sum the B
    // per-sequence grads, apply once" — the independent reference the batched
    // kernel must equal bit-for-bit at every step's weight-CRC.
    if (const char* refenv = std::getenv("TT_BATCH_REF")) {
        const int RB = std::atoi(refenv);
        if (RB < 1) { std::fprintf(stderr, "TT_BATCH_REF must be >= 1\n"); return 20; }
        // Per-seq region strides in the BATCH-staged source file (these match the
        // header's STRIDE_* and gen_buffer_init.py's; independent of this binary's
        // non-batch ML_BATCH_N==1 layout).
        const int SRC_STRIDE_TOKENS  = ML_L_SEQ;
        const int SRC_STRIDE_TARGETS = ML_L_SEQ;
        // The source file's OFF_TOKENS/OFF_TARGETS use the BATCH layout. Recompute
        // them here from the batch geometry (per-seq regions grown by RB). Weight
        // + grad + CFG + CE_LUT regions are layout-INVARIANT in their VALUES, but
        // their OFFSETS shift; we only need the source TOKENS/TARGETS + CFG/LUT.
        auto batch_off = [&](int nonbatch_pos_after_weights)->long { return 0; };
        (void)batch_off;
        const long B_TAP   = (long)RB * (ML_N_LAYERS * ML_L_SEQ * ML_D);
        const long B_PROJ  = (long)RB * (ML_N_LAYERS * ML_L_SEQ * ML_D);
        const long B_ADH   = (long)RB * (ML_L_SEQ * ML_V);
        const long B_BS    = (long)RB * (ML_N_LAYERS * ML_L_SEQ * ML_D);
        const long B_TOK   = (long)RB * ML_L_SEQ;
        const long B_TGT   = (long)RB * ML_L_SEQ;
        const long B_GLOG  = (long)RB * (ML_L_SEQ * ML_V);
        const long B_PRED  = (long)RB * ML_L_SEQ;
        // Cumulative batch offsets (mirror the header / gen_buffer_init for B=RB).
        long o = 0;
        const long bo_embed   = o; o += WORDS_EMBED;
        const long bo_mh      = o; o += WORDS_MH;
        const long bo_mp      = o; o += WORDS_MP;
        const long bo_mb      = o; o += WORDS_MB;
        const long bo_whead   = o; o += WORDS_W_HEAD;
        const long bo_gembed  = o; o += WORDS_G_EMBED;
        const long bo_gmh     = o; o += WORDS_G_MH;
        const long bo_gmp     = o; o += WORDS_G_MP;
        const long bo_gmb     = o; o += WORDS_G_MB;
        const long bo_gwhead  = o; o += WORDS_G_W_HEAD;
        const long bo_tap     = o; o += B_TAP;
        const long bo_proj    = o; o += B_PROJ;
        const long bo_adh     = o; o += B_ADH;
        const long bo_bs      = o; o += B_BS;
        const long bo_tokens  = o; o += B_TOK;
        const long bo_targets = o; o += B_TGT;
        const long bo_glog    = o; o += B_GLOG;
        const long bo_pred    = o; o += B_PRED;
        const long bo_cfg     = o; o += WORDS_CFG;
        const long bo_ce_lut  = ((bo_cfg + WORDS_CFG + 15) / 16) * 16;
        const long batch_words = bo_ce_lut + WORDS_CE_LUT;
        (void)bo_embed;(void)bo_mh;(void)bo_mp;(void)bo_mb;(void)bo_whead;
        (void)bo_gembed;(void)bo_gmh;(void)bo_gmp;(void)bo_gmb;(void)bo_gwhead;
        (void)bo_tap;(void)bo_proj;(void)bo_adh;(void)bo_bs;(void)bo_glog;(void)bo_pred;
        (void)SRC_STRIDE_TOKENS;(void)SRC_STRIDE_TARGETS;

        // Re-load the source file at its TRUE (batch) size (the initial fread used
        // this binary's small ML_BUFFER_WORDS — re-read the whole batch buffer).
        std::vector<int32_t> src(batch_words);
        {
            FILE* f = std::fopen(path, "rb");
            if (!f) { std::fprintf(stderr, "ref: cannot open %s\n", path); return 21; }
            size_t got = std::fread(src.data(), sizeof(int32_t), (size_t)batch_words, f);
            std::fclose(f);
            if (got != (size_t)batch_words) {
                std::fprintf(stderr, "ref: short read %zu/%ld words (B=%d)\n", got, batch_words, RB);
                return 22;
            }
        }

        // Build the NON-BATCH reference buffer: copy shared weights + CFG + LUT
        // from the batch source (their VALUES are layout-invariant); grads/scratch
        // start zero. Weights occupy identical offsets here (OFF_EMBED..OFF_W_HEAD).
        std::vector<int32_t> rbuf(ML_BUFFER_WORDS, 0);
        int32_t* R = rbuf.data();
        std::memcpy(R + OFF_EMBED,  src.data() + bo_embed,  WORDS_EMBED  * sizeof(int32_t));
        std::memcpy(R + OFF_MH,     src.data() + bo_mh,     WORDS_MH     * sizeof(int32_t));
        std::memcpy(R + OFF_MP,     src.data() + bo_mp,     WORDS_MP     * sizeof(int32_t));
        std::memcpy(R + OFF_MB,     src.data() + bo_mb,     WORDS_MB     * sizeof(int32_t));
        std::memcpy(R + OFF_W_HEAD, src.data() + bo_whead,  WORDS_W_HEAD * sizeof(int32_t));
        R[OFF_CFG + 0] = src[bo_cfg + 0];
        R[OFF_CFG + 1] = src[bo_cfg + 1];
        std::memcpy(R + OFF_CE_LUT, src.data() + bo_ce_lut, WORDS_CE_LUT * sizeof(int32_t));

        // Per-seq tokens/targets from the batch source.
        std::vector<std::vector<int>> rtok(RB, std::vector<int>(ML_L_SEQ));
        std::vector<std::vector<int>> rtgt(RB, std::vector<int>(ML_L_SEQ));
        for (int b = 0; b < RB; b++)
            for (int m = 0; m < ML_L_SEQ; m++) {
                rtok[b][m] = src[bo_tokens  + (long)b * ML_L_SEQ + m];
                rtgt[b][m] = src[bo_targets + (long)b * ML_L_SEQ + m];
            }

        for (int step = 0; step < STEPS; step++) {
            // (0) zero the 5 SHARED grad regions ONCE per step (grads SUM over b).
            zero_region(R, OFF_G_EMBED,  WORDS_G_EMBED);
            zero_region(R, OFF_G_MH,     WORDS_G_MH);
            zero_region(R, OFF_G_MP,     WORDS_G_MP);
            zero_region(R, OFF_G_MB,     WORDS_G_MB);
            zero_region(R, OFF_G_W_HEAD, WORDS_G_W_HEAD);

            std::vector<int> acc_correct(RB, 0);
            int32_t pred_b0[16] = {0};
            for (int b = 0; b < RB; b++) {
                // (1) zero BS (rolling conv cache) before this sequence's fwd.
                zero_region(R, OFF_BS_STREAM, WORDS_BS_STREAM);
                // (2) FORWARD m=0..L-1 for sequence b (b=0 slot in the ref buffer).
                for (int m = 0; m < ML_L_SEQ; m++) {
                    int rc = forge_train_step_morph_lm(R, ML_MODE_FWD, m, rtok[b][m], rtgt[b][m]);
                    if (rc != 0) { std::fprintf(stderr, "ref fwd rc=%d b=%d m=%d\n", rc, b, m); return 23; }
                }
                for (int m = 0; m < ML_L_SEQ; m++)
                    if (R[OFF_PRED + m] == rtgt[b][m]) acc_correct[b]++;
                if (b == 0) for (int m = 0; m < 16; m++) pred_b0[m] = R[OFF_PRED + m];
                // (3) zero BS (now grad_BS accum); grads NOT zeroed -> they SUM.
                zero_region(R, OFF_BS_STREAM, WORDS_BS_STREAM);
                // (4) BACKWARD m=L-1..0 (REVERSE) for sequence b.
                for (int m = ML_L_SEQ - 1; m >= 0; m--) {
                    int rc = forge_train_step_morph_lm(R, ML_MODE_BWD, m, rtok[b][m], 0);
                    if (rc != 0) { std::fprintf(stderr, "ref bwd rc=%d b=%d m=%d\n", rc, b, m); return 24; }
                }
            }
            // (5) ONE SignSGD over the SUMMED grads (same lr/thr as the batch run).
            int rc = forge_train_step_morph_lm(R, ML_MODE_SIGNSGD, SGD_LR, SGD_THR, 0);
            if (rc != 0) { std::fprintf(stderr, "ref sgd rc=%d step=%d\n", rc, step); return 25; }

            uint64_t wcrc = weight_crc(R);
            std::printf("{\"step\":%d,\"wcrc\":%llu,\"batch\":%d,\"accs\":[",
                        step, (unsigned long long)wcrc, RB);
            for (int b = 0; b < RB; b++)
                std::printf("%.6f%s", (double)acc_correct[b] / ML_L_SEQ, (b == RB - 1 ? "" : ","));
            std::printf("],\"preds16_b0\":[");
            for (int m = 0; m < 16; m++) std::printf("%d%s", (int)pred_b0[m], (m == 15 ? "" : ","));
            std::printf("]}\n");
            std::fflush(stdout);
        }
        return 0;
    }

    // tokens / targets are carried inside the buffer (OFF_TOKENS/OFF_TARGETS);
    // the kernel takes them via args, so we read them out of the buffer.
    std::vector<int> toks(ML_L_SEQ), tgts(ML_L_SEQ);
    for (int m = 0; m < ML_L_SEQ; m++) {
        toks[m] = B[OFF_TOKENS  + m];
        tgts[m] = B[OFF_TARGETS + m];
    }

    // Fused whole-run path (TT_FUSED_STEPS set): one mode-3 call runs the whole
    // training internally — this is the host reference for the on-fabric kernel.
    // Its final {wcrc,preds16} MUST equal the per-position loop's last step
    // (same arithmetic), which is the bit-exactness check for do_train_run.
    if (const char* fs = std::getenv("TT_FUSED_STEPS")) {
        int n = std::atoi(fs);
        int rc = forge_train_step_morph_lm(B, ML_MODE_TRAIN_RUN, n, SGD_LR, SGD_THR);
        if (rc != 0) { std::fprintf(stderr, "train_run rc=%d\n", rc); return 13; }
        int correct = 0;
        for (int m = 0; m < ML_L_SEQ; m++) if (B[OFF_PRED + m] == tgts[m]) correct++;
        uint64_t wcrc = weight_crc(B);
        std::printf("{\"fused_steps\":%d,\"acc\":%.6f,\"wcrc\":%llu,\"preds16\":[",
                    n, (double)correct / ML_L_SEQ, (unsigned long long)wcrc);
        for (int m = 0; m < 16; m++) std::printf("%d%s", (int)B[OFF_PRED + m], (m == 15 ? "" : ","));
        std::printf("]}\n");
        return 0;
    }

    for (int step = 0; step < STEPS; step++) {
        // (1) zero BS stream before the forward sweep (rolling conv cache).
        zero_region(B, OFF_BS_STREAM, WORDS_BS_STREAM);

        // (2) FORWARD sweep, forward position order m = 0..L-1.
        for (int m = 0; m < ML_L_SEQ; m++) {
            int rc = forge_train_step_morph_lm(B, ML_MODE_FWD, m, toks[m], tgts[m]);
            if (rc != 0) {
                std::fprintf(stderr, "fwd rc=%d at step=%d m=%d\n", rc, step, m);
                return 10;
            }
        }

        // accuracy + first-16 preds for THIS step (post-forward).
        int correct = 0;
        for (int m = 0; m < ML_L_SEQ; m++) {
            if (B[OFF_PRED + m] == tgts[m]) correct++;
        }
        double acc = (double)correct / (double)ML_L_SEQ;

        // (3) zero all 5 grad regions AND re-zero BS stream (now grad_BS accum).
        zero_region(B, OFF_G_EMBED,  WORDS_G_EMBED);
        zero_region(B, OFF_G_MH,     WORDS_G_MH);
        zero_region(B, OFF_G_MP,     WORDS_G_MP);
        zero_region(B, OFF_G_MB,     WORDS_G_MB);
        zero_region(B, OFF_G_W_HEAD, WORDS_G_W_HEAD);
        zero_region(B, OFF_BS_STREAM, WORDS_BS_STREAM);

        // (4) BACKWARD sweep, REVERSE position order m = L-1..0.
        for (int m = ML_L_SEQ - 1; m >= 0; m--) {
            int rc = forge_train_step_morph_lm(B, ML_MODE_BWD, m, toks[m], 0);
            if (rc != 0) {
                std::fprintf(stderr, "bwd rc=%d at step=%d m=%d\n", rc, step, m);
                return 11;
            }
        }

        // (5) SignSGD: consume grads, update weights, zero grads in-kernel.
        {
            int rc = forge_train_step_morph_lm(B, ML_MODE_SIGNSGD, SGD_LR, SGD_THR, 0);
            if (rc != 0) {
                std::fprintf(stderr, "sgd rc=%d at step=%d\n", rc, step);
                return 12;
            }
        }

        // checksum AFTER the weight update (so it changes as the model learns).
        uint64_t wcrc = weight_crc(B);

        // ---- emit one JSONL line ----
        std::printf("{\"step\":%d,\"acc\":%.6f,\"wcrc\":%llu,\"preds16\":[",
                    step, acc, (unsigned long long)wcrc);
        for (int m = 0; m < 16; m++) {
            std::printf("%d%s", (int)B[OFF_PRED + m], (m == 15 ? "" : ","));
        }
        std::printf("]}\n");
        std::fflush(stdout);
    }
    return 0;
}
