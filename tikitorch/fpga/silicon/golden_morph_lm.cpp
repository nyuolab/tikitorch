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
