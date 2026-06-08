// forge_train_step_tb.cpp — host-side csim for the monolithic training kernel.
// Verifies the period-4 LM toy task converges via the single-kernel dispatcher.
//
// Compile:
//   g++ -std=c++14 -O2 -DCSIM_ONLY -I.. forge_train_step.cpp \
//       forge_train_step_tb.cpp -o tb_train

#include "forge_buffer.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>

extern "C" int forge_train_step(int32_t* buffer, int mode, int arg1, int arg2, int arg3);

int main() {
    std::printf("[tb_train] FT_BUFFER_WORDS=%d (%.1f MiB)\n",
                FT_BUFFER_WORDS, FT_BUFFER_BYTES / (1024.0 * 1024.0));

    std::vector<int32_t> buf(FT_BUFFER_WORDS, 0);

    auto rng_seed = 0xC0FFEEu;
    auto rnd = [&]() { rng_seed = rng_seed * 1103515245u + 12345u; return (int16_t)(rng_seed >> 16); };
    auto fill = [&](int off, int n) {
        for (int i = 0; i < n; i++) buf[off + i] = rnd() % 17 - 8;
    };
    fill(OFF_EMBED,  WORDS_EMBED);
    fill(OFF_S_Q,    WORDS_S_Q);
    fill(OFF_S_K,    WORDS_S_K);
    fill(OFF_W1,     WORDS_W1);
    fill(OFF_W2,     WORDS_W2);
    fill(OFF_W_HEAD, WORDS_W_HEAD);

    const int PERIOD = 4;
    const int L_seq  = 16;
    const int LR     = 1;

    std::vector<int> tokens(L_seq), targets(L_seq);
    for (int m = 0; m < L_seq; m++) {
        tokens [m] = m % PERIOD;
        targets[m] = (m + 1) % PERIOD;
    }

    auto count_correct = [&]() -> int {
        int correct = 0;
        for (int m = 0; m < L_seq; m++) {
            bool zero = true;
            for (int v = 0; v < FT_V; v++) {
                if (buf[OFF_GRAD_LOGIT + m * FT_V + v] != 0) { zero = false; break; }
            }
            if (zero) correct++;
        }
        return correct;
    };

    int best_correct = 0;
    const int N_STEPS = 60;

    for (int step = 0; step < N_STEPS; step++) {
        for (int i = 0; i < (int)WORDS_K_CACHE; i++) buf[OFF_K_CACHE + i] = 0;
        for (int i = 0; i < (int)WORDS_V_CACHE; i++) buf[OFF_V_CACHE + i] = 0;

        // mode=0: fwd_loss
        for (int m = 0; m < L_seq; m++) {
            int rc = forge_train_step(buf.data(), 0, m, tokens[m], targets[m]);
            if (rc != 0) { std::fprintf(stderr, "fwd_loss m=%d rc=%d\n", m, rc); return 1; }
        }
        int correct = count_correct();

        for (int i = 0; i < (int)WORDS_K_CACHE; i++) buf[OFF_K_CACHE + i] = 0;
        for (int i = 0; i < (int)WORDS_V_CACHE; i++) buf[OFF_V_CACHE + i] = 0;

        // mode=1: bwd
        for (int mm = 0; mm < L_seq; mm++) {
            int m = L_seq - 1 - mm;
            int rc = forge_train_step(buf.data(), 1, m, tokens[m], 0);
            if (rc != 0) { std::fprintf(stderr, "bwd m=%d rc=%d\n", m, rc); return 1; }
        }

        // mode=2: signsgd
        int rc = forge_train_step(buf.data(), 2, LR, 0, 0);
        if (rc != 0) { std::fprintf(stderr, "signsgd rc=%d\n", rc); return 1; }

        if (correct > best_correct) best_correct = correct;
        if (step < 5 || step % 10 == 0 || step == N_STEPS - 1) {
            std::printf("[tb_train] step %3d  correct=%d/%d  best=%d/%d\n",
                        step, correct, L_seq, best_correct, L_seq);
        }
    }

    std::printf("[tb_train] best %d/%d positions correct over %d steps\n",
                best_correct, L_seq, N_STEPS);
    bool per_step_pass = (best_correct >= L_seq / 2);
    if (per_step_pass) {
        std::printf("[forge_train_step] per-step PASS — period-4 LM converges via mode 0/1/2 dispatcher\n");
    } else {
        std::fprintf(stderr, "[forge_train_step] per-step FAIL — convergence under 50%%\n");
        return 1;
    }

    // ------------------------------------------------------------------
    // Mode 3 (do_train_window) — same task, one kernel call per step.
    // ------------------------------------------------------------------
    std::printf("\n[tb_train] now testing mode 3 (in-kernel multi-step window)\n");

    std::vector<int32_t> buf3(FT_BUFFER_WORDS, 0);
    rng_seed = 0xC0FFEEu;  // re-seed for identical init
    auto fill3 = [&](int off, int n) {
        for (int i = 0; i < n; i++) buf3[off + i] = rnd() % 17 - 8;
    };
    fill3(OFF_EMBED,  WORDS_EMBED);
    fill3(OFF_S_Q,    WORDS_S_Q);
    fill3(OFF_S_K,    WORDS_S_K);
    fill3(OFF_W1,     WORDS_W1);
    fill3(OFF_W2,     WORDS_W2);
    fill3(OFF_W_HEAD, WORDS_W_HEAD);
    // Pre-stage tokens/targets — mode 3 reads them from the buffer.
    for (int m = 0; m < L_seq; m++) {
        buf3[OFF_TOKENS  + m] = tokens [m];
        buf3[OFF_TARGETS + m] = targets[m];
    }
    // Out-of-range sentinel for unused positions so mode 3 skips them.
    for (int m = L_seq; m < FT_L_MAX; m++) {
        buf3[OFF_TOKENS  + m] = -1;
        buf3[OFF_TARGETS + m] = -1;
    }

    int best_correct_w = 0;
    for (int step = 0; step < N_STEPS; step++) {
        for (int i = 0; i < (int)WORDS_K_CACHE; i++) buf3[OFF_K_CACHE + i] = 0;
        for (int i = 0; i < (int)WORDS_V_CACHE; i++) buf3[OFF_V_CACHE + i] = 0;

        int correct = forge_train_step(buf3.data(), 3, 0, 0, L_seq);
        if (correct < 0) { std::fprintf(stderr, "mode3 rc=%d\n", correct); return 1; }
        // Host applies the weight update separately now (matches silicon ABI).
        int rc2 = forge_train_step(buf3.data(), 2, LR, 0, 0);
        if (rc2 != 0) { std::fprintf(stderr, "signsgd after mode3 rc=%d\n", rc2); return 1; }
        if (correct > best_correct_w) best_correct_w = correct;
        if (step < 5 || step % 10 == 0 || step == N_STEPS - 1) {
            std::printf("[tb_train mode3] step %3d  correct=%d/%d  best=%d/%d\n",
                        step, correct, L_seq, best_correct_w, L_seq);
        }
    }

    std::printf("[tb_train mode3] best %d/%d positions correct over %d steps\n",
                best_correct_w, L_seq, N_STEPS);
    if (best_correct_w >= L_seq / 2) {
        std::printf("[forge_train_step] mode3 PASS — period-4 LM converges via single-call window dispatcher\n");
        return 0;
    }
    std::fprintf(stderr, "[forge_train_step] mode3 FAIL — convergence under 50%%\n");
    return 1;
}
