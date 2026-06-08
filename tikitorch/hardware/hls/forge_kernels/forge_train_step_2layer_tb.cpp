// forge_train_step_2layer_tb.cpp — csim for the 2-layer training kernel.
//
// Verifies the period-4 LM toy task converges via the 2-layer dispatcher,
// orchestrated from host: per training step,
//   1. zero K/V caches for both layers
//   2. layer 0 forward over all positions (writes OFF2_HIDDEN)
//   3. layer 1 forward over all positions (reads OFF2_HIDDEN, writes loss grad)
//   4. zero K/V caches for both layers (they become gradient accumulators)
//   5. layer 1 backward over all positions (writes grad to OFF2_HIDDEN)
//   6. layer 0 backward over all positions (consumes grad from OFF2_HIDDEN)
//   7. signsgd over all 14 (param, grad) pairs
//
// Compile:
//   g++ -std=c++14 -O2 -DCSIM_ONLY -I.. forge_train_step_2layer.cpp \
//       forge_train_step_2layer_tb.cpp -o tb_train_2layer
//
// Pass criterion: best-correct >= L_seq/2 on a 60-step period-4 LM run.
// We expect 2-layer to converge AT LEAST as fast as 1-layer (which hits
// 16/16 by step 3).

#include "forge_buffer_2layer.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>

extern "C" int forge_train_step_2layer(int32_t* buffer, int mode,
                                        int arg1, int arg2, int arg3, int layer_id);

int main() {
    std::printf("[tb_train_2layer] FT2_BUFFER_WORDS=%d (%.1f MiB)\n",
                FT2_BUFFER_WORDS, FT2_BUFFER_BYTES / (1024.0 * 1024.0));

    std::vector<int32_t> buf(FT2_BUFFER_WORDS, 0);

    auto rng_seed = 0xC0FFEEu;
    auto rnd = [&]() { rng_seed = rng_seed * 1103515245u + 12345u; return (int16_t)(rng_seed >> 16); };
    auto fill = [&](int off, int n) {
        for (int i = 0; i < n; i++) buf[off + i] = rnd() % 17 - 8;
    };
    // Shared params
    fill(OFF2_EMBED,    W2_EMBED);
    fill(OFF2_W_HEAD,   W2_W_HEAD);
    // Layer 0 params
    fill(OFF2_L0_S_Q,   W2_S_Q);
    fill(OFF2_L0_S_K,   W2_S_K);
    fill(OFF2_L0_W1,    W2_W1);
    fill(OFF2_L0_W2,    W2_W2);
    // Layer 1 params
    fill(OFF2_L1_S_Q,   W2_S_Q);
    fill(OFF2_L1_S_K,   W2_S_K);
    fill(OFF2_L1_W1,    W2_W1);
    fill(OFF2_L1_W2,    W2_W2);

    const int PERIOD = 4;
    const int L_seq  = 16;
    const int LR     = 1;

    std::vector<int> tokens(L_seq), targets(L_seq);
    for (int m = 0; m < L_seq; m++) {
        tokens [m] = m % PERIOD;
        targets[m] = (m + 1) % PERIOD;
    }

    // After the L1 forward loop, the kernel writes int16 logits (sign-extended
    // to int32) at OFF2_LOGITS.  Host reads them and writes the perceptron loss
    // gradient to OFF2_GRAD_LOGIT (the two regions are now distinct; see the CE
    // testbench for the closure-faithful softmax cross-entropy variant).
    auto host_loss = [&]() -> int {
        int correct = 0;
        for (int m = 0; m < L_seq; m++) {
            const int32_t* logit = &buf[OFF2_LOGITS    + m * FT2_V];
            int32_t*       grad  = &buf[OFF2_GRAD_LOGIT + m * FT2_V];
            int pred = 0; int32_t mx = logit[0];
            for (int v = 1; v < FT2_V; v++) {
                if (logit[v] > mx) { mx = logit[v]; pred = v; }
            }
            int tgt = targets[m];
            for (int v = 0; v < FT2_V; v++) grad[v] = 0;
            if (pred != tgt) {
                grad[pred] = +1;
                grad[tgt]  = -1;
            } else {
                correct++;
            }
        }
        return correct;
    };

    auto clear_caches = [&]() {
        for (int i = 0; i < W2_K_CACHE; i++) {
            buf[OFF2_L0_K_CACHE + i] = 0;
            buf[OFF2_L1_K_CACHE + i] = 0;
        }
        for (int i = 0; i < W2_V_CACHE; i++) {
            buf[OFF2_L0_V_CACHE + i] = 0;
            buf[OFF2_L1_V_CACHE + i] = 0;
        }
    };

    int best_correct = 0;
    const int N_STEPS = 60;
    for (int step = 0; step < N_STEPS; step++) {
        clear_caches();
        // Layer 0 fwd over all positions.
        for (int m = 0; m < L_seq; m++) {
            int rc = forge_train_step_2layer(buf.data(), MODE2_FWD_LOSS,
                                              m, tokens[m], -1, /*layer_id=*/0);
            if (rc != 0) { std::fprintf(stderr, "L0 fwd m=%d rc=%d\n", m, rc); return 1; }
        }
        // Layer 1 fwd + loss over all positions.
        for (int m = 0; m < L_seq; m++) {
            int rc = forge_train_step_2layer(buf.data(), MODE2_FWD_LOSS,
                                              m, /*token=ignored*/-1, targets[m],
                                              /*layer_id=*/1);
            if (rc != 0) { std::fprintf(stderr, "L1 fwd m=%d rc=%d\n", m, rc); return 1; }
        }
        int correct = host_loss();

        // Now backward.  Caches become gradient accumulators — zero them.
        clear_caches();
        // Layer 1 bwd in reverse position order.
        for (int mm = 0; mm < L_seq; mm++) {
            int m = L_seq - 1 - mm;
            int rc = forge_train_step_2layer(buf.data(), MODE2_BWD,
                                              m, /*token=ignored*/-1, 0,
                                              /*layer_id=*/1);
            if (rc != 0) { std::fprintf(stderr, "L1 bwd m=%d rc=%d\n", m, rc); return 1; }
        }
        // Layer 0 bwd in reverse position order (consumes grad from OFF2_HIDDEN).
        for (int mm = 0; mm < L_seq; mm++) {
            int m = L_seq - 1 - mm;
            int rc = forge_train_step_2layer(buf.data(), MODE2_BWD,
                                              m, tokens[m], 0, /*layer_id=*/0);
            if (rc != 0) { std::fprintf(stderr, "L0 bwd m=%d rc=%d\n", m, rc); return 1; }
        }
        // signsgd
        int rc = forge_train_step_2layer(buf.data(), MODE2_SIGNSGD,
                                          LR, 0, 0, /*layer_id=ignored*/-1);
        if (rc != 0) { std::fprintf(stderr, "signsgd rc=%d\n", rc); return 1; }

        if (correct > best_correct) best_correct = correct;
        if (step < 5 || step % 10 == 0 || step == N_STEPS - 1) {
            std::printf("[tb_train_2layer] step %3d  correct=%d/%d  best=%d/%d\n",
                        step, correct, L_seq, best_correct, L_seq);
        }
    }

    std::printf("[tb_train_2layer] best %d/%d positions correct over %d steps\n",
                best_correct, L_seq, N_STEPS);
    if (best_correct >= L_seq / 2) {
        std::printf("[forge_train_step_2layer] csim PASS — period-4 LM converges via 2-layer dispatcher\n");
        return 0;
    }
    std::fprintf(stderr, "[forge_train_step_2layer] csim FAIL — convergence under 50%%\n");
    return 1;
}
