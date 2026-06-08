// forge_train_step_3layer_tb.cpp — csim for the 3-layer training kernel.
//
// Per training step, host orchestrates:
//   1. zero K/V caches for all 3 layers
//   2. layer 0 forward over all positions (writes HIDDEN_0)
//   3. layer 1 forward over all positions (reads HIDDEN_0, writes HIDDEN_1)
//   4. layer 2 forward over all positions (reads HIDDEN_1, writes int16 logits
//      at OFF3_GRAD_LOGIT)
//   5. host computes perceptron grad from logits, writes back to OFF3_GRAD_LOGIT
//   6. zero K/V caches for all 3 layers (they become gradient accumulators)
//   7. layer 2 backward (reverse positions) writes grad to HIDDEN_1
//   8. layer 1 backward (reverse) writes grad to HIDDEN_0
//   9. layer 0 backward (reverse) accumulates into G_EMBED
//   10. signsgd over all 20 (param, grad) pairs (3 × 6 per-layer + 2 shared)
//
// Pass criterion: best-correct >= L_seq / 2 on a 60-step period-4 LM run.

#include "forge_buffer_3layer.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>

extern "C" int forge_train_step_3layer(int32_t* buffer, int mode,
                                        int arg1, int arg2, int arg3, int layer_id);

int main() {
    std::printf("[tb_train_3layer] FT3_BUFFER_WORDS=%d (%.1f MiB)\n",
                FT3_BUFFER_WORDS, FT3_BUFFER_BYTES / (1024.0 * 1024.0));

    std::vector<int32_t> buf(FT3_BUFFER_WORDS, 0);

    auto rng_seed = 0xC0FFEEu;
    auto rnd = [&]() { rng_seed = rng_seed * 1103515245u + 12345u; return (int16_t)(rng_seed >> 16); };
    auto fill = [&](int off, int n) {
        for (int i = 0; i < n; i++) buf[off + i] = rnd() % 17 - 8;
    };
    fill(OFF3_EMBED,    W3_EMBED);
    fill(OFF3_W_HEAD,   W3_W_HEAD);
    fill(OFF3_L0_S_Q,   W3_S_Q);  fill(OFF3_L0_S_K,   W3_S_K);
    fill(OFF3_L0_W1,    W3_W1);   fill(OFF3_L0_W2,    W3_W2);
    fill(OFF3_L1_S_Q,   W3_S_Q);  fill(OFF3_L1_S_K,   W3_S_K);
    fill(OFF3_L1_W1,    W3_W1);   fill(OFF3_L1_W2,    W3_W2);
    fill(OFF3_L2_S_Q,   W3_S_Q);  fill(OFF3_L2_S_K,   W3_S_K);
    fill(OFF3_L2_W1,    W3_W1);   fill(OFF3_L2_W2,    W3_W2);

    const int PERIOD = 4;
    const int L_seq  = 16;
    const int LR     = 1;

    std::vector<int> tokens(L_seq), targets(L_seq);
    for (int m = 0; m < L_seq; m++) {
        tokens [m] = m % PERIOD;
        targets[m] = (m + 1) % PERIOD;
    }

    // After all-layer fwd, OFF3_GRAD_LOGIT holds int16 logits.  Host computes
    // perceptron gradient (silicon-equivalent) here.
    auto host_loss = [&]() -> int {
        int correct = 0;
        for (int m = 0; m < L_seq; m++) {
            int32_t* row = &buf[OFF3_GRAD_LOGIT + m * FT3_V];
            int pred = 0; int32_t mx = row[0];
            for (int v = 1; v < FT3_V; v++) {
                if (row[v] > mx) { mx = row[v]; pred = v; }
            }
            int tgt = targets[m];
            for (int v = 0; v < FT3_V; v++) row[v] = 0;
            if (pred != tgt) { row[pred] = +1; row[tgt] = -1; }
            else             { correct++; }
        }
        return correct;
    };

    auto clear_caches = [&]() {
        for (int i = 0; i < W3_K_CACHE; i++) {
            buf[OFF3_L0_K_CACHE + i] = 0;
            buf[OFF3_L1_K_CACHE + i] = 0;
            buf[OFF3_L2_K_CACHE + i] = 0;
        }
        for (int i = 0; i < W3_V_CACHE; i++) {
            buf[OFF3_L0_V_CACHE + i] = 0;
            buf[OFF3_L1_V_CACHE + i] = 0;
            buf[OFF3_L2_V_CACHE + i] = 0;
        }
    };

    int best_correct = 0;
    const int N_STEPS = 60;
    for (int step = 0; step < N_STEPS; step++) {
        clear_caches();
        // Forward: L0 over all positions, then L1, then L2.
        for (int layer = 0; layer < FT3_N_LAYERS; layer++) {
            int target_arg = (layer == FT3_N_LAYERS - 1) ? 0 : -1;
            for (int m = 0; m < L_seq; m++) {
                // For layer 0 we pass token; for others token is ignored — but
                // the dispatch still needs SOMETHING; we pass the L=0 token id.
                int token_arg = (layer == 0) ? tokens[m] : -1;
                int target_pass = (layer == FT3_N_LAYERS - 1) ? targets[m] : -1;
                (void)target_arg;
                int rc = forge_train_step_3layer(buf.data(), MODE3_FWD_LOSS,
                                                  m, token_arg, target_pass, layer);
                if (rc != 0) { std::fprintf(stderr, "L%d fwd m=%d rc=%d\n", layer, m, rc); return 1; }
            }
        }
        int correct = host_loss();

        // Backward: zero K/V caches (used as grad accumulators), reverse-order over layers.
        clear_caches();
        for (int layer = FT3_N_LAYERS - 1; layer >= 0; layer--) {
            for (int mm = 0; mm < L_seq; mm++) {
                int m = L_seq - 1 - mm;
                int token_arg = (layer == 0) ? tokens[m] : -1;
                int rc = forge_train_step_3layer(buf.data(), MODE3_BWD,
                                                  m, token_arg, 0, layer);
                if (rc != 0) { std::fprintf(stderr, "L%d bwd m=%d rc=%d\n", layer, m, rc); return 1; }
            }
        }
        // signsgd
        int rc = forge_train_step_3layer(buf.data(), MODE3_SIGNSGD, LR, 0, 0, -1);
        if (rc != 0) { std::fprintf(stderr, "signsgd rc=%d\n", rc); return 1; }

        if (correct > best_correct) best_correct = correct;
        if (step < 5 || step % 10 == 0 || step == N_STEPS - 1) {
            std::printf("[tb_train_3layer] step %3d  correct=%d/%d  best=%d/%d\n",
                        step, correct, L_seq, best_correct, L_seq);
        }
    }

    std::printf("[tb_train_3layer] best %d/%d positions correct over %d steps\n",
                best_correct, L_seq, N_STEPS);
    if (best_correct >= L_seq / 2) {
        std::printf("[forge_train_step_3layer] csim PASS — period-4 LM converges via 3-layer dispatcher\n");
        return 0;
    }
    std::fprintf(stderr, "[forge_train_step_3layer] csim FAIL — convergence under 50%%\n");
    return 1;
}
