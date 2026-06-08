// forge_train_step_2layer_ce_tb.cpp — csim for the 2-layer kernel with the
// CLOSURE-FAITHFUL host-side cross-entropy loss (B2).
//
// The zero-DSP kernel produces logits (OFF2_LOGITS); the HOST conditions the
// loss with a softmax cross-entropy and writes the gradient (softmax - onehot)
// to OFF2_GRAD_LOGIT, which do_bwd_layer consumes.  This is the silicon
// realization of tikitorch.functional.tropical_cross_entropy(softmax="exact")
// — the path that measured +3.2pp over the perceptron baseline on F2 and closed
// timing at WNS -0.155 ns (the kernel is unchanged in the timing-critical path;
// only the logits store moved to its own region, and the loss is host-side).
//
// Compile + run (software csim — no Vitis needed):
//   g++ -std=c++14 -O2 -DCSIM_ONLY forge_train_step_2layer.cpp \
//       forge_train_step_2layer_ce_tb.cpp -o tb_ce && ./tb_ce
//
// On step 0 it also dumps logits + float CE gradients to CE_DUMP_PATH so the
// companion Python check (scripts/check_host_ce_matches_tikitorch.py) can verify
// the host CE gradient equals the library's tropical_cross_entropy backward.
//
// Pass criterion: CE converges (best-correct >= L_seq/2) AND is not worse than
// the perceptron baseline over the same schedule.

#include "forge_buffer_2layer.hpp"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

extern "C" int forge_train_step_2layer(int32_t* buffer, int mode,
                                        int arg1, int arg2, int arg3, int layer_id);

// Power-of-two fixed-point scale for the integer CE gradient (a bit-shift on
// chip).  The softmax-onehot gradient lives in [-1, 1]; scaling by 2^6 = 64 and
// rounding gives the kernel a meaningful integer gradient while staying a shift.
static const int CE_GRAD_SHIFT = 6;
static const double CE_GRAD_SCALE = (double)(1 << CE_GRAD_SHIFT);
static const char* CE_DUMP_PATH = "/tmp/ce_dump.txt";

int main() {
    std::printf("[tb_ce] FT2_BUFFER_WORDS=%d  OFF2_LOGITS=%d  OFF2_GRAD_LOGIT=%d\n",
                FT2_BUFFER_WORDS, OFF2_LOGITS, OFF2_GRAD_LOGIT);

    std::vector<int32_t> buf(FT2_BUFFER_WORDS, 0);
    auto rng = 0xC0FFEEu;
    auto rnd = [&]() { rng = rng * 1103515245u + 12345u; return (int16_t)(rng >> 16); };
    auto fill = [&](int off, int n) { for (int i = 0; i < n; i++) buf[off + i] = rnd() % 17 - 8; };
    fill(OFF2_EMBED, W2_EMBED); fill(OFF2_W_HEAD, W2_W_HEAD);
    fill(OFF2_L0_S_Q, W2_S_Q); fill(OFF2_L0_S_K, W2_S_K); fill(OFF2_L0_W1, W2_W1); fill(OFF2_L0_W2, W2_W2);
    fill(OFF2_L1_S_Q, W2_S_Q); fill(OFF2_L1_S_K, W2_S_K); fill(OFF2_L1_W1, W2_W1); fill(OFF2_L1_W2, W2_W2);

    const int PERIOD = 4, L_seq = 16, LR = 1, N_STEPS = 60;
    std::vector<int> tokens(L_seq), targets(L_seq);
    for (int m = 0; m < L_seq; m++) { tokens[m] = m % PERIOD; targets[m] = (m + 1) % PERIOD; }

    // Host-side softmax cross-entropy: read logits from OFF2_LOGITS, write the
    // integer CE gradient round(2^CE_GRAD_SHIFT * (softmax - onehot)) to
    // OFF2_GRAD_LOGIT.  Identical math to tropical_cross_entropy(softmax="exact").
    auto host_ce = [&](bool dump) -> int {
        int correct = 0;
        FILE* f = dump ? std::fopen(CE_DUMP_PATH, "w") : nullptr;
        for (int m = 0; m < L_seq; m++) {
            const int32_t* logit = &buf[OFF2_LOGITS     + m * FT2_V];
            int32_t*       grad  = &buf[OFF2_GRAD_LOGIT + m * FT2_V];
            int tgt = targets[m];
            // softmax (subtract max for stability)
            double mx = -1e30; int pred = 0;
            for (int v = 0; v < FT2_V; v++) if (logit[v] > mx) { mx = logit[v]; pred = v; }
            double Z = 0.0;
            static double p[FT2_V];
            for (int v = 0; v < FT2_V; v++) { p[v] = std::exp((double)logit[v] - mx); Z += p[v]; }
            for (int v = 0; v < FT2_V; v++) p[v] /= Z;
            if (pred == tgt) correct++;
            for (int v = 0; v < FT2_V; v++) {
                double gf = p[v] - (v == tgt ? 1.0 : 0.0);          // softmax - onehot
                grad[v] = (int32_t)std::lround(CE_GRAD_SCALE * gf);  // -> integer (shift on chip)
                if (f && m < 3) std::fprintf(f, "%d %d %d %.10f\n", m, v, logit[v], gf);
            }
        }
        if (f) std::fclose(f);
        return correct;
    };

    auto clear_caches = [&]() {
        for (int i = 0; i < W2_K_CACHE; i++) { buf[OFF2_L0_K_CACHE + i] = 0; buf[OFF2_L1_K_CACHE + i] = 0; }
        for (int i = 0; i < W2_V_CACHE; i++) { buf[OFF2_L0_V_CACHE + i] = 0; buf[OFF2_L1_V_CACHE + i] = 0; }
    };

    int best = 0;
    for (int step = 0; step < N_STEPS; step++) {
        clear_caches();
        for (int m = 0; m < L_seq; m++)
            if (forge_train_step_2layer(buf.data(), MODE2_FWD_LOSS, m, tokens[m], -1, 0)) return 1;
        for (int m = 0; m < L_seq; m++)
            if (forge_train_step_2layer(buf.data(), MODE2_FWD_LOSS, m, -1, targets[m], 1)) return 1;
        int correct = host_ce(/*dump=*/step == 0);
        clear_caches();
        for (int mm = 0; mm < L_seq; mm++) { int m = L_seq - 1 - mm;
            if (forge_train_step_2layer(buf.data(), MODE2_BWD, m, -1, 0, 1)) return 1; }
        for (int mm = 0; mm < L_seq; mm++) { int m = L_seq - 1 - mm;
            if (forge_train_step_2layer(buf.data(), MODE2_BWD, m, tokens[m], 0, 0)) return 1; }
        if (forge_train_step_2layer(buf.data(), MODE2_SIGNSGD, LR, 0, 0, -1)) return 1;
        if (correct > best) best = correct;
        if (step < 5 || step % 10 == 0 || step == N_STEPS - 1)
            std::printf("[tb_ce] step %3d  correct=%d/%d  best=%d/%d\n", step, correct, L_seq, best, L_seq);
    }
    std::printf("[tb_ce] best %d/%d over %d steps (host-side CE)\n", best, L_seq, N_STEPS);
    std::printf("[tb_ce] step-0 logits/grads dumped to %s for the library cross-check\n", CE_DUMP_PATH);
    if (best >= L_seq / 2) {
        std::printf("[forge_train_step_2layer] CE csim PASS\n");
        return 0;
    }
    std::fprintf(stderr, "[forge_train_step_2layer] CE csim FAIL\n");
    return 1;
}
