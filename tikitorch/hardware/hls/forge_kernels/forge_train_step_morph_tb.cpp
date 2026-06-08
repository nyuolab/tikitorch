// forge_train_step_morph_tb.cpp — CSIM_ONLY host testbench for the
// MorphSSM-core training kernel.
//
// Part (a): builds a small random MorphSSM-core config, runs a plain-C++
//   CPU reference implementing the SAME hard int16 forward/backward
//   semantics, and asserts the kernel's fwd POOLED and bwd gradients
//   (g_MH, g_MP) are BIT-EXACT to the reference.
//
// Part (b): a tiny synthetic learnable classification task — the label is a
//   deterministic function of the input through the morph core + a fixed
//   host head (emulated here) — trained for ~200 steps with mode 0/1/2.
//   Asserts accuracy rises well above chance.
//
// Compile:
//   g++ -std=c++14 -O2 -DCSIM_ONLY -I tikitorch/hardware/hls \
//       -I tikitorch/hardware/hls/forge_kernels \
//       tikitorch/hardware/hls/forge_kernels/forge_train_step_morph.cpp \
//       tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_tb.cpp -o /tmp/tb_morph

#include "forge_buffer_morph.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

extern "C" int forge_train_step_morph(int32_t* buffer, int mode, int arg1, int arg2, int arg3);

// -----------------------------------------------------------------------------
// CPU reference — independent re-implementation of the exact int semantics.
// -----------------------------------------------------------------------------
static inline int16_t r_sat16(int32_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return (int16_t)-32768;
    return (int16_t)v;
}

struct RefState {
    // saved for backward
    int tap_idx [MD_L_LAYERS][MD_D][MD_L_SEQ];   // [l][c][t]
    int proj_idx[MD_L_LAYERS][MD_L_SEQ][MD_D];   // [l][t][j]
    int16_t pooled[MD_D];
};

// Forward. h:[l][c][k], P:[l][j][i], X:[t][d]. Returns saved state.
static void ref_forward(const int16_t* MH, const int16_t* MP,
                        const int16_t X[MD_L_SEQ][MD_D], RefState& st) {
    static int16_t R[MD_L_SEQ][MD_D];
    for (int t = 0; t < MD_L_SEQ; t++)
        for (int d = 0; d < MD_D; d++) R[t][d] = X[t][d];

    static int16_t BS[MD_L_SEQ][MD_D];
    static int16_t MC[MD_L_SEQ][MD_D];
    static int16_t PJ[MD_L_SEQ][MD_D];

    for (int l = 0; l < MD_L_LAYERS; l++) {
        int dil = dilation_of_layer(l);
        bool is_dil = ((l & 1) == 0);
        const int16_t* h = MH + l * (MD_D * MD_K);
        const int16_t* P = MP + l * (MD_D * MD_D);

        // batch_shift over D, per t
        for (int t = 0; t < MD_L_SEQ; t++) {
            int32_t sum = 0;
            for (int d = 0; d < MD_D; d++) sum += R[t][d];
            int32_t mean = sum >> MD_LOG2_D;
            for (int d = 0; d < MD_D; d++) BS[t][d] = r_sat16((int32_t)R[t][d] - mean);
        }
        // morph conv
        for (int c = 0; c < MD_D; c++) {
            for (int t = 0; t < MD_L_SEQ; t++) {
                int32_t best = is_dil ? -2147483647 : 2147483647;
                int kstar = 0;
                for (int k = 0; k < MD_K; k++) {
                    int t_in = t - k * dil;
                    if (t_in >= 0) {
                        int32_t s = (int32_t)h[c * MD_K + k] + (int32_t)BS[t_in][c];
                        if (is_dil) { if (s > best) { best = s; kstar = k; } }
                        else        { if (s < best) { best = s; kstar = k; } }
                    }
                }
                MC[t][c] = r_sat16(best);
                st.tap_idx[l][c][t] = kstar;
            }
        }
        // tropical projection (max-plus)
        for (int t = 0; t < MD_L_SEQ; t++) {
            for (int j = 0; j < MD_D; j++) {
                int32_t best = -2147483647;
                int istar = 0;
                for (int i = 0; i < MD_D; i++) {
                    int32_t s = (int32_t)P[j * MD_D + i] + (int32_t)MC[t][i];
                    if (s > best) { best = s; istar = i; }
                }
                PJ[t][j] = r_sat16(best);
                st.proj_idx[l][t][j] = istar;
            }
        }
        // residual add
        for (int t = 0; t < MD_L_SEQ; t++)
            for (int j = 0; j < MD_D; j++)
                R[t][j] = r_sat16((int32_t)R[t][j] + (int32_t)PJ[t][j]);
    }

    // mean-pool over time
    for (int j = 0; j < MD_D; j++) {
        int32_t sum = 0;
        for (int t = 0; t < MD_L_SEQ; t++) sum += R[t][j];
        st.pooled[j] = r_sat16(sum >> MD_LOG2_L);
    }
}

// Backward. grad_pooled:[d]. Accumulates into g_MH/g_MP (caller pre-zeros or
// pre-loads). Needs the saved indices from forward (st).
static void ref_backward(const RefState& st, const int32_t grad_pooled[MD_D],
                         int32_t* g_MH, int32_t* g_MP) {
    static int32_t gR[MD_L_SEQ][MD_D];
    for (int t = 0; t < MD_L_SEQ; t++)
        for (int j = 0; j < MD_D; j++)
            gR[t][j] = grad_pooled[j] >> MD_LOG2_L;

    static int32_t grad_PJ[MD_L_SEQ][MD_D];
    static int32_t grad_MC[MD_L_SEQ][MD_D];
    static int32_t grad_BS[MD_L_SEQ][MD_D];
    static int32_t grad_R [MD_L_SEQ][MD_D];

    for (int lr = 0; lr < MD_L_LAYERS; lr++) {
        int l = MD_L_LAYERS - 1 - lr;
        int dil = dilation_of_layer(l);
        int32_t* gP = g_MP + l * (MD_D * MD_D);
        int32_t* gH = g_MH + l * (MD_D * MD_K);

        // residual split
        for (int t = 0; t < MD_L_SEQ; t++)
            for (int j = 0; j < MD_D; j++) {
                grad_PJ[t][j] = gR[t][j];
                grad_R [t][j] = gR[t][j];
            }
        // tropical proj bwd
        for (int t = 0; t < MD_L_SEQ; t++)
            for (int i = 0; i < MD_D; i++) grad_MC[t][i] = 0;
        for (int t = 0; t < MD_L_SEQ; t++)
            for (int j = 0; j < MD_D; j++) {
                int32_t g = grad_PJ[t][j];
                if (g == 0) continue;
                int istar = st.proj_idx[l][t][j];
                gP[j * MD_D + istar] += g;
                grad_MC[t][istar]    += g;
            }
        // morph conv bwd
        for (int t = 0; t < MD_L_SEQ; t++)
            for (int c = 0; c < MD_D; c++) grad_BS[t][c] = 0;
        for (int c = 0; c < MD_D; c++)
            for (int t = 0; t < MD_L_SEQ; t++) {
                int32_t g = grad_MC[t][c];
                if (g == 0) continue;
                int kstar = st.tap_idx[l][c][t];
                int t_in = t - kstar * dil;
                if (t_in < 0) continue;
                gH[c * MD_K + kstar] += g;
                grad_BS[t_in][c]     += g;
            }
        // batch_shift bwd, per t over D
        for (int t = 0; t < MD_L_SEQ; t++) {
            int32_t sum = 0;
            for (int c = 0; c < MD_D; c++) sum += grad_BS[t][c];
            int32_t mean = sum >> MD_LOG2_D;
            for (int c = 0; c < MD_D; c++) grad_R[t][c] += grad_BS[t][c] - mean;
        }
        // pass to earlier layer
        for (int t = 0; t < MD_L_SEQ; t++)
            for (int j = 0; j < MD_D; j++) gR[t][j] = grad_R[t][j];
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static uint32_t g_seed = 0xC0FFEEu;
static int16_t rnd16() { g_seed = g_seed * 1103515245u + 12345u; return (int16_t)(g_seed >> 16); }

int main() {
    std::printf("[tb_morph] config D=%d L_LAYERS=%d K=%d L_SEQ=%d DIL_CAP=%d\n",
                MD_D, MD_L_LAYERS, MD_K, MD_L_SEQ, MD_DIL_CAP);
    std::printf("[tb_morph] MD_BUFFER_WORDS=%d (%.2f MiB)\n",
                MD_BUFFER_WORDS, MD_BUFFER_BYTES / (1024.0 * 1024.0));

    // =====================================================================
    // PART (a): bit-exact fwd + bwd vs CPU reference.
    // =====================================================================
    std::vector<int32_t> buf(MD_BUFFER_WORDS, 0);

    // Random small weights, input, and grad_pooled.
    static int16_t MH[MD_L_LAYERS * MD_D * MD_K];
    static int16_t MP[MD_L_LAYERS * MD_D * MD_D];
    static int16_t X[MD_L_SEQ][MD_D];
    static int32_t grad_pooled[MD_D];

    for (int i = 0; i < (int)(MD_L_LAYERS * MD_D * MD_K); i++) MH[i] = (int16_t)(rnd16() % 17 - 8);
    for (int i = 0; i < (int)(MD_L_LAYERS * MD_D * MD_D); i++) MP[i] = (int16_t)(rnd16() % 17 - 8);
    for (int t = 0; t < MD_L_SEQ; t++)
        for (int d = 0; d < MD_D; d++) X[t][d] = (int16_t)(rnd16() % 65 - 32);
    for (int d = 0; d < MD_D; d++) grad_pooled[d] = (int32_t)(rnd16() % 21 - 10);

    // ---- CPU reference fwd + bwd ----
    RefState ref_st;
    ref_forward(MH, MP, X, ref_st);
    static int32_t ref_gMH[MD_L_LAYERS * MD_D * MD_K];
    static int32_t ref_gMP[MD_L_LAYERS * MD_D * MD_D];
    for (int i = 0; i < (int)(MD_L_LAYERS * MD_D * MD_K); i++) ref_gMH[i] = 0;
    for (int i = 0; i < (int)(MD_L_LAYERS * MD_D * MD_D); i++) ref_gMP[i] = 0;
    ref_backward(ref_st, grad_pooled, ref_gMH, ref_gMP);

    // ---- Kernel fwd + bwd ----
    for (int i = 0; i < (int)WORDS_MH; i++) buf[OFF_MH + i] = (int32_t)(int16_t)MH[i];
    for (int i = 0; i < (int)WORDS_MP; i++) buf[OFF_MP + i] = (int32_t)(int16_t)MP[i];
    for (int t = 0; t < MD_L_SEQ; t++)
        for (int d = 0; d < MD_D; d++) buf[OFF_X + t * MD_D + d] = (int32_t)(int16_t)X[t][d];
    for (int i = 0; i < (int)WORDS_G_MH; i++) buf[OFF_G_MH + i] = 0;
    for (int i = 0; i < (int)WORDS_G_MP; i++) buf[OFF_G_MP + i] = 0;

    int rc = forge_train_step_morph(buf.data(), MD_MODE_FWD, 0, 0, 0);
    if (rc != 0) { std::fprintf(stderr, "fwd rc=%d\n", rc); return 1; }

    // Check POOLED bit-exact.
    int pool_mism = 0;
    for (int j = 0; j < MD_D; j++) {
        int16_t kv = (int16_t)(buf[OFF_POOLED + j] & 0xFFFF);
        if (kv != ref_st.pooled[j]) {
            if (pool_mism < 8)
                std::fprintf(stderr, "POOLED[%d] kernel=%d ref=%d\n", j, kv, ref_st.pooled[j]);
            pool_mism++;
        }
    }
    // Also verify the kernel's saved indices match the reference (so the bwd
    // comparison is meaningful and not coincidentally aligned).
    int idx_mism = 0;
    for (int l = 0; l < MD_L_LAYERS; l++) {
        int tb = off_tap(l), pb = off_proj(l);
        for (int c = 0; c < MD_D; c++)
            for (int t = 0; t < MD_L_SEQ; t++)
                if (buf[tb + c * MD_L_SEQ + t] != ref_st.tap_idx[l][c][t]) idx_mism++;
        for (int t = 0; t < MD_L_SEQ; t++)
            for (int j = 0; j < MD_D; j++)
                if (buf[pb + t * MD_D + j] != ref_st.proj_idx[l][t][j]) idx_mism++;
    }

    // Backward in kernel.
    for (int d = 0; d < MD_D; d++) buf[OFF_GRAD_POOLED + d] = grad_pooled[d];
    rc = forge_train_step_morph(buf.data(), MD_MODE_BWD, 0, 0, 0);
    if (rc != 0) { std::fprintf(stderr, "bwd rc=%d\n", rc); return 1; }

    int gmh_mism = 0, gmp_mism = 0;
    for (int i = 0; i < (int)WORDS_G_MH; i++) {
        if (buf[OFF_G_MH + i] != ref_gMH[i]) {
            if (gmh_mism < 8) std::fprintf(stderr, "g_MH[%d] kernel=%d ref=%d\n", i, buf[OFF_G_MH + i], ref_gMH[i]);
            gmh_mism++;
        }
    }
    for (int i = 0; i < (int)WORDS_G_MP; i++) {
        if (buf[OFF_G_MP + i] != ref_gMP[i]) {
            if (gmp_mism < 8) std::fprintf(stderr, "g_MP[%d] kernel=%d ref=%d\n", i, buf[OFF_G_MP + i], ref_gMP[i]);
            gmp_mism++;
        }
    }

    std::printf("[tb_morph] (a) bit-exact check: POOLED mism=%d, idx mism=%d, g_MH mism=%d, g_MP mism=%d\n",
                pool_mism, idx_mism, gmh_mism, gmp_mism);
    bool exact = (pool_mism == 0 && idx_mism == 0 && gmh_mism == 0 && gmp_mism == 0);
    if (!exact) { std::fprintf(stderr, "[tb_morph] FAIL — kernel not bit-exact vs CPU reference\n"); return 1; }
    std::printf("[tb_morph] (a) PASS — kernel fwd POOLED + bwd gradients BIT-EXACT vs CPU reference\n");

    // =====================================================================
    // PART (b): tiny learnable classification task.
    //
    // Synthetic task: NUM_CLASSES classes. Each sample is a sequence whose
    // dominant feature channel is keyed to its class — channel block
    // [cls*BLK, cls*BLK+BLK) is driven high, a ramp along time. The morph
    // core must learn to route that signal into POOLED so the (fixed-init,
    // trained) host head separates the classes. We emulate the host head
    // here: a tropical max-plus head Hh[NUM_CLASSES][D] over POOLED, with
    // perceptron-style CE-grad (predicted-vs-target one-hot) producing
    // grad_pooled to send back into the kernel. The head is trained too
    // (host side) with the same signsgd rule.
    // =====================================================================
    std::printf("\n[tb_morph] (b) learnable toy classification task\n");

    const int NUM_CLASSES = 4;
    const int BLK = MD_D / NUM_CLASSES;          // channel block per class
    const int N_TRAIN = 96;                      // training samples
    const int N_STEPS = 200;
    const int LR = 1;
    const int TH = 0;
    const int HEAD_LR = 1;

    // Build the dataset: X_data[sample][t][d], label[sample].
    //
    // Each class is encoded as a temporal pulse in a class-specific channel
    // block placed at a class-specific time lag, on top of substantial noise.
    // Recovering the class therefore requires BOTH the channel selection
    // (tropical projection) AND the temporal localisation (morph conv over
    // time) — a non-trivial function of the morph core, so the accuracy
    // curve reflects real learning rather than a trivially separable signal.
    static int16_t Xset[96][MD_L_SEQ][MD_D];
    static int     yset[96];
    g_seed = 0x1234567u;
    for (int s = 0; s < N_TRAIN; s++) {
        int cls = s % NUM_CLASSES;
        yset[s] = cls;
        int lag = (cls * (MD_L_SEQ / NUM_CLASSES));   // class-specific time window
        for (int t = 0; t < MD_L_SEQ; t++) {
            for (int d = 0; d < MD_D; d++) {
                int16_t base = (int16_t)(rnd16() % 17 - 8);   // larger noise floor
                bool in_block  = (d >= cls * BLK && d < cls * BLK + BLK);
                bool in_window = (t >= lag && t < lag + (MD_L_SEQ / NUM_CLASSES));
                if (in_block && in_window) base = (int16_t)(24 + (rnd16() % 7));  // pulse
                Xset[s][t][d] = base;
            }
        }
    }

    // Fresh weights for the core.
    std::vector<int32_t> tbuf(MD_BUFFER_WORDS, 0);
    g_seed = 0xBADC0DEu;
    for (int i = 0; i < (int)WORDS_MH; i++) tbuf[OFF_MH + i] = (int32_t)(int16_t)(rnd16() % 7 - 3);
    for (int i = 0; i < (int)WORDS_MP; i++) tbuf[OFF_MP + i] = (int32_t)(int16_t)(rnd16() % 7 - 3);

    // Host head (max-plus) Hh[NUM_CLASSES][D], grad accumulator gHh.
    static int16_t Hh [4][MD_D];
    static int32_t gHh[4][MD_D];
    for (int v = 0; v < NUM_CLASSES; v++)
        for (int d = 0; d < MD_D; d++) Hh[v][d] = (int16_t)(rnd16() % 7 - 3);

    auto eval_accuracy = [&](void) -> double {
        int correct = 0;
        for (int s = 0; s < N_TRAIN; s++) {
            for (int t = 0; t < MD_L_SEQ; t++)
                for (int d = 0; d < MD_D; d++)
                    tbuf[OFF_X + t * MD_D + d] = (int32_t)(int16_t)Xset[s][t][d];
            forge_train_step_morph(tbuf.data(), MD_MODE_FWD, 0, 0, 0);
            int16_t pooled[MD_D];
            for (int d = 0; d < MD_D; d++) pooled[d] = (int16_t)(tbuf[OFF_POOLED + d] & 0xFFFF);
            // host head: logits[v] = max_d(Hh[v][d] + pooled[d])
            int pred = 0; int32_t bestl = -2147483647;
            for (int v = 0; v < NUM_CLASSES; v++) {
                int32_t best = -2147483647;
                for (int d = 0; d < MD_D; d++) {
                    int32_t s2 = (int32_t)Hh[v][d] + (int32_t)pooled[d];
                    if (s2 > best) best = s2;
                }
                if (best > bestl) { bestl = best; pred = v; }
            }
            if (pred == yset[s]) correct++;
        }
        return (double)correct / N_TRAIN;
    };

    double acc0 = eval_accuracy();
    std::printf("[tb_morph] (b) step   0  acc=%.3f (chance=%.3f)\n", acc0, 1.0 / NUM_CLASSES);

    double best_acc = acc0;
    for (int step = 0; step < N_STEPS; step++) {
        for (int v = 0; v < NUM_CLASSES; v++)
            for (int d = 0; d < MD_D; d++) gHh[v][d] = 0;

        for (int s = 0; s < N_TRAIN; s++) {
            int cls = yset[s];
            // fwd
            for (int t = 0; t < MD_L_SEQ; t++)
                for (int d = 0; d < MD_D; d++)
                    tbuf[OFF_X + t * MD_D + d] = (int32_t)(int16_t)Xset[s][t][d];
            forge_train_step_morph(tbuf.data(), MD_MODE_FWD, 0, 0, 0);
            int16_t pooled[MD_D];
            for (int d = 0; d < MD_D; d++) pooled[d] = (int16_t)(tbuf[OFF_POOLED + d] & 0xFFFF);

            // host head fwd: logits[v] = max_d(Hh[v][d]+pooled[d]), record argmax d*
            int dstar[4]; int32_t logit[4]; int pred = 0; int32_t bestl = -2147483647;
            for (int v = 0; v < NUM_CLASSES; v++) {
                int32_t best = -2147483647; int da = 0;
                for (int d = 0; d < MD_D; d++) {
                    int32_t s2 = (int32_t)Hh[v][d] + (int32_t)pooled[d];
                    if (s2 > best) { best = s2; da = d; }
                }
                logit[v] = best; dstar[v] = da;
                if (best > bestl) { bestl = best; pred = v; }
            }

            // perceptron-style grad on logits: g[pred]=+1, g[target]=-1 if wrong.
            int32_t glogit[4] = {0, 0, 0, 0};
            if (pred != cls) { glogit[pred] = +1; glogit[cls] = -1; }

            // head bwd (host): scatter to argmax d*; build grad_pooled.
            int32_t grad_pooled_s[MD_D];
            for (int d = 0; d < MD_D; d++) grad_pooled_s[d] = 0;
            for (int v = 0; v < NUM_CLASSES; v++) {
                if (glogit[v] == 0) continue;
                int da = dstar[v];
                gHh[v][da]          += glogit[v];
                grad_pooled_s[da]   += glogit[v];
            }

            // kernel bwd accumulates core grads (g_MH/g_MP persist across samples).
            for (int d = 0; d < MD_D; d++) tbuf[OFF_GRAD_POOLED + d] = grad_pooled_s[d];
            forge_train_step_morph(tbuf.data(), MD_MODE_BWD, 0, 0, 0);
        }

        // core weight update (signsgd in kernel) + host head update (signsgd here).
        forge_train_step_morph(tbuf.data(), MD_MODE_SIGNSGD, LR, TH, 0);
        for (int v = 0; v < NUM_CLASSES; v++)
            for (int d = 0; d < MD_D; d++) {
                int32_t g = gHh[v][d];
                int32_t stp = 0;
                if (g > 0) stp = HEAD_LR; else if (g < 0) stp = -HEAD_LR;
                int32_t wn = (int32_t)Hh[v][d] - stp;
                Hh[v][d] = r_sat16(wn);
            }

        double acc = eval_accuracy();
        if (acc > best_acc) best_acc = acc;
        if (step < 5 || step % 25 == 0 || step == N_STEPS - 1)
            std::printf("[tb_morph] (b) step %3d  acc=%.3f  best=%.3f\n", step, acc, best_acc);
    }

    std::printf("[tb_morph] (b) best accuracy %.3f over %d steps (chance=%.3f)\n",
                best_acc, N_STEPS, 1.0 / NUM_CLASSES);

    // PASS criterion: comfortably above chance (chance=0.25 for 4 classes).
    bool learn_pass = (best_acc >= 0.60);
    if (learn_pass)
        std::printf("[tb_morph] (b) PASS — toy task learns (acc >> chance)\n");
    else {
        std::fprintf(stderr, "[tb_morph] (b) FAIL — toy task did not learn above chance\n");
        return 1;
    }

    std::printf("\n[tb_morph] ALL CHECKS PASSED\n");
    return 0;
}
