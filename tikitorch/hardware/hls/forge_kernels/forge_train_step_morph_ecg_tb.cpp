// forge_train_step_morph_ecg_tb.cpp — CSIM_ONLY host testbench for the
// STREAMING ECG-scale MorphSSM-core training kernel.
//
// The kernel config (D / L_LAYERS / K / L_SEQ / DIL_CAP) is set at compile
// time via the -DMDE_*_CFG macros in forge_buffer_morph_ecg.hpp. This TB is
// compiled at a SMALL L_SEQ for a fast bit-exact C-sim; rebuild with a few
// different L_SEQ (and once at the full DIL/K) to exercise the sliding-window
// / cross-timestep-grad logic at multiple lengths.
//
// Part (a): a from-scratch DENSE CPU reference (the same hard int16 semantics)
//   asserts the streaming kernel's fwd POOLED and bwd g_MH/g_MP are BIT-EXACT.
//   The reference is dense (holds the whole sequence) — that is fine for a
//   small L_SEQ and is exactly what validates the streaming reformulation.
//
// Part (b): a tiny synthetic learnable classification task (host head emulated
//   here), trained with mode 0/1/2; asserts accuracy rises above chance.
//
// Compile (example, small L_SEQ; small DIL/K to keep C-sim fast):
//   g++ -std=c++14 -O2 -DCSIM_ONLY \
//       -DMDE_D_CFG=64 -DMDE_L_LAYERS_CFG=4 -DMDE_K_CFG=4 \
//       -DMDE_L_SEQ_CFG=128 -DMDE_DIL_CAP_CFG=16 \
//       -I forge/hardware/hls -I forge/hardware/hls/forge_kernels \
//       forge/hardware/hls/forge_kernels/forge_train_step_morph_ecg.cpp \
//       forge/hardware/hls/forge_kernels/forge_train_step_morph_ecg_tb.cpp -o /tmp/tb_ecg

#include "forge_buffer_morph_ecg.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

extern "C" int forge_train_step_morph_ecg(int32_t* buffer, int mode, int arg1, int arg2, int arg3);

// -----------------------------------------------------------------------------
// CPU reference — independent DENSE re-implementation of the exact int
// semantics (identical math to forge_train_step_morph's reference, retargeted
// to the ECG header constants). Holds full [L_SEQ][D] activations.
// -----------------------------------------------------------------------------
static inline int16_t r_sat16(int32_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return (int16_t)-32768;
    return (int16_t)v;
}

struct RefState {
    std::vector<int> tap_idx;    // [l][c][t] flattened
    std::vector<int> proj_idx;   // [l][t][j] flattened
    std::vector<int16_t> pooled; // [D]
    RefState()
        : tap_idx((size_t)MDE_L_LAYERS * MDE_D * MDE_L_SEQ),
          proj_idx((size_t)MDE_L_LAYERS * MDE_L_SEQ * MDE_D),
          pooled(MDE_D) {}
    int  tap (int l, int c, int t) const { return tap_idx[((size_t)l * MDE_D + c) * MDE_L_SEQ + t]; }
    int& tap (int l, int c, int t)       { return tap_idx[((size_t)l * MDE_D + c) * MDE_L_SEQ + t]; }
    int  proj(int l, int t, int j) const { return proj_idx[((size_t)l * MDE_L_SEQ + t) * MDE_D + j]; }
    int& proj(int l, int t, int j)       { return proj_idx[((size_t)l * MDE_L_SEQ + t) * MDE_D + j]; }
};

static void ref_forward(const int16_t* MH, const int16_t* MP,
                        const std::vector<int16_t>& X, RefState& st) {
    std::vector<int16_t> R((size_t)MDE_L_SEQ * MDE_D);
    for (size_t i = 0; i < R.size(); i++) R[i] = X[i];
    std::vector<int16_t> BS((size_t)MDE_L_SEQ * MDE_D);
    std::vector<int16_t> MC((size_t)MDE_L_SEQ * MDE_D);
    std::vector<int16_t> PJ((size_t)MDE_L_SEQ * MDE_D);

    for (int l = 0; l < MDE_L_LAYERS; l++) {
        int dil = dilation_of_layer_e(l);
        bool is_dil = ((l & 1) == 0);
        const int16_t* h = MH + l * (MDE_D * MDE_K);
        const int16_t* P = MP + l * (MDE_D * MDE_D);

        for (int t = 0; t < MDE_L_SEQ; t++) {
            int32_t sum = 0;
            for (int d = 0; d < MDE_D; d++) sum += R[(size_t)t * MDE_D + d];
            int32_t mean = sum >> MDE_LOG2_D;
            for (int d = 0; d < MDE_D; d++)
                BS[(size_t)t * MDE_D + d] = r_sat16((int32_t)R[(size_t)t * MDE_D + d] - mean);
        }
        for (int c = 0; c < MDE_D; c++) {
            for (int t = 0; t < MDE_L_SEQ; t++) {
                int32_t best = is_dil ? -2147483647 : 2147483647;
                int kstar = 0;
                for (int k = 0; k < MDE_K; k++) {
                    int t_in = t - k * dil;
                    if (t_in >= 0) {
                        int32_t s = (int32_t)h[c * MDE_K + k] + (int32_t)BS[(size_t)t_in * MDE_D + c];
                        if (is_dil) { if (s > best) { best = s; kstar = k; } }
                        else        { if (s < best) { best = s; kstar = k; } }
                    }
                }
                MC[(size_t)t * MDE_D + c] = r_sat16(best);
                st.tap(l, c, t) = kstar;
            }
        }
        for (int t = 0; t < MDE_L_SEQ; t++) {
            for (int j = 0; j < MDE_D; j++) {
                int32_t best = -2147483647;
                int istar = 0;
                for (int i = 0; i < MDE_D; i++) {
                    int32_t s = (int32_t)P[j * MDE_D + i] + (int32_t)MC[(size_t)t * MDE_D + i];
                    if (s > best) { best = s; istar = i; }
                }
                PJ[(size_t)t * MDE_D + j] = r_sat16(best);
                st.proj(l, t, j) = istar;
            }
        }
        for (int t = 0; t < MDE_L_SEQ; t++)
            for (int j = 0; j < MDE_D; j++) {
                size_t idx = (size_t)t * MDE_D + j;
                R[idx] = r_sat16((int32_t)R[idx] + (int32_t)PJ[idx]);
            }
    }

    for (int j = 0; j < MDE_D; j++) {
        int32_t sum = 0;
        for (int t = 0; t < MDE_L_SEQ; t++) sum += R[(size_t)t * MDE_D + j];
        st.pooled[j] = r_sat16(sum >> MDE_LOG2_L);
    }
}

static void ref_backward(const RefState& st, const int32_t grad_pooled[],
                         int32_t* g_MH, int32_t* g_MP,
                         int32_t* grad_X /* [L_SEQ][D], may be null */) {
    std::vector<int32_t> gR((size_t)MDE_L_SEQ * MDE_D);
    for (int t = 0; t < MDE_L_SEQ; t++)
        for (int j = 0; j < MDE_D; j++)
            gR[(size_t)t * MDE_D + j] = grad_pooled[j] >> MDE_LOG2_L;

    std::vector<int32_t> grad_PJ((size_t)MDE_L_SEQ * MDE_D);
    std::vector<int32_t> grad_MC((size_t)MDE_L_SEQ * MDE_D);
    std::vector<int32_t> grad_BS((size_t)MDE_L_SEQ * MDE_D);
    std::vector<int32_t> grad_R ((size_t)MDE_L_SEQ * MDE_D);

    for (int lr = 0; lr < MDE_L_LAYERS; lr++) {
        int l = MDE_L_LAYERS - 1 - lr;
        int dil = dilation_of_layer_e(l);
        int32_t* gP = g_MP + l * (MDE_D * MDE_D);
        int32_t* gH = g_MH + l * (MDE_D * MDE_K);

        for (int t = 0; t < MDE_L_SEQ; t++)
            for (int j = 0; j < MDE_D; j++) {
                size_t idx = (size_t)t * MDE_D + j;
                grad_PJ[idx] = gR[idx];
                grad_R [idx] = gR[idx];
            }
        for (size_t i = 0; i < grad_MC.size(); i++) grad_MC[i] = 0;
        for (int t = 0; t < MDE_L_SEQ; t++)
            for (int j = 0; j < MDE_D; j++) {
                int32_t g = grad_PJ[(size_t)t * MDE_D + j];
                if (g == 0) continue;
                int istar = st.proj(l, t, j);
                gP[j * MDE_D + istar]              += g;
                grad_MC[(size_t)t * MDE_D + istar] += g;
            }
        for (size_t i = 0; i < grad_BS.size(); i++) grad_BS[i] = 0;
        for (int c = 0; c < MDE_D; c++)
            for (int t = 0; t < MDE_L_SEQ; t++) {
                int32_t g = grad_MC[(size_t)t * MDE_D + c];
                if (g == 0) continue;
                int kstar = st.tap(l, c, t);
                int t_in = t - kstar * dil;
                if (t_in < 0) continue;
                gH[c * MDE_K + kstar]              += g;
                grad_BS[(size_t)t_in * MDE_D + c] += g;
            }
        for (int t = 0; t < MDE_L_SEQ; t++) {
            int32_t sum = 0;
            for (int c = 0; c < MDE_D; c++) sum += grad_BS[(size_t)t * MDE_D + c];
            int32_t mean = sum >> MDE_LOG2_D;
            for (int c = 0; c < MDE_D; c++)
                grad_R[(size_t)t * MDE_D + c] += grad_BS[(size_t)t * MDE_D + c] - mean;
        }
        for (size_t i = 0; i < gR.size(); i++) gR[i] = grad_R[i];
    }

    // After the last (lowest, l=0) layer, gR == grad wrt the layer-0 input =
    // grad_X. Same value the streaming kernel persists to OFF_GRAD_X_E.
    if (grad_X) {
        for (size_t i = 0; i < gR.size(); i++) grad_X[i] = gR[i];
    }
}

// -----------------------------------------------------------------------------
static uint32_t g_seed = 0xC0FFEEu;
static int16_t rnd16() { g_seed = g_seed * 1103515245u + 12345u; return (int16_t)(g_seed >> 16); }

int main() {
    std::printf("[tb_ecg] config D=%d L_LAYERS=%d K=%d L_SEQ=%d DIL_CAP=%d  (WSPAN=%d WPOW=%d)\n",
                MDE_D, MDE_L_LAYERS, MDE_K, MDE_L_SEQ, MDE_DIL_CAP, MDE_WSPAN, MDE_WPOW);
    std::printf("[tb_ecg] LOG2_D=%d LOG2_L=%d  FT_BUFFER_WORDS_E=%d (%.2f MiB)\n",
                MDE_LOG2_D, MDE_LOG2_L, FT_BUFFER_WORDS_E, FT_BUFFER_BYTES / (1024.0 * 1024.0));

    // =====================================================================
    // PART (a): bit-exact fwd + bwd vs dense CPU reference.
    // =====================================================================
    std::vector<int32_t> buf(FT_BUFFER_WORDS_E, 0);

    std::vector<int16_t> MH((size_t)MDE_L_LAYERS * MDE_D * MDE_K);
    std::vector<int16_t> MP((size_t)MDE_L_LAYERS * MDE_D * MDE_D);
    std::vector<int16_t> X ((size_t)MDE_L_SEQ * MDE_D);
    std::vector<int32_t> grad_pooled(MDE_D);

    for (size_t i = 0; i < MH.size(); i++) MH[i] = (int16_t)(rnd16() % 17 - 8);
    for (size_t i = 0; i < MP.size(); i++) MP[i] = (int16_t)(rnd16() % 17 - 8);
    for (size_t i = 0; i < X.size();  i++) X[i]  = (int16_t)(rnd16() % 65 - 32);
    for (int d = 0; d < MDE_D; d++) grad_pooled[d] = (int32_t)(rnd16() % 21 - 10);

    RefState ref_st;
    ref_forward(MH.data(), MP.data(), X, ref_st);
    std::vector<int32_t> ref_gMH((size_t)MDE_L_LAYERS * MDE_D * MDE_K, 0);
    std::vector<int32_t> ref_gMP((size_t)MDE_L_LAYERS * MDE_D * MDE_D, 0);
    std::vector<int32_t> ref_gradX((size_t)MDE_L_SEQ * MDE_D, 0);
    ref_backward(ref_st, grad_pooled.data(), ref_gMH.data(), ref_gMP.data(),
                 ref_gradX.data());

    for (size_t i = 0; i < MH.size(); i++) buf[OFF_MHE + i] = (int32_t)(int16_t)MH[i];
    for (size_t i = 0; i < MP.size(); i++) buf[OFF_MPE + i] = (int32_t)(int16_t)MP[i];
    for (size_t i = 0; i < X.size();  i++) buf[OFF_XE + i]  = (int32_t)(int16_t)X[i];
    for (int i = 0; i < (int)WORDS_G_MHE; i++) buf[OFF_G_MHE + i] = 0;
    for (int i = 0; i < (int)WORDS_G_MPE; i++) buf[OFF_G_MPE + i] = 0;

    int rc = forge_train_step_morph_ecg(buf.data(), MDE_MODE_FWD, 0, 0, 0);
    if (rc != 0) { std::fprintf(stderr, "fwd rc=%d\n", rc); return 1; }

    int pool_mism = 0;
    for (int j = 0; j < MDE_D; j++) {
        int16_t kv = (int16_t)(buf[OFF_POOLED_E + j] & 0xFFFF);
        if (kv != ref_st.pooled[j]) {
            if (pool_mism < 8) std::fprintf(stderr, "POOLED[%d] kernel=%d ref=%d\n", j, kv, ref_st.pooled[j]);
            pool_mism++;
        }
    }
    // Verify saved indices match too (so the bwd comparison is meaningful).
    int idx_mism = 0;
    for (int l = 0; l < MDE_L_LAYERS; l++) {
        for (int t = 0; t < MDE_L_SEQ; t++) {
            int tb = off_tap_row(l, t), pb = off_proj_row(l, t);
            for (int c = 0; c < MDE_D; c++)
                if (buf[tb + c] != ref_st.tap(l, c, t)) idx_mism++;
            for (int j = 0; j < MDE_D; j++)
                if (buf[pb + j] != ref_st.proj(l, t, j)) idx_mism++;
        }
    }

    for (int d = 0; d < MDE_D; d++) buf[OFF_GRAD_POOLED_E + d] = grad_pooled[d];
    rc = forge_train_step_morph_ecg(buf.data(), MDE_MODE_BWD, 0, 0, 0);
    if (rc != 0) { std::fprintf(stderr, "bwd rc=%d\n", rc); return 1; }

    int gmh_mism = 0, gmp_mism = 0;
    for (int i = 0; i < (int)WORDS_G_MHE; i++)
        if (buf[OFF_G_MHE + i] != ref_gMH[i]) {
            if (gmh_mism < 8) std::fprintf(stderr, "g_MH[%d] kernel=%d ref=%d\n", i, buf[OFF_G_MHE + i], ref_gMH[i]);
            gmh_mism++;
        }
    for (int i = 0; i < (int)WORDS_G_MPE; i++)
        if (buf[OFF_G_MPE + i] != ref_gMP[i]) {
            if (gmp_mism < 8) std::fprintf(stderr, "g_MP[%d] kernel=%d ref=%d\n", i, buf[OFF_G_MPE + i], ref_gMP[i]);
            gmp_mism++;
        }

    // NEW: grad_X (gradient wrt the embedded input) — bit-exact vs reference.
    int gradx_mism = 0;
    for (int t = 0; t < MDE_L_SEQ; t++) {
        int rb = off_grad_x_row(t);
        for (int d = 0; d < MDE_D; d++) {
            int32_t kv = buf[rb + d];
            int32_t rv = ref_gradX[(size_t)t * MDE_D + d];
            if (kv != rv) {
                if (gradx_mism < 8)
                    std::fprintf(stderr, "grad_X[t=%d][d=%d] kernel=%d ref=%d\n", t, d, kv, rv);
                gradx_mism++;
            }
        }
    }

    std::printf("[tb_ecg] (a) bit-exact: POOLED mism=%d, idx mism=%d, g_MH mism=%d, g_MP mism=%d, grad_X mism=%d\n",
                pool_mism, idx_mism, gmh_mism, gmp_mism, gradx_mism);
    bool exact = (pool_mism == 0 && idx_mism == 0 && gmh_mism == 0 && gmp_mism == 0 && gradx_mism == 0);
    if (!exact) { std::fprintf(stderr, "[tb_ecg] FAIL — streaming kernel not bit-exact vs dense CPU reference\n"); return 1; }
    std::printf("[tb_ecg] (a) PASS — streaming fwd POOLED + bwd gradients (incl. grad_X) BIT-EXACT vs dense CPU reference\n");

#ifdef TB_LEARN
    // =====================================================================
    // PART (b): tiny learnable classification task (host head emulated).
    //   Same construction as forge_train_step_morph_tb.cpp part (b).
    // =====================================================================
    std::printf("\n[tb_ecg] (b) learnable toy classification task\n");

    const int NUM_CLASSES = 4;
    const int BLK = MDE_D / NUM_CLASSES;
    const int N_TRAIN = 96;
    const int N_STEPS = 120;
    const int LR = 1;
    const int TH = 0;
    const int HEAD_LR = 1;

    std::vector<std::vector<int16_t>> Xset(N_TRAIN, std::vector<int16_t>((size_t)MDE_L_SEQ * MDE_D));
    std::vector<int> yset(N_TRAIN);
    g_seed = 0x1234567u;
    for (int s = 0; s < N_TRAIN; s++) {
        int cls = s % NUM_CLASSES;
        yset[s] = cls;
        int lag = (cls * (MDE_L_SEQ / NUM_CLASSES));
        for (int t = 0; t < MDE_L_SEQ; t++)
            for (int d = 0; d < MDE_D; d++) {
                int16_t base = (int16_t)(rnd16() % 17 - 8);
                bool in_block  = (d >= cls * BLK && d < cls * BLK + BLK);
                bool in_window = (t >= lag && t < lag + (MDE_L_SEQ / NUM_CLASSES));
                if (in_block && in_window) base = (int16_t)(24 + (rnd16() % 7));
                Xset[s][(size_t)t * MDE_D + d] = base;
            }
    }

    std::vector<int32_t> tbuf(FT_BUFFER_WORDS_E, 0);
    g_seed = 0xBADC0DEu;
    for (int i = 0; i < (int)WORDS_MHE; i++) tbuf[OFF_MHE + i] = (int32_t)(int16_t)(rnd16() % 7 - 3);
    for (int i = 0; i < (int)WORDS_MPE; i++) tbuf[OFF_MPE + i] = (int32_t)(int16_t)(rnd16() % 7 - 3);

    std::vector<std::vector<int16_t>> Hh(NUM_CLASSES, std::vector<int16_t>(MDE_D));
    std::vector<std::vector<int32_t>> gHh(NUM_CLASSES, std::vector<int32_t>(MDE_D));
    for (int v = 0; v < NUM_CLASSES; v++)
        for (int d = 0; d < MDE_D; d++) Hh[v][d] = (int16_t)(rnd16() % 7 - 3);

    auto eval_accuracy = [&](void) -> double {
        int correct = 0;
        for (int s = 0; s < N_TRAIN; s++) {
            for (size_t i = 0; i < (size_t)MDE_L_SEQ * MDE_D; i++)
                tbuf[OFF_XE + i] = (int32_t)(int16_t)Xset[s][i];
            forge_train_step_morph_ecg(tbuf.data(), MDE_MODE_FWD, 0, 0, 0);
            std::vector<int16_t> pooled(MDE_D);
            for (int d = 0; d < MDE_D; d++) pooled[d] = (int16_t)(tbuf[OFF_POOLED_E + d] & 0xFFFF);
            int pred = 0; int32_t bestl = -2147483647;
            for (int v = 0; v < NUM_CLASSES; v++) {
                int32_t best = -2147483647;
                for (int d = 0; d < MDE_D; d++) {
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
    std::printf("[tb_ecg] (b) step   0  acc=%.3f (chance=%.3f)\n", acc0, 1.0 / NUM_CLASSES);

    double best_acc = acc0;
    for (int step = 0; step < N_STEPS; step++) {
        for (int v = 0; v < NUM_CLASSES; v++)
            for (int d = 0; d < MDE_D; d++) gHh[v][d] = 0;

        for (int s = 0; s < N_TRAIN; s++) {
            int cls = yset[s];
            for (size_t i = 0; i < (size_t)MDE_L_SEQ * MDE_D; i++)
                tbuf[OFF_XE + i] = (int32_t)(int16_t)Xset[s][i];
            forge_train_step_morph_ecg(tbuf.data(), MDE_MODE_FWD, 0, 0, 0);
            std::vector<int16_t> pooled(MDE_D);
            for (int d = 0; d < MDE_D; d++) pooled[d] = (int16_t)(tbuf[OFF_POOLED_E + d] & 0xFFFF);

            std::vector<int> dstar(NUM_CLASSES); std::vector<int32_t> logit(NUM_CLASSES);
            int pred = 0; int32_t bestl = -2147483647;
            for (int v = 0; v < NUM_CLASSES; v++) {
                int32_t best = -2147483647; int da = 0;
                for (int d = 0; d < MDE_D; d++) {
                    int32_t s2 = (int32_t)Hh[v][d] + (int32_t)pooled[d];
                    if (s2 > best) { best = s2; da = d; }
                }
                logit[v] = best; dstar[v] = da;
                if (best > bestl) { bestl = best; pred = v; }
            }

            std::vector<int32_t> glogit(NUM_CLASSES, 0);
            if (pred != cls) { glogit[pred] = +1; glogit[cls] = -1; }

            std::vector<int32_t> grad_pooled_s(MDE_D, 0);
            for (int v = 0; v < NUM_CLASSES; v++) {
                if (glogit[v] == 0) continue;
                int da = dstar[v];
                gHh[v][da]        += glogit[v];
                grad_pooled_s[da] += glogit[v];
            }

            for (int d = 0; d < MDE_D; d++) tbuf[OFF_GRAD_POOLED_E + d] = grad_pooled_s[d];
            forge_train_step_morph_ecg(tbuf.data(), MDE_MODE_BWD, 0, 0, 0);
        }

        forge_train_step_morph_ecg(tbuf.data(), MDE_MODE_SIGNSGD, LR, TH, 0);
        for (int v = 0; v < NUM_CLASSES; v++)
            for (int d = 0; d < MDE_D; d++) {
                int32_t g = gHh[v][d];
                int32_t stp = (g > 0) ? HEAD_LR : (g < 0 ? -HEAD_LR : 0);
                Hh[v][d] = r_sat16((int32_t)Hh[v][d] - stp);
            }

        double acc = eval_accuracy();
        if (acc > best_acc) best_acc = acc;
        if (step < 5 || step % 20 == 0 || step == N_STEPS - 1)
            std::printf("[tb_ecg] (b) step %3d  acc=%.3f  best=%.3f\n", step, acc, best_acc);
    }

    std::printf("[tb_ecg] (b) best accuracy %.3f over %d steps (chance=%.3f)\n",
                best_acc, N_STEPS, 1.0 / NUM_CLASSES);
    if (best_acc >= 0.60)
        std::printf("[tb_ecg] (b) PASS — toy task learns (acc >> chance)\n");
    else { std::fprintf(stderr, "[tb_ecg] (b) FAIL — toy task did not learn above chance\n"); return 1; }
#endif  // TB_LEARN

    std::printf("\n[tb_ecg] ALL CHECKS PASSED\n");
    return 0;
}
