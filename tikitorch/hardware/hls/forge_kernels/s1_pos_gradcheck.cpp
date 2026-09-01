// s1_pos_gradcheck.cpp — FINITE-DIFFERENCE + WIRING gradient check for the S1 POSITIONAL EMBEDDING
// (S1_POS_EMBED=1). Drives the REAL kernel forge_train_step_s1 (g++ #else int64 path == F2 silicon).
//
// WHAT grad_POS IS: the forward adds POS[t*D+d] into the layer-0 residual input R[t][d] (a pure int add).
// So dLoss/dPOS[t][d] = dLoss/dR[t][d] = gR[t][d] EXACTLY — the SAME residual-input gradient the embed bwd
// scatters into the token rows. The kernel writes grad_POS[t][d] = gR[t][d] (a direct copy; POS rows are
// unique per (t,d), no accumulation). This is UNQUANTIZED, so it must match tightly.
//
// CHECK 1 (WIRING, BIT-EXACT — the load-bearing gate): grad_POS (S1_OFF_G_POS) == gR (S1_OFF_GR) for every
//   (t,d). This is the definition of grad_POS and directly verifies the backward write is correct.
//
// CHECK 2 (FINITE DIFFERENCE vs the kernel's OWN CE): perturb POS[t][d] by +/-h, run mode 0, reconstruct the
//   kernel's EXACT exp2-LUT cross-entropy loss (sum_t -ln p_model[tgt], nats — the loss the bwd is the adjoint
//   of), central-difference it, and compare to grad_POS. The forward is INTEGER with an exp2 LUT (the softmax
//   is piecewise-constant in the logits) and int16 saturation, so the FD has staircase noise; the honest gate
//   for a quantized/LUT pipeline is DIRECTION (sign agreement on the meaningfully-large elements + cosine), as
//   used by s1_head_gradcheck.cpp. A large-h average over a couple of step sizes smooths the LUT staircase.
//
// build: g++ -O2 -std=c++14 -I <forge_kernels> -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1 s1_pos_gradcheck.cpp \
//        <forge_kernels>/forge_train_step_s1.cpp -o /tmp/s1posgc
// run:   /tmp/s1posgc [seed]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include "forge_s1_config.hpp"
#include "forge_ce_lut.hpp"

extern "C" int forge_train_step_s1(int32_t*, int, int, int, int);

#if !S1_POS_EMBED
int main() { printf("[POSGC] built with S1_POS_EMBED=0 — nothing to check (POS disabled).\n"); return 0; }
#else

// the kernel's EXACT exp2-LUT cross-entropy at one position (== s1_charlm kernel_ce_bpc, but in NATS so it
// matches the grad scale). Returns -ln p_model[tgt].
static double kernel_ce_nats(const int32_t* logits_row /*[V]*/, int tgt, const int32_t* ce_lut) {
    int pred = 0; long mx = -(1L << 60);
    for (int v = 0; v < S1_V; v++) { long l = logits_row[v]; if (l > mx) { mx = l; pred = v; } }
    long e[S1_V]; long S = 0;
    for (int v = 0; v < S1_V; v++) {
        long zneg = (long)logits_row[pred] - (long)logits_row[v];
        long idx = zneg >> S1_CE_ZSHIFT; if (idx > 255) idx = 255; if (idx < 0) idx = 0;
        e[v] = ce_lut[idx]; S += e[v];
    }
    int sh = 0; for (int b = 0; b < 8; b++) { if ((S >> sh) > 65536) sh++; }
    long psum = 0, ptgt = 0;
    for (int v = 0; v < S1_V; v++) { long pq = e[v] >> sh; psum += pq; if (v == tgt) ptgt = pq; }
    if (psum <= 0) return log((double)S1_V);
    double p = (double)ptgt / (double)psum; if (p < 1e-12) p = 1e-12;
    return -log(p);
}

// total kernel CE over the whole window (nats), from the current buffer's LOGITS + TARGETS.
static double kernel_total_ce(const int32_t* buf) {
    const int32_t* ce_lut = buf + S1_OFF_CE_LUT;
    double tot = 0.0;
    for (int t = 0; t < S1_L_SEQ; t++) {
        int tgt = buf[S1_OFF_TARGETS + t]; if (tgt < 0) tgt = 0; if (tgt >= S1_V) tgt = S1_V - 1;
        tot += kernel_ce_nats(buf + S1_OFF_LOGITS + t * S1_V, tgt, ce_lut);
    }
    return tot;
}

int main(int argc, char** argv) {
    unsigned seed = (argc > 1) ? (unsigned)atoi(argv[1]) : 1234u;
    srand(seed);

    int32_t* buf = (int32_t*)calloc(S1_BUFLEN, sizeof(int32_t));

    // ---- stage a small, NON-DEGENERATE random instance (so grads are meaningfully nonzero everywhere) ----
    const long QC = (1L << S1_Q_CODE);
    // BACKBONE density (% nonzero dyadic codes). Set via env POSGC_DENSE (default 40). =0 -> PASSTHROUGH
    // backbone (attention/FFN contribute 0, R passes through), which makes gR a CLEAN gradient (only the
    // dyadic HEAD quantizes), isolating the POS-grad direction from the dense-backbone quantization noise.
    int densepct = 40; { const char* e = getenv("POSGC_DENSE"); if (e) densepct = atoi(e); }
    // dyadic code backbone (so attention/FFN carry real signal -> gR flows to POS through the block).
    auto fc = [&](int base, int O, int I) {
        for (int l = 0; l < S1_N_LAYERS; l++)
            for (int k = 0; k < O * I; k++) {
                int nz = (rand() % 100) < densepct; int sg = (rand() & 1) ? -1 : 1;
                buf[base + l * (O * I) + k] = (int32_t)(nz ? (long)sg * QC : 0);
            }
    };
    fc(S1_OFF_WFP_WQ, S1_D, S1_D); fc(S1_OFF_WFP_WK, S1_D, S1_D); fc(S1_OFF_WFP_WV, S1_D, S1_D);
    fc(S1_OFF_WFP_WO, S1_D, S1_D); fc(S1_OFF_WFP_W_UP, S1_FFN, S1_D); fc(S1_OFF_WFP_W_DOWN, S1_D, S1_FFN);
    for (int l = 0; l < S1_N_LAYERS; l++)
        for (int d = 0; d < S1_D; d++) {
            buf[S1_OFF_WFP_LN1_G + l * S1_D + d] = (int32_t)QC; buf[S1_OFF_WFP_LN2_G + l * S1_D + d] = (int32_t)QC;
        }
    for (int d = 0; d < S1_D; d++) buf[S1_OFF_WFP_LNF_G + d] = (int32_t)QC;
    for (int i = 0; i < S1_V * S1_D; i++) {
        buf[S1_OFF_WFP_EMBED + i] = (rand() % 33) - 16;
        int nz = (rand() % 100) < 40; int sg = (rand() & 1) ? -1 : 1;
        buf[S1_OFF_WFP_W_HEAD + i] = (int32_t)(nz ? (long)sg * QC : 0);   // HEAD_MODE=1: code master
    }
    // POS master: a NONZERO random positional table (so the FD operating point is generic, not the zero init).
    for (int i = 0; i < S1_POS_N; i++) buf[S1_OFF_WFP_POS + i] = (rand() % 41) - 20;   // |POS| <= 20
    for (int i = 0; i < S1_LUT_N; i++) { buf[S1_OFF_CE_LUT + i] = FT_CE_EXP2_Q16[i];
                                         buf[S1_OFF_ATTN_LUT + i] = FT_CE_EXP2_Q16[i]; }
    // a real token/target window (varied ids so the CE is non-trivial).
    for (int t = 0; t < S1_L_SEQ; t++) {
        buf[S1_OFF_TOKENS  + t] = rand() % S1_V;
        buf[S1_OFF_TARGETS + t] = rand() % S1_V;
    }
    // populate the live codes/values from the staged masters (init re-quantize).
    forge_train_step_s1(buf, 2, S1_OPT_INIT, S1_OPT_TRAIN_ALL, 0);

    // ---- analytic grad: ONE fwd + ONE bwd ----
    forge_train_step_s1(buf, 0, 0, 0, 0);   // forward (fills LOGITS + saves)
    forge_train_step_s1(buf, 1, 0, 0, 0);   // backward (fills g_* incl G_POS, GR)

    std::vector<int32_t> gPOS(S1_POS_N), gR(S1_L_SEQ * S1_D);
    for (int i = 0; i < S1_POS_N; i++)        gPOS[i] = buf[S1_OFF_G_POS + i];
    for (int i = 0; i < S1_L_SEQ * S1_D; i++) gR[i]   = buf[S1_OFF_GR   + i];

    // ===================== CHECK 1: WIRING (grad_POS == gR, bit-exact) =====================
    long wiring_mismatch = 0; long worst_wire = 0;
    for (int i = 0; i < S1_POS_N; i++) {
        long d = (long)gPOS[i] - (long)gR[i];
        if (d != 0) { wiring_mismatch++; if (labs(d) > worst_wire) worst_wire = labs(d); }
    }

    // ===================== CHECK 2: FINITE DIFFERENCE vs the kernel CE =====================
    // Perturb a sample of POS[t][d] by +/-h, recompute the kernel CE (nats), central-difference. The forward
    // is integer + exp2-LUT (piecewise-constant softmax) + int16 sat, so FD has staircase noise: we average a
    // couple of step sizes and gate on DIRECTION (sign + cosine on the meaningfully-large elements).
    // Restore POS to the analytic operating point before each perturb.
    auto set_pos = [&](int idx, int val) {
        buf[S1_OFF_POS + idx] = (int32_t)(int16_t)val;   // the forward reads POS as int16 value
    };
    auto get_pos0 = [&](int idx) { return (int)(int16_t)buf[S1_OFF_POS + idx]; };

    // sample positions across (t,d) for the FD sweep (full L*D is L_SEQ*D forwards*2*steps — keep it bounded).
    std::vector<int> sample;
    for (int t = 0; t < S1_L_SEQ; t++) for (int d = 0; d < S1_D; d++) {
        if ((rand() % 6) == 0) sample.push_back(t * S1_D + d);   // ~1/6 of elements
    }
    const int hs[] = { 8, 16, 32 };   // step sizes (large enough to cross LUT staircase; averaged)
    long sign_total = 0, sign_agree = 0;
    double dot = 0.0, na = 0.0, nb = 0.0;       // cosine(analytic, FD)
    double wsign_num = 0.0, wsign_den = 0.0;    // magnitude-weighted sign agreement
    double max_abs_g = 0.0;
    for (int i : sample) { double a = fabs((double)gPOS[i]); if (a > max_abs_g) max_abs_g = a; }
    double meaningful = max_abs_g * 0.02 + 1.0;  // ignore tiny-grad elements (LUT noise dominates them)

    for (int idx : sample) {
        int v0 = get_pos0(idx);
        // average central-difference over the step sizes
        double fd_acc = 0.0; int fd_cnt = 0;
        for (int s = 0; s < (int)(sizeof(hs)/sizeof(hs[0])); s++) {
            int h = hs[s];
            set_pos(idx, v0 + h); forge_train_step_s1(buf, 0, 0, 0, 0); double Lp = kernel_total_ce(buf);
            set_pos(idx, v0 - h); forge_train_step_s1(buf, 0, 0, 0, 0); double Lm = kernel_total_ce(buf);
            fd_acc += (Lp - Lm) / (2.0 * (double)h); fd_cnt++;
        }
        set_pos(idx, v0);   // restore
        double fd = fd_acc / fd_cnt;
        // the kernel grad is the CE-grad at the CE's INTERNAL Q-scale (Q16 logit grad); the FD-in-nats and the
        // kernel grad differ by a CONSTANT positive scale, so DIRECTION (sign/cosine) is the invariant.
        double an = (double)gPOS[idx];
        if (fabs(an) >= meaningful) {
            sign_total++;
            bool agree = (fd > 0) == (an > 0) || fd == 0.0;
            if ((fd > 0) == (an > 0)) sign_agree++;
            wsign_den += fabs(an);
            wsign_num += (((fd > 0) == (an > 0)) ? fabs(an) : -fabs(an));
        }
        dot += an * fd; na += an * an; nb += fd * fd;
    }
    double cosine = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    double sign_frac = sign_total ? (double)sign_agree / (double)sign_total : 0.0;
    double wsign_frac = wsign_den > 0 ? (wsign_num / wsign_den + 1.0) / 2.0 : 0.0;

    // ===================== CONTROL: same FD on EMBED (its bwd is silicon-validated; it shares gR) ==========
    // grad_EMBED[tok[t]*D+d] = sum_{t': tok[t']==tok[t]} gR[t'][d] — the SAME residual gradient gR that POS
    // uses, routed by token instead of position. Its backward is the bit-exact-on-F2 scatter. Running the
    // IDENTICAL FD-of-CE on EMBED gives the FD method's OWN ceiling under the kernel's LUT/quantization noise:
    // if EMBED's FD cosine ~ POS's FD cosine, the imperfect FD correlation is the KERNEL's quantization, not a
    // POS bug (POS and EMBED carry the same gR). We FD a per-(token,channel) embed element and compare to the
    // kernel's grad_EMBED. (Only meaningful when some token repeats — with random ids most rows are unique, so
    // grad_EMBED row == gR row for that single position; identical gradient, identical FD ceiling.)
    std::vector<int32_t> gEMB(S1_V * S1_D);
    for (int i = 0; i < S1_V * S1_D; i++) gEMB[i] = buf[S1_OFF_G_EMBED + i];
    auto set_emb = [&](int idx, int val) { buf[S1_OFF_EMBED + idx] = (int32_t)(int16_t)val; };
    auto get_emb0 = [&](int idx) { return (int)(int16_t)buf[S1_OFF_EMBED + idx]; };
    double e_dot = 0, e_na = 0, e_nb = 0; long e_sa = 0, e_st = 0;
    double e_max = 0; for (int i = 0; i < S1_V * S1_D; i++) { double a = fabs((double)gEMB[i]); if (a > e_max) e_max = a; }
    double e_mean = e_max * 0.02 + 1.0;
    // sample embed elements that ARE used by at least one token in the window (else grad is 0).
    std::vector<char> tok_used(S1_V, 0);
    for (int t = 0; t < S1_L_SEQ; t++) { int tk = buf[S1_OFF_TOKENS + t]; if (tk >= 0 && tk < S1_V) tok_used[tk] = 1; }
    for (int v = 0; v < S1_V; v++) { if (!tok_used[v]) continue;
        for (int d = 0; d < S1_D; d++) { if ((rand() % 6)) continue;
            int idx = v * S1_D + d; int v0 = get_emb0(idx);
            double fd_acc = 0; int fc2 = 0;
            for (int s = 0; s < 3; s++) { int h = hs[s];
                set_emb(idx, v0 + h); forge_train_step_s1(buf, 0, 0, 0, 0); double Lp = kernel_total_ce(buf);
                set_emb(idx, v0 - h); forge_train_step_s1(buf, 0, 0, 0, 0); double Lm = kernel_total_ce(buf);
                fd_acc += (Lp - Lm) / (2.0 * (double)h); fc2++;
            }
            set_emb(idx, v0);
            double fd = fd_acc / fc2; double an = (double)gEMB[idx];
            if (fabs(an) >= e_mean) { e_st++; if ((fd > 0) == (an > 0)) e_sa++; }
            e_dot += an * fd; e_na += an * an; e_nb += fd * fd;
        }
    }
    double e_cos = (e_na > 0 && e_nb > 0) ? e_dot / (sqrt(e_na) * sqrt(e_nb)) : 0.0;
    double e_signfrac = e_st ? (double)e_sa / (double)e_st : 0.0;

    // ===================== REPORT + GATES =====================
    printf("[POSGC] S1 positional-embed gradient check (seed=%u  L=%d D=%d V=%d  POS_N=%d  backbone_dense=%d%%)\n",
           seed, S1_L_SEQ, S1_D, S1_V, S1_POS_N, densepct);
    printf("[POSGC] (1) WIRING grad_POS==gR : mismatches = %ld / %d  (worst |diff| = %ld)  -> %s\n",
           wiring_mismatch, S1_POS_N, worst_wire, wiring_mismatch == 0 ? "BIT-EXACT" : "MISMATCH");
    printf("[POSGC] (2) FD vs kernel CE  POS  : sign-agree(meaningful)=%ld/%ld=%.3f  mag-weighted=%.4f  cosine=%.4f\n",
           sign_agree, sign_total, sign_frac, wsign_frac, cosine);
    printf("[POSGC] (3) FD vs kernel CE EMBED : sign-agree(meaningful)=%ld/%ld=%.3f  cosine=%.4f   <-- CONTROL "
           "(silicon-validated bwd, shares gR)\n",
           e_sa, e_st, e_signfrac, e_cos);
    printf("[POSGC]     -> POS and EMBED carry the SAME residual gradient gR; matching FD ceilings show the\n");
    printf("[POSGC]        imperfect FD cosine is the kernel's LUT/quantization, not a POS-grad bug.\n");

    bool ok_wire = (wiring_mismatch == 0);
    // FD-DIRECTION gate: POS must track the kernel CE AS WELL AS the silicon-validated EMBED grad does (they
    // share gR). Gate = POS cosine within noise of the EMBED control AND a positive directional correlation.
    bool ok_fd   = (cosine >= e_cos - 0.05) && (cosine > 0.20) && (wsign_frac >= 0.60);
    bool pass = ok_wire && ok_fd;
    printf("[POSGC] gates: WIRING(bit-exact)=%s  FD-DIRECTION(POS>=EMBED control)=%s\n",
           ok_wire ? "PASS" : "FAIL", ok_fd ? "PASS" : "FAIL");
    printf("[POSGC] %s\n", pass ? "ALL PASS" : "FAIL");
    free(buf);
    return pass ? 0 : 1;
}
#endif
