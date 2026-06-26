// Standalone g++ convergence GATE for D=64 dense TRAIN_ALL (and a FREEZE sanity anchor).
// Drives the REAL kernel forge_train_step_s1 g++ #else path (int64) = bit-exact to the silicon fwd/bwd AND to
// the host-side optimizer step. So the CE trajectory here == what the F2 host-optimizer run will produce.
//   build: g++ -O2 -std=c++14 -I <forge_kernels> s1_converge_gate.cpp <forge_kernels>/forge_train_step_s1.cpp -o gate
//   run:   ./gate <N_STEPS> <densepct> <embmag> <regime 0=TRAIN_ALL 1=FREEZE>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include "forge_s1_config.hpp"
#include "forge_ce_lut.hpp"

extern "C" int forge_train_step_s1(int32_t*, int, int, int, int);

// ---- dense-nonzero TRAIN_ALL init (copied verbatim from forge_train_step_s1_tb.cpp stage_train_all) ----
static void stage_train_all(int32_t *buf, unsigned seed, int densepct, int embmag) {
    for (int i = 0; i < S1_BUFLEN; i++) buf[i] = 0;
    srand(seed);
    const long QC = (1L << S1_Q_CODE);
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
            buf[S1_OFF_WFP_LN1_B + l * S1_D + d] = 0;            buf[S1_OFF_WFP_LN2_B + l * S1_D + d] = 0;
        }
    for (int d = 0; d < S1_D; d++) { buf[S1_OFF_WFP_LNF_G + d] = (int32_t)QC; buf[S1_OFF_WFP_LNF_B + d] = 0; }
    for (int i = 0; i < S1_V * S1_D; i++) {
        buf[S1_OFF_WFP_EMBED  + i] = (rand() % (2 * embmag + 1)) - embmag;
        buf[S1_OFF_WFP_W_HEAD + i] = (rand() % (2 * embmag + 1)) - embmag;
    }
    for (int i = 0; i < S1_LUT_N; i++) { buf[S1_OFF_CE_LUT + i] = FT_CE_EXP2_Q16[i];
                                         buf[S1_OFF_ATTN_LUT + i] = FT_CE_EXP2_Q16[i]; }
    const int PAT = 8;
    for (int t = 0; t < S1_L_SEQ; t++) { int sym = t % PAT; buf[S1_OFF_TOKENS + t] = sym; buf[S1_OFF_TARGETS + t] = (sym + 1) % PAT; }
}

// ---- the FREEZE (residual-identity) memorize init, for the sanity anchor (== stage_opt_buffer(memorize=true)
//      embed/head only; dense block zero). Simplified: zero backbone + small embed/head + gamma=+QC. ----
static void stage_freeze(int32_t *buf, unsigned seed, int embmag) {
    for (int i = 0; i < S1_BUFLEN; i++) buf[i] = 0;
    srand(seed);
    const long QC = (1L << S1_Q_CODE);
    for (int l = 0; l < S1_N_LAYERS; l++)
        for (int d = 0; d < S1_D; d++) {
            buf[S1_OFF_WFP_LN1_G + l * S1_D + d] = (int32_t)QC; buf[S1_OFF_WFP_LN2_G + l * S1_D + d] = (int32_t)QC;
        }
    for (int d = 0; d < S1_D; d++) buf[S1_OFF_WFP_LNF_G + d] = (int32_t)QC;
    for (int i = 0; i < S1_V * S1_D; i++) {
        buf[S1_OFF_WFP_EMBED  + i] = (rand() % (2 * embmag + 1)) - embmag;
        buf[S1_OFF_WFP_W_HEAD + i] = (rand() % (2 * embmag + 1)) - embmag;
    }
    for (int i = 0; i < S1_LUT_N; i++) { buf[S1_OFF_CE_LUT + i] = FT_CE_EXP2_Q16[i];
                                         buf[S1_OFF_ATTN_LUT + i] = FT_CE_EXP2_Q16[i]; }
    const int PAT = 8;
    for (int t = 0; t < S1_L_SEQ; t++) { int sym = t % PAT; buf[S1_OFF_TOKENS + t] = sym; buf[S1_OFF_TARGETS + t] = (sym + 1) % PAT; }
}

static const double CE_TEMP = 0.01;
static double true_ce(const int32_t *bb) {
    double tot = 0.0;
    for (int t = 0; t < S1_L_SEQ; t++) {
        int tgt = bb[S1_OFF_TARGETS + t]; if (tgt < 0) tgt = 0; if (tgt >= S1_V) tgt = S1_V - 1;
        double mx = -1e300;
        for (int v = 0; v < S1_V; v++) { double lv = (int32_t)bb[S1_OFF_LOGITS + t * S1_V + v]; if (lv > mx) mx = lv; }
        double Z = 0.0;
        for (int v = 0; v < S1_V; v++) Z += exp(((double)(int32_t)bb[S1_OFF_LOGITS + t * S1_V + v] - mx) * CE_TEMP);
        double lt = ((double)(int32_t)bb[S1_OFF_LOGITS + t * S1_V + tgt] - mx) * CE_TEMP;
        tot += -(lt - log(Z));
    }
    return tot / S1_L_SEQ;
}
static int count_correct(const int32_t *bb) {
    int c = 0;
    for (int t = 0; t < S1_L_SEQ; t++) {
        int tgt = bb[S1_OFF_TARGETS + t]; int am = 0; long best = -(1L<<62);
        for (int v = 0; v < S1_V; v++) { long lv = (int32_t)bb[S1_OFF_LOGITS + t * S1_V + v]; if (lv > best) { best = lv; am = v; } }
        if (am == tgt) c++;
    }
    return c;
}
static int8_t snapc[ (S1_D*S1_D)*4 + (S1_FFN*S1_D)*2 + S1_D*2 + 16 ];
static const int CDST[] = { S1_OFF_WQ, S1_OFF_WK, S1_OFF_WV, S1_OFF_WO, S1_OFF_W_UP, S1_OFF_W_DOWN, S1_OFF_LN1_G, S1_OFF_LN2_G };
static int CN(int r){ int ns[]={S1_D*S1_D,S1_D*S1_D,S1_D*S1_D,S1_D*S1_D,S1_FFN*S1_D,S1_D*S1_FFN,S1_D,S1_D}; return ns[r]; }
static void snapshot_codes(const int32_t *bb){ int off=0; for(int r=0;r<8;r++){ for(int p=0;p<CN(r);p++) snapc[off+p]=(int8_t)(bb[CDST[r]+p]&0xFF); off+=CN(r);} }
static int count_code_changes(const int32_t *bb){ int off=0,ch=0; for(int r=0;r<8;r++){ for(int p=0;p<CN(r);p++){ int8_t c=(int8_t)(bb[CDST[r]+p]&0xFF); if(c!=snapc[off+p]) ch++; } off+=CN(r);} return ch; }
static long peakR(const int32_t *bb){ long mx=0; int base=s1_off_R(S1_N_LAYERS); for(int t=0;t<S1_L_SEQ;t++) for(int d=0;d<S1_D;d++){ long a=labs((long)(int32_t)bb[base+t*S1_D+d]); if(a>mx) mx=a; } return mx; }

int main(int argc, char** argv) {
    int N      = argc>1 ? atoi(argv[1]) : 120;
    int dense  = argc>2 ? atoi(argv[2]) : 30;
    int embmag = argc>3 ? atoi(argv[3]) : 16;
    int regime = argc>4 ? atoi(argv[4]) : S1_OPT_TRAIN_ALL;   // 0=TRAIN_ALL (gate), 1=FREEZE (anchor)
    const char* rn = regime==S1_OPT_TRAIN_ALL ? "TRAIN_ALL" : "FREEZE";
    int32_t *buf = (int32_t*)calloc(S1_BUFLEN, sizeof(int32_t));
    if (regime==S1_OPT_TRAIN_ALL) stage_train_all(buf, 4242, dense, embmag);
    else                          stage_freeze(buf, 4242, embmag);
    forge_train_step_s1(buf, 2, S1_OPT_INIT, regime, 0);   // populate live code/value from masters
    snapshot_codes(buf);
    forge_train_step_s1(buf, 0, 0, 0, 0);                  // initial forward
    double ce0 = true_ce(buf); int acc0 = count_correct(buf);
    double floor_ce = log((double)S1_V);
    printf("[gate] D=%d L=%d regime=%s dense=%d%% embmag=%d N=%d  (CE floor=%.4f)\n",
           S1_D, S1_N_LAYERS, rn, dense, embmag, N, floor_ce);
    printf("[gate] step   0: CE=%.4f acc=%d/%d peakR=%ld\n", ce0, acc0, S1_L_SEQ, peakR(buf));
    for (int s = 1; s <= N; s++) {
        forge_train_step_s1(buf, 0, 0, 0, 0);              // fwd
        forge_train_step_s1(buf, 1, 0, 0, 0);              // bwd
        forge_train_step_s1(buf, 2, S1_OPT_STEP, regime, 0); // opt (bit-exact to host step)
        if (s==1 || s%10==0) {
            forge_train_step_s1(buf, 0, 0, 0, 0);          // refresh logits on post-opt weights
            printf("[gate] step %3d: CE=%.4f acc=%d/%d codes_moved=%d peakR=%ld\n",
                   s, true_ce(buf), count_correct(buf), S1_L_SEQ, count_code_changes(buf), peakR(buf));
        }
    }
    forge_train_step_s1(buf, 0, 0, 0, 0);
    double ce1 = true_ce(buf); int acc1 = count_correct(buf);
    int ce_drops = (ce0-ce1) >= 0.5, not_collapsed = ce1 < (floor_ce-0.25), acc_improves = acc1 > acc0, peak_ok = peakR(buf) < 32767;
    printf("[gate] ============================================================\n");
    printf("[gate] %s: CE %.4f -> %.4f (drop %.4f)  acc %d->%d/%d  codes_moved=%d  peakR=%ld\n",
           rn, ce0, ce1, ce0-ce1, acc0, acc1, S1_L_SEQ, count_code_changes(buf), peakR(buf));
    printf("[gate] gates: CE-drop>=0.5:%s below-floor:%s acc-improves:%s no-sat:%s -> %s\n",
           ce_drops?"yes":"NO", not_collapsed?"yes":"NO", acc_improves?"yes":"NO", peak_ok?"yes":"NO",
           (ce_drops&&not_collapsed&&acc_improves&&peak_ok)?"CONVERGES":"NO-CONVERGENCE");
    free(buf);
    return 0;
}
