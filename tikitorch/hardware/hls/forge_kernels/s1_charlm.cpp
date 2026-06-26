// s1_charlm.cpp — REAL char-LM trainer/evaluator for the S1 tropical (multiplier-free dyadic) transformer.
//
// Drives the REAL kernel forge_train_step_s1 (the g++ #else int64 path = BIT-EXACT to the F2 silicon fwd/bwd
// AND to the F2 host-side optimizer). So the held-out bits/char measured here PREDICTS the silicon bits/char.
// We do NOT reimplement the forward/backward/optimizer — every fwd/bwd/opt is a real kernel kick.
//
// Extends s1_converge_gate.cpp from the toy 64-token successor-MEMORIZATION task into a real char-LM:
//   - DATA: tinyshakespeare, lowercased -> 39 unique chars (fits V=64); 90% train / 10% val by char index.
//   - TRAIN: batch-1 SGD over random L_SEQ windows (TOKENS=ids[s..], TARGETS=ids[s+1..]); mode 0/1/2 per window.
//     Optimizer state (m/v/W_fp) PERSISTS in the buffer across windows (init only ONCE, before the loop).
//   - EVAL: a fixed deterministic set of VAL windows spanning the val split, mode 0 only, TRUE held-out
//     cross-entropy in BITS (log2) averaged over all val positions: bpc = mean_t -log2( softmax(CE_TEMP*logit)[tgt_t] ).
//   - BASELINES: train-split unigram entropy (beat = "learned something") and bigram conditional entropy
//     (beat = "uses context").
//
// THE OPEN QUESTION this harness answers: at D=64 / 1-layer, does TRAIN_ALL (the dyadic backbone trains ->
// attention can supply CONTEXT) beat FREEZE (only embed+head train -> a BIGRAM-class model, since the
// residual-identity backbone gives no context) on a REAL LM, measured by held-out bits/char?
//
// build: g++ -O2 -std=c++14 -I <forge_kernels> s1_charlm.cpp <forge_kernels>/forge_train_step_s1.cpp -o /tmp/s1charlm
// run:   /tmp/s1charlm <N_windows> <regime 0=TRAIN_ALL 1=FREEZE> <eval_every> <seed>
//
// ---------------------------------------------------------------------------------------------------------
// HOST-OPTIMIZER EXTENSION (the optimizer-vs-capacity fork). On F2 the optimizer runs HOST-SIDE (the
// on-fabric do_opt has a write wall), so we can swap a BETTER host optimizer WITHOUT a kernel/AFI rebuild.
// The kernel only provides forward (logits) + backward (grads in g_*). We add two host optimizers:
//
//   m1  MULT-FREE dyadic-Adam + grad ACCUMULATION over a power-of-2 batch K. Run K windows of {set
//       tokens/targets, mode0, mode1}; after EACH bwd read back every g_* region and SUM into an int64
//       host accumulator; after K windows write avg = sum >> log2(K) into the g_* regions and kick ONE
//       mode-2 do_opt STEP. STILL fully multiplier-free (int adds + shifts). Opt state m/v persists.
//       (Hypothesis: averaging the per-window sign cleans the gradient -> stops the mode-collapse divergence.)
//
//   m2  FP-MASTER QAT-Adam (the deconfounding UPPER BOUND). Maintain a host double[] master for EVERY
//       trainable element (init from the SAME staged init the kernel uses). Each step: (1) QUANTIZE master
//       -> deploy: is_code -> the kernel's dyad() (== s1_encode_fine init path) int8 code, value -> rounded
//       int16; write into the deploy regions (+ the WFP master regions). (2) run K windows fwd+bwd,
//       accumulate g_* into a host float grad (avg over K). (3) REAL float Adam (b1=0.9,b2=0.999,eps=1e-8,
//       lr tunable) updates the master. The FORWARD/BACKWARD stay mult-free on the QUANTIZED deploy
//       weights; ONLY the host master+optimizer is FP. Standard QAT (straight-through) loop.
//
// THE FORK: if even the m2 FP-master upper bound cannot beat bigram (3.5162 bpc) at D=64/1-layer, the
// ceiling is the CAPACITY (needs S2, a new AFI). If it can, the existing AFI suffices and we hunt the best
// (ideally mult-free) host optimizer.
//
// CLI: /tmp/s1charlm --opt {kernel-pm1|mf-accum-m1|fpmaster-m2} --regime {0|1} --N <windows> --K <batch>
//      --lr <float> --eval_every <n> --seed <s>   (legacy positional form still works for kernel-pm1)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include "forge_s1_config.hpp"
#include "forge_ce_lut.hpp"

extern "C" int forge_train_step_s1(int32_t*, int, int, int, int);

// CE_TEMP: the SAME fixed monotone output scale the gate uses on the int logits. The logits are O(1e3-1e4);
// CE_TEMP is part of the model's output distribution scaling, so the held-out softmax (hence the bits/char)
// is computed over softmax(CE_TEMP*logit). Reported as-is; noted in the summary.
static const double CE_TEMP = 0.01;

// ---------------------------------------------------------------------------------------------------------
// DATA: tinyshakespeare lowercased -> sorted-unique-char vocab (V_actual<=39, fits V=64), 90/10 train/val.
// ---------------------------------------------------------------------------------------------------------
static std::vector<int> g_train_ids, g_val_ids;
static int g_vocab = 0;

static void load_data(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[charlm] cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string txt; txt.resize(n);
    if (fread(&txt[0], 1, n, f) != (size_t)n) { fprintf(stderr, "[charlm] short read\n"); exit(1); }
    fclose(f);
    // lowercase
    for (char& c : txt) c = (char)tolower((unsigned char)c);
    // sorted-unique-char vocab
    std::vector<unsigned char> uniq(256, 0);
    for (char c : txt) uniq[(unsigned char)c] = 1;
    std::vector<int> char2id(256, -1);
    int vid = 0;
    for (int c = 0; c < 256; c++) if (uniq[c]) char2id[c] = vid++;
    g_vocab = vid;
    if (g_vocab > S1_V) { fprintf(stderr, "[charlm] vocab %d > S1_V %d\n", g_vocab, S1_V); exit(1); }
    // encode
    std::vector<int> ids(txt.size());
    for (size_t i = 0; i < txt.size(); i++) ids[i] = char2id[(unsigned char)txt[i]];
    // 90/10 split by char index
    size_t split = (size_t)((double)ids.size() * 0.90);
    g_train_ids.assign(ids.begin(), ids.begin() + split);
    g_val_ids.assign(ids.begin() + split, ids.end());
    printf("[charlm] corpus=%zu chars  vocab=%d (lowercased, sorted-unique)  train=%zu  val=%zu\n",
           ids.size(), g_vocab, g_train_ids.size(), g_val_ids.size());
}

// ---------------------------------------------------------------------------------------------------------
// BASELINES from the TRAIN split (the bars to beat).
//   unigram entropy        = -sum_c p(c) log2 p(c)                     bits/char  (beat = learned something)
//   bigram conditional ent = -sum_{c0,c1} p(c0,c1) log2 p(c1|c0)       bits/char  (beat = uses context)
// ---------------------------------------------------------------------------------------------------------
static void baselines(double& uni_bpc, double& bi_bpc) {
    const std::vector<int>& ids = g_train_ids;
    std::vector<double> uc(g_vocab, 0.0);
    std::vector<double> bc((size_t)g_vocab * g_vocab, 0.0);
    for (size_t i = 0; i < ids.size(); i++) uc[ids[i]] += 1.0;
    for (size_t i = 0; i + 1 < ids.size(); i++) bc[(size_t)ids[i] * g_vocab + ids[i + 1]] += 1.0;
    double total = (double)ids.size();
    // unigram entropy
    uni_bpc = 0.0;
    for (int c = 0; c < g_vocab; c++) if (uc[c] > 0) { double p = uc[c] / total; uni_bpc += -p * log2(p); }
    // bigram conditional entropy H(c1|c0) = -sum p(c0,c1) log2 p(c1|c0)
    double pair_total = total - 1.0;  // number of bigram pairs
    bi_bpc = 0.0;
    for (int c0 = 0; c0 < g_vocab; c0++) {
        // p(c0) over the bigram-LEFT marginal == count of c0 as a left context
        double c0_left = 0.0;
        for (int c1 = 0; c1 < g_vocab; c1++) c0_left += bc[(size_t)c0 * g_vocab + c1];
        if (c0_left <= 0) continue;
        for (int c1 = 0; c1 < g_vocab; c1++) {
            double n = bc[(size_t)c0 * g_vocab + c1];
            if (n <= 0) continue;
            double p_joint = n / pair_total;           // p(c0,c1)
            double p_cond  = n / c0_left;              // p(c1|c0)
            bi_bpc += -p_joint * log2(p_cond);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// INIT (copied from s1_converge_gate.cpp stage_train_all / stage_freeze). These stage the WEIGHTS + LUTs +
// the toy TOKENS/TARGETS pattern; the toy pattern is irrelevant here — the train/eval loop OVERWRITES
// TOKENS/TARGETS per window. The init runs ONCE; do_opt(INIT) populates the live codes/values from masters.
// ---------------------------------------------------------------------------------------------------------
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
}
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
}

// ---------------------------------------------------------------------------------------------------------
// per-window staging: write TOKENS = ids[s..s+L-1], TARGETS = ids[s+1..s+L] into the buffer (overwrites).
// ---------------------------------------------------------------------------------------------------------
static inline void stage_window(int32_t *buf, const std::vector<int>& ids, size_t s) {
    for (int i = 0; i < S1_L_SEQ; i++) {
        buf[S1_OFF_TOKENS  + i] = ids[s + i];
        buf[S1_OFF_TARGETS + i] = ids[s + 1 + i];
    }
}

// train-CE of the CURRENT buffer window (nats; the gate metric), for progress reporting only.
static double train_ce_window(const int32_t *bb) {
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

// ---------------------------------------------------------------------------------------------------------
// HELD-OUT bits/char: deterministic VAL windows spanning the val split; mode 0 only (no train); TRUE
// cross-entropy in BITS averaged over ALL val positions.
//   p = softmax(CE_TEMP * logit)        (numerically stable: subtract rowmax)
//   bpc_t = -log2( p[tgt_t] )           ; bpc = mean over windows*positions.
// We also track top-1 accuracy purely as a sanity readout (NOT the metric).
// ---------------------------------------------------------------------------------------------------------
static double eval_val_bpc(int32_t *buf, int n_windows, double& out_acc) {
    const std::vector<int>& ids = g_val_ids;
    if ((int)ids.size() < S1_L_SEQ + 2) { out_acc = 0; return NAN; }
    // deterministic starts spanning the val split (avoid the very last possible start; need s+1+L<=size)
    size_t max_start = ids.size() - (size_t)S1_L_SEQ - 1;
    int W = n_windows;
    double bits_sum = 0.0; long pos = 0; long correct = 0;
    for (int w = 0; w < W; w++) {
        size_t s = (W > 1) ? (size_t)((double)w / (double)(W - 1) * (double)max_start) : 0;
        if (s > max_start) s = max_start;
        stage_window(buf, ids, s);
        forge_train_step_s1(buf, 0, 0, 0, 0);   // forward only
        for (int t = 0; t < S1_L_SEQ; t++) {
            int tgt = buf[S1_OFF_TARGETS + t]; if (tgt < 0) tgt = 0; if (tgt >= S1_V) tgt = S1_V - 1;
            double mx = -1e300; int am = 0;
            for (int v = 0; v < S1_V; v++) {
                double lv = (int32_t)buf[S1_OFF_LOGITS + t * S1_V + v];
                if (lv > mx) { mx = lv; am = v; }
            }
            double Z = 0.0;
            for (int v = 0; v < S1_V; v++)
                Z += exp(((double)(int32_t)buf[S1_OFF_LOGITS + t * S1_V + v] - mx) * CE_TEMP);
            double lt = ((double)(int32_t)buf[S1_OFF_LOGITS + t * S1_V + tgt] - mx) * CE_TEMP;
            double logp_nats = lt - log(Z);            // ln p[tgt]
            double bpc_t = -logp_nats / log(2.0);      // -log2 p[tgt]
            bits_sum += bpc_t; pos++;
            if (am == tgt) correct++;
        }
    }
    out_acc = pos ? (double)correct / (double)pos : 0.0;
    return pos ? bits_sum / (double)pos : NAN;
}

// =========================================================================================================
// HOST OPTIMIZERS — m1 (mult-free grad-accum) + m2 (FP-master QAT). Both drive the REAL kernel for fwd/bwd;
// neither reimplements fwd/bwd. They differ only in how the host turns the kernel's g_* grads into a weight
// step (m1 = the kernel's own mode-2 dyadic-Adam on AVERAGED grads; m2 = a host FP-Adam on an FP master).
// =========================================================================================================

enum { OPT_KERNEL_PM1 = 0, OPT_MF_ACCUM_M1 = 1, OPT_FPMASTER_M2 = 2 };

// ---- the master->deploy QUANTIZER, BIT-MATCHED to the kernel forward decode. ----
// The forward reads is_code regions via wdecode(code) -> weight = (nz ? ±2^exp : 0); value regions as int16.
// So a faithful deploy MUST be the kernel's own quantizer:
//   is_code : dyad(round(master_value)) -> pack {sign(bit7), nz(bit5), exp(bits4..0)}  == s1_encode_fine
//             on the INIT path (allow_clamp=false: pure dyad(round), no prev/exp-clamp history). This is
//             EXACTLY what the kernel's do_opt(INIT) writes and what the forward wdecode() reverses.
//   value   : round(master_value) clamped to int16.
// dyad() is reproduced VERBATIM from forge_train_step_s1.cpp (priority-encode MSB; bump exp if the bit
// below MSB is set; sign separate; zero-guard) so kernel-decode(host-encode(w)) == the same dyadic w.
static inline void host_dyad(int v, bool& sgn, int& e, bool& nz) {
    unsigned a = (v < 0) ? (unsigned)(-v) : (unsigned)v;
    a &= 0xFFFFu;                                  // dyad() operates on the int16 magnitude (a is uint16)
    sgn = (v < 0);
    nz  = (a != 0);
    int msb = 0;
    for (int b = 15; b >= 1; b--) { if ((a >> b) & 1u) { msb = b; break; } }
    if (msb >= 1 && ((a >> (msb - 1)) & 1u)) msb = msb + 1;   // bump on bit below MSB
    e = msb;
}
// int16 saturating round of an FP master value (matches sat16 on a rounded integer).
static inline int host_round_i16(double w) {
    double r = (w < 0) ? -floor(-w + 0.5) : floor(w + 0.5);   // round-half-away-from-zero (== s1_encode_fine)
    if (r >  32767.0) r =  32767.0;
    if (r < -32768.0) r = -32768.0;
    return (int)r;
}
// is_code deploy byte (== s1_encode of the rounded master; init path, no exp-clamp).
static inline int32_t host_encode_code(double w) {
    int wr = host_round_i16(w);
    bool sgn, nz; int e;
    host_dyad(wr, sgn, e, nz);
    if (e > 31) e = 31;
    int c = (e & 0x1F);
    if (nz)  c |= (1 << 5);
    if (sgn) c |= (1 << 7);
    return (int32_t)(int8_t)(unsigned char)c;
}
// the deploy VALUE that the kernel forward sees for an is_code weight (for the QAT straight-through grad:
// fwd/bwd run on THIS quantized value, so the master moves but the realised weight is the dyadic snap).
static inline double host_decode_code_value(int32_t code) {
    unsigned c = (unsigned)(code & 0xFF);
    bool sgn = ((c >> 7) & 1) != 0;
    bool nz  = ((c >> 5) & 1) != 0;
    int  e   = (int)(c & 0x1F);
    if (!nz) return 0.0;
    double v = (double)(1u << e);   // ±2^e  (e<=16 in practice; 1u<<e safe)
    return sgn ? -v : v;
}

// each trainable element, flattened: which region, its deploy offset, WFP offset, count, is_code.
// (Mirror of S1_OPT_REGIONS but split into per-element addressing on the host.)
struct host_region_t { int dst_off, wfp_off, g_off, n, is_code; const char* name; };
static host_region_t HR[S1_NUM_OPT_REGIONS];
static int HR_N = 0;
static void build_host_regions() {
    static const char* names[S1_NUM_OPT_REGIONS] = {
        "EMBED","WQ","WK","WV","WO","W_UP","W_DOWN","LN1_G","LN1_B","LN2_G","LN2_B","LNF_G","LNF_B","W_HEAD" };
    for (int r = 0; r < S1_NUM_OPT_REGIONS; r++) {
        const s1_opt_region_t& S = S1_OPT_REGIONS[r];
        HR[r] = host_region_t{ S.dst_off, S.wfp_off, S.g_off, S.n, S.is_code, names[r] };
    }
    HR_N = S1_NUM_OPT_REGIONS;
}
// is this region FROZEN under the given regime? (FREEZE_CODES freezes the is_code dense backbone.)
static inline bool region_frozen(const host_region_t& R, int regime) {
    return (regime == S1_OPT_FREEZE_CODES) && (R.is_code != 0);
}

// run K windows {stage random window, mode0, mode1}, accumulate g_* into gacc (host int64), SUM over K.
// max_start bounds the random window start in the train ids. Returns the running CE proxy (nats) summed.
static double accumulate_grads_K(int32_t* buf, std::vector<std::vector<int64_t>>& gacc, int K,
                                 size_t max_start, double& ce_sum, int& ce_cnt) {
    for (int r = 0; r < HR_N; r++) std::fill(gacc[r].begin(), gacc[r].end(), (int64_t)0);
    for (int k = 0; k < K; k++) {
        size_t s = (size_t)(((unsigned long long)rand() * (unsigned long long)rand()) % max_start);
        stage_window(buf, g_train_ids, s);
        forge_train_step_s1(buf, 0, 0, 0, 0);                 // fwd + CE (fills LOGITS)
        ce_sum += train_ce_window(buf); ce_cnt++;
        forge_train_step_s1(buf, 1, 0, 0, 0);                 // bwd (OVERWRITES g_* — zeros its own accums)
        for (int r = 0; r < HR_N; r++) {
            const host_region_t& R = HR[r];
            const int32_t* g = buf + R.g_off;
            int64_t* a = gacc[r].data();
            for (int i = 0; i < R.n; i++) a[i] += (int64_t)g[i];
        }
    }
    return 0.0;
}

// =========================================================================================================
int main(int argc, char** argv) {
    // --- CLI: support both the new flag form and the legacy positional form (for kernel-pm1 compat). ---
    int opt        = OPT_KERNEL_PM1;
    int N          = 2000;
    int regime     = S1_OPT_TRAIN_ALL;
    int K          = 16;
    // m2 default lr is O(1): the master is at UNIT int16 scale (EMBED/W_HEAD ~ O(16)), and the value-region
    // requantize ROUNDS to int16 each step, so a sub-unit Adam step is swallowed by the rounding -> the deploy
    // never moves (verified: lr=2e-3 is FLAT, lr~1-2 trains). Tune via --lr. (For OPT_KERNEL_PM1/m1 lr is
    // unused — the kernel's dyadic-Adam pow2 step is governed by S1_LR_SHIFT in the config, not this flag.)
    double lr      = 1.0;
    int eval_every = 500;
    unsigned seed  = 4242u;
    // legacy positional: <N> <regime> <eval_every> <seed>  (only if argv[1] is not a --flag)
    if (argc > 1 && argv[1][0] != '-') {
        N          = atoi(argv[1]);
        if (argc > 2) regime     = atoi(argv[2]);
        if (argc > 3) eval_every = atoi(argv[3]);
        if (argc > 4) seed       = (unsigned)atoi(argv[4]);
    } else {
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            auto nx = [&](const char* d){ return (i + 1 < argc) ? argv[++i] : d; };
            if      (a == "--opt") {
                std::string v = nx("kernel-pm1");
                if      (v == "kernel-pm1")  opt = OPT_KERNEL_PM1;
                else if (v == "mf-accum-m1") opt = OPT_MF_ACCUM_M1;
                else if (v == "fpmaster-m2") opt = OPT_FPMASTER_M2;
                else { fprintf(stderr, "[charlm] unknown --opt %s\n", v.c_str()); return 1; }
            }
            else if (a == "--regime")     regime     = atoi(nx("0"));
            else if (a == "--N")          N          = atoi(nx("2000"));
            else if (a == "--K")          K          = atoi(nx("16"));
            else if (a == "--lr")         lr         = atof(nx("0.002"));
            else if (a == "--eval_every") eval_every = atoi(nx("500"));
            else if (a == "--seed")       seed       = (unsigned)atoi(nx("4242"));
            else { fprintf(stderr, "[charlm] unknown flag %s\n", a.c_str()); return 1; }
        }
    }
    // K must be a power of two for the m1 averaging shift.
    int log2K = 0; { int kk = K; while (kk > 1) { kk >>= 1; log2K++; } if ((1 << log2K) != K) {
        fprintf(stderr, "[charlm] K=%d must be a power of two\n", K); return 1; } }
    const char* optn = (opt == OPT_KERNEL_PM1) ? "kernel-pm1"
                     : (opt == OPT_MF_ACCUM_M1) ? "mf-accum-m1" : "fpmaster-m2";

    const int N_VAL_WINDOWS = 64;
    const int dense = 30, embmag = 16;                              // TRAIN_ALL densepct/embmag (gate defaults)
    const char* rn = (regime == S1_OPT_TRAIN_ALL) ? "TRAIN_ALL" : "FREEZE";

    load_data("/Users/ekoermann/Documents/tikitorch/data/tinyshakespeare.txt");

    double uni_bpc = 0, bi_bpc = 0;
    baselines(uni_bpc, bi_bpc);
    printf("[charlm] BASELINES (train split): unigram=%.4f bits/char  bigram(cond)=%.4f bits/char\n",
           uni_bpc, bi_bpc);
    printf("[charlm]   -> beat unigram = learned something; beat bigram = uses context.\n");
    printf("[charlm] note: held-out bpc uses softmax(CE_TEMP*logit), CE_TEMP=%.3f (the model's output scale).\n",
           CE_TEMP);

    build_host_regions();

    int32_t *buf = (int32_t*)calloc(S1_BUFLEN, sizeof(int32_t));
    if (regime == S1_OPT_TRAIN_ALL) stage_train_all(buf, seed, dense, embmag);
    else                            stage_freeze(buf, seed, embmag);
    forge_train_step_s1(buf, 2, S1_OPT_INIT, regime, 0);   // populate live codes/values from staged masters (ONCE)

    printf("[charlm] D=%d L_SEQ=%d V=%d H=%d DH=%d FFN=%d  regime=%s  opt=%s K=%d lr=%.2e  N=%d eval_every=%d seed=%u\n",
           S1_D, S1_L_SEQ, S1_V, S1_H, S1_DH, S1_FFN, rn, optn, K, lr, N, eval_every, seed);

    // initial eval (random init)
    double acc0 = 0; double bpc0 = eval_val_bpc(buf, N_VAL_WINDOWS, acc0);
    printf("[charlm] eval @win     0: val_bpc=%.4f  (val top1=%.3f)\n", bpc0, acc0);

    srand(seed ^ 0x9e3779b9u);
    size_t max_train_start = g_train_ids.size() - (size_t)S1_L_SEQ - 1;
    double ce_run = 0.0; int ce_cnt = 0;

    // ---------------------------------------------------------------------------------------------------
    // host grad accumulators (one int64 vector per region) — shared by m1 and m2.
    // ---------------------------------------------------------------------------------------------------
    std::vector<std::vector<int64_t>> gacc(HR_N);
    for (int r = 0; r < HR_N; r++) gacc[r].assign(HR[r].n, 0);

    if (opt == OPT_KERNEL_PM1) {
        // ===== BASELINE: the existing kernel mode-2 do_opt (the +/-1 sign-step), batch-1. =====
        for (int w = 1; w <= N; w++) {
            size_t s = (size_t)(((unsigned long long)rand() * (unsigned long long)rand()) % max_train_start);
            stage_window(buf, g_train_ids, s);
            forge_train_step_s1(buf, 0, 0, 0, 0);                  // fwd + CE (fills LOGITS + saves)
            ce_run += train_ce_window(buf); ce_cnt++;
            forge_train_step_s1(buf, 1, 0, 0, 0);                  // bwd (fills grads)
            forge_train_step_s1(buf, 2, S1_OPT_STEP, regime, 0);   // opt STEP (bit-exact to F2 host optimizer)
            if (w % eval_every == 0 || w == N) {
                double acc = 0; double bpc = eval_val_bpc(buf, N_VAL_WINDOWS, acc);
                printf("[charlm] win %5d: train_CE(nats,last %d)=%.4f  val_bpc=%.4f  (val top1=%.3f)\n",
                       w, ce_cnt, ce_run / ce_cnt, bpc, acc);
                ce_run = 0.0; ce_cnt = 0;
            }
        }
    }
    else if (opt == OPT_MF_ACCUM_M1) {
        // ===== m1: MULT-FREE dyadic-Adam on grads AVERAGED over a power-of-2 batch K. =====
        // Per OPT-STEP: accumulate g_* over K windows (host int64 sum), write avg = sum>>log2K into g_*,
        // then ONE kernel mode-2 do_opt STEP (the kernel's own dyadic-Adam; m/v persist). Fully mult-free.
        int n_steps = N / K; if (n_steps < 1) n_steps = 1;
        int total_windows = 0;
        for (int step = 1; step <= n_steps; step++) {
            accumulate_grads_K(buf, gacc, K, max_train_start, ce_run, ce_cnt);
            total_windows += K;
            // write the AVERAGED grad back into the kernel g_* regions (avg = sum >> log2K, sign-preserving
            // arithmetic shift). Mult-free: int adds during accumulate + one arithmetic shift here.
            for (int r = 0; r < HR_N; r++) {
                int32_t* g = buf + HR[r].g_off;
                const int64_t* a = gacc[r].data();
                for (int i = 0; i < HR[r].n; i++) g[i] = (int32_t)(a[i] >> log2K);   // arithmetic >>
            }
            forge_train_step_s1(buf, 2, S1_OPT_STEP, regime, 0);   // ONE kernel dyadic-Adam step on the avg
            if (total_windows / eval_every != (total_windows - K) / eval_every || step == n_steps) {
                double acc = 0; double bpc = eval_val_bpc(buf, N_VAL_WINDOWS, acc);
                printf("[charlm] win %5d: train_CE(nats,last %d)=%.4f  val_bpc=%.4f  (val top1=%.3f)\n",
                       total_windows, ce_cnt, ce_cnt ? ce_run / ce_cnt : 0.0, bpc, acc);
                ce_run = 0.0; ce_cnt = 0;
            }
        }
    }
    else { // OPT_FPMASTER_M2
        // ===== m2: FP-MASTER QAT-Adam (the deconfounding UPPER BOUND). =====
        // Host double master per trainable element, init from the SAME staged init the kernel uses
        // (read the staged WFP region — value regions are int16, code regions are at scale 2^Q_CODE so the
        // master VALUE = WFP / 2^Q_CODE). Each step: (1) quantize master -> deploy (kernel-faithful), (2) K
        // windows fwd+bwd accumulate g_* -> float grad (avg over K), (3) real float Adam updates the master.
        const double QC = (double)(1L << S1_Q_CODE);
        std::vector<std::vector<double>> M(HR_N), Mm(HR_N), Mv(HR_N);  // master, Adam m, Adam v
        for (int r = 0; r < HR_N; r++) {
            M[r].assign(HR[r].n, 0.0); Mm[r].assign(HR[r].n, 0.0); Mv[r].assign(HR[r].n, 0.0);
            const int32_t* wfp = buf + HR[r].wfp_off;
            double sc = HR[r].is_code ? (1.0 / QC) : 1.0;             // code master is stored at scale 2^Q_CODE
            for (int i = 0; i < HR[r].n; i++) M[r][i] = (double)wfp[i] * sc;
        }
        const double B1 = 0.9, B2 = 0.999, EPS = 1e-8;
        int n_steps = N / K; if (n_steps < 1) n_steps = 1;
        long adam_t = 0; int total_windows = 0;
        for (int step = 1; step <= n_steps; step++) {
            // (1) QUANTIZE master -> deploy regions (+ WFP) using the kernel-faithful quantizer.
            for (int r = 0; r < HR_N; r++) {
                const host_region_t& R = HR[r];
                int32_t* dst = buf + R.dst_off;
                int32_t* wfp = buf + R.wfp_off;
                for (int i = 0; i < R.n; i++) {
                    if (R.is_code) {
                        dst[i] = host_encode_code(M[r][i]);          // dyad(round(master)) == fwd wdecode source
                        wfp[i] = (int32_t)host_round_i16(M[r][i] * QC);  // keep WFP at scale 2^Q_CODE (consistency)
                    } else {
                        int wr = host_round_i16(M[r][i]);
                        dst[i] = (int32_t)(int16_t)wr;               // int16 value the forward reads
                        wfp[i] = (int32_t)(int16_t)wr;
                    }
                }
            }
            // (2) accumulate g_* over K windows (the kernel fwd/bwd run on the QUANTIZED deploy weights).
            accumulate_grads_K(buf, gacc, K, max_train_start, ce_run, ce_cnt);
            total_windows += K;
            adam_t++;
            double bc1 = 1.0 - pow(B1, (double)adam_t);
            double bc2 = 1.0 - pow(B2, (double)adam_t);
            // (3) real float Adam on the master. grad = (sum over K) / K. For is_code regions the kernel grad
            //     is dL/d(deploy weight); the master is the SAME weight (straight-through QAT), scale 1.
            for (int r = 0; r < HR_N; r++) {
                const host_region_t& R = HR[r];
                if (region_frozen(R, regime)) continue;              // FREEZE_CODES: leave the dense backbone
                const int64_t* a = gacc[r].data();
                double invK = 1.0 / (double)K;
                for (int i = 0; i < R.n; i++) {
                    double g = (double)a[i] * invK;
                    Mm[r][i] = B1 * Mm[r][i] + (1.0 - B1) * g;
                    Mv[r][i] = B2 * Mv[r][i] + (1.0 - B2) * g * g;
                    double mh = Mm[r][i] / bc1;
                    double vh = Mv[r][i] / bc2;
                    M[r][i] -= lr * mh / (sqrt(vh) + EPS);
                }
            }
            if (total_windows / eval_every != (total_windows - K) / eval_every || step == n_steps) {
                // requantize before eval so the forward sees the latest master (eval runs mode0 on deploy).
                for (int r = 0; r < HR_N; r++) {
                    const host_region_t& R = HR[r];
                    int32_t* dst = buf + R.dst_off;
                    for (int i = 0; i < R.n; i++)
                        dst[i] = R.is_code ? host_encode_code(M[r][i]) : (int32_t)(int16_t)host_round_i16(M[r][i]);
                }
                double acc = 0; double bpc = eval_val_bpc(buf, N_VAL_WINDOWS, acc);
                printf("[charlm] win %5d: train_CE(nats,last %d)=%.4f  val_bpc=%.4f  (val top1=%.3f)\n",
                       total_windows, ce_cnt, ce_cnt ? ce_run / ce_cnt : 0.0, bpc, acc);
                ce_run = 0.0; ce_cnt = 0;
            }
        }
        // final requantize so the closing eval is on the trained master.
        for (int r = 0; r < HR_N; r++) {
            const host_region_t& R = HR[r];
            int32_t* dst = buf + R.dst_off;
            for (int i = 0; i < R.n; i++)
                dst[i] = R.is_code ? host_encode_code(M[r][i]) : (int32_t)(int16_t)host_round_i16(M[r][i]);
        }
    }

    // final held-out bpc
    double accF = 0; double bpcF = eval_val_bpc(buf, N_VAL_WINDOWS, accF);
    printf("[charlm] ============================================================\n");
    printf("[charlm] %s/%s FINAL: val_bpc=%.4f bits/char  (init %.4f -> %.4f)  val top1=%.3f\n",
           rn, optn, bpcF, bpc0, bpcF, accF);
    printf("[charlm] vs baselines: unigram=%.4f  bigram=%.4f  |  beat-unigram:%s  beat-bigram:%s\n",
           uni_bpc, bi_bpc, (bpcF < uni_bpc ? "yes" : "NO"), (bpcF < bi_bpc ? "yes" : "NO"));
    free(buf);
    return 0;
}
