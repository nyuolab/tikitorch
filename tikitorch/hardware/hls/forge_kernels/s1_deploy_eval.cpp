// s1_deploy_eval.cpp — g++ FORWARD-deploy harness for the trained tropical LLM on the s1 kernel.
//
// Runs the REAL forge_train_step_s1 mode-0 forward (do_fwd_loss) — compiled with the deploy defines
//   -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1 -DS1_N_LAYERS=2
// — over held-out tinyshakespeare val windows, and measures the TRUE held-out bits/char from the kernel's
// saved S1_OFF_LOGITS (mean -log2 softmax(logit)[target]). It loads the int32 monolithic buffer produced by
// scripts/tropical_llm_export_s1.py (np.save .npy, little-endian int32), writes TOKENS+TARGETS per window
// (the export leaves them zero), kicks the kernel, reads the V logits per position, and accumulates bpc.
//
// The kernel forward is the F2 deploy path (csynth-confirmed DSP=0, 2.920ns, one SLR). This harness does
// NOT reimplement it (the constraint): it #includes forge_train_step_s1.cpp and calls the extern "C" entry.
//
// Build:
//   g++ -O2 -std=c++14 -I<repo> -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1 \
//       -DS1_N_LAYERS=2 s1_deploy_eval.cpp -o s1_deploy_eval
// Run:
//   ./s1_deploy_eval <buf.npy> <val_ids.bin (int32)> [nwin] [logit_temp]
// where val_ids.bin is the int32 array of the held-out 10% char ids (see the python driver).
//
// Prints: per-config deployable bpc + the bigram baseline (3.47) + the torch HARD reference (2.40/2.55).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "forge_s1_config.hpp"

// the REAL kernel forward (do_fwd_loss is mode 0; FWD_ONLY gates the training modes out).
#include "forge_train_step_s1.cpp"

// --- minimal .npy (int32, 1-D, little-endian, C order) loader ---
static std::vector<int32_t> load_npy_i32(const char *path, long &n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    unsigned char magic[6]; size_t r = fread(magic, 1, 6, f);
    if (r != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) { fprintf(stderr, "not a npy: %s\n", path); exit(2); }
    unsigned char ver[2]; r = fread(ver, 1, 2, f); (void)r;
    uint16_t hlen; r = fread(&hlen, 2, 1, f); (void)r;       // v1.0 header length (uint16 LE)
    std::string hdr(hlen, '\0'); r = fread(&hdr[0], 1, hlen, f); (void)r;
    // expect '<i4' dtype; find shape "(N,)"
    size_t sp = hdr.find("'shape':");
    long n = 0;
    if (sp != std::string::npos) {
        size_t lp = hdr.find('(', sp);
        sscanf(hdr.c_str() + lp + 1, "%ld", &n);
    }
    if (hdr.find("<i4") == std::string::npos)
        fprintf(stderr, "[warn] npy dtype not <i4 in header: %s\n", hdr.c_str());
    std::vector<int32_t> v(n);
    r = fread(v.data(), sizeof(int32_t), n, f);
    if ((long)r != n) { fprintf(stderr, "short read %s: %zu != %ld\n", path, r, n); exit(2); }
    fclose(f);
    n_out = n;
    return v;
}

// raw int32 binary loader (val ids)
static std::vector<int32_t> load_bin_i32(const char *path, long &n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    long n = bytes / 4;
    std::vector<int32_t> v(n);
    size_t r = fread(v.data(), 4, n, f); (void)r;
    fclose(f);
    n_out = n;
    return v;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <buf.npy> <val_ids.bin> [nwin=63] [logit_temp=512]\n", argv[0]);
        return 1;
    }
    const char *bufpath = argv[1];
    const char *valpath = argv[2];
    int nwin = (argc > 3) ? atoi(argv[3]) : 63;
    double logit_temp = (argc > 4) ? atof(argv[4]) : 512.0;
    int VREAL = (argc > 5) ? atoi(argv[5]) : 39;

    long blen = 0; std::vector<int32_t> buf0 = load_npy_i32(bufpath, blen);
    if (blen != S1_BUFLEN) {
        fprintf(stderr, "[FATAL] buffer length %ld != S1_BUFLEN %d (rebuild export at matching defines)\n",
                blen, S1_BUFLEN);
        return 3;
    }
    long vn = 0; std::vector<int32_t> val = load_bin_i32(valpath, vn);

    // work buffer (the kernel writes activation saves + logits in place)
    std::vector<int32_t> buf(S1_BUFLEN);

    double tot_bits = 0.0; long cnt = 0;
    int used = 0;
    for (int w = 0; w < nwin; w++) {
        long s = (long)w * S1_L_SEQ;
        if (s + S1_L_SEQ + 1 > vn) break;
        // fresh copy of the staged weights each window (kernel overwrites activation-save regions)
        memcpy(buf.data(), buf0.data(), sizeof(int32_t) * S1_BUFLEN);
        // write TOKENS + TARGETS for this window
        for (int t = 0; t < S1_L_SEQ; t++) {
            buf[S1_OFF_TOKENS + t]  = val[s + t];
            buf[S1_OFF_TARGETS + t] = val[s + t + 1];
        }
        // RUN THE REAL KERNEL FORWARD (mode 0).
        forge_train_step_s1(buf.data(), 0, 0, 0, 0);

        // read logits [L_SEQ][V] and accumulate true held-out bpc over the REAL vocab columns only.
        for (int t = 0; t < S1_L_SEQ; t++) {
            const int32_t *lg = &buf[S1_OFF_LOGITS + t * S1_V];
            int tgt = val[s + t + 1];
            // softmax over the real vocab [0,VREAL) at temperature logit_temp
            double mx = -1e30;
            for (int v = 0; v < VREAL; v++) { double z = lg[v] / logit_temp; if (z > mx) mx = z; }
            double sum = 0.0;
            for (int v = 0; v < VREAL; v++) sum += exp(lg[v] / logit_temp - mx);
            double pt = exp(lg[tgt] / logit_temp - mx) / sum;
            if (pt < 1e-30) pt = 1e-30;
            tot_bits += -log2(pt);
            cnt++;
        }
        used = w + 1;
    }
    double bpc = tot_bits / (double)cnt;
    printf("[s1_deploy_eval] windows=%d positions=%ld logit_temp=%.0f VREAL=%d\n", used, cnt, logit_temp, VREAL);
    printf("[s1_deploy_eval] DEPLOYABLE bpc = %.4f   | bigram = 3.47 | torch HARD ref = 2.40-2.55\n", bpc);
    printf("[s1_deploy_eval] beats_bigram = %s\n", (bpc < 3.47) ? "YES" : "NO");
    return 0;
}
