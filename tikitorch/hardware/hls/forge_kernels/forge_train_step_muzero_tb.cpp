// forge_train_step_muzero_tb.cpp — host-side CSIM for the dyadic MuZero forward kernel.
//
// Functional verification under plain g++ (no Vitis): the kernel compiles via the
// forge_hls_types.hpp shim, runs the zero-DSP shift+add forward, and we check the outputs
// are well-formed (int16 range), deterministic, and that the all-zero-weight case yields a
// zero (bias-only) output. Bit-exactness vs the trained PyTorch net is the integration test
// (export the shift schedule into this buffer via tikitorch.fpga.dyadic_export — next stage).
//
// Compile: g++ -O2 -std=c++17 -DCSIM_ONLY -I.. -I. forge_train_step_muzero.cpp \
//              forge_train_step_muzero_tb.cpp -o tb_muzero && ./tb_muzero
#include "forge_buffer_muzero.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>

extern "C" int forge_train_step_muzero(int32_t* buffer, int mode, int arg1, int arg2, int arg3);

int main() {
    std::printf("[tb_muzero] MZ_BUFFER_WORDS=%d (%.1f KiB)  OBS=%d L=%d A=%d DR=%d DP=%d\n",
                MZ_BUFFER_WORDS, MZ_BUFFER_BYTES / 1024.0, MZ_OBS, MZ_L, MZ_A, MZ_DR, MZ_DP);
    std::vector<int32_t> buf(MZ_BUFFER_WORDS, 0);

    uint32_t s = 0xC0FFEEu;
    auto rnd = [&](int lo, int hi) { s = s * 1103515245u + 12345u; return lo + (int)((s >> 16) % (hi - lo + 1)); };
    auto fill_dyadic = [&](int off_exp, int off_sgn, int n) {
        for (int i = 0; i < n; i++) {
            buf[off_exp + i] = rnd(-3, 2);          // exponent in [-3, 2] -> weight 2^-3..2^2
            buf[off_sgn + i] = rnd(0, 1) ? 1 : -1;  // sign +/-1
        }
    };
    fill_dyadic(OFF_R1_EXP, OFF_R1_SGN, WORDS_R1_EXP);
    fill_dyadic(OFF_R2_EXP, OFF_R2_SGN, WORDS_R2_EXP);
    fill_dyadic(OFF_P1_EXP, OFF_P1_SGN, WORDS_P1_EXP);
    fill_dyadic(OFF_P2_EXP, OFF_P2_SGN, WORDS_P2_EXP);
    fill_dyadic(OFF_PH_EXP, OFF_PH_SGN, WORDS_PH_EXP);
    fill_dyadic(OFF_VH_EXP, OFF_VH_SGN, WORDS_VH_EXP);
    for (int i = 0; i < MZ_OBS; i++) buf[OFF_OBS + i] = rnd(0, 1);   // one-hot-ish gridworld obs

    int rc = forge_train_step_muzero(buf.data(), 0, 0, 0, 0);
    if (rc != 0) { std::printf("FAIL: kernel returned %d\n", rc); return 1; }

    int fails = 0;
    auto in_range = [](int32_t v) { return v >= -32768 && v <= 32767; };
    bool any_nonzero = false;
    std::printf("policy:");
    for (int i = 0; i < MZ_A; i++) {
        std::printf(" %d", buf[OFF_POLICY + i]);
        if (!in_range(buf[OFF_POLICY + i])) fails++;
        if (buf[OFF_POLICY + i] != 0) any_nonzero = true;
    }
    std::printf("  value: %d\n", buf[OFF_VALUE]);
    if (!in_range(buf[OFF_VALUE])) fails++;
    if (!any_nonzero) { std::printf("FAIL: policy all zero (forward did not propagate)\n"); fails++; }

    // determinism: re-run -> identical outputs
    std::vector<int32_t> out1(buf.begin() + OFF_POLICY, buf.begin() + OFF_VALUE + 1);
    forge_train_step_muzero(buf.data(), 0, 0, 0, 0);
    for (int i = 0; i <= (OFF_VALUE - OFF_POLICY); i++)
        if (buf[OFF_POLICY + i] != out1[i]) { std::printf("FAIL: non-deterministic\n"); fails++; break; }

    // all-zero weights + zero bias -> zero output (bias-only path)
    std::fill(buf.begin(), buf.end(), 0);
    forge_train_step_muzero(buf.data(), 0, 0, 0, 0);
    for (int i = 0; i < MZ_A; i++) if (buf[OFF_POLICY + i] != 0) { std::printf("FAIL: zero-weights gave nonzero policy\n"); fails++; break; }

    std::printf(fails == 0 ? "[tb_muzero] PASS (zero-DSP forward, CSIM)\n"
                           : "[tb_muzero] FAIL (%d checks)\n", fails);
    return fails == 0 ? 0 : 1;
}
