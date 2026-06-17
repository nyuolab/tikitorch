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

    // ---- dynamics (mode 1) CSIM + INDEPENDENT bit-exact cross-check (catches offset/indexing/ABI bugs) ----
    {
        std::fill(buf.begin(), buf.end(), 0);
        fill_dyadic(OFF_R1_EXP, OFF_R1_SGN, WORDS_R1_EXP);  fill_dyadic(OFF_R2_EXP, OFF_R2_SGN, WORDS_R2_EXP);
        fill_dyadic(OFF_P1_EXP, OFF_P1_SGN, WORDS_P1_EXP);  fill_dyadic(OFF_P2_EXP, OFF_P2_SGN, WORDS_P2_EXP);
        fill_dyadic(OFF_PH_EXP, OFF_PH_SGN, WORDS_PH_EXP);  fill_dyadic(OFF_VH_EXP, OFF_VH_SGN, WORDS_VH_EXP);
        fill_dyadic(OFF_D0A_EXP, OFF_D0A_SGN, WORDS_D0A_EXP); fill_dyadic(OFF_D0B_EXP, OFF_D0B_SGN, WORDS_D0B_EXP);
        fill_dyadic(OFF_D1A_EXP, OFF_D1A_SGN, WORDS_D1A_EXP); fill_dyadic(OFF_D1B_EXP, OFF_D1B_SGN, WORDS_D1B_EXP);
        fill_dyadic(OFF_RH_EXP, OFF_RH_SGN, WORDS_RH_EXP);
        for (int i = 0; i < WORDS_AEMB; i++) buf[OFF_AEMB + i] = rnd(-4, 4);   // action-embedding table
        for (int i = 0; i < MZ_OBS; i++) buf[OFF_OBS + i] = rnd(0, 1);
        auto fill_bias = [&](int off, int n) { for (int i = 0; i < n; i++) buf[off + i] = rnd(-2, 2); };  // exercise bias path
        fill_bias(OFF_B_R1, WORDS_B_R1); fill_bias(OFF_B_R2, WORDS_B_R2); fill_bias(OFF_B_P1, WORDS_B_P1);
        fill_bias(OFF_B_P2, WORDS_B_P2); fill_bias(OFF_B_PH, WORDS_B_PH); fill_bias(OFF_B_VH, WORDS_B_VH);
        fill_bias(OFF_B_D0A, WORDS_B_D0A); fill_bias(OFF_B_D0B, WORDS_B_D0B); fill_bias(OFF_B_D1A, WORDS_B_D1A);
        fill_bias(OFF_B_D1B, WORDS_B_D1B); fill_bias(OFF_B_RH, WORDS_B_RH);
        const int action = 2 % MZ_A;

        forge_train_step_muzero(buf.data(), 0, 0, 0, 0);                       // forward -> OFF_H
        std::vector<data_t> h0(MZ_L); for (int i = 0; i < MZ_L; i++) h0[i] = (data_t)buf[OFF_H + i];
        forge_train_step_muzero(buf.data(), 1, action, 0, 0);                 // dynamics step (mode 1)

        // independent recompute with the SAME acc_t/data_t widths => bit-exact to the kernel arithmetic
        auto rmac = [](data_t x, int e, int s) { acc_t v = (e >= 0) ? ((acc_t)x << e) : ((acc_t)x >> (-e)); return s >= 0 ? v : (acc_t)(-v); };
        auto rsat = [](acc_t a) -> data_t { if (a > 32767) a = 32767; if (a < -32768) a = -32768; return (data_t)a; };
        auto rlin = [&](int oe, int os, int ob, const std::vector<data_t>& x, std::vector<data_t>& y, int OUT, int IN) {
            for (int o = 0; o < OUT; o++) { acc_t acc = (acc_t)buf[ob + o]; for (int i = 0; i < IN; i++) acc += rmac(x[i], buf[oe + o * IN + i], buf[os + o * IN + i]); y[o] = rsat(acc); } };
        auto rrelu = [&](std::vector<data_t>& a) { for (auto& v : a) if (v < 0) v = 0; };
        auto rsub = [&](std::vector<data_t>& a) { acc_t s = 0; for (auto v : a) s += v; data_t m = (data_t)(s / MZ_L); for (auto& v : a) v = rsat((acc_t)v - m); };

        std::vector<data_t> z(MZ_L), u(MZ_DD), c(MZ_L);
        for (int i = 0; i < MZ_L; i++) z[i] = rsat((acc_t)h0[i] + (acc_t)buf[OFF_AEMB + action * MZ_L + i]);
        rlin(OFF_D0A_EXP, OFF_D0A_SGN, OFF_B_D0A, z, u, MZ_DD, MZ_L); rrelu(u);
        rlin(OFF_D0B_EXP, OFF_D0B_SGN, OFF_B_D0B, u, c, MZ_L, MZ_DD);
        for (int i = 0; i < MZ_L; i++) z[i] = rsat((acc_t)z[i] + (acc_t)c[i]); rsub(z);
        rlin(OFF_D1A_EXP, OFF_D1A_SGN, OFF_B_D1A, z, u, MZ_DD, MZ_L); rrelu(u);
        rlin(OFF_D1B_EXP, OFF_D1B_SGN, OFF_B_D1B, u, c, MZ_L, MZ_DD);
        for (int i = 0; i < MZ_L; i++) z[i] = rsat((acc_t)z[i] + (acc_t)c[i]); rsub(z);
        std::vector<data_t> rr(1), tdp(MZ_DP), zz(MZ_L), pol(MZ_A), val(1);
        rlin(OFF_RH_EXP, OFF_RH_SGN, OFF_B_RH, z, rr, 1, MZ_L);
        rlin(OFF_P1_EXP, OFF_P1_SGN, OFF_B_P1, z, tdp, MZ_DP, MZ_L); rrelu(tdp);
        rlin(OFF_P2_EXP, OFF_P2_SGN, OFF_B_P2, tdp, zz, MZ_L, MZ_DP);
        rlin(OFF_PH_EXP, OFF_PH_SGN, OFF_B_PH, zz, pol, MZ_A, MZ_L);
        rlin(OFF_VH_EXP, OFF_VH_SGN, OFF_B_VH, zz, val, 1, MZ_L);

        int dfa = 0;
        for (int i = 0; i < MZ_L; i++) if (buf[OFF_H + i] != (int)z[i]) dfa++;
        if (buf[OFF_REWARD] != (int)rr[0]) dfa++;
        for (int i = 0; i < MZ_A; i++) if (buf[OFF_POLICY + i] != (int)pol[i]) dfa++;
        if (buf[OFF_VALUE] != (int)val[0]) dfa++;
        bool hchanged = false; for (int i = 0; i < MZ_L; i++) if (buf[OFF_H + i] != (int)h0[i]) hchanged = true;
        if (!hchanged) { std::printf("FAIL: dynamics did not change the latent\n"); fails++; }
        if (!in_range(buf[OFF_REWARD])) { std::printf("FAIL: reward out of int16 range\n"); fails++; }
        std::vector<int32_t> hsnap(buf.begin() + OFF_H, buf.begin() + OFF_H + MZ_L);
        for (int i = 0; i < MZ_L; i++) buf[OFF_H + i] = (int)h0[i];            // restore h, re-run -> determinism
        forge_train_step_muzero(buf.data(), 1, action, 0, 0);
        for (int i = 0; i < MZ_L; i++) if (buf[OFF_H + i] != hsnap[i]) { std::printf("FAIL: dynamics non-deterministic\n"); fails++; break; }
        if (dfa == 0) std::printf("[tb_muzero] dynamics mode 1 bit-exact vs independent recompute (h',r,policy,value) PASS\n");
        else { std::printf("[tb_muzero] FAIL: dynamics cross-check mismatch in %d outputs\n", dfa); fails++; }
    }

    // all-zero weights + zero bias -> zero output (bias-only path)
    std::fill(buf.begin(), buf.end(), 0);
    forge_train_step_muzero(buf.data(), 0, 0, 0, 0);
    for (int i = 0; i < MZ_A; i++) if (buf[OFF_POLICY + i] != 0) { std::printf("FAIL: zero-weights gave nonzero policy\n"); fails++; break; }

    std::printf(fails == 0 ? "[tb_muzero] PASS (zero-DSP forward, CSIM)\n"
                           : "[tb_muzero] FAIL (%d checks)\n", fails);
    return fails == 0 ? 0 : 1;
}
