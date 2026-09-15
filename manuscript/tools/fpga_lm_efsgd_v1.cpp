// CPU-only, separately versioned error-feedback SGD adapter. NOT an FPGA image.
// The frozen forward/backward is included unchanged; this file owns orchestration.
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__) || defined(TT_FPGA_PCI)
#error "EFSGD v1 is a sanitized CPU research adapter, not deployed FPGA logic"
#endif
#if !defined(TT_RESIDENT_W) || !defined(TT_ONCHIP_GRAD)
#error "EFSGD requires resident weights and gradient accumulators at every width"
#endif
#ifndef EF_SOURCE_SHA256
#error "Build must bind the complete source identity"
#endif
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#define forge_train_step_morph_lm ef_frozen_original_dispatch
#include "../../tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp"
#undef forge_train_step_morph_lm
#include "../../scripts/fpga_lm_layout.cpp"

namespace {
constexpr uint32_t WIRE_MAGIC = 0x45465331u;
constexpr int STATE_HEADER = 20;
constexpr int WEIGHT_WORDS = WORDS_EMBED + WORDS_MH + WORDS_MP + WORDS_MB + WORDS_W_HEAD;
constexpr int STATE_WORDS = STATE_HEADER + WORDS_CE_LUT + 2 * WEIGHT_WORDS;
constexpr int MAX_WORDS = STATE_WORDS > ML_BUFFER_WORDS + 1 ? STATE_WORDS : ML_BUFFER_WORDS + 1;
struct Region { int32_t* weights; int32_t* gradients; int offset; int count; };
Region regions[] = {{embed_oc, g_embed_oc, OFF_EMBED, WORDS_EMBED},
                    {mh_buf, g_mh_oc, OFF_MH, WORDS_MH},
                    {mp_buf, g_mp_oc, OFF_MP, WORDS_MP},
                    {mb_buf, g_mb_oc, OFF_MB, WORDS_MB},
                    {wh_buf, g_whead_oc, OFF_W_HEAD, WORDS_W_HEAD}};
std::vector<int32_t> buffer(ML_BUFFER_WORDS);
bool initialized = false;
bool poisoned = false;
int shift = 0;
uint64_t steps = 0;

uint32_t source_word(int index) {
    const char* text = EF_SOURCE_SHA256;
    uint32_t word = 0;
    for (int j = 0; j < 8; ++j) {
        char c = text[index * 8 + j];
        word = (word << 4) | uint32_t(c <= '9' ? c - '0' : c - 'a' + 10);
    }
    return word;
}
int64_t denominator(int k) { return int64_t(1) << k; }
bool int16_value(int64_t value) { return value >= -32768 && value <= 32767; }
bool valid_config(const int32_t* data) {
    if (data[OFF_CFG] < 0 || data[OFF_CFG] > 1 || data[OFF_CFG + 1] < 0 || data[OFF_CFG + 1] > 30) return false;
    for (int i = 0; i < WORDS_CE_LUT; ++i)
        if (data[OFF_CE_LUT + i] != FT_CE_EXP2_Q16[i]) return false;
    return true;
}
void load_weights() {
    for (const auto& region : regions)
        for (int i = 0; i < region.count; ++i) region.weights[i] = buffer[region.offset + i];
    for (int i = 0; i < WORDS_CE_LUT; ++i) ce_lut[i] = buffer[OFF_CE_LUT + i];
}
int reset(const std::vector<int32_t>& input) {
    if (input.size() != ML_BUFFER_WORDS + 1 || input[0] < 0 || input[0] > 30) return 11;
    if (!valid_config(input.data() + 1)) return 12;
    for (const auto& region : regions)
        for (int i = 0; i < region.count; ++i)
            if (!int16_value(input[1 + region.offset + i])) return 13;
    for (int i = 0; i < ML_BUFFER_WORDS; ++i) buffer[i] = input[i + 1];
    shift = input[0]; steps = 0;
    load_weights();
    for (const auto& region : regions)
        for (int i = 0; i < region.count; ++i) region.gradients[i] = 0;
    initialized = true;
    return 0;
}

// Independent int64 clipping preflight. It uses the same scratch BS region,
// overwritten again by the unchanged native forward. No output/update is taken
// from this check; every clipping event fails before backward or weight changes.
int preflight_forward(const int32_t* tokens) {
    std::vector<int64_t> residual(ML_D), conv(ML_D), projected(ML_D);
    zero_region_wide(buffer.data(), OFF_BS_STREAM, WORDS_BS_STREAM);
    for (int m = 0; m < ML_L_SEQ; ++m) {
        for (int d = 0; d < ML_D; ++d) residual[d] = embed_oc[tokens[m] * ML_D + d];
        for (int l = 0; l < ML_N_LAYERS; ++l) {
            int64_t sum = 0;
            for (int d = 0; d < ML_D; ++d) sum += residual[d];
            const int64_t mean = sum >> ML_LOG2_D;
            for (int d = 0; d < ML_D; ++d) {
                const int64_t value = residual[d] - mean;
                if (!int16_value(value)) return 31;
                buffer[off_bs(l, m) + d] = int32_t(value);
            }
            for (int d = 0; d < ML_D; ++d) {
                int64_t best = l % 2 ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int64_t>::min();
                for (int tap = 0; tap < ML_K; ++tap) {
                    int time = m - (tap << (l < 6 ? l : 6));
                    if (time < 0) continue;
                    int64_t value = int64_t(buffer[off_bs(l, time) + d]) + mh_buf[l * ML_D * ML_K + d * ML_K + tap];
                    if ((l % 2 && value < best) || (!(l % 2) && value > best)) best = value;
                }
                if (!int16_value(best)) return 32;
                conv[d] = best;
            }
            for (int j = 0; j < ML_D; ++j) {
                int64_t best = std::numeric_limits<int64_t>::min();
                for (int i = 0; i < ML_D; ++i) {
                    int64_t value = conv[i] + mp_buf[l * ML_D * ML_D + j * ML_D + i];
                    if (value > best) best = value;
                }
                projected[j] = best + mb_buf[l * ML_D + j];
                if (!int16_value(projected[j]) || !int16_value(residual[j] + projected[j])) return 33;
            }
            for (int j = 0; j < ML_D; ++j) residual[j] += projected[j];
        }
        for (int v = 0; v < ML_V; ++v) {
            int64_t best = std::numeric_limits<int64_t>::min();
            for (int d = 0; d < ML_D; ++d) {
                int64_t value = residual[d] + wh_buf[v * ML_D + d];
                if (value > best) best = value;
            }
            if (!int16_value(best)) return 34;
        }
    }
    return 0;
}

int apply_updates() {
    // Validate ALL regions before applying ANY weight update. int64 division
    // by a positive power of two truncates toward zero, including INT_MIN.
    for (const auto& region : regions)
        for (int i = 0; i < region.count; ++i) {
            int64_t update = int64_t(region.gradients[i]) / denominator(shift);
            if (!int16_value(int64_t(region.weights[i]) - update)) return 60;
        }
    for (const auto& region : regions)
        for (int i = 0; i < region.count; ++i) {
            int64_t accumulator = region.gradients[i];
            int64_t update = accumulator / denominator(shift);
            region.weights[i] = int32_t(int64_t(region.weights[i]) - update);
            region.gradients[i] = int32_t(accumulator % denominator(shift));
        }
    return 0;
}

int window_step(const std::vector<int32_t>& input, bool training, std::vector<int32_t>& output) {
    if (!initialized || input.size() != 1 + 2 * ML_L_SEQ || (input[0] != 0 && input[0] != 1)) return 20;
    for (int i = 1; i < 1 + 2 * ML_L_SEQ; ++i) if (input[i] < 0 || input[i] >= ML_V) return 21;
    if (training && steps == std::numeric_limits<uint64_t>::max()) return 22;
    const int32_t* tokens = input.data() + 1;
    const int32_t* targets = tokens + ML_L_SEQ;
    int status = preflight_forward(tokens);
    if (status) return status;
    zero_region_wide(buffer.data(), OFF_BS_STREAM, WORDS_BS_STREAM);
    for (int m = 0; m < ML_L_SEQ; ++m) {
        status = do_fwd_loss(buffer.data(), 0, m, tokens[m], targets[m], 0);
        if (status) return 40 + status;
    }
    if (training) {
        zero_region_wide(buffer.data(), OFF_BS_STREAM, WORDS_BS_STREAM);
        // Remainders are intentionally NOT cleared: parameter gradients are
        // additive sinks only. Nonrecovering UBSan catches every unsafe +=.
        for (int m = ML_L_SEQ - 1; m >= 0; --m) {
            status = do_bwd(buffer.data(), 0, m, tokens[m]);
            if (status) return 50 + status;
        }
        status = apply_updates();
        if (status) return status;
        ++steps;
    }
    output.push_back(int32_t(uint32_t(steps)));
    output.push_back(int32_t(uint32_t(steps >> 32)));
    for (int i = 0; i < WORDS_PRED; ++i) output.push_back(buffer[OFF_PRED + i]);
    for (int i = 0; i < WORDS_GRAD_LOGIT; ++i) output.push_back(buffer[OFF_GRAD_LOGIT + i]);
    if (input[0]) {
        for (int i = 0; i < WORDS_AD_HEAD; ++i) output.push_back(buffer[OFF_AD_HEAD + i]);
        for (int i = 0; i < WORDS_PROJ_IDX; ++i) output.push_back(buffer[OFF_PROJ_IDX + i]);
        for (int i = 0; i < WORDS_TAP_IDX; ++i) output.push_back(buffer[OFF_TAP_IDX + i]);
    }
    return 0;
}

std::vector<int32_t> export_state() {
    std::vector<int32_t> out = {int32_t(WIRE_MAGIC), 1, ML_D, ML_BUFFER_WORDS, WEIGHT_WORDS,
                              shift, buffer[OFF_CFG], buffer[OFF_CFG + 1],
                              int32_t(uint32_t(steps)), int32_t(uint32_t(steps >> 32)), WORDS_CE_LUT, 0};
    for (int i = 0; i < 8; ++i) out.push_back(int32_t(source_word(i)));
    for (int i = 0; i < WORDS_CE_LUT; ++i) out.push_back(ce_lut[i]);
    for (const auto& region : regions) for (int i = 0; i < region.count; ++i) out.push_back(region.weights[i]);
    for (const auto& region : regions) for (int i = 0; i < region.count; ++i) out.push_back(region.gradients[i]);
    return out;
}
int import_state(const std::vector<int32_t>& in) {
    if (in.size() != STATE_WORDS || uint32_t(in[0]) != WIRE_MAGIC || in[1] != 1 || in[2] != ML_D
        || in[3] != ML_BUFFER_WORDS || in[4] != WEIGHT_WORDS || in[5] < 0 || in[5] > 30
        || in[6] < 0 || in[6] > 1 || in[7] < 0 || in[7] > 30 || in[10] != WORDS_CE_LUT || in[11] != 0) return 70;
    for (int i = 0; i < 8; ++i) if (uint32_t(in[12 + i]) != source_word(i)) return 71;
    int offset = STATE_HEADER;
    for (int i = 0; i < WORDS_CE_LUT; ++i) if (in[offset + i] != FT_CE_EXP2_Q16[i]) return 72;
    offset += WORDS_CE_LUT;
    for (int i = 0; i < WEIGHT_WORDS; ++i) if (!int16_value(in[offset + i])) return 73;
    for (int i = 0; i < WEIGHT_WORDS; ++i) {
        int64_t remainder = in[offset + WEIGHT_WORDS + i];
        if (remainder <= -denominator(in[5]) || remainder >= denominator(in[5])) return 74;
    }
    shift = in[5]; steps = uint32_t(in[8]) | (uint64_t(uint32_t(in[9])) << 32);
    buffer[OFF_CFG] = in[6]; buffer[OFF_CFG + 1] = in[7];
    for (int i = 0; i < WORDS_CE_LUT; ++i) ce_lut[i] = buffer[OFF_CE_LUT + i] = in[STATE_HEADER + i];
    for (const auto& region : regions) for (int i = 0; i < region.count; ++i) region.weights[i] = in[offset++];
    for (const auto& region : regions) for (int i = 0; i < region.count; ++i) region.gradients[i] = in[offset++];
    initialized = true;
    return 0;
}
bool read_word(uint32_t& value) {
    unsigned char bytes[4];
    if (!std::cin.read(reinterpret_cast<char*>(bytes), 4)) return false;
    value = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) | (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    return true;
}
void write_word(uint32_t value) {
    char bytes[4];
    for (int i = 0; i < 4; ++i) bytes[i] = char((value >> (i * 8)) & 255u);
    std::cout.write(bytes, 4);
}
}  // namespace

int main() {
    static_assert(ML_BATCH_N == 1, "Only the declared single-window protocol is supported");
    std::ios::sync_with_stdio(false);
    while (true) {
        uint32_t magic, operation, count;
        if (!read_word(magic)) return 0;
        if (!read_word(operation) || !read_word(count) || magic != WIRE_MAGIC || count > uint32_t(MAX_WORDS)) return 2;
        std::vector<int32_t> input(count), output;
        for (uint32_t i = 0; i < count; ++i) { uint32_t value; if (!read_word(value)) return 2; input[i] = int32_t(value); }
        if (operation == 6 && count == 0) return 0;
        int status = poisoned ? 90 : 0;
        if (!status) {
            if (operation == 1) status = reset(input);
            else if (operation == 2 || operation == 3) status = window_step(input, operation == 2, output);
            else if (operation == 4 && count == 0 && initialized) output = export_state();
            else if (operation == 5) status = import_state(input);
            else if (operation == 7 && count == 0) {
                int values[128]; int n = tt_morph_lm_layout(values, 128);
                for (int i = 0; i < n; ++i) output.push_back(values[i]);
            } else status = 99;
        }
        if (status) { poisoned = true; output.clear(); }
        write_word(WIRE_MAGIC); write_word(uint32_t(status)); write_word(uint32_t(output.size()));
        for (int32_t value : output) write_word(uint32_t(value));
        std::cout.flush();
        if (!std::cout) return 3;
    }
}
