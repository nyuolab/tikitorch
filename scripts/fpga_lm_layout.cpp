// Host-side ABI probe. Compile alongside bridge.cpp with the SAME model flags.
// It is not FPGA logic and does not establish the identity of a loaded AFI.
#include "forge_buffer_morph_lm.hpp"

extern "C" int tt_morph_lm_backend() {
#ifdef TT_FPGA_PCI
    return 2;
#elif defined(CSIM_ONLY) && defined(TT_KERNEL_MORPH_LM)
    return 1;
#else
    return 0;
#endif
}

extern "C" int tt_morph_lm_layout(int* out, int capacity) {
    const int values[] = {
        1, ML_D, ML_V, ML_L_SEQ, ML_N_LAYERS, ML_K, ML_BATCH_N,
        ML_BUFFER_WORDS, ML_BUFFER_BYTES, ML_LOG2_D,
        OFF_EMBED, WORDS_EMBED, OFF_MH, WORDS_MH, OFF_MP, WORDS_MP,
        OFF_MB, WORDS_MB, OFF_W_HEAD, WORDS_W_HEAD,
        OFF_TAP_IDX, WORDS_TAP_IDX, OFF_PROJ_IDX, WORDS_PROJ_IDX,
        OFF_AD_HEAD, WORDS_AD_HEAD, OFF_BS_STREAM, WORDS_BS_STREAM,
        OFF_TOKENS, WORDS_TOKENS, OFF_TARGETS, WORDS_TARGETS,
        OFF_GRAD_LOGIT, WORDS_GRAD_LOGIT, OFF_PRED, WORDS_PRED,
        OFF_CFG, WORDS_CFG, OFF_CE_LUT, WORDS_CE_LUT
    };
    const int count = sizeof(values) / sizeof(values[0]);
    if (!out || capacity < count) return count;
    for (int i = 0; i < count; ++i) out[i] = values[i];
    return count;
}
