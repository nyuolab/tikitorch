// offsets_dump — print the kernel's exact buffer offsets + BUFLEN as JSON for the deploy config, so
// the Python export (tropical_llm_export_s1_faithful.py --offsets) can never drift from the kernel.
// build: g++ -std=c++14 -I . -DS1_KV2=1 -DS1_D=128 -DS1_LOG2_D=7 -DS1_H=4 -DS1_DH=32 -DS1_FFN=512 \
//        -DS1_N_LAYERS=4 -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1 offsets_dump.cpp -o /tmp/offdump
#include <cstdio>
#include "forge_s1_config.hpp"
int main() {
    printf("{\n");
    printf("  \"D\": %d, \"H\": %d, \"DH\": %d, \"FFN\": %d, \"T\": %d, \"V\": %d, \"NL\": %d,\n",
           S1_D, S1_H, S1_DH, S1_FFN, S1_L_SEQ, S1_V, S1_N_LAYERS);
    printf("  \"BUFLEN\": %d, \"OFF_POS\": %d, \"OFF_LOGITS\": %d,\n",
           (int)S1_BUFLEN, (int)S1_OFF_POS, (int)S1_OFF_LOGITS);
    printf("  \"offsets\": {\n");
    printf("    \"EMBED\": %d, \"WQ\": %d, \"WK\": %d, \"WV\": %d, \"WO\": %d,\n",
           (int)S1_OFF_EMBED, (int)S1_OFF_WQ, (int)S1_OFF_WK, (int)S1_OFF_WV, (int)S1_OFF_WO);
    printf("    \"W_UP\": %d, \"W_DOWN\": %d, \"LN1_G\": %d, \"LN1_B\": %d, \"LN2_G\": %d, \"LN2_B\": %d,\n",
           (int)S1_OFF_W_UP, (int)S1_OFF_W_DOWN, (int)S1_OFF_LN1_G, (int)S1_OFF_LN1_B, (int)S1_OFF_LN2_G, (int)S1_OFF_LN2_B);
    printf("    \"LNF_G\": %d, \"LNF_B\": %d, \"W_HEAD\": %d, \"CE_LUT\": %d, \"ATTN_LUT\": %d, \"TOKENS\": %d, \"TARGETS\": %d\n",
           (int)S1_OFF_LNF_G, (int)S1_OFF_LNF_B, (int)S1_OFF_W_HEAD, (int)S1_OFF_CE_LUT, (int)S1_OFF_ATTN_LUT, (int)S1_OFF_TOKENS, (int)S1_OFF_TARGETS);
    printf("  }\n}\n");
    return 0;
}
