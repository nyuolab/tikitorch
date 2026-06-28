// s1_dump — run the REAL kernel forward on ONE window and dump the whole buffer (with all the
// in-place activation saves: X1/R/LOGITS regions) so a python sim can diff layer-by-layer.
// build: g++ -O2 -std=c++14 -I . -DS1_GEMM_QX=0 -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 \
//        -DS1_POS_EMBED=1 -DS1_N_LAYERS=2 s1_dump.cpp -o /tmp/s1_dump
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include "forge_s1_config.hpp"
#include "forge_train_step_s1.cpp"

static std::vector<int32_t> load_npy_i32(const char *path, long &n) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(2); }
    char magic[6]; size_t r=fread(magic,1,6,f); (void)r; unsigned char ver[2]; r=fread(ver,1,2,f);
    unsigned short hlen; r=fread(&hlen,2,1,f); fseek(f, 10+hlen, SEEK_SET);
    fseek(f,0,SEEK_END); long bytes=ftell(f); long data=bytes-(10+hlen); fseek(f,10+hlen,SEEK_SET);
    n=data/4; std::vector<int32_t> v(n); r=fread(v.data(),4,n,f); fclose(f); return v;
}
static std::vector<int32_t> load_bin_i32(const char *path, long &n) {
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(2);} fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET);
    n=b/4; std::vector<int32_t> v(n); size_t r=fread(v.data(),4,n,f);(void)r; fclose(f); return v;
}

int main(int argc, char **argv) {
    // args: <buf.npy> <val_ids.bin> <window> <out.bin>
    const char *bufpath=argv[1], *valpath=argv[2]; int win=atoi(argv[3]); const char *out=argv[4];
    long blen=0; std::vector<int32_t> buf0=load_npy_i32(bufpath,blen);
    long vn=0; std::vector<int32_t> val=load_bin_i32(valpath,vn);
    std::vector<int32_t> buf(S1_BUFLEN);
    memcpy(buf.data(), buf0.data(), sizeof(int32_t)*S1_BUFLEN);
    long s=(long)win*S1_L_SEQ;
    for (int t=0;t<S1_L_SEQ;t++){ buf[S1_OFF_TOKENS+t]=val[s+t]; buf[S1_OFF_TARGETS+t]=val[s+t+1]; }
    forge_train_step_s1(buf.data(),0,0,0,0);
    FILE *f=fopen(out,"wb"); fwrite(buf.data(),4,S1_BUFLEN,f); fclose(f);
    // also print key offsets so python knows where to look
    printf("OFF_LOGITS=%d OFF_X1=%d OFF_R=%d OFF_TOKENS=%d L_SEQ=%d D=%d V=%d BUFLEN=%d\n",
           (int)S1_OFF_LOGITS, (int)S1_OFF_X1, (int)s1_off_R(1), (int)S1_OFF_TOKENS,
           (int)S1_L_SEQ, (int)S1_D, (int)S1_V, (int)S1_BUFLEN);
    return 0;
}
