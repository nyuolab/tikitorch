// gen_init.cpp — emit the s2 driver's EXACT self-init as master<l>.f32 files, at the current -D dims.
// Byte-identical to what forge_pc_xformer_s2_shard.cpp would self-init (same LCG, same field order,
// same per-chip seed = SEED + 1000*l), so the files serve as the MATCHED INIT for shim-vs-silicon
// comparisons at ANY config (e.g. -DS2_VOCAB=8192): both sides load the same files via INIT_DIR/INIT_S3.
//
//   g++ -O2 -std=c++14 [-DS2_VOCAB=8192 ...] -I../../../hardware/hls/forge_kernels gen_init.cpp -o gen_init
//   SEED=7 CHIPS=8 OUTDIR=data/v8k ./gen_init
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "forge_pc_xformer_s2_config.hpp"

// ==== verbatim from forge_pc_xformer_s2_shard.cpp (the init RNG + FMaster layout) ====
static unsigned g_rng = 1;
static inline double bf_u(){ g_rng = g_rng*1103515245u + 12345u; return ((g_rng>>8)&0xffffff)/(double)0x1000000; }
static inline double bf_n(){ double u=0; for(int k=0;k<12;k++) u+=bf_u(); return u-6.0; }
static void uni(float*W,long n,int fan){ double a=1.0/sqrt((double)fan); for(long i=0;i<n;i++) W[i]=(float)((bf_u()*2-1)*a); }
struct FMaster {
    float WQ[S2_D*S2_D], WK[S2_D*S2_D], WV[S2_D*S2_D], WO[S2_D*S2_D];
    float WUP[S2_FFN*S2_D], WDN[S2_D*S2_FFN], WHEAD[S2_VOCAB*S2_D];
    float EMB[S2_VOCAB*S2_D], POS[S2_SEQ*S2_D];
    float LN1G[S2_D], LN1B[S2_D], LN2G[S2_D], LN2B[S2_D], LNFG[S2_D], LNFB[S2_D];
};
static void init_master(FMaster &m, unsigned seed){
    g_rng = seed*2654435761u + 1u;
    uni(m.WQ,(long)S2_D*S2_D,S2_D); uni(m.WK,(long)S2_D*S2_D,S2_D); uni(m.WV,(long)S2_D*S2_D,S2_D); uni(m.WO,(long)S2_D*S2_D,S2_D);
    uni(m.WUP,(long)S2_FFN*S2_D,S2_D); uni(m.WDN,(long)S2_D*S2_FFN,S2_FFN); uni(m.WHEAD,(long)S2_VOCAB*S2_D,S2_D);
    for(long i=0;i<(long)S2_VOCAB*S2_D;i++) m.EMB[i]=(float)bf_n();
    for(long i=0;i<(long)S2_SEQ*S2_D;i++)   m.POS[i]=(float)bf_n();
    for(int d=0;d<S2_D;d++){ m.LN1G[d]=1;m.LN2G[d]=1;m.LNFG[d]=1; m.LN1B[d]=0;m.LN2B[d]=0;m.LNFB[d]=0; }
}
// ==== end verbatim ====

int main(){
    unsigned seed = getenv("SEED") ? (unsigned)atoi(getenv("SEED")) : 7u;
    int chips     = getenv("CHIPS") ? atoi(getenv("CHIPS")) : 8;
    const char* out = getenv("OUTDIR") ? getenv("OUTDIR") : ".";
    FMaster* m = (FMaster*)malloc(sizeof(FMaster));
    printf("[gen_init] D=%d FFN=%d SEQ=%d VOCAB=%d -> %zu floats (%.1f MB) per chip, seed=%u\n",
           S2_D, S2_FFN, S2_SEQ, S2_VOCAB, sizeof(FMaster)/sizeof(float), sizeof(FMaster)/1048576.0, seed);
    for(int l=0;l<chips;l++){
        init_master(*m, seed + 1000u*l);
        char p[512]; snprintf(p,512,"%s/master%d.f32",out,l);
        FILE* f=fopen(p,"wb"); if(!f){ fprintf(stderr,"open %s failed\n",p); return 1; }
        fwrite(m, sizeof(float), sizeof(FMaster)/sizeof(float), f); fclose(f);
        printf("[gen_init] chip %d -> %s\n", l, p);
    }
    free(m); return 0;
}
