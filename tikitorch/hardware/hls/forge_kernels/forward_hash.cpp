// forward_hash.cpp — FORWARD-IDENTITY check. The BF16 backward header must NOT change the deployed int
// forward (g_forward). We include the BF16 header (which pulls in the canonical golden block) and hash the
// int g_forward outputs (LG + GLOG) over a fixed sequence of batches. The hash is printed; compared against
// the SAME hash from the canonical golden block alone (forward_hash_ref.cpp) — they must be byte-identical.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef USE_BF16_HEADER
#include "forge_pc_xformer_bf16_golden_block.hpp"     // includes the canonical golden block + adds BF16 syms
#else
#include "forge_pc_xformer_golden_block.hpp"          // canonical golden ONLY (reference)
#endif

static int RV(double s){ double u=0; for(int k=0;k<6;k++) u+=rand()/(double)RAND_MAX; double g=(u-3.0)*s; int v=(int)lround(g); if(v>32767)v=32767; if(v<-32767)v=-32767; return v; }
static void init(GW &w){ srand(7); int wf=getenv("WFRAC")?atoi(getenv("WFRAC")):0; int ws=1<<wf;
    for(int i=0;i<X_D*X_D;i++){ w.WQ[i]=RV(8)*ws;w.WK[i]=RV(8)*ws;w.WV[i]=RV(8)*ws;w.WO[i]=RV(8)*ws;}
    for(int i=0;i<X_FFN*X_D;i++) w.WUP[i]=RV(8)*ws; for(int i=0;i<X_D*X_FFN;i++) w.WDN[i]=RV(8)*ws;
    for(int i=0;i<X_VOCAB*X_D;i++) w.WHEAD[i]=RV(8)*ws; for(int i=0;i<X_VOCAB*X_D;i++) w.EMB[i]=RV(16);
    for(int i=0;i<X_SEQ*X_D;i++) w.POS[i]=RV(8);
    for(int d=0;d<X_D;d++){ w.LN1G[d]=64;w.LN2G[d]=64;w.LNFG[d]=64; w.LN1B[d]=0;w.LN2B[d]=0;w.LNFB[d]=0; } }

// FNV-1a 64-bit over the int forward outputs
static unsigned long long fnv(unsigned long long h, i64 v){ unsigned char* p=(unsigned char*)&v; for(int b=0;b<8;b++){ h^=p[b]; h*=1099511628211ULL; } return h; }

int main(){
    g_lut_init(); g_exp_init();
    const char* path=getenv("SHAKESPEARE"); char def[256]; if(!path){ snprintf(def,256,"%s/.cache/tiur/tinyshakespeare.txt",getenv("HOME")); path=def; }
    FILE* fp=fopen(path,"rb"); fseek(fp,0,SEEK_END); long n=ftell(fp); fseek(fp,0,SEEK_SET); char* txt=(char*)malloc(n); long rd=fread(txt,1,n,fp); fclose(fp); (void)rd;
    int map[256]; for(int i=0;i<256;i++) map[i]=-1; int vocab=0;
    for(long i=0;i<n && vocab<X_VOCAB;i++){ unsigned char c=txt[i]; if(map[c]<0) map[c]=vocab++; }
    int* data=(int*)malloc(sizeof(int)*n); long nd=0; for(long i=0;i<n;i++){ int id=map[(unsigned char)txt[i]]; data[nd++]= id<0?0:id; }
    static GW w; static GA a; init(w);
    int tok[X_SEQ], tgt[X_SEQ];
    unsigned long long h=1469598103934665603ULL;
    for(int b=0;b<20;b++){ long off=1000+b*777; for(int t=0;t<X_SEQ;t++){ tok[t]=data[off+t]; tgt[t]=data[off+t+1]; }
        i64 loss=g_forward(w,a,tok,tgt); h=fnv(h,loss);
        for(int i=0;i<X_SEQ*X_VOCAB;i++){ h=fnv(h,a.LG[i]); h=fnv(h,a.GLOG[i]); } }
    printf("FORWARD_HASH %016llx  (%s)\n", h,
#ifdef USE_BF16_HEADER
        "with BF16 header"
#else
        "canonical golden only"
#endif
    );
    return 0;
}
