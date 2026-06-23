// forge_sparsecoding_pc_tb.cpp — CSIM: run the forge_sparsecoding_pc kernel in INFER mode (mode=1) and
// compare the settled codes s + reconstructions to an int64 reference of the SAME settle (dyadic D·s / Dᵀ·r
// shift-add, max-shift gradient, counting top-k WTA), bit-for-bit. INFER has no dictionary renorm, so it is
// the clean bit-exact gate (Gate A) before the AFI build.
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ap_int.h>

#define P 256
#define M 256
#define NB 32
#define K 24
#define RB 6
#define GBITS 6
#define LRS 1
#define WL 16
#define O_CTRL 0
#define O_D    (O_CTRL + WL)
#define O_X    (O_D    + P*M)
#define O_MASK (O_X    + NB*P)
#define O_S    (O_MASK + NB*P)
#define O_RECON (O_S   + NB*M)
#define BUFLEN  (O_RECON + NB*P)

void forge_sparsecoding_pc(ap_uint<512> *buffer);

static long rdyad(long v){ long a=v<0?-v:v; if(!a)return 0; if(a>32767)a=32767; int m=0; for(int b=15;b>=1;b--)if((a>>b)&1){m=b;break;} if(m>=1&&((a>>(m-1))&1))m++; long p=1L<<m; return v<0?-p:p; }
static int rshift(long mx,int t){ long a=mx<0?-mx:mx; int h=0; for(int b=47;b>=1;b--)if((a>>b)&1){h=b;break;} int s=(h+1)-t; return s>0?s:0; }
static long i16(long v){ return (int16_t)v; }
static int topk_thr(long s[M]){ int thr=0; for(int bit=15;bit>=0;bit--){ int cand=thr|(1<<bit); int cnt=0; for(int m=0;m<M;m++){ long a=s[m]<0?-s[m]:s[m]; if(a>=cand)cnt++; } if(cnt>=K)thr=cand; } return thr; }

int main(){
    int32_t *buf=(int32_t*)aligned_alloc(64,sizeof(int32_t)*BUFLEN);
    for(int i=0;i<BUFLEN;i++) buf[i]=0;
    srand(11);
    for(int i=0;i<P*M;i++) buf[O_D+i]=(rand()%9)-4;
    for(int n=0;n<NB;n++)for(int p=0;p<P;p++){ buf[O_X+n*P+p]=rand()%64; buf[O_MASK+n*P+p]=(rand()%2); }  // masked
    int NS=20; buf[O_CTRL+0]=1; buf[O_CTRL+1]=NS;     // mode=1 INFER, n_steps=20

    // ---- int64 reference settle ----
    static long D[P*M]; for(int i=0;i<P*M;i++) D[i]=buf[O_D+i];
    static long Sref[NB][M], Rref[NB][P];
    for(int n=0;n<NB;n++){
        static long s[M]; for(int m=0;m<M;m++) s[m]=0;
        for(int t=0;t<NS;t++){
            static long recon[P], r[P];
            for(int p=0;p<P;p++){ long acc=0; for(int m=0;m<M;m++) acc+=rdyad(s[m])*rdyad(D[p*M+m]); recon[p]=acc>>RB;
                if(t==NS-1) Rref[n][p]=i16(recon[p]);
                long rv=(long)buf[O_X+n*P+p]-recon[p]; if(buf[O_MASK+n*P+p]==0) rv=0; r[p]=i16(rv); }
            static long g[M]; long mxg=0;
            for(int m=0;m<M;m++){ long acc=0; for(int p=0;p<P;p++) acc+=rdyad(r[p])*rdyad(D[p*M+m]); g[m]=acc; long ag=acc<0?-acc:acc; if(ag>mxg)mxg=ag; }
            int shg=rshift(mxg,GBITS);
            for(int m=0;m<M;m++) s[m]=i16(s[m]+( (g[m]>>shg) >>LRS ));
            int thr=topk_thr(s);
            for(int m=0;m<M;m++){ long a=s[m]<0?-s[m]:s[m]; if(a<thr) s[m]=0; }
        }
        for(int m=0;m<M;m++) Sref[n][m]=s[m];
    }

    forge_sparsecoding_pc((ap_uint<512>*)buf);
    long dS=0,dR=0,nzS=0;
    for(int n=0;n<NB;n++){ for(int m=0;m<M;m++){ long a=labs((long)buf[O_S+n*M+m]-Sref[n][m]); if(a>dS)dS=a; if(buf[O_S+n*M+m])nzS++; }
                           for(int p=0;p<P;p++){ long a=labs((long)buf[O_RECON+n*P+p]-Rref[n][p]); if(a>dR)dR=a; } }
    printf("forge_sparsecoding_pc CSIM (INFER, NB=%d P=%d M=%d K=%d, %d steps): nz codes=%ld/%d\n",NB,P,M,K,NS,nzS,NB*M);
    printf("  s max|diff|=%ld  recon max|diff|=%ld\n",dS,dR);
    int ok=(dS|dR)==0;
    printf("  -> %s\n", ok?"BIT-EXACT (kernel settle == int64 reference)":"MISMATCH");
    free(buf); return ok?0:1;
}
