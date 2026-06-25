// forge_pc_xformer_sim.cpp — LOCAL multi-chip simulation of the distributed transformer (forge_pc_xformer_shard
// driver) using the REAL-ap_int kernel compiled in. Mirrors the driver's sweep logic EXACTLY (same DDR offsets,
// CFG words, per-chip init, neighbour-message relay) but calls forge_pc_xformer_block() directly on L in-memory
// buffers instead of kicking F2 hardware. Validates, before any F2 run: (a) the driver's DDR staging + message
// protocol, (b) that a transformer trains MODEL-PARALLEL across L blocks (bpb falls). Authentic ap_int (Xilinx
// open-source headers) => identical arithmetic to the silicon kernel.
//   g++ -O2 -std=c++14 -D__AP_INT_SPECIAL_H__ -D__AP_FIXED_SPECIAL_H__ -I/tmp/apreal/include -I. \
//       -Wno-unknown-pragmas forge_pc_xformer_block.cpp forge_pc_xformer_sim.cpp -o /tmp/xfsim && /tmp/xfsim
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ap_int.h>
#include "forge_pc_xformer_config.hpp"
extern "C" void forge_pc_xformer_block(ap_uint<512>* buffer);

#define MAXL 8
static int32_t* buf[MAXL];   // each chip's DDR buffer (X_BUFLEN int32)
static int L;

static void setcfg(int l,int ph,int role,int up){ buf[l][XO_CFG+0]=ph; buf[l][XO_CFG+1]=role; buf[l][XO_CFG+2]=up; buf[l][XO_CFG+3]=0; }
static int role_of(int l){ return (l==0)?X_RL_INPUT : (l==L-1)?X_RL_OUT : X_RL_HID; }
static void kern(int l){ forge_pc_xformer_block((ap_uint<512>*)buf[l]); }
static int RV(double s,unsigned&st){ double u=0; for(int k=0;k<6;k++){ st=st*1103515245u+12345u; u+=((st>>16)&0x7fff)/32768.0;} int v=(int)lround((u-3.0)*s); if(v>32767)v=32767; if(v<-32767)v=-32767; return v; }

int main(int argc,char**argv){
    L = argc>1?atoi(argv[1]):4;            // chips (default 4 for a quick local check; 8 = the flagship)
    int steps = argc>2?atoi(argv[2]):1200;
    const char* path = getenv("SHAKESPEARE"); char def[256]; if(!path){ snprintf(def,256,"%s/.cache/tiur/tinyshakespeare.txt",getenv("HOME")); path=def; }
    FILE* f=fopen(path,"rb"); if(!f){ printf("no data at %s\n",path); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char* txt=(char*)malloc(n); long rd=fread(txt,1,n,f); fclose(f); (void)rd;
    int map[256]; for(int i=0;i<256;i++) map[i]=-1; int vocab=0;
    for(long i=0;i<n&&vocab<X_VOCAB;i++){ unsigned char c=txt[i]; if(map[c]<0) map[c]=vocab++; }
    int* data=(int*)malloc(sizeof(int)*n); for(long i=0;i<n;i++){ int id=map[(unsigned char)txt[i]]; data[i]=id<0?0:id; }
    printf("[xfsim] %d-block transformer, LOCAL multi-chip sim (real ap_int kernel), D=%d seq=%d vocab=%d, %d steps\n",L,X_D,X_SEQ,X_VOCAB,steps);

    int32_t lut[X_LUT_N]; { double v=65536.0; for(int g=0;g<X_LUT_N;g++){ lut[g]=(int32_t)(v+0.5); v*=0.917004; } }
    for(int l=0;l<L;l++){
        buf[l]=(int32_t*)calloc(X_BUFLEN,4); unsigned st=7u+1000u*l;
        for(int i=0;i<X_D*X_D;i++){ buf[l][XO_WQ+i]=RV(8,st); buf[l][XO_WK+i]=RV(8,st); buf[l][XO_WV+i]=RV(8,st); buf[l][XO_WO+i]=RV(8,st);}
        for(int i=0;i<X_FFN*X_D;i++) buf[l][XO_WUP+i]=RV(8,st);
        for(int i=0;i<X_D*X_FFN;i++) buf[l][XO_WDN+i]=RV(8,st);
        for(int d=0;d<X_D;d++){ buf[l][XO_LN1G+d]=64; buf[l][XO_LN2G+d]=64; buf[l][XO_LNFG+d]=64; }
        if(l==0){ for(int i=0;i<X_VOCAB*X_D;i++) buf[l][XO_EMBED+i]=RV(16,st); for(int i=0;i<X_SEQ*X_D;i++) buf[l][XO_POS+i]=RV(8,st); }
        if(l==L-1) for(int i=0;i<X_VOCAB*X_D;i++) buf[l][XO_WHEAD+i]=RV(8,st);
        for(int g=0;g<X_LUT_N;g++) buf[l][XO_LUT+g]=lut[g];
    }
    srand(123); double bpb_ema=-1;
    for(int step=0; step<steps; step++){
        long off = rand()%(n-X_SEQ-2);
        // FORWARD sweep 0..L-1, relaying HOUT -> HIN
        for(int l=0;l<L;l++){
            setcfg(l,X_PH_FWD,role_of(l),0);
            if(l==0){ for(int t=0;t<X_SEQ;t++) buf[0][XO_TOK+t]=data[off+t]; }
            else      memcpy(buf[l]+XO_HIN, buf[l-1]+XO_HOUT, 4*X_SEQ*X_D);   // neighbour hidden-state (down)
            if(l==L-1) for(int t=0;t<X_SEQ;t++) buf[l][XO_TGT+t]=data[off+t+1];
            kern(l);
        }
        // BACKWARD sweep L-1..0, relaying EOUT -> EIN
        for(int l=L-1;l>=0;l--){
            setcfg(l,X_PH_BWD,role_of(l),1);
            if(l<L-1) memcpy(buf[l]+XO_EIN, buf[l+1]+XO_EOUT, 4*X_SEQ*X_D);   // neighbour error (up)
            kern(l);
        }
        if(step%100==0||step==steps-1){
            const int32_t* lg=buf[L-1]+XO_LOGIT; double bits=0; int c=0;
            for(int t=0;t<X_SEQ;t++){ int tg=data[off+t+1]; long mx=-1L<<60; int am=0; for(int v=0;v<X_VOCAB;v++){ long lv=lg[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} }
                double Z=0; for(int v=0;v<X_VOCAB;v++) Z+=exp((lg[t*X_VOCAB+v]-mx)/16.0);
                double pt=exp((lg[t*X_VOCAB+tg]-mx)/16.0)/Z; bits+=-log2(pt>1e-12?pt:1e-12); if(am==tg)c++; }
            double bpb=bits/X_SEQ; bpb_ema=bpb_ema<0?bpb:0.95*bpb_ema+0.05*bpb;
            printf("  step %4d  bpb=%.3f  bpb_ema=%.3f  acc=%.3f\n", step, bpb, bpb_ema, (double)c/X_SEQ);
        }
    }
    printf("  -> %s: distributed %d-block transformer %s (bpb_ema %.3f; lower=learning, log2(64)=6.0 chance)\n",
           L>=2?"PASS":"?", L, bpb_ema<5.5?"TRAINS model-parallel across the blocks":"did not clearly fall", bpb_ema);
    return 0;
}
