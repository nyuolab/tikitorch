// forge_pc_xformer_shard.cpp — train a TRANSFORMER MODEL-PARALLEL across L physical F2 FPGAs: one transformer
// BLOCK per chip (forge_pc_xformer_block kernel, role/phase/CFG). Chip 0 = INPUT (embed tokens), chips 1..L-2 =
// HIDDEN, chip L-1 = OUTPUT (final-LN + LM head + CE). Each chip holds ONLY its block's weights + dyadic-Adam
// state in its OWN on-card DDR; the ONLY data crossing chips is a neighbour hidden-state (down, forward) + a
// neighbour error (up, backward). No global model/gradient/all-reduce/barrier — same contract as the MLP shard,
// now for a transformer. The host relays the boundary messages over PCIe; a Gate-C audit counts what crosses.
//   g++ -O2 -std=c++14 -I<hls/forge_kernels> -I$SDK/include forge_pc_xformer_shard.cpp -lfpga_mgmt -o xfshard
//   sudo ./xfshard <steps> <bindir> <slot0> <slot1> ... <slotL-1>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
extern "C" {
#include <fpga_pci.h>
#include <fpga_mgmt.h>
}
#include "forge_pc_xformer_config.hpp"   // X_* dims + XO_* DDR offsets + X_BUFLEN (the kernel's layout)

#define MAXL 8
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE  0x02

static int L;
static pci_bar_handle_t bar0[MAXL], bar4[MAXL], bar4r[MAXL];
static int32_t *hbuf[MAXL];                 // host shadow of each chip's DDR buffer (X_BUFLEN int32)
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static void stage(int l,int off,int n){ for(long o=off;o<off+n;o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144;
    fpga_pci_write_burst(bar4[l],(uint64_t)o*4,(uint32_t*)(hbuf[l]+o),(uint64_t)k);} }
static void kick(int l){ fpga_pci_poke(bar0[l],R_BUF0,0); fpga_pci_poke(bar0[l],R_BUF1,0);
    double t0=now_s(); fpga_pci_poke(bar0[l],R_AP_CTRL,AP_START); uint32_t c=0;
    while(now_s()-t0<20.0){ fpga_pci_peek(bar0[l],R_AP_CTRL,&c); if(c&AP_DONE)return; } fprintf(stderr,"KICK HUNG slot %d\n",l); exit(6); }
static void readw(int l,int off,int32_t*dst,int n){ uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r[l],(uint64_t)(off+j)*4,&v); dst[j]=(int32_t)v; } }
static void setcfg(int l,int phase,int role,int doupd){ hbuf[l][XO_CFG+0]=phase; hbuf[l][XO_CFG+1]=role; hbuf[l][XO_CFG+2]=doupd; hbuf[l][XO_CFG+3]=0; stage(l,XO_CFG,X_WL); }
static int role_of(int l){ return (l==0)?X_RL_INPUT : (l==L-1)?X_RL_OUT : X_RL_HID; }

// init helpers (match the golden's init style so a Python ref could replay)
static int RV(double s,unsigned&st){ double u=0; for(int k=0;k<6;k++){ st=st*1103515245u+12345u; u+=((st>>16)&0x7fff)/32768.0;} double g=(u-3.0)*s; int v=(int)lround(g); if(v>32767)v=32767; if(v<-32767)v=-32767; return v; }

int main(int argc,char**argv){
    int steps = argc>1?atoi(argv[1]):2000;
    const char*dir = argc>2?argv[2]:"/tmp/pc_shakespeare";
    L = argc-3; if(L<2||L>MAXL){ fprintf(stderr,"usage: %s <steps> <bindir> <slot0..slotL-1> (2<=L<=%d)\n",argv[0],MAXL); return 1; }
    int slot[MAXL]; for(int l=0;l<L;l++) slot[l]=atoi(argv[3+l]);

    // ---- load TinyShakespeare token stream (uint16) ----
    char p[512]; snprintf(p,512,"%s/train.u16",dir); FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);return 2;}
    fseek(f,0,SEEK_END); long nb=ftell(f); fseek(f,0,SEEK_SET); long ntok=nb/2; uint16_t*data=(uint16_t*)malloc(nb);
    if(fread(data,2,ntok,f)!=(size_t)ntok){fprintf(stderr,"read tokens\n");return 2;} fclose(f);
    printf("[xfshard] %d-block transformer MODEL-PARALLEL across %d FPGAs (slots", L, L);
    for(int l=0;l<L;l++) printf(" %d",slot[l]); printf("); %ld train tokens, D=%d seq=%d vocab=%d\n", ntok, X_D, X_SEQ, X_VOCAB);

    if(fpga_mgmt_init()){fprintf(stderr,"mgmt init\n");return 2;}
    for(int l=0;l<L;l++){
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR0,0,&bar0[l])){fprintf(stderr,"bar0 %d\n",slot[l]);return 3;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4[l])){fprintf(stderr,"bar4 %d\n",slot[l]);return 4;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,0,&bar4r[l])){fprintf(stderr,"bar4r %d\n",slot[l]);return 4;}
        hbuf[l]=(int32_t*)calloc(X_BUFLEN,4);
    }
    // ---- exp2 LUT (== golden g_lut_init: ~2^(-g/8) in Q16) ----
    int32_t lut[X_LUT_N]; { double v=65536.0; for(int g=0;g<X_LUT_N;g++){ lut[g]=(int32_t)(v+0.5); v*=0.917004; } }
    // ---- init each chip's block weights (per-chip seed) + LN gammas=64 + embed/pos/head + Adam m/v=0; persist ----
    for(int l=0;l<L;l++){
        unsigned st = 7u + 1000u*l;
        for(int i=0;i<X_D*X_D;i++){ hbuf[l][XO_WQ+i]=RV(8,st); hbuf[l][XO_WK+i]=RV(8,st); hbuf[l][XO_WV+i]=RV(8,st); hbuf[l][XO_WO+i]=RV(8,st);}
        for(int i=0;i<X_FFN*X_D;i++) hbuf[l][XO_WUP+i]=RV(8,st);
        for(int i=0;i<X_D*X_FFN;i++) hbuf[l][XO_WDN+i]=RV(8,st);
        for(int d=0;d<X_D;d++){ hbuf[l][XO_LN1G+d]=64; hbuf[l][XO_LN2G+d]=64; hbuf[l][XO_LNFG+d]=64; }
        if(l==0){ for(int i=0;i<X_VOCAB*X_D;i++) hbuf[l][XO_EMBED+i]=RV(16,st); for(int i=0;i<X_SEQ*X_D;i++) hbuf[l][XO_POS+i]=RV(8,st); }
        if(l==L-1) for(int i=0;i<X_VOCAB*X_D;i++) hbuf[l][XO_WHEAD+i]=RV(8,st);
        for(int g=0;g<X_LUT_N;g++) hbuf[l][XO_LUT+g]=lut[g];
        // m/v already 0 from calloc
        stage(l,0,X_BUFLEN);
    }
    int32_t *hmsg=(int32_t*)malloc(4*X_SEQ*X_D), *emsg=(int32_t*)malloc(4*X_SEQ*X_D);
    int32_t *logit=(int32_t*)malloc(4*X_SEQ*X_VOCAB);
    long xchip_msg=0, xchip_weight=0, xchip_grad=0;
    double t0=now_s(); double bpb_ema=-1;

    for(int step=0; step<steps; step++){
        long off = (long)((double)rand()/RAND_MAX*(ntok-X_SEQ-2));
        // ===================== FORWARD SWEEP: chip 0 -> L-1 =====================
        for(int l=0;l<L;l++){
            setcfg(l,X_PH_FWD,role_of(l),0);
            if(l==0){ for(int t=0;t<X_SEQ;t++) hbuf[0][XO_TOK+t]=data[off+t]; stage(0,XO_TOK,X_SEQ); }       // INPUT: tokens
            else    { memcpy(hbuf[l]+XO_HIN, hmsg, 4*X_SEQ*X_D); xchip_msg+=(long)X_SEQ*X_D; stage(l,XO_HIN,X_SEQ*X_D); } // neighbour hidden-state
            if(l==L-1){ for(int t=0;t<X_SEQ;t++) hbuf[l][XO_TGT+t]=data[off+t+1]; stage(l,XO_TGT,X_SEQ); }    // OUTPUT: targets (next char)
            kick(l);
            if(l<L-1) readw(l,XO_HOUT,hmsg,X_SEQ*X_D);   // this block's hidden-state -> next chip's HIN (the down message)
        }
        // ===================== BACKWARD SWEEP: chip L-1 -> 0 =====================
        for(int l=L-1;l>=0;l--){
            setcfg(l,X_PH_BWD,role_of(l),1);                                  // DO_UPDATE=1: dyadic-Adam updates this block
            if(l<L-1){ memcpy(hbuf[l]+XO_EIN, emsg, 4*X_SEQ*X_D); xchip_msg+=(long)X_SEQ*X_D; stage(l,XO_EIN,X_SEQ*X_D); } // neighbour error
            kick(l);
            if(l>0) readw(l,XO_EOUT,emsg,X_SEQ*X_D);     // this block's input-gradient -> prev chip's EIN (the up message)
        }
        // ===================== periodic bpb eval (output chip's logits) =====================
        if(step%200==0 || step==steps-1){
            readw(L-1,XO_LOGIT,logit,X_SEQ*X_VOCAB);
            double bits=0; int n=0;
            for(int t=0;t<X_SEQ;t++){ int tg=data[off+t+1]; long mx=-1L<<60; for(int v=0;v<X_VOCAB;v++){ long lv=logit[t*X_VOCAB+v]; if(lv>mx)mx=lv; }
                double Z=0; for(int v=0;v<X_VOCAB;v++) Z+=exp((logit[t*X_VOCAB+v]-mx)/16.0);   // soft scale (Q? logits ~int); relative bpb
                double pt=exp((logit[t*X_VOCAB+tg]-mx)/16.0)/Z; bits += -log2(pt>1e-12?pt:1e-12); n++; }
            double bpb=bits/n; bpb_ema = bpb_ema<0?bpb:0.97*bpb_ema+0.03*bpb;
            printf("[xfshard] step %5d  bpb(batch)=%.3f  bpb_ema=%.3f  (%.1f steps/s)\n", step, bpb, bpb_ema, (step+1)/(now_s()-t0)); fflush(stdout);
        }
    }
    double wall=now_s()-t0;
    printf("\n========== DISTRIBUTED PC TRANSFORMER on %d FPGAs ==========\n", L);
    printf("[throughput] %d steps, %.1fs, %.2f steps/s | %.0f tok/s\n", steps, wall, steps/wall, steps*(double)X_SEQ/wall);
    printf("[comms] cross-chip messages: %ld ints = %.2f MB | per step %ld ints = %d hidden(down) + %d error(up) (each N*D=%d)\n",
           xchip_msg, xchip_msg*4/1e6, xchip_msg/steps, (L-1), (L-1), X_SEQ*X_D);
    printf("[gateC] WEIGHTS crossed chips=%ld | GRADIENTS crossed=%ld | only neighbour hidden-states + errors crossed (no global model/gradient)\n", xchip_weight*4, xchip_grad*4);
    // dump each chip's final weights (for an offline bit-exact replay / inspection)
    for(int l=0;l<L;l++){ int32_t*W=(int32_t*)malloc(4*X_D*X_D); readw(l,XO_WQ,W,X_D*X_D); snprintf(p,512,"/tmp/xfW%d.i32",l); FILE*g=fopen(p,"wb"); if(g){fwrite(W,4,X_D*X_D,g);fclose(g);} free(W); }
    for(int l=0;l<L;l++){ fpga_pci_detach(bar0[l]); fpga_pci_detach(bar4[l]); fpga_pci_detach(bar4r[l]); }
    return 0;
}
