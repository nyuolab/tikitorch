// forge_pc_block_tb.cpp — CSIM: orchestrate ONE distributed training step across 2 "chips" (2 buffers,
// each running the forge_pc_block kernel on its own layer) and compare the updated W0/W1 + the neighbour
// messages (AOUT0, AOUT1, EOUT1) to an int64 reference of the SAME 2-chip step, bit-for-bit. Proves the
// distributed block kernel is correct before any AFI build (Gate A in C-sim). Only AOUT (down) and EOUT
// (up) cross between buffers — never weights or a global gradient.
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ap_int.h>

#define TD 256
#define TN 32
#define TC 10
#define H1_BITS 7
#define LG_BITS 4
#define EDN_BITS 13
#define MCAP 7
#define PH_FWD 0
#define PH_BWD 1
#define RL_HIDDEN 0
#define RL_OUTPUT 1
#define WL 16
#define O_CFG  0
#define O_W    (O_CFG  + WL)
#define O_MOM  (O_W    + TD*TD)
#define O_AIN  (O_MOM  + TD*TD)
#define O_TGT  (O_AIN  + TN*TD)
#define O_EIN  (O_TGT  + TN*TD)
#define O_AOUT (O_EIN  + TN*TD)
#define O_MASK (O_AOUT + TN*TD)
#define O_EOUT (O_MASK + TN*TD)
#define BUFLEN (O_EOUT + TN*TD)

void forge_pc_block(ap_uint<512> *buffer);

static long rdyad(long v) {
    long a = v < 0 ? -v : v; if (a == 0) return 0; if (a > 32767) a = 32767;
    int msb = 0; for (int b = 15; b >= 1; b--) if ((a >> b) & 1) { msb = b; break; }
    if (msb >= 1 && ((a >> (msb - 1)) & 1)) msb++;
    long m = 1L << msb; return v < 0 ? -m : m;
}
static int rshift(long mx, int target) {                  // == kernel norm_shift (FIXED: hb+1-target)
    long a = mx < 0 ? -mx : mx; int hb = 0;
    for (int b = 47; b >= 1; b--) if ((a >> b) & 1) { hb = b; break; }
    int s = (hb + 1) - target; return s > 0 ? s : 0;
}
static long i16(long v){ return (int16_t)v; }              // kernel casts to ap_int<16> (truncate)
static int sgn(long v){ return v>0?1:(v<0?-1:0); }

static void setcfg(int32_t*buf,int phase,int role,int doupd){ buf[O_CFG+0]=phase;buf[O_CFG+1]=role;buf[O_CFG+2]=8;buf[O_CFG+3]=doupd; }

int main() {
    int32_t *b0 = (int32_t*)aligned_alloc(64, sizeof(int32_t)*BUFLEN);   // chip0: layer 0 (hidden)
    int32_t *b1 = (int32_t*)aligned_alloc(64, sizeof(int32_t)*BUFLEN);   // chip1: layer 1 (output)
    for (int i=0;i<BUFLEN;i++){ b0[i]=0; b1[i]=0; }
    srand(7);
    for (int i=0;i<TD*TD;i++){ b0[O_W+i]=(rand()%9)-4; b1[O_W+i]=(rand()%9)-4; b0[O_MOM+i]=(rand()%5)-2; b1[O_MOM+i]=(rand()%5)-2; }
    static long X[TN][TD]; static int lab[TN];
    for (int n=0;n<TN;n++){ for(int i=0;i<TD;i++){ X[n][i]=rand()%17; b0[O_AIN+n*TD+i]=X[n][i]; } lab[n]=rand()%TC; }

    // ---- int64 reference of the 2-chip step (snapshot W first) ----
    static long W0[TD*TD],W1[TD*TD],M0[TD*TD],M1[TD*TD];
    for(int i=0;i<TD*TD;i++){W0[i]=b0[O_W+i];W1[i]=b1[O_W+i];M0[i]=b0[O_MOM+i];M1[i]=b1[O_MOM+i];}
    // chip0 fwd (hidden): pre0, mask0, h=AOUT0
    static long pre0[TN][TD]; static int mask0[TN][TD]; long mxh=0;
    for(int n=0;n<TN;n++)for(int o=0;o<TD;o++){long acc=0;for(int i=0;i<TD;i++)acc+=rdyad(W0[o*TD+i])*X[n][i];pre0[n][o]=acc;mask0[n][o]=acc>0;long r=acc>0?acc:0;if(r>mxh)mxh=r;}
    int sh0=rshift(mxh,H1_BITS);
    static long H[TN][TD];
    for(int n=0;n<TN;n++)for(int o=0;o<TD;o++){long r=pre0[n][o]>0?pre0[n][o]:0;H[n][o]=i16(r>>sh0);}
    // chip1 fwd (output): logits=AOUT1
    static long lg[TN][TC]; long mxl=0;
    for(int n=0;n<TN;n++)for(int c=0;c<TC;c++){long acc=0;for(int i=0;i<TD;i++)acc+=rdyad(W1[c*TD+i])*H[n][i];lg[n][c]=acc;long al=acc<0?-acc:acc;if(al>mxl)mxl=al;}
    int sh1=rshift(mxl,LG_BITS);
    static long A1[TN][TC];
    for(int n=0;n<TN;n++)for(int c=0;c<TC;c++)A1[n][c]=i16(lg[n][c]>>sh1);
    // chip1 bwd (output): e1, gW1, e_down1 -> EOUT1, update W1
    static long e1[TN][TC];
    for(int n=0;n<TN;n++)for(int c=0;c<TC;c++)e1[n][c]=i16(A1[n][c]-(c==lab[n]?8:0));
    static long edn1[TN][TD]; long mxd1=0;
    for(int n=0;n<TN;n++)for(int i=0;i<TD;i++){long acc=0;for(int c=0;c<TC;c++)acc+=e1[n][c]*rdyad(W1[c*TD+i]);edn1[n][i]=acc;long ad=acc<0?-acc:acc;if(ad>mxd1)mxd1=ad;}
    int she1=rshift(mxd1,EDN_BITS);
    static long EOUT1[TN][TD];
    for(int n=0;n<TN;n++)for(int i=0;i<TD;i++)EOUT1[n][i]=i16(edn1[n][i]>>she1);
    static long gW1[TC][TD];
    for(int c=0;c<TC;c++)for(int i=0;i<TD;i++){long acc=0;for(int n=0;n<TN;n++)acc+=rdyad(e1[n][c])*rdyad(H[n][i]);gW1[c][i]=acc;}
    static long rW1[TD*TD]; for(int i=0;i<TD*TD;i++)rW1[i]=W1[i];
    for(int c=0;c<TC;c++)for(int i=0;i<TD;i++){int idx=c*TD+i;long m=M1[idx]+sgn(gW1[c][i]);m=m>MCAP?MCAP:(m<-MCAP?-MCAP:m);rW1[idx]=i16(W1[idx]-sgn(m));}
    // chip0 bwd (hidden): EIN0=EOUT1, delta0, gW0, update W0
    static long d0[TN][TD];
    for(int n=0;n<TN;n++)for(int o=0;o<TD;o++)d0[n][o]=mask0[n][o]?EOUT1[n][o]:0;
    static long gW0[TD*TD];
    for(int o=0;o<TD;o++)for(int i=0;i<TD;i++){long acc=0;for(int n=0;n<TN;n++)acc+=rdyad(d0[n][o])*rdyad(X[n][i]);gW0[o*TD+i]=acc;}
    static long rW0[TD*TD];
    for(int o=0;o<TD;o++)for(int i=0;i<TD;i++){int idx=o*TD+i;long m=M0[idx]+sgn(gW0[idx]);m=m>MCAP?MCAP:(m<-MCAP?-MCAP:m);rW0[idx]=i16(W0[idx]-sgn(m));}

    // ---- run the kernel over the 4 kicks (fwd0, fwd1, bwd1, bwd0) ----
    setcfg(b0,PH_FWD,RL_HIDDEN,0); forge_pc_block((ap_uint<512>*)b0);     // chip0 forward
    for(int n=0;n<TN;n++)for(int i=0;i<TD;i++) b1[O_AIN+n*TD+i]=b0[O_AOUT+n*TD+i];   // AOUT0 -> chip1 AIN (downstream msg)
    setcfg(b1,PH_FWD,RL_OUTPUT,0); forge_pc_block((ap_uint<512>*)b1);     // chip1 forward
    for(int n=0;n<TN;n++){ for(int c=0;c<TD;c++) b1[O_TGT+n*TD+c]=0; b1[O_TGT+n*TD+lab[n]]=8; }  // stage MSE target
    setcfg(b1,PH_BWD,RL_OUTPUT,1); forge_pc_block((ap_uint<512>*)b1);     // chip1 backward + update W1
    for(int n=0;n<TN;n++)for(int i=0;i<TD;i++) b0[O_EIN+n*TD+i]=b1[O_EOUT+n*TD+i];   // EOUT1 -> chip0 EIN (upstream msg)
    setcfg(b0,PH_BWD,RL_HIDDEN,1); forge_pc_block((ap_uint<512>*)b0);     // chip0 backward + update W0

    // ---- compare ----
    long dW0=0,dW1=0,nW0=0,nW1=0,dA0=0,dA1=0,dE=0;
    for(int i=0;i<TD*TD;i++){ long a=labs((long)b0[O_W+i]-rW0[i]); if(a>dW0)dW0=a; if(a)nW0++;
                              long b=labs((long)b1[O_W+i]-rW1[i]); if(b>dW1)dW1=b; if(b)nW1++; }
    for(int n=0;n<TN;n++)for(int o=0;o<TD;o++){ long a=labs((long)b0[O_AOUT+n*TD+o]-H[n][o]); if(a>dA0)dA0=a;
                                                long e=labs((long)b1[O_EOUT+n*TD+o]-EOUT1[n][o]); if(e>dE)dE=e; }
    for(int n=0;n<TN;n++)for(int c=0;c<TC;c++){ long a=labs((long)b1[O_AOUT+n*TD+c]-A1[n][c]); if(a>dA1)dA1=a; }
    printf("forge_pc_block CSIM (2-chip step, D=%d N=%d C=%d): sh0=%d sh1=%d she1=%d\n",TD,TN,TC,sh0,sh1,she1);
    printf("  AOUT0(h) max|diff|=%ld  AOUT1(logits) max|diff|=%ld  EOUT1(msg) max|diff|=%ld\n",dA0,dA1,dE);
    printf("  W0' max|diff|=%ld (%ld/%d)  W1' max|diff|=%ld (%ld/%d)\n",dW0,nW0,TD*TD,dW1,nW1,TC*TD);
    int ok=(dW0|dW1|dA0|dA1|dE)==0;
    printf("  -> %s\n", ok ? "BIT-EXACT (distributed 2-chip step == int64 reference; only AOUT/EOUT crossed)" : "MISMATCH");
    free(b0); free(b1);
    return ok?0:1;
}
