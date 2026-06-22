// forge_pc_train_step_tb.cpp — CSIM: run the actual forge_pc_train_step kernel for ONE training step and
// compare the updated W0/W1/mom0/mom1 to an int64 reference of the SAME step (dyad + max-shift norm + MSE
// + cascade + SignSGD-momentum), bit-for-bit. Validates the fused train-step C++ before the AFI build.
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ap_int.h>

#define TD 256
#define TN 32
#define TC 10
#define H1_BITS 7
#define LG_BITS 4
#define MCAP 7
#define O_W0   0
#define O_W1   (O_W0   + TD*TD)
#define O_MOM0 (O_W1   + TD*TD)
#define O_MOM1 (O_MOM0 + TD*TD)
#define O_X    (O_MOM1 + TD*TD)
#define O_TGT  (O_X    + TN*TD)
#define BUFLEN (O_TGT  + TN*TD)

void forge_pc_train_step(ap_uint<512> *buffer);

// reference dyad (== kernel: table-free 1.5x); returns sign*2^e
static long rdyad(long v) {
    long a = v < 0 ? -v : v; if (a == 0) return 0; if (a > 32767) a = 32767;
    int msb = 0; for (int b = 15; b >= 1; b--) if ((a >> b) & 1) { msb = b; break; }
    if (msb >= 1 && ((a >> (msb - 1)) & 1)) msb++;
    long m = 1L << msb; return v < 0 ? -m : m;
}
static int rshift(long mx, int target) {                  // == kernel norm_shift
    long a = mx < 0 ? -mx : mx; int hb = 0;
    for (int b = 47; b >= 1; b--) if ((a >> b) & 1) { hb = b; break; }
    int s = hb - target; return s > 0 ? s : 0;
}
static long clampi16(long v){ return v>32767?32767:(v<-32768?-32768:v); }
static int sgn(long v){ return v>0?1:(v<0?-1:0); }

int main() {
    int32_t *buf = (int32_t *)aligned_alloc(64, sizeof(int32_t) * BUFLEN);
    srand(7);
    for (int i = 0; i < TD*TD; i++) { buf[O_W0+i]=(rand()%9)-4; buf[O_W1+i]=(rand()%9)-4; buf[O_MOM0+i]=(rand()%5)-2; buf[O_MOM1+i]=(rand()%5)-2; }
    for (int i = 0; i < TN*TD; i++) { buf[O_X+i]=rand()%17; buf[O_TGT+i]=0; }
    for (int n = 0; n < TN; n++) buf[O_TGT + n*TD + (rand()%TC)] = 8;     // MSE target = 8 at the true class

    // ---- int64 reference of one step (snapshot inputs first) ----
    static long W0[TD*TD],W1[TD*TD],M0[TD*TD],M1[TD*TD]; static long X[TN][TD],T[TN][TD];
    for (int i=0;i<TD*TD;i++){W0[i]=buf[O_W0+i];W1[i]=buf[O_W1+i];M0[i]=buf[O_MOM0+i];M1[i]=buf[O_MOM1+i];}
    for (int n=0;n<TN;n++)for(int i=0;i<TD;i++){X[n][i]=buf[O_X+n*TD+i];T[n][i]=buf[O_TGT+n*TD+i];}
    // forward layer 0
    static long pre0[TN][TD]; static int mask0[TN][TD]; long mxh=0;
    for(int n=0;n<TN;n++)for(int o=0;o<TD;o++){long acc=0;for(int i=0;i<TD;i++)acc+=rdyad(W0[o*TD+i])*X[n][i];pre0[n][o]=acc;mask0[n][o]=acc>0;long r=acc>0?acc:0;if(r>mxh)mxh=r;}
    int sh0=rshift(mxh,H1_BITS);
    static long h1[TN][TD];
    for(int n=0;n<TN;n++)for(int o=0;o<TD;o++){long r=pre0[n][o]>0?pre0[n][o]:0;h1[n][o]=clampi16(r>>sh0);}
    // forward layer 1 (readout) + maxshift logits
    static long lg[TN][TC]; long mxl=0;
    for(int n=0;n<TN;n++)for(int c=0;c<TC;c++){long acc=0;for(int i=0;i<TD;i++)acc+=rdyad(W1[c*TD+i])*h1[n][i];lg[n][c]=acc;long al=acc<0?-acc:acc;if(al>mxl)mxl=al;}
    int sh1=rshift(mxl,LG_BITS);
    static long e1[TN][TC];
    for(int n=0;n<TN;n++)for(int c=0;c<TC;c++)e1[n][c]=clampi16((lg[n][c]>>sh1)-T[n][c]);
    // backward
    static long gW1[TC][TD];
    for(int c=0;c<TC;c++)for(int i=0;i<TD;i++){long acc=0;for(int n=0;n<TN;n++)acc+=rdyad(e1[n][c])*rdyad(h1[n][i]);gW1[c][i]=clampi16(acc);}
    static long edn[TN][TD]; long mxd=0;
    for(int n=0;n<TN;n++)for(int i=0;i<TD;i++){long acc=0;for(int c=0;c<TC;c++)acc+=e1[n][c]*rdyad(W1[c*TD+i]);long d=mask0[n][i]?acc:0;edn[n][i]=d;long ad=d<0?-d:d;if(ad>mxd)mxd=ad;}
    int shd=rshift(mxd,13);
    static long d0[TN][TD];
    for(int n=0;n<TN;n++)for(int o=0;o<TD;o++)d0[n][o]=clampi16(edn[n][o]>>shd);
    static long gW0[TD*TD];
    for(int o=0;o<TD;o++)for(int i=0;i<TD;i++){long acc=0;for(int n=0;n<TN;n++)acc+=rdyad(d0[n][o])*rdyad(X[n][i]);gW0[o*TD+i]=acc;}
    // update (reference): mom += sign(g) clamped; W -= sign(mom)
    static long rW0[TD*TD],rW1[TD*TD],rM0[TD*TD],rM1[TD*TD];
    for(int o=0;o<TD;o++)for(int i=0;i<TD;i++){int idx=o*TD+i;long m=M0[idx]+sgn(gW0[idx]);m=m>MCAP?MCAP:(m<-MCAP?-MCAP:m);rM0[idx]=m;rW0[idx]=clampi16(W0[idx]-sgn(m));}
    for(int i=0;i<TD*TD;i++){rW1[i]=W1[i];rM1[i]=M1[i];}                  // rows>=C untouched
    for(int c=0;c<TC;c++)for(int i=0;i<TD;i++){int idx=c*TD+i;long m=M1[idx]+sgn(gW1[c][i]);m=m>MCAP?MCAP:(m<-MCAP?-MCAP:m);rM1[idx]=m;rW1[idx]=clampi16(W1[idx]-sgn(m));}

    // ---- run the kernel + compare ----
    forge_pc_train_step((ap_uint<512>*)buf);
    long dW0=0,dW1=0,dM0=0,dM1=0;
    for(int i=0;i<TD*TD;i++){
        long a=labs((long)buf[O_W0+i]-rW0[i]); if(a>dW0)dW0=a;
        long b=labs((long)buf[O_W1+i]-rW1[i]); if(b>dW1)dW1=b;
        long c=labs((long)buf[O_MOM0+i]-rM0[i]); if(c>dM0)dM0=c;
        long d=labs((long)buf[O_MOM1+i]-rM1[i]); if(d>dM1)dM1=d;
    }
    printf("forge_pc_train_step CSIM (D=%d N=%d C=%d): one step\n", TD, TN, TC);
    printf("  sh0=%d sh1=%d shd=%d\n", sh0, sh1, shd);
    printf("  max|W0-ref|=%ld  max|W1-ref|=%ld  max|mom0-ref|=%ld  max|mom1-ref|=%ld\n", dW0,dW1,dM0,dM1);
    int ok = (dW0|dW1|dM0|dM1)==0;
    printf("  -> %s\n", ok ? "BIT-EXACT (kernel step == int64 reference)" : "MISMATCH");
    free(buf);
    return ok ? 0 : 1;
}
