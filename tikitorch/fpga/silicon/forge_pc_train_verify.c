// forge_pc_train_verify.c — ON-SILICON bit-exact verify of ONE forge_pc_train_step kick (like
// forge_pc_ddr_verify did for the backward). Stage known W/mom/x/tgt to DDR, kick once, read back the
// updated W0/W1/mom0/mom1, compare to an int64 reference of the same step. BIT-EXACT => the silicon RTL
// matches the C (no C/RTL divergence) -> a stuck training run is the host eval/dynamics, not the kernel.
// MISMATCH => a C/RTL divergence in the on-chip train-step (the real bug).
//   gcc -O2 -I$SDK/include forge_pc_train_verify.c -lfpga_mgmt -o v ; sudo ./v <slot>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>
#define D 256
#define N 32
#define C 10
#define H1_BITS 7
#define LG_BITS 4
#define MCAP 7
#define O_W0 0
#define O_W1 (O_W0+D*D)
#define O_MOM0 (O_W1+D*D)
#define O_MOM1 (O_MOM0+D*D)
#define O_X (O_MOM1+D*D)
#define O_TGT (O_X+N*D)
#define BUFLEN (O_TGT+N*D)
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE 0x02

static long rdyad(long v){ long a=v<0?-v:v; if(!a)return 0; if(a>32767)a=32767; int m=0; for(int b=15;b>=1;b--)if((a>>b)&1){m=b;break;} if(m>=1&&((a>>(m-1))&1))m++; long p=1L<<m; return v<0?-p:p; }
static int rshift(long mx,int t){ long a=mx<0?-mx:mx; int h=0; for(int b=47;b>=1;b--)if((a>>b)&1){h=b;break;} int s=h-t; return s>0?s:0; }
static long cl16(long v){ return v>32767?32767:(v<-32768?-32768:v); }
static int sgn(long v){ return v>0?1:(v<0?-1:0); }
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc,char**argv){
    int slot=argc>1?atoi(argv[1]):0;
    int32_t*h=aligned_alloc(64,4*BUFLEN); srand(7);
    for(int i=0;i<D*D;i++){ h[O_W0+i]=(rand()%9)-4; h[O_W1+i]=(rand()%9)-4; h[O_MOM0+i]=(rand()%5)-2; h[O_MOM1+i]=(rand()%5)-2; }
    for(int i=0;i<N*D;i++){ h[O_X+i]=rand()%17; h[O_TGT+i]=0; }
    for(int n=0;n<N;n++) h[O_TGT+n*D+(rand()%C)]=8;

    // ---- int64 reference of one step (snapshot inputs) ----
    static long W0[D*D],W1[D*D],M0[D*D],M1[D*D]; static long X[N][D],T[N][D];
    for(int i=0;i<D*D;i++){W0[i]=h[O_W0+i];W1[i]=h[O_W1+i];M0[i]=h[O_MOM0+i];M1[i]=h[O_MOM1+i];}
    for(int n=0;n<N;n++)for(int i=0;i<D;i++){X[n][i]=h[O_X+n*D+i];T[n][i]=h[O_TGT+n*D+i];}
    static long pre0[N][D]; static int mask0[N][D]; long mxh=0;
    for(int n=0;n<N;n++)for(int o=0;o<D;o++){long a=0;for(int i=0;i<D;i++)a+=rdyad(W0[o*D+i])*X[n][i];pre0[n][o]=a;mask0[n][o]=a>0;long r=a>0?a:0;if(r>mxh)mxh=r;}
    int sh0=rshift(mxh,H1_BITS); static long h1[N][D];
    for(int n=0;n<N;n++)for(int o=0;o<D;o++){long r=pre0[n][o]>0?pre0[n][o]:0;h1[n][o]=cl16(r>>sh0);}
    static long lg[N][C]; long mxl=0;
    for(int n=0;n<N;n++)for(int c=0;c<C;c++){long a=0;for(int i=0;i<D;i++)a+=rdyad(W1[c*D+i])*h1[n][i];lg[n][c]=a;long al=a<0?-a:a;if(al>mxl)mxl=al;}
    int sh1=rshift(mxl,LG_BITS); static long e1[N][C];
    for(int n=0;n<N;n++)for(int c=0;c<C;c++)e1[n][c]=cl16((lg[n][c]>>sh1)-T[n][c]);
    static long gW1[C][D];
    for(int c=0;c<C;c++)for(int i=0;i<D;i++){long a=0;for(int n=0;n<N;n++)a+=rdyad(e1[n][c])*rdyad(h1[n][i]);gW1[c][i]=cl16(a);}
    static long edn[N][D]; long mxd=0;
    for(int n=0;n<N;n++)for(int i=0;i<D;i++){long a=0;for(int c=0;c<C;c++)a+=e1[n][c]*rdyad(W1[c*D+i]);long d=mask0[n][i]?a:0;edn[n][i]=d;long ad=d<0?-d:d;if(ad>mxd)mxd=ad;}
    int shd=rshift(mxd,13); static long d0[N][D];
    for(int n=0;n<N;n++)for(int o=0;o<D;o++)d0[n][o]=cl16(edn[n][o]>>shd);
    static long gW0[D*D];
    for(int o=0;o<D;o++)for(int i=0;i<D;i++){long a=0;for(int n=0;n<N;n++)a+=rdyad(d0[n][o])*rdyad(X[n][i]);gW0[o*D+i]=a;}
    static long rW0[D*D],rW1[D*D],rM0[D*D],rM1[D*D];
    for(int o=0;o<D;o++)for(int i=0;i<D;i++){int idx=o*D+i;long m=M0[idx]+sgn(gW0[idx]);m=m>MCAP?MCAP:(m<-MCAP?-MCAP:m);rM0[idx]=m;rW0[idx]=cl16(W0[idx]-sgn(m));}
    for(int i=0;i<D*D;i++){rW1[i]=W1[i];rM1[i]=M1[i];}
    for(int c=0;c<C;c++)for(int i=0;i<D;i++){int idx=c*D+i;long m=M1[idx]+sgn(gW1[c][i]);m=m>MCAP?MCAP:(m<-MCAP?-MCAP:m);rM1[idx]=m;rW1[idx]=cl16(W1[idx]-sgn(m));}

    // ---- run one kick on silicon, read back, compare ----
    if(fpga_mgmt_init()){fprintf(stderr,"mgmt\n");return 2;}
    pci_bar_handle_t b0=PCI_BAR_HANDLE_INIT,b4=PCI_BAR_HANDLE_INIT,b4r=PCI_BAR_HANDLE_INIT;
    fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR0,0,&b0);
    fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&b4);
    fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR4,0,&b4r);
    for(long o=0;o<BUFLEN;o+=262144){long k=(BUFLEN-o<262144)?(BUFLEN-o):262144;fpga_pci_write_burst(b4,(uint64_t)o*4,(uint32_t*)(h+o),(uint64_t)k);}
    fpga_pci_poke(b0,R_BUF0,0);fpga_pci_poke(b0,R_BUF1,0);fpga_pci_poke(b0,R_AP_CTRL,AP_START);
    double t0=now_s();uint32_t cc=0;int done=0;while(now_s()-t0<10){fpga_pci_peek(b0,R_AP_CTRL,&cc);if(cc&AP_DONE){done=1;break;}}
    if(!done){fprintf(stderr,"HUNG\n");return 6;} printf("[v] AP_DONE %.2fms\n",(now_s()-t0)*1e3);
    int32_t*w0=malloc(4*D*D),*w1=malloc(4*D*D),*m0=malloc(4*D*D),*m1=malloc(4*D*D); uint32_t v;
    for(int j=0;j<D*D;j++){fpga_pci_peek(b4r,(uint64_t)(O_W0+j)*4,&v);w0[j]=v;}
    for(int j=0;j<D*D;j++){fpga_pci_peek(b4r,(uint64_t)(O_W1+j)*4,&v);w1[j]=v;}
    for(int j=0;j<D*D;j++){fpga_pci_peek(b4r,(uint64_t)(O_MOM0+j)*4,&v);m0[j]=v;}
    for(int j=0;j<D*D;j++){fpga_pci_peek(b4r,(uint64_t)(O_MOM1+j)*4,&v);m1[j]=v;}
    long dw0=0,dw1=0,dm0=0,dm1=0,n0=0;
    for(int j=0;j<D*D;j++){long a=labs(w0[j]-rW0[j]);if(a>dw0)dw0=a;if(a)n0++;long b=labs(w1[j]-rW1[j]);if(b>dw1)dw1=b;long c=labs(m0[j]-rM0[j]);if(c>dm0)dm0=c;long e=labs(m1[j]-rM1[j]);if(e>dm1)dm1=e;}
    printf("[v] sh0=%d sh1=%d shd=%d\n",sh0,sh1,shd);
    printf("[v] max|W0-ref|=%ld (%ld differ) max|W1-ref|=%ld max|mom0|=%ld max|mom1|=%ld\n",dw0,n0,dw1,dm0,dm1);
    int ok=(dw0|dw1|dm0|dm1)==0;
    printf("[v] -> %s\n", ok?"BIT-EXACT ON SILICON (kernel step == ref; training stuck is NOT the kernel)":"MISMATCH -> C/RTL divergence in the on-chip train-step");
    return ok?0:1;
}
