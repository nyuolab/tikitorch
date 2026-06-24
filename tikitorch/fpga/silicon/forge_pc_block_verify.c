// forge_pc_block_verify.c — localize the on-silicon stuck-at-chance bug for forge_pc_block. Runs ONE chip
// (slot 0, HIDDEN role) and checks each link of the integration the csim cannot test:
//   (A) stage W -> DDR -> read W back: round-trips bit-exact? (the stage/readback path)
//   (B) FWD kick: AOUT == int64 forward(W,x)? (weights loaded + forward compute on silicon)
//   (C) BWD kick (DO_UPDATE=1): does W CHANGE in DDR and match the int64 SignSGD ref? (update persists)
//   (D) CFG round-trip: read the CFG word back.
//   gcc -O2 -I$SDK/include forge_pc_block_verify.c -lfpga_mgmt -o forge_pc_block_verify ; sudo ./forge_pc_block_verify 0
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

#define D 256
#define N 32
#define C 10
#define H1_BITS 7
#define EDN_BITS 13
#define MCAP 7
#define WL 16
#define O_CFG  0
#define O_W    (O_CFG  + WL)
#define O_MOM  (O_W    + D*D)
#define O_AIN  (O_MOM  + D*D)
#define O_TGT  (O_AIN  + N*D)
#define O_EIN  (O_TGT  + N*D)
#define O_AOUT (O_EIN  + N*D)
#define O_MASK (O_AOUT + N*D)
#define O_EOUT (O_MASK + N*D)
#define BUFLEN (O_EOUT + N*D)
#define PH_FWD 0
#define PH_BWD 1
#define RL_HIDDEN 0
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE  0x02

static long rdyad(long v){ long a=v<0?-v:v; if(!a)return 0; if(a>32767)a=32767; int m=0; for(int b=15;b>=1;b--)if((a>>b)&1){m=b;break;} if(m>=1&&((a>>(m-1))&1))m++; long p=1L<<m; return v<0?-p:p; }
static int rsh(long mx,int t){ long a=mx<0?-mx:mx; int h=0; for(int b=47;b>=1;b--)if((a>>b)&1){h=b;break;} int s=(h+1)-t; return s>0?s:0; }
static long i16(long v){ return (int16_t)v; }
static int sgn(long v){ return v>0?1:(v<0?-1:0); }

static pci_bar_handle_t bar0,bar4,bar4r; static int32_t*hbuf;
static void stage(int off,int n){ for(long o=off;o<off+n;o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144; fpga_pci_write_burst(bar4,(uint64_t)o*4,(uint32_t*)(hbuf+o),(uint64_t)k);} }
static void kick(){ fpga_pci_poke(bar0,R_BUF0,0); fpga_pci_poke(bar0,R_BUF1,0); fpga_pci_poke(bar0,R_AP_CTRL,AP_START); uint32_t c=0; for(int i=0;i<2000000;i++){ fpga_pci_peek(bar0,R_AP_CTRL,&c); if(c&AP_DONE)return; } fprintf(stderr,"KICK HUNG\n"); exit(6); }
static void readw(int off,int32_t*dst,int n){ uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r,(uint64_t)(off+j)*4,&v); dst[j]=(int32_t)v; } }
static void setcfg(int ph,int role,int up){ hbuf[O_CFG+0]=ph; hbuf[O_CFG+1]=role; hbuf[O_CFG+2]=8; hbuf[O_CFG+3]=up; stage(O_CFG,WL); }

int main(int argc,char**argv){
    int slot=argc>1?atoi(argv[1]):0;
    if(fpga_mgmt_init()){fprintf(stderr,"mgmt\n");return 2;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR0,0,&bar0)){fprintf(stderr,"bar0\n");return 3;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4)){fprintf(stderr,"bar4\n");return 4;}
    if(fpga_pci_attach(slot,FPGA_APP_PF,APP_PF_BAR4,0,&bar4r)){fprintf(stderr,"bar4r\n");return 4;}
    hbuf=calloc(BUFLEN,4);
    srand(7);
    static long W0[D*D],M0[D*D],X[N][D],EI[N][D];
    for(int i=0;i<D*D;i++){ W0[i]=(rand()%9)-4; M0[i]=(rand()%5)-2; hbuf[O_W+i]=W0[i]; hbuf[O_MOM+i]=M0[i]; }
    for(int n=0;n<N;n++)for(int i=0;i<D;i++){ X[n][i]=rand()%17; EI[n][i]=(rand()%4001)-2000; hbuf[O_AIN+n*D+i]=X[n][i]; }
    stage(0,BUFLEN);

    // (A) stage->DDR->readback round-trip of W
    int32_t*Wb=malloc(4*D*D); readw(O_W,Wb,D*D);
    long dwr=0,nzw=0; for(int i=0;i<D*D;i++){ long a=labs((long)Wb[i]-W0[i]); if(a>dwr)dwr=a; if(Wb[i])nzw++; }
    printf("[A] W stage->DDR->readback: max|diff|=%ld, nonzero=%ld/%d  -> %s\n", dwr,nzw,D*D, dwr==0?"ROUND-TRIP OK":"READBACK/STAGE BROKEN");

    // (B) FWD kick -> AOUT vs int64 forward
    setcfg(PH_FWD,RL_HIDDEN,0); stage(O_AIN,N*D); kick();
    int32_t*AO=malloc(4*N*D); readw(O_AOUT,AO,N*D);
    static long pre[N][D]; long mxh=0;
    for(int n=0;n<N;n++)for(int o=0;o<D;o++){ long acc=0; for(int i=0;i<D;i++) acc+=rdyad(W0[o*D+i])*X[n][i]; pre[n][o]=acc; long r=acc>0?acc:0; if(r>mxh)mxh=r; }
    int sh=rsh(mxh,H1_BITS); long dao=0,nzao=0;
    for(int n=0;n<N;n++)for(int o=0;o<D;o++){ long h=i16((pre[n][o]>0?pre[n][o]:0)>>sh); long a=labs((long)AO[n*D+o]-h); if(a>dao)dao=a; if(AO[n*D+o])nzao++; }
    printf("[B] FWD AOUT vs ref: max|diff|=%ld (sh=%d), nonzero AOUT=%ld/%d  -> %s\n", dao,sh,nzao,N*D, dao==0?"FORWARD OK":"FORWARD/READ BROKEN");

    // (C) BWD kick (update) -> W changed + matches SignSGD ref
    for(int n=0;n<N;n++)for(int i=0;i<D;i++) hbuf[O_EIN+n*D+i]=EI[n][i];
    setcfg(PH_BWD,RL_HIDDEN,1); stage(O_EIN,N*D); kick();
    int32_t*Wa=malloc(4*D*D); readw(O_W,Wa,D*D);
    // int64 ref: mask from fwd pre>0; delta=EIN*mask; gW0=sum_n rdyad(delta)*rdyad(X); W-=sign(mom+sign(gW))
    static long gW[D*D];
    for(int o=0;o<D;o++)for(int i=0;i<D;i++){ long acc=0; for(int n=0;n<N;n++){ long d=(pre[n][o]>0)?EI[n][o]:0; acc+=rdyad(d)*rdyad(X[n][i]); } gW[o*D+i]=acc; }
    long dWu=0,changed=0,refchg=0;
    for(int o=0;o<D;o++)for(int i=0;i<D;i++){ int idx=o*D+i; long m=M0[idx]+sgn(gW[idx]); m=m>MCAP?MCAP:(m<-MCAP?-MCAP:m); long rW=i16(W0[idx]-sgn(m));
        long a=labs((long)Wa[idx]-rW); if(a>dWu)dWu=a; if(Wa[idx]!=W0[idx])changed++; if(rW!=W0[idx])refchg++; }
    printf("[C] BWD update: W changed on silicon=%ld, ref expects changed=%ld, max|W'-ref|=%ld  -> %s\n",
           changed,refchg,dWu, (dWu==0&&changed>0)?"UPDATE OK (persists, bit-exact)":(changed==0?"NO UPDATE (persistence/DO_UPDATE broken)":"UPDATE DIVERGES"));

    // (D) CFG round-trip
    int32_t cfg[4]; readw(O_CFG,cfg,4);
    printf("[D] CFG readback after BWD: phase=%d role=%d tgt=%d update=%d (expected 1 0 8 1)\n", cfg[0],cfg[1],cfg[2],cfg[3]);

    printf("VERDICT: A=%s B=%s C=%s\n", dwr==0?"ok":"FAIL", dao==0?"ok":"FAIL", (dWu==0&&changed>0)?"ok":"FAIL");
    fpga_pci_detach(bar0); fpga_pci_detach(bar4); fpga_pci_detach(bar4r);
    return 0;
}
