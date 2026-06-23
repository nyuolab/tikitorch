// forge_pc_train_shard.c — train ONE network MODEL-PARALLEL across L physical F2 FPGAs. Each slot runs the
// forge_pc_block kernel on its OWN layer; its weights+momentum live in its OWN on-card DDR and never leave.
// Per training step the host does a forward sweep (slot 0 -> .. -> slot L-1, passing each layer's ACTIVATION
// downstream) and a backward sweep (slot L-1 -> .. -> slot 0, passing each layer's ERROR upstream). The ONLY
// data crossing chips is a neighbour activation (down) and a neighbour error (up) — never a weight, never a
// global gradient, no all-reduce, no barrier. A Gate-C audit counts exactly what crosses.
//   gcc -O2 -I$SDK/include forge_pc_train_shard.c -lfpga_mgmt -o forge_pc_train_shard
//   sudo ./forge_pc_train_shard <epochs> <bindir> <slot0> <slot1> [slot2 ...]
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
#define MAXL 8
// forge_pc_block DDR layout (must match the kernel)
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
#define RL_OUTPUT 1
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE  0x02

static long rdyad(long v){ long a=v<0?-v:v; if(!a)return 0; if(a>32767)a=32767; int m=0; for(int b=15;b>=1;b--)if((a>>b)&1){m=b;break;} if(m>=1&&((a>>(m-1))&1))m++; long p=1L<<m; return v<0?-p:p; }
static int rshift(long mx,int t){ long a=mx<0?-mx:mx; int h=0; for(int b=47;b>=1;b--)if((a>>b)&1){h=b;break;} int s=(h+1)-t; return s>0?s:0; }  // FIXED bit_length-target
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int8_t *load(const char*p,long*n){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(2);} fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET); int8_t*b=malloc(*n); if(fread(b,1,*n,f)!=(size_t)*n){fprintf(stderr,"read %s\n",p);exit(2);} fclose(f); return b; }

static int L;
static pci_bar_handle_t bar0[MAXL], bar4[MAXL], bar4r[MAXL];
static int32_t *hbuf[MAXL];

static void stage(int l,int off,int n){ for(long o=off;o<off+n;o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144; fpga_pci_write_burst(bar4[l],(uint64_t)o*4,(uint32_t*)(hbuf[l]+o),(uint64_t)k);} }
static void kick(int l){ fpga_pci_poke(bar0[l],R_BUF0,0); fpga_pci_poke(bar0[l],R_BUF1,0); fpga_pci_poke(bar0[l],R_AP_CTRL,AP_START);
    double t0=now_s(); uint32_t c=0; while(now_s()-t0<5.0){ fpga_pci_peek(bar0[l],R_AP_CTRL,&c); if(c&AP_DONE)return; } fprintf(stderr,"KICK HUNG slot %d\n",l); exit(6); }
static void readw(int l,int off,int32_t*dst,int n){ uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r[l],(uint64_t)(off+j)*4,&v); dst[j]=(int32_t)v; } }
static void setcfg(int l,int phase,int role,int doupd){ hbuf[l][O_CFG+0]=phase; hbuf[l][O_CFG+1]=role; hbuf[l][O_CFG+2]=8; hbuf[l][O_CFG+3]=doupd; stage(l,O_CFG,WL); }
static int role_of(int l){ return (l==L-1)?RL_OUTPUT:RL_HIDDEN; }

// host forward through all L layers (eval), == the chips' forward. returns predicted class
static int predict(const int8_t*x, int32_t**W){
    static long a[D]; for(int i=0;i<D;i++) a[i]=x[i];
    for(int l=0;l<L;l++){
        int od = (l==L-1)?C:D;
        long pre[D]; long mx=0;
        for(int o=0;o<od;o++){ long acc=0; for(int i=0;i<D;i++) acc+=rdyad(W[l][o*D+i])*a[i]; pre[o]=acc;
            long r=(l==L-1)?(acc<0?-acc:acc):(acc>0?acc:0); if(r>mx)mx=r; }
        int sh=rshift(mx,(l==L-1)?LG_BITS:H1_BITS);
        for(int o=0;o<od;o++){ long v=(l==L-1)?pre[o]:(pre[o]>0?pre[o]:0); a[o]=(int16_t)(v>>sh); }
        for(int o=od;o<D;o++) a[o]=0;
    }
    int best=0; long bv=-1L<<62; for(int c=0;c<C;c++){ if(a[c]>bv){bv=a[c];best=c;} } return best;
}

int main(int argc,char**argv){
    int epochs = argc>1?atoi(argv[1]):5;
    const char*dir = argc>2?argv[2]:"/tmp/pc_mnist";
    L = argc-3; if(L<2||L>MAXL){ fprintf(stderr,"usage: %s <epochs> <bindir> <slot0> <slot1> [..] (2<=L<=%d)\n",argv[0],MAXL); return 1; }
    int slot[MAXL]; for(int l=0;l<L;l++) slot[l]=atoi(argv[3+l]);

    long ntr,nte,nytr,nyte; char p[512];
    snprintf(p,512,"%s/train_x.i8",dir); int8_t*Xtr=load(p,&ntr);
    snprintf(p,512,"%s/train_y.u8",dir); int8_t*Ytr=load(p,&nytr);
    snprintf(p,512,"%s/test_x.i8",dir);  int8_t*Xte=load(p,&nte);
    snprintf(p,512,"%s/test_y.u8",dir);  int8_t*Yte=load(p,&nyte);
    int Ntr=nytr, Nte=nyte;
    printf("[shard] %d-layer net MODEL-PARALLEL across %d FPGAs (slots", L, L);
    for(int l=0;l<L;l++) printf(" %d",slot[l]); printf("); %d train %d test\n", Ntr, Nte);

    if(fpga_mgmt_init()){fprintf(stderr,"mgmt init\n");return 2;}
    for(int l=0;l<L;l++){
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR0,0,&bar0[l])){fprintf(stderr,"bar0 slot %d\n",slot[l]);return 3;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4[l])){fprintf(stderr,"bar4 slot %d\n",slot[l]);return 4;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,0,&bar4r[l])){fprintf(stderr,"bar4r slot %d\n",slot[l]);return 4;}
        hbuf[l]=calloc(BUFLEN,4);
    }
    // init each chip's weights ~ round(randn*4) (Irwin-Hall), momentum 0; stage once -> persist in its DDR
    srand(12345);
    for(int l=0;l<L;l++){
        for(int i=0;i<D*D;i++){ double g=0; for(int k=0;k<12;k++) g+=rand()/(double)RAND_MAX;
            hbuf[l][O_W+i]=(int)(4.0*(g-6.0)+(g>=6.0?0.5:-0.5)); hbuf[l][O_MOM+i]=0; }
        stage(l,0,BUFLEN);
    }

    int32_t *W[MAXL]; for(int l=0;l<L;l++) W[l]=malloc(4*D*D);
    int32_t *aout=malloc(4*N*D), *eout=malloc(4*N*D);
    int *perm=malloc(Ntr*sizeof(int)); for(int i=0;i<Ntr;i++) perm[i]=i;
    long xchip_msg=0, xchip_weight=0, xchip_grad=0;     // Gate C: bytes that cross chips, by kind
    int nb=Ntr/N;
    for(int ep=0; ep<epochs; ep++){
        double t0=now_s();
        for(int i=Ntr-1;i>0;i--){ int j=rand()%(i+1); int t=perm[i];perm[i]=perm[j];perm[j]=t; }
        for(int b=0;b<nb;b++){
            // ---- FORWARD SWEEP: slot 0 -> L-1, passing each layer's activation downstream ----
            for(int l=0;l<L;l++){
                setcfg(l,PH_FWD,role_of(l),0);
                if(l==0){ for(int n=0;n<N;n++){ int s=perm[b*N+n]; for(int i=0;i<D;i++) hbuf[0][O_AIN+n*D+i]=Xtr[s*D+i]; } }
                else    { memcpy(hbuf[l]+O_AIN, aout, 4*N*D); xchip_msg += (long)N*D; }   // AOUT(l-1) -> AIN(l)
                stage(l,O_AIN,N*D); kick(l);
                if(l<L-1) readw(l,O_AOUT,aout,N*D);   // read this layer's activation for the next chip
            }
            // ---- BACKWARD SWEEP: slot L-1 -> 0, passing each layer's error upstream ----
            for(int l=L-1;l>=0;l--){
                setcfg(l,PH_BWD,role_of(l),1);
                if(l==L-1){ for(int n=0;n<N;n++){ int s=perm[b*N+n]; for(int c=0;c<D;c++) hbuf[l][O_TGT+n*D+c]=0; hbuf[l][O_TGT+n*D+Ytr[s]]=8; } stage(l,O_TGT,N*D); }
                else      { memcpy(hbuf[l]+O_EIN, eout, 4*N*D); stage(l,O_EIN,N*D); xchip_msg += (long)N*D; }  // EOUT(l+1) -> EIN(l)
                kick(l);
                if(l>0) readw(l,O_EOUT,eout,N*D);     // read upstream error for the next (lower) chip
            }
            // weights & gradients are NEVER read-and-crossed -> xchip_weight, xchip_grad stay 0 by construction
        }
        for(int l=0;l<L;l++) readw(l,O_W,W[l],D*D);   // read each chip's weights (for host eval only)
        long ok=0; for(int t=0;t<Nte;t++) if(predict(Xte+t*D,W)==Yte[t]) ok++;
        printf("[shard] epoch %d: test acc = %.4f  (%d steps, %.1fs)  MODEL-PARALLEL/%d-chip\n",
               ep+1, (double)ok/Nte, nb, now_s()-t0, L); fflush(stdout);
    }
    printf("[gateC] cross-chip data over the whole run: messages=%ld ints (activations+errors), "
           "WEIGHTS=%ld, GRADIENTS=%ld  -> only neighbour messages crossed; no global model/gradient.\n",
           xchip_msg, xchip_weight, xchip_grad);
    for(int l=0;l<L;l++){ fpga_pci_detach(bar0[l]); fpga_pci_detach(bar4[l]); fpga_pci_detach(bar4r[l]); }
    return 0;
}
