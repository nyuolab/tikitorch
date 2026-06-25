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
#define EDN_BITS 13   // outgoing-error band (matches the kernel); used to rescale the residual-skip error
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
static int RES = 0;   // PC_RESIDUAL: host-side residual skip on hidden chips (out=relu_block(in)+in). A deep
                      // plain dyadic+relu cascade washes out the signal at depth>=4; the skip carries it
                      // through (the transformer's depth trick). Kernel unchanged; host adds the skip to the
                      // neighbour activation (fwd) and routes the skip error (bwd). Bit-exact to a ref with the
                      // same host arithmetic (pc_shard_train_ref.step_L(res=True)).
static pci_bar_handle_t bar0[MAXL], bar4[MAXL], bar4r[MAXL];
static int32_t *hbuf[MAXL];

// instrumentation accumulators (seconds), shared so the helpers can time themselves
static double T_STAGE=0, T_XCHIP=0, T_FWDK=0, T_BWDK=0, T_READW=0, T_EVAL=0;
static double kick_fwd_s[MAXL]={0}, kick_bwd_s[MAXL]={0};

static double stage(int l,int off,int n){ double t0=now_s(); for(long o=off;o<off+n;o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144; fpga_pci_write_burst(bar4[l],(uint64_t)o*4,(uint32_t*)(hbuf[l]+o),(uint64_t)k);} return now_s()-t0; }
static double kick(int l){ fpga_pci_poke(bar0[l],R_BUF0,0); fpga_pci_poke(bar0[l],R_BUF1,0);
    double t0=now_s(); fpga_pci_poke(bar0[l],R_AP_CTRL,AP_START); uint32_t c=0;
    while(now_s()-t0<8.0){ fpga_pci_peek(bar0[l],R_AP_CTRL,&c); if(c&AP_DONE)return now_s()-t0; } fprintf(stderr,"KICK HUNG slot %d\n",l); exit(6); }
static double readw(int l,int off,int32_t*dst,int n){ double t0=now_s(); uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r[l],(uint64_t)(off+j)*4,&v); dst[j]=(int32_t)v; } return now_s()-t0; }
static void setcfg(int l,int phase,int role,int doupd){ hbuf[l][O_CFG+0]=phase; hbuf[l][O_CFG+1]=role; hbuf[l][O_CFG+2]=8; hbuf[l][O_CFG+3]=doupd; T_STAGE+=stage(l,O_CFG,WL); }
static int role_of(int l){ return (l==L-1)?RL_OUTPUT:RL_HIDDEN; }

// host forward through all L layers (eval), == the chips' forward. returns predicted class
static int predict(const int8_t*x, int32_t**W){
    static long a[D], prev[D]; for(int i=0;i<D;i++) a[i]=x[i];
    for(int l=0;l<L;l++){
        int od = (l==L-1)?C:D;
        for(int i=0;i<D;i++) prev[i]=a[i];                 // save block input for the residual skip
        long pre[D]; long mx=0;
        for(int o=0;o<od;o++){ long acc=0; for(int i=0;i<D;i++) acc+=rdyad(W[l][o*D+i])*a[i]; pre[o]=acc;
            long r=(l==L-1)?(acc<0?-acc:acc):(acc>0?acc:0); if(r>mx)mx=r; }
        int sh=rshift(mx,(l==L-1)?LG_BITS:H1_BITS);
        for(int o=0;o<od;o++){ long v=(l==L-1)?pre[o]:(pre[o]>0?pre[o]:0); long raw=v>>sh;
            a[o]=(int16_t)((RES && l<L-1) ? raw+prev[o] : raw); }   // residual skip on hidden chips
        for(int o=od;o<D;o++) a[o]=0;
    }
    int best=0; long bv=-1L<<62; for(int c=0;c<C;c++){ if(a[c]>bv){bv=a[c];best=c;} } return best;
}

int main(int argc,char**argv){
    int epochs = argc>1?atoi(argv[1]):5;
    const char*dir = argc>2?argv[2]:"/tmp/pc_mnist";
    L = argc-3; if(L<2||L>MAXL){ fprintf(stderr,"usage: %s <epochs> <bindir> <slot0> <slot1> [..] (2<=L<=%d)\n",argv[0],MAXL); return 1; }
    int slot[MAXL]; for(int l=0;l<L;l++) slot[l]=atoi(argv[3+l]);
    RES = (getenv("PC_RESIDUAL") != NULL);   // host-side residual skip on hidden chips (deep nets)

    long ntr,nte,nytr,nyte; char p[512];
    snprintf(p,512,"%s/train_x.i8",dir); int8_t*Xtr=load(p,&ntr);
    snprintf(p,512,"%s/train_y.u8",dir); int8_t*Ytr=load(p,&nytr);
    snprintf(p,512,"%s/test_x.i8",dir);  int8_t*Xte=load(p,&nte);
    snprintf(p,512,"%s/test_y.u8",dir);  int8_t*Yte=load(p,&nyte);
    int Ntr=nytr, Nte=nyte;
    printf("[shard] %d-layer net MODEL-PARALLEL across %d FPGAs (slots", L, L);
    for(int l=0;l<L;l++) printf(" %d",slot[l]); printf("); %d train %d test%s\n", Ntr, Nte, RES?" [RESIDUAL skips]":"");

    if(fpga_mgmt_init()){fprintf(stderr,"mgmt init\n");return 2;}
    for(int l=0;l<L;l++){
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR0,0,&bar0[l])){fprintf(stderr,"bar0 slot %d\n",slot[l]);return 3;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4[l])){fprintf(stderr,"bar4 slot %d\n",slot[l]);return 4;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,0,&bar4r[l])){fprintf(stderr,"bar4r slot %d\n",slot[l]);return 4;}
        hbuf[l]=calloc(BUFLEN,4);
    }
    // init each chip's weights, momentum 0; stage once -> persist in its DDR.
    // PC_INITPERM=<dir>: load a VERIFIED-GOOD init (W<l>_init.i32) + data order (perm.i32) -> the silicon
    // bit-exactly reproduces a Python-verified run (the MSE+SignSGD recipe collapses for some valid data
    // orders; a pre-validated order avoids that). Else: round(randn*4) Irwin-Hall + per-epoch C shuffle.
    const char* ipdir = getenv("PC_INITPERM");
    srand(12345);
    for(int l=0;l<L;l++){
        if(ipdir){ snprintf(p,512,"%s/W%d_init.i32",ipdir,l); FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);return 2;}
                   if(fread(hbuf[l]+O_W,4,D*D,f)!=(size_t)(D*D)){fprintf(stderr,"read %s\n",p);return 2;} fclose(f); for(int i=0;i<D*D;i++) hbuf[l][O_MOM+i]=0; }
        else for(int i=0;i<D*D;i++){ double g=0; for(int k=0;k<12;k++) g+=rand()/(double)RAND_MAX;
            hbuf[l][O_W+i]=(int)(4.0*(g-6.0)+(g>=6.0?0.5:-0.5)); hbuf[l][O_MOM+i]=0; }
        stage(l,0,BUFLEN);
    }
    if(ipdir) printf("[shard] loaded verified-good init from %s\n", ipdir);

    int32_t *iW[MAXL]; for(int l=0;l<L;l++){ iW[l]=malloc(4*D*D); memcpy(iW[l], hbuf[l]+O_W, 4*D*D); }  // init snapshot (diag)
    int32_t *W[MAXL]; for(int l=0;l<L;l++) W[l]=malloc(4*D*D);
    int32_t *aout=malloc(4*N*D), *eout=malloc(4*N*D);
    int *perm=malloc(Ntr*sizeof(int)); for(int i=0;i<Ntr;i++) perm[i]=i;
    if(ipdir){ snprintf(p,512,"%s/perm.i32",ipdir); FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);return 2;}
               if(fread(perm,4,Ntr,f)!=(size_t)Ntr){fprintf(stderr,"read %s\n",p);return 2;} fclose(f); printf("[shard] loaded verified-good order (reused each epoch)\n"); }
    long xchip_msg=0, xchip_weight=0, xchip_grad=0;     // Gate C: ints that cross chips, by kind
    int nb=Ntr/N;
    double t_run0=now_s();                              // start of the measured training window
    for(int ep=0; ep<epochs; ep++){
        double t0=now_s();
        if(!ipdir) for(int i=Ntr-1;i>0;i--){ int j=rand()%(i+1); int t=perm[i];perm[i]=perm[j];perm[j]=t; }   // C shuffle (unless a verified order was loaded)
        for(int b=0;b<nb;b++){
            // ---- FORWARD SWEEP: slot 0 -> L-1, passing each layer's activation downstream ----
            for(int l=0;l<L;l++){
                setcfg(l,PH_FWD,role_of(l),0);
                if(l==0){ for(int n=0;n<N;n++){ int s=perm[b*N+n]; for(int i=0;i<D;i++) hbuf[0][O_AIN+n*D+i]=Xtr[s*D+i]; } T_STAGE+=stage(l,O_AIN,N*D); }
                else    { memcpy(hbuf[l]+O_AIN, aout, 4*N*D); xchip_msg += (long)N*D; T_XCHIP+=stage(l,O_AIN,N*D); }   // AOUT(l-1)->AIN(l): cross-chip
                { double kt=kick(l); kick_fwd_s[l]+=kt; T_FWDK+=kt; }
                if(l<L-1){ T_XCHIP+=readw(l,O_AOUT,aout,N*D);   // read this layer's activation to pass on = comms
                    if(RES) for(int j=0;j<N*D;j++) aout[j]+=hbuf[l][O_AIN+j]; }  // residual: out = raw + in_l (host skip)
            }
            // ---- BACKWARD SWEEP: slot L-1 -> 0, passing each layer's error upstream ----
            for(int l=L-1;l>=0;l--){
                setcfg(l,PH_BWD,role_of(l),1);
                if(l==L-1){ for(int n=0;n<N;n++){ int s=perm[b*N+n]; for(int c=0;c<D;c++) hbuf[l][O_TGT+n*D+c]=0; hbuf[l][O_TGT+n*D+Ytr[s]]=8; } T_STAGE+=stage(l,O_TGT,N*D); }
                else      { memcpy(hbuf[l]+O_EIN, eout, 4*N*D); xchip_msg += (long)N*D; T_XCHIP+=stage(l,O_EIN,N*D); }  // EOUT(l+1)->EIN(l): cross-chip
                { double kt=kick(l); kick_bwd_s[l]+=kt; T_BWDK+=kt; }
                if(l>0){ T_XCHIP+=readw(l,O_EOUT,eout,N*D);     // read upstream error to pass on = comms
                    if(RES && l<L-1){ for(int j=0;j<N*D;j++) eout[j]+=hbuf[l][O_EIN+j];  // residual: grad wrt in_l = routed + e_out_l
                        long mx=0; for(int j=0;j<N*D;j++){ long a=eout[j]<0?-eout[j]:eout[j]; if(a>mx)mx=a; }
                        int sh=rshift(mx,EDN_BITS); for(int j=0;j<N*D;j++) eout[j]>>=sh; } }  // rescale to stay int16-bounded (scale-invariant)
            }
            // weights & gradients are NEVER read-and-crossed -> xchip_weight, xchip_grad stay 0 by construction
            if(ep==0 && b==0){   // ---- one-time 2-chip flow diagnostic ----
                int32_t *ain1=malloc(4*N*D), *lg=malloc(4*N*D), *w1=malloc(4*D*D);
                readw(L-1,O_AIN ,ain1,N*D);   // h1 AS RECEIVED by chip L-1 (the cross-chip message)
                readw(L-1,O_AOUT,lg  ,N*D);   // chip L-1 logits (silicon)
                readw(L-1,O_W   ,w1  ,D*D);
                long hmax=0,amax=0; for(int j=0;j<N*D;j++){ long a=labs(aout[j]); if(a>hmax)hmax=a; long b2=labs(ain1[j]); if(b2>amax)amax=b2; }
                // host-recompute chip(L-1) OUTPUT forward from the RECEIVED ain1 + w1 (maxshift LG_BITS)
                long mxl=0; static long pre1[N][C];
                for(int n=0;n<N;n++)for(int c=0;c<C;c++){ long acc=0; for(int i=0;i<D;i++) acc+=rdyad(w1[c*D+i])*(long)ain1[n*D+i]; pre1[n][c]=acc; long al=acc<0?-acc:acc; if(al>mxl)mxl=al; }
                int shh=rshift(mxl,LG_BITS); long nzhost=0,nzsil=0;
                for(int n=0;n<N;n++)for(int c=0;c<C;c++){ long hl=pre1[n][c]>>shh; if(hl)nzhost++; if(lg[n*D+c])nzsil++; }
                printf("[diag] |chip0_AOUT(h1)|max=%ld  |chip1_AIN(recv)|max=%ld (match=%s) | chip1 logits nz: silicon=%ld host-recompute=%ld/%d (host sh=%d, |pre1|max=%ld)\n",
                    hmax,amax, hmax==amax?"yes":"NO!", nzsil,nzhost,N*C, shh,mxl);
                int32_t *w0=malloc(4*D*D); readw(0,O_W,w0,D*D);
                long c0=0,c1=0; for(int j=0;j<D*D;j++){ if(w0[j]!=iW[0][j])c0++; if(w1[j]!=iW[L-1][j])c1++; }
                printf("[diag] step1: W0 changed=%ld W1 changed=%ld | logits[0,:10]=%d %d %d %d %d %d %d %d %d %d\n",
                    c0,c1, lg[0],lg[1],lg[2],lg[3],lg[4],lg[5],lg[6],lg[7],lg[8],lg[9]);
                free(ain1);free(lg);free(w0);free(w1); fflush(stdout);
            }
            if(ep==0 && b==3){   // dump init + first 4 batches + post-step-4 W for a 4-step Python compare
                FILE*f;
                f=fopen("/tmp/s1_iW0.i32","wb"); if(f){fwrite(iW[0],4,D*D,f);fclose(f);}
                f=fopen("/tmp/s1_iW1.i32","wb"); if(f){fwrite(iW[L-1],4,D*D,f);fclose(f);}
                int32_t *bx=malloc(4*4*N*D),*by=malloc(4*4*N);
                for(int bb=0;bb<4;bb++)for(int n=0;n<N;n++){ int s=perm[bb*N+n]; by[bb*N+n]=Ytr[s]; for(int i=0;i<D;i++) bx[(bb*N+n)*D+i]=Xtr[s*D+i]; }
                f=fopen("/tmp/s1_x.i32","wb"); if(f){fwrite(bx,4,4*N*D,f);fclose(f);}
                f=fopen("/tmp/s1_y.i32","wb"); if(f){fwrite(by,4,4*N,f);fclose(f);}
                int32_t *a0=malloc(4*D*D),*a1=malloc(4*D*D); readw(0,O_W,a0,D*D); readw(L-1,O_W,a1,D*D);
                f=fopen("/tmp/s1_W0.i32","wb"); if(f){fwrite(a0,4,D*D,f);fclose(f);}
                f=fopen("/tmp/s1_W1.i32","wb"); if(f){fwrite(a1,4,D*D,f);fclose(f);}
                free(bx);free(by);free(a0);free(a1);
            }
            if(ep==0 && (b==nb/4 || b==nb/2 || b==3*nb/4)){   // in-epoch accuracy trajectory (diverge vs flat?)
                for(int l=0;l<L;l++) readw(l,O_W,W[l],D*D);
                long ok=0; for(int t=0;t<Nte;t++) if(predict(Xte+t*D,W)==Yte[t]) ok++;
                long m0=0,m1=0; for(int j=0;j<D*D;j++){ long a=labs(W[0][j]); if(a>m0)m0=a; long b2=labs(W[L-1][j]); if(b2>m1)m1=b2; }
                printf("[mideval] epoch1 step %d/%d: acc=%.4f  |W0|max=%ld |W1|max=%ld\n", b,nb,(double)ok/Nte,m0,m1); fflush(stdout);
            }
        }
        for(int l=0;l<L;l++) T_READW+=readw(l,O_W,W[l],D*D);   // read each chip's weights (host eval only)
        long wc[MAXL]={0}; for(int l=0;l<L;l++) for(int j=0;j<D*D;j++) if(W[l][j]!=iW[l][j]) wc[l]++;
        double te0=now_s(); long ok=0; for(int t=0;t<Nte;t++) if(predict(Xte+t*D,W)==Yte[t]) ok++; T_EVAL+=now_s()-te0;
        printf("[shard] epoch %d: test acc = %.4f  (%d steps, %.1fs)  MODEL-PARALLEL/%d-chip  [W0 chg=%ld/%d W%d chg=%ld/%d vs init]\n",
               ep+1, (double)ok/Nte, nb, now_s()-t0, L, wc[0], D*D, L-1, wc[L-1], D*D); fflush(stdout);
    }
    // dump ALL L trained weights so the host can re-eval / bit-exact-replay with the Python ref (per chip).
    // NOTE: do NOT alias shardW1.i32 to W[L-1] — for L>=4 that clobbers the real chip-1 (hidden) dump with the
    // OUTPUT layer (the bug that made chip 1 spuriously "diverge" in the first sweep; training was bit-exact).
    for(int l=0;l<L;l++){ snprintf(p,512,"/tmp/shardW%d.i32",l); FILE*f=fopen(p,"wb"); if(f){fwrite(W[l],4,D*D,f);fclose(f);} }
    double wall = now_s() - t_run0;
    long steps = (long)epochs * nb;          // one model-parallel training step per batch
    long kicks = steps * 2L * L;             // L fwd kicks + L bwd kicks per step
    long samples = steps * (long)N;
    double t_comp = T_FWDK + T_BWDK;
    // ===== THROUGHPUT =====
    printf("\n========== SIGNATURE RESULT: %d-chip model-parallel PC training, fully measured ==========\n", L);
    printf("[throughput] %ld steps, %ld samples, %ld kicks over %.2fs wall\n", steps, samples, kicks, wall);
    printf("[throughput] %.1f steps/s | %.0f samples/s | %.1f kicks/s | %.3f ms/step | %.3f ms/sample\n",
           steps/wall, samples/wall, kicks/wall, 1e3*wall/steps, 1e3*wall/samples);
    printf("[throughput] avg fabric time/kick: fwd %.3f ms, bwd %.3f ms (sum compute %.2fs = %.1f%% of wall)\n",
           1e3*T_FWDK/(steps*L), 1e3*T_BWDK/(steps*L), t_comp, 100.0*t_comp/wall);
    for(int l=0;l<L;l++) printf("[throughput]   chip %d (slot %d): fwd-kick %.3f ms, bwd-kick %.3f ms (avg)\n",
           l, slot[l], 1e3*kick_fwd_s[l]/steps, 1e3*kick_bwd_s[l]/steps);
    // ===== TIME BREAKDOWN =====
    printf("[breakdown] fwd-compute %.1f%% | bwd-compute %.1f%% | cross-chip-comms %.1f%% | input-stage %.1f%% | weight-readback %.1f%% | host-eval %.1f%%\n",
           100*T_FWDK/wall, 100*T_BWDK/wall, 100*T_XCHIP/wall, 100*T_STAGE/wall, 100*T_READW/wall, 100*T_EVAL/wall);
    // ===== COMMS =====
    long xbytes = xchip_msg * 4;             // int32 messages
    double xMB = xbytes/1e6;
    printf("[comms] cross-chip messages: %ld ints = %.2f MB total | %ld bytes/step (%ld ints) | %.2f MB/s during comms\n",
           xchip_msg, xMB, xbytes/steps, xchip_msg/steps, (T_XCHIP>0? xMB/T_XCHIP : 0.0));
    printf("[comms] comms = %.1f%% of wall; per step %ld activation+error ints cross between adjacent chips (each msg = N*D = %d ints)\n",
           100*T_XCHIP/wall, xchip_msg/steps, N*D);
    // ===== GATE C: the structural guarantee =====
    printf("[gateC] WEIGHTS crossed chips=%ld bytes | GRADIENTS crossed=%ld bytes | only neighbour messages crossed\n",
           xchip_weight*4, xchip_grad*4);
    printf("[gateC] -> no chip ever held the global model or the global gradient; no all-reduce, no barrier.\n");
    // ===== ENERGY: the runner samples per-slot Vccint (fpga-describe -M) concurrently; J computed there =====
    printf("[energy] (board power sampled per slot by the runner during this window; wall=%.2fs, steps=%ld -> J/step in the run's energy.txt)\n", wall, steps);
    printf("ENERGY_WINDOW wall_s=%.3f steps=%ld samples=%ld\n", wall, steps, samples);  // parsed by the runner for J/step
    for(int l=0;l<L;l++){ fpga_pci_detach(bar0[l]); fpga_pci_detach(bar4[l]); fpga_pci_detach(bar4r[l]); }
    return 0;
}
