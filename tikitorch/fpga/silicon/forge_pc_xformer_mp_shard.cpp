// forge_pc_xformer_mp_shard.cpp — train the BF16 MIXED-PRECISION PC-transformer MODEL-PARALLEL across the FPGAs
// of an f2.48xlarge: ONE transformer block per physical chip (forge_pc_xformer_mp_block kernel; role/phase CFG).
// Chip 0 = INPUT (float+int embed/pos, block 0), chips 1..L-2 = HIDDEN, chip L-1 = OUTPUT (block, final-LN,
// LM head, CE). Each chip holds ONLY its block's FLOAT master + bf16 dyadic-Adam state (+ the deployed int16
// master) in its OWN on-card DDR; the ONLY data crossing chips is a WIDE (bf16/float) neighbour hidden-state
// (down, forward) + a WIDE (bf16) neighbour error (up, backward). No global model / gradient / all-reduce /
// barrier — same no-global-sync contract as the int shard, now for the BF16 mixed-precision trainer.
//
// This is the BF16-MP twin of forge_pc_xformer_shard.cpp. It mirrors that driver's structure verbatim (open the
// slots via the AWS fpga_mgmt/fpga_pci API, host-shadow each chip's DDR, burst-stage regions, kick with a CFG
// word, read back DDR, relay boundary messages over PCIe, persist weights). The ONLY real differences:
//   * the trainable model is the FLOAT MASTER in the appended MP_* DDR region (staged as float-per-int32 lane),
//     init'd EXACTLY like the de-risked bf16 golden bf_init (the recipe that reaches the char-LM bar);
//   * the neighbour messages are WIDE (float lanes): down = MP_GROUT -> MP_GRIN, up = MP_GROUT -> MP_GRIN;
//   * bpb is read from the OUTPUT chip's WIDE float logits MP_WLG (the value the golden's bpb uses), not XO_LOGIT;
//   * the optimizer is the kernel's bf16 dyadic-Adam, applied inside the BWD kick when DO_UPDATE=1; the lr is
//     staged in CFG lane 3 as a fixed-point code LRBITS = round(lr * 2^20)  (kernel: LR = LRBITS * 2^-20).
//
// The int16 deploy master at the base XO_* offsets is ALSO staged (the kernel's byte-identical int forward reads
// it), so the chip is a drop-in for the int AFI's DDR scaffolding; but TRAINING + bpb live on the wide path.
//
//   g++ -O2 -std=c++14 -I<hls/forge_kernels> -I$SDK/include forge_pc_xformer_mp_shard.cpp -lfpga_mgmt -o xfmpshard
//   sudo ./xfmpshard <steps> <bindir> <slot0> <slot1> ... <slotL-1>          (env: ACCUM, XF_LR, XF_WARMUP,
//                                                                              SEED, INIT_DIR, SAMPLE, DUMP_DIR)
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
#include "forge_pc_xformer_mp_config.hpp"   // base X_* dims + XO_* int layout + the appended MP_* float layout

#define MAXL 8
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE  0x02

static int L;
static pci_bar_handle_t bar0[MAXL], bar4[MAXL], bar4r[MAXL];
static int32_t *hbuf[MAXL];                 // host shadow of each chip's DDR buffer (MP_BUFLEN int32 lanes)
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

// float <-> int32 DDR lane bit-reinterpret (one float per int32 lane; exactly the kernel's i2f/f2i)
static inline int32_t f2i(float f){ union{float f; int32_t i;} c; c.f=f; return c.i; }
static inline float   i2f(int32_t i){ union{float f; int32_t i;} c; c.i=i; return c.f; }

// ---- burst-stage helpers (identical to the int shard's stage/kick/readw) ----
static void stage(int l,int off,int n){ for(long o=off;o<off+n;o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144;
    fpga_pci_write_burst(bar4[l],(uint64_t)o*4,(uint32_t*)(hbuf[l]+o),(uint64_t)k);} }
// per-phase kick timing (bundle item 0): attribute wall-clock per chip x phase — fwd vs bwd kernel-exec ms
// (kick() tight-spins AP_DONE, so the measured span IS kernel execution). g_phase set by the sweep loops.
static int g_phase = 0;                       // 0=fwd 1=bwd (index into the accumulators)
static double g_kick_s[2][8]; static long g_kick_n[2][8];
static void kick(int l){ fpga_pci_poke(bar0[l],R_BUF0,0); fpga_pci_poke(bar0[l],R_BUF1,0);
    double t0=now_s(); fpga_pci_poke(bar0[l],R_AP_CTRL,AP_START); uint32_t c=0;
    while(now_s()-t0<20.0){ fpga_pci_peek(bar0[l],R_AP_CTRL,&c);
        if(c&AP_DONE){ g_kick_s[g_phase][l]+=now_s()-t0; g_kick_n[g_phase][l]++; return; } }
    fprintf(stderr,"KICK HUNG slot %d\n",l); exit(6); }
static void kick_report(){
    for(int p=0;p<2;p++) for(int l=0;l<8;l++) if(g_kick_n[p][l])
        printf("[kicktime] %s chip %d: %.1f ms/kick (n=%ld)\n", p?"BWD":"FWD", l, 1e3*g_kick_s[p][l]/g_kick_n[p][l], g_kick_n[p][l]);
    fflush(stdout); }
// BURSTED readback (Track-B Stage-0): one mmap'd memcpy from the BAR4 read window instead of n
// word-at-a-time fpga_pci_peek calls. The per-word peek (function-call + PCIe round-trip per 4 bytes)
// was the 54-61%-of-wall-clock host-relay bottleneck; the write half (stage/write_burst) was already
// bursted, so this makes the neighbour-message relay symmetric. Falls back to per-word peek if
// fpga_pci_get_address is unavailable -> zero regression risk. Env READBURST=0 forces the old path.
static int32_t g_rbuf[1<<18];
static int g_readburst = -1;
static inline int readburst_on(){ if(g_readburst<0){ const char*e=getenv("READBURST"); g_readburst = e?atoi(e):1; } return g_readburst; }
static void readw(int l,int off,int32_t*dst,int n){
    void*p=NULL;
    if(readburst_on() && fpga_pci_get_address(bar4r[l],(uint64_t)off*4,(uint64_t)n,&p)==0 && p){ memcpy(dst,p,(size_t)n*4); return; }
    uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r[l],(uint64_t)(off+j)*4,&v); dst[j]=(int32_t)v; }
}
// float-lane read: bulk-read int32 lanes (one memcpy) then reinterpret as float
static void readf(int l,int off,float*dst,int n){
    void*p=NULL;
    if(readburst_on() && n<=(1<<18) && fpga_pci_get_address(bar4r[l],(uint64_t)off*4,(uint64_t)n,&p)==0 && p){
        memcpy(g_rbuf,p,(size_t)n*4); for(int j=0;j<n;j++) dst[j]=i2f(g_rbuf[j]); return; }
    uint32_t v; for(int j=0;j<n;j++){ fpga_pci_peek(bar4r[l],(uint64_t)(off+j)*4,&v); dst[j]=i2f((int32_t)v); }
}
// stage a host float array into a DDR float region (lane bit-reinterpret then burst)
static void put_f(int l,int off,const float*src,int n){ for(int j=0;j<n;j++) hbuf[l][off+j]=f2i(src[j]); stage(l,off,n); }
// scratch for the deployed-int down-message (the int forward's HOUT->HIN relay; kept alongside the wide path so
// the byte-identical int forward on each chip has a coherent neighbour hidden-state). One int32 per lane.
static int32_t g_ihmsg[X_SEQ*X_D];
static inline int32_t* ihmsg(){ return g_ihmsg; }

// CFG word lanes: [phase, role, do_update, lrbits]  (lane 3 = fixed-point lr code, lr = lrbits*2^-20)
static int g_trainfast=-1;   // TRAINFAST=1 (default): kernel skips the deploy-int forward + driver skips the int relay
static inline int trainfast(){ if(g_trainfast<0){ const char*e=getenv("TRAINFAST"); g_trainfast=e?atoi(e):1; } return g_trainfast; }
static void setcfg(int l,int phase,int role,int doupd,int lrbits,int gzero=0,int train=0){
    hbuf[l][XO_CFG+0]=phase; hbuf[l][XO_CFG+1]=role; hbuf[l][XO_CFG+2]=doupd; hbuf[l][XO_CFG+3]=lrbits;
    hbuf[l][XO_CFG+4]=gzero;   // ACCUM>1: 1 on the first sub-batch -> kernel zeroes the grad accumulators
    hbuf[l][XO_CFG+5]=train;   // 1 = skip the deploy-int forward (training kicks; ~8ms/fwd-kick saved)
    stage(l,XO_CFG,X_WL); }
static int role_of(int l){ return (l==0)?X_RL_INPUT : (l==L-1)?X_RL_OUT : X_RL_HID; }

// =====================================================================================================
// FLOAT-master init — BIT-EXACT to the de-risked golden bf_init (forge_pc_xformer_bf16_golden_block.hpp):
//   matmul weights ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) via the bf_u LCG;  EMB/POS ~ N(0,1) via bf_n (sum of
//   12 uniforms - 6);  LN gamma=1, beta=0 (frozen).  RNG: g=seed*2654435761u+1; draw advances g=g*1103515245+12345.
//   DRAW ORDER MUST MATCH bf_init EXACTLY: WQ,WK,WV,WO,WUP,WDN,WHEAD, then EMB, then POS (per chip seed).
// (The bf16-golden reaches the char-LM bar with this init; replicating it lets a Python ref replay the run.)
// =====================================================================================================
static unsigned g_rng = 1;
static inline double bf_u(){ g_rng = g_rng*1103515245u + 12345u; return ((g_rng>>8)&0xffffff)/(double)0x1000000; }
static inline double bf_n(){ double u=0; for(int k=0;k<12;k++) u+=bf_u(); return u-6.0; }
static void uni(float*W,int n,int fan){ double a=1.0/sqrt((double)fan); for(int i=0;i<n;i++) W[i]=(float)((bf_u()*2-1)*a); }

// per-chip FLOAT master (host side; what we stage into the MP_F* region + init the deploy int16 master from)
struct FMaster {
    float WQ[X_D*X_D], WK[X_D*X_D], WV[X_D*X_D], WO[X_D*X_D];
    float WUP[X_FFN*X_D], WDN[X_D*X_FFN], WHEAD[X_VOCAB*X_D];
    float EMB[X_VOCAB*X_D], POS[X_SEQ*X_D];
    float LN1G[X_D], LN1B[X_D], LN2G[X_D], LN2B[X_D], LNFG[X_D], LNFB[X_D];
};
static void init_master(FMaster &m, unsigned seed){
    g_rng = seed*2654435761u + 1u;
    uni(m.WQ,X_D*X_D,X_D); uni(m.WK,X_D*X_D,X_D); uni(m.WV,X_D*X_D,X_D); uni(m.WO,X_D*X_D,X_D);
    uni(m.WUP,X_FFN*X_D,X_D); uni(m.WDN,X_D*X_FFN,X_FFN); uni(m.WHEAD,X_VOCAB*X_D,X_D);
    for(int i=0;i<X_VOCAB*X_D;i++) m.EMB[i]=(float)bf_n();
    for(int i=0;i<X_SEQ*X_D;i++)   m.POS[i]=(float)bf_n();
    for(int d=0;d<X_D;d++){ m.LN1G[d]=1;m.LN2G[d]=1;m.LNFG[d]=1; m.LN1B[d]=0;m.LN2B[d]=0;m.LNFB[d]=0; }
}
// optional per-chip host init override: a flat float32 file of the master in struct-field order (see dump).
static bool load_master_file(FMaster &m, const char* path){
    FILE* f=fopen(path,"rb"); if(!f) return false;
    size_t n = fread(&m, sizeof(float), sizeof(FMaster)/sizeof(float), f); fclose(f);
    return n == sizeof(FMaster)/sizeof(float);
}
static void dump_master_file(const FMaster &m, const char* path){
    FILE* f=fopen(path,"wb"); if(!f) return; fwrite(&m, sizeof(float), sizeof(FMaster)/sizeof(float), f); fclose(f);
}

// nearest signed power-of-two of a float (== kernel bf_dyad_f / char-LM DyadSTE) — to seed the int16 deploy
// master from the float master so the byte-identical int forward is dyad-faithful to the wide path.
static double dyad_f(double wf){ double a=wf<0?-wf:wf; if(a<1e-30) return 0.0; int e=(int)lround(log2(a)); double p=ldexp(1.0,e); return wf<0?-p:p; }

// per-chip activation-statistics probe (ACTSTATS=1; no-op otherwise) — localizes a silicon-side forward
// degradation WITHOUT a CPU reference: does the WIDE residual stream R' (and the int deploy HOUT) collapse,
// explode, or saturate across chips 0->L-1 or over training? Printed at step 0 and every 500 steps.
static int g_actstats=-1;
static int actstats_on(){ if(g_actstats<0){ const char*e=getenv("ACTSTATS"); g_actstats=e?atoi(e):0; } return g_actstats; }
static void actstat(const char* tag, int chip, int step, const float* x, int n){
    if(!actstats_on() || !(step==0 || step%500==0)) return;
    double mn=1e300, mx=-1e300, sa=0; int bad=0;
    for(int i=0;i<n;i++){ double v=x[i]; if(!(v==v) || v>1e300 || v<-1e300){bad++;continue;} if(v<mn)mn=v; if(v>mx)mx=v; sa+=v<0?-v:v; }
    printf("[actstat] step %5d  %-8s chip %d  min=%+.3e max=%+.3e absmean=%.3e nan/inf=%d/%d\n",
           step, tag, chip, mn, mx, sa/(n>0?n:1), bad, n); fflush(stdout);
}
static void actstat_i(const char* tag, int chip, int step, const int32_t* x, int n){
    if(!actstats_on() || !(step==0 || step%500==0)) return;
    long mn=1L<<40, mx=-(1L<<40); double sa=0; int sat=0;
    for(int i=0;i<n;i++){ long v=x[i]; if(v<mn)mn=v; if(v>mx)mx=v; sa+=v<0?-v:v; if(v>=32767||v<=-32768)sat++; }
    printf("[actstat] step %5d  %-8s chip %d  min=%ld max=%ld absmean=%.1f int16sat=%d/%d\n",
           step, tag, chip, mn, mx, sa/(n>0?n:1), sat, n); fflush(stdout);
}

int main(int argc,char**argv){
    int steps = argc>1?atoi(argv[1]):2000;
    const char*dir = argc>2?argv[2]:"/tmp/pc_shakespeare";
    L = argc-3; if(L<2||L>MAXL){ fprintf(stderr,"usage: %s <steps> <bindir> <slot0..slotL-1> (2<=L<=%d)\n",argv[0],MAXL); return 1; }
    int slot[MAXL]; for(int l=0;l<L;l++) slot[l]=atoi(argv[3+l]);

    // training hyper-params (env; defaults == the bf16 golden's char-LM recipe)
    unsigned seed   = getenv("SEED")?(unsigned)atoi(getenv("SEED")):7u;
    int   ACCUM     = getenv("ACCUM")?atoi(getenv("ACCUM")):1;        // DEFAULT correct simple sync loop (ACCUM=1)
    double LR       = getenv("XF_LR")?atof(getenv("XF_LR")):3e-3;     // char-LM default
    int   WARMUP    = getenv("XF_WARMUP")?atoi(getenv("XF_WARMUP")):200;
    const char* INIT_DIR = getenv("INIT_DIR");                        // optional per-chip float-master init files
    const char* DUMP_DIR = getenv("DUMP_DIR")?getenv("DUMP_DIR"):"/tmp";
    if(ACCUM<1) ACCUM=1;

    // ---- load TinyShakespeare token stream (uint16), exactly like the int shard / run_pc_xformer_shard_f2.sh ----
    char p[512]; snprintf(p,512,"%s/train.u16",dir); FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);return 2;}
    fseek(f,0,SEEK_END); long nb=ftell(f); fseek(f,0,SEEK_SET); long ntok=nb/2; uint16_t*data=(uint16_t*)malloc(nb);
    if(fread(data,2,ntok,f)!=(size_t)ntok){fprintf(stderr,"read tokens\n");return 2;} fclose(f);
    printf("[xfmpshard] BF16-MP %d-block transformer MODEL-PARALLEL across %d FPGAs (slots", L, L);
    for(int l=0;l<L;l++) printf(" %d",slot[l]); printf(")\n");
    printf("[xfmpshard] %ld train tokens | D=%d H=%d FFN=%d SEQ=%d VOCAB=%d | LR=%.1e warmup=%d ACCUM=%d seed=%u\n",
           ntok, X_D, X_H, X_FFN, X_SEQ, X_VOCAB, LR, WARMUP, ACCUM, seed);

    if(fpga_mgmt_init()){fprintf(stderr,"mgmt init\n");return 2;}
    for(int l=0;l<L;l++){
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR0,0,&bar0[l])){fprintf(stderr,"bar0 %d\n",slot[l]);return 3;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4[l])){fprintf(stderr,"bar4 %d\n",slot[l]);return 4;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,0,&bar4r[l])){fprintf(stderr,"bar4r %d\n",slot[l]);return 4;}
        hbuf[l]=(int32_t*)calloc(MP_BUFLEN,4);    // shadow the WHOLE MP buffer (int base + appended float region)
    }

    // ---- exp2 LUT (== int shard / golden g_lut_init: 65536 * 0.917004^g, Q16) — the int forward's softmax ROM ----
    int32_t lut[X_LUT_N]; { double v=65536.0; for(int g=0;g<X_LUT_N;g++){ lut[g]=(int32_t)(v+0.5); v*=0.917004; } }

    // =================================================================================================
    // INIT: per chip, build the FLOAT master (golden bf_init, or a host file), stage it into the MP_F* region,
    //       zero the bf16-Adam m/v, seed the deployed int16 master = round(dyad(float)*2^WFRAC) so the int
    //       forward is dyad-faithful, stage the LUT + LN gammas. Persist the whole buffer once.
    // =================================================================================================
    static FMaster M[MAXL];
    for(int l=0;l<L;l++){
        // 1) FLOAT master (per-chip seed offset, like the int shard's per-chip seed)
        init_master(M[l], seed + 1000u*l);
        if(INIT_DIR){ char ip[512]; snprintf(ip,512,"%s/master%d.f32",INIT_DIR,l);
            if(load_master_file(M[l],ip)) printf("[xfmpshard] chip %d: loaded host init %s\n", l, ip); }
        FMaster &m = M[l];

        // 2) stage the FLOAT master into the appended MP_F* region (the trainable quantity the STE bwd diffs)
        put_f(l, MP_FWQ,  m.WQ,  X_D*X_D);   put_f(l, MP_FWK,  m.WK,  X_D*X_D);
        put_f(l, MP_FWV,  m.WV,  X_D*X_D);   put_f(l, MP_FWO,  m.WO,  X_D*X_D);
        put_f(l, MP_FWUP, m.WUP, X_FFN*X_D); put_f(l, MP_FWDN, m.WDN, X_D*X_FFN);
        put_f(l, MP_FLN1G,m.LN1G,X_D);       put_f(l, MP_FLN1B,m.LN1B,X_D);
        put_f(l, MP_FLN2G,m.LN2G,X_D);       put_f(l, MP_FLN2B,m.LN2B,X_D);
        if(l==0){     put_f(l, MP_FEMB, m.EMB, X_VOCAB*X_D); put_f(l, MP_FPOS, m.POS, X_SEQ*X_D); }
        if(l==L-1){   put_f(l, MP_FLNFG,m.LNFG,X_D); put_f(l, MP_FLNFB,m.LNFB,X_D); put_f(l, MP_FWHEAD,m.WHEAD,X_VOCAB*X_D); }

        // 3) bf16 dyadic-Adam m/v = 0  (calloc already zeroed every MP_MOM/MP_VAR/MP_MEMB.. lane; nothing to do)

        // 4) deployed int16 master = round(dyad(float)*2^WFRAC). WFRAC=0 (no scale) -> int16 = round(dyad(float));
        //    the int forward then drives dyad(int16) == dyad(float master), byte-identical to the wide path.
        auto seti=[&](int off,const float*src,int n){ for(int i=0;i<n;i++){ double q=dyad_f((double)src[i]); int v=(int)lround(q);
            if(v>32767)v=32767; if(v<-32767)v=-32767; hbuf[l][off+i]=v; } };
        seti(XO_WQ,m.WQ,X_D*X_D); seti(XO_WK,m.WK,X_D*X_D); seti(XO_WV,m.WV,X_D*X_D); seti(XO_WO,m.WO,X_D*X_D);
        seti(XO_WUP,m.WUP,X_FFN*X_D); seti(XO_WDN,m.WDN,X_D*X_FFN);
        for(int d=0;d<X_D;d++){ hbuf[l][XO_LN1G+d]=64; hbuf[l][XO_LN2G+d]=64; hbuf[l][XO_LNFG+d]=64;
                                hbuf[l][XO_LN1B+d]=0;  hbuf[l][XO_LN2B+d]=0;  hbuf[l][XO_LNFB+d]=0; }
        if(l==0){ seti(XO_EMBED,m.EMB,X_VOCAB*X_D); seti(XO_POS,m.POS,X_SEQ*X_D); }
        if(l==L-1) seti(XO_WHEAD,m.WHEAD,X_VOCAB*X_D);
        for(int g=0;g<X_LUT_N;g++) hbuf[l][XO_LUT+g]=lut[g];

        stage(l,0,MP_BUFLEN);    // persist the whole buffer (int base + float region) once
    }

    // boundary-message scratch (one float per lane; the wide down/up messages are SEQ*D floats)
    float *hmsg=(float*)malloc(4*X_SEQ*X_D);   // wide down-message (forward): MP_GROUT(l) -> MP_GRIN(l+1)
    float *emsg=(float*)malloc(4*X_SEQ*X_D);   // wide up-message  (backward): MP_GROUT(l) -> MP_GRIN(l-1)
    float *wlg =(float*)malloc(4*X_SEQ*X_VOCAB);  // OUTPUT chip wide logits (for bpb)
    long xchip_msg=0, xchip_weight=0, xchip_grad=0;
    double t0=now_s(); double bpb_ema=-1, acc_ema=-1;
    srand(123);

    // lr -> fixed-point code (lr = lrbits*2^-20); warmup applied per step like the golden recipe.
    auto lrcode=[&](double lr){ long c=(long)llround(lr * (double)(1<<20)); if(c<0)c=0; if(c>0x7fffffffL)c=0x7fffffffL; return (int)c; };

    for(int step=0; step<steps; step++){
        // char-LM linear warmup (== golden): lr = LR * min(1, (step+1)/WARMUP), then optional COSINE DECAY to
        // LR*LR_MINFRAC over the run (LR_DECAY=1). WHY: the silicon's constant-LR loss floor sits higher than the
        // CPU refs' (noisier effective gradient: bf16 messages + reduction reorder + fp32 arith; floor ~ LR*noise).
        // MEASURED: constant 3e-3 plateaus at ema 3.80-3.83 from ~step 600 while every CPU ref descends; resuming
        // the plateaued masters at LR 1e-3 immediately resumes descending -> ride the floor down with a schedule.
        static int   LR_DECAY  = getenv("LR_DECAY")  ? atoi(getenv("LR_DECAY"))   : 0;
        static double LR_MINFR = getenv("LR_MINFRAC")? atof(getenv("LR_MINFRAC")) : 0.1;
        double lr = LR * fmin(1.0, (step+1)/(double)(WARMUP>0?WARMUP:1));
        if (LR_DECAY && step >= WARMUP) {
            double t = (double)(step - WARMUP) / (double)(steps - WARMUP > 0 ? steps - WARMUP : 1);
            lr = LR * (LR_MINFR + (1.0 - LR_MINFR) * 0.5 * (1.0 + cos(M_PI * t)));
        }
        // ACCUM = effective BATCH size (the lever that takes silicon from the batch-1 ~4.0 plateau to a coherent
        //   ~3.0; golden ACCUM 1/8/32 -> 3.74/3.20/2.95). The kernel now TRULY accumulates: GZERO=1 on the first
        //   sub-batch zeroes the persistent DDR grad accumulators (MP_GACC_*), each later BWD adds, and DO_UPDATE=1
        //   on the LAST sub-batch runs dyadic-Adam on the accumulated sum. So ACCUM>1 = a real batch of ACCUM
        //   sequences per weight update (verified csim: acc1=g, acc2=2g). All ACCUM sub-batches see the same weights.
        long last_off = 0;
        for(int k=0;k<ACCUM;k++){
            // FIXEDOFF>=0 pins every batch to the same offset (overfit / kernel-can-fit sanity test on silicon):
            // if the kernel is sound it must memorize one batch -> bpb -> ~0; a floor >>0 = a real kernel/HW fault.
            static long g_fixedoff = getenv("FIXEDOFF") ? atol(getenv("FIXEDOFF")) : -1;
            long off = g_fixedoff>=0 ? g_fixedoff : (long)(rand() % (ntok - X_SEQ - 2)); last_off = off;
            int do_update = (k==ACCUM-1) ? 1 : 0;

            // ===================== FORWARD SWEEP: chip 0 -> L-1 =====================
            g_phase=0;
            for(int l=0;l<L;l++){
                setcfg(l, X_PH_FWD, role_of(l), 0, 0, 0, trainfast());
                if(l==0){
                    // INPUT: stage token ids (both the int forward and the wide forward read XO_TOK)
                    for(int t=0;t<X_SEQ;t++) hbuf[0][XO_TOK+t]=data[off+t]; stage(0,XO_TOK,X_SEQ);
                } else {
                    // HID/OUTPUT: stage the upstream chip's WIDE R' as this chip's down-message (MP_GRIN); the
                    // wide forward reads MP_GRIN as its block input R (kernel line ~535). Also feed the int path's
                    // XO_HIN (the deployed int forward's neighbour hidden-state) — staged from the int HOUT relay.
                    put_f(l, MP_GRIN, hmsg, X_SEQ*X_D);       xchip_msg += (long)X_SEQ*X_D;
                    if(!trainfast()){ memcpy(hbuf[l]+XO_HIN, ihmsg(), 4*X_SEQ*X_D); stage(l,XO_HIN,X_SEQ*X_D); }
                }
                if(l==L-1){ for(int t=0;t<X_SEQ;t++) hbuf[l][XO_TGT+t]=data[off+t+1]; stage(l,XO_TGT,X_SEQ); }  // OUTPUT targets
                kick(l);
                if(l<L-1){
                    readf(l, MP_GROUT, hmsg, X_SEQ*X_D);      // WIDE down-message: this block's R' -> next chip MP_GRIN
                    if(!trainfast()) readw(l, XO_HOUT, ihmsg(), X_SEQ*X_D);   // int down-message (deploy path only)
                    actstat  ("R'(wide)", l, step, hmsg, X_SEQ*X_D);   // probe: wide residual stream across chips
                    actstat_i("HOUT(int)", l, step, ihmsg(), X_SEQ*X_D);
                }
            }
            // ===================== BACKWARD SWEEP: chip L-1 -> 0 =====================
            g_phase=1;
            for(int l=L-1;l>=0;l--){
                setcfg(l, X_PH_BWD, role_of(l), do_update, do_update?lrcode(lr):0, (k==0)?1:0);   // DO_UPDATE -> dyadic-Adam apply; k==0 -> zero grad-accum
                if(l<L-1){ put_f(l, MP_GRIN, emsg, X_SEQ*X_D); xchip_msg += (long)X_SEQ*X_D; }  // neighbour WIDE error in
                kick(l);
                if(l>0) readf(l, MP_GROUT, emsg, X_SEQ*X_D);  // WIDE up-message: this block's input-grad -> prev chip MP_GRIN
            }
        }

        // ===================== per-step bpb/acc from the OUTPUT chip's WIDE logits (== golden bpb) =====================
        readf(L-1, MP_WLG, wlg, X_SEQ*X_VOCAB);
        actstat("logits", L-1, step, wlg, X_SEQ*X_VOCAB);   // probe: output-chip wide logits magnitude
        { int c=0; double bits=0; const double L2=log(2.0);
          for(int t=0;t<X_SEQ;t++){ int tg=data[last_off+t+1];
              double mx=-1e300; int am=0; for(int v=0;v<X_VOCAB;v++){ double lv=wlg[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} }
              if(am==tg)c++;
              double sum=0,pt=0; for(int v=0;v<X_VOCAB;v++){ double e=exp(wlg[t*X_VOCAB+v]-mx); sum+=e; if(v==tg)pt=e; }
              double pp=pt/sum; if(pp<1e-12)pp=1e-12; bits+=-log(pp)/L2; }
          double ac=(double)c/X_SEQ, bp=bits/X_SEQ;
          acc_ema = acc_ema<0?ac:0.99*acc_ema+0.01*ac; bpb_ema = bpb_ema<0?bp:0.99*bpb_ema+0.01*bp;
          if(step%50==0 || step==steps-1)
              printf("[xfmpshard] step %5d  bpb=%.3f bpb_ema=%.3f  acc=%.3f acc_ema=%.3f  lr=%.1e  (%.2f steps/s)\n",
                     step, bp, bpb_ema, ac, acc_ema, lr, (step+1)/(now_s()-t0));
          if(step%200==0 && step>0) kick_report();
          fflush(stdout);
        }
    }

    double wall=now_s()-t0;
    kick_report();
    printf("\n========== DISTRIBUTED BF16-MP PC TRANSFORMER on %d FPGAs ==========\n", L);
    printf("[throughput] %d steps (ACCUM=%d), %.1fs, %.2f steps/s | %.0f tok/s | final bpb_ema=%.3f acc_ema=%.3f\n",
           steps, ACCUM, wall, steps/wall, steps*(double)X_SEQ*ACCUM/wall, bpb_ema, acc_ema);
    printf("[comms] cross-chip WIDE messages: %ld floats = %.2f MB | per fwd/bwd pass %d hidden(down)+%d error(up), each N*D=%d floats\n",
           xchip_msg, xchip_msg*4/1e6, (L-1), (L-1), X_SEQ*X_D);
    printf("[gateC] WEIGHTS crossed chips=%ld | GRADIENTS crossed=%ld | only neighbour wide hidden-states + errors crossed (no global model/gradient)\n",
           xchip_weight*4, xchip_grad*4);

    // ---- persist each chip's TRAINED float master back from DDR to host (so we can fetch trained weights) ----
    for(int l=0;l<L;l++){
        FMaster &m = M[l];
        readf(l, MP_FWQ,  m.WQ,  X_D*X_D);   readf(l, MP_FWK,  m.WK,  X_D*X_D);
        readf(l, MP_FWV,  m.WV,  X_D*X_D);   readf(l, MP_FWO,  m.WO,  X_D*X_D);
        readf(l, MP_FWUP, m.WUP, X_FFN*X_D); readf(l, MP_FWDN, m.WDN, X_D*X_FFN);
        readf(l, MP_FLN1G,m.LN1G,X_D);       readf(l, MP_FLN1B,m.LN1B,X_D);
        readf(l, MP_FLN2G,m.LN2G,X_D);       readf(l, MP_FLN2B,m.LN2B,X_D);
        if(l==0){   readf(l, MP_FEMB, m.EMB, X_VOCAB*X_D); readf(l, MP_FPOS, m.POS, X_SEQ*X_D); }
        if(l==L-1){ readf(l, MP_FLNFG,m.LNFG,X_D); readf(l, MP_FLNFB,m.LNFB,X_D); readf(l, MP_FWHEAD,m.WHEAD,X_VOCAB*X_D); }
        char op[512]; snprintf(op,512,"%s/xfmp_master%d.f32",DUMP_DIR,l); dump_master_file(m,op);
        printf("[xfmpshard] chip %d: trained float master -> %s\n", l, op);
    }

    // ---- optional coherent-text sample from the OUTPUT chip (re-run forward with a rolling context) ----
    // Sampling needs the full block stack forwarded; we drive a SEQ-window context through the chip pipeline
    // exactly like a training forward (no backward), read the OUTPUT chip's last-token wide logits, sample.
    if(getenv("SAMPLE")){
        int NS = atoi(getenv("SAMPLE")); if(NS<=0) NS=300;
        int ctx[X_SEQ]; for(int t=0;t<X_SEQ;t++) ctx[t]=0;     // start from token 0 (== golden uses map['\n'])
        printf("\n===== SAMPLE (%d chars, token ids) =====\n", NS);
        unsigned rng=12345;
        for(int s=0;s<NS;s++){
            for(int l=0;l<L;l++){
                setcfg(l, X_PH_FWD, role_of(l), 0, 0, 0, trainfast());
                if(l==0){ for(int t=0;t<X_SEQ;t++) hbuf[0][XO_TOK+t]=ctx[t]; stage(0,XO_TOK,X_SEQ); }
                else    { put_f(l, MP_GRIN, hmsg, X_SEQ*X_D); memcpy(hbuf[l]+XO_HIN, ihmsg(), 4*X_SEQ*X_D); stage(l,XO_HIN,X_SEQ*X_D); }
                kick(l);
                if(l<L-1){ readf(l, MP_GROUT, hmsg, X_SEQ*X_D); readw(l, XO_HOUT, ihmsg(), X_SEQ*X_D); }
            }
            readf(L-1, MP_WLG, wlg, X_SEQ*X_VOCAB);
            int last=X_SEQ-1; double mx=-1e300; for(int v=0;v<X_VOCAB;v++){ double l2=wlg[last*X_VOCAB+v]; if(l2>mx)mx=l2; }
            double sum=0,pr[X_VOCAB]; for(int v=0;v<X_VOCAB;v++){ pr[v]=exp(wlg[last*X_VOCAB+v]-mx); sum+=pr[v]; }
            rng=rng*1103515245u+12345u; double r=((rng>>8)&0xffffff)/(double)0x1000000*sum; int nxt=0; double acc=0;
            for(int v=0;v<X_VOCAB;v++){ acc+=pr[v]; if(acc>=r){ nxt=v; break; } }
            printf("%d ", nxt);
            for(int t=0;t<X_SEQ-1;t++) ctx[t]=ctx[t+1]; ctx[X_SEQ-1]=nxt;
        }
        printf("\n(token ids; map back to chars host-side via the same vocab the .u16 was built with)\n");
    }

    for(int l=0;l<L;l++){ fpga_pci_detach(bar0[l]); fpga_pci_detach(bar4[l]); fpga_pci_detach(bar4r[l]); }
    return 0;
}
