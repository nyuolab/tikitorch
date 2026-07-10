// s2_twochip_shim.cpp — BIT-FAITHFUL 2-CHIP (NL=2) one-step reference for the S2 PC-transformer.
// Drives the REAL kernel forge_pc_xformer_s2_block (compiled locally via the ap_int/hls_math shims) as TWO
// separate chips exactly the way forge_pc_xformer_s2_shard.cpp does at NL=2 with HOST_EMBED=1 (the silicon
// default). This is bit-faithful to the kernel SOURCE, so a deployed AGFI whose RTL == source must match this
// EXACTLY. Any per-tensor step-1 delta mismatch vs silicon = the precise silicon-only backward bug.
//
// NL=2 topology (matches the driver serial sweep, PIPE=0, ACCUM sub-batches, HOST_EMBED on):
//   chip0 = kernel ROLE=HID (host_embed remaps l==0 INPUT -> HID); host computes R=EMB[tok]+POS -> chip0 S2_GRIN.
//   chip1 = kernel ROLE=OUT (final-LN + LM head + CE).
//   FWD: host->chip0.GRIN ; chip0 FWD ; chip0.GROUT (=R') -> chip1.GRIN ; chip1 FWD (+targets).
//   BWD: chip1 FWD? no — chip1 BWD (CE->head->...->gR) ; chip1.GROUT (=up-err) -> chip0.GRIN ;
//        chip0 BWD ; chip0.GROUT (=embed input grad gR) -> host scatters into EMB/POS grad, host dyadic-Adam.
//   Per BWD: DO_UP on k==ACCUM-1, GZERO on k==0 (both chips + host embed), lrbits=lrcode(lr).
//
// STEPS=1 by default. Dumps per-chip step-1 masters to OUTDIR/shim_master<l>.f32 in FMaster field order:
//   WQ,WK,WV,WO,WUP,WDN,WHEAD,EMB,POS,LN1G,LN1B,LN2G,LN2B,LNFG,LNFB  (== driver dump_master_file order).
//   chip0 (HID+host_embed): WQ..WDN + LN1/LN2 read back from buffer; EMB/POS = host-trained; WHEAD/LNF = init.
//   chip1 (OUT):            WQ..WDN + LN1/LN2 + WHEAD + LNF read back; EMB/POS = init (untouched by OUT).
//
// build: g++ -O2 -std=c++14 -D__AP_INT_SPECIAL_H__ -I_apshim -I_mpshim -I<kerneldir>
//        forge_pc_xformer_s2_block.cpp s2_twochip_shim.cpp -o s2twochip
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <ap_int.h>
#include "forge_pc_xformer_s2_config.hpp"

extern "C" void forge_pc_xformer_s2_block(ap_uint<512> *buffer);

static const int D=S2_D,H=S2_H,DH=S2_DH,FFN=S2_FFN,SEQ=S2_SEQ,VOCAB=S2_VOCAB;

// ---------------- bit helpers (float lives in one int32 lane, bit-reinterpret) ----------------
static int32_t f2i_(float f){ union{float f;int32_t i;}c; c.f=f; return c.i; }
static float   i2f_(int32_t i){ union{float f;int32_t i;}c; c.i=i; return c.f; }
static void put_f(int32_t*b,int off,const float*s,long n){ for(long i=0;i<n;i++) b[off+i]=f2i_(s[i]); }
static void put_i(int32_t*b,int off,const int32_t*s,long n){ for(long i=0;i<n;i++) b[off+i]=s[i]; }
static void get_f(const int32_t*b,int off,float*d,long n){ for(long i=0;i<n;i++) d[i]=i2f_(b[off+i]); }

// CFG lanes: [phase,role,do_up,lrbits,gzero,train,slot,tile_o,tile_i,seq_tile]
static void setcfg(int32_t*b,int ph,int role,int do_up,int lrbits,int gzero,int slot){
    for(int i=0;i<S2_WL;i++) b[S2O_CFG+i]=0;
    b[S2O_CFG+0]=ph; b[S2O_CFG+1]=role; b[S2O_CFG+2]=do_up; b[S2O_CFG+3]=lrbits; b[S2O_CFG+4]=gzero;
    b[S2O_CFG+5]=1 /*train*/; b[S2O_CFG+6]=slot;
}
// driver lrcode: llround(lr*2^20), clamped [0,0x7fffffff]
static int lrcode(double lr){ long c=(long)llround(lr*(double)(1<<20)); if(c<0)c=0; if(c>0x7fffffffL)c=0x7fffffffL; return (int)c; }
// driver dyad_adam (host EMB/POS): pow2 step from |m|/v ratio. RETURNS the delta to SUBTRACT from the weight.
static inline float dyad_adam(float g,float&m,float&v,float lr){ m=0.875f*m+0.125f*g; v=0.875f*v+0.125f*fabsf(g);
    float r=fabsf(m)/(v+1e-9f); float st=(r>0)?ldexpf(1.0f,(int)floorf(log2f(r))):0.0f; return lr*((m>0)-(m<0))*st; }

// ---------------- FMaster field order (== driver struct / dump_master_file, packed fp32, no padding) ----------------
struct FMaster {
    float WQ[D*D], WK[D*D], WV[D*D], WO[D*D];
    float WUP[FFN*D], WDN[D*FFN], WHEAD[VOCAB*D];
    float EMB[VOCAB*D], POS[SEQ*D];
    float LN1G[D], LN1B[D], LN2G[D], LN2B[D], LNFG[D], LNFB[D];
};
static bool load_master_file(FMaster&m,const char*path){
    FILE*f=fopen(path,"rb"); if(!f) return false;
    size_t n=fread(&m,sizeof(float),sizeof(FMaster)/sizeof(float),f); fclose(f);
    return n==sizeof(FMaster)/sizeof(float);
}
static void dump_master_file(const FMaster&m,const char*path){
    FILE*f=fopen(path,"wb"); if(!f){fprintf(stderr,"open %s\n",path);return;} fwrite(&m,sizeof(float),sizeof(FMaster)/sizeof(float),f); fclose(f);
}
// stage a chip's full FMaster into its S2_F* buffer region (matches driver stage_master; role-gated fields).
static void stage_master(int32_t*b,int role_is_out,const FMaster&m){
    put_f(b,S2_FWQ, m.WQ, (long)D*D);   put_f(b,S2_FWK, m.WK, (long)D*D);
    put_f(b,S2_FWV, m.WV, (long)D*D);   put_f(b,S2_FWO, m.WO, (long)D*D);
    put_f(b,S2_FWUP,m.WUP,(long)FFN*D); put_f(b,S2_FWDN,m.WDN,(long)D*FFN);
    put_f(b,S2_FLN1G,m.LN1G,D);         put_f(b,S2_FLN1B,m.LN1B,D);
    put_f(b,S2_FLN2G,m.LN2G,D);         put_f(b,S2_FLN2B,m.LN2B,D);
    // chip0 (HID): the driver stages EMB/POS only when l==0. Under host_embed chip0's role_of(0)==INPUT so the
    // driver's stage_master(l==0) DOES stage EMB/POS into S2_FEMB/S2_FPOS (init copy). The kernel HID path never
    // reads them (host computes R), so they stay the init copy in DDR. Stage them for byte-parity with the driver.
    if(!role_is_out){ put_f(b,S2_FEMB,m.EMB,(long)VOCAB*D); put_f(b,S2_FPOS,m.POS,(long)SEQ*D); }
    if(role_is_out){ put_f(b,S2_FLNFG,m.LNFG,D); put_f(b,S2_FLNFB,m.LNFB,D); put_f(b,S2_FWHEAD,m.WHEAD,(long)VOCAB*D); }
}

// ---------------- FWD_DUMP: per-chip per-stage forward-activation dumps (bit-faithful SOURCE) ----------------
// Mirrors the driver's FWD_DUMP (forge_pc_xformer_s2_shard.cpp). After a chip's FORWARD kick, read its S2_W* wide
// cache and write the FULL fp32 tensor to <OUTDIR>/fwd_<chip>_<name>.f32 in the kernel's forward order:
//   R, X1(LN1), Q,K,V(proj+qact), ATTN(attn+qact), M(=R+AO), X2(LN2), HD(FFN+relu+qact), RP(=M+FO);
//   + XF(LNF) and LG(logits) on the OUTPUT chip only. Chip index matches the driver (chip0, chip L-1=1 at NL=2),
// so fwd_<chip>_<name>.f32 diffs bit-for-bit against silicon's same-named dump at the matched init + fixed window.
static void dump_f32_(const char* dir, int chip, const char* name, const float* x, long n){
    char op[600]; snprintf(op,600,"%s/fwd_%d_%s.f32", dir, chip, name);
    FILE* f=fopen(op,"wb"); if(!f){ fprintf(stderr,"[FWD_DUMP] open %s failed\n",op); return; }
    fwrite(x,sizeof(float),(size_t)n,f); fclose(f);
    fprintf(stderr,"[FWD_DUMP] chip %d %-4s -> %s (%ld floats)\n", chip, name, op, n);
}
static void dump_fwd_chip(const int32_t* b, const char* dir, int chip, int is_out){
    // sized for BOTH SEQ*FFN (HD) and SEQ*VOCAB (LG) — at S2_VOCAB=8192 the old SEQ*FFN sizing
    // overflowed on the LG dump (heap smash / bus error)
    static std::vector<float> db((size_t)SEQ*FFN > (size_t)SEQ*VOCAB ? (size_t)SEQ*FFN : (size_t)SEQ*VOCAB);
    get_f(b,S2_WR,   db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"R",   db.data(),(long)SEQ*D);
    get_f(b,S2_WX1,  db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"X1",  db.data(),(long)SEQ*D);
    get_f(b,S2_WQ,   db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"Q",   db.data(),(long)SEQ*D);
    get_f(b,S2_WK,   db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"K",   db.data(),(long)SEQ*D);
    get_f(b,S2_WV,   db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"V",   db.data(),(long)SEQ*D);
    get_f(b,S2_WATTN,db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"ATTN",db.data(),(long)SEQ*D);
    get_f(b,S2_WM,   db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"M",   db.data(),(long)SEQ*D);
    get_f(b,S2_WX2,  db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"X2",  db.data(),(long)SEQ*D);
    get_f(b,S2_WHD,  db.data(),(long)SEQ*FFN);dump_f32_(dir,chip,"HD",  db.data(),(long)SEQ*FFN);
    get_f(b,S2_WRP,  db.data(),(long)SEQ*D);  dump_f32_(dir,chip,"RP",  db.data(),(long)SEQ*D);
    if(is_out){
        get_f(b,S2_WXF, db.data(),(long)SEQ*D);    dump_f32_(dir,chip,"XF",db.data(),(long)SEQ*D);
        get_f(b,S2_WLG, db.data(),(long)SEQ*VOCAB);dump_f32_(dir,chip,"LG",db.data(),(long)SEQ*VOCAB);
    }
}

// CE bits/token from the OUTPUT chip's wide logits S2_WLG and the target sequence.
static double ce_bits(const int32_t*b,const int32_t*tgt){
    std::vector<float> LG(SEQ*VOCAB); get_f(b,S2_WLG,LG.data(),(long)SEQ*VOCAB);
    double tot=0;
    for(int t=0;t<SEQ;t++){ double mx=-1e300; for(int v=0;v<VOCAB;v++){double l=LG[t*VOCAB+v]; if(l>mx)mx=l;}
        double sum=0; for(int v=0;v<VOCAB;v++) sum+=exp(LG[t*VOCAB+v]-mx);
        double logp=(LG[t*VOCAB+tgt[t]]-mx)-log(sum); tot+=-logp/log(2.0); }
    return tot/SEQ;
}

int main(int argc,char**argv){
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char* OUTDIR   = getenv("OUTDIR")   ? getenv("OUTDIR")   : ".";
    const char* INIT_DIR = getenv("INIT_DIR");
    const char* ENWIK    = getenv("ENWIK");                          // path to a train.u16 (uint16 tokens)
    int    STEPS  = getenv("STEPS")  ? atoi(getenv("STEPS"))  : 1;
    int    ACCUM  = getenv("ACCUM")  ? atoi(getenv("ACCUM"))  : 8;   if(ACCUM<1) ACCUM=1;
    double LR     = getenv("XF_LR")  ? atof(getenv("XF_LR"))  : 2.6e-4;
    int    WARMUP = getenv("XF_WARMUP")? atoi(getenv("XF_WARMUP")) : 0;
    long   FIXEDOFF = getenv("FIXEDOFF") ? atol(getenv("FIXEDOFF")) : 0;
    int    LR_DECAY = getenv("LR_DECAY") ? atoi(getenv("LR_DECAY")) : 0;
    int    FWD_DUMP = getenv("FWD_DUMP") ? atoi(getenv("FWD_DUMP")) : 0;   // dump per-chip fwd activations (step0/k0)
    if(!INIT_DIR){ fprintf(stderr,"FATAL: set INIT_DIR (loads INIT_DIR/master{0,1}.f32)\n"); return 2; }
    if(!ENWIK){ fprintf(stderr,"FATAL: set ENWIK (uint16 token stream; FIXEDOFF selects the window)\n"); return 2; }

    printf("=== s2_twochip_shim NL=%d HOST_EMBED  D=%d H=%d FFN=%d SEQ=%d VOCAB=%d K=%d | STEPS=%d ACCUM=%d LR=%.2e WARMUP=%d FIXEDOFF=%ld ROLL=%d ===\n",
           getenv("NL")?atoi(getenv("NL")):2, D,H,FFN,SEQ,VOCAB,S2_K,STEPS,ACCUM,LR,WARMUP,FIXEDOFF,
           getenv("ROLL")&&atoi(getenv("ROLL"))?1:0);

    // ---- load token stream ----
    FILE*tf=fopen(ENWIK,"rb"); if(!tf){fprintf(stderr,"open %s\n",ENWIK);return 2;}
    fseek(tf,0,SEEK_END); long nb=ftell(tf); fseek(tf,0,SEEK_SET); long ntok=nb/2;
    std::vector<uint16_t> data(ntok); if(fread(data.data(),2,ntok,tf)!=(size_t)ntok){fprintf(stderr,"read tok\n");return 2;} fclose(tf);
    if(ntok < FIXEDOFF + SEQ + 2){ fprintf(stderr,"FATAL: token stream too short (%ld) for FIXEDOFF=%ld + SEQ+2\n",ntok,FIXEDOFF); return 2; }
    printf("[shim] %ld tokens from %s\n", ntok, ENWIK);

    // ---- N-chip generalization (NL env, default 2): NL layers on NL chips.
    //      chip0 = HID with host-embed; chips 1..NL-2 = HID (block only); chip NL-1 = OUT (final-LN + head + CE).
    //      This is the L=8 deconfounder for the silicon-vs-recipe divergence question — the L=2 twin changed
    //      exp-exactness AND depth simultaneously; running L=8 with exact-libm exp isolates depth. ----
    int NL = getenv("NL") ? atoi(getenv("NL")) : 2; if(NL<2) NL=2;
    std::vector<FMaster*> M(NL);
    for(int l=0;l<NL;l++){ M[l]=(FMaster*)malloc(sizeof(FMaster)); if(!M[l]){fprintf(stderr,"malloc master %d\n",l);return 4;}
        char ip[512]; snprintf(ip,512,"%s/master%d.f32",INIT_DIR,l);
        if(!load_master_file(*M[l],ip)){fprintf(stderr,"load %s (need gen_init CHIPS=%d)\n",ip,NL);return 3;}
        printf("[shim] chip%d init <- %s\n",l,ip); }

    // ---- NL independent kernel buffers (calloc zeroes adam m/v + GACC, exactly like the driver) ----
    std::vector<int32_t*> b(NL);
    for(int l=0;l<NL;l++){ b[l]=(int32_t*)calloc((size_t)S2_BUFLEN,4); if(!b[l]){fprintf(stderr,"calloc buf %d\n",l);return 5;}
        stage_master(b[l], (l==NL-1)?1:0, *M[l]); }   // last=OUT(stages WHEAD/LNF); else HID(stages EMB/POS, unused by mid chips)

    // ---- host-embed state: hEMB/hPOS alias chip0's master EMB/POS (driver: hEMB=M[0]->EMB) ----
    float* hEMB=M[0]->EMB; float* hPOS=M[0]->POS;
    std::vector<float> hmE(VOCAB*D,0),hvE(VOCAB*D,0),hmP(SEQ*D,0),hvP(SEQ*D,0),hgE(VOCAB*D,0),hgP(SEQ*D,0);

    std::vector<float> hmsg(SEQ*D), emsg(SEQ*D);           // wide down / up message (SEQ*D floats)
    std::vector<int32_t> tgt(SEQ);
    auto ROLE=[&](int l){ return (l==NL-1)?S2_RL_OUT:S2_RL_HID; };

    for(int step=0; step<STEPS; step++){
        // driver LR schedule: linear warmup (guard WARMUP>0?WARMUP:1) then optional cosine decay.
        double lr = LR * fmin(1.0,(step+1)/(double)(WARMUP>0?WARMUP:1));
        if(LR_DECAY && step>=WARMUP){ double t=(double)(step-WARMUP)/(double)(STEPS-WARMUP>0?STEPS-WARMUP:1);
            double LR_MINFR=getenv("LR_MINFRAC")?atof(getenv("LR_MINFRAC")):0.1; lr=LR*(LR_MINFR+(1.0-LR_MINFR)*0.5*(1.0+cos(M_PI*t))); }
        int lrb=lrcode(lr);

        for(int k=0;k<ACCUM;k++){
            // ROLL=1: advance the window every sub-batch (step*ACCUM+k), wrapping — a REAL rolling-data trainer
            // (the true recipe-stability twin of the diverging silicon run). Default: FIXEDOFF fixed window.
            long off = (FIXEDOFF>=0)? FIXEDOFF : 0;        // FIXEDOFF fixes the window (same example every sub-batch)
            if(getenv("ROLL") && atoi(getenv("ROLL"))){ long span=ntok-SEQ-2; long base=(FIXEDOFF>=0)?FIXEDOFF:0;
                off = base + ((long)(step*ACCUM+k)*SEQ) % (span>0?span:1); }
            int do_update=(k==ACCUM-1)?1:0;
            int gz=(k==0)?1:0;

            // ============ FORWARD SWEEP chip0 -> chip NL-1 ============
            // chip0 (HID): host computes R = EMB[tok] + POS -> S2_GRIN; chips 1..NL-1 read the previous chip's R'.
            for(int t=0;t<SEQ;t++){ int v=(int)data[off+t]; if(v<0)v=0; if(v>=VOCAB)v=VOCAB-1;
                for(int d=0;d<D;d++) hmsg[t*D+d]=hEMB[(long)v*D+d]+hPOS[t*D+d]; }
            for(int l=0;l<NL;l++){
                put_f(b[l],S2_GRIN,hmsg.data(),(long)SEQ*D);
                if(l==NL-1){ for(int t=0;t<SEQ;t++) tgt[t]=data[off+t+1]; put_i(b[l],S2_TGT,tgt.data(),SEQ); }  // OUT targets
                setcfg(b[l],S2_PH_FWD,ROLE(l),0,0,0,0); forge_pc_xformer_s2_block((ap_uint<512>*)b[l]);
                if(FWD_DUMP && step==0 && k==0 && (l==0||l==NL-1)){ fprintf(stderr,"[FWD_DUMP] chip%d forward wide-cache -> %s\n",l,OUTDIR); dump_fwd_chip(b[l],OUTDIR,l,l==NL-1); }
                if(l==NL-1){ if(k==0){ double ce=ce_bits(b[l],tgt.data()); printf("[shim] step %d k0 CE(pre-update) = %.4f bits/tok (baseline log2(V)=%.3f)\n", step, ce, log2((double)VOCAB)); } }
                else get_f(b[l],S2_GROUT,hmsg.data(),(long)SEQ*D);   // this chip's wide R' -> next chip's input
            }

            // ============ BACKWARD SWEEP chip NL-1 -> chip0 ============
            // OUT chip seeds CE grad internally (no GRIN); each upstream HID chip reads the downstream up-error.
            for(int l=NL-1;l>=0;l--){
                if(l<NL-1) put_f(b[l],S2_GRIN,emsg.data(),(long)SEQ*D);   // up-error from the downstream chip
                setcfg(b[l],S2_PH_BWD,ROLE(l),do_update,do_update?lrb:0,gz,0); forge_pc_xformer_s2_block((ap_uint<512>*)b[l]);
                get_f(b[l],S2_GROUT,emsg.data(),(long)SEQ*D);            // up-error -> upstream chip (chip0: gR = embed input grad)
            }

            // host trains EMB/POS from gR (== emsg after chip0's backward; driver serial-sweep host_embed path).
            if(gz){ memset(hgE.data(),0,(size_t)VOCAB*D*4); memset(hgP.data(),0,(size_t)SEQ*D*4); }  // GZERO
            for(int t=0;t<SEQ;t++){ int v=(int)data[off+t]; if(v<0)v=0; if(v>=VOCAB)v=VOCAB-1;
                for(int d=0;d<D;d++){ float g=emsg[t*D+d]; hgE[(long)v*D+d]+=g; hgP[t*D+d]+=g; } }
            if(do_update && !getenv("EMB_FREEZE")){
                for(long i=0;i<(long)VOCAB*D;i++) hEMB[i]-=dyad_adam(hgE[i],hmE[i],hvE[i],(float)lr);
                for(long i=0;i<(long)SEQ*D;i++)   hPOS[i]-=dyad_adam(hgP[i],hmP[i],hvP[i],(float)lr);
            }
        }
        printf("[shim] step %d done (lr=%.3e lrbits=%d)\n", step, lr, lrb);
    }

    // ---- dump per-chip trained masters (only for the NL==2 one-step-reference use case; the L=8 stability
    //      sweep's deliverable is the CE trajectory above, so it skips the readback dump). ----
    if(NL==2){
        FMaster &M0=*M[0], &M1=*M[1]; int32_t *b0=b[0], *b1=b[1];
        get_f(b0,S2_FWQ, M0.WQ, (long)D*D);   get_f(b0,S2_FWK, M0.WK, (long)D*D);
        get_f(b0,S2_FWV, M0.WV, (long)D*D);   get_f(b0,S2_FWO, M0.WO, (long)D*D);
        get_f(b0,S2_FWUP,M0.WUP,(long)FFN*D); get_f(b0,S2_FWDN,M0.WDN,(long)D*FFN);
        get_f(b0,S2_FLN1G,M0.LN1G,D);         get_f(b0,S2_FLN1B,M0.LN1B,D);
        get_f(b0,S2_FLN2G,M0.LN2G,D);         get_f(b0,S2_FLN2B,M0.LN2B,D);
        get_f(b1,S2_FWQ, M1.WQ, (long)D*D);   get_f(b1,S2_FWK, M1.WK, (long)D*D);
        get_f(b1,S2_FWV, M1.WV, (long)D*D);   get_f(b1,S2_FWO, M1.WO, (long)D*D);
        get_f(b1,S2_FWUP,M1.WUP,(long)FFN*D); get_f(b1,S2_FWDN,M1.WDN,(long)D*FFN);
        get_f(b1,S2_FLN1G,M1.LN1G,D);         get_f(b1,S2_FLN1B,M1.LN1B,D);
        get_f(b1,S2_FLN2G,M1.LN2G,D);         get_f(b1,S2_FLN2B,M1.LN2B,D);
        get_f(b1,S2_FLNFG,M1.LNFG,D);         get_f(b1,S2_FLNFB,M1.LNFB,D);
        get_f(b1,S2_FWHEAD,M1.WHEAD,(long)VOCAB*D);
        char op[512];
        snprintf(op,512,"%s/shim_master0.f32",OUTDIR); dump_master_file(M0,op); printf("[shim] chip0 trained master -> %s\n",op);
        snprintf(op,512,"%s/shim_master1.f32",OUTDIR); dump_master_file(M1,op); printf("[shim] chip1 trained master -> %s\n",op);
    }

    for(int l=0;l<NL;l++){ free(b[l]); free(M[l]); }
    printf("=== s2_twochip_shim DONE (NL=%d) ===\n", NL);
    return 0;
}
